"""Shared scan, dedupe, evaluate, and run-evidence mechanics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from kb_artifacts.contracts import ArtifactRecipe, EvidenceRecord, SelectionDecision
from kb_artifacts.renderers.markdown import render
from kb_artifacts.sources.jsonl_bus import SourceInputError, expand_globs, scan_jsonl


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dedupe_key(record: EvidenceRecord) -> str:
    return record.provenance.text_sha256 or record.provenance.source_ref or record.record_id


def _representative_key(record: EvidenceRecord) -> tuple[int, int, int, str]:
    """Prefer the richest occurrence while retaining deterministic tie-breaking."""
    return (-len(record.annotations), -int(record.summary is not None), -int(record.title is not None), record.record_id)


def build(recipe: ArtifactRecipe, *, chunk_globs: Iterable[str], summary_globs: Iterable[str], output: Path, allow_empty: bool = False) -> dict:
    chunk_globs = list(chunk_globs)
    summary_globs = list(summary_globs)
    if output.exists() and any(output.iterdir()):
        raise SourceInputError(f"Output directory is not empty: {output}")
    chunk_paths = expand_globs(chunk_globs)
    summary_paths = expand_globs(summary_globs)
    if not (chunk_paths or summary_paths):
        raise SourceInputError("No input files matched the requested globs")
    chunk_records, chunk_errors = scan_jsonl(chunk_paths, source_kind="chunk")
    summary_records, summary_errors = scan_jsonl(summary_paths, source_kind="summary")
    records = list(chunk_records) + list(summary_records)
    errors = list(chunk_errors) + list(summary_errors)
    if not records:
        raise SourceInputError("No usable records were parsed from matched input files")
    records.sort(key=lambda record: (record.timestamp or datetime.min.replace(tzinfo=timezone.utc), record.record_id, record.provenance.partition, record.provenance.line_number))
    decisions: list[SelectionDecision] = []
    duplicate_groups: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        key = _dedupe_key(record)
        duplicate_groups.setdefault(key, []).append(record)
    representatives: dict[str, EvidenceRecord] = {}
    for key, occurrences in duplicate_groups.items():
        winner = min(occurrences, key=_representative_key)
        representatives[key] = winner
        for record in occurrences:
            if record.record_id != winner.record_id:
                decisions.append(SelectionDecision(record.record_id, "deduplicated", 0, ("duplicate_source_record",), {"dedupe_key": key}, winner.record_id))
    evaluated = [(record, recipe.evaluate(record)) for record in representatives.values()]
    decisions.extend(decision for _, decision in evaluated)
    selected = [(record, decision) for record, decision in evaluated if decision.disposition == "selected"]
    selected.sort(key=lambda item: (-item[1].score, item[0].timestamp or datetime.min.replace(tzinfo=timezone.utc), item[0].record_id))
    if not selected and not allow_empty:
        raise SourceInputError("No records were selected; rerun with --allow-empty only when an empty artifact is intentional")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        by_id = {record.record_id: record for record in records}
        for decision in sorted(decisions, key=lambda item: (item.record_id, item.disposition)):
            payload = _jsonable(asdict(decision))
            record = by_id.get(decision.record_id)
            if record:
                payload["provenance"] = _jsonable(asdict(record.provenance))
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    if errors:
        with (output / "errors.jsonl").open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "artifact.md").write_text(render(recipe, selected), encoding="utf-8")
    all_paths = chunk_paths + summary_paths
    manifest = {
        "recipe": recipe.id,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_configuration": {"chunk_globs": list(chunk_globs), "summary_globs": list(summary_globs)},
        "matched_partitions": [{"path": str(path), "sha256": _fingerprint(path)} for path in all_paths],
        "counts": {
            "scanned": len(records) + len(errors), "usable": len(records), "invalid": len(errors),
            "selected": len(selected), "rejected": sum(item.disposition == "rejected" for item in decisions),
            "deduplicated": sum(item.disposition == "deduplicated" for item in decisions),
        },
        "warnings": ["missing_summary" for record, _ in selected if record.summary is None],
        "outputs": ["manifest.json", "decisions.jsonl", "artifact.md"] + (["errors.jsonl"] if errors else []),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
