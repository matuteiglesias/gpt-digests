"""Shared scan, dedupe, evaluate, and run-evidence mechanics."""

from __future__ import annotations

import hashlib
import json
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from kb_artifacts.contracts import ArtifactRecipe, EvidenceRecord, SelectionDecision
from kb_artifacts.normalization import normalize_value
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


def _review_text(value: str | None, limit: int, *, excerpt_only: bool = False) -> str:
    """Keep calibration exports useful without exporting whole source bodies."""
    text = (value or "").replace("\n", " ").strip()
    if excerpt_only and len(text) <= limit:
        return ""
    return text[:limit]


def _write_review_packet(
    path: Path,
    recipe: ArtifactRecipe,
    evaluated: list[tuple[EvidenceRecord, SelectionDecision]],
) -> None:
    """Write a deterministic, bounded calibration packet; it does not affect decisions."""
    ranked = sorted(evaluated, key=lambda item: (-item[1].score, item[0].timestamp or datetime.min.replace(tzinfo=timezone.utc), item[0].record_id))
    selected = [(record, decision, "selected") for record, decision in ranked if decision.disposition == "selected"][:40]
    exclusions = [
        (record, decision, "policy_exclusion") for record, decision in ranked
        if decision.disposition == "rejected" and any(reason in {"recipe_only_material", "reusability_score=1"} for reason in decision.reasons)
    ][:20]
    exclusion_ids = {record.record_id for record, _, _ in exclusions}
    near = [
        (record, decision, "near_threshold") for record, decision in ranked
        if decision.disposition == "rejected"
        and decision.score >= recipe.near_threshold_score
        and record.record_id not in exclusion_ids
    ][:20]
    fields = ("note_type", "format_type", "msg_type", "stage", "snippet_type", "actionable", "reusability_score", "category", "domain", "medium")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("packet_section", "record_id", "source_kind", "source_ref", "title", "timestamp", "annotations", "score", "score_components", "summary", "text_excerpt", "current_disposition", "review_label", "review_comment", "expected_group"))
        writer.writeheader()
        for record, decision, section in selected + near + exclusions:
            annotations = {
                key: ([normalize_value(item) for item in value] if isinstance(value, (list, tuple)) else normalize_value(value))
                if key in {"note_type", "format_type", "msg_type", "stage", "snippet_type", "category", "domain", "medium"}
                else value
                for key, value in record.annotations.items() if key in fields
            }
            writer.writerow({
                "packet_section": section, "record_id": record.record_id, "source_kind": record.source_kind,
                "source_ref": record.provenance.source_ref or f"{record.provenance.partition}:{record.provenance.line_number}",
                "title": _review_text(record.title or record.summary or "Untitled evidence", 160),
                "timestamp": record.timestamp.isoformat() if record.timestamp else "",
                "annotations": json.dumps(_jsonable(annotations), ensure_ascii=False, sort_keys=True),
                "score": f"{decision.score:g}", "score_components": "; ".join(decision.reasons),
                "summary": _review_text(record.summary, 240), "text_excerpt": _review_text(record.text, 320, excerpt_only=True),
                "current_disposition": decision.disposition, "review_label": "", "review_comment": "", "expected_group": "",
            })


def build(
    recipe: ArtifactRecipe,
    *,
    chunk_globs: Iterable[str],
    summary_globs: Iterable[str],
    output: Path,
    allow_empty: bool = False,
    audit_all_decisions: bool = False,
    review_packet: Path | None = None,
) -> dict:
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
    duplicate_decisions: list[SelectionDecision] = []
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
                duplicate_decisions.append(SelectionDecision(record.record_id, "deduplicated", 0, ("duplicate_source_record",), {"dedupe_key": key}, winner.record_id))
    evaluated = [(record, recipe.evaluate(record)) for record in representatives.values()]
    selected = [(record, decision) for record, decision in evaluated if decision.disposition == "selected"]
    rejected = [(record, decision) for record, decision in evaluated if decision.disposition == "rejected"]
    candidate_rejected = [
        (record, decision) for record, decision in rejected
        if decision.score >= recipe.candidate_threshold
    ]
    near_threshold = [
        (record, decision) for record, decision in candidate_rejected
        if decision.score >= recipe.near_threshold_score
    ]
    near_threshold_ids = {decision.record_id for _, decision in near_threshold}
    ordinary_nonmatches = [
        (record, decision) for record, decision in rejected
        if decision.score < recipe.candidate_threshold
    ]
    nonmatch_reasons: dict[str, int] = {}
    for _, decision in rejected:
        for reason in decision.reasons:
            nonmatch_reasons[reason] = nonmatch_reasons.get(reason, 0) + 1
    # The normal ledger is deliberately a review ledger, not a copy of every
    # negative evaluation.  --audit-all-decisions is the explicit escape hatch.
    decisions = duplicate_decisions + [decision for _, decision in selected] + [decision for _, decision in candidate_rejected]
    if audit_all_decisions:
        decisions = duplicate_decisions + [decision for _, decision in evaluated]
    selected.sort(key=lambda item: (-item[1].score, item[0].timestamp or datetime.min.replace(tzinfo=timezone.utc), item[0].record_id))
    if not selected and not allow_empty:
        raise SourceInputError("No records were selected; rerun with --allow-empty only when an empty artifact is intentional")
    output.mkdir(parents=True, exist_ok=True)
    if review_packet:
        _write_review_packet(review_packet, recipe, evaluated)
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        by_id = {record.record_id: record for record in records}
        for decision in sorted(decisions, key=lambda item: (item.record_id, item.disposition)):
            payload = _jsonable(asdict(decision))
            if decision.record_id in near_threshold_ids:
                payload["review_flags"] = ["near_threshold"]
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
            "scanned": len(records) + len(errors),
            "invalid_or_unusable": len(errors),
            "deduplicated": len(duplicate_decisions),
            "evaluated_unique": len(evaluated),
            "selected": len(selected),
            "candidate_rejected": len(candidate_rejected),
            "near_threshold": len(near_threshold),
            "ordinary_nonmatch_count": len(ordinary_nonmatches),
            "aggregate_nonmatch_reason_counts": dict(sorted(nonmatch_reasons.items())),
        },
        "reconciliation": {
            "scanned": "invalid_or_unusable + deduplicated + evaluated_unique",
            "evaluated_unique": "selected + candidate_rejected + ordinary_nonmatch_count",
            "scanned_value": len(errors) + len(duplicate_decisions) + len(evaluated),
            "evaluated_unique_value": len(selected) + len(candidate_rejected) + len(ordinary_nonmatches),
        },
        "decision_ledger": {
            "audit_all_decisions": audit_all_decisions,
            "candidate_threshold": recipe.candidate_threshold,
            "near_threshold_score": recipe.near_threshold_score,
            "ordinary_nonmatches_omitted": 0 if audit_all_decisions else len(ordinary_nonmatches),
        },
        "warnings": ["missing_summary" for record, _ in selected if record.summary is None],
        "outputs": ["manifest.json", "decisions.jsonl", "artifact.md"] + ([str(review_packet)] if review_packet else []) + (["errors.jsonl"] if errors else []),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
