"""Generic, read-only evidence selection and multi-format export."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from kb_artifacts.classification import classify
from kb_artifacts.contracts import EvidenceRecord, SelectionDecision
from kb_artifacts.normalization import normalize_value, tag_lexeme
from kb_artifacts.sources.jsonl_bus import SourceInputError, expand_globs, scan_jsonl


@dataclass(frozen=True)
class SelectionRequest:
    chunk_globs: tuple[str, ...] = ()
    summary_globs: tuple[str, ...] = ()
    start: date | None = None
    end: date | None = None
    tags: tuple[str, ...] = ()
    fields: tuple[tuple[str, str], ...] = ()
    text_pattern: str | None = None
    families: tuple[str, ...] = ()
    maturities: tuple[str, ...] = ()
    limit: int | None = None
    deduplicate: bool = True
    group_by: str = "domain"


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(record: EvidenceRecord) -> str:
    return record.provenance.text_sha256 or record.record_id


def _sort_key(record: EvidenceRecord) -> tuple[datetime, str, str, int]:
    return (record.timestamp or datetime.min.replace(tzinfo=timezone.utc), record.record_id, record.provenance.partition, record.provenance.line_number)


def _field_matches(record: EvidenceRecord, field: str, wanted: str) -> bool:
    value = record.annotations.get(field)
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return any(normalize_value(item) == normalize_value(wanted) for item in values)


def _matches(record: EvidenceRecord, request: SelectionRequest, pattern: re.Pattern[str] | None) -> tuple[bool, list[str], object | None]:
    reasons: list[str] = []
    if request.start and (not record.timestamp or record.timestamp.date() < request.start): return False, reasons, None
    if request.end and (not record.timestamp or record.timestamp.date() > request.end): return False, reasons, None
    if request.start: reasons.append(f"from:{request.start.isoformat()}")
    if request.end: reasons.append(f"to:{request.end.isoformat()}")
    if request.tags:
        tags = {tag_lexeme(tag) for tag in record.tags}
        requested = {normalize_value(tag) for tag in request.tags}
        if not tags.intersection(requested): return False, reasons, None
        reasons.append("tag:" + sorted(tags.intersection(requested))[0])
    for field, wanted in request.fields:
        if not _field_matches(record, field, wanted): return False, reasons, None
        reasons.append(f"field:{field}={normalize_value(wanted)}")
    result = classify(record) if request.families or request.maturities else None
    if request.families and result.family not in set(request.families): return False, reasons, result
    if request.maturities and result.maturity not in set(request.maturities): return False, reasons, result
    if result and request.families: reasons.append(f"family:{result.family}")
    if result and request.maturities: reasons.append(f"maturity:{result.maturity}")
    if pattern and not pattern.search("\n".join(part for part in (record.title, record.summary, record.text) if part)):
        return False, reasons, result
    if pattern: reasons.append("text_match")
    return True, reasons, result


def _payload(record: EvidenceRecord, decision: SelectionDecision) -> dict[str, object]:
    return {"record_id": record.record_id, "source_kind": record.source_kind, "title": record.title,
            "summary": record.summary, "annotations": dict(record.annotations), "tags": list(record.tags),
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
            "provenance": asdict(record.provenance), "selection_reasons": list(decision.reasons),
            "artifact_family": decision.artifact_family, "artifact_maturity": decision.artifact_maturity}


def select(request: SelectionRequest, *, output: Path, allow_empty: bool = False) -> dict:
    """Select once from both buses and render JSONL, CSV, Markdown, and evidence."""
    if output.exists() and any(output.iterdir()):
        raise SourceInputError(f"Output directory is not empty: {output}")
    paths = [("chunk", path) for path in expand_globs(request.chunk_globs)] + [("summary", path) for path in expand_globs(request.summary_globs)]
    if not paths: raise SourceInputError("No input files matched the requested globs")
    try: pattern = re.compile(request.text_pattern, re.IGNORECASE) if request.text_pattern else None
    except re.error as error: raise SourceInputError(f"Invalid text pattern: {error}") from error
    records: list[EvidenceRecord] = []; errors: list[dict[str, object]] = []
    for kind, path in paths:
        found, diagnostics = scan_jsonl([path], source_kind=kind)
        records.extend(found); errors.extend(diagnostics)
    records.sort(key=_sort_key)
    unique: list[EvidenceRecord] = []; duplicate_count = 0; seen: set[str] = set()
    for record in records:
        if request.deduplicate and _key(record) in seen:
            duplicate_count += 1; continue
        seen.add(_key(record)); unique.append(record)
    selected: list[tuple[EvidenceRecord, SelectionDecision]] = []
    for record in unique:
        matched, reasons, result = _matches(record, request, pattern)
        if matched:
            selected.append((record, SelectionDecision(record.record_id, "selected", 1, tuple(reasons), {}, artifact_family=result.family if result else None, artifact_maturity=result.maturity if result else None, classification_reasons=result.reasons if result else ())))
    if request.limit is not None: selected = selected[:request.limit]
    if not selected and not allow_empty: raise SourceInputError("No records matched; rerun with --allow-empty only when an empty export is intentional")
    output.mkdir(parents=True, exist_ok=True)
    payloads = [_payload(record, decision) for record, decision in selected]
    with (output / "selected.jsonl").open("w", encoding="utf-8") as handle:
        for item in payloads: handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "selected.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_id", "source_kind", "timestamp", "title", "summary", "tags", "artifact_family", "artifact_maturity", "selection_reasons", "text_excerpt", "source_ref"))
        writer.writeheader()
        for (record, decision), item in zip(selected, payloads): writer.writerow({**{key: item.get(key, "") for key in writer.fieldnames}, "tags": ", ".join(record.tags), "selection_reasons": "; ".join(decision.reasons), "text_excerpt": (record.text or "")[:320], "source_ref": record.provenance.source_ref or record.record_id})
    groups: dict[str, list[tuple[EvidenceRecord, SelectionDecision]]] = {}
    for record, decision in selected:
        value = record.conversation_id if request.group_by == "conversation" else record.timestamp.date().isoformat() if request.group_by == "date" and record.timestamp else record.annotations.get(request.group_by)
        group = str(value or "Unclassified"); groups.setdefault(group, []).append((record, decision))
    lines = ["# Selected evidence", "", "Generated from read-only governed bus records.", ""]
    for group in sorted(groups, key=str.casefold):
        lines += [f"## {group}", ""]
        for record, decision in groups[group]:
            lines += [f"### {record.title or record.summary or (record.text or 'Untitled evidence')[:120]}", "", record.summary or record.text or "", "", f"- Selection: {', '.join(decision.reasons)}", f"- Source: `{record.provenance.source_ref or record.record_id}`", ""]
    (output / "artifact.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    manifest = {"selection_request": {"chunk_globs": list(request.chunk_globs), "summary_globs": list(request.summary_globs), "from": request.start.isoformat() if request.start else None, "to": request.end.isoformat() if request.end else None, "tags": list(request.tags), "fields": dict(request.fields), "text_pattern": request.text_pattern, "families": list(request.families), "maturities": list(request.maturities), "limit": request.limit, "deduplicate": request.deduplicate, "group_by": request.group_by}, "generated_at": datetime.now(timezone.utc).isoformat(), "matched_partitions": [{"source_kind": kind, "path": str(path), "sha256": _fingerprint(path)} for kind, path in paths], "counts": {"scanned": len(records) + len(errors), "invalid": len(errors), "deduplicated": duplicate_count, "selected": len(selected)}, "outputs": ["selected.jsonl", "selected.csv", "artifact.md", "manifest.json"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
