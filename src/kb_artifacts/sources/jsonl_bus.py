"""Read and normalize JSONL bus records without depending on legacy orchestration."""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kb_artifacts.contracts import EvidenceRecord, SourceReference


class SourceInputError(ValueError):
    """Raised when an explicitly requested input source cannot be used."""


def expand_globs(patterns: Iterable[str]) -> list[Path]:
    """Return sorted, deduplicated readable file paths for explicit patterns."""
    paths = {Path(path) for pattern in patterns for path in glob.glob(pattern, recursive=True)}
    return sorted(path for path in paths if path.is_file())


def parse_timestamp(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1_000_000_000_000:
            seconds /= 1_000
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _tags(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _stable_id(raw: Mapping[str, Any], source_kind: str, provenance: SourceReference, conversation_id: str | None, message_id: str | None) -> str:
    if provenance.source_ref:
        return provenance.source_ref
    if conversation_id and message_id:
        return f"{conversation_id}:{message_id}"
    if provenance.text_sha256:
        return f"sha256:{provenance.text_sha256}"
    stable = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    return f"{source_kind}:{digest}"


def normalize_record(raw: Mapping[str, Any], *, source_kind: str, partition: Path, line_number: int) -> EvidenceRecord:
    """Normalize known stable projections while retaining all upstream fields."""
    meta = _mapping(raw.get("meta"))
    outputs = _mapping(raw.get("outputs"))
    provenance_raw = _mapping(meta.get("provenance"))
    conversation_id = _text(raw.get("conversation_id")) or _text(meta.get("conversation_id"))
    message_id = _text(raw.get("message_id")) or _text(meta.get("message_id"))
    provenance = SourceReference(
        source_ref=_text(provenance_raw.get("source_ref")) or _text(raw.get("source_ref")),
        partition=str(partition),
        line_number=line_number,
        text_sha256=_text(raw.get("text_sha256")) or _text(meta.get("text_sha256")),
    )
    text = _text(raw.get("text")) or _text(raw.get("content")) or _text(outputs.get("summary_text"))
    summary = _text(raw.get("summary")) or _text(meta.get("summary")) or _text(outputs.get("summary"))
    title = _text(raw.get("title")) or _text(meta.get("title"))
    timestamp_value = next((value for value in (
        raw.get("timestamp"), raw.get("ts_abs"), raw.get("ts"), raw.get("created_at"),
        meta.get("timestamp"), meta.get("ts_abs"), raw.get("ts_abs_ms"),
    ) if value not in (None, "")), None)
    annotations = dict(meta)
    annotations.pop("provenance", None)
    return EvidenceRecord(
        record_id=_stable_id(raw, source_kind, provenance, conversation_id, message_id),
        source_kind=source_kind,
        text=text,
        summary=summary,
        title=title,
        timestamp=parse_timestamp(timestamp_value),
        conversation_id=conversation_id,
        message_id=message_id,
        tags=_tags(raw.get("tags")) or _tags(meta.get("tags")) or _tags(outputs.get("tags")),
        annotations=annotations,
        provenance=provenance,
        raw_record=dict(raw),
    )


def scan_jsonl(paths: Iterable[Path], *, source_kind: str, max_records: int | None = None) -> tuple[Iterator[EvidenceRecord], Iterator[dict[str, object]]]:
    """Return lazy record/error streams. The engine consumes both deterministically."""
    records: list[EvidenceRecord] = []
    errors: list[dict[str, object]] = []
    attempted = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if max_records is not None and attempted >= max_records:
                    return iter(records), iter(errors)
                attempted += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append({"partition": str(path), "line_number": line_number, "reason": "invalid_json", "detail": str(error)})
                    continue
                if not isinstance(raw, Mapping):
                    errors.append({"partition": str(path), "line_number": line_number, "reason": "not_object"})
                    continue
                if not isinstance(raw.get("meta", {}), Mapping):
                    errors.append({"partition": str(path), "line_number": line_number, "reason": "invalid_meta"})
                    continue
                record = normalize_record(raw, source_kind=source_kind, partition=path, line_number=line_number)
                if not (record.text or record.summary or record.title):
                    errors.append({"partition": str(path), "line_number": line_number, "reason": "missing_usable_content"})
                    continue
                records.append(record)
    return iter(records), iter(errors)
