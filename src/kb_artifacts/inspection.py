"""Bounded, privacy-preserving inspection of read-only JSONL bus partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Iterable

from kb_artifacts.normalization import normalized_tags
from kb_artifacts.sources.jsonl_bus import SourceInputError, expand_globs, scan_jsonl


FIELDS = (
    "note_type", "format_type", "msg_type", "stage", "snippet_type", "actionable",
    "reusability_score", "include_in_vault", "category", "subtopic", "domain", "medium", "topics", "tags",
)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _top(counter: Counter[str], limit: int = 20) -> list[dict[str, object]]:
    return [{"value": value, "count": count} for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def inspect_source(
    *,
    chunk_globs: Iterable[str],
    summary_globs: Iterable[str],
    max_files_per_kind: int | None = None,
    max_records: int | None = None,
    include_excerpts: bool = False,
    allow_empty: bool = False,
) -> dict:
    """Inspect only bounded metadata; source text is excluded unless requested."""
    chunk_globs, summary_globs = list(chunk_globs), list(summary_globs)
    matched = {
        "chunk": expand_globs(chunk_globs),
        "summary": expand_globs(summary_globs),
    }
    # Apply the file bound independently to each requested bus role.  This
    # guarantees that a mixed-source inspection cannot silently omit summaries
    # merely because chunk paths sort first.
    selected = {
        kind: paths[:max_files_per_kind] if max_files_per_kind is not None else paths
        for kind, paths in matched.items()
    }
    planned = [(kind, path) for kind in ("chunk", "summary") for path in selected[kind]]
    if not planned and not allow_empty:
        raise SourceInputError("No input files matched the requested globs")

    records = []
    errors = []
    remaining = max_records
    for kind, path in planned:
        if remaining is not None and remaining <= 0:
            break
        scanned, diagnostics = scan_jsonl([path], source_kind=kind, max_records=remaining)
        scanned, diagnostics = list(scanned), list(diagnostics)
        records.extend(scanned)
        errors.extend(diagnostics)
        if remaining is not None:
            remaining -= len(scanned) + len(diagnostics)
    if not records and not errors and not allow_empty:
        raise SourceInputError("No records were available within the requested inspection bounds")

    schema_keys: dict[str, Counter[str]] = {"top_level": Counter(), "meta": Counter()}
    missingness = Counter()
    value_frequencies = {field: Counter() for field in FIELDS}
    tag_raw, tag_normalized = Counter(), Counter()
    collisions: dict[str, set[str]] = defaultdict(set)
    diagnostics = Counter(error["reason"] for error in errors)
    source_kind_counts = Counter(record.source_kind for record in records)
    timestamp_repr, identity_repr = Counter(), Counter()
    samples = []
    for record in records:
        schema_keys["top_level"].update(record.raw_record.keys())
        schema_keys["meta"].update(record.annotations.keys())
        for field in FIELDS:
            source = record.tags if field == "tags" else record.annotations.get(field)
            if source in (None, "", [], ()): 
                missingness[field] += 1
            else:
                value_frequencies[field][_value(source)] += 1
        for tag in record.tags:
            tag_raw[tag] += 1
            normalized = normalized_tags(tag)[0]
            tag_normalized[normalized] += 1
            collisions[normalized].add(tag)
        missingness["summary"] += int(record.summary is None)
        missingness["text"] += int(record.text is None)
        missingness["upstream_provenance"] += int(record.provenance.source_ref is None)
        timestamp_repr["present" if record.timestamp else "missing_or_invalid"] += 1
        identity_repr[
            "source_ref" if record.provenance.source_ref else "conversation_message" if record.conversation_id and record.message_id else "text_sha256" if record.provenance.text_sha256 else "content_fingerprint"
        ] += 1
        sample = {"record_id": record.record_id, "source_kind": record.source_kind, "fields_present": sorted(record.annotations), "tags": list(normalized_tags(record.tags)), "has_text": record.text is not None, "has_summary": record.summary is not None}
        if include_excerpts:
            sample["text_excerpt"] = (record.text or "")[:160]
            sample["summary_excerpt"] = (record.summary or "")[:160]
        samples.append(sample)

    inventory = [{"source_kind": kind, "path": str(path), "sha256": _fingerprint(path)} for kind, path in planned]
    file_counts = {
        kind: {
            "matched_before_limit": len(matched[kind]),
            "sampled_after_limit": len(selected[kind]),
        }
        for kind in ("chunk", "summary")
    }
    return {
        "source_inventory": inventory,
        "counts": {"records_observed": len(records), "invalid_or_unsupported": len(errors), "by_source_kind": dict(sorted(source_kind_counts.items())), "files_by_source_kind": file_counts, "bounded_by_max_files_per_kind": max_files_per_kind, "bounded_by_max_records": max_records},
        "schema_keys": {key: _top(counter) for key, counter in schema_keys.items()},
        "field_missingness": {field: {"missing": missingness[field], "present": len(records) - missingness[field]} for field in (*FIELDS, "summary", "text", "upstream_provenance")},
        "field_value_frequencies": {field: _top(counter) for field, counter in value_frequencies.items()},
        "tag_statistics": {"raw": _top(tag_raw), "normalized": _top(tag_normalized)},
        "normalization_collisions": {normalized: sorted(values) for normalized, values in sorted(collisions.items()) if len(values) > 1},
        "diagnostics": {"by_reason": dict(sorted(diagnostics.items()))},
        "bounded_samples": samples[:20],
        "representations": {"timestamp": dict(sorted(timestamp_repr.items())), "identity": dict(sorted(identity_repr.items()))},
    }
