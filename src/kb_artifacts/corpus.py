"""Deterministic, read-only corpus exploration operations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from kb_artifacts.contracts import EvidenceRecord
from kb_artifacts.inspection import inspect_source
from kb_artifacts.normalization import normalize_value, tag_lexeme
from kb_artifacts.profiles import CorpusProfileError, CorpusProfiles, resolve_corpus_sources
from kb_artifacts.query import QueryExpression, evaluate_query, parse_query
from kb_artifacts.sources.jsonl_bus import SourceInputError, expand_globs, scan_jsonl


QueryInput = QueryExpression | Mapping[str, object] | None
_COMMON_FIELDS = {
    "record_id", "source_kind", "title", "summary", "text", "timestamp",
    "conversation_id", "message_id", "tags",
}


def _sources(
    chunk_globs: Iterable[str], summary_globs: Iterable[str], corpus: str | None,
    profiles: CorpusProfiles | None,
) -> tuple[tuple[str, ...], tuple[str, ...], object | None]:
    return resolve_corpus_sources(chunk_globs=chunk_globs, summary_globs=summary_globs, corpus=corpus, profiles=profiles)


def describe_corpus(
    *,
    chunk_globs: Iterable[str] = (),
    summary_globs: Iterable[str] = (),
    corpus: str | None = None,
    profiles: CorpusProfiles | None = None,
    max_files_per_kind: int | None = None,
    max_records: int | None = None,
    include_excerpts: bool = False,
    allow_empty: bool = False,
) -> dict:
    """Describe a corpus using the established inspection implementation."""
    chunks, summaries, profile = _sources(chunk_globs, summary_globs, corpus, profiles)
    if profile is not None and include_excerpts and not profile.excerpts_permitted_by_default:
        raise CorpusProfileError(f"Corpus profile {corpus} does not permit excerpts")
    report = inspect_source(
        chunk_globs=chunks,
        summary_globs=summaries,
        max_files_per_kind=max_files_per_kind,
        max_records=max_records,
        include_excerpts=include_excerpts,
        allow_empty=allow_empty,
    )
    if profile is not None:
        report["corpus"] = corpus
        for index, item in enumerate(report["source_inventory"], start=1):
            item["source_id"] = f"{item['source_kind']}:{index}"
            item.pop("path", None)
    return report


def _sort_key(record: EvidenceRecord) -> tuple[object, str, str, int]:
    # ISO UTC strings preserve timestamp order and avoid mixing aware/min datetimes.
    timestamp = record.timestamp.isoformat() if record.timestamp else ""
    return timestamp, record.record_id, record.provenance.partition, record.provenance.line_number


def _read_corpus(
    *,
    chunk_globs: Iterable[str],
    summary_globs: Iterable[str],
    allow_empty: bool,
    corpus: str | None = None,
    profiles: CorpusProfiles | None = None,
) -> tuple[list[EvidenceRecord], list[dict[str, object]], int]:
    chunk_globs, summary_globs, _profile = _sources(chunk_globs, summary_globs, corpus, profiles)
    paths = [
        *(("chunk", path) for path in expand_globs(chunk_globs)),
        *(("summary", path) for path in expand_globs(summary_globs)),
    ]
    if not paths and not allow_empty:
        raise SourceInputError("No input files matched the requested globs")
    records: list[EvidenceRecord] = []
    errors: list[dict[str, object]] = []
    for kind, path in paths:
        found, diagnostics = scan_jsonl([path], source_kind=kind)
        records.extend(found)
        errors.extend(diagnostics)
    records.sort(key=_sort_key)
    return records, errors, len(paths)


def _query(query: QueryInput) -> QueryExpression | None:
    return parse_query(query) if query is not None else None


def _field_value(record: EvidenceRecord, field: str) -> object:
    if field in _COMMON_FIELDS:
        return getattr(record, field)
    return record.annotations.get(field)


def _facet_values(record: EvidenceRecord, field: str) -> tuple[object, ...]:
    value = _field_value(record, field)
    if value is None or value == "" or value == () or value == []:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _normalized_facet_value(field: str, value: object) -> str:
    return tag_lexeme(value) if field == "tags" else normalize_value(value)


def facet_corpus(
    *,
    field: str,
    chunk_globs: Iterable[str] = (),
    summary_globs: Iterable[str] = (),
    corpus: str | None = None,
    profiles: CorpusProfiles | None = None,
    query: QueryInput = None,
    limit: int = 20,
    allow_empty: bool = False,
) -> dict:
    """Count normalized values for one field, optionally under a query."""
    if not field or not field.replace("_", "a").replace("-", "a").replace(".", "a").isalnum():
        raise ValueError("field must be a non-empty field name")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    expression = _query(query)
    records, errors, file_count = _read_corpus(
        chunk_globs=chunk_globs, summary_globs=summary_globs, allow_empty=allow_empty, corpus=corpus, profiles=profiles
    )
    considered = [record for record in records if expression is None or evaluate_query(record, expression)]
    values: Counter[str] = Counter()
    missing = 0
    for record in considered:
        items = _facet_values(record, field)
        if not items:
            missing += 1
            continue
        # A repeated value within one record counts once for corpus faceting.
        values.update(set(_normalized_facet_value(field, item) for item in items))
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    return {
        "field": field,
        "values": [{"value": value, "count": count} for value, count in ordered[:limit]],
        "missing": missing,
        "counts": {
            "files_matched": file_count,
            "records_scanned": len(records),
            "records_considered": len(considered),
            "invalid": len(errors),
            "distinct_values": len(values),
        },
        "bounds": {"result_limit": limit, "truncated": len(values) > limit},
    }


def count_corpus(
    *,
    chunk_globs: Iterable[str] = (),
    summary_globs: Iterable[str] = (),
    corpus: str | None = None,
    profiles: CorpusProfiles | None = None,
    query: QueryInput = None,
    allow_empty: bool = False,
) -> dict:
    """Count query matches without materializing a selected-evidence artifact."""
    expression = _query(query)
    records, errors, file_count = _read_corpus(
        chunk_globs=chunk_globs, summary_globs=summary_globs, allow_empty=allow_empty, corpus=corpus, profiles=profiles
    )
    matching = sum(expression is None or evaluate_query(record, expression) for record in records)
    return {
        "counts": {
            "files_matched": file_count,
            "records_scanned": len(records),
            "records_considered": len(records),
            "records_matching": matching,
            "invalid": len(errors),
        },
        "deduplication": {"applied": False, "reason": "exploration counts normalized source records"},
    }


def sample_corpus(
    *,
    chunk_globs: Iterable[str] = (),
    summary_globs: Iterable[str] = (),
    corpus: str | None = None,
    profiles: CorpusProfiles | None = None,
    query: QueryInput = None,
    limit: int = 10,
    excerpt_chars: int | None = None,
    allow_empty: bool = False,
) -> dict:
    """Return the first deterministic, bounded query matches without full bodies."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if excerpt_chars is not None and not 1 <= excerpt_chars <= 1000:
        raise ValueError("excerpt_chars must be between 1 and 1000")
    expression = _query(query)
    _chunks, _summaries, profile = _sources(chunk_globs, summary_globs, corpus, profiles)
    if profile is not None and excerpt_chars is not None and not profile.excerpts_permitted_by_default:
        raise CorpusProfileError(f"Corpus profile {corpus} does not permit excerpts")
    records, errors, file_count = _read_corpus(
        chunk_globs=chunk_globs, summary_globs=summary_globs, allow_empty=allow_empty, corpus=corpus, profiles=profiles
    )
    matching = [record for record in records if expression is None or evaluate_query(record, expression)]
    samples = []
    for record in matching[:limit]:
        item = {
            "record_id": record.record_id,
            "source_kind": record.source_kind,
            "title": record.title,
            "summary": record.summary,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
            "tags": list(record.tags),
            "annotations": dict(record.annotations),
            "provenance": {
                "source_ref": record.provenance.source_ref,
                "line_number": record.provenance.line_number,
                "text_sha256": record.provenance.text_sha256,
            },
        }
        if excerpt_chars is not None:
            item["text_excerpt"] = (record.text or "")[:excerpt_chars]
        samples.append(item)
    return {
        "samples": samples,
        "counts": {
            "files_matched": file_count,
            "records_scanned": len(records),
            "records_matching": len(matching),
            "invalid": len(errors),
            "returned": len(samples),
        },
        "bounds": {"limit": limit, "excerpt_chars": excerpt_chars},
        "ordering": "timestamp, record_id, source partition, line number; no randomness",
    }
