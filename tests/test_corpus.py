from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kb_artifacts import count_corpus, describe_corpus, facet_corpus, inspect_source, sample_corpus
from kb_artifacts.cli import app
from kb_artifacts.query import QueryValidationError


def _write_corpus(path: Path) -> None:
    rows = [
        {"title": "Later", "text": "PRIVATE AUTOMATION BODY", "tags": ["Play_book", "automation"], "text_sha256": "b", "meta": {"timestamp": "2026-02-01T00:00:00Z", "domain": "Automation", "stage": "ready", "score": 5, "provenance": {"source_ref": "fixture:b"}}},
        {"title": "Earlier", "summary": "Safe summary", "text": "PRIVATE SOFTWARE BODY", "tags": ["play-book"], "text_sha256": "a", "meta": {"timestamp": "2026-01-01T00:00:00Z", "domain": "software_engineering", "stage": "reflection", "score": 4, "provenance": {"source_ref": "fixture:a"}}},
        {"title": "Missing domain", "text": "PRIVATE OTHER BODY", "tags": [], "text_sha256": "c", "meta": {"timestamp": "2026-03-01T00:00:00Z", "stage": "ready", "score": "5"}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "{bad json}\n", encoding="utf-8")


def test_describe_is_inspection_compatible(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"; _write_corpus(source)
    arguments = {"chunk_globs": (str(source),), "summary_globs": (), "max_records": 2}
    assert describe_corpus(**arguments) == inspect_source(**arguments)


def test_facet_tags_metadata_missing_and_query(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"; _write_corpus(source)
    tags = facet_corpus(field="tags", chunk_globs=(str(source),), summary_globs=())
    assert tags["values"] == [
        {"value": "play book", "count": 2},
        {"value": "automation", "count": 1},
    ]
    assert tags["missing"] == 1
    domains = facet_corpus(
        field="domain", chunk_globs=(str(source),), summary_globs=(),
        query={"gte": {"field": "score", "value": 5}}, limit=1,
    )
    assert domains["values"] == [{"value": "automation", "count": 1}]
    assert domains["counts"]["records_considered"] == 1
    assert domains["counts"]["invalid"] == 1


def test_count_nested_query_and_no_deduplication(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"; _write_corpus(source)
    report = count_corpus(
        chunk_globs=(str(source),), summary_globs=(),
        query={"all": [{"gte": {"field": "score", "value": 4}}, {"not": {"eq": {"field": "stage", "value": "reflection"}}}]},
    )
    assert report["counts"] == {"files_matched": 1, "records_scanned": 3, "records_considered": 3, "records_matching": 1, "invalid": 1}
    assert report["deduplication"]["applied"] is False


def test_sample_is_bounded_deterministic_and_private_by_default(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"; _write_corpus(source)
    arguments = {"chunk_globs": (str(source),), "summary_globs": (), "limit": 2}
    first = sample_corpus(**arguments)
    second = sample_corpus(**arguments)
    assert first == second
    assert [item["record_id"] for item in first["samples"]] == ["fixture:a", "fixture:b"]
    assert first["counts"]["returned"] == 2
    serialized = json.dumps(first)
    assert "PRIVATE" not in serialized and "text_excerpt" not in serialized
    assert first["samples"][0]["provenance"]["source_ref"] == "fixture:a"
    excerpt = sample_corpus(**arguments, excerpt_chars=8)
    assert excerpt["samples"][0]["text_excerpt"] == "PRIVATE "
    assert excerpt["bounds"]["excerpt_chars"] == 8


def test_empty_exploration_and_malformed_query(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.jsonl")
    assert count_corpus(chunk_globs=(missing,), summary_globs=(), allow_empty=True)["counts"]["records_matching"] == 0
    assert facet_corpus(field="domain", chunk_globs=(missing,), summary_globs=(), allow_empty=True)["values"] == []
    assert sample_corpus(chunk_globs=(missing,), summary_globs=(), allow_empty=True)["samples"] == []
    with pytest.raises(QueryValidationError):
        count_corpus(chunk_globs=(missing,), summary_globs=(), query={"wat": {}}, allow_empty=True)


def test_corpus_cli_emits_parseable_json_and_accepts_query_file(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"; _write_corpus(source)
    query_file = tmp_path / "query.json"
    query_file.write_text(json.dumps({"eq": {"field": "domain", "value": "automation"}}), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["corpus", "count", "--chunk-glob", str(source), "--query-file", str(query_file)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["counts"]["records_matching"] == 1
    malformed = runner.invoke(app, ["corpus", "count", "--chunk-glob", str(source), "--query", "not json"])
    assert malformed.exit_code == 2 and "not valid JSON" in malformed.output


def test_corpus_describe_cli_json_does_not_require_output_file(tmp_path: Path) -> None:
    source = tmp_path / "corpus.jsonl"; _write_corpus(source)
    result = CliRunner().invoke(app, ["corpus", "describe", "--chunk-glob", str(source), "--max-records", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["counts"]["records_observed"] == 1
