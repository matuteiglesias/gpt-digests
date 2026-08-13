from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb_artifacts import (
    QueryExpression,
    QueryValidationError,
    SelectionRequest,
    evaluate_query,
    parse_query,
    select,
)
from kb_artifacts.sources.jsonl_bus import normalize_record


def _record(tmp_path: Path, **values: object):
    raw = {
        "text": values.pop("text", "Automate the release safely."),
        "title": values.pop("title", "Release playbook"),
        "tags": values.pop("tags", ["playbook", "automation"]),
        "meta": {
            "timestamp": values.pop("timestamp", "2026-06-01T12:00:00Z"),
            "domain": values.pop("domain", "automation"),
            "stage": values.pop("stage", "validated"),
            "reusability_score": values.pop("reusability_score", 4),
            **values,
        },
    }
    return normalize_record(raw, source_kind="chunk", partition=tmp_path / "fixture.jsonl", line_number=1)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ({"eq": {"field": "domain", "value": "AUTOMATION"}}, True),
        ({"in": {"field": "domain", "values": ["research", "automation"]}}, True),
        ({"contains": {"field": "tags", "value": "playbook"}}, True),
        ({"contains": {"field": "title", "value": "release"}}, True),
        ({"exists": {"field": "stage"}}, True),
        ({"exists": {"field": "missing"}}, False),
        ({"gte": {"field": "reusability_score", "value": 4}}, True),
        ({"lte": {"field": "reusability_score", "value": 3}}, False),
        ({"regex": {"target": "text", "pattern": r"release\s+safely"}}, True),
    ],
)
def test_every_primitive_predicate(tmp_path: Path, expression: dict, expected: bool) -> None:
    assert evaluate_query(_record(tmp_path), expression) is expected


def test_nested_boolean_operators_and_multiple_required_tags(tmp_path: Path) -> None:
    expression = {
        "all": [
            {"contains": {"field": "tags", "value": "playbook"}},
            {"contains": {"field": "tags", "value": "automation"}},
            {
                "any": [
                    {"eq": {"field": "domain", "value": "software_engineering"}},
                    {"eq": {"field": "domain", "value": "automation"}},
                ]
            },
            {"not": {"eq": {"field": "stage", "value": "reflection"}}},
        ]
    }
    assert evaluate_query(_record(tmp_path), expression)
    assert not evaluate_query(_record(tmp_path, tags=["playbook"]), expression)


def test_comparisons_fail_closed_without_type_coercion(tmp_path: Path) -> None:
    query = {"gte": {"field": "reusability_score", "value": 4}}
    assert not evaluate_query(_record(tmp_path, reusability_score="5"), query)
    assert not evaluate_query(_record(tmp_path, reusability_score=True), query)
    assert not evaluate_query(_record(tmp_path, reusability_score=None), query)
    with pytest.raises(QueryValidationError, match="must be numeric"):
        parse_query({"gte": {"field": "reusability_score", "value": "4"}})


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ({"unknown": {}}, "unknown operator"),
        ({"eq": {"field": "", "value": 1}}, "field name"),
        ({"eq": {"field": "domain"}}, "requires exactly"),
        ({"in": {"field": "domain", "values": []}}, "non-empty array"),
        ({"regex": {"target": "text", "pattern": "["}}, "is invalid"),
        ({"all": []}, "non-empty array"),
    ],
)
def test_malformed_queries_have_useful_errors(expression: dict, message: str) -> None:
    with pytest.raises(QueryValidationError, match=message):
        parse_query(expression)


def test_query_selection_is_deterministic_and_serialized(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    rows = [
        {"text": "second", "text_sha256": "b", "meta": {"message_id": "b", "timestamp": "2026-02-01T00:00:00Z", "score": 5}},
        {"text": "first", "text_sha256": "a", "meta": {"message_id": "a", "timestamp": "2026-01-01T00:00:00Z", "score": 4}},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    query = QueryExpression({"gte": {"field": "score", "value": 4}})
    outputs = []
    for name in ("one", "two"):
        output = tmp_path / name
        manifest = select(SelectionRequest(chunk_globs=(str(source),), query=query), output=output)
        outputs.append((output / "selected.jsonl").read_bytes())
        assert manifest["selection_request"]["query"] == query.to_dict()
    assert outputs[0] == outputs[1]
    assert [json.loads(line)["record_id"] for line in outputs[0].splitlines()] == ["sha256:a", "sha256:b"]


def test_legacy_request_does_not_change_output_or_manifest_shape(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"text": "legacy", "tags": ["runbook"], "text_sha256": "stable"}) + "\n", encoding="utf-8")
    request = SelectionRequest(chunk_globs=(str(source),), tags=("runbook",))
    first = select(request, output=tmp_path / "first")
    second = select(SelectionRequest(chunk_globs=(str(source),), tags=("runbook",), query=None), output=tmp_path / "second")
    assert "query" not in first["selection_request"]
    assert first["selection_request"] == second["selection_request"]
    assert (tmp_path / "first" / "selected.jsonl").read_bytes() == (tmp_path / "second" / "selected.jsonl").read_bytes()
