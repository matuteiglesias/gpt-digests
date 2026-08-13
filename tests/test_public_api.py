from __future__ import annotations

import json
from importlib.resources import files

import kb_artifacts
from kb_artifacts import SelectionRequest, inspect_source, select


def test_public_api_is_intentional() -> None:
    assert kb_artifacts.__all__ == (
        "CorpusProfileError",
        "CorpusProfiles",
        "EvidenceRecord",
        "QueryExpression",
        "QueryValidationError",
        "SelectionDecision",
        "SelectionRequest",
        "SourceReference",
        "count_corpus",
        "describe_corpus",
        "evaluate_query",
        "facet_corpus",
        "inspect_source",
        "load_corpus_profiles",
        "parse_query",
        "sample_corpus",
        "select",
    )
    assert callable(inspect_source)
    assert callable(select)


def test_distribution_declares_inline_types() -> None:
    assert files("kb_artifacts").joinpath("py.typed").is_file()


def test_public_select_accepts_a_string_output_path(tmp_path) -> None:
    source = tmp_path / "evidence.jsonl"
    source.write_text(
        json.dumps({"text": "Restart the service.", "tags": ["runbook"]}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "run"

    result = select(
        SelectionRequest(chunk_globs=(str(source),), tags=("runbook",)),
        output=str(output),
    )

    assert result["counts"]["selected"] == 1
    assert (output / "selected.jsonl").is_file()
