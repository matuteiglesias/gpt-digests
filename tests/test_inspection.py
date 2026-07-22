from __future__ import annotations

import json
from pathlib import Path
# import tomllib

from typer.testing import CliRunner

from kb_artifacts.cli import app
from kb_artifacts.normalization import canonical_tag, normalized_tags, tag_lexeme


def _record(identifier: str, text: str, **meta):
    return {"text": text, "meta": {"message_id": identifier, "conversation_id": "conversation", "timestamp": "2026-01-01T00:00:00Z", "tags": meta.pop("tags", []), "provenance": {"source_ref": f"bus:{identifier}"}, **meta}}


def test_tag_normalization_ports_safe_legacy_behavior() -> None:
    assert canonical_tag("Topic: Gestión Ágil") == "topic:gestion_agil"
    assert tag_lexeme("Topic: Gestión Ágil") == "gestion agil"
    assert normalized_tags("['SOP', 'topic:Gestión Ágil', 'SOP']") == ("free:sop", "topic:gestion_agil")


def test_inspect_source_is_bounded_private_and_distinguishes_kinds(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"; chunks.mkdir()
    summaries = tmp_path / "summaries"; summaries.mkdir()
    (chunks / "a.jsonl").write_text("\n".join((
        json.dumps(_record("one", "SECRET CHUNK TEXT", tags=["SOP", "sop"], note_type="procedure", actionable=True, reusability_score=4)),
        "{not json}",
        json.dumps({"text": "bad", "meta": []}),
    )) + "\n", encoding="utf-8")
    (summaries / "a.jsonl").write_text(json.dumps(_record("summary", "SUMMARY TEXT", summary="not copied by default", format_type="guide")) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"

    result = CliRunner().invoke(app, ["inspect", "source", "--chunk-glob", str(chunks / "*.jsonl"), "--summary-glob", str(summaries / "*.jsonl"), "--max-files-per-kind", "2", "--max-records", "4", "--output", str(output)])

    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"]["by_source_kind"] == {"chunk": 1, "summary": 1}
    assert report["counts"]["invalid_or_unsupported"] == 2
    assert report["diagnostics"]["by_reason"] == {"invalid_json": 1, "invalid_meta": 1}
    assert report["normalization_collisions"] == {"free:sop": ["SOP", "sop"]}
    assert all("text_excerpt" not in sample and "summary_excerpt" not in sample for sample in report["bounded_samples"])
    assert "SECRET CHUNK TEXT" not in output.read_text(encoding="utf-8")
    assert len(report["source_inventory"]) == 2
    assert all(len(item["sha256"]) == 64 for item in report["source_inventory"])


def test_inspect_zero_glob_and_allow_empty(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = str(tmp_path / "missing/*.jsonl")
    failed = runner.invoke(app, ["inspect", "source", "--chunk-glob", missing, "--output", str(tmp_path / "report.json")])
    assert failed.exit_code == 2
    allowed = runner.invoke(app, ["inspect", "source", "--chunk-glob", missing, "--allow-empty", "--output", str(tmp_path / "report.json")])
    assert allowed.exit_code == 0, allowed.output
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["counts"]["records_observed"] == 0


def test_inspect_applies_file_and_record_bounds_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "chunks"; source.mkdir()
    (source / "a.jsonl").write_text(json.dumps(_record("a", "first")) + "\n", encoding="utf-8")
    (source / "b.jsonl").write_text(json.dumps(_record("b", "second")) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    result = CliRunner().invoke(app, ["inspect", "source", "--chunk-glob", str(source / "*.jsonl"), "--max-files-per-kind", "1", "--max-records", "1", "--output", str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [Path(item["path"]).name for item in report["source_inventory"]] == ["a.jsonl"]
    assert report["bounded_samples"][0]["record_id"] == "bus:a"


def test_inspect_file_bound_is_applied_to_each_requested_source_kind(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"; chunks.mkdir()
    summaries = tmp_path / "summaries"; summaries.mkdir()
    for name in ("a", "b"):
        (chunks / f"{name}.jsonl").write_text(json.dumps(_record(f"chunk-{name}", "chunk")) + "\n", encoding="utf-8")
        (summaries / f"{name}.jsonl").write_text(json.dumps(_record(f"summary-{name}", "summary")) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"

    result = CliRunner().invoke(app, ["inspect", "source", "--chunk-glob", str(chunks / "*.jsonl"), "--summary-glob", str(summaries / "*.jsonl"), "--max-files-per-kind", "1", "--output", str(output)])

    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"]["by_source_kind"] == {"chunk": 1, "summary": 1}
    assert report["counts"]["files_by_source_kind"] == {
        "chunk": {"matched_before_limit": 2, "sampled_after_limit": 1},
        "summary": {"matched_before_limit": 2, "sampled_after_limit": 1},
    }


# def test_packaging_exposes_one_canonical_console_entrypoint() -> None:
#     project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
#     assert project["project"]["scripts"]["kb-artifact"] == "kb_artifacts.cli:main"


def test_new_spine_has_no_legacy_runtime_imports() -> None:
    package = Path(__file__).parents[1] / "src" / "kb_artifacts"
    legacy_package = "digests" + "_project"
    assert legacy_package not in "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py"))
