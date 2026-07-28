from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kb_artifacts.cli import app


FIXTURES = Path(__file__).parent / "fixtures" / "profile1"
EXPECTED_OUTPUTS = {
    "artifact.md",
    "manifest.json",
    "selected.csv",
    "selected.jsonl",
}


def _invoke(*arguments: str):
    return CliRunner().invoke(app, list(arguments))


def test_mixed_source_public_output_compatibility(tmp_path: Path) -> None:
    output = tmp_path / "mixed"
    result = _invoke(
        "select",
        "--chunk-glob",
        str(FIXTURES / "chunks.jsonl"),
        "--summary-glob",
        str(FIXTURES / "summaries.jsonl"),
        "--tag",
        "run_book",
        "--family",
        "operations",
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    assert {path.name for path in output.iterdir()} == EXPECTED_OUTPUTS

    jsonl_bytes = (output / "selected.jsonl").read_bytes()
    records = [json.loads(line) for line in jsonl_bytes.splitlines()]
    assert [record["record_id"] for record in records] == [
        "fixture:chunk:duplicate",
        "fixture:summary:candidate",
    ]
    assert records[0]["selection_reasons"] == ["tag:run book", "family:operations"]
    assert records[0]["artifact_family"] == "operations"
    assert records[0]["artifact_maturity"] == "fragment"
    assert records[1]["artifact_maturity"] == "candidate"

    csv_rows = list(csv.DictReader((output / "selected.csv").open(encoding="utf-8")))
    assert [row["record_id"] for row in csv_rows] == [record["record_id"] for record in records]
    assert (output / "artifact.md").read_text(encoding="utf-8") == (
        "# Selected evidence\n\n"
        "Generated from read-only governed bus records.\n\n"
        "## operations\n\n"
        "### Earlier duplicate view.\n\n"
        "Earlier duplicate view.\n\n"
        "- Selection: tag:run book, family:operations\n"
        "- Source: `fixture:chunk:duplicate`\n\n"
        "### 1. Inspect the queue.\n2. Clear the failed item.\nVerify completion.\n\n"
        "1. Inspect the queue.\n2. Clear the failed item.\nVerify completion.\n\n"
        "- Selection: tag:run book, family:operations\n"
        "- Source: `fixture:summary:candidate`\n"
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    generated_at = manifest.pop("generated_at")
    assert generated_at.endswith("+00:00")
    assert set(manifest) == {"counts", "matched_partitions", "outputs", "selection_request"}
    assert manifest["counts"] == {"deduplicated": 1, "invalid": 0, "scanned": 4, "selected": 2}
    assert manifest["outputs"] == ["selected.jsonl", "selected.csv", "artifact.md", "manifest.json"]
    assert [item["source_kind"] for item in manifest["matched_partitions"]] == ["chunk", "summary"]
    assert all(len(item["sha256"]) == 64 for item in manifest["matched_partitions"])


def test_intentional_empty_selection_exact_compatibility(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    result = _invoke(
        "select",
        "--chunk-glob",
        str(FIXTURES / "chunks.jsonl"),
        "--tag",
        "does-not-exist",
        "--allow-empty",
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    assert (output / "selected.jsonl").read_bytes() == b""
    assert (output / "selected.csv").read_text(encoding="utf-8") == (
        "record_id,source_kind,timestamp,title,summary,tags,artifact_family,"
        "artifact_maturity,selection_reasons,text_excerpt,source_ref\n"
    )
    assert (output / "artifact.md").read_text(encoding="utf-8") == (
        "# Selected evidence\n\nGenerated from read-only governed bus records.\n"
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"deduplicated": 1, "invalid": 0, "scanned": 3, "selected": 0}


def test_malformed_fixture_diagnostics_are_counted(tmp_path: Path) -> None:
    output = tmp_path / "malformed"
    result = _invoke(
        "select",
        "--chunk-glob",
        str(FIXTURES / "malformed.jsonl"),
        "--allow-empty",
        "--output",
        str(output),
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"deduplicated": 0, "invalid": 2, "scanned": 2, "selected": 0}


def test_current_user_failures_remain_exit_code_two(tmp_path: Path) -> None:
    source = str(FIXTURES / "chunks.jsonl")
    cases = [
        (["--chunk-glob", str(tmp_path / "missing-*.jsonl")], "No input files matched"),
        (["--chunk-glob", source, "--field", "invalid"], "NAME=VALUE"),
        (["--chunk-glob", source, "--from", "not-a-date"], "Invalid isoformat"),
        (["--chunk-glob", source, "--from", "2026-02-02", "--to", "2026-01-01"], "must not be after"),
        (["--chunk-glob", source, "--text", "["], "Invalid text pattern"),
        (["--chunk-glob", source, "--tag", "missing"], "No records matched"),
    ]

    for number, (arguments, message) in enumerate(cases):
        result = _invoke("select", *arguments, "--output", str(tmp_path / f"failure-{number}"))
        assert result.exit_code == 2, result.output
        assert message in result.output

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("do not replace", encoding="utf-8")
    result = _invoke("select", "--chunk-glob", source, "--output", str(occupied))
    assert result.exit_code == 2
    assert "Output directory is not empty" in result.output
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "do not replace"
