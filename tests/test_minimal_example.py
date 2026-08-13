from __future__ import annotations

import json
from pathlib import Path

from kb_artifacts import SelectionRequest, select


EXAMPLE = Path(__file__).parents[1] / "examples" / "basic" / "evidence.jsonl"


def test_basic_example_selects_one_runbook(tmp_path: Path) -> None:
    output = tmp_path / "selected"
    manifest = select(
        SelectionRequest(chunk_globs=(str(EXAMPLE),), tags=("runbook",)),
        output=output,
    )

    selected = [
        json.loads(line)
        for line in (output / "selected.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["counts"]["scanned"] == 2
    assert manifest["counts"]["selected"] == 1
    assert selected[0]["title"] == "Deploy app"
    assert {path.name for path in output.iterdir()} == {
        "artifact.md",
        "manifest.json",
        "selected.csv",
        "selected.jsonl",
    }


def test_legacy_selection_jsonl_is_byte_stable(tmp_path: Path) -> None:
    source = tmp_path / "legacy.jsonl"
    source.write_text(
        json.dumps({
            "title": "Stable record", "text": "Stable body", "tags": ["runbook"],
            "text_sha256": "stable-sha", "source_ref": "fixture:stable",
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "legacy-output"
    select(SelectionRequest(chunk_globs=(str(source),), tags=("runbook",)), output=output)
    expected = {
        "annotations": {}, "artifact_family": None, "artifact_maturity": None,
        "provenance": {"line_number": 1, "partition": str(source), "source_ref": "fixture:stable", "text_sha256": "stable-sha"},
        "record_id": "fixture:stable", "selection_reasons": ["tag:runbook"],
        "source_kind": "chunk", "summary": None, "tags": ["runbook"],
        "timestamp": None, "title": "Stable record",
    }
    expected_bytes = (json.dumps(expected, ensure_ascii=False, sort_keys=True) + "\n").encode()
    assert (output / "selected.jsonl").read_bytes() == expected_bytes
