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
