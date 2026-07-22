from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from kb_artifacts.cli import app


def _record(identifier: str, text: str, **meta):
    return {
        "text_sha256": meta.pop("text_sha256", identifier),
        "text": text,
        "meta": {
            "message_id": identifier,
            "conversation_id": "conversation-1",
            "timestamp": meta.pop("timestamp", "2026-06-30T01:30:21Z"),
            "provenance": {"source_ref": f"chatgpt:conversation-1:{identifier}"},
            **meta,
        },
    }


def _write_fixture(path: Path) -> None:
    records = [
        _record("sop-1", "Procedure: 1. Back up the database. 2. Verify the restore.", note_type="procedure", format_type="guide", actionable=True, reusability_score=4, domain="Operations"),
        _record("recipe-1", "Recipe: simmer vegetables for 30 minutes.", msg_type="instruction", format_type="guide", medium="recipe", actionable=True, reusability_score=4),
        _record("explain-1", "A workflow is a useful way to think about collaboration.", reusability_score=2),
        _record("runbook-1", "Runbook checklist for on-call incident handoff.", format_type="checklist", actionable=True, reusability_score=3, domain="Operations"),
        _record("dup-1", "Procedure: 1. Back up the database. 2. Verify the restore.", text_sha256="sop-1", note_type="procedure", actionable=True, reusability_score=4),
        {"text": "Malformed annotations", "meta": ["not", "an", "object"]},
        _record("spanish-1", "Procedimiento paso a paso para cerrar una incidencia.", note_type="procedimiento", actionable=True, reusability_score=4, domain="Operaciones"),
        _record("no-summary-1", "Checklist: validate deployment, then notify the owner.", format_type="checklist", actionable=True, reusability_score=3),
        _record("low-reuse-1", "SOP for a one-off experiment.", note_type="procedure", actionable=True, reusability_score=1),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_sop_build_writes_auditable_minimal_run(tmp_path: Path) -> None:
    fixture = tmp_path / "chunks.jsonl"
    _write_fixture(fixture)
    output = tmp_path / "run"

    result = CliRunner().invoke(app, ["build", "sop", "--chunk-glob", str(fixture), "--output", str(output)])

    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"deduplicated": 1, "invalid": 1, "rejected": 3, "scanned": 9, "selected": 4, "usable": 8}
    assert (output / "errors.jsonl").exists()
    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["record_id"] for item in decisions if item["disposition"] == "selected"} == {
        "chatgpt:conversation-1:sop-1", "chatgpt:conversation-1:runbook-1",
        "chatgpt:conversation-1:spanish-1", "chatgpt:conversation-1:no-summary-1",
    }
    duplicate = next(item for item in decisions if item["disposition"] == "deduplicated")
    assert duplicate["canonical_record_id"] == "chatgpt:conversation-1:sop-1"
    ledger_counts = {name: sum(item["disposition"] == name for item in decisions) for name in ("selected", "rejected", "deduplicated")}
    assert manifest["counts"]["scanned"] == ledger_counts["selected"] + ledger_counts["rejected"] + ledger_counts["deduplicated"] + manifest["counts"]["invalid"]
    artifact = (output / "artifact.md").read_text(encoding="utf-8")
    assert "Reusable SOPs and Procedures" in artifact
    assert "chatgpt:conversation-1:spanish-1" in artifact


def test_no_matching_input_fails(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["build", "sop", "--chunk-glob", str(tmp_path / "missing/*.jsonl"), "--output", str(tmp_path / "run")])
    assert result.exit_code == 2
    assert "No input files matched" in result.output


def test_zero_selection_requires_explicit_allow_empty(tmp_path: Path) -> None:
    fixture = tmp_path / "chunks.jsonl"
    fixture.write_text(json.dumps(_record("generic", "A broad conceptual explanation.", reusability_score=1)) + "\n", encoding="utf-8")
    output = tmp_path / "run"
    failed = CliRunner().invoke(app, ["build", "sop", "--chunk-glob", str(fixture), "--output", str(output)])
    assert failed.exit_code == 2
    allowed = CliRunner().invoke(app, ["build", "sop", "--chunk-glob", str(fixture), "--output", str(output), "--allow-empty"])
    assert allowed.exit_code == 0, allowed.output
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["counts"]["selected"] == 0
