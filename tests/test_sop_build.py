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
    assert manifest["counts"]["scanned"] == 9
    assert manifest["counts"]["invalid_or_unusable"] == 1
    assert manifest["counts"]["deduplicated"] == 1
    assert manifest["counts"]["evaluated_unique"] == 7
    assert manifest["counts"]["selected"] == 1
    assert (output / "errors.jsonl").exists()
    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["record_id"] for item in decisions if item["disposition"] == "selected"} == {
        "chatgpt:conversation-1:sop-1",
    }
    duplicate = next(item for item in decisions if item["disposition"] == "deduplicated")
    assert duplicate["canonical_record_id"] == "chatgpt:conversation-1:sop-1"
    assert not any(item["record_id"] == "chatgpt:conversation-1:explain-1" for item in decisions)
    assert manifest["reconciliation"]["scanned_value"] == manifest["counts"]["scanned"]
    assert manifest["reconciliation"]["evaluated_unique_value"] == manifest["counts"]["evaluated_unique"]
    artifact = (output / "artifact.md").read_text(encoding="utf-8")
    assert "Operations: SOPs and Runbooks" in artifact
    assert "chatgpt:conversation-1:sop-1" in artifact


def test_audit_all_decisions_restores_ordinary_nonmatches(tmp_path: Path) -> None:
    fixture = tmp_path / "chunks.jsonl"
    _write_fixture(fixture)
    output = tmp_path / "run"

    result = CliRunner().invoke(app, ["build", "sop", "--chunk-glob", str(fixture), "--output", str(output), "--audit-all-decisions"])

    assert result.exit_code == 0, result.output
    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(item["record_id"] == "chatgpt:conversation-1:explain-1" for item in decisions)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision_ledger"]["audit_all_decisions"] is True
    assert manifest["decision_ledger"]["ordinary_nonmatches_omitted"] == 0


def test_review_packet_is_bounded_private_and_has_reviewer_columns(tmp_path: Path) -> None:
    fixture = tmp_path / "chunks.jsonl"
    _write_fixture(fixture)
    output = tmp_path / "run"
    packet = tmp_path / "review" / "sop-review.csv"

    result = CliRunner().invoke(app, ["build", "sop", "--chunk-glob", str(fixture), "--output", str(output), "--review-packet", str(packet)])

    assert result.exit_code == 0, result.output
    rows = list(__import__("csv").DictReader(packet.open(encoding="utf-8", newline="")))
    assert rows
    assert {"review_family", "review_maturity", "review_action", "review_comment", "expected_group", "score_components", "text_excerpt"} <= set(rows[0])
    assert all(row["review_family"] == row["review_maturity"] == row["review_action"] == row["review_comment"] == row["expected_group"] == "" for row in rows)
    assert all(len(row["text_excerpt"]) <= 320 for row in rows)
    assert "Procedure: 1. Back up the database. 2. Verify the restore." not in packet.read_text(encoding="utf-8")


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
