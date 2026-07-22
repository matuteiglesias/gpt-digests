from __future__ import annotations

from kb_artifacts.classification import classify
from kb_artifacts.contracts import EvidenceRecord, SourceReference


def _record(text: str, **annotations: object) -> EvidenceRecord:
    return EvidenceRecord(
        record_id="record-1", source_kind="chunk", text=text, summary=None, title=None,
        timestamp=None, conversation_id=None, message_id=None, tags=(), annotations=annotations,
        provenance=SourceReference(None, "fixture", 1, None), raw_record={},
    )


def test_recipe_domain_precedes_procedure_label() -> None:
    result = classify(_record("1. Mix ingredients. 2. Bake. Verify the crust.", category="cooking", note_type="procedure"))
    assert (result.family, result.maturity) == ("recipe", "candidate")


def test_plan_precedes_operational_alias() -> None:
    result = classify(_record("1. Migrate database. 2. Verify the deployment.", note_type="migration plan", format_type="procedure", stage="plan"))
    assert result.family == "plan"


def test_playbook_precedes_template_and_operations() -> None:
    result = classify(_record("If a lead is qualified, route it for review.", note_type="operational playbook", format_type="template"))
    assert result.family == "playbook"


def test_operations_maturity_uses_structural_completeness() -> None:
    ready = classify(_record("Purpose: restore service. Before you begin, back up data.\n1. Stop service.\n2. Restore backup.\nVerify health checks.", note_type="runbook"))
    candidate = classify(_record("1. Stop service.\n2. Restore backup.\nVerify health checks.", note_type="runbook"))
    fragment = classify(_record("Run this command to inspect the service.", note_type="runbook"))
    assert ready.maturity == "ready"
    assert candidate.maturity == "candidate"
    assert fragment.maturity == "fragment"


def test_reference_is_not_promoted_by_actionable_or_reuse() -> None:
    result = classify(_record("The experiment improved recall by 4%.", actionable=True, reusability_score=5, snippet_type="workflow"))
    assert (result.family, result.maturity) == ("reference", "reference_only")
