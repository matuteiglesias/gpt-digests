"""Deterministic first-pass SOP candidate policy."""

from __future__ import annotations

import re
from kb_artifacts.contracts import ArtifactRecipe, EvidenceRecord, SelectionDecision
from kb_artifacts.normalization import normalize_value, tag_lexeme

STRONG = {"sop", "runbook", "procedure", "procedimiento", "checklist", "lista de verificacion", "workflow", "playbook", "how to", "workflow note", "workflow rule", "migration plan", "refactor plan", "execution plan"}
GUIDE = {"guide", "guia", "instructions", "instruction", "instrucciones", "template", "plantilla", "framework"}
RECIPE_VALUES = {"recipe", "receta"}
PHRASE = re.compile(r"\b(sop|runbook|procedure|procedimiento|checklist|lista de verificaci[oó]n|step[- ]by[- ]step|paso a paso|workflow)\b", re.I)
CATEGORICAL_FIELDS = ("note_type", "format_type", "msg_type", "stage", "snippet_type", "medium", "category", "domain")


def _norm(value: object) -> str:
    return normalize_value(value)


def _values(record: EvidenceRecord, *keys: str) -> set[str]:
    values = {tag_lexeme(tag) for tag in record.tags}
    for key in keys:
        value = record.annotations.get(key)
        if isinstance(value, (list, tuple)):
            values.update(_norm(item) for item in value)
        else:
            values.add(_norm(value))
    return values - {""}


def _number(record: EvidenceRecord, key: str) -> float | None:
    try:
        return float(record.annotations.get(key))
    except (TypeError, ValueError):
        return None


def evaluate(record: EvidenceRecord) -> SelectionDecision:
    # Every categorical field is accent/case/spacing normalized before policy
    # matching. Open domain and medium vocabularies are retained as context, not
    # treated as gates.
    values = _values(record, *CATEGORICAL_FIELDS)
    reasons: list[str] = []
    score = 0.0
    recipe_only = bool(values & RECIPE_VALUES) and not bool(values & STRONG)
    if recipe_only:
        return SelectionDecision(record.record_id, "rejected", -5, ("recipe_only_material",), {"values": sorted(values)})
    strong = sorted(values & STRONG)
    guides = sorted(values & GUIDE)
    if strong:
        score += 5
        reasons.append("strong_procedure_annotation:" + ",".join(strong))
    if guides:
        score += 3
        reasons.append("guide_or_template_annotation:" + ",".join(guides))
    if record.annotations.get("actionable") is True:
        score += 2
        reasons.append("actionable")
    if values & {"instruction"}:
        score += 1
        reasons.append("instructional_message_type")
    if values & {"execute", "plan"}:
        score += 1
        reasons.append("execution_or_plan_stage")
    reuse = _number(record, "reusability_score")
    if reuse is not None and reuse >= 4:
        score += 1
        reasons.append("reusability_score>=4")
    text_fields = " ".join(part for part in (record.title, record.summary, record.text) if part)
    if PHRASE.search(text_fields):
        score += 2
        reasons.append("procedural_phrase")
    if reuse is not None and reuse < 3:
        score -= 3
        reasons.append("reusability_score<3")
    if reuse == 1:
        reasons.append("reusability_score=1")
    disposition = "selected" if score >= 5 and reuse != 1 else "rejected"
    if disposition == "rejected" and not reasons:
        reasons.append("insufficient_procedural_evidence")
    return SelectionDecision(record.record_id, disposition, score, tuple(reasons), {"values": sorted(values), "reusability_score": reuse})


def group(record: EvidenceRecord) -> str:
    for key in ("domain", "category", "subtopic"):
        value = record.annotations.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unclassified"


RECIPE = ArtifactRecipe(id="sop.v0", title="Reusable SOPs and Procedures", evaluate=evaluate, group=group)
