"""Operations recipe built on the corpus-aware artifact classifier."""

from __future__ import annotations

from kb_artifacts.classification import classify
from kb_artifacts.contracts import ArtifactRecipe, EvidenceRecord, SelectionDecision


def evaluate(record: EvidenceRecord) -> SelectionDecision:
    result = classify(record)
    selected = result.family == "operations" and result.maturity in {"ready", "candidate"}
    score = 5.0 if result.maturity == "ready" else 3.0 if selected else 0.0
    reasons = ("routed_to_operations",) if selected else ("routed_to_other_artifact_family",)
    return SelectionDecision(
        record_id=record.record_id, disposition="selected" if selected else "rejected", score=score,
        reasons=reasons, matched_values={"structure": sorted(result.structure)},
        artifact_family=result.family, artifact_maturity=result.maturity,
        classification_reasons=result.reasons,
    )


def group(record: EvidenceRecord) -> str:
    for key in ("domain", "category", "subtopic"):
        value = record.annotations.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unclassified"


RECIPE = ArtifactRecipe(id="sop.v1", title="Operations: SOPs and Runbooks", evaluate=evaluate, group=group)
