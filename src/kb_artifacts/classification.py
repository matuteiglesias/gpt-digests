"""Deterministic routing and completeness assessment for evidence records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kb_artifacts.contracts import EvidenceRecord
from kb_artifacts.normalization import normalize_value, tag_lexeme

ROUTING_FIELDS = ("category", "domain", "medium", "note_type", "format_type")
RECIPE = ("recipe", "receta", "oven", "kitchen", "culinary", "cooking", "baking", "meal prep")
PLAN = ("execution plan", "implementation plan", "migration plan", "refactor plan", "experiment plan", "architecture plan", "action plan", "remediation plan")
PLAYBOOK = ("playbook", "policy", "workflow rule", "operating pattern", "escalation rule", "review rule")
OPERATIONS = ("sop", "runbook", "procedure", "procedimiento", "how to", "checklist", "troubleshooting guide", "workflow instructions", "remediation procedure", "diagnostic workflow", "recovery procedure")
TEMPLATE = ("template", "prompt pack", "script", "cover letter", "draft response", "interview response", "configuration template", "checklist skeleton")
STRATEGY = ("framework", "strategy", "strategy memo", "design principle", "architecture guidance", "architecture proposal", "planning rule", "decision framework", "experiment strategy")

NUMBERED_ACTIONS = re.compile(r"(?:^|\n|\s)\d+[.)]\s+", re.M)
BULLET_ACTIONS = re.compile(r"(?:^|\n)\s*[-*]\s+\S+", re.M)
COMMAND = re.compile(r"(?:^|\n)\s*(?:\$\s+|```(?:bash|sh)?|(?:git|make|python|systemctl|docker|pytest)\s+)", re.M | re.I)
PURPOSE = re.compile(r"\b(?:purpose|trigger|when |to (?:start|deploy|recover|validate|fix|create|run))\b", re.I)
PREREQUISITES = re.compile(r"\b(?:prerequisite|before you|requirements?|input[s]?|ensure |need(?:s)? )\b", re.I)
DECISIONS = re.compile(r"\b(?:if |unless |otherwise|when .* then|warning|exception|rollback)\b", re.I)
VALIDATION = re.compile(r"\b(?:verify|validate|test|expected output|done|complete|completion|confirm|check that)\b", re.I)


@dataclass(frozen=True)
class Classification:
    family: str
    maturity: str
    reasons: tuple[str, ...]
    structure: frozenset[str]


def _values(record: EvidenceRecord) -> set[str]:
    values = {tag_lexeme(tag) for tag in record.tags}
    for key in ROUTING_FIELDS:
        value = record.annotations.get(key)
        items = value if isinstance(value, (list, tuple)) else (value,)
        values.update(normalize_value(item) for item in items)
    return {value for value in values if value}


def _has(values: set[str], signals: tuple[str, ...]) -> list[str]:
    return sorted(signal for signal in signals if any(signal == value or signal in value for value in values))


def inspect_content_structure(record: EvidenceRecord) -> frozenset[str]:
    text = "\n".join(part for part in (record.title, record.summary, record.text) if part)
    features: set[str] = set()
    if len(NUMBERED_ACTIONS.findall(text)) >= 2 or len(BULLET_ACTIONS.findall(text)) >= 2 or COMMAND.search(text):
        features.add("ordered_actions")
    if PURPOSE.search(text):
        features.add("trigger_or_purpose")
    if PREREQUISITES.search(text):
        features.add("inputs_or_prerequisites")
    if DECISIONS.search(text):
        features.add("exceptions_or_decision_rules")
    if VALIDATION.search(text):
        features.add("validation_or_done_condition")
    return frozenset(features)


def classify(record: EvidenceRecord) -> Classification:
    """Route first, then assess completeness; weak quality fields never route."""
    values = _values(record)
    reasons: list[str] = []
    # This precedence intentionally keeps a cooking procedure out of operations
    # and a migration procedure out of generic SOP output.
    for family, signals in (("recipe", RECIPE), ("plan", PLAN), ("playbook", PLAYBOOK), ("operations", OPERATIONS), ("template", TEMPLATE), ("strategy", STRATEGY)):
        matched = _has(values, signals)
        if matched:
            reasons.append(f"{family}_annotation:" + ",".join(matched))
            break
    else:
        family = "reference"
        reasons.append("default_reference")
    structure = inspect_content_structure(record)
    stage = normalize_value(record.annotations.get("stage"))
    msg_type = normalize_value(record.annotations.get("msg_type"))
    actionable = record.annotations.get("actionable") is True
    if family == "reference":
        maturity = "reference_only"
    elif family == "operations":
        other = len(structure - {"ordered_actions"})
        if "ordered_actions" in structure and other >= 2:
            maturity = "ready"
        elif "ordered_actions" in structure and other >= 1:
            maturity = "candidate"
        else:
            maturity = "fragment"
    elif family == "recipe":
        maturity = "ready" if "ordered_actions" in structure and len(structure) >= 3 else "candidate" if "ordered_actions" in structure else "fragment"
    elif family == "plan":
        maturity = "ready" if "ordered_actions" in structure and (stage == "plan" or len(structure) >= 2) else "candidate"
    elif family == "playbook":
        maturity = "ready" if "ordered_actions" in structure and ("exceptions_or_decision_rules" in structure or len(structure) >= 3) else "candidate"
    else:
        # An explicit reusable asset or framework is usable when it contains
        # substantive text; procedural structure is not a requirement.
        body = " ".join(part for part in (record.title, record.summary, record.text) if part).strip()
        maturity = "ready" if len(body) >= 40 and msg_type not in {"idea", "reflection"} else "candidate"
    if actionable:
        reasons.append("actionable_supports_maturity")
    if stage:
        reasons.append(f"stage:{stage}")
    reasons.extend(f"structure:{feature}" for feature in sorted(structure))
    return Classification(family, maturity, tuple(reasons), structure)
