"""Producer-owned identity for the canonical selected-evidence payload."""

from __future__ import annotations

import hashlib


SELECTED_EVIDENCE_ID_ALGORITHM = "selected-evidence-sha256.v1"
SELECTED_EVIDENCE_ID_PREFIX = "selected-evidence.sha256."


def selected_evidence_sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of exact ``selected.jsonl`` bytes."""
    return hashlib.sha256(payload).hexdigest()


def selected_evidence_artifact_id(payload: bytes) -> str:
    """Return the version-1 producer-owned ID for exact ``selected.jsonl`` bytes."""
    return SELECTED_EVIDENCE_ID_PREFIX + selected_evidence_sha256(payload)
