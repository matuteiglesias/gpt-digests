"""Public API for inspecting and selecting JSONL evidence collections."""

from .contracts import EvidenceRecord, SelectionDecision, SourceReference
from .inspection import inspect_source
from .selection import SelectionRequest, select

__all__ = (
    "EvidenceRecord",
    "SelectionDecision",
    "SelectionRequest",
    "SourceReference",
    "inspect_source",
    "select",
)
