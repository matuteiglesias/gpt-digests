"""Small stable contracts for the canonical artifact-building path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal, Mapping


@dataclass(frozen=True)
class SourceReference:
    """Where a record came from without making the bus mutable or local."""

    source_ref: str | None
    partition: str
    line_number: int
    text_sha256: str | None


@dataclass(frozen=True)
class EvidenceRecord:
    """Stable projections plus the raw bus record needed for future recipes."""

    record_id: str
    source_kind: str
    text: str | None
    summary: str | None
    title: str | None
    timestamp: datetime | None
    conversation_id: str | None
    message_id: str | None
    tags: tuple[str, ...]
    annotations: Mapping[str, object]
    provenance: SourceReference
    raw_record: Mapping[str, object]


Disposition = Literal["selected", "rejected", "deduplicated", "ignored", "invalid"]


@dataclass(frozen=True)
class SelectionDecision:
    record_id: str
    disposition: Disposition
    score: float
    reasons: tuple[str, ...]
    matched_values: Mapping[str, object]
    canonical_record_id: str | None = None


Evaluate = Callable[[EvidenceRecord], SelectionDecision]
Group = Callable[[EvidenceRecord], str]


@dataclass(frozen=True)
class ArtifactRecipe:
    """Checked-in policy only; mechanics stay in the shared engine."""

    id: str
    title: str
    evaluate: Evaluate
    group: Group
