# bags_pipeline/core.py
"""
Core datatypes and lightweight type aliases.
No pandas, no numpy, no ingestion. Keep this dependency-free.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping, Union

# ——— Type aliases for lookup tables —————————————————————————————

EventIndex   = Dict[str, Dict[str, Any]]
SessionIndex = Dict[str, Dict[str, Any]]

# ——— Core dataclasses ————————————————————————————————————————

@dataclass(frozen=True)
class Event:
    """A single timestamped event from a log."""
    event_id:       str
    source:         str
    ts_abs:         Optional[str] = None       # absolute ISO8601 timestamp
    tz_local:       Optional[str] = None       # IANA timezone name, if known
    role:           Optional[str] = None
    conversation_id: Optional[str] = None
    title:          Optional[str] = None
    text:           str                     = ""
    tags:           Tuple[str, ...]         = ()
    extras:         Dict[str, Any]          = field(default_factory=dict)


@dataclass(frozen=True)
class Session:
    """A meeting or session with a start and end timestamp."""
    session_id: str
    start_ts:   Optional[str] = None   # ISO8601
    end_ts:     Optional[str] = None   # ISO8601
    labels:     Tuple[str, ...] = ()
    project:    Optional[str]   = None
    summary:    Dict[str, Any]  = field(default_factory=dict)
    extras:     Dict[str, Any]  = field(default_factory=dict)


@dataclass(frozen=True)
class Unit:
    """Atomic bag of content used to build digests.

    Attributes:
        unit_id: Stable identifier (sha256 of a stable JSON subset).
        unit_type: One of {"session","cohort","conversation","pairbag","tagbag"}.
        start_ts: ISO timestamp (local or UTC) for the earliest included item.
        end_ts: ISO timestamp for the latest included item.
        tags: Canonical tags used for selection and grouping.
        topic_ids: Optional topic identifiers (if you run topic modeling).
        sources: Provenance tuples, typically ("event", id) or ("session", id).
    """
    unit_id:   str                   # stable ID (e.g. sha256)
    unit_type: str                   # e.g. "session" | "cohort" | "pairbag" | "tagbag"
    start_ts:  str                   # ISO8601 of earliest item
    end_ts:    str                   # ISO8601 of latest item
    tags:      Tuple[str, ...]       = ()
    topic_ids: Tuple[str, ...]       = ()
    sources:   Tuple[Tuple[str,str], ...] = field(default_factory=tuple)
    extras:    Dict[str, Any]        = field(default_factory=dict)




from enum import Enum

class RenderMode(str, Enum):
    content = "content"
    summary = "summary"
    both    = "both"

@dataclass(frozen=True)
class L2Digest:
    digest_id: str
    level: str
    channel: str
    unit_id: str
    unit_type: str
    title: str
    start_ts: str
    end_ts: str
    topic_ids: tuple[str, ...]
    tags: tuple[str, ...]
    scores: Dict[str, Any]
    policy: Dict[str, Any]
    mdx: str



import re
# from .l2 import L2Digest

def slug_for_unit(d: L2Digest) -> str:
    # heurística: usa info de la unit si es bag
    def norm(t: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", t.lower())
    if d.unit_type == "pairbag":
        # p.ej: pairbag__category_programming+free_debugging
        return "pairbag__" + "+".join(norm(t.replace(":", "_")) for t in d.tags)
    if d.unit_type == "tagbag":
        # p.ej: tagbag__free_debugging
        return "tagbag__" + norm(d.tags[0].replace(":", "_")) if d.tags else f"tagbag__{d.digest_id[-6:]}"
    if d.unit_type == "session":
        return f"session__{d.start_ts[:10]}_{d.digest_id[-6:]}"
    if d.unit_type == "cohort":
        return f"cohort__{d.start_ts[:10]}_{d.digest_id[-6:]}"
    # fallback
    return d.digest_id






# ——— Small helper for duck-typing access —————————————————————————

def _get(ev: Any, key: str, default: Any = None) -> Any:
    """
    Try to retrieve `key` as an attribute, then as a dict-like key,
    then from its `extras` mapping if present.
    """
    # 1) attribute
    if hasattr(ev, key):
        try:
            v = getattr(ev, key)
            if v is not None:
                return v
        except Exception:
            pass

    # 2) mapping interface
    if isinstance(ev, Mapping) and key in ev:
        return ev[key]

    # 3) nested extras
    ex = getattr(ev, "extras", None) or (ev.get("extras") if isinstance(ev, Mapping) else None)
    if isinstance(ex, Mapping) and key in ex:
        return ex[key]

    return default

# from typing import Literal

# Quantiles = Dict[str, Dict[float, float]]
# Gates     = Dict[str, float]

# @dataclass(frozen=True)
# class GatePolicy:
#     # floors (absolute minima)
#     co_default_floor: int = 40
#     co_backbone_floor: int = 120
#     npmi_keep_floor:  float = 0.18
#     lift_keep_floor:  float = 3.0
#     # rounding behavior for count-based gates
#     round_counts: Literal["none","nearest","up"] = "nearest"
#     # which quantiles to use for each knob
#     q_for_co_default:  float = 0.50   # median co_docs
#     q_for_co_backbone: float = 0.75   # upper quartile co_docs
#     q_for_npmi_keep:   float = 0.50   # median nPMI
#     q_for_npmi_strong: float = 0.75   # strong nPMI
#     q_for_npmi_vstrong:float = 0.90   # very strong nPMI
#     q_for_lift_keep:   float = 0.50
#     q_for_lift_strong: float = 0.75
#     q_for_lift_vstrong:float = 0.90
#     # niche band strategy
#     niche_lo: int | None = 20   # if None → derive from upstream min_docs or a fraction
#     niche_hi: int | None = 60   # if None → derive from co_default/backbone
#     niche_hi_strategy: Literal["above_q75","below_default","fixed"] = "above_q75"
#     # bridge policy
#     min_bridge_floor: float = 0.26       # hard floor used in max(floor, NPMI_STRONG)
#     # optional softeners for sparse corpora
#     min_co_default_allowed: int = 20     # clamp CO_DEFAULT down to this if quantiles tiny
#     min_co_backbone_allowed: int = 40    # clamp CO_BACKBONE down to this if quantiles tiny


# def _round_count(x: float, how: str) -> int:
#     if how == "none":
#         return int(x)
#     if how == "up":
#         return int(math.ceil(x))
#     # "nearest"
#     return int(round(x))
    
# __all__ = ["Event", "Session", "Unit", "EventIndex", "SessionIndex", "GatePolicy", "_get"]
