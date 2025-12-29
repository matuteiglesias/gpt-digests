"""
Stable facade for the bags_pipeline package.

Import only from here in apps/CLI:
    from digests_project.bags_pipeline import (...)

This module re-exports the stable surface and provides thin wrappers where
we want back-compat (e.g., ignoring a legacy `tz=` argument).
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Dict, Tuple

# ── Config & types ─────────────────────────────────────────────────────────────
from .config import TZ_LOCAL, DIGESTS_DIR
from .ingest_logs import Event, load_events_from_logs
from .ingest_sessions import Session, load_sessions
from .unitize import Unit, units_from_sessions, cohort_units_from_logs, events_to_units_grouped, sessions_to_units_window
from .pairs import tagbag_units_from_units, pairbag_units_from_units
from .index import build_indices


# ── Quick adapters (no heavy deps) ────────────────────────────────────────────

# ── Selection / L2 orchestration ──────────────────────────────────────────────
from .tag_select import select_units
from .l2 import L2Digest, build_l2_digests as build_L2
from .l2 import write_l2

# ── Hydration / rendering (assembly, not analytics) ───────────────────────────
from .hydrate import (
    render_units_md,
    materialize_bag_markdown,
    # _trim_sources_for_window,
)

# ── Analytics primitives (no Unit coupling) ───────────────────────────────────
from .pairs import (
    co_tag_pairs,
    subsets,
    tag_communities_from_pairs,
)

# ── EDA “bridge” helpers (Unit-aware views for analytics) ─────────────────────
from .eda_bridge import long_from_units, pairs_from_units



# ── Public surface ─────────────────────────────────────────────────────────────
__all__ = [
    # config & types
    "TZ_LOCAL", "DIGESTS_DIR", "Event", "Session", "Unit",

    # ingestion
    "load_events_from_logs", "load_sessions",

    # unitization / adapters
    "units_from_sessions", "cohort_units_from_logs",
    "tagbag_units_from_units", "pairbag_units_from_units",

    # hydration/rendering
    "build_event_index", "build_session_index", "build_l2_digests",
    # "build_indices", "render_units_md", "materialize_bag_markdown",
    "render_units_md", "materialize_bag_markdown",

    # selection / L2
    "select_units", "L2Digest", "build_L2", "write_l2",

    # quick adapters
    "events_to_units_grouped", "sessions_to_units_window",

    # analytics (pure)
    "co_tag_pairs", "quantile_gates", "subsets", "tag_communities_from_pairs",

    # EDA bridges (unit-aware helpers)
    "long_from_units", "pairs_from_units",
]


# re-exports you already have above…
from .unitize import Unit  # and any other types you expose

# tiny ISO helper (no external deps)


# in digests_project/bags_pipeline/__init__.py

# def write_l2(units: Iterable["Unit"], out_base: str, *, layout: str = "onefile", filename_scheme: str = "slug_hash"):
#     from .l2 import write_l2 as _impl
#     return _impl(units, out_base=out_base, layout=layout, filename_scheme=filename_scheme)

# analytics re-exports
from .pairs import (
    co_tag_pairs, subsets, tag_communities_from_pairs
)

# EDA bridge adaptors (unit-aware)
from .eda_bridge import long_from_units, pairs_from_units


# -----------------------
# Ingestion (lazy wrappers)
# -----------------------

# def load_events_from_logs(paths: Iterable[str | "bytes" | "os.PathLike[str]"]) -> Iterable[Event]:
#     from .ingest_logs import load_events_from_logs as _impl
#     return _impl(paths)

# # def load_sessions(paths: Iterable[str | "bytes" | "os.PathLike[str]"]) -> Iterable[Session]:
# #     from .ingest_sessions import load_sessions as _impl
# #     return _impl(paths)

# # -----------------------
# # Unitization (lazy wrappers)
# # -----------------------

# def units_from_sessions(sessions: Iterable[Session]) -> Iterable[Unit]:
#     from .unitize import units_from_sessions as _impl
#     return _impl(sessions)

# # def cohort_units_from_logs(paths: Iterable[str | "bytes" | "os.PathLike[str]"], *, group_by: str = "day", combo_size: int = 2) -> Iterable[Unit]:
# #     from .unitize import cohort_units_from_logs as _impl
# #     return _impl(paths, group_by=group_by, combo_size=combo_size)

# def tagbag_units_from_units(units: Iterable[Unit], *, top_k_tags: int = 50, min_docs: int = 20) -> Iterable[Unit]:
#     from .unitize import tagbag_units_from_units as _impl
#     return _impl(units, top_k_tags=top_k_tags, min_docs=min_docs)

# # def pairbag_units_from_units(units: Iterable[Unit], *, min_support: int = 10, top_k: int = 50) -> Iterable[Unit]:
# #     from .unitize import pairbag_units_from_units as _impl
# #     return _impl(units, min_support=min_support, top_k=top_k)

# # ---------------------------------
# # Hydration / Rendering (lazy wrappers)
# # ---------------------------------

# # def build_event_index(patterns: Iterable[str | "os.PathLike[str]"], *, include_alias: bool = True) -> Dict[str, Dict[str, Any]]:
# #     from .index import build_event_index as _impl
# #     return _impl(patterns, include_alias=include_alias)

# # def build_session_index(patterns: Iterable[str | "os.PathLike[str]"]) -> Dict[str, Dict[str, Any]]:
# #     from .index import build_session_index as _impl
# #     return _impl(patterns)

# # def build_indices(*, logs_glob: Iterable[str | "os.PathLike[str]"] | None = None,
# #                   sessions_glob: Iterable[str | "os.PathLike[str]"] | None = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
# #     from .hydrate import build_indices as _impl
# #     return _impl(logs_glob=logs_glob, sessions_glob=sessions_glob)

# def render_units_md(units: Iterable[Unit], *, ev_idx=None, ss_idx=None, tz: str = "UTC") -> str:
#     from .hydrate import render_units_md as _impl
#     return _impl(units, ev_idx=ev_idx, ss_idx=ss_idx, tz=tz)

# # def materialize_bag_markdown(units: Iterable[Unit], event_idx: Dict[str, Dict[str, Any]] | None, session_idx: Dict[str, Dict[str, Any]] | None,
# #                              *, collapse: bool = True, max_items: int = 200,
# #                              since_iso: Optional[str] = None, until_iso: Optional[str] = None) -> str:
# #     from .hydrate import materialize_bag_markdown as _impl
# #     return _impl(units, event_idx, session_idx, collapse=collapse, max_items=max_items,
# #                  since_iso=since_iso, until_iso=until_iso)

# # -------------
# # Selection
# # -------------

# # def select_units(units: Iterable[Unit], *, tz: str = "UTC", per_tag_cap: int = 10) -> Iterable[Unit]:
# #     from .select import select_units as _impl
# #     return _impl(units, tz=tz, per_tag_cap=per_tag_cap)

# # ----------------
# # Quick adapters
# # ----------------

# # def events_to_units_grouped(events: Iterable[Event], *, group_by: str = "day", combo_size: int = 2) -> Iterable[Unit]:
# #     from .quick import events_to_units_grouped as _impl
# #     return _impl(events, group_by=group_by, combo_size=combo_size)

# def sessions_to_units_window(sessions: Iterable[Session], *, tz: str = "UTC") -> Iterable[Unit]:
#     from .quick import sessions_to_units_window as _impl
#     return _impl(sessions, tz=tz)

# # -----------
# # Analytics
# # -----------

# # def co_tag_pairs(long_df, top_k: int = 300, min_docs: int = 5, min_npmi: float = 0.05):
# #     from .pairs import co_tag_pairs as _impl
# #     return _impl(long_df, top_k=top_k, min_docs=min_docs, min_npmi=min_npmi)

# def quantile_gates(df):
#     from .pairs import quantile_gates as _impl
#     return _impl(df)

# def subsets(pairs_df, comm=None, stats=None, k_top: int = 12, gates: dict | None = None, bridges_npmi_floor: float | None = None):
#     from .pairs import subsets as _impl
#     return _impl(pairs_df, comm=comm, stats=stats, k_top=k_top, gates=gates, bridges_npmi_floor=bridges_npmi_floor)

# def tag_communities_from_pairs(pairs_df, min_npmi: float = 0.05, top_k: int = 12):
#     from .pairs import tag_communities_from_pairs as _impl
#     return _impl(pairs_df, min_npmi=min_npmi, top_k=top_k)