# digests_project/bags_pipeline/select.py
from __future__ import annotations
from typing import Iterable, List, Dict, Tuple, Optional, Set
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .textnorm import parse_utc_any
from .unitize import Unit
from zoneinfo import ZoneInfo

# from typing import Optional, Iterable, Tuple, List, Dict, Any
# ^ keep your existing imports; add re, datetime, timezone, Optional if missing

# from dataclasses import dataclass
# from datetime import datetime
# from dataclasses import asdict
# from datetime import datetime, date
# from math import log
# from .config import TZ_LOCAL, _to_dt_utc

from dataclasses import replace


# -------------------------------------------------------------------
# LEGACY helpers (kept for backward compatibility; thin shims)
# -------------------------------------------------------------------
def _get_field(obj: Any, names: Iterable[str]) -> Any:
    """Read first available attribute or dict key."""
    for name in names:
        # try attribute
        if hasattr(obj, name):
            try:
                v = getattr(obj, name)
                if v is not None:
                    return v
            except Exception:
                pass
        # try dict
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
    return None


def _in_window(u: Unit, since: Optional[str], until: Optional[str]) -> bool:
    """
    DEPRECATED: use UnitSelector._in_window.
    Return True if unit’s [start,end) overlaps [since,until).
    """
    # field lookup
    us_raw = _get_field(u, ("start_ts", "start", "ts_start", "begin_ts"))
    ue_raw = _get_field(u, ("end_ts",   "end",   "ts_end",   "finish_ts"))
    if us_raw is None or ue_raw is None:
        return False

    us = parse_utc_any(us_raw)
    ue = parse_utc_any(ue_raw)
    if us is None or ue is None:
        return False
    if us > ue:
        us, ue = ue, us

    s = parse_utc_any(since) if since else None
    e = parse_utc_any(until) if until else None

    if s is None and e is None:
        return True
    if s is not None and ue <= s:
        return False
    if e is not None and us >= e:
        return False
    return True


def _tags_ok(u: Unit, require_all: Tuple[str, ...], require_any: Tuple[str, ...]) -> bool:
    """
    DEPRECATED: use UnitSelector._tags_ok.
    Return True if u.tags satisfies AND/OR constraints.
    """
    S = set(u.tags or ())
    if require_all and not set(require_all).issubset(S):
        return False
    if require_any and not (set(require_any) & S):
        return False
    return True


def select_units(
    units: Iterable[Unit],
    types: Tuple[str, ...]     = (),
    tags_all: Tuple[str, ...]  = (),
    tags_any: Tuple[str, ...]  = (),
    since: Optional[str]       = None,
    until: Optional[str]       = None,
) -> List[Unit]:
    """
    LEGACY entrypoint: Filter Units by type, tags, and time window.
    Internally delegates to UnitSelector for consistency.
    """
    selector = UnitSelector(
        types=types,
        tags_all=tags_all,
        tags_any=tags_any,
        since=since, until=until
    )
    return selector.select(units)


# -------------------------------------------------------------------
# NEW: Encapsulated selector with clear concerns
# -------------------------------------------------------------------
class UnitSelector:
    """
    Encapsulates all filtering concerns for `Unit` objects:
      • types       : Allowed unit.unit_type values
      • tags_all    : Must include *all* of these tags
      • tags_any    : Must include *at least one* of these
      • since,until : ISO strings defining [since,until) time window
    """

    def __init__(
        self,
        types: Iterable[str]       = (),
        tags_all: Iterable[str]    = (),
        tags_any: Iterable[str]    = (),
        since: Optional[str]       = None,
        until: Optional[str]       = None,
    ):
        self.types    = set(types)
        self.tags_all = set(tags_all)
        self.tags_any = set(tags_any)
        self.since_dt = parse_utc_any(since) if since else None
        self.until_dt = parse_utc_any(until) if until else None

    def _in_window(self, u: Unit) -> bool:
        # reuse the same field‐lookup logic
        us_raw = _get_field(u, ("start_ts", "start", "ts_start", "begin_ts"))
        ue_raw = _get_field(u, ("end_ts",   "end",   "ts_end",   "finish_ts"))
        if us_raw is None or ue_raw is None:
            return False

        us = parse_utc_any(us_raw)
        ue = parse_utc_any(ue_raw)
        if us is None or ue is None:
            return False
        if us > ue:
            us, ue = ue, us

        # half‐open interval [since, until)
        if self.since_dt and ue <= self.since_dt:
            return False
        if self.until_dt and us >= self.until_dt:
            return False
        return True

    def _tags_ok(self, u: Unit) -> bool:
        S = set(u.tags or ())
        if self.tags_all and not self.tags_all.issubset(S):
            return False
        if self.tags_any and not (self.tags_any & S):
            return False
        return True

    def select(self, units: Iterable[Unit]) -> List[Unit]:
        """
        Perform sequential filtering:
          1) by type
          2) by time window (_in_window)
          3) by tag logic (_tags_ok)
        Optionally *trims* each Unit’s start/end to the window bounds.
        """
        out: List[Unit] = []
        for u in units:
            # type filter
            if self.types and u.unit_type not in self.types:
                continue
            # time overlap
            if not self._in_window(u):
                continue
            # tag constraints
            if not self._tags_ok(u):
                continue

            # optional timestamp truncation
            if self.since_dt or self.until_dt:
                st = parse_utc_any(u.start_ts)
                en = parse_utc_any(u.end_ts)
                new_st = max(st, self.since_dt) if self.since_dt else st
                new_en = min(en, self.until_dt) if self.until_dt else en
                if new_st <= new_en:
                    u = replace(
                        u,
                        start_ts=new_st.isoformat().replace("+00:00", "Z"),
                        end_ts  =new_en.isoformat().replace("+00:00", "Z"),
                    )
                else:
                    continue

            out.append(u)
        return out




# def _get_field(obj: Any, names: Iterable[str]) -> Any:
#     """Lee el primer campo disponible, sea atributo o dict."""
#     for name in names:
#         if hasattr(obj, name):
#             try:
#                 v = getattr(obj, name)
#                 if v is not None:
#                     return v
#             except Exception:
#                 pass
#         if isinstance(obj, dict) and name in obj and obj[name] is not None:
#             return obj[name]
#     return None


# def _in_window(u: "Unit", since: Optional[str], until: Optional[str]) -> bool:
#     """
#     True si la ventana del unit [start,end) SOLAPA la consulta [since,until).
#     Acepta nombres legacy: start_ts/end_ts o start/end. None = abierto.
#     Intervalos medio abiertos (incluye start, excluye end).
#     """
#     # 1) Extraer campos con tolerancia a nombres antiguos
#     us_raw = _get_field(u, ("start_ts", "start", "ts_start", "begin_ts"))
#     ue_raw = _get_field(u, ("end_ts",   "end",   "ts_end",   "finish_ts"))

#     if us_raw is None or ue_raw is None:
#         return False

#     # 2) Parsear todo a datetimes "aware" en UTC
#     us = parse_utc_any(us_raw)
#     ue = parse_utc_any(ue_raw)

#     if us is None or ue is None:
#         return False

#     # 3) Corregir si el rango viene invertido por algún bug
#     if us > ue:
#         us, ue = ue, us

#     # 4) Ventana de consulta
#     s = parse_utc_any(since) if since else None
#     e = parse_utc_any(until) if until else None

#     # 5) Sin filtro → siempre incluye
#     if s is None and e is None:
#         return True

#     # 6) Chequeos de no-solapamiento (medio abiertos)
#     if s is not None and ue <= s:
#         return False
#     if e is not None and us >= e:
#         return False

#     return True


# def _tags_ok(u: Unit, require_all: Tuple[str,...], require_any: Tuple[str,...]) -> bool:
#     S = set(u.tags)
#     if require_all and not set(require_all).issubset(S):
#         return False
#     if require_any and not (set(require_any) & S):
#         return False
#     return True


# def select_units(units: Iterable[Unit],
#                  types: Tuple[str,...] = (),
#                  tags_all: Tuple[str,...] = (),
#                  tags_any: Tuple[str,...] = (),
#                  since: Optional[str] = None,  # ISO UTC "YYYY-MM-DDTHH:MM:SSZ"
#                  until: Optional[str] = None) -> List[Unit]:
    
#     """Filter Units by type, tag logic, and optional time window.

#     Args:
#         units: Candidate units.
#         types: Allowed unit types (empty tuple means "any").
#         tags_all: All these tags must be present (AND).
#         tags_any: At least one of these tags must be present (OR).
#         since: ISO lower bound (inclusive) on start_ts.
#         until: ISO upper bound (exclusive) on end_ts.

#     Returns:
#         Filtered Units.
#     """

#     out = []
#     T = set(types or ())
#     for u in units:
#         if T and u.unit_type not in T:
#             continue
#         if not _in_window(u, since, until):
#             continue
#         if not _tags_ok(u, tags_all, tags_any):
#             continue
#         out.append(u)
#     return out


# class UnitSelector:
#     """
#     Encapsulates all the criteria for filtering Units:
#       - unit types
#       - tags_all (AND)
#       - tags_any (OR)
#       - time window [since, until)
#     """
#     def __init__(
#         self,
#         types: Iterable[str] = (),
#         tags_all: Iterable[str] = (),
#         tags_any: Iterable[str] = (),
#         since: Optional[str] = None,
#         until: Optional[str] = None,
#     ):
#         self.types     = set(types)
#         self.tags_all  = set(tags_all)
#         self.tags_any  = set(tags_any)
#         self.since_dt  = parse_utc_any(since) if since else None
#         self.until_dt  = parse_utc_any(until) if until else None

#     def _in_window(self, u: Unit) -> bool:
#         # Extract start/end (legacy-friendly)
#         us_raw = getattr(u, "start_ts", None) or getattr(u, "start", None)
#         ue_raw = getattr(u, "end_ts",   None) or getattr(u, "end",   None)
#         if not us_raw or not ue_raw:
#             return False
#         us = parse_utc_any(us_raw); ue = parse_utc_any(ue_raw)
#         if not us or not ue:
#             return False
#         if us > ue:
#             us, ue = ue, us
#         # Half-open interval [since, until)
#         if self.since_dt and ue <= self.since_dt:
#             return False
#         if self.until_dt and us >= self.until_dt:
#             return False
#         return True

#     def _tags_ok(self, u: Unit) -> bool:
#         S = set(u.tags or ())
#         if self.tags_all and not self.tags_all.issubset(S):
#             return False
#         if self.tags_any and not (self.tags_any & S):
#             return False
#         return True

#     def select(self, units: Iterable[Unit]) -> List[Unit]:
#         out: List[Unit] = []
#         for u in units:
#             if self.types and u.unit_type not in self.types:
#                 continue
#             if not self._in_window(u):
#                 continue
#             if not self._tags_ok(u):
#                 continue
#             # Optionally: trim the unit’s start/end to the window bounds:
#             if self.since_dt or self.until_dt:
#                 st = parse_utc_any(u.start_ts); en = parse_utc_any(u.end_ts)
#                 new_st = max(st, self.since_dt) if self.since_dt else st
#                 new_en = min(en, self.until_dt) if self.until_dt else en
#                 if new_st <= new_en:
#                     u = replace(
#                         u,
#                         start_ts=new_st.isoformat().replace("+00:00","Z"),
#                         end_ts  =new_en.isoformat().replace("+00:00","Z"),
#                     )
#                 else:
#                     continue
#             out.append(u)
#         return out




# Agrupadores simples: por día local, por topic_id, por par de tags
def group_by_day(units: Iterable[Unit], tz: str = "America/Argentina/Buenos_Aires") -> Dict[str, List[Unit]]:
    from collections import defaultdict
    out = defaultdict(list)
    for u in units:
        dt = parse_utc_any(u.start_ts).astimezone(ZoneInfo(tz))
        out[dt.date().isoformat()].append(u)
    return dict(out)

def group_by_topic(units: Iterable[Unit]) -> Dict[str, List[Unit]]:
    from collections import defaultdict
    out = defaultdict(list)
    for u in units:
        for t in (u.topic_ids or ()):
            out[t].append(u)
    return dict(out)

def tag_pairs_from_unit(u: Unit) -> List[Tuple[str,str]]:
    tags = sorted(set(u.tags))
    pairs = []
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            pairs.append((tags[i], tags[j]))
    return pairs

def group_by_tagpair(units: Iterable[Unit]) -> Dict[Tuple[str,str], List[Unit]]:
    from collections import defaultdict
    out = defaultdict(list)
    for u in units:
        for a,b in tag_pairs_from_unit(u):
            out[(a,b)].append(u)
    return dict(out)



# We assume Unit is a dataclass with attributes:
# unit_id: str
# unit_type: str
# tags: Tuple[str, ...]
# topic_ids: Tuple[str, ...]
# ts_range: Dict[str, str] with keys start, end, tz
# unit_ts: Optional[str]
# content: Dict
try:
    from digests_project.bags_pipeline.unitize import Unit
except Exception:
    Unit = object  # type: ignore


# def _parse_iso(s: str) -> datetime:
#     # Accepts "...Z" or "+00:00" and naive. Always returns aware in UTC first.
#     if not s:
#         raise ValueError("empty datetime string")
#     if s.endswith("Z"):
#         s = s[:-1] + "+00:00"
#     dt = datetime.fromisoformat(s)
#     if dt.tzinfo is None:
#         # treat naive as UTC, then caller can convert to local
#         dt = dt.replace(tzinfo=ZoneInfo("UTC"))
#     return dt


# def _unit_start_ts(u):
#     # primary: start_ts
#     ts = getattr(u, "start_ts", None)
#     if ts:
#         return ts
#     # secondary: ts_range.start (if present)
#     # tr = getattr(u, "ts_range", None)
#     # if isinstance(tr, dict):
#     #     ts = tr.get("start") or tr.get("end")
#     #     if ts:
#     #         return ts
#     # tertiary: old field if present (legacy compatibility only)
#     return getattr(u, "unit_ts", None)


# def _unit_local_day(u: Unit, local_tz: str) -> date:
#     z = ZoneInfo(local_tz)
#     # Prefer unit_ts if present, else ts_range.start
#     # ts = getattr(u, "unit_ts", None) or u.ts_range.get("start")
#     ts = _unit_start_ts(u)
#     dt = _parse_iso(ts).astimezone(z)
#     return dt.date()


# def _idf_weights(units: List[Unit]) -> Dict[str, float]:
#     # df counts per tag on this day's pool
#     N = max(1, len(units))
#     df: Dict[str, int] = {}
#     for u in units:
#         for t in set(u.tags or ()):  # set to avoid multiply counting same tag in one unit
#             df[t] = df.get(t, 0) + 1
#     # Smooth IDF
#     return {t: log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}



# def _topics_for(u: Unit) -> Tuple[str, ...]:
#     return tuple(u.topic_ids or ())


# def select_l3_daily(
#     units: List[Unit],
#     day: str | date,
#     cfg: Optional[Dict] = None,
#     tz: str = TZ_LOCAL,
# ) -> List[Unit]:
#     """
#     Select a daily L3 set:
#       - filter Units to local calendar 'day'
#       - score with IDF over tags
#       - enforce per_topic_cap and total_cap
#       - return sorted by -salience
#     cfg schema (defaults shown):
#     {
#       "per_topic_cap": 3,
#       "total_cap": 12,
#       "include_bridges": True,
#       "sort": "-salience"
#     }
#     """
#     cfg = cfg or {}
#     per_topic_cap: int = int(cfg.get("per_topic_cap", 3))
#     total_cap: int = int(cfg.get("total_cap", 12))
#     include_bridges: bool = bool(cfg.get("include_bridges", True))
#     # sort currently supports "-salience" or "salience"
#     sort_key: str = str(cfg.get("sort", "-salience"))

#     if isinstance(day, str):
#         # Expect YYYY-MM-DD or full ISO; normalize to local day
#         # Parse with ISO. If only date provided, interpret as local date.
#         if "T" in day:
#             d_loc = _parse_iso(day).astimezone(ZoneInfo(tz)).date()
#         else:
#             d_loc = datetime.fromisoformat(day).date()
#     else:
#         d_loc = day

#     # 1) filter by local day
#     pool: List[Unit] = []
#     for u in units:
#         try:
#             if _unit_local_day(u, tz) == d_loc:
#                 pool.append(u)
#         except Exception:
#             # if parsing fails, drop from this view
#             continue
#     if not pool:
#         return []

#     # 2) compute IDF on pool and score
#     idf = _idf_weights(pool)
#     scores: Dict[str, float] = {}
#     for u in pool:
#         scores[u.unit_id] = _score_unit(u, idf, include_bridges)

#     # 3) greedy selection with per-topic and total caps
#     selected: List[Unit] = []
#     per_topic_counts: Dict[str, int] = {}
#     # order by salience descending by default
#     ordered = sorted(pool, key=lambda u: scores[u.unit_id], reverse=True)

#     def can_take(u: Unit) -> bool:
#         if len(selected) >= total_cap:
#             return False
#         topics = _topics_for(u) or ("__none__",)
#         for t in topics:
#             if per_topic_counts.get(t, 0) >= per_topic_cap:
#                 # if any topic cap is hit, reject
#                 return False
#         return True

#     for u in ordered:
#         if not can_take(u):
#             continue
#         selected.append(u)
#         # annotate salience for downstream use
#         try:
#             setattr(u, "scores", {"salience": float(scores[u.unit_id])})
#         except Exception:
#             pass
#         for t in (_topics_for(u) or ("__none__",)):
#             per_topic_counts[t] = per_topic_counts.get(t, 0) + 1
#         if len(selected) >= total_cap:
#             break

#     # optional ascending sort if requested
#     if sort_key == "salience":
#         selected = sorted(selected, key=lambda u: getattr(u, "scores", {}).get("salience", 0.0))
#     else:
#         selected = sorted(selected, key=lambda u: getattr(u, "scores", {}).get("salience", 0.0), reverse=True)

#     return selected
