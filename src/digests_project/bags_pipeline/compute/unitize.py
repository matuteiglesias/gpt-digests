from __future__ import annotations
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict

from .io import read_jsonl
from .core import Event, Unit, Session
from .config import window_key_day, to_utc_dt, sha256_of, TZ_LOCAL, parse_utc_any

import pandas as pd

PathLike = Union[str, bytes, os.PathLike]

from .core import Unit
from .config import to_utc_dt

from .io import stable_id


# # from .normalize import _canon_tag
# def canonical_tag(s: str) -> str:
#     # normalize and keep namespacing if present ("category:software_dev")
#     s = s.strip()
#     s = s.replace(" ", "_")
#     return s.lower()


def _core_cohort_units_from_events(
    events: List[Event],
    *,
    combo_size: int = 2,
    group_by: str = "day",          # "day" | "week" | "month" | "session"
    tz: str = TZ_LOCAL,
    min_events: int = 4,
    top_k_tags: int = 30,
) -> List[Unit]:
    """
    Core logic: build 'cohort' Units from a list of Event objects.
    """
    def _iso_week_key(dt: datetime) -> str:
        day_key = window_key_day(dt.isoformat() + "Z", tz)
        y, m, d = map(int, day_key.split("-"))
        iso = datetime(y, m, d).isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _month_key(dt: datetime) -> str:
        return window_key_day(dt.isoformat() + "Z", tz)[:7]

    def _session_key(ev: Event) -> Optional[str]:
        return (
            ev.session_id
            or getattr(ev, "cluster_id", None)
            or ev.conversation_id
        )

    def _bucket_key(ev: Event) -> str:
        dt = to_utc_dt(ev.ts_abs)
        if group_by == "day":
            return window_key_day(dt.isoformat() + "Z", tz)
        if group_by == "week":
            return _iso_week_key(dt)
        if group_by == "month":
            return _month_key(dt)
        if group_by == "session":
            return _session_key(ev) or "session:unknown"
        return window_key_day(dt.isoformat() + "Z", tz)

    # bucket events
    by_bucket: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_bucket[_bucket_key(ev)].append(ev)

    units: List[Unit] = []
    for bucket, evs in by_bucket.items():
        if len(evs) < min_events:
            continue

        # compute bucket span
        starts = [to_utc_dt(e.ts_abs) for e in evs]
        start_iso = min(starts).isoformat().replace("+00:00","Z")
        end_iso   = max(starts).isoformat().replace("+00:00","Z")

        # count tags
        tag_counts: Counter[str] = Counter()
        for e in evs:
            tag_counts.update(set(e.tags))
        ranked = [t for t,_ in tag_counts.most_common(top_k_tags)]

        if combo_size > 0:
            for combo in combinations(ranked, combo_size):
                combo_set = set(combo)
                matched = [e.event_id for e in evs if combo_set.issubset(e.tags)]
                if len(matched) < min_events:
                    continue
                topics = topics_from_tags(combo)
                payload = {
                    "type":    "cohort",
                    "bucket":  bucket,
                    "group_by": group_by,
                    "combo":   tuple(sorted(combo)),
                    "start":   start_iso,
                    "end":     end_iso,
                    "topic_ids": topics,
                    "sources": [( "event", eid) for eid in sorted(set(matched))],
                }
                uid = "u_" + sha256_of(payload)
                units.append(Unit(
                    unit_id=uid,
                    unit_type="cohort",
                    start_ts=start_iso,
                    end_ts=end_iso,
                    tags=tuple(sorted(combo)),
                    topic_ids=topics,
                    sources=tuple(payload["sources"]),
                ))
        else:
            header = tuple(ranked[:5])
            topics = topics_from_tags(header) or ("unknown",)
            payload = {
                "type":    "cohort",
                "bucket":  bucket,
                "group_by": group_by,
                "tags":    header,
                "start":   start_iso,
                "end":     end_iso,
                "topic_ids": topics,
                "sources": [( "event", e.event_id) for e in evs],
            }
            uid = "u_" + sha256_of(payload)
            units.append(Unit(
                unit_id=uid,
                unit_type="cohort",
                start_ts=start_iso,
                end_ts=end_iso,
                tags=header,
                topic_ids=topics,
                sources=tuple(payload["sources"]),
            ))

    return units



def _read_units_jsonl(p: Path) -> list[Unit]:
    """
    Load a JSONL of Unit dicts from `p` and return a list of Unit objects.
    """
    from bags_pipeline.io import read_jsonl

    return [Unit(**d) for d in read_jsonl(p)]

def _write_units_jsonl(path: Path, units: Iterable[Unit]) -> None:
    """
    Write an iterable of Unit objects to JSONL at `path`.
    """
    from bags_pipeline.io import write_jsonl

    # each row is the Unit's __dict__
    write_jsonl(path,
                (u.__dict__ for u in units),
                ensure_ascii=False)


def cohort_units_from_logs(
    logs_or_events: Iterable[Union[PathLike, dict, Event]],
    *,
    group_by: str = "day",
    combo_size: int = 2,
    since: Optional[str] = None,
    until: Optional[str] = None,
    tz: Optional[str] = None,
    min_events: int = 4,
    top_k_tags: int = 30,
) -> List[Unit]:
    """
    Facade: accept either glob patterns, raw dicts, or Event objects,
    optionally slice by [since, until), and call the core builder.
    """
    # 1) load/normalize
    it = iter(logs_or_events)
    events: List[Event] = []
    try:
        first = next(it)
    except StopIteration:
        pass
    else:
        rest = [first, *it]
        if isinstance(first, (str, bytes, os.PathLike)):
            # treat each as a glob
            for pattern in rest:
                for p in Path(pattern).parent.glob(Path(pattern).name):
                    for raw in read_jsonl(p):
                        events.append(Event(**raw))
        else:
            # assume Event or dict
            for e in rest:
                events.append(e if isinstance(e, Event) else Event(**e))

    # 2) optional UTC slicing
    if since or until:
        since_dt = parse_utc_any(since) if since else None
        until_dt = parse_utc_any(until) if until else None
        def _in_window(ev: Event) -> bool:
            try:
                ts = parse_utc_any(ev.ts_abs)
            except Exception:
                return False
            if since_dt and ts < since_dt:
                return False
            if until_dt and ts >= until_dt:
                return False
            return True
        events = [e for e in events if _in_window(e)]

    # 3) delegate to core
    return _core_cohort_units_from_events(
        events,
        combo_size=combo_size,
        group_by=group_by,
        tz=tz or TZ_LOCAL,
        min_events=min_events,
        top_k_tags=top_k_tags,
    )



def pairbag_units_from_units(units: List[Unit],
                             pairs_df: pd.DataFrame | None = None,
                             top_n: int = 50,
                             min_docs: int = 2) -> List[Unit]:
    """
    Crea Units virtuales 'pairbag' por par de tags importante.
    - Si pairs_df es None, deriva pares rápidos por co-ocurrencia simple.
    - Agrupa todas las units originales que contengan ambos tags.
    """
    # 1) derivar pares si no viene un pairs_df listo
    if pairs_df is None or pairs_df.empty:
        from collections import Counter
        # contar co-ocurrencias por unit
        co = Counter()
        for u in units:
            tags = sorted(set(u.tags))
            for i in range(len(tags)):
                for j in range(i+1, len(tags)):
                    co[(tags[i], tags[j])] += 1
        # armar DF mínimo
        rows = [{"tag_a": a, "tag_b": b, "co_docs": n} for (a,b),n in co.items()]
        pairs_df = pd.DataFrame(rows)
        if pairs_df.empty:
            return []
        pairs_df["npmi"] = 0.0
        pairs_df["lift"] = 0.0
        pairs_df["edge_score"] = pairs_df["co_docs"]

    # 2) filtrar por soporte
    if "co_docs" in pairs_df.columns:
        pairs_df = pairs_df[pairs_df["co_docs"] >= min_docs].copy()
    if pairs_df.empty:
        return []

    # 3) seleccionar top_n por edge_score (o co_docs si no estuviera)
    key = "edge_score" if "edge_score" in pairs_df.columns else "co_docs"
    top = pairs_df.sort_values(key, ascending=False).head(top_n)

    # 4) materializar pairbags
    out: List[Unit] = []
    for _, r in top.iterrows():
        tag_a = r["tag_a"]; tag_b = r["tag_b"]
        matched = [u for u in units if set((tag_a, tag_b)).issubset(set(u.tags))]
        if not matched:
            continue
        start = min(u.start_ts for u in matched)
        end   = max(u.end_ts   for u in matched)
        # agregamos todas las sources de las units participantes
        sources = []
        for u in matched:
            sources.extend(list(u.sources))
        # id determinístico por tags + members
        payload = {
            "type": "pairbag",
            "tags": tuple(sorted((tag_a, tag_b))),
            "members": tuple(sorted(u.unit_id for u in matched)),
            "start": start, "end": end
        }
        unit_id = "u_" + sha256_of(payload)
        out.append(Unit(
            unit_id=unit_id,
            unit_type="pairbag",
            start_ts=start,
            end_ts=end,
            tags=tuple(sorted((tag_a, tag_b))),
            topic_ids=("unknown",),
            sources=tuple(sources)
        ))
    return out


def topics_from_tags(tags: Iterable[str]) -> tuple[str, ...]:
    topics = {t.split("topic:",1)[1] for t in tags if isinstance(t, str) and t.startswith("topic:")}
    return tuple(sorted(topics)) or ("unknown",)


def units_from_sessions(sessions: list["Session"]) -> list["Unit"]:
    out: list["Unit"] = []
    for s in sessions:
        tags = tuple(s.labels) if isinstance(s.labels, (list, tuple)) else tuple(s.labels or ())
        payload = {
            "type": "session",
            "start": s.start_ts,
            "end":   s.end_ts,
            "tags":  tags,
            "topic_ids": topics_from_tags(tags),
            "sources": [("session", s.session_id)],
        }
        unit_id = "u_" + sha256_of(payload)
        out.append(Unit(unit_id, "session", s.start_ts, s.end_ts,
                        tags, payload["topic_ids"], tuple(payload["sources"])))
    return out

from .core import _get
from zoneinfo import ZoneInfo
from .textnorm import infer_tags_from_text_like

import inspect
from dataclasses import is_dataclass, fields as dc_fields


# --- robust: discover Unit's accepted kwargs whether or not it's a dataclass ---
def _unit_field_names() -> set[str]:
    # 1) dataclass path
    try:
        if is_dataclass(Unit):
            return {f.name for f in dc_fields(Unit)}
    except Exception:
        pass
    # 2) signature-based path (class or callable)
    try:
        sig = inspect.signature(Unit)  # works for class __init__ too
        names = {name for name, p in sig.parameters.items()
                 if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        if names:
            return names
    except Exception:
        pass
    # 3) conservative fallback (the schema you actually use downstream)
    return {"unit_id", "unit_type", "tags", "topic_ids", "start_ts", "end_ts", "sources"}






def _filter_for_unit(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    fset = _unit_field_names()
    return {k: v for k, v in kwargs.items() if k in fset}

def sessions_to_units_window(sessions: List[Any], since_iso: str, until_iso: str,
                             require_tag: Optional[str] = None, tz: str = TZ_LOCAL) -> List[Unit]:
    """Select sessions in a time window and return them as Units.

    Args:
        sessions: Parsed session dicts.
        since_iso: Inclusive start in ISO (UTC or local; Z handled).
        until_iso: Exclusive end in ISO.
        require_tag: Canonical tag to filter. If None, no tag filter.
        tz: Target timezone for window evaluation.

    Returns:
        Units of type "session".
    """


    z = ZoneInfo(tz)
    def _to_aware(s: str) -> datetime:
        ss = s[:-1] + "+00:00" if s.endswith("Z") else s

        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(tz))
        return dt.astimezone(z)

    since, until = _to_aware(since_iso), _to_aware(until_iso)
    units: List[Unit] = []

    for s in sessions:
        sess_id = _get(s, "id") or _get(s, "session_id")  # pick your canonical field

        tags: list[str] = []
        tt = _get(s, "tags", [])
        if isinstance(tt, (list, tuple)):
            tags.extend(canonical_tag(t) for t in tt if isinstance(t, str))
        tags.extend(infer_tags_from_text_like(s))
        tags = sorted(set(t for t in tags if t))

        ts = _get(s, "unit_ts") or _get(s, "start") or _get(s, "created_at") or _get(s, "ts")
        title = _get(s, "title") or _get(s, "summary") or _get(s, "name") or ""

        if isinstance(ts, (int, float)):
            val = float(ts) / 1000.0 if float(ts) > 10_000_000_000 else float(ts)
            dt = datetime.fromtimestamp(val, tz=ZoneInfo("UTC")).astimezone(z)
        elif isinstance(ts, str):
            vv = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
            try:
                dt = datetime.fromisoformat(vv)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                dt = dt.astimezone(z)
            except Exception:
                continue
        else:
            continue

        if not (since <= dt < until):
            continue
        if require_tag and canonical_tag(require_tag) not in tags:
            continue

        # title/summary stay out of Unit; they’ll be used at digest/hydration time
        candidate = {
            "unit_id": stable_id({"sid": sess_id or title, "ts": dt.isoformat()}),
            "unit_type": "session",
            "tags": tuple(tags),
            "topic_ids": tuple(),
            "start_ts": dt.isoformat(),
            "end_ts": dt.isoformat(),
            # key change: give the hydrator something to resolve
            "sources": tuple((("session", sess_id),) if sess_id else ()),
        }
        units.append(Unit(**_filter_for_unit(candidate)))
    return units

from .quick import extract_event_tags, _event_session_key
from .config import _event_ts_iso

def events_to_units_grouped(events: List[Any], group_by: str = "session", tz: str = TZ_LOCAL) -> List[Unit]:
    """Group raw events into coarse Units.

    Groups by `session` when available, otherwise falls back to daily buckets.

    Args:
        events: Normalized events (as produced by `load_events_from_logs`).
        group_by: "session" or "day".
        tz: IANA timezone used to compute window labels.

    Returns:
        Units of type "cohort".
    """

    buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tags": set(), "events": []})
    for ev in events:
        tags = extract_event_tags(ev)
        key = None
        if group_by == "session":
            sid = _event_session_key(ev)
            if sid:
                key = f"session:{sid}"
        if not key:
            iso = _event_ts_iso(ev, tz)
            key = f"day:{iso[:10]}"
        buckets[key]["tags"].update(tags)
        buckets[key]["events"].append(ev)

    units: List[Unit] = []
    for key, payload in buckets.items():
        evs = payload["events"]
        if not evs:
            continue
        tss = sorted(_event_ts_iso(e, tz) for e in evs)
        ts_start, ts_end = tss[0], tss[-1]

        candidate = {
            "unit_id": stable_id({"k": key, "tags": sorted(payload["tags"]), "ts": ts_start}),
            "unit_type": "cohort",
            "tags": tuple(sorted(payload["tags"])),
            "topic_ids": tuple(),
            "start_ts": ts_start,
            "end_ts": ts_end,
            # "sources": tuple(),  # keep empty; hydration will pull events by index
            "sources": tuple(("event", getattr(e, "event_id", None)) for e in evs if getattr(e, "event_id", None)),
        }
        units.append(Unit(**_filter_for_unit(candidate)))
    return units





# def conversation_units_from_logs(events: List[Event]) -> List[Unit]:
#     from collections import defaultdict
#     by_conv: dict[str, list[Event]] = defaultdict(list)
#     for e in events:
#         if e.conversation_id:
#             by_conv[e.conversation_id].append(e)
#     units: List[Unit] = []
#     for cid, evs in by_conv.items():
#         start = min(e.ts_abs for e in evs)
#         end   = max(e.ts_abs for e in evs)
#         tags  = tuple(sorted({t for e in evs for t in e.tags}))
#         topics= topics_from_tags(tags)
#         payload = {
#             "type":"conversation",
#             "cid": cid,
#             "start": start,
#             "end": end,
#             "tags": tags,
#             "topic_ids": topics,
#             "sources": [("event", e.event_id) for e in evs]
#         }
#         unit_id = "u_" + sha256_of(payload)
#         units.append(Unit(unit_id, "conversation", start, end, tags, topics, tuple(payload["sources"])))
#     return units



# def find_unit
