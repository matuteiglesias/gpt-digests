# bags_pipeline/quick.py
from __future__ import annotations
from typing import Iterable, List, Dict, Any, Tuple, Optional
from zoneinfo import ZoneInfo
from .unitize import Unit  # same dataclass you already use
from .config import _get, parse_utc_any  # same dataclass you already use
from .core import _get
from .textnorm import infer_tags_from_text_like
from .normalize import canonical_tag


# # from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def extract_event_tags(ev: Any) -> Tuple[str, ...]:
    """
    Return canonical tag tuple for an event, combining:
      - ev.tags (tuple/list)
      - extras['tags'] (list)
      - extras['topics'] (list)   (treated as tags)
      - hashtags in text fields
    """
    out: set[str] = set()

    # ev.tags (dataclass attr) or mapping
    tags_attr = _get(ev, "tags", [])
    if isinstance(tags_attr, (list, tuple)):
        for t in tags_attr:
            if isinstance(t, str) and t.strip():
                out.add(canonical_tag(t))

    # extras.tags
    tags_ex = _get(ev, "tags", [])
    # If tags_attr was empty and we looked in mapping first, tags_ex == tags_attr. That’s fine.

    # extras['topics'] sometimes hold human tags
    topics_ex = _get(ev, "topics", [])
    for col in (tags_ex, topics_ex):
        if isinstance(col, (list, tuple)):
            for t in col:
                if isinstance(t, str) and t.strip():
                    out.add(canonical_tag(t))

    # hashtag inference
    for t in infer_tags_from_text_like(ev):
        out.add(t)

    # done
    return tuple(sorted(out))


def quick_unit_list_markdown(
        units: Iterable["Unit"],
        ev_idx: Optional[Dict[str, Dict[str, Any]]] = None,
        ss_idx: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        tz: str = "UTC",
        max_items: int = 200,
    ) -> str:
    """
    Simple renderer: group by day, list titles/summaries from resolvable sources.
    """
    z = ZoneInfo(tz)
    items: list[tuple[str | None, str, str]] = []

    for u in units:
        for kind, sid in getattr(u, "sources", ()) or ():
            rec = (
                ev_idx.get(sid) if kind == "event" and ev_idx is not None
                else ss_idx.get(sid) if kind == "session" and ss_idx is not None
                else None
            )
            if not rec:
                continue

            ts_raw = rec.get("ts_abs") or rec.get("timestamp") or rec.get("ts") or rec.get("created_at")
            ts = parse_utc_any(ts_raw, z)

            title = rec.get("title") or (rec.get("extras") or {}).get("title") or "…"
            summary = rec.get("summary") or (rec.get("extras") or {}).get("summary") or ""
            items.append((ts, title, summary))

            if len(items) >= max_items:
                break  # leaves inner loop
        if len(items) >= max_items:
            break      # leaves outer loop

    if not items:
        return ""

    by_day: dict[str, list[tuple[str | None, str, str]]] = {}
    for ts, title, summary in items:
        day = (ts or "")[:10] or "unknown"
        by_day.setdefault(day, []).append((ts, title, summary))

    lines: list[str] = []
    for day in sorted(by_day):
        lines.append(f"## {day}")
        for ts, title, summary in by_day[day]:
            ts_s = f" — {ts}" if ts else ""
            lines.append(f"### {title}{ts_s}")
            if summary:
                lines.append(summary.strip())
            lines.append("")
    return "\n".join(lines).strip()





# # —— NEW: filter kwargs to whatever Unit actually accepts ——
# def _filter_for_unit(kwargs: Dict[str, Any]) -> Dict[str, Any]:
#     try:
#         fset = {f.name for f in fields(Unit)}
#         return {k: v for k, v in kwargs.items() if k in fset}
#     except Exception:
#         return kwargs  # best-effort fallback
    
   
def _event_session_key(ev: Any) -> str | None:
    # prefer a stable conversation/thread id
    for k in ("session_id", "conversation_id", "thread_id", "conv_id"):
        v = _get(ev, k)
        if isinstance(v, str) and v.strip():
            return v
    return None

