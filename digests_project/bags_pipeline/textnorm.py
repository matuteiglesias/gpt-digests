# digests_project/bags_pipeline/textnorm.py
from __future__ import annotations


# tag normalization:
# from digests_project.bags_pipeline.normalize import parse_tags, canonical_tag
# raw = {"tags":["Guide","topic:Automation"],"stage":"Execute"}
# sorted({canonical_tag(t) for t in parse_tags(raw)})
# # ['free:guide', 'stage:execute', 'topic:automation']   (if you also project 'stage' elsewhere)

# time parsing:
# from digests_project.bags_pipeline.textnorm import to_utc_dt
# [str(to_utc_dt(x)) for x in ["2025-08-29T06:34:06Z", "2025-08-29T06:34:06+00:00", 1756559246]]




from datetime import datetime, timezone
import re, unicodedata
from typing import Any

__all__ = [
    "slug_value", "slugify",
    "to_utc_dt", "parse_utc_any",
    "coerce_text",
]

# ------------------------------- slugs ---------------------------------------

def slug_value(s: Any) -> str:
    """
    ASCII-only, lowercase, underscore-separated slug suitable for tags/namespaces.
    Example: 'Format-Type/Guide' -> 'format_type_guide'
    """
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    for ch in (" ", "-", "/"):
        s = s.replace(ch, "_")
    return s or "unknown"

def slugify(title: str, maxlen: int = 80) -> str:
    """URL/file friendly hyphen slug (kept separate from tag-oriented slug_value)."""
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return (s[:maxlen] or "untitled")





from dataclasses import is_dataclass, replace as _dc_replace
from types import SimpleNamespace
from typing import Optional, Iterable, Dict, Any, Tuple

from .config import parse_utc_any  # you already have this

def in_window_range(start: str|None, end: str|None,
                    since: str|None, until: str|None) -> bool:
    """Half-open overlap test: [start,end) vs [since,until). Empty/None is open."""
    s = parse_utc_any(since) if since else None
    e = parse_utc_any(until) if until else None
    a = parse_utc_any(start) if start else None
    b = parse_utc_any(end)   if end   else None
    if a is None or b is None:
        return False
    if b < a:
        a, b = b, a
    if s and b <= s:
        return False
    if e and a >= e:
        return False
    return True

def _copy_unit(u, **updates):
    """Dataclass-safe shallow copy with field overrides."""
    if is_dataclass(u):
        return _dc_replace(u, **updates)
    d = dict(getattr(u, "__dict__", {}))
    d.update(updates)
    return SimpleNamespace(**d)





# --------------------------- text-based inference ---------------------------

_HASHTAG_RE = re.compile(r"(?:^|\s)#([a-z0-9_]+)", re.IGNORECASE)
_TEXT_FIELDS = ("text", "content", "summary", "message", "body", "title")

def _get_attr_or_key(obj: Any, key: str) -> Any:
    """Minimal safe getter for dicts or objects. No external dependencies."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)

def _canon_free_token(s: str) -> str:
    # lower, trim, replace spaces with underscores for hashtag-like tokens
    s = (s or "").strip().replace(" ", "_").lower()
    return s

def infer_tags_from_text_like(ev: Any) -> List[str]:
    """
    Extract free tokens from common text fields using #hashtags.
    Returns raw strings (no ns), eg ["automation", "python"].
    """
    tags: List[str] = []
    for f in _TEXT_FIELDS:
        v = _get_attr_or_key(ev, f)
        if isinstance(v, str) and v:
            tags.extend(_HASHTAG_RE.findall(v))
    return [_canon_free_token(t) for t in tags]





# ────────────────────────────────────────────────────────────────────────────────


# # helpers_slicing.py (or top of hydrate.py / kbctl.py near other utilities)
# def _trim_sources_for_window(u, ev_idx, ss_idx, since_iso, until_iso):
#     """Return a view of `u` with sources trimmed to [since,until)."""
#     from .textnorm import parse_utc_any
#     since_dt = parse_utc_any(since_iso) if since_iso else None
#     until_dt = parse_utc_any(until_iso) if until_iso else None

#     if not (since_dt or until_dt):
#         return u  # nothing to trim

#     keep = []
#     seen = set()

#     def _in_window_ts(ts_raw):
#         from .textnorm import parse_utc_any
#         dt = parse_utc_any(ts_raw)
#         if dt is None:
#             return False
#         if since_dt and dt < since_dt:
#             return False
#         if until_dt and dt >= until_dt:
#             return False
#         return True

#     def _session_overlaps(ss):
#         # try a few common fields
#         s_ts = ss.get("ts_start") or ss.get("start_ts") or ss.get("ts") or (ss.get("summary", {}) or {}).get("ts")
#         e_ts = ss.get("ts_end")   or ss.get("end_ts")   or s_ts
#         from .textnorm import parse_utc_any
#         sdt = parse_utc_any(s_ts)
#         edt = parse_utc_any(e_ts) or sdt
#         if sdt is None or edt is None:
#             return False
#         # [sdt, edt) overlaps [since_dt, until_dt)
#         if since_dt and edt <= since_dt:
#             return False
#         if until_dt and sdt >= until_dt:
#             return False
#         return True

#     for kind, sid in getattr(u, "sources", ()) or ():
#         k = (kind, sid)
#         if k in seen:
#             continue
#         seen.add(k)

#         if kind == "event" and ev_idx:
#             rec = ev_idx.get(sid)
#             if not rec:
#                 continue
#             ts = rec.get("ts_abs") or rec.get("ts") or rec.get("timestamp") or rec.get("created_at")
#             if _in_window_ts(ts):
#                 keep.append((kind, sid))

#         elif kind == "session" and ss_idx:
#             key = sid if sid in ss_idx else ("cluster_" + sid[2:] if isinstance(sid, str) and sid.startswith("s_") else sid)
#             ss = ss_idx.get(key)
#             if ss and _session_overlaps(ss):
#                 keep.append((kind, sid))

#         # else: unresolved; skip for trimming

#     # Recompute window to the requested slice bounds (so we don't re-expand)
#     new_start = since_iso or getattr(u, "start_ts", "")
#     new_end   = until_iso or getattr(u, "end_ts", "")
#     return _copy_unit(u, start_ts=new_start, end_ts=new_end, sources=tuple(keep))
