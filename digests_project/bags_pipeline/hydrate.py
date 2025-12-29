# bags_pipeline/hydrate.py
from __future__ import annotations

import json
import re
# import html
from pathlib import Path
# from glob import glob as _glob
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

from .unitize import Unit
# from .ingest_logs import load_events_from_logs, normalize_log_line
from .config import coerce_text
from .config import parse_utc_any
from .quick import quick_unit_list_markdown

# from dataclasses import is_dataclass, replace as _dc_replace
# from types import SimpleNamespace
# from digests_project.bags_pipeline.ingest_logs import normalize_log_line

# convenience alias
PathLike = Union[str, Path]

###---###---###
### SNIPPET
###---###---###


# - for t in text.splitlines(): payload = json.loads(t)
# + from bags_pipeline.io import read_jsonl
# + for payload in read_jsonl_lines(text):  # you might wrap that helper

def _looks_like_code(s: str) -> bool:
    if not s:
        return False
    # If it's already fenced, it's "code", but caller will pass-through
    if _has_fenced_block(s):
        return True

    t = s.strip()

    # Parseable JSON
    if t.startswith("{") or t.startswith("["):
        try:
            json.loads(t)
            return True
        except Exception:
            pass

    # Python at line start
    if re.search(r"(?m)^\s*(def|class|from\s+\S+\s+import|import\s+\S+)\b", s):
        return True

    # Real SQL (require SELECT ... FROM or DDL/DML keywords)
    if re.search(r"(?is)\bSELECT\b.+\bFROM\b|\bCREATE\s+TABLE\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b", s):
        return True

    # 3+ lines indented like a code block
    if sum(1 for ln in s.splitlines() if re.match(r"^\s{4,}\S", ln)) >= 3:
        return True

    return False


def _render_snippet(content: str) -> str:
    s = content or ""
    # Pass through pre-fenced blocks unchanged (``` or ~~~)
    if _has_fenced_block(s):
        return s.strip()

    # Escape HTML **always** for non-fenced payloads to protect MDX
    safe = _escape_html(s)

    # If it now looks like code, fence it; otherwise let Markdown render
    if _looks_like_code(safe):
        return _fence(safe, _guess_lang(safe))
    return safe.strip()

def _has_fenced_block(s: str) -> bool:
    return bool(_FENCE_RE.search(s or ""))


def _escape_html(s: str) -> str:
    # Keep Markdown intact; only neutralize HTML
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# 1) Detect pre-fenced content (``` or ~~~ at start of a line)
_FENCE_RE = re.compile(r"(?m)^\s*(```|~~~)")

def _guess_lang(s: str) -> str:
    t = s.strip()
    if t.startswith("{") or t.startswith("["):
        return "json"
    if re.search(r"(?is)\bSELECT\b.+\bFROM\b|\bCREATE\s+TABLE\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b", s):
        return "sql"
    if re.search(r"(?m)^\s*(def|class|from\s+\S+\s+import|import\s+\S+)\b", s):
        return "python"
    return "text"

def _fence(s: str, lang: str = "text") -> str:
    # use ~~~ to avoid conflicts with ``` inside payloads
    return f"~~~{lang}\n{s.rstrip()}\n~~~\n"


def _render_mdx(front: dict, body: str) -> str:
    """
    Renders YAML front‐matter if possible, otherwise JSON comment,
    then the body.
    """
    try:
        import yaml
        fm_text = f"---\n{yaml.safe_dump(front, sort_keys=False)}---\n\n"
    except ImportError:
        fm_json = json.dumps({"front_matter": front}, ensure_ascii=False, indent=2)
        fm_text = f"<!--\n{fm_json}\n-->\n\n"
    return fm_text + body.lstrip()



###---###---###
### RENDERER
###---###---###

def hydrate_units_stream(docs: Iterator[dict], ev_idx=None, ss_idx=None) -> Iterator[dict]:
    for d in docs:
        yield _attach_snippets_if_any(d, ev_idx, ss_idx)



def _attach_snippets_if_any(unit: dict, ev_idx: dict | None, ss_idx: dict | None) -> dict:
    """
    If unit['sources'] carries resolvable pointers (('event'|'session', id)), attach
    a 'snippets' list under unit['content'].
    """
    u = dict(unit)
    content = dict(u.get("content", {}))
    snippets = list(content.get("snippets", []))

    if isinstance(u.get("sources"), (list, tuple)):
        for src in u["sources"]:
            if not (isinstance(src, (list, tuple)) and len(src) >= 2):
                continue
            kind, sid = src[0], src[1]
            if kind == "event" and isinstance(ev_idx, dict):
                rec = ev_idx.get(sid)
                if rec:
                    snippets.append({
                        "text": rec.get("text", "") or rec.get("summary", ""),
                        "meta": {"id": sid, "ts": rec.get("ts_abs") or rec.get("ts")},
                    })
            elif kind == "session" and isinstance(ss_idx, dict):
                rec = ss_idx.get(sid)
                if rec:
                    snippets.append({
                        "text": rec.get("summary") or rec.get("text", "") or "",
                        "meta": {"id": sid, "ts_start": rec.get("ts_start"), "ts_end": rec.get("ts_end")},
                    })

    # Normalize primary text fields
    if "text" not in content:
        content["text"] = coerce_text(content.get("summary") or "")

    if snippets:
        content["snippets"] = snippets
    u["content"] = content
    return u





from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def _fmt_when(ev: Dict[str, Any]) -> str:
    # Accepts normalize_log_line()-style keys: ts_abs (UTC) and tz_local
    ts = ev.get("ts_abs") or ev.get("timestamp") or ""
    z = ev.get("tz_local") or "UTC"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        loc = dt.astimezone(ZoneInfo(z))
        return f"{loc:%Y-%m-%d %H:%M} ({z})"
    except Exception:
        return ts or z
    
def _in_window_ts(ts: str | None, since_dt, until_dt) -> bool:
    if not (since_dt or until_dt):  # no window
        return True
    t = parse_utc_any(ts)
    if not t:
        return False
    if since_dt and t < since_dt: return False
    if until_dt and t >= until_dt: return False
    return True



# from .snippet_extractor import extract_snippets
# from .render_markdown import render_unit_md
# from .formatters      import format_event_snippet, format_session_snippet
# from .quick_unit_list import quick_unit_list_markdown  # import unchanged

# from typing import Any, List, Tuple, Optional
# from bags_pipeline.textnorm import parse_utc_any
# from bags_pipeline.hydrate import _in_window_ts

def extract_snippets(
    u: Any,  # your Unit type
    event_idx: Optional[dict[str, Any]],
    session_idx: Optional[dict[str, Any]],
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
) -> List[Tuple[str, dict[str, Any]]]:
    """
    Returns a list of (kind, payload) for each unique source in u.sources
    that passes the [since, until) filter.
    """
    since_dt = parse_utc_any(since_iso)
    until_dt = parse_utc_any(until_iso)

    seen = set()
    out: List[Tuple[str, dict[str, Any]]] = []

    for kind, sid in getattr(u, "sources", ()) or ():
        key = (kind, sid)
        if key in seen:
            continue
        seen.add(key)

        if kind == "event" and event_idx:
            ev = event_idx.get(sid)
            if not ev:
                continue
            ts_raw = ev.get("ts_abs") or ev.get("ts") or ev.get("timestamp") or ev.get("created_at")
            if not _in_window_ts(ts_raw, since_dt, until_dt):
                continue
            out.append(("event", ev))

        elif kind == "session" and session_idx:
            lookup = sid if sid in session_idx else ("cluster_" + sid[2:] if sid.startswith("s_") else sid)
            ss = session_idx.get(lookup)
            if not ss:
                continue
            # window check on sessions
            s_ts = ss.get("ts_start") or ss.get("start_ts") or ss.get("ts")
            e_ts = ss.get("ts_end")   or ss.get("end_ts")   or s_ts
            if since_dt or until_dt:
                sdt = parse_utc_any(s_ts)
                edt = parse_utc_any(e_ts) or sdt
                if since_dt and edt <= since_dt:
                    continue
                if until_dt and sdt >= until_dt:
                    continue
            out.append(("session", ss))

    return out


# from typing import Any, List
# from bags_pipeline.hydrate import _fmt_when, _escape_html, _render_snippet

def format_event_snippet(
    ev: dict[str, Any],
    since_iso: str | None,
    until_iso: str | None
) -> List[str]:
    """Turn a single event payload into its Markdown snippet block."""
    lines: List[str] = []

    # header
    sid = ev.get("event_id", "")
    lines.append(f"### event {sid}")

    # title
    title = ev.get("title") or ev.get("summary") or ""
    if title:
        lines.append(f"**{_escape_html(title).strip()}**")

    # timing meta
    when_local = str(_fmt_when(ev))
    meta = f"_{when_local}_"
    ts_utc = ev.get("ts_abs") or ev.get("ts") or ""
    if ts_utc:
        meta += f"\n_UTC: {ts_utc}_"
    lines.append(meta)

    # payload
    payload = ev.get("text") or ev.get("content") or ""
    if payload:
        lines.append(_render_snippet(payload))

    return lines


def format_session_snippet(
    ss: dict[str, Any],
    since_iso: str | None,
    until_iso: str | None
) -> List[str]:
    """Turn a single session payload into its Markdown snippet block."""
    lines: List[str] = []

    sid = ss.get("id", "")
    lines.append(f"### session {sid}")

    title = ss.get("title") or (ss.get("summary") or {}).get("name") or ""
    if title:
        lines.append(f"**{_escape_html(title).strip()}**")

    # meta
    lines.append(f"_session: {sid}_")

    payload = ss.get("text") or ss.get("content") or (ss.get("summary") or {}).get("description", "")
    if payload:
        lines.append(_render_snippet(payload))

    return lines


def materialize_bag_markdown(
    units: Iterable[Unit],
    event_idx: dict[str,Any] | None,
    session_idx: dict[str,Any] | None,
    *,
    collapse: bool    = True,
    max_items: int    = 200,
    since_iso: str|None  = None,
    until_iso: str|None  = None,
    render_mode = None,
) -> str:
    lines: list[str] = []
    for u in units:
        # 1) Pull raw snippets, already window-filtered
        raw_snips = extract_snippets(u, event_idx, session_idx, since_iso, until_iso)

        # 2) Format each into markdown blocks
        md_blocks = []
        for kind, payload in raw_snips:
            if kind == "event":
                md_blocks.append(format_event_snippet(payload, since_iso, until_iso))
            else:
                md_blocks.append(format_session_snippet(payload, since_iso, until_iso))

        # 3) Render per-unit and collect all lines
        lines += render_units_md(
            u,
            md_blocks,
            collapse=collapse,
            since_iso=since_iso,
            until_iso=until_iso
        )

    body = "\n".join(lines).rstrip() + "\n"
    # fallback if nothing rendered:
    return body or quick_unit_list_markdown(units, event_idx, session_idx, max_items=max_items)

from typing import Any, List
from dataclasses import replace
from .textnorm import parse_utc_any

def render_units_md(
    u: Any,  # your Unit
    snippets_md: List[List[str]],
    *,
    collapse: bool,
    since_iso: str | None,
    until_iso: str | None
) -> List[str]:
    """
    Assemble the unit‐level header + metadata lines, then wrap the
    snippets_md blocks in a <details> if collapse=True.
    """
    out: List[str] = []

    # unit header
    out.append(f"# Unit {u.unit_type} — {u.unit_id}")
    out.append(f"- Window: {u.start_ts} → {u.end_ts}")
    if getattr(u, "tags", None):
        out.append(f"- Tags: {', '.join(u.tags)}")
    if getattr(u, "topic_ids", None):
        out.append(f"- Topics: {', '.join(u.topic_ids)}")
    if since_iso or until_iso:
        out.append(f"- Slice: {since_iso or '…'} → {until_iso or '…'}")
    out.append("")

    # flatten all snippet blocks into one list of lines
    flat = []
    count = 0
    for block in snippets_md:
        flat.extend(block)
        flat.append("")  # separator between blocks

    if flat:
        if collapse:
            out += ["<details>", "<summary>🧵 Sources</summary>", ""]
            out += flat
            out += ["", "</details>"]
        else:
            out += flat

    out.append("\n---\n")
    return out





def render_channel(channel:str, u:Unit, scores:Dict[str,float]) -> str:
    # minimal templates; replace with your real renderers/prompts
    if channel=="memo":
        return f"# Memo — {', '.join(u.topic_ids)}\n\n- Window: {u.start_ts} → {u.end_ts}\n- Tags: {', '.join(u.tags)}\n- Salience: {scores['salience']:.2f}\n"
    if channel=="cheatsheet":
        return f"# Cheatsheet — {', '.join(u.topic_ids)}\n\n**Tags**: {', '.join(u.tags)}\n"
    if channel=="tech_debt":
        return f"# Tech Debt — {', '.join(u.topic_ids)}\n\n_Pendientes detectados…_\n"
    if channel=="achievements":
        return f"# Achievements — {', '.join(u.topic_ids)}\n\n_Hitos y cierres…_\n"

    # bags_pipeline/l2.py  (dentro de render_channel)
    if channel == "journal":
        return (
            f"# Journal — {', '.join(u.topic_ids)}\n\n"
            f"- Window: {u.start_ts} → {u.end_ts}\n"
            f"- Unit: {u.unit_type}\n"
            f"- Tags: {', '.join(u.tags)}\n\n"
            f"<!-- aquí podés luego inyectar snippets de sources si indexas eventos/sesiones -->\n"
        )
    
    return f"# {channel} — {', '.join(u.topic_ids)}\n"




# def materialize_bag_markdown(
#     units: Iterable["Unit"],
#     event_idx: Dict[str, Dict[str, Any]] | None,
#     session_idx: Dict[str, Dict[str, Any]] | None,
#     *,
#     collapse: bool = True,
#     max_items: int = 200,
#     since_iso: str | None = None,
#     until_iso: str | None = None,
# ) -> str:
#     since_dt = parse_utc_any(since_iso)
#     until_dt = parse_utc_any(until_iso)
#     out: list[str] = []

#     for u in units:
#         out.append(f"# Unit {u.unit_type} — {u.unit_id}")
#         out.append(f"- Window: {u.start_ts} → {u.end_ts}")
#         if getattr(u, "tags", None):
#             out.append(f"- Tags: {', '.join(u.tags)}")
#         if getattr(u, "topic_ids", None):
#             out.append(f"- Topics: {', '.join(u.topic_ids)}")
#         if since_dt or until_dt:
#             out.append(f"- Slice: {since_iso or '…'} → {until_iso or '…'}")
#         out.append("")

#         inner: list[str] = []
#         seen: set[Tuple[str, str]] = set()
#         n = 0

#         def _bump() -> bool:
#             nonlocal n
#             n += 1
#             return bool(max_items and n > max_items)

#         for kind, sid in getattr(u, "sources", ()) or ():
#             key = (kind, sid)
#             if key in seen:
#                 continue
#             seen.add(key)

#             if kind == "event" and event_idx:
#                 ev = event_idx.get(sid)
#                 if not ev:
#                     continue

#                 ts_raw = ev.get("ts_abs") or ev.get("ts") or ev.get("timestamp") or ev.get("created_at")
#                 if not _in_window_ts(ts_raw, since_dt, until_dt):
#                     continue

#                 title   = ev.get("title") or ev.get("summary") or ""
#                 payload = ev.get("text") or ev.get("content") or ""
#                 role    = ev.get("role")

#                 # _fmt_when may return datetime or str → always cast to str
#                 when_local = str(_fmt_when(ev))
#                 line = f"{when_local}" + (f", role: {role}" if role else "")
#                 meta = f"_{line}_"

#                 ts_utc = ev.get("ts_abs") or ev.get("ts") or ""
#                 if ts_utc:
#                     meta += f"\n_UTC: {ts_utc}_"

#                 inner.append(f"### event {sid}")
#                 if title:
#                     inner.append(f"**{_escape_html(title).strip()}**")
#                 inner.append(meta)
#                 if payload:
#                     inner.append(_render_snippet(payload))
#                 if _bump():
#                     break

#             elif kind == "session" and session_idx:
#                 # tolerate 'cluster_' prefix fallback
#                 lookup = sid if sid in session_idx else ("cluster_" + sid[2:] if sid.startswith("s_") else sid)
#                 ss = session_idx.get(lookup)
#                 if not ss:
#                     continue

#                 s_ts = ss.get("ts_start") or ss.get("start_ts") or ss.get("ts")
#                 e_ts = ss.get("ts_end")   or ss.get("end_ts")   or s_ts

#                 if since_dt or until_dt:
#                     sdt = parse_utc_any(s_ts)
#                     edt = parse_utc_any(e_ts) or sdt
#                     if sdt and edt:
#                         if since_dt and edt <= since_dt:
#                             continue
#                         if until_dt and sdt >= until_dt:
#                             continue

#                 title   = ss.get("title") or (ss.get("summary") or {}).get("name") or ""
#                 payload = ss.get("text") or ss.get("content") or (ss.get("summary") or {}).get("description", "")
#                 meta = f"_session: {lookup}_"

#                 inner.append(f"### session {sid}")
#                 if title:
#                     inner.append(f"**{_escape_html(title).strip()}**")
#                 inner.append(meta)
#                 if payload:
#                     inner.append(_render_snippet(payload))
#                 if _bump():
#                     break

#             else:
#                 continue

#         if inner:
#             if collapse:
#                 out += ["<details>", "<summary>🧵 Sources</summary>", ""]
#                 out.append("\n".join(inner).strip())
#                 out += ["", "</details>"]
#         out.append("\n---\n")

#     return "\n".join(out).rstrip() + "\n"




# def render_units_md(
#         units: Iterable["Unit"],
#         ev_idx: Optional[Dict[str, Dict[str, Any]]] = None,
#         ss_idx: Optional[Dict[str, Dict[str, Any]]] = None,
#         *,
#         tz: str = TZ_LOCAL,
#         max_items: int = 200,
#         collapse: bool = True,
#         since_iso: str | None = None,
#         until_iso: str | None = None,
#         render_mode,
#         **kwargs,
#     ) -> str:
#     """Render a robust Markdown body for a list of Units.

#     - Collapses sources emitted by unit bags.
#     - Looks up events/sessions in indices to pull text/summaries safely.
#     - Truncates output if it would be excessive.

#     Args:
#         units: Units to render (commonly length 1).
#         ev_idx: Event index from `build_event_index`, or None.
#         ss_idx: Session index from `build_session_index`, or None.
#         tz: IANA timezone for headers/window labeling (used by helpers).
#         max_items: Soft cap for item sections.
#         collapse: If True, wrap source list in <details>.
#         since_iso / until_iso: Slice window (UTC ISO strings). When provided,
#             only sources in this window are rendered.

#     Returns:
#         Markdown string (safe to embed after front matter).
#     """


#     body = materialize_bag_markdown(
#         units,
#         ev_idx,
#         ss_idx,
#         collapse=collapse,
#         max_items=max_items,
#         since_iso=since_iso,
#         until_iso=until_iso,
#     )

#     if not body or not body.strip():
#         # very small safety fallback; usually not reached
#         body = quick_unit_list_markdown(units, ev_idx, ss_idx, tz=tz, max_items=max_items)

#     return body


