# ingest_logs.py
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from digests_project.bags_pipeline.compute.config import TZ_LOCAL
from digests_project.bags_pipeline.compute.core import Event
from digests_project.bags_pipeline.compute.io import read_jsonl



def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _ms_to_iso_utc(ms: Any) -> Optional[str]:
    try:
        ms_i = int(ms)
    except Exception:
        return None
    if ms_i <= 0:
        return None
    dt = datetime.fromtimestamp(ms_i / 1000.0, tz=timezone.utc)
    # Use Z form to be friendly with downstream parsers
    return dt.isoformat().replace("+00:00", "Z")


def _guess_kind(raw: Dict[str, Any], source_path: Optional[Path]) -> str:
    # Prefer explicit
    for k in ("type", "kind", "record_type", "row_type"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Infer by filename
    if source_path is not None:
        name = source_path.name
        if name.endswith(".summary.jsonl"):
            return "summary"
        if name.endswith(".gpt_response.jsonl"):
            return "chunk"

    # Infer by fields
    if raw.get("summary") or raw.get("suggested_actions") or raw.get("scores"):
        return "summary"
    return "chunk"


def normalize_log_line(raw: Dict[str, Any], *, tz_default: str = TZ_LOCAL, source_path: Optional[Path] = None) -> Optional[Event]:
    """
    Normalize one row from chunk_bus or summary_bus into an Event.

    Design goal: never crash on schema drift. If it cannot extract any text-like
    signal at all, return None.
    """
    # if not isinstance(raw, dict):
    #     return None

    kind = _guess_kind(raw, source_path)

    # ---- timestamps ----
    ts_abs = None

    # Prefer bus timestamp
    if "ts_abs_ms" in raw:
        ts_abs = _ms_to_iso_utc(raw.get("ts_abs_ms"))

    # Accept ISO if present
    if not ts_abs:
        for k in ("ts_abs", "timestamp", "ts", "created_at", "time"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                ts_abs = v.strip()
                break

    # If still missing, keep None (downstream grouping should tolerate or you will QA later)
    # ---- identity ----
    meta = raw.get("meta") or outputs or {}

    conv = (
        raw.get("conversation_id")
        or meta.get("conversation_id")
    )

    msg = (
        raw.get("message_id")
        or meta.get("message_id")
    )

    if "ts_abs_ms" in raw:
        ts_abs = _ms_to_iso_utc(raw.get("ts_abs_ms"))

    if not ts_abs:
        ts_abs = _ms_to_iso_utc(meta.get("timestamp"))


    # Stable event_id if possible
    event_id = raw.get("event_id") or raw.get("id")
    if not event_id:
        if conv is not None and msg is not None:
            event_id = f"{conv}:{msg}"
        elif msg is not None:
            event_id = str(msg)
        else:
            # last resort: deterministic-ish fallback (filename + line hash)
            # note: we do not have line number here, so keep it simple
            event_id = f"{kind}:{hash(_coerce_text(raw))}"

    # ---- content extraction (be permissive) ----
    text = ""

    # chunk_bus payload
    if "text" in raw and isinstance(raw["text"], str):
        text = raw["text"].strip()

    # sometimes chunk text is JSON-encoded
    if text.startswith("{") and "}" in text:
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "text" in parsed:
                text = parsed["text"]
        except Exception:
            pass

    # summary_bus payload
    outputs = raw.get("outputs") or {}
    if not text:
        v = outputs.get("summary_text")
        if isinstance(v, str):
            text = v.strip()

    summary = outputs.get("summary_text") or raw.get("summary") or ""
    title = raw.get("title") or raw.get("meta", {}).get("title") or ""

    if not (text or summary or title):
        return None

    # ---- tags/actions/scores ----
    # tags = raw.get("tags") or raw.get("tag_ids") or raw.get("tagIds") or []



    tags = (
        raw.get("tags")
        or outputs.get("tags")
        or raw.get("meta", {}).get("tags")
        or []
    )

    suggested_actions = (
        raw.get("suggested_actions")
        or outputs.get("actions")
        or raw.get("meta", {}).get("suggested_actions")
        or []
    )



    scores = raw.get("scores") or {}
    if not isinstance(scores, dict):
        scores = {}

    # ---- build canonical Event ----
    ev = Event(
        event_id=str(event_id),
        ts_abs=ts_abs,
        # kind=str(kind),
        title=title or None,
        text=text or None,
        # summary=summary or None,
        tags=tuple(str(t) for t in tags if str(t).strip()),
        # suggested_actions=tuple(str(a) for a in suggested_actions if str(a).strip()),
        # scores=scores,
        conversation_id=str(conv) if conv is not None else None,
        # message_id=str(msg) if msg is not None else None,
        source=source_path,
    )
    return ev



import os
import glob
from pathlib import Path
from typing import Iterable, List

def _expand_glob(pat: str) -> str:
    # Expand $VARS first, then ~
    return os.path.expanduser(os.path.expandvars(pat))

def _glob_all(globs: Iterable[str]) -> List[str]:
    files: List[str] = []
    for g in globs:
        gg = _expand_glob(g)
        files.extend(glob.glob(gg))
    return sorted(set(files))


def load_events_from_logs(globs: Union[str, List[str]], tz_default: str = TZ_LOCAL) -> List[Event]:
    """
    Expand 1+ JSONL globs and return normalized Events.

    Behavior:
    - Skips rows with empty/blank text-like fields.
    - Never throws on schema drift: bad rows are ignored.
    """


    # print("GLOBS:", globs)

    # import glob
    # for g in globs:
    #     files = glob.glob(g)
    #     print("MATCH", g, "->", len(files))


    # patterns = [globs] if isinstance(globs, str) else list(globs)
    out: List[Event] = []
    seen_files: set[tuple[Path, int]] = set()

    paths = _glob_all(globs)

    # for pattern in patterns:
        # for filepath in sorted(glob(str(pattern))):

    for filepath in paths:
        p = Path(filepath)


        if not p.is_file():
            continue

        sig = (p, p.stat().st_mtime_ns)
        if sig in seen_files:
            continue
        seen_files.add(sig)

        for raw in read_jsonl(p):
            if not isinstance(raw, dict):
                continue

            # quick blank guard
            c = raw.get("content")
            if isinstance(c, str) and c.strip() == "":
                continue

            ev = normalize_log_line(raw, tz_default=tz_default, source_path=p)
            if ev is not None:
                    out.append(ev)

    return out


# ------------------------------------------------------------
# build_log_cohorts / write_log_cohorts remain as in your file
# ------------------------------------------------------------


# ------------------------------------------------------------
# 3) build_log_cohorts: uses Events; optional time window
# ------------------------------------------------------------
def build_log_cohorts(
    log_globs: List[Union[str, Path]],
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    group_by: str = "day",
    combo_size: int = 2,
    min_events: int = 4,
    top_k_tags: int = 30,
) -> List:
# ) -> List["Unit"]:
    """
    Load Events from JSONL (ignoring empty-content raws),
    slice by time window (if provided), then bucket into cohort Units.
    """
    # 1) load normalized events
    logs_globs = [str(p) for p in log_globs]
    events = load_events_from_logs(logs_globs)

    # 2) optional time slice (inclusive since, exclusive until)
    if since or until:
        t0 = parse_utc_any(since) if since else None
        t1 = parse_utc_any(until) if until else None

        def _in_range(ev: Event) -> bool:
            if ev.ts_abs is None:
                return False
            if t0 and ev.ts_abs < t0:
                return False
            if t1 and ev.ts_abs >= t1:
                return False
            return True

        events = [e for e in events if _in_range(e)]

    # 3) call core unitizer (expects Event objects)
    units = _core_cohorts(
        events,
        group_by=group_by,
        combo_size=combo_size,
        tz=TZ_LOCAL,
        min_events=min_events,
        top_k_tags=top_k_tags,
    )
    return units


# ------------------------------------------------------------
# 4) write_log_cohorts: facade that persists Units to JSONL
# ------------------------------------------------------------
def write_log_cohorts(
    log_globs: List[Union[str, Path]],
    out_path: Path,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    group_by: str = "day",
    combo_size: int = 2,
    **core_kwargs,
) -> int:
    """
    Full pipeline: build cohorts from logs, persist as JSONL.
    Returns number of units written.
    """
    units = build_log_cohorts(
        log_globs,
        since=since,
        until=until,
        group_by=group_by,
        combo_size=combo_size,
        **core_kwargs,
    )
    write_jsonl(out_path, (u.__dict__ for u in units))
    return len(units)
