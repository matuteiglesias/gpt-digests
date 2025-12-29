# bags_pipeline/logs.py  (or wherever these live)

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Union
from datetime import timezone

import pandas as pd  # optional; feel free to drop if unused

from .core import Event
from .io import read_jsonl, write_jsonl
from .unitize import cohort_units_from_logs as _core_cohorts

# single sources of truth (no duplication)
from .config import (
    TZ_LOCAL, sha256_of, coerce_text, parse_utc_any, window_key_day
)  # :contentReference[oaicite:2]{index=2}

from .normalize import (
    normalize_tags, canonical_tag, parse_tags
)  # :contentReference[oaicite:3]{index=3}


# ------------------------------------------------------------
# 1) normalize_log_line: drop if text is empty (content/text/summary)
# ------------------------------------------------------------
def normalize_log_line(raw: Dict[str, Any], tz_default: str = TZ_LOCAL) -> Optional[Event]:
    """
    Normalize one raw log dict to Event.
    - Reject if ALL text fields are empty (content/text/summary/title).
    - Timestamps parsed via config.parse_utc_any().
    - Tags normalized via normalize.normalize_tags().
    - Deterministic event_id via config.sha256_of().
    """
    # --- choose text once (ban empties early) ---
    text = coerce_text(raw.get("content"))
    if not text.strip():
        text = coerce_text(raw.get("text"))
    # if not text.strip():
    #     text = coerce_text(raw.get("summary"))
    if not text.strip():
        # also tolerate single-space titles; still reject if title has no body
        title_try = coerce_text(raw.get("title"))
        if not title_try.strip():
            return None  # 🚫 nothing to ingest

    # --- timestamp (prefer 'timestamp', 'ts', then 'extras.timestamp') ---
    ts_raw = raw.get("timestamp", raw.get("ts", raw.get("extras", {}).get("timestamp")))
    ts_abs_dt = parse_utc_any(ts_raw)
    if ts_abs_dt is None:
        # last-ditch: some logs only have ISO at 'ts_abs'/'created_at'
        for k in ("ts_abs", "created_at", "time", "time_iso"):
            ts_abs_dt = parse_utc_any(raw.get(k))
            if ts_abs_dt is not None:
                break
    if ts_abs_dt is None:
        return None  # 🚫 cannot place event in time

    # --- role / conv / title ---
    role = str(raw.get("role") or "assistant")
    conv = raw.get("conversation_id")
    title = (coerce_text(raw.get("title")) or "").strip() or None

    # --- tags (parse+canonicalize+dedupe by lexeme) ---
    tags_raw = []
    # merge structured categories into tags (they’ll be canonicalized)
    for k in ("stage", "msg_type", "domain", "note_type", "format_type",
              "category", "subtopic"):
        v = raw.get(k)
        if isinstance(v, str) and v:
            tags_raw.append(f"{k}:{v}")

    # also include any first-class tag containers
    for k in ("tags", "topics", "labels"):
        v = raw.get(k)
        if v is not None:
            tags_raw.extend(parse_tags(v))

    tags = tuple(sorted(normalize_tags(tags_raw, infer_when_empty=False)))

    # --- source & ids ---
    source = raw.get("source") or "gpt_log"
    ts_iso_z = ts_abs_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    event_id = sha256_of({
        "source": source,
        "ts_abs": ts_iso_z,
        "role": role,
        "title": title or "",
        "text_norm": (text or "").strip()[:700],
    })

    # --- extras: keep lightweight leftovers; avoid duplicating heavy text fields
    drop = {
        "content", "text", "summary", "timestamp", "ts", "source",
        "title", "role", "conversation_id", "tags", "topics", "labels",
    }
    extras = {k: v for k, v in raw.items() if k not in drop}
    extras.setdefault("ts_iso", ts_iso_z)

    return Event(
        event_id=event_id,
        source=source,
        ts_abs=ts_abs_dt,
        tz_local=tz_default,
        role=role,
        conversation_id=conv,
        title=title,
        text=text,
        tags=tags,
        extras=extras,
    )


# ------------------------------------------------------------
# 2) load_events_from_logs: pre-filter empties before normalizing
# ------------------------------------------------------------
def load_events_from_logs(globs: Union[str, List[str]], tz_default: str = TZ_LOCAL) -> List[Event]:
    """
    Expand one or more JSONL filename patterns and return normalized Events.
    - Pre-filters raws whose content/text/summary/title are all empty to save work.
    - Uses normalize_log_line() for canonical normalization.
    """
    from glob import glob

    patterns = [globs] if isinstance(globs, str) else list(globs)
    out: List[Event] = []
    seen_files: set[tuple[Path, int]] = set()

    for pattern in patterns:
        for filepath in sorted(glob(pattern)):
            p = Path(filepath)
            if not p.is_file():
                continue
            sig = (p, p.stat().st_mtime_ns)
            if sig in seen_files:
                continue
            seen_files.add(sig)

            for raw in read_jsonl(p):
                # 🚫 drop right here if content == "" (ignore blanks early)
                if isinstance(raw.get("content"), str) and raw.get("content").strip() == "":
                    continue

                # # 💡 super-cheap prefilter (no normalize/config duplication):
                # #    If *every* text-like field is empty/blank, skip.
                # if not any(coerce_text(raw.get(k)).strip() for k in ("content", "text", "summary", "title")):
                #     continue

                ev = normalize_log_line(raw, tz_default=tz_default)
                if ev is not None:
                    out.append(ev)

    return out


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
) -> List["Unit"]:
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
