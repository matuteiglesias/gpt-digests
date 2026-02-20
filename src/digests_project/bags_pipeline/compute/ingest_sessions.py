# ingest_sessions.py
from __future__ import annotations

from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from digests_project.bags_pipeline.compute.core import Session
from digests_project.bags_pipeline.compute.io import read_jsonl


def _ms_to_iso_utc(ms: Any) -> Optional[str]:
    try:
        ms_i = int(ms)
    except Exception:
        return None
    if ms_i <= 0:
        return None
    dt = datetime.fromtimestamp(ms_i / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _coerce_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str):
        return [t.strip() for t in x.split(",") if t.strip()]
    return [str(x)]


def normalize_session_line(raw: Dict[str, Any], *, source_path: Optional[Path] = None) -> Optional[Session]:
    """
    Normalize one session row. Accepts both:
    - your previous "summary.startTime/endTime" style
    - bus sessions style: {session_id, start_ts/end_ts} or {start_ms/end_ms}

    If timestamps are missing or 0, we keep them as None.
    """
    # if not isinstance(raw, dict):
    #     return None

    sid = raw.get("session_id") or raw.get("id") or raw.get("sessionId")
    if not sid:
        # last-resort stable-ish
        sid = f"sess:{hash(str(raw))}"

    # Prefer direct ISO
    # start_ts = raw.get("start_ts") or raw.get("start") or raw.get("startTime")
    # end_ts = raw.get("end_ts") or raw.get("end") or raw.get("endTime")

    # Old nested summary format
    # if not start_ts and isinstance(raw.get("summary"), dict):
    #     s = raw["summary"]
    #     start_ts = s.get("startTime") or s.get("start_ts")
    #     end_ts = s.get("endTime") or s.get("end_ts")

    # Bus numeric timestamps
    # if not start_ts:
    #     for k in ("start_ms", "ts_start_ms", "startTimeMs"):
    #         if k in raw:
    #             start_ts = _ms_to_iso_utc(raw.get(k))
    #             break
    # if not end_ts:
    #     for k in ("end_ms", "ts_end_ms", "endTimeMs"):
    #         if k in raw:
    #             end_ts = _ms_to_iso_utc(raw.get(k))
    #             break

    # If the bus currently writes 0, you get None here, which is fine for now.

    # labels = _coerce_list(raw.get("labels") or raw.get("tags") or raw.get("label_ids"))



    summary = raw.get("summary") or {}
    window = raw.get("window") or {}

    labels = summary.get("labels") or []

    start_ts = _ms_to_iso_utc(window.get("start_ts_ms")) if window.get("start_ts_ms") else None
    end_ts   = _ms_to_iso_utc(window.get("end_ts_ms")) if window.get("end_ts_ms") else None





    project = raw.get("project") or raw.get("workspace") or raw.get("project_name") or None

    # blocks: keep whatever, but ensure list
    blocks = raw.get("blocks") or []
    if not isinstance(blocks, list):
        blocks = [blocks]

    return Session(
        session_id=str(sid),
        start_ts=start_ts if isinstance(start_ts, str) and start_ts.strip() else None,
        end_ts=end_ts if isinstance(end_ts, str) and end_ts.strip() else None,
        labels=tuple(labels),
        # blocks=tuple(blocks),
        project=str(project) if project is not None else None,
    ) # summary, extras


def load_sessions(globs: Union[str, List[str]]) -> List[Session]:
    patterns = [globs] if isinstance(globs, str) else list(globs)
    out: List[Session] = []
    seen_files: set[tuple[Path, int]] = set()

    for pattern in patterns:
        for filepath in sorted(glob(str(pattern))):
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
                s = normalize_session_line(raw, source_path=p)
                if s is not None:
                    out.append(s)

    return out