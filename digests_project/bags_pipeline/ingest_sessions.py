# bags_pipeline/ingest_sessions.py
from pathlib import Path
from typing import List, Optional, Union

from .io import expand_globs, read_jsonl
from .core import Session
from .normalize import canonical_tag
from .config import sha256_of, to_iso_local, TZ_LOCAL

def normalize_session_line(
    day: str,
    raw: dict,
    tz: str = TZ_LOCAL
) -> Optional[Session]:
    """
    Turn one raw JSON dict into a Session, or return None on error.
    """
    try:
        s = raw["summary"]
        start = to_iso_local(day, s.get("startTime", "00:00"), tz)
        end   = to_iso_local(day, s.get("endTime",   "00:00"), tz)
        labels = tuple(sorted(
            canonical_tag(t) for t in s.get("labels", []) if t
        ))
        project = s.get("projectName")
        # stable id on content
        base_for_id = {"start":start, "end":end, "labels":labels, "project":project or ""}
        session_id = "s_" + sha256_of(base_for_id)
        return Session(
            session_id=session_id,
            start_ts=start,
            end_ts=end,
            labels=labels,
            project=project,
            summary=s
        )
    except Exception:
        return None


def load_sessions(
    patterns: Union[str, List[str]],
    tz: str = TZ_LOCAL
) -> List[Session]:
    """
    Expand one or more filename patterns (glob strings or paths),
    read every JSONL file found, normalize each line, dedupe by file+mtime,
    and return a flat list of Session objects.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    sessions: List[Session] = []
    seen_files = set()

    # 1) expand each glob into actual file paths
    files = []
    for pat in patterns:
        files.extend(expand_globs(pat))
    files = sorted(set(files), key=lambda p: str(p))


# def read_jsonl_globs(patterns: list[str]) -> pd.DataFrame:
#     rows=[]
#     for pat in patterns:
#         for fp in glob.glob(pat):
#             with open(fp,'r') as f:
#                 for line in f:
#                     try: rows.append(json.loads(line))
#                     except Exception: pass
#     return pd.DataFrame(rows)


    # 2) iterate, skip duplicates by (path, mtime)
    for fp in files:
        p = Path(fp)
        if not p.is_file():
            continue
        key = (p, p.stat().st_mtime_ns)
        if key in seen_files:
            continue
        seen_files.add(key)

        # 3) read & normalize each JSONL line
        for raw in read_jsonl(p):
            ss = normalize_session_line(p.stem, raw, tz=tz)
            if ss:
                sessions.append(ss)

    return sessions
