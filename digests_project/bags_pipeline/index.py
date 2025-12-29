
# bags_pipeline/index.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from .io       import read_jsonl, expand_globs
from .ingest_logs     import normalize_log_line
from .ingest_sessions import normalize_session_line
from .config          import TZ_LOCAL
from glob import glob
PathLike = Union[str, Path]

# def window_index_L2(l2s: List[L2Digest], tz: str) -> List[Dict]:
#     idx: List[Dict] = []
#     for d in l2s:
#         row = {
#             "digest_id": d.digest_id,
#             "channel": d.channel,
#             "unit_type": d.unit_type,
#             "topic_ids": list(d.topic_ids),
#             "salience": d.scores.get("salience",0.0),
#             "day": window_key_day(d.start_ts, tz),
#             "iso_week": iso_week_key(d.start_ts, tz),
#             "month": month_key(d.start_ts, tz)
#         }
#         idx.append(row)
#     return idx


# def write_index_json(idx: List[Dict], path: Path) -> Path:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(stable_json(idx), encoding="utf-8")
#     return path

def _expand_globs(globs: Iterable[PathLike]) -> List[Path]:
    """Turn a list of file-glob patterns into actual Path objects."""
    out: List[Path] = []
    for g in globs or []:
        for p in glob(str(g), recursive=False):
            out.append(Path(p))
    return out


def build_event_index(
    logs_globs: Iterable[PathLike],
    include_alias: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load every JSONL in `logs_globs`, normalize each line into an Event,
    and build a lookup by event_id (plus optional alias keys).
    """
    idx: Dict[str, Dict[str, Any]] = {
        "_meta": {
            "files":          [],
            "scanned":        0,
            "kept_primary":   0,
            "alias_keys":     0,
            "keys_total":     0,
        }
    }

# def normalize_log_line(raw: Dict[str, Any], tz_default: str = TZ_LOCAL) -> Optional[Event]:


    files = _expand_globs(logs_globs)
    idx["_meta"]["files"] = [str(p) for p in files]

    for path in files:
        for raw in read_jsonl(path):
            idx["_meta"]["scanned"] += 1
            ev = normalize_log_line(raw, tz_default=TZ_LOCAL)
            if ev is None:
                continue

            rec = {
                "kind":           "event",
                "event_id":       ev.event_id,
                "ts_abs":         ev.ts_abs,
                "tz_local":       ev.tz_local,
                "role":           ev.role,
                "conversation_id": getattr(ev, "conversation_id", None),
                "title":          ev.title or "",
                "text":           ev.text or "",
                "tags":           tuple(ev.tags or ()),
                "extras":         dict(ev.extras or {}),
                "source_path":    str(path),
            }

            # primary key
            if ev.event_id not in idx:
                idx[ev.event_id] = rec
                idx["_meta"]["kept_primary"] += 1

            # alias key?
            if include_alias:
                alias = rec["extras"].get("id")
                if isinstance(alias, str) and alias and alias not in idx:
                    idx[alias] = rec
                    idx["_meta"]["alias_keys"] += 1

    idx["_meta"]["keys_total"] = idx["_meta"]["kept_primary"] + idx["_meta"]["alias_keys"]
    return idx


def build_session_index(
    sessions_globs: Iterable[PathLike],
) -> Dict[str, Dict[str, Any]]:
    """
    Load every JSONL in `sessions_globs`, normalize each line into a Session,
    and build a lookup by session_id.
    """
    idx: Dict[str, Dict[str, Any]] = {
        "_meta": {
            "files":        [],
            "scanned":      0,
            "kept":         0,
            "keys_total":   0,
        }
    }

    files = _expand_globs(sessions_globs)
    idx["_meta"]["files"] = [str(p) for p in files]

    for path in files:
        for raw in read_jsonl(path):
            idx["_meta"]["scanned"] += 1
            ss = normalize_session_line(path.stem, raw, tz=TZ_LOCAL)
            if ss is None:
                continue

            rec = {
                "kind":       "session",
                "session_id": ss.session_id,
                "ts_start":   ss.start_ts,
                "ts_end":     ss.end_ts,
                "tags":       tuple(ss.labels or ()),
                "summary":    ss.summary or "",
                "extras":     dict(ss.extras or {}),
                "source_path": str(path),
            }
            idx[ss.session_id] = rec
            idx["_meta"]["kept"] += 1

    idx["_meta"]["keys_total"] = idx["_meta"]["kept"]
    return idx


def build_indices(
    event_globs: Iterable[PathLike],
    session_globs: Iterable[PathLike],
    *,
    include_event_alias: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Convenience: build both an event index and a session index.
    Returns (event_index, session_index).
    """
    ev_idx = build_event_index(event_globs, include_alias=include_event_alias) if event_globs else {}
    ss_idx = build_session_index(session_globs) if session_globs else {}
    return ev_idx, ss_idx


import json


def load_or_build_event_index(globs: list[str]) -> dict:
    cache_path = Path("cache/event_index.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    idx = build_event_index(globs)
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    cache_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return idx


def load_or_build_session_index(globs: list[str]) -> dict:
    cache_path = Path("cache/session_index.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    idx = build_session_index(globs)
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    cache_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return idx

