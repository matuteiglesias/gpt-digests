from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Mapping, Union
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json, re, hashlib

from .core import _get

# --- Global defaults ---
TZ_LOCAL = "America/Argentina/Buenos_Aires"
POLICY_VER = "p1"
PROMPT_VER = "v1"
MODEL_VER = "gpt-5-t"

# Directories (override from CLI)
DIGESTS_DIR = Path("/home/matias/repos/GPT_digests/digests")

# --- JSON & hashing utilities ---

def stable_json(obj: Any) -> str:
    """Deterministic JSON serialization."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":"))

def sha256_of(obj: Any) -> str:
    """16-char sha256 digest of stable_json(obj)."""
    h = hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()
    return h[:16]

# --- Text coercion ---

def coerce_text(x: Any) -> str:
    """Best-effort human-readable text from common containers."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return "\n".join(coerce_text(e) for e in x if e is not None)
    if isinstance(x, dict):
        for key in ("content","text","markdown","md","message","body","value"):
            v = x.get(key)
            if isinstance(v, (str, list, dict)):
                s = coerce_text(v)
                if s.strip():
                    return s
        flat = [str(v) for v in x.values() if isinstance(v, (str, int, float))]
        return " ".join(flat)
    return str(x)

# --- Time & date helpers ---
Timeish = Union[datetime, str, int, float]


def to_utc_dt(x: Any) -> datetime:
    """
    Convert various timestamp forms to an aware UTC datetime.
    Accepts epoch (s or ms), ISO strings with or without 'Z', or datetimes.
    """
    if x is None:
        raise ValueError("timestamp is None")
    # 1) epoch
    if isinstance(x, (int, float)):
        v = float(x)
        if v > 1e12:  # assume ms
            v /= 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc)
    # 2) datetime
    if isinstance(x, datetime):
        dt = x if x.tzinfo else x.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    # 3) string
    if isinstance(x, str):
        s = x.strip()
        # First, collapse any trailing "+00:00Z" → "+00:00"
        s = re.sub(r"\+00:00Z$" , "+00:00", s)
        # Then if it still ends with a lone "Z", drop it
        if s.endswith("Z"):
            s = s[:-1]
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"unsupported timestamp type: {type(x)!r}")

def window_key_day(ts: Timeish, tz: str = TZ_LOCAL) -> str:
    """
    Day key in LOCAL time zone, YYYY-MM-DD.
    """
    # parse into UTC, then convert to local tz for the date
    dt_local = to_utc_dt(ts).astimezone(ZoneInfo(tz))
    return dt_local.date().isoformat()


def parse_utc_any(ts_iso: Optional[str]) -> Optional[datetime]:
    """Parse ISO string to UTC datetime, return None on failure or blank."""
    if not ts_iso:
        return None
    try:
        return to_utc_dt(ts_iso)
    except Exception:
        return None




def iso_week_key(ts: Timeish, tz: str = TZ_LOCAL) -> str:
    dt_local = to_utc_dt(ts).astimezone(ZoneInfo(tz))
    iso = dt_local.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

def month_key(ts: Timeish, tz: str = TZ_LOCAL) -> str:
    dt_local = to_utc_dt(ts).astimezone(ZoneInfo(tz))
    return f"{dt_local.year:04d}-{dt_local.month:02d}"

# --- Mapping & _get helper ---

def _as_mapping(x: Any) -> Optional[Mapping[str,Any]]:
    if isinstance(x, Mapping):
        return x
    if is_dataclass(x):
        try:
            return asdict(x)
        except Exception:
            return None
    return None

# def _get(ev: Any, key: str, default: Any = None) -> Any:
#     # 1) attribute
#     if hasattr(ev, key):
#         try:
#             v = getattr(ev, key)
#             if v is not None:
#                 return v
#         except Exception:
#             pass
#     # 2) mapping
#     m = _as_mapping(ev)
#     if m and key in m:
#         return m[key]
#     # 3) extras
#     ex = getattr(ev, "extras", None) or (m.get("extras") if m else None)
#     if isinstance(ex, Mapping) and key in ex:
#         return ex[key]
#     return default

# --- Event timestamp formatter (local tz) ---

def _event_ts_iso(ev: Any, tz: str = TZ_LOCAL) -> str:
    """
    Pick a timestamp from ev and return as ISO in local tz.
    """
    # prefer ts_abs
    s = _get(ev, "ts_abs")
    if isinstance(s, str) and s:
        s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(ZoneInfo(tz)).isoformat()
        except Exception:
            pass
    # fallback to other fields
    for k in ("timestamp","ts","time","created_at","start"):
        v = _get(ev, k)
        if isinstance(v, (int,float)):
            sec = v/1000 if v>1e10 else v
            return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(ZoneInfo(tz)).isoformat()
        if isinstance(v, str) and v:
            v2 = v[:-1]+"+00:00" if v.endswith("Z") else v
            try:
                dt = datetime.fromisoformat(v2)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(ZoneInfo(tz)).isoformat()
            except Exception:
                continue
    return datetime.now(ZoneInfo(tz)).isoformat()



# --------------------- Core dataclasses (can be moved downstream) ---------------------


# def to_iso_z(ts_ms: int) -> str:
#     return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat().replace("+00:00","Z")

def to_iso_local(day_str: str, hhmm: str, tz: str = TZ_LOCAL) -> str:
    hh, mm = (hhmm or "00:00").split(":")[:2]
    dt = datetime.fromisoformat(f"{day_str}T{hh}:{mm}:00")
    return dt.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc).isoformat().replace("+00:00","Z")

# def window_key_day(ts_iso: str, tz: str = TZ_LOCAL) -> str:
#     dt = datetime.fromisoformat(ts_iso.replace("Z","+00:00")).astimezone(ZoneInfo(tz))
#     return dt.date().isoformat()

# def iso_week_key(ts_iso: str, tz: str = TZ_LOCAL) -> str:
#     dt = datetime.fromisoformat(ts_iso.replace("Z","+00:00")).astimezone(ZoneInfo(tz))
#     iso = dt.isocalendar()
#     return f"{iso.year}-W{iso.week:02d}"

# def month_key(ts_iso: str, tz: str = TZ_LOCAL) -> str:
#     dt = datetime.fromisoformat(ts_iso.replace("Z","+00:00")).astimezone(ZoneInfo(tz))
#     return f"{dt.year:04d}-{dt.month:02d}"


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
    
# def _coerce_iso(ts, z: ZoneInfo) -> str | None: 
#     """Return ISO8601 in tz z when possible; otherwise best-effort string.""" 
#     if ts is None: 
#         return None 
#     if isinstance(ts, (int, float)): 
#         unix = ts / 1000.0 
#     if ts > 10_000_000_000 
#         else ts 
#         return datetime.fromtimestamp(unix, tz=ZoneInfo("UTC")).astimezone(z).isoformat() 
#     s = str(ts).strip() 
#     if not s: 
#         return None 
    
#     # common case: ...Z 
#     if s.endswith("Z"):
#      s = s[:-1] + "+00:00" 
#     try: dt = datetime.fromisoformat(s) 
#     if dt.tzinfo is None: 
#         dt = dt.replace(tzinfo=ZoneInfo("UTC")) 
#         return dt.astimezone(z).isoformat() 
#     except Exception: 
#     return s
 # leave as-is if we can’t parse

def _in_window_range(s_raw: Optional[str], e_raw: Optional[str],
                        since_iso: Optional[str], until_iso: Optional[str]) -> bool:
    if not since_iso and not until_iso:
        return True
    s_dt = parse_utc_any(s_raw) if s_raw else None
    e_dt = parse_utc_any(e_raw) if e_raw else None
    if s_dt and e_dt and e_dt < s_dt:
        s_dt, e_dt = e_dt, s_dt
    since_dt = parse_utc_any(since_iso) if since_iso else None
    until_dt = parse_utc_any(until_iso) if until_iso else None
    if s_dt is None and e_dt is None:
        return True
    if since_dt and e_dt and e_dt <= since_dt:
        return False
    if until_dt and s_dt and s_dt >= until_dt:
        return False
    return True




__all__ = [
    "TZ_LOCAL", "DIGESTS_DIR",
    "stable_json", "sha256_of", "coerce_text",
    "to_utc_dt", "parse_utc_any", "window_key_day", "iso_week_key", "month_key",
    "_as_mapping", "_get", "_event_ts_iso", "_in_window_range"
]
