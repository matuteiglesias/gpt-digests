from __future__ import annotations

# bags_pipeline/io.py
import json, csv, re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from json import JSONDecoder, JSONDecodeError
from typing import Any, Iterable
import pandas as pd
import yaml  # you already depend on PyYAML

PathLike = Union[str, Path]

# from .index import build_event_index, build_session_index

def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL (one JSON doc per non-blank line)."""
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, docs: Iterable[Any], *,
                ensure_ascii: bool = False,
                warn_if_empty: bool = True) -> None:
    """
    Atomically write out JSONL, one JSON per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        count = 0
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    tmp.replace(path)
    if warn_if_empty and count == 0:
        print(f"⚠️  {path} is empty", flush=True)



# —————————————————————————————————————————————————————————
# JSON
# —————————————————————————————————————————————————————————

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any, *,
               ensure_ascii: bool = False,
               indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent),
                    encoding="utf-8")




def iter_jsonl(path: Path) -> Iterator[dict]:
    """
    Incremental JSON decoder that tolerates mixed whitespace and partial line garbage.
    Yields objects in order, skipping undecodable fragments by advancing to next newline.
    """
    text = path.read_text(encoding="utf-8")
    dec = JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except JSONDecodeError:
            nl = text.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
            continue
        yield obj
        i = end



from math import isfinite
import hashlib
# ---------------------------- JSON / hashing ---------------------------------

def canonical_json(obj: Any) -> str:
    """Normalize JSON for stable hashing (sort keys, scrub NaN/inf)."""
    def scrub(x):
        if isinstance(x, float):
            return x if isfinite(x) else None
        if isinstance(x, dict):
            return {k: scrub(x[k]) for k in sorted(x)}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        return x
    return json.dumps(scrub(obj), ensure_ascii=True, separators=(",", ":"), sort_keys=True)

# —————————————————————————————————————————————————————————
# CSV
# —————————————————————————————————————————————————————————

def write_csv(path: Path, df, **kwargs) -> None:
    """
    Write a DataFrame to CSV with parent-dir creation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)

import glob

def expand_globs(
    patterns: Union[str, Path, List[Union[str, Path]]],
    recursive: bool = True
) -> List[Path]:
    """
    Expand one or more glob patterns into a sorted, deduplicated list of Paths.
    
    Args:
        patterns: A single glob string/Path, or a list thereof.
        recursive: If True, allows “**” in patterns.
        
    Returns:
        A list of Path objects matching any of the patterns.
    """
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]

    out: List[Path] = []
    for pat in patterns:
        p = str(pat)
        # Use Python’s glob; recursive if asked for
        matches = glob.glob(p, recursive=recursive)
        for m in matches:
            path = Path(m)
            if path.is_file():
                out.append(path)
    # dedupe & sort
    unique = sorted(set(out), key=lambda p: p.as_posix())
    return unique


# def read_jsonl_globs(patterns: list[str]) -> pd.DataFrame:
#     rows=[]
#     for pat in patterns:
#         for fp in glob.glob(pat):
#             with open(fp,'r') as f:
#                 for line in f:
#                     try: rows.append(json.loads(line))
#                     except Exception: pass
#     return pd.DataFrame(rows)

# def read_lev_sess(paths, params):
#     lev = read_jsonl_globs(paths.lev_globs)
#     sess = read_jsonl_globs(paths.sess_globs)
#     return lev, sess


# write_csv(out_dir/"doc_tag_long.csv", long_df)

# def write_csv(df: pd.DataFrame, path: Path): ## ! args order
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, index=False)



from datetime import date

def data_filename(
    *,
    family: str,         # e.g. "logs_cohort"
    bag_type: str,       # "pairbags" | "tagbags" | "sessions"
    window: tuple[str,str],  # ("2025-05-01","2025-07-01")
) -> Path:
    since, until = window
    fn = f"{family}__{bag_type}__{since.replace('-','')}_{until.replace('-','')}.jsonl"
    return Path("data") / fn


import shutil
tmp_root = '.tmp/'

def write_build_tree(
    window_id: str,      # e.g. "2025-05"
    bag_type: str,       # "pairbags" or "tagbags"
    files: Iterable[Path],  # list of mdx files already written under tmp/
) -> None:
    dest = Path("build") / f"{window_id}_{bag_type}"
    if dest.exists():
        shutil.rmtree(dest)
    for src in files:
        # copy preserving relative tree under dest
        rel = src.relative_to(tmp_root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)



def stable_id(obj: Any) -> str:
    """Stable SHA-256 ID from canonical JSON."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()

_front_matter_re = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def read_mdx_front_matter(path: Path) -> Optional[Dict]:
    """
    Extracts front matter from a one‐file MDX:
    ---\n
    yaml...
    ---\n
    or an HTML comment containing a JSON blob.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = _front_matter_re.match(txt)
    if m:
        try:
            fm = yaml.safe_load(m.group(1))
            if isinstance(fm, dict):
                return fm
        except Exception:
            pass

    # fallback: look for <!-- { JSON } -->
    cm = re.search(r"<!--\s*(\{.*?\})\s*-->", txt, re.DOTALL)
    if cm:
        try:
            j = json.loads(cm.group(1))
            return j.get("front_matter", j) if isinstance(j, dict) else {}
        except Exception:
            pass

    return None


# import re, yaml

# def _frontmatter_from_mdx(path: Path) -> Dict[str, Any]:
#     text = path.read_text(encoding="utf-8", errors="ignore").lstrip()
#     if text.startswith("---"):
#         end = text.find("\n---", 3)
#         if end != -1 and yaml is not None:
#             try:
#                 data = yaml.safe_load(text[3:end]) or {}
#                 if isinstance(data, dict):
#                     return data
#             except Exception:
#                 pass
#     m = re.search(r"<!--\s*\{.*?\}\s*-->", text, re.DOTALL)
#     if m:
#         try:
#             j = json.loads(m.group(0).strip("<!-> \n"))
#             return j.get("front_matter", j) if isinstance(j, dict) else {}
#         except Exception:
#             pass
#     return {}




def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path) -> None:
    # atomic-ish copy
    shutil.copy2(src, dst)

def symlink_file(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src)
    except FileExistsError:
        dst.unlink()
        dst.symlink_to(src)


def read_lines(path: Path, encoding="utf-8"):
    with path.open("r", encoding=encoding) as f:
        yield from f

def write_text(path: Path, text: str, encoding="utf-8"):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(text, encoding=encoding)

# ── legacy ─────────────────────────────────────────────────────────────────────

# def read_parquets(paths: list[str]) -> pd.DataFrame:
#     dfs=[]
#     for p in paths:
#         if Path(p).exists():
#             dfs.append(pd.read_parquet(p))
#     return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# def iter_jsonl_files(globs: List[Union[str, Path]]) -> Iterator[Dict[str, Any]]:
#     """Parse heterogeneous JSONL logs into normalized `Event`s.

#     The function is resilient to different field names (`content` / `text` / `summary`),
#     and canonicalizes tags.

#     Args:
#         globs: One or more glob patterns for JSONL log files.
#         tz_default: Fallback IANA timezone for timestamps lacking zone info.

#     Returns:
#         List of normalized Event dataclass instances.
#     """
        
#     seen = set()
#     from glob import glob
#     for pattern in globs:
#         pat = str(pattern)  # ← coerce Path→str (fixes your traceback)
#         for path in sorted(glob(pat)):
#             p = Path(path)
#             if not p.is_file():
#                 continue
#             key = (p, p.stat().st_mtime_ns)
#             if key in seen:
#                 continue
#             seen.add(key)
#             with p.open("r", encoding="utf-8") as f:
#                 for ln, line in enumerate(f, 1):
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         yield __import__("json").loads(line)
#                     except Exception:
#                         continue