# bags_pipeline/dedupe.py
from __future__ import annotations
from typing import Tuple
from pathlib import Path
import json
from .io import iter_jsonl, canonical_json

def _get_by_dotted(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def dedupe_jsonl(inp: Path, out: Path, key: str = "unit_id") -> Tuple[int, int]:
    """
    Returns (kept, dropped). Writes kept to out.
    """
    seen = set()
    kept, dropped = 0, 0
    with out.open("w", encoding="utf-8") as fout:
        for d in iter_jsonl(inp):
            k = _get_by_dotted(d, key)
            if k is None:
                k = canonical_json(d)  # treat missing-key doc as unique by content
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped


# + from bags_pipeline.io import write_jsonl
# + write_jsonl(out, docs, ensure_ascii=False)