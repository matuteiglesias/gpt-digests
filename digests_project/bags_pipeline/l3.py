
from __future__ import annotations
from typing import List, Dict, Tuple

# Signatures only; implement selection & rendering in your env

# def build_L3_daily(idx_rows: List[Dict], day_key: str, *, per_topic_cap: int = 3, total_cap: int = 12) -> Tuple[Dict, str]:
#     """Return (manifest_dict, mdx_text) for the daily digest.
#     - idx_rows: list of index rows (L2-level) already filtered to day_key
#     - Enforce per-topic and total caps; sort by salience desc
#     """
#     # TODO: implement selection, compute topic_mix, assemble MDX
#     manifest = {"digest": {"level":"L3","title": f"Daily — {day_key}"}, "validated": False}
#     mdx = "# Daily Digest\n\n_TODO: render content_\n"
#     return manifest, mdx

# def build_L3W_weekly(idx_rows: List[Dict], week_key: str, *, per_topic_cap: int = 5, total_cap: int = 30) -> Tuple[Dict, str]:
#     """Return (manifest_dict, mdx_text) for the weekly digest."""
#     manifest = {"digest": {"level":"L3W","title": f"Weekly — {week_key}"}, "validated": False}
#     mdx = "# Weekly Digest\n\n_TODO: render content_\n"
#     return manifest, mdx
