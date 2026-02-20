# bags_pipeline/eda_bridge.py
from __future__ import annotations
from typing import Any, Dict, Iterable

import pandas as pd

from .core import Unit
from .pairs import (
    co_tag_pairs,
    tag_communities_from_pairs,
    GatePolicy,
    subsets,                    # now encapsulates gating via GatePolicy or gates override
)
from .normalize import normalize_tags



def long_from_units(units: Iterable[Any]) -> pd.DataFrame:
    """
    Build long doc-tag dataframe using your normalize pipeline.
    - Uses explicit `tags` if present; does NOT infer from text here (keep EDA stable).
    - De-duplicates (doc_id, tag) pairs.
    """
    rows: list[dict] = []
    for u in units:
        uid = (u.get("unit_id") if isinstance(u, dict) else getattr(u, "unit_id", None))
        if not uid:
            continue
        mapping = u if isinstance(u, dict) else vars(u)
        tags = normalize_tags(mapping.get("tags", []), infer_when_empty=False)
        for t in tags:
            rows.append({"doc_id": uid, "tag": t})

    if not rows:
        return pd.DataFrame(columns=["doc_id", "tag"])

    df = pd.DataFrame(rows, columns=["doc_id", "tag"]).drop_duplicates()
    return df


def pairs_from_units(
    units: Iterable[Unit | Dict[str, Any]],
    top_k: int = 300,
    min_docs: int = 5,
    min_npmi: float = 0.05,
    *,
    # Optional: let callers steer gating here (or pass None and let subsets() use defaults)
    gates: Dict[str, float] | None = None,
    gate_policy: GatePolicy | None = None,
    k_top: int = 12,
) -> Dict[str, pd.DataFrame]:
    """
    Compute pairs + communities + standard subsets, delegating gating to `subsets()`.

    - If `gates` is provided, it overrides policy/quantiles in `subsets()`.
    - Else if `gate_policy` is provided, `subsets()` computes gates from quantiles with that policy.
    - Else `subsets()` uses its default GatePolicy.
    """
    long_df = long_from_units(units)

    # Stable empty schema
    if long_df.empty:
        empty_pairs = pd.DataFrame(columns=["tag_a", "tag_b", "co_docs", "npmi", "lift", "edge_score"])
        empty_comm  = pd.DataFrame(columns=["tag", "community"])
        return {"pairs": empty_pairs, "communities": empty_comm, "subsets": {}}

    # Relax min_docs if corpus is tiny
    n_docs = int(long_df["doc_id"].nunique())
    eff_min_docs = min_docs if n_docs >= min_docs else 1

    pairs = co_tag_pairs(long_df, top_k=top_k, min_docs=eff_min_docs, min_npmi=min_npmi)
    if pairs is None or pairs.empty:
        empty_pairs = pd.DataFrame(columns=["tag_a", "tag_b", "co_docs", "npmi", "lift", "edge_score"])
        empty_comm  = pd.DataFrame(columns=["tag", "community"])
        return {"pairs": empty_pairs, "communities": empty_comm, "subsets": {}}

    comm = tag_communities_from_pairs(pairs)

    # Let subsets() handle gates/policy; no quantile_gates call here.
    out = subsets(
        pairs,
        comm=comm,
        stats=None,
        k_top=k_top,
        gates=gates,
        gate_policy=gate_policy,
    )

    return {"pairs": pairs, "communities": comm, "subsets": out}
