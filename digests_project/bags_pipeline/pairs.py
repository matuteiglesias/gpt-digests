# pairs.py  — analytics primitives only (no Unit imports, no I/O)

# pairs.py (or wherever these live)
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Tuple, Dict, List
import numpy as np
from .core import Unit

import pandas as pd

# NEW: use the shared normalizer
from digests_project.bags_pipeline.normalize import (
    canonical_tag as _canon,
    lexeme as _lexeme,
)

# ----------------------------- main primitives -------------------------------

def tag_communities_from_pairs(pairs: pd.DataFrame, min_npmi: float = 0.05, top_k: int = 12) -> pd.DataFrame:
    # Guards
    if pairs is None or pairs.empty or not {"tag_a", "tag_b", "npmi"}.issubset(pairs.columns):
        return pd.DataFrame(columns=["tag", "community"])

    try:
        import networkx as nx
    except Exception:
        return pd.DataFrame(columns=["tag", "community"])

    sub = pairs[pairs["npmi"] >= min_npmi].copy()

    # Canonicalize pair endpoints via normalize
    sub["tag_a"] = sub["tag_a"].map(_canon)
    sub["tag_b"] = sub["tag_b"].map(_canon)

    # Keep only top-k strongest neighbors per node
    adj: dict[str, list[tuple[str, float]]] = {}
    for _, r in sub.iterrows():
        adj.setdefault(r["tag_a"], []).append((r["tag_b"], float(r["npmi"])))
        adj.setdefault(r["tag_b"], []).append((r["tag_a"], float(r["npmi"])))

    keep: set[tuple[str, str]] = set()
    for n, neigh in adj.items():
        for nb, _w in sorted(neigh, key=lambda x: x[1], reverse=True)[:top_k]:
            keep.add(tuple(sorted((n, nb))))

    sub2 = sub[sub.apply(lambda r: tuple(sorted((r["tag_a"], r["tag_b"]))) in keep, axis=1)]

    G = nx.Graph()
    for _, r in sub2.iterrows():
        G.add_edge(r["tag_a"], r["tag_b"], weight=float(r["npmi"]))

    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=["tag", "community"])

    from networkx.algorithms.community import greedy_modularity_communities
    comms = greedy_modularity_communities(G, weight="weight")

    rows = []
    for cid, com in enumerate(comms):
        for t in com:
            rows.append({"tag": t, "community": cid})
    return pd.DataFrame(rows)


def co_tag_pairs(
    long_df: pd.DataFrame,
    *,
    top_k: int = 300,
    min_docs: int = 5,
    min_npmi: float = 0.05,
    canonicalize: bool = False,
) -> pd.DataFrame:
    """
    Compute co-occurring tag pairs with NPMI, Lift and an edge score.

    Input
    -----
    long_df : DataFrame with columns ["doc_id", "tag"] (assumed de-duplicated per doc).
    top_k   : keep only the top-K most frequent tags before pairing (sparsifies).
    min_docs: minimum co-document frequency to keep a pair.
    min_npmi: minimum NPMI to keep a pair.
    canonicalize : if True, canonicalize tags with project-wide normalizer.

    Returns
    -------
    DataFrame columns:
      ["tag_a","tag_b","co_docs","lift","npmi","lex_a","lex_b",
       "is_tautology_like","edge_score"]
    """
    if long_df is None or long_df.empty or not {"doc_id", "tag"}.issubset(long_df.columns):
        return pd.DataFrame(columns=["tag_a", "tag_b", "co_docs", "lift", "npmi", "edge_score"])

    df = long_df[["doc_id", "tag"]].copy()

    # Enforce per-doc tag dedupe locally for safety
    df = df.drop_duplicates(["doc_id", "tag"])

    if canonicalize:
        df["tag"] = df["tag"].map(_canon)

    # Focus on the frequent vocabulary (prevents giant cliques)
    top_vocab = df["tag"].value_counts().head(top_k).index.tolist()
    filt = df[df["tag"].isin(top_vocab)]

    N = int(filt["doc_id"].nunique()) or 1
    freq = filt["tag"].value_counts().to_dict()

    # Co-document counts
    co: Counter[Tuple[str, str]] = Counter()
    for _, g in filt.groupby("doc_id", sort=False):
        tags = sorted(pd.unique(g["tag"]))
        for a, b in combinations(tags, 2):
            co[(a, b)] += 1

    rows = []
    eps = 1e-12
    for (a, b), c in co.items():
        if c < min_docs:
            continue
        pa, pb = freq[a] / N, freq[b] / N
        pij = c / N
        pij_c = min(max(pij, eps), 1 - eps)
        lift = pij / (pa * pb) if pa * pb > 0 else 0.0
        npmi = (math.log(pij_c / (pa * pb + eps))) / (-math.log(pij_c))
        if npmi >= min_npmi:
            rows.append(
                {
                    "tag_a": a,
                    "tag_b": b,
                    "co_docs": int(c),
                    "lift": float(lift),
                    "npmi": float(npmi),
                    "lex_a": _lexeme(a),
                    "lex_b": _lexeme(b),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["is_tautology_like"] = (out["lex_a"] == out["lex_b"]) & (out["tag_a"] != out["tag_b"])
    out["edge_score"] = (
        out["npmi"].clip(lower=0)
        * pd.Series(out["co_docs"], dtype="float64").rpow(0).add(out["co_docs"]).apply(lambda x: math.log1p(max(x, 0)))
        * (out["lift"].clip(lower=1.0000001)).apply(math.log)
    )
    return out.sort_values(["edge_score", "npmi", "co_docs"], ascending=False)

# from .core import GatePolicy, Quantiles, Gates, _round_count

# gating.py (or inline above subsets() if you prefer keeping one file)
from dataclasses import dataclass
from typing import Dict, Literal, Tuple
import math
import pandas as pd

Quantiles = Dict[str, Dict[float, float]]
Gates     = Dict[str, float]


# {
#   "CO_DEFAULT": 20,      // was 25–40; 20 lets more real edges in
#   "CO_BACKBONE": 35,     // was 40–60; 35 is closer to your Q75
#   "CO_NICHE_LO": 8,      // was 10
#   "CO_NICHE_HI": 35,     // was 40
#   "NPMI_KEEP": 0.10,     // was 0.12
#   "LIFT_KEEP": 2.1,      // was 2.3
#   "NPMI_STRONG": 0.12,   // unchanged (use your Q75 if higher)
#   "LIFT_STRONG": 2.3,
#   "NPMI_VSTRONG": 0.16,  // unchanged
#   "LIFT_VSTRONG": 3.0,
#   "NPMI_BRIDGE": 0.18    // keep; works well for this corpus
# }


@dataclass(frozen=True)
class GatePolicy:
    # Floors (absolute minima)
    co_default_floor: int = 20
    co_backbone_floor: int = 35
    npmi_keep_floor:  float = 0.1
    lift_keep_floor:  float = 2.1

    # Count rounding
    round_counts: Literal["none","nearest","up"] = "nearest"


# {
#   "CO_DEFAULT": 25,
#   "CO_BACKBONE": 60,
#   "CO_NICHE_LO": 10,
#   "CO_NICHE_HI": 40,
#   "NPMI_KEEP": 0.12,
#   "LIFT_KEEP": 2.3,
#   "NPMI_STRONG": 0.12,
#   "LIFT_STRONG": 2.28,
#   "NPMI_VSTRONG": 0.167,
#   "LIFT_VSTRONG": 3.037,
#   "NPMI_BRIDGE": 0.18
# }



    # Quantiles used for each knob
    q_for_co_default:   float = 0.4
    q_for_co_backbone:  float = 0.6
    q_for_npmi_keep:    float = 0.4
    q_for_npmi_strong:  float = 0.75
    q_for_npmi_vstrong: float = 0.90
    q_for_lift_keep:    float = 0.4
    q_for_lift_strong:  float = 0.75
    q_for_lift_vstrong: float = 0.90

    # Niche band
    niche_lo: int | None = 8
    niche_hi: int | None = 55
    niche_hi_strategy: Literal["above_q75","below_default","fixed"] = "above_q75"

    # Bridges policy
    min_bridge_floor: float = 0.26

    # Softeners for sparse corpora
    min_co_default_allowed:  int = 10
    min_co_backbone_allowed: int = 20

def _round_count(x: float, how: str) -> int:
    if how == "none":
        return int(x)
    if how == "up":
        return int(math.ceil(x))
    return int(round(x))  # nearest



def compute_gates(df: pd.DataFrame, policy: GatePolicy) -> Tuple[Quantiles, Gates]:
    if df is None or df.empty:
        q: Quantiles = {"co": {}, "lift": {}, "npmi": {}}
        g: Gates = {
            "CO_DEFAULT":  float(policy.co_default_floor),
            "CO_BACKBONE": float(policy.co_backbone_floor),
            "CO_NICHE_LO": float(policy.niche_lo if policy.niche_lo is not None else 8),
            "CO_NICHE_HI": float(policy.niche_hi if policy.niche_hi is not None else 55),
            "NPMI_KEEP":   float(policy.npmi_keep_floor),
            "LIFT_KEEP":   float(policy.lift_keep_floor),
            "NPMI_STRONG": float(policy.npmi_keep_floor),
            "LIFT_STRONG": float(policy.lift_keep_floor),
            "NPMI_VSTRONG": float(max(policy.min_bridge_floor, policy.npmi_keep_floor)),
            "LIFT_VSTRONG": float(policy.lift_keep_floor),
            "NPMI_BRIDGE":  float(max(policy.min_bridge_floor, policy.npmi_keep_floor)),
        }
        return q, g

    base = df.copy()
    for c in ["co_docs", "lift", "npmi"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=["co_docs", "lift", "npmi"])

    qs = [0.50, 0.75, 0.90]
    q_co   = base["co_docs"].quantile(qs).to_dict()
    q_lift = base["lift"].quantile(qs).to_dict()
    q_npmi = base["npmi"].quantile(qs).to_dict()
    q: Quantiles = {"co": q_co, "lift": q_lift, "npmi": q_npmi}

    def qv(qmap: Dict[float, float], q: float, default: float = 0.0) -> float:
        return float(qmap.get(q, default) or 0.0)

    # raw from quantiles
    co_def_raw    = qv(q_co,   policy.q_for_co_default)
    co_bb_raw     = qv(q_co,   policy.q_for_co_backbone)
    npmi_keep_raw = qv(q_npmi, policy.q_for_npmi_keep)
    npmi_strong   = qv(q_npmi, policy.q_for_npmi_strong)
    npmi_vstrong  = qv(q_npmi, policy.q_for_npmi_vstrong)
    lift_keep_raw = qv(q_lift, policy.q_for_lift_keep)
    lift_strong   = qv(q_lift, policy.q_for_lift_strong)
    lift_vstrong  = qv(q_lift, policy.q_for_lift_vstrong)

    # floors + rounding
    CO_DEFAULT  = max(policy.co_default_floor,  _round_count(co_def_raw, policy.round_counts))
    CO_BACKBONE = max(policy.co_backbone_floor, _round_count(co_bb_raw,  policy.round_counts))
    CO_DEFAULT  = max(policy.min_co_default_allowed,  CO_DEFAULT)
    CO_BACKBONE = max(policy.min_co_backbone_allowed, CO_BACKBONE)

    NPMI_KEEP = max(policy.npmi_keep_floor, float(npmi_keep_raw))
    LIFT_KEEP = max(policy.lift_keep_floor, float(lift_keep_raw))

    # niche band
    if policy.niche_lo is not None:
        CO_NICHE_LO = int(policy.niche_lo)
    else:
        CO_NICHE_LO = _round_count(max(10, 0.5 * CO_DEFAULT), policy.round_counts)

    if policy.niche_hi is not None:
        CO_NICHE_HI = int(policy.niche_hi)
    else:
        if policy.niche_hi_strategy == "above_q75":
            CO_NICHE_HI = _round_count(max(CO_NICHE_LO + 1, qv(q_co, 0.75)), policy.round_counts)
        elif policy.niche_hi_strategy == "below_default":
            CO_NICHE_HI = max(CO_NICHE_LO + 1, CO_DEFAULT - 1)
        else:
            CO_NICHE_HI = 60

    # bridge & strength gates
    NPMI_STRONG  = float(npmi_strong)
    NPMI_VSTRONG = float(npmi_vstrong)
    LIFT_STRONG  = float(lift_strong)
    LIFT_VSTRONG = float(lift_vstrong)
    NPMI_BRIDGE  = max(policy.min_bridge_floor, NPMI_STRONG)

    g: Gates = {
        "CO_DEFAULT":   float(CO_DEFAULT),
        "CO_BACKBONE":  float(CO_BACKBONE),
        "CO_NICHE_LO":  float(CO_NICHE_LO),
        "CO_NICHE_HI":  float(CO_NICHE_HI),
        "NPMI_KEEP":    float(NPMI_KEEP),
        "LIFT_KEEP":    float(LIFT_KEEP),
        "NPMI_STRONG":  float(NPMI_STRONG),
        "LIFT_STRONG":  float(LIFT_STRONG),
        "NPMI_VSTRONG": float(NPMI_VSTRONG),
        "LIFT_VSTRONG": float(LIFT_VSTRONG),
        "NPMI_BRIDGE":  float(NPMI_BRIDGE),
    }
    return q, g





from typing import Any


# pairs.py (or wherever subsets() lives)
from typing import Any, Dict
import pandas as pd
import numpy as np

# import helpers you already have:
# _add_lex_cols, _edge_score, _sparsify_topk, _map_communities
# from .core import compute_gates, GatePolicy

def subsets(
    pairs: pd.DataFrame,
    *,
    comm: pd.DataFrame | None = None,
    stats: pd.DataFrame | None = None,
    k_top: int = 12,
    gates: Dict[str, float] | None = None,
    gate_policy: GatePolicy | None = None,   # <— NEW: pass a policy instead of gates
) -> Dict[str, Any]:
    """
    Build standard filtered edge sets: default, backbone, niche, bridges, audit.
    Returns dict with: quantiles (dict), gates (dict), edges_* (DFs), edges_*_topk (DFs), dehub_tags (list), counts (dict).
    """
    # ---------- empty guard ----------
    if pairs is None or pairs.empty:
        empty = pairs if isinstance(pairs, pd.DataFrame) else pd.DataFrame()
        return {
            "quantiles": {},
            "gates": {},
            "dehub_tags": [],
            "edges_default": empty,
            "edges_default_topk": empty,
            "edges_backbone": empty,
            "edges_backbone_topk": empty,
            "edges_niche": empty,
            "edges_niche_topk": empty,
            "edges_bridges": empty,
            "edges_bridges_topk": empty,
            "edges_audit": empty,
            "counts": {"default":0,"backbone":0,"niche":0,"bridges":0,"audit":0},
        }

    # ---------- normalize & enrich once ----------
    base = pairs.copy()
    for c in ["co_docs", "lift", "npmi"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.dropna(subset=["co_docs", "lift", "npmi"])
    base = base[~base["tag_a"].isin(["nan", "NaN"]) & ~base["tag_b"].isin(["nan", "NaN"])]

    base = _add_lex_cols(base)
    base["edge_score"] = _edge_score(base)
    base = _map_communities(base, comm)


    # ---------- thresholds ----------
    if gates is None:
        policy = gate_policy or GatePolicy()
        q, g = compute_gates(base, policy)
    else:
        # respect provided gates; still compute quantiles for reporting
        q, _ = compute_gates(base, GatePolicy())
        g = gates

    CO_DEFAULT   = int(g["CO_DEFAULT"])
    CO_BACKBONE  = int(g["CO_BACKBONE"])
    CO_NICHE_LO  = int(g["CO_NICHE_LO"])
    CO_NICHE_HI  = int(g["CO_NICHE_HI"])
    NPMI_KEEP    = float(g["NPMI_KEEP"])
    LIFT_KEEP    = float(g["LIFT_KEEP"])
    NPMI_STRONG  = float(g["NPMI_STRONG"])
    NPMI_VSTRONG = float(g["NPMI_VSTRONG"])
    LIFT_VSTRONG = float(g["LIFT_VSTRONG"])
    NPMI_BRIDGE  = float(g["NPMI_BRIDGE"])

    # ---------- exclude tautology-like from main views ----------
    base_main = base[~base["is_tautology_like"]].copy()

    # ---------- views ----------
    m_default = (base_main["co_docs"] >= CO_DEFAULT) & (
        (base_main["npmi"] >= NPMI_KEEP) | (base_main["lift"] >= LIFT_KEEP)
    )
    edges_default = base_main.loc[m_default].sort_values(
        ["edge_score", "npmi", "co_docs", "lift"], ascending=[False, False, False, False]
    )

    m_backbone = (base_main["co_docs"] >= CO_BACKBONE) & (base_main["npmi"] >= max(0.20, NPMI_STRONG))
    edges_backbone = base_main.loc[m_backbone].sort_values(
        ["co_docs", "npmi", "lift"], ascending=[False, False, False]
    )

    m_niche = base_main["co_docs"].between(CO_NICHE_LO, CO_NICHE_HI, inclusive="both") & (
        (base_main["npmi"] >= NPMI_VSTRONG) | (base_main["lift"] >= LIFT_VSTRONG)
    )
    edges_niche = base_main.loc[m_niche].sort_values(
        ["npmi", "co_docs", "lift"], ascending=[False, False, False]
    )

    # for c in ("ca", "cb"):
    #     if c in base.columns:
    #         base[c] = base[c].astype("Int64")

    if {"ca", "cb"}.issubset(base_main.columns):
        m_bridges = (
            base_main["ca"].notna()
            & base_main["cb"].notna()
            & (base_main["ca"] != base_main["cb"])
            & (base_main["npmi"] >= NPMI_BRIDGE)
        )
        edges_bridges = base_main.loc[m_bridges].sort_values(
            ["edge_score", "npmi", "co_docs", "lift"], ascending=[False, False, False, False]
        )
    else:
        edges_bridges = base_main.iloc[0:0].copy()

    m_thin = (base["co_docs"] < CO_NICHE_LO) & (
        (base["lift"] >= LIFT_VSTRONG) | (base["npmi"] >= NPMI_VSTRONG)
    )
    edges_audit = pd.concat(
        [base.loc[m_thin], base.loc[base["is_tautology_like"]]],
        ignore_index=True,
    ).drop_duplicates(["tag_a", "tag_b"])

    # ---------- sparsify only for *_topk ----------
    edges_default_topk  = _sparsify_topk(edges_default,  k_top)
    edges_backbone_topk = _sparsify_topk(edges_backbone, k_top)
    edges_niche_topk    = _sparsify_topk(edges_niche,    k_top)
    edges_bridges_topk  = _sparsify_topk(edges_bridges,  k_top)

    # ---------- dehub tags ----------
    dehub_tags: list[str] = []
    if stats is not None and not stats.empty and {"tag", "doc_freq"}.issubset(stats.columns):
        cutoff = stats["doc_freq"].quantile(0.98)
        dehub_tags = (
            stats.loc[stats["doc_freq"] >= cutoff, "tag"]
            .dropna().astype(str).drop_duplicates().tolist()
        )

    return {
        "quantiles": q,
        "gates": g,
        "edges_default": edges_default,
        "edges_backbone": edges_backbone,
        "edges_niche": edges_niche,
        "edges_bridges": edges_bridges,
        "edges_audit": edges_audit,
        "edges_default_topk": edges_default_topk,
        "edges_backbone_topk": edges_backbone_topk,
        "edges_niche_topk": edges_niche_topk,
        "edges_bridges_topk": edges_bridges_topk,
        "dehub_tags": dehub_tags,
        "counts": {
            "default":  int(edges_default.shape[0]),
            "backbone": int(edges_backbone.shape[0]),
            "niche":    int(edges_niche.shape[0]),
            "bridges":  int(edges_bridges.shape[0]),
            "audit":    int(edges_audit.shape[0]),
        },
    }


# ------------------------------- subset logic --------------------------------

def _add_lex_cols(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs.copy()
    df["lex_a"] = df["tag_a"].apply(_lexeme)
    df["lex_b"] = df["tag_b"].apply(_lexeme)
    df["is_tautology_like"] = (df["lex_a"] == df["lex_b"]) & (df["tag_a"] != df["tag_b"])
    return df


def _edge_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["npmi"].clip(lower=0.0)
        * np.log1p(df["co_docs"].clip(lower=0.0))
        * np.log(df["lift"].clip(lower=1.0000001))
    )


def _sparsify_topk(df_edges: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    if df_edges is None or df_edges.empty:
        return df_edges
    df_edges = df_edges.copy()
    df_edges["pair_key"] = df_edges.apply(lambda r: tuple(sorted((r["tag_a"], r["tag_b"]))), axis=1)
    df_edges = df_edges.drop_duplicates("pair_key")
    keep: set[Tuple[str, str]] = set()
    for a_col, b_col in [("tag_a", "tag_b"), ("tag_b", "tag_a")]:
        for _, sub in df_edges.groupby(a_col):
            sub = sub.sort_values(
                ["npmi", "co_docs", "lift", "edge_score"],
                ascending=[False, False, False, False],
            ).head(k)
            keep.update(sub["pair_key"].tolist())
    return df_edges[df_edges["pair_key"].isin(keep)].drop(columns=["pair_key"])


def _map_communities(pairs: pd.DataFrame, comm: pd.DataFrame | None) -> pd.DataFrame:
    """Attach community ids 'ca'/'cb' to pairs if a communities dataframe is provided."""
    if comm is None or comm.empty or not {"tag", "community"}.issubset(comm.columns):
        return pairs
    df = pairs.copy()
    ca = comm.rename(columns={"tag": "tag_a", "community": "ca"})
    cb = comm.rename(columns={"tag": "tag_b", "community": "cb"})
    df = df.merge(ca[["tag_a", "ca"]], on="tag_a", how="left")
    df = df.merge(cb[["tag_b", "cb"]], on="tag_b", how="left")
    return df


# --------------------------- community detection -----------------------------

# def tag_communities_from_pairs(
#     pairs: pd.DataFrame, *, min_npmi: float = 0.05, top_k: int = 12
# ) -> pd.DataFrame:
#     """
#     Lightweight community detection over tag pairs using greedy modularity.

#     Returns DataFrame: ["tag", "community"]. Empty on error or no edges.
#     """
#     if pairs is None or pairs.empty or not {"tag_a", "tag_b", "npmi"}.issubset(pairs.columns):
#         return pd.DataFrame(columns=["tag", "community"])

#     try:
#         import networkx as nx  # optional dependency
#         from networkx.algorithms.community import greedy_modularity_communities
#     except Exception:
#         return pd.DataFrame(columns=["tag", "community"])

#     sub = pairs.loc[pairs["npmi"] >= min_npmi, ["tag_a", "tag_b", "npmi"]].copy()
#     sub["tag_a"] = sub["tag_a"].map(_canon)
#     sub["tag_b"] = sub["tag_b"].map(_canon)

#     # keep strongest neighbors per node to reduce hairballs
#     adj: Dict[str, list[tuple[str, float]]] = {}
#     for _, r in sub.iterrows():
#         adj.setdefault(r["tag_a"], []).append((r["tag_b"], float(r["npmi"])))
#         adj.setdefault(r["tag_b"], []).append((r["tag_a"], float(r["npmi"])))

#     keep: set[Tuple[str, str]] = set()
#     for node, neigh in adj.items():
#         for nb, _w in sorted(neigh, key=lambda x: x[1], reverse=True)[:top_k]:
#             keep.add(tuple(sorted((node, nb))))

#     if not keep:
#         return pd.DataFrame(columns=["tag", "community"])

#     mask = sub.apply(lambda r: tuple(sorted((r["tag_a"], r["tag_b"]))) in keep, axis=1)
#     sub2 = sub.loc[mask]

#     G = nx.Graph()
#     for _, r in sub2.iterrows():
#         G.add_edge(r["tag_a"], r["tag_b"], weight=float(r["npmi"]))

#     if G.number_of_edges() == 0:
#         return pd.DataFrame(columns=["tag", "community"])

#     comms = greedy_modularity_communities(G, weight="weight")
#     rows = [{"tag": t, "community": int(cid)} for cid, com in enumerate(comms) for t in com]
#     return pd.DataFrame(rows)


from .config import sha256_of



def pairbag_units_from_units(units: List[Unit],
                             pairs_df: pd.DataFrame | None = None,
                             top_n: int = 50,
                             min_docs: int = 2) -> List[Unit]:
    """
    Crea Units virtuales 'pairbag' por par de tags importante.
    - Si pairs_df es None, deriva pares rápidos por co-ocurrencia simple.
    - Agrupa todas las units originales que contengan ambos tags.
    """
    # 1) derivar pares si no viene un pairs_df listo
    if pairs_df is None or pairs_df.empty:
        from collections import Counter
        # contar co-ocurrencias por unit
        co = Counter()
        for u in units:
            tags = sorted(set(u.tags))
            for i in range(len(tags)):
                for j in range(i+1, len(tags)):
                    co[(tags[i], tags[j])] += 1
        # armar DF mínimo
        rows = [{"tag_a": a, "tag_b": b, "co_docs": n} for (a,b),n in co.items()]
        pairs_df = pd.DataFrame(rows)
        if pairs_df.empty:
            return []
        pairs_df["npmi"] = 0.0
        pairs_df["lift"] = 0.0
        pairs_df["edge_score"] = pairs_df["co_docs"]

    # 2) filtrar por soporte
    if "co_docs" in pairs_df.columns:
        pairs_df = pairs_df[pairs_df["co_docs"] >= min_docs].copy()
    if pairs_df.empty:
        return []

    # 3) seleccionar top_n por edge_score (o co_docs si no estuviera)
    key = "edge_score" if "edge_score" in pairs_df.columns else "co_docs"
    top = pairs_df.sort_values(key, ascending=False).head(top_n)

    # 4) materializar pairbags
    out: List[Unit] = []
    for _, r in top.iterrows():
        tag_a = r["tag_a"]; tag_b = r["tag_b"]
        matched = [u for u in units if set((tag_a, tag_b)).issubset(set(u.tags))]
        if not matched:
            continue
        start = min(u.start_ts for u in matched)
        end   = max(u.end_ts   for u in matched)
        # agregamos todas las sources de las units participantes
        sources = []
        for u in matched:
            sources.extend(list(u.sources))
        # id determinístico por tags + members
        payload = {
            "type": "pairbag",
            "tags": tuple(sorted((tag_a, tag_b))),
            "members": tuple(sorted(u.unit_id for u in matched)),
            "start": start, "end": end
        }
        unit_id = "u_" + sha256_of(payload)
        out.append(Unit(
            unit_id=unit_id,
            unit_type="pairbag",
            start_ts=start,
            end_ts=end,
            tags=tuple(sorted((tag_a, tag_b))),
            topic_ids=("unknown",),
            sources=tuple(sources)
        ))
    return out

def tagbag_units_from_units(units: List[Unit],
                            top_k_tags: int = 50,
                            min_docs: int = 3) -> List[Unit]:
    """
    Crea Units 'tagbag' por tag individual con suficiente soporte.
    """
    from collections import Counter, defaultdict
    tag_docs = Counter()
    by_tag: dict[str, list[Unit]] = defaultdict(list)

    for u in units:
        for t in set(u.tags):
            tag_docs[t] += 1
            by_tag[t].append(u)

    # top-k por frecuencia
    top_tags = [t for t,_ in tag_docs.most_common(top_k_tags) if tag_docs[t] >= min_docs]
    out: List[Unit] = []
    for t in top_tags:
        matched = by_tag[t]
        start = min(u.start_ts for u in matched)
        end   = max(u.end_ts   for u in matched)
        sources = []
        for u in matched:
            sources.extend(list(u.sources))
        payload = {
            "type": "tagbag",
            "tag": t,
            "members": tuple(sorted(u.unit_id for u in matched)),
            "start": start, "end": end
        }
        unit_id = "u_" + sha256_of(payload)
        out.append(Unit(
            unit_id=unit_id,
            unit_type="tagbag",
            start_ts=start,
            end_ts=end,
            tags=(t,),
            topic_ids=("unknown",),
            sources=tuple(sources)
        ))
    return out





__all__ = [
    "co_tag_pairs",
    "compute_gates",
    "subsets",
    "pairbag_units_from_units",
    "tag_communities_from_pairs",
    "tagbag_units_from_units",

]
