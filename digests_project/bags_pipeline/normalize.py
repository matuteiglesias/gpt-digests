# normalize.py
from __future__ import annotations
import ast
import re
from typing import Any, Dict, Iterable, List, Sequence


from .textnorm import slug_value  # the one normalized slug source
from typing import Any, Dict, Iterable, List, Sequence, Set

__all__ = [
    "ALIAS_NS",
    "NS_PREFERENCE",
    "canonical_tag", "safe_canonical_tag", "lexeme",
    "parse_tags", "flatten_tag_dict",
    "unique_canonical_tags", "normalize_tags",
    "merge_tag_lists",
]




# ------------------------- namespace aliasing --------------------------------


# normalize.py (only the deltas)



# --- Namespace aliasing (single source of truth) ---
ALIAS_NS: Dict[str, Set[str]] = {
    "msg_type":     {"message_type", "msgtype", "messagetype"},
    "format_type":  {"format", "fmt", "content_type", "formattype"},
    "note_type":    {"notetype"},
    "rtype":        {"response_type", "responsetype"},
    "snippet_type": {"snippettype"},
    "topic":        {"topics"},
    "category":     set(), "domain": set(), "subtopic": set(), "medium": set(),
    "stage":        set(),
    "free":         set(),
}

def _canon_ns(ns: str) -> str:
    s = slug_value(ns)
    if s in ALIAS_NS:  # exact canonical hit
        return s
    for keep, aliases in ALIAS_NS.items():
        if s in aliases:
            return keep
    return s  # keep unknown ns visible; do not demote to 'free'

# --- Precedence used everywhere ---
NS_PREFERENCE: Sequence[str] = (
    "topic", "category", "domain", "stage",
    "msg_type", "format_type", "note_type", "rtype", "snippet_type",
    "free",
)

# NS_PREFERENCE = ("topic", "category", "domain", "stage", "msg_type", "format_type", "note_type", "rtype", "snippet_type", "free")





# ------------------------- VALUES NORMALIZE --------------------------------


# normalize_values.py (or keep in normalize.py if you prefer)

from typing import Dict
from .textnorm import slug_value




# normalize_values.py (or keep in normalize.py if you prefer)

from typing import Dict
from .textnorm import slug_value

# Canonical value tables per column; keys/values are *slugged* forms.
VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    "stage": {
        "execute": "execute", "execution": "execute", "run": "execute", "do": "execute",
        "plan": "plan", "planning": "plan", "design": "plan",
        "reflect": "reflect", "retro": "reflect", "review": "reflect", "postmortem": "reflect",
    },
    "msg_type": {
        "instruction": "instruction", "howto": "instruction", "how_to": "instruction", "spec": "instruction",
        "reflection": "reflection", "analysis": "reflection", "commentary": "reflection",
        "other": "other", "misc": "other",
    },
    "format_type": {
        "guide": "guide", "playbook": "guide", "walkthrough": "guide",
        "code": "code", "snippet": "code",
        "report": "report", "writeup": "report",
        "template": "template", "tmpl": "template", "skeleton": "template",
    },
    "note_type": {
        "insight": "insight", "aha": "insight", "learning": "insight",
    },
    "rtype": {
        "response": "response", "tool": "tool", "function": "tool", "call": "tool",
        "none": "none",
    },
    "category": {
        "software_dev": "software_dev", "software": "software_dev", "dev": "software_dev",
        "automation": "automation", "programming": "programming", "business": "business",
        "ai": "ai", "docs_and_planning": "docs_and_planning", "docs": "docs_and_planning",
    },
    "domain": {
        "python": "python", "py": "python",
        "software_dev": "software_dev",
    },
    # add similar for subtopic, medium, snippet_type, role if your data needs it
}

def normalize_value(col: str, val: object) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    key = slug_value(s)
    table = VALUE_SYNONYMS.get(col)
    if not table:
        return key  # unknown column: still return slugged value
    return table.get(key, key)


# ------------------------- SCHEMA NORMALIZE --------------------------------


# schema_normalize.py
import pandas as pd
from typing import Dict, List, Sequence, Tuple

from .textnorm import slug_value
# from .normalize import normalize_tags as norm_tags, lexeme
# from .normalize import normalize_tags as norm_tags, merge_tag_lists, NS_PREFERENCE
# from .values import normalize_value


def _pick_first(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Return the first existing column; else a NA Series of right length."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(pd.NA, index=df.index)

def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.map({True: True, False: False, "true": True, "false": False, "True": True, "False": False}).fillna(False)




# Canonical column order (you can extend this)
# CANONICAL_ORDER: List[str] = [
#     "id","conversation_id","title","ts","ts_iso","timestamp","day","week",
#     "stage","msg_type","format_type","note_type","rtype","category","domain",
#     "subtopic","medium","snippet_type","response_type","role",
#     "projectName","workspaceName","product_opportunity","leverage_potential",
#     "tags","labels","content","content_type","content_url","asset",
# ]

# Your synonym map (order matters inside each list)
SYNONYMS: Dict[str, List[str]] = {
    "id":["id","msg_id","event_id"],
    "conversation_id":["conversation_id","conv_id"],
    "title":["title","name","subject"],
    "timestamp":["timestamp","time_ms"],
    "ts_iso":["ts_iso","created_at","time_iso","datetime"],
    "ts":["ts"],
    "stage":["stage"], "msg_type":["msg_type","message_type","type"],
    "format_type":["format_type","format"], "note_type":["note_type"],
    "rtype":["rtype","response_type"], "category":["category"],
    "domain":["domain"], "subtopic":["subtopic"], "medium":["medium"],
    "snippet_type":["snippet_type"], "response_type":["response_type"],
    "role":["role"],
    "product_opportunity":["product_opportunity"],
    "leverage_potential":["leverage_potential"],
    "seeds_blog":["seeds_blog"],
    "tags":["tags","kwds","obsidian_link_targets"],
    "labels":["labels"],
    "projectName":["projectName","project","project_name"],
    "workspaceName":["workspaceName","workspace","workspace_name"],
    "content":["content","msg","text","msg_content","description","comments","comment","additional_notes"],
    "content_type":["content_type","format_type"],
    "content_url":["content_url","url","asset_pointer"],
    "asset":["asset","asset_pointer","audio_asset_pointer","video_asset_pointer"],
}



def normalize_schema(df_in: pd.DataFrame, *, include_labels_in_all_tags: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    1) Map schema synonyms → canonical columns.
    2) Normalize times (ts preferred, then timestamp(ms), then ts_iso).
    3) Normalize *values* in key categorical columns via VALUE_SYNONYMS.
    4) Normalize tags/labels via normalize.normalize_tags (dedupe by lexeme).
    5) Mint derived pairing tags from normalized categorical columns.
    6) Return normalized df and a report.
    """
    df = df_in.copy()
    report: Dict = {"columns": {}, "unmapped_source_columns": []}

    # (1) Schema mapping
    cols = {}
    for canon, syns in SYNONYMS.items():
        s = _pick_first(df, syns)
        cols[canon] = s
        src = next((n for n in syns if n in df.columns), None)
        report["columns"][canon] = {"source": src, "non_null": int(s.notna().sum())}

    out = pd.DataFrame(cols, index=df.index)

    # (2) Time normalization
    if out["ts"].isna().all():
        if out["timestamp"].notna().any():
            out["ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True, errors="coerce")
        elif out["ts_iso"].notna().any():
            out["ts"] = pd.to_datetime(out["ts_iso"], utc=True, errors="coerce")

    if out["ts_iso"].isna().all() and out["ts"].notna().any():
        # RFC3339-ish; drop tz colon if you prefer
        out["ts_iso"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Day/week convenience fields
    if out["ts"].notna().any():
        out["day"] = out["ts"].dt.date.astype("string")
        out["week"] = out["ts"].dt.to_period("W-SUN").astype(str)
    else:
        out["day"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out["week"] = pd.Series(pd.NA, index=out.index, dtype="string")

    # (3) Per-column value normalization
    cat_cols = ["stage","msg_type","format_type","note_type","rtype","category","domain",
                "subtopic","medium","snippet_type","response_type","role","projectName","workspaceName"]
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").map(lambda x: normalize_value(c, x) if not pd.isna(x) else x)

    # (4) Tags/labels normalization (dedupe by lexeme; namespaced preferred)
    if "tags" in out.columns:
        out["tags"] = out["tags"].apply(lambda x: normalize_tags(x, infer_when_empty=False))
    else:
        out["tags"] = [[]] * len(out)

    if "labels" in out.columns:
        out["labels"] = out["labels"].apply(lambda x: normalize_tags(x, infer_when_empty=False))

    # Optional booleans
    if "seeds_blog" in out.columns:
        out["seeds_blog"] = _to_bool_series(out["seeds_blog"])


    # (5) Derived pairing tags from normalized categoricals
    DERIVED_COLS = ("stage","msg_type","format_type","note_type","rtype","category","domain")
    def derive_row_tags(row) -> list[str]:
        kv = []
        for c in DERIVED_COLS:
            v = row.get(c)
            if isinstance(v, str) and v:
                kv.append(f"{slug_value(c)}:{slug_value(v)}")
        return kv

    out["derived_tags"] = out.apply(derive_row_tags, axis=1)

    # (5b) optional label merge
    base_tags = out["tags"]
    if include_labels_in_all_tags and "labels" in out.columns:
        base_tags = base_tags.combine(out["labels"], lambda a, b: (a or []) + (b or []))

    # (5c) final combine with precedence *once*
    out["all_tags"] = out.apply(
        lambda r: merge_tag_lists(base_tags.get(r.name), r.get("derived_tags"), ns_preference=NS_PREFERENCE),
        axis=1
    )

    # ✅ don’t forget to return
    return out, report

# ----------------------------- tag helpers ----------------------------------

from unicodedata import normalize as _u


def _lexeme(t: str) -> str:
    """Namespace-insensitive lexeme (used for tautology-like detection)."""
    core = str(t).split(":", 1)[-1]
    core = _u("NFKD", core).encode("ascii", "ignore").decode("ascii")
    return core.strip().lower()




def canonical_tag(tag: str) -> str:
    """
    Normalize a tag to 'ns:value'.
    - No namespace -> prefix 'free:'
    - Apply namespace alias map
    - Slugify both sides via slug_value
    """
    if not isinstance(tag, str) or not tag.strip():
        return "free:unknown"
    t = tag.strip()
    if ":" not in t:
        return f"free:{slug_value(t)}"
    ns, val = t.split(":", 1)
    return f"{_canon_ns(ns)}:{slug_value(val)}"


# # from .normalize import _canon_tag
# def canonical_tag(s: str) -> str:
#     # normalize and keep namespacing if present ("category:software_dev")
#     s = s.strip()
#     s = s.replace(" ", "_")
#     return s.lower()



def safe_canonical_tag(tag: Any) -> str:
    """
    Defensive canonicalization. Never raises, always returns 'ns:value'.
    """
    try:
        return canonical_tag(str(tag))
    except Exception:
        s = str(tag)
        if ":" in s:
            ns, val = s.split(":", 1)
            return f"{slug_value(ns)}:{slug_value(val)}"
        return f"free:{slug_value(s)}"

def lexeme(t: str) -> str:
    """
    Namespace-insensitive core token used for deduplication.
    Example: 'free:automation' and 'topic:automation' share lexeme 'automation'.
    """
    core = str(t).split(":", 1)[-1]
    return slug_value(core)




# --------------------------- parsing helpers --------------------------------
# --------------------------- parsing helpers --------------------------------
def flatten_tag_dict(d: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in (d or {}).items():
        if v is None or v == "":
            continue
        kslug = slug_value(k)
        if isinstance(v, (list, tuple, set)):
            out += [f"{kslug}:{slug_value(x)}" for x in v if x not in (None, "")]
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if vv is None or vv == "":
                    continue
                k2 = f"{kslug}.{slug_value(kk)}"
                if isinstance(vv, (list, tuple, set)):
                    out += [f"{k2}:{slug_value(x)}" for x in vv if x not in (None, "")]
                else:
                    out.append(f"{k2}:{slug_value(vv)}")
        else:
            out.append(f"{kslug}:{slug_value(v)}")
    return out

def parse_tags(x: Any) -> List[str]:
    if isinstance(x, dict):
        for k in ("tags", "kwds", "labels"):
            if isinstance(x.get(k), (list, tuple)):
                return [str(t) for t in x[k]]
        return flatten_tag_dict(x)  # NOTE: only pass dicts here if you *want* flattening
    if isinstance(x, (list, tuple, set)):
        return [str(t) for t in x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(t) for t in v]
            if isinstance(v, dict):
                for k in ("tags", "kwds", "labels"):
                    if isinstance(v.get(k), (list, tuple)):
                        return [str(t) for t in v[k]]
                return flatten_tag_dict(v)
        except Exception:
            pass
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return []

# ---------------------- canonicalization + dedupe ----------------------------


def unique_canonical_tags(
    raw_tags: Iterable[str],
    *,
    ns_preference: Sequence[str] = NS_PREFERENCE,
) -> List[str]:
    """
    Canonicalize to 'ns:value' and dedupe by lexeme.
    Preference order determines which tag survives when multiple share the same lexeme.
    - Default prefers semantic namespaces over 'free'.
    - If namespaces tie, pick the shorter representation.
    """
    by_lex: Dict[str, str] = {}
    rank = {ns: i for i, ns in enumerate(ns_preference)}

    for t in raw_tags:
        ct = safe_canonical_tag(t)
        lx = lexeme(ct)
        cur_ns = ct.split(":", 1)[0] if ":" in ct else "free"

        prev = by_lex.get(lx)
        if prev is None:
            by_lex[lx] = ct
            continue

        prev_ns = prev.split(":", 1)[0] if ":" in prev else "free"
        prev_rank = rank.get(prev_ns, len(rank))
        cur_rank  = rank.get(cur_ns, len(rank))

        if cur_rank < prev_rank:
            by_lex[lx] = ct
        elif cur_rank == prev_rank and len(ct) < len(prev):
            by_lex[lx] = ct

    return list(by_lex.values())

from .textnorm import infer_tags_from_text_like

def normalize_tags(
    x: Any,
    *,
    infer_when_empty: bool = False,
    ns_preference: Sequence[str] = NS_PREFERENCE,
) -> List[str]:
    """
    High-level helper:
    - parse raw tags from x (list/dict/str),
    - optionally infer from text-like fields if no tags,
    - canonicalize and dedupe by lexeme with namespace preference.
    """
    raw = parse_tags(x)
    if not raw and infer_when_empty:
        raw = infer_tags_from_text_like(x)
    return unique_canonical_tags(raw, ns_preference=ns_preference)



def merge_tag_lists(
    left: Iterable[str] | None,
    right: Iterable[str] | None,
    *,
    ns_preference: Sequence[str] = NS_PREFERENCE,
) -> List[str]:
    """
    Combine two tag lists that may already be canonical (e.g., derived tags)
    and/or raw (e.g., freeform user tags), then canon+dedupe once.
    """
    a = list(left or [])
    b = list(right or [])
    return unique_canonical_tags(a + b, ns_preference=ns_preference)





# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _as_mapping(u: Any) -> Dict[str, Any]:
#     """Return a dict-like view for parse_tags; works with dicts or dataclasses/objects."""
#     if isinstance(u, dict):
#         return u
#     if is_dataclass(u):
#         return asdict(u)
#     # Fallback: shallow attribute scrape
#     out = {}
#     for k in ("unit_id", "unit_type", "start_ts", "end_ts", "tags", "topic_ids", "sources", "extras"):
#         if hasattr(u, k):
#             out[k] = getattr(u, k)
#     return out

# def _unit_id(u: Any) -> str | None:
#     if isinstance(u, dict):
#         return u.get("unit_id") or u.get("id") or u.get("uid")
#     return getattr(u, "unit_id", None)

# def _canon_unique_tags(raw_tags: Iterable[str]) -> list[str]:
#     """
#     Canonicalize to ns:value and dedupe by lexeme.
#     Preference: keep a namespaced tag over 'free:*' when both share the same lexeme.
#     """
#     chosen_by_lex: dict[str, str] = {}
#     for t in raw_tags:
#         ct = canonical_tag(t)  # uses your ALIAS_NS and slugging
#         lx = lexeme(ct)
#         prev = chosen_by_lex.get(lx)
#         if prev is None:
#             chosen_by_lex[lx] = ct
#         else:
#             # prefer a namespaced (non-free) tag over free:*, else the shorter token
#             prev_ns, _ = prev.split(":", 1) if ":" in prev else ("free", prev)
#             cur_ns, _  = ct.split(":", 1) if ":" in ct  else ("free", ct)
#             if prev_ns == "free" and cur_ns != "free":
#                 chosen_by_lex[lx] = ct
#             elif cur_ns != "free" and len(ct) < len(prev):
#                 chosen_by_lex[lx] = ct
#     return list(chosen_by_lex.values())



# def _canon_ns(ns: str) -> str:
#     ns = slug_value(ns)
#     for keep, drops in ALIAS_NS.items():
#         if ns == keep or ns in drops:
#             return keep
#     return ns

# import re
# from .core import _get

# HASHTAG_RE = re.compile(r'(?:^|\s)#([a-z0-9_]+)', re.IGNORECASE)
# TEXT_FIELDS = ("text", "content", "summary", "message", "body", "title")

# def _infer_tags_from_text_like(ev: Any) -> List[str]:
#     tags: list[str] = []
#     # scan common text fields (attr, mapping, extras)
#     for f in TEXT_FIELDS:
#         v = _get(ev, f)
#         if isinstance(v, str):
#             tags.extend(HASHTAG_RE.findall(v))
#     return [canonical_tag(t) for t in tags]



# # ----------------------------- tag helpers -----------------------------------

# def canonical_tag(tag: str) -> str:
#     """
#     Normalize a tag to 'ns:value'.
#     - No namespace → prefix 'free:'
#     - Apply namespace aliases
#     - Slugify both sides with slug_value
#     """
#     if not isinstance(tag, str) or not tag.strip():
#         return "free:unknown"
#     t = tag.strip()
#     if ":" not in t:
#         return f"free:{slug_value(t)}"
#     ns, val = t.split(":", 1)
#     return f"{_canon_ns(ns)}:{slug_value(val)}"

# def lexeme(t: str) -> str:
#     """Namespace-insensitive token (for tautology-like detection)."""
#     core = str(t).split(":", 1)[-1]
#     return slug_value(core)

# def _canon(t: str) -> str:
#     """Safe wrapper; falls back to local slugging if something goes wrong."""
#     try:
#         return canonical_tag(t)
#     except Exception:
#         if ":" in str(t):
#             ns, val = str(t).split(":", 1)
#             return f"{slug_value(ns)}:{slug_value(val)}"
#         return f"free:{slug_value(t)}"

# # --------------------------- parsing helpers ---------------------------------

# def flatten_tag_dict(d: dict) -> list[str]:
#     """
#     Flatten nested dicts/lists into 'ns:value' tokens.
#     - list values -> multiple tags
#     - dict values -> 'k.sub:v' convention
#     """
#     out: list[str] = []
#     for k, v in (d or {}).items():
#         if v is None or v == "":
#             continue
#         kslug = slug_value(k)
#         if isinstance(v, (list, tuple, set)):
#             out += [f"{kslug}:{slug_value(x)}" for x in v if x not in (None, "")]
#         elif isinstance(v, dict):
#             for kk, vv in v.items():
#                 if vv is None or vv == "":
#                     continue
#                 k2 = f"{kslug}.{slug_value(kk)}"
#                 if isinstance(vv, (list, tuple, set)):
#                     out += [f"{k2}:{slug_value(x)}" for x in vv if x not in (None, "")]
#                 else:
#                     out.append(f"{k2}:{slug_value(vv)}")
#         else:
#             out.append(f"{kslug}:{slug_value(v)}")
#     return out

# def parse_tags(x: Any) -> list[str]:
#     """
#     Accepts:
#       - list of strings
#       - dicts like {'tags':[...]} or any k:v to flatten
#       - stringified list/dict or comma-separated string
#     Returns a list of raw tag strings (not yet canonicalized).
#     """
#     if isinstance(x, dict):
#         for k in ("tags", "kwds", "labels"):
#             if isinstance(x.get(k), (list, tuple)):
#                 return list(x[k])
#         return flatten_tag_dict(x)

#     if isinstance(x, list):
#         return [str(t) for t in x]

#     if isinstance(x, str):
#         s = x.strip()
#         if not s:
#             return []
#         try:
#             v = ast.literal_eval(s)
#             if isinstance(v, (list, tuple)):
#                 return [str(t) for t in v]
#             if isinstance(v, dict):
#                 for k in ("tags", "kwds", "labels"):
#                     if isinstance(v.get(k), (list, tuple)):
#                         return [str(t) for t in v[k]]
#                 return flatten_tag_dict(v)
#         except Exception:
#             pass
#         if "," in s:
#             return [t.strip() for t in s.split(",") if t.strip()]
#         return [s]

#     return []



# # # ------------------------------- tag helpers ---------------------------------

# # def _slug_value(s: str) -> str:
# #     s = _u("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
# #     return s.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")








# def add_meta_tags(df: pd.DataFrame) -> pd.DataFrame:
#     """Project selected categorical columns into the 'tags' list using canonical_tag()."""
#     df = df.copy()
#     # ensure tags column
#     if "tags" not in df.columns or df["tags"].isna().all():
#         df["tags"] = [[] for _ in range(len(df))]
#     else:
#         # make sure it's list-like
#         df["tags"] = df["tags"].apply(lambda v: v if isinstance(v, (list, tuple)) else ([] if pd.isna(v) else [v]))
#     # columns to project as tags (add/remove to taste)
#     cols = ["stage","msg_type","format_type","category","domain","note_type","snippet_type","medium","rtype"]
#     for c in cols:
#         if c in df.columns:
#             df[c] = df[c].astype(str)
#             df["tags"] = df.apply(lambda r: r["tags"] + [canonical_tag(f"{c}:{r[c]}")] if r[c] and r[c] != "nan" else r["tags"], axis=1)
#     # drop deprecated dup columns if you want it *clean* immediately
#     if "response_type" in df.columns and "rtype" in df.columns:
#         df = df.drop(columns=["response_type"])
#     # canonicalize any preexisting tag strings and de-duplicate
#     df["tags"] = df["tags"].apply(lambda xs: sorted({canonical_tag(x) for x in xs if isinstance(x, str) and ":" in x}))
#     return df





# # Inyector de meta-tags sobre columnas canónicas
# def add_meta_tags(df: pd.DataFrame, fields = ("stage","msg_type","format_type","note_type",
#                                              "rtype","category","domain","subtopic",
#                                              "medium","snippet_type","response_type","role")) -> pd.DataFrame:
#     if "tags" not in df.columns:
#         df["tags"] = [[]]*len(df)
#     def _row_tags(r):
#         base = list(r.get("tags") or [])
#         extra = []
#         for f in fields:
#             if f in df.columns:
#                 v = r.get(f)
#                 if pd.isna(v) or v in (None, ""): 
#                     continue
#                 if isinstance(v, (list, tuple, set)):
#                     extra += [f"{f}:{_slug_value(x)}" for x in v]
#                 else:
#                     extra.append(f"{f}:{_slug_value(v)}")
#         # canonizá todo
#         return sorted({canonical_tag(t) for t in (base + extra)})
#     df = df.copy()
#     df["tags"] = df.apply(_row_tags, axis=1)
#     return df

