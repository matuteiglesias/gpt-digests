from __future__ import annotations

# stdlib

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# project config & core
from .config import (
    MODEL_VER,
    POLICY_VER,
    PROMPT_VER,
    parse_utc_any,
    sha256_of,
)
from .core import Unit, L2Digest

# I/O helpers
from .io import (
    ensure_dir,
    read_json,
    read_jsonl,
    read_mdx_front_matter,
    write_json,
    write_text,
)

# indexing
from .index import build_indices
from .textnorm import _copy_unit, in_window_range
from .core import slug_for_unit


from dataclasses import dataclass

from .hydrate import (
    materialize_bag_markdown,
    _render_mdx,
)

# misc text/time utilities

Pathish = Union[str, Path]



def make_l2_id(channel:str, u:Unit, prompt_ver=PROMPT_VER, model_ver=MODEL_VER, policy_ver=POLICY_VER) -> str:
    payload = {"L2":channel,"unit_id":u.unit_id,"prompt_ver":prompt_ver,"model_ver":model_ver,"policy_ver":policy_ver}
    return "d_" + sha256_of(payload)


def trim_unit_sources_for_window(
    u,
    ev_idx: Optional[Dict[str, Dict[str, Any]]],
    ss_idx: Optional[Dict[str, Dict[str, Any]]],
    since_iso: str|None,
    until_iso: str|None,
):
    """Return a *view* of `u` with sources trimmed to [since,until) and window rewritten."""
    if not (since_iso or until_iso):
        return u

    since_dt = parse_utc_any(since_iso) if since_iso else None
    until_dt = parse_utc_any(until_iso) if until_iso else None

    def _in_ts(ts_raw) -> bool:
        dt = parse_utc_any(ts_raw)
        if dt is None:
            return False
        if since_dt and dt < since_dt:
            return False
        if until_dt and dt >= until_dt:
            return False
        return True

    def _session_overlaps(ss: Dict[str, Any]) -> bool:
        s_ts = ss.get("ts_start") or ss.get("start_ts") or ss.get("ts") or (ss.get("summary", {}) or {}).get("ts")
        e_ts = ss.get("ts_end")   or ss.get("end_ts")   or s_ts
        return in_window_range(s_ts, e_ts, since_iso, until_iso)

    keep: list[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    for kind, sid in getattr(u, "sources", ()) or ():
        key = (kind, sid)
        if key in seen:
            continue
        seen.add(key)

        if kind == "event" and ev_idx:
            ev = ev_idx.get(sid)
            if not ev:
                continue
            ts = ev.get("ts_abs") or ev.get("ts") or ev.get("timestamp") or ev.get("created_at")
            if _in_ts(ts):
                keep.append(key)

        elif kind == "session" and ss_idx:
            ss_key = sid if sid in ss_idx else ("cluster_" + sid[2:] if isinstance(sid, str) and sid.startswith("s_") else sid)
            ss = ss_idx.get(ss_key)
            if ss and _session_overlaps(ss):
                keep.append(key)

    # rewrite the window to the requested slice (so downstream doesn’t re-expand)
    return _copy_unit(u,
                      start_ts=(since_iso or getattr(u, "start_ts", "")),
                      end_ts=(until_iso or getattr(u, "end_ts", "")),
                      sources=tuple(keep))





def score_unit(u: Unit, history=None, centrality_weight=0.2, topic_weight=1.0) -> Dict[str,float]:
    # placeholder; replace with your metrics
    coverage = 0.7
    novelty = 0.5
    centrality = centrality_weight
    tweight = topic_weight
    salience = coverage * novelty * centrality * tweight
    return {"coverage":coverage,"novelty":novelty,"centrality":centrality,"topic_weight":tweight,"salience":salience}



def load_manifest_rows(root: Path) -> List[Dict[str, Any]]:
    """
    Walk `root` and return a list of {"id", "channel", ..., "path"} dicts
    for every digest in tree, flat or onefile layout.
    """
    rows: List[Dict[str, Any]] = []

    # tree
    for mf in root.rglob("manifest.json"):
        dg = read_json(mf).get("digest")
        if not isinstance(dg, dict):
            continue
        rows.append({
            "digest_id": dg["id"],
            "channel":   dg["channel"],
            "unit_type": dg["unit_type"],
            "topic_ids": dg.get("topic_ids", []),
            "tags":      dg.get("tags", []),
            "scores":    dg.get("scores", {}),
            "start_ts":  dg.get("start_ts"),
            "end_ts":    dg.get("end_ts"),
            "path":      str(mf.parent / "digest.mdx"),
        })

    # flat
    for js in root.rglob("*.json"):
        if js.name == "manifest.json":
            continue
        d = read_json(js).get("digest", {})
        mdx = js.with_suffix(".mdx")
        if not mdx.exists():
            continue
        rows.append({
            "digest_id": d.get("id"),
            "channel":   d.get("channel"),
            "unit_type": d.get("unit_type"),
            "topic_ids": d.get("topic_ids", []),
            "tags":      d.get("tags", []),
            "scores":    d.get("scores", {}),
            "start_ts":  d.get("start_ts"),
            "end_ts":    d.get("end_ts"),
            "path":      str(mdx),
        })

    # onefile
    for mdx in root.rglob("*.mdx"):
        if any(r["path"] == str(mdx) for r in rows):
            continue
        fm = read_mdx_front_matter(mdx)
        if not fm:
            continue
        rows.append({
            "digest_id": fm.get("id"),
            "channel":   fm.get("channel"),
            "unit_type": fm.get("unit_type"),
            "topic_ids": fm.get("topic_ids", []),
            "tags":      fm.get("tags", []),
            "scores":    fm.get("scores", {}),
            "start_ts":  fm.get("start_ts"),
            "end_ts":    fm.get("end_ts"),
            "path":      str(mdx),
        })

    return rows


def index_l2(
    root: Path,
    since: Optional[str],
    until: Optional[str],
    out_json: Path,
) -> None:
    """
    Load all manifest rows under `root`, optionally filter by time window,
    then write them to `out_json`.
    """
    rows = load_manifest_rows(root)
    if since or until:
        rows = [
            r for r in rows
            if in_window_range(r["start_ts"], r["end_ts"], since, until)
        ]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, rows, ensure_ascii=False, indent=2)




@dataclass
class L2Digest:
    digest_id: str
    unit_id:    str
    # channel:    str
    unit_type:  str
    level:      int
    # title:      str
    start_ts:   str
    end_ts:     str
    topic_ids:  Tuple[str, ...]
    tags:       Tuple[str, ...]
    scores:     Dict[str, float]
    policy:     str
    # add other fields here if needed


def build_L2(
    units: List[Unit],
    channels: List[str],
    *,
    level: int = 2,
    policy: str = "draft",
) -> List[L2Digest]:
    """
    From a flat list of Units, produce an L2Digest per unit whose
    unit.channel is in `channels`.  Computes a digest_id, runs your
    scoring function, and renders a short title/preview snippet.
    """
    digests: List[L2Digest] = []

    for u in units:
        # skip units on channels we're not building
        # if u.channel not in channels:
        #     continue

        # 1) Compute a stable id for this digest
        digest_id = "l2_" + sha256_of({
            "unit_id": u.unit_id,
            "unit_type": u.unit_type,
        })

        # 2) Score the unit however you like (stubbed here)
        #    Replace with your real scoring function:
        scores: Dict[str, float] = getattr(u, "scores", {}) or {}

        # 3) Build a human‐readable title or snippet
        #    You can swap in your real renderer:
        # try:
        #     title = render_snippet(u.channel, u)
        # except Exception:
        #     title = u.title or slug_for_unit(u)


    # title = ev.get("title") or ev.get("summary") or ""

        # 4) Gather metadata into the digest
        d = L2Digest(
            digest_id = digest_id,
            unit_id    = u.unit_id,
            # channel    = u.channel,
            unit_type  = u.unit_type,
            level      = level,
            # title      = title,
            start_ts   = u.start_ts,
            end_ts     = u.end_ts,
            topic_ids  = tuple(u.topic_ids),
            tags       = tuple(u.tags),
            scores     = scores,
            policy     = policy,
        )
        digests.append(d)

    return digests




def build_l2_digests(
    units_path: Path,
    channels: List[str],
    hydrate: bool,
    logs_globs: List[str],
    sess_globs: List[str],
    since: Optional[str],
    until: Optional[str],
    render_mode: str,
    max_items: int,
) -> List[Tuple[L2Digest, Unit, Optional[str]]]:
    """
    Returns a list of tuples (digest, corresponding Unit, optional hydrated body_md).
    """
    # 1) load & index units
    raw = read_jsonl(units_path)                # yields list[dict]
    all_units = [Unit(**u) for u in raw]
    units_by_id = {u.unit_id: u for u in all_units}

    # 2) build skeleton digests
    digests: List[L2Digest] = build_L2(all_units, channels)

    # 3) prepare optional indices
    if hydrate:
        ev_idx, ss_idx = build_indices(logs_globs, sess_globs)
    else:
        None

    out: List[Tuple[L2Digest, Unit, Optional[str]]] = []
    for d in digests:
        u = units_by_id[d.unit_id]

        # a) time-slice trim if requested
        if hydrate and (since or until):
            u = trim_unit_sources_for_window(u, ev_idx, ss_idx, since, until)
            d.start_ts, d.end_ts = u.start_ts, u.end_ts

        # b) optional hydration → full MD
        body_md: Optional[str] = None
        if hydrate:
            body_md = materialize_bag_markdown(
                [u],
                ev_idx, ss_idx,
                collapse=True,
                max_items=max_items,
                since_iso=since,
                until_iso=until,
                render_mode=render_mode,
            )

        out.append((d, u, body_md))

    return out



def write_l2_all(
    drafts: List[Tuple[L2Digest, Unit, Optional[str]]],
    out_base: Path,
    layout: str,
    filename_scheme: str,
    filename_suffix: str = "",
) -> int:
    """
    Writes each (digest,unit,body_md) to disk under out_base.
    Returns the number of files written.
    """
    ensure_dir(out_base)
    count = 0
    for digest, unit, body_md in drafts:
        write_l2(
            digest=digest,
            base=out_base,
            layout=layout,
            filename_scheme=filename_scheme,
            body_override_md=body_md,
            filename_suffix=filename_suffix,
        )
        count += 1
    return count



# from .l2 import L2Digest


import re

def write_mdx_file(path, body_content):
    path.parent.mkdir(exist_ok=True, parents=True)
    # ensure all <br> are self-closed for JSX/MDX compatibility
    body_fixed = body_content.replace('<br>', '<br/>')
    write_text(path, body_fixed)


# def write_text(path: Path, text: str, encoding="utf-8"):
#     path.parent.mkdir(exist_ok=True, parents=True)
#     path.write_text(text, encoding=encoding)

def write_l2(
    digest: L2Digest,
    base: Pathish,
    layout: str = "onefile",
    filename_scheme: str = "slug",
    body_override_md: Optional[str] = None,
    filename_suffix: str = "",
) -> Path:
    """
    Core writer: dumps one L2Digest to disk in tree|flat|onefile layout,
    using IO helpers for all file operations.
    """
    base = Path(base)
    # ch = re.sub(r"[^a-z0-9_\-]+", "_", digest.channel.lower())
    ut = re.sub(r"[^a-z0-9_\-]+", "_", digest.unit_type.lower())
    day = digest.start_ts[:10] or "0000-00-00"
    name = digest.digest_id if filename_scheme == "id" else slug_for_unit(digest)

    # choose body
    if body_override_md is not None:
        body = body_override_md
    else:
        body = (
            digest.body_md
            or digest.markdown
            or getattr(digest.content, "body_md", None)
            or getattr(digest.content, "markdown", None)
            or digest.mdx
            or ""
        )

    if layout == "tree":
        dest = base / ut / day / name
        # body file
        write_mdx_file(dest / f"{name}{filename_suffix}.mdx", body)
        # manifest
        write_json(
            dest / "manifest.json",
            {"digest": _digest_manifest(digest)},
            ensure_ascii=False,
            indent=2,
        )
        return dest

    if layout == "flat":
        dest = base / ut 
        write_mdx_file(dest / f"{name}{filename_suffix}.mdx", body)
        write_json(
            dest / f"{name}.json",
            {"digest": _digest_manifest(digest)},
            ensure_ascii=False,
            indent=2,
        )
        return dest / f"{name}{filename_suffix}.mdx"

    # onefile: front‐matter + body in single file
    if layout == "onefile":
        # dest = base / ut / ch
        dest = base / ut 
        # build front‐matter dict
        fm = _digest_manifest(digest)
        # write file
        write_mdx_file(dest / f"{name}{filename_suffix}.mdx", _render_mdx(fm, body))
        return dest / f"{name}{filename_suffix}.mdx"

    raise ValueError(f"unknown layout: {layout}")


def _digest_manifest(d: L2Digest) -> dict:
    return {
        "id": d.digest_id,
        "level": d.level,
        # "channel": d.channel,
        "unit_id": d.unit_id,
        "unit_type": d.unit_type,
        # "title": d.title,
        "start_ts": d.start_ts,
        "end_ts": d.end_ts,
        "topic_ids": list(d.topic_ids),
        "tags": list(d.tags),
        "scores": d.scores,
        "policy": d.policy,
    }






# # firma única y definitiva
# def write_l2(
#         digest: "L2Digest",
#         base: Pathish,
#         layout: str = "onefile",
#         filename_scheme: str = "slug",
#         body_override_md: Optional[str] = None,
#         filename_suffix: str = "",                   # ← NEW
#     ) -> Path:
#     """
#     Write a digest to disk.

#     If ``body_override_md`` is provided, it becomes the body; otherwise the
#     function falls back to any body/markdown fields on the digest.

#     Layout:
#         - "onefile" (front matter + body in a single .mdx)
#         - "flat"    (body only + sidecar JSON manifest)
#         - "tree"    (directory with manifest.json and digest.mdx)

#     Args:
#         digest: L2Digest-like object (attrs or dict), containing metadata and optional mdx.
#         base: Path-like base directory.
#         layout: "tree"|"flat"|"onefile".
#         filename_scheme: "slug" or "id".
#         body_override_md: If given, write this as the body; otherwise use digest.mdx or body fields.

#     Returns:
#         The output path (Path) or directory Path for "tree".

#     """

#     def _norm_channel(ch: str) -> str:
#         return re.sub(r"[^a-z0-9_\-]+", "_", (ch or "").strip().lower())

#     ch = _norm_channel(getattr(digest, "channel", "journal"))
#     unit_type = _norm_channel(getattr(digest, "unit_type", "unit"))
#     day = getattr(digest, "start_ts", "")[:10] or "0000-00-00"
#     name = digest.digest_id if filename_scheme == "id" else slug_for_unit(digest)


#     # A tiny helper to choose the body consistently
#     def _choose_body(d):
#         if body_override_md is not None:
#             return body_override_md
#         return (
#             getattr(d, "body_md", None)
#             or getattr(d, "markdown", None)
#             or (getattr(d, "content", {}) or {}).get("body_md")
#             or (getattr(d, "content", {}) or {}).get("markdown")
#             or getattr(d, "mdx", "")
#         )



#     if layout == "tree":
#         outdir = base / "L2" / unit_type / day / ch / name
#         outdir.mkdir(parents=True, exist_ok=True)
#         (outdir / f"{name}{filename_suffix}.mdx"
#          ).write_text(_choose_body(digest), encoding="utf-8")
        
#         # (outdir / "digest.mdx").write_text(digest.mdx, encoding="utf-8")
#         # write_json(outdir/"manifest.json", {"digest": {...}}, ensure_ascii=False, indent=2)
#         (outdir / "manifest.json").write_text(json.dumps({
#             "digest": {
#                 "id": digest.digest_id,
#                 "level": digest.level,
#                 "channel": digest.channel,
#                 "unit_id": digest.unit_id,
#                 "unit_type": digest.unit_type,
#                 "title": digest.title,
#                 "start_ts": digest.start_ts,
#                 "end_ts": digest.end_ts,
#                 "topic_ids": list(digest.topic_ids),
#                 "tags": list(digest.tags),
#                 "scores": digest.scores,
#                 "policy": digest.policy,
#             }
#         }, ensure_ascii=False, indent=2), encoding="utf-8")
#         (outdir / "digest.mdx").write_text(_choose_body(digest), encoding="utf-8")
#         return outdir
    

#     elif layout == "flat":
#         outdir = base / "L2" / unit_type / ch
#         outdir.mkdir(parents=True, exist_ok=True)
#         (outdir / f"{name}{filename_suffix}.mdx"
#          ).write_text(_choose_body(digest), encoding="utf-8")
#         (outdir / f"{name}.json").write_text(json.dumps({
#             "digest": {
#                 "id": digest.digest_id,
#                 "level": digest.level,
#                 "channel": digest.channel,
#                 "unit_id": digest.unit_id,
#                 "unit_type": digest.unit_type,
#                 "title": digest.title,
#                 "start_ts": digest.start_ts,
#                 "end_ts": digest.end_ts,
#                 "topic_ids": list(digest.topic_ids),
#                 "tags": list(digest.tags),
#                 "scores": digest.scores,
#                 "policy": digest.policy,
#             }
#         }, ensure_ascii=False, indent=2), encoding="utf-8")
#         return outdir / f"{name}.mdx"

#     if layout == "onefile":
#         outdir = base / "L2" / unit_type / ch
#         outdir.mkdir(parents=True, exist_ok=True)

#         front = {
#             "id": digest.digest_id,
#             "level": digest.level,
#             "channel": digest.channel,
#             "unit_id": digest.unit_id,
#             "unit_type": digest.unit_type,
#             "title": digest.title,
#             "start_ts": digest.start_ts,
#             "end_ts": digest.end_ts,
#             "topic_ids": list(digest.topic_ids),
#             "tags": list(digest.tags),
#             "scores": digest.scores,
#             "policy": digest.policy,
#         }
#         try:
#             import yaml
#             front_matter_text = f"---\n{yaml.safe_dump(front, sort_keys=False)}---\n"
#         except Exception:
#             fm = json.dumps({"front_matter": front}, ensure_ascii=False, indent=2)
#             front_matter_text = f"<!-- {fm} -->\n"

#     # def render_front_matter(obj: dict) -> str:
#     #     try:
#     #         import yaml
#     #         return f"---\n{yaml.safe_dump(obj, sort_keys=False)}---\n"
#     #     except ImportError:
#     #         return f"<!-- {json.dumps(obj)} -->\n"



#         # choose body
#         body = _choose_body(digest)
#         path = outdir / f"{name}{filename_suffix}.mdx"
#         with path.open("w", encoding="utf-8") as f:
#             f.write(front_matter_text)
#             if not front_matter_text.endswith("\n"):
#                 f.write("\n")
#             if body:
#                 f.write("\n" + body.lstrip())
#         return path


#     else:
#         raise ValueError(f"layout desconocido: {layout}")

# def digest_manifest(d: L2Digest) -> dict:
#     return {
#       "id": d.digest_id,
#       "level": d.level,
#       ...
#     }


# (manifest_path).write_text(
#     json.dumps({"digest": digest_manifest(d)}, indent=2),
#     encoding="utf-8"
# )


