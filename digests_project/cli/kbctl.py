# digests_project/cli/kbctl.py
from __future__ import annotations

# import json
import re
from dataclasses import replace
# from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional
from digests_project.bags_pipeline.l2 import trim_unit_sources_for_window
from digests_project.bags_pipeline.textnorm import in_window_range
from digests_project.bags_pipeline.ingest_logs import load_events_from_logs
from digests_project.bags_pipeline.ingest_sessions import load_sessions
from digests_project.bags_pipeline.index import build_indices
from digests_project.bags_pipeline.unitize import cohort_units_from_logs, units_from_sessions, sessions_to_units_window
from digests_project.bags_pipeline.pairs import pairbag_units_from_units
from digests_project.bags_pipeline.pairs import tagbag_units_from_units  # or import from .tagbags
from digests_project.bags_pipeline.tag_select import UnitSelector
from digests_project.bags_pipeline.hydrate import materialize_bag_markdown
from digests_project.bags_pipeline.config import parse_utc_any
from digests_project.bags_pipeline.io import (
    read_jsonl,
    write_jsonl,
    write_json,
    read_json,
    read_mdx_front_matter,
    iter_jsonl,
    write_csv,
)

from digests_project.bags_pipeline.core import RenderMode

import typer

from digests_project.bags_pipeline.config import TZ_LOCAL



app = typer.Typer(add_completion=False, no_args_is_help=True)



# # ------------------------------------------------------------------------------
# # bags-logs → cohort units
# # ------------------------------------------------------------------------------
# @app.command("bags-logs")
# def bags_logs(
#     logs_glob: List[str] = typer.Option(..., help="JSONL glob(s)"),
#     out: Path            = typer.Option(Path("runs/units_logs.jsonl")),
#     since: str           = typer.Option("", help="UTC ISO start (inclusive)"),
#     until: str           = typer.Option("", help="UTC ISO end (exclusive)"),
#     group_by: str        = typer.Option("day", help="day|week|month|session"),
#     combo_size: int      = typer.Option(2, help="0=no pairs, 1=single, 2=pairs"),
#     # min-events: int
# ):
#     """Build cohort units from logs."""
#     # 1) load & optional slice
#     # ev_idx = build_event_index(logs_glob)  # builds index and ._meta
#     events = load_events_from_logs(logs_glob)
#     if since or until:
#         t0, t1 = parse_utc_any(since), parse_utc_any(until)
#         events = [
#             e for e in events
#             if (not t0 or parse_utc_any(e.ts_abs) >= t0)
#             and (not t1 or parse_utc_any(e.ts_abs) <  t1)
#         ]

#     # 2) build cohorts
#     units = cohort_units_from_logs(
#         events,
#         group_by=group_by,
#         combo_size=combo_size,
#     )

#     # 3) write out
#     write_jsonl(out, (u.__dict__ for u in units))
#     typer.echo(f"[bags-logs] wrote {len(units)} units → {out}")

from digests_project.bags_pipeline.ingest_logs import (
    write_log_cohorts,
    build_log_cohorts,  # optional: for --dry-run previews
)

# ------------------------------------------------------------------------------
# bags-logs → cohort units  (new, consolidated)
# ------------------------------------------------------------------------------
@app.command("bags-logs")
def bags_logs(
    logs_glob: List[str]  = typer.Option([], help="0+ JSONL globs for logs"),
    out: Path            = typer.Option(Path("runs/units_logs.jsonl"), help="Output JSONL of Units"),
    since: str           = typer.Option("", help="UTC ISO start (inclusive), e.g. 2025-05-01T00:00:00Z"),
    until: str           = typer.Option("", help="UTC ISO end (exclusive), e.g. 2025-09-01T00:00:00Z"),
    group_by: str        = typer.Option("day", help="day|week|month|session"),
    combo_size: int      = typer.Option(2, help="0=no pair bundling; 1=single; 2=pairs of adjacent groups"),
    min_events: int      = typer.Option(4, help="Min events per group to form a Unit"),
    top_k_tags: int      = typer.Option(30, help="Keep top-K tags per Unit"),
    dry_run: bool        = typer.Option(False, help="Build but do not write (shows counts)"),
):
    """
    Build cohort Units from logs.
    Notes:
      - Raw lines with empty content/text/summary/title are ignored upstream.
      - Time slicing is inclusive since, exclusive until.
      - 'combo_size=2' bundles adjacent day/week buckets into pairs.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    # Normalize empty strings → None so ingestion handles slicing internally.
    sin = since or None
    unt = until or None

    if dry_run:
        units = build_log_cohorts(
            log_globs=logs_glob,
            since=sin, until=unt,
            group_by=group_by,
            combo_size=combo_size,
            min_events=min_events,
            top_k_tags=top_k_tags,
        )
        typer.echo(json.dumps({
            "units": len(units),
            "params": {
                "group_by": group_by,
                "combo_size": combo_size,
                "min_events": min_events,
                "top_k_tags": top_k_tags,
                "since": sin, "until": unt,
            },
            "out": str(out),
        }, indent=2))
        return

    n = write_log_cohorts(
        log_globs=logs_glob,
        out_path=out,
        since=sin,
        until=unt,
        group_by=group_by,
        combo_size=combo_size,
        min_events=min_events,
        top_k_tags=top_k_tags,
    )
    typer.echo(f"[bags-logs] wrote {n} units → {out}")



# ------------------------------------------------------------------------------
# bags-sessions → session‐based units
# ------------------------------------------------------------------------------
@app.command("bags-sessions")
def bags_sessions(
    sess_glob: List[str] = typer.Option(..., help="Session JSONL glob(s)"),
    out: Path            = typer.Option(Path("runs/units_sessions.jsonl")),
    since: str           = typer.Option("", help="UTC ISO start"),
    until: str           = typer.Option("", help="UTC ISO end"),
):
    """Build session units."""
    sessions = load_sessions(sess_glob)
    if since or until:
        t0, t1 = parse_utc_any(since), parse_utc_any(until)
        sessions = [
            s for s in sessions
            if (not t0 or parse_utc_any(s.start_ts) >= t0)
            and (not t1 or parse_utc_any(s.end_ts)   <  t1)
        ]

    units = units_from_sessions(sessions)
    write_jsonl(out, (u.__dict__ for u in units))
    typer.echo(f"[bags-sessions] wrote {len(units)} units → {out}")



# ------------------------------------------------------------------------------
# bags-pairs-from-units → pairbag units
# ------------------------------------------------------------------------------
@app.command("bags-pairs-from-units")
def bags_pairs_from_units(
    units_jsonl: Path    = typer.Option(...),
    out: Path            = typer.Option(Path("runs/units_pairs.jsonl")),
    top_n: int           = typer.Option(50),
    min_docs: int        = typer.Option(2),
    pairs_csv: Optional[Path] = typer.Option(None, help="Optional tag‐pair CSV"),
):
    """Build pairbag units from an existing units JSONL."""
    raw = read_jsonl(units_jsonl)
    units = [u if hasattr(u, "unit_id") else __import__("bags_pipeline").core.Unit(**u) for u in raw]
    pairbags = pairbag_units_from_units(
        units,
        pairs_df=(None if not pairs_csv or not pairs_csv.exists() else __import__("pandas").read_csv(pairs_csv)),
        top_n=top_n,
        min_docs=min_docs,
    )
    write_jsonl(out, (u.__dict__ for u in pairbags))
    typer.echo(f"[bags-pairs] wrote {len(pairbags)} pairbags → {out}")


# ------------------------------------------------------------------------------
# units-select → filter an existing units JSONL
# ------------------------------------------------------------------------------
@app.command("units-select")
def units_select(
    units_jsonl: Path  = typer.Option(...),
    types: str         = typer.Option("", help="csv of unit_types"),
    tags_all: str      = typer.Option("", help="csv of tags all must have"),
    tags_any: str      = typer.Option("", help="csv of tags any may have"),
    since: str         = typer.Option("", help="UTC ISO start"),
    until: str         = typer.Option("", help="UTC ISO end"),
    out: Path          = typer.Option(Path("runs/units_selected.jsonl")),
):
    """Select a subset of units by type, tags, and time window."""
    raw = read_jsonl(units_jsonl)
    units = [__import__("bags_pipeline").core.Unit(**d) for d in raw]
    selector = UnitSelector(
        types=[t for t in types.split(",") if t],
        tags_all=[t for t in tags_all.split(",") if t],
        tags_any=[t for t in tags_any.split(",") if t],
        since=since or None,
        until=until or None,
    )
    chosen = selector.select(units)
    write_jsonl(out, (u.__dict__ for u in chosen))
    typer.echo(f"[units-select] {len(chosen)} units → {out}")


# ------------------------------------------------------------------------------
# bags-tags-from-units → tagbag units
# ------------------------------------------------------------------------------
@app.command("bags-tags-from-units")
def bags_tags_from_units(
    units_jsonl: Path    = typer.Option(...),
    out: Path            = typer.Option(Path("runs/units_tags.jsonl")),
    top_k_tags: int      = typer.Option(50),
    min_docs: int        = typer.Option(3),
):
    """Build tagbag units from an existing units JSONL."""
    raw = read_jsonl(units_jsonl)
    units = [__import__("bags_pipeline").core.Unit(
        unit_id=d["unit_id"],
        unit_type=d["unit_type"],
        start_ts=d["start_ts"],
        end_ts=d["end_ts"],
        tags=tuple(d.get("tags",[])),
        topic_ids=tuple(d.get("topic_ids",[])),
        sources=tuple(tuple(x) for x in d.get("sources",[]))
    ) for d in raw]
    tagbags = tagbag_units_from_units(units, top_k_tags=top_k_tags, min_docs=min_docs)
    write_jsonl(out, (u.__dict__ for u in tagbags))
    typer.echo(f"[bags-tags] {len(tagbags)} tagbags → {out}")


# ------------------------------------------------------------------------------
# bag-md → render a single‐unit “bag” to Markdown
# ------------------------------------------------------------------------------
@app.command("bag-md")
def bag_md(
    units_jsonl: Path    = typer.Option(...),
    logs_glob: List[str] = typer.Option([], help="logs JSONL glob(s)"),
    sess_glob: List[str] = typer.Option([], help="session JSONL glob(s)"),
    out_md: Path         = typer.Option(Path("runs/bag.md")),
):
    """Render one‐unit bag Markdown for inspection."""
    raw = read_jsonl(units_jsonl)
    units = [__import__("bags_pipeline").core.Unit(**d) for d in raw]


    ev_idx, ss_idx = build_indices(logs_glob, sess_glob)  # returns (dict, dict)


    md = materialize_bag_markdown(
        units, ev_idx, ss_idx,
        collapse=True, max_items=25,
        since_iso=None, until_iso=None,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    typer.echo(f"[bag-md] WROTE → {out_md}")


# ------------------------------------------------------------------------------
# bags-merge
# ------------------------------------------------------------------------------
@app.command(name="bags-merge")
def bags_merge(
        inputs: list[Path] = typer.Option(..., help="One or more units JSONL files"),
        out: Path          = typer.Option(Path("runs/units_all.jsonl")),
    ):

    all_units = []
    seen = set()
    for inp in inputs:
        if not inp.exists():
            typer.echo(f"⚠️  bags-merge: missing {inp}")
            continue
        for d in read_jsonl(inp):
            uid = d.get("unit_id")
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            all_units.append(d)

    write_jsonl(out, all_units, ensure_ascii=False)
    typer.echo(f"[bags-merge] merged {len(all_units)} unique units → {out}")




# ------------------------------------------------------------------------------
# index-l2 / publish
# ------------------------------------------------------------------------------



@app.command(name="index-l2")
def index_l2(
    digests_root: Path = typer.Option(..., help="Root folder with L2 outputs"),
    tz: str         = typer.Option(TZ_LOCAL, help="IANA timezone for slicing"),
    out_json: Path  = typer.Option(Path("index/l2_by_window.json"), help="Where to write index"),
    since: str      = typer.Option("", help="UTC ISO slice start (inclusive)"),
    until: str      = typer.Option("", help="UTC ISO slice end (exclusive)"),
):
    """
    Walk L2 outputs (tree, flat, onefile), collect front-matter rows,
    optionally slice by time window, and dump JSON index.
    """
    rows = []

    # — tree layout —
    for mf in digests_root.rglob("manifest.json"):
        try:
            dg = read_json(mf)["digest"]
        except Exception:
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

    # — flat: any *.json + sidecar .mdx —
    for js in digests_root.rglob("*.json"):
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

    # — onefile: read front-matter from MDX —
    for mdx in digests_root.rglob("*.mdx"):
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

    # time-slice filter
    if since or until:
        rows = [
            r for r in rows
            if in_window_range(r["start_ts"], r["end_ts"], since or None, until or None)
        ]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, rows, ensure_ascii=False, indent=2)
    typer.echo(f"index wrote {len(rows)} rows → {out_json}")



# ev_idx = load_or_build_index(Path("cache/event_index.json"),
#                              lambda: build_event_index(logs_glob))


# ------------------------------------------------------------------------------
# l2-build (with optional hydration + slice)
# ------------------------------------------------------------------------------


import re
from pathlib import Path
import typer
from typing import List

from digests_project.bags_pipeline.l2 import build_l2_digests, write_l2_all
from digests_project.bags_pipeline.core import RenderMode
from digests_project.bags_pipeline.config import TZ_LOCAL


@app.command(name="l2-build")
def l2_build(
    units_jsonl: Path = typer.Option(..., help="Path to JSONL of L2 units"),
    channels: str    = typer.Option(
        "journal,memo,cheatsheet,tech_debt,achievements",
        help="Comma-separated channels"
    ),
    out_base: Path   = typer.Option(..., help="Output base directory"),
    layout: str      = typer.Option("onefile", help="tree|flat|onefile"),
    filename_scheme: str = typer.Option("slug", help="id|slug_hash"),
    hydrate: bool    = typer.Option(False, help="Attach snippets from logs/sessions"),
    logs_glob: List[str]  = typer.Option([], help="0+ JSONL globs for logs"),
    sess_glob: List[str]  = typer.Option([], help="0+ JSONL globs for sessions"),
    max_items: int   = typer.Option(120, help="Max snippets per unit"),
    since: str       = typer.Option("", help="ISO UTC slice start"),
    until: str       = typer.Option("", help="ISO UTC slice end"),
    render_mode: RenderMode = typer.Option(
        RenderMode.both, show_choices=True,
        help="Include content, summary, or both"
    ),
):
    """
    Build all L2 digests: read units, optionally hydrate with logs & sessions,
    slice by time window, render Markdown, and write to disk.
    """
    chans = [c for c in channels.split(",") if c]
    drafts = build_l2_digests(
        units_path=units_jsonl,
        channels=chans,
        hydrate=hydrate,
        logs_globs=logs_glob,
        sess_globs=sess_glob,
        since=since or None,
        until=until or None,
        render_mode=render_mode,
        max_items=max_items,
    )
    n = write_l2_all(
        drafts=drafts,
        out_base=out_base,
        layout=layout,
        filename_scheme=filename_scheme,
    )
    typer.echo(f"L2 wrote {n} digests → {out_base}")





from digests_project.bags_pipeline.publish import (
    publish_l2
)
# from digests_project.bags_pipeline.textnorm import parse_utc_any
from typing import Tuple, Dict, Any, List
from digests_project.bags_pipeline.config import _in_window_range, parse_utc_any



@app.command(name="publish")
def publish(
    digests_root: Path = typer.Option(..., help="Root folder that contains L2 outputs"),
    only_validated: bool = typer.Option(True, "--only-validated/--all"),
    out_dir: Optional[Path] = typer.Option(None, help="Defaults to <root>/_published"),
    link: bool = typer.Option(False, help="Symlink instead of copying"),
    since: str = typer.Option("", help="UTC ISO start (inclusive)"),
    until: str = typer.Option("", help="UTC ISO end (exclusive)"),
):
    """
    Publish L2 digests by walking manifests, filtering by validation
    and optional time window, then copying or symlinking MDX files.
    """
    rows: List[Dict[str, Any]] = []

    # — gather candidates from tree (manifest.json) —
    for mf in digests_root.rglob("manifest.json"):
        info = read_json(mf).get("digest", {})
        mdx = mf.with_name("digest.mdx")
        rows.append({
            "manifest": mf,
            "mdx":       mdx,
            "validated": info.get("policy","") == "validated",
            "start_ts":  info.get("start_ts"),
            "end_ts":    info.get("end_ts"),
        })

    # if no manifest.json found, fall back to any .mdx
    if not rows:
        for mdx in digests_root.rglob("*.mdx"):
            rows.append({"mdx": mdx, "validated": True, "manifest": None,
                         "start_ts": None, "end_ts": None})

    # apply time‐slice filter
    if since or until:
        rows = [
            r for r in rows
            if _in_window_range(r["start_ts"], r["end_ts"], since or None, until or None)
        ]
    # _load_items_from_manifests(root)


    target = out_dir or (digests_root / "_published")

    published = publish_l2(
        root=digests_root,
        # items=rows,
        only_validated=only_validated,
        out_dir=target,
        link=link,
    )

    typer.echo(f"published {len(published)} files → {target}")



# build_indices

# ------------------------------------------------------------------------------
# sessions-digest-tag-window (single markdown)
# ------------------------------------------------------------------------------
@app.command(name="sessions-digest-tag-window")
def sessions_digest_tag_window(
    sessions_glob: list[str] = typer.Option(...),
    tag: str                = typer.Option(...),
    since: str              = typer.Option(...),
    until: str              = typer.Option(...),
    out_md: Path            = typer.Option(Path("digests/L3/weekly/tag.md")),
    logs_glob: list[str]    = typer.Option([], help="0+ JSONL globs for logs"),
):
    sessions = load_sessions(sessions_glob)
    units = sessions_to_units_window(
        sessions,
        since_iso=since,
        until_iso=until,
        require_tag=tag,
    )
    ev_idx, ss_idx = build_indices(logs_glob, sessions_glob)

    md = materialize_bag_markdown(
        units,
        ev_idx, ss_idx,
        collapse   = True,
        max_items  = 25,
        since_iso  = since,
        until_iso  = until,
    )

    out_md.parent.mkdir(exist_ok=True, parents=True)
    out_md.write_text(md, encoding="utf-8")
    typer.echo(f"WROTE sessions digest → {out_md}  (n={len(units)})")

# ev_idx = load_or_build_index(Path("cache/event_index.json"),
#                              lambda: build_event_index(logs_glob))



# ------------------------------------------------------------------------------
# eda-tagpairs-from-units (re-exports from eda_bridge via facade)
# ------------------------------------------------------------------------------
from digests_project.bags_pipeline.eda_bridge import long_from_units, pairs_from_units  # or import from .tagbags
# cli/kbctl.py
import json
import typer
from pathlib import Path
import pandas as pd

from digests_project.bags_pipeline.eda_bridge import long_from_units, pairs_from_units
from digests_project.bags_pipeline.pairs import subsets
from digests_project.bags_pipeline.pairs import GatePolicy  # if you placed gating separately

@app.command(name="eda-tagpairs-from-units")
def eda_tagpairs_from_units_cmd(
    units_jsonl: Path = typer.Option(..., help="Units .jsonl"),
    out_dir: Path    = typer.Option(Path("outputs/eda_units")),
    top_k: int       = typer.Option(300),
    min_docs: int    = typer.Option(5),
    min_npmi: float  = typer.Option(0.05),

    # gating overrides (JSON and/or flags). Flags win over JSON.
    gates_json: Path | None = typer.Option(None, help="Optional JSON file with gate thresholds"),
    # atomic flags
    co_default_floor: int   = typer.Option(None, help="Floor for CO_DEFAULT"),
    co_backbone_floor: int  = typer.Option(None, help="Floor for CO_BACKBONE"),
    npmi_keep_floor: float  = typer.Option(None, help="Floor for NPMI_KEEP"),
    lift_keep_floor: float  = typer.Option(None, help="Floor for LIFT_KEEP"),
    niche_lo: int | None    = typer.Option(None, help="Niche low boundary (co_docs)"),
    niche_hi: int | None    = typer.Option(None, help="Niche high boundary (co_docs)"),
    min_bridge_floor: float = typer.Option(None, help="Floor for NPMI_BRIDGE"),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    units = list(read_jsonl(units_jsonl))

    # 1) long form
    long_df = long_from_units(units)
    write_csv(out_dir / "doc_tag_long.csv", long_df, index=False)

    # 2) pairs + communities
    results = pairs_from_units(units, top_k=top_k, min_docs=min_docs, min_npmi=min_npmi)
    write_csv(out_dir / "co_tag_pairs.csv",    results["pairs"],       index=False)
    write_csv(out_dir / "tag_communities.csv", results["communities"], index=False)

    # 3) gating: JSON overrides or policy flags
    custom_gates = None
    if gates_json and gates_json.exists():
        with gates_json.open("r", encoding="utf-8") as f:
            custom_gates = json.load(f)

    # Build a GatePolicy from flags (only set fields that were provided)
    policy = GatePolicy()
    kwargs = {}
    if co_default_floor is not None:  kwargs["co_default_floor"]  = co_default_floor
    if co_backbone_floor is not None: kwargs["co_backbone_floor"] = co_backbone_floor
    if npmi_keep_floor is not None:   kwargs["npmi_keep_floor"]   = npmi_keep_floor
    if lift_keep_floor is not None:   kwargs["lift_keep_floor"]   = lift_keep_floor
    if niche_lo is not None:          kwargs["niche_lo"]          = niche_lo
    if niche_hi is not None:          kwargs["niche_hi"]          = niche_hi
    if min_bridge_floor is not None:  kwargs["min_bridge_floor"]  = min_bridge_floor
    if kwargs:
        policy = GatePolicy(**{**policy.__dict__, **kwargs})

    # 4) subsets: if custom_gates is given, it takes precedence; else use policy
    subsets_out = subsets(
        results["pairs"],
        comm=results.get("communities"),
        stats=None,
        k_top=12,
        gates=custom_gates,
        gate_policy=None if custom_gates else policy,
    )

    # Write all subsets: DataFrames → CSV; dicts → JSON
    for name, obj in subsets_out.items():
        if isinstance(obj, pd.DataFrame):
            write_csv(out_dir / f"{name}.csv", obj, index=False)
        elif isinstance(obj, dict):
            write_json(out_dir / f"{name}.json", obj)

    manifest = {
        "pairs_rows":        int(results["pairs"].shape[0]),
        "communities_rows":  int(results["communities"].shape[0]),
        "params": {"top_k": top_k, "min_docs": min_docs, "min_npmi": min_npmi},
        "gates_source": "json" if custom_gates else ("policy_flags" if kwargs else "policy_defaults"),
    }
    write_json(out_dir / "index.json", manifest)

    if results["pairs"].empty:
        typer.echo("⚠️  pairs is empty; try --min-docs 1 and/or --min-npmi 0.0")
    typer.echo(f"EDA from units → {out_dir}")



# # --- small I/O helpers (you already had write_csv; add write_json if missing) ---
# def write_csv(path: Path, df, **kwargs) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, **kwargs)

# def write_json(path: Path, obj) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)





# @app.command(name="eda-tagpairs-from-units")
# def eda_tagpairs_from_units_cmd(
#     units_jsonl: Path = typer.Option(...),
#     out_dir: Path    = typer.Option(Path("outputs/eda_units")),
#     top_k: int       = typer.Option(300),
#     min_docs: int    = typer.Option(5),
#     min_npmi: float  = typer.Option(0.05),
# ):
#     import pandas as pd

#     out_dir.mkdir(parents=True, exist_ok=True)

#     units = list(read_jsonl(units_jsonl))

#     # 1) long form
#     long_df = long_from_units(units)
#     write_csv(out_dir / "doc_tag_long.csv", long_df, index=False)

#     # 2) co‐tag pairs + communities
#     results = pairs_from_units(
#         units,
#         top_k=top_k,
#         min_docs=min_docs,
#         min_npmi=min_npmi,
#     )

#     write_csv(out_dir / "co_tag_pairs.csv",    results["pairs"],       index=False)
#     write_csv(out_dir / "tag_communities.csv", results["communities"], index=False)

#     # 3) subsets: DataFrames → CSV, dicts → JSON
#     for subset_name, obj in results["subsets"].items():
#         if isinstance(obj, pd.DataFrame):
#             write_csv(out_dir / f"{subset_name}.csv", obj, index=False)
#         elif isinstance(obj, dict):
#             write_json(out_dir / f"{subset_name}.json", obj)
#         else:
#             # fallback: try to coerce dict-like to JSON, otherwise skip
#             try:
#                 write_json(out_dir / f"{subset_name}.json", dict(obj))
#             except Exception:
#                 typer.echo(f"⚠️  Skipping subset '{subset_name}' (unsupported type: {type(obj).__name__})")

#     if results["pairs"].empty:
#         typer.echo("⚠️  pairs is empty; try --min-docs 1 and/or --min-npmi 0.0")
#     typer.echo(f"EDA from units → {out_dir}")


#     manifest = {
#         "pairs_rows": int(results["pairs"].shape[0]),
#         "communities_rows": int(results["communities"].shape[0]),
#         "params": {"top_k": top_k, "min_docs": min_docs, "min_npmi": min_npmi},
#     }
#     write_json(out_dir / "index.json", manifest)

# @app.command(name="eda-tagpairs-from-units")
# def eda_tagpairs_from_units_cmd(
#     units_jsonl: Path = typer.Option(...),
#     out_dir: Path    = typer.Option(Path("outputs/eda_units")),
#     top_k: int       = typer.Option(300),
#     min_docs: int    = typer.Option(5),
#     min_npmi: float  = typer.Option(0.05),
#     gates_json: Path = typer.Option(None, help="Optional JSON with gating thresholds")
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     units = list(read_jsonl(units_jsonl))
#     long_df = long_from_units(units)
#     write_csv(out_dir / "doc_tag_long.csv", long_df, index=False)

#     results = pairs_from_units(units, top_k=top_k, min_docs=min_docs, min_npmi=min_npmi)

#     # Load custom gates (if provided) and rebuild subsets with overrides
#     custom_gates = None
#     if gates_json and gates_json.exists():
#         with gates_json.open("r", encoding="utf-8") as f:
#             custom_gates = json.load(f)

#     # Recompute subsets with optional gate overrides
#     from digests_project.bags_pipeline.pairs import subsets
#     subsets_out = subsets(
#         results["pairs"],
#         comm=results.get("communities"),
#         stats=None,
#         k_top=12,
#         gates=custom_gates
#     )
#     # write everything (DFs as CSV, dicts as JSON) – reuse your writers


# ------------------------------------------------------------------------------
# validate / dedupe / hydrate (streamy utilities that already exist)
# # ------------------------------------------------------------------------------

from digests_project.bags_pipeline.validate import load_schema, iter_validate

@app.command(name="validate")
def kbctl_validate(
    units_jsonl: Path         = typer.Option(...),
    schema: Path              = typer.Option(...),
    write_invalid: Path       = typer.Option(None),
):
    sch = load_schema(schema)
    valid, invalid = [], []

    for doc, errors in iter_validate(read_jsonl(units_jsonl), sch):
        if errors:
            invalid.append({"doc": doc, "errors": errors})
        else:
            valid.append(doc)

    if write_invalid:
        write_invalid.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(write_invalid, invalid, ensure_ascii=False)

    typer.echo(f"validated {len(valid)+len(invalid)} docs · valid={len(valid)} · invalid={len(invalid)}")
    if invalid:
        raise typer.Exit(code=1)




# # ------------------------------------------------------------------------------
# # hydrate-dryrun
# # ------------------------------------------------------------------------------
# @app.command(name="hydrate-dryrun")
# def hydrate_dryrun(
#     units_jsonl: Path = typer.Option(..., help="JSONL of units to hydrate"),
#     logs_glob:   List[str] = typer.Option([], help="globs for event logs"),
#     sess_glob:   List[str] = typer.Option([], help="globs for session logs"),
#     limit:       int       = typer.Option(5,   help="how many units to sample"),
#     verbose:     bool      = typer.Option(False, help="show missing sources"),
# ):
#     # 1) load
#     units = load_units(units_jsonl)

#     # 2) build event+session indices
#     ev_idx, ss_idx = build_indices(logs_glob, sess_glob)  # returns (dict, dict)

#     # 3) report meta
#     em, sm = ev_idx.get("_meta", {}), ss_idx.get("_meta", {})
#     typer.echo(f"[events]   files={len(em.get('files', []))}  scanned={em.get('scanned',0)}  kept={em.get('kept_primary',0)}")
#     typer.echo(f"[sessions] files={len(sm.get('files', []))}  scanned={sm.get('scanned',0)}  kept={sm.get('kept_primary',0)}")

#     # 4) sample first N units and check how many sources resolve
#     total_src = res_e = res_s = 0
#     for u in units[:limit]:
#         ok_e = ok_s = tot = 0
#         for kind, sid in u.sources:
#             tot += 1
#             if kind == "event"   and sid in ev_idx:  ok_e += 1
#             if kind == "session" and sid in ss_idx:  ok_s += 1

#         total_src += tot
#         res_e += ok_e
#         res_s += ok_s

#         typer.echo(f"- {u.unit_type}/{u.unit_id} → {ok_e} events + {ok_s} sessions of {tot}")
#         if verbose and tot and (ok_e + ok_s) == 0:
#             typer.echo(f"    tags={u.tags!r}")

#     typer.echo(f"TOTAL: {res_e} events + {res_s} sessions of {total_src} sources")


# # ------------------------------------------------------------------------------
# # hydrate
# # ------------------------------------------------------------------------------


# @app.command(name="hydrate")
# def kbctl_hydrate(
#     units_jsonl: Path        = typer.Option(..., help="JSONL of units to hydrate"),
#     out:        Path         = typer.Option(Path("runs/units_hydrated.jsonl")),
#     logs_glob:  List[str]    = typer.Option([], help="globs for event logs"),
#     sess_glob:  List[str]    = typer.Option([], help="globs for session logs"),
# ):
#     # 1) build indices
#     ev_idx, ss_idx = build_indices(logs_glob, sess_glob)

#     # 2) hydrate and write JSONL
#     out.parent.mkdir(parents=True, exist_ok=True)
#     hydrated = hydrate_units_stream(load_units(units_jsonl), ev_idx, ss_idx)
#     write_jsonl(out, hydrated)

#     typer.echo(f"hydrated units → {out}")


# ------------------------------------------------------------------------------
# units-stats
# ------------------------------------------------------------------------------
@app.command(name="units-stats")
def units_stats(units_jsonl: Path = typer.Option(...)):
    from collections import defaultdict
    n = 0
    type_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    for d in iter_jsonl(units_jsonl):
        n += 1
        type_counts[d.get("unit_type", "?")] += 1
        for tg in d.get("tags", []):
            tag_counts[str(tg)] += 1
    top_tags = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    typer.echo(f"Units: {n}")
    typer.echo(f"By type: {dict(type_counts)}")
    typer.echo(f"Top tags: {top_tags}")


@app.command(name="dedupe")
def kbctl_dedupe(inp: Path = typer.Option(..., "--in"),
                 key: str = typer.Option("unit_id"),
                 out: Path = typer.Option(Path("runs/units_dedup.jsonl"))):
    from digests_project.bags_pipeline.dedupe import dedupe_jsonl
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = dedupe_jsonl(inp, out, key)
    typer.echo(f"dedupe kept={kept} dropped={dropped} → {out}")



def main() -> None:
    app()


if __name__ == "__main__":
    main()
