# digests_project/cli/kbctl_compute.py
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import typer

from digests_project.bags_pipeline.compute.config import TZ_LOCAL, parse_utc_any
from digests_project.bags_pipeline.compute.core import RenderMode, Unit
from digests_project.bags_pipeline.compute.hydrate import materialize_bag_markdown
from digests_project.bags_pipeline.compute.index import build_indices
from digests_project.bags_pipeline.compute.ingest_logs import (
    build_log_cohorts,   # for dry run previews
    load_events_from_logs,
    write_log_cohorts,   # preferred writer
)
from digests_project.bags_pipeline.compute.ingest_sessions import load_sessions
from digests_project.bags_pipeline.compute.io import (
    iter_jsonl,
    read_json,
    read_jsonl,
    read_mdx_front_matter,
    write_csv,
    write_json,
    write_jsonl,
)
from digests_project.bags_pipeline.compute.l2 import trim_unit_sources_for_window
from digests_project.bags_pipeline.compute.pairs import pairbag_units_from_units, tagbag_units_from_units
from digests_project.bags_pipeline.compute.tag_select import UnitSelector
from digests_project.bags_pipeline.compute.textnorm import in_window_range
from digests_project.bags_pipeline.compute.unitize import (
    cohort_units_from_logs,
    sessions_to_units_window,
    units_from_sessions,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _load_units(units_jsonl: Path) -> List[Unit]:
    raw = read_jsonl(units_jsonl)
    out: List[Unit] = []
    for d in raw:
        if hasattr(d, "unit_id"):
            out.append(d)  # already a Unit-like object
        else:
            out.append(Unit(**d))
    return out


# ---------------------------------------------------------------------
# bags-logs -> Units (deterministic)
# ---------------------------------------------------------------------
@app.command("bags-logs")
def bags_logs(
    logs_glob: List[str] = typer.Option(..., help="JSONL glob(s). Point these at bus paths."),
    out: Path = typer.Option(Path("runs/units_logs.jsonl")),
    since: str = typer.Option("", help="UTC ISO start (inclusive)"),
    until: str = typer.Option("", help="UTC ISO end (exclusive)"),
    group_by: str = typer.Option("day", help="day|week|month|session"),
    combo_size: int = typer.Option(2, help="0=no pairs, 1=single, 2=pairs"),
    dry_run: bool = typer.Option(False, help="Print cohort preview only"),
):
    """
    Read log events (via globs), build cohort Units, write JSONL.
    """
    if dry_run:
        cohorts = build_log_cohorts(
            logs_glob=logs_glob,
            since=since or None,
            until=until or None,
            group_by=group_by,
            combo_size=combo_size,
        )
        preview = {
            "cohort_count": len(cohorts),
            "first_cohort": (cohorts[0].__dict__ if cohorts else None),
        }
        typer.echo(json.dumps(preview, ensure_ascii=False, indent=2))
        raise typer.Exit(code=0)

    # Keep behavior close to your earlier code path
    events = load_events_from_logs(logs_glob)
    if since or until:
        t0, t1 = parse_utc_any(since), parse_utc_any(until)
        events = [
            e for e in events
            # if (not t0 or parse_utc_any(e.ts_abs) >= t0)
            # and (not t1 or parse_utc_any(e.ts_abs) < t1)
        ]

    units = cohort_units_from_logs(events, group_by=group_by, combo_size=combo_size)
    write_jsonl(out, (u.__dict__ for u in units))
    typer.echo(f"[bags-logs] wrote {len(units)} units -> {out}")


# ---------------------------------------------------------------------
# bags-sessions -> Units (deterministic)
# ---------------------------------------------------------------------
@app.command("bags-sessions")
def bags_sessions(
    sessions_glob: List[str] = typer.Option(..., help="JSONL glob(s) for sessions bus"),
    out: Path = typer.Option(Path("runs/units_sessions.jsonl")),
    since: str = typer.Option("", help="UTC ISO start (inclusive)"),
    until: str = typer.Option("", help="UTC ISO end (exclusive)"),
):
    """
    Read sessions (via globs), build session Units, write JSONL.
    """
    sessions = load_sessions(sessions_glob)
    units = units_from_sessions(sessions)

    # Optional windowing (kept consistent with logs path)
    if since or until:
        selector = UnitSelector(since=since or None, until=until or None)
        units = selector.select(units)

    write_jsonl(out, (u.__dict__ for u in units))
    typer.echo(f"[bags-sessions] wrote {len(units)} units -> {out}")


# ---------------------------------------------------------------------
# bags-pairs-from-units -> Units
# ---------------------------------------------------------------------
@app.command("bags-pairs-from-units")
def bags_pairs_from_units(
    units_jsonl: Path = typer.Option(...),
    out: Path = typer.Option(Path("runs/units_pairs.jsonl")),
    pairs_csv: Optional[Path] = typer.Option(None, help="Optional existing pairs CSV"),
    top_n: int = typer.Option(150),
    min_docs: int = typer.Option(3),
):
    """
    Build pairbag units from an existing units JSONL.
    """
    units = _load_units(units_jsonl)

    pairs_df = None
    if pairs_csv and pairs_csv.exists():
        import pandas as pd  # local import keeps base CLI light
        pairs_df = pd.read_csv(pairs_csv)

    pairbags = pairbag_units_from_units(units, pairs_df=pairs_df, top_n=top_n, min_docs=min_docs)
    write_jsonl(out, (u.__dict__ for u in pairbags))
    typer.echo(f"[bags-pairs] wrote {len(pairbags)} pairbags -> {out}")


# ---------------------------------------------------------------------
# bags-tags-from-units -> Units
# ---------------------------------------------------------------------
@app.command("bags-tags-from-units")
def bags_tags_from_units(
    units_jsonl: Path = typer.Option(...),
    out: Path = typer.Option(Path("runs/units_tags.jsonl")),
    top_k_tags: int = typer.Option(50),
    min_docs: int = typer.Option(3),
):
    """
    Build tagbag units from an existing units JSONL.
    """
    units = _load_units(units_jsonl)
    tagbags = tagbag_units_from_units(units, top_k_tags=top_k_tags, min_docs=min_docs)
    write_jsonl(out, (u.__dict__ for u in tagbags))
    typer.echo(f"[bags-tags] wrote {len(tagbags)} tagbags -> {out}")


# ---------------------------------------------------------------------
# units-select -> filter Units JSONL
# ---------------------------------------------------------------------
@app.command("units-select")
def units_select(
    units_jsonl: Path = typer.Option(...),
    types: str = typer.Option("", help="csv of unit_types"),
    tags_all: str = typer.Option("", help="csv of tags all must have"),
    tags_any: str = typer.Option("", help="csv of tags any may have"),
    since: str = typer.Option("", help="UTC ISO start"),
    until: str = typer.Option("", help="UTC ISO end"),
    out: Path = typer.Option(Path("runs/units_selected.jsonl")),
):
    units = _load_units(units_jsonl)
    selector = UnitSelector(
        types=_csv_list(types),
        tags_all=_csv_list(tags_all),
        tags_any=_csv_list(tags_any),
        since=since or None,
        until=until or None,
    )
    chosen = selector.select(units)
    write_jsonl(out, (u.__dict__ for u in chosen))
    typer.echo(f"[units-select] {len(chosen)} units -> {out}")


# ---------------------------------------------------------------------
# bags-merge -> merge Units JSONL
# ---------------------------------------------------------------------
@app.command("bags-merge")
def bags_merge(
    inputs: List[Path] = typer.Option(..., help="One or more units jsonl files"),
    out: Path = typer.Option(Path("runs/units_merged.jsonl")),
):
    units: List[Unit] = []
    for p in inputs:
        units.extend(_load_units(p))
    write_jsonl(out, (u.__dict__ for u in units))
    typer.echo(f"[bags-merge] wrote {len(units)} units -> {out}")


# ---------------------------------------------------------------------
# bag-md -> render a bag markdown artifact (compute)
# ---------------------------------------------------------------------
@app.command("bag-md")
def bag_md(
    unit_json: Path = typer.Option(..., help="Single Unit json file (or jsonl with one row)"),
    out: Path = typer.Option(Path("outputs/bag.md")),
):
    if unit_json.suffix.lower() == ".jsonl":
        units = _load_units(unit_json)
        if len(units) != 1:
            raise typer.BadParameter("Expected exactly one unit in the jsonl")
        u = units[0]
    else:
        u = Unit(**read_json(unit_json))

    # md = materialize_bag_markdown(u, mode=mode)
    md = materialize_bag_markdown(u, collapse=True, max_items=25)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    typer.echo(f"[bag-md] wrote -> {out}")


# ---------------------------------------------------------------------
# sessions-digest-tag-window (kept as compute)
# ---------------------------------------------------------------------
@app.command("sessions-digest-tag-window")
def sessions_digest_tag_window(
    sessions_jsonl: Path = typer.Option(...),
    tag: str = typer.Option(...),
    since: str = typer.Option("", help="UTC ISO start"),
    until: str = typer.Option("", help="UTC ISO end"),
    out: Path = typer.Option(Path("outputs/sessions_digest.md")),
):
    sessions = list(iter_jsonl(sessions_jsonl))
    units = sessions_to_units_window(sessions, tag=tag, since=since or None, until=until or None)

    # This stays compute: it is a derived artifact
    md = materialize_bag_markdown(
        Unit(
            unit_id=f"sessions_digest:{tag}:{since}:{until}",
            unit_type="sessions_digest",
            start_ts=since,
            end_ts=until,
            tags=(tag,),
            topic_ids=(),
            sources=tuple(u.sources for u in units),
        ),
        mode=RenderMode.md,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    typer.echo(f"[sessions-digest-tag-window] wrote -> {out}")


# ---------------------------------------------------------------------
# validate + dedupe (compute QA)
# ---------------------------------------------------------------------
@app.command("validate")
def validate(
    units_jsonl: Path = typer.Option(...),
):
    units = _load_units(units_jsonl)
    bad = []
    for i, u in enumerate(units):
        if not u.unit_id or not u.unit_type:
            bad.append((i, "missing id/type"))
        if not getattr(u, "start_ts", None) or not getattr(u, "end_ts", None):
            bad.append((i, "missing start/end"))
    if bad:
        typer.echo(f"[validate] FAIL rows={len(bad)}")
        raise typer.Exit(code=1)
    typer.echo(f"[validate] OK units={len(units)}")


@app.command("dedupe")
def dedupe(
    units_jsonl: Path = typer.Option(...),
    out: Path = typer.Option(Path("runs/units_deduped.jsonl")),
):
    units = _load_units(units_jsonl)
    seen = set()
    kept: List[Unit] = []
    for u in units:
        k = (u.unit_id, u.unit_type, u.start_ts, u.end_ts)
        if k in seen:
            continue
        seen.add(k)
        kept.append(u)
    write_jsonl(out, (u.__dict__ for u in kept))
    typer.echo(f"[dedupe] kept={len(kept)} dropped={len(units) - len(kept)} -> {out}")


# ---------------------------------------------------------------------
# units-stats (kept here for now)
# ---------------------------------------------------------------------
@app.command("units-stats")
def units_stats(
    units_jsonl: Path = typer.Option(...),
    out: Path = typer.Option(Path("outputs/units_stats.json")),
):
    units = _load_units(units_jsonl)
    by_type = {}
    for u in units:
        by_type[u.unit_type] = by_type.get(u.unit_type, 0) + 1
    stats = {"units": len(units), "by_type": by_type}
    write_json(out, stats)
    typer.echo(f"[units-stats] wrote -> {out}")


# ---------------------------------------------------------------------
# EDA tagpairs from units (kept, but optional dependency)
# ---------------------------------------------------------------------
@app.command(name="eda-tagpairs-from-units")
def eda_tagpairs_from_units_cmd(
    units_jsonl: Path = typer.Option(..., help="Units .jsonl"),
    out_dir: Path = typer.Option(Path("outputs/eda_units")),
    top_k: int = typer.Option(300),
    min_docs: int = typer.Option(5),
    min_npmi: float = typer.Option(0.05),
    gates_json: Optional[Path] = typer.Option(None, help="Optional JSON file with gate thresholds"),
    co_default_floor: Optional[int] = typer.Option(None),
    co_backbone_floor: Optional[int] = typer.Option(None),
    npmi_keep_floor: Optional[float] = typer.Option(None),
    lift_keep_floor: Optional[float] = typer.Option(None),
    niche_lo: Optional[int] = typer.Option(None),
    niche_hi: Optional[int] = typer.Option(None),
    min_bridge_floor: Optional[float] = typer.Option(None),
):
    # Local imports to keep compute CLI usable without pandas unless you call this command
    import pandas as pd
    from digests_project.bags_pipeline.compute.pairs import (
        GatePolicy,
        subsets,
    )
    from digests_project.bags_pipeline.compute.eda_bridge import (
        long_from_units,
        pairs_from_units,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    units = list(read_jsonl(units_jsonl))

    long_df = long_from_units(units)
    write_csv(out_dir / "doc_tag_long.csv", long_df, index=False)

    results = pairs_from_units(units, top_k=top_k, min_docs=min_docs, min_npmi=min_npmi)
    write_csv(out_dir / "co_tag_pairs.csv", results["pairs"], index=False)
    write_csv(out_dir / "tag_communities.csv", results["communities"], index=False)

    custom_gates = None
    if gates_json and gates_json.exists():
        custom_gates = json.loads(gates_json.read_text(encoding="utf-8"))

    policy = GatePolicy()
    kwargs = {}
    if co_default_floor is not None:
        kwargs["co_default_floor"] = co_default_floor
    if co_backbone_floor is not None:
        kwargs["co_backbone_floor"] = co_backbone_floor
    if npmi_keep_floor is not None:
        kwargs["npmi_keep_floor"] = npmi_keep_floor
    if lift_keep_floor is not None:
        kwargs["lift_keep_floor"] = lift_keep_floor
    if niche_lo is not None:
        kwargs["niche_lo"] = niche_lo
    if niche_hi is not None:
        kwargs["niche_hi"] = niche_hi
    if min_bridge_floor is not None:
        kwargs["min_bridge_floor"] = min_bridge_floor
    if kwargs:
        policy = GatePolicy(**{**policy.__dict__, **kwargs})

    subsets_out = subsets(
        results["pairs"],
        comm=results.get("communities"),
        stats=None,
        k_top=12,
        gates=custom_gates,
        gate_policy=None if custom_gates else policy,
    )

    for name, obj in subsets_out.items():
        if isinstance(obj, pd.DataFrame):
            write_csv(out_dir / f"{name}.csv", obj, index=False)
        elif isinstance(obj, dict):
            write_json(out_dir / f"{name}.json", obj)

    manifest = {
        "pairs_rows": int(results["pairs"].shape[0]),
        "communities_rows": int(results["communities"].shape[0]),
        "params": {"top_k": top_k, "min_docs": min_docs, "min_npmi": min_npmi},
        "gates_source": "json" if custom_gates else ("policy_flags" if kwargs else "policy_defaults"),
    }
    write_json(out_dir / "index.json", manifest)
    typer.echo(f"[eda-tagpairs-from-units] wrote -> {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()