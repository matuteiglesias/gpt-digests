"""Thin command-line boundary for canonical artifact builds."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from kb_artifacts.engine import build as build_run
from kb_artifacts.inspection import inspect_source
from kb_artifacts.recipes.sop import RECIPE
from kb_artifacts.sources.jsonl_bus import SourceInputError

app = typer.Typer(add_completion=False, no_args_is_help=True)
inspect_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")


@app.command()
def version() -> None:
    """Print the canonical artifact compiler version."""
    typer.echo("kb-artifacts 0.1.0")


@app.command("build")
def build(
    recipe: Annotated[str, typer.Argument(help="Supported recipe ID: sop (operations)")],
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/runs/sop"),
    allow_empty: Annotated[bool, typer.Option("--allow-empty")] = False,
    audit_all_decisions: Annotated[bool, typer.Option("--audit-all-decisions", help="Write ordinary nonmatches to decisions.jsonl for an explicit audit.")] = False,
    review_packet: Annotated[Path | None, typer.Option("--review-packet", help="Write a bounded classification calibration CSV.")] = None,
) -> None:
    if recipe != "sop":
        raise typer.BadParameter("Only the 'sop' (operations) recipe is currently available")
    try:
        manifest = build_run(RECIPE, chunk_globs=chunk_glob, summary_globs=summary_glob, output=output, allow_empty=allow_empty, audit_all_decisions=audit_all_decisions, review_packet=review_packet)
    except SourceInputError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"[kb-artifact] selected={manifest['counts']['selected']} -> {output}")


@inspect_app.command("source")
def inspect_source_command(
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")] = [],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    max_files_per_kind: Annotated[int | None, typer.Option("--max-files-per-kind", min=1, help="Maximum files sampled independently from each source kind.")] = None,
    max_records: Annotated[int | None, typer.Option("--max-records", min=1)] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("source-report.json"),
    include_excerpts: Annotated[bool, typer.Option("--include-excerpts")] = False,
    allow_empty: Annotated[bool, typer.Option("--allow-empty")] = False,
) -> None:
    try:
        report = inspect_source(chunk_globs=chunk_glob, summary_globs=summary_glob, max_files_per_kind=max_files_per_kind, max_records=max_records, include_excerpts=include_excerpts, allow_empty=allow_empty)
    except SourceInputError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"[kb-artifact] observed={report['counts']['records_observed']} -> {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
