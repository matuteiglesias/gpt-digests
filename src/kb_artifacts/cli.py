"""Thin command-line boundary for canonical artifact builds."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from kb_artifacts.inspection import inspect_source
from kb_artifacts.selection import SelectionRequest, select
from kb_artifacts.sources.jsonl_bus import SourceInputError

app = typer.Typer(add_completion=False, no_args_is_help=True)
inspect_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")


def _field_pairs(values: list[str]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip() or not item.strip():
            raise typer.BadParameter("--field must use NAME=VALUE")
        pairs.append((key.strip(), item.strip()))
    return tuple(pairs)


@app.command("select")
def select_command(
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")] = [],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    start: Annotated[str | None, typer.Option("--from")]=None,
    end: Annotated[str | None, typer.Option("--to")]=None,
    tag: Annotated[list[str], typer.Option("--tag")]=[],
    field: Annotated[list[str], typer.Option("--field")]=[],
    text: Annotated[str | None, typer.Option("--text")]=None,
    family: Annotated[list[str], typer.Option("--family")]=[],
    maturity: Annotated[list[str], typer.Option("--maturity")]=[],
    limit: Annotated[int | None, typer.Option("--limit", min=1)]=None,
    no_deduplicate: Annotated[bool, typer.Option("--no-deduplicate")]=False,
    group_by: Annotated[str, typer.Option("--group-by")]= "domain",
    output: Annotated[Path, typer.Option("--output")]=Path("artifacts/runs/selection"),
    allow_empty: Annotated[bool, typer.Option("--allow-empty")]=False,
) -> None:
    """Select governed evidence once and export JSONL, CSV, and Markdown."""
    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
        if start_date and end_date and start_date > end_date:
            raise ValueError("--from must not be after --to")
        request = SelectionRequest(tuple(chunk_glob), tuple(summary_glob), start_date, end_date, tuple(tag), _field_pairs(field), text, tuple(family), tuple(maturity), limit, not no_deduplicate, group_by)
        manifest = select(request, output=output, allow_empty=allow_empty)
    except (SourceInputError, ValueError) as error:
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
