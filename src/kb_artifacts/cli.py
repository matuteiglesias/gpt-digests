"""Thin command-line boundary for canonical artifact builds."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Annotated

import typer

from kb_artifacts.corpus import count_corpus, describe_corpus, facet_corpus, sample_corpus
from kb_artifacts.inspection import inspect_source
from kb_artifacts.query import QueryValidationError, parse_query
from kb_artifacts.profiles import CorpusProfileError, CorpusProfiles, load_corpus_profiles
from kb_artifacts.selection import SelectionRequest, select
from kb_artifacts.sources.jsonl_bus import SourceInputError

app = typer.Typer(add_completion=False, no_args_is_help=True)
inspect_app = typer.Typer(add_completion=False, no_args_is_help=True)
corpus_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")
app.add_typer(corpus_app, name="corpus")


def _field_pairs(values: list[str]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key.strip() or not item.strip():
            raise typer.BadParameter("--field must use NAME=VALUE")
        pairs.append((key.strip(), item.strip()))
    return tuple(pairs)


def _query_input(query: str | None, query_file: Path | None):
    if query is not None and query_file is not None:
        raise ValueError("use only one of --query or --query-file")
    if query_file is not None:
        try:
            source = query_file.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"could not read query file: {error}") from error
    elif query is not None:
        source = query
    else:
        return None
    try:
        value = json.loads(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"query is not valid JSON: {error}") from error
    return parse_query(value)


def _emit_json(value: dict) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _load_profiles(path: Path | None) -> CorpusProfiles | None:
    return load_corpus_profiles(path) if path is not None else None


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
    corpus: Annotated[str | None, typer.Option("--corpus")]=None,
    profiles_file: Annotated[Path | None, typer.Option("--profiles-file", envvar="KB_ARTIFACT_CORPUS_PROFILES")]=None,
    query: Annotated[str | None, typer.Option("--query", help="Query expression as JSON.")]=None,
    query_file: Annotated[Path | None, typer.Option("--query-file", help="UTF-8 JSON query file.")]=None,
) -> None:
    """Select governed evidence once and export JSONL, CSV, and Markdown."""
    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
        if start_date and end_date and start_date > end_date:
            raise ValueError("--from must not be after --to")
        request = SelectionRequest(chunk_globs=tuple(chunk_glob), summary_globs=tuple(summary_glob), start=start_date, end=end_date, tags=tuple(tag), fields=_field_pairs(field), text_pattern=text, families=tuple(family), maturities=tuple(maturity), limit=limit, deduplicate=not no_deduplicate, group_by=group_by, query=_query_input(query, query_file), corpus=corpus)
        manifest = select(request, output=output, allow_empty=allow_empty, profiles=_load_profiles(profiles_file))
    except (SourceInputError, CorpusProfileError, ValueError) as error:
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
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"[kb-artifact] observed={report['counts']['records_observed']} -> {output}")


@corpus_app.command("describe")
def corpus_describe_command(
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")] = [],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    max_files_per_kind: Annotated[int | None, typer.Option("--max-files-per-kind", min=1)] = None,
    max_records: Annotated[int | None, typer.Option("--max-records", min=1)] = None,
    include_excerpts: Annotated[bool, typer.Option("--include-excerpts")] = False,
    allow_empty: Annotated[bool, typer.Option("--allow-empty")] = False,
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    profiles_file: Annotated[Path | None, typer.Option("--profiles-file", envvar="KB_ARTIFACT_CORPUS_PROFILES")] = None,
) -> None:
    """Describe corpus shape as JSON on stdout."""
    try:
        report = describe_corpus(chunk_globs=chunk_glob, summary_globs=summary_glob, corpus=corpus, profiles=_load_profiles(profiles_file), max_files_per_kind=max_files_per_kind, max_records=max_records, include_excerpts=include_excerpts, allow_empty=allow_empty)
    except (SourceInputError, CorpusProfileError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)
    _emit_json(report)


@corpus_app.command("facet")
def corpus_facet_command(
    field: str,
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")] = [],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    query: Annotated[str | None, typer.Option("--query", help="Query expression as JSON.")] = None,
    query_file: Annotated[Path | None, typer.Option("--query-file", help="UTF-8 JSON query file.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
    allow_empty: Annotated[bool, typer.Option("--allow-empty")] = False,
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    profiles_file: Annotated[Path | None, typer.Option("--profiles-file", envvar="KB_ARTIFACT_CORPUS_PROFILES")] = None,
) -> None:
    """Facet one normalized field as JSON on stdout."""
    try:
        report = facet_corpus(field=field, chunk_globs=chunk_glob, summary_globs=summary_glob, corpus=corpus, profiles=_load_profiles(profiles_file), query=_query_input(query, query_file), limit=limit, allow_empty=allow_empty)
    except (SourceInputError, CorpusProfileError, QueryValidationError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)
    _emit_json(report)


@corpus_app.command("count")
def corpus_count_command(
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")] = [],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    query: Annotated[str | None, typer.Option("--query", help="Query expression as JSON.")] = None,
    query_file: Annotated[Path | None, typer.Option("--query-file", help="UTF-8 JSON query file.")] = None,
    allow_empty: Annotated[bool, typer.Option("--allow-empty")] = False,
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    profiles_file: Annotated[Path | None, typer.Option("--profiles-file", envvar="KB_ARTIFACT_CORPUS_PROFILES")] = None,
) -> None:
    """Count query matches as JSON on stdout."""
    try:
        report = count_corpus(chunk_globs=chunk_glob, summary_globs=summary_glob, corpus=corpus, profiles=_load_profiles(profiles_file), query=_query_input(query, query_file), allow_empty=allow_empty)
    except (SourceInputError, CorpusProfileError, QueryValidationError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)
    _emit_json(report)


@corpus_app.command("sample")
def corpus_sample_command(
    chunk_glob: Annotated[list[str], typer.Option("--chunk-glob")] = [],
    summary_glob: Annotated[list[str], typer.Option("--summary-glob")] = [],
    query: Annotated[str | None, typer.Option("--query", help="Query expression as JSON.")] = None,
    query_file: Annotated[Path | None, typer.Option("--query-file", help="UTF-8 JSON query file.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 10,
    excerpt_chars: Annotated[int | None, typer.Option("--excerpt-chars", min=1, max=1000)] = None,
    allow_empty: Annotated[bool, typer.Option("--allow-empty")] = False,
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    profiles_file: Annotated[Path | None, typer.Option("--profiles-file", envvar="KB_ARTIFACT_CORPUS_PROFILES")] = None,
) -> None:
    """Return a bounded deterministic sample as JSON on stdout."""
    try:
        report = sample_corpus(chunk_globs=chunk_glob, summary_globs=summary_glob, corpus=corpus, profiles=_load_profiles(profiles_file), query=_query_input(query, query_file), limit=limit, excerpt_chars=excerpt_chars, allow_empty=allow_empty)
    except (SourceInputError, CorpusProfileError, QueryValidationError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)
    _emit_json(report)


@corpus_app.command("list")
def corpus_list_command(
    profiles_file: Annotated[Path, typer.Option("--profiles-file", envvar="KB_ARTIFACT_CORPUS_PROFILES")],
) -> None:
    """List configured corpus profiles as path-private JSON."""
    try:
        _emit_json(load_corpus_profiles(profiles_file).list())
    except CorpusProfileError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
