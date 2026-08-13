# CLI

The installed command is `kb-artifact`. Use `--help` at every command level for the
authoritative option list.

## Inspect a source

```bash
kb-artifact inspect source [OPTIONS]
```

| Option | Purpose |
| --- | --- |
| `--chunk-glob`, `--summary-glob` | Repeatable input path patterns. |
| `--max-files-per-kind` | Bound files independently for each source role. |
| `--max-records` | Bound the total attempted records. |
| `--include-excerpts` | Opt into short text and summary excerpts. |
| `--allow-empty` | Permit an inspection with no usable input. |
| `--output` | Report path; defaults to `source-report.json`. |

## Select records

```bash
kb-artifact select [OPTIONS]
```

| Option | Purpose |
| --- | --- |
| `--chunk-glob`, `--summary-glob` | Repeatable JSONL input patterns. |
| `--from`, `--to` | Inclusive ISO date boundaries. |
| `--tag` | Repeatable tag filter; requested tags use OR semantics. |
| `--field NAME=VALUE` | Repeatable annotation filter. |
| `--text` | Case-insensitive regular expression over title, summary, and text. |
| `--family`, `--maturity` | Optional classification filters. |
| `--limit` | Maximum selected records after deterministic ordering. |
| `--no-deduplicate` | Retain repeated stable identities. |
| `--group-by` | Field used to group `artifact.md`; defaults to `domain`. |
| `--allow-empty` | Permit an intentional zero-record export. |
| `--output` | New or empty output directory. |
| `--query`, `--query-file` | Optional explicit JSON query, supplied directly or from a UTF-8 file. |
| `--corpus`, `--profiles-file` | Optional local corpus-profile resolution; mutually exclusive with explicit globs. |

Different filter kinds combine with AND semantics. Input patterns expand to sorted,
deduplicated files, and selected records use deterministic ordering.

Errors use exit code 2 and do not promote a partial output as success.

## Explore a corpus

The `corpus` command group writes one machine-readable JSON object to stdout:

```bash
kb-artifact corpus describe --chunk-glob 'data/*.jsonl'
kb-artifact corpus facet domain --chunk-glob 'data/*.jsonl' --limit 20
kb-artifact corpus count --chunk-glob 'data/*.jsonl' --query-file query.json
kb-artifact corpus sample --chunk-glob 'data/*.jsonl' --query-file query.json --limit 10
```

`facet`, `count`, and `sample` accept either `--query '{...}'` for direct JSON or
`--query-file PATH`; the options are mutually exclusive. Prefer a query file for
nested expressions. Sampling returns metadata, title, summary, tags, annotations,
and stable references by default—not record bodies. `--excerpt-chars N` explicitly
opts into a text excerpt bounded to 1–1000 characters.

These exploratory commands scan source records without deduplication and do not
create or promote selected-evidence artifacts. Use `select` only after refining and
confirming the governed query.

### Corpus profiles

All exploration commands and `select` accept `--corpus NAME` with
`--profiles-file PATH` (or the `KB_ARTIFACT_CORPUS_PROFILES` environment variable).
Profile mode cannot be combined with explicit chunk or summary globs. Discover
configured profiles without exposing their paths:

```bash
kb-artifact corpus list --profiles-file corpora.toml
kb-artifact corpus count --corpus chatgpt-history --profiles-file corpora.toml
kb-artifact select --corpus chatgpt-history --profiles-file corpora.toml --output artifacts/run
```
