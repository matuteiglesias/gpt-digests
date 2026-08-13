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

Different filter kinds combine with AND semantics. Input patterns expand to sorted,
deduplicated files, and selected records use deterministic ordering.

Errors use exit code 2 and do not promote a partial output as success.
