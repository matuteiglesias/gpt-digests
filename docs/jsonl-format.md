# JSONL format

An input is UTF-8 newline-delimited JSON with one object per non-empty line. A record
must provide usable `title`, text, or summary content. Malformed lines are reported as
invalid rather than reconstructed.

## Recognized fields

| Projection | Accepted locations |
| --- | --- |
| Text | `text`, `content`, or `outputs.summary_text` |
| Summary | `summary`, `meta.summary`, or `outputs.summary` |
| Title | `title` or `meta.title` |
| Tags | `tags`, `meta.tags`, or `outputs.tags` |
| Timestamp | `timestamp`, `ts_abs`, `ts`, `created_at`, metadata timestamp, or epoch milliseconds |
| Source reference | `meta.provenance.source_ref` or `source_ref` |
| Text checksum | `text_sha256` or `meta.text_sha256` |

Unknown top-level fields are retained in the normalized record's `raw_record`.
Nested `meta` values become annotations, except for the provenance object.

## Minimal example

```json
{"title":"Deploy app","text":"Build and deploy the service","tags":["runbook"]}
```

## Stable record identity

Identity uses the first available value in this order:

1. an explicit source reference;
2. conversation and message IDs;
3. a supplied text SHA-256;
4. a SHA-256 fingerprint of the canonicalized JSON object.

This fallback makes a record stable for identical input content; it does not invent
missing provenance. See [Provenance](provenance.md).
