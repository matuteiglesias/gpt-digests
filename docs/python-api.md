# Python API

The public Python API is intentionally small. The names on this page are the API
that `kb-artifacts` supports throughout the `0.x` release series. Import them from
`kb_artifacts`; submodules and names not listed here are implementation details.

The distribution includes a `py.typed` marker, so type checkers can use the inline
annotations shipped with the package.

## Selecting records

```python
from kb_artifacts import SelectionRequest, select

request = SelectionRequest(
    chunk_globs=("data/*.jsonl",),
    tags=("runbook",),
)

manifest = select(request, output="artifacts/run")
print(manifest["counts"]["selected"])
```

### `SelectionRequest`

An immutable selection request. Its fields are:

| Field | Meaning |
| --- | --- |
| `chunk_globs` | JSONL path patterns treated as full-text records. |
| `summary_globs` | JSONL path patterns treated as summary records. |
| `start`, `end` | Optional inclusive `datetime.date` boundaries. |
| `tags` | Tags matched with OR semantics. |
| `fields` | `(name, value)` metadata filters; every pair must match. |
| `text_pattern` | Optional case-insensitive regular expression. |
| `families`, `maturities` | Optional classification filters. |
| `limit` | Optional maximum number of selected records. |
| `deduplicate` | Whether stable duplicate identities are collapsed. |
| `group_by` | Metadata field used to group `artifact.md`. |

Different filter types combine with AND semantics. The defaults do not add a
ranking or an implicit filter.

### `select(request, *, output, allow_empty=False)`

Reads every matched JSONL input without modifying it and atomically creates an output
directory containing `selected.jsonl`, `selected.csv`, `artifact.md`, and
`manifest.json`. `output` accepts a string or path-like object and must be absent or
empty. By default, a selection that matches no records raises `ValueError`; set
`allow_empty=True` only when an empty export is intentional.

In 0.1, the return value is the manifest mapping. Supported programmatic fields are
`manifest["counts"]["selected"]`, `"scanned"`, `"invalid"`, and `"deduplicated"`.
Read `manifest.json` for provenance evidence, but do not treat its remaining nested
layout as a general-purpose Python API. A dedicated result type may be introduced in
a later release without removing the mapping return during `0.x`.

## Inspecting records

### `inspect_source(...)`

```python
from kb_artifacts import inspect_source

report = inspect_source(
    chunk_globs=("data/*.jsonl",),
    summary_globs=(),
    max_records=100,
)
print(report["counts"]["records_observed"])
```

Returns a bounded metadata report without record bodies by default. Pass
`include_excerpts=True` to opt into short text excerpts, `max_files_per_kind` or
`max_records` to bound inspection, and `allow_empty=True` to accept no matching
input. Both glob arguments are explicit so callers state which source roles they
intend to inspect.

## Data contracts

### `SourceReference`

An immutable reference to a record's origin: its upstream `source_ref`, input
`partition`, one-based `line_number`, and optional `text_sha256`.

### `EvidenceRecord`

An immutable normalized view of one JSONL object. It provides stable identity,
source kind, common text and metadata projections, tags, provenance, and the
original object in `raw_record`.

### `SelectionDecision`

An immutable explanation of a record's selection disposition, reasons, matched
values, and optional classification. Applications normally consume exported records
rather than constructing decisions themselves.

## Stability boundary

The six package-root names documented here are supported through `0.x`:

- `EvidenceRecord`
- `SourceReference`
- `SelectionDecision`
- `SelectionRequest`
- `inspect_source`
- `select`

Modules beneath `kb_artifacts`, CLI implementation objects, classification helpers,
source adapters, and undocumented manifest details remain internal.
