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
| `query` | Optional validated `QueryExpression` or JSON-compatible expression mapping. |

Different filter types combine with AND semantics. The defaults do not add a
ranking or an implicit filter.

## Query expressions

`QueryExpression` is the stable, typed wrapper for a compact JSON-compatible query
algebra. `parse_query(mapping)` validates untrusted serialized input and returns the
same type. `evaluate_query(record, expression)` is the central, read-only evaluator
used by selection and is also available to callers that already hold normalized
`EvidenceRecord` objects.

Every expression object contains exactly one operator:

| Form | Semantics |
| --- | --- |
| `{"eq": {"field": F, "value": V}}` | A scalar equals `V`, or one collection member equals `V`. Strings use existing case/accent/separator normalization; other values require the same type and value. |
| `{"in": {"field": F, "values": [V, ...]}}` | The field, or one of its collection members, equals any listed value. The list must be non-empty. |
| `{"contains": {"field": F, "value": V}}` | Collection membership, or normalized substring containment when both operands are strings. |
| `{"exists": {"field": F}}` | The field is present and is not `null`. Empty strings and empty collections still exist. |
| `{"gte": {"field": F, "value": N}}` | The field is numeric and greater than or equal to numeric `N`. |
| `{"lte": {"field": F, "value": N}}` | The field is numeric and less than or equal to numeric `N`. |
| `{"regex": {"target": F, "pattern": P}}` | Case-insensitive Python regular-expression search of a string field. |
| `{"all": [E, ...]}` | Every nested expression matches (AND). The list must be non-empty. |
| `{"any": [E, ...]}` | At least one nested expression matches (OR). The list must be non-empty. |
| `{"not": E}` | The nested expression does not match. |

Fields resolve against common normalized projections (`record_id`, `source_kind`,
`title`, `summary`, `text`, `timestamp`, `conversation_id`, `message_id`, and
`tags`) and then annotation names. Missing fields fail closed. Regex evaluates only
the explicitly named target; choose `text`, `summary`, or `title` when searching a
text-bearing projection. Numeric operators accept JSON numbers, excluding booleans,
and never parse numeric strings. Invalid operators, shapes, field names, scalar
values, comparison operands, and regex patterns raise `QueryValidationError`.

Queries are filters, not ranking. They do not change ordering, assign relevance,
perform fuzzy matching, or introduce implicit scoring. An optional query combines
with legacy `SelectionRequest` filters using AND; requests that omit it retain the
legacy behavior and manifest shape.

### Agent-oriented examples

```python
from kb_artifacts import QueryExpression, SelectionRequest, parse_query, select

# An agent can emit this mapping directly as JSON.
query = parse_query({
    "all": [
        {"contains": {"field": "tags", "value": "playbook"}},
        {"contains": {"field": "tags", "value": "automation"}},
        {"in": {"field": "domain", "values": ["automation", "software_engineering"]}},
        {"not": {"eq": {"field": "stage", "value": "reflection"}}},
        {"gte": {"field": "reusability_score", "value": 4}},
    ]
})

request = SelectionRequest(chunk_globs=("data/*.jsonl",), query=query)
select(request, output="artifacts/agent-query")

# Other explicit predicates.
has_owner = QueryExpression({"exists": {"field": "owner"}})
text_match = QueryExpression({"regex": {"target": "summary", "pattern": r"deploy(ed|ment)"}})
```

`QueryExpression.to_dict()` returns a fresh JSON-compatible mapping suitable for
serialization. Query parsing is deliberately not natural-language planning: agents
must construct an explicit operator tree.

## Exploring a corpus

Four read-only operations support progressive corpus discovery without writing a
selected-evidence artifact:

- `describe_corpus(...)` returns the same machine-readable report as
  `inspect_source(...)`, including inventory, observed/invalid counts, schema keys,
  missingness, common values, tags, identity representation, normalization
  collisions, diagnostics, samples, and explicit bounds.
- `facet_corpus(field=..., query=..., limit=...)` counts normalized values for tags,
  normalized projections, or annotations. Results sort by descending count and then
  value. Missing values are counted separately.
- `count_corpus(query=...)` reports scanned, considered, matching, and invalid
  records. Exploratory counts do not deduplicate; the response states this.
- `sample_corpus(query=..., limit=..., excerpt_chars=None)` returns the first
  matching records in stable timestamp/identity/source order. Samples contain
  metadata, title, summary, tags, annotations, and stable provenance references.
  Full text is never returned; a text excerpt requires an explicit `excerpt_chars`
  between 1 and 1000.

All operations accept explicit `chunk_globs` and `summary_globs`, use the same query
evaluator as durable selection, scan normally without an index or cache, and support
`allow_empty=True`. They never mutate source data or promote their results.

`load_corpus_profiles(path)` loads local TOML configuration into `CorpusProfiles`.
Pass `corpus="name"` and `profiles=profiles` to any corpus operation, or set
`SelectionRequest(corpus="name")` and pass `profiles` to `select`. Profile mode and
explicit globs are mutually exclusive. `CorpusProfiles.list()` intentionally omits
the underlying paths while exposing IDs, descriptions, source roles, annotations,
and excerpt policy. Profiles default to disallowing excerpts; explicit excerpt
requests against such a profile fail with `CorpusProfileError`.

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

The package-root names documented here are supported through `0.x`:

- `CorpusProfileError`
- `CorpusProfiles`
- `EvidenceRecord`
- `QueryExpression`
- `QueryValidationError`
- `SourceReference`
- `count_corpus`
- `describe_corpus`
- `evaluate_query`
- `facet_corpus`
- `SelectionDecision`
- `SelectionRequest`
- `inspect_source`
- `load_corpus_profiles`
- `parse_query`
- `sample_corpus`
- `select`

Modules beneath `kb_artifacts`, CLI implementation objects, classification helpers,
source adapters, and undocumented manifest details remain internal.
