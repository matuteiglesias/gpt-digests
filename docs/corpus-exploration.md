# Agent / corpus exploration

Use progressive, bounded exploration when the JSONL corpus is unfamiliar:

```text
describe → facet → count → sample → refine query → select governed evidence
```

## Configure and discover profiles

Corpus profiles are local TOML configuration. They keep approved machine-specific
globs out of agent prompts, scripts, and ordinary machine-readable results:

```toml
[corpora.chatgpt-history]
description = "Historical ChatGPT bus"
chunk_globs = ["/locally/configured/chunks/**/*.jsonl"]
summary_globs = ["/locally/configured/summaries/**/*.jsonl"]
excerpts_permitted_by_default = false

[corpora.chatgpt-history.annotations]
access = "local-approved"
```

Set `KB_ARTIFACT_CORPUS_PROFILES` to this file or pass `--profiles-file`. The file is
local configuration and should not be committed when it contains private paths.
Agents begin with path-private discovery:

```bash
kb-artifact corpus list --profiles-file corpora.toml
kb-artifact corpus describe --corpus chatgpt-history --profiles-file corpora.toml
```

Profile IDs, descriptions, available source roles, access annotations, and excerpt
policy are returned as JSON; source globs are not. A corpus profile and explicit
globs are mutually exclusive rather than silently merged. Direct-glob APIs and CLI
options remain supported for generic and human-directed use.

## 1. Describe shape

```bash
kb-artifact corpus describe --chunk-glob 'data/chunks/*.jsonl' > description.json
```

Start with source/file and record counts, source kinds, schema keys, field
missingness, common values, tag normalization, identity/provenance representation,
and diagnostics. The report records all file, record, frequency, sample, and excerpt
bounds. Bodies are excluded unless inspection excerpts are explicitly enabled.

## 2. Facet useful fields

```bash
kb-artifact corpus facet domain --corpus chatgpt-history
kb-artifact corpus facet tags --corpus chatgpt-history --limit 50
```

Facet values are normalized and deterministically ordered by count and value. A
missing count shows how much of the considered corpus lacks the field.

## 3. Count a proposed query

Put nested expressions in a file to avoid shell escaping:

```json
{
  "all": [
    {"eq": {"field": "domain", "value": "automation"}},
    {"gte": {"field": "reusability_score", "value": 4}}
  ]
}
```

```bash
kb-artifact corpus count --corpus chatgpt-history --query-file query.json
kb-artifact corpus facet stage --corpus chatgpt-history --query-file query.json
```

Counts and facets use the same evaluator as durable selection. Exploration scans
normalized source records and does not deduplicate them; every response makes the
relevant counts or semantics explicit.

## 4. Inspect a bounded sample

```bash
kb-artifact corpus sample --corpus chatgpt-history \
  --query-file query.json --limit 10
```

Sampling is deterministic rather than random. It returns metadata, title, summary,
tags, annotations, and stable record/source references. It never returns full text.
To inspect a short body projection, explicitly add `--excerpt-chars 160` (maximum
1000).

## 5. Refine, then select

Repeat facets, counts, and bounded samples until the explicit query is suitable.
Exploratory JSON is not a selected-evidence artifact and is never promoted. Only the
separate `select` operation creates the governed durable outputs, with its existing
ordering, deduplication, provenance, and atomic-promotion behavior.

This workflow deliberately has no natural-language planner, dataframe engine,
index, cache, fuzzy ranker, embedding model, or external service.

## Automated-agent contract

An automated agent should: list profiles; describe the chosen corpus; inspect useful
facets; compile its intent into the documented explicit query schema; count; request
a small bounded sample; refine; and invoke `select --corpus NAME` only when the
candidate set is appropriate. `kb-artifacts` never interprets natural language—the
agent or LLM owns that compilation. No ranking is implicit, exploration is not
promotion, and selected evidence remains the durable governed product. Stable
record and source references are the bridge to a future hydration layer. Local
filesystem access policy remains outside this package.
