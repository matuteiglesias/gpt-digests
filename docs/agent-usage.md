# Agent usage

`kb-artifacts` owns deterministic inspection, explicit filtering, selected-evidence
exports, provenance, and atomic output promotion. It does not own source-document
meaning, natural-language interpretation, semantic ranking, arbitrary analysis,
MCP access policy, hydration, or downstream context routing.

## Responsibilities

The agent or calling application compiles intent into the documented JSON query
schema. The package validates and evaluates that expression without implicit scoring
or ranking. No command interprets a natural-language query.

Use this protocol:

1. `corpus list` to discover locally approved profile IDs.
2. `corpus describe` to inspect shape, missingness, identities, and bounds.
3. `corpus facet` to inspect normalized values.
4. Construct an explicit `eq`/`in`/`contains`/`exists`/`gte`/`lte`/`regex` expression,
   composed with `all`/`any`/`not`.
5. `corpus count` to measure the candidate set.
6. `corpus sample --limit N` to inspect deterministic metadata-only candidates.
7. Refine and repeat.
8. Run `select` only when the governed candidate set is appropriate.
9. Retain stable record/source references as the bridge to a future authorized
   hydration layer.

Exploration is not selection, promotion, or publication. Only selected evidence is
the durable governed product.

## Privacy and bounds

Profile discovery omits configured paths. Samples omit full bodies by default and
always have an explicit record limit. Text disclosure requires a bounded excerpt and
must also be permitted by the local profile. Local access policy remains outside this
package.

## Runnable offline example

From the repository root:

```bash
python examples/agent/progressive_query.py
```

The example lists a sanitized profile, describes and facets it, counts and samples a
candidate query, refines the expression, and creates a temporary selected-evidence
artifact. It is deterministic and performs no network access.

Indexes, caches, embeddings, fuzzy ranking, natural-language planning, MCP servers,
and hydration remain intentionally unsupported.
