# GPT Digests

A toolkit for turning messy conversational text into a navigable, auditable knowledge collection.

GPT Digests sits between raw text (ChatGPT exports, notes, summaries, sessions) and the outputs people actually want to use: daily and weekly digests, project summaries, tag views, curated briefings, and publishable pages. The point is not “more summaries”. The point is control: you can trace what went in, how it was grouped, what was selected, and what was published.

If you work with a lot of text and you want visibility and control, essentially, having it all at your fingertips, this tool can be the help you needed.

## What you can do with it

You start with text events (messages, chunks, summaries) and optionally sessions (work blocks, clusters). From there you can build:

- **Daily and weekly digests**
  - “What happened yesterday”
  - “What are the top decisions and next actions this week”
- **Topic and tag digests**
  - “Everything about pricing, hiring, or infra over the last 60 days”
  - “Recurring themes and what’s growing or fading”
- **Project briefs**
  - “What did we decide, what changed, and what’s blocked”
  - “A clean status memo for stakeholders”
- **Curated knowledge packs**
  - A Markdown page you can review or publish
  - A dataset (JSONL/CSV) you can analyze downstream

And crucially: you can do this with **observability** (indices, manifests, window slicing) and **guardrails** (validation, dedupe, selection rules), instead of ad hoc copy paste.

## Who this is for (examples)

This is intentionally useful outside engineering:

- **Researchers and scientists**: build literature style digests from conversations, notes, or field logs, and keep provenance.
- **Product and ops**: weekly execution reviews, decisions, and action lists from messy comms.
- **Consultants**: client ready briefs built from raw interactions, with traceability.
- **Legal, policy, and compliance**: structured collections of discussions and drafts with time windows and auditability.
- **Builders and knowledge workers**: turn years of logs into something searchable, sliceable, and publishable.

## How it works (conceptual pipeline)

1. **Ingest**
   - Read events and sessions from JSONL sources (typically buses: chunk, summary, session).
2. **Compute Units**
   - Group raw events into coherent windows (day, week, month, session).
3. **Derive Bags**
   - Build tag bags and pair bags (what co-occurs, what clusters).
4. **Select**
   - Filter units by type, tags, and time window.
5. **Render and Publish**
   - Produce Markdown or MDX digests and publish them into a site tree or output folder.

Everything is file-based and deterministic by default: you can rerun it, diff it, and debug it.

## Repository layout

- `src/digests_project/`
  - The codebase: pipeline components and CLI
- `docs/`
  - Documentation and API notes
- `artifacts/`
  - Outputs, indexes, published views (often gitignored depending on your workflow)
- `ops/`
  - Runbooks and operational scripts

## Quickstart

### 1) Inspect available commands
```bash
python -m digests_project.cli.kbctl --help
python -m digests_project.cli.kbctl compute --help
python -m digests_project.cli.kbctl publish --help
