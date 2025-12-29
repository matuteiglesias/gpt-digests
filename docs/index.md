# GPT Digests

A small pipeline to turn raw chat logs & session JSONL into publishable Markdown/MDX digests:

- **Ingest** logs/sessions → normalize into *Events* and *Sessions*
- **Unitize** into *Units* (cohort/day, session) and *Bags* (pairbag/tagbag)
- **Build L2** digests (memo/journal/etc.), optionally **hydrate** with resolved sources
- **Index & Publish** higher levels (L3/L4/L5) and copy artifacts for distribution

## What you’ll use most

- `kbctl bags-logs` / `kbctl bags-sessions` / `kbctl bags-merge`
- `kbctl bags-pairs-from-units` + `kbctl bags-tags-from-units`
- `kbctl l2-build --hydrate`
- `kbctl hydrate-dryrun` to sanity-check coverage
- `kbctl index-l2` + `kbctl publish`

Jump to the [Quickstart](guide/quickstart.md) to run end-to-end in minutes.
