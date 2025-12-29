
# Hydration

Hydration resolves `Unit.sources` (like `("event","<id>")` or `("session","<id>")`) into snippets.

- Build indices:
  - Events: `kbctl hydrate-dryrun --logs-glob "..."`
  - Sessions: same with `--sess-glob "..."`
- Run with `--hydrate` in `l2-build` to embed rendered snippets.

## Common pitfalls

- **Key mismatch**: Your events index prints a sample key (e.g. `61ae249d0af50e12`), but your Unit sources use UUIDs (e.g. `9e419d7d0daca797`). Make sure the source IDs match what `build_event_index()` keys on.
- **Empty coverage**: Start with a small subset using `kbctl units-select` then hydrate, to validate the pipeline before scaling.
