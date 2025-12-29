
# Quickstart

## 0) Install

```bash
pip install -e ".[docs]"
kbctl --help
```

## 1) Build Units from logs & sessions

```bash
kbctl bags-logs     --glob "/abs/path/to/logs/*.jsonl"    --out runs/units_logs.jsonl
kbctl bags-sessions --glob "/abs/path/to/sessions/*.jsonl" --out runs/units_sessions.jsonl
kbctl bags-merge    --inputs runs/units_logs.jsonl --inputs runs/units_sessions.jsonl \
                    --out runs/units_all.jsonl
```

## 2) Build pair/tag bags (optional but powerful)

```bash
kbctl bags-pairs-from-units --units-jsonl runs/units_all.jsonl --top-n 80 --min-docs 5 --out runs/units_pairs.jsonl
kbctl bags-tags-from-units  --units-jsonl runs/units_all.jsonl --top-k-tags 60 --min-docs 30 --out runs/units_tags.jsonl
kbctl bags-merge --inputs runs/units_all.jsonl --inputs runs/units_pairs.jsonl --inputs runs/units_tags.jsonl --out runs/units_all_plus_bags.jsonl
```

## 3) Hydrate coverage (dry-run check)

```bash
kbctl hydrate-dryrun \
  --units-jsonl runs/units_all_plus_bags.jsonl \
  --logs-glob "/path/to/logs/*.jsonl" \
  --sess-glob "/path/to/sessions/*.jsonl" \
  --limit 5 --verbose
```

* If you see `0 events + 0 sessions`, your `sources` don’t match the index keys. Use `--verbose` and see the sample keys the tool prints; reconcile the ID formats.

## 4) Build L2 digests

```bash
kbctl l2-build \
  --units-jsonl runs/units_all_plus_bags.jsonl \
  --channels "journal memo" \
  --layout onefile \
  --filename-scheme slug \
  --hydrate \
  --logs-glob "/path/to/logs/*.jsonl" \
  --sess-glob "/path/to/sessions/*.jsonl" \
  --out-base ./digests
```

Outputs land under `digests/L2/{unit_type}/{channel}/...`.

## 5) Index & publish

```bash
kbctl index-l2 --digests-root ./digests/L2 --out-json ./index/l2_by_window.json
kbctl publish --digests-root ./digests/L2 --out-dir ./digests/_published
```

That’s it!

 