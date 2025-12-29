
# Troubleshooting

### `No events matched; check --logs-glob.`
- Confirm the glob resolves: `ls /your/glob/*.jsonl | head`.
- Use absolute paths if the working directory is different.

### Hydration writes stubs only
- Run a dry-run: `kbctl hydrate-dryrun ... --limit 3 --verbose`
- Compare **index keys** vs **sources**; align formats.
- If Units were built from different raw sets, rebuild Units from the same sources you index.

### `ModuleNotFoundError: bags_pipeline`
- Your package is `digests_project`. Use `from digests_project.bags_pipeline import ...`
- A safety alias in `digests_project/__init__.py` can map `bags_pipeline` to the package path.

### `TypeError: Unit.__init__(...)`
- Your Unit dataclass changed; filter kwargs when constructing Units (you already added `_filter_for_unit`).
- Remove deprecated fields like `unit_ts`/`ts_range` if not in the dataclass.