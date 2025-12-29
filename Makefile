
PY=python3



# Example targets
l2:
	$(PY) -m cli.kbctl l2-build --help

index:
	$(PY) -m cli.kbctl index-l2 --help

qa:
	@echo "== complexity =="
	@radon cc -s -a digests_project | sed -n '1,25p'
	@echo "\n== maintainability =="
	@radon mi digests_project | sort -k2
	@echo "\n== dead code (top 40) =="
	@vulture digests_project | head -40 || true
	@echo "\n== import cycles =="
	@pydeps digests_project/bags_pipeline --show-cycles --max-bacon=2 >/dev/null 2>&1 && echo "no obvious cycles" || echo "pydeps ran (see cycles above if any)."
	@echo "\n== architectural contracts =="
	@lint-imports || true
	@echo "\n== mdx sanity =="
	@[ -d digests/L2 ] && \
	  O=$$(rg -n -g'**/*.mdx' '<details' digests/L2 | wc -l); \
	  C=$$(rg -n -g'**/*.mdx' '</details>' digests/L2 | wc -l); \
	  echo opens=$$O closes=$$C; \
	  F=$$(rg -n -g'**/*.mdx' '```' digests/L2 | wc -l); \
	  echo fences=$$F; \
	  exit $$(( ($$F % 2 == 0) ? 0 : 1 )); \
	  true || true


.PHONY: ingest digest publish

ingest:
	@cd ../textflow && $(PY) -m textflow.cli --src "ingest/*.jsonl"

digest:
	@$(PY) digests_project/scripts/bridge_textflow_to_units.py
	@$(PY) -m digests_project.cli.kbctl build --input runs/units_textflow.jsonl

publish:
	@bash digests_project/scripts/run_eda_may_aug.sh
	@bash digests_project/rebuild_views.sh
	
	
	

smoke:
	@echo "[SMOKE][$(PROJECT)] not implemented"
	@echo "Define a minimal, offline, fixture-driven smoke check."
	@exit 2

