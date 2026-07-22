PY ?= python3
.PHONY: test install smoke

test:
	$(PY) -m pytest -q

install:
	$(PY) -m pip install -e . --no-build-isolation

smoke:
	PYTHONPATH=src $(PY) -m kb_artifacts.cli --help
