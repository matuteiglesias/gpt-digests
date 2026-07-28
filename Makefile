PY ?= python3
.PHONY: test install smoke contract-release-verify

CONTRACT_RELEASE_MANIFEST ?= interop/vendor/kb-interop.v1-rc1/release.json
CONTRACT_RELEASE_ROOT ?= interop/vendor/kb-interop.v1-rc1

test:
	$(PY) -m pytest -q

install:
	$(PY) -m pip install -e . --no-build-isolation

smoke:
	PYTHONPATH=src $(PY) -m kb_artifacts.cli --help

contract-release-verify:
	$(PY) tools/verify_contract_release.py $(CONTRACT_RELEASE_MANIFEST) --bundle-root $(CONTRACT_RELEASE_ROOT)
