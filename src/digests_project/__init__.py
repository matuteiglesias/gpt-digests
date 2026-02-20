# Alias 'bags_pipeline' → 'digests_project.bags_pipeline.compute' for backward compat
import sys as _sys
from importlib import import_module as _im
try:
    _bp = _im("digests_project.bags_pipeline.compute")
    _sys.modules.setdefault("bags_pipeline", _bp)
except Exception:
    pass
