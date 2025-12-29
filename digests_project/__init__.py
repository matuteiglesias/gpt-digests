# Alias 'bags_pipeline' → 'digests_project.bags_pipeline' for backward compat
import sys as _sys
from importlib import import_module as _im
try:
    _bp = _im("digests_project.bags_pipeline")
    _sys.modules.setdefault("bags_pipeline", _bp)
except Exception:
    pass
