from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def test_progressive_agent_example_runs_offline_and_deterministically() -> None:
    command = [sys.executable, "examples/agent/progressive_query.py"]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    first = subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True).stdout
    assert first == second
    assert "records: 3" in first
    assert "candidate count: 2" in first
    assert "selected: 1" in first
