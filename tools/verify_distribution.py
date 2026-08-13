#!/usr/bin/env python3
"""Build and smoke-test kb-artifacts as an installed wheel."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def verify_distribution(dist_dir: Path) -> Path:
    """Build, install, import, and exercise the wheel outside the checkout."""
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    _run([sys.executable, "-m", "build", "--outdir", str(dist_dir)], cwd=ROOT)

    wheels = sorted(dist_dir.glob("kb_artifacts-*.whl"))
    sdists = sorted(dist_dir.glob("kb_artifacts-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("expected exactly one kb-artifacts wheel and one sdist")

    with tempfile.TemporaryDirectory(prefix="kb-artifacts-distribution-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        cli = scripts / ("kb-artifact.exe" if os.name == "nt" else "kb-artifact")

        _run([str(python), "-m", "pip", "install", str(wheels[0])], cwd=work)
        _run(
            [
                str(python),
                "-I",
                "-c",
                "from kb_artifacts import SelectionRequest, inspect_source, select",
            ],
            cwd=work,
        )
        _run([str(cli), "--help"], cwd=work)

        evidence = work / "evidence.jsonl"
        evidence.write_text(
            '{"title":"Deploy app","text":"Deploy service","tags":["runbook"]}\n'
            '{"title":"Buy groceries","text":"Milk","tags":["personal"]}\n',
            encoding="utf-8",
        )
        output = work / "selected"
        _run(
            [
                str(cli),
                "select",
                "--chunk-glob",
                str(evidence),
                "--tag",
                "runbook",
                "--output",
                str(output),
            ],
            cwd=work,
        )
        expected = {"selected.jsonl", "selected.csv", "artifact.md", "manifest.json"}
        if {path.name for path in output.iterdir()} != expected:
            raise RuntimeError("installed CLI produced an unexpected output file set")

    return wheels[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=ROOT / "dist",
        help="distribution output directory (default: repository dist/)",
    )
    arguments = parser.parse_args()
    wheel = verify_distribution(arguments.dist_dir.resolve())
    print(f"DISTRIBUTION VERIFIED: {wheel}")


if __name__ == "__main__":
    main()
