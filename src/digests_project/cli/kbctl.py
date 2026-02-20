# digests_project/cli/kbctl.py
from __future__ import annotations

import typer

from digests_project.cli.kbctl_compute import app as compute_app
from digests_project.cli.kbctl_publish import app as publish_app

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(compute_app, name="compute")
app.add_typer(publish_app, name="publish")

def main() -> None:
    app()

if __name__ == "__main__":
    main()