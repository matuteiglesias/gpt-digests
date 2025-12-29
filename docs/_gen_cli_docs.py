# docs/_gen_cli_docs.py
from __future__ import annotations
import sys
from pathlib import Path
import click
from typer.main import get_command

# Ensure repo root on sys.path so imports work when building docs
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your Typer app
from digests_project.cli.kbctl import app  # noqa: E402

OUT = ROOT / "docs" / "cli.md"


def render_help(cmd: click.BaseCommand, prog_name: str) -> str:
    """Return the --help text of a Click/Typer command as a string."""
    ctx = click.Context(cmd, info_name=prog_name)
    return cmd.get_help(ctx)


def main() -> None:
    root = get_command(app)         # Click Group
    md = []
    md.append("# `kbctl`")
    md.append("")
    md.append("```")
    md.append(render_help(root, "kbctl").rstrip())
    md.append("```")
    md.append("")

    # Subcommands
    for name, sub in sorted(root.commands.items()):
        md.append(f"## `kbctl {name}`")
        md.append("")
        md.append("```")
        md.append(render_help(sub, f"kbctl {name}").rstrip())
        md.append("```")
        md.append("")

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()