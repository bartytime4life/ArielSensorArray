"""Command-line interface for ArielSensorArray."""

from __future__ import annotations

import typer

app = typer.Typer(help="ArielSensorArray CLI")


@app.command()
def selftest(
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without side effects."),
) -> None:
    """Basic smoke test to ensure the CLI is wired correctly."""
    if dry_run:
        typer.echo("Selftest executed (dry run).")
    else:
        typer.echo("Selftest executed.")


if __name__ == "__main__":
    app()
