#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/cli/cli_diagnose.py

SpectraMind V50 — Diagnose CLI (includes `fft-fusion` subcommand)

This Typer-based CLI wires the `tools/generate_fft_symbolic_fusion.py` tool into the
unified `spectramind diagnose` workflow, with audit logging and safe subprocess
execution. You can run it standalone:

  python -m src.cli.cli_diagnose diagnose fft-fusion \
      --mu outputs/predictions/mu.npy \
      --symbolic outputs/diagnostics/symbolic_results.json \
      --entropy outputs/diagnostics/entropy.npy \
      --outdir outputs/fft_symbolic_fusion \
      --n-freq 50

Or, if your repo-level `spectramind.py` mounts this module's `app`, you can do:

  spectramind diagnose fft-fusion ... (same options)

Design notes
------------
• We invoke the tool as a subprocess to keep a single source of truth for its logic.
• We append an immutable audit entry to logs/v50_debug_log.md for every call.
• We validate paths and create `--outdir` if needed before calling the tool.
• All arguments map 1:1 onto `tools/generate_fft_symbolic_fusion.py`.

Requirements
------------
• Typer (CLI), Python 3.10+
• tools/generate_fft_symbolic_fusion.py present in repo

Author: SpectraMind V50 Team
License: MIT
"""

from __future__ import annotations

import sys
import shlex
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer

# -----------------------------------------------------------------------------
# Typer app structure
# -----------------------------------------------------------------------------

app = typer.Typer(name="spectramind", add_completion=False, help="SpectraMind V50 — Unified CLI")

diagnose = typer.Typer(name="diagnose", add_completion=False, help="Diagnostics and fusion tools")
app.add_typer(diagnose, name="diagnose")


# -----------------------------------------------------------------------------
# Utilities: filesystem + audit log
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _append_audit(header: str, payload: Dict[str, Any]) -> None:
    log_path = Path("logs") / "v50_debug_log.md"
    _ensure_dir(log_path.parent)
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n### {header} — {now}\n\n")
        for k, v in payload.items():
            f.write(f"- **{k}**: `{v}`\n")


def _python_exe() -> str:
    return sys.executable or "python"


def _fail(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _ok(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.GREEN)


# -----------------------------------------------------------------------------
# diagnose fft-fusion
# -----------------------------------------------------------------------------

@diagnose.command("fft-fusion")
def diagnose_fft_fusion(
    mu: Path = typer.Option(..., exists=True, readable=True, help="Input μ spectra (.npy or .csv), shape [N, 283]."),
    symbolic: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Symbolic results JSON (per-planet)."),
    entropy: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Entropy scores (.npy)."),
    outdir: Path = typer.Option(Path("outputs/fft_symbolic_fusion"), help="Output directory."),
    n_freq: int = typer.Option(50, min=4, max=256, help="Number of FFT frequencies to keep (head of rFFT)."),
    tool_path: Path = typer.Option(Path("tools/generate_fft_symbolic_fusion.py"),
                                   exists=True, readable=True,
                                   help="Path to the fusion tool script."),
    dry_run: bool = typer.Option(False, help="Print the command that would be run, then exit."),
) -> None:
    """
    Run the FFT × Symbolic fusion diagnostics, wiring through to the tool script.
    Produces UMAP/t-SNE interactive HTML and CSV/JSON exports in --outdir.
    """
    # Validate inputs and prepare output dir
    if not tool_path.exists():
        _fail(f"Tool not found at {tool_path}. Expected tools/generate_fft_symbolic_fusion.py")

    _ensure_dir(outdir)

    # Build command
    cmd: List[str] = [
        _python_exe(),
        str(tool_path),
        "--mu", str(mu),
        "--outdir", str(outdir),
        "--n-freq", str(n_freq),
    ]
    if symbolic is not None:
        cmd += ["--symbolic", str(symbolic)]
    if entropy is not None:
        cmd += ["--entropy", str(entropy)]

    # Audit: start
    _append_audit("diagnose.fft-fusion.start", {
        "mu": str(mu),
        "symbolic": str(symbolic) if symbolic else "",
        "entropy": str(entropy) if entropy else "",
        "outdir": str(outdir),
        "n_freq": n_freq,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
    })

    # Dry run?
    if dry_run:
        typer.echo("DRY-RUN: " + " ".join(shlex.quote(c) for c in cmd))
        _ok("No execution performed (dry-run).")
        return

    # Execute
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        # Stream tool stdout for user visibility
        sys.stdout.write(proc.stdout or "")
        if proc.stderr:
            sys.stderr.write(proc.stderr)
    except subprocess.CalledProcessError as e:
        # Audit: error
        _append_audit("diagnose.fft-fusion.error", {
            "returncode": e.returncode,
            "stderr": (e.stderr or "").strip(),
        })
        _fail(f"fft-fusion failed (exit {e.returncode}). See logs/v50_debug_log.md and tool output above.")

    # Audit: end
    _append_audit("diagnose.fft-fusion.end", {
        "outdir": str(outdir),
        "status": "success",
    })
    _ok(f"FFT × Symbolic fusion complete → {outdir}")


# -----------------------------------------------------------------------------
# __main__
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app()