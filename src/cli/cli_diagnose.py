#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/cli/cli_diagnose.py

SpectraMind V50 — Diagnose CLI (fft-fusion + utilities)

Upgrades in this version
------------------------
• Live streaming of the subprocess output (no "silent stall" while the tool runs)
• Robust audit logging (command, git hash, python path, duration, exit code)
• Extra safety & UX flags: --dry-run, --timeout, --verbose/--quiet, --save-cmd, --open-html
• More validations (input file types, write perms) and clearer error messages
• Optional passthrough of extra args to the tool (--tool-args "...") for future growth
• Single source of truth: still calls tools/generate_fft_symbolic_fusion.py as a subprocess

Usage (standalone)
------------------
python -m src.cli.cli_diagnose diagnose fft-fusion \
  --mu outputs/predictions/mu.npy \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --entropy outputs/diagnostics/entropy.npy \
  --outdir outputs/fft_symbolic_fusion \
  --n-freq 50 \
  --verbose

Or, if your repo-level `spectramind.py` mounts this module's `app`, then:
spectramind diagnose fft-fusion [options]

Requirements
------------
• Python 3.10+
• Typer installed
• tools/generate_fft_symbolic_fusion.py present and runnable

License
-------
MIT — SpectraMind V50 Team
"""

from __future__ import annotations

import os
import sys
import shlex
import time
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import typer

# ======================================================================================
# Typer app structure
# ======================================================================================

app = typer.Typer(name="spectramind", add_completion=False, help="SpectraMind V50 — Unified CLI")

diagnose = typer.Typer(name="diagnose", add_completion=False, help="Diagnostics and fusion tools")
app.add_typer(diagnose, name="diagnose")


# ======================================================================================
# Utilities
# ======================================================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _git_info() -> Dict[str, str]:
    """Best-effort git info for audit."""
    def _cmd(args: List[str]) -> str:
        try:
            out = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
            return out
        except Exception:
            return ""
    return {
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _cmd(["git", "status", "--porcelain"]),
    }


def _append_audit(header: str, payload: Dict[str, Any]) -> None:
    log_path = Path("logs") / "v50_debug_log.md"
    _ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n### {header} — {_now_str()}\n\n")
        for k, v in payload.items():
            f.write(f"- **{k}**: `{v}`\n")


def _python_exe() -> str:
    return sys.executable or "python"


def _fail(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _warn(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.YELLOW)


def _ok(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.GREEN)


def _info(msg: str, quiet: bool) -> None:
    if not quiet:
        typer.secho(msg, fg=typer.colors.CYAN)


def _validate_readable(p: Path, kind: str) -> None:
    if not p.exists():
        _fail(f"{kind} not found: {p}")
    if not os.access(p, os.R_OK):
        _fail(f"{kind} not readable: {p}")


def _validate_outdir(d: Path) -> None:
    _ensure_dir(d)
    test_file = d / ".write_test"
    try:
        with test_file.open("w", encoding="utf-8") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        _fail(f"Output directory not writable: {d}  ({e})")


def _save_cmd_script(outdir: Path, cmd: List[str]) -> Path:
    """Save a reproducible shell script with the exact command."""
    _ensure_dir(outdir)
    script = outdir / "reproduce_fft_fusion.sh"
    with script.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    os.chmod(script, 0o755)
    return script


def _open_html_if_exists(outdir: Path, filenames: Tuple[str, ...], quiet: bool) -> None:
    for name in filenames:
        p = outdir / name
        if p.exists():
            try:
                if sys.platform.startswith("darwin"):
                    subprocess.Popen(["open", str(p)])
                elif os.name == "nt":
                    os.startfile(str(p))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(p)])
                _info(f"Opened: {p}", quiet)
                return
            except Exception:
                _warn(f"Could not open file automatically: {p}")


def _run_with_live_output(cmd: List[str], timeout: Optional[int], quiet: bool) -> Tuple[int, str]:
    """
    Run a command streaming stdout/stderr live to the console.
    Returns (exit_code, last_lines) where last_lines is a compact tail for audit.
    """
    _info("Exec: " + " ".join(shlex.quote(c) for c in cmd), quiet)
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    last_lines: List[str] = []
    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                # Stream live
                sys.stdout.write(line)
                # Keep a small tail buffer for audit
                last_lines.append(line.rstrip())
                if len(last_lines) > 50:
                    last_lines.pop(0)
            elif proc.poll() is not None:
                break
            # Timeout handling
            if timeout is not None and (time.time() - start) > timeout:
                proc.send_signal(signal.SIGINT)
                time.sleep(0.5)
                if proc.poll() is None:
                    proc.kill()
                return (124, "\n".join(last_lines))
        return (proc.returncode or 0, "\n".join(last_lines))
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
            time.sleep(0.5)
            if proc.poll() is None:
                proc.kill()
        finally:
            _fail("Interrupted by user (SIGINT).")
    except Exception as e:
        try:
            if proc.poll() is None:
                proc.kill()
        finally:
            _fail(f"Subprocess failed to start or crashed: {e}")
    # Unreachable
    return (1, "\n".join(last_lines))


def _detect_file_kind(p: Path, expected_suffixes: Tuple[str, ...]) -> None:
    if not any(str(p).lower().endswith(s) for s in expected_suffixes):
        _warn(f"Unexpected file extension for {p}. Expected one of: {', '.join(expected_suffixes)}")


# ======================================================================================
# diagnose fft-fusion
# ======================================================================================

@diagnose.command("fft-fusion")
def diagnose_fft_fusion(
    mu: Path = typer.Option(..., exists=True, readable=True, help="Input μ spectra (.npy or .csv), shape [N, 283]."),
    symbolic: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Symbolic results JSON (per-planet)."),
    entropy: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Entropy scores (.npy)."),
    outdir: Path = typer.Option(Path("outputs/fft_symbolic_fusion"), help="Output directory for artifacts."),
    n_freq: int = typer.Option(50, min=4, max=512, help="Number of FFT frequencies to keep (head of rFFT)."),
    tool_path: Path = typer.Option(Path("tools/generate_fft_symbolic_fusion.py"),
                                   exists=True, readable=True,
                                   help="Path to the fusion tool script."),
    python_exe: Optional[Path] = typer.Option(None, help="Explicit Python interpreter for the tool (default: current)."),
    timeout: Optional[int] = typer.Option(None, help="Timeout in seconds (kills the tool if exceeded)."),
    tool_args: Optional[str] = typer.Option(None, help="Extra args passed verbatim to the tool (quoted string)."),
    save_cmd: bool = typer.Option(True, help="Save a reproduce shell script in --outdir."),
    open_html: bool = typer.Option(False, help="Try to open the produced HTML (UMAP/t-SNE) after success."),
    dry_run: bool = typer.Option(False, help="Print the command that would run, then exit."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose CLI messages."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce CLI chatter."),
) -> None:
    """
    Run the FFT × Symbolic fusion diagnostics via the tool script.
    Produces UMAP/t-SNE interactive HTML and CSV/JSON exports in --outdir.
    """
    # Normalize flags
    if verbose and quiet:
        _fail("Cannot use --verbose and --quiet together.")
    _validate_readable(tool_path, "Tool")
    _validate_readable(mu, "μ spectra")
    if symbolic:
        _validate_readable(symbolic, "Symbolic JSON")
    if entropy:
        _validate_readable(entropy, "Entropy array")
    _validate_outdir(outdir)

    # Friendly file-kind hints
    _detect_file_kind(mu, (".npy", ".csv"))
    if entropy:
        _detect_file_kind(entropy, (".npy",))
    if symbolic:
        _detect_file_kind(symbolic, (".json",))

    # Prepare command
    py = str(python_exe or _python_exe())
    cmd: List[str] = [
        py,
        str(tool_path),
        "--mu", str(mu),
        "--outdir", str(outdir),
        "--n-freq", str(n_freq),
    ]
    if symbolic is not None:
        cmd += ["--symbolic", str(symbolic)]
    if entropy is not None:
        cmd += ["--entropy", str(entropy)]
    if tool_args:
        # passthrough: split respecting quotes
        cmd += shlex.split(tool_args)

    # Audit: start
    git = _git_info()
    audit_start = {
        "mu": str(mu),
        "symbolic": str(symbolic) if symbolic else "",
        "entropy": str(entropy) if entropy else "",
        "outdir": str(outdir),
        "n_freq": n_freq,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "python": py,
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
    }
    _append_audit("diagnose.fft-fusion.start", audit_start)

    # Dry run?
    if dry_run:
        typer.echo("DRY-RUN: " + " ".join(shlex.quote(c) for c in cmd))
        _ok("No execution performed (dry-run).")
        return

    # Save reproduce script (optional)
    script_path: Optional[Path] = None
    if save_cmd:
        script_path = _save_cmd_script(outdir, cmd)
        _info(f"Saved reproduce script: {script_path}", quiet)

    # Execute with live output
    t0 = time.time()
    exit_code, tail = _run_with_live_output(cmd, timeout=timeout, quiet=quiet)
    dt = time.time() - t0

    # Audit: end
    _append_audit("diagnose.fft-fusion.end", {
        "status": "success" if exit_code == 0 else f"failure:{exit_code}",
        "duration_sec": f"{dt:.2f}",
        "tail": tail.replace("`", "'")[:4000],  # avoid md backticks and keep audit compact
        "reproduce_script": str(script_path) if script_path else "",
    })

    if exit_code != 0:
        _fail(f"fft-fusion failed (exit {exit_code}). See logs/v50_debug_log.md for details.")

    _ok(f"FFT × Symbolic fusion complete → {outdir}")

    # Convenience: try to open outputs
    if open_html:
        _open_html_if_exists(outdir, ("fusion_plot_umap.html", "fusion_plot_tsne.html"), quiet)


# ======================================================================================
# __main__
# ======================================================================================

if __name__ == "__main__":
    app()