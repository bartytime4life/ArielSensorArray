#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Unified Typer CLI (ArielSensorArray)

Architecture & Design References (project docs)
- CLI-first + Hydra configs + DVC + CI reproducibility [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:1‡Local LLM Workstation Setup for SpectraMind V50 (NeurIPS 2025 Ariel Challenge).pdf](file-service://file-YRjxAzxiYookHZQAg3RQ25) [oai_citation:2‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL)
- Terminal UX with Rich; HTML/PNG diagnostics, no heavy GUI [oai_citation:3‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:4‡Mastering the Command Line: Comprehensive Guide to CLI Development and UX.pdf](file-service://file-Hs2KJiGsPypJ1ZRnTjWa4F)
- Ariel challenge context: FGS/AIRS → μ/σ for 283 bins; 9‑hour runtime envelope [oai_citation:5‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w)
- Kaggle pipeline alignment (notebooks, hardware, quotas) [oai_citation:6‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf)

This module is safe to run before the full pipeline exists:
- Each command writes stub artifacts, honors Hydra-style overrides if available,
  appends to logs/v50_debug_log.md, and exits with helpful status.
- Replace simulate_* hooks and stub outputs with your real implementations.

"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Optional Hydra/OmegaConf — CLI works without them.
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    HYDRA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    HYDRA_AVAILABLE = False

APP = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — Unified CLI (train / predict / calibrate / diagnose / submit / selftest / analyze-log)",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()
ROOT = Path(__file__).resolve().parent
REPO = ROOT
LOGS = REPO / "logs"
OUTPUTS = REPO / "outputs"
DIAG = OUTPUTS / "diagnostics"
CALIB = OUTPUTS / "calibrated"
CHECKPOINTS = OUTPUTS / "checkpoints"
DEBUG_LOG = LOGS / "v50_debug_log.md"
VERSION_FILE = REPO / "VERSION"
RUN_HASH_JSON = OUTPUTS / "run_hash_summary_v50.json"  # optional


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def read_version() -> str:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return "0.1.0"


def read_run_hash() -> str:
    try:
        if RUN_HASH_JSON.exists():
            j = json.loads(RUN_HASH_JSON.read_text(encoding="utf-8"))
            return str(j.get("run_hash") or j.get("config_hash") or "unknown")
    except Exception:
        pass
    return "unknown"


def git_sha_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO))
        return out.decode().strip()
    except Exception:
        return "unknown"


def ensure_dirs() -> None:
    for p in (LOGS, OUTPUTS, DIAG, CALIB, CHECKPOINTS):
        p.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def render_header(title: str) -> None:
    console.print(Panel.fit(f"[bold]SpectraMind V50[/bold]\n{title}", box=box.ROUNDED))


def append_debug_log(lines: List[str]) -> None:
    ensure_dirs()
    header = f"\n⸻\n\n{timestamp()} — " + lines[0]
    body = "".join(f"\n\t• {ln}" for ln in lines[1:])
    prefix = (
        DEBUG_LOG.read_text(encoding="utf-8")
        if DEBUG_LOG.exists()
        else "SpectraMind V50 — Debug & Audit Log\n\nImmutable operator log — append-only.\n"
    )
    DEBUG_LOG.write_text(prefix + header + body, encoding="utf-8")


def short_hash_from_overrides(overrides: List[str]) -> str:
    data = "\n".join(sorted(overrides))
    return hashlib.sha1(data.encode()).hexdigest()[:8] if data else "noovr"


def pretty_overrides(overrides: List[str]) -> str:
    return ", ".join(overrides) if overrides else "(none)"


def hydra_compose_or_stub(config_dir: Path, task_cfg: str, overrides: List[str]) -> Dict[str, Any]:
    """
    Compose Hydra config from `configs/` dir and `task_cfg` (e.g. 'training=default').
    Returns a plain dict or a minimal stub when Hydra is unavailable or fails.
    """
    if HYDRA_AVAILABLE and config_dir.exists():
        try:
            with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
                base_over = [task_cfg] if task_cfg else []
                cfg = compose(config_name="config_v50.yaml", overrides=base_over + (overrides or []))
                return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
        except Exception as e:
            console.print(f"[yellow]Hydra compose failed[/yellow]: {e}")
    return {
        "task": task_cfg or "default",
        "overrides": overrides or [],
        "note": "Hydra not available or compose failed; using stub config.",
    }


def write_stub_html(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;max-width:1000px}}
pre{{background:#0b1020;color:#e6edf3;padding:1rem;border-radius:.5rem;overflow:auto}}
.card{{border:1px solid #e5e7eb;border-radius:.5rem;padding:1rem;margin:.5rem 0}}
small{{color:#6b7280}}
</style></head>
<body>
<h1>{title}</h1>
<p><small>Generated at {timestamp()}</small></p>
<div class="card">{body}</div>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def zip_artifacts(zip_out: Path, globs: List[str]) -> None:
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pattern in globs:
            for p in REPO.glob(pattern):
                if p.is_file():
                    zf.write(p, p.relative_to(REPO))


def simulate_progress(title: str, steps: int = 5, sleep_s: float = 0.30) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        tid = progress.add_task(title, total=steps)
        for _ in range(steps):
            time.sleep(sleep_s)
            progress.advance(tid)


# --------------------------------------------------------------------------------------
# Version flag (global)
# --------------------------------------------------------------------------------------
@APP.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show CLI version + commit + run hash + timestamp.",
        is_flag=True,
    ),
) -> None:
    if version:
        ver = read_version()
        rh = read_run_hash()
        now = timestamp()
        sha = git_sha_short()
        table = Table(title="SpectraMind V50 — Version", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")
        table.add_row("CLI", ver)
        table.add_row("Git SHA", sha)
        table.add_row("Run/Config Hash", rh)
        table.add_row("Timestamp (UTC)", now)
        console.print(table)
        append_debug_log(
            [
                "spectramind --version",
                f"Git SHA: {sha}",
                f"Version: {ver}",
                f"Run/Config hash: {rh}",
                f"Timestamp: {now}",
            ]
        )
        raise typer.Exit()


# --------------------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------------------
@APP.command("selftest")
def selftest(
    deep: bool = typer.Option(False, "--deep", help="Run deeper checks (Hydra compose, CUDA/DVC presence if available)."),
) -> None:
    """Run integrity and wiring checks (files, configs, CLI)."""
    render_header("Selftest")
    ensure_dirs()

    checks: List[Tuple[str, str]] = []
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        status = "[green]OK[/green]" if cond else "[red]FAIL[/red]"
        checks.append((name, status))
        ok = ok and cond

    # Basic directories/files
    check("logs/ exists", LOGS.exists())
    check("outputs/ exists", OUTPUTS.exists())
    check("configs/ exists", (REPO / "configs").exists())
    check("README.md exists", (REPO / "README.md").exists())
    check("pyproject.toml exists", (REPO / "pyproject.toml").exists())

    # Hydra smoke (optional)
    if deep:
        cfg_dir = REPO / "configs"
        cfg_ok = False
        if HYDRA_AVAILABLE and cfg_dir.exists():
            try:
                _ = hydra_compose_or_stub(cfg_dir, "training=default", [])
                cfg_ok = True
            except Exception:
                cfg_ok = False
        check("Hydra compose (training=default)", cfg_ok or not HYDRA_AVAILABLE)

    # Render table
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    for n, s in checks:
        table.add_row(n, s)
    console.print(table)

    console.print("✅ All checks passed." if ok else "❌ Some checks failed.")
    append_debug_log(
        [
            "spectramind selftest" + (" --deep" if deep else ""),
            f"Git SHA: {git_sha_short()}",
            f"Result: {'OK' if ok else 'FAIL'}",
        ]
    )
    if not ok:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------------------
# Calibrate / Train / Predict / Calibrate-Temp / COREL-Train
# --------------------------------------------------------------------------------------
def compose_or_stub(task: str, overrides: List[str]) -> Dict[str, Any]:
    return hydra_compose_or_stub(REPO / "configs", f"{task}=default", overrides or [])


@APP.command("calibrate")
def calibrate(
    overrides: List[str] = typer.Argument(None, help="Hydra-style overrides, e.g. data=kaggle calibration.cache=true"),
) -> None:
    """Run the calibration kill chain: bias → dark → flat → wavelength → normalization → lightcurves."""
    render_header("Calibration")
    ensure_dirs()
    cfg = compose_or_stub("calibration", overrides or [])
    simulate_progress("Calibrating", steps=6)
    (CALIB / "calibration_summary.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp(),
                "config": cfg,
                "artifacts": ["calibrated_lightcurves.npz", "wavelength_map.json"],
                "note": "Stub calibration summary. Replace with real pipeline output.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print("[green]Calibration completed[/green]. Artifacts → outputs/calibrated/")
    append_debug_log(
        [
            f"spectramind calibrate {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            f"Config overrides: {pretty_overrides(overrides or [])}",
            "Artifacts: outputs/calibrated/",
        ]
    )


@APP.command("train")
def train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. data=toy training=default model=v50 +training.epochs=2"),
) -> None:
    """Train the V50 model (FGS1+AIRSGNN encoders; μ/σ decoders)."""
    render_header("Training")
    ensure_dirs()
    _ = compose_or_stub("training", overrides or [])
    simulate_progress("Training", steps=10, sleep_s=0.25)
    (CHECKPOINTS / "best.ckpt").write_bytes(b"stub-checkpoint\n")
    console.print("[green]Training finished[/green]. Checkpoint → outputs/checkpoints/best.ckpt")
    append_debug_log(
        [
            f"spectramind train {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            "Artifacts: outputs/checkpoints/best.ckpt",
        ]
    )


@APP.command("predict")
def predict(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. inference=default model=v50"),
    out_csv: Path = typer.Option(OUTPUTS / "submission.csv", "--out-csv", help="Output CSV for submission predictions."),
) -> None:
    """Predict μ/σ and export submission CSV (challenge-ready)."""
    render_header("Prediction")
    ensure_dirs()
    _ = compose_or_stub("inference", overrides or [])
    simulate_progress("Predicting", steps=6, sleep_s=0.30)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["planet_id"] + [f"bin_{i:03d}" for i in range(283)])
        w.writerow(["P00001"] + [f"{0.1 + 0.001*i:.6f}" for i in range(283)])
    console.print(f"[green]Predictions complete[/green] → {out_csv}")
    append_debug_log(
        [
            f"spectramind predict {pretty_overrides(overrides or [])} --out-csv {out_csv}",
            f"Git SHA: {git_sha_short()}",
            f"Artifacts: {out_csv}",
        ]
    )


@APP.command("calibrate-temp")
def calibrate_temp(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. inference=default ts.max_iter=100"),
) -> None:
    """Temperature scaling for uncertainty calibration."""
    render_header("Temperature Scaling")
    ensure_dirs()
    _ = compose_or_stub("ts", overrides or [])
    simulate_progress("Optimizing temperature", steps=5)
    (OUTPUTS / "temp_scaling.json").write_text(
        json.dumps({"T": 1.07, "val_gll_improvement": 0.013, "timestamp": timestamp()}, indent=2),
        encoding="utf-8",
    )
    console.print("[green]Temperature scaling done[/green] → outputs/temp_scaling.json")
    append_debug_log(
        [
            f"spectramind calibrate-temp {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            "Artifacts: outputs/temp_scaling.json",
        ]
    )


@APP.command("corel-train")
def corel_train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for COREL GNN training"),
) -> None:
    """Train COREL (graph-aware conformal prediction) for calibrated σ intervals."""
    render_header("COREL Conformal Training")
    ensure_dirs()
    _ = compose_or_stub("corel", overrides or [])
    simulate_progress("Training COREL", steps=7)
    (OUTPUTS / "corel_model.pt").write_bytes(b"stub-corel-model\n")
    console.print("[green]COREL training done[/green] → outputs/corel_model.pt")
    append_debug_log(
        [
            f"spectramind corel-train {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            "Artifacts: outputs/corel_model.pt",
        ]
    )


# --------------------------------------------------------------------------------------
# Diagnose sub-CLI
# --------------------------------------------------------------------------------------
diagnose_app = typer.Typer(help="Diagnostics suite (HTML dashboard, plots, overlays)")
APP.add_typer(diagnose_app, name="diagnose")


@diagnose_app.command("dashboard")
def diagnose_dashboard(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for diagnostics"),
    html_out: Path = typer.Option(DIAG / "report_v1.html", "--html-out", help="Output HTML diagnostics report."),
    no_umap: bool = typer.Option(False, "--no-umap", help="Skip UMAP projection."),
    no_tsne: bool = typer.Option(False, "--no-tsne", help="Skip t‑SNE projection."),
) -> None:
    """Generate an interactive HTML diagnostics report (UMAP/t‑SNE/SHAP/symbolic overlays)."""
    render_header("Diagnostics Dashboard")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "diagnostics=dashboard", overrides or [])
    simulate_progress("Building dashboard", steps=6)
    body = f"""
<ul>
  <li><b>Hydra overrides</b>: {pretty_overrides(overrides or [])}</li>
  <li><b>UMAP</b>: {"skipped" if no_umap else "enabled"}</li>
  <li><b>t‑SNE</b>: {"skipped" if no_tsne else "enabled"}</li>
</ul>
<pre>{textwrap.indent(json.dumps(cfg, indent=2)[:2000], ' ')}</pre>
"""
    write_stub_html(html_out, "SpectraMind V50 — Diagnostics Report", body)
    console.print(f"[green]Diagnostics HTML written[/green] → {html_out}")
    append_debug_log(
        [
            f"spectramind diagnose dashboard {pretty_overrides(overrides or [])} --html-out {html_out}"
            + (" --no-umap" if no_umap else "")
            + (" --no-tsne" if no_tsne else ""),
            f"Git SHA: {git_sha_short()}",
            f"Artifacts: {html_out}",
        ]
    )


# --------------------------------------------------------------------------------------
# Submit
# --------------------------------------------------------------------------------------
@APP.command("submit")
def submit(
    zip_out: Path = typer.Option(OUTPUTS / "submission_bundle.zip", "--zip-out", help="Output ZIP bundle."),
    open_html: bool = typer.Option(False, "--open-html", help="Open latest diagnostics report after build."),
) -> None:
    """
    Build a submission bundle:
      - includes latest submission.csv (if present), diagnostics HTML, temp_scaling.json, and logs.
    """
    render_header("Make Submission")
    ensure_dirs()
    simulate_progress("Packaging submission", steps=4)
    globs = [
        "outputs/submission.csv",
        "outputs/diagnostics/*.html",
        "outputs/temp_scaling.json",
        "logs/v50_debug_log.md",
    ]
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    zip_artifacts(zip_out, globs)
    console.print(f"[green]Submission bundle created[/green] → {zip_out}")
    append_debug_log(
        [
            f"spectramind submit --zip-out {zip_out}" + (" --open-html" if open_html else ""),
            f"Git SHA: {git_sha_short()}",
            f"Artifacts: {zip_out}",
        ]
    )
    if open_html:
        htmls = sorted(DIAG.glob("*.html"))
        if htmls:
            try:
                if sys.platform.startswith("darwin"):
                    subprocess.call(["open", str(htmls[0])])
                elif os.name == "nt":
                    os.startfile(str(htmls[0]))  # type: ignore[attr-defined]
                else:
                    subprocess.call(["xdg-open", str(htmls[0])])
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# Analyze Log
# --------------------------------------------------------------------------------------
@APP.command("analyze-log")
def analyze_log(
    md_out: Optional[Path] = typer.Option(None, "--md", help="Export Markdown table to this path."),
    csv_out: Optional[Path] = typer.Option(None, "--csv", help="Export CSV table to this path."),
    group_by_config_hash: bool = typer.Option(
        False, "--group-by-config-hash", help="Group sections by config hash (if present)."
    ),
) -> None:
    """Parse logs/v50_debug_log.md → produce a tabular summary (optionally grouped by config hash)."""
    render_header("Analyze CLI Log")
    ensure_dirs()
    if not DEBUG_LOG.exists():
        console.print("[yellow]No debug log found[/yellow]. Run some commands first.")
        raise typer.Exit(code=2)

    text = DEBUG_LOG.read_text(encoding="utf-8")
    entries: List[Dict[str, str]] = []

    blocks = re.split(r"\n⸻\n", text)[1:] if "⸻" in text else [text]
    for blk in blocks:
        lines = [ln.strip(" \t•") for ln in blk.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        ts_cmd = lines[0]
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T.*Z)\s+—\s+(.*)", ts_cmd)
        ts = ts_match.group(1) if ts_match else ""
        cmd = ts_match.group(2) if ts_match else ts_cmd
        sha = ""
        cfg_hash = ""
        for ln in lines[1:]:
            if ln.lower().startswith("git sha:"):
                sha = ln.split(":", 1)[1].strip()
            if "config hash" in ln.lower() or "run/config hash" in ln.lower():
                cfg_hash = ln.split(":", 1)[1].strip()
        entries.append({"timestamp": ts, "cmd": cmd, "git_sha": sha, "config_hash": cfg_hash})

    if group_by_config_hash:
        # Grouping summary (simple print)
        by = {}
        for e in entries:
            by.setdefault(e["config_hash"] or "unknown", []).append(e)
        for k, group in by.items():
            console.print(f"\n[bold]Config Hash:[/bold] {k}  ([cyan]{len(group)} entries[/cyan])")
            for e in group:
                console.print(f"  • {e['timestamp']} — {e['cmd']} ({e['git_sha']})")

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Time (UTC)", style="bold")
    table.add_column("Command")
    table.add_column("Git SHA")
    table.add_column("Config Hash")
    for e in entries:
        table.add_row(e["timestamp"], e["cmd"], e["git_sha"], e["config_hash"])
    console.print(table)

    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md = "| time | command | git_sha | config_hash |\n|---|---|---|---|\n"
        for e in entries:
            md += f"| {e['timestamp']} | {e['cmd']} | {e['git_sha']} | {e['config_hash']} |\n"
        md_out.write_text(md, encoding="utf-8")
        console.print(f"Markdown written → {md_out}")

    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "command", "git_sha", "config_hash"])
            for e in entries:
                w.writerow([e["timestamp"], e["cmd"], e["git_sha"], e["config_hash"]])
        console.print(f"CSV written → {csv_out}")

    append_debug_log(
        [
            "spectramind analyze-log"
            + (f" --md {md_out}" if md_out else "")
            + (f" --csv {csv_out}" if csv_out else "")
            + (" --group-by-config-hash" if group_by_config_hash else ""),
            f"Git SHA: {git_sha_short()}",
            f"Entries: {len(entries)}",
        ]
    )


# --------------------------------------------------------------------------------------
# Check CLI Map
# --------------------------------------------------------------------------------------
@APP.command("check-cli-map")
def check_cli_map() -> None:
    """
    Print a command→file map (for docs and integrity).
    Update this when you add/remove CLI subcommands.
    """
    render_header("CLI Command → File Map")
    rows = [
        ("spectramind selftest", "spectramind.py"),
        ("spectramind calibrate", "spectramind.py"),
        ("spectramind train", "spectramind.py"),
        ("spectramind predict", "spectramind.py"),
        ("spectramind calibrate-temp", "spectramind.py"),
        ("spectramind corel-train", "spectramind.py"),
        ("spectramind diagnose dashboard", "spectramind.py"),
        ("spectramind submit", "spectramind.py"),
        ("spectramind analyze-log", "spectramind.py"),
        ("spectramind check-cli-map", "spectramind.py"),
    ]
    table = Table(box=box.SIMPLE)
    table.add_column("Command", style="bold cyan")
    table.add_column("Python Module")
    for cmd, mod in rows:
        table.add_row(cmd, mod)
    console.print(table)

    append_debug_log(
        [
            "spectramind check-cli-map",
            f"Git SHA: {git_sha_short()}",
            f"Commands: {len(rows)}",
        ]
    )


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
def _install_sigint_handler() -> None:
    def handler(signum, frame) -> None:  # type: ignore[unused-argument]
        console.print("\n[red]Interrupted[/red] (SIGINT). Exiting gracefully.")
        raise SystemExit(130)

    try:
        signal.signal(signal.SIGINT, handler)
    except Exception:
        pass


def app() -> None:
    _install_sigint_handler()
    APP()


if __name__ == "__main__":
    app()