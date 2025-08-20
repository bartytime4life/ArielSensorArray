#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Unified Typer CLI (ArielSensorArray)

Mission:
  • CLI-first orchestration with friendly Rich output
  • Safe stub artifacts for every command (works before full pipeline is wired)
  • Append-only operator audit log: logs/v50_debug_log.md
  • Optional Hydra compose snapshot to outputs/config_snapshot.yaml

References (engineering context only):
  - CLI-first, Hydra, DVC, CI reproducibility
  - Kaggle runtime envelope (≤9h), GPU quotas
  - Terminal UX (Rich), HTML diagnostics stubs
  - Physics-informed mindset for guardrails (e.g., log hygiene)

This CLI is intentionally safe:
  - Never deletes user data
  - Creates outputs/… folders and placeholder artifacts so Makefile/CI can proceed
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# -----------------------------------------------------------------------------
# Optional Hydra
# -----------------------------------------------------------------------------
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    HYDRA_AVAILABLE = True
except Exception:  # pragma: no cover
    HYDRA_AVAILABLE = False

# -----------------------------------------------------------------------------
# App & paths
# -----------------------------------------------------------------------------
APP = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — Unified CLI (calibrate/train/predict/diagnose/submit/selftest/analyze-log/check-cli-map)",
    add_completion=True,
    no_args_is_help=True,
)
console = Console()

REPO = Path(__file__).resolve().parent
ROOT = REPO  # alias
OUTPUTS = REPO / "outputs"
LOGS = REPO / "logs"
DIAG = OUTPUTS / "diagnostics"
CALIB = OUTPUTS / "calibrated"
CHECKPOINTS = OUTPUTS / "checkpoints"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSION = OUTPUTS / "submission"
SUBMISSION_ZIP = SUBMISSION / "bundle.zip"

DEBUG_LOG = LOGS / "v50_debug_log.md"
VERSION_FILE = REPO / "VERSION"
RUN_HASH_JSON = OUTPUTS / "run_hash_summary_v50.json"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def timestamp() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dirs() -> None:
    for p in (OUTPUTS, LOGS, DIAG, CALIB, CHECKPOINTS, PREDICTIONS, SUBMISSION):
        p.mkdir(parents=True, exist_ok=True)


def git_sha_short() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "0.1.0"


def read_run_hash() -> str:
    try:
        if RUN_HASH_JSON.exists():
            j = json.loads(RUN_HASH_JSON.read_text(encoding="utf-8"))
            return str(j.get("run_hash") or j.get("config_hash") or "unknown")
    except Exception:
        pass
    return "unknown"


def render_header(title: str) -> None:
    console.print(Panel.fit(f"[bold]SpectraMind V50[/bold]\n{title}", box=box.ROUNDED))


def append_debug_log(lines: Iterable[str]) -> None:
    ensure_dirs()
    lines = list(lines)
    header = f"\n⸻\n\n{timestamp()} — {lines[0] if lines else ''}"
    body = "".join(f"\n\t• {ln}" for ln in lines[1:])
    prefix = (
        DEBUG_LOG.read_text(encoding="utf-8")
        if DEBUG_LOG.exists()
        else "SpectraMind V50 — Debug & Audit Log\n\nAppend-only operator log (immutable).\n"
    )
    DEBUG_LOG.write_text(prefix + header + body, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_stub_html(path: Path, title: str, body_html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;max-width:1100px}}
pre{{background:#f6f8fa;padding:1rem;overflow-x:auto}}
code{{font-family:ui-monospace,Consolas,Menlo,monospace}}
hr{{border:0;border-top:1px solid #ddd;margin:2rem 0}}
.small{{color:#666}}
</style></head>
<body>
<h1>{title}</h1>
<div class="small">{timestamp()} • stub HTML</div>
<hr/>
{body_html}
</body></html>"""
    path.write_text(html, encoding="utf-8")


def simulate_progress(title: str, steps: int = 5, sleep_s: float = 0.25) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        tid = progress.add_task(title, total=steps)
        for _ in range(steps):
            time.sleep(sleep_s)
            progress.advance(tid)


def hydra_compose_or_stub(config_dir: Path, task_cfg: str, overrides: Optional[List[str]]) -> Dict[str, Any]:
    """
    Tries to compose config; if Hydra is unavailable or compose fails, returns a stub dict.
    Also writes a snapshot YAML under outputs/config_snapshot.yaml when possible.
    """
    if HYDRA_AVAILABLE and config_dir.exists():
        try:
            with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
                cfg = compose(config_name="config_v50.yaml", overrides=[task_cfg] + (overrides or []))
                snap = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
                write_text(OUTPUTS / "config_snapshot.yaml", OmegaConf.to_yaml(cfg))
                return snap
        except Exception as e:  # pragma: no cover
            console.print(f"[yellow]Hydra compose failed[/yellow]: {e}")
    return {"task": task_cfg, "overrides": overrides or [], "note": "Hydra unavailable; stub config."}


def zip_paths(zip_path: Path, paths: List[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists():
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(REPO))
                else:
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            zf.write(sub, arcname=sub.relative_to(REPO))


# -----------------------------------------------------------------------------
# Global --version
# -----------------------------------------------------------------------------
@APP.callback(invoke_without_command=False)
def _root_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", is_flag=True, help="Show CLI version & hashes"),
) -> None:
    if version:
        ver, sha, rh, now = read_version(), git_sha_short(), read_run_hash(), timestamp()
        t = Table(title="SpectraMind V50 — Version", box=box.MINIMAL_DOUBLE_HEAD)
        t.add_row("CLI", ver)
        t.add_row("Git SHA", sha)
        t.add_row("Run/Config Hash", rh)
        t.add_row("Timestamp (UTC)", now)
        console.print(t)
        append_debug_log(["spectramind --version", f"Git SHA: {sha}", f"Version: {ver}", f"Run/Config hash: {rh}"])
        raise typer.Exit()


# -----------------------------------------------------------------------------
# Selftest
# -----------------------------------------------------------------------------
@APP.command("selftest")
def selftest(
    deep: bool = typer.Option(False, "--deep", help="Check Hydra + DVC + CUDA availability.")
) -> None:
    """Fast environment & paths sanity with optional deeper checks."""
    render_header("Selftest")
    ensure_dirs()

    checks: List[Tuple[str, bool]] = []
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        checks.append((name, cond))
        ok &= cond

    # Basic presence
    check("logs dir exists", LOGS.exists())
    check("outputs dir exists", OUTPUTS.exists())
    check("configs dir present", (REPO / "configs").exists())
    check("README present", (REPO / "README.md").exists())

    # Optional deeper checks
    if deep:
        check("Hydra importable", HYDRA_AVAILABLE)
        check("DVC present (.dvc/)", (REPO / ".dvc").exists())
        # GPU presence (best-effort)
        try:
            subprocess.check_output(["nvidia-smi"])
            gpu_ok = True
        except Exception:
            gpu_ok = False
        check("CUDA (nvidia-smi)", gpu_ok)

    # Render table
    tb = Table(box=box.SIMPLE_HEAVY)
    tb.add_column("Check")
    tb.add_column("Status")
    for name, cond in checks:
        tb.add_row(name, "[green]OK[/green]" if cond else "[red]FAIL[/red]")
    console.print(tb)
    console.print("✅ Environment looks good." if ok else "❌ Selftest failed.")
    append_debug_log(["spectramind selftest" + (" --deep" if deep else ""), f"Result: {'OK' if ok else 'FAIL'}"])
    if not ok:
        raise typer.Exit(code=1)


# -----------------------------------------------------------------------------
# Calibrate / Train / Predict / Temp-scale / COREL-train
# -----------------------------------------------------------------------------
@APP.command("calibrate")
def calibrate(
    overrides: List[str] = typer.Argument(None, help="Hydra-style overrides, e.g., data=nominal +calib.version=v1")
) -> None:
    """Run the calibration kill chain (stubbed)."""
    render_header("Calibration")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "calibration=default", overrides)
    simulate_progress("Calibrating", steps=6)
    write_json(CALIB / "calibration_summary.json", {"cfg": cfg, "note": "stub"})
    console.print("[green]Calibration done[/green] → outputs/calibrated")
    append_debug_log(["spectramind calibrate", f"overrides={overrides}"])


@APP.command("calibrate-temp")
def calibrate_temp(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for temperature scaling")
) -> None:
    """Apply temperature scaling to logits/σ (stub)."""
    render_header("Temperature Scaling")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "calibration=temperature", overrides)
    simulate_progress("Calibrating temperature", steps=4)
    write_json(OUTPUTS / "temperature_scaling.json", {"cfg": cfg, "temp": 1.23, "note": "stub"})
    console.print("[green]Temperature scaling complete[/green] → outputs/temperature_scaling.json")
    append_debug_log(["spectramind calibrate-temp", f"overrides={overrides}"])


@APP.command("corel-train")
def corel_train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for COREL conformal training")
) -> None:
    """Train COREL (graph conformal calibration) — stubbed."""
    render_header("COREL Conformal Training")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "uncertainty=corel", overrides)
    simulate_progress("Training COREL", steps=8)
    write_json(OUTPUTS / "corel_model.json", {"cfg": cfg, "coverage": 0.95, "note": "stub"})
    console.print("[green]COREL saved[/green] → outputs/corel_model.json")
    append_debug_log(["spectramind corel-train", f"overrides={overrides}"])


@APP.command("train")
def train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. +training.epochs=1"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device string, e.g., cpu/gpu/cuda:0"),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Write artifacts to this dir (default checkpoints/)"),
) -> None:
    """Train the V50 model (stub)."""
    render_header("Training")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "training=default", overrides)
    simulate_progress(f"Training on {device}", steps=10)
    target_dir = (outdir or CHECKPOINTS)
    target_dir.mkdir(parents=True, exist_ok=True)
    write_text(target_dir / "best.ckpt", "stub-model-weights")
    write_json(target_dir / "train_summary.json", {"cfg": cfg, "device": device, "note": "stub"})
    console.print(f"[green]Training done[/green] → {target_dir}/best.ckpt")
    append_debug_log(["spectramind train", f"device={device}", f"overrides={overrides}", f"outdir={target_dir}"])


@APP.command("predict")
def predict(
    out_csv: Path = typer.Option(OUTPUTS / "submission.csv", "--out-csv", help="Path to write submission CSV"),
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for inference"),
) -> None:
    """Run inference and write a submission CSV (stub)."""
    render_header("Prediction")
    ensure_dirs()
    _ = hydra_compose_or_stub(REPO / "configs", "inference=default", overrides)
    simulate_progress("Predicting", steps=6)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["planet_id"] + [f"bin_{i:03d}" for i in range(283)])
        w.writerow(["P0001"] + [round(0.1 + i * 1e-3, 6) for i in range(283)])
    console.print(f"[green]CSV written[/green] → {out_csv}")
    append_debug_log(["spectramind predict", f"out_csv={out_csv}", f"overrides={overrides}"])


# -----------------------------------------------------------------------------
# Diagnose (group)
# -----------------------------------------------------------------------------
DIAG_APP = typer.Typer(help="Diagnostics subcommands (smoothness, dashboard)")
APP.add_typer(DIAG_APP, name="diagnose")


@DIAG_APP.command("smoothness")
def diag_smoothness(
    outdir: Path = typer.Option(DIAG, "--outdir", help="Output directory for smoothness artifacts")
) -> None:
    """Generate a stub smoothness map/HTML."""
    render_header("Diagnostics — Smoothness")
    ensure_dirs()
    simulate_progress("Computing smoothness", steps=4)
    outdir.mkdir(parents=True, exist_ok=True)
    write_stub_html(outdir / "smoothness.html", "Smoothness Map (stub)", "<p>No real data — stub output.</p>")
    console.print(f"[green]Smoothness HTML[/green] → {outdir}/smoothness.html")
    append_debug_log(["spectramind diagnose smoothness", f"outdir={outdir}"])


@DIAG_APP.command("dashboard")
def diag_dashboard(
    no_umap: bool = typer.Option(False, "--no-umap", help="Skip UMAP embedding generation"),
    no_tsne: bool = typer.Option(False, "--no-tsne", help="Skip t-SNE embedding generation"),
    outdir: Path = typer.Option(DIAG, "--outdir", help="Output directory for dashboard"),
) -> None:
    """Build the unified diagnostics dashboard (stub)."""
    render_header("Diagnostics — Dashboard")
    ensure_dirs()
    steps = 6 - int(no_umap) - int(no_tsne)
    simulate_progress("Assembling dashboard", steps=max(3, steps))
    # small stub report
    body = "<ul>"
    body += f"<li>UMAP: {'skipped' if no_umap else 'ok (stub)'}</li>"
    body += f"<li>t-SNE: {'skipped' if no_tsne else 'ok (stub)'}</li>"
    body += "<li>GLL: ok (stub)</li><li>SHAP: ok (stub)</li><li>Microlens audit: ok (stub)</li></ul>"
    write_stub_html(outdir / f"report_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html", "Diagnostics Report (stub)", body)
    console.print(f"[green]Dashboard built[/green] → {outdir}")
    append_debug_log(
        ["spectramind diagnose dashboard", f"outdir={outdir}", f"no_umap={no_umap}", f"no_tsne={no_tsne}"]
    )


# -----------------------------------------------------------------------------
# Submit (bundle)
# -----------------------------------------------------------------------------
@APP.command("submit")
def submit(
    zip_out: Path = typer.Option(SUBMISSION_ZIP, "--zip-out", help="Path to write submission ZIP"),
) -> None:
    """Bundle artifacts for leaderboard submission (stub)."""
    render_header("Submission Bundle")
    ensure_dirs()
    # Ensure a submission.csv exists (create stub if missing)
    sub_csv = PREDICTIONS / "submission.csv"
    if not sub_csv.exists():
        sub_csv.parent.mkdir(parents=True, exist_ok=True)
        with sub_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["planet_id"] + [f"bin_{i:03d}" for i in range(283)])
            w.writerow(["P0001"] + [0.0] * 283)
    # Create zip with common artifacts
    zip_paths(zip_out, [sub_csv, OUTPUTS / "config_snapshot.yaml", CHECKPOINTS, DIAG])
    console.print(f"[green]Bundle created[/green] → {zip_out}")
    append_debug_log(["spectramind submit", f"zip_out={zip_out}"])


# -----------------------------------------------------------------------------
# Analyze log
# -----------------------------------------------------------------------------
def _parse_debug_log(md_path: Path) -> List[Dict[str, str]]:
    if not md_path.exists():
        return []
    content = md_path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in content:
        if re.match(r"^\d{4}-\d{2}-\d{2}T", line.strip().split(" — ")[0] if " — " in line else ""):
            # new header
            if current:
                rows.append(current)
            ts_cmd = line.strip().split(" — ", 1)
            ts = ts_cmd[0].strip()
            cmd = ts_cmd[1].strip() if len(ts_cmd) > 1 else ""
            current = {"time": ts, "cmd": cmd, "git_sha": git_sha_short(), "cfg": (OUTPUTS / "config_snapshot.yaml").exists() and "snapshot" or "none"}
        elif line.strip().startswith("• "):
            # maybe parse useful key-values
            kv = line.strip()[2:]
            if ":" in kv:
                k, v = kv.split(":", 1)
                current.setdefault(k.strip().lower(), v.strip())
    if current:
        rows.append(current)
    return rows


@APP.command("analyze-log")
def analyze_log(
    md_out: Path = typer.Option(OUTPUTS / "log_table.md", "--md", help="Path to write Markdown table"),
    csv_out: Path = typer.Option(OUTPUTS / "log_table.csv", "--csv", help="Path to write CSV"),
) -> None:
    """Parse v50_debug_log.md into a small CSV/Markdown summary for CI/dashboards."""
    render_header("Analyze Log")
    ensure_dirs()
    rows = _parse_debug_log(DEBUG_LOG)
    # CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ["time", "cmd", "git_sha", "cfg"]
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h, "") for h in headers])
    # MD
    lines = ["# SpectraMind V50 — CLI Calls (Last N)\n", "", "| time | cmd | git_sha | cfg |", "|---|---|---|---|"]
    for r in rows[-50:]:
        lines.append(f"| {r.get('time','')} | {r.get('cmd','').replace('|','/')} | {r.get('git_sha','')} | {r.get('cfg','')} |")
    write_text(md_out, "\n".join(lines))
    console.print(f"[green]Wrote[/green] {md_out} and {csv_out}")
    append_debug_log(["spectramind analyze-log", f"md={md_out}", f"csv={csv_out}", f"rows={len(rows)}"])


# -----------------------------------------------------------------------------
# Check CLI → file map (dev aid)
# -----------------------------------------------------------------------------
@APP.command("check-cli-map")
def check_cli_map() -> None:
    """Emit a quick mapping of CLI commands → typical files produced (dev/CI aid)."""
    render_header("CLI → File map")
    rows = [
        ("selftest", "logs/v50_debug_log.md"),
        ("calibrate", "outputs/calibrated/calibration_summary.json"),
        ("train", "outputs/checkpoints/best.ckpt"),
        ("predict", "outputs/submission.csv (or outputs/predictions/submission.csv)"),
        ("diagnose smoothness", "outputs/diagnostics/smoothness.html"),
        ("diagnose dashboard", "outputs/diagnostics/report_*.html"),
        ("submit", "outputs/submission/bundle.zip"),
        ("analyze-log", "outputs/log_table.{csv,md}"),
    ]
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Command")
    table.add_column("Artifacts")
    for cmd, art in rows:
        table.add_row(cmd, art)
    console.print(table)
    append_debug_log(["spectramind check-cli-map"])


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def _install_sigint_handler() -> None:
    def h(sig, frm):
        console.print("\n[red]Interrupted[/red]")
        raise SystemExit(130)

    try:
        signal.signal(signal.SIGINT, h)
    except Exception:
        pass


def app() -> None:  # pragma: no cover
    _install_sigint_handler()
    APP()


if __name__ == "__main__":  # pragma: no cover
    app()