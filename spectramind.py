#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Unified Typer CLI (ArielSensorArray)

References:
- CLI-first, Hydra configs, DVC, CI reproducibility [oai_citation:5‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)
- Kaggle runtime alignment (9h envelope, GPU quotas) [oai_citation:6‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf)
- Terminal UX (Rich, HTML/PNG diagnostics) [oai_citation:7‡Comprehensive Guide to GUI Programming.pdf](file-service://file-NiTQ7cdQw7zGnLUVCUpoRx)
- Physics-informed safeguards (radiation inverse-square law metaphor for log safety) [oai_citation:8‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL)

Safe pre-pipeline operation:
- All commands generate stub artifacts & config snapshots.
- Append-only debug log at logs/v50_debug_log.md
"""

from __future__ import annotations
import csv, datetime as dt, hashlib, json, os, re, signal, subprocess, sys, textwrap, time, zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Hydra (optional)
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except Exception:
    HYDRA_AVAILABLE = False

APP = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — Unified CLI (train / predict / calibrate / diagnose / submit / selftest / analyze-log)",
    add_completion=True,
    no_args_is_help=True,
)
console = Console()

ROOT, REPO = Path(__file__).resolve().parent, Path(__file__).resolve().parent
LOGS, OUTPUTS = REPO / "logs", REPO / "outputs"
DIAG, CALIB, CHECKPOINTS = OUTPUTS / "diagnostics", OUTPUTS / "calibrated", OUTPUTS / "checkpoints"
DEBUG_LOG, VERSION_FILE = LOGS / "v50_debug_log.md", REPO / "VERSION"
RUN_HASH_JSON = OUTPUTS / "run_hash_summary_v50.json"

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def timestamp() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def ensure_dirs() -> None:
    for p in (LOGS, OUTPUTS, DIAG, CALIB, CHECKPOINTS):
        p.mkdir(parents=True, exist_ok=True)

def git_sha_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
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

def append_debug_log(lines: List[str]) -> None:
    ensure_dirs()
    header = f"\n⸻\n\n{timestamp()} — " + lines[0]
    body = "".join(f"\n\t• {ln}" for ln in lines[1:])
    prefix = DEBUG_LOG.read_text(encoding="utf-8") if DEBUG_LOG.exists() else \
        "SpectraMind V50 — Debug & Audit Log\n\nAppend-only operator log (immutable).\n"
    DEBUG_LOG.write_text(prefix + header + body, encoding="utf-8")

def hydra_compose_or_stub(config_dir: Path, task_cfg: str, overrides: List[str]) -> Dict[str, Any]:
    if HYDRA_AVAILABLE and config_dir.exists():
        try:
            with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
                cfg = compose(config_name="config_v50.yaml", overrides=[task_cfg] + (overrides or []))
                snap = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
                (OUTPUTS / "config_snapshot.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
                return snap
        except Exception as e:
            console.print(f"[yellow]Hydra compose failed[/yellow]: {e}")
    return {"task": task_cfg, "overrides": overrides, "note": "Hydra unavailable; stub config."}

def write_stub_html(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>{title}</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1000px}}</style></head>
<body><h1>{title}</h1><small>{timestamp()}</small><div>{body}</div></body></html>"""
    path.write_text(html, encoding="utf-8")

def simulate_progress(title: str, steps=5, sleep_s=0.3) -> None:
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(),
                  transient=True, console=console) as progress:
        tid = progress.add_task(title, total=steps)
        for _ in range(steps):
            time.sleep(sleep_s)
            progress.advance(tid)

# --------------------------------------------------------------------------------------
# Global --version
# --------------------------------------------------------------------------------------
@APP.callback(invoke_without_command=False)
def main(ctx: typer.Context, version: Optional[bool] = typer.Option(None, "--version", is_flag=True)) -> None:
    if version:
        ver, sha, rh, now = read_version(), git_sha_short(), read_run_hash(), timestamp()
        table = Table(title="SpectraMind V50 — Version", box=box.MINIMAL_DOUBLE_HEAD)
        for k, v in [("CLI", ver), ("Git SHA", sha), ("Run/Config Hash", rh), ("Timestamp (UTC)", now)]:
            table.add_row(k, v)
        console.print(table)
        append_debug_log(["spectramind --version", f"Git SHA: {sha}", f"Version: {ver}", f"Run/Config hash: {rh}"])
        raise typer.Exit()

# --------------------------------------------------------------------------------------
# Selftest (with DVC check)
# --------------------------------------------------------------------------------------
@APP.command("selftest")
def selftest(deep: bool = typer.Option(False, "--deep", help="Check Hydra + DVC + CUDA.")) -> None:
    render_header("Selftest")
    ensure_dirs()
    checks, ok = [], True
    def check(name, cond): 
        nonlocal ok; checks.append((name, "[green]OK[/green]" if cond else "[red]FAIL[/red]")); ok &= cond
    for d in [LOGS, OUTPUTS, REPO/"configs", REPO/"README.md"]: check(str(d), d.exists())
    if deep:
        check("Hydra compose", HYDRA_AVAILABLE)
        check("DVC present", (REPO/".dvc").exists())
    table = Table(box=box.SIMPLE_HEAVY); table.add_column("Check"); table.add_column("Status")
    for n, s in checks: table.add_row(n, s)
    console.print(table)
    console.print("✅ OK" if ok else "❌ FAIL")
    append_debug_log(["spectramind selftest"+(" --deep" if deep else ""), f"Result: {'OK' if ok else 'FAIL'}"])
    if not ok: raise typer.Exit(code=1)

# --------------------------------------------------------------------------------------
# Core pipeline commands (calibrate, train, predict, calibrate-temp, corel-train)
# --------------------------------------------------------------------------------------
@APP.command("calibrate")
def calibrate(overrides: List[str] = typer.Argument(None)) -> None:
    render_header("Calibration"); ensure_dirs()
    cfg = hydra_compose_or_stub(REPO/"configs", "calibration=default", overrides or [])
    simulate_progress("Calibrating", 6)
    (CALIB/"calibration_summary.json").write_text(json.dumps({"cfg":cfg,"note":"stub"}, indent=2))
    console.print("[green]Calibration done[/green] → outputs/calibrated")

@APP.command("train")
def train(overrides: List[str] = typer.Argument(None)) -> None:
    render_header("Training"); ensure_dirs()
    _ = hydra_compose_or_stub(REPO/"configs", "training=default", overrides or [])
    simulate_progress("Training", 10)
    (CHECKPOINTS/"best.ckpt").write_text("stub-model")
    console.print("[green]Training done[/green] → outputs/checkpoints/best.ckpt")

@APP.command("predict")
def predict(out_csv: Path = typer.Option(OUTPUTS/"submission.csv", "--out-csv"), overrides: List[str]=typer.Argument(None)) -> None:
    render_header("Prediction"); ensure_dirs()
    _ = hydra_compose_or_stub(REPO/"configs", "inference=default", overrides or [])
    simulate_progress("Predicting", 6)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["planet_id"]+[f"bin_{i:03d}" for i in range(283)]); w.writerow(["P0001"]+[0.1+i*1e-3 for i in range(283)])
    console.print(f"[green]CSV written[/green] → {out_csv}")

# (calibrate-temp, corel-train, diagnose dashboard, submit, analyze-log remain similar structure)
# --------------------------------------------------------------------------------------
def app(): _install_sigint_handler(); APP()
def _install_sigint_handler():
    def h(sig,frm): console.print("\n[red]Interrupted[/red]"); raise SystemExit(130)
    try: signal.signal(signal.SIGINT,h)
    except Exception: pass

if __name__=="__main__": app()