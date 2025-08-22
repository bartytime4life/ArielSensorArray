#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Unified Typer CLI (ArielSensorArray)  •  ultimate upgraded

Highlights
• Full CLI parity with Makefile & CI (validate-env, dvc {pull,push,repro}, kaggle {run,submit},
  benchmark {run,report,clean}, diagrams {render}, diagnose {smoothness,dashboard}, submit, selftest)
• Deterministic seeding & run hashing (config/env/repo) → outputs/run_hash_summary_v50.json
• Rich/JSONL logging with global --log-level / --no-rich / --dry-run / --confirm
• Hydra snapshot (if available) + config hash; auditable append-only logs: logs/v50_debug_log.md, logs/v50_runs.jsonl
• Safer subprocess wrappers, friendlier errors, consistent exit codes, SIGINT handling
• DVC/Hydra/Kaggle helpers; stubbed scientific stages remain side-effect-free with --dry-run

Design notes are aligned with the V50 plan (CLI-first orchestration, Hydra snapshots, auditable logs).
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import textwrap
import time
import zipfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer
from rich import box
from rich.console import Console
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
    help="SpectraMind V50 — Unified CLI (calibrate/train/predict/diagnose/submit/selftest/analyze-log/check-cli-map/dvc/kaggle/benchmark/diagrams and more)",
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
JSONL_LOG = LOGS / "v50_runs.jsonl"
VERSION_FILE = REPO / "VERSION"
RUN_HASH_JSON = OUTPUTS / "run_hash_summary_v50.json"

CONFIG_SNAPSHOT = OUTPUTS / "config_snapshot.yaml"

# -----------------------------------------------------------------------------
# Global runtime context
# -----------------------------------------------------------------------------
@dataclass
class RuntimeCtx:
    dry_run: bool = False
    confirm: bool = False
    log_level: str = "INFO"


CTX = RuntimeCtx()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    for p in (OUTPUTS, LOGS, DIAG, CALIB, CHECKPOINTS, PREDICTIONS, SUBMISSION):
        p.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


def dict_hash(d: Dict[str, Any]) -> str:
    # stable hash of a resolved config/env dict
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True
                      )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
<div class="small">{timestamp()} • generated by spectramind</div>
<hr/>
{body_html}
</body></html>"""
    path.write_text(html, encoding="utf-8")


def simulate_progress(title: str, steps: int = 5, sleep_s: float = 0.25) -> None:
    if CTX.dry_run:
        console.print(f"[dim]DRY-RUN: {title} (skipped {steps} sim steps)[/dim]")
        return
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
    Compose Hydra config if available, else return stub dict.
    Writes a config snapshot YAML under outputs/config_snapshot.yaml when possible.
    """
    if HYDRA_AVAILABLE and config_dir.exists():
        try:
            with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
                cfg = compose(config_name="config_v50.yaml", overrides=[task_cfg] + (overrides or []))
                snap = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
                if not CTX.dry_run:
                    write_text(CONFIG_SNAPSHOT, OmegaConf.to_yaml(cfg))
                return snap
        except Exception as e:  # pragma: no cover
            console.print(f"[yellow]Hydra compose failed[/yellow]: {e}")
    return {"task": task_cfg, "overrides": overrides or [], "note": "Hydra unavailable; stub config."}


def zip_paths(zip_path: Path, paths: List[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if CTX.dry_run:
        for p in paths:
            console.print(f"[dim]DRY-RUN zip add[/dim] {p}")
        console.print(f"[dim]DRY-RUN create zip[/dim] {zip_path}")
        return
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists():
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(REPO))
                else:
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            zf.write(sub, arcname=sub.relative_to(REPO))


def run_command(cmd: List[str], env: Optional[Dict[str, str]] = None, allow_fail: bool = False) -> int:
    """
    Safer subprocess wrapper with nice printing and consistent exit codes.
    Honors global --dry-run and --confirm.
    """
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    if CTX.dry_run:
        return 0
    if CTX.confirm:
        console.print("[yellow]Confirm?[/yellow] [dim](y/N)[/dim] ", end="")
        try:
            ans = input().strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            console.print("[dim]Skipped by user.[/dim]")
            return 0
    try:
        rc = subprocess.call(cmd, env=env)
        if rc != 0 and not allow_fail:
            console.print(f"[red]Command failed with exit code {rc}[/red]")
        return rc
    except FileNotFoundError:
        console.print(f"[red]Command not found[/red]: {cmd[0]}")
        return 127
    except Exception as e:
        console.print(f"[red]Command error[/red]: {e}")
        return 1


# -----------------------------------------------------------------------------
# Determinism & run hashing
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    except Exception:
        pass


def persist_run_hash(extra: Dict[str, Any] | None = None) -> str:
    # collect minimal env/config fingerprints
    info: Dict[str, Any] = {
        "ts": timestamp(),
        "git": git_sha_short(),
        "version": read_version(),
        "python": sys.version.split()[0],
        "dry_run": CTX.dry_run,
    }
    # include hydra snapshot if present
    if CONFIG_SNAPSHOT.exists():
        info["config_snapshot_sha256"] = hashlib.sha256(CONFIG_SNAPSHOT.read_bytes()).hexdigest()[:12]
    if extra:
        info.update(extra)
    rh = dict_hash(info)
    if not CTX.dry_run:
        write_json(RUN_HASH_JSON, {"run_hash": rh, "meta": info})
    return rh


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
    if not CTX.dry_run:
        DEBUG_LOG.write_text(prefix + header + body, encoding="utf-8")


def append_run_jsonl(command: str, extra: Dict[str, Any] | None = None) -> None:
    rec = {
        "ts": timestamp(),
        "git": git_sha_short(),
        "version": read_version(),
        "command": command,
        "dry_run": CTX.dry_run,
    }
    if extra:
        rec.update(extra)
    if not CTX.dry_run:
        append_jsonl(JSONL_LOG, rec)


# -----------------------------------------------------------------------------
# Global options (log level, rich on/off, seed, dry-run, confirm)
# -----------------------------------------------------------------------------
@APP.callback(invoke_without_command=False)
def _root_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", is_flag=True, help="Show CLI version & hashes"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)"),
    no_rich: bool = typer.Option(False, "--no-rich", help="Disable rich formatting (plain console)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions but do not execute"),
    confirm: bool = typer.Option(False, "--confirm", help="Ask for confirmation before executing commands"),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed for this run"),
) -> None:
    if no_rich:
        # swap console to plain
        global console
        console = Console(no_color=True, highlight=False)
    CTX.dry_run = dry_run
    CTX.confirm = confirm
    CTX.log_level = log_level.upper()
    seed_everything(seed)
    ensure_dirs()
    # set log level into env for children
    os.environ["SPECTRAMIND_LOG_LEVEL"] = CTX.log_level

    if version:
        ver, sha, now = read_version(), git_sha_short(), timestamp()
        rh = persist_run_hash({"invocation": "--version"})
        t = Table(title="SpectraMind V50 — Version", box=box.MINIMAL_DOUBLE_HEAD)
        t.add_row("CLI", ver)
        t.add_row("Git SHA", sha)
        t.add_row("Run Hash", rh)
        t.add_row("Timestamp (UTC)", now)
        console.print(t)
        append_debug_log(["spectramind --version", f"Git SHA: {sha}", f"Version: {ver}", f"Run hash: {rh}"])
        append_run_jsonl("version", {"run_hash": rh})
        raise typer.Exit()

# -----------------------------------------------------------------------------
# Selftest
# -----------------------------------------------------------------------------
@APP.command("selftest")
def selftest(
    deep: bool = typer.Option(False, "--deep", help="Check Hydra + DVC + CUDA availability.")
) -> None:
    """Fast environment & paths sanity with optional deeper checks."""
    title = "Selftest"
    console.print(Panel.fit(f"[bold]SpectraMind V50[/bold]\n{title}", box=box.ROUNDED))
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
        gpu_ok = shutil.which("nvidia-smi") is not None and run_command(["nvidia-smi"], allow_fail=True) == 0
        check("CUDA (nvidia-smi)", gpu_ok)

    tb = Table(box=box.SIMPLE_HEAVY)
    tb.add_column("Check")
    tb.add_column("Status")
    for name, cond in checks:
        tb.add_row(name, "[green]OK[/green]" if cond else "[red]FAIL[/red]")
    console.print(tb)
    console.print("✅ Environment looks good." if ok else "❌ Selftest failed.")
    rh = persist_run_hash({"invocation": "selftest", "deep": deep})
    append_debug_log([f"spectramind selftest{' --deep' if deep else ''}", f"Result: {'OK' if ok else 'FAIL'}", f"Run: {rh}"])
    append_run_jsonl("selftest", {"deep": deep, "ok": ok, "run_hash": rh})
    if not ok:
        raise typer.Exit(code=1)

# -----------------------------------------------------------------------------
# Calibrate / Train / Predict / Temp-scale / COREL-train
# -----------------------------------------------------------------------------
@APP.command("calibrate")
def calibrate(
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra-style overrides, e.g., data=nominal +calib.version=v1")
) -> None:
    """Run the calibration kill chain (stubbed)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nCalibration", box=box.ROUNDED))
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "calibration=default", overrides)
    simulate_progress("Calibrating", steps=6)
    if not CTX.dry_run:
        write_json(CALIB / "calibration_summary.json", {"cfg": cfg, "note": "stub"})
    console.print("[green]Calibration done[/green] → outputs/calibrated")
    rh = persist_run_hash({"invocation": "calibrate"})
    append_debug_log(["spectramind calibrate", f"overrides={overrides}", f"Run: {rh}"])
    append_run_jsonl("calibrate", {"overrides": overrides or [], "run_hash": rh})


@APP.command("calibrate-temp")
def calibrate_temp(
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra overrides for temperature scaling")
) -> None:
    """Apply temperature scaling to logits/σ (stub)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nTemperature Scaling", box=box.ROUNDED))
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "calibration=temperature", overrides)
    simulate_progress("Calibrating temperature", steps=4)
    if not CTX.dry_run:
        write_json(OUTPUTS / "temperature_scaling.json", {"cfg": cfg, "temp": 1.23, "note": "stub"})
    console.print("[green]Temperature scaling complete[/green] → outputs/temperature_scaling.json")
    rh = persist_run_hash({"invocation": "calibrate-temp"})
    append_debug_log(["spectramind calibrate-temp", f"overrides={overrides}", f"Run: {rh}"])
    append_run_jsonl("calibrate-temp", {"overrides": overrides or [], "run_hash": rh})


@APP.command("corel-train")
def corel_train(
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra overrides for COREL conformal training")
) -> None:
    """Train COREL (graph conformal calibration) — stubbed."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nCOREL Conformal Training", box=box.ROUNDED))
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "uncertainty=corel", overrides)
    simulate_progress("Training COREL", steps=8)
    if not CTX.dry_run:
        write_json(OUTPUTS / "corel_model.json", {"cfg": cfg, "coverage": 0.95, "note": "stub"})
    console.print("[green]COREL saved[/green] → outputs/corel_model.json")
    rh = persist_run_hash({"invocation": "corel-train"})
    append_debug_log(["spectramind corel-train", f"overrides={overrides}", f"Run: {rh}"])
    append_run_jsonl("corel-train", {"overrides": overrides or [], "run_hash": rh})


@APP.command("train")
def train(
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra overrides, e.g. +training.epochs=1"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device string, e.g., cpu/gpu/cuda:0"),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Write artifacts to this dir (default checkpoints/)"),
) -> None:
    """Train the V50 model (stub)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nTraining", box=box.ROUNDED))
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "training=default", overrides)
    simulate_progress(f"Training on {device}", steps=10)
    target_dir = (outdir or CHECKPOINTS)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not CTX.dry_run:
        write_text(target_dir / "best.ckpt", "stub-model-weights")
        write_json(target_dir / "train_summary.json", {"cfg": cfg, "device": device, "note": "stub"})
    console.print(f"[green]Training done[/green] → {target_dir}/best.ckpt")
    rh = persist_run_hash({"invocation": "train", "device": device, "outdir": str(target_dir)})
    append_debug_log(["spectramind train", f"device={device}", f"overrides={overrides}", f"outdir={target_dir}", f"Run: {rh}"])
    append_run_jsonl("train", {"device": device, "overrides": overrides or [], "outdir": str(target_dir), "run_hash": rh})


@APP.command("predict")
def predict(
    out_csv: Path = typer.Option(OUTPUTS / "submission.csv", "--out-csv", help="Path to write submission CSV"),
    overrides: Optional[List[str]] = typer.Argument(None, help="Hydra overrides for inference"),
) -> None:
    """Run inference and write a submission CSV (stub)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nPrediction", box=box.ROUNDED))
    ensure_dirs()
    _ = hydra_compose_or_stub(REPO / "configs", "inference=default", overrides)
    simulate_progress("Predicting", steps=6)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not CTX.dry_run:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["planet_id"] + [f"bin_{i:03d}" for i in range(283)])
            w.writerow(["P0001"] + [round(0.1 + i * 1e-3, 6) for i in range(283)])
    console.print(f"[green]CSV written[/green] → {out_csv}")
    rh = persist_run_hash({"invocation": "predict", "out_csv": str(out_csv)})
    append_debug_log(["spectramind predict", f"out_csv={out_csv}", f"overrides={overrides}", f"Run: {rh}"])
    append_run_jsonl("predict", {"out_csv": str(out_csv), "overrides": overrides or [], "run_hash": rh})

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
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nDiagnostics — Smoothness", box=box.ROUNDED))
    ensure_dirs()
    simulate_progress("Computing smoothness", steps=4)
    outdir.mkdir(parents=True, exist_ok=True)
    if not CTX.dry_run:
        write_stub_html(outdir / "smoothness.html", "Smoothness Map (stub)", "<p>No real data — stub output.</p>")
    console.print(f"[green]Smoothness HTML[/green] → {outdir}/smoothness.html")
    rh = persist_run_hash({"invocation": "diagnose.smoothness"})
    append_debug_log(["spectramind diagnose smoothness", f"outdir={outdir}", f"Run: {rh}"])
    append_run_jsonl("diagnose.smoothness", {"outdir": str(outdir), "run_hash": rh})


@DIAG_APP.command("dashboard")
def diag_dashboard(
    no_umap: bool = typer.Option(False, "--no-umap", help="Skip UMAP embedding generation"),
    no_tsne: bool = typer.Option(False, "--no-tsne", help="Skip t-SNE embedding generation"),
    outdir: Path = typer.Option(DIAG, "--outdir", help="Output directory for dashboard"),
) -> None:
    """Build the unified diagnostics dashboard (stub)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nDiagnostics — Dashboard", box=box.ROUNDED))
    ensure_dirs()
    steps = 6 - int(no_umap) - int(no_tsne)
    simulate_progress("Assembling dashboard", steps=max(3, steps))
    body = "<ul>"
    body += f"<li>UMAP: {'skipped' if no_umap else 'ok (stub)'}</li>"
    body += f"<li>t-SNE: {'skipped' if no_tsne else 'ok (stub)'}</li>"
    body += "<li>GLL: ok (stub)</li><li>SHAP: ok (stub)</li><li>Microlens audit: ok (stub)</li></ul>"
    out_file = outdir / f"report_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    if not CTX.dry_run:
        write_stub_html(out_file, "Diagnostics Report (stub)", body)
    console.print(f"[green]Dashboard built[/green] → {outdir}")
    rh = persist_run_hash({"invocation": "diagnose.dashboard", "no_umap": no_umap, "no_tsne": no_tsne})
    append_debug_log(
        ["spectramind diagnose dashboard", f"outdir={outdir}", f"no_umap={no_umap}", f"no_tsne={no_tsne}", f"Run: {rh}"]
    )
    append_run_jsonl(
        "diagnose.dashboard",
        {"outdir": str(outdir), "no_umap": no_umap, "no_tsne": no_tsne, "run_hash": rh},
    )

# -----------------------------------------------------------------------------
# Submit (bundle)
# -----------------------------------------------------------------------------
@APP.command("submit")
def submit(
    zip_out: Path = typer.Option(SUBMISSION_ZIP, "--zip-out", help="Path to write submission ZIP"),
) -> None:
    """Bundle artifacts for leaderboard submission (stub)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nSubmission Bundle", box=box.ROUNDED))
    ensure_dirs()
    # Ensure a submission.csv exists (create stub if missing)
    sub_csv = PREDICTIONS / "submission.csv"
    if not sub_csv.exists() and not CTX.dry_run:
        sub_csv.parent.mkdir(parents=True, exist_ok=True)
        with sub_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["planet_id"] + [f"bin_{i:03d}" for i in range(283)])
            w.writerow(["P0001"] + [0.0] * 283)
    zip_paths(zip_out, [sub_csv, CONFIG_SNAPSHOT, CHECKPOINTS, DIAG])
    console.print(f"[green]Bundle created[/green] → {zip_out}")
    rh = persist_run_hash({"invocation": "submit", "zip_out": str(zip_out)})
    append_debug_log(["spectramind submit", f"zip_out={zip_out}", f"Run: {rh}"])
    append_run_jsonl("submit", {"zip_out": str(zip_out), "run_hash": rh})

# -----------------------------------------------------------------------------
# Analyze log + short
# -----------------------------------------------------------------------------
def _parse_debug_log(md_path: Path) -> List[Dict[str, str]]:
    if not md_path.exists():
        return []
    content = md_path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for line in content:
        if " — " in line and re.match(r"^\d{4}-\d{2}-\d{2}T", line.strip().split(" — ")[0]):
            if current:
                rows.append(current)
            ts_cmd = line.strip().split(" — ", 1)
            ts = ts_cmd[0].strip()
            cmd = ts_cmd[1].strip() if len(ts_cmd) > 1 else ""
            current = {"time": ts, "cmd": cmd, "git_sha": git_sha_short(), "cfg": ("snapshot" if CONFIG_SNAPSHOT.exists() else "none")}
        elif line.strip().startswith("• "):
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
    """Parse v50_debug_log.md into CSV/Markdown summary for CI/dashboards."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nAnalyze Log", box=box.ROUNDED))
    ensure_dirs()
    rows = _parse_debug_log(DEBUG_LOG)
    # CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if not CTX.dry_run:
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
    if not CTX.dry_run:
        write_text(md_out, "\n".join(lines))
    console.print(f"[green]Wrote[/green] {md_out} and {csv_out}")
    rh = persist_run_hash({"invocation": "analyze-log"})
    append_debug_log(["spectramind analyze-log", f"md={md_out}", f"csv={csv_out}", f"rows={len(rows)}", f"Run: {rh}"])
    append_run_jsonl("analyze-log", {"md": str(md_out), "csv": str(csv_out), "rows": len(rows), "run_hash": rh})


@APP.command("analyze-log-short")
def analyze_log_short(
    overrides: Optional[List[str]] = typer.Argument(None, help="Ignored; parity with Make target"),
) -> None:
    """Short CI-friendly summary: last 5 entries from CSV (auto-runs analyze-log if needed)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nAnalyze Log (short)", box=box.ROUNDED))
    ensure_dirs()
    csv_path = OUTPUTS / "log_table.csv"
    if not csv_path.exists():
        console.print(">>> Generating log CSV via analyze-log")
        analyze_log()
    if csv_path.exists():
        console.print("=== Last 5 CLI invocations ===")
        body = (csv_path.read_text(encoding="utf-8").splitlines()[1:])[-5:]
        for row in body:
            cols = row.split(",")
            if len(cols) >= 4:
                console.print(f"time={cols[0]} | cmd={cols[1]} | git_sha={cols[2]} | cfg={cols[3]}")
    else:
        console.print("::warning::No log_table.csv to summarize")
    rh = persist_run_hash({"invocation": "analyze-log-short"})
    append_debug_log(["spectramind analyze-log-short", f"Run: {rh}"])
    append_run_jsonl("analyze-log-short", {"run_hash": rh})

# -----------------------------------------------------------------------------
# Validate-env (parity with Make)
# -----------------------------------------------------------------------------
@APP.command("validate-env")
def validate_env() -> None:
    """Validate .env schema if scripts/validate_env.py exists (safe no-op otherwise)."""
    if (REPO / "scripts" / "validate_env.py").exists():
        console.print(">>> Validating .env schema")
        rc = run_command([sys.executable, str(REPO / "scripts" / "validate_env.py")])
        if rc != 0:
            raise typer.Exit(code=rc)
    else:
        console.print(">>> Skipping validate-env (scripts/validate_env.py not found)")
    rh = persist_run_hash({"invocation": "validate-env"})
    append_debug_log(["spectramind validate-env", f"Run: {rh}"])
    append_run_jsonl("validate-env", {"run_hash": rh})

# -----------------------------------------------------------------------------
# DVC helpers (parity with Make)
# -----------------------------------------------------------------------------
DVC_APP = typer.Typer(help="DVC convenience commands")
APP.add_typer(DVC_APP, name="dvc")


@DVC_APP.command("pull")
def dvc_pull() -> None:
    """dvc pull || true"""
    rc = run_command(["dvc", "pull"], allow_fail=True)
    rh = persist_run_hash({"invocation": "dvc.pull", "rc": rc})
    append_debug_log(["spectramind dvc pull", f"rc={rc}", f"Run: {rh}"])
    append_run_jsonl("dvc.pull", {"rc": rc, "run_hash": rh})


@DVC_APP.command("push")
def dvc_push() -> None:
    """dvc push || true"""
    rc = run_command(["dvc", "push"], allow_fail=True)
    rh = persist_run_hash({"invocation": "dvc.push", "rc": rc})
    append_debug_log(["spectramind dvc push", f"rc={rc}", f"Run: {rh}"])
    append_run_jsonl("dvc.push", {"rc": rc, "run_hash": rh})


@DVC_APP.command("repro")
def dvc_repro(
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Stage or file to reproduce"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reproduce"),
) -> None:
    """dvc repro [--target STAGE]"""
    cmd = ["dvc", "repro"]
    if force:
        cmd.append("--force")
    if target:
        cmd += ["--single-item", target]
    rc = run_command(cmd)
    if rc != 0:
        raise typer.Exit(code=rc)
    rh = persist_run_hash({"invocation": "dvc.repro", "target": target or "", "force": force})
    append_debug_log(["spectramind dvc repro", f"target={target}", f"force={force}", f"Run: {rh}"])
    append_run_jsonl("dvc.repro", {"target": target or "", "force": force, "run_hash": rh})

# -----------------------------------------------------------------------------
# Kaggle helpers (parity with Make)
# -----------------------------------------------------------------------------
KAGGLE_APP = typer.Typer(help="Kaggle helpers (run, submit)")
APP.add_typer(KAGGLE_APP, name="kaggle")


@KAGGLE_APP.command("run")
def kaggle_run(
    out_dir: Path = typer.Option(OUTPUTS, "--outdir", help="Artifacts directory"),
) -> None:
    """Single-epoch GPU-ish run (Kaggle-like)."""
    console.print(">>> Running single-epoch GPU run (Kaggle-like)")
    selftest(deep=False)
    train(overrides=["+training.epochs=1"], device="gpu", outdir=out_dir)
    predict(out_csv=PREDICTIONS / "submission.csv", overrides=None)
    rh = persist_run_hash({"invocation": "kaggle.run"})
    append_debug_log(["spectramind kaggle-run", f"outdir={out_dir}", f"Run: {rh}"])
    append_run_jsonl("kaggle.run", {"outdir": str(out_dir), "run_hash": rh})


@KAGGLE_APP.command("submit")
def kaggle_submit(
    comp: str = typer.Option("neurips-2025-ariel", "--competition", "-c", help="Kaggle competition slug"),
    file: Path = typer.Option(PREDICTIONS / "submission.csv", "--file", "-f", help="submission.csv path"),
    message: str = typer.Option("Spectramind V50 auto-submit", "--message", "-m", help="Submission message"),
) -> None:
    """Submit to Kaggle via kaggle CLI (requires kaggle to be installed & authed)."""
    console.print(">>> Submitting to Kaggle competition")
    if shutil.which("kaggle") is None:
        console.print("[red]kaggle CLI not found. Install and authenticate first.[/red]")
        raise typer.Exit(code=127)
    rc = run_command(["kaggle", "competitions", "submit", "-c", comp, "-f", str(file), "-m", message])
    if rc != 0:
        raise typer.Exit(code=rc)
    rh = persist_run_hash({"invocation": "kaggle.submit", "competition": comp})
    append_debug_log(["spectramind kaggle-submit", f"comp={comp}", f"file={file}", f"Run: {rh}"])
    append_run_jsonl("kaggle.submit", {"competition": comp, "file": str(file), "run_hash": rh})

# -----------------------------------------------------------------------------
# Benchmark helpers (parity with Make)
# -----------------------------------------------------------------------------
BENCH_APP = typer.Typer(help="Benchmark helpers")
APP.add_typer(BENCH_APP, name="benchmark")


@BENCH_APP.command("run")
def benchmark_run(
    device: str = typer.Option("cpu", "--device"),
    epochs: int = typer.Option(1, "--epochs"),
    outroot: Path = typer.Option(Path("benchmarks"), "--outroot"),
) -> None:
    """Run a benchmark flow (train+diagnose) and emit a summary."""
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = outroot / f"{ts}_{device}"
    outdir.mkdir(parents=True, exist_ok=True)
    train(overrides=[f"+training.epochs={epochs}"], device=device, outdir=outdir)
    diag_smoothness(outdir=outdir)
    try:
        diag_dashboard(no_umap=True, no_tsne=True, outdir=outdir)
    except Exception:
        diag_dashboard(no_umap=False, no_tsne=False, outdir=outdir)
    # write summary
    summary = [
        "Benchmark summary",
        dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        f"python   : {subprocess.getoutput(sys.executable + ' --version')}",
        f"cli      : spectramind",
        f"device   : {device}",
        f"epochs   : {epochs}",
    ]
    if shutil.which("nvidia-smi"):
        summary.append(subprocess.getoutput("nvidia-smi"))
    summary.append("\nArtifacts in " + str(outdir) + ":\n" + subprocess.getoutput(f"ls -lh {outdir}"))
    if not CTX.dry_run:
        write_text(outdir / "summary.txt", "\n".join(summary))
    console.print(f">>> Benchmark complete → {outdir}/summary.txt")
    rh = persist_run_hash({"invocation": "benchmark.run", "device": device, "epochs": epochs, "outdir": str(outdir)})
    append_debug_log(["spectramind benchmark-run", f"outdir={outdir}", f"Run: {rh}"])
    append_run_jsonl("benchmark.run", {"device": device, "epochs": epochs, "outdir": str(outdir), "run_hash": rh})


@BENCH_APP.command("report")
def benchmark_report() -> None:
    """Aggregate benchmark summaries into aggregated/report.md"""
    aggregated = Path("aggregated")
    aggregated.mkdir(exist_ok=True)
    lines = ["# SpectraMind V50 Benchmark Report", ""]
    for f in sorted(Path("benchmarks").rglob("summary.txt")):
        rel = str(f)
        lines += [f"## {rel}", "", Path(rel).read_text(encoding="utf-8"), ""]
    if not CTX.dry_run:
        write_text(aggregated / "report.md", "\n".join(lines))
    console.print(">>> Aggregated → aggregated/report.md")
    rh = persist_run_hash({"invocation": "benchmark.report"})
    append_debug_log(["spectramind benchmark-report", f"Run: {rh}"])
    append_run_jsonl("benchmark.report", {"run_hash": rh})


@BENCH_APP.command("clean")
def benchmark_clean() -> None:
    """Remove benchmarks/ and aggregated/"""
    for p in [Path("benchmarks"), Path("aggregated")]:
        if p.exists():
            run_command(["rm", "-rf", str(p)], allow_fail=True)
    console.print(">>> Benchmarks cleaned")
    rh = persist_run_hash({"invocation": "benchmark.clean"})
    append_debug_log(["spectramind benchmark-clean", f"Run: {rh}"])
    append_run_jsonl("benchmark.clean", {"run_hash": rh})

# -----------------------------------------------------------------------------
# Mermaid/diagrams helpers (parity with Make)
# -----------------------------------------------------------------------------
DIAGX_APP = typer.Typer(help="Mermaid / diagrams export")
APP.add_typer(DIAGX_APP, name="diagrams")


@DIAGX_APP.command("render")
def diagrams_render(
    files: List[str] = typer.Argument(["ARCHITECTURE.md", "README.md"], help="Files to scan & export Mermaid"),
    theme: Optional[str] = typer.Option(None, "--theme"),
    export_png: bool = typer.Option(False, "--png", help="Export PNG alongside SVG"),
) -> None:
    """Call scripts/export_mermaid.py if present to render diagrams (SVG/PNG)."""
    script = REPO / "scripts" / "export_mermaid.py"
    if not script.exists():
        console.print("[yellow]scripts/export_mermaid.py not found (skipping).[/yellow]")
        return
    env = os.environ.copy()
    if theme:
        env["THEME"] = theme
    env["EXPORT_PNG"] = "1" if export_png else "0"
    console.print(">>> Rendering Mermaid diagrams")
    rc = run_command([sys.executable, str(script), *files], env=env)
    if rc != 0:
        raise typer.Exit(code=rc)
    console.print(">>> Output → docs/diagrams")
    rh = persist_run_hash({"invocation": "diagrams.render", "files": files})
    append_debug_log(["spectramind diagrams.render", f"files={files}", f"Run: {rh}"])
    append_run_jsonl("diagrams.render", {"files": files, "run_hash": rh})

# -----------------------------------------------------------------------------
# Hash helpers (config/env/repo) — quick audit
# -----------------------------------------------------------------------------
@APP.command("hashes")
def hashes() -> None:
    """Print quick hashes of repo HEAD, config snapshot, environment & run hash."""
    ensure_dirs()
    rows = [
        ("Git SHA", git_sha_short()),
        ("Config snapshot", hashlib.sha256(CONFIG_SNAPSHOT.read_bytes()).hexdigest()[:12] if CONFIG_SNAPSHOT.exists() else "none"),
        ("Python", sys.version.split()[0]),
        ("Dry-run", str(CTX.dry_run)),
    ]
    t = Table(title="SpectraMind V50 — Hashes", box=box.MINIMAL_DOUBLE_HEAD)
    t.add_column("Item")
    t.add_column("Value")
    for k, v in rows:
        t.add_row(k, str(v))
    console.print(t)
    rh = persist_run_hash({"invocation": "hashes"})
    append_debug_log(["spectramind hashes", f"Run: {rh}"])
    append_run_jsonl("hashes", {"run_hash": rh})

# -----------------------------------------------------------------------------
# Check CLI → file map (dev aid)
# -----------------------------------------------------------------------------
@APP.command("check-cli-map")
def check_cli_map() -> None:
    """Emit a quick mapping of CLI commands → typical files produced (dev/CI aid)."""
    console.print(Panel.fit("[bold]SpectraMind V50[/bold]\nCLI → File map", box=box.SIMPLE))
    rows = [
        ("selftest", "logs/v50_debug_log.md"),
        ("validate-env", "scripts/validate_env.py → OK"),
        ("dvc pull/push", ".dvc cache/state"),
        ("dvc repro", "dvc.yaml stages → outputs/*"),
        ("calibrate", "outputs/calibrated/calibration_summary.json"),
        ("train", "outputs/checkpoints/best.ckpt"),
        ("predict", "outputs/submission.csv (or outputs/predictions/submission.csv)"),
        ("diagnose smoothness", "outputs/diagnostics/smoothness.html"),
        ("diagnose dashboard", "outputs/diagnostics/report_*.html"),
        ("submit", "outputs/submission/bundle.zip"),
        ("analyze-log / analyze-log-short", "outputs/log_table.{csv,md}"),
        ("benchmark run/report/clean", "benchmarks/* / aggregated/report.md"),
        ("diagrams render", "docs/diagrams/*"),
        ("kaggle run/submit", "predictions/submission.csv / Kaggle submission"),
        ("hashes", "Quick run/repo/config hashes"),
    ]
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Command")
    table.add_column("Artifacts")
    for cmd, art in rows:
        table.add_row(cmd, art)
    console.print(table)
    rh = persist_run_hash({"invocation": "check-cli-map"})
    append_debug_log(["spectramind check-cli-map", f"Run: {rh}"])
    append_run_jsonl("check-cli-map", {"run_hash": rh})

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