\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
SpectraMind V50 — Unified Typer CLI (ArielSensorArray)  •  ultimate upgraded

Highlights
• Full CLI parity with Makefile & CI (validate-env, dvc {pull,push,repro}, kaggle {run,submit},
benchmark {run,report,clean}, diagrams {render}, diagnose {smoothness,dashboard}, submit, selftest)
• Deterministic seeding & run hashing (config/env/repo) → outputs/run\_hash\_summary\_v50.json
• Rich/JSONL logging with global --log-level / --no-rich / --dry-run / --confirm
• Hydra snapshot (if available) + config hash; auditable append-only logs: logs/v50\_debug\_log.md, logs/v50\_runs.jsonl
• Safer subprocess wrappers, friendlier errors, consistent exit codes, SIGINT handling
• DVC/Hydra/Kaggle helpers; stubbed scientific stages remain side-effect-free with --dry-run

Design notes are aligned with the V50 plan (CLI-first orchestration, Hydra snapshots, auditable logs).
"""

from **future** import annotations

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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import typer
from rich import box
from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# -----------------------------------------------------------------------------

# Optional Hydra

# -----------------------------------------------------------------------------

try:
from hydra import compose, initialize\_config\_dir
from omegaconf import OmegaConf

```
HYDRA_AVAILABLE = True
```

except Exception:  # pragma: no cover
HYDRA\_AVAILABLE = False

# -----------------------------------------------------------------------------

# App & paths

# -----------------------------------------------------------------------------

APP = typer.Typer(
name="spectramind",
help="SpectraMind V50 — Unified CLI (calibrate/train/predict/diagnose/submit/selftest/analyze-log/check-cli-map/dvc/kaggle/benchmark/diagrams and more)",
add\_completion=True,
no\_args\_is\_help=True,
)
console = Console()

REPO = Path(**file**).resolve().parent
ROOT = REPO  # alias
OUTPUTS = REPO / "outputs"
LOGS = REPO / "logs"
DIAG = OUTPUTS / "diagnostics"
CALIB = OUTPUTS / "calibrated"
CHECKPOINTS = OUTPUTS / "checkpoints"
PREDICTIONS = OUTPUTS / "predictions"
SUBMISSION = OUTPUTS / "submission"
SUBMISSION\_ZIP = SUBMISSION / "bundle.zip"

DEBUG\_LOG = LOGS / "v50\_debug\_log.md"
JSONL\_LOG = LOGS / "v50\_runs.jsonl"
VERSION\_FILE = REPO / "VERSION"
RUN\_HASH\_JSON = OUTPUTS / "run\_hash\_summary\_v50.json"

CONFIG\_SNAPSHOT = OUTPUTS / "config\_snapshot.yaml"

# -----------------------------------------------------------------------------

# Global runtime context

# -----------------------------------------------------------------------------

@dataclass
class RuntimeCtx:
dry\_run: bool = False
confirm: bool = False
log\_level: str = "INFO"

CTX = RuntimeCtx()

# -----------------------------------------------------------------------------

# Utilities

# -----------------------------------------------------------------------------

def ensure\_dirs() -> None:
for p in (OUTPUTS, LOGS, DIAG, CALIB, CHECKPOINTS, PREDICTIONS, SUBMISSION):
p.mkdir(parents=True, exist\_ok=True)

def timestamp() -> str:
return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def git\_sha\_short() -> str:
try:
return (
subprocess.check\_output(\["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO))
.decode()
.strip()
)
except Exception:
return "unknown"

def read\_version() -> str:
return VERSION\_FILE.read\_text(encoding="utf-8").strip() if VERSION\_FILE.exists() else "0.1.0"

def dict\_hash(d: Dict\[str, Any]) -> str:
\# stable hash of a resolved config/env dict
payload = json.dumps(d, sort\_keys=True, separators=(",", ":")).encode("utf-8")
return hashlib.sha256(payload).hexdigest()\[:12]

def write\_text(path: Path, text: str) -> None:
path.parent.mkdir(parents=True, exist\_ok=True)
path.write\_text(text, encoding="utf-8")

def write\_json(path: Path, data: Any) -> None:
path.parent.mkdir(parents=True, exist\_ok=True)
path.write\_text(json.dumps(data, indent=2), encoding="utf-8")

def append\_jsonl(path: Path, record: Dict\[str, Any]) -> None:
path.parent.mkdir(parents=True, exist\_ok=True)
with path.open("a", encoding="utf-8") as f:
f.write(json.dumps(record, ensure\_ascii=False) + "\n")

def write\_stub\_html(path: Path, title: str, body\_html: str) -> None:
path.parent.mkdir(parents=True, exist\_ok=True)
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

def simulate\_progress(title: str, steps: int = 5, sleep\_s: float = 0.25) -> None:
if CTX.dry\_run:
console.print(f"\[dim]DRY-RUN: {title} (skipped {steps} sim steps)\[/dim]")
return
with Progress(
SpinnerColumn(),
TextColumn("{task.description}"),
BarColumn(),
TimeElapsedColumn(),
transient=True,
console=console,
) as progress:
tid = progress.add\_task(title, total=steps)
for \_ in range(steps):
time.sleep(sleep\_s)
progress.advance(tid)

def hydra\_compose\_or\_stub(config\_dir: Path, task\_cfg: str, overrides: Optional\[List\[str]]) -> Dict\[str, Any]:
"""
Compose Hydra config if available, else return stub dict.
Writes a config snapshot YAML under outputs/config\_snapshot.yaml when possible.
"""
if HYDRA\_AVAILABLE and config\_dir.exists():
try:
with initialize\_config\_dir(version\_base=None, config\_dir=str(config\_dir.resolve())):
cfg = compose(config\_name="config\_v50.yaml", overrides=\[task\_cfg] + (overrides or \[]))
snap = dict(OmegaConf.to\_container(cfg, resolve=True))  # type: ignore
if not CTX.dry\_run:
write\_text(CONFIG\_SNAPSHOT, OmegaConf.to\_yaml(cfg))
return snap
except Exception as e:  # pragma: no cover
console.print(f"\[yellow]Hydra compose failed\[/yellow]: {e}")
return {"task": task\_cfg, "overrides": overrides or \[], "note": "Hydra unavailable; stub config."}

def zip\_paths(zip\_path: Path, paths: List\[Path]) -> None:
zip\_path.parent.mkdir(parents=True, exist\_ok=True)
if CTX.dry\_run:
for p in paths:
console.print(f"\[dim]DRY-RUN zip add\[/dim] {p}")
console.print(f"\[dim]DRY-RUN create zip\[/dim] {zip\_path}")
return
with zipfile.ZipFile(zip\_path, "w", compression=zipfile.ZIP\_DEFLATED) as zf:
for p in paths:
if p.exists():
if p.is\_file():
zf.write(p, arcname=p.relative\_to(REPO))
else:
for sub in p.rglob("\*"):
if sub.is\_file():
zf.write(sub, arcname=sub.relative\_to(REPO))

def run\_command(cmd: List\[str], env: Optional\[Dict\[str, str]] = None, allow\_fail: bool = False) -> int:
"""
Safer subprocess wrapper with nice printing and consistent exit codes.
Honors global --dry-run and --confirm.
"""
console.print(f"\[dim]\$ {' '.join(cmd)}\[/dim]")
if CTX.dry\_run:
return 0
if CTX.confirm:
console.print("\[yellow]Confirm?\[/yellow] [dim](y/N)\[/dim] ", end="")
try:
ans = input().strip().lower()
except EOFError:
ans = "n"
if ans not in ("y", "yes"):
console.print("\[dim]Skipped by user.\[/dim]")
return 0
try:
rc = subprocess.call(cmd, env=env)
if rc != 0 and not allow\_fail:
console.print(f"\[red]Command failed with exit code {rc}\[/red]")
return rc
except FileNotFoundError:
console.print(f"\[red]Command not found\[/red]: {cmd\[0]}")
return 127
except Exception as e:
console.print(f"\[red]Command error\[/red]: {e}")
return 1

# -----------------------------------------------------------------------------

# Determinism & run hashing

# -----------------------------------------------------------------------------

def seed\_everything(seed: int = 42) -> None:
random.seed(seed)
try:
import numpy as np  # type: ignore

```
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
```

def persist\_run\_hash(extra: Dict\[str, Any] | None = None) -> str:
\# collect minimal env/config fingerprints
info: Dict\[str, Any] = {
"ts": timestamp(),
"git": git\_sha\_short(),
"version": read\_version(),
"python": sys.version.split()\[0],
"dry\_run": CTX.dry\_run,
}
\# include hydra snapshot if present
if CONFIG\_SNAPSHOT.exists():
info\["config\_snapshot\_sha256"] = hashlib.sha256(CONFIG\_SNAPSHOT.read\_bytes()).hexdigest()\[:12]
if extra:
info.update(extra)
rh = dict\_hash(info)
if not CTX.dry\_run:
write\_json(RUN\_HASH\_JSON, {"run\_hash": rh, "meta": info})
return rh

def append\_debug\_log(lines: Iterable\[str]) -> None:
ensure\_dirs()
lines = list(lines)
header = f"\n⸻\n\n{timestamp()} — {lines\[0] if lines else ''}"
body = "".join(f"\n\t• {ln}" for ln in lines\[1:])
prefix = (
DEBUG\_LOG.read\_text(encoding="utf-8")
if DEBUG\_LOG.exists()
else "SpectraMind V50 — Debug & Audit Log\n\nAppend-only operator log (immutable).\n"
)
if not CTX.dry\_run:
DEBUG\_LOG.write\_text(prefix + header + body, encoding="utf-8")

def append\_run\_jsonl(command: str, extra: Dict\[str, Any] | None = None) -> None:
rec = {
"ts": timestamp(),
"git": git\_sha\_short(),
"version": read\_version(),
"command": command,
"dry\_run": CTX.dry\_run,
}
if extra:
rec.update(extra)
if not CTX.dry\_run:
append\_jsonl(JSONL\_LOG, rec)

# -----------------------------------------------------------------------------

# Global options (log level, rich on/off, seed, dry-run, confirm)

# -----------------------------------------------------------------------------

@APP.callback(invoke\_without\_command=False)
def \_root\_callback(
ctx: typer.Context,
version: Optional\[bool] = typer.Option(None, "--version", is\_flag=True, help="Show CLI version & hashes"),
log\_level: str = typer.Option("INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)"),
no\_rich: bool = typer.Option(False, "--no-rich", help="Disable rich formatting (plain console)"),
dry\_run: bool = typer.Option(False, "--dry-run", help="Print actions but do not execute"),
confirm: bool = typer.Option(False, "--confirm", help="Ask for confirmation before executing commands"),
seed: int = typer.Option(42, "--seed", help="Deterministic seed for this run"),
) -> None:
if no\_rich:
\# swap console to plain
global console
console = Console(no\_color=True, highlight=False)
CTX.dry\_run = dry\_run
CTX.confirm = confirm
CTX.log\_level = log\_level.upper()
seed\_everything(seed)
ensure\_dirs()
\# set log level into env for children
os.environ\["SPECTRAMIND\_LOG\_LEVEL"] = CTX.log\_level

```
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
```

# -----------------------------------------------------------------------------

# Selftest

# -----------------------------------------------------------------------------

@APP.command("selftest")
def selftest(
deep: bool = typer.Option(False, "--deep", help="Check Hydra + DVC + CUDA availability.")
) -> None:
"""Fast environment & paths sanity with optional deeper checks."""
title = "Selftest"
console.print(Panel.fit(f"\[bold]SpectraMind V50\[/bold]\n{title}", box=box.ROUNDED))
ensure\_dirs()

```
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
```

# -----------------------------------------------------------------------------

# Calibrate / Train / Predict / Temp-scale / COREL-train

# -----------------------------------------------------------------------------

@APP.command("calibrate")
def calibrate(
overrides: Optional\[List\[str]] = typer.Argument(None, help="Hydra-style overrides, e.g., data=nominal +calib.version=v1")
) -> None:
"""Run the calibration kill chain (stubbed)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nCalibration", box=box.ROUNDED))
ensure\_dirs()
cfg = hydra\_compose\_or\_stub(REPO / "configs", "calibration=default", overrides)
simulate\_progress("Calibrating", steps=6)
if not CTX.dry\_run:
write\_json(CALIB / "calibration\_summary.json", {"cfg": cfg, "note": "stub"})
console.print("\[green]Calibration done\[/green] → outputs/calibrated")
rh = persist\_run\_hash({"invocation": "calibrate"})
append\_debug\_log(\["spectramind calibrate", f"overrides={overrides}", f"Run: {rh}"])
append\_run\_jsonl("calibrate", {"overrides": overrides or \[], "run\_hash": rh})

@APP.command("calibrate-temp")
def calibrate\_temp(
overrides: Optional\[List\[str]] = typer.Argument(None, help="Hydra overrides for temperature scaling")
) -> None:
"""Apply temperature scaling to logits/σ (stub)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nTemperature Scaling", box=box.ROUNDED))
ensure\_dirs()
cfg = hydra\_compose\_or\_stub(REPO / "configs", "calibration=temperature", overrides)
simulate\_progress("Calibrating temperature", steps=4)
if not CTX.dry\_run:
write\_json(OUTPUTS / "temperature\_scaling.json", {"cfg": cfg, "temp": 1.23, "note": "stub"})
console.print("\[green]Temperature scaling complete\[/green] → outputs/temperature\_scaling.json")
rh = persist\_run\_hash({"invocation": "calibrate-temp"})
append\_debug\_log(\["spectramind calibrate-temp", f"overrides={overrides}", f"Run: {rh}"])
append\_run\_jsonl("calibrate-temp", {"overrides": overrides or \[], "run\_hash": rh})

@APP.command("corel-train")
def corel\_train(
overrides: Optional\[List\[str]] = typer.Argument(None, help="Hydra overrides for COREL conformal training")
) -> None:
"""Train COREL (graph conformal calibration) — stubbed."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nCOREL Conformal Training", box=box.ROUNDED))
ensure\_dirs()
cfg = hydra\_compose\_or\_stub(REPO / "configs", "uncertainty=corel", overrides)
simulate\_progress("Training COREL", steps=8)
if not CTX.dry\_run:
write\_json(OUTPUTS / "corel\_model.json", {"cfg": cfg, "coverage": 0.95, "note": "stub"})
console.print("\[green]COREL saved\[/green] → outputs/corel\_model.json")
rh = persist\_run\_hash({"invocation": "corel-train"})
append\_debug\_log(\["spectramind corel-train", f"overrides={overrides}", f"Run: {rh}"])
append\_run\_jsonl("corel-train", {"overrides": overrides or \[], "run\_hash": rh})

@APP.command("train")
def train(
overrides: Optional\[List\[str]] = typer.Argument(None, help="Hydra overrides, e.g. +training.epochs=1"),
device: str = typer.Option("cpu", "--device", "-d", help="Device string, e.g., cpu/gpu/cuda:0"),
outdir: Optional\[Path] = typer.Option(None, "--outdir", help="Write artifacts to this dir (default checkpoints/)"),
) -> None:
"""Train the V50 model (stub)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nTraining", box=box.ROUNDED))
ensure\_dirs()
cfg = hydra\_compose\_or\_stub(REPO / "configs", "training=default", overrides)
simulate\_progress(f"Training on {device}", steps=10)
target\_dir = (outdir or CHECKPOINTS)
target\_dir.mkdir(parents=True, exist\_ok=True)
if not CTX.dry\_run:
write\_text(target\_dir / "best.ckpt", "stub-model-weights")
write\_json(target\_dir / "train\_summary.json", {"cfg": cfg, "device": device, "note": "stub"})
console.print(f"\[green]Training done\[/green] → {target\_dir}/best.ckpt")
rh = persist\_run\_hash({"invocation": "train", "device": device, "outdir": str(target\_dir)})
append\_debug\_log(\["spectramind train", f"device={device}", f"overrides={overrides}", f"outdir={target\_dir}", f"Run: {rh}"])
append\_run\_jsonl("train", {"device": device, "overrides": overrides or \[], "outdir": str(target\_dir), "run\_hash": rh})

@APP.command("predict")
def predict(
out\_csv: Path = typer.Option(OUTPUTS / "submission.csv", "--out-csv", help="Path to write submission CSV"),
overrides: Optional\[List\[str]] = typer.Argument(None, help="Hydra overrides for inference"),
) -> None:
"""Run inference and write a submission CSV (stub)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nPrediction", box=box.ROUNDED))
ensure\_dirs()
\_ = hydra\_compose\_or\_stub(REPO / "configs", "inference=default", overrides)
simulate\_progress("Predicting", steps=6)
out\_csv.parent.mkdir(parents=True, exist\_ok=True)
if not CTX.dry\_run:
with out\_csv.open("w", newline="", encoding="utf-8") as f:
w = csv.writer(f)
w\.writerow(\["planet\_id"] + \[f"bin\_{i:03d}" for i in range(283)])
w\.writerow(\["P0001"] + \[round(0.1 + i \* 1e-3, 6) for i in range(283)])
console.print(f"\[green]CSV written\[/green] → {out\_csv}")
rh = persist\_run\_hash({"invocation": "predict", "out\_csv": str(out\_csv)})
append\_debug\_log(\["spectramind predict", f"out\_csv={out\_csv}", f"overrides={overrides}", f"Run: {rh}"])
append\_run\_jsonl("predict", {"out\_csv": str(out\_csv), "overrides": overrides or \[], "run\_hash": rh})

# -----------------------------------------------------------------------------

# Diagnose (group)

# -----------------------------------------------------------------------------

DIAG\_APP = typer.Typer(help="Diagnostics subcommands (smoothness, dashboard)")
APP.add\_typer(DIAG\_APP, name="diagnose")

@DIAG\_APP.command("smoothness")
def diag\_smoothness(
outdir: Path = typer.Option(DIAG, "--outdir", help="Output directory for smoothness artifacts")
) -> None:
"""Generate a stub smoothness map/HTML."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nDiagnostics — Smoothness", box=box.ROUNDED))
ensure\_dirs()
simulate\_progress("Computing smoothness", steps=4)
outdir.mkdir(parents=True, exist\_ok=True)
if not CTX.dry\_run:
write\_stub\_html(outdir / "smoothness.html", "Smoothness Map (stub)", "<p>No real data — stub output.</p>")
console.print(f"\[green]Smoothness HTML\[/green] → {outdir}/smoothness.html")
rh = persist\_run\_hash({"invocation": "diagnose.smoothness"})
append\_debug\_log(\["spectramind diagnose smoothness", f"outdir={outdir}", f"Run: {rh}"])
append\_run\_jsonl("diagnose.smoothness", {"outdir": str(outdir), "run\_hash": rh})

@DIAG\_APP.command("dashboard")
def diag\_dashboard(
no\_umap: bool = typer.Option(False, "--no-umap", help="Skip UMAP embedding generation"),
no\_tsne: bool = typer.Option(False, "--no-tsne", help="Skip t-SNE embedding generation"),
outdir: Path = typer.Option(DIAG, "--outdir", help="Output directory for dashboard"),
) -> None:
"""Build the unified diagnostics dashboard (stub)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nDiagnostics — Dashboard", box=box.ROUNDED))
ensure\_dirs()
steps = 6 - int(no\_umap) - int(no\_tsne)
simulate\_progress("Assembling dashboard", steps=max(3, steps))
body = "<ul>"
body += f"<li>UMAP: {'skipped' if no\_umap else 'ok (stub)'}</li>"
body += f"<li>t-SNE: {'skipped' if no\_tsne else 'ok (stub)'}</li>"
body += "<li>GLL: ok (stub)</li><li>SHAP: ok (stub)</li><li>Microlens audit: ok (stub)</li></ul>"
out\_file = outdir / f"report\_{dt.datetime.utcnow().strftime('%Y%m%d\_%H%M%S')}.html"
if not CTX.dry\_run:
write\_stub\_html(out\_file, "Diagnostics Report (stub)", body)
console.print(f"\[green]Dashboard built\[/green] → {outdir}")
rh = persist\_run\_hash({"invocation": "diagnose.dashboard", "no\_umap": no\_umap, "no\_tsne": no\_tsne})
append\_debug\_log(
\["spectramind diagnose dashboard", f"outdir={outdir}", f"no\_umap={no\_umap}", f"no\_tsne={no\_tsne}", f"Run: {rh}"]
)
append\_run\_jsonl(
"diagnose.dashboard",
{"outdir": str(outdir), "no\_umap": no\_umap, "no\_tsne": no\_tsne, "run\_hash": rh},
)

# -----------------------------------------------------------------------------

# Submit (bundle)

# -----------------------------------------------------------------------------

@APP.command("submit")
def submit(
zip\_out: Path = typer.Option(SUBMISSION\_ZIP, "--zip-out", help="Path to write submission ZIP"),
) -> None:
"""Bundle artifacts for leaderboard submission (stub)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nSubmission Bundle", box=box.ROUNDED))
ensure\_dirs()
\# Ensure a submission.csv exists (create stub if missing)
sub\_csv = PREDICTIONS / "submission.csv"
if not sub\_csv.exists() and not CTX.dry\_run:
sub\_csv.parent.mkdir(parents=True, exist\_ok=True)
with sub\_csv.open("w", newline="", encoding="utf-8") as f:
w = csv.writer(f)
w\.writerow(\["planet\_id"] + \[f"bin\_{i:03d}" for i in range(283)])
w\.writerow(\["P0001"] + \[0.0] \* 283)
zip\_paths(zip\_out, \[sub\_csv, CONFIG\_SNAPSHOT, CHECKPOINTS, DIAG])
console.print(f"\[green]Bundle created\[/green] → {zip\_out}")
rh = persist\_run\_hash({"invocation": "submit", "zip\_out": str(zip\_out)})
append\_debug\_log(\["spectramind submit", f"zip\_out={zip\_out}", f"Run: {rh}"])
append\_run\_jsonl("submit", {"zip\_out": str(zip\_out), "run\_hash": rh})

# -----------------------------------------------------------------------------

# Analyze log + short

# -----------------------------------------------------------------------------

def \_parse\_debug\_log(md\_path: Path) -> List\[Dict\[str, str]]:
if not md\_path.exists():
return \[]
content = md\_path.read\_text(encoding="utf-8").splitlines()
rows: List\[Dict\[str, str]] = \[]
current: Dict\[str, str] = {}
for line in content:
if re.match(r"^\d{4}-\d{2}-\d{2}T", line.strip().split(" — ")\[0] if " — " in line else ""):
if current:
rows.append(current)
ts\_cmd = line.strip().split(" — ", 1)
ts = ts\_cmd\[0].strip()
cmd = ts\_cmd\[1].strip() if len(ts\_cmd) > 1 else ""
current = {"time": ts, "cmd": cmd, "git\_sha": git\_sha\_short(), "cfg": (CONFIG\_SNAPSHOT.exists() and "snapshot" or "none")}
elif line.strip().startswith("• "):
kv = line.strip()\[2:]
if ":" in kv:
k, v = kv.split(":", 1)
current.setdefault(k.strip().lower(), v.strip())
if current:
rows.append(current)
return rows

@APP.command("analyze-log")
def analyze\_log(
md\_out: Path = typer.Option(OUTPUTS / "log\_table.md", "--md", help="Path to write Markdown table"),
csv\_out: Path = typer.Option(OUTPUTS / "log\_table.csv", "--csv", help="Path to write CSV"),
) -> None:
"""Parse v50\_debug\_log.md into CSV/Markdown summary for CI/dashboards."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nAnalyze Log", box=box.ROUNDED))
ensure\_dirs()
rows = \_parse\_debug\_log(DEBUG\_LOG)
\# CSV
csv\_out.parent.mkdir(parents=True, exist\_ok=True)
if not CTX.dry\_run:
with csv\_out.open("w", newline="", encoding="utf-8") as f:
w = csv.writer(f)
headers = \["time", "cmd", "git\_sha", "cfg"]
w\.writerow(headers)
for r in rows:
w\.writerow(\[r.get(h, "") for h in headers])
\# MD
lines = \["# SpectraMind V50 — CLI Calls (Last N)\n", "", "| time | cmd | git\_sha | cfg |", "|---|---|---|---|"]
for r in rows\[-50:]:
lines.append(f"| {r.get('time','')} | {r.get('cmd','').replace('|','/')} | {r.get('git\_sha','')} | {r.get('cfg','')} |")
if not CTX.dry\_run:
write\_text(md\_out, "\n".join(lines))
console.print(f"\[green]Wrote\[/green] {md\_out} and {csv\_out}")
rh = persist\_run\_hash({"invocation": "analyze-log"})
append\_debug\_log(\["spectramind analyze-log", f"md={md\_out}", f"csv={csv\_out}", f"rows={len(rows)}", f"Run: {rh}"])
append\_run\_jsonl("analyze-log", {"md": str(md\_out), "csv": str(csv\_out), "rows": len(rows), "run\_hash": rh})

@APP.command("analyze-log-short")
def analyze\_log\_short(
overrides: Optional\[List\[str]] = typer.Argument(None, help="Ignored; parity with Make target"),
) -> None:
"""Short CI-friendly summary: last 5 entries from CSV (auto-runs analyze-log if needed)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nAnalyze Log (short)", box=box.ROUNDED))
ensure\_dirs()
csv\_path = OUTPUTS / "log\_table.csv"
if not csv\_path.exists():
console.print(">>> Generating log CSV via analyze-log")
analyze\_log()
if csv\_path.exists():
console.print("=== Last 5 CLI invocations ===")
body = (csv\_path.read\_text(encoding="utf-8").splitlines()\[1:])\[-5:]
for row in body:
cols = row\.split(",")
if len(cols) >= 4:
console.print(f"time={cols\[0]} | cmd={cols\[1]} | git\_sha={cols\[2]} | cfg={cols\[3]}")
else:
console.print("::warning::No log\_table.csv to summarize")
rh = persist\_run\_hash({"invocation": "analyze-log-short"})
append\_debug\_log(\["spectramind analyze-log-short", f"Run: {rh}"])
append\_run\_jsonl("analyze-log-short", {"run\_hash": rh})

# -----------------------------------------------------------------------------

# Validate-env (parity with Make)

# -----------------------------------------------------------------------------

@APP.command("validate-env")
def validate\_env() -> None:
"""Validate .env schema if scripts/validate\_env.py exists (safe no-op otherwise)."""
if (REPO / "scripts" / "validate\_env.py").exists():
console.print(">>> Validating .env schema")
rc = run\_command(\[sys.executable, str(REPO / "scripts" / "validate\_env.py")])
if rc != 0:
raise typer.Exit(code=rc)
else:
console.print(">>> Skipping validate-env (scripts/validate\_env.py not found)")
rh = persist\_run\_hash({"invocation": "validate-env"})
append\_debug\_log(\["spectramind validate-env", f"Run: {rh}"])
append\_run\_jsonl("validate-env", {"run\_hash": rh})

# -----------------------------------------------------------------------------

# DVC helpers (parity with Make)

# -----------------------------------------------------------------------------

DVC\_APP = typer.Typer(help="DVC convenience commands")
APP.add\_typer(DVC\_APP, name="dvc")

@DVC\_APP.command("pull")
def dvc\_pull() -> None:
"""dvc pull || true"""
rc = run\_command(\["dvc", "pull"], allow\_fail=True)
rh = persist\_run\_hash({"invocation": "dvc.pull", "rc": rc})
append\_debug\_log(\["spectramind dvc pull", f"rc={rc}", f"Run: {rh}"])
append\_run\_jsonl("dvc.pull", {"rc": rc, "run\_hash": rh})

@DVC\_APP.command("push")
def dvc\_push() -> None:
"""dvc push || true"""
rc = run\_command(\["dvc", "push"], allow\_fail=True)
rh = persist\_run\_hash({"invocation": "dvc.push", "rc": rc})
append\_debug\_log(\["spectramind dvc push", f"rc={rc}", f"Run: {rh}"])
append\_run\_jsonl("dvc.push", {"rc": rc, "run\_hash": rh})

@DVC\_APP.command("repro")
def dvc\_repro(
target: Optional\[str] = typer.Option(None, "--target", "-t", help="Stage or file to reproduce"),
force: bool = typer.Option(False, "--force", "-f", help="Force reproduce"),
) -> None:
"""dvc repro \[--target STAGE]"""
cmd = \["dvc", "repro"]
if force:
cmd.append("--force")
if target:
cmd += \["--single-item", target]
rc = run\_command(cmd)
if rc != 0:
raise typer.Exit(code=rc)
rh = persist\_run\_hash({"invocation": "dvc.repro", "target": target or "", "force": force})
append\_debug\_log(\["spectramind dvc repro", f"target={target}", f"force={force}", f"Run: {rh}"])
append\_run\_jsonl("dvc.repro", {"target": target or "", "force": force, "run\_hash": rh})

# -----------------------------------------------------------------------------

# Kaggle helpers (parity with Make)

# -----------------------------------------------------------------------------

KAGGLE\_APP = typer.Typer(help="Kaggle helpers (run, submit)")
APP.add\_typer(KAGGLE\_APP, name="kaggle")

@KAGGLE\_APP.command("run")
def kaggle\_run(
out\_dir: Path = typer.Option(OUTPUTS, "--outdir", help="Artifacts directory"),
) -> None:
"""Single-epoch GPU-ish run (Kaggle-like)."""
console.print(">>> Running single-epoch GPU run (Kaggle-like)")
selftest(deep=False)
train(overrides=\["+training.epochs=1"], device="gpu", outdir=out\_dir)
predict(out\_csv=PREDICTIONS / "submission.csv", overrides=None)
rh = persist\_run\_hash({"invocation": "kaggle.run"})
append\_debug\_log(\["spectramind kaggle-run", f"outdir={out\_dir}", f"Run: {rh}"])
append\_run\_jsonl("kaggle.run", {"outdir": str(out\_dir), "run\_hash": rh})

@KAGGLE\_APP.command("submit")
def kaggle\_submit(
comp: str = typer.Option("neurips-2025-ariel", "--competition", "-c", help="Kaggle competition slug"),
file: Path = typer.Option(PREDICTIONS / "submission.csv", "--file", "-f", help="submission.csv path"),
message: str = typer.Option("Spectramind V50 auto-submit", "--message", "-m", help="Submission message"),
) -> None:
"""Submit to Kaggle via kaggle CLI (requires kaggle to be installed & authed)."""
console.print(">>> Submitting to Kaggle competition")
if shutil.which("kaggle") is None:
console.print("\[red]kaggle CLI not found. Install and authenticate first.\[/red]")
raise typer.Exit(code=127)
rc = run\_command(\["kaggle", "competitions", "submit", "-c", comp, "-f", str(file), "-m", message])
if rc != 0:
raise typer.Exit(code=rc)
rh = persist\_run\_hash({"invocation": "kaggle.submit", "competition": comp})
append\_debug\_log(\["spectramind kaggle-submit", f"comp={comp}", f"file={file}", f"Run: {rh}"])
append\_run\_jsonl("kaggle.submit", {"competition": comp, "file": str(file), "run\_hash": rh})

# -----------------------------------------------------------------------------

# Benchmark helpers (parity with Make)

# -----------------------------------------------------------------------------

BENCH\_APP = typer.Typer(help="Benchmark helpers")
APP.add\_typer(BENCH\_APP, name="benchmark")

@BENCH\_APP.command("run")
def benchmark\_run(
device: str = typer.Option("cpu", "--device"),
epochs: int = typer.Option(1, "--epochs"),
outroot: Path = typer.Option(Path("benchmarks"), "--outroot"),
) -> None:
"""Run a benchmark flow (train+diagnose) and emit a summary."""
ts = dt.datetime.utcnow().strftime("%Y%m%d\_%H%M%S")
outdir = outroot / f"{ts}\_{device}"
outdir.mkdir(parents=True, exist\_ok=True)
train(overrides=\[f"+training.epochs={epochs}"], device=device, outdir=outdir)
diag\_smoothness(outdir=outdir)
try:
diag\_dashboard(no\_umap=True, no\_tsne=True, outdir=outdir)
except Exception:
diag\_dashboard(no\_umap=False, no\_tsne=False, outdir=outdir)
\# write summary
summary = \[
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
if not CTX.dry\_run:
write\_text(outdir / "summary.txt", "\n".join(summary))
console.print(f">>> Benchmark complete → {outdir}/summary.txt")
rh = persist\_run\_hash({"invocation": "benchmark.run", "device": device, "epochs": epochs, "outdir": str(outdir)})
append\_debug\_log(\["spectramind benchmark-run", f"outdir={outdir}", f"Run: {rh}"])
append\_run\_jsonl("benchmark.run", {"device": device, "epochs": epochs, "outdir": str(outdir), "run\_hash": rh})

@BENCH\_APP.command("report")
def benchmark\_report() -> None:
"""Aggregate benchmark summaries into aggregated/report.md"""
aggregated = Path("aggregated")
aggregated.mkdir(exist\_ok=True)
lines = \["# SpectraMind V50 Benchmark Report", ""]
for f in sorted(Path("benchmarks").rglob("summary.txt")):
rel = str(f)
lines += \[f"## {rel}", "", Path(rel).read\_text(encoding="utf-8"), ""]
if not CTX.dry\_run:
write\_text(aggregated / "report.md", "\n".join(lines))
console.print(">>> Aggregated → aggregated/report.md")
rh = persist\_run\_hash({"invocation": "benchmark.report"})
append\_debug\_log(\["spectramind benchmark-report", f"Run: {rh}"])
append\_run\_jsonl("benchmark.report", {"run\_hash": rh})

@BENCH\_APP.command("clean")
def benchmark\_clean() -> None:
"""Remove benchmarks/ and aggregated/"""
for p in \[Path("benchmarks"), Path("aggregated")]:
if p.exists():
run\_command(\["rm", "-rf", str(p)], allow\_fail=True)
console.print(">>> Benchmarks cleaned")
rh = persist\_run\_hash({"invocation": "benchmark.clean"})
append\_debug\_log(\["spectramind benchmark-clean", f"Run: {rh}"])
append\_run\_jsonl("benchmark.clean", {"run\_hash": rh})

# -----------------------------------------------------------------------------

# Mermaid/diagrams helpers (parity with Make)

# -----------------------------------------------------------------------------

DIAGX\_APP = typer.Typer(help="Mermaid / diagrams export")
APP.add\_typer(DIAGX\_APP, name="diagrams")

@DIAGX\_APP.command("render")
def diagrams\_render(
files: List\[str] = typer.Argument(\["ARCHITECTURE.md", "README.md"], help="Files to scan & export Mermaid"),
theme: Optional\[str] = typer.Option(None, "--theme"),
export\_png: bool = typer.Option(False, "--png", help="Export PNG alongside SVG"),
) -> None:
"""Call scripts/export\_mermaid.py if present to render diagrams (SVG/PNG)."""
script = REPO / "scripts" / "export\_mermaid.py"
if not script.exists():
console.print("\[yellow]scripts/export\_mermaid.py not found (skipping).\[/yellow]")
return
env = os.environ.copy()
if theme:
env\["THEME"] = theme
env\["EXPORT\_PNG"] = "1" if export\_png else "0"
console.print(">>> Rendering Mermaid diagrams")
rc = run\_command(\[sys.executable, str(script), \*files], env=env)
if rc != 0:
raise typer.Exit(code=rc)
console.print(">>> Output → docs/diagrams")
rh = persist\_run\_hash({"invocation": "diagrams.render", "files": files})
append\_debug\_log(\["spectramind diagrams.render", f"files={files}", f"Run: {rh}"])
append\_run\_jsonl("diagrams.render", {"files": files, "run\_hash": rh})

# -----------------------------------------------------------------------------

# Hash helpers (config/env/repo) — quick audit

# -----------------------------------------------------------------------------

@APP.command("hashes")
def hashes() -> None:
"""Print quick hashes of repo HEAD, config snapshot, environment & run hash."""
ensure\_dirs()
rows = \[
("Git SHA", git\_sha\_short()),
("Config snapshot", CONFIG\_SNAPSHOT.exists() and hashlib.sha256(CONFIG\_SNAPSHOT.read\_bytes()).hexdigest()\[:12] or "none"),
("Python", sys.version.split()\[0]),
("Dry-run", str(CTX.dry\_run)),
]
t = Table(title="SpectraMind V50 — Hashes", box=box.MINIMAL\_DOUBLE\_HEAD)
t.add\_column("Item")
t.add\_column("Value")
for k, v in rows:
t.add\_row(k, str(v))
console.print(t)
rh = persist\_run\_hash({"invocation": "hashes"})
append\_debug\_log(\["spectramind hashes", f"Run: {rh}"])
append\_run\_jsonl("hashes", {"run\_hash": rh})

# -----------------------------------------------------------------------------

# Check CLI → file map (dev aid)

# -----------------------------------------------------------------------------

@APP.command("check-cli-map")
def check\_cli\_map() -> None:
"""Emit a quick mapping of CLI commands → typical files produced (dev/CI aid)."""
console.print(Panel.fit("\[bold]SpectraMind V50\[/bold]\nCLI → File map", box=box.SIMPLE))
rows = \[
("selftest", "logs/v50\_debug\_log.md"),
("validate-env", "scripts/validate\_env.py → OK"),
("dvc pull/push", ".dvc cache/state"),
("dvc repro", "dvc.yaml stages → outputs/*"),
("calibrate", "outputs/calibrated/calibration\_summary.json"),
("train", "outputs/checkpoints/best.ckpt"),
("predict", "outputs/submission.csv (or outputs/predictions/submission.csv)"),
("diagnose smoothness", "outputs/diagnostics/smoothness.html"),
("diagnose dashboard", "outputs/diagnostics/report\_*.html"),
("submit", "outputs/submission/bundle.zip"),
("analyze-log / analyze-log-short", "outputs/log\_table.{csv,md}"),
("benchmark run/report/clean", "benchmarks/\* / aggregated/report.md"),
("diagrams render", "docs/diagrams/\*"),
("kaggle run/submit", "predictions/submission.csv / Kaggle submission"),
("hashes", "Quick run/repo/config hashes"),
]
table = Table(box=box.SIMPLE\_HEAVY)
table.add\_column("Command")
table.add\_column("Artifacts")
for cmd, art in rows:
table.add\_row(cmd, art)
console.print(table)
rh = persist\_run\_hash({"invocation": "check-cli-map"})
append\_debug\_log(\["spectramind check-cli-map", f"Run: {rh}"])
append\_run\_jsonl("check-cli-map", {"run\_hash": rh})

# -----------------------------------------------------------------------------

# Entrypoint

# -----------------------------------------------------------------------------

def \_install\_sigint\_handler() -> None:
def h(sig, frm):
console.print("\n\[red]Interrupted\[/red]")
raise SystemExit(130)

```
try:
    signal.signal(signal.SIGINT, h)
except Exception:
    pass
```

def app() -> None:  # pragma: no cover
\_install\_sigint\_handler()
APP()

if **name** == "**main**":  # pragma: no cover
app()
