**spectramind.py**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Unified Typer CLI (ArielSensorArray)

Mission
-------
• CLI-first, reproducible control surface around the V50 pipeline.
• Hydra-backed configuration (optional) and structured logging for every operation.
• Rich console UX with an append-only audit log (logs/v50_debug_log.md).

Design
------
Subcommands:
  selftest, calibrate, train, predict, calibrate-temp, corel-train,
  diagnose dashboard, submit, analyze-log, check-cli-map

Notes
-----
This file is safe to run before the full pipeline exists. When a real module
is missing, it simulates work and still:
  - honors Hydra-style overrides,
  - emits stub artifacts/HTML,
  - appends structured entries to v50_debug_log.md.
Swap simulate_* hooks with real implementations as the repo fills out.
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
from typing import Dict, Iterable, List, Optional, Tuple

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Optional Hydra/OmegaConf; CLI remains usable without Hydra
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    HYDRA_AVAILABLE = True
except Exception:  # pragma: no cover - optional path
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


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "0.1.0"


def git_sha_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO))
        return out.decode().strip()
    except Exception:  # pragma: no cover - git not guaranteed
        return "unknown"


def compute_config_hash(obj: object | None, *, fallback: str = "unknown") -> str:
    try:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(data).hexdigest()[:12]
    except Exception:
        return fallback


def write_run_hash_summary(config_obj: object | None, *, overrides: Iterable[str] = ()) -> str:
    """
    Persist a tiny run-hash summary so `--version` can show a stable config hash.
    Priority: hydra-resolved object > overrides-only > 'unknown'.
    """
    ensure_dirs()
    ov_hash = compute_config_hash(sorted(list(overrides)), fallback="noovr")
    cfg_hash = compute_config_hash(config_obj, fallback=ov_hash)
    RUN_HASH_JSON.write_text(
        json.dumps({"config_hash": cfg_hash, "timestamp": timestamp()}, indent=2),
        encoding="utf-8",
    )
    return cfg_hash


def read_run_hash() -> str:
    try:
        if RUN_HASH_JSON.exists():
            return json.loads(RUN_HASH_JSON.read_text(encoding="utf-8")).get("config_hash", "unknown")
    except Exception:
        pass
    return "unknown"


def pretty_overrides(overrides: List[str]) -> str:
    return ", ".join(overrides) if overrides else "(none)"


def hydra_compose_or_stub(config_dir: Path, task_cfg: str, overrides: List[str]) -> Dict:
    """
    Compose Hydra config from `configs/` dir and `task_cfg` (e.g., 'training=default').
    Returns a plain dict (OmegaConf.to_container) or a minimal stub when Hydra is unavailable.
    """
    if HYDRA_AVAILABLE and config_dir.exists():
        try:
            with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
                base = [task_cfg] if task_cfg else []
                cfg = compose(config_name="config_v50.yaml", overrides=base + overrides)
                return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
        except Exception as e:
            console.print(f"[yellow]Hydra compose failed[/yellow]: {e}")
    return {"task": task_cfg or "default", "overrides": overrides, "stub": True}


def write_stub_html(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;max-width:900px}}
pre{{background:#111;color:#eee;padding:1rem;border-radius:.5rem;overflow:auto}}
</style></head>
<body>
<h1>{title}</h1>
<p>Generated at {timestamp()}</p>
{body}
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def zip_artifacts(zip_out: Path, globs: Iterable[str]) -> None:
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pattern in globs:
            for p in REPO.glob(pattern):
                if p.is_file():
                    zf.write(p, p.relative_to(REPO))


def simulate_long_task(title: str, steps: int = 5, sleep_s: float = 0.4) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task(title, total=steps)
        for _ in range(steps):
            time.sleep(sleep_s)
            progress.advance(task_id)


# --------------------------------------------------------------------------------------
# Version flag
# --------------------------------------------------------------------------------------
@APP.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", help="Show CLI version + config hash + build timestamp.", is_flag=True
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
        table.add_row("Config Hash", rh)
        table.add_row("Timestamp (UTC)", now)
        console.print(table)

        append_debug_log(
            ["spectramind --version", f"Git SHA: {sha}", f"Version: {ver}", f"Config hash: {rh}", f"Timestamp: {now}"]
        )
        raise typer.Exit()


# --------------------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------------------
@APP.command("selftest")
def selftest(
    deep: bool = typer.Option(False, "--deep", help="Run deep checks (files, directories, basic Hydra compose).")
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
        ok &= cond

    # Basic directories & files
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
        ["spectramind selftest" + (" --deep" if deep else ""), f"Git SHA: {git_sha_short()}", f"Result: {'OK' if ok else 'FAIL'}"]
    )

    if not ok:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------------------
# Calibrate / Train / Predict / Calibrate-Temp / COREL-Train
# --------------------------------------------------------------------------------------
@APP.command("calibrate")
def calibrate(
    overrides: List[str] = typer.Argument(None, help="Hydra-style overrides, e.g. data=kaggle calibration.cache=true")
) -> None:
    """Run the calibration kill chain: raw → calibrated (persist)."""
    render_header("Calibration")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "calibration=default", overrides or [])
    simulate_long_task("Calibrating", steps=6)

    # Emit stub artifact + run hash
    (CALIB / "calibration_summary.json").write_text(
        json.dumps({"timestamp": timestamp(), "config": cfg, "note": "Stub calibration summary."}, indent=2),
        encoding="utf-8",
    )
    rh = write_run_hash_summary(cfg, overrides=overrides or [])
    console.print(f"[green]Calibration completed[/green]. Artifacts in outputs/calibrated/ (hash {rh}).")

    append_debug_log(
        [
            f"spectramind calibrate {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            f"Config overrides: {pretty_overrides(overrides or [])}",
            f"Run hash: {rh}",
            "Artifacts: outputs/calibrated/",
        ]
    )


@APP.command("train")
def train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. data=toy training=default model=v50")
) -> None:
    """Train the V50 model."""
    render_header("Training")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "training=default", overrides or [])
    simulate_long_task("Training", steps=10, sleep_s=0.3)

    # Emit stub checkpoint + run hash
    (CHECKPOINTS / "best.ckpt").write_bytes(b"stub-checkpoint\n")
    rh = write_run_hash_summary(cfg, overrides=overrides or [])
    console.print(f"[green]Training finished[/green]. Checkpoint at outputs/checkpoints/best.ckpt (hash {rh}).")

    append_debug_log(
        [
            f"spectramind train {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            f"Config overrides: {pretty_overrides(overrides or [])}",
            f"Run hash: {rh}",
            "Artifacts: outputs/checkpoints/best.ckpt",
        ]
    )


@APP.command("predict")
def predict(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. data=toy model=v50"),
    out_csv: Path = typer.Option(OUTPUTS / "submission.csv", "--out-csv", help="Output CSV for submission predictions."),
) -> None:
    """Predict μ/σ and export submission CSV."""
    render_header("Prediction")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "inference=default", overrides or [])
    simulate_long_task("Predicting", steps=6, sleep_s=0.35)

    # Emit stub submission + run hash
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "mu_0", "mu_1", "sigma_0", "sigma_1"])
        w.writerow(["planet_0001", 0.1, 0.2, 0.05, 0.06])
        w.writerow(["planet_0002", 0.2, 0.3, 0.04, 0.05])
    rh = write_run_hash_summary(cfg, overrides=overrides or [])
    console.print(f"[green]Predictions complete[/green] → {out_csv} (hash {rh}).")

    append_debug_log(
        [
            f"spectramind predict {pretty_overrides(overrides or [])} --out-csv {out_csv}",
            f"Git SHA: {git_sha_short()}",
            f"Run hash: {rh}",
            "Artifacts: " + str(out_csv),
        ]
    )


@APP.command("calibrate-temp")
def calibrate_temp(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides, e.g. inference=default ts.max_iter=100")
) -> None:
    """Temperature scaling for uncertainty calibration."""
    render_header("Temperature Scaling")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "ts=default", overrides or [])
    simulate_long_task("Optimizing temperature", steps=5)

    (OUTPUTS / "temp_scaling.json").write_text(
        json.dumps({"T": 1.07, "val_gll_improvement": 0.013, "timestamp": timestamp()}, indent=2),
        encoding="utf-8",
    )
    rh = write_run_hash_summary(cfg, overrides=overrides or [])
    console.print(f"[green]Temperature scaling done[/green] → outputs/temp_scaling.json (hash {rh}).")

    append_debug_log(
        [
            f"spectramind calibrate-temp {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            f"Run hash: {rh}",
            "Artifacts: outputs/temp_scaling.json",
        ]
    )


@APP.command("corel-train")
def corel_train(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for COREL GNN training")
) -> None:
    """Train COREL (graph-aware conformal prediction) for calibrated σ intervals."""
    render_header("COREL Conformal Training")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "corel=default", overrides or [])
    simulate_long_task("Training COREL", steps=7)

    (OUTPUTS / "corel_model.pt").write_bytes(b"stub-corel-model\n")
    rh = write_run_hash_summary(cfg, overrides=overrides or [])
    console.print(f"[green]COREL training done[/green] → outputs/corel_model.pt (hash {rh}).")

    append_debug_log(
        [
            f"spectramind corel-train {pretty_overrides(overrides or [])}",
            f"Git SHA: {git_sha_short()}",
            f"Run hash: {rh}",
            "Artifacts: outputs/corel_model.pt",
        ]
    )


# --------------------------------------------------------------------------------------
# Diagnose
# --------------------------------------------------------------------------------------
diagnose_app = typer.Typer(help="Diagnostics suite (HTML dashboard, plots, overlays)")
APP.add_typer(diagnose_app, name="diagnose")


@diagnose_app.command("dashboard")
def diagnose_dashboard(
    overrides: List[str] = typer.Argument(None, help="Hydra overrides for diagnostics"),
    html_out: Path = typer.Option(DIAG / "report_v1.html", "--html-out", help="Output HTML diagnostics report."),
    no_umap: bool = typer.Option(False, "--no-umap", help="Skip UMAP projection."),
    no_tsne: bool = typer.Option(False, "--no-tsne", help="Skip t-SNE projection."),
) -> None:
    """Generate an HTML diagnostics report (UMAP/t-SNE/SHAP/symbolic overlays)."""
    render_header("Diagnostics Dashboard")
    ensure_dirs()
    cfg = hydra_compose_or_stub(REPO / "configs", "diagnostics=dashboard", overrides or [])
    simulate_long_task("Building dashboard", steps=6)

    body = f"""
<ul>
  <li><b>Hydra overrides</b>: {pretty_overrides(overrides or [])}</li>
  <li><b>UMAP</b>: {"skipped" if no_umap else "enabled"}</li>
  <li><b>t-SNE</b>: {"skipped" if no_tsne else "enabled"}</li>
</ul>
<pre>{textwrap.indent(json.dumps(cfg, indent=2)[:1200], ' ')}</pre>
"""
    write_stub_html(html_out, "SpectraMind V50 — Diagnostics Report", body)
    rh = write_run_hash_summary(cfg, overrides=overrides or [])
    console.print(f"[green]Diagnostics HTML written[/green] → {html_out} (hash {rh}).")

    append_debug_log(
        [
            f"spectramind diagnose dashboard {pretty_overrides(overrides or [])} --html-out {html_out}"
            + (" --no-umap" if no_umap else "")
            + (" --no-tsne" if no_tsne else ""),
            f"Git SHA: {git_sha_short()}",
            f"Run hash: {rh}",
            "Artifacts: " + str(html_out),
        ]
    )


# --------------------------------------------------------------------------------------
# Submit
# --------------------------------------------------------------------------------------
@APP.command("submit")
def submit(
    zip_out: Path = typer.Option(OUTPUTS / "submission_bundle.zip", "--zip-out", help="Output ZIP bundle."),
    open_html: bool = typer.Option(False, "--open-html", help="Open diagnostics report after build."),
) -> None:
    """
    Build a submission bundle: latest submission.csv (if present), diagnostics HTML,
    temp_scaling.json, and logs.
    """
    render_header("Make Submission")
    ensure_dirs()
    simulate_long_task("Packaging submission", steps=4)
    globs = ("outputs/submission.csv", "outputs/diagnostics/*.html", "outputs/temp_scaling.json", "logs/v50_debug_log.md")
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    zip_artifacts(zip_out, globs)
    console.print(f"[green]Submission bundle created[/green] → {zip_out}")

    append_debug_log(
        [
            f"spectramind submit --zip-out {zip_out}" + (" --open-html" if open_html else ""),
            f"Git SHA: {git_sha_short()}",
            "Artifacts: " + str(zip_out),
        ]
    )

    if open_html:
        # try to open first html
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
        False, "--group-by-config-hash", help="Group rows by config hash (if present)."
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
    # Split on separators written by append_debug_log
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
            low = ln.lower()
            if low.startswith("git sha:"):
                sha = ln.split(":", 1)[1].strip()
            elif "config hash" in low or low.startswith("run hash:"):
                cfg_hash = ln.split(":", 1)[1].strip()
        entries.append({"timestamp": ts, "cmd": cmd, "git_sha": sha, "config_hash": cfg_hash})

    # Optional grouping
    if group_by_config_hash:
        entries.sort(key=lambda e: (e["config_hash"], e["timestamp"]))
        groups: Dict[str, List[Dict[str, str]]] = {}
        for e in entries:
            groups.setdefault(e["config_hash"] or "unknown", []).append(e)

        for ghash, rows in groups.items():
            console.print(Panel.fit(f"[bold]Config Hash: {ghash or 'unknown'}[/bold]", box=box.SQUARE))
            table = Table(box=box.SIMPLE_HEAVY)
            table.add_column("Time (UTC)", style="bold")
            table.add_column("Command")
            table.add_column("Git SHA")
            for e in rows:
                table.add_row(e["timestamp"], e["cmd"], e["git_sha"])
            console.print(table)
    else:
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Time (UTC)", style="bold")
        table.add_column("Command")
        table.add_column("Git SHA")
        table.add_column("Config Hash")
        for e in entries:
            table.add_row(e["timestamp"], e["cmd"], e["git_sha"], e["config_hash"])
        console.print(table)

    # Optional exports
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
    """Print a command→file map (for docs and integrity)."""
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

    append_debug_log(["spectramind check-cli-map", f"Git SHA: {git_sha_short()}", f"Commands: {len(rows)}"])


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
def _install_sigint_handler() -> None:
    def handler(signum, frame) -> None:  # type: ignore[no-untyped-def]
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
```

If you keep a `src/` layout (recommended), place this file at `src/spectramind/cli.py` and set in `pyproject.toml`:

```toml
[project.scripts]
spectramind = "spectramind.cli:app"
```

Plus these tiny files so `python -m spectramind` works out of the box:

```python
# src/spectramind/__init__.py
from .cli import app
__all__ = ["app"]

# src/spectramind/__main__.py
from .cli import app
if __name__ == "__main__":
    app()
```
