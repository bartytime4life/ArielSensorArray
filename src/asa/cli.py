#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArielSensorArray CLI (SpectraMind V50)

Commands:
  selftest        Smoke checks (env, paths, optional deps)
  train           Toy "training" that writes a model artifact
  predict         Produce submission.csv (+ preds.pt) with deterministic toy preds
  calibrate       Temperature-scale predictions and write submission_calibrated.csv
  diagnose        Build a tiny HTML dashboard (summary stats)
  submit          Bundle and (optionally) validate submission artifacts
  analyze-log     Parse a markdown log into CSV/MD tables (optional helper)

Notes:
- Torch/Hydra are optional. If missing, we fall back to stdlib paths.
- All outputs under ./outputs by default.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import statistics
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import typer
from rich.console import Console
from rich.table import Table

# --- Optional deps (soft) ---
try:  # torch is optional
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

try:  # hydra is optional
    from hydra import compose, initialize  # type: ignore

    _HYDRA_OK = True
except Exception:
    _HYDRA_OK = False

try:  # importlib.metadata for --version
    from importlib.metadata import version as _pkg_version
except Exception:
    def _pkg_version(_: str) -> str:  # type: ignore
        return "0.0.0"

# ---------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------
app = typer.Typer(help="ArielSensorArray CLI (SpectraMind V50)")
console = Console()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
DEFAULT_OUT = Path("outputs")
DEFAULT_BINS = 283
DEFAULT_IDS = [str(i) for i in range(4)]  # tiny toy size by default


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    try:
        import numpy as np  # not required, just in case
        np.random.seed(seed)  # type: ignore
    except Exception:
        pass
    if _TORCH_OK:
        torch.manual_seed(seed)  # type: ignore
        if torch.cuda.is_available():  # type: ignore
            torch.cuda.manual_seed_all(seed)  # type: ignore


def _load_bins_from_hydra(config_path: Optional[str], config_name: Optional[str]) -> Optional[int]:
    if not _HYDRA_OK or not config_path or not config_name:
        return None
    # hydra.initialize changes CWD; scope it tightly.
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
    # Try common keys
    for key in ("bins", "model.bins", "data.bins"):
        node = cfg
        try:
            for part in key.split("."):
                node = node[part]
            val = int(node)
            if val > 0:
                return val
        except Exception:
            continue
    return None


def _write_csv(out_csv: Path, ids: Sequence[str], mu_rows: Sequence[Sequence[float]]) -> None:
    _ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        bins = len(mu_rows[0]) if mu_rows else DEFAULT_BINS
        w.writerow(["planet_id", *[f"bin{i}" for i in range(bins)]])
        for pid, row in zip(ids, mu_rows):
            w.writerow([pid, *[f"{x:.6f}" for x in row]])


def _save_preds_pt(path: Path, mu: Sequence[Sequence[float]], sigma: Sequence[Sequence[float]]) -> None:
    _ensure_dir(path.parent)
    payload = {"mu": mu, "sigma": sigma}
    if _TORCH_OK:
        # Save tensor-like payload (compatible with earlier scripts)
        tm = torch.tensor(mu)  # type: ignore
        ts = torch.tensor(sigma)  # type: ignore
        torch.save({"mu": tm, "sigma": ts}, path)  # type: ignore
    else:
        # Fallback: simple JSON bytes stored inside .pt for compatibility
        import pickle

        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        path.write_bytes(blob)


def _toy_predictor(bins: int, n: int, device: str = "cpu") -> Tuple[List[List[float]], List[List[float]]]:
    """
    Deterministic toy predictor. If torch is available we use it (fast),
    else we fall back to stdlib random.gauss to avoid extra dependencies.
    Returns (mu, sigma) with shapes (n, bins).
    """
    _set_seed(1337)
    if _TORCH_OK:
        with torch.no_grad():  # type: ignore
            fgs1 = torch.randn(n, 1, 128, device=device)  # type: ignore
            airs = torch.randn(n, 1, bins, device=device)  # type: ignore
            mu = airs.squeeze(1) + 0.1 * fgs1.mean(dim=-1, keepdim=True).expand(-1, bins)  # type: ignore
            base = 0.5 * airs.squeeze(1).abs().mean(dim=0).unsqueeze(0).expand(n, -1)  # type: ignore
            sigma = torch.nn.functional.softplus(base) + 1e-3  # type: ignore
            mu_l = mu.detach().cpu().tolist()
            sg_l = sigma.detach().cpu().tolist()
        return mu_l, sg_l

    # stdlib fallback
    mu: List[List[float]] = []
    sigma: List[List[float]] = []
    for _ in range(n):
        g_mean = sum(random.gauss(0.0, 1.0) for _ in range(128)) / 128.0
        row = [random.gauss(0.0, 1.0) + 0.1 * g_mean for _ in range(bins)]
        mu.append(row)
    # fixed per-bin stddev estimate
    per_bin = [abs(random.gauss(0.5, 0.2)) for _ in range(bins)]
    per_bin = [math.log1p(math.exp(x)) + 1e-3 for x in per_bin]  # softplus-ish
    for _ in range(n):
        sigma.append(per_bin[:])
    return mu, sigma


def _validate_submission_like(path: Path, bins: int) -> None:
    """
    Light-weight validator (subset of scripts/validate_submission.py).
    Ensures header shape and numeric, finite rows.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            raise ValueError("empty CSV")
        exp = ["planet_id"] + [f"bin{i}" for i in range(bins)]
        if header != exp:
            raise ValueError(f"header mismatch ({len(header)} cols), expected {len(exp)}")
        for i, row in enumerate(r, start=1):
            if len(row) != len(exp):
                raise ValueError(f"row {i} has {len(row)} cols (expected {len(exp)})")
            for j, v in enumerate(row[1:], start=1):
                x = float(v)
                if not math.isfinite(x):
                    raise ValueError(f"row {i} col {j} not finite")


def _html_report(out_path: Path, mu: Sequence[Sequence[float]]) -> None:
    """
    Tiny, dependency-light diagnostics HTML with a few summary stats.
    """
    _ensure_dir(out_path.parent)
    flat = [x for row in mu for x in row]
    if not flat:
        flat = [0.0]
    mean = statistics.fmean(flat)
    stdev = statistics.pstdev(flat)
    qmin = min(flat)
    qmax = max(flat)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SpectraMind V50 — Diagnostics</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 1rem; max-width: 680px; }}
code {{ background: #f6f8fa; padding: 0.2rem 0.4rem; border-radius: 4px; }}
</style></head><body>
<h1>Diagnostics</h1>
<div class="card">
  <p><b>Points:</b> {len(flat):,} &nbsp;|&nbsp; <b>Bins/row:</b> {len(mu[0]) if mu else 0} &nbsp;|&nbsp; <b>Rows:</b> {len(mu)}</p>
  <p><b>Mean:</b> {mean:.6f} &nbsp; <b>Std:</b> {stdev:.6f} &nbsp; <b>Min:</b> {qmin:.6f} &nbsp; <b>Max:</b> {qmax:.6f}</p>
  <p>Generated by <code>asa diagnose</code>.</p>
</div>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------
@app.callback()
def _version_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=lambda v: (_pkg_version("arielsensorarray") if v else None),
        help="Show package version and exit.",
        is_eager=True,
    )
) -> None:
    if version:
        console.print(_pkg_version("arielsensorarray"))
        raise typer.Exit()


@app.command()
def selftest(
    fast: bool = typer.Option(False, help="Run a lightweight subset."),
    dry_run: bool = typer.Option(False, help="Run without writing artifacts."),
) -> None:
    """Basic smoke test to ensure the CLI and environment are sane."""
    console.rule("[bold blue]Selftest")
    console.print(f"[cyan]Python[/]: {sys.version.split()[0]}")
    console.print(f"[cyan]Torch available[/]: {_TORCH_OK}")
    console.print(f"[cyan]Hydra available[/]: {_HYDRA_OK}")
    console.print(f"[cyan]CWD[/]: {Path.cwd()}")
    if not dry_run:
        _ensure_dir(DEFAULT_OUT / "diagnostics")
    if not fast:
        console.print("[green]Selftest completed[/]" + (" (dry run)" if dry_run else ""))
    else:
        console.print("[green]Fast selftest completed[/]" + (" (dry run)" if dry_run else ""))


@app.command()
def train(
    epochs: int = typer.Option(1, min=1, help="Toy epochs (no-op)"),
    out_dir: Path = typer.Option(DEFAULT_OUT, help="Where to write artifacts"),
    dry_run: bool = typer.Option(False, help="Run without side effects"),
) -> None:
    """Toy 'training' that writes a tiny model artifact."""
    console.rule("[bold blue]Train")
    _set_seed()
    if dry_run:
        console.print("[yellow]Dry-run: skipping writes[/]")
        return
    _ensure_dir(out_dir)
    (out_dir / "model.json").write_text(json.dumps({"epochs": epochs, "seed": 1337}), encoding="utf-8")
    console.print(f"[green]Wrote[/] {out_dir/'model.json'}")


@app.command()
def predict(
    out_csv: Path = typer.Option(DEFAULT_OUT / "submission.csv", help="Where to write submission CSV"),
    out_pt: Path = typer.Option(DEFAULT_OUT / "preds.pt", help="Where to write torch preds file"),
    bins: Optional[int] = typer.Option(None, help="Number of spectral bins (default: Hydra or 283)"),
    ids: Optional[Path] = typer.Option(None, help="Text file with planet_id (one per line) to set row order"),
    hydra_config_path: Optional[str] = typer.Option(None, help="Hydra config path (e.g., ../configs)"),
    hydra_config_name: Optional[str] = typer.Option(None, help="Hydra config name (e.g., config_v50)"),
) -> None:
    """Generate deterministic toy predictions and produce submission CSV (+ preds.pt)."""
    console.rule("[bold blue]Predict")
    _set_seed()
    bins_resolved = bins or _load_bins_from_hydra(hydra_config_path, hydra_config_name) or DEFAULT_BINS

    # IDs
    if ids and ids.exists():
        id_list = [s for s in ids.read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        id_list = DEFAULT_IDS

    device = "cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu"  # type: ignore
    mu, sigma = _toy_predictor(bins=bins_resolved, n=len(id_list), device=device)
    _write_csv(out_csv, id_list, mu)
    _save_preds_pt(out_pt, mu, sigma)

    console.print(
        f"[green]Wrote[/] {out_csv} and {out_pt} "
        f"([cyan]{len(id_list)}[/] rows × [cyan]{bins_resolved}[/] bins, device={device})"
    )


@app.command()
def calibrate(
    in_pt: Path = typer.Option(DEFAULT_OUT / "preds.pt", help="Input preds file"),
    out_csv: Path = typer.Option(DEFAULT_OUT / "submission_calibrated.csv", help="Output calibrated CSV"),
    temperature: float = typer.Option(1.0, min=1e-6, help="Temperature scaling factor (μ/temperature)"),
) -> None:
    """Apply simple temperature scaling to μ and write a calibrated submission CSV."""
    console.rule("[bold blue]Calibrate")
    if not in_pt.exists():
        raise typer.BadParameter(f"missing {in_pt} (run predict first)")
    # Load μ the way we saved it
    try:
        if _TORCH_OK:
            payload = torch.load(in_pt, map_location="cpu")  # type: ignore
            mu = payload["mu"].detach().cpu().tolist()
        else:
            import pickle

            payload = pickle.loads(in_pt.read_bytes())
            mu = payload["mu"]
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(f"failed to read {in_pt}: {e!r}")

    # Scale
    mu_cal = [[x / temperature for x in row] for row in mu]

    # IDs inferred as 0..N-1
    ids = [str(i) for i in range(len(mu_cal))]
    bins = len(mu_cal[0]) if mu_cal else DEFAULT_BINS
    _write_csv(out_csv, ids, mu_cal)
    console.print(f"[green]Wrote[/] {out_csv} (temperature={temperature}, bins={bins})")


@app.command("diagnose")
def diagnose_dashboard(
    html_out: Path = typer.Option(DEFAULT_OUT / "diagnostics" / "report.html", help="Diagnostics HTML output"),
    from_pt: Path = typer.Option(DEFAULT_OUT / "preds.pt", help="Load stats from preds.pt (μ)"),
) -> None:
    """Write a tiny HTML diagnostics dashboard with summary stats of μ."""
    console.rule("[bold blue]Diagnose")
    if not from_pt.exists():
        raise typer.BadParameter(f"missing {from_pt} (run predict first)")
    try:
        if _TORCH_OK:
            payload = torch.load(from_pt, map_location="cpu")  # type: ignore
            mu = payload["mu"].detach().cpu().tolist()
        else:
            import pickle

            payload = pickle.loads(from_pt.read_bytes())
            mu = payload["mu"]
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(f"failed to read {from_pt}: {e!r}")

    _html_report(html_out, mu)
    console.print(f"[green]Wrote[/] {html_out}")


@app.command()
def submit(
    bundle: bool = typer.Option(True, help="Create a submission.zip bundle in outputs/"),
    validate: bool = typer.Option(True, help="Validate header/shape/values before bundling"),
    csv_path: Path = typer.Option(DEFAULT_OUT / "submission.csv", help="CSV to include"),
    bins: int = typer.Option(DEFAULT_BINS, help="Expected number of bins for validation"),
) -> None:
    """Bundle submission.csv into outputs/submission.zip and (optionally) validate it."""
    console.rule("[bold blue]Submit")
    if validate:
        _validate_submission_like(csv_path, bins=bins)
        console.print(f"[green]Validation OK[/] for {csv_path}")

    if bundle:
        zip_path = csv_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, csv_path.name)
        console.print(f"[green]Wrote[/] {zip_path}")


@app.command()
def analyze_log(
    md_in: Path = typer.Option(Path("v50_debug_log.md"), help="Input markdown log"),
    md_out: Path = typer.Option(DEFAULT_OUT / "log_table.md", help="Output table (markdown)"),
    csv_out: Path = typer.Option(DEFAULT_OUT / "log_table.csv", help="Output table (CSV)"),
) -> None:
    """Very small helper that extracts lines starting with '[event]' into tables."""
    console.rule("[bold blue]Analyze Log")
    if not md_in.exists():
        console.print("[yellow]No log found, skipping[/]")
        return
    rows: List[Tuple[str, str]] = []
    for line in md_in.read_text(encoding="utf-8").splitlines():
        if line.startswith("[event]"):
            # format: [event] KEY: VALUE
            body = line[len("[event]") :].strip()
            if ":" in body:
                k, v = body.split(":", 1)
                rows.append((k.strip(), v.strip()))
    # write md
    _ensure_dir(md_out.parent)
    with md_out.open("w", encoding="utf-8") as f:
        f.write("| key | value |\n|---|---|\n")
        for k, v in rows:
            f.write(f"| {k} | {v} |\n")
    # write csv
    _ensure_dir(csv_out.parent)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "value"]); w.writerows(rows)
    console.print(f"[green]Wrote[/] {md_out} and {csv_out}")


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app()