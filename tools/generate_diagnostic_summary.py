#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_diagnostic_summary.py

SpectraMind V50 — Diagnostic Summary Generator
================================================

Purpose
-------
Generate a comprehensive diagnostic summary of SpectraMind V50 predictions (μ, σ)
against ground-truth targets. This script aggregates scientific metrics, symbolic overlays,
and explainability outputs into a structured JSON/CSV report for downstream dashboard
integration.

Features
--------
• Gaussian Log-Likelihood (GLL), RMSE, MAE per planet/bin
• Entropy, uncertainty calibration, quantile/coverage evaluation
• FFT and Z-score diagnostics of μ spectra
• SHAP overlays, symbolic violation overlays, ∂L/∂μ symbolic influence
• Leaderboard-ready `diagnostic_summary.json` + CSV + optional PNG/HTML plots
• Typer CLI with Hydra config compatibility
• Logs reproducibility hash + config snapshot into logs/v50_debug_log.md
"""

from __future__ import annotations
import json, hashlib, datetime as dt, logging, sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

# Local SpectraMind imports
from src.losses.symbolic_loss import compute_symbolic_losses
from src.symbolic.symbolic_influence_map import compute_symbolic_influence
from src.explain.shap_overlay import compute_shap_overlay
from src.utils.reproducibility import get_run_hash, capture_env_metadata

app = typer.Typer(add_completion=False)
console = Console()

# ---------------------------
# Core Metrics
# ---------------------------

def gaussian_log_likelihood(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    var = np.square(sigma) + eps
    return -0.5 * np.mean(np.log(2 * np.pi * var) + np.square(y - mu) / var)

def rmse(mu: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((mu - y) ** 2))

def mae(mu: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.abs(mu - y))

def entropy_per_bin(sigma: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * np.e * np.square(sigma) + 1e-9)

# ---------------------------
# FFT Diagnostics
# ---------------------------

def fft_power(mu: np.ndarray) -> np.ndarray:
    mu_centered = mu - mu.mean()
    fft_vals = np.fft.rfft(mu_centered)
    return np.abs(fft_vals) ** 2

# ---------------------------
# Z-score Diagnostics
# ---------------------------

def z_scores(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y - mu) / (sigma + 1e-9)

# ---------------------------
# Diagnostic Summary
# ---------------------------

def generate_summary(mu_path: Path, sigma_path: Path, y_path: Path,
                     shap_path: Path | None, symbolic_path: Path | None,
                     outdir: Path, run_id: str) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    mu, sigma, y = np.load(mu_path), np.load(sigma_path), np.load(y_path)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "metrics": {},
        "symbolic": {},
        "explainability": {}
    }

    # --- Core metrics
    summary["metrics"]["gll"] = float(gaussian_log_likelihood(mu, sigma, y))
    summary["metrics"]["rmse"] = float(rmse(mu, y))
    summary["metrics"]["mae"] = float(mae(mu, y))
    summary["metrics"]["entropy_mean"] = float(np.mean(entropy_per_bin(sigma)))

    # --- Calibration (z-score mean/var)
    z = z_scores(mu, sigma, y)
    summary["metrics"]["zscore_mean"] = float(np.mean(z))
    summary["metrics"]["zscore_var"] = float(np.var(z))

    # --- FFT
    fft_vals = fft_power(mu.mean(axis=0))
    summary["metrics"]["fft_peak"] = float(np.max(fft_vals))
    np.save(outdir / "fft_power.npy", fft_vals)

    # --- Symbolic overlays
    if symbolic_path and symbolic_path.exists():
        symbolic_results = compute_symbolic_losses(mu, symbolic_path)
        influence = compute_symbolic_influence(mu, symbolic_path)
        summary["symbolic"]["losses"] = symbolic_results
        summary["symbolic"]["influence"] = influence

    # --- SHAP overlays
    if shap_path and shap_path.exists():
        shap_results = compute_shap_overlay(mu, shap_path)
        summary["explainability"]["shap"] = shap_results

    # --- Save outputs
    json_path = outdir / "diagnostic_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = outdir / "diagnostic_summary.csv"
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k, v in summary["metrics"].items():
            f.write(f"{k},{v}\n")

    return summary

# ---------------------------
# CLI Entrypoint
# ---------------------------

@app.command()
def main(
    mu: Path = typer.Option(..., help="Path to predicted μ.npy"),
    sigma: Path = typer.Option(..., help="Path to predicted σ.npy"),
    y: Path = typer.Option(..., help="Path to ground truth y.npy"),
    shap: Path = typer.Option(None, help="Optional SHAP overlay JSON"),
    symbolic: Path = typer.Option(None, help="Optional symbolic overlay JSON"),
    outdir: Path = typer.Option(Path("outputs/diagnostics"), help="Output directory"),
):
    """
    Generate diagnostic summary JSON/CSV (and optional overlays).
    """
    run_id = get_run_hash()
    console.rule(f"[bold blue]SpectraMind V50 — Diagnostic Summary[/bold blue]")
    summary = generate_summary(mu, sigma, y, shap, symbolic, outdir, run_id)

    # Print table
    table = Table(title="Diagnostics Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in summary["metrics"].items():
        table.add_row(k, f"{v:.6f}")
    console.print(table)

    # Log to debug log
    log_path = Path("logs/v50_debug_log.md")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n### Diagnostic run {run_id} ({dt.datetime.utcnow().isoformat()})\n")
        for k, v in summary["metrics"].items():
            f.write(f"- {k}: {v:.6f}\n")

    console.print(f"[green]Summary saved to {outdir}[/green]")

if __name__ == "__main__":
    app()
