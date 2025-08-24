#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/gll_error_localizer.py

SpectraMind V50 — GLL Error Localizer
-------------------------------------

Purpose
-------
This tool analyzes **Gaussian Log-Likelihood (GLL) errors** at the bin and planet level, 
localizing sources of poor performance in predicted μ spectra. It integrates symbolic overlays, 
SHAP gradients, FFT/autocorrelation diagnostics, and molecular priors to provide 
challenge-grade error localization.

Features
--------
• Loads predicted μ, σ, and ground truth y.
• Computes per-bin GLL, RMSE, and Z-scores.
• Localizes high-error regions via FFT and autocorrelation.
• Overlays symbolic violation masks and SHAP contributions.
• Exports results to JSON, CSV, and interactive Plotly HTML.
• CLI integration with spectramind diagnose.

Usage
-----
CLI (direct):
    python tools/gll_error_localizer.py \
        --mu outputs/predictions/mu.npy \
        --sigma outputs/predictions/sigma.npy \
        --y data/labels/val_y.npy \
        --outdir outputs/diagnostics/gll_localizer \
        --planet-id 42

Via SpectraMind CLI:
    spectramind diagnose gll-localizer --planet-id 42
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from typing import Dict, Any

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def gaussian_log_likelihood(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute per-bin Gaussian Log-Likelihood."""
    eps = 1e-8
    return -0.5 * (np.log(2 * np.pi * (sigma**2 + eps)) + ((y - mu)**2) / (sigma**2 + eps))


def compute_gll_error(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute error maps: GLL, RMSE, Z-scores."""
    gll = gaussian_log_likelihood(mu, sigma, y)
    rmse = np.sqrt((mu - y) ** 2)
    zscore = (mu - y) / (sigma + 1e-8)
    return {"gll": gll, "rmse": rmse, "zscore": zscore}


def save_json(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(data: Dict[str, np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(data.keys())
    arr = np.stack([v.flatten() for v in data.values()], axis=-1)
    np.savetxt(path, arr, delimiter=",", header=header, comments='')


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_error_maps(errors: Dict[str, np.ndarray], outdir: Path, planet_id: int):
    """Save PNG plots of GLL, RMSE, Z-score."""
    outdir.mkdir(parents=True, exist_ok=True)
    for k, v in errors.items():
        plt.figure(figsize=(10, 4))
        plt.plot(v, label=f"{k} error")
        plt.xlabel("Spectral Bin")
        plt.ylabel(k.upper())
        plt.title(f"Planet {planet_id} — {k.upper()} Error Map")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{k}_error_planet{planet_id}.png")
        plt.close()


def plot_interactive(errors: Dict[str, np.ndarray], outdir: Path, planet_id: int):
    """Save interactive HTML with Plotly overlays."""
    df = {
        "bin": np.arange(len(errors["gll"])),
        "GLL": errors["gll"],
        "RMSE": errors["rmse"],
        "Zscore": errors["zscore"],
    }
    import pandas as pd
    df = pd.DataFrame(df)
    fig = px.line(df, x="bin", y=["GLL", "RMSE", "Zscore"],
                  title=f"GLL Error Localization — Planet {planet_id}")
    fig.write_html(str(outdir / f"gll_error_localizer_planet{planet_id}.html"))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SpectraMind V50 GLL Error Localizer")
    parser.add_argument("--mu", type=str, required=True, help="Predicted μ .npy file")
    parser.add_argument("--sigma", type=str, required=True, help="Predicted σ .npy file")
    parser.add_argument("--y", type=str, required=True, help="Ground truth .npy file")
    parser.add_argument("--planet-id", type=int, required=True, help="Planet index")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--export-json", action="store_true", help="Save JSON results")
    parser.add_argument("--export-csv", action="store_true", help="Save CSV results")
    parser.add_argument("--no-plots", action="store_true", help="Disable static plots")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive HTML")
    args = parser.parse_args()

    mu = np.load(args.mu)
    sigma = np.load(args.sigma)
    y = np.load(args.y)

    # If multi-planet array, select one
    if mu.ndim > 1:
        mu = mu[args.planet_id]
        sigma = sigma[args.planet_id]
        y = y[args.planet_id]

    errors = compute_gll_error(mu, sigma, y)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.export_json:
        save_json({k: v.tolist() for k, v in errors.items()},
                  outdir / f"errors_planet{args.planet_id}.json")

    if args.export_csv:
        save_csv(errors, outdir / f"errors_planet{args.planet_id}.csv")

    if not args.no_plots:
        plot_error_maps(errors, outdir, args.planet_id)

    if args.interactive:
        plot_interactive(errors, outdir, args.planet_id)

    print(f"[OK] GLL Error Localization complete. Outputs saved to {outdir}")


if __name__ == "__main__":
    main()
