#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot benchmark bars for GLL, MAE, Coverage, and Runtime and save to outputs/diagnostics/.
Assumes the table values in bestversion.md; update DATA dict below if numbers change.

Usage:
  python tools/plot_benchmarks.py
"""

import os
import matplotlib.pyplot as plt

# ==== Editable data (keep in sync with "Benchmark Results" table) ====
MODELS = [
    "Thang Do Duc",
    "V1ctorious3010",
    "Fawad Awan",
    "SpectraMind V50",
]

DATA = {
    "GLL": [0.495, 0.481, 0.487, 0.463],         # lower is better
    "MAE": [0.043, 0.041, 0.042, 0.038],         # lower is better
    "Coverage": [None, None, None, 0.81],        # higher is better (None => missing)
    "Runtime": [1.0, 8.5, 3.0, 7.5],             # lower is better (hours)
}

OUT_DIR = os.path.join("outputs", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

def _plot_metric(metric_name, values, fname, better="lower"):
    x = list(range(len(MODELS)))
    # Mask Nones (e.g., coverage missing for baselines)
    plot_x, plot_vals, plot_labels = [], [], []
    colors = []
    for i, v in enumerate(values):
        if v is None:
            continue
        plot_x.append(i)
        plot_vals.append(v)
        plot_labels.append(MODELS[i])
        colors.append("#2c7fb8" if "SpectraMind" not in MODELS[i] else "#41ab5d")

    plt.figure(figsize=(7.0, 3.8))
    plt.bar(plot_x, plot_vals, color=colors, edgecolor="#2b2b2b", linewidth=0.4)
    plt.xticks(plot_x, plot_labels, rotation=10, ha="right")
    plt.ylabel(metric_name + (" (lower is better)" if better == "lower" else " (higher is better)"))
    plt.title(f"{metric_name} comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=160)
    plt.close()

def main():
    _plot_metric("GLL", DATA["GLL"], "benchmark_gll.png", better="lower")
    _plot_metric("MAE", DATA["MAE"], "benchmark_mae.png", better="lower")
    _plot_metric("Coverage @80%", DATA["Coverage"], "benchmark_coverage.png", better="higher")
    _plot_metric("Runtime (hours)", DATA["Runtime"], "benchmark_runtime.png", better="lower")

if __name__ == "__main__":
    main()