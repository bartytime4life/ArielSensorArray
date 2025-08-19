#!/usr/bin/env python3
# scripts/generate_placeholder_plots.py
# -----------------------------------------------------------------------------
# Generate placeholder/demo plots for SpectraMind V50 and save them under
# assets/plots/. These are NON-SCIENTIFIC illustrations used for docs/CI.
#
# Run:
#   poetry run python scripts/generate_placeholder_plots.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

# Safe, headless matplotlib
import matplotlib
matplotlib.use("Agg")  # no GUI backend required
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PLOTS = ASSETS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(1337)


def _save(fig: plt.Figure, name: str, dpi: int = 160) -> None:
    out_png = PLOTS / f"{name}.png"
    out_svg = PLOTS / f"{name}.svg"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote: {out_png}  (and .svg)")


def plot_sample_spectrum():
    """Synthetic transmission spectrum with Gaussian absorption features."""
    wl = np.linspace(0.5, 3.5, 600)  # microns
    baseline = 0.015 + 0.002*np.sin(2*np.pi*wl/3.0)
    dips = (
        0.004 * np.exp(-0.5*((wl-1.4)/0.10)**2) +
        0.003 * np.exp(-0.5*((wl-1.9)/0.06)**2) +
        0.005 * np.exp(-0.5*((wl-2.8)/0.12)**2)
    )
    noise = RNG.normal(0, 0.0003, wl.shape)
    depth = baseline + dips + noise  # "transit depth"

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(wl, depth*1e3, lw=2, color="#1a237e")
    ax.set_title("Placeholder Transmission Spectrum (DEMO)", fontsize=12)
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Transit Depth (ppt)")  # parts-per-thousand for visual scale
    ax.grid(alpha=0.2)
    ax.set_xlim(wl.min(), wl.max())
    for mu in (1.4, 1.9, 2.8):
        ax.axvline(mu, color="#ff9800", lw=1.25, ls="--", alpha=0.6)
    ax.text(0.51, ax.get_ylim()[1]*0.92,
            "NOTE: Placeholder figure for docs/CI — not scientific output.",
            fontsize=9, color="#444")
    _save(fig, "sample_spectrum")


def plot_shap_overlay_example():
    """Synthetic SHAP overlay bar chart: top features with +/- contributions."""
    n = 20
    features = [f"feat_{i:02d}" for i in range(n)]
    shap_vals = RNG.normal(0, 0.25, n)
    order = np.argsort(np.abs(shap_vals))[::-1]
    features = [features[i] for i in order][:15]
    shap_vals = shap_vals[order][:15]
    colors = np.where(shap_vals >= 0, "#4caf50", "#f44336")

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(features))[::-1]
    ax.barh(y, shap_vals, color=colors, alpha=0.9)
    ax.set_yticks(y, features[::-1])
    ax.set_title("Placeholder SHAP Overlay (DEMO)", fontsize=12)
    ax.set_xlabel("SHAP Value (Δμ)")
    ax.grid(axis="x", alpha=0.2)
    ax.axvline(0, color="#333", lw=1)
    ax.text(0.01, 0.01, "DEMO — not computed from the model.",
            transform=ax.transAxes, fontsize=9, color="#444")
    _save(fig, "shap_overlay_example")


def _cluster_points(k=3, points_per=100, spread=0.5, seed=2025):
    rng = np.random.default_rng(seed)
    centers = np.array([[-2, -2], [0, 1.5], [2.5, -1.0]])[:k]
    X, y = [], []
    for i, c in enumerate(centers):
        X.append(c + rng.normal(0, spread, size=(points_per, 2)))
        y.append(np.full(points_per, i))
    return np.vstack(X), np.concatenate(y)


def plot_umap_like():
    """Synthetic 2D cluster scatter labeled as 'UMAP (DEMO)'."""
    X, y = _cluster_points(k=3, points_per=120, spread=0.55, seed=7)
    palette = ["#1a237e", "#ff9800", "#4caf50"]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i in np.unique(y):
        m = y == i
        ax.scatter(X[m, 0], X[m, 1], s=14, c=palette[i], label=f"cluster {i}", alpha=0.85)
    ax.legend(title="Cluster", loc="best", fontsize=9)
    ax.set_title("UMAP Projection (DEMO Placeholder)", fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    ax.text(0.01, 0.01, "DEMO — synthetic 2D clusters", transform=ax.transAxes, fontsize=9, color="#444")
    _save(fig, "umap_clusters")


def plot_tsne_like():
    """Synthetic 2D cluster scatter labeled as 't‑SNE (DEMO)'."""
    X, y = _cluster_points(k=3, points_per=120, spread=0.65, seed=42)
    palette = ["#3949ab", "#ffa726", "#66bb6a"]
    markers = ["o", "s", "D"]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i in np.unique(y):
        m = y == i
        ax.scatter(X[m, 0], X[m, 1], s=16, c=palette[i], marker=markers[i], label=f"cluster {i}", alpha=0.9)
    ax.legend(title="Cluster", loc="best", fontsize=9)
    ax.set_title("t‑SNE Projection (DEMO Placeholder)", fontsize=12)
    ax.set_xlabel("t‑SNE-1"); ax.set_ylabel("t‑SNE-2")
    ax.grid(alpha=0.2)
    ax.text(0.01, 0.01, "DEMO — synthetic 2D clusters", transform=ax.transAxes, fontsize=9, color="#444")
    _save(fig, "tsne_projection")


def main():
    print(f"[INFO] Writing assets to: {PLOTS}")
    plot_sample_spectrum()
    plot_shap_overlay_example()
    plot_umap_like()
    plot_tsne_like()
    print("[DONE] Placeholder plots generated.")


if __name__ == "__main__":
    main()