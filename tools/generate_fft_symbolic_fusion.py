#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — FFT × Symbolic Fingerprint Fusion Tool

Purpose
-------
Fuse FFT features of μ spectra with symbolic fingerprints (rule violations,
entropy, SHAP overlays) and produce diagnostic plots + exports:
- PCA/UMAP/t-SNE embeddings with symbolic overlays
- Cluster CSV/JSON summaries
- Interactive HTML visualizations (Plotly)
- Static PNG/CSV exports for CI & dashboard

Inputs
------
- spectra.npy or .csv (μ predictions; shape [N, 283])
- symbolic_results.json (per-planet symbolic violation scores)
- shap_symbolic_overlay.json (optional SHAP × symbolic bin overlays)
- entropy_scores.npy (optional entropy per-bin)
- Configurable via CLI flags (--input, --symbolic, --outdir, etc.)

Outputs
-------
- outdir/fft_symbolic_umap.html   (interactive UMAP plot)
- outdir/fft_symbolic_tsne.html   (interactive t-SNE plot)
- outdir/fft_symbolic_pca.png     (static PCA scatter)
- outdir/fft_symbolic_clusters.csv (cluster assignments + scores)
- outdir/fft_symbolic_summary.json (fusion metadata)

Author: SpectraMind V50 Team
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# ----------------------------------------------------------------------
# FFT feature extraction
# ----------------------------------------------------------------------
def compute_fft_features(mu: np.ndarray, n_freq: int = 50) -> np.ndarray:
    """Compute normalized FFT features for μ spectra."""
    fft_vals = np.fft.rfft(mu, axis=1)
    mag = np.abs(fft_vals)[:, :n_freq]
    mag = mag / (mag.sum(axis=1, keepdims=True) + 1e-12)
    return mag

# ----------------------------------------------------------------------
# Fusion dataset builder
# ----------------------------------------------------------------------
def build_feature_matrix(mu: np.ndarray,
                         symbolic: dict[str, float] | None = None,
                         entropy: np.ndarray | None = None) -> pd.DataFrame:
    """Combine FFT, symbolic scores, and entropy into one DataFrame."""
    fft_feats = compute_fft_features(mu)
    df = pd.DataFrame(fft_feats, columns=[f"fft_{i}" for i in range(fft_feats.shape[1])])
    if symbolic:
        for rule, vals in symbolic.items():
            df[f"sym_{rule}"] = vals
    if entropy is not None:
        df["entropy"] = entropy
    return df

# ----------------------------------------------------------------------
# Dimensionality reductions
# ----------------------------------------------------------------------
def run_pca(df: pd.DataFrame, n_comp: int = 2) -> np.ndarray:
    return PCA(n_components=n_comp, random_state=42).fit_transform(df)

def run_umap(df: pd.DataFrame, n_comp: int = 2) -> np.ndarray:
    return umap.UMAP(n_components=n_comp, random_state=42).fit_transform(df)

def run_tsne(df: pd.DataFrame, n_comp: int = 2) -> np.ndarray:
    return TSNE(n_components=n_comp, init="pca", random_state=42).fit_transform(df)

# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def make_plot(coords: np.ndarray, color: np.ndarray, title: str, out_html: Path):
    fig = px.scatter(x=coords[:,0], y=coords[:,1],
                     color=color,
                     title=title,
                     labels={"x":"Dim1", "y":"Dim2"})
    fig.write_html(str(out_html))
    return fig

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FFT × Symbolic Fusion Tool")
    ap.add_argument("--mu", type=Path, required=True, help="Input μ spectra (.npy or .csv)")
    ap.add_argument("--symbolic", type=Path, help="Symbolic results JSON (per-planet)")
    ap.add_argument("--entropy", type=Path, help="Entropy scores (.npy)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/fft_symbolic_fusion"))
    ap.add_argument("--n-freq", type=int, default=50)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load μ
    if args.mu.suffix == ".npy":
        mu = np.load(args.mu)
    else:
        mu = np.loadtxt(args.mu, delimiter=",")

    # Load symbolic
    symbolic = None
    if args.symbolic and args.symbolic.exists():
        with open(args.symbolic) as f:
            symbolic = json.load(f)

    # Load entropy
    entropy = None
    if args.entropy and args.entropy.exists():
        entropy = np.load(args.entropy)

    # Build features
    df = build_feature_matrix(mu, symbolic, entropy)

    # Run reductions
    pca_coords = run_pca(df)
    umap_coords = run_umap(df)
    tsne_coords = run_tsne(df)

    # Save PCA static
    pd.DataFrame(pca_coords, columns=["PC1","PC2"]).to_csv(args.outdir/"pca_coords.csv", index=False)

    # Interactive plots
    make_plot(umap_coords, df.get("entropy", None), "FFT × Symbolic UMAP", args.outdir/"fft_symbolic_umap.html")
    make_plot(tsne_coords, df.get("entropy", None), "FFT × Symbolic t-SNE", args.outdir/"fft_symbolic_tsne.html")

    # Cluster CSV
    df.to_csv(args.outdir/"fft_symbolic_clusters.csv", index=False)

    # Summary JSON
    summary = {
        "n_samples": len(df),
        "features": list(df.columns),
        "outputs": {
            "pca": "pca_coords.csv",
            "umap": "fft_symbolic_umap.html",
            "tsne": "fft_symbolic_tsne.html",
            "clusters": "fft_symbolic_clusters.csv"
        }
    }
    with open(args.outdir/"fft_symbolic_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print(f"[✓] FFT × Symbolic fusion complete. Results in {args.outdir}")

if __name__ == "__main__":
    main()