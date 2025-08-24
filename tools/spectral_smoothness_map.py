#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/spectral_smoothness_map.py

SpectraMind V50 — Spectral Smoothness Map & Diagnostics

Purpose
-------
Quantify and visualize spectral smoothness for predicted μ(λ) across samples and bins.
This tool produces:
  • Per-sample smoothness metrics: TV (|Δμ|), curvature (|Δ²μ|), rolling-std, FFT high‑freq ratio
  • Per-bin smoothness maps averaged over samples (hotspots of roughness)
  • Heatmaps (samples × bins) of |Δμ| and |Δ²μ|, sorted by smoothness score
  • Scatter overlays comparing smoothness vs entropy, GLL, SHAP, symbolic violations
  • Interactive Plotly HTML for quick exploration
  • CSV tables + a compact HTML report to embed into the unified diagnostics dashboard

Design
------
• CLI-first (Typer), reproducible, seed-stable for projections
• No heavy deps (NumPy/Pandas/Matplotlib/Plotly; UMAP optional for 2D embeddings)
• Aligns with V50 diagnostics ecosystem and audit logging (logs/v50_debug_log.md)

Inputs
------
--mu           N×B .npy of predicted μ (required)
--wavelengths  B .npy of λ values (optional; else uses bin index)
--sigma        N×B .npy predicted σ (optional; for GLL calc when --y provided)
--y            N×B .npy ground truth (optional; enables per-sample GLL)
--symbolic     JSON per-sample symbolic violation summary (optional)
--shap-bins    N×B .npy per-bin SHAP magnitudes (optional; for overlay correlation)
--outdir       Output directory (will be created)

Key Metrics (lower = smoother unless stated otherwise)
------------------------------------------------------
TV1: mean(|Δμ|)                     — first-difference total variation average
TV2: mean(|Δ²μ|)                    — curvature / second-difference variation average
RSTD[w]: mean(rolling std, window)  — average local volatility (window configurable)
HF_Ratio[k]: FFT high‑freq power / total (k = number of low freqs "kept")
Entropy: Shannon entropy of μ (overlay; not a smoothness metric)
GLL: mean Gaussian log-likelihood per sample (overlay; needs y and σ)
SHAP_mean: mean(|SHAP|) across bins (overlay; if SHAP provided)
Symbolic_violations: optional overlay (if JSON provided)

Outputs (in --outdir)
---------------------
tables/
  smoothness_per_sample.csv
  smoothness_per_bin.csv
  overlays_pairwise_corr.csv
plots/
  heatmap_tv1.png / heatmap_tv2.png
  heatmap_tv1.html / heatmap_tv2.html
  mean_bin_tv1.png / mean_bin_tv2.png (+ .html)
  scatter_smoothness_overlays_*.html
report_spectral_smoothness_map.html  (if --html)
summary.json

Examples
--------
# Minimal
python -m tools.spectral_smoothness_map --mu outputs/predictions/mu.npy --outdir outputs/smoothness

# With overlays & HTML
python -m tools.spectral_smoothness_map \
  --mu mu.npy --sigma sigma.npy --y labels.npy \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --shap-bins outputs/diagnostics/shap_bins.npy \
  --wavelengths data/wavelengths.npy \
  --window 21 --fft-keep 32 --html --save-png
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme

# Plotting
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Optional UMAP for embedding of sample metrics
try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

app = typer.Typer(add_completion=False, help="SpectraMind V50 — Spectral Smoothness Map & Diagnostics")
console = Console(theme=Theme({"info": "cyan", "warn": "yellow", "err": "bold red"}))


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} did not contain an ndarray")
    return arr


def append_audit_log(msg: str) -> None:
    try:
        logs = Path("logs")
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / "v50_debug_log.md", "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def now_str() -> str:
    import datetime as dt
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_global_seed(seed: int) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass


# ============================================================
# Core metrics
# ============================================================

def first_diff(mu: np.ndarray) -> np.ndarray:
    # mu: (N, B) -> returns (N, B-1)
    return np.diff(mu, axis=1)


def second_diff(mu: np.ndarray) -> np.ndarray:
    # mu: (N, B) -> returns (N, B-2)
    return mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]


def rolling_std(x: np.ndarray, win: int = 21) -> np.ndarray:
    """1D rolling std with reflect padding, odd window."""
    w = int(win) | 1
    pad = w // 2
    xr = np.pad(x, (pad, pad), mode="reflect")
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        seg = xr[i:i + w]
        out[i] = float(np.std(seg))
    return out


def entropy_row(mu_row: np.ndarray, eps: float = 1e-12) -> float:
    v = mu_row.astype(float)
    v = v - np.min(v)
    v = v + eps
    p = v / np.sum(v)
    return float(-np.sum(p * np.log(p + eps)))


def gll_row(mu_row: np.ndarray, sigma_row: np.ndarray, y_row: np.ndarray, eps: float = 1e-12) -> float:
    s2 = np.maximum(sigma_row.astype(float) ** 2, eps)
    resid2 = (y_row.astype(float) - mu_row.astype(float)) ** 2
    ll = -0.5 * (np.log(2 * np.pi * s2) + resid2 / s2)
    return float(np.mean(ll))


def fft_highfreq_ratio(mu_row: np.ndarray, keep: int = 32) -> float:
    fx = np.fft.rfft(mu_row)
    mag2 = (fx.real ** 2 + fx.imag ** 2)
    k = max(1, min(keep, len(mag2)))
    low = float(np.sum(mag2[:k]))
    high = float(np.sum(mag2[k:])) if k < len(mag2) else 0.0
    denom = low + high
    return float(high / denom) if denom > 0 else 0.0


@dataclass
class SmoothnessConfig:
    window: int = 21
    fft_keep: int = 32
    normalize_by_range: bool = False  # if True, scale μ row to [0,1] before metrics


def compute_smoothness_maps(
    mu: np.ndarray,
    cfg: SmoothnessConfig,
) -> Dict[str, Any]:
    """
    Compute per-sample and per-bin smoothness metrics.
    Returns:
      {
        "tv1_map": |Δμ| (N×B-1),
        "tv2_map": |Δ²μ| (N×B-2),
        "rstd_map": rolling-std (N×B),
        "per_sample": DataFrame with summary metrics per sample,
        "per_bin": DataFrame with mean tv1/tv2 per bin across samples,
      }
    """
    N, B = mu.shape
    M = mu.astype(float).copy()

    if cfg.normalize_by_range:
        # avoid division by zero; normalize each row by (max-min)
        row_min = np.nanmin(M, axis=1, keepdims=True)
        row_max = np.nanmax(M, axis=1, keepdims=True)
        denom = np.clip(row_max - row_min, 1e-12, None)
        M = (M - row_min) / denom

    d1 = first_diff(M)
    d2 = second_diff(M)
    tv1_map = np.abs(d1)                      # (N, B-1)
    tv2_map = np.abs(d2)                      # (N, B-2)

    # Rolling std per sample
    rstd_map = np.empty_like(M)
    for i in range(N):
        rstd_map[i] = rolling_std(M[i], win=cfg.window)

    # Summary per sample
    tv1_mean = np.mean(tv1_map, axis=1)
    tv2_mean = np.mean(tv2_map, axis=1) if tv2_map.shape[1] > 0 else np.zeros(N)
    rstd_mean = np.mean(rstd_map, axis=1)
    hf_ratio = np.array([fft_highfreq_ratio(M[i], keep=cfg.fft_keep) for i in range(N)], dtype=float)

    per_sample = pd.DataFrame({
        "sample": np.arange(N),
        "TV1_mean": tv1_mean,
        "TV2_mean": tv2_mean,
        "RSTD_mean": rstd_mean,
        "HF_Ratio": hf_ratio,
    })

    # Per-bin mean TV1/TV2 across samples (align to original bin grid)
    tv1_bin = np.mean(tv1_map, axis=0) if N > 0 else np.array([])
    tv2_bin = np.mean(tv2_map, axis=0) if N > 0 and tv2_map.shape[1] > 0 else np.array([])
    # Align TV1 (B-1) and TV2 (B-2) to a common B-length index via padding for plotting
    per_bin = pd.DataFrame({
        "bin": np.arange(B),
        "TV1_mean_bin": np.concatenate([[tv1_bin[0]], 0.5 * (tv1_bin[:-1] + tv1_bin[1:])]) if B > 1 else np.array([0.0]),
        "TV2_mean_bin": np.concatenate([[tv2_bin[0] if tv2_bin.size else 0.0],
                                        np.pad(tv2_bin, (0, max(0, B - 1 - tv2_bin.size)), constant_values=(tv2_bin[-1] if tv2_bin.size else 0.0))]),
    })

    return {
        "tv1_map": tv1_map,
        "tv2_map": tv2_map,
        "rstd_map": rstd_map,
        "per_sample": per_sample,
        "per_bin": per_bin,
    }


# ============================================================
# Visualization
# ============================================================

def plot_heatmap_png(
    mat: np.ndarray,
    title: str,
    out_png: Path,
    xlabel: str = "Bin",
    ylabel: str = "Sample (sorted)",
    cmap: str = "magma",
) -> None:
    plt.figure(figsize=(10, 6), dpi=120)
    plt.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_heatmap_html(
    mat: np.ndarray,
    title: str,
    out_html: Path,
) -> None:
    # Build a z dataframe to avoid huge HTML; downsample if enormous
    Z = mat
    max_w = 1800
    if Z.shape[1] > max_w:
        step = math.ceil(Z.shape[1] / max_w)
        Z = Z[:, ::step]
    fig = go.Figure(data=go.Heatmap(z=Z, colorscale="Magma"))
    fig.update_layout(title=title, xaxis_title="Bin (downsampled)" if Z is not mat else "Bin", yaxis_title="Sample (sorted)", height=520, template="plotly_white")
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)


def plot_mean_bin_curves(
    per_bin: pd.DataFrame,
    wavelengths: Optional[np.ndarray],
    out_png_tv1: Path,
    out_png_tv2: Path,
    out_html_tv1: Path,
    out_html_tv2: Path,
) -> None:
    x = wavelengths if wavelengths is not None else per_bin["bin"].values

    # TV1
    plt.figure(figsize=(10, 3.3), dpi=120)
    plt.plot(x, per_bin["TV1_mean_bin"].values, lw=2.0)
    plt.title("Mean |Δμ| per bin (TV1)")
    plt.xlabel("Wavelength" if wavelengths is not None else "Bin")
    plt.ylabel("TV1")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png_tv1)
    plt.close()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x, y=per_bin["TV1_mean_bin"].values, mode="lines", name="TV1"))
    fig1.update_layout(title="Mean |Δμ| per bin (TV1)", xaxis_title="Wavelength" if wavelengths is not None else "Bin", yaxis_title="TV1", template="plotly_white", height=320)
    pio.write_html(fig1, file=str(out_html_tv1), include_plotlyjs="cdn", full_html=True)

    # TV2
    plt.figure(figsize=(10, 3.3), dpi=120)
    plt.plot(x, per_bin["TV2_mean_bin"].values, lw=2.0)
    plt.title("Mean |Δ²μ| per bin (TV2)")
    plt.xlabel("Wavelength" if wavelengths is not None else "Bin")
    plt.ylabel("TV2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png_tv2)
    plt.close()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=per_bin["TV2_mean_bin"].values, mode="lines", name="TV2"))
    fig2.update_layout(title="Mean |Δ²μ| per bin (TV2)", xaxis_title="Wavelength" if wavelengths is not None else "Bin", yaxis_title="TV2", template="plotly_white", height=320)
    pio.write_html(fig2, file=str(out_html_tv2), include_plotlyjs="cdn", full_html=True)


def scatter_pair_html(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hover: Optional[pd.DataFrame],
    title: str,
    out_html: Path,
) -> None:
    plot_df = df[[x_col, y_col]].copy()
    if hover is not None:
        for c in hover.columns:
            plot_df[c] = hover[c]
    fig = px.scatter(plot_df, x=x_col, y=y_col, hover_data=hover.columns.tolist() if hover is not None else None, title=title, template="plotly_white")
    fig.update_traces(marker=dict(opacity=0.85))
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)


# ============================================================
# HTML report
# ============================================================

def build_html_report(
    outdir: Path,
    summary: Dict[str, Any],
    tables: Dict[str, str],
    plots_png: List[str],
    plots_html: List[str],
) -> str:
    report = outdir / "report_spectral_smoothness_map.html"
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; margin: 16px; color: #0e1116; }
    h1 { font-size: 20px; margin: 8px 0 12px; }
    h2 { font-size: 16px; margin: 16px 0 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    a { color: #0b5fff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { background: #f3f4f6; padding: 2px 4px; border-radius: 6px; }
    """
    imgs = "".join([f'<div class="card"><img src="plots/{fn}" style="width:100%;height:auto" alt="{fn}"/></div>' for fn in plots_png])
    links = "".join([f'<div class="card"><a href="plots/{fn}">{fn}</a></div>' for fn in plots_html])
    tbls = "".join([f'<div class="card"><a href="tables/{v}">{k}: {v}</a></div>' for k, v in tables.items()])

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — Spectral Smoothness Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{css}</style>
</head>
<body>
  <h1>SpectraMind V50 — Spectral Smoothness Map</h1>
  <div class="card"><pre>{json.dumps(summary, indent=2)}</pre></div>

  <h2>Static Plots</h2>
  <div class="grid">{imgs}</div>

  <h2>Interactive Plots</h2>
  <div class="grid">{links}</div>

  <h2>Tables</h2>
  <div class="grid">{tbls}</div>
</body>
</html>
"""
    report.write_text(html, encoding="utf-8")
    return str(report.name)


# ============================================================
# Orchestration
# ============================================================

def run_smoothness_map(
    mu_path: str,
    outdir: str,
    wavelengths_path: Optional[str] = None,
    sigma_path: Optional[str] = None,
    y_path: Optional[str] = None,
    symbolic_path: Optional[str] = None,
    shap_bins_path: Optional[str] = None,
    window: int = 21,
    fft_keep: int = 32,
    normalize_by_range: bool = False,
    save_png: bool = False,
    html_report: bool = False,
    seed: int = 42,
) -> None:
    t0 = time.time()
    set_global_seed(seed)

    out = Path(outdir)
    plots = out / "plots"
    tables = out / "tables"
    ensure_dir(out)
    ensure_dir(plots)
    ensure_dir(tables)

    console.rule("[info]SpectraMind V50 — Spectral Smoothness Map")
    console.print(f"[info]μ: {mu_path}")
    if wavelengths_path: console.print(f"[info]λ: {wavelengths_path}")
    if sigma_path: console.print(f"[info]σ: {sigma_path}")
    if y_path: console.print(f"[info]y: {y_path}")
    if symbolic_path: console.print(f"[info]symbolic: {symbolic_path}")
    if shap_bins_path: console.print(f"[info]shap-bins: {shap_bins_path}")

    # Load
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
        t_load = progress.add_task("Loading arrays", total=None)
        mu = load_npy(mu_path)
        if mu is None:
            raise FileNotFoundError("--mu is required")
        wavelengths = load_npy(wavelengths_path) if wavelengths_path else None
        sigma = load_npy(sigma_path) if sigma_path else None
        y = load_npy(y_path) if y_path else None
        shap_bins = load_npy(shap_bins_path) if shap_bins_path else None

        symbolic_json = None
        if symbolic_path and Path(symbolic_path).exists():
            symbolic_json = json.loads(Path(symbolic_path).read_text())
        progress.update(t_load, advance=1, visible=False)

    N, B = mu.shape
    console.print(f"[info]Loaded μ shape: {N}×{B}")

    # Compute smoothness maps & summaries
    cfg = SmoothnessConfig(window=window, fft_keep=fft_keep, normalize_by_range=normalize_by_range)
    maps = compute_smoothness_maps(mu, cfg)

    per_sample = maps["per_sample"]  # contains TV1_mean, TV2_mean, RSTD_mean, HF_Ratio

    # Overlays: entropy, GLL, SHAP_mean, symbolic_violations
    ent = np.array([entropy_row(mu[i]) for i in range(N)], dtype=float)
    per_sample["Entropy"] = ent

    if sigma is not None and y is not None:
        gll_vec = np.array([gll_row(mu[i], sigma[i], y[i]) for i in range(N)], dtype=float)
        per_sample["GLL"] = gll_vec
    else:
        gll_vec = None

    if shap_bins is not None:
        shap_mean = np.mean(np.abs(shap_bins), axis=1)
        per_sample["SHAP_mean"] = shap_mean

    sym_vec = None
    if symbolic_json is not None:
        try:
            if isinstance(symbolic_json, list) and len(symbolic_json) == N:
                sym_vec = np.array([float(d.get("violations_total", 0.0)) for d in symbolic_json], dtype=float)
            elif isinstance(symbolic_json, dict) and all(k.isdigit() for k in symbolic_json.keys()):
                sym_vec = np.array([float(symbolic_json.get(str(i), {}).get("violations_total", 0.0)) for i in range(N)], dtype=float)
        except Exception:
            sym_vec = None
    if sym_vec is not None:
        per_sample["Symbolic_violations"] = sym_vec

    # Define a composite "SmoothnessScore" (lower is smoother)
    # Weighted sum (tunable): TV1_mean + 0.5*TV2_mean + 0.25*RSTD_mean + HF_Ratio
    per_sample["SmoothnessScore"] = (
        per_sample["TV1_mean"].values
        + 0.5 * per_sample["TV2_mean"].values
        + 0.25 * per_sample["RSTD_mean"].values
        + 1.0 * per_sample["HF_Ratio"].values
    )

    # Save tables
    per_sample_csv = tables / "smoothness_per_sample.csv"
    per_sample.to_csv(per_sample_csv, index=False)

    per_bin_csv = tables / "smoothness_per_bin.csv"
    maps["per_bin"].to_csv(per_bin_csv, index=False)

    # Pairwise correlations (quick sanity)
    corr_cols = [c for c in ["TV1_mean", "TV2_mean", "RSTD_mean", "HF_Ratio", "Entropy", "GLL", "SHAP_mean", "Symbolic_violations", "SmoothnessScore"] if c in per_sample.columns]
    corr = per_sample[corr_cols].corr(numeric_only=True)
    corr_csv = tables / "overlays_pairwise_corr.csv"
    corr.to_csv(corr_csv)

    # Heatmaps: sort rows by SmoothnessScore descending (roughest first)
    order = np.argsort(-per_sample["SmoothnessScore"].values)
    tv1_sorted = maps["tv1_map"][order, :]
    tv2_sorted = maps["tv2_map"][order, :]

    # Plots — PNG
    pngs: List[str] = []
    if save_png:
        plot_heatmap_png(tv1_sorted, "Heatmap |Δμ| (TV1), samples sorted by roughness", plots / "heatmap_tv1.png")
        pngs.append("heatmap_tv1.png")
        if tv2_sorted.shape[1] > 0:
            plot_heatmap_png(tv2_sorted, "Heatmap |Δ²μ| (TV2), samples sorted by roughness", plots / "heatmap_tv2.png")
            pngs.append("heatmap_tv2.png")

    # Plots — HTML heatmaps
    plot_heatmap_html(tv1_sorted, "Heatmap |Δμ| (TV1), samples sorted by roughness", plots / "heatmap_tv1.html")
    plot_heatmap_html(tv2_sorted, "Heatmap |Δ²μ| (TV2), samples sorted by roughness", plots / "heatmap_tv2.html")
    htmls: List[str] = ["heatmap_tv1.html", "heatmap_tv2.html"]

    # Mean per-bin curves (PNG + HTML)
    plot_mean_bin_curves(
        per_bin=maps["per_bin"],
        wavelengths=wavelengths,
        out_png_tv1=plots / "mean_bin_tv1.png",
        out_png_tv2=plots / "mean_bin_tv2.png",
        out_html_tv1=plots / "mean_bin_tv1.html",
        out_html_tv2=plots / "mean_bin_tv2.html",
    )
    if save_png:
        pngs += ["mean_bin_tv1.png", "mean_bin_tv2.png"]
    htmls += ["mean_bin_tv1.html", "mean_bin_tv2.html"]

    # Scatter overlays: SmoothnessScore vs overlays
    hover_df = None  # attach custom hover later if desired
    def maybe_scatter(col: str, label: str):
        if col in per_sample.columns:
            out = plots / f"scatter_smoothness_vs_{col.lower()}.html"
            scatter_pair_html(per_sample, "SmoothnessScore", col, hover_df, f"SmoothnessScore vs {label}", out)
            htmls.append(out.name)

    for col, lbl in [("Entropy", "Entropy"),
                     ("GLL", "GLL"),
                     ("SHAP_mean", "mean(|SHAP|)"),
                     ("Symbolic_violations", "Symbolic Violations")]:
        maybe_scatter(col, lbl)

    # Summary JSON
    summary = {
        "timestamp": now_str(),
        "mu_path": mu_path,
        "wavelengths_path": wavelengths_path,
        "sigma_path": sigma_path,
        "y_path": y_path,
        "symbolic_path": symbolic_path,
        "shap_bins_path": shap_bins_path,
        "N": int(N),
        "B": int(B),
        "window": int(window),
        "fft_keep": int(fft_keep),
        "normalize_by_range": bool(normalize_by_range),
        "tables": {
            "smoothness_per_sample.csv": per_sample_csv.name,
            "smoothness_per_bin.csv": per_bin_csv.name,
            "overlays_pairwise_corr.csv": corr_csv.name,
        },
        "plots": {
            "png": pngs,
            "html": htmls,
        },
        "timing_sec": round(time.time() - t0, 3),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # HTML report
    if html_report:
        report_name = build_html_report(
            outdir=out,
            summary=summary,
            tables={
                "per_sample": per_sample_csv.name,
                "per_bin": per_bin_csv.name,
                "pairwise_corr": corr_csv.name,
            },
            plots_png=pngs,
            plots_html=htmls,
        )
        console.print(f"[info]Wrote HTML report → {out / report_name}")

    # Audit log (best-effort)
    append_audit_log(f"- {now_str()} | spectral_smoothness_map | mu={mu_path} out={outdir} N={N} B={B} window={window} fft_keep={fft_keep} normalize={normalize_by_range}")

    console.rule("[info]Done")
    console.print(f"[info]Elapsed: {round(time.time() - t0, 2)} s")
    console.print(f"[info]Artifacts in: {outdir}")


# ============================================================
# Typer CLI
# ============================================================

@app.command("run")
def cli_run(
    mu: str = typer.Option(..., help="Path to μ.npy (N×B)"),
    outdir: str = typer.Option(..., help="Output directory for artifacts"),
    wavelengths: Optional[str] = typer.Option(None, help="Path to wavelengths.npy (B,)"),
    sigma: Optional[str] = typer.Option(None, help="Path to σ.npy (N×B) for GLL"),
    y: Optional[str] = typer.Option(None, help="Path to labels.npy (N×B) for GLL"),
    symbolic: Optional[str] = typer.Option(None, help="Path to symbolic overlay JSON"),
    shap_bins: Optional[str] = typer.Option(None, help="Path to per-bin SHAP magnitudes .npy (N×B)"),
    window: int = typer.Option(21, help="Rolling std window (odd)"),
    fft_keep: int = typer.Option(32, help="Low-frequency count to keep for HF ratio"),
    normalize_by_range: bool = typer.Option(False, help="Normalize each μ row to [0,1] before metrics"),
    save_png: bool = typer.Option(False, help="Also save static PNGs"),
    html: bool = typer.Option(False, help="Emit compact HTML report"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """
    Compute smoothness maps and export heatmaps, curves, and overlay comparisons.
    """
    try:
        run_smoothness_map(
            mu_path=mu,
            outdir=outdir,
            wavelengths_path=wavelengths,
            sigma_path=sigma,
            y_path=y,
            symbolic_path=symbolic,
            shap_bins_path=shap_bins,
            window=window,
            fft_keep=fft_keep,
            normalize_by_range=normalize_by_range,
            save_png=save_png,
            html_report=html,
            seed=seed,
        )
    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", style="err"))
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
