#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/spectral_absorption_overlay_clustered.py

SpectraMind V50 — Clustered Spectral Absorption Overlay

Purpose
-------
Compute absorption features from predicted μ spectra, cluster the samples using band-depth
features (plus optional FFT/entropy descriptors), and render clustered overlays:

  • Per‑cluster mean/median spectrum with shaded IQR and band overlays
  • Heatmaps of spectra sorted by cluster and by band-depth score
  • Interactive Plotly figures (per‑cluster overlays, UMAP if available)
  • CSV tables: cluster assignments, band depths, cluster stats
  • Optional HTML report linking plots/tables for diagnostics dashboard inclusion

This tool is designed to integrate with the V50 diagnostics stack and the unified HTML
report (`generate_html_report.py`). All file outputs are safe to link/iframe.

Inputs
------
--mu            N×B .npy of predicted μ (required)
--wavelengths   B .npy of λ values (optional; if absent, bin indices are used)
--bands         YAML/JSON file specifying named bands (μm or indices) (optional)
--meta          CSV with N rows for hover/labels (optional)
--symbolic      JSON with per-sample symbolic violation summaries (optional)
--shap-bins     Optional N×B .npy of per-bin SHAP magnitudes for overlay scoring (optional)

Feature Engineering (defaults)
------------------------------
• Band depths per named band: mean(cont) - mean(band)
  - A "continuum" band can also be provided; else an automatic continuum is estimated
    via a robust rolling median or a low-order poly fit.
• Global descriptors (optional toggles):
  - Entropy of μ
  - FFT high-frequency energy ratio
  - Mean SHAP magnitude (if provided)
  - Symbolic violation count/score (if provided)

Clustering
----------
• Methods: KMeans (default), MiniBatchKMeans, DBSCAN, Agglomerative
• Features: concatenation of band-depth vector + optional globals (scaled)
• UMAP/TSNE embedding for visualization (optional and graceful if packages missing)

Outputs (in --outdir)
---------------------
tables/
  band_depths.csv
  cluster_assignments.csv
  cluster_stats.csv
plots/
  clusters_overlay_*.png / .html
  spectra_heatmap_by_cluster_*.png
  umap_clusters.html / tsne_clusters.html (if enabled)
report_spectral_absorption_overlay.html (if --html)

CLI Examples
------------
# Minimal:
python -m tools.spectral_absorption_overlay_clustered \
  --mu outputs/predictions/mu.npy \
  --outdir outputs/absorption_clusters

# With wavelengths, bands, and HTML report:
python -m tools.spectral_absorption_overlay_clustered \
  --mu outputs/predictions/mu.npy \
  --wavelengths data/wavelengths.npy \
  --bands configs/bands.yaml \
  --outdir outputs/absorption_clusters --html

# Include symbolic and SHAP overlays, use 6 clusters, add UMAP:
python -m tools.spectral_absorption_overlay_clustered \
  --mu mu.npy --wavelengths lam.npy --bands bands.yaml \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --shap-bins outputs/diagnostics/shap_bins.npy \
  --n-clusters 6 --umap --html --save-png
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme

# Plotting (PNG, static)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plotly (HTML, interactive)
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Clustering & projection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE

# Optional deps: umap-learn, pyyaml
try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# Typer CLI
app = typer.Typer(add_completion=False, help="SpectraMind V50 — Clustered Spectral Absorption Overlay")
console = Console(theme=Theme({"info": "cyan", "warn": "yellow", "err": "bold red"}))


# ============================================================
# Utility
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
        logdir = Path("logs")
        logdir.mkdir(parents=True, exist_ok=True)
        with open(logdir / "v50_debug_log.md", "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


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


def now_str() -> str:
    import datetime as dt
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Bands config
# ============================================================

@dataclass
class BandsConfig:
    """
    Holds named bands and an optional 'continuum' name.
    Bands can be specified in μm (with wavelengths provided) or in indices.
    """
    bands: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    continuum: Optional[Tuple[float, float]] = None  # optional explicit continuum window
    unit: str = "auto"  # "auto", "um", or "index"

    @staticmethod
    def from_file(path: Optional[str]) -> "BandsConfig":
        if path is None:
            return BandsConfig()
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML not installed; cannot parse YAML bands file.")
            data = yaml.safe_load(p.read_text())
        elif p.suffix.lower() == ".json":
            data = json.loads(p.read_text())
        else:
            raise ValueError(f"Unsupported band file type: {p.suffix}")
        bc = BandsConfig()
        bc.bands = {str(k): (float(v[0]), float(v[1])) for k, v in data.get("bands", {}).items()}
        cont = data.get("continuum", None)
        if cont is not None:
            bc.continuum = (float(cont[0]), float(cont[1]))
        bc.unit = str(data.get("unit", "auto"))
        return bc


def band_mask_from_bounds(
    B: int,
    bounds: Tuple[float, float],
    wavelengths: Optional[np.ndarray],
    unit: str = "auto",
) -> np.ndarray:
    """
    Return a binary mask for a band defined by (lo, hi), either in μm or indices.
    If unit="auto": use wavelengths if provided; else treat as indices.
    """
    lo, hi = bounds
    mask = np.zeros(B, dtype=np.float32)
    if (unit == "um") or (unit == "auto" and wavelengths is not None):
        lam = wavelengths if wavelengths is not None else np.arange(B)
        mask = ((lam >= lo) & (lam <= hi)).astype(np.float32)
    else:
        ia = max(0, min(B - 1, int(round(lo))))
        ib = max(0, min(B - 1, int(round(hi))))
        if ib < ia:
            ia, ib = ib, ia
        mask[ia:ib + 1] = 1.0
    return mask


# ============================================================
# Feature engineering
# ============================================================

def rolling_median(x: np.ndarray, win: int = 21) -> np.ndarray:
    """Simple rolling median with reflect pad; odd window size recommended."""
    w = max(3, win | 1)
    pad = w // 2
    xr = np.pad(x, (pad, pad), mode="reflect")
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = np.median(xr[i:i + w])
    return out


def polyfit_baseline(x: np.ndarray, y: np.ndarray, deg: int = 3) -> np.ndarray:
    """Low-order polynomial baseline fit."""
    try:
        c = np.polyfit(x, y, deg=deg)
        p = np.poly1d(c)
        return p(x)
    except Exception:
        # fallback to rolling median
        return rolling_median(y, win=min(51, max(7, len(y)//12*2+1)))


def estimate_continuum(mu_row: np.ndarray, wavelengths: Optional[np.ndarray]) -> np.ndarray:
    """
    Estimate a smooth baseline ("continuum") for a single μ spectrum.
    If wavelengths provided, polyfit over λ; else polyfit over bin index.
    """
    B = mu_row.shape[0]
    x = wavelengths if wavelengths is not None else np.arange(B)
    base = polyfit_baseline(x.astype(float), mu_row.astype(float), deg=3)
    # safety: ensure no zeros/negatives (not strictly required for "depth")
    return base


def band_mean(mu_row: np.ndarray, mask01: np.ndarray) -> float:
    w = mask01.astype(float)
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(mu_row * w) / np.sum(w))


def entropy_row(mu_row: np.ndarray, eps: float = 1e-12) -> float:
    v = mu_row.astype(float)
    v = v - np.min(v)
    v = v + eps
    p = v / np.sum(v)
    return float(-np.sum(p * np.log(p + eps)))


def fft_highfreq_ratio(mu_row: np.ndarray, keep: int = 32) -> float:
    fx = np.fft.rfft(mu_row)
    mag2 = (fx.real ** 2 + fx.imag ** 2)
    low = mag2[:max(1, keep)].sum()
    high = mag2[max(1, keep):].sum()
    denom = (low + high) if (low + high) > 0 else 1.0
    return float(high / denom)


@dataclass
class FeatureConfig:
    use_entropy: bool = True
    use_fft_ratio: bool = True
    use_shap_mean: bool = True
    use_symbolic_score: bool = True
    fft_keep: int = 32


def compute_band_depth_features(
    mu: np.ndarray,
    bands_cfg: BandsConfig,
    wavelengths: Optional[np.ndarray],
    shap_bins: Optional[np.ndarray],
    symbolic_json: Optional[Dict[str, Any]],
    feat_cfg: FeatureConfig,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute band-depth features for all samples. If a 'continuum' band is provided in bands_cfg,
    depth = mean(cont) - mean(band). Else estimate a per-sample continuum curve and define
    band mean relative to baseline: depth ~ mean(baseline) - mean(band region).

    Returns:
      features_df (N × (num_bands + extras))
      masks_by_name: dict name -> mask01 (B,)
    """
    N, B = mu.shape
    masks_by_name: Dict[str, np.ndarray] = {}
    # Build masks
    for name, bounds in bands_cfg.bands.items():
        masks_by_name[name] = band_mask_from_bounds(B, bounds, wavelengths, unit=bands_cfg.unit)
    cont_mask = None
    if bands_cfg.continuum is not None:
        cont_mask = band_mask_from_bounds(B, bands_cfg.continuum, wavelengths, unit=bands_cfg.unit)

    # Compute features
    cols = []
    data = []

    for name in bands_cfg.bands.keys():
        cols.append(f"depth::{name}")

    if feat_cfg.use_entropy:
        cols.append("entropy")
    if feat_cfg.use_fft_ratio:
        cols.append("fft_highfreq_ratio")
    if feat_cfg.use_shap_mean and shap_bins is not None:
        cols.append("shap_mean")
    if feat_cfg.use_symbolic_score and symbolic_json is not None:
        cols.append("symbolic_score")

    # Precompute per-sample extras
    # Parse symbolic score vector best-effort: allow list[dict] or dict[str]->dict
    sym_vec = None
    if symbolic_json is not None:
        try:
            if isinstance(symbolic_json, list) and len(symbolic_json) == N:
                sym_vec = np.array([float(d.get("violations_total", 0.0)) for d in symbolic_json], dtype=float)
            elif isinstance(symbolic_json, dict) and all(k.isdigit() for k in symbolic_json.keys()):
                sym_vec = np.array([float(symbolic_json.get(str(i), {}).get("violations_total", 0.0)) for i in range(N)])
        except Exception:
            sym_vec = None

    for i in range(N):
        row = []
        mu_i = mu[i, :]

        if cont_mask is not None:
            cont_mean = band_mean(mu_i, cont_mask)
            for name, mask in masks_by_name.items():
                row.append(cont_mean - band_mean(mu_i, mask))
        else:
            # Estimate continuum curve; compute baseline mean in each band region
            baseline = estimate_continuum(mu_i, wavelengths)
            for name, mask in masks_by_name.items():
                bmean = band_mean(mu_i, mask)
                cmean = band_mean(baseline, mask)
                row.append(cmean - bmean)

        if feat_cfg.use_entropy:
            row.append(entropy_row(mu_i))
        if feat_cfg.use_fft_ratio:
            row.append(fft_highfreq_ratio(mu_i, keep=feat_cfg.fft_keep))
        if feat_cfg.use_shap_mean and shap_bins is not None:
            row.append(float(np.nanmean(np.abs(shap_bins[i, :]))))
        if feat_cfg.use_symbolic_score and sym_vec is not None:
            row.append(float(sym_vec[i]))

        data.append(row)

    features_df = pd.DataFrame(data, columns=cols)
    return features_df, masks_by_name


# ============================================================
# Clustering & projections
# ============================================================

def cluster_features(
    X: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 5,
    seed: int = 42,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 10,
    agglom_linkage: str = "ward",
) -> np.ndarray:
    method = method.lower()
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        labels = model.fit_predict(X)
    elif method == "minibatchkmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=256)
        labels = model.fit_predict(X)
    elif method == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = model.fit_predict(X)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=agglom_linkage)
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return labels


def compute_projections(
    X: np.ndarray,
    do_umap: bool,
    do_tsne: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    pca = PCA(n_components=2, random_state=seed)
    out["pca2"] = pca.fit_transform(X)
    if do_umap and _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
        out["umap2"] = reducer.fit_transform(X)
    if do_tsne:
        tsne = TSNE(n_components=2, random_state=seed, init="random", learning_rate="auto", perplexity=30)
        out["tsne2"] = tsne.fit_transform(X)
    return out


# ============================================================
# Visualization
# ============================================================

def plot_cluster_overlays_png(
    mu: np.ndarray,
    labels: np.ndarray,
    wavelengths: Optional[np.ndarray],
    masks_by_name: Dict[str, np.ndarray],
    outdir: Path,
) -> List[str]:
    """
    Static PNG: per-cluster mean with shaded IQR; band regions overlaid.
    """
    B = mu.shape[1]
    x = wavelengths if wavelengths is not None else np.arange(B)
    uniq = np.unique(labels)
    outfiles: List[str] = []

    # Precompute band polygons
    band_regions = []
    for name, m in masks_by_name.items():
        idx = np.where(m > 0.5)[0]
        if idx.size == 0:
            continue
        lo, hi = idx.min(), idx.max()
        xl = x[lo]
        xh = x[hi]
        band_regions.append((name, xl, xh))

    for c in uniq:
        sel = (labels == c)
        if sel.sum() == 0:  # safety
            continue
        block = mu[sel, :]
        mean = np.nanmean(block, axis=0)
        q25 = np.nanpercentile(block, 25, axis=0)
        q75 = np.nanpercentile(block, 75, axis=0)

        plt.figure(figsize=(10, 4), dpi=120)
        # shade bands
        for (name, xl, xh) in band_regions:
            plt.axvspan(xl, xh, color="#dbeafe", alpha=0.35, lw=0, label=None)
        # IQR shaded
        plt.fill_between(x, q25, q75, alpha=0.25, color="#9ca3af", label="IQR")
        # Mean line
        plt.plot(x, mean, lw=2.2, color="#0b5fff", label=f"Cluster {int(c)} mean")
        plt.title(f"Cluster {int(c)} — mean ± IQR (N={int(sel.sum())})")
        plt.xlabel("Wavelength" if wavelengths is not None else "Bin")
        plt.ylabel("μ")
        plt.grid(alpha=0.25)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        fn = outdir / f"clusters_overlay_c{int(c)}.png"
        plt.savefig(fn)
        plt.close()
        outfiles.append(fn.name)

    return outfiles


def plot_cluster_overlays_html(
    mu: np.ndarray,
    labels: np.ndarray,
    wavelengths: Optional[np.ndarray],
    masks_by_name: Dict[str, np.ndarray],
    outdir: Path,
) -> List[str]:
    """
    Interactive Plotly lines: per-cluster mean ± shaded band regions via shapes.
    """
    B = mu.shape[1]
    x = wavelengths if wavelengths is not None else np.arange(B)
    uniq = np.unique(labels)
    outfiles: List[str] = []

    # Band shapes (rectangles)
    shapes = []
    for name, m in masks_by_name.items():
        idx = np.where(m > 0.5)[0]
        if idx.size == 0:
            continue
        lo, hi = idx.min(), idx.max()
        xl = x[lo]
        xh = x[hi]
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=float(xl),
            x1=float(xh),
            y0=0,
            y1=1,
            fillcolor="rgba(13,110,253,0.10)",
            line=dict(width=0),
            layer="below"
        ))

    for c in uniq:
        sel = (labels == c)
        if sel.sum() == 0:
            continue
        block = mu[sel, :]
        mean = np.nanmean(block, axis=0)
        q25 = np.nanpercentile(block, 25, axis=0)
        q75 = np.nanpercentile(block, 75, axis=0)

        fig = go.Figure()
        # IQR band
        fig.add_traces([
            go.Scatter(x=x, y=q75, mode="lines", line=dict(width=0), showlegend=False),
            go.Scatter(x=x, y=q25, mode="lines",
                       fill="tonexty", fillcolor="rgba(156,163,175,0.35)",
                       line=dict(width=0), showlegend=True, name="IQR"),
        ])
        # Mean
        fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=f"Cluster {int(c)} mean"))

        fig.update_layout(
            title=f"Cluster {int(c)} — mean ± IQR (N={int(sel.sum())})",
            xaxis_title="Wavelength" if wavelengths is not None else "Bin",
            yaxis_title="μ",
            template="plotly_white",
            height=420,
            shapes=shapes,
        )
        fn = outdir / f"clusters_overlay_c{int(c)}.html"
        pio.write_html(fig, file=str(fn), include_plotlyjs="cdn", full_html=True)
        outfiles.append(fn.name)

    return outfiles


def plot_heatmap_sorted_png(
    mu: np.ndarray,
    labels: np.ndarray,
    wavelengths: Optional[np.ndarray],
    outdir: Path,
    by: str = "cluster",
) -> str:
    """
    Static heatmap of μ sorted by cluster or by mean band-depth.
    """
    N, B = mu.shape
    if by == "cluster":
        order = np.argsort(labels)
        title = "Spectra heatmap (sorted by cluster)"
        fname = "spectra_heatmap_by_cluster.png"
    else:
        # fallback: sort by row mean
        order = np.argsort(mu.mean(axis=1))
        title = "Spectra heatmap (sorted by μ mean)"
        fname = "spectra_heatmap_by_mean.png"

    mu_sorted = mu[order, :]
    plt.figure(figsize=(10, 6), dpi=120)
    plt.imshow(mu_sorted, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(label="μ")
    plt.title(title)
    plt.xlabel("Wavelength" if wavelengths is not None else "Bin")
    plt.ylabel("Samples (sorted)")
    plt.tight_layout()
    fn = outdir / fname
    plt.savefig(fn)
    plt.close()
    return fn.name


def plot_embedding_html(
    emb: np.ndarray,
    labels: np.ndarray,
    hover: Optional[pd.DataFrame],
    title: str,
    out_html: Path,
) -> None:
    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "cluster": labels})
    if hover is not None:
        for c in hover.columns:
            df[c] = hover[c]
    fig = px.scatter(
        df, x="x", y="y", color="cluster",
        hover_data=hover.columns.tolist() if hover is not None else None,
        title=title, template="plotly_white"
    )
    fig.update_traces(marker=dict(opacity=0.9))
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)


# ============================================================
# HTML report
# ============================================================

def build_html_report(
    outdir: Path,
    summary: Dict[str, Any],
    tables: Dict[str, str],
    pngs: List[str],
    htmls: List[str],
) -> str:
    report = outdir / "report_spectral_absorption_overlay.html"
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
    imgs = "".join([f'<div class="card"><img src="plots/{fn}" style="width:100%;height:auto" alt="{fn}"/></div>' for fn in pngs])
    links = "".join([f'<div class="card"><a href="plots/{fn}">{fn}</a></div>' for fn in htmls])
    tbls = "".join([f'<div class="card"><a href="tables/{v}">{k}: {v}</a></div>' for k, v in tables.items()])

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — Clustered Spectral Absorption Overlay</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{css}</style>
</head>
<body>
  <h1>SpectraMind V50 — Clustered Spectral Absorption Overlay</h1>
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

def run_pipeline(
    mu_path: str,
    outdir: str,
    wavelengths_path: Optional[str] = None,
    bands_path: Optional[str] = None,
    meta_csv: Optional[str] = None,
    symbolic_path: Optional[str] = None,
    shap_bins_path: Optional[str] = None,
    n_clusters: int = 5,
    clustering: str = "kmeans",
    seed: int = 42,
    scale_features: bool = True,
    use_entropy: bool = True,
    use_fft_ratio: bool = True,
    fft_keep: int = 32,
    use_shap_mean: bool = True,
    use_symbolic_score: bool = True,
    umap_flag: bool = False,
    tsne_flag: bool = False,
    save_png: bool = False,
    html_report: bool = False,
) -> None:
    t0 = time.time()
    set_global_seed(seed)
    out = Path(outdir)
    plots = out / "plots"
    tables = out / "tables"
    ensure_dir(out)
    ensure_dir(plots)
    ensure_dir(tables)

    console.rule("[info]SpectraMind V50 — Clustered Spectral Absorption Overlay")
    console.print(f"[info]μ: {mu_path}")
    if wavelengths_path: console.print(f"[info]λ: {wavelengths_path}")
    if bands_path: console.print(f"[info]bands: {bands_path}")
    if meta_csv: console.print(f"[info]meta: {meta_csv}")
    if symbolic_path: console.print(f"[info]symbolic: {symbolic_path}")
    if shap_bins_path: console.print(f"[info]shap-bins: {shap_bins_path}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
        t_load = progress.add_task("Loading data", total=None)
        mu = load_npy(mu_path)
        if mu is None:
            raise FileNotFoundError("--mu is required")
        wavelengths = load_npy(wavelengths_path) if wavelengths_path else None
        shap_bins = load_npy(shap_bins_path) if shap_bins_path else None
        bands_cfg = BandsConfig.from_file(bands_path)
        meta_df = pd.read_csv(meta_csv) if meta_csv else None
        symbolic_json = None
        if symbolic_path and Path(symbolic_path).exists():
            symbolic_json = json.loads(Path(symbolic_path).read_text())
        progress.update(t_load, advance=1, visible=False)

    N, B = mu.shape
    console.print(f"[info]Loaded μ shape: {N}×{B}")

    # Compute features
    feat_cfg = FeatureConfig(
        use_entropy=use_entropy,
        use_fft_ratio=use_fft_ratio,
        use_shap_mean=(use_shap_mean and shap_bins is not None),
        use_symbolic_score=(use_symbolic_score and symbolic_json is not None),
        fft_keep=fft_keep,
    )
    features_df, masks_by_name = compute_band_depth_features(
        mu=mu,
        bands_cfg=bands_cfg,
        wavelengths=wavelengths,
        shap_bins=shap_bins,
        symbolic_json=symbolic_json,
        feat_cfg=feat_cfg,
    )

    # Save band depths/features table
    band_depths_csv = tables / "band_depths.csv"
    features_df.to_csv(band_depths_csv, index=False)

    # Prepare feature matrix
    X = features_df.values.astype(float)
    if scale_features:
        X = StandardScaler().fit_transform(X)

    # Cluster
    labels = cluster_features(
        X, method=clustering, n_clusters=n_clusters, seed=seed,
    )
    # Fix DBSCAN noise label (-1) by shifting to a separate cluster index at end
    if labels.min() < 0:
        labels = labels.copy()
        labels[labels < 0] = labels.max() + 1

    # Save assignments
    assign_df = pd.DataFrame({"sample": np.arange(N), "cluster": labels})
    if meta_df is not None:
        assign_df = pd.concat([assign_df, meta_df.reset_index(drop=True)], axis=1)
    cluster_assign_csv = tables / "cluster_assignments.csv"
    assign_df.to_csv(cluster_assign_csv, index=False)

    # Cluster stats
    rows = []
    uniq = np.unique(labels)
    for c in uniq:
        sel = (labels == c)
        rows.append({"cluster": int(c), "count": int(sel.sum()), "fraction": float(sel.mean())})
    cluster_stats_df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    cluster_stats_csv = tables / "cluster_stats.csv"
    cluster_stats_df.to_csv(cluster_stats_csv, index=False)

    # Plots — overlays per cluster
    pngs_overlays = plot_cluster_overlays_png(
        mu=mu, labels=labels, wavelengths=wavelengths, masks_by_name=masks_by_name, outdir=plots
    ) if save_png else []

    html_overlays = plot_cluster_overlays_html(
        mu=mu, labels=labels, wavelengths=wavelengths, masks_by_name=masks_by_name, outdir=plots
    )

    # Heatmap sorted by cluster
    heatmap_by_cluster = plot_heatmap_sorted_png(mu=mu, labels=labels, wavelengths=wavelengths, outdir=plots, by="cluster")
    if not save_png:
        # Ensure at least one static plot exists for HTML report preview
        pngs_overlays = [heatmap_by_cluster]
    else:
        pngs_overlays.append(heatmap_by_cluster)

    # Projections (optional)
    proj = compute_projections(X, do_umap=umap_flag, do_tsne=tsne_flag, seed=seed)
    if "pca2" in proj:
        plot_embedding_html(proj["pca2"], labels, meta_df, "PCA — clusters", plots / "pca_clusters.html")
        html_overlays.append("pca_clusters.html")
    if "umap2" in proj:
        plot_embedding_html(proj["umap2"], labels, meta_df, "UMAP — clusters", plots / "umap_clusters.html")
        html_overlays.append("umap_clusters.html")
    if "tsne2" in proj:
        plot_embedding_html(proj["tsne2"], labels, meta_df, "t-SNE — clusters", plots / "tsne_clusters.html")
        html_overlays.append("tsne_clusters.html")

    # Summary JSON
    summary = {
        "timestamp": now_str(),
        "mu_path": mu_path,
        "wavelengths_path": wavelengths_path,
        "bands_path": bands_path,
        "meta_csv": meta_csv,
        "symbolic_path": symbolic_path,
        "shap_bins_path": shap_bins_path,
        "N": int(N),
        "B": int(B),
        "n_clusters": int(n_clusters),
        "clustering": clustering,
        "features_cols": features_df.columns.tolist(),
        "tables": {
            "band_depths.csv": band_depths_csv.name,
            "cluster_assignments.csv": cluster_assign_csv.name,
            "cluster_stats.csv": cluster_stats_csv.name,
        },
        "plots": {
            "png": pngs_overlays,
            "html": html_overlays,
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
                "band_depths": band_depths_csv.name,
                "cluster_assignments": cluster_assign_csv.name,
                "cluster_stats": cluster_stats_csv.name,
            },
            pngs=pngs_overlays,
            htmls=html_overlays,
        )
        console.print(f"[info]Wrote HTML report → {out / report_name}")

    # Audit log (best-effort)
    append_audit_log(f"- {now_str()} | spectral_absorption_overlay_clustered | mu={mu_path} out={outdir} bands={bands_path or 'none'} clusters={n_clusters} method={clustering} N={N} B={B}")

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
    bands: Optional[str] = typer.Option(None, help="Bands YAML/JSON: {unit:'um'|'index'|'auto', bands:{name:[lo,hi]}, continuum:[lo,hi]?}"),
    meta: Optional[str] = typer.Option(None, help="Metadata CSV (N rows) for hover/labels (optional)"),
    symbolic: Optional[str] = typer.Option(None, help="Symbolic overlay JSON (optional)"),
    shap_bins: Optional[str] = typer.Option(None, help="Per-bin SHAP N×B .npy (optional)"),
    n_clusters: int = typer.Option(5, min=1, help="Number of clusters (ignored for DBSCAN unless noise-only)"),
    clustering: str = typer.Option("kmeans", help="kmeans|minibatchkmeans|dbscan|agglomerative"),
    seed: int = typer.Option(42, help="Random seed"),
    scale_features: bool = typer.Option(True, help="Standardize features before clustering"),
    use_entropy: bool = typer.Option(True, help="Include entropy feature"),
    use_fft_ratio: bool = typer.Option(True, help="Include FFT high-frequency ratio feature"),
    fft_keep: int = typer.Option(32, help="Number of low freqs to 'keep' when computing high-frequency ratio"),
    use_shap_mean: bool = typer.Option(True, help="Include mean(|SHAP|) if --shap-bins provided"),
    use_symbolic_score: bool = typer.Option(True, help="Include symbolic violation score if --symbolic provided"),
    umap_flag: bool = typer.Option(False, "--umap/--no-umap", help="Compute UMAP projection if available"),
    tsne_flag: bool = typer.Option(False, "--tsne/--no-tsne", help="Compute t-SNE projection"),
    save_png: bool = typer.Option(False, help="Save static PNG overlays/heatmaps"),
    html: bool = typer.Option(False, help="Emit compact HTML report"),
):
    """
    Cluster spectra using band-depth features and render per-cluster overlays + tables.
    """
    try:
        run_pipeline(
            mu_path=mu,
            outdir=outdir,
            wavelengths_path=wavelengths,
            bands_path=bands,
            meta_csv=meta,
            symbolic_path=symbolic,
            shap_bins_path=shap_bins,
            n_clusters=n_clusters,
            clustering=clustering,
            seed=seed,
            scale_features=scale_features,
            use_entropy=use_entropy,
            use_fft_ratio=use_fft_ratio,
            fft_keep=fft_keep,
            use_shap_mean=use_shap_mean,
            use_symbolic_score=use_symbolic_score,
            umap_flag=umap_flag,
            tsne_flag=tsne_flag,
            save_png=save_png,
            html_report=html,
        )
    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", style="err"))
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
