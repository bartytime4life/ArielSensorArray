#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_fft_power_cluster_compare.py

SpectraMind V50 — FFT Power Cluster Compare (Upgraded)
======================================================

Purpose
-------
Compute and compare **FFT power spectra** of μ(λ) (or any 1D series per planet) across
clusters (e.g., symbolic/UMAP clusters), then export interactive and static diagnostics:

• Per‑cluster mean/median FFT power curves with confidence bands
• Peak detection and per‑cluster top‑K frequency tables
• Heatmaps (cluster × frequency) and optional per‑planet small multiples
• Optional detrend/window/normalization controls for robust spectral analysis
• Interactive Plotly HTML + static PNG/SVG (via kaleido)
• CLI‑ready; consistent with SpectraMind V50 terminal‑first diagnostics

References
----------
• FFT/spectral analysis for patterns and periodicity (Fourier components):contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}  
• CLI‑first workflow and diagnostics philosophy (Typer/Hydra/terminal‑light dashboards):contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}  
• Scientific modeling rigor & reproducibility mindset (NASA‑grade M&S):contentReference[oaicite:4]{index=4}  

Inputs
------
1) --spectra  : .npy or .csv with shape (N, B) (N=planets/items, B=bins or time/wavelength)
                 If .csv: numeric columns are taken as series; optional 'planet_id' column used as IDs.
2) --labels   : .csv with columns ['planet_id', 'cluster', ...] (cluster grouping required)
3) --wavelengths : optional vector for axis labeling (.npy/.csv with length B)
4) Optional normalization/detrending/windowing parameters.

Outputs
-------
• {outdir}/fft_cluster_compare.html
• {outdir}/fft_cluster_compare.png / .svg  (if requested)
• {outdir}/fft_cluster_power_{aggregate}.csv
• {outdir}/fft_cluster_peaks_topk.csv
• {outdir}/fft_heatmap_{aggregate}.png/.html (if requested)

Examples
--------
python tools/plot_fft_power_cluster_compare.py \
  --spectra outputs/predictions/mu.npy \
  --labels outputs/diagnostics/planet_labels.csv \
  --wavelengths data/meta/wavelengths.csv \
  --aggregate mean --topk 5 --html --png --outdir outputs/fft_compare

python tools/plot_fft_power_cluster_compare.py \
  --spectra outputs/predictions/mu.csv \
  --labels outputs/diagnostics/planet_labels.csv \
  --normalize unit --detrend poly:1 --window hann \
  --freq-max 0.5 --aggregate median --heatmap --html --svg \
  --outdir outputs/fft_compare

Author
------
SpectraMind V50 — Architect & Master Programmer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# I/O Utilities
# ---------------------------------------------------------------------------

def _load_spectra(path: str, id_col: Optional[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load spectra as (meta_df, X) where X shape = (N, B).
    If CSV:
      • If id_col provided or 'planet_id' exists => keep it as string id column.
      • All numeric columns are treated as the series.
    If NPY:
      • Returns meta index 0..N-1 as 'planet_id' strings.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        X = np.load(p)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D npy array (N,B), got {X.shape}")
        meta = pd.DataFrame({"planet_id": [str(i) for i in range(X.shape[0])]})
        return meta, X

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        cols = df.columns.tolist()
        if id_col and id_col in df.columns:
            ids = df[id_col].astype(str)
            feat = df.drop(columns=[id_col])
        elif "planet_id" in df.columns:
            ids = df["planet_id"].astype(str)
            feat = df.drop(columns=["planet_id"])
        else:
            # Use first non-numeric column as id if exists; else index
            nonnum = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            if nonnum:
                ids = df[nonnum[0]].astype(str)
                feat = df.drop(columns=[nonnum[0]])
            else:
                ids = pd.Series([str(i) for i in range(len(df))])
                feat = df
        feat = feat.select_dtypes(include=[np.number])
        X = feat.values
        meta = pd.DataFrame({"planet_id": ids})
        return meta, X

    raise ValueError(f"Unsupported spectra file type: {path}")


def _load_labels(path: Optional[str]) -> pd.DataFrame:
    if not path:
        raise ValueError("--labels is required (must contain 'planet_id' and 'cluster')")
    df = pd.read_csv(path)
    if "planet_id" not in df.columns or "cluster" not in df.columns:
        raise ValueError("Labels CSV must contain columns: 'planet_id', 'cluster'")
    df["planet_id"] = df["planet_id"].astype(str)
    df["cluster"] = df["cluster"].astype(str)
    return df


def _load_axis_vector(path: Optional[str], B: int) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() == ".npy":
        v = np.load(p)
    else:
        v = pd.read_csv(p, header=None).iloc[:, 0].values
    v = np.asarray(v).reshape(-1)
    if len(v) != B:
        raise ValueError(f"Wavelength/axis length mismatch: got {len(v)}, expected {B}")
    return v


# ---------------------------------------------------------------------------
# Preprocessing (detrend / normalization / window)
# ---------------------------------------------------------------------------

def _detrend(series: np.ndarray, mode: str) -> np.ndarray:
    """
    Detrend per-row series.
    mode: 'none' | 'poly:K' (K >=1) | 'mean' | 'linear'
    """
    if mode == "none":
        return series
    X = series.copy()
    N, B = X.shape
    if mode == "mean":
        X -= X.mean(axis=1, keepdims=True)
        return X
    if mode == "linear":
        # simple linear fit per row
        x = np.linspace(0.0, 1.0, B)
        A = np.c_[x, np.ones_like(x)]
        for i in range(N):
            m, c = np.linalg.lstsq(A, X[i], rcond=None)[0]
            X[i] = X[i] - (m * x + c)
        return X
    if mode.startswith("poly:"):
        try:
            k = int(mode.split(":")[1])
        except Exception:
            raise ValueError("poly detrend expects 'poly:K' with integer K>=1")
        x = np.linspace(-1.0, 1.0, B)
        V = np.vstack([x**d for d in range(k, -1, -1)]).T  # [x^k ... x^0]
        for i in range(N):
            coef, *_ = np.linalg.lstsq(V, X[i], rcond=None)
            fit = V @ coef
            X[i] = X[i] - fit
        return X
    raise ValueError(f"Unknown detrend mode: {mode}")


def _normalize(series: np.ndarray, mode: str) -> np.ndarray:
    """
    Normalize per-row series.
    mode: 'none' | 'zscore' | 'unit' (scale to max abs 1) | 'minmax'
    """
    if mode == "none":
        return series
    X = series.copy()
    if mode == "zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mu) / sd
    if mode == "unit":
        s = np.max(np.abs(X), axis=1, keepdims=True) + 1e-12
        return X / s
    if mode == "minmax":
        mn = np.min(X, axis=1, keepdims=True)
        mx = np.max(X, axis=1, keepdims=True)
        return (X - mn) / (mx - mn + 1e-12)
    raise ValueError(f"Unknown normalize mode: {mode}")


def _apply_window(series: np.ndarray, window: str) -> np.ndarray:
    """
    Apply a data window per-row to reduce spectral leakage (Hann/Hamming/Blackman).
    window: 'none'|'hann'|'hamming'|'blackman'
    """
    if window == "none":
        return series
    N, B = series.shape
    if window == "hann":
        w = np.hanning(B)
    elif window == "hamming":
        w = np.hamming(B)
    elif window == "blackman":
        w = np.blackman(B)
    else:
        raise ValueError(f"Unknown window: {window}")
    return series * w.reshape(1, -1)


# ---------------------------------------------------------------------------
# FFT Computation
# ---------------------------------------------------------------------------

def _rfft_power(X: np.ndarray, freq_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided real FFT power for each row.
    Returns (freqs, P) where P shape = (N, F).
    freq axis normalized to [0, 1) by default (Nyquist=0.5) assuming unit sample spacing.
    If freq_max provided (0<freq_max<=0.5), truncate to that max.
    """
    N, B = X.shape
    # Real FFT
    F = np.fft.rfft(X, axis=1)
    P = (F * np.conj(F)).real / B  # normalized power
    freqs = np.fft.rfftfreq(B, d=1.0)  # Nyquist at 0.5 for unit spacing
    if freq_max is not None:
        mask = freqs <= freq_max
        return freqs[mask], P[:, mask]
    return freqs, P


def _aggregate_by_cluster(P: np.ndarray, clusters: pd.Series, aggregate: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Aggregate power by cluster label using mean/median; also compute per-cluster std for bands.
    Returns:
      df_clusters: mapping of row index -> cluster (aligns with P rows)
      agg: dict cluster -> aggregated power (1D)
    """
    if aggregate not in {"mean", "median"}:
        raise ValueError("--aggregate must be 'mean' or 'median'")
    dfc = pd.DataFrame({"cluster": clusters.values, "idx": np.arange(len(clusters))})
    agg: Dict[str, np.ndarray] = {}
    for cl, g in dfc.groupby("cluster"):
        Pg = P[g["idx"].values]
        if aggregate == "mean":
            agg[cl] = Pg.mean(axis=0)
        else:
            agg[cl] = np.median(Pg, axis=0)
    return dfc, agg


def _per_cluster_std(P: np.ndarray, clusters: pd.Series) -> Dict[str, np.ndarray]:
    stds: Dict[str, np.ndarray] = {}
    dfc = pd.DataFrame({"cluster": clusters.values, "idx": np.arange(len(clusters))})
    for cl, g in dfc.groupby("cluster"):
        Pg = P[g["idx"].values]
        stds[cl] = Pg.std(axis=0)
    return stds


def _find_topk_peaks(y: np.ndarray, freqs: np.ndarray, k: int) -> List[Tuple[float, float]]:
    """
    Simple peak finder: select top-K values (excluding DC if desired later).
    Returns list of (freq, value) sorted by descending value.
    """
    if k <= 0:
        return []
    idx = np.argsort(-y)[:k]
    pairs = [(float(freqs[i]), float(y[i])) for i in idx]
    pairs.sort(key=lambda t: -t[1])
    return pairs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_clusters(freqs: np.ndarray,
                   agg: Dict[str, np.ndarray],
                   stds: Optional[Dict[str, np.ndarray]],
                   title: str,
                   reference: Optional[str] = None) -> go.Figure:
    """
    Interactive overlay of per-cluster spectra with optional ±1σ bands.
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
    keys = sorted(agg.keys())
    for i, cl in enumerate(keys):
        y = agg[cl]
        color = palette[i % len(palette)]
        width = 3.0 if (reference is not None and cl == reference) else 2.0
        fig.add_trace(go.Scatter(
            x=freqs, y=y, mode="lines", name=f"cluster={cl}",
            line=dict(color=color, width=width),
            hovertemplate="f=%{x:.4f}<br>P=%{y:.6g}<extra>"+f"{cl}</extra>"
        ))
        if stds and cl in stds:
            s = stds[cl]
            fig.add_trace(go.Scatter(
                x=np.concatenate([freqs, freqs[::-1]]),
                y=np.concatenate([y - s, (y + s)[::-1]]),
                fill='toself', fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"),
                line=dict(color='rgba(0,0,0,0)'), name=f"{cl} ±1σ",
                hoverinfo='skip', showlegend=False
            ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Frequency (Nyquist=0.5, Δx=1)",
        yaxis_title="Power",
        legend=dict(itemsizing="constant"),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def _plot_heatmap(freqs: np.ndarray, agg: Dict[str, np.ndarray], title: str) -> go.Figure:
    """
    Heatmap clusters × freq of aggregated power.
    """
    clusters = sorted(agg.keys())
    Z = np.vstack([agg[c] for c in clusters])
    fig = go.Figure(data=go.Heatmap(
        z=Z, x=freqs, y=clusters, colorscale="Viridis", colorbar=dict(title="Power")
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Frequency",
        yaxis_title="Cluster",
        margin=dict(l=70, r=20, t=60, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run(
    spectra_path: str,
    labels_path: str,
    wavelengths_path: Optional[str],
    id_col: Optional[str],
    aggregate: str,
    topk: int,
    detrend_mode: str,
    normalize_mode: str,
    window: str,
    freq_max: Optional[float],
    reference_cluster: Optional[str],
    include_heatmap: bool,
    outdir: str,
    save_html: bool,
    save_png: bool,
    save_svg: bool,
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    meta, X = _load_spectra(spectra_path, id_col=id_col)
    labels = _load_labels(labels_path)
    if X.ndim != 2:
        raise ValueError("Spectra must be 2D (N,B)")
    B = X.shape[1]
    wl = _load_axis_vector(wavelengths_path, B)  # optional, not mandatory

    # Align labels to spectra by planet_id
    meta["planet_id"] = meta["planet_id"].astype(str)
    df = meta.merge(labels[["planet_id", "cluster"]], on="planet_id", how="left")
    if df["cluster"].isna().any():
        missing = df[df["cluster"].isna()]["planet_id"].head(10).tolist()
        raise ValueError(f"Missing cluster labels for some planet_id, e.g. {missing[:5]} ...")

    # Preprocess series
    Xp = _detrend(X, detrend_mode)
    Xp = _normalize(Xp, normalize_mode)
    Xp = _apply_window(Xp, window)

    # FFT power
    freqs, P = _rfft_power(Xp, freq_max=freq_max)

    # Aggregate
    _, agg = _aggregate_by_cluster(P, df["cluster"], aggregate=aggregate)
    stds = _per_cluster_std(P, df["cluster"])

    # Peaks
    rows = []
    for cl, y in agg.items():
        # Exclude DC from peaks (index 0) to avoid dominating constant offset
        y_sel = y.copy()
        y_sel[0] = 0.0
        peaks = _find_topk_peaks(y_sel, freqs, topk)
        for rank, (f, val) in enumerate(peaks, start=1):
            rows.append({"cluster": cl, "rank": rank, "freq": f, "power": val})
    peaks_df = pd.DataFrame(rows).sort_values(["cluster", "rank"])
    peaks_df.to_csv(out / f"fft_cluster_peaks_topk.csv", index=False)

    # Export aggregated power as CSV
    agg_df = pd.DataFrame({"freq": freqs})
    for cl, y in agg.items():
        agg_df[f"power_{cl}"] = y
    agg_df.to_csv(out / f"fft_cluster_power_{aggregate}.csv", index=False)

    # Plots
    title = f"FFT Power — Cluster Compare ({aggregate})"
    fig = _plot_clusters(freqs, agg, stds, title=title, reference=reference_cluster)

    # Save main figure
    stem = "fft_cluster_compare"
    if save_html:
        fig.write_html(str(out / f"{stem}.html"), include_plotlyjs="cdn", full_html=True)
    if save_png:
        try:
            fig.write_image(str(out / f"{stem}.png"), scale=2)
        except Exception as e:
            print(f"[WARN] PNG export failed (install kaleido): {e}")
    if save_svg:
        try:
            fig.write_image(str(out / f"{stem}.svg"))
        except Exception as e:
            print(f"[WARN] SVG export failed (install kaleido): {e}")

    # Optional heatmap
    if include_heatmap:
        hfig = _plot_heatmap(freqs, agg, title=f"FFT Power Heatmap ({aggregate})")
        if save_html:
            hfig.write_html(str(out / f"{stem}_heatmap.html"), include_plotlyjs="cdn", full_html=True)
        if save_png:
            try:
                hfig.write_image(str(out / f"{stem}_heatmap.png"), scale=2)
            except Exception as e:
                print(f"[WARN] Heatmap PNG export failed: {e}")
        if save_svg:
            try:
                hfig.write_image(str(out / f"{stem}_heatmap.svg"))
            except Exception as e:
                print(f"[WARN] Heatmap SVG export failed: {e}")

    # Console summary
    print(f"[OK] Saved peak table: {out / 'fft_cluster_peaks_topk.csv'}")
    print(f"[OK] Saved cluster power CSV: {out / f'fft_cluster_power_{aggregate}.csv'}")
    if save_html:
        print(f"[OK] Saved HTML: {out / f'{stem}.html'}")
        if include_heatmap:
            print(f"[OK] Saved Heatmap HTML: {out / f'{stem}_heatmap.html'}")
    if save_png:
        print(f"[OK] Saved PNG: {out / f'{stem}.png'}")
        if include_heatmap:
            print(f"[OK] Saved Heatmap PNG: {out / f'{stem}_heatmap.png'}")
    if save_svg:
        print(f"[OK] Saved SVG: {out / f'{stem}.svg'}")
        if include_heatmap:
            print(f"[OK] Saved Heatmap SVG: {out / f'{stem}_heatmap.svg'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpectraMind V50 — FFT Power Cluster Compare (Upgraded)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spectra", type=str, required=True,
                   help=".npy or .csv (N,B) with per-planet series (μ(λ) or time). CSV may include 'planet_id'.")
    p.add_argument("--labels", type=str, required=True,
                   help="CSV with columns ['planet_id','cluster', ...] to group spectra.")
    p.add_argument("--wavelengths", type=str, default=None,
                   help="Optional wavelengths axis (.npy/.csv length B); used only for sanity checks/metadata.")
    p.add_argument("--id-col", type=str, default=None,
                   help="Explicit id column name in spectra CSV if not using 'planet_id'.")

    # Processing controls
    p.add_argument("--detrend", dest="detrend_mode", type=str, default="mean",
                   choices=["none", "mean", "linear"] + [f"poly:{k}" for k in range(1, 6)],
                   help="Detrend each series before FFT; e.g., mean removal, linear, or polynomial fit.")
    p.add_argument("--normalize", dest="normalize_mode", type=str, default="unit",
                   choices=["none", "zscore", "unit", "minmax"],
                   help="Normalize each series prior to FFT.")
    p.add_argument("--window", type=str, default="hann",
                   choices=["none", "hann", "hamming", "blackman"],
                   help="Apply data window to reduce leakage.")

    # FFT & aggregation
    p.add_argument("--freq-max", type=float, default=None,
                   help="Optional max frequency (<=0.5) to truncate the spectrum.")
    p.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"],
                   help="Aggregate across cluster members.")
    p.add_argument("--topk", type=int, default=5, help="Top-K peak frequencies per cluster.")
    p.add_argument("--reference-cluster", type=str, default=None,
                   help="Highlight this cluster (thicker line).")
    p.add_argument("--heatmap", action="store_true", help="Also export a cluster × freq heatmap.")

    # Outputs
    p.add_argument("--outdir", type=str, default="outputs/fft_cluster_compare", help="Output directory.")
    p.add_argument("--html", action="store_true", help="Export interactive HTML.")
    p.add_argument("--png", action="store_true", help="Export static PNG (requires kaleido).")
    p.add_argument("--svg", action="store_true", help="Export static SVG (requires kaleido).")
    return p


def main():
    args = build_argparser().parse_args()
    run(
        spectra_path=args.spectra,
        labels_path=args.labels,
        wavelengths_path=args.wavelengths,
        id_col=args.id_col,
        aggregate=args.aggregate,
        topk=int(args.topk),
        detrend_mode=args.detrend_mode,
        normalize_mode=args.normalize_mode,
        window=args.window,
        freq_max=float(args.freq_max) if args.freq_max is not None else None,
        reference_cluster=args.reference_cluster,
        include_heatmap=bool(args.heatmap),
        outdir=args.outdir,
        save_html=bool(args.html),
        save_png=bool(args.png),
        save_svg=bool(args.svg),
    )


if __name__ == "__main__":
    main()
