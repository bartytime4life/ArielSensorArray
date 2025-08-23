#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/fft_power_compare.py

SpectraMind V50 — FFT Power Compare (FGS1 & AIRS, pre/post calibration)

Purpose
-------
Compute and compare frequency-domain power (FFT/Welch) of:
  • FGS1 time-series (photometry) and
  • AIRS time-series per wavelength (or aggregated),
before and after calibration/detrending. Export rich diagnostics:
  • Matplotlib PNG plots (log–log power vs frequency)
  • Optional Plotly HTML with hover
  • CSV/JSON summaries (peak bands, band power, ratios)
  • Markdown mini-report (optional)
  • Append audit entry to logs/v50_debug_log.md

Design
------
• CLI-first (Typer) with robust flags & friendly output (Rich).
• Safe readers: .npy, .csv, .txt, and optional HDF5 (h5py if available).
• AIRS: support (T, W) arrays or (P, T, W); reduce via mean/median across W or per-band.
• FGS1: support (T,) or (P, T).
• Welch PSD (scipy.signal.welch) with tunable nperseg/overlap/window.
• Physics-aware helpers: mark low-f jitter, remove DC, show slope fits.

References (project context, CLI strategy, challenge & instruments)
------------------------------------------------------------------
• SpectraMind V50 CLI-first, Typer, Rich console dashboards, artifact logging
   
• Reproducibility & logging (config-as-code; append-to-log practice)
   
• CLI UX best practices (help, progress, human-friendly errors)
   
• Ariel mission instruments (FGS / AIRS) and noisy low-frequency systematics
  
• Kaggle pipelines & runtime context (notebooks/datasets/limits)
  

Usage
-----
  # Minimal: FGS1 pre/post (1D .npy) and AIRS pre/post (2D .npy: T×W)
  python tools/fft_power_compare.py \
      --fgs1-pre  data/fgs1_pre.npy  --fgs1-post  data/fgs1_post.npy \
      --airs-pre  data/airs_pre.npy  --airs-post  data/airs_post.npy \
      --rate 0.5  --outdir outputs/fft_compare

  # HDF5 (auto-discovery) with Welch tuning and HTML:
  python tools/fft_power_compare.py \
      --h5 cal/LightCurves.h5 \
      --fgs1-key /FGS1/cal --airs-key /AIRS/cal \
      --compare-raw True \
      --welch-nperseg 8192 --welch-overlap 0.5 --welch-window hann \
      --html True --open-html False

Notes
-----
• Sample rate (--rate) = samples per second (Hz). For the challenge’s simulated cadence,
  pass the correct sampling rate for FGS1/AIRS time axis to get physical Hz.
• If unknown, you can set --rate 1.0 (dimensionless frequency).
• HDF5 auto-discovery expects groups like /FGS1/{time,raw,cal} and /AIRS/{time,raw,cal}.

Outputs
-------
outdir/
  ├── fft_compare_fgs1.png
  ├── fft_compare_airs.png
  ├── fft_compare_airs_heatmap.png (optional, if --airs-heatmap)
  ├── fft_compare_plotly.html (if --html)
  ├── fft_summary.json
  ├── fft_summary.csv
  └── fft_report.md (optional)

"""

from __future__ import annotations

import json
import math
import os
import sys
import csv
import time
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

# Optional imports guarded
try:
    import h5py  # type: ignore
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

try:
    from scipy.signal import welch, get_window  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import plotly.graph_objs as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


app = typer.Typer(help="SpectraMind V50 — FFT Power Compare (FGS1/AIRS, pre/post)",
                  add_completion=False)
console = Console()


# ----------------------------- Helpers & I/O --------------------------------- #

def _ensure_dir(path: str | os.PathLike) -> str:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _load_simple(path: str) -> np.ndarray:
    """
    Load .npy / .npz (arr) / .csv / .txt. Returns float64 array.
    Shapes commonly:
      FGS1: (T,) or (P,T)
      AIRS: (T,W) or (P,T,W)
    """
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in [".npy"]:
        arr = np.load(path)
    elif suffix in [".npz"]:
        with np.load(path) as npz:
            # Heuristic: pick first array
            key = list(npz.keys())[0]
            arr = npz[key]
    elif suffix in [".csv", ".txt"]:
        arr = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported format: {path}")
    return np.asarray(arr, dtype=np.float64)


def _h5_read(h5_path: str,
             fgs1_key: Optional[str],
             airs_key: Optional[str],
             fgs1_raw_key: Optional[str],
             airs_raw_key: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Open HDF5 and read datasets. Returns:
       (fgs1_cal, fgs1_raw, airs_cal, airs_raw)
    Keys:
       fgs1_key like '/FGS1/cal' or '/FGS1/flux_cal'
       airs_key like '/AIRS/cal'  or '/AIRS/flux_cal'
       fgs1_raw_key '/FGS1/raw' (optional)
       airs_raw_key '/AIRS/raw'  (optional)
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py not installed. Install it or pass array files instead.")

    fgs1_cal = fgs1_raw = airs_cal = airs_raw = None

    with h5py.File(h5_path, "r") as h5:
        def get_or_none(k: Optional[str]):
            if not k:
                return None
            if k in h5:
                return np.asarray(h5[k], dtype=np.float64)
            # try flexible lookup
            try:
                return np.asarray(h5[k], dtype=np.float64)
            except Exception:
                return None

        fgs1_cal = get_or_none(fgs1_key)
        airs_cal = get_or_none(airs_key)
        fgs1_raw = get_or_none(fgs1_raw_key)
        airs_raw = get_or_none(airs_raw_key)

        # Lightweight auto-discovery if not provided
        if fgs1_cal is None:
            for cand in ["/FGS1/cal", "/FGS1/flux_cal", "/FGS1/flux"]:
                if cand in h5:
                    fgs1_cal = np.asarray(h5[cand], dtype=np.float64)
                    break
        if airs_cal is None:
            for cand in ["/AIRS/cal", "/AIRS/flux_cal", "/AIRS/flux"]:
                if cand in h5:
                    airs_cal = np.asarray(h5[cand], dtype=np.float64)
                    break
        if fgs1_raw is None:
            for cand in ["/FGS1/raw", "/FGS1/flux_raw"]:
                if cand in h5:
                    fgs1_raw = np.asarray(h5[cand], dtype=np.float64)
                    break
        if airs_raw is None:
            for cand in ["/AIRS/raw", "/AIRS/flux_raw"]:
                if cand in h5:
                    airs_raw = np.asarray(h5[cand], dtype=np.float64)
                    break

    return fgs1_cal, fgs1_raw, airs_cal, airs_raw


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is 2D for Welch handling:
      • (T,) -> (1,T)
      • (P,T) -> (P,T)
      • (T,W) -> treat as (T,W) will be reduced by aggregate path
      • (P,T,W) -> (P,T,W) handled in higher-level
    """
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    return arr  # leave higher dims to caller


@dataclass
class WelchConfig:
    nperseg: int = 4096
    overlap: float = 0.5
    window: str = "hann"
    detrend: str = "constant"
    scaling: str = "density"
    average: str = "mean"

    def window_array(self) -> Optional[np.ndarray]:
        if not HAS_SCIPY:
            return None
        try:
            return get_window(self.window, self.nperseg, fftbins=True)  # type: ignore
        except Exception:
            return None


def _welch_psd(x: np.ndarray, fs: float, cfg: WelchConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Welch per row of x (shape: N_series × T).
    Returns (freqs, mean_psd) averaged across series.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for Welch PSD (install scipy).")
    if x.ndim != 2:
        raise ValueError("Welch expects 2D array (N_series × T)")

    nperseg = min(cfg.nperseg, x.shape[-1])
    noverlap = int(cfg.overlap * nperseg)

    psds = []
    freqs = None
    window = cfg.window_array()

    for row in x:
        # Remove NaNs quietly
        row = np.asarray(row, dtype=np.float64)
        if np.any(np.isnan(row)):
            row = np.nan_to_num(row, nan=np.nanmedian(row))
        f, Pxx = welch(row, fs=fs, window=window if window is not None else cfg.window,
                       nperseg=nperseg, noverlap=noverlap, detrend=cfg.detrend,
                       scaling=cfg.scaling, average=cfg.average)  # type: ignore
        psds.append(Pxx)
        if freqs is None:
            freqs = f
    psds = np.asarray(psds)
    mean_psd = np.nanmean(psds, axis=0)
    return freqs, mean_psd


def _reduce_airs(airs: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Reduce AIRS cube to 2D series for PSD:
      • (T,W) -> reduce across W   => (1,T) or multiple if method='none'
      • (P,T,W) -> reduce across W => (P,T)
    method: 'mean' | 'median' | 'p50' | 'none' (no reduction; handled separately)
    """
    if airs.ndim == 2:
        T, W = airs.shape
        if method == "none":
            # return (W,T) for per-wavelength PSD then aggregate outside
            return airs.T  # (W,T)
        if method in ("median", "p50"):
            return np.median(airs, axis=1, keepdims=True).T  # (1,T)
        return np.mean(airs, axis=1, keepdims=True).T  # (1,T)
    if airs.ndim == 3:
        P, T, W = airs.shape
        if method == "none":
            # Return (P*W, T)
            return airs.reshape(P * T, W).T  # (W, P*T) → nope; keep consistent
        if method in ("median", "p50"):
            return np.median(airs, axis=2)  # (P,T)
        return np.mean(airs, axis=2)  # (P,T)
    raise ValueError("AIRS array must be (T,W) or (P,T,W)")


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    """Integrate PSD over a frequency band using trapezoidal rule."""
    fmin, fmax = band
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


# ----------------------------- Plotting -------------------------------------- #

def _plot_loglog(freqs, pre_psd, post_psd, out_png, title, bands=None, annotate_slope=True):
    if not HAS_MPL:
        return
    plt.figure(figsize=(9.5, 6.5), dpi=130)
    plt.loglog(freqs, pre_psd + 1e-30, label="Pre", alpha=0.85)
    plt.loglog(freqs, post_psd + 1e-30, label="Post", alpha=0.85)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density")
    plt.title(title)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend()

    if bands:
        ymin, ymax = np.min([pre_psd, post_psd]), np.max([pre_psd, post_psd])
        for (bmin, bmax, name, color) in bands:
            plt.fill_betweenx([ymin, ymax], bmin, bmax, color=color, alpha=0.08, label=name)
    if annotate_slope:
        # crude slope annotation over a mid band (between 2nd and 3rd decade if possible)
        try:
            idx = (freqs > np.percentile(freqs, 30)) & (freqs < np.percentile(freqs, 70))
            lf = np.log10(freqs[idx] + 1e-30)
            lp = np.log10(post_psd[idx] + 1e-30)
            # linear fit in log-log
            a = np.polyfit(lf, lp, 1)
            slope = a[0]
            x0 = 10 ** np.mean(lf)
            y0 = 10 ** np.polyval(a, np.mean(lf))
            x_line = np.logspace(np.min(lf), np.max(lf), 50)
            y_line = 10 ** np.polyval(a, np.log10(x_line))
            plt.loglog(x_line, y_line, 'k--', alpha=0.4, label=f"slope≈{slope:.2f}")
            plt.legend()
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_airs_heatmap(freqs, psd_matrix, out_png, title):
    """
    Heatmap of per-wavelength PSD:
      psd_matrix: shape (W, F) or (N_series, F) after reduction 'none'
    """
    if not HAS_MPL:
        return
    plt.figure(figsize=(10.5, 6.0), dpi=130)
    # Log10 for visibility
    im = plt.imshow(np.log10(psd_matrix + 1e-30), aspect='auto',
                    origin='lower', extent=[freqs[0], freqs[-1], 0, psd_matrix.shape[0]])
    plt.xscale("log")
    plt.xlabel("Frequency [Hz] (log)")
    plt.ylabel("Wavelength index (or series idx)")
    plt.title(title)
    plt.colorbar(im, label="log10(PSD)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plotly_dual(freqs, pre_psd, post_psd, title) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=freqs, y=pre_psd, mode='lines', name='Pre', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=freqs, y=post_psd, mode='lines', name='Post', line=dict(width=2)))
    fig.update_layout(title=title, xaxis_type="log", yaxis_type="log",
                      xaxis_title="Frequency [Hz]", yaxis_title="Power Spectral Density",
                      template="plotly_white", legend=dict(x=0.02, y=0.98))
    return fig


# ----------------------------- Core Routine ---------------------------------- #

def _compute_compare(
    pre_arr: Optional[np.ndarray],
    post_arr: Optional[np.ndarray],
    fs: float,
    welch_cfg: WelchConfig,
    label: str,
    outdir: str,
    bands_for_summary: List[Tuple[float, float, str]],
    is_airs: bool = False,
    airs_reduce: str = "mean",
    airs_heatmap: bool = False,
    html: bool = False,
    html_name: str = "fft_compare_plotly.html",
    html_open: bool = False,
) -> Dict[str, Any]:
    """
    Compute Welch PSD for pre/post, export plots & summary.
    For AIRS if reduce='none', produce heatmap as well.
    """
    result: Dict[str, Any] = {"label": label, "bands": {}, "figures": []}

    if pre_arr is None and post_arr is None:
        console.print(f"[yellow]Skipping {label}: no arrays provided[/yellow]")
        return result

    if pre_arr is None:
        pre_arr = post_arr
    if post_arr is None:
        post_arr = pre_arr

    # Reduce shapes
    if is_airs:
        if pre_arr is not None:
            if airs_reduce == "none":
                # (T,W) -> (W,T)
                if pre_arr.ndim == 2:
                    pre_series = pre_arr.T
                elif pre_arr.ndim == 3:
                    # (P,T,W) flatten by median across P then transpose to (W,T)
                    pre_series = np.median(pre_arr, axis=0).T
                else:
                    raise ValueError("AIRS array expected (T,W) or (P,T,W)")
            else:
                pre_series = _to_2d(_reduce_airs(pre_arr, method=airs_reduce))
        else:
            pre_series = None

        if post_arr is not None:
            if airs_reduce == "none":
                if post_arr.ndim == 2:
                    post_series = post_arr.T
                elif post_arr.ndim == 3:
                    post_series = np.median(post_arr, axis=0).T
                else:
                    raise ValueError("AIRS array expected (T,W) or (P,T,W)")
            else:
                post_series = _to_2d(_reduce_airs(post_arr, method=airs_reduce))
        else:
            post_series = None
    else:
        pre_series = _to_2d(pre_arr)
        post_series = _to_2d(post_arr)

    # Welch
    freqs_pre, psd_pre = _welch_psd(pre_series, fs, welch_cfg)
    freqs_post, psd_post = _welch_psd(post_series, fs, welch_cfg)

    # Plot PNG
    png = os.path.join(outdir, f"fft_compare_{label.lower()}.png")
    bands_vis = [(bmin, bmax, name, "#ff7f0e") for (bmin, bmax, name) in [(b[0], b[1], b[2]) for b in bands_for_summary]]
    _plot_loglog(freqs_post, psd_pre, psd_post, png, f"{label}: FFT Power (Welch)", bands=bands_vis)
    result["figures"].append(os.path.basename(png))

    # Optional heatmap for AIRS when reduce='none'
    if is_airs and airs_reduce == "none" and airs_heatmap:
        # Build per-wavelength PSD matrices (pre/post), store combined heatmap for 'post/pre ratio'
        def per_series_psd(series_2d):
            F = None
            PSDs = []
            for row in series_2d:
                f, pxx = _welch_psd(row[None, :], fs, welch_cfg)
                PSDs.append(pxx)
                if F is None:
                    F = f
            return F, np.asarray(PSDs)

        f_pre, m_pre = per_series_psd(pre_series)
        f_post, m_post = per_series_psd(post_series)
        # Align just in case
        assert np.allclose(f_pre, f_post)
        ratio = (m_post + 1e-30) / (m_pre + 1e-30)
        png_hm = os.path.join(outdir, f"fft_compare_{label.lower()}_heatmap.png")
        _plot_airs_heatmap(f_post, ratio, png_hm, f"{label}: per-wavelength POST/PRE PSD ratio (log10)")
        result["figures"].append(os.path.basename(png_hm))

    # Optional HTML
    if html and HAS_PLOTLY:
        fig = _plotly_dual(freqs_post, psd_pre, psd_post, f"{label}: Welch PSD (log–log)")
        html_path = os.path.join(outdir, html_name)
        if os.path.exists(html_path):
            # append as a new figure using a simple wrapper page
            with open(html_path, "a", encoding="utf-8") as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        else:
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        result["figures"].append(os.path.basename(html_path))
        if html_open:
            try:
                import webbrowser  # stdlib
                webbrowser.open(f"file://{os.path.abspath(html_path)}", new=2)
            except Exception:
                pass

    # Band power summary
    for (bmin, bmax, name) in bands_for_summary:
        bp_pre = _band_power(freqs_post, psd_pre, (bmin, bmax))
        bp_post = _band_power(freqs_post, psd_post, (bmin, bmax))
        ratio = bp_post / (bp_pre + 1e-30)
        result["bands"][name] = {
            "band": [bmin, bmax],
            "pre": bp_pre,
            "post": bp_post,
            "ratio": ratio
        }

    # Peak frequency (argmax) summary
    k_pre = int(np.argmax(psd_pre))
    k_post = int(np.argmax(psd_post))
    result["peak"] = {
        "pre": {"f": float(freqs_post[k_pre]), "psd": float(psd_pre[k_pre])},
        "post": {"f": float(freqs_post[k_post]), "psd": float(psd_post[k_post])}
    }

    return result


# ----------------------------- CLI ------------------------------------------- #

@app.command("run")
def run(
    # Inputs (either pass arrays directly OR one HDF5)
    fgs1_pre: Optional[str] = typer.Option(None, help="FGS1 pre-cal (npy/npz/csv/txt)"),
    fgs1_post: Optional[str] = typer.Option(None, help="FGS1 post-cal (npy/npz/csv/txt)"),
    airs_pre: Optional[str] = typer.Option(None, help="AIRS pre-cal (T×W npy/...)"),
    airs_post: Optional[str] = typer.Option(None, help="AIRS post-cal (T×W npy/...)"),
    h5: Optional[str] = typer.Option(None, help="HDF5 file for auto-discovery"),
    fgs1_key: Optional[str] = typer.Option(None, help="H5 dataset key for FGS1 cal (e.g. /FGS1/cal)"),
    fgs1_raw_key: Optional[str] = typer.Option(None, help="H5 dataset key for FGS1 raw (e.g. /FGS1/raw)"),
    airs_key: Optional[str] = typer.Option(None, help="H5 dataset key for AIRS cal (e.g. /AIRS/cal)"),
    airs_raw_key: Optional[str] = typer.Option(None, help="H5 dataset key for AIRS raw (e.g. /AIRS/raw)"),

    # Sampling & Welch
    rate: float = typer.Option(1.0, help="Sampling rate [Hz]"),
    welch_nperseg: int = typer.Option(4096, help="Welch nperseg"),
    welch_overlap: float = typer.Option(0.5, help="Welch overlap fraction [0..1)"),
    welch_window: str = typer.Option("hann", help="Welch window (scipy name)"),
    welch_detrend: str = typer.Option("constant", help="Welch detrend (constant/linear/None)"),
    welch_scaling: str = typer.Option("density", help="Welch scaling (density/spectrum)"),

    # AIRS aggregation
    airs_reduce: str = typer.Option("mean", help="AIRS per-wavelength reduction: mean|median|none"),
    airs_heatmap: bool = typer.Option(False, help="If reduce=none, export PSD ratio heatmap"),

    # Bands of interest
    low_band: str = typer.Option("1e-5,5e-4", help="Low-f band (e.g. spacecraft jitter) as 'fmin,fmax'"),
    mid_band: str = typer.Option("5e-4,5e-3", help="Mid band 'fmin,fmax'"),
    high_band: str = typer.Option("5e-3,5e-2", help="High band 'fmin,fmax'"),

    # Outputs
    outdir: str = typer.Option("outputs/fft_compare", help="Output directory"),
    html: bool = typer.Option(False, help="Write plotly HTML"),
    open_html: bool = typer.Option(False, help="Open HTML in browser"),
    report_md: bool = typer.Option(False, help="Write markdown mini-report"),
    log_append: bool = typer.Option(True, help="Append audit line to logs/v50_debug_log.md"),
    pretty: bool = typer.Option(True, help="Pretty-print summary table")
):
    """
    Run FFT/Welch power comparison on FGS1 and AIRS pre/post calibration streams.
    """
    t0 = time.time()
    console.rule("[bold cyan]SpectraMind V50 — FFT Power Compare")
    outdir = _ensure_dir(outdir)

    # Prepare Welch config
    welch_cfg = WelchConfig(
        nperseg=welch_nperseg,
        overlap=welch_overlap,
        window=welch_window,
        detrend=welch_detrend,
        scaling=welch_scaling,
        average="mean",
    )

    # Parse bands
    def parse_band(s: str, name: str) -> Tuple[float, float, str]:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise typer.BadParameter(f"Invalid {name} band: {s}")
        return (float(parts[0]), float(parts[1]), name)

    bands_for_summary = [
        parse_band(low_band, "low"),
        parse_band(mid_band, "mid"),
        parse_band(high_band, "high"),
    ]

    # Load inputs
    fgs1_pre_arr = fgs1_post_arr = airs_pre_arr = airs_post_arr = None

    if h5:
        console.print(f"[green]Loading HDF5:[/green] {h5}")
        if not HAS_H5PY:
            console.print("[red]h5py not installed; cannot read HDF5[/red]")
            raise typer.Exit(1)
        fgs1_cal, fgs1_raw, airs_cal, airs_raw = _h5_read(h5, fgs1_key, airs_key, fgs1_raw_key, airs_raw_key)
        # Use *cal as 'post'; optionally compare raw vs cal if present
        if fgs1_raw is not None:
            fgs1_pre_arr = fgs1_raw
        if fgs1_cal is not None:
            fgs1_post_arr = fgs1_cal
        if airs_raw is not None:
            airs_pre_arr = airs_raw
        if airs_cal is not None:
            airs_post_arr = airs_cal

    # Override from file flags if provided
    if fgs1_pre:
        fgs1_pre_arr = _load_simple(fgs1_pre)
    if fgs1_post:
        fgs1_post_arr = _load_simple(fgs1_post)
    if airs_pre:
        airs_pre_arr = _load_simple(airs_pre)
    if airs_post:
        airs_post_arr = _load_simple(airs_post)

    # Sanity
    if fgs1_pre_arr is None and fgs1_post_arr is None and airs_pre_arr is None and airs_post_arr is None:
        console.print("[red]No inputs provided. See --help for options.[/red]")
        raise typer.Exit(2)

    # Compute
    summary: Dict[str, Any] = {
        "sampling_rate_hz": rate,
        "welch": welch_cfg.__dict__,
        "bands": {b[2]: [b[0], b[1]] for b in bands_for_summary},
        "results": {}
    }

    with console.status("[bold green]Computing Welch PSDs...[/bold green]"):
        if fgs1_pre_arr is not None or fgs1_post_arr is not None:
            res_fgs1 = _compute_compare(
                fgs1_pre_arr, fgs1_post_arr, rate, welch_cfg,
                label="FGS1", outdir=outdir, bands_for_summary=bands_for_summary,
                is_airs=False, html=html, html_open=open_html
            )
            summary["results"]["FGS1"] = res_fgs1

        if airs_pre_arr is not None or airs_post_arr is not None:
            res_airs = _compute_compare(
                airs_pre_arr, airs_post_arr, rate, welch_cfg,
                label="AIRS", outdir=outdir, bands_for_summary=bands_for_summary,
                is_airs=True, airs_reduce=airs_reduce, airs_heatmap=airs_heatmap,
                html=html, html_open=open_html
            )
            summary["results"]["AIRS"] = res_airs

    # Export JSON/CSV
    json_path = os.path.join(outdir, "fft_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]Wrote[/green] {json_path}")

    # CSV table (bands × {pre,post,ratio} for each label)
    csv_path = os.path.join(outdir, "fft_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["label", "band", "fmin", "fmax", "pre_power", "post_power", "ratio", "peak_f_pre", "peak_f_post"]
        writer.writerow(header)
        for label, res in summary["results"].items():
            if not res or "bands" not in res:
                continue
            peak_pre = res.get("peak", {}).get("pre", {}).get("f", "")
            peak_post = res.get("peak", {}).get("post", {}).get("f", "")
            for band_name, row in res["bands"].items():
                writer.writerow([
                    label, band_name, row["band"][0], row["band"][1],
                    row["pre"], row["post"], row["ratio"], peak_pre, peak_post
                ])
    console.print(f"[green]Wrote[/green] {csv_path}")

    # Markdown mini-report
    if report_md:
        md_path = os.path.join(outdir, "fft_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# FFT Power Compare Report\n\n")
            f.write(f"- Sampling rate: **{rate} Hz**\n")
            f.write(f"- Welch: `nperseg={welch_nperseg}`, `overlap={welch_overlap}`, `window={welch_window}`\n\n")
            for label, res in summary["results"].items():
                f.write(f"## {label}\n\n")
                figs = res.get("figures", [])
                for fig in figs:
                    if fig.endswith(".png"):
                        f.write(f"![{fig}]({fig})\n\n")
                if "peak" in res:
                    pk = res["peak"]
                    f.write(f"- Peak pre: f={pk['pre']['f']:.6g} Hz; post: f={pk['post']['f']:.6g} Hz\n\n")
                f.write("| Band | fmin | fmax | Pre | Post | Ratio |\n")
                f.write("|---:|---:|---:|---:|---:|---:|\n")
                for bname, row in res["bands"].items():
                    f.write(f"| {bname} | {row['band'][0]:.3e} | {row['band'][1]:.3e} | "
                            f"{row['pre']:.3e} | {row['post']:.3e} | {row['ratio']:.3f} |\n")
                f.write("\n")
        console.print(f"[green]Wrote[/green] {md_path}")

    # Pretty print table
    if pretty:
        table = Table(title="Band Power Summary", box=box.SIMPLE_HEAVY)
        table.add_column("Label", justify="left", no_wrap=True)
        table.add_column("Band", justify="right")
        table.add_column("fmin..fmax [Hz]", justify="right")
        table.add_column("Pre", justify="right")
        table.add_column("Post", justify="right")
        table.add_column("Ratio", justify="right")
        for label, res in summary["results"].items():
            for bname, row in res.get("bands", {}).items():
                table.add_row(
                    label,
                    bname,
                    f"{row['band'][0]:.2e}..{row['band'][1]:.2e}",
                    f"{row['pre']:.3e}",
                    f"{row['post']:.3e}",
                    f"{row['ratio']:.3f}"
                )
        console.print(Panel(table, title="FFT Compare", border_style="cyan"))

    # Append to audit log (CLI call trace)
    if log_append:
        log_dir = _ensure_dir("logs")
        log_path = os.path.join(log_dir, "v50_debug_log.md")
        cmd = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"- [{time.strftime('%Y-%m-%d %H:%M:%S')}] fft_power_compare : `{cmd}` "
                    f"=> outdir `{outdir}` ; elapsed={time.time()-t0:.2f}s\n")
        console.print(f"[blue]Appended audit[/blue] {log_path}")

    console.rule("[bold green]Done")
    return


# ----------------------------- Entrypoint ------------------------------------ #

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
