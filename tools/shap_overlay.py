#!/usr/bin/env python3
"""
shap_overlay.py — Tools/: SHAP-based spectral attribution overlays for SpectraMind V50

Purpose
-------
Compute SHAP attributions for model predictions and generate publication-ready overlays that
align feature contributions with predicted exoplanet transmission spectra. Designed to work
with tabular or sequence-to-spectrum models used in the NeurIPS Ariel Data Challenge pipeline.

Key Features
------------
1) CLI (Typer) with rich progress, logging, and helpful --help docs
2) Flexible model loading:
   - TorchScript (.pt / .pth scripted)
   - Torch state_dict + user-specified factory import path
3) Dataset loading from .npy, .csv, or parquet; supports slicing and single-sample focus
4) SHAP explainers:
   - KernelExplainer (model-agnostic)
   - DeepExplainer (PyTorch; if model is compatible)
   - GradientExplainer (fallback when gradients available)
5) Spectral overlays:
   - Overlay SHAP contributions on predicted spectrum (line + positive/negative bands)
   - Per-wavelength contributions, optional smoothing
   - Optional grouping by spectral bands (e.g., molecule masks) for stacked areas
6) Batch or single-sample modes; export: PNG, SVG, HTML (Plotly), and CSV of attributions
7) Reproducibility: fixed seeds, clear metadata in outputs

Usage
-----
# Basic (TorchScript model, NumPy data)
python tools/shap_overlay.py run \
  --model-path checkpoints/model.ts \
  --input /data/x_val.npy \
  --output-dir artifacts/shap_overlay \
  --topk 8 \
  --explainer auto \
  --spectral-axis /data/wavelengths.npy

# State dict with factory
python tools/shap_overlay.py run \
  --model-path checkpoints/state_dict.pt \
  --factory "mylib.models:create_model" \
  --factory-kwargs '{"in_dim":2048,"out_dim":283}' \
  --input /data/x_val.npy \
  --output-dir artifacts/shap_overlay

# Single sample focus
python tools/shap_overlay.py run \
  --model-path checkpoints/model.ts \
  --input /data/x_val.npy \
  --index 42 \
  --output-dir artifacts/shap_overlay/s42

Notes
-----
- If you pass --band-map YAML (wavelength band -> mask indices), plots will include grouped stacks.
- If your model expects tensors shaped (B, T, F) but data is (B, F), use --reshape to expand dims.

Author: SpectraMind V50 (Neuro‑Symbolic Pipeline)
License: MIT
"""

from __future__ import annotations

import json
import os
import sys
import math
import time
import types
import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch

# Optional SHAP backends
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import plotly.graph_objects as go  # type: ignore
    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    _HAS_PLOTLY = False

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from rich.theme import Theme

app = typer.Typer(add_completion=True, no_args_is_help=True)
console = Console(theme=Theme({"ok": "green", "warn": "yellow", "err": "bold red"}))


# ---------------------------- Utilities & Config --------------------------------- #

@dataclass
class OverlayConfig:
    model_path: Path
    factory: Optional[str] = None                  # "module.submod:callable"
    factory_kwargs: Optional[Dict[str, Any]] = None
    input_path: Path = Path("")
    spectral_axis: Optional[Path] = None           # 1D array of wavelengths (nm or um)
    band_map: Optional[Path] = None                # YAML/JSON mapping "band_name": indices or [start, end]
    output_dir: Path = Path("artifacts/shap_overlay")
    index: Optional[int] = None                    # Single sample index (otherwise batch)
    batch: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    explainer: str = "auto"                        # auto|deep|kernel|gradient
    background: Optional[Path] = None              # optional background samples for KernelExplainer
    background_size: int = 128
    topk: int = 10                                 # for tabular feature-ranking
    smooth: int = 0                                # Savitzky-Golay window; 0 disables
    export_html: bool = True
    export_csv: bool = True
    dpi: int = 160
    seed: int = 42
    reshape: Optional[str] = None                  # e.g. "B,F->B,1,F" or "B,F->B,F,1"
    dtype: str = "float32"


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_callable(path: str):
    """Load a callable from 'module.submodule:attr' path."""
    if ":" not in path:
        raise ValueError("Factory path must be 'module.submodule:callable'")
    mod, attr = path.split(":")
    m = importlib.import_module(mod)
    fn = getattr(m, attr)
    if not callable(fn):
        raise TypeError(f"{path} is not callable.")
    return fn


def _load_array_any(path: Path) -> np.ndarray:
    """Load 1D/2D array from npy/csv/parquet."""
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path).values
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path).values
    raise ValueError(f"Unsupported data format: {path}")


def _maybe_reshape(x: np.ndarray, spec: Optional[str]) -> np.ndarray:
    if not spec:
        return x
    try:
        src, dst = spec.split("->")
        # Basic helpers for common patterns
        if src == "B,F" and dst == "B,1,F":
            return x[:, None, :]
        if src == "B,F" and dst == "B,F,1":
            return x[:, :, None]
        if src == "B,T,F" and dst == "B,T*F":
            b, t, f = x.shape
            return x.reshape(b, t * f)
        if src == "B,T*F" and dst == "B,T,F":
            raise ValueError("Provide explicit T,F to reshape back from 'B,T*F'.")
        # Fallback: eval shapes by dictionary
        raise ValueError(f"Unsupported reshape spec: {spec}")
    except Exception as e:
        raise ValueError(f"Bad --reshape spec '{spec}': {e}") from e


def _load_band_map(path: Path, n_out: int) -> Dict[str, np.ndarray]:
    """Load band map from YAML/JSON; return dict of name->indices boolean mask or index array."""
    import yaml  # lightweight dependency; common in envs
    with open(path, "r") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            obj = yaml.safe_load(f)
        else:
            obj = json.load(f)
    bands: Dict[str, np.ndarray] = {}
    for k, v in obj.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v):
            start, end = v
            idx = np.arange(start, end)
        elif isinstance(v, list) and all(isinstance(x, int) for x in v):
            idx = np.array(v, dtype=int)
        else:
            raise ValueError(f"Band map value for '{k}' must be [start,end] or [idx,...].")
        bands[k] = idx[(idx >= 0) & (idx < n_out)]
    return bands


# ---------------------------- Model Loading -------------------------------------- #

def load_model(cfg: OverlayConfig) -> torch.nn.Module:
    """
    Load a PyTorch model from:
      - TorchScript file (.pt/.pth) OR
      - state_dict via user-specified factory callable "module:factory"
    """
    model_path = cfg.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try TorchScript
    try:
        model = torch.jit.load(str(model_path), map_location=cfg.device)
        model.eval().to(cfg.device)
        return model
    except Exception:
        pass

    # Try state_dict + factory
    if not cfg.factory:
        raise ValueError(
            "Non-TorchScript model requires --factory 'module.submodule:create_model'."
        )
    fn = _resolve_callable(cfg.factory)
    kwargs = cfg.factory_kwargs or {}
    model = fn(**kwargs)
    sd = torch.load(str(model_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
        # Handle Lightning-style keys
        new_sd = {}
        for k, v in sd.items():
            new_sd[k.replace("model.", "").replace("net.", "")] = v
        sd = new_sd
    model.load_state_dict(sd, strict=False)
    model.eval().to(cfg.device)
    return model


@torch.no_grad()
def forward_model(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Run model forward pass on numpy batch -> numpy predictions."""
    t = torch.from_numpy(x).to(device)
    if t.dtype != torch.float32:
        t = t.float()
    y = model(t)
    if isinstance(y, (list, tuple)):
        y = y[0]
    return y.detach().cpu().numpy()


# ---------------------------- SHAP Explainers ------------------------------------ #

def build_explainer(cfg: OverlayConfig, model, background: np.ndarray, data_sample: np.ndarray):
    """
    Build appropriate SHAP explainer by cfg.explainer. 'auto' tries Deep -> Gradient -> Kernel.
    background: background batch for Kernel (and optionally for Deep)
    data_sample: a batch (B, ...) used to shape the explainer
    """
    if not _HAS_SHAP:
        raise RuntimeError("SHAP not installed. Please `pip install shap`.")

    explainer_type = cfg.explainer.lower()

    if explainer_type == "deep" or (explainer_type == "auto"):
        try:
            if isinstance(model, torch.nn.Module):
                e = shap.DeepExplainer(model, torch.from_numpy(background).to(cfg.device))
                return e, "deep"
        except Exception:
            if explainer_type == "deep":
                raise

    if explainer_type == "gradient" or (explainer_type == "auto"):
        try:
            if isinstance(model, torch.nn.Module):
                e = shap.GradientExplainer(model, torch.from_numpy(background).to(cfg.device))
                return e, "gradient"
        except Exception:
            if explainer_type == "gradient":
                raise

    # Fallback to KernelExplainer (model-agnostic)
    # Wrap model as callable f(x) -> y
    def fwrap(x):
        with torch.no_grad():
            xx = torch.from_numpy(x).to(cfg.device)
            if xx.dtype != torch.float32:
                xx = xx.float()
            yy = model(xx)
            if isinstance(yy, (list, tuple)):
                yy = yy[0]
            return yy.detach().cpu().numpy()

    e = shap.KernelExplainer(fwrap, background)
    return e, "kernel"


def compute_shap_values(
    explainer,
    x_batch: np.ndarray,
    output_dim: int,
    method_name: str,
    nsamples: Optional[int] = None,
) -> np.ndarray:
    """
    Return SHAP attributions shaped (B, input_dim, output_dim) or (B, input_dim) depending on explainer.
    For multi-output regression, SHAP returns a list per output OR a 3D array with axis for output.
    We harmonize to (B, input_dim, output_dim).
    """
    if method_name in {"deep", "gradient"}:
        # Deep/Gradient explainers in SHAP for PyTorch often support multi-output directly
        vals = explainer.shap_values(torch.from_numpy(x_batch))
        # vals can be:
        # - list of length output_dim, each (B, input_dim)
        # - or a single array (B, input_dim)
        if isinstance(vals, list) and len(vals) == output_dim:
            arr = np.stack([v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in vals], axis=2)
            return arr  # (B, input_dim, output_dim)
        if torch.is_tensor(vals):
            vals = vals.detach().cpu().numpy()
        if vals.ndim == 2:
            # replicate across outputs if only single explained head
            return np.repeat(vals[:, :, None], output_dim, axis=2)
        if vals.ndim == 3:
            return vals  # (B, input_dim, output_dim)
        raise RuntimeError("Unexpected SHAP shape for deep/gradient explainer.")

    # KernelExplainer: use shap_values(X, nsamples=...) which may return list per output
    vals = explainer.shap_values(x_batch, nsamples=nsamples)
    if isinstance(vals, list) and len(vals) == output_dim:
        arr = np.stack(vals, axis=2)  # (B, input_dim, output_dim)
        return arr
    if isinstance(vals, np.ndarray):
        if vals.ndim == 2:
            return np.repeat(vals[:, :, None], output_dim, axis=2)
        if vals.ndim == 3:
            return vals
    raise RuntimeError("Unexpected SHAP shape for kernel explainer.")


# ---------------------------- Plotting & Exports -------------------------------- #

def _savgol_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    from scipy.signal import savgol_filter  # lazy import
    # ensure odd window
    win = window if window % 2 == 1 else window + 1
    win = max(3, win)
    return savgol_filter(y, window_length=win, polyorder=min(3, win - 1))


def plot_overlay(
    outdir: Path,
    wavelengths: Optional[np.ndarray],
    y_pred: np.ndarray,             # (O,)
    shap_in_to_out: np.ndarray,     # (I, O) or (I,) if collapsed
    x_sample: np.ndarray,           # (I,) input features for sample
    topk: int,
    dpi: int,
    smooth: int,
    band_map: Optional[Dict[str, np.ndarray]] = None,
    html: bool = True,
    title: Optional[str] = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    O = y_pred.shape[0]
    xs = wavelengths if wavelengths is not None and len(wavelengths) == O else np.arange(O)

    # Aggregate feature contributions into per-wavelength effect
    if shap_in_to_out.ndim == 2:
        # contributions per input feature to each output wavelength
        contrib_w = shap_in_to_out.sum(axis=0)  # (O,)
    else:
        contrib_w = shap_in_to_out  # already collapsed shape (O,)

    if smooth > 0:
        y_plot = _savgol_smooth(y_pred, smooth)
        c_plot = _savgol_smooth(contrib_w, smooth)
    else:
        y_plot = y_pred
        c_plot = contrib_w

    # Matplotlib overlay
    fig, ax = plt.subplots(figsize=(11, 5), dpi=dpi)
    ax.plot(xs, y_plot, lw=2.0, color="#1f77b4", label="Predicted spectrum")

    # Positive/negative bands
    pos = np.clip(c_plot, 0, None)
    neg = np.clip(c_plot, None, 0)
    ax.fill_between(xs, y_plot, y_plot + pos, color="#2ca02c", alpha=0.25, label="Positive SHAP influence")
    ax.fill_between(xs, y_plot, y_plot + neg, color="#d62728", alpha=0.25, label="Negative SHAP influence")

    # Optional band stacks
    if band_map:
        for name, idx in band_map.items():
            idx = idx[(idx >= 0) & (idx < O)]
            if idx.size == 0:
                continue
            band_contrib = c_plot[idx]
            if band_contrib.size == 0:
                continue
            ax.plot(xs[idx], y_plot[idx] + band_contrib, lw=1.2, alpha=0.8, label=f"{name} stack")

    ax.set_xlabel("Wavelength" + ("" if wavelengths is None else " (axis units)"))
    ax.set_ylabel("Transit depth / Relative flux")
    if title:
        ax.set_title(title)
    ax.legend(ncols=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "overlay.png")
    fig.savefig(outdir / "overlay.svg")
    plt.close(fig)

    # Plot top-K features by total |shap|
    if shap_in_to_out.ndim == 2:
        scores = np.sum(np.abs(shap_in_to_out), axis=1)  # (I,)
        idxs = np.argsort(scores)[::-1][: min(topk, scores.size)]
        table = Table(title="Top-K features by |SHAP|", box=box.SIMPLE_HEAVY)
        table.add_column("Rank", justify="right")
        table.add_column("Feature Index", justify="right")
        table.add_column("|SHAP| Sum", justify="right")
        for r, i in enumerate(idxs, 1):
            table.add_row(str(r), str(int(i)), f"{scores[i]:.5g}")
        console.print(table)

        # Horizontal bar plot
        fig2, ax2 = plt.subplots(figsize=(8, 0.4 * len(idxs) + 1.5), dpi=dpi)
        ax2.barh(
            [str(i) for i in idxs[::-1]],
            scores[idxs][::-1],
            color="#9467bd",
            alpha=0.9,
        )
        ax2.set_xlabel("|SHAP| contribution (summed across wavelengths)")
        ax2.set_ylabel("Feature index")
        ax2.set_title(f"Top-{len(idxs)} features by |SHAP|")
        fig2.tight_layout()
        fig2.savefig(outdir / "topk_features.png")
        plt.close(fig2)

    # HTML interactive
    if html and _HAS_PLOTLY:
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=xs, y=y_plot, mode="lines", name="Predicted spectrum"))
        figp.add_trace(go.Scatter(x=xs, y=y_plot + pos, mode="lines", name="y + positive SHAP", line=dict(color="green")))
        figp.add_trace(go.Scatter(x=xs, y=y_plot + neg, mode="lines", name="y + negative SHAP", line=dict(color="red")))
        figp.update_layout(
            title=title or "SHAP Overlay",
            xaxis_title="Wavelength",
            yaxis_title="Transit depth / Relative flux",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        figp.write_html(str(outdir / "overlay.html"))


def export_csv(outdir: Path, wavelengths: Optional[np.ndarray], y_pred: np.ndarray, shap_in_to_out: np.ndarray):
    outdir.mkdir(parents=True, exist_ok=True)
    O = y_pred.shape[0]
    xs = wavelengths if wavelengths is not None and len(wavelengths) == O else np.arange(O)
    df = pd.DataFrame({"wavelength": xs, "y_pred": y_pred})
    if shap_in_to_out.ndim == 1:
        df["shap_sum"] = shap_in_to_out
    else:
        # sum across inputs for per-wavelength aggreg
        df["shap_sum"] = shap_in_to_out.sum(axis=0)
    df.to_csv(outdir / "overlay_data.csv", index=False)


# ---------------------------- CLI Commands --------------------------------------- #

@app.command()
def run(
    model_path: Path = typer.Option(..., help="Path to TorchScript (.pt/.pth) or state_dict."),
    input: Path = typer.Option(..., "--input", help="Path to input features (.npy/.csv/.parquet)."),
    output_dir: Path = typer.Option(Path("artifacts/shap_overlay"), help="Output directory."),
    factory: Optional[str] = typer.Option(None, help="Factory callable 'module.submodule:create_model' (if not TorchScript)."),
    factory_kwargs: Optional[str] = typer.Option(None, help="JSON dict of kwargs passed to factory."),
    spectral_axis: Optional[Path] = typer.Option(None, help="Optional 1D array of wavelength axis."),
    band_map: Optional[Path] = typer.Option(None, help="Optional YAML/JSON mapping of bands to indices or [start,end]."),
    index: Optional[int] = typer.Option(None, help="Single sample index to explain (default: batch explain)."),
    batch: int = typer.Option(32, help="Batch size for model forward (and batched SHAP if supported)."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)."),
    explainer: str = typer.Option("auto", help="SHAP explainer: auto|deep|kernel|gradient"),
    background: Optional[Path] = typer.Option(None, help="Background samples for Kernel/Deep (same format as --input)."),
    background_size: int = typer.Option(128, help="Number of background rows to use (if background not provided)."),
    topk: int = typer.Option(10, help="Top-K features to list in bar chart."),
    smooth: int = typer.Option(0, help="Savitzky-Golay window size; 0 to disable."),
    export_html: bool = typer.Option(True, help="Export interactive HTML (requires plotly)."),
    export_csv_flag: bool = typer.Option(True, "--export-csv", help="Export CSV with wavelengths,y_pred,shap_sum"),
    dpi: int = typer.Option(160, help="Matplotlib figure DPI."),
    seed: int = typer.Option(42, help="Random seed."),
    reshape: Optional[str] = typer.Option(None, help="e.g., 'B,F->B,1,F' to add a time/channel dim."),
    dtype: str = typer.Option("float32", help="Input dtype: float32/float64."),
):
    """
    Compute SHAP attributions and create spectral overlays.
    """
    t0 = time.time()
    set_seed(seed)

    # Build config
    cfg = OverlayConfig(
        model_path=model_path,
        factory=factory,
        factory_kwargs=json.loads(factory_kwargs) if factory_kwargs else None,
        input_path=input,
        spectral_axis=spectral_axis,
        band_map=band_map,
        output_dir=output_dir,
        index=index,
        batch=batch,
        device=device,
        explainer=explainer,
        background=background,
        background_size=background_size,
        topk=topk,
        smooth=smooth,
        export_html=export_html,
        export_csv=export_csv_flag,
        dpi=dpi,
        seed=seed,
        reshape=reshape,
        dtype=dtype,
    )

    console.rule("[ok]SHAP Overlay")
    console.print(f"[ok] Model: {cfg.model_path}\n[ok] Data : {cfg.input_path}\n[ok] Out  : {cfg.output_dir}")
    if cfg.factory:
        console.print(f"[ok] Factory: {cfg.factory}  kwargs={cfg.factory_kwargs}")

    # Load data
    X = _load_array_any(cfg.input_path).astype(cfg.dtype)
    if cfg.reshape:
        X = _maybe_reshape(X, cfg.reshape)

    # Sanity dims
    if X.ndim < 2:
        raise ValueError(f"Input X must be 2D or 3D (batch-first). Got shape {X.shape}")

    # Load wavelengths if provided
    wavelengths = None
    if cfg.spectral_axis and cfg.spectral_axis.exists():
        wavelengths = _load_array_any(cfg.spectral_axis).squeeze()

    # Load model
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), transient=True) as progress:
        task = progress.add_task("[cyan]Loading model...", total=None)
        model = load_model(cfg)
        progress.update(task, completed=1)

    # Probe output dimension
    sample0 = X[: min(2, len(X))]
    with torch.no_grad():
        y0 = forward_model(model, sample0, cfg.device)
    if y0.ndim == 1:
        y0 = y0[None, :]
    Bp, O = y0.shape
    console.print(f"[ok] Output dimension (spectrum points) = {O}")

    # Load/create background
    if cfg.background and cfg.background.exists():
        BG = _load_array_any(cfg.background).astype(cfg.dtype)
        if cfg.reshape:
            BG = _maybe_reshape(BG, cfg.reshape)
    else:
        # Random subset of X as background
        if len(X) < 2:
            BG = np.repeat(X, repeats=2, axis=0)
        else:
            nbg = min(cfg.background_size, len(X))
            sel = np.random.choice(len(X), size=nbg, replace=len(X) < nbg)
            BG = X[sel]

    # Build explainer
    explainer, mname = build_explainer(cfg, model, BG, sample0)
    console.print(f"[ok] Using explainer: [bold]{mname}[/bold]")

    # Select data to explain
    if cfg.index is not None:
        # Single sample
        sidx = int(cfg.index)
        if sidx < 0 or sidx >= len(X):
            raise IndexError(f"--index {sidx} out of range [0,{len(X)-1}]")
        Xexp = X[sidx : sidx + 1]
        y_pred = forward_model(model, Xexp, cfg.device)[0]  # (O,)
        # SHAP for one sample
        if mname in {"deep", "gradient"}:
            vals = explainer.shap_values(torch.from_numpy(Xexp))
        else:
            vals = explainer.shap_values(Xexp)
        # Harmonize to (1, I, O)
        vals_h = compute_shap_values(explainer, Xexp, O, mname, nsamples=None)[0]  # (I,O)
        # Overlay
        band_dict = _load_band_map(cfg.band_map, O) if cfg.band_map else None
        subdir = cfg.output_dir / f"sample_{sidx}"
        title = f"SHAP Overlay — sample {sidx}"
        plot_overlay(subdir, wavelengths, y_pred, vals_h, Xexp[0].reshape(-1), cfg.topk, cfg.dpi, cfg.smooth, band_dict, cfg.export_html, title)
        if cfg.export_csv:
            export_csv(subdir, wavelengths, y_pred, vals_h)
        console.print(f"[ok] Wrote overlays to: {subdir}")
    else:
        # Batch mode: explain a small batch for speed, then summarize
        Btake = min( min(64, len(X)), max(4, cfg.batch) )
        Xexp = X[:Btake]
        y_pred = forward_model(model, Xexp, cfg.device)  # (B,O)
        # compute shap
        vals_h = compute_shap_values(explainer, Xexp, O, mname, nsamples=None)  # (B,I,O)
        # Aggregate across batch: mean absolute contribution
        shap_mean = np.mean(np.abs(vals_h), axis=0)  # (I,O)
        y_mean = np.mean(y_pred, axis=0)            # (O,)
        band_dict = _load_band_map(cfg.band_map, O) if cfg.band_map else None
        subdir = cfg.output_dir / "batch_summary"
        title = f"SHAP Overlay — batch mean (N={Btake})"
        plot_overlay(subdir, wavelengths, y_mean, shap_mean, Xexp[0].reshape(-1), cfg.topk, cfg.dpi, cfg.smooth, band_dict, cfg.export_html, title)
        if cfg.export_csv:
            export_csv(subdir, wavelengths, y_mean, shap_mean)
        console.print(f"[ok] Wrote batch overlays to: {subdir}")

    # Metadata dump
    meta = {
        "model_path": str(cfg.model_path),
        "factory": cfg.factory,
        "factory_kwargs": cfg.factory_kwargs,
        "input": str(cfg.input_path),
        "spectral_axis": str(cfg.spectral_axis) if cfg.spectral_axis else None,
        "band_map": str(cfg.band_map) if cfg.band_map else None,
        "index": cfg.index,
        "explainer": mname,
        "device": cfg.device,
        "seed": cfg.seed,
        "version": "shap_overlay_v1.0",
        "runtime_sec": round(time.time() - t0, 3),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "overlay_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    console.print(f"[ok] Done in {meta['runtime_sec']}s")


@app.command()
def version():
    """Print version and environment info."""
    table = Table(title="shap_overlay environment", box=box.SIMPLE_HEAVY)
    table.add_column("Key"); table.add_column("Value")
    table.add_row("torch", torch.__version__)
    table.add_row("numpy", np.__version__)
    table.add_row("shap", getattr(shap, "__version__", "not installed") if _HAS_SHAP else "not installed")
    table.add_row("plotly", "installed" if _HAS_PLOTLY else "not installed")
    table.add_row("cuda_available", str(torch.cuda.is_available()))
    console.print(table)


if __name__ == "__main__":
    app()
