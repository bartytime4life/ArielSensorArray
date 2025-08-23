#!/usr/bin/env python3
"""
shap_overlay.py — SpectraMind V50
---------------------------------
Overlay SHAP attributions on predicted spectra (and optional uncertainty bands),
with a clean, CLI-first workflow and robust model/data auto-detection.

Features
- Loads models from: PyTorch (.pt/.pth), scikit-learn/lightgbm/xgboost pickles, or ONNX.
- Auto-picks a SHAP explainer: TreeExplainer, LinearExplainer, DeepExplainer, or KernelExplainer fallback.
- Accepts features from CSV/Parquet/NPY/NPZ; selects columns by regex or a YAML list.
- Plots: spectrum line; optional ±k·sigma band; SHAP contributions as a signed colormap “ribbon”; saves PNG & HTML.
- Saves raw SHAP arrays to .npy for downstream use; emits a JSONL event log for reproducibility.
- Rich, friendly CLI via Typer; verbose/debug logging; deterministic seeds.

Usage
------
$ python shap_overlay.py \
    --model models/v50_spectrum_regressor.pt \
    --features data/X_eval.parquet \
    --predictions data/preds_eval.npz \
    --id-col planet_id \
    --sample-id WASP-39b \
    --wavelengths data/wavelengths.npy \
    --select-cols "^wl_" \
    --outdir artifacts/shap/ \
    --sigma-k 2.0 \
    --smoothing 5 \
    --background 256 \
    --fig-width 1200 \
    --theme dark \
    --save-html

If you don’t have precomputed predictions, the script will call model(X) to obtain μ (and, if your
model returns it, σ). If σ isn’t available it’ll only plot μ + SHAP.

Notes
- For neural nets, SHAP KernelExplainer can be slow; use --background to subsample background rows.
- If your feature dimension is exactly the number of wavelengths (e.g., 283), the x-axis will use
  provided wavelengths; else, an index (0..d-1) axis is used.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
import types
import pickle
import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Optional deps are lazily imported where used
import typer
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.traceback import install as rich_install

# Matplotlib is used for static figures (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plotly (optional) for interactive HTML output
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# SHAP
import shap

# Torch optional
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# LightGBM/XGBoost optional for TreeExplainer
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


APP = typer.Typer(add_completion=False, no_args_is_help=True)

THEME = Theme({
    "good": "bold green",
    "warn": "bold yellow",
    "bad": "bold red",
    "info": "cyan",
    "head": "bold",
})
console = Console(theme=THEME)
rich_install(show_locals=False)


# ----------------------------- Utilities ------------------------------------ #

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore


# ----------------------------- Data loading --------------------------------- #

def load_array_or_frame(path: str) -> T.Tuple[np.ndarray | pd.DataFrame, str]:
    """
    Load a 2D matrix-like dataset:
    - .parquet/.pq -> DataFrame
    - .csv/.tsv -> DataFrame
    - .npy -> ndarray
    - .npz -> ndarray (first array found or 'X')
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return df, "df"
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        return df, "df"
    if ext == ".npy":
        arr = np.load(path)
        return arr, "nd"
    if ext == ".npz":
        z = np.load(path)
        if "X" in z:
            return z["X"], "nd"
        # else take the first
        first_key = list(z.keys())[0]
        return z[first_key], "nd"
    raise ValueError(f"Unsupported features file: {path}")


def extract_matrix(
    source: np.ndarray | pd.DataFrame,
    select_cols: str | None = None,
    feature_list: str | None = None,
) -> T.Tuple[np.ndarray, T.Optional[T.List[str]]]:
    """
    Return (X, colnames). If original is ndarray, colnames=None.
    select_cols: regex to pick columns if source is DataFrame
    feature_list: path to text/yaml listing columns to keep (one per line or YAML list)
    """
    if isinstance(source, np.ndarray):
        return source, None

    df = source.copy()
    cols = list(df.columns)

    if feature_list:
        # Accept YAML or plain text list
        try:
            import yaml  # type: ignore
            with open(feature_list, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
            wanted = spec if isinstance(spec, list) else []
        except Exception:
            with open(feature_list, "r", encoding="utf-8") as f:
                wanted = [ln.strip() for ln in f if ln.strip()]
        df = df[wanted]
        return df.values.astype(float), list(df.columns)

    if select_cols:
        pat = re.compile(select_cols)
        keep = [c for c in cols if pat.search(c)]
        if not keep:
            raise ValueError(f"No columns matched regex: {select_cols}")
        df = df[keep]
        return df.values.astype(float), keep

    # no selection → numeric columns only
    df = df.select_dtypes(include=[np.number])
    return df.values.astype(float), list(df.columns)


def load_predictions(path: str | None) -> T.Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load predictions if provided:
    - .npz with 'mu' and optional 'sigma'
    - .npy with shape (..., d) (assume μ only)
    - .csv/parquet: columns 'mu_*' and optionally 'sigma_*'
    """
    if path is None:
        return None, None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path)
        mu = z["mu"]
        sigma = z["sigma"] if "sigma" in z else None
        return mu, sigma
    if ext == ".npy":
        mu = np.load(path)
        return mu, None
    if ext in (".csv", ".tsv", ".parquet", ".pq"):
        if ext in (".csv", ".tsv"):
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(path, sep=sep)
        else:
            df = pd.read_parquet(path)
        mu_cols = [c for c in df.columns if c.startswith("mu_")]
        if not mu_cols:
            raise ValueError("No columns starting with 'mu_' found in predictions file.")
        mu = df[mu_cols].values
        sigma_cols = [c for c in df.columns if c.startswith("sigma_")]
        sigma = df[sigma_cols].values if sigma_cols else None
        return mu, sigma
    raise ValueError(f"Unsupported predictions file: {path}")


def load_wavelengths(wavelengths: str | None, d: int) -> np.ndarray:
    if wavelengths is None:
        return np.arange(d, dtype=float)
    w_ext = os.path.splitext(wavelengths)[1].lower()
    if w_ext == ".npy":
        w = np.load(wavelengths).astype(float).reshape(-1)
        if w.shape[0] != d:
            raise ValueError(f"Wavelengths length {w.shape[0]} != feature dim {d}")
        return w
    if w_ext in (".csv", ".tsv"):
        sep = "," if w_ext == ".csv" else "\t"
        w = pd.read_csv(wavelengths, sep=sep).iloc[:, 0].values.astype(float)
        if w.shape[0] != d:
            raise ValueError(f"Wavelengths length {w.shape[0]} != feature dim {d}")
        return w
    raise ValueError(f"Unsupported wavelengths file: {wavelengths}")


# ----------------------------- Model loading -------------------------------- #

@dataclass
class ModelAdapter:
    model: T.Any
    kind: str  # 'torch' | 'sklearn' | 'xgb' | 'lgb' | 'onnx' | 'callable'
    device: str = "cpu"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "torch":
            self.model.eval()
            with torch.no_grad():
                xt = torch.from_numpy(X.astype(np.float32))
                if self.device == "cuda":
                    xt = xt.cuda()
                out = self.model(xt)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                mu = out.detach().cpu().numpy()
            return mu
        elif self.kind in {"sklearn", "xgb", "lgb"}:
            return self.model.predict(X)
        elif self.kind == "onnx":
            import onnxruntime as ort  # type: ignore
            sess: ort.InferenceSession = self.model
            inp = {sess.get_inputs()[0].name: X.astype(np.float32)}
            mu = sess.run(None, inp)[0]
            return mu
        elif self.kind == "callable":
            return self.model(X)
        else:
            raise RuntimeError(f"Unknown model kind: {self.kind}")


def load_model(path: str | None, device: str = "cpu") -> ModelAdapter | None:
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    # Torch
    if _HAS_TORCH and ext in (".pt", ".pth", ".torch"):
        obj = torch.load(path, map_location="cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        # Either a torch.nn.Module or a state_dict with a builder in the file
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, dict) and "state_dict" in obj and "builder" in obj:
            builder = obj["builder"]
            if isinstance(builder, types.FunctionType):
                model = builder()
                model.load_state_dict(obj["state_dict"])
            else:
                raise ValueError("Torch checkpoint missing callable 'builder'.")
        else:
            raise ValueError("Unrecognized torch checkpoint format. Provide a Module or dict with state_dict+builder.")
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        return ModelAdapter(model=model, kind="torch", device=("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"))
    # ONNX
    if ext == ".onnx":
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return ModelAdapter(model=sess, kind="onnx", device=device)
    # Pickle for sklearn/xgb/lgb
    with open(path, "rb") as f:
        obj = pickle.load(f)
    kind = "sklearn"
    if xgb is not None and isinstance(obj, xgb.XGBModel):
        kind = "xgb"
    if lgb is not None and isinstance(obj, lgb.Booster):
        kind = "lgb"
    return ModelAdapter(model=obj, kind=kind, device=device)


# ----------------------------- SHAP explainers ------------------------------- #

def pick_explainer(adapter: ModelAdapter | None, X_bg: np.ndarray):
    """
    Choose an appropriate SHAP explainer.
    Fallback order: TreeExplainer → LinearExplainer → DeepExplainer → KernelExplainer.
    """
    m = adapter.model if adapter is not None else None
    # Try tree-based
    try:
        if (xgb is not None and isinstance(m, xgb.XGBModel)) or (lgb is not None and isinstance(m, lgb.Booster)):
            return shap.TreeExplainer(m)
    except Exception:
        pass
    # Try linear
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
        if hasattr(m, "coef_") or isinstance(m, LinearRegression):
            return shap.LinearExplainer(m, X_bg, feature_dependence="correlation")
    except Exception:
        pass
    # Try deep
    try:
        if _HAS_TORCH and isinstance(m, torch.nn.Module):
            return shap.DeepExplainer(m, torch.from_numpy(X_bg.astype(np.float32)))
    except Exception:
        pass
    # Kernel fallback
    return shap.KernelExplainer(m.predict if adapter is not None else (lambda x: x), X_bg)


def compute_shap(
    adapter: ModelAdapter | None,
    X: np.ndarray,
    background: int = 256,
    seed: int = 42
) -> np.ndarray:
    """
    Compute SHAP values with an automatic explainer.
    Returns shap_vals with shape (n_samples, n_features) or (n_samples, n_outputs, n_features).
    """
    _seed_everything(seed)
    n = X.shape[0]
    if background > 0 and background < n:
        idx = np.random.default_rng(seed).choice(n, size=background, replace=False)
        X_bg = X[idx]
    else:
        X_bg = X

    explainer = pick_explainer(adapter, X_bg)

    # For models that output vectors, SHAP may return list per output; we standardize.
    if isinstance(explainer, shap.KernelExplainer):
        # KernelExplainer needs small batch size for speed; we can vectorize by chunks.
        # But for simplicity here, we'll explain all at once (caller can reduce n via sample-id).
        shap_vals = explainer.shap_values(X, l1_reg="num_features(10)")
    else:
        shap_vals = explainer.shap_values(X)

    # Standardize to ndarray of shape (n, d) or (n, o, d)
    if isinstance(shap_vals, list):
        # list of outputs → (o, n, d) → (n, o, d)
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 3:
            shap_vals = np.transpose(shap_vals, (1, 0, 2))
    else:
        shap_vals = np.asarray(shap_vals)

    return shap_vals


# ----------------------------- Plotting ------------------------------------- #

def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return y
    k = max(1, int(k))
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(ypad, ker, mode="valid")


def plot_overlay_static(
    out_png: str,
    wavelengths: np.ndarray,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    shap_row: np.ndarray,
    sigma_k: float = 2.0,
    smoothing: int = 1,
    title: str = "",
    figsize=(13, 6),
    theme: str = "dark",
) -> None:
    plt.close("all")
    plt.style.use("dark_background" if theme.lower() == "dark" else "default")
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    x = wavelengths
    # Plot mean spectrum if present
    if mu is not None:
        y_mu = _smooth(mu, smoothing) if smoothing > 1 else mu
        ax1.plot(x, y_mu, color="#1f77b4", lw=2.0, label="μ (predicted)")
        # Uncertainty band
        if sigma is not None:
            y_lo = y_mu - sigma_k * sigma
            y_hi = y_mu + sigma_k * sigma
            ax1.fill_between(x, y_lo, y_hi, color="#1f77b4", alpha=0.15, label=f"±{sigma_k}·σ")

    # SHAP ribbon: normalize for colormap impact
    shp = shap_row.astype(float)
    shp_s = _smooth(shp, smoothing) if smoothing > 1 else shp
    vmax = np.percentile(np.abs(shp_s), 99) + 1e-12
    colors = plt.cm.seismic((shp_s / (2 * vmax) + 0.5).clip(0, 1))
    ax1.scatter(x, (ax1.get_ylim()[0] + ax1.get_ylim()[1]) / 2 * np.ones_like(x),
                c=colors, s=12, marker="s", alpha=0.9, label="SHAP (signed)")

    ax1.set_xlabel("Wavelength" if np.any(wavelengths != np.arange(len(wavelengths))) else "Feature index")
    ax1.set_ylabel("Flux / normalized units")
    if title:
        ax1.set_title(title)
    ax1.grid(True, ls=":", alpha=0.25)
    # Create a legend without duplicate handles
    handles, labels = ax1.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax1.legend(uniq.values(), uniq.keys(), loc="best")

    _mkdir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_html(
    out_html: str,
    wavelengths: np.ndarray,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    shap_row: np.ndarray,
    sigma_k: float = 2.0,
    smoothing: int = 1,
    title: str = "",
) -> None:
    if not _HAS_PLOTLY:
        raise RuntimeError("Plotly not installed; install plotly to use --save-html")
    x = wavelengths
    shp = shap_row.astype(float)
    shp_s = _smooth(shp, smoothing) if smoothing > 1 else shp

    fig = go.Figure()

    if mu is not None:
        y_mu = _smooth(mu, smoothing) if smoothing > 1 else mu
        fig.add_trace(go.Scatter(x=x, y=y_mu, name="μ (predicted)", mode="lines",
                                 line=dict(color="#1f77b4", width=2)))
        if sigma is not None:
            y_lo = y_mu - sigma_k * sigma
            y_hi = y_mu + sigma_k * sigma
            fig.add_trace(go.Scatter(x=x, y=y_hi, name=f"+{sigma_k}·σ", mode="lines",
                                     line=dict(color="rgba(31,119,180,0.2)")))
            fig.add_trace(go.Scatter(x=x, y=y_lo, name=f"-{sigma_k}·σ", mode="lines",
                                     line=dict(color="rgba(31,119,180,0.2)"),
                                     fill="tonexty", fillcolor="rgba(31,119,180,0.15)"))

    vmax = np.percentile(np.abs(shp_s), 99) + 1e-12
    # Color scale from blue (neg) → red (pos)
    cmap = [[0.0, "rgb(33,102,172)"], [0.5, "rgb(247,247,247)"], [1.0, "rgb(178,24,43)"]]
    fig.add_trace(go.Scatter(x=x, y=[0]*len(x), mode="markers", name="SHAP",
                             marker=dict(color=(shp_s / (2*vmax) + 0.5).clip(0,1),
                                         colorscale=cmap, size=6, symbol="square")))

    fig.update_layout(template="plotly_dark", title=title, xaxis_title="Wavelength", yaxis_title="Flux")
    _mkdir(os.path.dirname(out_html))
    fig.write_html(out_html, include_plotlyjs="cdn")


# ----------------------------- CLI command ---------------------------------- #

@APP.command("overlay")
def overlay(
    model: str = typer.Option(None, "--model", help="Path to model (.pt/.pth, .pkl, .onnx). If omitted, requires --predictions."),
    features: str = typer.Option(..., "--features", help="Features matrix: CSV/TSV/Parquet/NPY/NPZ."),
    predictions: str = typer.Option(None, "--predictions", help="Optional predictions file (.npz with 'mu' and optional 'sigma', or CSV with mu_* columns)."),
    id_col: str = typer.Option(None, "--id-col", help="ID column name (if features is a DataFrame) to select a sample by --sample-id."),
    sample_id: str = typer.Option(None, "--sample-id", help="Row identifier to visualize (works with --id-col). If omitted, will use the first row."),
    row_index: int = typer.Option(None, "--row-index", help="Alternative to --sample-id: zero-based row index to visualize."),
    wavelengths: str = typer.Option(None, "--wavelengths", help="Optional wavelengths vector .npy/.csv; otherwise 0..d-1."),
    select_cols: str = typer.Option(None, "--select-cols", help="Regex to select feature columns if features is a DataFrame."),
    feature_list: str = typer.Option(None, "--feature-list", help="Text/YAML list of feature columns to keep."),
    outdir: str = typer.Option("artifacts/shap", "--outdir", help="Output directory."),
    outbase: str = typer.Option(None, "--outbase", help="Basename for outputs (default auto-generated)."),
    sigma_k: float = typer.Option(2.0, "--sigma-k", help="Uncertainty band ±k·σ."),
    smoothing: int = typer.Option(1, "--smoothing", help="Rolling window for smoothing (1=off)."),
    background: int = typer.Option(256, "--background", help="Background size for SHAP KernelExplainer."),
    device: str = typer.Option("cpu", "--device", help="cuda or cpu (if torch model)."),
    save_html: bool = typer.Option(False, "--save-html", help="Also save interactive HTML (requires plotly)."),
    fig_width: int = typer.Option(1200, "--fig-width", help="PNG width in pixels (height auto)."),
    theme: str = typer.Option("dark", "--theme", help="dark or light theme for PNG."),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logs."),
    debug: bool = typer.Option(False, "--debug", help="Debug logs."),
):
    """
    Compute SHAP on one sample and overlay attributions with the predicted spectrum (and optional σ band).
    """
    console.rule("[head]SHAP Overlay[/head]")
    _seed_everything(seed)

    # Logging setup
    level = "DEBUG" if debug else ("VERBOSE" if verbose else "INFO")
    console.print(f"[info]Log level:[/info] {level}")

    # Load features
    src, kind = load_array_or_frame(features)
    df = None
    if kind == "df":
        df = T.cast(pd.DataFrame, src)
        # Keep a copy to resolve id lookup before col filtering
        id_series = df[id_col] if (id_col and id_col in df.columns) else None
        X, colnames = extract_matrix(df, select_cols=select_cols, feature_list=feature_list)
    else:
        X = T.cast(np.ndarray, src)
        colnames = None
        id_series = None

    n, d = X.shape
    console.print(f"[info]Loaded features:[/info] shape = {n} x {d}")

    # Row selection
    row = 0
    if id_series is not None and sample_id is not None:
        matches = np.where(id_series.values == sample_id)[0]
        if len(matches) == 0:
            raise ValueError(f"sample-id '{sample_id}' not found in column '{id_col}'")
        row = int(matches[0])
    elif row_index is not None:
        row = int(row_index)
        if row < 0 or row >= n:
            raise ValueError(f"--row-index out of range (0..{n-1})")
    title_id = sample_id if sample_id is not None else (str(row) if id_series is None else str(id_series.values[row]))
    console.print(f"[info]Selected row:[/info] {row} (id: {title_id})")

    # Load or make predictions
    mu, sigma = load_predictions(predictions)
    adapter = None
    if mu is None and model is not None:
        adapter = load_model(model, device=device)
        console.print(f"[info]Model loaded:[/info] kind={adapter.kind if adapter else 'None'}")
        mu = adapter.predict(X)
        # If model returns (mu, sigma)
        if isinstance(mu, (list, tuple)):
            mu, sigma = mu  # type: ignore
            mu = np.asarray(mu)
            if sigma is not None:
                sigma = np.asarray(sigma)
    if mu is not None:
        console.print(f"[info]Predictions provided:[/info] mu shape={mu.shape} sigma={'None' if sigma is None else sigma.shape}")
        # If predictions include many targets, ensure 1D for a single spectrum
        if mu.ndim == 2 and mu.shape[0] == n:
            mu_row = mu[row]
        elif mu.ndim == 1 and mu.shape[0] == d:
            mu_row = mu
        else:
            raise ValueError(f"Unexpected mu shape {mu.shape}; expected (n,d) or (d,)")
        if sigma is not None:
            if sigma.ndim == 2 and sigma.shape[0] == n:
                sigma_row = sigma[row]
            elif sigma.ndim == 1 and sigma.shape[0] == d:
                sigma_row = sigma
            else:
                raise ValueError(f"Unexpected sigma shape {sigma.shape}; expected (n,d) or (d,)")
        else:
            sigma_row = None
    else:
        mu_row, sigma_row = None, None
        console.print("[warn]No predictions available. Will plot SHAP only.[/warn]")

    # Wavelength axis
    wl = load_wavelengths(wavelengths, d)

    # SHAP computation on selected sample (and small background)
    # To ensure the background is representative, we use a random subset (or all data if small).
    # KernelExplainer cost scales with #features * #background * #samples; tune --background accordingly.
    console.print(f"[info]Computing SHAP with background={background}...[/info]")
    adapter_for_shap = adapter if adapter is not None else None
    shap_vals = compute_shap(adapter_for_shap, X, background=background, seed=seed)

    # Extract row
    if shap_vals.ndim == 3:
        # (n, outputs, d) → expect outputs==d (multioutput regression mapping to wavelengths) or pick first output
        if shap_vals.shape[1] == 1:
            shap_row = shap_vals[row, 0, :]
        else:
            # If the model emits multiple outputs but you want a single spectrum column,
            # here we take the first output; customize as needed.
            shap_row = shap_vals[row, 0, :]
            console.print("[warn]Multi-output SHAP detected; using output[0] for overlay.[/warn]")
    elif shap_vals.ndim == 2:
        shap_row = shap_vals[row, :]
    else:
        raise RuntimeError(f"Unexpected SHAP shape {shap_vals.shape}")

    # Outputs
    run_id = outbase or f"shap_{title_id}_{uuid.uuid4().hex[:8]}"
    outdir = os.path.join(outdir, str(title_id))
    _mkdir(outdir)
    out_png = os.path.join(outdir, f"{run_id}.png")
    out_html = os.path.join(outdir, f"{run_id}.html")
    out_npy = os.path.join(outdir, f"{run_id}_shap.npy")
    out_evt = os.path.join(outdir, f"{run_id}_events.jsonl")

    # Save SHAP raw row
    np.save(out_npy, shap_row)

    # Plot PNG
    # Set PNG size from fig_width with 16:9-ish ratio
    dpi = 100
    w_in = fig_width / dpi
    h_in = max(6.0, w_in * 0.5)
    title = f"SHAP Overlay — id={title_id}"
    plot_overlay_static(
        out_png=out_png,
        wavelengths=wl,
        mu=(mu_row if mu is not None else None),
        sigma=(sigma_row if sigma is not None else None),
        shap_row=shap_row,
        sigma_k=sigma_k,
        smoothing=smoothing,
        title=title,
        figsize=(w_in, h_in),
        theme=theme,
    )
    console.print(f"[good]Saved PNG:[/good] {out_png}")

    # Optional HTML
    if save_html:
        if not _HAS_PLOTLY:
            console.print("[warn]plotly not installed; skipping HTML export[/warn]")
        else:
            plot_overlay_html(
                out_html=out_html,
                wavelengths=wl,
                mu=(mu_row if mu is not None else None),
                sigma=(sigma_row if sigma is not None else None),
                shap_row=shap_row,
                sigma_k=sigma_k,
                smoothing=smoothing,
                title=title,
            )
            console.print(f"[good]Saved HTML:[/good] {out_html}")

    # Event log for reproducibility
    evt = {
        "ts": _now_iso(),
        "cmd": "overlay",
        "run_id": run_id,
        "features": os.path.abspath(features),
        "model": (os.path.abspath(model) if model else None),
        "predictions": (os.path.abspath(predictions) if predictions else None),
        "row_index": row,
        "sample_id": title_id,
        "n": int(n), "d": int(d),
        "sigma_k": float(sigma_k),
        "smoothing": int(smoothing),
        "background": int(background),
        "device": device,
        "outputs": {
            "png": os.path.abspath(out_png),
            "html": (os.path.abspath(out_html) if (save_html and _HAS_PLOTLY) else None),
            "shap_npy": os.path.abspath(out_npy),
        },
        "seed": seed,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "shap": shap.__version__,
            "torch": (torch.__version__ if _HAS_TORCH else None),
        }
    }
    _save_jsonl(out_evt, evt)
    console.print(f"[info]Event log:[/info] {out_evt}")
    console.rule("[head]Done[/head]")


# ----------------------------- Entrypoint ----------------------------------- #

def main():
    APP()


if __name__ == "__main__":
    main()
