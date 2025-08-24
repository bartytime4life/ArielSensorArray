#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/diagnostics/spectral_shap_gradient.py

SpectraMind V50 — Spectral SHAP + Gradient Visualizer (Upgraded, Challenge‑Grade)
==================================================================================

Purpose
-------
Compute and visualize sensitivity of model outputs with respect to inputs for exoplanet spectroscopy:
• ∂μ/∂input (spectrum mean head)
• ∂σ/∂input (uncertainty head)
• ∂GLL/∂input (Gaussian Log-Likelihood wrt inputs, given targets)
Optionally overlay/import SHAP attributions to compare with gradient-based saliency.

Outputs
-------
• grads.npy                          (float32, shape [N, D]) per-sample gradient map magnitude (L2 or abs)
• grads_raw.npy                      (float32, shape [N, D]) signed gradient (for scalar objective)
• optional per-bin gradients (if requested) saved with suffix _bin{k}.npy
• JSON summary (grad_summary.json)
• Plots (PNG): heatmap, top-k bar, optional SHAP vs Grad comparison
• Optional HTML report (single-file) embedding figures
• Append-only audit entry to logs/v50_debug_log.md

Assumptions
-----------
• Model returns a dict or tuple with μ and σ (in that order) OR just μ.
• σ is either already positive or requires activation (exp/softplus/none controlled via flag).
• Inputs X are [N, D] numpy float arrays (flattened features already prepared).
• Optional targets Y are [N, B] when computing ∂GLL/∂input. μ/σ heads must output [N, B].
• Planet IDs (optional .txt/.csv first column) length N, used for plots/labels.

CLI Highlights
--------------
Typer-based CLI. Typical run:

  python -m src.diagnostics.spectral_shap_gradient run \
      --model-ts models/v50_model.ts \
      --inputs outputs/features/X.npy \
      --targets data/labels/Y.npy \
      --mode gll \
      --sigma-activation softplus \
      --outdir outputs/diag_grad \
      --html --open-html

Author
------
SpectraMind V50 — Architect & Master Programmer

License
-------
MIT
"""

from __future__ import annotations

import os
import io
import csv
import json
import math
import base64
import types
import importlib
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Torch / plotting
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# ======================================================================================
# Utilities
# ======================================================================================

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _append_audit(message: str, log_path: Path) -> None:
    """Append an immutable audit entry to logs/v50_debug_log.md (best-effort)."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{stamp}] spectral_shap_gradient: {message}\n")
    except Exception:
        pass

def _read_npy(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        console.print(f"[yellow]WARN[/] npy not found: {path}")
        return None
    return np.load(str(path))

def _read_planet_ids(path: Optional[Path], n: Optional[int]) -> Optional[List[str]]:
    if path is None:
        if n is None:
            return None
        return [f"planet_{i:04d}" for i in range(n)]
    if not path.exists():
        console.print(f"[yellow]WARN[/] planet-ids file not found: {path}")
        return None
    if path.suffix.lower() == ".txt":
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # CSV first column
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            out.append(str(row[0]).strip())
    return out

def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)

def _sigma_activation(raw_sigma: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Enforce positivity of σ depending on provided activation mode.
    mode ∈ {'exp','softplus','none'}.
    """
    if mode == "exp":
        return torch.exp(raw_sigma)
    if mode == "softplus":
        return F.softplus(raw_sigma)
    # 'none' – assume model already returns positive σ (or you accept negatives during debug)
    return raw_sigma

def _encode_png_to_base64(png_path: Path) -> str:
    b = png_path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

# ======================================================================================
# Model Loader
# ======================================================================================

class WrappedModel(nn.Module):
    """
    Thin wrapper that standardizes output as (mu, sigma or None).
    - support TorchScript model (single callable returning tuple/dict)
    - support Python class + state_dict load

    Expected forward output variants:
      • tuple: (mu, sigma) or (mu,) or (mu)  (mu shape [N,B], sigma shape [N,B])
      • dict:  keys like {'mu': ..., 'sigma': ...} (any case accepted)
      • tensor: just mu
    """
    def __init__(self, inner: nn.Module, sigma_key: Optional[str] = None, mu_key: Optional[str] = None):
        super().__init__()
        self.inner = inner
        self.mu_key = mu_key
        self.sigma_key = sigma_key

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.inner(x)
        mu = None
        sigma = None
        if isinstance(out, (list, tuple)):
            if len(out) >= 1:
                mu = out[0]
            if len(out) >= 2:
                sigma = out[1]
        elif isinstance(out, dict):
            # normalize keys
            keys = {k.lower(): k for k in out.keys()}
            k_mu = self.mu_key or ("mu" if "mu" in keys else next((k for k in keys if "mu" in k), None))
            k_sigma = self.sigma_key or ("sigma" if "sigma" in keys else next((k for k in keys if "sig" in k), None))
            if k_mu is not None:
                mu = out[keys[k_mu]]
            else:
                # first value fallback
                mu = list(out.values())[0]
            if k_sigma is not None:
                sigma = out[keys[k_sigma]]
        elif torch.is_tensor(out):
            mu = out
        else:
            raise TypeError("Unsupported model output type.")
        return mu, sigma

def load_torchscript(path: Path, device: torch.device, sigma_key: Optional[str], mu_key: Optional[str]) -> WrappedModel:
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return WrappedModel(model, sigma_key=sigma_key, mu_key=mu_key).to(device)

def load_python_model(
    module_path: str,
    class_name: str,
    state_dict_path: Optional[Path],
    device: torch.device,
    sigma_key: Optional[str],
    mu_key: Optional[str],
    init_kwargs: Optional[str] = None
) -> WrappedModel:
    """
    Dynamically import module and class, construct model (optionally with init kwargs JSON),
    optionally load state_dict, set to eval.
    """
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    kwargs = {}
    if init_kwargs:
        try:
            kwargs = json.loads(init_kwargs)
        except Exception as e:
            raise ValueError(f"Invalid JSON for init_kwargs: {e}")
    model = cls(**kwargs)
    if state_dict_path:
        sd = torch.load(str(state_dict_path), map_location="cpu")
        # allow both raw state_dict or {'state_dict': ...}
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
    model.eval()
    return WrappedModel(model, sigma_key=sigma_key, mu_key=mu_key).to(device)

# ======================================================================================
# Core Gradient Logic
# ======================================================================================

def gaussian_log_likelihood(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample, per-bin GLL (not reduced).
    GLL = -0.5 * [ log(2πσ^2) + (y - μ)^2 / σ^2 ]
    """
    sigma = torch.clamp(sigma, min=eps)
    resid2 = (y - mu) ** 2
    return -0.5 * (torch.log(2 * math.pi * sigma**2) + resid2 / (sigma**2))

def _reduce_scalar(obj: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce a tensor to a scalar objective.
    reduction ∈ {'mean','sum'}
    """
    if reduction == "sum":
        return obj.sum()
    return obj.mean()

@torch.no_grad()
def _infer_mu_sigma(model: WrappedModel, x: torch.Tensor, device: torch.device, sigma_act: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    model.eval()
    with torch.no_grad():
        mu, sigma = model(x.to(device))
        if sigma is not None:
            sigma = _sigma_activation(sigma, sigma_act)
    return mu, sigma

def compute_gradients(
    model: WrappedModel,
    x: torch.Tensor,             # [N, D], requires_grad will be set inside
    mode: str,                   # 'mu'|'sigma'|'gll'
    sigma_act: str,
    target: Optional[torch.Tensor],  # needed for 'gll' mode, shape [N,B]
    bin_index: Optional[int],    # focus on a specific output bin for mu/sigma; if None, reduce across bins
    reduction: str,              # 'mean' or 'sum'
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (grads_raw [N,D], grads_mag [N,D]) as numpy.
    The raw gradient is ∂objective/∂input.
    The magnitude is |raw| (abs) for visualization; can be L2 or abs — here we use abs per-feature.
    """
    model.eval()
    x = x.detach().to(device).requires_grad_(True)
    mu, sigma = model(x)

    # Handle sigma activation
    if sigma is not None:
        sigma = _sigma_activation(sigma, sigma_act)

    # Build scalar objective depending on mode
    if mode == "mu":
        if bin_index is not None:
            # objective = mean/sum of μ[:, bin_index] across batch
            obj = mu[:, bin_index]
        else:
            # reduce across bins, then across batch
            obj = mu  # [N,B]
        obj_scalar = _reduce_scalar(obj, reduction)
    elif mode == "sigma":
        if sigma is None:
            raise RuntimeError("Model did not produce σ, cannot compute ∂σ/∂input.")
        if bin_index is not None:
            obj = sigma[:, bin_index]
        else:
            obj = sigma
        obj_scalar = _reduce_scalar(obj, reduction)
    elif mode == "gll":
        if target is None:
            raise ValueError("targets are required to compute ∂GLL/∂input")
        if sigma is None:
            raise RuntimeError("Model did not produce σ required for GLL.")
        gll = gaussian_log_likelihood(mu, sigma, target.to(device))  # [N,B]
        obj_scalar = _reduce_scalar(gll, reduction)
    else:
        raise ValueError("mode must be one of {'mu','sigma','gll'}")

    # Backprop to inputs
    obj_scalar.backward()
    grads_raw = x.grad.detach().cpu().numpy().astype(np.float32)  # [N, D]
    # Use absolute value per feature for magnitude (saliency). L2 across features could also be used; we expose abs.
    grads_mag = np.abs(grads_raw).astype(np.float32)
    return grads_raw, grads_mag

# ======================================================================================
# Plotting
# ======================================================================================

def save_heatmap(matrix: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str, aspect: str = "auto") -> None:
    """
    Save a heatmap for a 2D matrix (e.g., [N,D]).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(min(16, max(6, matrix.shape[1] * 0.04)), min(10, max(4, matrix.shape[0] * 0.06))))
    im = ax.imshow(matrix, aspect=aspect, interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

def save_topk_bar(values: np.ndarray, feature_names: Optional[List[str]], out: Path, title: str, k: int = 20) -> None:
    """
    Save a horizontal bar chart for top-k features by magnitude for a single sample vector (length D).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    k = min(k, values.size)
    idx = np.argsort(-np.abs(values))[:k]
    vals = values[idx]
    names = [feature_names[i] if feature_names and i < len(feature_names) else f"f{i}" for i in idx]
    fig, ax = plt.subplots(figsize=(9, max(4, k * 0.35)))
    ax.barh(range(k), np.abs(vals))
    ax.set_yticks(range(k))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("|gradient|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

def save_scatter_compare(x: np.ndarray, y: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    """
    Save a scatter comparing two vectors (e.g., SHAP vs Grad per feature for a sample).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=10, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

def write_html_report(
    out_html: Path,
    title: str,
    key_stats: Dict[str, Any],
    image_paths: List[Tuple[str, Path]]
) -> None:
    """
    Render a minimal single-file HTML with embedded base64 PNGs and key/value stats.
    """
    out_html.parent.mkdir(parents=True, exist_ok=True)
    kv = "\n".join([f"<div>{k}</div><div>{v}</div>" for k, v in key_stats.items()])
    cards = []
    for caption, p in image_paths:
        if p.exists():
            data_uri = _encode_png_to_base64(p)
            cards.append(f"<div class='card'><h3>{caption}</h3><img src='{data_uri}'/></div>")
        else:
            cards.append(f"<div class='card'><h3>{caption}</h3><div class='muted'>Not generated</div></div>")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
:root{{--bg:#0b1220;--fg:#e6edf3;--muted:#8b96a8;--card:#121a2a;--border:#22304a;--accent:#7aa2ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Ubuntu}}
.container{{max-width:1200px;margin:0 auto;padding:20px}}
h1{{font-size:20px;margin:0 0 10px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px}}
.kv{{display:grid;grid-template-columns:200px 1fr;gap:6px;margin-bottom:12px}}
img{{width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:#0c1526}}
.muted{{color:var(--muted);font-size:12px}}
</style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div class="card">
      <h3>Run Stats</h3>
      <div class="kv">{kv}</div>
    </div>
    <div class="grid">
      {"".join(cards)}
    </div>
  </div>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")

# ======================================================================================
# CLI
# ======================================================================================

@app.command("run")
def cli_run(
    # ---- Model options
    model_ts: Optional[Path] = typer.Option(None, help="Path to TorchScript model (.pt / .ts)."),
    model_py: Optional[str] = typer.Option(None, help="Python module path for model class (e.g., 'src.models.model_v50')."),
    model_class: Optional[str] = typer.Option(None, help="Class name for model in module (e.g., 'V50Model')."),
    state_dict: Optional[Path] = typer.Option(None, help="Optional state_dict file (.pt/.pth) when using Python model."),
    model_init_kwargs: Optional[str] = typer.Option(None, help="JSON string of kwargs to pass to model constructor."),
    mu_key: Optional[str] = typer.Option(None, help="If model returns dict, name of μ key (case-insensitive)."),
    sigma_key: Optional[str] = typer.Option(None, help="If model returns dict, name of σ key (case-insensitive)."),

    # ---- Data
    inputs: Path = typer.Option(..., help="Path to X.npy (float, shape [N,D])."),
    targets: Optional[Path] = typer.Option(None, help="Path to Y.npy (float, shape [N,B]) for GLL mode."),
    planet_ids: Optional[Path] = typer.Option(None, help="Optional .txt or .csv with first column planet_id."),
    feature_names: Optional[Path] = typer.Option(None, help="Optional .txt with feature names, one per line (length D)."),

    # ---- Gradient mode
    mode: str = typer.Option(..., help="One of {'mu','sigma','gll'}"),
    sigma_activation: str = typer.Option("softplus", help="Sigma activation: 'softplus'|'exp'|'none'"),
    bin_index: Optional[int] = typer.Option(None, help="For mu/sigma modes, focus on a single output bin index; if omitted, reduce across bins."),
    reduction: str = typer.Option("mean", help="Scalar reduction across batch/bins: 'mean'|'sum'"),
    device_str: str = typer.Option("cuda:0", help="Torch device, e.g., 'cuda:0' or 'cpu'"),

    # ---- Visualization / output
    outdir: Path = typer.Option(Path("outputs/diag_gradients"), help="Output directory."),
    normalize_inputs: bool = typer.Option(False, "--normalize/--no-normalize", help="Zero-mean unit-std normalize inputs per feature before grad."),
    topk: int = typer.Option(20, help="Top-K features for per-sample bar chart."),
    sample_index: int = typer.Option(0, help="Sample index to visualize in top-K plots."),
    shap_values: Optional[Path] = typer.Option(None, help="Optional SHAP .npy with per-sample per-feature attributions [N,D]."),
    html: bool = typer.Option(False, "--html/--no-html", help="Write a lightweight HTML report embedding figures."),
    open_html: bool = typer.Option(False, "--open-html/--no-open-html", help="Open HTML report after writing."),

    # ---- Misc
    log_path: Path = typer.Option(Path("logs/v50_debug_log.md"), help="Append-only audit log."),
):
    """
    Compute ∂μ/∂input, ∂σ/∂input, or ∂GLL/∂input and generate visual diagnostics.

    Notes
    -----
    • If using GLL mode, you must provide --targets and model must output μ and σ with shape [N,B].
    • For mu/sigma mode without --bin-index, gradients collapse across bins (objective reduces across bins).
    • Gradients are computed for the scalar objective = {mean|sum} across batch (and bins if applicable).
      The raw gradient is ∂objective/∂input (signed). We save both raw and |raw| for diagnostics.
    """
    try:
        outdir = _ensure_dir(outdir)
        plots_dir = _ensure_dir(outdir / "plots")

        device = torch.device(device_str if torch.cuda.is_available() or "cpu" not in device_str else "cpu")

        # Load inputs / targets
        X = _read_npy(inputs)
        if X is None:
            raise FileNotFoundError(f"Missing inputs: {inputs}")
        if X.ndim != 2:
            raise ValueError("inputs must be [N, D]")
        N, D = X.shape

        Y = _read_npy(targets) if targets else None
        if mode == "gll" and Y is None:
            raise ValueError("targets are required in GLL mode.")
        if Y is not None and Y.shape[0] != N:
            raise ValueError("targets first dimension must match inputs N.")

        pids = _read_planet_ids(planet_ids, n=N)
        feat_names = None
        if feature_names and feature_names.exists():
            feat_names = [ln.strip() for ln in feature_names.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(feat_names) != D:
                console.print(f"[yellow]WARN[/] feature_names length {len(feat_names)} != D {D}; ignoring.")
                feat_names = None

        if sample_index < 0 or sample_index >= N:
            raise ValueError(f"sample_index must be in [0, {N-1}]")

        shap_np = _read_npy(shap_values) if shap_values else None
        if shap_np is not None and (shap_np.shape != (N, D)):
            console.print(f"[yellow]WARN[/] shap_values shape {shap_np.shape} != (N,D); ignoring.")
            shap_np = None

        # Optional input normalization for gradients (does NOT alter saved X)
        X_for_grad = X.copy()
        norm_stats = {}
        if normalize_inputs:
            mu_f = X_for_grad.mean(axis=0, keepdims=True)
            std_f = X_for_grad.std(axis=0, keepdims=True) + 1e-12
            X_for_grad = (X_for_grad - mu_f) / std_f
            norm_stats = {"normalize": True, "mean": mu_f.squeeze().tolist(), "std": std_f.squeeze().tolist()}
        else:
            norm_stats = {"normalize": False}

        # Build torch tensors
        x_t = torch.from_numpy(X_for_grad).float()
        y_t = torch.from_numpy(Y).float() if Y is not None else None

        # Load model
        if model_ts:
            model = load_torchscript(model_ts, device, sigma_key=sigma_key, mu_key=mu_key)
        elif model_py and model_class:
            model = load_python_model(
                module_path=model_py,
                class_name=model_class,
                state_dict_path=state_dict,
                device=device,
                sigma_key=sigma_key,
                mu_key=mu_key,
                init_kwargs=model_init_kwargs,
            )
        else:
            raise ValueError("Provide either --model-ts OR (--model-py and --model-class).")

        # Sanity forward (no grad) to infer output shapes
        mu_pred, sigma_pred = _infer_mu_sigma(model, x_t, device, sigma_activation)
        B_dim = mu_pred.shape[1] if mu_pred is not None and mu_pred.ndim == 2 else None
        if mode in ("mu", "sigma"):
            if bin_index is not None:
                if B_dim is None:
                    raise RuntimeError("Model output μ/σ is not [N,B]; cannot index bin. Either omit --bin-index or adjust model.")
                if bin_index < 0 or bin_index >= B_dim:
                    raise ValueError(f"--bin-index out of range [0,{B_dim-1}]")
        if mode == "gll":
            if mu_pred is None or sigma_pred is None:
                raise RuntimeError("Model must output (μ, σ) for GLL mode.")
            if y_t is None or y_t.shape[1] != mu_pred.shape[1]:
                raise RuntimeError("targets must have same B dimension as μ/σ.")

        console.print(Panel.fit(f"Computing gradients… mode={mode}, reduction={reduction}", style="cyan"))

        # Compute gradients
        grads_raw, grads_mag = compute_gradients(
            model=model,
            x=x_t,
            mode=mode.lower(),
            sigma_act=sigma_activation.lower(),
            target=y_t,
            bin_index=bin_index,
            reduction=reduction.lower(),
            device=device,
        )

        # Save arrays
        np.save(outdir / "grads_raw.npy", grads_raw)
        np.save(outdir / "grads.npy", grads_mag)

        # Heatmap across samples (|grad|)
        save_heatmap(
            grads_mag,
            plots_dir / "grad_heatmap.png",
            title=f"Gradient Magnitude Heatmap (mode={mode})",
            xlabel="feature index",
            ylabel="sample index",
            aspect="auto",
        )

        # Top-k bar for a selected sample
        save_topk_bar(
            grads_mag[sample_index, :],
            feat_names,
            plots_dir / "grad_topk_sample.png",
            title=f"Top-{topk} |grad| — sample {sample_index} ({(pids[sample_index] if pids else f'#{sample_index}')})",
            k=topk,
        )

        # SHAP comparison (optional)
        shap_scatter = None
        if shap_np is not None:
            # Scatter SHAP vs |grad| for sample_index
            shap_scatter = plots_dir / "shap_vs_grad_scatter.png"
            save_scatter_compare(
                x=shap_np[sample_index, :],
                y=grads_mag[sample_index, :],
                out=shap_scatter,
                title=f"SHAP vs |Grad| — sample {sample_index}",
                xlabel="SHAP",
                ylabel="|Grad|",
            )

        # Summary JSON
        summary = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "mode": mode,
            "sigma_activation": sigma_activation,
            "bin_index": bin_index,
            "reduction": reduction,
            "shapes": {"N": int(N), "D": int(D), "B": (int(B_dim) if B_dim is not None else None)},
            "normalize_inputs": norm_stats["normalize"],
            "paths": {
                "inputs": str(inputs),
                "targets": (str(targets) if targets else None),
                "grads_raw": str((outdir / "grads_raw.npy").as_posix()),
                "grads_mag": str((outdir / "grads.npy").as_posix()),
                "plots": {
                    "heatmap": str((plots_dir / "grad_heatmap.png").as_posix()),
                    "topk_sample": str((plots_dir / "grad_topk_sample.png").as_posix()),
                    "shap_vs_grad_scatter": (str(shap_scatter.as_posix()) if shap_scatter else None),
                },
            },
            "sample_index": int(sample_index),
            "planet_id": (pids[sample_index] if pids else None),
        }
        (outdir / "grad_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # HTML (optional)
        if html:
            images = [
                ("Gradient Heatmap", plots_dir / "grad_heatmap.png"),
                (f"Top-{topk} |Grad| — sample {sample_index}", plots_dir / "grad_topk_sample.png"),
            ]
            if shap_scatter:
                images.append(("SHAP vs |Grad| — sample", shap_scatter))

            key_stats = {
                "Mode": mode,
                "Sigma activation": sigma_activation,
                "Reduction": reduction,
                "Bin index": bin_index if bin_index is not None else "all (reduced)",
                "N (samples)": N,
                "D (features)": D,
                "B (bins)": (B_dim if B_dim is not None else "N/A"),
                "Inputs": inputs.name,
                "Targets": targets.name if targets else "N/A",
                "Sample shown": f"{sample_index} ({pids[sample_index] if pids else 'N/A'})",
            }
            html_path = outdir / "report.html"
            write_html_report(html_path, f"SpectraMind V50 — {mode} Gradient Report", key_stats, images)
            if open_html:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{html_path.resolve().as_posix()}")
                except Exception:
                    pass

        # Console summary
        table = Table(title="SpectraMind V50 — Gradient Summary", show_header=True, header_style="bold")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Mode", mode)
        table.add_row("Sigma activation", sigma_activation)
        table.add_row("Reduction", reduction)
        table.add_row("Bin index", str(bin_index) if bin_index is not None else "all (reduced)")
        table.add_row("N,D,B", f"{N},{D},{B_dim if B_dim is not None else 'N/A'}")
        table.add_row("Inputs", inputs.name)
        table.add_row("Targets", targets.name if targets else "N/A")
        console.print(table)

        # Audit
        _append_audit(
            f"outdir={outdir.as_posix()} mode={mode} N={N} D={D} "
            f"B={(B_dim if B_dim is not None else 'NA')} normalize={normalize_inputs}",
            log_path=log_path
        )

        console.print(Panel.fit(f"Done. Artifacts: {outdir}", style="green"))

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted[/]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(Panel.fit(f"ERROR: {e}", style="red"))
        raise typer.Exit(code=1)

@app.callback()
def _cb():
    """
    SpectraMind V50 — Spectral SHAP + Gradient Visualizer

    Modes:
      • mu     : ∂μ/∂input at a specified bin (--bin-index) or reduced over bins (default).
      • sigma  : ∂σ/∂input at a specified bin (--bin-index) or reduced over bins (default).
      • gll    : ∂GLL/∂input, requires --targets and model to output both μ and σ.
                 GLL per bin = -0.5 * [ log(2πσ^2) + (y - μ)^2 / σ^2 ]; objective reduced via --reduction.

    Outputs:
      grads_raw.npy (signed ∂objective/∂input), grads.npy (|raw|), heatmap + bar plots, JSON, optional HTML.

    Notes:
      • If feature_names are provided, top-k bar labels reflect those names; else 'f{i}'.
      • If shap_values [N,D] are provided, a scatter compares SHAP vs |Grad| for one sample.
      • For stability, σ is clamped via activation ('softplus' default) to enforce positivity in GLL.
    """
    pass

if __name__ == "__main__":
    app()
