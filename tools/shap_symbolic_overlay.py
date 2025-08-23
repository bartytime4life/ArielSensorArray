#!/usr/bin/env python3
"""
shap_symbolic_overlay.py

Neuro‑symbolic SHAP overlay for SpectraMind V50:
- computes feature attributions (SHAP) for the spectrum head of a trained model,
- computes per‑wavelength symbolic rule pressures (constraint-violation magnitudes),
- produces aligned heatmaps and a fused overlay to diagnose why the model predicts what it does
  and whether it does so in a physically consistent way.

Design notes (refs):
- CLI‑first Typer + Hydra config workflow; structured logs/artifacts for reproducibility. :contentReference[oaicite:0]{index=0} 
- “UI‑light” rich console and saved plots/HTML, no heavy GUI. :contentReference[oaicite:1]{index=1} 
- Neuro‑symbolic loss with physics rules (non‑negativity, bounded flux, correlated features). :contentReference[oaicite:2]{index=2} 
- GNN spectral priors and molecular-band coherence → we re-use band groupings for a “coherence” rule. :contentReference[oaicite:3]{index=3} 
- FFT/UMAP/diagnostics are part of the toolbox; this module focuses on SHAP + symbolic overlays. :contentReference[oaicite:4]{index=4}

"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


# -----------------------------
# Utility / Config structures
# -----------------------------

@dataclass
class OverlayConfig:
    # Paths
    model_path: str
    data_path: str
    output_dir: str = "outputs/diagnostics/shap_symbolic"
    # Selection
    n_samples: int = 64
    seed: int = 42
    # Spectrum specifics
    n_wavelengths: int = 283
    wl_start: float = 0.5  # microns (example)
    wl_end: float = 7.8    # microns (example)
    # Rules
    max_flux: float = 1.0          # upper bound (relative to stellar continuum)
    smooth_lambda: float = 1.0      # weight for smoothness rule
    nonneg_lambda: float = 1.0      # weight for non-negativity rule
    bound_lambda: float = 1.0       # weight for upper bound rule
    bandcoh_lambda: float = 1.0     # weight for molecular band coherence rule
    # Coherence bands: list of (name, [index ranges])
    bands: Dict[str, List[Tuple[int, int]]] = None
    # SHAP
    shap_background: int = 64
    shap_max_evals: int = 1024
    shap_method: str = "auto"  # "deep", "kernel", "auto"
    device: str = "auto"       # "auto","cpu","cuda"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Data / Model loading
# -----------------------------

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Expects a torch scripted or state-dict saved model that outputs
    either:
      - spectrum (B, 283)
      - or (mu, log_sigma) pair as a dict with keys ['mu','log_sigma']
    """
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        model.to(device).eval()
        return model

    # Fallback: state dict under 'state_dict' or top-level
    # A minimal generic MLP head definition for restoration if class is not present
    class SimpleHead(torch.nn.Module):
        def __init__(self, n_in: int, n_out: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_in, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, n_out),
            )

        def forward(self, x):
            return self.net(x)

    state_dict = ckpt.get("state_dict", ckpt)
    # Infer shapes crudely
    # NOTE: users should prefer saving scripted models; this is a convenience.
    example_in = state_dict.get("net.0.weight", None)
    if example_in is None:
        raise RuntimeError("Cannot infer model architecture; provide a scripted model or known class.")
    n_in = example_in.shape[1]
    # Try detecting mu/log_sigma twin heads
    has_mu = any(k.startswith("mu_head") for k in state_dict.keys())
    has_logsig = any(k.startswith("logsig_head") for k in state_dict.keys())

    if has_mu and has_logsig:
        class TwinHead(torch.nn.Module):
            def __init__(self, n_in: int, n_out: int):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(n_in, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(),
                )
                self.mu_head = torch.nn.Linear(512, n_out)
                self.logsig_head = torch.nn.Linear(512, n_out)

            def forward(self, x):
                h = self.backbone(x)
                return {"mu": self.mu_head(h), "log_sigma": self.logsig_head(h)}

        model = TwinHead(n_in, 283)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model

    model = SimpleHead(n_in, 283)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def load_dataset_npz(data_path: str, n_samples: int, n_wavelengths: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an .npz with keys:
      X : features for the spectrum head (B, F)   (e.g., engineered features or latent repr)
      Y : target spectrum (B, n_wavelengths)      (optional; used only for overlay CSV)
    """
    with np.load(data_path) as d:
        X = d["X"]
        Y = d.get("Y", None)
    if n_samples > 0 and X.shape[0] > n_samples:
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[idx]
        if Y is not None:
            Y = Y[idx]
    if Y is None:
        Y = np.zeros((X.shape[0], n_wavelengths), dtype=np.float32)
    return X, Y


# -----------------------------
# Symbolic rules
# -----------------------------

def rule_nonneg(spectrum: np.ndarray) -> np.ndarray:
    """Penalty for negative values; zero if >= 0, else magnitude below zero."""
    return np.maximum(0.0, -spectrum)


def rule_upper_bound(spectrum: np.ndarray, max_flux: float) -> np.ndarray:
    """Penalty for exceeding an upper physical bound (e.g., stellar continuum)."""
    return np.maximum(0.0, spectrum - max_flux)


def rule_smoothness(spectrum: np.ndarray) -> np.ndarray:
    """
    Smoothness pressure ~ |second derivative|.
    Returns per-wavelength penalties, centered (pad edges with zeros).
    """
    pen = np.zeros_like(spectrum)
    # finite differences
    # d2[i] = s[i+1] - 2 s[i] + s[i-1]
    d2 = np.zeros_like(spectrum)
    d2[1:-1] = spectrum[2:] - 2.0 * spectrum[1:-1] + spectrum[:-2]
    pen = np.abs(d2)
    return pen


def rule_band_coherence(spectrum: np.ndarray, bands: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
    """
    Within each molecular band region, penalize inconsistency from the band's median response:
      penalty[i] = |s[i] - median_band|
    Accumulate across bands (sum if overlapping).
    """
    pen = np.zeros_like(spectrum)
    for _, ranges in (bands or {}).items():
        for (a, b) in ranges:
            a = max(0, int(a))
            b = min(len(spectrum), int(b))
            if b <= a: 
                continue
            seg = spectrum[a:b]
            med = np.median(seg)
            pen[a:b] += np.abs(seg - med)
    return pen


def compute_symbolic_pressures(
    spectra: np.ndarray,
    cfg: OverlayConfig
) -> Dict[str, np.ndarray]:
    """
    Returns dict of per-sample per-wavelength pressures.
      keys: 'nonneg','upper','smooth','bandcoh','total'
      shapes: (B, n_wavelengths)
    """
    B, W = spectra.shape
    out = {
        "nonneg": np.zeros_like(spectra),
        "upper": np.zeros_like(spectra),
        "smooth": np.zeros_like(spectra),
        "bandcoh": np.zeros_like(spectra),
    }
    for i in range(B):
        s = spectra[i]
        out["nonneg"][i] = rule_nonneg(s)
        out["upper"][i] = rule_upper_bound(s, cfg.max_flux)
        out["smooth"][i] = rule_smoothness(s)
        out["bandcoh"][i] = rule_band_coherence(s, cfg.bands or {})
    total = (
        cfg.nonneg_lambda * out["nonneg"]
        + cfg.bound_lambda * out["upper"]
        + cfg.smooth_lambda * out["smooth"]
        + cfg.bandcoh_lambda * out["bandcoh"]
    )
    out["total"] = total
    return out


# -----------------------------
# SHAP computation
# -----------------------------

def predict_spectrum_head(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Normalize model outputs to shape (B, W) always.
    Accepts dict {'mu','log_sigma'} or plain tensor. Uses 'mu' if present.
    """
    with torch.no_grad():
        y = model(x)
        if isinstance(y, dict):
            y = y.get("mu", next(iter(y.values())))
        return y


def shap_explainer_for_model(model: torch.nn.Module, Xb: np.ndarray, device: torch.device, method: str, max_evals: int):
    """
    Build a SHAP explainer; prefer DeepExplainer if model is PyTorch native and supports gradient,
    else fall back to KernelExplainer (model-agnostic).
    """
    model.eval()

    def f_predict(inp: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(inp).float().to(device)
        out = predict_spectrum_head(model, t).detach().cpu().numpy()
        return out

    # Try deep explainer if possible (single-output per call requirement is avoided by looping)
    if method in ("deep", "auto"):
        try:
            # Create a wrapper that returns a single dimension at a time so DeepExplainer can handle
            # multi-output by looping outside. We'll still return the explainer factory.
            return ("deep", f_predict)
        except Exception:
            pass

    # Fallback: kernel
    return ("kernel", f_predict)


def compute_shap_values(
    model: torch.nn.Module,
    X: np.ndarray,
    X_bg: np.ndarray,
    device: torch.device,
    method: str,
    max_evals: int,
) -> np.ndarray:
    """
    Returns SHAP values with shape (B, F, W_out).
    Strategy:
      - For DeepExplainer: loop over each output wavelength; compute shap for that scalar output.
      - For KernelExplainer: same loop (expensive but general).
    """
    B, F = X.shape
    # Probe output width
    with torch.no_grad():
        out = predict_spectrum_head(model, torch.from_numpy(X[:1]).float().to(device)).cpu().numpy()
    W_out = out.shape[1]

    expl_kind, f_predict = shap_explainer_for_model(model, X_bg, device, method, max_evals)
    console.log(f"[cyan]SHAP backend[/cyan]: {expl_kind} | W_out={W_out}")

    shap_values = np.zeros((B, F, W_out), dtype=np.float32)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Computing SHAP per wavelength", total=W_out)

        if expl_kind == "deep":
            # DeepExplainer expects a torch model; we use grad-based shap by building per-dim heads
            # We implement gradient × input as a fast proxy (when DeepExplainer is not instantiated).
            # For robustness across arbitrary models, fallback to gradient×input here.
            X_t = torch.from_numpy(X).float().to(device).requires_grad_(True)
            model.eval()
            for j in range(W_out):
                model.zero_grad(set_to_none=True)
                y = predict_spectrum_head(model, X_t)[:, j]  # (B,)
                # gradient × input attribution
                y.sum().backward(retain_graph=True)
                grad = X_t.grad.detach().cpu().numpy()  # (B,F)
                shap_values[:, :, j] = grad * X  # simple proxy
                X_t.grad.zero_()
                progress.advance(task)
        else:
            # KernelExplainer (model-agnostic)
            # Use small background sample
            bg = shap.kmeans(X_bg, min(len(X_bg), 32))
            for j in range(W_out):
                # scalar-output function for dimension j
                f_j = lambda inp: f_predict(inp)[:, j]
                expl = shap.KernelExplainer(f_j, bg)
                vals = expl.shap_values(
                    X,
                    nsamples=max_evals,
                    l1_reg="aic",
                )  # returns (B,F)
                shap_values[:, :, j] = np.asarray(vals, dtype=np.float32)
                progress.advance(task)

    return shap_values  # (B,F,W)


# -----------------------------
# Visualization / Saving
# -----------------------------

def wavelength_axis(cfg: OverlayConfig) -> np.ndarray:
    return np.linspace(cfg.wl_start, cfg.wl_end, cfg.n_wavelengths)


def plot_heatmap(
    M: np.ndarray,
    x_axis: np.ndarray,
    title: str,
    outpath: Path,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
):
    plt.figure(figsize=(12, 4))
    vmin, vmax = (None, None)
    if center_zero:
        vmax = np.nanpercentile(np.abs(M), 99.0)
        vmin = -vmax
    plt.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
               extent=[x_axis[0], x_axis[-1], 0, M.shape[0]])
    plt.colorbar(label="magnitude")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Sample")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_overlay_csv(
    spectra: np.ndarray,
    shap_vals: np.ndarray,
    sym: Dict[str, np.ndarray],
    wl: np.ndarray,
    out_csv: Path,
):
    """
    Save per-sample summaries: mean |SHAP| per wavelength, mean rule pressures, and spectrum.
    """
    B, W = spectra.shape
    F = shap_vals.shape[1]
    # Aggregate |SHAP| over features dimension
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=1)  # (B,W)

    records = []
    for i in range(B):
        rec = {
            "sample": int(i),
            **{f"spectrum_{k}": float(v) for k, v in zip(wl, spectra[i])},
            **{f"shap_{k}": float(v) for k, v in zip(wl, mean_abs_shap[i])},
            **{f"nonneg_{k}": float(v) for k, v in zip(wl, sym["nonneg"][i])},
            **{f"upper_{k}": float(v) for k, v in zip(wl, sym["upper"][i])},
            **{f"smooth_{k}": float(v) for k, v in zip(wl, sym["smooth"][i])},
            **{f"bandcoh_{k}": float(v) for k, v in zip(wl, sym["bandcoh"][i])},
            **{f"total_{k}": float(v) for k, v in zip(wl, sym["total"][i])},
        }
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)


def fuse_overlay(mean_abs_shap: np.ndarray, sym_total: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Simple fusion: normalize each map to [0,1] per-sample then weighted sum.
    """
    B, W = mean_abs_shap.shape
    s1 = mean_abs_shap - mean_abs_shap.min(axis=1, keepdims=True)
    s1 = s1 / (s1.max(axis=1, keepdims=True) + 1e-9)
    s2 = sym_total - sym_total.min(axis=1, keepdims=True)
    s2 = s2 / (s2.max(axis=1, keepdims=True) + 1e-9)
    return (1 - alpha) * s1 + alpha * s2


# -----------------------------
# Main CLI
# -----------------------------

@app.command()
def run(
    cfg_path: str = typer.Option(..., help="YAML config for overlay (paths, rules, SHAP params)"),
):
    """
    Example config YAML:

    model_path: "artifacts/model.pt"
    data_path: "data/npz/val_latents.npz"
    output_dir: "outputs/diagnostics/shap_symbolic"
    n_samples: 64
    seed: 42
    n_wavelengths: 283
    wl_start: 0.5
    wl_end: 7.8
    max_flux: 1.0
    smooth_lambda: 1.0
    nonneg_lambda: 1.0
    bound_lambda: 1.0
    bandcoh_lambda: 1.0
    bands:
      H2O:
        - [30, 60]
        - [140, 165]
      CO2:
        - [200, 230]
    shap_background: 64
    shap_max_evals: 1024
    shap_method: "auto"
    device: "auto"
    """
    cfgd = load_yaml(cfg_path)
    cfg = OverlayConfig(**cfgd)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfgd, out_dir / "overlay_config.yaml")

    # Load
    console.rule("[bold]Loading model & data")
    model = load_model(cfg.model_path, device)
    X, Y = load_dataset_npz(cfg.data_path, cfg.n_samples, cfg.n_wavelengths)

    console.print(f"[green]Model[/green]: {type(model).__name__} on {device}")
    console.print(f"[green]Data[/green]: X={X.shape}, Y={Y.shape}")

    # Probe predictions (spectra)
    with torch.no_grad():
        spectra = predict_spectrum_head(
            model, torch.from_numpy(X).float().to(device)
        ).cpu().numpy()  # (B,W)

    # Compute symbolic pressures
    console.rule("[bold]Computing symbolic rule pressures")
    sym = compute_symbolic_pressures(spectra, cfg)

    # SHAP values
    console.rule("[bold]Computing SHAP values")
    # Background for kernel/approx: sample from X
    bg_n = min(cfg.shap_background, X.shape[0])
    X_bg = X[np.random.choice(X.shape[0], bg_n, replace=False)]

    shap_vals = compute_shap_values(
        model=model,
        X=X,
        X_bg=X_bg,
        device=device,
        method=cfg.shap_method,
        max_evals=cfg.shap_max_evals,
    )  # (B,F,W)

    # Aggregate |SHAP| over features to compare with per-wavelength symbolic pressures
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=1)  # (B,W)

    # Save artifacts
    wl = wavelength_axis(cfg)
    console.rule("[bold]Saving artifacts")
    # Heatmaps
    plot_heatmap(
        mean_abs_shap,
        wl,
        "Mean |SHAP| per wavelength (samples x wavelengths)",
        out_dir / "heatmap_mean_abs_shap.png",
    )
    plot_heatmap(
        sym["total"],
        wl,
        "Total symbolic pressure per wavelength",
        out_dir / "heatmap_symbolic_total.png",
        cmap="magma",
        center_zero=False,
    )
    fused = fuse_overlay(mean_abs_shap, sym["total"], alpha=0.5)
    plot_heatmap(
        fused,
        wl,
        "Fused overlay (normalized SHAP ⊕ symbolic pressure)",
        out_dir / "heatmap_fused_overlay.png",
        cmap="viridis",
        center_zero=False,
    )

    # Per-sample CSV
    save_overlay_csv(
        spectra=spectra,
        shap_vals=shap_vals,
        sym=sym,
        wl=wl,
        out_csv=out_dir / "overlay_per_sample.csv",
    )

    # JSON summary (global)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": cfg.model_path,
        "data_path": cfg.data_path,
        "n_samples": int(cfg.n_samples),
        "seed": int(cfg.seed),
        "device": str(device),
        "shap_backend": cfg.shap_method,
        "mean_symbolic_total": float(np.mean(sym["total"])),
        "mean_abs_shap": float(np.mean(mean_abs_shap)),
    }
    with open(out_dir / "overlay_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console table
    tbl = Table(title="Overlay Summary")
    tbl.add_column("Metric")
    tbl.add_column("Value")
    for k, v in summary.items():
        tbl.add_row(k, str(v))
    console.print(tbl)

    console.print(f"[bold green]Saved overlays to:[/bold green] {out_dir.resolve()}")
    console.rule("[bold]Done")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted[/red]")
        sys.exit(130)
