#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/symbolic_influence_map.py

SpectraMind V50 — Symbolic Influence Map (∂L_sym/∂μ) Generator

Purpose
-------
Compute per‑rule symbolic influence maps on predicted spectra μ(λ) by differentiating
physics‑/symbolic‑informed losses w.r.t. μ. The result is an influence magnitude
|∂L_rule/∂μ| per bin, per sample, with flexible rule definitions (YAML/JSON) and
built‑in canonical rules (smoothness, nonnegativity, band constraints, edge/monotone,
and FFT penalty). Outputs include:
  • Per‑rule mean influence curves over bins (CSV + PNG + Plotly HTML)
  • Per‑sample, per‑rule top‑K influential bins (CSV)
  • Global summary JSON (timings, config, rules)
  • Optional compact HTML report linking all artifacts
  • Optional overlays from entropy, GLL, and symbolic violation counts

Design Notes
------------
• CLI‑first (Typer), reproducibility‑aware, deterministic seeds.
• Uses PyTorch autograd to obtain ∂L/∂μ for each rule.
• Accepts μ (N×B), optional σ and y for diagnostics, optional wavelength vector λ (B,)
  and metadata (CSV) for rule conditioning.
• Rules can be provided in a YAML/JSON file or you can rely on the built‑in defaults.
• Aggregations: mean/sum/max across samples; top‑K bins per sample & rule.
• Visualization: Matplotlib PNG (static) and Plotly HTML (interactive).
• Logs an auditable entry to logs/v50_debug_log.md (if writeable).

Typical Usage
-------------
# Minimal (defaults to built‑in rules)
python -m tools.symbolic_influence_map \
  --mu outputs/predictions/mu.npy \
  --outdir outputs/symbolic_influence

# With wavelengths, custom rules, and HTML report
python -m tools.symbolic_influence_map \
  --mu outputs/predictions/mu.npy \
  --wavelengths data/wavelengths.npy \
  --rules configs/symbolic_rules.yaml \
  --outdir outputs/symbolic_influence --html

# Include σ and y (for diagnostics overlays only)
python -m tools.symbolic_influence_map \
  --mu mu.npy --sigma sigma.npy --y labels.npy \
  --outdir out --topk 8 --fft-n 48 --seed 123 --html

Inputs
------
--mu           Path to N×B .npy (predicted μ)
--sigma        Optional N×B .npy (predicted σ) for diagnostics only
--y            Optional N×B .npy (targets) for diagnostics only (GLL)
--wavelengths  Optional B .npy of λ values (μm or index units)
--metadata     Optional CSV with N rows; can be used to condition rules
--rules        Optional YAML/JSON rules file (see schema below)
--outdir       Output directory

Rule Schema (YAML/JSON)
-----------------------
Each rule entry is a dict with a 'type' and parameters. Supported types:

- smoothness:
    type: smoothness
    weight: 1.0
    order: 2                 # finite-diff order: 1 or 2
    l2_factor: 1.0          # multiplier for L2 norm of finite diffs

- nonnegativity:
    type: nonnegativity
    weight: 0.5
    margin: 0.0             # penalty on (margin - μ)+

- band_min:                 # encourage lower μ inside a band (or multiple bands)
    type: band_min
    weight: 1.0
    bands:
      - [λ_lo, λ_hi]
      - [λ_lo2, λ_hi2]
    target: 0.0             # minimize mean(μ[band]) - target

- band_max:                 # encourage higher μ inside a band
    type: band_max
    weight: 0.8
    bands:
      - [λ_lo, λ_hi]
    target: 1.0

- monotone:                 # encourage monotonicity (increasing or decreasing)
    type: monotone
    weight: 0.6
    direction: "increasing" # or "decreasing"

- edge:                     # encourage a step/edge around λ0 ± width
    type: edge
    weight: 0.9
    lambda0: 1.40
    width: 0.05             # band half‑width around lambda0
    sign: "drop"            # "rise" or "drop"

- fft_smooth:               # penalize high‑frequency energy
    type: fft_smooth
    weight: 0.4
    n_freq_keep: 32         # keep low‑freq; penalize the rest

If 'wavelengths' are not provided, band/edge rules interpret bins as indices.

Outputs
-------
outdir/
  tables/
    influence_mean_per_rule.csv
    influence_topk_bins_per_sample_rule.csv
  plots/
    rule_<name>_mean_influence.png
    rule_<name>_mean_influence.html
  summary.json
  report_symbolic_influence.html   (if --html)

Author
------
SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme

# Torch for autograd
try:
    import torch
    _HAS_TORCH = True
except Exception as _e:
    _HAS_TORCH = False

# Plotting
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio

# YAML for rule configs
try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

app = typer.Typer(add_completion=False, help="SpectraMind V50 — Symbolic Influence Map")
console = Console(theme=Theme({"info": "cyan", "warn": "yellow", "err": "bold red"}))


# =========================
# Utility & IO
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} did not contain an ndarray")
    return arr


def write_debug_log(entry: str) -> None:
    """Append a short audit entry to logs/v50_debug_log.md (best-effort)."""
    try:
        logdir = Path("logs")
        logdir.mkdir(parents=True, exist_ok=True)
        with open(logdir / "v50_debug_log.md", "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n")
    except Exception:
        # do not fail the run if logging isn't possible
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
    if _HAS_TORCH:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def now_str() -> str:
    import datetime as dt
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Rule Model
# =========================

@dataclass
class Rule:
    name: str
    type: str
    weight: float = 1.0
    # Specific params:
    order: int = 2                     # smoothness
    l2_factor: float = 1.0             # smoothness
    margin: float = 0.0                # nonnegativity
    bands: List[Tuple[float, float]] = field(default_factory=list)  # band_min / band_max
    target: float = 0.0                # band target
    direction: str = "increasing"      # monotone
    lambda0: float = 0.0               # edge center
    width: float = 0.0                 # edge half-width
    sign: str = "drop"                 # "rise" or "drop" for edge
    n_freq_keep: int = 32              # fft_smooth

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d


def _default_rules(wavelengths: Optional[np.ndarray], B: int) -> List[Rule]:
    """
    Built-in canonical rules if none are provided:
    - 2nd-order smoothness
    - nonnegativity (margin=0)
    - FFT smoothing (keep first 32 freqs)
    - Example molecular-like bands (if λ provided), else index-based bands
    """
    rules: List[Rule] = [
        Rule(name="smoothness_o2", type="smoothness", weight=1.0, order=2, l2_factor=1.0),
        Rule(name="nonnegativity", type="nonnegativity", weight=0.5, margin=0.0),
        Rule(name="fft_smooth_k32", type="fft_smooth", weight=0.5, n_freq_keep=min(32, B // 2)),
    ]
    # Add example bands (heuristics)
    if wavelengths is not None and len(wavelengths) == B:
        # Example: water & CO2-ish regions (user should supply precise masks via rules file)
        # This is intentionally generic and can be overridden by --rules.
        lam = wavelengths
        bands_h2o = [(1.32, 1.50), (1.80, 2.00)]
        bands_co2 = [(1.95, 2.15), (4.15, 4.45)]
        rules.append(Rule(name="band_max_h2o_like", type="band_max", weight=0.6, bands=bands_h2o, target=1.0))
        rules.append(Rule(name="band_max_co2_like", type="band_max", weight=0.6, bands=bands_co2, target=1.0))
        # Example edge near 1.4 μm
        rules.append(Rule(name="edge_drop_1p4", type="edge", weight=0.4, lambda0=1.40, width=0.05, sign="drop"))
        # Example monotone between 0.9–1.2 μm if present
        rules.append(Rule(name="monotone_inc_short", type="monotone", weight=0.4, direction="increasing"))
    else:
        # Index-based bands: split B into thirds
        t = B // 3
        bands_a = [(0, max(1, t - 1))]
        bands_b = [(t, max(t + 1, 2 * t - 1))]
        rules.append(Rule(name="band_max_A", type="band_max", weight=0.5, bands=bands_a, target=1.0))
        rules.append(Rule(name="band_max_B", type="band_max", weight=0.5, bands=bands_b, target=1.0))
    return rules


def _load_rules(path: Optional[str], wavelengths: Optional[np.ndarray], B: int) -> List[Rule]:
    if path is None:
        return _default_rules(wavelengths, B)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    try:
        if p.suffix.lower() in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML is not installed; cannot read YAML rules. Install pyyaml or use JSON.")
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        else:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
        rules: List[Rule] = []
        for i, r in enumerate(raw):
            rule = Rule(
                name=r.get("name", f"rule_{i}"),
                type=r["type"],
                weight=float(r.get("weight", 1.0)),
                order=int(r.get("order", 2)),
                l2_factor=float(r.get("l2_factor", 1.0)),
                margin=float(r.get("margin", 0.0)),
                bands=[(float(a), float(b)) for a, b in r.get("bands", [])],
                target=float(r.get("target", 0.0)),
                direction=str(r.get("direction", "increasing")),
                lambda0=float(r.get("lambda0", 0.0)),
                width=float(r.get("width", 0.0)),
                sign=str(r.get("sign", "drop")),
                n_freq_keep=int(r.get("n_freq_keep", 32)),
            )
            rules.append(rule)
        return rules
    except Exception as e:
        raise RuntimeError(f"Failed to parse rules file {path}: {e}")


# =========================
# Losses (differentiable)
# =========================

def _finite_diff(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    # x: (..., B)
    if order == 1:
        return x[..., 1:] - x[..., :-1]
    elif order == 2:
        return x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    else:
        raise ValueError("order must be 1 or 2")


def loss_smoothness(mu: torch.Tensor, order: int = 2, l2_factor: float = 1.0) -> torch.Tensor:
    d = _finite_diff(mu, order=order)
    return l2_factor * torch.mean(d ** 2)


def loss_nonnegativity(mu: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    # Penalize (margin - μ)+
    return torch.mean(torch.clamp(margin - mu, min=0.0) ** 2)


def _band_mask_from_lambda(wavelengths: Optional[np.ndarray], B: int, bands: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(B, dtype=np.float32)
    if wavelengths is None:
        # Interpret bands as index ranges
        for a, b in bands:
            ia = int(round(a))
            ib = int(round(b))
            ia = max(0, min(B - 1, ia))
            ib = max(0, min(B - 1, ib))
            if ib < ia:
                ia, ib = ib, ia
            mask[ia:ib + 1] = 1.0
        return mask
    lam = wavelengths
    for a, b in bands:
        mask = np.where((lam >= a) & (lam <= b), 1.0, mask)
    return mask.astype(np.float32)


def loss_band_min(mu: torch.Tensor, band_mask: torch.Tensor, target: float = 0.0) -> torch.Tensor:
    # Encourage mean(μ in band) to approach 'target' (lower μ -> absorption if needed)
    eps = 1e-8
    wsum = torch.clamp(band_mask.sum(), min=eps)
    mean_band = (mu * band_mask).sum() / wsum
    return (mean_band - float(target)) ** 2


def loss_band_max(mu: torch.Tensor, band_mask: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    # Encourage mean(μ in band) to approach 'target' (higher μ)
    eps = 1e-8
    wsum = torch.clamp(band_mask.sum(), min=eps)
    mean_band = (mu * band_mask).sum() / wsum
    return (float(target) - mean_band) ** 2


def loss_monotone(mu: torch.Tensor, direction: str = "increasing") -> torch.Tensor:
    d = _finite_diff(mu, order=1)
    if direction.lower().startswith("inc"):
        # Penalize negative slopes
        return torch.mean(torch.clamp(-d, min=0.0) ** 2)
    else:
        # Penalize positive slopes
        return torch.mean(torch.clamp(d, min=0.0) ** 2)


def loss_edge(mu: torch.Tensor, wavelengths: Optional[np.ndarray], lambda0: float, width: float, sign: str = "drop") -> torch.Tensor:
    # Encourage a "drop" or "rise" in the window (lambda0 ± width): maximize |Δμ|
    # Implement as penalty on lack of edge magnitude in window.
    B = mu.numel()
    if wavelengths is None:
        # Use index‑centered window
        c = int(round(lambda0))
        hw = max(1, int(round(width)))
        lo = max(0, c - hw)
        hi = min(B - 1, c + hw)
        idx = torch.arange(B, device=mu.device)
        mask = ((idx >= lo) & (idx <= hi)).float()
    else:
        lam = torch.tensor(wavelengths, device=mu.device, dtype=mu.dtype)
        mask = ((lam >= (lambda0 - width)) & (lam <= (lambda0 + width))).float()

    d = _finite_diff(mu, order=1)
    # Align mask size to d (B-1)
    mask_d = mask[1:]
    # Edge magnitude inside window (mean of |d| masked)
    eps = 1e-8
    wsum = torch.clamp(mask_d.sum(), min=eps)
    edge_mag = torch.sum(torch.abs(d) * mask_d) / wsum

    # For "drop", we want negative d dominant; for "rise", positive
    if sign.lower() == "drop":
        desirability = torch.sum(torch.clamp(-d, min=0.0) * mask_d) / wsum
    else:
        desirability = torch.sum(torch.clamp(d, min=0.0) * mask_d) / wsum

    # Penalize inverse of desirability + encourage edge magnitude
    return (1.0 / (desirability + 1e-3)) + (1.0 / (edge_mag + 1e-3))


def loss_fft_smooth(mu: torch.Tensor, n_freq_keep: int = 32) -> torch.Tensor:
    # Penalize energy outside first n_freq_keep FFT components
    x = mu
    B = x.numel()
    fx = torch.fft.rfft(x)  # length B//2 + 1
    mag2 = (fx.real ** 2 + fx.imag ** 2)
    k = max(1, n_freq_keep)
    keep = mag2[:k]
    drop = mag2[k:]
    if drop.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return torch.mean(drop)  # penalize average high‑freq energy


def rule_loss(rule: Rule, mu: torch.Tensor, wavelengths: Optional[np.ndarray]) -> torch.Tensor:
    """Compute a single rule's differentiable loss on a 1D μ tensor (B,)."""
    t = rule.type.lower()
    if t == "smoothness":
        return rule.weight * loss_smoothness(mu, order=rule.order, l2_factor=rule.l2_factor)
    if t == "nonnegativity":
        return rule.weight * loss_nonnegativity(mu, margin=rule.margin)
    if t == "band_min":
        mask = _band_mask_from_lambda(wavelengths, mu.numel(), rule.bands)
        band = torch.tensor(mask, device=mu.device, dtype=mu.dtype)
        return rule.weight * loss_band_min(mu, band, target=rule.target)
    if t == "band_max":
        mask = _band_mask_from_lambda(wavelengths, mu.numel(), rule.bands)
        band = torch.tensor(mask, device=mu.device, dtype=mu.dtype)
        return rule.weight * loss_band_max(mu, band, target=rule.target)
    if t == "monotone":
        return rule.weight * loss_monotone(mu, direction=rule.direction)
    if t == "edge":
        return rule.weight * loss_edge(mu, wavelengths, rule.lambda0, rule.width, sign=rule.sign)
    if t == "fft_smooth":
        return rule.weight * loss_fft_smooth(mu, n_freq_keep=rule.n_freq_keep)
    raise ValueError(f"Unknown rule type: {rule.type}")


# =========================
# Diagnostics (entropy/GLL)
# =========================

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


# =========================
# Core Influence Computation
# =========================

@dataclass
class InfluenceResults:
    # Per-rule mean influence over samples (B,)
    mean_influence: Dict[str, np.ndarray]
    # Per-sample topK bins for each rule: dict[rule] -> list of lists of bin indices (N × K)
    topk_bins: Dict[str, List[List[int]]]
    # Optional overlays
    entropy: Optional[np.ndarray] = None
    gll: Optional[np.ndarray] = None


def compute_influence_maps(
    mu: np.ndarray,
    rules: List[Rule],
    wavelengths: Optional[np.ndarray],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    topk: int = 8,
) -> InfluenceResults:
    """
    For each sample i and each rule r:
      1) μ_i -> torch tensor with requires_grad
      2) L_r(μ_i) -> scalar; backprop to get ∂L_r/∂μ_i (B,)
      3) Influence_i,r = |∂L_r/∂μ_i|
    Aggregate across N to produce mean influence curves and per‑sample top‑K bins.
    """
    assert _HAS_TORCH, "PyTorch is required for symbolic_influence_map."
    N, B = mu.shape
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    mean_influence: Dict[str, np.ndarray] = {r.name: np.zeros(B, dtype=np.float64) for r in rules}
    topk_bins: Dict[str, List[List[int]]] = {r.name: [[] for _ in range(N)] for r in rules}

    wl = wavelengths if wavelengths is None else np.asarray(wavelengths, dtype=np.float32)

    for i in range(N):
        mui = torch.tensor(mu[i], device=device, dtype=dtype, requires_grad=True)
        for r in rules:
            mui.grad = None  # reset grad
            L = rule_loss(r, mui, wl)
            if mui.grad is not None:
                mui.grad.zero_()
            L.backward()
            grad = mui.grad.detach().abs().to("cpu").numpy()  # (B,)
            mean_influence[r.name] += grad
            # record top‑K bins indices
            top_idx = np.argsort(-grad)[:topk].tolist()
            topk_bins[r.name][i] = top_idx

    # Average across samples
    for r in rules:
        mean_influence[r.name] = mean_influence[r.name] / float(N)
    return InfluenceResults(mean_influence=mean_influence, topk_bins=topk_bins)


# =========================
# Visualization & Exports
# =========================

def plot_mean_influence_png(
    wavelengths: Optional[np.ndarray],
    mean_influence: Dict[str, np.ndarray],
    outdir: Path,
) -> List[str]:
    outfiles: List[str] = []
    x = wavelengths if wavelengths is not None else np.arange(len(next(iter(mean_influence.values()))))
    for rname, infl in mean_influence.items():
        plt.figure(figsize=(9.5, 3.6), dpi=120)
        plt.plot(x, infl, linewidth=2.0)
        plt.xlabel("Wavelength" if wavelengths is not None else "Bin")
        plt.ylabel("Mean |∂L/∂μ|")
        plt.title(f"Mean Influence — {rname}")
        plt.grid(True, alpha=0.25)
        fn = outdir / f"rule_{rname}_mean_influence.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        outfiles.append(fn.name)
    return outfiles


def plot_mean_influence_html(
    wavelengths: Optional[np.ndarray],
    mean_influence: Dict[str, np.ndarray],
    outdir: Path,
) -> List[str]:
    outfiles: List[str] = []
    x = wavelengths if wavelengths is not None else np.arange(len(next(iter(mean_influence.values()))))
    for rname, infl in mean_influence.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=infl, mode="lines", name=rname))
        fig.update_layout(
            title=f"Mean Influence — {rname}",
            xaxis_title="Wavelength" if wavelengths is not None else "Bin",
            yaxis_title="Mean |∂L/∂μ|",
            template="plotly_white",
            height=360,
        )
        fn = outdir / f"rule_{rname}_mean_influence.html"
        pio.write_html(fig, file=str(fn), include_plotlyjs="cdn", full_html=True)
        outfiles.append(fn.name)
    return outfiles


def export_tables(
    wavelengths: Optional[np.ndarray],
    mean_influence: Dict[str, np.ndarray],
    topk_bins: Dict[str, List[List[int]]],
    outdir: Path,
) -> Tuple[str, str]:
    tables_dir = outdir / "tables"
    ensure_dir(tables_dir)

    # Mean influence table
    all_rows = []
    for rname, infl in mean_influence.items():
        B = len(infl)
        if wavelengths is not None and len(wavelengths) == B:
            for b in range(B):
                all_rows.append({"rule": rname, "bin": b, "lambda": float(wavelengths[b]), "mean_influence": float(infl[b])})
        else:
            for b in range(B):
                all_rows.append({"rule": rname, "bin": b, "mean_influence": float(infl[b])})
    df_mean = pd.DataFrame(all_rows)
    mean_csv = tables_dir / "influence_mean_per_rule.csv"
    df_mean.to_csv(mean_csv, index=False)

    # Top‑K per sample
    rows = []
    for rname, lists in topk_bins.items():
        for i, bins in enumerate(lists):
            rows.append({
                "sample": i,
                "rule": rname,
                "topk_bins": json.dumps(bins),
            })
    df_topk = pd.DataFrame(rows)
    topk_csv = tables_dir / "influence_topk_bins_per_sample_rule.csv"
    df_topk.to_csv(topk_csv, index=False)

    return mean_csv.name, topk_csv.name


def build_html_report(
    outdir: Path,
    png_files: List[str],
    html_files: List[str],
    tables: Tuple[str, str],
    summary: Dict[str, Any],
) -> str:
    report_path = outdir / "report_symbolic_influence.html"
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; margin: 16px; color: #0e1116; }
    h1 { font-size: 20px; margin: 8px 0 12px; }
    h2 { font-size: 16px; margin: 16px 0 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    a { color: #0b5fff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { background: #f3f4f6; padding: 2px 4px; border-radius: 6px; }
    table { border-collapse: collapse; font-size: 12px; width: 100%; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
    """
    imgs = "".join([f'<div class="card"><img src="plots/{fn}" alt="{fn}" style="width:100%;height:auto"/></div>' for fn in png_files])
    links = "".join([f'<div class="card"><a href="plots/{fn}">{fn}</a></div>' for fn in html_files])

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SpectraMind V50 — Symbolic Influence Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{css}</style>
</head>
<body>
  <h1>SpectraMind V50 — Symbolic Influence Map</h1>
  <div class="card"><pre>{json.dumps(summary, indent=2)}</pre></div>

  <h2>Static Mean Influence PNGs</h2>
  <div class="grid">{imgs}</div>

  <h2>Interactive Mean Influence (Plotly)</h2>
  <div class="grid">{links}</div>

  <h2>Tables</h2>
  <div class="grid">
    <div class="card"><a href="tables/{tables[0]}">{tables[0]}</a></div>
    <div class="card"><a href="tables/{tables[1]}">{tables[1]}</a></div>
  </div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path.name


# =========================
# Orchestration
# =========================

def run_symbolic_influence(
    mu_path: str,
    outdir: str,
    rules_path: Optional[str] = None,
    wavelengths_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    sigma_path: Optional[str] = None,
    y_path: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
    topk: int = 8,
    seed: int = 42,
    html_report: bool = False,
    fft_n: Optional[int] = None,
) -> None:
    t0 = time.time()
    set_global_seed(seed)

    out = Path(outdir)
    plots_dir = out / "plots"
    ensure_dir(out)
    ensure_dir(plots_dir)

    console.rule("[info]SpectraMind V50 — Symbolic Influence Map")
    console.print(f"[info]μ: {mu_path}")
    if sigma_path: console.print(f"[info]σ: {sigma_path}")
    if y_path: console.print(f"[info]y: {y_path}")
    if wavelengths_path: console.print(f"[info]λ: {wavelengths_path}")
    if rules_path: console.print(f"[info]rules: {rules_path}")
    if metadata_path: console.print(f"[info]metadata: {metadata_path}")

    # Load inputs
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
        t_load = progress.add_task("Loading arrays", total=None)
        mu = load_npy(mu_path)
        if mu is None:
            raise FileNotFoundError("--mu is required")
        sigma = load_npy(sigma_path) if sigma_path else None
        y = load_npy(y_path) if y_path else None
        wavelengths = load_npy(wavelengths_path) if wavelengths_path else None
        meta_df = pd.read_csv(metadata_path) if metadata_path else None
        progress.update(t_load, advance=1, visible=False)

    N, B = mu.shape
    console.print(f"[info]Loaded μ: {N}×{B}")

    # Rules
    rules = _load_rules(rules_path, wavelengths, B)
    # Allow quick override for fft rule via --fft-n
    if fft_n is not None:
        for r in rules:
            if r.type.lower() == "fft_smooth":
                r.n_freq_keep = int(fft_n)

    # Compute optional overlays
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
        t_diag = progress.add_task("Diagnostics overlays", total=N)
        entropy_vec = np.zeros(N, dtype=np.float64)
        gll_vec = np.full(N, np.nan, dtype=np.float64)
        for i in range(N):
            entropy_vec[i] = entropy_row(mu[i, :])
            if sigma is not None and y is not None:
                gll_vec[i] = gll_row(mu[i, :], sigma[i, :], y[i, :])
            progress.advance(t_diag)

    # Compute influence maps
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available. Please install torch to compute influence maps.")

    torch_dtype = getattr(torch, dtype, torch.float32)
    device_sel = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        console.print("[warn]CUDA requested but not available; falling back to CPU.")
        device_sel = "cpu"

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as progress:
        t_infl = progress.add_task("Computing |∂L/∂μ| per rule", total=None)
        infl = compute_influence_maps(mu=mu, rules=rules, wavelengths=wavelengths, device=device_sel, dtype=torch_dtype, topk=topk)
        progress.update(t_infl, advance=1, visible=False)

    # Exports
    mean_csv, topk_csv = export_tables(
        wavelengths=wavelengths,
        mean_influence=infl.mean_influence,
        topk_bins=infl.topk_bins,
        outdir=out,
    )
    pngs = plot_mean_influence_png(wavelengths, infl.mean_influence, outdir=plots_dir)
    htmls = plot_mean_influence_html(wavelengths, infl.mean_influence, outdir=plots_dir)

    # Summary JSON
    summary = {
        "timestamp": now_str(),
        "mu_path": mu_path,
        "sigma_path": sigma_path,
        "y_path": y_path,
        "wavelengths_path": wavelengths_path,
        "rules_path": rules_path,
        "metadata_path": metadata_path,
        "N": int(N),
        "B": int(B),
        "topk": int(topk),
        "device": device_sel,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "rules": [r.to_dict() for r in rules],
        "tables": {"mean": mean_csv, "topk": topk_csv},
        "plots": {"png": pngs, "html": htmls},
        "diagnostics": {
            "entropy_mean": float(np.nanmean(entropy_vec)),
            "gll_mean": float(np.nanmean(gll_vec)) if np.isfinite(gll_vec).any() else None,
        },
        "timing_sec": round(time.time() - t0, 3),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # HTML report
    if html_report:
        report = build_html_report(out, png_files=pngs, html_files=htmls, tables=(mean_csv, topk_csv), summary=summary)
        console.print(f"[info]Wrote HTML report → {out / report}")

    # Append to audit log (best-effort)
    write_debug_log(f"- {now_str()} | symbolic_influence_map | mu={mu_path} rules={rules_path or 'builtin'} out={outdir} N={N} B={B} topk={topk} device={device_sel} dtype={summary['dtype']}")

    console.rule("[info]Done")
    console.print(f"[info]Elapsed: {round(time.time() - t0, 2)} s")
    console.print(f"[info]Artifacts in: {outdir}")


# =========================
# Typer CLI
# =========================

@app.command("run")
def cli_run(
    mu: str = typer.Option(..., help="Path to μ.npy (N×B)"),
    outdir: str = typer.Option(..., help="Output directory for artifacts"),
    rules: Optional[str] = typer.Option(None, help="Rules file (.yaml/.yml/.json). If omitted, use built‑in defaults."),
    wavelengths: Optional[str] = typer.Option(None, help="Path to wavelengths.npy (B,)"),
    metadata: Optional[str] = typer.Option(None, help="Optional metadata CSV (N rows); not required for built‑in rules"),
    sigma: Optional[str] = typer.Option(None, help="Optional σ.npy (N×B) for diagnostics overlays"),
    y: Optional[str] = typer.Option(None, help="Optional labels.npy (N×B) for GLL overlay"),
    device: str = typer.Option("cpu", help="Device: 'cpu' or 'cuda' (if available)"),
    dtype: str = typer.Option("float32", help="Torch dtype: float32, float64, etc."),
    topk: int = typer.Option(8, min=1, help="Top‑K influential bins per sample & rule"),
    seed: int = typer.Option(42, help="Random seed"),
    html: bool = typer.Option(False, help="Emit compact HTML report"),
    fft_n: Optional[int] = typer.Option(None, help="Override n_freq_keep for any fft_smooth rule"),
):
    """
    Compute symbolic influence maps (|∂L_sym/∂μ|) per rule and export plots/tables/report.
    """
    try:
        run_symbolic_influence(
            mu_path=mu,
            outdir=outdir,
            rules_path=rules,
            wavelengths_path=wavelengths,
            metadata_path=metadata,
            sigma_path=sigma,
            y_path=y,
            device=device,
            dtype=dtype,
            topk=topk,
            seed=seed,
            html_report=html,
            fft_n=fft_n,
        )
    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", style="err"))
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
