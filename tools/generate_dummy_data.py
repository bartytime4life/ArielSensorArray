#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_dummy_data.py

SpectraMind V50 — Ultimate Dummy Data Generator (challenge-grade, CLI-first)

Purpose
-------
Create a *fully wired* synthetic dataset that mirrors the shapes, file formats, and
diagnostics hooks used by SpectraMind V50 for the NeurIPS 2025 Ariel Data Challenge.

This generator is designed to unblock end-to-end pipeline development, CI, and
diagnostics without needing the real challenge data. It produces:

  • Spectral products (N × B):
      - y.npy         : "ground-truth" mean spectrum μ* per sample
      - mu.npy        : "predicted" mean spectrum with controllable noise/bias
      - sigma.npy     : per-bin predicted uncertainty (>0), correlated with SNR
      - shap_bins.npy : optional per-bin |SHAP| magnitudes (proxy)
      - latents.npy   : optional N × D latent vectors (diagnostics)

  • Axes / metadata:
      - wavelengths.npy : B-length wavelength axis (μm, default 0.9–5.0)
      - metadata.csv    : tabular per-sample metadata (planet/star/instrument knobs)

  • Symbolic & diagnostics:
      - symbolic_results.json : per-sample symbolic summary (violations_total, hints)
      - (optional) calibrated lightcurves HDF5 with simple FGS1/AIRS groups
        -> outputs formatted as: calibrated/lightcurves.h5 (see --write-h5)

  • Logs / summary:
      - summary.json   : generator configuration & quick stats
      - logs/v50_debug_log.md (append-only audit entry)

Design Principles
-----------------
• Reproducibility first: deterministic RNG seed; JSON summary of all inputs.
• Physics-informed shape: compose absorption bands (H2O/CO2/CH4-like) as Gaussians
  on a sloped continuum, add per-sample temperature / gravity trends and aerosol haze.
• Noise realism: heteroscedastic noise tied to SNR & stellar magnitude; σ follows.
• Symbolic viability: includes basic rule-like signals so our symbolic tools have
  something meaningful to chew on.
• Safety: no external deps beyond numpy/pandas/h5py (optional), typer, rich, plotly-kaleido (optional).

CLI Examples
------------
# Minimal, quick:
python -m tools.generate_dummy_data --outdir outputs/dummy --n 128 --b 283

# Heavier with latents, SHAP, HDF5, and JSON symbolic overlays:
python -m tools.generate_dummy_data \
  --outdir outputs/dummy \
  --n 256 --b 283 --latent-d 32 \
  --write-shap --write-h5 --seed 123

# Wide spectrum and noisier predictions:
python -m tools.generate_dummy_data \
  --outdir outputs/dummy_wide --b 356 \
  --lam-min 0.6 --lam-max 7.8 \
  --noise 0.0008 --bias 0.0002 --sigma-scale 1.2

File Layout (generated)
-----------------------
<outdir>/
  mu.npy
  sigma.npy
  y.npy
  wavelengths.npy
  latents.npy                 (optional)
  shap_bins.npy               (optional)
  metadata.csv
  symbolic_results.json
  summary.json
  calibrated/lightcurves.h5   (optional if --write-h5)
  plots/preview_spectra.png   (quick sanity plot, if --save-png)

Author
------
SpectraMind V50 — Architect & Master Programmer
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme

# Optional: HDF5 writer for simple lightcurves stub
try:
    import h5py  # type: ignore
    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

# Optional: PNG preview (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False, help="SpectraMind V50 — Dummy Data Generator")
console = Console(theme=Theme({"info": "cyan", "warn": "yellow", "err": "bold red"}))


# ============================================================
# Utilities / IO
# ============================================================

def ensure_dir(path: Path) -> None:
    """Create a directory (parents OK)."""
    path.mkdir(parents=True, exist_ok=True)


def write_npy(path: Path, arr: np.ndarray) -> None:
    """Atomic-ish write for .npy arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def append_audit_log(entry: str) -> None:
    """Append an audit log line to logs/v50_debug_log.md (best-effort)."""
    try:
        logs = Path("logs")
        logs.mkdir(parents=True, exist_ok=True)
        with open(logs / "v50_debug_log.md", "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n")
    except Exception:
        pass


def now_str() -> str:
    import datetime as dt
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_global_seed(seed: int) -> None:
    """Deterministic RNG for numpy (and Python stdlib if available)."""
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
# Spectral model (dummy but physics-inspired)
# ============================================================

@dataclass
class BandSpec:
    """Simple Gaussian absorption band specification."""
    center_um: float
    width_um: float
    depth_scale: float  # typical per-band depth scaling


@dataclass
class GenConfig:
    """All knobs for the generator (keep in summary.json)."""
    n: int = 128
    b: int = 283
    lam_min: float = 0.9
    lam_max: float = 5.0
    latent_d: int = 0
    noise: float = 6e-4         # additive mu noise stdev for prediction
    bias: float = 1.0e-4        # additive small bias on mu wrt y
    sigma_scale: float = 1.0     # multiplier for sigma baseline
    snr_floor: float = 50.0      # baseline SNR for bright stars (higher = lower sigma)
    snr_ceiling: float = 200.0   # upper SNR bound for the best cases
    shap_scale: float = 0.02     # scale factor for synthetic |SHAP| (relative)
    symbolic_noise: float = 0.3  # noise on symbolic violations
    haze_strength: float = 0.015 # amplitude of aerosol haze curvature
    # Band templates (H2O/CO2/CH4-like regions; configurable if needed)
    bands: Tuple[BandSpec, ...] = (
        BandSpec(center_um=1.40, width_um=0.08, depth_scale=0.035),  # H2O-ish
        BandSpec(center_um=1.90, width_um=0.10, depth_scale=0.025),  # H2O-ish
        BandSpec(center_um=2.00, width_um=0.06, depth_scale=0.020),  # CO2-ish
        BandSpec(center_um=3.30, width_um=0.10, depth_scale=0.030),  # CH4-ish
        BandSpec(center_um=4.30, width_um=0.08, depth_scale=0.018),  # CO2-ish
    )
    # Time-series (HDF5) stub sizes — NOT the real challenge sizes, just lightweight demo
    fgs1_time: int = 1200       # number of time samples for FGS1 lightcurve
    airs_time: int = 400        # number of time samples for AIRS single channel
    write_h5: bool = False
    write_shap: bool = False
    save_png: bool = False
    seed: int = 42


def make_wavelengths(b: int, lam_min: float, lam_max: float) -> np.ndarray:
    """Uniform wavelength grid (μm)."""
    return np.linspace(float(lam_min), float(lam_max), int(b), dtype=np.float64)


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / max(1e-9, sigma)) ** 2)


def synth_baseline(lam: np.ndarray, teff: float, haze_strength: float) -> np.ndarray:
    """
    Build a smooth baseline continuum:
      - slight slope driven by stellar Teff
      - very gentle curvature term for aerosol haze
      - returns around ~1.0 with small trends
    """
    x = (lam - lam.min()) / max(1e-9, (lam.max() - lam.min()))
    slope = (5800.0 - teff) / 5800.0 * 0.02     # cooler stars -> slightly steeper slope
    curve = haze_strength * (x - 0.5) ** 2
    base = 1.0 + slope * (x - 0.5) - curve
    return base


def synth_absorption(lam: np.ndarray, bands: Tuple[BandSpec, ...], abundances: np.ndarray) -> np.ndarray:
    """
    Compose Gaussian absorptions with per-molecule abundance scaling.
    abundances has length == len(bands); depths additively subtract from baseline.
    """
    y = np.zeros_like(lam, dtype=np.float64)
    for (bspec, a) in zip(bands, abundances):
        # Convert width_um (approx stddev) to sigma
        g = gaussian(lam, bspec.center_um, bspec.width_um)
        y += a * bspec.depth_scale * g
    return y


def synth_true_spectrum(
    lam: np.ndarray,
    teff: float,
    logg: float,
    haze_strength: float,
    bands: Tuple[BandSpec, ...],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Produce a "ground-truth" μ* using baseline + molecular absorptions.
    Abundances are drawn log-uniform-ish and modulated by gravity/temperature.
    """
    base = synth_baseline(lam, teff=teff, haze_strength=haze_strength)
    k = len(bands)
    # Random "molecular abundances" (unitless proxy), lightly tied to teff/logg
    # Lower teff → stronger H2O; higher logg → slightly weaker features
    raw = rng.lognormal(mean=-3.0, sigma=0.6, size=k)  # ~[0.01 .. 0.2] typical
    mod_teff = np.clip((6500.0 - teff) / 2500.0, 0.0, 1.0)  # cooler => up to +1
    mod_logg = np.clip((logg - 3.0) / 2.0, 0.0, 1.0)        # higher g => up to +1
    abund = raw * (1.0 + 0.8 * mod_teff) * (1.0 - 0.25 * mod_logg)
    # Compose absorption "depths" then subtract from baseline (transmission dip)
    depth = synth_absorption(lam, bands, abundances=abund)
    mu_star = base - depth
    # Enforce non-negativity and light clamp near [0, 1.2]
    mu_star = np.clip(mu_star, 0.0, 1.2)
    return mu_star, {
        "h2o_like": float(abund[0] + abund[1]),
        "co2_like": float(abund[2] + abund[4]),
        "ch4_like": float(abund[3]),
    }


def add_prediction_noise(
    y_true: np.ndarray,
    noise: float,
    bias: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Synthesize "predicted" μ by adding small bias + Gaussian noise; clamp to [0, 1.2].
    """
    mu_pred = y_true + bias + rng.normal(loc=0.0, scale=noise, size=y_true.shape)
    return np.clip(mu_pred, 0.0, 1.2)


def synth_sigma_from_snr(
    y_true: np.ndarray,
    snr: float,
    sigma_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build σ that loosely matches an SNR (larger SNR -> smaller sigma).
    Add light heteroscedastic variation across bins.
    """
    B = y_true.shape[-1]
    base = (1.0 / max(1.0, snr)) * sigma_scale
    trend = 0.6 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, B))  # gentle wiggle
    jitter = rng.uniform(0.9, 1.1, size=B)
    sigma = base * trend * jitter
    sigma = np.clip(sigma, 1e-6, None)
    return sigma.astype(np.float64)


def synth_symbolic_violations(
    lam: np.ndarray,
    mu_row: np.ndarray,
    y_row: np.ndarray,
    teff: float,
    logg: float,
    rng: np.random.Generator,
    cfg: GenConfig,
) -> Dict[str, Any]:
    """
    Very lightweight symbolic "violations_total" proxy so that symbolic tools
    don't see a constant zero vector. Heuristics:
      - Nonnegativity pass unless numerical jitter; we won't punish that here.
      - If band depth at ~1.4 μm (H2O-ish) is shallow, add small penalty.
      - If σ would be too small for given residual (mu - y), add penalty-like term.
    Returns a dict keyed for convenience by our dashboards.
    """
    # Depth around 1.4 μm (±0.05)
    mask = (lam >= 1.35) & (lam <= 1.45)
    if mask.sum() == 0:
        depth_14 = 0.0
    else:
        cont = float(np.mean(mu_row[~mask])) if (~mask).sum() > 0 else 1.0
        depth_14 = float(cont - np.mean(mu_row[mask]))

    # Residual energy (proxy for calibration failure)
    resid = float(np.mean(np.abs(mu_row - y_row)))

    # Temperature prior penalty (too hot with deep H2O)
    teff_pen = float(max(0.0, (teff - 6200.0) / 1000.0) * max(0.0, 0.02 - depth_14) * 100.0)
    # Residual penalty scaled
    resid_pen = 1000.0 * resid

    noise = rng.normal(0.0, cfg.symbolic_noise)
    total = max(0.0, 0.25 + teff_pen + resid_pen + noise)

    return {
        "violations_total": float(total),
        "depth_h2o_1p4": depth_14,
        "residual_abs_mean": resid,
        "star_teff": float(teff),
        "logg": float(logg),
    }


def synth_shap_bins(mu_row: np.ndarray, y_row: np.ndarray, lam: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """
    Produce a rough per-bin |SHAP| proxy:
      - larger where |mu - y| is large or spectral curvature is higher
      - smoothed and scaled to a friendly dynamic range
    """
    B = mu_row.shape[-1]
    resid = np.abs(mu_row - y_row)
    curv = np.zeros_like(mu_row)
    if B > 2:
        curv[1:-1] = np.abs(mu_row[2:] - 2 * mu_row[1:-1] + mu_row[:-2])
    proto = 0.6 * resid + 0.4 * curv
    # Smooth by simple convolution
    k = np.array([0.25, 0.5, 0.25])
    proto = np.convolve(proto, k, mode="same")
    # Scale and small noise
    out = scale * (proto / (proto.max() + 1e-9))
    out += rng.normal(0.0, scale * 0.03, size=B)
    return np.clip(out, 0.0, None).astype(np.float64)


def synth_metadata_row(i: int, teff: float, logg: float, snr: float, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Make a friendly metadata row per sample; extend as needed.
    """
    planet_id = f"PL_{i:04d}"
    star_mag = float(np.clip(10.0 - math.log10(max(1.0, snr)) * 2.5 + rng.normal(0, 0.2), 4.0, 14.0))
    inst_profile = rng.choice(["AIRS-CH0", "AIRS-CH1", "FGS1"], p=[0.45, 0.25, 0.30])
    return {
        "planet_id": planet_id,
        "star_teff": float(teff),
        "logg": float(logg),
        "snr_proxy": float(snr),
        "star_mag": star_mag,
        "instrument_profile": inst_profile,
    }


# ============================================================
# HDF5 lightcurves (optional stub)
# ============================================================

def write_lightcurves_h5(
    path: Path,
    lam: np.ndarray,
    mu: np.ndarray,
    rng: np.random.Generator,
    fgs1_time: int,
    airs_time: int,
) -> None:
    """
    Create a *lightweight* HDF5 file with stubbed groups:

      /FGS1/time        (T1,)
      /FGS1/cal         (T1,)      — synthetic white-light lightcurve
      /AIRS/time        (T2,)
      /AIRS/cal         (T2, B)    — synthetic spectroscopic lightcurves

    This is a demo scaffold, not a physically rigorous simulation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    T1 = int(fgs1_time)
    T2 = int(airs_time)
    B = lam.shape[0]

    t1 = np.linspace(-3.0, 3.0, T1, dtype=np.float64)
    t2 = np.linspace(-3.0, 3.0, T2, dtype=np.float64)

    # Transit-like dip model (toy)
    def transit(t, depth=0.005, width=0.8):
        return 1.0 - depth * np.exp(-0.5 * (t / max(1e-9, width)) ** 2)

    # White-light FGS1
    fgs1 = transit(t1) + rng.normal(0.0, 5e-4, size=T1)

    # AIRS spectroscopic (depth varies with wavelength band)
    # Use smaller depth near clear regions, larger where absorption templates hit.
    depth_vec = 0.003 + 0.004 * np.sin(2 * np.pi * (lam - lam.min()) / (lam.max() - lam.min()))
    airs = np.vstack([transit(t2, depth=float(d)) + rng.normal(0.0, 8e-4, size=T2) for d in depth_vec]).T

    with h5py.File(str(path), "w") as h:
        g1 = h.create_group("FGS1")
        g1.create_dataset("time", data=t1, compression="gzip")
        g1.create_dataset("cal", data=fgs1, compression="gzip")

        g2 = h.create_group("AIRS")
        g2.create_dataset("time", data=t2, compression="gzip")
        g2.create_dataset("cal", data=airs, compression="gzip")
        g2.create_dataset("wavelengths", data=lam, compression="gzip")


# ============================================================
# Main generation orchestration
# ============================================================

def generate_dummy_dataset(cfg: GenConfig, outdir: Path) -> Dict[str, Any]:
    """
    Generate a complete, pipeline-ready dummy dataset.

    Returns a summary dict that is also written to summary.json.
    """
    set_global_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Prepare outputs
    ensure_dir(outdir)
    ensure_dir(outdir / "plots")
    if cfg.write_h5:
        ensure_dir(outdir / "calibrated")

    # 1) Axes
    lam = make_wavelengths(cfg.b, cfg.lam_min, cfg.lam_max)  # (B,)
    write_npy(outdir / "wavelengths.npy", lam)

    # 2) Allocate arrays
    y = np.zeros((cfg.n, cfg.b), dtype=np.float64)
    mu = np.zeros_like(y)
    sigma = np.zeros_like(y)

    if cfg.latent_d > 0:
        latents = np.zeros((cfg.n, cfg.latent_d), dtype=np.float64)
    else:
        latents = None

    shap_bins = np.zeros_like(y) if cfg.write_shap else None

    # 3) Per-sample metadata and symbolic
    meta_rows: List[Dict[str, Any]] = []
    symbolic_list: List[Dict[str, Any]] = []

    # Sampling distributions for star/planet knobs
    teff_dist = lambda: float(rng.normal(5500.0, 500.0))     # Kelvin
    logg_dist = lambda: float(rng.normal(4.0, 0.3))          # cgs
    snr_dist  = lambda: float(rng.uniform(cfg.snr_floor, cfg.snr_ceiling))

    # 4) Generate rows
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Synthesizing samples", total=cfg.n)

        for i in range(cfg.n):
            teff = teff_dist()
            logg = np.clip(logg_dist(), 3.0, 5.0)
            snr  = snr_dist()

            # True spectrum
            y_i, molec = synth_true_spectrum(
                lam=lam,
                teff=teff,
                logg=logg,
                haze_strength=cfg.haze_strength,
                bands=cfg.bands,
                rng=rng,
            )

            # Predicted mu + sigma
            mu_i = add_prediction_noise(y_i, noise=cfg.noise, bias=cfg.bias, rng=rng)
            sig_i = synth_sigma_from_snr(y_i, snr=snr, sigma_scale=cfg.sigma_scale, rng=rng)

            # Store
            y[i, :] = y_i
            mu[i, :] = mu_i
            sigma[i, :] = sig_i

            # Latents (optional): encode coarse molecule/teff/logg info + noise
            if latents is not None:
                # A simplest "meaningful" latent: [teff_z, logg_z, h2o, co2, ch4, ...noise]
                teff_z = (teff - 5500.0) / 500.0
                logg_z = (logg - 4.0) / 0.3
                seed_lat = np.array([teff_z, logg_z, molec["h2o_like"], molec["co2_like"], molec["ch4_like"]], dtype=np.float64)
                if cfg.latent_d <= seed_lat.size:
                    latents[i, :] = seed_lat[:cfg.latent_d]
                else:
                    latents[i, :seed_lat.size] = seed_lat
                    latents[i, seed_lat.size:] = rng.normal(0.0, 0.33, size=cfg.latent_d - seed_lat.size)

            # SHAP proxy (optional)
            if shap_bins is not None:
                shap_bins[i, :] = synth_shap_bins(mu_row=mu_i, y_row=y_i, lam=lam, scale=cfg.shap_scale, rng=rng)

            # Symbolic summary
            sym = synth_symbolic_violations(lam=lam, mu_row=mu_i, y_row=y_i, teff=teff, logg=logg, rng=rng, cfg=cfg)
            symbolic_list.append(sym)

            # Metadata row
            meta_rows.append({**synth_metadata_row(i, teff, logg, snr, rng), **molec})

            progress.advance(task)

    # 5) Save arrays
    write_npy(outdir / "y.npy", y)
    write_npy(outdir / "mu.npy", mu)
    write_npy(outdir / "sigma.npy", sigma)
    if latents is not None:
        write_npy(outdir / "latents.npy", latents)
    if shap_bins is not None:
        write_npy(outdir / "shap_bins.npy", shap_bins)

    # 6) Save metadata & symbolic
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(outdir / "metadata.csv", index=False)
    with open(outdir / "symbolic_results.json", "w", encoding="utf-8") as f:
        json.dump(symbolic_list, f, indent=2)

    # 7) Optional HDF5 lightcurves
    if cfg.write_h5:
        if not _HAS_H5PY:
            console.print("[warn]h5py not installed; skipping HDF5 lightcurves.")
        else:
            write_lightcurves_h5(
                path=outdir / "calibrated" / "lightcurves.h5",
                lam=lam,
                mu=mu,
                rng=rng,
                fgs1_time=cfg.fgs1_time,
                airs_time=cfg.airs_time,
            )

    # 8) Optional quick PNG preview (few random spectra)
    if cfg.save_png:
        try:
            sel = np.random.default_rng(cfg.seed + 1).choice(cfg.n, size=min(8, cfg.n), replace=False)
            plt.figure(figsize=(10, 4), dpi=120)
            for j in sel:
                plt.plot(lam, y[j], alpha=0.85, lw=1.2)
            plt.title("Dummy ground-truth μ* (random subset)")
            plt.xlabel("Wavelength (μm)")
            plt.ylabel("μ*")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(outdir / "plots" / "preview_spectra.png")
            plt.close()
        except Exception:
            console.print("[warn]Failed to save preview PNG; continuing.")

    # 9) Build summary
    summary: Dict[str, Any] = {
        "timestamp": now_str(),
        "outdir": str(outdir),
        "config": asdict(cfg),
        "shapes": {
            "mu": list(mu.shape),
            "sigma": list(sigma.shape),
            "y": list(y.shape),
            "wavelengths": list(lam.shape),
            "latents": list(latents.shape) if latents is not None else None,
            "shap_bins": list(shap_bins.shape) if shap_bins is not None else None,
        },
        "stats": {
            "mu_min": float(mu.min()),
            "mu_max": float(mu.max()),
            "sigma_mean": float(sigma.mean()),
            "y_mean": float(y.mean()),
            "symbolic_mean": float(np.mean([d["violations_total"] for d in symbolic_list])),
        },
        "artifacts": {
            "mu.npy": "mu.npy",
            "sigma.npy": "sigma.npy",
            "y.npy": "y.npy",
            "wavelengths.npy": "wavelengths.npy",
            "latents.npy": "latents.npy" if latents is not None else None,
            "shap_bins.npy": "shap_bins.npy" if shap_bins is not None else None,
            "metadata.csv": "metadata.csv",
            "symbolic_results.json": "symbolic_results.json",
            "calibrated/lightcurves.h5": ("calibrated/lightcurves.h5" if (cfg.write_h5 and _HAS_H5PY) else None),
            "plots/preview_spectra.png": ("plots/preview_spectra.png" if cfg.save_png else None),
        },
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 10) Audit log
    append_audit_log(f"- {now_str()} | generate_dummy_data | out={outdir} N={cfg.n} B={cfg.b} seed={cfg.seed} latents={cfg.latent_d} shap={cfg.write_shap} h5={cfg.write_h5}")

    return summary


# ============================================================
# Typer CLI
# ============================================================

@app.command("run")
def cli_run(
    outdir: str = typer.Option(..., help="Output directory for the dummy dataset"),
    n: int = typer.Option(128, help="Number of samples (planets)"),
    b: int = typer.Option(283, help="Number of spectral bins"),
    lam_min: float = typer.Option(0.9, help="Minimum wavelength (μm)"),
    lam_max: float = typer.Option(5.0, help="Maximum wavelength (μm)"),
    latent_d: int = typer.Option(0, help="Latent embedding dimensionality (0 to disable)"),
    noise: float = typer.Option(6e-4, help="Prediction noise stdev for μ"),
    bias: float = typer.Option(1e-4, help="Small additive bias on μ vs y"),
    sigma_scale: float = typer.Option(1.0, help="Multiplier for σ baseline"),
    snr_floor: float = typer.Option(50.0, help="Lower SNR bound (bigger → smaller σ)"),
    snr_ceiling: float = typer.Option(200.0, help="Upper SNR bound"),
    shap_scale: float = typer.Option(0.02, help="Scale factor for synthetic per-bin |SHAP|"),
    symbolic_noise: float = typer.Option(0.3, help="Noise on symbolic violations_total"),
    haze_strength: float = typer.Option(0.015, help="Aerosol haze curvature amplitude"),
    write_h5: bool = typer.Option(False, help="Also write a small HDF5 lightcurves file"),
    write_shap: bool = typer.Option(False, help="Also write shap_bins.npy (proxy magnitudes)"),
    save_png: bool = typer.Option(False, help="Save quick preview PNG of a few ground-truth spectra"),
    fgs1_time: int = typer.Option(1200, help="FGS1 time samples in HDF5 (stub)"),
    airs_time: int = typer.Option(400, help="AIRS time samples in HDF5 (stub)"),
    seed: int = typer.Option(42, help="RNG seed"),
):
    """
    Generate a complete, pipeline-ready dummy dataset under --outdir.

    Artifacts: mu.npy, sigma.npy, y.npy, wavelengths.npy, metadata.csv,
               symbolic_results.json, summary.json, (optional) latents.npy,
               shap_bins.npy, calibrated/lightcurves.h5, preview PNG.
    """
    try:
        cfg = GenConfig(
            n=int(n),
            b=int(b),
            lam_min=float(lam_min),
            lam_max=float(lam_max),
            latent_d=int(latent_d),
            noise=float(noise),
            bias=float(bias),
            sigma_scale=float(sigma_scale),
            snr_floor=float(snr_floor),
            snr_ceiling=float(snr_ceiling),
            shap_scale=float(shap_scale),
            symbolic_noise=float(symbolic_noise),
            haze_strength=float(haze_strength),
            write_h5=bool(write_h5),
            write_shap=bool(write_shap),
            save_png=bool(save_png),
            fgs1_time=int(fgs1_time),
            airs_time=int(airs_time),
            seed=int(seed),
        )

        console.rule("[info]SpectraMind V50 — Dummy Data Generator")
        console.print(f"[info]Outdir: {outdir}")
        console.print(f"[info]N={cfg.n}, B={cfg.b}, λ∈[{cfg.lam_min},{cfg.lam_max}] μm, seed={cfg.seed}")
        if cfg.latent_d > 0:
            console.print(f"[info]Latents: D={cfg.latent_d}")
        if cfg.write_shap:
            console.print("[info]Writing SHAP proxy: shap_bins.npy")
        if cfg.write_h5:
            console.print("[info]Writing HDF5 stub: calibrated/lightcurves.h5")

        summary = generate_dummy_dataset(cfg, Path(outdir))
        console.rule("[info]Done")
        console.print(f"[info]Artifacts written to: {outdir}")
        console.print(Panel.fit(json.dumps(summary["shapes"], indent=2), title="Shapes", subtitle="(quick view)"))

    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", style="err"))
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
