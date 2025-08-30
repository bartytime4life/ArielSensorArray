#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectramind/data/ariel_toy_dataset.py

SpectraMind V50 — Toy Dataset Generator (FGS1 + AIRS)
=====================================================
Purpose
-------
Generate a *minimal, deterministic* synthetic dataset compatible with the
`configs/data/toy.yaml` you provided:
  • Numpy arrays:
      - FGS1 cube: shape (N, fgs1_len, 32)   →  data/raw/toy/fgs1_toy.npy
      - AIRS cube: shape (N, 32, 283)        →  data/raw/toy/airs_toy.npy
  • Labels CSV:
      - Columns: planet_id, split, target_mu_0..282, target_sigma_0..282
      → data/raw/toy/labels_toy.csv
  • Split indices (reproducible):
      - {train,val,test}_idx.npy in data/processed/toy/splits/

Design Goals
------------
  • Deterministic: fixed `seed` controls RNG across all arrays.
  • Fast: small shapes for CI smoke tests (< 2 minutes end-to-end).
  • Physically plausible signals (toy level):
      - AIRS: time×wavelength cube constructed from a per-planet μ(λ) with a
        simple transit time profile w(t), and Gaussian noise + mild jitter.
      - FGS1: photometric time series replicated across 32 channels with small
        per-channel variation, showing a transit dip correlated with AIRS timing.
  • Hydra-friendly: accepts a path to your YAML (same fields as configs/data/toy.yaml).
  • Idempotent: skips generation if all files exist (unless --force).

Usage
-----
# Most convenient: from your repo root (reads the toy config)
python -m spectramind.data.ariel_toy_dataset --config configs/data/toy.yaml

# Force regeneration (overwrites existing)
python -m spectramind.data.ariel_toy_dataset --config configs/data/toy.yaml --force

# Show help
python -m spectramind.data.ariel_toy_dataset --help

Notes
-----
This module does not depend on Hydra at runtime; it just reads the YAML so it
works in CI and simple shells. The generated files are small and CI/Kaggle-safe.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from datetime import datetime


# --------------------------------------------------------------------------------------
# Dataclasses for a minimal, typed view of the config we care about (robust to extras)
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class PathsCfg:
    raw_dir: str
    processed_dir: str
    fgs1_file: str
    airs_file: str
    labels_file: str
    submission_template: str


@dataclasses.dataclass
class GenerationCfg:
    n_samples: int
    fgs1_len: int
    bins: int
    with_targets: bool
    seed: int


@dataclasses.dataclass
class SplitsCfg:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    stratify: bool
    seed: int
    export_paths: Dict[str, str]  # expects keys: indices_dir


@dataclasses.dataclass
class DatasetCfg:
    paths: PathsCfg
    generation: GenerationCfg
    splits: SplitsCfg


# --------------------------------------------------------------------------------------
# Utility: pretty logging (print-only to avoid extra deps)
# --------------------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def info(msg: str) -> None:
    print(f"[{_now()}] [INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[{_now()}] [WARN] {msg}")


def err(msg: str) -> None:
    print(f"[{_now()}] [ERROR] {msg}")


# --------------------------------------------------------------------------------------
# YAML loader → dataclasses (tolerant to extra keys)
# --------------------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select(cfg: dict, key_path: str, default=None):
    """
    Safely select nested keys via dotted path, e.g. "paths.raw_dir".
    Returns `default` if any key along the path is missing.
    """
    cur = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _coerce_dataset_cfg(cfg: dict) -> DatasetCfg:
    # Paths
    paths = PathsCfg(
        raw_dir=_select(cfg, "paths.raw_dir"),
        processed_dir=_select(cfg, "paths.processed_dir"),
        fgs1_file=_select(cfg, "paths.fgs1_file"),
        airs_file=_select(cfg, "paths.airs_file"),
        labels_file=_select(cfg, "paths.labels_file"),
        submission_template=_select(cfg, "paths.submission_template"),
    )
    # Generation
    generation = GenerationCfg(
        n_samples=int(_select(cfg, "generation.n_samples")),
        fgs1_len=int(_select(cfg, "generation.fgs1_len")),
        bins=int(_select(cfg, "generation.bins")),
        with_targets=bool(_select(cfg, "generation.with_targets")),
        seed=int(_select(cfg, "generation.seed")),
    )
    # Splits
    splits = SplitsCfg(
        train_fraction=float(_select(cfg, "splits.train_fraction")),
        val_fraction=float(_select(cfg, "splits.val_fraction")),
        test_fraction=float(_select(cfg, "splits.test_fraction")),
        stratify=bool(_select(cfg, "splits.stratify")),
        seed=int(_select(cfg, "splits.seed")),
        export_paths=dict(_select(cfg, "splits.export_paths")),
    )
    return DatasetCfg(paths=paths, generation=generation, splits=splits)


# --------------------------------------------------------------------------------------
# Synthetic signal primitives (fast, deterministic)
# --------------------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    """Create a NumPy Generator from a seed for reproducibility."""
    return np.random.default_rng(seed)


def _make_molecular_lines(rng: np.random.Generator, bins: int, n_lines: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random spectral line centers (in bin index) and widths (stddev in bins).
    This emulates several absorption features over the wavelength axis.
    """
    centers = rng.choice(bins, size=n_lines, replace=False)
    # Wider features at longer wavelengths just for variety; keep widths modest.
    widths = rng.uniform(1.5, 6.0, size=n_lines)
    return centers.astype(np.int32), widths.astype(np.float32)


def _gaussian_profile(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute a normalized Gaussian profile over x with center mu and std sigma."""
    return np.exp(-0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2)


def _build_mu_spectrum(
    rng: np.random.Generator,
    bins: int,
    base_depth_ppm: float = 2500.0,
    depth_jitter_ppm: float = 1500.0,
    n_lines: int = 6,
) -> np.ndarray:
    """
    Build a nonnegative target μ(λ) for AIRS [0..283).
    Toy approach: sum of a few Gaussian absorption features scaled by small depths.
    Returns shape (bins,) float32 with values around ~few × 10^-3.
    """
    lam = np.arange(bins, dtype=np.float32)
    centers, widths = _make_molecular_lines(rng, bins, n_lines=n_lines)

    # Depth scale in fractional units (ppm → fraction)
    base = (base_depth_ppm + rng.uniform(-depth_jitter_ppm, depth_jitter_ppm)) * 1e-6
    base = float(max(base, 100e-6))  # floor to avoid zero-ish depths

    mu = np.zeros(bins, dtype=np.float32)
    for c, w in zip(centers, widths):
        line_amp = base * rng.uniform(0.8, 1.4)  # small per-line variation
        mu += line_amp * _gaussian_profile(lam, float(c), float(w)).astype(np.float32)

    # Clip to ensure nonnegativity and reasonable upper bound (toy)
    mu = np.clip(mu, 0.0, 0.02).astype(np.float32)  # <= 2% depth
    return mu


def _build_sigma_spectrum(
    rng: np.random.Generator,
    bins: int,
    min_sigma: float = 0.002,
    max_sigma: float = 0.006,
) -> np.ndarray:
    """
    Build a per-bin σ(λ) (aleatoric uncertainty) with a gentle random structure.
    """
    # Smooth random field via cumulative filter
    noise = rng.uniform(-1.0, 1.0, size=bins).astype(np.float32)
    smooth = np.cumsum(noise)
    smooth -= smooth.min()
    smooth /= max(smooth.ptp(), 1e-6)
    sigma = min_sigma + (max_sigma - min_sigma) * smooth
    sigma = sigma.astype(np.float32)
    return sigma


def _transit_weight_profile(
    t: np.ndarray,
    t0: float,
    width: float,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    A simple, smooth transit weight 0..1 over time t (shape (T,)):
      - 0 outside transit
      - peaks near t0 by a smooth 'U' (inverted Gaussian-like)
    We model w(t) ∈ [0, 1] that multiplies μ(λ) to modulate AIRS.
    """
    # Smooth 'U' profile: 1.0 at center, tapering to ~0 away
    w = np.exp(-0.5 * ((t - t0) / max(width, 1e-6)) ** 2)
    # Normalize to [0,1]
    w = (w - w.min()) / max(w.ptp(), 1e-6)
    return (w * depth_scale).astype(np.float32)


def _low_freq_jitter(
    rng: np.random.Generator,
    length: int,
    max_amp: float = 0.0025,
    n_harm: int = 3,
) -> np.ndarray:
    """
    Create a small, smooth jitter time-series to simulate instrument variations.
    Summation of a few low-frequency sinusoids with random phases.
    """
    t = np.linspace(0, 2 * np.pi, length, endpoint=False, dtype=np.float32)
    series = np.zeros(length, dtype=np.float32)
    for k in range(1, n_harm + 1):
        amp = (max_amp / k) * rng.uniform(0.5, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        series += amp * np.sin(k * t + phase).astype(np.float32)
    return series


# --------------------------------------------------------------------------------------
# Core generators for AIRS and FGS1 cubes
# --------------------------------------------------------------------------------------

def _generate_airs_cube_for_planet(
    rng: np.random.Generator,
    T: int,
    bins: int,
    mu: np.ndarray,
    per_bin_sigma: np.ndarray,
) -> np.ndarray:
    """
    AIRS: (T=32, bins=283) cube for a single planet.
    Flux(t, λ) = 1 - w(t) * μ(λ) + ε(t, λ)
      • w(t): smooth transit profile in [0,1]
      • ε: Gaussian noise with std drawn from per_bin_sigma, plus slight time jitter coupling
    """
    assert mu.shape == (bins,), "mu must be (bins,)"
    assert per_bin_sigma.shape == (bins,), "sigma must be (bins,)"

    # Time grid in arbitrary units [0..1]
    t_grid = np.linspace(0.0, 1.0, T, dtype=np.float32)

    # Transit parameters per planet (deterministic draws)
    t0 = float(rng.uniform(0.35, 0.65))            # transit center
    width = float(rng.uniform(0.08, 0.18))         # transit width
    depth_scale = float(rng.uniform(0.9, 1.1))     # slight amplitude scaling
    w = _transit_weight_profile(t_grid, t0=t0, width=width, depth_scale=depth_scale)  # (T,)

    # Jitter across time for AIRS (same for all bins, small amplitude)
    jitter_t = _low_freq_jitter(rng, T, max_amp=0.0015, n_harm=2)  # (T,)

    # Broadcast w(t) * mu(λ) to T×bins and add noise
    base = 1.0 - np.outer(w, mu).astype(np.float32)               # (T, bins)
    # Add a weak time jitter that scales with median(mu) to keep magnitude bounded
    base += jitter_t[:, None] * float(np.median(mu) * 0.4)

    # Per-bin noise scaled by sigma(λ); also slight time modulation
    # Keep noise small to preserve transit visibility
    noise = rng.normal(loc=0.0, scale=1.0, size=(T, bins)).astype(np.float32)
    noise *= (per_bin_sigma[None, :] * rng.uniform(0.8, 1.2))     # scale around sigma
    base += noise

    # Clip to nonnegative flux toy range
    base = np.clip(base, 0.0, 2.0).astype(np.float32)
    return base  # (T, bins)


def _generate_fgs1_cube_for_planet(
    rng: np.random.Generator,
    fgs1_len: int,
    channels: int,
    mu: np.ndarray,
) -> np.ndarray:
    """
    FGS1: (fgs1_len, channels=32) time series 'images' for one planet.
    We create a transit-like dip correlated with AIRS timing (loosely) for toy consistency.

    Approach:
      - Build a time-domain transit profile u(t) with random center/width/depth from μ summary.
      - Replicate across channels with slight per-channel variation + noise + low-freq jitter.
    """
    # Summarize μ(λ) to pick an overall depth in photometry (scaled down)
    depth_mu = float(np.clip(mu.mean() * 0.6, 5e-4, 1.5e-2))  # typical ~ 0.001..0.01

    t = np.linspace(0.0, 1.0, fgs1_len, dtype=np.float32)
    t0 = float(rng.uniform(0.35, 0.65))
    width = float(rng.uniform(0.08, 0.18))

    # Smooth U-shaped dip in photometry: u(t) ∈ [0, depth_mu]
    u = _transit_weight_profile(t, t0=t0, width=width, depth_scale=depth_mu)  # 0..depth_mu
    # Convert to flux: 1 - u
    base = 1.0 - u

    # Jitter shared across channels
    jitter = _low_freq_jitter(rng, fgs1_len, max_amp=0.0025, n_harm=3)

    # Channel replication with slight per-channel scale and noise
    cube = np.zeros((fgs1_len, channels), dtype=np.float32)
    for c in range(channels):
        scale = rng.uniform(0.98, 1.02)  # mild gain differences per channel
        ch = base * scale + 0.5 * jitter
        # Add small white noise
        ch += rng.normal(0.0, 0.0015, size=fgs1_len).astype(np.float32)
        cube[:, c] = ch.astype(np.float32)

    cube = np.clip(cube, 0.0, 2.0).astype(np.float32)
    return cube  # (fgs1_len, channels)


# --------------------------------------------------------------------------------------
# Split helpers (reproducible)
# --------------------------------------------------------------------------------------

def _make_splits(
    rng: np.random.Generator,
    n_samples: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic split indices (train/val/test) by fractions. Fractions must sum to 1 within tolerance.
    """
    fr_sum = train_fraction + val_fraction + test_fraction
    if not (abs(fr_sum - 1.0) < 1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {fr_sum:.6f}")

    all_idx = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(all_idx)

    n_train = int(round(n_samples * train_fraction))
    n_val = int(round(n_samples * val_fraction))
    n_test = n_samples - n_train - n_val

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------------------
# Generation pipeline
# --------------------------------------------------------------------------------------

def generate_toy(cfg: DatasetCfg, force: bool = False) -> None:
    """
    Main generation routine: creates AIRS/FGS1 cubes, labels CSV, and split indices.
    Skips generation if all outputs exist (unless force=True).
    """
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    fgs1_file = Path(cfg.paths.fgs1_file)
    airs_file = Path(cfg.paths.airs_file)
    labels_file = Path(cfg.paths.labels_file)
    splits_dir = Path(cfg.splits.export_paths["indices_dir"])

    # Ensure directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Skip if everything exists (idempotent)
    if (fgs1_file.exists() and airs_file.exists() and labels_file.exists()
            and (splits_dir / "train_idx.npy").exists()
            and (splits_dir / "val_idx.npy").exists()
            and (splits_dir / "test_idx.npy").exists()
            and not force):
        info("Toy dataset already exists. Use --force to regenerate.")
        return

    info("Generating SpectraMind V50 toy dataset ...")

    # Deterministic RNG
    rng = _rng(cfg.generation.seed)

    # Shapes from config
    N = cfg.generation.n_samples
    T_fgs1 = cfg.generation.fgs1_len
    T_airs = 32                    # fixed per toy schema
    BINS = cfg.generation.bins
    CH_FGS1 = 32                   # fixed per toy schema

    # Pre-allocate
    airs_all = np.zeros((N, T_airs, BINS), dtype=np.float32)
    fgs1_all = np.zeros((N, T_fgs1, CH_FGS1), dtype=np.float32)
    mu_all = np.zeros((N, BINS), dtype=np.float32)
    sigma_all = np.zeros((N, BINS), dtype=np.float32)

    # Per-planet generation
    for i in range(N):
        # Derive a per-planet RNG (deterministic, independent)
        sub_rng = _rng(cfg.generation.seed + 10_000 * (i + 1))

        mu = _build_mu_spectrum(sub_rng, BINS)
        sigma = _build_sigma_spectrum(sub_rng, BINS)
        mu_all[i] = mu
        sigma_all[i] = sigma

        airs = _generate_airs_cube_for_planet(sub_rng, T=T_airs, bins=BINS, mu=mu, per_bin_sigma=sigma)
        fgs1 = _generate_fgs1_cube_for_planet(sub_rng, fgs1_len=T_fgs1, channels=CH_FGS1, mu=mu)
        airs_all[i] = airs
        fgs1_all[i] = fgs1

    # Save arrays
    np.save(airs_file, airs_all)
    np.save(fgs1_file, fgs1_all)
    info(f"Saved AIRS cube → {airs_file}  shape={tuple(airs_all.shape)}  dtype={airs_all.dtype}")
    info(f"Saved FGS1 cube → {fgs1_file}  shape={tuple(fgs1_all.shape)}  dtype={fgs1_all.dtype}")

    # Create splits
    split_rng = _rng(cfg.splits.seed)
    train_idx, val_idx, test_idx = _make_splits(
        split_rng,
        n_samples=N,
        train_fraction=cfg.splits.train_fraction,
        val_fraction=cfg.splits.val_fraction,
        test_fraction=cfg.splits.test_fraction,
    )
    np.save(splits_dir / "train_idx.npy", train_idx)
    np.save(splits_dir / "val_idx.npy", val_idx)
    np.save(splits_dir / "test_idx.npy", test_idx)
    info(f"Saved split indices → {splits_dir}")

    # Build labels CSV with planet_id + split + per-bin μ/σ columns
    planet_ids = [f"toy_{i:04d}" for i in range(N)]
    split_labels = np.array(["train"] * N, dtype=object)
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    # Column names
    mu_cols = [f"target_mu_{k}" for k in range(BINS)]
    sg_cols = [f"target_sigma_{k}" for k in range(BINS)]

    # Assemble DataFrame
    df = pd.DataFrame({"planet_id": planet_ids, "split": split_labels})
    df_mu = pd.DataFrame(mu_all, columns=mu_cols)
    df_sg = pd.DataFrame(sigma_all, columns=sg_cols)
    df = pd.concat([df, df_mu, df_sg], axis=1)

    # Save CSV (explicit quoting for safety)
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(labels_file, index=False, quoting=csv.QUOTE_MINIMAL)
    info(f"Saved labels CSV → {labels_file}  rows={len(df)}  cols={len(df.columns)}")

    # Write a small manifest (optional, handy for audits)
    manif
