#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
configs/dat/ariel_toy_dataset.py

SpectraMind V50 — Toy/Debug Dataset Generator (FGS1 + AIRS)
===========================================================

Purpose
-------
Generate *deterministic* synthetic datasets compatible with the V50 pipeline:
  • Tiny "debug" slice for CI/self-test (fast, ~seconds)
  • Small "toy" set for local development (lightweight but richer)

Artifacts (configurable via CLI)
--------------------------------
Writes *packed* .pkl files (default) and/or optional .npy arrays:

  outdir/
    ├── train_debug.pkl or train.pkl
    ├── val_debug.pkl   or val.pkl
    ├── test_debug.pkl  or test.pkl
    ├── fgs1_train_debug.npy (optional)
    ├── airs_train_debug.npy (optional)
    ├── labels_debug.csv     (optional)
    ├── splits/
    │     ├── train_idx.npy
    │     ├── val_idx.npy
    │     └── test_idx.npy
    └── toy_manifest.json

Data shapes & semantics
-----------------------
• FGS1 time series cube: (N, T_fgs1, 32)        with T_fgs1=128 (toy) or 64 (debug)
• AIRS time×wavelength: (N, T_airs=32, BINS=283)
• Targets:
    mu:    (N, 283)  nonnegative spectral absorption depths
    sigma: (N, 283)  aleatoric uncertainties (small positive)
• IDs:     list[str], e.g. "toy_0000"
• Splits:  deterministic planet-holdout by seed

Usage examples
--------------
# Generate the tiny debug slice
python configs/dat/ariel_toy_dataset.py --mode debug --outdir data/debug --force

# Generate the toy set with both PKL and NPY outputs
python configs/dat/ariel_toy_dataset.py --mode toy --outdir data/toy --write-npy --write-pkl

Notes
-----
• No network calls; pure NumPy (+pandas optional for CSV labels).
• Deterministic under fixed --seed.
• Produces split indices, manifest JSON, and optional binmap 356→283 if requested.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import pandas as pd  # optional (used for labels.csv)
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    _HAS_PANDAS = False


# --------------------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class GenDims:
    n: int
    t_fgs1: int
    t_airs: int
    bins: int
    channels_fgs1: int = 32


@dataclasses.dataclass
class GenCfg:
    mode: str
    outdir: Path
    seed: int
    write_pkl: bool
    write_npy: bool
    write_csv_labels: bool
    save_binmap: bool
    train_name: str
    val_name: str
    test_name: str
    splits: Tuple[float, float, float]  # (train, val, test)
    dims: GenDims


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------------------
# Synthetic spectrum/time-series primitives
# --------------------------------------------------------------------------------------

def make_line_centers_and_widths(rng: np.random.Generator, bins: int, n_lines: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    centers = rng.choice(bins, size=n_lines, replace=False)
    widths = rng.uniform(1.5, 6.0, size=n_lines)
    return centers.astype(np.int32), widths.astype(np.float32)


def gauss_profile(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_mu_sigma(
    rng: np.random.Generator,
    bins: int,
    base_depth_ppm: float = 2500.0,
    depth_jitter_ppm: float = 1500.0,
    min_sigma: float = 0.002,
    max_sigma: float = 0.006,
    n_lines: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.arange(bins, dtype=np.float32)
    centers, widths = make_line_centers_and_widths(rng, bins, n_lines=n_lines)
    base = (base_depth_ppm + rng.uniform(-depth_jitter_ppm, depth_jitter_ppm)) * 1e-6
    base = float(max(base, 100e-6))
    mu = np.zeros(bins, dtype=np.float32)
    for c, w in zip(centers, widths):
        amp = base * rng.uniform(0.8, 1.4)
        mu += amp * gauss_profile(lam, float(c), float(w)).astype(np.float32)
    mu = np.clip(mu, 0.0, 0.02).astype(np.float32)

    noise = rng.uniform(-1.0, 1.0, size=bins).astype(np.float32)
    smooth = np.cumsum(noise)
    smooth -= smooth.min()
    smooth /= max(smooth.ptp(), 1e-6)
    sigma = min_sigma + (max_sigma - min_sigma) * smooth
    sigma = sigma.astype(np.float32)
    return mu, sigma


def transit_weight(t: np.ndarray, t0: float, width: float, scale: float = 1.0) -> np.ndarray:
    width = max(float(width), 1e-6)
    w = np.exp(-0.5 * ((t - t0) / width) ** 2)
    w = (w - w.min()) / max(w.ptp(), 1e-6)
    return (w * scale).astype(np.float32)


def low_freq_jitter(rng: np.random.Generator, length: int, max_amp: float = 0.0025, n_harm: int = 3) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, length, endpoint=False, dtype=np.float32)
    s = np.zeros(length, dtype=np.float32)
    for k in range(1, n_harm + 1):
        amp = (max_amp / k) * rng.uniform(0.5, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        s += amp * np.sin(k * t + phase).astype(np.float32)
    return s


# --------------------------------------------------------------------------------------
# Generate per-planet AIRS/FGS1
# --------------------------------------------------------------------------------------

def gen_airs_for_planet(
    rng: np.random.Generator,
    t_steps: int,
    bins: int,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    assert mu.shape == (bins,)
    assert sigma.shape == (bins,)

    t_grid = np.linspace(0.0, 1.0, t_steps, dtype=np.float32)
    t0 = float(rng.uniform(0.35, 0.65))
    width = float(rng.uniform(0.08, 0.18))
    scale = float(rng.uniform(0.9, 1.1))
    w = transit_weight(t_grid, t0, width, scale)

    jitter_t = low_freq_jitter(rng, t_steps, max_amp=0.0015, n_harm=2)

    base = 1.0 - np.outer(w, mu).astype(np.float32)
    base += jitter_t[:, None] * float(np.median(mu) * 0.4)

    noise = rng.normal(0.0, 1.0, size=(t_steps, bins)).astype(np.float32)
    noise *= (sigma[None, :] * rng.uniform(0.8, 1.2))
    base += noise

    return np.clip(base, 0.0, 2.0).astype(np.float32)


def gen_fgs1_for_planet(
    rng: np.random.Generator,
    t_fgs1: int,
    channels: int,
    mu: np.ndarray,
) -> np.ndarray:
    depth_mu = float(np.clip(mu.mean() * 0.6, 5e-4, 1.5e-2))
    t = np.linspace(0.0, 1.0, t_fgs1, dtype=np.float32)
    t0 = float(rng.uniform(0.35, 0.65))
    width = float(rng.uniform(0.08, 0.18))
    u = transit_weight(t, t0, width, depth_mu)
    base = 1.0 - u
    jitter = low_freq_jitter(rng, t_fgs1, max_amp=0.0025, n_harm=3)

    cube = np.zeros((t_fgs1, channels), dtype=np.float32)
    for c in range(channels):
        scale = rng.uniform(0.98, 1.02)
        ch = base * scale + 0.5 * jitter
        ch += rng.normal(0.0, 0.0015, size=t_fgs1).astype(np.float32)
        cube[:, c] = ch.astype(np.float32)

    return np.clip(cube, 0.0, 2.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Generation pipeline
# --------------------------------------------------------------------------------------

def compute_splits(
    rng: np.random.Generator, n: int, fr_train: float, fr_val: float, fr_test: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = fr_train + fr_val + fr_test
    if not (abs(s - 1.0) < 1e-6):
        raise ValueError(f"Split fractions must sum to 1.0; got {s:.6f}")
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_train = int(round(n * fr_train))
    n_val = int(round(n * fr_val))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def pack_dict(
    fgs1: np.ndarray,
    airs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    ids: np.ndarray,
    split: np.ndarray,
) -> dict:
    return {
        "fgs1": fgs1,    # (N, T_fgs1, 32)
        "airs": airs,    # (N, T_airs=32, BINS=283)
        "mu": mu,        # (N, 283)
        "sigma": sigma,  # (N, 283)
        "ids": ids,      # (N,)
        "split": split,  # (N,) 'train'/'val'/'test'
    }


def write_pkl(path: Path, obj: dict) -> None:
    import pickle
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def maybe_write_csv_labels(path: Path, ids: np.ndarray, split: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
    if not _HAS_PANDAS:
        warn("pandas not available; skipping labels CSV.")
        return
    mu_cols = [f"target_mu_{k}" for k in range(mu.shape[1])]
    sg_cols = [f"target_sigma_{k}" for k in range(sigma.shape[1])]
    df = pd.DataFrame({"planet_id": ids, "split": split})
    df_mu = pd.DataFrame(mu, columns=mu_cols)
    df_sg = pd.DataFrame(sigma, columns=sg_cols)
    df = pd.concat([df, df_mu, df_sg], axis=1)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def save_binmap(path: Path, source_bins: int = 356, target_bins: int = 283) -> None:
    """
    Simple deterministic 356→283 mapping (down-select indices uniformly).
    In real pipelines, you'd use the official instrument mapping. This is for smoke/toy only.
    """
    if target_bins > source_bins:
        raise ValueError("target_bins must be ≤ source_bins")
    # Uniformly sample indices from 0..source_bins-1
    idx = np.linspace(0, source_bins - 1, target_bins, dtype=np.int64)
    np.save(path, idx)


def generate(cfg: GenCfg) -> None:
    ensure_dir(cfg.outdir)
    splits_dir = cfg.outdir / "splits"
    ensure_dir(splits_dir)

    rng = rng_from_seed(cfg.seed)
    N = cfg.dims.n
    T_F = cfg.dims.t_fgs1
    T_A = cfg.dims.t_airs
    B = cfg.dims.bins
    CH = cfg.dims.channels_fgs1

    info(f"Generating {cfg.mode} dataset → {cfg.outdir} (N={N}, bins={B}, T_fgs1={T_F}, T_airs={T_A})")

    fgs1_all = np.zeros((N, T_F, CH), dtype=np.float32)
    airs_all = np.zeros((N, T_A, B), dtype=np.float32)
    mu_all = np.zeros((N, B), dtype=np.float32)
    sigma_all = np.zeros((N, B), dtype=np.float32)

    ids = np.array([f"{cfg.mode}_{i:04d}" for i in range(N)], dtype=object)

    for i in range(N):
        sub_rng = rng_from_seed(cfg.seed + 10000 * (i + 1))
        mu, sigma = build_mu_sigma(sub_rng, B)
        mu_all[i] = mu
        sigma_all[i] = sigma
        airs_all[i] = gen_airs_for_planet(sub_rng, T_A, B, mu, sigma)
        fgs1_all[i] = gen_fgs1_for_planet(sub_rng, T_F, CH, mu)

    # Splits
    train_frac, val_frac, test_frac = cfg.splits
    train_idx, val_idx, test_idx = compute_splits(rng, N, train_frac, val_frac, test_frac)
    np.save(splits_dir / "train_idx.npy", train_idx)
    np.save(splits_dir / "val_idx.npy", val_idx)
    np.save(splits_dir / "test_idx.npy", test_idx)

    split_arr = np.array(["train"] * N, dtype=object)
    split_arr[val_idx] = "val"
    split_arr[test_idx] = "test"

    # Pack partitions (by indices) into PKL files
    if cfg.write_pkl:
        # train
        d_train = pack_dict(
            fgs1_all[train_idx], airs_all[train_idx], mu_all[train_idx], sigma_all[train_idx],
            ids[train_idx], split_arr[train_idx],
        )
        write_pkl(cfg.outdir / cfg.train_name, d_train)

        # val
        d_val = pack_dict(
            fgs1_all[val_idx], airs_all[val_idx], mu_all[val_idx], sigma_all[val_idx],
            ids[val_idx], split_arr[val_idx],
        )
        write_pkl(cfg.outdir / cfg.val_name, d_val)

        # test
        d_test = pack_dict(
            fgs1_all[test_idx], airs_all[test_idx], mu_all[test_idx], sigma_all[test_idx],
            ids[test_idx], split_arr[test_idx],
        )
        write_pkl(cfg.outdir / cfg.test_name, d_test)

        info(f"Wrote PKL: {cfg.train_name}, {cfg.val_name}, {cfg.test_name}")

    # Optional NPY exports for smoke alignment with debug configs
    if cfg.write_npy:
        np.save(cfg.outdir / "fgs1_train.npy", fgs1_all[train_idx])
        np.save(cfg.outdir / "airs_train.npy", airs_all[train_idx])
        np.save(cfg.outdir / "labels.csv.npy", np.stack([train_idx], axis=0))  # placeholder marker
        np.save(cfg.outdir / "fgs1_test.npy", fgs1_all[test_idx])
        np.save(cfg.outdir / "airs_test.npy", airs_all[test_idx])
        info("Wrote NPY arrays for train/test.")

    # Optional labels CSV (planet_id, split, target_mu_*, target_sigma_*)
    if cfg.write_csv_labels:
        labels_csv = cfg.outdir / ("labels_debug.csv" if cfg.mode == "debug" else "labels.csv")
        maybe_write_csv_labels(labels_csv, ids, split_arr, mu_all, sigma_all)
        info(f"Wrote labels CSV → {labels_csv}")

    # Optional 356→283 binmap
    if cfg.save_binmap:
        map_path = cfg.outdir / (f"binmap_356_to_{B}_{cfg.mode}.npy")
        save_binmap(map_path, source_bins=356, target_bins=B)
        info(f"Wrote binmap → {map_path}")

    # Manifest for audits
    manifest = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "N": int(N),
        "T_fgs1": int(T_F),
        "T_airs": int(T_A),
        "bins": int(B),
        "channels_fgs1": int(CH),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "write_pkl": bool(cfg.write_pkl),
        "write_npy": bool(cfg.write_npy),
        "write_csv_labels": bool(cfg.write_csv_labels),
        "paths": {
            "outdir": str(cfg.outdir),
            "splits_dir": str(splits_dir),
            "train_pkl": str(cfg.outdir / cfg.train_name) if cfg.write_pkl else None,
            "val_pkl": str(cfg.outdir / cfg.val_name) if cfg.write_pkl else None,
            "test_pkl": str(cfg.outdir / cfg.test_name) if cfg.write_pkl else None,
        },
    }
    save_json(cfg.outdir / "toy_manifest.json", manifest)
    info("Toy/debug dataset generation complete.")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ariel_toy_dataset",
        description="Generate deterministic toy/debug datasets for SpectraMind V50.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["toy", "debug"], default="debug", help="Dataset preset.")
    p.add_argument("--outdir", type=str, default="data/debug", help="Output directory for artifacts.")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed.")
    p.add_argument("--write-pkl", action="store_true", help="Write packed .pkl files.")
    p.add_argument("--write-npy", action="store_true", help="Write .npy arrays for smoke tests.")
    p.add_argument("--write-csv-labels", action="store_true", help="Write labels CSV (requires pandas).")
    p.add_argument("--save-binmap", action="store_true", help="Write a simple 356→283 binmap.")
    p.add_argument("--force", action="store_true", help="Overwrite existing artifacts if present.")

    # Advanced knobs
    p.add_argument("--n", type=int, default=None, help="Override number of planets.")
    p.add_argument("--t-fgs1", type=int, default=None, help="Override FGS1 time length.")
    p.add_argument("--t-airs", type=int, default=None, help="Override AIRS time steps.")
    p.add_argument("--bins", type=int, default=None, help="Override number of bins (default 283).")
    p.add_argument("--train-name", type=str, default=None, help="Override train pkl filename.")
    p.add_argument("--val-name", type=str, default=None, help="Override val pkl filename.")
    p.add_argument("--test-name", type=str, default=None, help="Override test pkl filename.")
    p.add_argument("--train-frac", type=float, default=None, help="Train fraction.")
    p.add_argument("--val-frac", type=float, default=None, help="Val fraction.")
    p.add_argument("--test-frac", type=float, default=None, help="Test fraction.")
    return p.parse_args()


def make_cfg(args: argparse.Namespace) -> GenCfg:
    if args.mode == "debug":
        dims = GenDims(
            n=5 if args.n is None else args.n,
            t_fgs1=64 if args.t_fgs1 is None else args.t_fgs1,
            t_airs=32 if args.t_airs is None else args.t_airs,
            bins=283 if args.bins is None else args.bins,
            channels_fgs1=32,
        )
        train_name = args.train_name or "train_debug.pkl"
        val_name = args.val_name or "val_debug.pkl"
        test_name = args.test_name or "test_debug.pkl"
        splits = (
            0.8 if args.train_frac is None else args.train_frac,
            0.2 if args.val_frac is None else args.val_frac,
            0.0 if args.test_frac is None else args.test_frac,
        )
    else:  # toy
        dims = GenDims(
            n=64 if args.n is None else args.n,
            t_fgs1=128 if args.t_fgs1 is None else args.t_fgs1,
            t_airs=32 if args.t_airs is None else args.t_airs,
            bins=283 if args.bins is None else args.bins,
            channels_fgs1=32,
        )
        train_name = args.train_name or "train.pkl"
        val_name = args.val_name or "val.pkl"
        test_name = args.test_name or "test.pkl"
        splits = (
            0.8 if args.train_frac is None else args.train_frac,
            0.1 if args.val_frac is None else args.val_frac,
            0.1 if args.test_frac is None else args.test_frac,
        )

    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Invalid splits (must sum to 1.0): {splits}")

    cfg = GenCfg(
        mode=args.mode,
        outdir=Path(args.outdir),
        seed=int(args.seed),
        write_pkl=bool(args.write_pkl or not args.write_npy),  # default: write PKL if nothing else chosen
        write_npy=bool(args.write_npy),
        write_csv_labels=bool(args.write_csv_labels),
        save_binmap=bool(args.save_binmap),
        train_name=train_name,
        val_name=val_name,
        test_name=test_name,
        splits=splits,
        dims=dims,
    )
    return cfg


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)

    # Idempotency: skip if outputs exist (unless --force)
    if not args.force:
        maybe = [
            cfg.outdir / cfg.train_name,
            cfg.outdir / cfg.val_name,
            cfg.outdir / cfg.test_name,
        ]
        if any(p.exists() for p in maybe if cfg.write_pkl):
            warn("Outputs already exist; use --force to overwrite.")
            return

    generate(cfg)


if __name__ == "__main__":
    main()
