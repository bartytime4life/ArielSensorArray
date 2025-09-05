#!/usr/bin/env python3

-- coding: utf-8 --

“””
configs/dat/ariel_toy_dataset.py

SpectraMind V50 — Toy/Debug Dataset Generator (FGS1 + AIRS)

Purpose

Generate deterministic synthetic datasets compatible with the V50 pipeline:
• Tiny “debug” slice for CI/self-test (fast, ~seconds)
• Small “toy” set for local development (lightweight but richer)

Key upgrades in this version

• Deterministic RNG routing per-planet + manifest hashing (SHA256) for audit
• Backward-compatible flags with a new –format switch (pkl|npy|both)
• Strong shape validation, guardrails (nonnegativity/finite checks), and error messages
• Optional CSV labels (pandas) + CSV split index exports + NPZ packing for arrays
• Idempotent writes with –force and per-file overwrite checks
• Clear, single-pass generation (low memory churn), progress logging, and structured manifest

Artifacts (configurable via CLI)

Writes packed .pkl files (default) and/or optional .npy/.npz arrays:

outdir/
├── train_debug.pkl or train.pkl
├── val_debug.pkl   or val.pkl
├── test_debug.pkl  or test.pkl
├── fgs1_train_debug.npz (optional; contains array ‘fgs1’)
├── airs_train_debug.npz (optional; contains array ‘airs’)
├── labels_debug.csv     (optional)
├── splits/
│     ├── train_idx.npy
│     ├── val_idx.npy
│     ├── test_idx.npy
│     ├── train_idx.csv
│     ├── val_idx.csv
│     └── test_idx.csv
└── toy_manifest.json

Data shapes & semantics

• FGS1 time series cube: (N, T_fgs1, 32)        with T_fgs1=128 (toy) or 64 (debug)
• AIRS time×wavelength: (N, T_airs=32, BINS=283)
• Targets:
mu:    (N, 283)  nonnegative spectral absorption depths
sigma: (N, 283)  aleatoric uncertainties (small positive)
• IDs:     list[str], e.g. “toy_0000”
• Splits:  deterministic planet-holdout by seed

Usage examples

Generate the tiny debug slice

python configs/dat/ariel_toy_dataset.py –mode debug –outdir data/debug –force

Generate the toy set with both PKL and NPY outputs

python configs/dat/ariel_toy_dataset.py –mode toy –outdir data/toy –format both

Generate toy set with CSV labels and 356→283 binmap

python configs/dat/ariel_toy_dataset.py –mode toy –outdir data/toy –write-csv-labels –save-binmap –force

Notes

• No network calls; pure NumPy (+pandas optional for CSV labels).
• Deterministic under fixed –seed.
• Produces split indices, manifest JSON (with hashes), and optional binmap 356→283 if requested.
“””

from future import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np

try:
import pandas as pd  # optional (used for labels.csv)
_HAS_PANDAS = True
except Exception:  # pragma: no cover
_HAS_PANDAS = False

–––––––––––––––––––––––––––––––––––––––––––

Metadata & versioning

–––––––––––––––––––––––––––––––––––––––––––

VERSION = “1.2.0”  # bump on functional change

–––––––––––––––––––––––––––––––––––––––––––

Config dataclasses

–––––––––––––––––––––––––––––––––––––––––––

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
id_prefix: str
id_pad: int
array_format: str  # ‘npy’ or ‘npz’ for array exports

–––––––––––––––––––––––––––––––––––––––––––

Utilities

–––––––––––––––––––––––––––––––––––––––––––

def rng_from_seed(seed: int) -> np.random.Generator:
“”“Create a NumPy Generator deterministically from a seed.”””
return np.random.default_rng(int(seed))

def ensure_dir(p: Path) -> None:
“”“Create directory if missing (parents OK).”””
p.mkdir(parents=True, exist_ok=True)

def info(msg: str) -> None:
“”“Informational log.”””
print(f”[INFO] {msg}”)

def warn(msg: str) -> None:
“”“Warning log.”””
print(f”[WARN] {msg}”)

def err(msg: str) -> None:
“”“Error log.”””
print(f”[ERROR] {msg}”)

def save_json(path: Path, obj: dict) -> None:
“”“Write a JSON file with UTF-8 encoding and indentation.”””
with path.open(“w”, encoding=“utf-8”) as f:
json.dump(obj, f, indent=2)

def sha256_of_file(path: Path) -> Optional[str]:
“”“Compute SHA256 of a file, returning None if it doesn’t exist.”””
if not path.exists() or not path.is_file():
return None
h = hashlib.sha256()
with path.open(“rb”) as f:
for chunk in iter(lambda: f.read(1024 * 1024), b””):
h.update(chunk)
return h.hexdigest()

def write_csv_indices(path: Path, idx: np.ndarray) -> None:
“”“Write a simple CSV with one index per row.”””
with path.open(“w”, newline=””, encoding=“utf-8”) as f:
writer = csv.writer(f)
writer.writerow([“index”])
for v in idx.tolist():
writer.writerow([int(v)])

–––––––––––––––––––––––––––––––––––––––––––

Synthetic spectrum/time-series primitives

–––––––––––––––––––––––––––––––––––––––––––

def make_line_centers_and_widths(rng: np.random.Generator, bins: int, n_lines: int = 6) -> Tuple[np.ndarray, np.ndarray]:
“”“Choose distinct centers and random Gaussian widths for spectral lines.”””
centers = rng.choice(bins, size=n_lines, replace=False)
widths = rng.uniform(1.5, 6.0, size=n_lines)
return centers.astype(np.int32), widths.astype(np.float32)

def gauss_profile(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
“”“Stable Gaussian profile (σ clamped).”””
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
“””
Build a toy μ(λ) with several Gaussian absorption features and a smoothly varying σ(λ).
μ is clipped to a small depth range; σ varies monotonically after normalization.
“””
lam = np.arange(bins, dtype=np.float32)
centers, widths = make_line_centers_and_widths(rng, bins, n_lines=n_lines)

# Base line depth in fraction (ppm → fraction)
base = (base_depth_ppm + rng.uniform(-depth_jitter_ppm, depth_jitter_ppm)) * 1e-6
base = float(max(base, 100e-6))  # floor to avoid degenerate all-zero lines

mu = np.zeros(bins, dtype=np.float32)
for c, w in zip(centers, widths):
    amp = base * rng.uniform(0.8, 1.4)
    mu += amp * gauss_profile(lam, float(c), float(w)).astype(np.float32)

mu = np.clip(mu, 0.0, 0.02).astype(np.float32)

# σ: create smooth, positive variation across bins
noise = rng.uniform(-1.0, 1.0, size=bins).astype(np.float32)
smooth = np.cumsum(noise)
smooth -= smooth.min()
smooth /= max(smooth.ptp(), 1e-6)
sigma = min_sigma + (max_sigma - min_sigma) * smooth
sigma = sigma.astype(np.float32)

return mu, sigma

def transit_weight(t: np.ndarray, t0: float, width: float, scale: float = 1.0) -> np.ndarray:
“”“Gaussian-shaped transit weighting over time in [0,1].”””
width = max(float(width), 1e-6)
w = np.exp(-0.5 * ((t - t0) / width) ** 2)
w = (w - w.min()) / max(w.ptp(), 1e-6)
return (w * scale).astype(np.float32)

def low_freq_jitter(rng: np.random.Generator, length: int, max_amp: float = 0.0025, n_harm: int = 3) -> np.ndarray:
“”“Compose a few sines with decreasing amplitudes to simulate low-frequency jitter.”””
t = np.linspace(0, 2 * np.pi, length, endpoint=False, dtype=np.float32)
s = np.zeros(length, dtype=np.float32)
for k in range(1, n_harm + 1):
amp = (max_amp / k) * rng.uniform(0.5, 1.0)
phase = rng.uniform(0, 2 * np.pi)
s += amp * np.sin(k * t + phase).astype(np.float32)
return s

–––––––––––––––––––––––––––––––––––––––––––

Generate per-planet AIRS/FGS1

–––––––––––––––––––––––––––––––––––––––––––

def gen_airs_for_planet(
rng: np.random.Generator,
t_steps: int,
bins: int,
mu: np.ndarray,
sigma: np.ndarray,
) -> np.ndarray:
“”“Generate AIRS (time × bins) array with time-varying transit + jitter + noise.”””
assert mu.shape == (bins,)
assert sigma.shape == (bins,)

t_grid = np.linspace(0.0, 1.0, t_steps, dtype=np.float32)
t0 = float(rng.uniform(0.35, 0.65))
width = float(rng.uniform(0.08, 0.18))
scale = float(rng.uniform(0.9, 1.1))
w = transit_weight(t_grid, t0, width, scale)

jitter_t = low_freq_jitter(rng, t_steps, max_amp=0.0015, n_harm=2)

# Transit modulation across bins
base = 1.0 - np.outer(w, mu).astype(np.float32)

# Add small correlated temporal jitter
base += jitter_t[:, None] * float(np.median(mu) * 0.4)

# Heteroscedastic noise per-bin
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
“”“Generate FGS1 (time × channels) cube (each channel with slight scale/noise differences).”””
depth_mu = float(np.clip(mu.mean() * 0.6, 5e-4, 1.5e-2))  # average depth scaled
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

–––––––––––––––––––––––––––––––––––––––––––

Validation helpers

–––––––––––––––––––––––––––––––––––––––––––

def is_finite(arr: np.ndarray) -> bool:
“”“Check all values finite.”””
return np.isfinite(arr).all()

def validate_pack(d: dict) -> None:
“”“Validate the packed dictionary shapes and basic constraints.”””
required = [“fgs1”, “airs”, “mu”, “sigma”, “ids”, “split”]
for k in required:
if k not in d:
raise ValueError(f”Missing key in pack: {k}”)

fgs1 = d["fgs1"]
airs = d["airs"]
mu = d["mu"]
sigma = d["sigma"]
ids = d["ids"]
split = d["split"]

if fgs1.ndim != 3:
    raise ValueError(f"fgs1 must be (N,T,CH), got {fgs1.shape}")
if airs.ndim != 3:
    raise ValueError(f"airs must be (N,T,B), got {airs.shape}")
if mu.ndim != 2 or sigma.ndim != 2:
    raise ValueError("mu and sigma must be (N,B)")
if ids.shape[0] != mu.shape[0] or split.shape[0] != mu.shape[0]:
    raise ValueError("ids/split length must match N")

N, T_f, CH = fgs1.shape
N2, T_a, B = airs.shape
if N != N2 or mu.shape != (N, B) or sigma.shape != (N, B):
    raise ValueError("Shape mismatch among fgs1/airs/mu/sigma")

# Guardrails
if not is_finite(fgs1) or not is_finite(airs) or not is_finite(mu) or not is_finite(sigma):
    raise ValueError("Non-finite values detected in arrays")

if (mu < 0).any():
    raise ValueError("Negative values detected in mu (must be nonnegative)")

if (sigma <= 0).any():
    raise ValueError("Nonpositive values detected in sigma (must be > 0)")

if (fgs1 < 0).any() or (airs < 0).any():
    warn("Negative flux values exist after generation; arrays are clipped later in pipeline.")

–––––––––––––––––––––––––––––––––––––––––––

Packing & IO

–––––––––––––––––––––––––––––––––––––––––––

def pack_dict(
fgs1: np.ndarray,
airs: np.ndarray,
mu: np.ndarray,
sigma: np.ndarray,
ids: np.ndarray,
split: np.ndarray,
) -> dict:
“”“Pack arrays + metadata into a dict for pickling.”””
d = {
“fgs1”: fgs1,    # (N, T_fgs1, 32)
“airs”: airs,    # (N, T_airs=32, BINS=283)
“mu”: mu,        # (N, 283)
“sigma”: sigma,  # (N, 283)
“ids”: ids,      # (N,)
“split”: split,  # (N,) ‘train’/‘val’/‘test’
“version”: VERSION,
}
validate_pack(d)
return d

def write_pkl(path: Path, obj: dict) -> None:
“”“Write a dict to pickle using highest protocol.”””
import pickle
with path.open(“wb”) as f:
pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def maybe_write_csv_labels(path: Path, ids: np.ndarray, split: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
“”“Optionally write labels CSV if pandas is available.”””
if not HAS_PANDAS:
warn(“pandas not available; skipping labels CSV.”)
return
mu_cols = [f”target_mu{k}” for k in range(mu.shape[1])]
sg_cols = [f”target_sigma_{k}” for k in range(sigma.shape[1])]
df = pd.DataFrame({“planet_id”: ids, “split”: split})
df_mu = pd.DataFrame(mu, columns=mu_cols)
df_sg = pd.DataFrame(sigma, columns=sg_cols)
df = pd.concat([df, df_mu, df_sg], axis=1)
df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)

def save_binmap(path: Path, source_bins: int = 356, target_bins: int = 283) -> None:
“””
Simple deterministic 356→283 mapping (down-select indices uniformly).
In real pipelines, you’d use the official instrument mapping. This is for smoke/toy only.
“””
if target_bins > source_bins:
raise ValueError(“target_bins must be ≤ source_bins”)
# Uniformly sample indices from 0..source_bins-1
idx = np.linspace(0, source_bins - 1, target_bins, dtype=np.int64)
np.save(path, idx)

def write_array(path: Path, arr: np.ndarray, fmt: str, key: Optional[str] = None) -> None:
“””
Save an array either as .npy or .npz (with a named key).
For .npz, if key is None, use ‘arr’.
“””
fmt = fmt.lower()
if fmt == “npy”:
np.save(path.with_suffix(”.npy”), arr)
elif fmt == “npz”:
np.savez_compressed(path.with_suffix(”.npz”), **{key or “arr”: arr})
else:
raise ValueError(f”Unsupported array format: {fmt}”)

–––––––––––––––––––––––––––––––––––––––––––

Generation pipeline

–––––––––––––––––––––––––––––––––––––––––––

def compute_splits(
rng: np.random.Generator, n: int, fr_train: float, fr_val: float, fr_test: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
“”“Deterministic shuffle + partition into train/val/test by fractions (sum=1).”””
s = fr_train + fr_val + fr_test
if not (abs(s - 1.0) < 1e-6):
raise ValueError(f”Split fractions must sum to 1.0; got {s:.6f}”)
idx = np.arange(n, dtype=np.int64)
rng.shuffle(idx)
n_train = int(round(n * fr_train))
n_val = int(round(n * fr_val))
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]
return train_idx, val_idx, test_idx

def generate_ids(prefix: str, pad: int, n: int) -> np.ndarray:
“”“Generate zero-padded IDs like ‘toy_0000’.”””
return np.array([f”{prefix}{str(i).zfill(pad)}” for i in range(n)], dtype=object)

def generate(cfg: GenCfg, out_format: str) -> Dict[str, Optional[str]]:
“””
Generate arrays, write requested artifacts, and return a dict of produced file hashes.
out_format: one of {‘pkl’,‘npy’,‘both’}
“””
ensure_dir(cfg.outdir)
splits_dir = cfg.outdir / “splits”
ensure_dir(splits_dir)

rng = rng_from_seed(cfg.seed)
N = cfg.dims.n
T_F = cfg.dims.t_fgs1
T_A = cfg.dims.t_airs
B = cfg.dims.bins
CH = cfg.dims.channels_fgs1

info(f"Generating {cfg.mode} dataset → {cfg.outdir} (N={N}, bins={B}, T_fgs1={T_F}, T_airs={T_A})")

# Pre-allocate (single pass)
fgs1_all = np.zeros((N, T_F, CH), dtype=np.float32)
airs_all = np.zeros((N, T_A, B), dtype=np.float32)
mu_all = np.zeros((N, B), dtype=np.float32)
sigma_all = np.zeros((N, B), dtype=np.float32)

ids = generate_ids(cfg.id_prefix, cfg.id_pad, N)

# Deterministic planet-wise generation (sub-RNG routed by index + seed)
for i in range(N):
    sub_rng = rng_from_seed(cfg.seed + 10000 * (i + 1))
    mu, sigma = build_mu_sigma(sub_rng, B)
    mu_all[i] = mu
    sigma_all[i] = sigma
    airs_all[i] = gen_airs_for_planet(sub_rng, T_A, B, mu, sigma)
    fgs1_all[i] = gen_fgs1_for_planet(sub_rng, T_F, CH, mu)

    # Light progress log on small N
    if N <= 64 and (i + 1) % max(1, N // 8) == 0:
        info(f"  progress: {i + 1}/{N}")

# Splits
train_frac, val_frac, test_frac = cfg.splits
train_idx, val_idx, test_idx = compute_splits(rng, N, train_frac, val_frac, test_frac)
np.save(splits_dir / "train_idx.npy", train_idx)
np.save(splits_dir / "val_idx.npy", val_idx)
np.save(splits_dir / "test_idx.npy", test_idx)
write_csv_indices(splits_dir / "train_idx.csv", train_idx)
write_csv_indices(splits_dir / "val_idx.csv", val_idx)
write_csv_indices(splits_dir / "test_idx.csv", test_idx)

split_arr = np.array(["train"] * N, dtype=object)
split_arr[val_idx] = "val"
split_arr[test_idx] = "test"

# Prepare outputs and hashes
file_hashes: Dict[str, Optional[str]] = {}

# Pack partitions (by indices) into PKL files
if out_format in ("pkl", "both") and cfg.write_pkl:
    # train
    d_train = pack_dict(
        fgs1_all[train_idx], airs_all[train_idx], mu_all[train_idx], sigma_all[train_idx],
        ids[train_idx], split_arr[train_idx],
    )
    p_train = cfg.outdir / cfg.train_name
    write_pkl(p_train, d_train)
    file_hashes[str(p_train)] = sha256_of_file(p_train)

    # val
    d_val = pack_dict(
        fgs1_all[val_idx], airs_all[val_idx], mu_all[val_idx], sigma_all[val_idx],
        ids[val_idx], split_arr[val_idx],
    )
    p_val = cfg.outdir / cfg.val_name
    write_pkl(p_val, d_val)
    file_hashes[str(p_val)] = sha256_of_file(p_val)

    # test
    d_test = pack_dict(
        fgs1_all[test_idx], airs_all[test_idx], mu_all[test_idx], sigma_all[test_idx],
        ids[test_idx], split_arr[test_idx],
    )
    p_test = cfg.outdir / cfg.test_name
    write_pkl(p_test, d_test)
    file_hashes[str(p_test)] = sha256_of_file(p_test)

    info(f"Wrote PKL: {cfg.train_name}, {cfg.val_name}, {cfg.test_name}")

# Optional array exports for smoke alignment with debug configs
if out_format in ("npy", "both") and cfg.write_npy:
    # Use .npz compressed to keep artifact sizes small
    write_array(cfg.outdir / "fgs1_train", fgs1_all[train_idx], cfg.array_format, key="fgs1")
    write_array(cfg.outdir / "airs_train", airs_all[train_idx], cfg.array_format, key="airs")
    write_array(cfg.outdir / "fgs1_test", fgs1_all[test_idx], cfg.array_format, key="fgs1")
    write_array(cfg.outdir / "airs_test", airs_all[test_idx], cfg.array_format, key="airs")
    info("Wrote array exports for train/test.")
    # Hash whichever extension used
    for stem in ["fgs1_train", "airs_train", "fgs1_test", "airs_test"]:
        ext = ".npz" if cfg.array_format == "npz" else ".npy"
        file_hashes[str(cfg.outdir / (stem + ext))] = sha256_of_file(cfg.outdir / (stem + ext))

# Optional labels CSV (planet_id, split, target_mu_*, target_sigma_*)
if cfg.write_csv_labels:
    labels_csv = cfg.outdir / ("labels_debug.csv" if cfg.mode == "debug" else "labels.csv")
    maybe_write_csv_labels(labels_csv, ids, split_arr, mu_all, sigma_all)
    info(f"Wrote labels CSV → {labels_csv}")
    file_hashes[str(labels_csv)] = sha256_of_file(labels_csv)

# Optional 356→283 binmap
if cfg.save_binmap:
    map_path = cfg.outdir / (f"binmap_356_to_{B}_{cfg.mode}.npy")
    save_binmap(map_path, source_bins=356, target_bins=B)
    info(f"Wrote binmap → {map_path}")
    file_hashes[str(map_path)] = sha256_of_file(map_path)

# Manifest for audits
manifest = {
    "version": VERSION,
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
    "array_format": cfg.array_format,
    "write_csv_labels": bool(cfg.write_csv_labels),
    "save_binmap": bool(cfg.save_binmap),
    "paths": {
        "outdir": str(cfg.outdir),
        "splits_dir": str(splits_dir),
        "train_pkl": str(cfg.outdir / cfg.train_name) if cfg.write_pkl else None,
        "val_pkl": str(cfg.outdir / cfg.val_name) if cfg.write_pkl else None,
        "test_pkl": str(cfg.outdir / cfg.test_name) if cfg.write_pkl else None,
    },
    "hashes": file_hashes,
}
save_json(cfg.outdir / "toy_manifest.json", manifest)
info("Toy/debug dataset generation complete.")
return file_hashes

–––––––––––––––––––––––––––––––––––––––––––

CLI

–––––––––––––––––––––––––––––––––––––––––––

def parse_args() -> argparse.Namespace:
p = argparse.ArgumentParser(
prog=“ariel_toy_dataset”,
description=“Generate deterministic toy/debug datasets for SpectraMind V50.”,
formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
p.add_argument(”–mode”, choices=[“toy”, “debug”], default=“debug”, help=“Dataset preset.”)
p.add_argument(”–outdir”, type=str, default=“data/debug”, help=“Output directory for artifacts.”)
p.add_argument(”–seed”, type=int, default=1337, help=“RNG seed.”)
# Back-compat flags:
p.add_argument(”–write-pkl”, action=“store_true”, help=“Write packed .pkl files.”)
p.add_argument(”–write-npy”, action=“store_true”, help=“Write arrays (.npy/.npz) for smoke tests.”)
# New unified format flag (overrides legacy if provided):
p.add_argument(
“–format”,
choices=[“pkl”, “npy”, “both”],
default=None,
help=“Output format (overrides –write-pkl/–write-npy if provided).”,
)
p.add_argument(”–array-format”, choices=[“npy”, “npz”], default=“npz”, help=“Array file container for –format npy/both.”)
p.add_argument(”–write-csv-labels”, action=“store_true”, help=“Write labels CSV (requires pandas).”)
p.add_argument(”–save-binmap”, action=“store_true”, help=“Write a simple 356→283 binmap.”)
p.add_argument(”–force”, action=“store_true”, help=“Overwrite existing artifacts if present.”)

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

# ID controls
p.add_argument("--id-prefix", type=str, default=None, help="Planet ID prefix (default: '<mode>_').")
p.add_argument("--id-pad", type=int, default=4, help="Zero-padding width for IDs (default: 4).")

return p.parse_args()

def decide_out_format(args: argparse.Namespace) -> str:
“””
Choose output format based on –format, or legacy flags if –format not provided.
Returns one of {‘pkl’,‘npy’,‘both’}.
“””
if args.format:
return args.format
# Legacy behavior: if neither given, default to PKL; if both flags, ‘both’
if args.write_pkl and args.write_npy:
return “both”
if args.write_npy:
return “npy”
# default
return “pkl”

def make_cfg(args: argparse.Namespace) -> GenCfg:
“”“Build the generator configuration from parsed CLI args, applying mode presets.”””
if args.mode == “debug”:
dims = GenDims(
n=5 if args.n is None else args.n,
t_fgs1=64 if args.t_fgs1 is None else args.t_fgs1,
t_airs=32 if args.t_airs is None else args.t_airs,
bins=283 if args.bins is None else args.bins,
channels_fgs1=32,
)
train_name = args.train_name or “train_debug.pkl”
val_name = args.val_name or “val_debug.pkl”
test_name = args.test_name or “test_debug.pkl”
splits = (
0.8 if args.train_frac is None else args.train_frac,
0.2 if args.val_frac is None else args.val_frac,
0.0 if args.test_frac is None else args.test_frac,
)
default_id_prefix = “debug_”
else:  # toy
dims = GenDims(
n=64 if args.n is None else args.n,
t_fgs1=128 if args.t_fgs1 is None else args.t_fgs1,
t_airs=32 if args.t_airs is None else args.t_airs,
bins=283 if args.bins is None else args.bins,
channels_fgs1=32,
)
train_name = args.train_name or “train.pkl”
val_name = args.val_name or “val.pkl”
test_name = args.test_name or “test.pkl”
splits = (
0.8 if args.train_frac is None else args.train_frac,
0.1 if args.val_frac is None else args.val_frac,
0.1 if args.test_frac is None else args.test_frac,
)
default_id_prefix = “toy_”

if abs(sum(splits) - 1.0) > 1e-6:
    raise ValueError(f"Invalid splits (must sum to 1.0): {splits}")

cfg = GenCfg(
    mode=args.mode,
    outdir=Path(args.outdir),
    seed=int(args.seed),
    write_pkl=True,   # default on; output format is decided separately
    write_npy=True,   # default on; output format is decided separately
    write_csv_labels=bool(args.write_csv_labels),
    save_binmap=bool(args.save_binmap),
    train_name=train_name,
    val_name=val_name,
    test_name=test_name,
    splits=splits,
    dims=dims,
    id_prefix=(args.id_prefix if args.id_prefix is not None else default_id_prefix),
    id_pad=int(args.id_pad),
    array_format=args.array_format.lower(),
)
return cfg

def preflight_overwrite_checks(cfg: GenCfg, out_format: str, force: bool) -> None:
“””
If not –force, check whether we would overwrite existing artifacts and abort if so.
Checks PKL filenames and array stems we plan to write (train/test).
“””
if force:
return
to_check: List[Path] = []

if out_format in ("pkl", "both") and cfg.write_pkl:
    to_check += [cfg.outdir / cfg.train_name, cfg.outdir / cfg.val_name, cfg.outdir / cfg.test_name]

if out_format in ("npy", "both") and cfg.write_npy:
    ext = ".npz" if cfg.array_format == "npz" else ".npy"
    to_check += [
        cfg.outdir / ("fgs1_train" + ext),
        cfg.outdir / ("airs_train" + ext),
        cfg.outdir / ("fgs1_test" + ext),
        cfg.outdir / ("airs_test" + ext),
    ]

existing = [p for p in to_check if p.exists()]
if existing:
    warn("Outputs already exist; use --force to overwrite.")
    for p in existing:
        warn(f"  exists: {p}")
    raise SystemExit(0)

def main() -> None:
args = parse_args()
out_format = decide_out_format(args)
cfg = make_cfg(args)

# Idempotency: skip if outputs exist (unless --force)
preflight_overwrite_checks(cfg, out_format, force=bool(args.force))

# Generate and write artifacts
file_hashes = generate(cfg, out_format)

# Print a compact summary at the end (paths + hashes)
info("Artifact hashes (first 8 chars):")
for p, h in file_hashes.items():
    if h:
        print(f"  {p}: {h[:8]}...")

if name == “main”:
try:
main()
except SystemExit:
# allow sys.exit or our early exit without traceback noise
raise
except Exception as e:
err(f”Unhandled exception: {e}”)
raise