#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_lightcurve_preview.py

SpectraMind V50 — Calibration Plot Generator (Lightcurve Preview)

Purpose
-------
Produce `calibration_plots/lightcurve_preview.png` showing representative
raw vs. calibrated light curves for FGS1 (photometer) and AIRS (spectrometer)
for a chosen planet. The figure is designed for quick human verification that
calibration stages (ADC/dark/flat/CDS, background removal, phase/jitter
alignment) behaved as expected.

Inputs (flexible; robust to missing datasets)
---------------------------------------------
- Calibrated HDF5 (default: outputs/calibrated/lightcurves.h5)
  Expected groups/datasets (any of these patterns will be auto-discovered):
    • /FGS1/cal, /FGS1/raw, /FGS1/time
    • /AIRS/cal, /AIRS/raw, /AIRS/time
  Supported shapes:
    • FGS1: (N_time,) for a single-curve OR (N_planets, N_time)
    • AIRS:  (N_time, N_wavs) OR (N_planets, N_time, N_wavs)
- Optional raw HDF5 (if raw resides separately): will be merged if provided.

Outputs
-------
- PNG at: <outdir>/lightcurve_preview.png
  where <outdir> defaults to "calibration_plots/"

Features
--------
- Auto-discovery of common dataset names and shapes
- Optional planet selection by index or id (best-effort)
- Median-normalization for clarity (keeps relative transit depth visible)
- Rolling median overlay for noise-reduction preview (configurable)
- Tight legends, color-consistent theming, and dark-mode friendly colors
- Reproducible (seeded) and CI-safe (headless Matplotlib)

Usage
-----
  python tools/generate_lightcurve_preview.py \
    --cal-h5 outputs/calibrated/lightcurves.h5 \
    --outdir calibration_plots \
    --planet-idx 0 \
    --airs-wav-idx auto \
    --rolling 51

Design Notes
------------
- We avoid strong assumptions about HDF5 layout and attempt to locate
  "raw/cal/time" datasets by name heuristics.
- For AIRS (2D time×wavelength), we auto-pick a representative wavelength
  with maximal transit depth variability or accept a user-specified index.
- All axes are annotated; we include statistics (SNR proxy, min/max) inline.

Author
------
SpectraMind V50 Team — NeurIPS 2025 Ariel Data Challenge
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import h5py
import matplotlib

# Headless backend for CI/Kaggle
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Utilities ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centered rolling median (odd window), reflect padding at edges."""
    if window <= 1 or x.ndim != 1:
        return x
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    # Use stride trick via sliding windows
    shape = (x.size, w)
    strides = (xp.strides[0], xp.strides[0])
    sw = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    return np.median(sw, axis=1)


def robust_norm(x: np.ndarray) -> np.ndarray:
    """Normalize by median to visualize relative variations (transit depth preserved)."""
    med = np.nanmedian(x)
    if not np.isfinite(med) or np.isclose(med, 0.0):
        return x
    return x / med


def snr_proxy(y: np.ndarray) -> float:
    """
    A simple SNR proxy: median(|y|) / MAD of high-pass residuals.
    Not a physical SNR; used for comparative visualization only.
    """
    # High-pass via subtracting rolling median (window ≈ 151 by default fallback)
    w = min(max(51, y.size // 50 * 2 + 1), 1001)  # odd, ~2% of length
    trend = rolling_median(y, w)
    resid = y - trend
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-12
    return float(np.nanmedian(np.abs(y)) / mad)


@dataclass
class CurvePack:
    time: np.ndarray          # (T,)
    raw: Optional[np.ndarray] # (T,) or None
    cal: Optional[np.ndarray] # (T,) or None


# ---------- HDF5 discovery ----------------------------------------------------------------------

NAME_CANDIDATES = {
    "time":  [r"time", r"t", r"phase"],
    "raw":   [r"raw", r"uncal", r"preproc", r"before"],
    "cal":   [r"cal", r"calib", r"proc", r"after"],
}

def _match_name(name: str, patterns: list[str]) -> bool:
    lname = name.lower()
    return any(re.fullmatch(pat, lname) or re.search(fr"(^|/|_){pat}($|/|_)", lname) for pat in patterns)

def _find_first_dset(h5: h5py.File, group_path: str, key: str) -> Optional[np.ndarray]:
    """
    Find the first dataset under group_path whose name matches NAME_CANDIDATES[key].
    """
    if group_path not in h5:
        return None
    grp = h5[group_path]
    # Direct children search
    for k, obj in grp.items():
        if isinstance(obj, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
            return obj[()]
    # Deep search (walk)
    for root, _, items in grp.items():
        del root  # unused
        break  # h5py groups are not iterable like walk; fallback to global scan below

    # Global search fallback inside group subtree
    def walker(g: h5py.Group, base: str) -> Optional[np.ndarray]:
        for k, v in g.items():
            path = f"{base}/{k}"
            if isinstance(v, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
                return v[()]
            if isinstance(v, h5py.Group):
                out = walker(v, path)
                if out is not None:
                    return out
        return None

    return walker(grp, group_path)

def _auto_pick_wavelength(airs_tc: np.ndarray) -> int:
    """
    Heuristic: choose wavelength index that maximizes temporal variance (transit depth signal).
    Supports shapes (T, W) or (P, T, W) with P=1 (single planet).
    """
    if airs_tc.ndim == 2:
        # (T, W)
        vari = np.nanvar(airs_tc, axis=0)
    elif airs_tc.ndim == 3 and airs_tc.shape[0] == 1:
        vari = np.nanvar(airs_tc[0], axis=0)
    else:
        # Fallback
        return 0
    idx = int(np.nanargmax(vari))
    return max(0, min(idx, airs_tc.shape[-1] - 1))

def _select_planet(arr: np.ndarray, planet_idx: int) -> np.ndarray:
    """
    Select planet index if leading dimension is planet; otherwise return as-is.
    Accepts:
      - (T,)                   -> return
      - (P, T) or (P, T, W)   -> select planet_idx
      - (T, W)                -> return
    """
    if arr.ndim >= 2 and arr.shape[0] > 8 and arr.shape[0] < arr.shape[-1]:
        # Heuristic: leading dim could be planets (P), typical P ~ 1..1100
        if planet_idx < 0 or planet_idx >= arr.shape[0]:
            raise IndexError(f"planet_idx {planet_idx} out of range 0..{arr.shape[0]-1}")
        return arr[planet_idx]
    return arr


def discover_curves(
    cal_h5_path: str,
    raw_h5_path: Optional[str],
    planet_idx: int,
    airs_wav_idx: Optional[int | str],
) -> Tuple[CurvePack, CurvePack]:
    """
    Returns (fgs1, airs) where each is a CurvePack(time, raw, cal).
    For AIRS, 'raw' and 'cal' are single wavelength time-series (after selecting wav index).
    """
    if not os.path.isfile(cal_h5_path):
        raise FileNotFoundError(f"Calibrated H5 not found: {cal_h5_path}")
    raw_h5 = None
    if raw_h5_path:
        if not os.path.isfile(raw_h5_path):
            print(f"[warn] Raw H5 not found at {raw_h5_path}; continuing with calibrated only.")
        else:
            raw_h5 = h5py.File(raw_h5_path, "r")

    with h5py.File(cal_h5_path, "r") as cal_h5:
        # --- FGS1 ---
        fgs1_time = None
        fgs1_raw = None
        fgs1_cal = None
        for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
            if grp.strip("/") in cal_h5:
                gpath = f"/{grp.strip('/')}"
                # time
                fgs1_time = _find_first_dset(cal_h5, gpath, "time")
                # raw/cal from calibrated file first
                fgs1_raw = _find_first_dset(cal_h5, gpath, "raw")
                fgs1_cal = _find_first_dset(cal_h5, gpath, "cal")
                break

        # If raw missing, try raw_h5
        if fgs1_raw is None and raw_h5 is not None:
            for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
                if grp.strip("/") in raw_h5:
                    fgs1_raw = _find_first_dset(raw_h5, f"/{grp.strip('/')}", "raw")
                    if fgs1_time is None:
                        fgs1_time = _find_first_dset(raw_h5, f"/{grp.strip('/')}", "time")
                    break

        # Select planet if needed
        if fgs1_time is not None and fgs1_time.ndim > 1:
            fgs1_time = _select_planet(fgs1_time, planet_idx)
        if fgs1_raw is not None and fgs1_raw.ndim > 1:
            fgs1_raw = _select_planet(fgs1_raw, planet_idx)
        if fgs1_cal is not None and fgs1_cal.ndim > 1:
            fgs1_cal = _select_planet(fgs1_cal, planet_idx)

        # --- AIRS ---
        airs_time = None
        airs_raw_tc = None  # time × wav
        airs_cal_tc = None
        for grp in ("AIRS", "airs", "/AIRS", "/airs"):
            if grp.strip("/") in cal_h5:
                gpath = f"/{grp.strip('/')}"
                airs_time = _find_first_dset(cal_h5, gpath, "time")
                airs_raw_tc = _find_first_dset(cal_h5, gpath, "raw")
                airs_cal_tc = _find_first_dset(cal_h5, gpath, "cal")
                break

        if airs_raw_tc is None and raw_h5 is not None:
            for grp in ("AIRS", "airs", "/AIRS", "/airs"):
                if grp.strip("/") in raw_h5:
                    airs_raw_tc = _find_first_dset(raw_h5, f"/{grp.strip('/')}", "raw")
                    if airs_time is None:
                        airs_time = _find_first_dset(raw_h5, f"/{grp.strip('/')}", "time")
                    break

        # Planet selection
        if airs_time is not None and airs_time.ndim > 1:
            airs_time = _select_planet(airs_time, planet_idx)
        if airs_raw_tc is not None and airs_raw_tc.ndim >= 2:
            airs_raw_tc = _select_planet(airs_raw_tc, planet_idx)
        if airs_cal_tc is not None and airs_cal_tc.ndim >= 2:
            airs_cal_tc = _select_planet(airs_cal_tc, planet_idx)

        # Choose wavelength index
        wav_choice = None
        if airs_cal_tc is not None and airs_cal_tc.ndim == 2:
            if isinstance(airs_wav_idx, str) and airs_wav_idx.lower() == "auto":
                wav_choice = _auto_pick_wavelength(airs_cal_tc)
            else:
                wav_choice = int(airs_wav_idx or 0)
            wav_choice = max(0, min(wav_choice, airs_cal_tc.shape[1] - 1))
        elif airs_raw_tc is not None and airs_raw_tc.ndim == 2:
            if isinstance(airs_wav_idx, str) and airs_wav_idx.lower() == "auto":
                wav_choice = _auto_pick_wavelength(airs_raw_tc)
            else:
                wav_choice = int(airs_wav_idx or 0)
            wav_choice = max(0, min(wav_choice, airs_raw_tc.shape[1] - 1))
        else:
            wav_choice = 0

        # Slice the selected wavelength to 1D time-series
        if airs_raw_tc is not None and airs_raw_tc.ndim == 2:
            airs_raw = airs_raw_tc[:, wav_choice]
        else:
            airs_raw = airs_raw_tc
        if airs_cal_tc is not None and airs_cal_tc.ndim == 2:
            airs_cal = airs_cal_tc[:, wav_choice]
        else:
            airs_cal = airs_cal_tc

        fgs1 = CurvePack(
            time=fgs1_time if fgs1_time is not None else np.arange(fgs1_cal.size) if fgs1_cal is not None else np.arange(100),
            raw=fgs1_raw,
            cal=fgs1_cal,
        )
        airs = CurvePack(
            time=airs_time if airs_time is not None else np.arange(airs_cal.size) if airs_cal is not None else np.arange(100),
            raw=airs_raw,
            cal=airs_cal,
        )
        return fgs1, airs


# ---------- Plotting ---------------------------------------------------------------------------

def plot_lightcurve_preview(
    fgs1: CurvePack,
    airs: CurvePack,
    out_path: str,
    rolling: int,
    title_suffix: str,
) -> None:
    # Styling
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False, constrained_layout=True)
    ax1, ax2 = axes

    # --- FGS1 ---
    ax1.set_title("FGS1 Light Curve — Raw vs Calibrated" + title_suffix, fontsize=12, weight="bold")
    # Normalize to median so transit stands out
    if fgs1.raw is not None:
        y_raw = robust_norm(np.asarray(fgs1.raw).astype(float))
        ax1.plot(fgs1.time, y_raw, color="#8888ff", alpha=0.55, lw=1.2, label="raw (median-norm)")
        if rolling > 1:
            ax1.plot(fgs1.time, rolling_median(y_raw, rolling), color="#4c4cff", lw=1.8, label=f"raw rolling med (w={rolling})")
        s_raw = snr_proxy(y_raw)
        ax1.text(0.01, 0.93, f"SNR proxy (raw): {s_raw:.1f}", transform=ax1.transAxes, fontsize=9, color="#4c4cff")
    if fgs1.cal is not None:
        y_cal = robust_norm(np.asarray(fgs1.cal).astype(float))
        ax1.plot(fgs1.time, y_cal, color="#88ff88", alpha=0.65, lw=1.2, label="calibrated (median-norm)")
        if rolling > 1:
            ax1.plot(fgs1.time, rolling_median(y_cal, rolling), color="#21c15b", lw=1.8, label=f"cal rolling med (w={rolling})")
        s_cal = snr_proxy(y_cal)
        ax1.text(0.01, 0.86, f"SNR proxy (cal): {s_cal:.1f}", transform=ax1.transAxes, fontsize=9, color="#21c15b")

    ax1.set_ylabel("Flux (median-normalized)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # --- AIRS ---
    ax2.set_title("AIRS Light Curve at Representative λ — Raw vs Calibrated" + title_suffix, fontsize=12, weight="bold")
    if airs.raw is not None:
        y_raw = robust_norm(np.asarray(airs.raw).astype(float))
        ax2.plot(airs.time, y_raw, color="#ff9888", alpha=0.55, lw=1.2, label="raw (median-norm)")
        if rolling > 1:
            ax2.plot(airs.time, rolling_median(y_raw, rolling), color="#ff6a4c", lw=1.8, label=f"raw rolling med (w={rolling})")
        s_raw = snr_proxy(y_raw)
        ax2.text(0.01, 0.93, f"SNR proxy (raw): {s_raw:.1f}", transform=ax2.transAxes, fontsize=9, color="#ff6a4c")

    if airs.cal is not None:
        y_cal = robust_norm(np.asarray(airs.cal).astype(float))
        ax2.plot(airs.time, y_cal, color="#88e5ff", alpha=0.65, lw=1.2, label="calibrated (median-norm)")
        if rolling > 1:
            ax2.plot(airs.time, rolling_median(y_cal, rolling), color="#36b3d6", lw=1.8, label=f"cal rolling med (w={rolling})")
        s_cal = snr_proxy(y_cal)
        ax2.text(0.01, 0.86, f"SNR proxy (cal): {s_cal:.1f}", transform=ax2.transAxes, fontsize=9, color="#36b3d6")

    ax2.set_xlabel("Time (arbitrary units or phase)")
    ax2.set_ylabel("Flux (median-normalized)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # Footer note
    footer = textwrap.dedent(
        """\
        Notes:
        • Curves are median-normalized to emphasize relative depth.
        • Rolling median overlays are for visual noise/smoothness inspection only.
        • SNR proxy = median(|y|)/MAD(high-pass residuals); for comparative inspection (not physical SNR).
        """
    )
    fig.text(0.01, 0.01, footer, fontsize=8, color="#aaaaaa", va="bottom", ha="left")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------- Main --------------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate calibration_plots/lightcurve_preview.png (raw vs calibrated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cal-h5", type=str, default="outputs/calibrated/lightcurves.h5",
                   help="Path to calibrated HDF5 containing FGS1/AIRS groups.")
    p.add_argument("--raw-h5", type=str, default=None,
                   help="Optional path to raw HDF5 if raw curves are not present in cal-h5.")
    p.add_argument("--outdir", type=str, default="calibration_plots",
                   help="Output directory for the PNG.")
    p.add_argument("--planet-idx", type=int, default=0,
                   help="Planet index to visualize if arrays are stacked by planet.")
    p.add_argument("--airs-wav-idx", default="auto",
                   help="AIRS wavelength index (int) or 'auto' to pick a representative channel.")
    p.add_argument("--rolling", type=int, default=51,
                   help="Centered rolling-median window size (odd). Set 1 to disable.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.outdir)

    try:
        fgs1, airs = discover_curves(
            cal_h5_path=args.cal_h5,
            raw_h5_path=args.raw_h5,
            planet_idx=args.planet_idx,
            airs_wav_idx=args.airs_wav_idx,
        )
    except Exception as e:
        print(f"[error] Failed to read curves: {e}", file=sys.stderr)
        return 2

    title_suffix = f"  (planet idx={args.planet_idx}, AIRS λ={args.airs_wav_idx})"
    out_path = os.path.join(args.outdir, "lightcurve_preview.png")

    try:
        plot_lightcurve_preview(
            fgs1=fgs1,
            airs=airs,
            out_path=out_path,
            rolling=int(args.rolling),
            title_suffix=title_suffix,
        )
        print(f"[ok] Wrote {out_path}")
        return 0
    except Exception as e:
        print(f"[error] Plotting failed: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())