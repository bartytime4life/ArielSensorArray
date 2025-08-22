#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_fft_alignment.py

SpectraMind V50 — Calibration Plot Generator (FFT Alignment)

Purpose
-------
Produce `calibration_plots/fft_alignment.png` comparing pre- and post-calibration
frequency content for FGS1 and AIRS (single representative wavelength channel).
This helps verify that calibration suppressed instrument/systematic bands and
preserved astrophysical transit power at low frequencies.

Inputs
------
- Calibrated HDF5 (default: outputs/calibrated/lightcurves.h5)
  Expected group layout (auto-discovered; flexible):
    /FGS1/{time,raw,cal}, /AIRS/{time,raw,cal}
  Shapes supported:
    FGS1: (T,) or (P,T)   | AIRS: (T,W), (P,T,W)
- Optional raw HDF5 (to fetch 'raw' series if cal file lacks them)

Outputs
-------
- PNG at: <outdir>/fft_alignment.png

Features
--------
- Robust HDF5 discovery (similar heuristics to generate_lightcurve_preview.py)
- Planet selection and AIRS wavelength auto-pick (max temporal variance)
- Hann window + median-detrend prior to FFT for stable spectra
- Frequency axis in cycles per unit time (c / Δt); marks top-N peaks
- Dual panels (FGS1 & AIRS); overlays raw vs calibrated amplitude spectra
- Optional notch-mark annotation (user-provided) for known jitter frequencies

Usage
-----
  python tools/generate_fft_alignment.py \
      --cal-h5 outputs/calibrated/lightcurves.h5 \
      --raw-h5 data/raw/lightcurves_raw.h5 \
      --outdir calibration_plots \
      --planet-idx 0 \
      --airs-wav-idx auto \
      --topk 5 \
      --notches 0.001,0.0025

Notes
-----
- Frequency units depend on input time units. If time spacing is uneven,
  FFT uses median Δt for sampling frequency estimate.
- This plot is diagnostic; not a physics-grade periodogram.

Author
------
SpectraMind V50 Team — NeurIPS 2025 Ariel Data Challenge
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import h5py
import matplotlib

# Headless for CI/Kaggle
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------- Utility --------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def median_detrend(y: np.ndarray) -> np.ndarray:
    med = np.nanmedian(y)
    return y - med if np.isfinite(med) else y


def hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n))
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def estimate_dt(t: np.ndarray) -> float:
    """Estimate uniform sampling interval using median of diffs; fall back to 1.0."""
    t = np.asarray(t).astype(float)
    if t.ndim != 1 or t.size < 2:
        return 1.0
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1.0
    dt = float(np.median(diffs))
    return dt if (np.isfinite(dt) and dt > 0) else 1.0


def compute_fft_amp(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided amplitude spectrum for real-valued y(t).
    Returns (freq, amp), where freq in cycles per time-unit, amp normalized.
    """
    t = np.asarray(t).astype(float)
    y = np.asarray(y).astype(float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    n = t.size
    if n < 8:
        # Degenerate
        return np.array([0.0]), np.array([0.0])
    # Uniformize assumption via median dt; resample not performed (diagnostic)
    dt = estimate_dt(t)
    # Detrend + window
    yw = median_detrend(y) * hann_window(n)
    # FFT
    Y = np.fft.rfft(yw, n=n)
    freq = np.fft.rfftfreq(n, d=dt)  # cycles per time-unit
    # Amplitude normalization: 2/N for one-sided, preserve window energy scale
    amp = (2.0 / n) * np.abs(Y)
    return freq, amp


def top_k_peaks(freq: np.ndarray, amp: np.ndarray, k: int, fmin: float = 0.0) -> List[int]:
    """Return indices of top-k peaks above minimum frequency."""
    mask = freq >= fmin
    if not np.any(mask):
        return []
    f = freq[mask]
    a = amp[mask]
    # Avoid DC / zero-freq if present
    if f[0] == 0:
        a = a.copy()
        a[0] = 0.0
    idx_sorted = np.argsort(a)[::-1]
    picks = idx_sorted[: max(0, k)]
    # Map back to original indices
    base = np.where(mask)[0][0]
    return (base + picks).tolist()


@dataclass
class CurvePack:
    time: np.ndarray
    raw: Optional[np.ndarray]
    cal: Optional[np.ndarray]


# --------------------- HDF5 Discovery (same patterns used elsewhere) ------------------------

NAME_CANDIDATES = {
    "time":  [r"time", r"t", r"phase"],
    "raw":   [r"raw", r"uncal", r"preproc", r"before"],
    "cal":   [r"cal", r"calib", r"proc", r"after"],
}

def _match_name(name: str, patterns: list[str]) -> bool:
    lname = name.lower()
    return any(re.fullmatch(pat, lname) or re.search(fr"(^|/|_){pat}($|/|_)", lname) for pat in patterns)

def _find_first_dset(h5: h5py.File, group_path: str, key: str) -> Optional[np.ndarray]:
    if group_path not in h5:
        return None
    grp = h5[group_path]
    # Direct children first
    for k, obj in grp.items():
        if isinstance(obj, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
            return obj[()]
    # Deep search
    def walker(g: h5py.Group, base: str) -> Optional[np.ndarray]:
        for k, v in g.items():
            if isinstance(v, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
                return v[()]
            if isinstance(v, h5py.Group):
                out = walker(v, f"{base}/{k}")
                if out is not None:
                    return out
        return None
    return walker(grp, group_path)

def _select_planet(arr: np.ndarray, planet_idx: int) -> np.ndarray:
    if arr.ndim >= 2 and arr.shape[0] > 8 and arr.shape[0] < arr.shape[-1]:
        if planet_idx < 0 or planet_idx >= arr.shape[0]:
            raise IndexError(f"planet_idx {planet_idx} out of range 0..{arr.shape[0]-1}")
        return arr[planet_idx]
    return arr

def _auto_pick_wavelength(airs_tc: np.ndarray) -> int:
    if airs_tc.ndim == 2:
        vari = np.nanvar(airs_tc, axis=0)
    elif airs_tc.ndim == 3 and airs_tc.shape[0] == 1:
        vari = np.nanvar(airs_tc[0], axis=0)
    else:
        return 0
    return int(np.nanargmax(vari))


def discover_curves(cal_h5_path: str,
                    raw_h5_path: Optional[str],
                    planet_idx: int,
                    airs_wav_idx: Optional[int | str]) -> Tuple[CurvePack, CurvePack]:
    if not os.path.isfile(cal_h5_path):
        raise FileNotFoundError(f"Calibrated H5 not found: {cal_h5_path}")
    raw_h5 = None
    if raw_h5_path and os.path.isfile(raw_h5_path):
        raw_h5 = h5py.File(raw_h5_path, "r")

    with h5py.File(cal_h5_path, "r") as cal_h5:
        # FGS1
        fgs1_time = fgs1_raw = fgs1_cal = None
        for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
            if grp.strip("/") in cal_h5:
                gpath = f"/{grp.strip('/')}"
                fgs1_time = _find_first_dset(cal_h5, gpath, "time")
                fgs1_raw  = _find_first_dset(cal_h5, gpath, "raw")
                fgs1_cal  = _find_first_dset(cal_h5, gpath, "cal")
                break
        if fgs1_raw is None and raw_h5 is not None:
            for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
                if grp.strip("/") in raw_h5:
                    fgs1_raw = _find_first_dset(raw_h5, f"/{grp.strip('/')}", "raw")
                    if fgs1_time is None:
                        fgs1_time = _find_first_dset(raw_h5, f"/{grp.strip('/')}", "time")
                    break
        if fgs1_time is not None and fgs1_time.ndim > 1:
            fgs1_time = _select_planet(fgs1_time, planet_idx)
        if fgs1_raw is not None and fgs1_raw.ndim > 1:
            fgs1_raw = _select_planet(fgs1_raw, planet_idx)
        if fgs1_cal is not None and fgs1_cal.ndim > 1:
            fgs1_cal = _select_planet(fgs1_cal, planet_idx)

        # AIRS (time×wav)
        airs_time = airs_raw_tc = airs_cal_tc = None
        for grp in ("AIRS", "airs", "/AIRS", "/airs"):
            if grp.strip("/") in cal_h5:
                gpath = f"/{grp.strip('/')}"
                airs_time   = _find_first_dset(cal_h5, gpath, "time")
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
        if airs_time is not None and airs_time.ndim > 1:
            airs_time = _select_planet(airs_time, planet_idx)
        if airs_raw_tc is not None and airs_raw_tc.ndim >= 2:
            airs_raw_tc = _select_planet(airs_raw_tc, planet_idx)
        if airs_cal_tc is not None and airs_cal_tc.ndim >= 2:
            airs_cal_tc = _select_planet(airs_cal_tc, planet_idx)

        # wavelength choice
        wav_idx = None
        if isinstance(airs_wav_idx, str) and airs_wav_idx.lower() == "auto":
            if airs_cal_tc is not None and airs_cal_tc.ndim == 2:
                wav_idx = _auto_pick_wavelength(airs_cal_tc)
            elif airs_raw_tc is not None and airs_raw_tc.ndim == 2:
                wav_idx = _auto_pick_wavelength(airs_raw_tc)
            else:
                wav_idx = 0
        else:
            wav_idx = int(airs_wav_idx or 0)

        # Extract 1D AIRS curves
        airs_raw = None
        airs_cal = None
        if airs_raw_tc is not None and airs_raw_tc.ndim == 2:
            wav_idx = np.clip(wav_idx, 0, airs_raw_tc.shape[1]-1)
            airs_raw = airs_raw_tc[:, wav_idx]
        elif airs_raw_tc is not None and airs_raw_tc.ndim == 1:
            airs_raw = airs_raw_tc
        if airs_cal_tc is not None and airs_cal_tc.ndim == 2:
            wav_idx = np.clip(wav_idx, 0, airs_cal_tc.shape[1]-1)
            airs_cal = airs_cal_tc[:, wav_idx]
        elif airs_cal_tc is not None and airs_cal_tc.ndim == 1:
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


# --------------------- Plotting ---------------------------------------------------------------

def plot_fft_alignment(
    fgs1: CurvePack,
    airs: CurvePack,
    out_path: str,
    topk: int,
    notch_freqs: Optional[List[float]],
    title_suffix: str,
) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    ax1, ax2 = axes

    # --- FGS1 ---
    ax1.set_title("FGS1 — FFT Amplitude (raw vs calibrated)" + title_suffix, fontsize=12, weight="bold")
    # RAW
    if fgs1.raw is not None:
        fr, ar = compute_fft_amp(fgs1.time, fgs1.raw)
        ax1.plot(fr, ar, color="#8a8aff", lw=1.25, alpha=0.9, label="raw")
        picks = top_k_peaks(fr, ar, topk, fmin=0.0)
        for i in picks:
            ax1.plot([fr[i]], [ar[i]], "o", color="#5a5aff", ms=4)
    # CAL
    if fgs1.cal is not None:
        fc, ac = compute_fft_amp(fgs1.time, fgs1.cal)
        ax1.plot(fc, ac, color="#20c15b", lw=1.35, alpha=0.95, label="calibrated")
        picks = top_k_peaks(fc, ac, topk, fmin=0.0)
        for i in picks:
            ax1.plot([fc[i]], [ac[i]], "o", color="#158f42", ms=4)

    if notch_freqs:
        for nf in notch_freqs:
            ax1.axvline(nf, color="#ffb347", ls="--", lw=1.0, alpha=0.8)
    ax1.set_xlabel("Frequency (cycles / time-unit)")
    ax1.set_ylabel("Amplitude (arb. norm.)")
    ax1.set_xlim(left=0.0)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # --- AIRS ---
    ax2.set_title("AIRS (representative λ) — FFT Amplitude (raw vs calibrated)" + title_suffix, fontsize=12, weight="bold")
    if airs.raw is not None:
        fr, ar = compute_fft_amp(airs.time, airs.raw)
        ax2.plot(fr, ar, color="#ff8c8c", lw=1.25, alpha=0.9, label="raw")
        picks = top_k_peaks(fr, ar, topk, fmin=0.0)
        for i in picks:
            ax2.plot([fr[i]], [ar[i]], "o", color="#ff5e5e", ms=4)
    if airs.cal is not None:
        fc, ac = compute_fft_amp(airs.time, airs.cal)
        ax2.plot(fc, ac, color="#36b3d6", lw=1.35, alpha=0.95, label="calibrated")
        picks = top_k_peaks(fc, ac, topk, fmin=0.0)
        for i in picks:
            ax2.plot([fc[i]], [ac[i]], "o", color="#1d87a7", ms=4)

    if notch_freqs:
        for nf in notch_freqs:
            ax2.axvline(nf, color="#ffb347", ls="--", lw=1.0, alpha=0.8)

    ax2.set_xlabel("Frequency (cycles / time-unit)")
    ax2.set_ylabel("Amplitude (arb. norm.)")
    ax2.set_xlim(left=0.0)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # Footer text
    footer = (
        "Hann-windowed, median-detrended amplitude spectra; "
        "frequency in cycles per time-unit using median Δt. "
        "Dashed lines mark user-specified notch frequencies."
    )
    fig.text(0.01, 0.01, footer, fontsize=8, color="#aaaaaa", va="bottom", ha="left")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------- CLI -------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate calibration_plots/fft_alignment.png (pre/post calibration FFT).",
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
    p.add_argument("--topk", type=int, default=5,
                   help="Highlight top-K peaks in each spectrum.")
    p.add_argument("--notches", type=str, default=None,
                   help="Comma-separated list of notch frequencies to annotate, e.g. '0.001,0.0025'.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.outdir)

    notches = None
    if args.notches:
        try:
            notches = [float(x.strip()) for x in args.notches.split(",") if x.strip() != ""]
        except Exception:
            print("[warn] Failed to parse --notches; ignoring.", file=sys.stderr)
            notches = None

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
    out_path = os.path.join(args.outdir, "fft_alignment.png")

    try:
        plot_fft_alignment(
            fgs1=fgs1,
            airs=airs,
            out_path=out_path,
            topk=max(0, int(args.topk)),
            notch_freqs=notches,
            title_suffix=title_suffix,
        )
        print(f"[ok] Wrote {out_path}")
        return 0
    except Exception as e:
        print(f"[error] Plotting failed: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())