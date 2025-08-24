#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_phase_alignment.py

SpectraMind V50 — Calibration Plot Generator (Phase Alignment & Jitter)

Purpose
-------
Produce `calibration_plots/phase_alignment.png` that visualizes:

  • Phase-aligned light curves (FGS1 + AIRS representative λ), raw vs. calibrated
  • Estimated transit center (t0) and optional period folding
  • Cross-correlation (raw ↔ calibrated) to quantify lag removed by calibration
  • Jitter/residual overlays and before/after RMS metrics

The plot is designed for quick QA: did calibration reduce time-domain jitter and align
the curves around the transit? If a period is provided, we fold; otherwise we normalize
time to a local "phase" relative to an estimated transit center.

Inputs (flexible; robust to missing datasets)
---------------------------------------------
- Calibrated HDF5 (default: outputs/calibrated/lightcurves.h5)
  Expected groups/datasets (auto-discovered with heuristics):
    • /FGS1/{time, raw, cal}
    • /AIRS/{time, raw, cal}
- Optional raw HDF5 if raw curves are stored separately.

Outputs
-------
- PNG at: <outdir>/phase_alignment.png  (default outdir = calibration_plots/)

Features
--------
- HDF5 discovery with tolerant naming patterns
- Planet selection if leading dimension indexes planets
- AIRS wavelength auto-pick by variance, or user-specified index
- Transit center estimation (by argmin of smoothed calibrated curve) if --t0 unspecified
- Optional period folding if --period > 0 (with optional --t0)
- Cross-correlation to estimate raw→cal shift and annotate peak lag
- Jitter/residual diagnostic with RMS before/after

Usage
-----
  python tools/generate_phase_alignment.py \
      --cal-h5 outputs/calibrated/lightcurves.h5 \
      --raw-h5 data/raw/lightcurves_raw.h5 \
      --outdir calibration_plots \
      --planet-idx 0 \
      --airs-wav-idx auto \
      --rolling 101 \
      --period 0.0 \
      --t0 None

Notes
-----
- If you do not know the orbital period (typical for a single-transit chunk), leave --period at 0.
  The script will compute a "relative phase" axis centered at the estimated transit time.
- If you do know period and an approximate t0 (epoch), pass both to produce a proper phase fold.

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
from matplotlib.gridspec import GridSpec


# --------------------- Utilities --------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median with reflect padding. Odd window enforced."""
    if window <= 1 or x.ndim != 1:
        return x
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x, pad_width=pad, mode="reflect")
    shape = (x.size, w)
    strides = (xp.strides[0], xp.strides[0])
    sw = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    return np.nanmedian(sw, axis=1)


def robust_norm(x: np.ndarray) -> np.ndarray:
    """Median-normalize."""
    med = np.nanmedian(x)
    return x / med if np.isfinite(med) and not np.isclose(med, 0.0) else x


def robust_rms(x: np.ndarray) -> float:
    """Robust RMS proxy based on MAD."""
    # MAD ~ 1.4826 * sigma for Gaussian
    m = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - m))
    return float(1.4826 * mad)


def cross_correlation_lag(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute normalized cross-correlation between x and y, return (lags, corr, lag_at_peak_time).
    Assumes uniform sampling for lag→time conversion via median dt.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or y.size < 3:
        return np.array([0.0]), np.array([0.0]), 0.0

    # Normalize (zero-mean, unit-var) to compute NCC
    xz = x - np.mean(x)
    yz = y - np.mean(y)
    sx = np.std(xz) + 1e-12
    sy = np.std(yz) + 1e-12
    xz /= sx
    yz /= sy

    corr = np.correlate(xz, yz, mode="full")
    corr /= (xz.size)  # simple normalization
    lags = np.arange(-xz.size + 1, yz.size)

    # Convert lags to time using median dt
    tmask = np.isfinite(t)
    dt = np.median(np.diff(t[tmask])) if np.sum(tmask) >= 2 else 1.0
    lag_time = lags * dt

    # Peak lag
    pk = int(np.nanargmax(corr))
    lag_at_peak = lag_time[pk]

    return lag_time, corr, float(lag_at_peak)


def estimate_t0(time: np.ndarray, y_cal: np.ndarray, rolling: int = 101) -> float:
    """
    Estimate transit center as argmin of a smoothed calibrated curve.
    Returns time value (t0).
    """
    if time.size < 3:
        return float(time[0] if time.size else 0.0)
    yc = robust_norm(np.asarray(y_cal, float))
    sm = rolling_median(yc, window=rolling)
    idx = int(np.nanargmin(sm))
    return float(time[idx])


def phase_from_time(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """
    Compute phase in [-0.5, 0.5) given period and t0. If period <= 0, return None.
    """
    if period is None or period <= 0:
        return None
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    return phase


def local_phase(time: np.ndarray, t0: float) -> np.ndarray:
    """
    Compute a local "phase-like" axis centered at t0, scaled by half-range so that roughly in [-1, 1].
    This is not physical phase; it's for visualization when period is unknown.
    """
    if time.size == 0:
        return np.array([])
    # Scale by half span
    half_span = max(np.max(time) - t0, t0 - np.min(time), 1e-12)
    return (time - t0) / half_span


# --------------------- HDF5 Discovery --------------------------------------------------------

NAME_CANDIDATES = {
    "time": [r"time", r"t", r"phase"],
    "raw":  [r"raw", r"uncal", r"preproc", r"before"],
    "cal":  [r"cal", r"calib", r"proc", r"after"],
}

def _match_name(name: str, patterns: List[str]) -> bool:
    lname = name.lower()
    return any(re.fullmatch(pat, lname) or re.search(fr"(^|/|_){pat}($|/|_)", lname) for pat in patterns)

def _find_first_dset(h5: h5py.File, group_path: str, key: str) -> Optional[np.ndarray]:
    if group_path not in h5:
        return None
    grp = h5[group_path]
    # direct children
    for k, obj in grp.items():
        if isinstance(obj, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
            return obj[()]
    # recursive
    def walker(g: h5py.Group) -> Optional[np.ndarray]:
        for k, v in g.items():
            if isinstance(v, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
                return v[()]
            if isinstance(v, h5py.Group):
                out = walker(v)
                if out is not None:
                    return out
        return None
    return walker(grp)

def _likely_planet_leading_dim(arr: np.ndarray) -> bool:
    if arr.ndim < 2:
        return False
    P = arr.shape[0]
    rest = int(np.prod(arr.shape[1:]))
    return 1 < P < 100000 and P < rest

def _select_planet(arr: Optional[np.ndarray], planet_idx: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim >= 2 and _likely_planet_leading_dim(arr):
        if not (0 <= planet_idx < arr.shape[0]):
            raise IndexError(f"planet_idx {planet_idx} out of range 0..{arr.shape[0]-1}")
        return arr[planet_idx]
    return arr

def _auto_pick_wavelength(airs_tc: np.ndarray) -> int:
    if airs_tc.ndim == 3 and airs_tc.shape[0] == 1:
        airs_tc = airs_tc[0]
    if airs_tc.ndim != 2:
        return 0
    vari = np.nanvar(airs_tc, axis=0)
    idx = int(np.nanargmax(vari))
    return max(0, min(idx, airs_tc.shape[1] - 1))


@dataclass
class CurvePack:
    time: np.ndarray
    raw: Optional[np.ndarray]
    cal: Optional[np.ndarray]


def discover_curves(
    cal_h5_path: str,
    raw_h5_path: Optional[str],
    planet_idx: int,
    airs_wav_idx: Optional[int | str],
) -> Tuple[CurvePack, CurvePack]:
    if not os.path.isfile(cal_h5_path):
        raise FileNotFoundError(f"Calibrated H5 not found: {cal_h5_path}")

    raw_h5 = h5py.File(raw_h5_path, "r") if (raw_h5_path and os.path.isfile(raw_h5_path)) else None
    try:
        with h5py.File(cal_h5_path, "r") as cal_h5:
            # FGS1
            fgs1_time = fgs1_raw = fgs1_cal = None
            for grp in ("FGS1", "fgs1"):
                if grp in cal_h5:
                    gpath = f"/{grp}"
                    fgs1_time = _find_first_dset(cal_h5, gpath, "time")
                    fgs1_raw  = _find_first_dset(cal_h5, gpath, "raw")
                    fgs1_cal  = _find_first_dset(cal_h5, gpath, "cal")
                    break
            if fgs1_raw is None and raw_h5 is not None:
                for grp in ("FGS1", "fgs1"):
                    if grp in raw_h5:
                        fgs1_raw = _find_first_dset(raw_h5, f"/{grp}", "raw")
                        if fgs1_time is None:
                            fgs1_time = _find_first_dset(raw_h5, f"/{grp}", "time")
                        break
            fgs1_time = _select_planet(fgs1_time, planet_idx) if fgs1_time is not None else None
            fgs1_raw  = _select_planet(fgs1_raw,  planet_idx) if fgs1_raw  is not None else None
            fgs1_cal  = _select_planet(fgs1_cal,  planet_idx) if fgs1_cal  is not None else None
            if fgs1_time is None:
                n = fgs1_cal.size if (fgs1_cal is not None) else (fgs1_raw.size if fgs1_raw is not None else 100)
                fgs1_time = np.arange(n)

            # AIRS (time×wav)
            airs_time = airs_raw_tc = airs_cal_tc = None
            for grp in ("AIRS", "airs"):
                if grp in cal_h5:
                    gpath = f"/{grp}"
                    airs_time   = _find_first_dset(cal_h5, gpath, "time")
                    airs_raw_tc = _find_first_dset(cal_h5, gpath, "raw")
                    airs_cal_tc = _find_first_dset(cal_h5, gpath, "cal")
                    break
            if airs_raw_tc is None and raw_h5 is not None:
                for grp in ("AIRS", "airs"):
                    if grp in raw_h5:
                        airs_raw_tc = _find_first_dset(raw_h5, f"/{grp}", "raw")
                        if airs_time is None:
                            airs_time = _find_first_dset(raw_h5, f"/{grp}", "time")
                        break
            airs_time   = _select_planet(airs_time,   planet_idx) if airs_time   is not None else None
            airs_raw_tc = _select_planet(airs_raw_tc, planet_idx) if airs_raw_tc is not None else None
            airs_cal_tc = _select_planet(airs_cal_tc, planet_idx) if airs_cal_tc is not None else None

            # wavelength choice
            if isinstance(airs_wav_idx, str) and airs_wav_idx.lower() == "auto":
                if airs_cal_tc is not None and airs_cal_tc.ndim >= 2:
                    wav_idx = _auto_pick_wavelength(airs_cal_tc)
                elif airs_raw_tc is not None and airs_raw_tc.ndim >= 2:
                    wav_idx = _auto_pick_wavelength(airs_raw_tc)
                else:
                    wav_idx = 0
            else:
                wav_idx = int(airs_wav_idx or 0)

            def _slice_w(tc: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if tc is None:
                    return None
                arr = tc
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim == 2:
                    w = int(np.clip(wav_idx, 0, arr.shape[1] - 1))
                    return arr[:, w]
                return arr  # already 1D

            airs_raw = _slice_w(airs_raw_tc)
            airs_cal = _slice_w(airs_cal_tc)

            if airs_time is None:
                n = airs_cal.size if (airs_cal is not None) else (airs_raw.size if airs_raw is not None else 100)
                airs_time = np.arange(n)

            fgs1 = CurvePack(time=np.asarray(fgs1_time), raw=fgs1_raw, cal=fgs1_cal)
            airs  = CurvePack(time=np.asarray(airs_time), raw=airs_raw, cal=airs_cal)
            return fgs1, airs
    finally:
        if raw_h5 is not None:
            raw_h5.close()


# --------------------- Plotting ---------------------------------------------------------------

def plot_phase_alignment(
    fgs1: CurvePack,
    airs: CurvePack,
    out_path: str,
    rolling: int,
    period: float,
    t0_opt: Optional[float],
    phase_window: float,
    title_suffix: str,
) -> None:
    """
    Build a 2x2 figure:
      [0,0] Phase-aligned FGS1 raw vs cal (with smoothed overlays)
      [0,1] Phase-aligned AIRS (representative λ) raw vs cal
      [1,0] Cross-correlation (FGS1 raw vs cal) lag curve with peak annotated
      [1,1] Jitter/residual RMS before/after summary with small hist
    """
    # Prepare FGS1 signals
    t_f = np.asarray(fgs1.time, float)
    yfr = robust_norm(fgs1.raw) if fgs1.raw is not None else None
    yfc = robust_norm(fgs1.cal) if fgs1.cal is not None else None

    # Prepare AIRS signals (optional)
    t_a = np.asarray(airs.time, float)
    yar = robust_norm(airs.raw) if airs.raw is not None else None
    yac = robust_norm(airs.cal) if airs.cal is not None else None

    # Estimate t0 if needed (via calibrated FGS1 minimum)
    t0 = float(t0_opt) if t0_opt is not None else (
        estimate_t0(t_f, yfc if yfc is not None else (yfr if yfr is not None else t_f), rolling=rolling)
    )

    # Phase
    phi_f = phase_from_time(t_f, period=period, t0=t0)
    phi_a = phase_from_time(t_a, period=period, t0=t0)
    if phi_f is None:
        phi_f = local_phase(t_f, t0)
    if phi_a is None:
        phi_a = local_phase(t_a, t0)

    # Optional crop to +/- phase_window for clarity
    def crop(phi, *ys):
        if phi.size == 0:
            return (phi,) + ys
        m = np.abs(phi) <= phase_window if phase_window > 0 else np.ones_like(phi, dtype=bool)
        out = (phi[m],)
        for arr in ys:
            out += ((arr[m] if (arr is not None and arr.size == phi.size) else arr),)
        return out

    phi_f, yfr, yfc = crop(phi_f, yfr, yfc)
    phi_a, yar, yac = crop(phi_a, yar, yac)

    # Smoothed versions
    yfr_s = rolling_median(yfr, rolling) if yfr is not None else None
    yfc_s = rolling_median(yfc, rolling) if yfc is not None else None
    yar_s = rolling_median(yar, rolling) if yar is not None else None
    yac_s = rolling_median(yac, rolling) if yac is not None else None

    # Cross-correlation lag (FGS1 raw vs cal) on the overlapping domain
    lag_time, corr, lag_pk = (np.array([0.0]), np.array([0.0]), 0.0)
    if yfr is not None and yfc is not None:
        # Use original time grid for lag calculation to preserve dt
        lag_time, corr, lag_pk = cross_correlation_lag(
            (fgs1.raw if fgs1.raw is not None else yfr),
            (fgs1.cal if fgs1.cal is not None else yfc),
            t_f
        )

    # Residual/jitter: subtract smoothed curve
    resid_raw = (yfr - yfr_s) if (yfr is not None and yfr_s is not None) else None
    resid_cal = (yfc - yfc_s) if (yfc is not None and yfc_s is not None) else None
    rms_raw = robust_rms(resid_raw) if resid_raw is not None else np.nan
    rms_cal = robust_rms(resid_cal) if resid_cal is not None else np.nan

    # ---- Figure
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 0.9], width_ratios=[1.1, 1.0])

    # [0,0] FGS1 phase plot
    ax_f = fig.add_subplot(gs[0, 0])
    ax_f.set_title("FGS1 — Phase Alignment (raw vs calibrated)" + title_suffix, fontsize=12, weight="bold")
    if yfr is not None:
        ax_f.plot(phi_f, yfr, color="#8a8aff", alpha=0.55, lw=1.0, label="raw")
        if yfr_s is not None:
            ax_f.plot(phi_f, yfr_s, color="#5a5aff", alpha=0.9, lw=1.5, label=f"raw median (w={rolling})")
    if yfc is not None:
        ax_f.plot(phi_f, yfc, color="#20c15b", alpha=0.65, lw=1.0, label="calibrated")
        if yfc_s is not None:
            ax_f.plot(phi_f, yfc_s, color="#158f42", alpha=0.95, lw=1.5, label=f"cal median (w={rolling})")

    ax_f.set_xlabel("Phase" if period and period > 0 else "Relative phase (centered)")
    ax_f.set_ylabel("Flux (median-normalized)")
    ax_f.legend(loc="best", fontsize=9)
    ax_f.grid(alpha=0.25)

    # [0,1] AIRS phase plot
    ax_a = fig.add_subplot(gs[0, 1])
    ax_a.set_title("AIRS (representative λ) — Phase Alignment (raw vs calibrated)" + title_suffix,
                   fontsize=12, weight="bold")
    if yar is not None:
        ax_a.plot(phi_a, yar, color="#ff8c8c", alpha=0.55, lw=1.0, label="raw")
        if yar_s is not None:
            ax_a.plot(phi_a, yar_s, color="#ff5e5e", alpha=0.9, lw=1.5, label=f"raw median (w={rolling})")
    if yac is not None:
        ax_a.plot(phi_a, yac, color="#36b3d6", alpha=0.65, lw=1.0, label="calibrated")
        if yac_s is not None:
            ax_a.plot(phi_a, yac_s, color="#1d87a7", alpha=0.95, lw=1.5, label=f"cal median (w={rolling})")
    ax_a.set_xlabel("Phase" if period and period > 0 else "Relative phase (centered)")
    ax_a.set_ylabel("Flux (median-normalized)")
    ax_a.legend(loc="best", fontsize=9)
    ax_a.grid(alpha=0.25)

    # [1,0] Cross-correlation lag
    ax_cc = fig.add_subplot(gs[1, 0])
    ax_cc.set_title("FGS1 Cross-Correlation (raw vs calibrated)", fontsize=11, weight="bold")
    if lag_time.size > 1:
        ax_cc.plot(lag_time, corr, color="#444", lw=1.2)
        ax_cc.axvline(lag_pk, color="#f59e0b", ls="--", lw=1.2, alpha=0.9, label=f"peak lag ~ {lag_pk:.4g} time units")
        ax_cc.set_xlabel("Lag (time units)")
        ax_cc.set_ylabel("Normalized Cross-Correlation")
        ax_cc.legend(loc="best", fontsize=9)
        ax_cc.grid(alpha=0.25)
    else:
        ax_cc.text(0.5, 0.5, "Insufficient data for cross-correlation", ha="center", va="center",
                   transform=ax_cc.transAxes, color="#888")
        ax_cc.set_axis_off()

    # [1,1] Residual RMS + small hist
    ax_r = fig.add_subplot(gs[1, 1])
    ax_r.set_title("FGS1 Residual/Jitter (median-detrended)", fontsize=11, weight="bold")
    text_lines = []
    if not np.isnan(rms_raw):
        text_lines.append(f"RMS(raw residual)  : {rms_raw:.4g}")
    if not np.isnan(rms_cal):
        text_lines.append(f"RMS(cal residual)  : {rms_cal:.4g}")
    if not np.isnan(rms_raw) and not np.isnan(rms_cal):
        improv = (rms_raw - rms_cal) / (rms_raw + 1e-12) * 100.0
        text_lines.append(f"Improvement (↓RMS): {improv:.2f}%")

    # Draw small histograms if available
    if resid_raw is not None or resid_cal is not None:
        bins = 50
        if resid_raw is not None:
            ax_r.hist(resid_raw, bins=bins, alpha=0.5, color="#8a8aff", edgecolor="white", label="raw residual")
        if resid_cal is not None:
            ax_r.hist(resid_cal, bins=bins, alpha=0.6, color="#20c15b", edgecolor="white", label="cal residual")
        ax_r.legend(loc="best", fontsize=9)
        ax_r.set_xlabel("Residual (ppm, approx.)")
        ax_r.set_ylabel("Count")
        ax_r.grid(alpha=0.15)
    else:
        ax_r.text(0.5, 0.5, "No residuals available", ha="center", va="center",
                  transform=ax_r.transAxes, color="#888")
        ax_r.set_axis_off()

    # Footer
    footer = []
    footer.append(f"t0 = {t0:.6g} (estimated)" if t0_opt is None else f"t0 = {t0:.6g} (user)")
    footer.append(f"period = {period:.6g}" if period and period > 0 else "period = unknown (relative phase)")
    if lag_time.size > 1:
        footer.append(f"xcorr peak lag ≈ {lag_pk:.6g} time units")
    if text_lines:
        footer += text_lines
    fig.text(0.01, 0.01, " | ".join(footer), fontsize=8.5, color="#777", ha="left", va="bottom")

    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# --------------------- CLI -------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate calibration_plots/phase_alignment.png (phase fold & jitter diagnostics).",
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
    p.add_argument("--rolling", type=int, default=101,
                   help="Rolling median window (odd) for smoothing.")
    p.add_argument("--period", type=float, default=0.0,
                   help="Orbital period for phase folding. If <=0, use relative phase around t0.")
    p.add_argument("--t0", type=float, default=None,
                   help="Epoch/center time for phase folding. If None, estimated from calibrated FGS1.")
    p.add_argument("--phase-window", type=float, default=0.6,
                   help="Show only |phase| <= this value if >0 (for clarity).")
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
    out_path = os.path.join(args.outdir, "phase_alignment.png")

    try:
        plot_phase_alignment(
            fgs1=fgs1,
            airs=airs,
            out_path=out_path,
            rolling=int(args.rolling),
            period=float(args.period),
            t0_opt=(args.t0 if args.t0 is not None else None),
            phase_window=float(args.phase_window),
            title_suffix=title_suffix,
        )
        print(f"[ok] Wrote {out_path}")
        return 0
    except Exception as e:
        print(f"[error] Plotting failed: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())