#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_fft_alignment.py

SpectraMind V50 — Calibration Plot Generator (FFT Alignment) • Rebuilt

Overview
--------
Generates `calibration_plots/fft_alignment.png` comparing pre- and post-calibration
frequency spectra for FGS1 and AIRS (one representative wavelength channel). Also
supports CSV export of detected peak frequencies/amplitudes and optional log-scale
y-axis for clarity on weak bands.

Design improvements (this rebuild)
----------------------------------
1) Hardened HDF5 discovery
   • Heuristics for planet-leading dimension selection
   • Robust wavelength auto-pick for AIRS via max temporal variance
   • Graceful fallbacks for missing datasets/time vectors

2) FFT engineering for diagnostics (not physics-grade periodogram)
   • Median-detrend + Hann window
   • One-sided amplitude spectrum from rFFT with 2/N normalization
   • Median-Δt frequency axis for uneven sampling
   • Top-K peak detection and optional notch overlays

3) CLI & logging
   • Headless Matplotlib for CI/Kaggle
   • Structured logging (INFO/WARN/ERROR)
   • Options: --ylog, --dpi, --save-csv, --notches
   • Optional annotation from calibration JSON (smoothness/alignment)

Inputs
------
- Calibrated HDF5 (default: outputs/calibrated/lightcurves.h5)
  Expected flexible groups:
    /FGS1/{time, raw, cal}, /AIRS/{time, raw, cal}
  Supported shapes:
    FGS1: (T,) or (P,T)
    AIRS : (T,W), (P,T,W) or (T,) after λ selection

- Optional raw HDF5 (to fetch 'raw' series if cal H5 lacks them)

Outputs
-------
- <outdir>/fft_alignment.png
- (optional) <outdir>/fft_alignment_peaks.csv

Usage
-----
  python tools/generate_fft_alignment.py \
      --cal-h5 outputs/calibrated/lightcurves.h5 \
      --raw-h5 data/raw/lightcurves_raw.h5 \
      --outdir calibration_plots \
      --planet-idx 0 \
      --airs-wav-idx auto \
      --topk 5 \
      --notches 0.001,0.0025 \
      --ylog \
      --save-csv

Notes
-----
• Frequency units follow input time units. With uneven sampling, we estimate Δt via median.
• This is a diagnostic visual, not a physics-grade spectral estimator.

Author
------
SpectraMind V50 Team — NeurIPS 2025 Ariel Data Challenge
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import h5py
import matplotlib

# Headless backend for CI and non-GUI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------- Logging ---------------------------------------------------------------

LOG = logging.getLogger("fft_alignment")


def setup_logging(verbosity: int = 1) -> None:
    """Set up root logger with level based on verbosity (0=WARNING,1=INFO,2=DEBUG)."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "[%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    LOG.setLevel(level)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)


# --------------------- Small utilities -------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def median_detrend(y: np.ndarray) -> np.ndarray:
    """Remove median to suppress DC."""
    med = np.nanmedian(y)
    return y - med if np.isfinite(med) else y


def hann_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n))
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def estimate_dt(t: np.ndarray) -> float:
    """Robust Δt via median of diffs. Fall back to 1.0 if invalid."""
    t = np.asarray(t, dtype=float)
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
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    n = t.size
    if n < 8:
        LOG.warning("Too few samples for FFT (n=%d). Returning degenerate spectrum.", n)
        return np.array([0.0]), np.array([0.0])

    dt = estimate_dt(t)
    yw = median_detrend(y) * hann_window(n)
    Y = np.fft.rfft(yw, n=n)
    freq = np.fft.rfftfreq(n, d=dt)
    amp = (2.0 / n) * np.abs(Y)  # one-sided normalization
    return freq, amp


def top_k_peaks(freq: np.ndarray, amp: np.ndarray, k: int, fmin: float = 0.0) -> List[int]:
    """Return indices of top-k peaks above minimum frequency."""
    if k <= 0:
        return []
    mask = freq >= fmin
    if not np.any(mask):
        return []
    f = freq[mask]
    a = amp[mask].copy()
    if f[0] == 0:
        a[0] = 0.0  # drop DC
    picks = np.argsort(a)[::-1][:k]
    base = np.where(mask)[0][0]
    return (base + picks).tolist()


# --------------------- HDF5 discovery --------------------------------------------------------

NAME_CANDIDATES = {
    "time": [r"time", r"t", r"phase"],
    "raw":  [r"raw", r"uncal", r"preproc", r"before"],
    "cal":  [r"cal", r"calib", r"proc", r"after"],
}


def _match_name(name: str, patterns: list[str]) -> bool:
    lname = name.lower()
    return any(
        re.fullmatch(pat, lname) or re.search(fr"(^|/|_){pat}($|/|_)", lname)
        for pat in patterns
    )


def _find_first_dset(h5: h5py.File, group_path: str, key: str) -> Optional[np.ndarray]:
    """Find the first dataset under group_path whose name matches common aliases."""
    if group_path not in h5:
        return None
    grp = h5[group_path]

    # Direct children
    for k, obj in grp.items():
        if isinstance(obj, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
            return obj[()]

    # DFS for nested layouts
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
    """
    Heuristic: a leading 'planet' dimension tends to be the smallest high-level batch dim and
    smaller than the product of trailing dims.
    """
    if arr.ndim < 2:
        return False
    P = arr.shape[0]
    rest = int(np.prod(arr.shape[1:]))
    return 1 < P < 100_000 and P < rest


def _select_planet(arr: Optional[np.ndarray], planet_idx: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim >= 2 and _likely_planet_leading_dim(arr):
        if not (0 <= planet_idx < arr.shape[0]):
            raise IndexError(f"planet_idx {planet_idx} out of range [0, {arr.shape[0]-1}]")
        return arr[planet_idx]
    return arr


def _auto_pick_wavelength(airs_tc: np.ndarray) -> int:
    """Select wavelength index by maximal temporal variance."""
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
    """Load FGS1 and AIRS curves from HDF5 files with robust fallbacks."""
    if not os.path.isfile(cal_h5_path):
        raise FileNotFoundError(f"Calibrated H5 not found: {cal_h5_path}")

    raw_h5 = h5py.File(raw_h5_path, "r") if (raw_h5_path and os.path.isfile(raw_h5_path)) else None
    try:
        with h5py.File(cal_h5_path, "r") as cal_h5:
            # -------- FGS1 --------
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

            # Provide time fallback
            if fgs1_time is None:
                n = fgs1_cal.size if (fgs1_cal is not None) else (fgs1_raw.size if fgs1_raw is not None else 100)
                fgs1_time = np.arange(n)

            # -------- AIRS --------
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

            # Choose wavelength index
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

            # Provide time fallback
            if airs_time is None:
                n = airs_cal.size if (airs_cal is not None) else (airs_raw.size if airs_raw is not None else 100)
                airs_time = np.arange(n)

            fgs1 = CurvePack(time=np.asarray(fgs1_time), raw=fgs1_raw, cal=fgs1_cal)
            airs  = CurvePack(time=np.asarray(airs_time), raw=airs_raw, cal=airs_cal)
            return fgs1, airs
    finally:
        if raw_h5 is not None:
            raw_h5.close()


# --------------------- Plotting --------------------------------------------------------------

def plot_fft_alignment(
    fgs1: CurvePack,
    airs: CurvePack,
    out_path: str,
    topk: int,
    notch_freqs: Optional[List[float]],
    title_suffix: str,
    ylog: bool,
    dpi: int,
    annotate: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce the side-by-side FFT amplitude comparison.
    Returns (fgs1_f, fgs1_a, airs_f, airs_a) for optional CSV saving.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.5), constrained_layout=True)
    ax1, ax2 = axes

    # Compute spectra
    fgs1_fr = fgs1_ar = fgs1_fc = fgs1_ac = None
    airs_fr = airs_ar = airs_fc = airs_ac = None

    # --- FGS1 ---
    ax1.set_title("FGS1 — FFT Amplitude (raw vs calibrated)" + title_suffix, fontsize=12, weight="bold")
    if fgs1.raw is not None:
        fgs1_fr, fgs1_ar = compute_fft_amp(fgs1.time, fgs1.raw)
        ax1.plot(fgs1_fr, fgs1_ar, color="#8a8aff", lw=1.25, alpha=0.95, label="raw")
        for i in top_k_peaks(fgs1_fr, fgs1_ar, topk, fmin=0.0):
            ax1.plot([fgs1_fr[i]], [fgs1_ar[i]], "o", color="#5a5aff", ms=4)
    if fgs1.cal is not None:
        fgs1_fc, fgs1_ac = compute_fft_amp(fgs1.time, fgs1.cal)
        ax1.plot(fgs1_fc, fgs1_ac, color="#20c15b", lw=1.35, alpha=0.95, label="calibrated")
        for i in top_k_peaks(fgs1_fc, fgs1_ac, topk, fmin=0.0):
            ax1.plot([fgs1_fc[i]], [fgs1_ac[i]], "o", color="#158f42", ms=4)

    if notch_freqs:
        for nf in notch_freqs:
            ax1.axvline(nf, color="#ffb347", ls="--", lw=1.0, alpha=0.85)

    if ylog:
        ax1.set_yscale("log")
    ax1.set_xlabel("Frequency (cycles / time-unit)")
    ax1.set_ylabel("Amplitude (arb. norm.)")
    ax1.set_xlim(left=0.0)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # --- AIRS ---
    ax2.set_title("AIRS (representative λ) — FFT Amplitude (raw vs calibrated)" + title_suffix, fontsize=12, weight="bold")
    if airs.raw is not None:
        airs_fr, airs_ar = compute_fft_amp(airs.time, airs.raw)
        ax2.plot(airs_fr, airs_ar, color="#ff8c8c", lw=1.25, alpha=0.95, label="raw")
        for i in top_k_peaks(airs_fr, airs_ar, topk, fmin=0.0):
            ax2.plot([airs_fr[i]], [airs_ar[i]], "o", color="#ff5e5e", ms=4)
    if airs.cal is not None:
        airs_fc, airs_ac = compute_fft_amp(airs.time, airs.cal)
        ax2.plot(airs_fc, airs_ac, color="#36b3d6", lw=1.35, alpha=0.95, label="calibrated")
        for i in top_k_peaks(airs_fc, airs_ac, topk, fmin=0.0):
            ax2.plot([airs_fc[i]], [airs_ac[i]], "o", color="#1d87a7", ms=4)

    if notch_freqs:
        for nf in notch_freqs:
            ax2.axvline(nf, color="#ffb347", ls="--", lw=1.0, alpha=0.85)

    if ylog:
        ax2.set_yscale("log")
    ax2.set_xlabel("Frequency (cycles / time-unit)")
    ax2.set_ylabel("Amplitude (arb. norm.)")
    ax2.set_xlim(left=0.0)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # Footer & annotation
    footer = "Hann-windowed, median-detrended amplitude spectra; frequency via median Δt. Dashed lines = notch markers."
    if annotate:
        extra = []
        if "fft_smoothness" in annotate:
            extra.append(f"smoothness={annotate['fft_smoothness']:.3f}")
        if "photonic_alignment_score" in annotate:
            extra.append(f"photonic_alignment={annotate['photonic_alignment_score']:.3f}")
        if extra:
            footer += "  |  " + "  •  ".join(extra)
    fig.text(0.01, 0.01, footer, fontsize=8, color="#777777", va="bottom", ha="left")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Return the most relevant spectra (calibrated preferred, else raw) for CSV saving.
    fgs_f = fgs1_fc if fgs1_fc is not None else (fgs1_fr if fgs1_fr is not None else np.array([]))
    fgs_a = fgs1_ac if fgs1_ac is not None else (fgs1_ar if fgs1_ar is not None else np.array([]))
    air_f = airs_fc if airs_fc is not None else (airs_fr if airs_fr is not None else np.array([]))
    air_a = airs_ac if airs_ac is not None else (airs_ar if airs_ar is not None else np.array([]))

    return fgs_f, fgs_a, air_f, air_a


# --------------------- CSV helpers -----------------------------------------------------------

def save_peaks_csv(
    out_csv: str,
    label: str,
    freq: np.ndarray,
    amp: np.ndarray,
    topk: int,
) -> None:
    """Save top-K peak list as CSV: columns = [series, rank, freq, amp]."""
    if freq.size == 0 or amp.size == 0 or topk <= 0:
        return
    ranks = top_k_peaks(freq, amp, topk, fmin=0.0)
    with open(out_csv, "a", encoding="utf-8") as f:
        if os.stat(out_csv).st_size == 0:
            f.write("series,rank,freq,amp\n")
        for r, idx in enumerate(ranks, start=1):
            f.write(f"{label},{r},{freq[idx]:.12g},{amp[idx]:.12g}\n")


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
                   help="Output directory for the PNG/CSV.")
    p.add_argument("--planet-idx", type=int, default=0,
                   help="Planet index to visualize if arrays are stacked by planet.")
    p.add_argument("--airs-wav-idx", default="auto",
                   help="AIRS wavelength index (int) or 'auto' to pick a representative channel.")
    p.add_argument("--topk", type=int, default=5,
                   help="Highlight top-K peaks in each spectrum.")
    p.add_argument("--notches", type=str, default=None,
                   help="Comma-separated list of notch frequencies, e.g. '0.001,0.0025'.")
    p.add_argument("--ylog", action="store_true",
                   help="Use logarithmic y-scale for amplitude.")
    p.add_argument("--save-csv", action="store_true",
                   help="Export peak lists to <outdir>/fft_alignment_peaks.csv.")
    p.add_argument("--dpi", type=int, default=170,
                   help="Output figure DPI.")
    p.add_argument("--annot-calibration-json", type=str, default=None,
                   help="Optional JSON (e.g., logs/calibration.json) to annotate footer with metrics.")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="Increase verbosity: -v (INFO), -vv (DEBUG); -q for warnings only.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Quiet mode (warnings only).")
    return p.parse_args(argv)


def load_annotation(json_path: Optional[str]) -> Optional[dict]:
    if not json_path:
        return None
    if not os.path.isfile(json_path):
        LOG.warning("Annotation JSON not found: %s", json_path)
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ann = {}
        # Try common keys used in our calibration log format
        if "diagnostics" in data and isinstance(data["diagnostics"], dict):
            diag = data["diagnostics"]
            if "fft_smoothness" in diag:
                ann["fft_smoothness"] = diag["fft_smoothness"]
        # Sometimes photonic alignment appears under 'symbolic_constraint_violations' or similar
        for k in ("symbolic_constraint_violations",):
            if k in data and isinstance(data[k], dict):
                sc = data[k]
                if "photonic_alignment_score" in sc:
                    ann["photonic_alignment_score"] = sc["photonic_alignment_score"]
        # Also allow at root level
        for k in ("fft_smoothness", "photonic_alignment_score"):
            if k in data and k not in ann:
                ann[k] = data[k]
        return ann or None
    except Exception as e:
        LOG.warning("Failed to parse annotation JSON: %s", e)
        return None


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Logging level
    if args.quiet:
        setup_logging(verbosity=0)
    else:
        setup_logging(verbosity=min(2, max(0, args.verbose)))

    ensure_dir(args.outdir)

    # Parse notches
    notches = None
    if args.notches:
        try:
            notches = [float(x.strip()) for x in args.notches.split(",") if x.strip()]
            LOG.info("Using notch markers at: %s", ", ".join(f"{nf:g}" for nf in notches))
        except Exception:
            LOG.warning("Failed to parse --notches; ignoring.")
            notches = None

    # Load curves
    try:
        fgs1, airs = discover_curves(
            cal_h5_path=args.cal_h5,
            raw_h5_path=args.raw_h5,
            planet_idx=args.planet_idx,
            airs_wav_idx=args.airs_wav_idx,
        )
        LOG.info("Loaded curves | FGS1: time=%s raw=%s cal=%s | AIRS: time=%s raw=%s cal=%s",
                 fgs1.time.shape, None if fgs1.raw is None else fgs1.raw.shape, None if fgs1.cal is None else fgs1.cal.shape,
                 airs.time.shape,  None if airs.raw is None else airs.raw.shape,  None if airs.cal is None else airs.cal.shape)
    except Exception as e:
        LOG.error("Failed to read curves: %s", e)
        return 2

    title_suffix = f"  (planet idx={args.planet-idx if hasattr(args,'planet-idx') else args.planet_idx}, AIRS λ={args.airs_wav_idx})"
    # Python identifiers can't have hyphens; ensure above renders properly
    title_suffix = f"  (planet idx={args.planet_idx}, AIRS λ={args.airs_wav_idx})"

    out_png = os.path.join(args.outdir, "fft_alignment.png")
    ann = load_annotation(args.annot_calibration_json)

    try:
        fgs_f, fgs_a, air_f, air_a = plot_fft_alignment(
            fgs1=fgs1,
            airs=airs,
            out_path=out_png,
            topk=max(0, int(args.topk)),
            notch_freqs=notches,
            title_suffix=title_suffix,
            ylog=bool(args.ylog),
            dpi=max(50, int(args.dpi)),
            annotate=ann,
        )
        LOG.info("Wrote %s", out_png)
    except Exception as e:
        LOG.error("Plotting failed: %s", e)
        return 3

    # CSV export
    if args.save_csv:
        out_csv = os.path.join(args.outdir, "fft_alignment_peaks.csv")
        try:
            # Ensure (re)create file if first time
            if not os.path.exists(out_csv):
                open(out_csv, "w", encoding="utf-8").close()
            save_peaks_csv(out_csv, "FGS1", fgs_f, fgs_a, max(0, int(args.topk)))
            save_peaks_csv(out_csv, "AIRS",  air_f, air_a, max(0, int(args.topk)))
            LOG.info("Saved peaks CSV: %s", out_csv)
        except Exception as e:
            LOG.warning("Failed to save peaks CSV: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())