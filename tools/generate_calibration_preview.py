#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_calibration_preview.py

SpectraMind V50 — Calibration Plot Generator (Preview Overview)

Purpose
-------
Generate `calibration_plots/calibration_preview.png` to quickly assess the effect
of calibration across instruments. The figure includes:

  • FGS1 panel: raw vs calibrated light curve (and residuals).
  • AIRS panel: raw vs calibrated for a representative wavelength channel
    (auto-picked by variance, or user-selected), plus residuals.
  • Residual distributions (FGS1 & AIRS) with robust spread statistics.
  • Coverage/flag summary if per-bin flags are available (optional).

Inputs
------
- Calibrated HDF5 (default: outputs/calibrated/lightcurves.h5)
  Expected (but flexible) groups and dataset names (auto-discovered with heuristics):
      /FGS1/{time,raw,cal}, /AIRS/{time,raw,cal}
  Shapes supported:
      FGS1: (T,) or (P,T)
      AIRS : (T,W), (P,T,W)
- Optional raw HDF5 to backfill 'raw' series if the cal file lacks them.

Outputs
-------
- PNG at: <outdir>/calibration_preview.png

Notes
-----
- Uses median-detrending for residuals (y - median(y_raw)) to emphasize shape changes.
- If time spacing is irregular, plotting uses native sample order; no resampling is done.
- If AIRS has multiple wavelengths, 'auto' picks the channel with maximum temporal variance.

Usage
-----
  python tools/generate_calibration_preview.py \
      --cal-h5 outputs/calibrated/lightcurves.h5 \
      --raw-h5 data/raw/lightcurves_raw.h5 \
      --outdir calibration_plots \
      --planet-idx 0 \
      --airs-wav-idx auto \
      --title "Calibration Preview — Run abc123"

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
from typing import Optional, Tuple, List, Dict

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


def robust_median(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else 0.0


def robust_scale(x: np.ndarray) -> float:
    """Return a robust scale estimate: 1.4826 * MAD (median absolute deviation)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def median_detrend(y: np.ndarray, y_ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Subtract median of reference (or self) to emphasize shape over offset."""
    base = y_ref if y_ref is not None else y
    med = robust_median(base)
    return np.asarray(y, float) - med


def auto_variance_channel(tc: np.ndarray) -> int:
    """Pick wavelength index by maximum temporal variance."""
    # tc shape: (T, W)
    vari = np.nanvar(tc, axis=0)
    return int(np.nanargmax(vari))


# --------------------- HDF5 Discovery ---------------------------------------------------------

NAME_CANDIDATES = {
    "time": [r"time", r"t", r"phase"],
    "raw":  [r"raw", r"uncal", r"preproc", r"before"],
    "cal":  [r"cal", r"calib", r"proc", r"after"],
    "flag": [r"flag", r"flags", r"mask", r"bad", r"quality"]
}

def _match_name(name: str, patterns: List[str]) -> bool:
    lname = name.lower()
    return any(re.fullmatch(pat, lname) or re.search(fr"(^|/|_){pat}($|/|_)", lname) for pat in patterns)

def _find_first_dset(h5: h5py.File, group_path: str, key: str) -> Optional[np.ndarray]:
    if group_path not in h5:
        return None
    grp = h5[group_path]
    # Direct children
    for k, obj in grp.items():
        if isinstance(obj, h5py.Dataset) and _match_name(k, NAME_CANDIDATES[key]):
            return obj[()]
    # Deep search
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

def _select_planet(arr: np.ndarray, planet_idx: int) -> np.ndarray:
    if arr is None:
        return arr
    a = np.asarray(arr)
    if a.ndim >= 2 and a.shape[0] > 1 and a.shape[0] < a.size:
        if not (0 <= planet_idx < a.shape[0]):
            raise IndexError(f"planet_idx {planet_idx} out of range 0..{a.shape[0]-1}")
        return a[planet_idx]
    return a

@dataclass
class CurvePack:
    time: np.ndarray
    raw: Optional[np.ndarray]
    cal: Optional[np.ndarray]
    flags: Optional[np.ndarray] = None  # optional 0/1 (good/bad) or boolean per-sample/ per-bin


def discover_curves(cal_h5_path: str,
                    raw_h5_path: Optional[str],
                    planet_idx: int,
                    airs_wav_idx: Optional[int | str]) -> Tuple[CurvePack, CurvePack, Dict[str, int]]:
    if not os.path.isfile(cal_h5_path):
        raise FileNotFoundError(f"Calibrated H5 not found: {cal_h5_path}")
    raw_h5 = None
    if raw_h5_path and os.path.isfile(raw_h5_path):
        raw_h5 = h5py.File(raw_h5_path, "r")

    meta = {}

    with h5py.File(cal_h5_path, "r") as cal_h5:
        # ------------- FGS1 -------------
        fgs1_time = fgs1_raw = fgs1_cal = fgs1_flag = None
        for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
            gkey = grp.strip("/")
            if gkey in cal_h5:
                gpath = f"/{gkey}"
                fgs1_time = _find_first_dset(cal_h5, gpath, "time")
                fgs1_raw  = _find_first_dset(cal_h5, gpath, "raw")
                fgs1_cal  = _find_first_dset(cal_h5, gpath, "cal")
                fgs1_flag = _find_first_dset(cal_h5, gpath, "flag")
                break
        if fgs1_raw is None and raw_h5 is not None:
            for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
                gkey = grp.strip("/")
                if gkey in raw_h5:
                    fgs1_raw = _find_first_dset(raw_h5, f"/{gkey}", "raw")
                    if fgs1_time is None:
                        fgs1_time = _find_first_dset(raw_h5, f"/{gkey}", "time")
                    if fgs1_flag is None:
                        fgs1_flag = _find_first_dset(raw_h5, f"/{gkey}", "flag")
                    break
        if fgs1_time is not None and fgs1_time.ndim > 1:
            fgs1_time = _select_planet(fgs1_time, planet_idx)
        if fgs1_raw is not None and fgs1_raw.ndim > 1:
            fgs1_raw = _select_planet(fgs1_raw, planet_idx)
        if fgs1_cal is not None and fgs1_cal.ndim > 1:
            fgs1_cal = _select_planet(fgs1_cal, planet_idx)
        if fgs1_flag is not None and fgs1_flag.ndim > 1:
            fgs1_flag = _select_planet(fgs1_flag, planet_idx)

        # ------------- AIRS -------------
        airs_time = airs_raw_tc = airs_cal_tc = airs_flag_tc = None
        for grp in ("AIRS", "airs", "/AIRS", "/airs"):
            gkey = grp.strip("/")
            if gkey in cal_h5:
                gpath = f"/{gkey}"
                airs_time    = _find_first_dset(cal_h5, gpath, "time")
                airs_raw_tc  = _find_first_dset(cal_h5, gpath, "raw")
                airs_cal_tc  = _find_first_dset(cal_h5, gpath, "cal")
                airs_flag_tc = _find_first_dset(cal_h5, gpath, "flag")
                break
        if airs_raw_tc is None and raw_h5 is not None:
            for grp in ("AIRS", "airs", "/AIRS", "/airs"):
                gkey = grp.strip("/")
                if gkey in raw_h5:
                    airs_raw_tc = _find_first_dset(raw_h5, f"/{gkey}", "raw")
                    if airs_time is None:
                        airs_time = _find_first_dset(raw_h5, f"/{gkey}", "time")
                    if airs_flag_tc is None:
                        airs_flag_tc = _find_first_dset(raw_h5, f"/{gkey}", "flag")
                    break
        if airs_time is not None and airs_time.ndim > 1:
            airs_time = _select_planet(airs_time, planet_idx)
        if airs_raw_tc is not None and airs_raw_tc.ndim >= 2:
            airs_raw_tc = _select_planet(airs_raw_tc, planet_idx)  # (T,W)
        if airs_cal_tc is not None and airs_cal_tc.ndim >= 2:
            airs_cal_tc = _select_planet(airs_cal_tc, planet_idx)  # (T,W)
        if airs_flag_tc is not None and airs_flag_tc.ndim >= 2:
            airs_flag_tc = _select_planet(airs_flag_tc, planet_idx)  # (T,W) or (W,) etc.

        # wavelength choice
        if isinstance(airs_wav_idx, str) and airs_wav_idx.lower() == "auto":
            if airs_cal_tc is not None and airs_cal_tc.ndim == 2:
                wav_idx = auto_variance_channel(airs_cal_tc)
            elif airs_raw_tc is not None and airs_raw_tc.ndim == 2:
                wav_idx = auto_variance_channel(airs_raw_tc)
            else:
                wav_idx = 0
        else:
            wav_idx = int(airs_wav_idx or 0)
        meta["airs_wav_idx"] = wav_idx

        # Extract 1D AIRS series
        airs_raw = airs_cal = airs_flag = None
        if airs_raw_tc is not None and airs_raw_tc.ndim == 2:
            wav_idx = int(np.clip(wav_idx, 0, airs_raw_tc.shape[1] - 1))
            airs_raw = airs_raw_tc[:, wav_idx]
        elif airs_raw_tc is not None and airs_raw_tc.ndim == 1:
            airs_raw = airs_raw_tc
        if airs_cal_tc is not None and airs_cal_tc.ndim == 2:
            wav_idx = int(np.clip(wav_idx, 0, airs_cal_tc.shape[1] - 1))
            airs_cal = airs_cal_tc[:, wav_idx]
        elif airs_cal_tc is not None and airs_cal_tc.ndim == 1:
            airs_cal = airs_cal_tc
        if airs_flag_tc is not None:
            # try to align flags per time or per (time,wav)
            a = np.asarray(airs_flag_tc)
            if a.ndim == 2 and a.shape[1] == (airs_raw_tc.shape[1] if airs_raw_tc is not None else a.shape[1]):
                airs_flag = a[:, wav_idx]
            elif a.ndim == 1:
                airs_flag = a

        fgs1 = CurvePack(
            time=fgs1_time if fgs1_time is not None else (np.arange(len(fgs1_cal)) if fgs1_cal is not None else np.arange(100)),
            raw=fgs1_raw,
            cal=fgs1_cal,
            flags=fgs1_flag
        )
        airs = CurvePack(
            time=airs_time if airs_time is not None else (np.arange(len(airs_cal)) if airs_cal is not None else np.arange(100)),
            raw=airs_raw,
            cal=airs_cal,
            flags=airs_flag
        )

        return fgs1, airs, meta


# --------------------- Plotting ---------------------------------------------------------------

def _residuals(raw: Optional[np.ndarray], cal: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if raw is None or cal is None:
        return None
    return median_detrend(cal, raw)

def _annot_stats(ax, data: np.ndarray, label: str):
    mu = float(np.nanmean(data))
    sd = float(np.nanstd(data))
    mad = robust_scale(data)
    ax.text(0.98, 0.95,
            f"{label}\nμ={mu:.3g}\nσ={sd:.3g}\nMAD={mad:.3g}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#666666")

def _coverage_from_flags(flags: Optional[np.ndarray]) -> Tuple[int, int]:
    if flags is None:
        return (0,0)
    f = np.asarray(flags)
    # interpret True/1 as "flagged/bad"
    bad = int(np.nansum((f.astype(float) > 0).astype(int)))
    tot = int(f.size)
    return bad, tot

def plot_preview(fgs1: CurvePack, airs: CurvePack, out_path: str, title: str, meta: Dict[str,int]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.2, 0.8])

    # Panels:
    ax_fgs1 = fig.add_subplot(gs[0, 0])
    ax_airs = fig.add_subplot(gs[0, 1])
    ax_res_fgs1 = fig.add_subplot(gs[1, 0])
    ax_res_airs = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_cov  = fig.add_subplot(gs[2, 1])

    # ---- FGS1 time series ----
    ax_fgs1.set_title("FGS1 — Light Curve (raw vs calibrated)", fontsize=12, weight="bold")
    if fgs1.raw is not None:
        ax_fgs1.plot(fgs1.time, fgs1.raw, lw=1.0, alpha=0.85, color="#8a8aff", label="raw")
    if fgs1.cal is not None:
        ax_fgs1.plot(fgs1.time, fgs1.cal, lw=1.2, alpha=0.95, color="#20c15b", label="calibrated")
    ax_fgs1.set_xlabel("Time (native units)")
    ax_fgs1.set_ylabel("Flux (arb.)")
    ax_fgs1.legend(loc="best", fontsize=9)
    ax_fgs1.grid(alpha=0.25)

    # ---- AIRS representative λ ----
    wtxt = f"λ idx={meta.get('airs_wav_idx', 0)}"
    ax_airs.set_title(f"AIRS — Representative channel ({wtxt})", fontsize=12, weight="bold")
    if airs.raw is not None:
        ax_airs.plot(airs.time, airs.raw, lw=1.0, alpha=0.85, color="#ff8c8c", label="raw")
    if airs.cal is not None:
        ax_airs.plot(airs.time, airs.cal, lw=1.2, alpha=0.95, color="#36b3d6", label="calibrated")
    ax_airs.set_xlabel("Time (native units)")
    ax_airs.set_ylabel("Flux (arb.)")
    ax_airs.legend(loc="best", fontsize=9)
    ax_airs.grid(alpha=0.25)

    # ---- Residuals (median-detrended, cal - median(raw)) ----
    r_fgs1 = _residuals(fgs1.raw, fgs1.cal)
    r_airs = _residuals(airs.raw, airs.cal)

    ax_res_fgs1.set_title("FGS1 — Residuals (cal - median(raw))", fontsize=11, weight="bold")
    if r_fgs1 is not None:
        ax_res_fgs1.plot(fgs1.time, r_fgs1, lw=0.9, color="#158f42")
        _annot_stats(ax_res_fgs1, r_fgs1, "FGS1 residuals")
    ax_res_fgs1.axhline(0, color="#888888", lw=0.8, ls="--", alpha=0.7)
    ax_res_fgs1.set_xlabel("Time"); ax_res_fgs1.set_ylabel("ΔFlux")
    ax_res_fgs1.grid(alpha=0.25)

    ax_res_airs.set_title("AIRS — Residuals (cal - median(raw))", fontsize=11, weight="bold")
    if r_airs is not None:
        ax_res_airs.plot(airs.time, r_airs, lw=0.9, color="#1d87a7")
        _annot_stats(ax_res_airs, r_airs, "AIRS residuals")
    ax_res_airs.axhline(0, color="#888888", lw=0.8, ls="--", alpha=0.7)
    ax_res_airs.set_xlabel("Time"); ax_res_airs.set_ylabel("ΔFlux")
    ax_res_airs.grid(alpha=0.25)

    # ---- Residual histograms ----
    ax_hist.set_title("Residual Distributions", fontsize=11, weight="bold")
    bins = 40
    if r_fgs1 is not None and np.isfinite(r_fgs1).any():
        ax_hist.hist(r_fgs1[np.isfinite(r_fgs1)], bins=bins, alpha=0.6, color="#20c15b", label="FGS1")
    if r_airs is not None and np.isfinite(r_airs).any():
        ax_hist.hist(r_airs[np.isfinite(r_airs)], bins=bins, alpha=0.6, color="#36b3d6", label="AIRS")
    ax_hist.set_xlabel("ΔFlux"); ax_hist.set_ylabel("Count")
    ax_hist.legend(loc="best", fontsize=9)
    ax_hist.grid(alpha=0.25)

    # ---- Coverage / flags (optional) ----
    ax_cov.set_title("Coverage / Flags (optional)", fontsize=11, weight="bold")
    f_bad, f_tot = _coverage_from_flags(fgs1.flags)
    a_bad, a_tot = _coverage_from_flags(airs.flags)
    cats = ["FGS1", "AIRS"]
    good = [max(f_tot - f_bad, 0), max(a_tot - a_bad, 0)]
    bad  = [f_bad, a_bad]
    width = 0.5
    if (f_tot + a_tot) > 0:
        ax_cov.bar(cats, good, width=width, color="#4caf50", label="clean")
        ax_cov.bar(cats, bad,  width=width, bottom=good, color="#f44336", label="flagged")
        for i, (g, b, t) in enumerate(zip(good, bad, [f_tot, a_tot])):
            if t > 0:
                pct = 100.0 * g / t
                ax_cov.text(i, g + b * 0.02, f"{pct:.1f}% clean", ha="center", va="bottom", fontsize=9)
        ax_cov.set_ylabel("Samples")
        ax_cov.legend(loc="best", fontsize=9)
        ax_cov.set_ylim(0, max(g + b for g, b in zip(good, bad)) * 1.12)
    else:
        ax_cov.text(0.5, 0.5, "No flag/coverage info available", ha="center", va="center", transform=ax_cov.transAxes, color="#888888")
        ax_cov.set_xticks([]); ax_cov.set_yticks([])

    # ---- Title & footer ----
    if title:
        fig.suptitle(title, fontsize=14, weight="bold", y=1.02)
    footer = ("Median-detrended residuals shown (cal - median(raw)). "
              "AIRS channel chosen by variance unless provided. "
              "Flags treated as bad=1/True if present.")
    fig.text(0.01, 0.01, footer, fontsize=8, color="#777777", va="bottom", ha="left")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------- CLI -------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate calibration_plots/calibration_preview.png (overview of raw vs calibrated, residuals, coverage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cal-h5", type=str, default="outputs/calibrated/lightcurves.h5",
                   help="Path to calibrated HDF5 containing FGS1/AIRS groups.")
    p.add_argument("--raw-h5", type=str, default=None,
                   help="Optional path to raw HDF5 to backfill 'raw' if missing in cal-h5.")
    p.add_argument("--outdir", type=str, default="calibration_plots",
                   help="Output directory for the PNG.")
    p.add_argument("--planet-idx", type=int, default=0,
                   help="Planet index to visualize if arrays are stacked by planet.")
    p.add_argument("--airs-wav-idx", default="auto",
                   help="AIRS wavelength index (int) or 'auto' to pick a representative channel.")
    p.add_argument("--title", type=str, default="Calibration Preview",
                   help="Optional title for the figure.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.outdir)

    try:
        fgs1, airs, meta = discover_curves(
            cal_h5_path=args.cal_h5,
            raw_h5_path=args.raw_h5,
            planet_idx=args.planet_idx,
            airs_wav_idx=args.airs_wav_idx,
        )
    except Exception as e:
        print(f"[error] Failed to read curves: {e}", file=sys.stderr)
        return 2

    out_path = os.path.join(args.outdir, "calibration_preview.png")
    try:
        plot_preview(fgs1=fgs1, airs=airs, out_path=out_path, title=args.title, meta=meta)
        print(f"[ok] Wrote {out_path}")
        return 0
    except Exception as e:
        print(f"[error] Plotting failed: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())