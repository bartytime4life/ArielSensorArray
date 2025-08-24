#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_coverage_heatmap.py

SpectraMind V50 — Calibration Plot Generator (Coverage/Flags Heatmap)

Purpose
-------
Produce `calibration_plots/coverage_heatmap.png` summarizing calibration coverage
using any available flag/mask datasets in the calibrated (and optional raw) HDF5.
Flags are interpreted as 1/True = "bad/flagged" and 0/False = "good/clean".

This figure contains:
  • Top: Time × Wavelength heatmap of AIRS flags (bad=1 shown in color).
  • Right: Per‑wavelength clean fraction bar plot (fraction of good samples).
  • Bottom: Per‑time bad fraction line (fraction flagged at each time index).
  • Optional: FGS1 clean/bad summary (if flags exist) as a compact bar.

Additionally, a CSV summary can be written with per‑wavelength clean_fraction
and per‑time bad_fraction if requested.

Inputs (flexible; robust to missing datasets)
---------------------------------------------
- Calibrated HDF5 (default: outputs/calibrated/lightcurves.h5)
  Expected (but flexible) groups/datasets (auto-discovered):
    • /AIRS/{time, flag|flags|mask|bad|quality}
    • /FGS1/{time, flag|flags|mask|bad|quality}   (optional)
  Shapes supported:
    • AIRS flags: (T, W) preferred; (W,) or (T,) also tolerated with best‑effort handling.
    • FGS1 flags: (T,) or (P,T) → planet selection logic applied.

- Optional raw HDF5 to backfill flags if not present in the cal file.

Outputs
-------
- PNG at: <outdir>/coverage_heatmap.png
- Optional CSV with per‑wavelength & per‑time summaries.

Usage
-----
  python tools/generate_coverage_heatmap.py \
      --cal-h5 outputs/calibrated/lightcurves.h5 \
      --raw-h5 data/raw/lightcurves_raw.h5 \
      --outdir calibration_plots \
      --planet-idx 0 \
      --csv-out calibration_plots/coverage_summary.csv

Author
------
SpectraMind V50 Team — NeurIPS 2025 Ariel Data Challenge
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional, Tuple, Dict

import numpy as np
import h5py
import matplotlib

# Headless backend for CI/Kaggle
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv


# --------------------- Utilities --------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_bool01(arr: np.ndarray) -> np.ndarray:
    """
    Convert any numeric/boolean mask to {0,1} with NaNs preserved.
    Nonzero/True -> 1 (bad), zero/False -> 0 (good).
    """
    a = np.asarray(arr)
    out = np.zeros_like(a, dtype=float)
    finite = np.isfinite(a.astype(float, copy=False)) if np.issubdtype(a.dtype, np.number) else np.ones(a.shape, bool)
    # treat non-finite as NaN
    out[~finite] = np.nan
    # Booleans or anything nonzero
    nz = (a != 0)
    out[nz & finite] = 1.0
    return out


def _clean_fraction(mask01: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute fraction of clean samples along given axis.
    mask01: bad=1, good=0 (NaNs ignored).
    clean_fraction = (# good) / (# finite)
    """
    bad = np.nansum(mask01, axis=axis)
    tot = np.sum(np.isfinite(mask01), axis=axis)
    good = np.maximum(tot - bad, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = good / tot
    return frac


def _bad_fraction(mask01: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute fraction of bad samples along given axis.
    bad_fraction = (# bad) / (# finite)
    """
    bad = np.nansum(mask01, axis=axis)
    tot = np.sum(np.isfinite(mask01), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = bad / tot
    return frac


# --------------------- HDF5 Discovery ---------------------------------------------------------

NAME_CANDIDATES = {
    "time": [r"time", r"t", r"phase"],
    "flag": [r"flag", r"flags", r"mask", r"bad", r"quality"],
}

def _match_name(name: str, patterns) -> bool:
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
    # deep search
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

def _find_first_time(h5: h5py.File, group_path: str) -> Optional[np.ndarray]:
    if group_path not in h5:
        return None
    grp = h5[group_path]
    for k, obj in grp.items():
        if isinstance(obj, h5py.Dataset) and _match_name(k, NAME_CANDIDATES["time"]):
            return obj[()]
    # deep search
    def walker(g: h5py.Group) -> Optional[np.ndarray]:
        for k, v in g.items():
            if isinstance(v, h5py.Dataset) and _match_name(k, NAME_CANDIDATES["time"]):
                return v[()]
            if isinstance(v, h5py.Group):
                out = walker(v)
                if out is not None:
                    return out
        return None
    return walker(grp)

def _select_planet(arr: Optional[np.ndarray], planet_idx: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim >= 2 and a.shape[0] > 1 and a.shape[0] < a.size:
        if not (0 <= planet_idx < a.shape[0]):
            raise IndexError(f"planet_idx {planet_idx} out of range 0..{a.shape[0]-1}")
        return a[planet_idx]
    return a

def discover_flags(cal_h5_path: str,
                   raw_h5_path: Optional[str],
                   planet_idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Discover AIRS and FGS1 flags/time arrays. Returns:
      data = {
        "AIRS_flags": (T,W) mask (bad=1), possibly 2D,
        "AIRS_time":  (T,) time vector (or arange),
        "FGS1_flags": (T,) mask (bad=1) if present,
        "FGS1_time":  (T,) time vector (or arange),
      }
      meta = {
        "AIRS_shape": (T,W or 1),
        "FGS1_shape": (T or 0,),
      }
    """
    if not os.path.isfile(cal_h5_path):
        raise FileNotFoundError(f"Calibrated H5 not found: {cal_h5_path}")
    raw_h5 = None
    if raw_h5_path and os.path.isfile(raw_h5_path):
        raw_h5 = h5py.File(raw_h5_path, "r")

    data, meta = {}, {}

    with h5py.File(cal_h5_path, "r") as cal:
        # AIRS
        a_flags = None
        a_time = None
        for grp in ("AIRS", "airs", "/AIRS", "/airs"):
            gkey = grp.strip("/")
            if gkey in cal:
                gpath = f"/{gkey}"
                a_flags = _find_first_dset(cal, gpath, "flag")
                a_time  = _find_first_time(cal, gpath)
                break
        if a_flags is None and raw_h5 is not None:
            for grp in ("AIRS", "airs", "/AIRS", "/airs"):
                gkey = grp.strip("/")
                if gkey in raw_h5:
                    a_flags = _find_first_dset(raw_h5, f"/{gkey}", "flag")
                    if a_time is None:
                        a_time = _find_first_time(raw_h5, f"/{gkey}")
                    break

        # FGS1
        f_flags = None
        f_time = None
        for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
            gkey = grp.strip("/")
            if gkey in cal:
                gpath = f"/{gkey}"
                f_flags = _find_first_dset(cal, gpath, "flag")
                f_time  = _find_first_time(cal, gpath)
                break
        if f_flags is None and raw_h5 is not None:
            for grp in ("FGS1", "fgs1", "/FGS1", "/fgs1"):
                gkey = grp.strip("/")
                if gkey in raw_h5:
                    f_flags = _find_first_dset(raw_h5, f"/{gkey}", "flag")
                    if f_time is None:
                        f_time = _find_first_time(raw_h5, f"/{gkey}")
                    break

        # shape handling & planet selection
        if f_flags is not None and f_flags.ndim > 1:
            f_flags = _select_planet(f_flags, planet_idx)
        if f_time is not None and f_time.ndim > 1:
            f_time = _select_planet(f_time, planet_idx)
        if a_flags is not None and a_flags.ndim > 2:
            a_flags = _select_planet(a_flags, planet_idx)
        if a_time is not None and a_time.ndim > 1:
            a_time = _select_planet(a_time, planet_idx)

        # fallback times
        if a_time is None and a_flags is not None:
            T = a_flags.shape[0] if a_flags.ndim >= 1 else int(a_flags.size)
            a_time = np.arange(T)
        if f_time is None and f_flags is not None:
            T = f_flags.shape[0] if f_flags.ndim >= 1 else int(f_flags.size)
            f_time = np.arange(T)

        # normalize to 0/1 bad mask
        if a_flags is not None:
            a_flags = as_bool01(a_flags)
            # ensure (T,W) for heatmap; if 1D, expand as (T,1) or (1,W)
            if a_flags.ndim == 1:
                a_flags = a_flags.reshape(-1, 1)
            elif a_flags.ndim == 0:
                a_flags = a_flags.reshape(1, 1)

        if f_flags is not None:
            f_flags = as_bool01(f_flags)
            if f_flags.ndim != 1:
                f_flags = f_flags.reshape(-1)

        # package
        data["AIRS_flags"] = a_flags
        data["AIRS_time"]  = a_time
        data["FGS1_flags"] = f_flags
        data["FGS1_time"]  = f_time

        meta["AIRS_shape"] = tuple(a_flags.shape) if a_flags is not None else (0, 0)
        meta["FGS1_shape"] = tuple(f_flags.shape) if f_flags is not None else (0,)

    if raw_h5 is not None:
        raw_h5.close()

    return data, meta


# --------------------- Plotting ---------------------------------------------------------------

def plot_coverage_heatmap(data: Dict[str, np.ndarray],
                          meta: Dict[str, np.ndarray],
                          out_path: str,
                          title: str) -> None:
    """
    Create a composite figure:
      - AIRS heatmap (T×W) of bad=1 masks
      - Per-wavelength clean fraction (right)
      - Per-time bad fraction (bottom)
      - Optional FGS1 summary (compact bar)
    """
    airs_flags = data.get("AIRS_flags")
    airs_time  = data.get("AIRS_time")
    fgs1_flags = data.get("FGS1_flags")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 0.9, 0.45], width_ratios=[1.0, 0.15, 0.35])

    ax_hm  = fig.add_subplot(gs[0:2, 0])    # heatmap
    ax_cw  = fig.add_subplot(gs[0:2, 1], sharey=ax_hm)  # clean fraction (wavelength) aligned with heatmap y
    ax_bt  = fig.add_subplot(gs[2, 0])      # bad fraction over time
    ax_fgs = fig.add_subplot(gs[2, 2])      # FGS1 compact bar
    ax_legend = fig.add_subplot(gs[0:2, 2]) # text/legend panel

    # ---- AIRS Heatmap ----
    if airs_flags is None:
        ax_hm.text(0.5, 0.5, "No AIRS flags available", ha="center", va="center", transform=ax_hm.transAxes, color="#888888")
        ax_hm.set_axis_off()
    else:
        # Show bad=1 in color, good=0 as white
        im = ax_hm.imshow(airs_flags.T, aspect="auto", origin="lower",
                          interpolation="nearest", cmap="Reds",
                          vmin=0.0, vmax=1.0)
        ax_hm.set_title("AIRS Flags Heatmap (bad=1, good=0)", fontsize=12, weight="bold")
        ax_hm.set_ylabel("Wavelength index")
        ax_hm.set_xlabel("Time index")
        cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("Bad (1) / Good (0)")

        # ---- Per-wavelength clean fraction (right) ----
        clean_w = _clean_fraction(airs_flags, axis=0)  # along time
        yy = np.arange(clean_w.size)
        ax_cw.barh(yy, clean_w, height=0.8, color="#4caf50")
        ax_cw.set_xlim(0, 1)
        ax_cw.set_xlabel("Clean fraction")
        ax_cw.grid(alpha=0.25)
        # hide y tick labels (shared with heatmap)
        plt.setp(ax_cw.get_yticklabels(), visible=False)
        ax_cw.tick_params(axis='y', which='both', length=0)

        # ---- Per-time bad fraction (bottom) ----
        bad_t = _bad_fraction(airs_flags, axis=1)  # along wavelength
        ax_bt.plot(bad_t, color="#f44336", lw=1.2)
        ax_bt.set_ylim(0, 1)
        ax_bt.set_ylabel("Bad fraction")
        ax_bt.set_xlabel("Time index")
        ax_bt.set_title("Per‑Time Bad Fraction", fontsize=11, weight="bold")
        ax_bt.grid(alpha=0.25)

    # ---- FGS1 compact summary ----
    ax_fgs.set_title("FGS1 Coverage (optional)", fontsize=11, weight="bold")
    if fgs1_flags is not None:
        bad = float(np.nansum(fgs1_flags))
        tot = float(np.sum(np.isfinite(fgs1_flags)))
        good = max(tot - bad, 0.0)
        if tot > 0:
            clean_frac = good / tot
            ax_fgs.bar(["FGS1"], [good], color="#4caf50", width=0.5, label="clean")
            ax_fgs.bar(["FGS1"], [bad], bottom=[good], color="#f44336", width=0.5, label="flagged")
            ax_fgs.set_ylim(0, tot * 1.1)
            ax_fgs.text(0, good + bad * 0.02, f"{clean_frac*100:.1f}% clean", ha="center", va="bottom")
            ax_fgs.set_ylabel("Samples")
            ax_fgs.legend(loc="best", fontsize=9)
        else:
            ax_fgs.text(0.5, 0.5, "No finite samples", ha="center", va="center", transform=ax_fgs.transAxes, color="#888888")
            ax_fgs.set_xticks([]); ax_fgs.set_yticks([])
    else:
        ax_fgs.text(0.5, 0.5, "No FGS1 flags available", ha="center", va="center", transform=ax_fgs.transAxes, color="#888888")
        ax_fgs.set_xticks([]); ax_fgs.set_yticks([])

    # ---- Legend / Notes panel ----
    ax_legend.axis("off")
    notes = []
    notes.append("Notes")
    notes.append("• Heatmap shows AIRS bad=1 (flagged) and good=0 (clean).")
    notes.append("• Clean fraction (right) is per‑wavelength: (#good)/(#finite).")
    notes.append("• Bad fraction (bottom) is per‑time: (#bad)/(#finite).")
    if meta.get("AIRS_shape", (0,0)) != (0,0):
        T, W = meta["AIRS_shape"]
        notes.append(f"• AIRS shape: T={T}, W={W}")
    if meta.get("FGS1_shape", (0,)) != (0,):
        notes.append(f"• FGS1 flags: T={meta['FGS1_shape'][0]}")
    ax_legend.text(0.0, 1.0, "\n".join(notes), va="top", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, weight="bold", y=1.02)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------- CSV Export -------------------------------------------------------------

def write_csv_summary(csv_path: str,
                      data: Dict[str, np.ndarray]) -> None:
    """
    Write per-wavelength clean_fraction and per-time bad_fraction to CSV if AIRS flags exist.
    """
    airs_flags = data.get("AIRS_flags")
    if airs_flags is None:
        # still write a stub for consistency
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["message"])
            w.writerow(["No AIRS flags available; summary not generated."])
        return

    clean_w = _clean_fraction(airs_flags, axis=0)  # per wavelength
    bad_t   = _bad_fraction(airs_flags, axis=1)    # per time

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "index", "fraction"])
        for i, v in enumerate(clean_w):
            w.writerow(["per_wavelength_clean_fraction", i, f"{float(v):.6f}"])
        for i, v in enumerate(bad_t):
            w.writerow(["per_time_bad_fraction", i, f"{float(v):.6f}"])


# --------------------- CLI -------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate coverage/flags heatmap for AIRS (and FGS1 summary).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cal-h5", type=str, default="outputs/calibrated/lightcurves.h5",
                   help="Path to calibrated HDF5 containing FGS1/AIRS groups.")
    p.add_argument("--raw-h5", type=str, default=None,
                   help="Optional path to raw HDF5 to backfill flags if missing in cal-h5.")
    p.add_argument("--outdir", type=str, default="calibration_plots",
                   help="Output directory for the PNG/CSV.")
    p.add_argument("--planet-idx", type=int, default=0,
                   help="Planet index to visualize if arrays are stacked by planet.")
    p.add_argument("--title", type=str, default="Calibration Coverage / Flags",
                   help="Figure title.")
    p.add_argument("--csv-out", type=str, default=None,
                   help="Optional CSV summary path to write per‑wavelength/time fractions.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ensure_dir(args.outdir)

    try:
        data, meta = discover_flags(
            cal_h5_path=args.cal_h5,
            raw_h5_path=args.raw_h5,
            planet_idx=args.planet_idx,
        )
    except Exception as e:
        print(f"[error] Failed to read flags: {e}", file=sys.stderr)
        return 2

    out_png = os.path.join(args.outdir, "coverage_heatmap.png")
    try:
        plot_coverage_heatmap(data=data, meta=meta, out_path=out_png, title=args.title)
        print(f"[ok] Wrote {out_png}")
    except Exception as e:
        print(f"[error] Plotting failed: {e}", file=sys.stderr)
        return 3

    if args.csv_out:
        try:
            write_csv_summary(args.csv_out, data=data)
            print(f"[ok] Wrote {args.csv_out}")
        except Exception as e:
            print(f"[warn] Failed to write CSV summary: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())