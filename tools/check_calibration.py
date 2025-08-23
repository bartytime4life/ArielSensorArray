#!/usr/bin/env python3
"""
check_calibration.py — SpectraMind V50
Cal/QA: fast, reproducible checks for bias/dark/flat/wavelength calibration packs.

Features
- CLI-first (Typer) with rich logs & progress
- Hydra config support (paths, thresholds, report dir)
- Works with FITS (*.fits, *.fit) and NumPy stacks (*.npy)
- Metrics:
  * Bias level (e-), spatial structure (RSD %)
  * Dark current rate (e-/px/s), hot-pixel fraction
  * Flat-field non‑uniformity (RSD %), gradient (%/kpx)
  * Wavelength solution drift (px and nm) via cross‑correlation / line centroid
- Artifacts:
  * Markdown summary report
  * JSON metrics dump
  * JSONL events log
  * PNG quick‑look plots (optional)
"""

from __future__ import annotations

import json
import os
import sys
import time
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Optional deps (auto-detect)
try:
    from astropy.io import fits  # type: ignore
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# CLI / Config / Logs
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Hydra is optional; if not present you can pass arguments via CLI flags only.
try:
    import hydra  # type: ignore
    from omegaconf import DictConfig, OmegaConf  # type: ignore
    HAS_HYDRA = True
except Exception:
    HAS_HYDRA = False


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console(record=False)


# ------------------------------ Utilities ------------------------------ #

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, out: Path):
    out.write_text(json.dumps(data, indent=2))


def append_jsonl(evt: dict, out: Path):
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")


def robust_stats(x: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean, std, rsd_percent) with robust std via MAD."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.nanmean(x))
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))  # ~1.4826 * sigma for normal
    std = float(1.4826 * mad) if mad > 0 else float(np.nanstd(x))
    rsd = (std / mean) * 100.0 if (std == std and mean != 0.0) else np.nan  # rsd%
    return mean, std, rsd


def finite_diff_gradient(img: np.ndarray) -> Tuple[float, float]:
    """Estimate gradient magnitude along x and y as % per kilo-pixel."""
    h, w = img.shape
    # normalize by median to express gradient in %
    med = np.nanmedian(img)
    if med == 0 or not np.isfinite(med):
        return np.nan, np.nan
    # gradient (simple central diff)
    gx = np.nanmedian(np.abs(np.diff(img, axis=1))) / med * 100.0  # % per pixel
    gy = np.nanmedian(np.abs(np.diff(img, axis=0))) / med * 100.0
    # scale to % per kilo-pixel
    return gx * 1000.0, gy * 1000.0


def cross_correlation_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Return shift (in pixels) that best aligns b to a (positive = b -> right)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-12)
    n = len(a) + len(b) - 1
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    cc = np.fft.irfft(fa * np.conj(fb), n)  # cross-correlation
    shift = int(np.argmax(np.abs(cc))) - (len(b) - 1)
    # subpixel refinement via quadratic fit around peak if possible
    idx = int(np.argmax(np.abs(cc)))
    if 1 <= idx < n - 1:
        y0, y1, y2 = cc[idx - 1], cc[idx], cc[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            return float(shift + delta)
    return float(shift)


# ------------------------------ Loading ------------------------------ #

def load_stack(path: Union[str, Path]) -> np.ndarray:
    """
    Load calibration stack:
      - .npy  -> (N,H,W) or (H,W)
      - .fits/.fit -> single HDU or multiple extensions -> stacked into (N,H,W)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array in npy, got shape {arr.shape}")
        return np.asarray(arr, dtype=np.float64)

    if path.suffix.lower() in {".fits", ".fit"}:
        if not HAS_ASTROPY:
            raise RuntimeError("astropy is required to read FITS files. `pip install astropy`")
        with fits.open(path) as hdul:
            # Gather all IMAGE HDUs
            imgs = []
            for hdu in hdul:
                if getattr(hdu, "data", None) is not None and isinstance(hdu.data, np.ndarray):
                    data = np.asarray(hdu.data, dtype=np.float64)
                    if data.ndim == 2:
                        imgs.append(data)
                    elif data.ndim == 3:
                        imgs.extend([frame for frame in data])
            if not imgs:
                raise ValueError(f"No image data found in {path}")
            stack = np.stack(imgs, axis=0)
            return stack

    raise ValueError(f"Unsupported file format: {path.suffix}")


# ------------------------------ Metrics ------------------------------ #

@dataclass
class BiasMetrics:
    mean_e: float
    std_e: float
    rsd_pct: float
    frame_rsd_pct: float


def measure_bias(stack: np.ndarray) -> BiasMetrics:
    # Global stats
    mean, std, rsd = robust_stats(stack)
    # Per-frame RS D to detect temporal structure
    per = [robust_stats(f)[2] for f in stack]
    frame_rsd = float(np.nanmedian(per))
    return BiasMetrics(mean_e=mean, std_e=std, rsd_pct=rsd, frame_rsd_pct=frame_rsd)


@dataclass
class DarkMetrics:
    rate_e_per_px_s: float
    hot_px_fraction: float


def measure_dark(dark_stack: np.ndarray, exptime_s: float, hot_sigma: float = 5.0) -> DarkMetrics:
    # Median dark image -> electrons (assuming input already in e-; if ADU, provide gain in config)
    dark_med = np.nanmedian(dark_stack, axis=0)
    rate = float(np.nanmedian(dark_med) / max(exptime_s, 1e-9))  # e-/px/s
    # Hot pixels: per‑pixel median relative to spatial distribution
    med = float(np.nanmedian(dark_med))
    std = float(np.nanstd(dark_med))
    thr = med + hot_sigma * std
    hot_frac = float(np.mean(dark_med > thr)) if np.isfinite(thr) else np.nan
    return DarkMetrics(rate_e_per_px_s=rate, hot_px_fraction=hot_frac)


@dataclass
class FlatMetrics:
    rsd_pct: float
    grad_x_pct_per_kpx: float
    grad_y_pct_per_kpx: float


def measure_flat(flat_stack: np.ndarray) -> FlatMetrics:
    flat_norm = np.nanmedian(flat_stack, axis=0)
    mean, std, rsd = robust_stats(flat_norm)
    gx, gy = finite_diff_gradient(flat_norm)
    return FlatMetrics(rsd_pct=rsd, grad_x_pct_per_kpx=gx, grad_y_pct_per_kpx=gy)


@dataclass
class WaveMetrics:
    px_shift: float
    nm_shift: Optional[float]


def measure_wavelength_drift(
    stack: np.ndarray,
    axis_spectral: int,
    ref_spectrum: Optional[np.ndarray] = None,
    disp_nm_per_px: Optional[float] = None,
) -> WaveMetrics:
    """
    Reduce stack to a 1D spectrum (median-collapsed across spatial axis),
    then measure shift vs reference (or stack self‑reference if None).
    """
    arr = np.nanmedian(stack, axis=0)  # (H,W)
    if axis_spectral not in (0, 1):
        raise ValueError("axis_spectral must be 0 (rows) or 1 (cols)")
    spec = np.nanmedian(arr, axis=1-axis_spectral)

    if ref_spectrum is None:
        ref_spectrum = spec.copy()

    shift_px = cross_correlation_shift(ref_spectrum, spec)
    nm_shift = float(shift_px * disp_nm_per_px) if disp_nm_per_px is not None else None
    return WaveMetrics(px_shift=float(shift_px), nm_shift=nm_shift)


# ------------------------------ QA / Thresholds ------------------------------ #

@dataclass
class Thresholds:
    bias_rsd_max_pct: float = 1.0
    bias_frame_rsd_max_pct: float = 1.5
    dark_rate_max_e_px_s: float = 0.02
    dark_hot_px_max_frac: float = 0.01
    flat_rsd_max_pct: float = 2.0
    flat_grad_max_pct_per_kpx: float = 1.0
    wave_max_px_drift: float = 0.5
    wave_max_nm_drift: Optional[float] = None  # if disp provided


def pass_fail(val: Optional[float], limit: Optional[float], sense: str = "max") -> Optional[bool]:
    if val is None or limit is None:
        return None
    if sense == "max":
        return bool(val <= limit)
    if sense == "min":
        return bool(val >= limit)
    return None


# ------------------------------ Reports ------------------------------ #

def write_markdown_report(
    out_md: Path,
    meta: dict,
    bias: BiasMetrics,
    dark: Optional[DarkMetrics],
    flat: Optional[FlatMetrics],
    wave: Optional[WaveMetrics],
    thr: Thresholds,
    figures: List[Path],
):
    lines = []
    lines.append(f"# Calibration Check Report\n")
    lines.append(f"- Generated: `{_now_iso()}`")
    lines.append(f"- Source: `{meta.get('source')}`")
    lines.append(f"- Report dir: `{out_md.parent}`\n")
    lines.append("## Summary\n")
    tbl = [
        ("Metric", "Value", "Limit", "Pass?"),
        ("Bias RSD (%)", f"{bias.rsd_pct:.3f}", f"≤ {thr.bias_rsd_max_pct}", str(pass_fail(bias.rsd_pct, thr.bias_rsd_max_pct))),
        ("Bias Frame RSD (%)", f"{bias.frame_rsd_pct:.3f}", f"≤ {thr.bias_frame_rsd_max_pct}", str(pass_fail(bias.frame_rsd_pct, thr.bias_frame_rsd_max_pct))),
    ]
    if dark is not None:
        tbl.extend([
            ("Dark rate (e-/px/s)", f"{dark.rate_e_per_px_s:.5f}", f"≤ {thr.dark_rate_max_e_px_s}", str(pass_fail(dark.rate_e_per_px_s, thr.dark_rate_max_e_px_s))),
            ("Hot pixel fraction", f"{dark.hot_px_fraction:.4f}", f"≤ {thr.dark_hot_px_max_frac}", str(pass_fail(dark.hot_px_fraction, thr.dark_hot_px_max_frac))),
        ])
    if flat is not None:
        tbl.extend([
            ("Flat RSD (%)", f"{flat.rsd_pct:.3f}", f"≤ {thr.flat_rsd_max_pct}", str(pass_fail(flat.rsd_pct, thr.flat_rsd_max_pct))),
            ("Flat ∇x (%/kpx)", f"{flat.grad_x_pct_per_kpx:.3f}", f"≤ {thr.flat_grad_max_pct_per_kpx}", str(pass_fail(flat.grad_x_pct_per_kpx, thr.flat_grad_max_pct_per_kpx))),
            ("Flat ∇y (%/kpx)", f"{flat.grad_y_pct_per_kpx:.3f}", f"≤ {thr.flat_grad_max_pct_per_kpx}", str(pass_fail(flat.grad_y_pct_per_kpx, thr.flat_grad_max_pct_per_kpx))),
        ])
    if wave is not None:
        nm_limit = thr.wave_max_nm_drift if thr.wave_max_nm_drift is not None else "—"
        nm_val = f"{wave.nm_shift:.4f}" if wave.nm_shift is not None else "—"
        tbl.extend([
            ("Wavelength drift (px)", f"{wave.px_shift:.4f}", f"≤ {thr.wave_max_px_drift}", str(pass_fail(wave.px_shift, thr.wave_max_px_drift))),
            ("Wavelength drift (nm)", nm_val, f"≤ {nm_limit}", str(pass_fail(wave.nm_shift, thr.wave_max_nm_drift)) if wave.nm_shift is not None and thr.wave_max_nm_drift is not None else "—"),
        ])

    # render table
    lines.append("| " + " | ".join(tbl[0]) + " |")
    lines.append("|" + "|".join(["---"] * len(tbl[0])) + "|")
    for row in tbl[1:]:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if figures:
        lines.append("## Figures\n")
        for fig in figures:
            lines.append(f"![{fig.name}]({fig.name})")
        lines.append("")

    lines.append("## Metadata\n")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2))
    lines.append("```")
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------ Plotting ------------------------------ #

def make_quicklook_figs(
    out_dir: Path,
    bias_stack: Optional[np.ndarray],
    dark_stack: Optional[np.ndarray],
    flat_stack: Optional[np.ndarray],
    axis_spectral: int,
    disp_nm_per_px: Optional[float] = None,
) -> List[Path]:
    figs: List[Path] = []
    if not HAS_MPL:
        return figs

    if bias_stack is not None and bias_stack.size:
        plt.figure(figsize=(7, 4))
        plt.title("Bias median frame")
        plt.imshow(np.nanmedian(bias_stack, axis=0), cmap="magma")
        plt.colorbar(label="e-")
        p = out_dir / "bias_median.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        figs.append(p)

    if dark_stack is not None and dark_stack.size:
        plt.figure(figsize=(7, 4))
        plt.title("Dark median frame")
        plt.imshow(np.nanmedian(dark_stack, axis=0), cmap="magma")
        plt.colorbar(label="e-")
        p = out_dir / "dark_median.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        figs.append(p)

    if flat_stack is not None and flat_stack.size:
        plt.figure(figsize=(7, 4))
        plt.title("Flat median frame (normalized)")
        fmed = np.nanmedian(flat_stack, axis=0)
        plt.imshow(fmed / (np.nanmedian(fmed) + 1e-9), cmap="viridis", vmin=0.97, vmax=1.03)
        plt.colorbar(label="relative")
        p = out_dir / "flat_median.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        figs.append(p)

    # Spectrum quick‑look (if any stack exists)
    stack = None
    if flat_stack is not None and flat_stack.size:
        stack = flat_stack
    elif dark_stack is not None and dark_stack.size:
        stack = dark_stack
    elif bias_stack is not None and bias_stack.size:
        stack = bias_stack

    if stack is not None:
        arr = np.nanmedian(stack, axis=0)
        spec = np.nanmedian(arr, axis=1-axis_spectral)
        plt.figure(figsize=(7, 3.5))
        x = np.arange(len(spec))
        xlabel = "Pixel"
        if disp_nm_per_px is not None:
            # plot in nm relative scale as secondary axis
            ax = plt.gca()
            ax2 = ax.secondary_xaxis('top', functions=(lambda p: p * disp_nm_per_px,
                                                       lambda nm: nm / disp_nm_per_px))
            ax2.set_xlabel("Δλ (nm)")
        plt.plot(x, spec, lw=1.2)
        plt.xlabel(xlabel); plt.ylabel("e- (arb.)"); plt.title("Median spectrum")
        p = out_dir / "spectrum.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        figs.append(p)

    return figs


# ------------------------------ Config Model ------------------------------ #

@dataclass
class InputPaths:
    bias: Optional[str] = None
    dark: Optional[str] = None
    flat: Optional[str] = None


@dataclass
class WaveConf:
    axis_spectral: int = 1  # 1 = columns (x) vary with wavelength; 0 = rows
    dispersion_nm_per_px: Optional[float] = None
    ref_spectrum_npy: Optional[str] = None  # optional reference spectrum (.npy 1D)


@dataclass
class ReportConf:
    out_dir: str = "reports/cal_check"
    tag: Optional[str] = None
    save_plots: bool = True


@dataclass
class CalCheckConfig:
    inputs: InputPaths = InputPaths()
    wave: WaveConf = WaveConf()
    report: ReportConf = ReportConf()
    thresholds: Thresholds = Thresholds()
    # Optional: if frames are in ADU, provide e-/ADU gain for each type; default assumes already electrons
    gain_e_per_adu_bias: Optional[float] = None
    gain_e_per_adu_dark: Optional[float] = None
    gain_e_per_adu_flat: Optional[float] = None
    # Dark exposure time (seconds) for rate calc
    dark_exptime_s: Optional[float] = None
    # Hot pixel sigma threshold
    dark_hot_sigma: float = 5.0


# ------------------------------ Runner ------------------------------ #

def run_check(cfg: CalCheckConfig) -> int:
    out_dir = ensure_dir(Path(cfg.report.out_dir) / (cfg.report.tag or _now_iso().replace(":", "")))
    metrics_json = out_dir / "metrics.json"
    events_jsonl = out_dir / "events.jsonl"
    report_md = out_dir / "report.md"

    append_jsonl({"ts": _now_iso(), "event": "start", "cfg": _cfg_to_plain(cfg)}, events_jsonl)

    # Load stacks (with progress)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as prog:
        bias_stack = dark_stack = flat_stack = None

        if cfg.inputs.bias:
            t = prog.add_task(f"Loading bias: {cfg.inputs.bias}", total=None)
            bias_stack = load_stack(cfg.inputs.bias)
            if cfg.gain_e_per_adu_bias:
                bias_stack = bias_stack * cfg.gain_e_per_adu_bias
            prog.remove_task(t)

        if cfg.inputs.dark:
            t = prog.add_task(f"Loading dark: {cfg.inputs.dark}", total=None)
            dark_stack = load_stack(cfg.inputs.dark)
            if cfg.gain_e_per_adu_dark:
                dark_stack = dark_stack * cfg.gain_e_per_adu_dark
            prog.remove_task(t)

        if cfg.inputs.flat:
            t = prog.add_task(f"Loading flat: {cfg.inputs.flat}", total=None)
            flat_stack = load_stack(cfg.inputs.flat)
            if cfg.gain_e_per_adu_flat:
                flat_stack = flat_stack * cfg.gain_e_per_adu_flat
            prog.remove_task(t)

    # Compute metrics
    figures: List[Path] = []
    bias_metrics = dark_metrics = flat_metrics = wave_metrics = None

    if bias_stack is not None:
        bias_metrics = measure_bias(bias_stack)
        append_jsonl({"ts": _now_iso(), "event": "bias_metrics", **bias_metrics.__dict__}, events_jsonl)

    if dark_stack is not None:
        if cfg.dark_exptime_s is None:
            console.print("[yellow]dark_exptime_s not provided; dark rate cannot be normalized. Using exptime=1s.")
        exptime = cfg.dark_exptime_s or 1.0
        dark_metrics = measure_dark(dark_stack, exptime_s=exptime, hot_sigma=cfg.dark_hot_sigma)
        append_jsonl({"ts": _now_iso(), "event": "dark_metrics", **dark_metrics.__dict__}, events_jsonl)

    if flat_stack is not None:
        flat_metrics = measure_flat(flat_stack)
        append_jsonl({"ts": _now_iso(), "event": "flat_metrics", **flat_metrics.__dict__}, events_jsonl)

    # Wavelength drift
    if any(x is not None for x in (flat_stack, dark_stack, bias_stack)):
        stack_for_wave = flat_stack if flat_stack is not None else (dark_stack if dark_stack is not None else bias_stack)
        ref_spectrum = None
        if cfg.wave.ref_spectrum_npy:
            p = Path(cfg.wave.ref_spectrum_npy)
            if p.exists():
                ref_spectrum = np.load(p, allow_pickle=False)
        wave_metrics = measure_wavelength_drift(
            stack_for_wave,
            axis_spectral=cfg.wave.axis_spectral,
            ref_spectrum=ref_spectrum,
            disp_nm_per_px=cfg.wave.dispersion_nm_per_px,
        )
        append_jsonl({"ts": _now_iso(), "event": "wave_metrics", **wave_metrics.__dict__}, events_jsonl)

    # Plots
    if cfg.report.save_plots:
        figures = make_quicklook_figs(
            out_dir=out_dir,
            bias_stack=bias_stack,
            dark_stack=dark_stack,
            flat_stack=flat_stack,
            axis_spectral=cfg.wave.axis_spectral,
            disp_nm_per_px=cfg.wave.dispersion_nm_per_px,
        )

    # Aggregate & write
    metrics = {
        "generated": _now_iso(),
        "source": {
            "bias": cfg.inputs.bias,
            "dark": cfg.inputs.dark,
            "flat": cfg.inputs.flat,
            "tag": cfg.report.tag,
        },
        "bias": (bias_metrics.__dict__ if bias_metrics else None),
        "dark": (dark_metrics.__dict__ if dark_metrics else None),
        "flat": (flat_metrics.__dict__ if flat_metrics else None),
        "wavelength": (wave_metrics.__dict__ if wave_metrics else None),
        "thresholds": cfg.thresholds.__dict__,
    }
    save_json(metrics, metrics_json)

    # Render console table
    table = Table(title="Calibration QA Summary", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Limit")
    table.add_column("Pass?")
    def _add_row(name, val, lim, ok):
        table.add_row(name, val, lim, ok if ok is not None else "—")

    if bias_metrics:
        _add_row("Bias RSD (%)", f"{bias_metrics.rsd_pct:.3f}",
                 f"≤ {cfg.thresholds.bias_rsd_max_pct}",
                 str(pass_fail(bias_metrics.rsd_pct, cfg.thresholds.bias_rsd_max_pct)))
        _add_row("Bias Frame RSD (%)", f"{bias_metrics.frame_rsd_pct:.3f}",
                 f"≤ {cfg.thresholds.bias_frame_rsd_max_pct}",
                 str(pass_fail(bias_metrics.frame_rsd_pct, cfg.thresholds.bias_frame_rsd_max_pct)))

    if dark_metrics:
        _add_row("Dark rate (e-/px/s)", f"{dark_metrics.rate_e_per_px_s:.5f}",
                 f"≤ {cfg.thresholds.dark_rate_max_e_px_s}",
                 str(pass_fail(dark_metrics.rate_e_per_px_s, cfg.thresholds.dark_rate_max_e_px_s)))
        _add_row("Hot px fraction", f"{dark_metrics.hot_px_fraction:.4f}",
                 f"≤ {cfg.thresholds.dark_hot_px_max_frac}",
                 str(pass_fail(dark_metrics.hot_px_fraction, cfg.thresholds.dark_hot_px_max_frac)))

    if flat_metrics:
        _add_row("Flat RSD (%)", f"{flat_metrics.rsd_pct:.3f}",
                 f"≤ {cfg.thresholds.flat_rsd_max_pct}",
                 str(pass_fail(flat_metrics.rsd_pct, cfg.thresholds.flat_rsd_max_pct)))
        _add_row("Flat ∇x (%/kpx)", f"{flat_metrics.grad_x_pct_per_kpx:.3f}",
                 f"≤ {cfg.thresholds.flat_grad_max_pct_per_kpx}",
                 str(pass_fail(flat_metrics.grad_x_pct_per_kpx, cfg.thresholds.flat_grad_max_pct_per_kpx)))
        _add_row("Flat ∇y (%/kpx)", f"{flat_metrics.grad_y_pct_per_kpx:.3f}",
                 f"≤ {cfg.thresholds.flat_grad_max_pct_per_kpx}",
                 str(pass_fail(flat_metrics.grad_y_pct_per_kpx, cfg.thresholds.flat_grad_max_pct_per_kpx)))

    if wave_metrics:
        nm_limit = cfg.thresholds.wave_max_nm_drift if cfg.thresholds.wave_max_nm_drift is not None else "—"
        nm_val = f"{wave_metrics.nm_shift:.4f}" if wave_metrics.nm_shift is not None else "—"
        _add_row("Wave drift (px)", f"{wave_metrics.px_shift:.4f}",
                 f"≤ {cfg.thresholds.wave_max_px_drift}",
                 str(pass_fail(wave_metrics.px_shift, cfg.thresholds.wave_max_px_drift)))
        _add_row("Wave drift (nm)", nm_val,
                 f"≤ {nm_limit}",
                 str(pass_fail(wave_metrics.nm_shift, cfg.thresholds.wave_max_nm_drift)) if wave_metrics.nm_shift is not None and cfg.thresholds.wave_max_nm_drift is not None else "—")

    console.print(table)

    meta = {
        "source": str(Path(cfg.inputs.flat or cfg.inputs.dark or cfg.inputs.bias or 'N/A')),
        "cfg": _cfg_to_plain(cfg),
        "out_dir": str(out_dir),
    }

    write_markdown_report(
        out_md=report_md,
        meta=meta,
        bias=bias_metrics if bias_metrics else BiasMetrics(np.nan, np.nan, np.nan, np.nan),
        dark=dark_metrics,
        flat=flat_metrics,
        wave=wave_metrics,
        thr=cfg.thresholds,
        figures=[p.relative_to(out_dir) for p in figures],
    )

    append_jsonl({"ts": _now_iso(), "event": "done", "report": str(report_md)}, events_jsonl)
    console.print(f"[green]Report saved:[/green] {report_md}")
    console.print(f"[green]Metrics saved:[/green] {metrics_json}")
    return 0


def _cfg_to_plain(cfg: CalCheckConfig) -> dict:
    # Convert dataclasses to vanilla dict for logging
    return {
        "inputs": cfg.inputs.__dict__,
        "wave": cfg.wave.__dict__,
        "report": cfg.report.__dict__,
        "thresholds": cfg.thresholds.__dict__,
        "gain_e_per_adu_bias": cfg.gain_e_per_adu_bias,
        "gain_e_per_adu_dark": cfg.gain_e_per_adu_dark,
        "gain_e_per_adu_flat": cfg.gain_e_per_adu_flat,
        "dark_exptime_s": cfg.dark_exptime_s,
        "dark_hot_sigma": cfg.dark_hot_sigma,
    }


# ------------------------------ CLI ------------------------------ #

@app.command("run")
def cli_run(
    bias: Optional[Path] = typer.Option(None, help="Bias stack path (.fits/.fit/.npy)"),
    dark: Optional[Path] = typer.Option(None, help="Dark stack path (.fits/.fit/.npy)"),
    flat: Optional[Path] = typer.Option(None, help="Flat stack path (.fits/.fit/.npy)"),

    out_dir: Path = typer.Option("reports/cal_check", help="Output directory root"),
    tag: Optional[str] = typer.Option(None, help="Optional run tag (used as subfolder)"),
    save_plots: bool = typer.Option(True, help="Save PNG quick‑look plots"),

    axis_spectral: int = typer.Option(1, help="1 = wavelength along X/columns, 0 = along Y/rows"),
    dispersion_nm_per_px: Optional[float] = typer.Option(None, help="Dispersion scale (nm/px) for nm drift"),
    ref_spectrum_npy: Optional[Path] = typer.Option(None, help="Optional reference spectrum (.npy 1D)"),

    bias_rsd_max_pct: float = typer.Option(1.0, help="Max allowed bias RSD (%)"),
    bias_frame_rsd_max_pct: float = typer.Option(1.5, help="Max allowed median per-frame bias RSD (%)"),
    dark_rate_max_e_px_s: float = typer.Option(0.02, help="Max allowed dark current rate (e-/px/s)"),
    dark_hot_px_max_frac: float = typer.Option(0.01, help="Max allowed hot pixel fraction"),
    flat_rsd_max_pct: float = typer.Option(2.0, help="Max allowed flat non-uniformity RSD (%)"),
    flat_grad_max_pct_per_kpx: float = typer.Option(1.0, help="Max allowed flat gradient magnitude (%/kpx)"),
    wave_max_px_drift: float = typer.Option(0.5, help="Max allowed wavelength drift (px)"),
    wave_max_nm_drift: Optional[float] = typer.Option(None, help="Max allowed wavelength drift (nm)"),

    gain_e_per_adu_bias: Optional[float] = typer.Option(None, help="Gain e-/ADU for bias frames"),
    gain_e_per_adu_dark: Optional[float] = typer.Option(None, help="Gain e-/ADU for dark frames"),
    gain_e_per_adu_flat: Optional[float] = typer.Option(None, help="Gain e-/ADU for flat frames"),
    dark_exptime_s: Optional[float] = typer.Option(None, help="Dark exposure time (seconds)"),
    dark_hot_sigma: float = typer.Option(5.0, help="Sigma threshold for hot pixels in dark"),
):
    """
    Run calibration QA with CLI flags (no Hydra). Example:

    python check_calibration.py run --bias bias.npy --dark dark.npy --flat flat.npy --dark-exptime-s 60
    """
    cfg = CalCheckConfig(
        inputs=InputPaths(
            bias=str(bias) if bias else None,
            dark=str(dark) if dark else None,
            flat=str(flat) if flat else None,
        ),
        wave=WaveConf(
            axis_spectral=axis_spectral,
            dispersion_nm_per_px=dispersion_nm_per_px,
            ref_spectrum_npy=str(ref_spectrum_npy) if ref_spectrum_npy else None,
        ),
        report=ReportConf(
            out_dir=str(out_dir),
            tag=tag,
            save_plots=save_plots,
        ),
        thresholds=Thresholds(
            bias_rsd_max_pct=bias_rsd_max_pct,
            bias_frame_rsd_max_pct=bias_frame_rsd_max_pct,
            dark_rate_max_e_px_s=dark_rate_max_e_px_s,
            dark_hot_px_max_frac=dark_hot_px_max_frac,
            flat_rsd_max_pct=flat_rsd_max_pct,
            flat_grad_max_pct_per_kpx=flat_grad_max_pct_per_kpx,
            wave_max_px_drift=wave_max_px_drift,
            wave_max_nm_drift=wave_max_nm_drift,
        ),
        gain_e_per_adu_bias=gain_e_per_adu_bias,
        gain_e_per_adu_dark=gain_e_per_adu_dark,
        gain_e_per_adu_flat=gain_e_per_adu_flat,
        dark_exptime_s=dark_exptime_s,
        dark_hot_sigma=dark_hot_sigma,
    )
    try:
        raise SystemExit(run_check(cfg))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if console.is_terminal:
            traceback.print_exc()
        raise SystemExit(1)


# ------------------------------ Hydra entry (optional) ------------------------------ #

if HAS_HYDRA:
    @hydra.main(version_base=None, config_path=None, config_name=None)
    def hydra_entry(_cfg: DictConfig) -> None:
        """
        Hydra mode:
        Create a YAML matching CalCheckConfig structure and pass via:
          python check_calibration.py hydra_entry \
            inputs.bias=bias.npy inputs.dark=dark.npy inputs.flat=flat.npy \
            dark_exptime_s=60 report.out_dir=reports/cal_check
        """
        # Convert a flexible DictConfig into our dataclasses
        def dc_from(dc_cls, section):
            if section is None:
                return dc_cls()
            # map nested
            kwargs = {}
            for k, v in dict(section).items():
                if isinstance(v, dict):
                    # nested dataclass field?
                    # We'll construct manually where needed
                    kwargs[k] = v
                else:
                    kwargs[k] = v
            return dc_cls(**kwargs)  # type: ignore

        cfg = CalCheckConfig(
            inputs=dc_from(InputPaths, _cfg.get("inputs")),
            wave=dc_from(WaveConf, _cfg.get("wave")),
            report=dc_from(ReportConf, _cfg.get("report")),
            thresholds=dc_from(Thresholds, _cfg.get("thresholds")),
            gain_e_per_adu_bias=_cfg.get("gain_e_per_adu_bias"),
            gain_e_per_adu_dark=_cfg.get("gain_e_per_adu_dark"),
            gain_e_per_adu_flat=_cfg.get("gain_e_per_adu_flat"),
            dark_exptime_s=_cfg.get("dark_exptime_s"),
            dark_hot_sigma=_cfg.get("dark_hot_sigma", 5.0),
        )
        raise SystemExit(run_check(cfg))


# ------------------------------ Main ------------------------------ #

if __name__ == "__main__":
    # If you prefer Hydra, run:  python check_calibration.py hydra_entry …
    # Otherwise, use the Typer CLI:
    app()
