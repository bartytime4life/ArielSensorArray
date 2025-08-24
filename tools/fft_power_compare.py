#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/fft_power_compare.py

SpectraMind V50 — FFT Power Compare (Ultimate, Challenge‑Grade)

Purpose
-------
Compute and compare FFT power spectra between two conditions (e.g., pre‑ vs post‑calibration;
method A vs B) across Ariel‑like data. Works on either:

  • Time-domain cubes (FGS1/AIRS‑style):      X.shape ∈ {(T,W), (P,T,W)}
      -> FFT along time axis (T) for each [planet, wavelength] channel
  • Spectral-domain matrices (μ spectra):     μ.shape ∈ {(B,), (P,B)}
      -> FFT along wavelength/bins axis (B) per planet

It produces:
  • Aggregated power spectra for condition A and B (mean/median over selected channels)
  • Power ratio (B / A) diagnostics
  • CSV exports for reproducibility
  • PNG static plots + optional Plotly HTML
  • Self-contained dashboard and manifest + run-hash log
  • Append-only audit logs (logs/v50_debug_log.md / logs/v50_runs.jsonl)

Examples
--------
# 1) Time-domain comparison (pre vs post calibration), averaging over all wavelengths
python tools/fft_power_compare.py \
  --a outputs/calibration/pre_cube.npy \
  --b outputs/calibration/post_cube.npy \
  --mode time --aggregate mean --nfft 4096 --freq-max 0.5 \
  --outdir outputs/fft_compare_pre_post --open-browser

# 2) Spectral-domain comparison for μ spectra (bins axis FFT)
python tools/fft_power_compare.py \
  --a outputs/predictions/mu_a.npy \
  --b outputs/predictions/mu_b.npy \
  --mode spectrum --aggregate median \
  --outdir outputs/fft_compare_mu --open-browser

Input Notes
-----------
• Arrays are loaded via NPY/NPZ/CSV/TSV/Parquet/Feather.
• Time-domain expects shape (T,W) or (P,T,W). If (T,W), planets=1. If (P,T,W), FFT runs along axis=1.
• Spectral-domain expects shape (B,) or (P,B). FFT runs along last axis.
• Optional wavelength/time axes can be supplied to label plots; otherwise normalized axes are used.

Outputs
-------
outdir/
  power_a.csv, power_b.csv, power_ratio.csv    # frequency vs power (aggregated)
  power_a.png/.html, power_b.png/.html         # spectra
  power_ratio.png/.html                        # ratio B/A
  fft_power_compare_manifest.json
  run_hash_summary_v50.json
  dashboard.html

Design
------
• Deterministic computations (no RNG). No external web calls.
• Graceful fallbacks if Plotly/Matplotlib missing (CSV always written).
• Robust input normalization & channel selection (wavelength masks / planet filters).

"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Tabular I/O
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required. Please `pip install pandas`.") from e

# Optional viz libs
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False


# ==============================================================================
# Utilities — time, dirs, hashing, audit logging
# ==============================================================================

def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_jsonable(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


@dataclass
class AuditLogger:
    md_path: Path
    jsonl_path: Path

    def log(self, event: Dict[str, Any]) -> None:
        _ensure_dir(self.md_path.parent)
        _ensure_dir(self.jsonl_path.parent)
        row = dict(event)
        row.setdefault("timestamp", _now_iso())
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        md = textwrap.dedent(f"""
        ---
        time: {row["timestamp"]}
        tool: fft_power_compare
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        a: {row.get("a","")}
        b: {row.get("b","")}
        mode: {row.get("mode","time")}
        aggregate: {row.get("aggregate","mean")}
        nfft: {row.get("nfft","")}
        freq_max: {row.get("freq_max","")}
        outdir: {row.get("outdir","")}
        message: {row.get("message","")}
        """).strip() + "\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(md)


def _update_run_hash_summary(outdir: Path, manifest: Dict[str, Any]) -> None:
    p = outdir / "run_hash_summary_v50.json"
    payload = {"runs": []}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "runs" not in payload:
                payload = {"runs": []}
        except Exception:
            payload = {"runs": []}
    payload["runs"].append({"hash": _hash_jsonable(manifest), "timestamp": _now_iso(), "manifest": manifest})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ==============================================================================
# Robust loaders
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if s == ".npz":
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            return np.asarray(z[k])
        raise ValueError(f"No arrays found in {path}")
    if s in {".csv", ".tsv"}:
        df = pd.read_csv(path) if s == ".csv" else pd.read_csv(path, sep="\t")
        return df.to_numpy()
    if s == ".parquet":
        return pd.read_parquet(path).to_numpy()
    if s == ".feather":
        return pd.read_feather(path).to_numpy()
    raise ValueError(f"Unsupported array format: {path}")


def _normalize_time_input(X: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """
    Normalize time-domain input to (P,T,W). Accepts (T,W) or (P,T,W).
    """
    if X.ndim == 2:
        T, W = X.shape
        return X[None, :, :], 1, T, W
    if X.ndim == 3:
        P, T, W = X.shape
        return X, P, T, W
    raise ValueError(f"Time-domain input must be (T,W) or (P,T,W); got {X.shape}")


def _normalize_spectrum_input(mu: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Normalize spectral-domain input to (P,B). Accepts (B,) or (P,B).
    """
    if mu.ndim == 1:
        B = mu.shape[0]
        return mu[None, :], 1, B
    if mu.ndim == 2:
        P, B = mu.shape
        return mu, P, B
    raise ValueError(f"Spectrum input must be (B,) or (P,B); got {mu.shape}")


# ==============================================================================
# FFT helpers
# ==============================================================================

def _fft_power_time(X: np.ndarray, nfft: int, detrend: bool, window: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT power along time axis for X ∈ R^{P×T×W}.
    Returns (freqs, power) with power shape P×F×W, F = nfft//2+1 (rfft).
    Detrending: subtract mean per [p,w] channel if requested.
    Window: 'hann' or 'none' (applied along time).
    """
    P, T, W = X.shape
    F = nfft // 2 + 1
    Xw = X.copy()
    if detrend:
        Xw = Xw - Xw.mean(axis=1, keepdims=True)
    if window == "hann":
        win = np.hanning(T).astype(Xw.dtype)
        norm = np.sqrt((win**2).sum())
        Xw = Xw * win[None, :, None] / (norm + 1e-12)
    freqs = np.fft.rfftfreq(nfft, d=1.0)  # normalized freq (cycles/sample)
    # rfft along axis=1 (time). If nfft>T, zero-pad; if smaller, truncate via rfft arg.
    power = np.empty((P, F, W), dtype=np.float64)
    for p in range(P):
        # rfft over time for each wavelength
        Y = np.fft.rfft(Xw[p], n=nfft, axis=0)  # shape F×W
        power[p] = (Y.real**2 + Y.imag**2)
    return freqs, power


def _fft_power_bins(mu: np.ndarray, nfft: int, detrend: bool, window: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT power along spectral bins for μ ∈ R^{P×B}.
    Returns (k, power) with power shape P×K, K = nfft//2+1.
    """
    P, B = mu.shape
    K = nfft // 2 + 1
    Z = mu.copy()
    if detrend:
        Z = Z - Z.mean(axis=1, keepdims=True)
    if window == "hann":
        win = np.hanning(B).astype(Z.dtype)
        norm = np.sqrt((win**2).sum())
        Z = Z * win[None, :] / (norm + 1e-12)
    k = np.fft.rfftfreq(nfft, d=1.0)
    Y = np.fft.rfft(Z, n=nfft, axis=1)  # P×K
    power = (Y.real**2 + Y.imag**2)
    return k, power


def _aggregate_power(PW: np.ndarray, aggregate: str, axis: Optional[Tuple[int, ...]]) -> np.ndarray:
    """
    Aggregate power array over given axes using 'mean' or 'median'.
    """
    if axis is None:
        return PW
    if aggregate == "mean":
        return PW.mean(axis=axis)
    if aggregate == "median":
        return np.median(PW, axis=axis)
    raise ValueError(f"Unknown aggregate: {aggregate}")


# ==============================================================================
# Plotting
# ==============================================================================

def _plot_lines(x: np.ndarray, y: Dict[str, np.ndarray], title: str, xlabel: str, ylabel: str,
                out_png: Path, out_html: Path) -> None:
    _ensure_dir(out_png.parent)
    if _MPL_OK:
        plt.figure(figsize=(11, 5))
        for name, vec in y.items():
            plt.plot(x, vec, label=name, lw=2.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=170)
        plt.close()
    if _PLOTLY_OK:
        fig = go.Figure()
        for name, vec in y.items():
            fig.add_trace(go.Scatter(x=x, y=vec, mode="lines", name=name))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
        pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclass
class Config:
    a_path: Path
    b_path: Path
    mode: str                 # 'time' or 'spectrum'
    nfft: int
    detrend: bool
    window: str
    aggregate: str
    freq_max: Optional[float]
    select_planet: Optional[int]
    select_wl_from: Optional[int]
    select_wl_to: Optional[int]
    outdir: Path
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load inputs
    A = _load_array_any(cfg.a_path)
    B = _load_array_any(cfg.b_path)

    manifest_inputs = {"a": str(cfg.a_path), "b": str(cfg.b_path), "mode": cfg.mode}

    if cfg.mode == "time":
        A3, PA, TA, WA = _normalize_time_input(A)
        B3, PB, TB, WB = _normalize_time_input(B)
        if (TA != TB) or (WA != WB):
            # pad/truncate smaller to match larger along T,W
            T = min(TA, TB); W = min(WA, WB)
            A3 = A3[:, :T, :W]
            B3 = B3[:, :T, :W]
        else:
            T, W = TA, WA

        # Optional channel subset (wavelength slice)
        wl_lo = 0 if cfg.select_wl_from is None else max(0, int(cfg.select_wl_from))
        wl_hi = W if cfg.select_wl_to is None else min(W, int(cfg.select_wl_to))
        wl_slice = slice(wl_lo, wl_hi)
        A3 = A3[:, :, wl_slice]
        B3 = B3[:, :, wl_slice]
        P = max(PA, PB)

        # If both have planet dim >1 and select_planet specified: take that planet
        if cfg.select_planet is not None:
            p = int(cfg.select_planet)
            p = max(0, min(p, A3.shape[0]-1, B3.shape[0]-1))
            A3 = A3[p:p+1]
            B3 = B3[p:p+1]

        # Compute FFT power (average over wavelength and planets per aggregate)
        freqs, PwA = _fft_power_time(A3, cfg.nfft, cfg.detrend, cfg.window)  # P×F×W
        _,     PwB = _fft_power_time(B3, cfg.nfft, cfg.detrend, cfg.window)

        # Aggregate over (P,W) to get per-frequency curves
        pA = _aggregate_power(PwA, cfg.aggregate, axis=(0, 2))
        pB = _aggregate_power(PwB, cfg.aggregate, axis=(0, 2))

        # Truncate frequency max
        if cfg.freq_max is not None:
            mask = freqs <= float(cfg.freq_max)
            freqs, pA, pB = freqs[mask], pA[mask], pB[mask]

        # Ratio (avoid div-by-zero)
        ratio = pB / np.maximum(pA, 1e-18)

        # Save CSVs
        dfA = pd.DataFrame({"freq": freqs, "power_a": pA})
        dfB = pd.DataFrame({"freq": freqs, "power_b": pB})
        dfR = pd.DataFrame({"freq": freqs, "power_ratio_B_over_A": ratio})
        dfA.to_csv(cfg.outdir / "power_a.csv", index=False)
        dfB.to_csv(cfg.outdir / "power_b.csv", index=False)
        dfR.to_csv(cfg.outdir / "power_ratio.csv", index=False)

        # Plots
        _plot_lines(freqs, {"A": pA, "B": pB}, "FFT Power (Time Domain)", "Frequency (cycles/sample)", "Power",
                    cfg.outdir / "power_time.png", cfg.outdir / "power_time.html")
        _plot_lines(freqs, {"B/A": ratio}, "FFT Power Ratio (Time Domain)", "Frequency (cycles/sample)", "Ratio (B/A)",
                    cfg.outdir / "power_ratio_time.png", cfg.outdir / "power_ratio_time.html")

    elif cfg.mode == "spectrum":
        A2, PA, BA = _normalize_spectrum_input(A)
        B2, PB, BB = _normalize_spectrum_input(B)
        if BA != BB:
            K = min(BA, BB)
            A2 = A2[:, :K]
            B2 = B2[:, :K]
        else:
            K = BA

        # Select planet if provided and both have P>1
        if cfg.select_planet is not None:
            p = int(cfg.select_planet)
            p = max(0, min(p, A2.shape[0]-1, B2.shape[0]-1))
            A2 = A2[p:p+1]
            B2 = B2[p:p+1]

        k, pA2 = _fft_power_bins(A2, cfg.nfft, cfg.detrend, cfg.window)  # P×Kf
        _, pB2 = _fft_power_bins(B2, cfg.nfft, cfg.detrend, cfg.window)

        pA = _aggregate_power(pA2, cfg.aggregate, axis=(0,))  # Kf
        pB = _aggregate_power(pB2, cfg.aggregate, axis=(0,))  # Kf

        if cfg.freq_max is not None:
            mask = k <= float(cfg.freq_max)
            k, pA, pB = k[mask], pA[mask], pB[mask]

        ratio = pB / np.maximum(pA, 1e-18)

        dfA = pd.DataFrame({"k": k, "power_a": pA})
        dfB = pd.DataFrame({"k": k, "power_b": pB})
        dfR = pd.DataFrame({"k": k, "power_ratio_B_over_A": ratio})
        dfA.to_csv(cfg.outdir / "power_a.csv", index=False)
        dfB.to_csv(cfg.outdir / "power_b.csv", index=False)
        dfR.to_csv(cfg.outdir / "power_ratio.csv", index=False)

        _plot_lines(k, {"A": pA, "B": pB}, "FFT Power (Spectral)", "Spatial Frequency (1/bin)", "Power",
                    cfg.outdir / "power_spectrum.png", cfg.outdir / "power_spectrum.html")
        _plot_lines(k, {"B/A": ratio}, "FFT Power Ratio (Spectral)", "Spatial Frequency (1/bin)", "Ratio (B/A)",
                    cfg.outdir / "power_ratio_spectrum.png", cfg.outdir / "power_ratio_spectrum.html")
    else:
        raise ValueError("mode must be 'time' or 'spectrum'.")

    # Dashboard
    html_name = cfg.html_name if cfg.html_name.endswith(".html") else "fft_power_compare.html"
    dash = cfg.outdir / html_name
    links = []
    # collect existing files
    for name in ["power_a.csv", "power_b.csv", "power_ratio.csv",
                 "power_time.png", "power_time.html", "power_ratio_time.png", "power_ratio_time.html",
                 "power_spectrum.png", "power_spectrum.html", "power_ratio_spectrum.png", "power_ratio_spectrum.html"]:
        p = cfg.outdir / name
        if p.exists():
            links.append(f'<li><a href="{p.name}" target="_blank" rel="noopener">{p.name}</a></li>')
    dash.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — FFT Power Compare</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<style>
  :root {{ --bg:#0b0e14; --fg:#e6edf3; --muted:#9aa4b2; --card:#111827; --border:#2b3240; --brand:#0b5fff; }}
  body {{ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:2rem; line-height:1.5; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.25rem; margin-bottom:1rem; }}
  a {{ color:var(--brand); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
  <header class="card">
    <h1>FFT Power Compare — SpectraMind V50</h1>
    <div>Generated: {_now_iso()} • mode: {cfg.mode} • aggregate: {cfg.aggregate} • nfft: {cfg.nfft}</div>
  </header>
  <section class="card">
    <h2>Artifacts</h2>
    <ul>
      {''.join(links)}
    </ul>
  </section>
  <footer class="card">
    <small>© SpectraMind V50 • fft_power_compare</small>
  </footer>
</body>
</html>
""", encoding="utf-8")

    # Manifest
    manifest = {
        "tool": "fft_power_compare",
        "timestamp": _now_iso(),
        "inputs": manifest_inputs,
        "params": {
            "nfft": cfg.nfft,
            "detrend": cfg.detrend,
            "window": cfg.window,
            "aggregate": cfg.aggregate,
            "freq_max": cfg.freq_max,
            "select_planet": cfg.select_planet,
            "select_wl_from": cfg.select_wl_from,
            "select_wl_to": cfg.select_wl_to,
        },
        "outputs": {
            "dashboard_html": str(dash),
            "csv_a": str(cfg.outdir / "power_a.csv"),
            "csv_b": str(cfg.outdir / "power_b.csv"),
            "csv_ratio": str(cfg.outdir / "power_ratio.csv"),
        }
    }
    with open(cfg.outdir / "fft_power_compare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    audit.log({
        "action": "run", "status": "ok",
        "a": str(cfg.a_path), "b": str(cfg.b_path),
        "mode": cfg.mode, "aggregate": cfg.aggregate, "nfft": cfg.nfft, "freq_max": cfg.freq_max,
        "outdir": str(cfg.outdir),
        "message": f"FFT compare complete; dashboard={dash.name}"
    })

    if cfg.open_browser and dash.exists():
        try:
            import webbrowser
            webbrowser.open_new_tab(dash.as_uri())
        except Exception:
            pass

    return 0


# ==============================================================================
# CLI
# ==============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fft_power_compare",
        description="Compare FFT power between two arrays (pre/post; A/B) in time or spectrum modes."
    )
    p.add_argument("--a", type=Path, required=True, help="Condition A filepath (NPY/NPZ/CSV/TSV/Parquet/Feather).")
    p.add_argument("--b", type=Path, required=True, help="Condition B filepath.")
    p.add_argument("--mode", type=str, default="time", choices=["time", "spectrum"],
                   help="'time' for time-domain cubes (T×W or P×T×W); 'spectrum' for μ spectra (B or P×B).")
    p.add_argument("--nfft", type=int, default=4096, help="FFT size (rfft).")
    p.add_argument("--detrend", action="store_true", help="Subtract per-channel mean before FFT.")
    p.add_argument("--window", type=str, default="hann", choices=["hann", "none"], help="Window along FFT axis.")
    p.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"],
                   help="Aggregate power across channels/planets.")
    p.add_argument("--freq-max", type=float, default=None, help="Max frequency to display (normalized).")
    p.add_argument("--select-planet", type=int, default=None, help="If provided, compare only this planet index.")
    p.add_argument("--select-wl-from", type=int, default=None, help="For time mode: inclusive start wavelength index.")
    p.add_argument("--select-wl-to", type=int, default=None, help="For time mode: exclusive end wavelength index.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    p.add_argument("--html-name", type=str, default="fft_power_compare.html", help="Dashboard HTML name.")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in browser.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        a_path=args.a.resolve(),
        b_path=args.b.resolve(),
        mode=str(args.mode),
        nfft=int(args.nfft),
        detrend=bool(args.detrend),
        window=str(args.window),
        aggregate=str(args.aggregate),
        freq_max=float(args.freq_max) if args.freq_max is not None else None,
        select_planet=int(args.select_planet) if args.select_planet is not None else None,
        select_wl_from=int(args.select_wl_from) if args.select_wl_from is not None else None,
        select_wl_to=int(args.select_wl_to) if args.select_wl_to is not None else None,
        outdir=args.outdir.resolve(),
        html_name=str(args.html_name),
        open_browser=bool(args.open_browser),
    )

    audit = AuditLogger(
        md_path=Path("logs") / "v50_debug_log.md",
        jsonl_path=Path("logs") / "v50_runs.jsonl",
    )
    audit.log({
        "action": "start", "status": "running",
        "a": str(cfg.a_path), "b": str(cfg.b_path),
        "mode": cfg.mode, "aggregate": cfg.aggregate, "nfft": cfg.nfft, "freq_max": cfg.freq_max,
        "outdir": str(cfg.outdir), "message": "Starting fft_power_compare"
    })

    try:
        rc = run(cfg, audit)
        return rc
    except Exception as e:
        import traceback
        traceback.print_exc()
        audit.log({
            "action": "run", "status": "error",
            "a": str(cfg.a_path), "b": str(cfg.b_path),
            "mode": cfg.mode, "aggregate": cfg.aggregate,
            "outdir": str(cfg.outdir), "message": f"{type(e).__name__}: {e}"
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
