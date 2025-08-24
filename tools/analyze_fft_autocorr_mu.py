#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/diagnostics/analyze_fft_autocorr_mu.py

SpectraMind V50 — FFT + Autocorrelation Analyzer for μ(λ) Spectra (Upgraded, Challenge‑Grade)
============================================================================================

Purpose
-------
Analyze per‑planet μ spectra with:
• FFT power diagnostics (peak frequency, high‑frequency fraction, band power)
• Autocorrelation diagnostics (dominant lag, normalized peak)
• Optional symbolic + SHAP overlays (aggregations per planet)
• Optional molecule template matching (H2O / CO2 / CH4) in wavelength or frequency space
• Exports: JSON summary, per‑planet CSV, plots (PNG), and a lightweight HTML report
• Append‑only audit entry to logs/v50_debug_log.md

Inputs
------
--mu                .npy array [N_planets, N_bins] or [N_bins]
--wavelengths       .npy of bin centers (μm) [N_bins] (recommended for template masks)
--planet-ids        .txt or .csv (first col) list of planet IDs, length = N_planets (optional)
--symbolic-json     JSON mapping {planet_id: {rule: score or vector}} (optional; flexible)
--shap-json         JSON mapping {planet_id: ...} (optional; flexible)
--templates-json    Optional JSON with molecule band definitions (μm ranges) or frequency masks
                    Example:
                    {
                      "H2O": {"bands_um": [[1.3,1.5],[1.8,2.0]]},
                      "CO2": {"bands_um": [[1.95,2.05],[4.15,4.35]]},
                      "CH4": {"bands_um": [[3.2,3.5]]}
                    }

Outputs (in --outdir)
---------------------
• fft_autocorr_summary.json
• fft_autocorr_per_planet.csv
• plots/
    - fft_power_mean.png
    - fft_peak_hist.png
    - fft_hf_fraction_hist.png
    - autocorr_peak_lag_hist.png
    - template_correlation_bar.png   (if templates provided)
• report.html  (if --html)
• Append audit line: logs/v50_debug_log.md

CLI Example
-----------
python -m src.diagnostics.analyze_fft_autocorr_mu run \
  --mu outputs/predictions/mu.npy \
  --wavelengths data/metadata/wavelengths.npy \
  --planet-ids data/metadata/planet_ids.csv \
  --symbolic-json outputs/diagnostics/symbolic_results.json \
  --shap-json outputs/diagnostics/shap_overlay.json \
  --templates-json configs/molecule_bands.json \
  --outdir outputs/diagnostics/fft_autocorr \
  --hf-cut 0.25 \
  --html --open-html
"""

from __future__ import annotations

import csv
import json
import math
import os
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _append_audit(message: str, log_path: Path) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{dt.datetime.now().isoformat(timespec='seconds')}] analyze_fft_autocorr_mu: {message}\n")
    except Exception:
        pass

def _read_npy(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        console.print(f"[yellow]WARN[/] npy not found: {path}")
        return None
    return np.load(str(path))

def _read_planet_ids(path: Optional[Path], n: Optional[int]) -> Optional[List[str]]:
    if path is None:
        if n is None:
            return None
        # fabricate IDs if not provided
        return [f"planet_{i:04d}" for i in range(n)]
    if not path.exists():
        console.print(f"[yellow]WARN[/] planet-ids file not found: {path}")
        return None
    if path.suffix.lower() == ".txt":
        ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return ids
    # CSV, first column
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            out.append(str(row[0]).strip())
    return out

def _read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        console.print(f"[yellow]WARN[/] JSON not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[yellow]WARN[/] Failed to read JSON {path}: {e}")
        return None


# --------------------------------------------------------------------------------------
# Core computations
# --------------------------------------------------------------------------------------

def compute_fft_power(x: np.ndarray) -> np.ndarray:
    """
    Returns rFFT power spectrum (non-negative frequencies).
    Input x: 1D array (length B). Removes mean prior to FFT.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    X = np.fft.rfft(x)
    power = (X * np.conj(X)).real
    return power  # len = floor(B/2)+1

def compute_autocorr(x: np.ndarray, norm: bool = True) -> np.ndarray:
    """
    Full autocorrelation via FFT-based or numpy correlate (here np.correlate).
    Returns centered autocorr with same length as x (lags -(B-1)..(B-1)); we then take non-negative lags.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if np.allclose(x, 0):
        ac = np.zeros(2 * len(x) - 1)
    else:
        ac = np.correlate(x, x, mode="full")
    if norm and np.max(np.abs(ac)) > 0:
        ac = ac / np.max(np.abs(ac))
    return ac  # length 2B-1

def hf_fraction(power: np.ndarray, cut_ratio: float) -> float:
    """
    High-frequency fraction of total power above a cutoff ratio of Nyquist.
    cut_ratio in (0, 1]. Uses index threshold int(cut_ratio * (N-1)) on rFFT bins.
    """
    if power.size <= 1:
        return 0.0
    nyq_idx = power.size - 1
    idx_cut = max(1, min(nyq_idx, int(math.floor(cut_ratio * nyq_idx))))
    total = np.sum(power[1:]) + 1e-12
    hf = np.sum(power[idx_cut:]) / total
    return float(hf)

def dominant_fft_peak(power: np.ndarray) -> Tuple[int, float]:
    """
    Returns (peak_index, peak_value) excluding DC (index 0).
    """
    if power.size <= 1:
        return 0, 0.0
    idx = int(np.argmax(power[1:])) + 1
    return idx, float(power[idx])

def dominant_ac_lag(ac: np.ndarray) -> Tuple[int, float]:
    """
    From autocorr (length 2B-1), take non-negative lags and find dominant lag > 0 (exclude zero-lag).
    Returns (lag_index, value) where lag index is sample lag (1..B-1).
    """
    n = (len(ac) + 1) // 2  # B
    pos = ac[n-1:]          # lags 0..B-1
    if pos.size <= 1:
        return 0, 0.0
    lag = int(np.argmax(pos[1:])) + 1
    return lag, float(pos[lag])

def band_mask_from_ranges(wavelengths: np.ndarray, ranges_um: List[List[float]]) -> np.ndarray:
    """
    Build boolean mask for wavelengths in any of the provided [lo, hi] μm ranges.
    """
    mask = np.zeros_like(wavelengths, dtype=bool)
    for lo, hi in ranges_um:
        mask |= (wavelengths >= lo) & (wavelengths <= hi)
    return mask

def template_match_score(mu: np.ndarray, mask: np.ndarray, method: str = "corr") -> float:
    """
    Compute simple template score using masked region:
    method "corr": Pearson corr between (mu within mask) and a flat depth (mean-subtracted).
    method "contrast": mean(mu in mask) - mean(mu outside mask)
    """
    mu = np.asarray(mu, dtype=float)
    if mask.sum() < 2 or (~mask).sum() < 2:
        return float("nan")
    if method == "contrast":
        return float(np.nanmean(mu[mask]) - np.nanmean(mu[~mask]))
    # correlation to a constant-depth template: use normalized mu inside mask
    a = mu[mask] - np.nanmean(mu[mask])
    b = np.ones_like(a)
    if np.allclose(a.std(), 0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def summarize_symbolic(symbolic_json: Optional[Dict[str, Any]], planet_id: str) -> float:
    """
    Produce a single scalar per planet from a flexible symbolic JSON:
    - If per-planet dict of rule->scalar: mean absolute
    - If contains vectors per wavelength: take mean absolute over all entries
    """
    if symbolic_json is None:
        return float("nan")
    p = symbolic_json.get(planet_id)
    if p is None:
        return float("nan")
    vals = []
    if isinstance(p, dict):
        for v in p.values():
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(abs(float(v)))
            elif isinstance(v, list):
                arr = np.asarray(v, dtype=float)
                if arr.size > 0 and np.isfinite(arr).any():
                    vals.append(float(np.nanmean(np.abs(arr))))
    elif isinstance(p, list):
        arr = np.asarray(p, dtype=float)
        if arr.size > 0 and np.isfinite(arr).any():
            vals.append(float(np.nanmean(np.abs(arr))))
    return float(np.nanmean(vals)) if vals else float("nan")

def summarize_shap(shap_json: Optional[Dict[str, Any]], planet_id: str) -> float:
    """
    SHAP JSON can vary; try common keys: mean_abs/magnitude/avg_abs, else arrays -> mean(|.|).
    """
    if shap_json is None:
        return float("nan")
    p = shap_json.get(planet_id)
    if p is None:
        return float("nan")
    if isinstance(p, dict):
        for key in ("mean_abs", "magnitude", "avg_abs", "avg_abs_shap"):
            if key in p and isinstance(p[key], (int, float)) and np.isfinite(p[key]):
                return float(p[key])
        for key in ("values", "bins", "per_bin", "per_bin_abs"):
            if key in p and isinstance(p[key], list):
                arr = np.asarray(p[key], dtype=float)
                if arr.size > 0 and np.isfinite(arr).any():
                    return float(np.nanmean(np.abs(arr)))
    elif isinstance(p, list):
        arr = np.asarray(p, dtype=float)
        if arr.size > 0 and np.isfinite(arr).any():
            return float(np.nanmean(np.abs(arr)))
    return float("nan")


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def plot_fft_power_mean(power_stack: np.ndarray, out: Path) -> None:
    mean_power = np.nanmean(power_stack, axis=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mean_power)
    ax.set_title("Mean rFFT Power (μ)")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

def plot_hist(arr: np.ndarray, title: str, xlabel: str, out: Path, bins: int = 50) -> None:
    valid = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid, bins=bins, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)

def plot_template_bar(scores: Dict[str, float], out: Path, title: str = "Template Correlation") -> None:
    if not scores:
        return
    keys = list(scores.keys())
    vals = [scores[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(keys, vals, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Mean score")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)

def write_html(out_html: Path,
               imgs: List[Tuple[str, str]],
               key_stats: Dict[str, Any],
               timestamp: str,
               run_name: str) -> None:
    rows = "\n".join([f"<div class='card'><h3>{title}</h3><img src='{src}'/></div>" for title, src in imgs])
    kv = "\n".join([f"<div>{k}</div><div>{v}</div>" for k, v in key_stats.items()])
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FFT + Autocorr — SpectraMind V50</title>
<style>
:root{{--bg:#0b1220;--fg:#e6edf3;--muted:#8b96a8;--card:#121a2a;--border:#22304a;--accent:#7aa2ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Ubuntu}}
.container{{max-width:1100px;margin:0 auto;padding:20px}}
h1{{font-size:20px;margin:0 0 10px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px}}
.kv{{display:grid;grid-template-columns:160px 1fr;gap:6px;margin-bottom:12px}}
img{{width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:#0c1526}}
.muted{{color:var(--muted);font-size:12px}}
</style>
</head>
<body>
  <div class="container">
    <h1>FFT + Autocorr Diagnostics — SpectraMind V50</h1>
    <div class="muted">Run: {run_name} • {timestamp}</div>
    <div class="card">
      <h3>Key Stats</h3>
      <div class="kv">{kv}</div>
    </div>
    <div class="grid">{rows}</div>
    <div class="muted" style="margin-top:16px">Generated by analyze_fft_autocorr_mu.py</div>
  </div>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

@app.command("run")
def cli_run(
    mu: Path = typer.Option(..., help="Path to μ.npy [N,B] or [B]"),
    wavelengths: Optional[Path] = typer.Option(None, help="Optional wavelengths.npy [B] (μm)"),
    planet_ids: Optional[Path] = typer.Option(None, help="Optional planet IDs (.txt or first col .csv)"),
    symbolic_json: Optional[Path] = typer.Option(None, help="Optional symbolic JSON per planet"),
    shap_json: Optional[Path] = typer.Option(None, help="Optional SHAP JSON per planet"),
    templates_json: Optional[Path] = typer.Option(None, help="Optional molecule band ranges JSON"),
    outdir: Path = typer.Option(Path("outputs/diagnostics/fft_autocorr"), help="Output directory"),
    hf_cut: float = typer.Option(0.25, help="High-frequency cutoff ratio (0,1] for HF fraction"),
    run_name: str = typer.Option("run", help="Run label"),
    seed: int = typer.Option(1337, help="Random seed (unused but logged)"),
    html: bool = typer.Option(False, "--html/--no-html", help="Write a lightweight HTML report"),
    open_html: bool = typer.Option(False, "--open-html/--no-open-html", help="Open HTML after writing"),
    log_path: Path = typer.Option(Path("logs/v50_debug_log.md"), help="Append-only audit log path"),
):
    """
    Analyze FFT and autocorrelation metrics for μ spectra; overlay symbolic/SHAP; optional molecule templates.
    """
    try:
        outdir = _ensure_dir(outdir)
        plots_dir = _ensure_dir(outdir / "plots")

        MU = _read_npy(mu)
        if MU is None:
            raise FileNotFoundError(f"Missing mu: {mu}")
        if MU.ndim == 1:
            MU = MU[None, :]
        N, B = MU.shape

        WL = _read_npy(wavelengths)
        if WL is not None and WL.size != B:
            console.print(f"[yellow]WARN[/] wavelengths size {WL.size} != B {B}; ignoring WL")
            WL = None

        ids = _read_planet_ids(planet_ids, n=N)
        if ids is None or len(ids) != N:
            ids = [f"planet_{i:04d}" for i in range(N)]

        symbolic = _read_json(symbolic_json)
        shap = _read_json(shap_json)
        templates = _read_json(templates_json)

        # Build molecule masks if templates and wavelengths provided
        mol_masks: Dict[str, np.ndarray] = {}
        if WL is not None and templates:
            for mol, cfg in templates.items():
                bands = cfg.get("bands_um") if isinstance(cfg, dict) else None
                if isinstance(bands, list) and all(isinstance(r, list) and len(r) == 2 for r in bands):
                    mol_masks[mol] = band_mask_from_ranges(WL, bands)

        # FFT & Autocorr stacks
        fft_power_stack = np.zeros((N, (B // 2) + 1), dtype=float)
        fft_peak_bins = np.full(N, np.nan)
        hf_fracs = np.full(N, np.nan)
        ac_peak_lags = np.full(N, np.nan)
        ac_peak_vals = np.full(N, np.nan)

        # Template scores per molecule
        mol_scores: Dict[str, List[float]] = {mol: [] for mol in mol_masks.keys()}

        # Symbolic/SHAP aggregations
        symbolic_scores = np.full(N, np.nan)
        shap_scores = np.full(N, np.nan)

        # Iterate planets
        for i in range(N):
            x = MU[i]
            # FFT power
            p = compute_fft_power(x)
            fft_power_stack[i] = p
            pk_idx, pk_val = dominant_fft_peak(p)
            fft_peak_bins[i] = pk_idx
            hf_fracs[i] = hf_fraction(p, cut_ratio=hf_cut)
            # Autocorr
            ac = compute_autocorr(x, norm=True)
            lag, lag_val = dominant_ac_lag(ac)
            ac_peak_lags[i] = lag
            ac_peak_vals[i] = lag_val
            # Templates
            if mol_masks:
                for mol, mask in mol_masks.items():
                    score = template_match_score(x, mask, method="contrast")
                    mol_scores[mol].append(score)
            # Symbolic/SHAP
            pid = ids[i]
            symbolic_scores[i] = summarize_symbolic(symbolic, pid) if symbolic else float("nan")
            shap_scores[i] = summarize_shap(shap, pid) if shap else float("nan")

        # Aggregate template means
        mol_means = {mol: float(np.nanmean(vals)) if len(vals) else float("nan") for mol, vals in mol_scores.items()}

        # Save per-planet CSV
        csv_path = outdir / "fft_autocorr_per_planet.csv"
        header = ["planet_id", "fft_peak_bin", "fft_hf_fraction", "ac_peak_lag", "ac_peak_val", "symbolic_agg", "shap_agg"]
        header += [f"template_{mol}" for mol in mol_masks.keys()]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(N):
                row = [
                    ids[i],
                    int(fft_peak_bins[i]) if np.isfinite(fft_peak_bins[i]) else "",
                    round(float(hf_fracs[i]), 6) if np.isfinite(hf_fracs[i]) else "",
                    int(ac_peak_lags[i]) if np.isfinite(ac_peak_lags[i]) else "",
                    round(float(ac_peak_vals[i]), 6) if np.isfinite(ac_peak_vals[i]) else "",
                    round(float(symbolic_scores[i]), 6) if np.isfinite(symbolic_scores[i]) else "",
                    round(float(shap_scores[i]), 6) if np.isfinite(shap_scores[i]) else "",
                ]
                for mol in mol_masks.keys():
                    val = mol_scores[mol][i] if i < len(mol_scores[mol]) else float("nan")
                    row.append(round(float(val), 6) if np.isfinite(val) else "")
                w.writerow(row)

        # Plots
        plot_fft_power_mean(fft_power_stack, plots_dir / "fft_power_mean.png")
        plot_hist(fft_peak_bins, "FFT Dominant Peak (bin index)", "bin index", plots_dir / "fft_peak_hist.png", bins=min(60, max(10, int(np.nanmax(fft_peak_bins) + 1))))
        plot_hist(hf_fracs, "High-Frequency Power Fraction", "fraction", plots_dir / "fft_hf_fraction_hist.png", bins=40)
        plot_hist(ac_peak_lags, "Autocorr Dominant Lag (samples)", "lag", plots_dir / "autocorr_peak_lag_hist.png", bins=50)
        if mol_masks:
            plot_template_bar(mol_means, plots_dir / "template_correlation_bar.png", "Template Mean Contrast (μ in-band minus out-of-band)")

        # JSON summary
        summary = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "run_name": run_name,
            "shape": {"planets": int(N), "bins": int(B), "rfft_bins": int((B // 2) + 1)},
            "params": {"hf_cut": float(hf_cut)},
            "metrics": {
                "fft_peak_bin_mean": float(np.nanmean(fft_peak_bins)),
                "fft_hf_fraction_mean": float(np.nanmean(hf_fracs)),
                "ac_peak_lag_mean": float(np.nanmean(ac_peak_lags)),
                "symbolic_agg_mean": float(np.nanmean(symbolic_scores)) if symbolic is not None else None,
                "shap_agg_mean": float(np.nanmean(shap_scores)) if shap is not None else None,
            },
            "template_means": mol_means if mol_masks else {},
            "artifacts": {
                "csv": csv_path.as_posix(),
                "plots": {
                    "fft_power_mean": (plots_dir / "fft_power_mean.png").as_posix(),
                    "fft_peak_hist": (plots_dir / "fft_peak_hist.png").as_posix(),
                    "hf_fraction_hist": (plots_dir / "fft_hf_fraction_hist.png").as_posix(),
                    "autocorr_peak_lag_hist": (plots_dir / "autocorr_peak_lag_hist.png").as_posix(),
                    "template_mean_bar": (plots_dir / "template_correlation_bar.png").as_posix() if mol_masks else None,
                },
            },
        }
        json_path = outdir / "fft_autocorr_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        # HTML
        if html:
            imgs = [
                ("Mean rFFT Power", summary["artifacts"]["plots"]["fft_power_mean"]),
                ("FFT Peak (hist)", summary["artifacts"]["plots"]["fft_peak_hist"]),
                ("High-Freq Fraction (hist)", summary["artifacts"]["plots"]["hf_fraction_hist"]),
                ("Autocorr Peak Lag (hist)", summary["artifacts"]["plots"]["autocorr_peak_lag_hist"]),
            ]
            if mol_masks:
                imgs.append(("Template Mean Contrast", summary["artifacts"]["plots"]["template_mean_bar"]))
            key_stats = {
                "Planets": N,
                "Bins": B,
                "HF cutoff": hf_cut,
                "FFT peak bin (mean)": f"{summary['metrics']['fft_peak_bin_mean']:.3f}",
                "HF fraction (mean)": f"{summary['metrics']['fft_hf_fraction_mean']:.4f}",
                "AC peak lag (mean)": f"{summary['metrics']['ac_peak_lag_mean']:.3f}",
            }
            if symbolic is not None:
                key_stats["Symbolic agg (mean)"] = f"{summary['metrics']['symbolic_agg_mean']:.4f}"
            if shap is not None:
                key_stats["SHAP agg (mean)"] = f"{summary['metrics']['shap_agg_mean']:.4f}"

            html_path = outdir / "report.html"
            write_html(
                html_path,
                imgs=imgs,
                key_stats=key_stats,
                timestamp=summary["timestamp"],
                run_name=run_name,
            )
            # update JSON with html path
            summary["artifacts"]["html"] = html_path.as_posix()
            json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            if open_html:
                try:
                    import webbrowser
                    webbrowser.open(f"file://{html_path.resolve().as_posix()}")
                except Exception:
                    pass

        # Console summary
        console.print(Panel.fit(
            f"[bold]FFT+Autocorr Summary[/]\n"
            f"N={N} B={B} hf_cut={hf_cut}\n"
            f"fft_peak_bin_mean={summary['metrics']['fft_peak_bin_mean']:.3f}\n"
            f"hf_fraction_mean={summary['metrics']['fft_hf_fraction_mean']:.4f}\n"
            f"ac_peak_lag_mean={summary['metrics']['ac_peak_lag_mean']:.3f}\n"
            + (f"symbolic_agg_mean={summary['metrics']['symbolic_agg_mean']:.4f}\n" if symbolic is not None else "")
            + (f"shap_agg_mean={summary['metrics']['shap_agg_mean']:.4f}\n" if shap is not None else "")
            + (f"templates={list(mol_masks.keys())}\n" if mol_masks else ""),
            title="SpectraMind V50 — FFT & Autocorr", style="cyan"))

        # Audit
        _append_audit(
            f"outdir={outdir.as_posix()} N={N} B={B} hf_cut={hf_cut} html={'yes' if html else 'no'}",
            log_path=log_path
        )

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted[/]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(Panel.fit(f"ERROR: {e}", style="red"))
        raise typer.Exit(code=1)


@app.callback()
def _cb():
    """
    SpectraMind V50 — FFT + Autocorrelation Analyzer for μ spectra with symbolic/SHAP overlays and molecule templates.
    Exports per‑planet CSV, JSON summary, PNG plots, and an optional HTML report; logs to v50_debug_log.md.
    """
    pass


if __name__ == "__main__":
    app()
