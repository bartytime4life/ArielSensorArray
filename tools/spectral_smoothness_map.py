#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/diagnostics/spectral_smoothness_map.py

SpectraMind V50 — Spectral Smoothness Map (Upgraded, Challenge‑Grade)
=====================================================================

Purpose
-------
Quantify and visualize spectral smoothness/roughness of predicted μ(λ) across planets and bins.
Produces per‑planet scalar metrics and per‑planet×bin roughness maps, with optional FFT high‑freq
fraction and total variation (TV). Exports JSON/CSV summaries and diagnostic PNGs ready for the
unified HTML dashboard. CLI‑first, CI/Kaggle‑friendly, with audit logging.

Metrics
-------
• Curvature (C2): mean squared second‑difference across bins (per‑planet scalar)
• Total Variation (TV1): mean absolute first‑difference across bins (per‑planet scalar)
• FFT High‑Frequency Fraction (HF_frac): fraction of power above a cutoff (per‑planet scalar)
• Roughness Map (|Δ²μ|): per‑planet×bin (aligned to inner bins) used for heatmap

Inputs
------
• --mu:           .npy of shape [N_planets, N_bins] (predicted μ)
• --wavelengths:  optional .npy [N_bins] for X‑axis ticks in plots
• --planet-ids:   .txt (one per line) or .csv (first column) with N_planets IDs
• --labels-csv:   optional CSV with columns: planet_id,label (categorical)
• --confidence-csv: optional CSV with columns: planet_id,confidence in [0,1]
• --outdir:       output directory (JSON/CSV/PNGs saved here)

Outputs
-------
• outdir/smoothness_summary.json        (per‑planet metrics dictionary)
• outdir/smoothness_summary.csv         (per‑planet table)
• outdir/roughness_map.npy              (|Δ²μ| map [N, B-2])
• outdir/plots/roughness_heatmap.png    (per‑planet×bin heatmap)
• outdir/plots/metrics_hist.png         (histograms of C2/TV1/HF_frac)
• outdir/plots/c2_vs_confidence.png     (if confidence available)
• outdir/plots/c2_by_label.png          (if labels available)
• logs/v50_debug_log.md                 (appended audit line)

Typical usage
-------------
python -m src.diagnostics.spectral_smoothness_map run \
  --mu outputs/predictions/mu.npy \
  --wavelengths data/metadata/wavelengths.npy \
  --planet-ids data/metadata/planet_ids.csv \
  --labels-csv outputs/labels.csv \
  --confidence-csv outputs/diagnostics/confidence.csv \
  --fft-cut 0.25 \
  --normalize \
  --title "Spectral Smoothness — SpectraMind V50" \
  --outdir outputs/smoothness \
  --open-plots
"""
from __future__ import annotations

import csv
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# ======================================================================================
# I/O helpers
# ======================================================================================

def _read_npy(path: Optional[Path]) -> Optional[np.ndarray]:
    if not path:
        return None
    if not path.exists():
        console.print(f"[yellow]WARN[/] npy not found: {path}")
        return None
    return np.load(str(path))

def _read_planet_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing planet_ids file: {path}")
    if path.suffix.lower() == ".txt":
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # CSV first column
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            out.append(str(row[0]).strip())
    return out

def _read_labels_csv(path: Optional[Path]) -> Dict[str, str]:
    if not path: return {}
    if not path.exists():
        console.print(f"[yellow]WARN[/] labels csv not found: {path}")
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("planet_id","")).strip()
            lab = str(row.get("label","")).strip()
            if pid: out[pid] = lab
    return out

def _read_confidence_csv(path: Optional[Path]) -> Dict[str, float]:
    if not path: return {}
    if not path.exists():
        console.print(f"[yellow]WARN[/] confidence csv not found: {path}")
        return {}
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row.get("planet_id","")).strip()
            try:
                val = float(row.get("confidence",""))
            except Exception:
                val = float("nan")
            if pid and np.isfinite(val):
                out[pid] = float(np.clip(val, 0.0, 1.0))
    return out

def _append_audit(log_path: Path, message: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- [{dt.datetime.now().isoformat(timespec='seconds')}] spectral_smoothness_map: {message}\n")

# ======================================================================================
# Smoothness metrics
# ======================================================================================

def _second_diff(mu: np.ndarray) -> np.ndarray:
    """Δ²μ along wavelength axis: shape [N, B-2]."""
    return np.diff(mu, n=2, axis=1)

def curvature_c2(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        roughness_map = |Δ²μ| (N, B-2)
        c2_per_planet = mean( (Δ²μ)^2 ) per planet (N,)
    """
    d2 = _second_diff(mu)
    rough = np.abs(d2)
    c2 = np.mean(d2 ** 2, axis=1)
    return rough, c2

def total_variation_tv1(mu: np.ndarray) -> np.ndarray:
    """TV1 per planet: mean |Δμ| across bins."""
    d1 = np.diff(mu, n=1, axis=1)
    return np.mean(np.abs(d1), axis=1)

def fft_highfreq_fraction(mu: np.ndarray, cut_ratio: float = 0.25) -> np.ndarray:
    """
    Fraction of rFFT power above a normalized cutoff in (0, 0.5].
    For each planet, rFFT over bins, compute sum(power[f>=fc]) / sum(power[all]).
    """
    N, B = mu.shape
    if B < 4:
        return np.zeros(N, dtype=float)
    # center each spectrum
    mu_c = mu - np.mean(mu, axis=1, keepdims=True)
    # rFFT along bins
    fft_vals = np.fft.rfft(mu_c, axis=1)
    power = (fft_vals * np.conj(fft_vals)).real
    # frequency index cutoff (exclude DC at index 0)
    max_idx = power.shape[1] - 1
    fc_idx = int(np.floor(cut_ratio * max_idx))
    fc_idx = max(1, min(fc_idx, max_idx))
    high = np.sum(power[:, fc_idx:], axis=1)
    total = np.sum(power[:, 1:], axis=1) + 1e-12
    return (high / total).astype(float)

def normalize_per_planet(mu: np.ndarray) -> np.ndarray:
    """Optional normalization: zero‑mean, unit‑std per planet (robust to constant spectra)."""
    m = np.mean(mu, axis=1, keepdims=True)
    s = np.std(mu, axis=1, keepdims=True)
    s = np.where(s <= 1e-12, 1.0, s)
    return (mu - m) / s

# ======================================================================================
# Plotting
# ======================================================================================

def _save_heatmap(rough_map: np.ndarray, out_png: Path, wavelengths: Optional[np.ndarray], planet_ids: List[str], title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(rough_map, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_ylabel("Planet index")
    ax.set_xlabel("Bin index (inner, after Δ²)")
    if wavelengths is not None and wavelengths.size == rough_map.shape[1]:
        # label some ticks
        ticks = np.linspace(0, rough_map.shape[1]-1, num=min(10, rough_map.shape[1]), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{wavelengths[t]:.2f}" for t in ticks], rotation=45, ha="right")
        ax.set_xlabel("Wavelength (μm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Δ²μ|")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def _save_hist(c2: np.ndarray, tv1: np.ndarray, hf: np.ndarray, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(c2, bins=40, color="#4f83cc", alpha=0.9)
    axes[0].set_title("C2 (curvature)")
    axes[1].hist(tv1, bins=40, color="#10b981", alpha=0.9)
    axes[1].set_title("TV1 (total variation)")
    axes[2].hist(hf, bins=40, color="#eab308", alpha=0.9)
    axes[2].set_title("HF_frac (FFT)")
    fig.suptitle(title)
    for ax in axes:
        ax.grid(alpha=0.25, ls="--")
        ax.set_ylabel("Count")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def _save_scatter_c2_conf(c2: np.ndarray, planet_ids: List[str], conf_map: Dict[str, float], out_png: Path):
    vals = np.array([conf_map.get(pid, np.nan) for pid in planet_ids], dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(vals[mask], c2[mask], s=18, alpha=0.8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("C2 curvature")
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def _save_c2_by_label(c2: np.ndarray, planet_ids: List[str], labels: Dict[str, str], out_png: Path):
    if not labels:
        return
    labvals: Dict[str, List[float]] = {}
    for i, pid in enumerate(planet_ids):
        lab = labels.get(pid, "unknown")
        labvals.setdefault(lab, []).append(float(c2[i]))
    if not labvals:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labs = sorted(labvals.keys())
    data = [labvals[k] for k in labs]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.boxplot(data, labels=labs, showfliers=False)
    ax.set_ylabel("C2 curvature")
    ax.set_title("C2 by label")
    ax.grid(alpha=0.25, ls="--", axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

# ======================================================================================
# Save artifacts
# ======================================================================================

def _save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _save_csv(path: Path, header: List[str], rows: List[List[Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ======================================================================================
# CLI
# ======================================================================================

@app.command("run")
def cli_run(
    mu: Path = typer.Option(..., help="Path to μ.npy [N,B]"),
    wavelengths: Optional[Path] = typer.Option(None, help="Optional wavelengths.npy [B] (for X ticks)"),
    planet_ids: Path = typer.Option(..., help="Planet IDs file (.txt or first col .csv)"),
    labels_csv: Optional[Path] = typer.Option(None, help="Optional labels CSV (planet_id,label)"),
    confidence_csv: Optional[Path] = typer.Option(None, help="Optional confidence CSV (planet_id,confidence∈[0,1])"),
    fft_cut: float = typer.Option(0.25, help="Normalized FFT cutoff ratio in (0,0.5] for HF fraction."),
    normalize: bool = typer.Option(False, "--normalize/--no-normalize", help="Zero-mean/unit-std per planet before metrics."),
    title: str = typer.Option("Spectral Smoothness — SpectraMind V50", help="Plot title prefix."),
    outdir: Path = typer.Option(Path("outputs/smoothness"), help="Output directory."),
    open_plots: bool = typer.Option(False, "--open-plots/--no-open-plots", help="Open generated plots in default viewer."),
    log_path: Path = typer.Option(Path("logs/v50_debug_log.md"), help="Append audit line here."),
):
    """
    Compute spectral smoothness metrics/maps and export JSON/CSV/PNGs.
    """
    try:
        console.print(Panel.fit("Loading inputs...", style="cyan"))
        MU = _read_npy(mu)
        if MU is None:
            raise FileNotFoundError(f"Missing mu: {mu}")
        N, B = MU.shape
        pids = _read_planet_ids(planet_ids)
        if len(pids) != N:
            raise ValueError(f"Row mismatch: mu N={N} vs planet_ids={len(pids)}")
        WL = _read_npy(wavelengths)
        labels = _read_labels_csv(labels_csv)
        conf = _read_confidence_csv(confidence_csv)

        if normalize:
            MU = normalize_per_planet(MU)

        console.print(Panel.fit("Computing smoothness metrics...", style="cyan"))
        rough_map, c2 = curvature_c2(MU)            # (N, B-2), (N,)
        tv1 = total_variation_tv1(MU)               # (N,)
        hf = fft_highfreq_fraction(MU, cut_ratio=float(fft_cut))  # (N,)

        # Save arrays and metrics
        outdir.mkdir(parents=True, exist_ok=True)
        plots_dir = outdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        np.save(outdir / "roughness_map.npy", rough_map)

        # JSON summary
        summary_dict: Dict[str, Any] = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "shape": {"planets": int(N), "bins": int(B), "rough_bins": int(rough_map.shape[1])},
            "fft_cut": float(fft_cut),
            "normalize": bool(normalize),
            "metrics": {
                "c2_mean": float(np.mean(c2)),
                "c2_median": float(np.median(c2)),
                "tv1_mean": float(np.mean(tv1)),
                "tv1_median": float(np.median(tv1)),
                "hf_mean": float(np.mean(hf)),
                "hf_median": float(np.median(hf)),
            }
        }
        _save_json(outdir / "smoothness_summary.json", summary_dict)

        # CSV per-planet
        rows = []
        for i, pid in enumerate(pids):
            rows.append([pid, float(c2[i]), float(tv1[i]), float(hf[i])])
        _save_csv(outdir / "smoothness_summary.csv", ["planet_id", "c2", "tv1", "hf_frac"], rows)

        # Heatmap of |Δ²μ| (N, B-2)
        wl_inner = None
        if WL is not None and WL.size == B:
            wl_inner = WL[1:-1]  # align with B-2 interior after Δ²
        _save_heatmap(
            rough_map=rough_map,
            out_png=plots_dir / "roughness_heatmap.png",
            wavelengths=wl_inner,
            planet_ids=pids,
            title=f"{title}: |Δ²μ| heatmap"
        )

        # Histograms of scalar metrics
        _save_hist(c2, tv1, hf, plots_dir / "metrics_hist.png", title=f"{title}: metrics hist")

        # Scatter against confidence (if present)
        if conf:
            _save_scatter_c2_conf(c2, pids, conf, plots_dir / "c2_vs_confidence.png")

        # Boxplot C2 by label (if present)
        if labels:
            _save_c2_by_label(c2, pids, labels, plots_dir / "c2_by_label.png")

        # Audit and finish
        _append_audit(log_path, f"outdir={outdir.as_posix()} N={N} B={B} normalize={normalize} fft_cut={fft_cut}")
        console.print(Panel.fit(f"Smoothness artifacts written to: {outdir}", style="green"))

        if open_plots:
            try:
                import webbrowser
                for p in [
                    plots_dir / "roughness_heatmap.png",
                    plots_dir / "metrics_hist.png",
                    plots_dir / "c2_vs_confidence.png",
                    plots_dir / "c2_by_label.png",
                ]:
                    if p.exists():
                        webbrowser.open(f"file://{p.resolve().as_posix()}")
            except Exception:
                pass

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user.[/]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(Panel.fit(f"ERROR: {e}", style="red"))
        raise typer.Exit(code=1)

@app.callback()
def _cb():
    """
    SpectraMind V50 — Spectral Smoothness Map:
    • Curvature C2, TV1, FFT high‑frequency fraction
    • |Δ²μ| heatmap, per‑planet CSV/JSON, plots for dashboard
    • Optional normalization and wavelength tick labeling
    • Labels/confidence overlays, CLI audit logging
    """
    pass

if __name__ == "__main__":
    app()
