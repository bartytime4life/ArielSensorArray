#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
shap_attention_overlay.py

Overlay SHAP attributions on attention maps for SpectraMind V50 spectral channels (N≈283).

What it does
------------
1) Loads per-channel SHAP values (vector length N) and an attention matrix (N×N or H×N×N).
2) Harmonizes shapes (averages heads if needed), normalizes, and validates alignment with an
   optional wavelength grid (N vector).
3) Produces:
   • A heatmap figure of the attention matrix with a SHAP “importance ribbon” aligned to axes.
   • A ranked bar chart of top‑K channels by |SHAP|.
   • A CSV report with per‑channel metrics (wavelength, shap_value, |shap| rank, attention degree).
4) Saves artifacts (PNG/HTML optional) to a chosen output directory with reproducible file names.

CLI examples
------------
# Minimal: attn (npy/npz/csv), shap (npy/npz/csv)
python shap_attention_overlay.py \
  --attention ./artifacts/attn_heads.npz \
  --shap ./artifacts/shap_values.npy \
  --wavelengths ./data/wavelengths_283.csv \
  --out-dir ./reports/diag_run_01

# Specify top-K and dpi
python shap_attention_overlay.py \
  --attention attn.npy --shap shap.csv --wavelengths waves.npy \
  --topk 20 --dpi 250 --title "Planet-123 SHAP×Attention Overlay" --out-dir ./reports/planet_123

Design notes
------------
• CLI-first, reproducible artifacts, rich console logs, and self-contained validation per SpectraMind V50
  engineering standards (Typer + Rich; headless plotting).  
• Produces lightweight “UI-like” diagnostics (PNG/CSV) that can be consumed by the CLI/CI workflow. 

Author: SpectraMind V50 Tools
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering for CI/servers
import matplotlib.pyplot as plt

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

app = typer.Typer(add_completion=False, help="Overlay SHAP attributions on attention maps.")
console = Console()


# ---------- Utilities ----------

def _load_vector_or_matrix(path: Path, key: Optional[str] = None) -> np.ndarray:
    """
    Load a vector or matrix from .npy / .npz(/key) / .csv file.
    CSV: loads numeric frame → np.ndarray (squeeze).
    NPZ: uses first array if key not provided.
    """
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if key is not None:
            if key not in data.files:
                raise KeyError(f"Key '{key}' not found in NPZ ({data.files}).")
            arr = data[key]
        else:
            # Pick the first array
            first = data.files[0]
            arr = data[first]
    elif ext == ".csv":
        df = pd.read_csv(path, header=None)
        arr = df.values
        if arr.ndim == 2 and (arr.shape[1] == 1 or arr.shape[0] == 1):
            arr = arr.squeeze()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return np.asarray(arr)


def _ensure_2d_attention(attn: np.ndarray) -> np.ndarray:
    """
    Convert attention to (N, N). If attn is (H, N, N), average across heads.
    """
    if attn.ndim == 3:
        # (H, N, N) → mean over heads
        attn2 = np.nanmean(attn, axis=0)
    elif attn.ndim == 2:
        attn2 = attn
    else:
        raise ValueError(f"Attention array must be 2D or 3D; got shape {attn.shape}.")
    # Replace NaNs/infs
    attn2 = np.nan_to_num(attn2, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip small negatives due to numeric noise
    attn2 = np.clip(attn2, a_min=0.0, a_max=None)
    # Normalize to [0,1] (robust)
    maxv = attn2.max()
    if maxv > 0:
        attn2 = attn2 / maxv
    return attn2


def _normalize_shap(shap: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    Normalize SHAP values for visualization; keep sign for coloring.
      mode="zscore": (x - mu) / std
      mode="maxabs": x / max(|x|)
      mode="none":   unchanged (but NaNs→0)
    """
    shap = shap.astype(float)
    shap = np.nan_to_num(shap, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "zscore":
        mu = shap.mean()
        sd = shap.std()
        if sd > 0:
            out = (shap - mu) / sd
        else:
            out = shap - mu
    elif mode == "maxabs":
        denom = np.max(np.abs(shap))
        out = shap / denom if denom > 0 else shap
    elif mode == "none":
        out = shap
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return out


def _compute_attention_degree(attn: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """
    Compute an attention-degree score per channel (row/col sums).
    If symmetrize=True, do (attn + attn.T) / 2 first.
    """
    A = attn
    if symmetrize:
        A = 0.5 * (A + A.T)
    # Degree-like measure (sum of connections)
    deg = A.sum(axis=1)
    return deg


def _safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_csv_report(path: Path, wavelengths: Optional[np.ndarray], shap: np.ndarray,
                     shap_norm: np.ndarray, attn_deg: np.ndarray):
    """
    Save a per-channel report as CSV.
    """
    N = len(shap)
    idx = np.arange(N)
    df = pd.DataFrame({
        "index": idx,
        "wavelength": wavelengths if wavelengths is not None else np.repeat(np.nan, N),
        "shap_value": shap,
        "shap_abs": np.abs(shap),
        "shap_norm": shap_norm,
        "attn_degree": attn_deg
    })
    # Rank by |shap|
    df["shap_abs_rank"] = df["shap_abs"].rank(ascending=False, method="dense").astype(int)
    df.sort_values("shap_abs_rank", inplace=True)
    df.to_csv(path, index=False)


def _pretty_title(title: Optional[str], fallback: str) -> str:
    return title if (title is not None and len(title.strip()) > 0) else fallback


# ---------- Plotting ----------

def plot_attention_with_shap(
    out_png: Path,
    attn: np.ndarray,
    shap_norm: np.ndarray,
    wavelengths: Optional[np.ndarray],
    title: str,
    dpi: int = 200,
    cmap_attn: str = "viridis",
    cmap_shap: str = "coolwarm",
) -> None:
    """
    Render a figure: attention heatmap with SHAP importance ribbons on top/left axes.
    """
    N = attn.shape[0]
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = fig.add_gridspec(nrows=6, ncols=6)

    # Top ribbon (SHAP across columns)
    ax_top = fig.add_subplot(gs[0, 1:6])
    # Left ribbon (SHAP across rows) – align orientation
    ax_left = fig.add_subplot(gs[1:6, 0])
    # Main heatmap
    ax = fig.add_subplot(gs[1:6, 1:6])

    # Main heatmap
    hm = ax.imshow(attn, interpolation="nearest", aspect="auto", cmap=cmap_attn)
    cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Attention (normalized)")

    # X/Y ticks hint (reduce clutter)
    step = max(1, N // 10)
    xticks = np.arange(0, N, step)
    yticks = np.arange(0, N, step)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if wavelengths is not None and wavelengths.shape[0] == N:
        # Show wavelengths on coarse ticks
        xlabels = [f"{wavelengths[i]:.3f}" for i in xticks]
        ylabels = [f"{wavelengths[i]:.3f}" for i in yticks]
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel("Wavelength index (tick labels = μm)")
        ax.set_ylabel("Wavelength index (tick labels = μm)")
    else:
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")

    ax.set_title(title, fontsize=12)

    # Top SHAP ribbon (color = shap_norm; magnitude encoded by color)
    shap_top = shap_norm.reshape(1, -1)  # (1, N)
    im_top = ax_top.imshow(shap_top, aspect="auto", cmap=cmap_shap, vmin=-np.max(np.abs(shap_norm)), vmax=np.max(np.abs(shap_norm)))
    ax_top.set_yticks([])
    ax_top.set_xticks(xticks)
    if wavelengths is not None and wavelengths.shape[0] == N:
        ax_top.set_xticklabels([f"{wavelengths[i]:.3f}" for i in xticks], rotation=45, ha="right", fontsize=8)
        ax_top.set_title("SHAP (normalized) across channels", fontsize=10)
    else:
        ax_top.set_xticklabels(xticks, rotation=45, ha="right", fontsize=8)
        ax_top.set_title("SHAP (normalized) across channels", fontsize=10)

    # Left SHAP ribbon (vertical)
    shap_left = shap_norm.reshape(-1, 1)  # (N, 1)
    im_left = ax_left.imshow(shap_left, aspect="auto", cmap=cmap_shap, vmin=-np.max(np.abs(shap_norm)), vmax=np.max(np.abs(shap_norm)))
    ax_left.set_xticks([])
    ax_left.set_yticks(yticks)
    if wavelengths is not None and wavelengths.shape[0] == N:
        ax_left.set_yticklabels([f"{wavelengths[i]:.3f}" for i in yticks], fontsize=8)
    else:
        ax_left.set_yticklabels(yticks, fontsize=8)

    # Colorbar for SHAP ribbons (shared)
    cbar2 = fig.colorbar(im_left, ax=[ax_left, ax_top], fraction=0.046, pad=0.02)
    cbar2.set_label("SHAP (normalized)")

    fig.suptitle("Attention × SHAP Overlay", fontsize=13, y=0.99)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_topk_shap_bars(
    out_png: Path,
    shap: np.ndarray,
    wavelengths: Optional[np.ndarray],
    topk: int = 15,
    dpi: int = 200,
    palette: str = "coolwarm"
) -> None:
    """
    Plot the top-K channels by |SHAP| as a horizontal bar chart.
    """
    N = len(shap)
    idx = np.arange(N)
    df = pd.DataFrame({
        "index": idx,
        "wavelength": wavelengths if wavelengths is not None else np.repeat(np.nan, N),
        "shap_value": shap,
        "shap_abs": np.abs(shap)
    })
    df = df.sort_values("shap_abs", ascending=False).head(topk)
    df = df.iloc[::-1]  # largest on top
    labels = [f"{i} | {w:.3f} μm" if not np.isnan(w) else f"{i}" for i, w in zip(df["index"], df["wavelength"])]

    # Color by sign (positive vs negative SHAP)
    colors = [plt.get_cmap(palette)(0.85) if v >= 0 else plt.get_cmap(palette)(0.15) for v in df["shap_value"]]

    fig, ax = plt.subplots(figsize=(8, max(4, int(topk * 0.35))))
    ax.barh(labels, df["shap_value"], color=colors)
    ax.axvline(0.0, color="k", linewidth=1)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"Top-{topk} Channels by |SHAP|")
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ---------- CLI ----------

@app.command()
def run(
    attention: Path = typer.Option(..., exists=True, help="Path to attention matrix .npy/.npz/.csv (N×N or H×N×N)."),
    shap: Path = typer.Option(..., exists=True, help="Path to SHAP vector .npy/.npz/.csv (length N)."),
    wavelengths: Optional[Path] = typer.Option(None, exists=True, help="Optional wavelengths .npy/.npz/.csv (length N)."),
    attn_key: Optional[str] = typer.Option(None, help="Optional NPZ key for attention."),
    shap_key: Optional[str] = typer.Option(None, help="Optional NPZ key for SHAP."),
    waves_key: Optional[str] = typer.Option(None, help="Optional NPZ key for wavelengths."),
    normalize: str = typer.Option("zscore", help="SHAP normalization: zscore | maxabs | none"),
    topk: int = typer.Option(15, min=1, help="Top‑K channels by |SHAP| to show in bar chart."),
    dpi: int = typer.Option(200, help="Figure DPI."),
    title: Optional[str] = typer.Option(None, help="Custom title on heatmap."),
    out_dir: Path = typer.Option(Path("./reports/shap_attention_overlay"), help="Output directory for artifacts."),
    cmap_attn: str = typer.Option("viridis", help="Matplotlib colormap for attention."),
    cmap_shap: str = typer.Option("coolwarm", help="Matplotlib colormap for SHAP."),
    symmetrize: bool = typer.Option(True, help="Symmetrize attention before computing degree."),
):
    """
    Create overlay diagnostics of SHAP attributions and attention matrices and export a CSV report.
    """
    console.rule("[bold cyan]SHAP × Attention Overlay")
    _safe_mkdir(out_dir)

    with console.status("[bold green]Loading inputs..."):
        attn_arr = _load_vector_or_matrix(attention, key=attn_key)
        shap_vec = _load_vector_or_matrix(shap, key=shap_key).squeeze()
        waves = None
        if wavelengths is not None:
            waves = _load_vector_or_matrix(wavelengths, key=waves_key).squeeze()

    # Validate shapes
    attn2 = _ensure_2d_attention(attn_arr)
    N = attn2.shape[0]
    if attn2.shape[0] != attn2.shape[1]:
        raise ValueError(f"Attention must be square (N×N); got {attn2.shape}.")
    if shap_vec.ndim != 1:
        shap_vec = shap_vec.squeeze()
    if shap_vec.shape[0] != N:
        raise ValueError(f"SHAP length {shap_vec.shape[0]} != attention size {N}.")
    if waves is not None and waves.shape[0] != N:
        raise ValueError(f"Wavelength length {waves.shape[0]} != attention size {N}.")

    # Normalize SHAP (retain sign for color)
    shap_norm = _normalize_shap(shap_vec, mode=normalize)

    # Compute per-channel attention degree (useful in CSV)
    attn_degree = _compute_attention_degree(attn2, symmetrize=symmetrize)

    # Prepare outputs
    base = _pretty_title(title, "Attention×SHAP")
    fname_base = base.lower().replace(" ", "_").replace("/", "_")
    heatmap_png = out_dir / f"{fname_base}_heatmap.png"
    bars_png = out_dir / f"{fname_base}_top{topk}_bars.png"
    report_csv = out_dir / f"{fname_base}_report.csv"
    meta_json = out_dir / f"{fname_base}_meta.json"

    # Plot
    console.log(f"[bold]N[/bold]={N} • Rendering heatmap → {heatmap_png.name}")
    plot_attention_with_shap(
        out_png=heatmap_png,
        attn=attn2,
        shap_norm=shap_norm,
        wavelengths=waves,
        title=base,
        dpi=dpi,
        cmap_attn=cmap_attn,
        cmap_shap=cmap_shap,
    )

    console.log(f"Rendering top‑{topk} SHAP bars → {bars_png.name}")
    plot_topk_shap_bars(
        out_png=bars_png,
        shap=shap_vec,
        wavelengths=waves,
        topk=topk,
        dpi=dpi,
    )

    # CSV report
    console.log(f"Saving per‑channel CSV report → {report_csv.name}")
    _save_csv_report(report_csv, waves, shap_vec, shap_norm, attn_degree)

    # Meta
    meta = {
        "attention_path": str(attention),
        "shap_path": str(shap),
        "wavelengths_path": str(wavelengths) if wavelengths else None,
        "normalize": normalize,
        "symmetrize": symmetrize,
        "N": int(N),
        "topk": int(topk),
        "figures": {
            "heatmap_png": str(heatmap_png),
            "topk_bars_png": str(bars_png),
        },
        "report_csv": str(report_csv),
        "colormaps": {"attention": cmap_attn, "shap": cmap_shap},
        "notes": "Artifacts created for SHAP×Attention overlay diagnostics.",
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Pretty console summary
    table = Table(title="Artifacts", box=box.SIMPLE_HEAVY)
    table.add_column("Type", style="bold")
    table.add_column("Path")
    table.add_row("Heatmap", str(heatmap_png))
    table.add_row("Top‑K Bars", str(bars_png))
    table.add_row("CSV Report", str(report_csv))
    table.add_row("Meta JSON", str(meta_json))
    console.print(table)

    # Final panel
    console.print(
        Panel.fit(
            f"[green]Done![/green]\nSaved artifacts to: [bold]{out_dir}[/bold]\n"
            f"[dim]Tip: include these diagnostics in your CLI/CI runs for reproducibility.[/dim]",
            title="SHAP × Attention Overlay",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
