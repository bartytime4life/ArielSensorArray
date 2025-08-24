#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_umap_fusion_latents_v50.py

SpectraMind V50 — Tools: UMAP projection for fused latent spaces.

What this does
--------------
- Loads one or more latent arrays (e.g., *.npy or *.pt) produced by different subsystems
  (time, spectral, graph, metadata encoders).
- Optionally standardizes and/or PCA pre-reduces each latent block.
- Fuses blocks (concat by feature axis, aligned by row/row-index) into a single matrix.
- Runs UMAP to produce a 2D (or 3D) embedding.
- Plots and saves publication-ready figures (PNG/SVG), plus CSV with coordinates & labels.
- Can color the embedding by a label column (categorical) or a numeric column (continuous).
- Offers KMeans clustering overlay and silhouette score to sanity-check structure.
- Deterministic runs via `--seed` (umap-learn + numpy + torch).

Why
---
UMAP helps diagnose whether the fused representation separates scientifically meaningful
classes (e.g., planet types, stellar classes) or encodes spurious artifacts. Put this in the
diagnostic phase to quickly eyeball the geometry of your latent space across runs.

Usage (examples)
----------------
# Minimal (two latent blocks, color by a label CSV):
python tools/plot_umap_fusion_latents_v50.py \
  --latents path/to/time_latents.npy path/to/spectral_latents.pt \
  --labels-csv path/to/labels.csv --label-col planet_class \
  --outdir outputs/umap/ --title "V50 UMAP (time+spectral)"

# With standardization, PCA per-block, and KMeans overlay:
python tools/plot_umap_fusion_latents_v50.py \
  --latents enc/time.npy enc/spec.npy enc/gnn.pt \
  --std --pca-k 64 --kmeans 8 --silhouette \
  --labels-csv meta/labels.csv --label-col star_type \
  --metric cosine --neighbors 30 --min-dist 0.05 \
  --outdir outputs/umap/ --title "Fusion UMAP" --seed 42

Dependencies
------------
- numpy, pandas, matplotlib, seaborn, scikit-learn, umap-learn, torch (optional for *.pt)
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# Optional torch support for *.pt latent tensors
try:
    import torch

    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

# Rich logging (pretty console)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    from rich.logging import RichHandler
    import logging

    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, markup=True)],
    )
    log = logging.getLogger("umap_fusion")
except Exception:
    # Fallback if rich isn't available
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("umap_fusion")
    console = None  # type: ignore


app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


# ------------------------------- Utilities --------------------------------- #


def _set_all_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ensure some determinism, may slow down
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    except Exception:
        pass


def _load_latent(path: Union[str, Path]) -> np.ndarray:
    """Load a latent block from .npy or .pt; returns float32 numpy array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"latent file not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
        arr = np.asarray(arr, dtype=np.float32)
        return arr
    if path.suffix.lower() == ".pt":
        if not _TORCH_OK:
            raise RuntimeError(f"torch not available to load {path}")
        t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.nn.Module):
            raise ValueError(f"{path} appears to be a module; provide tensor array")
        if isinstance(t, (tuple, list)):
            # try first element
            t = t[0]
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Unsupported PT object in {path}: {type(t)}")
        return t.detach().float().cpu().numpy()
    raise ValueError(f"Unsupported latent extension: {path.suffix}")


def _align_blocks(blocks: List[np.ndarray]) -> List[np.ndarray]:
    """Ensure all blocks share the same number of rows. If not, raise error."""
    rows = [b.shape[0] for b in blocks]
    if len(set(rows)) != 1:
        raise ValueError(f"Row count mismatch across blocks: {rows} — latents must align by row.")
    return blocks


def _standardize(block: np.ndarray) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(block)


def _pca_reduce(block: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= block.shape[1]:
        return block
    p = PCA(n_components=k, random_state=0, svd_solver="auto")
    return p.fit_transform(block)


def _merge_blocks(blocks: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _make_outdir(outdir: Union[str, Path]) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_csv_coords(
    coords: np.ndarray,
    labels_df: Optional[pd.DataFrame],
    out_csv: Path,
    label_col: Optional[str],
    color_by: Optional[str],
) -> None:
    df = pd.DataFrame(coords, columns=["umap_1", "umap_2"] if coords.shape[1] == 2 else ["umap_1", "umap_2", "umap_3"])
    if labels_df is not None:
        df = pd.concat([df, labels_df.reset_index(drop=True)], axis=1)
    # Reorder columns to put coloring columns up front if present
    cols = list(df.columns)
    front = []
    for c in [label_col, color_by]:
        if c and c in df.columns and c not in front:
            front.append(c)
    cols = [*front, *[c for c in cols if c not in front]]
    df = df[cols]
    df.to_csv(out_csv, index=False)


def _auto_figsize(n: int) -> Tuple[int, int]:
    # heuristic
    if n < 2_000:
        return (6, 5)
    if n < 20_000:
        return (7, 6)
    return (8, 7)


def _scatter_plot(
    coords: np.ndarray,
    labels_df: Optional[pd.DataFrame],
    out_png: Path,
    title: str,
    label_col: Optional[str],
    color_by: Optional[str],
    alpha: float = 0.8,
    point_size: float = 8.0,
) -> None:
    import seaborn as sns

    plt.figure(figsize=_auto_figsize(coords.shape[0]))
    ax = plt.gca()

    if labels_df is None or (color_by is None and label_col is None):
        # plain scatter
        plt.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha, c="#4C78A8")
        plt.title(title)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(out_png, dpi=250)
        plt.close()
        return

    key = color_by or label_col
    series = labels_df[key].reset_index(drop=True)
    if series.dtype.kind in "ifu":  # numeric
        sc = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size,
            alpha=alpha,
            c=series.values,
            cmap="viridis",
        )
        cbar = plt.colorbar(sc, pad=0.01)
        cbar.set_label(key)
    else:
        # categorical
        palette = sns.color_palette("husl", n_colors=series.nunique())
        cat2idx = {cat: i for i, cat in enumerate(series.astype("category").cat.categories)}
        idx = series.astype("category").cat.codes.values
        plt.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha, c=np.array(palette)[idx])
        # build legend (up to 20 entries to keep readable)
        uniq = series.astype("category").cat.categories
        handles = []
        from matplotlib.lines import Line2D

        for i, cat in enumerate(uniq[:20]):
            handles.append(Line2D([0], [0], marker="o", color="w", label=str(cat), markerfacecolor=palette[i], markersize=6))
        if len(uniq) > 20:
            handles.append(Line2D([0], [0], marker="o", color="w", label="... (truncated)", markerfacecolor="#bbb", markersize=6))
        ax.legend(handles=handles, title=key, bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ------------------------------- CLI Core ---------------------------------- #


@app.command("run")
def run_umap(
    latents: List[Path] = typer.Option(..., "--latents", "-i", help="Paths to latent blocks (*.npy or *.pt). Order matters."),
    outdir: Path = typer.Option("outputs/umap", "--outdir", "-o", help="Output directory."),
    labels_csv: Optional[Path] = typer.Option(None, "--labels-csv", help="Optional CSV with labels/metadata aligned by row."),
    label_col: Optional[str] = typer.Option(None, "--label-col", help="Categorical label column for legend."),
    color_by: Optional[str] = typer.Option(None, "--color-by", help="Column to color points (categorical or numeric)."),
    std: bool = typer.Option(False, "--std", help="Standardize each block before fusion."),
    pca_k: int = typer.Option(0, "--pca-k", help="Optional PCA dims per block before fusion (0 = no PCA)."),
    n_components: int = typer.Option(2, "--components", help="UMAP embedding dimensions (2 or 3)."),
    neighbors: int = typer.Option(15, "--neighbors", "-k", help="UMAP n_neighbors."),
    min_dist: float = typer.Option(0.1, "--min-dist", help="UMAP min_dist."),
    metric: str = typer.Option("euclidean", "--metric", help="UMAP metric (euclidean, cosine, etc.)."),
    densmap: bool = typer.Option(False, "--densmap", help="Use densMAP variant (umap-learn>=0.5)."),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for determinism."),
    sample: int = typer.Option(0, "--sample", help="Optional subsample N rows before UMAP (0 = all)."),
    kmeans: int = typer.Option(0, "--kmeans", help="Optional overlay KMeans clusters (k>1)."),
    silhouette: bool = typer.Option(False, "--silhouette", help="Compute silhouette score (if kmeans>1)."),
    title: str = typer.Option("UMAP (fusion latents)", "--title", help="Figure title."),
    save_svg: bool = typer.Option(False, "--svg", help="Also save SVG figure."),
    point_size: float = typer.Option(8.0, "--point-size", help="Marker size."),
    alpha: float = typer.Option(0.8, "--alpha", help="Marker alpha."),
):
    """
    Fuse latent blocks, run UMAP, and save plots + CSV.

    Assumptions:
    - Each latent file is [N x D_i]; all share the same N (row alignment).
    - labels_csv (if provided) has N rows aligned with the latent rows in the same order.
    """
    _set_all_seeds(seed)
    outdir = _make_outdir(outdir)

    # Load
    blocks: List[np.ndarray] = []
    log.info(f"[bold]- Loading {len(latents)} latent blocks[/bold]")
    for p in latents:
        log.info(f"  • {p}")
        arr = _load_latent(p)
        if arr.ndim != 2:
            raise ValueError(f"Latent must be 2D [N x D], got {arr.shape} in {p}")
        blocks.append(arr)

    _align_blocks(blocks)
    N = blocks[0].shape[0]
    log.info(f"Rows: {N} | Block dims: {[b.shape[1] for b in blocks]}")

    # Optional per-block preprocessing
    if std:
        blocks = [track((_standardize(b) for b in blocks), description="Standardizing blocks", total=len(blocks))]
        blocks = list(blocks)
    if pca_k > 0:
        blocks = [track((_pca_reduce(b, pca_k) for b in blocks), description=f"PCA→{pca_k} per block", total=len(blocks))]
        blocks = list(blocks)

    # Fuse
    X = _merge_blocks(blocks)
    log.info(f"Fused latent: {X.shape}")

    # Optional subsampling
    if sample and sample > 0 and sample < X.shape[0]:
        idx = np.random.RandomState(seed).choice(X.shape[0], size=sample, replace=False)
        X = X[idx]
        idx_map = idx
        log.info(f"Subsampled: {X.shape[0]} rows")
    else:
        idx_map = np.arange(X.shape[0])

    # Load labels if provided and align with subsample
    labels_df = None
    if labels_csv is not None:
        labels_df = pd.read_csv(labels_csv)
        if len(labels_df) != N:
            raise ValueError(f"labels_csv rows ({len(labels_df)}) != N ({N})")
        labels_df = labels_df.iloc[idx_map].reset_index(drop=True)

    # UMAP
    log.info("[bold]- Running UMAP[/bold]")
    umap = UMAP(
        n_components=n_components,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric=metric,
        densmap=densmap,
        random_state=seed,
    )
    coords = umap.fit_transform(X)
    log.info(f"UMAP coords: {coords.shape}")

    # Optional KMeans
    km_labels = None
    km_info = {}
    if kmeans and kmeans > 1:
        log.info(f"[bold]- KMeans (k={kmeans})[/bold]")
        km = KMeans(n_clusters=kmeans, n_init="auto", random_state=seed)
        km_labels = km.fit_predict(coords)
        km_info = {
            "inertia": float(km.inertia_),
            "centroids": km.cluster_centers_.tolist(),
        }
        if silhouette and coords.shape[0] > kmeans:
            try:
                sil = silhouette_score(coords, km_labels)
                km_info["silhouette"] = float(sil)
                log.info(f"Silhouette score: {sil:.4f}")
            except Exception as e:
                log.warning(f"Silhouette failed: {e}")

    # Save CSV (coords + labels + kmeans)
    out_csv = outdir / "umap_coords.csv"
    df_labels = labels_df.copy() if labels_df is not None else None
    if km_labels is not None:
        if df_labels is None:
            df_labels = pd.DataFrame()
        df_labels["kmeans"] = km_labels
    _save_csv_coords(coords, df_labels, out_csv, label_col, color_by)
    log.info(f"Saved: {out_csv}")

    # Save JSON meta
    meta = {
        "N": int(coords.shape[0]),
        "d_fused": int(X.shape[1]),
        "blocks": [{"path": str(p), "shape": list(b.shape)} for p, b in zip(latents, blocks)],
        "umap": {
            "n_components": n_components,
            "n_neighbors": neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "densmap": densmap,
            "random_state": seed,
        },
        "kmeans": km_info,
        "label_col": label_col,
        "color_by": color_by,
        "subsample": int(sample) if sample else 0,
    }
    with open(outdir / "umap_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Plot
    out_png = outdir / "umap.png"
    _scatter_plot(
        coords=coords[:, :2],
        labels_df=df_labels,
        out_png=out_png,
        title=title,
        label_col=label_col,
        color_by=color_by,
        alpha=alpha,
        point_size=point_size,
    )
    log.info(f"Saved: {out_png}")
    if save_svg:
        out_svg = outdir / "umap.svg"
        # Re-render in SVG to keep vector quality
        _scatter_plot(
            coords=coords[:, :2],
            labels_df=df_labels,
            out_png=out_svg,
            title=title,
            label_col=label_col,
            color_by=color_by,
            alpha=alpha,
            point_size=point_size,
        )
        log.info(f"Saved: {out_svg}")

    # Small terminal table
    if console is not None:
        tbl = Table(title="UMAP Fusion Latents — Summary")
        tbl.add_column("Key", style="bold cyan")
        tbl.add_column("Value")
        tbl.add_row("Rows", str(coords.shape[0]))
        tbl.add_row("Fused Dim", str(X.shape[1]))
        tbl.add_row("Blocks", ", ".join([f"{p.name}({b.shape[1]})" for p, b in zip(latents, blocks)]))
        if label_col:
            tbl.add_row("Label Col", label_col)
        if color_by:
            tbl.add_row("Color By", color_by)
        tbl.add_row("Metric", metric)
        tbl.add_row("Neighbors", str(neighbors))
        tbl.add_row("Min Dist", str(min_dist))
        if kmeans and km_info:
            tbl.add_row("KMeans k", str(kmeans))
            if "silhouette" in km_info:
                tbl.add_row("Silhouette", f"{km_info['silhouette']:.4f}")
        console.print(tbl)


@app.callback()
def _callback():
    """SpectraMind V50 — UMAP fusion latent plotting tool."""
    pass


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
