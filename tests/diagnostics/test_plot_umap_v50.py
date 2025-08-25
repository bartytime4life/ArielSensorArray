# tests/diagnostics/test_plot_umap_v50.py
"""
SpectraMind V50 — Diagnostics: UMAP plot smoke & fidelity tests.

Goals
-----
- Exercise a tiny, deterministic UMAP projection on synthetic "planet-like" latent data.
- Produce a non-interactive PNG (Agg backend) with clear legend/labels.
- Assert the artifact exists and is non-empty; assert basic numerical stability (no NaNs / infs).
- Run FAST by default; mark SLOW parametrizations separately.
- Skip cleanly if optional deps (umap-learn, matplotlib) are unavailable.

Design notes
------------
- No seaborn. Matplotlib (Agg) only.
- Deterministic seeds across numpy / random / umap (where applicable).
- No Internet / no external files. Everything is generated in-test.
- The test is self-sufficient even if the repo doesn’t yet expose a plotting helper.
"""

from __future__ import annotations

import importlib.util
import io
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# --- Optional deps handling (skip if absent) ---------------------------------
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

pytestmark = pytest.mark.skipif(
    not (_has_module("matplotlib") and _has_module("umap")),
    reason="Requires matplotlib and umap-learn",
)

# Now that we know matplotlib is present, configure the headless backend.
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import umap  # noqa: E402


# --- Utilities ----------------------------------------------------------------
def _set_all_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # umap uses numpy RNG under the hood; passing random_state as well.


def _synthetic_planet_latents(
    n_planets: int = 80,
    latent_dim: int = 32,
    n_groups: int = 4,
    cluster_spread: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make small, structured synthetic "planet latent" data.

    We create n_groups cluster centroids on a hypersphere, then sample points
    around them. Output:
      X: [n_planets, latent_dim] float32
      y: [n_planets] int labels in [0..n_groups-1]
    """
    assert n_planets >= n_groups, "n_planets must be >= n_groups"
    _set_all_seeds(1337)

    # Sample group centroids uniformly on the unit hypersphere
    centroids = np.random.normal(0, 1, size=(n_groups, latent_dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    # Allocate roughly equal membership per group
    base = n_planets // n_groups
    rem = n_planets % n_groups
    counts = [base + (1 if i < rem else 0) for i in range(n_groups)]

    X_list, y_list = [], []
    for gid, n in enumerate(counts):
        # Sample around centroid
        noise = np.random.normal(0, cluster_spread, size=(n, latent_dim))
        pts = centroids[gid : gid + 1] + noise
        X_list.append(pts)
        y_list.extend([gid] * n)

    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)

    # Small shuffle (deterministic)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


def _safe_umap_embed(
    X: np.ndarray,
    n_neighbors: int = 10,
    min_dist: float = 0.05,
    metric: str = "euclidean",
    random_state: int = 2025,
) -> np.ndarray:
    """
    Compute a 2D UMAP embedding; assert no NaNs/Infs and shape correctness.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    emb = reducer.fit_transform(X)
    assert emb.shape == (X.shape[0], 2), "Unexpected UMAP output shape"
    assert np.isfinite(emb).all(), "UMAP produced non-finite values"
    return emb.astype(np.float32)


def _plot_umap_scatter(
    emb: np.ndarray,
    labels: Optional[np.ndarray],
    title: str,
    out_path: Path,
    cmap: str = "tab10",
    point_size: float = 14.0,
) -> None:
    """
    Simple 2D scatter with legend (if labels provided). Saves PNG atomically.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    if labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=point_size, c="#1f77b4", alpha=0.9)
    else:
        # Label-driven colors
        labels = labels.astype(int)
        classes = np.unique(labels)
        for cls in classes:
            mask = labels == cls
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=point_size,
                alpha=0.95,
                label=f"group={cls}",
            )
        ax.legend(loc="best", frameon=True, fontsize=8)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)
    fig.tight_layout()

    # Atomic write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    data = buf.getvalue()
    tmp = out_path.with_suffix(".tmp.png")
    tmp.write_bytes(data)
    tmp.replace(out_path)

    # Sanity checks on file
    assert out_path.exists(), "PNG not written"
    assert out_path.stat().st_size > 1024, "PNG too small; likely empty"


# --- Tests --------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_neighbors, min_dist",
    [
        (10, 0.05),  # fast, default-like
        pytest.param(25, 0.01, marks=pytest.mark.slow),  # a bit denser; marked slow
    ],
)
def test_umap_plot_smoke(tmp_path: Path, n_neighbors: int, min_dist: float) -> None:
    """
    Smoke test: produce a UMAP projection + PNG artifact and verify basic health.
    """
    _set_all_seeds(7)

    # Small, structured synthetic set
    X, y = _synthetic_planet_latents(
        n_planets=90, latent_dim=24, n_groups=3, cluster_spread=0.22
    )

    # Projection
    emb = _safe_umap_embed(
        X, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=99
    )

    # Basic geometry sanity
    # (spread shouldn't degenerate; bounding box area > ~tiny epsilon)
    xspan = float(emb[:, 0].max() - emb[:, 0].min())
    yspan = float(emb[:, 1].max() - emb[:, 1].min())
    area = xspan * yspan
    assert xspan > 1e-4 and yspan > 1e-4, "Embedding collapsed to a line/point"
    assert area > 1e-6, "Degenerate area"

    # Produce artifact
    out_png = tmp_path / f"umap_v50_n{n_neighbors}_d{min_dist:.2f}.png"
    _plot_umap_scatter(emb, y, "UMAP — SpectraMind V50 Latent Snapshot", out_png)


def test_umap_handles_constant_feature(tmp_path: Path) -> None:
    """
    Robustness: UMAP on data that includes a constant feature (common in real pipelines).
    """
    _set_all_seeds(123)
    X, y = _synthetic_planet_latents(n_planets=60, latent_dim=8, n_groups=3, cluster_spread=0.3)

    # Append a constant column and a near-constant column
    const_col = np.ones((X.shape[0], 1), dtype=X.dtype)
    near_const = (np.ones((X.shape[0], 1)) * 0.5 + np.random.randn(X.shape[0], 1) * 1e-5).astype(
        X.dtype
    )
    X_aug = np.concatenate([X, const_col, near_const], axis=1)

    emb = _safe_umap_embed(X_aug, n_neighbors=8, min_dist=0.03, metric="euclidean", random_state=1)
    assert np.isfinite(emb).all()

    out_png = tmp_path / "umap_v50_constant_feature.png"
    _plot_umap_scatter(emb, y, "UMAP — Constant/near-constant Feature Robustness", out_png)


def test_umap_png_is_deterministic(tmp_path: Path) -> None:
    """
    Determinism smoke: with the same seeds and params, the PNG byte-size should match.
    (Strict pixel-wise determinism can vary across platforms/backends; we use a lenience proxy.)
    """
    _set_all_seeds(2025)
    X, y = _synthetic_planet_latents(n_planets=64, latent_dim=16, n_groups=4, cluster_spread=0.2)

    emb = _safe_umap_embed(X, n_neighbors=12, min_dist=0.05, random_state=2025)
    out_png_1 = tmp_path / "umap_det_1.png"
    _plot_umap_scatter(emb, y, "UMAP — Determinism Check #1", out_png_1)

    # Reset seeds and redo
    _set_all_seeds(2025)
    emb2 = _safe_umap_embed(X, n_neighbors=12, min_dist=0.05, random_state=2025)
    out_png_2 = tmp_path / "umap_det_2.png"
    _plot_umap_scatter(emb2, y, "UMAP — Determinism Check #2", out_png_2)

    # Same size is a good proxy for stable output (exact bytes may differ slightly across OS/Matplotlib)
    s1 = out_png_1.stat().st_size
    s2 = out_png_2.stat().st_size
    # Allow a tiny tolerance
    assert abs(int(s1) - int(s2)) <= 128, f"PNG sizes differ too much: {s1} vs {s2}"


def test_umap_handles_small_sample(tmp_path: Path) -> None:
    """
    Edge case: very small n with neighbors adjusted. Confirms graceful behavior.
    """
    _set_all_seeds(9)
    X, y = _synthetic_planet_latents(n_planets=12, latent_dim=10, n_groups=3, cluster_spread=0.35)

    # For tiny sample sizes, n_neighbors must be < n_samples
    emb = _safe_umap_embed(X, n_neighbors=5, min_dist=0.1, random_state=9)
    assert emb.shape[0] == X.shape[0]

    out_png = tmp_path / "umap_small_n.png"
    _plot_umap_scatter(emb, y, "UMAP — Small Sample", out_png)
