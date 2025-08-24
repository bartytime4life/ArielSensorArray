# tests/diagnostics/test_plot_umap_fusion_latents_v50.py
"""
SpectraMind V50 — Diagnostics: UMAP embeddings of fused latents

This test suite validates that our diagnostics layer can:
  1) Render a 2D embedding (UMAP preferred, PCA/TSNE fallback) of fused latent vectors
     in headless mode and persist it as a raster image (PNG).
  2) Operate deterministically with a fixed RNG seed to align with our
     reproducibility standards (Hydra/seed discipline).
  3) Gracefully degrade (fallback) if optional UMAP is unavailable.

Notes
-----
• The SpectraMind V50 plan explicitly calls for generating and saving headless
  plots (e.g., spectral plots and UMAP embeddings of latent features) as part
  of the `diagnose` flow.

• If the project exposes a CLI path for this diagnostic, prefer exercising it.
  Otherwise, a direct functional entrypoint is resolved dynamically.

• To keep the tests portable across developer machines and CI, matplotlib uses
  the 'Agg' backend (no X server required).
"""

from __future__ import annotations

import importlib
import io
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest

# Force headless rendering for CI / containerized runs
import matplotlib

matplotlib.use("Agg")


# ---------- Helpers ----------

def _resolve_plotter() -> Optional[Callable]:
    """
    Try to locate the diagnostic plotting entrypoint across a few
    conventional module paths. Return the callable if found; else None.

    Expected signature:
        plot_umap_fusion_latents(
            latents: np.ndarray,           # (N, D)
            labels: Optional[np.ndarray],  # (N,) cluster or class labels (int/str)
            out_path: str | os.PathLike,
            method: str = "umap",          # "umap" | "tsne" | "pca"
            random_state: int = 42,
            n_neighbors: int = 15,
            min_dist: float = 0.1,
            title: Optional[str] = None,
        ) -> Path | str
    """
    candidates = [
        # prefer fully-qualified "spectramind" namespace if present
        ("spectramind.diagnostics.umap_fusion", "plot_umap_fusion_latents"),
        ("spectramind.diagnostics.plotting", "plot_umap_fusion_latents"),
        ("spectramind.diagnostics", "plot_umap_fusion_latents"),
        # common "src" layouts
        ("src.diagnostics.umap_fusion", "plot_umap_fusion_latents"),
        ("src.diagnostics.plotting", "plot_umap_fusion_latents"),
        ("diagnostics.umap_fusion", "plot_umap_fusion_latents"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except ModuleNotFoundError:
            continue
    return None


def _synthetic_latents(n: int = 240, d: int = 64, k: int = 4, seed: int = 7):
    """
    Create a simple, reproducible synthetic fused-latent matrix with k Gaussian
    clusters in d dimensions. Returns (X, labels).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=3.0, size=(k, d))
    X = []
    y = []
    per = n // k
    for i in range(k):
        Xi = centers[i] + rng.normal(scale=0.7, size=(per, d))
        X.append(Xi)
        y.extend([i] * per)
    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y)
    return X, y


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


# ---------- Tests ----------

@pytest.mark.parametrize("method", ["umap", "pca"])  # pca ensures fallback path works everywhere
def test_umap_fusion_latents_produces_png(tmp_path: Path, method: str):
    """
    GIVEN fused latent vectors and cluster labels
    WHEN we request a 2D embedding diagnostic
    THEN a non-empty PNG is produced deterministically at the requested path.

    This hits the happy path for UMAP (if installed) and the universal PCA fallback.
    """
    plotter = _resolve_plotter()
    if plotter is None:
        pytest.skip("plot_umap_fusion_latents entrypoint not found in repository.")

    # Reproducible synthetic data
    X, y = _synthetic_latents(n=240, d=64, k=4, seed=13)

    out_png = tmp_path / f"fusion_latents_{method}.png"
    returned = plotter(
        latents=X,
        labels=y,
        out_path=out_png,
        method=method,
        random_state=42,   # enforce deterministic embedding
        n_neighbors=10,    # small but valid
        min_dist=0.05,
        title=f"Fusion Latents ({method.upper()})",
    )
    # The function can return the path or nothing; in either case, verify on disk.
    assert out_png.exists(), f"Expected PNG at {out_png}, but file not found (returned={returned})"
    size = _file_size(out_png)
    assert size > 2_048, f"PNG seems suspiciously small ({size} bytes)."

    # Quick sanity: it's a PNG signature
    with open(out_png, "rb") as fh:
        sig = fh.read(8)
    assert sig == b"\x89PNG\r\n\x1a\n", "Output does not start with PNG signature."


def test_fallback_when_umap_missing(monkeypatch, tmp_path: Path):
    """
    GIVEN UMAP is unavailable at runtime
    WHEN the plotting function is invoked with method='umap'
    THEN it should fall back (e.g., to PCA/TSNE) and still produce a valid PNG.

    This validates graceful degradation rather than hard failure when optional deps
    (umap-learn) are not installed on the runner.
    """
    plotter = _resolve_plotter()
    if plotter is None:
        pytest.skip("plot_umap_fusion_latents entrypoint not found in repository.")

    # Simulate umap import failure inside the plotted function's module by
    # temporarily removing 'umap' from sys.modules and blocking import.
    real_import = importlib.import_module

    def _blocked(name, *args, **kwargs):
        if name.startswith("umap"):
            raise ModuleNotFoundError("Simulated umap absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _blocked)

    X, y = _synthetic_latents(n=160, d=32, k=3, seed=23)
    out_png = tmp_path / "fusion_latents_umap_fallback.png"
    returned = plotter(
        latents=X,
        labels=y,
        out_path=out_png,
        method="umap",     # request UMAP explicitly; implementation should catch and fallback
        random_state=123,
        n_neighbors=8,
        min_dist=0.05,
        title="Fusion Latents (UMAP fallback)",
    )

    assert out_png.exists(), f"Expected PNG at {out_png}, but file not found (returned={returned})"
    assert _file_size(out_png) > 2_048
    with open(out_png, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_determinism_with_seed(tmp_path: Path):
    """
    Embedding scatter should be deterministic under a fixed random_state.

    We don't compare pixel-by-pixel (renderers can differ), but we do assert
    that two invocations with same seed result in identical file bytes when
    all else (latents, labels, figure size, DPI) is held constant.
    """
    plotter = _resolve_plotter()
    if plotter is None:
        pytest.skip("plot_umap_fusion_latents entrypoint not found in repository.")

    X, y = _synthetic_latents(n=180, d=48, k=3, seed=99)

    out1 = tmp_path / "emb1.png"
    out2 = tmp_path / "emb2.png"

    args = dict(
        latents=X,
        labels=y,
        method="pca",        # universal fallback ensures availability
        random_state=2025,   # fixed seed (V50 reproducibility)
        n_neighbors=12,
        min_dist=0.10,
        title=None,
    )
    plotter(out_path=out1, **args)
    plotter(out_path=out2, **args)

    # On identical RNG and rendering settings, PNGs should match byte-for-byte.
    b1 = Path(out1).read_bytes()
    b2 = Path(out2).read_bytes()
    assert b1 == b2, "Embeddings are not deterministic under fixed seed and identical inputs."
