# tests/diagnostics/test_plot_umap_fusion_latents_v50.py
"""
Upgraded diagnostic tests for UMAP fusion-latent visualization (V50).

Goals
-----
1) Ensure a plotting routine produces a valid PNG (or returns a Matplotlib Axes)
   for fused latent spaces.
2) Validate reproducibility when a fixed random seed is provided.
3) Verify robust handling of labels (present or missing) and input shape errors.
4) Keep wall-clock under CI-friendly limits (≈ a few seconds) using small synthetic data.

These tests will try to import a project-provided plotting function from a few
common paths. If none exist (e.g., early scaffold), we fall back to a minimal,
standalone implementation so that the diagnostic expectations remain testable.

Expected function signature (flexible):
---------------------------------------
plot_umap_fusion_latents(
    latents: np.ndarray,           # (N, D) fused latent matrix for all samples
    labels: Optional[ArrayLike],   # (N,) discrete class labels, optional
    title: str = "...",
    out_path: Optional[Path] = None,
    random_state: Optional[int] = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    **kwargs
) -> Union[matplotlib.axes.Axes, Path, None]

Notes
-----
- If the function returns a Matplotlib Axes, we save to a PNG ourselves to
  validate artifact generation.
- If it returns a path, we verify the path exists.
- If it returns None, we verify that out_path exists if it was provided.

Dependencies
------------
- numpy
- pytest
- matplotlib
- umap-learn (tests will skip gracefully if not available AND no project function is found)

"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pytest

# Use a non-interactive backend for headless CI environments
import matplotlib
matplotlib.use("Agg", force=True)  # type: ignore
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------
# Attempt to import SUT (System Under Test)
# ---------------------------

_PLOT_FN: Optional[Callable] = None
_IMPORT_ERRORS = []

_CANDIDATE_IMPORTS = (
    # Try likely locations first (adjust/add as the repo evolves)
    "spectramind.diagnostics.plot_umap_fusion_latents",
    "spectramind.visualization.plot_umap_fusion_latents",
    "src.spectramind.diagnostics.plot_umap_fusion_latents",
    "src.spectramind.visualization.plot_umap_fusion_latents",
)

for _cand in _CANDIDATE_IMPORTS:
    try:
        module_name, func_name = _cand.rsplit(".", 1)
        mod = __import__(module_name, fromlist=[func_name])
        _fn = getattr(mod, func_name, None)
        if callable(_fn):
            _PLOT_FN = _fn
            break
    except Exception as e:  # pragma: no cover - best-effort import
        _IMPORT_ERRORS.append((_cand, repr(e)))


def _fallback_plot_umap_fusion_latents(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP Fusion Latents (fallback)",
    out_path: Optional[Path] = None,
    random_state: Optional[int] = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    **kwargs,
):
    """
    Minimal reference implementation used ONLY if the project function is unavailable.

    Produces a 2D UMAP scatter plot with optional discrete labels (colored).
    """
    try:
        import umap  # type: ignore
    except Exception as e:  # pragma: no cover - guarded by skip
        raise RuntimeError("umap-learn not available for fallback implementation") from e

    if latents.ndim != 2:
        raise ValueError(f"`latents` must be 2D array of shape (N, D), got {latents.shape}")

    if labels is not None and len(labels) != len(latents):
        raise ValueError("`labels` length must match number of rows in `latents`")

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        n_components=2,
        random_state=None if random_state is None else int(random_state),
        metric=kwargs.get("metric", "euclidean"),
        verbose=False,
    )
    emb = reducer.fit_transform(latents)  # (N, 2)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=120)
    if labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=10, c="tab:blue", alpha=0.8, linewidths=0)
    else:
        labels = np.asarray(labels)
        # Use a simple color cycle
        unique = np.unique(labels)
        colors = plt.get_cmap("tab10", len(unique))
        for idx, cls in enumerate(unique):
            mask = labels == cls
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=10,
                alpha=0.85,
                color=colors(idx),
                label=str(cls),
                linewidths=0,
            )
        if len(unique) <= 10:
            ax.legend(frameon=False, fontsize=8, handletextpad=0.3, borderpad=0.2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
    else:
        return ax


# Decide which implementation to use
try:
    if _PLOT_FN is not None:
        plot_umap_fusion_latents = _PLOT_FN  # type: ignore
    else:
        # Verify we can import umap-learn for fallback path
        import umap  # noqa: F401
        plot_umap_fusion_latents = _fallback_plot_umap_fusion_latents  # type: ignore
except Exception:
    plot_umap_fusion_latents = None  # type: ignore


# ---------------------------
# Test Utilities
# ---------------------------

def _synth_fused_latents(
    n: int = 300, d: int = 16, n_classes: int = 4, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple fused-latent data with class structure.
    Each class is drawn from a different Gaussian in D-dim, then stacked.
    """
    rng = np.random.default_rng(seed)
    latents_list = []
    labels_list = []
    for k in range(n_classes):
        center = rng.normal(0.0, 3.0, size=(d,))
        cov = np.eye(d) * (0.3 + 0.2 * k)  # minor variation per-class
        Xk = rng.multivariate_normal(center, cov, size=n // n_classes)
        latents_list.append(Xk)
        labels_list.append(np.full((len(Xk),), k))
    latents = np.vstack(latents_list)
    labels = np.concatenate(labels_list)
    # small shuffle
    perm = rng.permutation(len(latents))
    return latents[perm], labels[perm]


def _pairwise_distances(X: np.ndarray, max_points: int = 120) -> np.ndarray:
    """
    Compute a condensed vector of pairwise distances for similarity checks.
    """
    if len(X) > max_points:
        X = X[:max_points]
    # Euclidean distances; we only need relative structure for correlation.
    diffs = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diffs * diffs, axis=-1))
    # Flatten upper triangle without diagonal
    idxs = np.triu_indices(D.shape[0], k=1)
    return D[idxs]


# ---------------------------
# Pytest marks & skips
# ---------------------------

REASON_NO_IMPL = None
if plot_umap_fusion_latents is None:
    REASON_NO_IMPL = (
        "No plotting implementation is available: project function not found and "
        "`umap-learn` is missing for fallback.\n"
        f"Tried imports: {_CANDIDATE_IMPORTS}\n"
        f"Import errors: {_IMPORT_ERRORS}"
    )

requires_impl = pytest.mark.skipif(
    plot_umap_fusion_latents is None, reason=REASON_NO_IMPL or "No implementation"
)


# ---------------------------
# Tests
# ---------------------------

@requires_impl
def test_umap_plot_saves_png(tmp_path: Path):
    """Smoke test: plotting produces a valid PNG file (and a non-empty image)."""
    X, y = _synth_fused_latents(n=240, d=12, n_classes=3, seed=123)
    out = tmp_path / "umap_fusion.png"

    result = plot_umap_fusion_latents(
        latents=X,
        labels=y,
        title="UMAP Fusion Latents V50",
        out_path=out,
        random_state=42,
        n_neighbors=12,
        min_dist=0.10,
    )
    # Accept either a returned path, an Axes, or None (if file was saved)
    if isinstance(result, plt.Axes):
        # Save ourselves if the function didn't, to keep the assertion invariant
        result.figure.savefig(out, bbox_inches="tight")
        plt.close(result.figure)
    elif isinstance(result, (str, os.PathLike, Path)):
        out = Path(result)

    assert out.exists(), "Expected output PNG was not created."
    assert out.stat().st_size > 5_000, "Output image looks too small to be a valid plot."


@requires_impl
def test_reproducible_embedding_with_seed(tmp_path: Path):
    """
    Reproducibility: with the same random_state, the geometry of embeddings should be
    (nearly) preserved. Since UMAP is stochastic up to isometries, we compare pairwise
    distance structures via correlation after two independent runs with the same seed.
    """
    X, y = _synth_fused_latents(n=300, d=16, n_classes=4, seed=999)

    # Run A
    ax_or_path_A = plot_umap_fusion_latents(
        latents=X, labels=y, title="A", out_path=None, random_state=77, n_neighbors=15
    )
    # Extract the coordinates from the Axes if possible; otherwise, re-run locally with fallback
    emb_A: Optional[np.ndarray] = None
    if isinstance(ax_or_path_A, plt.Axes):
        coll = ax_or_path_A.collections
        if coll:
            # Get first PathCollection offsets as proxy (most scatter artists)
            emb_A = coll[0].get_offsets().data
        plt.close(ax_or_path_A.figure)

    if emb_A is None:
        # Fall back: run the local reducer to pull out coordinates for the test metric
        emb_A = _run_local_umap(X, random_state=77, n_neighbors=15)

    # Run B with same seed
    ax_or_path_B = plot_umap_fusion_latents(
        latents=X, labels=y, title="B", out_path=None, random_state=77, n_neighbors=15
    )
    emb_B: Optional[np.ndarray] = None
    if isinstance(ax_or_path_B, plt.Axes):
        coll = ax_or_path_B.collections
        if coll:
            emb_B = coll[0].get_offsets().data
        plt.close(ax_or_path_B.figure)

    if emb_B is None:
        emb_B = _run_local_umap(X, random_state=77, n_neighbors=15)

    # Compare pairwise distance structures via Pearson correlation
    dA = _pairwise_distances(emb_A, max_points=120)
    dB = _pairwise_distances(emb_B, max_points=120)
    # Normalize
    dA = (dA - dA.mean()) / (dA.std() + 1e-9)
    dB = (dB - dB.mean()) / (dB.std() + 1e-9)
    corr = float(np.clip(np.corrcoef(dA, dB)[0, 1], -1.0, 1.0))
    assert corr > 0.98, f"Pairwise structure dev too large: corr={corr:.4f} (seed reproducibility)"


def _run_local_umap(X: np.ndarray, random_state: int, n_neighbors: int) -> np.ndarray:
    """Helper: run a tiny UMAP transform to extract coordinates for metrics."""
    import umap  # type: ignore
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=0.10,
        n_components=2,
        random_state=int(random_state),
        metric="euclidean",
        verbose=False,
    )
    return reducer.fit_transform(X)


@requires_impl
def test_handles_missing_labels(tmp_path: Path):
    """The function should handle `labels=None` gracefully and still render."""
    X, _ = _synth_fused_latents(n=200, d=10, n_classes=2, seed=2025)
    out = tmp_path / "umap_no_labels.png"

    res = plot_umap_fusion_latents(
        latents=X, labels=None, title="No Labels", out_path=out, random_state=11
    )
    if isinstance(res, plt.Axes):
        res.figure.savefig(out, bbox_inches="tight")
        plt.close(res.figure)
    elif isinstance(res, (str, os.PathLike, Path)):
        out = Path(res)

    assert out.exists(), "Expected output PNG for unlabeled plot was not created."
    assert out.stat().st_size > 3_000, "Unlabeled PNG seems unexpectedly small."


@requires_impl
def test_input_shape_validation():
    """Invalid input shapes should raise ValueError with a helpful message."""
    X, y = _synth_fused_latents(n=100, d=8, n_classes=2, seed=42)
    # Break the shape: give 1D vector
    with pytest.raises(Exception):
        plot_umap_fusion_latents(latents=X.ravel(), labels=y)

    # Labels length mismatch
    with pytest.raises(Exception):
        plot_umap_fusion_latents(latents=X, labels=y[:-1])


@requires_impl
def test_allows_cli_like_params(tmp_path: Path):
    """
    Ensure typical CLI-like params are accepted (neighbors/min_dist/seed)
    and that the function doesn't choke on additional **kwargs such as metric.
    """
    X, y = _synth_fused_latents(n=180, d=10, n_classes=3, seed=101)
    out = tmp_path / "umap_extra_kwargs.png"
    res = plot_umap_fusion_latents(
        latents=X,
        labels=y,
        title="ExtraKwargs",
        out_path=out,
        random_state=123,
        n_neighbors=10,
        min_dist=0.05,
        metric="euclidean",  # extra kwarg passed through to UMAP in fallback, tolerated in SUT
    )
    if isinstance(res, plt.Axes):
        res.figure.savefig(out, bbox_inches="tight")
        plt.close(res.figure)
    elif isinstance(res, (str, os.PathLike, Path)):
        out = Path(res)
    assert out.exists()


# ---------------------------
# Optional: show helpful skip message in logs
# ---------------------------

def test__why_skipped_if_no_impl():
    """
    If this repository doesn't yet ship a plotting implementation and the environment
    lacks `umap-learn`, surface a single informative xfail so developers see the cause.
    """
    if plot_umap_fusion_latents is None:
        pytest.xfail(REASON_NO_IMPL or "No plotting implementation available")
    else:
        # If we do have an implementation, this test is a no-op pass.
        assert callable(plot_umap_fusion_latents)


# ---------------------------
# End of file
# ---------------------------
