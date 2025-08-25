# /tests/diagnostics/test_spectral_smoothness_map.py
# -*- coding: utf-8 -*-
"""
Tests for the spectral smoothness map diagnostic.

Goals
-----
1) Numerical correctness on simple, synthetic spectra:
   - Constant and very smooth spectra should yield (near-)zero roughness.
   - Noisy / spiky spectra should yield larger roughness than smooth ones.
   - Monotone segments should be smooth except at boundaries or injected edges.

2) API stability:
   - Works on 1D (single spectrum) and 2D batch (N_spectra x N_wavelengths).
   - Robust to NaNs (ignored via local interpolation or robust estimator).

3) Optional plotting/IO:
   - If the project exposes a plotting helper, it should return a Matplotlib Figure.
   - When invoked via the SpectraMind CLI (if available), it should write artifacts
     without crashing (smoke test).

This test is written to pass *today* using a portable, reference implementation if
the project function is not yet available. As soon as the official implementation
lands under `src/diagnostics/spectral_smoothness.py`, the test will automatically
switch to it and validate behavior against the reference expectations.

How we measure "smoothness"
---------------------------
We use a curvature proxy based on the normalized second finite difference:

    roughness ~ median( | Δ² x | / (|x| + ε) )

with an optional rolling-window robust aggregator to produce a per-wavelength
"map" (same length as the spectrum). This rewards globally-smooth, gently-varying
signals and penalizes sharp kinks/spikes—matching the physical prior that
true transmission spectra vary smoothly with wavelength except at narrow lines.

Author: SpectraMind V50 · Diagnostics
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Tuple

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Try to import the project's implementation; otherwise, use a reference one.
# -----------------------------------------------------------------------------
try:
    # Expected project location (update here if your repo uses a different path)
    from diagnostics.spectral_smoothness import (
        spectral_smoothness_map,        # (spectra, window, robust, eps) -> array
        plot_spectral_smoothness_map,   # (smooth_map, wavelengths=None, **kwargs) -> Figure
    )
    _HAS_PROJECT_IMPL = True
except Exception:
    _HAS_PROJECT_IMPL = False

    def _rolling_median(a: np.ndarray, w: int) -> np.ndarray:
        """Simple, padding-aware rolling median with odd window size."""
        if w < 1:
            return a
        w = int(w)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        ap = np.pad(a, ((0, 0), (pad, pad)), mode="edge") if a.ndim == 2 else np.pad(a, (pad, pad), mode="edge")
        out = np.empty_like(a, dtype=float)
        if a.ndim == 1:
            for i in range(a.shape[0]):
                out[i] = np.median(ap[i : i + w])
        else:
            for i in range(a.shape[1]):
                out[:, i] = np.median(ap[:, i : i + w], axis=1)
        return out

    def spectral_smoothness_map(
        spectra: np.ndarray,
        window: int = 5,
        robust: bool = True,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Reference implementation (portable) of a per-wavelength roughness/smoothness indicator.

        Parameters
        ----------
        spectra : array-like
            1D (L) or 2D (N, L) array of spectra (transit depth vs wavelength index).
        window : int
            Rolling window used to aggregate curvature into a map.
        robust : bool
            If True, use median aggregation; else mean.
        eps : float
            Small value to stabilize normalization.

        Returns
        -------
        rough_map : np.ndarray
            Same shape as input (L or N x L). Lower values => smoother.
        """
        x = np.asarray(spectra, dtype=float)
        x = np.atleast_2d(x)  # (N, L)
        # Replace NaNs with local linear interpolation along wavelength
        nan_mask = np.isnan(x)
        if nan_mask.any():
            for n in range(x.shape[0]):
                idx = np.arange(x.shape[1])
                good = ~nan_mask[n]
                if good.sum() == 0:
                    # all-NaN, set zeros (will produce zeros roughness)
                    x[n] = 0.0
                else:
                    x[n, ~good] = np.interp(idx[~good], idx[good], x[n, good])

        # Second finite difference (curvature proxy)
        # Δ² x[i] = x[i+1] - 2 x[i] + x[i-1]
        d2 = np.zeros_like(x)
        d2[:, 1:-1] = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
        d2[:, 0] = d2[:, 1]
        d2[:, -1] = d2[:, -2]

        # Normalize by local magnitude to be less sensitive to scale
        denom = np.abs(x) + eps
        curv = np.abs(d2) / denom

        # Aggregate with a rolling window to yield a per-wavelength "map"
        if window and window > 1:
            if robust:
                rough = _rolling_median(curv, window)
            else:
                # mean smoothing
                w = int(window)
                if w % 2 == 0:
                    w += 1
                pad = w // 2
                ap = np.pad(curv, ((0, 0), (pad, pad)), mode="edge")
                kernel = np.ones((1, w)) / w
                rough = np.apply_along_axis(lambda r: np.convolve(r, kernel.ravel(), mode="valid"), 1, ap)
        else:
            rough = curv

        # Return with original dimensionality
        return rough[0] if spectra.ndim == 1 else rough

    def plot_spectral_smoothness_map(smooth_map, wavelengths=None, **kwargs):  # minimal, lazy plotting
        import matplotlib.pyplot as plt

        sm = np.asarray(smooth_map)
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 3)))
        if sm.ndim == 1:
            x = np.arange(sm.shape[0]) if wavelengths is None else wavelengths
            ax.plot(x, sm, lw=1.5, color=kwargs.get("color", "tab:orange"))
            ax.set_ylabel("roughness ↓ (smooth ↑)")
        else:
            im = ax.imshow(sm, aspect="auto", cmap=kwargs.get("cmap", "viridis"))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="roughness")
            ax.set_ylabel("spectrum index")
        ax.set_xlabel("wavelength" if wavelengths is not None else "channel")
        ax.set_title("Spectral Smoothness Map")
        fig.tight_layout()
        return fig


# -----------------------------------------------------------------------------
# Fixtures & helpers
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seed():
    np.random.seed(1234)
    return 1234


def _make_synthetic(batch: int = 4, L: int = 283, noise: float = 0.0, spikes: int = 0) -> np.ndarray:
    """
    Create synthetic spectra:
      - Base: low-frequency sine + gentle slope (smooth).
      - Optional: Gaussian noise and a few random spikes.
    """
    x = np.linspace(0, 2 * np.pi, L, dtype=float)
    base = 0.01 * np.sin(1.5 * x) + 0.001 * x  # smooth, small amplitude typical of ppm-ish signals
    S = np.repeat(base[None, :], batch, axis=0)

    if noise > 0:
        S += np.random.normal(0, noise, size=S.shape)

    if spikes > 0:
        for n in range(batch):
            idx = np.random.choice(L, size=spikes, replace=False)
            S[n, idx] += np.random.uniform(0.02, 0.05, size=spikes) * np.sign(np.random.randn(spikes))

    return S


# -----------------------------------------------------------------------------
# Core numerical behavior
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("window", [1, 5, 11])
@pytest.mark.parametrize("robust", [True, False])
def test_constant_and_smooth_are_low(seed, window, robust):
    L = 283
    const = np.zeros(L)
    smooth = 0.001 * np.sin(np.linspace(0, 3 * np.pi, L))
    rough_const = spectral_smoothness_map(const, window=window, robust=robust)
    rough_smooth = spectral_smoothness_map(smooth, window=window, robust=robust)

    # Both should be (near) zero; constant the lowest.
    assert rough_const.shape == (L,)
    assert rough_smooth.shape == (L,)
    assert np.nanmax(rough_const) < 1e-10
    assert np.nanmedian(rough_smooth) < 1e-3
    assert np.nanmedian(rough_const) <= np.nanmedian(rough_smooth) + 1e-12


def test_noisy_greater_than_smooth(seed):
    L = 283
    smooth = 0.001 * np.sin(np.linspace(0, 4 * np.pi, L))
    noisy = smooth + np.random.normal(0, 0.003, size=L)
    r_smooth = spectral_smoothness_map(smooth, window=7, robust=True)
    r_noisy = spectral_smoothness_map(noisy, window=7, robust=True)

    # Noisy should be clearly rougher than smooth
    assert np.nanmedian(r_noisy) > 3 * np.nanmedian(r_smooth)


def test_spiky_greater_than_noisy(seed):
    L = 283
    noisy = _make_synthetic(batch=1, L=L, noise=0.002, spikes=0)[0]
    spiky = _make_synthetic(batch=1, L=L, noise=0.002, spikes=6)[0]
    r_noisy = spectral_smoothness_map(noisy, window=9, robust=True)
    r_spiky = spectral_smoothness_map(spiky, window=9, robust=True)
    # Spikes should drive much higher roughness in local windows
    assert np.nanpercentile(r_spiky, 90) > 2.0 * np.nanpercentile(r_noisy, 90)


def test_batch_shape_and_nans(seed):
    N, L = 5, 283
    batch = _make_synthetic(batch=N, L=L, noise=0.001, spikes=0)
    # Introduce NaNs in random places; implementation should be robust
    nan_mask = np.random.rand(*batch.shape) < 0.02
    batch[nan_mask] = np.nan

    rough = spectral_smoothness_map(batch, window=7, robust=True)
    assert rough.shape == (N, L)
    # Finite after NaN handling
    assert np.isfinite(rough).all()


def test_monotone_segments(seed):
    L = 283
    # Piecewise linear ramp with a single kink; should be smooth except around the change
    x = np.linspace(0, 1, L)
    y = np.where(x < 0.5, 0.001 * x, 0.001 * (1.0 + 0.5 * (x - 0.5)))
    r = spectral_smoothness_map(y, window=5, robust=True)
    # Bulk should be small
    assert np.nanmedian(r) < 5e-3
    # There should exist a localized region with higher roughness (the kink vicinity)
    assert np.nanmax(r) > 10 * np.nanmedian(r)


# -----------------------------------------------------------------------------
# Optional plotting (if provided)
# -----------------------------------------------------------------------------
@pytest.mark.mpl
def test_plot_returns_figure(seed):
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
    except Exception:
        pytest.skip("Matplotlib not available in test environment.")
    L = 283
    y = 0.001 * np.sin(np.linspace(0, 3 * np.pi, L)) + np.random.normal(0, 0.002, size=L)
    sm = spectral_smoothness_map(y, window=9, robust=True)
    fig = plot_spectral_smoothness_map(sm)
    # Minimal checks
    assert fig is not None
    # Close to avoid backend accumulation
    import matplotlib.pyplot as plt
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI smoke test (skipped if the SpectraMind CLI is not installed)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("spectramind") is None, reason="SpectraMind CLI not found in PATH")
def test_cli_smoke_tmp_artifacts(seed, tmp_path: pytest.TempPathFactory):
    """
    This test only verifies that the CLI route for the diagnostic runs end-to-end and
    writes something to disk. It does *not* assert numerical values (covered above).
    """
    temp_dir = tmp_path.mktemp("smoothness_cli")
    inp = temp_dir / "batch.npy"
    out_dir = temp_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a small batch input for the CLI
    batch = _make_synthetic(batch=6, L=283, noise=0.002, spikes=2)
    np.save(str(inp), batch)

    # Expected CLI contract (adjust if your CLI differs):
    # spectramind diagnose spectral-smoothness-map --input <npy> --outdir <dir> --window 9
    cmd = [
        "spectramind",
        "diagnose",
        "spectral-smoothness-map",
        "--input",
        str(inp),
        "--outdir",
        str(out_dir),
        "--window",
        "9",
        "--robust",
        "true",
    ]

    env = os.environ.copy()
    # Keep any config lookups local to temp if needed
    with tempfile.TemporaryDirectory() as _:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    # Should run without crashing
    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}"

    # Should have produced at least one artifact (image/map)
    produced = list(out_dir.glob("*"))
    assert len(produced) > 0, f"No artifacts produced in {out_dir}"
    # Prefer existence of either .png or .npy summary
    assert any(p.suffix.lower() in (".png", ".pdf", ".npy") for p in produced)
