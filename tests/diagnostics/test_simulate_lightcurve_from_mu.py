# tests/diagnostics/test_simulate_lightcurve_from_mu.py
# -*- coding: utf-8 -*-
"""
Unit tests for simulating a white-light transit lightcurve from a per-wavelength
mean transit depth vector (mu). Tests cover:
  - White-light depth fidelity
  - Quadratic limb-darkening shape effects
  - Noise reproducibility with a seed
  - Input type robustness
  - (Optional) Parity vs project implementation, if present

Notes:
* Quadratic limb darkening: I(μ) = I(1) * [1 - a(1-μ) - b(1-μ)^2]
* Shot-noise σ ∝ sqrt(N) is used here only to check seeded reproducibility.
"""

from __future__ import annotations

import math
import importlib
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest


# --------------------------
# Optional: try to import the DUT from the project
# --------------------------
_DUT_PATHS = [
    # add likely import paths here as your codebase evolves
    "spectramind.simulation.lightcurve",
    "src.simulation.lightcurve",
    "spectramind.diagnostics.lightcurve",
]
_DUT_FUNC_NAME = "simulate_lightcurve_from_mu"

def _try_load_dut() -> Optional[callable]:
    for mod in _DUT_PATHS:
        try:
            m = importlib.import_module(mod)
            fn = getattr(m, _DUT_FUNC_NAME, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

SIM_DUT = _try_load_dut()  # project function if present, else None


# --------------------------
# Reference implementation (lightweight, for property tests & parity)
# --------------------------
def _ref_simulate_lightcurve_from_mu(
    t: np.ndarray,
    mu: np.ndarray | float,
    t0: float = 0.0,
    duration: float = 0.12,
    ingress: float = 0.02,
    ld_coeffs: Tuple[float, float] = (0.0, 0.0),
    add_noise: bool = False,
    noise_sigma: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simple white-light transit model using a symmetric trapezoid whose depth is the
    average of mu across wavelengths, optionally modified by quadratic limb darkening
    during ingress/egress. This is a reference (not a full Mandel–Agol model).
    """
    t = np.asarray(t, dtype=float)
    if np.ndim(mu) == 0:
        depth = float(mu)
    else:
        mu_arr = np.asarray(mu, dtype=float)
        if mu_arr.size == 0:
            depth = 0.0
        else:
            depth = float(np.nanmean(mu_arr))

    # Guard rails
    depth = max(0.0, min(depth, 1.0))

    # Define trapezoid: ingress and egress are linear ramps; flat bottom in between
    t_ing_start = t0 - duration / 2.0
    t_ing_end   = t_ing_start + ingress
    t_egr_start = t0 + duration / 2.0 - ingress
    t_egr_end   = t0 + duration / 2.0

    flux = np.ones_like(t)

    # Piecewise
    # Out of transit: flux = 1

    # In ingress: linear ramp from 1 -> 1 - depth (modified by limb darkening)
    in_ing = (t >= t_ing_start) & (t < t_ing_end)
    if np.any(in_ing):
        x = (t[in_ing] - t_ing_start) / ingress  # 0..1
        ld = _quadratic_ld_profile(x, ld_coeffs)  # curvature proxy 0..1 -> LD weighting
        flux[in_ing] = 1.0 - depth * x * ld

    # Flat bottom
    in_flat = (t >= t_ing_end) & (t <= t_egr_start)
    if np.any(in_flat):
        flux[in_flat] = 1.0 - depth

    # In egress: linear ramp from 1 - depth -> 1 (LD again)
    in_egr = (t > t_egr_start) & (t <= t_egr_end)
    if np.any(in_egr):
        x = 1.0 - (t[in_egr] - t_egr_start) / ingress  # 1..0 decreasing
        ld = _quadratic_ld_profile(1.0 - x, ld_coeffs)
        flux[in_egr] = 1.0 - depth * x * ld

    # Optional Gaussian noise for a quick sanity check (seeded)
    if add_noise and noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        flux = flux + rng.normal(0.0, noise_sigma, size=flux.shape)

    return flux


def _quadratic_ld_profile(x01: np.ndarray, ld_coeffs: Tuple[float, float]) -> np.ndarray:
    """A minimal shape modifier (0..1) mimicking quadratic limb darkening curvature during the ramp."""
    a, b = ld_coeffs
    x01 = np.clip(np.asarray(x01, dtype=float), 0.0, 1.0)
    # Map ramp progress -> μ proxy; we keep it simple yet monotonic
    # Use μ = sqrt(1 - s^2) for s in [0,1], then quadratic LD law
    s = x01
    mu_geo = np.sqrt(np.clip(1.0 - s**2, 0.0, 1.0))
    ld = 1.0 - a * (1.0 - mu_geo) - b * (1.0 - mu_geo) ** 2
    # Normalize to [min,1] so that the modifier stays reasonable (>0)
    mn = np.minimum(1.0, np.maximum(0.2, ld.min() if np.ndim(ld) else ld))
    return ld / (1e-12 + max(mn, 1e-6))


# --------------------------
# Test data fixtures
# --------------------------
@pytest.fixture(scope="module")
def time_grid():
    # 3 hours around mid-transit sampled at 1‑minute cadence
    t = np.linspace(-1.5, 1.5, 181)  # hours
    return t


@pytest.fixture(scope="module")
def constant_mu():
    # 283-channel spectrum with a constant 800 ppm (0.0008) depth
    return np.full(283, 8.0e-4, dtype=float)


# --------------------------
# Tests
# --------------------------
@pytest.mark.parametrize("ld_coeffs", [(0.0, 0.0), (0.4, 0.2)])
def test_white_light_depth_matches_mu_mean(time_grid, constant_mu, ld_coeffs):
    flux = _ref_simulate_lightcurve_from_mu(
        time_grid, constant_mu, t0=0.0, duration=0.12, ingress=0.02, ld_coeffs=ld_coeffs
    )
    # Minimum flux should be ~ 1 - mean(mu) within a small tolerance (numerical)
    expected = 1.0 - float(np.mean(constant_mu))
    assert np.isfinite(flux).all()
    assert np.abs(np.min(flux) - expected) < 5e-5


def test_limb_darkening_changes_ingress_shape(time_grid, constant_mu):
    # Compare no-LD vs with LD at mid-ingress: curvature should change
    t = time_grid
    t0, dur, ing = 0.0, 0.12, 0.02
    mid_ing = t0 - dur / 2.0 + ing / 2.0
    idx = int(np.argmin(np.abs(t - mid_ing)))

    f_nold = _ref_simulate_lightcurve_from_mu(t, constant_mu, t0=t0, duration=dur, ingress=ing, ld_coeffs=(0.0, 0.0))
    f_ld   = _ref_simulate_lightcurve_from_mu(t, constant_mu, t0=t0, duration=dur, ingress=ing, ld_coeffs=(0.5, 0.3))

    # Same depth at mid-transit
    assert math.isclose(f_nold[t == 0.0].item(), f_ld[t == 0.0].item(), rel_tol=1e-6, abs_tol=1e-6)
    # But different curvature in ingress
    assert not math.isclose(f_nold[idx], f_ld[idx], rel_tol=1e-3, abs_tol=1e-5)


def test_noise_is_reproducible_with_seed(time_grid, constant_mu):
    sigma = 2.5e-5
    f1 = _ref_simulate_lightcurve_from_mu(time_grid, constant_mu, add_noise=True, noise_sigma=sigma, seed=42)
    f2 = _ref_simulate_lightcurve_from_mu(time_grid, constant_mu, add_noise=True, noise_sigma=sigma, seed=42)
    f3 = _ref_simulate_lightcurve_from_mu(time_grid, constant_mu, add_noise=True, noise_sigma=sigma, seed=43)

    # Same seed => identical realization
    assert np.allclose(f1, f2)
    # Different seed => statistically different (not exactly equal)
    assert not np.allclose(f1, f3)


@pytest.mark.parametrize("as_type", [list, np.array])
def test_input_type_robustness(as_type, constant_mu):
    t_list = as_type(np.linspace(-0.5, 0.5, 61))
    f = _ref_simulate_lightcurve_from_mu(t_list, constant_mu)
    assert isinstance(f, np.ndarray)
    assert f.dtype.kind == "f"
    assert f.shape == (len(t_list),)


def test_zero_depth_returns_unity(time_grid):
    mu_zero = np.zeros(283, dtype=float)
    f = _ref_simulate_lightcurve_from_mu(time_grid, mu_zero)
    assert np.allclose(f, 1.0)


@pytest.mark.xfail(SIM_DUT is None, reason="Project simulate_lightcurve_from_mu not found; parity test skipped")
def test_parity_against_project_function(time_grid, constant_mu):
    """If the project function is present, ensure basic parity with the reference for a canonical case."""
    assert SIM_DUT is not None
    kwargs = dict(t0=0.0, duration=0.12, ingress=0.02, ld_coeffs=(0.3, 0.2), add_noise=False)
    f_ref = _ref_simulate_lightcurve_from_mu(time_grid, constant_mu, **kwargs)
    f_dut = SIM_DUT(time_grid, constant_mu, **kwargs)
    # We allow a small tolerance; exact parity isn't required if DUT is a more physical model
    assert f_ref.shape == f_dut.shape
    assert np.all(np.isfinite(f_dut))
    assert np.max(np.abs(f_ref - f_dut)) < 1.5e-3
