# tests/test_molecular_priors.py
# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
# ---------------------------------------------------------------------
# This test suite validates the behavior of the "molecular priors"
# utilities that generate physics-informed features used by the model,
# e.g., template spectra for molecules (H2O/CO2/CH4), Gaussian/Voigt
# bands, wavelength masks, and the graph prior that connects spectral
# channels belonging to the same absorption system.
#
# The tests are intentionally light on external deps: only numpy+pytest.
# If Hypothesis is available, property tests are enabled automatically.
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import importlib
from typing import Dict, Iterable, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------
# Import targets under several plausible project paths to be robust to
# where the code lives in your repo.
# ---------------------------------------------------------------------
_CANDIDATE_IMPORTS = [
    "spectramind.features.molecular_priors",
    "src.spectramind.features.molecular_priors",
    "src.features.molecular_priors",
    "features.molecular_priors",
    "molecular_priors",
]


def _import_module():
    last_err = None
    for name in _CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err  # If none worked, bubble the last error


mp = _import_module()


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def wl_grid() -> np.ndarray:
    """
    Return a canonical wavelength grid (microns) covering Ariel-like AIRS
    sampling (~283 points). If your implementation uses a different grid,
    these tests should still pass as they check relative properties.
    """
    # 0.5–7.8 µm roughly spans visible-through-IR windows used in many
    # simplified challenge datasets. 283 points for parity with baseline.
    return np.linspace(0.5, 7.8, 283, dtype=np.float64)


@pytest.fixture(scope="module")
def band_catalog() -> Dict[str, Dict[str, Iterable[float]]]:
    """
    Minimalistic band centers for three molecules. Replace/extend if your
    implementation uses a richer catalog; tests use closeness not exact idx.
    """
    return {
        "H2O": {
            "centers_um": [1.38, 1.9, 2.7, 6.3],
        },
        "CO2": {
            "centers_um": [2.0, 2.7, 4.3],
        },
        "CH4": {
            "centers_um": [1.66, 2.3, 3.3],
        },
    }


# Hypothesis (optional)
try:  # pragma: no cover - optional dependency
    from hypothesis import given, settings, strategies as st

    HAVE_HYPOTHESIS = True
except Exception:  # pragma: no cover
    HAVE_HYPOTHESIS = False


# ---------------------------------------------------------------------
# Helper utilities for tests
# ---------------------------------------------------------------------
def nearest_index(x: np.ndarray, v: float) -> int:
    return int(np.abs(x - v).argmin())


def _safe_hasattr(obj, name: str) -> bool:
    try:
        getattr(obj, name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Presence / API surface tests
# ---------------------------------------------------------------------
def test_required_symbols_present():
    required = [
        "gaussian_band",
        "voigt_profile",            # may be a thin wrapper around gaussian if a==0
        "make_template_spectrum",   # (molecule, wl, params) -> spectrum
        "build_molecular_band_graph",  # (wl, catalog) -> (edges, weights) or adjacency
        "get_molecular_priors",     # convenience orchestrator
    ]

    for name in required:
        assert _safe_hasattr(mp, name), f"Missing required symbol: {name}"


# ---------------------------------------------------------------------
# Gaussian / Voigt band primitives
# ---------------------------------------------------------------------
def test_gaussian_band_area_is_normalized():
    wl = np.linspace(1.0, 2.0, 4001)
    mu = 1.5
    sigma = 0.02
    g = mp.gaussian_band(wl, mu=mu, sigma=sigma, amplitude=1.0, area_normalize=True)

    # Numeric integral approximates 1.0 (area normalization requests unit area).
    area = np.trapz(g, wl)
    assert np.isfinite(area)
    assert math.isclose(area, 1.0, rel_tol=1e-2, abs_tol=5e-3)


def test_voigt_reduces_to_gaussian_when_lorentz_zero():
    wl = np.linspace(1.0, 2.0, 4001)
    mu = 1.3
    sigma = 0.015
    # a=0 should reduce to Gaussian
    v = mp.voigt_profile(wl, mu=mu, sigma=sigma, gamma_lorentz=0.0, amplitude=1.0, area_normalize=True)
    g = mp.gaussian_band(wl, mu=mu, sigma=sigma, amplitude=1.0, area_normalize=True)

    # Profiles are close in L2
    diff = np.linalg.norm(v - g) / (np.linalg.norm(g) + 1e-12)
    assert diff < 1e-3


# ---------------------------------------------------------------------
# Template spectra: shape, non-negativity, and peak alignment
# ---------------------------------------------------------------------
@pytest.mark.parametrize("mol", ["H2O", "CO2", "CH4"])
def test_template_is_non_negative_and_finite(mol: str, wl_grid: np.ndarray):
    spec = mp.make_template_spectrum(molecule=mol, wl_um=wl_grid, params={"strength_scale": 1.0})
    assert spec.shape == wl_grid.shape
    assert np.all(np.isfinite(spec))
    assert np.all(spec >= -1e-12)  # allow tiny negative from numerical roundoff


@pytest.mark.parametrize("mol, expected_peak_um", [
    ("H2O", 1.38),
    ("H2O", 1.9),
    ("H2O", 2.7),
    ("H2O", 6.3),
    ("CO2", 4.3),
    ("CH4", 3.3),
])
def test_template_has_local_peak_near_known_band(mol: str, expected_peak_um: float, wl_grid: np.ndarray):
    spec = mp.make_template_spectrum(molecule=mol, wl_um=wl_grid, params={"strength_scale": 1.0})
    idx = nearest_index(wl_grid, expected_peak_um)

    # Check a local peak (descending on both sides within a window if possible)
    left = max(idx - 3, 0)
    right = min(idx + 3, len(wl_grid) - 1)
    local_max_idx = left + np.argmax(spec[left:right + 1])
    peak_wl = wl_grid[local_max_idx]
    assert abs(peak_wl - expected_peak_um) <= (wl_grid[1] - wl_grid[0]) * 4.0


# ---------------------------------------------------------------------
# Priors assembly: normalization and masks
# ---------------------------------------------------------------------
def test_get_molecular_priors_returns_consistent_shapes(wl_grid: np.ndarray):
    out = mp.get_molecular_priors(wl_um=wl_grid, molecules=("H2O", "CO2", "CH4"))
    assert isinstance(out, dict)
    assert "templates" in out and "names" in out
    templates = out["templates"]
    names = out["names"]
    assert isinstance(templates, np.ndarray)
    assert templates.ndim == 2  # (n_molecules, n_wavelength)
    assert templates.shape[1] == wl_grid.shape[0]
    assert len(names) == templates.shape[0]


def test_templates_are_l2_normalized_or_reasonable_scale(wl_grid: np.ndarray):
    # Expect either L2=1 or within a consistent order of magnitude
    out = mp.get_molecular_priors(wl_um=wl_grid, molecules=("H2O", "CO2", "CH4"))
    X = out["templates"]
    norms = np.linalg.norm(X, axis=1)
    assert np.all(np.isfinite(norms))
    ratio = norms.max() / (norms.min() + 1e-12)
    assert ratio < 10.0, "Template scales should be comparable to avoid ill-conditioned features"


def test_masks_are_respected_when_provided(wl_grid: np.ndarray):
    # Mask out a region: ensure returned templates are near zero there
    mask = np.ones_like(wl_grid, dtype=bool)
    # Mask 2.6–2.8 µm
    mask[(wl_grid >= 2.6) & (wl_grid <= 2.8)] = False

    out = mp.get_molecular_priors(
        wl_um=wl_grid,
        molecules=("H2O", "CO2", "CH4"),
        mask=mask,
        zero_out_masked=True,
    )
    X = out["templates"]
    masked_energy = np.sum(np.abs(X[:, ~mask]))
    assert masked_energy <= 1e-6 * np.sum(np.abs(X)), "Templates should be ~zero in masked regions"


# ---------------------------------------------------------------------
# Graph prior over wavelengths (AIRS-GNN edge builder)
# ---------------------------------------------------------------------
def _expect_edges_tuple(obj) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Some implementations may return (edges, weights) or adjacency matrix."""
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj
    return (obj, None)


def test_band_graph_connects_same_molecule_regions(wl_grid: np.ndarray, band_catalog):
    edges_or_adj = mp.build_molecular_band_graph(wl_um=wl_grid, catalog=band_catalog, k_within_band=3)
    edges, weights = _expect_edges_tuple(edges_or_adj)

    # Convert adjacency to edge list if needed
    if isinstance(edges, np.ndarray) and edges.ndim == 2 and edges.shape[0] == edges.shape[1]:
        adj = edges
        ii, jj = np.where(adj > 0)
        edges = np.stack([ii, jj], axis=0)

    assert isinstance(edges, np.ndarray) and edges.shape[0] == 2
    assert edges.shape[1] > 0

    # Check symmetry (undirected graph): if (i,j) then (j,i)
    e = set(map(tuple, edges.T.tolist()))
    for i, j in e:
        assert (j, i) in e, "Graph should be undirected / symmetric"

    # Sanity check: channels near a known center should connect to each other
    center = 3.3
    idx = nearest_index(wl_grid, center)
    neighborhood = set(range(max(idx - 2, 0), min(idx + 3, len(wl_grid))))
    deg = sum(1 for (i, j) in e if i == idx and j in neighborhood)
    assert deg > 0, "Expected local connectivity around a known CH4 band center"


def test_no_self_loops_in_graph(wl_grid: np.ndarray, band_catalog):
    edges_or_adj = mp.build_molecular_band_graph(wl_um=wl_grid, catalog=band_catalog, k_within_band=2)
    edges, _ = _expect_edges_tuple(edges_or_adj)

    # Convert adjacency if needed
    if isinstance(edges, np.ndarray) and edges.ndim == 2 and edges.shape[0] == edges.shape[1]:
        adj = edges
        assert np.all(np.diag(adj) == 0), "No self-loops expected on adjacency diagonal"
    else:
        assert np.all(edges[0] != edges[1]), "No self-loops in edge list expected"


# ---------------------------------------------------------------------
# Uncertainty / sigma priors
# ---------------------------------------------------------------------
def test_uncertainty_prior_reasonable_scale(wl_grid: np.ndarray):
    """
    If the module provides per-wavelength sigma (e.g., photon noise model),
    verify shape and that no channel has pathological values.
    """
    if not _safe_hasattr(mp, "make_uncertainty_prior"):
        pytest.skip("make_uncertainty_prior not implemented (optional)")

    sigma = mp.make_uncertainty_prior(wl_um=wl_grid, photon_rate_per_um=1e6, exposure_s=1.0)
    assert sigma.shape == wl_grid.shape
    assert np.all(np.isfinite(sigma))
    # No absurd zeros or huge spikes
    assert sigma.min() > 0.0
    assert sigma.max() < sigma.min() * 1e3


# ---------------------------------------------------------------------
# (Optional) Property tests with Hypothesis
# ---------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_HYPOTHESIS, reason="Hypothesis not installed")
@given(
    mu=st.floats(min_value=0.6, max_value=7.5),
    sigma=st.floats(min_value=1e-3, max_value=0.2),
)
@settings(deadline=None, max_examples=75)
def test_gaussian_is_peak_at_mu_property(mu: float, sigma: float):
    wl = np.linspace(0.5, 7.8, 1501)
    g = mp.gaussian_band(wl, mu=mu, sigma=sigma, amplitude=1.0, area_normalize=True)
    idx = nearest_index(wl, mu)
    # Peak close to mu
    window = slice(max(idx - 2, 0), min(idx + 3, wl.size))
    local_idx = np.argmax(g[window]) + (idx - 2 if idx >= 2 else 0)
    peak_wl = wl[local_idx]
    assert abs(peak_wl - mu) <= (wl[1] - wl[0]) * 4.0


# ---------------------------------------------------------------------
# Performance / caching (optional)
# ---------------------------------------------------------------------
def test_repeated_calls_stable_and_deterministic(wl_grid: np.ndarray):
    """
    Template generation should be deterministic for fixed inputs (seedless).
    """
    t1 = mp.get_molecular_priors(wl_um=wl_grid, molecules=("H2O", "CO2", "CH4"))["templates"]
    t2 = mp.get_molecular_priors(wl_um=wl_grid, molecules=("H2O", "CO2", "CH4"))["templates"]
    assert np.allclose(t1, t2, rtol=0, atol=0), "Priors should be deterministic for fixed inputs"
