# tests/diagnostics/test_shap_overlay.py
"""
SpectraMind V50 — SHAP overlay diagnostics tests.

Goals
-----
- Exercise a minimal, fast SHAP workflow on synthetic data (no internet).
- Verify shapes, stability, top‑k ranking, and plotting/saving behavior.
- Keep hard dependencies optional: skip cleanly if `shap`/`sklearn`/`matplotlib` absent.
- Be deterministic (fixed seed), file‑system safe (temp dirs), and CI‑friendly.

Expected API (soft contract)
----------------------------
The project should provide a diagnostics helper module, tentatively:

    spectramind.diagnostics.shap_overlay

with the following functions (or compatible signatures):

    compute_shap_values(model, X, feature_names=None, **kwargs) -> dict:
        Returns {
          "values": np.ndarray [n_samples, n_features],
          "base_values": np.ndarray [n_samples],
          "feature_names": List[str]
        }

    rank_top_features(shap_values, feature_names, k=10) -> List[Tuple[str, float]]:
        Returns top‑k by mean |SHAP| value across samples.

    make_overlay_chart(
        shap_values, feature_names, *,
        top_k=10,
        out_path: str | Path | None = None,
        return_ax: bool = False,
        **kwargs,
    ) -> Optional[Path | matplotlib.axes.Axes]:
        Creates a compact overlay plot (e.g., bar overlay of mean |SHAP| with CI/range).
        If out_path is provided, saves a PNG there and returns the Path.
        If return_ax is True (and out_path is None), returns an Axes.

If your implementation differs slightly, adapt this test or provide light wrappers.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

shap = pytest.importorskip("shap", reason="`shap` is required for SHAP overlay tests")
sklearn = pytest.importorskip("sklearn", reason="`scikit-learn` is required for SHAP overlay tests")
plt = pytest.importorskip("matplotlib.pyplot", reason="`matplotlib` is required for plotting")

# Try multiple import paths to be forgiving of repo layout.
_overlay_mod = None  # will be assigned after import attempts
_import_errors = []


def _try_import(modname: str):
    try:
        mod = __import__(modname, fromlist=["*"])
        return mod
    except Exception as e:  # pragma: no cover (diagnostic path)
        _import_errors.append((modname, repr(e)))
        return None


for candidate in (
    "spectramind.diagnostics.shap_overlay",
    "src.spectramind.diagnostics.shap_overlay",
    "spectramind_v50.diagnostics.shap_overlay",
):
    _overlay_mod = _try_import(candidate)
    if _overlay_mod:
        break


needs_overlay = pytest.mark.skipif(
    _overlay_mod is None,
    reason=(
        "Could not import shap_overlay module. Tried:\n  - "
        + "\n  - ".join(name for name, _ in _import_errors)
        + "\nLast errors:\n  "
        + "\n  ".join(f"{name}: {err}" for name, err in _import_errors)
    ),
)


# ---------------------------
# Fixtures & test utilities
# ---------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(1337)


@pytest.fixture(scope="module")
def toy_linear_data(rng):
    """
    Small deterministic linear system with 8 features.
    y = X @ beta + eps, where only a subset truly matters.
    """
    n, p = 64, 8
    X = rng.normal(0, 1, size=(n, p))
    # Choose betas with clear magnitudes for top-k testing
    beta = np.array([2.0, -1.5, 0.0, 0.0, 0.75, 0.0, -0.25, 0.0])
    y = X @ beta + rng.normal(0, 0.10, size=n)  # small noise so SHAP is stable
    feature_names = [f"f{i}" for i in range(p)]
    return X, y, beta, feature_names


@pytest.fixture(scope="module")
def fitted_linear_model(toy_linear_data):
    from sklearn.linear_model import LinearRegression

    X, y, *_ = toy_linear_data
    model = LinearRegression()
    model.fit(X, y)
    return model


# ----------------------------------------
# Core tests against the overlay helpers
# ----------------------------------------


@needs_overlay
def test_compute_shap_values_shapes(fitted_linear_model, toy_linear_data):
    X, _, _, feature_names = toy_linear_data

    out = _overlay_mod.compute_shap_values(fitted_linear_model, X, feature_names=feature_names)
    assert isinstance(out, dict), "compute_shap_values should return a dict"
    assert {"values", "base_values", "feature_names"} <= set(out), "Missing expected keys"

    shap_values = out["values"]
    base_values = out["base_values"]
    fnames = out["feature_names"]

    assert shap_values.shape == X.shape, "SHAP values shape must match input (n, p)"
    assert base_values.shape == (X.shape[0],), "Base values must be 1D over samples"
    assert list(fnames) == list(feature_names), "Feature names must round-trip unchanged"

    # Basic finite checks
    assert np.isfinite(shap_values).all()
    assert np.isfinite(base_values).all()


@needs_overlay
def test_rank_top_features_matches_signal_strength(fitted_linear_model, toy_linear_data):
    """Top-k by mean |SHAP| should align with the strongest true coefficients."""
    X, _, beta, feature_names = toy_linear_data

    out = _overlay_mod.compute_shap_values(fitted_linear_model, X, feature_names=feature_names)
    shap_values = out["values"]
    ranked: List[Tuple[str, float]] = _overlay_mod.rank_top_features(shap_values, feature_names, k=3)

    assert len(ranked) == 3
    names, scores = zip(*ranked)
    assert all(s >= 0 for s in scores)

    # The top-3 absolute beta are f0 (2.0), f1 (1.5), f4 (0.75)
    expected_top3 = {"f0", "f1", "f4"}
    assert expected_top3.issubset(set(names)), f"Top-3 should include {expected_top3}, got {set(names)}"


@needs_overlay
def test_make_overlay_chart_saves_png(tmp_path: Path, fitted_linear_model, toy_linear_data):
    X, _, feature_names = toy_linear_data[0], toy_linear_data[1], toy_linear_data[3]
    out = _overlay_mod.compute_shap_values(fitted_linear_model, X, feature_names=feature_names)
    shap_values = out["values"]

    out_file = tmp_path / "shap_overlay.png"
    result = _overlay_mod.make_overlay_chart(
        shap_values,
        feature_names,
        top_k=5,
        out_path=out_file,
        return_ax=False,
    )
    assert isinstance(result, (str, Path)), "Expected a path-like return value when saving"
    saved = Path(result)
    assert saved.exists() and saved.suffix.lower() == ".png", "Overlay image not saved as PNG"


@needs_overlay
def test_make_overlay_chart_return_ax(fitted_linear_model, toy_linear_data):
    X, _, feature_names = toy_linear_data[0], toy_linear_data[1], toy_linear_data[3]
    out = _overlay_mod.compute_shap_values(fitted_linear_model, X, feature_names=feature_names)
    shap_values = out["values"]

    ax = _overlay_mod.make_overlay_chart(
        shap_values,
        feature_names,
        top_k=4,
        out_path=None,
        return_ax=True,
    )
    # Delayed import here to avoid hard dependency if not requested
    import matplotlib.axes

    assert isinstance(ax, matplotlib.axes.Axes)


@needs_overlay
def test_compute_shap_values_nan_guard(fitted_linear_model, toy_linear_data):
    """
    If upstream code accidentally passes NaNs into the overlay, the helper should be defensive.
    We expect a ValueError with a clear message rather than silent failure.
    """
    X, _, feature_names = toy_linear_data[0].copy(), toy_linear_data[1], toy_linear_data[3]
    X[0, 0] = np.nan

    with pytest.raises((ValueError, AssertionError)):
        _overlay_mod.compute_shap_values(fitted_linear_model, X, feature_names=feature_names)


# ----------------------------------------
# Optional: tiny e2e sanity via SHAP LinearExplainer
# ----------------------------------------


@pytest.mark.fast
def test_linearexplainer_e2e_sanity(rng, toy_linear_data, fitted_linear_model):
    """
    Sanity check: SHAP LinearExplainer itself returns stable shapes for a linear model.
    This does not touch project helpers — it verifies environment/tooling in CI.
    """
    X, _, feature_names = toy_linear_data[0], toy_linear_data[1], toy_linear_data[3]

    explainer = shap.LinearExplainer(fitted_linear_model, X, feature_names=feature_names)
    sv = explainer.shap_values(X)
    base = explainer.expected_value

    # SHAP may return list for multi-output; for 1D regression it's a 2D array
    assert isinstance(sv, np.ndarray) and sv.shape == X.shape
    assert np.isfinite(sv).all()
    assert (np.isfinite(base) and np.ndim(base) == 0) or (
        isinstance(base, np.ndarray) and base.shape in [(1,), ()]
    )


# ----------------------------------------
# CLI (optional) smoke — skipped if CLI not present
# ----------------------------------------


@needs_overlay
@pytest.mark.skipif(
    not hasattr(_overlay_mod, "make_overlay_chart"),
    reason="No overlay plotting helper available",
)
def test_cli_like_usage_smoke(tmp_path: Path, fitted_linear_model, toy_linear_data, monkeypatch):
    """
    Simulate a thin CLI flow by calling the helpers as a pipeline:
      model -> SHAP -> rank -> plot/save
    Ensures no side effects (figures closed) and returns cleanly.
    """
    X, _, feature_names = toy_linear_data[0], toy_linear_data[1], toy_linear_data[3]

    # Compute SHAP
    out = _overlay_mod.compute_shap_values(fitted_linear_model, X, feature_names=feature_names)
    shap_values = out["values"]

    # Rank
    top = _overlay_mod.rank_top_features(shap_values, feature_names, k=5)
    assert len(top) == 5

    # Plot/save
    out_file = tmp_path / "overlay_cli.png"
    returned = _overlay_mod.make_overlay_chart(
        shap_values,
        feature_names,
        top_k=5,
        out_path=out_file,
        return_ax=False,
    )
    assert Path(returned).exists()

    # Extra: ensure Matplotlib figure count doesn't balloon (resource/CI hygiene)
    import matplotlib

    open_figs_before = len(matplotlib.pyplot.get_fignums())
    # Trigger another save to see resources cleaned (if implementation opens new fig)
    _ = _overlay_mod.make_overlay_chart(shap_values, feature_names, top_k=3, out_path=tmp_path / "again.png")
    open_figs_after = len(matplotlib.pyplot.get_fignums())
    # Allow equal or fewer (implementations may close)
    assert open_figs_after <= open_figs_before + 1
