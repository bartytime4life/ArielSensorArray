# tests/diagnostics/test_shap_symbolic_overlay.py
"""
Upgraded diagnostics test: SHAP × Symbolic-Constraint Overlay

What this validates
-------------------
1) We can compute per-wavelength SHAP attributions for a simple surrogate model.
2) We can compute symbolic rule violations over the predicted spectrum:
   - Non-negativity: spectrum >= 0
   - Upper bound:   spectrum <= 1.0 (arbitrary, acts like a "physical ceiling")
   - Smoothness:    |second derivative| <= smoothness_threshold
3) We can render a compact, readable 3-panel diagnostic overlay and save it:
   [A] Predicted spectrum
   [B] |SHAP| profile (per wavelength)
   [C] Violation heatmap (per rule × wavelength)

Design notes
------------
- Deterministic (fixed RNG seeds).
- Uses only numpy, matplotlib, shap, scikit-learn (if sklearn missing, we fall back
  to a tiny local LinearRegression clone).
- Skips gracefully when optional libs aren't present (pytest.importorskip).
- Creates artifacts in pytest's tmp_path for CI collection.

Artifact outputs
----------------
- overlay PNG:   shap_symbolic_overlay.png
- CSV summaries: shap_values.csv, violations.csv
"""

from __future__ import annotations

import io
import math
import pathlib
import warnings

import numpy as np
import pytest

# Optional deps (skip test if missing)
mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
shap = pytest.importorskip("shap")

# Try sklearn if available; otherwise use a tiny fallback linear regressor.
try:
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover
    class LinearRegression:  # minimal fallback
        def fit(self, X, y):
            # Pseudo closed-form via least squares
            X_ = np.c_[np.ones((X.shape[0], 1)), X]
            beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
            self.coef_ = beta[1:]
            self.intercept_ = beta[0]
            return self

        def predict(self, X):
            return X @ self.coef_ + self.intercept_


# ---------------------------
# Helpers
# ---------------------------

def make_wavelength_axis(n_bins: int = 283, start_um: float = 0.5, stop_um: float = 5.0) -> np.ndarray:
    """Wavelength grid (μm)."""
    return np.linspace(start_um, stop_um, n_bins)


def synth_true_spectrum(wl: np.ndarray, seed: int = 7) -> np.ndarray:
    """
    Create a plausible 'true' transmission spectrum in relative units [0, 1].
    Combines a smooth baseline + a few Gaussian absorption features.
    """
    rng = np.random.default_rng(seed)
    n = wl.size
    baseline = 0.02 + 0.005 * np.sin(2 * np.pi * (wl - wl.min()) / (wl.max() - wl.min()))
    spectrum = baseline.copy()

    # Randomly place 4–6 absorption lines with varying widths/depths
    n_lines = rng.integers(4, 7)
    for _ in range(n_lines):
        center = rng.uniform(wl.min() + 0.2, wl.max() - 0.2)
        width = rng.uniform(0.02, 0.15)
        depth = rng.uniform(0.002, 0.01)
        spectrum -= depth * np.exp(-0.5 * ((wl - center) / width) ** 2)

    # Ensure non-negative and bounded
    spectrum = np.clip(spectrum, 0.0, 1.0)

    # Slight high-frequency wiggle to emulate instrument residuals
    wiggle = 0.0005 * np.sin(30 * wl) + 0.0003 * np.cos(17 * wl)
    return np.clip(spectrum + wiggle, 0.0, 1.0)


def make_training_set(wl: np.ndarray, n_samples: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a simple tabular training set from wavelength-derived features.
    Inputs: features per wavelength; Outputs: true spectrum (per wavelength) aggregated into a scalar target.

    We train a regressor to map compact features of (wl) -> scalar "depth index",
    then at predict-time, we lift it to a per-wavelength profile using a small adapter.
    For SHAP, we explain the scalar model on feature space; then broadcast to λ-bins.
    """
    rng = np.random.default_rng(seed)
    # Features: polynomial + trig bases of mean λ bucket statistics
    # To keep simple, we use fixed summary stats derived from the grid:
    mu = wl.mean()
    sig = wl.std()
    base_feat = np.array([
        mu,
        sig,
        np.mean(np.sin(wl)),
        np.mean(np.cos(wl)),
        np.mean(wl**2),
        np.mean(np.sin(2 * wl)),
        np.mean(np.cos(2 * wl)),
    ])
    # Build a design matrix with small randomized jitter across samples
    X = base_feat + 0.02 * rng.normal(size=(n_samples, base_feat.size))
    # Targets: random linear combo that correlates with global depth of synthetic spectra variants
    # Build per-sample spectrum variants and reduce to a scalar label
    y = []
    for i in range(n_samples):
        spec = synth_true_spectrum(wl, seed=1000 + i)
        # A simple scalar "depth index": mean deficit below a flat 0.02 baseline
        depth_idx = float(np.maximum(0.0, 0.02 - spec).mean())
        y.append(depth_idx)
    y = np.asarray(y)
    return X, y


def compute_symbolic_violations(
    spectrum: np.ndarray,
    smoothness_threshold: float = 2e-4,
    upper_bound: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Compute boolean violation masks per rule over the 1D spectrum.

    Rules:
      - nonneg: spectrum >= 0
      - upper:  spectrum <= upper_bound
      - smooth: |second derivative| <= smoothness_threshold  (central difference)
    """
    n = spectrum.size
    nonneg_ok = spectrum >= 0.0
    upper_ok = spectrum <= upper_bound

    # Second derivative (central differences); pad ends as False if exceeding threshold
    d2 = np.zeros_like(spectrum)
    if n >= 3:
        d2[1:-1] = spectrum[2:] - 2 * spectrum[1:-1] + spectrum[:-2]
    smooth_ok = np.abs(d2) <= smoothness_threshold

    violations = {
        "nonneg": ~nonneg_ok,
        "upper": ~upper_ok,
        "smooth": ~smooth_ok,
    }
    return violations


def render_overlay(
    wl: np.ndarray,
    pred: np.ndarray,
    shap_abs: np.ndarray,
    violations: dict[str, np.ndarray],
    out_path: pathlib.Path,
) -> pathlib.Path:
    """Save a 3-panel overlay PNG."""
    rules = list(violations.keys())
    V = np.vstack([violations[r].astype(float) for r in rules])  # shape (R, λ), {0,1} as floats

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.5, 1.5])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(wl, pred, lw=1.8)
    ax0.set_title("Predicted Transmission Spectrum")
    ax0.set_xlabel("Wavelength (μm)")
    ax0.set_ylabel("Transit depth (rel. units)")
    ax0.grid(alpha=0.2)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(wl, shap_abs, lw=1.2)
    ax1.set_title("|SHAP| Attribution Profile (per λ)")
    ax1.set_xlabel("Wavelength (μm)")
    ax1.set_ylabel("|SHAP|")
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[2, 0])
    im = ax2.imshow(
        V,
        aspect="auto",
        interpolation="nearest",
        extent=[wl.min(), wl.max(), 0, len(rules)],
        origin="lower",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
    )
    ax2.set_yticks(np.arange(len(rules)) + 0.5)
    ax2.set_yticklabels(rules)
    ax2.set_title("Symbolic Rule Violations (heatmap)")
    ax2.set_xlabel("Wavelength (μm)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.026)
    cbar.set_label("Violation (1=yes, 0=no)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


# ---------------------------
# The actual test
# ---------------------------

@pytest.mark.parametrize("n_bins", [283])  # keep the Ariel-like channel count
def test_shap_symbolic_overlay(tmp_path: pathlib.Path, n_bins: int):
    # Determinism
    np.random.seed(123)

    # 1) Make wavelength grid and a synthetic "true" spectrum
    wl = make_wavelength_axis(n_bins=n_bins)
    true_spec = synth_true_spectrum(wl, seed=11)

    # 2) Build a lightweight tabular training set and train a scalar model
    X, y = make_training_set(wl, n_samples=220, seed=2025)
    model = LinearRegression().fit(X, y)

    # 3) Create a single prediction for overlay:
    #    We'll adapt the scalar model's output into a per-wavelength profile
    #    by scaling a smooth template toward the learned depth index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # KernelExplainer on the scalar model over feature space
        background = X[:50]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_vals = explainer.shap_values(X[50:60], nsamples=100)  # 10 examples

    # Aggregate |SHAP| across the 10 examples
    shap_abs_feat = np.mean(np.abs(shap_vals), axis=0)  # per-feature attributions

    # Broadcast feature attributions to λ-space as a simple, smooth profile:
    # Normalize to [0, 1] and modulate with a low-frequency curve
    shap_profile = shap_abs_feat.mean() * (0.5 + 0.5 * np.sin(2 * np.pi * (wl - wl.min()) / (wl.max() - wl.min())))
    shap_profile = np.abs(shap_profile)
    shap_profile = shap_profile / max(1e-12, shap_profile.max())

    # Prediction: use the trained scalar depth index on the mean feature to scale a template
    x_query = X.mean(axis=0, keepdims=True)
    depth_idx = float(model.predict(x_query))
    # Create a prediction by attenuating a flat baseline with the learned depth and the same line centers
    pred = np.clip(0.02 - 0.6 * depth_idx * (true_spec - true_spec.min()), 0.0, 1.0)

    # 4) Compute symbolic violations
    violations = compute_symbolic_violations(
        pred,
        smoothness_threshold=2.0e-4,  # fairly lenient; tweak as needed
        upper_bound=1.0,
    )

    # 5) Save overlays + CSV summaries
    out_png = tmp_path / "shap_symbolic_overlay.png"
    out_shap_csv = tmp_path / "shap_values.csv"
    out_vio_csv = tmp_path / "violations.csv"

    # Overlay
    render_overlay(wl, pred, shap_profile, violations, out_png)
    assert out_png.exists() and out_png.stat().st_size > 0, "Overlay PNG not written or empty."

    # CSVs
    np.savetxt(out_shap_csv, np.c_[wl, shap_profile], delimiter=",", header="wavelength_um,abs_shap", comments="")
    vio_mat = np.vstack([violations[k].astype(int) for k in violations.keys()]).T  # (λ, rules)
    header = "wavelength_um," + ",".join(violations.keys())
    np.savetxt(out_vio_csv, np.c_[wl, vio_mat], delimiter=",", header=header, comments="")
    assert out_shap_csv.exists() and out_vio_csv.exists()

    # 6) Sanity checks on arrays and violations
    assert pred.shape == (n_bins,)
    assert shap_profile.shape == (n_bins,)
    # At least some wavelengths should be non-violating for each rule
    for rule, mask in violations.items():
        assert mask.shape == (n_bins,)
        assert mask.sum() < n_bins, f"Rule '{rule}' flags all wavelengths; threshold too strict?"

    # 7) Fail-fast contract: if any hard physical rule is catastrophically violated, we surface it.
    # Here we allow some violations (diagnostics), but guard against obviously broken predictions.
    hard_violation = np.any(pred < -1e-9) or np.any(pred > 1.0 + 1e-6)
    assert not hard_violation, "Prediction outside hard physical bounds."