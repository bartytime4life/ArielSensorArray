# tests/regression/test_predict_ensemble_50v.py
"""
Regression tests for the V50 ensemble predictor.

These tests focus on:
- Weighted ensembling of multiple model means/variances
- Optional MC-Dropout epistemic variance
- Post-hoc sigma temperature calibration
- Determinism (seeded RNG paths)
- Shape, dtype, and error handling

They assume an API like:

    from spectramind.predict.ensemble_50v import ensemble_predict

    mu, sigma = ensemble_predict(
        models,                # iterable of callables or model objects
        inputs,                # np.ndarray [..., features]
        weights=None,          # optional list/array of non-negative weights
        mc_dropout=0,          # int, number of stochastic forward passes per model
        sigma_temperature=None,# optional float > 0, multiplies predicted sigma
        rng=None,              # optional np.random.Generator for determinism
        ignore_nan=False,      # if True, ignore NaN-producing models for a sample
    )

- Each model when called must return (mu, sigma) for given inputs, where
  mu:    np.ndarray [N, D]    – per-sample mean predictions
  sigma: np.ndarray [N, D]    – per-sample (strictly positive) standard deviations

If your project exposes a different symbol or signature, feel free to adapt the
import block or add a thin adapter in spectramind.predict.ensemble_50v.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import pytest


# ---- Import under test (skip the whole file if module is missing) ----------------
ensemble_mod = pytest.importorskip(
    "spectramind.predict.ensemble_50v",
    reason="spectramind.predict.ensemble_50v not importable; skip regression tests",
)
ensemble_predict = getattr(ensemble_mod, "ensemble_predict", None)
if ensemble_predict is None:
    pytest.skip("ensemble_predict symbol missing in spectramind.predict.ensemble_50v", allow_module_level=True)


# ---- Helpers --------------------------------------------------------------------
class DummyModel:
    """A deterministic dummy model producing affine means and constant sigmas.

    mu = A @ x + b   (broadcast to [N, D])
    sigma = s (scalar or array-like) broadcast to [N, D]

    Optionally supports MC-dropout via `stochastic=True`, adding Gaussian noise to mu.
    """

    def __init__(
        self,
        A: float,
        b: float,
        s: float | np.ndarray,
        D: int,
        stochastic: bool = False,
        noise_scale: float = 0.01,
        seed: int = 0,
    ):
        self.A = float(A)
        self.b = float(b)
        self.s = np.array(s, dtype=np.float32)
        self.D = int(D)
        self.stochastic = stochastic
        self.noise_scale = float(noise_scale)
        self.rng = np.random.default_rng(seed)

    def __call__(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = X.shape[0]
        # Produce a simple feature -> output mapping: mean depends on the first feature only.
        base = (self.A * X[:, :1] + self.b).astype(np.float32)  # [N, 1]
        mu = np.repeat(base, self.D, axis=1)                    # [N, D]
        if self.stochastic:
            mu = mu + self.rng.normal(loc=0.0, scale=self.noise_scale, size=mu.shape).astype(np.float32)
        sigma = np.broadcast_to(self.s, (N, self.D)).astype(np.float32)
        sigma = np.clip(sigma, 1e-9, None)  # guard positivity
        return mu, sigma


def _analytical_weighted_ensemble(
    mus: Iterable[np.ndarray], sigmas: Iterable[np.ndarray], weights: np.ndarray | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ensemble mean and sigma using first/second-moment combination.

    Given model i with mean μ_i and (aleatoric) variance σ_i^2, and non-negative weights w_i, the
    ensemble mean μ̄ = (Σ w_i μ_i) / W and the total variance (aleatoric + between-model) per element is:

        Var_total = (Σ w_i * (σ_i^2 + μ_i^2)) / W - μ̄^2

    Return (μ̄, sqrt(Var_total)).
    """
    mus = list(mus)
    sigmas = list(sigmas)
    assert len(mus) == len(sigmas) >= 2
    W = len(mus) if weights is None else float(np.sum(weights))
    w = np.ones(len(mus), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    w = w / (np.sum(w) + 1e-12)

    mu_stack = np.stack(mus, axis=0).astype(np.float64)         # [M, N, D]
    var_stack = np.stack([s**2 for s in sigmas], axis=0).astype(np.float64)
    w_col = w[:, None, None]                                    # [M, 1, 1]

    mu_bar = np.sum(w_col * mu_stack, axis=0)                   # [N, D]
    second_moment = np.sum(w_col * (var_stack + mu_stack**2), axis=0)  # [N, D]
    var_total = np.clip(second_moment - mu_bar**2, 0.0, None)
    return mu_bar.astype(np.float32), np.sqrt(var_total).astype(np.float32)


# ---- Fixtures -------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def toy_inputs(rng) -> np.ndarray:
    # 12 samples, 7 features – first feature used by DummyModel
    X = rng.normal(size=(12, 7)).astype(np.float32)
    return X


# ---- Tests ----------------------------------------------------------------------
@pytest.mark.regression
@pytest.mark.parametrize("D", [1, 3, 17])
def test_weighted_mean_and_variance_matches_analytical(D: int, toy_inputs: np.ndarray, rng):
    """Ensembling multiple models should match analytical first/second-moment combination."""
    # Three diverse models
    m1 = DummyModel(A=0.5, b=0.1, s=0.02, D=D)
    m2 = DummyModel(A=-0.2, b=0.0, s=0.05, D=D)
    m3 = DummyModel(A=0.9, b=-0.3, s=0.01, D=D)

    models = [m1, m2, m3]
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Reference combination
    mus, sigmas = zip(*(m(toy_inputs) for m in models))
    mu_ref, sigma_ref = _analytical_weighted_ensemble(mus, sigmas, weights)

    # Under test
    mu_hat, sigma_hat = ensemble_predict(
        models=models,
        inputs=toy_inputs,
        weights=weights,
        mc_dropout=0,
        sigma_temperature=None,
        rng=np.random.default_rng(7),
    )

    assert mu_hat.shape == mu_ref.shape == (toy_inputs.shape[0], D)
    assert sigma_hat.shape == sigma_ref.shape == (toy_inputs.shape[0], D)

    # Tolerances chosen to be tight for purely deterministic path
    np.testing.assert_allclose(mu_hat, mu_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sigma_hat, sigma_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.regression
def test_sigma_temperature_calibration_scales_sigma(toy_inputs: np.ndarray):
    """Post-hoc temperature scaling should scale sigma but leave means intact."""
    D = 5
    T = 1.37
    models = [DummyModel(A=0.3, b=0.0, s=0.05, D=D), DummyModel(A=0.6, b=0.1, s=0.07, D=D)]

    mu_base, sigma_base = ensemble_predict(models, toy_inputs, mc_dropout=0, sigma_temperature=None)
    mu_T, sigma_T = ensemble_predict(models, toy_inputs, mc_dropout=0, sigma_temperature=T)

    np.testing.assert_allclose(mu_T, mu_base, rtol=0, atol=0)  # means unchanged
    np.testing.assert_allclose(sigma_T, sigma_base * T, rtol=1e-6, atol=1e-7)


@pytest.mark.regression
def test_mc_dropout_adds_epistemic_variance(toy_inputs: np.ndarray):
    """MC-Dropout with stochastic models should produce larger total variance than deterministic path."""
    D = 4
    models = [
        DummyModel(A=0.5, b=0.0, s=0.03, D=D, stochastic=True, noise_scale=0.02, seed=1),
        DummyModel(A=0.2, b=0.1, s=0.03, D=D, stochastic=True, noise_scale=0.02, seed=2),
    ]
    # No MC sampling (deterministic path)
    _, sigma_det = ensemble_predict(models, toy_inputs, mc_dropout=0, rng=np.random.default_rng(777))
    # With MC sampling (epistemic captured)
    _, sigma_mc = ensemble_predict(models, toy_inputs, mc_dropout=12, rng=np.random.default_rng(777))

    # The MC-derived sigma should be >= deterministic (elementwise, allow equality in rare cases)
    assert np.all(sigma_mc >= sigma_det - 1e-8)
    # And strictly larger for most elements
    frac_strict = float(np.mean(sigma_mc > sigma_det + 1e-8))
    assert frac_strict > 0.5, f"Expected >50% elements to increase; got {frac_strict:.2%}"


@pytest.mark.regression
def test_determinism_with_seed(toy_inputs: np.ndarray):
    """Given a fixed RNG seed, ensemble outputs must be bitwise identical."""
    D = 6
    models = [
        DummyModel(A=0.41, b=-0.02, s=0.02, D=D, stochastic=True, noise_scale=0.03, seed=123),
        DummyModel(A=-0.17, b=0.10, s=0.05, D=D, stochastic=True, noise_scale=0.04, seed=456),
    ]
    rng1 = np.random.default_rng(2025)
    rng2 = np.random.default_rng(2025)

    mu1, s1 = ensemble_predict(models, toy_inputs, mc_dropout=9, rng=rng1)
    mu2, s2 = ensemble_predict(models, toy_inputs, mc_dropout=9, rng=rng2)

    # Bitwise identical expectations
    assert mu1.dtype == mu2.dtype == np.float32
    assert s1.dtype == s2.dtype == np.float32
    assert mu1.shape == mu2.shape == (toy_inputs.shape[0], D)
    assert s1.shape == s2.shape == (toy_inputs.shape[0], D)
    assert np.array_equal(mu1, mu2)
    assert np.array_equal(s1, s2)


@pytest.mark.regression
def test_shape_and_dtype_contract(toy_inputs: np.ndarray):
    """Outputs should be float32 and have shape [N, D]."""
    D = 9
    models = [DummyModel(A=0.1, b=0.0, s=0.02, D=D), DummyModel(A=0.2, b=0.0, s=0.02, D=D)]
    mu, sigma = ensemble_predict(models, toy_inputs, mc_dropout=0)

    assert mu.shape == (toy_inputs.shape[0], D)
    assert sigma.shape == (toy_inputs.shape[0], D)
    assert mu.dtype == np.float32
    assert sigma.dtype == np.float32
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0)


@pytest.mark.regression
def test_dimension_mismatch_raises_value_error(toy_inputs: np.ndarray):
    """If a model returns a different output dimensionality, the ensemble should raise a ValueError."""
    D1, D2 = 5, 7
    m_ok = DummyModel(A=0.3, b=0.1, s=0.02, D=D1)
    m_bad = DummyModel(A=0.6, b=-0.2, s=0.03, D=D2)

    with pytest.raises((ValueError, AssertionError)):
        _ = ensemble_predict([m_ok, m_bad], toy_inputs, mc_dropout=0)


@pytest.mark.regression
def test_ignore_nan_model_outputs_if_configured(toy_inputs: np.ndarray):
    """If a model emits NaNs and ignore_nan=True, the ensemble should fallback to the remaining models."""
    D = 3

    class NaNModel(DummyModel):
        def __call__(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mu, sigma = super().__call__(X)
            mu[::2, :] = np.nan  # inject NaNs in half the rows
            return mu, sigma

    m1 = DummyModel(A=0.1, b=0.0, s=0.05, D=D)
    m2 = NaNModel(A=0.2, b=0.1, s=0.02, D=D)

    # Without ignore_nan -> should raise
    with pytest.raises((ValueError, AssertionError)):
        _ = ensemble_predict([m1, m2], toy_inputs, mc_dropout=0, ignore_nan=False)

    # With ignore_nan=True -> should succeed & match single-model m1 on NaN rows fallback
    mu_single, sigma_single = ensemble_predict([m1], toy_inputs, mc_dropout=0)
    mu_ens, sigma_ens = ensemble_predict([m1, m2], toy_inputs, mc_dropout=0, ignore_nan=True)

    # On the rows where NaNs were injected, ensemble should reduce to m1
    rows_nan = np.arange(toy_inputs.shape[0])[::2]
    np.testing.assert_allclose(mu_ens[rows_nan], mu_single[rows_nan], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sigma_ens[rows_nan], sigma_single[rows_nan], rtol=1e-6, atol=1e-7)


@pytest.mark.regression
@pytest.mark.slow
def test_cli_predict_ensemble_smoke(tmp_path: Path, toy_inputs: np.ndarray):
    """If the CLI is exposed as `spectramind predict --ensemble`, run a tiny smoke test.

    Skips if the `spectramind` CLI isn't available on PATH.
    """
    cli = shutil.which("spectramind")
    if cli is None:
        pytest.skip("`spectramind` CLI not found on PATH; skipping CLI smoke test")

    # Prepare inputs on disk as .npy for the CLI (common simple interchange)
    X_path = tmp_path / "X.npy"
    np.save(X_path, toy_inputs)

    out_mu = tmp_path / "mu.npy"
    out_sigma = tmp_path / "sigma.npy"

    # Minimal CLI args; adapt if your CLI uses different flags
    cmd = [
        cli, "predict",
        "--ensemble",
        "--inputs", str(X_path),
        "--out-mu", str(out_mu),
        "--out-sigma", str(out_sigma),
        "--mc-dropout", "4",
        "--sigma-temperature", "1.1",
        "--seed", "42",
    ]
    # If your CLI requires a model registry path or config, consider honoring environment defaults.
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        pytest.fail(f"CLI ensemble predict failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    # Verify artifacts
    assert out_mu.exists() and out_sigma.exists()
    mu = np.load(out_mu)
    sigma = np.load(out_sigma)
    assert mu.shape == sigma.shape == (toy_inputs.shape[0], mu.shape[1])
    assert mu.dtype == np.float32 and sigma.dtype == np.float32
    assert np.all(sigma > 0.0)
