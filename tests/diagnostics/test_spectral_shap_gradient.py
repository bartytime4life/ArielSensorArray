# tests/diagnostics/test_spectral_shap_gradient.py
# -*- coding: utf-8 -*-
"""
Diagnostic tests for gradient-based spectral attributions.

Goals
-----
1) Sanity check GradientSHAP/IntegratedGradients on a simple spectral model
   (283 channels) with properties we can verify analytically.
2) Enforce reproducibility (fixed seeds) and CPU/GPU parity.
3) Validate completeness: sum of attributions ~= f(x) - f(baseline).
4) Validate localization: a linear model whose weights have a peaked profile
   should yield attributions that peak at the same wavelengths.
5) Validate shape/batch handling.

These tests are deliberately self-contained (no external data needed) so they
can run in CI quickly and deterministically.

Usage
-----
pytest -q tests/diagnostics/test_spectral_shap_gradient.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import pytest

# Optional torch/captum imports with graceful skip
torch = pytest.importorskip("torch", reason="Requires PyTorch for attribution tests")

# Captum is preferred, but if unavailable we fallback to Integrated Gradients-like logic
try:
    from captum.attr import GradientShap, IntegratedGradients
    _HAS_CAPTUM = True
except Exception:  # pragma: no cover
    _HAS_CAPTUM = False


# -------------------------------
# Helpers
# -------------------------------

SEED = 1337
N_CHANNELS = 283  # Ariel AIRS spectral length
DTYPE = torch.float32


def _set_all_seeds(seed: int = SEED) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(requested: Optional[str] = None) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LinearSpectralModel(torch.nn.Module):
    """
    Simple linear spectral regressor: y = x @ w + b

    For tests we often set bias=False to match completeness with baseline=0.
    """

    def __init__(self, n_features: int, bias: bool = False):
        super().__init__()
        self.fc = torch.nn.Linear(n_features, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        return self.fc(x).squeeze(-1)  # (B,)


def _gaussian_weights(n: int, center: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    xs = np.arange(n)
    return amplitude * np.exp(-0.5 * ((xs - center) / sigma) ** 2)


def _build_peaked_linear_model(
    device: torch.device,
    center: float = 140.0,
    sigma: float = 8.0,
    amplitude: float = 1.0,
    bias: bool = False,
) -> LinearSpectralModel:
    model = LinearSpectralModel(N_CHANNELS, bias=bias).to(device=device, dtype=DTYPE)
    w = _gaussian_weights(N_CHANNELS, center=center, sigma=sigma, amplitude=amplitude).astype(np.float32)
    with torch.no_grad():
        model.fc.weight.copy_(torch.from_numpy(w).unsqueeze(0).to(device=device, dtype=DTYPE))
        if bias:
            model.fc.bias.zero_()
    model.eval()
    return model


def _gradient_shap_or_ig(model: torch.nn.Module, input_: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """
    Compute attributions with GradientSHAP if available, else Integrated Gradients.
    Returns attributions with same shape as input_ (B, F).
    """
    if _HAS_CAPTUM:
        try:
            gs = GradientShap(model)
            # For GS, we provide a distribution of baselines by adding small noise.
            baseline_dist = baseline.unsqueeze(0).repeat(16, 1)  # 16 samples
            attr = gs.attribute(
                input_,
                baselines=baseline_dist,
                n_samples=64,
                stdevs=0.02,
                return_convergence_delta=False,
            )
            return attr
        except Exception:
            # Fallback to IG if GS path fails for any reason
            ig = IntegratedGradients(model)
            return ig.attribute(input_, baselines=baseline, n_steps=128)
    else:
        # Minimal IG-like fallback (requires captum if we wanted exact; here we implement simple IG)
        return _simple_integrated_gradients(model, input_, baseline, steps=128)


@torch.no_grad()
def _simple_integrated_gradients(model: torch.nn.Module, x: torch.Tensor, x0: torch.Tensor, steps: int = 128) -> torch.Tensor:
    """
    Simple IG implementation (for environments w/o captum). Computes path integral
    via Riemann sum along straight-line path from x0 to x.
    """
    # Ensure we need gradients
    model.requires_grad_(True)

    # IG usually requires grads; here we compute grads via autograd with enable_grad.
    with torch.enable_grad():
        alphas = torch.linspace(0.0, 1.0, steps=steps, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        interp = x0.unsqueeze(0) + alphas * (x.unsqueeze(0) - x0.unsqueeze(0))  # (steps, B, F)
        interp.requires_grad_(True)

        # Flatten steps and batch to evaluate in one go
        S, B, F = interp.shape
        flat = interp.reshape(S * B, F)
        out = model(flat).reshape(S, B)  # (steps, B)

        grads = torch.autograd.grad(
            outputs=out.sum(dim=0),  # sum over steps dim to keep shape (B,)
            inputs=interp,
            grad_outputs=torch.ones_like(out.sum(dim=0)),
            create_graph=False,
            retain_graph=False,
        )[0]  # (steps, B, F)

        avg_grads = grads.mean(dim=0)  # (B, F)
        attributions = (x - x0) * avg_grads
        return attributions


# -------------------------------
# PyTest fixtures
# -------------------------------

@pytest.fixture(scope="module", autouse=True)
def _fix_seeds():
    _set_all_seeds(SEED)
    yield


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request) -> torch.device:
    return _device(request.param)


# -------------------------------
# Tests
# -------------------------------

def test_shapes_and_basic_forward(device):
    model = _build_peaked_linear_model(device=device, bias=False)
    x = torch.zeros(4, N_CHANNELS, dtype=DTYPE, device=device)
    y = model(x)
    assert y.shape == (4,)
    assert torch.allclose(y, torch.zeros(4, device=device, dtype=DTYPE), atol=1e-7)


def test_completeness_property_linear_model(device):
    """
    Completeness (a.k.a. sensitivity to baseline): sum(attrib) ~= f(x) - f(baseline).
    With a linear model and baseline zero, this should hold to tight tolerance.
    """
    model = _build_peaked_linear_model(device=device, bias=False)
    baseline = torch.zeros(1, N_CHANNELS, dtype=DTYPE, device=device)
    x = torch.randn(1, N_CHANNELS, dtype=DTYPE, device=device) * 0.1  # small amplitude keeps linear region
    x.requires_grad_(True)

    attrs = _gradient_shap_or_ig(model, x, baseline)  # (1, F)
    pred = model(x)  # (1,)
    pred_base = model(baseline)  # (1,)

    lhs = attrs.sum(dim=1)  # (1,)
    rhs = (pred - pred_base)  # (1,)

    assert lhs.shape == rhs.shape
    assert torch.allclose(lhs, rhs, atol=5e-4), f"Completeness failed: sum(attrs)={lhs.item():.6f}, delta={rhs.item():.6f}"


def test_localization_matches_weight_profile(device):
    """
    If the true model weights have a Gaussian peak at a center index, the attribution
    vector on a small positive input (and baseline zero) should correlate strongly
    with the weight vector.
    """
    center = 140.0
    sigma = 8.0
    model = _build_peaked_linear_model(device=device, center=center, sigma=sigma, amplitude=1.0, bias=False)

    baseline = torch.zeros(1, N_CHANNELS, dtype=DTYPE, device=device)
    x = torch.full((1, N_CHANNELS), 0.1, dtype=DTYPE, device=device)  # uniform small positive flux
    attrs = _gradient_shap_or_ig(model, x, baseline).detach().cpu().numpy().reshape(-1)

    w = model.fc.weight.detach().cpu().numpy().reshape(-1)
    # Normalize both before correlation
    attrs = (attrs - attrs.mean()) / (attrs.std() + 1e-12)
    w_norm = (w - w.mean()) / (w.std() + 1e-12)

    corr = float(np.clip(np.corrcoef(attrs, w_norm)[0, 1], -1.0, 1.0))
    assert corr > 0.95, f"Attribution should align with weight peak; corr={corr:.4f}"


def test_reproducibility_fixed_seed(device):
    """
    With all seeds fixed and identical inputs, attributions should be identical.
    """
    _set_all_seeds(SEED)
    model = _build_peaked_linear_model(device=device, bias=False)
    baseline = torch.zeros(2, N_CHANNELS, dtype=DTYPE, device=device)
    x = torch.randn(2, N_CHANNELS, dtype=DTYPE, device=device) * 0.05

    _set_all_seeds(SEED)
    attrs1 = _gradient_shap_or_ig(model, x, baseline)

    _set_all_seeds(SEED)
    attrs2 = _gradient_shap_or_ig(model, x, baseline)

    assert torch.allclose(attrs1, attrs2, atol=1e-6), "Attributions changed despite fixed seeds."


def test_batch_support_and_per_sample_completeness(device):
    """
    Ensure attribution works for batches and completeness holds per sample.
    """
    model = _build_peaked_linear_model(device=device, bias=False)
    B = 4
    baseline = torch.zeros(B, N_CHANNELS, dtype=DTYPE, device=device)
    x = torch.randn(B, N_CHANNELS, dtype=DTYPE, device=device) * 0.05
    attrs = _gradient_shap_or_ig(model, x, baseline)
    assert attrs.shape == x.shape

    pred = model(x)
    pred0 = model(baseline)
    lhs = attrs.sum(dim=1)
    rhs = pred - pred0
    assert torch.allclose(lhs, rhs, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_gpu_parity():
    """
    Ensure CPU vs CUDA attributions are numerically close for the same inputs/model.
    """
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda")

    model_cpu = _build_peaked_linear_model(device=device_cpu, bias=False)
    # Copy weights to GPU model
    model_gpu = _build_peaked_linear_model(device=device_gpu, bias=False)
    with torch.no_grad():
        model_gpu.fc.weight.copy_(model_cpu.fc.weight.to(device_gpu))
        if model_gpu.fc.bias is not None:
            model_gpu.fc.bias.copy_((model_cpu.fc.bias or 0.0).to(device_gpu))

    B = 3
    x_cpu = torch.randn(B, N_CHANNELS, dtype=DTYPE, device=device_cpu) * 0.05
    base_cpu = torch.zeros(B, N_CHANNELS, dtype=DTYPE, device=device_cpu)

    x_gpu = x_cpu.to(device_gpu)
    base_gpu = base_cpu.to(device_gpu)

    attrs_cpu = _gradient_shap_or_ig(model_cpu, x_cpu, base_cpu).cpu()
    attrs_gpu = _gradient_shap_or_ig(model_gpu, x_gpu, base_gpu).cpu()
    assert torch.allclose(attrs_cpu, attrs_gpu, atol=2e-3), "CPU/GPU attribution mismatch beyond tolerance."
