```python
# tests/loss/test_gll_loss.py
# ==============================================================================
# Upgraded Unit Tests for Gaussian Log-Likelihood (GLL) loss configuration/behavior
# Target: configs/loss/gll.yaml semantics
#
# This suite:
#   • Implements a reference GLL loss (pure PyTorch) mirroring config options.
#   • Covers stability (σ clamp), reductions, weighting schemes, dtypes/devices.
#   • Verifies gradients wrt μ/σ, and validates symbolic hooks (smoothness/FFT/nonneg).
#   • Exercises broadcasting and JSON export paths used by CI diagnostics.
#
# The tests do NOT import repository code; they form a spec for config semantics.
# ==============================================================================

import math
import json
from pathlib import Path

import pytest
import torch


# ----------------------------
# Helpers
# ----------------------------
def torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_weighting(per_bin, scheme="uniform", custom=None, molecule_regions=None, entropy=None):
    """
    Apply per-bin weights.

    Args:
        per_bin: (..., B) tensor
        scheme: 'uniform' | 'molecule' | 'entropy' | 'custom'
        custom: (..., B) tensor if scheme == 'custom'
        molecule_regions: (..., B) binary mask (1 for molecule bins) if scheme == 'molecule'
        entropy: (..., B) positive weights if scheme == 'entropy'

    Returns:
        weighted_per_bin, weights
    """
    if scheme in (None, "uniform"):
        w = torch.ones_like(per_bin)
    elif scheme == "custom":
        assert custom is not None, "custom weighting requires a weight vector"
        w = custom
    elif scheme == "molecule":
        assert molecule_regions is not None, "molecule weighting requires molecule_regions mask"
        w = torch.ones_like(per_bin) + molecule_regions  # mol bins doubled
    elif scheme == "entropy":
        assert entropy is not None, "entropy weighting requires entropy weights"
        eps = torch.finfo(per_bin.dtype).eps
        w = entropy / (entropy.mean() + eps)
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return per_bin * w, w


def gll_loss(mu, sigma, y, cfg=None, molecule_regions=None, entropy_w=None, custom_w=None):
    """
    Reference Gaussian log-likelihood with optional symbolic hooks.

    L = 0.5 * Σ [ log(2πσ²) + (y-μ)² / σ² ]

    Config keys under 'gll':
      - reduction: 'mean'|'sum'|'none'
      - sigma_min: float
      - clamp_sigma: bool
      - detach_sigma_grad: bool
      - weighting: 'uniform'|'molecule'|'entropy'|'custom'
      - symbolic: dict with:
          • smoothness_penalty: float (curvature)
          • nonnegativity: bool
          • fft_suppression: float
    """
    if cfg is None:
        cfg = {}
    gll_cfg = cfg.get("gll", cfg)

    reduction = gll_cfg.get("reduction", "mean")
    sigma_min = float(gll_cfg.get("sigma_min", 1e-4))
    clamp = bool(gll_cfg.get("clamp_sigma", True))
    detach_sigma_grad = bool(gll_cfg.get("detach_sigma_grad", False))
    weighting = gll_cfg.get("weighting", "uniform")

    if clamp:
        sigma_eff = torch.clamp(sigma, min=sigma_min)
    else:
        sigma_eff = sigma
    if detach_sigma_grad:
        sigma_eff = sigma_eff.detach()

    var = sigma_eff**2
    eps = torch.finfo(mu.dtype).eps
    per_bin = 0.5 * (torch.log(2 * math.pi * var + eps) + (y - mu) ** 2 / (var + eps))

    # Weighting
    per_bin, _w = _apply_weighting(
        per_bin,
        scheme=weighting,
        custom=custom_w,
        molecule_regions=molecule_regions,
        entropy=entropy_w,
    )

    # Symbolic hooks
    sym = gll_cfg.get("symbolic", {})

    # Smoothness (curvature) penalty
    smooth_lambda = float(sym.get("smoothness_penalty", 0.0))
    if smooth_lambda > 0.0 and mu.shape[-1] >= 3:
        mu_left = mu[..., :-2]
        mu_mid = mu[..., 1:-1]
        mu_right = mu[..., 2:]
        curvature = (mu_right - 2 * mu_mid + mu_left) ** 2
        pad_left = torch.zeros_like(mu[..., :1])
        pad_right = torch.zeros_like(mu[..., :1])
        curvature_full = torch.cat([pad_left, curvature, pad_right], dim=-1)
        per_bin = per_bin + smooth_lambda * curvature_full

    # Nonnegativity (soft hinge)
    if bool(sym.get("nonnegativity", False)):
        per_bin = per_bin + (torch.relu(-mu)) ** 2

    # FFT suppression (frequency-weighted magnitude)
    fft_w = float(sym.get("fft_suppression", 0.0))
    if fft_w > 0.0:
        spec = torch.fft.rfft(mu, dim=-1)
        # freq 0..Nf-1 normalized
        freq_idx = torch.arange(spec.shape[-1], device=spec.device, dtype=mu.real.dtype)
        denom = torch.maximum(freq_idx.max(), torch.tensor(1.0, device=spec.device, dtype=mu.real.dtype))
        freq_w = freq_idx / denom
        fft_mag2 = (spec.real**2 + spec.imag**2) * freq_w
        pad = mu.shape[-1] - fft_mag2.shape[-1]
        fft_term = torch.nn.functional.pad(fft_mag2, (0, pad), value=0.0)
        per_bin = per_bin + fft_w * fft_term

    if reduction == "mean":
        return per_bin.mean()
    if reduction == "sum":
        return per_bin.sum()
    if reduction == "none":
        return per_bin
    raise ValueError(f"Unknown reduction: {reduction}")


# ----------------------------
# Tests
# ----------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gll_basic_shapes_and_reduction(dtype):
    device = torch_device()
    torch.manual_seed(0)
    B, bins = 4, 8
    mu = torch.zeros(B, bins, device=device, dtype=dtype)
    sigma = torch.ones(B, bins, device=device, dtype=dtype) * 0.5
    y = torch.randn(B, bins, device=device, dtype=dtype) * 0.1

    cfg = {"gll": {"reduction": "mean", "sigma_min": 1e-4}}
    v_mean = gll_loss(mu, sigma, y, cfg)
    assert v_mean.ndim == 0

    cfg["gll"]["reduction"] = "sum"
    v_sum = gll_loss(mu, sigma, y, cfg)
    assert torch.isfinite(v_sum)
    assert torch.isclose(v_sum, v_mean * B * bins, rtol=1e-4, atol=1e-4)

    cfg["gll"]["reduction"] = "none"
    v_none = gll_loss(mu, sigma, y, cfg)
    assert v_none.shape == (B, bins)


def test_gll_matches_analytic_when_mu_equals_y():
    device = torch_device()
    dtype = torch.float64
    B, bins = 3, 7
    mu = torch.randn(B, bins, device=device, dtype=dtype)
    y = mu.clone()  # zero residuals
    sigma = torch.rand(B, bins, device=device, dtype=dtype) * 0.9 + 0.1
    cfg = {"gll": {"reduction": "sum", "sigma_min": 1e-12, "clamp_sigma": True}}
    val = gll_loss(mu, sigma, y, cfg)
    analytic = 0.5 * torch.sum(torch.log(2 * math.pi * sigma**2 + torch.finfo(dtype).eps))
    assert torch.isclose(val, analytic, rtol=1e-6, atol=1e-6)


def test_gll_sigma_clamping_and_detach():
    device = torch_device()
    dtype = torch.float32
    mu = torch.zeros(2, 5, device=device, dtype=dtype)
    y = torch.zeros(2, 5, device=device, dtype=dtype)
    sigma = torch.full((2, 5), 1e-12, device=device, dtype=dtype, requires_grad=True)

    cfg = {"gll": {"reduction": "sum", "sigma_min": 1e-4, "clamp_sigma": True, "detach_sigma_grad": False}}
    loss = gll_loss(mu, sigma, y, cfg)
    loss.backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()

    sigma.grad.zero_()
    cfg["gll"]["detach_sigma_grad"] = True
    loss2 = gll_loss(mu, sigma, y, cfg)
    loss2.backward()
    assert sigma.grad is None or torch.allclose(sigma.grad, torch.zeros_like(sigma.grad))


@pytest.mark.parametrize("scheme", ["uniform", "custom", "molecule", "entropy"])
def test_gll_weighting_schemes(scheme):
    device = torch_device()
    torch.manual_seed(0)
    B, bins = 2, 10
    mu = torch.zeros(B, bins, device=device)
    sigma = torch.ones(B, bins, device=device)
    y = torch.zeros(B, bins, device=device)

    cfg = {"gll": {"reduction": "sum", "weighting": scheme}}
    kwargs = {}
    if scheme == "custom":
        kwargs["custom_w"] = torch.linspace(0.5, 1.5, steps=bins, device=device).repeat(B, 1)
    if scheme == "molecule":
        mask = torch.zeros(B, bins, device=device)
        mask[..., bins // 2 :] = 1.0
        kwargs["molecule_regions"] = mask
    if scheme == "entropy":
        ent = torch.rand(B, bins, device=device) + 0.1
        kwargs["entropy_w"] = ent

    loss = gll_loss(mu, sigma, y, cfg, **kwargs)
    assert torch.isfinite(loss)
    assert loss > 0.0


def test_gll_symbolic_hooks_smoothness_nonneg_fft():
    device = torch_device()
    torch.manual_seed(0)
    B, bins = 1, 32
    x = torch.linspace(0, 8 * math.pi, bins, device=device)
    mu = 0.5 * torch.sin(4 * x).unsqueeze(0)  # high frequency, negatives present
    y = torch.zeros_like(mu)
    sigma = torch.full_like(mu, 0.2)

    cfg_off = {"gll": {"reduction": "sum", "sigma_min": 1e-4}}
    base = gll_loss(mu, sigma, y, cfg_off)

    cfg = {
        "gll": {
            "reduction": "sum",
            "sigma_min": 1e-4,
            "symbolic": {"smoothness_penalty": 1.0, "nonnegativity": True, "fft_suppression": 0.5},
        }
    }
    reg = gll_loss(mu, sigma, y, cfg)

    assert reg > base  # adding penalties should increase loss


def test_smoothness_penalty_distinguishes_oscillation():
    device = torch_device()
    bins = 64
    x = torch.linspace(0, 4 * math.pi, bins, device=device)
    mu_osc = 0.2 * torch.sin(6 * x).unsqueeze(0)  # more oscillatory
    mu_smooth = 0.2 * torch.sin(1 * x).unsqueeze(0)
    y = torch.zeros_like(mu_osc)
    sigma = torch.ones_like(mu_osc) * 0.3
    cfg = {"gll": {"reduction": "sum", "symbolic": {"smoothness_penalty": 1.0}}}
    loss_osc = gll_loss(mu_osc, sigma, y, cfg)
    loss_smooth = gll_loss(mu_smooth, sigma, y, cfg)
    assert loss_osc > loss_smooth


def test_fft_penalty_distinguishes_frequency_content():
    device = torch_device()
    bins = 64
    x = torch.linspace(0, 4 * math.pi, bins, device=device)
    mu_low = 0.2 * torch.sin(1 * x).unsqueeze(0)
    mu_high = 0.2 * torch.sin(10 * x).unsqueeze(0)
    y = torch.zeros_like(mu_low)
    sigma = torch.ones_like(mu_low) * 0.2
    cfg = {"gll": {"reduction": "sum", "symbolic": {"fft_suppression": 0.5}}}
    loss_low = gll_loss(mu_low, sigma, y, cfg)
    loss_high = gll_loss(mu_high, sigma, y, cfg)
    assert loss_high > loss_low


def test_nonnegativity_penalty_zero_for_positive():
    device = torch_device()
    mu_pos = torch.rand(2, 16, device=device)  # >=0
    mu_mixed = mu_pos.clone()
    mu_mixed[..., :4] = -torch.abs(mu_mixed[..., :4])  # inject negatives
    y = torch.zeros_like(mu_pos)
    sigma = torch.ones_like(mu_pos) * 0.5

    cfg_on = {"gll": {"reduction": "sum", "symbolic": {"nonnegativity": True}}}
    cfg_off = {"gll": {"reduction": "sum"}}

    loss_pos_on = gll_loss(mu_pos, sigma, y, cfg_on)
    loss_pos_off = gll_loss(mu_pos, sigma, y, cfg_off)
    loss_mix_on = gll_loss(mu_mixed, sigma, y, cfg_on)

    assert torch.isclose(loss_pos_on, loss_pos_off, rtol=1e-6, atol=1e-6)  # no penalty needed
    assert loss_mix_on > loss_pos_on


def test_gradients_wrt_mu_exist_and_finite():
    device = torch_device()
    torch.manual_seed(0)
    mu = torch.randn(3, 12, device=device, requires_grad=True)
    y = torch.randn_like(mu)
    sigma = torch.rand_like(mu) * 0.5 + 0.2
    cfg = {"gll": {"reduction": "mean", "sigma_min": 1e-6}}
    loss = gll_loss(mu, sigma, y, cfg)
    loss.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()


def test_broadcasting_sigma_and_y():
    device = torch_device()
    B, bins = 5, 24
    mu = torch.zeros(B, bins, device=device)
    sigma = torch.full((B, 1), 0.3, device=device)  # broadcast across bins
    y = torch.zeros(1, bins, device=device)         # broadcast across batch
    cfg = {"gll": {"reduction": "sum"}}
    val = gll_loss(mu, sigma, y, cfg)
    assert torch.isfinite(val) and val.ndim == 0


def test_dtype_compatibility_and_numerical_stability():
    device = torch_device()
    for dtype in (torch.float32, torch.float64):
        mu = torch.randn(2, 10, device=device, dtype=dtype)
        y = torch.randn_like(mu)
        # small sigma but clamped
        sigma = (torch.rand_like(mu) * 1e-8).to(dtype)
        cfg = {"gll": {"reduction": "mean", "sigma_min": 1e-4, "clamp_sigma": True}}
        val = gll_loss(mu, sigma, y, cfg)
        assert torch.isfinite(val)


def test_config_json_save(tmp_path: Path):
    cfg = {
        "gll": {
            "reduction": "mean",
            "sigma_min": 1e-4,
            "weighting": "uniform",
            "symbolic": {"smoothness_penalty": 0.0, "nonnegativity": False, "fft_suppression": 0.0},
            "save_json": True,
        }
    }
    out = tmp_path / "gll_loss_config.json"
    with out.open("w") as f:
        json.dump(cfg, f, indent=2)
    assert out.exists() and out.read_text().strip().startswith("{")
```
