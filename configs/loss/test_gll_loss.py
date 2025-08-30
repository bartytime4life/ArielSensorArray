# tests/loss/test_gll_loss.py
# ==============================================================================
# Unit tests for Gaussian Log-Likelihood (GLL) loss configuration/behavior
# Target: configs/loss/gll.yaml semantics
#
# These tests implement a reference GLL loss in-Python (PyTorch) that mirrors the
# config options (sigma_min, reduction, weighting, symbolic hooks). They do not
# depend on repository-specific modules, enabling CI to validate math & config.
# ==============================================================================

import math
from pathlib import Path
import json
import torch
import pytest


def _apply_weighting(per_bin, scheme="uniform", custom=None, molecule_regions=None, entropy=None):
    """
    per_bin: (..., B) tensor of per-bin values
    scheme : 'uniform' | 'molecule' | 'entropy' | 'custom'
    custom : optional (..., B) tensor of weights if scheme == 'custom'
    molecule_regions: optional mask (..., B) with higher weights in molecule bands
    entropy: optional (..., B) weights proportional to entropy
    """
    if scheme == "uniform" or scheme is None:
        w = torch.ones_like(per_bin)
    elif scheme == "custom":
        assert custom is not None, "custom weighting requires a weight vector"
        w = custom
    elif scheme == "molecule":
        assert molecule_regions is not None, "molecule weighting requires molecule_regions mask"
        # Weight molecule regions higher; continuum = 1.0
        w = torch.ones_like(per_bin)
        w = w + 1.0 * molecule_regions  # molecule bins get +1 → weight 2.0
    elif scheme == "entropy":
        assert entropy is not None, "entropy weighting requires entropy weights"
        # Normalize entropy weights to mean ~1.0 to keep scale comparable
        eps = 1e-8
        e = entropy / (entropy.mean() + eps)
        w = e
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return per_bin * w, w


def gll_loss(mu, sigma, y, cfg=None, molecule_regions=None, entropy_w=None, custom_w=None):
    """
    Reference Gaussian log-likelihood:
      L = 0.5 * Σ [ log(2πσ^2) + (y-μ)^2 / σ^2 ]
    Implements:
      - sigma_min clamp
      - optional detach_sigma_grad
      - weighting schemes
      - reductions: mean/sum/none
      - symbolic hooks (smoothness, nonnegativity, fft) as additive regularizers
    """
    if cfg is None:
        cfg = {}
    gll_cfg = cfg.get("gll", cfg)

    reduction = gll_cfg.get("reduction", "mean")
    sigma_min = float(gll_cfg.get("sigma_min", 1e-4))
    clamp = bool(gll_cfg.get("clamp_sigma", True))
    detach_sigma_grad = bool(gll_cfg.get("detach_sigma_grad", False))
    weighting = gll_cfg.get("weighting", "uniform")

    # Stabilize sigma
    if clamp:
        sigma_eff = torch.clamp(sigma, min=sigma_min)
    else:
        sigma_eff = sigma
    if detach_sigma_grad:
        sigma_eff = sigma_eff.detach()

    var = sigma_eff ** 2
    # GLL per-bin
    per_bin = 0.5 * (torch.log(2 * math.pi * var) + (y - mu) ** 2 / (var + 1e-12))

    # Weighting
    per_bin, weights_used = _apply_weighting(
        per_bin,
        scheme=weighting,
        custom=custom_w,
        molecule_regions=molecule_regions,
        entropy=entropy_w,
    )

    # Symbolic/physics hooks (optional)
    sym = gll_cfg.get("symbolic", {})
    # Smoothness penalty ~ L2 curvature on mu across bins
    smooth_lambda = float(sym.get("smoothness_penalty", 0.0))
    smooth_pen = 0.0
    if smooth_lambda > 0.0 and mu.shape[-1] >= 3:
        mu_left = mu[..., :-2]
        mu_mid = mu[..., 1:-1]
        mu_right = mu[..., 2:]
        curvature = (mu_right - 2 * mu_mid + mu_left) ** 2
        # pad to match length
        pad_left = torch.zeros_like(mu[..., :1])
        pad_right = torch.zeros_like(mu[..., :1])
        curvature_full = torch.cat([pad_left, curvature, pad_right], dim=-1)
        smooth_pen = smooth_lambda * curvature_full

    # Nonnegativity (soft hinge) on mu
    nonneg = bool(sym.get("nonnegativity", False))
    nonneg_pen = 0.0
    if nonneg:
        # quadratic hinge on negative μ
        nonneg_pen = (torch.relu(-mu)) ** 2

    # FFT suppression — simple high-frequency L2 (optional)
    fft_w = float(sym.get("fft_suppression", 0.0))
    fft_pen = 0.0
    if fft_w > 0.0:
        # naive DFT magnitude (real-only simplification for test)
        spec = torch.fft.rfft(mu, dim=-1)
        # weight high-frequency components linearly with frequency index
        freq_idx = torch.arange(spec.shape[-1], device=spec.device, dtype=spec.real.dtype)
        freq_w = freq_idx / (freq_idx.max().clamp(min=1))
        fft_mag2 = (spec.real**2 + spec.imag**2) * freq_w
        # Broadcast to original bin-length through irfft-equivalent length scaling
        fft_pen = fft_w * torch.nn.functional.pad(fft_mag2, (0, mu.shape[-1] - fft_mag2.shape[-1]), value=0.0)

    total_per_bin = per_bin + smooth_pen + nonneg_pen + fft_pen

    if reduction == "mean":
        return total_per_bin.mean()
    elif reduction == "sum":
        return total_per_bin.sum()
    elif reduction == "none":
        return total_per_bin
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def test_gll_basic_shapes_and_reduction():
    torch.manual_seed(0)
    B = 4
    bins = 8
    mu = torch.zeros(B, bins)
    sigma = torch.ones(B, bins) * 0.5
    y = torch.randn(B, bins) * 0.1

    # mean reduction
    cfg = {"gll": {"reduction": "mean", "sigma_min": 1e-4}}
    v_mean = gll_loss(mu, sigma, y, cfg)
    assert v_mean.ndim == 0

    # sum reduction
    cfg["gll"]["reduction"] = "sum"
    v_sum = gll_loss(mu, sigma, y, cfg)
    assert torch.isclose(v_sum, v_mean * B * bins, rtol=1e-4, atol=1e-4)

    # none reduction
    cfg["gll"]["reduction"] = "none"
    v_none = gll_loss(mu, sigma, y, cfg)
    assert v_none.shape == (B, bins)


def test_gll_sigma_clamping_and_detach():
    torch.manual_seed(0)
    mu = torch.zeros(2, 5)
    y = torch.zeros(2, 5)
    # dangerously small sigma
    sigma = torch.full((2, 5), 1e-8, requires_grad=True)

    cfg = {"gll": {"reduction": "sum", "sigma_min": 1e-4, "clamp_sigma": True, "detach_sigma_grad": False}}
    loss = gll_loss(mu, sigma, y, cfg)
    loss.backward()
    # Gradient should exist on sigma (not detached)
    assert sigma.grad is not None

    sigma.grad.zero_()
    cfg["gll"]["detach_sigma_grad"] = True
    loss2 = gll_loss(mu, sigma, y, cfg)
    loss2.backward()
    # Now sigma grad should be None or zero because we detached
    assert sigma.grad is None or torch.allclose(sigma.grad, torch.zeros_like(sigma.grad))


def test_gll_weighting_schemes_custom_and_molecule():
    torch.manual_seed(0)
    B, bins = 2, 6
    mu = torch.zeros(B, bins)
    sigma = torch.ones(B, bins)
    y = torch.zeros(B, bins)

    # custom weights: emphasize latter half
    custom_w = torch.cat([torch.ones(bins // 2), torch.ones(bins - bins // 2) * 2.0]).repeat(B, 1)
    cfg = {"gll": {"reduction": "sum", "weighting": "custom"}}
    loss_custom = gll_loss(mu, sigma, y, cfg, custom_w=custom_w)

    # molecule mask: latter half as molecule bins → weight ~2.0
    mol_mask = torch.zeros(B, bins)
    mol_mask[:, bins // 2 :] = 1.0
    cfg2 = {"gll": {"reduction": "sum", "weighting": "molecule"}}
    loss_molecule = gll_loss(mu, sigma, y, cfg2, molecule_regions=mol_mask)

    # Both strategies should up-weight latter bins; magnitudes should be similar
    assert loss_molecule > 0 and loss_custom > 0
    ratio = float(loss_custom / loss_molecule)
    assert 0.5 <= ratio <= 2.0


def test_gll_symbolic_hooks_smoothness_nonneg_fft_no_crash_and_effects():
    torch.manual_seed(0)
    B, bins = 1, 16
    # create oscillatory μ that should incur smoothness + fft penalty and some negatives
    x = torch.linspace(0, 4 * math.pi, bins)
    mu = 0.2 * torch.sin(x).unsqueeze(0)
    y = torch.zeros_like(mu)
    sigma = torch.ones_like(mu) * 0.2

    cfg = {
        "gll": {
            "reduction": "sum",
            "sigma_min": 1e-4,
            "symbolic": {"smoothness_penalty": 1.0, "nonnegativity": True, "fft_suppression": 0.1},
        }
    }
    loss = gll_loss(mu, sigma, y, cfg)
    # Turning off symbolic should reduce loss (remove regularizers)
    cfg_off = {"gll": {"reduction": "sum", "sigma_min": 1e-4}}
    loss_off = gll_loss(mu, sigma, y, cfg_off)
    assert loss > loss_off


def test_gll_save_config_json(tmp_path: Path = None):
    # Simulate saving config for diagnostics (as per config flags)
    cfg = {
        "gll": {
            "reduction": "mean",
            "sigma_min": 1e-4,
            "weighting": "uniform",
            "symbolic": {"smoothness_penalty": 0.0, "nonnegativity": False, "fft_suppression": 0.0},
            "save_json": True,
        }
    }
    if tmp_path is None:
        tmp_path = Path.cwd() / "tmp_gll_test"
    tmp_path.mkdir(exist_ok=True)
    out = tmp_path / "gll_loss_config.json"
    with out.open("w") as f:
        json.dump(cfg, f, indent=2)
    assert out.exists()
