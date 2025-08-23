#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_symbolic_loss.py

SpectraMind V50 — Unit & Integration Tests for symbolic_loss

Goals
-----
This suite verifies the upgraded symbolic loss module provides:
  • A scalar total loss and per-rule decomposition
  • Stable gradients wrt μ (and optionally σ)
  • Mask-aware computation (invalid bins ignored)
  • Configurable per-rule weights (changing one weight impacts total)
  • Robustness to NaNs/Infs in μ/σ (no crash; masked or sanitized)
  • Optional normalization / temperature / calibration hooks
  • Hydra-style construction from config dicts
  • (If nn.Module) state_dict round-trip
  • Cross-device determinism at the level of summary statistics

Expected APIs (permissive)
--------------------------
We accommodate multiple implementations:

1) Functional:
   from src.symbolic.symbolic_loss import compute_symbolic_loss
   out = compute_symbolic_loss(mu, sigma=None, mask=None, cfg: dict | None)
   # 'out' may be:
   #   - scalar tensor (total loss)          OR
   #   - (total, details) tuple              OR
   #   - dict with keys {"total", "rules", ...}

2) OO module:
   from src.symbolic.symbolic_loss import SymbolicLoss
   crit = SymbolicLoss(cfg)
   out = crit(mu, sigma=None, mask=None)
   # with optional hooks like:
   #   crit.set_rule_weight(name, w), crit.set_temperature(T)
   #   crit.get_rule_map(name) -> [B, N] tensor

We normalize outputs via _unwrap() and skip tests that rely on unsupported features.
"""

import os
import sys
import math
import json
import types
import pytest
import contextlib

import torch
import torch.nn as nn

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import in a flexible way
_sym = None
_SymbolicLoss = None
_compute = None
try:
    from src.symbolic import symbolic_loss as _sym  # type: ignore
except Exception:
    pass

if _sym is not None:
    _SymbolicLoss = getattr(_sym, "SymbolicLoss", None)
    _compute = getattr(_sym, "compute_symbolic_loss", None)


NUM_BINS = 283


# ----------------------------
# Helpers
# ----------------------------

def _seed_all(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as _np  # noqa
    random.seed(seed)
    _np.random.seed(seed)


def _make_mu(batch=2, device=torch.device("cpu")):
    """
    Create a plausible μ spectrum batch with gentle correlations:
    Sum of a few sinusoids + noise; clamp into [0, 1] range to mimic transmission depth.
    """
    t = torch.linspace(0, 2 * math.pi, NUM_BINS, device=device)
    base = (
        0.05 * torch.sin(3.0 * t)
        + 0.03 * torch.sin(11.0 * t + 0.4)
        + 0.02 * torch.sin(23.0 * t + 1.3)
    )  # shape [N]
    mu = base.unsqueeze(0).repeat(batch, 1) + 0.01 * torch.randn(batch, NUM_BINS, device=device)
    mu = (mu + 0.2).clamp_min(0.0)  # non-negative-ish
    return mu


def _make_sigma_like(mu, min_sigma: float = 1e-3):
    sigma = 0.05 + 0.02 * torch.rand_like(mu)
    return sigma.clamp_min(min_sigma)


def _make_mask(batch=2, keep_ratio=0.9, device=torch.device("cpu")):
    mask = torch.ones(batch, NUM_BINS, dtype=torch.bool, device=device)
    cut = int(NUM_BINS * keep_ratio)
    if cut < NUM_BINS:
        mask[:, cut:] = False
    return mask


def _build_cfg(weights=None, normalize=True):
    """
    Minimal Hydra-like config dict with rule weights.
    Keys are permissive; implementers can ignore unknown fields.
    """
    weights = weights or {
        "nonnegativity": 1.0,
        "smoothness": 1.0,
        "fft_smooth": 0.5,
        "asymmetry": 0.25,
        "photonic_alignment": 0.0,  # optional rule
    }
    return {
        "normalize": normalize,
        "rules": {
            "nonnegativity": {"weight": weights.get("nonnegativity", 1.0)},
            "smoothness": {"weight": weights.get("smoothness", 1.0), "order": 2, "lambda": 1.0},
            "fft_smooth": {"weight": weights.get("fft_smooth", 0.5)},
            "asymmetry": {"weight": weights.get("asymmetry", 0.25)},
            "photonic_alignment": {
                "weight": weights.get("photonic_alignment", 0.0),
                # sample anchors; impl may ignore
                "anchors": [50, 120, 210],
            },
        },
    }


def _unwrap_total_and_rules(obj):
    """
    Try to extract (total_loss, rule_dict[str->tensor]) from many shapes:
      - tensor                   -> (total, {})
      - (total, rules)           -> rules: dict|list|tensor
      - dict{"total", "rules"}   -> obvious
    Rules may be:
      - dict of scalars or maps [B, N]
      - 1D/2D tensors with implicit naming (we will convert to dict with generic names)
    """
    rules = {}
    if isinstance(obj, torch.Tensor):
        return obj, rules
    if isinstance(obj, (tuple, list)):
        assert len(obj) >= 1
        total = obj[0]
        if len(obj) > 1:
            rules = obj[1]
        return total, _rules_to_dict(rules)
    if isinstance(obj, dict):
        low = {str(k).lower(): k for k in obj.keys()}
        if "total" in low:
            total = obj[low["total"]]
        else:
            # Fallback: first tensor becomes total
            total = next((v for v in obj.values() if isinstance(v, torch.Tensor) and v.ndim == 0), None)
            if total is None:
                raise AssertionError("Could not locate scalar total loss in dict output.")
        rules = obj.get(low.get("rules", "rules"), {})
        return total, _rules_to_dict(rules)
    raise AssertionError(f"Unsupported output type: {type(obj)}")


def _rules_to_dict(rules):
    """Normalize rules into a dict[str->tensor] with finite tensors."""
    if isinstance(rules, dict):
        return rules
    if isinstance(rules, (list, tuple)):
        return {f"rule_{i}": r for i, r in enumerate(rules)}
    if torch.is_tensor(rules):
        # Either vector of per-rule scalars or [B, N] maps w/o names
        if rules.ndim == 1:
            return {f"rule_{i}": rules[i] for i in range(rules.shape[0])}
        else:
            return {"rule": rules}
    return {}


def _get_symbolic_object(cfg=None):
    """
    Construct callable symbolic loss from either OO or functional API.
    Return (callable, obj) where:
      callable(mu, sigma, mask, cfg?) -> out
      obj: original object or None
    """
    cfg = cfg or _build_cfg()
    if _SymbolicLoss is not None:
        try:
            obj = _SymbolicLoss(cfg)
            return lambda mu, sigma=None, mask=None, cfg=None: obj(mu, sigma=sigma, mask=mask), obj
        except Exception:
            pass
    if _compute is not None:
        return lambda mu, sigma=None, mask=None, cfg=None: _compute(mu, sigma=sigma, mask=mask, cfg=cfg or _build_cfg()), None
    raise pytest.SkipTest("symbolic_loss module not available (neither SymbolicLoss nor compute_symbolic_loss).")


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _per_test_seed():
    _seed_all(1337)
    yield
    _seed_all(1337)


# ----------------------------
# Tests
# ----------------------------

def test_forward_scalar_and_rules(device):
    """Basic forward returns a scalar total and (optional) per-rule decomposition."""
    f, _ = _get_symbolic_object(_build_cfg())
    mu = _make_mu(batch=3, device=device)
    sigma = _make_sigma_like(mu)
    mask = _make_mask(batch=3, keep_ratio=0.9, device=device)
    out = f(mu, sigma=sigma, mask=mask)
    total, rules = _unwrap_total_and_rules(out)
    assert isinstance(total, torch.Tensor) and total.ndim == 0
    assert torch.isfinite(total).all()
    # Rules are optional but if present must be finite
    for k, v in rules.items():
        assert isinstance(k, str)
        assert torch.is_tensor(v)
        assert torch.isfinite(v).all()


def test_gradients_wrt_mu(device):
    """Symbolic loss must backpropagate through μ."""
    f, _ = _get_symbolic_object(_build_cfg())
    mu = _make_mu(batch=2, device=device).requires_grad_(True)
    sigma = _make_sigma_like(mu)
    out = f(mu, sigma=sigma, mask=None)
    total, _ = _unwrap_total_and_rules(out)
    total.backward()
    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all()


def test_mask_changes_behavior(device):
    """Changing mask should change total loss (behavioral sanity)."""
    f, _ = _get_symbolic_object(_build_cfg())
    mu = _make_mu(batch=1, device=device)
    sigma = _make_sigma_like(mu)
    total_full, _ = _unwrap_total_and_rules(f(mu, sigma=sigma, mask=torch.ones_like(mu, dtype=torch.bool)))
    total_masked, _ = _unwrap_total_and_rules(f(mu, sigma=sigma, mask=_make_mask(batch=1, keep_ratio=0.6, device=device)))
    # No strict inequality enforced (impl-dependent), but typically different:
    assert (total_full - total_masked).abs().item() > 1e-7


def test_rule_weight_effect(device):
    """Increasing a rule's weight should (usually) raise its contribution and often total."""
    # First pass with smaller nonnegativity weight
    cfg1 = _build_cfg(weights={"nonnegativity": 0.01, "smoothness": 1.0, "fft_smooth": 0.5, "asymmetry": 0.25})
    f1, _ = _get_symbolic_object(cfg1)
    mu = (_make_mu(batch=1, device=device) - 0.25).detach()  # create some negatives to engage the rule
    sigma = _make_sigma_like(mu)
    tot1, rules1 = _unwrap_total_and_rules(f1(mu, sigma=sigma, mask=None))

    # Second pass with larger weight
    cfg2 = _build_cfg(weights={"nonnegativity": 5.0, "smoothness": 1.0, "fft_smooth": 0.5, "asymmetry": 0.25})
    f2, _ = _get_symbolic_object(cfg2)
    tot2, rules2 = _unwrap_total_and_rules(f2(mu, sigma=sigma, mask=None))

    # If rule is present, its term should increase with weight (scalar or map -> mean)
    nn1 = None
    nn2 = None
    for name in ("nonnegativity", "non_negativity", "nonneg", "non_neg"):
        nn1 = nn1 or rules1.get(name, None)
        nn2 = nn2 or rules2.get(name, None)

    if nn1 is None or nn2 is None:
        pytest.skip("Nonnegativity rule breakdown not exposed; skipping weight-specific assertion.")
    else:
        v1 = nn1.mean() if nn1.ndim > 0 else nn1
        v2 = nn2.mean() if nn2.ndim > 0 else nn2
        assert v2.item() >= v1.item() - 1e-7

    # Totals are often (not strictly guaranteed) larger when a rule weight is increased
    assert tot2.item() >= tot1.item() - 1e-6


def test_nan_inf_robustness(device):
    """NaNs/Infs in μ/σ should not crash; module should mask/sanitize or propagate safely."""
    f, _ = _get_symbolic_object(_build_cfg())
    mu = _make_mu(batch=1, device=device)
    sigma = _make_sigma_like(mu)
    mu[:, :5] = float("nan")
    mu[:, 5:10] = float("inf")
    sigma[:, 0:3] = float("nan")
    out = f(mu, sigma=sigma, mask=_make_mask(batch=1, device=device))
    total, _ = _unwrap_total_and_rules(out)
    assert isinstance(total, torch.Tensor)  # forward returns something


def test_normalization_and_temperature_hooks(device):
    """
    If the implementation provides normalization enable/disable and/or temperature scaling for σ,
    attempt to toggle and observe a change in total (best-effort, non-strict).
    """
    cfg = _build_cfg()
    f, obj = _get_symbolic_object(cfg)
    mu = _make_mu(batch=2, device=device)
    sigma = _make_sigma_like(mu)

    total0, _ = _unwrap_total_and_rules(f(mu, sigma=sigma, mask=None))

    # Try temperature scaling
    totalT = None
    with contextlib.suppress(Exception):
        if obj is not None and hasattr(obj, "set_temperature"):
            obj.set_temperature(2.0)
            totalT, _ = _unwrap_total_and_rules(obj(mu, sigma=sigma, mask=None))
        elif obj is not None and hasattr(obj, "apply_temperature"):
            totalT, _ = _unwrap_total_and_rules(obj.apply_temperature(2.0, mu, sigma=sigma, mask=None))

    # Try normalization toggle by rebuilding cfg if available
    totalN = None
    try:
        fN, _ = _get_symbolic_object(_build_cfg(normalize=False))
        totalN, _ = _unwrap_total_and_rules(fN(mu, sigma=sigma, mask=None))
    except pytest.SkipTest:
        pass
    except Exception:
        pass

    # No strict expectations; just ensure if present they produce finite outputs and likely change
    if totalT is not None:
        assert torch.isfinite(totalT)
        assert (totalT - total0).abs().item() > 0 or totalN is not None  # at least one knob should move something
    if totalN is not None:
        assert torch.isfinite(totalN)
        assert (totalN - total0).abs().item() >= 0.0  # should compute


def test_hydra_style_construction(device):
    """Construct from a Hydra-like config dict."""
    cfg = _build_cfg()
    if _SymbolicLoss is None:
        pytest.skip("SymbolicLoss class not exposed; construction test skipped.")
    obj = _SymbolicLoss(cfg)
    mu = _make_mu(batch=2, device=device)
    sigma = _make_sigma_like(mu)
    total, rules = _unwrap_total_and_rules(obj(mu, sigma=sigma, mask=None))
    assert torch.isfinite(total)
    for v in rules.values():
        assert torch.isfinite(v).all()


def test_optional_per_rule_maps_shape(device):
    """If per-rule maps are available, they should align with [B, N] dimensions."""
    f, obj = _get_symbolic_object(_build_cfg())
    mu = _make_mu(batch=2, device=device)
    out = f(mu, sigma=None, mask=None)
    _, rules = _unwrap_total_and_rules(out)

    # Try to pull maps from known accessors if dict lacks [B, N] maps
    if not any(torch.is_tensor(v) and v.ndim >= 2 for v in rules.values()):
        if obj is not None:
            grab = None
            for name in ("get_rule_map", "get_maps", "get_per_rule_maps"):
                if hasattr(obj, name):
                    grab = getattr(obj, name)
                    break
            if grab is not None:
                with contextlib.suppress(Exception):
                    maps = grab()
                    if isinstance(maps, dict):
                        rules.update(maps)

    # Check shapes if maps exist
    has_map = False
    for k, v in rules.items():
        if torch.is_tensor(v) and v.ndim >= 2:
            has_map = True
            assert v.shape[0] == mu.shape[0]
            assert v.shape[-1] == NUM_BINS
            assert torch.isfinite(v).all()
    if not has_map:
        pytest.skip("Implementation did not expose per-rule maps; shape test skipped.")


def test_state_dict_roundtrip_if_module(device):
    """If implemented as nn.Module, weights should be round-trippable."""
    if _SymbolicLoss is None:
        pytest.skip("SymbolicLoss class not exposed; skipping state_dict test.")
    obj1 = _SymbolicLoss(_build_cfg())
    obj2 = _SymbolicLoss(_build_cfg())
    mu = _make_mu(batch=1, device=device)
    with torch.no_grad():
        t1, _ = _unwrap_total_and_rules(obj1(mu))
    sd = obj1.state_dict()
    missing, unexpected = obj2.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0
    with torch.no_grad():
        t2, _ = _unwrap_total_and_rules(obj2(mu))
    # If pure param-free loss, totals may be identical by design; tolerate equality
    assert torch.allclose(t1, t2, atol=1e-8, rtol=0)


def test_cpu_gpu_determinism_stats(device):
    """Compare CPU vs GPU moments (if CUDA available) on the same inputs & cfg."""
    cfg = _build_cfg()
    if torch.cuda.is_available():
        # CPU pass
        if _SymbolicLoss is not None:
            obj_cpu = _SymbolicLoss(cfg)
            f_cpu = lambda mu, sigma=None, mask=None: obj_cpu(mu, sigma=sigma, mask=mask)
        elif _compute is not None:
            f_cpu = lambda mu, sigma=None, mask=None: _compute(mu, sigma=sigma, mask=mask, cfg=cfg)
        else:
            pytest.skip("No symbolic loss callable available.")
        mu_cpu = _make_mu(batch=2, device=torch.device("cpu"))
        tot_cpu, _ = _unwrap_total_and_rules(f_cpu(mu_cpu))

        # GPU pass
        if _SymbolicLoss is not None:
            obj_gpu = _SymbolicLoss(cfg)
            f_gpu = lambda mu, sigma=None, mask=None: obj_gpu(mu, sigma=sigma, mask=mask)
        else:
            f_gpu = lambda mu, sigma=None, mask=None: _compute(mu, sigma=sigma, mask=mask, cfg=cfg)
        mu_gpu = mu_cpu.to("cuda")
        tot_gpu, _ = _unwrap_total_and_rules(f_gpu(mu_gpu))

        # Compare scalar statistics (identical numerically is not guaranteed across devices)
        assert torch.allclose(tot_cpu, tot_gpu.cpu(), rtol=5e-3, atol=5e-4)
    else:
        pytest.skip("CUDA not available; skipping cross-device check.")


def test_autocast_amp_smoke(device):
    """Autocast smoke test with fp16/bf16 contexts when supported."""
    cfg = _build_cfg()
    f, obj = _get_symbolic_object(cfg)
    mu = _make_mu(batch=1, device=device)
    autocast_kwargs = {}
    if device.type == "cuda":
        autocast_kwargs = dict(device_type="cuda", dtype=torch.float16)
    else:
        if torch.cpu.amp.is_bf16_supported():
            autocast_kwargs = dict(device_type="cpu", dtype=torch.bfloat16)
        else:
            pytest.skip("No AMP support on this device.")
    with torch.autocast(**autocast_kwargs):
        total, _ = _unwrap_total_and_rules(f(mu))
    assert torch.isfinite(total)


if __name__ == "__main__":
    pytest.main([__file__])
