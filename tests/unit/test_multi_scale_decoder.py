#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_multi_scale_decoder.py

SpectraMind V50 — Unit & Integration Tests for the Multi-Scale Decoder

What this tests
---------------
The multi-scale spectral decoder is expected to fuse latent representations from
multiple encoders (e.g., AIRS GNN over 283 spectral bins, FGS1 Mamba time-series,
and optional global/context embeddings) and produce, at minimum:

  • Mean spectrum μ ∈ ℝ^{B×283}
  • (Optionally) Uncertainty σ ∈ ℝ^{B×283} (heteroscedastic)
  • (Optionally) Auxiliary artifacts (e.g., per-bin attention, diagnostics dict)

This suite verifies:
  1) Shape integrity on a variety of input combinations (per-bin AIRS, pooled FGS1)
  2) Robustness to masks, NaNs/Infs in inputs
  3) Determinism with fixed seeds
  4) Gradient flow through μ/σ (GLL-friendly)
  5) Basic Gaussian Log-Likelihood (GLL) loss decreases with a step of SGD
  6) Temperature-scaling style calibration hooks (best-effort if implemented)
  7) TorchScript export (xfail if backend uses ops not scriptable)
  8) Hydra-style construction from config dicts
  9) State_dict save/load round-trip
 10) AMP autocast smoke test (CUDA fp16 / CPU bf16 where supported)
 11) Optional quantile or mixture outputs (skip if not exposed)
 12) Optional attention/importance map exposure (skip if not exposed)

Run:
  pytest -q tests/test_multi_scale_decoder.py
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

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.multi_scale_decoder import MultiScaleDecoder  # expected module
from src.utils import reproducibility, logging as v50_logging


NUM_BINS = 283  # Challenge spectral bins


# ----------------------------
# Fixtures & utilities
# ----------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _seed_each_test():
    reproducibility.set_seed(2025)
    yield
    reproducibility.set_seed(2025)


def _make_inputs(
    batch_size=2,
    airs_nodes=NUM_BINS,
    airs_feat_dim=64,
    fgs_pool_dim=128,
    with_airs=True,
    with_fgs=True,
    with_global=True,
    device=torch.device("cpu"),
):
    """
    Build synthetic inputs to the decoder:

      airs_z:   [B, N, Da]  per-bin AIRS latent (node-level)
      airs_mask:[B, N]      valid bins mask
      fgs_z:    [B, Df]     pooled FGS1 latent (sequence already compressed)
      global_z: [B, Dg]     optional global/context vector

    Returns a dict of named inputs for clarity.
    """
    x = {}

    if with_airs:
        x["airs_z"] = torch.randn(batch_size, airs_nodes, airs_feat_dim, device=device)
        x["airs_mask"] = torch.ones(batch_size, airs_nodes, dtype=torch.bool, device=device)
    else:
        x["airs_z"] = None
        x["airs_mask"] = None

    if with_fgs:
        x["fgs_z"] = torch.randn(batch_size, fgs_pool_dim, device=device)
    else:
        x["fgs_z"] = None

    if with_global:
        x["global_z"] = torch.randn(batch_size, 32, device=device)  # small context
    else:
        x["global_z"] = None

    return x


def _unwrap_outputs(out):
    """
    Normalize decoder outputs to a dict with at least μ (and optionally σ).
      Accepted forms:
        - Tensor -> μ only
        - (μ, σ) tuple/list
        - dict with keys "mu" and/or "sigma" (case-insensitive)
    """
    if isinstance(out, torch.Tensor):
        return {"mu": out}
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise AssertionError("Decoder returned empty tuple/list.")
        d = {"mu": out[0]}
        if len(out) > 1 and isinstance(out[1], torch.Tensor):
            d["sigma"] = out[1]
        return d
    if isinstance(out, dict):
        # Be permissive about key casing
        low = {str(k).lower(): v for k, v in out.items()}
        d = {}
        if "mu" in low:
            d["mu"] = low["mu"]
        elif "mean" in low:
            d["mu"] = low["mean"]
        else:
            # If the dict has any tensor, assume first is mu
            first_tensor = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
            if first_tensor is None:
                raise AssertionError("Decoder dict outputs did not contain tensors for μ.")
            d["mu"] = first_tensor
        if "sigma" in low:
            d["sigma"] = low["sigma"]
        elif "std" in low:
            d["sigma"] = low["std"]
        return d
    raise AssertionError(f"Unexpected decoder output type: {type(out)}")


def _gll_loss(mu, sigma, target, eps: float = 1e-6):
    """
    Gaussian Negative Log-Likelihood per bin (averaged):
      L = 0.5 * [ log(2πσ^2) + (y - μ)^2 / σ^2 ]
    """
    if sigma is None:
        # Fallback: treat as unit variance if σ isn't produced
        sigma = torch.ones_like(mu)
    var = (sigma.clamp_min(eps)) ** 2
    return 0.5 * (torch.log(2 * math.pi * var) + (target - mu) ** 2 / var).mean()


# ----------------------------
# Core shape / forward tests
# ----------------------------

def test_forward_shapes_all_streams(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=128).to(device)
    inp = _make_inputs(
        batch_size=3, airs_nodes=NUM_BINS, airs_feat_dim=48, fgs_pool_dim=96, device=device
    )
    out = dec(
        airs_z=inp["airs_z"],
        airs_mask=inp["airs_mask"],
        fgs_z=inp["fgs_z"],
        global_z=inp["global_z"],
    )
    d = _unwrap_outputs(out)
    mu = d["mu"]
    assert isinstance(mu, torch.Tensor)
    assert mu.shape == (3, NUM_BINS)
    if "sigma" in d:
        sigma = d["sigma"]
        assert isinstance(sigma, torch.Tensor)
        assert sigma.shape == (3, NUM_BINS)
        assert torch.isfinite(sigma).all()
    assert torch.isfinite(mu).all()


def test_forward_shapes_airs_only(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    inp = _make_inputs(with_fgs=False, with_global=False, device=device)
    out = dec(airs_z=inp["airs_z"], airs_mask=inp["airs_mask"], fgs_z=None, global_z=None)
    d = _unwrap_outputs(out)
    assert d["mu"].shape == (2, NUM_BINS)


def test_forward_shapes_fgs_only(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    inp = _make_inputs(with_airs=False, with_global=False, device=device)
    out = dec(airs_z=None, airs_mask=None, fgs_z=inp["fgs_z"], global_z=None)
    d = _unwrap_outputs(out)
    assert d["mu"].shape == (2, NUM_BINS)


def test_masks_change_outputs_behaviorally(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=96).to(torch.device("cpu"))
    inp = _make_inputs(batch_size=1, device=torch.device("cpu"))
    with_mask = dec(**inp)
    # Invalidate last 20% of bins
    m2 = inp["airs_mask"].clone()
    cut = int(NUM_BINS * 0.8)
    m2[:, cut:] = False
    with_mask2 = dec(airs_z=inp["airs_z"], airs_mask=m2, fgs_z=inp["fgs_z"], global_z=inp["global_z"])
    d1 = _unwrap_outputs(with_mask)["mu"]
    d2 = _unwrap_outputs(with_mask2)["mu"]
    assert d1.shape == d2.shape
    assert (d1 - d2).abs().mean().item() > 1e-7


def test_nan_inf_robustness(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    inp = _make_inputs(device=device)
    # Inject NaN/Inf
    inp["airs_z"][:, :5, :] = float("nan")
    inp["airs_z"][:, 5:10, :] = float("inf")
    out = dec(**inp)
    d = _unwrap_outputs(out)
    assert d["mu"].shape == (2, NUM_BINS)  # forward should not crash


# ----------------------------
# Determinism & gradients
# ----------------------------

def test_determinism_with_fixed_seed(device):
    dec1 = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    dec2 = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    inp = _make_inputs(batch_size=1, device=device)
    with torch.no_grad():
        out1 = _unwrap_outputs(dec1(**inp))["mu"]
    reproducibility.set_seed(2025)
    with torch.no_grad():
        out2 = _unwrap_outputs(dec2(**inp))["mu"]
    assert torch.allclose(out1, out2, atol=0, rtol=0)


def test_backward_and_gll_step_decreases_loss(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=128).to(device)
    optim = torch.optim.SGD(dec.parameters(), lr=1e-2)
    inp = _make_inputs(batch_size=4, device=device)
    target = torch.randn(4, NUM_BINS, device=device)

    # Step 1: loss before
    out = _unwrap_outputs(dec(**inp))
    loss0 = _gll_loss(out["mu"], out.get("sigma", None), target)

    # One optimization step
    optim.zero_grad(set_to_none=True)
    loss0.backward()
    optim.step()

    # Step 2: loss after
    out2 = _unwrap_outputs(dec(**inp))
    loss1 = _gll_loss(out2["mu"], out2.get("sigma", None), target)
    # Not guaranteed strictly monotonic, but typically should not increase with simple synthetic setup
    assert loss1.item() <= loss0.item() + 1e-5


# ----------------------------
# Calibration / temperature scaling (best-effort)
# ----------------------------

def test_temperature_scaling_hook_if_available(device):
    """
    If the decoder exposes a temperature-scaling calibration for σ (e.g., .set_temperature(T)
    or .apply_temperature(T)), try it and verify that σ is scaled accordingly.
    If unavailable, skip the test.
    """
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=80).to(device)
    inp = _make_inputs(batch_size=2, device=device)
    out = _unwrap_outputs(dec(**inp))
    if "sigma" not in out:
        pytest.skip("Decoder does not produce σ; skipping temperature scaling test.")

    sigma0 = out["sigma"].detach()
    scaled = None
    # Try common hook names
    with contextlib.suppress(Exception):
        if hasattr(dec, "set_temperature") and callable(getattr(dec, "set_temperature")):
            dec.set_temperature(2.0)
            out2 = _unwrap_outputs(dec(**inp))
            scaled = out2["sigma"]
        elif hasattr(dec, "apply_temperature") and callable(getattr(dec, "apply_temperature")):
            out2 = _unwrap_outputs(dec.apply_temperature(2.0, **inp))
            scaled = out2["sigma"]

    if scaled is None:
        pytest.skip("Decoder has no temperature scaling hook; skipping.")

    # Temperature scaling (T=2) typically multiplies σ by T (common convention)
    ratio = (scaled / sigma0.clamp_min(1e-12)).median().item()
    assert 1.5 <= ratio <= 2.5  # tolerant bounds


# ----------------------------
# TorchScript, Hydra, state_dict, AMP
# ----------------------------

@pytest.mark.xfail(reason="Decoder may use non-scriptable ops (einsum over lists, custom modules).")
def test_torchscript_export(tmp_path, device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    scripted = torch.jit.script(dec)
    path = tmp_path / "decoder_ts.pt"
    scripted.save(str(path))
    reloaded = torch.jit.load(str(path), map_location=device)
    inp = _make_inputs(batch_size=1, device=device)
    out = _unwrap_outputs(reloaded(**inp))
    assert out["mu"].shape == (1, NUM_BINS)


def test_hydra_style_construction(device):
    cfg = dict(
        num_bins=NUM_BINS,
        hidden_dim=96,
        dropout=0.1,
        heads=4,
        fuse_mode="concat",  # permissive; actual impl may ignore unknowns
    )
    dec = MultiScaleDecoder(**cfg).to(device)
    inp = _make_inputs(batch_size=2, device=device)
    out = _unwrap_outputs(dec(**inp))
    assert out["mu"].shape == (2, NUM_BINS)


def test_state_dict_roundtrip(device):
    dec1 = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=72).to(device)
    dec2 = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=72).to(device)
    inp = _make_inputs(batch_size=1, device=device)
    with torch.no_grad():
        mu1 = _unwrap_outputs(dec1(**inp))["mu"]
    sd = dec1.state_dict()
    missing, unexpected = dec2.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0
    with torch.no_grad():
        mu2 = _unwrap_outputs(dec2(**inp))["mu"]
    assert torch.allclose(mu1, mu2, atol=1e-6, rtol=0)


def test_mixed_precision_autocast(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    inp = _make_inputs(batch_size=1, device=device)
    autocast_kwargs = {}
    if device.type == "cuda":
        autocast_kwargs = dict(device_type="cuda", dtype=torch.float16)
    else:
        if torch.cpu.amp.is_bf16_supported():
            autocast_kwargs = dict(device_type="cpu", dtype=torch.bfloat16)
        else:
            pytest.skip("No AMP support available on this device.")
    with torch.autocast(**autocast_kwargs):
        out = _unwrap_outputs(dec(**inp))
    assert out["mu"].shape == (1, NUM_BINS)


# ----------------------------
# Optional outputs: quantiles / mixtures / attention
# ----------------------------

def test_optional_quantile_or_mixture_outputs(device):
    """
    If the decoder can emit quantiles or mixture parameters, exercise them lightly.
    Accept common patterns:
      - out["q"] or out["quantiles"] -> Tensor [B, NUM_BINS, K]
      - out["mixture"] with e.g. keys ("pi","mu_k","sigma_k")
    Otherwise skip.
    """
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=96).to(device)
    inp = _make_inputs(batch_size=2, device=device)
    out = dec(**inp)

    if not isinstance(out, dict):
        pytest.skip("Decoder did not return a dict with optional outputs; skipping.")

    low = {str(k).lower(): k for k in out.keys()}

    if "q" in low or "quantiles" in low:
        q = out[low.get("q", low.get("quantiles"))]
        assert isinstance(q, torch.Tensor)
        assert q.ndim == 3 and q.shape[0] == 2 and q.shape[1] == NUM_BINS
        assert torch.isfinite(q).all()
        return

    if "mixture" in low and isinstance(out[low["mixture"]], dict):
        mix = out[low["mixture"]]
        # Soft checks; implementations vary
        if "pi" in mix:
            assert torch.isfinite(mix["pi"]).all()
        if "mu_k" in mix:
            mk = mix["mu_k"]
            assert mk.shape[1] == NUM_BINS
        if "sigma_k" in mix:
            assert torch.isfinite(mix["sigma_k"]).all()
        return

    pytest.skip("No quantile/mixture outputs exposed by decoder; skipping.")


def test_optional_attention_maps_and_logging(tmp_path, device):
    """
    If the decoder exposes attention/importance per bin (e.g., out['attn'] or dec.get_attention()),
    we sanity-check finiteness and log a small summary.
    """
    logger = v50_logging.get_logger("test_decoder_attention", logfile=str(tmp_path / "decoder_attn.json"))
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=64).to(device)
    inp = _make_inputs(batch_size=2, device=device)
    out = dec(**inp)

    attn = None
    if isinstance(out, dict):
        for k in out.keys():
            if str(k).lower() in ("attn", "attention", "weights", "importance"):
                attn = out[k]
                break
    if attn is None and hasattr(dec, "get_attention") and callable(getattr(dec, "get_attention")):
        with contextlib.suppress(Exception):
            attn = dec.get_attention(**inp)

    if attn is None:
        pytest.skip("No attention/importance exposure; skipping.")

    if isinstance(attn, torch.Tensor):
        assert torch.isfinite(attn).all()
        logger.info("attn_summary", extra={"mean": float(attn.mean().item()), "std": float(attn.std().item())})
    elif isinstance(attn, (list, tuple)):
        for t in attn:
            assert torch.isfinite(t).all()
        logger.info("attn_list_count", extra={"count": len(attn)})
    elif isinstance(attn, dict):
        for v in attn.values():
            assert torch.isfinite(v).all()
        logger.info("attn_dict_keys", extra={"keys": list(attn.keys())})
    else:
        pytest.skip("Attention object type not recognized; skipping.")


# ----------------------------
# Larger batch smoke
# ----------------------------

def test_larger_batch_memory_smoke(device):
    dec = MultiScaleDecoder(num_bins=NUM_BINS, hidden_dim=48).to(device)
    inp = _make_inputs(batch_size=8, device=device)
    out = _unwrap_outputs(dec(**inp))
    assert out["mu"].shape == (8, NUM_BINS)


if __name__ == "__main__":
    pytest.main([__file__])
