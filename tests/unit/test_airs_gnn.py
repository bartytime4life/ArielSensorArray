#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_airs_gnn.py

Unit & integration tests for the AIRS GNN encoder in SpectraMind V50.

This suite validates:
• Forward pass on realistic AIRS-shaped inputs (283 spectral bins)
• Support for edge indices and edge attributes (distance / type)
• Optional masks for valid bins and robustness to NaNs/Infs
• Determinism with fixed seeds across CPU/GPU where applicable
• Gradient flow & backward pass
• TorchScript export (best-effort; xfail if backend not scriptable)
• Hydra-style construction from config dicts
• State_dict save/load round-trip
• Optional attention/edge-weight extraction hooks (skip if unavailable)

Run:
  pytest -q tests/test_airs_gnn.py
"""

import os
import sys
import math
import json
import types
import pytest
import random
import tempfile
import contextlib

import torch
import numpy as np

# Ensure the repository root is in sys.path for local imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Project-specific imports (paths reflect SpectraMind V50 layout).
from src.models.airs_gnn import AIRSGNNEncoder
from src.utils import reproducibility, logging as v50_logging


# ----------------------------
# Fixtures & helpers
# ----------------------------

NUM_BINS = 283  # AIRS spectral bins

@pytest.fixture(scope="module")
def device():
    """Prefer CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _seed_every_test():
    """Deterministic seeding before each test."""
    reproducibility.set_seed(1337)
    yield
    reproducibility.set_seed(1337)


def _build_linear_chain_edges(n: int):
    """Create a simple linear chain over n bins: edges (i -> i+1) and (i+1 -> i)."""
    src = torch.arange(0, n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    # Undirected by doubling
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)  # [2, 2*(n-1)]
    # Distances: absolute wavelength bin distance (1 for neighbors)
    dist = torch.ones(edge_index.shape[1], 1, dtype=torch.float32)
    return edge_index, dist


def _build_cluster_edges(n: int, cluster_size: int = 7, p_intra: float = 0.6):
    """
    Build edges that mimic molecule-band clusters:
    - Partition bins into clusters of size ~cluster_size
    - Higher probability of edges within a cluster (p_intra)
    - Some sparse inter-cluster edges
    """
    clusters = [list(range(i, min(i + cluster_size, n))) for i in range(0, n, cluster_size)]
    edges = set()
    rng = random.Random(99)

    for c in clusters:
        # Dense-ish intra-cluster
        for i in c:
            for j in c:
                if i != j and rng.random() < p_intra:
                    edges.add((i, j))
        # Sparse inter-cluster to the next cluster head (if exists)
        head = c[0]
        nxt = head + cluster_size
        if nxt < n:
            edges.add((head, nxt))
            edges.add((nxt, head))

    if not edges:
        edges.add((0, 1))

    edge_index = torch.tensor(list(edges), dtype=torch.long).T  # [2, E]
    # Distance attribute (bin difference) and a simple "type" channel (intra vs inter)
    diff = (edge_index[0] - edge_index[1]).abs().float().unsqueeze(1)
    intra = (diff <= (cluster_size - 1)).float()
    edge_attr = torch.cat([diff, intra], dim=1)  # [E, 2]
    return edge_index, edge_attr


def make_synthetic_airs_batch(
    batch_size: int = 2,
    num_bins: int = NUM_BINS,
    feat_dim: int = 1,
    with_edges: bool = True,
    cluster_edges: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """
    Construct a synthetic AIRS batch:
      x: [B, N, F] spectral features (e.g., normalized transit depth / flux)
      mask: [B, N] valid-bins mask
      edge_index: [2, E] (optional)
      edge_attr: [E, A] (optional)
    """
    x = torch.randn(batch_size, num_bins, feat_dim, device=device)
    mask = torch.ones(batch_size, num_bins, dtype=torch.bool, device=device)

    if not with_edges:
        return x, mask, None, None

    if cluster_edges:
        edge_index, edge_attr = _build_cluster_edges(num_bins)
    else:
        edge_index, dist = _build_linear_chain_edges(num_bins)
        edge_attr = dist  # single distance channel

    return x, mask, edge_index.to(device), edge_attr.to(device)


# ----------------------------
# Tests
# ----------------------------

def test_forward_shape_default(device):
    """Basic forward on synthetic AIRS batch -> shape sanity and finiteness."""
    encoder = AIRSGNNEncoder(hidden_dim=128, num_layers=3).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=3, feat_dim=2, device=device
    )
    out = encoder(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    assert isinstance(out, torch.Tensor)
    # Accept either pooled [B, D] or per-bin embeddings [B, N, D]
    if out.ndim == 2:
        assert out.shape[0] == x.shape[0]
        assert out.shape[1] >= 16
    elif out.ndim == 3:
        assert out.shape[:2] == x.shape[:2]
        assert out.shape[2] >= 16
    else:
        raise AssertionError(f"Unexpected output ndim: {out.ndim}")
    assert torch.isfinite(out).all()


def test_backward_and_grad_flow(device):
    """Ensure gradients propagate from pooled/per-bin outputs back to inputs."""
    encoder = AIRSGNNEncoder(hidden_dim=64, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=2, feat_dim=3, device=device
    )
    x.requires_grad_(True)
    out = encoder(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mask_effect_behavioral(device):
    """If mask zeroes out bins, outputs should differ from full-mask forward."""
    encoder = AIRSGNNEncoder(hidden_dim=64, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=2, device=device
    )
    with_mask = encoder(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    # Invalidate last 20% of bins
    mask2 = mask.clone()
    cut = int(0.8 * NUM_BINS)
    mask2[:, cut:] = False
    with_mask2 = encoder(x, mask=mask2, edge_index=edge_index, edge_attr=edge_attr)
    # Behaviorally different but both finite
    assert torch.isfinite(with_mask).all() and torch.isfinite(with_mask2).all()
    diff = (with_mask - with_mask2).abs().mean().item()
    assert diff > 1e-7


def test_nan_inputs_are_handled(device):
    """Ensure NaNs/Infs do not crash forward (output may contain NaN—acceptable)."""
    encoder = AIRSGNNEncoder(hidden_dim=48, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=1, device=device
    )
    x[:, :10, :] = float("nan")
    x[:, 10:20, :] = float("inf")
    out = encoder(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    assert out.shape[0] == x.shape[0]


def test_determinism_cpu_vs_gpu_stats(device):
    """If CUDA is available, CPU vs GPU means/std should be close on same seed."""
    encoder_cpu = AIRSGNNEncoder(hidden_dim=32, num_layers=2).cpu()
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=2, device=torch.device("cpu")
    )
    with torch.no_grad():
        out_cpu = encoder_cpu(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    if device.type == "cuda":
        encoder_gpu = AIRSGNNEncoder(hidden_dim=32, num_layers=2).to(device)
        xg = x.to(device)
        maskg = mask.to(device)
        eig = edge_index.to(device) if edge_index is not None else None
        eag = edge_attr.to(device) if edge_attr is not None else None
        with torch.no_grad():
            out_gpu = encoder_gpu(xg, mask=maskg, edge_index=eig, edge_attr=eag)
        # Compare first two moments instead of full tensor equality
        def _moments(t):
            return t.mean().cpu(), t.std().cpu()
        m_cpu, s_cpu = _moments(out_cpu)
        m_gpu, s_gpu = _moments(out_gpu)
        assert torch.allclose(m_cpu, m_gpu, rtol=5e-2, atol=5e-2)
        assert torch.allclose(s_cpu, s_gpu, rtol=5e-2, atol=5e-2)


def test_hydra_style_construction(device):
    """Construct encoder from a dict like a Hydra config would provide."""
    cfg = {
        "hidden_dim": 96,
        "num_layers": 4,
        "dropout": 0.1,
        "edge_attr_dim": 2,  # our synthetic cluster edges have 2 channels
        "readout": "mean",
    }
    enc = AIRSGNNEncoder(**cfg).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=2, feat_dim=3, device=device
    )
    out = enc(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    assert out.shape[0] == x.shape[0]


def test_edge_shape_mismatch_raises(device):
    """Mismatched edge_index/edge_attr lengths should raise a friendly error."""
    enc = AIRSGNNEncoder(hidden_dim=64, num_layers=3).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=2, device=device
    )
    # Corrupt edge_attr to wrong E
    wrong_attr = edge_attr[:-5, :] if edge_attr.shape[0] > 5 else torch.randn(1, edge_attr.shape[1], device=device)
    with pytest.raises(Exception):
        _ = enc(x, mask=mask, edge_index=edge_index, edge_attr=wrong_attr)


@pytest.mark.xfail(reason="Some GNN backends are not TorchScript-compatible; best-effort export.")
def test_torchscript_export(device, tmp_path):
    """Attempt to script and run a saved AIRS GNN; xfail if not supported by backend."""
    enc = AIRSGNNEncoder(hidden_dim=64, num_layers=2).to(device)
    scripted = torch.jit.script(enc)
    path = tmp_path / "airs_gnn.pt"
    scripted.save(str(path))
    reloaded = torch.jit.load(str(path), map_location=device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=1, device=device
    )
    out = reloaded(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    assert out.shape[0] == 1


def test_state_dict_roundtrip(device):
    """Save/load state_dict and confirm close outputs."""
    enc1 = AIRSGNNEncoder(hidden_dim=48, num_layers=2).to(device)
    enc2 = AIRSGNNEncoder(hidden_dim=48, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=2, device=device
    )
    with torch.no_grad():
        out1 = enc1(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    sd = enc1.state_dict()
    missing, unexpected = enc2.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0
    with torch.no_grad():
        out2 = enc2(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-5)


def test_optional_attention_weights_extraction(device):
    """
    If the encoder exposes attention/edge weights (e.g., via get_attention_weights or returns aux dict),
    retrieve them and perform basic checks. Otherwise, skip.
    """
    enc = AIRSGNNEncoder(hidden_dim=32, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=2, device=device
    )
    out = enc(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)

    # Try common patterns:
    weights = None
    if hasattr(enc, "get_attention_weights") and callable(getattr(enc, "get_attention_weights")):
        with contextlib.suppress(Exception):
            weights = enc.get_attention_weights()
    elif isinstance(out, (tuple, list)) and len(out) >= 2 and torch.is_tensor(out[1]):
        weights = out[1]
        out = out[0]
    elif isinstance(out, dict) and "attn" in out:
        weights = out["attn"]

    if weights is None:
        pytest.skip("No attention/edge-weight interface exposed by AIRSGNNEncoder.")
    else:
        # Weights should be finite; shape may vary by implementation.
        if torch.is_tensor(weights):
            assert torch.isfinite(weights).all()
        elif isinstance(weights, (list, tuple)):
            assert all(torch.is_tensor(w) and torch.isfinite(w).all() for w in weights)
        else:
            # Allow dict of tensors
            assert all(torch.is_tensor(v) and torch.isfinite(v).all() for v in weights.values())


def test_logging_hook_integration(tmp_path, device):
    """Ensure logging works with encoder outputs and metadata."""
    logger = v50_logging.get_logger("test_airs_gnn", logfile=str(tmp_path / "airs_gnn_test.json"))
    enc = AIRSGNNEncoder(hidden_dim=40, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=2, feat_dim=3, device=device
    )
    out = enc(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    logger.info("AIRS forward", extra={"batch": x.shape[0], "out_dim": int(out.shape[-1]) if out.ndim >= 2 else int(out.shape[1])})
    # Read back log file to ensure write happened
    with open(logger.handlers[0].baseFilename, "r") as f:
        content = f.read()
    assert "AIRS forward" in content and "out_dim" in content


def test_mixed_precision_autocast(device):
    """Smoke test autocast (CUDA or CPU BF16 if supported)."""
    enc = AIRSGNNEncoder(hidden_dim=32, num_layers=2).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=1, feat_dim=1, device=device
    )
    # Choose dtype context
    autocast_kwargs = {}
    if device.type == "cuda":
        autocast_kwargs = dict(device_type="cuda", dtype=torch.float16)
    else:
        # On some CPUs with BF16 support
        if torch.cpu.amp.is_bf16_supported():
            autocast_kwargs = dict(device_type="cpu", dtype=torch.bfloat16)
        else:
            pytest.skip("No AMP support on this device.")
    with torch.autocast(**autocast_kwargs):
        out = enc(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    assert isinstance(out, torch.Tensor)


def test_large_batch_efficiency(device):
    """Ensure encoder can handle a moderately large batch without OOM (uses small dims)."""
    enc = AIRSGNNEncoder(hidden_dim=24, num_layers=2, dropout=0.0).to(device)
    x, mask, edge_index, edge_attr = make_synthetic_airs_batch(
        batch_size=8, feat_dim=1, device=device
    )
    out = enc(x, mask=mask, edge_index=edge_index, edge_attr=edge_attr)
    if out.ndim == 2:
        assert out.shape == (8, out.shape[1])
    else:
        assert out.shape[:2] == (8, NUM_BINS)


if __name__ == "__main__":
    pytest.main([__file__])
