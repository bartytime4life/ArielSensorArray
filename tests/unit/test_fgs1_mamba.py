#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_fgs1_mamba.py

Unit and integration tests for the FGS1Mamba encoder module in SpectraMind V50.

This test suite verifies:
• Input/output shape integrity for synthetic FGS1 sequences
• Deterministic behavior with fixed seeds
• Gradient flow and backward pass
• Integration with Hydra configs and reproducibility utils
• Symbolic-aware hooks (logging, overlays)
• TorchScript export and fast inference

Run:
  pytest -v tests/test_fgs1_mamba.py
"""

import os
import sys
import json
import tempfile
import pytest
import torch
import numpy as np

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fgs1_mamba import FGS1MambaEncoder
from src.utils import reproducibility, logging as v50_logging


@pytest.fixture(scope="module")
def device():
    """Pick device (GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True, scope="module")
def set_seed():
    """Ensure deterministic seeds before each test run."""
    reproducibility.set_seed(42)
    yield
    reproducibility.set_seed(42)


def test_forward_shape(device):
    """Check encoder output shape with synthetic input."""
    B, T, H, W = 2, 135000, 32, 32  # typical FGS1 shape
    encoder = FGS1MambaEncoder(embed_dim=128, depth=4).to(device)
    x = torch.randn(B, T, H, W, device=device)
    out = encoder(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == B
    assert out.ndim in (2, 3), "Output must be [B,D] or [B,L,D]"


def test_deterministic_forward(device):
    """Ensure deterministic outputs with fixed seeds."""
    B, T, H, W = 1, 256, 32, 32
    encoder = FGS1MambaEncoder(embed_dim=64, depth=2).to(device)
    x = torch.randn(B, T, H, W, device=device)
    out1 = encoder(x)
    reproducibility.set_seed(42)
    encoder2 = FGS1MambaEncoder(embed_dim=64, depth=2).to(device)
    x2 = torch.randn(B, T, H, W, device=device)
    out2 = encoder2(x2)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_backward_pass(device):
    """Check gradients flow correctly."""
    B, T, H, W = 2, 512, 32, 32
    encoder = FGS1MambaEncoder(embed_dim=32, depth=2).to(device)
    x = torch.randn(B, T, H, W, device=device, requires_grad=True)
    out = encoder(x)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_torchscript_export(tmp_path, device):
    """Verify TorchScript export and reload."""
    encoder = FGS1MambaEncoder(embed_dim=32, depth=2).to(device)
    scripted = torch.jit.script(encoder)
    export_path = tmp_path / "fgs1_mamba.pt"
    scripted.save(str(export_path))
    reloaded = torch.jit.load(str(export_path), map_location=device)
    x = torch.randn(1, 512, 32, 32, device=device)
    out1 = encoder(x)
    out2 = reloaded(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_symbolic_hook_logging(device, tmp_path):
    """Simulate symbolic overlay hooks logging outputs."""
    log_path = tmp_path / "fgs1_mamba_log.json"
    logger = v50_logging.get_logger("test_fgs1_mamba", logfile=str(log_path))
    encoder = FGS1MambaEncoder(embed_dim=16, depth=1).to(device)

    x = torch.randn(1, 128, 32, 32, device=device)
    out = encoder(x)

    logger.info("Test run", extra={"output_norm": out.norm().item()})
    logger.handlers[0].flush()

    with open(log_path, "r") as f:
        data = f.read()
    assert "output_norm" in data


def test_integration_with_hydra_config(device):
    """Integration test: build encoder from config dict."""
    cfg = {"embed_dim": 64, "depth": 3}
    encoder = FGS1MambaEncoder(**cfg).to(device)
    x = torch.randn(1, 1024, 32, 32, device=device)
    out = encoder(x)
    assert out.shape[0] == 1


def test_nan_propagation(device):
    """Ensure NaNs in input propagate safely."""
    encoder = FGS1MambaEncoder(embed_dim=16, depth=2).to(device)
    x = torch.full((1, 64, 32, 32), float("nan"), device=device)
    out = encoder(x)
    assert torch.isnan(out).any()


def test_large_batch_runtime(device):
    """Smoke test with large batch to ensure no OOM with gradient checkpointing."""
    encoder = FGS1MambaEncoder(embed_dim=32, depth=2, checkpoint=True).to(device)
    x = torch.randn(4, 2048, 32, 32, device=device)
    out = encoder(x)
    assert out.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__])
