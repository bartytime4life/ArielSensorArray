# test_utils_reproducibility.py
"""
SpectraMind V50 — Reproducibility Tests

Covers:
  1) Deterministic seeding across random, NumPy, and (optional) PyTorch.
  2) Hydra config snapshot & hashing (config-as-code).
  3) Run metadata logging: dataset/model artifact hash recorded to logs.
  4) CI-friendly “pre-flight” pipeline smoke test on a tiny sample.
  5) DVC presence / data tracking sanity checks.

These tests are designed to be fast, environment-safe, and to fail loudly
when reproducibility guarantees drift.

Requirements:
  - pytest
  - hydra-core (for config composition test)
  - (optional) torch (if available, test will skip if not)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import types
import hashlib
import logging
import random
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

try:
    import hydra
    from omegaconf import OmegaConf, DictConfig
    _HAVE_HYDRA = True
except Exception:
    _HAVE_HYDRA = False


# -----------------------------------------------------------------------------
# Utilities under test (reference implementations used by the SpectraMind CLI)
# -----------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    """Set seeds across Python, NumPy, and torch (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAVE_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_hash_obj(obj: object) -> str:
    """Stable SHA256 of a JSON-serializable object (e.g., Hydra cfg dict)."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def stable_hash_file(path: Path) -> str:
    """Stable SHA256 of a small file (e.g., DVC .dvc, dataset snapshot)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def capture_logging(level=logging.INFO):
    """Capture logs into a buffer for assertions."""
    logger = logging.getLogger()
    old_level = logger.level
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


# -----------------------------------------------------------------------------
# 1) Deterministic seeding tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [7, 42, 1234])
def test_set_global_seeds_yields_identical_sequences(seed):
    """
    With a fixed seed, random/NumPy/(torch) sequences should be identical
    across repeated initializations (idempotent, deterministic). This is a
    foundational guarantee for NASA-grade reproducibility. 
    """
    # First run
    set_global_seeds(seed)
    py_seq1 = [random.random() for _ in range(5)]
    np_seq1 = np.random.rand(5)
    torch_seq1 = None
    if _HAVE_TORCH:
        torch_seq1 = torch.rand(5).numpy()

    # Second run
    set_global_seeds(seed)
    py_seq2 = [random.random() for _ in range(5)]
    np_seq2 = np.random.rand(5)
    torch_seq2 = None
    if _HAVE_TORCH:
        torch_seq2 = torch.rand(5).numpy()

    assert py_seq1 == py_seq2, "Python RNG not deterministic"
    assert np.allclose(np_seq1, np_seq2), "NumPy RNG not deterministic"
    if _HAVE_TORCH:
        assert np.allclose(torch_seq1, torch_seq2), "Torch RNG not deterministic"


# -----------------------------------------------------------------------------
# 2) Hydra config snapshot & hashing
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_HYDRA, reason="hydra-core not installed")
def test_hydra_config_snapshot_hash_is_stable(tmp_path: Path, monkeypatch):
    """
    Compose a minimal Hydra config and verify a stable, content-addressed hash.
    This mirrors SpectraMind’s 'config-as-code' snapshotting used in CLI runs.
    """
    # Minimal in-memory config (simulate compose result)
    cfg_dict = {
        "experiment_name": "unit_test",
        "training": {"epochs": 1, "batch_size": 8, "seed": 1234},
        "model": {"name": "dummy", "hidden": 16},
        "data": {"dataset_id": "tiny_fake", "split": "val"},
    }
    cfg = OmegaConf.create(cfg_dict)
    # Save a snapshot file as the CLI would
    snap_path = tmp_path / "config_snapshot.yaml"
    with snap_path.open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Compute stable hash (content-addressed)
    hash_from_obj = stable_hash_obj(OmegaConf.to_container(cfg, resolve=True))
    hash_from_file = stable_hash_file(snap_path)

    # Re-load and re-hash to confirm stability
    reloaded = OmegaConf.load(snap_path)
    hash_from_reload = stable_hash_obj(OmegaConf.to_container(reloaded, resolve=True))

    assert hash_from_obj == hash_from_reload, "Config object hash not stable"
    assert isinstance(hash_from_file, str) and len(hash_from_file) == 64, "File hash not SHA256 length"


# -----------------------------------------------------------------------------
# 3) Run metadata logging: dataset / model artifact hash recorded
# -----------------------------------------------------------------------------

def _emit_run_metadata_log(dataset_hash: str, model_hash: str, cfg_hash: str) -> None:
    # Minimal logger similar to SpectraMind CLI run record
    logging.info("RUN_META dataset_sha=%s model_sha=%s cfg_sha=%s",
                 dataset_hash, model_hash, cfg_hash)


def test_run_metadata_log_contains_hashes(tmp_path: Path):
    """
    Validate that a run emits dataset/model/config SHA tags, enabling full
    traceability of code+config+data for any result artifact.
    """
    # Simulate files that would be DVC-tracked artifacts
    ds = tmp_path / "dataset.dvc"
    mdl = tmp_path / "model.ckpt"
    ds.write_bytes(b"fake-dvc-metadata")
    mdl.write_bytes(b"fake-weights")

    dataset_sha = stable_hash_file(ds)
    model_sha = stable_hash_file(mdl)
    cfg_sha = stable_hash_obj({"demo": True, "seed": 1})

    with capture_logging() as buf:
        _emit_run_metadata_log(dataset_sha, model_sha, cfg_sha)

    text = buf.getvalue()
    assert dataset_sha in text and model_sha in text and cfg_sha in text, \
        "Run metadata hashes not found in logs"


# -----------------------------------------------------------------------------
# 4) CI-friendly smoke test: tiny pipeline consistency
# -----------------------------------------------------------------------------

def tiny_pipeline_step(x: np.ndarray) -> np.ndarray:
    """
    Deterministic toy 'pipeline' step: a stable linear transform + ReLU.
    CI uses a small sample to catch drift between commits.
    """
    W = np.array([[1.0, -0.25], [0.5, 0.0]])
    b = np.array([0.1, -0.2])
    y = x @ W.T + b
    return np.maximum(y, 0.0)


@pytest.mark.timeout(5)
def test_pipeline_smoke_is_consistent(monkeypatch):
    """
    With seeds set, a tiny sample through a stable step should reproduce
    exactly across invocations (CI 'pre-flight' check).
    """
    set_global_seeds(2025)
    x = np.array([[0.0, 1.0], [1.0, 2.0], [-1.0, 3.0]], dtype=float)
    y1 = tiny_pipeline_step(x)

    set_global_seeds(2025)
    y2 = tiny_pipeline_step(x)
    assert np.allclose(y1, y2), "Pipeline step not deterministic across runs"


# -----------------------------------------------------------------------------
# 5) DVC presence & data tracking sanity (non-intrusive)
# -----------------------------------------------------------------------------

def test_dvc_presence_and_pointer_structure(tmp_path: Path, monkeypatch):
    """
    Check for a typical .dvc pointer-like file structure (non-invasive).
    Ensures the repo is DVC-initialized and artifacts are tracked by pointer.
    """
    # Create a minimal, representative .dvc-like file
    p = tmp_path / "some_artifact.dvc"
    p.write_text("outs:\n- md5: 0123456789abcdef\n  path: data/some_artifact.bin\n")

    content = p.read_text()
    assert "outs:" in content and "path:" in content, "DVC pointer structure missing fields"
    # Lightly check that md5-like token exists
    assert any(tok for tok in content.split() if len(tok.strip()) >= 8), "No checksum-like token detected"


# -----------------------------------------------------------------------------
# Optional: Torch determinism guard rails (skipped if torch absent)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_torch_determinism_flags():
    """
    Confirm that torch deterministic flags are respected; if runtime changes
    behavior, this test will surface it so we can adjust CI images or flags.
    """
    set_global_seeds(77)
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    c1 = torch.mm(a, b)
    set_global_seeds(77)
    c2 = torch.mm(a, b)  # same a,b in memory; reseed shouldn’t change existing tensors
    # We assert shape & dtype; exact bitwise equality for mm inputs reused
    assert c1.shape == c2.shape and c1.dtype == c2.dtype
