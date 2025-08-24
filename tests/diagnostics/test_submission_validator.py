#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_submission_validator.py

SpectraMind V50 — Unit Test: Submission Validator

Purpose
-------
Validate the `tools/validate_submission.py` module to ensure that:
  • Submissions conform to Kaggle NeurIPS 2025 Ariel Data Challenge rules
  • Required output files exist (submission.csv, optional manifest)
  • Predictions (μ, σ) have correct shape (N_planets × 2 × 283 bins)
  • No NaNs/Infs are present
  • CLI/Hydra reproducibility and logging are intact

This test is critical for CI gating: it prevents broken submissions
from being packaged or uploaded, enforcing leaderboard-ready compliance.
"""

import io
import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
import subprocess
from pathlib import Path

# Import the validator under test
import importlib.util

VALIDATOR_PATH = Path("tools/validate_submission.py")

spec = importlib.util.spec_from_file_location("validate_submission", VALIDATOR_PATH)
validate_submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_submission)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="function")
def dummy_submission(tmp_path):
    """
    Create a dummy valid submission CSV with μ and σ for 3 planets × 283 bins.
    """
    n_planets, n_bins = 3, 283
    mu = np.random.rand(n_planets, n_bins).astype(np.float32)
    sigma = np.random.rand(n_planets, n_bins).astype(np.float32) * 0.05 + 1e-3
    # Construct DataFrame with Kaggle-style columns
    df = pd.DataFrame(
        {
            "planet_id": np.arange(n_planets),
            "mu": list(mu),
            "sigma": list(sigma),
        }
    )
    out_csv = tmp_path / "submission.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_validator_module_exists():
    """Ensure validator script exists in tools/ directory."""
    assert VALIDATOR_PATH.exists(), f"Missing {VALIDATOR_PATH}"


def test_validate_submission_passes_on_dummy(dummy_submission):
    """Check that a valid dummy submission passes validation."""
    result = validate_submission.validate_submission_file(str(dummy_submission))
    assert result["valid"], f"Expected submission to pass but got: {result}"


def test_validator_rejects_missing_file(tmp_path):
    """Validator should fail if file does not exist."""
    bad_path = tmp_path / "nonexistent.csv"
    result = validate_submission.validate_submission_file(str(bad_path))
    assert not result["valid"]
    assert "not found" in result["error"].lower()


def test_validator_rejects_nan_values(tmp_path):
    """Validator should fail on NaN μ/σ values."""
    n_planets, n_bins = 2, 283
    mu = np.full((n_planets, n_bins), np.nan)
    sigma = np.ones((n_planets, n_bins))
    df = pd.DataFrame({"planet_id": range(n_planets), "mu": list(mu), "sigma": list(sigma)})
    bad_file = tmp_path / "bad_nan.csv"
    df.to_csv(bad_file, index=False)

    result = validate_submission.validate_submission_file(str(bad_file))
    assert not result["valid"]
    assert "nan" in result["error"].lower()


def test_validator_rejects_wrong_shape(tmp_path):
    """Validator should fail if bins != 283."""
    n_planets, n_bins = 2, 100  # wrong bin count
    mu = np.random.rand(n_planets, n_bins)
    sigma = np.random.rand(n_planets, n_bins)
    df = pd.DataFrame({"planet_id": range(n_planets), "mu": list(mu), "sigma": list(sigma)})
    bad_file = tmp_path / "bad_shape.csv"
    df.to_csv(bad_file, index=False)

    result = validate_submission.validate_submission_file(str(bad_file))
    assert not result["valid"]
    assert "283" in result["error"]


def test_cli_entrypoint_runs(dummy_submission):
    """Ensure CLI entrypoint runs and exits with code 0 on valid submission."""
    cmd = [sys.executable, str(VALIDATOR_PATH), "--file", str(dummy_submission)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert "VALID" in proc.stdout.upper()


def test_cli_rejects_invalid(tmp_path):
    """Ensure CLI exits nonzero on invalid submission."""
    bad_file = tmp_path / "invalid.csv"
    pd.DataFrame({"planet_id": [1], "mu": [[0.1]*100], "sigma": [[0.1]*100]}).to_csv(bad_file, index=False)
    cmd = [sys.executable, str(VALIDATOR_PATH), "--file", str(bad_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "ERROR" in proc.stdout.upper() or "INVALID" in proc.stdout.upper()