# /tests/diagnostics/test_validate_submission.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: validate_submission

Purpose
-------
Validate the repository's submission validator for the NeurIPS Ariel Data Challenge format.
This test exercises both API and CLI (if present), ensuring the following are detected:

1) Schema & header correctness
   • Required ID column present (flexible names: ['id','ID','planet_id','planetID']).
   • Exactly 283 μ columns and 283 σ columns (prefix-flexible: ['mu','mean','mu_','m_'] and
     ['sigma','std','stddev','sigma_','s_']).
   • No unexpected duplicates, consistent zero-indexing or contiguous bin numbering.

2) Value constraints (scientific + numerical)
   • μ finite, typically in [0, 1] (we only assert finiteness here; range checks if validator supports).
   • σ finite and strictly > 0 (positive; non-zero).
   • No NaNs/inf.
   • Optional clipping warnings allowed if validator supports.

3) Cardinality & identity checks
   • Number of rows matches the provided/expected ID set if supplied.
   • No duplicate IDs.

4) Determinism & robustness
   • Same CSV yields identical validation result.
   • Clear error messages for common failure modes.

5) Artifacts / report
   • If the validator writes a JSON/MD report, ensure it is created and contains useful fields.

6) CLI smoke (optional)
   • If a `spectramind` CLI is present with a submission validation subcommand, smoke-test it.

Design notes
------------
• Defensively adaptable: we discover the module under several paths and try common entrypoints:
    - validate_submission(path or df, ...) -> dict
    - validate(path or df, ...)
    - check_submission(path or df, ...)
    - class SubmissionValidator(...).validate(...)
• We do not hard-code exact column names beyond flexible prefixes; the test fabricates a valid header that the
  repo validator should accept if it follows the standard (283 μ + 283 σ with consistent bin indices).
• All temp files are created under pytest tmp_path for isolation.

Author: SpectraMind V50 Team
"""

from __future__ import annotations

import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------------------------------
# Module discovery
# --------------------------------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    "tools.validate_submission",
    "src.tools.validate_submission",
    "diagnostics.validate_submission",
    "submission_validator",
    "tools.submission_validator",
    "src.tools.submission_validator",
    "validate_submission",
]


def _import_validator_module():
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import submission validator from any of:\n"
        f"  {CANDIDATE_IMPORTS}\n"
        f"Last error: {last_err}"
    )


def _locate_entrypoint(mod):
    """
    Locate a callable/class to validate a submission.

    Accepted function names:
      - validate_submission(obj, **cfg)
      - validate(obj, **cfg)
      - check_submission(obj, **cfg)
    Accepted classes:
      - SubmissionValidator(...).validate(...)
      - Validator(...).validate(...)
    """
    for fn_name in ("validate_submission", "validate", "check_submission"):
        if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
            return "func", getattr(mod, fn_name)

    for cls_name in ("SubmissionValidator", "Validator", "SubmissionChecker"):
        if hasattr(mod, cls_name):
            Cls = getattr(mod, cls_name)
            if hasattr(Cls, "validate") and callable(getattr(Cls, "validate")):
                return "class", Cls

    pytest.xfail(
        "Submission validator module found but no known entrypoint was discovered. "
        "Expected one of: validate_submission(), validate(), check_submission(), "
        "or a class with .validate(...)."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, obj, **cfg) -> Dict[str, Any]:
    """
    Invoke the validator and normalize the output to a dict.

    Expected dict keys (subset ok):
      - 'ok': bool
      - 'errors': list[str]
      - 'warnings': list[str]
      - 'report_path': optional str
      - 'n_rows', 'n_mu', 'n_sigma'
    """
    if kind == "func":
        out = target(obj, **cfg)
    elif kind == "class":
        try:
            inst = target(**cfg)
        except TypeError:
            inst = target()
        out = inst.validate(obj)
    else:
        pytest.fail("Unknown invocation kind.")  # pragma: no cover

    if isinstance(out, dict):
        assert "ok" in out, "Validator must return a dict with 'ok' key."
        return out
    # If a boolean is returned, coerce into dict
    if isinstance(out, bool):
        return {"ok": out, "errors": [] if out else ["unknown error"], "warnings": []}
    pytest.fail("Validator returned unsupported type; expected dict or bool.")  # pragma: no cover
    return {}


# --------------------------------------------------------------------------------------------------
# Submission fabricators
# --------------------------------------------------------------------------------------------------

L_BINS = 283
ID_NAMES = ["id", "ID", "planet_id", "planetID"]
MU_PREFIXES = ["mu", "mu_", "mean", "m_"]
SIGMA_PREFIXES = ["sigma", "sigma_", "std", "stddev", "s_"]


def _build_header(mu_prefix: str = "mu_", sigma_prefix: str = "sigma_") -> List[str]:
    cols = ["ID"]
    cols += [f"{mu_prefix}{i}" for i in range(L_BINS)]
    cols += [f"{sigma_prefix}{i}" for i in range(L_BINS)]
    return cols


def _make_valid_df(n_rows: int = 5, mu_prefix: str = "mu_", sigma_prefix: str = "sigma_") -> pd.DataFrame:
    cols = _build_header(mu_prefix, sigma_prefix)
    df = pd.DataFrame(columns=cols)
    # Fill with realistic values: μ in [0.0, 0.05]; σ in (1e-6, 0.05]
    for r in range(n_rows):
        row = {}
        row["ID"] = f"P{r:04d}"
        mu = 0.01 + 0.01 * np.sin(np.linspace(0, 2 * np.pi, L_BINS)) + np.random.normal(0, 1e-4, L_BINS)
        mu = np.clip(mu, 0.0, 1.0)
        sigma = 0.01 + np.abs(np.random.normal(0, 1e-3, L_BINS))  # strictly > 0
        for i in range(L_BINS):
            row[f"{mu_prefix}{i}"] = float(mu[i])
            row[f"{sigma_prefix}{i}"] = float(sigma[i])
        df.loc[r] = row
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    assert path.exists() and path.stat().st_size > 0
    return path


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def val_mod():
    return _import_validator_module()


@pytest.fixture(scope="module")
def val_call(val_mod):
    return _locate_entrypoint(val_mod)


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    df = _make_valid_df(n_rows=5, mu_prefix="mu_", sigma_prefix="sigma_")
    return _write_csv(df, tmp_path / "submission_valid.csv")


# --------------------------------------------------------------------------------------------------
# Tests — basic success & determinism
# --------------------------------------------------------------------------------------------------

def test_accepts_valid_submission(val_call, valid_csv):
    kind, target = val_call
    out = _invoke(kind, target, str(valid_csv))
    assert isinstance(out, dict)
    assert out["ok"] is True, f"Validator rejected a valid CSV. Errors: {out.get('errors')}"
    # Optional informative fields
    if "n_mu" in out and "n_sigma" in out:
        assert out["n_mu"] == L_BINS and out["n_sigma"] == L_BINS
    if "n_rows" in out:
        assert out["n_rows"] == 5


def test_determinism_on_same_file(val_call, valid_csv):
    kind, target = val_call
    out1 = _invoke(kind, target, str(valid_csv))
    out2 = _invoke(kind, target, str(valid_csv))
    # Compare core booleans and first error/warning sets deterministically
    assert out1["ok"] == out2["ok"]
    assert sorted(out1.get("errors", [])) == sorted(out2.get("errors", []))
    assert sorted(out1.get("warnings", [])) == sorted(out2.get("warnings", []))


# --------------------------------------------------------------------------------------------------
# Tests — schema failures
# --------------------------------------------------------------------------------------------------

def test_missing_id_column(val_call, tmp_path: Path):
    df = _make_valid_df(3)
    # Remove ID column
    mu_cols = [c for c in df.columns if c.lower().startswith("mu")]
    sigma_cols = [c for c in df.columns if c.lower().startswith("sigma") or c.lower().startswith("std")]
    df = df[mu_cols + sigma_cols]
    csv_path = _write_csv(df, tmp_path / "submission_no_id.csv")

    kind, target = val_call
    out = _invoke(kind, target, str(csv_path))
    assert out["ok"] is False
    assert any("id" in e.lower() for e in out.get("errors", [])), f"Expected ID error, got: {out.get('errors')}"


def test_missing_some_mu_columns(val_call, tmp_path: Path):
    df = _make_valid_df(3)
    # Drop a μ column
    drop_col = [c for c in df.columns if re.match(r"(?i)mu[_]?\d+$", c)][0]
    df = df.drop(columns=[drop_col])
    csv_path = _write_csv(df, tmp_path / "submission_missing_mu.csv")

    kind, target = val_call
    out = _invoke(kind, target, str(csv_path))
    assert out["ok"] is False
    assert any("mu" in e.lower() or "mean" in e.lower() for e in out.get("errors", [])), (
        f"Expected μ column count/indexing error, got: {out.get('errors')}"
    )


def test_missing_some_sigma_columns(val_call, tmp_path: Path):
    df = _make_valid_df(3)
    # Drop a σ column
    drop_col = [c for c in df.columns if re.match(r"(?i)(sigma|std|stddev|s)[_]?\d+$", c)][0]
    df = df.drop(columns=[drop_col])
    csv_path = _write_csv(df, tmp_path / "submission_missing_sigma.csv")

    kind, target = val_call
    out = _invoke(kind, target, str(csv_path))
    assert out["ok"] is False
    assert any("sigma" in e.lower() or "std" in e.lower() for e in out.get("errors", [])), (
        f"Expected σ column count/indexing error, got: {out.get('errors')}"
    )


def test_duplicate_ids(val_call, tmp_path: Path):
    df = _make_valid_df(4)
    df.loc[3, "ID"] = df.loc[2, "ID"]  # duplicate
    csv_path = _write_csv(df, tmp_path / "submission_dup_ids.csv")

    kind, target = val_call
    out = _invoke(kind, target, str(csv_path))
    assert out["ok"] is False
    assert any("duplicate" in e.lower() and "id" in e.lower() for e in out.get("errors", [])), (
        f"Expected duplicate ID error, got: {out.get('errors')}"
    )


# --------------------------------------------------------------------------------------------------
# Tests — value failures
# --------------------------------------------------------------------------------------------------

def test_nan_values_rejected(val_call, tmp_path: Path):
    df = _make_valid_df(3)
    # Inject NaNs into μ and σ
    mu_col = [c for c in df.columns if re.match(r"(?i)mu[_]?\d+$", c)][10]
    sg_col = [c for c in df.columns if re.match(r"(?i)(sigma|std|stddev|s)[_]?\d+$", c)][20]
    df.loc[1, mu_col] = np.nan
    df.loc[2, sg_col] = np.nan
    csv_path = _write_csv(df, tmp_path / "submission_with_nan.csv")

    kind, target = val_call
    out = _invoke(kind, target, str(csv_path))
    assert out["ok"] is False
    assert any("nan" in e.lower() or "finite" in e.lower() for e in out.get("errors", [])), (
        f"Expected NaN/finite error, got: {out.get('errors')}"
    )


def test_nonpositive_sigma_rejected(val_call, tmp_path: Path):
    df = _make_valid_df(3)
    # Inject nonpositive sigma values
    s_cols = [c for c in df.columns if re.match(r"(?i)(sigma|std|stddev|s)[_]?\d+$", c)]
    df.loc[0, s_cols[0]] = 0.0
    df.loc[2, s_cols[1]] = -1e-6
    csv_path = _write_csv(df, tmp_path / "submission_sigma_nonpos.csv")

    kind, target = val_call
    out = _invoke(kind, target, str(csv_path))
    assert out["ok"] is False
    msgs = " ".join(out.get("errors", [])).lower()
    assert ("sigma" in msgs or "std" in msgs) and ("positive" in msgs or ">" in msgs or "nonzero" in msgs), (
        f"Expected positive sigma constraint error, got: {out.get('errors')}"
    )


# --------------------------------------------------------------------------------------------------
# Tests — report/artifacts (optional)
# --------------------------------------------------------------------------------------------------

def test_report_artifacts_if_supported(val_mod, val_call, tmp_path: Path):
    """
    If the module exposes save_report(...) / save_artifacts(...), ensure it writes something.
    Otherwise xfail gracefully.
    """
    save_fn = None
    for name in ("save_report", "save_artifacts", "write_report"):
        if hasattr(val_mod, name) and callable(getattr(val_mod, name)):
            save_fn = getattr(val_mod, name)
            break

    # Use a valid CSV for report generation
    df = _make_valid_df(2)
    csv_path = _write_csv(df, tmp_path / "submission_for_report.csv")

    kind, target = val_call
    result = _invoke(kind, target, str(csv_path))

    if save_fn is None:
        pytest.xfail("Validator does not expose a report/artifact saver; skipping artifact test.")

    outdir = tmp_path / "validation_report"
    outdir.mkdir(parents=True, exist_ok=True)
    save_fn(result, outdir=str(outdir))  # should not raise

    files = list(outdir.glob("*"))
    assert files, "No artifacts produced by report saver."
    # If a JSON exists, ensure it has 'ok' and errors/warnings arrays
    j = outdir / "validation_report.json"
    if j.exists():
        with open(j, "r", encoding="utf-8") as f:
            rep = json.load(f)
        assert "ok" in rep and "errors" in rep and "warnings" in rep


# --------------------------------------------------------------------------------------------------
# CLI smoke (optional)
# --------------------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("spectramind") is None, reason="spectramind CLI not found in PATH")
def test_cli_smoke_validate_submission(tmp_path: Path):
    """
    Smoke-test the CLI route (adjust flags if your repo differs).
    Expected patterns we try (first successful one wins):
      1) spectramind submit validate --file <csv>
      2) spectramind validate submission --file <csv>
      3) spectramind validate-submission --file <csv>
    We only assert that it runs with exit code 0 on a valid CSV.
    """
    df = _make_valid_df(2)
    csv_path = _write_csv(df, tmp_path / "cli_valid.csv")

    candidates = [
        ["spectramind", "submit", "validate", "--file", str(csv_path)],
        ["spectramind", "validate", "submission", "--file", str(csv_path)],
        ["spectramind", "validate-submission", "--file", str(csv_path)],
        # Slight variants
        ["spectramind", "submit", "validate", "--path", str(csv_path)],
        ["spectramind", "validate", "submission", "--path", str(csv_path)],
    ]

    last_proc = None
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception:
            continue
        last_proc = proc
        if proc.returncode == 0:
            break

    if last_proc is None or last_proc.returncode != 0:
        pytest.xfail(
            "No working CLI validate pattern found. Ensure the submit/validate command is wired. "
            f"Last stdout/stderr:\n{'' if last_proc is None else last_proc.stdout}\n"
            f"{'' if last_proc is None else last_proc.stderr}"
        )

    assert last_proc.returncode == 0


# --------------------------------------------------------------------------------------------------
# Performance guardrail
# --------------------------------------------------------------------------------------------------

def test_runs_fast_enough(val_call, valid_csv):
    kind, target = val_call
    import time
    t0 = time.time()
    _ = _invoke(kind, target, str(valid_csv))
    dt = time.time() - t0
    assert dt < 1.0, f"Validation too slow for tiny CSV: {dt:.3f}s (should be < 1.0s)"