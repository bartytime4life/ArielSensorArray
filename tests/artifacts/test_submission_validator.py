# tests/artifacts/test_submission_validator.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Artifact Tests
File: tests/artifacts/test_submission_validator.py

Purpose
-------
End-to-end tests for the submission bundle validator. These tests generate tiny
synthetic submission CSVs (and .gz variants), then exercise the validator either
as a Python API (preferred) or via its CLI (fallback). The goal is to ensure the
validator catches common failure modes and accepts a correct file.

What this test asserts
----------------------
1) Correct “happy path” submission passes validation.
2) Wrong column count is rejected.
3) NaN / Inf values are rejected.
4) Non‑positive σ (sigma) values are rejected.
5) Duplicate planet_ids are rejected.
6) Missing required planet_ids are rejected.
7) Extra / unknown planet_ids are rejected.
8) Wrong header names are rejected.
9) Gzip’ed submissions (.csv.gz) are supported (happy path).

Assumptions (kept minimal and documented)
-----------------------------------------
• The repository provides a validator in one of these forms:
  A) Python API:
       tools/validate_submission.py  exposing a function like:
         validate_submission(
             submission_path: str,
             expected_ids: Optional[Sequence[str]] = None,
             bins: int = 283,
             require_sigma: bool = True,
             strict_headers: bool = True,
         ) -> Tuple[bool, List[str]]
     The exact signature can vary; the test tries to adapt.
  B) CLI:
       python tools/validate_submission.py --submission SUB.csv \
              --ids IDS.txt --bins 283 --require-sigma --strict-headers
     Exit code 0 = pass; non‑zero = fail. Errors are printed to stderr/stdout.

• Output shape for Ariel Data Challenge 2025:
  − 283 spectral bins for μ and 283 for σ  → 566 numeric columns total
  − 1 identifier column: planet_id
  − Header layout (strict mode):
       "planet_id", "mu_0"... "mu_282", "sigma_0"... "sigma_282"

We intentionally run with a *small* expected ID list (3 fake planets) so tests
are fast and hermetic. The test passes those IDs to the validator (API or CLI).

If your local validator offers different flag names, the CLI fallback uses
sensible alternatives and will fail with a helpful message if no recognized
entrypoint is found. Adjust the `CLI_CANDIDATES` / `CLI_FLAGS_MATRIX` below if
your CLI interface differs.

Author: SpectraMind V50 — Test Suite
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pytest


# ------------------------------- Constants --------------------------------- #

# We use a minimal “toy” set of planets to keep tests fast and hermetic.
TEST_PLANET_IDS = ["P0001", "P0002", "P0003"]

# Spectral bins for Ariel V50 (challenge standard). Keep as the canonical value.
BINS = 283

# Column name templates
MU_COLS = [f"mu_{i}" for i in range(BINS)]
SIGMA_COLS = [f"sigma_{i}" for i in range(BINS)]

STRICT_HEADER = ["planet_id"] + MU_COLS + SIGMA_COLS

# Where we *expect* to find the validator (API or CLI).
VALIDATOR_MODULE_REL = Path("tools") / "validate_submission.py"

# Candidate python invocations used by the CLI fallback (try in order).
PYTHON_BIN_CANDIDATES = [sys.executable, "python", "python3"]

# Some repos may wire a console-script entry point; include common variants here.
CLI_CANDIDATES = [
    # Explicit module file
    ["-m", "tools.validate_submission"],
    # Direct script call (path will be prefixed at runtime with repo root)
    [str(VALIDATOR_MODULE_REL)],
]

# Different repos sometimes choose slightly different flag names; try these.
CLI_FLAGS_MATRIX = [
    # Preferred flag style
    dict(submission="--submission", ids="--ids", bins="--bins",
         require_sigma="--require-sigma", strict_headers="--strict-headers"),
    # Alt spellings (if any)
    dict(submission="--submission-path", ids="--expected-ids", bins="--bins",
         require_sigma="--sigma-required", strict_headers="--strict-headers"),
]


# ------------------------------- Utilities --------------------------------- #

@dataclass
class ValidatorAPI:
    """Holds callable interface to a Python API validator (if available)."""
    func: Callable[..., Tuple[bool, List[str]]]
    supports_kwargs: Dict[str, bool]


def repo_root() -> Path:
    """Return repository root (heuristic: nearest dir containing pyproject.toml or .git)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


def discover_api_validator() -> Optional[ValidatorAPI]:
    """
    Try to import a Python API validator from tools/validate_submission.py.
    We detect supported kwargs by introspection; unknown kwargs won’t be passed.
    """
    root = repo_root()
    module_path = root / VALIDATOR_MODULE_REL
    if not module_path.exists():
        return None

    # Add repo root to sys.path for import
    sys.path.insert(0, str(root))
    try:
        mod = __import__("tools.validate_submission", fromlist=["*"])
    except Exception:
        # Could not import (syntax error or missing); fall back to CLI.
        return None

    # Discover candidate function names
    for name in ["validate_submission", "run_validation", "validate"]:
        func = getattr(mod, name, None)
        if callable(func):
            # Probe supported kwargs by looking at the signature string
            src = getattr(func, "__doc__", "") or ""
            sig_txt = str(func.__annotations__) + " " + src
            # Simple capability flags (best effort)
            supports = {
                "expected_ids": bool(re.search(r"expected_ids|ids", sig_txt)),
                "bins": "bins" in sig_txt or "n_bins" in sig_txt,
                "require_sigma": "require_sigma" in sig_txt or "sigma" in sig_txt,
                "strict_headers": "strict_headers" in sig_txt or "strict" in sig_txt,
            }
            return ValidatorAPI(func=func, supports_kwargs=supports)

    # Function not found; fall back to CLI
    return None


def run_cli_validator(submission: Path, ids_txt: Path, bins: int,
                      require_sigma: bool = True, strict_headers: bool = True) -> Tuple[bool, List[str]]:
    """
    Invoke the validator via CLI (fallback). Tries multiple python and flag spellings.
    Returns (ok, messages).
    """
    root = repo_root()
    messages: List[str] = []

    for py in PYTHON_BIN_CANDIDATES:
        for entry in CLI_CANDIDATES:
            for flags in CLI_FLAGS_MATRIX:
                cmd = [py]
                # If entry is a direct path, resolve against repo root
                if entry and entry[0].endswith(".py"):
                    cmd.append(str(root / entry[0]))
                else:
                    cmd.extend(entry)

                cmd.extend([flags["submission"], str(submission)])
                cmd.extend([flags["ids"], str(ids_txt)])
                cmd.extend([flags["bins"], str(bins)])
                if require_sigma and flags.get("require_sigma"):
                    cmd.append(flags["require_sigma"])
                if strict_headers and flags.get("strict_headers"):
                    cmd.append(flags["strict_headers"])

                try:
                    proc = subprocess.run(
                        cmd, cwd=str(root), check=False,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                except FileNotFoundError:
                    continue  # this python may not exist on PATH
                except Exception as e:
                    messages.append(f"[CLI ERROR] {cmd} → {e}")
                    continue

                out = (proc.stdout or "") + (proc.stderr or "")
                messages.append(f"$ {' '.join(cmd)}\nexit={proc.returncode}\n{out}")

                if proc.returncode == 0:
                    return True, messages
                # else: try next flags or entry
    # All attempts failed
    return False, messages


def write_ids_txt(path: Path, ids: Sequence[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def write_submission_csv(path: Path,
                         rows: Sequence[Tuple[str, List[float], List[float]]],
                         strict_header: bool = True,
                         gzip_compress: bool = False) -> None:
    """
    Write a submission CSV (or CSV.GZ) with provided rows.
    rows: sequence of (planet_id, mu[0..BINS-1], sigma[0..BINS-1])
    """
    header = STRICT_HEADER if strict_header else (["planet_id"] + MU_COLS + SIGMA_COLS)
    # Use in-memory buffer for both .csv and .gz to avoid CRLF surprises
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for pid, mu, sig in rows:
        w.writerow([pid, *mu, *sig])
    data = buf.getvalue()

    if gzip_compress or str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as gz:
            gz.write(data)
    else:
        path.write_text(data, encoding="utf-8")


def tiny_valid_rows(ids: Sequence[str], bins: int = BINS) -> List[Tuple[str, List[float], List[float]]]:
    """Create a tiny set of valid rows with strictly positive σ and finite μ."""
    rows = []
    for j, pid in enumerate(ids):
        mu = [0.001 * (i + 1) + 0.01 * j for i in range(bins)]
        sigma = [0.1 + 1e-6 * (i + 1) for i in range(bins)]  # strictly positive
        rows.append((pid, mu, sigma))
    return rows


def mutate_make_nan(rows: List[Tuple[str, List[float], List[float]]]) -> None:
    rows[0][1][10] = float("nan")  # μ_10 → NaN


def mutate_make_inf(rows: List[Tuple[str, List[float], List[float]]]) -> None:
    rows[0][1][11] = float("inf")  # μ_11 → +Inf


def mutate_make_nonpositive_sigma(rows: List[Tuple[str, List[float], List[float]]]) -> None:
    rows[0][2][12] = 0.0  # σ_12 → 0 (invalid)


def try_api_then_cli(submission: Path,
                     ids_txt: Path,
                     bins: int = BINS,
                     require_sigma: bool = True,
                     strict_headers: bool = True) -> Tuple[bool, List[str]]:
    """
    Attempt API validation first; fall back to CLI.
    Returns (ok, messages).
    """
    api = discover_api_validator()
    if api is not None:
        kwargs = {}
        if api.supports_kwargs.get("expected_ids", True):
            kwargs["expected_ids"] = TEST_PLANET_IDS
        if api.supports_kwargs.get("bins", True):
            kwargs["bins"] = bins
        if api.supports_kwargs.get("require_sigma", True):
            kwargs["require_sigma"] = require_sigma
        if api.supports_kwargs.get("strict_headers", True):
            kwargs["strict_headers"] = strict_headers

        try:
            ok, errors = api.func(str(submission), **kwargs)  # type: ignore[arg-type]
            messages = [f"[API] ok={ok} errors={errors}"]
            return bool(ok), messages + (errors or [])
        except TypeError:
            # Signature mismatch; defer to CLI.
            pass
        except Exception as e:
            return False, [f"[API-EXCEPTION] {e!r}"]

    # CLI fallback
    return run_cli_validator(submission=submission, ids_txt=ids_txt, bins=bins,
                             require_sigma=require_sigma, strict_headers=strict_headers)


# ------------------------------- Fixtures ---------------------------------- #

@pytest.fixture(scope="module")
def tmp_artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temporary directory for all generated artifacts."""
    return tmp_path_factory.mktemp("artifacts_submission_tests")


@pytest.fixture()
def ids_file(tmp_artifacts_dir: Path) -> Path:
    p = tmp_artifacts_dir / "expected_ids.txt"
    write_ids_txt(p, TEST_PLANET_IDS)
    return p


# ------------------------------- Test Cases -------------------------------- #

def test_valid_submission_passes(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "valid_submission.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    write_submission_csv(sub, rows, strict_header=True, gzip_compress=False)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    if not ok:
        pytest.fail("Valid submission should pass.\n" + "\n".join(map(str, msgs)))


def test_valid_gzip_submission_passes(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "valid_submission.csv.gz"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    write_submission_csv(sub, rows, strict_header=True, gzip_compress=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    if not ok:
        pytest.fail("Valid gzip submission should pass.\n" + "\n".join(map(str, msgs)))


def test_wrong_column_count_is_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    # Drop last sigma column to cause a column-count failure
    sub = tmp_artifacts_dir / "bad_columns.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    # Manually write with one fewer column in header
    header = ["planet_id"] + MU_COLS + SIGMA_COLS[:-1]  # missing sigma_282
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for pid, mu, sig in rows:
        w.writerow([pid, *mu, *sig[:-1]])
    sub.write_text(buf.getvalue(), encoding="utf-8")

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission with wrong column count must be rejected."


def test_nan_values_are_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_nan.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    mutate_make_nan(rows)
    write_submission_csv(sub, rows, strict_header=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission containing NaN must be rejected."


def test_inf_values_are_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_inf.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    mutate_make_inf(rows)
    write_submission_csv(sub, rows, strict_header=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission containing Inf must be rejected."


def test_non_positive_sigma_is_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_sigma.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    mutate_make_nonpositive_sigma(rows)
    write_submission_csv(sub, rows, strict_header=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission with non‑positive sigma must be rejected."


def test_duplicate_ids_are_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_dupe_ids.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    # Duplicate the first row (duplicate planet_id)
    dup = (rows[0][0], rows[0][1][:], rows[0][2][:])
    rows.append(dup)
    write_submission_csv(sub, rows, strict_header=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission with duplicate planet_id must be rejected."


def test_missing_ids_are_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_missing_ids.csv"
    # Write a file with only two of the three expected IDs
    rows = tiny_valid_rows(TEST_PLANET_IDS[:2], bins=BINS)
    write_submission_csv(sub, rows, strict_header=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission missing required planet_ids must be rejected."


def test_extra_ids_are_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_extra_ids.csv"
    # Add an unknown planet_id not present in TEST_PLANET_IDS
    rows = tiny_valid_rows(TEST_PLANET_IDS + ["P9999"], bins=BINS)
    write_submission_csv(sub, rows, strict_header=True)

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission with extra / unknown planet_ids must be rejected."


def test_wrong_headers_are_rejected(tmp_artifacts_dir: Path, ids_file: Path):
    sub = tmp_artifacts_dir / "bad_headers.csv"
    rows = tiny_valid_rows(TEST_PLANET_IDS, bins=BINS)
    # Write custom header with a typo to force header validation failure
    header = ["planet_id"] + [f"mu_{i}" for i in range(BINS)]
    header += [f"sigmaa_{i}" for i in range(BINS)]  # typo: 'sigmaa_'
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for pid, mu, sig in rows:
        w.writerow([pid, *mu, *sig])
    sub.write_text(buf.getvalue(), encoding="utf-8")

    ok, msgs = try_api_then_cli(submission=sub, ids_txt=ids_file,
                                bins=BINS, require_sigma=True, strict_headers=True)
    assert not ok, "Submission with incorrect header names must be rejected."


# --------------------------- Helpful Failure Dump -------------------------- #

def pytest_assertrepr_compare(op, left, right):
    """Richer assertion output when comparing (ok == False) if needed."""
    if isinstance(left, bool) and isinstance(right, bool) and op == "==":
        return [f"Boolean compare: {left} == {right}"]


def pytest_runtest_makereport(item, call):
    """
    On failure, show environment hints to debug validator discovery quickly.
    """
    if call.excinfo is not None and call.when == "call":
        root = repo_root()
        extras = [
            f"[debug] repo_root = {root}",
            f"[debug] validator (module) exists = {(root / VALIDATOR_MODULE_REL).exists()}",
            f"[debug] python = {sys.executable}",
        ]
        sys.stderr.write("\n".join(extras) + "\n")
