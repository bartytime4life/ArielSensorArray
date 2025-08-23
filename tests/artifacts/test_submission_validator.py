# tests/artifacts/test_submission_validator.py
# SpectraMind V50 — NeurIPS Ariel Data Challenge 2025
#
# Purpose
# -------
# Guard-rail tests for the final submission artifact to Kaggle.
# These tests are intentionally resilient to small schema variations across
# experiments while still catching the common failure cases before CI/CD ships
# a bad file.
#
# What we validate (in order of importance):
# 1) A submission file exists (or is provided via env/pytest -k).
# 2) It’s a readable CSV (optionally .gz), has a string ID column with unique IDs.
# 3) All numeric prediction columns are finite (no NaN/inf).
# 4) If uncertainty columns exist, they are strictly positive.
# 5) Optional, schema sanity (column counts / naming) if the team provides hints
#    via environment variables (see “Tunable expectations” below).
#
# Tunable expectations (all optional; leave unset for looser checks):
#  - SUBMISSION_PATH            : explicit path to test instead of globbing
#  - SUBMISSION_GLOB            : glob used to discover a candidate file
#                                 (default: artifacts/**/submission*.csv*)
#  - SUBMISSION_ID_COL          : exact id column (default: auto-detect)
#  - EXPECT_MEAN_PREFIX         : prefix for mean columns (e.g. "mu_" or "y_")
#  - EXPECT_SIGMA_PREFIX        : prefix for sigma columns (e.g. "sigma_" or "s_")
#  - EXPECT_WAVELENGTHS         : integer expected number of wavelengths (e.g. 283)
#
# Usage
# -----
#   pytest -q tests/artifacts/test_submission_validator.py
#
# Notes
# -----
#  - This test does *not* mutate your repository or artifacts.
#  - If you want stricter checks (e.g., exact column names), pin the environment
#    variables above in your CI job for this test job only.
#
# © SpectraMind V50 Team — NASA-grade reproducibility, data-first rigor.

from __future__ import annotations

import os
import gzip
import io
import glob
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import pytest


# -----------------------------
# Helpers
# -----------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default


def _discover_submission_file() -> Path:
    """Find the submission artifact.

    Priority:
      1) SUBMISSION_PATH if set
      2) First match of SUBMISSION_GLOB (default: artifacts/**/submission*.csv*)

    Raises:
      AssertionError if not found.
    """
    explicit = _env("SUBMISSION_PATH")
    if explicit:
        p = Path(explicit)
        assert p.exists(), f"SUBMISSION_PATH does not exist: {p}"
        return p

    pattern = _env("SUBMISSION_GLOB", "artifacts/**/submission*.csv*")
    matches = sorted([Path(p) for p in glob.glob(pattern, recursive=True)])
    assert matches, (
        "No submission file found.\n"
        f"Tried: SUBMISSION_PATH={explicit!r} or glob={pattern!r}\n"
        "Tip: export SUBMISSION_PATH=artifacts/submissions/latest.csv"
    )
    return matches[0]


def _read_csv_any(path: Path) -> pd.DataFrame:
    """Read CSV or CSV.GZ with pandas, preserving strings for ID columns."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
        return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(path)


def _autodetect_id_col(df: pd.DataFrame) -> str:
    """Pick a reasonable ID column if not provided.

    Heuristics:
      - Respect SUBMISSION_ID_COL if set.
      - Else first column named any of: ["id", "ID", "planet_id", "object_id"] if present (case-insensitive).
      - Else first non-numeric column.
      - Else raise.
    """
    forced = _env("SUBMISSION_ID_COL")
    if forced:
        assert forced in df.columns, f"SUBMISSION_ID_COL={forced!r} not in columns"
        return forced

    lower = {c.lower(): c for c in df.columns}
    for candidate in ("id", "planet_id", "object_id"):
        if candidate in lower:
            return lower[candidate]

    # fallback: first non-numeric column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise AssertionError(
        "Could not detect ID column. Provide SUBMISSION_ID_COL or include a non-numeric identifier column."
    )


def _split_columns(
    df: pd.DataFrame, id_col: str
) -> Tuple[List[str], List[str], List[str]]:
    """Split columns into: id, mean candidates, sigma candidates.

    Mean candidates:
      - numeric columns that do NOT appear to be uncertainty/sigma
    Sigma candidates:
      - columns whose names contain one of {"sigma", "std", "unc", "err"}
      - OR columns with EXPECT_SIGMA_PREFIX
    """
    expect_mu_prefix = _env("EXPECT_MEAN_PREFIX")
    expect_s_prefix = _env("EXPECT_SIGMA_PREFIX")

    # exclude id
    numeric_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]

    sigma_flags = set()
    for c in numeric_cols:
        name = c.lower()
        if expect_s_prefix and c.startswith(expect_s_prefix):
            sigma_flags.add(c)
            continue
        if any(tag in name for tag in ("sigma", "std", "unc", "err")):
            sigma_flags.add(c)

    sigma_cols = sorted(list(sigma_flags))
    mu_cols = sorted([c for c in numeric_cols if c not in sigma_flags])

    # If user provided a mean prefix, filter for extra confidence (but keep all numeric if empty)
    if expect_mu_prefix:
        prefixed = [c for c in mu_cols if c.startswith(expect_mu_prefix)]
        if prefixed:
            mu_cols = sorted(prefixed)

    return [id_col], mu_cols, sigma_cols


def _assert_monotonic_wavelengths_if_named(mu_cols: List[str]) -> None:
    """Light heuristic: if mean columns look like wavelength-indexed names
    (e.g., w000, w001, ... or mu_000, mu_001, ...) ensure they’re contiguous.
    This is a soft sanity check; it won’t fail if names are free-form.
    """
    # pull trailing integer tokens
    idxs = []
    for c in mu_cols:
        tok = "".join(ch for ch in c if ch.isdigit())
        if tok != "":
            try:
                idxs.append(int(tok))
            except ValueError:
                pass
    if len(idxs) >= 4:
        idxs = sorted(set(idxs))
        diffs = np.diff(idxs)
        assert np.all(diffs == 1), (
            "Mean columns appear indexed but are not contiguous. "
            f"Detected indices like {idxs[:10]} ...; please ensure no gaps."
        )


# -----------------------------
# Tests
# -----------------------------
@pytest.fixture(scope="session")
def submission_path() -> Path:
    return _discover_submission_file()


@pytest.fixture(scope="session")
def submission_df(submission_path: Path) -> pd.DataFrame:
    df = _read_csv_any(submission_path)
    assert not df.empty, f"Submission is empty: {submission_path}"
    return df


def test_submission_has_id_and_predictions(submission_df: pd.DataFrame) -> None:
    id_col = _autodetect_id_col(submission_df)
    assert id_col in submission_df.columns, "ID column not found"
    assert submission_df[id_col].notna().all(), "ID column has missing values"
    # Must have at least one numeric prediction column
    numeric_cols = [c for c in submission_df.columns if c != id_col and pd.api.types.is_numeric_dtype(submission_df[c])]
    assert numeric_cols, "No numeric prediction columns found"


def test_id_values_unique(submission_df: pd.DataFrame) -> None:
    id_col = _autodetect_id_col(submission_df)
    dupes = submission_df[id_col][submission_df[id_col].duplicated()].unique()
    assert dupes.size == 0, f"Duplicate IDs detected: {dupes[:10]}"


def test_numeric_values_are_finite(submission_df: pd.DataFrame) -> None:
    id_col = _autodetect_id_col(submission_df)
    value_cols = [c for c in submission_df.columns if c != id_col and pd.api.types.is_numeric_dtype(submission_df[c])]
    assert value_cols, "No numeric columns to validate"
    arr = submission_df[value_cols].to_numpy(dtype=np.float64, copy=False)
    assert np.isfinite(arr).all(), "Found NaN/inf in prediction columns"


def test_sigma_columns_positive_if_present(submission_df: pd.DataFrame) -> None:
    id_col = _autodetect_id_col(submission_df)
    _, _, sigma_cols = _split_columns(submission_df, id_col)
    if not sigma_cols:
        pytest.skip("No sigma/uncertainty columns detected — skipping positivity check")
    sig = submission_df[sigma_cols].to_numpy(dtype=np.float64, copy=False)
    assert (sig > 0).all(), f"Sigma/uncertainty columns must be strictly positive; offending columns: {sigma_cols}"


def test_expected_wavelength_count_if_configured(submission_df: pd.DataFrame) -> None:
    expected = _env("EXPECT_WAVELENGTHS")
    if not expected:
        pytest.skip("EXPECT_WAVELENGTHS not set — skipping strict count check")
    exp = int(expected)
    id_col = _autodetect_id_col(submission_df)
    _, mu_cols, sigma_cols = _split_columns(submission_df, id_col)

    # If both μ and σ are present, require each to match exp (common in GLL settings).
    if sigma_cols:
        assert len(mu_cols) == exp, f"Expected {exp} mean columns, found {len(mu_cols)}"
        assert len(sigma_cols) == exp, f"Expected {exp} sigma columns, found {len(sigma_cols)}"
    else:
        # Mean-only submissions
        assert len(mu_cols) == exp, f"Expected {exp} prediction columns, found {len(mu_cols)}"

    # Soft heuristic on naming consistency
    _assert_monotonic_wavelengths_if_named(mu_cols)


def test_values_within_sane_bounds(submission_df: pd.DataFrame) -> None:
    """Catch egregious scale bugs (exploding values). We keep this conservative.

    Ariel spectra are typically order ~ppm or small fractions; we allow a very wide bound to avoid
    false positives in unit changes while still catching accidental 1e12 spikes.
    """
    id_col = _autodetect_id_col(submission_df)
    numeric_cols = [c for c in submission_df.columns if c != id_col and pd.api.types.is_numeric_dtype(submission_df[c])]
    arr = submission_df[numeric_cols].to_numpy(dtype=np.float64, copy=False)
    max_abs = np.nanmax(np.abs(arr))
    assert max_abs < 1e9, f"Values look wildly out of range (|max|={max_abs:.3g}); check unit scaling/bugs"


def test_optional_schema_quality_notes(submission_df: pd.DataFrame) -> None:
    """Non-failing test that emits helpful notes to the log when schema is unusual."""
    id_col = _autodetect_id_col(submission_df)
    _, mu_cols, sigma_cols = _split_columns(submission_df, id_col)

    if not sigma_cols:
        # Encourage but do not force uncertainty reporting
        print(
            "[NOTE] No sigma/uncertainty columns detected. "
            "If your leaderboard metric rewards calibrated uncertainties (e.g., GLL), "
            "consider including per-wavelength sigma."
        )

    if len(mu_cols) <= 10:
        print(
            f"[NOTE] Only {len(mu_cols)} numeric mean columns detected. "
            "If this is not a toy run, your submission may be missing wavelengths."
        )
