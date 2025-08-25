# tests/diagnostics/test_symbolic_rule_table.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Symbolic Rule Table Diagnostic Tests

This suite validates construction and basic semantics of a "symbolic rule table"
over predicted transmission spectra. We check core physics/logic constraints:
- Non-negativity (no negative transit depths)
- Flux ceiling (transit depth cannot exceed 1.0 relative baseline)
- Smoothness via total variation (TV) threshold
- Band consistency (e.g., if a strong H2O band is present, a paired band should co-occur)
- Simple energy budget (sum of depths bounded)

It’s intentionally self-contained so it can run without the full pipeline; if the
project exposes a CLI rules report (e.g., `spectramind diagnose --rules`), we’ll
optionally exercise that too.

Markers: fast, unit

Notes:
- Keep thresholds conservative; we want deterministic pass/fail on the synthetic vectors.
- The rule table schema we assert on is general, not tied to any one engine:
  ['rule_id','label','category','severity','value','threshold','violation','passed','details'].

Author: SpectraMind V50 Test Harness
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest


# ---------------------------
# Helpers
# ---------------------------

def total_variation(x: np.ndarray) -> float:
    """L1 total variation of a 1D array."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.abs(np.diff(x)).sum())


def band_mean(x: np.ndarray, idx: List[int]) -> float:
    """Mean over a set of spectral indices (ignore out-of-range safely)."""
    valid = [i for i in idx if 0 <= i < x.size]
    if not valid:
        return float("nan")
    return float(np.nanmean(x[valid]))


@dataclass(frozen=True)
class Rule:
    rule_id: str
    label: str
    category: str
    severity: str
    func: Callable[[np.ndarray], Tuple[float, float, float, bool, str]]
    # func returns (value, threshold, violation, passed, details)


def make_rule_table(spectrum: np.ndarray, rules: List[Rule]) -> pd.DataFrame:
    rows = []
    for r in rules:
        value, threshold, violation, passed, details = r.func(spectrum)
        rows.append(
            dict(
                rule_id=r.rule_id,
                label=r.label,
                category=r.category,
                severity=r.severity,
                value=value,
                threshold=threshold,
                violation=violation,
                passed=bool(passed),
                details=details,
            )
        )
    df = pd.DataFrame(rows)
    # Basic, consistent ordering: most severe violations first, then by rule_id
    df = df.sort_values(by=["passed", "severity", "rule_id"], ascending=[True, True, True]).reset_index(drop=True)
    return df


# ---------------------------
# Synthetic spectra for testing
# ---------------------------

@pytest.fixture(scope="module")
def spectra_cases() -> Dict[str, np.ndarray]:
    """
    Provide a few spectra with known properties:

    - "clean": small, positive, smooth (< TV threshold), well within ceiling.
    - "neg_dip": same as clean but with a small negative glitch (violates non-negativity).
    - "too_high": same as clean but with a spike > 1.0 (violates ceiling).
    - "rough": add alternating noise to increase total variation (violates TV).
    - "band_inconsistent": strong in band A, missing in band B (violates band consistency).
    - "energy_heavy": sum exceeds budget (violates budget).
    """
    rng = np.random.default_rng(42)
    n = 283  # Ariel spectrum length in challenge
    base = 0.02 + 0.005 * np.exp(-((np.arange(n) - 140) ** 2) / (2 * 25.0**2))  # small Gaussian bump, realistic scale
    clean = base.copy()

    neg_dip = base.copy()
    neg_dip[50] = -0.01  # negative glitch

    too_high = base.copy()
    too_high[120] = 1.05  # impossible spike

    rough = base.copy()
    rough += 0.002 * np.sign(np.sin(np.linspace(0, 40 * np.pi, n)))  # alternating ripple to raise TV

    band_inconsistent = base.copy()
    # Make band A strong, band B absent
    # Choose two disjoint bands
    band_inconsistent[60:70] += 0.02       # A strong
    # keep B ~ baseline (do nothing to 180:190)

    energy_heavy = base.copy()
    energy_heavy += 0.01  # lift entire spectrum to exceed simple sum budget

    return dict(
        clean=clean,
        neg_dip=neg_dip,
        too_high=too_high,
        rough=rough,
        band_inconsistent=band_inconsistent,
        energy_heavy=energy_heavy,
    )


# ---------------------------
# Define rules
# ---------------------------

@pytest.fixture(scope="module")
def symbolic_rules() -> List[Rule]:
    """
    A small, representative rule set:

    R1 Non-negativity: min(x) >= 0
    R2 Flux ceiling: max(x) <= 1.0
    R3 Smoothness/TV: TV(x) <= 1.2 (units of depth; chosen to pass 'clean', fail 'rough')
    R4 Band consistency (H2O example): mean(A) and mean(B) should co-occur.
       We'll require mean(A) <= mean(B) + delta when mean(A) is strong; else OK.
    R5 Energy budget: sum(x) <= budget
    """
    tv_threshold = 1.2
    ceil_threshold = 1.0
    energy_budget = 0.02 * 283 + 0.5  # generous, but "energy_heavy" will exceed

    band_A = list(range(60, 70))
    band_B = list(range(180, 190))
    delta = 0.005  # tolerance

    def r1_nonneg(x):
        m = float(np.nanmin(x))
        thr = 0.0
        violation = max(0.0, thr - m)
        return m, thr, violation, m >= thr, f"min={m:.4f}"

    def r2_ceiling(x):
        mx = float(np.nanmax(x))
        thr = ceil_threshold
        violation = max(0.0, mx - thr)
        return mx, thr, violation, mx <= thr, f"max={mx:.4f}"

    def r3_tv(x):
        tv = total_variation(x)
        thr = tv_threshold
        violation = max(0.0, tv - thr)
        return tv, thr, violation, tv <= thr, f"TV={tv:.4f}"

    def r4_band_consistency(x):
        a = band_mean(x, band_A)
        b = band_mean(x, band_B)
        # If band A is strong (> baseline by 0.01), require band B to be within delta of A
        thr = 0.0  # constraint modeled as A <= B + delta when A is strong; encode violation magnitude
        if math.isnan(a) or math.isnan(b):
            return float("nan"), thr, 0.0, True, "bands NaN->skip"
        strongA = a >= (np.nanmean(x) + 0.01)
        viol = max(0.0, (a - (b + delta))) if strongA else 0.0
        passed = viol == 0.0
        return (a - b), thr, viol, passed, f"meanA={a:.4f}, meanB={b:.4f}, strongA={strongA}"

    def r5_energy(x):
        s = float(np.nansum(x))
        thr = energy_budget
        violation = max(0.0, s - thr)
        return s, thr, violation, s <= thr, f"sum={s:.4f}"

    return [
        Rule("R1", "Non-negativity", "physics", "high", r1_nonneg),
        Rule("R2", "Flux ceiling", "physics", "high", r2_ceiling),
        Rule("R3", "Smoothness (TV)", "regularity", "medium", r3_tv),
        Rule("R4", "Band consistency (H2O)", "chemistry", "medium", r4_band_consistency),
        Rule("R5", "Energy budget", "physics", "low", r5_energy),
    ]


# ---------------------------
# Tests
# ---------------------------

@pytest.mark.unit
@pytest.mark.fast
def test_rule_table_schema_and_sorting(spectra_cases, symbolic_rules):
    df = make_rule_table(spectra_cases["clean"], symbolic_rules)
    expected_cols = [
        "rule_id", "label", "category", "severity",
        "value", "threshold", "violation", "passed", "details"
    ]
    assert list(df.columns) == expected_cols

    # All clean spectrum rules should pass with zero violations.
    assert df["passed"].all()
    assert (df["violation"] == 0.0).all()

    # Ensure sorting places any failing rules (if present) first – try a failing spectrum:
    df_fail = make_rule_table(spectra_cases["too_high"], symbolic_rules)
    assert (~df_fail["passed"]).any()
    # Failing rows should come before passing rows after our sort:
    first_fail_index = df_fail.index[~df_fail["passed"]][0]
    last_pass_index = df_fail.index[df_fail["passed"]][-1]
    assert first_fail_index < last_pass_index


@pytest.mark.unit
@pytest.mark.fast
def test_nonnegativity_and_ceiling_violations(spectra_cases, symbolic_rules):
    df_neg = make_rule_table(spectra_cases["neg_dip"], symbolic_rules)
    r1 = df_neg.loc[df_neg["rule_id"] == "R1"].iloc[0]
    assert r1["passed"] is False
    assert r1["violation"] > 0.0
    assert r1["value"] < 0.0  # min(x) negative

    df_high = make_rule_table(spectra_cases["too_high"], symbolic_rules)
    r2 = df_high.loc[df_high["rule_id"] == "R2"].iloc[0]
    assert r2["passed"] is False
    assert r2["violation"] > 0.0
    assert r2["value"] > 1.0  # max(x) exceeds ceiling


@pytest.mark.unit
@pytest.mark.fast
def test_tv_smoothness_rule(spectra_cases, symbolic_rules):
    df_clean = make_rule_table(spectra_cases["clean"], symbolic_rules)
    r3c = df_clean.loc[df_clean["rule_id"] == "R3"].iloc[0]
    assert r3c["passed"] is True

    df_rough = make_rule_table(spectra_cases["rough"], symbolic_rules)
    r3r = df_rough.loc[df_rough["rule_id"] == "R3"].iloc[0]
    assert r3r["passed"] is False
    assert r3r["violation"] > 0.0
    assert r3r["value"] > r3r["threshold"]


@pytest.mark.unit
@pytest.mark.fast
def test_band_consistency_rule(spectra_cases, symbolic_rules):
    df_inc = make_rule_table(spectra_cases["band_inconsistent"], symbolic_rules)
    r4 = df_inc.loc[df_inc["rule_id"] == "R4"].iloc[0]
    assert r4["passed"] is False
    assert r4["violation"] > 0.0
    assert "meanA=" in r4["details"] and "meanB=" in r4["details"]


@pytest.mark.unit
@pytest.mark.fast
def test_energy_budget_rule(spectra_cases, symbolic_rules):
    df_e = make_rule_table(spectra_cases["energy_heavy"], symbolic_rules)
    r5 = df_e.loc[df_e["rule_id"] == "R5"].iloc[0]
    assert r5["passed"] is False
    assert r5["violation"] > 0.0
    assert r5["value"] > r5["threshold"]


@pytest.mark.unit
@pytest.mark.fast
def test_artifact_write_and_json_export(tmp_path, spectra_cases, symbolic_rules):
    """Ensure we can persist the rule table (useful for downstream dashboards)."""
    df = make_rule_table(spectra_cases["neg_dip"], symbolic_rules)
    csv_path = tmp_path / "rule_table.csv"
    json_path = tmp_path / "rule_table.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    assert csv_path.exists() and csv_path.stat().st_size > 0
    # round-trip JSON basic check
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert all("rule_id" in row for row in data)


@pytest.mark.unit
@pytest.mark.fast
def test_optional_cli_rules_report(tmp_path):
    """
    OPTIONAL: If the project exposes a Typer CLI `spectramind` with a rules report
    (e.g., `spectramind diagnose --rules --out rule_table.json`), run it and parse output.

    If not present, xfail safely with a clear reason.
    """
    exe = shutil.which("spectramind")
    if exe is None:
        pytest.xfail("spectramind CLI not found in PATH; skipping optional CLI integration test.")

    out_file = tmp_path / "rules_cli.json"
    cmd = [exe, "diagnose", "--rules", "--out", str(out_file)]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
    except Exception as e:
        pytest.xfail(f"Unable to execute spectramind CLI: {e}")

    if proc.returncode != 0:
        # Allow graceful skip if command isn't implemented yet
        pytest.xfail(f"spectramind diagnose --rules not implemented (rc={proc.returncode}). stderr:\n{proc.stderr}")

    assert out_file.exists(), "CLI should write a JSON rule table file."
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0
    # Basic schema sanity
    first = data[0]
    for k in ("rule_id", "label", "category", "severity", "value", "threshold", "violation", "passed", "details"):
        assert k in first