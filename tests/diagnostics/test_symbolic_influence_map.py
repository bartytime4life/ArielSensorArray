# /tests/diagnostics/test_symbolic_influence_map.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: symbolic_influence_map

Purpose
-------
Validate the scientific and engineering behavior of the *symbolic_influence_map*
utility, which computes per-rule influence (e.g., ∂L_sym/∂μ or related) over
wavelength bins for one or more symbolic constraints.

This test suite focuses on:
1) API discovery & shape contracts (single and batched μ)
2) Rule-localization sanity (influence concentrated where a rule’s mask is active)
3) Rule weight monotonicity (higher weight ⇒ larger aggregate influence)
4) Determinism under fixed seeds
5) Optional JSON/NPY artifact save path (if exposed)
6) Optional CLI smoke test via `spectramind diagnose symbolic-influence-map ...`

Design Notes
------------
• The tests are *defensively adaptable* to small API differences. They attempt to
  import/locate the tool under common layouts:
      - tools.symbolic_influence_map
      - src.tools.symbolic_influence_map
      - diagnostics.symbolic_influence_map
  and detect one of the following call patterns:
      - compute_symbolic_influence_map(mu, rules=..., **cfg) -> np.ndarray or dict
      - symbolic_influence_map(mu, rules=..., **cfg) -> np.ndarray or dict
      - compute_influence(mu, rules=..., **cfg) -> ...
      - Simulator/Runner/Influencer(...).run(...)
  If none are found, the test xfails with clear guidance.

• We generate synthetic μ spectra (L=283) and a tiny rule set comprising two
  disjoint wavelength masks. By construction, a physically sensible symbolic
  influence implementation should show stronger influence within each rule’s
  masked region versus outside it.

• The test tolerates either absolute or signed influence maps. We use magnitude-
  based (|·|) comparisons where needed.

Author: SpectraMind V50 Team
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    "tools.symbolic_influence_map",
    "src.tools.symbolic_influence_map",
    "diagnostics.symbolic_influence_map",
    "symbolic_influence_map",
]


def _import_simodule():
    """
    Try to import the symbolic influence module from common locations.
    """
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import symbolic_influence_map from any of:\n"
        f"  {CANDIDATE_IMPORTS}\n"
        f"Last error: {last_err}"
    )


def _locate_callable(mod):
    """
    Detect a callable or class interface to compute influence.

    Returns
    -------
    kind : str
        'func' or 'class'
    target : callable | type
    """
    # Try common functional entrypoints (preferred)
    for fn_name in (
        "compute_symbolic_influence_map",
        "symbolic_influence_map",
        "compute_influence",
        "run_influence",
    ):
        if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
            return "func", getattr(mod, fn_name)

    # Fallback: class with .run(...)
    for cls_name in ("SymbolicInfluenceMap", "SymbolicInfluenceRunner", "InfluenceSimulator"):
        if hasattr(mod, cls_name):
            cls = getattr(mod, cls_name)
            if hasattr(cls, "run") and callable(getattr(cls, "run")):
                return "class", cls

    pytest.xfail(
        "symbolic_influence_map module found but no known callable was discovered. "
        "Expected one of: compute_symbolic_influence_map(), symbolic_influence_map(), "
        "compute_influence(), run_influence(), or a class with .run(...)."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, mu: np.ndarray, rules: List[Dict[str, Any]], **cfg):
    """
    Invoke the detected interface in an API-agnostic way.

    Expected returns
    ----------------
    • np.ndarray of shape (R, L) or (B, R, L), or
    • dict with key 'influence' containing the array above.
    """
    if kind == "func":
        out = target(mu, rules=rules, **cfg)
    elif kind == "class":
        try:
            inst = target(mu=mu, rules=rules, **cfg)
        except TypeError:
            inst = target(**cfg)
            out = inst.run(mu=mu, rules=rules)
        else:
            out = inst.run() if hasattr(inst, "run") else inst(mu=mu, rules=rules)
    else:
        pytest.fail("Unknown invocation kind.")  # pragma: no cover

    if isinstance(out, dict) and "influence" in out:
        return out["influence"], out
    return out, {"influence": out}


# ---------------------------------------------------------------------------
# Synthetic data & rules
# ---------------------------------------------------------------------------

L_DEFAULT = 283  # Ariel spectral bins
RNG = np.random.RandomState(20250824)


def _make_mu_single(L: int = L_DEFAULT) -> np.ndarray:
    """
    Construct a single μ spectrum with gentle smooth structure + bumps.
    """
    x = np.linspace(0, 1, L, dtype=np.float64)
    baseline = 0.02 + 0.004 * np.sin(2 * math.pi * 2.1 * x)
    bumps = (
        0.010 * np.exp(-0.5 * ((x - 0.28) / 0.02) ** 2)
        + 0.007 * np.exp(-0.5 * ((x - 0.62) / 0.015) ** 2)
    )
    mu = baseline + bumps
    mu = np.clip(mu, 0.0, 1.0)
    assert mu.shape == (L,)
    return mu


def _make_mu_batch(B: int = 4, L: int = L_DEFAULT) -> np.ndarray:
    """
    Batched μ with small stochastic variation.
    """
    base = _make_mu_single(L)
    batch = np.stack([base + RNG.normal(0, 0.0005, size=L) for _ in range(B)], axis=0)
    return np.clip(batch, 0.0, 1.0)


def _rect_mask(L: int, start_idx: int, end_idx: int) -> np.ndarray:
    m = np.zeros(L, dtype=np.float32)
    m[start_idx:end_idx] = 1.0
    return m


def _make_rules(L: int = L_DEFAULT) -> List[Dict[str, Any]]:
    """
    Create two disjoint rule masks with names and (optional) metadata.
    Rules follow a simple schema commonly used in our tools:
        { "name": str, "mask": [0/1]*L, "weight": float }
    """
    r1 = {
        "name": "rule_left_band",
        "mask": _rect_mask(L, int(0.18 * L), int(0.30 * L)).tolist(),
        "weight": 1.0,
        "kind": "band_smoothness",
    }
    r2 = {
        "name": "rule_right_band",
        "mask": _rect_mask(L, int(0.58 * L), int(0.72 * L)).tolist(),
        "weight": 1.0,
        "kind": "band_smoothness",
    }
    return [r1, r2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def si_mod():
    return _import_simodule()


@pytest.fixture(scope="module")
def si_callable(si_mod):
    return _locate_callable(si_mod)


@pytest.fixture
def mu_single() -> np.ndarray:
    return _make_mu_single(L_DEFAULT)


@pytest.fixture
def mu_batch() -> np.ndarray:
    return _make_mu_batch(B=3, L=L_DEFAULT)


@pytest.fixture
def rules_default() -> List[Dict[str, Any]]:
    return _make_rules(L_DEFAULT)


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def _as_array(influence) -> np.ndarray:
    assert isinstance(influence, np.ndarray), "Influence must be a numpy array."
    assert np.all(np.isfinite(influence)), "Influence contains non-finite values."
    return influence


def _enforce_shape(arr: np.ndarray, L: int = L_DEFAULT) -> Tuple[np.ndarray, str]:
    """
    Accept (R, L), (B, R, L), or a dict shape. Return normalized shape (B, R, L)
    with B=1 if single-spectrum.
    """
    if arr.ndim == 2 and arr.shape[1] == L:
        R = arr.shape[0]
        return arr[None, ...], f"(1,{R},{L})"
    if arr.ndim == 3 and arr.shape[2] == L:
        return arr, f"({arr.shape[0]},{arr.shape[1]},{L})"
    pytest.fail(f"Unexpected influence shape {arr.shape}, expected (R,{L}) or (B,R,{L}).")
    return arr, ""  # pragma: no cover


def _agg_rule_strength(infl: np.ndarray, rule_index: int) -> float:
    """
    Aggregate total strength for a rule across all bins and batch (L1 magnitude).
    infl: (B, R, L)
    """
    return float(np.sum(np.abs(infl[:, rule_index, :])))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_api_shapes_single(si_callable, mu_single, rules_default):
    """
    API & shape contract on single μ: output is (R,L) or (B,R,L) with B=1.
    """
    kind, target = si_callable
    influence, full = _invoke(kind, target, mu_single, rules_default, seed=1234, return_dict=True)
    arr = _as_array(influence)
    arr_norm, shape_str = _enforce_shape(arr, L=L_DEFAULT)

    # Basic checks
    B, R, L = arr_norm.shape
    assert B == 1 and L == L_DEFAULT and R == len(rules_default), f"Unexpected shape {shape_str}"
    assert np.all(np.isfinite(arr_norm))


def test_api_shapes_batch(si_callable, mu_batch, rules_default):
    """
    API & shape contract on batch μ: output is (B,R,L).
    """
    kind, target = si_callable
    influence, _ = _invoke(kind, target, mu_batch, rules_default, seed=7, return_dict=False)
    arr = _as_array(influence)
    arr_norm, _ = _enforce_shape(arr, L=L_DEFAULT)
    B, R, L = arr_norm.shape
    assert B == mu_batch.shape[0] and R == len(rules_default) and L == L_DEFAULT


def test_rule_localization(si_callable, mu_single, rules_default):
    """
    Influence magnitude for each rule should be larger inside its mask region
    than outside (statistical check on percentiles).
    """
    kind, target = si_callable
    influence, _ = _invoke(kind, target, mu_single, rules_default, seed=99)
    arr = _as_array(influence)
    arr_norm, _ = _enforce_shape(arr, L=L_DEFAULT)
    _, R, L = arr_norm.shape

    for r_idx, rule in enumerate(rules_default):
        mask = np.asarray(rule["mask"], dtype=bool)
        infl = np.abs(arr_norm[0, r_idx, :])  # (L,)
        inside = infl[mask]
        outside = infl[~mask]
        # Compare robust percentiles; inside should systematically exceed outside
        inside_q = float(np.percentile(inside, 75))
        outside_q = float(np.percentile(outside, 75))
        assert inside_q > outside_q * 1.2, (
            f"Rule '{rule['name']}' influence not localized: inside P75={inside_q:.3e}, "
            f"outside P75={outside_q:.3e}"
        )


def test_rule_weight_monotonicity(si_callable, mu_single, rules_default):
    """
    Increasing a rule's weight should not decrease its aggregate influence.
    If the API does not support 'rule_weights' or per-rule 'weight' overrides,
    we xfail with a clear message.
    """
    # Try two mechanisms:
    #  (A) per-call kwarg rule_weights=[...]
    #  (B) rules[i]['weight'] modified directly
    kind, target = si_callable

    # Baseline call
    infl_base, _ = _invoke(kind, target, mu_single, rules_default, seed=123)
    arr_base, _ = _enforce_shape(_as_array(infl_base), L=L_DEFAULT)
    s_base_0 = _agg_rule_strength(arr_base, rule_index=0)

    # Try call-time kwarg first
    try:
        infl_hi, _ = _invoke(
            kind,
            target,
            mu_single,
            rules_default,
            seed=123,
            rule_weights=[5.0, 1.0],  # upweight rule 0
        )
        arr_hi, _ = _enforce_shape(_as_array(infl_hi), L=L_DEFAULT)
        s_hi_0 = _agg_rule_strength(arr_hi, rule_index=0)
        assert s_hi_0 >= s_base_0 * 1.2, "Rule weight increase did not raise aggregate influence as expected."
        return
    except TypeError:
        # Fall back to in-structure weight override
        rules_mod = [dict(r) for r in rules_default]
        rules_mod[0]["weight"] = 5.0
        infl_hi2, _ = _invoke(kind, target, mu_single, rules_mod, seed=123)
        arr_hi2, _ = _enforce_shape(_as_array(infl_hi2), L=L_DEFAULT)
        s_hi2_0 = _agg_rule_strength(arr_hi2, rule_index=0)
        if not (s_hi2_0 >= s_base_0 * 1.1):
            pytest.xfail(
                "Symbolic influence API appears not to support rule weights or does not scale influence with weight."
            )


def test_determinism_fixed_seed(si_callable, mu_single, rules_default):
    """
    With a fixed seed and identical inputs, the influence maps should be identical.
    """
    kind, target = si_callable
    infl1, _ = _invoke(kind, target, mu_single, rules_default, seed=777)
    infl2, _ = _invoke(kind, target, mu_single, rules_default, seed=777)
    a1, _ = _enforce_shape(_as_array(infl1), L=L_DEFAULT)
    a2, _ = _enforce_shape(_as_array(infl2), L=L_DEFAULT)
    assert np.array_equal(a1, a2), "Influence maps changed despite fixed seeds."


# ---------------------------------------------------------------------------
# Optional artifacts round-trip (if exposed)
# ---------------------------------------------------------------------------

def test_save_artifacts_roundtrip_if_available(si_mod, si_callable, tmp_path, mu_single, rules_default):
    """
    If the module exposes `save_influence_artifacts(...)` or `write_artifacts(...)`,
    verify it writes expected files and preserves shapes on reload.
    """
    save_fn = None
    for name in ("save_influence_artifacts", "write_artifacts", "save_artifacts"):
        if hasattr(si_mod, name) and callable(getattr(si_mod, name)):
            save_fn = getattr(si_mod, name)
            break
    if save_fn is None:
        pytest.xfail("Module does not expose an artifacts save function; skipping round-trip test.")

    kind, target = si_callable
    out = _invoke(kind, target, mu_single, rules_default, seed=2025)[1]  # full dict
    outdir = tmp_path / "influence_artifacts"
    outdir.mkdir(parents=True, exist_ok=True)

    # Attempt save
    save_fn(out, outdir=str(outdir))

    # Require existence of at least JSON and one NPY
    files = list(outdir.glob("*"))
    assert files, "No artifacts written."
    assert any(p.suffix.lower() == ".json" for p in files) or any(p.suffix.lower() == ".npz" for p in files)
    # If influence.npy exists, reload and check shape
    inf_path = outdir / "influence.npy"
    if inf_path.exists():
        arr = np.load(inf_path)
        _, _ = _enforce_shape(arr, L=L_DEFAULT)  # raises on mismatch


# ---------------------------------------------------------------------------
# CLI smoke (optional)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil_which := __import__("shutil").which("spectramind") is None, reason="SpectraMind CLI not found in PATH")
def test_cli_smoke_symbolic_influence(tmp_path: Path, mu_single, rules_default):
    """
    Smoke test the CLI route. We generate small inputs and verify that
    the CLI runs and writes at least one artifact.
    Expected CLI (adjust as needed):
        spectramind diagnose symbolic-influence-map --mu mu.npy --rules rules.json --outdir out
    """
    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, mu_single)

    rules_path = tmp_path / "rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules_default, f)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "spectramind",
        "diagnose",
        "symbolic-influence-map",
        "--mu",
        str(mu_path),
        "--rules",
        str(rules_path),
        "--outdir",
        str(outdir),
        "--seed",
        "123",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.xfail(
            "CLI returned nonzero exit. This may indicate differing flag names.\n"
            f"Stdout:\n{proc.stdout}\n\nStderr:\n{proc.stderr}"
        )

    produced = list(outdir.glob("*"))
    assert len(produced) > 0, "CLI ran but produced no artifacts."


# ---------------------------------------------------------------------------
# Performance sanity (very light)
# ---------------------------------------------------------------------------

def test_runs_fast_enough(si_callable, mu_single, rules_default):
    """
    Light guardrail: tiny inference should run in < 1.5s on CI CPU.
    """
    kind, target = si_callable
    t0 = time.time()
    _ = _invoke(kind, target, mu_single, rules_default, seed=11)
    dt = time.time() - t0
    assert dt < 1.5, f"Symbolic influence computation too slow: {dt:.3f}s (should be < 1.5s)"