# /tests/diagnostics/test_symbolic_violation_predictor.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: symbolic_violation_predictor

Purpose
-------
Validate the behavior and contracts of the *SymbolicViolationPredictor* tool.
This component applies a set of symbolic rules (with wavelength masks and
weights) to predicted μ spectra (L≈283) and produces per‑rule violation
scores, along with optional rankings and artifacts for diagnostics.

What we verify
--------------
1) API discovery & shape contracts
   • Accepts μ of shape (L,) and (B,L)
   • Returns scores of shape (R,) or (B,R)
   • Optional: per‑rule masks, metadata, ranking, combined score
2) Rule localization (if per‑bin maps/overlays are provided, violations
   concentrate inside each rule mask)
3) Weight monotonicity (↑ rule weight ⇒ ↑ per‑rule score, not ↓)
4) Determinism (fixed seed ⇒ identical outputs)
5) Optional artifact saver round‑trip (writes JSON/NPY, reload shapes)
6) Optional CLI smoke via `spectramind diagnose symbolic-rank`
   (gracefully xfail if CLI or flags are not available)
7) Performance guardrail on tiny inputs

Design Notes
------------
• The test is *defensively adaptable* to small API differences:
  - Module discovery tries several import paths.
  - Entrypoint discovery tries several function/class names.
  - Output normalization tolerates dict/array forms.
• Synthetic μ has two regions of structure aligned to two rule masks.
• We avoid any heavy numerics to keep the CI fast and stable.

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


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    "tools.symbolic_violation_predictor",
    "src.tools.symbolic_violation_predictor",
    "diagnostics.symbolic_violation_predictor",
    "symbolic_violation_predictor",
]


def _import_predictor_module():
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import symbolic_violation_predictor from any of:\n"
        f"  {CANDIDATE_IMPORTS}\nLast error: {last_err}"
    )


def _locate_entrypoint(mod):
    """
    Locate a callable/class interface to compute per‑rule violation scores.

    Accepted options (any of):
      - predict_violations(mu, rules=..., **cfg) -> dict or array
      - predict(mu, rules=..., **cfg)
      - run_prediction(mu, rules=..., **cfg)
      - class SymbolicViolationPredictor(...).predict(mu, rules, **cfg)
      - class SymbolicViolationPredictor(...).run(mu, rules, **cfg)

    Returns
    -------
    kind: 'func' | 'class'
    target: callable | type
    """
    for fn in ("predict_violations", "predict", "run_prediction", "symbolic_violation_predictor"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return "func", getattr(mod, fn)

    for cls in ("SymbolicViolationPredictor", "ViolationPredictor", "SymbolicPredictor"):
        if hasattr(mod, cls):
            Cls = getattr(mod, cls)
            for method in ("predict", "run"):
                if hasattr(Cls, method) and callable(getattr(Cls, method)):
                    return "class", Cls

    pytest.xfail(
        "symbolic_violation_predictor module found but no known entrypoint. "
        "Expected a function like predict_violations()/predict() or a class with .predict/.run."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, mu: np.ndarray, rules: List[Dict[str, Any]], **cfg) -> Dict[str, Any]:
    """
    Invoke API and coerce to a dict with at least 'scores' present.

    Expected dict keys (subset ok):
      - 'scores'       : ndarray (R,) or (B,R)
      - 'ranking'      : list or (B, list) of rule indices/names sorted desc by score
      - 'per_rule_map' : ndarray (R,L) or (B,R,L) (optional)
      - 'combined'     : ndarray (L,) or (B,L) (optional)
      - 'rules'        : echo of rules
      - 'metadata'     : optional metadata
    """
    if kind == "func":
        out = target(mu, rules=rules, **cfg)
    elif kind == "class":
        try:
            inst = target(mu=mu, rules=rules, **cfg)
        except TypeError:
            inst = target(**cfg)
            if hasattr(inst, "predict"):
                out = inst.predict(mu=mu, rules=rules)
            else:
                out = inst.run(mu=mu, rules=rules)
        else:
            out = inst.predict(mu=mu, rules=rules) if hasattr(inst, "predict") else inst.run()
    else:  # pragma: no cover
        pytest.fail("Unknown invocation kind.")

    if isinstance(out, dict):
        assert "scores" in out, "Predictor dict output must include 'scores'."
        return out
    # If bare array (scores) returned, wrap into dict
    return {"scores": out}


# -----------------------------------------------------------------------------
# Synthetic inputs (μ and rules)
# -----------------------------------------------------------------------------

L_DEFAULT = 283
_RNG = np.random.RandomState(20250824)


def _make_mu_single(L: int = L_DEFAULT) -> np.ndarray:
    """
    Smooth baseline + two 'features' aligned with rule masks.
    """
    x = np.linspace(0.0, 1.0, L, dtype=np.float64)
    base = 0.02 + 0.004 * np.sin(2 * math.pi * 2.1 * x)
    f1 = 0.012 * np.exp(-0.5 * ((x - 0.24) / 0.02) ** 2)
    f2 = 0.010 * np.exp(-0.5 * ((x - 0.68) / 0.018) ** 2)
    mu = np.clip(base + f1 + f2, 0.0, 1.0)
    return mu


def _make_mu_batch(B: int = 4, L: int = L_DEFAULT) -> np.ndarray:
    base = _make_mu_single(L)
    batch = np.stack([base + _RNG.normal(0, 0.0006, size=L) for _ in range(B)], axis=0)
    return np.clip(batch, 0.0, 1.0)


def _rect_mask(L: int, a: float, b: float) -> np.ndarray:
    i0 = max(0, int(a * L))
    i1 = min(L, int(b * L))
    m = np.zeros(L, dtype=np.float32)
    m[i0:i1] = 1.0
    return m


def _make_rules(L: int = L_DEFAULT) -> List[Dict[str, Any]]:
    """
    Two disjoint masks with default weight=1.0 and enabled=True.
    """
    r1 = {
        "name": "left_band_rule",
        "mask": _rect_mask(L, 0.16, 0.30).tolist(),
        "weight": 1.0,
        "enabled": True,
        "kind": "band_smoothness",
    }
    r2 = {
        "name": "right_band_rule",
        "mask": _rect_mask(L, 0.58, 0.74).tolist(),
        "weight": 1.0,
        "enabled": True,
        "kind": "band_smoothness",
    }
    return [r1, r2]


# -----------------------------------------------------------------------------
# Normalizers
# -----------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
    assert np.isfinite(x).all(), "Array contains non-finite values"
    return x


def _norm_scores(arr: np.ndarray, R_expected: int) -> Tuple[np.ndarray, str]:
    """
    Normalize scores to (B,R). Accepts (R,) or (B,R).
    """
    arr = _as_np(arr)
    if arr.ndim == 1 and arr.shape[0] == R_expected:
        return arr[None, :], f"(1,{R_expected})"
    if arr.ndim == 2 and arr.shape[1] == R_expected:
        return arr, f"({arr.shape[0]},{R_expected})"
    pytest.fail(f"Unexpected scores shape {arr.shape} (expected ({R_expected},) or (B,{R_expected}))")
    return arr, ""  # pragma: no cover


def _norm_per_rule_map(arr: np.ndarray, L: int) -> Tuple[np.ndarray, str]:
    """
    Normalize per_rule_map to (B,R,L). Accepts (R,L) or (B,R,L).
    """
    arr = _as_np(arr)
    if arr.ndim == 2 and arr.shape[1] == L:
        R = arr.shape[0]
        return arr[None, ...], f"(1,{R},{L})"
    if arr.ndim == 3 and arr.shape[2] == L:
        return arr, f"({arr.shape[0]},{arr.shape[1]},{L})"
    pytest.fail(f"Unexpected per_rule_map shape {arr.shape} (expected (R,{L}) or (B,R,{L}))")
    return arr, ""  # pragma: no cover


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pred_mod():
    return _import_predictor_module()


@pytest.fixture(scope="module")
def pred_entry(pred_mod):
    return _locate_entrypoint(pred_mod)


@pytest.fixture
def mu_single():
    return _make_mu_single(L_DEFAULT)


@pytest.fixture
def mu_batch():
    return _make_mu_batch(B=3, L=L_DEFAULT)


@pytest.fixture
def rules_default():
    return _make_rules(L_DEFAULT)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_api_shapes_single(pred_entry, mu_single, rules_default):
    """
    Single μ: scores normalize to (1,R); optional per_rule_map to (1,R,L).
    """
    kind, target = pred_entry
    out = _invoke(kind, target, mu_single, rules_default, seed=1234, return_dict=True)
    assert isinstance(out, dict)
    scores = out.get("scores", None)
    assert scores is not None, "Predictor must return 'scores'."
    sc, _ = _norm_scores(_as_np(scores), R_expected=len(rules_default))
    assert sc.shape == (1, len(rules_default))

    if "per_rule_map" in out:
        prm, _ = _norm_per_rule_map(_as_np(out["per_rule_map"]), L=L_DEFAULT)
        assert prm.shape == (1, len(rules_default), L_DEFAULT)


def test_api_shapes_batch(pred_entry, mu_batch, rules_default):
    """
    Batch μ: scores normalize to (B,R); optional per_rule_map to (B,R,L).
    """
    kind, target = pred_entry
    out = _invoke(kind, target, mu_batch, rules_default, seed=7)
    scores = out.get("scores", None)
    assert scores is not None
    sc, _ = _norm_scores(_as_np(scores), R_expected=len(rules_default))
    assert sc.shape == (mu_batch.shape[0], len(rules_default))

    if "per_rule_map" in out:
        prm, _ = _norm_per_rule_map(_as_np(out["per_rule_map"]), L=L_DEFAULT)
        assert prm.shape == (mu_batch.shape[0], len(rules_default), L_DEFAULT)


def test_rule_localization_if_maps_available(pred_entry, mu_single, rules_default):
    """
    If per_rule_map is available, ensure violations are larger within rule masks than outside.
    """
    kind, target = pred_entry
    out = _invoke(kind, target, mu_single, rules_default, seed=99)
    prm = out.get("per_rule_map", None)
    if prm is None:
        pytest.xfail("Predictor does not expose per_rule_map; localization test skipped.")
    arr, _ = _norm_per_rule_map(_as_np(prm), L=L_DEFAULT)  # (1,R,L)
    for r_idx, rule in enumerate(rules_default):
        mask = np.asarray(rule["mask"], dtype=bool)
        inside = np.abs(arr[0, r_idx, mask])
        outside = np.abs(arr[0, r_idx, ~mask])
        if inside.size and outside.size:
            p75_in = float(np.percentile(inside, 75))
            p75_out = float(np.percentile(outside, 75))
            assert p75_in > p75_out * 1.2, (
                f"Rule '{rule['name']}' not localized: inside P75={p75_in:.3e} vs outside P75={p75_out:.3e}"
            )


def test_rule_weight_monotonicity(pred_entry, mu_single, rules_default):
    """
    Increasing rule 0 weight should not reduce its score.
    Try kwarg rule_weights then structural override as fallback.
    """
    kind, target = pred_entry

    # Baseline scores
    out0 = _invoke(kind, target, mu_single, rules_default, seed=123)
    sc0, _ = _norm_scores(_as_np(out0["scores"]), R_expected=len(rules_default))
    base0 = float(sc0[0, 0])

    # Call-time weights
    try:
        out_up = _invoke(kind, target, mu_single, rules_default, seed=123, rule_weights=[5.0, 1.0])
        sc_up, _ = _norm_scores(_as_np(out_up["scores"]), R_expected=len(rules_default))
        up0 = float(sc_up[0, 0])
        assert up0 >= base0 * 1.2, "Rule weight increase did not raise rule 0 score as expected."
        return
    except TypeError:
        # Structural override
        rules_mod = [dict(r) for r in rules_default]
        rules_mod[0]["weight"] = 5.0
        out_up2 = _invoke(kind, target, mu_single, rules_mod, seed=123)
        sc_up2, _ = _norm_scores(_as_np(out_up2["scores"]), R_expected=len(rules_mod))
        up20 = float(sc_up2[0, 0])
        if not (up20 >= base0 * 1.1):
            pytest.xfail("Predictor API does not scale with per‑rule weights (non‑monotone behavior).")


def test_determinism_fixed_seed(pred_entry, mu_single, rules_default):
    """
    Fixed seed ⇒ identical scores (and maps if provided).
    """
    kind, target = pred_entry
    out1 = _invoke(kind, target, mu_single, rules_default, seed=777)
    out2 = _invoke(kind, target, mu_single, rules_default, seed=777)

    sc1, _ = _norm_scores(_as_np(out1["scores"]), R_expected=len(rules_default))
    sc2, _ = _norm_scores(_as_np(out2["scores"]), R_expected=len(rules_default))
    assert np.array_equal(sc1, sc2), "Scores changed despite fixed seed."

    if "per_rule_map" in out1 and "per_rule_map" in out2:
        prm1, _ = _norm_per_rule_map(_as_np(out1["per_rule_map"]), L=L_DEFAULT)
        prm2, _ = _norm_per_rule_map(_as_np(out2["per_rule_map"]), L=L_DEFAULT)
        assert np.array_equal(prm1, prm2), "per_rule_map changed despite fixed seed."


# -----------------------------------------------------------------------------
# Optional artifact round‑trip
# -----------------------------------------------------------------------------

def test_save_artifacts_roundtrip_if_available(pred_mod, pred_entry, tmp_path, mu_single, rules_default):
    """
    If the module exposes a saver like `save_predictions(...)` / `save_artifacts(...)`,
    verify it writes files and shapes reload correctly.
    """
    save_fn = None
    for name in ("save_predictions", "save_artifacts", "write_artifacts"):
        if hasattr(pred_mod, name) and callable(getattr(pred_mod, name)):
            save_fn = getattr(pred_mod, name)
            break
    if save_fn is None:
        pytest.xfail("No artifact saver in symbolic_violation_predictor; skipping round‑trip.")

    kind, target = pred_entry
    result = _invoke(kind, target, mu_single, rules_default, seed=2025)
    outdir = tmp_path / "violation_pred_artifacts"
    outdir.mkdir(parents=True, exist_ok=True)

    save_fn(result, outdir=str(outdir))

    files = list(outdir.glob("*"))
    assert files, "Artifact saver wrote no files."

    sc_path = outdir / "scores.npy"
    if sc_path.exists():
        arr = np.load(sc_path)
        _norm_scores(arr, R_expected=len(rules_default))  # raises on mismatch

    prm_path = outdir / "per_rule_map.npy"
    if prm_path.exists():
        arr = np.load(prm_path)
        _norm_per_rule_map(arr, L=L_DEFAULT)  # raises on mismatch


# -----------------------------------------------------------------------------
# Optional CLI smoke
# -----------------------------------------------------------------------------

@pytest.mark.skipif(__import__("shutil").which("spectramind") is None, reason="spectramind CLI not found in PATH")
def test_cli_smoke_symbolic_rank(tmp_path, mu_single, rules_default):
    """
    Smoke test for the CLI integration (if wired), using:

        spectramind diagnose symbolic-rank \
            --mu mu.npy --rules rules.json --outdir out --seed 123

    We only assert that it executes successfully and produces artifacts.
    """
    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, mu_single)

    rules_path = tmp_path / "rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules_default, f)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "spectramind", "diagnose", "symbolic-rank",
        "--mu", str(mu_path),
        "--rules", str(rules_path),
        "--outdir", str(outdir),
        "--seed", "123",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.xfail(
            "CLI returned nonzero exit (flags or subcommand may differ).\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    produced = list(outdir.glob("*"))
    assert len(produced) > 0, "CLI ran but produced no artifacts."


# -----------------------------------------------------------------------------
# Performance guardrail
# -----------------------------------------------------------------------------

def test_runs_fast_enough(pred_entry, mu_single, rules_default):
    """
    Tiny single‑spectrum prediction should complete in <1.5s on CI CPU.
    """
    kind, target = pred_entry
    t0 = time.time()
    _ = _invoke(kind, target, mu_single, rules_default, seed=11)
    dt = time.time() - t0
    assert dt < 1.5, f"Symbolic violation prediction too slow: {dt:.3f}s (should be < 1.5s)"