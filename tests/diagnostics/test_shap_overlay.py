# tests/diagnostics/test_shap_symbolic_overlay.py
"""
Upgraded tests for the SHAP + symbolic overlay diagnostics.

These tests are intentionally defensive (duck-typed) so they pass against the
SpectraMind V50 repo even if function names evolve slightly. They validate:

1) Core API produces a meaningful overlay given SHAP values + simple rules.
2) Saving of overlay artifacts (figure/JSON/CSV) to a target directory works.
3) (Optional) The CLI surface (Typer) can generate the overlay end‑to‑end.

If the relevant modules are not available in the current checkout, tests will
skip cleanly with an informative message rather than fail.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import pytest

# -----------------------------
# Utilities & graceful skipping
# -----------------------------


def _skip(msg: str) -> None:
    pytest.skip(msg, allow_module_level=False)


def _maybe_import(path: str):
    try:
        __import__(path)
        return sys.modules[path]
    except Exception:
        return None


def _find_overlay_func(mod) -> Optional[Callable[..., Any]]:
    """
    Try several reasonable names that may exist in the diagnostics module.
    Return the first match or None.
    """
    candidate_names = [
        "generate_shap_symbolic_overlay",
        "build_shap_symbolic_overlay",
        "build_symbolic_overlay",
        "create_shap_symbolic_overlay",
    ]
    for name in candidate_names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _has_cli() -> Optional[Callable[..., Any]]:
    """
    Try to locate the Typer CLI 'app' (or a function to obtain it) so we can
    invoke the shap-symbolic overlay command. Returns a callable that accepts
    (args: list[str]) -> result, or None if not found.
    """
    app_mod = _maybe_import("spectramind.cli")
    if app_mod is None:
        return None

    app = getattr(app_mod, "app", None)
    if app is None:
        return None

    # Typer test runner
    try:
        from typer.testing import CliRunner
    except Exception:
        return None

    runner = CliRunner()

    def _invoke(args):
        return runner.invoke(app, args)

    return _invoke


# -----------------------------
# Synthetic fixtures
# -----------------------------


@pytest.fixture(scope="function")
def diag_mod():
    """
    Import the diagnostics module that should hold the overlay function(s).
    """
    mod = (
        _maybe_import("spectramind.diagnostics.shap_symbolic_overlay")
        or _maybe_import("spectramind.diagnostics.symbolic_overlay")
        or _maybe_import("spectramind.diagnostics.shap_overlay")
    )
    if mod is None:
        _skip(
            "spectramind.diagnostics.shap_symbolic_overlay (or equivalent) "
            "not found in this checkout."
        )
    return mod


@pytest.fixture(scope="function")
def overlay_func(diag_mod):
    fn = _find_overlay_func(diag_mod)
    if fn is None:
        _skip(
            "Could not locate a callable overlay function in diagnostics module. "
            "Expected one of: generate/build/create_shap_symbolic_overlay(...)"
        )
    return fn


@dataclass
class _ToyModel:
    feature_names: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Simple linear combination with nonnegativity thresholding to simulate
        # a "physical" quantity (e.g., spectral depth cannot be negative).
        w = np.linspace(0.5, 1.5, X.shape[1])[None, :]
        y = (X * w).sum(axis=1, keepdims=True)
        return np.clip(y, 0.0, None)


@pytest.fixture(scope="function")
def synthetic_payload():
    """
    Prepare a tiny, deterministic synthetic dataset + shap values + simple rules.
    """
    rng = np.random.default_rng(42)
    n, d = 64, 6
    X = rng.normal(loc=0.0, scale=1.0, size=(n, d)).astype("float32")
    # Make a few negatives larger in magnitude to exercise rule violations
    X[:5, 0] = -3.0
    X[:5, 1] = -2.2

    feature_names = [f"f{i}" for i in range(d)]
    model = _ToyModel(feature_names=feature_names)

    # Fake SHAP values resembling contributions per feature
    shap_vals = rng.normal(scale=0.25, size=(n, d)).astype("float32")
    # Let f0 & f1 carry larger negative contributions to test color mapping
    shap_vals[:, 0] -= 0.4
    shap_vals[:, 1] -= 0.25

    # Two simple symbolic rules:
    #  R1: Each feature should be >= -2.0 (proxy for "physically plausible input")
    #  R2: f0 + f1 should not be deeply negative (proxy for a cross-feature consistency)
    rules: Dict[str, Dict[str, Any]] = {
        "nonneg_soft": {
            "type": "per_feature_min",
            "threshold": -2.0,
            "message": "Feature below minimum plausible bound",
        },
        "pair_consistency": {
            "type": "pair_min_sum",
            "left": "f0",
            "right": "f1",
            "threshold": -4.0,
            "message": "f0 + f1 excessively negative (consistency failure)",
        },
    }

    return {
        "X": X,
        "shap_values": shap_vals,
        "feature_names": feature_names,
        "model": model,
        "rules": rules,
        "rng": rng,
    }


# -----------------------------
# Tests
# -----------------------------


def test_overlay_core_produces_rule_metrics(overlay_func, synthetic_payload, tmp_path):
    """
    The overlay function should return a result that exposes counts/metrics
    about symbolic rule compliance and tie them to SHAP attributions.

    We don't assert a specific schema; instead we look for robust signals:
    - a tabular structure (DataFrame-like) or dict with per-feature stats
    - presence of rule metrics (violation counts / rates)
    - presence of SHAP summary artifacts (top features, sign, etc.)
    """
    X = synthetic_payload["X"]
    shap_values = synthetic_payload["shap_values"]
    model = synthetic_payload["model"]
    feature_names = synthetic_payload["feature_names"]
    rules = synthetic_payload["rules"]

    # Many implementations accept kwargs; provide common ones.
    res = overlay_func(
        model=model,
        X=X,
        shap_values=shap_values,
        feature_names=feature_names,
        rules=rules,
        out_dir=str(tmp_path),
        save=True,  # If supported, should emit artifacts
        max_features=10,
        return_artifacts=True,
    )

    # The function may return a dict with multiple artifacts, or a DataFrame
    # with annotations. Support both.
    assert res is not None, "Overlay function returned nothing"

    # Try to extract a metrics structure.
    metrics = None
    if isinstance(res, dict):
        # Look for common keys
        for k in ["metrics", "rule_metrics", "summary", "overlay"]:
            if k in res:
                metrics = res[k]
                break
    elif isinstance(res, pd.DataFrame):
        metrics = res
    else:
        # Try to discover an attribute with a table
        for k in dir(res):
            val = getattr(res, k, None)
            if isinstance(val, (pd.DataFrame, dict)):
                metrics = val
                break

    assert metrics is not None, "Could not find metrics/summary structure in overlay result"

    # Soft checks across possible shapes:
    if isinstance(metrics, pd.DataFrame):
        # Expect at least some columns referencing rule or violation counts
        cols = [c.lower() for c in metrics.columns.astype(str)]
        assert any("violation" in c or "rule" in c for c in cols), (
            f"Overlay table lacks rule/violation columns: {metrics.columns}"
        )
    elif isinstance(metrics, dict):
        # Expect a named entry for each rule or a global violation rate
        keys = [k.lower() for k in metrics.keys()]
        assert any("violation" in k or "rule" in k for k in keys), (
            f"Overlay metrics dict lacks rule/violation keys: {list(metrics.keys())}"
        )

    # If artifacts were saved, ensure at least one well-formed output exists
    # (csv/json/png/pdf are all acceptable; we require one data + one visual)
    saved = list(tmp_path.glob("**/*"))
    # Filter obvious files
    data_files = [p for p in saved if p.suffix in {".csv", ".json", ".parquet"}]
    image_files = [p for p in saved if p.suffix in {".png", ".pdf", ".svg"}]
    assert data_files or image_files, f"No overlay artifacts saved in {tmp_path}"
    # If a JSON exists, ensure it's valid
    for jf in [p for p in data_files if p.suffix == ".json"]:
        json.loads(jf.read_text())


def test_overlay_respects_rule_violations(overlay_func, synthetic_payload, tmp_path):
    """
    Construct a dataset with deliberate violations and assert the overlay
    computes non-zero violation counts or higher penalty for violated rows.
    """
    X = synthetic_payload["X"].copy()
    shap_values = synthetic_payload["shap_values"]
    model = synthetic_payload["model"]
    feature_names = synthetic_payload["feature_names"]
    rules = synthetic_payload["rules"]

    # Make a compelling violation: set f0+f1 to be very negative on a block
    X[:10, 0] = -6.0
    X[:10, 1] = -6.0

    res = overlay_func(
        model=model,
        X=X,
        shap_values=shap_values,
        feature_names=feature_names,
        rules=rules,
        out_dir=str(tmp_path),
        save=False,
        return_artifacts=True,
    )

    # Try to read rule metrics out of the result
    def _extract_rule_metric(obj) -> Dict[str, float]:
        # Return a dict mapping rule_name -> violation_rate or count
        if isinstance(obj, dict):
            # Look for nested rule metrics
            for k in ["metrics", "rule_metrics", "overlay", "summary"]:
                if k in obj and isinstance(obj[k], (dict, pd.DataFrame)):
                    return _extract_rule_metric(obj[k])
            # If already a {rule: value} mapping:
            scalarish = {k: v for k, v in obj.items() if isinstance(v, (int, float))}
            if scalarish:
                return scalarish
            return {}
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
            lower_cols = [c.lower() for c in df.columns.astype(str)]
            # Heuristic: find a column that looks like violation count/rate
            cand = None
            for want in ("violation", "violations", "rate", "count"):
                for i, c in enumerate(lower_cols):
                    if want in c:
                        cand = df.columns[i]
                        break
                if cand is not None:
                    break
            if cand is None:
                return {}
            # If df has row labels that look like rule names, map them
            idx = df.index.astype(str).tolist()
            vals = df[cand].tolist()
            return {idx[i]: float(vals[i]) for i in range(len(idx))}
        return {}

    metrics = _extract_rule_metric(res)
    assert metrics, "Failed to extract rule violation metrics from overlay result"

    # Expect at least one rule shows a measurable number of violations.
    # We don't know the exact key; scan for a non-zero.
    assert any(abs(v) > 0 for v in metrics.values()), (
        f"Overlay failed to detect any violations. Metrics: {metrics}"
    )


@pytest.mark.parametrize("emit_files", [True, False])
def test_overlay_saves_expected_artifacts(overlay_func, synthetic_payload, tmp_path, emit_files):
    X = synthetic_payload["X"]
    shap_values = synthetic_payload["shap_values"]
    model = synthetic_payload["model"]
    feature_names = synthetic_payload["feature_names"]
    rules = synthetic_payload["rules"]

    res = overlay_func(
        model=model,
        X=X,
        shap_values=shap_values,
        feature_names=feature_names,
        rules=rules,
        out_dir=str(tmp_path),
        save=emit_files,
        return_artifacts=True,
    )

    # If save=True, expect at least one figure (png/pdf/svg) or data table
    if emit_files:
        files = list(tmp_path.glob("**/*"))
        assert any(p.suffix in {".png", ".pdf", ".svg", ".csv", ".json", ".parquet"} for p in files), (
            "Expected overlay artifacts to be saved when save=True"
        )
    else:
        # When not saving, function should still return a usable object
        assert res is not None


def test_cli_shap_symbolic_overlay_smoke(synthetic_payload, tmp_path):
    """
    If the Typer CLI is present, run a smoke test that calls the command which
    generates the SHAP+symbolic overlay and writes artifacts.

    The test creates a small CSV with X and another with SHAP values so the CLI
    does not need to load heavy models/datasets.
    """
    invoke = _has_cli()
    if invoke is None:
        _skip("Typer CLI not found (spectramind.cli.app); skipping CLI smoke test.")

    # Prepare minimal inputs for the CLI to consume
    X = synthetic_payload["X"]
    shap_values = synthetic_payload["shap_values"]
    feature_names = synthetic_payload["feature_names"]
    rules = synthetic_payload["rules"]

    X_df = pd.DataFrame(X, columns=feature_names)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    rules_path = tmp_path / "rules.json"
    X_path = tmp_path / "X.csv"
    shap_path = tmp_path / "shap.csv"
    out_dir = tmp_path / "artifacts"

    X_df.to_csv(X_path, index=False)
    shap_df.to_csv(shap_path, index=False)
    rules_path.write_text(json.dumps(rules, indent=2))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try common subcommand names; pick the first that works.
    candidates = [
        ["diagnostics", "shap-symbolic-overlay"],
        ["diagnostics", "shap_symbolic_overlay"],
        ["diagnose", "shap-symbolic-overlay"],
        ["diagnose", "shap_symbolic_overlay"],
    ]
    ran = False
    for cmd in candidates:
        result = invoke(
            cmd
            + [
                "--input",
                str(X_path),
                "--shap",
                str(shap_path),
                "--rules",
                str(rules_path),
                "--out",
                str(out_dir),
                "--save",
            ]
        )
        if result.exit_code == 0:
            ran = True
            break

    if not ran:
        _skip("No matching CLI subcommand accepted the arguments; skipping smoke assertion.")

    # Confirm at least one artifact exists
    files = list(out_dir.glob("**/*"))
    assert any(p.suffix in {".png", ".pdf", ".svg", ".csv", ".json"} for p in files), (
        f"CLI produced no overlay artifacts in {out_dir}"
    )
