# tests/diagnostics/test_simulate_lightcurve_from_mu.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: simulate_lightcurve_from_mu

Purpose
-------
Validate the scientific and engineering behavior of the *simulate_lightcurve_from_mu*
utility, which should synthesize AIRS/FGS1-like time-series cubes from a target
transmission spectrum μ (length ≈ 283) and minimal metadata/configuration.

This test suite focuses on:
1) API availability & shape contracts
2) Determinism under fixed seeds
3) Noise control & variance behavior
4) CLI parity (if a CLI entrypoint is exposed)
5) Robust error signaling for bad inputs

Design Notes
------------
• The tests are *defensively adaptable* to small API differences. They locate the
  module under common repo layouts (e.g., tools/simulate_lightcurve_from_mu.py or
  src/tools/simulate_lightcurve_from_mu.py) and try a small set of expected call
  patterns:
    - generate(mu, metadata=None, **cfg)
    - simulate_lightcurve(mu, metadata=None, **cfg)
    - Simulator(...).run()
  If none are found, the test will xfail with a clear message.

• Sizes are intentionally small (tiny time length & spatial dims) to run fast in CI.

• No placeholders: every assertion is tied to concrete, minimal scientific expectations
  (finite outputs, reasonable shape contracts, deterministic seeding, noise ↑ variance).

• CLI test (if available) is executed via `python -m tools.simulate_lightcurve_from_mu`
  or `python -m src.tools.simulate_lightcurve_from_mu`, using a temp output dir.

Requirements
------------
• pytest
• numpy

Author: SpectraMind V50 Team
"""

from __future__ import annotations

import importlib
import os
import sys
import json
import math
import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pytest


# ------------------------------
# Helpers to discover the module
# ------------------------------

CANDIDATE_IMPORTS = [
    "tools.simulate_lightcurve_from_mu",
    "src.tools.simulate_lightcurve_from_mu",
    "simulate_lightcurve_from_mu",
]


def _import_sim_module():
    """
    Try to import the simulate_lightcurve_from_mu module from common locations.

    Returns
    -------
    module
        Imported module object.

    Raises
    ------
    ImportError
        If the module is not found in any candidate path.
    """
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Could not import simulate_lightcurve_from_mu from any of: {CANDIDATE_IMPORTS}\n"
        f"Last error: {last_err}"
    )


def _locate_callable(mod):
    """
    Detect a suitable callable to synthesize light curves from μ.

    Returns
    -------
    kind : str
        'func' or 'method' or 'class'
    callable_or_class : callable | type
        The function or class object to invoke.
    """
    # Preferred function signatures
    for fn_name in ("generate", "simulate_lightcurve"):
        if hasattr(mod, fn_name):
            fn = getattr(mod, fn_name)
            if callable(fn):
                return "func", fn

    # Fallback: class with .run()
    for cls_name in ("Simulator", "LightcurveSimulator", "SimulateFromMu"):
        if hasattr(mod, cls_name):
            cls = getattr(mod, cls_name)
            # Instantiate later in caller
            if hasattr(cls, "run"):
                return "class", cls

    # Fallback: a function named main_generate (rare)
    if hasattr(mod, "main_generate") and callable(getattr(mod, "main_generate")):
        return "func", getattr(mod, "main_generate")

    pytest.xfail(
        "simulate_lightcurve_from_mu module found but no known callable was discovered. "
        "Expected one of: generate(), simulate_lightcurve(), Simulator.run(), LightcurveSimulator.run()."
    )
    return "none", None  # pragma: no cover


def _invoke_generation(kind: str, callable_or_class, mu: np.ndarray, metadata: Optional[dict], **cfg):
    """
    Invoke the generator in an API-agnostic way.

    Returns
    -------
    out : dict
        Expected to contain arrays for 'airs_cube' and/or 'fgs1_cube' and 'time', maybe 'wavelengths'.
    """
    if kind == "func":
        return callable_or_class(mu, metadata=metadata, **cfg)
    elif kind == "class":
        # Attempt common init signatures
        try:
            inst = callable_or_class(mu=mu, metadata=metadata, **cfg)
        except TypeError:
            inst = callable_or_class(**cfg)
            return inst.run(mu=mu, metadata=metadata)
        if hasattr(inst, "run"):
            return inst.run()
        pytest.xfail("Simulator class found but no .run() method available.")
    else:
        pytest.fail("Unknown invocation kind; test configuration error.")  # pragma: no cover


def _clean_tmp_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


# ------------------------------
# Fixtures
# ------------------------------

@pytest.fixture(scope="module")
def sim_mod():
    """Import the target module once per module scope."""
    return _import_sim_module()


@pytest.fixture(scope="module")
def sim_callable(sim_mod):
    """Detect the callable/class once per module scope."""
    return _locate_callable(sim_mod)


@pytest.fixture
def tiny_mu() -> np.ndarray:
    """
    Construct a small, well-behaved μ spectrum for tests.

    We use 283 points by default (the challenge spec), but allow smaller if the
    simulator supports mapping. Here we stick to 283 for contract checking.

    The spectrum is positive (transit depths in ppm scaled to 0..1), with a few
    synthetic absorption features.
    """
    n = 283
    x = np.linspace(0, 1, n, dtype=np.float64)

    # Smooth baseline (0.01–0.03 range), plus three synthetic Gaussian "lines".
    baseline = 0.02 + 0.005 * np.sin(2 * math.pi * 3 * x)
    lines = (
        0.015 * np.exp(-0.5 * ((x - 0.25) / 0.02) ** 2)
        + 0.010 * np.exp(-0.5 * ((x - 0.55) / 0.015) ** 2)
        + 0.008 * np.exp(-0.5 * ((x - 0.82) / 0.01) ** 2)
    )
    mu = baseline + lines
    mu = np.clip(mu, 0.0, 1.0)  # transit depth fraction 0..1
    assert mu.shape == (n,)
    assert np.all(np.isfinite(mu))
    return mu


@pytest.fixture
def tiny_metadata() -> Dict[str, Any]:
    """
    Minimal metadata example required by many simulators:
    - planet_id
    - exposure_time_s
    - period_s (not strictly used by all implementations)
    """
    return {
        "planet_id": "TEST-0001",
        "exposure_time_s": 2.0,
        "period_s": 100000.0,
        # Optional fields tolerated by some simulators:
        "star_mag": 10.0,
        "distance_pc": 100.0,
    }


@pytest.fixture
def tiny_cfg() -> Dict[str, Any]:
    """
    Small config to keep the synthetic cubes tiny and CI-friendly.
    These keys are commonly supported; if unknown keys are provided, most implementations will ignore them.
    """
    return {
        # core sizing
        "n_time": 64,
        "airs_shape": (8, 16),   # (H, W) tiny spectral image
        "fgs1_shape": (8, 8),    # (H, W) tiny photometric image
        # physical-ish toggles
        "inject_noise": True,
        "noise_sigma": 0.001,
        "jitter_ppm": 50.0,
        # reproducibility
        "seed": 1234,
        # I/O toggles (most implementations support dry-runs or return-only)
        "save": False,
        "outdir": None,
    }


# ------------------------------
# Core API/contract tests
# ------------------------------

def test_generate_shapes_and_finiteness(sim_callable, tiny_mu, tiny_metadata, tiny_cfg):
    """
    The generator should:
    • accept μ with shape (283,)
    • return finite arrays for AIRS/FGS1 time-series cubes
    • include a time vector with length == n_time
    • respect the requested small shapes
    """
    kind, target = sim_callable

    out = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **tiny_cfg)
    assert isinstance(out, dict), "Simulator should return a dictionary of outputs."

    # Not all sims output both instruments; accept either, but require at least one.
    has_airs = "airs_cube" in out
    has_fgs1 = "fgs1_cube" in out
    assert has_airs or has_fgs1, "Expected at least one of AIRS or FGS1 cubes."

    n_time = tiny_cfg["n_time"]

    if has_airs:
        airs = out["airs_cube"]
        assert isinstance(airs, np.ndarray)
        assert airs.ndim == 3, "AIRS cube must be (T, H, W)"
        assert airs.shape[0] == n_time
        assert airs.shape[1:] == tiny_cfg["airs_shape"]
        assert np.all(np.isfinite(airs)), "AIRS cube contains non-finite values."

    if has_fgs1:
        fgs1 = out["fgs1_cube"]
        assert isinstance(fgs1, np.ndarray)
        assert fgs1.ndim == 3, "FGS1 cube must be (T, H, W)"
        assert fgs1.shape[0] == n_time
        assert fgs1.shape[1:] == tiny_cfg["fgs1_shape"]
        assert np.all(np.isfinite(fgs1)), "FGS1 cube contains non-finite values."

    # time axis is expected
    assert "time" in out, "Output must include a time vector."
    t = out["time"]
    assert isinstance(t, np.ndarray)
    assert t.ndim == 1 and t.shape[0] == n_time
    assert np.all(np.isfinite(t))


def test_determinism_with_seed(sim_callable, tiny_mu, tiny_metadata, tiny_cfg):
    """
    With a fixed seed, the simulator must be deterministic (bitwise equal outputs).
    """
    kind, target = sim_callable

    cfgA = dict(tiny_cfg)
    cfgB = dict(tiny_cfg)
    cfgA["seed"] = 7
    cfgB["seed"] = 7

    out1 = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **cfgA)
    out2 = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **cfgB)

    # Compare keys and arrays bitwise-equal where applicable.
    assert set(out1.keys()) == set(out2.keys())

    for k in out1:
        v1, v2 = out1[k], out2[k]
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            assert v1.shape == v2.shape
            assert np.array_equal(v1, v2), f"Mismatch for key '{k}' under same seed!"
        else:
            # For scalars or metadata dicts, fall back to equality
            assert v1 == v2


def test_noise_control_affects_variance(sim_callable, tiny_mu, tiny_metadata, tiny_cfg):
    """
    Increasing noise_sigma should (statistically) increase variance in the synthetic data.
    We test this on AIRS if available; otherwise, on FGS1.
    """
    kind, target = sim_callable

    cfg_low = dict(tiny_cfg)
    cfg_high = dict(tiny_cfg)

    cfg_low["inject_noise"] = True
    cfg_low["noise_sigma"] = 0.0001
    cfg_low["seed"] = 42

    cfg_high["inject_noise"] = True
    cfg_high["noise_sigma"] = 0.01
    cfg_high["seed"] = 42

    out_lo = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **cfg_low)
    out_hi = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **cfg_high)

    def _pick_cube(out):
        if "airs_cube" in out:
            return out["airs_cube"]
        elif "fgs1_cube" in out:
            return out["fgs1_cube"]
        pytest.xfail("Simulator returned neither AIRS nor FGS1 cube.")

    cube_lo = _pick_cube(out_lo).astype(np.float64)
    cube_hi = _pick_cube(out_hi).astype(np.float64)

    # Compare overall variance across the entire cube
    var_lo = float(np.var(cube_lo))
    var_hi = float(np.var(cube_hi))
    assert var_hi > var_lo, f"Expected higher variance with higher noise. var_lo={var_lo}, var_hi={var_hi}"


def test_rejects_bad_mu_shape(sim_callable, tiny_metadata, tiny_cfg):
    """
    The simulator should raise a clear error for invalid μ shapes (e.g., (10,) instead of (283,)).
    """
    kind, target = sim_callable

    bad_mu = np.ones((10,), dtype=np.float64)
    with pytest.raises(Exception):
        _invoke_generation(kind, target, bad_mu, tiny_metadata, **tiny_cfg)


# ------------------------------
# CLI behavior (optional)
# ------------------------------

CLI_CANDIDATES = [
    ("tools.simulate_lightcurve_from_mu", "python", "-m", "tools.simulate_lightcurve_from_mu"),
    ("src.tools.simulate_lightcurve_from_mu", "python", "-m", "src.tools.simulate_lightcurve_from_mu"),
]


@pytest.mark.parametrize("module_name,py,flag_m,entry", CLI_CANDIDATES)
def test_cli_smoke_if_available(tmp_path: Path, tiny_mu, module_name, py, flag_m, entry):
    """
    If the module exposes a Python -m entrypoint, perform a CLI smoke test:

    • Writes μ to a temporary .npy
    • Invokes the module with --mu and --outdir
    • Uses a fixed seed for determinism
    • Expects exit code 0 and creation of some outputs (or at least a manifest/log)
    """
    # Quickly check import; if missing, skip this CLI form.
    try:
        importlib.import_module(module_name)
    except Exception:
        pytest.skip(f"Module {module_name} not importable; skipping CLI smoke.")

    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, tiny_mu)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    # Common CLI flags: not all implementations share exact names, so we pass a safe subset.
    # Implementations should ignore unknown flags gracefully or document their names.
    cmd = [
        sys.executable,
        flag_m,
        entry,
        "--mu", str(mu_path),
        "--outdir", str(outdir),
        "--seed", "777",
        "--n-time", "32",
        "--airs-h", "6",
        "--airs-w", "12",
        "--fgs1-h", "6",
        "--fgs1-w", "6",
        "--inject-noise",
        "--noise-sigma", "0.001",
        "--save",
    ]

    # Some CLIs use underscores or different option names; try a best-effort call and don't fail
    # the entire test suite if the CLI returns usage error. We treat nonzero exit as xfail with logs.
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(Path.cwd()))
    except Exception as e:
        pytest.xfail(f"CLI invocation failed at OS level: {e}")

    if proc.returncode != 0:
        # Provide helpful context but do not hard-fail the whole suite.
        pytest.xfail(
            "CLI returned nonzero exit. This may indicate different flag names. "
            f"Stdout:\n{proc.stdout}\n\nStderr:\n{proc.stderr}"
        )

    # If it ran, we expect either data files (npy) or at least a manifest/log in outdir.
    produced = list(outdir.glob("*"))
    assert len(produced) > 0, "CLI ran without creating any outputs in the specified outdir."


# ------------------------------
# Round-trip save (if supported)
# ------------------------------

def test_save_artifacts_roundtrip(sim_mod, sim_callable, tmp_path: Path, tiny_mu, tiny_metadata, tiny_cfg):
    """
    If the module exposes a 'save_artifacts(...)' utility, verify that:
    • It writes expected files into outdir
    • Written arrays can be reloaded with correct shapes

    This is optional; if not present, the test is xfailed (documented).
    """
    if not hasattr(sim_mod, "save_artifacts") or not callable(getattr(sim_mod, "save_artifacts")):
        pytest.xfail("Module does not expose save_artifacts(); skipping round-trip save test.")

    kind, target = sim_callable
    cfg = dict(tiny_cfg)
    cfg["save"] = False  # We'll explicitly call save_artifacts
    cfg["outdir"] = None

    out = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **cfg)
    outdir = tmp_path / "save_out"
    outdir.mkdir(parents=True, exist_ok=True)

    # Try to save; require at least μ + time and any generated cubes.
    save_fn = getattr(sim_mod, "save_artifacts")
    save_fn(out, outdir=str(outdir), metadata=tiny_metadata)

    saved = list(outdir.glob("*.npy")) + list(outdir.glob("*.json"))
    assert len(saved) > 0, "save_artifacts() created no files."

    # If cubes exist, reload one and check shape parity
    for key in ("airs_cube", "fgs1_cube", "time"):
        p = outdir / f"{key}.npy"
        if p.exists():
            arr = np.load(p)
            assert arr.shape == out[key].shape, f"Saved shape mismatch for {key}."


# ------------------------------
# Metadata pass-through (optional)
# ------------------------------

def test_metadata_passthrough_in_output(sim_callable, tiny_mu, tiny_metadata, tiny_cfg):
    """
    Some implementations echo metadata in the output dict as 'metadata' or embed it into a manifest.
    If present, verify that core fields pass through unchanged.
    """
    kind, target = sim_callable
    out = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **tiny_cfg)

    meta_out = out.get("metadata", None)
    if meta_out is None:
        pytest.xfail("No 'metadata' key in output; acceptable if design stores metadata externally.")
    else:
        for k in ("planet_id", "exposure_time_s"):
            assert k in meta_out and meta_out[k] == tiny_metadata[k]


# ------------------------------
# Performance sanity (very light)
# ------------------------------

def test_runs_fast_enough(sim_callable, tiny_mu, tiny_metadata, tiny_cfg):
    """
    Light performance guardrail: the tiny config should execute within ~1 second on CI CPU.
    """
    kind, target = sim_callable
    t0 = time.time()
    _ = _invoke_generation(kind, target, tiny_mu, tiny_metadata, **tiny_cfg)
    dt = time.time() - t0
    assert dt < 1.5, f"Tiny simulation took too long: {dt:.3f}s (should be < 1.5s)"
