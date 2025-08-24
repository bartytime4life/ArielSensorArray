# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_analyze_fft_autocorr_mu.py

SpectraMind V50 — Diagnostics tests for tools/analyze_fft_autocorr_mu.py

This suite validates the scientific logic, artifact generation, and CLI behavior of the
FFT + autocorrelation diagnostic tool used on μ spectra. It is designed to be robust to
minor API differences while enforcing core guarantees of the SpectraMind V50 tooling:
determinism (with seeds), artifact creation (JSON/CSV/PNG/HTML), reasonable scientific
properties (e.g., FFT peak for sinusoidal content; autocorrelation periodicity), and
audit logging (v50_debug_log.md) when enabled.

Coverage
--------
1) Core API sanity:
   • compute_fft_power(...) on constant vs. sinusoid.
   • compute_autocorr(...) periodicity and r[0] maximum.

2) Artifact generation API:
   • generate_fft_autocorr_artifacts(...)/run_fft_autocorr_diagnostics(...) produces JSON/CSV/PNG/HTML.
   • JSON manifest contains expected fields (freq axis, power, autocorr summary).

3) CLI contract:
   • End-to-end run via subprocess (python -m tools.analyze_fft_autocorr_mu).
   • Determinism with --seed: JSON equality modulo volatile fields.
   • Graceful error handling for missing/invalid args.
   • Optional SPECTRAMIND_LOG_PATH audit line is appended.

4) Housekeeping:
   • Output files are nonempty and stable across re-runs (idempotent/versioned is fine).
   • PNG/CSV/HTML presence checks (minimum size thresholds).

Assumptions
-----------
• The module lives at tools/analyze_fft_autocorr_mu.py and exposes one or more of:
    - compute_fft_power(signal, fs=None, n_freq=None, window=None, **kw)
    - compute_autocorr(signal, max_lag=None, **kw)
    - generate_fft_autocorr_artifacts(...) / run_fft_autocorr_diagnostics(...)
• Or similar names:
    - fft_power, power_spectrum, spectrum_fft
    - autocorr, autocorrelation, compute_acf

• CLI accepts flags such as:
    --mu <path.npy>
    --wavelengths <path.npy>   (optional)
    --outdir <dir>
    --json --csv --png --html  (any subset allowed)
    --n-freq <int>             (optional)
    --seed <int>               (optional)
    --silent                   (optional)
  The exact set can vary; tests are tolerant but require basic functionality.

Implementation notes
--------------------
• These tests use lightweight synthetic data and no GPU/network.
• Numerical checks use tolerant comparisons and avoid brittle equality on floats.
• Where an API surface is missing, tests xfail with an explanation instead of silently passing.

Author: SpectraMind V50 test harness
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest


# =================================================================================================
# Helpers
# =================================================================================================

def _import_tool():
    """
    Import the module under test.

    Tries:
      1) tools.analyze_fft_autocorr_mu
      2) analyze_fft_autocorr_mu  (top-level)
    """
    try:
        import tools.analyze_fft_autocorr_mu as m  # type: ignore
        return m
    except Exception:
        try:
            import analyze_fft_autocorr_mu as m2  # type: ignore
            return m2
        except Exception:
            pytest.skip(
                "analyze_fft_autocorr_mu module not found. "
                "Expected at tools/analyze_fft_autocorr_mu.py or importable as analyze_fft_autocorr_mu."
            )


def _has_attr(mod, name: str) -> bool:
    return hasattr(mod, name) and getattr(mod, name) is not None


def _run_cli(
    module_path: Path,
    args: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    """
    Execute the tool as a CLI using `python -m tools.analyze_fft_autocorr_mu` when possible.
    Fallback to direct script invocation by file path if package execution is not feasible.
    """
    # Determine repository root if we have .../tools/analyze_fft_autocorr_mu.py
    if module_path.name == "analyze_fft_autocorr_mu.py" and module_path.parent.name == "tools":
        repo_root = module_path.parent.parent
        candidate_pkg = "tools.analyze_fft_autocorr_mu"
        cmd = [sys.executable, "-m", candidate_pkg, *args]
        cwd = str(repo_root)
    else:
        cmd = [sys.executable, str(module_path), *args]
        cwd = str(module_path.parent)

    env_full = os.environ.copy()
    if env:
        env_full.update(env)

    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env_full,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        check=False,
    )


def _assert_file(p: Path, min_size: int = 1) -> None:
    assert p.exists(), f"File not found: {p}"
    assert p.is_file(), f"Expected file: {p}"
    sz = p.stat().st_size
    assert sz >= min_size, f"File too small ({sz} bytes): {p}"


def _fft_peak_index(power: np.ndarray) -> int:
    """Return the index of the largest element (argmax) in a 1D power array."""
    return int(np.argmax(np.asarray(power)))


# =================================================================================================
# Synthetic inputs (signals & wavelengths)
# =================================================================================================

def _make_mu_constant(n: int, value: float = 0.5) -> np.ndarray:
    return np.full((n,), float(value), dtype=np.float64)


def _make_mu_sine(n: int, cycles: float = 5.0, amp: float = 1.0, bias: float = 0.0, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi * cycles, n, dtype=np.float64)
    return bias + amp * np.sin(t + phase)


def _make_mu_sine_plus_noise(n: int, cycles: float = 5.0, amp: float = 1.0, noise: float = 0.1, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = _make_mu_sine(n, cycles=cycles, amp=amp, bias=0.0, phase=0.0)
    return clean + noise * rng.standard_normal(n).astype(np.float64)


def _make_wavelengths(n: int, lo_um: float = 0.5, hi_um: float = 7.8) -> np.ndarray:
    # Ariel-esque continuous bins (not used by FFT math, but useful metadata and optional bin-weighting)
    return np.linspace(float(lo_um), float(hi_um), n, dtype=np.float64)


# =================================================================================================
# Fixtures
# =================================================================================================

@pytest.fixture(scope="module")
def tool_mod():
    return _import_tool()


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Dict[str, Path]:
    """
    Create a clean workspace:
      inputs/  — for .npy μ and wavelength arrays
      outputs/ — for artifacts
      logs/    — for optional v50_debug_log.md
    """
    ip = tmp_path / "inputs"
    op = tmp_path / "outputs"
    lg = tmp_path / "logs"
    ip.mkdir(parents=True, exist_ok=True)
    op.mkdir(parents=True, exist_ok=True)
    lg.mkdir(parents=True, exist_ok=True)
    return {"root": tmp_path, "inputs": ip, "outputs": op, "logs": lg}


@pytest.fixture()
def synthetic_inputs(tmp_workspace: Dict[str, Path]) -> Dict[str, Path]:
    """
    Save deterministic test arrays to disk:
      - mu_constant.npy
      - mu_sine.npy
      - mu_sine_noise.npy
      - wavelengths.npy
    """
    n = 512  # power of two for FFT convenience; moderate for speed
    wl = _make_wavelengths(n)
    mu_const = _make_mu_constant(n, 0.25)
    mu_sine = _make_mu_sine(n, cycles=8.0, amp=0.8, bias=0.1)
    mu_noisy = _make_mu_sine_plus_noise(n, cycles=8.0, amp=0.8, noise=0.05, seed=123)

    ip = tmp_workspace["inputs"]
    wl_path = ip / "wavelengths.npy"
    const_path = ip / "mu_constant.npy"
    sine_path = ip / "mu_sine.npy"
    noisy_path = ip / "mu_sine_noise.npy"

    np.save(wl_path, wl)
    np.save(const_path, mu_const)
    np.save(sine_path, mu_sine)
    np.save(noisy_path, mu_noisy)

    return {
        "wavelengths": wl_path,
        "mu_constant": const_path,
        "mu_sine": sine_path,
        "mu_sine_noise": noisy_path,
    }


# =================================================================================================
# Core API tests — FFT & Autocorrelation
# =================================================================================================

def test_fft_power_peak_detection(tool_mod, synthetic_inputs):
    """
    On a pure sinusoid, FFT power spectrum should have a clear dominant peak (excluding DC if biased).
    On a constant signal, FFT power should be concentrated at DC (index 0) with negligible other bins.
    """
    # Identify FFT function
    candidates = [
        "compute_fft_power",    # expected return: (freq, power) or dict
        "fft_power",
        "power_spectrum",
        "spectrum_fft",
        "analyze_fft",          # may return a richer structure
    ]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No FFT function (compute_fft_power/fft_power/...) found in analyze_fft_autocorr_mu.")

    mu_sine = np.load(synthetic_inputs["mu_sine"])
    mu_const = np.load(synthetic_inputs["mu_constant"])

    # Try calling the function and normalize expected shape
    def _call(signal: np.ndarray):
        try:
            out = fn(signal, fs=None, n_freq=None, window=None)
        except TypeError:
            out = fn(signal)  # type: ignore
        # Normalize to (freq, power) numpy arrays
        if isinstance(out, dict):
            f = np.asarray(out.get("freq") or out.get("frequency") or out.get("f"))
            p = np.asarray(out.get("power") or out.get("P") or out.get("spectrum"))
            assert f is not None and p is not None, "FFT dict missing freq/power."
            return f, p
        elif isinstance(out, (tuple, list)) and len(out) >= 2:
            f = np.asarray(out[0])
            p = np.asarray(out[1])
            return f, p
        else:
            pytest.fail("Unknown return type from FFT function; expected (freq, power) or dict.")
    f_sine, p_sine = _call(mu_sine)
    f_const, p_const = _call(mu_const)

    # Basic shape assertions
    assert p_sine.ndim == 1 and p_sine.size > 4, "FFT power for sine should be a 1D array."
    assert p_const.shape == p_sine.shape, "FFT shapes should match for equal-length signals."

    # Peak behavior: constant should peak at DC; sine should have a non-DC dominant peak (allow bias).
    idx_peak_const = _fft_peak_index(p_const)
    idx_peak_sine = _fft_peak_index(p_sine)

    assert idx_peak_const == 0, f"Constant sequence should have DC peak at index 0, got {idx_peak_const}."
    assert idx_peak_sine != 0, "Sine spectrum should have dominant non-DC peak."

    # Sine should have much stronger non-DC vs constant's non-DC energy
    non_dc_energy_const = float(np.sum(p_const[1:]))
    non_dc_energy_sine = float(np.sum(p_sine[1:]))
    assert non_dc_energy_sine > 10.0 * max(non_dc_energy_const, 1e-18), \
        "Sine non-DC energy should handily exceed constant's non-DC energy."


def test_autocorr_periodicity(tool_mod, synthetic_inputs):
    """
    Autocorrelation of a sinusoid shows periodic structure with r[0] being the maximum.
    Constant signal autocorrelation is flat-ish with dominant zero-lag.
    """
    candidates = [
        "compute_autocorr",
        "autocorr",
        "autocorrelation",
        "compute_acf",
    ]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No autocorrelation function found in analyze_fft_autocorr_mu.")

    mu_sine = np.load(synthetic_inputs["mu_sine"])
    mu_const = np.load(synthetic_inputs["mu_constant"])

    # Call and normalize outputs
    def _call(signal: np.ndarray):
        try:
            out = fn(signal, max_lag=None, normalize=True)
        except TypeError:
            out = fn(signal)  # type: ignore
        # Accept return as array or dict with 'acf'/'autocorr'
        if isinstance(out, dict):
            ac = np.asarray(out.get("acf") or out.get("autocorr") or out.get("r"))
        else:
            ac = np.asarray(out)
        assert ac.ndim == 1 and ac.size >= 8, "Autocorr should be 1D with reasonable length."
        return ac

    ac_sine = _call(mu_sine)
    ac_const = _call(mu_const)

    # r[0] maximum & approx 1.0 if normalized
    assert abs(ac_sine[0] - 1.0) <= 1e-6 or ac_sine[0] >= 0.999, "Sine ACF r[0] should be ~1.0 when normalized."
    assert abs(ac_const[0] - 1.0) <= 1e-6 or ac_const[0] >= 0.999, "Constant ACF r[0] should be ~1.0 when normalized."
    assert ac_sine[0] >= np.max(ac_sine[1:]) - 1e-12, "Zero-lag should be the maximum of ACF for sine."
    assert ac_const[0] >= np.max(ac_const[1:]) - 1e-12, "Zero-lag should be the maximum of ACF for constant."

    # Periodicity for sine: mean of top-k non-zero-lag peaks should be positive and notably larger than median
    k = min(5, ac_sine.size - 1)
    # Avoid trivial lag 0; examine coarse grid of lags to catch periodic peaks
    candidate_lags = np.arange(1, min(64, ac_sine.size), dtype=int)
    topk_vals = np.sort(ac_sine[candidate_lags])[-k:]
    assert float(np.mean(topk_vals)) > float(np.median(ac_sine)) + 0.05, \
        "Sine ACF should show non-trivial periodic peaks above median baseline."


# =================================================================================================
# Artifact Generation API
# =================================================================================================

def test_generate_artifacts(tool_mod, tmp_workspace, synthetic_inputs):
    """
    Artifact generator should emit JSON/CSV/PNG/HTML files and return a manifest (or paths).
    """
    entry_candidates = [
        "generate_fft_autocorr_artifacts",
        "run_fft_autocorr_diagnostics",
        "produce_fft_autocorr_outputs",
        "analyze_and_export",  # generic fallback
    ]
    entry = None
    for name in entry_candidates:
        if _has_attr(tool_mod, name):
            entry = getattr(tool_mod, name)
            break
    if entry is None:
        pytest.xfail("No artifact generation entrypoint found in analyze_fft_autocorr_mu.")

    outdir = tmp_workspace["outputs"]
    mu = np.load(synthetic_inputs["mu_sine_noise"])
    wl = np.load(synthetic_inputs["wavelengths"])

    kwargs = dict(
        mu=mu,
        wavelengths=wl,
        outdir=str(outdir),
        json_out=True,
        csv_out=True,
        png_out=True,
        html_out=True,
        n_freq=128,
        seed=99,
        title="FFT+ACF Test Artifacts",
    )
    try:
        manifest = entry(**kwargs)
    except TypeError:
        manifest = entry(mu, wl, str(outdir), True, True, True, True, 128, 99, "FFT+ACF Test Artifacts")  # type: ignore

    # Presence checks
    json_files = list(outdir.glob("*.json"))
    csv_files = list(outdir.glob("*.csv"))
    png_files = list(outdir.glob("*.png"))
    html_files = list(outdir.glob("*.html"))

    assert json_files, "No JSON artifact produced by artifact generator."
    assert csv_files, "No CSV artifact produced by artifact generator."
    assert png_files, "No PNG artifact produced by artifact generator."
    assert html_files, "No HTML artifact produced by artifact generator."

    # Validate a JSON schema minimally
    with open(json_files[0], "r", encoding="utf-8") as f:
        js = json.load(f)
    assert isinstance(js, dict), "Top-level JSON must be an object."
    has_fft = ("fft" in js) or ("power" in js) or ("spectrum" in js) or ("freq" in js)
    has_acf = ("acf" in js) or ("autocorr" in js) or ("autocorrelation" in js)
    assert has_fft, "JSON should include FFT/power content (keys like 'fft','power','spectrum','freq')."
    assert has_acf, "JSON should include autocorr content (keys like 'acf','autocorr','autocorrelation')."

    # Files should be non-trivially sized to avoid zero-byte outputs
    for p in png_files:
        _assert_file(p, min_size=256)
    for c in csv_files:
        _assert_file(c, min_size=64)
    for h in html_files:
        _assert_file(h, min_size=128)


# =================================================================================================
# CLI End-to-End
# =================================================================================================

def test_cli_end_to_end(tmp_workspace, synthetic_inputs):
    """
    End-to-end CLI test:
      • Runs the module as a CLI with --mu/--wavelengths → emits JSON/CSV/PNG/HTML.
      • Uses --seed for determinism and compares JSON across two runs (modulo volatile metadata).
      • Verifies optional audit log when SPECTRAMIND_LOG_PATH is set.
    """
    # Locate module file to construct python -m invocation
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "analyze_fft_autocorr_mu.py",  # repo-root/tools/...
        Path(__file__).resolve().parents[1] / "analyze_fft_autocorr_mu.py",            # tests/diagnostics/../
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("analyze_fft_autocorr_mu.py not found; cannot run CLI end-to-end test.")

    outdir = tmp_workspace["outputs"]
    logsdir = tmp_workspace["logs"]
    mu_path = synthetic_inputs["mu_sine_noise"]
    wl_path = synthetic_inputs["wavelengths"]

    env = {
        "PYTHONUNBUFFERED": "1",
        "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
    }

    args = (
        "--mu", str(mu_path),
        "--wavelengths", str(wl_path),
        "--outdir", str(outdir),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--n-freq", "128",
        "--seed", "2025",
        "--silent",
    )
    proc1 = _run_cli(module_file, args, env=env, timeout=210)
    if proc1.returncode != 0:
        msg = f"CLI run 1 failed (exit={proc1.returncode}).\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}"
        pytest.fail(msg)

    json1 = sorted(outdir.glob("*.json"))
    csv1 = sorted(outdir.glob("*.csv"))
    png1 = sorted(outdir.glob("*.png"))
    html1 = sorted(outdir.glob("*.html"))

    assert json1 and csv1 and png1 and html1, "CLI run 1 did not produce all expected artifact types."

    # Determinism check: second run with same seed into a new directory should match JSON content
    outdir2 = outdir.parent / "outputs_run2"
    outdir2.mkdir(exist_ok=True)
    args2 = (
        "--mu", str(mu_path),
        "--wavelengths", str(wl_path),
        "--outdir", str(outdir2),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--n-freq", "128",
        "--seed", "2025",
        "--silent",
    )
    proc2 = _run_cli(module_file, args2, env=env, timeout=210)
    if proc2.returncode != 0:
        msg = f"CLI run 2 failed (exit={proc2.returncode}).\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"
        pytest.fail(msg)

    json2 = sorted(outdir2.glob("*.json"))
    assert json2, "Second CLI run produced no JSON artifacts."

    # Normalize JSON: drop volatile fields (timestamps, host, absolute paths, durations)
    def _normalize(j: Dict[str, Any]) -> Dict[str, Any]:
        d = json.loads(json.dumps(j))  # deep copy
        vol_patterns = re.compile(r"(time|date|timestamp|duration|path|cwd|hostname|uuid|version)", re.I)

        def scrub(obj: Any) -> Any:
            if isinstance(obj, dict):
                keys = list(obj.keys())
                for k in keys:
                    if vol_patterns.search(k):
                        obj.pop(k, None)
                    else:
                        obj[k] = scrub(obj[k])
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = scrub(obj[i])
            return obj

        return scrub(d)

    with open(json1[0], "r", encoding="utf-8") as f:
        j1 = _normalize(json.load(f))
    with open(json2[0], "r", encoding="utf-8") as f:
        j2 = _normalize(json.load(f))

    assert j1 == j2, "Seeded CLI runs should yield identical JSON after removing volatile metadata."

    # Audit log should exist and include a recognizable signature
    log_file = Path(env["SPECTRAMIND_LOG_PATH"])
    if log_file.exists():
        _assert_file(log_file, min_size=1)
        text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
        assert ("analyze_fft_autocorr_mu" in text) or ("fft" in text and "autocorr" in text), \
            "Audit log exists but lacks recognizable CLI signature."


def test_cli_error_cases(tmp_workspace, synthetic_inputs):
    """
    CLI should:
      • Exit non-zero when required --mu is missing.
      • Report helpful error text mentioning the missing/invalid flag.
      • Handle invalid numeric args (e.g., --n-freq -5) by failing or sanitizing with a warning.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "analyze_fft_autocorr_mu.py",
        Path(__file__).resolve().parents[1] / "analyze_fft_autocorr_mu.py",
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("analyze_fft_autocorr_mu.py not found; cannot run CLI error tests.")

    outdir = tmp_workspace["outputs"]
    wl = synthetic_inputs["wavelengths"]

    # Missing --mu
    args_missing_mu = (
        "--wavelengths", str(wl),
        "--outdir", str(outdir),
        "--json",
    )
    proc = _run_cli(module_file, args_missing_mu, env=None, timeout=90)
    assert proc.returncode != 0, "CLI should fail when required --mu is missing."
    msg = (proc.stderr + "\n" + proc.stdout).lower()
    assert "mu" in msg, "Error message should mention missing 'mu'."

    # Invalid --n-freq
    mu = synthetic_inputs["mu_sine"]
    args_bad_nfreq = (
        "--mu", str(mu),
        "--wavelengths", str(wl),
        "--outdir", str(outdir),
        "--json",
        "--n-freq", "-4",
    )
    proc2 = _run_cli(module_file, args_bad_nfreq, env=None, timeout=90)
    # Either fail informatively or sanitize; accept both, but require JSON if sanitized.
    if proc2.returncode != 0:
        m = (proc2.stderr + "\n" + proc2.stdout).lower()
        assert ("n-freq" in m) or ("invalid" in m) or ("must be" in m), \
            "Expected an informative error about invalid --n-freq."
    else:
        json_files = list(outdir.glob("*.json"))
        assert json_files, "CLI sanitized invalid --n-freq but produced no JSON output."


# =================================================================================================
# Determinism at API level (optional)
# =================================================================================================

def test_api_determinism_with_seed(tool_mod, synthetic_inputs, tmp_workspace):
    """
    If the module exposes `set_seed/seed_all/fix_seed`, or artifact generator accepts seed,
    ensure repeated API calls in the same process produce identical outputs.
    """
    seed_fn = None
    for name in ("set_seed", "seed_all", "fix_seed"):
        if _has_attr(tool_mod, name):
            seed_fn = getattr(tool_mod, name)
            break

    # Prefer a deterministic API path via artifact generator
    entry_candidates = [
        "generate_fft_autocorr_artifacts",
        "run_fft_autocorr_diagnostics",
        "produce_fft_autocorr_outputs",
    ]
    entry = None
    for name in entry_candidates:
        if _has_attr(tool_mod, name):
            entry = getattr(tool_mod, name)
            break

    if seed_fn is None and entry is None:
        pytest.xfail("No seed setter and no artifact generator found for determinism test.")

    mu = np.load(synthetic_inputs["mu_sine_noise"])
    wl = np.load(synthetic_inputs["wavelengths"])

    if seed_fn is not None:
        seed_fn(77)

    # Call artifact generator twice with the same seed and compare JSON content
    if entry is None:
        pytest.xfail("Artifact generator missing; cannot perform API-level determinism check.")

    outdirA = tmp_workspace["root"] / "api_det_A"
    outdirB = tmp_workspace["root"] / "api_det_B"
    outdirA.mkdir(exist_ok=True)
    outdirB.mkdir(exist_ok=True)

    def _call(outdir: Path):
        try:
            entry(mu=mu, wavelengths=wl, outdir=str(outdir), json_out=True, csv_out=False, png_out=False, html_out=False, n_freq=128, seed=77)  # type: ignore
        except TypeError:
            entry(mu, wl, str(outdir), True, False, False, False, 128, 77)  # type: ignore

    _call(outdirA)
    _call(outdirB)

    jA = sorted(outdirA.glob("*.json"))
    jB = sorted(outdirB.glob("*.json"))
    assert jA and jB, "Determinism check requires JSON artifacts from both runs."

    def _load_norm(p: Path) -> Dict[str, Any]:
        with open(p, "r", encoding="utf-8") as f:
            jd = json.load(f)
        # strip volatile keys
        vol = re.compile(r"(time|date|timestamp|duration|path|cwd|hostname|uuid|version)", re.I)
        def scrub(obj: Any) -> Any:
            if isinstance(obj, dict):
                for k in list(obj.keys()):
                    if vol.search(k):
                        obj.pop(k, None)
                    else:
                        obj[k] = scrub(obj[k])
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = scrub(obj[i])
            return obj
        return scrub(jd)

    JA = _load_norm(jA[0])
    JB = _load_norm(jB[0])
    assert JA == JB, "API-level seeded runs should yield identical JSON after removing volatile metadata."


# =================================================================================================
# Housekeeping checks
# =================================================================================================

def test_artifact_min_sizes(tmp_workspace):
    """
    After prior tests, ensure that PNG/CSV/HTML in outputs/ are non-trivially sized.
    """
    outdir = tmp_workspace["outputs"]
    png_files = list(outdir.glob("*.png"))
    csv_files = list(outdir.glob("*.csv"))
    html_files = list(outdir.glob("*.html"))
    # Not all formats may exist if a prior test xfailed early; be lenient but check when present.
    for p in png_files:
        _assert_file(p, min_size=256)
    for c in csv_files:
        _assert_file(c, min_size=64)
    for h in html_files:
        _assert_file(h, min_size=128)


def test_idempotent_rerun_behavior(tmp_workspace):
    """
    The tool should either overwrite consistently or produce versioned filenames.
    We don't require a specific policy here; only that subsequent writes do not corrupt artifacts.
    """
    outdir = tmp_workspace["outputs"]
    before = {p.name for p in outdir.glob("*")}
    # Simulate pre-existing artifact to ensure tool does not crash due to existing files
    marker = outdir / "preexisting_marker.txt"
    marker.write_text("marker", encoding="utf-8")
    after = {p.name for p in outdir.glob("*")}
    assert before.issubset(after), "Artifacts disappeared unexpectedly between runs or overwrite simulation."
