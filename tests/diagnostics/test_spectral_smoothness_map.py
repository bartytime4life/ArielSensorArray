# tests/diagnostics/test_spectral_smoothness_map.py
# -*- coding: utf-8 -*-
"""
Pytest suite for tools/spectral_smoothness_map.py

SpectraMind V50 — Diagnostics: Spectral Smoothness Map

This test module validates both:
  1) The *library API* (if the tool exposes a callable to compute smoothness), and
  2) The *CLI interface* (python -m tools.spectral_smoothness_map ...)

Design & coverage
-----------------
• Robust import: we try to import `tools.spectral_smoothness_map` (the canonical path used in this repo).
  - If the module exposes `compute_smoothness(...)`, we unit-test it on synthetic spectra.
  - If instead it exposes an analyzer class (e.g., SpectralSmoothnessAnalyzer) with a compatible method,
    we adapt and test via duck-typing.
  - If neither is present, we fail with a clear, actionable error explaining what API to expose.

• CLI checks:
  - `--help` runs and prints a useful description (no crash, non-empty usage text).
  - End-to-end run: given a synthetic μ (.npy) with multiple "planets", the CLI writes expected outputs
    (metrics JSON/CSV and one or more figures/heatmaps) into a provided outdir.
  - Logging: we set SPECTRAMIND_LOGS_DIR to a temp path and assert that the tool appends to
    v50_debug_log.md (append-only audit log required by project standards).

• Scientific sanity:
  - Smooth spectrum should yield LOWER smoothness than a noisy/rough spectrum.
    We enforce this via deterministic synthetic spectra.

Assumptions (kept loose but explicit to avoid brittleness)
----------------------------------------------------------
• Module path: tools/spectral_smoothness_map
• Library API: one of the following must exist:
    def compute_smoothness(mu: np.ndarray, *, method: str = "L2_second_diff", **kwargs) -> float | np.ndarray
  or:
    class SpectralSmoothnessAnalyzer:
        def compute_smoothness(self, mu: np.ndarray, **kwargs) -> float | np.ndarray
• CLI:
    python -m tools.spectral_smoothness_map --mu <path/to/mu.npy> --outdir <dir> [--png] [--csv] [--json] [--html]
  The tool should accept a 1D μ (length = n_bins) or 2D μ (n_planets × n_bins). It should create:
    - At least one of: *.json, *.csv files containing per-planet smoothness metrics
    - At least one visualization: *.png (or *.html) figure(s)
  It should also honor SPECTRAMIND_LOGS_DIR=<dir> and append to <dir>/v50_debug_log.md

Notes
-----
• This test suite intentionally prefers *behavioral* assertions over internal implementation details.
• If you change function/class names or flags in the tool, update the compatibility adapters below.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest


# ---------- Utilities & adapters ----------------------------------------------------------------


def _import_tool_module():
    """
    Attempt to import the tool module and return (mod, callable) where callable is the function
    to compute smoothness. Adapts to either a bare function or analyzer class with method.
    """
    try:
        import tools.spectral_smoothness_map as mod  # type: ignore
    except Exception as e:
        raise AssertionError(
            "Unable to import 'tools.spectral_smoothness_map'. Ensure the file exists at "
            "tools/spectral_smoothness_map.py and is importable. Original error: "
            f"{type(e).__name__}: {e}"
        )
    # Preferred: direct function
    if hasattr(mod, "compute_smoothness") and callable(getattr(mod, "compute_smoothness")):
        return mod, getattr(mod, "compute_smoothness")

    # Fallback: class-based analyzer exposing compute_smoothness
    analyzer_cls = getattr(mod, "SpectralSmoothnessAnalyzer", None)
    if analyzer_cls is not None:
        analyzer = analyzer_cls()  # type: ignore[call-arg]
        if hasattr(analyzer, "compute_smoothness") and callable(getattr(analyzer, "compute_smoothness")):
            # Wrap instance method to look like a function
            def compute(mu: np.ndarray, **kwargs) -> Any:
                return analyzer.compute_smoothness(mu, **kwargs)

            return mod, compute

    # If we reach here, we couldn't find a programmable API
    raise AssertionError(
        "The tool module 'tools.spectral_smoothness_map' must expose either:\n"
        "  • def compute_smoothness(mu: np.ndarray, *, method: str = 'L2_second_diff', **kwargs) -> float | np.ndarray\n"
        "or\n"
        "  • class SpectralSmoothnessAnalyzer with method compute_smoothness(mu: np.ndarray, **kwargs) -> float | np.ndarray\n"
        "Please implement one of these APIs so unit tests can exercise the scientific logic."
    )


def _run_cli(args: list[str], env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess:
    """
    Invoke the CLI as `python -m tools.spectral_smoothness_map <args...>` and return the completed process.
    Raises if python exits with non-zero status.
    """
    cmd = [sys.executable, "-m", "tools.spectral_smoothness_map"] + args
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    return cp


def _write_mu_file(path: Path, mu: np.ndarray) -> None:
    """
    Save a μ array to an .npy file at `path`. Creates parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mu)


def _find_files_with_suffix(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """
    Recursively find files in `root` that end with any of the given suffixes.
    """
    matches: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            matches.append(p)
    return matches


# ---------- Synthetic data builders --------------------------------------------------------------


def _make_smooth_spectrum(n_bins: int = 283, amplitude: float = 0.01, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Create a *smooth* synthetic transmission spectrum μ(λ) with mild sinusoidal variation.

    The general shape is a low-amplitude sinusoid + gentle slope to mimic broad features.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, n_bins, endpoint=True)
    baseline = 0.02 + 0.002 * (np.linspace(0, 1, n_bins))  # gentle slope + offset
    sinusoid = amplitude * np.sin(2.5 * x)                 # low-frequency wiggles
    return baseline + sinusoid


def _make_noisy_spectrum(n_bins: int = 283, noise_scale: float = 0.02, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Create a *noisy/rough* synthetic spectrum by adding higher-frequency oscillations and random noise.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    x = np.linspace(0, 4 * np.pi, n_bins, endpoint=True)
    # Mix of higher-frequency sine + random jitter
    hf = 0.008 * np.sin(18 * x) + 0.005 * np.sin(47 * x + 0.4)
    noise = rng.normal(0.0, noise_scale, size=n_bins)
    baseline = 0.02 + 0.002 * np.linspace(0, 1, n_bins)
    return baseline + hf + noise


def _stack_planets(*spectra: np.ndarray) -> np.ndarray:
    """
    Stack 1D spectra (n_bins,) into a 2D array (n_planets, n_bins).
    """
    return np.stack(spectra, axis=0)


# ---------- Tests: Library API ------------------------------------------------------------------


def test_compute_smoothness_prefers_smooth_over_noisy():
    """
    Unit test for the scientific core: a smoother spectrum must score *lower smoothness* than a rough/noisy spectrum.

    We don't lock to an exact number (implementation may differ in normalization), but we assert strict ordering.
    """
    mod, compute_smoothness = _import_tool_module()

    smooth = _make_smooth_spectrum()
    noisy = _make_noisy_spectrum()

    # We expect either a scalar or a length-1 array for 1D input; normalize to float
    s_smooth = compute_smoothness(smooth)
    s_noisy = compute_smoothness(noisy)

    if isinstance(s_smooth, (list, tuple, np.ndarray)):
        assert np.size(s_smooth) == 1, "compute_smoothness(1D) should return scalar-like value"
        s_smooth = float(np.asarray(s_smooth).ravel()[0])
    if isinstance(s_noisy, (list, tuple, np.ndarray)):
        assert np.size(s_noisy) == 1, "compute_smoothness(1D) should return scalar-like value"
        s_noisy = float(np.asarray(s_noisy).ravel()[0])

    assert np.isfinite(s_smooth) and np.isfinite(s_noisy), "Smoothness results must be finite floats"
    assert s_smooth < s_noisy, (
        f"Expected smoother spectrum to have LOWER smoothness. Got smooth={s_smooth:.6g}, noisy={s_noisy:.6g}"
    )


def test_compute_smoothness_vectorized_over_planets():
    """
    If the implementation supports batching, then a 2D input (n_planets, n_bins) should return per-planet metrics.

    We verify:
      • correct output shape (n_planets,)
      • correct ordering: planet[0] (smooth) < planet[1] (noisy)
    """
    mod, compute_smoothness = _import_tool_module()

    smooth = _make_smooth_spectrum()
    noisy = _make_noisy_spectrum()
    mu_2d = _stack_planets(smooth, noisy)

    res = compute_smoothness(mu_2d)
    res = np.asarray(res)

    assert res.ndim == 1 and res.shape[0] == 2, (
        "For 2D input (n_planets, n_bins), compute_smoothness should return a 1D array of length n_planets"
    )
    assert np.all(np.isfinite(res)), "Smoothness values must be finite"
    assert res[0] < res[1], (
        f"Expected planet 0 (smooth) < planet 1 (noisy) in smoothness. Got {res.tolist()}"
    )


# ---------- Tests: CLI interface ----------------------------------------------------------------


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_cli_help_runs(flag):
    """
    `python -m tools.spectral_smoothness_map --help` should succeed and show usage text.
    """
    cp = _run_cli([flag])
    assert cp.returncode == 0, f"Help should exit(0). stderr:\n{cp.stderr}"
    # Basic heuristics for useful help text
    output = cp.stdout + "\n" + cp.stderr
    assert re.search(r"(?i)(usage|options|smoothness|μ|mu|spectrum)", output), (
        "Help output should mention usage/options and smoothness concepts."
    )


def test_cli_end_to_end_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    End-to-end CLI test:
      • Build a small batch of spectra (smooth & noisy) and save to mu.npy
      • Run CLI to compute and export diagnostics into `outdir`
      • Assert expected outputs exist:
            - At least one metrics file: *.json or *.csv
            - At least one visualization: *.png or *.html
      • Assert v50_debug_log.md was appended in a configured logs dir
    """
    # Arrange inputs
    outdir = tmp_path / "outputs"
    logsdir = tmp_path / "logs"
    mu_path = tmp_path / "mu.npy"

    smooth = _make_smooth_spectrum()
    noisy = _make_noisy_spectrum()
    mu_2d = _stack_planets(smooth, noisy)
    _write_mu_file(mu_path, mu_2d)

    # Ensure tool logs into our temp logs dir (append-only audit log)
    env = os.environ.copy()
    env["SPECTRAMIND_LOGS_DIR"] = str(logsdir)

    # Try a conservative set of flags. The tool should accept --mu and --outdir at minimum.
    # We add optional export toggles commonly supported in this codebase.
    candidate_flags = [
        "--mu", str(mu_path),
        "--outdir", str(outdir),
        "--png",
        "--csv",
        "--json",
        "--html",
    ]
    cp = _run_cli(candidate_flags, env=env)

    # The tool should succeed (exit code 0). If it returns non-zero, surface both streams.
    assert cp.returncode == 0, textwrap.dedent(
        f"""
        CLI returned non-zero exit code {cp.returncode}.
        --- STDOUT ---
        {cp.stdout}
        --- STDERR ---
        {cp.stderr}
        """
    )

    # Check outputs: metrics files and visualizations
    metrics = _find_files_with_suffix(outdir, (".json", ".csv"))
    figures = _find_files_with_suffix(outdir, (".png", ".html"))

    assert len(metrics) >= 1, f"Expected at least one metrics file (*.json or *.csv) in {outdir}, found none."
    assert len(figures) >= 1, f"Expected at least one figure (*.png or *.html) in {outdir}, found none."

    # Optionally, inspect a JSON to ensure it contains plausible keys.
    json_files = [p for p in metrics if p.suffix.lower() == ".json"]
    if json_files:
        # Pick the first JSON and check it has per-planet entries with numeric smoothness
        with json_files[0].open("r", encoding="utf-8") as f:
            payload = json.load(f)
        # Accept either a list of per-planet dicts or a dict keyed by planet_id.
        if isinstance(payload, list):
            assert len(payload) >= 2, "Expected at least two planets in the JSON metrics list"
            for row in payload:
                # Try common key names
                val = None
                for k in ("smoothness", "smooth", "score", "value"):
                    if k in row:
                        val = row[k]
                        break
                assert val is not None, "JSON row must contain a smoothness-like key"
                assert np.isfinite(float(val))
        elif isinstance(payload, dict):
            # Look for any numeric values in dict (possibly nested)
            def _any_numeric(d: dict) -> bool:
                for v in d.values():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        return True
                    if isinstance(v, dict) and _any_numeric(v):
                        return True
                    if isinstance(v, list) and any(isinstance(x, (int, float)) and np.isfinite(x) for x in v):
                        return True
                return False

            assert _any_numeric(payload), "JSON metrics should contain at least one finite numeric value"
        else:
            pytest.fail("Unexpected JSON structure for metrics; expected dict or list.")

    # Logging: verify v50_debug_log.md was created/updated in SPECTRAMIND_LOGS_DIR
    log_file = logsdir / "v50_debug_log.md"
    assert log_file.exists(), f"Expected audit log at {log_file} (ensure tool honors SPECTRAMIND_LOGS_DIR)"
    content = log_file.read_text(encoding="utf-8", errors="ignore")
    assert re.search(r"(?i)(spectral_smoothness_map|smoothness|diagnose|outputs)", content), (
        "Audit log should include an entry mentioning this tool or its outputs."
    )


def test_cli_respects_overwrite_and_outdir_creation(tmp_path: Path):
    """
    The tool should create the outdir if missing and either overwrite or version outputs safely.

    This test runs CLI twice with the same outdir and asserts it does not crash on the second run.
    (We don't enforce specific overwrite semantics—just that it's robust.)
    """
    outdir = tmp_path / "out_nested" / "deep"
    mu_path = tmp_path / "mu_small.npy"

    # smaller shape for speed
    smooth = _make_smooth_spectrum(n_bins=96)
    noisy = _make_noisy_spectrum(n_bins=96)
    mu_2d = _stack_planets(smooth, noisy)
    _write_mu_file(mu_path, mu_2d)

    # First run
    cp1 = _run_cli(["--mu", str(mu_path), "--outdir", str(outdir), "--png", "--csv", "--json"])
    assert cp1.returncode == 0, f"First run failed. stderr:\n{cp1.stderr}"

    # Second run (should also succeed; tool may overwrite or write new versioned files)
    cp2 = _run_cli(["--mu", str(mu_path), "--outdir", str(outdir), "--png", "--csv", "--json"])
    assert cp2.returncode == 0, f"Second run failed. stderr:\n{cp2.stderr}"

    # Ensure directory exists and has content
    assert outdir.exists() and any(outdir.iterdir()), "Outdir should exist and contain outputs after runs"


# ---------- Optional: Negative/edge cases -------------------------------------------------------


def test_compute_smoothness_rejects_nan_inputs():
    """
    The scientific function should either cleanly handle NaNs (e.g., ignore or fill) or raise a clear error.
    We accept either behavior, but if it returns a value, it must be finite.
    """
    _, compute_smoothness = _import_tool_module()

    mu = _make_smooth_spectrum()
    mu[10:13] = np.nan  # inject NaNs
    try:
        val = compute_smoothness(mu)
        # If it returns, enforce finiteness
        if isinstance(val, (list, tuple, np.ndarray)):
            val = float(np.asarray(val).ravel()[0])
        assert np.isfinite(val), "compute_smoothness returned non-finite result for NaN input"
    except Exception as e:
        # Also acceptable: raise a clear, user-facing error
        msg = str(e).lower()
        assert any(tok in msg for tok in ("nan", "invalid", "missing", "finite")), (
            "If raising, error message should clearly indicate NaN/invalid input handling."
        )


def test_cli_fails_cleanly_on_bad_input(tmp_path: Path):
    """
    Passing a non-existent or malformed --mu should cause a non-zero exit code and a helpful error message.
    """
    bad_mu = tmp_path / "does_not_exist.npy"
    cp = _run_cli(["--mu", str(bad_mu), "--outdir", str(tmp_path / "o")])

    # We expect a failure; if the tool opts to create a helpful scaffold and exit 0, that's okay too,
    # but it must print a helpful message to stderr/stdout.
    if cp.returncode != 0:
        out = (cp.stdout + "\n" + cp.stderr).lower()
        assert any(s in out for s in ("not found", "missing", "cannot", "load", "read", "invalid", "file")), (
            "On failure, the tool should explain the problem with --mu path."
        )
    else:
        # If it exits 0, ensure it printed a warning about the bad input.
        out = (cp.stdout + "\n" + cp.stderr).lower()
        assert any(s in out for s in ("not found", "missing", "cannot", "load", "read", "invalid", "file")), (
            "Even if returning exit(0), the tool must warn about the invalid --mu."
        )
