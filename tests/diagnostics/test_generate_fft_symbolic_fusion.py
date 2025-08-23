# tests/diagnostics/test_generate_fft_symbolic_fusion.py
"""
Diagnostics: FFT × Symbolic-Fusion tests

This suite validates a deterministic, reproducible diagnostic that fuses:
  • frequency-domain power (FFT on residuals / noise traces),
  • symbolic safety/physics rules (e.g., non-negativity & bounds),
and emits a single "fusion" score + artifacts for the run.

Design goals:
  - Works offline, no network, no heavy runtime assumptions.
  - Deterministic with seeds.
  - Play nicely with Hydra output dirs & Typer CLI (if present in repo).

Author: SpectraMind V50 diagnostics
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

# Optional SciPy: we prefer it for windowing / periodograms if available.
try:
    from scipy.signal import get_window
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -------------------------------
# Utilities under test (inline)
# -------------------------------

def _fft_power(signal: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided frequency & power density for a real-valued trace.

    Parameters
    ----------
    signal : np.ndarray
        1D time-domain trace.
    fs_hz : float
        Sample rate in Hz.

    Returns
    -------
    f_hz : np.ndarray
        Frequency bins (one-sided, including DC & Nyquist).
    pxx : np.ndarray
        Power spectral density-like magnitude (not strictly PSD; enough for peak diagnostics).
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n == 0:
        return np.array([]), np.array([])

    # Detrend (mean removal) for diagnostics
    x = x - np.mean(x)

    if _HAVE_SCIPY:
        # Light window to reduce spectral leakage
        w = get_window("hann", n, fftbins=True)
        xw = x * w
    else:
        xw = x

    # rfft is one-sided for real input
    X = np.fft.rfft(xw)
    # Normalize like periodogram-ish so relative peaks are meaningful
    # (scale by N and window power if any; this keeps the test deterministic)
    scale = n
    mag = (np.abs(X) ** 2) / scale

    f_hz = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return f_hz, mag


@dataclass(frozen=True)
class SymbolicRules:
    """
    Simple symbolic constraints for spectra/series.

    - nonneg: spectrum values must be >= 0
    - max_amp: absolute bound (optional), e.g. physical/engineering sanity cap
    """
    nonneg: bool = True
    max_amp: float | None = None

    def violations(self, arr: np.ndarray) -> Dict[str, int]:
        v: Dict[str, int] = {}
        if self.nonneg:
            v["nonneg"] = int(np.sum(arr < 0))
        if self.max_amp is not None:
            v["max_amp"] = int(np.sum(np.abs(arr) > float(self.max_amp)))
        return v


def fuse_fft_and_symbolic(
    residual_trace: np.ndarray,
    fs_hz: float,
    spectrum_estimate: np.ndarray,
    rules: SymbolicRules,
    jitter_interest_hz: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    bandwidth_hz: float = 0.05,
    weights: Tuple[float, float] = (0.65, 0.35),
) -> Dict[str, float | Dict[str, int]]:
    """
    Produce a single 'fusion' diagnostic score combining frequency peaks (jitter)
    and symbolic rule violations.

    Fusion logic (deterministic & deliberately simple):
      - FFT: sum power in narrow bands around specified jitter_interest_hz.
      - Symbolic: count violations and map to penalty via a stable function.
      - Blend via weights (fft_w, sym_w) into 0..1 score where 0 = best, 1 = worst.
    """
    assert 0.0 < weights[0] < 1.0 and 0.0 < weights[1] < 1.0 and (abs(sum(weights) - 1.0) < 1e-9)
    f_hz, pxx = _fft_power(residual_trace, fs_hz)

    # If empty, we still want deterministic outputs
    if f_hz.size == 0:
        fft_peak_sum = 0.0
    else:
        fft_peak_sum = 0.0
        for f0 in jitter_interest_hz:
            band = (f_hz >= (f0 - bandwidth_hz)) & (f_hz <= (f0 + bandwidth_hz))
            if np.any(band):
                fft_peak_sum += float(np.sum(pxx[band]))

        # Normalize peak sum by overall energy to keep score in a bounded-ish range
        total = float(np.sum(pxx)) + 1e-12
        fft_component = fft_peak_sum / total
        # squash to [0,1) with a soft saturation so pathological cases stay bounded
        fft_term = 1.0 - math.exp(-fft_component)
        # numerical hygiene
        fft_term = max(0.0, min(1.0, fft_term))

    # Symbolic
    vmap = rules.violations(np.asarray(spectrum_estimate, dtype=float))
    vcount = sum(vmap.values())
    # Map counts to a 0..1 penalty via 1 - exp(-k * count). k picked to be gentle yet visible.
    k = 0.25
    sym_term = 1.0 - math.exp(-k * float(vcount))

    # Blend
    fft_w, sym_w = weights
    fused = fft_w * (fft_term if f_hz.size else 0.0) + sym_w * sym_term

    return {
        "fft_term": float(fft_term if f_hz.size else 0.0),
        "symbolic_term": float(sym_term),
        "fusion_score": float(fused),
        "violations": vmap,
        "fft_peak_sum": float(fft_peak_sum),
    }


# -------------------------------
# Unit tests
# -------------------------------

@pytest.mark.parametrize("seed", [7, 42, 1234])
def test_fft_symbolic_fusion_repro(seed: int) -> None:
    """Deterministic fusion across seeds when input is fixed."""
    rng = np.random.default_rng(0)  # fixed input; 'seed' is unused on purpose
    fs = 20.0  # Hz
    t = np.arange(0, 40.0, 1.0 / fs)
    # Two small jitter tones at 1 Hz and 2 Hz + white noise
    jitter = 0.015 * np.sin(2 * np.pi * 1.0 * t) + 0.02 * np.sin(2 * np.pi * 2.0 * t)
    residual = jitter + 0.005 * rng.standard_normal(t.size)

    # A toy spectrum with a few nonneg violations and one clipping violation:
    spectrum = np.ones(283) * 0.1
    spectrum[10] = -0.05      # 1 violation
    spectrum[200] = -0.02     # 2 violations
    spectrum[50] = 1.5        # will trigger max_amp if set low

    rules = SymbolicRules(nonneg=True, max_amp=1.0)
    out = fuse_fft_and_symbolic(residual, fs, spectrum, rules)

    # Basic sanity
    assert 0.0 <= out["fft_term"] <= 1.0
    assert 0.0 <= out["symbolic_term"] <= 1.0
    assert 0.0 <= out["fusion_score"] <= 1.0
    assert isinstance(out["violations"], dict)
    # Expected exact counts from above edits
    assert out["violations"]["nonneg"] == 2
    assert out["violations"]["max_amp"] == 1

    # Deterministic: calling again yields same numbers
    out2 = fuse_fft_and_symbolic(residual, fs, spectrum, rules)
    np.testing.assert_allclose(
        [out["fft_term"], out["symbolic_term"], out["fusion_score"]],
        [out2["fft_term"], out2["symbolic_term"], out2["fusion_score"]],
        rtol=0, atol=1e-12,
    )


def test_fft_handles_edge_cases() -> None:
    """Edge-cases: empty, zeros, NaNs are handled gracefully."""
    rules = SymbolicRules(nonneg=True, max_amp=0.5)

    # Empty residual => no FFT term contribution, only symbolic counts matter
    out_empty = fuse_fft_and_symbolic(np.array([]), 10.0, np.array([0.1, -0.2]), rules)
    assert out_empty["fft_term"] == 0.0
    assert out_empty["violations"]["nonneg"] == 1

    # All-zero residual => small numerical total, bounded fft_term
    out_zeros = fuse_fft_and_symbolic(np.zeros(128), 64.0, np.array([0.0, 0.0]), rules)
    assert 0.0 <= out_zeros["fft_term"] <= 1.0

    # Spectrum with NaN: treat NaN as not violating nonnegativity but still bounded in amplitude test
    spec = np.array([np.nan, 0.2, -0.1, 0.8])
    # Replace NaNs safely before rules (simulate upstream sanitizer)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    out_nans = fuse_fft_and_symbolic(np.zeros(256), 128.0, spec, rules)
    assert out_nans["violations"]["nonneg"] == 1
    assert out_nans["violations"]["max_amp"] == int(np.sum(np.abs(spec) > 0.5))


def test_fft_detects_known_jitter_lines() -> None:
    """FFT term grows when the known jitter lines are present."""
    fs = 100.0
    t = np.arange(0, 20.0, 1.0 / fs)

    # Case A: no jitter, just noise
    rng = np.random.default_rng(123)
    residual_a = 0.01 * rng.standard_normal(t.size)

    # Case B: clear 0.5 Hz and 2.0 Hz lines
    residual_b = residual_a + 0.05 * np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.sin(2 * np.pi * 2.0 * t)

    rules = SymbolicRules(nonneg=True, max_amp=None)
    spec_dummy = np.ones(283) * 0.1
    out_a = fuse_fft_and_symbolic(residual_a, fs, spec_dummy, rules)
    out_b = fuse_fft_and_symbolic(residual_b, fs, spec_dummy, rules)

    assert out_b["fft_term"] > out_a["fft_term"]
    # Fusion should also increase if symbolic term is identical
    assert out_b["fusion_score"] > out_a["fusion_score"]


@pytest.mark.skipif(
    not any(p.stem.startswith("spectramind") for p in Path(".").glob("**/*.py")),
    reason="CLI module 'spectramind' not present in this environment.",
)
def test_cli_diagnose_produces_artifacts_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If the Typer CLI 'spectramind' is available, validate that:
      - `spectramind diagnose` (or equivalent) can run a minimal FFT+symbolic fusion diagnostic,
      - writes artifacts into a target dir under tmp_path,
      - saves a run configuration / log so the diagnostic is reproducible.

    Notes:
      • We don't assert exact CLI name/flags—only behavior. Adapt if your subcommand differs.
      • Test is isolated in tmp_path to avoid touching real work areas.
    """
    # Prefer to import the local CLI entry (Typer app) if available
    # Expected pattern: spectramind.py exposes `app` or `cli` (Typer)
    mod = None
    app = None
    for candidate in ("spectramind", "src.spectramind", "app.spectramind"):
        try:
            mod = __import__(candidate, fromlist=["*"])
            app = getattr(mod, "app", None) or getattr(mod, "cli", None)
            if app is not None:
                break
        except Exception:
            continue

    if app is None:
        pytest.skip("Typer app not found on import.")

    from click.testing import CliRunner

    runner = CliRunner(mix_stderr=False)
    # Point any output / hydra run dir to tmp_path if the CLI supports it.
    # We pass a minimal JSON payload via stdin or flag (adjust to your command shape).
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Minimal diagnostic input to exercise FFT+symbolic:
    payload = {
        "fs_hz": 50.0,
        "residual": (np.sin(2 * np.pi * 1.0 * np.arange(0, 10, 1/50.0))).tolist(),
        "spectrum": (np.ones(283) * 0.1).tolist(),
        "rules": {"nonneg": True, "max_amp": 1.0},
        "out_dir": str(artifacts_dir),
    }
    input_json = json.dumps(payload)

    # Try a few likely subcommand names; adapt to your actual command in the repo.
    possible_cmds = [
        ["diagnose", "fft-symbolic-fusion"],
        ["diagnostics", "fft-symbolic-fusion"],
        ["diagnose"],  # maybe the diagnostic is selected by config
    ]

    last_result = None
    for cmd in possible_cmds:
        result = runner.invoke(app, [*cmd, "--stdin-json"], input=input_json, catch_exceptions=False)
        last_result = result
        if result.exit_code == 0:
            break

    assert last_result is not None, "No CLI attempt was executed."
    assert last_result.exit_code == 0, f"CLI failed: {last_result.output}"

    # Check artifacts exist: a JSON summary & a small plot/log expected from diagnostics
    summary = artifacts_dir / "fft_symbolic_fusion_summary.json"
    assert summary.exists(), f"Missing diagnostic summary at {summary}"
    data = json.loads(summary.read_text())
    for k in ("fft_term", "symbolic_term", "fusion_score"):
        assert k in data and isinstance(data[k], (int, float))

    # Check reproducibility breadcrumbs (Hydra/Config/Log). We don’t enforce exact filenames,
    # only that something config-ish landed in the output tree.
    config_like = list(artifacts_dir.glob("**/*config*.*")) + list(artifacts_dir.glob("**/*hydra*.*")) + list(artifacts_dir.glob("**/*log*.*"))
    assert len(config_like) > 0, "Expected at least one config/log artifact for reproducibility."


def test_weight_blend_is_bounded() -> None:
    """Weights must sum to 1 and final score must stay in [0,1]."""
    fs = 32.0
    t = np.arange(0, 8.0, 1.0 / fs)
    residual = 0.02 * np.sin(2 * np.pi * 2.0 * t)
    spectrum = np.zeros(283)

    rules = SymbolicRules()
    out = fuse_fft_and_symbolic(residual, fs, spectrum, rules, weights=(0.7, 0.3))
    assert 0.0 <= out["fusion_score"] <= 1.0

    with pytest.raises(AssertionError):
        _ = fuse_fft_and_symbolic(residual, fs, spectrum, rules, weights=(0.9, 0.2))  # sums != 1

