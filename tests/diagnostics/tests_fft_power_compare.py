#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_fft_power_compare.py

SpectraMind V50 — Diagnostics Tests (FFT power compare)

This suite validates the FFT power comparison / clustering diagnostics tool.
It is tolerant to different filenames/entrypoints your repo may use, such as:

  • tools/plot_fft_power_cluster_compare.py
  • tools/fft_power_compare.py
  • tools/analyze_fft_autocorr_mu.py   (some teams fold FFT compare into this script)

Objectives
----------
1) End-to-end CLI run on tiny synthetic inputs (time-series and/or μ spectra).
2) Verify at least one plot (PNG/SVG/PDF) is created in the specified --outdir.
3) Confirm append-only audit logging into logs/v50_debug_log.md.
4) Ensure --outdir is respected (no stray writes outside, except logs/ and run-hash JSON).
5) Exercise edge cases (very short sequences, NaNs) without crashing.
6) Module invocation mode (`python -m tools.<name>`) works if importable.

Design
------
• Self-contained repo scaffold (tools/, logs/, outputs/).
• Synthetic signals: sine + jitter with distinct power bands to make clustering plausible.
• Flexible CLI args: we pass common flags, but tests don't fail if your tool ignores unknown ones.
• We do NOT over-constrain exact filenames—only require that at least one image artifact exists.

Run
---
pytest -q tests/diagnostics/test_fft_power_compare.py
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

# Force headless plotting for CI environments
os.environ.setdefault("MPLBACKEND", "Agg")


# --------------------------------------
# Repo scaffold + synthetic inputs
# --------------------------------------

CANDIDATE_SCRIPTS = [
    "plot_fft_power_cluster_compare.py",
    "fft_power_compare.py",
    "analyze_fft_autocorr_mu.py",
]

CANDIDATE_MODULES = [
    "tools.plot_fft_power_cluster_compare",
    "tools.fft_power_compare",
    "tools.analyze_fft_autocorr_mu",
]


def _ensure_repo_scaffold(repo_root: Path) -> None:
    """
    Create a minimal repo-like layout so default relative paths used by the tool
    (e.g., logs/, outputs/) resolve cleanly.
    If none of the candidate tools exists, write a shim that exits with a helpful message.
    """
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "diagnostics").mkdir(parents=True, exist_ok=True)

    tool_dir = repo_root / "tools"
    if not any((tool_dir / name).exists() for name in CANDIDATE_SCRIPTS):
        # Create a shim for the first candidate to make the skip reasoning explicit
        shim_path = tool_dir / CANDIDATE_SCRIPTS[0]
        shim_path.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                # Shim placeholder for FFT diagnostics tool.
                # Replace with one of: {", ".join(CANDIDATE_SCRIPTS)}
                import sys
                if __name__ == "__main__":
                    sys.exit("Shim placeholder: replace with a real FFT diagnostics tool in tools/.")
                """
            ),
            encoding="utf-8",
        )


def _make_sine_mix_time_series(
    n_series: int = 8,
    length: int = 256,
    seed: int = 123,
    sample_rate_hz: float = 1.0,
) -> np.ndarray:
    """
    Construct tiny multi-series signals:
      x[t] = a1*sin(2π f1 t) + a2*sin(2π f2 t) + jitter
    with per-series random amplitudes/frequencies to create distinct FFT signatures.
    Returns array of shape (n_series, length).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64) / sample_rate_hz

    X = []
    for i in range(n_series):
        # Choose two frequencies in different bands (normalized)
        f1 = rng.uniform(0.03, 0.08)  # low-ish
        f2 = rng.uniform(0.15, 0.25)  # higher
        a1 = rng.uniform(0.5, 1.5)
        a2 = rng.uniform(0.3, 1.2)
        sig = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
        sig += rng.normal(0, 0.1, size=length)
        # small trend for realism
        sig += 0.001 * (i + 1) * (t - t.mean())
        X.append(sig)
    return np.vstack(X)


def _make_mu_spectra_from_signals(signals: np.ndarray) -> np.ndarray:
    """
    Derive a tiny 'μ spectra' proxy by taking the magnitude of the FFT at a handful
    of pseudo-'wavelength bins'. This is only for testing file I/O paths where the tool
    expects μ(λ) arrays. Shape: (n_series, n_bins).
    """
    n_series, length = signals.shape
    fft = np.abs(np.fft.rfft(signals, axis=1))
    # pick a few stable bins (excluding DC)
    idx = np.linspace(2, min(fft.shape[1] - 1, 60), 17).astype(int)
    mu = fft[:, idx] / (fft[:, idx].max(axis=1, keepdims=True) + 1e-9)
    return mu.astype(np.float64)


def _write_inputs(repo_root: Path) -> Dict[str, Path]:
    """
    Write minimal inputs for both "time-series" and "μ spectra" workflows so the tool
    can choose either. Returns a dict of useful paths.
    """
    inputs = repo_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    signals = _make_sine_mix_time_series(n_series=8, length=192, seed=321)
    mu = _make_mu_spectra_from_signals(signals)

    ts_path = inputs / "time_series.npy"
    mu_path = inputs / "mu.npy"
    labels_path = inputs / "labels.txt"

    np.save(ts_path, signals)
    np.save(mu_path, mu)
    labels = [f"group_{i%2}" for i in range(signals.shape[0])]
    labels_path.write_text("\n".join(labels), encoding="utf-8")

    return {"time_series": ts_path, "mu": mu_path, "labels": labels_path}


def _find_tool_script(repo_root: Path) -> Optional[Path]:
    """
    Return a path to the first candidate tool found in tools/.
    """
    for name in CANDIDATE_SCRIPTS:
        p = repo_root / "tools" / name
        if p.exists():
            return p
    return None


def _find_tool_module() -> Optional[str]:
    """
    Return the first importable candidate module path (tools.<name>).
    """
    for mod in CANDIDATE_MODULES:
        try:
            __import__(mod)
            return mod
        except Exception:
            continue
    return None


def _run_tool_subprocess(
    repo_root: Path,
    tool_path: Optional[Path],
    module_name: Optional[str],
    inputs: Dict[str, Path],
    outdir: Path,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """
    Execute the FFT tool either by file path or as a module.
    Prefer a direct file path when available; module is a fallback.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SPECTRAMIND_TEST", "1")

    base_args: List[str] = []
    # Supply both time-series and μ inputs; accept that the tool may use only one.
    # Provide tolerant flags that many implementations accept; unknown flags should be ignored gracefully.
    base_args += [
        # common flags seen across repos/tools
        "--outdir", str(outdir),
        "--no-open",
        "--save-plots",
        "--quiet",
        "--version", "test",
    ]

    # Try a variety of input flag spellings (tool should accept at least one).
    # If the tool rejects unknowns with non-zero exit, later tests will xfail with context.
    input_flag_sets = [
        ["--time-series", str(inputs["time_series"]), "--labels", str(inputs["labels"])],
        ["--ts", str(inputs["time_series"]), "--labels", str(inputs["labels"])],
        ["--mu", str(inputs["mu"]), "--labels", str(inputs["labels"])],
    ]

    # We'll attempt up to 3 runs with alternative flag sets until one succeeds.
    tried_cmds: List[List[str]] = []

    for flagset in input_flag_sets:
        if module_name:
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", module_name]
        else:
            assert tool_path is not None
            cmd = [sys.executable, str(tool_path)]

        args = base_args + flagset
        if extra_args:
            args += list(extra_args)

        tried_cmds.append(cmd + args)
        proc = subprocess.run(
            cmd + args,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if proc.returncode == 0:
            return proc
        # else try next flag set; continue

    # If none succeeded, return the last proc for debugging
    return proc  # type: ignore[name-defined]


def _scan_new_artifacts(root: Path, before: Sequence[Path]) -> List[Path]:
    """
    Find new image artifacts (PNG/SVG/PDF) in outdir after running tool.
    """
    before_set = set(before)
    exts = {".png", ".svg", ".pdf", ".jpg", ".jpeg"}
    after = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return [p for p in after if p not in before_set]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ----------------------
# Pytest fixtures
# ----------------------

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path) -> Path:
    _ensure_repo_scaffold(tmp_path)
    return tmp_path


@pytest.fixture(scope="function")
def tiny_inputs(repo_tmp: Path) -> Dict[str, Path]:
    return _write_inputs(repo_tmp)


# ---------------------------------------
# Core tests — end-to-end and robustness
# ---------------------------------------

@pytest.mark.integration
def test_fft_power_compare_generates_plots(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    End-to-end run:
      • Execute tool with small time-series/μ inputs.
      • Expect at least one image artifact in --outdir.
      • Confirm audit log appended.
    """
    tool_path = _find_tool_script(repo_tmp)
    module_name = _find_tool_module() if tool_path is None else None

    # If neither script nor module is present, skip with a clear message.
    if tool_path is None and module_name is None:
        pytest.skip("No FFT diagnostics tool found under tools/. Add one of the candidates to enable this test.")

    outdir = repo_tmp / "outputs" / "diagnostics" / "fft_compare_run"
    outdir.mkdir(parents=True, exist_ok=True)

    before = list(outdir.rglob("*"))

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool_path,
        module_name=module_name,
        inputs=tiny_inputs,
        outdir=outdir,
        extra_args=["--dpi", "120", "--figsize", "8,6"],
    )

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, "FFT diagnostics tool should exit successfully."

    artifacts = _scan_new_artifacts(outdir, before)
    assert artifacts, "Expected at least one plot artifact (PNG/SVG/PDF) in the outdir."

    # Audit log presence
    log_path = repo_tmp / "logs" / "v50_debug_log.md"
    assert log_path.exists(), "Expected audit log logs/v50_debug_log.md"
    log_text = _read_text(log_path).lower()
    assert "fft" in log_text or "power" in log_text or "autocorr" in log_text, \
        "Audit log should mention FFT/power/autocorr diagnostics."


@pytest.mark.integration
def test_outdir_respected_no_stray_writes(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Ensure the tool writes under --outdir (plus logs/); no other stray writes in repo root.
    """
    tool_path = _find_tool_script(repo_tmp)
    module_name = _find_tool_module() if tool_path is None else None
    if tool_path is None and module_name is None:
        pytest.skip("No FFT diagnostics tool available; skipping.")

    outdir = repo_tmp / "outputs" / "diagnostics" / "fft_compare_outdir"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool_path,
        module_name=module_name,
        inputs=tiny_inputs,
        outdir=outdir,
        extra_args=["--dpi", "110", "--quiet"],
    )
    assert proc.returncode == 0

    after = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())
    new_files = sorted(list(after - before))

    # Allowed: anything under outdir, logs/*, and optionally outputs/run_hash_summary*.json
    disallowed = []
    out_rel = outdir.relative_to(repo_tmp).as_posix()
    for rel in new_files:
        if rel.startswith("logs/"):
            continue
        if rel.startswith(out_rel):
            continue
        if rel.startswith("outputs/") and re.search(r"run_hash_summary.*\.json$", rel):
            continue
        if rel.endswith(".pyc") or "/__pycache__/" in rel:
            continue
        disallowed.append(rel)

    assert not disallowed, f"FFT tool wrote unexpected files outside --outdir: {disallowed}"


@pytest.mark.integration
def test_handles_short_sequences_and_nans(repo_tmp: Path) -> None:
    """
    Edge cases:
      • Very short sequences (length ~ 32).
      • Inject NaNs; tool should either sanitize or skip gracefully and still complete.
    """
    tool_path = _find_tool_script(repo_tmp)
    module_name = _find_tool_module() if tool_path is None else None
    if tool_path is None and module_name is None:
        pytest.skip("No FFT diagnostics tool available; skipping.")

    # Build tiny inputs with NaNs
    inputs_dir = repo_tmp / "inputs_edge"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    ts = _make_sine_mix_time_series(n_series=4, length=32, seed=7)
    ts[1, 3] = np.nan  # inject
    mu = _make_mu_spectra_from_signals(ts)

    ts_path = inputs_dir / "ts_short.npy"
    mu_path = inputs_dir / "mu_short.npy"
    labels_path = inputs_dir / "labels.txt"
    np.save(ts_path, ts)
    np.save(mu_path, mu)
    labels_path.write_text("A\nB\nA\nB\n", encoding="utf-8")

    outdir = repo_tmp / "outputs" / "diagnostics" / "fft_compare_edge"
    outdir.mkdir(parents=True, exist_ok=True)

    # Try with μ first (some tools operate on μ-only); then fallback to time-series flags.
    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool_path,
        module_name=module_name,
        inputs={"time_series": ts_path, "mu": mu_path, "labels": labels_path},
        outdir=outdir,
        extra_args=["--dpi", "120", "--max-series", "4"],
    )

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0, "Tool should handle short sequences and NaNs without crashing."

    artifacts = list(outdir.glob("**/*.*"))
    # Not strict on extension here; just verify something was produced.
    assert any(p.suffix.lower() in {".png", ".svg", ".pdf"} for p in artifacts), \
        "Expected at least one plot artifact even for edge-case inputs."


@pytest.mark.integration
def test_module_invocation_if_importable(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Validate `python -m tools.<name>` execution path if a candidate tool module is importable.
    If only a shim exists, xfail gracefully with context.
    """
    module_name = _find_tool_module()
    if module_name is None:
        pytest.xfail("No importable tools.<name> FFT module found; acceptable if your tool is file-invoked only.")

    # Detect shim by reading file text if accessible (best-effort)
    # We'll just run it and expect success; if shim raises, we xfail.
    outdir = repo_tmp / "outputs" / "diagnostics" / "fft_compare_module_mode"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=None,
        module_name=module_name,
        inputs=tiny_inputs,
        outdir=outdir,
        extra_args=["--dpi", "120", "--figsize", "7,5"],
    )

    if proc.returncode != 0:
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "Shim placeholder" in combined:
            pytest.xfail("FFT tool shim detected — replace with real implementation to pass this test.")
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0

    # At least one artifact expected
    imgs = list(outdir.glob("**/*"))
    assert any(p.suffix.lower() in {".png", ".svg", ".pdf"} for p in imgs), \
        "Expected at least one image artifact in module mode."
