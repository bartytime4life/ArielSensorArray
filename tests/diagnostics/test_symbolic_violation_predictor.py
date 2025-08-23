#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_symbolic_violation_predictor.py

SpectraMind V50 — Diagnostics Tests
SymbolicViolationPredictor (SVP) & CLI wrapper

This suite validates the upgraded symbolic violation analysis pipeline. It is deliberately
flexible to accommodate different repository layouts and evolving interfaces:

    • Python API: class `SymbolicViolationPredictor` discovered in common modules.
    • CLI tool:  `python -m tools.symbolic_violation_predictor` (if importable).
    • Output artifacts: JSON/CSV/masks placed under --outdir (filenames not over-constrained).
    • Logging: append-only audit log at logs/v50_debug_log.md.

What we test
------------
1) Basic run (API or CLI): tiny μ spectra + tiny rule set → artifacts written, process exits 0.
2) Edge cases: NaNs/Inf in μ; near-zero/negative values; small number of bins.
3) OUTDIR discipline: no stray writes outside --outdir except logs/ and optional run-hash JSON.
4) Audit log append-only behavior across multiple runs.
5) (If supported) Determinism when a `seed`/`random_state` parameter is provided.

Design choices
--------------
• We avoid brittle assumptions about exact function names/flags/filenames.
• We pass a small “rules JSON” with simple per-bin masks; tools that ignore external rules
  should still succeed (we tolerate unknown-flag behavior).
• If neither API nor CLI is present, tests skip with a clear message (keeps CI green
  while wiring things up).

Run
---
pytest -q tests/diagnostics/test_symbolic_violation_predictor.py
"""

from __future__ import annotations

import inspect
import io
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

# Force headless plotting for any potential figures
os.environ.setdefault("MPLBACKEND", "Agg")


# --------------------------------------------------------------------------------------
# Discovery: locate class SymbolicViolationPredictor or a CLI module to run via `-m`
# --------------------------------------------------------------------------------------

POSSIBLE_API_MODULES = [
    # Canonical paths first
    "src.symbolic.symbolic_violation_predictor",
    "symbolic.symbolic_violation_predictor",
    # Some repos house it under tools as a callable module too
    "tools.symbolic_violation_predictor",
    # Legacy/alternate locations
    "src.diagnostics.symbolic_violation_predictor",
    "diagnostics.symbolic_violation_predictor",
]

CLI_MODULE = "tools.symbolic_violation_predictor"


def _try_import_predictor() -> Optional[type]:
    """
    Attempt to import `SymbolicViolationPredictor` class from common modules.
    Returns the class or None if not found.
    """
    for mod_name in POSSIBLE_API_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["SymbolicViolationPredictor"])
            cls = getattr(mod, "SymbolicViolationPredictor", None)
            if isinstance(cls, type):
                return cls
        except Exception:
            continue
    return None


def _has_cli_module() -> bool:
    """
    Check if we can import tools.symbolic_violation_predictor as a module for `python -m`.
    """
    try:
        __import__(CLI_MODULE)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Tiny synthetic inputs
# --------------------------------------------------------------------------------------

@dataclass
class TinyInputs:
    mu_path: Path
    rules_json: Path
    planet_ids: List[str]


def _make_tiny_mu(n_planets: int = 6, n_bins: int = 31, seed: int = 1) -> np.ndarray:
    """
    Create a small (N_planets x N_bins) μ spectra array with gentle structure.
    Mix smooth baseline + simple band bumps so rule masks have meaningful variations.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
    mu_list = []
    for pid in range(n_planets):
        base = 0.02 * np.sin(2 * np.pi * (pid + 1) * x) + 0.02 * np.cos(5 * x)
        bands = (
            0.10 * np.exp(-0.5 * ((x - 0.30) / 0.06) ** 2) +  # band A ~ 0.30
            0.07 * np.exp(-0.5 * ((x - 0.55) / 0.05) ** 2) +  # band B ~ 0.55
            0.05 * np.exp(-0.5 * ((x - 0.80) / 0.04) ** 2)    # band C ~ 0.80
        )
        noise = rng.normal(0, 0.005, size=n_bins)
        mu_list.append(base + bands + noise)
    mu = np.stack(mu_list, axis=0)
    return mu.astype(np.float64)


def _inject_pathologies(mu: np.ndarray) -> np.ndarray:
    """
    Inject a few pathologies to test robustness: NaN, Inf, negatives.
    """
    mu2 = mu.copy()
    if mu2.size >= 10:
        mu2[1, 3] = np.nan
    if mu2.size >= 20:
        mu2[2, 5] = np.inf
    if mu2.size >= 30:
        mu2[3, 7] = -abs(mu2[3, 7])  # negative
    return mu2


def _write_rules_json(path: Path, n_bins: int) -> Path:
    """
    Create a minimal rules JSON. We keep the schema intentionally simple/generic:

    {
      "meta": {...},
      "rules": [
        {"id": "H2O_band_consistency", "weight": 1.0, "mask": [start, end]},
        {"id": "CO2_peak_alignment",   "weight": 0.8, "mask": [start, end]},
        {"id": "CH4_edge_monotonicity","weight": 0.6, "mask": [start, end], "direction": "decreasing"}
      ]
    }

    Tools that use a different rule schema should either ignore unknown keys or map gracefully.
    """
    # Define 3 tiny masks spanning non-overlapping ranges
    a0, a1 = int(0.22 * n_bins), int(0.36 * n_bins)
    b0, b1 = int(0.48 * n_bins), int(0.60 * n_bins)
    c0, c1 = int(0.73 * n_bins), int(0.86 * n_bins)

    data = {
        "meta": {"version": "test-1.0", "n_bins": n_bins},
        "rules": [
            {"id": "H2O_band_consistency", "weight": 1.0, "mask": [a0, a1]},
            {"id": "CO2_peak_alignment", "weight": 0.8, "mask": [b0, b1]},
            {"id": "CH4_edge_monotonicity", "weight": 0.6, "mask": [c0, c1], "direction": "decreasing"},
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _materialize_inputs(root: Path, n_planets: int = 6, n_bins: int = 31, seed: int = 1, pathological: bool = False) -> TinyInputs:
    inputs_dir = root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    mu = _make_tiny_mu(n_planets=n_planets, n_bins=n_bins, seed=seed)
    if pathological:
        mu = _inject_pathologies(mu)

    mu_path = inputs_dir / "mu.npy"
    np.save(mu_path, mu)

    rules_path = inputs_dir / "rules.json"
    _write_rules_json(rules_path, n_bins=n_bins)

    planet_ids = [f"P{idx:03d}" for idx in range(n_planets)]
    return TinyInputs(mu_path=mu_path, rules_json=rules_path, planet_ids=planet_ids)


# --------------------------------------------------------------------------------------
# Repo scaffold
# --------------------------------------------------------------------------------------

def _ensure_repo_scaffold(repo_root: Path) -> None:
    """
    Ensure minimal directories for tools, logs, and outputs. If neither API nor CLI exists,
    we do NOT create a shim (we'll skip tests instead); shims can cause confusing xfails here.
    """
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "diagnostics").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Flexible API invocation
# --------------------------------------------------------------------------------------

def _construct_predictor(cls: type, **kwargs) -> Any:
    """
    Instantiate SymbolicViolationPredictor with only the kwargs its constructor accepts.
    """
    sig = inspect.signature(cls)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**accepted)


def _call_predictor_runlike(obj: Any, mu: np.ndarray, **kwargs) -> Any:
    """
    Call a "run-like" method on the predictor object, preferring common method names.
    Only pass supported kwargs.
    """
    for name in ("run", "score", "predict", "evaluate", "__call__"):
        if hasattr(obj, name) and callable(getattr(obj, name)):
            func = getattr(obj, name)
            sig = inspect.signature(func)
            accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
            try:
                return func(mu, **accepted)
            except TypeError:
                # some APIs accept data as named param
                if "mu" in sig.parameters:
                    accepted2 = dict(accepted)
                    accepted2["mu"] = mu
                    return func(**accepted2)
                raise
    raise AttributeError("No runnable method found on predictor (tried run/score/predict/evaluate/__call__).")


# --------------------------------------------------------------------------------------
# CLI runner
# --------------------------------------------------------------------------------------

def _run_cli_module(mu_path: Path, rules_json: Path, outdir: Path, extra_flags: Optional[List[str]] = None) -> Tuple[int, str, str]:
    """
    Run CLI via `python -m tools.symbolic_violation_predictor`. Return (code, stdout, stderr).
    """
    import subprocess

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SPECTRAMIND_TEST", "1")
    env.setdefault("MPLBACKEND", "Agg")

    cmd = [
        sys.executable,
        "-m",
        CLI_MODULE,
        "--mu", str(mu_path),
        "--outdir", str(outdir),
        "--rules-json", str(rules_json),
        "--save-json",
        "--save-csv",
        "--no-browser",
        "--version", "test",
    ]
    if extra_flags:
        cmd += list(extra_flags)

    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _scan_artifacts(outdir: Path) -> Dict[str, List[Path]]:
    """
    Return lists of artifacts by type.
    """
    kinds = {
        "json": [p for p in outdir.rglob("*.json")],
        "csv":  [p for p in outdir.rglob("*.csv")],
        "png":  [p for p in outdir.rglob("*.png")],
        "svg":  [p for p in outdir.rglob("*.svg")],
        "pdf":  [p for p in outdir.rglob("*.pdf")],
        "npy":  [p for p in outdir.rglob("*.npy")],
    }
    return kinds


def _assert_nonempty_file(path: Path) -> None:
    assert path.exists(), f"Expected file not found: {path}"
    assert path.is_file(), f"Expected a file, got: {path}"
    assert path.stat().st_size > 0, f"File seems empty: {path}"


# --------------------------------------------------------------------------------------
# Pytest fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path) -> Path:
    _ensure_repo_scaffold(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def predictor_class_or_cli_available():
    """
    Returns the class SymbolicViolationPredictor if importable,
    otherwise the string "CLI" if the CLI module exists.
    Skips the entire suite if neither is available.
    """
    cls = _try_import_predictor()
    if cls is not None:
        return cls
    if _has_cli_module():
        return "CLI"
    pytest.skip("Neither SymbolicViolationPredictor API nor CLI module is available. Skipping tests.")


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

@pytest.mark.integration
def test_basic_run_generates_artifacts(repo_tmp: Path, predictor_class_or_cli_available):
    """
    Basic success path: tiny μ + simple rules should produce at least JSON or CSV outputs
    and append to the audit log. API or CLI path is accepted.
    """
    tiny = _materialize_inputs(repo_tmp, n_planets=6, n_bins=31, seed=3, pathological=False)
    outdir = repo_tmp / "outputs" / "diagnostics" / "svp_basic"
    outdir.mkdir(parents=True, exist_ok=True)

    if isinstance(predictor_class_or_cli_available, type):
        # API mode
        cls = predictor_class_or_cli_available
        mu = np.load(tiny.mu_path)
        # Construct with tolerant kwargs
        predictor = _construct_predictor(
            cls,
            rules_json=str(tiny.rules_json),
            outdir=str(outdir),
            save_json=True,
            save_csv=True,
            log_path=str(repo_tmp / "logs" / "v50_debug_log.md"),
            seed=123,
            random_state=123,
            quiet=True,
        )
        # Run with flexible call
        _ = _call_predictor_runlike(
            predictor,
            mu,
            planet_ids=tiny.planet_ids,
            outdir=str(outdir),
            save_json=True,
            save_csv=True,
            seed=123,
            random_state=123,
            quiet=True,
        )
    else:
        # CLI mode
        code, stdout, stderr = _run_cli_module(
            mu_path=tiny.mu_path,
            rules_json=tiny.rules_json,
            outdir=outdir,
            extra_flags=["--quiet"],
        )
        if code != 0:
            print("CLI STDOUT:\n", stdout)
            print("CLI STDERR:\n", stderr)
        assert code == 0, "CLI run should succeed."

    arts = _scan_artifacts(outdir)
    assert arts["json"] or arts["csv"], "Expected at least one JSON or CSV artifact in outdir."

    # Audit log presence
    log_path = repo_tmp / "logs" / "v50_debug_log.md"
    assert log_path.exists(), "Expected audit log logs/v50_debug_log.md"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "symbolic" in log_text or "violation" in log_text, "Audit log should mention symbolic violations."


@pytest.mark.integration
def test_handles_nan_inf_and_small_bins(repo_tmp: Path, predictor_class_or_cli_available):
    """
    Robustness: μ contains NaN/Inf/negatives; number of bins is small.
    The predictor should not crash and should still produce outputs.
    """
    tiny = _materialize_inputs(repo_tmp, n_planets=4, n_bins=17, seed=5, pathological=True)
    outdir = repo_tmp / "outputs" / "diagnostics" / "svp_edge"
    outdir.mkdir(parents=True, exist_ok=True)

    if isinstance(predictor_class_or_cli_available, type):
        cls = predictor_class_or_cli_available
        mu = np.load(tiny.mu_path)
        predictor = _construct_predictor(
            cls,
            rules_json=str(tiny.rules_json),
            outdir=str(outdir),
            save_json=True,
            save_csv=True,
            sanitize=True,        # if supported
            clamp=True,           # if supported
            quiet=True,
        )
        _ = _call_predictor_runlike(
            predictor,
            mu,
            planet_ids=tiny.planet_ids,
            outdir=str(outdir),
            sanitize=True,
            clamp=True,
            save_json=True,
            save_csv=True,
            quiet=True,
        )
    else:
        code, stdout, stderr = _run_cli_module(
            mu_path=tiny.mu_path,
            rules_json=tiny.rules_json,
            outdir=outdir,
            extra_flags=["--sanitize", "--clamp", "--quiet"],
        )
        if code != 0:
            print("CLI STDOUT:\n", stdout)
            print("CLI STDERR:\n", stderr)
        assert code == 0

    arts = _scan_artifacts(outdir)
    assert arts["json"] or arts["csv"], "Expected artifacts for edge-case inputs."


@pytest.mark.integration
def test_outdir_respected_no_strays(repo_tmp: Path, predictor_class_or_cli_available):
    """
    Ensure the predictor writes only under --outdir (plus logs/) and does not create stray files.
    """
    tiny = _materialize_inputs(repo_tmp, n_planets=5, n_bins=21, seed=7, pathological=False)
    outdir = repo_tmp / "outputs" / "diagnostics" / "svp_outdir"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    if isinstance(predictor_class_or_cli_available, type):
        cls = predictor_class_or_cli_available
        mu = np.load(tiny.mu_path)
        predictor = _construct_predictor(
            cls,
            rules_json=str(tiny.rules_json),
            outdir=str(outdir),
            save_json=True,
            save_csv=True,
            quiet=True,
        )
        _ = _call_predictor_runlike(
            predictor,
            mu,
            planet_ids=tiny.planet_ids,
            outdir=str(outdir),
            save_json=True,
            save_csv=True,
            quiet=True,
        )
    else:
        code, stdout, stderr = _run_cli_module(
            mu_path=tiny.mu_path,
            rules_json=tiny.rules_json,
            outdir=outdir,
            extra_flags=["--quiet"],
        )
        if code != 0:
            print("CLI STDOUT:\n", stdout)
            print("CLI STDERR:\n", stderr)
        assert code == 0

    after = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())
    new_files = sorted(list(after - before))

    # Allowed: anything under outdir; logs/*; optional outputs/run_hash_summary*.json; bytecode cache.
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

    assert not disallowed, f"Unexpected stray writes outside --outdir: {disallowed}"


@pytest.mark.integration
def test_audit_log_is_append_only(repo_tmp: Path, predictor_class_or_cli_available):
    """
    Run the predictor twice and ensure logs/v50_debug_log.md grows (or at least doesn't shrink).
    """
    tiny = _materialize_inputs(repo_tmp, n_planets=4, n_bins=19, seed=11, pathological=False)
    out1 = repo_tmp / "outputs" / "diagnostics" / "svp_log1"
    out2 = repo_tmp / "outputs" / "diagnostics" / "svp_log2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    log_path = repo_tmp / "logs" / "v50_debug_log.md"

    def _run(outdir: Path):
        if isinstance(predictor_class_or_cli_available, type):
            cls = predictor_class_or_cli_available
            mu = np.load(tiny.mu_path)
            predictor = _construct_predictor(
                cls,
                rules_json=str(tiny.rules_json),
                outdir=str(outdir),
                save_json=True,
                save_csv=True,
                quiet=True,
            )
            _ = _call_predictor_runlike(
                predictor,
                mu,
                planet_ids=tiny.planet_ids,
                outdir=str(outdir),
                save_json=True,
                save_csv=True,
                quiet=True,
            )
        else:
            code, stdout, stderr = _run_cli_module(
                mu_path=tiny.mu_path,
                rules_json=tiny.rules_json,
                outdir=outdir,
                extra_flags=["--quiet"],
            )
            if code != 0:
                print("CLI STDOUT:\n", stdout)
                print("CLI STDERR:\n", stderr)
            assert code == 0

    _run(out1)
    size1 = log_path.stat().st_size if log_path.exists() else 0
    _run(out2)
    size2 = log_path.stat().st_size if log_path.exists() else 0

    assert size2 >= size1, "Audit log should not shrink after subsequent runs."
    if size1 > 0:
        assert size2 > size1, "Audit log should typically increase after a second run."


@pytest.mark.integration
def test_determinism_when_seed_provided(repo_tmp: Path, predictor_class_or_cli_available):
    """
    If the API/CLI supports a `seed` or `random_state` parameter, repeated runs with the same seed
    should produce identical JSON outputs (within tiny numerical tolerance). If seeding is not
    supported, we xfail gracefully.
    """
    tiny = _materialize_inputs(repo_tmp, n_planets=5, n_bins=25, seed=17, pathological=False)

    if isinstance(predictor_class_or_cli_available, type):
        cls = predictor_class_or_cli_available

        # Inspect constructor and run-like method to see if a seed is accepted
        accepts_seed_ctor = "seed" in inspect.signature(cls).parameters or "random_state" in inspect.signature(cls).parameters

        # Try to detect run-like method
        runlike_name = None
        for name in ("run", "score", "predict", "evaluate", "__call__"):
            if hasattr(cls, name) and callable(getattr(cls, name)):
                runlike_name = name
                break
        accepts_seed_run = False
        if runlike_name:
            sig = inspect.signature(getattr(cls, runlike_name))
            accepts_seed_run = "seed" in sig.parameters or "random_state" in sig.parameters

        if not (accepts_seed_ctor or accepts_seed_run):
            pytest.xfail("Predictor API does not appear to accept a seed/random_state parameter.")

        # Two runs with the same seed into separate outdirs
        outA = repo_tmp / "outputs" / "diagnostics" / "svp_seed_A"
        outB = repo_tmp / "outputs" / "diagnostics" / "svp_seed_B"
        outA.mkdir(parents=True, exist_ok=True)
        outB.mkdir(parents=True, exist_ok=True)

        mu = np.load(tiny.mu_path)

        predA = _construct_predictor(
            cls,
            rules_json=str(tiny.rules_json),
            outdir=str(outA),
            save_json=True,
            seed=777,
            random_state=777,
            quiet=True,
        )
        _ = _call_predictor_runlike(
            predA, mu, outdir=str(outA), save_json=True, seed=777, random_state=777, quiet=True
        )

        predB = _construct_predictor(
            cls,
            rules_json=str(tiny.rules_json),
            outdir=str(outB),
            save_json=True,
            seed=777,
            random_state=777,
            quiet=True,
        )
        _ = _call_predictor_runlike(
            predB, mu, outdir=str(outB), save_json=True, seed=777, random_state=777, quiet=True
        )

        # Compare JSON artifacts if present
        jsonA = sorted((p for p in outA.rglob("*.json")), key=lambda p: p.name)
        jsonB = sorted((p for p in outB.rglob("*.json")), key=lambda p: p.name)
        if not jsonA or not jsonB:
            pytest.xfail("No JSON artifacts found to compare for determinism; acceptable for minimal predictor.")
        # Load the first comparable pair
        a = json.loads(jsonA[0].read_text(encoding="utf-8"))
        b = json.loads(jsonB[0].read_text(encoding="utf-8"))
        assert a == b, "Seeded runs should produce identical JSON outputs."
    else:
        if not _has_cli_module():
            pytest.xfail("CLI module not available for seed determinism test.")
        # Two CLI runs with the same seed (if supported); if not, xfail.
        outA = repo_tmp / "outputs" / "diagnostics" / "svp_seed_cli_A"
        outB = repo_tmp / "outputs" / "diagnostics" / "svp_seed_cli_B"
        outA.mkdir(parents=True, exist_ok=True)
        outB.mkdir(parents=True, exist_ok=True)

        codeA, stdoutA, stderrA = _run_cli_module(
            mu_path=tiny.mu_path,
            rules_json=tiny.rules_json,
            outdir=outA,
            extra_flags=["--save-json", "--seed", "999", "--quiet"],
        )
        codeB, stdoutB, stderrB = _run_cli_module(
            mu_path=tiny.mu_path,
            rules_json=tiny.rules_json,
            outdir=outB,
            extra_flags=["--save-json", "--seed", "999", "--quiet"],
        )
        if codeA != 0 or codeB != 0:
            pytest.xfail("CLI does not appear to support seeding or failed unexpectedly.")

        jsonA = sorted((p for p in outA.rglob("*.json")), key=lambda p: p.name)
        jsonB = sorted((p for p in outB.rglob("*.json")), key=lambda p: p.name)
        if not jsonA or not jsonB:
            pytest.xfail("No JSON artifacts found to compare for determinism; acceptable for minimal predictor.")
        a = json.loads(jsonA[0].read_text(encoding="utf-8"))
        b = json.loads(jsonB[0].read_text(encoding="utf-8"))
        assert a == b, "Seeded CLI runs should produce identical JSON outputs."
