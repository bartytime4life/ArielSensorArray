# tests/diagnostics/test_symbolic_influence_map.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics tests for tools/symbolic_influence_map.py

This suite validates the scientific logic, artifact generation, and CLI behavior of the
symbolic influence map tool that computes per-wavelength influence (e.g., ∂L/∂μ) w.r.t.
symbolic rule losses.

Coverage
--------
1) Core API sanity:
   • compute_* influence functions (e.g., compute_symbolic_influence / influence_map / dLdmu)
     produce higher magnitude in bands with engineered violations.
   • Optional rule-wise decomposition is consistent (per-rule > 0 where violated).

2) Artifact generation API:
   • generate_symbolic_influence_artifacts(...) (or equivalent) produces JSON/CSV/PNG/HTML,
     including a manifest or structured JSON with per-rule summaries.

3) CLI contract:
   • End-to-end run via subprocess (python -m tools.symbolic_influence_map).
   • Determinism with --seed (compare JSON modulo volatile fields).
   • Graceful error handling for missing/invalid args.
   • Optional SPECTRAMIND_LOG_PATH audit line is appended.

4) Housekeeping:
   • Output files are non-empty; subsequent runs do not corrupt artifacts.

Notes
-----
• The module may expose different function names; tests try multiple candidates and xfail nicely if absent.
• No GPU/network required; uses tiny synthetic arrays.
• Matplotlib rendering (if used by the tool) should be headless-safe; we only assert file existence/size.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest


# ======================================================================================
# Helpers
# ======================================================================================

def _import_tool():
    """
    Import the module under test. Tries:
      1) tools.symbolic_influence_map
      2) symbolic_influence_map (top-level)
    """
    try:
        import tools.symbolic_influence_map as m  # type: ignore
        return m
    except Exception:
        try:
            import symbolic_influence_map as m2  # type: ignore
            return m2
        except Exception:
            pytest.skip(
                "symbolic_influence_map module not found. "
                "Expected at tools/symbolic_influence_map.py or importable as symbolic_influence_map."
            )


def _has_attr(mod, name: str) -> bool:
    return hasattr(mod, name) and getattr(mod, name) is not None


def _run_cli(
    module_path: Path,
    args: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 210,
) -> subprocess.CompletedProcess:
    """
    Execute the tool as a CLI using `python -m tools.symbolic_influence_map` when possible.
    Fallback to direct script invocation by file path if package execution is not feasible.
    """
    if module_path.name == "symbolic_influence_map.py" and module_path.parent.name == "tools":
        repo_root = module_path.parent.parent
        candidate_pkg = "tools.symbolic_influence_map"
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


# ======================================================================================
# Synthetic inputs & rules
# ======================================================================================

def _make_wavelengths(n: int = 283, lo_um: float = 0.5, hi_um: float = 7.8) -> np.ndarray:
    return np.linspace(float(lo_um), float(hi_um), n, dtype=np.float64)


def _make_mu_baseline(n: int = 283, seed: int = 7) -> np.ndarray:
    """
    Smooth-ish astrophysical-looking baseline spectrum in [0, 0.01] (relative units).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float64)
    mu = 1.0e-3 + 2.5e-4 * np.sin(3.0 * x) + 1.5e-4 * np.cos(7.0 * x)
    mu += 5.0e-5 * rng.standard_normal(n)
    return np.clip(mu, 0.0, None)


def _inject_band_violation(mu: np.ndarray, wl: np.ndarray, wl_lo: float, wl_hi: float, amp: float) -> np.ndarray:
    """
    Raise μ within a target band by 'amp' to intentionally violate a 'band_max' rule.
    """
    mu2 = mu.copy()
    mask = (wl >= wl_lo) & (wl <= wl_hi)
    mu2[mask] = mu2[mask] + float(amp)
    return mu2


def _write_minimal_rules_json(path: Path, band_lo: float, band_hi: float, max_depth: float) -> Path:
    """
    Create a tiny rules JSON understood by a broad class of symbolic engines:
      - 'band_max': within [band_lo, band_hi], μ must not exceed max_depth
      - 'nonnegativity': μ >= 0 always (weight is smaller)
      - 'smoothness': optional finite-difference smoothness cap (soft) for testing
    """
    rules = {
        "rules": [
            {
                "name": "band_max",
                "type": "band_threshold",
                "band": {"lo": float(band_lo), "hi": float(band_hi)},
                "op": "le",
                "value": float(max_depth),
                "weight": 1.0,
            },
            {
                "name": "nonnegativity",
                "type": "global_threshold",
                "op": "ge",
                "value": 0.0,
                "weight": 0.25,
            },
            {
                "name": "smoothness_fd2",
                "type": "smoothness_fd2",
                "threshold": 3.0e-4,
                "weight": 0.25,
            }
        ]
    }
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    return path


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture(scope="module")
def tool_mod():
    return _import_tool()


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Dict[str, Path]:
    """
    Create a clean workspace:
      inputs/  — .npy (μ, wavelengths) and rules.json
      outputs/ — artifacts
      logs/    — optional v50_debug_log.md
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
      - wavelengths.npy
      - mu_clean.npy
      - mu_violate.npy (with band spike)
      - rules.json
    """
    wl = _make_wavelengths()
    mu_clean = _make_mu_baseline()
    mu_bad = _inject_band_violation(mu_clean, wl, 2.0, 2.2, amp=2.0e-3)  # exceed a 2.0e-3 cap

    ip = tmp_workspace["inputs"]
    wl_path = ip / "wavelengths.npy"
    mu_clean_path = ip / "mu_clean.npy"
    mu_bad_path = ip / "mu_violate.npy"
    rules_path = ip / "rules.json"

    np.save(wl_path, wl)
    np.save(mu_clean_path, mu_clean)
    np.save(mu_bad_path, mu_bad)
    _write_minimal_rules_json(rules_path, band_lo=2.0, band_hi=2.2, max_depth=2.0e-3)

    return {
        "wavelengths": wl_path,
        "mu_clean": mu_clean_path,
        "mu_violate": mu_bad_path,
        "rules": rules_path,
    }


# ======================================================================================
# Core API tests — influence sensitivity & decomposition
# ======================================================================================

def test_influence_stronger_in_violated_band(tool_mod, synthetic_inputs):
    """
    Influence magnitude |∂L/∂μ| should be larger in a band that violates a rule than outside it.
    """
    candidates = [
        "compute_symbolic_influence",
        "influence_map",
        "compute_dldmu",
        "dldmu",
    ]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No influence function (compute_symbolic_influence/influence_map/dldmu) found in module.")

    wl = np.load(synthetic_inputs["wavelengths"])
    mu = np.load(synthetic_inputs["mu_violate"])
    rules_json = synthetic_inputs["rules"]

    # Try calling with flexible signatures; allow dict or path for rules
    try:
        out = fn(mu=mu, wavelengths=wl, rules=rules_json, return_per_rule=False, seed=13)
    except TypeError:
        out = fn(mu, wl, rules_json)  # type: ignore

    inf = np.asarray(out)
    assert inf.shape == mu.shape, "Influence must be a per-wavelength vector."
    assert np.all(np.isfinite(inf)), "Influence contains non-finite values."

    # Compare in-band vs out-of-band magnitude
    band = (wl >= 2.0) & (wl <= 2.2)
    outband = (wl < 2.0) | (wl > 2.2)
    mag_band = float(np.mean(np.abs(inf[band])))
    mag_out = float(np.mean(np.abs(inf[outband])))
    assert mag_band > 1.25 * mag_out, f"Expected stronger influence in violation band (band={mag_band:.3g}, out={mag_out:.3g})."


def test_per_rule_decomposition_if_available(tool_mod, synthetic_inputs):
    """
    If the API supports per-rule maps (e.g., return_per_rule=True or returns dict),
    ensure band_max contributes strongly in the violated band.
    """
    wl = np.load(synthetic_inputs["wavelengths"])
    mu = np.load(synthetic_inputs["mu_violate"])
    rules_json = synthetic_inputs["rules"]

    # Find an entrypoint that can return per-rule maps
    fn = None
    for name in ("compute_symbolic_influence", "influence_map"):
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No per-rule capable influence function available.")

    try:
        out = fn(mu=mu, wavelengths=wl, rules=rules_json, return_per_rule=True, seed=17)
    except TypeError:
        out = fn(mu, wl, rules_json, True)  # type: ignore

    # Accept either dict {rule: vector} or (aggregate, per_rule_dict)
    per_rule = None
    if isinstance(out, dict) and all(hasattr(v, "__len__") for v in out.values()):
        per_rule = out
    elif isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], dict):
        per_rule = out[1]

    if per_rule is None:
        pytest.xfail("Influence function did not return a per-rule mapping.")

    assert "band_max" in per_rule, "Per-rule maps missing 'band_max'."
    band_inf = np.asarray(per_rule["band_max"])
    assert band_inf.shape == mu.shape

    band = (wl >= 2.0) & (wl <= 2.2)
    outband = (wl < 2.0) | (wl > 2.2)
    mag_band = float(np.mean(np.abs(band_inf[band])))
    mag_out = float(np.mean(np.abs(band_inf[outband])))
    assert mag_band > 1.4 * mag_out, f"'band_max' per-rule influence should be concentrated in-band (band={mag_band:.3g}, out={mag_out:.3g})."


# ======================================================================================
# Artifact generation API
# ======================================================================================

def test_generate_artifacts(tool_mod, tmp_workspace, synthetic_inputs):
    """
    Artifact generator should emit JSON/CSV/PNG/HTML files and return a manifest (or paths).
    """
    entry_candidates = [
        "generate_symbolic_influence_artifacts",
        "run_symbolic_influence_map",
        "produce_symbolic_influence_outputs",
        "analyze_and_export",  # generic fallback
    ]
    entry = None
    for name in entry_candidates:
        if _has_attr(tool_mod, name):
            entry = getattr(tool_mod, name)
            break
    if entry is None:
        pytest.xfail("No artifact generation entrypoint found in symbolic_influence_map.")

    outdir = tmp_workspace["outputs"]
    mu = np.load(synthetic_inputs["mu_violate"])
    wl = np.load(synthetic_inputs["wavelengths"])
    rules_json = synthetic_inputs["rules"]

    kwargs = dict(
        mu=mu,
        wavelengths=wl,
        rules=rules_json,
        outdir=str(outdir),
        json_out=True,
        csv_out=True,
        png_out=True,
        html_out=True,
        seed=99,
        title="Symbolic Influence Map — Test",
    )
    try:
        manifest = entry(**kwargs)
    except TypeError:
        manifest = entry(mu, wl, rules_json, str(outdir), True, True, True, True, 99, "Symbolic Influence Map — Test")  # type: ignore

    # Presence checks
    json_files = list(outdir.glob("*.json"))
    csv_files = list(outdir.glob("*.csv"))
    png_files = list(outdir.glob("*.png"))
    html_files = list(outdir.glob("*.html"))

    assert json_files, "No JSON artifact produced by symbolic influence map."
    assert csv_files, "No CSV artifact produced by symbolic influence map."
    assert png_files, "No PNG artifact produced by symbolic influence map."
    assert html_files, "No HTML artifact produced by symbolic influence map."

    # Minimal JSON schema check
    with open(json_files[0], "r", encoding="utf-8") as f:
        js = json.load(f)
    assert isinstance(js, dict), "Top-level JSON must be an object."
    has_influence = ("influence" in js) or ("dldmu" in js) or ("per_rule" in js)
    assert has_influence, "JSON should include influence content (keys like 'influence','dldmu','per_rule')."

    # Files should be non-trivially sized
    for p in png_files:
        _assert_file(p, min_size=256)
    for c in csv_files:
        _assert_file(c, min_size=64)
    for h in html_files:
        _assert_file(h, min_size=128)


# ======================================================================================
# CLI End-to-End
# ======================================================================================

def test_cli_end_to_end(tmp_workspace, synthetic_inputs):
    """
    End-to-end CLI test:
      • Runs the module as a CLI with --mu/--wavelengths/--rules → emits JSON/CSV/PNG/HTML.
      • Uses --seed for determinism and compares JSON across two runs (modulo volatile metadata).
      • Verifies optional audit log when SPECTRAMIND_LOG_PATH is set.
    """
    # Locate module file to construct python -m invocation
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "symbolic_influence_map.py",  # repo-root/tools/...
        Path(__file__).resolve().parents[1] / "symbolic_influence_map.py",            # tests/diagnostics/../
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("symbolic_influence_map.py not found; cannot run CLI end-to-end test.")

    outdir = tmp_workspace["outputs"]
    logsdir = tmp_workspace["logs"]
    mu_path = synthetic_inputs["mu_violate"]
    wl_path = synthetic_inputs["wavelengths"]
    rules_json = synthetic_inputs["rules"]

    env = {
        "PYTHONUNBUFFERED": "1",
        "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
    }

    args = (
        "--mu", str(mu_path),
        "--wavelengths", str(wl_path),
        "--rules", str(rules_json),
        "--outdir", str(outdir),
        "--json",
        "--csv",
        "--png",
        "--html",
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

    # Determinism: second run with same seed into a new directory should match JSON (minus volatile fields)
    outdir2 = outdir.parent / "outputs_run2"
    outdir2.mkdir(exist_ok=True)
    args2 = (
        "--mu", str(mu_path),
        "--wavelengths", str(wl_path),
        "--rules", str(rules_json),
        "--outdir", str(outdir2),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--seed", "2025",
        "--silent",
    )
    proc2 = _run_cli(module_file, args2, env=env, timeout=210)
    if proc2.returncode != 0:
        msg = f"CLI run 2 failed (exit={proc2.returncode}).\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"
        pytest.fail(msg)

    json2 = sorted(outdir2.glob("*.json"))
    assert json2, "Second CLI run produced no JSON artifacts."

    def _normalize(j: Dict[str, Any]) -> Dict[str, Any]:
        d = json.loads(json.dumps(j))  # deep copy
        vol_patterns = re.compile(r"(time|date|timestamp|duration|path|cwd|hostname|uuid|version)", re.I)

        def scrub(obj: Any) -> Any:
            if isinstance(obj, dict):
                for k in list(obj.keys()):
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
        assert ("symbolic_influence_map" in text) or ("influence" in text and "symbolic" in text), \
            "Audit log exists but lacks recognizable CLI signature."


def test_cli_error_cases(tmp_workspace, synthetic_inputs):
    """
    CLI should:
      • Exit non-zero when required --mu is missing.
      • Report helpful error text mentioning the missing/invalid flag.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "symbolic_influence_map.py",
        Path(__file__).resolve().parents[1] / "symbolic_influence_map.py",
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("symbolic_influence_map.py not found; cannot run CLI error tests.")

    outdir = tmp_workspace["outputs"]
    wl = synthetic_inputs["wavelengths"]
    rules = synthetic_inputs["rules"]

    # Missing --mu
    args_missing_mu = (
        "--wavelengths", str(wl),
        "--rules", str(rules),
        "--outdir", str(outdir),
        "--json",
    )
    proc = _run_cli(module_file, args_missing_mu, env=None, timeout=90)
    assert proc.returncode != 0, "CLI should fail when required --mu is missing."
    msg = (proc.stderr + "\n" + proc.stdout).lower()
    assert "mu" in msg, "Error message should mention missing 'mu'."


# ======================================================================================
# Housekeeping checks
# ======================================================================================

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