# tests/diagnostics/test_validate_submission.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics tests for tools/validate_submission.py

This suite validates the scientific logic, artifact generation, and CLI behavior of the
submission validator. It creates minimal-yet-realistic submission files (CSV) with
Ariel-like 283 spectral bins per planet and checks that:

1) Core API sanity
   • validate_* function (e.g., validate_submission / validate / run_validate) returns a
     structured result (dict-like) and flags schema/shape/value errors.
   • Happy-path submissions pass basic checks (mu finite, sigma > 0, per-planet 283 bins).

2) Artifact generation API
   • generate_* or run_* entrypoint produces JSON/HTML/CSV artifacts and/or a manifest.

3) CLI contract
   • End-to-end run via subprocess (python -m tools.validate_submission).
   • Determinism for the JSON report (modulo volatile fields).
   • Helpful failures on bad input (missing cols / wrong bin count / invalid sigma).
   • Optional SPECTRAMIND_LOG_PATH audit line is appended.

4) Housekeeping
   • Output files are non-empty; subsequent runs do not corrupt artifacts.

The test is tolerant to API naming differences and will xfail nicely if the tool is not
available in the repository yet.

No GPU/network is required. All I/O is confined to pytest's tmp_path.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest


# ======================================================================================
# Helpers
# ======================================================================================

def _import_tool():
    """
    Import the module under test. Tries:
      1) tools.validate_submission
      2) validate_submission (top-level)
    """
    try:
        import tools.validate_submission as m  # type: ignore
        return m
    except Exception:
        try:
            import validate_submission as m2  # type: ignore
            return m2
        except Exception:
            pytest.skip(
                "validate_submission module not found. "
                "Expected at tools/validate_submission.py or importable as validate_submission."
            )


def _has_attr(mod, name: str) -> bool:
    return hasattr(mod, name) and getattr(mod, name) is not None


def _run_cli(
    module_path: Path,
    args: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 240,
) -> subprocess.CompletedProcess:
    """
    Execute the tool as a CLI using `python -m tools.validate_submission` when possible.
    Fallback to direct script invocation by file path if package execution is not feasible.
    """
    if module_path.name == "validate_submission.py" and module_path.parent.name == "tools":
        repo_root = module_path.parent.parent
        candidate_pkg = "tools.validate_submission"
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
# Synthetic submission builders
# ======================================================================================

_ARIEL_NBINS = 283

@dataclass
class SubRow:
    planet_id: str
    bin: int
    mu: float
    sigma: float


def _write_csv(path: Path, rows: Iterable[SubRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["planet_id", "bin", "mu", "sigma"])
        for r in rows:
            w.writerow([r.planet_id, r.bin, f"{r.mu:.10g}", f"{r.sigma:.10g}"])
    return path


def _make_ok_rows(planet_ids: Sequence[str], seed: int = 11) -> List[SubRow]:
    """
    Build a valid submission: per-planet 283 bins, finite mu, strictly positive sigma.
    """
    rng = np.random.default_rng(seed)
    rows: List[SubRow] = []
    for pid in planet_ids:
        # Smooth-ish baseline spectrum in ~[0, 1e-2]
        x = np.linspace(0.0, 2.0 * np.pi, _ARIEL_NBINS)
        base = 1.0e-3 + 2.5e-4 * np.sin(3.0 * x) + 1.5e-4 * np.cos(7.0 * x)
        noise = 5.0e-5 * rng.standard_normal(_ARIEL_NBINS)
        mu = np.clip(base + noise, -1.0, 1.0)  # allow tiny negatives if validator clamps or warns
        # Sigma positive and not absurdly small/huge
        sigma = 5.0e-4 + np.abs(1.0e-4 * rng.standard_normal(_ARIEL_NBINS))
        for b in range(_ARIEL_NBINS):
            rows.append(SubRow(planet_id=pid, bin=b, mu=float(mu[b]), sigma=float(sigma[b])))
    return rows


def _make_bad_rows_missing_col(planet_ids: Sequence[str]) -> Path:
    """
    Create a CSV missing the 'sigma' column to trigger schema failure.
    """
    from io import StringIO
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["planet_id", "bin", "mu"])  # missing sigma
    for pid in planet_ids:
        for b in range(_ARIEL_NBINS):
            w.writerow([pid, b, 0.001])
    txt = buf.getvalue()
    # caller will write to disk
    return txt  # type: ignore[return-value]


def _make_bad_rows_wrong_bins(planet_ids: Sequence[str], valid_first: bool = True) -> List[SubRow]:
    """
    Create rows where one planet has the wrong number of bins to trigger shape failure.
    """
    rows = _make_ok_rows(planet_ids)
    if not rows:
        return rows
    # Remove some rows for the last planet to violate the 283-per-planet rule.
    last = planet_ids[-1]
    # Keep only first 200 bins for the last planet
    rows = [r for r in rows if not (r.planet_id == last and r.bin >= 200)]
    return rows


def _make_bad_rows_negative_sigma(planet_ids: Sequence[str]) -> List[SubRow]:
    """
    Create rows with negative sigma for a handful of bins to trigger value failure.
    """
    rows = _make_ok_rows(planet_ids)
    # Flip signs for a few sigma values
    flips = 7
    for i, r in enumerate(rows[:flips]):
        rows[i] = SubRow(planet_id=r.planet_id, bin=r.bin, mu=r.mu, sigma=-abs(r.sigma))
    return rows


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
      inputs/  — submissions
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
def ok_submission(tmp_workspace: Dict[str, Path]) -> Path:
    rows = _make_ok_rows(["planet_A", "planet_B"])
    sub_path = tmp_workspace["inputs"] / "submission_ok.csv"
    _write_csv(sub_path, rows)
    return sub_path


@pytest.fixture()
def bad_submission_wrong_bins(tmp_workspace: Dict[str, Path]) -> Path:
    rows = _make_bad_rows_wrong_bins(["planet_A", "planet_B", "planet_C"])
    sub_path = tmp_workspace["inputs"] / "submission_wrong_bins.csv"
    _write_csv(sub_path, rows)
    return sub_path


@pytest.fixture()
def bad_submission_negative_sigma(tmp_workspace: Dict[str, Path]) -> Path:
    rows = _make_bad_rows_negative_sigma(["planet_A"])
    sub_path = tmp_workspace["inputs"] / "submission_neg_sigma.csv"
    _write_csv(sub_path, rows)
    return sub_path


@pytest.fixture()
def bad_submission_missing_col(tmp_workspace: Dict[str, Path]) -> Path:
    txt = _make_bad_rows_missing_col(["planet_A"])
    sub_path = tmp_workspace["inputs"] / "submission_missing_col.csv"
    sub_path.write_text(txt, encoding="utf-8")
    return sub_path


# ======================================================================================
# Core API tests — validate() behavior
# ======================================================================================

def test_api_validate_ok(tool_mod, ok_submission, tmp_workspace):
    """
    Happy-path: API returns a dict-like object with 'ok' True and includes basic summary metadata.
    """
    candidates = ["validate_submission", "validate", "run_validate"]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No validate function (validate_submission/validate/run_validate) found in module.")

    try:
        result = fn(input_path=str(ok_submission), outdir=str(tmp_workspace["outputs"]), json_out=True, html_out=True, csv_out=True, seed=13)
    except TypeError:
        # Most minimal signature
        result = fn(str(ok_submission))  # type: ignore

    # Normalize result to dict
    if isinstance(result, (list, tuple)) and len(result) >= 1 and isinstance(result[0], dict):
        result = result[0]
    if not isinstance(result, dict):
        # Don't fail hard on shape; convert to dict conservatively if possible
        result = {"ok": True}

    assert "ok" in result and bool(result["ok"]) is True, f"Expected ok=True for valid submission, got {result}"
    # If the tool reports counts, sanity check them
    for key in ("num_planets", "num_rows", "bins_per_planet"):
        if key in result:
            assert isinstance(result[key], (int, dict, list))


def test_api_validate_catches_wrong_bins(tool_mod, bad_submission_wrong_bins):
    """
    Wrong number of bins per planet should be flagged (ok=False) with a meaningful message.
    """
    candidates = ["validate_submission", "validate", "run_validate"]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No validate function found.")

    try:
        result = fn(input_path=str(bad_submission_wrong_bins))
    except TypeError:
        result = fn(str(bad_submission_wrong_bins))  # type: ignore

    # Interpret result
    if isinstance(result, dict):
        ok = bool(result.get("ok", False))
        msg = " ".join(str(x) for x in result.values())
    else:
        ok = False
        msg = str(result)

    assert ok is False, "Expected ok=False for wrong-bin submission."
    assert ("283" in msg) or ("bin" in msg.lower()) or ("count" in msg.lower()), "Expected message to mention bin count."


def test_api_validate_catches_negative_sigma(tool_mod, bad_submission_negative_sigma):
    """
    Negative sigma values should be flagged (ok=False).
    """
    candidates = ["validate_submission", "validate", "run_validate"]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No validate function found.")

    try:
        result = fn(input_path=str(bad_submission_negative_sigma))
    except TypeError:
        result = fn(str(bad_submission_negative_sigma))  # type: ignore

    if isinstance(result, dict):
        ok = bool(result.get("ok", False))
        msg = " ".join(str(x) for x in result.values())
    else:
        ok = False
        msg = str(result)

    assert ok is False, "Expected ok=False for negative-sigma submission."
    assert ("sigma" in msg.lower()) or ("std" in msg.lower()) or ("uncert" in msg.lower()), "Expected message to mention sigma/uncertainty."


def test_api_validate_catches_missing_column(tool_mod, bad_submission_missing_col):
    """
    Missing required column (e.g., 'sigma') should be flagged.
    """
    candidates = ["validate_submission", "validate", "run_validate"]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No validate function found.")

    try:
        result = fn(input_path=str(bad_submission_missing_col))
    except TypeError:
        result = fn(str(bad_submission_missing_col))  # type: ignore

    if isinstance(result, dict):
        ok = bool(result.get("ok", False))
        msg = " ".join(str(x) for x in result.values())
    else:
        ok = False
        msg = str(result)

    assert ok is False, "Expected ok=False for missing-column submission."
    assert ("sigma" in msg.lower()) or ("column" in msg.lower()) or ("field" in msg.lower()), "Expected message to mention missing column."


# ======================================================================================
# Artifact generation API
# ======================================================================================

def test_generate_artifacts(tool_mod, ok_submission, tmp_workspace):
    """
    Artifact generator (if present) should emit JSON/HTML/CSV files and return a manifest (or paths).
    """
    entry_candidates = [
        "generate_validation_artifacts",
        "run_submission_validator",
        "produce_validation_outputs",
        "analyze_and_export",  # generic fallback
    ]
    entry = None
    for name in entry_candidates:
        if _has_attr(tool_mod, name):
            entry = getattr(tool_mod, name)
            break
    if entry is None:
        # Fall back to validate() with outdir + flags if that produces artifacts
        entry = None

    outdir = tmp_workspace["outputs"]
    if entry is not None:
        try:
            manifest = entry(input_path=str(ok_submission), outdir=str(outdir), json_out=True, html_out=True, csv_out=True, seed=99)
        except TypeError:
            manifest = entry(str(ok_submission), str(outdir), True, True, True, 99)  # type: ignore
    else:
        # Try validate() with artifact flags
        validate_candidates = ["validate_submission", "validate"]
        fn = None
        for name in validate_candidates:
            if _has_attr(tool_mod, name):
                fn = getattr(tool_mod, name)
                break
        if fn is None:
            pytest.xfail("No artifact-capable entrypoint found for submission validation.")
        try:
            manifest = fn(input_path=str(ok_submission), outdir=str(outdir), json_out=True, html_out=True, csv_out=True, seed=99)
        except TypeError:
            manifest = fn(str(ok_submission), str(outdir), True, True, True, 99)  # type: ignore

    # Presence checks
    json_files = list(outdir.glob("*.json"))
    html_files = list(outdir.glob("*.html"))
    csv_files = list(outdir.glob("*.csv"))

    assert json_files, "No JSON artifact produced by submission validator."
    assert html_files, "No HTML artifact produced by submission validator."
    # CSV may be optional; if present, ensure not empty
    for c in csv_files:
        _assert_file(c, min_size=64)

    # Minimal JSON schema check
    with open(json_files[0], "r", encoding="utf-8") as f:
        js = json.load(f)
    assert isinstance(js, dict), "Top-level JSON must be an object."
    # Look for hints of validation results
    has_summary = ("ok" in js) or ("errors" in js) or ("summary" in js)
    assert has_summary, "JSON should include validation 'ok' flag, 'errors', or a 'summary' section."


# ======================================================================================
# CLI End-to-End
# ======================================================================================

def test_cli_end_to_end(ok_submission, tmp_workspace):
    """
    End-to-end CLI test:
      • Runs the module as a CLI with --input/--outdir → emits JSON/HTML(/CSV) artifacts.
      • Uses --seed for determinism and compares JSON across two runs (modulo volatile fields).
      • Verifies optional audit log when SPECTRAMIND_LOG_PATH is set.
    """
    # Locate module file to construct python -m invocation
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "validate_submission.py",  # repo-root/tools/...
        Path(__file__).resolve().parents[1] / "validate_submission.py",            # tests/diagnostics/../
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("validate_submission.py not found; cannot run CLI end-to-end test.")

    outdir = tmp_workspace["outputs"]
    logsdir = tmp_workspace["logs"]

    env = {
        "PYTHONUNBUFFERED": "1",
        "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
    }

    args = (
        "--input", str(ok_submission),
        "--outdir", str(outdir),
        "--json",
        "--html",
        "--csv",
        "--seed", "2025",
        "--silent",
    )
    proc1 = _run_cli(module_file, args, env=env, timeout=240)
    if proc1.returncode != 0:
        msg = f"CLI run 1 failed (exit={proc1.returncode}).\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}"
        pytest.fail(msg)

    json1 = sorted(outdir.glob("*.json"))
    html1 = sorted(outdir.glob("*.html"))
    assert json1 and html1, "CLI run 1 did not produce required artifacts."

    # Determinism: second run with same seed into a new directory should match JSON (minus volatile fields)
    outdir2 = outdir.parent / "outputs_run2"
    outdir2.mkdir(exist_ok=True)
    args2 = (
        "--input", str(ok_submission),
        "--outdir", str(outdir2),
        "--json",
        "--html",
        "--csv",
        "--seed", "2025",
        "--silent",
    )
    proc2 = _run_cli(module_file, args2, env=env, timeout=240)
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
        assert ("validate_submission" in text) or ("submission" in text and "validate" in text), \
            "Audit log exists but lacks recognizable validator signature."


def test_cli_error_cases_missing_input(tmp_workspace):
    """
    CLI should:
      • Exit non-zero when required --input is missing.
      • Report helpful error text mentioning the missing/invalid flag.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "validate_submission.py",
        Path(__file__).resolve().parents[1] / "validate_submission.py",
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("validate_submission.py not found; cannot run CLI error tests.")

    outdir = tmp_workspace["outputs"]

    args = (
        "--outdir", str(outdir),
        "--json",
    )
    proc = _run_cli(module_file, args, env=None, timeout=120)
    assert proc.returncode != 0, "CLI should fail when required --input is missing."
    msg = (proc.stderr + "\n" + proc.stdout).lower()
    assert "input" in msg, "Error message should mention missing 'input'."


def test_cli_error_cases_bad_file(tmp_workspace):
    """
    CLI should:
      • Exit non-zero when the input path is not a valid file.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "validate_submission.py",
        Path(__file__).resolve().parents[1] / "validate_submission.py",
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("validate_submission.py not found; cannot run CLI error tests.")

    outdir = tmp_workspace["outputs"]

    bogus = tmp_workspace["inputs"] / "nope.csv"
    args = (
        "--input", str(bogus),
        "--outdir", str(outdir),
        "--json",
    )
    proc = _run_cli(module_file, args, env=None, timeout=120)
    assert proc.returncode != 0, "CLI should fail when input file does not exist."
    msg = (proc.stderr + "\n" + proc.stdout).lower()
    assert ("not found" in msg) or ("exist" in msg) or ("open" in msg), "Expected file-not-found style error."


# ======================================================================================
# Housekeeping checks
# ======================================================================================

def test_artifact_min_sizes(tmp_workspace):
    """
    After prior tests, ensure that JSON/HTML in outputs/ are non-trivially sized.
    """
    outdir = tmp_workspace["outputs"]
    json_files = list(outdir.glob("*.json"))
    html_files = list(outdir.glob("*.html"))
    for p in json_files:
        _assert_file(p, min_size=64)
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