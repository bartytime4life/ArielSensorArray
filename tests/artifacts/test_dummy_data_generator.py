#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/artifacts/test_dummy_data_generator.py

SpectraMind V50 — Artifact Tests
Dummy Test Data Generator (tools/generate_dummy_data.py)

Goal
----
Validate that the dummy data generator can be invoked (API or CLI), writes a small,
self-contained synthetic dataset to the requested --outdir, behaves deterministically
with a fixed seed, and respects audit logging & filesystem hygiene.

This suite is **signature-agnostic** and **repo-layout-tolerant**:
- API discovery attempts several common module paths and callables.
- CLI discovery attempts `python -m tools.generate_dummy_data`.
- Direct script execution is attempted as a final fallback.
- If nothing is found yet, tests skip gracefully with a clear message.

What we check
-------------
1) End-to-end generation of a *tiny* dataset into a temp outdir:
   - At least one expected artifact exists (e.g., mu.npy, sigma.npy, airs.npy, fgs1.npy, metadata.json, labels.csv).
2) Determinism:
   - Two runs with the same seed produce byte-identical artifacts.
3) Outdir discipline:
   - No stray writes outside the requested --outdir (except logs/ and optional outputs/run_hash_summary*.json).
4) JSON metadata (if present) is valid JSON and minimally structured.
5) Audit log append-only behavior across repeated runs.

Usage
-----
pytest -q tests/artifacts/test_dummy_data_generator.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest


# ======================================================================================
# Discovery (API / module / script)
# ======================================================================================

# Candidate Python API modules and callables
API_CANDIDATES: List[Tuple[str, Iterable[str]]] = [
    ("tools.generate_dummy_data", ("run", "main", "generate", "generate_dummy_data")),
    ("src.tools.generate_dummy_data", ("run", "main", "generate", "generate_dummy_data")),
    ("spectramind.tools.generate_dummy_data", ("run", "main", "generate", "generate_dummy_data")),
]

# Candidate module names for `python -m`
MODULE_CANDIDATES = [
    "tools.generate_dummy_data",
    "src.tools.generate_dummy_data",
    "spectramind.tools.generate_dummy_data",
]

# Candidate script filenames under tools/
SCRIPT_CANDIDATES = [
    "generate_dummy_data.py",
    "generate_dummy_dataset.py",
    "generate_dummy_inputs.py",
]


def _discover_api() -> Optional[Callable]:
    """
    Try to import a callable generator from common module paths.
    Returns the first callable found, else None.
    """
    for mod_name, names in API_CANDIDATES:
        try:
            mod = __import__(mod_name, fromlist=list(names))
        except Exception:
            continue
        for nm in names:
            fn = getattr(mod, nm, None)
            if callable(fn):
                return fn
    return None


def _discover_module() -> Optional[str]:
    for mod in MODULE_CANDIDATES:
        try:
            __import__(mod)
            return mod
        except Exception:
            continue
    return None


def _discover_script(repo_root: Path) -> Optional[Path]:
    for nm in SCRIPT_CANDIDATES:
        p = repo_root / "tools" / nm
        if p.exists():
            return p
    return None


# ======================================================================================
# Helpers
# ======================================================================================

EXPECTED_FILENAMES = {
    # Likely names — generator may produce any subset of these
    "mu.npy",
    "sigma.npy",
    "airs.npy",
    "fgs1.npy",
    "X.npy",
    "y.npy",
    "metadata.json",
    "labels.csv",
    "planets.csv",
}


def _sha256_file(p: Path) -> str:
    h = sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_artifacts(outdir: Path) -> Dict[str, List[Path]]:
    files = list(outdir.glob("*")) + list(outdir.rglob("*"))
    # Only files at or under outdir
    files = [p for p in files if p.is_file()]
    out: Dict[str, List[Path]] = {"npy": [], "json": [], "csv": [], "other": []}
    for p in files:
        ext = p.suffix.lower()
        if ext == ".npy":
            out["npy"].append(p)
        elif ext == ".json":
            out["json"].append(p)
        elif ext == ".csv":
            out["csv"].append(p)
        else:
            out["other"].append(p)
    return out


def _pick_expected_paths(outdir: Path) -> List[Path]:
    """
    Return a set of expected artifact paths found under outdir,
    based on names we commonly see from the generator.
    """
    found = []
    for name in EXPECTED_FILENAMES:
        p = outdir / name
        if p.exists():
            found.append(p)
    return sorted(found)


def _ensure_repo_scaffold(repo_root: Path) -> None:
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs").mkdir(parents=True, exist_ok=True)


# ======================================================================================
# Runners (API / module / script)
# ======================================================================================

def _run_via_api(
    fn: Callable,
    outdir: Path,
    seed: int = 123,
    n_planets: int = 6,
    n_bins: int = 17,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Call API with flexible kwargs; only pass those the function appears to accept.
    """
    import inspect

    kwargs: Dict[str, Any] = {
        "outdir": str(outdir),
        "seed": seed,
        "n_planets": n_planets,
        "n_bins": n_bins,
        "quiet": True,
        "no_browser": True,
        "overwrite": True,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        pass

    return fn(**kwargs)


def _run_via_module(
    module_name: str,
    outdir: Path,
    seed: int = 123,
    n_planets: int = 6,
    n_bins: int = 17,
    extra_flags: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SPECTRAMIND_TEST", "1")

    cmd = [
        sys.executable, "-m", module_name,
        "--outdir", str(outdir),
        "--seed", str(seed),
        "--n-planets", str(n_planets),
        "--n-bins", str(n_bins),
        "--overwrite",
        "--quiet",
        "--no-browser",
    ]
    if extra_flags:
        cmd += list(extra_flags)

    return subprocess.run(
        cmd, cwd=str(Path.cwd()), env=env, capture_output=True, text=True, timeout=90
    )


def _run_via_script(
    script_path: Path,
    outdir: Path,
    seed: int = 123,
    n_planets: int = 6,
    n_bins: int = 17,
    extra_flags: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SPECTRAMIND_TEST", "1")

    cmd = [
        sys.executable, str(script_path),
        "--outdir", str(outdir),
        "--seed", str(seed),
        "--n-planets", str(n_planets),
        "--n-bins", str(n_bins),
        "--overwrite",
        "--quiet",
        "--no-browser",
    ]
    if extra_flags:
        cmd += list(extra_flags)

    return subprocess.run(
        cmd, cwd=str(script_path.parent.parent if script_path.parent.name == "tools" else Path.cwd()),
        env=env, capture_output=True, text=True, timeout=90
    )


# ======================================================================================
# Pytest fixtures
# ======================================================================================

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path) -> Path:
    _ensure_repo_scaffold(tmp_path)
    return tmp_path


# ======================================================================================
# Tests
# ======================================================================================

@pytest.mark.integration
def test_generate_tiny_dataset(repo_tmp: Path) -> None:
    """
    End-to-end: generate a tiny dataset and confirm at least one expected artifact exists.
    """
    outdir = repo_tmp / "outputs" / "dummy" / "tiny"
    outdir.mkdir(parents=True, exist_ok=True)

    api = _discover_api()
    mod = _discover_module() if api is None else None
    script = _discover_script(repo_tmp) if (api is None and mod is None) else None

    if api is not None:
        _ = _run_via_api(api, outdir=outdir, seed=777, n_planets=5, n_bins=13)
    elif mod is not None:
        proc = _run_via_module(mod, outdir=outdir, seed=777, n_planets=5, n_bins=13)
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        assert proc.returncode == 0
    elif script is not None:
        proc = _run_via_script(script, outdir=outdir, seed=777, n_planets=5, n_bins=13)
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        assert proc.returncode == 0
    else:
        pytest.skip("No dummy data generator found. Add tools/generate_dummy_data.py to enable this test.")

    # Verify artifacts
    arts = _scan_artifacts(outdir)
    found_named = _pick_expected_paths(outdir)
    assert arts["npy"] or arts["json"] or arts["csv"] or found_named, \
        "No expected artifacts produced by dummy data generator."

    # If metadata.json exists, ensure valid JSON
    meta = outdir / "metadata.json"
    if meta.exists():
        j = json.loads(meta.read_text(encoding="utf-8"))
        assert isinstance(j, dict), "metadata.json must be a JSON object"
        # Soft expectations (tolerant)
        maybe_keys = {"n_planets", "n_bins", "seed", "generator"}
        assert any(k in j for k in maybe_keys), "metadata.json should include basic fields like n_planets/n_bins/seed"


@pytest.mark.integration
def test_determinism_with_fixed_seed(repo_tmp: Path) -> None:
    """
    Two runs with the same seed should yield byte-identical artifacts (for all files present).
    """
    outA = repo_tmp / "outputs" / "dummy" / "seedA"
    outB = repo_tmp / "outputs" / "dummy" / "seedB"
    outA.mkdir(parents=True, exist_ok=True)
    outB.mkdir(parents=True, exist_ok=True)

    api = _discover_api()
    mod = _discover_module() if api is None else None
    script = _discover_script(repo_tmp) if (api is None and mod is None) else None

    # Run A
    if api is not None:
        _ = _run_via_api(api, outdir=outA, seed=1234, n_planets=4, n_bins=11)
    elif mod is not None:
        assert _run_via_module(mod, outdir=outA, seed=1234, n_planets=4, n_bins=11).returncode == 0
    elif script is not None:
        assert _run_via_script(script, outdir=outA, seed=1234, n_planets=4, n_bins=11).returncode == 0
    else:
        pytest.skip("No dummy data generator available.")

    # Run B
    if api is not None:
        _ = _run_via_api(api, outdir=outB, seed=1234, n_planets=4, n_bins=11)
    elif mod is not None:
        assert _run_via_module(mod, outdir=outB, seed=1234, n_planets=4, n_bins=11).returncode == 0
    elif script is not None:
        assert _run_via_script(script, outdir=outB, seed=1234, n_planets=4, n_bins=11).returncode == 0

    # Compare overlapping files by name
    filesA = {p.name: p for p in (list(outA.glob("*")) + list(outA.rglob("*"))) if p.is_file()}
    filesB = {p.name: p for p in (list(outB.glob("*")) + list(outB.rglob("*"))) if p.is_file()}
    common = sorted(set(filesA.keys()) & set(filesB.keys()))
    assert common, "No overlapping artifact filenames to compare for determinism."

    for name in common:
        hA = _sha256_file(filesA[name])
        hB = _sha256_file(filesB[name])
        assert hA == hB, f"Artifact differs between same-seed runs: {name}"


@pytest.mark.integration
def test_outdir_respected_and_no_strays(repo_tmp: Path) -> None:
    """
    Ensure the generator writes under --outdir (plus logs/) and does not create stray files.
    """
    outdir = repo_tmp / "outputs" / "dummy" / "sandbox"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    api = _discover_api()
    mod = _discover_module() if api is None else None
    script = _discover_script(repo_tmp) if (api is None and mod is None) else None

    if api is not None:
        _ = _run_via_api(api, outdir=outdir, seed=999, n_planets=3, n_bins=9)
    elif mod is not None:
        assert _run_via_module(mod, outdir=outdir, seed=999, n_planets=3, n_bins=9).returncode == 0
    elif script is not None:
        assert _run_via_script(script, outdir=outdir, seed=999, n_planets=3, n_bins=9).returncode == 0
    else:
        pytest.skip("No dummy data generator available.")

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
        if "/__pycache__/" in rel or rel.endswith(".pyc"):
            continue
        disallowed.append(rel)

    assert not disallowed, f"Generator wrote unexpected files outside --outdir: {disallowed}"


@pytest.mark.integration
def test_metadata_json_if_present(repo_tmp: Path) -> None:
    """
    If metadata.json is present, ensure basic fields look sane.
    """
    outdir = repo_tmp / "outputs" / "dummy" / "meta_check"
    outdir.mkdir(parents=True, exist_ok=True)

    api = _discover_api()
    mod = _discover_module() if api is None else None
    script = _discover_script(repo_tmp) if (api is None and mod is None) else None

    if api is not None:
        _ = _run_via_api(api, outdir=outdir, seed=42, n_planets=5, n_bins=15)
    elif mod is not None:
        assert _run_via_module(mod, outdir=outdir, seed=42, n_planets=5, n_bins=15).returncode == 0
    elif script is not None:
        assert _run_via_script(script, outdir=outdir, seed=42, n_planets=5, n_bins=15).returncode == 0
    else:
        pytest.skip("No dummy data generator available.")

    meta = outdir / "metadata.json"
    if not meta.exists():
        pytest.xfail("metadata.json not produced — acceptable if the minimal generator omits metadata.")
    blob = json.loads(meta.read_text(encoding="utf-8"))
    assert isinstance(blob, dict)
    # Loose checks
    if "n_planets" in blob:
        assert int(blob["n_planets"]) > 0
    if "n_bins" in blob:
        assert int(blob["n_bins"]) > 0
    if "seed" in blob:
        assert int(blob["seed"]) >= 0
    # Optional marker about the generator
    if "generator" in blob:
        assert isinstance(blob["generator"], str) and blob["generator"].strip()


@pytest.mark.integration
def test_audit_log_append_only(repo_tmp: Path) -> None:
    """
    Two invocations should append to logs/v50_debug_log.md (or at least not shrink it).
    """
    log_path = repo_tmp / "logs" / "v50_debug_log.md"

    out1 = repo_tmp / "outputs" / "dummy" / "log1"
    out2 = repo_tmp / "outputs" / "dummy" / "log2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    api = _discover_api()
    mod = _discover_module() if api is None else None
    script = _discover_script(repo_tmp) if (api is None and mod is None) else None

    def _do(outdir: Path, seed: int):
        if api is not None:
            _ = _run_via_api(api, outdir=outdir, seed=seed, n_planets=3, n_bins=7)
        elif mod is not None:
            assert _run_via_module(mod, outdir=outdir, seed=seed, n_planets=3, n_bins=7).returncode == 0
        elif script is not None:
            assert _run_via_script(script, outdir=outdir, seed=seed, n_planets=3, n_bins=7).returncode == 0
        else:
            pytest.skip("No dummy data generator available.")

    _do(out1, seed=101)
    size1 = log_path.stat().st_size if log_path.exists() else 0
    _do(out2, seed=101)
    size2 = log_path.stat().st_size if log_path.exists() else 0

    assert size2 >= size1, "Audit log size should not shrink."
    if size1 > 0:
        assert size2 > size1, "Audit log should typically grow after a second run."
