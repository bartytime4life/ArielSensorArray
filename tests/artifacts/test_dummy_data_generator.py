# tests/artifacts/test_dummy_data_generator.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Artifact Tests
File: tests/artifacts/test_dummy_data_generator.py

Purpose
-------
Validate that the dummy data generator produces well-formed, reproducible,
challenge-shaped synthetic artifacts for quick pipeline tests.

This test suite will:
  1) Invoke the generator via Python API if available, else via CLI (fallback).
  2) Generate a tiny dataset (N=5) with the challenge-standard 283 bins.
  3) Validate array shapes, finiteness, and physical plausibility:
       • mu: shape (N, 283)
       • sigma: shape (N, 283), strictly positive
  4) Validate identifiers/metadata (if produced).
  5) Validate reproducibility w/ same seed; and sensitivity w/ different seed.

Assumptions
-----------
• The repository includes a generator at tools/generate_dummy_data.py
  exposing either:
    A) API function (any of the following names):
         - generate_dummy_data(outdir:str, n:int=…, bins:int=…, seed:int=…, **opts)
         - run(outdir=…, n=…, bins=…, seed=…)
         - main_generate(outdir=…, n=…, bins=…, seed=…)
       Return value is optional; artifacts must be written to outdir.

    B) CLI (one of these entry styles):
         $ python tools/generate_dummy_data.py --outdir OUT --n 5 --bins 283 --seed 123
         $ python -m tools.generate_dummy_data --outdir OUT --n 5 --bins 283 --seed 123
       We try multiple python/flag variants (see CLI_FLAGS_MATRIX below).

• Expected outputs (best-effort discovery; at least mu & sigma arrays must exist):
    OUT/
      mu.npy or mu.npz (key 'mu')
      sigma.npy or sigma.npz (key 'sigma')
      metadata.json or metadata.csv (optional but preferred)

If your actual generator uses different flags or filenames, you can add variants
in CLI_FLAGS_MATRIX or in _find_arrays() heuristics below.
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytest

# ------------------------------- Constants --------------------------------- #

BINS = 283
N_SMALL = 5

GENERATOR_REL = Path("tools") / "generate_dummy_data.py"

PY_BIN_CANDS = [sys.executable, "python", "python3"]
ENTRY_CANDS = [
    # module entry
    ["-m", "tools.generate_dummy_data"],
    # direct path (resolved against repo root)
    [str(GENERATOR_REL)],
]
CLI_FLAGS_MATRIX = [
    # preferred
    dict(outdir="--outdir", n="--n", bins="--bins", seed="--seed"),
    # alternates
    dict(outdir="--out", n="--num", bins="--bins", seed="--seed"),
    dict(outdir="--outdir", n="--count", bins="--bins", seed="--seed"),
]

# Heuristic filename patterns for arrays and metadata
MU_PATS = ("mu.npy", "mu.npz")
SIGMA_PATS = ("sigma.npy", "sigma.npz")
META_PATS = ("metadata.json", "metadata.csv")


# ------------------------------- Utilities --------------------------------- #

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path.cwd().resolve()


@dataclass
class APIGen:
    func: callable
    supports: Dict[str, bool]


def discover_api() -> Optional[APIGen]:
    """
    Try to import a Python API from tools/generate_dummy_data.py
    """
    root = repo_root()
    mod_path = root / GENERATOR_REL
    if not mod_path.exists():
        return None
    sys.path.insert(0, str(root))
    try:
        mod = __import__("tools.generate_dummy_data", fromlist=["*"])
    except Exception:
        return None

    for name in ("generate_dummy_data", "run", "main_generate"):
        fn = getattr(mod, name, None)
        if callable(fn):
            # best-effort capability sniff
            sigtxt = (getattr(fn, "__doc__", "") or "") + str(getattr(fn, "__annotations__", ""))
            supports = {
                "outdir": "outdir" in sigtxt or "output_dir" in sigtxt,
                "n": "n" in sigtxt or "num" in sigtxt or "count" in sigtxt,
                "bins": "bins" in sigtxt,
                "seed": "seed" in sigtxt,
            }
            return APIGen(fn, supports)
    return None


def run_cli(outdir: Path, n: int, bins: int, seed: int) -> Tuple[bool, List[str]]:
    """
    CLI fallback: try multiple python executables / entry styles / flag spellings.
    """
    root = repo_root()
    logs: List[str] = []
    for py in PY_BIN_CANDS:
        for entry in ENTRY_CANDS:
            for flags in CLI_FLAGS_MATRIX:
                cmd = [py]
                if entry and entry[0].endswith(".py"):
                    cmd.append(str(root / entry[0]))
                else:
                    cmd.extend(entry)
                cmd.extend([flags["outdir"], str(outdir)])
                cmd.extend([flags["n"], str(n)])
                cmd.extend([flags["bins"], str(bins)])
                cmd.extend([flags["seed"], str(seed)])

                try:
                    proc = subprocess.run(
                        cmd, cwd=str(root), check=False,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logs.append(f"[CLI ERROR] {' '.join(cmd)} → {e}")
                    continue
                out = (proc.stdout or "") + (proc.stderr or "")
                logs.append(f"$ {' '.join(cmd)}\nexit={proc.returncode}\n{out}")
                if proc.returncode == 0:
                    return True, logs
    return False, logs


def run_generator(outdir: Path, n: int, bins: int, seed: int) -> Tuple[bool, List[str]]:
    """
    Try API first; on failure, run CLI.
    """
    api = discover_api()
    if api:
        kwargs = {}
        if api.supports.get("outdir", True):
            kwargs["outdir"] = str(outdir)
        if api.supports.get("n", True):
            kwargs["n"] = n
        if api.supports.get("bins", True):
            kwargs["bins"] = bins
        if api.supports.get("seed", True):
            kwargs["seed"] = seed
        try:
            api.func(**kwargs)  # type: ignore[arg-type]
            return True, [f"[API] called {api.func.__name__} with {kwargs}"]
        except TypeError:
            # signature mismatch → fall back to CLI
            pass
        except Exception as e:
            return False, [f"[API-EXCEPTION] {e!r}"]

    # CLI fallback
    return run_cli(outdir=outdir, n=n, bins=bins, seed=seed)


def _load_npy(p: Path) -> np.ndarray:
    return np.load(p)


def _load_npz(p: Path, key: str) -> Optional[np.ndarray]:
    with np.load(p) as npz:
        if key in npz.files:
            return npz[key]
        # Tolerate alternate keys
        for alt in (key.upper(), key.capitalize()):
            if alt in npz.files:
                return npz[alt]
    return None


def _find_arrays(outdir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locate μ and σ arrays (npy or npz). Return (mu, sigma) as ndarrays.
    """
    mu_arr = None
    sig_arr = None

    # Direct .npy
    for name in MU_PATS:
        p = outdir / name
        if p.exists() and p.suffix == ".npy":
            mu_arr = _load_npy(p)
            break
    for name in SIGMA_PATS:
        p = outdir / name
        if p.exists() and p.suffix == ".npy":
            sig_arr = _load_npy(p)
            break

    # .npz container
    if mu_arr is None:
        for name in MU_PATS:
            p = outdir / name
            if p.exists() and p.suffix == ".npz":
                mu_arr = _load_npz(p, "mu")
                break
    if sig_arr is None:
        for name in SIGMA_PATS:
            p = outdir / name
            if p.exists() and p.suffix == ".npz":
                sig_arr = _load_npz(p, "sigma")
                break

    if mu_arr is None or sig_arr is None:
        raise AssertionError(f"Could not find mu/sigma arrays in {outdir}")

    return mu_arr, sig_arr


def _load_metadata(outdir: Path) -> Optional[dict]:
    for name in META_PATS:
        p = outdir / name
        if p.exists() and p.suffix == ".json":
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        if p.exists() and p.suffix == ".csv":
            try:
                rows = list(csv.DictReader(p.open("r", encoding="utf-8")))
                return {"rows": rows, "rows_count": len(rows)}
            except Exception:
                return None
    return None


def _basic_array_checks(mu: np.ndarray, sigma: np.ndarray, n: int, bins: int) -> None:
    assert mu.shape == (n, bins), f"mu shape {mu.shape} != {(n, bins)}"
    assert sigma.shape == (n, bins), f"sigma shape {sigma.shape} != {(n, bins)}"
    assert np.isfinite(mu).all(), "mu contains non‑finite values"
    assert np.isfinite(sigma).all(), "sigma contains non‑finite values"
    assert (sigma > 0).all(), "sigma must be strictly positive"
    # sanity: mu not all constant; per‑planet variance should be > 0 on average
    per_var = mu.var(axis=1)
    assert (per_var > 0).mean() > 0.5, "most generated spectra should have non‑zero variance"
    # magnitude sanity (broad check; adjust if your generator uses a different scale)
    assert np.abs(mu).max() < 10.0, "mu magnitude looks implausible (>10)"
    assert sigma.max() < 10.0, "sigma magnitude looks implausible (>10)"


# ------------------------------- Fixtures ---------------------------------- #

@pytest.fixture(scope="module")
def tmp_out_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("dummy_data_gen")


# --------------------------------- Tests ----------------------------------- #

def test_generate_happy_path(tmp_out_root: Path):
    out = tmp_out_root / "run1"
    out.mkdir(parents=True, exist_ok=True)
    ok, logs = run_generator(outdir=out, n=N_SMALL, bins=BINS, seed=123)
    if not ok:
        pytest.fail("Dummy data generator failed to run.\n" + "\n".join(logs))

    mu, sigma = _find_arrays(out)
    _basic_array_checks(mu, sigma, n=N_SMALL, bins=BINS)

    meta = _load_metadata(out)
    if meta is not None:
        # If JSON, expect a small set of keys; be tolerant
        if "rows" in meta:
            # CSV-style
            assert meta["rows_count"] in (0, N_SMALL), "CSV metadata rows_count mismatch"
        else:
            # JSON-style dict
            # Optional: planet_ids
            pids = meta.get("planet_ids")
            if pids is not None:
                assert isinstance(pids, list) and len(pids) == N_SMALL, "planet_ids length mismatch"
            # Optional: bins
            bins_meta = meta.get("bins")
            if bins_meta is not None:
                assert int(bins_meta) == BINS, "metadata 'bins' mismatch"


def test_reproducible_same_seed(tmp_out_root: Path):
    out_a = tmp_out_root / "run_seed123_a"
    out_b = tmp_out_root / "run_seed123_b"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    ok_a, log_a = run_generator(outdir=out_a, n=N_SMALL, bins=BINS, seed=123)
    ok_b, log_b = run_generator(outdir=out_b, n=N_SMALL, bins=BINS, seed=123)
    if not (ok_a and ok_b):
        pytest.fail("Generator failed (same seed case):\n" + "\n".join(log_a + log_b))

    mu_a, sigma_a = _find_arrays(out_a)
    mu_b, sigma_b = _find_arrays(out_b)

    assert np.array_equal(mu_a, mu_b), "mu must be identical for same seed"
    assert np.array_equal(sigma_a, sigma_b), "sigma must be identical for same seed"


def test_different_seed_changes_output(tmp_out_root: Path):
    out_a = tmp_out_root / "run_seed111"
    out_b = tmp_out_root / "run_seed222"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    ok_a, log_a = run_generator(outdir=out_a, n=N_SMALL, bins=BINS, seed=111)
    ok_b, log_b = run_generator(outdir=out_b, n=N_SMALL, bins=BINS, seed=222)
    if not (ok_a and ok_b):
        pytest.fail("Generator failed (different seed case):\n" + "\n".join(log_a + log_b))

    mu_a, sigma_a = _find_arrays(out_a)
    mu_b, sigma_b = _find_arrays(out_b)

    # Hamming distance-like check (allow rare equality but expect changes)
    same_mu = np.array_equal(mu_a, mu_b)
    same_sigma = np.array_equal(sigma_a, sigma_b)
    assert not (same_mu and same_sigma), "different seeds should change generated data"

    # Also check global correlation is not trivially 1.0
    corr = np.corrcoef(mu_a.ravel(), mu_b.ravel())[0, 1]
    assert corr < 0.999, f"unexpectedly high correlation ({corr:.6f}) for different seeds"


def test_bins_argument_respected(tmp_out_root: Path):
    # Use fewer bins (e.g., 57) to enforce the generator honors the flag
    alt_bins = 57
    out = tmp_out_root / "run_bins57"
    out.mkdir(parents=True, exist_ok=True)
    ok, logs = run_generator(outdir=out, n=3, bins=alt_bins, seed=7)
    if not ok:
        pytest.fail("Generator failed for non-default bins.\n" + "\n".join(logs))
    mu, sigma = _find_arrays(out)
    _basic_array_checks(mu, sigma, n=3, bins=alt_bins)


# --------------------------- Debug failure context ------------------------- #

def pytest_runtest_makereport(item, call):
    """
    On failure, append useful path hints to stderr.
    """
    if call.excinfo is not None and call.when == "call":
        root = repo_root()
        sys.stderr.write(
            f"[debug] repo_root={root}\n"
            f"[debug] generator_exists={(root / GENERATOR_REL).exists()}\n"
        )
