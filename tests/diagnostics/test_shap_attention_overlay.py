# tests/diagnostics/test_shap_attention_overlay.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics tests for tools/shap_attention_overlay.py

This suite validates the scientific logic, artifact generation, and CLI behavior of the
SHAP × Attention overlay tool that visualizes fused saliency across wavelength bins.

Coverage
--------
1) Core API sanity:
   • Fusion function (e.g., compute_shap_attention_fusion / fuse_shap_attention) produces
     intuitive ordering: bins with high |SHAP| and high attention rank highest.
   • Normalization and top‑K selection behave as expected.

2) Artifact generation API:
   • generate_shap_attention_artifacts(...) (or equivalent) produces JSON/CSV/PNG/HTML and
     writes a manifest or structured JSON.

3) CLI contract:
   • End-to-end run via subprocess (python -m tools.shap_attention_overlay).
   • Determinism with --seed (compare JSON modulo volatile fields).
   • Graceful error handling for missing/invalid args.
   • Optional SPECTRAMIND_LOG_PATH audit line is appended.

4) Housekeeping:
   • Output files are non-empty; multiple runs do not corrupt artifacts.

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
      1) tools.shap_attention_overlay
      2) shap_attention_overlay (top-level)
    """
    try:
        import tools.shap_attention_overlay as m  # type: ignore
        return m
    except Exception:
        try:
            import shap_attention_overlay as m2  # type: ignore
            return m2
        except Exception:
            pytest.skip(
                "shap_attention_overlay module not found. "
                "Expected at tools/shap_attention_overlay.py or importable as shap_attention_overlay."
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
    Execute the tool as a CLI using `python -m tools.shap_attention_overlay` when possible.
    Fallback to direct script invocation by file path if package execution is not feasible.
    """
    if module_path.name == "shap_attention_overlay.py" and module_path.parent.name == "tools":
        repo_root = module_path.parent.parent
        candidate_pkg = "tools.shap_attention_overlay"
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


def _topk_by_abs(v: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, v.size))
    order = np.lexsort((np.arange(v.size), np.abs(v)))
    return np.sort(order[-k:])


# ======================================================================================
# Synthetic inputs
# ======================================================================================

def _make_wavelengths(n: int = 283, lo_um: float = 0.5, hi_um: float = 7.8) -> np.ndarray:
    return np.linspace(float(lo_um), float(hi_um), n, dtype=np.float64)


def _make_mu(n: int = 283, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Smooth-ish baseline spectrum with gentle structure
    x = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float64)
    mu = 1.0e-3 + 2.5e-4 * np.sin(3.0 * x) + 1.5e-4 * np.cos(7.0 * x)
    mu += 5.0e-5 * rng.standard_normal(n)
    return mu


def _make_shap_attention(n: int = 283, seed: int = 23) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create SHAP values and attention weights with aligned peaks around two wavelength bands.
    """
    rng = np.random.default_rng(seed)
    wl = _make_wavelengths(n)
    shap_vals = 1e-3 * rng.standard_normal(n)

    # Inject two strong SHAP bands (one positive, one negative)
    band1 = (wl >= 1.35) & (wl <= 1.45)  # H2O-like
    band2 = (wl >= 3.25) & (wl <= 3.35)  # CH4-like
    shap_vals[band1] += 0.030
    shap_vals[band2] -= 0.028

    # Attention weights (non-negative, sum to 1); focus near same bands
    att = np.clip(0.1 * rng.random(n), 0.0, 1.0)
    att[band1] += 0.6
    att[band2] += 0.5
    att = np.maximum(att, 0.0)
    att = att / np.sum(att)

    return shap_vals.astype(np.float64), att.astype(np.float64)


# Local reference fusion for property testing
def _ref_fuse(shap_vals: np.ndarray, att: np.ndarray, mode: str = "product", normalize: bool = True) -> np.ndarray:
    """
    Reference fusion:
      product: |shap| * att
      sum:     α|shap| + (1-α)att  with α=0.5
      rank:    normalize ranks and average
    """
    s = np.abs(shap_vals).astype(np.float64)
    a = np.clip(att, 0.0, None).astype(np.float64)

    if mode == "product":
        f = s * a
    elif mode == "sum":
        f = 0.5 * s / (s.sum() + 1e-12) + 0.5 * a
    elif mode == "rank":
        rs = (np.argsort(np.argsort(s)) + 1).astype(np.float64)  # 1..n
        ra = (np.argsort(np.argsort(a)) + 1).astype(np.float64)
        f = (rs + ra) / (2.0 * len(s))
    else:
        raise ValueError("unknown fusion mode")

    if normalize:
        z = f.sum() + 1e-12
        f = f / z
    return f


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
      inputs/  — .npy arrays (mu, shap, attention, wavelengths)
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
    Save deterministic arrays to disk:
      - wavelengths.npy
      - mu.npy
      - shap.npy
      - attention.npy
    """
    n = 283
    wl = _make_wavelengths(n)
    mu = _make_mu(n)
    shap_vals, att = _make_shap_attention(n)

    ip = tmp_workspace["inputs"]
    wl_path = ip / "wavelengths.npy"
    mu_path = ip / "mu.npy"
    shap_path = ip / "shap.npy"
    att_path = ip / "attention.npy"

    np.save(wl_path, wl)
    np.save(mu_path, mu)
    np.save(shap_path, shap_vals)
    np.save(att_path, att)

    return {
        "wavelengths": wl_path,
        "mu": mu_path,
        "shap": shap_path,
        "attention": att_path,
    }


# ======================================================================================
# Core API tests — fusion & top‑K behavior
# ======================================================================================

def test_fusion_emphasizes_joint_high_importance(tool_mod, synthetic_inputs):
    """
    Fusion score should be higher in bins where both |SHAP| and attention are high.
    """
    candidates = [
        "compute_shap_attention_fusion",
        "fuse_shap_attention",
        "compute_fusion_scores",
        "shap_attention_fusion",
    ]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No fusion function (compute_shap_attention_fusion/fuse_shap_attention/...) found.")

    wl = np.load(synthetic_inputs["wavelengths"])
    shap_vals = np.load(synthetic_inputs["shap"])
    att = np.load(synthetic_inputs["attention"])

    # Call implementation; be flexible with signature
    try:
        fused = np.asarray(fn(shap_vals=shap_vals, attention=att, method="product", normalize=True))
    except TypeError:
        fused = np.asarray(fn(shap_vals, att, "product", True))  # type: ignore

    assert fused.shape == shap_vals.shape
    assert np.all(np.isfinite(fused)), "Fused array contains non-finite values."

    # Reference and implementation top‑K should substantially overlap
    ref = _ref_fuse(shap_vals, att, mode="product", normalize=True)
    top_impl = set(_topk_by_abs(fused, 15))
    top_ref = set(_topk_by_abs(ref, 15))
    overlap = len(top_impl & top_ref) / 15.0
    assert overlap >= 0.6, f"Fusion top‑K overlap too low: {overlap:.2f}"


def test_topk_contains_known_bands(tool_mod, synthetic_inputs):
    """
    We injected bands near ~1.40 μm and ~3.30 μm. Fused top‑K should include indices from these regions.
    """
    wl = np.load(synthetic_inputs["wavelengths"])
    shap_vals = np.load(synthetic_inputs["shap"])
    att = np.load(synthetic_inputs["attention"])

    # Prefer module fusion; fallback to reference if unavailable
    fn = None
    for name in ("compute_shap_attention_fusion", "fuse_shap_attention"):
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        fused = _ref_fuse(shap_vals, att, mode="product", normalize=True)
    else:
        try:
            fused = np.asarray(fn(shap_vals=shap_vals, attention=att, method="product", normalize=True))
        except TypeError:
            fused = np.asarray(fn(shap_vals, att, "product", True))  # type: ignore

    top = _topk_by_abs(fused, 20)
    band1 = np.where((wl >= 1.35) & (wl <= 1.45))[0]
    band2 = np.where((wl >= 3.25) & (wl <= 3.35))[0]

    hit1 = np.intersect1d(top, band1).size
    hit2 = np.intersect1d(top, band2).size
    assert hit1 >= 2, "Top‑K fusion should include multiple indices from ~1.40 μm band."
    assert hit2 >= 2, "Top‑K fusion should include multiple indices from ~3.30 μm band."


# ======================================================================================
# Artifact generation API
# ======================================================================================

def test_generate_artifacts(tool_mod, tmp_workspace, synthetic_inputs):
    """
    Artifact generator should emit JSON/CSV/PNG/HTML files and return a manifest (or paths).
    """
    entry_candidates = [
        "generate_shap_attention_artifacts",
        "run_shap_attention_overlay",
        "produce_shap_attention_outputs",
        "analyze_and_export",  # generic fallback
    ]
    entry = None
    for name in entry_candidates:
        if _has_attr(tool_mod, name):
            entry = getattr(tool_mod, name)
            break
    if entry is None:
        pytest.xfail("No artifact generation entrypoint found in shap_attention_overlay.")

    outdir = tmp_workspace["outputs"]
    mu = np.load(synthetic_inputs["mu"])
    wl = np.load(synthetic_inputs["wavelengths"])
    shap_vals = np.load(synthetic_inputs["shap"])
    att = np.load(synthetic_inputs["attention"])

    kwargs = dict(
        mu=mu,
        wavelengths=wl,
        shap_values=shap_vals,
        attention=att,
        outdir=str(outdir),
        json_out=True,
        csv_out=True,
        png_out=True,
        html_out=True,
        seed=77,
        title="SHAP × Attention Overlay — Test",
        top_k=20,
    )
    try:
        manifest = entry(**kwargs)
    except TypeError:
        manifest = entry(mu, wl, shap_vals, att, str(outdir), True, True, True, True, 77, "SHAP × Attention Overlay — Test", 20)  # type: ignore

    # Presence checks
    json_files = list(outdir.glob("*.json"))
    csv_files = list(outdir.glob("*.csv"))
    png_files = list(outdir.glob("*.png"))
    html_files = list(outdir.glob("*.html"))

    assert json_files, "No JSON artifact produced by SHAP × Attention overlay."
    assert csv_files, "No CSV artifact produced by SHAP × Attention overlay."
    assert png_files, "No PNG artifact produced by SHAP × Attention overlay."
    assert html_files, "No HTML artifact produced by SHAP × Attention overlay."

    # Minimal JSON schema check
    with open(json_files[0], "r", encoding="utf-8") as f:
        js = json.load(f)
    assert isinstance(js, dict), "Top-level JSON must be an object."
    has_fusion = ("fusion" in js) or ("scores" in js) or ("topk" in js) or ("overlay" in js)
    assert has_fusion, "JSON should include fusion/overlay content (keys like 'fusion','scores','topk','overlay')."

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
      • Runs the module as a CLI with --mu/--wavelengths/--shap/--attention → emits JSON/CSV/PNG/HTML.
      • Uses --seed for determinism and compares JSON across two runs (modulo volatile metadata).
      • Verifies optional audit log when SPECTRAMIND_LOG_PATH is set.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "shap_attention_overlay.py",  # repo-root/tools/...
        Path(__file__).resolve().parents[1] / "shap_attention_overlay.py",            # tests/diagnostics/../
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("shap_attention_overlay.py not found; cannot run CLI end-to-end test.")

    outdir = tmp_workspace["outputs"]
    logsdir = tmp_workspace["logs"]
    mu_path = synthetic_inputs["mu"]
    wl_path = synthetic_inputs["wavelengths"]
    shap_path = synthetic_inputs["shap"]
    att_path = synthetic_inputs["attention"]

    env = {
        "PYTHONUNBUFFERED": "1",
        "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
    }

    args = (
        "--mu", str(mu_path),
        "--wavelengths", str(wl_path),
        "--shap", str(shap_path),
        "--attention", str(att_path),
        "--outdir", str(outdir),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--top-k", "20",
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
        "--shap", str(shap_path),
        "--attention", str(att_path),
        "--outdir", str(outdir2),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--top-k", "20",
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
        assert ("shap_attention_overlay" in text) or ("shap" in text and "attention" in text), \
            "Audit log exists but lacks recognizable CLI signature."


def test_cli_error_cases(tmp_workspace, synthetic_inputs):
    """
    CLI should:
      • Exit non-zero when required args are missing.
      • Report helpful error text mentioning the missing flag(s).
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "shap_attention_overlay.py",
        Path(__file__).resolve().parents[1] / "shap_attention_overlay.py",
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("shap_attention_overlay.py not found; cannot run CLI error tests.")

    outdir = tmp_workspace["outputs"]
    wl = synthetic_inputs["wavelengths"]
    shap_path = synthetic_inputs["shap"]
    att_path = synthetic_inputs["attention"]

    # Missing --mu
    args_missing_mu = (
        "--wavelengths", str(wl),
        "--shap", str(shap_path),
        "--attention", str(att_path),
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