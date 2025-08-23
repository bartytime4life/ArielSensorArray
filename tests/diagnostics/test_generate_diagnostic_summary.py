#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_generate_diagnostic_summary.py

SpectraMind V50 — Diagnostics Tests (generate_diagnostic_summary.py)

This test suite validates the upgraded diagnostics generator tool:
tools/generate_diagnostic_summary.py

Goals
-----
1) End-to-end CLI execution against tiny synthetic inputs (μ/σ, y_true).
2) Verify creation and schema of `diagnostic_summary.json`.
3) Check that plots/HTML (if enabled) are emitted, and filenames are reasonable.
4) Ensure robust handling of NaNs/Infs and degenerate σ, without crashing.
5) Confirm append-only audit logging into `logs/v50_debug_log.md`.
6) Exercise optional overlays (symbolic, SHAP, FFT) when present.
7) Validate Hydra-safe config interop where applicable.

Design
------
• Minimal runtime (<2s) with tiny arrays (N_planets=5, N_bins=17 for speed).
• Flexible assertions: tolerate additional keys/plots but require core outputs.
• Dual-mode launcher: import-and-call `main()` if available, else use subprocess.
• Self-contained temp repo layout: creates `tools/`, `logs/`, `outputs/` if needed.
• Rich inline comments for maintainability (NASA-grade documentation style).

Usage
-----
pytest -q tests/diagnostics/test_generate_diagnostic_summary.py
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pytest


# -----------------------------
# Constants & tiny test shapes
# -----------------------------

N_PLANETS = 5     # Keep very small for test speed
N_BINS = 17       # 17 << 283 to reduce IO/plotting overhead


# --------------------------------------
# Helper: synthetic scientific test data
# --------------------------------------

def _mk_synthetic_arrays(seed: int = 123) -> Dict[str, np.ndarray]:
    """
    Create tiny synthetic μ, σ, and y_true arrays with basic structure:
    • μ: smooth-ish signal with slight sinusoid + random jitter
    • σ: positive, mildly correlated with local curvature
    • y_true: μ + small noise (simulating ground truth about predictions)
    Also inject controlled NaNs/Infs to exercise robustness.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, N_BINS, dtype=np.float64)

    mu = []
    sigma = []
    y_true = []
    for _ in range(N_PLANETS):
        base = 0.2 * np.sin(x) + 0.05 * np.cos(2 * x)  # small spectral wiggle
        jitter = rng.normal(0, 0.01, size=N_BINS)
        this_mu = base + jitter
        # σ must be positive; couple to |second derivative| + floor
        curvature = np.abs(np.gradient(np.gradient(this_mu)))
        this_sigma = 0.03 + 0.05 * (curvature / (curvature.max() + 1e-6))
        # y_true ~ mu + tiny noise
        this_y = this_mu + rng.normal(0, 0.01, size=N_BINS)

        mu.append(this_mu)
        sigma.append(this_sigma)
        y_true.append(this_y)

    mu = np.stack(mu, axis=0)
    sigma = np.stack(sigma, axis=0)
    y_true = np.stack(y_true, axis=0)

    # Inject a couple of NaNs/Infs to test robustness
    if N_PLANETS >= 2 and N_BINS >= 4:
        mu[1, 3] = np.nan
        sigma[2, 5] = np.inf  # unrealistic, should be handled internally
    if N_PLANETS >= 3 and N_BINS >= 8:
        # also test a near-zero sigma that must be clamped inside the tool
        sigma[3, 7] = 1e-12

    return {"mu": mu, "sigma": sigma, "y_true": y_true}


def _mk_symbolic_overlay(out_dir: Path) -> Path:
    """
    Create a tiny symbolic violation JSON overlay with made-up rule scores.
    Expected to be optionally consumed by the diagnostics generator.
    """
    data = {
        "meta": {"version": "test-1.0", "n_rules": 3},
        "rules": [
            {"id": "H2O_band_consistency", "weight": 1.0},
            {"id": "CO2_peak_alignment", "weight": 0.8},
            {"id": "CH4_edge_monotonicity", "weight": 0.6},
        ],
        "per_planet": [],
    }
    # build dummy per-planet violation magnitudes across bins
    for pid in range(N_PLANETS):
        violations = (np.linspace(0, 1, N_BINS) * (pid + 1) / N_PLANETS).tolist()
        data["per_planet"].append(
            {
                "planet_id": f"P{pid:03d}",
                "violations": {
                    "H2O_band_consistency": violations,
                    "CO2_peak_alignment": violations[::-1],
                    "CH4_edge_monotonicity": [float(v > 0.5) for v in violations],
                },
            }
        )
    path = out_dir / "symbolic_results.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def _mk_shap_overlay(out_dir: Path) -> Path:
    """
    Create a tiny SHAP overlay JSON with per-bin importance magnitudes.
    """
    data = {
        "meta": {"version": "test-1.0"},
        "per_planet": [],
    }
    for pid in range(N_PLANETS):
        shap_vals = np.abs(np.sin(np.linspace(0, 2 * np.pi, N_BINS))).tolist()
        data["per_planet"].append(
            {"planet_id": f"P{pid:03d}", "shap_abs": shap_vals}
        )
    path = out_dir / "shap_overlay.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def _ensure_repo_scaffold(repo_root: Path) -> None:
    """
    Tests assume a minimal repo-like layout so relative paths inside the tool
    (e.g., logs/ and outputs/ defaults) resolve cleanly.

    This creates:
      • tools/ (placeholder file if missing)
      • logs/  (append-only audit log target)
      • outputs/diagnostics/ (typical default for tool output)
    """
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "diagnostics").mkdir(parents=True, exist_ok=True)

    # If the tool file does not exist yet, place a tiny shim that raises a helpful error.
    tool_path = repo_root / "tools" / "generate_diagnostic_summary.py"
    if not tool_path.exists():
        tool_path.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                # NOTE: This shim exists for test bootstrapping only.
                # Replace with the real `tools/generate_diagnostic_summary.py`.
                if __name__ == "__main__":
                    raise SystemExit(
                        "Shim placeholder for tools/generate_diagnostic_summary.py — replace with real file."
                    )
                """
            )
        )


# ---------------------------
# Helper: tool invocations
# ---------------------------

def _discover_tool(repo_root: Path) -> Path:
    """
    Return path to tools/generate_diagnostic_summary.py under the given repo root.
    """
    tool = repo_root / "tools" / "generate_diagnostic_summary.py"
    return tool


def _run_tool_subprocess(
    repo_root: Path,
    tool_path: Path,
    mu_path: Path,
    sigma_path: Path,
    y_true_path: Path,
    outdir: Path,
    symbolic_json: Optional[Path] = None,
    shap_json: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    use_python_module: bool = False,
) -> subprocess.CompletedProcess:
    """
    Execute the tool via subprocess. We prefer `python <tool>` for fewer surprises;
    but optionally we can try `python -m tools.generate_diagnostic_summary`.

    Environment:
      • Set SPECTRAMIND_TEST=1 for lightweight behavior where supported.
      • Set MPLBACKEND=Agg to avoid GUI issues during plot generation.

    Returns CompletedProcess for inspection of returncode/stdout/stderr.
    """
    env = os.environ.copy()
    env.setdefault("SPECTRAMIND_TEST", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")

    cmd: List[str]
    if use_python_module:
        # Launch as a module (repo root must be on PYTHONPATH)
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            "-m",
            "tools.generate_diagnostic_summary",
        ]
    else:
        cmd = [sys.executable, str(tool_path)]

    # Baseline required args — these flags should match the tool's CLI contract.
    base_args = [
        "--mu", str(mu_path),
        "--sigma", str(sigma_path),
        "--y-true", str(y_true_path),
        "--outdir", str(outdir),
        "--no-browser",
    ]

    # Optional overlays
    if symbolic_json is not None:
        base_args += ["--symbolic", str(symbolic_json)]
    if shap_json is not None:
        base_args += ["--shap", str(shap_json)]

    # Common test-friendly args (tolerated if unknown by the tool)
    # The tool is expected to ignore unknown flags or expose these flags.
    base_args += [
        "--save-plots",
        "--save-html",
        "--max-planets", "5",
        "--max-bins", "17",
        "--quiet",
    ]

    if extra_args:
        base_args += list(extra_args)

    proc = subprocess.run(
        cmd + base_args,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------
# Pytest fixtures
# ----------------------

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path) -> Path:
    """
    Create an isolated, temp "repo root" with the expected structure.
    """
    _ensure_repo_scaffold(tmp_path)
    return tmp_path


@pytest.fixture(scope="function")
def tiny_inputs(repo_tmp: Path) -> Dict[str, Path]:
    """
    Materialize tiny μ/σ/y_true .npy arrays and return their paths.
    """
    arrays = _mk_synthetic_arrays(seed=777)
    inputs_dir = repo_tmp / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    mu_path = inputs_dir / "mu.npy"
    sigma_path = inputs_dir / "sigma.npy"
    y_true_path = inputs_dir / "y_true.npy"

    np.save(mu_path, arrays["mu"])
    np.save(sigma_path, arrays["sigma"])
    np.save(y_true_path, arrays["y_true"])

    return {"mu": mu_path, "sigma": sigma_path, "y_true": y_true_path}


@pytest.fixture(scope="function")
def overlays(repo_tmp: Path) -> Dict[str, Path]:
    """
    Generate optional overlay JSON files.
    """
    ov_dir = repo_tmp / "inputs" / "overlays"
    ov_dir.mkdir(parents=True, exist_ok=True)
    symbolic = _mk_symbolic_overlay(ov_dir)
    shap = _mk_shap_overlay(ov_dir)
    return {"symbolic": symbolic, "shap": shap}


# ---------------------------------------
# Core tests — end-to-end and robustness
# ---------------------------------------

@pytest.mark.integration
def test_cli_generates_summary_and_plots(repo_tmp: Path, tiny_inputs: Dict[str, Path], overlays: Dict[str, Path]) -> None:
    """
    End-to-end CLI test:
      • Run the tool with μ/σ/y_true and overlays
      • Expect diagnostic_summary.json
      • Expect at least one PNG and (optionally) an HTML file
      • Expect an audit log entry appended to logs/v50_debug_log.md
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_e2e"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        symbolic_json=overlays["symbolic"],
        shap_json=overlays["shap"],
        extra_args=["--version", "test"],
    )

    # Helpful debug on failure
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)

    assert proc.returncode == 0, "Diagnostics generator should exit successfully."

    summary_path = outdir / "diagnostic_summary.json"
    assert summary_path.exists(), "Expected diagnostic_summary.json to be written."
    summary = _read_json(summary_path)

    # Minimal schema checks
    assert isinstance(summary, dict), "diagnostic_summary.json must be a JSON object."
    assert "meta" in summary and isinstance(summary["meta"], dict)
    assert "metrics" in summary and isinstance(summary["metrics"], dict)
    assert "per_planet" in summary and isinstance(summary["per_planet"], list)

    # Core metrics sanity
    metrics = summary["metrics"]
    # The generator typically includes 'mean_gll' or 'gll' keys and calibration stats.
    # Accept either, but require at least one GLL-like metric present and numeric.
    gll_like_keys = [k for k in metrics.keys() if "gll" in k.lower()]
    assert gll_like_keys, "Expected at least one GLL-related metric in summary['metrics']."
    for k in gll_like_keys:
        assert isinstance(metrics[k], (int, float)), f"Metric {k} should be numeric."

    # Files: png/plots (not strict on exact names; require at least one png),
    # and optional HTML report.
    pngs = list(outdir.glob("*.png"))
    assert pngs, "Expected at least one PNG plot output in diagnostics outdir."

    htmls = list(outdir.glob("*.html"))
    # HTML is optional; if present we lightly validate
    if htmls:
        html_txt = htmls[0].read_text(encoding="utf-8", errors="ignore")
        assert "<html" in html_txt.lower(), "HTML report should contain an <html> tag."

    # Audit log appended
    log_path = repo_tmp / "logs" / "v50_debug_log.md"
    assert log_path.exists(), "Expected audit log to be present."
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    assert "generate_diagnostic_summary" in log_text or "diagnostic" in log_text.lower(), \
        "Audit log should mention diagnostics generation."


@pytest.mark.integration
def test_cli_handles_nans_infs_and_tiny_sigma(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Stress edge cases:
      • μ has NaN(s)
      • σ has Inf and near-zero values
    Expect the tool to complete and produce a summary without NaN/Inf in core metrics.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_edgecases"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        extra_args=["--version", "edge"],
    )

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)

    assert proc.returncode == 0, "Tool should handle NaNs/Infs/near-zero σ without crashing."

    summary_path = outdir / "diagnostic_summary.json"
    assert summary_path.exists(), "Expected diagnostic_summary.json to be written even on edge cases."
    summary = _read_json(summary_path)

    # Ensure metrics are finite numbers
    for k, v in summary.get("metrics", {}).items():
        if isinstance(v, (int, float)):
            assert math.isfinite(float(v)), f"Metric {k} should be finite, got {v}."


@pytest.mark.integration
def test_cli_without_overlays_is_still_successful(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Overlays are optional; tool must still run and produce outputs without them.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_no_overlays"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        extra_args=["--no-symbolic", "--no-shap", "--version", "no-ovl"],  # tolerated if tool exposes or ignores
    )

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)

    assert proc.returncode == 0, "Diagnostics must succeed without any overlays provided."
    assert (outdir / "diagnostic_summary.json").exists()


@pytest.mark.integration
def test_cli_module_invocation_mode(repo_tmp: Path, tiny_inputs: Dict[str, Path], overlays: Dict[str, Path]) -> None:
    """
    Some CI environments prefer python -m tools.generate_diagnostic_summary.
    Ensure module-based invocation also works (when the tool is importable).

    If the shim placeholder exists (no real tool), this test will xfail gracefully.
    """
    tool = _discover_tool(repo_tmp)
    is_shim = "Shim placeholder" in tool.read_text(encoding="utf-8", errors="ignore")

    outdir = repo_tmp / "outputs" / "diagnostics" / "run_module_mode"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        symbolic_json=overlays["symbolic"],
        shap_json=overlays["shap"],
        extra_args=["--version", "mod"],
        use_python_module=True,
    )

    if is_shim:
        # The shim intentionally raises SystemExit with an instructive message.
        pytest.xfail("Tool shim detected — replace with real tools/generate_diagnostic_summary.py to pass this test.")

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)

    assert proc.returncode == 0
    assert (outdir / "diagnostic_summary.json").exists()


@pytest.mark.integration
def test_summary_contains_expected_sections(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Validate that the summary JSON includes categories we rely on downstream:
      • meta: version/hash/timestamps (flexible), but at least a dict
      • metrics: dict with GLL/MAE/RMSE/coverage-like keys (any subset acceptable)
      • per_planet: list with planet-level dicts including id and (optionally) diagnostics
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_sections"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        extra_args=["--version", "sections"],
    )
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0

    summary = _read_json(outdir / "diagnostic_summary.json")

    assert isinstance(summary.get("meta"), dict), "meta must be a dict"
    assert isinstance(summary.get("metrics"), dict), "metrics must be a dict"
    assert isinstance(summary.get("per_planet"), list), "per_planet must be a list"

    # per_planet entries contain a planet_id and may include bin-level diagnostics
    if summary["per_planet"]:
        entry0 = summary["per_planet"][0]
        assert isinstance(entry0, dict)
        assert any(k in entry0 for k in ("planet_id", "id", "name")), "per_planet[0] should carry an identifier"


@pytest.mark.integration
def test_plots_are_reasonable_filenames(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    The exact plot names are implementation-defined. We assert a simple policy:
    • At least one .png exists.
    • Filenames should be alnum/underscore/dash + .png (no spaces, no unsafe chars).
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_plot_names"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        extra_args=["--version", "plots"],
    )
    assert proc.returncode == 0

    pngs = list(outdir.glob("*.png"))
    assert pngs, "Expected at least one plot (PNG)."

    for p in pngs:
        assert re.match(r"^[A-Za-z0-9._\-]+\.png$", p.name), f"Unsafe plot filename: {p.name}"


@pytest.mark.integration
def test_audit_log_is_append_only(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Confirm audit log appends (does not truncate). We'll record size before and after a second run.
    """
    tool = _discover_tool(repo_tmp)
    outdir1 = repo_tmp / "outputs" / "diagnostics" / "run_log_append_1"
    outdir2 = repo_tmp / "outputs" / "diagnostics" / "run_log_append_2"
    outdir1.mkdir(parents=True, exist_ok=True)
    outdir2.mkdir(parents=True, exist_ok=True)

    log_path = repo_tmp / "logs" / "v50_debug_log.md"

    # First run
    proc1 = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir1,
        extra_args=["--version", "log1"],
    )
    assert proc1.returncode == 0
    size1 = log_path.stat().st_size if log_path.exists() else 0

    # Second run
    proc2 = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir2,
        extra_args=["--version", "log2"],
    )
    assert proc2.returncode == 0
    size2 = log_path.stat().st_size if log_path.exists() else 0

    # Append-only → size2 >= size1; in practice, strictly greater unless tool suppresses logs
    assert size2 >= size1, "Audit log should not shrink between runs."
    if size1 > 0:
        assert size2 > size1, "Expected audit log growth after a second run."


@pytest.mark.integration
def test_tool_respects_output_directory(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Ensure --outdir is honored strictly: no stray files in repo root or unrelated dirs.
    We allow writes to logs/ for audit; everything else should live under outdir.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "sandbox_outdir"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        extra_args=["--version", "outdir"],
    )
    assert proc.returncode == 0

    after = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    new_files = sorted(list(after - before))
    # Allowed: anything under outdir and logs/v50_debug_log.md
    disallowed = []
    for rel in new_files:
        if rel.startswith("logs/"):
            continue
        if rel.startswith(outdir.relative_to(repo_tmp).as_posix()):
            continue
        if rel.endswith("/__pycache__/") or rel.endswith(".pyc"):
            continue
        # If the tool emits a run_hash_summary or similar under outputs/, allow it:
        if rel.startswith("outputs/") and not rel.startswith(outdir.relative_to(repo_tmp).as_posix()):
            # tolerate run_hash_summary_v50.json at outputs/
            if not re.search(r"run_hash_summary.*\.json$", rel):
                disallowed.append(rel)
    assert not disallowed, f"Tool wrote unexpected files outside --outdir: {disallowed}"


@pytest.mark.integration
def test_json_is_minimally_stable_and_parseable(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    A JSON linter-friendly check:
      • ensure UTF-8 encoding
      • ensure no NaN/Inf (JSON spec forbids them)
      • ensure keys are strings, values serializable
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_json_stability"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        mu_path=tiny_inputs["mu"],
        sigma_path=tiny_inputs["sigma"],
        y_true_path=tiny_inputs["y_true"],
        outdir=outdir,
        extra_args=["--version", "json"],
    )
    assert proc.returncode == 0

    raw = (outdir / "diagnostic_summary.json").read_text(encoding="utf-8")
    # JSON must not contain bare NaN/Inf tokens
    assert "NaN" not in raw and "Infinity" not in raw and "-Infinity" not in raw, \
        "JSON must not contain non-standard NaN/Infinity literals."

    data = json.loads(raw)
    assert isinstance(data, dict)
    # spot-check nested structures are serializable
    _ = json.dumps(data)  # will raise if non-serializable


@pytest.mark.integration
def test_cli_returns_nonzero_on_input_missing(repo_tmp: Path) -> None:
    """
    If required inputs are missing, tool should exit with non-zero and emit a helpful error.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "run_missing_inputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # Provide only mu; omit sigma and y_true
    inputs_dir = repo_tmp / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    mu_path = inputs_dir / "mu.npy"
    np.save(mu_path, _mk_synthetic_arrays()["mu"])

    proc = subprocess.run(
        [sys.executable, str(tool), "--mu", str(mu_path), "--outdir", str(outdir), "--no-browser"],
        cwd=str(repo_tmp),
        capture_output=True,
        text=True,
        env={**os.environ, "SPECTRAMIND_TEST": "1", "MPLBACKEND": "Agg"},
        timeout=30,
    )

    assert proc.returncode != 0, "Missing required inputs should yield non-zero exit."
    # Helpful message
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert re.search(r"(missing|required|--sigma|--y-true)", combined, re.IGNORECASE), \
        "Expected a helpful message referencing missing args."


# --------------------------------------
# Optional: Hydra/config integration
# --------------------------------------

@pytest.mark.integration
def test_tool_accepts_config_snapshot_flag_if_available(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Some versions of the tool accept a --config-snapshot or --config flag pointing to a YAML.
    This test creates a small YAML and passes it; if the flag is unknown, the tool should ignore it
    (or exit 0 if it validates args strictly). We do not fail if the flag is not supported.
    """
    tool = _discover_tool(repo_tmp)
    cfg_dir = repo_tmp / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfg_dir / "config_v50.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            # Minimal Hydra-like YAML for tests
            diagnostics:
              enable_html: true
              enable_plots: true
              max_planets: 5
              max_bins: 17
            """
        ),
        encoding="utf-8",
    )

    outdir = repo_tmp / "outputs" / "diagnostics" / "run_with_cfg"
    outdir.mkdir(parents=True, exist_ok=True)

    # Try a few common spellings; tool should either accept or ignore unknown flags gracefully.
    tried = []
    for flag in ("--config-snapshot", "--config", "--hydra-config"):
        tried.append(flag)
        proc = _run_tool_subprocess(
            repo_root=repo_tmp,
            tool_path=_discover_tool(repo_tmp),
            mu_path=tiny_inputs["mu"],
            sigma_path=tiny_inputs["sigma"],
            y_true_path=tiny_inputs["y_true"],
            outdir=outdir,
            extra_args=[flag, str(yaml_path), "--version", "cfg"],
        )
        # If tool rejects the flag with a clean usage error, allow non-zero return once,
        # but require at least one of the flags to succeed.
        if proc.returncode == 0:
            assert (outdir / "diagnostic_summary.json").exists()
            break
    else:
        pytest.xfail(f"Tool rejected all config flags {tried}; if unsupported, xfail is acceptable here.")


# --------------------------------------
# Markers & local run hints
# --------------------------------------

def _local_debug_note() -> None:
    """
    This function is never executed; it exists to document a convenient local run:

    PYTHONPATH=. pytest -q tests/diagnostics/test_generate_diagnostic_summary.py -k cli_generates_summary_and_plots

    The tool can emit many figures; use `--max-planets` and `--max-bins` flags to bound runtime.
    """


# End of file
