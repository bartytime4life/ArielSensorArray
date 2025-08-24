#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_calibration_checker.py

SpectraMind V50 — Diagnostics Test: tools/check_calibration.py

This test suite validates the behavior of the calibration checker tool. It is designed
to be *adaptive* to minor CLI variations while enforcing our project standards for:
- Discoverability (`--help` works and mentions calibration).
- Safe execution (dry-run / selftest style) without heavy compute.
- Artifact behavior (writes or clearly *plans* JSON/CSV/HTML/PNG outputs).
- Logging (appends to logs/v50_debug_log.md with recognizable wording).
- Optional inputs (μ, σ, and ground-truth arrays) when flags are available.

The tests are intentionally verbose, defensive, and self-contained. They do not
require GPUs, large datasets, or network access. They prefer a *safe* path using
dry-run/selftest flags when available. If the tool requires tiny input arrays,
we synthesise them as small .npy files.

Repository assumptions (adapted defensively if absent):
- Tool path: tools/check_calibration.py (canonical), with optional aliases.
- Top-level CLI wrapper: `spectramind diagnose calibration` or similar (optional).
- logs/v50_debug_log.md exists or is created by the tool on first run.
- outputs/ is writable. (This test writes to a temp dir.)

What “calibration checker” artifacts may look like (any subset acceptable):
- JSON summaries: coverage metrics, z-score stats, quantile calibration metrics.
- PNG/HTML plots: z-score histogram, per-bin coverage heatmaps, quantile reliability curves.
- CSV: per-bin calibration table.
If a pure dry-run is requested, the tool should at least *report* intended paths.

NOTE: This test does not validate *numerical correctness* of calibration — it validates
the CLI/UX contract and artifact/logging behavior for reproducibility and diagnostics.

Author: SpectraMind V50 QA
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest


# ======================================================================================
# Helpers — repo root / candidate entrypoints
# ======================================================================================

def repo_root() -> Path:
    """
    Heuristically resolve repository root by walking up until a 'tools' directory is found.
    Falls back to two levels up from this test file (tests/diagnostics/...).
    """
    here = Path(__file__).resolve()
    for anc in [here] + list(here.parents):
        if (anc / "tools").is_dir():
            return anc
    return Path(__file__).resolve().parents[2]


def tool_script_candidates() -> List[Path]:
    """
    Candidate script locations for the calibration checker tool.
    Canonical: tools/check_calibration.py
    Aliases supported defensively: tools/calibration_checker.py, tools/calibration_check.py
    """
    root = repo_root()
    cands = [
        root / "tools" / "check_calibration.py",
        root / "tools" / "calibration_checker.py",
        root / "tools" / "calibration_check.py",
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional CLI wrappers. We try these if direct script/module isn't available.
    Prefer diagnose path if present; fall back to generic.
    """
    return [
        ["spectramind", "diagnose", "calibration"],
        ["spectramind", "diagnose", "check-calibration"],
        ["spectramind", "diagnose", "check_calibration"],
        ["spectramind", "calibration"],
        [sys.executable, "-m", "spectramind", "diagnose", "calibration"],
        [sys.executable, "-m", "spectramind", "diagnose", "check_calibration"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "calibration"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "check_calibration"],
    ]


# ======================================================================================
# Subprocess helpers
# ======================================================================================

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 120) -> Tuple[int, str, str]:
    """
    Run a subprocess and capture (exit_code, stdout, stderr). Always text mode.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ},
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out, err


def python_module_invocation(module: str, *args: str) -> List[str]:
    return [sys.executable, "-m", module, *args]


def python_script_invocation(script: Path, *args: str) -> List[str]:
    return [sys.executable, str(script), *args]


# ======================================================================================
# Flag discovery
# ======================================================================================

# Aliases we will search for in --help to adapt to the tool’s actual flags.
FLAG_ALIASES: Dict[str, List[str]] = {
    # Help
    "help": ["--help", "-h"],

    # Safe-run / dry mode / selftest / planning mode
    "dry_run": ["--dry-run", "--dryrun", "--selftest", "--plan", "--check", "--no-exec"],

    # Output directory
    "outdir": ["--outdir", "--out-dir", "--output", "--output-dir", "-o"],

    # Inputs (allow many spellings)
    "mu": ["--mu", "--pred-mu", "--mu-npy", "--mu_path", "--pred_mu"],
    "sigma": ["--sigma", "--pred-sigma", "--sigma-npy", "--sigma_path", "--pred_sigma", "--sigmas"],
    "y_true": ["--y", "--y-true", "--y_true", "--targets", "--labels", "--truth"],

    # Optional behavior flags
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "csv": ["--csv", "--csv-out", "--per-bin-csv", "--write-csv"],

    # Calibration-specific knobs (optional)
    "quantile": ["--quantile", "--check-quantiles", "--quantile-eval"],
    "zscore": ["--zscore", "--check-zscore", "--z-score-eval"],
    "coverage": ["--coverage", "--check-coverage", "--coverage-eval"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Parse --help text to determine which of our abstract flags are supported by the tool.
    Uses word-boundary searches to avoid collisions with descriptions.
    """
    mapping: Dict[str, str] = {}
    for abstract, aliases in FLAG_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(^|\s){re.escape(alias)}(\s|,|$)", help_text):
                mapping[abstract] = alias
                break
    return mapping


# ======================================================================================
# Artifact probing
# ======================================================================================

def recent_files_with_suffix(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    """
    Collect files recursively in root whose suffix is in suffixes, ordered by mtime desc.
    """
    if not root.exists():
        return []
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(p)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def find_any_pathname_match(root: Path, patterns: List[str]) -> List[Path]:
    """
    Find files under root with a filename containing any of the substr patterns (case-insensitive).
    """
    hits: List[Path] = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if p.is_file() and any(s.lower() in p.name.lower() for s in patterns):
            hits.append(p)
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture(scope="module")
def project_root() -> Path:
    return repo_root()


@pytest.fixture
def temp_outdir(tmp_path: Path) -> Path:
    """
    Dedicated temp output directory for this test run.
    """
    d = tmp_path / "calibration_checker_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def ensure_logs_dir(project_root: Path) -> Path:
    """
    Ensure logs dir + debug log exist to allow logging assertions.
    """
    logs = project_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    dbg = logs / "v50_debug_log.md"
    if not dbg.exists():
        dbg.write_text("# v50 Debug Log\n", encoding="utf-8")
    return logs


@pytest.fixture
def tiny_inputs(tmp_path: Path) -> Dict[str, Path]:
    """
    Generate tiny, well-formed μ, σ, and y_true arrays to feed the tool when
    it supports explicit input flags. Arrays are small to avoid heavy compute
    and to keep runs deterministic.

    Shapes:
      - y_true: (N, B)
      - mu:     (N, B)
      - sigma:  (N, B) with small positive values
    """
    N, B = 3, 7
    rng = np.random.default_rng(42)

    y = rng.normal(loc=0.0, scale=1.0, size=(N, B)).astype(np.float32)
    mu = y + rng.normal(loc=0.0, scale=0.1, size=(N, B)).astype(np.float32)
    sigma = np.abs(rng.normal(loc=0.2, scale=0.05, size=(N, B)).astype(np.float32)) + 1e-3

    y_path = tmp_path / "y_true.npy"
    mu_path = tmp_path / "mu.npy"
    sigma_path = tmp_path / "sigma.npy"
    np.save(y_path, y)
    np.save(mu_path, mu)
    np.save(sigma_path, sigma)

    return {"y_true": y_path, "mu": mu_path, "sigma": sigma_path}


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    The tool must be discoverable and provide a meaningful --help output mentioning calibration.
    We try multiple entrypoints:
      1) python tools/check_calibration.py --help
      2) python -m tools.check_calibration --help (if tools is a package)
      3) spectramind diagnose calibration --help (optional wrapper)
    """
    help_blobs: List[str] = []

    # 1) Direct script
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 2) Module form
    if (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.check_calibration", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 3) spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for calibration checker."
    combined = "\n\n".join(help_blobs).lower()
    assert any(k in combined for k in ["calibration", "z-score", "coverage", "quantile"]), \
        "Help output does not appear to describe a calibration checker."


def test_safe_invocation_and_artifact_behavior(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the tool in safe mode (dry/selftest if available). If the tool supports input
    flags for μ/σ/y, pass tiny arrays. Validate:
      - Exit code 0
      - Logging appended to logs/v50_debug_log.md
      - Output directory exists
      - Either files are produced (PNG/HTML/JSON/CSV) OR the tool reports intended paths.

    We adapt to the tool's actual flags by parsing --help first.
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    pre_len = len(read_text_or_empty(debug_log))

    # Obtain help and choose a working base command.
    help_text = ""
    base_cmd: Optional[List[str]] = None

    # Try direct script
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    # Try module
    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.check_calibration", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.check_calibration")

    # Try spectramind wrapper
    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for calibration checker."
    flags = discover_supported_flags(help_text)

    # Prepare command with safe flags and outputs
    cmd = list(base_cmd)

    # Safe mode
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])

    # Output dir
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])

    # Optional inputs
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "sigma" in flags:
        cmd.extend([flags["sigma"], str(tiny_inputs["sigma"])])
    if "y_true" in flags:
        cmd.extend([flags["y_true"], str(tiny_inputs["y_true"])])

    # Request exports if available (lightweight or planned)
    if "html" in flags:
        cmd.append(flags["html"])
    if "md" in flags:
        cmd.append(flags["md"])
    if "csv" in flags:
        cmd.append(flags["csv"])
    if "quantile" in flags:
        cmd.append(flags["quantile"])
    if "zscore" in flags:
        cmd.append(flags["zscore"])
    if "coverage" in flags:
        cmd.append(flags["coverage"])

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=180)
    assert code == 0, f"Calibration checker safe invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()

    # Logging must be appended
    post_len = len(read_text_or_empty(debug_log))
    assert post_len >= pre_len, "v50_debug_log.md did not grow after calibration checker invocation."
    appended = read_text_or_empty(debug_log)[pre_len:]
    assert re.search(r"(calibration|check_calibration|z[- ]?score|coverage|quantile)", appended, re.IGNORECASE), \
        "No recognizable calibration-related text found in the appended debug log segment."

    # Output dir should exist
    assert temp_outdir.exists(), "Output directory missing after invocation."

    # Check for artifacts OR at least planned outputs in stdout/stderr
    # Look for likely JSON/CSV/HTML/PNG artifacts
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))
    produced_csv = recent_files_with_suffix(temp_outdir, (".csv",))
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_png = recent_files_with_suffix(temp_outdir, (".png",))

    # Also try filename-based discovery for common names
    named_hits = find_any_pathname_match(temp_outdir, [
        "coverage", "zscore", "z-score", "quantile", "reliability", "calibration", "per_bin"
    ])

    if not (produced_json or produced_csv or produced_html or produced_png or named_hits):
        # Accept dry-run: must *mention* intended outputs
        assert any(tok in combined for tok in ["outdir", "output", "write", "would", ".json", ".csv", ".html", ".png"]), \
            "No artifacts found and no mention of intended outputs in tool output."


def test_json_summary_sanity_if_emitted(temp_outdir: Path):
    """
    If a JSON summary was produced, validate that it decodes and contains at least
    one recognizable calibration key. We allow flexibility and only enforce *some*
    structure if present.
    """
    json_files = recent_files_with_suffix(temp_outdir, (".json",))
    for jf in json_files[:5]:
        text = read_text_or_empty(jf).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception as e:
            pytest.fail(f"Malformed JSON at {jf}: {e}")

        # Minimal sanity assertions
        assert isinstance(obj, (dict, list)), f"Unexpected JSON top-level type in {jf}"

        if isinstance(obj, dict):
            lower_keys = {str(k).lower() for k in obj.keys()}
            # Heuristic set of expected keys (any subset is fine).
            expected_any = {"coverage", "zscore", "z_score", "quantile", "per_bin", "bins", "summary", "metrics"}
            assert lower_keys & expected_any, \
                f"JSON {jf} lacks recognizable calibration keys (found {sorted(lower_keys)[:10]})."


def test_idempotent_safe_runs_do_not_accumulate_heavy_artifacts(project_root: Path, temp_outdir: Path, tiny_inputs: Dict[str, Path]):
    """
    Invoke the tool twice in safe mode and confirm heavy artifacts (e.g., >5MB or .ckpt/.pt)
    do not accumulate. If none exist at all, the test passes trivially.
    """
    # Resolve entrypoint and flags
    help_text = ""
    base_cmd: Optional[List[str]] = None

    # script
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    # module
    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.check_calibration", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.check_calibration")

    # wrapper
    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain a working entrypoint for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "sigma" in flags:
        cmd.extend([flags["sigma"], str(tiny_inputs["sigma"])])
    if "y_true" in flags:
        cmd.extend([flags["y_true"], str(tiny_inputs["y_true"])])

    # Count heavy artifacts in outdir (heuristic)
    def count_heavy(root: Path) -> int:
        if not root.exists():
            return 0
        n = 0
        for p in root.rglob("*"):
            if p.is_file():
                if p.suffix.lower() in {".ckpt", ".pt"} or p.stat().st_size > 5 * 1024 * 1024:
                    n += 1
        return n

    pre = count_heavy(temp_outdir)
    code1, out1, err1 = run_proc(cmd, cwd=project_root, timeout=150)
    code2, out2, err2 = run_proc(cmd, cwd=project_root, timeout=150)
    assert code1 == 0 and code2 == 0, f"Safe calibration runs failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased between safe runs: before={pre}, after={post}"


def test_stdout_mentions_core_concepts(project_root: Path):
    """
    A quick semantic sniff test: the tool's --help should mention at least one of the
    core calibration concepts to aid discoverability for users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.check_calibration", "--help"), cwd=project_root)
        if code == 0:
            help_texts.append(out + "\n" + err)

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for calibration checker."
    combined = "\n\n".join(help_texts).lower()

    # Require at least one calibration domain concept in help for user guidance.
    required_any = ["calibration", "coverage", "z-score", "zscore", "quantile", "reliability"]
    assert any(tok in combined for tok in required_any), \
        f"Help lacks core calibration keywords. Expected any of {required_any}."


# ======================================================================================
# End of file
# ======================================================================================
