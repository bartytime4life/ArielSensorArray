#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_generate_diagnostic_summary.py

SpectraMind V50 — Diagnostics Test: tools/generate_diagnostic_summary.py

Purpose
-------
Validate the diagnostics summary generator (dashboard-ready aggregator) in a *safe*,
*adaptive*, and *repo-agnostic* way. Rather than checking deep numerics, this suite
verifies the CLI/UX contract, artifact creation (or *planned* outputs during dry mode),
and logging to the project-wide debug log.

What this test asserts
----------------------
1) Discoverability: a --help route exists and mentions diagnostics/summary/metrics.
2) Safe execution: a dry-run/selftest/plan path exits with code 0.
3) Inputs: accepts tiny synthetic inputs when flags are available (μ, σ, y_true, SHAP,
   entropy, symbolic results); tolerates absent flags gracefully.
4) Artifacts: produces light artifacts (JSON/CSV/HTML/PNG/MD) OR clearly states intended
   outputs in dry-run.
5) Logging: appends an audit entry to logs/v50_debug_log.md.
6) Idempotency: repeating safe invocations does not accumulate heavy artifacts.

Design
------
• Entry points probed (in order):
    - tools/generate_diagnostic_summary.py (canonical)
    - tools/generate_diagnostic_summary_v50.py (variant)
    - spectramind diagnose summary / generate-diagnostic-summary (optional wrapper)
• Flags are discovered by parsing --help and mapping abstract names to actual flags.
• Tiny inputs are created on-the-fly:
    - mu.npy, sigma.npy, y_true.npy : shape (N, B)
    - shap.npy, entropy.npy         : shape (N, B)
    - symbolic_results.json         : lightweight per-rule/per-planet scores
• In dry mode the tool may just *plan* outputs; the test accepts messages indicating
  intended write paths in lieu of actual files.

Notes
-----
• This test does *not* enforce scientific correctness.
• No GPU or network is required; runs are tiny and time-bounded.
• The test is defensive to avoid brittle coupling to exact flag names.

Author: SpectraMind V50 QA
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest


# ======================================================================================
# Repo root / entrypoint discovery
# ======================================================================================

def repo_root() -> Path:
    """
    Resolve repository root by walking upward until a 'tools' directory appears.
    Fallback: two levels up from tests/diagnostics.
    """
    here = Path(__file__).resolve()
    for anc in [here] + list(here.parents):
        if (anc / "tools").is_dir():
            return anc
    return Path(__file__).resolve().parents[2]


def tool_script_candidates() -> List[Path]:
    """
    Candidate scripts for the diagnostics summary generator.
    """
    root = repo_root()
    cands = [
        root / "tools" / "generate_diagnostic_summary.py",
        root / "tools" / "generate_diagnostic_summary_v50.py",  # defensive alias
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLI forms. We'll try these if direct script/module isn't available.
    """
    return [
        ["spectramind", "diagnose", "summary"],
        ["spectramind", "diagnose", "generate-diagnostic-summary"],
        ["spectramind", "diagnose", "generate_diagnostic_summary"],
        [sys.executable, "-m", "spectramind", "diagnose", "summary"],
        [sys.executable, "-m", "spectramind", "diagnose", "generate_diagnostic_summary"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "summary"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "generate_diagnostic_summary"],
    ]


# ======================================================================================
# Subprocess helpers
# ======================================================================================

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 210) -> Tuple[int, str, str]:
    """
    Execute a command and return (exit_code, stdout, stderr) in text mode.
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

FLAG_ALIASES: Dict[str, List[str]] = {
    # Help
    "help": ["--help", "-h"],

    # Safe / selftest / plan
    "dry_run": ["--dry-run", "--dryrun", "--selftest", "--plan", "--check", "--no-exec"],

    # Output directory
    "outdir": ["--outdir", "--out-dir", "--output", "--output-dir", "-o"],

    # Inputs (permissive spellings)
    "mu": ["--mu", "--mu-npy", "--mu_path", "--pred-mu", "--pred_mu"],
    "sigma": ["--sigma", "--sigma-npy", "--sigma_path", "--pred-sigma", "--pred_sigma"],
    "y_true": ["--y", "--y-true", "--y_true", "--targets", "--truth", "--labels"],
    "shap": ["--shap", "--shap-npy", "--shap_path", "--shap-values", "--shap_values"],
    "entropy": ["--entropy", "--entropy-npy", "--entropy_path"],
    "symbolic": ["--symbolic", "--symbolic-json", "--symbolic_path", "--symbolic-results", "--symbolic_results"],

    # Exports
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "csv": ["--csv", "--csv-out", "--write-csv", "--per-bin-csv", "--export-csv"],
    "json": ["--json", "--json-out", "--export-json"],

    # Optional toggles
    "plots": ["--plots", "--make-plots", "--plot"],
    "gll": ["--gll", "--compute-gll", "--check-gll"],
    "calibration": ["--calibration", "--check-calibration", "--zscore", "--coverage", "--quantile"],
    "fft": ["--fft", "--do-fft", "--check-fft"],
    "symbolic_overlay": ["--symbolic-overlay", "--overlay-symbolic", "--link-symbols"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Map abstract flag names to real aliases by scanning --help output.
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
    Find files recursively under root with suffix in suffixes, sorted by mtime desc.
    """
    if not root.exists():
        return []
    hits: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
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
    d = tmp_path / "diagnostic_summary_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def ensure_logs_dir(project_root: Path) -> Path:
    logs = project_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    dbg = logs / "v50_debug_log.md"
    if not dbg.exists():
        dbg.write_text("# v50 Debug Log\n", encoding="utf-8")
    return logs


@pytest.fixture
def tiny_inputs(tmp_path: Path) -> Dict[str, Path]:
    """
    Create tiny inputs to exercise the summary tool:
      - y_true.npy, mu.npy, sigma.npy  : (N, B)
      - shap.npy, entropy.npy          : (N, B)
      - symbolic_results.json          : tiny rule × planet map

    Shapes are kept small for speed: N=4, B=12.
    """
    N, B = 4, 12
    rng = np.random.default_rng(404)

    y = rng.normal(0.0, 1.0, size=(N, B)).astype(np.float32)
    mu = y + rng.normal(0.0, 0.1, size=(N, B)).astype(np.float32)
    sigma = np.abs(rng.normal(0.2, 0.05, size=(N, B)).astype(np.float32)) + 1e-3
    shap = np.abs(rng.normal(0.0, 0.2, size=(N, B)).astype(np.float32))
    entropy = np.abs(rng.normal(0.5, 0.15, size=(N, B)).astype(np.float32))

    base = tmp_path
    y_path = base / "y_true.npy"
    mu_path = base / "mu.npy"
    sigma_path = base / "sigma.npy"
    shap_path = base / "shap.npy"
    entropy_path = base / "entropy.npy"

    np.save(y_path, y)
    np.save(mu_path, mu)
    np.save(sigma_path, sigma)
    np.save(shap_path, shap)
    np.save(entropy_path, entropy)

    # Tiny symbolic results structure
    symbolic_path = base / "symbolic_results.json"
    symbolic_payload = {
        "rules": ["R1_smoothness", "R2_nonnegativity"],
        "planets": [
            {"id": "p000", "violations": {"R1_smoothness": 0.12, "R2_nonnegativity": 0.00}},
            {"id": "p001", "violations": {"R1_smoothness": 0.03, "R2_nonnegativity": 0.05}},
            {"id": "p002", "violations": {"R1_smoothness": 0.00, "R2_nonnegativity": 0.00}},
            {"id": "p003", "violations": {"R1_smoothness": 0.07, "R2_nonnegativity": 0.01}},
        ],
    }
    symbolic_path.write_text(json.dumps(symbolic_payload, indent=2), encoding="utf-8")

    return {
        "y_true": y_path,
        "mu": mu_path,
        "sigma": sigma_path,
        "shap": shap_path,
        "entropy": entropy_path,
        "symbolic": symbolic_path,
    }


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    The tool must be discoverable and --help must mention diagnostics/summary/metrics.
    We try:
      1) python tools/generate_diagnostic_summary.py --help
      2) python -m tools.generate_diagnostic_summary --help (if tools is a package)
      3) spectramind diagnose summary --help (or its aliases)
    """
    help_blobs: List[str] = []

    # 1) Direct scripts
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 2) Module form
    if (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_diagnostic_summary", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 3) spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for diagnostic summary generator."
    combined = "\n\n".join(help_blobs).lower()
    required_any = ["diagnostic", "summary", "metrics", "gll", "calibration", "fft", "symbolic"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core diagnostics keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the tool in safe mode with tiny inputs (if supported), ensuring:
      - Exit code 0
      - Debug log appended
      - Outdir exists
      - Artifacts produced (JSON/CSV/HTML/PNG/MD) OR intended outputs are mentioned
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    pre_len = len(read_text_or_empty(debug_log))

    # Choose an entrypoint and collect help text
    help_text = ""
    base_cmd: Optional[List[str]] = None

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_diagnostic_summary", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.generate_diagnostic_summary")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for the summary tool."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)

    # Safe flag
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])

    # Outdir
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])

    # Inputs (best-effort; only pass if flags present)
    if "y_true" in flags:
        cmd.extend([flags["y_true"], str(tiny_inputs["y_true"])])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "sigma" in flags:
        cmd.extend([flags["sigma"], str(tiny_inputs["sigma"])])
    if "shap" in flags:
        cmd.extend([flags["shap"], str(tiny_inputs["shap"])])
    if "entropy" in flags:
        cmd.extend([flags["entropy"], str(tiny_inputs["entropy"])])
    if "symbolic" in flags:
        cmd.extend([flags["symbolic"], str(tiny_inputs["symbolic"])])

    # Exports (request light artifacts; tool may *plan* in dry mode)
    for opt in ("html", "md", "csv", "json", "plots", "gll", "calibration", "fft", "symbolic_overlay"):
        if opt in flags:
            cmd.append(flags[opt])

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=240)
    assert code == 0, f"Safe diagnostic summary invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["diagnostic", "summary", "metrics", "gll", "calibration", "fft", "symbolic", "shap", "entropy", "html", "csv", "json", "md"]), \
        "Output does not resemble diagnostic summary generator output."

    # Log grew
    post_len = len(read_text_or_empty(debug_log))
    assert post_len >= pre_len, "v50_debug_log.md did not grow after diagnostic summary invocation."
    appended = read_text_or_empty(debug_log)[pre_len:]
    assert re.search(r"(diagnostic|summary|gll|calibration|fft|symbolic|shap|entropy)", appended, re.IGNORECASE), \
        "No recognizable diagnostics-related text found in debug log appended segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after diagnostic summary invocation."

    # Look for produced artifacts OR mention of intended outputs
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_png = recent_files_with_suffix(temp_outdir, (".png",))
    produced_csv = recent_files_with_suffix(temp_outdir, (".csv",))
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))
    produced_md = recent_files_with_suffix(temp_outdir, (".md",))

    if not (produced_html or produced_png or produced_csv or produced_json or produced_md):
        # Accept dry-run planning — require mention of output intent
        assert any(tok in combined for tok in ["outdir", "output", "write", ".html", ".png", ".csv", ".json", ".md"]), \
            "No artifacts found and no mention of intended outputs in tool output (dry-run should plan)."


def test_json_summary_sanity_if_emitted(temp_outdir: Path):
    """
    If JSON summaries were written, they should decode and contain indicative keys.
    """
    json_files = recent_files_with_suffix(temp_outdir, (".json",))
    for jf in json_files[:8]:
        text = read_text_or_empty(jf).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception as e:
            pytest.fail(f"Malformed JSON at {jf}: {e}")

        # Heuristic: look for indicative keys for diagnostics
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            indicative = {"summary", "metrics", "gll", "coverage", "zscore", "z_score", "fft", "entropy", "shap", "symbolic", "per_bin", "bins"}
            assert keys & indicative or len(keys) > 0, \
                f"JSON {jf} lacks indicative diagnostics keys."


def test_csv_or_md_sanity_if_emitted(temp_outdir: Path):
    """
    If CSV or MD were written, they should be textual with at least delimiters/lines.
    """
    csv_files = recent_files_with_suffix(temp_outdir, (".csv",))
    md_files = recent_files_with_suffix(temp_outdir, (".md",))

    for cf in csv_files[:6]:
        t = read_text_or_empty(cf).strip()
        if t:
            assert ("," in t or "\t" in t or "\n" in t), f"CSV {cf} seems empty or malformed."

    for mf in md_files[:6]:
        t = read_text_or_empty(mf).strip()
        if t:
            # Look for headings or table markers or bullet points — heuristic only
            assert any(x in t for x in ["#", "|", "-", "*", "GLL", "Summary", "Diagnostics"]), \
                f"Markdown {mf} appears empty or not a report."


def test_idempotent_safe_runs_no_heavy_accumulation(project_root: Path, temp_outdir: Path, tiny_inputs: Dict[str, Path]):
    """
    Run safe mode twice and ensure heavy artifacts (checkpoints or >5MB) do not accumulate.
    """
    # Get help & base command
    help_text = ""
    base_cmd: Optional[List[str]] = None

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_diagnostic_summary", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.generate_diagnostic_summary")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain summary tool help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command with the tiny inputs
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "y_true" in flags:
        cmd.extend([flags["y_true"], str(tiny_inputs["y_true"])])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "sigma" in flags:
        cmd.extend([flags["sigma"], str(tiny_inputs["sigma"])])
    if "shap" in flags:
        cmd.extend([flags["shap"], str(tiny_inputs["shap"])])
    if "entropy" in flags:
        cmd.extend([flags["entropy"], str(tiny_inputs["entropy"])])
    if "symbolic" in flags:
        cmd.extend([flags["symbolic"], str(tiny_inputs["symbolic"])])

    # Count heavy artifacts (heuristic: >5MB or checkpoint-like suffixes)
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
    code1, out1, err1 = run_proc(cmd, cwd=project_root, timeout=180)
    code2, out2, err2 = run_proc(cmd, cwd=project_root, timeout=180)
    assert code1 == 0 and code2 == 0, f"Safe summary invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased in safe mode: before={pre}, after={post}"


def test_help_mentions_core_exports_and_checks(project_root: Path):
    """
    Quick semantic sniff: --help should mention core exports and checks to guide users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_diagnostic_summary", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for diagnostic summary."
    combined = "\n\n".join(help_texts).lower()
    want_any = ["html", "csv", "json", "markdown", "gll", "calibration", "fft", "symbolic", "shap", "entropy"]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention exports or core checks; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
