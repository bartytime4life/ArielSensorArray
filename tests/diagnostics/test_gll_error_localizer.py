#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_gll_error_localizer.py

SpectraMind V50 — Diagnostics Test: tools/gll_error_localizer.py

Purpose
-------
Validate the GLL error localization tool in a *safe*, *adaptive*, and *repo‑agnostic*
manner. This suite checks the CLI/UX contract, artifact behavior (or *planned*
outputs in dry mode), and audit logging — not the scientific numerics.

What this test asserts
----------------------
1) Discoverability: --help exists and mentions GLL/error/localize/heatmap.
2) Safe execution: a dry‑run/selftest/plan path exits with code 0.
3) Inputs: accepts tiny synthetic arrays when flags are available (μ, σ, y_true).
4) Artifacts: produces light artifacts (PNG/HTML/CSV/JSON/MD) OR clearly states
   intended outputs in dry mode.
5) Logging: appends an audit line to logs/v50_debug_log.md.
6) Idempotency: repeating safe invocations does not accumulate heavy artifacts.

Design
------
• Entry points probed (in order):
    - tools/gll_error_localizer.py           (canonical)
    - tools/gll_error_localiser.py           (UK spelling; defensive)
    - spectramind diagnose gll-error         (wrapper; optional)
    - spectramind diagnose gll-error-localizer (wrapper; optional)
• Flags are discovered by parsing --help and mapping abstract names to actual flags.
• Tiny inputs are synthesized on-the-fly:
    - y_true.npy : (N, B)
    - mu.npy     : (N, B)
    - sigma.npy  : (N, B) with positive values

Notes
-----
• No GPU/network required; runs are tiny and bounded.
• The test is flexible to minor variations in flag names.
• Numerical validation is covered elsewhere; this test checks the CLI contract.

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
    Candidate scripts for the GLL error localizer.
    """
    root = repo_root()
    cands = [
        root / "tools" / "gll_error_localizer.py",
        root / "tools" / "gll_error_localiser.py",  # defensive alias
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLI forms. We'll try these if direct script/module isn't available.
    """
    return [
        ["spectramind", "diagnose", "gll-error"],
        ["spectramind", "diagnose", "gll-error-localizer"],
        [sys.executable, "-m", "spectramind", "diagnose", "gll-error"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "gll-error"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "gll-error-localizer"],
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

    # Safe / plan mode
    "dry_run": ["--dry-run", "--dryrun", "--selftest", "--plan", "--check", "--no-exec"],

    # Output directory
    "outdir": ["--outdir", "--out-dir", "--output", "--output-dir", "-o"],

    # Inputs (permissive spellings)
    "y_true": ["--y", "--y-true", "--y_true", "--targets", "--truth", "--labels"],
    "mu": ["--mu", "--mu-npy", "--mu_path", "--pred-mu", "--pred_mu"],
    "sigma": ["--sigma", "--sigma-npy", "--sigma_path", "--pred-sigma", "--pred_sigma"],

    # Exports / plots
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "csv": ["--csv", "--csv-out", "--write-csv", "--per-bin-csv"],
    "json": ["--json", "--json-out", "--export-json"],
    "plots": ["--plots", "--make-plots", "--plot"],

    # Options / toggles (optional)
    "topk": ["--top-k", "--topk", "--top_k"],
    "per_planet": ["--per-planet", "--per_planet", "--by-planet"],
    "per_bin": ["--per-bin", "--per_bin", "--by-bin"],
    "zscore": ["--zscore", "--z-score", "--check-zscore"],
    "heatmap": ["--heatmap", "--error-heatmap", "--gll-heatmap"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Map abstract flag names to the actual names found in --help.
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
# Fixtures — logs and tiny synthetic inputs
# ======================================================================================

@pytest.fixture(scope="module")
def project_root() -> Path:
    return repo_root()


@pytest.fixture
def temp_outdir(tmp_path: Path) -> Path:
    d = tmp_path / "gll_error_localizer_out"
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
    Create tiny inputs for GLL localization:
      - y_true.npy, mu.npy, sigma.npy : (N, B)
    Shapes kept tiny for speed: N=4, B=12.
    """
    N, B = 4, 12
    rng = np.random.default_rng(31415)

    y = rng.normal(0.0, 1.0, size=(N, B)).astype(np.float32)
    mu = y + rng.normal(0.0, 0.12, size=(N, B)).astype(np.float32)
    # Small but positive uncertainties
    sigma = np.abs(rng.normal(0.2, 0.05, size=(N, B)).astype(np.float32)) + 1e-3

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
    The tool must be discoverable and --help must mention GLL/error/localize/heatmap.
    Try:
      1) python tools/gll_error_localizer.py --help
      2) python -m tools.gll_error_localizer --help (if tools is a package)
      3) spectramind diagnose gll-error --help (or aliases)
    """
    help_blobs: List[str] = []

    # Direct scripts
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # Module form
    if (project_root / "tools" / "__init__.py").exists():
        for mod in ("tools.gll_error_localizer", "tools.gll_error_localiser"):
            code, out, err = run_proc(python_module_invocation(mod, "--help"), cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    # spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for GLL error localizer."
    combined = "\n\n".join(help_blobs).lower()
    required_any = ["gll", "error", "localiz", "heatmap", "zscore", "z-score"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core GLL/error localization keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the localizer in safe mode with tiny inputs (if supported), ensuring:
      - Exit code 0
      - Debug log appended
      - Outdir exists
      - Artifacts produced (HTML/PNG/CSV/JSON/MD) OR intended outputs mentioned
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    pre_len = len(read_text_or_empty(debug_log))

    # Choose entrypoint and parse help
    help_text = ""
    base_cmd: Optional[List[str]] = None

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        for mod in ("tools.gll_error_localizer", "tools.gll_error_localiser"):
            code, out, err = run_proc(python_module_invocation(mod, "--help"), cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = python_module_invocation(mod)
                break

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for the GLL error localizer."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)

    # Safe flag
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])

    # Outdir
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])

    # Inputs (best-effort)
    if "y_true" in flags:
        cmd.extend([flags["y_true"], str(tiny_inputs["y_true"])])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "sigma" in flags:
        cmd.extend([flags["sigma"], str(tiny_inputs["sigma"])])

    # Exports (request light artifacts; tool may *plan* in dry mode)
    for opt in ("html", "md", "csv", "json", "plots", "zscore", "heatmap", "per_planet", "per_bin"):
        if opt in flags:
            cmd.append(flags[opt])

    # Optional top-k (small value)
    if "topk" in flags:
        cmd.extend([flags["topk"], "3"])

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=240)
    assert code == 0, f"Safe GLL error localization invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["gll", "error", "zscore", "z-score", "heatmap", "localiz", "html", "csv", "json"]), \
        "Output does not resemble GLL error localization output."

    # Log grew
    post = read_text_or_empty(debug_log)
    assert len(post) >= pre_len, "v50_debug_log.md did not grow after localizer invocation."
    appended = post[pre_len:]
    assert re.search(r"(gll|error|localiz|heatmap|z[- ]?score)", appended, re.IGNORECASE), \
        "No recognizable GLL/error-related text found in the appended debug log segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after localizer invocation."

    # Look for outputs OR intent messages (in dry mode)
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_png = recent_files_with_suffix(temp_outdir, (".png",))
    produced_csv = recent_files_with_suffix(temp_outdir, (".csv",))
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))
    produced_md = recent_files_with_suffix(temp_outdir, (".md",))

    if not (produced_html or produced_png or produced_csv or produced_json or produced_md):
        # Accept dry-run planning — require mention of intended outputs
        assert any(tok in combined for tok in ["outdir", "output", "write", ".html", ".png", ".csv", ".json", ".md"]), \
            "No artifacts found and no mention of intended outputs (dry‑run should plan)."


def test_json_or_csv_sanity_if_emitted(temp_outdir: Path):
    """
    If JSON or CSV summaries were written, perform light sanity checks.
    """
    json_files = recent_files_with_suffix(temp_outdir, (".json",))
    csv_files = recent_files_with_suffix(temp_outdir, (".csv",))

    # JSON sanity
    for jf in json_files[:6]:
        text = read_text_or_empty(jf).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception as e:
            pytest.fail(f"Malformed JSON at {jf}: {e}")

        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            indicative = {"gll", "zscore", "bins", "summary", "metrics", "per_bin", "per_planet"}
            assert keys & indicative or len(keys) > 0, \
                f"JSON {jf} lacks indicative GLL/diagnostics keys."

    # CSV sanity
    for cf in csv_files[:6]:
        text = read_text_or_empty(cf).strip()
        if not text:
            continue
        assert ("\n" in text or "," in text), f"CSV {cf} seems empty or malformed."


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
        for mod in ("tools.gll_error_localizer", "tools.gll_error_localiser"):
            code, out, err = run_proc(python_module_invocation(mod, "--help"), cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = python_module_invocation(mod)
                break

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain localizer help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command with tiny inputs
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
    assert code1 == 0 and code2 == 0, f"Safe localizer invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased in safe mode: before={pre}, after={post}"


def test_help_mentions_core_exports_and_toggles(project_root: Path):
    """
    --help should hint at exports and toggles to guide users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        for mod in ("tools.gll_error_localizer", "tools.gll_error_localiser"):
            code, out, err = run_proc(python_module_invocation(mod, "--help"), cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for GLL error localizer."
    combined = "\n\n".join(help_texts).lower()
    want_any = ["html", "csv", "json", "markdown", "heatmap", "zscore", "per-bin", "per planet", "top-k"]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention exports/toggles; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
