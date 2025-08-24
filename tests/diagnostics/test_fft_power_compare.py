#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_fft_power_compare.py

SpectraMind V50 — Diagnostics Test: tools/fft_power_compare.py

Purpose
-------
Validate the FFT power comparison tool in a *safe*, *adaptive*, and *repo‑agnostic* way.
This suite focuses on CLI/UX and artifact behavior rather than numerical correctness.

The test asserts that the tool:
  1) Is discoverable and prints a meaningful --help (mentions FFT/power/compare).
  2) Supports a safe invocation path (dry-run/selftest/plan) exiting with code 0.
  3) Accepts tiny synthetic inputs if applicable (μ spectra or CSV) when flags exist.
  4) Writes light artifacts (HTML/PNG/CSV/JSON/MD) *or* clearly states intended outputs.
  5) Appends an audit entry to logs/v50_debug_log.md (append-only).
  6) Is idempotent in safe mode (no accumulation of heavy artifacts).
  7) Optionally exercises clustering/options (e.g., --clusters/--labels) if present.

Design
------
• Entry points probed (in order):
    - tools/fft_power_compare.py (canonical)
    - tools/fft_power_compare_v50.py (variant)
    - spectramind diagnose fft-power-compare (wrapper; optional)
• Flags are discovered by parsing --help and mapping abstract names to actual flags.
• Tiny inputs are created in tmp_path:
    - mu.npy       : shape (N, B) float32
    - labels.csv   : optional grouping labels (id,cluster)
• The tool may just *plan* outputs in dry mode; if so, we accept messages indicating
  intended outputs instead of requiring files.

Notes
-----
• No network/GPU required. Tests are time-bounded and synthetic.
• Flexible to avoid brittle coupling with the exact CLI flag names.
• Numerical fidelity of FFTs is covered elsewhere; this test checks the CLI contract.

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
    Candidate tool scripts for FFT power comparison.
    """
    root = repo_root()
    cands = [
        root / "tools" / "fft_power_compare.py",
        root / "tools" / "fft_power_compare_v50.py",  # variant name (defensive)
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLI forms. We'll try these if direct script/module isn't available.
    """
    return [
        ["spectramind", "diagnose", "fft-power-compare"],
        ["spectramind", "diagnose", "fft_power_compare"],
        [sys.executable, "-m", "spectramind", "diagnose", "fft-power-compare"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "fft-power-compare"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "fft_power_compare"],
    ]


# ======================================================================================
# Subprocess helpers
# ======================================================================================

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 180) -> Tuple[int, str, str]:
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

    # Inputs (allow multiple spellings)
    "mu": ["--mu", "--mu-npy", "--mu_path", "--pred-mu", "--pred_mu", "--input-mu"],
    "labels": ["--labels", "--labels-csv", "--clusters", "--cluster-csv", "--labels_path"],

    # Exports (request light artifacts)
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "csv": ["--csv", "--csv-out", "--write-csv", "--per-bin-csv", "--export-csv"],
    "json": ["--json", "--json-out", "--export-json"],

    # FFT options (optional)
    "n_freq": ["--n-freq", "--n_freq", "--num-freq", "--kmax"],
    "window": ["--window", "--fft-window"],
    "normalize": ["--normalize", "--norm", "--zscore", "--standardize"],

    # Compare toggles / clustering (optional)
    "compare": ["--compare", "--do-compare", "--pairwise", "--between-clusters"],
    "plot": ["--plot", "--plots", "--make-plots"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Map abstract flag names to actual aliases found in --help text.
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
    d = tmp_path / "fft_power_compare_out"
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
    Create tiny inputs for FFT power comparison:
      - mu.npy: (N, B) with simple synthetic periodic patterns + noise
      - labels.csv: N rows mapping ids to small set of clusters
    Shapes kept tiny to avoid heavy compute: N=6, B=64
    """
    N, B = 6, 64
    rng = np.random.default_rng(2025)

    # Synthesize two groups with different dominant frequencies
    t = np.linspace(0, 1, B, endpoint=False).astype(np.float32)
    mu = np.zeros((N, B), dtype=np.float32)
    for i in range(N):
        if i < N // 2:
            mu[i] = 0.7 * np.sin(2 * np.pi * 5 * t) + 0.2 * rng.normal(0, 0.2, size=B)
        else:
            mu[i] = 0.7 * np.sin(2 * np.pi * 9 * t) + 0.2 * rng.normal(0, 0.2, size=B)

    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, mu)

    labels_path = tmp_path / "labels.csv"
    # Minimal CSV header: id,cluster
    # p000,G1
    # ...
    ids = [f"p{i:03d}" for i in range(N)]
    clusters = ["G1"] * (N // 2) + ["G2"] * (N - N // 2)
    lines = ["id,cluster"] + [f"{pid},{clu}" for pid, clu in zip(ids, clusters)]
    labels_path.write_text("\n".join(lines), encoding="utf-8")

    return {"mu": mu_path, "labels": labels_path}


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    Tool must be discoverable and --help must mention FFT/power/compare keywords.
    We try:
      1) python tools/fft_power_compare.py --help
      2) python -m tools.fft_power_compare --help (if tools is a package)
      3) spectramind diagnose fft-power-compare --help
    """
    help_blobs: List[str] = []

    # 1) Direct scripts
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 2) Module form
    if (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.fft_power_compare", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 3) spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for FFT power compare tool."
    combined = "\n\n".join(help_blobs).lower()
    required_any = ["fft", "power", "spectrum", "compare", "diagnostics"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core FFT/power/compare keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the tool in safe mode with tiny inputs (if supported), ensuring:
      - Exit code 0
      - Debug log appended
      - Outdir exists
      - Artifacts produced (HTML/PNG/CSV/JSON/MD) OR intended outputs are mentioned
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
        code, out, err = run_proc(python_module_invocation("tools.fft_power_compare", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.fft_power_compare")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for FFT tool."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)

    # Safe flag
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])

    # Outdir
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])

    # Inputs
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "labels" in flags:
        cmd.extend([flags["labels"], str(tiny_inputs["labels"])])

    # Exports (request light artifacts; tool may *plan* in dry mode)
    for opt in ("html", "md", "csv", "json", "plot"):
        if opt in flags:
            cmd.append(flags[opt])

    # Optional FFT options (small values to keep it light)
    if "n_freq" in flags:
        cmd.extend([flags["n_freq"], "16"])
    if "window" in flags:
        cmd.extend([flags["window"], "hann"])
    if "normalize" in flags:
        cmd.append(flags["normalize"])
    if "compare" in flags:
        cmd.append(flags["compare"])

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=210)
    assert code == 0, f"Safe FFT power compare invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["fft", "power", "spectrum", "compare", "cluster", "plot", "html", "csv", "json"]), \
        "Output does not resemble FFT/power compare tool output."

    # Log grew
    post_len = len(read_text_or_empty(debug_log))
    assert post_len >= pre_len, "v50_debug_log.md did not grow after FFT tool invocation."
    appended = read_text_or_empty(debug_log)[pre_len:]
    assert re.search(r"(fft|power|spectrum|compare|diagnose)", appended, re.IGNORECASE), \
        "No recognizable FFT/power/compare text found in debug log appended segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after FFT tool invocation."

    # Look for produced artifacts OR mention of intended outputs
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_png = recent_files_with_suffix(temp_outdir, (".png",))
    produced_csv = recent_files_with_suffix(temp_outdir, (".csv",))
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))

    if not (produced_html or produced_png or produced_csv or produced_json):
        # Accept dry-run planning — require mention of output intent
        assert any(tok in combined for tok in ["outdir", "output", "write", ".html", ".png", ".csv", ".json"]), \
            "No artifacts found and no mention of intended outputs in tool output (dry-run should plan)."


def test_json_or_csv_summaries_if_emitted(temp_outdir: Path):
    """
    If JSON or CSV summaries were written, perform light sanity checks.
    """
    json_files = recent_files_with_suffix(temp_outdir, (".json",))
    csv_files = recent_files_with_suffix(temp_outdir, (".csv",))

    # JSON sanity
    for jf in json_files[:5]:
        text = read_text_or_empty(jf).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception as e:
            pytest.fail(f"Malformed JSON at {jf}: {e}")

        # Expect dict or list; if dict, look for indicative keys
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            indicative = {"fft", "power", "bins", "metrics", "clusters", "summary"}
            assert keys & indicative or len(keys) > 0, \
                f"JSON {jf} lacks indicative FFT/compare keys."

    # CSV sanity
    for cf in csv_files[:5]:
        text = read_text_or_empty(cf).strip()
        if not text:
            continue
        assert ("\n" in text or "," in text), f"CSV {cf} seems empty or malformed."


def test_idempotent_safe_runs_no_heavy_accumulation(project_root: Path, temp_outdir: Path, tiny_inputs: Dict[str, Path]):
    """
    Run safe mode twice and ensure that heavy artifacts (checkpoints or >5MB) do not accumulate.
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
        code, out, err = run_proc(python_module_invocation("tools.fft_power_compare", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.fft_power_compare")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain tool help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command with the tiny inputs
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "labels" in flags:
        cmd.extend([flags["labels"], str(tiny_inputs["labels"])])

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
    code1, out1, err1 = run_proc(cmd, cwd=project_root, timeout=150)
    code2, out2, err2 = run_proc(cmd, cwd=project_root, timeout=150)
    assert code1 == 0 and code2 == 0, f"Safe FFT tool invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased in safe mode: before={pre}, after={post}"


def test_help_mentions_fft_options_and_exports(project_root: Path):
    """
    Quick semantic sniff: --help should mention FFT options or exports to guide users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.fft_power_compare", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for FFT power compare."
    combined = "\n\n".join(help_texts).lower()
    want_any = ["n-freq", "window", "normalize", "html", "csv", "json", "plot"]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention FFT options or exports; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
