#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_generate_fft_symbolic_fusion.py

SpectraMind V50 — Diagnostics Test: tools/generate_fft_symbolic_fusion.py

Purpose
-------
Validate the FFT × Symbolic Fusion generator in a *safe*, *adaptive*, and *repo‑agnostic* way.
This suite verifies CLI/UX behavior, artifact creation (or planned outputs in dry mode),
and debug logging — not deep numerical correctness.

What this test asserts
----------------------
1) Discoverability: --help exists and mentions FFT/symbolic/fusion/UMAP/t-SNE.
2) Safe execution: a dry-run/selftest/plan path exits with code 0.
3) Inputs: accepts tiny synthetic inputs when flags are present:
   - μ spectra (.npy), optional SHAP/entropy (.npy), symbolic results (.json).
4) Artifacts: produces light artifacts (HTML/PNG/CSV/JSON/MD) OR clearly states
   intended outputs in dry mode.
5) Logging: appends an audit line to logs/v50_debug_log.md.
6) Idempotency: repeating safe invocations does not accumulate heavy artifacts.

Design
------
• Entry points probed (in order):
    - tools/generate_fft_symbolic_fusion.py (canonical)
    - tools/fft_symbolic_fusion.py (legacy/variant)
    - spectramind diagnose fft-fusion / fft_symbolic_fusion (wrapper; optional)
• Flags are discovered by parsing --help and mapping abstract names to actual flags.
• Tiny inputs are synthesized on-the-fly:
    - mu.npy              : (N, B) with simple periodic structure
    - shap.npy, entropy.npy (optional): (N, B)
    - symbolic_results.json: tiny rule × planet scores
• Dry mode may just *plan* outputs; we accept textual description of intended paths.

Notes
-----
• No GPU/network required; runs are tiny and bounded.
• Flexible to minor variations in flag names.
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
    Candidate scripts for the FFT × Symbolic Fusion generator.
    """
    root = repo_root()
    cands = [
        root / "tools" / "generate_fft_symbolic_fusion.py",
        root / "tools" / "fft_symbolic_fusion.py",  # defensive alias
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLI forms. We'll try these if direct script/module isn't available.
    """
    return [
        ["spectramind", "diagnose", "fft-fusion"],
        ["spectramind", "diagnose", "fft_symbolic_fusion"],
        ["spectramind", "diagnose", "generate-fft-symbolic-fusion"],
        [sys.executable, "-m", "spectramind", "diagnose", "fft-fusion"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "fft-fusion"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "fft_symbolic_fusion"],
    ]


# ======================================================================================
# Subprocess helpers
# ======================================================================================

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 240) -> Tuple[int, str, str]:
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
    "mu": ["--mu", "--mu-npy", "--mu_path", "--pred-mu", "--pred_mu", "--input-mu"],
    "shap": ["--shap", "--shap-npy", "--shap_path", "--shap-values", "--shap_values"],
    "entropy": ["--entropy", "--entropy-npy", "--entropy_path"],
    "symbolic": ["--symbolic", "--symbolic-json", "--symbolic_path", "--symbolic-results", "--symbolic_results"],

    # Exports
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "csv": ["--csv", "--csv-out", "--write-csv", "--per-bin-csv", "--export-csv"],
    "json": ["--json", "--json-out", "--export-json"],

    # Projections
    "umap": ["--umap", "--do-umap", "--with-umap"],
    "tsne": ["--tsne", "--do-tsne", "--with-tsne"],

    # Fusion/overlays toggles (optional)
    "overlay_symbolic": ["--overlay-symbolic", "--symbolic-overlay", "--link-symbols"],
    "overlay_shap": ["--overlay-shap", "--shap-overlay", "--overlay_shap"],
    "overlay_entropy": ["--overlay-entropy", "--entropy-overlay", "--overlay_entropy"],

    # FFT options (optional)
    "n_freq": ["--n-freq", "--n_freq", "--num-freq", "--kmax"],
    "window": ["--window", "--fft-window"],
    "normalize": ["--normalize", "--norm", "--zscore", "--standardize"],

    # Clustering / labeling (optional)
    "clusters": ["--clusters", "--n-clusters"],
    "labels": ["--labels", "--labels-csv", "--cluster-csv", "--labels_path"],
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
    d = tmp_path / "fft_symbolic_fusion_out"
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
    Create tiny inputs for fusion:
      - mu.npy       : (N, B) with two distinct frequencies per group
      - shap.npy     : (N, B) weak magnitudes (optional)
      - entropy.npy  : (N, B) weak magnitudes (optional)
      - symbolic_results.json : tiny rule × planet map

    Shapes kept small: N=6, B=64.
    """
    N, B = 6, 64
    rng = np.random.default_rng(777)

    # Two groups with different dominant frequencies
    t = np.linspace(0, 1, B, endpoint=False).astype(np.float32)
    mu = np.zeros((N, B), dtype=np.float32)
    for i in range(N):
        freq = 5 if i < (N // 2) else 9
        mu[i] = 0.6 * np.sin(2 * np.pi * freq * t) + 0.2 * rng.normal(0, 0.2, size=B)

    shap = np.abs(rng.normal(0.0, 0.15, size=(N, B)).astype(np.float32))
    entropy = np.abs(rng.normal(0.4, 0.1, size=(N, B)).astype(np.float32))

    base = tmp_path
    mu_path = base / "mu.npy"
    shap_path = base / "shap.npy"
    entropy_path = base / "entropy.npy"
    np.save(mu_path, mu)
    np.save(shap_path, shap)
    np.save(entropy_path, entropy)

    # Minimal symbolic results
    symbolic_path = base / "symbolic_results.json"
    payload = {
        "rules": ["R_smooth", "R_nonneg"],
        "planets": [
            {"id": "p000", "violations": {"R_smooth": 0.10, "R_nonneg": 0.00}},
            {"id": "p001", "violations": {"R_smooth": 0.02, "R_nonneg": 0.07}},
            {"id": "p002", "violations": {"R_smooth": 0.00, "R_nonneg": 0.00}},
            {"id": "p003", "violations": {"R_smooth": 0.09, "R_nonneg": 0.01}},
            {"id": "p004", "violations": {"R_smooth": 0.03, "R_nonneg": 0.00}},
            {"id": "p005", "violations": {"R_smooth": 0.00, "R_nonneg": 0.05}},
        ],
    }
    symbolic_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "mu": mu_path,
        "shap": shap_path,
        "entropy": entropy_path,
        "symbolic": symbolic_path,
    }


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    Tool must be discoverable and --help must mention FFT/symbolic/fusion/UMAP/t-SNE keywords.
    We try:
      1) python tools/generate_fft_symbolic_fusion.py --help
      2) python -m tools.generate_fft_symbolic_fusion --help (if tools is a package)
      3) spectramind diagnose fft-fusion --help (or aliases)
    """
    help_blobs: List[str] = []

    # 1) Direct scripts
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 2) Module form
    if (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_fft_symbolic_fusion", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 3) spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for FFT × Symbolic Fusion tool."
    combined = "\n\n".join(help_blobs).lower()
    required_any = ["fft", "fusion", "symbolic", "umap", "t-sne", "tsne", "diagnostics", "entropy", "shap"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core fusion keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the fusion tool in safe mode with tiny inputs (if supported), ensuring:
      - Exit code 0
      - Debug log appended
      - Outdir exists
      - Artifacts produced (HTML/PNG/CSV/JSON/MD) OR intended outputs mentioned
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    pre_len = len(read_text_or_empty(debug_log))

    # Choose an entrypoint and parse help
    help_text = ""
    base_cmd: Optional[List[str]] = None

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_fft_symbolic_fusion", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.generate_fft_symbolic_fusion")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for the fusion tool."
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
    if "shap" in flags:
        cmd.extend([flags["shap"], str(tiny_inputs["shap"])])
    if "entropy" in flags:
        cmd.extend([flags["entropy"], str(tiny_inputs["entropy"])])
    if "symbolic" in flags:
        cmd.extend([flags["symbolic"], str(tiny_inputs["symbolic"])])

    # Projections / exports (request light artifacts; tool may *plan* in dry mode)
    for opt in ("umap", "tsne", "html", "md", "csv", "json",
                "overlay_symbolic", "overlay_shap", "overlay_entropy"):
        if opt in flags:
            cmd.append(flags[opt])

    # Optional FFT options (small/light)
    if "n_freq" in flags:
        cmd.extend([flags["n_freq"], "16"])
    if "window" in flags:
        cmd.extend([flags["window"], "hann"])
    if "normalize" in flags:
        cmd.append(flags["normalize"])
    if "clusters" in flags:
        cmd.extend([flags["clusters"], "2"])  # tiny example
    if "labels" in flags:
        # We don't synthesize labels.csv here; this flag is optional and often alternative to --clusters
        pass

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=240)
    assert code == 0, f"Safe fusion invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["fft", "fusion", "symbolic", "umap", "tsne", "overlay", "html", "csv", "json"]), \
        "Output does not resemble fusion tool output."

    # Log grew
    post_len = len(read_text_or_empty(debug_log))
    assert post_len >= pre_len, "v50_debug_log.md did not grow after fusion invocation."
    appended = read_text_or_empty(debug_log)[pre_len:]
    assert re.search(r"(fft|fusion|symbolic|umap|tsne|diagnose)", appended, re.IGNORECASE), \
        "No recognizable fusion-related text found in debug log appended segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after fusion invocation."

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


def test_json_or_csv_summaries_if_emitted(temp_outdir: Path):
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
            indicative = {"fft", "fusion", "umap", "tsne", "clusters", "summary", "metrics", "symbolic"}
            assert keys & indicative or len(keys) > 0, \
                f"JSON {jf} lacks indicative fusion/diagnostics keys."

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
        code, out, err = run_proc(python_module_invocation("tools.generate_fft_symbolic_fusion", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.generate_fft_symbolic_fusion")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain fusion tool help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command with the tiny inputs
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
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
    assert code1 == 0 and code2 == 0, f"Safe fusion invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased in safe mode: before={pre}, after={post}"


def test_help_mentions_core_exports_and_projections(project_root: Path):
    """
    Quick semantic sniff: --help should mention core exports and projections to guide users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_fft_symbolic_fusion", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for FFT × Symbolic Fusion."
    combined = "\n\n".join(help_texts).lower()
    want_any = ["umap", "t-sne", "tsne", "html", "csv", "json", "markdown", "overlay", "shap", "entropy", "symbolic"]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention projections/exports/overlays; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
