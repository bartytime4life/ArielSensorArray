#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_plot_tsne_interactive.py

SpectraMind V50 — Diagnostics Test: tools/plot_tsne_interactive.py

Purpose
-------
Validate the interactive t‑SNE latent plot generator in a *safe*, *adaptive*, and
*repo‑agnostic* manner. The suite checks the CLI/UX contract, artifact creation (or
a clear *plan* in dry mode), and logging — not the scientific numerics.

This suite asserts:
  1) Discoverability: --help exists and mentions t‑SNE/interactive/HTML/diagnostics.
  2) Safe execution: a dry-run/selftest/plan path exits with code 0.
  3) Inputs: accepts tiny placeholder inputs when flags are available:
        • latents.npy (N×D)
        • labels.csv  (id,label)
        • ids.csv     (id only)  (optional, if tool supports)
  4) Artifacts: produces HTML/JSON/PNG/MD outputs *or* clearly states intended outputs in dry mode.
  5) Logging: appends an audit line to logs/v50_debug_log.md.
  6) Idempotency: repeating safe invocations does not accumulate heavy artifacts.

Entry points probed (in order):
  • tools/plot_tsne_interactive.py                 (canonical)
  • tools/plot_tsne.py / tools/tsne_interactive.py (defensive aliases)
  • spectramind diagnose tsne‑latents / tsne       (optional wrapper)

Flag discovery is dynamic — we parse --help and map abstract names to actual flags.

Notes
-----
• No GPU/network required; runs are tiny and bounded.
• We do not import sklearn; the test only prepares inputs and exercises the CLI.

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
    Resolve repo root by walking upward until a 'tools' directory appears.
    Fallback: two levels up from tests/diagnostics.
    """
    here = Path(__file__).resolve()
    for anc in [here] + list(here.parents):
        if (anc / "tools").is_dir():
            return anc
    return Path(__file__).resolve().parents[2]


def tool_script_candidates() -> List[Path]:
    """
    Candidate t‑SNE scripts we try to invoke.
    """
    root = repo_root()
    cands = [
        root / "tools" / "plot_tsne_interactive.py",
        root / "tools" / "plot_tsne.py",
        root / "tools" / "tsne_interactive.py",
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLI forms.
    """
    return [
        ["spectramind", "diagnose", "tsne-latents"],
        ["spectramind", "diagnose", "tsne"],
        [sys.executable, "-m", "spectramind", "diagnose", "tsne-latents"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "tsne-latents"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "tsne"],
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

    # Safe / plan
    "dry_run": ["--dry-run", "--dryrun", "--selftest", "--plan", "--check", "--no-exec"],

    # Output dir
    "outdir": ["--outdir", "--out-dir", "--output", "--output-dir", "-o"],

    # Inputs
    "latents": ["--latents", "--latents-npy", "--latents_path", "--embeddings", "--embeddings-npy"],
    "labels": ["--labels", "--labels-csv", "--label-csv", "--cluster-csv", "--labels_path"],
    "ids": ["--ids", "--ids-csv", "--id-csv", "--ids_path"],

    # Exports
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "json": ["--json", "--json-out", "--export-json"],
    "png": ["--png", "--png-out", "--export-png"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "no_open": ["--no-open", "--no-open-html", "--no_open_html"],

    # t-SNE tuning (optional)
    "perplexity": ["--perplexity", "-pplx"],
    "learning_rate": ["--learning-rate", "--lr"],
    "n_iter": ["--n-iter", "--n_iter"],
    "metric": ["--metric"],
    "seed": ["--seed"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Map abstract flag names to actual names found in --help.
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
# Fixtures — logs and tiny inputs
# ======================================================================================

@pytest.fixture(scope="module")
def project_root() -> Path:
    return repo_root()


@pytest.fixture
def temp_outdir(tmp_path: Path) -> Path:
    d = tmp_path / "tsne_interactive_out"
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
    Prepare tiny (N=60, D=8) synthetic latents with three separable blobs and
    a labels.csv with cluster ids, plus an ids.csv file (optional).
    """
    rng = np.random.default_rng(20250824)
    N, D = 60, 8
    K = 3
    per = N // K

    centers = np.stack([
        np.linspace(0.5, 1.0, D),
        np.linspace(-1.0, -0.5, D),
        np.concatenate([np.ones(D // 2), -np.ones(D - D // 2)]) * 0.75,
    ], axis=0)  # [3, D]

    latents = []
    labels = []
    for k in range(K):
        cov = np.diag(np.linspace(0.02, 0.08, D))
        pts = rng.multivariate_normal(mean=centers[k], cov=cov, size=per).astype(np.float32)
        latents.append(pts)
        labels += [f"C{k}"] * per
    latents = np.vstack(latents)  # [N, D]

    # If N not divisible by K, pad last cluster
    while len(labels) < N:
        latents = np.vstack([latents, latents[-1:]])
        labels.append("C2")

    ids = [f"p{i:03d}" for i in range(N)]

    latents_path = tmp_path / "latents.npy"
    labels_path = tmp_path / "labels.csv"
    ids_path = tmp_path / "ids.csv"

    np.save(latents_path, latents)
    labels_path.write_text("id,label\n" + "\n".join(f"{pid},{lab}" for pid, lab in zip(ids, labels)), encoding="utf-8")
    ids_path.write_text("id\n" + "\n".join(ids), encoding="utf-8")

    return {"latents": latents_path, "labels": labels_path, "ids": ids_path}


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    The tool must be discoverable and --help must mention t‑SNE/interactive/HTML/diagnostics.
    Try:
      1) python tools/plot_tsne_interactive.py --help
      2) python -m tools.plot_tsne_interactive --help (if tools is a package)
      3) spectramind diagnose tsne‑latents --help (or aliases)
    """
    help_blobs: List[str] = []

    # 1) Direct scripts
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 2) Module form
    if (project_root / "tools" / "__init__.py").exists():
        for module_name in ("tools.plot_tsne_interactive", "tools.plot_tsne", "tools.tsne_interactive"):
            code, out, err = run_proc(python_module_invocation(module_name, "--help"), cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    # 3) spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for t‑SNE interactive plotter."
    combined = "\n\n".join(help_blobs).lower()
    required_any = ["tsne", "t-sne", "interactive", "html", "diagnostic", "latents"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core t‑SNE keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the t‑SNE plotter in safe mode with tiny inputs (if supported), ensuring:
      - Exit code 0
      - Debug log appended
      - Outdir exists
      - Artifacts produced (HTML/JSON/PNG/MD) OR intended outputs mentioned
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
        for module_name in ("tools.plot_tsne_interactive", "tools.plot_tsne", "tools.tsne_interactive"):
            code, out, err = run_proc(python_module_invocation(module_name, "--help"), cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = python_module_invocation(module_name)
                break

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for the t‑SNE plotter."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "no_open" in flags:
        cmd.append(flags["no_open"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])

    # Required-ish inputs (best-effort)
    if "latents" in flags:
        cmd.extend([flags["latents"], str(tiny_inputs["latents"])])
    if "labels" in flags:
        cmd.extend([flags["labels"], str(tiny_inputs["labels"])])
    if "ids" in flags:
        cmd.extend([flags["ids"], str(tiny_inputs["ids"])])

    # Exports (request light artifacts; tool may *plan* in dry mode)
    for opt in ("html", "json", "png", "md"):
        if opt in flags:
            cmd.append(flags[opt])

    # Tiny t‑SNE params (if available)
    if "perplexity" in flags:
        cmd.extend([flags["perplexity"], "10"])
    if "learning_rate" in flags:
        cmd.extend([flags["learning_rate"], "100"])
    if "n_iter" in flags:
        cmd.extend([flags["n_iter"], "250"])
    if "metric" in flags:
        cmd.extend([flags["metric"], "euclidean"])
    if "seed" in flags:
        cmd.extend([flags["seed"], "42"])

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=240)
    assert code == 0, f"Safe t‑SNE invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["tsne", "t-sne", "interactive", "html", "latents", "embedding", "plot"]), \
        "Output does not resemble t‑SNE interactive plotter output."

    # Log grew
    post = read_text_or_empty(debug_log)
    assert len(post) >= pre_len, "v50_debug_log.md did not grow after t‑SNE invocation."
    appended = post[pre_len:]
    assert re.search(r"(tsne|t[- ]?sne|interactive|html|diagnose)", appended, re.IGNORECASE), \
        "No recognizable t‑SNE-related text found in the appended debug log segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after t‑SNE invocation."

    # Look for outputs OR intent messages (in dry mode)
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))
    produced_png = recent_files_with_suffix(temp_outdir, (".png",))
    produced_md = recent_files_with_suffix(temp_outdir, (".md",))

    if not (produced_html or produced_json or produced_png or produced_md):
        assert any(tok in combined for tok in ["outdir", "output", "write", ".html", ".json", ".png", ".md"]), \
            "No artifacts found and no mention of intended outputs (dry‑run should plan)."


def test_html_or_json_sanity_if_emitted(temp_outdir: Path):
    """
    If HTML or JSON was written, perform light sanity checks.
    """
    html_files = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    json_files = recent_files_with_suffix(temp_outdir, (".json",))

    for hf in html_files[:5]:
        text = read_text_or_empty(hf)
        if text.strip():
            # Heuristics: HTML doctype or common plotly div markers
            assert ("<!doctype html" in text.lower()) or ("plotly" in text.lower()) or ("<html" in text.lower()), \
                f"HTML {hf} does not look like an interactive report."

    for jf in json_files[:5]:
        text = read_text_or_empty(jf).strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception as e:
            pytest.fail(f"Malformed JSON at {jf}: {e}")
        assert isinstance(obj, (dict, list)), f"Unexpected JSON top-level type in {jf}"


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
        for module_name in ("tools.plot_tsne_interactive", "tools.plot_tsne", "tools.tsne_interactive"):
            code, out, err = run_proc(python_module_invocation(module_name, "--help"), cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = python_module_invocation(module_name)
                break

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain t‑SNE tool help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command with tiny inputs
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "no_open" in flags:
        cmd.append(flags["no_open"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "latents" in flags:
        cmd.extend([flags["latents"], str(tiny_inputs["latents"])])
    if "labels" in flags:
        cmd.extend([flags["labels"], str(tiny_inputs["labels"])])
    if "ids" in flags:
        cmd.extend([flags["ids"], str(tiny_inputs["ids"])])

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
    assert code1 == 0 and code2 == 0, f"Safe t‑SNE invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased in safe mode: before={pre}, after={post}"


def test_help_mentions_core_options_and_exports(project_root: Path):
    """
    --help should hint at core options/exports to guide users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        for module_name in ("tools.plot_tsne_interactive", "tools.plot_tsne", "tools.tsne_interactive"):
            code, out, err = run_proc(python_module_invocation(module_name, "--help"), cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for t‑SNE plotter."
    combined = "\n\n".join(help_texts).lower()
    want_any = ["html", "json", "png", "markdown", "perplexity", "n-iter", "metric", "seed", "latents", "labels"]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention options/exports; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
