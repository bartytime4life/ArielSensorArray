#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_explain_shap_metadata_v50.py

SpectraMind V50 — Diagnostics Test: tools/explain_shap_metadata_v50.py

Purpose
-------
Validate the SHAP + metadata explainer tool without running heavy computation.
This suite is *adaptive* to minor CLI differences and checks that the tool:

1) Is discoverable and prints a meaningful --help that references SHAP/metadata.
2) Supports a safe invocation path (dry-run/selftest/plan) that exits 0.
3) Accepts tiny synthetic inputs when flags are available (μ, SHAP, metadata).
4) Writes light artifacts (HTML/PNG/CSV/JSON/MD) *or* clearly states intended outputs.
5) Appends an audit entry to logs/v50_debug_log.md.
6) Is idempotent in safe mode (no heavy artifact accumulation).
7) Optionally honors UMAP/t-SNE flags or overlay toggles if present (best-effort).

Design
------
• The test adapts to these candidate entrypoints:
    - tools/explain_shap_metadata_v50.py (canonical)
    - tools/explain_shap_metadata.py      (legacy)
    - spectramind diagnose shap-metadata  (wrapper; optional)
• Flags are discovered by parsing --help and mapping abstract names to real aliases.
• Tiny inputs are created on-the-fly in tmp_path:
    - mu.npy       : shape (N, B)
    - shap.npy     : shape (N, B)     (simple random values)
    - metadata.csv : N rows with id and a categorical/group column
• The test tolerates pure "plan-only" behavior during dry-run (no files produced).
  In that case, it expects the tool to mention intended outputs.

Notes
-----
• This test does not verify scientific correctness. It checks CLI/UX and artifacts.
• If your tool names/flags are different, ensure --help documents them; the test
  discovers flags by searching help text.
• No network/GPU is required. Runs are time-bounded and tiny.

Author: SpectraMind V50 QA
"""

import json
import os
import re
import shutil
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
    Resolve repository root by walking upward until we find a 'tools' directory.
    Fallback: two levels up from tests/diagnostics.
    """
    here = Path(__file__).resolve()
    for anc in [here] + list(here.parents):
        if (anc / "tools").is_dir():
            return anc
    return Path(__file__).resolve().parents[2]


def tool_script_candidates() -> List[Path]:
    """
    Candidate explainer scripts.
    """
    root = repo_root()
    cands = [
        root / "tools" / "explain_shap_metadata_v50.py",
        root / "tools" / "explain_shap_metadata.py",  # legacy alias
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLI forms. We'll try these if direct script/module isn't available.
    """
    return [
        ["spectramind", "diagnose", "shap-metadata"],
        ["spectramind", "diagnose", "explain-shap-metadata"],
        [sys.executable, "-m", "spectramind", "diagnose", "shap-metadata"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "shap-metadata"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "explain-shap-metadata"],
    ]


# ======================================================================================
# Subprocess helpers
# ======================================================================================

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 150) -> Tuple[int, str, str]:
    """
    Execute a command and return (exit_code, stdout, stderr).
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

    # Safe / selftest
    "dry_run": ["--dry-run", "--dryrun", "--selftest", "--plan", "--check", "--no-exec"],

    # Output directory
    "outdir": ["--outdir", "--out-dir", "--output", "--output-dir", "-o"],

    # Inputs
    "mu": ["--mu", "--mu-npy", "--mu_path", "--pred-mu", "--pred_mu"],
    "shap": ["--shap", "--shap-npy", "--shap_path", "--shap-values", "--shap_values"],
    "metadata": ["--metadata", "--metadata-csv", "--meta", "--meta-csv", "--metadata_path"],

    # Exports
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "csv": ["--csv", "--csv-out", "--write-csv", "--per-bin-csv", "--export-csv"],
    "json": ["--json", "--json-out", "--export-json"],

    # Projection toggles (optional)
    "umap": ["--umap", "--do-umap", "--with-umap"],
    "tsne": ["--tsne", "--do-tsne", "--with-tsne"],

    # Overlay toggles (optional)
    "overlay_symbolic": ["--overlay-symbolic", "--symbolic-overlay", "--link-symbols", "--overlay_symbols"],
    "overlay_entropy": ["--overlay-entropy", "--entropy-overlay", "--overlay_entropy"],
    "overlay_gll": ["--overlay-gll", "--gll-overlay", "--overlay_gll"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Map abstract flag names to the actual aliases found in the tool's help text.
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


def find_any_pathname_match(root: Path, substrings: List[str]) -> List[Path]:
    """
    Find files whose filename contains any of substrings (case-insensitive).
    """
    if not root.exists():
        return []
    hits: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and any(s.lower() in p.name.lower() for s in substrings):
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
    d = tmp_path / "explain_shap_metadata_out"
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
    Create tiny inputs to exercise the tool in safe mode.

    - mu.npy       : (N, B) small float32 array
    - shap.npy     : (N, B) small float32 SHAP magnitudes (arbitrary)
    - metadata.csv : id, group columns with N rows

    Chosen shapes:
      N=4 examples, B=8 bins
    """
    N, B = 4, 8
    rng = np.random.default_rng(123)

    mu = rng.normal(0.0, 1.0, size=(N, B)).astype(np.float32)
    shap = np.abs(rng.normal(0.0, 0.2, size=(N, B)).astype(np.float32))

    mu_path = tmp_path / "mu.npy"
    shap_path = tmp_path / "shap.npy"
    meta_path = tmp_path / "metadata.csv"

    np.save(mu_path, mu)
    np.save(shap_path, shap)

    # Minimal metadata CSV
    # id,group
    # p000,A
    # p001,B
    # ...
    ids = [f"p{i:03d}" for i in range(N)]
    groups = ["A", "B", "A", "B"]
    lines = ["id,group"] + [f"{pid},{grp}" for pid, grp in zip(ids, groups)]
    meta_path.write_text("\n".join(lines), encoding="utf-8")

    return {"mu": mu_path, "shap": shap_path, "metadata": meta_path}


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    The tool must be discoverable and the help output should mention SHAP/metadata/UMAP/t-SNE.
    We try (in order):
      1) python tools/explain_shap_metadata_v50.py --help
      2) python -m tools.explain_shap_metadata_v50 --help (if tools is a package)
      3) spectramind diagnose shap-metadata --help
    """
    help_blobs: List[str] = []

    # 1) Direct script candidates
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 2) Module form (if tools is a package)
    if (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.explain_shap_metadata_v50", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # 3) spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for SHAP+metadata explainer."
    combined = "\n\n".join(help_blobs).lower()
    # Look for any core concept mention
    required_any = ["shap", "metadata", "umap", "t-sne", "tsne", "projection", "diagnostics"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_inputs: Dict[str, Path]):
    """
    Execute the explainer in safe mode, passing tiny inputs if supported.
    Validate:
      - exit code 0
      - debug log appended
      - outdir exists
      - artifacts produced (HTML/PNG/CSV/JSON/MD) OR intended outputs mentioned.
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    pre_len = len(read_text_or_empty(debug_log))

    # Obtain working base command + help text
    help_text = ""
    base_cmd: Optional[List[str]] = None

    # Try direct script first
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    # Try module
    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.explain_shap_metadata_v50", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.explain_shap_metadata_v50")

    # Try spectramind wrapper
    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Unable to obtain a working entrypoint for the explainer."
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
    if "metadata" in flags:
        cmd.extend([flags["metadata"], str(tiny_inputs["metadata"])])

    # Exports (request light artifacts; tool may just *plan* in dry mode)
    if "html" in flags:
        cmd.append(flags["html"])
    if "md" in flags:
        cmd.append(flags["md"])
    if "csv" in flags:
        cmd.append(flags["csv"])
    if "json" in flags:
        cmd.append(flags["json"])

    # Optional projections/overlays (best-effort; harmless in dry mode)
    for opt in ("umap", "tsne", "overlay_symbolic", "overlay_entropy", "overlay_gll"):
        if opt in flags:
            cmd.append(flags[opt])

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=210)
    assert code == 0, f"Safe explainer invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["shap", "metadata", "umap", "tsne", "projection", "overlay"]), \
        "Output does not resemble SHAP+metadata explainer output."

    # Log grew
    post_len = len(read_text_or_empty(debug_log))
    assert post_len >= pre_len, "v50_debug_log.md did not grow after explainer invocation."
    appended = read_text_or_empty(debug_log)[pre_len:]
    assert re.search(r"(explain|shap|metadata|umap|tsne|diagnose)", appended, re.IGNORECASE), \
        "No recognizable explainer-related text found in debug log appended segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after explainer invocation."

    # Look for produced artifacts OR mention of intended outputs
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_png = recent_files_with_suffix(temp_outdir, (".png",))
    produced_csv = recent_files_with_suffix(temp_outdir, (".csv",))
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))
    named_hits = find_any_pathname_match(temp_outdir, ["umap", "tsne", "shap", "metadata", "overlay", "diagnostics"])

    if not (produced_html or produced_png or produced_csv or produced_json or named_hits):
        # Accept dry-run planning — require mention of output intent
        assert any(tok in combined for tok in ["outdir", "output", "write", ".html", ".png", ".csv", ".json"]), \
            "No artifacts found and no mention of intended outputs in tool output (dry-run should plan)."


def test_json_or_csv_summaries_if_emitted(temp_outdir: Path):
    """
    If the tool wrote JSON or CSV summaries, perform light sanity checks.
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

        # Expect dictionary-like content for metrics/summary if non-empty
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            # Not strict; just look for indicative keys
            indicative = {"summary", "metrics", "umap", "tsne", "bins", "rules", "entropy", "shap"}
            assert keys & indicative or len(keys) > 0, \
                f"JSON {jf} lacks indicative diagnostic keys."

    # CSV sanity
    for cf in csv_files[:5]:
        text = read_text_or_empty(cf).strip()
        if not text:
            continue
        # Should have at least a header line or commas
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
        code, out, err = run_proc(python_module_invocation("tools.explain_shap_metadata_v50", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.explain_shap_metadata_v50")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain explainer help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "mu" in flags:
        cmd.extend([flags["mu"], str(tiny_inputs["mu"])])
    if "shap" in flags:
        cmd.extend([flags["shap"], str(tiny_inputs["shap"])])
    if "metadata" in flags:
        cmd.extend([flags["metadata"], str(tiny_inputs["metadata"])])

    # Count heavy artifacts (heuristic: >5MB or checkpoint suffixes)
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
    assert code1 == 0 and code2 == 0, f"Safe explainer invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased in safe mode: before={pre}, after={post}"


def test_stdout_mentions_projection_or_overlays(project_root: Path):
    """
    A quick semantic sniff: --help should mention projections (UMAP/t-SNE) or overlays
    (symbolic/entropy/gll) to guide users to the feature set.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.explain_shap_metadata_v50", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for the explainer."
    combined = "\n\n".join(help_texts).lower()
    want_any = ["umap", "t-sne", "tsne", "overlay", "symbolic", "entropy", "gll", "diagnostics", "html"]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention projections/overlays; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
