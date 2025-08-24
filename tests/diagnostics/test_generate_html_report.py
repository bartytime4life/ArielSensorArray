#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_generate_html_report.py

SpectraMind V50 — Diagnostics Test: tools/generate_html_report.py

Purpose
-------
Validate the HTML diagnostics dashboard/report generator in a *safe*, *adaptive*,
and *repo‑agnostic* manner. We test the CLI/UX contract, artifact creation (or
a clear *plan* in dry mode), and logging — not the scientific internals.

This suite asserts:
  1) Discoverability: --help exists and mentions report/dashboard/html/diagnostics.
  2) Safe execution: a dry-run/selftest/plan path exits with code 0.
  3) Inputs: accepts tiny placeholder artifacts when flags are available
     (diagnostic summary, UMAP/t‑SNE HTML, SHAP/Symbolic snippets, log table, etc.).
  4) Artifacts: produces versioned HTML/MD/JSON *or* states intended outputs in dry mode.
  5) Logging: appends to logs/v50_debug_log.md.
  6) Idempotency: repeating safe runs does not accumulate heavy artifacts.

Entry points probed (in order):
  • tools/generate_html_report.py         (canonical)
  • tools/generate_diagnostic_report.py   (defensive alias)
  • spectramind diagnose dashboard/report (optional wrapper)

Flag discovery is dynamic — we parse --help and map abstract names to actual flags.

Author: SpectraMind V50 QA
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    Candidate report generator scripts.
    """
    root = repo_root()
    cands = [
        root / "tools" / "generate_html_report.py",
        root / "tools" / "generate_diagnostic_report.py",  # alias/variant
    ]
    return [c for c in cands if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Optional wrapper CLIs for the dashboard/report.
    """
    return [
        ["spectramind", "diagnose", "dashboard"],
        ["spectramind", "diagnose", "report"],
        ["spectramind", "diagnose", "html-report"],
        [sys.executable, "-m", "spectramind", "diagnose", "dashboard"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "dashboard"],
        [sys.executable, "-m", "src.cli.cli_diagnose", "report"],
    ]


# ======================================================================================
# Subprocess helpers
# ======================================================================================

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 240) -> Tuple[int, str, str]:
    """
    Execute a command and return (exit_code, stdout, stderr) with text I/O.
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

    # Primary outputs and toggles
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html", "--report-html"],
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],
    "json": ["--json", "--json-out", "--export-json", "--summary-json"],
    "versioned": ["--versioned", "--versioned-out", "--versioned_html", "--versioned-html"],

    # Optional inputs (accept many spellings defensively)
    "diag_summary": ["--summary", "--diagnostic-summary", "--summary-json", "--diagnostic_summary"],
    "umap_html": ["--umap-html", "--umap_html", "--umap", "--umap-report"],
    "tsne_html": ["--tsne-html", "--tsne_html", "--tsne", "--tsne-report"],
    "shap_json": ["--shap-json", "--shap_json", "--shap-summary", "--shap_summary"],
    "symbolic_json": ["--symbolic-json", "--symbolic_json", "--symbolic-results", "--symbolic_results"],
    "rule_table_html": ["--rule-table-html", "--symbolic-rule-table", "--rule_table_html"],
    "log_table_md": ["--log-md", "--log-table-md", "--log_table_md", "--cli-log-md"],
    "manifest_json": ["--manifest-json", "--manifest_json", "--manifest"],
    "runhash_json": ["--run-hash-json", "--run_hash_json", "--run-hash-summary", "--run_hash_summary"],

    # Skips/toggles (optional)
    "no_umap": ["--no-umap", "--skip-umap"],
    "no_tsne": ["--no-tsne", "--skip-tsne"],
    "no_open": ["--no-open", "--no-open-html", "--no_open_html"],
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
# Fixtures — logs and tiny placeholder inputs
# ======================================================================================

@pytest.fixture(scope="module")
def project_root() -> Path:
    return repo_root()


@pytest.fixture
def temp_outdir(tmp_path: Path) -> Path:
    d = tmp_path / "html_report_out"
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
def tiny_report_inputs(tmp_path: Path) -> Dict[str, Path]:
    """
    Create tiny placeholder inputs that a report might stitch together.
    None of these need to be scientifically meaningful — they just exercise I/O paths.

    Files produced:
      - diagnostic_summary.json : tiny metrics dict
      - umap.html, tsne.html    : minimal HTML shells
      - shap.json               : minimal dict
      - symbolic.json           : tiny rule×planet
      - rule_table.html         : minimal table
      - log_table.md            : minimal CLI call history
      - run_hash_summary_v50.json: tiny hash/version stub
      - manifest.json           : tiny artifact list
    """
    base = tmp_path

    diag = base / "diagnostic_summary.json"
    diag.write_text(json.dumps({
        "summary": {"gll_mean": 1.234, "rmse": 0.056},
        "metrics": {"coverage": 0.91, "zscore_mean": 0.02}
    }, indent=2), encoding="utf-8")

    umap = base / "umap.html"
    umap.write_text("<!doctype html><html><body><h1>UMAP</h1></body></html>", encoding="utf-8")

    tsne = base / "tsne.html"
    tsne.write_text("<!doctype html><html><body><h1>t-SNE</h1></body></html>", encoding="utf-8")

    shap = base / "shap.json"
    shap.write_text(json.dumps({"bins": [0, 1, 2], "avg_shap": [0.1, 0.2, 0.05]}, indent=2), encoding="utf-8")

    symbolic = base / "symbolic_results.json"
    symbolic.write_text(json.dumps({
        "rules": ["R1", "R2"],
        "planets": [{"id": "p000", "violations": {"R1": 0.1, "R2": 0.0}}]
    }, indent=2), encoding="utf-8")

    rule_table = base / "symbolic_rule_table.html"
    rule_table.write_text("<html><body><table><tr><th>Rule</th></tr><tr><td>R1</td></tr></table></body></html>",
                          encoding="utf-8")

    log_table = base / "log_table.md"
    log_table.write_text("# CLI Calls\n\n| time | cmd |\n|---|---|\n| now | spectramind diagnose dashboard |\n", encoding="utf-8")

    runhash = base / "run_hash_summary_v50.json"
    runhash.write_text(json.dumps({"cli_version": "v50.0.1", "config_hash": "deadbeef", "timestamp": "2025-08-24T12:00:00Z"}, indent=2),
                       encoding="utf-8")

    manifest = base / "report_manifest.json"
    manifest.write_text(json.dumps({"artifacts": ["umap.html", "tsne.html", "symbolic_rule_table.html"]}, indent=2),
                        encoding="utf-8")

    return {
        "diag_summary": diag,
        "umap_html": umap,
        "tsne_html": tsne,
        "shap_json": shap,
        "symbolic_json": symbolic,
        "rule_table_html": rule_table,
        "log_table_md": log_table,
        "runhash_json": runhash,
        "manifest_json": manifest,
    }


# ======================================================================================
# Tests
# ======================================================================================

def test_discoverable_and_help(project_root: Path):
    """
    The tool must be discoverable and --help must mention report/dashboard/html/diagnostics.
    Try:
      1) python tools/generate_html_report.py --help
      2) python -m tools.generate_html_report --help (if tools is a package)
      3) spectramind diagnose dashboard --help (or aliases)
    """
    help_blobs: List[str] = []

    # Direct scripts
    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # Module form
    if (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_html_report", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_blobs.append(out + "\n" + err)

    # spectramind wrapper
    if not help_blobs:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_blobs.append(out + "\n" + err)
                break

    assert help_blobs, "No --help output found for HTML report generator."
    combined = "\n\n".join(help_blobs).lower()
    required_any = ["report", "dashboard", "html", "diagnostic", "summary"]
    assert any(tok in combined for tok in required_any), \
        f"--help lacks core report keywords; expected any of {required_any}"


def test_safe_invocation_and_artifacts(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path, tiny_report_inputs: Dict[str, Path]):
    """
    Execute the report generator in safe mode, passing tiny placeholder inputs if supported:
      - Exit code 0
      - Debug log appended
      - Outdir exists
      - Versioned HTML produced OR intended outputs mentioned (dry‑run acceptable)
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
        code, out, err = run_proc(python_module_invocation("tools.generate_html_report", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.generate_html_report")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = cli

    assert base_cmd is not None, "Unable to obtain a working entrypoint for the HTML report tool."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "no_open" in flags:
        cmd.append(flags["no_open"])  # avoid launching a browser during tests
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])
    if "versioned" in flags:
        cmd.append(flags["versioned"])
    if "html" in flags:
        cmd.append(flags["html"])
    if "md" in flags:
        cmd.append(flags["md"])
    if "json" in flags:
        cmd.append(flags["json"])

    # Optional inputs (best-effort; pass only if the flag exists)
    for abstract in ("diag_summary", "umap_html", "tsne_html", "shap_json", "symbolic_json",
                     "rule_table_html", "log_table_md", "runhash_json", "manifest_json"):
        if abstract in flags:
            cmd.extend([flags[abstract], str(tiny_report_inputs[abstract])])

    # Optional skips — if supported, flip them to ensure paths are respected
    for abstract in ("no_umap", "no_tsne"):
        if abstract in flags:
            # We *won't* pass these; we prefer to exercise embedding if tool supports it.
            pass

    # Execute
    code, out, err = run_proc(cmd, cwd=project_root, timeout=240)
    assert code == 0, f"Safe HTML report invocation failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
    combined = (out + "\n" + err).lower()
    assert any(k in combined for k in ["report", "dashboard", "html", "summary", "diagnostic"]), \
        "Output does not resemble an HTML diagnostics report tool."

    # Log append check
    post = read_text_or_empty(debug_log)
    assert len(post) >= pre_len, "v50_debug_log.md did not grow after report invocation."
    appended = post[pre_len:]
    assert re.search(r"(report|dashboard|html|diagnostic|summary)", appended, re.IGNORECASE), \
        "No recognizable report-related text found in the appended debug log segment."

    # Outdir exists
    assert temp_outdir.exists(), "Output directory missing after report invocation."

    # Look for outputs OR intent messages (in dry mode)
    produced_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    produced_md = recent_files_with_suffix(temp_outdir, (".md",))
    produced_json = recent_files_with_suffix(temp_outdir, (".json",))

    if not (produced_html or produced_md or produced_json):
        # Accept dry-run planning — require mention of intended outputs
        assert any(tok in combined for tok in ["outdir", "output", "write", ".html", ".md", ".json"]), \
            "No artifacts found and no mention of intended outputs (dry‑run should plan)."


def test_versioned_filename_if_requested(temp_outdir: Path, project_root: Path):
    """
    If the tool supports a 'versioned' flag or prints about versioned HTML,
    we expect at least one HTML filename containing 'v' or a timestamp-like token.
    This test is tolerant and only inspects existing outputs; it passes if none exist
    (pure dry-run case is covered elsewhere).
    """
    html_files = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    # If nothing was produced (pure dry-run), pass gracefully.
    if not html_files:
        return
    # Heuristic: look for 'v' or YYYY‑MM‑DD/HHMM style tokens in filename
    names = [p.name.lower() for p in html_files]
    has_versionish = any(re.search(r"(v\d+)|(\d{4}-?\d{2}-?\d{2})|(\d{6,})", n) for n in names)
    assert has_versionish, f"No version-like token found in produced HTML names: {names}"


def test_idempotent_safe_runs_do_not_accumulate_heavy_artifacts(project_root: Path, temp_outdir: Path):
    """
    Re-run a minimal safe command and ensure heavy artifacts (>5MB) do not accumulate.
    We do not require the tool to create heavy artifacts at all — this is a guardrail.
    """
    # Find entrypoint and flags from help
    help_text = ""
    base_cmd: Optional[List[str]] = None

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_script_invocation(script)
            break

    if base_cmd is None and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_html_report", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            base_cmd = python_module_invocation("tools.generate_html_report")

    if base_cmd is None:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                base_cmd = cli
                break

    assert base_cmd is not None, "Could not obtain report tool help for idempotency test."
    flags = discover_supported_flags(help_text)

    cmd = list(base_cmd)
    if "dry_run" in flags:
        cmd.append(flags["dry_run"])
    if "no_open" in flags:
        cmd.append(flags["no_open"])
    if "outdir" in flags:
        cmd.extend([flags["outdir"], str(temp_outdir)])

    # Count heavy artifacts
    def count_heavy(root: Path) -> int:
        if not root.exists():
            return 0
        n = 0
        for p in root.rglob("*"):
            if p.is_file() and p.stat().st_size > 5 * 1024 * 1024:
                n += 1
        return n

    pre = count_heavy(temp_outdir)
    code1, _, err1 = run_proc(cmd, cwd=project_root, timeout=200)
    code2, _, err2 = run_proc(cmd, cwd=project_root, timeout=200)
    assert code1 == 0 and code2 == 0, f"Safe report invocations failed.\n1) {err1}\n2) {err2}"
    post = count_heavy(temp_outdir)
    assert post <= pre, f"Heavy artifacts increased between safe runs: before={pre}, after={post}"


def test_help_mentions_core_sections_and_exports(project_root: Path):
    """
    --help should hint at key sections/exports to guide users.
    """
    help_texts: List[str] = []

    for script in tool_script_candidates():
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.generate_html_report", "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    if not help_texts:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0 and (out or err):
                help_texts.append(out + "\n" + err)
                break

    assert help_texts, "Unable to capture --help for HTML report generator."
    combined = "\n\n".join(help_texts).lower()
    want_any = [
        "html", "dashboard", "report", "summary", "gll", "calibration", "fft",
        "symbolic", "umap", "t-sne", "tsne", "markdown", "json", "versioned"
    ]
    assert any(tok in combined for tok in want_any), \
        f"--help should mention sections/exports; expected any of {want_any}"


# ======================================================================================
# End of file
# ======================================================================================
