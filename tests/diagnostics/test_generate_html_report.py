#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_generate_html_report.py

SpectraMind V50 — Diagnostics Tests (generate_html_report.py)

This test suite validates the upgraded HTML diagnostics report generator:
tools/generate_html_report.py

Objectives
----------
1) End-to-end CLI run with a tiny `diagnostic_summary.json` and a few dummy artifacts.
2) Verify main HTML is created, minimally valid (<html> tag), and references key sections.
3) Ensure assets (PNG/CSV/JSON fragments) are copied/linked into the output directory.
4) Confirm append-only audit logging into `logs/v50_debug_log.md`.
5) Check that OUTDIR is respected; no stray writes outside of it (except logs/).
6) Exercise optional inputs (UMAP/t-SNE HTML, CLI log table, symbolic table) when present.
7) Module invocation mode (`python -m tools.generate_html_report`) also works (if tool importable).

Design
------
• Self-contained temp repo scaffold (tools/, logs/, outputs/).
• Synthetic summary JSON is tiny and stable; assets are small (one PNG/CSV).
• Flexible assertions: tolerate extra sections/figures, but require core presence.
• Generous inline comments (NASA-grade documentation style).

Run
---
pytest -q tests/diagnostics/test_generate_html_report.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pytest


# -----------------------------
# Constants for tiny fixtures
# -----------------------------

N_PLANETS = 4
N_BINS = 13


# --------------------------------------
# Repo scaffold + synthetic artifacts
# --------------------------------------

def _ensure_repo_scaffold(repo_root: Path) -> None:
    """
    Create a minimal repo-like layout so default relative paths used by the tool
    (e.g., logs/ and outputs/) resolve cleanly.
    """
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "diagnostics").mkdir(parents=True, exist_ok=True)

    tool_path = repo_root / "tools" / "generate_html_report.py"
    if not tool_path.exists():
        # Place a tiny shim to make it obvious if the real tool isn't present
        tool_path.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                # Shim placeholder for tools/generate_html_report.py
                # Replace with the real diagnostics report generator.
                import sys
                if __name__ == "__main__":
                    sys.exit("Shim placeholder: replace tools/generate_html_report.py with the real implementation.")
                """
            ),
            encoding="utf-8",
        )


def _make_tiny_summary(outdir: Path) -> Path:
    """
    Create a minimal but realistic diagnostic_summary.json with tiny shapes.
    """
    summary = {
        "meta": {
            "version": "test",
            "created_at": "2025-08-23T12:00:00Z",
            "config_hash": "deadbeefcafebabe",
        },
        "metrics": {
            "mean_gll": -1.234,
            "rmse": 0.0123,
            "mae": 0.0101,
            "coverage_80": 0.81,
            "coverage_95": 0.95,
        },
        "per_planet": [],
    }
    rng = np.random.default_rng(7)
    for pid in range(N_PLANETS):
        planet = {
            "planet_id": f"P{pid:03d}",
            "gll": float(-1.0 - 0.05 * pid),
            "rmse": float(0.01 + 0.001 * pid),
            "mae": float(0.009 + 0.001 * pid),
            "n_bins": N_BINS,
            "symbolic": {
                "top_rule": "H2O_band_consistency" if pid % 2 == 0 else "CO2_peak_alignment",
                "violation_sum": float(rng.uniform(0.0, 1.0)),
            },
        }
        summary["per_planet"].append(planet)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "diagnostic_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _make_dummy_assets(assets_dir: Path) -> Dict[str, Path]:
    """
    Create tiny placeholder artifacts that the report may embed or link to:
      • A plot PNG
      • A small CSV
      • Optional UMAP/t-SNE HTML fragments
      • Optional symbolic rule table HTML
      • A tiny CLI log Markdown table
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {}

    # 1) PNG (single pixel) — keep tiny to avoid IO overhead
    png_path = assets_dir / "gll_heatmap.png"
    # Minimal valid PNG header with IHDR+IEND (we'll just write any bytes that look like PNG)
    # Note: For tests, existence is enough; the report tool should not parse it.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"  # signature
        b"\x00\x00\x00\rIHDR"  # IHDR chunk
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"  # 1x1, RGB
        b"\x90wS\xde"
        b"\x00\x00\x00\x0bIDAT\x08\xd7c``\x00\x00\x00\x04\x00\x01"  # tiny IDAT
        b"\x0e\xfb\x03\xfa"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_path.write_bytes(png_bytes)
    files["png"] = png_path

    # 2) CSV
    csv_path = assets_dir / "metrics_table.csv"
    csv_path.write_text("planet_id,gll,rmse\nP000,-1.0,0.01\nP001,-1.05,0.011\n", encoding="utf-8")
    files["csv"] = csv_path

    # 3) Optional UMAP / t-SNE fragments
    umap_html = assets_dir / "umap.html"
    umap_html.write_text("<!doctype html><html><body><div id='umap'>UMAP</div></body></html>", encoding="utf-8")
    files["umap"] = umap_html

    tsne_html = assets_dir / "tsne.html"
    tsne_html.write_text("<!doctype html><html><body><div id='tsne'>TSNE</div></body></html>", encoding="utf-8")
    files["tsne"] = tsne_html

    # 4) Optional symbolic rule table
    sym_html = assets_dir / "symbolic_rule_table.html"
    sym_html.write_text(
        "<!doctype html><html><body><table><tr><th>Rule</th><th>Score</th></tr>"
        "<tr><td>H2O_band_consistency</td><td>0.7</td></tr></table></body></html>",
        encoding="utf-8",
    )
    files["symbolic"] = sym_html

    # 5) Tiny CLI log markdown table (some generators embed this)
    log_md = assets_dir / "log_table.md"
    log_md.write_text("| time | cmd |\n|---|---|\n| 12:00 | spectramind diagnose dashboard |\n", encoding="utf-8")
    files["log_md"] = log_md

    return files


# ---------------------------
# Helper: tool invocations
# ---------------------------

def _discover_tool(repo_root: Path) -> Path:
    return repo_root / "tools" / "generate_html_report.py"


def _run_tool_subprocess(
    repo_root: Path,
    tool_path: Path,
    summary_path: Path,
    outdir: Path,
    art_dir: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    use_python_module: bool = False,
) -> subprocess.CompletedProcess:
    """
    Execute the generator via subprocess. Prefer `python <tool>`.
    """
    env = os.environ.copy()
    env.setdefault("SPECTRAMIND_TEST", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")

    if use_python_module:
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "tools.generate_html_report"]
    else:
        cmd = [sys.executable, str(tool_path)]

    args = [
        "--summary", str(summary_path),
        "--outdir", str(outdir),
        "--title", "SpectraMind V50 — Diagnostics Report (Test)",
        "--no-open",
    ]

    # Common test-friendly flags; if unknown, the tool should ignore or exit cleanly
    args += [
        "--embed-umap", str((art_dir / "umap.html") if art_dir else ""),
        "--embed-tsne", str((art_dir / "tsne.html") if art_dir else ""),
        "--embed-symbolic-table", str((art_dir / "symbolic_rule_table.html") if art_dir else ""),
        "--assets-dir", str(art_dir) if art_dir else "",
        "--include-csv", str((art_dir / "metrics_table.csv") if art_dir else ""),
        "--include-png", str((art_dir / "gll_heatmap.png") if art_dir else ""),
        "--version", "test",
    ]

    if extra_args:
        args += list(extra_args)

    proc = subprocess.run(
        cmd + args,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# ----------------------
# Pytest fixtures
# ----------------------

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path) -> Path:
    _ensure_repo_scaffold(tmp_path)
    return tmp_path


@pytest.fixture(scope="function")
def tiny_inputs(repo_tmp: Path) -> Dict[str, Path]:
    """
    Materialize a tiny diagnostic summary and a handful of artifact files.
    """
    inputs_dir = repo_tmp / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    summary = _make_tiny_summary(inputs_dir)
    assets = _make_dummy_assets(inputs_dir / "assets")
    return {"summary": summary, "assets_dir": assets["png"].parent}


# ---------------------------------------
# Core tests — end-to-end and robustness
# ---------------------------------------

@pytest.mark.integration
def test_cli_generates_report_with_embeds(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Run the HTML report generator with a small summary and embedding flags:
      • Expect a main HTML file (report.html or index.html-like).
      • Validate it contains <html> and key text markers.
      • Ensure assets are either copied or linked.
      • Confirm logs/v50_debug_log.md is appended.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "report_embeds"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir,
        art_dir=tiny_inputs["assets_dir"],
        extra_args=["--report-name", "diagnostics_report_test.html"],  # tolerated if supported
    )

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)

    assert proc.returncode == 0, "Report generator should exit successfully."

    # The tool may choose a default name; check common possibilities.
    candidates = [
        outdir / "diagnostics_report_test.html",
        outdir / "diagnostics_report.html",
        outdir / "report.html",
        outdir / "index.html",
    ]
    report_path = None
    for c in candidates:
        if c.exists():
            report_path = c
            break
    assert report_path is not None, "Expected an HTML report to be written."

    html = _read(report_path)
    assert "<html" in html.lower(), "HTML report should contain an <html> tag."
    assert "SpectraMind V50" in html or "Diagnostics" in html, "Expected report title/branding marker."

    # If the generator embeds UMAP/t-SNE/symbolic table, check presence by id/keyword
    # (tolerate implementations that inline or iframe them)
    inline_markers = ("umap", "tsne", "symbolic", "H2O_band_consistency", "CO2_peak_alignment")
    assert any(m.lower() in html.lower() for m in inline_markers), \
        "Expected at least one embedded or referenced diagnostic section (UMAP/TSNE/Symbolic)."

    # Ensure at least one PNG or IMG reference is present (embedded or linked)
    assert (outdir / "gll_heatmap.png").exists() or ("<img" in html.lower()), \
        "Expected a plot asset to be copied or an <img> tag included."

    # Audit log
    log_path = repo_tmp / "logs" / "v50_debug_log.md"
    assert log_path.exists(), "Expected audit log to be present."
    log_text = _read(log_path)
    assert "generate_html_report" in log_text or "diagnostics report" in log_text.lower(), \
        "Audit log should mention HTML diagnostics generation."


@pytest.mark.integration
def test_cli_module_invocation_mode(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Validate `python -m tools.generate_html_report` invocation. If only a shim exists,
    the test xfails with an instructive note.
    """
    tool = _discover_tool(repo_tmp)
    is_shim = "Shim placeholder" in _read(tool)

    outdir = repo_tmp / "outputs" / "diagnostics" / "report_module_mode"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir,
        art_dir=tiny_inputs["assets_dir"],
        use_python_module=True,
        extra_args=["--report-name", "module_mode.html"],
    )

    if is_shim:
        pytest.xfail("Tool shim detected — replace with real tools/generate_html_report.py to pass this test.")

    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0

    # Expect report
    assert (outdir / "module_mode.html").exists() or any(p.suffix == ".html" for p in outdir.glob("*.html"))


@pytest.mark.integration
def test_outdir_respected_and_no_strays(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Ensure that files are written under --outdir (plus logs/). No stray writes to repo root.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "report_outdir"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir,
        art_dir=tiny_inputs["assets_dir"],
        extra_args=["--report-name", "outdir_ok.html"],
    )
    assert proc.returncode == 0

    after = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())
    new_files = sorted(list(after - before))

    # Allowed: logs/* and anything under outdir. Tolerate an outputs/run_hash_summary*.json.
    disallowed = []
    out_rel = outdir.relative_to(repo_tmp).as_posix()
    for rel in new_files:
        if rel.startswith("logs/"):
            continue
        if rel.startswith(out_rel):
            continue
        if rel.startswith("outputs/") and re.search(r"run_hash_summary.*\.json$", rel):
            continue
        if rel.endswith(".pyc") or "/__pycache__/" in rel:
            continue
        disallowed.append(rel)

    assert not disallowed, f"Report generator wrote unexpected files outside --outdir: {disallowed}"


@pytest.mark.integration
def test_missing_optional_embeds_do_not_fail(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    If UMAP/t-SNE/symbolic table paths aren't provided (or don't exist),
    the report generator should still succeed and produce a valid HTML report.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "report_no_embeds"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir,
        art_dir=None,  # no assets dir; no embed flags
        extra_args=["--report-name", "no_embeds.html"],
    )
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
    assert proc.returncode == 0

    # HTML exists and is minimally valid
    candidates = [outdir / "no_embeds.html"] + list(outdir.glob("*.html"))
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    assert found is not None, "Expected an HTML report to be created even without optional embeds."
    html = _read(found)
    assert "<html" in html.lower(), "Report should contain an <html> tag."
    # It should still include some header/title or summary reference
    assert "diagnostic_summary" in html.lower() or "metrics" in html.lower() or "SpectraMind".lower() in html.lower()


@pytest.mark.integration
def test_html_contains_basic_sections_and_links(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Sanity: the report should include references to metrics and per-planet sections
    and potentially link to assets (PNG/CSV).
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "report_sections"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir,
        art_dir=tiny_inputs["assets_dir"],
        extra_args=["--report-name", "sections.html"],
    )
    assert proc.returncode == 0

    html = _read(outdir / "sections.html")
    # Look for metric keywords and at least a couple of link-ish patterns
    expect_keys = ("mean_gll", "rmse", "mae", "coverage")
    assert any(k in html for k in expect_keys), "Expected metrics keys to appear in report HTML."

    # Links to PNG/CSV (can be <img> or <a href>)
    linkish = re.findall(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    # Not strict on exact filenames; require at least one asset reference
    assert linkish, "Expected at least one asset link/reference (PNG/CSV/HTML)."


@pytest.mark.integration
def test_audit_log_is_append_only(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Ensure audit log grows (or at least doesn't shrink) across runs.
    """
    tool = _discover_tool(repo_tmp)
    outdir1 = repo_tmp / "outputs" / "diagnostics" / "report_log1"
    outdir2 = repo_tmp / "outputs" / "diagnostics" / "report_log2"
    outdir1.mkdir(parents=True, exist_ok=True)
    outdir2.mkdir(parents=True, exist_ok=True)

    log_path = repo_tmp / "logs" / "v50_debug_log.md"

    p1 = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir1,
        art_dir=tiny_inputs["assets_dir"],
        extra_args=["--report-name", "r1.html"],
    )
    assert p1.returncode == 0
    size1 = log_path.stat().st_size if log_path.exists() else 0

    p2 = _run_tool_subprocess(
        repo_root=repo_tmp,
        tool_path=tool,
        summary_path=tiny_inputs["summary"],
        outdir=outdir2,
        art_dir=tiny_inputs["assets_dir"],
        extra_args=["--report-name", "r2.html"],
    )
    assert p2.returncode == 0
    size2 = log_path.stat().st_size if log_path.exists() else 0

    assert size2 >= size1, "Audit log should not shrink after a subsequent run."
    if size1 > 0:
        assert size2 > size1, "Audit log should typically grow across runs."


@pytest.mark.integration
def test_cli_returns_nonzero_on_missing_summary(repo_tmp: Path) -> None:
    """
    If the required --summary file is missing, the tool should exit non-zero
    with a helpful error message in stdout/stderr.
    """
    tool = _discover_tool(repo_tmp)
    outdir = repo_tmp / "outputs" / "diagnostics" / "report_missing_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [sys.executable, str(tool), "--summary", "does_not_exist.json", "--outdir", str(outdir), "--no-open"],
        cwd=str(repo_tmp),
        capture_output=True,
        text=True,
        env={**os.environ, "SPECTRAMIND_TEST": "1", "MPLBACKEND": "Agg"},
        timeout=30,
    )

    assert proc.returncode != 0, "Missing --summary should cause non-zero exit."
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert re.search(r"(missing|not found|--summary)", combined, re.IGNORECASE), \
        "Expected helpful error mentioning the missing summary file."


# --------------------------------------
# Local debug helper (documentation only)
# --------------------------------------

def _local_debug_note() -> None:
    """
    Not executed; documents a handy local run for a single test:

    PYTHONPATH=. pytest -q tests/diagnostics/test_generate_html_report.py -k report_with_embeds

    Tip: If the generator supports `--report-name`, you can set predictable names in CI.
    """


# End of file
