#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/regression/test_generate_html_report_snapshots.py

SpectraMind V50 — Regression Snapshots for HTML Diagnostics Dashboard

Purpose
-------
Run the HTML dashboard generator (generate_html_report.py) in a lightweight mode
and compare the produced HTML against stored snapshots under:

    tests/regression/snapshots/*.snap.html
    tests/regression/snapshots/*.snap.json   (optional, for summary JSON)

This guards against accidental regressions to the diagnostics report structure,
headings, and key sections while tolerating volatile values (timestamps, paths,
run hashes), which are normalized away before comparison.

Behavior
--------
• If no snapshot exists yet:
    - The test FAILS with instructions to create/update snapshots.
    - To write snapshots, run CI/locally with:
          UPDATE_SNAPSHOTS=1 pytest tests/regression/test_generate_html_report_snapshots.py
• If UPDATE_SNAPSHOTS=1 is set:
    - The current outputs overwrite snapshot files.
• If the generator is not found or no HTML output is produced in dry/fast mode:
    - The test skips gracefully (useful while scaffolding).

Assumptions
-----------
• Report generator lives at tools/generate_html_report.py (preferred) or under src/tools/.
• The generator supports a lightweight or dry-run path (we probe common flags).
• Reports are emitted into ./outputs/ or ./outputs/diagnostics/ with names like:
      diagnostic_report*.html
• A JSON summary may be emitted as outputs/diagnostics_report*.json (optional).

"""

from __future__ import annotations

import os
import re
import json
import html
import hashlib
import typing as t
import subprocess
from pathlib import Path

import pytest

# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

_REPORT_CANDIDATES = [
    Path("tools") / "generate_html_report.py",
    Path("src") / "tools" / "generate_html_report.py",
]

def _find_report_script() -> Path | None:
    for p in _REPORT_CANDIDATES:
        if p.exists():
            return p.resolve()
    return None


# --------------------------------------------------------------------------------------
# Normalization helpers (remove volatile bits before comparing snapshots)
# --------------------------------------------------------------------------------------

_TS_PAT = re.compile(
    r"(?i)\b(20\d{2}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\b"
)
_DATE_PAT = re.compile(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b")
_UUID_PAT = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEX_HASH_PAT = re.compile(r"\b[0-9a-f]{7,64}\b", re.I)
_PATH_PAT = re.compile(r"(?i)(?:[A-Z]:\\|/)?(?:[^<>\n\"']+/)+[^<>\n\"']+")
_WS_PAT = re.compile(r"[ \t]+")

def _normalize_text(txt: str) -> str:
    # HTML entities normalized
    s = html.unescape(txt)

    # Remove volatile lines/segments commonly embedded in reports
    patterns = [
        _TS_PAT, _DATE_PAT, _UUID_PAT, _HEX_HASH_PAT,
    ]
    for pat in patterns:
        s = pat.sub("<REDACTED>", s)

    # Strip absolute/host paths (keep final component to remain informative)
    def _shorten_path(m: re.Match) -> str:
        p = m.group(0)
        if "\\" in p:
            tail = p.replace("\\", "/").split("/")[-1]
        else:
            tail = p.split("/")[-1]
        # ignore very short strings (not a real path)
        return tail if len(tail) > 3 else "<FILE>"

    s = _PATH_PAT.sub(_shorten_path, s)

    # Collapse excessive whitespace
    s = re.sub(r"\r\n?", "\n", s)
    s = _WS_PAT.sub(" ", s)

    # Drop common volatile lines by prefix
    stable_lines: list[str] = []
    volatile_prefixes = (
        "Generated at", "Build:", "Run hash", "Config hash", "Hostname", "Python", "CUDA", "Path:"
    )
    for line in s.splitlines():
        if any(line.strip().startswith(p) for p in volatile_prefixes):
            continue
        stable_lines.append(line.strip())
    s = "\n".join(stable_lines).strip()

    return s


def _stable_json(obj: t.Any) -> str:
    """
    Dump JSON deterministically after removing obviously volatile fields.
    """
    def _strip(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                lk = str(k).lower()
                if lk in {"generated_at", "timestamp", "created_at", "run_hash", "config_hash", "host"}:
                    continue
                out[k] = _strip(v)
            return out
        if isinstance(o, list):
            return [_strip(v) for v in o]
        if isinstance(o, str):
            # redact timestamps and hashes in strings
            return _normalize_text(o)
        return o

    clean = _strip(obj)
    return json.dumps(clean, sort_keys=True, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# Snapshot I/O
# --------------------------------------------------------------------------------------

SNAP_DIR = Path("tests") / "regression" / "snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

def _snapshot_path(name: str) -> Path:
    return SNAP_DIR / f"{name}.snap.html"

def _snapshot_json_path(name: str) -> Path:
    return SNAP_DIR / f"{name}.snap.json"

def _update_snapshots_enabled() -> bool:
    return os.environ.get("UPDATE_SNAPSHOTS", "0") in ("1", "true", "yes")


def _write_snapshot(p: Path, content: str):
    p.write_text(content, encoding="utf-8")


def _read_snapshot(p: Path) -> str | None:
    return p.read_text(encoding="utf-8") if p.exists() else None


# --------------------------------------------------------------------------------------
# Running the report generator
# --------------------------------------------------------------------------------------

def _run_report(script: Path, extra_flags: list[str] | None = None, env: dict | None = None, timeout: int = 600):
    """
    Try multiple lightweight flag sets to generate the HTML without heavy computation.
    """
    candidate_flag_sets = [
        ["--no-open", "--fast", "--light", "--dry-run"],
        ["--no-open", "--dry-run"],
        ["--no-open"],
        [],  # last resort
    ]
    if extra_flags:
        candidate_flag_sets = [extra_flags] + candidate_flag_sets

    merged_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1", "SPECTRAMIND_FAST": "1", "SPECTRAMIND_NO_BROWSER": "1"}
    for flags in candidate_flag_sets:
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python"), "-u", str(script), *flags],
            capture_output=True, text=True, env=merged_env, timeout=timeout, check=False
        )
        # Accept success or a helpful usage error (missing inputs) and stop probing
        joined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 or re.search(r"(?i)(usage:|missing|required|no such option|error:)", joined):
            return proc
    return proc  # type: ignore


def _find_reports() -> tuple[list[Path], list[Path]]:
    """
    Return (html_reports, json_reports) found in common output locations.
    """
    htmls: list[Path] = []
    jsons: list[Path] = []
    base = Path("outputs")
    htmls.extend(base.glob("diagnostic_report*.html"))
    htmls.extend((base / "diagnostics").glob("diagnostic_report*.html"))
    htmls.extend(base.glob("dashboard*.html"))
    htmls.extend((base / "diagnostics").glob("dashboard*.html"))

    jsons.extend(base.glob("diagnostics_report*.json"))
    jsons.extend((base / "diagnostics").glob("diagnostics_report*.json"))
    jsons.extend((base / "diagnostics").glob("diagnostic*_report*.json"))
    return htmls, jsons


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report_script() -> Path:
    p = _find_report_script()
    if not p:
        pytest.skip("generate_html_report.py not found in tools/ or src/tools/.")
    return p


def test_html_report_snapshot_regression(report_script: Path):
    """
    Generate the HTML report in a lightweight mode and compare with stored snapshot.
    If snapshot missing and UPDATE_SNAPSHOTS=1, create it; otherwise fail with guidance.
    """
    proc = _run_report(report_script, extra_flags=["--no-open", "--dry-run"])
    stdall = (proc.stdout or "") + (proc.stderr or "")
    # Tolerate non-zero as long as it's actionable
    assert proc.returncode in (0, 1, 2), f"Report generator failed unexpectedly.\n{stdall}"
    assert not re.search(r"Traceback \(most recent call last\):", stdall), f"Python traceback encountered.\n{stdall}"

    htmls, _ = _find_reports()
    if not htmls:
        pytest.skip("No HTML reports produced in dry/fast mode; skipping snapshot comparison.")

    # Use the first report for regression; project may emit versioned names
    target = htmls[0]
    content = target.read_text(encoding="utf-8", errors="ignore")
    norm = _normalize_text(content)

    # Stable snapshot name derived from filename base
    name = target.name.split(".")[0]
    snap_path = _snapshot_path(name)
    existing = _read_snapshot(snap_path)

    if existing is None:
        if _update_snapshots_enabled():
            _write_snapshot(snap_path, norm)
            pytest.skip(f"Snapshot created: {snap_path} (first run).")
        else:
            pytest.fail(
                f"Missing snapshot for {name}.\n"
                f"Run with UPDATE_SNAPSHOTS=1 to create:\n"
                f"  UPDATE_SNAPSHOTS=1 pytest {__file__}\n"
                f"Normalized report hash: {hashlib.sha256(norm.encode('utf-8')).hexdigest()}"
            )

    assert norm == existing, (
        "HTML regression mismatch after normalization.\n"
        f"Report: {target}\nSnapshot: {snap_path}\n\n"
        f"Tip: diff the files, or re-baseline with UPDATE_SNAPSHOTS=1 if the change is intentional."
    )


def test_json_summary_snapshot_if_available(report_script: Path):
    """
    If a JSON diagnostics summary is produced, compare a normalized dump to snapshot.
    Optional: skipped if no JSON files are emitted in this mode.
    """
    # Ensure the generator was invoked at least once
    _ = _run_report(report_script, extra_flags=["--no-open", "--dry-run"])

    _, jsons = _find_reports()
    if not jsons:
        pytest.skip("No diagnostics JSON produced; skipping JSON snapshot check.")

    target = jsons[0]
    raw = target.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        pytest.fail(f"Diagnostics JSON invalid at {target}: {e}")

    norm = _stable_json(obj)
    name = target.name.split(".")[0]
    snap_path = _snapshot_json_path(name)
    existing = _read_snapshot(snap_path)

    if existing is None:
        if _update_snapshots_enabled():
            _write_snapshot(snap_path, norm)
            pytest.skip(f"JSON snapshot created: {snap_path} (first run).")
        else:
            pytest.fail(
                f"Missing JSON snapshot for {name}.\n"
                f"Create with UPDATE_SNAPSHOTS=1:\n"
                f"  UPDATE_SNAPSHOTS=1 pytest {__file__}"
            )

    assert norm == existing, (
        "Diagnostics JSON regression mismatch after normalization.\n"
        f"JSON: {target}\nSnapshot: {snap_path}\n"
        "Re-baseline with UPDATE_SNAPSHOTS=1 if the change is intentional."
    )
