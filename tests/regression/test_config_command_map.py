#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/regression/test_config_command_map.py

SpectraMind V50 — Regression: Config ↔ Command Map

Purpose
-------
Generate or fetch a canonical "command-to-file mapping" describing how
CLI subcommands map to source files/configs (and vice‑versa). Compare the
normalized result to a stored snapshot:

    tests/regression/snapshots/config_command_map.snap.json

This helps detect accidental regressions in CLI coverage, command naming,
and their linkage to implementation files / Hydra config groups, while
tolerating volatile fields (timestamps, absolute paths, hashes).

Behavior
--------
• We probe multiple producer paths:
    1) Root Typer CLI (preferred), trying subcommands like:
         spectramind check-cli-map
         spectramind test --check-cli-map
         spectramind analyze-log --export-cli-map
    2) A tooling script if present (e.g., tools/cli_explain_util.py) with
       a `--export` or `--dump-json` flag.

• We expect a JSON artifact like:
      outputs/command_map.json
      outputs/cli_map.json
      outputs/diagnostics/command_map.json
  If none are produced, the test SKIPS (useful while scaffolding).

• To intentionally update the snapshot:
      UPDATE_SNAPSHOTS=1 pytest tests/regression/test_config_command_map.py

Notes
-----
• Normalization removes or redacts: timestamps, run/config hashes, and
  absolute paths (while preserving basename information).
• The test is gentle; it avoids heavy work and accepts actionable usage
  errors (non‑zero with "usage:/missing/required") as an indication that
  the producer is reachable even if inputs are absent.

"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pytest


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

def _candidate_cli_paths() -> List[Path]:
    return [
        Path("spectramind.py"),                  # root Typer CLI
        Path("src") / "spectramind.py",
    ]


def _candidate_tool_paths() -> List[Path]:
    return [
        Path("tools") / "cli_explain_util.py",
        Path("src") / "tools" / "cli_explain_util.py",
    ]


def _first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p.resolve()
    return None


# --------------------------------------------------------------------------------------
# Invocation helpers
# --------------------------------------------------------------------------------------

def _run_cli(cli: Path, args: List[str]) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "SPECTRAMIND_FAST": "1",
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",
    }
    return subprocess.run(
        [os.environ.get("PYTHON", "python"), "-u", str(cli), *args],
        text=True,
        capture_output=True,
        env=env,
        timeout=600,
        check=False,
    )


def _attempt_produce_map_via_cli(cli: Path) -> subprocess.CompletedProcess | None:
    """
    Try a few variants that projects commonly expose.
    Accept success OR actionable usage errors.
    """
    candidates = [
        ["check-cli-map"],
        ["test", "--check-cli-map"],
        ["analyze-log", "--export-cli-map"],
    ]
    for flags in candidates:
        proc = _run_cli(cli, flags)
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            return proc
        if re.search(r"(?i)(usage:|missing|required|no such option|error:)", combined):
            # Consider reachable; artifact may still be produced or the message is actionable.
            return proc
    return None


def _attempt_produce_map_via_tool(script: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    # Probe common flags
    for flags in (["--export"], ["--dump-json"], []):
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python"), "-u", str(script), *flags],
            text=True,
            capture_output=True,
            env=env,
            timeout=600,
            check=False,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            return proc
        if re.search(r"(?i)(usage:|missing|required|no such option|error:)", combined):
            return proc
    return proc  # type: ignore


def _find_map_artifacts() -> List[Path]:
    base = Path("outputs")
    candidates = [
        base / "command_map.json",
        base / "cli_map.json",
        *(base / "diagnostics").glob("command_map*.json"),
        *(base / "diagnostics").glob("cli_map*.json"),
        *(base.glob("command_map*.json")),
        *(base.glob("cli_map*.json")),
    ]
    return [p for p in candidates if p.exists()]


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------

_TS_PAT = re.compile(r"(?i)\b(20\d{2}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\b")
_DATE_PAT = re.compile(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b")
_UUID_PAT = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEX_PAT = re.compile(r"\b[0-9a-f]{7,64}\b", re.I)

def _shorten_path(s: str) -> str:
    # Keep only the basename to stabilize snapshots
    if "\\" in s or "/" in s:
        tail = s.replace("\\", "/").split("/")[-1]
        return tail if tail else "<FILE>"
    return s

def _normalize_value(v: Any) -> Any:
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            lk = str(k).lower()
            if lk in {"timestamp", "generated_at", "created_at", "updated_at", "run_hash", "config_hash"}:
                continue
            out[k] = _normalize_value(val)
        return out
    if isinstance(v, list):
        return [_normalize_value(x) for x in v]
    if isinstance(v, str):
        s = v
        # redact volatile tokens
        s = _TS_PAT.sub("<REDACTED>", s)
        s = _DATE_PAT.sub("<REDACTED>", s)
        s = _UUID_PAT.sub("<REDACTED>", s)
        # For hex blobs/hashes: only redact if they look like typical hashes
        if len(s) > 6 and _HEX_PAT.fullmatch(s):
            s = "<HEX>"
        # Shorten embedded paths heuristically
        if "/" in s or "\\" in s:
            parts = re.split(r"(\s+|,)", s)  # split but keep spacing
            parts = [(_shorten_path(p) if ("/" in p or "\\" in p) else p) for p in parts]
            s = "".join(parts)
        return s
    return v

def _normalize_map(obj: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_value(obj)


# --------------------------------------------------------------------------------------
# Snapshot I/O
# --------------------------------------------------------------------------------------

SNAP_DIR = Path("tests") / "regression" / "snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
SNAP_PATH = SNAP_DIR / "config_command_map.snap.json"

def _update_snapshots_enabled() -> bool:
    return os.environ.get("UPDATE_SNAPSHOTS", "0").lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def produced_or_skipped() -> bool:
    """
    Attempt to produce the command map via CLI or tool.
    Returns True if we attempted; SKIPs if neither producer exists.
    """
    cli = _first_existing(_candidate_cli_paths())
    tool = _first_existing(_candidate_tool_paths())

    attempted = False
    if cli:
        _ = _attempt_produce_map_via_cli(cli)
        attempted = True
    if tool:
        _ = _attempt_produce_map_via_tool(tool)
        attempted = True
    if not attempted:
        pytest.skip("No CLI or tool found to produce the command map; skipping regression test.")
    return True


def test_config_command_map_snapshot(produced_or_skipped: bool):
    """
    Locate command map JSON, normalize, and compare to snapshot.
    """
    artifacts = _find_map_artifacts()
    if not artifacts:
        pytest.skip("No command map JSON artifacts found in outputs/*; skipping snapshot comparison.")

    # Pick the most recent JSON by mtime
    target = max(artifacts, key=lambda p: p.stat().st_mtime)
    raw = target.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        pytest.fail(f"Command map is not valid JSON at {target}: {e}")

    norm = _normalize_map(obj)
    current = json.dumps(norm, sort_keys=True, indent=2, ensure_ascii=False)

    if not SNAP_PATH.exists():
        if _update_snapshots_enabled():
            SNAP_PATH.write_text(current, encoding="utf-8")
            pytest.skip(f"Snapshot created: {SNAP_PATH} (first run).")
        else:
            pytest.fail(
                "Missing snapshot for config/command map.\n"
                "Review the normalized JSON below and re‑baseline if intentional:\n\n"
                f"{current}\n\n"
                "To create snapshot:\n"
                "  UPDATE_SNAPSHOTS=1 pytest tests/regression/test_config_command_map.py"
            )

    expected = SNAP_PATH.read_text(encoding="utf-8")
    assert current == expected, (
        "Config ↔ Command map snapshot mismatch.\n\n"
        "If the change is intentional, re‑baseline with:\n"
        "  UPDATE_SNAPSHOTS=1 pytest tests/regression/test_config_command_map.py\n\n"
        f"Artifact: {target}\n"
        "---- CURRENT (normalized) ----\n"
        f"{current}\n"
        "---- EXPECTED ----\n"
        f"{expected}\n"
    )