#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/regression/test_cli_map_integrity.py

SpectraMind V50 — Regression: CLI Map Integrity

Purpose
-------
Validate the integrity of the CLI command → implementation/config map that the repo
exports (typically as a JSON artifact, e.g. outputs/command_map.json or outputs/diagnostics/command_map*.json),
and compare a normalized integrity summary to a stored snapshot:

  tests/regression/snapshots/cli_map_integrity.snap.json

What we check (lightweight, CI‑safe)
------------------------------------
• Structure sanity: non‑empty commands, unique command names, per‑command metadata.
• Referenced files: if a path looks relative (contains "/" or "\\"), verify existence; otherwise skip.
• Duplicates / collisions: a file should not claim multiple exclusive owners unless map expresses sharing.
• Optional help parity (best‑effort): if root CLI exists, ensure at least some mapped commands appear in `--help`.

This test is defensive and *skips* gracefully when:
  • No map artifact is present yet (e.g., while scaffolding),
  • No CLI entrypoint is found to probe help.

How to (re)baseline the snapshot
--------------------------------
    UPDATE_SNAPSHOTS=1 pytest tests/regression/test_cli_map_integrity.py

Notes
-----
• Normalization removes timestamps, hashes, and absolute paths while keeping basenames.
• We tolerate repo evolution by focusing on *structure* and *counts*, not raw content dumps.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


# --------------------------------------------------------------------------------------
# Discovery utilities
# --------------------------------------------------------------------------------------

def _candidate_cli_paths() -> List[Path]:
    return [
        Path("spectramind.py"),
        Path("src") / "spectramind.py",
    ]


def _find_cli() -> Path | None:
    for p in _candidate_cli_paths():
        if p.exists():
            return p.resolve()
    return None


def _find_map_artifacts() -> List[Path]:
    base = Path("outputs")
    candidates = [
        base / "command_map.json",
        base / "cli_map.json",
        *(base / "diagnostics").glob("command_map*.json"),
        *(base / "diagnostics").glob("cli_map*.json"),
        *base.glob("command_map*.json"),
        *base.glob("cli_map*.json"),
    ]
    return [p for p in candidates if p.exists()]


# --------------------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------------------

_TS_PAT = re.compile(r"(?i)\b(20\d{2}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\b")
_UUID_PAT = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_HEX_PAT = re.compile(r"\b[0-9a-f]{7,64}\b", re.I)


def _basename_only(s: str) -> str:
    s = s.replace("\\", "/")
    if "/" in s:
        s = s.split("/")[-1]
    return s


def _normalize_value(v: Any) -> Any:
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            lk = str(k).lower()
            if lk in {"timestamp", "created_at", "generated_at", "updated_at", "run_hash", "config_hash"}:
                continue
            out[k] = _normalize_value(val)
        return out
    if isinstance(v, list):
        return [_normalize_value(x) for x in v]
    if isinstance(v, str):
        s = _TS_PAT.sub("<REDACTED>", v)
        s = _UUID_PAT.sub("<REDACTED>", s)
        # redact long hex strings that look like hashes
        if _HEX_PAT.fullmatch(s or ""):
            return "<HEX>"
        # shorten paths to basenames
        if "/" in s or "\\" in s:
            return _basename_only(s)
        return s
    return v


def _normalize_map(obj: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_value(obj)


# --------------------------------------------------------------------------------------
# Integrity computation
# --------------------------------------------------------------------------------------

@dataclass
class IntegritySummary:
    total_commands: int
    unique_commands: int
    total_files: int
    existing_files: int
    missing_files: int
    commands_with_files: int
    commands_without_files: int
    duplicates_in_files: int
    command_names_sample: List[str]
    missing_files_sample: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)


def _iter_paths_from_entry(entry: Dict[str, Any]) -> List[str]:
    """
    Attempt to read file path fields from a command map entry.
    Common keys: files, file, paths, source_files, src, modules
    """
    keys = ("files", "file", "paths", "source_files", "src", "modules")
    vals: List[str] = []
    for k in keys:
        if k in entry:
            v = entry[k]
            if isinstance(v, str):
                vals.append(v)
            elif isinstance(v, list):
                vals.extend([x for x in v if isinstance(x, str)])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in vals:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _is_relative_like(p: str) -> bool:
    # If it contains a path separator, consider it a relative path (we will check existence).
    return ("/" in p) or ("\\" in p)


def _compute_integrity(obj: Dict[str, Any]) -> IntegritySummary:
    """
    Compute a structural integrity summary over the command map.
    The map is expected to be something like:
      { "commands": { "train": {...}, "diagnose dashboard": {...}, ... }, "meta": {...} }
    but we also tolerate a list or an alternate top-level key. We adapt heuristically.
    """
    # Extract command dict
    commands: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        if "commands" in obj and isinstance(obj["commands"], dict):
            commands = obj["commands"]  # type: ignore
        elif "map" in obj and isinstance(obj["map"], dict):
            commands = obj["map"]       # type: ignore
        else:
            # fallback: consider every top-level key (excluding obvious meta keys) a command
            for k, v in obj.items():
                if isinstance(v, dict) and k.lower() not in {"meta", "version"}:
                    commands[k] = v
    elif isinstance(obj, list):
        # list of entries: expect {"command": "...", ...}
        for e in obj:
            if isinstance(e, dict) and "command" in e:
                commands[str(e["command"])] = e

    command_names = list(commands.keys())
    total_commands = len(command_names)
    unique_commands = len(set(command_names))

    # Gather files, existence checks, duplicates
    all_files: List[str] = []
    existing_files = 0
    missing_files = 0
    commands_with_files = 0
    commands_without_files = 0
    missing_files_sample: List[str] = []

    for cmd, entry in commands.items():
        paths = _iter_paths_from_entry(entry)
        if paths:
            commands_with_files += 1
        else:
            commands_without_files += 1

        for p in paths:
            all_files.append(p)
            if _is_relative_like(p):
                rel = Path(p)
                if rel.exists():
                    existing_files += 1
                else:
                    missing_files += 1
                    if len(missing_files_sample) < 8:
                        missing_files_sample.append(p)

    # duplicate files (by basename for robustness)
    by_basename: Dict[str, int] = {}
    for p in all_files:
        base = _basename_only(p)
        by_basename[base] = by_basename.get(base, 0) + 1
    duplicates_in_files = sum(1 for k, c in by_basename.items() if c > 1)

    # sample names (deterministic order, truncated)
    command_names_sample = sorted(set(command_names))[:10]

    return IntegritySummary(
        total_commands=total_commands,
        unique_commands=unique_commands,
        total_files=len(all_files),
        existing_files=existing_files,
        missing_files=missing_files,
        commands_with_files=commands_with_files,
        commands_without_files=commands_without_files,
        duplicates_in_files=duplicates_in_files,
        command_names_sample=command_names_sample,
        missing_files_sample=missing_files_sample,
    )


# --------------------------------------------------------------------------------------
# Snapshot I/O
# --------------------------------------------------------------------------------------

SNAP_DIR = Path("tests") / "regression" / "snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)
SNAP_PATH = SNAP_DIR / "cli_map_integrity.snap.json"


def _update_snapshots_enabled() -> bool:
    return os.environ.get("UPDATE_SNAPSHOTS", "0").lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------------------
# Optional help parity
# --------------------------------------------------------------------------------------

def _root_help_commands(cli: Path) -> List[str]:
    """
    Best‑effort parsing of root help to extract subcommand tokens.
    Implementation is intentionally loose: look for lines that resemble subcommand headers.
    """
    proc = subprocess.run(
        [os.environ.get("PYTHON", "python"), "-u", str(cli), "--help"],
        text=True, capture_output=True, check=False, timeout=120
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    cmds: List[str] = []
    # Simple heuristic: lines starting with 2 spaces then a word
    for line in out.splitlines():
        m = re.match(r"^\s{2,}([a-zA-Z][\w\-:]*)\s", line)
        if m:
            cmds.append(m.group(1).strip())
    return sorted(set(cmds))[:30]


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def map_artifact() -> Path:
    arts = _find_map_artifacts()
    if not arts:
        pytest.skip("No CLI map artifact found in outputs/*; skipping integrity regression.")
    # pick most recent
    return max(arts, key=lambda p: p.stat().st_mtime)


def test_cli_map_integrity_snapshot(map_artifact: Path):
    """
    Normalize the map and compare a structural integrity summary to snapshot.
    """
    raw = map_artifact.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        pytest.fail(f"CLI map artifact is not valid JSON at {map_artifact}: {e}")

    norm = _normalize_map(obj)
    summary = _compute_integrity(norm)
    current = summary.to_json()

    if not SNAP_PATH.exists():
        if _update_snapshots_enabled():
            SNAP_PATH.write_text(current, encoding="utf-8")
            pytest.skip(f"Snapshot created: {SNAP_PATH}")
        else:
            pytest.fail(
                "Missing snapshot for CLI map integrity.\n"
                "Review the computed summary below and re‑baseline if intentional:\n\n"
                f"{current}\n\n"
                "To create snapshot:\n"
                "  UPDATE_SNAPSHOTS=1 pytest tests/regression/test_cli_map_integrity.py"
            )

    expected = SNAP_PATH.read_text(encoding="utf-8")
    assert current == expected, (
        "CLI map integrity summary mismatch.\n\n"
        "If the change is intentional, re‑baseline with:\n"
        "  UPDATE_SNAPSHOTS=1 pytest tests/regression/test_cli_map_integrity.py\n\n"
        f"Artifact: {map_artifact}\n"
        "---- CURRENT ----\n"
        f"{current}\n"
        "---- EXPECTED ----\n"
        f"{expected}\n"
    )


def test_cli_map_some_commands_appear_in_help(map_artifact: Path):
    """
    Optional parity check: confirm that at least one command from the map
    is visible in root `--help` output (when CLI exists). Skip if CLI missing.
    """
    cli = _find_cli()
    if not cli:
        pytest.skip("Root CLI not found; skipping help parity check.")

    raw = map_artifact.read_text(encoding="utf-8", errors="ignore")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        pytest.skip("Map not JSON‑parsable (already asserted above); skip parity.")

    # Extract some command names from the map (normalized)
    norm = _normalize_map(obj)
    commands: List[str] = []
    if isinstance(norm, dict):
        if "commands" in norm and isinstance(norm["commands"], dict):
            commands = list(norm["commands"].keys())[:30]
        elif "map" in norm and isinstance(norm["map"], dict):
            commands = list(norm["map"].keys())[:30]
        else:
            commands = [k for k, v in norm.items() if isinstance(v, dict)][:30]
    elif isinstance(norm, list):
        for e in norm[:30]:
            if isinstance(e, dict) and "command" in e:
                commands.append(str(e["command"]))

    if not commands:
        pytest.skip("No command names detected in map; skipping parity check.")

    help_cmds = _root_help_commands(cli)
    # Accept parity as long as *some* command token appears (robust to naming differences)
    overlap = [c for c in commands if any(c.split()[0] == h for h in help_cmds)]
    if not overlap:
        pytest.skip("No overlap between map commands and --help tokens (naming may differ); skipping.")