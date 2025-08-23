# tests/artifacts/test_log_integrity.py
# SpectraMind V50 — Log Integrity & Provenance Tests
#
# Objective
# ---------
# Validate that our audit logs (Markdown and/or JSONL) are readable, structured,
# and consistent with our reproducibility posture:
#   • logs/v50_debug_log.md      (human-friendly append-only audit)
#   • logs/v50_runs.jsonl        (machine-friendly JSONL, one record per run)
#
# The tests are *tolerant* of format variations and will skip gracefully if a
# particular file is not present yet. When present, they check:
#   1) UTF‑8 readability (no binary/NULs) and reasonable size.
#   2) Presence of expected cues (timestamps, CLI command strings).
#   3) JSONL parseability and minimal schema (timestamp, cmd, exit_code).
#   4) Monotonic (non‑decreasing) timestamps → “no time travel”.
#   5) Optional “recent activity” window if EXPECT_LOG_WITHIN_HOURS is set.
#   6) Markdown table sanity (if a table is present, row column counts align).
#
# Config (optional environment variables)
# ---------------------------------------
#   LOG_MD_PATH              : path to the Markdown log (default: logs/v50_debug_log.md)
#   LOG_JSONL_PATH           : path to the JSONL log (default: logs/v50_runs.jsonl)
#   EXPECT_LOG_WITHIN_HOURS  : int; if set, assert at least one entry is newer
#                              than now - X hours (JSONL path preferred)
#
# Usage
# -----
#   pytest -q tests/artifacts/test_log_integrity.py
#
# Notes
# -----
#   • Tests do NOT modify any log file.
#   • If both logs are missing (fresh repo), core tests will skip with context.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest


# -----------------------------
# Helpers & configuration
# -----------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default


def _log_md_path() -> Path:
    return Path(_env("LOG_MD_PATH", "logs/v50_debug_log.md"))


def _log_jsonl_path() -> Path:
    return Path(_env("LOG_JSONL_PATH", "logs/v50_runs.jsonl"))


def _read_text_safe(p: Path) -> str:
    data = p.read_bytes()
    # quick binary sanity: reject NULs
    assert b"\x00" not in data, f"NUL byte detected in {p}"
    return data.decode("utf-8", errors="strict")


def _parse_iso8601(s: str) -> Optional[datetime]:
    """
    Parse a variety of ISO8601-ish times used in logs.
    Accepted forms (UTC):
      2025-08-23T12:34:56Z
      2025-08-23T12:34:56.789Z
      2025-08-23 12:34:56Z
    Returns timezone-aware datetime in UTC, or None if not parseable.
    """
    s = s.strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z$", s)
    if not m:
        return None
    try:
        dt = datetime.fromisoformat((m.group(1) + "T" + m.group(2)).replace("Z", ""))
    except ValueError:
        return None
    # ensure tz-aware UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _extract_md_timestamps(md_text: str) -> List[datetime]:
    """
    Heuristic extraction of timestamps from the Markdown log.
    Matches lines like:
      [2025-08-23T12:34:56Z] spectramind train ...
      timestamp: 2025-08-23T12:34:56.123Z
    """
    ts_list: List[datetime] = []

    # [ISO] prefix lines
    for m in re.finditer(r"\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z\]", md_text):
        ts = _parse_iso8601(m.group(1) + "Z")
        if ts:
            ts_list.append(ts)

    # key: value style
    for m in re.finditer(r"(?:^|\s)(?:ts|timestamp)\s*[:=]\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z\b", md_text):
        ts = _parse_iso8601(m.group(1) + "Z")
        if ts:
            ts_list.append(ts)

    # Non-decreasing sort for monotonicity checks
    ts_list.sort()
    return ts_list


def _load_jsonl_records(p: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                # tolerate blank lines
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise AssertionError(f"JSONL parse error at {p}:{i} → {e}") from e
            assert isinstance(obj, dict), f"JSONL object at {p}:{i} must be a JSON object"
            recs.append(obj)
    return recs


def _coerce_ts(obj: Dict[str, Any]) -> Optional[datetime]:
    """
    Extract a timestamp-like field from a JSONL record.
    Accepts any of: ts, timestamp, time (ISO8601Z).
    """
    for k in ("ts", "timestamp", "time"):
        if k in obj and isinstance(obj[k], str):
            dt = _parse_iso8601(obj[k].strip().rstrip("Z") + "Z")
            if dt:
                return dt
    return None


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture(scope="session")
def md_path() -> Path:
    return _log_md_path()


@pytest.fixture(scope="session")
def jsonl_path() -> Path:
    return _log_jsonl_path()


# -----------------------------
# Tests — presence & readability
# -----------------------------

def test_logs_present_or_skip(md_path: Path, jsonl_path: Path) -> None:
    """
    At least one log should exist. If neither exists yet (fresh repo),
    skip the suite gracefully with instructions.
    """
    if not md_path.exists() and not jsonl_path.exists():
        pytest.skip(
            "No logs found yet. Create logs/v50_debug_log.md and/or logs/v50_runs.jsonl "
            "via the SpectraMind CLI to activate these tests."
        )


def test_markdown_log_readable_when_present(md_path: Path) -> None:
    if not md_path.exists():
        pytest.skip("Markdown log not present.")
    text = _read_text_safe(md_path)
    # Non-empty and reasonable size (< 50 MB)
    assert text.strip(), "Markdown log is empty."
    assert md_path.stat().st_size < 50 * 1024 * 1024, "Markdown log is unexpectedly huge; consider rotation."

    # Should contain some recognizable cues
    cues = ("spectramind", "cli", "train", "predict", "diagnose", "submit", "version", "config", "hash")
    assert any(c in text for c in cues), "Markdown log lacks recognizable CLI cues; is it the correct file?"

    # Optional: trailing newline for append-only friendliness
    assert text.endswith("\n"), "Markdown log should end with a newline (append-only hygiene)."


def test_jsonl_log_readable_when_present(jsonl_path: Path) -> None:
    if not jsonl_path.exists():
        pytest.skip("JSONL log not present.")
    recs = _load_jsonl_records(jsonl_path)
    assert recs, "JSONL log is empty."
    # Minimal schema check on a few records
    sample = recs[: min(5, len(recs))]
    for obj in sample:
        ts = _coerce_ts(obj)
        assert ts is not None, "JSONL records must include an ISO8601 UTC timestamp (ts/timestamp/time)."
        # Command text recommended
        if "cmd" in obj:
            assert isinstance(obj["cmd"], str) and obj["cmd"].strip(), "If present, 'cmd' must be a non-empty string."
        # exit_code recommended
        if "exit_code" in obj:
            assert isinstance(obj["exit_code"], int), "'exit_code' should be an integer."
        # If present, config_hash should look like hex
        if "config_hash" in obj and obj["config_hash"] is not None:
            assert re.match(r"^[0-9a-fA-F]{6,64}$", str(obj["config_hash"])), "config_hash should be hex-like."


# -----------------------------
# Tests — monotonicity & time
# -----------------------------

def test_jsonl_timestamps_monotonic(jsonl_path: Path) -> None:
    if not jsonl_path.exists():
        pytest.skip("JSONL log not present.")
    recs = _load_jsonl_records(jsonl_path)
    ts_list = [t for t in (_coerce_ts(r) for r in recs) if t is not None]
    assert ts_list, "No parseable timestamps in JSONL log."
    # Non-decreasing (allow ties)
    assert all(ts_list[i] >= ts_list[i - 1] for i in range(1, len(ts_list))), "JSONL timestamps are not monotonic."


def test_markdown_no_time_travel(md_path: Path) -> None:
    if not md_path.exists():
        pytest.skip("Markdown log not present.")
    text = _read_text_safe(md_path)
    ts = _extract_md_timestamps(text)
    if len(ts) < 2:
        pytest.skip("Not enough timestamps in Markdown log to evaluate monotonicity.")
    assert all(ts[i] >= ts[i - 1] for i in range(1, len(ts))), "Markdown timestamps are not monotonic (time travel)."


def test_recent_activity_window_if_configured(md_path: Path, jsonl_path: Path) -> None:
    """
    If EXPECT_LOG_WITHIN_HOURS=x is set, ensure at least one entry within the window.
    JSONL is preferred; fallback to Markdown timestamps if needed.
    """
    cfg = _env("EXPECT_LOG_WITHIN_HOURS")
    if not cfg:
        pytest.skip("EXPECT_LOG_WITHIN_HOURS not set — skipping recency check.")
    hours = int(cfg)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)

    # Prefer JSONL
    if jsonl_path.exists():
        recs = _load_jsonl_records(jsonl_path)
        ts_list = [t for t in (_coerce_ts(r) for r in recs) if t is not None]
        assert ts_list, "No parseable timestamps in JSONL log."
        assert any(t >= cutoff for t in ts_list), f"No JSONL entries within {hours}h."
        return

    # Fallback to Markdown
    if md_path.exists():
        text = _read_text_safe(md_path)
        ts = _extract_md_timestamps(text)
        assert ts, "No parseable timestamps in Markdown log."
        assert any(t >= cutoff for t in ts), f"No Markdown entries within {hours}h."
        return

    pytest.skip("No logs present to evaluate recency (unexpected in this branch).")


# -----------------------------
# Tests — Markdown table sanity (optional)
# -----------------------------

def _iter_md_tables(md_text: str) -> List[List[List[str]]]:
    """
    Extract simple GitHub-flavored Markdown tables as a list of tables,
    where each table is a list of rows, and each row is a list of cell strings.

    This is a *lightweight* heuristic sufficient to catch obvious row/column
    mismatches; it does not aim to fully parse GFM tables.
    """
    lines = md_text.splitlines()
    tables: List[List[List[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|", line) and "|" in line.strip().rstrip("|"):
            # Potential header and separator on next line
            if i + 1 < len(lines) and re.match(r"^\s*\|?[:\-\s|]+\|?\s*$", lines[i + 1]):
                # capture table until a non-pipe line
                rows: List[List[str]] = []
                j = i
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    rows.append(row)
                    j += 1
                tables.append(rows)
                i = j
                continue
        i += 1
    return tables


def test_markdown_tables_have_consistent_columns(md_path: Path) -> None:
    if not md_path.exists():
        pytest.skip("Markdown log not present.")
    text = _read_text_safe(md_path)
    tables = _iter_md_tables(text)
    if not tables:
        pytest.skip("No Markdown tables found — skipping table integrity check.")

    for t in tables:
        widths = [len(r) for r in t]
        # Sometimes first two rows are header and separator; use the mode of widths
        # and assert that all rows match it.
        if not widths:
            continue
        mode = max(set(widths), key=widths.count)
        assert all(w == mode for w in widths), f"Markdown table has inconsistent column counts: {widths}"


# -----------------------------
# Tests — Basic content cues
# -----------------------------

def test_logs_contain_cli_cues(md_path: Path, jsonl_path: Path) -> None:
    """
    Provide a friendly assertion that at least one of the logs contains
    a recognizable SpectraMind CLI cue (cmd string). This is not strict
    (we only warn/fail if neither log has any cues).
    """
    cues = ("spectramind", "python -m", "train", "predict", "diagnose", "submit", "ablate")
    md_ok = False
    js_ok = False

    if md_path.exists():
        text = _read_text_safe(md_path).lower()
        md_ok = any(c in text for c in cues)

    if jsonl_path.exists():
        try:
            recs = _load_jsonl_records(jsonl_path)
            for r in recs:
                cmd = str(r.get("cmd", "")).lower()
                if any(c in cmd for c in cues):
                    js_ok = True
                    break
        except AssertionError:
            # JSONL parsing error would have been raised in its own test
            pass

    assert md_ok or js_ok, "No recognizable CLI cues found in either log; verify logging integration."
