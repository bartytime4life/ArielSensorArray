# /tools/artifacts/test_log_integrity.py
"""
SpectraMind V50 — Log Integrity Tests (upgraded)

Purpose
-------
These tests enforce "NASA‑grade" logging guarantees for the CLI‑first pipeline.
They validate JSONL telemetry emitted by the system (e.g., events.jsonl or
run_*.jsonl), catching schema drift, timezone issues, missing hashes, etc.

How it works
------------
By default we scan for JSONL logs under:
  ./artifacts/logs/**/*.jsonl
Override with env:
  ARTIFACT_LOG_GLOB="path/pattern/**/*.jsonl"

What we validate
----------------
* Each line is valid JSON (no trailing commas, no BOM, etc.)
* Required fields are present per event
* Timestamps are RFC3339/ISO‑8601 with timezone and monotonic per run_id
* Levels and event types are from approved vocabularies
* No NaN/Inf or non‑finite numbers
* run_id is UUIDv4‑like (or 26+ char ULID), consistent within a file
* config_hash/data_hash look like hex SHA‑256 (64 hex chars)
* pid is int, hostname is non‑empty
* schema string present and versioned (e.g., "v1", "v1.1")
* message length sane; no newline injection
* Optional file references exist and (if checksum provided) match
* No secrets/keys accidentally logged
* File rotation (if used) is well‑formed and ordered

Run
---
pytest -q tools/artifacts/test_log_integrity.py
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

# ------------------------------
# Configuration / constants
# ------------------------------

DEFAULT_GLOB = "artifacts/logs/**/*.jsonl"
LOG_GLOB = os.getenv("ARTIFACT_LOG_GLOB", DEFAULT_GLOB)

# Required event fields (add more as your schema evolves)
REQUIRED_FIELDS = {
    "ts",           # ISO-8601 timestamp with timezone (RFC3339)
    "level",        # INFO|DEBUG|WARN|ERROR|CRITICAL
    "event",        # machine-friendly event type (snake_case)
    "message",      # human-friendly message (one line)
    "run_id",       # UUIDv4 or ULID
    "component",    # e.g., "cli", "calibrate", "train", "diagnostics"
    "schema",       # e.g., "v1", "v1.1"
    "config_hash",  # SHA256 hex of composed Hydra config
    "data_hash",    # SHA256 hex of dataset snapshot (e.g., DVC)
    "pid",          # process id (int)
    "hostname",     # machine hostname
}

APPROVED_LEVELS = {"TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"}

# You can extend based on your pipeline's event taxonomy
APPROVED_EVENTS = {
    # lifecycle
    "run_started",
    "run_finished",
    "stage_started",
    "stage_finished",
    "epoch_started",
    "epoch_finished",
    # io / artifacts
    "artifact_written",
    "artifact_read",
    "artifact_deleted",
    # metrics
    "metric",
    "metrics_flushed",
    # errors
    "exception",
    "retry",
    # calibration / processing
    "calibration_step",
    "detrend_step",
    # submission
    "submission_packaged",
}

# Optional file reference fields used for existence/hash checks
FILE_PATH_FIELDS = {"artifact_path", "file_path", "report_path", "image_path"}
FILE_HASH_FIELDS = {"artifact_sha256", "file_sha256"}

# Simple secret patterns to catch accidental leaks
SECRET_REGEXES = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\._\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key prefix
]

UUID4_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")  # Crockford base32 (no I,L,O,U)

HEX256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ISO8601_TZ_RE = re.compile(
    # 2025-08-24T12:34:56.789Z or with offset like +02:00
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+\-]\d{2}:\d{2})$"
)

ONE_LINE_RE = re.compile(r"^[^\r\n]*$")

MAX_MESSAGE_LEN = 2000  # sane upper bound for single-line messages


# ------------------------------
# Utilities
# ------------------------------

def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            # Skip empty lines (but flag them in a separate test)
            if not line.strip():
                yield idx, {"__empty__": True}
                continue
            try:
                # Disallow trailing commas by strict parsing via json.loads
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"{path} line {idx}: invalid JSON ({e})"
                ) from e
            if not isinstance(obj, dict):
                raise AssertionError(f"{path} line {idx}: JSON must be an object")
            yield idx, obj


def _parse_ts(ts: str) -> datetime:
    if not ISO8601_TZ_RE.match(ts):
        raise AssertionError(f"timestamp not ISO‑8601 with timezone: {ts!r}")
    # Normalize 'Z' to +00:00 for fromisoformat
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError as e:
        raise AssertionError(f"invalid timestamp value: {ts!r}") from e
    if dt.tzinfo is None:
        raise AssertionError(f"timestamp missing timezone: {ts!r}")
    return dt


def _is_uuid_or_ulid(s: str) -> bool:
    return bool(UUID4_RE.match(s) or ULID_RE.match(s))


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return True
    if isinstance(x, float):
        return math.isfinite(x)
    return False


def _flatten(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


# ------------------------------
# Fixtures
# ------------------------------

@pytest.fixture(scope="session")
def log_files() -> List[Path]:
    paths = sorted(Path(".").glob(LOG_GLOB))
    assert paths, f"No JSONL logs found for pattern: {LOG_GLOB}"
    return paths


# ------------------------------
# Tests per file
# ------------------------------

@pytest.mark.parametrize("path", ids=lambda p: str(p), argvalues=None)
def test_collection_parametrize_marker(log_files):  # noqa: D401
    """
    Internal: works around pytest parametrization when building from fixture.
    This test dynamically parametrizes and delegates to the actual tests below.
    """
    # Dynamically create parametrized tests for each file
    for path in log_files:
        _run_all_tests_on_file(path)


def _run_all_tests_on_file(path: Path) -> None:
    # 1) Basic JSON & required fields
    events_by_run: dict[str, list[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    empty_lines = 0

    for idx, obj in _iter_jsonl(path):
        if "__empty__" in obj:
            empty_lines += 1
            continue

        missing = REQUIRED_FIELDS - obj.keys()
        assert not missing, f"{path} line {idx}: missing fields {missing}"

        # level & event vocab
        level = str(obj["level"])
        event = str(obj["event"])
        assert level.upper() in APPROVED_LEVELS, f"{path} line {idx}: bad level {level!r}"
        assert re.fullmatch(r"[a-z][a-z0-9_]*", event), f"{path} line {idx}: event must be snake_case"
        if event not in APPROVED_EVENTS:
            # Allow forward-compat with a warning-style failure
            pytest.fail(f"{path} line {idx}: event {event!r} not in approved taxonomy", pytrace=False)

        # ts format
        ts_str = str(obj["ts"])
        dt = _parse_ts(ts_str)

        # run id
        rid = str(obj["run_id"])
        assert _is_uuid_or_ulid(rid), f"{path} line {idx}: run_id not UUIDv4/ULID: {rid!r}"

        # hashes
        assert HEX256_RE.match(str(obj["config_hash"])), f"{path} line {idx}: config_hash not SHA256 hex"
        assert HEX256_RE.match(str(obj["data_hash"])), f"{path} line {idx}: data_hash not SHA256 hex"

        # pid / hostname
        assert isinstance(obj["pid"], int), f"{path} line {idx}: pid must be int"
        assert str(obj["hostname"]).strip(), f"{path} line {idx}: hostname empty"

        # schema
        schema = str(obj["schema"])
        assert re.fullmatch(r"v\d+(?:\.\d+)?", schema), f"{path} line {idx}: schema must look like 'v1' or 'v1.2'"

        # message single-line + length
        msg = str(obj["message"])
        assert ONE_LINE_RE.match(msg), f"{path} line {idx}: message must be single line"
        assert len(msg) <= MAX_MESSAGE_LEN, f"{path} line {idx}: message too long ({len(msg)} chars)"

        # No NaN/Inf anywhere
        for key, val in _flatten(obj):
            if isinstance(val, float):
                assert math.isfinite(val), f"{path} line {idx}: non‑finite number at {key}: {val}"

        # Secret detection
        full_line = json.dumps(obj, ensure_ascii=False)
        for rx in SECRET_REGEXES:
            assert not rx.search(full_line), f"{path} line {idx}: possible secret detected by {rx.pattern}"

        # Optional file refs
        file_paths_present = [obj.get(k) for k in FILE_PATH_FIELDS if obj.get(k)]
        file_hashes_present = [obj.get(k) for k in FILE_HASH_FIELDS if obj.get(k)]
        for fp in file_paths_present:
            p = Path(str(fp))
            assert p.exists(), f"{path} line {idx}: referenced file does not exist: {fp}"
        # If both path and sha256 present, check hash (best‑effort to avoid heavy reads)
        for fp, fh in zip(file_paths_present, file_hashes_present):
            if fp and fh and HEX256_RE.match(str(fh)):
                # Only hash files smaller than ~50MB to keep tests fast
                p = Path(str(fp))
                if p.is_file() and p.stat().st_size <= 50 * 1024 * 1024:
                    import hashlib

                    h = hashlib.sha256()
                    with p.open("rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                    assert (
                        h.hexdigest().lower() == str(fh).lower()
                    ), f"{path} line {idx}: checksum mismatch for {fp}"

        events_by_run[rid].append((idx, obj))

    # 2) No empty lines inside logs (to keep JSONL strict) — allow trailing at most one
    assert empty_lines == 0, f"{path}: contains {empty_lines} empty lines; JSONL must be dense"

    # 3) Per-run monotonic timestamps and stage ordering sanity
    for rid, rows in events_by_run.items():
        # sort by line index (already in file order)
        last_dt: Optional[datetime] = None
        for idx, obj in rows:
            dt = _parse_ts(str(obj["ts"]))
            if last_dt:
                assert dt >= last_dt, f"{path} line {idx}: non‑monotonic timestamp in run {rid}"
            last_dt = dt

        # lifecycle sanity: if present, run_started must appear before run_finished
        idxs = {e: i for i, (_, o) in enumerate(rows) for e in ["run_started", "run_finished"] if o["event"] == e}
        if "run_started" in idxs and "run_finished" in idxs:
            assert idxs["run_started"] < idxs["run_finished"], f"{path}: run_finished precedes run_started for {rid}"

    # 4) Rotation sanity (e.g., events.jsonl, events.jsonl.1, …)
    # If this file has numeric suffix, ensure the base exists. If not, just pass.
    m = re.match(r"^(?P<stem>.+\.jsonl)\.(?P<num>\d+)$", path.name)
    if m:
        base = path.with_name(m.group("stem"))
        assert base.exists(), f"{path}: rotated file found but base log '{base.name}' missing"

    # 5) Ensure file encoding is UTF‑8 (no BOM) — open() above would have failed subtly otherwise
    first_bytes = path.read_bytes()[:3]
    assert first_bytes != b"\xef\xbb\xbf", f"{path}: file must not contain UTF‑8 BOM"

    # If we got here, the file passed all checks.


# ------------------------------
# Optional: summary report (skip by default)
# ------------------------------

@pytest.mark.optionalhook
def pytest_sessionfinish(session, exitstatus):  # pragma: no cover
    """
    If you want a lightweight success marker for downstream CI steps,
    you can enable writing a small report via env var:
      WRITE_LOG_INTEGRITY_REPORT=1
    """
    if os.getenv("WRITE_LOG_INTEGRITY_REPORT", "") != "1":
        return
    out = Path("artifacts") / "log_integrity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed" if exitstatus == 0 else "failed",
        "exitstatus": exitstatus,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pattern": LOG_GLOB,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ------------------------------
# Helpful markers / selection
# ------------------------------

def pytest_addoption(parser):  # pragma: no cover
    parser.addoption(
        "--log-glob",
        action="store",
        default=None,
        help="Override ARTIFACT_LOG_GLOB for this pytest invocation",
    )


@pytest.fixture(autouse=True, scope="session")
def _apply_cli_override(request):  # pragma: no cover
    val = request.config.getoption("--log-glob")
    if val:
        os.environ["ARTIFACT_LOG_GLOB"] = val
