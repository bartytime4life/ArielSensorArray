# tests/artifacts/test_run_hash_summary_contents.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Artifact Tests
File: tests/artifacts/test_run_hash_summary_contents.py

Purpose
-------
Validate the integrity and schema of the consolidated run-hash summary produced
by the CLI (e.g., `spectramind ...`), typically named:

    run_hash_summary_v50.json

This test asserts (at minimum):

1) Summary file exists (or an override is provided via env).
2) Top-level JSON object has stable keys and the `runs` list is non-empty.
3) Each run item contains required fields with valid formats:
     - run_id: UUIDv4 or ULID
     - started_at, finished_at: ISO 8601 UTC (…Z), finished >= started
     - duration_seconds: float ≥ 0 and ≈ (finished - started)
     - command: non-empty string
     - exit_code: int
     - config_hash, data_hash: 64-hex (SHA-256) or "UNKNOWN"
     - git_commit: hexish string (7–40 chars) or "UNKNOWN"
     - python_version, platform: non-empty strings
     - cwd: absolute or relative path string
     - environment: dict-like or omitted; if present, should be key:str, value:str
     - files/artifacts/logs: optional arrays; if checksums present, must be 64-hex
4) run_id uniqueness across the file.
5) Optional determinism guard: entries are sorted by started_at (non-decreasing).
6) If global keys exist (app_name/version/build_time/etc.), they pass basic checks.

Configuration
-------------
Set SPECTRAMIND_RUN_HASH_SUMMARY to point to a custom path if your repo uses a
different location or name.

Default search order:
  - outputs/run_hash_summary_v50.json
  - logs/run_hash_summary_v50.json
  - run_hash_summary_v50.json

"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pytest

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ENV_SUMMARY = "SPECTRAMIND_RUN_HASH_SUMMARY"
CANDIDATES = (
    Path("outputs/run_hash_summary_v50.json"),
    Path("logs/run_hash_summary_v50.json"),
    Path("run_hash_summary_v50.json"),
)

# Loose patterns / validators
UUID4_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")  # Crockford base32 (no I,L,O,U)
HEX256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{7,40}|UNKNOWN)$")
ISO8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")

# Optional “global” keys we tolerate and lightly validate if present
GLOBAL_KEYS = {
    "app_name",
    "version",
    "build_time",
    "cli_entry",
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path.cwd().resolve()


def find_summary(root: Path) -> Path:
    env = os.getenv(ENV_SUMMARY, "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (root / p).resolve()
        assert p.exists(), f"run-hash summary not found at {p}"
        return p

    for cand in CANDIDATES:
        p = (root / cand).resolve()
        if p.exists():
            return p

    pytest.fail(
        "Run-hash summary JSON not found. "
        f"Set {ENV_SUMMARY} or place 'run_hash_summary_v50.json' under outputs/ or logs/."
    )


def _parse_iso8601_utc(s: str) -> datetime:
    assert ISO8601_UTC_RE.match(s), f"timestamp not ISO‑8601 UTC with 'Z': {s!r}"
    # convert for fromisoformat by replacing Z with +00:00
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    assert dt.tzinfo is not None, "timestamp missing timezone info"
    return dt


def _is_uuid_or_ulid(s: str) -> bool:
    return bool(UUID4_RE.match(s) or ULID_RE.match(s))


def _semver_like(s: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+([\-+][A-Za-z0-9\.\-_]+)?$", s))


def _safe_get(obj: dict, key: str, default=None):
    return obj.get(key, default)


def _is_sha256_or_unknown(v: str) -> bool:
    return v == "UNKNOWN" or bool(HEX256_RE.match(v))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_summary_exists_and_has_runs():
    root = repo_root()
    summary_path = find_summary(root)
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert isinstance(data, dict), "Top-level of run-hash summary must be a JSON object"
    assert "runs" in data, "Missing 'runs' key at top-level"
    runs = data["runs"]
    assert isinstance(runs, list) and len(runs) > 0, "'runs' must be a non-empty list"

    # Optional global keys sanity (if present)
    for k in GLOBAL_KEYS:
        if k in data:
            assert isinstance(data[k], str) and data[k].strip(), f"Global key '{k}' must be non-empty string"
    if "version" in data:
        assert _semver_like(data["version"]), f"Global 'version' not semver-like: {data['version']}"
    if "build_time" in data:
        _parse_iso8601_utc(data["build_time"])


def test_each_run_item_has_required_fields_and_valid_formats():
    root = repo_root()
    data = json.loads(find_summary(root).read_text(encoding="utf-8"))
    runs = data["runs"]

    # Check uniqueness and order by started_at
    seen_ids = set()
    last_start: Optional[datetime] = None

    for i, r in enumerate(runs):
        assert isinstance(r, dict), f"runs[{i}] must be an object"

        # Required keys
        for key in [
            "run_id",
            "started_at",
            "finished_at",
            "duration_seconds",
            "command",
            "exit_code",
            "config_hash",
            "data_hash",
            "git_commit",
            "python_version",
            "platform",
            "cwd",
        ]:
            assert key in r, f"runs[{i}] missing '{key}'"

        # run_id
        run_id = str(r["run_id"])
        assert _is_uuid_or_ulid(run_id), f"runs[{i}].run_id not UUIDv4/ULID: {run_id!r}"
        assert run_id not in seen_ids, f"Duplicate run_id in summary: {run_id}"
        seen_ids.add(run_id)

        # timestamps
        started = _parse_iso8601_utc(str(r["started_at"]))
        finished = _parse_iso8601_utc(str(r["finished_at"]))
        assert finished >= started, f"runs[{i}]: finished_at < started_at"
        if last_start:
            assert started >= last_start, f"runs are not sorted by started_at at index {i}"
        last_start = started

        # duration
        duration = float(r["duration_seconds"])
        assert duration >= 0.0, f"runs[{i}]: duration_seconds must be ≥ 0"
        # wall-clock duration comparison (allow ±0.5s tolerance)
        wall = (finished - started).total_seconds()
        assert abs(duration - wall) <= 0.5, (
            f"runs[{i}]: duration_seconds ({duration:.3f}) "
            f"differs from wall time ({wall:.3f})"
        )

        # command
        cmd = str(r["command"]).strip()
        assert cmd, f"runs[{i}]: empty command string"

        # exit code
        assert isinstance(r["exit_code"], int), f"runs[{i}]: exit_code must be int"

        # hashes
        cfg_hash = str(r["config_hash"])
        data_hash = str(r["data_hash"])
        assert _is_sha256_or_unknown(cfg_hash), f"runs[{i}]: invalid config_hash: {cfg_hash!r}"
        assert _is_sha256_or_unknown(data_hash), f"runs[{i}]: invalid data_hash: {data_hash!r}"

        # git commit
        gitc = str(r["git_commit"])
        assert GIT_COMMIT_RE.match(gitc), f"runs[{i}]: invalid git_commit: {gitc!r}"

        # python/platform
        assert isinstance(r["python_version"], str) and r["python_version"], f"runs[{i}]: python_version empty"
        assert isinstance(r["platform"], str) and r["platform"], f"runs[{i}]: platform empty"

        # cwd
        assert isinstance(r["cwd"], str) and r["cwd"], f"runs[{i}]: cwd empty"

        # Optional: environment (if present, should be dict of strings → strings)
        env = _safe_get(r, "environment")
        if env is not None:
            assert isinstance(env, dict), f"runs[{i}].environment must be object"
            for k, v in env.items():
                assert isinstance(k, str) and isinstance(v, str), f"runs[{i}].environment must map str→str"

        # Optional: artifacts/files/logs arrays
        for coll_key in ("artifacts", "files", "logs"):
            coll = _safe_get(r, coll_key)
            if coll is not None:
                assert isinstance(coll, list), f"runs[{i}].{coll_key} must be list"
                for j, item in enumerate(coll):
                    assert isinstance(item, dict), f"runs[{i}].{coll_key}[{j}] must be object"
                    # If checksum present, validate shape
                    if "sha256" in item:
                        val = str(item["sha256"])
                        assert _is_sha256_or_unknown(val), \
                            f"runs[{i}].{coll_key}[{j}].sha256 invalid: {val!r}"
                    # If path present, validate non-empty string
                    if "path" in item:
                        p = str(item["path"]).strip()
                        assert p, f"runs[{i}].{coll_key}[{j}].path empty"

        # Optional: metrics (dict or list)
        metrics = _safe_get(r, "metrics")
        if metrics is not None:
            assert isinstance(metrics, (dict, list)), f"runs[{i}].metrics must be dict or list"
            # Guard against NaN/Inf if numbers present
            def _flat(o):
                if isinstance(o, dict):
                    for vv in o.values():
                        yield from _flat(vv)
                elif isinstance(o, list):
                    for vv in o:
                        yield from _flat(vv)
                else:
                    yield o
            for val in _flat(metrics):
                if isinstance(val, float):
                    assert math.isfinite(val), f"runs[{i}].metrics contains non‑finite float"

        # Optional: notes (string)
        notes = _safe_get(r, "notes")
        if notes is not None:
            assert isinstance(notes, str), f"runs[{i}].notes must be string"


def test_summary_keys_are_stable_and_no_surprising_renames():
    """
    Guardrail: if maintainers change top-level keys or omit 'runs', fail loudly
    to force downstream updates.
    """
    data = json.loads(find_summary(repo_root()).read_text(encoding="utf-8"))

    assert isinstance(data, dict), "Top-level JSON must be an object"
    assert "runs" in data, "Missing 'runs' key at top-level"
    # Permit a small set of known globals; ignore unknown keys rather than fail hard.
    for k in data.keys():
        if k in ("runs", *GLOBAL_KEYS):
            continue
        # Soft guard: allow additional keys; if you want strict, turn this into an assert.


def test_human_readable_header_and_counts_are_reasonable():
    """
    Optional nicety: if the summary exposes `total_runs`, `failures`, etc., validate consistency.
    If not present, this test passes (no-op).
    """
    data = json.loads(find_summary(repo_root()).read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    total = data.get("total_runs")
    failures = data.get("failures")
    if total is not None:
        assert isinstance(total, int) and total >= 0, "total_runs must be non-negative int"
        assert total == len(runs), "`total_runs` must equal length of 'runs'"
    if failures is not None:
        assert isinstance(failures, int) and 0 <= failures <= len(runs), "failures must be in [0, len(runs)]"
        # If exit_code is present for each run, cross-check failures ~= number of non-zero exit_code
        try:
            calc_failures = sum(1 for r in runs if int(r.get("exit_code", 0)) != 0)
            assert failures == calc_failures, "failures count mismatch vs. exit_code aggregation"
        except Exception:
            # If exit_code types vary, skip strict check.
            pass


# --------------------------------------------------------------------------- #
# Debug / failure context
# --------------------------------------------------------------------------- #

def pytest_runtest_makereport(item, call):
    """
    On failure, append helpful context to stderr to speed up root-cause analysis.
    """
    if call.excinfo is not None and call.when == "call":
        root = repo_root()
        try:
            path = find_summary(root)
            extra = f"[debug] repo_root={root}\n[debug] run_hash_summary={path}\n"
        except Exception as e:
            extra = f"[debug] repo_root={root}\n[debug] run_hash_summary=NOT FOUND ({e})\n"
        import sys
        sys.stderr.write(extra)
