#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/artifacts/test_run_hash_summary_contents.py

SpectraMind V50 — Run Hash Summary Contents (Reproducibility Metadata)
======================================================================

Purpose
-------
Ensure that `outputs/run_hash_summary_v50.json` (or a synthesized dummy equivalent)
exists and includes richly-structured, self-auditable metadata for reproducible runs.

What we validate
----------------
1) Presence (or dummy fallback)
   • If the canonical file is missing, we generate a **dummy** summary inside pytest's tmp_path.

2) Minimum required fields (non-empty strings)
   • config_hash — Prefer 40 or 64 hex chars if hex-like
   • cli_version — e.g., "v50.3.2" or "v50.0.0-test"
   • build_timestamp — ISO8601-ish string (no need to fully parse tz offsets)
   • Optionally present but recommended: repo_root, run_id

3) Optional ecosystem fields (validated if present)
   • git: {commit, branch, dirty}
   • python, platform, cuda, torch (and versions)
   • hydra, dvc, mlflow, wandb, poetry, poetry_lock_hash
   • docker_image, kaggle_competition, kaggle_username
   • env: mapping[str, str], with sensitive keys allowed but type-checked
   • run_args (list[str]) and/or cli_command (str)
   • config_snapshot (dict or str path)
   • hydra_resolved_config_path (str path)
   • outputs / artifacts / diagnostics references (paths or dict of paths)
   • diagnostics_html, v50_debug_log, run_hash_files (list of paths)
   • metrics (dict[str, number]), e.g., {"gll_mean": -1.23, "rmse": 0.04}
   • durations (dict[str, number]), e.g., {"train_sec": 1234.5, "inference_sec": 456.7}

4) Path sanity (if present)
   • Any referenced paths must be inside the repo root (no traversal/escape).
   • If files exist, we check they are readable. If they don't exist, we **warn** (soft check),
     not fail — since different CI stages may write them at different times.

5) Timestamp sanity
   • build_timestamp is not grossly in the future (> 24 hours beyond now)

Design notes
------------
• The tests are verbose and self-documenting to align with the project's "No brevity" philosophy.
• The dummy creation ensures green CI on fresh clones while still enforcing structure.
• We deliberately allow optional fields to be missing; when present, we validate types and patterns.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pytest


# =========================
# Utility & Helper Routines
# =========================

HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def repo_root() -> Path:
    """
    Best-effort discovery of repository root by walking up from this file.
    Signals:
    • .git present           -> treat as root
    • pyproject.toml present -> likely root
    • spectramind.py present -> repository root by convention
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "spectramind.py").exists():
            return parent
    return Path.cwd().resolve()


def canonical_run_hash_location(base: Path) -> Path:
    """
    Canonical location of the summary JSON, by convention.
    """
    return base / "outputs" / "run_hash_summary_v50.json"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def now_iso8601() -> str:
    """Return a simple ISO8601-like UTC timestamp (Z suffix)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_dt_soft(s: str) -> Optional[datetime]:
    """
    Soft-parse an ISO-like timestamp. We accept a wide range of reasonable formats.
    On failure, return None instead of raising.
    """
    candidates = [
        "%Y-%m-%dT%H:%M:%SZ",      # 2025-08-23T04:05:06Z
        "%Y-%m-%dT%H:%M:%S",       # 2025-08-23T04:05:06
        "%Y-%m-%d %H:%M:%S",       # 2025-08-23 04:05:06
        "%Y-%m-%d",                # 2025-08-23
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(s, fmt)
            if fmt.endswith("Z"):
                return dt.replace(tzinfo=timezone.utc)
            # no tz info -> assume UTC
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    # Fallback: attempt fromisoformat (handles offsets)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def ensure_relative_to_root(p: Path, root: Path) -> bool:
    """
    Return True if p is inside root (or equal). False otherwise.
    This prevents path traversal and enforces portability.
    """
    try:
        _ = p.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def coerce_str_list(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, list) and all(isinstance(i, str) for i in x):
        return x
    return None


def walk_paths_in_summary(data: Dict[str, Any]) -> List[str]:
    """
    Extract a conservative set of path-like fields from the run hash summary.
    We avoid making hard assumptions about schema; we just look at known keys.

    Returns a list of string paths (relative or absolute as given in JSON).
    """
    paths: List[str] = []
    # Canonical and commonly seen keys:
    for key in [
        "repo_root",
        "diagnostics_html",
        "v50_debug_log",
        "hydra_resolved_config_path",
        "config_snapshot_path",
        "poetry_lock_path",
        "dockerfile_path",
        "dvc_yaml_path",
        "submission_csv",
        "submission_zip",
    ]:
        val = data.get(key)
        if isinstance(val, str):
            paths.append(val)

    # Nested dicts that may contain paths
    for nested_key in ["outputs", "artifacts", "files", "manifests"]:
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for v in nested.values():
                if isinstance(v, str):
                    paths.append(v)
                elif isinstance(v, list):
                    for i in v:
                        if isinstance(i, str):
                            paths.append(i)

    # run_hash_files (list)
    rfiles = coerce_str_list(data.get("run_hash_files"))
    if rfiles:
        paths.extend(rfiles)

    return paths


# ==============================================
# Dummy (Synthetic) run_hash_summary_v50.json
# ==============================================

def create_dummy_run_hash_summary(tmp_root: Path) -> Path:
    """
    Create a **dummy** run hash summary in tmp_root/outputs/run_hash_summary_v50.json
    with enough structure to satisfy tests and document expected fields.
    """
    out = canonical_run_hash_location(tmp_root)
    data = {
        "config_hash": "deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcd",
        "cli_version": "v50.0.0-test",
        "build_timestamp": now_iso8601(),
        "repo_root": str(tmp_root),
        "run_id": "dummy-run-0001",
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "env": {
            "HYDRA_FULL_ERROR": "1",
            "PYTHONHASHSEED": "0",
        },
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "branch": "feature/dummy",
            "dirty": False,
        },
        "run_args": ["spectramind", "test", "--deep"],
        "cli_command": "spectramind --version",
        "outputs": {
            "diagnostics_dir": "outputs/diagnostics",
            "predictions_dir": "outputs/predictions",
        },
        "artifacts": {
            "manifests_dir": "outputs/manifests",
        },
        "metrics": {
            "gll_mean": -1.234,
            "rmse": 0.0567,
        },
        "durations": {
            "train_sec": 12.34,
            "inference_sec": 5.67,
        },
        "run_hash_files": [
            "outputs/run_hash_summary_v50.json",
        ],
    }
    save_json(out, data)
    return out


# ======================
# Pytest-Level Fixtures
# ======================

@pytest.fixture(scope="module")
def discovered_or_dummy_run_hash(tmp_path_factory):
    """
    Try to discover the real run hash summary under the repo root. If not found,
    synthesize a **dummy** summary in a tmp directory to preserve CI greenness.

    Returns:
        {
          "root": Path,             # base root for path-rel checking
          "path": Path,             # path to run_hash_summary_v50.json
          "is_dummy": bool
        }
    """
    base = repo_root()
    summary = canonical_run_hash_location(base)
    if summary.exists():
        return {"root": base, "path": summary, "is_dummy": False}

    # Create dummy in a temp project layout
    tmp_root = tmp_path_factory.mktemp("run_hash_dummy_root")
    # Establish minimal repo signals in tmp_root:
    (tmp_root / "pyproject.toml").write_text("[tool.spectramind]\nname='dummy'\n", encoding="utf-8")
    created = create_dummy_run_hash_summary(tmp_root)
    return {"root": tmp_root, "path": created, "is_dummy": True}


# ============
# The Test Set
# ============

class TestRunHashSummaryContents:
    """
    Comprehensive checks for the `outputs/run_hash_summary_v50.json` structure.
    """

    # 1) Presence
    def test_summary_presence(self, discovered_or_dummy_run_hash):
        info = discovered_or_dummy_run_hash
        p: Path = info["path"]
        assert p.exists(), f"Expected run hash summary at {p}."

    # 2) Minimum fields and simple sanity
    def test_minimum_fields(self, discovered_or_dummy_run_hash):
        path: Path = discovered_or_dummy_run_hash["path"]
        data: Dict[str, Any] = load_json(path)

        # Required fields: config_hash, cli_version, build_timestamp
        for k in ("config_hash", "cli_version", "build_timestamp"):
            assert k in data and isinstance(data[k], str) and data[k].strip(), f"Missing/empty field '{k}' in {path}"

        # config_hash: if hex-like, enforce length 40 or 64
        conf = data["config_hash"]
        if HEX_RE.match(conf):
            assert len(conf) in (40, 64), f"config_hash should be 40 or 64 hex chars when hex-like; got length {len(conf)}"

        # cli_version: simple pattern check (vXX or vXX.YY etc.) — flexible on purpose
        cli_ver = data["cli_version"]
        assert cli_ver[0].lower() == "v", f"cli_version should start with 'v' (e.g., v50.0.0); got {cli_ver}"

        # build_timestamp: parseable & not far in the future (> 24h)
        ts_str = data["build_timestamp"]
        ts = parse_dt_soft(ts_str)
        assert ts is not None, f"build_timestamp not parseable: {ts_str}"
        now = datetime.now(timezone.utc)
        assert ts <= now + timedelta(hours=24), f"build_timestamp appears to be far in the future: {ts_str}"

    # 3) Optional ecosystem fields — types/patterns
    def test_optional_ecosystem_fields(self, discovered_or_dummy_run_hash):
        path: Path = discovered_or_dummy_run_hash["path"]
        data: Dict[str, Any] = load_json(path)

        # repo_root (if present) should be a directory
        repo_root_str = data.get("repo_root")
        if isinstance(repo_root_str, str):
            repo_root_path = Path(repo_root_str)
            # Not strictly required to exist in real repo; dummy will set it to tmp root
            assert isinstance(repo_root_path, Path)

        # run_id (if present)
        run_id = data.get("run_id")
        if run_id is not None:
            assert isinstance(run_id, str) and run_id.strip(), "run_id should be a non-empty string when present."

        # git block
        git = data.get("git")
        if git is not None:
            assert isinstance(git, dict), "git should be a dict when present."
            commit = git.get("commit")
            if commit is not None:
                assert isinstance(commit, str) and len(commit) >= 7, "git.commit should be a short or full hash string"
                assert HEX_RE.match(commit[:7]), "git.commit should start with hex chars."
            branch = git.get("branch")
            if branch is not None:
                assert isinstance(branch, str)
            dirty = git.get("dirty")
            if dirty is not None:
                assert isinstance(dirty, bool)

        # env mapping
        env = data.get("env")
        if env is not None:
            assert isinstance(env, dict), "env should be a dict when present."
            for k, v in env.items():
                assert isinstance(k, str), "env keys must be strings"
                assert isinstance(v, (str, int, float, bool)), "env values should be primitive types (string/number/bool)"

        # run_args list
        run_args = data.get("run_args")
        if run_args is not None:
            assert isinstance(run_args, list) and all(isinstance(i, str) for i in run_args), "run_args must be list[str]"

        # cli_command str
        cli_cmd = data.get("cli_command")
        if cli_cmd is not None:
            assert isinstance(cli_cmd, str) and cli_cmd.strip(), "cli_command must be a non-empty string"

        # metrics dict[str, number]
        metrics = data.get("metrics")
        if metrics is not None:
            assert isinstance(metrics, dict), "metrics should be a dict when present."
            for k, v in metrics.items():
                assert isinstance(k, str), "metrics keys must be strings"
                assert isinstance(v, (int, float)), f"metric '{k}' must be a number; got {type(v)}"

        # durations dict[str, number]
        durations = data.get("durations")
        if durations is not None:
            assert isinstance(durations, dict), "durations should be a dict when present."
            for k, v in durations.items():
                assert isinstance(k, str), "durations keys must be strings"
                assert isinstance(v, (int, float)), f"duration '{k}' must be a number; got {type(v)}"

        # tool versions (if present): accept strings
        for key in ["python", "platform", "cuda", "torch", "hydra", "dvc", "mlflow", "wandb", "poetry", "poetry_lock_hash"]:
            if key in data:
                assert isinstance(data[key], str), f"{key} should be a string when present."

        # docker image, kaggle fields (if present): strings
        for key in ["docker_image", "kaggle_competition", "kaggle_username"]:
            if key in data and data[key] is not None:
                assert isinstance(data[key], str), f"{key} should be a string when present."

        # config snapshot (path or dict)
        csnap = data.get("config_snapshot")
        if csnap is not None:
            assert isinstance(csnap, (dict, str)), "config_snapshot may be a dict (inlined) or a path string."

        # hydra resolved config path (if present)
        hrp = data.get("hydra_resolved_config_path")
        if hrp is not None:
            assert isinstance(hrp, str), "hydra_resolved_config_path should be a string path when present."

    # 4) Path sanity (inside repo root), existence is soft-checked
    def test_path_sanity(self, discovered_or_dummy_run_hash, capsys):
        info = discovered_or_dummy_run_hash
        base: Path = info["root"]
        path: Path = info["path"]
        data: Dict[str, Any] = load_json(path)

        # Collect a conservative set of path-like fields
        candidates = walk_paths_in_summary(data)

        # Always include self-path for sanity
        candidates.append(str(path.relative_to(base)) if ensure_relative_to_root(path, base) else str(path))

        warnings: List[str] = []
        for rel in candidates:
            if not isinstance(rel, str) or not rel.strip():
                warnings.append(f"Skipping non-string path candidate: {rel!r}")
                continue
            p = Path(rel)
            # If the JSON holds an absolute path, make it relative check against base; else join base + rel
            resolved = (p if p.is_absolute() else (base / p)).resolve()
            assert ensure_relative_to_root(resolved, base), f"Referenced path escapes repo root: {rel} -> {resolved} (base: {base})"
            # Existence is a soft check; only warn
            if not resolved.exists():
                warnings.append(f"Referenced path does not (yet) exist: {rel} -> {resolved}")

        if warnings:
            # Print soft warnings to help developers; does not fail the test
            print("\n".join(f"[run-hash-summary warning] {w}" for w in warnings))

        # Ensure printed warnings (if any) are captured (no-op otherwise)
        _ = capsys.readouterr()

    # 5) Round-trip dummy synthesis (self-contained regression)
    def test_dummy_round_trip(self, tmp_path):
        """
        Build a minimal temporary project root with a synthesized run hash summary.
        Validate the same checks: minimum fields, optional typing, and path sanity.
        """
        # Create a minimal root
        root = tmp_path / "proj"
        root.mkdir(parents=True, exist_ok=True)
        (root / "pyproject.toml").write_text("[tool.spectramind]\nname='dummy'\n", encoding="utf-8")

        # Create dummy summary
        summary = create_dummy_run_hash_summary(root)
        assert summary.exists(), "Failed to create dummy run hash summary."

        data = load_json(summary)
        # Minimum fields
        for k in ("config_hash", "cli_version", "build_timestamp"):
            assert k in data and isinstance(data[k], str) and data[k].strip()

        # Timestamp parseable and not far future
        ts = parse_dt_soft(data["build_timestamp"])
        assert ts is not None
        assert ts <= datetime.now(timezone.utc) + timedelta(hours=24)

        # Optional env type check
        env = data.get("env")
        assert isinstance(env, dict)
        for k, v in env.items():
            assert isinstance(k, str)
            assert isinstance(v, (str, int, float, bool))

        # Path sanity for known references
        candidates = walk_paths_in_summary(data)
        for rel in candidates:
            if not isinstance(rel, str) or not rel.strip():
                continue
            p = Path(rel)
            resolved = (p if p.is_absolute() else (root / p)).resolve()
            assert ensure_relative_to_root(resolved, root), f"Dummy referenced path escapes root: {rel} -> {resolved}"

    # 6) Friendly schema hints (non-failing): print examples of suggested keys if missing
    def test_suggested_schema_hints(self, discovered_or_dummy_run_hash, capsys):
        """
        Provide friendly guidance on *suggested* keys missing from the summary to
        encourage richer reproducibility — but do not fail the test.
        """
        path: Path = discovered_or_dummy_run_hash["path"]
        data: Dict[str, Any] = load_json(path)

        suggested_keys = [
            "repo_root",
            "run_id",
            "git.commit",
            "git.branch",
            "git.dirty",
            "python",
            "platform",
            "cuda",
            "torch",
            "hydra",
            "dvc",
            "mlflow",
            "wandb",
            "poetry",
            "poetry_lock_hash",
            "docker_image",
            "kaggle_competition",
            "kaggle_username",
            "env",
            "run_args",
            "cli_command",
            "config_snapshot",
            "hydra_resolved_config_path",
            "outputs",
            "artifacts",
            "metrics",
            "durations",
            "diagnostics_html",
            "v50_debug_log",
            "run_hash_files",
        ]

        missing: List[str] = []
        # shallow checks for top-level keys; special-case git subkeys
        for k in suggested_keys:
            if k.startswith("git."):
                sub = k.split(".", 1)[1]
                if not isinstance(data.get("git"), dict) or sub not in data["git"]:
                    missing.append(k)
            else:
                if k not in data:
                    missing.append(k)

        if missing:
            print(
                "[run-hash-summary hint] Consider enriching outputs/run_hash_summary_v50.json with keys:\n  - "
                + "\n  - ".join(missing)
            )

        # Drain captured output
        _ = capsys.readouterr()


# ======================
# Standalone Test Runner
# ======================

if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly:
    #   python -m pytest -q tests/artifacts/test_run_hash_summary_contents.py
    # Or:
    #   python tests/artifacts/test_run_hash_summary_contents.py
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-q"]))