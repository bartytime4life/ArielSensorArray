#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/artifacts/test_report_manifest_integrity.py

SpectraMind V50 — Artifact & Manifest Integrity Tests
=====================================================

Purpose
-------
Validate that our artifact manifests and run reports are internally consistent,
auditable, and reproducible. This covers:

1) Manifest Presence & Shape
   • JSON and/or CSV manifests exist (canonical default: outputs/manifests/manifest_v50.{json,csv})
   • Required fields present (path, sha256, size, kind [optional but recommended], created_at [optional])

2) Digest & Filesystem Integrity
   • Every entry's sha256 and size match the actual file on disk
   • Paths are relative to repo root (recommended) and point to existing files

3) Cross-Format Consistency (JSON ↔ CSV)
   • If both JSON and CSV variants are available, verify they describe the same set of files

4) Run Report & Reproducibility Metadata
   • outputs/run_hash_summary_v50.json exists (if not, a dummy scenario is exercised in tmp space)
   • Contains minimum fields: config_hash, cli_version, build_timestamp
   • (Optional) Contains environment snapshot fields if produced by the pipeline

5) Graceful Fallback for Clean Repos
   • If the canonical manifests are not present (e.g., on a fresh clone), the test **creates**
     a minimal dummy artifact set in a temp directory, synthesizes a compliant manifest,
     and validates that round-trip integrity passes. This keeps CI green while still enforcing rules.

Usage
-----
• Pytest discovers and runs this file automatically.
• No internet or external services required.

Implementation Notes
--------------------
• We never mutate user artifacts. Any synthetic data is confined to pytest's tmp_path.
• SHA-256 is used for reproducible digests.
• Tests are verbose and self-documenting to align with the project's "No brevity" philosophy.

"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest


# =========================
# Utilities & Data Classes
# =========================

@dataclass(frozen=True)
class ManifestEntry:
    """Normalized representation of a single manifest row/record."""
    path: str
    sha256: str
    size: int
    kind: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def norm_path(self) -> str:
        """Return a normalized POSIX-like path for comparison."""
        return self.path.replace("\\", "/")

    def as_dict(self) -> Dict[str, Any]:
        """Dict view (useful for comparisons)."""
        d = {"path": self.norm_path, "sha256": self.sha256, "size": int(self.size)}
        if self.kind is not None:
            d["kind"] = self.kind
        if self.created_at is not None:
            d["created_at"] = self.created_at
        return d


def repo_root() -> Path:
    """
    Best-effort discovery of repository root based on this file's location.
    Falls back to current working directory if heuristic fails.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "spectramind.py").exists():
            return parent
    return Path.cwd().resolve()


def compute_sha256(p: Path, buf_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 digest of a file with streaming reads."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def now_iso8601() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_manifest_json(path: Path) -> List[ManifestEntry]:
    """
    Accepts structures:
      • {"artifacts": [ {path, sha256, size, kind?, created_at?}, ... ]}
      • or a top-level list [ {path, sha256, size, ...}, ... ]
    """
    raw = load_json(path)
    if isinstance(raw, dict) and "artifacts" in raw:
        rows = raw["artifacts"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise AssertionError(f"Unrecognized JSON manifest structure in {path}")

    entries: List[ManifestEntry] = []
    for i, r in enumerate(rows):
        try:
            entries.append(
                ManifestEntry(
                    path=str(r["path"]),
                    sha256=str(r["sha256"]),
                    size=int(r["size"]),
                    kind=str(r["kind"]) if "kind" in r and r["kind"] is not None else None,
                    created_at=str(r["created_at"]) if "created_at" in r and r["created_at"] is not None else None,
                )
            )
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"Malformed record at index {i} in {path}: {e}\nRecord: {r!r}")
    return entries


def read_manifest_csv(path: Path) -> List[ManifestEntry]:
    """
    CSV columns (min):
      • path, sha256, size
    Optional:
      • kind, created_at
    Extra columns are ignored.
    """
    entries: List[ManifestEntry] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "sha256", "size"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise AssertionError(f"CSV manifest {path} missing required columns: {sorted(missing)}")

        for i, row in enumerate(reader):
            try:
                entries.append(
                    ManifestEntry(
                        path=str(row["path"]),
                        sha256=str(row["sha256"]),
                        size=int(row["size"]),
                        kind=str(row["kind"]) if "kind" in row and row["kind"] else None,
                        created_at=str(row["created_at"]) if "created_at" in row and row["created_at"] else None,
                    )
                )
            except Exception as e:  # pragma: no cover
                raise AssertionError(f"Malformed CSV row {i} in {path}: {e}\nRow: {row!r}")
    return entries


def index_by_path(entries: Iterable[ManifestEntry]) -> Dict[str, ManifestEntry]:
    """Index entries by normalized path."""
    out: Dict[str, ManifestEntry] = {}
    for e in entries:
        key = e.norm_path
        if key in out:
            raise AssertionError(f"Duplicate artifact path in manifest: {key}")
        out[key] = e
    return out


def ensure_relative_to_root(p: Path, root: Path) -> bool:
    """
    Return True if p is inside root (or equal). False otherwise.
    Paths in manifest should be relative to repo root for portability.
    """
    try:
        _ = p.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def canonical_manifest_locations(base: Path) -> Dict[str, Path]:
    """
    Standard/canonical locations used by the repo. These can be absent safely.
    """
    return {
        "json": base / "outputs" / "manifests" / "manifest_v50.json",
        "csv": base / "outputs" / "manifests" / "manifest_v50.csv",
        "run_hash": base / "outputs" / "run_hash_summary_v50.json",
    }


# =========================================
# Synthetic (Dummy) Artifact/Manifest Maker
# =========================================

def create_dummy_artifacts_and_manifest(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """
    Create a tiny synthetic artifact set and both JSON and CSV manifests that
    comply with our schema. Also creates a minimal run_hash_summary_v50.json.
    Returns: (json_manifest, csv_manifest, run_hash_summary_json)
    """
    # Prepare directories
    out_dir = tmp_path / "outputs"
    art_dir = out_dir / "artifacts" / "dummy"
    man_dir = out_dir / "manifests"
    art_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    # Create a couple of tiny files to represent artifacts
    a = art_dir / "example_a.txt"
    b = art_dir / "nested" / "example_b.bin"
    b.parent.mkdir(parents=True, exist_ok=True)

    a.write_text("SpectraMind V50 — dummy artifact A\n", encoding="utf-8")
    b.write_bytes(b"\x00\x01\x02\x03\x04SpectraMind V50 dummy artifact B\xff")

    # Compute metadata
    def meta(p: Path, kind: Optional[str]) -> Dict[str, Any]:
        return {
            "path": str(p.relative_to(tmp_path)).replace("\\", "/"),
            "sha256": compute_sha256(p),
            "size": p.stat().st_size,
            "kind": kind,
            "created_at": now_iso8601(),
        }

    artifacts = [
        meta(a, "text"),
        meta(b, "binary"),
    ]

    # Write JSON manifest
    json_manifest = man_dir / "manifest_v50.json"
    save_json(json_manifest, {"artifacts": artifacts})

    # Write CSV manifest
    csv_manifest = man_dir / "manifest_v50.csv"
    with csv_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sha256", "size", "kind", "created_at"])
        writer.writeheader()
        for r in artifacts:
            writer.writerow(r)

    # Minimal run_hash_summary
    run_hash = out_dir / "run_hash_summary_v50.json"
    run_summary = {
        "config_hash": "deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcd",
        "cli_version": "v50.0.0-test",
        "build_timestamp": now_iso8601(),
        # Optional helpful fields:
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "notes": "Dummy summary for manifest integrity tests.",
    }
    save_json(run_hash, run_summary)

    return json_manifest, csv_manifest, run_hash


# =====================
# Pytest Helper Fixtures
# =====================

@pytest.fixture(scope="module")
def discovered_or_dummy_manifests(tmp_path_factory):
    """
    Try to discover real manifests in the repo. If not found, create a
    dummy set under a dedicated tmp_path and use those for tests.

    Returns a dict:
      {
        "root": Path,  # base path to resolve manifest-relative file paths
        "json": Optional[Path],
        "csv": Optional[Path],
        "run_hash": Optional[Path],
        "is_dummy": bool
      }
    """
    base = repo_root()
    locs = canonical_manifest_locations(base)

    found_any = False
    result: Dict[str, Any] = {
        "root": base,
        "json": None,
        "csv": None,
        "run_hash": None,
        "is_dummy": False,
    }

    if locs["json"].exists():
        result["json"] = locs["json"]
        found_any = True
    if locs["csv"].exists():
        result["csv"] = locs["csv"]
        found_any = True
    if locs["run_hash"].exists():
        result["run_hash"] = locs["run_hash"]
        found_any = True

    if not found_any:
        # build dummy in tmp
        tmp = tmp_path_factory.mktemp("manifest_integrity_dummy")
        jm, cm, rh = create_dummy_artifacts_and_manifest(tmp)
        result.update({"root": tmp, "json": jm, "csv": cm, "run_hash": rh, "is_dummy": True})

    return result


# ============
# The Test Set
# ============

class TestReportManifestIntegrity:
    """
    Comprehensive integrity checks for artifact manifests and run reports.
    """

    def test_manifest_presence_or_dummy(self, discovered_or_dummy_manifests):
        """
        Ensure we have something to test: either real manifests in the repo or a synthetic set.
        """
        info = discovered_or_dummy_manifests
        assert info["json"] or info["csv"] or info["run_hash"], "Neither real nor dummy manifests were found or created."
        # Assert files exist if paths are provided
        for key in ("json", "csv", "run_hash"):
            p: Optional[Path] = info.get(key)
            if p is not None:
                assert p.exists(), f"Expected {key} at {p} to exist."

    def test_json_manifest_shape_if_present(self, discovered_or_dummy_manifests):
        """
        Validate JSON manifest schema and basic record structure.
        """
        jm: Optional[Path] = discovered_or_dummy_manifests.get("json")
        if not jm:
            pytest.skip("JSON manifest not present; skipping JSON manifest shape test.")
        entries = read_manifest_json(jm)
        assert len(entries) > 0, f"JSON manifest {jm} has no artifacts."
        for e in entries:
            assert e.path and isinstance(e.path, str), f"Invalid path in {jm}: {e}"
            assert e.sha256 and len(e.sha256) == 64 and all(c in "0123456789abcdef" for c in e.sha256), \
                f"Invalid sha256 in {jm}: {e.sha256}"
            assert isinstance(e.size, int) and e.size >= 0, f"Invalid size in {jm}: {e.size}"

        # No duplicate paths
        _ = index_by_path(entries)

    def test_csv_manifest_shape_if_present(self, discovered_or_dummy_manifests):
        """
        Validate CSV manifest schema and basic record structure.
        """
        cm: Optional[Path] = discovered_or_dummy_manifests.get("csv")
        if not cm:
            pytest.skip("CSV manifest not present; skipping CSV manifest shape test.")
        entries = read_manifest_csv(cm)
        assert len(entries) > 0, f"CSV manifest {cm} has no artifacts."
        for e in entries:
            assert e.path and isinstance(e.path, str), f"Invalid path in {cm}: {e}"
            assert e.sha256 and len(e.sha256) == 64 and all(c in "0123456789abcdef" for c in e.sha256), \
                f"Invalid sha256 in {cm}: {e.sha256}"
            assert isinstance(e.size, int) and e.size >= 0, f"Invalid size in {cm}: {e.size}"

        # No duplicate paths
        _ = index_by_path(entries)

    def test_digest_and_size_match_filesystem(self, discovered_or_dummy_manifests):
        """
        For each entry, verify the sha256 and size exactly match the on-disk file.
        We allow either JSON or CSV (or both). If both exist, validate union of both.
        """
        base: Path = discovered_or_dummy_manifests["root"]
        jm: Optional[Path] = discovered_or_dummy_manifests.get("json")
        cm: Optional[Path] = discovered_or_dummy_manifests.get("csv")

        # Aggregate entries (map by path to dedupe if both exist)
        entries_map: Dict[str, ManifestEntry] = {}
        if jm:
            for e in read_manifest_json(jm):
                entries_map[e.norm_path] = e
        if cm:
            for e in read_manifest_csv(cm):
                entries_map[e.norm_path] = e

        if not entries_map:
            pytest.skip("No manifest entries available; skipping digest/size verification.")

        # Validate each
        for path_key, entry in entries_map.items():
            file_path = (base / entry.path).resolve()
            assert file_path.exists(), f"Manifest path not found on disk: {entry.path} (resolved: {file_path})"
            # Recommend relative paths inside repo/dummy root
            assert ensure_relative_to_root(file_path, base), \
                f"Manifest path should be within repo root: {entry.path} (resolved: {file_path}, base: {base})"

            actual_size = file_path.stat().st_size
            assert actual_size == entry.size, f"Size mismatch for {entry.path}: manifest {entry.size} != fs {actual_size}"

            actual_sha = compute_sha256(file_path)
            assert actual_sha == entry.sha256, f"SHA256 mismatch for {entry.path}: manifest {entry.sha256} != fs {actual_sha}"

    def test_json_csv_cross_consistency_if_both_present(self, discovered_or_dummy_manifests):
        """
        If both JSON and CSV manifests exist, they should cover the same set of files
        (sha256 and size must match per path).
        """
        jm: Optional[Path] = discovered_or_dummy_manifests.get("json")
        cm: Optional[Path] = discovered_or_dummy_manifests.get("csv")
        if not (jm and cm):
            pytest.skip("Both JSON and CSV manifests are not present; skipping cross-consistency test.")

        j_entries = index_by_path(read_manifest_json(jm))
        c_entries = index_by_path(read_manifest_csv(cm))

        # Compare the sets of paths
        j_paths = set(j_entries.keys())
        c_paths = set(c_entries.keys())
        assert j_paths == c_paths, f"JSON/CSV path sets differ.\nOnly in JSON: {sorted(j_paths - c_paths)}\nOnly in CSV: {sorted(c_paths - j_paths)}"

        # Compare record-level fields for matching paths
        for p in sorted(j_paths):
            je, ce = j_entries[p], c_entries[p]
            assert je.sha256 == ce.sha256, f"Digest mismatch for {p}: JSON {je.sha256} vs CSV {ce.sha256}"
            assert je.size == ce.size, f"Size mismatch for {p}: JSON {je.size} vs CSV {ce.size}"
            # 'kind' and 'created_at' may be absent; only compare if both exist
            if je.kind is not None and ce.kind is not None:
                assert je.kind == ce.kind, f"Kind mismatch for {p}: JSON {je.kind} vs CSV {ce.kind}"

    def test_run_hash_summary_minimum_fields(self, discovered_or_dummy_manifests):
        """
        Validate the presence and minimum shape of outputs/run_hash_summary_v50.json
        (or dummy equivalent). This ensures basic reproducibility metadata.
        """
        rh: Optional[Path] = discovered_or_dummy_manifests.get("run_hash")
        if not rh:
            pytest.skip("run_hash_summary_v50.json not present; skipping run hash validation.")
        data = load_json(rh)

        # Required minimal fields
        for k in ("config_hash", "cli_version", "build_timestamp"):
            assert k in data and isinstance(data[k], str) and data[k], f"Missing/empty field '{k}' in {rh}"

        # config_hash sanity (prefer 64-hex)
        conf = data["config_hash"]
        assert len(conf) >= 32, f"config_hash seems too short: {conf}"
        # If it's a pure hex hash, enforce hex chars; otherwise allow non-hex for legacy formats.
        if all(c in "0123456789abcdef" for c in conf.lower()):
            assert len(conf) in (40, 64), f"config_hash should be 40 or 64 hex chars when hex-like; got length {len(conf)}"

        # build_timestamp should be parseable (basic sanity check)
        ts = data["build_timestamp"]
        assert ("T" in ts or ts.endswith("Z") or ":" in ts or "-" in ts), f"build_timestamp doesn't look ISO-like: {ts}"

    def test_manifest_references_do_not_escape_root(self, discovered_or_dummy_manifests):
        """
        Security/portability check: paths should not traverse outside repo root (e.g., via ../../)
        """
        base: Path = discovered_or_dummy_manifests["root"]
        jm: Optional[Path] = discovered_or_dummy_manifests.get("json")
        cm: Optional[Path] = discovered_or_dummy_manifests.get("csv")

        entries: List[ManifestEntry] = []
        if jm:
            entries.extend(read_manifest_json(jm))
        if cm:
            entries.extend(read_manifest_csv(cm))
        if not entries:
            pytest.skip("No manifest entries available; skipping root-escape check.")

        for e in entries:
            resolved = (base / e.path).resolve()
            assert ensure_relative_to_root(resolved, base), \
                f"Manifest path escapes root: {e.path} (resolved: {resolved}, base: {base})"

    def test_manifest_rejects_duplicate_paths(self, discovered_or_dummy_manifests, tmp_path):
        """
        Synthesize a minimal manifest with a duplicate path to ensure our indexer raises.
        """
        # Build a tiny synthetic JSON manifest with duplicates
        man = tmp_path / "dup" / "manifest_v50.json"
        man.parent.mkdir(parents=True, exist_ok=True)
        dup_entries = [
            {"path": "a.txt", "sha256": "0" * 64, "size": 0},
            {"path": "a.txt", "sha256": "f" * 64, "size": 1},
        ]
        save_json(man, {"artifacts": dup_entries})

        with pytest.raises(AssertionError, match="Duplicate artifact path"):
            index_by_path(read_manifest_json(man))

    def test_manifest_rejects_malformed_rows(self, tmp_path):
        """
        Synthesize malformed rows to ensure our readers enforce schema.
        """
        # JSON: missing 'size'
        man_json = tmp_path / "bad" / "m.json"
        man_json.parent.mkdir(parents=True, exist_ok=True)
        save_json(man_json, {"artifacts": [{"path": "x", "sha256": "0" * 64}]})
        with pytest.raises(AssertionError, match="Malformed record"):
            _ = read_manifest_json(man_json)

        # CSV: missing required columns
        man_csv = tmp_path / "bad" / "m.csv"
        with man_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["only_path_here"])
            writer.writerow(["x"])
        with pytest.raises(AssertionError, match="missing required columns"):
            _ = read_manifest_csv(man_csv)

    def test_manifest_round_trip_dummy(self, tmp_path):
        """
        Create dummy artifacts, emit JSON & CSV manifests, then read them back and
        verify digest/size & cross-consistency. This is a self-contained regression check.
        """
        jm, cm, rh = create_dummy_artifacts_and_manifest(tmp_path)
        # Ensure shape is valid
        j_entries = read_manifest_json(jm)
        c_entries = read_manifest_csv(cm)
        assert len(j_entries) == len(c_entries) >= 1

        # Cross-check by path
        j_map = index_by_path(j_entries)
        c_map = index_by_path(c_entries)
        assert set(j_map) == set(c_map)

        # Verify digests/sizes actually match the real files
        base = tmp_path
        for k in j_map:
            je, ce = j_map[k], c_map[k]
            assert je.sha256 == ce.sha256
            assert je.size == ce.size
            p = (base / je.path).resolve()
            assert p.exists()
            assert compute_sha256(p) == je.sha256
            assert p.stat().st_size == je.size

        # Run hash summary sanity
        data = load_json(rh)
        assert "config_hash" in data and data["config_hash"], "Missing config_hash in dummy run hash summary."


# ======================
# Standalone Test Runner
# ======================

if __name__ == "__main__":  # pragma: no cover
    # Allow running this file directly for ad-hoc checks:
    #   python -m pytest -q tests/artifacts/test_report_manifest_integrity.py
    # Or:
    #   python tests/artifacts/test_report_manifest_integrity.py
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-q"]))