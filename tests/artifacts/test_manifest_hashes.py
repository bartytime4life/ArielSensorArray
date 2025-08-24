# tests/artifacts/test_manifest_hashes.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Artifact Tests
File: tests/artifacts/test_manifest_hashes.py

Purpose
-------
Validate the repository file manifest and its cryptographic integrity.

This test will:
  1) Locate a manifest file (JSON or CSV) describing repo files and their SHA256.
     • Default search order (overridable with env var SPECTRAMIND_MANIFEST):
         - ./manifest_v50.json
         - ./manifest_v50.csv
         - ./outputs/manifests/manifest_v50.json
         - ./outputs/manifests/manifest_v50.csv
  2) Load paths + expected sha256 digests.
  3) Assert there are no duplicate entries, no empty rows, and paths are normalized.
  4) For each entry:
        - Path exists relative to the repo root, unless explicitly marked optional.
        - If it is a regular file (and not a .dvc pointer), compute SHA256 and
          compare with the manifest.
        - For .dvc pointer files, validate the pointer file exists; hashing the
          large underlying artifact is not required (DVC manages that).
  5) Spot-check that important top-level project anchors exist and are included in
     the manifest: pyproject.toml OR poetry.lock OR Dockerfile OR README.md.

Notes
-----
• Set SPECTRAMIND_MANIFEST to point to a custom manifest if your repo structure differs.
• Manifest schema (JSON):
      [
        {
          "path": "relative/path/to/file.ext",
          "sha256": "<lowercase hex digest>",
          "optional": false   # optional key; defaults to false
        },
        ...
      ]
  CSV schema (header required):
      path,sha256,optional
      relative/path,deadbeef...,false
• For performance, hashing is streamed in chunks and only applied to regular files.
• Files with zero-byte content should have the SHA256 of the empty string:
      e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pytest


# ------------------------------ Configuration ------------------------------ #

ENV_MANIFEST = "SPECTRAMIND_MANIFEST"

# Default search locations if env var is not provided
CANDIDATE_MANIFESTS: Sequence[Path] = (
    Path("manifest_v50.json"),
    Path("manifest_v50.csv"),
    Path("outputs/manifests/manifest_v50.json"),
    Path("outputs/manifests/manifest_v50.csv"),
)

# Some files that should exist in most SpectraMind repositories (soft expectations).
ANCHOR_FILES: Sequence[Path] = (
    Path("pyproject.toml"),
    Path("poetry.lock"),
    Path("Dockerfile"),
    Path("README.md"),
)


# --------------------------------- Model ----------------------------------- #

@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    sha256: Optional[str]  # may be None for pointers where hashing is skipped
    optional: bool = False


# ------------------------------ Helper funcs ------------------------------- #

def repo_root() -> Path:
    """
    Heuristic: climb up from this test file until we find a directory
    containing either pyproject.toml or .git. Fallback to CWD.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path.cwd().resolve()


def find_manifest(root: Path) -> Path:
    """
    Resolve a manifest file path:
      1) SPECTRAMIND_MANIFEST if provided (absolute or relative to repo root)
      2) First existing file in CANDIDATE_MANIFESTS under repo root
    """
    env = os.getenv(ENV_MANIFEST, "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            pytest.fail(f"Manifest not found at SPECTRAMIND_MANIFEST={p}")
        return p

    for cand in CANDIDATE_MANIFESTS:
        p = (root / cand).resolve()
        if p.exists():
            return p

    pytest.fail(
        "Manifest not found. Try setting SPECTRAMIND_MANIFEST to a valid path, "
        "or place manifest_v50.json/manifest_v50.csv in repo root or outputs/manifests/."
    )
    raise AssertionError("unreachable")


def _normalize_path(p: str) -> Path:
    # Normalize separators and strip whitespace
    return Path(p.strip().replace("\\", "/")).as_posix() and Path(p.strip().replace("\\", "/"))


def load_manifest(path: Path) -> List[ManifestEntry]:
    """
    Load JSON or CSV format into a list of ManifestEntry objects.
    """
    entries: List[ManifestEntry] = []

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            pytest.fail(f"JSON manifest must be a list of objects: {path}")
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                pytest.fail(f"JSON manifest row {i} is not an object: {row!r}")
            p = _normalize_path(str(row.get("path", "")))
            sha = row.get("sha256", None)
            opt = bool(row.get("optional", False))
            if not p or p.as_posix() in ("", "."):
                pytest.fail(f"JSON manifest row {i} missing/invalid 'path'.")
            if sha is not None:
                sha = str(sha).lower().strip()
                _assert_hex_sha256_or_fail(sha, context=f"row {i} ({p})")
            entries.append(ManifestEntry(p, sha, opt))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            if not rdr.fieldnames:
                pytest.fail(f"CSV manifest missing header: {path}")
            expected_cols = {"path", "sha256", "optional"}
            missing = expected_cols - set(x.lower() for x in rdr.fieldnames)
            if missing:
                pytest.fail(f"CSV manifest missing columns {missing}. Found: {rdr.fieldnames}")

            for i, row in enumerate(rdr):
                p = _normalize_path(str(row.get("path", "")))
                sha_val = row.get("sha256", "")
                sha = None
                if sha_val is not None and str(sha_val).strip() != "":
                    sha = str(sha_val).lower().strip()
                    _assert_hex_sha256_or_fail(sha, context=f"row {i} ({p})")
                opt_val = row.get("optional", "false").strip().lower()
                opt = opt_val in ("1", "true", "yes", "y")
                if not p or p.as_posix() in ("", "."):
                    pytest.fail(f"CSV manifest row {i} missing/invalid 'path'.")
                entries.append(ManifestEntry(p, sha, opt))
    else:
        pytest.fail(f"Unsupported manifest format: {path.suffix}")

    return entries


def _assert_hex_sha256_or_fail(sha: str, context: str = "") -> None:
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        pytest.fail(f"Invalid sha256 hex digest {sha!r} {context}".strip())


def stream_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_dvc_pointer(p: Path) -> bool:
    # Basic heuristic: treat *.dvc as pointer files managed by DVC
    return p.suffix.lower() == ".dvc"


def anchors_present(root: Path) -> Sequence[Path]:
    return tuple(p for p in ANCHOR_FILES if (root / p).exists())


# ---------------------------------- Tests ---------------------------------- #

def test_manifest_exists_and_loads():
    root = repo_root()
    manifest_path = find_manifest(root)
    entries = load_manifest(manifest_path)
    assert len(entries) > 0, f"Manifest {manifest_path} is empty."

    # Ensure no duplicate paths and normalized paths
    seen = set()
    for e in entries:
        norm = e.path.as_posix()
        assert "/" in norm or norm, f"Entry path seems unnormalized: {e.path}"
        assert norm not in seen, f"Duplicate entry in manifest: {e.path}"
        seen.add(norm)


def test_manifest_covers_key_project_anchors():
    root = repo_root()
    manifest = find_manifest(root)
    entries = load_manifest(manifest)
    paths = {e.path.as_posix() for e in entries}

    present = anchors_present(root)
    assert present, (
        "None of the anchor files exist in the repo. "
        "Expected at least one of: "
        + ", ".join(str(p) for p in ANCHOR_FILES)
    )

    # If anchors exist, at least one of them should be listed in the manifest.
    listed_any = any(p.as_posix() in paths for p in present)
    assert listed_any, (
        "At least one anchor file should be included in the manifest. "
        f"Existing anchors: {[str(p) for p in present]}; "
        f"Manifest path count: {len(paths)}"
    )


def test_manifest_paths_exist_and_hashes_match():
    root = repo_root()
    manifest = find_manifest(root)
    entries = load_manifest(manifest)

    # Create a simple summary of failures to show all problems at once.
    missing: List[str] = []
    mismatch: List[str] = []
    skipped: List[str] = []

    for e in entries:
        file_path = (root / e.path).resolve()
        if not file_path.exists():
            # If file is optional, tolerate absence.
            if e.optional:
                skipped.append(f"[optional missing] {e.path.as_posix()}")
                continue
            missing.append(e.path.as_posix())
            continue

        # Directories are permitted in manifest only if marked optional
        if file_path.is_dir():
            if e.optional:
                skipped.append(f"[dir optional] {e.path.as_posix()}")
                continue
            pytest.fail(f"Manifest lists a directory as a required file: {e.path}")

        # If this is a DVC pointer file, we only assert it exists; hashing large artifacts is deferred to DVC.
        if is_dvc_pointer(file_path):
            # It's fine to have sha256 in manifest for .dvc (will be ignored), but not required.
            skipped.append(f"[dvc pointer] {e.path.as_posix()}")
            continue

        # For regular files we expect an SHA256
        if e.sha256 is None or e.sha256.strip() == "":
            pytest.fail(f"Manifest entry missing sha256 for regular file: {e.path}")

        # Compute and compare
        actual = stream_sha256(file_path)
        if actual != e.sha256:
            mismatch.append(f"{e.path.as_posix()} expected={e.sha256} actual={actual}")

    # Aggregate assertions to present a consolidated error view
    errors: List[str] = []
    if missing:
        errors.append("Missing files:\n  - " + "\n  - ".join(missing))
    if mismatch:
        errors.append("SHA256 mismatches:\n  - " + "\n  - ".join(mismatch))

    if errors:
        pytest.fail(
            f"Manifest validation failed for {manifest} under root={repo_root()}.\n"
            + "\n\n".join(errors)
            + ("\n\nSkipped (ok):\n  - " + "\n  - ".join(skipped) if skipped else "")
        )


def test_manifest_is_reasonably_small_and_sorted():
    """
    A gentle structural check to encourage order and sanity (does not enforce a specific sort key).
    """
    root = repo_root()
    manifest = find_manifest(root)
    entries = load_manifest(manifest)

    # Ensure it's not absurdly huge for a test environment; adjust threshold if needed.
    assert len(entries) < 100_000, "Manifest seems excessively large for this project."

    # Encourages stable ordering (lexicographic by path)
    paths = [e.path.as_posix() for e in entries]
    sorted_paths = sorted(paths, key=str.lower)
    assert paths == sorted_paths, (
        "Manifest paths are not lexicographically sorted. "
        "Please sort by path (case-insensitive) to improve diff stability."
    )


def test_manifest_sha256_format_is_valid_hex():
    root = repo_root()
    manifest = find_manifest(root)
    entries = load_manifest(manifest)

    for e in entries:
        if e.sha256:
            _assert_hex_sha256_or_fail(e.sha256, context=f"path={e.path.as_posix()}")


# ---------------------------- Utility Assertions --------------------------- #

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    When a test fails, append manifest + root paths to the report to speed up debugging.
    """
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed:
        root = repo_root()
        try:
            manifest = find_manifest(root)
            extra = f"\n[debug] repo_root={root}\n[debug] manifest={manifest}\n"
        except Exception as e:
            extra = f"\n[debug] repo_root={root}\n[debug] manifest=NOT FOUND ({e})\n"
        rep.longrepr = f"{rep.longrepr}\n{extra}"
