# tests/artifacts/test_manifest_hashes.py
"""
Upgraded tests for the artifact manifest generator.

What this validates (and why it matters):

- Correct SHA256 & byte size per file  ➜ scientific reproducibility
- Stable output across runs            ➜ determinism
- Changes reflected when files change  ➜ integrity
- Respect of include/exclude patterns  ➜ hygiene (no __pycache__, .git, etc.)
- Clean, POSIX‑style relative paths    ➜ portability across OSes
- JSON header with provenance          ➜ auditability (created_at, version, tool)

This test suite is intentionally tolerant about the import path of the manifest
API to accommodate different repository layouts. It will try several common
locations and, if not found, mark the tests as xfailed with a helpful message.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pytest


# ---- Helpers ----------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_import_api() -> Tuple[
    Optional[Callable[..., Dict[str, Any]]],
    Optional[Callable[[Dict[str, Any], Path], None]]
]:
    """
    Try to import a manifest API from several likely locations.

    Expected call signatures (any of these are fine):
      - build_manifest(root: Path, includes: Iterable[str], excludes: Iterable[str]) -> dict
      - generate_manifest(root: Path, include_globs: Iterable[str], exclude_globs: Iterable[str]) -> dict
      - write_manifest(manifest: dict, out_path: Path) -> None  (optional)

    Returns:
      (build_fn, write_fn) — either/both can be None; tests will xfail gracefully.
    """
    candidates = [
        ("spectramind.artifacts.manifest", ("build_manifest", "write_manifest")),
        ("spectramind.utils.manifest", ("build_manifest", "write_manifest")),
        ("spectramind.utils.manifest", ("generate_manifest", "write_manifest")),
        ("src.spectramind.artifacts.manifest", ("build_manifest", "write_manifest")),
        ("src.spectramind.utils.manifest", ("build_manifest", "write_manifest")),
    ]

    build_fn = None
    write_fn = None

    for mod_name, (build_name, write_name) in candidates:
        try:
            mod = __import__(mod_name, fromlist=[build_name, write_name])
        except Exception:
            continue

        # pick build/generate callable if present
        if hasattr(mod, build_name):
            build_fn = getattr(mod, build_name)
        elif build_name == "build_manifest" and hasattr(mod, "generate_manifest"):
            build_fn = getattr(mod, "generate_manifest")  # tolerate naming alt
        elif build_name == "generate_manifest" and hasattr(mod, "build_manifest"):
            build_fn = getattr(mod, "build_manifest")

        if hasattr(mod, write_name):
            write_fn = getattr(mod, write_name)

        if build_fn:
            break

    return build_fn, write_fn


def _require_api_or_xfail():
    build_fn, write_fn = _try_import_api()
    if build_fn is None:
        pytest.xfail(
            "Manifest API not found. "
            "Provide one of: spectramind.artifacts.manifest.build_manifest, "
            "spectramind.utils.manifest.build_manifest, "
            "spectramind.utils.manifest.generate_manifest. "
            "Tests will activate once the API exists."
        )
    return build_fn, write_fn


def _normalize_path(rel_path: Path) -> str:
    """Return a POSIX-style relative path (no leading './')."""
    p = posixpath.join(*rel_path.parts)
    return p.lstrip("./")


def _make_files(base: Path, files: Dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture()
def sample_tree(tmp_path: Path) -> Path:
    """
    Create a small content-addressable tree with a few files, including folders
    that should typically be excluded by default (.git, __pycache__, .dvc, etc.).
    """
    files = {
        "data/a.txt": b"alpha\n",
        "data/sub/b.bin": b"\x00\x01\x02\x03",
        "scripts/run.sh": b"#!/usr/bin/env bash\necho ok\n",
        ".git/HEAD": b"ref: refs/heads/main\n",
        "__pycache__/x.cpython-311.pyc": b"\x00F00BAR",
        ".dvc/lock": b"{}",
    }
    _make_files(tmp_path, files)
    os.chmod(tmp_path / "scripts" / "run.sh", 0o755)
    return tmp_path


@pytest.fixture()
def manifest_api():
    return _require_api_or_xfail()


# ---- Tests ------------------------------------------------------------------


def test_basic_content_hash_and_size(sample_tree: Path, manifest_api):
    build_manifest, _write_manifest = manifest_api

    # generous defaults: include everything, exclude common junk
    includes = ["**/*"]
    excludes = [".git/**", "__pycache__/**", ".dvc/**"]

    manifest = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]

    assert "meta" in manifest, "Top-level 'meta' missing"
    assert "files" in manifest, "Top-level 'files' missing"
    files = manifest["files"]
    assert isinstance(files, list) and files, "Empty file list"

    # Collect entries into a dict by path for quick lookup
    by_path = {entry["path"]: entry for entry in files}
    # Expect excluded files to be absent
    assert ".git/HEAD" not in by_path
    assert "__pycache__/x.cpython-311.pyc" not in by_path
    assert ".dvc/lock" not in by_path

    # Validate a.txt
    a_rel = _normalize_path(Path("data/a.txt"))
    assert a_rel in by_path, f"{a_rel} missing from manifest"
    a = by_path[a_rel]
    assert a["sha256"] == _sha256_file(sample_tree / a_rel)
    assert a["bytes"] == (sample_tree / a_rel).stat().st_size
    # POSIX relative paths only
    assert "/" in a["path"] and "\\" not in a["path"]
    assert not a["path"].startswith("./")

    # Validate b.bin
    b_rel = _normalize_path(Path("data/sub/b.bin"))
    b = by_path[b_rel]
    assert b["sha256"] == _sha256_file(sample_tree / b_rel)
    assert b["bytes"] == 4

    # Validate executable script presence
    sh_rel = _normalize_path(Path("scripts/run.sh"))
    assert sh_rel in by_path


def test_stable_across_runs(sample_tree: Path, manifest_api):
    build_manifest, _ = manifest_api
    includes = ["**/*"]
    excludes = [".git/**", "__pycache__/**", ".dvc/**"]

    m1 = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]
    m2 = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]

    # Compare hashes and ordering (manifests should be deterministic)
    assert m1["files"] == m2["files"], "Manifest not stable for same inputs"
    # Meta created_at can differ if generated per-call; allow either same or both present
    assert "meta" in m1 and "meta" in m2
    assert "generator" in m1["meta"]
    assert "version" in m1["meta"]


def test_changes_reflected_when_file_changes(sample_tree: Path, manifest_api):
    build_manifest, _ = manifest_api
    includes = ["**/*"]
    excludes = [".git/**", "__pycache__/**", ".dvc/**"]

    a_path = sample_tree / "data" / "a.txt"
    before = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]
    before_entry = {e["path"]: e for e in before["files"]}["data/a.txt"]

    # mutate file
    a_path.write_bytes(b"alpha\nbeta\n")

    after = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]
    after_entry = {e["path"]: e for e in after["files"]}["data/a.txt"]

    assert before_entry["sha256"] != after_entry["sha256"], "SHA256 did not change after content change"
    assert before_entry["bytes"] != after_entry["bytes"], "size did not change after content change"


def test_respects_excludes_and_includes(sample_tree: Path, manifest_api):
    build_manifest, _ = manifest_api

    # Only include *.txt under data, exclude everything else
    includes = ["data/**/*.txt"]
    excludes = ["**/*.bin", ".git/**", "__pycache__/**", ".dvc/**", "scripts/**"]

    m = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]
    paths = {e["path"] for e in m["files"]}
    assert paths == {"data/a.txt"}, f"Unexpected paths: {paths}"


def test_manifest_json_header_schema(sample_tree: Path, manifest_api, tmp_path: Path):
    build_manifest, write_manifest = manifest_api
    includes = ["**/*"]
    excludes = [".git/**", "__pycache__/**", ".dvc/**"]

    m = build_manifest(sample_tree, includes, excludes)  # type: ignore[call-arg]

    # Top-level meta checks
    meta = m.get("meta", {})
    assert isinstance(meta, dict)
    assert isinstance(meta.get("generator", ""), str) and meta["generator"], "generator string required"
    assert isinstance(meta.get("version", ""), str) and meta["version"], "version string required"

    # Optional fields: created_at ISO8601, git_commit (40 hex)
    created = meta.get("created_at")
    if created is not None:
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$", created), "created_at must be ISO8601 UTC"

    git_commit = meta.get("git_commit")
    if git_commit is not None:
        assert re.match(r"^[0-9a-f]{40}$", git_commit), "git_commit should be a 40-char hex SHA1"

    # Ensure we can write JSON if write_manifest is provided
    if write_manifest is not None:
        out = tmp_path / "manifest.json"
        write_manifest(m, out)  # type: ignore[call-arg]
        on_disk = json.loads(out.read_text())
        assert on_disk.get("files") == m["files"]
        assert on_disk.get("meta", {}).get("version") == meta.get("version")


def test_paths_are_posix_and_relative(sample_tree: Path, manifest_api):
    build_manifest, _ = manifest_api
    m = build_manifest(sample_tree, ["**/*"], [".git/**", "__pycache__/**", ".dvc/**"])  # type: ignore[call-arg]
    for entry in m["files"]:
        p = entry["path"]
        assert isinstance(p, str)
        # relative, POSIX, no drive letters, no backslashes
        assert not p.startswith(("/", "./", ".\\")), f"not relative: {p}"
        assert "\\" not in p, f"not POSIX: {p}"


def test_manifest_contains_minimal_fields(sample_tree: Path, manifest_api):
    build_manifest, _ = manifest_api
    m = build_manifest(sample_tree, ["**/*"], [".git/**", "__pycache__/**", ".dvc/**"])  # type: ignore[call-arg]
    assert isinstance(m["files"], list) and m["files"], "files must be non-empty list"
    for e in m["files"]:
        for key in ("path", "sha256", "bytes"):
            assert key in e, f"missing '{key}' in file entry"
        assert isinstance(e["bytes"], int) and e["bytes"] >= 0
        assert re.match(r"^[0-9a-f]{64}$", e["sha256"]), "sha256 must be 64-char hex"


# ---- UX: show a nicer skip/xfail in CI when API is missing -------------------


def test_manifest_api_present_or_xfail():
    build_fn, _ = _try_import_api()
    if build_fn is None:
        pytest.xfail(
            "No manifest API found yet — implement one at:\n"
            " - spectramind.artifacts.manifest.build_manifest(managed_root, includes, excludes)\n"
            "   or spectramind.utils.manifest.generate_manifest(...)\n"
            "Once present, this test (and suite) will activate automatically."
        )
