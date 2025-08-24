# tests/artifacts/test_report_manifest_integrity.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Artifact Tests
File: tests/artifacts/test_report_manifest_integrity.py

Purpose
-------
Validate the integrity of the *report manifest* produced by diagnostics and
report-generation tools (e.g., generate_html_report.py, dashboards, packaging).

This test ensures:
  1) A manifest exists (or a custom path is provided via env).
  2) Schema sanity (required keys, types, no duplicates).
  3) Each file entry exists, byte-size matches, SHA256 matches.
  4) Timestamps are ISO‑8601 UTC with 'Z'.
  5) content_type is from an allowed list (below) and matches basic heuristics.
  6) For HTML, local asset links (img/src, link href, script src) resolve.
  7) For CSV/JSON assets referenced by HTML (e.g., embedded charts), links exist.
  8) Optional dependency DAG is acyclic (simple cycle check).
  9) The manifest is deterministic: entries are sorted lexicographically by path.

Assumptions
-----------
• Default manifest search locations (overridable with env var
  SPECTRAMIND_REPORT_MANIFEST):
     - outputs/reports/report_manifest.json
     - outputs/diagnostics/report_manifest.json
     - outputs/report_manifest.json
     - report_manifest.json
• Manifest format (JSON). Two forms are accepted:
    A) List[Entry]
    B) {"reports": List[Entry], ...}
  Where Entry has:
    path          : str (relative path to repo root)
    sha256        : str (64 hex chars)
    bytes         : int (file size in bytes)
    content_type  : str (e.g., "text/html", "image/png", "application/pdf")
    created_at    : str (ISO‑8601 UTC, e.g., "2025-08-18T21:14:55Z")
    title         : str (optional, recommended for human-readable dashboards)
    depends_on    : List[str] (optional list of other entry paths)
    tags          : List[str] (optional)
• If you generate different keys, you can add them—tests ignore unknowns.

"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest

# ------------------------------ Configuration ------------------------------ #

ENV_REPORT_MANIFEST = "SPECTRAMIND_REPORT_MANIFEST"

CANDIDATES = (
    Path("outputs/reports/report_manifest.json"),
    Path("outputs/diagnostics/report_manifest.json"),
    Path("outputs/report_manifest.json"),
    Path("report_manifest.json"),
)

# Allowed content types and minimal file signature heuristics
ALLOWED_CONTENT_TYPES = {
    "text/html",
    "text/csv",
    "application/json",
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "application/pdf",
    "text/plain",
    "application/zip",
}

# HTML link extraction patterns (very lightweight, not a full parser by design)
HTML_HREF_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"'#]+)["']""", re.IGNORECASE)

ISO8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")

HEX256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


# --------------------------------- Model ----------------------------------- #

@dataclass(frozen=True)
class ReportEntry:
    path: Path
    sha256: str
    size: int
    ctype: str
    created_at: str
    title: Optional[str]
    depends_on: Tuple[Path, ...]
    tags: Tuple[str, ...]


# ------------------------------ Helper funcs ------------------------------- #

def repo_root() -> Path:
    """Find repo root by seeking pyproject.toml or .git upwards; fallback to CWD."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return Path.cwd().resolve()


def find_manifest(root: Path) -> Path:
    env = os.getenv(ENV_REPORT_MANIFEST, "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (root / p).resolve()
        assert p.exists(), f"Manifest not found at {p}"
        return p
    for cand in CANDIDATES:
        p = (root / cand).resolve()
        if p.exists():
            return p
    pytest.fail(
        "Report manifest not found. "
        f"Set {ENV_REPORT_MANIFEST} or place a manifest at one of: "
        + ", ".join(str(c) for c in CANDIDATES)
    )


def _normalize_path(s: str) -> Path:
    # Normalize to posix separators
    return Path(s.strip().replace("\\", "/"))


def load_manifest(path: Path) -> List[ReportEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        entries = raw.get("reports", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        pytest.fail(f"Manifest top-level must be a list or dict with 'reports': {type(raw)}")

    out: List[ReportEntry] = []
    for i, row in enumerate(entries):
        if not isinstance(row, dict):
            pytest.fail(f"Manifest entry {i} not an object: {row!r}")

        p = _normalize_path(str(row.get("path", "")))
        if not p.as_posix() or p.as_posix() in (".", "/"):
            pytest.fail(f"Entry {i} has invalid path: {p!r}")

        sha = str(row.get("sha256", "")).lower().strip()
        if not HEX256_RE.match(sha):
            pytest.fail(f"Entry {i} sha256 invalid for {p}: {sha!r}")

        b = row.get("bytes", None)
        if not isinstance(b, int) or b < 0:
            pytest.fail(f"Entry {i} bytes invalid for {p}: {b!r}")

        ctype = str(row.get("content_type", "")).strip()
        if ctype not in ALLOWED_CONTENT_TYPES:
            pytest.fail(f"Entry {i} content_type not allowed for {p}: {ctype!r}")

        created = str(row.get("created_at", "")).strip()
        if not ISO8601_UTC_RE.match(created):
            pytest.fail(f"Entry {i} created_at not ISO‑8601 UTC for {p}: {created!r}")

        title = row.get("title", None)
        if title is not None and not str(title).strip():
            pytest.fail(f"Entry {i} has empty title for {p}")

        depends = tuple(_normalize_path(x) for x in row.get("depends_on", []) or [])
        tags = tuple(str(x) for x in row.get("tags", []) or [])

        out.append(ReportEntry(p, sha, int(b), ctype, created, title if title else None, depends, tags))
    return out


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def detect_html_links(html_text: str) -> List[str]:
    # Extract href/src values; ignore anchors (#) already filtered by regex
    return [m.group(1).strip() for m in HTML_HREF_RE.finditer(html_text)]


def _content_heuristic(path: Path, ctype: str) -> None:
    """
    Minimal magic header checks to catch mismatches (fast, not exhaustive).
    """
    try:
        head = path.read_bytes()[:8]
    except Exception:
        head = b""

    if ctype == "application/pdf":
        assert head.startswith(b"%PDF"), f"{path}: does not start with %PDF"
    elif ctype == "image/png":
        assert head.startswith(b"\x89PNG\r\n\x1a\n"), f"{path}: not PNG signature"
    elif ctype == "image/jpeg":
        assert head[:2] == b"\xff\xd8", f"{path}: not JPEG SOI"
    elif ctype == "image/svg+xml":
        # SVG is XML/text; no strict magic, but should contain '<svg' near top
        txt = path.read_text(encoding="utf-8", errors="ignore")[:2000]
        assert "<svg" in txt.lower(), f"{path}: missing <svg tag for SVG"
    elif ctype == "text/html":
        txt = path.read_text(encoding="utf-8", errors="ignore")[:2000].lower()
        assert "<html" in txt or "<!doctype html" in txt, f"{path}: html doctype not detected"
    # For JSON/CSV/TXT/ZIP, we skip deep checks here.


def _check_html_assets(root: Path, html_path: Path) -> List[str]:
    """
    Parse a small subset of HTML to extract local resource references, ensuring they exist.
    Only relative local links are validated; http(s) links are ignored.
    """
    errs: List[str] = []
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    links = detect_html_links(text)
    for href in links:
        if href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:"):
            continue
        # Resolve relative to the HTML file
        p = (html_path.parent / href).resolve()
        # Constrain within repo
        try:
            p.relative_to(root)
        except ValueError:
            errs.append(f"{html_path}: link escapes repo root -> {href}")
            continue
        if not p.exists():
            errs.append(f"{html_path}: missing linked asset -> {href}")
    return errs


def _assert_no_cycles(entries: List[ReportEntry]) -> None:
    # Build adjacency and perform Kahn’s algorithm for cycle detection
    path_to_idx = {e.path.as_posix(): i for i, e in enumerate(entries)}
    adj = defaultdict(set)  # node -> set of deps
    indeg = defaultdict(int)
    for e in entries:
        for dep in e.depends_on:
            dposix = dep.as_posix()
            if dposix not in path_to_idx:
                # Non-strict: external deps are ignored for cycles
                continue
            if dposix not in adj[e.path.as_posix()]:
                adj[e.path.as_posix()].add(dposix)
                indeg[dposix] += 1
            # ensure nodes are registered
            _ = indeg[e.path.as_posix()]

    # Kahn
    q = deque([k for k in path_to_idx.keys() if indeg[k] == 0])
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # If we visited fewer than the number of nodes with edges, there is a cycle
    nodes_with_edges = set(indeg.keys()) | set(adj.keys())
    if nodes_with_edges and visited < len(nodes_with_edges):
        pytest.fail("Dependency graph contains a cycle among report entries.")


# ---------------------------------- Tests ---------------------------------- #

def test_manifest_exists_and_loads():
    root = repo_root()
    manifest_path = find_manifest(root)
    entries = load_manifest(manifest_path)
    assert entries, f"Manifest {manifest_path} is empty."

    # Paths must be unique and sorted lexicographically (case-insensitive)
    paths = [e.path.as_posix() for e in entries]
    assert len(paths) == len(set(paths)), "Duplicate paths found in manifest."
    assert paths == sorted(paths, key=str.lower), "Manifest paths are not lexicographically sorted."


def test_each_entry_file_exists_and_hash_matches():
    root = repo_root()
    manifest_path = find_manifest(root)
    entries = load_manifest(manifest_path)

    missing: List[str] = []
    size_mismatch: List[str] = []
    hash_mismatch: List[str] = []

    for e in entries:
        f = (root / e.path).resolve()
        if not f.exists():
            missing.append(e.path.as_posix())
            continue

        # Size check
        actual_size = f.stat().st_size
        if actual_size != e.size:
            size_mismatch.append(f"{e.path.as_posix()} expected={e.size} actual={actual_size}")

        # Hash check
        actual_hash = sha256_file(f)
        if actual_hash.lower() != e.sha256.lower():
            hash_mismatch.append(f"{e.path.as_posix()} expected={e.sha256} actual={actual_hash}")

        # Minimal content heuristics
        _content_heuristic(f, e.ctype)

    errs: List[str] = []
    if missing:
        errs.append("Missing files:\n  - " + "\n  - ".join(missing))
    if size_mismatch:
        errs.append("Size mismatches:\n  - " + "\n  - ".join(size_mismatch))
    if hash_mismatch:
        errs.append("SHA256 mismatches:\n  - " + "\n  - ".join(hash_mismatch))
    if errs:
        pytest.fail(f"Report manifest integrity failed for {manifest_path}.\n" + "\n\n".join(errs))


def test_timestamps_and_content_types_valid():
    root = repo_root()
    manifest_path = find_manifest(root)
    entries = load_manifest(manifest_path)
    for e in entries:
        assert e.ctype in ALLOWED_CONTENT_TYPES, f"{e.path}: content_type not allowed: {e.ctype}"
        assert ISO8601_UTC_RE.match(e.created_at), f"{e.path}: created_at not ISO‑8601 UTC: {e.created_at}"


def test_html_assets_exist_when_referenced():
    root = repo_root()
    manifest_path = find_manifest(root)
    entries = load_manifest(manifest_path)

    html_errs: List[str] = []
    for e in entries:
        if e.ctype != "text/html":
            continue
        f = (root / e.path).resolve()
        if not f.exists():
            # file existence handled elsewhere
            continue
        html_errs.extend(_check_html_assets(root, f))

    if html_errs:
        pytest.fail("HTML asset link failures:\n  - " + "\n  - ".join(html_errs))


def test_dependency_graph_is_acyclic():
    root = repo_root()
    manifest_path = find_manifest(root)
    entries = load_manifest(manifest_path)
    _assert_no_cycles(entries)


def test_manifest_has_at_least_one_human_viewable_report():
    """
    A sanity guard: ensure there is at least one HTML or PDF in the manifest so that
    humans can open something meaningful from a diagnostics run.
    """
    root = repo_root()
    entries = load_manifest(find_manifest(root))
    has_viewable = any(e.ctype in ("text/html", "application/pdf") for e in entries)
    assert has_viewable, "Manifest should contain at least one HTML or PDF report."


# --------------------------- Debug failure context ------------------------- #

def pytest_runtest_makereport(item, call):
    """
    On failure, append helpful paths to stderr.
    """
    if call.excinfo is not None and call.when == "call":
        root = repo_root()
        try:
            manifest_path = find_manifest(root)
            extra = f"[debug] repo_root={root}\n[debug] report_manifest={manifest_path}\n"
        except Exception as e:
            extra = f"[debug] repo_root={root}\n[debug] report_manifest=NOT FOUND ({e})\n"
        # Write to stderr (pytest will include in output)
        import sys
        sys.stderr.write(extra)
