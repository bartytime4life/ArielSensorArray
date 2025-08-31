```python
# scr/server/artifacts.py
# =============================================================================
# 📦 SpectraMind V50 — Artifacts Locator, Metadata, and Secure I/O Utilities
# -----------------------------------------------------------------------------
# Scope
#   Thin, server-safe helpers for locating, listing, and reading CLI-produced
#   artifacts (HTML/PNG/CSV/JSON/etc). This module NEVER computes diagnostics;
#   artifacts are created by the CLI (e.g., `spectramind diagnose dashboard`)
#   into ARTIFACTS_DIR.
#
# Design highlights
#   • Path traversal protection (sandboxed to ARTIFACTS_DIR)
#   • Rich metadata (size, mtime/ctime, sha256, etag, mime)
#   • Flexible listings (by ext, glob, depth, pagination, search)
#   • Common finders (summary JSON, UMAP/t-SNE HTML, plots)
#   • Safe binary reads with optional HTTP Range support
#   • Lightweight manifest generator (for GUI prefetch)
#
# Environment
#   • PROJECT_ROOT (default: repo root resolved from this file)
#   • ARTIFACTS_DIR (default: {PROJECT_ROOT}/artifacts)
#   • DIAGNOSTICS_SUMMARY_FILE (default: diagnostic_summary.json)
#
# Notes
#   • The module is framework-agnostic; FastAPI/Starlette handlers can import
#     these helpers directly to serve files with proper headers.
#   • All public APIs return plain Python types (dict/list/bytes/Path).
#   • No external deps; standard library only.
# =============================================================================

from __future__ import annotations

import fnmatch
import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, Any

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).resolve()
DIAGNOSTICS_SUMMARY_FILE = os.getenv("DIAGNOSTICS_SUMMARY_FILE", "diagnostic_summary.json")

# Hard limit used by read helpers to avoid excessive memory usage (bytes)
DEFAULT_MAX_READ_BYTES = int(os.getenv("ARTIFACTS_MAX_READ_BYTES", str(50 * 1024 * 1024)))  # 50 MiB

# Default MIME fallbacks
mimetypes.init()
_MIME_FALLBACKS = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".parquet": "application/octet-stream",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------

class ArtifactSecurityError(PermissionError):
    """Raised on path traversal or sandbox violations."""


class ArtifactNotFound(FileNotFoundError):
    """Raised when an expected artifact is missing."""


class ArtifactReadTooLarge(IOError):
    """Raised when a read exceeds the configured safety threshold."""


class ArtifactRangeNotSatisfiable(IOError):
    """Raised when an HTTP Range-like request is invalid for the resource."""


# -----------------------------------------------------------------------------
# Safe path helpers
# -----------------------------------------------------------------------------

def _safe_join(*parts: Path | str) -> Path:
    """
    Join parts under ARTIFACTS_DIR and ensure the result remains within it.
    Prevents path traversal with user-supplied segments.
    """
    base = ARTIFACTS_DIR
    candidate = (base.joinpath(*[str(p) for p in parts])).resolve()
    if not str(candidate).startswith(str(base)):
        raise ArtifactSecurityError("Path traversal detected outside of ARTIFACTS_DIR")
    return candidate


def ensure_artifacts_dir() -> Path:
    """
    Ensure ARTIFACTS_DIR exists (create if missing) and return its path.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


# -----------------------------------------------------------------------------
# Metadata model
# -----------------------------------------------------------------------------

@dataclass
class ArtifactInfo:
    name: str
    relpath: str
    abspath: str
    size_bytes: int
    mtime_epoch: float
    ctime_epoch: float
    ext: str
    mime: str
    sha256: Optional[str] = None
    etag: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _guess_mime(path: Path) -> str:
    mtype, _ = mimetypes.guess_type(str(path))
    if mtype:
        return mtype
    return _MIME_FALLBACKS.get(path.suffix.lower(), "application/octet-stream")


def _file_sha256(path: Path, max_bytes: Optional[int] = None) -> str:
    """
    Efficient SHA-256 computation with optional cap (for huge files).
    If max_bytes is provided, hash only first max_bytes (still deterministic).
    """
    h = hashlib.sha256()
    chunk = 1024 * 1024
    total = 0
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            total += len(b)
            if max_bytes is not None and total > max_bytes:
                # Only hash up to max_bytes deterministically
                over = total - max_bytes
                h.update(b[:-over] if over > 0 else b)
                break
            h.update(b)
    return h.hexdigest()


def _to_info(path: Path, *, compute_hash: bool = False, hash_limit_bytes: int = 5 * 1024 * 1024) -> ArtifactInfo:
    st = path.stat()
    rel = path.relative_to(ARTIFACTS_DIR)
    mime = _guess_mime(path)
    sha = _file_sha256(path, hash_limit_bytes) if compute_hash else None
    # Simple ETag strategy: size-hex + mtime-int + (optional sha prefix)
    etag_core = f'{st.st_size:x}-{int(st.st_mtime):x}'
    etag = f'{etag_core}-{sha[:12]}' if sha else etag_core
    return ArtifactInfo(
        name=path.name,
        relpath=str(rel).replace("\\", "/"),
        abspath=str(path),
        size_bytes=st.st_size,
        mtime_epoch=st.st_mtime,
        ctime_epoch=st.st_ctime,
        ext=path.suffix.lower().lstrip("."),
        mime=mime,
        sha256=sha,
        etag=etag,
    )


# -----------------------------------------------------------------------------
# Listing & filtering
# -----------------------------------------------------------------------------

DEFAULT_EXCLUDES = {".DS_Store", "Thumbs.db"}


def _depth_of(p: Path) -> int:
    try:
        return len(p.relative_to(ARTIFACTS_DIR).parts)
    except Exception:
        return 0


def _natural_key(s: str) -> Tuple:
    # Human-friendly sort keys (e.g., file2 < file10)
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s))


def iter_artifacts(
    *,
    include_ext: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
    glob_patterns: Optional[Iterable[str]] = None,
    max_depth: Optional[int] = None,
    compute_hash: bool = False,
    search: Optional[str] = None,
) -> Iterator[ArtifactInfo]:
    """
    Walk ARTIFACTS_DIR and yield ArtifactInfo entries filtered by:
      • include_ext: e.g., ["html","png","json"]
      • exclude_names: exact file names to skip
      • glob_patterns: Unix-style patterns relative to ARTIFACTS_DIR
      • max_depth: limit recursion depth (0 = top only)
      • compute_hash: compute sha256 + etag for each file (expensive)
      • search: substring (case-insensitive) to match against relpath
    """
    ensure_artifacts_dir()
    include_ext_set = {e.lower().lstrip(".") for e in include_ext or []}
    exclude_names_set = set(exclude_names or []) | DEFAULT_EXCLUDES
    patterns = list(glob_patterns or [])
    needle = (search or "").lower()

    for root, dirs, files in os.walk(ARTIFACTS_DIR):
        root_path = Path(root)
        if max_depth is not None and _depth_of(root_path) > max_depth:
            dirs[:] = []
            continue

        # Stable natural sort for deterministic listings
        files.sort(key=_natural_key)

        for name in files:
            if name in exclude_names_set:
                continue
            p = root_path / name
            if not p.is_file():
                continue
            info = _to_info(p, compute_hash=compute_hash)

            if include_ext_set and info.ext not in include_ext_set:
                continue

            if patterns:
                rel = info.relpath
                if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
                    continue

            if needle and needle not in info.relpath.lower():
                continue

            yield info


def list_artifacts(
    *,
    include_ext: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
    glob_patterns: Optional[Iterable[str]] = None,
    max_depth: Optional[int] = None,
    compute_hash: bool = False,
    search: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[ArtifactInfo]:
    """
    Materialize iter_artifacts with optional pagination.
    """
    items = list(
        iter_artifacts(
            include_ext=include_ext,
            exclude_names=exclude_names,
            glob_patterns=glob_patterns,
            max_depth=max_depth,
            compute_hash=compute_hash,
            search=search,
        )
    )
    if offset < 0:
        offset = 0
    if limit is None or limit <= 0:
        return items[offset:]
    return items[offset : offset + limit]


# -----------------------------------------------------------------------------
# Common lookups
# -----------------------------------------------------------------------------

def find_summary() -> Optional[Path]:
    """
    Return the path to the primary diagnostic summary JSON (if present).
    """
    candidate = _safe_join(DIAGNOSTICS_SUMMARY_FILE)
    return candidate if candidate.exists() else None


@lru_cache(maxsize=8)
def read_summary_json() -> Optional[Dict[str, Any]]:
    """
    Read and parse the diagnostic summary JSON if available (cached).
    """
    p = find_summary()
    if not p:
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_first(*relative_candidates: str) -> Optional[Path]:
    """
    Return the first existing artifact for provided relative paths.
    Example:
        find_first("umap.html", "plots/umap.html")
    """
    for rel in relative_candidates:
        p = _safe_join(rel)
        if p.exists():
            return p
    return None


def newest_by_ext(ext: str) -> Optional[Path]:
    """
    Find the most recently modified artifact with the given extension.
    """
    items = list_artifacts(include_ext=[ext])
    if not items:
        return None
    items.sort(key=lambda i: i.mtime_epoch, reverse=True)
    return Path(items[0].abspath)


# -----------------------------------------------------------------------------
# Paths for common dashboard panels
# -----------------------------------------------------------------------------

def artifact_umap_html() -> Optional[Path]:
    return find_first("umap.html", "plots/umap.html", "dashboard/umap.html")


def artifact_tsne_html() -> Optional[Path]:
    return find_first("tsne.html", "plots/tsne.html", "dashboard/tsne.html")


def artifact_gll_heatmap() -> Optional[Path]:
    return find_first("gll_heatmap.png", "plots/gll_heatmap.png", "dashboard/gll_heatmap.png")


def artifact_shap_overlay() -> Optional[Path]:
    return find_first("shap_overlay.png", "plots/shap_overlay.png", "dashboard/shap_overlay.png")


def artifact_calibration_plot() -> Optional[Path]:
    return find_first("calibration_plot.png", "plots/calibration.png", "dashboard/calibration_plot.png")


# -----------------------------------------------------------------------------
# Public API (discovery & resolution)
# -----------------------------------------------------------------------------

def artifacts_root() -> Path:
    return ensure_artifacts_dir()


def resolve_artifact(relpath: str, *, must_exist: bool = True) -> Path:
    """
    Resolve a relative artifact path safely within ARTIFACTS_DIR.
    Raises if the file does not exist (when must_exist=True).
    """
    p = _safe_join(relpath)
    if must_exist and (not p.exists() or not p.is_file()):
        raise ArtifactNotFound(f"Artifact not found: {relpath}")
    return p


def stat_artifact(relpath: str, *, compute_hash: bool = False) -> ArtifactInfo:
    """
    Return ArtifactInfo (with optional sha256/etag) for a relative artifact path.
    """
    p = resolve_artifact(relpath)
    return _to_info(p, compute_hash=compute_hash)


def list_for_dashboard() -> Dict[str, Optional[str]]:
    """
    Minimal manifest of commonly used dashboard artifacts (relative paths).
    """
    def rel_or_none(p: Optional[Path]) -> Optional[str]:
        if not p:
            return None
        return str(p.relative_to(ARTIFACTS_DIR)).replace("\\", "/")

    return {
        "summary_json": rel_or_none(find_summary()),
        "umap_html": rel_or_none(artifact_umap_html()),
        "tsne_html": rel_or_none(artifact_tsne_html()),
        "gll_heatmap_png": rel_or_none(artifact_gll_heatmap()),
        "shap_overlay_png": rel_or_none(artifact_shap_overlay()),
        "calibration_plot_png": rel_or_none(artifact_calibration_plot()),
    }


def list_all_html(max_depth: Optional[int] = None, **kwargs) -> List[ArtifactInfo]:
    return list_artifacts(include_ext=["html", "htm"], max_depth=max_depth, **kwargs)


def list_all_images(max_depth: Optional[int] = None, **kwargs) -> List[ArtifactInfo]:
    return list_artifacts(include_ext=["png", "jpg", "jpeg", "svg", "webp"], max_depth=max_depth, **kwargs)


def list_all_data(max_depth: Optional[int] = None, **kwargs) -> List[ArtifactInfo]:
    return list_artifacts(include_ext=["json", "csv", "parquet"], max_depth=max_depth, **kwargs)


# -----------------------------------------------------------------------------
# Safe reads (binary), with simple HTTP Range support
# -----------------------------------------------------------------------------

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def _parse_range(range_header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """
    Parse an HTTP Range header value like 'bytes=START-END' into a (start,end) inclusive tuple.
    Returns None if no/invalid range provided.
    """
    if not range_header:
        return None
    m = _RANGE_RE.match(range_header.strip())
    if not m:
        return None
    start_s, end_s = m.groups()
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        # suffix: last N bytes
        length = int(end_s)
        if length <= 0:
            return None
        start = max(0, size - length)
        end = size - 1
    elif end_s == "":
        start = int(start_s)
        if start >= size:
            raise ArtifactRangeNotSatisfiable("Range start beyond file size")
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s)
        if start > end or start >= size:
            raise ArtifactRangeNotSatisfiable("Invalid range")
        end = min(end, size - 1)
    return (start, end)


def read_bytes(
    relpath: str,
    *,
    range_header: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Read an artifact's bytes with optional HTTP Range support.
    Returns (data, headers) suitable for a web response layer.

    Headers include: Content-Type, Content-Length, Accept-Ranges, ETag, Last-Modified,
    and for ranged responses: Content-Range, Status (206).
    """
    p = resolve_artifact(relpath)
    st = p.stat()
    size = st.st_size
    info = _to_info(p, compute_hash=False)

    # Parse range
    byte_range = _parse_range(range_header, size)
    if byte_range:
        start, end = byte_range
        length = end - start + 1
    else:
        start, end = 0, size - 1
        length = size

    if length > max_bytes:
        raise ArtifactReadTooLarge(
            f"Refusing to read {length} bytes from '{relpath}' (limit {max_bytes} bytes)"
        )

    # Read the requested slice
    with p.open("rb") as f:
        f.seek(start)
        data = f.read(length)

    # Compose headers
    headers: Dict[str, Any] = {
        "Content-Type": info.mime,
        "Content-Length": str(len(data)),
        "Accept-Ranges": "bytes",
        "ETag": info.etag,
        "Last-Modified": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(info.mtime_epoch)),
    }
    if byte_range:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Status"] = 206  # consumers can map to HTTP 206 Partial Content
    else:
        headers["Status"] = 200

    return data, headers


# -----------------------------------------------------------------------------
# Manifest utilities
# -----------------------------------------------------------------------------

def generate_manifest(
    *,
    include_ext: Optional[Iterable[str]] = None,
    glob_patterns: Optional[Iterable[str]] = None,
    compute_hash: bool = False,
    max_depth: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Generate a lightweight manifest for GUI prefetching and caching.
    """
    items = list_artifacts(
        include_ext=include_ext,
        glob_patterns=glob_patterns,
        compute_hash=compute_hash,
        max_depth=max_depth,
        limit=limit,
        offset=offset,
    )
    return {
        "root": str(ARTIFACTS_DIR),
        "count": len(items),
        "items": [i.as_dict() for i in items],
        "generated_at": int(time.time()),
    }


def write_manifest(path: Optional[Path] = None, **kwargs) -> Path:
    """
    Write a manifest JSON to disk and return the path.
    """
    manifest = generate_manifest(**kwargs)
    target = path or _safe_join("manifest.json")
    with target.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return target


# -----------------------------------------------------------------------------
# CLI (optional): quick inspection
# -----------------------------------------------------------------------------

def _print(obj: Any) -> None:
    import sys as _sys, json as _json
    _sys.stdout.write(_json.dumps(obj, indent=2) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print({
            "usage": "python -m scr.server.artifacts <cmd>",
            "cmds": [
                "root", "manifest", "html", "images", "data", "summary",
                "stat <relpath>", "cat <relpath> [--range bytes=START-END]",
            ],
        })
        return 0

    cmd = args[0]
    if cmd == "root":
        _print({"artifacts_root": str(artifacts_root())}); return 0
    if cmd == "manifest":
        _print(generate_manifest()); return 0
    if cmd == "html":
        _print([a.as_dict() for a in list_all_html()]); return 0
    if cmd == "images":
        _print([a.as_dict() for a in list_all_images()]); return 0
    if cmd == "data":
        _print([a.as_dict() for a in list_all_data()]); return 0
    if cmd == "summary":
        _print(read_summary_json() or {}); return 0
    if cmd == "stat":
        if len(args) < 2:
            _print({"error": "stat requires <relpath>"}); return 2
        _print(stat_artifact(args[1], compute_hash=True).as_dict()); return 0
    if cmd == "cat":
        if len(args) < 2:
            _print({"error": "cat requires <relpath> [--range bytes=..]"}); return 2
        rel = args[1]
        range_hdr = None
        for a in args[2:]:
            if a.startswith("--range="):
                range_hdr = a.split("=", 1)[1]
        try:
            data, headers = read_bytes(rel, range_header=range_hdr)
        except Exception as e:
            _print({"error": str(e)}); return 1
        # Write headers then data to stdout (human-friendly preview only)
        _print({"headers": headers})
        try:
            # Print as utf-8 text if likely text; else note binary length.
            if headers.get("Content-Type", "").startswith(("text/", "application/json")):
                print(data.decode("utf-8", errors="replace"))
            else:
                print(f"<{len(data)} bytes binary>")
        except Exception:
            print(f"<{len(data)} bytes>")
        return 0

    _print({"error": f"unknown cmd: {cmd}"})
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```
