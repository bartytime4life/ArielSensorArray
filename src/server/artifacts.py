# scr/server/artifacts.py
# =============================================================================
# 📦 SpectraMind V50 — Artifacts Locator & Metadata Utilities
# -----------------------------------------------------------------------------
# This module provides safe helpers for locating, listing, and describing
# CLI-produced artifacts (HTML/PNG/CSV/JSON, etc.) for the GUI and API layers.
#
# It does **not** compute diagnostics. All artifacts are created by the CLI
# (e.g., `spectramind diagnose dashboard`) and written into ARTIFACTS_DIR.
#
# Key features:
#   • Safe path resolution (prevents traversal outside ARTIFACTS_DIR)
#   • Lightweight metadata (size, mtime) for directory listings
#   • Simple filters by extension or glob patterns
#   • Helpers to find common files (e.g., diagnostic_summary.json)
#
# Environment:
#   • PROJECT_ROOT (default: repo root resolved from this file)
#   • ARTIFACTS_DIR (default: {PROJECT_ROOT}/artifacts)
#   • DIAGNOSTICS_SUMMARY_FILE (default: diagnostic_summary.json)
# =============================================================================

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, Any

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).resolve()
DIAGNOSTICS_SUMMARY_FILE = os.getenv("DIAGNOSTICS_SUMMARY_FILE", "diagnostic_summary.json")

# -----------------------------------------------------------------------------
# Safe path helpers
# -----------------------------------------------------------------------------

def _safe_join(*parts: Path | str) -> Path:
    """
    Join parts under ARTIFACTS_DIR and ensure the result remains within it.
    Prevents path traversal when user-supplied segments are present.
    """
    base = ARTIFACTS_DIR
    candidate = (base.joinpath(*[str(p) for p in parts])).resolve()
    if not str(candidate).startswith(str(base)):
        raise PermissionError("Path traversal detected outside of ARTIFACTS_DIR")
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
    ext: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _to_info(path: Path) -> ArtifactInfo:
    st = path.stat()
    rel = path.relative_to(ARTIFACTS_DIR)
    return ArtifactInfo(
        name=path.name,
        relpath=str(rel).replace("\\", "/"),
        abspath=str(path),
        size_bytes=st.st_size,
        mtime_epoch=st.st_mtime,
        ext=path.suffix.lower().lstrip("."),
    )

# -----------------------------------------------------------------------------
# Listing & filtering
# -----------------------------------------------------------------------------

DEFAULT_EXCLUDES = {".DS_Store", "Thumbs.db"}

def iter_artifacts(
    *,
    include_ext: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
    glob_patterns: Optional[Iterable[str]] = None,
    max_depth: Optional[int] = None,
) -> Iterator[ArtifactInfo]:
    """
    Walk ARTIFACTS_DIR and yield ArtifactInfo entries filtered by:
      • include_ext: e.g., ["html","png","json"]
      • exclude_names: exact file names to skip
      • glob_patterns: Unix-style patterns relative to ARTIFACTS_DIR
      • max_depth: limit recursion depth (0 = top only)
    """
    ensure_artifacts_dir()
    include_ext_set = {e.lower().lstrip(".") for e in include_ext or []}
    exclude_names_set = set(exclude_names or []) | DEFAULT_EXCLUDES
    patterns = list(glob_patterns or [])

    def depth_of(p: Path) -> int:
        try:
            return len(p.relative_to(ARTIFACTS_DIR).parts)
        except Exception:
            return 0

    for root, dirs, files in os.walk(ARTIFACTS_DIR):
        root_path = Path(root)
        if max_depth is not None and depth_of(root_path) > max_depth:
            # prevent descending further
            dirs[:] = []
            continue

        for name in files:
            if name in exclude_names_set:
                continue
            p = root_path / name
            if not p.is_file():
                continue
            info = _to_info(p)

            if include_ext_set and info.ext not in include_ext_set:
                continue

            if patterns:
                rel = info.relpath
                if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
                    continue

            yield info

def list_artifacts(**kwargs) -> List[ArtifactInfo]:
    return list(iter_artifacts(**kwargs))

# -----------------------------------------------------------------------------
# Common lookups
# -----------------------------------------------------------------------------

def find_summary() -> Optional[Path]:
    """
    Return the path to the primary diagnostic summary JSON (if present).
    """
    candidate = _safe_join(DIAGNOSTICS_SUMMARY_FILE)
    return candidate if candidate.exists() else None

def read_summary_json() -> Optional[Dict[str, Any]]:
    """
    Read and parse the diagnostic summary JSON if available.
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
# Public API
# -----------------------------------------------------------------------------

def artifacts_root() -> Path:
    return ensure_artifacts_dir()

def resolve_artifact(relpath: str) -> Path:
    """
    Resolve a relative artifact path safely within ARTIFACTS_DIR.
    Raises if the file does not exist.
    """
    p = _safe_join(relpath)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Artifact not found: {relpath}")
    return p

def list_for_dashboard() -> Dict[str, Optional[str]]:
    """
    Return a minimal manifest of commonly used dashboard artifacts.
    Paths are returned relative to ARTIFACTS_DIR for easy static mounting.
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

def list_all_html(max_depth: Optional[int] = None) -> List[ArtifactInfo]:
    return list_artifacts(include_ext=["html"], max_depth=max_depth)

def list_all_images(max_depth: Optional[int] = None) -> List[ArtifactInfo]:
    return list_artifacts(include_ext=["png", "jpg", "jpeg", "svg", "webp"], max_depth=max_depth)

def list_all_data(max_depth: Optional[int] = None) -> List[ArtifactInfo]:
    return list_artifacts(include_ext=["json", "csv", "parquet"], max_depth=max_depth)

# -----------------------------------------------------------------------------
# CLI (optional): quick inspection
# -----------------------------------------------------------------------------

def _print(obj: Any) -> None:
    import sys, json as _json
    sys.stdout.write(_json.dumps(obj, indent=2) + "\n")

def main(argv: Optional[List[str]] = None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print({
            "usage": "python -m scr.server.artifacts <cmd>",
            "cmds": ["root", "manifest", "html", "images", "data", "summary"],
        })
        return 0

    cmd = args[0]
    if cmd == "root":
        _print({"artifacts_root": str(artifacts_root())}); return 0
    if cmd == "manifest":
        _print(list_for_dashboard()); return 0
    if cmd == "html":
        _print([a.as_dict() for a in list_all_html()]); return 0
    if cmd == "images":
        _print([a.as_dict() for a in list_all_images()]); return 0
    if cmd == "data":
        _print([a.as_dict() for a in list_all_data()]); return 0
    if cmd == "summary":
        _print(read_summary_json() or {}); return 0

    _print({"error": f"unknown cmd: {cmd}"})
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
