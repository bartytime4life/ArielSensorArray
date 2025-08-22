# src/spectramind/logging_utils.py
# ==============================================================================
# Lightweight helpers for mission‑grade JSON logging (no external deps).
# ==============================================================================

from __future__ import annotations
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(p: str | os.PathLike) -> Path:
    """Create directory p if missing and return Path object."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    """Write a dict to path as pretty JSON (UTF‑8)."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def git_commit_hash(fallback: str = "unknown") -> str:
    """Return current git commit hash if available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return fallback


def system_fingerprint() -> Dict[str, str]:
    """Collect basic environment details for audit."""
    return {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
    }