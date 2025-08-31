# src/server/api/diagnostics.py
# =============================================================================
# 🚀 FastAPI Diagnostics API for SpectraMind V50 (CLI-first, GUI-optional)
# -----------------------------------------------------------------------------
# Responsibilities
#   • Serve CLI-produced artifacts and summaries to the GUI.
#   • Never compute analytics directly — only orchestrate CLI and stream results.
#   • Provide safe, reproducible access to `diagnostic_summary.json`,
#     UMAP/t-SNE HTML, SHAP/FFT/Calibration plots, etc.
#
# Endpoints
#   GET  /api/diagnostics/health
#   GET  /api/diagnostics/summary
#   POST /api/diagnostics/run        # optional: execute CLI to refresh artifacts
#
# Static Mount
#   /artifacts → {ARTIFACTS_DIR} (default: ./artifacts)
#
# Environment / Config
#   • ARTIFACTS_DIR: directory containing CLI outputs (default: ./artifacts)
#   • DIAGNOSTICS_SUMMARY_FILE: file name for summary JSON (default: diagnostic_summary.json)
#   • SPECTRAMIND_CLI: CLI entrypoint (default: spectramind)
#   • DIAGNOSE_SUBCOMMAND: diagnose subcommand to run (default: diagnose dashboard)
#   • CLI_TIMEOUT_SECONDS: max seconds for CLI call (default: 1800)
#
# Notes
#   • This module exposes a `register_routes(app: FastAPI)` helper to:
#       - include the router
#       - mount StaticFiles at "/artifacts"
#   • The GUI (React) loads:
#       - GET /api/diagnostics/summary
#       - Static pages like /artifacts/umap.html, /artifacts/tsne.html, etc.
# =============================================================================

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _env(key: str, default: Optional[str] = None) -> str:
    v = os.getenv(key, default)
    assert v is not None, f"Missing required environment variable: {key}"
    return v

PROJECT_ROOT = pathlib.Path(os.getenv("PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[3]))
ARTIFACTS_DIR = pathlib.Path(_env("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))).resolve()
DIAGNOSTICS_SUMMARY_FILE = _env("DIAGNOSTICS_SUMMARY_FILE", "diagnostic_summary.json")
SPECTRAMIND_CLI = _env("SPECTRAMIND_CLI", "spectramind")
DIAGNOSE_SUBCOMMAND = _env("DIAGNOSE_SUBCOMMAND", "diagnose dashboard")
CLI_TIMEOUT_SECONDS = int(_env("CLI_TIMEOUT_SECONDS", "1800"))

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

def _read_json_file(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

def _safe_join(base: pathlib.Path, *paths: str) -> pathlib.Path:
    """
    Prevent path traversal by resolving against a base directory.
    """
    candidate = (base.joinpath(*paths)).resolve()
    if not str(candidate).startswith(str(base)):
        raise PermissionError("Path traversal detected.")
    return candidate

@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "artifacts_dir": str(ARTIFACTS_DIR)}

@router.get("/summary")
def get_summary() -> JSONResponse:
    """
    Return the CLI-produced diagnostic summary JSON.

    Expected layout:
      {ARTIFACTS_DIR}/diagnostic_summary.json
    """
    summary_path = _safe_join(ARTIFACTS_DIR, DIAGNOSTICS_SUMMARY_FILE)
    try:
        data = _read_json_file(summary_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return JSONResponse(content=data)

@router.post("/run")
def run_diagnostics(force: bool = False) -> Dict[str, Any]:
    """
    Optional: execute CLI to (re)generate dashboard artifacts.

    This endpoint is a thin wrapper around the CLI:
      spectramind diagnose dashboard

    It returns basic execution metadata and does not stream logs. The
    CLI itself writes full logs to v50_debug_log.md (or equivalent).
    """
    # Build command
    # Users can override DIAGNOSE_SUBCOMMAND via env (e.g., "diagnose dashboard --no-tsne").
    full_cmd = f"{SPECTRAMIND_CLI} {DIAGNOSE_SUBCOMMAND}"
    if force:
        full_cmd += " --force"

    # Split by shell rules safely
    cmd = shlex.split(full_cmd)

    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            status_code=504,
            detail=f"CLI timed out after {CLI_TIMEOUT_SECONDS}s: {e}",
        ) from e
    duration = time.time() - started

    result: Dict[str, Any] = {
        "command": cmd,
        "cwd": str(PROJECT_ROOT),
        "returncode": proc.returncode,
        "duration_sec": round(duration, 3),
        "stdout_tail": proc.stdout.splitlines()[-50:] if proc.stdout else [],
        "stderr_tail": proc.stderr.splitlines()[-50:] if proc.stderr else [],
    }

    if proc.returncode != 0:
        # Non-zero exit: surface stderr tail for quick triage
        raise HTTPException(
            status_code=500,
            detail={
                "msg": "Diagnostics CLI failed",
                "result": result,
            },
        )
    return result

# -----------------------------------------------------------------------------
# Registration helper
# -----------------------------------------------------------------------------

def register_routes(app: FastAPI) -> None:
    """
    Attach diagnostics router and mount static artifacts directory.
    Also apply a permissive CORS policy *only if* none configured.
    """
    # Include API routes
    app.include_router(router)

    # Mount artifacts static files at /artifacts
    #   e.g., /artifacts/umap.html, /artifacts/tsne.html, *.png, *.svg
    artifacts_dir = str(ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=artifacts_dir), name="artifacts")

    # Add permissive CORS if not already present
    if not any(isinstance(m, CORSMiddleware) for m in getattr(app.user_middleware, "_middlewares", [])):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

# -----------------------------------------------------------------------------
# Optional: Standalone app for quick testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal local runner: uvicorn src.server.api.diagnostics:app --reload
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="SpectraMind V50 — Diagnostics API")
    register_routes(app)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
