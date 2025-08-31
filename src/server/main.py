# src/server/main.py
# =============================================================================
# 🚀 SpectraMind V50 — FastAPI Server Entrypoint
# -----------------------------------------------------------------------------
# Philosophy
#   • CLI-first, GUI-optional: this server does not compute diagnostics; it
#     exposes CLI-produced artifacts and summaries to a thin web GUI.
#   • Reproducible by construction: all heavy lifting is delegated to the
#     Typer CLI (`spectramind …`). The API is a read/trigger façade.
#   • Minimal, secure defaults: optional header-based authz, optional CORS,
#     zero third-party services required (works air-gapped).
#
# What this provides
#   • FastAPI app with:
#       - /            → basic info
#       - /health      → liveness
#       - /version     → VERSION + hashes (if present)
#       - /me          → echo current user from authz dependency
#   • Mounts diagnostics API and static /artifacts (HTML/PNG/JSON).
#   • Uvicorn entrypoint (python -m src.server.main).
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Local modules
#   - Diagnostics API registers /api/diagnostics/* and mounts /artifacts
from src.server.api.diagnostics import register_routes as register_diagnostics
#   - Lightweight header-based auth (optional; AUTHZ_MODE env)
from src.server.authz import get_current_user, User

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
VERSION_FILE = Path(os.getenv("VERSION_FILE", PROJECT_ROOT / "VERSION")).resolve()
RUN_HASH_FILE = Path(os.getenv("RUN_HASH_FILE", PROJECT_ROOT / "run_hash_summary_v50.json")).resolve()

APP_TITLE = os.getenv("APP_TITLE", "SpectraMind V50 — Server")
APP_DESC = os.getenv(
    "APP_DESC",
    "CLI-first orchestration server that exposes diagnostics artifacts and metadata.",
)
APP_VERSION = (VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "0.0.0-dev")

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")  # comma-separated or "*"
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "0") in {"1", "true", "TRUE"}
CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "GET,POST,OPTIONS")
CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "Content-Type,Authorization,X-API-Key")

# -----------------------------------------------------------------------------
# App factory
# -----------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESC,
        version=APP_VERSION,
        docs_url=os.getenv("DOCS_URL", "/docs"),
        redoc_url=os.getenv("REDOC_URL", "/redoc"),
        openapi_url=os.getenv("OPENAPI_URL", "/openapi.json"),
    )

    # CORS (optional)
    allow_origins = (
        ["*"]
        if CORS_ALLOW_ORIGINS.strip() == "*"
        else [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=[m.strip() for m in CORS_ALLOW_METHODS.split(",") if m.strip()],
        allow_headers=[h.strip() for h in CORS_ALLOW_HEADERS.split(",") if h.strip()],
    )

    # Register feature routers / static mounts
    register_diagnostics(app)

    # Basic routes
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        # Redirect to docs for convenience, can change to a GUI index if desired
        return RedirectResponse(url=app.docs_url or "/docs")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": APP_VERSION}

    @app.get("/version")
    def version() -> Dict[str, Any]:
        """
        Return app version and any run-hash metadata if available.
        """
        payload: Dict[str, Any] = {
            "version": APP_VERSION,
            "project_root": str(PROJECT_ROOT),
        }
        if RUN_HASH_FILE.exists():
            try:
                with RUN_HASH_FILE.open("r", encoding="utf-8") as f:
                    payload["run_hash"] = json.load(f)
            except Exception:
                payload["run_hash"] = {"error": "failed to read run hash file"}
        # Simple integrity hash of VERSION file (if present)
        if VERSION_FILE.exists():
            try:
                b = VERSION_FILE.read_bytes()
                payload["version_sha256"] = hashlib.sha256(b).hexdigest()
            except Exception:
                payload["version_sha256"] = None
        return payload

    @app.get("/me")
    def me(user: User = Depends(get_current_user)) -> JSONResponse:
        """
        Echo current user from the authorization dependency.
        AUTHZ_MODE:
          • OFF             → returns synthetic 'dev' admin user
          • HEADER_API_KEY  → returns policy user for supplied key/bearer token
        """
        return JSONResponse(
            content={
                "id": user.id,
                "name": user.name,
                "roles": user.roles,
                "scopes": user.scopes,
            }
        )

    return app


app = create_app()

# -----------------------------------------------------------------------------
# Uvicorn Entrypoint
# -----------------------------------------------------------------------------

def _as_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return s in {"1", "true", "TRUE", "yes", "YES"}

if __name__ == "__main__":
    # Example:
    #   uvicorn src.server.main:app --reload --port 8000
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = _as_bool(os.getenv("RELOAD"), default=False)

    uvicorn.run(app, host=host, port=port, reload=reload)
