# src/server/main.py

# =============================================================================

# 🚀 SpectraMind V50 — FastAPI Server Entrypoint

# -----------------------------------------------------------------------------

# Philosophy

# • CLI-first, GUI-optional: this server does not compute diagnostics; it

# exposes CLI-produced artifacts and summaries to a thin web GUI.

# • Reproducible by construction: all heavy lifting is delegated to the

# Typer CLI (`spectramind …`). The API is a read/trigger façade.

# • Minimal, secure defaults: optional header-based authz, optional CORS,

# zero third-party services required (works air-gapped).

#

# What this provides

# • FastAPI app with:

# - /               → redirect to API docs (or GUI index if configured)

# - /health         → liveness

# - /ready          → readiness with simple filesystem checks

# - /version        → VERSION + hashes (if present)

# - /me             → echo current user from authz dependency

# - /settings       → sanitized server/runtime settings (no secrets)

# • Mounts diagnostics API and static /artifacts (HTML/PNG/JSON).

# • Optional lightweight security & performance middleware.

# • Uvicorn entrypoint (python -m src.server.main).

# -----------------------------------------------------------------------------

# Upgrades in this version

# • Static artifacts mount configurable via env (ARTIFACTS\_DIR).

# • Optional security headers middleware (X-Content-Type-Options, etc.).

# • Optional GZip compression middleware.

# • Optional TrustedHost filter.

# • Readiness probe that checks for VERSION, run-hash JSON, and artifacts dir.

# • Settings endpoint to aid the GUI in environment discovery (safe values).

# • Hooks for future global dependencies if we later want to wire authz guards.

# =============================================================================

from **future** import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.staticfiles import StaticFiles

# Local modules

# - Diagnostics API registers /api/diagnostics/\* and (optionally) uses /artifacts

from src.server.api.diagnostics import register\_routes as register\_diagnostics

# - Lightweight header-based auth (optional; AUTHZ\_MODE env)

from src.server.authz import (
get\_current\_user,
User,
\# Optional guards available to routers; not globally applied here by default.
require\_ip\_allowlist,
require\_readonly\_guard,
require\_rate\_limit,
)

# -----------------------------------------------------------------------------

# Configuration helpers

# -----------------------------------------------------------------------------

def \_as\_bool(s: Optional\[str], default: bool = False) -> bool:
"""
Interpret an environment string as boolean with common truthy values.
"""
if s is None:
return default
return s.strip().lower() in {"1", "true", "yes", "on"}

def \_split\_csv\_env(s: Optional\[str]) -> list\[str]:
"""
Split a comma-separated env string into a clean list of tokens.
"""
if not s:
return \[]
return \[tok.strip() for tok in s.split(",") if tok.strip()]

# -----------------------------------------------------------------------------

# Paths & runtime constants

# -----------------------------------------------------------------------------

# Project root discovery: default to repo root (…/… from this file).

PROJECT\_ROOT = Path(os.getenv("PROJECT\_ROOT", Path(**file**).resolve().parents\[2])).resolve()
VERSION\_FILE = Path(os.getenv("VERSION\_FILE", PROJECT\_ROOT / "VERSION")).resolve()
RUN\_HASH\_FILE = Path(os.getenv("RUN\_HASH\_FILE", PROJECT\_ROOT / "run\_hash\_summary\_v50.json")).resolve()

# Where HTML/PNG/JSON artifacts are written by the CLI (diagnostics, reports, etc.).

ARTIFACTS\_DIR = Path(os.getenv("ARTIFACTS\_DIR", PROJECT\_ROOT / "artifacts")).resolve()

APP\_TITLE = os.getenv("APP\_TITLE", "SpectraMind V50 — Server")
APP\_DESC = os.getenv(
"APP\_DESC",
"CLI-first orchestration server that exposes diagnostics artifacts and metadata.",
)
APP\_VERSION = (VERSION\_FILE.read\_text(encoding="utf-8").strip() if VERSION\_FILE.exists() else "0.0.0-dev")

# CORS knobs

CORS\_ALLOW\_ORIGINS = os.getenv("CORS\_ALLOW\_ORIGINS", "*")  # comma-separated or "*"
CORS\_ALLOW\_CREDENTIALS = \_as\_bool(os.getenv("CORS\_ALLOW\_CREDENTIALS", "0"))
CORS\_ALLOW\_METHODS = os.getenv("CORS\_ALLOW\_METHODS", "GET,POST,OPTIONS")
CORS\_ALLOW\_HEADERS = os.getenv("CORS\_ALLOW\_HEADERS", "Content-Type,Authorization,X-API-Key")

# Middleware knobs

ENABLE\_GZIP = \_as\_bool(os.getenv("ENABLE\_GZIP", "1"), default=True)
SECURITY\_HEADERS = \_as\_bool(os.getenv("SECURITY\_HEADERS", "1"), default=True)
TRUSTED\_HOSTS = \_split\_csv\_env(os.getenv("TRUSTED\_HOSTS", ""))  # e.g., "localhost,127.0.0.1,example.com"

# Docs mount paths

DOCS\_URL = os.getenv("DOCS\_URL", "/docs")
REDOC\_URL = os.getenv("REDOC\_URL", "/redoc")
OPENAPI\_URL = os.getenv("OPENAPI\_URL", "/openapi.json")

# GUI index override (optional): if set, "/" will redirect here instead of /docs

GUI\_INDEX\_URL = os.getenv("GUI\_INDEX\_URL")  # e.g., "/app" (if a SPA is mounted elsewhere)

# -----------------------------------------------------------------------------

# Security headers middleware

# -----------------------------------------------------------------------------

class \_SecurityHeadersMiddleware(BaseHTTPMiddleware):
"""
Adds a small set of conservative security headers suitable for local/offline use.
CSP is intentionally omitted here to avoid breaking the FastAPI docs UI; projects
can add a stricter CSP if serving a locked-down GUI.
"""
async def dispatch(self, request: Request, call\_next):
response = await call\_next(request)
\# MIME type sniffing protection
response.headers.setdefault("X-Content-Type-Options", "nosniff")
\# Clickjacking protection (sameorigin ok for docs; adjust for GUI if needed)
response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
\# Referrer policy to avoid leaking paths to external sites
response.headers.setdefault("Referrer-Policy", "no-referrer")
\# Minimal permissions policy
response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
return response

# -----------------------------------------------------------------------------

# App factory

# -----------------------------------------------------------------------------

def create\_app() -> FastAPI:
"""
Build the FastAPI application with optional CORS, gzip, security headers,
trusted host filter, static artifact mount, and diagnostics routes.
"""
app = FastAPI(
title=APP\_TITLE,
description=APP\_DESC,
version=APP\_VERSION,
docs\_url=DOCS\_URL,
redoc\_url=REDOC\_URL,
openapi\_url=OPENAPI\_URL,
)

```
# Optional TrustedHost filter (no effect if list empty)
if TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

# Optional GZip compression
if ENABLE_GZIP:
    # Default minimum size 500 bytes balances CPU vs. bandwidth for plots/JSON
    app.add_middleware(GZipMiddleware, minimum_size=int(os.getenv("GZIP_MIN_BYTES", "500")))

# Optional security headers
if SECURITY_HEADERS:
    app.add_middleware(_SecurityHeadersMiddleware)

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
    expose_headers=["Content-Disposition"],  # helpful for file downloads
)

# Mount static artifacts if directory exists; otherwise create a placeholder.
# This path is used by the GUI to browse diagnostics, HTML reports, PNGs, etc.
if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR), html=False), name="artifacts")
else:
    # Do not auto-create in case the path is DVC-managed; readiness will report its absence.
    pass

# Register feature routers / static mounts
# NOTE: We keep global dependencies off here to avoid breaking /health and docs.
# Routers that need guards can import and use:
#   require_ip_allowlist(), require_rate_limit(), require_readonly_guard()
# from src.server.authz within their own route definitions.
register_diagnostics(app)

# -------------------------------
# Basic + ops-friendly endpoints
# -------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """
    Redirect to a GUI index (if configured) or the interactive API docs.
    """
    target = GUI_INDEX_URL or (app.docs_url or "/docs")
    return RedirectResponse(url=target)

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Lightweight liveness probe; returns 200 if the server process is responsive.
    """
    return {"status": "ok", "version": APP_VERSION}

@app.get("/ready")
def ready() -> JSONResponse:
    """
    Readiness probe that checks for presence/readability of:
      • VERSION file (optional)
      • run_hash_summary_v50.json (optional)
      • artifacts directory (recommended for GUI browsing)
    This keeps the probe cheap and filesystem-only (air-gapped friendly).
    """
    checks: Dict[str, Any] = {}

    # VERSION
    checks["version_file_exists"] = VERSION_FILE.exists()
    if VERSION_FILE.exists():
        try:
            _ = VERSION_FILE.read_text(encoding="utf-8")
            checks["version_file_readable"] = True
        except Exception as e:
            checks["version_file_readable"] = False
            checks["version_file_error"] = repr(e)

    # RUN HASH
    checks["run_hash_exists"] = RUN_HASH_FILE.exists()
    if RUN_HASH_FILE.exists():
        try:
            with RUN_HASH_FILE.open("r", encoding="utf-8") as f:
                json.load(f)
            checks["run_hash_readable"] = True
        except Exception as e:
            checks["run_hash_readable"] = False
            checks["run_hash_error"] = repr(e)

    # ARTIFACTS
    checks["artifacts_dir"] = str(ARTIFACTS_DIR)
    checks["artifacts_dir_exists"] = ARTIFACTS_DIR.exists()
    checks["artifacts_dir_is_dir"] = ARTIFACTS_DIR.is_dir() if ARTIFACTS_DIR.exists() else False

    status_code = status.HTTP_200_OK if (checks.get("artifacts_dir_exists") and checks.get("artifacts_dir_is_dir")) or True else status.HTTP_503_SERVICE_UNAVAILABLE  # artifacts optional; always 200 unless you prefer strict
    return JSONResponse(content={"status": "ok", "checks": checks}, status_code=status_code)

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

@app.get("/settings")
def settings() -> Dict[str, Any]:
    """
    Expose sanitized runtime settings to the GUI (no secrets, no API keys).
    This helps the front-end decide what features to show, where artifacts live, etc.
    """
    # Important: never emit secrets (API keys, bearer tokens). Only structural flags/paths.
    # Anything security-sensitive (like the authz mode) is safe to disclose at a high level.
    cors_origins = (
        ["*"]
        if CORS_ALLOW_ORIGINS.strip() == "*"
        else [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
    )
    return {
        "app": {
            "title": APP_TITLE,
            "version": APP_VERSION,
            "docs_url": DOCS_URL,
            "redoc_url": REDOC_URL,
            "openapi_url": OPENAPI_URL,
            "project_root": str(PROJECT_ROOT),
        },
        "artifacts": {
            "dir": str(ARTIFACTS_DIR),
            "mounted": ARTIFACTS_DIR.exists(),
            "static_mount": "/artifacts" if ARTIFACTS_DIR.exists() else None,
        },
        "cors": {
            "allow_origins": cors_origins,
            "allow_credentials": CORS_ALLOW_CREDENTIALS,
            "allow_methods": [m.strip() for m in CORS_ALLOW_METHODS.split(",") if m.strip()],
            "allow_headers": [h.strip() for h in CORS_ALLOW_HEADERS.split(",") if h.strip()],
        },
        "security": {
            "security_headers": SECURITY_HEADERS,
            "trusted_hosts": TRUSTED_HOSTS or None,
            "gzip": ENABLE_GZIP,
            # Whether guards are available to routers (not applied globally here).
            "authz_guards_available": True,
        },
    }

return app
```

# The ASGI application object

app = create\_app()

# -----------------------------------------------------------------------------

# Uvicorn Entrypoint

# -----------------------------------------------------------------------------

if **name** == "**main**":
\# Example:
\#   uvicorn src.server.main\:app --reload --port 8000
import uvicorn

```
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))
reload = _as_bool(os.getenv("RELOAD"), default=False)

uvicorn.run(app, host=host, port=port, reload=reload)
```
