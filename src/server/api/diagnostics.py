# src/server/api/diagnostics.py

# =============================================================================

# 🚀 FastAPI Diagnostics API for SpectraMind V50 (CLI-first, GUI-optional)

# -----------------------------------------------------------------------------

# Responsibilities

# • Serve CLI-produced artifacts and summaries to the GUI.

# • Never compute analytics directly — only orchestrate CLI and stream results.

# • Provide safe, reproducible access to `diagnostic_summary.json`,

# UMAP/t-SNE HTML, SHAP/FFT/Calibration plots, etc.

#

# Endpoints (this module)

# GET   /api/diagnostics/health

# GET   /api/diagnostics/summary                 # returns diagnostic\_summary.json

# GET   /api/diagnostics/list                    # list artifacts (filtered)

# GET   /api/diagnostics/logs                    # tail v50\_debug\_log.md (optional)

# POST  /api/diagnostics/run                     # execute CLI to refresh artifacts (optional)

#

# Static Mount (mounted here if not already present by the app)

# /artifacts  → {ARTIFACTS\_DIR} (default: ./artifacts)

#

# Environment / Config

# • PROJECT\_ROOT              : repo root (default: 3 parents up from this file)

# • ARTIFACTS\_DIR             : directory for CLI outputs (default: {PROJECT\_ROOT}/artifacts)

# • DIAGNOSTICS\_SUMMARY\_FILE  : summary JSON filename (default: diagnostic\_summary.json)

# • SPECTRAMIND\_CLI           : CLI entrypoint (default: spectramind)

# • DIAGNOSE\_SUBCOMMAND       : subcommand string (default: "diagnose dashboard")

# • CLI\_TIMEOUT\_SECONDS       : max seconds for CLI call (default: 1800)

# • DIAGNOSTICS\_ALLOW\_RUN     : "1" to enable POST /run (default: "1")

# • DIAGNOSTICS\_ALLOWED\_EXT   : comma list of extensions for /list (default: json,html,png,svg,jpg,jpeg,csv,md)

#

# Security / Auth (provided by src.server.authz)

# • Read endpoints may be guarded with scopes like "diagnostics\:read".

# • Run endpoint may require "diagnostics\:run" or "pipeline\:run".

# • Optional IP allowlist, read-only guard, and rate limit dependencies can be used.

# =============================================================================

from **future** import annotations

import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

# Optional guards (available; not mandatory)

from src.server.authz import (
require\_scopes,
require\_any\_scope,
require\_rate\_limit,
require\_readonly\_guard,
require\_ip\_allowlist,
User,
get\_current\_user,
)

# -----------------------------------------------------------------------------

# Configuration helpers

# -----------------------------------------------------------------------------

def \_env(key: str, default: Optional\[str] = None) -> str:
v = os.getenv(key, default)
assert v is not None, f"Missing required environment variable: {key}"
return v

def \_bool\_env(key: str, default: bool = False) -> bool:
raw = os.getenv(key)
if raw is None:
return default
return raw\.strip().lower() in {"1", "true", "yes", "on"}

def \_csv\_env(key: str, default: str) -> List\[str]:
raw = os.getenv(key, default)
return \[s.strip().lower() for s in raw\.split(",") if s.strip()]

# Paths

PROJECT\_ROOT = pathlib.Path(os.getenv("PROJECT\_ROOT", pathlib.Path(**file**).resolve().parents\[3])).resolve()
ARTIFACTS\_DIR = pathlib.Path(\_env("ARTIFACTS\_DIR", str(PROJECT\_ROOT / "artifacts"))).resolve()
DIAGNOSTICS\_SUMMARY\_FILE = \_env("DIAGNOSTICS\_SUMMARY\_FILE", "diagnostic\_summary.json")
SPECTRAMIND\_CLI = \_env("SPECTRAMIND\_CLI", "spectramind")
DIAGNOSE\_SUBCOMMAND = \_env("DIAGNOSE\_SUBCOMMAND", "diagnose dashboard")
CLI\_TIMEOUT\_SECONDS = int(\_env("CLI\_TIMEOUT\_SECONDS", "1800"))
DIAGNOSTICS\_ALLOW\_RUN = \_bool\_env("DIAGNOSTICS\_ALLOW\_RUN", default=True)
DIAGNOSTICS\_ALLOWED\_EXT = set("." + ext for ext in \_csv\_env("DIAGNOSTICS\_ALLOWED\_EXT", "json,html,png,svg,jpg,jpeg,csv,md"))

# -----------------------------------------------------------------------------

# Router

# -----------------------------------------------------------------------------

router = APIRouter(prefix="/api/diagnostics", tags=\["diagnostics"])

# -----------------------------------------------------------------------------

# Filesystem helpers

# -----------------------------------------------------------------------------

def \_safe\_join(base: pathlib.Path, \*paths: str) -> pathlib.Path:
"""
Prevent path traversal by resolving against a base directory.
"""
candidate = (base.joinpath(\*paths)).resolve()
if not str(candidate).startswith(str(base)):
raise PermissionError("Path traversal detected.")
return candidate

def \_read\_json\_file(path: pathlib.Path) -> Dict\[str, Any]:
if not path.exists() or not path.is\_file():
raise FileNotFoundError(f"File not found: {path}")
try:
with path.open("r", encoding="utf-8") as f:
return json.load(f)
except json.JSONDecodeError as e:
raise ValueError(f"Invalid JSON in {path}: {e}") from e

def \_hash\_file\_sha256(path: pathlib.Path, chunk\_size: int = 1 << 20) -> str:
h = hashlib.sha256()
with path.open("rb") as f:
while True:
chunk = f.read(chunk\_size)
if not chunk:
break
h.update(chunk)
return h.hexdigest()

def \_list\_artifacts(base: pathlib.Path, allowed\_ext: set\[str]) -> List\[Dict\[str, Any]]:
results: List\[Dict\[str, Any]] = \[]
if not base.exists():
return results
for p in base.rglob("\*"):
if not p.is\_file():
continue
ext = p.suffix.lower()
if ext and ext in allowed\_ext:
try:
st = p.stat()
results.append(
{
"path": str(p.relative\_to(base)).replace("\\", "/"),
"size": st.st\_size,
"mtime": int(st.st\_mtime),
"ext": ext.lstrip("."),
}
)
except Exception:
continue
\# Newest first
results.sort(key=lambda d: d\["mtime"], reverse=True)
return results

# -----------------------------------------------------------------------------

# Endpoints

# -----------------------------------------------------------------------------

@router.get("/health")
def health() -> Dict\[str, Any]:
return {
"status": "ok",
"artifacts\_dir": str(ARTIFACTS\_DIR),
"artifacts\_exists": ARTIFACTS\_DIR.exists(),
}

@router.get(
"/summary",
dependencies=\[
Depends(require\_ip\_allowlist()),
Depends(require\_rate\_limit()),
\# Require either explicit diagnostics read scope or wildcard read
Depends(require\_any\_scope("diagnostics\:read", "read:*", "*")),
],
)
def get\_summary(
request: Request,
response: Response,
filename: str = Query(DIAGNOSTICS\_SUMMARY\_FILE, description="Summary filename inside artifacts dir"),
if\_none\_match: Optional\[str] = Header(default=None, alias="If-None-Match"),
) -> JSONResponse:
"""
Return the CLI-produced diagnostic summary JSON.

```
Expected layout:
  {ARTIFACTS_DIR}/{filename}   (defaults to diagnostic_summary.json)

Conditional caching:
  • Computes ETag = sha256(file).
  • If client sends If-None-Match and it matches, returns 304.
"""
summary_path = _safe_join(ARTIFACTS_DIR, filename)
try:
    if not summary_path.exists() or not summary_path.is_file():
        raise FileNotFoundError(f"File not found: {summary_path}")
    etag = _hash_file_sha256(summary_path)
    # ETag handling
    if if_none_match and if_none_match.strip('"') == etag:
        return JSONResponse(status_code=status.HTTP_304_NOT_MODIFIED, content=None)
    data = _read_json_file(summary_path)
    # Set caching headers
    response.headers["ETag"] = f"\"{etag}\""
    response.headers["Cache-Control"] = "no-cache"
except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e)) from e
except ValueError as e:
    raise HTTPException(status_code=422, detail=str(e)) from e
return JSONResponse(content=data)
```

@router.get(
"/list",
dependencies=\[
Depends(require\_ip\_allowlist()),
Depends(require\_rate\_limit()),
Depends(require\_any\_scope("diagnostics\:read", "read:*", "*")),
],
)
def list\_artifacts(
subdir: Optional\[str] = Query(None, description="Optional subdirectory within artifacts"),
limit: int = Query(200, ge=1, le=5000, description="Max files to return"),
exts: Optional\[str] = Query(None, description="Comma list of allowed extensions (overrides default)"),
) -> Dict\[str, Any]:
"""
List recent artifact files (filtered by extension), newest first.
"""
base = ARTIFACTS\_DIR if not subdir else \_safe\_join(ARTIFACTS\_DIR, subdir)
allowed = DIAGNOSTICS\_ALLOWED\_EXT if not exts else set("." + e.strip().lower() for e in exts.split(",") if e.strip())
items = \_list\_artifacts(base, allowed)
return {
"dir": str(base),
"count": min(limit, len(items)),
"total": len(items),
"items": items\[:limit],
}

@router.get(
"/logs",
dependencies=\[
Depends(require\_ip\_allowlist()),
Depends(require\_rate\_limit()),
Depends(require\_any\_scope("diagnostics\:read", "read:*", "*")),
],
)
def tail\_logs(
path: str = Query("v50\_debug\_log.md", description="Path relative to artifacts dir"),
n: int = Query(200, ge=1, le=2000, description="Number of lines from end"),
) -> Dict\[str, Any]:
"""
Return the last N lines of a log-like text file (UTF-8), relative to ARTIFACTS\_DIR.
Useful for showing recent CLI calls and diagnostics summaries in the GUI.
"""
log\_path = \_safe\_join(ARTIFACTS\_DIR, path)
if not log\_path.exists() or not log\_path.is\_file():
raise HTTPException(status\_code=404, detail=f"File not found: {log\_path}")
try:
with log\_path.open("r", encoding="utf-8", errors="replace") as f:
lines = f.read().splitlines()
tail = lines\[-n:] if len(lines) > n else lines
return {
"file": str(log\_path),
"lines": tail,
"total\_lines": len(lines),
"returned": len(tail),
}
except Exception as e:
raise HTTPException(status\_code=500, detail=f"Failed to read log: {e}") from e

@router.post(
"/run",
dependencies=\[
Depends(require\_ip\_allowlist()),
Depends(require\_readonly\_guard()),
Depends(require\_rate\_limit()),
\# Require a run scope; allow pipeline\:run or diagnostics\:run or wildcard
Depends(require\_any\_scope("diagnostics\:run", "pipeline\:run", "write:*", "*")),
],
)
def run\_diagnostics(
force: bool = Query(False, description="Force regeneration if supported by CLI"),
extra\_args: Optional\[str] = Query(None, description="Extra CLI args appended to the diagnose subcommand"),
) -> Dict\[str, Any]:
"""
Execute CLI to (re)generate dashboard artifacts.

```
This endpoint is a thin wrapper around the CLI:
  spectramind diagnose dashboard [extra_args]

Returns:
  • command, cwd, returncode, duration_sec
  • stdout_tail, stderr_tail (last 50 lines)
"""
if not DIAGNOSTICS_ALLOW_RUN:
    raise HTTPException(status_code=403, detail="Diagnostics run is disabled by server configuration")

# Build command safely
# Only allow diagnose-related subcommands to avoid arbitrary execution.
base_cmd = f"{SPECTRAMIND_CLI} {DIAGNOSE_SUBCOMMAND}"
if force:
    base_cmd += " --force"
if extra_args:
    # Safe-ish: split and append; deny ';', '&&', etc. by using shlex.split and no shell=True.
    base_cmd += f" {extra_args}"

cmd = shlex.split(base_cmd)

# Prevent abuse: ensure the command starts with "spectramind" and includes "diagnose"
if not cmd or os.path.basename(cmd[0]) != SPECTRAMIND_CLI or "diagnose" not in cmd:
    raise HTTPException(status_code=400, detail={"msg": "Invalid subcommand; only 'spectramind diagnose …' is allowed"})

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
    "artifacts_dir": str(ARTIFACTS_DIR),
    "stdout_tail": proc.stdout.splitlines()[-50:] if proc.stdout else [],
    "stderr_tail": proc.stderr.splitlines()[-50:] if proc.stderr else [],
}

if proc.returncode != 0:
    raise HTTPException(
        status_code=500,
        detail={"msg": "Diagnostics CLI failed", "result": result},
    )
return result
```

# -----------------------------------------------------------------------------

# Registration helper

# -----------------------------------------------------------------------------

def \_is\_artifacts\_mounted(app: FastAPI) -> bool:
mount\_paths = {getattr(r, "path", "") for r in getattr(app, "routes", \[])}
return "/artifacts" in mount\_paths

def register\_routes(app: FastAPI) -> None:
"""
Attach diagnostics router and mount static artifacts directory.
Also apply a permissive CORS policy *only if* none configured.
"""
\# Include API routes
app.include\_router(router)

```
# Mount artifacts static files at /artifacts (if not already mounted)
if ARTIFACTS_DIR.exists() and not _is_artifacts_mounted(app):
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR), html=False), name="artifacts")

# Add permissive CORS if not already present (best-effort check)
if not any(isinstance(m, CORSMiddleware) for m in getattr(app, "user_middleware", [])):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
```

# -----------------------------------------------------------------------------

# Optional: Standalone app for quick testing

# -----------------------------------------------------------------------------

if **name** == "**main**":
\# Minimal local runner: uvicorn src.server.api.diagnostics\:app --reload
from fastapi import FastAPI
import uvicorn

```
app = FastAPI(title="SpectraMind V50 — Diagnostics API")
register_routes(app)
port = int(os.getenv("PORT", "8000"))
uvicorn.run(app, host="0.0.0.0", port=port, reload=_bool_env("RELOAD", False))
```
