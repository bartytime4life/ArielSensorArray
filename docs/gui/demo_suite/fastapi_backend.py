#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — FastAPI Backend (Upgraded Thin Contract)
- POST /api/run       : run a spectramind command (validated, logged, reproducible)
- POST /api/artifacts : list artifacts by glob (with optional filter/sort)
- GET  /api/log       : tail the audit log
"""
import asyncio
import glob
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------------------------------------------------------
# FastAPI app metadata
# ----------------------------------------------------------------------
app = FastAPI(
    title="SpectraMind V50 Backend",
    version="0.2.0",
    description="Thin API contract wrapping CLI-first SpectraMind pipeline."
)

# ----------------------------------------------------------------------
# Paths and repo context
# ----------------------------------------------------------------------
REPO = Path.cwd()
LOG_PATH = REPO / "logs" / "v50_debug_log.md"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Request/Response models
# ----------------------------------------------------------------------
class RunRequest(BaseModel):
    args: List[str]                 # e.g. ["diagnose","dashboard","--outputs.dir","outputs/diag_vX"]
    cli: Optional[str] = "spectramind"
    cwd: Optional[str] = None       # working dir override
    timeout: Optional[int] = 3600   # max seconds (default 1 hr)


class RunResponse(BaseModel):
    returncode: int
    stdout: str
    stderr: str
    command: str
    cwd: str
    timestamp: str


class ArtifactsRequest(BaseModel):
    glob: str                       # e.g. "outputs/diag_vX/**/*.html"
    sort: Optional[bool] = True
    limit: Optional[int] = None


class ArtifactsResponse(BaseModel):
    files: List[str]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sanitize_args(args: List[str]) -> List[str]:
    """Reject dangerous shell control characters; allow Hydra-style args."""
    if any((";" in a) or ("&&" in a) or ("|" in a) for a in args):
        raise HTTPException(status_code=400, detail="Invalid characters in args")
    return args


def _write_audit_log(entry: str) -> None:
    """Append structured audit logs for reproducibility."""
    timestamp = datetime.utcnow().isoformat()
    env_info = f"py={sys.version.split()[0]} os={platform.system()} git={_get_git_hash()}"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] {entry} | {env_info}\n")


def _get_git_hash() -> str:
    """Get current git commit hash if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO)
        ).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------
@app.post("/api/run", response_model=RunResponse)
async def api_run(req: RunRequest):
    cli = req.cli or "spectramind"
    args = _sanitize_args(req.args)
    cmd = [cli] + args
    cwd = Path(req.cwd).resolve() if req.cwd else REPO

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=req.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            _write_audit_log(f"TIMEOUT: {cmd} in {cwd}")
            raise HTTPException(status_code=408, detail="Command timed out")

        result = RunResponse(
            returncode=proc.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            command=" ".join(shlex.quote(a) for a in cmd),
            cwd=str(cwd),
            timestamp=datetime.utcnow().isoformat(),
        )
        _write_audit_log(f"RUN: {result.command} rc={result.returncode}")
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{cli} not found on PATH")


@app.get("/api/log", response_model=str)
async def api_log_tail(n: int = 50000):
    """Return last N bytes from debug log."""
    if not LOG_PATH.exists():
        return ""
    with LOG_PATH.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n), os.SEEK_SET)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return text.split("\n", 1)[-1] if size > n else text


@app.post("/api/artifacts", response_model=ArtifactsResponse)
async def api_artifacts(req: ArtifactsRequest):
    """Return artifact file paths by glob, optionally sorted/limited."""
    files = [str(Path(p).resolve()) for p in glob.glob(req.glob, recursive=True)]
    if req.sort:
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if req.limit:
        files = files[: req.limit]
    _write_audit_log(f"ARTIFACTS: {req.glob} -> {len(files)} files")
    return ArtifactsResponse(files=files)
