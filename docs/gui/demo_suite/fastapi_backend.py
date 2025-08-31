#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — FastAPI Backend (Thin Contract)
- POST /api/run : run a spectramind command (validated)
- GET  /api/artifacts : list artifacts by glob
- GET  /api/log       : tail the audit log
"""
import asyncio
import glob
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SpectraMind V50 Backend", version="0.1.0")

REPO = Path.cwd()
LOG_PATH = REPO / "logs" / "v50_debug_log.md"

class RunRequest(BaseModel):
    args: List[str]  # e.g. ["diagnose","dashboard","--outputs.dir","outputs/diag_vX"]
    cli: Optional[str] = "spectramind"
    cwd: Optional[str] = None  # default: current working dir

class RunResponse(BaseModel):
    returncode: int
    stdout: str
    stderr: str

class ArtifactsRequest(BaseModel):
    glob: str  # e.g. "outputs/diag_vX/**/*.html"

class ArtifactsResponse(BaseModel):
    files: List[str]

def _sanitize_args(args: List[str]) -> List[str]:
    # Prevent dangerous shell constructs by ensuring we do not pass through a shell
    # and keep to a vetted set (we still allow Hydra flags and values).
    if any((";" in a) or ("&&" in a) or ("|" in a) for a in args):
        raise HTTPException(status_code=400, detail="Invalid characters in args")
    return args

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
        stdout, stderr = await proc.communicate()
        return RunResponse(
            returncode=proc.returncode,
            stdout=(stdout.decode("utf-8", errors="replace") if stdout else ""),
            stderr=(stderr.decode("utf-8", errors="replace") if stderr else ""),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="spectramind not found on PATH")

@app.get("/api/log", response_model=str)
async def api_log_tail(n: int = 50000):
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
    files = [str(Path(p).resolve()) for p in glob.glob(req.glob, recursive=True)]
    return ArtifactsResponse(files=files)
