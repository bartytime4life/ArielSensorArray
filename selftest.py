#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — selftest.py

Mission:
  Run a fast, deterministic self‑test to validate that the SpectraMind V50
  environment, configs, and CLI are correctly wired before running the full
  pipeline (CI/Kaggle/Local). Produces both console output and a JSON summary
  that CI can parse.

What this does:
  1) Environment checks (Python version, OS info).
  2) Dependency presence (import checks).
  3) File/dir sanity (configs, src, outputs/logs).
  4) Git state (commit, dirty flag).
  5) Compute a "run hash" (config hash + env snapshot).
  6) CLI checks (help, train/predict dry‑runs, diagnose smoke).
  7) Optional deep checks (torch CUDA, DVC presence).
  8) Emit JSON + Markdown summaries.

Usage:
  python selftest.py --fast
  python selftest.py --deep --json-out outputs/selftest_summary.json

Exit codes:
  0 = success, 1 = failure
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------
# Constants & Defaults
# ------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path("configs/config_v50.yaml")
DEFAULT_OUTDIR = Path("outputs")
DEFAULT_LOGSDIR = Path("logs")
DEFAULT_DIAGDIR = DEFAULT_OUTDIR / "diagnostics"
DEFAULT_JSON_SUMMARY = DEFAULT_OUTDIR / "selftest_summary.json"
DEFAULT_MD_SUMMARY = DEFAULT_OUTDIR / "selftest_summary.md"
RUN_HASH_FILE = Path(os.getenv("RUN_HASH_FILE", "run_hash_summary_v50.json"))

CLI_CANDIDATES = [
    # 1) poetry managed script
    "poetry run spectramind",
    # 2) direct path script (if installed)
    "spectramind",
    # 3) module execution
    f"{shlex.quote(sys.executable)} -m spectramind",
]

TIMEOUT_SHORT = 30
TIMEOUT_MED = 60
TIMEOUT_LONG = 120


# ------------------------------
# Data Classes
# ------------------------------
@dc.dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    took_s: float = 0.0


@dc.dataclass
class SelfTestSummary:
    status: str
    timestamp: str
    python: str
    platform: str
    project_root: str
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    config_path: str
    config_hash_sha256: Optional[str]
    cli_cmd_used: Optional[str]
    cuda_available: Optional[bool]
    dvc_installed: Optional[bool]
    checks: List[CheckResult]
    hints: List[str]

    def to_json(self) -> str:
        def enc(o: Any):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, CheckResult):
                return dc.asdict(o)
            return o

        return json.dumps(dc.asdict(self), indent=2, default=enc)


# ------------------------------
# Utils
# ------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: str, timeout: int = TIMEOUT_MED, cwd: Optional[Path] = None) -> Tuple[int, str, str, float]:
    """Run shell command and capture (rc, out, err, took)."""
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        took = time.time() - t0
        return 124, out, f"TIMEOUT after {timeout}s\n{err}", took
    took = time.time() - t0
    return proc.returncode, out, err, took


def which(cmd: str) -> Optional[str]:
    rc, out, _, _ = run(f"command -v {shlex.quote(cmd)}", TIMEOUT_SHORT)
    return out.strip() if rc == 0 and out.strip() else None


def try_import(mod: str) -> Tuple[bool, Optional[str]]:
    try:
        m = __import__(mod)
        version = getattr(m, "__version__", None)
        return True, version
    except Exception as e:
        return False, str(e)


def sha256_file(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_status() -> Tuple[Optional[str], Optional[bool]]:
    # commit hash
    rc, out, _, _ = run("git rev-parse HEAD", TIMEOUT_SHORT, cwd=ROOT)
    commit = out.strip() if rc == 0 and out.strip() else None
    # dirty flag
    rc, out, _, _ = run("git status --porcelain", TIMEOUT_SHORT, cwd=ROOT)
    dirty = None
    if rc == 0:
        dirty = bool(out.strip())
    return commit, dirty


def ensure_dirs(paths: List[Path]) -> List[CheckResult]:
    results = []
    for p in paths:
        t0 = time.time()
        try:
            p.mkdir(parents=True, exist_ok=True)
            ok = p.exists() and p.is_dir()
            results.append(CheckResult(name=f"mkdir:{p}", ok=ok, detail="ok" if ok else "failed", took_s=time.time() - t0))
        except Exception as e:
            results.append(CheckResult(name=f"mkdir:{p}", ok=False, detail=str(e), took_s=time.time() - t0))
    return results


def choose_cli() -> Optional[str]:
    # If POETRY is installed and pyproject exists, prefer poetry run path
    if which("poetry") and (ROOT / "pyproject.toml").exists():
        return CLI_CANDIDATES[0]
    # Otherwise try direct script
    if which("spectramind"):
        return CLI_CANDIDATES[1]
    # Fallback to python -m spectramind
    return CLI_CANDIDATES[2]


def md_table(checks: List[CheckResult]) -> str:
    rows = ["| Check | Status | Time (s) | Detail |", "|------|--------|----------:|--------|"]
    for c in checks:
        status = "✅ OK" if c.ok else "❌ FAIL"
        rows.append(f"| `{c.name}` | {status} | {c.took_s:.2f} | {c.detail.replace('|','\\|')} |")
    return "\n".join(rows)


# ------------------------------
# Self-test steps
# ------------------------------
def check_python() -> CheckResult:
    t0 = time.time()
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    detail = f"Python {platform.python_version()} (>=3.10 required)"
    return CheckResult("python-version", ok, detail, time.time() - t0)


def check_imports() -> List[CheckResult]:
    wanted = [
        "yaml",            # PyYAML (Hydra deps)
        "omegaconf",       # Hydra core dep
        "hydra",           # Hydra entry
        "typer",           # CLI framework
        "numpy",
        "pandas",
    ]
    optional = [
        "torch",
        "networkx",
        "sklearn",
        "matplotlib",
        "dvc",
    ]
    results = []
    for m in wanted:
        t0 = time.time()
        ok, ver = try_import(m)
        results.append(CheckResult(f"import:{m}", ok, f"version={ver}" if ok else f"missing({ver})", time.time() - t0))
    for m in optional:
        t0 = time.time()
        ok, ver = try_import(m)
        results.append(CheckResult(f"import:{m}", ok, f"version={ver}" if ok else f"missing({ver})", time.time() - t0))
    return results


def check_paths(config_path: Path) -> List[CheckResult]:
    t0 = time.time()
    results = []
    results += ensure_dirs([DEFAULT_OUTDIR, DEFAULT_LOGSDIR, DEFAULT_DIAGDIR])
    results.append(CheckResult("path:src", (ROOT / "src").exists(), "src/ exists" if (ROOT / "src").exists() else "missing src/", time.time() - t0))
    results.append(CheckResult("path:configs", (ROOT / "configs").exists(), "configs/ exists" if (ROOT / "configs").exists() else "missing configs/", 0.0))
    if config_path:
        exists = config_path.exists()
        chash = sha256_file(config_path) if exists else None
        results.append(CheckResult("config:file", exists, f"{config_path} (sha256={chash})" if exists else "missing config", 0.0))
    return results


def check_git() -> List[CheckResult]:
    t0 = time.time()
    commit, dirty = get_git_status()
    res = []
    res.append(CheckResult("git:commit", commit is not None, f"{commit or 'n/a'}", time.time() - t0))
    res.append(CheckResult("git:dirty", dirty is not None, "dirty" if dirty else "clean" if dirty is not None else "n/a", 0.0))
    return res


def check_cli(cli_cmd: str, fast: bool = True) -> List[CheckResult]:
    checks: List[CheckResult] = []

    # --help
    t0 = time.time()
    rc, out, err, took = run(f"{cli_cmd} --help", TIMEOUT_SHORT)
    checks.append(CheckResult("cli:help", rc == 0, "ok" if rc == 0 else err[:2000], time.time() - t0))

    # --version (if supported)
    t0 = time.time()
    rc, out, err, took = run(f"{cli_cmd} --version", TIMEOUT_SHORT)
    ok = rc == 0 or ("Usage" in out + err)  # tolerate if command doesn't support --version
    checks.append(CheckResult("cli:version", ok, (out or err).strip()[:2000], time.time() - t0))

    # train (dry-run or fast)
    t0 = time.time()
    train_cmd = f"{cli_cmd} train --dry-run" if fast else f"{cli_cmd} train +training.epochs=1"
    rc, out, err, _ = run(train_cmd, TIMEOUT_MED)
    checks.append(CheckResult("cli:train", rc == 0, "ok" if rc == 0 else (out + "\n" + err)[:2000], time.time() - t0))

    # predict (dry-run)
    t0 = time.time()
    rc, out, err, _ = run(f"{cli_cmd} predict --dry-run", TIMEOUT_MED)
    checks.append(CheckResult("cli:predict", rc == 0, "ok" if rc == 0 else (out + "\n" + err)[:2000], time.time() - t0))

    # diagnose (smoke)
    t0 = time.time()
    rc, out, err, _ = run(f"{cli_cmd} diagnose dashboard --no-umap --no-tsne --outdir {shlex.quote(str(DEFAULT_DIAGDIR))}", TIMEOUT_LONG)
    checks.append(CheckResult("cli:diagnose", rc == 0, "ok" if rc == 0 else (out + "\n" + err)[:2000], time.time() - t0))

    return checks


def compute_run_hash(config_path: Path) -> str:
    """
    Create a simple run hash to help identify reproducible runs:
      sha256( config sha256 + python_version + platform + git_commit + pip_freeze )
    """
    parts: List[str] = []
    chash = sha256_file(config_path) or "no-config"
    parts.append(chash)
    parts.append(platform.python_version())
    parts.append(platform.platform())

    commit, _ = get_git_status()
    parts.append(commit or "no-git")

    # pip freeze (safely)
    rc, out, err, _ = run("python -m pip freeze", TIMEOUT_MED)
    parts.append(out.strip() if rc == 0 else "no-freeze")

    h = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return h


def check_cuda() -> CheckResult:
    t0 = time.time()
    ok, ver = try_import("torch")
    detail = "torch missing"
    if ok:
        try:
            import torch  # type: ignore
            detail = f"torch={torch.__version__}, cuda={torch.cuda.is_available()}"
            ok = True
        except Exception as e:
            detail = f"torch import error: {e}"
            ok = False
    return CheckResult("cuda:torch", ok, detail, time.time() - t0)


def check_dvc() -> CheckResult:
    t0 = time.time()
    rc, out, err, _ = run("dvc --version", TIMEOUT_SHORT)
    ok = rc == 0
    return CheckResult("dvc:version", ok, (out or err).strip(), time.time() - t0)


# ------------------------------
# Main
# ------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="SpectraMind V50 self-test (fast, deterministic sanity checks)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--fast", action="store_true", help="Run fast checks (default).")
    ap.add_argument("--deep", action="store_true", help="Run deeper checks (CUDA, DVC).")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to Hydra config file.")
    ap.add_argument("--json-out", type=str, default=str(DEFAULT_JSON_SUMMARY), help="Write JSON summary here.")
    ap.add_argument("--md-out", type=str, default=str(DEFAULT_MD_SUMMARY), help="Write Markdown summary here.")
    args = ap.parse_args(argv)

    # Prepare dirs
    DEFAULT_OUTDIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOGSDIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_DIAGDIR.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    checks: List[CheckResult] = []
    hints: List[str] = []

    # Environment checks
    checks.append(check_python())
    for cr in check_imports():
        checks.append(cr)

    # Paths
    cfg_path = Path(args.config)
    for cr in check_paths(cfg_path):
        checks.append(cr)

    # Git
    for cr in check_git():
        checks.append(cr)

    # Compute run hash
    run_hash = compute_run_hash(cfg_path)
    try:
        RUN_HASH_FILE.write_text(json.dumps({"timestamp": ts, "run_hash": run_hash}, indent=2))
    except Exception as e:
        hints.append(f"Could not write run hash file '{RUN_HASH_FILE}': {e}")

    # CLI check
    cli_cmd = choose_cli()
    if not cli_cmd:
        checks.append(CheckResult("cli:resolve", False, "Could not resolve CLI command (poetry/spectramind/python -m)", 0.0))
    else:
        checks.append(CheckResult("cli:resolve", True, f"Using '{cli_cmd}'", 0.0))
        for cr in check_cli(cli_cmd, fast=not args.deep):
            checks.append(cr)

    # Optional deeper checks
    cuda_ok = None
    dvc_ok = None
    if args.deep:
        cuda_res = check_cuda()
        checks.append(cuda_res)
        cuda_ok = "cuda=True" in cuda_res.detail
        dvc_res = check_dvc()
        checks.append(dvc_res)
        dvc_ok = dvc_res.ok

    # Determine status
    ok_all = all(c.ok for c in checks)
    status = "ok" if ok_all else "fail"

    # Build summary
    commit, dirty = get_git_status()
    cfg_hash = sha256_file(cfg_path) if cfg_path.exists() else None
    summary = SelfTestSummary(
        status=status,
        timestamp=ts,
        python=platform.python_version(),
        platform=platform.platform(),
        project_root=str(ROOT),
        git_commit=commit,
        git_dirty=dirty,
        config_path=str(cfg_path),
        config_hash_sha256=cfg_hash,
        cli_cmd_used=cli_cmd,
        cuda_available=cuda_ok,
        dvc_installed=dvc_ok,
        checks=checks,
        hints=hints,
    )

    # Emit JSON
    try:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(summary.to_json())
        log(f"[selftest] JSON summary → {args.json_out}")
    except Exception as e:
        log(f"[selftest] WARN: Failed to write json summary: {e}")

    # Emit Markdown
    md_lines = [
        f"# SpectraMind V50 — Self‑Test Summary",
        f"- **Timestamp:** {ts}",
        f"- **Status:** {'✅ OK' if status == 'ok' else '❌ FAIL'}",
        f"- **Python:** {summary.python}",
        f"- **Platform:** {summary.platform}",
        f"- **Project root:** `{summary.project_root}`",
        f"- **Git commit:** `{summary.git_commit}` {'(dirty)' if summary.git_dirty else '(clean)' if summary.git_dirty is not None else ''}",
        f"- **Config:** `{summary.config_path}`",
        f"- **Config sha256:** `{summary.config_hash_sha256}`",
        f"- **CLI:** `{summary.cli_cmd_used}`",
        "",
        "## Checks",
        md_table(checks),
    ]
    if hints:
        md_lines += ["", "## Hints"]
        md_lines += [f"- {h}" for h in hints]
    try:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text("\n".join(md_lines) + "\n")
        log(f"[selftest] Markdown summary → {args.md_out}")
    except Exception as e:
        log(f"[selftest] WARN: Failed to write markdown summary: {e}")

    # Print concise console recap
    log("\n=== SpectraMind V50 — Self‑Test (recap) ===")
    log(f"Status: {'OK' if status == 'ok' else 'FAIL'} | CLI: {summary.cli_cmd_used or 'n/a'} | Config: {cfg_path}")
    for c in checks:
        badge = "OK " if c.ok else "ERR"
        log(f"  [{badge}] {c.name} ({c.took_s:.2f}s) - {c.detail.splitlines()[0][:120]}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())