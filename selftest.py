#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — selftest.py

Mission:
  Run a fast, deterministic self‑test to validate that the SpectraMind V50
  environment, configs, and CLI are correctly wired before running the full
  pipeline (CI/Kaggle/Local). Produces both console output and a JSON summary
  that CI can parse.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

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
        return json.dumps(
            dc.asdict(self),
            indent=2,
            default=lambda o: str(o) if isinstance(o, Path) else o,
        )


# ------------------------------
# Utils
# ------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def safe_run(
    args: Sequence[str],
    timeout: int = TIMEOUT_MED,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> Tuple[int, str, str, float]:
    """Run a process (no shell), capture output and timing."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        took = time.time() - t0
        return proc.returncode, proc.stdout, proc.stderr, took
    except subprocess.TimeoutExpired as e:
        took = time.time() - t0
        return 124, e.stdout or "", f"TIMEOUT after {timeout}s\n{e.stderr or ''}", took
    except Exception as e:
        took = time.time() - t0
        return 1, "", f"{type(e).__name__}: {e}", took


def try_import(mod: str) -> Tuple[bool, Optional[str]]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", None)
        return True, ver
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
    commit = None
    dirty = None
    rc, out, _, _ = safe_run(["git", "rev-parse", "HEAD"], TIMEOUT_SHORT, ROOT)
    if rc == 0 and out.strip():
        commit = out.strip()
    rc, out, _, _ = safe_run(["git", "status", "--porcelain"], TIMEOUT_SHORT, ROOT)
    if rc == 0:
        dirty = bool(out.strip())
    return commit, dirty


def ensure_dirs(paths: List[Path]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for p in paths:
        t0 = time.time()
        ok = False
        detail = ""
        try:
            p.mkdir(parents=True, exist_ok=True)
            ok = p.exists() and p.is_dir()
            detail = "ok" if ok else "failed to create"
        except Exception as e:
            ok = False
            detail = str(e)
        results.append(CheckResult(name=f"mkdir:{p}", ok=ok, detail=detail, took_s=time.time() - t0))
    return results


def resolve_cli() -> Tuple[Optional[str], List[str]]:
    """
    Return (display, argv) for the CLI to use.
    Preference:
      1) poetry run spectramind (if poetry & pyproject present)
      2) python -m spectramind
      3) spectramind (if on PATH)
    """
    poetry = shutil.which("poetry")
    pyproject = (ROOT / "pyproject.toml").exists()
    if poetry and pyproject:
        return "poetry run spectramind", [poetry, "run", "spectramind"]

    # python -m spectramind (most reliable in dev envs)
    returncode, _, _, _ = safe_run([sys.executable, "-m", "spectramind", "--help"], TIMEOUT_SHORT)
    if returncode == 0:
        return f"{sys.executable} -m spectramind", [sys.executable, "-m", "spectramind"]

    # plain spectramind from PATH
    sp = shutil.which("spectramind")
    if sp:
        return sp, [sp]

    return None, []


def md_table(checks: List[CheckResult]) -> str:
    def esc(x: str) -> str:
        return x.replace("|", r"\|")
    lines = ["| Check | Status | Time (s) | Detail |", "|------|--------|---------:|--------|"]
    for c in checks:
        status = "✅ OK" if c.ok else "❌ FAIL"
        lines.append(f"| `{esc(c.name)}` | {status} | {c.took_s:.2f} | {esc(c.detail)} |")
    return "\n".join(lines)


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
        "yaml",        # PyYAML
        "omegaconf",   # Hydra dependency
        "hydra",       # Hydra entry
        "typer",       # CLI
        "numpy",
        "pandas",
    ]
    optional = ["torch", "networkx", "sklearn", "matplotlib", "dvc"]
    results: List[CheckResult] = []
    for m in wanted + optional:
        t0 = time.time()
        ok, ver = try_import(m)
        results.append(CheckResult(f"import:{m}", ok, f"version={ver}" if ok else f"missing({ver})", time.time() - t0))
    return results


def check_paths(config_path: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    results += ensure_dirs([DEFAULT_OUTDIR, DEFAULT_LOGSDIR, DEFAULT_DIAGDIR])
    for name in ("src", "configs"):
        p = ROOT / name
        results.append(CheckResult(f"path:{name}", p.exists(), f"{name}/ {'exists' if p.exists() else 'missing'}", 0.0))
    exists = config_path.exists()
    chash = sha256_file(config_path) if exists else None
    results.append(CheckResult("config:file", exists, f"{config_path} (sha256={chash})" if exists else "missing config", 0.0))
    return results


def check_cli(cli_argv: List[str], fast: bool = True) -> List[CheckResult]:
    checks: List[CheckResult] = []

    # --help
    t0 = time.time()
    rc, out, err, _ = safe_run([*cli_argv, "--help"], TIMEOUT_SHORT)
    checks.append(CheckResult("cli:help", rc == 0, "ok" if rc == 0 else (out or err)[:2000], time.time() - t0))

    # --version (tolerate missing)
    t0 = time.time()
    rc, out, err, _ = safe_run([*cli_argv, "--version"], TIMEOUT_SHORT)
    ok = rc == 0 or "Usage" in (out + err)
    checks.append(CheckResult("cli:version", ok, (out or err).strip()[:2000], time.time() - t0))

    # train (dry-run/fast)
    t0 = time.time()
    train_argv = [*cli_argv, "train", "--dry-run"] if fast else [*cli_argv, "train", "+training.epochs=1"]
    rc, out, err, _ = safe_run(train_argv, TIMEOUT_MED)
    checks.append(CheckResult("cli:train", rc == 0, "ok" if rc == 0 else (out + "\n" + err)[:2000], time.time() - t0))

    # predict (dry-run)
    t0 = time.time()
    rc, out, err, _ = safe_run([*cli_argv, "predict", "--dry-run"], TIMEOUT_MED)
    checks.append(CheckResult("cli:predict", rc == 0, "ok" if rc == 0 else (out + "\n" + err)[:2000], time.time() - t0))

    # diagnose (smoke)
    t0 = time.time()
    rc, out, err, _ = safe_run([*cli_argv, "diagnose", "dashboard", "--no-umap", "--no-tsne", "--outdir", str(DEFAULT_DIAGDIR)], TIMEOUT_LONG)
    checks.append(CheckResult("cli:diagnose", rc == 0, "ok" if rc == 0 else (out + "\n" + err)[:2000], time.time() - t0))

    return checks


def compute_run_hash(config_path: Path) -> str:
    """
    Create a simple run hash to help identify reproducible runs:
      sha256( config sha256 + python_version + platform + git_commit + pip_freeze )
    """
    parts: List[str] = []
    parts.append(sha256_file(config_path) or "no-config")
    parts.append(platform.python_version())
    parts.append(platform.platform())
    commit, _ = get_git_status()
    parts.append(commit or "no-git")
    rc, out, _, _ = safe_run([sys.executable, "-m", "pip", "freeze"], TIMEOUT_MED)
    parts.append(out.strip() if rc == 0 else "no-freeze")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def check_cuda() -> Tuple[CheckResult, Optional[bool]]:
    t0 = time.time()
    ok, ver = try_import("torch")
    detail = "torch missing"
    cuda_flag: Optional[bool] = None
    if ok:
        try:
            import torch  # type: ignore

            cuda_flag = torch.cuda.is_available()
            detail = f"torch={torch.__version__}, cuda={cuda_flag}"
            ok = True
        except Exception as e:
            ok = False
            detail = f"torch import error: {e}"
    return CheckResult("cuda:torch", ok, detail, time.time() - t0), cuda_flag


def check_dvc() -> Tuple[CheckResult, Optional[bool]]:
    t0 = time.time()
    rc, out, err, _ = safe_run(["dvc", "--version"], TIMEOUT_SHORT)
    ok = rc == 0
    return CheckResult("dvc:version", ok, (out or err).strip(), time.time() - t0), ok if ok else None


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
    for d in (DEFAULT_OUTDIR, DEFAULT_LOGSDIR, DEFAULT_DIAGDIR):
        d.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    checks: List[CheckResult] = []
    hints: List[str] = []

    # Env & imports
    checks.append(check_python())
    checks.extend(check_imports())

    # Paths
    cfg_path = Path(args.config)
    checks.extend(check_paths(cfg_path))

    # Git
    commit, dirty = get_git_status()
    checks.append(CheckResult("git:commit", commit is not None, f"{commit or 'n/a'}", 0.0))
    checks.append(CheckResult("git:dirty", dirty is not None, "dirty" if dirty else "clean" if dirty is not None else "n/a", 0.0))

    # Run hash
    run_hash = compute_run_hash(cfg_path)
    try:
        RUN_HASH_FILE.write_text(json.dumps({"timestamp": ts, "run_hash": run_hash}, indent=2))
    except Exception as e:
        hints.append(f"Could not write run hash file '{RUN_HASH_FILE}': {e}")

    # CLI check
    cli_display, cli_argv = resolve_cli()
    if not cli_argv:
        checks.append(CheckResult("cli:resolve", False, "Could not resolve CLI (poetry / python -m spectramind / spectramind).", 0.0))
    else:
        checks.append(CheckResult("cli:resolve", True, f"Using '{cli_display}'", 0.0))
        checks.extend(check_cli(cli_argv, fast=not args.deep))

    # Optional deeper checks
    cuda_ok: Optional[bool] = None
    dvc_ok: Optional[bool] = None
    if args.deep:
        cuda_res, cuda_ok = check_cuda()
        checks.append(cuda_res)
        dvc_res, dvc_ok = check_dvc()
        checks.append(dvc_res)

    # Determine status
    ok_all = all(c.ok for c in checks)
    status = "ok" if ok_all else "fail"

    # Build summary
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
        cli_cmd_used=cli_display,
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
        md_lines += ["", "## Hints", *[f"- {h}" for h in hints]]
    try:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text("\n".join(md_lines) + "\n")
        log(f"[selftest] Markdown summary → {args.md_out}")
    except Exception as e:
        log(f"[selftest] WARN: Failed to write markdown summary: {e}")

    # Console recap
    log("\n=== SpectraMind V50 — Self‑Test (recap) ===")
    log(f"Status: {'OK' if status == 'ok' else 'FAIL'} | CLI: {summary.cli_cmd_used or 'n/a'} | Config: {cfg_path}")
    for c in checks:
        badge = "OK " if c.ok else "ERR"
        head = (c.detail.splitlines()[0] if c.detail else "").strip()
        log(f"  [{badge}] {c.name} ({c.took_s:.2f}s) - {head[:120]}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())