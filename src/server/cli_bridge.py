# src/server/cli_bridge.py
# =============================================================================
# 🔗 SpectraMind V50 — CLI Bridge (Server-Safe Wrapper for `spectramind …`)
# -----------------------------------------------------------------------------
# Why this exists
#   • The GUI is a thin veneer over the CLI. This module provides a *safe*,
#     reproducible way for the server to invoke CLI subcommands and return
#     structured results to the frontend — without doing any analytics here.
#
# Design goals
#   • CLI-first & reproducible: we never compute or mutate artifacts directly.
#     All work is delegated to the Typer CLI (`spectramind …`) which logs and
#     writes artifacts under versioned, Hydra-managed outputs.
#   • Security-minded execution: no shell=True; strict argv lists; redaction of
#     sensitive env/flags; path traversal protection is done at API layer.
#   • Developer ergonomics: a single `run_cli()` with convenience wrappers for
#     common subcommands (calibrate/train/diagnose/submit).
#
# What this module provides
#   • run_cli(argv: list[str], *, timeout, cwd, env_overrides, stream) -> dict|gen
#   • spectramind_run(subcommand_str, extra_args) -> dict
#   • helpers: calibrate(), train(), diagnose_dashboard(), submit_bundle()
#
# NOTE: This module does NOT expose FastAPI routes; import it from your API
#       modules (e.g., src/server/api/diagnostics.py) to execute the CLI.
# =============================================================================

from __future__ import annotations

import os
import sys
import shlex
import json
import time
import queue
import signal
import threading
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Union

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_CLI = os.getenv("SPECTRAMIND_CLI", "spectramind")
DEFAULT_TIMEOUT = int(os.getenv("CLI_TIMEOUT_SECONDS", "1800"))  # 30 min default
DEFAULT_LOG_TAIL = int(os.getenv("CLI_LOG_TAIL_LINES", "200"))
DEFAULT_ENV_REDACT_KEYS = {
    "HF_TOKEN",
    "WANDB_API_KEY",
    "MLFLOW_TRACKING_PASSWORD",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}

# Optional canonical debug log file (written by the CLI)
DEFAULT_DEBUG_LOG = os.getenv("V50_DEBUG_LOG", str(PROJECT_ROOT / "v50_debug_log.md"))

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_list(argv: Union[str, Iterable[str]]) -> List[str]:
    """
    Accept either a string (split with shlex) or an iterable of strings
    and return a safe argv list.
    """
    if isinstance(argv, str):
        return shlex.split(argv)
    return list(argv)

def _redact_env(env: Mapping[str, str], redact_keys: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Return a copy of env with sensitive values redacted for logging."""
    redact = set(DEFAULT_ENV_REDACT_KEYS if redact_keys is None else redact_keys)
    safe = {}
    for k, v in env.items():
        if k in redact:
            safe[k] = "********"
        else:
            safe[k] = v
    return safe

def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _tail_lines(text: str, n: int) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    return lines[-max(0, n):]

# -----------------------------------------------------------------------------
# Streaming helpers
# -----------------------------------------------------------------------------

@dataclass
class StreamEvent:
    ts: str
    stream: str   # "stdout" | "stderr"
    line: str

def _enqueue_stream(pipe, label: str, q: "queue.Queue[StreamEvent]") -> None:
    for raw in iter(pipe.readline, ""):
        q.put(StreamEvent(ts=_now_ts(), stream=label, line=raw.rstrip("\n")))
    pipe.close()

def _iter_stream(proc: subprocess.Popen, q: "queue.Queue[StreamEvent]") -> Iterator[StreamEvent]:
    """
    Yield lines from stdout/stderr in real-time until process terminates.
    """
    while True:
        try:
            evt = q.get(timeout=0.05)
            yield evt
        except queue.Empty:
            if proc.poll() is not None and q.empty():
                break

# -----------------------------------------------------------------------------
# Result model
# -----------------------------------------------------------------------------

@dataclass
class CliResult:
    command: List[str]
    cwd: str
    start_ts: str
    end_ts: str
    duration_sec: float
    returncode: int
    stdout_tail: List[str]
    stderr_tail: List[str]
    debug_log: Optional[str] = None
    env_logged: Optional[Dict[str, str]] = None
    note: Optional[str] = None

    def as_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------

def run_cli(
    argv: Union[str, Iterable[str]],
    *,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    stream: bool = False,
    log_tail_lines: int = DEFAULT_LOG_TAIL,
) -> Union[CliResult, Iterator[StreamEvent]]:
    """
    Execute a command safely (no shell=True) and return a structured result,
    or a real-time stream of events if stream=True.

    Parameters
    ----------
    argv : str | Iterable[str]
        The command and arguments. If a string is provided, it is split
        with shlex to form argv.
    timeout : int | None
        Max execution seconds (defaults to DEFAULT_TIMEOUT).
    cwd : str | Path | None
        Working directory (defaults to PROJECT_ROOT).
    env_overrides : Mapping[str, str] | None
        Environment overlay; values redact-logged.
    stream : bool
        If True, returns an iterator of StreamEvent (stdout/stderr lines).
    log_tail_lines : int
        Number of lines from stdout/stderr to include in the summary.

    Returns
    -------
    CliResult | Iterator[StreamEvent]
    """
    argv_list = _ensure_list(argv)
    if not argv_list:
        raise ValueError("run_cli(): empty argv")

    workdir = Path(cwd or PROJECT_ROOT).resolve()
    if not workdir.exists():
        raise FileNotFoundError(f"Working directory not found: {workdir}")

    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    safe_env_for_log = _redact_env(env)

    # Launch subprocess
    start_ts = _now_ts()
    started = time.time()
    proc = subprocess.Popen(
        argv_list,
        cwd=str(workdir),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    if stream:
        # Real-time streaming mode
        q: "queue.Queue[StreamEvent]" = queue.Queue()
        t_out = threading.Thread(target=_enqueue_stream, args=(proc.stdout, "stdout", q), daemon=True)
        t_err = threading.Thread(target=_enqueue_stream, args=(proc.stderr, "stderr", q), daemon=True)
        t_out.start(); t_err.start()

        def _iter() -> Iterator[StreamEvent]:
            try:
                deadline = None if timeout is None else (time.time() + timeout)
                for evt in _iter_stream(proc, q):
                    yield evt
                    if deadline is not None and time.time() > deadline:
                        try:
                            proc.send_signal(signal.SIGINT)
                            time.sleep(1.0)
                            if proc.poll() is None:
                                proc.kill()
                        finally:
                            break
                # Drain remaining queue if any
                while not q.empty():
                    yield q.get()
            finally:
                # Ensure threads have ended
                t_out.join(timeout=0.2)
                t_err.join(timeout=0.2)
        return _iter()

    # Non-streaming: wait and collect
    try:
        out, err = proc.communicate(timeout=(timeout or DEFAULT_TIMEOUT))
    except subprocess.TimeoutExpired:
        # Try graceful interruption
        try:
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=5)
        except Exception:
            proc.kill()
            out, err = proc.communicate()
        end_ts = _now_ts()
        duration = time.time() - started
        return CliResult(
            command=argv_list,
            cwd=str(workdir),
            start_ts=start_ts,
            end_ts=end_ts,
            duration_sec=duration,
            returncode=124,  # common "timeout" code
            stdout_tail=_tail_lines(out, log_tail_lines),
            stderr_tail=_tail_lines(err, log_tail_lines),
            debug_log=_safe_read_tail(DEFAULT_DEBUG_LOG, 1200),
            env_logged=safe_env_for_log,
            note=f"Process timed out after {(timeout or DEFAULT_TIMEOUT)}s",
        )

    end_ts = _now_ts()
    duration = time.time() - started
    return CliResult(
        command=argv_list,
        cwd=str(workdir),
        start_ts=start_ts,
        end_ts=end_ts,
        duration_sec=duration,
        returncode=proc.returncode,
        stdout_tail=_tail_lines(out, log_tail_lines),
        stderr_tail=_tail_lines(err, log_tail_lines),
        debug_log=_safe_read_tail(DEFAULT_DEBUG_LOG, 1200),
        env_logged=safe_env_for_log,
    )

def _safe_read_tail(path: Optional[str], max_chars: int) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) <= max_chars:
            return text
        return text[-max(0, max_chars):]
    except Exception:
        return None

# -----------------------------------------------------------------------------
# SpectraMind convenience wrappers
# -----------------------------------------------------------------------------

def spectramind_run(
    subcommand: str,
    *,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    stream: bool = False,
) -> Union[CliResult, Iterator[StreamEvent]]:
    """
    Execute: spectramind {subcommand} [extra_args...]
    Example:
        spectramind_run("diagnose dashboard", extra_args=["--no-tsne"])
    """
    base = _ensure_list([DEFAULT_CLI] + shlex.split(subcommand))
    argv = base + ([] if not extra_args else _ensure_list(extra_args))
    return run_cli(
        argv,
        timeout=timeout or DEFAULT_TIMEOUT,
        cwd=cwd or PROJECT_ROOT,
        env_overrides=env_overrides,
        stream=stream,
    )

def calibrate(
    *,
    config_overrides: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
) -> CliResult:
    """
    Run the calibration stage:
        spectramind calibrate [Hydra overrides...]
    """
    extra = _ensure_list(config_overrides or [])
    return spectramind_run("calibrate", extra_args=extra, timeout=timeout, cwd=cwd, env_overrides=env_overrides)  # type: ignore[return-value]

def train(
    *,
    config_overrides: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
) -> CliResult:
    """
    Run model training:
        spectramind train [Hydra overrides...]
    """
    extra = _ensure_list(config_overrides or [])
    return spectramind_run("train", extra_args=extra, timeout=timeout, cwd=cwd, env_overrides=env_overrides)  # type: ignore[return-value]

def diagnose_dashboard(
    *,
    flags: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
) -> CliResult:
    """
    Build/refresh diagnostics dashboard artifacts:
        spectramind diagnose dashboard [flags]
    Common flags: --no-umap, --no-tsne, --open-html=false, --versioned=true
    """
    extra = _ensure_list(flags or [])
    return spectramind_run("diagnose dashboard", extra_args=extra, timeout=timeout, cwd=cwd, env_overrides=env_overrides)  # type: ignore[return-value]

def submit_bundle(
    *,
    flags: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
) -> CliResult:
    """
    Create leaderboard-ready submission bundle:
        spectramind submit [flags]
    """
    extra = _ensure_list(flags or [])
    return spectramind_run("submit", extra_args=extra, timeout=timeout, cwd=cwd, env_overrides=env_overrides)  # type: ignore[return-value]

# -----------------------------------------------------------------------------
# Optional: small CLI for smoke testing this bridge
# -----------------------------------------------------------------------------

def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))

def main(argv: Optional[List[str]] = None) -> int:
    """
    Example usage:
        python -m src.server.cli_bridge diagnose -- --no-tsne
        python -m src.server.cli_bridge train -- data=nominal trainer=ci_fast
        python -m src.server.cli_bridge run -- spectramind --version
    """
    args = _ensure_list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("usage: python -m src.server.cli_bridge <command> [-- extra args...]")
        print("  commands: calibrate | train | diagnose | submit | run")
        return 2

    cmd = args[0]
    extra: List[str] = []
    if "--" in args:
        dash = args.index("--")
        extra = args[dash + 1 :]
        args = args[:dash]

    try:
        if cmd == "calibrate":
            res = calibrate(config_overrides=extra)
            _print_json(asdict(res)); return 0
        if cmd == "train":
            res = train(config_overrides=extra)
            _print_json(asdict(res)); return 0
        if cmd == "diagnose":
            res = diagnose_dashboard(flags=extra)
            _print_json(asdict(res)); return 0
        if cmd == "submit":
            res = submit_bundle(flags=extra)
            _print_json(asdict(res)); return 0
        if cmd == "run":
            if not extra:
                print("usage: … run -- <argv...>")
                return 2
            res = run_cli(extra)
            if isinstance(res, CliResult):
                _print_json(asdict(res)); return res.returncode
            else:
                # stream mode for manual testing
                for evt in res:  # type: ignore
                    print(f"[{evt.ts}] {evt.stream}: {evt.line}")
                return 0
        print(f"unknown command: {cmd}")
        return 2
    except Exception as e:
        _print_json({"error": str(e)})
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
