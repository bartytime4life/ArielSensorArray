# src/server/cli_bridge.py
# =============================================================================
# 🔗 SpectraMind V50 — CLI Bridge (Server-Safe Wrapper for `spectramind …`)
# -----------------------------------------------------------------------------
# Why this exists
#   • The GUI and API are thin veneers over the CLI. This module provides a *safe*,
#     reproducible way for the server to invoke CLI subcommands and return
#     structured results — without performing analytics here.
#
# Design goals
#   • CLI-first & reproducible: never compute or mutate artifacts here. All work
#     is delegated to the Typer CLI (`spectramind …`) which logs and writes
#     artifacts under Hydra/DVC-managed outputs.
#   • Security-minded execution: no shell=True; strict argv lists; redaction of
#     sensitive env; cwd sandboxing; process-group termination and timeouts.
#   • Developer ergonomics: a single `run_cli()` with convenience wrappers for
#     common subcommands (calibrate/train/diagnose/submit), optional streaming,
#     JSON tail extraction, result dataclass, simple retry/backoff.
#
# What this module provides
#   • run_cli(argv, *, timeout, cwd, env_overrides, stream, tee_path) -> CliResult|iter
#   • spectramind_run(subcommand_str, extra_args, **opts) -> CliResult|iter
#   • helpers: calibrate(), train(), diagnose_dashboard(), submit_bundle()
#   • graceful_kill(result) to terminate lingering processes by group
#
# NOTE: This module does NOT expose FastAPI routes; import it from API modules
#       (e.g., src/server/api/diagnostics.py) to execute the CLI.
# =============================================================================

from __future__ import annotations

import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
DEFAULT_CLI = os.getenv("SPECTRAMIND_CLI", "spectramind")
DEFAULT_TIMEOUT = int(os.getenv("CLI_TIMEOUT_SECONDS", "1800"))  # 30 min
DEFAULT_LOG_TAIL = int(os.getenv("CLI_LOG_TAIL_LINES", "200"))
DEFAULT_ENV_REDACT_KEYS = {
    "HF_TOKEN",
    "WANDB_API_KEY",
    "MLFLOW_TRACKING_PASSWORD",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_KEY",
}

# Optional canonical debug log file (written by the CLI)
DEFAULT_DEBUG_LOG = os.getenv("V50_DEBUG_LOG", str(PROJECT_ROOT / "v50_debug_log.md"))

# Optional sandbox for CWD safety (defaults to PROJECT_ROOT)
CWD_SANDBOX = Path(os.getenv("CWD_SANDBOX", str(PROJECT_ROOT))).resolve()

# Retry defaults
DEFAULT_RETRIES = int(os.getenv("CLI_RETRIES", "0"))
DEFAULT_RETRY_BACKOFF = float(os.getenv("CLI_RETRY_BACKOFF", "1.5"))  # seconds multiplier


# -----------------------------------------------------------------------------
# Types & Errors
# -----------------------------------------------------------------------------

class CliBridgeError(RuntimeError):
    """Base error for CLI bridge failures."""


class CliSecurityError(CliBridgeError):
    """Raised when cwd or argv validation fails."""


class CliTimeout(CliBridgeError):
    """Raised when a CLI invocation times out."""


@dataclass
class StreamEvent:
    ts: str
    stream: str   # "stdout" | "stderr"
    line: str


@dataclass
class CliResult:
    command: List[str]
    cwd: str
    start_ts: str
    end_ts: str
    duration_sec: float
    returncode: int
    pid: Optional[int]
    timed_out: bool
    stdout_tail: List[str]
    stderr_tail: List[str]
    debug_log: Optional[str] = None
    env_logged: Optional[Dict[str, str]] = None
    note: Optional[str] = None

    def as_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_list(argv: Union[str, Iterable[str]]) -> List[str]:
    """
    Accept either a string (split with shlex) or an iterable of strings and return a safe argv list.
    """
    if isinstance(argv, str):
        return shlex.split(argv)
    return [str(a) for a in argv]


def _redact_env(env: Mapping[str, str], redact_keys: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Return a copy of env with sensitive values redacted for logging."""
    redact = set(DEFAULT_ENV_REDACT_KEYS if redact_keys is None else redact_keys)
    safe: Dict[str, str] = {}
    for k, v in env.items():
        safe[k] = "********" if k in redact else v
    return safe


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _tail_lines(text: str, n: int) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    return lines[-max(0, n):]


def _safe_read_tail(path: Optional[str], max_chars: int) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= max_chars else text[-max(0, max_chars):]
    except Exception:
        return None


def _validate_cwd(cwd: Path) -> Path:
    """
    Ensure cwd exists and is within the CWD_SANDBOX.
    """
    c = cwd.resolve()
    if not c.exists():
        raise CliSecurityError(f"CWD does not exist: {c}")
    sandbox = CWD_SANDBOX
    if not str(c).startswith(str(sandbox)):
        raise CliSecurityError(f"CWD '{c}' not within sandbox '{sandbox}'")
    return c


def _popen_kwargs_for_platform() -> Dict[str, Any]:
    """
    Provide platform-safe Popen kwargs for starting a new process group.
    This allows graceful group termination on timeout/interrupts.
    """
    kwargs: Dict[str, Any] = {}
    if os.name == "posix":
        kwargs["preexec_fn"] = os.setsid  # new process group
    else:
        # On Windows, CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    return kwargs


def _terminate_process(proc: subprocess.Popen, *, grace_sec: float = 1.0) -> None:
    """
    Terminate a process group gracefully, then force kill if needed.
    """
    try:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:
                proc.send_signal(signal.SIGINT)
        else:
            # Windows: CTRL_BREAK_EVENT only works for console processes with new groups;
            # fall back to terminate.
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except Exception:
                proc.terminate()
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Streaming helpers
# -----------------------------------------------------------------------------

def _enqueue_stream(pipe, label: str, q: "queue.Queue[StreamEvent]") -> None:
    for raw in iter(pipe.readline, ""):
        q.put(StreamEvent(ts=_now_ts(), stream=label, line=raw.rstrip("\n")))
    try:
        pipe.close()
    except Exception:
        pass


def _iter_stream(proc: subprocess.Popen, q: "queue.Queue[StreamEvent]") -> Iterator[StreamEvent]:
    """
    Yield lines from stdout/stderr in real-time until process terminates and queues drain.
    """
    while True:
        try:
            evt = q.get(timeout=0.05)
            yield evt
        except queue.Empty:
            if proc.poll() is not None and q.empty():
                break


# -----------------------------------------------------------------------------
# Core runner (with optional streaming & retry)
# -----------------------------------------------------------------------------

def run_cli(
    argv: Union[str, Iterable[str]],
    *,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    stream: bool = False,
    log_tail_lines: int = DEFAULT_LOG_TAIL,
    tee_path: Optional[Union[str, Path]] = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Union[CliResult, Iterator[StreamEvent]]:
    """
    Execute a command safely (no shell=True) and return a structured CliResult,
    or a real-time iterator of StreamEvent if stream=True.

    Parameters
    ----------
    argv : str | Iterable[str]
        Command and arguments. If a string is provided, it is split with shlex.
    timeout : int | None
        Max execution seconds. Defaults to DEFAULT_TIMEOUT.
    cwd : str | Path | None
        Working directory (validated within CWD_SANDBOX). Defaults to PROJECT_ROOT.
    env_overrides : Mapping[str, str] | None
        Environment overlay; values are redact-logged.
    stream : bool
        If True, returns an iterator of StreamEvent (stdout/stderr lines).
    log_tail_lines : int
        Number of lines from stdout/stderr to include in the summary tails.
    tee_path : str | Path | None
        If provided, stdout/stderr are mirrored to this file (append mode).
    retries : int
        Number of immediate retries on non-zero exit (except timeouts).
    retry_backoff : float
        Backoff multiplier (seconds) between retries.

    Returns
    -------
    CliResult | Iterator[StreamEvent]
    """
    argv_list = _ensure_list(argv)
    if not argv_list:
        raise ValueError("run_cli(): empty argv")

    workdir = _validate_cwd(Path(cwd or PROJECT_ROOT))
    timeout = timeout or DEFAULT_TIMEOUT

    # Prepare environment
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    safe_env_for_log = _redact_env(env)
    popen_extra = _popen_kwargs_for_platform()

    # Optional tee file
    tee_file = None
    if tee_path is not None:
        tee_file = Path(tee_path)
        try:
            tee_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    attempt = 0
    last_err: Optional[str] = None

    while True:
        attempt += 1
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
            **popen_extra,
        )

        if stream:
            # Streaming mode: yield events, then compose a final CliResult when finished.
            q: "queue.Queue[StreamEvent]" = queue.Queue()
            t_out = threading.Thread(target=_enqueue_stream, args=(proc.stdout, "stdout", q), daemon=True)
            t_err = threading.Thread(target=_enqueue_stream, args=(proc.stderr, "stderr", q), daemon=True)
            t_out.start(); t_err.start()

            def _iter() -> Iterator[StreamEvent]:
                deadline = None if timeout is None else (time.time() + timeout)
                try:
                    for evt in _iter_stream(proc, q):
                        # Optional tee
                        if tee_file is not None:
                            try:
                                tee_file.write_text("", encoding="utf-8")  # touch on first write
                            except Exception:
                                pass
                            try:
                                with tee_file.open("a", encoding="utf-8") as f:
                                    f.write(f"{evt.ts} [{evt.stream}] {evt.line}\n")
                            except Exception:
                                pass
                        yield evt
                        if deadline is not None and time.time() > deadline:
                            _terminate_process(proc)
                            break
                    # Drain remaining (if any)
                    while not q.empty():
                        yield q.get()
                finally:
                    try:
                        t_out.join(timeout=0.2)
                        t_err.join(timeout=0.2)
                    except Exception:
                        pass
            return _iter()

        # Non-streaming: wait and collect with timeout handling
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process(proc)
            out, err = proc.communicate()

        end_ts = _now_ts()
        duration = time.time() - started

        # Tee full outputs if requested
        if tee_file is not None:
            try:
                with tee_file.open("a", encoding="utf-8") as f:
                    if out:
                        f.write(out if out.endswith("\n") else out + "\n")
                    if err:
                        f.write(err if err.endswith("\n") else err + "\n")
            except Exception:
                pass

        result = CliResult(
            command=argv_list,
            cwd=str(workdir),
            start_ts=start_ts,
            end_ts=end_ts,
            duration_sec=duration,
            returncode=proc.returncode,
            pid=proc.pid,
            timed_out=timed_out,
            stdout_tail=_tail_lines(out, log_tail_lines),
            stderr_tail=_tail_lines(err, log_tail_lines),
            debug_log=_safe_read_tail(DEFAULT_DEBUG_LOG, 2000),
            env_logged=safe_env_for_log,
            note=(
                f"Process timed out after {timeout}s"
                if timed_out else
                (f"Attempt {attempt}/{1 + retries}" if retries > 0 else None)
            ),
        )

        if timed_out:
            # Do not retry timeouts (caller should adjust timeout if needed)
            raise CliTimeout(result.as_json())

        if proc.returncode == 0 or attempt > retries + 1:
            return result

        # Retry on non-zero exit
        last_err = f"Non-zero exit ({proc.returncode}) on attempt {attempt}; retrying…"
        time.sleep(max(0.1, retry_backoff * attempt))


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
    tee_path: Optional[Union[str, Path]] = None,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
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
        timeout=timeout,
        cwd=cwd or PROJECT_ROOT,
        env_overrides=env_overrides,
        stream=stream,
        tee_path=tee_path,
        retries=retries,
        retry_backoff=retry_backoff,
    )


def calibrate(
    *,
    config_overrides: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    tee_path: Optional[Union[str, Path]] = None,
    retries: int = DEFAULT_RETRIES,
) -> CliResult:
    """
    Run the calibration stage:
        spectramind calibrate [Hydra overrides...]
    """
    extra = _ensure_list(config_overrides or [])
    return spectramind_run(
        "calibrate",
        extra_args=extra,
        timeout=timeout or DEFAULT_TIMEOUT,
        cwd=cwd,
        env_overrides=env_overrides,
        stream=False,
        tee_path=tee_path,
        retries=retries,
    )  # type: ignore[return-value]


def train(
    *,
    config_overrides: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    tee_path: Optional[Union[str, Path]] = None,
    retries: int = DEFAULT_RETRIES,
) -> CliResult:
    """
    Run model training:
        spectramind train [Hydra overrides...]
    """
    extra = _ensure_list(config_overrides or [])
    return spectramind_run(
        "train",
        extra_args=extra,
        timeout=timeout or DEFAULT_TIMEOUT,
        cwd=cwd,
        env_overrides=env_overrides,
        stream=False,
        tee_path=tee_path,
        retries=retries,
    )  # type: ignore[return-value]


def diagnose_dashboard(
    *,
    flags: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    tee_path: Optional[Union[str, Path]] = None,
    retries: int = DEFAULT_RETRIES,
) -> CliResult:
    """
    Build/refresh diagnostics dashboard artifacts:
        spectramind diagnose dashboard [flags]
    Common flags: --no-umap, --no-tsne, --open-html=false, --versioned=true
    """
    extra = _ensure_list(flags or [])
    return spectramind_run(
        "diagnose dashboard",
        extra_args=extra,
        timeout=timeout or DEFAULT_TIMEOUT,
        cwd=cwd,
        env_overrides=env_overrides,
        stream=False,
        tee_path=tee_path,
        retries=retries,
    )  # type: ignore[return-value]


def submit_bundle(
    *,
    flags: Optional[Iterable[str]] = None,
    timeout: Optional[int] = None,
    cwd: Optional[Union[str, Path]] = None,
    env_overrides: Optional[Mapping[str, str]] = None,
    tee_path: Optional[Union[str, Path]] = None,
    retries: int = DEFAULT_RETRIES,
) -> CliResult:
    """
    Create leaderboard-ready submission bundle:
        spectramind submit [flags]
    """
    extra = _ensure_list(flags or [])
    return spectramind_run(
        "submit",
        extra_args=extra,
        timeout=timeout or DEFAULT_TIMEOUT,
        cwd=cwd,
        env_overrides=env_overrides,
        stream=False,
        tee_path=tee_path,
        retries=retries,
    )  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Graceful kill for an in-flight process (if using streaming externally)
# -----------------------------------------------------------------------------

def graceful_kill(proc: subprocess.Popen, *, grace_sec: float = 1.0) -> None:
    """
    Public helper to terminate a process (group) started elsewhere by callers who
    used streaming `run_cli(..., stream=True)`.
    """
    _terminate_process(proc, grace_sec=grace_sec)


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
        python -m src.server.cli_bridge stream -- spectramind diagnose dashboard
    """
    args = _ensure_list(argv if argv is not None else sys.argv[1:])
    if not args:
        print("usage: python -m src.server.cli_bridge <command> [-- extra args...]")
        print("  commands: calibrate | train | diagnose | submit | run | stream")
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
                # stream mode shouldn't happen via 'run'
                return 0
        if cmd == "stream":
            if not extra:
                print("usage: … stream -- <argv...>")
                return 2
            it = run_cli(extra, stream=True)
            # Consume stream to stdout
            for evt in it:  # type: ignore
                print(f"[{evt.ts}] {evt.stream}: {evt.line}")
            return 0

        print(f"unknown command: {cmd}")
        return 2
    except CliTimeout as e:
        _print_json({"timeout": str(e)})
        return 124
    except CliSecurityError as e:
        _print_json({"security_error": str(e)})
        return 111
    except Exception as e:
        _print_json({"error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
