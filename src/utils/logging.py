#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/logging.py — SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge)

Mission‑grade logging utilities for a CLI‑first, Hydra‑driven scientific pipeline.

Goals
-----
• One‑liner initialization that “just works” locally, on CI, and on Kaggle:
    logger = init_logger("logs/train.log", level="INFO", rich=True)
• Rich (color) console + plain file + optional JSONL event stream.
• Rank‑aware safe logging (does nothing on non‑rank‑0 unless enabled).
• Minimal deps (Rich is optional); safe when unavailable.
• Handy helpers: log_metrics(), JsonlLogger, Timer, tee_stdout(), capture_exceptions().
• Opt‑in integrations: MLflow/TensorBoard if present (no hard dependency).

Typical use
-----------
    from utils.logging import init_logger, JsonlLogger, log_metrics, Timer
    logger = init_logger("logs/train.log", level="INFO", rich=True)
    jsonl  = JsonlLogger("logs/events.jsonl")

    with Timer() as t:
        ... work ...
    log_metrics({"epoch": 3, "train_loss": 0.123, "t_s": t.seconds}, run_hash)

Design notes
------------
• Keeps formatting lean in files (machine‑parseable), pretty in TTY (when Rich available).
• All writers create parent dirs; writes are atomic where feasible.
• No global mutable state besides a tiny module cache for handlers to avoid duplicates.

Copyright
---------
(c) 2025 SpectraMind Team. MIT License.
"""

from __future__ import annotations

import os
import io
import sys
import json
import time
import atexit
import queue
import types
import signal
import shutil
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union, Tuple, Callable, ContextManager
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------
try:  # Rich is optional (for pretty console)
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install as rich_install
    _HAS_RICH = True
except Exception:
    Console = None  # type: ignore
    RichHandler = object  # type: ignore
    rich_install = lambda **_: None  # type: ignore
    _HAS_RICH = False

try:  # MLflow is optional
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

try:  # TensorBoard is optional
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAS_TB = True
except Exception:
    SummaryWriter = None  # type: ignore
    _HAS_TB = False


# -----------------------------------------------------------------------------
# Helpers & module cache
# -----------------------------------------------------------------------------
_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_CONSOLE: Optional[Console] = None
_RANK: int = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
_IS_MASTER: bool = (_RANK == 0)


def _ensure_dir(path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _fmt_plain() -> logging.Formatter:
    # timestamp | level | name | message
    return logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def init_logger(
    file_path: Optional[Union[str, Path]] = None,
    *,
    level: str = "INFO",
    rich: bool = True,
    name: str = "",
    propagate: bool = False,
    also_return_console: bool = False,
) -> Union[logging.Logger, Tuple[logging.Logger, Optional[Console]]]:
    """
    Initialize a root (or named) logger with:
      • Rich pretty console (if available and desired) OR plain StreamHandler
      • Optional file handler with plain format (machine‑friendly)
    Avoids duplicate handlers on repeated calls.

    Args
    ----
    file_path: path to the log file (created if provided)
    level:     logging level string ("DEBUG", "INFO", ...)
    rich:      use RichHandler if available and TTY
    name:      logger name ("" means root)
    propagate: whether to propagate to parent loggers
    also_return_console: return (logger, console)

    Returns
    -------
    logging.Logger or (logger, Console|None)
    """
    global _CONSOLE
    if name in _LOGGER_CACHE:
        lg = _LOGGER_CACHE[name]
        return (lg, _CONSOLE) if also_return_console else lg

    logger = logging.getLogger(name if name else None)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = bool(propagate)

    # Clear preexisting handlers (only first time) to ensure consistent format
    if not logger.handlers:
        # Console handler
        if rich and _HAS_RICH and _is_tty(sys.stdout):
            _CONSOLE = Console(highlight=True)
            ch = RichHandler(console=_CONSOLE, show_time=False, show_level=True, show_path=False, markup=True)
            ch.setLevel(getattr(logging, level.upper(), logging.INFO))
            logger.addHandler(ch)
            # Pretty tracebacks
            try:
                rich_install(show_locals=False, width=shutil.get_terminal_size((120, 20)).columns)
            except Exception:
                pass
        else:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(getattr(logging, level.upper(), logging.INFO))
            ch.setFormatter(_fmt_plain())
            logger.addHandler(ch)

        # File handler
        if file_path:
            _ensure_dir(file_path)
            fh = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
            fh.setLevel(getattr(logging, level.upper(), logging.INFO))
            fh.setFormatter(_fmt_plain())
            logger.addHandler(fh)

    _LOGGER_CACHE[name] = logger
    return (logger, _CONSOLE) if also_return_console else logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger with the same handlers (init_logger() must be called at least once).
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    lg = logging.getLogger(name)
    if not lg.handlers and "" in _LOGGER_CACHE:
        # inherit root handlers if we initialized the root
        for h in _LOGGER_CACHE[""].handlers:
            lg.addHandler(h)
        lg.setLevel(_LOGGER_CACHE[""].level)
        lg.propagate = False
    _LOGGER_CACHE[name] = lg
    return lg


# -----------------------------------------------------------------------------
# JSONL writer & metrics helpers
# -----------------------------------------------------------------------------
class JsonlLogger:
    """
    Minimal JSONL event logger. Always appends; creates parent dirs.
    Designed to be safe in multi-process (best effort, not lock-based).
    """
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        _ensure_dir(self.path)

    def log(self, event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_metrics(
    metrics: Mapping[str, Any],
    run_hash: Optional[str] = None,
    *,
    jsonl: Optional[JsonlLogger] = None,
    logger: Optional[logging.Logger] = None,
    step: Optional[int] = None,
    mlflow_prefix: Optional[str] = None,
    tb_writer: Optional["SummaryWriter"] = None,  # noqa: F821
) -> None:
    """
    Log metrics in three places:
      • Human console/file via `logger` (pretty key=val string)
      • Append to JSONL (if provided)
      • MLflow/TensorBoard when available (optional)

    Example:
        log_metrics({"epoch": 3, "train_loss": 0.123}, run_hash, jsonl=j, logger=lg, step=3)
    """
    kv = dict(metrics)
    if step is not None:
        kv["step"] = step
    if run_hash:
        kv["run_hash"] = run_hash

    # Console/file
    if logger is None:
        logger = get_logger("spectramind.metrics")
    try:
        as_line = " ".join([f"{k}={v}" for k, v in kv.items()])
        logger.info(as_line)
    except Exception:
        logger.info(str(kv))

    # JSONL
    if jsonl is not None:
        try:
            jsonl.log(kv)
        except Exception:
            logger.debug("Failed to write JSONL metrics.", exc_info=True)

    # MLflow
    if _HAS_MLFLOW:
        try:
            if step is not None:
                mlflow.log_metrics({str(k): float(v) for k, v in metrics.items() if _is_number(v)}, step=step)
            else:
                mlflow.log_metrics({str(k): float(v) for k, v in metrics.items() if _is_number(v)})
        except Exception:
            logger.debug("MLflow logging failed.", exc_info=True)

    # TensorBoard
    if _HAS_TB and tb_writer is not None:
        try:
            for k, v in metrics.items():
                if _is_number(v):
                    tb_writer.add_scalar(f"{mlflow_prefix + '/' if mlflow_prefix else ''}{k}", float(v), global_step=step or 0)
        except Exception:
            logger.debug("TensorBoard logging failed.", exc_info=True)


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Timers & progress
# -----------------------------------------------------------------------------
@dataclass
class Timer:
    """Simple wall‑clock timer context."""
    start: float = 0.0
    end: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.time()

    @property
    def seconds(self) -> float:
        return (self.end or time.time()) - self.start

    def __str__(self) -> str:
        return f"{self.seconds:.3f}s"


# -----------------------------------------------------------------------------
# Exception capture & fail‑safe hooks
# -----------------------------------------------------------------------------
def capture_exceptions(logger: Optional[logging.Logger] = None) -> None:
    """
    Install a global exception hook that logs uncaught exceptions (rank‑0 only by default).
    """
    lg = logger or get_logger("spectramind")

    def _hook(exc_type, exc, tb):
        if _IS_MASTER:
            msg = "".join(traceback.format_exception(exc_type, exc, tb))
            lg.error("Uncaught exception:\n%s", msg)
        # Call the default hook afterwards to preserve behavior
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


@contextmanager
def tee_stdout(file_path: Union[str, Path]) -> ContextManager[None]:
    """
    Context manager that tees stdout/stderr to a file (append mode) while still writing to console.
    """
    _ensure_dir(file_path)
    f = open(file_path, "a", encoding="utf-8")

    class Tee(io.TextIOBase):
        def __init__(self, a, b):
            self.a, self.b = a, b
        def write(self, s):
            self.a.write(s)
            self.b.write(s)
            self.a.flush()
            self.b.flush()
            return len(s)
        def flush(self):
            self.a.flush()
            self.b.flush()

    try:
        with redirect_stdout(Tee(sys.stdout, f)), redirect_stderr(Tee(sys.stderr, f)):
            yield
    finally:
        f.close()


# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------
def log_artifact(path: Union[str, Path], *, logger: Optional[logging.Logger] = None) -> None:
    """
    Best‑effort artifact logging with MLflow if present. Always no‑op on non‑master ranks.
    """
    if not _IS_MASTER:
        return
    p = Path(path)
    if logger is None:
        logger = get_logger("spectramind.artifacts")
    if not p.exists():
        logger.warning("Artifact not found: %s", p)
        return
    logger.info("Artifact: %s", p)
    if _HAS_MLFLOW:
        try:
            mlflow.log_artifact(str(p))
        except Exception:
            logger.debug("MLflow artifact logging failed.", exc_info=True)


# -----------------------------------------------------------------------------
# Self‑test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    lg = init_logger("logs/selftest.log", level="INFO", rich=True)
    lg.info("Hello from utils.logging self‑test.")
    j = JsonlLogger("logs/selftest_events.jsonl")
    log_metrics({"a": 1, "b": 2.5}, run_hash="abc123", jsonl=j, logger=lg, step=1)
    with Timer() as t:
        time.sleep(0.15)
    log_metrics({"phase": "sleep", "t": t.seconds}, jsonl=j, logger=lg)
    capture_exceptions(lg)
    try:
        with tee_stdout("logs/selftest_tee.log"):
            print("Tee stdout line.")
            raise ValueError("Demonstration exception")
    except Exception:
        lg.exception("Caught demo exception")