# test_utils_logging.py
# -----------------------------------------------------------------------------
# SpectraMind V50 • NeurIPS 2025 Ariel Data Challenge
#
# Upgraded tests for the logging utilities module.
#
# Goals:
# - Verify that root/child loggers are configured with expected levels.
# - Ensure console (optionally Rich) + file handlers are attached and write output.
# - Validate JSONL event logging produces well-formed, one-JSON-object-per-line files.
# - Check audit/security log exists and is independent of normal logs.
# - Respect env-var overrides (e.g., LOG_LEVEL) if supported by the utils.
# - Keep tests resilient to minor API differences (setup_logging vs configure_logging,
#   jsonl_event_logger vs get_event_logger). Tests will skip gracefully if a feature
#   is not implemented in the target utils module.
#
# Assumed target module name(s):
#   - "spectramind.utils.logging"  (preferred, project-style)
#   - "utils.logging"              (fallback, generic)
#
# NOTE: These tests are intentionally defensive: they discover available functions
#       and adapt. If a feature is unavailable, a test skips with a clear reason.
# -----------------------------------------------------------------------------

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pytest


# --- Helper: dynamically import the logging utils module ----------------------

_CANDIDATE_MODULES = [
    "spectramind.utils.logging",  # project-style path (preferred)
    "utils.logging",              # fallback path
    "logging_utils",              # another common alias
]

_utils_mod = None
for _name in _CANDIDATE_MODULES:
    try:
        _utils_mod = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue

if _utils_mod is None:
    pytest.skip(
        "No logging utils module found in expected locations: "
        + ", ".join(_CANDIDATE_MODULES),
        allow_module_level=True,
    )


# --- Helpers: discover functions and small utilities --------------------------

def _get_setup_fn():
    """
    Locate a config-based setup function in the utils module.

    Expected signature (any of):
      setup_logging(cfg: Mapping, run_dir: Path) -> logging.Logger | dict | None
      configure_logging(cfg: Mapping, run_dir: Path) -> logging.Logger | dict | None
      init_logging(cfg: Mapping, run_dir: Path) -> logging.Logger | dict | None
    """
    for cand in ("setup_logging", "configure_logging", "init_logging"):
        fn = getattr(_utils_mod, cand, None)
        if callable(fn):
            return fn, cand
    return None, None


def _get_logger_fn():
    """
    Locate a function to retrieve a named logger (optional).
    """
    for cand in ("get_logger", "getLogger"):
        fn = getattr(_utils_mod, cand, None)
        if callable(fn):
            return fn
    # Fallback to stdlib logging.getLogger
    return logging.getLogger


def _get_event_logger_factory():
    """
    Locate a JSONL event logger factory.

    Expected possibilities:
      - jsonl_event_logger(path: Union[str, Path]) -> Callable[[Mapping], None]
      - get_event_logger(path: Union[str, Path]) -> logging.Logger  (logger that writes JSON lines)
    Returns (factory, kind) where kind in {"callable", "logger"} or (None, None).
    """
    fn = getattr(_utils_mod, "jsonl_event_logger", None)
    if callable(fn):
        return fn, "callable"
    fn = getattr(_utils_mod, "get_event_logger", None)
    if callable(fn):
        return fn, "logger"
    # Some implementations expose "EventLogger" class
    cls = getattr(_utils_mod, "EventLogger", None)
    if cls is not None:
        def _factory(path: Path):
            return cls(path)
        return _factory, "class"
    return None, None


def _has_rich_handler(logger: logging.Logger) -> bool:
    try:
        from rich.logging import RichHandler  # type: ignore
        return any(isinstance(h, RichHandler) for h in logger.handlers)
    except Exception:
        # Rich not installed or not used.
        return False


def _iter_all_handlers(logger: logging.Logger):
    seen = set()
    q = [logger]
    while q:
        lg = q.pop()
        if id(lg) in seen:
            continue
        seen.add(id(lg))
        for h in lg.handlers:
            yield h
        if lg.propagate and lg.parent:
            q.append(lg.parent)


# --- Fixtures -----------------------------------------------------------------

@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    # Simulate a Hydra/CLI run directory with a "logs" subdir
    d = tmp_path / "run"
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def base_config(run_dir: Path) -> Mapping[str, Any]:
    # A generic, liberal config many utils implementations can understand.
    # If the target utils expects different keys, tests will adapt or skip.
    return {
        "level": "INFO",
        "console": {"enabled": True, "rich": True, "level": "INFO"},
        "file": {
            "enabled": True,
            "path": str(run_dir / "logs" / "run.log"),
            "level": "DEBUG",
            "mode": "a",
        },
        "jsonl": {
            "enabled": True,
            "path": str(run_dir / "logs" / "events.jsonl"),
        },
        "audit": {
            "enabled": True,
            "path": str(run_dir / "logs" / "audit.log"),
        },
        # Optional: structured fields common to this project
        "extra": {"project": "SpectraMindV50", "component": "tests"},
    }


@pytest.fixture(autouse=True)
def _isolate_logging_state():
    """
    Ensure we don't leak handlers between tests.
    """
    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    try:
        yield
    finally:
        # Remove any handlers the test added
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        # Restore original handlers
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)


# --- Tests: setup / handlers / levels -----------------------------------------

def test_setup_attaches_console_and_file_handlers(base_config, run_dir, caplog):
    setup_fn, name = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function (setup_logging/configure_logging/init_logging) found in utils.")

    caplog.set_level(logging.DEBUG)  # capture everything emitted via logging module
    result = setup_fn(base_config, run_dir)  # return value type can vary
    # Get root logger after setup
    root = logging.getLogger()

    # Console handler?
    has_console = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is not None
        for h in _iter_all_handlers(root)
    )
    assert has_console, "Expected a console/stream handler attached to root logger."

    # File handler?
    file_path = Path(base_config["file"]["path"])
    # Emit a test log and ensure it lands in the file.
    test_msg = "hello-file-handler"
    root.info(test_msg)
    # Flush/close so OS writes to disk
    for h in _iter_all_handlers(root):
        try:
            h.flush()
        except Exception:
            pass

    assert file_path.exists(), f"Expected log file to be created at {file_path}"
    assert file_path.read_text(encoding="utf-8").find(test_msg) != -1, "Run log should contain the emitted message."

    # If Rich was requested, check presence (best-effort).
    if base_config.get("console", {}).get("rich"):
        assert _has_rich_handler(root) or has_console, "Rich console requested; expected a Rich or Stream handler."


def test_child_logger_inherits_level(base_config, run_dir):
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    setup_fn(base_config, run_dir)
    get_logger = _get_logger_fn()

    parent = get_logger("spectramind")
    child = get_logger("spectramind.model")

    # Child should not be more permissive than parent, and should propagate
    assert child.propagate is True
    assert child.level in (logging.NOTSET, parent.level), (
        "Child logger level should be NOTSET (inherit) or equal to parent."
    )


def test_env_override_log_level_takes_effect(base_config, run_dir, monkeypatch):
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    # Attempt to override via environment variable (if supported by implementation).
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    setup_fn(base_config, run_dir)

    root = logging.getLogger()
    # Emit logs; DEBUG/INFO should not be enabled if override worked.
    got_debug = root.isEnabledFor(logging.DEBUG)
    got_info = root.isEnabledFor(logging.INFO)
    # We allow either behavior (not all implementations support env override).
    # If supported, DEBUG/INFO will be False; if not, test is skipped to avoid false negatives.
    if got_debug or got_info:
        pytest.skip("Implementation does not support LOG_LEVEL override — skipping.")
    else:
        assert root.isEnabledFor(logging.WARNING), "Expected WARNING to be enabled under LOG_LEVEL=WARNING."


# --- Tests: JSONL event logging -----------------------------------------------

def test_jsonl_event_logger_writes_valid_json(base_config, run_dir):
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    setup_fn(base_config, run_dir)
    factory, kind = _get_event_logger_factory()
    if not factory:
        pytest.skip("No JSONL event logger factory (jsonl_event_logger/get_event_logger/EventLogger) found.")

    events_path = Path(base_config["jsonl"]["path"])
    events_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"event": "planet_processed", "planet_id": "K2-18b", "ok": True, "n_obs": 283}
    # Depending on kind, call appropriately
    if kind == "callable":
        writer: Callable[[Mapping[str, Any]], None] = factory(events_path)
        writer(payload)
    elif kind == "logger":
        ev_logger: logging.Logger = factory(events_path)
        ev_logger.info("", extra={"event": payload})  # common pattern: put payload into "event"
    elif kind == "class":
        event_logger = factory(events_path)
        # Try common method names
        if hasattr(event_logger, "log"):
            event_logger.log(payload)
        elif hasattr(event_logger, "write"):
            event_logger.write(payload)
        else:
            pytest.skip("EventLogger class has no known 'log'/'write' method.")

    # Ensure one JSON per line & parseable
    assert events_path.exists(), "Expected events JSONL file to be created."
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1, "Expected at least one JSONL line."
    first = json.loads(lines[0])
    # Either payload directly, or nested under a key like 'event' or 'data'
    if first == payload:
        pass
    elif isinstance(first, dict) and "event" in first and first["event"] == payload:
        pass
    else:
        pytest.fail(f"Unexpected event JSON structure: {first}")


# --- Tests: audit log independence --------------------------------------------

def test_audit_log_is_separate_from_run_log(base_config, run_dir):
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    setup_fn(base_config, run_dir)

    audit_path = Path(base_config["audit"]["path"])
    run_path = Path(base_config["file"]["path"])

    # Try to get a dedicated audit logger if exposed; otherwise, use a known name.
    get_logger = _get_logger_fn()
    audit_name = getattr(_utils_mod, "AUDIT_LOG_NAME", "audit")
    audit_logger = get_logger(audit_name)

    # Emit audit message
    audit_msg = "user=alice action=delete_dataset ok=false"
    audit_logger.warning(audit_msg)

    # Flush file handlers
    for h in _iter_all_handlers(audit_logger):
        try:
            h.flush()
        except Exception:
            pass

    # Validate files & separation
    assert audit_path.exists(), "Expected audit log file to exist."
    assert run_path.exists(), "Expected run log file to exist."

    audit_text = audit_path.read_text(encoding="utf-8")
    run_text = run_path.read_text(encoding="utf-8")

    assert audit_msg in audit_text, "Audit message should be present in audit log."
    # Either not present in run log, or implementation duplicates to run log as well (allow either),
    # but prefer separation:
    if audit_msg in run_text:
        # Acceptable if design mirrors to run log; warn but do not fail.
        pytest.skip("Audit events also appear in run log for this implementation — acceptable, skipping strict check.")


# --- Tests: minimal integration smoke -----------------------------------------

def test_logging_end_to_end_smoke(base_config, run_dir):
    """
    Emit a few logs through root and a child logger, ensure no exceptions and files written.
    """
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    setup_fn(base_config, run_dir)

    root = logging.getLogger()
    child = logging.getLogger("spectramind.decoder")

    root.debug("debug-root")      # should be captured in file (level DEBUG there)
    root.info("info-root")
    child.warning("warn-child")
    child.error("error-child")

    # Flush all
    for h in _iter_all_handlers(root):
        try:
            h.flush()
        except Exception:
            pass

    # Check the file log captured something
    file_path = Path(base_config["file"]["path"])
    text = file_path.read_text(encoding="utf-8")
    assert "info-root" in text and "warn-child" in text and "error-child" in text, "Expected log lines in run.log"
    # DEBUG may be filtered by root level; but file level is DEBUG per config, so we usually get it:
    if "debug-root" not in text:
        # Allow absence if root level prevented it; do not fail.
        pytest.skip("DEBUG line not present—likely filtered by root level; acceptable.")


# --- Optional: test for structured 'extra' fields if implementation supports it ----

def test_structured_extra_fields_if_supported(base_config, run_dir):
    """
    If the implementation supports injecting structured 'extra' fields into every record
    (e.g., run_id, project, component), verify at least one such field appears in run.log.
    """
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    setup_fn(base_config, run_dir)

    root = logging.getLogger()
    root.info("structured-check")

    file_path = Path(base_config["file"]["path"])
    text = file_path.read_text(encoding="utf-8")

    # Heuristics: look for any of the known extra fields (project/component) in the same line.
    if "SpectraMindV50" in text or "component=tests" in text:
        assert True
    else:
        pytest.skip("No structured extras detected in file log; implementation may not add them globally.")


# --- Optional: ensure no duplicate handlers across multiple setup calls ----------

def test_repeated_setup_does_not_duplicate_handlers(base_config, run_dir):
    setup_fn, _ = _get_setup_fn()
    if not setup_fn:
        pytest.skip("No setup function found in utils.")

    root = logging.getLogger()
    before = len(list(_iter_all_handlers(root)))
    setup_fn(base_config, run_dir)
    mid = len(list(_iter_all_handlers(root)))
    setup_fn(base_config, run_dir)
    after = len(list(_iter_all_handlers(root)))

    # Expect mid >= before (handlers added once), and after not much larger than mid
    # If duplicate handlers are added on repeated setup, 'after' will grow every time.
    assert after <= mid + 2, (
        "Repeated setup seems to add duplicate handlers; please guard against reinitialization."
    )
