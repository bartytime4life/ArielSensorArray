# /tests/artifacts/test_cli_version_stamp.py
# ---------------------------------------------------------------------------
# SpectraMind V50 — CLI "version stamp" tests
#
# Goals:
#  1) The CLI must expose a version surface that humans and machines can rely on.
#  2) It should be accessible via BOTH `spectramind --version` and
#     `spectramind version` (and, if the Typer app is importable, via CliRunner).
#  3) A machine-readable (JSON) format must exist and include a stable set
#     of keys: at minimum [app_name, version, python_version, platform,
#     build_time, git_commit, dvc_data_hash, config_hash, run_id].
#
# These tests are intentionally defensive:
#  - If the Typer `app` is importable (e.g., from `spectramind.cli`), we run via CliRunner.
#  - Otherwise, we fall back to invoking the installed console script `spectramind`.
#  - If neither import nor console script is available, we mark the test as xfail
#    with a helpful message (so local devs know why it didn’t run).
#
# Implementation notes for the CLI implementer:
#  - `spectramind --version` SHOULD print a one‐liner for humans (non‑JSON).
#  - `spectramind version --json` SHOULD print a single JSON object (one line OK).
#  - Recommended keys for JSON:
#      app_name        : str    (e.g., "SpectraMind V50")
#      version         : str    (semantic version, e.g., "0.9.3+abc123")
#      python_version  : str    (e.g., "3.11.8")
#      platform        : str    (e.g., "Linux-6.8.0-...-x86_64-with-glibc2.35")
#      build_time      : str    (ISO 8601 UTC, e.g., "2025-08-18T21:14:55Z")
#      git_commit      : str    (7-40 char hex or "UNKNOWN")
#      dvc_data_hash   : str    (hash or "UNKNOWN")
#      config_hash     : str    (hash of the Hydra-composed config, if available)
#      run_id          : str    (UUIDv4 or similar unique token)
#      extras          : dict   (OPTIONAL bucket for additional stamps)
#
#  - If git/DVC/config hash are unknown in the current environment, set them
#    explicitly to "UNKNOWN" (don’t omit keys).
#
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Optional, Tuple

import pytest


# ------------------------------
# Utilities
# ------------------------------

def _import_cli_app() -> Optional[Tuple[object, object]]:
    """
    Try to import a Typer app `app` so we can use CliRunner without spawning a new process.

    We try a couple of plausible module paths used across this codebase:
      - spectramind.cli:app
      - src.spectramind.cli:app (when running tests with editable installs)
    Returns:
      (app, CliRunner) or None if not importable.
    """
    candidates = [
        ("spectramind.cli", "app"),
        ("src.spectramind.cli", "app"),
        ("spectramind", "app"),
    ]
    for mod, attr in candidates:
        try:
            module = __import__(mod, fromlist=[attr])
            app = getattr(module, attr, None)
            if app is not None:
                from typer.testing import CliRunner  # import here to avoid dependency when unused
                return app, CliRunner
        except Exception:
            continue
    return None


def _run_cli_capture(args: list[str], env: Optional[dict[str, str]] = None) -> Tuple[int, str, str]:
    """
    Run the CLI either via Typer CliRunner (preferred) or by subprocess.

    Args:
      args: arguments after the program name (e.g., ["--version"] or ["version", "--json"])
      env:  optional environment overrides

    Returns:
      (exit_code, stdout, stderr)
    """
    # 1) Try Typer CliRunner path if available
    cli_tuple = _import_cli_app()
    if cli_tuple:
        app, CliRunner = cli_tuple
        runner = CliRunner()
        # Typer runner returns result with exit_code, stdout (via result.output)
        result = runner.invoke(app, args, env=env)
        return result.exit_code, result.output, ""

    # 2) Fallback to subprocess using the console script if on PATH
    exe = shutil.which("spectramind")
    if exe:
        proc = subprocess.run(
            [exe] + args,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        return proc.returncode, proc.stdout, proc.stderr

    # 3) Nothing found; mark as "xfail" at call sites.
    return 127, "", "spectramind CLI not found (neither importable nor on PATH)"


def _assert_semver_like(version: str) -> None:
    """
    Accepts semver-ish strings, including optional local/build metadata.
      Examples that should pass:
        0.1.0
        1.2.3+abc123
        1.2.3-rc.1
        2.0.0b1
    """
    # Be generous: digits-dot-digits-dot-digits then optional suffixes
    pattern = r"^\d+\.\d+\.\d+([\-+][A-Za-z0-9\.\-_]+)?$"
    assert re.match(pattern, version), f"version '{version}' is not semver-like"


def _assert_iso8601_utc(ts: str) -> None:
    """
    Require a simple, safe subset of ISO 8601 in UTC with trailing 'Z'.
      e.g., 2025-08-18T21:14:55Z or 2025-08-18T21:14:55.123Z
    """
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$"
    assert re.match(pattern, ts), f"build_time '{ts}' is not ISO-8601 UTC with trailing 'Z'"


def _maybe_xfail_not_available(exit_code: int, stderr: str) -> None:
    if exit_code == 127 and "not found" in stderr.lower():
        pytest.xfail("spectramind CLI not available (neither importable Typer app nor console script on PATH).")


# ------------------------------
# Tests
# ------------------------------

def test_version_flag_human_line():
    """
    `spectramind --version` SHOULD print a single human-readable line and exit 0.
    It SHOULD include the app name and a semver-like version token.
    """
    exit_code, out, err = _run_cli_capture(["--version"])
    _maybe_xfail_not_available(exit_code, err)
    assert exit_code == 0, f"--version failed: {err or out}"

    # Expect a short line; avoid assuming exact phrasing.
    line = out.strip()
    assert line, "empty --version output"

    # Heuristics: contains a name-like token + a version-like token
    # Accept either "SpectraMind V50 0.9.3+abc123" or "spectramind 0.9.3" etc.
    tokens = line.split()
    assert len(tokens) >= 2, f"unexpected --version line: {line}"
    version_token = tokens[-1]
    _assert_semver_like(version_token)


@pytest.mark.parametrize("args", [["version"], ["version", "--human"]])
def test_version_subcommand_human_equivalent(args):
    """
    `spectramind version` (optionally with --human) SHOULD behave like `--version`,
    returning a human-readable one-liner with semver-like token.
    """
    exit_code, out, err = _run_cli_capture(args)
    _maybe_xfail_not_available(exit_code, err)
    assert exit_code == 0, f"{args} failed: {err or out}"
    line = out.strip()
    assert line, f"empty output for {args}"
    tokens = line.split()
    assert len(tokens) >= 2, f"unexpected output for {args}: {line}"
    _assert_semver_like(tokens[-1])


def test_version_json_core_fields_present_and_valid():
    """
    `spectramind version --json` SHOULD emit a JSON object with a stable schema.

    Required keys (must exist even if value is "UNKNOWN"):
      app_name, version, python_version, platform, build_time,
      git_commit, dvc_data_hash, config_hash, run_id
    """
    exit_code, out, err = _run_cli_capture(["version", "--json"])
    _maybe_xfail_not_available(exit_code, err)
    assert exit_code == 0, f"`version --json` failed: {err or out}"
    out = out.strip()
    assert out, "empty JSON output from `version --json`"

    try:
        data = json.loads(out)
    except Exception as e:
        pytest.fail(f"`version --json` did not return valid JSON: {e}\nRaw: {out}")

    required = [
        "app_name",
        "version",
        "python_version",
        "platform",
        "build_time",
        "git_commit",
        "dvc_data_hash",
        "config_hash",
        "run_id",
    ]
    for key in required:
        assert key in data, f"missing key `{key}` in version JSON"

    # Basic validations
    assert isinstance(data["app_name"], str) and data["app_name"], "app_name must be non-empty string"
    _assert_semver_like(str(data["version"]))
    assert isinstance(data["python_version"], str) and data["python_version"], "python_version must be string"
    assert isinstance(data["platform"], str) and data["platform"], "platform must be string"
    _assert_iso8601_utc(str(data["build_time"]))

    # git_commit/dvc_data_hash/config_hash are allowed to be "UNKNOWN", but must be strings.
    for k in ("git_commit", "dvc_data_hash", "config_hash"):
        v = data[k]
        assert isinstance(v, str), f"{k} must be string"
        assert v == "UNKNOWN" or len(v) >= 4, f"{k} appears too short/unset: {v}"

    # run_id should look UUID-ish (don’t require exact variant)
    run_id = str(data["run_id"])
    assert re.match(r"^[a-fA-F0-9\-]{8,}$", run_id), f"run_id does not look UUID-like: {run_id}"


def test_version_json_respects_env_overrides(monkeypatch):
    """
    If the CLI honors environment-provided stamps (recommended for CI builds),
    setting env vars should be reflected in the JSON.

    This test sets:
      SPECTRAMIND_GIT_COMMIT=deadbeefcafebabe
      SPECTRAMIND_DVC_DATA_HASH=abcd1234
      SPECTRAMIND_CONFIG_HASH=f00dcafe
    and expects the JSON fields to include those exact values.
    """
    env = {
        "SPECTRAMIND_GIT_COMMIT": "deadbeefcafebabe",
        "SPECTRAMIND_DVC_DATA_HASH": "abcd1234",
        "SPECTRAMIND_CONFIG_HASH": "f00dcafe",
    }
    # Ensure env is applied for both CliRunner and subprocess path
    exit_code, out, err = _run_cli_capture(["version", "--json"], env=env)
    _maybe_xfail_not_available(exit_code, err)
    assert exit_code == 0, f"`version --json` with env failed: {err or out}"

    data = json.loads(out.strip())
    assert data.get("git_commit") in (
        env["SPECTRAMIND_GIT_COMMIT"],
        # Implementations may truncate to 7-12 chars; accept prefix match:
        env["SPECTRAMIND_GIT_COMMIT"][:7],
        env["SPECTRAMIND_GIT_COMMIT"][:12],
    ), f"git_commit did not reflect env override: {data.get('git_commit')!r}"

    assert data.get("dvc_data_hash") == env["SPECTRAMIND_DVC_DATA_HASH"], \
        f"dvc_data_hash did not reflect env override"

    assert data.get("config_hash") == env["SPECTRAMIND_CONFIG_HASH"], \
        f"config_hash did not reflect env override"


def test_version_json_is_single_object_not_array_or_multiline_blob():
    """
    For ease of machine parsing, `version --json` SHOULD emit exactly one JSON object.
    Allow leading/trailing whitespace, but there should be exactly one top-level object.
    """
    exit_code, out, err = _run_cli_capture(["version", "--json"])
    _maybe_xfail_not_available(exit_code, err)
    assert exit_code == 0, f"`version --json` failed: {err or out}"
    s = out.strip()

    # Quick shape checks
    assert s.startswith("{") and s.endswith("}"), f"expected single JSON object, got: {s[:60]}..."
    # Should parse cleanly to dict
    obj = json.loads(s)
    assert isinstance(obj, dict), "top-level JSON is not an object"


def test_version_keys_are_stable_and_no_surprising_renames():
    """
    Guardrail: if the team changes key names in the JSON, break the build here.
    It nudges maintainers to update downstream consumers and this test list.
    """
    exit_code, out, err = _run_cli_capture(["version", "--json"])
    _maybe_xfail_not_available(exit_code, err)
    assert exit_code == 0, f"`version --json` failed: {err or out}"
    data = json.loads(out.strip())

    # Minimal stable set
    stable_keys = {
        "app_name",
        "version",
        "python_version",
        "platform",
        "build_time",
        "git_commit",
        "dvc_data_hash",
        "config_hash",
        "run_id",
    }
    missing = [k for k in stable_keys if k not in data]
    assert not missing, f"version JSON missing stable keys: {missing}"


# ------------------------------
# Optional: CLI smoke test via console binary only
# ------------------------------

def test_console_script_presence_or_app_importable():
    """
    This is a light smoke assertion to help developers:
      - Either the Typer app is importable (preferred)
      - Or the console script `spectramind` is on PATH (after `pip install -e .`)
    If neither is true in a local dev context, we xfail with guidance.
    """
    if _import_cli_app():
        return
    if shutil.which("spectramind"):
        return
    pytest.xfail(
        "Neither `spectramind.cli:app` importable nor `spectramind` console script found. "
        "Run `pip install -e .` or ensure your PYTHONPATH is set. "
        "If running in CI, make sure the CLI entrypoint is installed."
    )
