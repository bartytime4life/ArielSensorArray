# tests/integration/test_cli_core.py
# SpectraMind V50 — CLI core integration tests
#
# These tests intentionally avoid running any heavy pipeline work. They only exercise the
# public CLI surface (Typer + Click) to ensure commands exist and help/usage renders correctly.
#
# Requirements:
#   - pytest
#   - spectramind package installed (or importable in editable mode)
#
# Reference (design intent):
#   - Typer-based unified CLI with subcommands {calibrate, train, diagnose, submit}
#     and Hydra-backed configuration composition. (See SpectraMind V50 plan.)  # noqa
#     (Commands: calibrate, train, diagnose, submit)  # noqa
#
# If your project exposes the Typer app under a different module path, adjust `CLI_IMPORTS` below.

from __future__ import annotations

import os
import types
from typing import Iterable, Set

import pytest
from click.testing import CliRunner

# Attempt a few common import locations for the Typer app.
CLI_IMPORTS = (
    "spectramind.cli",         # e.g., app = Typer() defined here
    "spectramind.__main__",    # sometimes CLI is exposed here
    "spectramind.app",         # fallback if you export app in a submodule
)


def _load_typer_app():
    """
    Resolve the Typer application object (`app`) from one of the known modules.

    The function raises ImportError if the CLI cannot be located, which will
    surface a clear failure explaining why tests could not run.
    """
    last_err: Exception | None = None
    for mod_path in CLI_IMPORTS:
        try:
            mod = __import__(mod_path, fromlist=["*"])
        except Exception as e:  # pragma: no cover - import resolution path
            last_err = e
            continue
        # Heuristic: look for `app` first, then any Typer-like attribute that has `.to_click()`
        candidate = getattr(mod, "app", None)
        if candidate is None:
            # scan module for a Typer-like object
            for name in dir(mod):
                obj = getattr(mod, name)
                if hasattr(obj, "to_click") and isinstance(getattr(obj, "to_click"), types.BuiltinFunctionType) or callable(getattr(obj, "to_click", None)):
                    candidate = obj
                    break
        if candidate is not None and callable(getattr(candidate, "to_click", None)):
            return candidate
    raise ImportError(
        f"Could not locate a Typer app `app` in any of {CLI_IMPORTS}. "
        f"Last import error: {last_err}"
    )


@pytest.fixture(scope="session")
def typer_app():
    # Ensure non-interactive / reproducible CLI behavior for tests
    # (These envs are harmless if the app ignores them.)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    return _load_typer_app()


@pytest.fixture()
def runner():
    return CliRunner(mix_stderr=False)


def _extract_command_names(app) -> Set[str]:
    """
    Introspect Typer/Click to list registered subcommand names.
    Supports Typer ≥0.6 via app.to_click() returning a click.Group.
    """
    click_cmd = app.to_click()
    if hasattr(click_cmd, "commands"):
        return set(click_cmd.commands.keys())
    # Fallback: iterate over subcommands if available
    names: Set[str] = set()
    for attr in ("list_commands", "commands"):
        if hasattr(click_cmd, attr):
            try:
                maybe = getattr(click_cmd, attr)
                if callable(maybe):
                    names.update(maybe(None) if maybe.__code__.co_argcount else maybe())
                else:
                    names.update(list(maybe.keys()) if hasattr(maybe, "keys") else list(maybe))
            except Exception:  # pragma: no cover
                pass
    return names


# --- Top-level CLI behaviour -------------------------------------------------


def test_cli_top_level_help_shows_commands(typer_app, runner):
    """Top-level `--help` should succeed and list the core subcommands."""
    result = runner.invoke(typer_app, ["--help"])
    assert result.exit_code == 0, result.output
    # Basic sanity checks on help text
    assert "Usage:" in result.output
    assert "Options:" in result.output
    # Expect to see these subcommand names in the help body
    for sub in ("calibrate", "train", "diagnose", "submit"):
        assert sub in result.output, f"Expected `{sub}` to appear in CLI help."


def test_cli_invoked_without_args_shows_usage(typer_app, runner):
    """
    Invoking the CLI with no args should not crash.
    Some Typer/Click apps exit with code 0 and print help, others use code 2.
    We accept both as long as 'Usage:' appears and no traceback is present.
    """
    result = runner.invoke(typer_app, [])
    assert result.exit_code in (0, 2), f"Unexpected exit code {result.exit_code}\n{result.output}"
    assert "Usage:" in result.output
    assert "Traceback" not in result.output


def test_cli_registers_expected_subcommands(typer_app):
    """Ensure the four core subcommands are actually registered with Click."""
    names = _extract_command_names(typer_app)
    missing = {"calibrate", "train", "diagnose", "submit"} - names
    assert not missing, f"CLI is missing expected commands: {sorted(missing)}"


# --- Subcommand help smoke tests --------------------------------------------

@pytest.mark.parametrize("subcmd", ["calibrate", "train", "diagnose", "submit"])
def test_each_subcommand_help_renders(subcmd, typer_app, runner):
    """Each core subcommand should provide `--help` and exit cleanly."""
    result = runner.invoke(typer_app, [subcmd, "--help"])
    assert result.exit_code == 0, f"{subcmd} --help failed:\n{result.output}"
    assert "Usage:" in result.output
    assert "Options:" in result.output
    # A tiny UX check: the subcommand name should be visible in the help header
    assert subcmd in result.output


# --- Minimal UX/robustness guards -------------------------------------------

def test_unknown_command_produces_click_error(typer_app, runner):
    """Unknown subcommands should produce a helpful Click error message."""
    result = runner.invoke(typer_app, ["frobnicate", "--help"])
    assert result.exit_code != 0
    # Click typically says: "Error: No such command 'frobnicate'."
    assert "No such command" in result.output or "Error" in result.output


def test_version_option_if_present_does_not_crash(typer_app, runner):
    """
    If the CLI exposes a --version option (optional), it should not crash.
    We don't assert the exact version text (to avoid coupling to packaging).
    """
    result = runner.invoke(typer_app, ["--version"])
    if result.exit_code == 0:
        # If supported, ensure it printed something version-like.
        assert result.output.strip(), "Expected some version output."
    else:
        # It's fine if the app does not implement --version; just ensure it didn't traceback.
        assert "Traceback" not in result.output
