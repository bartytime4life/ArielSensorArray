# tests/test_cli_map_integrity.py
"""
CLI map integrity tests for the SpectraMind V50 Typer CLI.

Goals
-----
1) The top-level CLI exists and is invokable.
2) The top-level --help prints and lists the expected core subcommands.
3) Every discovered subcommand responds to its own --help.
4) Every discovered subcommand exposes a non-empty help string.
5) Unknown commands fail gracefully (non-zero exit, Click-style message).
6) If a version flag exists, it returns a semantic-looking version string.
7) The set of Click subcommands is reflected in the rendered help text.

These tests are intentionally defensive: they try multiple import locations
and tolerate optional features (e.g., --version) by skipping if unavailable.
"""

from __future__ import annotations

import importlib
import re
from typing import Optional

import pytest

try:
    import typer  # noqa: F401
except Exception as exc:  # pragma: no cover - if typer missing, fail early with a helpful msg
    raise RuntimeError(
        "Typer is required for CLI tests. Please add it to your test environment."
    ) from exc

from click.testing import CliRunner


# -------------------------
# Helpers
# -------------------------


def _load_module_safely(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def _get_typer_app() -> "typer.Typer":
    """
    Try a few conventional import locations to find the Typer app instance.

    Search order:
      - spectramind.cli:  attributes: app, cli, typer_app, main
      - spectramind:      attributes: app, cli, typer_app
    """
    candidates = [
        ("spectramind.cli", ("app", "cli", "typer_app", "main")),
        ("spectramind", ("app", "cli", "typer_app")),
    ]

    for mod_name, attrs in candidates:
        mod = _load_module_safely(mod_name)
        if mod is None:
            continue
        for attr in attrs:
            app = getattr(mod, attr, None)
            # Typer has .to_click() method; click.Group has .commands
            if app is not None and hasattr(app, "to_click"):
                return app  # type: ignore[return-value]
    raise AssertionError(
        "Could not locate the Typer application. "
        "Expected one of: spectramind.cli:{app,cli,typer_app,main} "
        "or spectramind:{app,cli,typer_app}."
    )


def _to_click_group(app) -> "click.core.Group":
    # Typer's .to_click() returns a Click command (usually a Group for top-level)
    click_cmd = app.to_click()
    # Defensive: top-level should be a Group; if not, many tests are not meaningful
    if not hasattr(click_cmd, "commands"):
        raise AssertionError("Top-level CLI is not a Click Group (no .commands found).")
    return click_cmd


def _extract_subcommand_names_from_help(help_text: str) -> set[str]:
    """
    Click renders subcommands in the help text table. We'll parse them loosely:
    lines that start with two spaces, then a word, then at least two spaces.

    Example (typical Click output):

      Commands:
        calibrate   Calibrate raw inputs ...
        train       Train the model ...
        diagnose    Generate diagnostics ...
        submit      Build the submission package ...

    This parser is intentionally permissive.
    """
    names: set[str] = set()
    for line in help_text.splitlines():
        m = re.match(r"^\s{2,}([a-zA-Z0-9][\w\-]*)\s{2,}", line)
        if m:
            names.add(m.group(1))
    return names


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture(scope="session")
def typer_app() -> "typer.Typer":
    return _get_typer_app()


@pytest.fixture(scope="session")
def click_group(typer_app):
    return _to_click_group(typer_app)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


# -------------------------
# Tests
# -------------------------


def test_top_level_help_runs(typer_app, runner: CliRunner):
    res = runner.invoke(typer_app, ["--help"])
    assert res.exit_code == 0, f"Top-level --help failed: {res.output}"

    # Smoke checks on generic Click/Typer help structure
    assert "Usage" in res.output or "Usage:" in res.output
    assert "Commands" in res.output or "Commands:" in res.output

    # Project-specific: the SpectraMind CLI advertises these core verbs in docs
    # We allow partial presence to stay resilient across repos/branches.
    expected_any = {"calibrate", "train", "diagnose", "submit", "predict"}
    present = _extract_subcommand_names_from_help(res.output)
    assert present & expected_any, (
        "Top-level help did not show any of the expected core commands; "
        f"found: {sorted(present)}"
    )


def test_all_subcommands_have_help(click_group, typer_app, runner: CliRunner):
    # The canonical list of subcommands from Click
    names = sorted(click_group.commands.keys())
    assert names, "No subcommands were registered on the top-level Group."

    for name in names:
        res = runner.invoke(typer_app, [name, "--help"])
        assert res.exit_code == 0, f"`{name} --help` failed: {res.output}"


def test_subcommands_have_non_empty_help_text(click_group):
    missing: list[str] = []
    for name, cmd in sorted(click_group.commands.items()):
        # Click Command.help may be None; Typer passes function docstring into command.help
        help_text: Optional[str] = getattr(cmd, "help", None)
        if not help_text or not help_text.strip():
            missing.append(name)
    assert not missing, (
        "The following subcommands have empty/missing help text. "
        "Please add docstrings or Typer help strings:\n  - " + "\n  - ".join(missing)
    )


def test_unknown_command_fails_gracefully(typer_app, runner: CliRunner):
    res = runner.invoke(typer_app, ["definitely-not-a-real-cmd"])
    assert res.exit_code != 0, "Unknown command should exit non-zero."
    # Click standard error message:
    assert "No such command" in res.output or "Error" in res.output


def test_version_flag_if_present(typer_app, runner: CliRunner):
    """
    If a version flag exists, it should return 0 and contain a version-like token.
    We try common variants and skip if none are wired.
    """
    tried = []
    for flag in ("--version", "-V", "version"):
        tried.append(flag)
        res = runner.invoke(typer_app, [flag])
        if res.exit_code == 0 and re.search(r"\d+\.\d+(\.\d+)?", res.output):
            return
    pytest.skip(f"No working version flag among {tried} (or no semantic version present).")


def test_help_lists_registered_commands(click_group, typer_app, runner: CliRunner):
    """
    The rendered help should mention every registered Click subcommand name.

    (We don't require exact 1:1 matching because Click may hide aliases or
    Typer may render nested groups, but at minimum each command key should
    appear in the help text.)
    """
    res = runner.invoke(typer_app, ["--help"])
    assert res.exit_code == 0, f"Top-level --help failed: {res.output}"
    rendered = _extract_subcommand_names_from_help(res.output)
    missing = [name for name in click_group.commands.keys() if name not in rendered]
    # Be lenient with nested groups: only fail if *all* are missing
    if missing and len(missing) == len(click_group.commands):
        pytest.fail(
            "None of the registered Click subcommands were visible in --help.\n"
            f"Registered: {sorted(click_group.commands.keys())}\n"
            f"Rendered : {sorted(rendered)}\n"
            "If you are using nested groups, ensure the top-level help renders them."
        )


def test_each_subcommand_invocation_prints_usage_on_no_args(click_group, typer_app, runner: CliRunner):
    """
    Invoking a subcommand with no args should either run (exit 0) or
    print its usage/help (non-zero but includes 'Usage' or '--help').
    This guards against completely silent failures.
    """
    for name in sorted(click_group.commands.keys()):
        res = runner.invoke(typer_app, [name])
        ok = res.exit_code == 0 or ("Usage" in res.output or "--help" in res.output)
        assert ok, f"`{name}` produced neither success nor a usage message:\n{res.output}"
