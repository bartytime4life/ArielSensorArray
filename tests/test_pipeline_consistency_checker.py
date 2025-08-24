# tests/test_pipeline_consistency_checker.py
# -----------------------------------------------------------------------------
# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
#
# Upgraded tests for the pipeline_consistency_checker module.
#
# Design goals
#  • Dynamic, API-flexible discovery of functions (run_checks, generate_report, etc.)
#  • Spin up a tiny, synthetic repo skeleton in a temp dir (configs/, dvc.yaml,
#    src/cli/, logs/, outputs/) so the checker can parse realistic artifacts
#  • Validate that high-level status/sections are returned and that common
#    sub-checks (CLI registration, config scan, DVC stages, logs/hash parsing,
#    symbolic module presence) behave sanely
#  • Gracefully skip when a feature isn’t implemented in the target module
#
# Supported import paths:
#   - src.utils.pipeline_consistency_checker
#   - utils.pipeline_consistency_checker
#   - pipeline_consistency_checker
#
# NOTE: These tests are defensive to accommodate minor differences between
#       implementations. Assertions prefer semantic checks over brittle
#       exact-structure matches, and will skip with a clear message if an
#       optional feature is missing.
# -----------------------------------------------------------------------------

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pytest
import yaml


# -----------------------------------------------------------------------------
# Dynamic import of the checker module
# -----------------------------------------------------------------------------
_CANDIDATE_IMPORTS = [
    "src.utils.pipeline_consistency_checker",
    "utils.pipeline_consistency_checker",
    "pipeline_consistency_checker",
]

checker = None
for _mod in _CANDIDATE_IMPORTS:
    try:
        checker = importlib.import_module(_mod)
        break
    except ModuleNotFoundError:
        continue

if checker is None:
    pytest.skip(
        "pipeline_consistency_checker module not found in expected locations: "
        + ", ".join(_CANDIDATE_IMPORTS),
        allow_module_level=True,
    )


# -----------------------------------------------------------------------------
# Helpers to discover optional functions
# -----------------------------------------------------------------------------
def _get_fn(names: Iterable[str]):
    for n in names:
        f = getattr(checker, n, None)
        if callable(f):
            return f, n
    return None, None


def _get_run_checks():
    return _get_fn(("run_checks", "run_all_checks", "check_all"))


def _get_generate_report():
    return _get_fn(("generate_report", "write_report", "save_report"))


def _get_check_cli():
    return _get_fn(("check_cli_registration", "check_cli", "validate_cli"))


def _get_check_configs():
    return _get_fn(("check_config_files", "check_configs", "scan_configs"))


def _get_check_dvc():
    return _get_fn(("check_dvc_stages", "check_dvc", "scan_dvc_yaml"))


def _get_check_logs():
    return _get_fn(("check_logging_outputs", "check_logs", "scan_logs"))


def _get_check_symbolic():
    return _get_fn(("check_symbolic_modules", "check_symbolics", "validate_symbolic_modules"))


def _get_main():
    return _get_fn(("main",))


# -----------------------------------------------------------------------------
# Synthetic repo skeleton (temp dir)
# -----------------------------------------------------------------------------
@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """
    Create a tiny repo skeleton the checker can scan without hitting the network.

    Layout:
      repo/
        configs/
          config_v50.yaml
          model/v50.yaml
          data/nominal.yaml
        src/cli/spectramind.py            (Typer-style placeholder or simple text)
        src/symbolic/symbolic_loss.py     (required module marker)
        src/symbolic/molecular_priors.py  (required module marker)
        dvc.yaml                          (two stages with outs)
        logs/v50_debug_log.md             (contains sample config hash)
        outputs/run_hash_summary_v50.json (contains sample config hash + env)
    """
    root = tmp_path / "repo"
    (root / "configs" / "model").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "data").mkdir(parents=True, exist_ok=True)
    (root / "src" / "cli").mkdir(parents=True, exist_ok=True)
    (root / "src" / "symbolic").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)

    # Minimal YAML configs
    (root / "configs" / "config_v50.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": [
                    {"model": "v50"},
                    {"data": "nominal"},
                ],
                "training": {"epochs": 1, "seed": 1234},
                "diagnostics": {"dashboard": True},
            }
        ),
        encoding="utf-8",
    )
    (root / "configs" / "model" / "v50.yaml").write_text(
        yaml.safe_dump({"name": "v50", "fgs1": {"embed_dim": 64}, "airs": {"hidden_dim": 128}}),
        encoding="utf-8",
    )
    (root / "configs" / "data" / "nominal.yaml").write_text(
        yaml.safe_dump({"dataset": "ariel_sim", "split": "val"}),
        encoding="utf-8",
    )

    # CLI placeholder (Typer-like signature string to be greppable by checkers)
    (root / "src" / "cli" / "spectramind.py").write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "# CLI placeholder with subcommands: train, diagnose, submit, test\n"
        "COMMANDS = ['train', 'diagnose', 'submit', 'test']\n",
        encoding="utf-8",
    )

    # Symbolic module presence markers
    (root / "src" / "symbolic" / "symbolic_loss.py").write_text(
        "class SymbolicLoss: pass\n", encoding="utf-8"
    )
    (root / "src" / "symbolic" / "molecular_priors.py").write_text(
        "def make_template_spectrum(*args, **kwargs):\n    return []\n", encoding="utf-8"
    )

    # dvc.yaml with a couple stages
    (root / "dvc.yaml").write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "calibrate": {"cmd": "python -m pipeline.calibrate", "outs": ["outputs/calib.h5"]},
                    "train": {"cmd": "python -m pipeline.train", "deps": ["outputs/calib.h5"], "outs": ["outputs/model.pt"]},
                }
            }
        ),
        encoding="utf-8",
    )

    # Logs: include a config hash line + a couple entries
    (root / "logs" / "v50_debug_log.md").write_text(
        "2025-08-20 10:00:00 [INFO] CLI version=1.0.0 config_hash=deadbeefcafebabefeed1234567890aa\n"
        "2025-08-20 10:00:01 [INFO] train epochs=1 seed=1234\n",
        encoding="utf-8",
    )

    # outputs: run-hash summary JSON
    (root / "outputs" / "run_hash_summary_v50.json").write_text(
        json.dumps(
            {
                "config_hash": "deadbeefcafebabefeed1234567890aa",
                "env": {"python_version": "3.11.x", "platform": "linux"},
                "timestamp": "2025-08-20T10:00:05Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return root


# -----------------------------------------------------------------------------
# Tests: module exports & run_checks
# -----------------------------------------------------------------------------
def test_module_exports_expected_symbols():
    # Plurality of implementations: verify at least one of the main entrypoints exists
    run_checks, _ = _get_run_checks()
    main, _ = _get_main()
    assert run_checks or main, "Expected at least one entry point (run_checks/main) in checker module"


def test_run_checks_returns_struct_with_status(repo: Path):
    run_checks, _ = _get_run_checks()
    if run_checks is None:
        pytest.skip("run_checks / run_all_checks not implemented—skipping.")
    res = run_checks(repo_root=str(repo))
    # Allow dict-like or tuple (status, details)
    if isinstance(res, tuple) and len(res) >= 1:
        status = res[0]
        details = res[1] if len(res) > 1 else {}
    elif isinstance(res, dict):
        status = res.get("status", True)
        details = res
    else:
        pytest.skip(f"run_checks returned unsupported type: {type(res)}")
    assert isinstance(status, (bool, int))
    # Check presence of at least some sections
    joined = json.dumps(details).lower()
    assert any(k in joined for k in ("cli", "config", "dvc", "log", "symbolic")), "Missing expected sections in results"


# -----------------------------------------------------------------------------
# Tests: specific sub-checks (optional)
# -----------------------------------------------------------------------------
def test_cli_registration_check_detects_missing_command(repo: Path):
    check_cli, _ = _get_check_cli()
    if check_cli is None:
        pytest.skip("check_cli_registration not implemented—skipping.")

    # Initial should pass
    ok0 = check_cli(repo_root=str(repo))
    assert isinstance(ok0, (bool, dict))

    # Remove 'diagnose' from CLI placeholder and expect a warning/fail
    cli_file = repo / "src" / "cli" / "spectramind.py"
    txt = cli_file.read_text(encoding="utf-8")
    txt = txt.replace(" 'diagnose',", " ")
    cli_file.write_text(txt, encoding="utf-8")

    out = check_cli(repo_root=str(repo))
    # Accept either a bool status or a dict with 'status'/messages
    if isinstance(out, bool):
        assert out in (False, 0), "Expected CLI check to flag missing 'diagnose' command."
    elif isinstance(out, dict):
        st = out.get("status", True)
        msg = json.dumps(out).lower()
        assert (not st) or ("diagnose" in msg and "missing" in msg or "not found" in msg), \
            "Expected CLI check to report 'diagnose' missing."
    else:
        pytest.skip("check_cli_registration returned unsupported type.")


def test_config_scan_finds_core_files(repo: Path):
    check_cfg, _ = _get_check_configs()
    if check_cfg is None:
        pytest.skip("check_config_files not implemented—skipping.")
    res = check_cfg(repo_root=str(repo))
    # Allow bool or dict with file lists
    assert isinstance(res, (bool, dict))
    if isinstance(res, dict):
        joined = json.dumps(res).lower()
        assert "config_v50.yaml" in joined and "model" in joined and "data" in joined


def test_dvc_stage_scan_parses_stages(repo: Path):
    check_dvc, _ = _get_check_dvc()
    if check_dvc is None:
        pytest.skip("check_dvc_stages not implemented—skipping.")
    res = check_dvc(repo_root=str(repo))
    assert isinstance(res, (bool, dict))
    if isinstance(res, dict):
        j = json.dumps(res).lower()
        assert "calibrate" in j and "train" in j, "Expected at least 'calibrate' and 'train' stages detected."


def test_log_scan_extracts_config_hash(repo: Path):
    check_logs, _ = _get_check_logs()
    if check_logs is None:
        pytest.skip("check_logging_outputs not implemented—skipping.")
    res = check_logs(repo_root=str(repo))
    assert isinstance(res, (bool, dict))
    if isinstance(res, dict):
        j = json.dumps(res).lower()
        assert "config_hash" in j and "deadbeef" in j, "Expected to recover config hash from debug log."


def test_symbolic_module_presence(repo: Path):
    check_symb, _ = _get_check_symbolic()
    if check_symb is None:
        pytest.skip("check_symbolic_modules not implemented—skipping.")
    res = check_symb(repo_root=str(repo))
    assert isinstance(res, (bool, dict))
    if isinstance(res, dict):
        j = json.dumps(res).lower()
        assert "symbolic_loss" in j and "molecular_priors" in j


# -----------------------------------------------------------------------------
# Tests: report generation (optional)
# -----------------------------------------------------------------------------
def test_generate_report_produces_md_and_json(repo: Path, tmp_path: Path):
    gen_report, _ = _get_generate_report()
    run_checks, _ = _get_run_checks()
    if gen_report is None or run_checks is None:
        pytest.skip("generate_report/run_checks not implemented—skipping.")

    results = run_checks(repo_root=str(repo))
    out_md = tmp_path / "consistency_report.md"
    out_json = tmp_path / "consistency_report.json"

    # Flexible call signatures
    try:
        gen_report(results, md_path=str(out_md), json_path=str(out_json))
    except TypeError:
        try:
            gen_report(results, md_path=out_md, json_path=out_json)
        except TypeError:
            gen_report(results, output_dir=str(tmp_path))

    # Check outputs exist
    md_exists = out_md.exists() or any(p.suffix == ".md" for p in tmp_path.glob("*.md"))
    json_exists = out_json.exists() or any(p.suffix == ".json" for p in tmp_path.glob("*.json"))
    assert md_exists and json_exists, "Expected report writer to produce .md and .json outputs."


# -----------------------------------------------------------------------------
# Tests: exit code semantics via main (optional)
# -----------------------------------------------------------------------------
def test_main_exit_code_reflects_failure(repo: Path, monkeypatch):
    main, _ = _get_main()
    if main is None:
        pytest.skip("main() not implemented—skipping.")

    # Break a core artifact to induce failure (remove dvc.yaml)
    (repo / "dvc.yaml").unlink(missing_ok=True)

    # Provide argv-like args if the CLI expects them; otherwise call with kwargs
    try:
        rc = main(["--repo-root", str(repo), "--no-color"])
    except TypeError:
        rc = main(repo_root=str(repo))

    assert isinstance(rc, (int, bool))
    assert rc not in (0, True), "Expected non-zero/False exit when core artifact is missing."
