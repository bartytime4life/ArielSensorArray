# tests/conftest.py
"""
Global pytest fixtures & test wiring for SpectraMind V50.

Design goals
------------
- Deterministic tests (fixed seeds across numpy / Python / PyTorch).
- Hermetic runs (no unwanted net / GUI / non‑deterministic backends).
- Fast dev loop (mark/skip slow by default; easy opt-in with --runslow).
- Helpful utilities (temp project dir, sample spectrum tensors, CLI runner).

All fixtures are safe to import even if optional deps (torch, hydra, typer)
are not installed — they degrade gracefully.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Global, once-per-session test configuration
# -----------------------------------------------------------------------------

# Make matplotlib fully headless in any test (if present).
os.environ.setdefault("MPLBACKEND", "Agg")

# Silence some noisy tooling for CI speed/stability.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
os.environ.setdefault("PIP_NO_PYTHON_VERSION_WARNING", "1")

# Optional: encourage libraries to act deterministically when they can.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")       # if TF present
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # cudnn/cublas (if used)

# Prefer reproducible thread counts if libs respect it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    try:  # optional dependency
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Ensure CuDNN is deterministic where possible
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        # It's okay if torch isn't available during docs/lint jobs
        pass


@pytest.fixture(scope="session", autouse=True)
def _session_determinism() -> None:
    """Force determinism for the entire test session."""
    seed = int(os.environ.get("PYTEST_GLOBAL_SEED", "42"))
    _seed_all(seed)


@pytest.fixture(autouse=True)
def _function_rng() -> Generator[np.random.Generator, None, None]:
    """
    A per-test NumPy Generator seeded from the global seed + nodeid hash.
    This keeps tests independent yet reproducible.
    """
    # The calling test nodeid is not available here directly; use a stable fallback.
    # For better isolation, individual tests can pass their own seed.
    seed = int(os.environ.get("PYTEST_GLOBAL_SEED", "42")) ^ (hash(os.getcwd()) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    yield rng


# -----------------------------------------------------------------------------
# CLI helpers (optional Typer integration)
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def cli_app() -> Optional["typer.Typer"]:
    """
    Provide the Typer app if the SpectraMind CLI is importable.
    Returns None if Typer or the app is not available so tests can xfail/skip.
    """
    try:
        import typer  # type: ignore
        from spectramind import cli  # project: src/spectramind/cli.py with `app = Typer(...)`

        if hasattr(cli, "app"):
            return cli.app  # type: ignore
    except Exception:
        pass
    return None


@pytest.fixture()
def cli_runner(cli_app) -> "typer.testing.CliRunner":
    """
    Typer CliRunner tied to the SpectraMind app. If the app is missing, tests
    that require it should call `pytest.skip(...)`.
    """
    try:
        from typer.testing import CliRunner  # type: ignore
    except Exception as e:
        pytest.skip(f"typer.testing.CliRunner not available: {e}")

    if cli_app is None:
        pytest.skip("SpectraMind CLI app not importable.")
    return CliRunner()


# -----------------------------------------------------------------------------
# PyTest knobs: marks, options, and slow test handling
# -----------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: tests requiring CUDA/GPU")
    config.addinivalue_line("markers", "integration: full pipeline or external resources")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# -----------------------------------------------------------------------------
# Logging, temp project dir, and handy data fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _caplog_level(caplog: pytest.LogCaptureFixture) -> None:
    # Ensure useful INFO logs appear during failing tests
    caplog.set_level("INFO")


@pytest.fixture()
def tmp_project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Create a throwaway 'project root' with expected subdirs so code that writes
    into logs/, outputs/, artifacts/ works in tests without touching the repo.
    """
    for sub in ("logs", "outputs", "artifacts", "data"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    # Point any project-level envs to the temp dir if code relies on them.
    monkeypatch.setenv("SPECTRAMIND_HOME", str(tmp_path))
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")  # helpful in CI
    # Avoid accidental online calls during tests (opt-out if needed per-test)
    monkeypatch.setenv("KAGGLE_OFFLINE", "1")

    # Add tmp project to sys.path for dynamic module discovery (optional)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    return tmp_path


@pytest.fixture()
def sample_wavelengths() -> np.ndarray:
    """
    A canonical 283-length wavelength grid (float64) spanning 0.5–5.0 microns.
    Adjust bounds to match your pipeline if needed.
    """
    return np.linspace(0.5, 5.0, 283, dtype=np.float64)


@pytest.fixture()
def sample_spectrum(sample_wavelengths: np.ndarray, _function_rng) -> np.ndarray:
    """
    A physically-plausible toy transmission spectrum: smooth baseline plus
    a few Gaussian absorption features. Non-negative and bounded in [0, 1].
    """
    wl = sample_wavelengths
    baseline = 0.02 + 0.005 * np.sin(2 * np.pi * wl / wl.max())

    def gauss(mu: float, sigma: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((wl - mu) / sigma) ** 2)

    # Mix a few lines; parameters chosen to be gentle.
    spectrum = baseline.copy()
    spectrum += gauss(mu=1.4, sigma=0.04, amp=0.015)
    spectrum += gauss(mu=2.7, sigma=0.06, amp=0.010)
    spectrum += gauss(mu=4.3, sigma=0.08, amp=0.008)

    # Add tiny noise to avoid pathological exact-equality assertions
    spectrum += _function_rng.normal(0.0, 5e-5, size=spectrum.shape)

    # Clamp to physically meaningful range
    return np.clip(spectrum, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Optional: Hypothesis defaults (if installed) for fast & stable property tests
# -----------------------------------------------------------------------------
try:
    from hypothesis import HealthCheck, Phase, settings

    settings.register_profile(
        "spectramind_fast",
        deadline=None,
        max_examples=100,
        suppress_health_check=(HealthCheck.too_slow,),
        phases=(Phase.generate, Phase.target),
        print_blob=True,
    )
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "spectramind_fast"))
except Exception:
    # Hypothesis not installed — fine for CI shards where not needed.
    pass


# -----------------------------------------------------------------------------
# Optional Hydra isolation (if your tests compose configs)
# -----------------------------------------------------------------------------
@pytest.fixture()
def hydra_clear_global_state() -> Generator[None, None, None]:
    """
    Ensure Hydra's global state does not bleed between tests that call compose().
    Only activates if Hydra is present in the environment.
    """
    try:
        from hydra import initialize, compose  # noqa: F401
        from hydra.core.global_hydra import GlobalHydra

        # Before: ensure no prior state
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        yield
        # After: clear again
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        # If hydra isn't installed or used, do nothing
        yield
