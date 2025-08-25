# tests/conftest.py
"""
Global pytest fixtures & test wiring for SpectraMind V50.

Design goals
------------
- Deterministic tests (fixed seeds across numpy / Python / PyTorch).
- Hermetic runs (no unwanted net / GUI / non‑deterministic backends).
- Fast dev loop (mark/skip slow by default; easy opt-in with --runslow).
- Helpful utilities (temp project dir, sample spectrum tensors, CLI runners).

All fixtures degrade gracefully if optional deps (torch, hydra, typer) are absent.
"""

from __future__ import annotations

import os
import random
import sys
import subprocess
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Global, once-per-session test configuration
# -----------------------------------------------------------------------------

# Headless rendering for any matplotlib usage
os.environ.setdefault("MPLBACKEND", "Agg")

# Silence some noisy tooling for CI speed/stability.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
os.environ.setdefault("PIP_NO_PYTHON_VERSION_WARNING", "1")

# Encourage deterministic backends (when present)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# Pin BLAS threads for reproducibility on shared runners
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Helpful in CI for Hydra tracebacks
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# Prefer offline behavior during tests (opt-out per-test if needed)
os.environ.setdefault("KAGGLE_OFFLINE", "1")

# -----------------------------------------------------------------------------
# Seeding helpers
# -----------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        # Optional dependency; fine if unavailable
        pass


@pytest.fixture(scope="session", autouse=True)
def _session_determinism() -> None:
    """Force determinism for the entire test session."""
    seed = int(os.environ.get("PYTEST_GLOBAL_SEED", "42"))
    _seed_all(seed)


@pytest.fixture(autouse=True)
def _function_rng(request: pytest.FixtureRequest) -> Generator[np.random.Generator, None, None]:
    """
    Per-test NumPy Generator seeded from global seed + nodeid hash for independence.
    """
    base = int(os.environ.get("PYTEST_GLOBAL_SEED", "42"))
    node_hash = hash(request.node.nodeid) & 0xFFFFFFFF
    rng = np.random.default_rng(base ^ node_hash)
    yield rng

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
    parser.addoption(
        "--cli",
        action="store",
        default="spectramind",
        help='Subprocess CLI entry to invoke (e.g., "spectramind" or "python -m spectramind").',
    )
    parser.addoption(
        "--no-network",
        action="store_true",
        default=False,
        help="Disallow network access during tests (export SPECTRAMIND_NO_NETWORK=1).",
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
# Logging: ensure INFO is visible on failures
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _caplog_level(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")

# -----------------------------------------------------------------------------
# Paths & temp project layout
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    # Resolve repo root by walking up until pyproject or .git
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur.parent if cur.name == "tests" else cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(__file__).resolve().parent.parent

@pytest.fixture()
def tmp_project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Create a throwaway 'project root' with expected subdirs so code that writes
    into logs/, outputs/, artifacts/ works in tests without touching the repo.
    """
    for sub in ("logs", "outputs", "artifacts", "data"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    # Point project-level envs to the temp dir if code relies on them.
    monkeypatch.setenv("SPECTRAMIND_HOME", str(tmp_path))
    if os.environ.get("HYDRA_FULL_ERROR") != "1":
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    if os.environ.get("KAGGLE_OFFLINE") != "1":
        monkeypatch.setenv("KAGGLE_OFFLINE", "1")

    # Ensure temporary project is importable if tests dynamically import modules there
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    return tmp_path

# -----------------------------------------------------------------------------
# CLI helpers (Typer & subprocess)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cli_app() -> Optional["typer.Typer"]:
    """
    Provide the Typer app if the SpectraMind CLI is importable.
    Returns None if Typer or the app is not available so tests can xfail/skip.
    """
    try:
        import typer  # type: ignore
        from spectramind import cli  # src/spectramind/cli.py with `app = Typer(...)`
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


@pytest.fixture()
def cli_subprocess(pytestconfig: pytest.Config):
    """
    Subprocess fallback/alternative CLI runner.

    Usage:
        out = cli_subprocess("diagnose --help")
    """
    entry = pytestconfig.getoption("--cli")

    def _run(args: str, cwd: Optional[Path] = None) -> str:
        if isinstance(entry, str) and entry.strip().startswith("python -m"):
            cmd = entry.split() + args.split()
        else:
            cmd = [entry] + args.split()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"CLI failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc.stdout

    return _run

# -----------------------------------------------------------------------------
# Useful data fixtures
# -----------------------------------------------------------------------------

@pytest.fixture()
def sample_wavelengths() -> np.ndarray:
    """
    Canonical 283-length wavelength grid (float64) spanning 0.5–5.0 microns.
    Adjust bounds to match the pipeline if needed.
    """
    return np.linspace(0.5, 5.0, 283, dtype=np.float64)


@pytest.fixture()
def sample_spectrum(sample_wavelengths: np.ndarray, _function_rng) -> np.ndarray:
    """
    Toy transmission spectrum: smooth baseline plus a few Gaussian absorption features.
    Non-negative and bounded in [0, 1].
    """
    wl = sample_wavelengths
    baseline = 0.02 + 0.005 * np.sin(2 * np.pi * wl / wl.max())

    def gauss(mu: float, sigma: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((wl - mu) / sigma) ** 2)

    spectrum = baseline.copy()
    spectrum += gauss(mu=1.4, sigma=0.04, amp=0.015)
    spectrum += gauss(mu=2.7, sigma=0.06, amp=0.010)
    spectrum += gauss(mu=4.3, sigma=0.08, amp=0.008)

    spectrum += _function_rng.normal(0.0, 5e-5, size=spectrum.shape)
    return np.clip(spectrum, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Optional: Hypothesis profile for fast & stable property tests
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
    pass

# -----------------------------------------------------------------------------
# Optional Hydra isolation (if tests compose configs)
# -----------------------------------------------------------------------------

@pytest.fixture()
def hydra_clear_global_state() -> Generator[None, None, None]:
    """
    Ensure Hydra's global state does not bleed between tests that call compose().
    Only activates if Hydra is present in the environment.
    """
    try:
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        yield
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        yield

# -----------------------------------------------------------------------------
# Optional session-wide network guard
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _network_guard(pytestconfig: pytest.Config) -> None:
    if pytestconfig.getoption("--no-network"):
        os.environ["SPECTRAMIND_NO_NETWORK"] = "1"