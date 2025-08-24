# tests/diagnostics/test_shap_overlay.py
"""
SpectraMind V50 — SHAP Overlay Diagnostics tests

What this verifies
------------------
1) CLI smoke: `spectramind diagnose --shap-overlay` creates overlay artifacts (PNG/HTML/JSON)
2) API smoke: a plotting function (if exposed) saves a SHAP overlay for a simple synthetic case
3) Determinism: top-K feature selection from SHAP magnitudes is stable for fixed inputs

Notes
-----
- Tests are skip-friendly: if `spectramind` CLI, `shap`, or your overlay API are missing, tests
  will skip cleanly rather than fail hard.
- This aligns with the plan where `spectramind diagnose` emits SHAP explanations/plots as part of
  the post-training diagnostics pipeline. See the V50 design on diagnostics & SHAP.  # noqa: E501
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:  # pragma: no cover
    matplotlib = None

# Optional SHAP; tests will skip gracefully if not present
try:
    import shap  # type: ignore
    _HAVE_SHAP = True
except Exception:  # pragma: no cover
    _HAVE_SHAP = False


# ---------- helpers ----------

def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _find_artifacts(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(root.rglob(pat))
    return found


def _topk_indices_by_abs(values: np.ndarray, k: int) -> np.ndarray:
    """Return stable top-k by |value| with index tie-breaker (ascending)."""
    assert values.ndim == 1
    k = max(1, min(k, values.size))
    # argsort by (abs, then index) for stability
    order = np.lexsort((np.arange(values.size), np.abs(values)))
    topk = order[-k:]
    return np.sort(topk)  # stable monotonically increasing indices


# ---------- Test 1: CLI smoke ----------

@pytest.mark.slow
def test_cli_generates_shap_overlay(tmp_path: Path):
    """
    Smoke test the CLI path:
      spectramind diagnose --shap-overlay --limit 1 --out <tmp>
    Expectations:
      - non-zero overlay artifacts appear (png/html/json)
      - process exits 0
    Skips if CLI is not installed on PATH.
    """
    spectramind = _which("spectramind")
    if spectramind is None:
        pytest.skip("spectramind CLI not found on PATH")

    out_dir = tmp_path / "diag_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        spectramind,
        "diagnose",
        "--shap-overlay",
        "--limit", "1",          # keep it light & fast
        "--out", str(out_dir),
        "--headless",            # if your CLI supports it; otherwise remove
    ]

    # Let Hydra/CLI print full errors if misconfig happens
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
        check=False,
    )

    # Helpful output on failure
    if proc.returncode != 0:
        print("=== spectramind diagnose output ===")
        print(proc.stdout)

    assert proc.returncode == 0, "diagnose command failed"

    # Look for any SHAP overlay artifacts
    artifacts = _find_artifacts(
        out_dir,
        patterns=(
            "*shap*overlay*.png",
            "*shap*overlay*.html",
            "*shap*overlay*.json",
            "*shap*explain*.png",
            "*shap*explain*.html",
            "*shap*explain*.json",
        ),
    )
    assert artifacts, f"No SHAP overlay artifacts found in {out_dir}"


# ---------- Test 2: API smoke (if overlay function is exposed) ----------

@pytest.mark.parametrize("k", [5, 10])
def test_overlay_api_smoke(tmp_path: Path, k: int):
    """
    If your project exposes a plotting function, exercise it with synthetic data.

    Expected callable (any one of these names):
      - spectramind.diagnostics.shap_overlay.plot_shap_overlay(...)
      - src.diagnostics.shap_overlay.plot_shap_overlay(...)
      - diagnostics.shap_overlay.plot_shap_overlay(...)

    Signature (flexible):
      plot_shap_overlay(
          spectrum: np.ndarray,           # (n_wl,)
          shap_values: np.ndarray,        # (n_wl,)
          wavelengths: np.ndarray|None,   # (n_wl,)
          top_k: int = 10,
          out_path: str|Path = "..."
      ) -> "matplotlib.figure.Figure" | None

    The test will skip if no such function is importable.
    """
    overlay_fn = None
    tried = []

    candidates = [
        "spectramind.diagnostics.shap_overlay",
        "src.diagnostics.shap_overlay",
        "diagnostics.shap_overlay",
        "spectramind_diagnostics.shap_overlay",  # alternate naming, just in case
    ]

    for mod_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=["plot_shap_overlay"])
            if hasattr(mod, "plot_shap_overlay"):
                overlay_fn = getattr(mod, "plot_shap_overlay")
                break
            tried.append(mod_name + ": no plot_shap_overlay")
        except Exception as e:  # pragma: no cover
            tried.append(f"{mod_name}: import failed ({e})")

    if overlay_fn is None:
        pytest.skip("No plot_shap_overlay API found "
                    f"(tried: {', '.join(tried)})")

    # Synthesize a tiny fake spectrum and SHAP values
    rng = np.random.default_rng(42)
    n_wl = 283
    wavelengths = np.linspace(0.8, 5.0, n_wl)  # micron, just for x-axis
    spectrum = (0.02 * np.sin(2 * np.pi * wavelengths)
                + 0.01 * np.cos(5 * np.pi * wavelengths))
    spectrum += 0.001 * rng.standard_normal(n_wl)

    shap_values = 0.001 * rng.standard_normal(n_wl)
    # inject a few obvious "top" contributors
    shap_values[[10, 77, 140, 201, 250]] += np.array([0.03, -0.025, 0.02, -0.018, 0.022])

    out_path = tmp_path / f"shap_overlay_top{k}.png"
    fig = overlay_fn(
        spectrum=spectrum,
        shap_values=shap_values,
        wavelengths=wavelengths,
        top_k=k,
        out_path=out_path,
    )

    assert out_path.exists(), "Overlay plot was not saved"
    # If the function returns a figure, ensure it's a Matplotlib figure and close it
    if fig is not None and matplotlib is not None:
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt  # noqa: F401
        plt.close(fig)


# ---------- Test 3: Determinism of top-K selection ----------

def test_topk_selection_is_deterministic():
    """
    Top-K by absolute SHAP magnitude should be stable for fixed inputs.
    This test uses a small, known vector and verifies the chosen indices.
    """
    vals = np.array([0.0, -0.2, 0.15, 0.15, -0.05, 0.2, -0.2, 0.001], dtype=float)
    # |vals| = [0, .2, .15, .15, .05, .2, .2, .001]
    # top3 by |.| are magnitudes .2 at indices [1, 5, 6]; tie-breaker by index (asc)
    top3 = _topk_indices_by_abs(vals, 3)
    np.testing.assert_array_equal(top3, np.array([1, 5, 6]))


# ---------- (Optional) SHAP end-to-end tiny example (skippable) ----------

@pytest.mark.skipif(not _HAVE_SHAP or matplotlib is None, reason="requires shap and matplotlib")
def test_tiny_shap_pipeline_and_overlay(tmp_path: Path):
    """
    Optional end-to-end: build a tiny linear model, compute SHAP, and plot overlay.

    Skips if shap/matplotlib are unavailable.
    """
    rng = np.random.default_rng(0)
    n_wl = 283
    n_samples = 50

    # Synthetic training data (features ~ simple sinusoids)
    X = np.stack([
        np.sin(np.linspace(0, 2*np.pi, n_wl) + phi)
        for phi in np.linspace(0, np.pi, n_samples)
    ])
    # True weights emphasizing a few regions
    w = np.zeros(n_wl)
    w[[15, 63, 118, 170, 230]] = np.array([0.8, -0.6, 0.7, -0.5, 0.9])
    y = X @ w + 0.05 * rng.standard_normal(n_samples)

    class TinyLinear:
        def __init__(self):
            # closed form ridge (very light)
            lam = 1e-3
            XtX = X.T @ X + lam * np.eye(n_wl)
            self.coef_ = np.linalg.solve(XtX, X.T @ y)
            self.intercept_ = 0.0

        def predict(self, Z):
            return Z @ self.coef_ + self.intercept_

    model = TinyLinear()

    # Pick one example to explain
    x0 = X[3:4, :]
    background = X[:20, :]

    expl = shap.LinearExplainer(model, background)
    shap_vals = expl.shap_values(x0)[0]  # (n_wl,)

    # Try to import the project's overlay function; otherwise make a minimal overlay here
    overlay_fn = None
    for mod_name in [
        "spectramind.diagnostics.shap_overlay",
        "src.diagnostics.shap_overlay",
        "diagnostics.shap_overlay",
    ]:
        try:
            mod = __import__(mod_name, fromlist=["plot_shap_overlay"])
            if hasattr(mod, "plot_shap_overlay"):
                overlay_fn = getattr(mod, "plot_shap_overlay")
                break
        except Exception:  # pragma: no cover
            pass

    out_path = tmp_path / "shap_overlay_e2e.png"
    wavelengths = np.linspace(0.8, 5.0, n_wl)

    if overlay_fn is not None:
        fig = overlay_fn(
            spectrum=x0.ravel(),
            shap_values=shap_vals,
            wavelengths=wavelengths,
            top_k=8,
            out_path=out_path,
        )
        assert out_path.exists()
        if fig is not None:
            import matplotlib.pyplot as plt  # noqa: F401
            plt.close(fig)
    else:
        # Minimal inline overlay (fallback to still validate output)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wavelengths, x0.ravel(), label="spectrum", lw=1.5, c="#1f77b4")
        ax2 = ax.twinx()
        ax2.stem(wavelengths, shap_vals, linefmt="C3-", markerfmt=" ", basefmt=" ")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Transit depth")
        ax2.set_ylabel("SHAP value")
        ax.set_title("SHAP Overlay (fallback)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        assert out_path.exists(), "Fallback overlay not saved"

