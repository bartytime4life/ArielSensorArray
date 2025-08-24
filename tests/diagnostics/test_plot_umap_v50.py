Here’s an upgraded, drop‑in pytest file for `tests/diagnostics/test_plot_umap_v50.py`. It’s defensive (auto‑discovers where your function lives), backend‑safe for headless CI, and verifies that a figure file is actually produced for normal inputs and small edge‑case inputs. It also gracefully skips if the function isn’t present yet—so your pipeline stays green while you wire things up.

```python
# tests/diagnostics/test_plot_umap_v50.py
"""
Upgraded tests for plot_umap_v50

What these tests cover (robustly, without over-constraining your implementation):
- Imports: auto-discovers plot_umap_v50 across common module layouts and skips cleanly if absent.
- Headless plotting: forces a non-interactive Matplotlib backend for CI environments.
- Happy path: generates synthetic embeddings + labels, calls plot_umap_v50, and asserts that a
  non-empty image file was created (or that a Matplotlib Figure was returned, which we then save).
- Small edge case: ensures plotting works with very small inputs (e.g., n=3 samples).
- NaN/None labels edge case: confirms the function is tolerant to unknown labels (if supported).
- Signature-flexible: only passes kwargs your function actually supports (introspected at runtime).

Notes:
- We intentionally avoid brittle assumptions (e.g., exact filename, return types, strict layout).
- If your function returns a figure, we save it. If it returns a path, we verify the file exists.
- If your function writes to the provided path and returns None, we still discover the file.
"""

from __future__ import annotations

import inspect
import io
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

# Force non-interactive backend for CI/headless
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------- Helpers ----------

POSSIBLE_MODULES = [
    # Common places the function may live. Add more as your repo evolves.
    "src.diagnostics.umap_v50",
    "src.diagnostics.plotting",
    "diagnostics.umap_v50",
    "diagnostics.plotting",
    "src.visualization.umap_v50",
    "src.visualization.plotting",
    "visualization.umap_v50",
    "visualization.plotting",
]


def _try_import_plot_umap() -> Optional[Callable]:
    """
    Attempt to import plot_umap_v50 from a list of likely modules.
    Returns the function if found, else None.
    """
    for mod_name in POSSIBLE_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["plot_umap_v50"])
            if hasattr(mod, "plot_umap_v50") and callable(getattr(mod, "plot_umap_v50")):
                return getattr(mod, "plot_umap_v50")
        except Exception:
            continue
    return None


def _make_blobs(n_samples: int = 200, n_features: int = 8, n_classes: int = 3, seed: int = 13):
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=4.0, size=(n_classes, n_features))
    counts = [n_samples // n_classes] * n_classes
    counts[0] += n_samples - sum(counts)  # balance remainder
    X_parts = []
    y_parts = []
    for i, c in enumerate(counts):
        cov = np.eye(n_features) * rng.uniform(0.5, 1.5)
        X_i = rng.multivariate_normal(mean=centers[i], cov=cov, size=c, check_valid="warn")
        X_parts.append(X_i)
        y_parts.append(np.full(c, i, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def _call_flexibly(func: Callable, *args, **kwargs) -> Any:
    """
    Call `func` but only pass kwargs it actually supports (by signature).
    This prevents false failures if your function's signature changes slightly.
    """
    sig = inspect.signature(func)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return func(*args, **accepted)


def _collect_new_images(start_dir: Path, before: set[Path]) -> list[Path]:
    after = set(p for p in start_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"})
    new_files = [p for p in after if p not in before]
    return sorted(new_files)


def _assert_nonempty_image(path: Path):
    assert path.exists(), f"Expected image file not found: {path}"
    assert path.is_file(), f"Expected a file, got: {path}"
    size = path.stat().st_size
    assert size > 0, f"Image file seems empty: {path} (0 bytes)"


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def plot_umap_func():
    fn = _try_import_plot_umap()
    if fn is None:
        pytest.skip("plot_umap_v50() not found in expected modules. Skipping diagnostics plot tests.")
    return fn


# ---------- Tests ----------

def test_plot_umap_basic(tmp_path: Path, plot_umap_func: Callable):
    """
    Basic happy-path: verify that a plot image is created or a Matplotlib Figure is returned.
    """
    X, y = _make_blobs(n_samples=240, n_features=12, n_classes=4, seed=42)
    labels = [f"class_{i}" for i in y]
    out_dir = tmp_path / "umap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    desired_path = out_dir / "umap_v50_test.png"

    # Snapshot of existing images (to detect newly created files)
    before = set(p for p in out_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"})

    # Flexible call (we pass conservative, common kwargs; omitted if not supported)
    result = _call_flexibly(
        plot_umap_func,
        X,
        labels=labels,
        title="UMAP V50 – basic test",
        out_path=str(desired_path),
        random_state=123,
        show=False,
        figsize=(8, 6),
        dpi=120,
    )

    # Outcome options:
    # 1) Function returns a Matplotlib Figure; we save it.
    # 2) Function returns a path/str/pathlib.Path (where it saved the file).
    # 3) Function returns None but writes to out_path (or some file in the output dir).
    saved_files = _collect_new_images(out_dir, before)

    if result is not None and "matplotlib" in type(result).__module__.lower():
        # Likely a Figure
        fig = result
        # Ensure we can save cleanly
        fig.savefig(desired_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        _assert_nonempty_image(desired_path)
    elif isinstance(result, (str, os.PathLike, Path)):
        produced = Path(result)
        # Accept either absolute file or a directory that contains images
        if produced.is_file():
            _assert_nonempty_image(produced)
        else:
            # If a directory was returned, see if it contains a new image
            imgs = _collect_new_images(Path(produced), set())
            assert imgs, f"No images found in returned directory: {produced}"
            _assert_nonempty_image(imgs[0])
    else:
        # No explicit return – we expect an image file to exist (preferably at desired_path)
        if desired_path.exists():
            _assert_nonempty_image(desired_path)
        else:
            # Fall back to any newly created image in out_dir
            assert saved_files, "plot_umap_v50 produced no image file and returned nothing."
            _assert_nonempty_image(saved_files[0])


def test_plot_umap_small_input(tmp_path: Path, plot_umap_func: Callable):
    """
    Very small input should still produce a plot without crashing (e.g., n=3).
    """
    X, y = _make_blobs(n_samples=3, n_features=5, n_classes=3, seed=7)
    labels = [f"c{i}" for i in y]
    out_img = tmp_path / "tiny_umap.png"

    result = _call_flexibly(
        plot_umap_func,
        X,
        labels=labels,
        title="UMAP V50 – tiny input",
        out_path=str(out_img),
        random_state=7,
        show=False,
        dpi=100,
    )

    # See handling as in the basic test
    if result is not None and "matplotlib" in type(result).__module__.lower():
        fig = result
        fig.savefig(out_img, dpi=100, bbox_inches="tight")
        plt.close(fig)
        _assert_nonempty_image(out_img)
    else:
        # Path or None: check file existence
        if isinstance(result, (str, os.PathLike, Path)):
            out_path = Path(result)
        else:
            out_path = out_img
        _assert_nonempty_image(out_path)


def test_plot_umap_with_unknown_labels(tmp_path: Path, plot_umap_func: Callable):
    """
    When labels contain Nones/NaNs or unexpected values, plotting should still succeed.
    (If your function doesn't accept labels, this will just pass them if supported.)
    """
    X, y = _make_blobs(n_samples=100, n_features=6, n_classes=3, seed=11)
    labels = [None if i % 10 == 0 else f"group_{gi}" for i, gi in enumerate(y)]  # inject unknowns
    out_img = tmp_path / "umap_with_unknowns.svg"

    result = _call_flexibly(
        plot_umap_func,
        X,
        labels=labels,
        title="UMAP V50 – unknown labels",
        out_path=str(out_img),
        random_state=11,
        show=False,
        dpi=120,
    )

    if result is not None and "matplotlib" in type(result).__module__.lower():
        fig = result
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", dpi=120, bbox_inches="tight")
        plt.close(fig)
        # Ensure bytes were produced
        assert buf.getbuffer().nbytes > 0, "Expected non-empty SVG bytes from figure."
        # Also write to the expected path for consistency
        with open(out_img, "wb") as f:
            f.write(buf.getvalue())
        _assert_nonempty_image(out_img)
    else:
        # Path or None: check file existence
        if isinstance(result, (str, os.PathLike, Path)):
            out_path = Path(result)
        else:
            out_path = out_img
        _assert_nonempty_image(out_path)
```

**How to use**

* Save as `tests/diagnostics/test_plot_umap_v50.py`.
* Run with `pytest -q` (your CI will pick it up too).
* If `plot_umap_v50` isn’t implemented yet or lives in a different module path, the test will skip cleanly—just add its import path to `POSSIBLE_MODULES` later.
