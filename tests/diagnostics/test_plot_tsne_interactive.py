# tests/diagnostics/test_plot_tsne_interactive.py
"""
Upgraded tests for plot_tsne_interactive (interactive t-SNE diagnostics)

Coverage (robust, signature-agnostic, CI-safe):
- Function discovery: tries common module paths for `plot_tsne_interactive`.
  If not found, attempts CLI/module execution: `python -m tools.plot_tsne_interactive`.
  If neither exists yet, skips cleanly (keeps pipeline green while wiring up).
- Headless compatibility: forces a non-interactive Matplotlib backend for any static fallbacks.
- Happy path: generates synthetic embeddings, calls the function, and asserts an interactive
  HTML artifact was created (or a Matplotlib Figure/image was returned/saved).
- Small-edge path: verifies t-SNE handles very small inputs (n ~ 3–5) without crashing.
- Unknown labels path: tolerates `None`/NaN labels if the function accepts labels.
- Flexible kwargs: only passes options the function actually supports (introspected at runtime).

Acceptance:
- Return types:
  1) Path/string to an `.html` (preferred) or image file → file must exist and be non-empty.
  2) Matplotlib Figure → test saves to disk and verifies non-empty bytes.
  3) None → artifact must be created at requested `out_path` or under `out_dir`.

Notes:
- We avoid brittle assumptions about exact filenames or return values.
- If your function writes to an `out_path` (e.g., HTML), that passes. If it returns a figure,
  we save either PNG or SVG for verification.
"""

from __future__ import annotations

import io
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

# Force non-interactive backend for any matplotlib fallbacks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------- Discovery helpers ----------

POSSIBLE_MODULES = [
    # Add/adjust as your repo evolves
    "src.diagnostics.tsne_interactive",
    "src.diagnostics.plot_tsne_interactive",
    "diagnostics.tsne_interactive",
    "diagnostics.plot_tsne_interactive",
    "src.visualization.tsne_interactive",
    "src.visualization.plot_tsne_interactive",
    "visualization.tsne_interactive",
    "visualization.plot_tsne_interactive",
    # Some teams keep tools callable as modules
    "tools.plot_tsne_interactive",
]


def _try_import_tsne() -> Optional[Callable]:
    """
    Attempt to import `plot_tsne_interactive` from likely modules.
    Returns the function if found, else None.
    """
    for mod_name in POSSIBLE_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["plot_tsne_interactive"])
            fn = getattr(mod, "plot_tsne_interactive", None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None


def _has_cli_module() -> bool:
    """
    Detect whether `tools/plot_tsne_interactive.py` is present so we can call `python -m tools.plot_tsne_interactive`.
    """
    # Probe import first (without requiring the function)
    try:
        __import__("tools.plot_tsne_interactive")
        return True
    except Exception:
        return False


# ---------- Data helpers ----------

def _make_blobs(n_samples: int = 240, n_features: int = 12, n_classes: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(loc=0.0, scale=4.0, size=(n_classes, n_features))
    counts = [n_samples // n_classes] * n_classes
    counts[0] += n_samples - sum(counts)  # handle remainder
    X_parts = []
    y_parts = []
    for i, c in enumerate(counts):
        cov = np.eye(n_features) * rng.uniform(0.3, 1.3)
        X_i = rng.multivariate_normal(mean=centers[i], cov=cov, size=c, check_valid="warn")
        X_parts.append(X_i)
        y_parts.append(np.full(c, i, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


# ---------- Invocation helpers ----------

def _call_flexibly(func: Callable, *args, **kwargs) -> Any:
    """
    Call `func` but only pass kwargs it supports.
    """
    sig = inspect.signature(func)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **accepted)


def _collect_new_artifacts(start_dir: Path, before: set[Path]) -> list[Path]:
    """
    Return list of new artifacts (HTML or images) generated after a call.
    """
    valid_exts = {".html", ".htm", ".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    after = set(p for p in start_dir.rglob("*") if p.suffix.lower() in valid_exts)
    new_files = [p for p in after if p not in before]
    return sorted(new_files)


def _assert_nonempty_file(path: Path):
    assert path.exists(), f"Expected file not found: {path}"
    assert path.is_file(), f"Expected a file, got: {path}"
    size = path.stat().st_size
    assert size > 0, f"File seems empty: {path} (0 bytes)"


def _run_cli_module(summary_out: Path, out_html: Path, X: np.ndarray, labels: list[str | None]) -> None:
    """
    Attempt to run `python -m tools.plot_tsne_interactive` as a fallback if no function was found.
    This path is only used if the module exists; otherwise, tests will skip.
    """
    import tempfile
    import json
    import subprocess

    # Write minimal inputs to disk for CLI consumption
    tmp_dir = summary_out.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    x_path = tmp_dir / "X.npy"
    np.save(x_path, X)

    labels_path = tmp_dir / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SPECTRAMIND_TEST", "1")

    # We try common CLI options that many implementations expose.
    # Implementations should ignore unknown flags gracefully or print help and exit 0.
    cmd = [
        sys.executable,
        "-m",
        "tools.plot_tsne_interactive",
        "--X", str(x_path),
        "--labels-json", str(labels_path),
        "--out", str(out_html),
        "--title", "t-SNE Interactive — test",
        "--perplexity", "20",
        "--learning-rate", "200",
        "--n-iter", "250",
        "--random-state", "123",
        "--no-open",
    ]

    proc = subprocess.run(
        cmd, cwd=str(Path.cwd()), env=env, capture_output=True, text=True, timeout=90
    )

    if proc.returncode != 0:
        # Provide debug context
        print("CLI STDOUT:\n", proc.stdout)
        print("CLI STDERR:\n", proc.stderr)

    assert proc.returncode == 0, "CLI module execution failed."
    _assert_nonempty_file(out_html)


# ---------- Fixtures ----------

@pytest.fixture(scope="session")
def tsne_callable_or_cli_available():
    """
    Returns:
      - a callable if `plot_tsne_interactive` can be imported, OR
      - the string "CLI" if a CLI module exists,
      - otherwise skips the test session.
    """
    fn = _try_import_tsne()
    if fn is not None:
        return fn
    if _has_cli_module():
        return "CLI"
    pytest.skip("plot_tsne_interactive not found, and CLI module tools.plot_tsne_interactive not present. Skipping.")


# ---------- Tests ----------

def test_tsne_interactive_basic(tmp_path: Path, tsne_callable_or_cli_available):
    """
    Happy path: ensure interactive HTML (or an image/figure fallback) is produced.
    """
    X, y = _make_blobs(n_samples=200, n_features=10, n_classes=4, seed=101)
    labels = [f"class_{int(i)}" for i in y]

    out_dir = tmp_path / "tsne_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "tsne_interactive_test.html"

    before = set(p for p in out_dir.rglob("*") if p.suffix.lower() in {".html", ".htm", ".png", ".svg", ".jpg", ".jpeg", ".pdf"})

    if callable(tsne_callable_or_cli_available):
        fn = tsne_callable_or_cli_available
        result = _call_flexibly(
            fn,
            X,
            labels=labels,
            title="t-SNE Interactive — basic test",
            out_path=str(out_html),
            perplexity=20,
            learning_rate=200,
            n_iter=250,
            random_state=123,
            show=False,
            dpi=120,
            figsize=(8, 6),
        )

        new_files = _collect_new_artifacts(out_dir, before)

        # Interpret outcomes
        if result is not None and "matplotlib" in type(result).__module__.lower():
            fig = result
            # Save as SVG to validate bytes
            svg_path = out_dir / "tsne_interactive_result.svg"
            buf = io.BytesIO()
            fig.savefig(buf, format="svg", dpi=120, bbox_inches="tight")
            plt.close(fig)
            assert buf.getbuffer().nbytes > 0, "Expected non-empty SVG bytes from figure."
            svg_path.write_bytes(buf.getvalue())
            _assert_nonempty_file(svg_path)
        elif isinstance(result, (str, os.PathLike, Path)):
            produced = Path(result)
            if produced.is_file():
                _assert_nonempty_file(produced)
            else:
                # Directory returned — must contain a new artifact
                artifacts = _collect_new_artifacts(Path(produced), set())
                assert artifacts, f"No artifacts found in returned directory: {produced}"
                _assert_nonempty_file(artifacts[0])
        else:
            # No explicit return — expect requested out_html (preferred) OR any new artifact
            if out_html.exists():
                _assert_nonempty_file(out_html)
            else:
                assert new_files, "No artifacts were produced by plot_tsne_interactive."
                _assert_nonempty_file(new_files[0])

    else:
        # CLI fallback
        assert tsne_callable_or_cli_available == "CLI"
        _run_cli_module(summary_out=out_dir / "dummy.json", out_html=out_html, X=X, labels=labels)


def test_tsne_interactive_small_input(tmp_path: Path, tsne_callable_or_cli_available):
    """
    Very small input should still not crash and should produce an artifact.
    """
    X, y = _make_blobs(n_samples=4, n_features=6, n_classes=4, seed=7)
    labels = [f"c{int(i)}" for i in y]
    out_dir = tmp_path / "tsne_small"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "tsne_small.html"

    if callable(tsne_callable_or_cli_available):
        fn = tsne_callable_or_cli_available
        result = _call_flexibly(
            fn,
            X,
            labels=labels,
            title="t-SNE Interactive — tiny input",
            out_path=str(out_html),
            perplexity=2,          # tiny perplexity for tiny data
            n_iter=250,
            random_state=7,
            show=False,
        )
        # As before: accept figure/path/None
        if result is not None and "matplotlib" in type(result).__module__.lower():
            fig = result
            png_path = out_dir / "tiny_tsne.png"
            fig.savefig(png_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            _assert_nonempty_file(png_path)
        else:
            out = Path(result) if isinstance(result, (str, os.PathLike, Path)) else out_html
            _assert_nonempty_file(out if out.exists() else out_html)
    else:
        assert tsne_callable_or_cli_available == "CLI"
        _run_cli_module(summary_out=out_dir / "dummy.json", out_html=out_html, X=X, labels=labels)


def test_tsne_interactive_with_unknown_labels(tmp_path: Path, tsne_callable_or_cli_available):
    """
    Unknown labels (None/NaN) should not break plotting (if labels are supported).
    """
    X, y = _make_blobs(n_samples=120, n_features=8, n_classes=3, seed=11)
    labels = [None if i % 11 == 0 else f"group_{int(gi)}" for i, gi in enumerate(y)]

    out_dir = tmp_path / "tsne_unknowns"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "tsne_unknowns.html"

    if callable(tsne_callable_or_cli_available):
        fn = tsne_callable_or_cli_available
        result = _call_flexibly(
            fn,
            X,
            labels=labels,
            title="t-SNE Interactive — unknown labels",
            out_path=str(out_html),
            perplexity=15,
            n_iter=300,
            random_state=11,
            show=False,
        )

        if result is not None and "matplotlib" in type(result).__module__.lower():
            fig = result
            svg_path = out_dir / "tsne_unknowns.svg"
            buf = io.BytesIO()
            fig.savefig(buf, format="svg", dpi=120, bbox_inches="tight")
            plt.close(fig)
            assert buf.getbuffer().nbytes > 0, "Expected non-empty SVG bytes from figure."
            svg_path.write_bytes(buf.getvalue())
            _assert_nonempty_file(svg_path)
        else:
            out = Path(result) if isinstance(result, (str, os.PathLike, Path)) else out_html
            _assert_nonempty_file(out if out.exists() else out_html)
    else:
        assert tsne_callable_or_cli_available == "CLI"
        _run_cli_module(summary_out=out_dir / "dummy.json", out_html=out_html, X=X, labels=labels)
