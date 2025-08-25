# tests/diagnostics/test_shap_attention_overlay.py
# SpectraMind V50 — Diagnostics
#
# Purpose
# -------
# Contract tests for the "SHAP × Attention overlay" explainer utilities.
# These tests do **not** assume a particular internal formula. Instead, they
# assert interface/shape/bounds, α-behavior (mixing weight), batching support,
# plotting side‑effects, and error handling for mismatched lengths.
#
# The test suite dynamically resolves the implementation so it can run against:
# - spectramind.explain.shap_attention.overlay:generate_overlay / overlay_heatmap
# - spectramind.explain.shap_attention:generate_overlay / overlay_heatmap
# - spectramind.explain:generate_shap_attention_overlay / plot_shap_attention_overlay
#
# If no known symbol is found, tests are skipped (not failed), keeping CI green
# until the module ships. When the module appears, these tests immediately start
# validating it without code changes.
#
# Design Notes
# ------------
# * Deterministic seed: reproducible synthetic fixtures.
# * α-extremes: α=1 should be dominated by SHAP, α=0 by Attention — we verify
#   with a constructed case where SHAP and Attention prefer different tokens.
# * Normalization: overlay importance should be bounded in [0, 1] per contract.
# * Batching: function should accept (B, T, T) attention and (B, T) shap.
# * Plotting: heatmap path should be created and non-empty.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import io
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pytest


# --------- Dynamic resolver ---------------------------------------------------


@dataclass
class OverlayAPI:
    """Resolved overlay API surface."""
    generate_overlay: Callable[..., np.ndarray]
    overlay_heatmap: Optional[Callable[..., Any]]  # may be None if not present


def _try_import(path: str, name: str) -> Optional[Callable[..., Any]]:
    try:
        mod = importlib.import_module(path)
    except Exception:
        return None
    return getattr(mod, name, None)


def resolve_overlay_api() -> Optional[OverlayAPI]:
    # 1) Preferred module path
    gen = _try_import("spectramind.explain.shap_attention.overlay", "generate_overlay")
    heat = _try_import("spectramind.explain.shap_attention.overlay", "overlay_heatmap")
    if gen:
        return OverlayAPI(generate_overlay=gen, overlay_heatmap=heat)

    # 2) Fallback package-level variant
    gen = _try_import("spectramind.explain.shap_attention", "generate_overlay")
    heat = _try_import("spectramind.explain.shap_attention", "overlay_heatmap")
    if gen:
        return OverlayAPI(generate_overlay=gen, overlay_heatmap=heat)

    # 3) Legacy names
    gen = _try_import("spectramind.explain", "generate_shap_attention_overlay")
    heat = _try_import("spectramind.explain", "plot_shap_attention_overlay")
    if gen:
        return OverlayAPI(generate_overlay=gen, overlay_heatmap=heat)

    return None


overlay_api = resolve_overlay_api()

skip_reason = (
    "SHAP×Attention overlay implementation was not found. "
    "Expected one of:\n"
    "  spectramind.explain.shap_attention.overlay.generate_overlay\n"
    "  spectramind.explain.shap_attention.generate_overlay\n"
    "  spectramind.explain.generate_shap_attention_overlay\n"
    "Once the overlay module is added to the repo, these tests will run."
)


# --------- Fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(1337)


def make_attention_matrix(T: int, prefer_idx: int) -> np.ndarray:
    """
    Construct a simple (T, T) attention matrix that concentrates mass onto
    a single target position `prefer_idx`. Symmetric for simplicity.
    """
    attn = np.zeros((T, T), dtype=np.float32)
    # Put high mass on the preferred column across all query positions.
    attn[:, prefer_idx] = 1.0
    # Normalize each row to sum 1 (already true, but keep contract explicit).
    # Handle degenerate rows just in case.
    row_sums = attn.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    attn = attn / row_sums
    return attn


def make_shap_values(T: int, prefer_idx: int) -> np.ndarray:
    """
    Construct SHAP values that prefer a (different) index by giving it a
    larger positive contribution.
    """
    shap = np.zeros((T,), dtype=np.float32)
    shap[prefer_idx] = 10.0
    return shap


# --------- Tests: contract & behavior ----------------------------------------


@pytest.mark.skipif(overlay_api is None, reason=skip_reason)
@pytest.mark.parametrize("T", [5, 16])
def test_shapes_bounds_and_contract(T: int, rng: np.random.Generator):
    """
    Contract:
      * Input: attention (T,T), shap (T,)
      * Output: overlay importance (T,) in [0, 1]
      * Deterministic on given inputs/alpha
    """
    gen = overlay_api.generate_overlay

    attn = rng.random((T, T), dtype=np.float32)
    # row-normalize attention
    attn = attn / np.clip(attn.sum(axis=1, keepdims=True), 1e-9, None)
    shap = rng.normal(size=(T,)).astype(np.float32)

    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = gen(attention=attn, shap=shap, alpha=alpha)
        assert isinstance(out, np.ndarray), "Overlay must return a numpy array"
        assert out.shape == (T,), "Overlay must return (T,) vector"
        assert np.all(np.isfinite(out)), "Overlay must not contain NaNs/Inf"
        # Bounded [0, 1] importance scores
        assert np.min(out) >= -1e-6 and np.max(out) <= 1 + 1e-6, (
            "Overlay importance should be normalized to [0, 1]"
        )

        # Determinism: repeated call with same inputs returns same values
        out2 = gen(attention=attn, shap=shap, alpha=alpha)
        np.testing.assert_allclose(out, out2, rtol=0, atol=0, err_msg="Overlay must be deterministic")


@pytest.mark.skipif(overlay_api is None, reason=skip_reason)
def test_alpha_extremes_prefer_sources():
    """
    Behavioral sanity:
      * α=1 → overlay dominated by SHAP ranking
      * α=0 → overlay dominated by Attention ranking
    We create a setting where the SHAP and Attention favor different tokens.
    """
    gen = overlay_api.generate_overlay
    T = 8
    idx_shap = 2
    idx_attn = 6

    attn = make_attention_matrix(T=T, prefer_idx=idx_attn)
    shap = make_shap_values(T=T, prefer_idx=idx_shap)

    # α = 1 → SHAP-only
    out_shap = gen(attention=attn, shap=shap, alpha=1.0)
    top_idx_shap = int(np.argmax(out_shap))
    assert top_idx_shap == idx_shap, (
        f"With alpha=1.0 (SHAP-dominant), top token should be {idx_shap}, "
        f"got {top_idx_shap}"
    )

    # α = 0 → Attention-only
    out_attn = gen(attention=attn, shap=shap, alpha=0.0)
    top_idx_attn = int(np.argmax(out_attn))
    assert top_idx_attn == idx_attn, (
        f"With alpha=0.0 (Attention-dominant), top token should be {idx_attn}, "
        f"got {top_idx_attn}"
    )

    # α = 0.5 → in-between; should include both indices among top-K in many implementations
    out_mid = gen(attention=attn, shap=shap, alpha=0.5)
    top2 = np.argsort(out_mid)[-2:]
    assert idx_shap in top2 and idx_attn in top2, (
        "With alpha=0.5, both SHAP- and Attention‑preferred tokens should rank highly."
    )


@pytest.mark.skipif(overlay_api is None, reason=skip_reason)
def test_batched_inputs_supported_or_meaningful_error():
    """
    If implementation supports batching:
      attention: (B, T, T), shap: (B, T) → out: (B, T)
    Otherwise, the function should raise a clear ValueError/TypeError.
    """
    gen = overlay_api.generate_overlay
    B, T = 3, 7

    attn = np.stack([make_attention_matrix(T, prefer_idx=i % T) for i in range(B)], axis=0)  # (B,T,T)
    shap = np.stack([make_shap_values(T, prefer_idx=(i + 2) % T) for i in range(B)], axis=0)  # (B,T)

    try:
        out = gen(attention=attn, shap=shap, alpha=0.5)
    except (ValueError, TypeError) as e:
        # Accept clear error about unsupported batching
        assert "batch" in str(e).lower() or "shape" in str(e).lower()
        return

    # If it did not raise, assert proper shape & bounds.
    assert isinstance(out, np.ndarray)
    assert out.shape == (B, T)
    assert np.min(out) >= -1e-6 and np.max(out) <= 1 + 1e-6


@pytest.mark.skipif(overlay_api is None, reason=skip_reason)
def test_mismatched_lengths_raise():
    """
    attention: (T,T), shap: (T2,) with T2 != T → must raise a clear error.
    """
    gen = overlay_api.generate_overlay
    T, T2 = 6, 5
    attn = make_attention_matrix(T, prefer_idx=1)
    shap = make_shap_values(T2, prefer_idx=2)

    with pytest.raises((ValueError, AssertionError, TypeError)):
        _ = gen(attention=attn, shap=shap, alpha=0.5)


@pytest.mark.skipif(overlay_api is None or overlay_api.overlay_heatmap is None, reason=skip_reason)
def test_overlay_heatmap_writes_file(tmp_path):
    """
    overlay_heatmap(attention, shap, alpha, tokens, save_path=...) should
    create a non-empty image artifact when provided.
    """
    heat = overlay_api.overlay_heatmap
    assert heat is not None, "overlay_heatmap not found (resolver bug)."

    T = 9
    attn = make_attention_matrix(T, prefer_idx=4)
    shap = make_shap_values(T, prefer_idx=2)
    tokens = [f"t{i}" for i in range(T)]
    out_path = tmp_path / "overlay.png"

    # Some implementations accept **kwargs like dpi, cmap, title; keep minimal.
    heat(attention=attn, shap=shap, alpha=0.6, tokens=tokens, save_path=str(out_path))

    assert out_path.exists(), "overlay_heatmap did not create the file"
    assert out_path.stat().st_size > 0, "overlay_heatmap wrote an empty file"


# --------- Optional smoke: PNG buffer support (if overlay_heatmap returns bytes) ---------


@pytest.mark.skipif(overlay_api is None, reason=skip_reason)
def test_overlay_heatmap_buffer_mode_if_supported():
    """
    Some implementations may optionally return bytes/BytesIO when save_path is None.
    If present, verify the buffer is non-empty PNG.
    """
    heat = overlay_api.overlay_heatmap
    if heat is None:
        pytest.skip("overlay_heatmap is not available")

    T = 6
    attn = make_attention_matrix(T, prefer_idx=1)
    shap = make_shap_values(T, prefer_idx=4)
    tokens = [f"x{i}" for i in range(T)]

    try:
        buf = heat(attention=attn, shap=shap, alpha=0.4, tokens=tokens, save_path=None)
    except TypeError:
        # Implementation might require save_path → acceptable
        pytest.skip("overlay_heatmap does not support buffer mode")
        return

    # Accept either bytes or a BytesIO-like object
    if hasattr(buf, "getvalue"):
        data = buf.getvalue()
    else:
        data = buf
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 16
    # PNG signature
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Expected PNG buffer from overlay_heatmap"


# --------- CLI hook (non-failing smoke) --------------------------------------


@pytest.mark.skipif(overlay_api is None, reason=skip_reason)
def test_api_is_importable_and_documented():
    """
    Very light smoke to ensure resolved function has a docstring.
    Helps keep API self-documented.
    """
    gen = overlay_api.generate_overlay
    assert callable(gen)
    doc = getattr(gen, "__doc__", "") or ""
    # Do not fail hard; nudge toward documentation.
    assert "overlay" in doc.lower() or len(doc) > 24, "Consider adding a brief docstring to generate_overlay()"

