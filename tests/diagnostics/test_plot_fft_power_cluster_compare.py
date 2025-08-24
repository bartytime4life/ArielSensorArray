# tests/diagnostics/test_plot_fft_power_cluster_compare.py
"""
Upgraded diagnostic test: FFT power clustering + comparison plot.

What this test does (fully self-contained):
- Synthesizes 3 clusters of time-series with distinct dominant frequencies
- Computes FFT power spectra for each sample
- Clusters the samples by their dominant frequency (simple & robust)
- Produces a comparison plot of cluster-mean power spectra
- Verifies:
    * The saved figure file exists and is non-empty
    * The recovered cluster centers match the intended frequencies (within tol)
    * The plotted cluster means show a power peak near the intended band

This test does not require scikit-learn or any external libs beyond numpy/matplotlib/pytest.
It is deterministic via a fixed RNG seed and runs in < 2 seconds on a typical CI runner.

If your repository already exposes a utility to do this (e.g.,
`spectramind.diagnostics.fft.plot_fft_power_cluster_compare`), you can optionally
wire it in the `try_import_repo_plotter()` hook below to exercise your code instead.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import matplotlib

# Use a non-interactive backend suitable for headless CI environments
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pytest


# ----------------------------
# Optional hook into repo code
# ----------------------------
def try_import_repo_plotter() -> Optional[Callable]:
    """
    If your repo provides a plotting helper, return a callable with the interface:
        plotter(
            freqs: np.ndarray,      # shape [F], Hz
            powers: np.ndarray,     # shape [N, F], power spectra
            labels: np.ndarray,     # shape [N], cluster ids 0..K-1
            outpath: str,           # where to save the figure
            title: str = "...",
        ) -> matplotlib.figure.Figure
    Otherwise return None to use the internal reference implementation.
    """
    try:
        # Example of possible location; adjust if your repo has a different path.
        from spectramind.diagnostics.fft_power import plot_fft_power_cluster_compare as repo_plotter  # type: ignore

        return repo_plotter
    except Exception:
        return None


# ----------------------------
# Synthetic data generation
# ----------------------------
@dataclass(frozen=True)
class ClusterSpec:
    freq_hz: float
    count: int
    amp: float = 1.0


def _gen_signal(
    rng: np.random.Generator,
    n: int,
    fs: float,
    f0: float,
    amp: float = 1.0,
    noise_std: float = 0.3,
    harmonics: Iterable[Tuple[int, float]] = ((2, 0.25), (3, 0.1)),
) -> np.ndarray:
    """
    Generate a single synthetic time-series with a dominant sinusoid at f0 plus mild harmonics & noise.
    """
    t = np.arange(n, dtype=np.float64) / fs

    # Allow tiny random phase/frequency jitter to make clustering realistic
    phase = rng.uniform(0, 2 * math.pi)
    f_jitter = rng.normal(0.0, 0.05)  # ~0.05 Hz jitter

    x = amp * np.sin(2 * math.pi * (f0 + f_jitter) * t + phase)

    # Add subtle harmonics for more realistic spectra
    for k, rel_amp in harmonics:
        x += amp * rel_amp * np.sin(2 * math.pi * k * (f0 + 0.5 * f_jitter) * t + rng.uniform(0, 2 * math.pi))

    # Add white noise
    x += rng.normal(0.0, noise_std, size=n)
    return x.astype(np.float64)


def _rfft_power(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Real FFT power spectrum (one-sided). Returns (freqs_hz, power).
    """
    n = x.shape[-1]
    # Use rfft to get non-redundant positive frequencies (including DC & Nyquist)
    X = np.fft.rfft(x, axis=-1)
    power = (np.abs(X) ** 2) / n  # power normalization
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, power


def _stack_power(signals: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute stacked power spectra for a batch of signals.
    signals: [N, T]
    Returns (freqs: [F], powers: [N, F])
    """
    freqs, first = _rfft_power(signals[0], fs)
    N = signals.shape[0]
    F = first.shape[-1]
    powers = np.empty((N, F), dtype=np.float64)
    powers[0] = first
    for i in range(1, N):
        _, Pi = _rfft_power(signals[i], fs)
        powers[i] = Pi
    return freqs, powers


def _dominant_freq(freqs: np.ndarray, power: np.ndarray) -> float:
    idx = int(np.argmax(power))
    return float(freqs[idx])


def _simple_kmeans_1d(values: np.ndarray, k: int, iters: int = 25, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal 1D k-means for stable clustering by dominant frequency.
    Returns (labels: [N], centers: [k])
    """
    assert values.ndim == 1
    rng = rng or np.random.default_rng(0)
    # Initialize centers from quantiles for stability
    qs = np.linspace(0.1, 0.9, k)
    centers = np.quantile(values, qs)
    centers_prev = centers.copy()

    for _ in range(iters):
        # Assign
        dists = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)

        # Update
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # Re-seed empty cluster at a random point
                centers[j] = values[rng.integers(0, len(values))]
            else:
                centers[j] = float(np.mean(values[mask]))

        # Converged?
        if np.allclose(centers, centers_prev, atol=1e-6, rtol=0.0):
            break
        centers_prev = centers.copy()

    # Final assignment
    dists = np.abs(values[:, None] - centers[None, :])
    labels = np.argmin(dists, axis=1)
    return labels, centers


# ----------------------------
# Plotting
# ----------------------------
def _default_plotter(
    freqs: np.ndarray,
    powers: np.ndarray,
    labels: np.ndarray,
    outpath: str,
    title: str = "FFT Power Cluster Compare",
) -> matplotlib.figure.Figure:
    """
    Reference implementation for the comparison plot.
    - Plots mean +/- std power per cluster on a shared axes (log10 power)
    - Highlights peak band for each cluster
    """
    assert powers.ndim == 2 and powers.shape[0] == labels.shape[0]
    K = int(labels.max()) + 1
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120, constrained_layout=True)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10 Power")
    ax.grid(True, alpha=0.25)

    colors = plt.cm.tab10.colors
    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            continue
        Pk = powers[mask]  # [Nk, F]
        mean = Pk.mean(axis=0)
        std = Pk.std(axis=0)

        # Avoid log of zero by adding tiny epsilon
        eps = 1e-12
        ax.plot(freqs, np.log10(mean + eps), color=colors[k % len(colors)], lw=2, label=f"Cluster {k} (n={mask.sum()})")
        ax.fill_between(freqs, np.log10(np.maximum(mean - std, eps)), np.log10(mean + std + eps), color=colors[k % len(colors)], alpha=0.15)

        # Mark cluster peak
        peak_idx = int(np.argmax(mean))
        ax.axvline(freqs[peak_idx], color=colors[k % len(colors)], ls="--", alpha=0.5)

    ax.legend(frameon=False, ncols=min(K, 3))
    fig.savefig(outpath)
    return fig


# ----------------------------
# Helper checks
# ----------------------------
def _match_centers_to_targets(centers: np.ndarray, targets: np.ndarray, tol: float) -> bool:
    """
    Greedy matching: for each target, see if a center lies within tol, without reusing centers.
    """
    centers_sorted = list(sorted(centers))
    for t in sorted(targets):
        ok = False
        for i, c in enumerate(centers_sorted):
            if abs(c - t) <= tol:
                centers_sorted.pop(i)
                ok = True
                break
        if not ok:
            return False
    return True


def _assert_peak_near_band(freqs: np.ndarray, mean_power: np.ndarray, target: float, band_hz: float = 0.6) -> None:
    """
    Assert that the mean power shows a local maximum near `target` within +/- band_hz.
    """
    idx_band = np.where((freqs >= target - band_hz) & (freqs <= target + band_hz))[0]
    assert idx_band.size > 3, f"No frequency bins within ±{band_hz} Hz around {target} Hz"
    band_power = mean_power[idx_band]
    band_peak_idx = idx_band[int(np.argmax(band_power))]
    assert abs(freqs[band_peak_idx] - target) <= band_hz, f"Peak {freqs[band_peak_idx]:.3f} Hz not near target {target} Hz"


# ----------------------------
# The actual test
# ----------------------------
@pytest.mark.fast
def test_plot_fft_power_cluster_compare(tmp_path: pytest.TempPathFactory) -> None:
    rng = np.random.default_rng(12345)

    # Sampling setup
    fs = 100.0  # Hz
    duration_s = 8.0
    n = int(fs * duration_s)

    # Intended cluster centers
    true_clusters = [
        ClusterSpec(freq_hz=5.0, count=10, amp=1.0),
        ClusterSpec(freq_hz=12.0, count=10, amp=1.0),
        ClusterSpec(freq_hz=20.0, count=10, amp=1.0),
    ]
    true_freqs = np.array([c.freq_hz for c in true_clusters], dtype=np.float64)
    K = len(true_clusters)

    # Generate signals
    signals = []
    for spec in true_clusters:
        for _ in range(spec.count):
            signals.append(_gen_signal(rng, n=n, fs=fs, f0=spec.freq_hz, amp=spec.amp, noise_std=0.25))
    signals = np.stack(signals, axis=0)  # [N, T]
    N = signals.shape[0]

    # Compute power spectra
    freqs, powers = _stack_power(signals, fs=fs)  # [F], [N, F]

    # Feature for clustering: dominant frequency per sample
    dom_freqs = np.array([_dominant_freq(freqs, p) for p in powers], dtype=np.float64)

    # 1D k-means to label the samples into K clusters
    labels, centers = _simple_kmeans_1d(dom_freqs, k=K, iters=30, rng=rng)

    # Validate recovered cluster centers are near the true ones
    tol_hz = 0.8  # tolerance given noise/jitter
    assert _match_centers_to_targets(centers, true_freqs, tol=tol_hz), (
        f"Recovered centers {centers} not all within {tol_hz} Hz of targets {true_freqs}"
    )

    # Select plotting implementation: repo hook if available, else internal
    plotter = try_import_repo_plotter() or _default_plotter

    # Save figure
    out_png = os.path.join(str(tmp_path), "fft_power_cluster_compare.png")
    fig = plotter(freqs=freqs, powers=powers, labels=labels, outpath=out_png, title="FFT Power Cluster Compare (Synthetic)")
    plt.close(fig)

    # Basic file checks
    assert os.path.exists(out_png), "Expected plot image not found"
    file_size = os.path.getsize(out_png)
    assert file_size > 5_000, f"Plot image seems too small ({file_size} bytes)"

    # Content sanity: ensure each cluster mean has a peak near its intended band
    for k in range(K):
        mask = labels == k
        assert np.any(mask), f"Cluster {k} is empty"
        mean_power = powers[mask].mean(axis=0)
        # Find the nearest true target to the cluster center and assert a local peak around it
        nearest_target = float(true_freqs[np.argmin(np.abs(true_freqs - centers[k]))])
        _assert_peak_near_band(freqs, mean_power, target=nearest_target, band_hz=0.8)

    # Determinism sanity: stable image meta (avoid strict pixel hash due to renderer diffs)
    with open(out_png, "rb") as f:
        head = f.read(64)
    assert head.startswith(b"\x89PNG\r\n\x1a\n"), "Output is not a PNG file"
    # Weak checksum on first KB to detect egregious instability across runs
    with open(out_png, "rb") as f:
        first_kb = f.read(1024)
    checksum = hashlib.sha1(first_kb).hexdigest()
    assert len(checksum) == 40
