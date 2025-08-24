# tests/diagnostics/test_plot_fft_power_cluster_compare.py
"""
Diagnostics: FFT Power Spectrum Cluster Comparison

This test generates synthetic multi-sample time-series grouped into clusters
with distinct dominant frequencies. It then:

1) Computes normalized FFT power spectra per sample
2) Clusters samples in power-spectrum space (KMeans)
3) Compares predicted vs. true clusters quantitatively (ARI) and qualitatively
   by saving a figure with mean spectra per (true/predicted) cluster.

Why it exists:
- Verifies our spectral feature pipeline (FFT power) is stable & informative
- Provides a repeatable, headless diagnostic artifact for quick visual checks
- Serves as a regression test: small code changes shouldn’t break clustering
  separability on basic synthetic cases.

This test is headless (uses Agg backend) and deterministic (fixed RNG seeds).
It writes artifacts under pytest’s tmp_path to avoid polluting the repo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib

# Force headless for CI / non-GUI environments
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def _synthesize_clustered_timeseries(
    n_clusters: int = 3,
    samples_per_cluster: int = 40,
    n_time: int = 2048,
    fs: float = 128.0,
    snr_db: float = 10.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic time-series grouped by clusters with distinct dominant frequencies.

    Returns
    -------
    x : array, shape (N, n_time)
        Time-domain signals for all samples (stacked).
    y_true : array, shape (N,)
        True cluster labels in [0, n_clusters-1].
    dom_freqs : array, shape (n_clusters,)
        The dominant frequency (Hz) used per cluster.
    """
    if rng is None:
        rng = np.random.default_rng(7)

    N = n_clusters * samples_per_cluster
    t = np.arange(n_time) / fs

    # Choose well-separated dominant frequencies (Hz)
    # Keep within Nyquist range: (0, fs/2)
    # Example: 6 Hz, 14 Hz, 28 Hz for fs=128Hz
    dom_freqs = np.linspace(6.0, fs / 4.0, n_clusters)

    x = np.zeros((N, n_time), dtype=np.float64)
    y_true = np.repeat(np.arange(n_clusters, dtype=int), samples_per_cluster)

    # Signal-to-noise
    snr_lin = 10 ** (snr_db / 10)

    idx = 0
    for k, f0 in enumerate(dom_freqs):
        for _ in range(samples_per_cluster):
            # Base sinusoid + slight frequency jitter + multi-harmonic content
            f_jitter = f0 * (1.0 + rng.normal(0.0, 0.01))
            phase = rng.uniform(0, 2 * np.pi)
            s = np.sin(2 * np.pi * f_jitter * t + phase)

            # Add a weaker harmonic and amplitude variation
            s += 0.35 * np.sin(2 * np.pi * 2.0 * f_jitter * t + rng.uniform(0, 2 * np.pi))
            s += 0.15 * np.sin(2 * np.pi * 3.0 * f_jitter * t + rng.uniform(0, 2 * np.pi))
            s *= rng.uniform(0.9, 1.1)

            # White Gaussian noise to achieve target SNR
            power_s = np.mean(s**2)
            noise_power = power_s / snr_lin
            n = rng.normal(0.0, np.sqrt(noise_power), size=n_time)

            x[idx] = s + n
            idx += 1

    # Optionally, apply a tiny DC detrend so low-freq bias doesn’t dominate
    x -= x.mean(axis=1, keepdims=True)

    return x, y_true, dom_freqs


def _power_spectrum(
    x: np.ndarray,
    fs: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided normalized power spectrum via rFFT for each sample.

    Parameters
    ----------
    x : array, shape (N, T)
        Time-series per row (N samples).
    fs : float
        Sampling rate (Hz).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    freqs : array, shape (F,)
        Frequency axis (Hz), one-sided.
    Pn : array, shape (N, F)
        Normalized power spectra (each row sums to 1).
    """
    N, T = x.shape
    # rfft returns T//2+1 frequency bins (one-sided)
    X = np.fft.rfft(x, axis=1)
    P = (np.abs(X) ** 2) / T  # power
    # Normalize each sample to unit sum for clustering robustness
    Pn = P / (P.sum(axis=1, keepdims=True) + eps)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    return freqs, Pn


def _cluster_power(Pn: np.ndarray, n_clusters: int, seed: int = 11) -> np.ndarray:
    """
    Cluster normalized power spectra rows using KMeans.
    """
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    y_pred = km.fit_predict(Pn)
    return y_pred


def _plot_cluster_compare(
    out_png: Path,
    freqs: np.ndarray,
    Pn: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dom_freqs: np.ndarray,
    max_freq: float | None = None,
    title: str = "FFT Power Cluster Compare",
) -> None:
    """
    Save a figure comparing mean spectra per true vs predicted clusters.
    """
    n_clusters = len(np.unique(y_true))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Limit frequency axis if requested
    mask = np.ones_like(freqs, dtype=bool)
    if max_freq is not None:
        mask = freqs <= max_freq

    # True clusters
    ax = axes[0]
    for k in range(n_clusters):
        mean_P = Pn[y_true == k].mean(axis=0)
        ax.plot(freqs[mask], mean_P[mask], label=f"True k={k}")
    for f0 in dom_freqs:
        ax.axvline(f0, color="k", lw=0.8, ls="--", alpha=0.35)
    ax.set_title("Mean spectra by TRUE cluster")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    # Predicted clusters
    ax = axes[1]
    for k in range(n_clusters):
        mean_P = Pn[y_pred == k].mean(axis=0)
        ax.plot(freqs[mask], mean_P[mask], label=f"Pred k={k}")
    for f0 in dom_freqs:
        ax.axvline(f0, color="k", lw=0.8, ls="--", alpha=0.35)
    ax.set_title("Mean spectra by PREDICTED cluster")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized power")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


@pytest.mark.parametrize(
    "snr_db,expected_ari",
    [
        (12.0, 0.90),  # easier: higher SNR -> ARI should be very high
        (8.0, 0.80),   # moderate SNR -> ARI should still be good
    ],
)
def test_fft_power_cluster_compare(tmp_path: Path, snr_db: float, expected_ari: float) -> None:
    """
    End-to-end diagnostic:
    - synthesize clustered time-series
    - FFT power -> normalized
    - KMeans clustering
    - Evaluate ARI
    - Save comparison plot

    Assertions:
      * ARI >= expected_ari
      * output image file exists and is non-empty
      * basic sanity on cluster membership counts
    """
    # Allow a "fast" mode for CI by shortening the series
    fast = os.getenv("SPECTRAMIND_FAST", "0") == "1"
    n_time = 1024 if fast else 2048
    samples_per_cluster = 25 if fast else 40

    fs = 128.0
    rng = np.random.default_rng(123)

    x, y_true, dom_freqs = _synthesize_clustered_timeseries(
        n_clusters=3,
        samples_per_cluster=samples_per_cluster,
        n_time=n_time,
        fs=fs,
        snr_db=snr_db,
        rng=rng,
    )
    freqs, Pn = _power_spectrum(x, fs=fs)

    y_pred = _cluster_power(Pn, n_clusters=3, seed=2025)
    ari = adjusted_rand_score(y_true, y_pred)

    # Persist a headless comparison plot for humans to inspect if needed
    out_png = tmp_path / "fft_power_cluster_compare.png"
    _plot_cluster_compare(
        out_png=out_png,
        freqs=freqs,
        Pn=Pn,
        y_true=y_true,
        y_pred=y_pred,
        dom_freqs=dom_freqs,
        max_freq=fs / 2.5,  # zoom a bit for readability
        title=f"FFT Power Cluster Compare (SNR={snr_db:.1f} dB, ARI={ari:.3f})",
    )

    # Assertions
    assert ari >= expected_ari, f"ARI {ari:.3f} < expected {expected_ari:.3f}"
    assert out_png.exists() and out_png.stat().st_size > 0, "Expected plot file to be written"
    # Each predicted cluster should have at least a few members (no collapse)
    unique, counts = np.unique(y_pred, return_counts=True)
    assert len(unique) == 3 and counts.min() >= max(3, samples_per_cluster // 10)


def test_fft_power_shapes_and_normalization(tmp_path: Path) -> None:
    """
    Sanity checks: power spectrum shape and normalization invariants.
    """
    fs = 64.0
    x = np.stack(
        [
            np.sin(2 * np.pi * 5 * np.arange(512) / fs),
            np.sin(2 * np.pi * 12 * np.arange(512) / fs),
        ],
        axis=0,
    )

    freqs, Pn = _power_spectrum(x, fs=fs)
    # rFFT length should be T//2+1
    assert Pn.shape == (2, 512 // 2 + 1)
    assert freqs.shape == (512 // 2 + 1,)

    # Normalization to unit sum per sample
    row_sums = Pn.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    # Plot once to ensure no exceptions in plotting utility
    out = tmp_path / "sanity_plot.png"
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    _plot_cluster_compare(out, freqs, Pn, y_true, y_pred, dom_freqs=np.array([5.0, 12.0]))
    assert out.exists() and out.stat().st_size > 0
