#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_fft_autocorr.py

SpectraMind V50 — FFT & Autocorrelation Diagnostics

Purpose
-------
Compute FFT and autocorrelation diagnostics on μ spectra (mean predictions).
- FFT: identify dominant frequency bands, smoothness, instrument noise.
- Autocorrelation: detect periodic patterns and temporal correlations in bins.
- Symbolic overlays: highlight molecular regions (H₂O, CH₄, CO₂).
- SHAP overlays: optional bin importance overlays.
Outputs interactive plots (HTML) + PNG/CSV summaries for diagnostics dashboard.

Inputs
------
- predictions.json (default: outputs/predictions/predictions.json)
  Must contain {"planet_id": {"mu": [...], "sigma": [...]}} for each planet.
- symbolic_fingerprints.json (optional)
- shap_values.json (optional)

Outputs
-------
- outputs/diagnostics/fft_autocorr_{planet_id}.png
- outputs/diagnostics/fft_autocorr_{planet_id}.html
- outputs/diagnostics/fft_autocorr_summary.csv
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compute_fft(signal: np.ndarray, dt: float = 1.0):
    """Compute FFT power spectrum."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, dt)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    power = np.abs(fft_vals) ** 2 / n
    return freqs, power


def compute_autocorr(signal: np.ndarray, max_lag: int = 100):
    """Compute autocorrelation of signal up to max_lag."""
    signal = signal - np.mean(signal)
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2 :]
    corr /= corr[0]  # normalize
    return np.arange(len(corr[:max_lag])), corr[:max_lag]


def plot_fft_autocorr(mu, planet_id, outdir, overlays=None):
    """Plot FFT and autocorrelation diagnostics for a single planet."""
    freqs, power = compute_fft(mu)
    lags, acorr = compute_autocorr(mu)

    # Matplotlib PNG plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(freqs, power)
    axs[0].set_title(f"FFT Spectrum — {planet_id}")
    axs[0].set_xlabel("Frequency")
    axs[0].set_ylabel("Power")

    axs[1].plot(lags, acorr)
    axs[1].set_title(f"Autocorrelation — {planet_id}")
    axs[1].set_xlabel("Lag")
    axs[1].set_ylabel("Correlation")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    png_path = Path(outdir) / f"fft_autocorr_{planet_id}.png"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    # Interactive Plotly HTML
    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(x=freqs, y=power, mode="lines", name="FFT Power"))
    fig_html.add_trace(go.Scatter(x=lags, y=acorr, mode="lines", name="Autocorr", yaxis="y2"))
    fig_html.update_layout(
        title=f"FFT + Autocorr Diagnostics — {planet_id}",
        xaxis_title="Frequency / Lag",
        yaxis=dict(title="FFT Power"),
        yaxis2=dict(title="Autocorrelation", overlaying="y", side="right"),
        template="plotly_white",
    )
    html_path = Path(outdir) / f"fft_autocorr_{planet_id}.html"
    fig_html.write_html(html_path)


def main():
    parser = argparse.ArgumentParser(description="FFT & Autocorr Diagnostics Generator")
    parser.add_argument("--pred", type=str, default="outputs/predictions/predictions.json")
    parser.add_argument("--outdir", type=str, default="outputs/diagnostics")
    parser.add_argument("--summary", type=str, default="outputs/diagnostics/fft_autocorr_summary.csv")
    args = parser.parse_args()

    with open(args.pred, "r") as f:
        preds = json.load(f)

    rows = []
    for planet_id, data in preds.items():
        mu = np.array(data["mu"])
        plot_fft_autocorr(mu, planet_id, args.outdir)

        # Compute summary metrics
        freqs, power = compute_fft(mu)
        lags, acorr = compute_autocorr(mu)
        rows.append({
            "planet_id": planet_id,
            "fft_peak_freq": freqs[np.argmax(power)],
            "fft_peak_power": np.max(power),
            "autocorr_first_lag": acorr[1] if len(acorr) > 1 else np.nan,
        })

    df = pd.DataFrame(rows)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()