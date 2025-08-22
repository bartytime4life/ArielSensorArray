#!/usr/bin/env python3
"""
generate_fft_autocorr.py

Compute the FFT power spectrum and (normalized) autocorrelation of a 1D signal,
optionally apply detrending/windowing, and save results to disk.

Features
- Robust CLI (file or stdin)
- Optional detrend and window (hann, hamming, blackman, boxcar)
- Autocorrelation via FFT (O(N log N)), normalized to R[0] = 1
- Power spectrum in physical frequency units when --fs is provided
- CSV outputs + optional PNG plots
- Reproducible (fixed seed only used for demo generation)

Usage
------
# From a CSV with one column of samples:
python generate_fft_autocorr.py --input signal.csv --fs 1000 --window hann --detrend
python generate_fft_autocorr.py --input signal.csv --fs 1000 --plot

# Read from stdin (one sample per line):
cat signal.txt | python generate_fft_autocorr.py --fs 500 --output-dir out --plot

# Generate a synthetic demo signal:
python generate_fft_autocorr.py --demo --fs 200 --duration 2.0 --freqs 10 40 --snr 5 --plot
"""

import argparse
import sys
import os
import math
import csv
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import get_window, detrend as sp_detrend
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _read_1d_series(path: Optional[str]) -> np.ndarray:
    """Read a single-column numeric series from CSV/TSV or newline-delimited text.
    If path is None, read from stdin."""
    if path is None or path == "-":
        data = sys.stdin.read().strip().splitlines()
        vals = [float(x) for x in data if x.strip() != ""]
        return np.asarray(vals, dtype=float)

    # Heuristic: try CSV/TSV, else plain text
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv", ".tsv"}:
        delim = "," if ext == ".csv" else "\t"
        vals: List[float] = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                if not row:
                    continue
                # Use first column
                try:
                    vals.append(float(row[0]))
                except ValueError:
                    continue
        return np.asarray(vals, dtype=float)
    else:
        with open(path, "r") as f:
            vals = [float(x) for x in f.read().strip().splitlines() if x.strip() != ""]
        return np.asarray(vals, dtype=float)


def _maybe_detrend(x: np.ndarray, do: bool) -> np.ndarray:
    if not do:
        return x
    if _HAS_SCIPY:
        return sp_detrend(x, type="linear")
    # Fallback: remove linear trend via least squares
    n = len(x)
    t = np.arange(n, dtype=float)
    A = np.column_stack((t, np.ones(n)))
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    trend = A @ beta
    return x - trend


def _apply_window(x: np.ndarray, name: Optional[str]) -> Tuple[np.ndarray, float]:
    """Apply window; return (windowed_signal, window_power_correction)."""
    if not name or name.lower() in ("none", "boxcar"):
        w = np.ones_like(x)
    else:
        if not _HAS_SCIPY:
            raise RuntimeError("Windowing requires SciPy (scipy.signal.get_window).")
        w = get_window(name.lower(), len(x), fftbins=True)
    xw = x * w
    # Power correction factor so average signal power is preserved approximately
    corr = (np.sum(w**2) / len(w))
    return xw, corr


def power_spectrum(x: np.ndarray, fs: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return one-sided power spectrum (magnitude squared) and frequency vector.
    If fs is None, frequencies are in FFT bins (0..N/2).
    """
    n = len(x)
    # Next power of two (optional; comment out to use raw N)
    nfft = int(2 ** math.ceil(math.log2(max(1, n))))
    X = np.fft.rfft(x, n=nfft)
    Pxx = (np.abs(X) ** 2) / n  # energy or power proxy
    if fs is None:
        freqs = np.fft.rfftfreq(nfft, d=1.0)
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, Pxx


def autocorr_via_fft(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute full autocorrelation via FFT, return non-negative lags [0..N-1]."""
    n = len(x)
    nfft = int(2 ** math.ceil(math.log2(2 * n - 1)))
    X = np.fft.rfft(x, n=nfft)
    S = X * np.conj(X)  # power spectral density (up to scale)
    r = np.fft.irfft(S, n=nfft)
    r = r[:n]  # non-negative lags
    if normalize and r[0] != 0:
        r = r / r[0]
    return r


def save_csv(path: str, header: Tuple[str, str], a: np.ndarray, b: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ai, bi in zip(a, b):
            writer.writerow([ai, bi])


def plot_results(
    t: Optional[np.ndarray],
    x: np.ndarray,
    freqs: np.ndarray,
    Pxx: np.ndarray,
    r: np.ndarray,
    fs: Optional[float],
    outdir: str,
    prefix: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Time-domain
    plt.figure(figsize=(10, 3))
    if t is None:
        plt.plot(x, lw=1)
        plt.xlabel("Sample")
    else:
        plt.plot(t, x, lw=1)
        plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Signal")
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_signal.png"), dpi=150)
    plt.close()

    # Power spectrum
    plt.figure(figsize=(10, 3))
    plt.semilogy(freqs, np.maximum(Pxx, 1e-20), lw=1)
    plt.xlabel("Frequency [Hz]" if fs is not None else "Frequency bin")
    plt.ylabel("Power")
    plt.title("One-sided Power Spectrum")
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_spectrum.png"), dpi=150)
    plt.close()

    # Autocorrelation
    lags = np.arange(len(r))
    x_axis = lags / fs if fs is not None else lags
    plt.figure(figsize=(10, 3))
    plt.plot(x_axis, r, lw=1)
    plt.xlabel("Lag [s]" if fs is not None else "Lag [samples]")
    plt.ylabel("Autocorr")
    plt.title("Autocorrelation (non-negative lags)")
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_autocorr.png"), dpi=150)
    plt.close()


def generate_demo(fs: float, duration: float, freqs: List[float], snr: float, seed: int = 1234) -> np.ndarray:
    """Generate a synthetic multi-tone + white noise signal at a given SNR (dB)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs, dtype=float)
    x = np.zeros_like(t)
    for f in freqs:
        x += np.sin(2 * np.pi * f * t)
    # Normalize signal power to 1
    sig_power = np.mean(x**2) if np.any(x) else 1.0
    x = x / math.sqrt(sig_power)
    # Add noise for requested SNR (dB): SNR = 10 log10(Ps/Pn)
    if snr is not None:
        pn = 1.0 / (10 ** (snr / 10.0))  # since Ps=1 after normalization
        noise = rng.normal(0.0, math.sqrt(pn), size=len(t))
        x = x + noise
    return x


def main():
    p = argparse.ArgumentParser(
        description="Compute FFT power spectrum and autocorrelation of a 1D signal."
    )
    p.add_argument("--input", "-i", type=str, default=None,
                   help="Path to input series (CSV/TSV/plain). Omit or '-' to read from stdin. Use --demo to generate.")
    p.add_argument("--fs", type=float, default=None,
                   help="Sampling rate in Hz (improves frequency/lag axes).")
    p.add_argument("--detrend", action="store_true",
                   help="Remove linear trend before analysis.")
    p.add_argument("--window", type=str, default="hann",
                   help="Window name (hann, hamming, blackman, boxcar). Requires SciPy for non-boxcar.")
    p.add_argument("--normalize-ac", action="store_true",
                   help="Normalize autocorrelation so r[0] = 1 (recommended).")
    p.add_argument("--output-dir", "-o", type=str, default="outputs",
                   help="Directory to write CSV/plots.")
    p.add_argument("--prefix", type=str, default="result",
                   help="Filename prefix for outputs.")
    p.add_argument("--plot", action="store_true",
                   help="Save PNG plots (signal, spectrum, autocorr).")
    # Demo options
    p.add_argument("--demo", action="store_true",
                   help="Generate a synthetic signal instead of reading input.")
    p.add_argument("--duration", type=float, default=2.0,
                   help="Demo duration [s].")
    p.add_argument("--freqs", type=float, nargs="*", default=[10.0, 40.0],
                   help="Demo tone frequencies [Hz].")
    p.add_argument("--snr", type=float, default=10.0,
                   help="Demo SNR in dB (signal power normalized to 1).")
    args = p.parse_args()

    # Acquire signal
    if args.demo:
        if args.fs is None:
            raise SystemExit("When using --demo, please provide --fs.")
        x = generate_demo(fs=args.fs, duration=args.duration, freqs=args.freqs, snr=args.snr)
        t = np.arange(len(x)) / args.fs
    else:
        x = _read_1d_series(args.input)
        t = (np.arange(len(x)) / args.fs) if args.fs is not None else None

    if len(x) < 2:
        raise SystemExit("Need at least 2 samples.")

    # Preprocess
    x = _maybe_detrend(x, args.detrend)
    xw, power_corr = _apply_window(x, args.window)

    # FFT power spectrum
    freqs, Pxx = power_spectrum(xw, fs=args.fs)
    # compensate for window power loss
    if power_corr > 0:
        Pxx = Pxx / power_corr

    # Autocorrelation (normalize recommended)
    r = autocorr_via_fft(xw, normalize=args.normalize_ac)

    # Save CSVs
    os.makedirs(args.output_dir, exist_ok=True)
    save_csv(os.path.join(args.output_dir, f"{args.prefix}_spectrum.csv"),
             ("frequency_hz" if args.fs is not None else "frequency_bin", "power"),
             freqs, Pxx)
    lags = np.arange(len(r))
    lag_axis = (lags / args.fs) if args.fs is not None else lags
    save_csv(os.path.join(args.output_dir, f"{args.prefix}_autocorr.csv"),
             ("lag_s" if args.fs is not None else "lag_samples", "autocorr"),
             lag_axis, r)

    # Save signal CSV too (handy for provenance)
    sig_axis = (t if t is not None else np.arange(len(x)))
    save_csv(os.path.join(args.output_dir, f"{args.prefix}_signal.csv"),
             ("time_s" if t is not None else "sample", "amplitude"),
             sig_axis, x)

    # Plots
    if args.plot:
        plot_results(t, x, freqs, Pxx, r, args.fs, args.output_dir, args.prefix)

    print(f"[OK] Wrote outputs to: {os.path.abspath(args.output_dir)}")
    print(f"     Files: {args.prefix}_signal.csv, {args.prefix}_spectrum.csv, {args.prefix}_autocorr.csv")
    if args.plot:
        print(f"     Plots: {args.prefix}_signal.png, {args.prefix}_spectrum.png, {args.prefix}_autocorr.png")


if __name__ == "__main__":
    main()