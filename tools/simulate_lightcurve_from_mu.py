#!/usr/bin/env python3
"""
simulate_lightcurve_from_mu.py
--------------------------------
SpectraMind V50 • NeurIPS Ariel Data Challenge 2025

Upgraded, CLI-first simulator that generates a (time × wavelength) lightcurve
cube given a transmission spectrum μ(λ) (i.e., depth per channel), adding
deterministic instrument systematics (drift), correlated jitter, and photon
noise. Produces fully reproducible outputs with rich logs.

Highlights
- Typer CLI with --help and sane defaults (no notebooks needed).
- Deterministic seeding for exact reproducibility across runs.
- Physics-aware transit shape (small-planet + limb darkening along the chord).
- Correlated jitter via Ornstein–Uhlenbeck (OU) process.
- Vectorized, fast NumPy implementation; zero heavy external deps.
- Saves NPZ (time, wavelengths, flux cube) and optional PNG diagnostics plot.
- JSONL event log for machine-readable telemetry + pretty Rich console logs.

Usage
------
$ python simulate_lightcurve_from_mu.py simulate \
    --mu-file data/mu_depth_ppm.npy \
    --wavelengths-file data/wavelengths_nm.npy \
    --out-npz out/sim_lightcurve.npz \
    --plot out/sim_lightcurve.png \
    --duration-h 6 --cadence-s 10 \
    --period-d 3.14159 --t0-d 0.25 \
    --rp-rs-ref 0.1 --impact 0.3 \
    --u1 0.3 --u2 0.2 \
    --jitter-ppm 150 --jitter-tau-s 120.0 \
    --photon-ppm 200 \
    --drift-ppm-per-h 30 \
    --seed 42

Inputs
------
- μ(λ): 1D array of transit depth per wavelength channel (ppm). Typical length: 283.
- wavelengths: 1D array (same length as μ), in nm (or any units, used for metadata).

Outputs
-------
- NPZ with arrays:
    time_d              : (T,)          time in days
    wavelengths         : (L,)          wavelength grid
    flux                : (T,) or (T,L) simulated lightcurve(s)
    flux_white          : (T,)          white-light curve (always present)
    meta                : dict           configuration snapshot (JSON-serializable)
- Optional PNG quicklook plot.
- JSONL telemetry log (events.jsonl) + human-friendly run_log.md in the same folder as NPZ.

Notes
-----
This uses the small-planet approximation with a limb-darkened local chord
intensity. In-transit occulted intensity is approximated by I(mu_chord)/<I>,
and the flux deficit scales with depth(λ) ≈ μ_ppm(λ) × 1e-6. This captures
chromatic depth variation while keeping runtime light and avoiding heavy
dependencies (e.g., BATMAN). For most diagnostic and augmentation scenarios,
the approximation is more than sufficient.

Author: SpectraMind V50 Architect
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import typer

try:
    # Optional pretty console; degrade gracefully if not present
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

try:
    import matplotlib.pyplot as plt
    MPL = True
except Exception:
    MPL = False


# ----------------------------
# Dataclasses & Config
# ----------------------------

@dataclass
class StellarTransitConfig:
    period_d: float = 3.0          # orbital period (days)
    t0_d: float = 0.25             # mid-transit epoch (days)
    rp_rs_ref: float = 0.1         # reference Rp/Rs (used only for ingress smooth time)
    impact: float = 0.2            # impact parameter (0=center)
    a_rs: float = 12.0             # scaled semi-major axis (kepler-ish, only for kinematics)
    inc_deg: float = 89.0          # inclination (deg) ~ used for z(t) along chord (approx)
    u1: float = 0.3                # limb-darkening coefficient (quadratic)
    u2: float = 0.2                # limb-darkening coefficient (quadratic)


@dataclass
class NoiseConfig:
    jitter_ppm: float = 100.0      # OU jitter std (ppm)
    jitter_tau_s: float = 120.0    # OU timescale (seconds)
    photon_ppm: float = 200.0      # white noise (ppm)
    drift_ppm_per_h: float = 0.0   # linear drift (ppm/hour)


@dataclass
class TimeConfig:
    duration_h: float = 6.0        # total simulation duration (hours)
    cadence_s: float = 10.0        # cadence (seconds)


@dataclass
class IOConfig:
    mu_file: str = "mu_depth_ppm.npy"           # 1D ppm
    wavelengths_file: str = "wavelengths_nm.npy"
    out_npz: str = "sim_lightcurve.npz"
    out_plot: Optional[str] = None
    out_dir: Optional[str] = None  # if set, overrides outputs' parent folder


@dataclass
class SimConfig:
    stellar: StellarTransitConfig = StellarTransitConfig()
    noise: NoiseConfig = NoiseConfig()
    time: TimeConfig = TimeConfig()
    io: IOConfig = IOConfig()
    per_wavelength: bool = True            # output (T,L) cube; otherwise returns only white
    seed: Optional[int] = 42               # RNG seed for reproducibility
    ingress_smooth_factor: float = 1.0     # smoothing factor for ingress/egress (>= 0)
    # If >0, the transit box edges are smoothed by convolving with a Gaussian of sigma equal to
    # (... planet diameter crossing time ...) * factor; 0 => no smoothing.


# ----------------------------
# Utility: Logging (human + JSONL)
# ----------------------------

def _log_event(events_path: str, **payload):
    os.makedirs(os.path.dirname(events_path), exist_ok=True)
    payload = {"ts": datetime.utcnow().isoformat() + "Z", **payload}
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _human(msg: str):
    if RICH and console is not None:
        console.log(msg)
    else:
        print(msg)


# ----------------------------
# Core Physics Helpers
# ----------------------------

def quadratic_ld_intensity(mu: float, u1: float, u2: float) -> float:
    """
    Quadratic limb darkening local intensity normalized to I_center = 1:
    I(μ) / I(0) = 1 - u1*(1-μ) - u2*(1-μ)^2
    We renormalize by the disk-averaged intensity so that <I> = 1.
    """
    # disk-averaged intensity for quadratic LD is 1 - u1/3 - u2/6 (relative to I_center)
    I_center = 1.0
    I_mu = I_center * (1.0 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)
    I_avg = I_center * (1.0 - u1 / 3.0 - u2 / 6.0)
    return I_mu / I_avg


def projected_separation(time_d: np.ndarray,
                         period_d: float,
                         t0_d: float,
                         a_rs: float,
                         inc_deg: float) -> np.ndarray:
    """
    Projected center-to-center separation z(t) in stellar radii (circular orbit).
    For a circular orbit: true anomaly f(t) ~ 2π (t - t0)/P, then
    z(t) = a/Rs * sqrt( sin^2(f) * cos^2(i) + cos^2(f) )
    Reference: standard exoplanet transit geometry (approx).
    """
    inc = np.deg2rad(inc_deg)
    n = 2.0 * np.pi / period_d
    f = n * (time_d - t0_d)  # phase angle
    sinf = np.sin(f)
    cosf = np.cos(f)
    z = a_rs * np.sqrt((sinf ** 2) * (np.cos(inc) ** 2) + (cosf ** 2))
    return z


def small_planet_transit_shape(time_d: np.ndarray,
                               cfg: StellarTransitConfig,
                               rp_rs_lambda: Optional[np.ndarray] = None,
                               ingress_smooth: float = 0.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute a chromatic transit shape using small-planet approximation:
    - If rp_rs_lambda is None, returns a white transit mask (unit depth) curve,
      where 'depth' should be multiplied externally.
    - If rp_rs_lambda is provided (L,), returns a (T, L) chromatic mask.

    Occulted flux fraction δ(t, λ) ≈ (Rp/Rs)^2(λ) * I_chord(t)/<I>,
    where I_chord is the local (limb-darkened) intensity at the chord.
    We only apply deficit when z(t) < 1 (planet center within stellar disk).

    Ingress/egress smoothing: Gaussian blur on δ(t,·) edges with sigma (in time samples)
    equal to ingress_smooth samples, approximating finite-planet edge.
    """
    T = time_d.size
    z = projected_separation(time_d, cfg.period_d, cfg.t0_d, cfg.a_rs, cfg.inc_deg)

    # mu_chord from z (μ = cos θ = sqrt(1 - r^2) for ray at radius r), but at mid-chord we take r = impact approx.
    # We approximate local mu along transverse chord using the instantaneous "impact-like" radial distance:
    r = np.clip(z, 0.0, 1.0)  # clamp inside the disk for intensity calc
    mu_chord = np.sqrt(np.clip(1.0 - r ** 2, 0.0, 1.0))
    I_loc = quadratic_ld_intensity(mu_chord, cfg.u1, cfg.u2)  # (T,)

    in_disk = (z < 1.0).astype(float)  # small-planet step mask

    if rp_rs_lambda is None:
        # Unit-depth lightcurve shape (white). Multiply by scalar depth later.
        delta = in_disk * I_loc  # shape (T,)
        if ingress_smooth > 0.0:
            delta = _gaussian_blur1d(delta, sigma_samples=ingress_smooth)
        return delta, None

    # Chromatic: (T, L)
    rp2 = (rp_rs_lambda ** 2)[None, :]  # (1, L)
    delta_T = (in_disk * I_loc)[:, None] * rp2  # (T,L)

    if ingress_smooth > 0.0:
        delta_T = _gaussian_blur2d_time(delta_T, sigma_samples=ingress_smooth)

    return delta_T, rp2  # rp2 is returned for convenience


def _gaussian_blur1d(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0.0:
        return x
    # 6σ kernel
    L = int(max(3, math.ceil(6 * sigma_samples)))
    t = np.arange(-L, L + 1)
    g = np.exp(-0.5 * (t / sigma_samples) ** 2)
    g /= g.sum()
    return np.convolve(x, g, mode="same")


def _gaussian_blur2d_time(X: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0.0:
        return X
    L = int(max(3, math.ceil(6 * sigma_samples)))
    t = np.arange(-L, L + 1)
    g = np.exp(-0.5 * (t / sigma_samples) ** 2)
    g /= g.sum()
    # Convolve along time axis for each channel (separable)
    from numpy.lib.stride_tricks import sliding_window_view
    pad_width = ((L, L), (0, 0))
    Xp = np.pad(X, pad_width=pad_width, mode="edge")
    # sliding windows over time
    sw = sliding_window_view(Xp, window_shape=(2 * L + 1, 1))
    # sw shape: (T, 1, 2L+1, L) — simplify by manual conv
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i, :] = (Xp[i:i + 2 * L + 1, :] * g[:, None]).sum(axis=0)
    return out


def ou_jitter(times_s: np.ndarray, tau_s: float, sigma_ppm: float, rng: np.random.Generator) -> np.ndarray:
    """
    Ornstein–Uhlenbeck process sampled at arbitrary cadence using exact discretization.
    dx = - (1/tau) x dt + sqrt(2 sigma^2 / tau) dW
    For non-uniform dt, use:
    x_{n+1} = x_n e^{-dt/tau} + N(0, sigma^2 (1 - e^{-2 dt/tau}))
    """
    x = np.zeros_like(times_s)
    if times_s.size == 0:
        return x
    var = sigma_ppm ** 2
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        if dt <= 0:
            x[i] = x[i - 1]
            continue
        phi = np.exp(-dt / tau_s)
        var_dt = var * (1.0 - phi ** 2)
        x[i] = phi * x[i - 1] + rng.normal(0.0, np.sqrt(max(var_dt, 0.0)))
    return x


def linear_drift_ppm(times_h: np.ndarray, slope_ppm_per_h: float) -> np.ndarray:
    return slope_ppm_per_h * (times_h - times_h.mean())


# ----------------------------
# I/O helpers
# ----------------------------

def load_mu_and_wavelengths(mu_file: str, wavelengths_file: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mu = np.load(mu_file)
    mu = np.squeeze(mu).astype(float)
    if mu.ndim != 1:
        raise ValueError(f"μ(λ) must be 1D; got shape {mu.shape}")
    wl = None
    if wavelengths_file is not None and os.path.exists(wavelengths_file):
        wl = np.load(wavelengths_file)
        wl = np.squeeze(wl).astype(float)
        if wl.ndim != 1 or wl.shape[0] != mu.shape[0]:
            raise ValueError("wavelengths must be 1D and same length as μ(λ)")
    return mu, wl


def make_time_grid(cfg_time: TimeConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(round(cfg_time.duration_h * 3600.0 / cfg_time.cadence_s))
    n = max(3, n)
    times_s = np.arange(n) * cfg_time.cadence_s
    t0_s = times_s[n // 2]
    # center around 0 for numerical niceness
    times_s = times_s - t0_s
    times_h = times_s / 3600.0
    times_d = times_s / (86400.0)
    return times_d, times_h, times_s


# ----------------------------
# Main Simulation
# ----------------------------

def simulate(cfg: SimConfig) -> dict:
    # RNG & telemetry
    rng = np.random.default_rng(cfg.seed)

    out_folder = cfg.io.out_dir or os.path.dirname(cfg.io.out_npz) or "."
    os.makedirs(out_folder, exist_ok=True)
    events_path = os.path.join(out_folder, "events.jsonl")
    runlog_path = os.path.join(out_folder, "run_log.md")

    _log_event(events_path, event="start", config=asdict(cfg))
    _human("[bold]SpectraMind V50[/bold]: simulate_lightcurve_from_mu starting...") if RICH else None

    # Load μ(λ) depth (ppm) and wavelengths
    mu_ppm, wavelengths = load_mu_and_wavelengths(cfg.io.mu_file, cfg.io.wavelengths_file)
    L = mu_ppm.size

    if RICH:
        table = Table("Array", "Length", "Path/Info")
        table.add_row("μ(λ)", str(L), cfg.io.mu_file)
        table.add_row("λ (opt.)", str(L if wavelengths is not None else 0),
                      (cfg.io.wavelengths_file or "None"))
        console.print(table)

    # Time grid
    t_d, t_h, t_s = make_time_grid(cfg.time)
    T = t_d.size

    # Convert μ_ppm(λ) to fractional depth per channel
    depth_lambda = mu_ppm * 1e-6  # (L,)

    # From depth to Rp/Rs per channel (small-planet): depth ≈ (Rp/Rs)^2
    rp_rs_lambda = np.sqrt(np.clip(depth_lambda, 0.0, None))  # (L,)

    # Approximate ingress/egress smooth sigma in samples
    # crossing time ~ (2 * Rp/Rs_ref) * P / (π a/Rs)
    P_s = cfg.stellar.period_d * 86400.0
    cross_s = (2.0 * cfg.stellar.rp_rs_ref) * P_s / (np.pi * cfg.stellar.a_rs)
    sigma_ing = (cfg.ingress_smooth_factor * cross_s) / cfg.time.cadence_s

    # Compute chromatic transit shape (deficit)
    if cfg.per_wavelength:
        delta_TL, _ = small_planet_transit_shape(
            t_d, cfg.stellar, rp_rs_lambda=rp_rs_lambda, ingress_smooth=sigma_ing
        )  # (T,L)
        delta_white = delta_TL.mean(axis=1)
    else:
        # white shape with unit "shape", multiply by mean depth
        delta_T, _ = small_planet_transit_shape(
            t_d, cfg.stellar, rp_rs_lambda=None, ingress_smooth=sigma_ing
        )  # (T,)
        depth_white = depth_lambda.mean()
        delta_white = depth_white * delta_T  # (T,)
        delta_TL = None

    # Baseline flux = 1 - δ
    flux_white = 1.0 - delta_white  # (T,)
    if delta_TL is not None:
        flux_cube = 1.0 - delta_TL  # (T,L)
    else:
        flux_cube = None

    # Systematics & noise (ppm → fraction)
    drift = linear_drift_ppm(t_h, cfg.noise.drift_ppm_per_h) * 1e-6  # (T,)
    jitter = ou_jitter(t_s, cfg.noise.jitter_tau_s, cfg.noise.jitter_ppm, rng) * 1e-6  # (T,)
    photon = rng.normal(0.0, cfg.noise.photon_ppm * 1e-6, size=T)  # (T,)

    # Apply to white & cube (additive in flux)
    flux_white_sys = flux_white + drift + jitter + photon
    if flux_cube is not None:
        # Apply same systematics to each wavelength channel; optionally add independent photon noise per channel
        photon_L = rng.normal(0.0, cfg.noise.photon_ppm * 1e-6, size=(T, 1))
        flux_cube_sys = flux_cube + (drift + jitter)[:, None] + photon_L
    else:
        flux_cube_sys = None

    # Bundle outputs
    meta = {
        "config": asdict(cfg),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "T": T,
        "L": L,
        "period_d": cfg.stellar.period_d,
        "t0_d": cfg.stellar.t0_d,
        "seed": cfg.seed,
        "ingress_sigma_samples": float(sigma_ing),
    }

    out_npz = cfg.io.out_npz
    if cfg.io.out_dir:
        base = os.path.basename(out_npz)
        out_npz = os.path.join(cfg.io.out_dir, base)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    np.savez_compressed(
        out_npz,
        time_d=t_d,
        time_h=t_h,
        wavelengths=(wavelengths if wavelengths is not None else np.arange(L)),
        flux_white=flux_white_sys,
        flux=(flux_cube_sys if flux_cube_sys is not None else np.empty((0, 0))),
        meta=json.dumps(meta, indent=2),
    )

    _log_event(events_path, event="saved_npz", path=out_npz, T=T, L=L)

    # Plot (optional)
    if cfg.io.out_plot:
        out_plot = cfg.io.out_plot
        if cfg.io.out_dir:
            out_plot = os.path.join(cfg.io.out_dir, os.path.basename(out_plot))
        if MPL:
            fig, ax = plt.subplots(2 if cfg.per_wavelength else 1, 1, figsize=(10, 6), sharex=True)
            if not isinstance(ax, np.ndarray):
                ax = np.array([ax])

            ax[0].plot(t_h, flux_white_sys, color="k", lw=1.0, label="white")
            ax[0].set_ylabel("Flux (white)")
            ax[0].legend(loc="lower left")
            if cfg.per_wavelength:
                # show a few channels
                sel = np.linspace(0, L - 1, num=min(6, L), dtype=int)
                for j in sel:
                    ax[1].plot(t_h, flux_cube_sys[:, j], lw=0.6, alpha=0.8, label=f"ch {j}")
                ax[1].set_ylabel("Flux (channels)")
                ax[1].legend(ncols=3, fontsize=8, loc="lower left")
            ax[-1].set_xlabel("Time (hours, centered)")
            fig.tight_layout()
            fig.savefig(out_plot, dpi=150)
            plt.close(fig)
            _log_event(events_path, event="saved_plot", path=out_plot)
            _human(f"Saved plot → {out_plot}")
        else:
            _human("matplotlib not available; skipping plot.")

    # Human run-log
    with open(runlog_path, "a", encoding="utf-8") as f:
        f.write(f"# simulate_lightcurve_from_mu run @ {meta['created_utc']}\n\n")
        f.write(f"- Output: `{out_npz}`\n")
        if cfg.io.out_plot:
            f.write(f"- Plot  : `{cfg.io.out_plot}`\n")
        f.write(f"- T × L : {T} × {L}\n")
        f.write(f"- Seed  : {cfg.seed}\n")
        f.write(f"- Config:\n\n```json\n{json.dumps(meta['config'], indent=2)}\n```\n\n")

    _log_event(events_path, event="end", status="ok")
    _human("[green]Simulation complete.[/green]") if RICH else print("Simulation complete.")
    return {
        "time_d": t_d,
        "wavelengths": wavelengths if wavelengths is not None else np.arange(L),
        "flux_white": flux_white_sys,
        "flux": flux_cube_sys,
        "meta": meta,
        "out_npz": out_npz
    }


# ----------------------------
# CLI (Typer)
# ----------------------------

app = typer.Typer(add_completion=True, help="SpectraMind V50 · simulate_lightcurve_from_mu")


@app.command("simulate")
def cli_simulate(
    # IO
    mu_file: str = typer.Option(..., "--mu-file", "-m", help="Path to μ(λ) depth array (ppm). .npy 1D"),
    wavelengths_file: Optional[str] = typer.Option(None, "--wavelengths-file", help="Optional λ array (.npy)"),
    out_npz: str = typer.Option("sim_lightcurve.npz", "--out-npz", help="Output NPZ path"),
    out_plot: Optional[str] = typer.Option(None, "--plot", help="Optional quicklook PNG output"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Override output folder (NPZ/PNG to this dir)"),
    # Time
    duration_h: float = typer.Option(6.0, "--duration-h", help="Total duration (hours)"),
    cadence_s: float = typer.Option(10.0, "--cadence-s", help="Cadence (seconds)"),
    # Transit
    period_d: float = typer.Option(3.0, "--period-d", help="Orbital period (days)"),
    t0_d: float = typer.Option(0.25, "--t0-d", help="Mid-transit epoch (days, relative to time grid center)"),
    rp_rs_ref: float = typer.Option(0.1, "--rp-rs-ref", help="Reference Rp/Rs for ingress smoothing calc"),
    impact: float = typer.Option(0.2, "--impact", help="Impact parameter b"),
    a_rs: float = typer.Option(12.0, "--a-rs", help="Scaled semi-major axis a/Rs"),
    inc_deg: float = typer.Option(89.0, "--inc-deg", help="Inclination (deg)"),
    u1: float = typer.Option(0.3, "--u1", help="Quadratic LD u1"),
    u2: float = typer.Option(0.2, "--u2", help="Quadratic LD u2"),
    # Noise/Systematics
    jitter_ppm: float = typer.Option(100.0, "--jitter-ppm", help="OU jitter std (ppm)"),
    jitter_tau_s: float = typer.Option(120.0, "--jitter-tau-s", help="OU jitter timescale (s)"),
    photon_ppm: float = typer.Option(200.0, "--photon-ppm", help="Photon white noise (ppm)"),
    drift_ppm_per_h: float = typer.Option(0.0, "--drift-ppm-per-h", help="Linear drift slope (ppm/hour)"),
    # Controls
    per_wavelength: bool = typer.Option(True, "--per-wavelength/--white-only",
                                        help="Emit (T,L) cube vs just white-light"),
    seed: Optional[int] = typer.Option(42, "--seed", help="RNG seed for reproducibility"),
    ingress_smooth_factor: float = typer.Option(1.0, "--ingress-smooth-factor",
                                                help="≥0: scale for ingress/egress Gaussian smoothing"),
):
    """
    Generate a synthetic lightcurve cube from a transmission spectrum μ(λ).
    """
    # Prepare config
    cfg = SimConfig(
        stellar=StellarTransitConfig(
            period_d=period_d,
            t0_d=t0_d,
            rp_rs_ref=rp_rs_ref,
            impact=impact,
            a_rs=a_rs,
            inc_deg=inc_deg,
            u1=u1,
            u2=u2,
        ),
        noise=NoiseConfig(
            jitter_ppm=jitter_ppm,
            jitter_tau_s=jitter_tau_s,
            photon_ppm=photon_ppm,
            drift_ppm_per_h=drift_ppm_per_h,
        ),
        time=TimeConfig(duration_h=duration_h, cadence_s=cadence_s),
        io=IOConfig(mu_file=mu_file, wavelengths_file=wavelengths_file, out_npz=out_npz,
                    out_plot=out_plot, out_dir=out_dir),
        per_wavelength=per_wavelength,
        seed=seed,
        ingress_smooth_factor=max(0.0, ingress_smooth_factor),
    )

    # Summary to console
    if RICH:
        console.rule("[bold cyan]simulate_lightcurve_from_mu")
        cfg_tbl = Table("Section", "Key", "Value")
        for k, v in [
            ("Time", "duration_h", cfg.time.duration_h),
            ("Time", "cadence_s", cfg.time.cadence_s),
            ("Transit", "period_d", cfg.stellar.period_d),
            ("Transit", "t0_d", cfg.stellar.t0_d),
            ("Transit", "impact", cfg.stellar.impact),
            ("Transit", "a_rs", cfg.stellar.a_rs),
            ("LD", "u1", cfg.stellar.u1),
            ("LD", "u2", cfg.stellar.u2),
            ("Noise", "jitter_ppm", cfg.noise.jitter_ppm),
            ("Noise", "jitter_tau_s", cfg.noise.jitter_tau_s),
            ("Noise", "photon_ppm", cfg.noise.photon_ppm),
            ("Noise", "drift_ppm_per_h", cfg.noise.drift_ppm_per_h),
            ("Ctrl", "per_wavelength", cfg.per_wavelength),
            ("Ctrl", "seed", cfg.seed),
        ]:
            cfg_tbl.add_row(k, v if isinstance(v, str) else str(v), "")
        console.print(cfg_tbl)
    else:
        print("Config:", json.dumps(asdict(cfg), indent=2))

    # Run simulation
    simulate(cfg)


# Entry
if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        if RICH and console is not None:
            console.print_exception()
        else:
            print("ERROR:", repr(e), file=sys.stderr)
        sys.exit(1)
