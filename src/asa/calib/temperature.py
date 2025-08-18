#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch


@dataclass
class TempCalibResult:
    T: float | torch.Tensor          # float (global) or (B,) tensor (perbin)
    n_points: int                    # total elements considered (finite only)
    used_targets: bool
    mode: Literal["global", "perbin"]
    bins: int
    rows: int


# ---------- math helpers ----------

def _finite_mask(*tensors: torch.Tensor) -> torch.Tensor:
    m = torch.ones_like(tensors[0], dtype=torch.bool)
    for t in tensors:
        m &= torch.isfinite(t)
    return m


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian NLL (no reduction)."""
    s = sigma.clamp_min(1e-8)
    var = s * s
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var)


@torch.no_grad()
def fit_temperature(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    mode: Literal["global", "perbin"] = "global",
    l2: float = 0.0,
) -> torch.Tensor | float:
    """
    Closed-form minimizer of mean NLL for sigma' = T * sigma.

    Without regularization:
        global:  T^2 = mean( (y - mu)^2 / sigma^2 )
        perbin:  T_b^2 = mean_over_rows( (y - mu)^2 / sigma^2 ) for each bin b

    With L2 regularization (toward T=1):
        argmin_T  E[z/T^2 + 2log T] + l2*(T-1)^2
        -> solved via 1D (or per-bin) Newton iterations (few steps, closed-ish).
    """
    mu = mu.to(dtype=torch.float64)
    sigma = sigma.to(dtype=torch.float64).clamp_min(1e-8)
    y = y.to(dtype=torch.float64)

    m = _finite_mask(mu, sigma, y)
    if not m.any():
        return 1.0 if mode == "global" else torch.ones(mu.shape[1], dtype=torch.float32)

    # z = (y - mu)^2 / sigma^2
    z = ((y - mu) ** 2) / (sigma ** 2)
    z = z[m]

    if mode == "global":
        if l2 <= 0:
            T2 = z.mean().item()
            return float(math.sqrt(max(T2, 1e-12)))
        # small Newton solve for l2>0
        # minimize f(T) = E[z/T^2 + 2 log T] + l2*(T-1)^2
        T = torch.tensor(1.0, dtype=torch.float64)
        Ez = z.mean()
        for _ in range(12):
            # f'(T) = E[-2z/T^3 + 2/T] + 2l2(T-1)
            g = (-2 * Ez / (T ** 3)) + (2 / T) + 2 * l2 * (T - 1)
            # f''(T) = E[6z/T^4 - 2/T^2] + 2l2
            h = (6 * Ez / (T ** 4)) - (2 / (T ** 2)) + 2 * l2
            step = g / h
            T = (T - step).clamp_min(1e-6)
            if abs(step.item()) < 1e-8:
                break
        return float(T.item())

    # per-bin
    B = mu.shape[1]
    T = torch.ones(B, dtype=torch.float64)
    # compute per-bin E[z]
    Z = ((y - mu) ** 2) / (sigma ** 2)
    # mask invalid
    Z[~_finite_mask(Z)] = float("nan")
    Ez = torch.nanmean(Z, dim=0)  # (B,)
    if l2 <= 0:
        T = torch.sqrt(torch.clamp(Ez, min=1e-12))
    else:
        # per-bin Newton
        T = T.clone()
        for b in range(B):
            tb = T[b]
            ez = Ez[b]
            if not torch.isfinite(ez):
                T[b] = 1.0
                continue
            for _ in range(12):
                g = (-2 * ez / (tb ** 3)) + (2 / tb) + 2 * l2 * (tb - 1)
                h = (6 * ez / (tb ** 4)) - (2 / (tb ** 2)) + 2 * l2
                step = g / h
                tb = (tb - step).clamp_min(1e-6)
                if float(abs(step)) < 1e-8:
                    break
            T[b] = tb
    return T.to(dtype=torch.float32)


def apply_temperature(sigma: torch.Tensor, T: float | torch.Tensor) -> torch.Tensor:
    if isinstance(T, torch.Tensor):
        # per-bin broadcast: (N,B) * (B,)
        return sigma * T.view(1, -1)
    return sigma * float(T)


# ---------- I/O helpers ----------

def _read_ids(ids_path: Path | None, n_rows: int) -> list[str]:
    if ids_path and ids_path.exists():
        ids = [s.strip() for s in ids_path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if ids:
            return ids
    return [str(i) for i in range(n_rows)]


def _write_submission_csv(out_csv: Path, ids: Sequence[str], mu: torch.Tensor) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        B = mu.shape[1]
        w.writerow(["planet_id", *[f"bin{i}" for i in range(B)]])
        for i, pid in enumerate(ids):
            w.writerow([pid, *[f"{x:.6f}" for x in mu[i].tolist()]])


# ---------- main API ----------

@torch.no_grad()
def calibrate_from_preds(
    preds_pt: str = "outputs/preds.pt",
    targets_pt: str | None = None,
    out_csv: str = "outputs/submission_calibrated.csv",
    ids_path: str | None = None,
    mode: Literal["global", "perbin"] = "global",
    l2: float = 0.0,
) -> TempCalibResult:
    """
    Load preds {'mu': (N,B), 'sigma': (N,B)} and optional targets (N,B).
    Fits T (global or per-bin). If no usable targets, uses T=1 (no-op).
    Always writes calibrated CSV of μ (unchanged), and saves
    'outputs/preds_calibrated.pt' with calibrated σ and T.

    Args:
        preds_pt: Path to preds.pt (expects keys 'mu', 'sigma').
        targets_pt: Optional targets (N,B); required for actual calibration.
        out_csv: Output calibrated submission CSV (μ only).
        ids_path: Optional text file with planet_ids (one per line). Defaults to 0..N-1.
        mode: 'global' for scalar T, 'perbin' for vector T (B,).
        l2: L2 regularization toward T=1.0 (0 disables).

    Returns:
        TempCalibResult with T, counts, and flags.
    """
    dev = "cpu"
    payload: dict = torch.load(preds_pt, map_location=dev)
    mu: torch.Tensor = payload["mu"].to(dev).to(torch.float32)       # (N,B)
    sigma: torch.Tensor = payload["sigma"].to(dev).to(torch.float32) # (N,B)

    used_targets = False
    T: float | torch.Tensor = 1.0

    if targets_pt and Path(targets_pt).exists():
        y: torch.Tensor = torch.load(targets_pt, map_location=dev).to(torch.float32)
        if y.shape == mu.shape and _finite_mask(mu, sigma, y).any():
            T = fit_temperature(mu, sigma, y, mode=mode, l2=l2)
            used_targets = True

    sigma_cal = apply_temperature(sigma, T)

    # Write calibrated CSV (μ only; submission schema)
    ids = _read_ids(Path(ids_path) if ids_path else None, n_rows=mu.shape[0])
    _write_submission_csv(Path(out_csv), ids, mu)

    # Save calibrated preds for diagnostics
    out_pt = Path("outputs/preds_calibrated.pt")
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": mu, "sigma": sigma_cal, "T": T}, out_pt)

    n_points = int(torch.isfinite(mu).sum().item())  # report finite count
    return TempCalibResult(T=T, n_points=n_points, used_targets=used_targets,
                           mode=mode, bins=mu.shape[1], rows=mu.shape[0])


# ---------- tiny CLI ----------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Temperature calibration for Gaussian outputs.")
    ap.add_argument("--preds", default="outputs/preds.pt")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--out-csv", default="outputs/submission_calibrated.csv")
    ap.add_argument("--ids", default=None, help="Optional path to planet_id list")
    ap.add_argument("--mode", choices=["global", "perbin"], default="global")
    ap.add_argument("--l2", type=float, default=0.0, help="L2 toward T=1.0 (0 = off)")
    return ap.parse_args(list(argv) if argv is not None else None)


def _main() -> None:
    a = _parse_args()
    res = calibrate_from_preds(
        preds_pt=a.preds,
        targets_pt=a.targets,
        out_csv=a.out_csv,
        ids_path=a.ids,
        mode=a.mode,
        l2=a.l2,
    )
    # nice single-line summary
    t_fmt = (f"{res.T:.6f}" if isinstance(res.T, float)
             else f"vector[{res.T.numel()}], mean={float(res.T.mean()):.6f}")
    print(f"[calibrate] mode={res.mode} T={t_fmt} used_targets={res.used_targets} "
          f"rows={res.rows} bins={res.bins} finite_points={res.n_points}")


if __name__ == "__main__":
    _main()