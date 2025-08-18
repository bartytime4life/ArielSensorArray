from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TempCalibResult:
    T: float
    n_points: int
    used_targets: bool


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    var = sigma.clamp_min(1e-8) ** 2
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var)


def fit_temperature(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> float:
    """
    Closed-form T for sigma' = T * sigma minimizing Gaussian NLL:
        T^2 = mean( (y - mu)^2 / sigma^2 )
    """
    num = ((y - mu) ** 2 / (sigma.clamp_min(1e-8) ** 2)).mean().item()
    T = math.sqrt(max(num, 1e-8))
    return float(T)


def apply_temperature(sigma: torch.Tensor, T: float) -> torch.Tensor:
    return sigma * T


def calibrate_from_preds(
    preds_pt: str = "outputs/preds.pt",
    targets_pt: str | None = None,
    out_csv: str = "outputs/submission_calibrated.csv",
) -> TempCalibResult:
    """
    Loads preds {'mu': (N,B), 'sigma': (N,B)} from preds.pt and optional targets (N,B).
    If targets not present, keeps T=1.0 (no-op) but still writes calibrated CSV.
    """
    device = "cpu"
    preds: dict[str, torch.Tensor] = torch.load(preds_pt, map_location=device)
    mu: torch.Tensor = preds["mu"].to(device)  # (N, bins)
    sigma: torch.Tensor = preds["sigma"].to(device)  # (N, bins)

    T = 1.0
    used_targets = False
    if targets_pt and Path(targets_pt).exists():
        y: torch.Tensor = torch.load(targets_pt, map_location=device)
        # shape sanity
        if y.shape == mu.shape:
            T = fit_temperature(mu, sigma, y)
            used_targets = True

    sigma_cal = apply_temperature(sigma, T)

    # Write calibrated CSV (μ only, as required by submission)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = mu.shape[1]
    header = "planet_id," + ",".join([f"bin{i}" for i in range(bins)]) + "\n"
    lines = [header]
    for i in range(mu.shape[0]):
        row = [str(i)] + [f"{v:.6f}" for v in mu[i].tolist()]
        lines.append(",".join(row) + "\n")
    out_path.write_text("".join(lines))

    # Save calibrated preds alongside (handy for report/diagnostics)
    torch.save({"mu": mu, "sigma": sigma_cal, "T": T}, "outputs/preds_calibrated.pt")
    return TempCalibResult(T=T, n_points=mu.numel(), used_targets=used_targets)


if __name__ == "__main__":
    res = calibrate_from_preds()
    print(f"[calibrate] T={res.T:.4f} used_targets={res.used_targets} points={res.n_points}")
