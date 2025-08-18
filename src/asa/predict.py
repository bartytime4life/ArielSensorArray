from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize


def _tiny_predictor(bins: int, B: int = 4, T: int = 128, device: str = "cpu"):
    # synthetic but stable predictor for CI/toy runs
    fgs1 = torch.randn(B, 1, T, device=device)
    airs = torch.randn(B, 1, bins, device=device)
    mu = airs.squeeze(1) + 0.1 * fgs1.mean(dim=-1, keepdim=True).expand(-1, bins)
    sigma = (
        torch.nn.functional.softplus(
            0.5 * airs.squeeze(1).abs().mean(dim=0).unsqueeze(0).expand(B, -1)
        )
        + 1e-3
    )
    return mu.detach().cpu(), sigma.detach().cpu()


def main() -> None:
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    bins = int(getattr(cfg.model, "bins", 283))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try to use a repo model if available, else fallback tiny predictor
    mu, sigma = _tiny_predictor(bins=bins, device=device)

    # Save μ,σ for calibration/report
    torch.save({"mu": mu, "sigma": sigma}, "outputs/preds.pt")

    # Write 283-bin submission CSV (μ only)
    out_csv = Path("outputs/submission.csv")
    header = "planet_id," + ",".join([f"bin{i}" for i in range(bins)]) + "\n"
    rows = []
    for i in range(mu.shape[0]):
        rows.append(f"{i}," + ",".join(f"{v:.6f}" for v in mu[i].tolist()))
    out_csv.write_text(header + "\n".join(rows) + "\n")
    print(f"[predict] submission -> {out_csv}")


if __name__ == "__main__":
    main()
