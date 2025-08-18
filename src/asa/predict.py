from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize

from .pipeline.model_def import ArielModel  # to access σ if needed


def main() -> None:
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")

    bins = int(getattr(cfg.model, "bins", 283))
    # Run predictor to obtain μ, and reconstruct σ from the model on a fresh pass (or keep predictor returning both)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ArielModel(bins=bins).to(device)
    # If checkpoint exists, load it
    ckpt = Path("outputs/checkpoints/model.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Synthesize a small batch (B=4) to generate predictions (works for CI/toy)
    B, T, W = 4, 128, bins
    fgs1 = torch.randn(B, 1, T, device=device)
    airs = torch.randn(B, 1, W, device=device)
    with torch.no_grad():
        out = model({"fgs1": fgs1, "airs": airs})
        mu: torch.Tensor = out["mu"].detach().cpu()
        sigma: torch.Tensor = out["sigma"].detach().cpu()

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
