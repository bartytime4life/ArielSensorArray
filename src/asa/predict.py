#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

# Optional Hydra config (kept soft so the script runs without Hydra present)
try:
    from hydra import compose, initialize
    _HYDRA_OK = True
except Exception:  # pragma: no cover
    _HYDRA_OK = False


def _set_seed(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep algorithms fast in CI while still stable enough for RNG-based toy preds
    # torch.use_deterministic_algorithms(True)  # uncomment if you need full determinism


@torch.no_grad()
def _tiny_predictor(bins: int, n: int, T: int = 128, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Synthetic, stable-ish predictor for CI/toy runs.
    Returns:
        mu:    (n, bins)
        sigma: (n, bins)
    """
    # Random features whose statistics drive mean/uncertainty
    fgs1 = torch.randn(n, 1, T, device=device)
    airs = torch.randn(n, 1, bins, device=device)

    mu = airs.squeeze(1) + 0.1 * fgs1.mean(dim=-1, keepdim=True).expand(-1, bins)

    # per-bin magnitude -> positive stddev; broadcast to all rows
    base = 0.5 * airs.squeeze(1).abs().mean(dim=0).unsqueeze(0).expand(n, -1)
    sigma = torch.nn.functional.softplus(base) + 1e-3

    return mu.detach().cpu(), sigma.detach().cpu()


def _load_hydra_bins(config_path: str | None, config_name: str | None) -> int | None:
    if not _HYDRA_OK or not config_path or not config_name:
        return None
    # hydra.initialize chdir behavior is contained; we only extract a single value.
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
    # Try several common locations
    for key in ("bins", "model.bins", "data.bins"):
        node = cfg
        try:
            for part in key.split("."):
                node = node[part]
            val = int(node)
            if val > 0:
                return val
        except Exception:
            continue
    return None


def _read_ids(ids_path: Path) -> List[str]:
    ids: List[str] = []
    for line in ids_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    if not ids:
        raise ValueError(f"--ids file is empty: {ids_path}")
    return ids


def _write_submission_csv(out_csv: Path, ids: Sequence[str], mu: torch.Tensor) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["planet_id", *[f"bin{i}" for i in range(mu.shape[1])]])
        for idx, pid in enumerate(ids):
            # Guard against any accidental size mismatch
            row_mu = mu[idx].tolist()
            # format compactly but clearly
            w.writerow([pid, *[f"{x:.6f}" for x in row_mu]])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate toy predictions and submission CSV.")
    ap.add_argument("--bins", type=int, default=None, help="Number of spectral bins (default: Hydra or 283)")
    ap.add_argument("--n", type=int, default=None, help="Number of rows to predict (defaults to len(--ids) or 4)")
    ap.add_argument("--ids", type=Path, help="Path to file of planet_id, one per line; enforces row order/count")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory (default: outputs/)")
    ap.add_argument("--preds-pt", type=str, default="preds.pt", help='Filename for torch outputs (default: "preds.pt")')
    ap.add_argument("--csv", type=str, default="submission.csv", help='Filename for CSV (default: "submission.csv")')
    # Optional Hydra inputs (kept explicit so normal runs don’t require Hydra)
    ap.add_argument("--hydra-config-path", type=str, default=None, help="Hydra config path (e.g., ../configs)")
    ap.add_argument("--hydra-config-name", type=str, default=None, help="Hydra config name (e.g., config_v50)")
    return ap.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    _set_seed(1337)

    # Resolve bins
    bins = (
        args.bins
        or _load_hydra_bins(args.hydra_config_path, args.hydra_config_name)
        or 283
    )
    if bins <= 0:
        raise SystemExit(f"Invalid bins: {bins}")

    # Resolve ids / n
    ids: List[str]
    if args.ids:
        ids = _read_ids(args.ids)
        n = len(ids)
    else:
        n = args.n or 4
        ids = [str(i) for i in range(n)]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Predict
    mu, sigma = _tiny_predictor(bins=bins, n=n, device=device)

    # Write artifacts
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"mu": mu, "sigma": sigma}, out_dir / args.preds_pt)
    _write_submission_csv(out_dir / args.csv, ids=ids, mu=mu)

    print(f"[predict] wrote {out_dir/args.csv} and {out_dir/args.preds_pt}  "
          f"({n} rows x {bins} bins, device={device})")


if __name__ == "__main__":
    main()