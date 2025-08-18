from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ArielDataConfig:
    root: str = "data/"
    split: str = "toy"  # "toy" or "real"
    batch_size: int = 8
    num_workers: int = 0
    bins: int = 283  # AIRS spectral bins (challenge target)
    fgs1_len: int = 128  # toy time length for FGS1
    n_samples: int = 64  # toy dataset size per split
    with_targets: bool = True  # toy-only targets for training/calibration


class ArielToyDataset(Dataset):
    def __init__(self, cfg: ArielDataConfig, seed: int = 1337) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.bins = cfg.bins
        self.fgs1 = torch.randn(cfg.n_samples, 1, cfg.fgs1_len, generator=g)
        self.airs = torch.randn(cfg.n_samples, 1, cfg.bins, generator=g)
        # Create toy targets correlated with inputs for a meaningful training signal
        # target ~ linear combo of (mean over time of FGS1, AIRS) + noise
        z1 = self.fgs1.mean(dim=-1)  # (N,1)
        z2 = self.airs.mean(dim=-1)  # (N,1)
        base = z1 + z2  # (N,1)
        self.target = base.repeat(1, cfg.bins) + 0.1 * torch.randn(
            cfg.n_samples, cfg.bins, generator=g
        )

    def __len__(self) -> int:
        return self.fgs1.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "fgs1": self.fgs1[idx],  # (1, T)
            "airs": self.airs[idx],  # (1, bins)
        }
        # Include synthetic target for training/calibration convenience
        item["target"] = self.target[idx]  # (bins,)
        return item


class ArielRealDataset(Dataset):
    """
    Expects files (per split) under root:
      - fgs1_{split}.pt : torch tensor (N, 1, T)
      - airs_{split}.pt : torch tensor (N, 1, bins)
      - y_{split}.pt    : optional torch tensor (N, bins) with ground truth
    """

    def __init__(self, cfg: ArielDataConfig) -> None:
        super().__init__()
        root = Path(cfg.root)
        f_fgs1 = root / f"fgs1_{cfg.split}.pt"
        f_airs = root / f"airs_{cfg.split}.pt"
        if not f_fgs1.exists() or not f_airs.exists():
            raise FileNotFoundError(
                f"Expected {f_fgs1} and {f_airs}. " "Provide .pt tensors or use split='toy'."
            )
        self.fgs1 = torch.load(f_fgs1, map_location="cpu")
        self.airs = torch.load(f_airs, map_location="cpu")
        self.target = None
        f_y = root / f"y_{cfg.split}.pt"
        if f_y.exists():
            self.target = torch.load(f_y, map_location="cpu")

        # basic shape checks
        assert self.fgs1.dim() == 3 and self.fgs1.size(1) == 1
        assert self.airs.dim() == 3 and self.airs.size(1) == 1
        if self.target is not None:
            assert self.target.shape[0] == self.fgs1.shape[0]
            assert self.target.shape[1] == self.airs.size(-1)

    def __len__(self) -> int:
        return self.fgs1.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "fgs1": self.fgs1[idx],  # (1, T)
            "airs": self.airs[idx],  # (1, bins)
        }
        if self.target is not None:
            item["target"] = self.target[idx]  # (bins,)
        return item


def make_dataloader(cfg: ArielDataConfig) -> DataLoader:
    if cfg.split == "toy":
        ds: Dataset = ArielToyDataset(cfg)
    else:
        ds = ArielRealDataset(cfg)
    return DataLoader(
        ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, drop_last=False
    )


def build_loaders(
    data_root: str = "data/",
    mode: str = "toy",
    batch_size: int = 8,
    num_workers: int = 0,
    bins: int = 283,
    fgs1_len: int = 128,
    n_samples: int = 64,
    with_targets: bool = True,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Convenience for train/val splits in toy mode (50/50 split). In real mode,
    call this twice with split='train' and split='val'.
    """
    if mode == "toy":
        train_cfg = ArielDataConfig(
            root=data_root,
            split="toy",
            batch_size=batch_size,
            num_workers=num_workers,
            bins=bins,
            fgs1_len=fgs1_len,
            n_samples=n_samples,
            with_targets=with_targets,
        )
        val_cfg = ArielDataConfig(
            root=data_root,
            split="toy",
            batch_size=batch_size,
            num_workers=num_workers,
            bins=bins,
            fgs1_len=fgs1_len,
            n_samples=max(8, n_samples // 2),
            with_targets=with_targets,
        )
        return make_dataloader(train_cfg), make_dataloader(val_cfg)
    else:
        train_cfg = ArielDataConfig(
            root=data_root, split="train", batch_size=batch_size, num_workers=num_workers, bins=bins
        )
        return make_dataloader(train_cfg), None
