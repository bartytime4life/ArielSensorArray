from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class ArielDataConfig:
    """
    Dataset configuration.

    Args:
        root: Base directory for real .pt tensors (ignored for toy).
        split: One of {"toy", "train", "val", "test"}; "toy" generates synthetic data.
        batch_size: Batch size for DataLoader.
        num_workers: DataLoader workers (0 = main process).
        bins: Number of AIRS spectral bins (challenge target width).
        fgs1_len: Toy FGS1 temporal length.
        n_samples: Number of samples per split (toy only).
        with_targets: Whether examples include 'target' in item dict.
        seed: RNG seed used for toy data and DataLoader generator (stability in CI).
    """
    root: str = "data/"
    split: str = "toy"         # "toy" | "train" | "val" | "test"
    batch_size: int = 8
    num_workers: int = 0
    bins: int = 283
    fgs1_len: int = 128
    n_samples: int = 64
    with_targets: bool = True
    seed: int = 1337


# ---------------------------------------------------------------------
# Toy dataset
# ---------------------------------------------------------------------
class ArielToyDataset(Dataset):
    """Synthetic but structured toy dataset for fast CI/dev loops."""

    def __init__(self, cfg: ArielDataConfig) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(cfg.seed)
        self.bins = int(cfg.bins)
        self.with_targets = bool(cfg.with_targets)

        # Inputs
        self.fgs1 = torch.randn(cfg.n_samples, 1, cfg.fgs1_len, generator=g)   # (N,1,T)
        self.airs = torch.randn(cfg.n_samples, 1, self.bins, generator=g)      # (N,1,B)

        # Optional synthetic targets with weak correlation to inputs
        if self.with_targets:
            z1 = self.fgs1.mean(dim=-1)                  # (N,1)
            z2 = self.airs.mean(dim=-1)                  # (N,1)
            base = z1 + z2                               # (N,1)
            self.target = base.repeat(1, self.bins) + 0.1 * torch.randn(
                cfg.n_samples, self.bins, generator=g
            )                                            # (N,B)
        else:
            self.target = None

    def __len__(self) -> int:
        return self.fgs1.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "fgs1": self.fgs1[idx],   # (1,T)
            "airs": self.airs[idx],   # (1,B)
        }
        if self.target is not None:
            item["target"] = self.target[idx]  # (B,)
        return item


# ---------------------------------------------------------------------
# Real dataset
# ---------------------------------------------------------------------
class ArielRealDataset(Dataset):
    """
    Loads tensors for a given split from disk:

      root/fgs1_{split}.pt : torch.Tensor of shape (N, 1, T)
      root/airs_{split}.pt : torch.Tensor of shape (N, 1, B)
      root/y_{split}.pt    : optional torch.Tensor of shape (N, B)

    Notes:
        - Will raise if required files are missing.
        - If 'y' exists, it's exposed as 'target' in items.
    """

    def __init__(self, cfg: ArielDataConfig) -> None:
        super().__init__()
        self.bins = int(cfg.bins)
        root = Path(cfg.root)
        split = cfg.split

        f_fgs1 = root / f"fgs1_{split}.pt"
        f_airs = root / f"airs_{split}.pt"
        f_y    = root / f"y_{split}.pt"

        if not f_fgs1.exists() or not f_airs.exists():
            raise FileNotFoundError(
                f"Expected files not found: {f_fgs1} and/or {f_airs}. "
                f"Provide .pt tensors or use split='toy'."
            )

        self.fgs1 = torch.load(f_fgs1, map_location="cpu")  # (N,1,T)
        self.airs = torch.load(f_airs, map_location="cpu")  # (N,1,B)

        # Basic shape checks
        if self.fgs1.ndim != 3 or self.fgs1.size(1) != 1:
            raise ValueError(f"fgs1 tensor must be (N,1,T); got {tuple(self.fgs1.shape)}")
        if self.airs.ndim != 3 or self.airs.size(1) != 1:
            raise ValueError(f"airs tensor must be (N,1,B); got {tuple(self.airs.shape)}")
        if self.airs.size(-1) != self.bins:
            raise ValueError(f"airs bins {self.airs.size(-1)} != cfg.bins {self.bins}")
        if self.fgs1.size(0) != self.airs.size(0):
            raise ValueError("fgs1 and airs must have the same N")

        # Optional targets
        self.target = None
        if f_y.exists():
            y = torch.load(f_y, map_location="cpu")  # (N,B)
            if y.ndim != 2 or y.size(0) != self.fgs1.size(0) or y.size(1) != self.bins:
                raise ValueError(f"y tensor must be (N,{self.bins}); got {tuple(y.shape)}")
            self.target = y

    def __len__(self) -> int:
        return self.fgs1.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "fgs1": self.fgs1[idx],  # (1,T)
            "airs": self.airs[idx],  # (1,B)
        }
        if self.target is not None:
            item["target"] = self.target[idx]  # (B,)
        return item


# ---------------------------------------------------------------------
# Collation & Loader builders
# ---------------------------------------------------------------------
def _collate(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Safe collation for dict samples; stacks keys present in the first item.
    Avoids surprises if some items omit 'target'.
    """
    batch = list(batch)
    if not batch:
        return {}
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [b[k] for b in batch if k in b]
        out[k] = torch.stack(tensors, dim=0)  # default_collate-like
    return out


def _loader_kwargs(cfg: ArielDataConfig, for_train: bool) -> dict:
    pin_mem = torch.cuda.is_available()
    return {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_mem,
        "persistent_workers": bool(cfg.num_workers > 0),
        "shuffle": bool(for_train),
        "drop_last": False,
        "collate_fn": _collate,
        "generator": torch.Generator().manual_seed(cfg.seed),
    }


def make_dataloader(cfg: ArielDataConfig) -> DataLoader:
    """Construct a single DataLoader for the given config."""
    if cfg.split == "toy":
        ds: Dataset = ArielToyDataset(cfg)
        for_train = True
    else:
        ds = ArielRealDataset(cfg)
        # convention: shuffle only for training split
        for_train = cfg.split.lower() in {"train", "tr", "training"}
    return DataLoader(ds, **_loader_kwargs(cfg, for_train=for_train))


def build_loaders(
    data_root: str = "data/",
    mode: str = "toy",
    batch_size: int = 8,
    num_workers: int = 0,
    bins: int = 283,
    fgs1_len: int = 128,
    n_samples: int = 64,
    with_targets: bool = True,
    seed: int = 1337,
) -> Tuple[DataLoader, DataLoader | None]:
    """
    Convenience builder.

    - toy: returns (train_loader, val_loader) with a simple 50/50 size split (independent RNG).
    - real: returns (train_loader, None); call again with split='val' for validation.

    For toy mode we generate two independent toy datasets so there's no leakage from shuffling.
    """
    if mode == "toy":
        train_cfg = ArielDataConfig(
            root=data_root, split="toy",
            batch_size=batch_size, num_workers=num_workers,
            bins=bins, fgs1_len=fgs1_len, n_samples=n_samples,
            with_targets=with_targets, seed=seed,
        )
        val_cfg = ArielDataConfig(
            root=data_root, split="toy",
            batch_size=batch_size, num_workers=num_workers,
            bins=bins, fgs1_len=fgs1_len, n_samples=max(8, n_samples // 2),
            with_targets=with_targets, seed=seed + 1,   # different RNG for val
        )
        return make_dataloader(train_cfg), make_dataloader(val_cfg)

    # real mode: give caller control over exact splits by calling twice
    train_cfg = ArielDataConfig(
        root=data_root, split="train",
        batch_size=batch_size, num_workers=num_workers,
        bins=bins, fgs1_len=fgs1_len, n_samples=n_samples,
        with_targets=with_targets, seed=seed,
    )
    return make_dataloader(train_cfg), None