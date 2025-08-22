# src/data/loaders.py
# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
# -----------------------------------------------------------------------------
# This module builds reproducible, cache-aware PyTorch datasets and dataloaders
# for Ariel-like spatio-temporal spectral cubes and target transmission spectra.
#
# Highlights
# - Hydra-friendly: accepts a DictConfig (or dict) for all knobs/toggles
# - Deterministic: fixed seeding, worker_init_fn, and optional torch determinism
# - Cache-aware: memoizes preprocessed tensors on disk keyed by content+params
# - Flexible IO: supports .npy/.npz/.pt for cubes/targets and CSV manifest
# - Light physics: optional wavelength-wise detrend & per-channel normalization
# - Rich logging, clear errors, and a tight API: build_dataloaders(cfg) -> dict
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import io
import csv
import json
import math
import time
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required for data/loaders.py") from e

try:
    from omegaconf import DictConfig, OmegaConf  # Hydra-friendly, optional import
except Exception:
    DictConfig = dict  # type: ignore


# ----------------------------- Logging setup ---------------------------------

log = logging.getLogger("spectramind.data")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ------------------------------- Utilities -----------------------------------

def _as_dict(cfg: Union[DictConfig, dict]) -> dict:
    """Convert Hydra DictConfig to a plain dict (safe for hashing/printing)."""
    try:
        from omegaconf import OmegaConf
        if isinstance(cfg, DictConfig):
            return json.loads(OmegaConf.to_json(cfg))
    except Exception:
        pass
    return dict(cfg)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path, max_bytes: int = 2_000_000) -> str:
    """Hash a file (truncated read to be quick but stable for content-based keys)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        read = 0
        chunk = f.read(65536)
        while chunk and read < max_bytes:
            h.update(chunk)
            read += len(chunk)
            chunk = f.read(65536)
    h.update(str(path.stat().st_size).encode())
    h.update(str(int(path.stat().st_mtime)).encode())
    return h.hexdigest()


def _stable_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return _sha256_bytes(blob)


def _np_load_any(path: Path) -> np.ndarray:
    """Load .npy/.npz/.pt tensors into numpy arrays (zero-copy when possible)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".npz":
        with np.load(path, mmap_mode="r") as z:
            # prefer 'arr_0' if single, else stack values in sorted key order
            if len(z.files) == 1:
                return z[z.files[0]]
            return np.stack([z[k] for k in sorted(z.files)], axis=0)
    if suffix == ".pt":
        t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.Tensor):
            return t.numpy()
        raise ValueError(f"Unsupported .pt content in {path!s}: expected a tensor.")
    raise ValueError(f"Unsupported tensor file type: {path.name}")


def _maybe_make_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Set numpy/torch seeds for reproducibility."""
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _worker_init_fn(worker_id: int) -> None:
    """Deterministic worker seeding."""
    base_seed = torch.initial_seed() % (2**31 - 1)
    np.random.seed((base_seed + worker_id) % (2**31 - 1))


# ------------------------------- Disk Cache ----------------------------------

@dataclass
class DiskCache:
    root: Path

    def __post_init__(self) -> None:
        _maybe_make_dir(self.root)

    def key_for(self, *, manifest_hash: str, sample_id: str, params: dict) -> Path:
        key = _stable_key(manifest_hash, sample_id, params)
        return self.root / f"{key}.npy"

    def exists(self, path: Path) -> bool:
        return path.exists()

    def save(self, path: Path, array: np.ndarray) -> None:
        # Save atomically
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, path)

    def load(self, path: Path) -> np.ndarray:
        return np.load(path, mmap_mode="r")


# --------------------------- Manifest & Schema -------------------------------

@dataclass
class SampleRow:
    sample_id: str
    cube_path: Path
    target_path: Optional[Path]
    split: str
    meta: Dict[str, Any]


def _read_manifest(manifest_csv: Path) -> List[SampleRow]:
    """
    Expected CSV columns (header required):
        id,cube_path,target_path,split,meta_json
    - target_path may be empty for test/unlabeled
    - meta_json is optional; if missing or empty, meta={}
    """
    rows: List[SampleRow] = []
    with manifest_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "cube_path", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        for r in reader:
            sid = r["id"].strip()
            cube = Path(r["cube_path"]).expanduser().resolve()
            tgt = Path(r["target_path"]).expanduser().resolve() if "target_path" in r and r["target_path"] else None
            split = r["split"].strip().lower()
            meta = {}
            if "meta_json" in r and r["meta_json"]:
                try:
                    meta = json.loads(r["meta_json"])
                except Exception as e:
                    log.warning("Could not parse meta_json for id=%s: %s", sid, e)
            rows.append(SampleRow(sid, cube, tgt, split, meta))
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_csv}")
    return rows


def _manifest_hash(rows: List[SampleRow]) -> str:
    content = [(r.sample_id, str(r.cube_path), str(r.target_path) if r.target_path else None, r.split, r.meta)
               for r in rows]
    return _stable_key(content)


# ----------------------------- Preprocessing ---------------------------------

@dataclass
class PreprocessParams:
    detrend: bool = True
    normalize: bool = True
    eps: float = 1e-6


def detrend_wavelengthwise(cube: np.ndarray) -> np.ndarray:
    """
    Simple jitter/systematics removal: subtract per-wavelength median over time.
    cube: (T, W) or (T, W, *extras) -> returns same shape
    """
    if cube.ndim < 2:
        return cube
    # Assume (T, W) in the first two axes
    T, W = cube.shape[0], cube.shape[1]
    flat = cube.reshape(T, W, -1)
    med = np.median(flat, axis=0, keepdims=True)  # (1, W, F)
    flat = flat - med
    return flat.reshape(cube.shape)


def normalize_per_channel(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize per-wavelength channel (zero mean, unit variance) over time.
    """
    if cube.ndim < 2:
        return cube
    T, W = cube.shape[0], cube.shape[1]
    flat = cube.reshape(T, W, -1)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    flat = (flat - mean) / (std + eps)
    return flat.reshape(cube.shape)


def preprocess_cube(cube: np.ndarray, params: PreprocessParams) -> np.ndarray:
    x = cube
    if params.detrend:
        x = detrend_wavelengthwise(x)
    if params.normalize:
        x = normalize_per_channel(x, eps=params.eps)
    return x


# ------------------------------- Dataset -------------------------------------

class ArielSpectralDataset(Dataset):
    """
    Dataset for Ariel-like spectral cubes and target transmission spectra.

    Expected sample structure from manifest:
      - cube_path -> npy/npz/pt array of shape (T, W) or (T, W, C)
      - target_path (optional) -> npy/npz/pt array of shape (W,) or (W, K)
    """

    def __init__(
        self,
        rows: List[SampleRow],
        preprocess: PreprocessParams,
        cache: Optional[DiskCache] = None,
        manifest_hash: Optional[str] = None,
        max_time: Optional[int] = None,
        wavelength_axis: int = 1,
        dtype: str = "float32",
    ):
        self.rows = rows
        self.params = preprocess
        self.cache = cache
        self.manifest_hash = manifest_hash or "nohash"
        self.max_time = max_time
        self.wavelength_axis = wavelength_axis
        self.dtype = np.float32 if dtype == "float32" else np.float16

    def __len__(self) -> int:
        return len(self.rows)

    def _load_cube(self, path: Path) -> np.ndarray:
        x = _np_load_any(path).astype(self.dtype, copy=False)
        # Ensure (T, W, ...) layout with wavelength at axis 1
        if self.wavelength_axis != 1:
            # rotate axes: find where W is; here we assume input is (T, W, ...) already
            pass
        if self.max_time and x.shape[0] > self.max_time:
            x = x[: self.max_time]
        return x

    def _load_target(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        y = _np_load_any(path).astype(self.dtype, copy=False)
        return y

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        cache_tensor: Optional[np.ndarray] = None
        cache_path: Optional[Path] = None

        if self.cache is not None:
            params_dict = {"detrend": self.params.detrend,
                           "normalize": self.params.normalize,
                           "eps": self.params.eps,
                           "max_time": self.max_time}
            cache_path = self.cache.key_for(
                manifest_hash=self.manifest_hash,
                sample_id=r.sample_id,
                params=params_dict,
            )
            if self.cache.exists(cache_path):
                try:
                    cache_tensor = self.cache.load(cache_path)
                except Exception as e:
                    log.warning("Failed to load cache for %s: %s (recomputing)", r.sample_id, e)

        if cache_tensor is None:
            cube = self._load_cube(r.cube_path)
            cube = preprocess_cube(cube, self.params)
            if self.cache is not None and cache_path is not None:
                try:
                    self.cache.save(cache_path, cube)
                except Exception as e:
                    log.warning("Failed to save cache for %s: %s", r.sample_id, e)
        else:
            cube = cache_tensor

        target = self._load_target(r.target_path)

        sample = {
            "id": r.sample_id,
            "cube": torch.from_numpy(np.ascontiguousarray(cube)),       # (T, W, ...)
            "target": torch.from_numpy(np.ascontiguousarray(target)) if target is not None else None,
            "meta": r.meta,
        }
        return sample


# ------------------------------- Collation -----------------------------------

def pad_time_dim(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that pads the time dimension to the max length in the batch.
    Assumes 'cube' shape (T, W, ...). Targets are not padded (assumed (W,) or None).
    """
    ids = [b["id"] for b in batch]
    metas = [b["meta"] for b in batch]

    cubes = [b["cube"] for b in batch]
    T_max = max([c.shape[0] for c in cubes])
    W = cubes[0].shape[1]

    padded = []
    lengths = []
    for c in cubes:
        T = c.shape[0]
        lengths.append(T)
        if T == T_max:
            padded.append(c)
        else:
            pad_shape = (T_max - T, *c.shape[1:])
            pad = torch.zeros(pad_shape, dtype=c.dtype)
            padded.append(torch.cat([c, pad], dim=0))
    X = torch.stack(padded, dim=0)  # (B, T_max, W, ...)

    targets = [b["target"] for b in batch]
    if any(t is None for t in targets):
        Y = None
    else:
        Y = torch.stack(targets, dim=0)  # (B, W) or (B, W, K)

    return {
        "id": ids,
        "cube": X,
        "lengths": torch.tensor(lengths, dtype=torch.int32),
        "target": Y,
        "meta": metas,
    }


# ------------------------------- Builders ------------------------------------

def build_datasets(cfg: Union[DictConfig, dict]) -> Dict[str, ArielSpectralDataset]:
    """
    Build train/val/test datasets from a manifest CSV and preprocessing config.

    cfg.data:
      manifest: str (CSV path)
      cache_dir: Optional[str]
      split_names: {train: "train", val: "val", test: "test"}
      max_time: Optional[int]
      dtype: "float32" | "float16"
      preprocess:
        detrend: bool
        normalize: bool
        eps: float
    """
    dcfg = _as_dict(cfg).get("data", {})
    manifest = Path(dcfg.get("manifest", "")).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest}")

    rows = _read_manifest(manifest)
    mh = _manifest_hash(rows)

    cache_dir = dcfg.get("cache_dir", None)
    cache = DiskCache(Path(cache_dir)) if cache_dir else None

    pparams = PreprocessParams(
        detrend=bool(dcfg.get("preprocess", {}).get("detrend", True)),
        normalize=bool(dcfg.get("preprocess", {}).get("normalize", True)),
        eps=float(dcfg.get("preprocess", {}).get("eps", 1e-6)),
    )

    max_time = dcfg.get("max_time", None)
    dtype = dcfg.get("dtype", "float32")

    splits = dcfg.get("split_names", {"train": "train", "val": "val", "test": "test"})
    split_map = {"train": [], "val": [], "test": []}
    for r in rows:
        if r.split == splits.get("train", "train"):
            split_map["train"].append(r)
        elif r.split == splits.get("val", "val"):
            split_map["val"].append(r)
        elif r.split == splits.get("test", "test"):
            split_map["test"].append(r)

    datasets = {}
    for key in ["train", "val", "test"]:
        if split_map[key]:
            datasets[key] = ArielSpectralDataset(
                rows=split_map[key],
                preprocess=pparams,
                cache=cache,
                manifest_hash=mh,
                max_time=max_time,
                dtype=dtype,
            )
            log.info("Built %s dataset: %d samples", key, len(datasets[key]))
        else:
            log.info("No samples for split '%s' in manifest", key)

    return datasets


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    drop_last: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_time_dim,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


def build_dataloaders(cfg: Union[DictConfig, dict]) -> Dict[str, DataLoader]:
    """
    One-stop factory for train/val/test dataloaders.

    cfg.training.data_loader:
      train:
        batch_size: int
        shuffle: true
        num_workers: int
        pin_memory: true
        persistent_workers: true
        drop_last: true
      val: {...}
      test: {...}
    cfg.training.deterministic:
      enable: bool
      seed: int
    """
    tcfg = _as_dict(cfg).get("training", {})
    det = tcfg.get("deterministic", {"enable": True, "seed": 42})
    seed = int(det.get("seed", 42))
    set_global_seed(seed)
    if det.get("enable", True):
        try:
            torch.use_deterministic_algorithms(True)
            log.info("Torch deterministic algorithms enabled")
        except Exception as e:
            log.warning("Deterministic algorithms not fully supported: %s", e)

    g = torch.Generator()
    g.manual_seed(seed)

    datasets = build_datasets(cfg)

    loaders: Dict[str, DataLoader] = {}
    dl_cfg = tcfg.get("data_loader", {})
    for split in ["train", "val", "test"]:
        ds = datasets.get(split, None)
        if ds is None:
            continue
        scfg = dl_cfg.get(split, {})
        loaders[split] = build_dataloader(
            ds,
            batch_size=int(scfg.get("batch_size", 8)),
            shuffle=bool(scfg.get("shuffle", split == "train")),
            num_workers=int(scfg.get("num_workers", 4)),
            pin_memory=bool(scfg.get("pin_memory", True)),
            persistent_workers=bool(scfg.get("persistent_workers", True)),
            drop_last=bool(scfg.get("drop_last", split == "train")),
            generator=g,
        )
        log.info(
            "Built %s dataloader: batch_size=%d, workers=%d",
            split, int(scfg.get("batch_size", 8)), int(scfg.get("num_workers", 4))
        )

    return loaders


# ------------------------------- Example CLI ---------------------------------
# This block is optional; you can import the builders from train/predict scripts.
if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Quick sanity-check for loaders.py")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    # Minimal ad-hoc config for quick check (Hydra users ignore this)
    cfg = {
        "data": {
            "manifest": args.manifest,
            "cache_dir": args.cache_dir,
            "preprocess": {"detrend": True, "normalize": True, "eps": 1e-6},
            "split_names": {"train": "train", "val": "val", "test": "test"},
            "max_time": None,
            "dtype": "float32",
        },
        "training": {
            "deterministic": {"enable": True, "seed": 42},
            "data_loader": {
                "train": {"batch_size": 4, "shuffle": True, "num_workers": 2, "pin_memory": True, "persistent_workers": True, "drop_last": True},
                "val":   {"batch_size": 4, "shuffle": False, "num_workers": 2, "pin_memory": True, "persistent_workers": True, "drop_last": False},
            },
        },
    }

    loaders = build_dataloaders(cfg)
    for split, dl in loaders.items():
        for batch in dl:
            log.info("Split=%s batch keys=%s cube=%s target=%s",
                     split, list(batch.keys()),
                     tuple(batch["cube"].shape),
                     None if batch["target"] is None else tuple(batch["target"].shape))
            break