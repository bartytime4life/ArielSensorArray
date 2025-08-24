#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/reproducibility.py — SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge)

Mission-grade reproducibility helpers for the CLI-first V50 pipeline.

Highlights
----------
• Deterministic seeding across Python, NumPy, and PyTorch (CPU/CUDA).
• Toggle PyTorch deterministic algorithms (cuDNN) safely.
• Stable config hashing (ΩConf/JSON/YAML → SHA-256) for run identity.
• Environment snapshot: Python / OS / CUDA / cuDNN / torch / numpy / git.
• Append-only audit logs (JSONL + Markdown) and run-summary artifact writer.
• RNG state freezer (context manager) and worker_init for DataLoader workers.

These utilities are intentionally dependency-light and work even when Hydra or
OmegaConf is not installed (they gracefully downgrade to JSON/YAML).
"""

from __future__ import annotations

import os
import re
import io
import sys
import json
import time
import math
import uuid
import yaml
import hashlib
import random
import getpass
import platform
import datetime as dt
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Tuple, Union, Mapping, Sequence
from contextlib import contextmanager

# Optional imports
try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from omegaconf import OmegaConf, DictConfig  # type: ignore
    _HAS_OMEGA = True
except Exception:  # pragma: no cover
    OmegaConf, DictConfig = None, None  # type: ignore
    _HAS_OMEGA = False


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------

def _maybe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    _maybe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_jsonl(path: Union[str, Path], obj: Mapping[str, Any]) -> None:
    path = Path(path)
    _maybe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_markdown(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    _maybe_mkdir(path.parent)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n---\n**{stamp}**\n\n{text.rstrip()}\n")


# -----------------------------------------------------------------------------
# Config canonicalization & hashing
# -----------------------------------------------------------------------------

def _to_canonical_dict(cfg: Any) -> Any:
    """
    Convert cfg into a JSON-serializable structure with stable key order.
    Supports OmegaConf DictConfig / dataclasses / objects with __dict__ / dict-like.
    """
    # OmegaConf first
    if _HAS_OMEGA and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    # Dict-like
    if isinstance(cfg, Mapping):
        return {str(k): _to_canonical_dict(v) for k, v in sorted(cfg.items(), key=lambda kv: str(kv[0]))}
    # Sequence (but not str/bytes)
    if isinstance(cfg, (list, tuple)):
        return [_to_canonical_dict(v) for v in cfg]
    # Dataclass-like
    if hasattr(cfg, "__dict__") and not isinstance(cfg, type):
        return _to_canonical_dict(vars(cfg))
    # Fallback scalar
    return cfg


def config_to_yaml(cfg: Any) -> str:
    """
    Render cfg as a canonical YAML string. Keeps keys sorted and values resolved.
    """
    canon = _to_canonical_dict(cfg)
    # Use safe_dump for stable ordering
    return yaml.safe_dump(canon, sort_keys=True, allow_unicode=True)


def hash_config(cfg: Any, *, algo: str = "sha256") -> str:
    """
    Compute a stable short hash for the composed config (default SHA-256 / 16 hex chars).
    """
    text = config_to_yaml(cfg)
    h = hashlib.new(algo)
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


# -----------------------------------------------------------------------------
# Seeding & determinism
# -----------------------------------------------------------------------------

def set_seed(seed: int, *, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs and optionally enable deterministic algorithms.

    Args:
        seed: integer seed.
        deterministic: if True, torch.use_deterministic_algorithms(True).
        benchmark: set torch.backends.cudnn.benchmark flag (False improves reproducibility).
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    if _HAS_NUMPY:
        np.random.seed(seed)

    # PyTorch
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN flags
        try:
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = bool(benchmark)
        except Exception:
            pass
        # Opt-in determinism where supported
        try:
            torch.use_deterministic_algorithms(bool(deterministic))
        except Exception:
            # Not all builds support it (or certain ops are not deterministic)
            pass


def enable_torch_determinism(enable: bool = True) -> None:
    """
    Toggle PyTorch deterministic algorithms (and safest cuDNN flags).
    """
    if not _HAS_TORCH:
        return
    try:
        torch.use_deterministic_algorithms(bool(enable))
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = bool(enable)
        # In strict reproducibility, benchmark should be False.
        torch.backends.cudnn.benchmark = False if enable else torch.backends.cudnn.benchmark
    except Exception:
        pass


def dataloader_worker_init(worker_id: int) -> None:
    """
    Deterministic worker initializer for DataLoader.
    """
    if not _HAS_TORCH:
        return
    base_seed = torch.initial_seed() % (2**31 - 1)
    if _HAS_NUMPY:
        np.random.seed((base_seed + worker_id) % (2**31 - 1))
    random.seed((base_seed + worker_id) % (2**31 - 1))


@contextmanager
def freeze_torch_rng_state():
    """
    Context manager to save/restore PyTorch RNG state (CPU & CUDA).
    """
    if not _HAS_TORCH:
        yield
        return
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


# -----------------------------------------------------------------------------
# Environment & git snapshot
# -----------------------------------------------------------------------------

def _run(cmd: Sequence[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=3)
        return 0, out.decode("utf-8", errors="replace").strip()
    except Exception as e:  # pragma: no cover
        return 1, str(e)


def get_git_info(root: Union[str, Path] = ".") -> Dict[str, Optional[str]]:
    """
    Collect basic git info (if repo is present): commit, branch, dirty, remote.
    """
    root = str(root)
    code, commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    code_b, branch = _run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"])
    code_s, status = _run(["git", "-C", root, "status", "--porcelain"])
    code_r, remote = _run(["git", "-C", root, "config", "--get", "remote.origin.url"])
    return {
        "commit": commit if code == 0 else None,
        "branch": branch if code_b == 0 else None,
        "dirty": ("yes" if (code_s == 0 and len(status) > 0) else ("no" if code_s == 0 else None)),
        "remote": remote if code_r == 0 else None,
    }


def get_env_fingerprint(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Summarize key runtime info for audit logs and run manifests.
    """
    env: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user": getpass.getuser() if hasattr(getpass, "getuser") else None,
        "hostname": platform.node(),
        "platform": {
            "os": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "libraries": {},
        "cuda": {},
        "git": get_git_info("."),
    }

    # Library versions
    if _HAS_NUMPY:
        env["libraries"]["numpy"] = getattr(np, "__version__", None)
    if _HAS_TORCH:
        env["libraries"]["torch"] = getattr(torch, "__version__", None)
        env["cuda"]["available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda"]["device_count"] = int(torch.cuda.device_count())
            env["cuda"]["current_device"] = int(torch.cuda.current_device())
            env["cuda"]["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
            env["cuda"]["torch_cuda_version"] = torch.version.cuda
            try:
                env["cuda"]["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                env["cuda"]["cudnn_version"] = None

    if extra:
        env.update(extra)
    return env


# -----------------------------------------------------------------------------
# Run metadata / audit trail
# -----------------------------------------------------------------------------

def record_run_metadata(
    out_dir: Union[str, Path],
    cfg: Any,
    *,
    run_hash: Optional[str] = None,
    tag: Optional[str] = None,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write a canonical set of run metadata artifacts under `out_dir` and append logs.

    Artifacts:
        - out_dir/config_composed.yaml             (canonical resolved config)
        - out_dir/run_hash_summary_v50.json        (key metadata for this run)
        - logs/v50_runs.jsonl                      (append-only machine-readable log)
        - logs/v50_debug_log.md                    (append-only human-readable audit)
    """
    out_dir = Path(out_dir).absolute()
    _maybe_mkdir(out_dir)

    # Compose config YAML and hash
    cfg_yaml = config_to_yaml(cfg)
    cfg_path = out_dir / "config_composed.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        f.write(cfg_yaml)

    rhash = run_hash or hash_config(cfg)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = tag or "run"

    # Environment snapshot
    env = get_env_fingerprint(extra_env)

    # Summaries
    summary = {
        "run_hash": rhash,
        "tag": tag,
        "timestamp_local": stamp,
        "out_dir": str(out_dir),
        "config_path": str(cfg_path),
        "config_hash": rhash,   # alias for clarity
        "git": env.get("git"),
        "platform": env.get("platform"),
        "libraries": env.get("libraries"),
        "cuda": env.get("cuda"),
    }
    write_json(out_dir / "run_hash_summary_v50.json", summary)

    # Append-only JSONL (machine)
    append_jsonl(Path("logs") / "v50_runs.jsonl", {**summary, "ts": time.time()})

    # Append-only Markdown (human)
    md = []
    md.append(f"### SpectraMind V50 — {tag}  \n`{rhash}` @ `{stamp}`")
    md.append(f"- out_dir: `{summary['out_dir']}`")
    md.append(f"- config: `{summary['config_path']}`")
    if summary["git"]:
        git = summary["git"]
        md.append(f"- git: `{git.get('commit')}` (branch `{git.get('branch')}`, dirty={git.get('dirty')})")
    libs = summary.get("libraries") or {}
    if libs:
        lib_line = ", ".join([f"{k}={v}" for k, v in libs.items() if v])
        md.append(f"- libs: {lib_line}")
    cuda = summary.get("cuda") or {}
    if cuda.get("available"):
        md.append(f"- cuda: device `{cuda.get('current_device')}` / {cuda.get('device_name')} "
                  f"(cuDNN={cuda.get('cudnn_version')}, torch.cuda={cuda.get('torch_cuda_version')})")
    append_markdown(Path("logs") / "v50_debug_log.md", "\n".join(md))

    return summary


# -----------------------------------------------------------------------------
# Convenience one-liners
# -----------------------------------------------------------------------------

def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
    worker_init: bool = True,
) -> Dict[str, Any]:
    """
    Opinionated seeding helper for quick-start scripts and tests.
    Returns a dict describing the applied policy (for logging).
    """
    set_seed(seed, deterministic=deterministic, benchmark=benchmark)
    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "benchmark": bool(benchmark),
        "worker_init_fn": "utils.reproducibility.dataloader_worker_init" if worker_init else None,
    }


# -----------------------------------------------------------------------------
# Minimal self-check
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    # Simulate a composed config
    cfg = {
        "runtime": {"seed": 1337, "cuda": True},
        "model": {"name": "spectramind_v50", "d_model": 256},
        "training": {"epochs": 2, "lr": 1e-3},
    }

    print("Config YAML:")
    print(config_to_yaml(cfg))

    rh = hash_config(cfg)
    print(f"Config hash: {rh}")

    set_seed(123, deterministic=True, benchmark=False)
    s = seed_everything(42)
    print("seed_everything:", s)

    summary = record_run_metadata("outputs/example_run", cfg, tag="selftest")
    print("Recorded:", summary)