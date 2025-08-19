import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

def iter_npy_files(root: Path) -> Iterable[Path]:
    yield from sorted(p for p in root.rglob("*.npy") if p.is_file())

def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)

def load_optional_npy(path: Path) -> np.ndarray:
    return np.load(path) if path.is_file() else None

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def chunks(seq: Iterable[Any], n: int) -> Iterable[Tuple[Any, ...]]:
    bucket = []
    for x in seq:
        bucket.append(x)
        if len(bucket) == n:
            yield tuple(bucket)
            bucket = []
    if bucket:
        yield tuple(bucket)