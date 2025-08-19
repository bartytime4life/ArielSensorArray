"""
Calibrate raw spectral timeseries cubes with simple, physically-motivated steps:

- Bias subtraction
- Dark-current subtraction
- Flat-field correction
- (Optional) wavelength registration (identity here but hook provided)

This mirrors standard spectroscopic pipelines in a compact form.  See the plan for
NASA-grade steps and rationale. 
"""
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from .paths import RAW, PROCESSED
from .io_utils import iter_npy_files, save_npy, load_optional_npy
from .logging_utils import console, log_event_jsonl, summary_table

def _apply_bias(frame: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    return frame if bias is None else frame - bias

def _apply_dark(frame: np.ndarray, dark: Optional[np.ndarray]) -> np.ndarray:
    return frame if dark is None else frame - dark

def _apply_flat(frame: np.ndarray, flat: Optional[np.ndarray]) -> np.ndarray:
    if flat is None:
        return frame
    eps = 1e-9
    return frame / (flat + eps)

def _register_wavelength(cube: np.ndarray, mapping: Optional[np.ndarray]) -> np.ndarray:
    # Hook for wavelength calibration. If mapping provided (dst indices), remap columns.
    if mapping is None:
        return cube
    # mapping is expected as float or int index array length = n_cols
    n_t, n_y, n_x = cube.shape
    out = np.zeros_like(cube)
    # Simple nearest neighbor remap across wavelength axis (x)
    xm = np.clip(np.rint(mapping).astype(int), 0, n_x - 1)
    out[:, :, :] = cube[:, :, xm]
    return out

def calibrate_cube(
    path_raw_cube: Path,
    bias: Optional[np.ndarray],
    dark: Optional[np.ndarray],
    flat: Optional[np.ndarray],
    wave_map: Optional[np.ndarray],
) -> np.ndarray:
    # cube shape: (time, spatial, wavelength) or (t, y, x)
    cube = np.load(path_raw_cube)
    # Operate per-frame to mimic real pipelines (and avoid peak memory)
    out = np.empty_like(cube, dtype=np.float32)
    for i in range(cube.shape[0]):
        frame = _apply_bias(cube[i], bias)
        frame = _apply_dark(frame, dark)
        frame = _apply_flat(frame, flat)
        out[i] = frame
    out = _register_wavelength(out, wave_map)
    return out

def run_calibration(
    raw_dir: Path = RAW,
    processed_dir: Path = PROCESSED,
    bias_path: Optional[Path] = None,
    dark_path: Optional[Path] = None,
    flat_path: Optional[Path] = None,
    wave_path: Optional[Path] = None,
) -> Dict[str, int]:
    bias = load_optional_npy(bias_path) if bias_path else None
    dark = load_optional_npy(dark_path) if dark_path else None
    flat = load_optional_npy(flat_path) if flat_path else None
    wave = load_optional_npy(wave_path) if wave_path else None

    event_log = processed_dir / "calibrate_events.jsonl"
    n_in, n_ok = 0, 0
    console().rule("[bold cyan]Calibration")
    summary_table("Reference frames", {
        "bias": str(bias_path) if bias_path else "None",
        "dark": str(dark_path) if dark_path else "None",
        "flat": str(flat_path) if flat_path else "None",
        "wavelength_map": str(wave_path) if wave_path else "None",
    })

    for cube_path in iter_npy_files(raw_dir):
        n_in += 1
        rel = cube_path.relative_to(raw_dir)
        out_path = processed_dir / "calibrated" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cal = calibrate_cube(cube_path, bias, dark, flat, wave)
            save_npy(out_path, cal.astype(np.float32))
            n_ok += 1
            log_event_jsonl(event_log, {"event": "calibrated", "src": str(cube_path), "dst": str(out_path)})
        except Exception as e:  # keep going; record failure
            log_event_jsonl(event_log, {"event": "error", "src": str(cube_path), "error": repr(e)})

    summary_table("Calibration summary", {"input_cubes": n_in, "calibrated": n_ok})
    return {"input_cubes": n_in, "calibrated": n_ok}