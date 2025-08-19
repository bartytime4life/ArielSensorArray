"""
Diagnostics & quick-look visuals (saved to disk for headless review).
"""
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

from .paths import PROCESSED, OUTPUTS
from .io_utils import iter_npy_files
from .logging_utils import console, summary_table

def plot_example_spectrum(
    calibrated_dir: Path = PROCESSED / "calibrated",
    out_dir: Path = OUTPUTS / "diagnostics"
) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    console().rule("[bold cyan]Diagnostics")

    any_file = next(iter_npy_files(calibrated_dir), None)
    if any_file is None:
        summary_table("Diagnostics", {"status": "no calibrated cubes found"})
        return {"figures": 0}

    cube = np.load(any_file)
    spec = np.median(cube, axis=(0, 1))
    plt.figure(figsize=(9, 3))
    plt.plot(spec, lw=1.5)
    plt.title(f"Median spectrum: {any_file.name}")
    plt.xlabel("Wavelength channel")
    plt.ylabel("Relative flux (a.u.)")
    png = out_dir / "example_spectrum.png"
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()
    summary_table("Diagnostics", {"figure_saved": str(png)})
    return {"figures": 1}