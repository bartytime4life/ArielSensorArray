"""
Bundle predictions into a submission archive (CSV + ZIP).
"""
from pathlib import Path
from typing import Dict
import numpy as np
import csv
import zipfile

from .paths import OUTPUTS
from .io_utils import iter_npy_files
from .logging_utils import console, summary_table

def build_submission(
    preds_dir: Path = OUTPUTS / "predictions",
    csv_out: Path = OUTPUTS / "submission.csv",
    zip_out: Path = OUTPUTS / "submission.zip"
) -> Dict[str, str]:
    console().rule("[bold cyan]Submission")
    rows = []
    for p in iter_npy_files(preds_dir):
        spec = np.load(p)
        # Row format: id, v0, v1, ..., vN (simple, generic CSV)
        rows.append([p.stem.replace(".pred", "")] + [float(x) for x in spec])

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)

    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_out, arcname=csv_out.name)

    summary_table("Submission artifacts", {"csv": str(csv_out), "zip": str(zip_out)})
    return {"csv": str(csv_out), "zip": str(zip_out)}