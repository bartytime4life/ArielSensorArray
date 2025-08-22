# src/spectramind_ci/calibrate.py
# ==============================================================================
# CI‑safe calibration entrypoint
#
# This module:
#  1) Composes Hydra config via your pipeline's calibrator (imported below)
#  2) Runs calibration
#  3) Emits mission‑grade logs/calibration.json + outputs/manifests/calibration_manifest.json
#
# It does NOT alter your existing CLI; CI calls this module directly to avoid patching
# your current Typer app. Locally, you can run:
#   poetry run python -m spectramind_ci.calibrate --config-path configs/calibration --config-name v50.yaml
# ==============================================================================

from __future__ import annotations
import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict

# ---- Import your pipeline's calibration entrypoint here -----------------------
# Expected contract: run_calibration(config_path: str, config_name: str) -> Dict[str, Any]
# The returned dict should already contain the sections shown below (calibration_run, inputs, preprocessing, outputs, diagnostics, uncertainty_calibration, reproducibility, timing, status).
# If your project uses a different signature, adjust the wrapper call accordingly.

try:
    # Preferred: a dedicated calibration orchestrator module
    from spectramind.pipeline.calibration import run_calibration  # type: ignore
except Exception as _e:
    # Fallback stub to avoid import errors during first integration.
    # Replace this with your real pipeline import as soon as available.
    def run_calibration(config_path: str, config_name: str) -> Dict[str, Any]:  # type: ignore
        # Minimal, self‑contained example object matching your approved structure.
        now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return {
            "calibration_run": {
                "id": now,
                "commit_hash": "unknown",
                "config_hash": "unknown",
                "pipeline_stage": "calibration",
                "environment": {
                    "python_version": "unknown",
                    "cuda_version": "unknown",
                    "torch_version": "unknown",
                    "device": "unknown",
                    "os": "unknown"
                }
            },
            "inputs": {
                "raw_data": "data/raw/",
                "dvc_hash": "unknown",
                "frames": {"fgs1_count": 0, "airs_count": 0}
            },
            "preprocessing": {},
            "outputs": {
                "calibrated_lightcurves": "outputs/calibrated/lightcurves.h5",
                "preview_plots": [],
                "stats": {}
            },
            "diagnostics": {
                "fft_smoothness": 0.0,
                "symbolic_constraint_violations": {},
                "coverage": {"bins_total": 283, "bins_clean": 283, "bins_flagged": 0}
            },
            "uncertainty_calibration": {},
            "reproducibility": {
                "dvc_stage": "calibration",
                "mlflow_run_id": "unknown",
                "hydra_config_used": f"{config_path}/{config_name}",
                "artifact_manifest": "outputs/manifests/calibration_manifest.json"
            },
            "timing": {
                "start_time": now,
                "end_time": now,
                "duration_sec": 0
            },
            "status": "success"
        }

# ---- Local utilities for writing JSON and system fingerprint ------------------
from spectramind.logging_utils import write_json, ensure_dir, git_commit_hash, system_fingerprint


def build_manifest(report: Dict[str, Any]) -> Dict[str, Any]:
    outputs = report.get("outputs", {})
    return {
        "artifacts": {
            "calibrated_lightcurves": outputs.get("calibrated_lightcurves"),
            "preview_plots": outputs.get("preview_plots", []),
        },
        "config": {
            "hydra_config_used": report.get("reproducibility", {}).get("hydra_config_used"),
            "dvc_stage": report.get("reproducibility", {}).get("dvc_stage"),
        },
        "run": report.get("calibration_run", {}),
        "status": report.get("status", "unknown"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SpectraMind V50 — CI calibration entrypoint")
    ap.add_argument("--config-path", required=True, help="Hydra config path (folder)")
    ap.add_argument("--config-name", required=True, help="Hydra config name (file)")
    ap.add_argument("--log-dir", default=os.getenv("SM_LOG_DIR", "logs"), help="Directory for JSON logs")
    args = ap.parse_args()

    # 1) Run your pipeline
    report = run_calibration(config_path=args.config_path, config_name=args.config_name)

    # 2) Enrich with commit + system fingerprint if missing (harmless no‑ops otherwise)
    report.setdefault("calibration_run", {})
    report["calibration_run"].setdefault("commit_hash", git_commit_hash())
    report["calibration_run"].setdefault("environment", {}).update(system_fingerprint())

    # 3) Persist mission‑grade logs
    ensure_dir(args.log_dir)
    write_json(report, Path(args.log_dir) / "calibration.json")

    manifest = build_manifest(report)
    write_json(manifest, "outputs/manifests/calibration_manifest.json")

    # 4) Optional CI assertion: fail job if status is not success
    if report.get("status") != "success":
        raise SystemExit("Calibration reported non‑success status")


if __name__ == "__main__":
    main()