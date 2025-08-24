# bootstrap_notebooks.sh
set -euo pipefail

# 1) ensure dir
mkdir -p notebooks

# 2) README (policy, usage, DVC/Kaggle notes)
cat > notebooks/README.md << 'MD'
# SpectraMind V50 Notebooks (orchestration-only)

These notebooks are **thin orchestration wrappers** over the official CLI (`spectramind`) and Hydra configs.
They never implement pipeline logic; they just call the CLI, render quick plots, and link to DVC-tracked artifacts.
All large artifacts live under `outputs/` and `logs/` (DVC‑backed). This keeps runs reproducible and auditable.  
References: CLI-first + Hydra + DVC reproducibility; logs include config and dataset hash.  

## How to run (local)
- Install the project per repo instructions.
- Use the cells to run `spectramind ...` with Hydra overrides (e.g. `training.fast_dev_run=true`).
- Artifacts land in `outputs/` and logs in `logs/`. Commit code & configs; track data/models with DVC.

## Kaggle handoff
These notebooks avoid web calls at run time and assume models/data are bundled as Kaggle Datasets; pin environments and write outputs under `/kaggle/working/outputs`. 

## Guardrails
- **CLI > Hydra > DVC** only; no hidden logic here. 
- All charts/HTML saved to `outputs/` (PNG/HTML/JSON/ZIP). Logs in `logs/`. 
- Each run records command, merged Hydra config, and commit/hash in the diary. 
MD

# 3) notebook template function (percent-format Python; Jupytext-friendly)
emit_nb () {
  local fname="$1"; shift
  local title="$1"; shift
  local body="$*"

  cat > "notebooks/${fname}.py" <<PY
# %% [markdown]
# # ${title}
# Thin orchestration notebook. Calls the official CLI with Hydra overrides and leaves artifacts in outputs/ + logs/.
# References: CLI-first, Hydra configs, DVC artifacts & immutable logs.  

# %%
import os, sys, json, subprocess, textwrap, pathlib, datetime

def run(cmd:list, env=None):
    print("\\n$ " + " ".join(cmd))
    res = subprocess.run(cmd, env=env, check=True)
    return res

ROOT = pathlib.Path(".").resolve()
OUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
OUTS.mkdir(exist_ok=True, parents=True)
LOGS.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Commands

# %%
${body}

# %% [markdown]
# ## Quick links
# - outputs/: saved PNG/HTML/JSON/ZIP
# - logs/: run diaries (config hash, dataset hash, timestamps)
PY
}

# 4) emit the 16 notebooks

emit_nb 00_quickstart "00 Quickstart" '
run(["spectramind", "--version"])
run(["spectramind", "train", "training.fast_dev_run=true"])
'

emit_nb 01_data_exploration "01 Data Exploration (FGS1 & AIRS)" '
run(["spectramind", "data", "explore", "dataset=nominal", "sample_limit=200"])
'

emit_nb 02_calibration_walkthrough "02 Calibration Walkthrough" '
# raw -> calibrated; verify HDF5 artifacts; pre/post FFT/PSD
run(["spectramind", "calibrate", "dataset=nominal", "calibrate.sample=10"])
run(["spectramind", "diagnose", "fft", "scope=calibration", "sample=10"])
'

emit_nb 03_train_v50_demo "03 Train V50 Demo" '
run(["spectramind", "train", "model=v50", "training.seed=42", "trainer.max_epochs=1"])
run(["spectramind", "diagnose", "loss_curve", "last_run=true"])
'

emit_nb 04_predict_v50_demo "04 Predict V50 Demo" '
run(["spectramind", "predict", "model=v50", "inference.split=validation", "predict.sample=12"])
run(["spectramind", "diagnose", "spectra_preview", "last_run=true"])
'

emit_nb 05_diagnostics_suite "05 Diagnostics Suite" '
run(["spectramind", "diagnose", "summary", "last_run=true"])
run(["spectramind", "diagnose", "umap_v50", "last_run=true"])
run(["spectramind", "diagnose", "tsne_interactive", "last_run=true"])
'

emit_nb 06_symbolic_overlays "06 Symbolic Overlays" '
run(["spectramind", "diagnose", "symbolic_overlays", "last_run=true"])
'

emit_nb 07_shap_explainability "07 SHAP Explainability" '
run(["spectramind", "explain", "shap_overlay", "last_run=true"])
run(["spectramind", "explain", "shap_attention_overlay", "last_run=true"])
'

emit_nb 08_fft_autocorr_analysis "08 FFT & Autocorrelation Analysis" '
run(["spectramind", "diagnose", "fft", "last_run=true"])
run(["spectramind", "diagnose", "autocorr", "last_run=true"])
'

emit_nb 09_ablation_study "09 Ablation Study" '
run(["spectramind", "ablate", "auto=true", "trainer.max_epochs=1"])
run(["spectramind", "diagnose", "ablation_leaderboard", "last_run=true"])
'

emit_nb 10_kaggle_submission_pipeline "10 Kaggle Submission Pipeline" '
# end-to-end: predict -> validate -> package
run(["spectramind", "predict", "inference.split=test"])
run(["spectramind", "validate", "last_run=true"])
run(["spectramind", "submit", "format=zip", "dest=outputs/submission.zip"])
'

emit_nb 11_corel_calibration_demo "11 COREL Conformal Calibration Demo" '
run(["spectramind", "calibrate", "corel", "last_run=true"])
run(["spectramind", "diagnose", "uncertainty_coverage", "last_run=true"])
'

emit_nb 12_benchmark_models_comparison "12 Benchmark Models Comparison" '
run(["spectramind", "benchmarks", "compare", "sources=kaggle_baselines,v50"])
'

emit_nb 13_gui_dashboard_demo "13 GUI Dashboard Demo (optional)" '
# launch static HTML diagnostics dashboard (UI-light), served from outputs/.
run(["spectramind", "dashboard", "serve=false", "export=outputs/dashboard"])
'

emit_nb 14_radiation_and_noise_modeling "14 Radiation & Noise Modeling (Edu)" '
# educational plots; saves under outputs/edu/
run(["spectramind", "edu", "radiation_noise", "save_dir=outputs/edu"])
'

emit_nb 15_gravitational_lensing_demo "15 Gravitational Lensing Demo (Edu)" '
# tutorial visuals on lensing impacts on transit spectra; ties to symbolic overlays
run(["spectramind", "edu", "lensing_demo", "save_dir=outputs/edu"])
'

echo "Notebooks bootstrapped in notebooks/"