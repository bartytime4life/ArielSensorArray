Here’s a fully-rebuilt README.md for the /configs directory, aligned with the SpectraMind V50 philosophy and your uploaded technical docs. It explains purpose, structure, usage, reproducibility principles, and ties configs into CLI + Hydra + DVC + Kaggle.

⸻

🗂️ /configs — SpectraMind V50 Configuration System

0. Purpose & Scope

The /configs directory defines all experiment parameters for the SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).
It is the single source of truth for:
	•	Data paths, calibration parameters, and preprocessing
	•	Model architectures (FGS1 Mamba encoder, AIRS GNN, decoders)
	•	Training hyperparameters, curriculum schedules, and loss weights
	•	Symbolic/physics constraints (smoothness, non-negativity, molecular priors)
	•	Diagnostics, explainability, and uncertainty calibration settings
	•	Runtime and environment overrides (local, Kaggle, CI)

This ensures experiments are Hydra-safe, reproducible, and auditable: every run is traceable to its config snapshot ￼ ￼.

⸻

1. Design Philosophy
	•	Hydra-first: All configs are modular YAML files, dynamically composed at runtime ￼.
	•	No hard-coding: Pipeline behavior is changed only via configs or CLI overrides, never by editing code ￼.
	•	Hierarchical layering: Defaults are composed from subgroups (data/, model/, optimizer/, etc.), with runtime overrides allowed at any depth.
	•	Versioned & logged: Each run stores its exact config in the Hydra output dir, logged to logs/v50_debug_log.md with a hash ￼ ￼.
	•	DVC integration: Data/model artifacts referenced in configs are DVC-tracked for reproducibility ￼.

⸻

2. Directory Structure

configs/
├── train.yaml              # Main entrypoint config (composes defaults)
├── predict.yaml            # Inference config
├── ablate.yaml             # Config grid for ablation sweeps
├── selftest.yaml           # Lightweight smoke/self-test config
│
├── data/                   # Dataset + calibration options
│   ├── nominal.yaml
│   ├── kaggle.yaml
│   └── debug.yaml
│
├── model/                  # Model architectures
│   ├── v50.yaml
│   ├── fgs1_mamba.yaml
│   ├── airs_gnn.yaml
│   └── decoder.yaml
│
├── optimizer/              # Optimizer + scheduler choices
│   ├── adam.yaml
│   ├── adamw.yaml
│   └── sgd.yaml
│
├── loss/                   # Physics/symbolic loss weights
│   ├── gll.yaml
│   ├── smoothness.yaml
│   └── symbolic.yaml
│
├── trainer/                # Training loop options
│   ├── default.yaml
│   ├── gpu.yaml
│   └── kaggle_safe.yaml
│
├── logger/                 # Logging & tracking
│   ├── tensorboard.yaml
│   ├── wandb.yaml
│   └── mlflow.yaml
│
└── local/                  # Machine-specific overrides (git-ignored)
    └── default.yaml


⸻

3. Usage

Running with defaults

python train_v50.py

Loads configs/train.yaml, which composes defaults across all groups.

Overriding values

python train_v50.py optimizer=adamw training.epochs=20 model=airs_gnn

Multirun sweeps

python train_v50.py -m optimizer=adam,sgd training.batch_size=32,64

Kaggle-safe run

spectramind train --config-name train.yaml trainer=kaggle_safe


⸻

4. Best Practices
	•	Keep configs in Git: All YAMLs under /configs (except /local/) must be version-controlled ￼.
	•	Use /local/ for secrets/paths: Machine-specific overrides (scratch dirs, cluster queue names) go here and are .gitignored ￼.
	•	Leverage interpolation: Use ${...} to link values across groups (e.g. num_classes: ${data.num_classes} in a model).
	•	Snapshot every run: Hydra auto-saves composed configs to outputs/YYYY-MM-DD_HH-MM-SS/. Never run without a saved config.
	•	Sync with DVC: Ensure any file path in configs is DVC-tracked for reproducibility ￼.

⸻

5. Integration
	•	CLI: All commands (spectramind train, spectramind diagnose, etc.) load configs through Hydra ￼ ￼.
	•	CI: GitHub Actions rebuilds configs, runs self-test, and executes sample pipelines to verify integrity ￼.
	•	Kaggle: Configs ensure ≤9 hr runtime, safe memory use, and no internet calls ￼ ￼.
	•	Dashboard: Configs feed into diagnostics (generate_html_report.py), producing versioned HTML reports with config metadata ￼.

⸻

6. References
	•	Hydra configuration best practices ￼
	•	SpectraMind V50 Technical Plan ￼
	•	Project Analysis of repo structure and configs ￼
	•	Strategy for updating and extending configs ￼

⸻

✅ With this setup, /configs is not just parameters: it is the flight plan for every SpectraMind V50 experiment, ensuring NASA-grade reproducibility and Kaggle-safe deployment.

⸻