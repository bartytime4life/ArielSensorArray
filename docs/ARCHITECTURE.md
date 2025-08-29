
⸻

🌌 SpectraMind V50 — System Architecture

Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge

⸻

0. Purpose & Scope

SpectraMind V50 is a neuro-symbolic, physics-informed system engineered for ESA’s Ariel exoplanet mission data.
It processes telescope observations (FGS1, AIRS) into reconstructed spectra (μ ± σ) while enforcing scientific rigor, reproducibility, and CI/CD safety.
	•	Git + Hydra → reproducible logic & configs
	•	DVC → reproducible data & models
	•	Typer CLI → reproducible execution

🔗 For data subsystem internals, see .dvc/architecture.md.

⸻

1. High-Level Layout

SpectraMind V50/
├── src/                # Core pipeline code
├── configs/            # Hydra YAML configs (training, model, diagnostics, submission)
├── data/               # Raw + processed telescope data (tracked via DVC)
├── outputs/            # Model predictions, diagnostics (tracked via DVC)
├── models/             # Trained models (DVC)
├── artifacts/          # Submission bundles, exports (DVC)
├── .dvc/               # DVC control system (see .dvc/architecture.md)
├── .github/            # CI/CD workflows (lint, CI, submission, security)
└── docs/               # System documentation (this file, subsystem guides)

	•	Code → deterministic logic (Git)
	•	Configs → structured Hydra composition (YAMLs)
	•	Data & Models → binary artifacts (DVC-managed)

⸻

2. Pipeline Stages

The system is defined as a DVC DAG (dvc.yaml) and orchestrated via the unified CLI:
	1.	Calibration
spectramind calibrate → raw FGS1/AIRS → calibrated data
	•	DVC stage: calibrate
	•	Hydra config: configs/data/calibration.yaml
	2.	Training
spectramind train → train FGS1+GNN model
	•	DVC stage: train
	•	Hydra config: configs/training/config_v50.yaml
	3.	Prediction
spectramind predict → μ, σ spectra predictions
	•	DVC stage: predict
	•	Hydra config: configs/inference/predict_v50.yaml
	4.	Diagnostics
spectramind diagnose dashboard → GLL, FFT, SHAP, symbolic overlays
	•	DVC stage: diagnose
	•	Hydra config: configs/diagnostics/diagnose_v50.yaml
	5.	Submission
spectramind submit → Kaggle-ready bundle
	•	DVC stage: submit
	•	Hydra config: configs/submission/submit_v50.yaml

🔗 See .dvc/architecture.md for how these stages are stored and versioned in DVC.

⸻

3. Integration of Subsystems
	•	Hydra Config-as-Code
Structured YAML configs define all parameters, overridable at CLI runtime.
	•	Typer CLI
One entrypoint: spectramind
	•	Subcommands: calibrate, train, predict, diagnose, submit
	•	Auto-generated --help, tab completion, reproducibility logging
	•	DVC Data Backbone
	•	Tracks all artifacts in data/, models/, outputs/, artifacts/
	•	Provides git checkout && dvc checkout reproducibility
	•	Remote sync (S3, GCS, Azure, local) ensures team-wide consistency
🔗 Detailed in .dvc/architecture.md.
	•	CI/CD Guardrails
	•	GitHub Actions validate Hydra configs + DVC stages
	•	Pre-flight: dvc status + selftest
	•	Fail fast if pointers/configs drift

⸻

4. Diagnostics & Explainability

Diagnostics run via CLI (spectramind diagnose) and are exported as both static plots and interactive HTML dashboards.
	•	DVC Plots (.dvc/plots/) define JSON templates for:
	•	Training loss
	•	Calibration reliability & coverage
	•	FFT power spectrum
	•	Symbolic rule violations
	•	Dashboard Export
	•	SHAP overlays
	•	Symbolic logic graphs
	•	FFT autocorr, entropy maps
	•	Full run log trace (v50_debug_log.md)

⸻

5. Best Practices
	•	Always run pipeline stages via CLI (spectramind ...), never ad-hoc scripts
	•	Hydra overrides, not hardcoding, for dataset/model switching
	•	Run dvc push after every successful run → sync with team/CI remote
	•	CI requires dvc status clean before merging PRs
	•	Never commit blobs in .dvc/cache/

⸻

6. Mission-Grade Acceptance Criteria

SpectraMind V50 is production-ready when:
	•	Any past run is restorable via git checkout && dvc checkout
	•	All stages are represented in dvc.yaml
	•	CI/CD enforces consistency (Hydra + DVC checks pass)
	•	All diagnostics integrate into HTML dashboard & DVC plots
	•	Kaggle bundles are byte-identical across environments

⸻

7. Cross-References
	•	.dvc/architecture.md — Deep dive into DVC subsystem
	•	[docs/diagnostics.md] — Full diagnostics overlay and symbolic explainability
	•	[docs/ci.md] — CI/CD guardrails and GitHub Actions workflows

⸻

✅ With this cross-linking, root docs/architecture.md = overview, and .dvc/architecture.md = deep dive.

⸻