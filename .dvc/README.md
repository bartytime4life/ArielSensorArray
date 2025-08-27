.dvc/ — Data Version Control Internals

The .dvc/ directory is the internal control center for Data Version Control (DVC) in the SpectraMind V50 repository.
It is automatically created when you run dvc init and should always be tracked in Git (except for transient cache files).

DVC acts as the data & artifact versioning system for the entire SpectraMind pipeline:
	•	✅ Ensures raw telescope data, processed spectra, models, and diagnostics are all tied to Git commits.
	•	✅ Guarantees byte-for-byte reproducibility across machines, Kaggle, or HPC clusters.
	•	✅ Provides pipeline orchestration (dvc repro) from calibration → training → prediction → diagnostics → submission.
	•	✅ Keeps large binary artifacts out of Git, storing them in cache/remotes but leaving behind lightweight pointers.

⸻

📂 Contents

Inside .dvc/ you will find several subdirectories and control files:

File/Dir	Purpose
config	Global DVC configuration (remotes, cache locations, default options).
config.local	Developer-specific overrides (ignored by Git, safe for local secrets).
tmp/	Temporary files used during DVC operations.
plots/	Plot templates for dvc plots diff (metrics and diagnostics charts).
cache/	Content-addressable storage for all tracked files (data, models, outs).
lock files (if any)	Pipeline state snapshots to guarantee exact reproduction.

⚠️ Do not manually edit files inside .dvc/ unless you know what you’re doing. Use dvc config, dvc remote, or dvc repro commands to make changes safely.

⸻

🔑 Key Concepts
	•	Stage tracking: Each step of the pipeline (e.g. calibration, training) is defined in dvc.yaml.
	•	Pointer files: Adding data (dvc add data/raw) creates a .dvc pointer file in the repo root.
	•	Cache system: The actual large file lives in .dvc/cache/ or a remote (S3, GCS, SSH, etc.),
while Git only sees the lightweight .dvc file.
	•	Reproduction: At any commit, running dvc repro ensures outputs are regenerated consistently.

⸻

🚀 Workflow
	1.	Track new data/artifact

dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw Ariel telescope data with DVC"


	2.	Define pipeline stages
All stages live in dvc.yaml:
	•	calibrate → train → predict → diagnose → submit
Run the whole chain with:

dvc repro


	3.	Version & share artifacts
Push/pull large files with:

dvc push   # upload to remote storage
dvc pull   # download artifacts for current commit



⸻

🛰️ SpectraMind V50 Standards
	•	Hydra + DVC integration: configs drive every stage, DVC logs tie outputs to exact configs.
	•	NASA-grade reproducibility: any leaderboard submission can be recreated from a Git hash.
	•	CI/CD enforcement: GitHub Actions run dvc repro on sample data to ensure integrity.
	•	Remote compatibility: Designed for Kaggle dataset remotes + S3/lakeFS storage for collaboration.

⸻

✅ Best Practices
	•	Always commit pointer files (*.dvc) and dvc.yaml, never large data itself.
	•	Add secrets (remote URLs, tokens) in config.local only — never commit to Git.
	•	Run dvc status often to verify data consistency.
	•	Use dvc exp run + dvc exp show for experiment tracking with config overrides.

⸻

🔒 In summary:
The .dvc/ directory is the mission control for data versioning in SpectraMind V50.
It ensures that all telescope inputs, trained models, and diagnostic outputs are fully reproducible, shareable, and scientifically trustworthy.

⸻
