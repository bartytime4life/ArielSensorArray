
⸻

🛰️ .dvc/architecture.md

SpectraMind V50 — Data Version Control (DVC) Subsystem
Neuro-symbolic, physics-informed pipeline for the NeurIPS 2025 Ariel Data Challenge

⸻

📌 Purpose of .dvc/

The .dvc/ directory is the control center for data and model versioning.
	•	Git controls code + configs → logic reproducibility
	•	DVC controls data + models → artifact reproducibility

Every run of the SpectraMind V50 pipeline is thus tied to:
	1.	A Git commit hash (immutable code + config)
	2.	A DVC snapshot (datasets, calibration, model artifacts)

🔗 Cross-link: See docs/architecture.md for the full pipeline design.
This .dvc/ subsystem corresponds to the Data Layer described there.

⸻

📂 Directory Layout

.dvc/
├── cache/          # Local cache of binary blobs (never commit)
├── tmp/            # Ephemeral staging (safe to delete)
├── config          # Global DVC config (committed)
├── config.local    # Local overrides (ignored by Git)
├── plots/          # Plot templates (loss curves, calibration metrics)
└── lock/           # Auto-generated locks (commit-safe)

🔒 Commit Policy
	•	✅ Commit: .dvc/config, .dvcignore, dvc.yaml, *.dvc pointer files, .dvc/plots/*
	•	❌ Ignore: .dvc/cache/, .dvc/tmp/, .dvc/config.local

See also:
	•	.dvc/.dvcignore.readme.md
	•	.dvc/.gitattributes.readme.md

⸻

⚙️ Integration with SpectraMind V50
	1.	Hydra Configs → Typer CLI → DVC
	•	Commands like spectramind calibrate or spectramind train:
	•	Hydra composes configs
	•	CLI executes stages
	•	DVC snapshots outputs into .dvc pointers
	2.	Reproducibility Loop

git checkout <commit>
dvc checkout

Restores the exact datasets, models, and diagnostics.

	3.	CI/CD Enforcement
	•	GitHub Actions checks dvc status on every PR.
	•	Missing or broken .dvc pointers → merge blocked.
	•	Matches the “pre-flight safety check” model in root architecture docs.

⸻

📊 Plots & Metrics
	•	dvc plots renders loss curves, calibration reliability, FFT, symbolic metrics
	•	.dvc/plots/ holds JSON/YAML templates → reused across runs
	•	Outputs are embedded into the diagnostics dashboard (report.html)

🔗 Cross-link: see docs/architecture.md → Diagnostics Layer

⸻

🚀 Typical Workflows

Track a dataset

dvc add data/raw/fgs1_lightcurves.fits
git add data/raw/fgs1_lightcurves.fits.dvc .gitignore
git commit -m "Track raw FGS1 lightcurves with DVC"

Reproduce pipeline

dvc repro

Push artifacts to remote

dvc push -r s3-ariel

Pull artifacts from remote

dvc pull -r s3-ariel


⸻

🌌 Best Practices
	•	Always run via CLI (spectramind ...), never raw Python scripts
	•	Use Hydra overrides for dataset/model paths, never hardcode
	•	Run dvc push after success to sync remotes
	•	CI requires clean status before merging
	•	Never commit blobs in cache/ or tmp/

⸻

✅ Acceptance Criteria

The .dvc/ subsystem is mission-grade when:
	•	git checkout && dvc checkout fully restores any run
	•	All stages (calibrate → train → predict → diagnose → submit) exist in dvc.yaml
	•	CI enforces dvc status clean before merges
	•	S3/GCS/Azure remotes continuously synced
	•	Kaggle artifacts identical to local runs

⸻

🛡️ Alignment with Root Architecture
	•	Glass-box reproducibility ￼
	•	CLI-first workflows ￼
	•	Config-as-code with Hydra + DVC ￼
	•	CI/CD enforced data integrity

This subsystem ensures that the Data & Artifact layer described in docs/architecture.md is fully reproducible, transparent, and mission-ready.

⸻