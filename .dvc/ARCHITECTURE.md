
⸻

🛰️ .dvc/architecture.md

SpectraMind V50 — Data Version Control (DVC) Subsystem
Neuro-symbolic, physics-informed pipeline for the NeurIPS 2025 Ariel Data Challenge

⸻

📌 Purpose of .dvc/

The .dvc/ directory is the internal control center for DVC, the system we use to track large datasets, calibration outputs, and model artifacts.
	•	Git controls code + configs (reproducible logic).
	•	DVC controls data + models (large, evolving artifacts).

This ensures every run of the SpectraMind pipeline is tied to an immutable commit AND the exact input/output data used.

⸻

📂 Directory Layout

.dvc/
├── cache/          # Local cache of binary blobs (never commit to Git)
├── tmp/            # Ephemeral staging area (safe to delete)
├── config          # Versioned global DVC config (committed)
├── config.local    # Local-only overrides (ignored by Git)
├── plots/          # Plot templates for DVC (loss curves, metrics)
└── lock/           # Auto-generated locks for stages (commit-safe)

🔒 Git Policy
	•	Commit: .dvc/config, .dvcignore, dvc.yaml, *.dvc pointer files, .dvc/plots/*
	•	Ignore: .dvc/cache/, .dvc/tmp/, .dvc/config.local

See the root .gitignore for enforced rules.

⸻

⚙️ Integration with SpectraMind V50
	1.	Hydra Configs → CLI → DVC
	•	When you run spectramind calibrate or spectramind train, Hydra loads configs, and outputs are tracked with DVC.
	•	Each dataset/model snapshot is pinned via a .dvc pointer file.
	2.	Reproducibility Loop
	•	git checkout <commit> + dvc checkout restores the exact dataset + model used.
	•	Every diagnostic run is reproducible at the byte level.
	3.	CI/CD Enforcement
	•	GitHub Actions (.github/workflows/) verifies DVC integrity on every PR.
	•	Broken or missing .dvc pointers fail pre-flight checks.

⸻

📊 Plots & Metrics

DVC integrates with SpectraMind diagnostics:
	•	dvc plots renders loss curves, calibration error, and symbolic violation metrics.
	•	Results sync with outputs/ and can be included in the HTML diagnostics dashboard.

Plots under .dvc/plots/ define reusable JSON templates for standard metrics.

⸻

🚀 Typical Workflows

Add a dataset

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
	•	Always run via CLI: spectramind ... (never bypass with ad-hoc python ...).
	•	Use Hydra overrides to select datasets/models; never edit code for paths.
	•	Run dvc push after every successful run to sync with the remote.
	•	CI checks enforce dvc status clean before merging.
	•	Never commit blobs in cache/ or tmp/.

⸻

✅ Acceptance Criteria

The .dvc/ subsystem is considered mission-grade when:
	•	Every dataset/model is reproducible via git checkout && dvc checkout.
	•	All pipeline stages (calibrate → train → diagnose → submit) are defined in dvc.yaml.
	•	CI passes DVC consistency checks.
	•	Remote storage is continuously synchronized.

⸻
