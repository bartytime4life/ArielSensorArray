
⸻

.dvc/ — Data Version Control (DVC) Guide

Mission: keep code + configs in Git, data/models in DVC, and ephemera out of both — so every result is exactly reproducible ￼.

⸻

Why DVC here?

SpectraMind V50 moves large, evolving artifacts (calibrated lightcurves, feature stores, model checkpoints, submissions) out of Git and into a DVC remote.
Small, stable descriptors (dvc.yaml, *.dvc, configs) stay in Git.
This keeps the Git history clean and lets anyone dvc pull the exact blobs tied to a commit/config ￼.

⸻

What goes in Git (commit these)
	•	Pipeline descriptors
	•	dvc.yaml (stage DAG)
	•	*.dvc pointer files
	•	Config (shared)
	•	.dvc/config (safe, no secrets)
	•	.dvcignore
	•	Plots/templates
	•	.dvc/plots/* (plot specs, JSON templates for metrics)

Rule of thumb: If it’s human-readable & stable and needed to recreate a run, it belongs in Git ￼.

⸻

What goes in DVC (tracked, not committed)
	•	Large or volatile artifacts
	•	Raw & calibrated datasets
	•	Intermediate feature stores
	•	Trained models & ensembles
	•	Heavy eval metrics CSV/JSONs, submission bundles
	•	Anything ≥ a few MB or expected to churn

Tracked via:

dvc add data/features.parquet
git add data/features.parquet.dvc
git commit -m "Track features via DVC"
dvc push


⸻

What is ignored (keep out of Git)
	•	Runtime internals (already in .dvc/.gitignore):
	•	.dvc/cache/, .dvc/tmp/, .dvc/state/, .dvc/experiments/, .dvc/logs/, stage.lock
	•	Local/session noise:
	•	.dvc/config.local, scratch remotes, temp datasets

These are ephemeral, machine-local, or secret — never commit ￼.

⸻

Typical Workflow (CLI-first)
	1.	Define/extend a stage in dvc.yaml
Inputs: data + code + Hydra config
Outputs: calibrated data / model / submission
	2.	Run pipeline

dvc repro              # or `make` / `spectramind train`


	3.	Track/push artifacts

dvc add models/v50.ckpt
git add models/v50.ckpt.dvc
git commit -m "Add V50 checkpoint"
dvc push


	4.	Share run

git push
dvc push
# teammate:
git pull && dvc pull



⸻

Experiments & Metrics
	•	Run variants with Hydra overrides or spectramind ablate:

dvc exp run
dvc exp show
dvc exp gc --workspace   # clean dangling exps


	•	Log metrics to small JSON/CSV + declare in dvc.yaml:

metrics:
  - outputs/metrics.json
plots:
  - outputs/gll_curve.json



⸻

Do / Don’t Checklist

✅ Do
	•	Commit dvc.yaml, *.dvc, .dvc/config, .dvcignore, plot templates
	•	Use DVC for artifacts; push after Git push
	•	Reference outputs in stages (reproducible > ad-hoc)
	•	Share a team remote (S3/GS/Azure/lakeFS) ￼

❌ Don’t
	•	Commit .dvc/cache/, .dvc/tmp/, stage.lock
	•	Commit large blobs to Git
	•	Hard-code local paths/mounts in configs

⸻

Troubleshooting
	•	File missing after git pull → run dvc pull
	•	Slow pulls → check dvc remote list; enable cache on fast disk
	•	*.dvc conflicts → re-add artifact and recommit
	•	Stale outputs after code change → dvc repro will rebuild only invalidated stages

⸻

CI & Kaggle Notes
	•	CI: run read-only dvc pull to fetch cached artifacts for tests ￼.
	•	Kaggle: since kernels have no remote creds, package needed artifacts as Kaggle Datasets or export minimal files in repo (small only) ￼Ariel Data Challenge Dataset.pdf.

⸻

One-liner Policy

Large or churning → DVC
Small & pipeline-defining → Git
Local/secrets → ignore ￼

⸻