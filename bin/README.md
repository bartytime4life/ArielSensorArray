🛰️ SpectraMind V50 — bin/ Directory

The bin/ directory contains executable scripts that orchestrate the SpectraMind V50 pipeline end-to-end.
Every script is CLI-first (driving the Typer CLI spectramind), Hydra-safe (no hard-coded params), and reproducibility-compliant (logs + manifests).

⸻

🎯 Purpose
	•	Centralize shell entrypoints outside Python modules.
	•	Provide one-command workflows for developers and CI/CD (submission, repair, diagnostics, benchmark).
	•	Enforce guardrails (selftests, config hashing, logging) before heavy tasks.
	•	Keep Git + DVC + Kaggle flows consistent and auditable.

⸻

📦 Contents
	•	analyze-log.sh — Parse logs/v50_debug_log.md → outputs/log_table.{md,csv}.
	•	benchmark.sh — Standardized benchmark pass (train→diagnose), summary + optional manifest.
	•	diagnostics.sh — Rich diagnostics (smoothness, dashboard, optional symbolic overlays).
	•	make-submission.sh — Predict → validate → bundle; Kaggle-ready; --dry-run safe.
	•	repair_and_push.sh — Git/DVC repair + commit/push; tagging & manifest optional.
	•	.gitkeep — Keeps bin/ tracked if empty.

Tip: All scripts adhere to common contracts (see Conventions & Contracts).

⸻

🚀 Quick Start

Run from the repo root with an activated environment (Poetry venv or Docker):

# One-shot dry-run submission (logs but no ZIP write)
./bin/make-submission.sh --dry-run --tag dev-check

# Real bundle + open folder
./bin/make-submission.sh --no-dry-run --open --tag v50.1.0

# Repair + push (with preflight tests and hooks)
./bin/repair_and_push.sh --msg "Symbolic loss clamp fix" --run-tests --run-pre-commit

# Benchmark (CPU) with manifest + auto-open report
./bin/benchmark.sh --profile cpu --epochs 1 --tag smoke --manifest --open-report

# Diagnostics with symbolic overlays
./bin/diagnostics.sh --symbolic --open

Assumptions:
	•	Run from repo root.
	•	Dependencies installed (Poetry environment or Docker image).
	•	spectramind available (e.g., poetry run spectramind behind the scenes).

⸻

🧭 Conventions & Contracts (all scripts)
	•	Header & strict mode: #!/usr/bin/env bash + set -Eeuo pipefail.
	•	Help: --help prints purpose, options, examples.
	•	Echo intent: Print effective command(s) before executing.
	•	Selftest first: Light spectramind test --fast prior to heavy work (where applicable).
	•	Logging: Append a single concise line to logs/v50_debug_log.md for each run.
	•	Idempotence: Support --dry-run and safe re-runs.
	•	No hard-coding: Parameters flow from Hydra configs / CLI overrides.
	•	Exit codes: 0 success; non-zero on failure (usage=2; selftest=3; DVC=4; push=5).

⸻

⚙️ Option Cheat-Sheets

make-submission.sh

Option	Type	Default	Description
--dry-run	flag	true	Run predict/validate but skip ZIP write.
--open	flag	false	Open submissions/ after success.
--tag <str>	string	“”	Include version tag in bundle & logs.

repair_and_push.sh

Option	Type	Default	Description
--msg "<text>"	string	(required*)	Commit message (\* unless --allow-empty).
--allow-empty	flag	false	Permit empty commit.
--allow-non-main	flag	false	Allow pushes from non-main branch.
--run-tests	flag	false	Run spectramind test --fast preflight.
--run-pre-commit	flag	false	Run pre-commit hooks before staging.
--no-dvc	flag	false	Skip DVC (status/add/push).
--no-push	flag	false	Local commit only.
--tag "<vX.Y.Z>"	string	“”	Create/push annotated tag.
--manifest	flag	false	Write JSON manifest under outputs/manifests/.

benchmark.sh

Option	Type	Default	Description
`–profile {cpu	gpu}`	choice	gpu
--epochs <N>	int	1	Training epochs.
--seed <N>	int	42	Deterministic seed.
--overrides "<hydra>"	string	“”	Hydra overrides (quoted).
--extra "<cli>"	string	“”	Extra args pass-through.
--outdir <dir>	string	benchmarks/<ts>_<profile>	Output directory.
--tag <str>	string	“”	Label used in logs/summary.
--dry-run	flag	false	Plan only; no execution.
--open-report	flag	false	Open latest HTML after run.
--manifest	flag	false	Write run manifest in outdir.

diagnostics.sh

Option	Type	Default	Description
--outdir <dir>	string	outputs/diagnostics/<ts>	Diagnostics output dir.
--source <path>	string	“”	Optional source (e.g., predictions.csv).
--overrides "<hydra>"	string	“”	Hydra overrides.
--extra "<cli>"	string	“”	Extra args pass-through.
--no-umap	flag	false	Skip UMAP plot in dashboard.
--no-tsne	flag	false	Skip t-SNE plot in dashboard.
--symbolic	flag	false	Include symbolic overlays & tables.
--open	flag	false	Open latest HTML after run.
--manifest	flag	false	Write manifest in outdir.


⸻

🧪 Safety & Failure Modes
	•	Selftest failed: abort with hint to run spectramind test --help.
	•	DVC mismatch: print dvc status; suggest dvc repro / dvc add / dvc push.
	•	Nothing to commit: warn but allow tag/write manifest.
	•	Push failed: retry with backoff; print remote error on final fail.
	•	Unknown option: print --help then exit 2.

Exit Codes (normalized)
	•	0 OK
	•	2 Usage/args error
	•	3 Selftest failed
	•	4 DVC inconsistency
	•	5 Git/DVC push failure
	•	1 Generic failure

⸻

🧑‍💻 Developer Notes

When adding a new script:
	1.	Add header + strict mode.
	2.	Provide --help with purpose/usage/examples.
	3.	Support --dry-run where sensible.
	4.	Append log line(s) to logs/v50_debug_log.md.
	5.	Favor Hydra overrides over hard-coded params.
	6.	Keep outputs in outputs/* or benchmarks/* with per-run subfolders.
	7.	Ensure CI-friendly (no interactive prompts; use flags).
	8.	Add a Makefile target (e.g., submission-bin, benchmark-bin, diagnostics-bin).

Use consistent, human-friendly output (and colors) for key steps; all errors must be actionable.

⸻

🧩 CI/CD Integration
	•	Workflows should call bin/ scripts directly (e.g., smoke test: ./bin/benchmark.sh --profile cpu --epochs 1 --manifest then ./bin/diagnostics.sh --no-umap --no-tsne --manifest).
	•	Upload artifacts (benchmarks/**, outputs/diagnostics/**, logs/v50_debug_log.md) for inspection.
	•	For submission validation pipelines, invoke ./bin/make-submission.sh --dry-run by default.

⸻

📈 Diagrams

make-submission.sh — end-to-end flow

flowchart TD
  A[Start: make-submission.sh] --> B{Self-test passes?}
  B -- no --> Bx[Exit ❌\nprint 'Selftest failed'] --> Z[End]
  B -- yes --> C[Predict\nspectramind predict]
  C --> D[Validate\nspectramind validate]
  D --> E{--dry-run?}
  E -- yes --> Ey[Log would-run bundle cmd\n(no file writes)] --> Z
  E -- no --> F[Bundle\nspectramind bundle → submissions/bundle.zip]
  F --> G{--open?}
  G -- yes --> Gy[Open submissions/\n(open/xdg-open)] --> H
  G -- no --> H[Append run metadata → logs/v50_debug_log.md]
  H --> I[Record: git SHA • hydra cfg hash • timestamps]
  I --> Z[End ✅]

repair_and_push.sh — Git+DVC sync

flowchart TD
  A[Start: repair_and_push.sh] --> B{On main/master\nor --allow-non-main?}
  B -- no --> Bx[Exit ❌\nrefuse non-protected branch] --> Z[End]
  B -- yes --> C[Optional: spectramind test --fast\n(--run-tests)]
  C --> D[DVC status]
  D --> E[git add -A]
  E --> F[dvc add data/* (best-effort)]
  F --> G{Changes detected?}
  G -- no --> Gx[Warn: 'nothing to commit'] --> J
  G -- yes --> H[git commit -m "$MSG"\n(or --allow-empty)]
  H --> J{--no-push?}
  J -- yes --> Jx[Skip pushes → log only] --> P
  J -- no --> K[git push origin <branch>]
  K --> L{Tag requested?}
  L -- yes --> M[git tag -a <TAG> && git push --tags]
  L -- no --> N[Skip tag push]
  M --> O[DVC push (if not --no-dvc)]
  N --> O[DVC push (if not --no-dvc)]
  O --> P[Write manifest (--manifest)\noutputs/manifests/repair_manifest_<RUN_ID>.json]
  P --> Q[Append run metadata → logs/v50_debug_log.md]
  Q --> Z[End ✅]


⸻

✅ Next Steps
	•	Add wrappers for additional flows as needed (e.g., ablate, specialized dashboards).
	•	Ensure each script is wired into Makefile targets and covered by selftest.py.
	•	Keep scripts short; if logic grows, move complexity into Python modules and leave bin/ as orchestration.

⸻

📖 This folder is part of the CLI-first, Hydra-safe, mission-grade SpectraMind V50 ecosystem for the NeurIPS 2025 Ariel Data Challenge.
