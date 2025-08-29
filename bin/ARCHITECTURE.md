🛰️ SpectraMind V50 — bin/ Architecture

0) Purpose & Scope

The bin/ directory is the operational toolkit for the SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).
It contains entrypoint scripts that wrap the unified CLI (spectramind) with reproducibility, safety, and Kaggle-integration guarantees ￼.

These scripts provide:
	•	One-command workflows (submission, repair, diagnostics).
	•	Guardrails (self-tests, config hashing, logging).
	•	Integration with Git, DVC, Hydra, and Kaggle ￼.
	•	NASA-grade reproducibility (every run logs config, code, data versions) ￼.

In short: bin/ ensures the repository is not just code, but a mission-grade instrument.

⸻

1) Design Philosophy
	•	CLI-first: Every operation goes through spectramind (Typer CLI + Hydra configs) ￼.
	•	Safety nets: All scripts run spectramind test before heavy tasks, preventing invalid runs ￼.
	•	Reproducibility: Append logs to logs/v50_debug_log.md (timestamp, config hash, Git SHA).
	•	Automation-ready: CI-safe, headless, Kaggle-safe ￼.
	•	Glass-box transparency: Intermediate artifacts (predictions, diagnostics, manifests) are DVC-versioned ￼.

⸻

2) Directory Contracts (every script must)
	•	Shebang & strict mode: #!/usr/bin/env bash + set -Eeuo pipefail.
	•	Help & usage: --help prints purpose, options, and examples.
	•	Echo intent: Print the effective command(s) before executing.
	•	Logging: Append a structured line to logs/v50_debug_log.md for every run ￼.
	•	Exit codes: Non-zero on failure; zero on success (see §9).
	•	Idempotence: Safe to re-run; support --dry-run.
	•	No hardcoding: All parameters flow from Hydra configs or CLI overrides ￼.

⸻

3) Core Scripts

3.1 make-submission.sh

Purpose: Full Kaggle submission workflow ￼.
Pipeline:
	1.	spectramind test (integrity check)
	2.	spectramind predict (inference → CSV)
	3.	spectramind validate (sanity checks)
	4.	spectramind bundle (package ZIP for Kaggle)

Options: --dry-run, --open, --tag <string>
Guarantee: Never produces a broken submission — fails fast if checks fail ￼.

⸻

3.2 repair_and_push.sh

Purpose: Repository repair & synchronization.
Workflow:
	1.	dvc status (pipeline consistency)
	2.	Stage changes (git add -A; dvc add data/*)
	3.	Commit (with message or --allow-empty)
	4.	Push Git + DVC (with retries) ￼

Guarantee: Git + DVC states stay in sync.

⸻

3.3 analyze-log.sh

Purpose: Summarize CLI call history ￼.
Pipeline: Parse logs/v50_debug_log.md → export outputs/log_table.md + outputs/log_table.csv.
Options: --since <ISOdate>, --tail <N>, --clean.
Usage: CI dashboards, symbolic violation overlays, trend analysis ￼.

⸻

4) Integration with Repository
	•	Configs: Hydra YAMLs (configs/) drive all runs; scripts never hard-code params ￼.
	•	Data & Models: Managed via DVC; bin/ ensures artifacts are tracked ￼.
	•	Logging: Every run appends a structured line to logs/v50_debug_log.md.
	•	CI/CD: Workflows call bin/make-submission.sh and bin/repair_and_push.sh for submission and repo sync ￼.
	•	Kaggle: Scripts respect runtime limits (≤9h GPU, offline-safe patterns) ￼.

⸻

5) Relation to Scientific Goals
	•	Reproducibility: Each submission links code commit, Hydra config, and DVC artifacts ￼.
	•	Scientific Integrity: Pipeline only bundles spectra validated by symbolic & physics constraints ￼ ￼.
	•	Challenge-readiness: Outputs follow competition schema; bundles are compressible and traceable ￼.

⸻

6) Roadmap Extensions
	•	Add bin/tune.sh — wrap spectramind tune for hyperparameter sweeps ￼.
	•	Add bin/simulate.sh — run forward physics models for cycle consistency ￼.
	•	Extend bin/analyze-log.sh to generate CLI usage heatmaps & symbolic overlays ￼.
	•	Add optional GUI-light launcher bin/open-dashboard.sh for diagnostics HTML ￼ ￼.

⸻

7) Diagrams

7.1 Submission Flow (make-submission.sh)

flowchart TD
  A[Start: make-submission.sh] --> B{Self-test passes?}
  B -- no --> Bx[Exit ❌ print "Selftest failed"] --> Z[End]
  B -- yes --> C[Predict\nspectramind predict]
  C --> D[Validate\nspectramind validate]
  D --> E{--dry-run?}
  E -- yes --> Ey[Log would-run bundle cmd] --> Z
  E -- no --> F[Bundle\nspectramind bundle → submissions/bundle.zip]
  F --> G{--open?}
  G -- yes --> Gy[Open submissions/] --> H
  G -- no --> H[Append run metadata → logs/v50_debug_log.md]
  H --> I[Record git SHA • Hydra hash • timestamp]
  I --> Z[End ✅]

7.2 Repo Repair Flow (repair_and_push.sh)

flowchart TD
  A[Start: repair_and_push.sh] --> B{On main? or --allow-non-main?}
  B -- no --> Bx[Exit ❌ refuse non-main] --> Z[End]
  B -- yes --> C[Optional: spectramind test --fast]
  C --> D[DVC status]
  D --> E[git add -A]
  E --> F[dvc add data/*]
  F --> G{Changes detected?}
  G -- no --> Gx[Warn: nothing to commit] --> J
  G -- yes --> H[git commit -m "$MSG"]
  H --> J{--no-push?}
  J -- yes --> Jx[Skip pushes] --> P
  J -- no --> K[git push origin <branch>]
  K --> L{Tag requested?}
  L -- yes --> M[git tag -a <TAG> && git push --tags]
  L -- no --> N[Skip tag push]
  M --> O[DVC push]
  N --> O[DVC push]
  O --> P[Write manifest (--manifest)]
  P --> Q[Append run metadata → logs/v50_debug_log.md]
  Q --> Z[End ✅]


⸻

8) Option Cheat-Sheets

Tables for make-submission.sh and repair_and_push.sh are preserved (see your draft) with explicit Hydra config notes and Kaggle alignment ￼.

⸻

9) Safety & Failure Modes

Exit codes:
	•	0 success
	•	1 generic failure
	•	2 usage error
	•	3 self-test failed
	•	4 DVC inconsistency
	•	5 push failure ￼

⸻

10) Logging & Manifests

Log schema (single line, Markdown):

[ISO8601] cmd=<script> git=<sha> cfg_hash=<hash> tag=<tag_or_-> pred=<path> bundle=<path> notes="..."

Manifest (JSON):
	•	run_id, ts_utc, git_sha, cfg_hash, device, epochs, seed
	•	inputs/outputs (CSV, bundle paths)
	•	Kaggle fields (competition slug, message)

⸻

11) Quickstart
	•	Dry-run:

./bin/make-submission.sh --dry-run --tag dev-check


	•	Real bundle & open folder:

./bin/make-submission.sh --no-dry-run --open --tag v50.1.0


	•	Repair & push:

./bin/repair_and_push.sh --msg "Symbolic loss clamp fix" --run-tests --run-pre-commit


	•	Summarize logs:

./bin/analyze-log.sh --since 2025-08-01 --tail 50



⸻

12) Summary

The bin/ directory is mission control for SpectraMind V50:
CLI-driven, reproducible, Kaggle-compliant, scientifically rigorous.
Every script here is a reproducibility lock:
	•	no run without logs,
	•	no submission without validation,
	•	no push without DVC sync ￼.

⸻
