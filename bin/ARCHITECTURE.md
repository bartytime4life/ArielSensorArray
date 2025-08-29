🛰️ SpectraMind V50 — bin/ Architecture

0) Purpose & Scope

The bin/ directory is the operational toolkit for the SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).
It contains entrypoint scripts that wrap the unified CLI (spectramind) with reproducibility, safety, and Kaggle integration guarantees.

These scripts provide:
	•	One-command workflows (submission, repair, diagnostics).
	•	Guardrails (self-tests, config hashing, logging).
	•	Integration with Git, DVC, and Kaggle.
	•	NASA-grade reproducibility (every run logs config, code, data versions).

In short: bin/ ensures the repository is not just code, but a mission-grade instrument.

⸻

1) Design Philosophy
	•	CLI-first: Every operation goes through spectramind (Typer CLI + Hydra configs).
	•	Safety nets: All scripts run selftest before heavy tasks, preventing invalid runs.
	•	Reproducibility: Append logs to logs/v50_debug_log.md (timestamp, config hash, Git SHA).
	•	Automation-ready: Scripts are CI-compatible, headless, Kaggle-safe.
	•	Glass-box transparency: Intermediate artifacts (predictions, diagnostics, manifests) are DVC-versioned.

⸻

2) Directory Contracts (what every script must do)
	•	Shebang & strict mode: #!/usr/bin/env bash + set -Eeuo pipefail.
	•	Help & usage: --help prints purpose, options, and examples.
	•	Echo intent: Print the effective command(s) before executing.
	•	Logging: Append a single-line summary to logs/v50_debug_log.md for every run.
	•	Exit codes: Non-zero on failure; zero on success (see §9).
	•	Idempotence: Safe to re-run; use --dry-run where appropriate.
	•	No hardcoding: All parameters flow from Hydra configs or CLI overrides; never bake paths.

⸻

3) Core Scripts

3.1 make-submission.sh

Purpose: Full Kaggle submission workflow.
Pipeline:
	1.	spectramind test (integrity check)
	2.	spectramind predict (inference → CSV)
	3.	spectramind validate (sanity checks)
	4.	spectramind bundle (package ZIP)

Options: --dry-run, --open, --tag <string>
Guarantee: Never produces a broken submission (fails fast if checks fail).

⸻

3.2 repair_and_push.sh

Purpose: Repository repair & synchronization.
Workflow:
	1.	dvc status (pipeline consistency)
	2.	Stage changes (git add -A; dvc add data/* best-effort)
	3.	Commit (message or --allow-empty)
	4.	Push Git + DVC (with retries)

Guarantee: Git + DVC states stay in sync.

⸻

3.3 analyze-log.sh

Purpose: Summarize CLI call history.
Pipeline: Parse logs/v50_debug_log.md → export outputs/log_table.md + outputs/log_table.csv.
Options: --since <ISOdate>, --tail <N>, --clean.
Usage: CI dashboards, trend analysis of runs.

⸻

4) Integration with Repository
	•	Configs: Hydra YAMLs (configs/) drive all runs; scripts never hard-code parameters.
	•	Data & Models: Managed via DVC; bin/ scripts ensure artifacts are tracked.
	•	Logging: Every run appends a concise, structured line to logs/v50_debug_log.md.
	•	CI/CD: Workflows call bin/make-submission.sh for submission validation and bin/repair_and_push.sh for consistent pushes.
	•	Kaggle: Scripts respect runtime limits (≤9h GPU budget, offline-safe patterns).

⸻

5) Relation to Scientific Goals
	•	Reproducibility: Each submission links code commit, Hydra config, and DVC artifacts.
	•	Scientific Integrity: Pipeline only bundles spectra validated by symbolic & physics constraints.
	•	Challenge-readiness: Outputs follow competition schema; bundles are compressible and traceable.

⸻

6) Roadmap Extensions
	•	Add spectramind tune wrapper as bin/tune.sh for hyperparameter sweeps.
	•	Add bin/simulate.sh to run forward physics models for cycle consistency.
	•	Extend bin/analyze-log.sh to generate CLI usage heatmaps & symbolic violation overlays.
	•	Optional GUI-light launcher bin/open-dashboard.sh to open diagnostics HTML.

⸻

7) Diagrams

7.1 make-submission.sh — end-to-end submission flow

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

Notes
	•	Every step writes to logs/v50_debug_log.md (timestamp, Git SHA, Hydra config hash, paths).
	•	--tag <str> is appended to the bundle command and recorded in the log.
	•	--dry-run still runs predict/validate for quick signal, but skips bundle file writes.

⸻

7.2 repair_and_push.sh — repo repair & sync (Git + DVC)

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

Notes
	•	Safe by default: refuses non-main pushes unless --allow-non-main.
	•	Retries on git push/dvc push mitigate transient failures.
	•	--run-pre-commit executes pre-commit hooks prior to staging.
	•	--no-dvc lets you operate in code-only repos.

⸻

8) Option Cheat-Sheets

8.1 make-submission.sh

Option	Type	Default	Description
--dry-run	flag	true	Skip bundle write; still predict/validate for fast feedback.
--open	flag	false	Open submissions/ (macOS open, Linux xdg-open).
--tag <str>	string	“”	Append a version tag to the bundle + log it.

Examples

./bin/make-submission.sh --dry-run --tag v50.0.1
./bin/make-submission.sh --no-dry-run --open --tag v50.0.2


⸻

8.2 repair_and_push.sh

Option	Type	Default	Description
--msg "<text>"	string	(required*)	Commit message (\*unless --allow-empty).
--allow-empty	flag	false	Permit empty commits.
--allow-non-main	flag	false	Allow pushes from non-main branches.
--run-tests	flag	false	Run spectramind test --fast before commit.
--run-pre-commit	flag	false	Run pre-commit hooks before staging.
--no-dvc	flag	false	Disable DVC steps (status/add/push).
--no-push	flag	false	Skip remote pushes; local commit only.
--tag "<vX.Y.Z>"	string	“”	Create and push annotated tag.
--manifest	flag	false	Write JSON manifest under outputs/manifests/.

Examples

./bin/repair_and_push.sh --msg "Cal pipeline hotfix"
./bin/repair_and_push.sh --msg "Diagnostics UX" --run-tests --run-pre-commit
./bin/repair_and_push.sh --msg "Docs" --no-dvc --no-push --allow-non-main
./bin/repair_and_push.sh --msg "v50.0.3" --tag v50.0.3 --manifest


⸻

9) Safety & Failure Modes

Common failure points & responses
	•	Self-test failure: abort immediately; print actionable hint (spectramind test --help).
	•	DVC mismatch: print dvc status diff; suggest dvc repro or re-add artifacts.
	•	Nothing to commit: warn and continue where sensible (e.g., allow tagging).
	•	Push failed: retry with backoff (3x); on final failure, print exact remote error.
	•	Unknown options: print help and exit 2.

Exit codes
	•	0 success
	•	1 generic failure
	•	2 usage / invalid arguments
	•	3 self-test failed
	•	4 DVC inconsistency
	•	5 push failure

⸻

10) Logging & Manifests

Log line schema (single-line append to logs/v50_debug_log.md)

[ISO8601] cmd=<script_name> git=<sha> cfg_hash=<hash> tag=<tag_or_->
pred=<path_or_-> bundle=<path_or_-> notes="<freeform>"

Run manifest (JSON; when --manifest)
	•	run_id, ts_utc, git_sha, cfg_hash, device, epochs, seed
	•	inputs/outputs (pred CSV path, bundle path)
	•	Kaggle fields (competition slug, message) when used

⸻

11) Quickstart

Dry-run a submission

./bin/make-submission.sh --dry-run --tag dev-check

Create a real bundle & open folder

./bin/make-submission.sh --no-dry-run --open --tag v50.1.0

Repair & push (with tests and hooks)

./bin/repair_and_push.sh --msg "Symbolic loss: clamp fix" --run-tests --run-pre-commit

Summarize recent CLI activity

./bin/analyze-log.sh --since 2025-08-01 --tail 50 --md outputs/log_table.md --csv outputs/log_table.csv


⸻

12) Summary

The bin/ directory is mission control for SpectraMind V50: CLI-driven, reproducible, Kaggle-compliant, scientifically rigorous.
Every script here is a reproducibility lock: no run without logs, no submission without validation, no push without DVC sync.

⸻
