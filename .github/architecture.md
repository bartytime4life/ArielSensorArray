🏗️ .github/ Architecture — SpectraMind V50 (Upgraded)

This is the source-of-truth for everything under .github/: CI/CD workflows, issue/PR governance, review automation, and security/compliance guardrails. It encodes NASA-grade reproducibility and CLI-first operation for the NeurIPS 2025 Ariel Data Challenge.

⸻

0) Design Goals
	•	Reproducibility First — every result reconstructable from CLI → Hydra config → run hash → DVC artifact.
	•	CI as Pre-Flight — treat the pipeline like flight hardware: selftest, consistency checks, smoke E2E on every PR.
	•	Kaggle-Aware Discipline — enforce ≤ 9 h GPU budget, deterministic seeds where feasible, explicit perf variance tracking.
	•	Security & Least Privilege — minimal GITHUB_TOKEN scopes, pinned actions, secret hygiene, container/image scanning.

⸻

1) Directory Map (contract)

.github/
├─ README.md                         # Meta & governance (what this folder is)
├─ ARCHITECTURE.md                   # ← You are here (wiring & policies)
├─ CODEOWNERS                        # Auto-review routing by path
├─ CONTRIBUTING.md                   # Contributor rules, local dev, style, tests
├─ SECURITY.md                       # Coordinated disclosure & contacts
├─ SUPPORT.md                        # Support channels & triage pathways
├─ PULL_REQUEST_TEMPLATE.md          # Author-side reproducibility checklist
├─ REVIEW_CHECKLIST.md               # One-pager triage table (for reviewers)
├─ ISSUE_TEMPLATE/
│  ├─ bug_report.yml
│  ├─ feature_request.yml
│  ├─ documentation_request.yml
│  ├─ performance_issue.yml
│  ├─ security_report.yml
│  ├─ config_update.yml
│  ├─ task_tracking.yml
│  └─ README.md
└─ workflows/
   ├─ ci.yml                         # build + quick smoke
   ├─ tests.yml                      # unit/integration matrix
   ├─ diagnostics.yml                # artifacts: GLL heatmap, FFT/UMAP/t-SNE, symbolic overlays
   ├─ ci-dashboard.yml               # job summary → CI_DASHBOARD.md
   ├─ submission.yml                 # pack + validate (manual or tag)
   ├─ kaggle-submit.yml              # guarded dispatch for leaderboard
   ├─ hash-check.yml                 # config composition + DVC pointer integrity
   ├─ docs.yml                       # docs build (MkDocs/Pages bundle)
   ├─ pages.yml                      # publish docs/diagnostics preview
   ├─ lint.yml                       # ruff/black/isort/markdownlint/yamllint
   ├─ bandit.yml                     # python SAST
   ├─ codeql.yml                     # code scanning
   ├─ pip-audit.yml                  # Python vuln scan
   ├─ docker-trivy.yml               # image/package scan
   ├─ hadolint.yml                   # Dockerfile lint
   ├─ artifact-sweeper.yml           # storage hygiene
   ├─ benchmark.yml                  # perf drift (≤ 9 h guard)
   ├─ pr-title-lint.yml              # Conventional PR titles
   ├─ labeler.yml                    # path-based labels
   └─ pr-review-checklist.yml        # bot triage table (idempotent comment)

Companion docs
	•	workflows/README.md — job-by-job details & required checks
	•	ISSUE_TEMPLATE/README.md — template intent and triage flow
	•	PULL_REQUEST_TEMPLATE.md + REVIEW_CHECKLIST.md — author/reviewer contracts

⸻

2) Workflow Architecture

2.1 High-level execution graph

PR opened/synchronized
	•	ci.yml (build + import + quick smoke)
	•	tests.yml (matrix: unit/integration)
	•	diagnostics.yml (plots/artifacts) → ci-dashboard.yml → pages.yml (preview)
	•	Static/security: lint.yml, bandit.yml, codeql.yml, pip-audit.yml, hadolint.yml, docker-trivy.yml
	•	Integrity: hash-check.yml (Hydra/DVC/run-hash)
	•	Reviewer assist: pr-review-checklist.yml (single auto-updated comment)

Tag / release / manual dispatch
	•	submission.yml (pack + validate)
	•	kaggle-submit.yml (guarded manual submit; no-internet runtime parity)

2.2 Minimal permissions (least privilege)

Every workflow declares the narrowest scopes it needs:

permissions:
  contents: read
  pull-requests: write        # only if commenting
  issues: write               # only if posting issue comments
  security-events: write      # only for code scanning upload
  id-token: write             # only for OIDC-based publish

Action pins are commit-SHA locked; marketplace tags are not allowed.

2.3 Concurrency guards

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

Prevents duplicate runners on rapid force-pushes.

2.4 Required checks (branch protection)

Merges to main require:
	•	Build (ci.yml)
	•	Tests (tests.yml matrix)
	•	Static/Security (lint, bandit, codeql, pip-audit, hadolint, docker-trivy)
	•	Integrity (hash-check.yml)
	•	(Optional) Diagnostics artifacts attached for reviewer inspection

⸻

3) Governance: Issues → PRs → Reviews

3.1 Issues (YAML forms)

Each template forces repro context:
	•	Bug — exact spectramind ... CLI, Hydra overrides, config/run hash, environment, log excerpt.
	•	Feature — CLI mapping, Hydra diffs, data/artifact lineage plan, validation metrics to be added.
	•	Performance — metrics table (baseline vs PR), hardware, seeds, variance bounds, Kaggle budget.
	•	Security — safe/dry-run PoC, affected surface, impact, suggested mitigation & acceptance criteria.
	•	Config update — explicit Hydra group/override changes; update hash-check baselines.

3.2 PRs
	•	Authors must include: copy-pasteable CLI, Hydra diffs, DVC stage updates, run hash, plots, runtime numbers, risk notes.
	•	Reviewers use: REVIEW_CHECKLIST.md + full Review Guide.
The bot (pr-review-checklist.yml) posts an idempotent triage comment updated on every sync, edit, label change, or draft toggle.

⸻

4) CI/CD Contracts

4.1 Reproducibility loop
	•	CLI-first: jobs shell out to spectramind subcommands (no notebook-only logic).
	•	Hydra logging: config composition captured & hashed; diff visible in PR.
	•	DVC discipline: large artifacts versioned; no model binaries in Git.
	•	Run hash: emitted to logs/v50_debug_log.md; linked in PR body and CI summary.

4.2 Diagnostics & Dashboard
	•	diagnostics.yml renders: GLL heatmap, FFT/UMAP/t-SNE, symbolic/physics overlays, calibration checks.
	•	ci-dashboard.yml builds a Markdown/HTML summary (status tiles, last runtimes, perf deltas).
pages.yml can publish a preview (PR-scoped) when docs flag is set.

4.3 Kaggle-aware constraints
	•	Enforce ≤ 9 h aggregate GPU budget (w/ seed determinism where feasible).
	•	benchmark.yml detects perf drift; significant slowdowns block merge unless justified.

⸻

5) Security Architecture
	•	Action pinning — all third-party actions are pinned to commit SHAs.
	•	Secret hygiene — secrets never echoed; environment protections enforced; no secrets in PR artifacts.
	•	Scanning suite — bandit, codeql, pip-audit, docker-trivy, hadolint run on PRs and daily on main.
	•	Disclosure — SECURITY.md + security_report.yml for coordinated reporting; 72 h acknowledgement, triage & advisory flow.
	•	Permissions audits — quarterly review of workflow scopes & environment rules.

⸻

6) Automation Inventory

Automation	Purpose	Trigger
pr-review-checklist.yml	Auto-comment reviewer triage (updates in place)	PR open/sync/edit/label
labeler.yml	Path-based labels (areas, stacks, CI, docs/tests)	PR opened/sync
pr-title-lint.yml	Conventional title enforcement	PR opened/edited
artifact-sweeper.yml	Clean aged artifacts (cost control)	Scheduled (weekly)
docs.yml → pages.yml	Build docs & diagnostics preview	PR docs/diag changes
ci-dashboard.yml	Job status → CI dashboard	On workflow completions
benchmark.yml	Track perf drift vs baseline	Nightly + PR (heavy labels)


⸻

7) CODEOWNERS & Review Routing

Keep ownership minimal but complete to avoid orphans:

# Models & training
/src/models/            @team-ml
/src/training/          @team-ml

# Diagnostics, dashboard
/tools/                 @team-dx

# CI/Infra
/.github/               @team-infra

# Configs
/configs/               @team-ml @team-infra


⸻

8) Maintenance & SRE Playbook
	•	Action refresh — quarterly validate pinned SHAs; review release notes for breaking changes.
	•	Permissions audit — quarterly review workflow permissions + environment protections.
	•	Required checks — change only via PR; approval from @team-infra required.
	•	Incident mode — if CI outage, apply temporary label ci-bypass-approved with incident link; remove within 24 h.

⸻

9) Author / Reviewer TL;DR
	•	Authors: submit exact CLI, Hydra diffs, DVC stage updates, run hash, plots, runtime; call out breaking changes, migrations, and risk mitigations.
	•	Reviewers: block merges lacking repro, metrics/plots, Kaggle budget compliance, or with security red flags.

⸻

10) Implementation Notes (how to keep this real)
	•	Hash & integrity: hash-check.yml recomputes config composition hashes, verifies DVC pointers and expected files; PR must update baselines intentionally.
	•	Pinned runners: self-hosted or ubuntu-latest with explicit toolchain versions (Python, Poetry, CUDA) reproducibly installed via setup steps.
	•	Determinism: CI injects seeds via env; reports per-seed variance; regression gates compare medians & 95% CI.
	•	Notebook parity: submission workflows re-use the same library entrypoints; Kaggle kernels are thin wrappers calling spectramind commands; no internet.

⸻