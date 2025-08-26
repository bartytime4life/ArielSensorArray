# 🏗️ `.github/` Architecture — SpectraMind V50

This document is the **source-of-truth architecture** for everything under the `.github/` directory:  
workflows (CI/CD), issue/PR governance, review automation, and security/compliance.  
It codifies **NASA-grade reproducibility** and **CLI-first** operation for the NeurIPS 2025 Ariel Data Challenge [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

---

## 0) Design Goals

- **Reproducibility First** — every run is reconstructable from **CLI → Hydra config → run hash → DVC artifact** [oai_citation:2‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL) [oai_citation:3‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w)  
- **CI as Pre-Flight** — treat pipeline like a flight instrument: **selftest**, **consistency checks**, **smoke E2E** on each PR [oai_citation:4‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w)  
- **Kaggle-aware Discipline** — enforce ≤9h GPU budget and deterministic seeds where possible [oai_citation:5‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf)  
- **Security & Least Privilege** — minimal GITHUB_TOKEN perms, pinned actions, secret hygiene [oai_citation:6‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL)

---

## 1) Directory Map (contract)

```text
.github/
├─ README.md                          # Meta & governance (what this folder is)
├─ ARCHITECTURE.md                    # ← You are here (wiring & policies)
├─ CODEOWNERS                         # Auto-review routing by path
├─ CONTRIBUTING.md                    # Contributor rules, local dev, style, tests
├─ SECURITY.md                        # Coordinated disclosure policy & contacts
├─ SUPPORT.md                         # Support channels & triage pathways
├─ PULL_REQUEST_TEMPLATE.md           # Author-side reproducibility checklist
├─ ISSUE_TEMPLATE/
│  ├─ bug_report.yml
│  ├─ feature_request.yml
│  ├─ documentation_request.yml
│  ├─ performance_issue.yml
│  ├─ security_report.yml
│  ├─ config_update.yml
│  ├─ task_tracking.yml
│  └─ README.md
├─ workflows/
│  ├─ ci.yml
│  ├─ tests.yml
│  ├─ diagnostics.yml
│  ├─ ci-dashboard.yml
│  ├─ submission.yml
│  ├─ kaggle-submit.yml
│  ├─ hash-check.yml
│  ├─ docs.yml
│  ├─ pages.yml
│  ├─ lint.yml
│  ├─ bandit.yml
│  ├─ codeql.yml
│  ├─ pip-audit.yml
│  ├─ docker-trivy.yml
│  ├─ hadolint.yml
│  ├─ artifact-sweeper.yml
│  ├─ benchmark.yml
│  ├─ pr-title-lint.yml
│  ├─ labeler.yml
│  └─ pr-review-checklist.yml         # Auto-comment Quick Triage table
└─ REVIEW_CHECKLIST.md                # One-pager triage table (for reviewers)

🔗 Companion docs:
	•	Workflows/README & architecture — job-by-job details
	•	Issue templates/architecture — template intent & triage flow
	•	PR templates & review guide — author/reviewer responsibilities

⸻

2) Workflow Architecture

2.1 Execution graph (high-level)

PR opened/sync
   ├─ ci.yml (build + unit smoke) ─┬─ tests.yml (matrix) ─┬─ diagnostics.yml
   │                               │                     └─ ci-dashboard.yml → pages.yml
   ├─ lint.yml / bandit.yml / codeql.yml / pip-audit.yml / hadolint.yml / docker-trivy.yml
   ├─ hash-check.yml (config hashes, DVC pointers) [oai_citation:7‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w)
   └─ pr-review-checklist.yml (bot comment with triage table)

Tag / release / manual:
   ├─ submission.yml (pack + validate)
   └─ kaggle-submit.yml (guarded, manual-only)

2.2 Minimal Permissions

Every workflow specifies:

permissions:
  contents: read
  pull-requests: write    # only if commenting is required
  security-events: write  # only for code scanning jobs
  id-token: write         # only for OIDC publish steps

Why? Principle of least privilege reduces blast radius ￼.

2.3 Concurrency Guards

All long-running workflows use:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

Prevents duplicate runners on rapid pushes.

2.4 Required Checks (branch protection)

Main requires the following to merge:
	•	ci / build green (sets up env, imports, quick smoke)
	•	tests / unit+integration green (matrix as defined)
	•	lint + bandit + codeql + pip-audit green
	•	hash-check green (Hydra/DVC config integrity) ￼
	•	(optional) diagnostics artifacts uploaded for reviewer inspection

⸻

3) Governance: Issues → PRs → Reviews

3.1 Issues (YAML forms)

Each template enforces complete repro context. Examples:
	•	Bug report requires: exact spectramind CLI, config hash, log excerpt, environment ￼
	•	Feature request requires: CLI mapping + Hydra config changes + artifacts/logging plan ￼
	•	Performance issue requires: metrics table, variance, hardware & seeds (Kaggle-aware) ￼
	•	Security report: safe/dry-run repro, no secrets, mitigation & acceptance criteria ￼

3.2 PRs
	•	Author must use PULL_REQUEST_TEMPLATE.md (CLI commands, Hydra diffs, DVC stages, run hash) ￼
	•	Reviewer uses REVIEW_CHECKLIST.md + full Review Guide; bot posts triage table on every PR
	•	Auto-update triage: .github/workflows/pr-review-checklist.yml updates a single comment via marker

⸻

4) CI/CD Contracts

4.1 Reproducibility loop
	•	CLI-first: all jobs shell out to spectramind subcommands ￼
	•	Hydra logging: config composition is saved + hashed per run ￼
	•	DVC discipline: large artifacts tracked/versioned, no binaries in Git ￼
	•	Run hash: emitted to logs/v50_debug_log.md and linked in PR

4.2 Diagnostics & Dashboard
	•	diagnostics.yml produces GLL heatmaps, FFT/UMAP/t-SNE, symbolic overlays, attaches as artifacts ￼ ￼
	•	ci-dashboard.yml bundles HTML dashboard; pages.yml can publish preview

4.3 Kaggle-aware constraints
	•	Enforce aggregate runtime budget (≤9h for ~1100 planets) and deterministic seeds where feasible ￼
	•	benchmark.yml tracks drift; perf regressions block merge until justified

⸻

5) Security Architecture
	•	Action pinning: all third-party actions pinned to a commit SHA
	•	Secret hygiene: no secrets in repo; use environment-protected secrets for deploy; bots must not echo secrets
	•	Scanning suite: bandit, codeql, pip-audit, docker-trivy, hadolint run on PRs
	•	Disclosure: SECURITY.md + security_report.yml for coordinated, responsible reporting ￼

⸻

6) Automation Inventory

Automation	Purpose	Trigger
pr-review-checklist.yml	Auto-comment reviewer triage table; updates in place	PR open/sync/label
Labeler + PR title lint	Apply area labels, enforce conventional titles	PR opened/edited
Artifact sweeper	Clean aged artifacts (reduce storage costs)	Scheduled (weekly)
Docs + Pages	Build docs & publish dashboard bundle (preview)	PR/docs changes / manual


⸻

7) CODEOWNERS & Review Routing

Use path-based ownership to auto-request reviews:

# Models & training
/src/models/        @team-ml
/src/training/      @team-ml

# Diagnostics, dashboard
/tools/             @team-dx

# CI/Infra
/.github/           @team-infra

# Configs
/configs/           @team-ml @team-infra

Keep ownership minimal but complete to avoid orphaned changes.

⸻

8) Maintenance & SRE Playbook
	•	Action refresh: quarterly validate pinned SHAs & major updates
	•	Permissions audit: quarterly review workflow permissions & env protections
	•	Required checks: adjust only via PR + approval from @team-infra
	•	Incident: if CI outage, set temporary label ci-bypass-approved with link to incident doc; remove within 24h

⸻

9) Author/Reviewer Contracts (TL;DR)
	•	Authors: Provide copy-pasteable CLI, Hydra diffs, DVC stage updates, run hash, plots, runtime numbers ￼ ￼
	•	Reviewers: Block merges lacking repro, metrics/plots, Kaggle budget compliance, or with security red flags ￼ ￼

⸻

10) References
	•	SpectraMind V50 Technical Plan — CLI-first, Hydra, DVC, CI guardrails ￼
	•	SpectraMind V50 Project Analysis — repo structure & reproducibility audit ￼
	•	Hydra for AI Projects — composition, overrides, config hashing ￼
	•	Kaggle Platform Guide — runtime/leaderboard constraints & environment model ￼
	•	Physics/Spectroscopy refs — scientific integrity & symbolic/physics checks ￼ ￼

⸻

Mission Reminder

CI is pre-flight.
Only reproducible, validated, secure, and Kaggle-compliant changes fly.

