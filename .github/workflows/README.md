# ⚙️ GitHub Actions Workflows — SpectraMind V50

This folder contains all **GitHub Actions workflows** that power the continuous integration (CI), continuous delivery (CD), diagnostics, reproducibility checks, and Kaggle submission automation for the **SpectraMind V50** pipeline (NeurIPS 2025 Ariel Data Challenge).

Workflows are designed to ensure:
- **NASA-grade reproducibility** (Hydra configs, DVC, MLflow logging).
- **Terminal-first integrity** (`spectramind` CLI is the single entrypoint).
- **Security and auditability** (hash logs, artifact retention, SBOM scans).
- **End-to-end automation** from calibration → training → diagnostics → submission.

---

## 📂 Workflow Index

### ✅ CI / Testing / Quality
- `ci.yml` — Core CI build and test.
- `tests.yml` — Full test suite (unit, integration, regression).
- `lint.yml` — Python linting and style checks.
- `bandit.yml` — Static security lint.
- `codeql.yml` — GitHub CodeQL static analysis.
- `pip-audit.yml` — Dependency vulnerability scan.

### 🔬 Pipeline & Diagnostics
- `calibration-ci.yml` — Runs calibration stage checks.
- `nightly-e2e.yml` — Nightly end-to-end sample run.
- `diagnostics.yml` — Scientific diagnostics (FFT, SHAP, UMAP, GLL).
- `ci-dashboard.yml` — Build & upload diagnostics dashboard.
- `hash-check.yml` — Validate `run_hash_summary_v50.json` integrity.
- `submission.yml` — Build & validate Kaggle submission ZIP.
- `kaggle-submit.yml` — Manual Kaggle submission trigger (with guardrails).

### 📦 Packaging, Docs & Releases
- `docs.yml` — Build documentation (MkDocs/Sphinx).
- `pages.yml` — Publish docs/diagnostics to GitHub Pages.
- `release.yml` — Automate release tagging & artifact upload.

### 🛡️ Containers, SBOM & Security
- `docker-trivy.yml` — Scan images with Trivy.
- `hadolint.yml` — Lint Dockerfiles.
- `sbom-refresh.yml` — Generate Software Bill of Materials.

### 🧹 Housekeeping & Automation
- `artifact-sweeper.yml` — Auto-cleanup of old CI artifacts.
- `branch-protection.yml` — Enforce branch rules.
- `dependabot-auto-merge.yml` — Merge safe Dependabot PRs.
- `labeler.yml` — Auto-label PRs by file patterns.
- `pr-title-lint.yml` — PR title lint (conventional commits).
- `stale.yml` — Mark inactive issues/PRs as stale.

### 📊 Benchmarking & Visualization
- `benchmark.yml` — Performance regression detection.
- `mermaid-export.yml` — Export Mermaid DAG/architecture diagrams.

---

## ⚠️ Misplaced Files

The following belong at `.github/` (not inside `workflows/`):
- `CONTRIBUTING.md` → `.github/CONTRIBUTING.md`
- `SECURITY.md` → `.github/SECURITY.md`
- `FUNDING.yml` → `.github/FUNDING.yml`
- `PULL_REQUEST_TEMPLATE.md` → `.github/PULL_REQUEST_TEMPLATE.md`

Keeping only workflows here avoids confusion and ensures all files are runnable by GitHub Actions.

---

## 🔐 Best Practices (apply to all workflows)

```yaml
permissions:
  contents: read
  actions: read
  checks: read
  security-events: write   # only if needed
  id-token: write          # only if using OIDC

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
