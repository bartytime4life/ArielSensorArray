```markdown
# 🏗️ Architecture — GitHub Actions for SpectraMind V50

This document explains **how the workflows in `.github/workflows/` fit together** to deliver CI/CD, scientific diagnostics, and reproducibility guarantees for the SpectraMind V50 pipeline.

> TL;DR: Every path runs through the **`spectramind` CLI** with Hydra configs and DVC-tracked data. CI verifies integrity on every PR, nightly runs build dashboards, and guarded jobs produce and (optionally) submit Kaggle bundles.

---

## 1) System Overview

### Goals
- **Reproducibility-first**: same CLI, same config ⇒ same result.
- **Defense-in-depth**: security scans, least-privilege tokens, SBOM.
- **Auditability**: artifact logs, hash manifests, dashboard builds.
- **Reliability**: required checks and branch protections on `main`.

### Workflow Families
- **Core CI/QA**: `ci.yml`, `tests.yml`, `lint.yml`, `bandit.yml`, `codeql.yml`, `pip-audit.yml`
- **Pipeline Runs & Diagnostics**: `calibration-ci.yml`, `nightly-e2e.yml`, `diagnostics.yml`, `ci-dashboard.yml`, `hash-check.yml`
- **Release & Docs**: `docs.yml`, `pages.yml`, `release.yml`
- **Security & Supply Chain**: `docker-trivy.yml`, `hadolint.yml`, `sbom-refresh.yml`
- **Automation & Hygiene**: `artifact-sweeper.yml`, `branch-protection.yml`, `dependabot-auto-merge.yml`, `labeler.yml`, `pr-title-lint.yml`, `stale.yml`
- **Perf & Viz**: `benchmark.yml`, `mermaid-export.yml`
- **Kaggle**: `submission.yml`, `kaggle-submit.yml` (manual, guarded)

---

## 2) Trigger Topology

```

pull\_request ->  ci.yml, tests.yml, lint.yml, bandit.yml, hash-check.yml
push(main)  ->  ci.yml, docs.yml (if docs changed), ci-dashboard.yml (if diag changed)
schedule    ->  nightly-e2e.yml, benchmark.yml, sbom-refresh.yml, docker-trivy.yml
workflow\_dispatch -> submission.yml, kaggle-submit.yml, diagnostics.yml

```

> All workflows share a **common setup step**: checkout, Python toolchain, cache restore, and `poetry install` (or `pip`), then run the **CLI**.

---

## 3) Job Graph (Conceptual)

```

```
      PR opened
          |
     +----v----+
     |  ci.yml |--(needs)--> lint.yml
     +----+----+             bandit.yml
          |                   tests.yml
          |                   hash-check.yml
          v
 status checks pass? ----> required for merge to main
```

Nightly (cron):
nightly-e2e.yml -> diagnostics.yml -> ci-dashboard.yml -> pages.yml

Manual:
submission.yml -> (validate) -> artifact upload
kaggle-submit.yml -> (needs submission) -> submit via token (guarded)

````

---

## 4) Conventions

### Naming
- **Workflows**: _kebab-case_, concise (`hash-check.yml`, `artifact-sweeper.yml`).
- **Jobs**: `setup`, `lint`, `unit`, `integration`, `package`, `publish`.
- **Artifacts**: `logs-<sha>`, `diagnostics-<date>`, `submission-<tag>`.

### Paths
- Workflows live **only** under `.github/workflows/`.
- Non-workflow GitHub files (e.g., `FUNDING.yml`, `SECURITY.md`, `PULL_REQUEST_TEMPLATE.md`) sit in `.github/`.

---

## 5) Reusable Patterns

### 5.1 Caching
Use modular caches to speed runs:

```yaml
- uses: actions/setup-python@v5
  with: { python-version: '3.11' }

- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      .venv
    key: ${{ runner.os }}-py311-${{ hashFiles('**/poetry.lock', '**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-py311-
````

### 5.2 Matrix

Run tests across platforms/versions when needed:

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest]
    py: ['3.10', '3.11']
runs-on: ${{ matrix.os }}
steps:
  - uses: actions/setup-python@v5
    with: { python-version: ${{ matrix.py }} }
```

### 5.3 Artifacts

* Upload minimal but sufficient logs (keep under retention).
* Prefer compressed artifacts; include **config snapshots**, **CLI logs**, **reports**.

```yaml
- uses: actions/upload-artifact@v4
  with:
    name: logs-${{ github.sha }}
    path: |
      logs/**
      outputs/reports/**
    retention-days: 14
```

---

## 6) Security Model

### Permissions & Concurrency (baseline)

Add to every workflow:

```yaml
permissions:
  contents: read
  actions: read
  checks: read
  security-events: write  # only when needed
  id-token: write         # only if using OIDC for registry or cloud

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### Hardening

* **Protected branches**: `main` requires passing checks (CI, lint, tests, hash-check, bandit).
* **Environment protection** for publishing (Pages, Releases, Submissions).
* **Dependabot** + **auto-merge** only after checks pass.
* **Secrets**: read by deployment jobs only; never in logs.
* **Image & SBOM**: `docker-trivy.yml`, `sbom-refresh.yml` run on a cadence.

---

## 7) Environments & Secrets

Define environments in repo settings:

* `staging`: pages deploys, dashboard publish.
* `release`: tag & GitHub Release permissions.
* `kaggle`: holds Kaggle API token; guarded by reviewers.

Use `environment` on jobs that publish:

```yaml
environment:
  name: kaggle
  url: https://www.kaggle.com/
```

Secrets expected:

* `KAGGLE_USERNAME`, `KAGGLE_KEY` (for guarded submit).
* `PAGES_TOKEN` or built-in `GITHUB_TOKEN` with Pages.
* Registry/cloud OIDC setup if pushing images.

---

## 8) Workflow Details (by family)

### 8.1 Core CI

* **ci.yml**: quick build + smoke checks (`spectramind --version`, config probe).
* **tests.yml**: unit/integration/regression via `pytest -q`.
* **lint.yml**: `ruff`, `black --check`, `isort --check-only`, `mypy` (optional).
* **bandit.yml**: static security scan.
* **codeql.yml**: code scanning alerts.
* **pip-audit.yml**: Python dependency CVE scan.

### 8.2 Pipeline & Diagnostics

* **calibration-ci.yml**: verifies calibration stage CLI on sample inputs.
* **nightly-e2e.yml**: runs a small end-to-end job and collects outputs.
* **diagnostics.yml**: FFT/UMAP/SHAP/GLL analyses; uploads plots + JSON reports.
* **ci-dashboard.yml**: builds an HTML dashboard; hands off to `pages.yml`.
* **hash-check.yml**: validates `run_hash_summary_v50.json` & manifests.
* **submission.yml**: packages submission zip and validates schema.
* **kaggle-submit.yml**: manual, requires `kaggle` env and maintainer approval.

### 8.3 Docs & Release

* **docs.yml**: build docs (MkDocs/Sphinx).
* **pages.yml**: push built site or dashboard artifacts to GitHub Pages.
* **release.yml**: tag, changelog, upload artifacts (submission, dashboard snapshot, SBOM).

### 8.4 Supply Chain

* **docker-trivy.yml**: scan base and app images.
* **hadolint.yml**: Dockerfile style & best practices.
* **sbom-refresh.yml**: CycloneDX/Syft JSON uploaded as artifact.

### 8.5 Automation

* **artifact-sweeper.yml**: deletes old redundant artifacts (retention policy).
* **branch-protection.yml**: ensures rulesets remain enforced (organization dependent).
* **dependabot-auto-merge.yml**: safe upgrade path with tests as gate.
* **labeler.yml**: PR labeling for review routing.
* **pr-title-lint.yml**: enforce Conventional Commits.
* **stale.yml**: triage dormant issues/PRs.

### 8.6 Performance & Viz

* **benchmark.yml**: micro/meso perf timing; posts PR comment deltas.
* **mermaid-export.yml**: exports architecture graphs (workflow/DAG diagrams) into artifacts and docs.

---

## 9) Retention & Cost Controls

* **Artifacts**: default 14 days for logs/diagnostics; 90 days for releases.
* **Pages**: keep last successful build.
* **Cache**: segment by lock hash; restore-keys to reduce misses.
* **Nightlies**: compress outputs; prefer JSON over CSV where possible.

---

## 10) PR Policy & Required Checks

**Required checks on PR → main** (example):

* `ci / build`
* `tests / unit+integration`
* `lint / style`
* `security / bandit`
* `hash-check / manifests`

Also enabled:

* **Dismiss stale reviews** on new commits.
* **Require linear history** (optional).
* **Require signed commits** (recommended).

---

## 11) Failure Modes & Troubleshooting

* **Cache miss / long installs**: update `restore-keys`, pin lockfiles, check Python version drift.
* **Hash-check fails**: re-generate `run_hash_summary_v50.json`, confirm paths in workflow.
* **Pages deploy fails**: verify build artifact name matches `pages.yml` input; environment permissions.
* **Kaggle submit blocked**: ensure secrets scoped to `kaggle` environment; reviewer approval present.
* **CodeQL timeouts**: reduce language matrix or schedule separate scan job.
* **SBOM upload large**: compress JSON (`.json.gz`) and note in artifact description.

---

## 12) Migration Guide

* **Adding a new workflow**: copy `permissions` + `concurrency` block; reuse the common setup composite action if present; add to README index.
* **Renaming**: keep kebab-case; update required checks in branch protection.
* **Secrets**: rotate and rebind to environments; avoid plain `secrets.*` in non-deploy jobs.
* **Refactoring**: prefer **reusable workflows** (call via `workflow_call`) for common setup.

---

## 13) Example: Common Setup Snippet

```yaml
jobs:
  setup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Cache deps
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/pip
            .venv
          key: ${{ runner.os }}-py311-${{ hashFiles('**/poetry.lock', '**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-py311-
      - name: Install
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -U pip
          if [ -f poetry.lock ]; then pip install poetry && poetry install --no-interaction; else pip install -r requirements.txt; fi
      - name: Show CLI
        run: |
          source .venv/bin/activate
          spectramind --version || true
```

---

## 14) Design Principles (Recap)

* **Single entrypoint**: *everything* flows through `spectramind` CLI.
* **Config, not code**: Hydra-managed YAMLs; configs are artifacts.
* **Security by default**: least privilege, guarded envs, scans on schedule.
* **Artifacts as evidence**: logs, hashes, diagnostics uploaded every run.
* **Docs are living**: workflows own their README and this architecture file.

---

**Questions or updates?**
Open a small PR that modifies both: `README.md` (index) and this `architecture.md` (wiring). Keep them in lockstep so contributors always have a coherent map.

```
```
