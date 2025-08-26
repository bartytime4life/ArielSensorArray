# 🏗️ Architecture — Issue Templates for SpectraMind V50

This document describes the **architecture, intent, and flow** of the issue templates in  
`.github/issue_template/` for the **SpectraMind V50** repository.  
These templates enforce **structured, reproducible reporting** and align with our pipeline’s **NASA-grade rigor**.

---

## 1. System Goals

- **Consistency** — All issues follow the same structured format.  
- **Reproducibility** — Every bug/feature request includes configs, logs, and hashes.  
- **Scientific Traceability** — Issues capture expected vs. actual scientific/physical behavior.  
- **Fast Triage** — Maintainers get all context up front (CLI calls, config hashes, workflow run links).  

---

## 2. Template Topology

```

.github/
└── issue\_template/
├── bug\_report.yml
├── feature\_request.yml
├── documentation\_request.yml
├── performance\_issue.yml
├── security\_report.yml
├── config\_update.yml
├── task\_tracking.yml
└── README.md

```

Each YAML file defines a **GitHub Issue Form**, using required fields, dropdowns, and checkboxes.  
This ensures completeness while keeping inputs consistent across contributors.

---

## 3. Workflow Integration

### Trigger Path

```

Contributor → New Issue → Select Template → Fill Required Fields →
Template captures CLI logs + config hash → Maintainers triage →
Workflows link back to issue (CI, Kaggle runs) → Resolution documented

````

### Cross-links

- `bug_report.yml` requires attaching:
  - CLI command (`spectramind …`)
  - `run_hash_summary_v50.json` entry
  - Logs (`logs/v50_debug_log.md`)
- `feature_request.yml` links to **Hydra configs** and **workflows** for feasibility.
- `security_report.yml` routes to maintainers under `.github/SECURITY.md`.
- All templates recommend attaching **workflow run links** from `.github/workflows/`.

---

## 4. Template Roles

| Template                   | Purpose                                                                 | Critical Fields                                                      |
| -------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `bug_report.yml`           | Capture CLI or pipeline bugs.                                           | Steps to reproduce, CLI call, config hash, log excerpt.              |
| `feature_request.yml`      | Propose new scientific/CLI features.                                    | Motivation, expected outcome, reproducibility impact.                |
| `documentation_request.yml`| Request doc updates (README, guides, manifests).                       | File/section, missing detail, suggested addition.                    |
| `performance_issue.yml`    | Report bottlenecks or runtime failures (e.g., Kaggle 9h limit).         | Hardware, runtime duration, profiling logs, expected vs. actual time. |
| `security_report.yml`      | Private vulnerability disclosure.                                       | Environment, dependency version, vulnerability details.               |
| `config_update.yml`        | Suggest Hydra/DVC config changes.                                       | Config file path, before/after snippet, reason.                       |
| `task_tracking.yml`        | Track infrastructure/maintenance chores.                               | Task type, description, expected deliverable.                         |

---

## 5. Alignment with V50 Principles

- **CLI-First** — Issues always reference `spectramind` commands.  
- **Hydra-Safe Configs** — Configs attached to issues ensure repeatable runs.  
- **Integrated Logging** — Required `v50_debug_log.md` excerpt connects issues to runtime.  
- **Scientific Integrity** — Bug reports & feature requests document physical/scientific expectations.  
- **Reproducibility Guardrails** — Every issue can be replayed via configs + hashes.  

---

## 6. Maintenance Strategy

- **Quarterly review** — Ensure templates match current CLI subcommands and Hydra configs.  
- **Versioning** — When new CLI or pipeline features are added, update relevant templates.  
- **Security** — Keep `security_report.yml` minimal but mandatory for vulnerabilities.  
- **Expansion** — Add templates for new domains (e.g. symbolic logic, dashboard tasks) as the repo evolves.  

---

## 7. Example Flow (Bug Report)

1. User runs:

   ```bash
   spectramind train --config configs/model/v50.yaml
````

2. Training fails. User opens a **Bug Report**:

   * Fills in **CLI call**, **config hash**, and **excerpt from v50\_debug\_log.md**.
   * Attaches workflow run link from GitHub Actions.

3. Maintainer triages:

   * Confirms config hash reproducibility.
   * Re-runs via CI workflow.
   * Links fix commit back to the issue.

---

## 8. Summary

The **issue template architecture** is a **frontline reproducibility guard**:
it transforms ad-hoc reports into **structured, reproducible tickets**.
Every issue carries its own scientific record (CLI command, config, logs),
ensuring SpectraMind V50 evolves with **traceability, rigor, and speed**.

---

```
