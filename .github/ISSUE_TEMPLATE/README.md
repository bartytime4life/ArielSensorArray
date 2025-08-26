# 📝 GitHub Issue Templates — SpectraMind V50

This directory (`.github/issue_template/`) contains **GitHub Issue Form templates** for managing all issues in the **SpectraMind V50** repository.  
Templates enforce a **structured, reproducible, and scientifically rigorous** approach to reporting bugs, proposing features, or updating documentation.

---

## 📂 Templates

| File                        | Purpose                                                                 |
| --------------------------- | ----------------------------------------------------------------------- |
| `bug_report.yml`            | For reporting **pipeline/CLI bugs**. Requires CLI command, config hash, and logs. |
| `feature_request.yml`       | For suggesting **new features, modules, or diagnostics**. Captures motivation, scientific context, and reproducibility impact. |
| `documentation_request.yml` | For requesting **README, guide, or manifest updates**. Includes expected location and missing details. |
| `performance_issue.yml`     | For reporting **runtime bottlenecks, memory leaks, or Kaggle runtime-limit failures**. Requires profiling evidence or metrics. |
| `security_report.yml`       | For reporting **security vulnerabilities or unsafe dependencies**. Routed to maintainers under the security policy. |
| `config_update.yml`         | For suggesting changes to **Hydra/DVC configuration files or overrides**. |
| `task_tracking.yml`         | For general tracking of **infrastructure tasks, chores, or CI/CD improvements**. |

All forms are written in **YAML Issue Forms** format, supported by GitHub’s [issue template system](https://docs.github.com/en/issues/building-community/using-templates-to-encourage-useful-issues).

---

## 🚀 How to Use

When opening a new issue:

1. Click **“New Issue”** in the repository.
2. Select the most relevant template from the dropdown.
3. Complete **all required fields** (marked `*`).
4. Attach supporting artifacts:
   - CLI command (`spectramind train …`)
   - Config hash (`run_hash_summary_v50.json`)
   - Logs (`logs/v50_debug_log.md`)
   - Artifact IDs or workflow run links (for CI/Kaggle)

---

## 🔒 Security Policy

- Use **`security_report.yml`** for vulnerabilities.  
- Do **not** disclose security issues publicly until triaged.  
- Attach sensitive logs/configs privately if needed.  

See **`.github/SECURITY.md`** for full details.

---

## 🧭 Alignment with SpectraMind V50 Principles

- **Reproducibility First** — Every issue must include configs, logs, or hashes.  
- **Scientific Integrity** — Issues document expected vs. actual results scientifically.  
- **Automated Usability** — Clear templates → faster CI/CD integration → reproducible fixes.  
- **Integrated Logging** — Always link back to `v50_debug_log.md` entries.  

---

## 🧹 Maintenance

- Keep templates aligned with **CLI changes** (e.g. new subcommands).  
- Add new templates when we introduce new workflows (e.g. symbolic logic, dashboard tasks).  
- Review templates quarterly to ensure they match current **SpectraMind V50 protocols**.  

---

📌 **Reminder**: If your issue relates to **Kaggle or CI workflows**, always attach the **workflow run link** from `.github/workflows/`.  
This helps maintainers trace the problem quickly and preserve our **NASA-grade reproducibility**.
