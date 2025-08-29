# 🛰️ SpectraMind V50 — `bin/` Directory

The `bin/` directory contains **executable scripts** for orchestrating and maintaining the SpectraMind V50 pipeline.  
These scripts provide **CLI-first automation** of reproducible tasks — wrapping around the Typer CLI (`spectramind`) and Hydra configurations to ensure end-to-end reproducibility.

---

## 📌 Purpose

- Centralize all shell and helper scripts that are **outside of Python modules**.
- Provide **shortcuts** for developers and CI/CD pipelines (e.g., `make-submission.sh`, `analyze-log.sh`).
- Maintain repository integrity by ensuring common workflows are accessible with **one command**.
- Guarantee that everything here is **CLI-first, Hydra-safe, and reproducibility-compliant** [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK) [oai_citation:2‡Strategy for Updating and Extending SpectraMind V50 for NeurIPS 2025 Ariel Challenge.pdf](file-service://file-9VcfypHeuBNRcRQdCMaKS4).

---

## 📂 Contents

- `analyze-log.sh` — Parse and summarize CLI history from `logs/v50_debug_log.md` into Markdown/CSV tables.
- `make-submission.sh` — Bundle predictions, diagnostics, and manifests into a Kaggle-ready submission package.
- `repair_and_push.sh` — Git/DVC sync helper for patching and pushing consistent states.
- `phase_x_polish.sh` — Phase-end checklist: lint, tests, selftest, DVC repro, diagnostics, submission dry-run.
- `.gitkeep` — Placeholder file to keep the `bin/` directory tracked when empty.

---

## ⚙️ Usage

All scripts are executable from the root of the repository:

```bash
./bin/analyze-log.sh --md outputs/log_table.md --csv outputs/log_table.csv
./bin/make-submission.sh --dry-run
./bin/repair_and_push.sh "Fix calibration pipeline hashes"

Scripts assume:
	•	You are running from the repo root.
	•	Python virtualenv/Docker image is already activated with dependencies installed.
	•	spectramind CLI commands are available in PATH.

⸻

🧭 Design Principles
	•	CLI-First: Every major operation must be runnable from spectramind Typer CLI; scripts here only orchestrate combinations.
	•	Hydra-Safe: All run parameters live in configs/*.yaml, not hard-coded in scripts.
	•	Reproducibility: Every script logs its activity into logs/v50_debug_log.md with timestamps and config hashes.
	•	Transparency: Scripts are documented, auditable, and minimal — no hidden behavior.

⸻

🛠️ Developer Notes
	•	New scripts must include:
	•	#!/usr/bin/env bash header
	•	Top-of-file docstring with purpose, usage, and options
	•	Safe defaults (e.g., --dry-run mode)
	•	Logging to logs/v50_debug_log.md
	•	Use Rich CLI outputs and consistent formatting for user-facing messages.
	•	Integration with CI: All critical scripts should be invocable in .github/workflows/*.yml.

⸻

✅ Next Steps
	•	Add wrapper scripts for:
	•	spectramind diagnose dashboard
	•	spectramind ablate
	•	spectramind analyze-log
	•	Ensure each script passes selftest.py checks and is included in Makefile targets.

⸻

📖 This folder is part of the SpectraMind V50 CLI-first ecosystem for the NeurIPS 2025 Ariel Data Challenge.
