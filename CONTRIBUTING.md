
# Contributing to SpectraMind V50 — ArielSensorArray

Welcome, and thank you for considering a contribution to the **SpectraMind V50** project for the NeurIPS 2025 Ariel Data Challenge.  
This guide sets the standards, workflows, and expectations to keep contributions aligned with the project’s goals of **reproducibility, scientific rigor, and engineering excellence**.

---

## 1) Project Principles

- **Reproducibility First** — All code, configs, and data flows must be deterministic, Hydra-configurable, and version-controlled.  
- **Scientific Integrity** — Physics-informed modeling and symbolic rules are mandatory; shortcuts that violate astrophysical realism will not be accepted.  
- **CLI-First Design** — Every function must be callable via the unified `spectramind` CLI.  
- **Auditability** — Every run must log to `logs/v50_debug_log.md` with config hash + Git SHA.  
- **Open Collaboration** — Code, docs, and configs must be clear and accessible to teammates and the scientific community.  

---

## 2) Repository Layout

Key directories:

ArielSensorArray/
├── configs/        # Hydra YAML configs (data, model, training, diagnostics, calibration, logging)
├── src/            # Core source code (pipeline, calibration, diagnostics, CLI)
├── data/           # Raw and processed data (DVC-tracked, not committed directly)
├── outputs/        # Checkpoints, diagnostics, predictions (gitignored/DVC)
├── logs/           # Debug and event logs
├── docs/           # Documentation and MkDocs site
└── tests/          # Unit and integration tests

---

## 3) How to Contribute

### Step 1 — Fork and Branch
```bash
git checkout -b feat/<short-description>

Step 2 — Code Standards
	•	Python 3.10+ with strict type hints
	•	PEP8 + Black formatting
	•	ruff + isort before committing
	•	Full NumPy/SciPy-style docstrings
	•	All features exposed via CLI (spectramind.py)
	•	No hardcoded constants — everything must live in Hydra configs

Step 3 — Logging & Reproducibility
	•	Every CLI call must append to logs/v50_debug_log.md
	•	Always log: Git SHA, config hash, seed, dataset version
	•	Use deterministic seeds for toy/CI configs

Step 4 — Tests
	•	Add unit tests in tests/
	•	Run:

pytest -q


	•	Cover new modules with CLI-integrated tests
	•	CI will reject contributions that break reproducibility or tests

Step 5 — Documentation
	•	Update docs/ and ARCHITECTURE.md
	•	New CLI commands must be documented in docs/cli.md
	•	Update CHANGELOG.md if functionality changes

Step 6 — Commit and Push
	•	Follow Conventional Commits:
	•	feat(model): add symbolic influence map
	•	fix(cli): correct config hash logging
	•	docs: update calibration step in architecture
	•	Push and open a Pull Request (PR) against main

⸻

4) Review Process

All PRs must pass:
	•	spectramind selftest
	•	GitHub Actions CI (.github/workflows/ci.yml)
	•	Pre-commit hooks (ruff, black, isort, YAML)

Maintainers may request:
	•	Benchmark evidence (≤ 9 hrs runtime on Kaggle full dataset)
	•	Reproducibility demo (re-run via config hash + Git SHA)
	•	Extra tests or documentation

⸻

5) Contribution Etiquette
	•	Be respectful and collaborative — we’re building a mission-grade scientific system.
	•	Write explicit comments/docstrings; no “magic” code.
	•	Open an Issue before large changes.
	•	Use exp/ branches and [Draft] PRs for experiments.

⸻

6) Adding New Features

Ask first:
	1.	Is it physics-informed, symbolic, reproducible?
	2.	Can it be expressed as a Hydra config or CLI flag?
	3.	Is it testable, loggable, diagnosable?

If yes, proceed. If not, propose in an Issue first.

⸻

7) License & Attribution
	•	Contributions are licensed under MIT License
	•	Cite via CITATION.cff if used in research

⸻

8) Pre-PR Checklist
	•	Code formatted (black, ruff, isort)
	•	Configs updated & tested
	•	spectramind selftest passes
	•	Tests added & passing (pytest)
	•	Documentation updated
	•	Commit message follows convention

⸻

Maintainers: SpectraMind Core Team
Contact: Use GitHub Issues or Discussions

---
