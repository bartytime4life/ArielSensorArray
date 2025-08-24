
# Contributing to SpectraMind V50 — ArielSensorArray  
**Neuro-Symbolic, Physics-Informed AI Pipeline for the NeurIPS 2025 Ariel Data Challenge**

---

## 🌟 Philosophy

SpectraMind V50 is engineered for **NASA-grade reproducibility**, **neuro-symbolic scientific rigor**, and **CLI-first usability**.  
All contributions must align with these principles:

- **Reproducibility First** → every result must be re-runnable (Hydra configs, DVC data hashes, Docker/Poetry environments).  
- **Documentation-First** → code without docstrings or diagnostics is incomplete.  
- **Scientific Integrity** → no hacks; physics-informed modeling and symbolic constraints must be respected.  
- **Automation Everywhere** → every workflow reachable via CLI, CI, and diagnostics dashboards.  

---

## 🛠️ Development Workflow

### 1. Fork & Branch
- Fork this repo and create feature branches from `main`.
- Use naming convention:  
  - `feat/<feature>` (new functionality)  
  - `fix/<issue>` (bug fix)  
  - `docs/<section>` (documentation)  
  - `exp/<experiment>` (exploratory trial, subject to cleanup)  

### 2. Environment Setup
- Use **Poetry** for dependencies:
  ```bash
  poetry install

	•	Use DVC + lakeFS for dataset synchronization:

dvc pull


	•	Use Docker or Ubuntu setup guide for system parity.

3. Coding Standards
	•	Python ≥3.10 required.
	•	Enforce style via pre-commit hooks:

pre-commit install
pre-commit run --all-files


	•	Linting & formatting stack:
	•	ruff (static analysis + autofix)
	•	black (formatting)
	•	isort (imports, black profile)
	•	mypy (type checking)
	•	Docstrings: All public functions/classes must use Google or NumPy style docstrings with explicit typing.
	•	Logging: Use logging with Rich formatting. Never print().

4. Config Standards
	•	All parameters must live in Hydra YAML configs (configs/).
	•	Never hardcode constants in code — use Hydra defaults & overrides.
	•	Use group syntax (model/, data/, training/, diagnostics/).
	•	Document every new config with inline comments.

⸻

✅ Pre-Submission Checklist

Before opening a PR, run:

make selftest     # Runs spectramind test --deep
make lint         # Runs ruff, black, isort, mypy
make diagnostics  # Generates small dashboard slice

Confirm that:
	•	✅ All CLI commands (spectramind <subcommand>) run without error
	•	✅ Hydra configs resolve with no missing defaults
	•	✅ DVC stages (dvc repro) succeed on toy splits
	•	✅ v50_debug_log.md captures your CLI calls
	•	✅ Unit tests (pytest) all pass

⸻

🔬 Scientific Requirements
	•	Physics-informed models: All modeling must respect physical realism (no negative flux, spectra smoothness).
	•	Symbolic constraints: All training/evaluation must log symbolic losses and violations.
	•	Uncertainty: If your code outputs μ spectra, you must also implement σ calibration (temperature scaling, COREL, or conformal).
	•	Diagnostics: Every new module must integrate with at least one diagnostic visualization (FFT, SHAP, symbolic overlay, etc.).

⸻

📦 Submissions & CI

CI/CD is enforced via GitHub Actions:
	•	Linting (lint.yml)
	•	Diagnostics smoke tests (diagnostics.yml)
	•	Submission packaging (kaggle-submit.yml)
	•	End-to-End nightly run (nightly-e2e.yml)

Your branch will not merge unless CI passes.
Use make ci-smoke locally to preview.

⸻

🧪 Testing
	•	Add unit tests under tests/ for every new function or bugfix.
	•	Use pytest and ensure coverage is ≥90% for scientific modules.
	•	New CLI subcommands must include:
	•	--help coverage test
	•	--dry-run test
	•	CLI log entry in v50_debug_log.md

⸻

📖 Documentation
	•	Update README.md and ARCHITECTURE.md if your change alters project structure.
	•	Add diagrams (Mermaid or Graphviz) to assets/diagrams/ if relevant.
	•	Any scientific upgrade should be reflected in the PDF knowledge docs (AI Design, Modeling, Radiation, Lensing, etc.).

⸻

📝 Commit Conventions

Follow Conventional Commits:

feat(train): add gradient accumulation and resume support
fix(corel): correct σ coverage plot normalization
docs(gui): expand dashboard section with t-SNE integration

	•	Prefix: feat, fix, docs, test, ci, refactor, chore.
	•	Scope: folder or subsystem (train, corel, cli, gui, data, etc.).
	•	Subject: short imperative sentence.

⸻

🎯 Pull Requests
	•	Fill out the PR template.
	•	Reference related issues (Fixes #123).
	•	Include screenshots/plots if adding diagnostics.
	•	Keep PRs small and focused. Large changes must be split.

⸻

🌍 Community Guidelines
	•	Be respectful and constructive in reviews.
	•	Share scientific reasoning for changes (include equations or citations if applicable).
	•	Kaggle discussions & community notebooks may be linked in PRs for context.

⸻

🚀 Quickstart Commands for Contributors

# Run calibration on toy split
spectramind calibrate +data.split=toy

# Train small model
spectramind train model=v50_small.yaml training.epochs=2

# Run diagnostics dashboard
spectramind diagnose dashboard --no-umap --fast

# Package submission
spectramind submit --selftest --bundle


⸻

🤝 Attribution

This project is guided by:
	•	ESA Ariel Mission data simulation
	•	NeurIPS 2025 Ariel Data Challenge rules
	•	NASA-grade reproducibility practices

All contributors will be listed in CITATION.cff and included in leaderboard acknowledgements.

⸻

Thank you for helping make SpectraMind V50 the leading open scientific AI pipeline for exoplanet discovery!

---
