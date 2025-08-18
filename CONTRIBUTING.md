# Contributing to SpectraMind V50 — ArielSensorArray

Welcome, and thank you for considering a contribution to the **SpectraMind V50** project for the NeurIPS 2025 Ariel Data Challenge.  
This guide outlines the standards, workflows, and expectations that ensure all contributions maintain the project’s goals of **reproducibility, scientific rigor, and engineering excellence**.

---

## 1. Project Principles

- **Reproducibility First** — All code, configs, and data flows must be deterministic, Hydra-configurable, and version-controlled.  
- **Scientific Integrity** — Physics-informed modeling and symbolic rules are mandatory; shortcuts that violate astrophysical realism will not be accepted.  
- **CLI-First Design** — All functions must be callable via the unified `spectramind` CLI.  
- **Auditability** — Every run must log to `v50_debug_log.md` and record config hash + Git SHA.  
- **Open Collaboration** — Code, documentation, and configs should be clear and accessible to all team members and the broader scientific community.  

---

## 2. Repository Layout

Key directories and files you will interact with:

```

ArielSensorArray/
├── configs/        # Hydra YAML configs (data, model, training, diagnostics, calibration, logging)
├── src/            # Core source code (pipeline, calibration, diagnostics, CLI)
├── data/           # Raw and processed data (DVC-tracked, not committed directly)
├── outputs/        # Model checkpoints, diagnostics, predictions (gitignored, tracked via DVC)
├── logs/           # Debug and event logs
├── docs/           # Documentation and MkDocs site
└── tests/          # Unit and integration tests

````

---

## 3. How to Contribute

### Step 1 — Fork and Branch
1. Fork the repository to your GitHub account.  
2. Clone your fork locally.  
3. Create a new branch:  
   ```bash
   git checkout -b feat/<short-description>
````

### Step 2 — Code Standards

* Use **Python 3.10+** with strict type hints.
* Follow **PEP8 + Black** formatting.
* Run **ruff** and **isort** before committing.
* Include full **docstrings** (NumPy/SciPy style).
* Every new feature must be exposed via the **CLI** (Typer app in `spectramind.py`).
* No hardcoded constants — all parameters belong in Hydra configs.

### Step 3 — Logging and Reproducibility

* Ensure every CLI call appends an entry to `logs/v50_debug_log.md`.
* Always log: Git SHA, config hash, seed, dataset version.
* Use deterministic seeds for toy/CI configs.

### Step 4 — Tests

* Add unit tests in `tests/`.
* Run all tests before submitting:

  ```bash
  pytest -q
  ```
* Ensure new modules are covered by **CLI-integrated tests**.
* CI will reject contributions that break reproducibility or fail tests.

### Step 5 — Documentation

* Update relevant docs in `docs/` and `ARCHITECTURE.md`.
* If you add a new CLI command, document it in `docs/cli.md`.
* Update `CHANGELOG.md` if functionality changes.

### Step 6 — Commit and Push

* Use conventional commit messages (examples below).

  * `feat(model): add symbolic influence map module`
  * `fix(cli): correct logging of config hash`
  * `docs: update architecture with new calibration step`
* Push your branch and open a **Pull Request (PR)** to `main`.

---

## 4. Review Process

* PRs require **at least one approving review** from a maintainer.
* All PRs must pass:

  * `spectramind selftest`
  * CI workflows (`.github/workflows/ci.yml`)
  * Pre-commit hooks (ruff, black, isort, YAML checks)
* Maintainers may request:

  * Benchmark evidence (runtime ≤ 9 hrs for full dataset).
  * Reproducibility demo (re-run using config hash + Git SHA).
  * Additional tests or documentation.

---

## 5. Contribution Etiquette

* Be respectful and collaborative — we are building a **mission-grade scientific system**.
* Write clear, explicit comments and docstrings; **no “magic” code**.
* When in doubt, open an **Issue** before starting large changes.
* For experimental work, prefix branch names with `exp/` and PR titles with `[Draft]`.

---

## 6. Adding New Features

Before adding a feature, ask:

1. Does it align with **physics-informed, symbolic, reproducible AI**?
2. Can it be expressed as a Hydra config or CLI option?
3. Is it testable, loggable, and diagnosable?

If **yes**, proceed; if not, propose first in an Issue.

---

## 7. License and Attribution

* All contributions are licensed under the **MIT License** (same as project).
* Cite the project via `CITATION.cff` if you use it in research.

---

## 8. Quick Checklist Before PR

* [ ] Code formatted (black, ruff, isort).
* [ ] All configs updated and tested.
* [ ] `spectramind selftest` passes.
* [ ] Tests added and passing (`pytest`).
* [ ] Documentation updated.
* [ ] Commit message follows convention.

---

**Maintainers:** SpectraMind Core Team
**Contact:** Use GitHub Issues or Discussions for support.

```
