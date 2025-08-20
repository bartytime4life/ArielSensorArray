* **Kaggle-specific rules** (daily submission limits, 9-hour GPU budget, leaderboard etiquette).
* **Competitor learnings** (why V50 avoids overly deep MLPs like the 80-block model, why calibration-first matters).
* **Physics-informed rules** beyond general integrity: lensing, radiation, calibration noise modeling.
* **Reproducibility stack** (Hydra configs, DVC dataset hashes, CI toy runs).
* **GUI/Diagnostics roadmap** (FastAPI + React integration, HTML dashboards).

Here’s the **upgraded `contributing.md`** with those integrations folded in:

```markdown
# Contributing to SpectraMind V50 — ArielSensorArray

Welcome, and thank you for considering a contribution to the **SpectraMind V50** project for the NeurIPS 2025 Ariel Data Challenge.  
This guide sets the standards, workflows, and expectations to keep contributions aligned with the project’s goals of **reproducibility, scientific rigor, Kaggle-competition compliance, and engineering excellence**.

---

## 1) Project Principles

- **Reproducibility First** — All code, configs, and data flows must be deterministic, Hydra-configurable, and version-controlled:contentReference[oaicite:6]{index=6}.  
- **Scientific Integrity** — Physics-informed modeling and symbolic rules are mandatory. Constraints include smoothness, non-negativity, FFT suppression, and astrophysical realism (radiative transfer, lensing, radiation noise):contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}.  
- **CLI-First Design** — Every function must be callable via the unified `spectramind` CLI:contentReference[oaicite:9]{index=9}.  
- **Auditability** — Every run must log to `logs/v50_debug_log.md` with config hash, Git SHA, and DVC dataset hash:contentReference[oaicite:10]{index=10}.  
- **Kaggle-Conscious Development** — Pipeline must remain within Kaggle’s constraints: ≤9 hrs runtime on GPU, ≤ daily submission limits, no reliance on external data:contentReference[oaicite:11]{index=11}.  
- **Open Collaboration** — Code, docs, and configs must be clear, tested, and accessible to teammates and the scientific community.  

---

## 2) Repository Layout

```

ArielSensorArray/
├── configs/        # Hydra YAML configs (data, model, training, diagnostics, calibration, logging)
├── src/            # Core source code (pipeline, calibration, diagnostics, CLI)
├── data/           # Raw and processed data (DVC-tracked, not committed directly)
├── outputs/        # Checkpoints, diagnostics, predictions (gitignored/DVC)
├── logs/           # Debug and event logs (v50\_debug\_log.md, event JSONL)
├── docs/           # Documentation and MkDocs site
└── tests/          # Unit and integration tests

````

---

## 3) How to Contribute

### Step 1 — Fork and Branch
```bash
git checkout -b feat/<short-description>
````

### Step 2 — Code Standards

* Python 3.10+ with strict type hints.
* PEP8 + Black formatting.
* `ruff` + `isort` before committing.
* Full NumPy/SciPy-style docstrings.
* All features exposed via CLI (`spectramind`).
* No hardcoded constants — everything must live in Hydra configs.

### Step 3 — Logging & Reproducibility

* Every CLI call must append to `logs/v50_debug_log.md`.
* Always log: Git SHA, config hash, seed, dataset version.
* Use deterministic seeds for toy/CI configs.
* All large artifacts tracked via DVC (never commit blobs directly).

### Step 4 — Tests

* Add unit tests in `tests/`.
* Run:

  ```bash
  pytest -q
  ```
* Cover new modules with CLI-integrated toy-mode tests:

  ```bash
  python -m spectramind train +data.split=toy
  python -m spectramind diagnose dashboard --html out/test_report.html
  ```

### Step 5 — Documentation

* Update `docs/` and `ARCHITECTURE.md`.
* New CLI commands must be documented in `docs/cli.md`.
* Update `CHANGELOG.md` if functionality changes.

### Step 6 — Commit and Push

* Follow Conventional Commits:

  * `feat(model): add symbolic influence map`
  * `fix(cli): correct config hash logging`
  * `docs: update calibration step in architecture`
* Push and open a Pull Request (PR) against `main`.

---

## 4) Review Process

All PRs must pass:

* `spectramind selftest`
* GitHub Actions CI (`.github/workflows/ci.yml`)
* Pre-commit hooks (ruff, black, isort, mypy, yaml)

Maintainers may request:

* Benchmark evidence (≤9 hrs runtime on Kaggle full dataset)
* Reproducibility demo (re-run via config hash + Git SHA)
* Extra tests or documentation

---

## 5) Contribution Etiquette

* Be respectful and collaborative — we’re building a mission-grade scientific system.
* Write explicit comments/docstrings; no “magic” code.
* Open an Issue before large changes.
* Use `exp/` branches and \[Draft] PRs for experiments.
* Respect Kaggle’s **submission quotas** and avoid leaderboard overfitting (“shake-up” risk).

---

## 6) Adding New Features

Ask first:

1. Is it physics-informed, symbolic, and reproducible?
2. Can it be expressed as a Hydra config or CLI flag?
3. Is it testable, loggable, diagnosable?
4. Does it respect Kaggle runtime/leaderboard constraints?

If yes, proceed. If not, propose in an Issue first.

---

## 7) License & Attribution

* Contributions are licensed under Apache 2.0 License.
* Cite via `CITATION.cff` if used in research.

---

## 8) Pre-PR Checklist

* Code formatted (black, ruff, isort).
* Configs updated & tested.
* `spectramind selftest` passes.
* Tests added & passing (pytest).
* Documentation updated.
* Commit message follows convention.

---

## 9) Inspiration from Kaggle Competitors

* Avoid **overly deep MLPs** like “80bl-128hd” — runtime inefficient under Kaggle limits.
* Use **calibration-first preprocessing** — baseline models lacked this and suffered.
* Always prefer **uncertainty-calibrated predictions** over point estimates.
* Build **explainability hooks** into every feature — leaderboard score isn’t enough.

---

**Maintainers:** SpectraMind Core Team
**Contact:** Use GitHub Issues or Discussions

---

```

---

✅ This new `contributing.md` now folds in:  
- **Kaggle rules and etiquette**:contentReference[oaicite:19]{index=19}  
- **Competitor insights**:contentReference[oaicite:20]{index=20}  
- **Physics-informed priorities**:contentReference[oaicite:21]{index=21}:contentReference[oaicite:22]{index=22}  
- **Reproducibility and CI stack**:contentReference[oaicite:23]{index=23}  
- **GUI/diagnostics roadmap alignment**:contentReference[oaicite:24]{index=24}  
