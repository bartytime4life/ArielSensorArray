# 🤝 Contributing to SpectraMind V50

Welcome to **SpectraMind V50** — the neuro-symbolic, physics-informed AI pipeline for the
[NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/neurips-2025-ariel).
We are thrilled you are here!

This guide sets expectations for contributors and ensures that every change remains
**reproducible, scientifically rigorous, and production-ready**.

---

## 📜 Philosophy

- **CLI-first** — All functionality is available via the `spectramind` Typer CLI.
- **Hydra-safe configs** — No hard-coded params; all runs are config-driven.
- **Reproducibility** — Seeds, hashes, and manifests logged on every run.
- **Scientific integrity** — Physics-informed, symbolic constraints enforced.
- **CI-ready** — Lint, test, and Docker smoke tests must pass before merge.

---

## 🚀 Quickstart (Contributor Mode)

```bash
# 1. Fork & clone
git clone https://github.com/YOURNAME/spectramind-v50
cd spectramind-v50

# 2. Install with poetry
poetry install

# 3. Enable pre-commit hooks
pre-commit install

# 4. Verify environment + CLI
make quickstart   # runs selftest, hash check, dummy pipeline
```

---

## 🧪 Development Workflow

1. **Branch naming**
   - `feat/<topic>` — new features
   - `fix/<topic>` — bug fixes
   - `docs/<topic>` — documentation changes
   - `chore/<topic>` — maintenance / CI / deps

2. **Pre-commit checks**
   - Auto-formatting (`black`, `isort`)
   - Lint (`ruff`, `yamllint`, `markdownlint`, `shellcheck`)
   - Security (`bandit`)
   - Run locally:
     ```bash
     pre-commit run --all-files
     ```

3. **Run tests**
   - Unit + integration tests live in `/tests/`
   - Execute with:
     ```bash
     pytest -q
     ```
   - Use `--runslow` to include heavier scientific tests.

4. **Selftest pipeline**
   - Always verify before PR:
     ```bash
     spectramind test --deep
     ```

5. **Docs**
   - Update relevant `README.md`, `ARCHITECTURE.md`, and manifests.
   - Add docstrings + type hints to new code.

---

## 🛡️ CI & Security

- GitHub Actions enforce:
  - `lint.yml` → pre-commit only (blocking)
  - `tests.yml` → pytest + mypy (matrix: 3.10/3.11/3.12)
  - `docker-smoke.yml` → build image & run `spectramind selftest`
  - `codeql-analysis.yml` → CodeQL scans
- Dependabot auto-bumps pip / actions / Docker.
- All PRs must be green before merge.

---

## 📂 Data & Reproducibility

- **Never commit data** — all experiment data is tracked via DVC/lakeFS.
- **Outputs ignored** — ensure `outputs/` and `logs/` never enter Git.
- **Seeds & hashes** — every CLI run logs its config hash + Git commit.
- **Artifacts** — submissions, dashboards, diagnostics are exported under `/outputs`.

---

## 🧭 Scientific Integrity

- Symbolic constraints (`src/symbolic/`) are **mandatory** in training runs.
- Diagnostics (`spectramind diagnose`) must show no hard violations
  (negative flux, unstable FFT, broken logic).
- All contributions should **respect physics realism** and maintain leaderboard compliance.

---

## ✅ Pull Request Checklist

- [ ] Branch named correctly (`feat/…`, `fix/…`, etc.)
- [ ] Pre-commit hooks passed (`pre-commit run --all-files`)
- [ ] Unit & integration tests passed (`pytest`)
- [ ] `spectramind selftest` passes (fast + deep modes)
- [ ] Docs updated (README/ARCHITECTURE/manifests)
- [ ] No data committed (`dvc status` clean)

---

## 📬 Getting Help

- **Issues** — File bug reports or feature requests via GitHub Issues.
- **Discussions** — Use GitHub Discussions for design questions & community chat.
- **Security** — Report vulnerabilities privately to maintainers.

---

### 🛰️ Thank you!

Your contributions help us build a **NASA-grade, challenge-winning pipeline**.  
Together, we’ll make **SpectraMind V50** a reference standard in neuro-symbolic AI.