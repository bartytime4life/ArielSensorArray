
````markdown
# Contributing to SpectraMind V50 — ArielSensorArray

Welcome! 🎉  
Thank you for considering contributing to **SpectraMind V50**, our neuro-symbolic, physics-informed AI pipeline for the [NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025).

This document provides guidelines for contributors, covering setup, coding standards, commit practices, and competition-specific rules.

---

## 🚀 Getting Started

### Environment Setup
We support **two parallel stacks**:

1. **Poetry (recommended for development)**  
   ```bash
   poetry install
   poetry shell
````

2. **Conda/Mamba (for CI, Kaggle, HPC)**

   ```bash
   mamba env create -f CONDA_ENV.yml
   conda activate spectramindv50
   ```

🔑 **Notes:**

* For **GPU installs**, pin PyTorch to your CUDA build:

  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
* For **CPU-only installs**, swap `pytorch-cuda` with `cpuonly` in `CONDA_ENV.yml`.

### Dependencies

* Core stack: **NumPy, SciPy, Pandas, PyArrow**
* DL stack: **PyTorch 2.2+, Torch Geometric (wheels), Mamba-SSM**
* Config/CLI: **Hydra, Typer, Rich**
* Diagnostics: **SHAP, UMAP, Plotly, Matplotlib**
* Versioning: **DVC** (with S3/GDrive/GS backends)
* Tracking: **MLflow, W\&B (optional)**

See [`requirements.txt`](./requirements.txt) and [`requirements-dev.txt`](./requirements-dev.txt) for details.

---

## 📦 Project Layout

```bash
SpectraMindV50/
├── spectramind/          # Core pipeline (FGS1 Mamba, AIRS GNN, decoders, calibration, diagnostics)
├── configs/              # Hydra YAML configs
├── scripts/              # CLI utilities, DVC hooks
├── notebooks/            # Prototypes and experiments
├── tests/                # pytest suite (unit, integration, symbolic checks)
├── assets/               # Diagrams, dashboards, generated HTML reports
├── .github/workflows/    # CI/CD pipelines
```

---

## 🧪 Development Workflow

1. **Fork & Branch**

   * Fork the repo, create feature branches from `main`:

     ```bash
     git checkout -b feat/my-feature
     ```

2. **Pre-commit Hooks**

   * Install hooks:

     ```bash
     pre-commit install -t pre-commit -t pre-push
     ```
   * Hooks enforce: `ruff`, `black`, `isort`, `mypy`.

3. **Testing**

   * Run full test suite:

     ```bash
     pytest -q --cov=spectramind
     ```

4. **Documentation**

   * MkDocs drives docs. To preview:

     ```bash
     mkdocs serve
     ```

5. **CLI Self-test**

   * Before pushing:

     ```bash
     spectramind test --deep
     ```

---

## 🎯 Contribution Guidelines

* **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/) with enforced linters.
* **Commits:** Use [Conventional Commits](https://www.conventionalcommits.org/):

  * `feat: add symbolic loss decomposition`
  * `fix: correct calibration σ scaling`
* **PRs:** One logical feature/fix per PR. Include:

  * ✅ Tests
  * 📖 Docs (if user-facing)
  * 📝 Changelog entry

---

## 🧬 Kaggle Integration

* **Competition Code:** Ensure Kaggle notebooks use `requirements.txt` or `CONDA_ENV.yml` to replicate the environment.
* **Resource Limits:** Kaggle provides \~12h GPU runtime (Tesla P100, 16GB RAM). Optimize configs accordingly.
* **Reproducibility:** Always pin `torch`/`torch-geometric` to compatible wheels to avoid runtime mismatch.
* **Leaderboard Etiquette:** Respect the [Kaggle Code of Conduct](https://www.kaggle.com/code-of-conduct). Share public starter notebooks, but avoid leaking private test insights.

---

## 🔍 CI/CD

Our GitHub Actions workflows check:

* **Linting:** `lint.yml`
* **Diagnostics:** `diagnostics.yml` (runs UMAP/t-SNE, SHAP, symbolic overlays)
* **Docs:** `docs.yml`
* **Benchmarking:** `benchmark.yml`

CI **must pass** before merging into `main`.

---

## 🛠️ Advanced Tips

* **Torch Geometric:** Install with CUDA-matched wheels:

  ```bash
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  ```
* **Mamba-SSM:** If your environment uses a fork, adjust `requirements.txt`.
* **DVC Remotes:** Configure only the remotes you use:

  ```bash
  dvc remote add -d s3remote s3://bucket/path
  ```

---

## 🤝 Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
Please treat all contributors and community members with respect.

---

## 📜 License

SpectraMind V50 is released under the **Apache 2.0 License**.
By contributing, you agree your code will be licensed under the same.

---

## 🙌 Acknowledgments

* ESA Ariel Mission science team
* NeurIPS 2025 Ariel Challenge organizers
* Open-source contributors in PyTorch, Hydra, DVC, SHAP, UMAP, and beyond
* Kaggle community notebooks & discussions

---

## 🔗 References

* [NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025)
* [Torch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
* [Hydra Documentation](https://hydra.cc/)
* [DVC Documentation](https://dvc.org/)

```
