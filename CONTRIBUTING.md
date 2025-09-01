# 🤝 Contributing to SpectraMind V50 — ArielSensorArray

Welcome! 🎉  
Thank you for considering contributing to **SpectraMind V50**, the neuro-symbolic, physics-informed AI pipeline for the [NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025).

This guide defines how to set up your environment, follow coding standards, commit properly, and integrate with the competition/Kaggle ecosystem — while maintaining **NASA-grade reproducibility**.

---

## 🚀 Getting Started

### Environment Setup

We support **two parallel stacks**:

1. **Poetry (recommended for local development)**  
   ```bash
   poetry install
   poetry shell
   ```

2. **Conda/Mamba (for Kaggle, CI, HPC)**  
   ```bash
   mamba env create -f CONDA_ENV.yml
   conda activate spectramindv50
   ```

🔑 **Notes:**
* For **GPU installs**, pin PyTorch to your CUDA version:
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```
* For **CPU-only**, replace `pytorch-cuda` with `cpuonly` in `CONDA_ENV.yml`.
* Kaggle environments reset — always include package installs in your notebook or pin dependencies.

---

## 📦 Project Layout

```bash
SpectraMindV50/
├── src/spectramind/     # Core pipeline (FGS1 Mamba, AIRS GNN, decoders, symbolic/diagnostics)
├── configs/             # Hydra configs (data, model, training, uncertainty, GUI, etc.)
├── scripts/             # CLI utilities, DVC hooks
├── tests/               # pytest suite (unit, integration, symbolic, CLI)
├── assets/              # Diagrams, dashboards, generated HTML reports
├── .github/workflows/   # CI/CD (lint, test, diagnostics, diagrams, docs)
├── Dockerfile           # Reproducible base image
├── dvc.yaml             # DVC pipeline (calibration → train → diagnostics → submit)
```

See [`README.md`](./README.md) for a quickstart guide.

---

## 🧪 Development Workflow

1. **Fork & Branch**
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Pre-commit Hooks**
   ```bash
   pre-commit install -t pre-commit -t pre-push
   ```
   Enforces: `ruff`, `black`, `isort`, `mypy`.

3. **Testing**
   ```bash
   pytest -q --cov=src/spectramind
   spectramind test --deep   # full CLI self-test with config+artifact checks
   ```

4. **Documentation**
   ```bash
   mkdocs serve
   ```

5. **Reproducibility Check**
   ```bash
   dvc repro       # re-run full pipeline
   spectramind submit --selftest   # dry-run submission check
   ```

---

## 🎯 Contribution Guidelines

* **Style:** PEP 8 + enforced linters; type hints required.
* **Commits:** Use [Conventional Commits](https://www.conventionalcommits.org/):
  * `feat: add symbolic violation predictor`
  * `fix: correct calibration σ scaling`
* **Pull Requests:** One logical change per PR, with:
  * ✅ Tests
  * 📖 Docs (if user-facing)
  * 📝 `CHANGELOG.md` entry

---

## 🧬 Kaggle Integration

* **Competition Code:** Use `requirements.txt` or `CONDA_ENV.yml` for reproducibility.
* **Limits:** Kaggle GPU quota ≈ 12h (Tesla P100, 16 GB). Optimize configs accordingly.
* **Reproducibility:** Always pin `torch`/`torch-geometric` wheels to prevent mismatch.
* **Best Practices:**
  * Keep training < 9 hrs on Kaggle.
  * Use Kaggle’s **dataset versioning** for data+checkpoints.
  * Respect the [Kaggle Code of Conduct](https://www.kaggle.com/code-of-conduct).

---

## 🔍 CI/CD

GitHub Actions workflows run on every PR:

* **`lint.yml`** — ruff, black, isort, mypy  
* **`test.yml`** — unit + CLI + symbolic tests  
* **`diagnostics.yml`** — UMAP/t-SNE, SHAP, symbolic overlays, FFT checks  
* **`docs.yml`** — build + link-check MkDocs  
* **`benchmark.yml`** — optional regression benchmarks  

✅ All must pass before merging.

---

## 🛠️ Advanced Tips

* **Torch Geometric Wheels (CUDA):**
  ```bash
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  ```
* **Mamba-SSM:** Ensure correct fork/version in `requirements.txt`.
* **DVC Remotes:**
  ```bash
  dvc remote add -d s3remote s3://bucket/path
  ```

---

## 🤝 Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/).  
Please treat all contributors and community members with respect.

---

## 📜 License

SpectraMind V50 is released under the **MIT License** (with citation requirement).  
By contributing, you agree your code will be licensed under the same.

---

## 🙌 Acknowledgments

* ESA Ariel Mission science team  
* NeurIPS 2025 Ariel Challenge organizers  
* Kaggle community notebooks & discussions  
* Open-source contributors in PyTorch, Hydra, DVC, SHAP, UMAP, Hugging Face, and beyond  

---

## 🔗 References

* [NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025)  
* [SpectraMind V50 Technical Plan](/docs/SpectraMindV50_TechnicalPlan.pdf):contentReference[oaicite:3]{index=3}  
* [SpectraMind V50 Project Analysis](/docs/SpectraMindV50_ProjectAnalysis.pdf):contentReference[oaicite:4]{index=4}  
* [Strategy for Updating & Extending V50](/docs/SpectraMindV50_Strategy.pdf):contentReference[oaicite:5]{index=5}  
* [Kaggle Platform Guide](/docs/Kaggle_Platform_Guide.pdf):contentReference[oaicite:6]{index=6}  
