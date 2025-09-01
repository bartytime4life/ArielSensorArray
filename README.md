# 🌌 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

**Mission:** Build a **neuro-symbolic, physics-informed AI pipeline** to extract exoplanetary spectra (μ, σ for 283 bins) from ESA Ariel telescope simulation data.  
Designed for **NASA-grade reproducibility**, **CLI-first workflows**, and **Kaggle runtime compliance** (≤9h on ~1,100 planets).

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/<YOUR_ORG>/SpectraMindV50.git
cd SpectraMindV50

# Create environment
poetry install   # or: pip install -r requirements.txt

# Fetch data/models (DVC)
dvc pull

# Run pipeline self-test
spectramind test --deep

# Train and predict
spectramind train +experiment=nominal
spectramind submit --config configs/train/nominal.yaml
````

---

## 🧭 Repository Structure

```
SpectraMindV50/
├── configs/          # Hydra configs (data, model, training, diagnostics)
├── src/              # Core pipeline code (encoders, decoders, diagnostics)
├── docs/             # Documentation, diagrams, guides
├── data/             # Raw + processed (DVC-tracked, not stored in Git)
├── tests/            # Pytest suite for all modules
├── artifacts/        # Outputs: logs, HTML dashboards, submission bundles
├── pyproject.toml    # Poetry environment
├── Dockerfile        # Reproducible build (Ubuntu CUDA base)
└── README.md         # You are here
```

---

## 🗺️ End-to-End Workflow (CLI → Hydra → Pipeline → Kaggle)

```mermaid
flowchart TD
  A[User] -->|invokes| B[spectramind CLI]
  B -->|compose + override| C[Hydra Configs\n(configs/*.yaml)]
  C --> D[Pipeline Orchestrator]
  D --> E[Calibration\n(FGS/AIRS processing)]
  E --> F[Model Training\n(SSM + GNN → μ, σ)]
  F --> G[Diagnostics & Explainability\n(GLL, FFT, SHAP, Symbolic)]
  G --> H[Submission Bundler\n(selftest, manifest, zip)]
  H --> I[Kaggle Leaderboard]

  %% Artifact side-rails
  C -. logs, overrides .-> J[(Artifacts\noutputs/YYYY-MM-DD/HH-MM-SS)]
  E -. DVC tracked data .-> J
  F -. checkpoints, metrics .-> J
  G -. HTML dashboard .-> J
  H -. submission.zip .-> J
```

---

## 📦 Features (Implemented vs Planned)

* ✅ **CLI-first orchestration** via `spectramind` Typer app
* ✅ **Hydra configs** for full experiment control
* ✅ **DVC** for data/model versioning
* ✅ **NASA-grade reproducibility**: config hashes, JSONL logs, CI checks
* ✅ **Scientific constraints**: smoothness, non-negativity, symbolic rule checks
* ✅ **Diagnostics**: FFT, SHAP, UMAP/t-SNE, calibration plots
* 🚧 **Experiment tracking UI** — *MLflow planned*
* 🚧 **GUI dashboard** (React + FastAPI, CLI artifact mirror)
* 🚧 **Advanced uncertainty calibration** (COREL, conformal prediction)
* 🚧 **Interpretability suite** (attention maps, symbolic overlays)

---

## 📊 Example Outputs

* 🖼️ **UMAP / t-SNE latent projections** (interactive HTML)
* 📈 **Per-bin GLL heatmaps** (diagnostics/plot)
* 🧩 **Symbolic rule violation overlays** (smoothness, physical constraints)
* 📜 **`diagnostic_summary.json`** — JSON export for dashboard integration

*(See `/artifacts/` after running `spectramind diagnose dashboard`)*

---

## 🧪 Development Philosophy

* **CLI-first, GUI-optional** — all operations reproducible via Typer + Hydra
* **Reproducibility First** — configs in Git, data in DVC, hashes logged
* **Scientific Integrity** — physics-informed ML, symbolic constraints
* **Automation & CI** — GitHub Actions enforce reproducibility
* **Mission-grade Logging** — JSONL + Rich console + HTML dashboards

---

## 🧩 Roadmap (2025)

**Phase 1: Audit & Gap Close**

* Expand Hydra config coverage (uncertainty calibration, GUI settings)
* Fully capture pipeline stages in DVC for caching and reuse

**Phase 2: Diagnostics Expansion**

* SHAP overlays + GNN explainers
* Symbolic rule influence mapping
* Extended per-bin diagnostics

**Phase 3: Uncertainty Calibration**

* COREL GNN calibration
* Per-bin conformal prediction
* Coverage plots + symbolic weighting

**Phase 4: GUI Layer**

* React/Vite dashboard on top of FastAPI server
* Purely artifact-driven (no drift from CLI runs)

**Phase 5: Leaderboard Polishing**

* Hyperparameter sweeps (Hydra multirun)
* Ensemble inference
* Ablation engine with symbolic overlays

---

## 📜 License

This project is released under the **MIT License** (OSI approved).

* ✅ Open-source friendly
* ✅ Compatible with Kaggle distribution
* ✅ Aligns with NASA/ESA open data sharing principles

See [`LICENSE`](LICENSE) for details.

---

## 📝 Citations

* [SpectraMind V50 Technical Plan]: contentReference[oaicite:27]{index=27}
* [SpectraMind V50 Project Analysis]: contentReference[oaicite:28]{index=28}
* [Strategy for Updating V50]: contentReference[oaicite:29]{index=29}
* [Hydra for AI Projects Guide]: contentReference[oaicite:30]{index=30}
* [Mermaid Diagrams in GitHub Markdown]: contentReference[oaicite:31]{index=31}
* [Engineering Guide to GUI Development]: contentReference[oaicite:32]{index=32}
* [Computational Physics Modeling]: contentReference[oaicite:33]{index=33}
* [Gravitational Lensing in Observation]: contentReference[oaicite:34]{index=34}
* [Radiation Technical Reference]: contentReference[oaicite:35]{index=35}

---

## 🧑‍💻 Maintainers

**SpectraMind Core Team** — NeurIPS 2025 Ariel Data Challenge
Contact: `<your_email@domain>`

---

<p align="center">
  <a href="https://github.com/<YOUR_ORG>/SpectraMindV50/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/<YOUR_ORG>/SpectraMindV50/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://github.com/<YOUR_ORG>/SpectraMindV50/releases">
    <img alt="Release" src="https://img.shields.io/github/v/release/<YOUR_ORG>/SpectraMindV50?display_name=tag&sort=semver">
  </a>
  <a href="https://hub.docker.com/r/<YOUR_DOCKER_NS>/spectramind-v50">
    <img alt="Docker pulls" src="https://img.shields.io/docker/pulls/<YOUR_DOCKER_NS>/spectramind-v50.svg">
  </a>
  <a href="https://www.kaggle.com/competitions/ariel-data-challenge-2025">
    <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-NeurIPS%202025%20Ariel-20BEFF.svg">
  </a>
</p>
```

---
