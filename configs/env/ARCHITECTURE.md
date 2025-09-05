# 🛰️ `/configs/env/ARCHITECTURE.md` — Environment Configuration Architecture

---

## 0) Purpose & Scope

`/configs/env` defines the **runtime environment stack** for the SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).  
This ensures that **local, Kaggle, and CI/CD environments** are deterministic, reproducible, and fully aligned with the **Master Coder Protocol (MCP)**:

- **Hydra-first composition** of environment profiles (`default.yaml`, `kaggle.yaml`, `ci.yaml`, `docker.yaml`).
- **Mission-grade reproducibility**: pinned Python version, deterministic CUDA/cuDNN, and controlled random seeds.
- **Multi-platform parity**: the same pipeline runs seamlessly on **local dev, Kaggle GPUs (no internet), Docker containers, and GitHub Actions**.
- **Documentation-first ethos**: every environment config is self-describing and tracked in Git for audit compliance.

---

## 1) Design Philosophy

- **Single Source of Truth (SoT)**: All environment definitions live in YAML configs under `/configs/env`.  
- **Hydra Overrides**: Profiles are composable; e.g.  
  ```bash
  spectramind train +env=kaggle +data=nominal +model=v50