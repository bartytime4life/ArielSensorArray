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

	•	Separation of Concerns: Code (src/), data (data/ via DVC/lakeFS), and environment (configs/env/) are strictly separated.
	•	Audit & Traceability: Every run logs its environment hash (run_hash_summary_v50.json) and writes metadata to v50_debug_log.md.

⸻

2) Directory Layout

configs/
└── env/
    ├── default.yaml      # Baseline dev environment (Poetry, Python, CUDA)
    ├── kaggle.yaml       # Kaggle GPU runtime (no internet, 9h limit)
    ├── ci.yaml           # GitHub Actions / CI runners
    ├── docker.yaml       # Docker container environment
    ├── readme.md         # Environment README (overview + usage)
    └── architecture.md   # (this file)


⸻

3) Environment Profiles

🔹 default.yaml
	•	Use case: Local development.
	•	Python: 3.10.x
	•	Package manager: Poetry (strict lockfile, hashes pinned).
	•	CUDA/cuDNN: matched to Kaggle runtime for parity.
	•	MLflow + W&B optional for experiment tracking.

🔹 kaggle.yaml
	•	Use case: Kaggle kernels (A100/T4 GPUs, no internet).
	•	Constraints:
	•	Runtime ≤ 9 hours.
	•	No pip install at runtime except pre-approved packages.
	•	Datasets mounted under /kaggle/input/….
	•	Mirrors default.yaml with stripped extras (no internet calls, minimal logging).
	•	Validated via bin/kaggle-boot.sh bootstrap script.

🔹 ci.yaml
	•	Use case: GitHub Actions.
	•	Deterministic Python environment (setup-python@v5).
	•	Jobs: lint, unit tests, diagnostics, submission validation.
	•	SBOM (CycloneDX/SPDX) + security scans (Trivy, pip-audit).
	•	Mirrors Kaggle runtime where possible.

🔹 docker.yaml
	•	Use case: Reproducible containerized builds.
	•	Base image: python:3.10-slim + pinned CUDA toolkit.
	•	Poetry + Hydra + DVC baked into container.
	•	CI builds and pushes to GitHub Container Registry.
	•	Used for local parity and archival.

⸻

4) Configuration Flow

flowchart TD
    A[default.yaml] -->|override| B[kaggle.yaml]
    A -->|override| C[ci.yaml]
    A -->|override| D[docker.yaml]
    B -->|runs on| K[Kaggle GPU (no internet)]
    C -->|runs on| G[GitHub Actions CI]
    D -->|runs on| L[Local Docker / HPC]
    subgraph Hydra
        A;B;C;D
    end

	•	Hydra composes environment configs with model/data/training configs.
	•	Overrides (+env=kaggle) select environment at runtime.
	•	CLI (spectramind) logs chosen profile + hash for reproducibility.

⸻

5) Reproducibility & Validation
	•	Hydra snapshots: every run saves merged env config to outputs/YYYY-MM-DD/HH-MM-SS/.hydra.
	•	Self-tests: spectramind test validates environment integrity (Python, CUDA, Kaggle mounts, DVC).
	•	Hash logging: run_hash_summary_v50.json maps env profiles to git commit, Hydra config, and MLflow run IDs.
	•	SBOM exports: environment dependencies are captured as CycloneDX SPDX docs in CI.

⸻

6) Integration with Kaggle & MCP
	•	Kaggle notebooks mirror GitHub repo via auto-exported inference/training kernels.
	•	MCP alignment: every environment config includes rationale, reproducibility proof, and audit trails.
	•	GitHub → Kaggle sync handled by CI (.github/workflows/kaggle-sync.yml).
	•	Reproducibility checklist: each run documents Python version, package lock, CUDA/cuDNN, Kaggle kernel version.

⸻

7) Future Extensions
	•	Cross-platform parity testing: nightly workflow runs spectramind test across all profiles.
	•	HPC extension: add hpc.yaml for Slurm/cluster environments.
	•	Symbolic-aware env overrides: toggle libraries for symbolic reasoning (SymPy, Z3).
	•	Long-term archival: freeze Docker images + Poetry lockfiles for submission reproducibility.

