# 🧰 `/configs/env/` — Environment & Runtime Profiles (SpectraMind V50)

> Mission: **one place** to define how the pipeline runs on **local dev**, **CI**, **Kaggle**, **Docker**, and **HPC** — hermetically and reproducibly.  
> Everything here is **Hydra-composable**, **.env-driven**, and **DVC/Kaggle-safe**.

---

## 0) What lives here?

- **Profile YAMLs** that toggle runtime behavior:
  - `local.yaml` – developer laptop/desktop
  - `ci.yaml` – GitHub Actions (CPU), smoke-fast
  - `kaggle.yaml` – Kaggle kernels, **no internet**, ≤9 hr
  - `docker.yaml` – containerized reproducible runs
  - `hpc.yaml` – SLURM/LSF clusters, multi-GPU knobs
- **Env scaffolding**:
  - `.env.example` – copy → `.env` for local dev
  - `.env.schema.json` – optional JSON Schema for validation
- **Docs & helpers**:
  - This `readme.md`, optional `hooks.md` (pre-commit), `secrets.md` (how to store secrets safely)

> Tip: keep **all** paths/flags in YAML or `.env` — **never** bake them into code.

---

## 1) Design Principles

- **Hydra-first**: all environment toggles are YAML or CLI overrides.
- **Hermetic**: `.env` + profiles fully define runtime — no hidden state.
- **Reproducible**: fixed random seeds, deterministic PyTorch flags, version-pinned images.
- **Portable**: the same pipeline behaves identically across local, CI, Kaggle, Docker, HPC (within hardware limits).
- **Secure-by-default**: no secrets in Git; use env/keychain or CI secrets; Kaggle has **no internet**.

---

## 2) Quickstart

### A) Local dev (single GPU or CPU)
```bash
cp .env.example .env
# Adjust CUDA and cache paths if needed
spectramind --config-name train.yaml env=local data=nominal

B) CI smoke (CPU, seconds)

spectramind --config-name train.yaml env=ci data=debug training.epochs=1

C) Kaggle (offline)

# Inside a Kaggle Notebook or Script cell
# Note: Kaggle has no interactive shell exports; rely on env/kaggle.yaml
spectramind --config-name train.yaml env=kaggle data=kaggle

D) Docker (exact reproducibility)

# Build image (optional: pin PYTORCH/CUDA via args)
docker build -t spectramind:v50 -f docker/Dockerfile .
# Run container with mounted project & cache
docker run --rm -it \
  -v $PWD:/workspace -v $PWD/.cache:/workspace/.cache \
  --gpus all spectramind:v50 \
  spectramind --config-name train.yaml env=docker data=nominal

E) HPC (SLURM)

# Submit single-GPU job
sbatch scripts/slurm/train_nominal.slurm
# Or interactive test
srun --gres=gpu:1 -c 8 --mem=32G \
  spectramind --config-name train.yaml env=hpc data=nominal


⸻

3) .env variables

Create a local .env (never commit real secrets). These feed Hydra via env= profiles.

# ---- CORE PATHS ----
PROJECT_ROOT=/abs/path/to/repo
DATA_ROOT=${PROJECT_ROOT}/data
CACHE_ROOT=${PROJECT_ROOT}/.cache
ARTIFACTS_ROOT=${PROJECT_ROOT}/outputs

# ---- COMPUTE ----
DEVICE=auto            # auto|cpu|cuda
NUM_WORKERS=4
SEED=1337

# ---- CUDA / DETERMINISM ----
CUDA_VISIBLE_DEVICES=0
TORCH_DETERMINISTIC=1 # 1=deterministic, may reduce perf
CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- DVC / STORAGE ----
DVC_REMOTE=storage
DVC_CACHE_DIR=${PROJECT_ROOT}/.dvc/cache

# ---- LOGGING ----
WANDB_MODE=disabled        # or online|offline (if used)
MLFLOW_TRACKING_URI=       # empty = local file store

# ---- KAGGLE (used only in kaggle env profile) ----
KAGGLE_INPUT=/kaggle/input
KAGGLE_WORKING=/kaggle/working
KAGGLE_TEMP=/kaggle/temp

Validate .env with .env.schema.json (optional) via a simple script or CI step.

⸻

4) Profile matrix

Profile	Device	IO Roots	Internet	Time budget	Notes
local	auto/CPU/GPU	${DATA_ROOT}, ${ARTIFACTS_ROOT}	allowed	user-defined	full diagnostics ok
ci	CPU	repo-local, small	blocked	< 60 s	data=debug, minimal workers
kaggle	GPU (T4/L4)	/kaggle/input, /kaggle/working	blocked	≤ 9 hr	no pip installs from net; memmap ok
docker	GPU/CPU	mounted volumes	allowed (optional)	user-defined	pin base image & torch versions
hpc	multi/single	shared FS / scratch	allowed (policy)	queued	SLURM scripts, nodelist constraints


⸻

5) Hydra environment keys (typical)

Each profile sets (example keys):

# configs/env/local.yaml
env:
  device: ${oc.env:DEVICE,auto}
  num_workers: ${oc.env:NUM_WORKERS,4}
  cache_dir: ${oc.env:CACHE_ROOT,.cache}
  artifacts_dir: ${oc.env:ARTIFACTS_ROOT,outputs}
  reproducibility:
    torch_deterministic: ${oc.env:TORCH_DETERMINISTIC,1}
    seed: ${oc.env:SEED,1337}
  dvc:
    remote: ${oc.env:DVC_REMOTE,storage}
    cache_dir: ${oc.env:DVC_CACHE_DIR,.dvc/cache}

Other profiles override subsets (e.g., Kaggle fixes IO roots and num_workers≤2, enforces no_internet=true).

⸻

6) Determinism & Seeds
	•	Torch: set torch.use_deterministic_algorithms(True) when TORCH_DETERMINISTIC=1.
	•	CUDA: export CUBLAS_WORKSPACE_CONFIG=:4096:8; disable nondeterministic ops if needed.
	•	NumPy/Random: seed once at entry; avoid per-worker time-based seeds.
	•	Dataloaders: worker_init_fn uses global seed; shuffle is deterministic when seed set.

Be aware: strict determinism may reduce performance.

⸻

7) Secrets & credentials
	•	Do not commit secrets to Git or notebooks.
	•	Prefer CI secrets, local keychains, or .env ignored by Git.
	•	Kaggle: secrets are generally not available; design code to run without them.

⸻

8) Recipes

A. Force CPU on local

DEVICE=cpu spectramind --config-name train.yaml env=local data=nominal

B. Fast smoke with preview artifacts

spectramind --config-name train.yaml env=ci data=debug training.epochs=1 diagnostics.save_plots=true

C. Kaggle conservative loader

spectramind --config-name train.yaml env=kaggle data=kaggle loader.batch_size=48 runtime.reduce_heavy_ops=true

D. Reproducible Docker run with pinned image

docker run --rm -it --gpus all -v $PWD:/w -v $PWD/.cache:/w/.cache \
  spectramind:v50 spectramind --config-name train.yaml env=docker data=nominal

E. Multi-GPU HPC (DDP)

srun --gres=gpu:4 -c 16 --mem=64G \
  spectramind --config-name train.yaml env=hpc training.accelerator=ddp training.devices=4


⸻

9) CI & Pre-commit
	•	CI runs env=ci data=debug with training.epochs=1, num_workers=0, minimal diagnostics.
	•	Optional pre-commit hooks:
	•	YAML lint, schema check vs .env.schema.json
	•	Black/ruff for Python
	•	DVC status guard (refuse untracked artifacts)

⸻

10) Troubleshooting
	•	CUDA OOM: lower loader.batch_size, set precision=bf16/fp16 if supported, disable heavy diagnostics.
	•	Stalls on Kaggle: set num_workers=2, avoid multiprocess previews, keep FFT off; ensure no net calls.
	•	Non-deterministic diffs: verify determinism flags and avoid random ops in augmentations for CI/Kaggle.
	•	DVC not found: ensure DVC_REMOTE exists; for Kaggle, rely on /kaggle/input datasets (no remote pulls).

⸻

11) Change Log (excerpt)
	•	2025-09-05: Added toy synthetic support across env profiles; unified .npz/.npy/.pkl ingestion.
	•	2025-09-04: Hardened Kaggle profile (no net, memmap, legacy-shape acceptance).
	•	2025-09-03: CI profile reduced to sub-60s with deterministic loader and preview caps.

⸻

12) Appendix

.env.example

PROJECT_ROOT=/abs/path/to/repo
DATA_ROOT=${PROJECT_ROOT}/data
CACHE_ROOT=${PROJECT_ROOT}/.cache
ARTIFACTS_ROOT=${PROJECT_ROOT}/outputs
DEVICE=auto
NUM_WORKERS=4
SEED=1337
CUDA_VISIBLE_DEVICES=0
TORCH_DETERMINISTIC=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
DVC_REMOTE=storage
DVC_CACHE_DIR=${PROJECT_ROOT}/.dvc/cache
WANDB_MODE=disabled
MLFLOW_TRACKING_URI=
KAGGLE_INPUT=/kaggle/input
KAGGLE_WORKING=/kaggle/working
KAGGLE_TEMP=/kaggle/temp

Hydra selection patterns

# Select env + dataset in one go
spectramind --config-name train.yaml env=kaggle data=kaggle
spectramind --config-name train.yaml env=ci data=debug
spectramind --config-name train.yaml env=local data=nominal


⸻

TL;DR: /configs/env gives you repeatable runs everywhere: pick env=<local|ci|kaggle|docker|hpc> and go. Keep configs + .env as the single source of truth — your future self will thank you.

