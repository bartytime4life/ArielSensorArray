
⸻

🛰️ SpectraMind V50 — Bug Report

Fill out every section. If something doesn’t apply, write N/A and explain why. This template enforces our NASA‑grade reproducibility, physics‑informed modeling, and CLI‑first standards.

⸻

0) Pre‑submission Checklist (must pass before filing)
	•	I searched existing issues and discussions for this bug.
	•	I reproduced on latest main (paste commit hash below).
	•	I ran spectramind selftest --fast and spectramind selftest --deep and captured logs.
	•	I ran pre-commit run -a (ruff/black/isort/yaml checks) and fixed style issues.
	•	I can reproduce with a clean environment (e.g., fresh Poetry env or container).
	•	I attached key artifacts (logs, HTML diagnostics, config snapshot) listed in §5.

Current commit (git rev-parse HEAD):
<hash>

⸻

1) Summary

Title:
[BUG] <concise problem statement>

Observed Behavior (what happened):
<clear, factual description>

Expected Behavior (what you thought should happen):


Severity: ☐ Blocker (prevents submission) ☐ High ☐ Medium ☐ Low
Scope: ☐ CLI ☐ Hydra Configs ☐ Data/Calibration ☐ Model Training ☐ Inference ☐ Diagnostics/Dashboard ☐ DVC ☐ CI/GitHub Actions ☐ Docker/Container ☐ Packaging/Submission ☐ Other: <specify>

Reproducibility: ☐ Always ☐ Often (≥70%) ☐ Intermittent (30–70%) ☐ Rare (<30%)
First seen on: <date> (America/Chicago)

⸻

2) Environment

Paste exact outputs where indicated. Do not paraphrase.
	•	Platform: ☐ Linux (distro + version) ☐ macOS ☐ Windows ☐ Kaggle ☐ Other
	•	Kernel (Linux): uname -a
	•	Container: ☐ Docker ☐ Bare‑metal
Image/tag: <e.g., spectramindv50:dev>  Dockerfile hash (if custom): <hash>
	•	GPU(s): <e.g., 1×RTX 5080 16GB>  Driver: nvidia-smi top lines
	•	CUDA/cuDNN: python -c "import torch;print('torch',torch.__version__);import torch,platform;print('cuda?',torch.cuda.is_available(),'cuda_ver',torch.version.cuda)" and python -c "import torch.backends.cudnn as c;c.enabled and print('cudnn',c.version())"
	•	Python/Poetry: python --version, poetry --version
	•	Core libs: hydra-core, dvc, mlflow (versions) — pip show hydra-core dvc mlflow | grep -E 'Name|Version'
	•	Project CLI: spectramind --version (include CLI version, config hash, timestamp printed by the app)
	•	Git: git rev-parse HEAD (again, for clarity)
	•	Data location/mode: ☐ toy ☐ challenge ☐ custom; ☐ local ☐ DVC remote ☐ Kaggle dataset

⸻

3) Minimal Reproduction (exact steps & commands)

Provide the smallest scenario that still reproduces the problem. Prefer the toy split first. Include all flags/overrides, exact working directory, and any environment variables used.

Working directory: <absolute or repo‑relative path>

Commands (copy‑paste exactly as run):

# Example (adjust to your case)
spectramind selftest --deep

# Toy CI/dev loop (fast)
spectramind train --config-name=config_v50 +data.split=toy +training.seed=1337

# Full repro
spectramind calibrate --config-name=config_v50
spectramind train --config-name=config_v50 +training.seed=1337 training.mixed_precision=true
spectramind predict --out-csv outputs/submission.csv
spectramind diagnose dashboard --open-html

Hydra overrides used:
<e.g., +data.split=toy +training.seed=1337 model.decoder.head=gaussian_corel>

DVC context (if relevant):

dvc repro -f
dvc status
dvc doctor

Dataset / Planets involved (IDs or count):
<list or range> (note any that consistently fail)

Seeds:
<list> (confirm if seed changes alter outcome)

⸻

4) Regression Info (if applicable)
	•	Last known good commit: <hash> (commands & metrics attached below)
	•	First bad commit: <hash> (suspected)
	•	Δ Changes (high‑level): <e.g., upgraded GATConv; new calibration stage>
	•	Metrics Δ (baseline → now):

Metric	Baseline (commit)	Current (commit)	Δ
GLL (val/public)			
MAE(μ)			
Calibration z‑mean/z‑std			
Coverage@95%			
Runtime (planet avg)			


⸻

5) Logs & Artifacts (attach or link)
	•	logs/v50_debug_log.md (relevant excerpt + full file if small)
	•	logs/v50_event_log.jsonl (or events.jsonl)
	•	outputs/diagnostics/report_vX.html (dashboard)
	•	outputs/diagnostics/diagnostic_summary.json
	•	constraint_violation_log.json (symbolic)
	•	run_hash_summary_v50.json
	•	Any relevant PNG/CSV (e.g., GLL heatmap, calibration plots)

If artifacts are large, use DVC and paste the .dvc meta + remote path.

⸻

6) Config Snapshot (exact composed Hydra config)

Paste the full OmegaConf.to_yaml(cfg) for the main failing run (redact secrets).

# --- BEGIN HYDRA CONFIG SNAPSHOT ---
<insert exact composed config>
# --- END HYDRA CONFIG SNAPSHOT ---


⸻

7) Symbolic / Physics Integrity

Check all that apply and describe:
	•	New or increased symbolic rule violations (which rules?):
	•	☐ Smoothness ☐ Nonnegativity ☐ Asymmetry ☐ FFT prior ☐ Photonic alignment
	•	Molecule region inconsistencies: ☐ H₂O ☐ CH₄ ☐ CO₂ ☐ CO ☐ Other: <...>
	•	Gravitational lensing or calibration anomalies suspected
	•	Physical constraints broken in specific bins / wavelengths (list):
	•	<bin indices / wavelength ranges>

Details / screenshots:
<describe and/or attach figures>

⸻

8) Error Trace / Console Output

<paste the exact error/traceback or relevant console snippet>

If a hang/deadlock, provide:
	•	Location (stage/command)
	•	Last visible log line(s)
	•	nvidia-smi top summary during hang (utilization, memory)
	•	Whether --debug or increased logging reveals progress

⸻

9) Workarounds Tried
	•	Disable mixed precision (training.mixed_precision=false)
	•	Reduce batch size (+data.batch_size=<n>)
	•	Different seed(s)
	•	Rebuild environment / re‑install deps
	•	Pin Torch/CUDA versions
	•	Skip diagnostics (diagnose --no-umap/--no-tsne)
	•	Other: <what else?>

Effect of each attempt:
<brief results; include timings/metrics if relevant>

⸻

10) Security & Data Compliance
	•	No credentials or secrets leaked in logs/configs
	•	No prohibited data uploaded (competition rules compliant)
	•	Licenses/attributions unchanged

⸻

11) Additional Context

Anything else we should know? Links to PRs, discussions, Slack threads, or external references.

⸻

Maintainer Triage (to be filled by core team)
	•	Category: ☐ Data/Calibration ☐ Model/GNN/Decoder ☐ CLI/Config ☐ Diagnostics ☐ Infra/CI ☐ Packaging/Submission ☐ Docs
	•	Owner: @assignee
	•	Priority: P0 / P1 / P2
	•	Root Cause (when known): <summary>
	•	Next Actions: <checklist>
	•	Target Release/Tag: <e.g., v0.1.2>
	•	Labels to apply: bug, needs-repro, physics, symbolic, perf, infra, docs, help-wanted

⸻


<details>
<summary><strong>How to Collect Useful Evidence (quick guide)</strong></summary>


# 1) Version & hash
spectramind --version
git rev-parse HEAD

# 2) Self-tests
spectramind selftest --fast
spectramind selftest --deep

# 3) Minimal toy repro
spectramind train --config-name=config_v50 +data.split=toy +training.seed=1337

# 4) Full pipeline (if needed)
spectramind calibrate --config-name=config_v50
spectramind train --config-name=config_v50 +training.seed=1337
spectramind predict --out-csv outputs/submission.csv
spectramind diagnose dashboard

# 5) Collect logs & artifacts
ls -lah logs/ outputs/diagnostics/

If you have DVC:

dvc repro -f
dvc status
dvc doctor

For CUDA/Torch:

nvidia-smi
python - <<'PY'
import torch, json
print(json.dumps({
  "torch": torch.__version__,
  "cuda_available": torch.cuda.is_available(),
  "cuda_version": getattr(torch.version, "cuda", None),
  "device_count": torch.cuda.device_count()
}, indent=2))
PY

</details>
