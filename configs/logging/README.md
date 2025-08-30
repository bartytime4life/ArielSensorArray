# 📝 `/configs/logging` — Logging & Experiment Tracking

## 0. Purpose & Scope

The **`/configs/logging`** directory defines how the SpectraMind V50 pipeline records, stores, and visualizes all logs, metrics, and experiment traces.  

Logging is treated as a **first-class scientific artifact**: every run must be auditable, reproducible, and linked back to its exact **config hash**, **Git commit**, and **DVC-tracked data snapshot** [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

This directory ensures:
- 📊 **Metrics & Scalars** — Loss curves, calibration scores, violation maps
- 📑 **Structured Logs** — JSON/CSV + rich console streams
- 📦 **Artifacts** — Checkpoints, plots, diagnostic reports
- 🌐 **Experiment Tracking** — MLflow/W&B integration (optional)

---

## 1. Design Philosophy

- **Reproducibility-first** — every run logs its full Hydra config, dataset hash, and CLI call [oai_citation:2‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).
- **CLI-driven** — logs are written automatically when using `spectramind` subcommands [oai_citation:3‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).
- **UI-light** — rich console feedback via [Rich](https://github.com/Textualize/rich), optional HTML dashboards, no heavy GUIs [oai_citation:4‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).
- **Pluggable backends** — switch between TensorBoard, MLflow, CSV, or W&B via config overrides [oai_citation:5‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL).
- **Separation of concerns** — logging configs live here, independent of training code.

---

## 2. Directory Structure

```bash
configs/logging/
├── tensorboard.yaml     # TensorBoard logger
├── mlflow.yaml          # MLflow backend
├── wandb.yaml           # Weights & Biases integration
├── csv.yaml             # Minimal CSV logger
└── default.yaml         # Safe baseline (CSV + Rich console)


⸻

3. Usage in Hydra (train.yaml snippet)

defaults:
  - logger: tensorboard     # or mlflow | wandb | csv

Example CLI overrides

Run with MLflow:

spectramind train logger=mlflow logger.mlflow.tracking_uri="http://localhost:5000"

Run with TensorBoard + CSV dual logging:

spectramind train logger=[tensorboard,csv]

Run with W&B:

spectramind train logger=wandb logger.wandb.project="SpectraMindV50"


⸻

4. Logging Backends

🖥️ Console (default, always on)
	•	Rich-enhanced progress bars, live metrics, color-coded diagnostics ￼.
	•	Minimal overhead, Kaggle-safe.

📊 TensorBoard (tensorboard.yaml)
	•	Logs scalars, histograms, images.
	•	Compatible with Kaggle & local workflows.

🧪 MLflow (mlflow.yaml)
	•	Tracks params, metrics, artifacts to MLflow server.
	•	Enables experiment comparisons across runs ￼.

📦 Weights & Biases (wandb.yaml)
	•	Optional cloud experiment tracking.
	•	Supports team collaboration.

📑 CSV (csv.yaml)
	•	Lightweight, filesystem-only logs.
	•	Always safe for Kaggle offline mode.

⸻

5. Best Practices
	•	Always version configs — commit YAMLs to Git, never hard-code logging in code ￼.
	•	Use outputs/ structure — Hydra saves each run’s logs + configs in timestamped folders.
	•	Keep Kaggle-safe — default to console + CSV; heavy backends (MLflow/W&B) only when external network access is available ￼.
	•	Link logs to physics constraints — violations, FFT overlays, symbolic rule scores should also be logged ￼ ￼.
	•	Visualize via dashboards — diagnostics HTML reports (generate_html_report.py) consume logs and metrics for interpretability ￼.

⸻

6. Example Run Lifecycle
	1.	CLI call:

spectramind train --config-name train data=nominal model=v50


	2.	Hydra composes configs (incl. logging/) ￼.
	3.	Logger(s) initialized → logdir: outputs/train/YYYY-MM-DD_HH-MM-SS/.
	4.	Console streams live metrics; scalars saved to TensorBoard/CSV.
	5.	Artifacts (checkpoints, plots) stored and linked in logs.
	6.	Config snapshot + dataset hash written for full reproducibility ￼.

⸻

7. References
	•	SpectraMind V50 Technical Plan — Logging + Hydra integration ￼
	•	SpectraMind V50 Project Analysis — Config-driven reproducibility ￼
	•	Hydra for AI Projects: A Comprehensive Guide — Config groups + logging ￼
	•	CLI Technical Reference (MCP) — Logging standards for reproducibility ￼

⸻

✅ With this setup, /configs/logging transforms logs into scientific evidence — every result is fully traceable, Kaggle-safe, and NASA-grade reproducible.