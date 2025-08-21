Awesome — here’s the complete assets/KAGGLE_GUIDE.md, ready to paste.

# SpectraMind V50 — Kaggle Integration Guide

**Purpose.** This document is the practical bridge between our architecture and a reproducible, leaderboard-safe Kaggle workflow. It explains how Kaggle works, how to run SpectraMind V50 within Kaggle’s constraints, and how to submit competitively while preserving scientific rigor.

---

## 1) Kaggle Platform — What matters for SpectraMind V50

### Core components
- **Competitions/Leaderboards.** You submit predictions that are scored on hidden test data. The **public leaderboard** uses a *subset* of test data for real‑time feedback; the **private leaderboard** uses the rest and determines final ranks (“shake‑up” often happens if models overfit the public split) [oai_citation:0‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Datasets.** Share versioned datasets privately or publicly; attach them to notebooks as inputs. Dataset versions allow exact reproducibility [oai_citation:1‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Notebooks.** Hosted Jupyter environments with CPU/GPU/TPU. Containers are standardized; internet is typically off for submissions; attach inputs via the right‑hand **Data** panel [oai_citation:2‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Community.** Discussion forums and shared notebooks help with debugging and strategy; contributions also count toward user progression tiers [oai_citation:3‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).

### Runtime constraints (typical free tier)
- **GPU:** Tesla P100 class (~13 GB VRAM), limited concurrent sessions.  
- **Session length:** ~12 hours per run; weekly GPU quotas (~30 hours) [oai_citation:4‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Internet:** **Off** by default for reliability; prepare all resources via attached datasets/models.  
- **Datasets:** Size and file‑count limits; use versions to freeze exact inputs [oai_citation:5‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).

**Implication for V50.** We must:
1) Package code + configs + models as attached datasets,  
2) Fit within GPU VRAM (batching, mixed precision),  
3) Keep wall‑time < 12 h (full pipeline over ~1,100 planets was designed for strict budget) [oai_citation:6‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w),  
4) Avoid public‑LB overfitting (robust CV and uncertainty handling).

---

## 2) Leaderboard Strategy — Avoid the “shake‑up”

- **Public vs Private.** Optimize for *generalization*, not only the public split; expect re‑ranking at reveal [oai_citation:7‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Submission caps.** Use daily limits wisely; track experiment metadata tightly (config hash, commit, seed).  
- **Reproducible CV.** Deterministic splits & seeds; capture config snapshots (Hydra) and data hashes (DVC) [oai_citation:8‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:9‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).  
- **Uncertainty sanity.** Penalized metrics (like Gaussian log‑likelihood) reward calibrated σ. Use temperature scaling and COREL‑style relational calibration to reduce overconfidence on OOD cases [oai_citation:10‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).

---

## 3) Lessons from public baselines (and how V50 improves)

**Compared Kaggle models** (publicly shared; details summarized):  
- **Thang Do Duc “0.329 LB” baseline.** Residual MLP; simple preprocessing; robust reference; no explicit uncertainty; ~0.329 public LB [oai_citation:11‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
- **V1ctorious3010 “80bl‑128hd‑impact”.** Very deep residual MLP (~80 blocks); improved feature capacity; risk of variance/overfit; batchnorm+dropout [oai_citation:12‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
- **Fawad Awan “Spectrum Regressor”.** Multi‑output spectrum regressor; stable and interpretable; slightly lower public LB than deep model [oai_citation:13‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).

**SpectraMind V50 upgrades (why it generalizes better):**  
- **Encoders:** FGS1 → **Mamba SSM** for ultra‑long sequences; AIRS → **Graph NN** with λ‑adjacency & molecular/region edges (prior‑aware message passing) [oai_citation:14‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).  
- **Symbolic constraints:** Smoothness, non‑negativity, FFT coherence, molecular alignment — as regularizers and diagnostics overlays [oai_citation:15‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).  
- **Uncertainty:** **Temperature scaling + COREL GNN** for binwise coverage and relational calibration [oai_citation:16‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).  
- **Reproducibility:** Typer CLI + **Hydra configs** + **DVC** + CI self‑test; config hash & run logs ensure exact recovery of leaderboard submissions [oai_citation:17‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:18‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

---

## 4) Running V50 on Kaggle — Step‑by‑step

> Goal: Execute **predict → calibrate‑σ → validate → package → submit**, all without internet, inside Kaggle.

### A. Prepare inputs as Kaggle Datasets (done locally, then upload)
1. **Code bundle**: repository subset needed for inference/packaging (e.g., `src/`, `spectramind.py`, `configs/`, minimal `pyproject.toml`/wheel).  
2. **Weights & artifacts**: trained checkpoints, COREL calibration artifacts, tokenizer/graph meta.  
3. **Runtime configs**: Hydra `configs/` with production defaults; freeze versions.  
4. **Diagnostic HTML template** (optional): `assets/report.html` + `assets/diagnostics_dashboard.html`.

> Version each dataset after any change; reference them by **version** in notebooks for immutability [oai_citation:19‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).

### B. Create the Kaggle Notebook
- **Hardware**: Select **GPU** (P100 class).  
- **Data**: Attach the above datasets in the right sidebar (`Add data`).  
- **Internet**: Keep **off**; rely only on attached data.  
- **Environment**: Use pinned image; avoid surprise upgrades (pin option) [oai_citation:20‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).

### C. Notebook inference commands (example skeleton)
```bash
# 1) Self‑test (fast) – verify env, paths, shapes
!python -m spectramind test --mode=fast

# 2) Predict μ, σ_raw
!python -m spectramind predict \
  +data.split=test +runtime.kaggle=true \
  +paths.input=/kaggle/input/ADC2025 \
  +paths.out=/kaggle/working/output \
  +model.ckpt=/kaggle/input/v50-weights/model.ckpt

# 3) Calibrate σ (temperature + COREL)
!python -m spectramind calibrate-temp \
  +calib.corel=true +paths.out=/kaggle/working/output

# 4) Validate & package (CSV/ZIP per rules)
!python -m spectramind submit \
  +submit.bundle=/kaggle/working/submission.zip \
  +report.html=true

Notes
	•	All CLI commands capture and save the composed Hydra config and run logs; DVC hashes are printed when applicable ￼.
	•	Keep memory within the P100 envelope (batch size, AMP).
	•	Ensure wall‑time < 12 h by using our optimized loaders and vectorized computations ￼.

⸻

5) Submission packaging and checks
	•	Validator enforces shape/bins/coverage; fails fast with actionable messages (console + HTML) ￼.
	•	Bundle includes: predictions CSV, config snapshot, optional HTML report, minimal manifest for audit.
	•	Kaggle submit from the right panel or programmatically in the notebook (if enabled).

⸻

6) Reproducibility & governance
	•	Hydra: store outputs/*/hydra/*.yaml with final overrides.
	•	Run hash & logs: persist logs/v50_debug_log.md & JSONL events for audit trails ￼.
	•	DVC: point to exact dataset/model versions; pull is offline via attached datasets on Kaggle ￼.
	•	CI mirror: our GitHub Actions smoke‑test runs a small E2E to ensure the notebook steps won’t regress ￼.

⸻

7) Troubleshooting (Kaggle‑specific)
	•	Out of memory (OOM): lower batch size; enable AMP; shard inference.
	•	Runtime limit: cache intermediates; pre‑export graph features; avoid heavy plots; use our fast I/O.
	•	Missing internet: every dependency must be vendored or in attached datasets.
	•	Submission rejects: re‑run validator; check column order/headers and exact bin count.

⸻

8) Quick FAQ

Q: Why not train on Kaggle?
A: Free tier quotas/time may be tight for full V50 training. We train elsewhere; Kaggle focuses on inference + calibration + packaging with versioned artifacts.

Q: How do we avoid public‑LB overfit?
A: Strong CV, calibrated σ, and conservative model selection; never cherry‑pick purely on public LB.

Q: Can I fork and reproduce locally?
A: Yes — the same Typer CLI + Hydra configs + DVC workflow works locally and on CI; Kaggle is a sandbox with locked inputs.

⸻

9) References
	•	Kaggle: platform, datasets, notebooks, leaderboards ￼.
	•	Public model comparisons for the Ariel Data Challenge 2025 ￼.
	•	SpectraMind V50 technical plan: CLI, Hydra/DVC, uncertainty calibration, CI self‑test ￼.
	•	Project analysis & reproducibility patterns ￼.

⸻
