# 🧪 SpectraMind V50 — Training & Ablation Config Architecture

## 0) Purpose & Scope

This document explains how the **training** and **ablation** configuration layers in `configs/train/` plug into the CLI-first SpectraMind V50 pipeline.  
It shows the control flow from **CLI → Hydra composition → Ablation engine → Diagnostics/Leaderboard**,  
the major config groups involved, and the artifacts produced for **reproducibility** and **scientific auditability**:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 1) High-Level Execution DAG

The diagram below captures the end-to-end path for an ablation run, including Hydra config composition, loss/symbolic-rule toggles, job execution, diagnostics, and leaderboard export.

```mermaid
flowchart TD
  A0[User CLI] -->|spectramind ablate ... overrides| A1[Typer CLI Entrypoint]
  A1 --> A2[Hydra Compose \n defaults + overrides]
  A2 --> A3[Resolved Config \n (train/ablation.yaml)]
  A3 --> A4[Ablation Engine]
  A4 -->|spawn 1..N runs| A5[Trainer]
  A5 --> A6[Metrics & Artifacts]
  A6 --> A7[Diagnostics]
  A7 --> A8[Leaderboard Export \n (MD/HTML, top-N)]
  A6 --> L1[Logs \n v50_debug_log.md \n events.jsonl]
  A6 --> L2[Artifacts \n checkpoints, summaries]
  A7 --> L3[Reports \n HTML + PNGs]
  A3 --> C1[Symbolic/Loss Toggles \n (smoothness, FFT, nonneg, fp rules)]
  C1 --> A4
  A3 --> C2[Optimizer/Scheduler \n (AdamW, cosine, warmup)]
  C2 --> A5
  A3 --> C3[Trainer Limits \n (epochs, batch, AMP)]
  C3 --> A5
````

**Key ideas:**

* **CLI** is the control plane; **Hydra** composes the final config (base + overrides).
* **Ablation engine** mutates configs (physics-aware toggles) and dispatches runs.
* **Diagnostics** collate metrics into **leaderboards** (MD + HTML), while **logs** capture full provenance.

---

## 2) Control-Flow (Sequence View)

```mermaid
sequenceDiagram
  participant U as User
  participant CLI as spectramind (Typer)
  participant HY as Hydra
  participant AB as Ablation Engine
  participant TR as Trainer
  participant DG as Diagnostics

  U->>CLI: spectramind ablate loss.composite.fft.enabled=false ...
  CLI->>HY: compose(defaults, overrides)
  HY-->>CLI: resolved cfg (train/ablation.yaml)
  CLI->>AB: start ablation(cfg)
  AB->>AB: generate run grid (rules/weights/toggles)
  loop for each config
    AB->>TR: train(cfg_i)
    TR-->>AB: metrics, artifacts
  end
  AB->>DG: collate metrics, compute rankings
  DG-->>U: leaderboard.md / leaderboard.html, reports
```

---

## 3) Config Groups and Where They Live

* **`/configs/train/ablation.yaml`** — ablation-focused trainer defaults, symbolic/loss toggles, logging, leaderboard outputs.
* **`/configs/loss/composite.yaml`** — switchboard for physics-aware terms (GLL, smoothness, nonnegativity, FFT, asymmetry).
* **`/configs/model/*.yaml`** — model variants (V50 encoders/decoders).
* **`/configs/optimizer/*.yaml`** — optimizers (AdamW, SGD, Lookahead).
* **`/configs/trainer/*.yaml`** — runtime (epochs, batch, AMP, checkpoint).
* **`/configs/data/*.yaml`** — dataset/calibration composition (nominal, CI-slice, Kaggle-safe).

Hydra **defaults** in `train/*.yaml` bind these groups together and allow CLI overrides.

---

## 4) What the Ablation Layer Controls

* **Loss mix**: toggle/weight `smoothness`, `FFT`, `nonnegativity`, `asymmetry`, always keep `GLL`.
* **Symbolic constraints**: enable/disable **rule families** (fingerprints, lensing, photonic alignment).
* **Runtime safety**: Kaggle-safe (≤9h, ≤16GB), AMP, grad clipping, accumulation.
* **Export/reporting**: HTML/Markdown leaderboard; JSONL + Markdown logs with run hashes.

---

## 5) CLI Usage Patterns

### Disable Smoothness + FFT

```bash
spectramind ablate loss.composite.smoothness.enabled=false \
  loss.composite.fft.enabled=false
```

### Sweep Smoothness Weights

```bash
spectramind ablate -m loss.composite.smoothness.weight=0.0,0.05,0.1,0.2
```

### Toggle Symbolic Rule Families

```bash
spectramind ablate symbolic.rules[0].enabled=true symbolic.rules[1].enabled=false
```

### CI Fast Smoke

```bash
spectramind ablate trainer.epochs=2 ablation.parallel_runs=1 ablation.dry_run=true
```

---

## 6) Artifacts & Reproducibility

* **Logs**:

  * `logs/v50_debug_log.md` (CLI invocations, run hash)
  * `logs/events.jsonl` (structured metrics/events)
* **Diagnostics**:

  * `outputs/ablation_leaderboard.md` / `.html`
  * Plots (loss curves, FFT, symbolic overlays)
* **Models/Summaries**: checkpoints, JSON summaries, per-run snapshots
* **Repro steps**: re-run with same Hydra overrides or config snapshot

---

## 7) Config Snippet (Anchor)

```yaml
trainer:
  epochs: 30
  batch_size: 32
  gradient_accumulation: 2
  mixed_precision: true

loss.composite:
  gll: {enabled: true, weight: 1.0}
  smoothness: {enabled: true, weight: 0.1, cutoff: 50}
  fft: {enabled: true, weight: 0.1, cutoff_freq: 40}
  nonnegativity: {enabled: true, weight: 0.05}

symbolic:
  enabled: true
  rules:
    - {name: molecular_fingerprint, enabled: true, weight: 0.1}
    - {name: gravitational_lensing, enabled: false, weight: 0.05}

logging:
  markdown_log: logs/v50_debug_log.md
  jsonl_stream: logs/events.jsonl
  html_report: true

leaderboard:
  markdown: outputs/ablation_leaderboard.md
  html: outputs/ablation_leaderboard.html
```

---

## 8) Why This Layout Works

* **Hydra-first composition** = reproducibility + override power.
* **Symbolic-aware ablations** isolate contributions of physics-informed losses.
* **Reproducibility**: config snapshots + run hashes + DVC tie results to exact inputs.
* **CI-ready**: tiny configs run in <5 min, ensuring the pipeline is always green.

---

## 9) Extending the DAG

* Add new **loss terms/symbolic rules** → append to `loss.composite` or `symbolic.rules`.
* Register new **diagnostics** → export to HTML dashboard.
* Integrate experiment tracking (MLflow/W\&B) → track metrics/artifacts alongside configs.

---

```
```
