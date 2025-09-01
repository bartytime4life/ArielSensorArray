# 📟 SpectraMind V50 — CLI Flow

This diagram shows how the unified `spectramind` CLI routes user intents to reproducible pipeline steps with Hydra configs, logging, and artifacts.

---

## 0) Quick reference

- Root command: `spectramind`
- Core subcommands: `calibrate`, `train`, `diagnose`, `submit`, `ablate`, `test`, `analyze-log`
- Configs: Hydra YAMLs under `configs/`
- Logs: `events.jsonl`, `v50_debug_log.md`
- Artifacts: saved under run-specific output folders

---

## 1) CLI flow (high level)

```mermaid
flowchart LR
  %% Root
  S[User] --> C[spectramind CLI]

  %% Subcommands
  C --> CAL[calibrate]
  C --> TR[train]
  C --> DIA[diagnose]
  C --> SUB[submit]
  C --> ABL[ablate]
  C --> TST[test]
  C --> LOG[analyze-log]

  %% Shared runtime
  CAL --> HY[Hydra compose configs]
  TR  --> HY
  DIA --> HY
  SUB --> HY
  ABL --> HY
  TST --> HY
  LOG --> HY

  HY --> RUN[Run step with typed params]
  RUN --> EVT[(events.jsonl)]
  RUN --> OUT[(artifacts dir)]
  RUN --> REP[reports md and html]
  RUN --> DBG[v50_debug_log.md]

  %% Typical data flow
  CAL --> DAT[(calibrated data)]
  TR  --> CKPT[(checkpoints)]
  DIA --> RPT[(diagnostic figures)]
  SUB --> PKG[(submission package)]
  ABL --> LBD[(leaderboard md html)]
````

---

## 2) Examples

```bash
# Calibrate raw inputs with a specific profile
spectramind calibrate data=nominal calibration.profile=std

# Train with overrides (learning rate and epochs)
spectramind train model=v50 optimizer.lr=0.0005 training.epochs=30

# Generate diagnostics and an HTML report
spectramind diagnose inputs.run_id=last report.html_out=reports/diag.html

# Produce a submission bundle
spectramind submit inputs.run_id=best output.dir=submit_v1

# Run an ablation grid
spectramind ablate run model=v50_a,v50_b optimizer.lr=0.0005,0.001 training.epochs=10

# Self test and integrity checks
spectramind test --deep

# Parse recent CLI calls into a compact table
spectramind analyze-log --limit 50 --md out/log_table.md
```

---

## 3) Notes

* All subcommands load Hydra configs first, then execute the step.
* Each run writes structured logs and artifacts to a timestamped output directory.
* Use CLI overrides to avoid editing code; reproducibility is maintained via saved configs and logs.
* Keep artifact paths short and stable for CI and packaging.

---

```
```
