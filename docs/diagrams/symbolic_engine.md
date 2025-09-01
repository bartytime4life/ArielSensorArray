# 🧠 SpectraMind V50 — Symbolic Engine Flow

This document shows how the **symbolic engine** evaluates physics and domain rules over model outputs, produces **violation maps** and **influence scores**, and feeds those signals back into **training**, **diagnostics**, and **calibration**. It is GitHub-ready and the Mermaid diagrams below use safe ASCII labels only.

---

## 0) What the symbolic engine does

- **Inputs**
  - Model predictions: `mu` spectrum per target, optional `sigma`
  - Metadata: wavelength grid, molecule bands, detector regions, flags
  - Rule set: YAML or JSON list of symbolic constraints and weights

- **Core steps**
  - Parse rules → build masks and operators for each rule
  - Evaluate rules on `mu` (and optional `sigma`) to get **per-bin** and **per-rule** losses
  - Aggregate to **per-target** violation scores and **global** metrics
  - Compute **influence maps** (rule and bin attribution) for explainability

- **Outputs**
  - Violation maps (`.npy`, `.csv`, or `.json`)
  - Rule tables and summaries (Markdown and HTML)
  - Influence and heatmap figures (PNG or HTML)
  - Optional feedback terms for training loss
  - Signals for uncertainty calibration and conformal layers

---

## 1) High level flow

```mermaid
flowchart TD
  %% Inputs
  MU[Predictions mu] --> PARSE[Parse rules]
  SG[Sigma optional] --> PARSE
  MD[Metadata] --> PARSE
  RS[Rules YAML] --> PARSE

  %% Build
  PARSE --> MASKS[Build masks per rule]
  PARSE --> OPS[Build rule operators]

  %% Evaluate
  MASKS --> EVAL[Evaluate rules]
  OPS --> EVAL
  MU --> EVAL
  SG --> EVAL

  %% Aggregate and explain
  EVAL --> AGG[Aggregate scores]
  AGG --> SUMM[Rule and target summaries]
  EVAL --> INFL[Influence map and attribution]

  %% Outputs
  SUMM --> OUT1[(violation tables)]
  INFL --> OUT2[(influence figures)]
  SUMM --> OUT3[(html and md reports)]

  %% Feedback to other systems
  AGG --> TRAIN[training loss terms]
  AGG --> DIAG[diagnostics]
  AGG --> CAL[uncertainty and calibration]
````

---

## 2) Components and data model

```mermaid
flowchart LR
  RS[Rules YAML] --> RSET[Rule set]
  MD[Metadata] --> BANDS[Molecule bands]
  MD --> REG[Detector regions]
  MD --> GRID[Wavelength grid]

  RSET --> BUILD[Mask and operator build]
  BANDS --> BUILD
  REG --> BUILD
  GRID --> BUILD

  BUILD --> CORE[Rule evaluator]
  MU[mu] --> CORE
  SG[sigma optional] --> CORE

  CORE --> VEC[Per bin vectors]
  CORE --> RSC[Per rule scores]
  VEC --> AGG[Aggregations]
  RSC --> AGG
  AGG --> REP[Reports and plots]
```

---

## 3) Rule examples (conceptual)

> Keep rules in a single YAML and keep labels short and ASCII only.

```yaml
# rules.yaml
- id: nonneg
  desc: spectrum is non negative
  weight: 1.0
  type: threshold
  op: mu >= 0.0

- id: smooth_l2
  desc: second diff penalty
  weight: 0.5
  type: smooth_l2
  window: 3

- id: band_coherence
  desc: molecule band coherence
  weight: 0.7
  type: group_mean
  groups: bands.h2o
  op: abs(mu - group_mean) <= thresh

- id: detector_region
  desc: region consistency
  weight: 0.3
  type: region_consistency
  groups: regions.detA
```

---

## 4) CLI quick reference

> Use short overrides to keep runs reproducible and logged.

```bash
# Evaluate rules and write violation tables and plots
spectramind diagnose symbolic \
  inputs.run_id=last \
  rules=rules.yaml \
  outputs.dir=artifacts/symbolic_last

# Rank rules by mean violation and export a markdown table
spectramind diagnose symbolic-rank \
  inputs.run_id=last \
  rules=rules.yaml \
  export.md=reports/symbolic_rank.md

# Generate influence maps for top rules
spectramind diagnose symbolic-influence \
  inputs.run_id=last \
  rules=rules.yaml \
  top_k=5 \
  export.dir=reports/symbolic_influence

# Train with symbolic loss enabled
spectramind train \
  model=v50 \
  loss.symbolic.enable=true \
  loss.symbolic.rules=rules.yaml \
  loss.symbolic.weight=0.2
```

---

## 5) Outputs and files

* `symbolic_violation_table.csv` — per target x per rule scores
* `symbolic_violation_map.npy` — per target x bin violation vectors
* `symbolic_rank.md` — rule ranking by mean or weighted score
* `symbolic_influence_*` — influence figures per target and rule
* `symbolic_report.html` — compact HTML report with links to artifacts

---

## 6) Notes for robust usage

* Keep **labels ASCII** and **short** to avoid renderer errors.
* Avoid line breaks inside Mermaid node labels.
* Validate rules on a small set first; fail fast on YAML errors.
* Version rules and metadata alongside the run for full reproducibility.
* Log rule hash and config snapshot with every diagnosis run.

---

```
```
