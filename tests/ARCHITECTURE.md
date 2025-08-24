# 🧪 Tests — Top‑Level Architecture

**SpectraMind V50** · *Neuro‑symbolic, physics‑informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*

The `tests/` tree is the **verification nervous system** of SpectraMind V50.  
It validates **reproducibility (artifacts), scientific correctness (diagnostics), end‑to‑end behavior (integration)**, and **stability over time (regression)** — all wired into CI so only **audit‑clean** code reaches `main`.

---

## 📂 Directory Map

```

tests/
├─ artifacts/      # Reproducibility & audit trail tests (manifests, hashes, logs, submissions)
├─ diagnostics/    # Scientific diagnostics tests (FFT/ACF, SHAP, symbolic overlays, HTML reports)
├─ integration/    # End‑to‑end CLI pipelines, configs wiring, packaging, submission flows
└─ regression/     # Golden snapshots, non‑flaky metrics, report diffs, CLI log trend checks

````

---

## 🧩 Test Suites & Responsibilities

| Suite          | Primary Focus                            | Guarantees                                                                                           |
|----------------|-------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **artifacts**  | Reproducibility & auditability            | Submissions schema/pass; manifests ↔ checksums; run‑hash/config capture; logs append‑only + timestamped |
| **diagnostics**| Scientific validity & explainability      | FFT/ACF math correct; SHAP & symbolic overlays deterministic; dashboards embed all plots/artifacts       |
| **integration**| End‑to‑end orchestration & packaging      | CLI commands, Hydra configs, DVC hooks, COREL/calibration, bundling, submission validator              |
| **regression** | Long‑horizon stability & non‑regression   | Golden outputs/plots/JSON stable; toleranced diffs; CLI heatmaps/log trend checks; perf guardrails      |

> **Fail‑fast rule:** any suite failure blocks CI merges.

---

## 🔄 End‑to‑End Data Flow (Tests × Pipeline × CI)

```mermaid
flowchart TD
    subgraph Inputs[Versioned Inputs]
      CODE[Pipeline Code]
      CFG[Hydra Configs]
      DATA[AIRS/FGS1 Data]
    end

    subgraph Runtime[Execution]
      CLI[spectramind CLI]
      RUN[Train/Infer/Diagnose]
    end

    subgraph Artifacts[Pipeline Outputs]
      SUB[Submission CSV (μ, σ)]
      LOGS[v50_debug_log.md / runs.jsonl]
      MAN[manifest_v50.json/csv]
      HASH[run_hash_summary_v50.json]
      REP[Diagnostics HTML/PNGs/JSON]
    end

    subgraph Tests[/tests]
      A[artifacts/]
      D[diagnostics/]
      I[integration/]
      R[regression/]
    end

    CI[GitHub Actions CI]:::ci
    KAG[Kaggle Submission]:::kaggle
    DASH[Diagnostics Dashboard]:::dash

    CODE --> CLI --> RUN --> Artifacts
    CFG  --> CLI
    DATA --> RUN
    Artifacts --> Tests
    Tests --> CI
    CI -->|on pass| KAG
    CI -->|on pass| DASH

    classDef ci fill=#0f62fe,stroke=#fff,color=#fff
    classDef kaggle fill=#20c997,stroke=#fff,color=#fff
    classDef dash fill=#9b59b6,stroke=#fff,color=#fff
````

---

## 🧪 Suite Details

### 1) `tests/artifacts`

* **What it checks:** submission schema (283 bins, numeric μ/σ), kaggle size/format, manifests ↔ checksum, append‑only logs, run‑hash content (config/env/git).
* **Why it matters:** artifacts are the **DNA** of runs; this suite guarantees **rebuildability** and **audit‑clean** provenance.

### 2) `tests/diagnostics`

* **What it checks:** FFT/ACF math; SHAP × μ overlays; symbolic violation & influence maps; UMAP/t‑SNE projections; HTML dashboard embed & manifest coverage.
* **Why it matters:** diagnostics are the **eyes** of the pipeline; this suite prevents **misleading science** and ensures **explainability** is reproducible.

### 3) `tests/integration`

* **What it checks:** full CLI paths (`calibrate`, `train`, `diagnose dashboard`, `submit`); Hydra overrides; packaging; validators; COREL/temperature‑scaling hooks.
* **Why it matters:** protects the **operator experience** — one command, correct artifacts, correct bundles, correct submissions.

### 4) `tests/regression`

* **What it checks:** golden JSON/CSV/PNG/HTML snapshots; toleranced numerical diffs (e.g., GLL, coverage); CLI log trend/heatmap consistency; performance envelopes.
* **Why it matters:** detects **drift** and **silent quality loss** across refactors or dependency bumps.

---

## 🛠️ Running Tests

Run everything (verbose):

```bash
poetry run pytest tests -v
```

Target a suite:

```bash
poetry run pytest tests/artifacts -v
poetry run pytest tests/diagnostics -v
poetry run pytest tests/integration -v
poetry run pytest tests/regression -v
```

Generate coverage:

```bash
poetry run pytest --cov=src --cov-report=term-missing
```

---

## 🔐 CI Policy (merge gate)

* **Every push / PR**: all four suites run in GitHub Actions.
* **Hard gate**: *any* failure blocks merge.
* **Artifacts**: CI uploads Markdown + JSON test reports for archival.
* **Reproducibility**: CI invokes the **same CLI** paths as humans (no hidden codepaths).

---

## 📎 Conventions & Guardrails

* **Append‑only logs**: `v50_debug_log.md` and `runs.jsonl` verified for monotonic timestamps & no duplicate run‑ids.
* **Cryptographic manifests**: manifest↔file checksums enforced; mismatches fail.
* **Deterministic diagnostics**: fixed seeds; accept small tolerances for numerics; non‑determinism = fail.
* **Golden snapshots**: stored under `tests/regression/_golden/` with README + update protocol.

---

## 🧭 Update Protocol (when outputs legitimately change)

1. Land the modeling/diagnostics change on a feature branch.
2. Run `pytest tests/…` locally; inspect diffs in `regression` failures.
3. If differences are **intended**, regenerate goldens via the suite‑provided helper (documented per test).
4. Open PR with: rationale, before/after metrics, and regenerated goldens.
5. CI must pass on all suites.

---

## ✅ Success Criteria

* **Reproducible**: same config + commit ⇒ same artifacts.
* **Explainable**: diagnostics render & embed consistently.
* **Operable**: CLI flows work end‑to‑end with clear errors.
* **Stable**: goldens only change with deliberate, reviewed updates.

---

## 📣 Notes for Contributors

* Prefer **small, focused tests** with clear failure messages.
* Keep **test data minimal**; use synthetic generators for heavy cases.
* When adding a tool or artifact, **add the test the same day**.
* If a diagnostic figures in the dashboard, **add a dashboard embed test**.

---

```

---
