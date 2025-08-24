# 🏗️ Tests / Artifacts — Architecture

**SpectraMind V50** · *Neuro‑symbolic, physics‑informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*

This document explains how the **artifact‑integrity test suite** is wired into the SpectraMind V50 pipeline, how data flows through manifests and logs, and how CI blocks merges when artifact guarantees are not met.

---

## 🎯 What this suite guarantees

- **Reproducibility** — Artifacts are regenerable from code+config; digests match.
- **Auditability** — Logs and manifests form an append‑only, cross‑checked trail.
- **Submission safety** — Bundles pass validator checks before leaderboard upload.

---

## 📦 Components

| Component / File                          | Role in the system                                                                                 |
|------------------------------------------|------------------------------------------------------------------------------------------------------|
| `spectramind` CLI                         | Single entrypoint that builds all artifacts (submissions, reports, manifests, logs).                |
| `manifest_v50.(json|csv)`                 | Canon of artifacts (path, size, SHA256, created_at, run_id, config_hash).                          |
| `run_hash_summary_v50.json`               | Run metadata (git commit, env snapshot, config hash, timestamps).                                   |
| `logs/v50_debug_log.md` + `*.jsonl`       | Append‑only audit trail (invocation, merged config, result pointers, errors).                        |
| `/tests/artifacts/*`                      | Verification layer (see test matrix below).                                                         |
| CI workflow (`.github/workflows/*.yml`)   | Enforces gating: generate → verify → publish (fail fast on any artifact mismatch).                  |

---

## 🧪 Test matrix (what each spec asserts)

| Test file                                | Primary assertions                                                                                                  |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `test_submission_validator.py`           | Submission shape/headers/NaN policy; optional size limits; row–ID coverage.                                          |
| `test_manifest_hashes.py`                | Every manifest entry exists; recomputed SHA256 matches; orphan/missing file detection.                               |
| `test_log_integrity.py`                  | `v50_debug_log.md` is append‑only; timestamps monotonic; no duplicate run IDs; CLI line parses.                      |
| `test_dummy_data_generator.py`           | Determinism under fixed seed; schema (dims/metadata) correct; basic numeric sanity (ranges, sparsity).               |
| `test_cli_version_stamp.py`              | CLI `--version`/config hash recorded; matches `run_hash_summary_v50.json`; echoed into debug log.                    |
| `test_report_manifest_integrity.py`      | Report bundles (HTML/PNG/JSON) are manifest‑tracked; no dangling file refs; intra‑report links resolvable.           |
| `test_run_hash_summary_contents.py`      | Git commit present; toolchain/env fingerprint present; config digest equals pipeline merged config digest.            |

---

## 🔄 Data & control flow

```mermaid
flowchart TD
    subgraph CLI["spectramind (Typer)"]
      A1[calibrate] -->|artifacts| AO[outputs/…]
      A2[train] --> AO
      A3[diagnose] --> AO
      A4[submit] --> AO
      A1 --> L[logs/v50_debug_log.md]
      A2 --> L
      A3 --> L
      A4 --> L
      A5[any] --> H[run_hash_summary_v50.json]
      A5 --> M[manifest_v50.json/csv]
    end

    subgraph Tests["/tests/artifacts"]
      T1[test_submission_validator]
      T2[test_manifest_hashes]
      T3[test_log_integrity]
      T4[test_dummy_data_generator]
      T5[test_cli_version_stamp]
      T6[test_report_manifest_integrity]
      T7[test_run_hash_summary_contents]
    end

    AO --> T1 & T2 & T6
    L  --> T3 & T5
    H  --> T5 & T7
    M  --> T2 & T6

    subgraph CI["GitHub Actions"]
      C1[setup env]
      C2[run spectramind (sample)]
      C3[pytest tests/artifacts -v]
      C4[gated publish]
    end

    CLI -->|invoked by| C2
    Tests -->|executed by| C3
    C3 -->|all pass| C4
    C3 -->|fail any| X([❌ block merge])
````

**Reading the graph**

1. The CLI creates **artifacts**, **manifest**, **run summary**, and **logs**.
2. The **tests** recompute and cross‑check those views of truth.
3. **CI** runs a small, deterministic pipeline then executes the artifact suite; failures block merge.

---

## 🧰 CI hooks (reference sequence)

1. **Build**: create Python env (Poetry/UV/pip), cache deps, pin versions.
2. **Sample run**: `spectramind calibrate/train/diagnose --fast-dev-run --seed 1337`
3. **Tests**: `pytest tests/artifacts -v --maxfail=1`
4. **Gate**: on success, proceed (optionally upload artifacts as CI evidence); on failure, **stop**.

---

## 🚨 Failure modes this suite catches

* **Hash drift**: file changed without manifest update (or vice‑versa).
* **Audit holes**: missing CLI call; duplicate run IDs; non‑monotonic timestamps.
* **Submission regressions**: wrong shapes, NaNs, unexpected headers or ID gaps.
* **Report rot**: HTML/PNG/JSON files not in manifest, broken intra‑bundle links.
* **Provenance gaps**: missing git commit or mismatched config digest.
* **Nondeterministic stubs**: dummy‑data not reproducible under fixed seed.

---

## ➕ Extending the suite

* **New artifact type?**

  1. Add it to `manifest_v50.*` with SHA256 + metadata.
  2. Create `tests/artifacts/test_<new>.py` to recompute and assert digests + schema.
  3. Wire generation into a CI sample run so the test has real inputs.

* **Custom checks** (examples):

  * Enforce max bundle size before upload (safety against infra limits).
  * Enforce required tags/metadata in manifest entries (e.g., instrument, profile).
  * Validate cross‑run invariants (e.g., static assets identical across builds).

---

## 🏁 Quick commands

```bash
# run only artifact tests
pytest tests/artifacts -v

# regenerate sample artifacts locally (deterministic)
spectramind diagnose dashboard --fast-dev-run --seed 1337

# sanity: show current CLI stamp used by the tests
spectramind --version
```

---

## 📌 Design principles distilled

* **One source of truth per view**: manifest (files), run summary (provenance), logs (human audit).
* **Cross‑validation, not trust**: every view is recomputed and checked against the others.
* **Gate early**: tiny sample runs + fast tests catch issues before costly training/inference.

> *Artifacts are the DNA of experiments; this suite performs the genome integrity check.*

```
```
