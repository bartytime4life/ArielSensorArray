# 🧪 Tests — Artifacts Directory

**SpectraMind V50** · *Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*:contentReference[oaicite:0]{index=0}

This directory contains **unit + integrity tests** for all **artifact-level reproducibility files**.  
Artifacts here are *not models or spectra*, but the **audit trail** that guarantees every run can be  
**reproduced, validated, and trusted** — the backbone of our *NASA-grade reproducibility mandate*:contentReference[oaicite:1]{index=1}.

---

## 🎯 Purpose

The `/tests/artifacts` suite ensures:

1. **Submission Validity** — Submissions pass formatting, metric, and hash checks before Kaggle upload:contentReference[oaicite:2]{index=2}.  
2. **Manifest Integrity** — Every file in `/artifacts/` (submissions, logs, manifests) matches its declared hash:contentReference[oaicite:3]{index=3}.  
3. **Run Hash Coherence** — Config hash (`run_hash_summary_v50.json`) and pipeline outputs are cryptographically consistent:contentReference[oaicite:4]{index=4}.  
4. **Log Auditing** — CLI logs (`logs/v50_debug_log.md`, `logs/v50_runs.jsonl`) are append-only, well-formed, and timestamp-aligned.  
5. **Dummy Data Assurance** — Synthetic test data generators produce valid AIRS/FGS1 cubes for CI regression runs.  

These tests are central to **continuous integration (CI)** — any failed artifact integrity check blocks merges,  
treating the pipeline like a *scientific instrument under calibration*:contentReference[oaicite:5]{index=5}.

---

## 📂 File Index

| Test Script                          | Purpose                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------- |
| `test_submission_validator.py`       | Verifies CSV submissions: schema, μ/σ bins = 283, numeric validity, size limits:contentReference[oaicite:6]{index=6}. |
| `test_manifest_hashes.py`            | Ensures manifests (`manifest_v50.json/csv`) match file checksums.        |
| `test_log_integrity.py`              | Validates CLI logs: no missing timestamps, duplicate entries, or truncation. |
| `test_dummy_data_generator.py`       | Confirms synthetic AIRS/FGS1 cubes conform to shape + metadata specs.   |
| `test_cli_version_stamp.py`          | Verifies CLI `--version` outputs correct build hash + timestamp.        |
| `test_report_manifest_integrity.py`  | Cross-checks diagnostic reports vs manifest hashes for dashboard exports. |
| `test_run_hash_summary_contents.py`  | Validates run hash JSON: config/env/git commit all captured reproducibly. |

---

## 🔬 How to Run

Run all artifact tests from project root:

```bash
poetry run pytest tests/artifacts -v
````

Or run a specific test:

```bash
poetry run pytest tests/artifacts/test_submission_validator.py::test_valid_submission
```

CI automatically runs these on **every commit** via GitHub Actions.

---

## 🛡️ Scientific & Competition Guardrails

* **Reproducibility First**: Every artifact carries cryptographic fingerprints.
* **Kaggle Compliance**: Submissions pre-validated to avoid format errors or leaderboard rejection.
* **Fail-Fast Principle**: Any log or manifest corruption blocks merges, preventing silent drift.
* **Audit Ready**: Each test outputs machine-parsable JSON + human-readable Markdown for archives.

---

## 🌌 Philosophy

Artifacts are the **DNA of experiments**: logs, manifests, hashes, and submissions.
The `/tests/artifacts` directory enforces that this DNA is **untainted, traceable, and regenerable** —
critical for **mission-grade exoplanet spectroscopy research**.

---

```

---
