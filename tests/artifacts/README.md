# 🧪 Tests — Artifacts Validation Suite

**SpectraMind V50** · *Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*

This directory contains **unit and regression tests** for validating the **integrity of pipeline artifacts**.  
Artifacts in this context are the **diagnostic bundles, manifests, hashes, and dummy data** produced by the pipeline.  
These tests ensure that **reproducibility, auditability, and leaderboard-ready compliance** are maintained at all times.

---

## 📂 File Index

| Test Script                          | Purpose                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------- |
| `test_submission_validator.py`       | Ensures generated Kaggle submission files have correct shape & schema.   |
| `test_manifest_hashes.py`            | Validates that run hashes and manifests match across pipeline artifacts. |
| `test_log_integrity.py`              | Checks that `logs/v50_debug_log.md` entries are consistent & complete.   |
| `test_dummy_data_generator.py`       | Verifies that synthetic test data is reproducible & schema-aligned.      |
| `test_cli_version_stamp.py`          | Confirms CLI `--version` stamps are recorded in manifests & logs.        |
| `test_report_manifest_integrity.py`  | Ensures HTML/Markdown/JSON reports reference correct artifact hashes.    |
| `test_run_hash_summary_contents.py`  | Validates structure and correctness of `run_hash_summary_v50.json`.      |

---

## 🧭 Testing Philosophy

- **Reproducibility First**: All tests assert **hash consistency** across runs.  
- **Auditability**: Logs and manifests are validated as append-only audit trails.  
- **CLI-Driven**: Every artifact tested here is created via `spectramind` CLI commands.  
- **Fail Fast**: Tests provide clear errors for missing, corrupt, or mismatched artifacts.

---

## 🚀 Running Tests

To execute only the artifacts validation suite:

```bash
pytest tests/artifacts -v
````

Or run **all tests** in the repository (integration + diagnostics + regression):

```bash
make test
# or
pytest -v
```

---

## ✅ Example Outputs

* **Submission Validation**: Ensures `submission.csv` has `(1100, 566)` shape with no NaNs.
* **Manifest Integrity**: Confirms every artifact listed in `manifest.json` exists and hash-matches.
* **Log Consistency**: Detects duplicate or missing CLI entries in `logs/v50_debug_log.md`.
* **Dummy Data**: Confirms `generate_dummy_data.py` output is deterministic under fixed seed.
* **Version Stamps**: Cross-checks CLI version metadata against `run_hash_summary_v50.json`.

---

## 🔒 CI Integration

These tests run automatically in the **GitHub Actions CI pipeline**:

* On each commit, artifacts are generated in a sandbox run.
* All artifact tests (`/tests/artifacts/`) must pass before merging.
* Failures block merges, preserving **leaderboard-safe reproducibility**.

---

## 🏆 Mission Alignment

Artifact validation is central to SpectraMind V50’s mission:

* **NASA-grade rigor** (hashes, manifests, logs must match)
* **Challenge compliance** (submission files always Kaggle-ready)
* **Audit trail integrity** (no silent drift in configs or artifacts)

This ensures every run of SpectraMind V50 can be trusted, reproduced, and submitted with confidence.

```
