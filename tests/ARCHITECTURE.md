# 🏗️ SpectraMind V50 — Testing Architecture

The `/tests` directory is designed to **mirror the SpectraMind V50 pipeline**, ensuring every subsystem has automated coverage.  

---

## 📐 Layered Testing Design

1. **Diagnostics Tests (`/diagnostics`)**  
   - Unit tests for scientific components.  
   - Examples: symbolic logic engine, FFT overlays, SHAP fusion, calibration checks.  

2. **Integration Tests (`/integration`)**  
   - Run full pipeline segments (calibration → training → prediction → dashboard).  
   - Ensures CLI subcommands chain correctly.  

3. **Regression Tests (`/regression`)**  
   - Locks in baseline metrics (e.g., GLL per-bin) to prevent silent degradation.  
   - Tests ablations and config sweeps (`auto_ablate_v50.py`).  

4. **Artifact Tests (`/artifacts`)**  
   - Validates output bundles, manifests, and logs.  
   - Example: submission ZIP structure, manifest hashes, debug logs.  

---

## 🔄 Testing Workflow

- **Local**: `pytest tests/` (developer validation).
- **CLI**: `spectramind test` (official entrypoint).
- **CI/CD**: GitHub Actions triggers all layers on each push:
  1. Build environment (Docker + Poetry).
  2. Run `/tests` in fast mode (`--fast-dev-run`).
  3. Run `/tests` deep mode nightly (`--deep`).
  4. Validate artifacts and configs.  

---

## 🛡️ Reproducibility & Logging

- All test runs log to `logs/v50_debug_log.md`.  
- Config + dataset hashes verified against `run_hash_summary_v50.json`.  
- DVC tracks artifacts, enabling exact reproduction.  

---

## 📊 Outputs

- **Pass/Fail** → pytest report + GitHub CI.  
- **Artifacts** → JSON/Markdown summaries in `/tests/artifacts`.  
- **Dashboards** → integrated into `generate_html_report.py` for visual inspection.  

---

## 🌌 Philosophy

The test suite treats **the pipeline as a scientific instrument**:  
- Each test = a calibration check.  
- No step unverified.  
- CI ensures that **every commit is “flight-ready.”**