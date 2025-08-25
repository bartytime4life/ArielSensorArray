# 🧪 SpectraMind V50 — Tests Directory

This directory hosts the **full testing suite** for the SpectraMind V50 pipeline used in the NeurIPS 2025 Ariel Data Challenge.  

All tests follow the **Master Coder Protocol (MCP)**: documentation-first, CLI-driven, reproducibility-safe.  

---

## 📂 Structure

- `diagnostics/` — unit & scientific diagnostics tests (FFT, SHAP, symbolic overlays, calibration).
- `integration/` — end-to-end pipeline tests (calibrate → train → predict → diagnose).
- `regression/` — regression & ablation tests; ensures new commits do not break prior behavior.
- `artifacts/` — artifact validation (submission bundles, manifests, logs).

---

## 🧭 Usage

Run all tests with:

```bash
pytest -v tests/