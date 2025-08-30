# 🔬 Calibration Configs — SpectraMind V50

This folder defines **calibration pipeline configurations** for the SpectraMind V50 system in the **NeurIPS 2025 Ariel Data Challenge**.  
Calibration is the **first stage** of the pipeline: converting raw ESA Ariel telescope simulation frames into **physically corrected, science-ready light curves**.

---

## 📂 Purpose

Calibration ensures that **instrumental systematics** are corrected before modeling.  
Each YAML config specifies the calibration **kill chain** (corrections and normalizations) and runtime behavior.

The calibration stage includes:
- **ADC correction** (analog-to-digital)  
- **Nonlinearity correction**  
- **Dark subtraction**  
- **Flat fielding**  
- **Correlated double sampling (CDS)**  
- **Photometric extraction**  
- **Trace normalization**  
- **Phase alignment**

---

## 📂 Files (expected)

- **nominal.yaml**  
  Default calibration chain for leaderboard submissions.  
  - All steps enabled.  
  - Saves intermediate cubes for diagnostics.  
  - Kaggle-safe for ≤9h runs.

- **fast.yaml**  
  Lightweight calibration for CI/smoke tests.  
  - Minimal corrections (skip heavy steps like CDS or flat-fielding).  
  - Faster runtime, less physically rigorous.  

- **strict.yaml**  
  Full science-grade calibration.  
  - All corrections enabled, highest precision.  
  - Saves full intermediate outputs for audit.  
  - Recommended for **diagnostics** and **offline validation**, not Kaggle runs.

---

## 🔧 Usage

Calibration configs are composed in `train.yaml` or invoked directly via CLI:

### Run default calibration
```bash
spectramind calibrate --config-name nominal

Switch to fast mode (CI smoke test)

spectramind calibrate --config-name fast

Run strict calibration (science validation)

spectramind calibrate --config-name strict


⸻

📊 DVC Integration

Calibration is tracked as a DVC stage, ensuring reproducibility:
	•	Inputs: raw FGS1/AIRS frames + calibration config.
	•	Outputs: calibrated cubes (saved under data/processed/<profile>/calibrated).
	•	DVC caches results — if raw data & config are unchanged, calibration will be skipped automatically.

This guarantees deterministic, versioned calibration outputs for every experiment.

⸻

📝 Notes
	•	All calibration configs are Hydra-composable and can be overridden at runtime:

spectramind calibrate calibration.save_intermediate=false calibration.steps=[adc_correction,photometric_extraction]


	•	Lite/fast configs are intended for CI pipelines and debugging.
	•	Nominal configs are Kaggle-ready (≤9h safe mode).
	•	Strict configs are intended for NASA-grade diagnostics and may exceed Kaggle limits.

⸻

✅ With this setup, calibration configs act as the launchpad for every SpectraMind V50 run — ensuring reproducibility, physics-informed rigor, and Kaggle-safe deployment.

---