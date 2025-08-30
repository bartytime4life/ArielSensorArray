# ⚙️ Calibration Methods — SpectraMind V50

This folder defines **method-level calibration profiles** for the SpectraMind V50 pipeline in the  
**NeurIPS 2025 Ariel Data Challenge**.  

Each file here specifies how **individual calibration steps** (ADC correction, dark subtraction, flat fielding, etc.)  
are performed, parameterized, and composed into the main calibration chain.

---

## 📂 Purpose

Calibration in SpectraMind V50 follows a **kill chain** of corrections:  
1. **ADC correction** — analog-to-digital conversion fix.  
2. **Nonlinearity correction** — removes detector non-linear response.  
3. **Dark subtraction** — subtracts detector dark current signal.  
4. **Flat fielding** — corrects pixel-to-pixel sensitivity variations.  
5. **Correlated double sampling (CDS)** — suppresses read noise.  
6. **Photometric extraction** — aperture or PSF-based extraction of stellar flux.  
7. **Trace normalization** — normalize extracted flux to baseline continuum.  
8. **Phase alignment** — aligns light curves to transit phase for model readiness.  

This folder allows **step-specific overrides**, so you can swap methods (e.g., polynomial vs spline detrending)  
without editing core pipeline code.

---

## 📂 Files (expected)

- **adc.yaml**  
  Parameters for analog-to-digital conversion correction.  
  - Offsets, gain tables, bit-depth adjustments.  

- **dark.yaml**  
  Dark frame subtraction configuration.  
  - Dark current maps, scaling factors.  

- **flat.yaml**  
  Flat-field calibration.  
  - Pixel sensitivity maps, normalization flags.  

- **cds.yaml**  
  Correlated double sampling (CDS) setup.  
  - Frame pairing rules, noise thresholds.  

- **photometry.yaml**  
  Photometric extraction options.  
  - Method: `aperture` | `psf` | `optimal`.  
  - Aperture radius, PSF kernel, centroiding algorithm.  

- **trace.yaml**  
  Trace normalization.  
  - Polynomial vs spline normalization, baseline windows.  

- **phase.yaml**  
  Phase alignment.  
  - Ephemeris source, transit window size, binning scheme.  

---

## 🔧 Usage

Each method config is Hydra-composable and can be overridden at runtime. For example:

### Run calibration with aperture photometry (radius=8 px)
```bash
spectramind calibrate calib/method=photometry calib.method.photometry.type=aperture calib.method.photometry.radius=8

Switch normalization method to spline

spectramind calibrate calib/method=trace calib.method.trace.type=spline


⸻

📝 Notes
	•	All method configs are atomic — they define one stage only.
	•	The master calib/nominal.yaml or calib/strict.yaml chains these methods in order.
	•	DVC tracks outputs of each method; unchanged inputs skip re-run automatically.
	•	This modular design allows rapid experimentation with different calibration techniques.

⸻

✅ With this structure, /configs/calib/method/ acts as the toolbox for the calibration pipeline — each YAML corresponds to one correction method, ensuring reproducibility, modularity, and Kaggle-safe overrides.

---