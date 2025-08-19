# SpectraMind V50 — Gravitational Lensing: Modeling & Mitigation (ArielSensorArray)

> Scope: make lensing effects first‑class citizens in the SpectraMind V50 pipeline — from theory → detection → correction — with CLI hooks, Hydra configs, and tests. Target stability: physics is time‑invariant and safe to vendor into the repo.

---

## 0) TL;DR

* **What**: Gravitational lensing magnifies and distorts background sources due to mass along the line of sight. For our use case (exoplanet transmission spectroscopy), the dominant risk is **time‑variable microlensing** (achromatic magnification) that can **perturb transit depths** and **spectral normalization** if its timescale overlaps the transit window.
* **When it matters**: Rare, but non‑negligible for dense fields; more likely at faint magnitudes and crowded lines of sight. Constant magnification **cancels** in the in/out‑of‑transit ratio; **time‑variable** magnification does **not**.
* **What we do**:

  1. **Detect**: catalog cross‑match (Gaia/2MASS) + residual diagnostics (wavelet/FFT) + simple microlensing curve fits.
  2. **Mitigate**: (A) re‑normalize with a smooth multiplicative model, or (B) **joint‑fit** transit × microlens with nuisance parameters, or (C) down‑weight affected windows.
  3. **Govern**: log flags/metrics; wire to CLI + Hydra; export to diagnostics HTML.

---

## 1) Where this plugs into SpectraMind V50

* **Calibration** (`src/asa/calib/`):

  * Add optional **lensing pre‑fit** and **multiplicative correction** after photometry/extraction but before normalization.
  * Emit `calibration/lensing.json` (per‑target) and `calibration/lensing_mask.npy`.
* **Modeling** (`src/asa/pipeline/`):

  * Expose nuisance variables (e.g., constant or slow drift in magnification) and pass as **metadata features**.
* **Diagnostics** (`src/asa/diagnostics/`):

  * New plots: residual power vs. microlensing templates; fit quality; per‑bin sensitivity impact.
* **Symbolic rules**:

  * Soft constraints: `magnification(t) ≥ 0.9`, smoothness on dA/dt, encourage achromaticity across bins.

---

## 2) Physics Essentials (minimal but sufficient)

### 2.1 Lens equation (thin‑lens approximation)

Let **β** be the unlensed angular position, **θ** the observed image position, and **α(θ)** the reduced deflection:

$$
\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta}), \quad
\boldsymbol{\alpha}(\boldsymbol{\theta}) = \frac{D_{LS}}{D_S}\,\hat{\boldsymbol{\alpha}}(\boldsymbol{\theta})
$$

Distances: $D_L$ (observer→lens), $D_S$ (observer→source), $D_{LS}$ (lens→source).

### 2.2 Einstein radius (point mass lens)

$$
\theta_E = \sqrt{\frac{4 G M}{c^2}\,\frac{D_{LS}}{D_L D_S}}
$$

Typical scale: micro‑ to milli‑arcseconds for stellar masses at kpc distances.

### 2.3 Point lens magnification (unresolved source)

Let $u = \frac{\beta}{\theta_E}$ be the dimensionless impact parameter. Total magnification:

$$
A(u) = \frac{u^2 + 2}{u\sqrt{u^2 + 4}}
$$

**Microlensing light curve** (Paczyński curve):

$$
u(t) = \sqrt{u_0^2 + \left(\frac{t - t_0}{t_E}\right)^2}
$$

Parameters: $u_0$ (closest approach), $t_0$ (epoch), $t_E$ (Einstein time scale).

### 2.4 SIS + shear (for completeness)

For a **singular isothermal sphere (SIS)** with velocity dispersion $\sigma_v$:

$$
\theta_E^{\text{SIS}} = 4\pi \left( \frac{\sigma_v}{c} \right)^2 \frac{D_{LS}}{D_S}, \quad
\mu = \frac{\theta}{\theta - \theta_E}
$$

External shear $\gamma$ modifies the mapping via the Jacobian $\mathbf{A}^{-1} = \mathbf{I} - \nabla \boldsymbol{\alpha}$; in weak lensing terms with convergence $\kappa$ and shear $\gamma$, total magnification $\mu = 1 / \left[(1-\kappa)^2 - \gamma^2\right]$.

### 2.5 Achromaticity and finite source caveat

* **Ideal GR lensing is achromatic** (no wavelength dependence).
* Apparent chromaticity may arise through **finite source effects** (brightness profile) and **differential blending/extinction**. For transmission spectra, assume **magnification ≈ multiplicative scalar** applied to all wavelengths at time $t$, unless evidence suggests otherwise.

---

## 3) Impact on Transmission Spectroscopy

* If **A(t)** is **constant** during baseline and in‑transit windows → cancels in $F_\text{in}/F_\text{out}$.
* If **A(t)** varies across ingress/egress/transit → biases derived **transit depth** and **spectral normalization**. This can masquerade as subtle features; must be modeled or removed.
* **Practical rule**: model **slow $A(t)$** as a low‑order polynomial or Paczyński template; verify **achromaticity** across bins (σ‑consistent residuals).

---

## 4) Detection & Mitigation Workflow

1. **Catalog cross‑match (optional, offline)**

   * Look for potential lenses/companions near the line of sight (e.g., `data/meta/gaia_neighbors.csv` produced offline).
   * Feature: **crowding index**, **nearest neighbor separation**, **magnitude contrast**.

2. **Residual diagnostics (mandatory)**

   * After standard calibration, compute **pre‑transit baseline residuals** and **in‑transit residuals**.
   * **Wavelet / FFT** scan for **low‑frequency trends** matching microlensing timescales ($t_E \gg$ exposure time; $t_E$ hours–days typical).
   * **Achromaticity test**: regress per‑bin residuals on a **common template**; demand consistent scaling across wavelengths.

3. **Template fitting (candidate confirmation)**

   * Fit a **Paczyński curve** $A(t; u_0, t_0, t_E)$ OR a **low‑order spline** across the window.
   * Model selection (AIC/BIC) against a null (constant or polynomial).

4. **Mitigation**

   * **Re‑normalize** by $\widehat{A}(t)$ (multiplicative correction), OR
   * **Joint‑fit** transit × microlensing in a single likelihood (preferred), OR
   * **Down‑weight / mask** time ranges with strongest inferred lensing.

5. **Governance**

   * Save `outputs/calibrated/<planet_id>/lensing.json` with parameters, evidence, p‑values.
   * Flag target in `diagnostic_summary.json`.
   * Include figures in HTML dashboard.

---

## 5) Hydra Config (drop‑in)

```yaml
# configs/calibration/lensing.yaml
lensing:
  enabled: true
  mode: "auto"        # ["off","poly","paczynski","auto"]
  poly_order: 2       # when mode=poly
  paczynski_init:
    u0: 0.5           # initial guess
    tE: 1.0           # days
    t0_offset: 0.0    # relative to mid-transit
  achromaticity_check: true
  achrom_threshold: 0.1   # max fractional spread across bins
  fit_window_pad: 0.25    # fraction of transit duration to include on each side
  logging_level: "INFO"
  save_artifacts: true
```

Enable from CLI:

```bash
python -m spectramind calibrate data=kaggle calibration=lensing
python -m spectramind diagnose lensing --html-out outputs/diagnostics/lensing_<planet>.html
```

---

## 6) Minimal Python Reference (vendor into `src/asa/calib/lensing.py`)

```python
# src/asa/calib/lensing.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

# --------------------------------------------------------------------------------------
# Microlensing core: Paczyński flux template and helpers
# --------------------------------------------------------------------------------------

def _u_of_t(t: NDArray[np.floating], u0: float, t0: float, tE: float) -> NDArray[np.floating]:
    """
    Dimensionless impact parameter over time:
        u(t) = sqrt(u0^2 + ((t - t0) / tE)^2)
    All times in the same units (e.g., days). tE > 0 enforced by caller.
    """
    return np.sqrt(u0**2 + ((t - t0) / tE) ** 2)

def paczynski_magnification(t: NDArray[np.floating], u0: float, t0: float, tE: float) -> NDArray[np.floating]:
    """
    Point-lens total magnification:
        A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
    """
    u = _u_of_t(t, u0, t0, tE)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

def poly_magnification(t: NDArray[np.floating], *coeffs: float) -> NDArray[np.floating]:
    """
    Smooth multiplicative trend model for weak/slow magnification.
    coeffs: c0, c1, ..., cK such that A(t) = 1 + c0 + c1*t + c2*t^2 + ...
    """
    deg = len(coeffs)
    powers = np.vstack([t**k for k in range(deg)])
    return 1.0 + (coeffs @ powers)

@dataclass
class LensingFitResult:
    mode: str                         # "none" | "poly" | "paczynski"
    params: Dict[str, float]          # fitted parameter dict
    param_cov: Optional[NDArray[np.floating]]
    achromatic_spread: Optional[float]
    aic: float
    bic: float
    success: bool

def fit_paczynski(t: NDArray[np.floating], y: NDArray[np.floating]) -> Tuple[LensingFitResult, NDArray[np.floating]]:
    """
    Fit a Paczyński magnification to a 1D time-series using non-linear least squares.
    Returns (fit_result, A_hat(t)).
    """
    # normalize around 1 to match multiplicative interpretation
    y_norm = y / np.nanmedian(y)
    p0 = (0.5, float(np.nanmedian(t)), max(0.25, float((np.nanmax(t)-np.nanmin(t))/2)))
    bounds = ((1e-3, np.nanmin(t)-10, 1e-3), (5.0, np.nanmax(t)+10, 1e3))
    try:
        popt, pcov = curve_fit(paczynski_magnification, t, y_norm, p0=p0, bounds=bounds, maxfev=20000)
        u0, t0, tE = map(float, popt)
        A = paczynski_magnification(t, u0, t0, tE)
        resid = y_norm - A
        n = len(t)
        k = 3
        sse = float(np.nansum(resid**2))
        aic = n * np.log(sse / n + 1e-12) + 2 * k
        bic = n * np.log(sse / n + 1e-12) + k * np.log(n)
        return (
            LensingFitResult(
                mode="paczynski",
                params={"u0": u0, "t0": t0, "tE": tE},
                param_cov=pcov,
                achromatic_spread=None,  # filled by caller if multi-bin
                aic=aic,
                bic=bic,
                success=True,
            ),
            A,
        )
    except Exception:
        return (
            LensingFitResult(
                mode="paczynski",
                params={},
                param_cov=None,
                achromatic_spread=None,
                aic=np.inf,
                bic=np.inf,
                success=False,
            ),
            np.ones_like(t),
        )

def apply_magnification_correction(flux: NDArray[np.floating], A_hat: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Divide observed flux by estimated magnification (achromatic assumption).
    """
    eps = 1e-12
    return flux / np.clip(A_hat, eps, None)

def estimate_achromatic_spread(A_hats_per_bin: NDArray[np.floating]) -> float:
    """
    Compute fractional spread across wavelength bins for achromaticity test:
        spread = (p95 - p5) / median
    """
    p5 = np.nanpercentile(A_hats_per_bin, 5, axis=0)
    p95 = np.nanpercentile(A_hats_per_bin, 95, axis=0)
    med = np.nanmedian(A_hats_per_bin, axis=0) + 1e-12
    return float(np.nanmedian((p95 - p5) / med))

def save_lensing_report(path: Path, result: LensingFitResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "mode": result.mode,
                "params": result.params,
                "aic": result.aic,
                "bic": result.bic,
                "achromatic_spread": result.achromatic_spread,
                "success": result.success,
            },
            f,
            indent=2,
        )
```

---

## 7) Quick Example: Joint Transit × Microlensing (conceptual)

Let $F_*(t)$ be the stellar flux, $T(t;\,\theta)$ the transit model (fractional dip; $0< T \le 1$), and $A(t;\,\phi)$ the magnification:

$$
F_\text{obs}(t, \lambda) \approx A(t;\phi)\, F_*(t, \lambda)\, T(t;\theta)
$$

* Fit either sequentially (estimate $A$, de‑magnify, then fit transit) or jointly (preferred if SNR permits).
* In our pipeline, we keep $A(t)$ **achromatic** unless diagnostics show bin‑dependent residual structure.

---

## 8) CLI Hooks (to register)

* `spectramind calibrate --with-lensing`
  Runs detection/fit; writes `lensing.json`, corrected lightcurves, and masks.
* `spectramind diagnose lensing`
  Generates plots: raw vs. corrected flux, residual PSD, Paczyński fit, achromatic spread, flags.

---

## 9) Tests (checklist)

* **Unit**:

  * `paczynski_magnification` monotonicity vs. $u$; symmetry about $t_0$.
  * Fit recovery on synthetic data with noise; tolerance on $u_0, t_E$.
  * Achromaticity spread returns \~0 on perfectly shared A(t) across bins.
* **Integration**:

  * Calibration end‑to‑end on toy dataset: baseline→fit→correction improves GLL or residual variance.
  * Diagnostics produce expected JSON/PNG/HTML artifacts.
* **Edge cases**:

  * Short windows ($<0.3\,t_E$) → prefer poly mode.
  * Flat signals → model selection returns “none” or “poly” with small coefficients.
  * NaNs masked safely.

---

## 10) Logging & Governance

* Append to `v50_debug_log.md`:

  * mode, AIC/BIC, $u_0,t_E,t_0$ if used, achromatic spread, artifacts saved.
* Emit `diagnostic_summary.json` fields:

  * `lensing.detected: bool`, `lensing.mode: str`, `lensing.achrom_spread: float`, `lensing.aic_delta: float`.

---

## 11) Limitations & Notes

* True chromatic effects from lensing are second‑order; if detected, they often implicate blending/extinction rather than GR. Treat with caution.
* Paczyński fits are **degenerate** with other slow systematics; prefer **model comparison** (AIC/BIC) + achromatic check.
* For extremely crowded fields, carry a **blending factor $f_\text{blend}$** (constant flux contribution) in the fit.

---

## 12) References (general)

* Standard microlensing literature (Paczyński 1986), weak/strong lensing reviews, and gravitational optics textbooks.
* Mission ops docs on systematics disentanglement for transit photometry.

---

## 13) Developer To‑Dos (fast path)

* [ ] Add `src/asa/calib/lensing.py` (above) + wire into calibration graph.
* [ ] Register `spectramind diagnose lensing` subcommand.
* [ ] Add `configs/calibration/lensing.yaml`; default `enabled: false`.
* [ ] Unit tests: `tests/calib/test_lensing.py`.
* [ ] Dashboard tiles: lensing fit panel, achromaticity badge, AIC/BIC deltas.

---

*End of document.*
