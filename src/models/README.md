## 📂 Directory Contents

```

models/
├── fgs1\_mamba.py          # FGS1 encoder (Mamba state-space model for long photometric sequences)
├── airs\_gnn.py            # AIRS encoder (Graph Neural Network over spectral bins with edge features)
└── multi\_scale\_decoder.py # Multi-head decoder for μ/σ outputs (supports symbolic overlays, explainability)

````

---

## 🔑 Modules

### **`fgs1_mamba.py`**
- Encoder for **FGS1 photometric time series** (~135k × 32 × 32).  
- Based on **Mamba State-Space Models (SSM)** for efficient long-sequence modeling.  
- Handles **temporal jitter injection** and **photometric alignment**.  
- Outputs compressed latent embeddings aligned with AIRS features.  
- Integrated hooks for **symbolic overlays** (smoothness, priors).

---

### **`airs_gnn.py`**
- Encoder for **AIRS spectral channels** (~11k × 32 × 356).  
- Graph structure:  
  - **Nodes** = wavelength bins  
  - **Edges** =  
    • spectral proximity,  
    • molecular co-bands (H₂O, CO₂, CH₄),  
    • detector region adjacency.  
- Supports **edge features** (distance, molecule type, detector region).  
- Configurable GNN backends (e.g., **GATConv**, **RGCNConv**, **NNConv**).  
- Outputs latent AIRS embeddings that align with FGS1 features.  

---

### **`multi_scale_decoder.py`**
- **Fusion decoder** combining FGS1 and AIRS latent embeddings.  
- Outputs:  
  - **μ (mean transmission spectrum)**  
  - **σ (uncertainty estimate)**  
  - Optional **quantile/diffusion heads** for richer uncertainty modeling.  
- Symbolic overlays integrated:  
  - smoothness penalties,  
  - molecular priors,  
  - attention × symbolic fusion.  
- Supports **explainability**: attention weight tracing and symbolic influence overlays.  

---

## 🧭 Data Flow

```mermaid
flowchart LR
  FGS1[FGS1 calibrated time series] --> ENC1[fgs1_mamba.py]
  AIRS[AIRS calibrated spectra] --> ENC2[airs_gnn.py]
  ENC1 --> DEC[multi_scale_decoder.py]
  ENC2 --> DEC
  DEC --> MU[μ spectra]
  DEC --> SIG[σ spectra]
````

---

## ✅ Guarantees

* **Physics-informed**
  Encoders align with telescope data structures:
  • FGS1 → photometric time series
  • AIRS → spectral bins

* **Symbolic-ready**
  Decoder supports overlays for smoothness, molecular priors, and symbolic constraints.

* **Reproducible**
  Configurable via Hydra (`configs/config_v50.yaml`), version-logged with run hashes.

* **CI-tested**
  Shapes, outputs, and symbolic hooks validated in `selftest.py`.

---

## 🌌 Integration Notes

* Works in tandem with **`src/symbolic/`** for physics-informed loss shaping and diagnostics.
* Integrated with **`train_v50.py`** (training) and **`predict_v50.py`** (inference & submission).
* Outputs are consumed by **diagnostics tools** (e.g., SHAP overlays, UMAP embeddings, FFT smoothness maps).

---

> **Together, `models/` and `symbolic/` define the neuro-symbolic heart of SpectraMind V50 — blending deep learning encoders with astrophysical priors and interpretable symbolic overlays.**

```
```
