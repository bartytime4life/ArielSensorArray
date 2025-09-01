# 🛰️ `.dvc/ARCHITECTURE.md`

## SpectraMind V50 — Data Version Control (DVC) Subsystem  
*Neuro-symbolic, physics-informed pipeline for the NeurIPS 2025 Ariel Data Challenge*

---

## 📌 Purpose of `.dvc/`

The `.dvc/` directory is the **control center for artifact reproducibility** in SpectraMind V50:

- **Git** controls *code + configs* → logic reproducibility.  
- **DVC** controls *data + models* → artifact reproducibility.  

Every run of the pipeline is tied to:
1. A **Git commit hash** (immutable code + configs).  
2. A **DVC snapshot** (datasets, calibration outputs, model artifacts).  

🔗 See also: [docs/architecture.md](../docs/architecture.md) for the *full pipeline design*.  
This `.dvc/` subsystem is the **Data & Artifact Layer** of the architecture:contentReference[oaicite:0]{index=0}.

---

## 📂 Directory Layout

```plaintext
.dvc/
├── cache/          # Local cache of binary blobs (never commit)
├── tmp/            # Ephemeral staging (safe to delete)
├── config          # Global DVC config (committed, multi-cloud ready)
├── config.local    # Local overrides (ignored by Git)
├── plots/          # Plot templates (loss, calibration, FFT, symbolic metrics)
├── lock/           # Auto-generated locks (not committed)
└── state/          # SQLite + exp states (ignored)
````

### 🔒 Commit Policy

* ✅ **Commit**: `.dvc/config`, `.dvcignore`, `dvc.yaml`, `*.dvc` pointers, `.dvc/plots/*`
* ❌ **Ignore**: `.dvc/cache/`, `.dvc/tmp/`, `.dvc/config.local`, `.dvc/state/`

Supporting docs:

* `.dvc/.dvcignore.readme.md`
* `.dvc/.gitattributes.readme.md`

---

## ⚙️ Integration with SpectraMind V50

1. **Hydra Configs → Typer CLI → DVC**

   * Commands like `spectramind calibrate` or `spectramind train`
   * Hydra composes configs
   * CLI executes pipeline stage
   * DVC snapshots outputs → `.dvc` pointers

2. **Reproducibility Loop**

```bash
git checkout <commit>
dvc checkout
```

→ restores exact datasets, models, and diagnostics.

3. **CI/CD Enforcement**

   * GitHub Actions checks `dvc status` on every PR
   * Broken/missing pointers → **merge blocked**
   * Mirrors the “pre-flight safety check” in root architecture

---

## 📊 Plots & Metrics

* `dvc plots` renders:

  * Loss curves
  * Calibration reliability
  * FFT & autocorr diagnostics
  * Symbolic violation metrics

* `.dvc/plots/` holds JSON/YAML templates, reused across runs.

* Outputs are embedded into `report.html` diagnostics dashboard.

---

## 🚀 Typical Workflows

### Track a dataset

```bash
dvc add data/raw/fgs1_lightcurves.fits
git add data/raw/fgs1_lightcurves.fits.dvc .gitignore
git commit -m "Track raw FGS1 lightcurves with DVC"
```

### Reproduce pipeline

```bash
dvc repro
```

### Push artifacts to remote

```bash
dvc push -r storage
```

### Pull artifacts from remote

```bash
dvc pull -r storage
```

---

## 🌌 Best Practices

* Always run via **CLI (`spectramind …`)**, never raw Python.
* Use **Hydra overrides** for dataset/model paths; never hardcode.
* Run `dvc push` after success to sync remotes.
* CI requires **clean `dvc status`** before merging.
* Never commit blobs in `cache/` or `tmp/`.

---

## ✅ Acceptance Criteria

The `.dvc/` subsystem is **mission-grade** when:

* `git checkout && dvc checkout` fully restores any run.
* All stages (calibrate → train → predict → diagnose → submit) exist in `dvc.yaml`.
* CI enforces clean `dvc status` before merges.
* Multi-cloud remotes (S3/GCS/Azure) are continuously synced.
* Kaggle submissions reproduce bit-identical artifacts.

---

## 🛡️ Alignment with Root Architecture

* **Glass-box reproducibility**
* **CLI-first workflows**
* **Config-as-code (Hydra + DVC)**
* **CI/CD enforced artifact integrity**

This subsystem guarantees that the Data & Artifact Layer of SpectraMind V50 remains **fully transparent, scientifically verifiable, and Kaggle/CI reproducible**.

---

```
