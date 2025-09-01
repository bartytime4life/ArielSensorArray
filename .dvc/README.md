# `.dvc/` – Data Version Control (DVC) Guide

> Mission: keep **code + configs in Git**, **data/models in DVC**, and **ephemera out of both** — so every result is exactly reproducible.

---

## Why DVC here?

SpectraMind V50 moves large, evolving artifacts (calibrated data, features, model checkpoints, submissions) **out of Git** and into a DVC remote, while keeping **small, stable descriptors** (like `dvc.yaml`, `*.dvc`, and configs) **in Git**.  
This preserves a clean Git history and lets anyone `dvc pull` the exact bits tied to a commit/config.

---

## What goes **in Git** (commit these)

- **Pipeline descriptors**
  - `dvc.yaml` (stages/DAG)
  - `*.dvc` pointer files (for tracked artifacts outside the DAG)
- **DVC config (shared)**
  - `.dvc/config` (safe, no secrets)
  - `.dvcignore`
- **Plots & templates**
  - `.dvc/plots/*` (plot specs/templates you want versioned)

> Rule of thumb: If it’s **human-readable & stable** and needed to *recreate* a run, it belongs in Git.

---

## What goes **in DVC** (tracked, not committed)

- **Large, changing artifacts**
  - Raw & calibrated datasets
  - Intermediate feature stores
  - Trained model checkpoints, ensembling artifacts
  - Evaluation metrics JSON/CSVs (large), submission files
- **Any file ≥ a few MB** or likely to churn

> Track with `dvc add path/to/artifact` or as stage outputs in `dvc.yaml`. DVC stores a **pointer** in Git and the **content** in your remote (S3/GS/Azure/etc.).

---

## What is **ignored** (keep out of Git)

- DVC runtime internals & noise (already covered by `.dvc/.gitignore`)
  - `.dvc/cache/`, `.dvc/tmp/`, `.dvc/state/`, `.dvc/experiments/`, `.dvc/exp/`, `.dvc/logs/`, `.dvc/events/`, `.dvc/plots/tmp/`, `.dvc/stage.lock`, `.dvc/config.local`, etc.
- Any **local** remotes or user/session scratch under `.dvc/`

> These are machine-local, ephemeral, or contain secrets. Do not commit them.

---

## Typical workflow (CLI-first)

1. **Define/extend a stage** in `dvc.yaml`  
   Inputs: data + code + Hydra config.  
   Outputs: calibrated data / model / reports.

2. **Run the pipeline**  
   ```bash
   dvc repro                # or use `make` / `spectramind ...`
````

3. **Track & push artifacts**

   ```bash
   dvc add path/to/big_file         # if not produced by a stage
   git add path/to/big_file.dvc
   git commit -m "Track artifact via DVC"
   dvc push                         # upload to remote storage
   ```

4. **Share your run**

   ```bash
   git push
   # teammates do:
   git pull && dvc pull
   ```

---

## Experiments & metrics

* Run variants with Hydra overrides or `spectramind tune`, then:

  ```bash
  dvc exp run
  dvc exp show            # tabular compare of params/metrics
  dvc exp gc --workspace  # clean up dangling experiments (careful)
  ```
* Prefer logging key metrics to small JSON/CSV **and** declare them as `metrics` in `dvc.yaml` for easy comparison.

---

## Do / Don’t (quick checklist)

**Do**

* ✅ Commit `dvc.yaml`, `*.dvc`, `.dvc/config`, `.dvcignore`, plot templates.
* ✅ Track big/volatile artifacts with DVC; `dvc push` after `git push`.
* ✅ Reference artifacts as **stage outputs** where possible (repro > ad-hoc).
* ✅ Use a **shared remote** (S3/GS/Azure/etc.) and confirm everyone can `dvc pull`.
* ✅ Keep secrets in **`.dvc/config.local`** (never commit).

**Don’t**

* ❌ Don’t commit `.dvc/cache/`, `.dvc/tmp/`, `.dvc/state/`, experiments scratch, or `stage.lock`.
* ❌ Don’t commit large artifacts directly to Git.
* ❌ Don’t rely on local paths or machine-specific mounts in tracked configs.

---

## Troubleshooting

* **“File missing” after `git pull`** → run `dvc pull` (fetch blobs from remote).
* **“Cache not found”/slow pulls** → confirm remote is set (`dvc remote list`) and you have access; consider enabling `dvc cache dir` on a fast disk.
* **Merge conflicts in `*.dvc`** → they are small JSONs; resolve or regenerate by re-adding the artifact and recommitting.
* **Changed code, stale outputs** → `dvc repro` will rebuild only invalidated stages (smart caching).

---

## CI & Kaggle notes

* CI should run read-only `dvc pull` for artifacts required by smoke tests.
* For Kaggle notebooks without remote credentials, package needed artifacts as a Kaggle Dataset or export minimal files in the repo (small!).

---

## One-liner policy

> **If it’s large or changes often → DVC**.
> **If it’s small and defines the pipeline → Git**.
> **If it’s local noise or secrets → ignore**.

```
