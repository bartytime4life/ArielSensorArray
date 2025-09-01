Here’s a drop-in **`docs/diagrams/dvc-structure.md`** with a Mermaid diagram and a concise explainer for new contributors.

````markdown
# DVC Structure & Commit Policy (SpectraMind V50)

This page clarifies **what lives in Git vs DVC vs is ignored** across the SpectraMind V50 pipeline.  
It pairs with our CLI-first + Hydra configs + DVC versioning strategy for NASA-grade reproducibility:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

> TL;DR  
> - **Git tracks**: code, Hydra YAMLs, `dvc.yaml`, `*.dvc`, shared `.dvc/config`, plots specs.  
> - **DVC tracks**: big/volatile artifacts (datasets, intermediates, model checkpoints).  
> - **Ignored**: machine-local cache/locks/state/logs under `.dvc/…` (never commit).

---

## Visual Map

```mermaid
flowchart LR
  subgraph GIT[Git-tracked (commit to repo)]
    A[src/** (code)]
    B[configs/** (Hydra YAMLs)]
    C[dvc.yaml (pipeline DAG)]
    D["*.dvc (pointers)"]
    E[".dvc/config (shared, no secrets)"]
    F[".dvc/plots/** (specs/templates)"]
  end

  subgraph DVC[DVC-tracked (stored in remote, referenced in Git)]
    G[data/** (raw, processed)]
    H[artifacts/** (intermediates)]
    I[models/** (checkpoints)]
    J[outputs/** (large reports)]
  end

  subgraph IGNORED[Ignored (machine-local, non-reproducible)]
    K[".dvc/cache/**"]
    L[".dvc/tmp/**  (locks, runs, exp cache)"]
    M[".dvc/state/**  (sqlite)"]
    N[".dvc/experiments/**  & .dvc/exp/**"]
    O[".dvc/logs/** & .dvc/events/**"]
    P[".dvc/stage.lock  (runtime locks)"]
    Q[".dvc/remotes/local/** (local-only mirrors)"]
    R[".dvc/plots/tmp/** (render cache)"]
    S[".dvc/config.local (secrets, dev overrides)"]
  end

  A -->|uses| C
  B -->|composed by| C
  C -->|produces pointers| D
  D -->|references| G & H & I & J

  K & L & M & N & O & P & Q & R & S -.->|never commit| IGNORED
````

**Legend**

* **Git-tracked**: versioned in repo → portable, reviews, CI.
* **DVC-tracked**: heavy data/model artifacts → referenced by `*.dvc` or `dvc.yaml`, fetched with `dvc pull`.
* **Ignored**: local, ephemeral, or secret-bearing paths under `.dvc/` → **must not** enter Git.

> GitHub renders Mermaid diagrams natively in Markdown; no extra scripts are needed.

---

## Why this split?

* **Reproducibility & Traceability**: Hydra captures run configs; DVC ties exact data/model snapshots to Git commits for full provenance.
* **Performance & CI**: Heavy artifacts live in remotes (not in Git), while CI and collaborators use `dvc pull` to retrieve required versions.
* **Safety**: `.dvc/config.local`, caches, locks, and experiments are machine-local and excluded to avoid leaking secrets/host specifics and to keep history clean.

---

## Commit Rules (Quick Checklist)

**Commit (YES):**

* `src/**`, `configs/**` (Hydra), `dvc.yaml`, `*.dvc`, `.dvc/config`, `.dvc/plots/**`.

**DVC-track (YES, don’t Git-add the binaries):**

* `data/**`, `artifacts/**`, `models/**`, large `outputs/**` → add via `dvc add` or declare as `outs` in `dvc.yaml`.

**Ignore (NEVER COMMIT):**

* `.dvc/cache/**`, `.dvc/tmp/**`, `.dvc/state/**`, `.dvc/experiments/**`, `.dvc/exp/**`, `.dvc/logs/**`, `.dvc/events/**`, `.dvc/stage.lock`, `.dvc/remotes/local/**`, `.dvc/plots/tmp/**`, `.dvc/config.local`.

---

## Typical Workflow

1. Define/extend stages in `dvc.yaml` (inputs/outs).
2. Run the pipeline via CLI (`spectramind …`) so Hydra logs the exact config and DVC updates outs.
3. `git add dvc.yaml *.dvc` + code/config changes → `git commit`.
4. `dvc push` to upload artifacts to the remote; teammates/CI run `dvc pull` to reproduce.

---

### References

* SpectraMind V50 Technical Plan — CLI/Hydra/DVC reproducibility & CI guardrails
* SpectraMind V50 Project Analysis — repository layout, Hydra configs, DVC usage
* Strategy Guide — DVC stage/caching guidance & config layering notes
* GitHub Mermaid support — render diagrams in Markdown

```

This gives folks a one-screen mental model and the exact do/don’t rules, consistent with our Hydra+DVC+CLI stack.
```
