Here are the upgraded dependency files, aligned with the V50 stack (Mamba SSM, GNNs, Hydra/Typer, DVC, SHAP/UMAP, calibration, and diagnostics). I’ve included pragmatic notes where installation depends on CUDA/torch wheels.

---

### `requirements.txt`

```text
# ===== Core numerical stack =====
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.2.0
pyarrow>=15.0.0

# ===== Deep learning =====
# Pin torch to your CUDA stack explicitly in production. Example:
#   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0

# Graph / GNN utilities (NetworkX used even when TG is absent)
networkx>=3.2.1

# Torch Geometric (install wheels matching your torch/CUDA version from https://pytorch-geometric.readthedocs.io/)
# Keep the metapackage here for resolver visibility; a proper install often requires extra wheel URLs.
torch-geometric>=2.5.0

# ===== Sequence / SSM (FGS1 encoder) =====
# Mamba SSM reference implementation. If your platform uses a different package, adjust accordingly.
mamba-ssm>=1.2.0

# ===== Config + CLI + logging =====
hydra-core>=1.3.2
omegaconf>=2.3.0
typer>=0.12.4
rich>=13.7.1

# ===== Uncertainty / calibration / stats =====
statsmodels>=0.14.2
scikit-learn>=1.4.0
# Optional: conformal prediction helpers (generic libs)
mapie==0.8.4

# ===== Explainability & visualization =====
shap>=0.44.0
umap-learn>=0.5.5
matplotlib>=3.8.0
plotly>=5.20.0

# ===== Experiment tracking (optional) =====
mlflow>=2.12.1

# ===== Data & artifact versioning =====
dvc>=3.50.0
# Choose appropriate remote(s); comment out the ones you don’t need.
dvc-s3>=3.0.1
dvc-gs>=3.0.0
dvc-gdrive>=3.0.0

# ===== Misc utilities =====
tqdm>=4.66.2
pyyaml>=6.0.1
```

---

### `requirements-dev.txt`

```text
# ===== Linters / formatters / typing =====
ruff>=0.6.9
black>=24.8.0
isort>=5.13.2
mypy>=1.11.0
types-PyYAML>=6.0.12.20240808

# ===== Testing =====
pytest>=8.1.0
pytest-cov>=5.0.0
hypothesis>=6.105.1

# ===== Pre-commit hooks =====
pre-commit>=3.7.1

# ===== Docs / notebooks (optional) =====
mkdocs>=1.6.0
mkdocs-material>=9.5.27
jupyterlab>=4.2.4
notebook>=7.2.2
```

**Notes & tips**

* **PyTorch + CUDA:** For reproducible installs, always pin the torch build to your CUDA version (e.g., `cu121`) using the official index URL. The plain `pip torch` above will default to CPU or whatever wheel PyPI offers.
* **torch-geometric:** Install wheels that match your exact PyTorch & CUDA versions (their docs provide one-liners). Keeping the requirement here helps resolvers, but your CI should install via the TG wheel URL for consistency.
* **mamba-ssm:** If your environment uses a different SSM package name or a compiled fork, adjust accordingly. The requirement above follows commonly used naming.
* **DVC remotes:** Keep only the remote backends you actually use (e.g., remove `dvc-gs` if you don’t use Google Cloud Storage).
* **Docs/notebooks:** The dev file includes MkDocs + Material and Jupyter; drop these if you want a slimmer dev image.
