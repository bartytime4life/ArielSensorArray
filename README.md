[![CI](https://github.com/bartytime4life/ArielSensorArray/actions/workflows/ci.yml/badge.svg)](https://github.com/bartytime4life/ArielSensorArray/actions/workflows/ci.yml) [![Release](https://github.com/bartytime4life/ArielSensorArray/actions/workflows/release.yml/badge.svg)](https://github.com/bartytime4life/ArielSensorArray/actions/workflows/release.yml)

# ArielSensorArray — NeurIPS 2025 Ariel Data Challenge

Mission-grade, **reproducible** scaffold for Ariel’s sensor array challenge:
- Inputs: FGS1 & AIRS time-series frames
- Outputs: mean (μ) and uncertainty (σ) for 283 spectral bins
- Stack: Hydra configs, Typer CLI, DVC pipeline, CI, pre-commit

## Quickstart
```bash
# one-time
pipx install poetry
poetry install
poetry run pre-commit install

# sanity
poetry run asa --help
poetry run asa selftest --dry-run
```
