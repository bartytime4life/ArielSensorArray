from pathlib import Path

# Centralized project paths (override with CLI flags if desired)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"

for p in (DATA, RAW, PROCESSED, MODELS, OUTPUTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)