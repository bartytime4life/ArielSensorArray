# Ensures pytest can import from the `src/` layout without external plugins.
from __future__ import annotations
import sys
from pathlib import Path

# repo_root/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
