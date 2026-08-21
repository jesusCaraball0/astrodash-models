"""Put sibling training packages on ``sys.path`` without editing them."""

from __future__ import annotations

import sys
from pathlib import Path

NORM_FLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NORM_FLOW_DIR.parent
WISEREP_DIR = PROJECT_ROOT / "WiserepData"
ZMODEL_DIR = PROJECT_ROOT / "zmodel_training"


def setup_imports() -> Path:
    for path in (NORM_FLOW_DIR, PROJECT_ROOT, WISEREP_DIR, ZMODEL_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return PROJECT_ROOT
