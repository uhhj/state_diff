"""Small JSON/NPZ helpers for Phase 0C."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def clean_json(value: Any):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
