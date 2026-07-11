"""Formal state-v2 rollout helpers."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .data_io import get_extras
from .schema_v2 import state_v2_from_info


def state_from_live_info(info: Dict[str, Any]) -> np.ndarray:
    state, _, _ = state_v2_from_info(info)
    return state


def pad_history(history: List[np.ndarray], th: int) -> np.ndarray:
    if th <= 0 or not history:
        raise ValueError("history must be non-empty and th positive")
    arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history]
    if any(item.size != arrays[-1].size for item in arrays):
        raise ValueError("history entries have inconsistent dimensions")
    selected = arrays[-int(th) :]
    if len(selected) < int(th):
        selected = [selected[0].copy() for _ in range(int(th) - len(selected))] + selected
    result = np.stack(selected, axis=0).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("history contains NaN or Inf")
    return result


def final_fraction_from_info(info: Dict[str, Any]) -> float:
    extras = get_extras(info)
    for key in ("task.final_fraction", "final_fraction", "task.fraction", "fraction"):
        if key in extras:
            return float(extras[key])
    if extras.get("nb_beads"):
        return float(extras.get("nb_zone", 0)) / float(extras["nb_beads"])
    return float("nan")
