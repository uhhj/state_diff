"""Explicit-state comparison helpers for Soft BlockPush pairing."""
from __future__ import annotations

from typing import Dict

import numpy as np


def capture_explicit_state(env) -> Dict[str, np.ndarray]:
    """Capture one environment explicit-state dictionary."""
    return env.capture_explicit_state()


def max_state_difference(first: Dict[str, np.ndarray],
                         second: Dict[str, np.ndarray]) -> float:
    """Return maximum absolute numeric difference across equal state schemas."""
    if set(first) != set(second):
        raise ValueError("explicit state keys differ")
    maximum = 0.0
    for key in sorted(first):
        left, right = np.asarray(first[key]), np.asarray(second[key])
        if left.shape != right.shape:
            raise ValueError("state shape differs for {}".format(key))
        if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
            if not np.array_equal(left, right):
                return float("inf")
        elif left.size:
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    return maximum


def save_explicit_state(path: str, state: Dict[str, np.ndarray]) -> None:
    """Save an explicit snapshot as a compressed NPZ."""
    np.savez_compressed(path, **state)
