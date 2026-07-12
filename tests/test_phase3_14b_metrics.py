from __future__ import annotations

import numpy as np

from ccda_phase3.phase314b_metrics import (
    branch_support_for_pair,
    candidate_final_chamfer,
    nested_best_of_k,
)
from ccda_phase3.schema_v2 import STATE_DIM


def state_with_offset(offset: float) -> np.ndarray:
    state = np.zeros(STATE_DIM, dtype=np.float32)
    xy = np.zeros((24, 2), dtype=np.float32)
    xy[:, 0] = np.linspace(0.0, 0.23, 24)
    xy[:, 1] = float(offset)
    state[:48] = xy.reshape(-1)
    state[83:87] = [0.0, 0.0, 0.0, 1.0]
    return state


def future_with_offset(offset: float) -> np.ndarray:
    return np.repeat(
        state_with_offset(offset)[None, :],
        4,
        axis=0,
    )


def test_nested_best_of_k_is_monotonic():
    target = future_with_offset(0.0)[None, ...]
    samples = np.stack(
        [
            target + 0.03,
            target + 0.02,
            target + 0.01,
            target,
        ],
        axis=0,
    )
    distance = candidate_final_chamfer(samples, target)
    nested = nested_best_of_k(distance, (1, 2, 4))
    assert nested[1][0] >= nested[2][0] >= nested[4][0]
    assert nested[4][0] == 0.0


def test_branch_support_detects_two_modes():
    free = state_with_offset(0.0)
    hidden = state_with_offset(0.02)
    samples = np.stack([free, hidden], axis=0)
    result = branch_support_for_pair(
        samples,
        free_final=free,
        hidden_final=hidden,
    )
    assert result["eligible"]
    assert result["free_supported"]
    assert result["hidden_supported"]
    assert result["both_supported"]
