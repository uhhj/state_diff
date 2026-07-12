from __future__ import annotations

import numpy as np

from ccda_phase3.phase314a_metrics import (
    final_valid_indices,
    group_bootstrap_mean_ci,
    masked_mse,
    trajectory_chamfer_rows,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


def test_final_valid_index_uses_last_true_horizon():
    mask = np.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    np.testing.assert_array_equal(
        final_valid_indices(mask),
        [0, 1, 3],
    )


def test_masked_mse_ignores_padded_errors():
    target = np.zeros((1, DEFAULT_TF, STATE_DIM), dtype=np.float32)
    prediction = target.copy()
    prediction[:, 2:] = 1000.0
    mask = np.array([[True, True, False, False]])
    assert masked_mse(prediction, target, mask) == 0.0


def test_trajectory_chamfer_zero_for_identical_future():
    value = np.zeros((2, DEFAULT_TF, STATE_DIM), dtype=np.float32)
    mask = np.ones((2, DEFAULT_TF), dtype=np.bool_)
    result = trajectory_chamfer_rows(value, value, mask)
    np.testing.assert_array_equal(result, np.zeros(2))


def test_group_bootstrap_uses_independent_groups():
    result = group_bootstrap_mean_ci(
        [1.0, 1.0, 3.0, 3.0],
        [10, 10, 20, 20],
        iterations=200,
        seed=1,
    )
    assert result["groups"] == 2
    assert result["mean"] == 2.0
    assert result["ci_low"] <= 2.0 <= result["ci_high"]
