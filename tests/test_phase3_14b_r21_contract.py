from __future__ import annotations

import numpy as np

from ccda_phase3.phase314b_r21_contract import (
    conformal_quantile,
    train_visible_seed_partition,
)


def test_conformal_quantile_is_finite_and_monotonic():
    values = np.arange(100, dtype=np.float64)
    q1 = conformal_quantile(values, alpha=0.10)
    q2 = conformal_quantile(values, alpha=0.01)
    assert np.isfinite(q1)
    assert np.isfinite(q2)
    assert q2 >= q1


def test_train_partition_is_visible_seed_disjoint():
    train_rows = np.arange(40, dtype=np.int64)
    arrays = {
        "visible_seed": np.repeat(
            np.arange(20, 40, dtype=np.int64),
            2,
        )
    }
    fit, calibration = train_visible_seed_partition(
        arrays,
        train_rows,
    )
    fit_seeds = set(arrays["visible_seed"][fit].tolist())
    calibration_seeds = set(
        arrays["visible_seed"][calibration].tolist()
    )
    assert not fit_seeds.intersection(calibration_seeds)
    assert set(np.concatenate([fit, calibration]).tolist()) == set(
        train_rows.tolist()
    )
