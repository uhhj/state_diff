from __future__ import annotations

import numpy as np

from ccda_phase3.phase314b_r21_geometry import (
    aggregate_geometry_rows,
    calibrated_validity,
    candidate_geometry_rows,
    fit_calibrated_geometry_contract,
    ordered_xy,
    segment_lengths,
)
from ccda_phase3.schema_v2 import STATE_DIM


def make_future(
    *,
    y_offset: float = 0.0,
    noise: float = 0.0,
) -> np.ndarray:
    state = np.zeros((4, STATE_DIM), dtype=np.float32)
    xy = np.zeros((4, 24, 2), dtype=np.float32)
    base_x = np.linspace(0.30, 0.53, 24, dtype=np.float32)
    for horizon in range(4):
        xy[horizon, :, 0] = base_x
        xy[horizon, :, 1] = (
            y_offset + 0.002 * horizon
        )
    if noise:
        rng = np.random.default_rng(7)
        xy += rng.normal(0.0, noise, size=xy.shape).astype(
            np.float32
        )
    state[:, :48] = xy.reshape(4, -1)
    state[:, 83:87] = [0.0, 0.0, 0.0, 1.0]
    return state


def test_calibrated_contract_accepts_train_like_future():
    fit = np.stack(
        [
            make_future(y_offset=value, noise=1e-4)
            for value in np.linspace(-0.02, 0.02, 40)
        ],
        axis=0,
    )
    calibration = np.stack(
        [
            make_future(y_offset=value, noise=1e-4)
            for value in np.linspace(-0.019, 0.019, 40)
        ],
        axis=0,
    )
    contract = fit_calibrated_geometry_contract(
        fit_future=fit,
        calibration_future=calibration,
    )
    result = calibrated_validity(
        calibration[None, ...],
        contract,
    )
    assert result["sample_validity_rate"] >= 0.90


def test_calibrated_contract_rejects_gross_edge_stretch():
    fit = np.stack(
        [make_future(y_offset=value) for value in np.linspace(-0.02, 0.02, 40)],
        axis=0,
    )
    contract = fit_calibrated_geometry_contract(
        fit_future=fit[:20],
        calibration_future=fit[20:],
    )
    bad = fit[:2].copy()
    bad_xy = ordered_xy(bad).copy()
    bad_xy[:, :, 12, 0] += 0.5
    bad[..., :48] = bad_xy.reshape(2, 4, 48)
    result = calibrated_validity(
        bad[None, ...],
        contract,
    )
    assert result["sample_validity_rate"] == 0.0


def test_point_set_chamfer_can_hide_order_failure():
    target = make_future()[None, ...]
    prediction = target.copy()
    xy = ordered_xy(prediction)
    permutation = np.array(
        list(range(0, 24, 2)) + list(range(1, 24, 2))
    )
    xy[:] = xy[..., permutation, :]
    prediction[..., :48] = xy.reshape(
        prediction.shape[:-1] + (48,)
    )
    rows = candidate_geometry_rows(
        sample_pool=prediction[None, ...],
        target_future=target,
        row_indices=np.array([10]),
        pair_keys=np.array(["val|1|0"]),
        conditions=np.array(["free"]),
        visible_seeds=np.array([1]),
        segment_center=segment_lengths(target)[0],
    )
    aggregate = aggregate_geometry_rows(rows)
    assert aggregate["final_chamfer_mean"] < 1e-6
    assert aggregate["final_ordered_rmse_mean"] > 0.01
    assert aggregate["permutation_gap_mean"] > 0.01
