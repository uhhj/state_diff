from __future__ import annotations

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r21_geometry import CalibratedGeometryContract
from ccda_phase3.phase314b_r22_geometry import GeometryNormalizers
from ccda_phase3.phase314b_r23_diagnostics import (
    EXPECTED_PAIRED_CONDITIONS,
    classify_diagnosis,
    diagnostic_tail_components,
    failure_decomposition,
    paired_selection_metadata,
    select_balanced_paired_rows,
)


def make_future(spacing: float = 0.01) -> np.ndarray:
    future = np.zeros((1, 4, 87), dtype=np.float32)
    x = np.arange(24, dtype=np.float32) * spacing
    future[:, :, :48] = np.stack(
        [x, np.zeros_like(x)],
        axis=-1,
    ).reshape(1, 1, 48)
    future[:, :, 83:87] = np.asarray([0, 0, 0, 1], dtype=np.float32)
    return future


def make_contract() -> CalibratedGeometryContract:
    return CalibratedGeometryContract(
        coordinate_lower=np.asarray([-1.0, -1.0], dtype=np.float32),
        coordinate_upper=np.asarray([1.0, 1.0], dtype=np.float32),
        segment_center=np.full((4, 23), 0.01, dtype=np.float32),
        segment_scale=np.full((4, 23), 0.001, dtype=np.float32),
        segment_score_threshold=3.0,
        chain_center=np.full((4,), 0.23, dtype=np.float32),
        chain_scale=np.full((4,), 0.005, dtype=np.float32),
        chain_score_threshold=3.0,
    )


def test_failure_decomposition_detects_single_stretched_edge() -> None:
    target = make_future()
    prediction = target.copy()
    xy = prediction[:, :, :48].reshape(1, 4, 24, 2)
    xy[:, :, 12:, 0] += 0.05
    prediction[:, :, :48] = xy.reshape(1, 4, 48)

    valid = failure_decomposition(target[None], target, make_contract())
    invalid = failure_decomposition(prediction[None], target, make_contract())

    assert valid["sample_validity_rate"] == 1.0
    assert invalid["sample_validity_rate"] == 0.0
    assert invalid["segment_family_failure_rate"] == 1.0
    assert invalid["bad_segment_constraint_count_p50"] == 4.0
    assert invalid["gross_stretch_fraction_center"] > 0.0


def test_tail_component_responds_to_sparse_outlier() -> None:
    target = torch.from_numpy(make_future())
    prediction = target.clone()
    prediction[:, :, 24:48] += 0.0
    xy = prediction[:, :, :48].reshape(1, 4, 24, 2)
    xy[:, :, 12:, 0] += 0.05
    prediction[:, :, :48] = xy.reshape(1, 4, 48)
    normalizers = GeometryNormalizers(
        ordered_xy_scale=0.23,
        edge_vector_scale=0.01,
        segment_length_scale=0.01,
        chain_length_scale=0.23,
        temporal_edge_scale=0.01,
    )
    exact = diagnostic_tail_components(target, target, normalizers)
    outlier = diagnostic_tail_components(prediction, target, normalizers)
    assert float(exact["tail_segment"].max()) == 0.0
    assert float(outlier["tail_segment"].min()) > 0.0
    assert float(outlier["tail_ordered"].min()) > 0.0


def test_classifier_prefers_scale_failure_when_tail_control_overfits() -> None:
    checkpoint = {
        "scheduler_parity_max_abs": 0.0,
        "gradient_rows": [
            {
                "config": config,
                "component": "ordered_xy",
                "zero_gradient": False,
                "weighted_to_v_gradient_ratio": 1e-4,
            }
            for config in ("ordered_edge", "ordered_edge_temporal")
        ]
        + [
            {
                "config": "ordered_edge",
                "component": component,
                "zero_gradient": False,
                "weighted_to_v_gradient_ratio": 0.1,
            }
            for component in (
                "edge_vector",
                "segment_length",
                "chain_length",
            )
        ],
        "familywise_tail_signature": {"supported": True},
        "reverse_accumulation_signature": {"supported": False},
    }
    tiny = {
        "runs": {
            "r22_exact_fixed": {"gate_pass": False},
            "tail_control_fixed": {"gate_pass": True},
            "r22_exact_random": {"gate_pass": False},
            "tail_control_random": {"gate_pass": True},
        }
    }
    result = classify_diagnosis(checkpoint, tiny)
    assert result["root_cause"] == (
        "phase314b_r23_ordered_scale_and_mean_reduction_failure_supported"
    )


def make_paired_selection_arrays(seed_count: int = 51):
    conditions = np.asarray(EXPECTED_PAIRED_CONDITIONS)
    return {
        "visible_seed": np.repeat(
            np.arange(seed_count, dtype=np.int64),
            len(conditions),
        ),
        "condition_name": np.tile(conditions, seed_count),
        "pair_key": np.repeat(
            np.asarray([f"pair-{i:03d}" for i in range(seed_count)]),
            len(conditions),
        ),
        "split_name": np.full(
            seed_count * len(conditions),
            "val",
        ),
    }


def test_select_balanced_paired_rows_treats_64_as_rows() -> None:
    arrays = make_paired_selection_arrays(51)
    candidates = np.arange(102, dtype=np.int64)

    selected = select_balanced_paired_rows(
        arrays,
        candidates,
        64,
    )
    metadata = paired_selection_metadata(arrays, selected)

    assert len(selected) == 64
    assert metadata["complete_pair_count"] == 32
    assert metadata["distinct_visible_seed_count"] == 32
    assert metadata["distinct_pair_key_count"] == 32
    assert metadata["condition_counts"] == {
        "free": 32,
        "hidden_slack_breakaway_pin_v2": 32,
    }


def test_select_balanced_paired_rows_supports_trace_subset() -> None:
    arrays = make_paired_selection_arrays(51)
    candidates = np.arange(102, dtype=np.int64)

    validation = select_balanced_paired_rows(
        arrays,
        candidates,
        64,
    )
    trace = select_balanced_paired_rows(
        arrays,
        validation,
        16,
    )
    metadata = paired_selection_metadata(arrays, trace)

    assert len(trace) == 16
    assert metadata["complete_pair_count"] == 8
    assert metadata["distinct_visible_seed_count"] == 8


def test_select_balanced_paired_rows_rejects_odd_count() -> None:
    arrays = make_paired_selection_arrays(51)
    candidates = np.arange(102, dtype=np.int64)

    with pytest.raises(ValueError, match="divisible"):
        select_balanced_paired_rows(arrays, candidates, 63)


def test_select_balanced_paired_rows_rejects_insufficient_pairs() -> None:
    arrays = make_paired_selection_arrays(10)
    candidates = np.arange(20, dtype=np.int64)

    with pytest.raises(RuntimeError, match="not enough complete"):
        select_balanced_paired_rows(arrays, candidates, 22)


def test_select_balanced_paired_rows_uses_one_pair_per_seed() -> None:
    arrays = make_paired_selection_arrays(4)

    arrays = {
        key: np.concatenate(
            [
                value,
                (
                    np.asarray([0, 0], dtype=np.int64)
                    if key == "visible_seed"
                    else np.asarray(EXPECTED_PAIRED_CONDITIONS)
                    if key == "condition_name"
                    else np.asarray(["pair-extra", "pair-extra"])
                    if key == "pair_key"
                    else np.asarray(["train", "train"])
                ),
            ]
        )
        for key, value in arrays.items()
    }

    candidates = np.arange(len(arrays["visible_seed"]), dtype=np.int64)
    selected = select_balanced_paired_rows(
        arrays,
        candidates,
        8,
    )

    assert len(set(arrays["visible_seed"][selected].tolist())) == 4
