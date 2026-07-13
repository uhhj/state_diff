from __future__ import annotations

import numpy as np
import torch

from ccda_phase3.phase314b_r231_controls import (
    EXACT_REPLAY_GATE,
    EXPECTED_CONDITIONS,
    classify_controls,
    gate_reconstruction,
    make_noise,
    pair_ambiguity_metrics,
    select_unique_condition_rows,
    target_reconstruction_metrics,
)


def make_arrays(seed_count: int = 20):
    conditions = np.asarray(EXPECTED_CONDITIONS)
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
        "split_name": np.full(seed_count * len(conditions), "train"),
    }


def test_select_unique_condition_rows_uses_one_seed_each() -> None:
    arrays = make_arrays(20)
    candidates = np.arange(40, dtype=np.int64)
    rows = select_unique_condition_rows(
        arrays,
        candidates,
        condition="free",
        row_count=16,
    )
    assert rows.shape == (16,)
    assert len(set(arrays["visible_seed"][rows].tolist())) == 16
    assert set(arrays["condition_name"][rows].tolist()) == {"free"}


def test_target_reconstruction_metrics_exact_prediction_passes() -> None:
    target_z = np.zeros((3, 4, 87), dtype=np.float32)
    target_raw = np.zeros((3, 4, 87), dtype=np.float32)
    # Give the cable nonzero, regular edges.
    for bead in range(24):
        target_raw[:, :, 2 * bead] = 0.01 * bead
    active = np.ones((4, 87), dtype=bool)

    metrics = target_reconstruction_metrics(
        predicted_z=target_z.copy(),
        target_z=target_z,
        predicted_raw=target_raw.copy(),
        target_raw=target_raw,
        active_mask=active,
    )
    assert metrics["z_mse"] == 0.0
    assert metrics["ordered_rmse"]["max"] == 0.0
    assert metrics["segment_relative_error"]["max"] == 0.0
    assert metrics["chain_relative_error"]["max"] == 0.0
    assert gate_reconstruction(metrics, EXACT_REPLAY_GATE)


def test_make_noise_distinguishes_training_and_fresh_seed() -> None:
    clean = torch.zeros((2, 4, 87), dtype=torch.float32)
    active = torch.ones((4, 87), dtype=torch.bool)
    training = make_noise(clean, active, seed=100)
    replay = make_noise(clean, active, seed=100)
    fresh = make_noise(clean, active, seed=101)

    assert torch.equal(training, replay)
    assert not torch.equal(training, fresh)


def test_pair_ambiguity_metrics_detects_equal_inputs_divergent_targets() -> None:
    arrays = make_arrays(2)
    rows = np.arange(4, dtype=np.int64)
    condition = np.zeros((4, 261), dtype=np.float32)
    target = np.zeros((4, 4, 87), dtype=np.float32)
    target[1, :, :48] = 0.02
    target[3, :, :48] = 0.03

    metrics = pair_ambiguity_metrics(arrays, rows, condition, target)
    assert metrics["pair_count"] == 2
    assert metrics["exact_input_equal_fraction"] == 1.0
    assert metrics["input_max_abs"]["max"] == 0.0
    assert metrics["target_ordered_rmse"]["median"] > 0.0


def _run(kind: str, passed: bool):
    if kind == "direct":
        return {"gate_pass": passed}
    if kind == "fixed":
        return {"exact_replay_gate_pass": passed}
    if kind == "random":
        return {"single_branch_t50_gate_pass": passed}
    raise ValueError(kind)


def test_classification_separates_paired_ambiguity_from_capacity() -> None:
    report = {
        "runs": {
            "direct_one_row": _run("direct", True),
            "direct_unique_free_16": _run("direct", True),
            "fixed_one_row_v_only": _run("fixed", True),
            "fixed_pairs_v_only_adamw": _run("fixed", True),
            "fixed_pairs_v_only_adam": _run("fixed", True),
            "fixed_pairs_v_only_wide": _run("fixed", True),
            "fixed_pairs_r22_geometry": _run("fixed", True),
            "random_one_row": _run("random", True),
            "random_unique_free_16": _run("random", True),
            "random_paired_16": _run("random", False),
        },
        "pair_ambiguity": {
            "input_max_abs": {"median": 0.0},
            "target_ordered_rmse": {"median": 0.03},
        },
    }
    result = classify_controls(report)
    assert result["root_cause"] == (
        "phase314b_r231_paired_branch_ambiguity_not_capacity_supported"
    )


def test_classification_stops_at_direct_one_row_failure() -> None:
    report = {
        "runs": {
            "direct_one_row": _run("direct", False),
        },
        "pair_ambiguity": {},
    }
    result = classify_controls(report)
    assert result["root_cause"] == (
        "phase314b_r231_direct_one_row_optimizer_or_capacity_failure"
    )
