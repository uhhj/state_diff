from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccda_phase3.phase314b_r256_stagea_contract import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    EXPECTED_ROWS,
    EXPECTED_TRAIN_ROWS,
    IDM_CABLE_TRAJECTORY_DIM,
    IDM_NEXT_CABLE_DIM,
    AuditSpec,
    StageAContractError,
    build_contract_arrays,
    deterministic_npz_bytes,
    deterministic_stratified_permutation,
    fit_masked_cable_standardizer,
    group_fold_assignment,
    paired_action_intervention_audit,
    run_train_only_idm_audit,
)


def synthetic_cache(rows=32):
    rng = np.random.RandomState(7)
    state_dim = 67
    paper = rng.randn(rows, DEFAULT_TH * state_dim).astype(np.float32)
    state_action = np.concatenate(
        [paper, rng.randn(rows, DEFAULT_TH * ACTION_DIM).astype(np.float32)],
        axis=1,
    )
    future = rng.randn(rows, DEFAULT_TF, state_dim).astype(np.float32)
    current = paper.reshape(rows, DEFAULT_TH, state_dim)[:, -1].copy()
    action = rng.randn(rows, ACTION_DIM).astype(np.float32)
    pair_key = np.asarray([f"train|{i // 2}|g{i // 2}|0" for i in range(rows)])
    # Paired rows share target action by construction.
    action[1::2] = action[0::2]
    episode_group = np.asarray([f"train|{i // 2}|g{i // 2}" for i in range(rows)])
    return {
        "paper_x": paper,
        "state_action_x": state_action,
        "y_state": future,
        "y_final_state": future[:, -1].copy(),
        "y_action": action,
        "future_valid_mask": np.ones((rows, DEFAULT_TF), dtype=np.bool_),
        "state_history_valid_mask":
            np.ones((rows, DEFAULT_TH), dtype=np.bool_),
        "action_history_valid_mask":
            np.ones((rows, DEFAULT_TH), dtype=np.bool_),
        "condition_name":
            np.asarray(["free", "hidden_slack_breakaway_pin_v2"] * (rows // 2)),
        "split_name": np.asarray(["train"] * rows),
        "visible_seed": np.arange(rows) // 2,
        "pair_group": np.asarray([f"g{i // 2}" for i in range(rows)]),
        "pair_key": pair_key,
        "episode_group_key": episode_group,
        "source_file": np.asarray([f"f{i}.pkl" for i in range(rows)]),
        "source_pickle_sha256": np.asarray(["0" * 64] * rows),
        "episode_index": np.arange(rows),
        "window_t": np.zeros(rows, dtype=np.int64),
        "pre_engagement": np.ones(rows, dtype=np.bool_),
        "current_state": current.astype(np.float32),
        "row_index": np.arange(rows, dtype=np.int64),
    }


def test_masked_cable_standardizer_shape():
    future = np.arange(10 * 4 * 48, dtype=np.float32).reshape(10, 4, 48)
    mask = np.ones((10, 4), dtype=np.bool_)
    value = fit_masked_cable_standardizer(future, mask)
    assert value.mean.shape == (4, 48)
    assert value.scale.shape == (4, 48)


def test_group_fold_assignment_keeps_groups_together():
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    assignment, mapping = group_fold_assignment(groups, folds=4)
    assert len(mapping) == 4
    for group in mapping:
        assert len(set(assignment[groups == group])) == 1


def test_deterministic_npz_bytes_are_exact():
    arrays = {"b": np.arange(5), "a": np.ones((2, 3), dtype=np.float32)}
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(arrays)


def test_permutation_preserves_condition_and_time_strata():
    features = np.arange(8 * 2).reshape(8, 2)
    condition = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    time = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.asarray(["g0", "g1", "g0", "g1", "g0", "g1", "g0", "g1"])
    shuffled, contract = deterministic_stratified_permutation(
        features,
        condition_name=condition,
        window_t=time,
        episode_group_key=groups,
        shift=1,
    )
    assert shuffled.shape == features.shape
    assert contract["condition_preserved"]
    assert contract["window_t_preserved"]
    assert not np.array_equal(shuffled, features)


def test_paired_action_audit_detects_no_intervention():
    action = np.asarray([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    cable = np.arange(4 * 4 * 3).reshape(4, 4, 3)
    result = paired_action_intervention_audit(
        ["a", "a", "b", "b"],
        action,
        cable,
        epsilon=1.0e-8,
    )
    assert result["paired_action_diverse_fraction"] == 0.0
    assert result["same_action_pair_contract"]


def test_paired_action_audit_detects_intervention():
    action = np.asarray([[1.0], [2.0], [3.0], [3.0]])
    cable = np.arange(4 * 4 * 3).reshape(4, 4, 3)
    result = paired_action_intervention_audit(
        ["a", "a", "b", "b"],
        action,
        cable,
        epsilon=1.0e-8,
    )
    assert result["paired_action_diverse_keys"] == 1


def test_audit_classifies_missing_paired_action_intervention():
    rows = 64
    rng = np.random.RandomState(3)
    history = rng.randn(rows, 8)
    action = np.zeros((rows, ACTION_DIM))
    action[:, 0] = history[:, 0]
    action[1::2] = action[0::2]
    cable = rng.randn(rows, 4, 48)
    next_feature = np.concatenate([history, cable[:, 0]], axis=1)
    trajectory = np.concatenate([history, cable.reshape(rows, -1)], axis=1)
    arrays = {
        "action_target": action.astype(np.float32),
        "episode_group_key":
            np.asarray([f"g{i // 2}" for i in range(rows)]),
        "condition_name":
            np.asarray(["free", "hidden_slack_breakaway_pin_v2"] * (rows // 2)),
        "window_t": np.zeros(rows, dtype=np.int64),
        "pair_key": np.asarray([f"p{i // 2}" for i in range(rows)]),
        "diffusion_target_cable": cable.astype(np.float32),
        "idm_history_x": history.astype(np.float32),
        "idm_next_cable_x": next_feature.astype(np.float32),
        "idm_next_cable_delta_x": next_feature.astype(np.float32),
        "idm_cable_trajectory_x": trajectory.astype(np.float32),
        "idm_cable_delta_trajectory_x": trajectory.astype(np.float32),
    }
    result = run_train_only_idm_audit(
        arrays,
        spec=AuditSpec(
            folds=8,
            minimum_unique_action_vectors=8,
            minimum_active_action_dims=1,
        ),
    )
    assert not result["formal_idm_data_ready"]
    assert "paired_action_intervention_missing" in result["root_cause"]


def test_contract_dimensions_are_cable_only():
    assert CABLE_DIM == 48
    assert IDM_NEXT_CABLE_DIM == 291
    assert IDM_CABLE_TRAJECTORY_DIM == 435


def test_object_dtype_rejected_by_deterministic_writer():
    with pytest.raises(ValueError):
        deterministic_npz_bytes({"bad": np.asarray([object()], dtype=object)})
