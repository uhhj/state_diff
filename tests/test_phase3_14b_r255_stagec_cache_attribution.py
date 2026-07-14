from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3.phase314b_r255_stagec_attribution import (
    AttributionSpec,
    action_prediction_probe,
    classify_attribution,
    constant_velocity_baseline,
    group_fold_assignment,
    grouped_oof_future_robot,
    grouped_oof_regression,
    robot_prediction_metrics,
    run_train_only_attribution,
    schema_audit,
)
from ccda_phase3.phase314b_r255_stagec_cache import (
    EXPECTED_PAIR_KEYS,
    EXPECTED_ROWS,
    EXPECTED_TRAIN_ROWS,
    StageCCacheError,
    build_cache_arrays,
    deterministic_npz_bytes,
    fit_masked_future_standardizer,
    fit_standardizer,
    load_npz_strict,
    make_pair_key,
    validate_stage_b_windows,
)
from ccda_phase3.schema_v3 import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    PAPER_X_DIM,
    ROBOT_EE_QUATERNION_SLICE,
    ROBOT_PROXY_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
)


def _strings(values):
    width = max(1, max(len(str(value)) for value in values))
    return np.asarray([str(value) for value in values], dtype=f"<U{width}")


def make_full_windows() -> dict:
    rows = EXPECTED_ROWS
    pair_counts = {"train": 1220, "val": 454, "test": 454}
    split = []
    condition = []
    seed = []
    group = []
    source = []
    episode = []
    pair_index = 0
    row_index = 0
    for split_name, count in pair_counts.items():
        for local_pair in range(count):
            visible_seed = 100000 + pair_index
            pair_group = f"{split_name}_pair_{local_pair:04d}"
            for condition_name in ("free", "hidden_slack_breakaway_pin_v2"):
                split.append(split_name)
                condition.append(condition_name)
                seed.append(visible_seed)
                group.append(pair_group)
                source.append(
                    f"raw/{split_name}/{condition_name}/{visible_seed:06d}.pkl"
                )
                episode.append(row_index)
                row_index += 1
            pair_index += 1
    assert row_index == rows
    assert pair_index == EXPECTED_PAIR_KEYS

    row = np.arange(rows, dtype=np.float32)[:, None, None]
    time = np.arange(DEFAULT_TH, dtype=np.float32)[None, :, None]
    history = np.zeros((rows, DEFAULT_TH, STATE_DIM), dtype=np.float32)
    history[..., :CABLE_DIM] = (
        row / 1000.0
        + time / 100.0
        + np.arange(CABLE_DIM, dtype=np.float32)[None, None] / 10000.0
    )
    history[..., CABLE_DIM : CABLE_DIM + 12] = (
        row / 2000.0
        + time / 50.0
        + np.arange(12, dtype=np.float32)[None, None] / 1000.0
    )
    history[..., CABLE_DIM + 12 : CABLE_DIM + 15] = (
        row / 3000.0 + time / 70.0
    )
    history[..., -4:] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    horizon = np.arange(DEFAULT_TF, dtype=np.float32)[None, :, None]
    future = np.repeat(history[:, -1:, :], DEFAULT_TF, axis=1)
    future[..., :CABLE_DIM] += (horizon + 1.0) / 500.0
    future[..., CABLE_DIM : CABLE_DIM + 12] += (horizon + 1.0) / 800.0
    future[..., CABLE_DIM + 12 : CABLE_DIM + 15] += (horizon + 1.0) / 900.0
    future[..., -4:] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    action_history = np.zeros((rows, DEFAULT_TH, ACTION_DIM), dtype=np.float32)
    pair_value = (np.arange(rows, dtype=np.float32) // 2.0)
    action_history[..., 0] = pair_value[:, None] / 100.0
    paper = history.reshape(rows, PAPER_X_DIM)
    state_action = np.concatenate(
        [paper, action_history.reshape(rows, -1)], axis=1
    ).astype(np.float32)
    action = np.zeros((rows, ACTION_DIM), dtype=np.float32)
    action[:, 0] = pair_value / 100.0
    action[:, 1] = (pair_value % 2).astype(np.float32)
    arrays = {
        "paper_x": paper.astype(np.float32),
        "state_action_x": state_action,
        "y_state": future.astype(np.float32),
        "y_final_state": future[:, -1].astype(np.float32),
        "y_action": action,
        "state_history_valid_mask": np.ones(
            (rows, DEFAULT_TH), dtype=np.bool_
        ),
        "future_valid_mask": np.ones(
            (rows, DEFAULT_TF), dtype=np.bool_
        ),
        "action_history_valid_mask": np.ones(
            (rows, DEFAULT_TH), dtype=np.bool_
        ),
        "condition_name": _strings(condition),
        "visible_seed": np.asarray(seed, dtype=np.int64),
        "split_name": _strings(split),
        "source_file": _strings(source),
        "source_pickle_sha256": _strings(
            [f"{index:064x}" for index in range(rows)]
        ),
        "pair_group": _strings(group),
        "episode_index": np.asarray(episode, dtype=np.int64),
        "window_t": np.zeros(rows, dtype=np.int64),
        "success": np.asarray(
            [name == "free" for name in condition], dtype=np.bool_
        ),
        "final_fraction": np.asarray(
            [1.0 if name == "free" else 0.0 for name in condition],
            dtype=np.float32,
        ),
        "engagement_step": np.ones(rows, dtype=np.int64),
        "release_step": np.full(rows, 2, dtype=np.int64),
        "pre_engagement": np.zeros(rows, dtype=np.bool_),
    }
    return arrays


def make_small_train_view(rows: int = 64) -> dict:
    if rows % 2:
        raise ValueError("rows must be even")
    rng = np.random.RandomState(123)
    groups = np.repeat(np.arange(rows // 2), 2)
    history = np.zeros((rows, DEFAULT_TH, STATE_DIM), dtype=np.float32)
    latent = rng.normal(size=(rows, 2)).astype(np.float32)
    history[:, :, 0] = latent[:, :1]
    history[:, :, CABLE_DIM] = latent[:, 1:2]
    history[..., -4:] = np.asarray([0, 0, 0, 1], dtype=np.float32)
    action_history = np.zeros((rows, DEFAULT_TH, ACTION_DIM), dtype=np.float32)
    state_action = np.concatenate(
        [history.reshape(rows, -1), action_history.reshape(rows, -1)], axis=1
    ).astype(np.float32)
    target_action = rng.normal(size=(rows, ACTION_DIM)).astype(np.float32)
    future = np.repeat(history[:, -1:, :], DEFAULT_TF, axis=1)
    for horizon in range(DEFAULT_TF):
        future[:, horizon, CABLE_DIM : CABLE_DIM + 14] = (
            target_action * float(horizon + 1)
        )
    future[..., -4:] = np.asarray([0, 0, 0, 1], dtype=np.float32)
    return {
        "paper_x": history.reshape(rows, -1).astype(np.float32),
        "state_action_x": state_action,
        "y_state": future.astype(np.float32),
        "y_action": target_action,
        "future_valid_mask": np.ones((rows, DEFAULT_TF), dtype=np.bool_),
        "state_history_valid_mask": np.ones((rows, DEFAULT_TH), dtype=np.bool_),
        "action_history_valid_mask": np.ones((rows, DEFAULT_TH), dtype=np.bool_),
        "condition_name": _strings(
            ["free" if index % 2 == 0 else "hidden_slack_breakaway_pin_v2" for index in range(rows)]
        ),
        "visible_seed": groups.astype(np.int64),
        "pair_group": _strings([f"pair_{value}" for value in groups]),
        "source_file": _strings([f"train/{index}.pkl" for index in range(rows)]),
        "episode_index": np.arange(rows, dtype=np.int64),
        "window_t": np.zeros(rows, dtype=np.int64),
        "pre_engagement": np.zeros(rows, dtype=np.bool_),
        "source_row_index": np.arange(rows, dtype=np.int64),
        "pair_key": _strings([f"train|{value}|pair_{value}|0" for value in groups]),
        "episode_group_key": _strings([f"train|{value}|pair_{value}" for value in groups]),
        "robot_history": history[..., CABLE_DIM:].astype(np.float32),
        "robot_future": future[..., CABLE_DIM:].astype(np.float32),
        "split_name": _strings(["train"] * rows),
    }


def test_fit_standardizer_marks_constant_dimensions_inactive():
    value = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]], dtype=np.float32)
    result = fit_standardizer(value)
    assert result.active.tolist() == [False, True]
    assert result.scale[0] == 1.0
    assert result.mean.tolist() == pytest.approx([1.0, 4.0])


def test_masked_future_standardizer_ignores_padded_rows():
    future = np.zeros((3, DEFAULT_TF, STATE_DIM), dtype=np.float32)
    mask = np.ones((3, DEFAULT_TF), dtype=np.bool_)
    future[:, :, 0] = np.asarray([1.0, 2.0, 3.0])[:, None]
    mask[2, 3] = False
    future[2, 3, 0] = 10000.0
    result = fit_masked_future_standardizer(future, mask)
    assert result.mean[3, 0] == pytest.approx(1.5)


def test_deterministic_npz_bytes_are_byte_exact():
    arrays = {
        "b": np.arange(5, dtype=np.int64),
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
    }
    assert deterministic_npz_bytes(arrays) == deterministic_npz_bytes(dict(reversed(list(arrays.items()))))


def test_deterministic_npz_rejects_object_arrays():
    with pytest.raises(ValueError):
        deterministic_npz_bytes({"bad": np.asarray([object()], dtype=object)})


def test_load_npz_strict_round_trip(tmp_path: Path):
    path = tmp_path / "value.npz"
    path.write_bytes(deterministic_npz_bytes({"x": np.arange(4, dtype=np.float32)}))
    loaded = load_npz_strict(path)
    np.testing.assert_array_equal(loaded["x"], np.arange(4, dtype=np.float32))


def test_validate_stage_b_windows_accepts_full_contract():
    result = validate_stage_b_windows(make_full_windows())
    assert result["rows"] == EXPECTED_ROWS
    assert result["train_rows"] == EXPECTED_TRAIN_ROWS
    assert result["pair_keys"] == EXPECTED_PAIR_KEYS


def test_build_cache_arrays_has_state_v3_dimensions():
    cache, train, _ = build_cache_arrays(make_full_windows())
    assert cache["current_state"].shape == (EXPECTED_ROWS, STATE_DIM)
    assert cache["idm_pair_x"].shape == (EXPECTED_ROWS, STATE_DIM * 2)
    assert cache["idm_trajectory_x"].shape == (
        EXPECTED_ROWS,
        PAPER_X_DIM + DEFAULT_TF * STATE_DIM,
    )
    assert train["paper_x"].shape == (EXPECTED_TRAIN_ROWS, PAPER_X_DIM)


def test_pair_key_includes_pair_group_and_window_time():
    arrays = make_full_windows()
    keys = make_pair_key(arrays)
    assert arrays["pair_group"][0] in keys[0]
    assert keys[0].endswith("|0")
    assert keys[0] == keys[1]


def test_train_view_contains_only_train_rows():
    _, train, _ = build_cache_arrays(make_full_windows())
    assert set(train["split_name"].astype(str).tolist()) == {"train"}
    assert train["source_row_index"].shape == (EXPECTED_TRAIN_ROWS,)


def test_standardizers_are_fitted_from_train_rows_only():
    original = make_full_windows()
    changed = {key: value.copy() for key, value in original.items()}
    validation = changed["split_name"].astype(str) != "train"
    changed["paper_x"][validation, 0] += 1000000.0
    changed["state_action_x"][validation, 0] += 1000000.0
    cache_a, _, _ = build_cache_arrays(original)
    cache_b, _, _ = build_cache_arrays(changed)
    np.testing.assert_array_equal(cache_a["paper_x_mean"], cache_b["paper_x_mean"])
    np.testing.assert_array_equal(cache_a["state_action_x_mean"], cache_b["state_action_x_mean"])


def test_noncanonical_quaternion_is_rejected():
    arrays = make_full_windows()
    arrays["paper_x"][0, -1] = -1.0
    with pytest.raises(StageCCacheError):
        validate_stage_b_windows(arrays)


def test_split_overlap_is_rejected():
    arrays = make_full_windows()
    train_index = int(np.flatnonzero(arrays["split_name"].astype(str) == "train")[0])
    val_index = int(np.flatnonzero(arrays["split_name"].astype(str) == "val")[0])
    arrays["visible_seed"][val_index] = arrays["visible_seed"][train_index]
    with pytest.raises(StageCCacheError):
        validate_stage_b_windows(arrays)


def test_group_fold_assignment_keeps_pairs_together():
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    folds, mapping = group_fold_assignment(groups, folds=4)
    assert len(mapping) == 4
    for group in set(groups.tolist()):
        assert len(set(folds[groups == group].tolist())) == 1


def test_grouped_oof_regression_recovers_linear_relation():
    groups = np.repeat(np.arange(20), 2)
    x = np.arange(40, dtype=np.float64)[:, None]
    y = np.concatenate([2.0 * x + 1.0, -3.0 * x + 2.0], axis=1)
    prediction, contract = grouped_oof_regression(
        features=x,
        targets=y,
        groups=groups,
        folds=5,
        regularization=1.0e-8,
    )
    assert contract["group_integrity_pass"] is True
    assert float(np.max(np.abs(prediction - y))) < 1.0e-4


def test_grouped_future_prediction_respects_valid_mask():
    view = make_small_train_view(40)
    mask = view["future_valid_mask"].copy()
    mask[::3, -1] = False
    prediction, contract = grouped_oof_future_robot(
        features=view["state_action_x"],
        robot_future=view["robot_future"],
        future_valid_mask=mask,
        groups=view["episode_group_key"],
        spec=AttributionSpec(folds=4),
    )
    assert np.all(np.isfinite(prediction))
    assert contract["group_integrity_pass"] is True


def test_constant_velocity_baseline_preserves_unit_quaternion():
    view = make_small_train_view(16)
    prediction = constant_velocity_baseline(view["robot_history"])
    quaternion = prediction[..., ROBOT_EE_QUATERNION_SLICE]
    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=-1), 1.0)


def test_schema_and_action_probe_support_state_v3_semantics():
    view = make_small_train_view(64)
    spec = AttributionSpec(folds=4, robot_action_incremental_gain_min=0.01)
    schema = schema_audit(
        view["robot_history"],
        view["robot_future"],
        view["future_valid_mask"],
        spec=spec,
    )
    probe = action_prediction_probe(
        paper_x=view["paper_x"],
        state_action_x=view["state_action_x"],
        robot_history=view["robot_history"],
        robot_future=view["robot_future"],
        cable_future=view["y_state"][..., :CABLE_DIM],
        action_target=view["y_action"],
        groups=view["episode_group_key"],
        spec=spec,
    )
    assert schema["robot_proxy_dim"] == ROBOT_PROXY_DIM
    assert schema["schema_contract_pass"] is True
    assert probe["robot_incremental_gain"] > 0.5


def test_run_train_only_attribution_is_data_side_and_blocked():
    view = make_small_train_view(64)
    report = run_train_only_attribution(
        view,
        spec=AttributionSpec(
            folds=4,
            deployable_nmse_max=2.0,
            action_conditioned_gain_min=0.01,
            robot_action_incremental_gain_min=0.01,
        ),
    )
    assert report["verdict"] == "PASS"
    assert report["scientific_status"] == "BLOCKED"
    assert report["train_only"] is True
    assert report["validation_targets_used"] is False
    assert report["model_attribution_performed"] is False
    assert report["diffusion_training"] is False
