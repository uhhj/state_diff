from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccda_phase3.phase314b_r255_stageb_dataset import (
    StateV3MigrationError,
    canonical_source_file,
    compare_build_directories,
    compare_legacy_windows,
    deterministic_npz_bytes,
    optional_int,
    split_isolation,
    stage_a_robot_digest,
    validate_window_pairs,
)
from ccda_phase3.schema_v3 import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    PAPER_X_DIM,
    ROBOT_PROXY_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    SchemaV3Manifest,
    build_window,
    state_v3_from_components,
)


def test_schema_v3_dimensions_and_layout_are_fixed():
    manifest = SchemaV3Manifest()
    manifest.validate()
    assert CABLE_DIM == 48
    assert ROBOT_PROXY_DIM == 19
    assert STATE_DIM == 67
    assert PAPER_X_DIM == 201
    assert STATE_ACTION_X_DIM == 243


def test_state_v3_rejects_nonfinite_and_wrong_shape():
    cable = np.zeros((24, 2), dtype=np.float32)
    robot = np.zeros(19, dtype=np.float32)
    robot[-1] = 1.0
    assert state_v3_from_components(cable, robot).shape == (67,)
    with pytest.raises(ValueError):
        state_v3_from_components(cable[:-1], robot)
    robot[0] = np.nan
    with pytest.raises(ValueError):
        state_v3_from_components(cable, robot)




def test_state_v3_rejects_nonunit_or_noncanonical_quaternion():
    cable = np.zeros((24, 2), dtype=np.float32)
    robot = np.zeros(19, dtype=np.float32)
    with pytest.raises(ValueError):
        state_v3_from_components(cable, robot)
    robot[-1] = -1.0
    with pytest.raises(ValueError):
        state_v3_from_components(cable, robot)


def test_stage_a_robot_digest_is_independent_of_dataset_iteration_order():
    left = ("raw/test/free/b.pkl", 0, np.asarray([2.0], dtype=np.float32))
    right = ("raw/train/free/a.pkl", 1, np.asarray([1.0], dtype=np.float32))
    assert stage_a_robot_digest([left, right]) == stage_a_robot_digest([right, left])


def test_optional_int_preserves_legitimate_zero():
    assert optional_int(None) == -1
    assert optional_int("") == -1
    assert optional_int(0) == 0
    assert optional_int("0") == 0


def test_window_shapes_and_padding_masks():
    states = np.arange(5 * STATE_DIM, dtype=np.float32).reshape(5, STATE_DIM)
    actions = np.arange(4 * ACTION_DIM, dtype=np.float32).reshape(4, ACTION_DIM)
    window = build_window(states=states, action_vectors=actions, current_index=0)
    assert window["paper_x"].shape == (DEFAULT_TH * STATE_DIM,)
    assert window["state_action_x"].shape == (STATE_ACTION_X_DIM,)
    assert window["y_state"].shape == (DEFAULT_TF, STATE_DIM)
    np.testing.assert_array_equal(
        window["state_history_valid_mask"],
        np.asarray([False, False, True]),
    )
    np.testing.assert_array_equal(
        window["action_history_valid_mask"],
        np.asarray([False, False, False]),
    )


def test_deterministic_npz_bytes_ignore_input_mapping_order():
    left = deterministic_npz_bytes(
        {"b": np.arange(3, dtype=np.int64), "a": np.ones(2, dtype=np.float32)}
    )
    right = deterministic_npz_bytes(
        {"a": np.ones(2, dtype=np.float32), "b": np.arange(3, dtype=np.int64)}
    )
    assert left == right


def test_deterministic_npz_rejects_object_dtype():
    with pytest.raises(ValueError):
        deterministic_npz_bytes({"bad": np.asarray([object()], dtype=object)})


def _synthetic_windows(rows: int = 2):
    new_history = np.arange(rows * DEFAULT_TH * STATE_DIM, dtype=np.float32).reshape(
        rows, DEFAULT_TH, STATE_DIM
    )
    new_future = np.arange(rows * DEFAULT_TF * STATE_DIM, dtype=np.float32).reshape(
        rows, DEFAULT_TF, STATE_DIM
    )
    actions = np.arange(rows * DEFAULT_TH * ACTION_DIM, dtype=np.float32).reshape(
        rows, DEFAULT_TH * ACTION_DIM
    )
    y_action = np.arange(rows * ACTION_DIM, dtype=np.float32).reshape(rows, ACTION_DIM)

    old_history = np.zeros((rows, DEFAULT_TH, 87), dtype=np.float32)
    old_history[:, :, :CABLE_DIM] = new_history[:, :, :CABLE_DIM]
    old_future = np.zeros((rows, DEFAULT_TF, 87), dtype=np.float32)
    old_future[:, :, :CABLE_DIM] = new_future[:, :, :CABLE_DIM]
    common = {
        "condition_name": np.asarray(["free", "hidden_slack_breakaway_pin_v2"]),
        "visible_seed": np.asarray([1, 1], dtype=np.int64),
        "split_name": np.asarray(["train", "train"]),
        "pair_group": np.asarray(["pair", "pair"]),
        "window_t": np.asarray([0, 0], dtype=np.int64),
        "success": np.asarray([True, False], dtype=np.bool_),
        "final_fraction": np.asarray([1.0, 0.0], dtype=np.float32),
        "engagement_step": np.asarray([-1, -1], dtype=np.int64),
        "release_step": np.asarray([-1, -1], dtype=np.int64),
        "pre_engagement": np.asarray([True, True], dtype=np.bool_),
    }
    legacy = {
        **common,
        "source_file": np.asarray(
            [
                "/data/state_diff2/data/phase3_state_v2_slack/raw/train/free/a.pkl",
                "/data/state_diff2/data/phase3_state_v2_slack/raw/train/hidden_slack_breakaway_pin_v2/b.pkl",
            ]
        ),
        "paper_x": old_history.reshape(rows, -1),
        "state_action_x": np.concatenate([old_history.reshape(rows, -1), actions], axis=1),
        "y_state": old_future,
        "y_final_state": old_future[:, -1],
        "y_action": y_action,
    }
    migrated = {
        **common,
        "source_file": np.asarray(
            [
                "raw/train/free/a.pkl",
                "raw/train/hidden_slack_breakaway_pin_v2/b.pkl",
            ]
        ),
        "paper_x": new_history.reshape(rows, -1),
        "state_action_x": np.concatenate([new_history.reshape(rows, -1), actions], axis=1),
        "y_state": new_future,
        "y_final_state": new_future[:, -1],
        "y_action": y_action,
    }
    return legacy, migrated


def test_legacy_equivalence_compares_only_preserved_cable_and_action():
    legacy, migrated = _synthetic_windows()
    result = compare_legacy_windows(legacy, migrated)
    assert result["cable_and_action_exact"] is True
    assert result["cable_history_max_abs"] == 0.0
    assert result["target_action_max_abs"] == 0.0


def test_legacy_zero_step_bug_has_explicit_narrow_correction():
    legacy, migrated = _synthetic_windows()
    migrated["engagement_step"] = np.asarray([0, -1], dtype=np.int64)
    result = compare_legacy_windows(legacy, migrated)
    assert result["zero_step_metadata_corrections"]["engagement_step"] == 1
    migrated["engagement_step"] = np.asarray([1, -1], dtype=np.int64)
    with pytest.raises(StateV3MigrationError):
        compare_legacy_windows(legacy, migrated)


def test_canonical_source_requires_raw_component():
    assert canonical_source_file("/x/raw/train/free/a.pkl") == "raw/train/free/a.pkl"
    with pytest.raises(StateV3MigrationError):
        canonical_source_file("train/free/a.pkl")


def test_pair_and_split_audits():
    _, arrays = _synthetic_windows()
    arrays["source_file"] = np.asarray(["raw/train/free/a.pkl", "raw/train/hidden_slack_breakaway_pin_v2/b.pkl"])
    pair = validate_window_pairs(arrays)
    assert pair["pair_window_count"] == 1
    isolation = split_isolation(arrays)
    assert not any(isolation.values())


def test_pair_audit_rejects_duplicate_condition_rows():
    _, arrays = _synthetic_windows()
    duplicated = {
        key: np.concatenate([value, value[:1]], axis=0)
        for key, value in arrays.items()
    }
    with pytest.raises(StateV3MigrationError):
        validate_window_pairs(duplicated)


def test_directory_comparison_is_byte_exact(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a").write_bytes(b"same")
    (right / "a").write_bytes(b"same")
    assert compare_build_directories(left, right)["exact"] is True
    (right / "a").write_bytes(b"different")
    with pytest.raises(StateV3MigrationError):
        compare_build_directories(left, right)
