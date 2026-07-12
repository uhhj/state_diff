from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from ccda_phase3.phase314a_contract import (
    ACTION_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    PAPER_X_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    action_history_valid_mask,
    canonical_source_path,
    current_state_from_paper_x,
    fit_masked_future_standardizer,
    future_valid_mask,
    load_raw_action_count,
    state_history_valid_mask,
    strict_json_dump,
    validate_formal_windows,
)


def formal_arrays(rows: int = 4):
    paper = np.zeros((rows, PAPER_X_DIM), dtype=np.float32)
    state_action = np.zeros(
        (rows, STATE_ACTION_X_DIM),
        dtype=np.float32,
    )
    state_action[:, :PAPER_X_DIM] = paper
    future = np.zeros(
        (rows, DEFAULT_TF, STATE_DIM),
        dtype=np.float32,
    )
    final = np.zeros((rows, STATE_DIM), dtype=np.float32)
    action = np.zeros((rows, ACTION_DIM), dtype=np.float32)
    conditions = np.array(
        ["free", "hidden_slack_breakaway_pin_v2"] * (rows // 2),
        dtype="<U64",
    )
    split = np.array(
        ["train", "val", "test", "train"],
        dtype="<U8",
    )
    return {
        "paper_x": paper,
        "state_action_x": state_action,
        "y_state": future,
        "y_final_state": final,
        "y_action": action,
        "condition_name": conditions,
        "visible_seed": np.arange(rows, dtype=np.int64),
        "split_name": split,
        "source_file": np.array(["x"] * rows, dtype="<U8"),
        "pair_group": np.array(["g"] * rows, dtype="<U8"),
        "window_t": np.arange(rows, dtype=np.int64),
        "success": np.zeros(rows, dtype=np.bool_),
        "final_fraction": np.zeros(rows, dtype=np.float32),
        "engagement_step": np.full(rows, -1, dtype=np.int64),
        "release_step": np.full(rows, -1, dtype=np.int64),
        "pre_engagement": np.ones(rows, dtype=np.bool_),
    }


def test_distinct_state_and_action_history_masks():
    np.testing.assert_array_equal(
        state_history_valid_mask(0),
        [False, False, True],
    )
    np.testing.assert_array_equal(
        action_history_valid_mask(0),
        [False, False, False],
    )
    np.testing.assert_array_equal(
        state_history_valid_mask(1),
        [False, True, True],
    )
    np.testing.assert_array_equal(
        action_history_valid_mask(1),
        [False, False, True],
    )


def test_future_mask_excludes_right_padding():
    np.testing.assert_array_equal(
        future_valid_mask(5, 0),
        [True, True, True, True],
    )
    np.testing.assert_array_equal(
        future_valid_mask(5, 3),
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        future_valid_mask(5, 4),
        [True, False, False, False],
    )


def test_current_state_is_last_history_frame():
    paper = np.arange(PAPER_X_DIM, dtype=np.float32).reshape(1, -1)
    current = current_state_from_paper_x(paper)
    np.testing.assert_array_equal(
        current[0],
        paper[0, -STATE_DIM:],
    )


def test_masked_future_standardizer_ignores_padding():
    future = np.zeros((4, 4, STATE_DIM), dtype=np.float32)
    future[:, 0] = 1.0
    future[:, 1] = 2.0
    future[:, 2] = 1000.0
    future[:, 3] = 2000.0
    mask = np.array(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    fitted = fit_masked_future_standardizer(future, mask)
    assert np.all(fitted.mean[0] == 1.0)
    assert np.all(fitted.mean[1] == 2.0)
    assert np.all(fitted.mean[2] == 1000.0)
    assert np.all(fitted.mean[3] == 2000.0)
    # Only one real row at horizon 3 is not enough for formal fitting.
    with pytest.raises(ValueError):
        fit_masked_future_standardizer(
            future[:3],
            mask[:3],
        )


def test_canonical_source_relinks_staging_path(tmp_path):
    formal = tmp_path / "formal"
    target = formal / "raw/train/free/episode.pkl"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    resolved, relinked = canonical_source_path(
        formal,
        split="train",
        condition="free",
        recorded_source="/old/staging/raw/train/free/episode.pkl",
    )
    assert resolved == target.resolve()
    assert relinked


def test_raw_action_count_contract(tmp_path):
    path = tmp_path / "episode.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "actions": [1, 2],
                "infos": [{}, {}],
                "last_info": {},
            },
            handle,
        )
    assert load_raw_action_count(path) == 2


def test_strict_json_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        strict_json_dump(tmp_path / "bad.json", {"x": np.nan})


def test_formal_validator_detects_wrong_prefix():
    arrays = formal_arrays()
    arrays["state_action_x"][0, 0] = 1.0
    with pytest.raises(ValueError):
        validate_formal_windows(arrays, expected_rows=4)
