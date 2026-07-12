from __future__ import annotations

import numpy as np

from ccda_phase3.phase314b_contract import (
    fixed_balanced_eval_indices,
    full_horizon_mask,
    paired_key_index,
)


def synthetic_arrays():
    conditions = np.array(
        [
            "free",
            "hidden_slack_breakaway_pin_v2",
            "free",
            "hidden_slack_breakaway_pin_v2",
        ],
        dtype="<U64",
    )
    return {
        "future_valid_mask": np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, False],
                [True, True, False, False],
            ],
            dtype=np.bool_,
        ),
        "split_name": np.array(
            ["val", "val", "val", "val"],
            dtype="<U8",
        ),
        "pre_engagement": np.ones(4, dtype=np.bool_),
        "pair_key": np.array(
            ["val|1|0", "val|1|0", "val|2|1", "val|2|1"],
            dtype="<U64",
        ),
        "condition_name": conditions,
    }


def test_full_horizon_mask_is_exact():
    arrays = synthetic_arrays()
    np.testing.assert_array_equal(
        full_horizon_mask(arrays),
        [True, True, False, False],
    )


def test_paired_key_index_requires_both_conditions():
    arrays = synthetic_arrays()
    selected = full_horizon_mask(arrays)
    pairs = paired_key_index(arrays, selected)
    assert pairs == {"val|1|0": (0, 1)}


def test_fixed_eval_indices_are_pair_balanced():
    arrays = synthetic_arrays()
    indices = fixed_balanced_eval_indices(
        arrays,
        split="val",
        max_pair_keys=1,
    )
    np.testing.assert_array_equal(indices, [0, 1])
