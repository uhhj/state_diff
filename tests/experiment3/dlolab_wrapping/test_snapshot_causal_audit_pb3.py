import copy

import numpy as np
import pytest

from scripts.experiment3.dlolab_wrapping.snapshot_causal_audit_pb3 import (
    derive_rank_order_greedy_shortlist,
    gate4_threshold,
    ordered_displacement_field_rmse,
    validate_shortlist_against_pb2c,
    winding_stratum,
)


def test_displacement_metric_removes_static_translation_difference():
    a0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ah = a0 + np.array([0.1, 0.0, 0.0])

    b0 = a0 + np.array([5.0, -3.0, 2.0])
    bh = b0 + np.array([0.1, 0.0, 0.0])

    value = ordered_displacement_field_rmse(
        a0,
        ah,
        b0,
        bh,
    )

    assert np.isclose(value, 0.0, atol=1e-12)


def test_displacement_metric_detects_differential_future_motion():
    a0 = np.zeros((4, 3), dtype=np.float64)
    b0 = np.ones((4, 3), dtype=np.float64)

    ah = a0.copy()
    bh = b0.copy()
    bh[:, 0] += 0.002

    value = ordered_displacement_field_rmse(
        a0,
        ah,
        b0,
        bh,
    )

    assert np.isclose(value, 0.002, atol=1e-12)


def test_gate4_threshold_uses_absolute_floor_when_repeat_is_tiny():
    threshold = gate4_threshold(
        1e-6,
        absolute_effect_min_m=0.001,
        repeat_floor_multiplier=5.0,
    )

    assert np.isclose(threshold, 0.001)


def test_gate4_threshold_uses_repeat_relative_floor_when_larger():
    threshold = gate4_threshold(
        4e-4,
        absolute_effect_min_m=0.001,
        repeat_floor_multiplier=5.0,
    )

    assert np.isclose(threshold, 0.002)


def test_winding_stratum_is_orientation_invariant():
    a = winding_stratum(
        [0, 0, 0],
        [0, 1, 0],
    )
    b = winding_stratum(
        [0, 1, 0],
        [0, 0, 0],
    )

    assert a == b


def _candidate(a, b, t=13):
    return {
        "rollout_a": a,
        "rollout_b": b,
        "time_index": t,
        "winding_index_a": [0, 0, 0],
        "winding_index_b": [0, 1, 0],
        "differing_post_indices": [1],
        "visible_rope_history_chamfer_m": 0.001 + 1e-6 * a,
        "pair_max_winding_integer_residual": 1e-8,
        "replay_a": {
            "batch_index": a // 32,
            "env_index": a % 32,
            "seed": 123 + a // 32,
        },
        "replay_b": {
            "batch_index": b // 32,
            "env_index": b % 32,
            "seed": 123 + b // 32,
        },
    }


def _candidate_doc():
    # rank 1 accepted: 1-2
    # rank 2 overlaps rollout 1 -> must be skipped
    # rank 3 accepted: 3-4
    # then enough disjoint pairs to reach ten.
    rows = [
        _candidate(1, 2),
        _candidate(1, 99),
        _candidate(3, 4),
        _candidate(5, 6),
        _candidate(7, 8),
        _candidate(9, 10),
        _candidate(11, 12),
        _candidate(13, 14),
        _candidate(15, 16),
        _candidate(17, 18),
        _candidate(19, 20),
    ]
    return {
        "top_candidates": rows,
    }


def _shortlist_from_expected(expected):
    pairs = []
    for index, row in enumerate(expected):
        pair = copy.deepcopy(row)
        pair["pair_id"] = f"pb3_pair_{index:02d}"
        pairs.append(pair)

    return {
        "selection_rule": {
            "future_information_used": False,
            "target_pair_count": 10,
            "max_rollout_use_count": 1,
        },
        "pairs": pairs,
    }


def test_rank_order_scan_derivation_skips_only_used_rollouts():
    doc = _candidate_doc()
    expected = derive_rank_order_greedy_shortlist(
        doc,
        10,
    )

    assert [row["source_rank"] for row in expected] == [
        1, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]


def test_exact_rank_order_shortlist_passes():
    doc = _candidate_doc()
    expected = derive_rank_order_greedy_shortlist(
        doc,
        10,
    )
    shortlist = _shortlist_from_expected(expected)

    result = validate_shortlist_against_pb2c(
        shortlist,
        doc,
    )

    assert result["rank_order_scan_verified"] is True
    assert result["derived_source_ranks"] == [
        1, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]


def test_monotonic_unique_but_skipping_admissible_rank_is_rejected():
    doc = _candidate_doc()
    expected = derive_rank_order_greedy_shortlist(
        doc,
        10,
    )
    shortlist = _shortlist_from_expected(expected)

    # Old validation (membership + unique rollouts + monotonic ranks) could
    # miss this class of error. Replace the rank-3 accepted pair with a later
    # admissible pair while keeping monotonic ranks and rollout uniqueness.
    replacement = copy.deepcopy(doc["top_candidates"][3])  # source rank 4
    shortlist["pairs"][1] = {
        "pair_id": "pb3_pair_01",
        "source_rank": 4,
        **replacement,
    }

    # Remove the original rank-4 occurrence so the final list remains unique
    # and monotonically ranked, then use an extra later disjoint candidate.
    shortlist["pairs"][2:] = shortlist["pairs"][3:]
    shortlist["pairs"].append(
        {
            "pair_id": "pb3_pair_09",
            "source_rank": 11,
            **copy.deepcopy(doc["top_candidates"][10]),
        }
    )

    with pytest.raises(
        RuntimeError,
        match="rank-order greedy scan",
    ):
        validate_shortlist_against_pb2c(
            shortlist,
            doc,
        )


def test_tampered_candidate_metadata_is_rejected():
    doc = _candidate_doc()
    expected = derive_rank_order_greedy_shortlist(
        doc,
        10,
    )
    shortlist = _shortlist_from_expected(expected)

    shortlist["pairs"][0][
        "visible_rope_history_chamfer_m"
    ] += 1e-4

    with pytest.raises(
        RuntimeError,
        match="rank-order greedy scan",
    ):
        validate_shortlist_against_pb2c(
            shortlist,
            doc,
        )
