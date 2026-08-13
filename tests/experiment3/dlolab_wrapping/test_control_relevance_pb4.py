import copy

import numpy as np

from scripts.experiment3.dlolab_wrapping.control_relevance_pb4 import (
    ACTION_FAILED,
    LIVE_FAILED,
    NEGATIVE_VERDICT,
    POSITIVE_VERDICT,
    SNAPSHOT_FAILED,
    analyze_control_relevance,
    masked_qpos_command,
    max_pairwise_abs,
    prefix_validity_from_failure_steps,
    published_score_transition,
    recompute_task_score,
    summarize_branch,
    validate_result_boundaries,
)


def _actions():
    return [
        {
            "action_id": "both_continue",
            "arm1_mask": 1.0,
            "arm2_mask": 1.0,
        },
        {
            "action_id": "arm1_continue_arm2_hold",
            "arm1_mask": 1.0,
            "arm2_mask": 0.0,
        },
        {
            "action_id": "arm1_hold_arm2_continue",
            "arm1_mask": 0.0,
            "arm2_mask": 1.0,
        },
        {
            "action_id": "hold_both",
            "arm1_mask": 0.0,
            "arm2_mask": 0.0,
        },
    ]


def _config():
    return {
        "action_library": {
            "actions": _actions(),
        },
        "control_relevance": {
            "absolute_regret_min": 0.05,
            "repeat_floor_multiplier": 5.0,
            "minimum_passing_pairs": 3,
            "minimum_passing_winding_strata": 2,
        },
    }


def test_masked_qpos_action_library_is_armwise_continue_or_hold():
    qpos = np.zeros((50, 18), dtype=np.float64)
    qpos[10] = np.arange(18, dtype=np.float64)
    qpos[15] = qpos[10] + 10.0

    both = masked_qpos_command(
        qpos,
        10,
        5,
        _actions()[0],
    )
    arm1 = masked_qpos_command(
        qpos,
        10,
        5,
        _actions()[1],
    )
    arm2 = masked_qpos_command(
        qpos,
        10,
        5,
        _actions()[2],
    )
    hold = masked_qpos_command(
        qpos,
        10,
        5,
        _actions()[3],
    )

    assert np.array_equal(
        both,
        qpos[15],
    )
    assert np.array_equal(
        arm1[:9],
        qpos[15, :9],
    )
    assert np.array_equal(
        arm1[9:],
        qpos[10, 9:],
    )
    assert np.array_equal(
        arm2[:9],
        qpos[10, :9],
    )
    assert np.array_equal(
        arm2[9:],
        qpos[15, 9:],
    )
    assert np.array_equal(
        hold,
        qpos[10],
    )


def test_action_command_does_not_read_hidden_state():
    qpos = np.arange(
        40 * 18,
        dtype=np.float64,
    ).reshape(40, 18)
    action = _actions()[1]

    first = masked_qpos_command(
        qpos,
        13,
        7,
        action,
    )
    second = masked_qpos_command(
        qpos.copy(),
        13,
        7,
        copy.deepcopy(action),
    )

    assert np.array_equal(
        first,
        second,
    )


def test_repeat_floor_is_max_pairwise_absolute_difference():
    assert np.isclose(
        max_pairwise_abs(
            [1.0, 1.1, 0.9]
        ),
        0.2,
    )


def _branch_records(rollout, t, scores_by_action):
    rows = []
    for action_id, scores in scores_by_action.items():
        for repeat, score in enumerate(scores):
            rows.append(
                {
                    "rollout_id": rollout,
                    "time_index": t,
                    "action_id": action_id,
                    "repeat_index": repeat,
                    "task_score": float(score),
                    "survived_horizon": True,
                }
            )
    return rows


def test_unique_best_requires_absolute_or_repeat_floor_margin():
    actions = _actions()

    records = _branch_records(
        1,
        13,
        {
            "both_continue": [1.00, 1.00, 1.00],
            "arm1_continue_arm2_hold": [0.97, 0.97, 0.97],
            "arm1_hold_arm2_continue": [0.80, 0.80, 0.80],
            "hold_both": [0.70, 0.70, 0.70],
        },
    )
    summary = summarize_branch(
        records,
        actions,
        0.05,
        5.0,
    )
    assert summary["best_action_id"] == "both_continue"
    assert summary["unique_best"] is False

    records = _branch_records(
        1,
        13,
        {
            "both_continue": [1.10, 1.10, 1.10],
            "arm1_continue_arm2_hold": [1.00, 1.00, 1.00],
            "arm1_hold_arm2_continue": [0.80, 0.80, 0.80],
            "hold_both": [0.70, 0.70, 0.70],
        },
    )
    summary = summarize_branch(
        records,
        actions,
        0.05,
        5.0,
    )
    assert summary["unique_best"] is True


def _shortlist():
    pairs = []
    for index in range(10):
        if index < 5:
            ia = [1, 1, 0]
            ib = [0, 1, 0]
        else:
            ia = [0, 1, 0]
            ib = [0, 0, 0]
        pairs.append(
            {
                "pair_id": f"p{index+1}",
                "source_rank": index + 1,
                "rollout_a": 2 * index,
                "rollout_b": 2 * index + 1,
                "time_index": 13,
                "winding_index_a": ia,
                "winding_index_b": ib,
            }
        )
    return {"pairs": pairs}


def _live_records(shortlist, overrides=None):
    overrides = {} if overrides is None else overrides
    rows = []
    for pair in shortlist["pairs"]:
        live_a, live_b = overrides.get(
            pair["pair_id"],
            (
                pair["winding_index_a"],
                pair["winding_index_b"],
            ),
        )
        a = ",".join(str(int(v)) for v in live_a)
        b = ",".join(str(int(v)) for v in live_b)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "valid": True,
                "live_winding_index_a": [int(v) for v in live_a],
                "live_winding_index_b": [int(v) for v in live_b],
                "pb4_live_winding_stratum": "<->".join(sorted((a, b))),
                "prefix_valid_through_t": True,
            }
        )
    return rows


def _passing_pair_records(pair, best_a, best_b):
    actions = [row["action_id"] for row in _actions()]
    records = []

    for side, rollout, best in (
        ("a", pair["rollout_a"], best_a),
        ("b", pair["rollout_b"], best_b),
    ):
        scores = {}
        for action_id in actions:
            scores[action_id] = [0.6, 0.6, 0.6]
        scores[best] = [1.0, 1.0, 1.0]
        records.extend(
            _branch_records(
                rollout,
                pair["time_index"],
                scores,
            )
        )
    return records


def test_control_relevance_positive_requires_three_pairs_across_two_strata():
    shortlist = _shortlist()
    records = []

    # Three passing switches: p1,p2 in stratum 1; p6 in stratum 2.
    passing_indices = {0, 1, 5}
    for index, pair in enumerate(shortlist["pairs"]):
        if index in passing_indices:
            records.extend(
                _passing_pair_records(
                    pair,
                    "arm1_continue_arm2_hold",
                    "arm1_hold_arm2_continue",
                )
            )
        else:
            # Same best action => not a formal control switch.
            records.extend(
                _passing_pair_records(
                    pair,
                    "both_continue",
                    "both_continue",
                )
            )

    audit = analyze_control_relevance(
        _config(),
        shortlist,
        records,
        _live_records(shortlist),
    )
    assert audit["confirmed"] is True
    assert audit["passing_pair_count"] == 3
    assert audit["passing_winding_strata"] == 2


def test_two_passing_pairs_are_not_enough():
    shortlist = _shortlist()
    records = []
    passing_indices = {0, 5}

    for index, pair in enumerate(shortlist["pairs"]):
        if index in passing_indices:
            records.extend(
                _passing_pair_records(
                    pair,
                    "arm1_continue_arm2_hold",
                    "arm1_hold_arm2_continue",
                )
            )
        else:
            records.extend(
                _passing_pair_records(
                    pair,
                    "both_continue",
                    "both_continue",
                )
            )

    audit = analyze_control_relevance(
        _config(),
        shortlist,
        records,
        _live_records(shortlist),
    )
    assert audit["confirmed"] is False
    assert audit["passing_pair_count"] == 2


def test_robust_cross_regret_rejects_repeat_overlap():
    shortlist = _shortlist()
    pair = shortlist["pairs"][0]
    records = []

    # Medians switch, but repeat ranges overlap heavily.
    a_scores = {
        "both_continue": [1.00, 0.80, 1.00],
        "arm1_continue_arm2_hold": [0.90, 0.90, 0.90],
        "arm1_hold_arm2_continue": [0.60, 0.60, 0.60],
        "hold_both": [0.50, 0.50, 0.50],
    }
    b_scores = {
        "both_continue": [0.60, 0.60, 0.60],
        "arm1_continue_arm2_hold": [0.70, 0.70, 0.70],
        "arm1_hold_arm2_continue": [1.00, 0.80, 1.00],
        "hold_both": [0.50, 0.50, 0.50],
    }
    records.extend(
        _branch_records(
            pair["rollout_a"],
            13,
            a_scores,
        )
    )
    records.extend(
        _branch_records(
            pair["rollout_b"],
            13,
            b_scores,
        )
    )

    # Fill remaining pairs with same-best actions so the analyzer can run.
    for other in shortlist["pairs"][1:]:
        records.extend(
            _passing_pair_records(
                other,
                "both_continue",
                "both_continue",
            )
        )

    audit = analyze_control_relevance(
        _config(),
        shortlist,
        records,
        _live_records(shortlist),
    )
    row = audit["pairs"][0]
    assert row["pair_control_relevance_pass"] is False



def test_prefix_validity_rejects_failure_at_or_before_branch_time():
    valid = prefix_validity_from_failure_steps(
        13,
        20,
        10**9,
    )
    assert valid["prefix_valid_through_t"] is True
    assert valid["failed_stretch_through_t"] is False

    failed_at_t = prefix_validity_from_failure_steps(
        13,
        13,
        10**9,
    )
    assert failed_at_t["prefix_valid_through_t"] is False
    assert failed_at_t["failed_stretch_through_t"] is True

    failed_before_t = prefix_validity_from_failure_steps(
        13,
        10**9,
        10,
    )
    assert failed_before_t["prefix_valid_through_t"] is False
    assert failed_before_t["failed_rope_nan_through_t"] is True


def test_reward_nan_zeroes_only_current_contribution():
    step = published_score_transition(
        alive=True,
        rope_nan=False,
        stretch_failed=False,
        reward=np.nan,
    )
    assert step["alive"] is True
    assert step["contribution"] == 0.0
    assert step["reward_nan"] is True
    assert step["physical_failure"] is False

    next_step = published_score_transition(
        alive=step["alive"],
        rope_nan=False,
        stretch_failed=False,
        reward=0.25,
    )
    assert next_step["alive"] is True
    assert np.isclose(
        next_step["contribution"],
        1.25,
    )


def test_physical_failure_permanently_clears_alive():
    step = published_score_transition(
        alive=True,
        rope_nan=False,
        stretch_failed=True,
        reward=0.5,
    )
    assert step["alive"] is False
    assert step["contribution"] == 0.0
    assert step["physical_failure"] is True

    next_step = published_score_transition(
        alive=step["alive"],
        rope_nan=False,
        stretch_failed=False,
        reward=1.0,
    )
    assert next_step["alive"] is False
    assert next_step["contribution"] == 0.0


def test_task_score_reconstructs_from_20_contributions():
    contributions = [1.0] * 10 + [0.0] * 10
    assert np.isclose(
        recompute_task_score(
            contributions,
            20,
        ),
        0.5,
    )


def test_phase_strata_use_pb4_live_not_frozen_strata():
    shortlist = _shortlist()
    records = []
    passing_indices = {0, 1, 5}

    for index, pair in enumerate(shortlist["pairs"]):
        if index in passing_indices:
            records.extend(
                _passing_pair_records(
                    pair,
                    "arm1_continue_arm2_hold",
                    "arm1_hold_arm2_continue",
                )
            )
        else:
            records.extend(
                _passing_pair_records(
                    pair,
                    "both_continue",
                    "both_continue",
                )
            )

    # Force all three passing pairs into the same PB4 live stratum even though
    # p6 has a different frozen PB3-B1 stratum.
    same_live = {
        "p1": ([0, 0, 0], [0, 1, 0]),
        "p2": ([0, 0, 0], [0, 1, 0]),
        "p6": ([0, 0, 0], [0, 1, 0]),
    }
    audit = analyze_control_relevance(
        _config(),
        shortlist,
        records,
        _live_records(
            shortlist,
            same_live,
        ),
    )
    assert audit["passing_pair_count"] == 3
    assert audit["passing_winding_strata"] == 1
    assert audit["confirmed"] is False

    two_live = dict(same_live)
    two_live["p6"] = (
        [0, 1, 0],
        [1, 1, 0],
    )
    audit = analyze_control_relevance(
        _config(),
        shortlist,
        records,
        _live_records(
            shortlist,
            two_live,
        ),
    )
    assert audit["passing_winding_strata"] == 2
    assert audit["confirmed"] is True


def _scientific_stub(
        *,
        live_valid,
        restore_valid,
        restore_records,
        audit,
        blocked,
    ):
    return {
        "live_pair_barrier": {
            "valid": live_valid,
        },
        "snapshot_restore_barrier": {
            "valid": restore_valid,
        },
        "snapshot_restore_records": restore_records,
        "audit": audit,
        "blocked_details": blocked,
    }


def test_blocked_result_boundaries_live_failure():
    scientific = _scientific_stub(
        live_valid=False,
        restore_valid=False,
        restore_records=[],
        audit=None,
        blocked={
            "failure_component":
                "live_cohort_revalidation",
        },
    )
    mode = validate_result_boundaries(
        LIVE_FAILED,
        scientific,
        [],
    )
    assert mode == "blocked_live"


def test_blocked_result_boundaries_snapshot_failure():
    scientific = _scientific_stub(
        live_valid=True,
        restore_valid=False,
        restore_records=[
            {"valid": False}
        ],
        audit=None,
        blocked={
            "failure_component":
                "snapshot_restore_validation",
        },
    )
    mode = validate_result_boundaries(
        SNAPSHOT_FAILED,
        scientific,
        [],
    )
    assert mode == "blocked_snapshot"


def test_blocked_result_boundaries_action_failure():
    scientific = _scientific_stub(
        live_valid=True,
        restore_valid=True,
        restore_records=[
            {"valid": True}
        ] * 60,
        audit=None,
        blocked={
            "failure_component":
                "incomplete_action_matrix",
        },
    )
    mode = validate_result_boundaries(
        ACTION_FAILED,
        scientific,
        [],
    )
    assert mode == "blocked_action"
