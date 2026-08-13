import copy

import numpy as np

from scripts.experiment3.dlolab_wrapping.prospective_live_cohort_pb3b1 import (
    STRATA_INSUFFICIENT_VERDICT,
    SUCCESS_VERDICT,
    evaluate_live_candidate,
    mine_prefix_only_candidate_queue,
    prefix_valid_indices_at_time,
    scan_live_cohort,
    valid_through_branch,
)


def _pb2c_config():
    return {
        "discovery": {
            "pair_batch_size": 64,
        },
        "partial_state_observation": {
            "history_samples": 3,
            "rope_component": {
                "post_radius_m": 0.015,
                "rope_radius_m": 0.01,
                "local_occlusion_radius_multiplier": 2.0,
                "max_visible_history_chamfer_m": 0.01,
            },
            "robot_component": {
                "max_dual_ee_position_history_mean_m": 0.01,
                "max_dual_ee_quaternion_geodesic_history_mean_rad":
                    0.08726646259971647,
                "max_dual_motor_qpos_history_rms_rad": 0.05,
            },
        }
    }


def _config():
    return {
        "cohort": {
            "target_pair_count": 10,
            "max_rollout_use_count": 1,
            "minimum_live_winding_strata_to_proceed": 2,
        }
    }


def _live_data(n_rollouts=30, n_times=25):
    rope = np.zeros(
        (n_rollouts, n_times, 50, 3),
        dtype=np.float32,
    )
    rope[:, :, :, 0] = np.linspace(
        0.0,
        0.2,
        50,
        dtype=np.float32,
    )[None, None, :]

    posts = np.zeros(
        (n_rollouts, n_times, 3, 3),
        dtype=np.float32,
    )
    posts[..., 0] = 10.0
    posts[..., 1] = 10.0

    quat = np.zeros(
        (n_rollouts, n_times, 4),
        dtype=np.float32,
    )
    quat[..., 0] = 1.0

    winding = np.zeros(
        (n_rollouts, n_times, 3),
        dtype=np.float32,
    )

    data = {
        "rollout_id":
            np.arange(n_rollouts, dtype=np.int32),

        "batch_index":
            np.zeros(
                n_rollouts,
                dtype=np.int16,
            ),

        "env_index":
            np.arange(
                n_rollouts,
                dtype=np.int16,
            ),

        "batch_seed":
            np.full(
                n_rollouts,
                123,
                dtype=np.int32,
            ),

        "official_rollout_valid":
            np.ones(
                n_rollouts,
                dtype=bool,
            ),

        "rope_xyz":
            rope,

        "post_xyz":
            posts,

        "ee1_pos":
            np.zeros(
                (n_rollouts, n_times, 3),
                dtype=np.float32,
            ),

        "ee1_quat":
            quat.copy(),

        "ee2_pos":
            np.zeros(
                (n_rollouts, n_times, 3),
                dtype=np.float32,
            ),

        "ee2_quat":
            quat.copy(),

        "motor_qpos_1":
            np.zeros(
                (n_rollouts, n_times, 7),
                dtype=np.float32,
            ),

        "motor_qpos_2":
            np.zeros(
                (n_rollouts, n_times, 7),
                dtype=np.float32,
            ),

        "signed_winding_turns":
            winding,

        "official_failed_rope_nan":
            np.zeros(
                n_rollouts,
                dtype=bool,
            ),

        "official_failed_stretch":
            np.zeros(
                n_rollouts,
                dtype=bool,
            ),

        "official_first_fail_step":
            np.full(
                n_rollouts,
                24,
                dtype=np.int32,
            ),
    }
    return data


def _candidate(rank, a, b, time_index=13):
    return {
        "source_rank": rank,
        "rollout_a": a,
        "rollout_b": b,
        "time_index": time_index,
        "replay_a": {
            "batch_index": 0,
            "env_index": a,
            "seed": 123,
        },
        "replay_b": {
            "batch_index": 0,
            "env_index": b,
            "seed": 123,
        },
        "frozen_candidate_diagnostic": {},
    }


def _cache(data):
    from scripts.experiment3.dlolab_wrapping.prospective_live_cohort_pb3b1 import (
        prepare_live_screen_cache,
    )
    return prepare_live_screen_cache(
        data,
        _pb2c_config(),
    )


def test_validity_failure_after_branch_does_not_reject_branch():
    data = _live_data()
    data[
        "official_failed_stretch"
    ][0] = True
    data[
        "official_first_fail_step"
    ][0] = 20

    assert valid_through_branch(
        data,
        0,
        13,
    ) is True

    assert valid_through_branch(
        data,
        0,
        20,
    ) is False


def test_live_candidate_uses_only_history_through_branch():
    data = _live_data()
    data[
        "signed_winding_turns"
    ][1, :, 1] = 1.0

    candidate = _candidate(
        1,
        0,
        1,
        time_index=13,
    )

    before = evaluate_live_candidate(
        candidate,
        data,
        _pb2c_config(),
        _cache(data),
    )

    changed = copy.deepcopy(
        data
    )

    # Change only post-branch states. Admission must not change.
    changed[
        "rope_xyz"
    ][0, 14:, :, :] = 1000.0
    changed[
        "ee1_pos"
    ][0, 14:, :] = 1000.0
    changed[
        "signed_winding_turns"
    ][0, 14:, :] = 1000.0

    after = evaluate_live_candidate(
        candidate,
        changed,
        _pb2c_config(),
        _cache(changed),
    )

    assert before == after
    assert before["valid"] is True


def test_live_candidate_rejects_same_live_winding_index():
    data = _live_data()

    candidate = _candidate(
        1,
        0,
        1,
        time_index=13,
    )

    row = evaluate_live_candidate(
        candidate,
        data,
        _pb2c_config(),
        _cache(data),
    )

    assert row["valid"] is False
    assert row["passes"][
        "live_winding_index_different"
    ] is False


def _sources():
    return {
        "pb2c_config": _pb2c_config(),
    }


def _protocol():
    return {
        "verdict": "PB3B1_PROTOCOL_PREREGISTERED",
    }


def test_scan_freezes_first_10_and_does_not_continue_to_rescue_strata():
    data = _live_data(
        n_rollouts=24,
        n_times=25,
    )

    candidates = []

    # First ten pairs are all 0,0,0 <-> 0,1,0.
    for rank in range(1, 11):
        a = 2 * (rank - 1)
        b = a + 1
        data[
            "signed_winding_turns"
        ][b, :, 1] = 1.0

        candidates.append(
            _candidate(
                rank,
                a,
                b,
                13,
            )
        )

    # Rank 11 would provide a second stratum, but frozen rule must stop at 10.
    data[
        "signed_winding_turns"
    ][21, :, 0] = 1.0

    candidates.append(
        _candidate(
            11,
            20,
            21,
            13,
        )
    )

    queue = {
        "source_candidate_count":
            11,

        "eligible_candidate_count":
            11,

        "eligible_candidates":
            candidates,
    }

    cohort, funnel, scan_log = (
        scan_live_cohort(
            _config(),
            _sources(),
            _protocol(),
            queue,
            data,
        )
    )

    assert cohort[
        "selection"
    ][
        "accepted_pair_count"
    ] == 10

    assert cohort[
        "selection"
    ][
        "unique_rollout_count"
    ] == 20

    assert funnel[
        "scanned_until_stop"
    ] == 10

    assert len(
        scan_log
    ) == 10

    assert cohort[
        "live_winding_strata_count"
    ] == 1

    assert cohort[
        "verdict"
    ] == STRATA_INSUFFICIENT_VERDICT


def test_scan_accepts_10_rank_order_disjoint_pairs_when_two_strata_present():
    data = _live_data(
        n_rollouts=24,
        n_times=25,
    )
    candidates = []

    for rank in range(1, 11):
        a = 2 * (rank - 1)
        b = a + 1

        if rank < 10:
            data[
                "signed_winding_turns"
            ][b, :, 1] = 1.0
        else:
            data[
                "signed_winding_turns"
            ][b, :, 0] = 1.0

        candidates.append(
            _candidate(
                rank,
                a,
                b,
                13,
            )
        )

    queue = {
        "source_candidate_count":
            10,

        "eligible_candidate_count":
            10,

        "eligible_candidates":
            candidates,
    }

    cohort, funnel, _ = scan_live_cohort(
        _config(),
        _sources(),
        _protocol(),
        queue,
        data,
    )

    assert cohort[
        "verdict"
    ] == SUCCESS_VERDICT

    assert [
        row[
            "source_rank"
        ]
        for row in cohort[
            "pairs"
        ]
    ] == list(
        range(
            1,
            11,
        )
    )

    assert cohort[
        "live_winding_strata_count"
    ] == 2

    assert funnel[
        "accepted_pairs"
    ] == 10


def test_rollout_reuse_is_never_admitted_twice():
    data = _live_data(
        n_rollouts=24,
        n_times=25,
    )

    for odd in range(
        1,
        24,
        2,
    ):
        data[
            "signed_winding_turns"
        ][odd, :, 1] = 1.0

    candidates = [
        _candidate(
            1,
            0,
            1,
            13,
        ),
        # Shares rollout 0 and must be skipped.
        _candidate(
            2,
            0,
            3,
            13,
        ),
    ]

    rank = 3
    for a in range(
        2,
        22,
        2,
    ):
        candidates.append(
            _candidate(
                rank,
                a,
                a + 1,
                13,
            )
        )
        rank += 1

    # Make final pair a second stratum.
    data[
        "signed_winding_turns"
    ][21, :, :] = 0.0
    data[
        "signed_winding_turns"
    ][21, :, 0] = 1.0

    queue = {
        "source_candidate_count":
            len(
                candidates
            ),

        "eligible_candidate_count":
            len(
                candidates
            ),

        "eligible_candidates":
            candidates,
    }

    cohort, funnel, _ = scan_live_cohort(
        _config(),
        _sources(),
        _protocol(),
        queue,
        data,
    )

    used = [
        rollout
        for row in cohort[
            "pairs"
        ]
        for rollout in (
            row[
                "rollout_a"
            ],
            row[
                "rollout_b"
            ],
        )
    ]

    assert len(
        used
    ) == len(
        set(
            used
        )
    )

    assert funnel[
        "rejected_rollout_already_used"
    ] >= 1



def test_prefix_queue_does_not_condition_on_future_survival():
    data = _live_data(
        n_rollouts=3,
        n_times=25,
    )

    # Pair 0/1 is a valid hidden-winding pair at t=13.
    data[
        "signed_winding_turns"
    ][1, :, 1] = 1.0

    # Rollout 1 later fails at t=20 and therefore is NOT full-rollout valid.
    data[
        "official_failed_stretch"
    ][1] = True
    data[
        "official_first_fail_step"
    ][1] = 20
    data[
        "official_rollout_valid"
    ][1] = False

    prefix_13 = prefix_valid_indices_at_time(
        data,
        13,
        3,
    )
    assert 1 in set(
        prefix_13.tolist()
    )

    prefix_20 = prefix_valid_indices_at_time(
        data,
        20,
        3,
    )
    assert 1 not in set(
        prefix_20.tolist()
    )

    mined = mine_prefix_only_candidate_queue(
        data,
        _pb2c_config(),
    )

    pair_at_13 = [
        row
        for row in mined[
            "candidates"
        ]
        if (
            row[
                "rollout_a"
            ] == 0
            and row[
                "rollout_b"
            ] == 1
            and row[
                "time_index"
            ] == 13
        )
    ]

    assert len(
        pair_at_13
    ) == 1

    row = pair_at_13[0]
    assert row[
        "selection_used_full_rollout_survival"
    ] is False
    assert row[
        "full_rollout_valid_b_diagnostic"
    ] is False


def test_prefix_queue_is_invariant_to_post_branch_failure_time_if_after_branch():
    data_a = _live_data(
        n_rollouts=3,
        n_times=25,
    )
    data_a[
        "signed_winding_turns"
    ][1, :, 1] = 1.0
    data_a[
        "official_failed_stretch"
    ][1] = True
    data_a[
        "official_first_fail_step"
    ][1] = 18
    data_a[
        "official_rollout_valid"
    ][1] = False

    data_b = copy.deepcopy(
        data_a
    )
    data_b[
        "official_first_fail_step"
    ][1] = 24

    a = mine_prefix_only_candidate_queue(
        data_a,
        _pb2c_config(),
    )
    b = mine_prefix_only_candidate_queue(
        data_b,
        _pb2c_config(),
    )

    def signature_at_13(result):
        return [
            (
                row[
                    "rollout_a"
                ],
                row[
                    "rollout_b"
                ],
                row[
                    "time_index"
                ],
                row[
                    "visible_rope_history_chamfer_m"
                ],
            )
            for row in result[
                "candidates"
            ]
            if row[
                "time_index"
            ] == 13
        ]

    assert signature_at_13(
        a
    ) == signature_at_13(
        b
    )
