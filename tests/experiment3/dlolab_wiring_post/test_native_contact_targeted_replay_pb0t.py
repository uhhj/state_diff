from scripts.experiment3.dlolab_wiring_post.native_contact_targeted_replay_pb0t import (
    build_confirmed_shortlist,
    finalize_interval_aggregate,
    init_interval_aggregate,
    pair_confirmation,
    replay_locator,
    update_interval_aggregate,
)


def _step(post0=False, post1=False):
    return {
        "posts": [
            {
                "rigid_post_collision":
                    post0,
                "hidden_rod_collision":
                    False,
                "combined_native_post_contact":
                    post0,
                "rigid_collision_vertex_ids":
                    [3] if post0 else [],
                "hidden_rod_collision_rope_edge_ids":
                    [],
                "rigid_max_penetration_m":
                    0.001 if post0 else 0.0,
                "hidden_rod_max_constraint_penetration_m":
                    0.0,
            },
            {
                "rigid_post_collision":
                    post1,
                "hidden_rod_collision":
                    False,
                "combined_native_post_contact":
                    post1,
                "rigid_collision_vertex_ids":
                    [8] if post1 else [],
                "hidden_rod_collision_rope_edge_ids":
                    [],
                "rigid_max_penetration_m":
                    0.002 if post1 else 0.0,
                "hidden_rod_max_constraint_penetration_m":
                    0.0,
            },
        ]
    }


def test_interval_aggregate():
    aggregate = init_interval_aggregate()
    update_interval_aggregate(
        aggregate,
        _step(post0=True),
    )
    update_interval_aggregate(
        aggregate,
        _step(post0=False),
    )

    result = finalize_interval_aggregate(
        aggregate
    )

    assert result["scene_steps"] == 2
    assert (
        result["posts"][0][
            "combined_contact_scene_step_fraction"
        ]
        == 0.5
    )
    assert (
        result["posts"][0][
            "rigid_collision_vertex_ids"
        ]
        == [3]
    )


def test_replay_locator_pb0_layout():
    assert replay_locator(
        138,
        32,
        123,
    ) == {
        "rollout_id": 138,
        "batch_index": 4,
        "env_index": 10,
        "batch_seed": 127,
    }


def _record(post0_by_time, post1_by_time=None):
    if post1_by_time is None:
        post1_by_time = {
            key: False
            for key
            in post0_by_time
        }

    samples = {}

    for time_index in sorted(
            set(post0_by_time)
            | set(post1_by_time)):
        p0 = bool(
            post0_by_time.get(
                time_index,
                False,
            )
        )
        p1 = bool(
            post1_by_time.get(
                time_index,
                False,
            )
        )

        samples[str(time_index)] = {
            "end": {
                "posts": [
                    {
                        "combined_native_post_contact": p0,
                        "rigid_post_collision": p0,
                        "hidden_rod_collision": False,
                    },
                    {
                        "combined_native_post_contact": p1,
                        "rigid_post_collision": p1,
                        "hidden_rod_collision": False,
                    },
                ]
            },
            "interval": {
                "posts": [
                    {
                        "combined_contact_scene_step_fraction":
                            1.0 if p0 else 0.0,
                    },
                    {
                        "combined_contact_scene_step_fraction":
                            1.0 if p1 else 0.0,
                    },
                ]
            },
        }

    return {
        "replay_match_within_tolerance": True,
        "samples": samples,
    }


def _candidate():
    # Old proxy says: on post 0, A contact / B non-contact.
    return {
        "rollout_a": 1,
        "rollout_b": 2,
        "time_index": 18,
        "observable_chamfer_m": 0.004,
        "ee_position_distance_m": 0.003,
        "hidden_robustness": {
            "mismatch_posts": [
                {
                    "post_index": 0,
                    "clearance_a_m": 0.002,
                    "clearance_b_m": 0.004,
                }
            ]
        },
    }


def test_native_mismatch_matching_proxy_is_confirmed():
    records = {
        1: _record(
            {17: True, 18: True, 19: True}
        ),
        2: _record(
            {17: False, 18: False, 19: False}
        ),
    }

    result = pair_confirmation(
        _candidate(),
        records,
        contact_threshold_m=0.003,
        target_sample=18,
        core_samples=[17, 18, 19],
    )

    assert (
        result[
            "native_mismatch_confirmed_at_target"
        ]
        is True
    )
    assert result["stable_core_support"] is True
    assert (
        result[
            "proxy_orientation_agreement_post_indices"
        ]
        == [0]
    )


def test_native_mismatch_opposite_proxy_direction_is_still_confirmed():
    # Old proxy orientation is A contact / B non-contact.
    # Native simulator says the reverse. This is still a genuine native
    # current-state mismatch; proxy agreement is only diagnostic.
    records = {
        1: _record(
            {17: False, 18: False, 19: False}
        ),
        2: _record(
            {17: True, 18: True, 19: True}
        ),
    }

    result = pair_confirmation(
        _candidate(),
        records,
        contact_threshold_m=0.003,
        target_sample=18,
        core_samples=[17, 18, 19],
    )

    assert (
        result[
            "native_mismatch_confirmed_at_target"
        ]
        is True
    )
    assert result["stable_core_support"] is True
    assert (
        result[
            "proxy_orientation_disagreement_post_indices"
        ]
        == [0]
    )


def test_native_mismatch_on_non_proxy_post_is_confirmed():
    # Proxy implicated post 0, but the actual simulator-native difference is
    # on post 1. Hidden-state qualification must inspect both published posts.
    records = {
        1: _record(
            {17: False, 18: False, 19: False},
            {17: True, 18: True, 19: True},
        ),
        2: _record(
            {17: False, 18: False, 19: False},
            {17: False, 18: False, 19: False},
        ),
    }

    result = pair_confirmation(
        _candidate(),
        records,
        contact_threshold_m=0.003,
        target_sample=18,
        core_samples=[17, 18, 19],
    )

    assert (
        result[
            "native_mismatch_confirmed_at_target"
        ]
        is True
    )
    assert result[
        "native_mismatch_post_indices"
    ] == [1]
    assert result["stable_core_support"] is True


def test_stable_core_support_requires_same_native_direction():
    # Mismatch exists at every core sample, but A/B direction flips.
    records = {
        1: _record(
            {17: True, 18: False, 19: True}
        ),
        2: _record(
            {17: False, 18: True, 19: False}
        ),
    }

    result = pair_confirmation(
        _candidate(),
        records,
        contact_threshold_m=0.003,
        target_sample=18,
        core_samples=[17, 18, 19],
    )

    assert (
        result[
            "native_mismatch_confirmed_at_target"
        ]
        is True
    )
    assert result["stable_core_support"] is False
    assert (
        result["post_results"][0][
            "persistent_core_mismatch_any_direction"
        ]
        is True
    )


def test_equal_native_states_are_not_confirmed_even_if_proxy_differs():
    records = {
        1: _record(
            {17: True, 18: True, 19: True}
        ),
        2: _record(
            {17: True, 18: True, 19: True}
        ),
    }

    result = pair_confirmation(
        _candidate(),
        records,
        contact_threshold_m=0.003,
        target_sample=18,
        core_samples=[17, 18, 19],
    )

    assert (
        result[
            "native_mismatch_confirmed_at_target"
        ]
        is False
    )
    assert (
        result["status"]
        == "NATIVE_POST_CONTACT_MISMATCH_NOT_CONFIRMED"
    )


def test_replay_invalid_blocks_native_confirmation():
    records = {
        1: _record(
            {17: True, 18: True, 19: True}
        ),
        2: _record(
            {17: False, 18: False, 19: False}
        ),
    }
    records[1][
        "replay_match_within_tolerance"
    ] = False

    result = pair_confirmation(
        _candidate(),
        records,
        contact_threshold_m=0.003,
        target_sample=18,
        core_samples=[17, 18, 19],
    )

    assert (
        result[
            "native_mismatch_confirmed_at_target"
        ]
        is False
    )
    assert (
        result["status"]
        == "REPLAY_ALIGNMENT_INVALID"
    )



def _pair_result(
        *,
        confirmed=True,
        stable=True,
        chamfer=0.004,
        ee=0.003):
    return {
        "native_mismatch_confirmed_at_target":
            bool(confirmed),
        "stable_core_support":
            bool(stable),
        "observable_chamfer_m":
            float(chamfer),
        "ee_position_distance_m":
            float(ee),
    }


def test_global_alignment_failure_forces_empty_confirmed_shortlist():
    shortlist = build_confirmed_shortlist(
        [
            _pair_result(
                confirmed=True,
                stable=True,
            )
        ],
        10,
        global_replay_alignment_valid=False,
    )

    assert shortlist[
        "global_replay_alignment_valid"
    ] is False
    assert shortlist[
        "blocked_reason"
    ] == "PB0T_TARGETED_REPLAY_ALIGNMENT_FAILED"
    assert shortlist["candidates"] == []


def test_global_alignment_success_allows_confirmed_shortlist():
    shortlist = build_confirmed_shortlist(
        [
            _pair_result(
                confirmed=True,
                stable=False,
                chamfer=0.005,
            ),
            _pair_result(
                confirmed=True,
                stable=True,
                chamfer=0.006,
            ),
            _pair_result(
                confirmed=False,
                stable=True,
                chamfer=0.001,
            ),
        ],
        10,
        global_replay_alignment_valid=True,
    )

    assert shortlist[
        "global_replay_alignment_valid"
    ] is True
    assert len(
        shortlist["candidates"]
    ) == 2
    assert shortlist[
        "candidates"
    ][0][
        "stable_core_support"
    ] is True
