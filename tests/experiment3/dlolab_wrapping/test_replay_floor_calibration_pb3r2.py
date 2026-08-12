import copy

import numpy as np

from scripts.experiment3.dlolab_wrapping.replay_floor_calibration_pb3r2 import (
    attempt_paths,
    derive_calibration_set,
    derive_resume_policy,
    evenly_spaced_positions,
    full_alignment_metrics,
    quantile_summary,
    sample_finiteness,
    validate_reusable_worker_artifact,
)


def test_evenly_spaced_positions_cover_endpoints():
    assert evenly_spaced_positions(21, 5) == [
        0, 5, 10, 15, 20
    ]


def _config():
    return {
        "calibration_selection": {
            "batch_indices": [0, 1, 2, 3],
            "states_per_batch": 5,
            "total_states": 20,
            "time_pattern_even_batch": [
                13, 20, 13, 20, 13
            ],
            "time_pattern_odd_batch": [
                20, 13, 20, 13, 20
            ],
            "selection_rule": "test rule",
        }
    }


def _frozen():
    rollout_id = np.arange(
        128,
        dtype=np.int32,
    )
    batch_index = (
        rollout_id // 32
    ).astype(np.int16)
    env_index = (
        rollout_id % 32
    ).astype(np.int16)
    batch_seed = (
        123 + batch_index
    ).astype(np.int32)

    frozen = {
        "rollout_id":
            rollout_id,

        "batch_index":
            batch_index,

        "env_index":
            env_index,

        "batch_seed":
            batch_seed,

        "official_rollout_valid":
            np.ones(
                128,
                dtype=bool,
            ),

        # State values are deliberately present but selection must ignore them.
        "rope_xyz":
            np.random.default_rng(
                1
            ).normal(
                size=(
                    128,
                    21,
                    50,
                    3,
                )
            ),
    }

    return frozen


def test_calibration_set_is_20_unique_and_formal_disjoint():
    frozen = _frozen()
    formal = [
        2, 5, 6, 10, 18, 24,
        36, 39, 46, 55, 61,
        69, 72, 73, 78, 83,
        97, 105, 109, 124,
    ]

    row = derive_calibration_set(
        _config(),
        frozen,
        formal,
    )

    ids = [
        state[
            "rollout_id"
        ]
        for state in row[
            "states"
        ]
    ]

    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert set(ids).isdisjoint(
        formal
    )
    assert row["time_counts"] == {
        "13": 10,
        "20": 10,
    }

    for batch in range(4):
        assert sum(
            state[
                "batch_index"
            ] == batch
            for state in row[
                "states"
            ]
        ) == 5


def test_calibration_selection_is_independent_of_rope_state_values():
    frozen_a = _frozen()
    frozen_b = copy.deepcopy(
        frozen_a
    )

    frozen_b[
        "rope_xyz"
    ] = np.random.default_rng(
        99
    ).normal(
        loc=1000.0,
        scale=500.0,
        size=frozen_b[
            "rope_xyz"
        ].shape,
    )

    formal = [
        2, 5, 6, 10, 18, 24,
        36, 39, 46, 55, 61,
        69, 72, 73, 78, 83,
        97, 105, 109, 124,
    ]

    a = derive_calibration_set(
        _config(),
        frozen_a,
        formal,
    )
    b = derive_calibration_set(
        _config(),
        frozen_b,
        formal,
    )

    assert a == b


def test_official_invalid_rollout_is_not_selected():
    frozen = _frozen()
    frozen[
        "official_rollout_valid"
    ][0] = False

    row = derive_calibration_set(
        _config(),
        frozen,
        [],
    )

    ids = {
        state[
            "rollout_id"
        ]
        for state in row[
            "states"
        ]
    }

    assert 0 not in ids


def test_quantile_summary_reports_requested_core_statistics():
    row = quantile_summary(
        np.arange(
            1,
            101,
            dtype=np.float64,
        )
    )

    assert row["count"] == 100
    assert np.isclose(
        row["median"],
        50.5,
    )
    assert row["max"] == 100.0
    assert row["p99"] > row["p95"]


def test_selection_uses_only_metadata_not_formal_candidate_metrics():
    frozen = _frozen()

    row = derive_calibration_set(
        _config(),
        frozen,
        [24, 109],
    )

    assert row[
        "selection_used_state_value"
    ] is False
    assert row[
        "selection_used_replay_error"
    ] is False
    assert row[
        "selection_used_future"
    ] is False
    assert row[
        "selection_used_winding"
    ] is False
    assert row[
        "selection_used_reward_force_or_sensor"
    ] is False



def _alignment_config():
    return {
        "frozen_pb3_alignment_reference": {
            "rope_max_abs_m": 5e-5,
            "ee_max_abs_m": 5e-5,
            "motor_qpos_max_abs_rad": 5e-5,
            "require_winding_index_exact": True,
        }
    }


def _live_and_frozen_for_alignment():
    rope = np.zeros(
        (50, 3),
        dtype=np.float64,
    )

    live = {
        "rope_xyz":
            rope.copy(),

        "ee1_pos":
            np.zeros(
                3,
                dtype=np.float64,
            ),

        "ee2_pos":
            np.zeros(
                3,
                dtype=np.float64,
            ),

        "motor_qpos_1":
            np.zeros(
                7,
                dtype=np.float64,
            ),

        "motor_qpos_2":
            np.zeros(
                7,
                dtype=np.float64,
            ),

        "signed_winding_turns":
            np.zeros(
                3,
                dtype=np.float64,
            ),
    }

    frozen = {
        "rope_xyz":
            rope[
                None,
                None,
            ].copy(),

        "ee1_pos":
            np.zeros(
                (1, 1, 3),
                dtype=np.float64,
            ),

        "ee2_pos":
            np.zeros(
                (1, 1, 3),
                dtype=np.float64,
            ),

        "motor_qpos_1":
            np.zeros(
                (1, 1, 7),
                dtype=np.float64,
            ),

        "motor_qpos_2":
            np.zeros(
                (1, 1, 7),
                dtype=np.float64,
            ),

        "signed_winding_turns":
            np.zeros(
                (1, 1, 3),
                dtype=np.float64,
            ),
    }

    return live, frozen


def test_target_finiteness_rejects_nonfinite_live_state():
    live, frozen = _live_and_frozen_for_alignment()
    live[
        "rope_xyz"
    ][
        17,
        1,
    ] = np.nan

    row = sample_finiteness(
        live,
        frozen,
        0,
        0,
    )

    assert row[
        "live_all_finite"
    ] is False
    assert row[
        "all_finite"
    ] is False
    assert row[
        "live_field_finite"
    ][
        "rope_xyz"
    ] is False


def test_target_finiteness_rejects_nonfinite_frozen_reference():
    live, frozen = _live_and_frozen_for_alignment()
    frozen[
        "signed_winding_turns"
    ][
        0,
        0,
        1,
    ] = np.inf

    row = sample_finiteness(
        live,
        frozen,
        0,
        0,
    )

    assert row[
        "frozen_all_finite"
    ] is False
    assert row[
        "all_finite"
    ] is False


def test_50um_coverage_is_rope_only_not_full_alignment():
    live, frozen = _live_and_frozen_for_alignment()

    # Rope is an exact match and therefore passes the pure 50 um rope
    # reference. EE deliberately fails the historical full alignment.
    live[
        "ee1_pos"
    ][0] = 1e-3

    row = full_alignment_metrics(
        live,
        frozen,
        0,
        0,
        _alignment_config(),
    )

    assert row[
        "old_50um_rope_reference_pass"
    ] is True

    assert row[
        "old_full_pb3_alignment_pass"
    ] is False


def test_50um_rope_reference_fails_only_when_rope_exceeds_limit():
    live, frozen = _live_and_frozen_for_alignment()

    live[
        "rope_xyz"
    ][
        28,
        0,
    ] = 5.1e-5

    row = full_alignment_metrics(
        live,
        frozen,
        0,
        0,
        _alignment_config(),
    )

    assert row[
        "old_50um_rope_reference_pass"
    ] is False


def _complete_worker_artifact(root, calibration_set, batch, repeat):
    states = [
        state for state in calibration_set["states"]
        if state["batch_index"] == batch
    ]
    stem = f"batch{batch}_repeat{repeat}"
    npz_path = root / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        rollout_id=np.asarray([s["rollout_id"] for s in states]),
        time_index=np.asarray([s["time_index"] for s in states]),
        rope_xyz=np.zeros((5, 50, 3), dtype=np.float32),
    )
    (root / f"{stem}.json").write_text(
        __import__("json").dumps(
            {
                "worker_verdict": "PB3R2_WORKER_COMPLETE",
                "future_steps_executed": 20,
                "target_metrics": [
                    {
                        "rollout_id": s["rollout_id"],
                        "time_index": s["time_index"],
                    }
                    for s in states
                ],
                "all_target_samples_finite": True,
            }
        ),
        encoding="utf-8",
    )


def test_reusable_artifact_requires_complete_json_npz_and_finite_ropes(tmp_path):
    calibration_set = derive_calibration_set(_config(), _frozen(), [])
    _complete_worker_artifact(tmp_path, calibration_set, 0, 0)

    valid = validate_reusable_worker_artifact(
        tmp_path, calibration_set, 0, 0
    )
    absent = validate_reusable_worker_artifact(
        tmp_path, calibration_set, 2, 1
    )

    assert valid["reusable"] is True
    assert absent == {
        "reusable": False,
        "reason": "final_artifact_missing",
    }


def test_resume_policy_reuses_seven_and_allows_only_five_new_workers(tmp_path):
    config = _config()
    config["replay"] = {"fresh_process_repeats": 3}
    calibration_set = derive_calibration_set(config, _frozen(), [])

    for batch, repeat in [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2), (2, 0)]:
        _complete_worker_artifact(
            tmp_path, calibration_set, batch, repeat
        )

    policy = derive_resume_policy(
        config, calibration_set, tmp_path
    )

    assert len(policy["reusable_workers"]) == 7
    assert policy["maximum_new_fresh_workers"] == 5
    assert [row["logical_worker_id"] for row in policy["resume_actions"]] == [
        "batch2_repeat1",
        "batch2_repeat2",
        "batch3_repeat0",
        "batch3_repeat1",
        "batch3_repeat2",
    ]
    assert policy["resume_actions"][0]["attempt_index"] == 1
    assert policy["resume_actions"][0]["action"] == "retry_once"


def test_attempt_paths_are_per_attempt_and_not_final_worker_artifacts(tmp_path):
    paths = attempt_paths(tmp_path, 2, 1, 1)

    assert paths["stage"].name == "batch2_repeat1_attempt1.stage.jsonl"
    assert paths["stdout"].name == "batch2_repeat1_attempt1.stdout.log"
    assert paths["stderr"].name == "batch2_repeat1_attempt1.stderr.log"
