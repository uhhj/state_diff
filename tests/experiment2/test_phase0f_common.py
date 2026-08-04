import copy

import numpy as np
import pytest

from scripts.experiment2.phase0.phase0f_common import (
    oracle_contact_feature,
    rank_phase0f,
    sensor_contact_feature,
    summarize_phase0f_candidate,
)


def trace():
    return {
        "phase": np.asarray(["no_action", "preload", "preload"], dtype="U16"),
        "contact_force_norm": np.asarray([0.0, 1.0, 2.0]),
        "contact_max_force_norm": np.asarray([0.0, 0.5, 1.5]),
        "contact_active_beads": np.asarray([0, 1, 2]),
        "contact_mean_speed": np.asarray([0.0, 0.1, 0.2]),
        "sensor_joint_motor_torque": np.asarray(
            [[0, 0], [1, 2], [2, 4]], dtype=np.float64
        ),
        "sensor_joint_motor_torque_norm": np.asarray(
            [0.0, np.sqrt(5), np.sqrt(20)]
        ),
        "sensor_suction_force_norm": np.asarray([0.0, 3.0, 4.0]),
        "sensor_suction_torque_norm": np.asarray([0.0, 1.0, 2.0]),
        "sensor_grasp_active": np.asarray([0, 1, 1]),
        "sensor_constraint_available": np.asarray([0, 1, 1]),
    }


def test_oracle_feature():
    value = oracle_contact_feature(trace(), 100, 1)
    assert value.shape == (6,)
    assert value[0] == pytest.approx(0.03)


def test_sensor_feature():
    value = sensor_contact_feature(trace(), 100, 1)
    assert value.shape == (13,)
    assert np.all(np.isfinite(value))


def targets():
    return {
        "max_initial_abs_xy": 1e-12,
        "max_arm_jump": 1e-12,
        "max_median_no_action_drift": 0.0015,
        "max_median_preload_visible_difference": 0.004,
        "min_median_contact_impulse_gap": 0.02,
        "min_median_main_branch_ade": 0.004,
        "min_median_main_branch_fde": 0.01,
        "min_median_branch_amplification": 2.0,
        "max_grouped_vision_accuracy": 0.625,
        "min_grouped_sensor_accuracy": 0.75,
        "min_grouped_oracle_accuracy": 0.875,
        "min_sensor_over_vision_margin": 0.125,
        "min_seed_pass_fraction": 0.625,
    }


def pair_row():
    return {
        "action_hash_match": True,
        "free_base_state_hash_match": True,
        "hidden_base_state_hash_match": True,
        "free_hidden_initial_hash_match": True,
        "max_initial_state_difference": 0.0,
        "free_arm_max_abs_jump": 0.0,
        "hidden_arm_max_abs_jump": 0.0,
        "free_no_action_drift": 0.0001,
        "hidden_no_action_drift": 0.0001,
        "preload_end_max_abs_xy": 0.002,
        "contact_impulse_gap": 0.04,
        "main_branch_ade": 0.008,
        "main_branch_fde": 0.02,
        "branch_amplification": 10.0,
    }


def classifier(accuracy):
    return {"accuracy": accuracy, "groups": ["a", "b", "c"]}


def test_formal_sensor_and_vision_checks_control_eligibility():
    rows = [copy.deepcopy(pair_row()) for _ in range(8)]
    candidate = {
        "id": "native",
        "action": {"protocol": "planar_microprobe"},
        "friction": {"mechanism": "native_segment"},
    }
    eligible = summarize_phase0f_candidate(
        "native", candidate, rows, targets(),
        classifier(0.5), classifier(0.5),
        classifier(0.875), classifier(1.0),
    )
    assert eligible["eligible"]

    weak_sensor = summarize_phase0f_candidate(
        "weak", candidate, rows, targets(),
        classifier(0.5), classifier(0.5),
        classifier(0.625), classifier(1.0),
    )
    assert not weak_sensor["eligible"]
    assert not weak_sensor["checks"]["formal_sensor_classifier"]

    leaky_vision = summarize_phase0f_candidate(
        "leaky", candidate, rows, targets(),
        classifier(0.75), classifier(0.5),
        classifier(1.0), classifier(1.0),
    )
    assert not leaky_vision["eligible"]
    assert not leaky_vision["checks"]["vision_screen"]


def test_ranking_prefers_eligible_candidate():
    rows = [copy.deepcopy(pair_row()) for _ in range(8)]
    candidate = {
        "id": "native",
        "action": {"protocol": "planar_microprobe"},
        "friction": {"mechanism": "native_segment"},
    }
    eligible = summarize_phase0f_candidate(
        "eligible", candidate, rows, targets(),
        classifier(0.5), classifier(0.5),
        classifier(0.875), classifier(1.0),
    )
    review = copy.deepcopy(eligible)
    review["candidate_id"] = "review"
    review["eligible"] = False
    review["score"] += 100.0
    assert rank_phase0f([review, eligible])[0]["candidate_id"] == "eligible"
