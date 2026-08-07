import copy

import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.analyze_highrate_pair import (
    evaluate_highrate_arrays)
from scripts.experiment3.phase0c_hidden_dynamics.common import load_config


def branches(sensor_onset=30, state_onset=42, sensor_spike=False):
    n = 72
    phase = np.asarray(["no_action"] * 24 + ["test"] * (n - 24))
    base = {
        "state": np.zeros((4, 74)),
        "goal_distance": np.zeros(4),
        "state_hr": np.zeros((n, 74)),
        "contact_sensor_hr": np.zeros((n, 45)),
        "robot_proprio_extended_hr": np.zeros((n, 24)),
        "goal_distance_hr": np.zeros(n),
        "spring_cap_count_hr": np.zeros(n),
        "spring_evaluation_count_hr": np.full(n, 100),
        "oracle_patch_tangential_force_hr": np.zeros(n),
        "oracle_patch_mean_slip_speed_hr": np.zeros(n),
        "outer_step_hr": np.arange(1, n + 1),
        "phase_hr": phase,
        "joint_target": np.zeros((n, 6)),
        "ee_target": np.zeros((n, 3)),
        "command_action": np.zeros((3, 2)),
        "command_phase": phase,
    }
    low, high, repeat = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    if state_onset is not None:
        high["state_hr"][state_onset - 1:, 72] = .01
    if sensor_onset is not None:
        if sensor_spike:
            high["contact_sensor_hr"][sensor_onset - 1, 0] = .04
        else:
            high["contact_sensor_hr"][sensor_onset - 1:, 0] = .04
    high["oracle_patch_tangential_force_hr"][24:] = .01
    return low, high, repeat


def evaluate(*args, **kwargs):
    config = load_config("configs/experiment3/hlf_sbp_phase0c_r1_highrate.json")
    return evaluate_highrate_arrays(*branches(*args, **kwargs), config,
                                    {"low_mu": .2, "high_mu": 1.6})


def test_sensor_before_state_completes_observability():
    result = evaluate(sensor_onset=30, state_onset=42)
    assert result["verdict"] == "PHASE0C_R1_SENSOR_OBSERVABILITY_COMPLETE"
    assert result["sensor_onset_outer_step"] == 30
    assert result["state_onset_outer_step"] == 42
    assert result["sensor_lead_outer_steps"] == 12
    assert result["peak_deformable_rmse_m"] == 0.0
    assert result["peak_ee_xy_rmse_m"] > 0.0


def test_sensor_after_state_is_still_late():
    assert evaluate(sensor_onset=50, state_onset=42)["verdict"] == (
        "PHASE0C_R1_SENSOR_STILL_LATE")


def test_absent_sensor_is_not_separable():
    assert evaluate(sensor_onset=None, state_onset=42)["verdict"] == (
        "PHASE0C_R1_SENSOR_NOT_SEPARABLE")


def test_single_outer_step_spike_does_not_trigger_onset():
    result = evaluate(sensor_onset=30, state_onset=42, sensor_spike=True)
    assert result["sensor_onset_outer_step"] is None
    assert result["verdict"] == "PHASE0C_R1_SENSOR_NOT_SEPARABLE"
