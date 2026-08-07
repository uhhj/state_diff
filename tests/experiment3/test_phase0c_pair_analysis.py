import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.analyze_pair import (
    evaluate_pair_arrays)
from scripts.experiment3.phase0c_hidden_dynamics.common import load_config


def fixture(sensor=True, state=True, repeat_gap=0.0):
    n = 6
    base = {
        "policy_step": np.arange(n),
        "phase": np.asarray(["initial", "no_action", "no_action", "probe", "test", "post_test"]),
        "state": np.zeros((n, 74)), "contact_sensor": np.zeros((n, 45)),
        "robot_proprio_extended": np.zeros((n, 24)),
        "goal_distance": np.ones(n), "spring_cap_count": np.zeros(n),
        "spring_evaluation_count": np.full(n, 100),
        "oracle_patch_tangential_force": np.zeros(n),
        "oracle_patch_mean_slip_speed": np.zeros(n),
        "joint_target": np.zeros((10, 6)), "ee_target": np.zeros((10, 3)),
        "command_action": np.zeros((2, 2)),
        "command_phase": np.asarray(["probe"] * 10)}
    low = {key: value.copy() for key, value in base.items()}
    high = {key: value.copy() for key, value in base.items()}
    repeat = {key: value.copy() for key, value in base.items()}
    if sensor:
        high["contact_sensor"][3:, 0] = .03
    if state:
        high["state"][4:] = .001
    if repeat_gap:
        repeat["state"][4:] = repeat_gap
    high["oracle_patch_tangential_force"][3:] = 1.0
    return low, high, repeat


def evaluate(**kwargs):
    config = load_config("configs/experiment3/hlf_sbp_phase0c_dynamics.json")
    return evaluate_pair_arrays(*fixture(**kwargs), config,
                                {"low_mu": .2, "high_mu": 1.6})


def test_sensor_early_state_later_completes_without_deformation_gate():
    result = evaluate()
    assert result["verdict"] == "PHASE0C_PAIR_COMPLETE"
    assert result["t_physics_separable"] < result["t_state_divergence"]
    assert result["deformation_diagnostics"]["used_as_hard_gate"] is False


def test_state_without_formal_sensor_is_not_established():
    assert evaluate(sensor=False)["verdict"] == "PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED"


def test_cross_gap_below_repeat_floor_is_not_established():
    assert evaluate(repeat_gap=.0008)["verdict"] == "PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED"
