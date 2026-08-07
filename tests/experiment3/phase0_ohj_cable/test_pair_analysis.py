import copy

import numpy as np

from scripts.experiment3.phase0_ohj_cable.analyze_pair import (
    evaluate_pair, recovery_curve)
from scripts.experiment3.phase0_ohj_cable.common import load_config


CONFIG = load_config("configs/experiment3/phase0/ohj_cable_phase0d.json")


def synthetic():
    phase = np.asarray(
        ["no_action"] * 6 + ["probe_forward"] * 3
        + ["probe_hold"] * 3 + ["probe_return"] * 3
        + ["post_probe"] * 3 + ["test_pull"] * 6 + ["post_test"] * 3)
    n = len(phase)
    base = {
        "physics_step": np.arange(1, n + 1), "phase": phase,
        "statediff_state": np.zeros((n, 51)),
        "formal_wrench": np.zeros((n, 6)),
        "extraction_progress_m": np.linspace(0., .04, n),
        "oracle_latch_contact_force": np.zeros(n),
        "oracle_latch_contact_count": np.zeros(n),
        "command_phase": np.asarray(["probe", "test"]),
        "command_ee_target": np.zeros((2, 3)),
        "command_joint_target": np.zeros((2, 6)),
    }
    free, jam, repeat = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    jam["formal_wrench"][7:13, 0] = .1
    jam["statediff_state"][18:, :48] = .006
    jam["oracle_latch_contact_force"][7:13] = 1.
    jam["oracle_latch_contact_count"][7:13] = 1
    metadata = {name: {
        "initial_statediff_state": np.zeros(51).tolist(),
        "post_probe_statediff_state": np.zeros(51).tolist()}
        for name in ("free", "jam_right", "free_repeat")}
    return free, jam, repeat, metadata


def test_complete_pair_has_sensor_then_future_branch():
    metrics = evaluate_pair(*synthetic(), CONFIG)
    assert metrics["verdict"] == "PHASE0D_PAIR_COMPLETE"
    assert metrics["sensor_onset_step"] == 8
    assert metrics["future_peak_visible_rmse_m"] == .006


def test_sensor_absent_fails_sensor_gate():
    free, jam, repeat, metadata = synthetic()
    jam["formal_wrench"][:] = free["formal_wrench"]
    assert evaluate_pair(free, jam, repeat, metadata, CONFIG)["verdict"] == (
        "PHASE0D_SENSOR_NOT_OBSERVABLE")


def test_post_probe_difference_fails_equivalence_first():
    free, jam, repeat, metadata = synthetic()
    metadata["jam_right"]["post_probe_statediff_state"][0] = .02
    assert evaluate_pair(free, jam, repeat, metadata, CONFIG)["verdict"] == (
        "PHASE0D_OBSERVABLE_EQUIVALENCE_FAIL")


def test_future_under_repeat_floor_fails_future_gate():
    free, jam, repeat, metadata = synthetic()
    repeat["statediff_state"][18:, :48] = .003
    assert evaluate_pair(free, jam, repeat, metadata, CONFIG)["verdict"] == (
        "PHASE0D_FUTURE_BRANCH_NOT_ESTABLISHED")


def test_recovery_curve_tracks_return_to_one_second():
    phase = np.asarray(["probe_return"] + ["post_probe"] * 240)
    free = {"phase": phase, "statediff_state": np.zeros((241, 51)),
            "oracle_latch_contact_count": np.zeros(241),
            "oracle_latch_contact_force": np.zeros(241)}
    jam = copy.deepcopy(free)
    repeat = copy.deepcopy(free)
    jam["statediff_state"][0] = .006
    jam["statediff_state"][1:25] = .004
    jam["statediff_state"][25:49] = .003
    jam["statediff_state"][49:121] = .002
    jam["statediff_state"][121:] = .001
    repeat["statediff_state"][:] = .0002
    config = load_config(
        "configs/experiment3/phase0/ohj_cable_phase0d_r1_recovery.json")
    recovery = recovery_curve(free, jam, repeat, config)
    assert len(recovery["checkpoints"]) == 5
    assert (recovery["final_free_jam_rmse_m"]
            < recovery["immediate_free_jam_rmse_m"])
    assert recovery["recovery_fraction"] > .5
    assert recovery["final_branch_excess_over_repeat_m"] > 0.0
