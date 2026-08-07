import copy

import numpy as np

from scripts.experiment3.phase0_ohj_cable.analyze_pair import (
    evaluate_pair, probe_contact_diagnostics, recovery_curve,
    repeatability_by_phase)
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
        "gripper_surface_tactile_force": np.zeros((n, 3)),
        "gripper_surface_contact_count": np.zeros(n),
        "formal_sensor": np.zeros((n, 9)),
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
    jam["oracle_latch_contact_count"][1:241:2] = 1
    repeat["statediff_state"][:] = .0002
    config = load_config(
        "configs/experiment3/phase0/ohj_cable_phase0d_r1_recovery.json")
    recovery = recovery_curve(free, jam, repeat, config)
    assert len(recovery["checkpoints"]) == 5
    assert (recovery["final_free_jam_rmse_m"]
            < recovery["immediate_free_jam_rmse_m"])
    assert recovery["recovery_fraction"] > .5
    assert recovery["final_branch_excess_over_repeat_m"] > 0.0
    assert np.isclose(
        recovery["final_repeat_fraction_of_free_jam"], 0.2)
    assert np.isclose(
        recovery["final_branch_excess_fraction_of_free_jam"], 0.8)
    assert np.isclose(
        recovery["post_probe_jam_latch_contact_fraction"], 0.5)


def test_repeatability_by_phase_localizes_divergence():
    phase = np.asarray(
        ["no_action"] * 3 + ["probe_forward"] * 3 + ["post_probe"] * 3)
    free = {
        "phase": phase,
        "statediff_state": np.zeros((9, 51)),
    }
    repeat = {
        "phase": phase,
        "statediff_state": np.zeros((9, 51)),
    }
    repeat["statediff_state"][3:6, :48] = 0.003
    config = load_config(
        "configs/experiment3/phase0/ohj_cable_phase0d_r3_holdrepair.json")
    result = repeatability_by_phase(free, repeat, config)
    assert (result["first_phase_above_equivalence_threshold"]
            == "probe_forward")
    rows = {row["phase"]: row for row in result["phases"]}
    assert rows["no_action"]["peak_51d_rmse_m"] == 0.0
    assert rows["probe_forward"]["peak_keypoint_rmse_m"] > 0.0015


def test_probe_contact_diagnostics_are_phase_local():
    phase = np.asarray(
        ["no_action"] * 2 + ["probe_forward"] * 2
        + ["probe_hold"] * 2 + ["probe_return"] * 2
        + ["post_probe"] * 2 + ["test_pull"] * 2)
    n = len(phase)
    free = {
        "phase": phase,
        "oracle_latch_contact_count": np.zeros(n),
        "oracle_latch_contact_force": np.zeros(n),
    }
    jam = {
        "phase": phase,
        "oracle_latch_contact_count": np.zeros(n),
        "oracle_latch_contact_force": np.zeros(n),
    }
    jam["oracle_latch_contact_count"][3] = 1
    jam["oracle_latch_contact_count"][5] = 1
    jam["oracle_latch_contact_force"][3] = 0.4
    jam["oracle_latch_contact_force"][5] = 0.7
    jam["oracle_latch_contact_count"][10:] = 1
    jam["oracle_latch_contact_force"][10:] = 3.0
    result = probe_contact_diagnostics(free, jam)
    assert result["free_probe_contact_samples"] == 0
    assert result["jam_probe_contact_samples"] == 2
    assert np.isclose(result["jam_probe_contact_fraction"], 2.0 / 6.0)
    assert np.isclose(result["jam_probe_peak_latch_force_n"], 0.7)


def test_r5_gate3_can_be_triggered_by_surface_tactile():
    free, jam, repeat, metadata = synthetic()
    r5 = load_config(
        "configs/experiment3/phase0/ohj_cable_phase0d_r5_tactile9d.json")
    for branch in (free, jam, repeat):
        branch["formal_sensor"] = np.zeros(
            (len(branch["phase"]), 9), dtype=np.float64)
        branch["gripper_surface_contact_count"] = np.ones(
            len(branch["phase"]), dtype=np.int64)
    probe_idx = np.flatnonzero(np.isin(
        jam["phase"], ["probe_forward", "probe_hold", "probe_return"]))
    jam["formal_sensor"][probe_idx[:4], 6] = 0.2
    metrics = evaluate_pair(free, jam, repeat, metadata, r5)
    assert metrics["gates"]["sensor_observability"] is True
    assert metrics["grasp_only_peak_fused_gap"] == 0.0
    assert metrics["tactile_only_peak_fused_gap"] >= 1.0
    assert metrics["sensor_trigger"]["channel"] == "tactile_Fx"
