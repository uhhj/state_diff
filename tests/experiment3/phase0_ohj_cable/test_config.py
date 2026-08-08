import json
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_ohj_cable.common import load_config


CONFIG = "configs/experiment3/phase0/ohj_cable_phase0d.json"


def test_phase0d_config_is_minimal_ohj_contract():
    config = load_config(CONFIG)
    assert config["conditions"] == ["free", "jam_right"]
    assert config["state"]["state_dim"] == 51
    assert config["sensor"]["sensor_dim"] == 6
    assert config["sensor"]["sampling_hz"] == 240
    assert [row["id"] for row in config["control_relevance"]["candidates"]] == [
        "straight", "left_release", "right_release"]


def _load(name):
    path = (Path(__file__).resolve().parents[3] / "configs" / "experiment3"
            / "phase0" / name)
    return json.loads(path.read_text(encoding="utf-8"))


def test_r1_only_extends_post_probe_recovery():
    base = _load("ohj_cable_phase0d.json")
    r1 = _load("ohj_cable_phase0d_r1_recovery.json")
    assert base["execution"]["post_probe_steps"] == 48
    assert r1["execution"]["post_probe_steps"] == 240
    assert (r1["motion"]["probe_delta_xyz_m"]
            == base["motion"]["probe_delta_xyz_m"]
            == [0.002, 0.0, 0.0])
    assert (r1["geometry"]["jam_surface_clearance_m"]
            == base["geometry"]["jam_surface_clearance_m"] == 0.0005)
    assert (r1["analysis"]["post_probe_visible_rmse_max_m"]
            == base["analysis"]["post_probe_visible_rmse_max_m"] == 0.0015)


def test_r2_only_reduces_probe_amplitude():
    r1 = _load("ohj_cable_phase0d_r1_recovery.json")
    r2 = _load("ohj_cable_phase0d_r2_probe1mm.json")
    assert r1["motion"]["probe_delta_xyz_m"] == [0.002, 0.0, 0.0]
    assert r2["motion"]["probe_delta_xyz_m"] == [0.001, 0.0, 0.0]
    assert (r2["execution"]["post_probe_steps"]
            == r1["execution"]["post_probe_steps"] == 240)
    assert (r2["geometry"]["jam_surface_clearance_m"]
            == r1["geometry"]["jam_surface_clearance_m"] == 0.0005)
    assert (r2["motion"]["probe_forward_steps"]
            == r1["motion"]["probe_forward_steps"])
    assert (r2["motion"]["probe_hold_steps"]
            == r1["motion"]["probe_hold_steps"])
    assert (r2["motion"]["probe_return_steps"]
            == r1["motion"]["probe_return_steps"])
    assert (r2["analysis"]["post_probe_visible_rmse_max_m"]
            == r1["analysis"]["post_probe_visible_rmse_max_m"] == 0.0015)
    assert (r2["analysis"]["sensor_normalized_gap_min"]
            == r1["analysis"]["sensor_normalized_gap_min"] == 1.0)
    assert (r2["analysis"]["future_vs_repeat_multiplier"]
            == r1["analysis"]["future_vs_repeat_multiplier"] == 3.0)


def test_r3_keeps_r2_physics_and_enables_hold_repair():
    r2 = _load("ohj_cable_phase0d_r2_probe1mm.json")
    r3 = _load("ohj_cable_phase0d_r3_holdrepair.json")
    assert (r3["motion"]["probe_delta_xyz_m"]
            == r2["motion"]["probe_delta_xyz_m"]
            == [0.001, 0.0, 0.0])
    assert (r3["execution"]["post_probe_steps"]
            == r2["execution"]["post_probe_steps"] == 240)
    assert (r3["geometry"]["jam_surface_clearance_m"]
            == r2["geometry"]["jam_surface_clearance_m"] == 0.0005)
    assert (r3["analysis"]["post_probe_visible_rmse_max_m"]
            == r2["analysis"]["post_probe_visible_rmse_max_m"] == 0.0015)
    assert (r3["analysis"]["sensor_normalized_gap_min"]
            == r2["analysis"]["sensor_normalized_gap_min"] == 1.0)
    assert (r3["analysis"]["future_vs_repeat_multiplier"]
            == r2["analysis"]["future_vs_repeat_multiplier"] == 3.0)
    assert r3["execution"]["reassert_canonical_hold_after_restore"] is True


def test_r4_only_reduces_jam_clearance():
    r3 = _load("ohj_cable_phase0d_r3_holdrepair.json")
    r4 = _load("ohj_cable_phase0d_r4_clearance025.json")
    assert r3["geometry"]["jam_surface_clearance_m"] == 0.0005
    assert r4["geometry"]["jam_surface_clearance_m"] == 0.00025
    assert (r4["motion"]["probe_delta_xyz_m"]
            == r3["motion"]["probe_delta_xyz_m"]
            == [0.001, 0.0, 0.0])
    assert (r4["execution"]["post_probe_steps"]
            == r3["execution"]["post_probe_steps"] == 240)
    assert r4["execution"]["reassert_canonical_hold_after_restore"] is True
    assert (r4["analysis"]["post_probe_visible_rmse_max_m"]
            == r3["analysis"]["post_probe_visible_rmse_max_m"] == 0.0015)
    assert (r4["analysis"]["sensor_normalized_gap_min"]
            == r3["analysis"]["sensor_normalized_gap_min"] == 1.0)
    assert (r4["analysis"]["future_visible_rmse_min_m"]
            == r3["analysis"]["future_visible_rmse_min_m"] == 0.005)


def test_r5_changes_sensor_not_task_physics():
    r4 = _load("ohj_cable_phase0d_r4_clearance025.json")
    r5 = _load("ohj_cable_phase0d_r5_tactile9d.json")
    assert r5["motion"] == r4["motion"]
    assert r5["geometry"] == r4["geometry"]
    assert r5["state"] == r4["state"]
    assert r5["analysis"] == r4["analysis"]
    assert r5["execution"]["post_probe_steps"] == 240
    assert r5["execution"]["reassert_canonical_hold_after_restore"] is True
    assert r4["sensor"]["sensor_dim"] == 6
    assert r5["sensor"]["sensor_dim"] == 9
    assert r5["sensor"]["trace_field"] == "formal_sensor"
    assert len(r5["sensor"]["channel_floor"]) == 9


def test_r6_only_adds_spatial_tactile():
    r5 = _load("ohj_cable_phase0d_r5_tactile9d.json")
    r6 = _load("ohj_cable_phase0d_r6_tactile4patch18d.json")
    assert r6["motion"] == r5["motion"]
    assert r6["geometry"] == r5["geometry"]
    assert r6["execution"] == r5["execution"]
    assert r6["state"] == r5["state"]
    assert r6["analysis"] == r5["analysis"]
    assert r5["sensor"]["sensor_dim"] == 9
    assert r6["sensor"]["sensor_dim"] == 18
    assert r6["sensor"]["trace_field"] == "formal_sensor_spatial"
    assert r6["sensor"]["tactile_patch_count"] == 4
    assert r6["sensor"]["tactile_patch_layout"] == "tip_xy_quadrants"
    assert len(r6["sensor"]["channel_floor"]) == 18


def test_r7_only_adds_spatial_repeat_corrected_load_audit():
    r6 = _load("ohj_cable_phase0d_r6_tactile4patch18d.json")
    r7 = _load("ohj_cable_phase0d_r7_spatial_loadpath.json")
    assert r7["execution"] == r6["execution"]
    assert r7["geometry"] == r6["geometry"]
    assert r7["motion"] == r6["motion"]
    assert r7["state"] == r6["state"]
    assert r7["sensor"] == r6["sensor"]
    assert r7["analysis"] == r6["analysis"]
    diagnostic = r7["diagnostic"]
    assert diagnostic["mode"] == (
        "spatial_repeat_corrected_load_path_information")
    assert (diagnostic["branch_vs_repeat_multiplier"]
            == r7["analysis"]["future_vs_repeat_multiplier"] == 3.0)
    assert diagnostic["mechanical_reference_floor_n"] == 0.05
    assert "oracle_internal_constraint_segments" not in diagnostic


def test_r8_doubles_excursion_at_fixed_probe_speed():
    r7 = _load("ohj_cable_phase0d_r7_spatial_loadpath.json")
    r8 = _load("ohj_cable_phase0d_r8_probe2mm_fixedspeed.json")
    assert r8["execution"] == r7["execution"]
    assert r8["geometry"] == r7["geometry"]
    assert r8["state"] == r7["state"]
    assert r8["sensor"] == r7["sensor"]
    assert r8["diagnostic"] == r7["diagnostic"]
    assert r8["analysis"] == r7["analysis"]
    assert r8["control_relevance"] == r7["control_relevance"]
    r7_motion = dict(r7["motion"])
    r8_motion = dict(r8["motion"])
    r7_delta = r7_motion.pop("probe_delta_xyz_m")
    r8_delta = r8_motion.pop("probe_delta_xyz_m")
    r7_forward = r7_motion.pop("probe_forward_steps")
    r8_forward = r8_motion.pop("probe_forward_steps")
    r7_return = r7_motion.pop("probe_return_steps")
    r8_return = r8_motion.pop("probe_return_steps")
    assert r7_motion == r8_motion
    assert r7_delta == [0.001, 0.0, 0.0]
    assert r8_delta == [0.002, 0.0, 0.0]
    assert r7_forward == r7_return == 24
    assert r8_forward == r8_return == 48
    hz = r8["execution"]["hz"]
    r7_speed = r7_delta[0] / (r7_forward / hz)
    r8_speed = r8_delta[0] / (r8_forward / hz)
    assert abs(r7_speed - 0.01) < 1e-12
    assert abs(r8_speed - r7_speed) < 1e-12
    repair = r8["repair"]
    assert repair["mode"] == (
        "single_fixed_speed_probe_excursion_amplification")
    assert repair["stop_after_this_trial"] is True
    assert repair["baseline"]["result_sha"] == (
        "aa5600408d835f344b580f155e61780c737930d1")


def test_r9_changes_only_probe_direction_from_r7():
    r7 = _load("ohj_cable_phase0d_r7_spatial_loadpath.json")
    r9 = _load("ohj_cable_phase0d_r9_contactloading_minusy1mm.json")
    assert r9["execution"] == r7["execution"]
    assert r9["geometry"] == r7["geometry"]
    assert r9["state"] == r7["state"]
    assert r9["sensor"] == r7["sensor"]
    assert r9["diagnostic"] == r7["diagnostic"]
    assert r9["analysis"] == r7["analysis"]
    assert r9["control_relevance"] == r7["control_relevance"]
    r7_motion = dict(r7["motion"])
    r9_motion = dict(r9["motion"])
    r7_delta = np.asarray(
        r7_motion.pop("probe_delta_xyz_m"), dtype=np.float64)
    r9_delta = np.asarray(
        r9_motion.pop("probe_delta_xyz_m"), dtype=np.float64)
    assert r7_motion == r9_motion
    np.testing.assert_allclose(r7_delta, [0.001, 0.0, 0.0])
    np.testing.assert_allclose(r9_delta, [0.0, -0.001, 0.0])
    assert np.isclose(np.linalg.norm(r7_delta), 0.001)
    assert np.isclose(np.linalg.norm(r9_delta), 0.001)
    r7_direction = r7_delta / np.linalg.norm(r7_delta)
    r9_direction = r9_delta / np.linalg.norm(r9_delta)
    assert np.isclose(np.dot(r7_direction, r9_direction), 0.0)
    hz = r9["execution"]["hz"]
    r7_speed = np.linalg.norm(r7_delta) / (
        r7["motion"]["probe_forward_steps"] / hz)
    r9_speed = np.linalg.norm(r9_delta) / (
        r9["motion"]["probe_forward_steps"] / hz)
    assert np.isclose(r7_speed, 0.01)
    assert np.isclose(r9_speed, r7_speed)
    repair = r9["repair"]
    assert repair["mode"] == "single_orthogonal_contact_loading_probe"
    assert repair["stop_after_this_trial"] is True
    assert repair["baseline"]["result_sha"] == (
        "aa5600408d835f344b580f155e61780c737930d1")


def test_r10_only_extends_r9_same_action_future_horizon():
    r9 = _load("ohj_cable_phase0d_r9_contactloading_minusy1mm.json")
    r10 = _load("ohj_cable_phase0d_r10_future1s.json")
    for key in (
            "conditions", "geometry", "motion", "state", "sensor",
            "diagnostic", "repair", "analysis", "control_relevance"):
        assert r10[key] == r9[key]
    r9_execution = dict(r9["execution"])
    r10_execution = dict(r10["execution"])
    assert r9_execution.pop("post_test_steps") == 48
    assert r10_execution.pop("post_test_steps") == 240
    assert r10_execution == r9_execution
    assert r10["motion"]["straight_pull_steps"] == 96
    assert r10["motion"]["probe_delta_xyz_m"] == [0.0, -0.001, 0.0]
