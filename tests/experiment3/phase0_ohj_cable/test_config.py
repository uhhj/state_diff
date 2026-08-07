import json
from pathlib import Path

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
