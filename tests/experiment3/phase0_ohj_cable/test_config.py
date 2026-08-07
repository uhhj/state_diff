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
