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
