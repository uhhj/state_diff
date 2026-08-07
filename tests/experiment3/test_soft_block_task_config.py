import copy

import pytest

from state_diff.env.block_pushing.soft_block_task_config import (
    load_hlf_sbp_config, validate_hlf_sbp_config)

CONFIG = "configs/experiment3/hlf_sbp_phase0c_dynamics.json"
HIGHRATE_CONFIG = "configs/experiment3/hlf_sbp_phase0c_r1_highrate.json"


def test_active_config_contract():
    config = load_hlf_sbp_config(CONFIG)
    assert config["physics"]["microsteps_per_outer"] == 8
    assert config["physics"]["outer_timestep_s"] == pytest.approx(1 / 240)
    assert config["state"]["state_dim"] == 74
    assert config["sensor"]["sensor_dim"] == 45
    assert config["soft_block"]["material_model"] == "kelvin_voigt"


@pytest.mark.parametrize("path,value", [
    (("physics", "microsteps_per_outer"), 4),
    (("state", "state_dim"), 98),
    (("sensor", "sensor_dim"), 51),
    (("soft_block", "material_model"), "p2p_legacy")])
def test_active_config_rejects_superseded_modes(path, value):
    config = load_hlf_sbp_config(CONFIG)
    invalid = copy.deepcopy(config); invalid[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_hlf_sbp_config(invalid)


def test_highrate_sensor_window_matches_policy_interval():
    config = load_hlf_sbp_config(HIGHRATE_CONFIG)
    assert config["sensor"]["outer_rate_recording"] is True
    assert config["sensor"]["outer_samples_per_policy"] == 24
    invalid = copy.deepcopy(config)
    invalid["sensor"]["outer_samples_per_policy"] = 23
    with pytest.raises(ValueError, match="outer sensor window length mismatch"):
        validate_hlf_sbp_config(invalid)
