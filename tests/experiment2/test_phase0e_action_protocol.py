import copy

import numpy as np
import pytest

from scripts.experiment2.phase0.run_hidden_friction_pairs import (
    _environment_action,
    generate_action_script,
)


def base_config():
    return {
        "workspace_bounds": {"x": [0.0, 1.0], "y": [-0.5, 0.5]},
        "action": {
            "pick_z": 0.01,
            "preload_distance": 0.02,
            "main_pull_distance": 0.10,
        },
    }


def beads():
    return np.asarray([[0.4, 0.0, 0.01], [0.6, 0.0, 0.01]])


def position(action, name):
    return np.asarray(action[name]["position"], dtype=np.float64)


def test_default_direct_has_two_actions_and_continuous_main_start():
    actions = generate_action_script(base_config(), beads())
    assert [row["primitive"] for row in actions] == ["pick_place", "pick_place"]
    assert [row["phase"] for row in actions] == ["preload", "main_pull"]
    np.testing.assert_array_equal(position(actions[1], "pose0"), position(actions[0], "pose1"))


def test_probe_return_uses_original_start_for_main_pull():
    config = copy.deepcopy(base_config())
    config["action"].update({"protocol": "probe_return", "probe_hold_steps": 17})
    actions = generate_action_script(config, beads())
    assert [row["primitive"] for row in actions] == ["pick_probe_return", "pick_place"]
    np.testing.assert_array_equal(position(actions[1], "pose0"), position(actions[0], "pose0"))
    assert actions[0]["hold_steps"] == 17


def test_environment_action_converts_all_probe_poses_and_hold_steps():
    config = copy.deepcopy(base_config())
    config["action"].update({"protocol": "probe_return", "probe_hold_steps": 23})
    action = _environment_action(generate_action_script(config, beads())[0])
    assert action["primitive"] == "pick_probe_return"
    assert action["params"]["hold_steps"] == 23
    for key in ("pose0", "pose_probe", "pose_return"):
        assert isinstance(action["params"][key], tuple)
        assert isinstance(action["params"][key][0], tuple)
        assert isinstance(action["params"][key][1], tuple)


def test_unknown_protocol_and_primitive_are_rejected():
    config = base_config()
    config["action"]["protocol"] = "mystery"
    with pytest.raises(ValueError, match="unsupported preload protocol"):
        generate_action_script(config, beads())
    with pytest.raises(ValueError, match="unsupported primitive"):
        _environment_action({"primitive": "mystery"})
