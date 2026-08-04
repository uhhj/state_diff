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
            "pick_z": 0.001,
            "protocol": "planar_microprobe",
            "preload_distance": 0.0005,
            "main_pull_distance": 0.1,
            "microprobe_lift_height": 0.0015,
            "microprobe_hold_steps": 60,
            "microprobe_return_hold_steps": 120,
            "microprobe_post_release_steps": 180,
            "microprobe_approach_height": 0.02,
            "microprobe_retreat_z": 0.3,
        },
    }


def beads():
    return np.asarray([[0.4, 0.0, 0.001], [0.6, 0.0, 0.001]])


def test_planar_microprobe_action_and_original_main_start():
    actions = generate_action_script(base_config(), beads())
    assert [a["primitive"] for a in actions] == [
        "pick_planar_microprobe",
        "pick_place",
    ]
    probe = actions[0]
    assert actions[1]["pose0"] == probe["pose0"]
    assert probe["lift_height"] == 0.0015
    assert probe["hold_steps"] == 60
    assert probe["return_hold_steps"] == 120
    assert probe["post_release_steps"] == 180
    assert probe["approach_height"] == 0.02
    assert probe["retreat_z"] == 0.3


def test_environment_action_converts_microprobe_parameters():
    probe = generate_action_script(base_config(), beads())[0]
    action = _environment_action(probe)
    assert action["primitive"] == "pick_planar_microprobe"
    assert action["params"]["pose0"][0] == tuple(probe["pose0"]["position"])
    assert action["params"]["pose_probe"][0] == tuple(
        probe["pose_probe"]["position"]
    )
    assert action["params"]["pose_return"][0] == tuple(
        probe["pose_return"]["position"]
    )
    for key in (
        "lift_height", "hold_steps", "return_hold_steps",
        "post_release_steps", "approach_height", "retreat_z",
    ):
        assert action["params"][key] == probe[key]


def test_unknown_protocol_is_rejected():
    config = copy.deepcopy(base_config())
    config["action"]["protocol"] = "unknown"
    with pytest.raises(ValueError, match="unsupported preload protocol"):
        generate_action_script(config, beads())
