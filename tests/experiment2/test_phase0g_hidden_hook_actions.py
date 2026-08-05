import json
from pathlib import Path

import numpy as np

from scripts.experiment2.phase0.hidden_hook_common import generate_hidden_hook_action_script
from scripts.experiment2.phase0.run_hidden_friction_pairs import _environment_action


ROOT = Path(__file__).resolve().parents[2]


def config():
    phase0g = json.loads(
        (ROOT / "configs/experiment2/phase0/hidden_hook_phase0g.json").read_text()
    )
    phase0g["hook"] = phase0g["candidates"][1]["hook"]
    return phase0g


def beads():
    return np.column_stack((
        np.linspace(0.35, 0.65, 25),
        np.zeros(25),
        np.full(25, 0.001),
    ))


def test_action_script_contains_precise_probe_and_main_pull():
    actions = generate_hidden_hook_action_script(config(), beads())
    assert [row["primitive"] for row in actions] == [
        "pick_precise_probe_return", "pick_place"
    ]
    assert [row["phase"] for row in actions] == ["preload", "main_pull"]
    assert len(actions[0]["layout"]["boxes"]) == 3
    converted = _environment_action(actions[0])
    assert converted["params"]["joint_tolerance"] == 1e-4
    assert converted["params"]["min_achieved_fraction"] == 0.8


def test_probe_and_main_targets_are_inside_workspace():
    actions = generate_hidden_hook_action_script(config(), beads())
    targets = [actions[0]["pose_probe"]["position"], actions[1]["pose1"]["position"]]
    for x, y, _ in targets:
        assert 0.25 < x < 0.75
        assert -0.45 < y < 0.45
