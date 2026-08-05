import inspect
import json
from pathlib import Path

import numpy as np

from scripts.experiment2.phase0.hidden_hook_common import (
    generate_hidden_latch_action_script,
)
from scripts.experiment2.phase0.run_hidden_friction_pairs import _environment_action
from scripts.experiment2.phase0 import run_hidden_latch_phase0i as runner


ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads(
        (ROOT / "configs/experiment2/phase0/hidden_latch_phase0i.json").read_text()
    )


def beads():
    return np.column_stack((
        np.linspace(0.35, 0.65, 25), np.zeros(25), np.full(25, 0.005)
    ))


def test_latch_action_has_vertical_probe_and_fixed_main_pull():
    actions = generate_hidden_latch_action_script(config(), beads())
    assert [row["primitive"] for row in actions] == [
        "pick_precise_latch_probe", "pick_place"
    ]
    probe = actions[0]
    assert "pose_probe" not in probe
    converted = _environment_action(probe)
    assert converted["params"]["lift_height"] == 0.004
    assert converted["params"]["joint_tolerance"] == 1e-4
    assert converted["params"]["cartesian_tolerance"] == 2e-4


def test_runner_uses_corrected_sensor_separate_from_oracle_and_freezes_scope():
    source = inspect.getsource(runner)
    assert "stage_aligned_formal_sensor_feature" in source
    assert "sensor_contact_feature" not in source
    assert '"formal_sensor": formal' in source
    assert '"oracle_contact": oracle_contact_feature' in source
    assert '"training_performed": False' in source
    assert '"geometry_search_performed": False' in source
    assert "not training or Scientific PASS" in source
