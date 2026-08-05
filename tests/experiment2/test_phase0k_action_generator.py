import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = ROOT / "external" / "deformable-ravens"
for path in (ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.hidden_hook_common import (
    generate_hidden_latch_action_script,
    generate_hidden_latch_tension_action_script,
)
from scripts.experiment2.phase0.run_hidden_friction_pairs import _environment_action


def config():
    return json.loads(
        (ROOT / "configs/experiment2/phase0/hidden_latch_phase0k.json").read_text()
    )


def beads():
    return np.column_stack((
        np.linspace(0.35, 0.65, 25), np.zeros(25), np.full(25, 0.005)
    ))


def main(actions):
    rows = [row for row in actions if row["phase"] == "main_pull"]
    assert len(rows) == 1
    return rows[0]


def test_stage1_pose_matches_phase0i_target_byte_for_byte():
    cfg = config()
    base = main(generate_hidden_latch_action_script(cfg, beads()))
    tension = main(generate_hidden_latch_tension_action_script(cfg, beads()))
    assert tension["pose_stage1"] == base["pose1"]
    assert tension["pose0"] == base["pose0"]


def test_final_pose_is_fixed_extension_and_path_is_collinear():
    action = main(generate_hidden_latch_tension_action_script(config(), beads()))
    start = np.asarray(action["pose0"]["position"][:2])
    stage1 = np.asarray(action["pose_stage1"]["position"][:2])
    final = np.asarray(action["pose1"]["position"][:2])
    assert np.isclose(np.linalg.norm(stage1 - start), 0.08)
    assert np.isclose(np.linalg.norm(final - stage1), 0.04)
    assert np.isclose(np.linalg.norm(final - start), 0.12)
    assert np.allclose(
        (stage1 - start) / np.linalg.norm(stage1 - start),
        (final - start) / np.linalg.norm(final - start),
    )


def test_generator_returns_one_main_action_and_environment_params():
    action = main(generate_hidden_latch_tension_action_script(config(), beads()))
    assert action["primitive"] == "pick_precise_tension_extension"
    converted = _environment_action(action)
    assert converted["params"]["pose_stage1"][0] == tuple(
        action["pose_stage1"]["position"]
    )
    assert converted["params"]["min_achieved_fraction"] == 0.8


def test_generator_uses_same_endpoint_and_no_regrasp():
    actions = generate_hidden_latch_tension_action_script(config(), beads())
    assert [row["phase"] for row in actions] == ["preload", "main_pull"]
    assert sum(row["phase"] == "main_pull" for row in actions) == 1
    assert "pose_stage1" in main(actions)


def test_workspace_violation_stops_without_search():
    cfg = config()
    cfg["action"]["tension_extension_distance"] = 1.0
    with pytest.raises(RuntimeError, match="do not search"):
        generate_hidden_latch_tension_action_script(cfg, beads())


def test_phase0i_generator_is_unchanged():
    cfg = config()
    before = generate_hidden_latch_action_script(cfg, beads())
    generate_hidden_latch_tension_action_script(cfg, beads())
    after = generate_hidden_latch_action_script(cfg, beads())
    assert before == after
    assert main(after)["primitive"] == "pick_place"


@pytest.mark.parametrize("field,value", [
    ("topology_id", "wide_stop_z_latch_v2"),
    ("intervention_id", "different"),
])
def test_generator_rejects_scope_changes(field, value):
    cfg = copy.deepcopy(config())
    cfg[field] = value
    with pytest.raises(ValueError):
        generate_hidden_latch_tension_action_script(cfg, beads())
