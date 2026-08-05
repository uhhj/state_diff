import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "configs/experiment2/phase0" / name).read_text())


def test_phase0k_restores_phase0i_geometry_and_frozen_contract():
    phase0i = load("hidden_latch_phase0i.json")
    phase0k = load("hidden_latch_phase0k.json")
    for key in (
        "task_name", "conditions", "topology_id", "hz", "trace_stride",
        "no_action_steps", "post_main_steps", "workspace_bounds", "execution",
        "observation", "latch", "seeds", "selection_targets",
    ):
        assert phase0k[key] == phase0i[key]
    assert phase0k["latch"]["wall_width"] == 0.032
    assert phase0k["seeds"] == [71001, 71002, 71003]


def test_phase0k_adds_only_fixed_tension_action_fields():
    phase0i = load("hidden_latch_phase0i.json")
    phase0k = load("hidden_latch_phase0k.json")
    added = {
        "tension_extension_distance", "tension_lift_height",
        "tension_approach_height", "tension_retreat_z",
    }
    assert set(phase0k["action"]) - set(phase0i["action"]) == added
    for key, value in phase0i["action"].items():
        assert phase0k["action"][key] == value
    assert phase0k["action"]["main_pull_distance"] == 0.08
    assert phase0k["action"]["tension_extension_distance"] == 0.04
    assert phase0k["fixed_intervention"] == {
        "stage1_distance": 0.08,
        "extension_distance": 0.04,
        "final_distance": 0.12,
        "same_grasp": True,
        "collinear": True,
    }
    assert not {
        "candidates", "action_candidates", "extension_candidates"
    }.intersection(phase0k)

