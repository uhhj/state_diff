import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "configs/experiment2/phase0" / name).read_text())


def test_phase0i_has_one_fixed_topology_three_seeds_and_frozen_tolerances():
    config = load("hidden_latch_phase0i.json")
    assert config["topology_policy"] == "single_fixed_topology_no_grid_search"
    assert config["topology_id"] == "delayed_z_latch_v1"
    assert config["seeds"] == [71001, 71002, 71003]
    assert "candidates" not in config
    assert config["action"]["joint_tolerance"] == 1e-4
    assert config["action"]["cartesian_tolerance"] == 2e-4
    assert config["action"]["probe_lift_height"] == 0.004


def test_phase0i_selection_targets_are_identical_to_phase0h():
    phase0i = load("hidden_latch_phase0i.json")
    phase0h = load("hidden_hook_phase0h.json")
    assert phase0i["selection_targets"] == phase0h["selection_targets"]
