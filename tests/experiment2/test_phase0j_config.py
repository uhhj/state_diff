import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "configs/experiment2/phase0" / name).read_text())


def test_phase0j_changes_only_named_topology_and_wall_width():
    phase0i = load("hidden_latch_phase0i.json")
    phase0j = load("hidden_latch_phase0j.json")
    assert phase0i["latch"]["wall_width"] == 0.032
    assert phase0j["latch"]["wall_width"] == 0.080
    assert phase0j["topology_id"] == "wide_stop_z_latch_v2"
    assert phase0j["fixed_change"] == {
        "field": "latch.wall_width",
        "phase0i_value": 0.032,
        "phase0j_value": 0.080,
    }
    normalized_i = copy.deepcopy(phase0i)
    normalized_j = copy.deepcopy(phase0j)
    normalized_i.pop("topology_id")
    normalized_j.pop("topology_id")
    normalized_j.pop("fixed_change")
    normalized_j["latch"]["wall_width"] = normalized_i["latch"]["wall_width"]
    assert normalized_i == normalized_j


def test_phase0j_freezes_action_seeds_thresholds_observation_and_execution():
    phase0i = load("hidden_latch_phase0i.json")
    phase0j = load("hidden_latch_phase0j.json")
    for key in ("action", "seeds", "selection_targets", "observation", "execution"):
        assert phase0j[key] == phase0i[key]
    assert phase0j["seeds"] == [71001, 71002, 71003]
    assert phase0j["topology_policy"] == "single_fixed_topology_no_grid_search"
    assert "candidates" not in phase0j

