import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / "configs/experiment2/phase0/hidden_routing_gate_phase0l.json").read_text())


def test_phase0l_fixed_scope_and_outcome():
    cfg = config()
    assert cfg["topology_id"] == "hidden_routing_gate_v1"
    assert cfg["candidate_id"] == "hidden_routing_gate_v1_fixed_pull_v1"
    assert cfg["official_outcome"] == "pulled_endpoint_target_success_gap"
    assert cfg["seeds"] == [71001, 71002, 71003]
    assert not {"candidates", "barrier_candidates", "action_candidates", "target_candidates"}.intersection(cfg)


def test_phase0l_geometry_and_gate_are_preregistered():
    cfg = config()
    action = cfg["action"]
    gate = cfg["routing_gate"]
    assert gate["barrier_offset"] + gate["barrier_thickness"] / 2 + 0.005 < gate["target_plane_offset"]
    assert gate["target_plane_offset"] < action["final_pull_distance"]
    assert action["stage1_pull_distance"] == 0.080
    assert action["final_pull_distance"] == 0.120
    assert cfg["selection_targets"]["min_median_routing_success_gap"] == 1.0
