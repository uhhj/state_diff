import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/experiment2/phase0"


def load(name):
    return json.loads((CONFIGS / name).read_text())


def test_resume2_preserves_scientific_definition():
    previous = load("hidden_routing_gate_phase0l_resume1.json")
    resume2 = load("hidden_routing_gate_phase0l_resume2.json")
    for key in (
        "task_name", "conditions", "condition_semantics", "topology_policy",
        "action_policy", "official_outcome", "hz", "trace_stride",
        "no_action_steps", "post_main_steps", "workspace_bounds", "execution",
        "observation", "action", "seeds", "selection_targets",
    ):
        assert resume2[key] == previous[key]


def test_resume2_changes_only_constructor_semantics():
    previous = load("hidden_routing_gate_phase0l_resume1.json")
    resume2 = load("hidden_routing_gate_phase0l_resume2.json")
    for key, value in previous["routing_gate"].items():
        if key not in {"barrier_width"}:
            assert resume2["routing_gate"][key] == value
    assert resume2["routing_gate"]["barrier_mode"] == "endpoint_corridor"
    assert resume2["routing_gate"]["barrier_safety_margin"] == 0.005


def test_resume2_width_is_derived_not_manual():
    gate = load("hidden_routing_gate_phase0l_resume2.json")["routing_gate"]
    assert gate["barrier_width"] == 0.0
    assert gate["barrier_mode"] == "endpoint_corridor"


def test_resume2_formula_matches_recorded_width():
    config = load("hidden_routing_gate_phase0l_resume2.json")
    gate = config["routing_gate"]
    expected = 2 * (
        gate["target_corridor_half_width"]
        + np.sqrt(3) * 0.005
        + gate["barrier_safety_margin"]
    )
    assert np.isclose(expected, config["geometry_repair"]["resolved_barrier_width"])


def test_resume2_keeps_workspace_and_action():
    previous = load("hidden_routing_gate_phase0l_resume1.json")
    resume2 = load("hidden_routing_gate_phase0l_resume2.json")
    assert resume2["workspace_bounds"] == previous["workspace_bounds"]
    assert resume2["action"] == previous["action"]


def test_resume2_contains_no_candidate_lists():
    config = load("hidden_routing_gate_phase0l_resume2.json")
    assert not {"candidates", "barrier_candidates", "action_candidates"}.intersection(config)


def test_phase0l_and_resume1_configs_are_unchanged():
    original = load("hidden_routing_gate_phase0l.json")
    resume1 = load("hidden_routing_gate_phase0l_resume1.json")
    assert original["routing_gate"]["barrier_width"] == 0.340
    assert "barrier_mode" not in original["routing_gate"]
    assert resume1["routing_gate"]["barrier_width"] == 0.350
    assert "barrier_mode" not in resume1["routing_gate"]
