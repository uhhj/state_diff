import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "configs/experiment2/phase0" / name).read_text())


def test_resume1_changes_only_fixed_engineering_fields():
    original = load("hidden_routing_gate_phase0l.json")
    resume = load("hidden_routing_gate_phase0l_resume1.json")
    for key in (
        "stage_name", "resume_of",
        "eligible_verdict", "blocked_verdict", "geometry_repair",
        "required_provenance_report",
    ):
        resume.pop(key)
    resume["topology_id"] = original["topology_id"]
    resume["candidate_id"] = original["candidate_id"]
    resume["routing_gate"] = copy.deepcopy(resume["routing_gate"])
    resume["routing_gate"]["barrier_width"] = 0.340
    assert resume == original


def test_original_phase0l_config_is_unchanged():
    original = load("hidden_routing_gate_phase0l.json")
    assert original["topology_id"] == "hidden_routing_gate_v1"
    assert original["candidate_id"] == "hidden_routing_gate_v1_fixed_pull_v1"
    assert original["routing_gate"]["barrier_width"] == 0.340


def test_resume1_width_exceeds_analytic_minimum():
    resume = load("hidden_routing_gate_phase0l_resume1.json")
    repair = resume["geometry_repair"]
    assert resume["routing_gate"]["barrier_width"] == 0.350
    assert repair["analytic_nominal_minimum"] == 0.3425896274215007
    assert repair["phase0l_value"] < repair["analytic_nominal_minimum"]
    assert repair["resume1_value"] > repair["analytic_nominal_minimum"]


def test_resume1_contains_no_candidate_lists():
    resume = load("hidden_routing_gate_phase0l_resume1.json")
    forbidden = {"candidates", "barrier_candidates", "action_candidates", "target_candidates"}
    assert not forbidden.intersection(resume)


def test_scientific_task_and_thresholds_are_unchanged():
    original = load("hidden_routing_gate_phase0l.json")
    resume = load("hidden_routing_gate_phase0l_resume1.json")
    for key in (
        "task_name", "conditions", "condition_semantics", "action_policy",
        "official_outcome", "hz", "trace_stride", "no_action_steps",
        "post_main_steps", "workspace_bounds", "execution", "observation",
        "action", "seeds", "selection_targets",
    ):
        assert resume[key] == original[key]
    for key, value in original["routing_gate"].items():
        if key != "barrier_width":
            assert resume["routing_gate"][key] == value
