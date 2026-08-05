import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    ROOT
    / "configs"
    / "experiment2"
    / "phase0"
)


def load(name):
    return json.loads(
        (
            CONFIGS / name
        ).read_text(
            encoding="utf-8"
        )
    )


def test_resume7_changes_only_probe_execution_and_status():
    resume6 = load(
        "hidden_routing_gate_"
        "phase0l_resume6.json"
    )
    resume7 = load(
        "hidden_routing_gate_"
        "phase0l_resume7.json"
    )

    assert resume7["action"][
        "probe_acquisition_motion_mode"
    ] == "precise_endpoint_recovery"

    for key in (
        "main_pull_frame",
        "acquisition_motion_mode",
        "pick_z",
        "stage1_pull_distance",
        "final_pull_distance",
        "probe_lift_height",
        "probe_hold_steps",
        "probe_return_hold_steps",
        "probe_post_release_steps",
        "probe_approach_height",
        "probe_retreat_z",
        "routing_lift_height",
        "routing_approach_height",
        "routing_retreat_z",
        "joint_tolerance",
        "cartesian_tolerance",
        "min_achieved_fraction",
    ):
        assert (
            resume7["action"][key]
            == resume6["action"][key]
        )

    for key in (
        "routing_gate",
        "workspace_bounds",
        "execution",
        "observation",
        "selection_targets",
        "official_outcome",
        "no_action_steps",
        "post_main_steps",
        "seeds",
        "topology_id",
    ):
        assert (
            resume7[key]
            == resume6[key]
        )


def test_resume7_adds_no_scientific_or_search_fields():
    config = load(
        "hidden_routing_gate_"
        "phase0l_resume7.json"
    )
    text = json.dumps(
        config,
        sort_keys=True,
    )
    for forbidden in (
        "new_classifier",
        "new_sensor_feature",
        "new_outcome",
        "geometry_candidates",
        "action_candidates",
        "distance_candidates",
        "threshold_candidates",
        "retry_count",
    ):
        assert forbidden not in text
