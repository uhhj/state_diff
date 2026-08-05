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


def test_resume9_changes_only_descent_execution():
    resume8 = load(
        "hidden_routing_gate_"
        "phase0l_resume8.json"
    )
    resume9 = load(
        "hidden_routing_gate_"
        "phase0l_resume9.json"
    )

    assert resume9["action"][
        "acquisition_descent_mode"
    ] == "single_pass_target"

    for key in (
        "target_grasp_mode",
        "probe_acquisition_motion_mode",
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
            resume9["action"][key]
            == resume8["action"][key]
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
            resume9[key]
            == resume8[key]
        )


def test_resume9_adds_no_scientific_or_search_fields():
    config = load(
        "hidden_routing_gate_"
        "phase0l_resume9.json"
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
        "descent_candidates",
        "target_candidates",
        "threshold_candidates",
        "retry_count",
        "capture_radius",
        "proximity_threshold",
    ):
        assert forbidden not in text
