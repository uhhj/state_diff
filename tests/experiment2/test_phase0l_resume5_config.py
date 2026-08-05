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


def test_resume5_changes_only_main_pull_frame():
    resume4 = load(
        "hidden_routing_gate_"
        "phase0l_resume4.json"
    )
    resume5 = load(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )

    assert resume5["topology_id"] == (
        resume4["topology_id"]
    )
    assert resume5["action"][
        "main_pull_frame"
    ] == (
        "current_endpoint_frozen_normal"
    )

    for key in (
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
            resume5["action"][key]
            == resume4["action"][key]
        )

    assert (
        resume5["routing_gate"]
        == resume4["routing_gate"]
    )
    assert (
        resume5["workspace_bounds"]
        == resume4["workspace_bounds"]
    )
    assert (
        resume5["execution"]
        == resume4["execution"]
    )
    assert (
        resume5["observation"]
        == resume4["observation"]
    )
    assert (
        resume5["selection_targets"]
        == resume4["selection_targets"]
    )
    assert (
        resume5["official_outcome"]
        == resume4["official_outcome"]
    )
    assert (
        resume5["no_action_steps"]
        == resume4["no_action_steps"]
    )
    assert resume5["seeds"] == [
        71001,
        71002,
        71003,
    ]


def test_resume5_has_no_search_fields():
    config = load(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )
    text = json.dumps(
        config,
        sort_keys=True,
    )
    for forbidden in (
        "distance_candidates",
        "direction_candidates",
        "frame_candidates",
        "target_candidates",
        "action_candidates",
        "geometry_candidates",
    ):
        assert forbidden not in text
