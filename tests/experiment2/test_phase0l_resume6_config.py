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


def test_resume6_changes_only_acquisition_execution():
    resume5 = load(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )
    resume6 = load(
        "hidden_routing_gate_"
        "phase0l_resume6.json"
    )

    assert resume6["topology_id"] == (
        resume5["topology_id"]
    )
    assert resume6["action"][
        "main_pull_frame"
    ] == resume5["action"][
        "main_pull_frame"
    ]
    assert resume6["action"][
        "acquisition_motion_mode"
    ] == "precise_endpoint_recovery"

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
            resume6["action"][key]
            == resume5["action"][key]
        )

    assert (
        resume6["routing_gate"]
        == resume5["routing_gate"]
    )
    assert (
        resume6["workspace_bounds"]
        == resume5["workspace_bounds"]
    )
    assert (
        resume6["execution"]
        == resume5["execution"]
    )
    assert (
        resume6["observation"]
        == resume5["observation"]
    )
    assert (
        resume6["selection_targets"]
        == resume5["selection_targets"]
    )
    assert (
        resume6["official_outcome"]
        == resume5["official_outcome"]
    )
    assert resume6["seeds"] == [
        71001,
        71002,
        71003,
    ]


def test_resume6_has_no_acquisition_search():
    config = load(
        "hidden_routing_gate_"
        "phase0l_resume6.json"
    )
    text = json.dumps(
        config,
        sort_keys=True,
    )
    for forbidden in (
        "acquisition_candidates",
        "pick_candidates",
        "lowering_candidates",
        "grasp_candidates",
        "tolerance_candidates",
        "distance_candidates",
    ):
        assert forbidden not in text
