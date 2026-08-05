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


def test_resume4_changes_only_layout_lifecycle():
    resume3 = load(
        "hidden_routing_gate_"
        "phase0l_resume3.json"
    )
    resume4 = load(
        "hidden_routing_gate_"
        "phase0l_resume4.json"
    )

    assert resume4[
        "routing_gate"
    ][
        "public_layout_mode"
    ] == "frozen_at_branch_arm"

    for key in (
        "center_ratio",
        "probe_selector_mode",
        "probe_roof_clearance",
        "probe_roof_depth",
        "probe_roof_width",
        "probe_roof_thickness",
        "barrier_mode",
        "barrier_offset",
        "barrier_thickness",
        "barrier_width",
        "barrier_safety_margin",
        "barrier_height",
        "target_plane_offset",
        "target_zone_depth",
        "target_corridor_half_width",
        "leading_segment_size",
        "min_initial_clearance",
    ):
        assert (
            resume4["routing_gate"][key]
            == resume3["routing_gate"][key]
        )

    assert (
        resume4["action"]
        == resume3["action"]
    )
    assert (
        resume4["workspace_bounds"]
        == resume3["workspace_bounds"]
    )
    assert (
        resume4["selection_targets"]
        == resume3["selection_targets"]
    )
    assert (
        resume4["official_outcome"]
        == resume3["official_outcome"]
    )
    assert (
        resume4["no_action_steps"]
        == resume3["no_action_steps"]
    )
    assert resume4["seeds"] == [
        71001,
        71002,
        71003,
    ]


def test_resume4_has_no_search_fields():
    config = load(
        "hidden_routing_gate_"
        "phase0l_resume4.json"
    )
    text = json.dumps(
        config,
        sort_keys=True,
    )
    for forbidden in (
        "layout_candidates",
        "target_candidates",
        "action_candidates",
        "probe_candidates",
        "tolerance_candidates",
    ):
        assert forbidden not in text
