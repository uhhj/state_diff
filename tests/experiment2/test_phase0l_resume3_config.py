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


def test_resume3_changes_only_probe_selector():
    resume2 = load(
        "hidden_routing_gate_"
        "phase0l_resume2.json"
    )
    resume3 = load(
        "hidden_routing_gate_"
        "phase0l_resume3.json"
    )

    assert resume3[
        "routing_gate"
    ][
        "probe_selector_mode"
    ] == (
        "all_bead_clearance_"
        "nearest_center"
    )

    for key in (
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
            resume3["routing_gate"][key]
            == resume2["routing_gate"][key]
        )

    assert (
        resume3["action"]
        == resume2["action"]
    )
    assert (
        resume3["workspace_bounds"]
        == resume2["workspace_bounds"]
    )
    assert (
        resume3["selection_targets"]
        == resume2["selection_targets"]
    )
    assert (
        resume3["official_outcome"]
        == resume2["official_outcome"]
    )
    assert (
        resume3["seeds"]
        == [71001, 71002, 71003]
    )


def test_resume3_has_no_probe_index_list():
    config = load(
        "hidden_routing_gate_"
        "phase0l_resume3.json"
    )
    text = json.dumps(
        config,
        sort_keys=True,
    )
    assert "probe_indices" not in text
    assert "probe_candidates" not in text
    assert "probe_index_override" not in text
