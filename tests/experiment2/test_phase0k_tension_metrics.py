import copy

import numpy as np

from scripts.experiment2.phase0.hidden_hook_metrics import hidden_hook_outcome_metrics
from scripts.experiment2.phase0.phase0k_tension_metrics import (
    TENSION_STAGES,
    tension_extension_metrics,
    tension_motion_valid,
)


def action_script():
    return [{
        "phase": "main_pull",
        "primitive": "pick_precise_tension_extension",
        "pose0": {"position": [0.0, 0.0, 0.0]},
        "pose_stage1": {"position": [0.0, 0.08, 0.0]},
        "pose1": {"position": [0.0, 0.12, 0.0]},
    }]


def trace(progress):
    steps = np.asarray([0, 4, 8, 12, 16, 20])
    phases = np.asarray([
        "preload", "main_pull", "main_pull", "main_pull", "main_pull",
        "post_main",
    ])
    base = np.column_stack((
        np.linspace(0.0, 0.04, 5), np.zeros(5), np.zeros(5)
    ))
    positions = np.stack([
        base + np.asarray([0.0, value, 0.0]) for value in progress
    ])
    return {
        "physics_step": steps,
        "phase": phases,
        "bead_positions": positions,
        "contact_active_beads": np.zeros(len(steps)),
        "oracle_contact": np.arange(len(steps)),
    }


def metadata(stage1_step, stage2_step, grasp=True):
    ends = {
        "tension_pull_lift": 4,
        "tension_pull_stage1": stage1_step,
        "tension_pull_stage2": stage2_step,
        "tension_pull_lower_release": 16,
    }
    rows = []
    for stage in TENSION_STAGES:
        row = {
            "stage": stage,
            "success": True,
            "achieved_fraction": 1.0,
            "physics_step_end": ends[stage],
        }
        if stage != "tension_pull_lower_release":
            row["grasp_active_after"] = grasp
        else:
            row["grasp_active_before_release"] = grasp
        rows.append(row)
    return {"motion_events": rows}


def case():
    free = trace([0.0, 0.01, 0.04, 0.07, 0.10, 0.11])
    hidden = trace([0.0, 0.005, 0.015, 0.025, 0.035, 0.04])
    return free, hidden, metadata(8, 16), metadata(12, 16)


def test_positions_use_own_event_end_steps_and_gap_increases():
    free, hidden, free_meta, hidden_meta = case()
    result = tension_extension_metrics(
        free, hidden, free_meta, hidden_meta, action_script()
    )
    assert np.isclose(result["stage1"]["mean_progress_gap"], 0.015)
    assert np.isclose(result["stage2"]["mean_progress_gap"], 0.065)
    assert result["stage2"]["mean_progress_gap"] > result["stage1"]["mean_progress_gap"]


def test_final_gap_matches_unchanged_official_definition():
    free, hidden, free_meta, hidden_meta = case()
    diagnostic = tension_extension_metrics(
        free, hidden, free_meta, hidden_meta, action_script()
    )
    official = hidden_hook_outcome_metrics(free, hidden, action_script())
    assert np.isclose(
        diagnostic["final"]["mean_progress_gap"],
        official["mean_cable_progress_gap"],
    )
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["changes_official_progress_gate"] is False


def test_tension_motion_requires_all_stages_and_grasp_retention():
    valid = metadata(8, 16)
    assert tension_motion_valid(valid, 0.8)
    missing = copy.deepcopy(valid)
    missing["motion_events"].pop()
    try:
        tension_motion_valid(missing, 0.8)
        assert False
    except ValueError:
        pass
    lost = metadata(8, 16, grasp=False)
    assert not tension_motion_valid(lost, 0.8)
    short = metadata(8, 16)
    short["motion_events"][1]["achieved_fraction"] = 0.7
    assert not tension_motion_valid(short, 0.8)


def test_metrics_ignore_oracle_contact_fields():
    free, hidden, free_meta, hidden_meta = case()
    first = tension_extension_metrics(
        free, hidden, free_meta, hidden_meta, action_script()
    )
    free["oracle_contact"][:] = 999
    hidden["oracle_contact"][:] = -999
    second = tension_extension_metrics(
        free, hidden, free_meta, hidden_meta, action_script()
    )
    assert first == second
