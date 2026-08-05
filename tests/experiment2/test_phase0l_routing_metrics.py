import copy

import numpy as np
import pytest

from scripts.experiment2.phase0.hidden_routing_gate_metrics import (
    hidden_routing_gate_outcome_metrics,
    summarize_hidden_routing_gate,
)


def action(corridor=0.035):
    return [{
        "phase": "main_pull",
        "primitive": "pick_precise_tension_extension",
        "public_task_layout": {
            "endpoint_index": 0,
            "leading_segment_indices": [0, 1],
            "target_plane_point_xy": [0.10, 0.0],
            "normal_xy": [1.0, 0.0],
            "tangent_xy": [0.0, 1.0],
            "target_corridor_half_width": corridor,
        },
    }]


def trace(endpoint, second=(0.11, 0.0), oracle=0.0):
    positions = np.zeros((1, 5, 3), dtype=float)
    positions[0, 0, :2] = endpoint
    positions[0, 1, :2] = second
    return {
        "phase": np.asarray(["post_main"]),
        "bead_positions": positions,
        "oracle_contact": np.asarray([oracle]),
    }


def test_official_success_requires_plane_crossing_and_corridor():
    result = hidden_routing_gate_outcome_metrics(
        trace((0.12, 0.0)), trace((0.09, 0.0)), action()
    )
    assert result["free_endpoint_target_success"] is True
    assert result["hidden_endpoint_target_success"] is False
    assert result["routing_success_gap"] == 1
    outside = hidden_routing_gate_outcome_metrics(
        trace((0.12, 0.04)), trace((0.09, 0.0)), action()
    )
    assert outside["free_endpoint_target_success"] is False


def test_success_gap_is_signed_free_minus_hidden():
    result = hidden_routing_gate_outcome_metrics(
        trace((0.09, 0.0)), trace((0.12, 0.0)), action()
    )
    assert result["routing_success_gap"] == -1


def test_leading_crossing_is_diagnostic():
    result = hidden_routing_gate_outcome_metrics(
        trace((0.12, 0.0), (0.11, 0.0)),
        trace((0.09, 0.0), (0.11, 0.0)),
        action(),
    )
    assert result["free_leading_crossing_fraction"] == 1.0
    assert result["hidden_leading_crossing_fraction"] == 0.5
    assert result["leading_crossing_fraction_gap"] == 0.5


def test_metric_rejects_hidden_layout_fields():
    leaked = action()
    leaked[0]["public_task_layout"]["routing_barrier"] = {}
    with pytest.raises(ValueError, match="hidden layout"):
        hidden_routing_gate_outcome_metrics(trace((0.12, 0.0)), trace((0.09, 0.0)), leaked)


def test_metric_ignores_oracle_contact_arrays():
    first = hidden_routing_gate_outcome_metrics(
        trace((0.12, 0.0), oracle=1.0), trace((0.09, 0.0), oracle=2.0), action()
    )
    second = hidden_routing_gate_outcome_metrics(
        trace((0.12, 0.0), oracle=999.0), trace((0.09, 0.0), oracle=-999.0), action()
    )
    assert first == second


def pair_row():
    return {
        "action_hash_match": True,
        "free_base_state_hash_match": True,
        "hidden_base_state_hash_match": True,
        "free_hidden_initial_hash_match": True,
        "max_initial_state_difference": 0.0,
        "free_arm_max_abs_jump": 0.0,
        "hidden_arm_max_abs_jump": 0.0,
        "free_no_action_drift": 0.0,
        "hidden_no_action_drift": 0.0,
        "preload_end_max_abs_xy": 0.001,
        "contact_impulse_gap": 0.03,
        "main_branch_ade": 0.01,
        "main_branch_fde": 0.02,
        "branch_amplification": 20.0,
        "routing_success_gap": 1,
        "endpoint_target_margin_gap": 0.03,
        "leading_crossing_fraction_gap": 0.5,
        "free_endpoint_target_success": True,
        "hidden_endpoint_target_success": False,
        "hidden_gate_engagement_fraction": 0.5,
        "probe_motion_valid": True,
        "routing_motion_valid": True,
        "mean_cable_progress_gap": -0.02,
    }


def targets():
    return {
        "max_initial_abs_xy": 1e-12,
        "max_arm_jump": 1e-12,
        "max_median_no_action_drift": 0.0015,
        "max_median_preload_visible_difference": 0.004,
        "min_median_contact_impulse_gap": 0.02,
        "min_median_main_branch_ade": 0.006,
        "min_median_main_branch_fde": 0.015,
        "min_median_branch_amplification": 2.0,
        "max_grouped_vision_accuracy": 0.7,
        "min_grouped_sensor_accuracy": 0.7,
        "min_grouped_oracle_accuracy": 0.9,
        "min_sensor_over_vision_margin": 0.1,
        "min_median_routing_success_gap": 1.0,
        "min_median_engagement_fraction": 0.1,
        "min_seed_pass_fraction": 0.6,
    }


def classifier(accuracy):
    return {"accuracy": accuracy}


def test_summary_uses_routing_gap_not_mean_progress():
    row = pair_row()
    summary = summarize_hidden_routing_gate(
        "candidate", {}, [row], targets(), classifier(0.5), classifier(0.5),
        classifier(0.8), classifier(1.0),
    )
    assert row["mean_cable_progress_gap"] < 0
    assert summary["checks"]["routing_outcome"] is True
    assert summary["eligible"] is True


def test_failed_routing_gap_cannot_be_rescued_by_legacy_progress():
    row = pair_row()
    row["routing_success_gap"] = 0
    row["mean_cable_progress_gap"] = 100.0
    summary = summarize_hidden_routing_gate(
        "candidate", {}, [row], targets(), classifier(0.5), classifier(0.5),
        classifier(0.8), classifier(1.0),
    )
    assert summary["checks"]["routing_outcome"] is False
    assert summary["eligible"] is False
