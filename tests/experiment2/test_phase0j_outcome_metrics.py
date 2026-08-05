import copy

import numpy as np
import pytest

from scripts.experiment2.phase0.phase0j_outcome_metrics import (
    blocked_indices_from_layout,
    hidden_latch_outcome_decomposition,
    ordered_segment_indices,
)


def layout(width=0.03, endpoint_index=0):
    return {
        "probe_index": 9,
        "endpoint_index": endpoint_index,
        "tangent_xy": [1.0, 0.0],
        "bead_collision_half_extent": 0.005,
        "boxes": [{
            "name": "stop_wall",
            "center_xy": [0.09, 0.02],
            "center_z": 0.01,
            "half_extents": [width / 2, 0.001, 0.01],
            "yaw": 0.0,
        }],
    }


def synthetic_case():
    before = np.column_stack((
        np.arange(20) * 0.01,
        np.zeros(20),
        np.full(20, 0.005),
    ))
    free_disp = np.zeros_like(before)
    free_disp[:, 1] = 0.02
    hidden_disp = np.zeros_like(before)
    hidden_disp[:7, 1] = 0.015
    hidden_disp[7:12, 1] = 0.001
    hidden_disp[12:, 1] = 0.0
    hidden_disp[:, 0] = np.linspace(0.0, 0.004, 20)
    hidden_disp[:, 2] = np.linspace(0.0, 0.002, 20)
    phases = np.asarray(["preload", "post_main"])
    free = {
        "phase": phases,
        "bead_positions": np.stack((before, before + free_disp)),
        "oracle_contact": np.asarray([0.0, 0.0]),
    }
    hidden = {
        "phase": phases,
        "bead_positions": np.stack((before, before + hidden_disp)),
        "oracle_contact": np.asarray([999.0, -999.0]),
    }
    actions = [
        {"phase": "preload", "layout": layout()},
        {
            "phase": "main_pull",
            "pose0": {"position": [0.0, 0.0, 0.0]},
            "pose1": {"position": [0.0, 0.08, 0.0]},
        },
    ]
    return free, hidden, actions


def test_blocked_indices_expand_with_wall_width():
    positions = synthetic_case()[0]["bead_positions"][0]
    narrow = blocked_indices_from_layout(positions, layout(0.03))
    wide = blocked_indices_from_layout(positions, layout(0.08))
    assert narrow.tolist() == [7, 8, 9, 10, 11]
    assert wide.size > narrow.size
    assert set(narrow).issubset(set(wide))


def test_blocked_indices_ignore_projection_only_remote_segments():
    positions = synthetic_case()[0]["bead_positions"][0].copy()
    positions[0, 0] = 0.09
    assert blocked_indices_from_layout(
        positions, layout(0.03)
    ).tolist() == [7, 8, 9, 10, 11]


def test_ordered_partition_for_endpoint_0_and_last_endpoint():
    blocked = np.arange(7, 12)
    first = ordered_segment_indices(20, blocked, 0)
    last = ordered_segment_indices(20, blocked, 19)
    assert first["pulled_side"].tolist() == list(range(7))
    assert first["trailing_side"].tolist() == list(range(12, 20))
    assert last["pulled_side"].tolist() == list(range(12, 20))
    assert last["trailing_side"].tolist() == list(range(7))


def test_decomposition_matches_all_bead_definition_and_localizes_segments():
    free, hidden, actions = synthetic_case()
    result = hidden_latch_outcome_decomposition(free, hidden, actions)
    expected = np.mean([
        result_row["progress_gap"] for result_row in result["per_bead"]
    ])
    assert result["segments"]["all"]["progress_gap"] == pytest.approx(expected)
    assert result["segments"]["blocked"]["progress_gap"] > result["segments"]["pulled_side"]["progress_gap"]
    assert result["segments"]["trailing_side"]["progress_gap"] > result["segments"]["pulled_side"]["progress_gap"]
    assert result["segments"]["all"]["progress_gap"] > 0
    assert result["motion_difference_components"]["normal_rms"] > 0
    assert result["motion_difference_components"]["tangent_rms"] > 0
    assert result["motion_difference_components"]["vertical_rms"] > 0
    assert result["diagnostic_only"] is True
    assert result["changes_official_progress_gate"] is False


def test_oracle_fields_do_not_affect_result():
    free, hidden, actions = synthetic_case()
    first = hidden_latch_outcome_decomposition(free, hidden, actions)
    free_changed, hidden_changed = copy.deepcopy(free), copy.deepcopy(hidden)
    free_changed["oracle_contact"][:] = 123456
    hidden_changed["oracle_contact"][:] = -123456
    second = hidden_latch_outcome_decomposition(
        free_changed, hidden_changed, actions
    )
    assert first == second


@pytest.mark.parametrize("blocked,endpoint", [([], 0), ([0, 1], 0), ([18, 19], 19)])
def test_invalid_or_empty_segments_raise(blocked, endpoint):
    with pytest.raises(ValueError):
        ordered_segment_indices(20, np.asarray(blocked), endpoint)
