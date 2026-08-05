import numpy as np
import pytest

from scripts.experiment2.phase0.hidden_hook_metrics import hidden_hook_outcome_metrics


def trace(progress, active):
    before = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    after = before.copy()
    after[:, 0] += progress
    return {
        "phase": np.asarray(["preload", "main_pull", "main_pull", "post_main", "post_main"]),
        "bead_positions": np.asarray([before, before, before, after, after]),
        "contact_active_beads": np.asarray(active),
    }


def test_progress_gap_and_engagement_fraction():
    actions = [{
        "phase": "main_pull",
        "pose0": {"position": [1.0, 0.0, 0.0]},
        "pose1": {"position": [1.1, 0.0, 0.0]},
    }]
    metrics = hidden_hook_outcome_metrics(
        trace(0.04, [0, 0, 0, 0, 0]),
        trace(0.01, [0, 0, 1, 1, 1]),
        actions,
    )
    assert metrics["mean_cable_progress_gap"] == pytest.approx(0.03)
    assert metrics["opposite_endpoint_progress_gap"] == pytest.approx(0.03)
    assert metrics["hidden_hook_engagement_fraction"] == pytest.approx(0.75)
