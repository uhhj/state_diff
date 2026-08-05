"""Hidden-Hook outcome metrics."""
from __future__ import annotations

import numpy as np

from scripts.experiment2.phase0.common import phase_slice


def _last_positions(trace, phase):
    value = phase_slice(trace, phase)
    positions = np.asarray(value["bead_positions"], dtype=np.float64)
    if positions.shape[0] == 0:
        raise ValueError(f"empty phase {phase}")
    return positions[-1]


def hidden_hook_outcome_metrics(free_trace, hidden_trace, action_script):
    main = [action for action in action_script if action["phase"] == "main_pull"]
    if len(main) != 1:
        raise ValueError("expected one main_pull action")
    start = np.asarray(main[0]["pose0"]["position"][:2], dtype=np.float64)
    target = np.asarray(main[0]["pose1"]["position"][:2], dtype=np.float64)
    vector = target - start
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        raise ValueError("zero main pull distance")
    direction = vector / distance

    free_before = _last_positions(free_trace, "preload")
    hidden_before = _last_positions(hidden_trace, "preload")
    free_after = _last_positions(free_trace, "post_main")
    hidden_after = _last_positions(hidden_trace, "post_main")
    free_projection = (free_after[:, :2] - free_before[:, :2]) @ direction
    hidden_projection = (hidden_after[:, :2] - hidden_before[:, :2]) @ direction

    hidden_main = phase_slice(hidden_trace, "main_pull")
    hidden_post = phase_slice(hidden_trace, "post_main")
    active = np.concatenate([
        np.asarray(hidden_main["contact_active_beads"], dtype=np.float64),
        np.asarray(hidden_post["contact_active_beads"], dtype=np.float64),
    ])
    return {
        "free_mean_cable_progress": float(np.mean(free_projection)),
        "hidden_mean_cable_progress": float(np.mean(hidden_projection)),
        "mean_cable_progress_gap": float(np.mean(free_projection) - np.mean(hidden_projection)),
        "free_opposite_endpoint_progress": float(min(free_projection[0], free_projection[-1])),
        "hidden_opposite_endpoint_progress": float(min(hidden_projection[0], hidden_projection[-1])),
        "opposite_endpoint_progress_gap": float(
            min(free_projection[0], free_projection[-1])
            - min(hidden_projection[0], hidden_projection[-1])
        ),
        "hidden_hook_engagement_fraction": float(np.mean(active > 0)) if active.size else 0.0,
    }
