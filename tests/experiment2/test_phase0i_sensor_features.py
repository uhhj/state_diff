import copy

import numpy as np
import pytest

from scripts.experiment2.phase0.phase0i_sensor_features import (
    motion_stage_mask,
    reaction_only_feature,
    stage_aligned_formal_sensor_feature,
)


def synthetic_trace():
    count = 12
    trace = {
        "physics_step": np.arange(count) * 4,
        "sensor_joint_motor_torque": np.ones((count, 6)),
        "sensor_joint_reaction_force_torque": np.ones((count, 6, 6)),
        "sensor_suction_force_xyz": np.ones((count, 3)),
        "sensor_suction_torque_xyz": np.ones((count, 3)) * 2,
        "sensor_grasp_active": np.ones(count, dtype=np.int64),
        "sensor_constraint_available": np.ones(count, dtype=np.int64),
        "contact_force_norm": np.zeros(count),
        "contact_active_beads": np.zeros(count),
    }
    events = [
        {"stage": "latch_probe_lift", "physics_step_start": 8, "physics_step_end": 20},
        {"stage": "latch_probe_hold", "physics_step_start": 20, "physics_step_end": 24},
        {"stage": "latch_probe_lower_release", "physics_step_start": 28, "physics_step_end": 36},
        {"stage": "latch_probe_post_release", "physics_step_start": 36, "physics_step_end": 40},
    ]
    return trace, events


def test_formal_feature_is_frozen_finite_49d_and_closed_interval():
    trace, events = synthetic_trace()
    feature, schema = stage_aligned_formal_sensor_feature(trace, events, 480, 4)
    assert feature.shape == (49,)
    assert feature.dtype == np.float64
    assert len(schema) == len(set(schema)) == 49
    assert np.all(np.isfinite(feature))
    mask = motion_stage_mask(trace["physics_step"], events, ["latch_probe_lift"])
    assert np.flatnonzero(mask).tolist() == [2, 3, 4, 5]


def test_reaction_changes_formal_and_reaction_only_features():
    trace, events = synthetic_trace()
    first, _ = stage_aligned_formal_sensor_feature(trace, events, 480, 4)
    first_reaction, _ = reaction_only_feature(trace, events, 480, 4)
    changed = copy.deepcopy(trace)
    changed["sensor_joint_reaction_force_torque"][2:7] *= 3
    second, _ = stage_aligned_formal_sensor_feature(changed, events, 480, 4)
    second_reaction, schema = reaction_only_feature(changed, events, 480, 4)
    assert not np.array_equal(first, second)
    assert not np.array_equal(first_reaction, second_reaction)
    assert second_reaction.shape == (18,)
    assert len(schema) == 18


def test_oracle_and_privileged_metadata_do_not_change_feature():
    trace, events = synthetic_trace()
    first, schema = stage_aligned_formal_sensor_feature(trace, events, 480, 4)
    trace["contact_force_norm"][:] = 999
    trace["contact_active_beads"][:] = 7
    trace["hidden_condition"] = np.asarray(["hidden"] * 12)
    trace["hook_layout"] = np.arange(12)
    second, second_schema = stage_aligned_formal_sensor_feature(trace, events, 480, 4)
    assert np.array_equal(first, second)
    assert schema == second_schema


def test_missing_event_window_fails_without_preload_fallback():
    trace, events = synthetic_trace()
    with pytest.raises(ValueError, match="unload"):
        stage_aligned_formal_sensor_feature(trace, events[:2], 480, 4)
