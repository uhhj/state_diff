import numpy as np
import pytest

from state_diff.env.block_pushing.policy_rate_metrics import (
    displacement_from_reference, formal_feature_scale_floors, onset_with_floor,
    resample_phase_aligned_last, resample_phase_aligned_mean)


def test_phase_aligned_resampling_and_boundaries():
    values = np.arange(96).reshape(48, 2)
    phases = np.asarray(["a"] * 24 + ["b"] * 24)
    steps = np.arange(1, 49)
    mean = resample_phase_aligned_mean(values, phases, steps, 24)
    last = resample_phase_aligned_last(values, phases, steps, 24)
    assert np.array_equal(mean.phase, ["a", "b"])
    assert np.array_equal(mean.physics_step_end, [24, 48])
    assert np.array_equal(last.values, [values[23], values[47]])
    with pytest.raises(ValueError):
        resample_phase_aligned_mean(values[:-1], phases[:-1], steps[:-1], 24)


def test_scale_layout_nonzero_threshold_and_policy_lead():
    config = {"analysis": {"sensor_scale_floors": {
        "joint_motor_torque_nm": .02, "joint_reaction_force_n": .1,
        "joint_reaction_torque_nm": .01, "ee_tracking_error_m": .0002,
        "ee_contact_force_n": .05, "ee_contact_torque_nm": .005}}}
    floors = formal_feature_scale_floors(config)
    assert floors.shape == (45,)
    assert np.array_equal(floors[6:12], [.1, .1, .1, .01, .01, .01])
    threshold, onset = onset_with_floor(
        np.array([0., 0., 2., 3.]), np.array([True, True, False, False]), 5, 1, 1)
    assert threshold == 1 and onset == 2
    _, same = onset_with_floor(
        np.array([0., 2.]), np.array([True, False]), 5, 1, 1)
    assert same == 1


def test_local_displacement_metric():
    reference = np.zeros((4, 3))
    positions = np.stack([reference, reference + [1, 2, 0]])
    result = displacement_from_reference(positions, reference, np.array([0, 2]))
    assert np.array_equal(result[1], [1, 2, 0])
