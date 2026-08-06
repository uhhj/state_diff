import numpy as np

from scripts.experiment3.phase0.analyze_occp_single_pair import (
    analyze_pair_data)


def fake_trace(jam=False):
    length, beads = 16, 8
    phase = np.asarray(
        ['no_action'] * 4 + ['probe'] * 4
        + ['test_pull'] * 4 + ['post_test'] * 4)
    positions = np.zeros((length, beads, 3), dtype=np.float64)
    positions[:, :, 0] = np.arange(beads)[None, :] * 0.01
    if jam:
        positions[7:, :, 1] = np.linspace(0., 0.02, length - 7)[:, None]
    zeros_j = np.zeros((length, 6))
    reaction = np.zeros((length, 6, 6))
    suction = np.ones((length, 3))
    if jam:
        suction[5:, 0] += 2.
    return {
        'physics_step': np.arange(length) * 2,
        'phase': phase,
        'bead_positions': positions,
        'bead_velocities': np.zeros_like(positions),
        'joint_positions': zeros_j,
        'joint_velocities': zeros_j,
        'ee_position': np.zeros((length, 3)),
        'ee_orientation': np.tile([0., 0., 0., 1.], (length, 1)),
        'ee_linear_velocity': np.zeros((length, 3)),
        'ee_angular_velocity': np.zeros((length, 3)),
        'ee_tracking_error': np.zeros(length),
        'sensor_joint_motor_torque': np.ones((length, 6)),
        'sensor_joint_reaction_force_torque': reaction,
        'sensor_suction_force_xyz': suction,
        'sensor_suction_torque_xyz': np.zeros((length, 3)),
        'oracle_pin_contact_force': (
            np.r_[np.zeros(6), np.ones(length - 6)] if jam
            else np.zeros(length)),
        'oracle_pin_contact_count': (
            np.r_[np.zeros(6), np.ones(length - 6)] if jam
            else np.zeros(length)),
    }


def test_analysis_keeps_oracle_separate_and_detects_future_branch():
    branch = {
        'execution_failure': None,
        'grasp': {'retained': True},
    }
    metadata = {
        'free_to_jam_max_abs': 0.,
        'action_payload_equal': True,
        'layout': {
            'visible_readout_indices': [0, 1, 2],
            'active_endpoint_index': 7,
            'pin_center': [0.04, 0.01, 0.03],
            'test_delta': [0.05, 0.02, 0.],
        },
        'branches': {
            'free': branch,
            'right_hidden_jam': branch,
        },
    }
    metrics, timeseries = analyze_pair_data(
        metadata, fake_trace(False), fake_trace(True))
    assert metrics['formal_sensor_nonzero']
    assert metrics['same_action_produces_future_branch']
    assert metrics['oracle_first_pin_contact_step_ORACLE_ONLY'] is not None
    assert 'sensor_gap' in timeseries
    assert not any('oracle' in key for key in metrics if key.startswith('formal_'))
