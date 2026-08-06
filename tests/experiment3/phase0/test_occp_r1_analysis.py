import copy

import numpy as np

from scripts.experiment3.phase0.analyze_occp_single_pair_r1 import analyze_pair_data


def _synthetic():
    count, nodes = 14, 3
    phases = np.asarray(
        ['no_action'] * 4 + ['probe'] * 4 + ['post_probe'] * 2
        + ['test_pull'] * 2 + ['post_test'] * 2)
    steps = np.arange(1, count + 1)
    beads = np.zeros((count, nodes, 3), dtype=np.float64)
    beads[:, :, 0] = np.array([0., .1, .2])
    free_beads, jam_beads = beads.copy(), beads.copy()
    jam_beads[10:, 2, 0] -= .0033
    jam_beads[10:, 1, 1] += .003

    def trace(bead_values):
        payload = {
            'physics_step': steps.copy(), 'phase': phases.copy(),
            'bead_positions': bead_values,
            'oracle_pin_contact_count': np.zeros(count, dtype=np.int64),
            'oracle_pin_contact_force': np.zeros(count),
            'oracle_pin_min_signed_distance': np.full(count, .01),
        }
        fields = {
            'sensor_joint_motor_torque': (6,),
            'sensor_joint_reaction_force_torque': (6, 6),
            'sensor_suction_force_xyz': (3,),
            'sensor_suction_torque_xyz': (3,),
            'ee_tracking_error': (),
        }
        for field, shape in fields.items():
            payload[field] = np.ones((count,) + shape, dtype=np.float64)
        return payload

    free, jam = trace(free_beads), trace(jam_beads)
    for field in (
            'sensor_joint_motor_torque', 'sensor_joint_reaction_force_torque',
            'sensor_suction_force_xyz', 'sensor_suction_torque_xyz',
            'ee_tracking_error'):
        jam[field][5:] += .1
    jam['oracle_pin_contact_count'][5:] = 1
    jam['oracle_pin_contact_force'][5:] = 2.
    jam['oracle_pin_min_signed_distance'][5:] = -.001
    metadata = {
        'layout': {'visible_readout_indices': [2], 'active_endpoint_index': 2,
                   'active_exit_index': 1, 'test_delta': [.03, 0., 0.]},
        'analysis': {'sigma_multiplier': 5., 'consecutive_samples': 3,
                     'final_visible_min_m': .0025,
                     'final_full_cable_min_m': .001,
                     'progress_gap_min_m': .0025,
                     'branch_amplification_min': 2.},
        'fixed_command_arrays_equal': True,
        'physics_step_arrays_equal': True, 'phase_arrays_equal': True,
        'free_to_jam_max_abs': 0., 'initial_visible_bead_rmse': 0.,
        'initial_robot_state_rmse': 0.,
        'branches': {
            'free': {'grasp': {'retained': True}, 'execution_failures': []},
            'right_hidden_jam': {
                'grasp': {'retained': True}, 'execution_failures': []}}}
    return metadata, free, jam


def test_jam_probe_contact_and_free_no_contact_passes_oracle_mechanism():
    metrics, _ = analyze_pair_data(*_synthetic())
    assert metrics['oracle_mechanism_ok']


def test_fused_sensor_onset_uses_absolute_steps():
    metrics, _ = analyze_pair_data(*_synthetic())
    assert metrics['fused_sensor_separation_step'] == 6


def test_sensor_must_precede_visual():
    metrics, _ = analyze_pair_data(*_synthetic())
    assert metrics['fused_sensor_separation_step'] < metrics['visual_divergence_step']


def test_no_action_oracle_contact_forces_scientific_fail():
    metadata, free, jam = _synthetic()
    jam['oracle_pin_contact_count'][1] = 1
    metrics, _ = analyze_pair_data(metadata, free, jam)
    assert metrics['verdict'] == 'PHASE0A_R1_SCIENTIFIC_FAIL'
    assert metrics['failure_cause'] == 'baseline_contact_contamination'


def test_missing_visual_divergence_fails():
    metadata, free, jam = _synthetic()
    jam['bead_positions'][:] = free['bead_positions']
    metrics, _ = analyze_pair_data(metadata, free, jam)
    assert metrics['verdict'] == 'PHASE0A_R1_SCIENTIFIC_FAIL'
    assert metrics['visual_divergence_step'] is None


def test_weak_submillimeter_branch_fails():
    metadata, free, jam = _synthetic()
    delta = free['bead_positions'] - jam['bead_positions']
    jam['bead_positions'][:] = free['bead_positions'] - delta * .1
    metrics, _ = analyze_pair_data(metadata, free, jam)
    assert not metrics['same_fixed_commands_produce_future_branch']


def test_step_array_mismatch_is_engineering_blocker():
    metadata, free, jam = _synthetic()
    jam['physics_step'] = jam['physics_step'] + 1
    metrics, _ = analyze_pair_data(metadata, free, jam)
    assert metrics['verdict'] == 'PHASE0A_R1_ENGINEERING_BLOCKED'
    assert 'physics_step_arrays_differ' in metrics['engineering_blockers']


def test_oracle_onset_cannot_replace_formal_sensor_onset():
    metadata, free, jam = _synthetic()
    for field in (
            'sensor_joint_motor_torque', 'sensor_joint_reaction_force_torque',
            'sensor_suction_force_xyz', 'sensor_suction_torque_xyz',
            'ee_tracking_error'):
        jam[field][:] = free[field]
    metrics, _ = analyze_pair_data(metadata, free, jam)
    assert metrics['jam_oracle_contact_onset_step'] is not None
    assert metrics['fused_sensor_separation_step'] is None
    assert metrics['verdict'] == 'PHASE0A_R1_SCIENTIFIC_FAIL'
