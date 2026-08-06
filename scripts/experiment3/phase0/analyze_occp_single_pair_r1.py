"""Absolute-step analysis for the OCCP Phase 0A-R1 single-pair audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0.occp_common import write_json


FORMAL_CHANNELS = {
    'joint_motor_torque': 'sensor_joint_motor_torque',
    'joint_reaction_wrench': 'sensor_joint_reaction_force_torque',
    'suction_force': 'sensor_suction_force_xyz',
    'suction_torque': 'sensor_suction_torque_xyz',
    'ee_tracking_error': 'ee_tracking_error',
}


def _matrix(trace, field):
    values = np.asarray(trace[field], dtype=np.float64)
    return values.reshape(values.shape[0], -1)


def _gap(free, jam, baseline):
    pooled = np.concatenate([free[baseline], jam[baseline]], axis=0)
    mean = np.mean(pooled, axis=0)
    std = np.maximum(np.std(pooled, axis=0), 1e-6)
    return np.sqrt(np.mean(np.square(
        (free - mean) / std - (jam - mean) / std), axis=1))


def _first_sustained(values, threshold, valid, consecutive):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    for index in range(len(values) - consecutive + 1):
        if (np.all(valid[index:index + consecutive])
                and np.all(values[index:index + consecutive] > threshold)):
            return int(index)
    return None


def _onset(gap, baseline, consecutive):
    baseline_values = np.asarray(gap)[baseline]
    threshold = float(
        np.mean(baseline_values) + 5.0 * np.std(baseline_values))
    index = _first_sustained(gap, threshold, ~baseline, consecutive)
    return threshold, index


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _mean_curvature(beads):
    edges = np.diff(np.asarray(beads, dtype=np.float64)[:, :2], axis=0)
    norms = np.linalg.norm(edges, axis=1)
    cosine = np.sum(edges[:-1] * edges[1:], axis=1) / np.maximum(
        norms[:-1] * norms[1:], 1e-12)
    return float(np.mean(np.arccos(np.clip(cosine, -1, 1))))


def analyze_pair_data(pair_metadata, free, jam):
    free_steps = np.asarray(free['physics_step'], dtype=np.int64)
    jam_steps = np.asarray(jam['physics_step'], dtype=np.int64)
    free_phase = np.asarray(free['phase']).astype(str)
    jam_phase = np.asarray(jam['phase']).astype(str)
    blockers = []
    if not np.array_equal(free_steps, jam_steps):
        blockers.append('physics_step_arrays_differ')
    if not np.array_equal(free_phase, jam_phase):
        blockers.append('phase_arrays_differ')
    common = min(len(free_steps), len(jam_steps))
    if common == 0:
        raise ValueError('empty trace')
    steps, phases = free_steps[:common], free_phase[:common]
    baseline = phases == 'no_action'
    if not np.any(baseline):
        raise ValueError('trace has no no_action baseline')
    sigma = float(pair_metadata.get('analysis', {}).get('sigma_multiplier', 5.0))
    if sigma != 5.0:
        blockers.append('sigma_multiplier_is_not_five')
    consecutive = int(pair_metadata.get('analysis', {}).get(
        'consecutive_samples', 3))

    channel_gaps, channel_metrics = {}, {}
    free_fused, jam_fused = [], []
    raw_nonzero = True
    for name, field in FORMAL_CHANNELS.items():
        left, right = _matrix(free, field)[:common], _matrix(jam, field)[:common]
        free_fused.append(left)
        jam_fused.append(right)
        raw_nonzero = raw_nonzero and bool(
            max(np.max(np.abs(left)), np.max(np.abs(right))) > 1e-8)
        gap = _gap(left, right, baseline)
        threshold = float(np.mean(gap[baseline])
                          + sigma * np.std(gap[baseline]))
        index = _first_sustained(gap, threshold, ~baseline, consecutive)
        channel_gaps[name] = gap
        channel_metrics[name] = {
            'threshold': threshold,
            'post_baseline_max': float(np.max(gap[~baseline])),
            'onset_index': index,
            'onset_step': None if index is None else int(steps[index])}
    fused_gap = _gap(np.concatenate(free_fused, axis=1),
                     np.concatenate(jam_fused, axis=1), baseline)
    fused_threshold = float(np.mean(fused_gap[baseline])
                            + sigma * np.std(fused_gap[baseline]))
    fused_index = _first_sustained(
        fused_gap, fused_threshold, ~baseline, consecutive)
    fused_step = None if fused_index is None else int(steps[fused_index])

    free_beads = np.asarray(free['bead_positions'], dtype=np.float64)[:common]
    jam_beads = np.asarray(jam['bead_positions'], dtype=np.float64)[:common]
    visible_indices = np.asarray(
        pair_metadata['layout']['visible_readout_indices'], dtype=np.int64)
    visible_gap = np.sqrt(np.mean(np.square(
        free_beads[:, visible_indices] - jam_beads[:, visible_indices]),
        axis=(1, 2)))
    all_gap = np.sqrt(np.mean(np.square(free_beads - jam_beads), axis=(1, 2)))
    visual_threshold = float(np.mean(visible_gap[baseline])
                             + sigma * np.std(visible_gap[baseline]))
    visual_index = _first_sustained(
        visible_gap, visual_threshold, ~baseline, consecutive)
    visual_step = None if visual_index is None else int(steps[visual_index])

    free_contact = ((np.asarray(free['oracle_pin_contact_count'])[:common] > 0)
                    | (np.asarray(free['oracle_pin_contact_force'])[:common] > 0))
    jam_contact = ((np.asarray(jam['oracle_pin_contact_count'])[:common] > 0)
                   | (np.asarray(jam['oracle_pin_contact_force'])[:common] > 0))
    free_distance = np.asarray(
        free['oracle_pin_min_signed_distance'], dtype=np.float64)[:common]
    jam_distance = np.asarray(
        jam['oracle_pin_min_signed_distance'], dtype=np.float64)[:common]
    free_contact_indices = np.flatnonzero(free_contact)
    jam_contact_indices = np.flatnonzero(jam_contact)
    jam_crossing = np.flatnonzero(jam_distance <= 0)
    jam_onset = None if not len(jam_contact_indices) else int(
        steps[jam_contact_indices[0]])
    free_onset = None if not len(free_contact_indices) else int(
        steps[free_contact_indices[0]])
    pretest = np.isin(phases, ['no_action', 'probe', 'post_probe'])
    jam_phase_ok = bool(len(jam_contact_indices) and phases[
        jam_contact_indices[0]] in ('probe', 'post_probe'))
    baseline_contaminated = bool(
        np.any(free_contact[baseline]) or np.any(jam_contact[baseline]))
    oracle_mechanism_ok = bool(
        not baseline_contaminated and jam_phase_ok
        and not np.any(free_contact[pretest]))

    test_delta = np.asarray(pair_metadata['layout']['test_delta'], dtype=np.float64)
    direction = test_delta / np.linalg.norm(test_delta)
    active = int(pair_metadata['layout']['active_endpoint_index'])
    exit_index = int(pair_metadata['layout']['active_exit_index'])
    free_progress = np.sum(
        (free_beads[:, active] - free_beads[0, active]) * direction, axis=1)
    jam_progress = np.sum(
        (jam_beads[:, active] - jam_beads[0, active]) * direction, axis=1)
    progress_gap = float(free_progress[-1] - jam_progress[-1])
    final_visible = float(visible_gap[-1])
    final_full = _rmse(free_beads[-1] - jam_beads[-1])
    exit_gap = float(abs(
        (free_beads[-1, exit_index, 1] - free_beads[0, exit_index, 1])
        - (jam_beads[-1, exit_index, 1] - jam_beads[0, exit_index, 1])))
    curvature_gap = float(abs(
        _mean_curvature(free_beads[-1]) - _mean_curvature(jam_beads[-1])))
    pre_probe_visible = float(visible_gap[np.flatnonzero(baseline)[-1]])
    amplification = float(final_visible / max(pre_probe_visible, 1e-6))
    gates = pair_metadata.get('analysis', {})
    future_branch_ok = bool(
        final_visible >= float(gates.get('final_visible_min_m', 0.0025))
        and final_full >= float(gates.get('final_full_cable_min_m', 0.001))
        and abs(progress_gap) >= float(gates.get('progress_gap_min_m', 0.0025))
        and amplification >= float(gates.get('branch_amplification_min', 2.0)))

    branch_meta = pair_metadata['branches']
    failures = (list(branch_meta['free'].get('execution_failures', []))
                + list(branch_meta['right_hidden_jam'].get(
                    'execution_failures', [])))
    engineering_ok = bool(
        not blockers
        and pair_metadata.get('fixed_command_arrays_equal')
        and pair_metadata.get('physics_step_arrays_equal')
        and pair_metadata.get('phase_arrays_equal')
        and float(pair_metadata.get('free_to_jam_max_abs', np.inf)) <= 1e-7
        and branch_meta['free']['grasp']['retained']
        and branch_meta['right_hidden_jam']['grasp']['retained']
        and not failures and raw_nonzero)
    contact_precedes_visual = bool(
        fused_step is not None and visual_step is not None
        and fused_step < visual_step)
    if not engineering_ok:
        verdict = 'PHASE0A_R1_ENGINEERING_BLOCKED'
    elif oracle_mechanism_ok and contact_precedes_visual and future_branch_ok:
        verdict = 'PHASE0A_R1_SINGLE_PAIR_COMPLETE'
    else:
        verdict = 'PHASE0A_R1_SCIENTIFIC_FAIL'
    failure_cause = None
    if blockers or not engineering_ok:
        failure_cause = ','.join(blockers + failures) or 'engineering_gate_failed'
    elif baseline_contaminated:
        failure_cause = 'baseline_contact_contamination'
    elif not oracle_mechanism_ok:
        failure_cause = 'probe_only_oracle_mechanism_failed'
    elif not contact_precedes_visual:
        failure_cause = 'formal_sensor_does_not_precede_visual'
    elif not future_branch_ok:
        failure_cause = 'future_branch_below_r1_thresholds'

    metrics = {
        'verdict': verdict,
        'scientific_status': 'single-pair smoke only; no held-out conclusion',
        'failure_cause': failure_cause, 'engineering_blockers': blockers,
        'initial_full_state_max_abs': float(
            pair_metadata.get('free_to_jam_max_abs', np.inf)),
        'initial_visible_bead_rmse': float(
            pair_metadata.get('initial_visible_bead_rmse', np.nan)),
        'initial_robot_state_rmse': float(
            pair_metadata.get('initial_robot_state_rmse', np.nan)),
        'fixed_command_arrays_equal': bool(
            pair_metadata.get('fixed_command_arrays_equal')),
        'physics_step_arrays_equal': bool(np.array_equal(free_steps, jam_steps)),
        'phase_arrays_equal': bool(np.array_equal(free_phase, jam_phase)),
        'formal_sensor_nonzero': raw_nonzero,
        'formal_sensor_channels': channel_metrics,
        'fused_sensor_threshold': fused_threshold,
        'fused_sensor_post_baseline_max': float(np.max(fused_gap[~baseline])),
        'fused_sensor_separation_step': fused_step,
        'visual_threshold': visual_threshold,
        'visual_divergence_step': visual_step,
        'free_no_action_oracle_contact_samples': int(
            np.count_nonzero(free_contact[baseline])),
        'jam_no_action_oracle_contact_samples': int(
            np.count_nonzero(jam_contact[baseline])),
        'free_no_action_min_signed_distance': float(np.min(free_distance[baseline])),
        'jam_no_action_min_signed_distance': float(np.min(jam_distance[baseline])),
        'jam_oracle_contact_onset_step': jam_onset,
        'free_oracle_contact_onset_step': free_onset,
        'jam_min_signed_distance_crossing_step': (
            None if not len(jam_crossing) else int(steps[jam_crossing[0]])),
        'free_min_signed_distance_min': float(np.min(free_distance)),
        'free_contact_through_post_probe': bool(np.any(free_contact[pretest])),
        'oracle_mechanism_ok': oracle_mechanism_ok,
        'formal_contact_precedes_visual_divergence': contact_precedes_visual,
        'oracle_to_sensor_latency_steps': (
            None if jam_onset is None or fused_step is None else fused_step - jam_onset),
        'sensor_to_visual_lead_steps': (
            None if fused_step is None or visual_step is None else visual_step - fused_step),
        'final_visible_rmse': final_visible,
        'final_full_cable_rmse': final_full,
        'active_endpoint_progress_gap': progress_gap,
        'exit_lateral_displacement_gap': exit_gap,
        'mean_curvature_gap': curvature_gap,
        'branch_amplification_ratio': amplification,
        'same_fixed_commands_produce_future_branch': future_branch_ok,
        'execution_failures': failures}
    timeseries = {
        'physics_step': steps, 'phase': phases,
        'fused_sensor_gap': fused_gap,
        'visible_gap_active_readout': visible_gap,
        'visible_gap_all_nodes': all_gap,
        'free_oracle_signed_distance': free_distance,
        'jam_oracle_signed_distance': jam_distance,
        'free_endpoint_progress': free_progress,
        'jam_endpoint_progress': jam_progress}
    for name, values in channel_gaps.items():
        timeseries['formal_' + name + '_gap'] = values
    return metrics, timeseries


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _summary(metrics):
    return '\n'.join([
        '# OCCP Phase 0A-R1 single-pair summary', '',
        '- Verdict: `{}`'.format(metrics['verdict']),
        '- Scientific status: {}'.format(metrics['scientific_status']),
        '- Oracle mechanism OK: {}'.format(metrics['oracle_mechanism_ok']),
        '- Fused formal sensor onset: {}'.format(
            metrics['fused_sensor_separation_step']),
        '- Visual onset: {}'.format(metrics['visual_divergence_step']),
        '- Final active-readout RMSE: {:.8f} m'.format(
            metrics['final_visible_rmse']),
        '- Failure cause: {}'.format(metrics['failure_cause']), '',
        'Oracle fields are diagnostic only and are excluded from formal sensors.',
        'This is a single-pair smoke result, not a held-out conclusion.', ''])


def analyze_pair(pair_dir, report_dir):
    pair_dir, report_dir = Path(pair_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((pair_dir / 'metadata.json').read_text('utf-8'))
    free = load_trajectory(pair_dir / 'free' / 'trajectory.npz')
    jam = load_trajectory(pair_dir / 'right_hidden_jam' / 'trajectory.npz')
    metrics, timeseries = analyze_pair_data(metadata, free, jam)
    write_json(report_dir / 'single_pair_r1_summary.json', metrics)
    (report_dir / 'single_pair_r1_summary.md').write_text(
        _summary(metrics), encoding='utf-8')
    np.savez_compressed(report_dir / 'single_pair_r1_timeseries.npz', **timeseries)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    print(analyze_pair(args.pair_dir, args.report_dir)['verdict'])


if __name__ == '__main__':
    main()
