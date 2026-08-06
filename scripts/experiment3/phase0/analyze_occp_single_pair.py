"""Analyze one OCCP smoke pair without fitting a classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0.occp_common import write_json


def _rmse(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _phase_aligned_indices(free, jam):
    free_phase = np.asarray(free['phase']).astype(str)
    jam_phase = np.asarray(jam['phase']).astype(str)
    free_indices, jam_indices, phases = [], [], []
    for phase in ('no_action', 'probe', 'test_pull', 'post_test'):
        left = np.flatnonzero(free_phase == phase)
        right = np.flatnonzero(jam_phase == phase)
        target = min(len(left), len(right))
        if target == 0:
            continue
        left_pick = np.rint(np.linspace(0, len(left) - 1, target)).astype(int)
        right_pick = np.rint(np.linspace(0, len(right) - 1, target)).astype(int)
        free_indices.extend(left[left_pick].tolist())
        jam_indices.extend(right[right_pick].tolist())
        phases.extend([phase] * target)
    if not free_indices:
        raise ValueError('paired traces have no common phases')
    return (
        np.asarray(free_indices, dtype=np.int64),
        np.asarray(jam_indices, dtype=np.int64),
        np.asarray(phases),
    )


def _sensor_matrix(trace, indices):
    fields = (
        'sensor_joint_motor_torque',
        'sensor_joint_reaction_force_torque',
        'sensor_suction_force_xyz',
        'sensor_suction_torque_xyz',
        'ee_tracking_error')
    columns = []
    for field in fields:
        values = np.asarray(trace[field], dtype=np.float64)[indices]
        columns.append(values.reshape(values.shape[0], -1))
    return np.concatenate(columns, axis=1)


def _first_sustained(values, threshold, start_index, consecutive=3):
    values = np.asarray(values, dtype=np.float64)
    for index in range(int(start_index), len(values) - consecutive + 1):
        if np.all(values[index:index + consecutive] > threshold):
            return int(index)
    return None


def _mean_curvature(beads):
    xy = np.asarray(beads, dtype=np.float64)[:, :2]
    edges = np.diff(xy, axis=0)
    norms = np.linalg.norm(edges, axis=1)
    valid = (norms[:-1] > 1e-12) & (norms[1:] > 1e-12)
    if not np.any(valid):
        return 0.0
    cosine = np.sum(edges[:-1] * edges[1:], axis=1) / np.maximum(
        norms[:-1] * norms[1:], 1e-12)
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.mean(angles[valid]))


def analyze_pair_data(pair_metadata, free, jam):
    free_idx, jam_idx, phases = _phase_aligned_indices(free, jam)
    free_beads = np.asarray(free['bead_positions'], dtype=np.float64)[free_idx]
    jam_beads = np.asarray(jam['bead_positions'], dtype=np.float64)[jam_idx]
    visible_indices = np.asarray(
        pair_metadata['layout']['visible_readout_indices'], dtype=np.int64)
    visible_gap = np.sqrt(np.mean(np.square(
        free_beads[:, visible_indices] - jam_beads[:, visible_indices]),
        axis=(1, 2)))

    free_sensor = _sensor_matrix(free, free_idx)
    jam_sensor = _sensor_matrix(jam, jam_idx)
    baseline = phases == 'no_action'
    if not np.any(baseline):
        raise ValueError('trace has no no_action baseline')
    baseline_values = np.concatenate(
        [free_sensor[baseline], jam_sensor[baseline]], axis=0)
    sensor_mean = np.mean(baseline_values, axis=0)
    sensor_std = np.std(baseline_values, axis=0)
    sensor_std = np.where(sensor_std < 1e-6, 1e-6, sensor_std)
    free_z = (free_sensor - sensor_mean) / sensor_std
    jam_z = (jam_sensor - sensor_mean) / sensor_std
    sensor_gap = np.sqrt(np.mean(np.square(free_z - jam_z), axis=1))

    baseline_end = int(np.flatnonzero(baseline)[-1] + 1)
    sensor_baseline = sensor_gap[baseline]
    visible_baseline = visible_gap[baseline]
    sensor_threshold = float(
        np.mean(sensor_baseline) + 5 * np.std(sensor_baseline))
    visible_threshold = float(
        np.mean(visible_baseline) + 5 * np.std(visible_baseline))
    sensor_onset = _first_sustained(
        sensor_gap, sensor_threshold, baseline_end, consecutive=3)
    visual_onset = _first_sustained(
        visible_gap, visible_threshold, baseline_end, consecutive=3)

    initial_visible_rmse = float(
        pair_metadata.get('initial_visible_bead_rmse', _rmse(
            free_beads[0, visible_indices] - jam_beads[0, visible_indices])))
    final_visible_rmse = float(visible_gap[-1])
    final_full_rmse = _rmse(free_beads[-1] - jam_beads[-1])
    test_delta = np.asarray(pair_metadata['layout']['test_delta'], dtype=np.float64)
    direction = test_delta / np.linalg.norm(test_delta)
    active_index = int(pair_metadata['layout']['active_endpoint_index'])
    free_progress = np.sum(
        (free_beads[:, active_index] - free_beads[0, active_index]) * direction,
        axis=1)
    jam_progress = np.sum(
        (jam_beads[:, active_index] - jam_beads[0, active_index]) * direction,
        axis=1)
    progress_gap = float(free_progress[-1] - jam_progress[-1])
    pin_x = float(pair_metadata['layout']['pin_center'][0])
    pin_index = int(np.argmin(np.abs(free_beads[0, :, 0] - pin_x)))
    exit_lateral_gap = float(abs(
        (free_beads[-1, pin_index, 1] - free_beads[0, pin_index, 1])
        - (jam_beads[-1, pin_index, 1] - jam_beads[0, pin_index, 1])))
    curvature_gap = float(abs(
        _mean_curvature(free_beads[-1]) - _mean_curvature(jam_beads[-1])))
    pre_probe_visible = float(visible_gap[np.flatnonzero(baseline)[-1]])
    amplification = float(
        final_visible_rmse / max(pre_probe_visible, 1e-6))

    jam_oracle = np.asarray(
        jam['oracle_pin_contact_force'], dtype=np.float64)[jam_idx]
    jam_oracle_count = np.asarray(
        jam['oracle_pin_contact_count'], dtype=np.int64)[jam_idx]
    oracle_indices = np.flatnonzero((jam_oracle > 0) | (jam_oracle_count > 0))
    oracle_onset = int(oracle_indices[0]) if len(oracle_indices) else None
    raw_sensor_nonzero = bool(
        np.max(np.abs(free_sensor)) > 1e-8
        and np.max(np.abs(jam_sensor)) > 1e-8)
    free_meta = pair_metadata['branches']['free']
    jam_meta = pair_metadata['branches']['right_hidden_jam']
    execution_failures = [
        value for value in (
            free_meta.get('execution_failure'),
            jam_meta.get('execution_failure')) if value]
    engineering_ok = bool(
        pair_metadata.get('action_payload_equal')
        and float(pair_metadata.get('free_to_jam_max_abs', np.inf)) <= 1e-7
        and free_meta['grasp']['retained']
        and jam_meta['grasp']['retained']
        and not execution_failures)
    contact_precedes_visual = bool(
        sensor_onset is not None
        and visual_onset is not None
        and sensor_onset < visual_onset)
    future_branch = bool(final_full_rmse > 1e-4 and final_visible_rmse > 1e-5)
    if not engineering_ok or not raw_sensor_nonzero:
        verdict = 'PHASE0A_BLOCKED'
    elif contact_precedes_visual and future_branch:
        verdict = 'PHASE0A_SINGLE_PAIR_COMPLETE'
    else:
        verdict = 'PHASE0A_SCIENTIFIC_FAIL'

    aligned_step = np.rint((
        np.asarray(free['physics_step'])[free_idx]
        + np.asarray(jam['physics_step'])[jam_idx]) / 2).astype(np.int64)
    metrics = {
        'verdict': verdict,
        'scientific_status': 'single-pair smoke only; no held-out conclusion',
        'initial_full_state_max_abs': float(
            pair_metadata['free_to_jam_max_abs']),
        'initial_visible_bead_rmse': initial_visible_rmse,
        'initial_robot_state_rmse': float(
            pair_metadata.get('initial_robot_state_rmse', 0.0)),
        'initial_velocity_rmse': float(
            pair_metadata.get('initial_velocity_rmse', 0.0)),
        'action_payload_equal': bool(pair_metadata['action_payload_equal']),
        'grasp_retained_free': bool(free_meta['grasp']['retained']),
        'grasp_retained_jam': bool(jam_meta['grasp']['retained']),
        'formal_sensor_nonzero': raw_sensor_nonzero,
        'formal_sensor_threshold': sensor_threshold,
        'formal_sensor_baseline_mean': float(np.mean(sensor_baseline)),
        'formal_sensor_baseline_std': float(np.std(sensor_baseline)),
        'formal_sensor_baseline_max': float(np.max(sensor_baseline)),
        'formal_sensor_post_baseline_max': float(
            np.max(sensor_gap[baseline_end:])) if baseline_end < len(sensor_gap)
            else 0.0,
        'formal_sensor_separation_index': sensor_onset,
        'formal_sensor_separation_step': (
            None if sensor_onset is None else int(aligned_step[sensor_onset])),
        'visual_threshold': visible_threshold,
        'visual_baseline_mean': float(np.mean(visible_baseline)),
        'visual_baseline_std': float(np.std(visible_baseline)),
        'visual_baseline_max': float(np.max(visible_baseline)),
        'visual_post_baseline_max': float(
            np.max(visible_gap[baseline_end:])) if baseline_end < len(visible_gap)
            else 0.0,
        'visual_divergence_index': visual_onset,
        'visual_divergence_step': (
            None if visual_onset is None else int(aligned_step[visual_onset])),
        'final_visible_rmse': final_visible_rmse,
        'final_full_cable_rmse': final_full_rmse,
        'active_endpoint_progress_gap': progress_gap,
        'exit_lateral_displacement_gap': exit_lateral_gap,
        'mean_curvature_gap': curvature_gap,
        'branch_amplification_ratio': amplification,
        'oracle_first_pin_contact_index_ORACLE_ONLY': oracle_onset,
        'oracle_first_pin_contact_step_ORACLE_ONLY': (
            None if oracle_onset is None else int(aligned_step[oracle_onset])),
        'oracle_peak_pin_force_ORACLE_ONLY': float(np.max(jam_oracle)),
        'oracle_contact_bead_count_ORACLE_ONLY': int(np.max(jam_oracle_count)),
        'formal_contact_precedes_visual_divergence': contact_precedes_visual,
        'same_action_produces_future_branch': future_branch,
        'execution_failures': execution_failures,
        'smoke_debug_history': pair_metadata.get('smoke_debug_history', []),
        'smoke_threshold_note': (
            '5-sigma/K=3 onset and minimal nonzero deformation checks are '
            'diagnostic smoke criteria, not frozen scientific gates'),
    }
    timeseries = {
        'physics_step': aligned_step,
        'phase': phases,
        'sensor_gap': sensor_gap,
        'visible_gap': visible_gap,
        'free_endpoint_progress': free_progress,
        'jam_endpoint_progress': jam_progress,
        'jam_oracle_pin_force': jam_oracle,
    }
    return metrics, timeseries


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _summary_markdown(metrics):
    lines = [
        '# OCCP Phase 0A single-pair summary',
        '',
        '- Verdict: `{}`'.format(metrics['verdict']),
        '- Scientific status: {}'.format(metrics['scientific_status']),
        '- Formal sensor separation step: {}'.format(
            metrics['formal_sensor_separation_step']),
        '- Visual divergence step: {}'.format(metrics['visual_divergence_step']),
        '- Final visible RMSE: {:.8f} m'.format(
            metrics['final_visible_rmse']),
        '- Final full cable RMSE: {:.8f} m'.format(
            metrics['final_full_cable_rmse']),
        '- Formal contact precedes visual divergence: {}'.format(
            metrics['formal_contact_precedes_visual_divergence']),
        '- Same action produces a future branch: {}'.format(
            metrics['same_action_produces_future_branch']),
        '',
        'Oracle fields are marked `ORACLE_ONLY` and were not used as formal input.',
        '',
        'This is a single-pair smoke result, not a held-out scientific conclusion.',
    ]
    if metrics.get('smoke_debug_history'):
        lines.extend(['', '## Smoke debugging history', ''])
        for change in metrics['smoke_debug_history']:
            lines.append(
                '- Trial {trial}: `{parameter}` {before} → {after}. {reason}'
                .format(**change))
    return '\n'.join(lines) + '\n'


def analyze_pair(pair_dir, report_dir):
    pair_dir = Path(pair_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pair_metadata = json.loads((pair_dir / 'metadata.json').read_text('utf-8'))
    free = load_trajectory(pair_dir / 'free' / 'trajectory.npz')
    jam = load_trajectory(
        pair_dir / 'right_hidden_jam' / 'trajectory.npz')
    metrics, timeseries = analyze_pair_data(pair_metadata, free, jam)
    with np.load(pair_dir / 'base_state.npz', allow_pickle=False) as base:
        beads = np.asarray(base['bead_positions'], dtype=np.float64)
        pin = np.asarray(pair_metadata['layout']['pin_center'], dtype=np.float64)
        planar = np.linalg.norm(beads[:, :2] - pin[None, :2], axis=1)
        nearest = int(np.argmin(planar))
        cable_radius = float(pair_metadata['layout']['cable_diameter']) / 2
        pin_radius = float(pair_metadata['layout']['pin_radius'])
        metrics['snapshot_bead_y_min'] = float(np.min(beads[:, 1]))
        metrics['snapshot_bead_y_max'] = float(np.max(beads[:, 1]))
        metrics['snapshot_pin_nearest_bead_index'] = nearest
        metrics['snapshot_pin_surface_clearance'] = float(
            planar[nearest] - cable_radius - pin_radius)
        metrics['snapshot_active_bead_y'] = float(
            beads[int(pair_metadata['layout']['active_endpoint_index']), 1])
        metrics['snapshot_ee_y'] = float(base['ee_position'][1])
        metrics['snapshot_bead_speed_max'] = float(np.max(np.linalg.norm(
            np.asarray(base['bead_linear_velocities'], dtype=np.float64),
            axis=1)))
        metrics['snapshot_joint_speed_max'] = float(np.max(np.abs(
            np.asarray(base['joint_velocities'], dtype=np.float64))))
    write_json(report_dir / 'single_pair_summary.json', metrics)
    (report_dir / 'single_pair_summary.md').write_text(
        _summary_markdown(metrics), encoding='utf-8')
    np.savez_compressed(report_dir / 'single_pair_timeseries.npz', **timeseries)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    metrics = analyze_pair(args.pair_dir, args.report_dir)
    print(metrics['verdict'])


if __name__ == '__main__':
    main()
