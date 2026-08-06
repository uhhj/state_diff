"""Render independent OCCP R1 diagnostics and a paired comparison video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _markers(axis, steps, phases, metrics):
    for phase in ('probe', 'post_probe', 'test_pull', 'post_test'):
        indices = np.flatnonzero(phases == phase)
        if len(indices):
            axis.axvline(steps[indices[0]], linestyle='--', linewidth=0.8)
    for key, style, label in (
            ('jam_oracle_contact_onset_step', ':', 'ORACLE ONLY'),
            ('fused_sensor_separation_step', '-.', 'FORMAL SENSOR'),
            ('visual_divergence_step', '--', 'VISUAL')):
        value = metrics.get(key)
        if value is not None:
            axis.axvline(value, linestyle=style, linewidth=1.2, label=label)


def _single_plot(path, steps, phases, metrics, series, ylabel):
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_axes([0.12, 0.14, 0.83, 0.78])
    for label, values in series:
        axis.plot(steps, values, label=label)
    _markers(axis, steps, phases, metrics)
    axis.set_xlabel('absolute physics step')
    axis.set_ylabel(ylabel)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_plots(report_dir):
    report_dir = Path(report_dir)
    metrics = json.loads(
        (report_dir / 'single_pair_r1_summary.json').read_text('utf-8'))
    with np.load(report_dir / 'single_pair_r1_timeseries.npz',
                 allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    steps, phases = data['physics_step'], data['phase'].astype(str)
    _single_plot(
        report_dir / 'formal_sensor_gap.png', steps, phases, metrics,
        [('fused FORMAL SENSOR', data['fused_sensor_gap'])],
        'pooled z-score RMS gap')
    channels = sorted(key for key in data if
                      key.startswith('formal_') and key.endswith('_gap'))
    _single_plot(
        report_dir / 'formal_sensor_channels.png', steps, phases, metrics,
        [(key[7:-4], data[key]) for key in channels],
        'per-channel z-score RMS gap')
    _single_plot(
        report_dir / 'visible_readout_gap.png', steps, phases, metrics,
        [('active readout', data['visible_gap_active_readout']),
         ('all nodes diagnostic', data['visible_gap_all_nodes'])],
        'position RMSE (m)')
    _single_plot(
        report_dir / 'oracle_signed_distance.png', steps, phases, metrics,
        [('Free ORACLE ONLY', data['free_oracle_signed_distance']),
         ('Jam ORACLE ONLY', data['jam_oracle_signed_distance'])],
        'pin minimum signed distance (m)')
    _single_plot(
        report_dir / 'endpoint_progress.png', steps, phases, metrics,
        [('Free', data['free_endpoint_progress']),
         ('Jam', data['jam_endpoint_progress'])],
        'active endpoint progress (m)')
    return metrics, data


def render_comparison(pair_dir, report_dir):
    pair_dir, report_dir = Path(pair_dir), Path(report_dir)
    metrics, data = render_plots(report_dir)
    left_frames = sorted((pair_dir / 'free' / 'frames').glob('*.png'))
    right_frames = sorted(
        (pair_dir / 'right_hidden_jam' / 'frames').glob('*.png'))
    count = min(len(left_frames), len(right_frames))
    output = report_dir / 'free_vs_jam_r1.mp4'
    if count == 0:
        return {'video_available': False, 'error': 'no paired PNG frames'}
    left = cv2.imread(str(left_frames[0]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_frames[0]), cv2.IMREAD_COLOR)
    height = min(left.shape[0], right.shape[0])
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30.,
        (left.shape[1] + right.shape[1], height))
    if not writer.isOpened():
        return {'video_available': False, 'error': 'MP4 writer unavailable'}
    steps, phases = data['physics_step'], data['phase'].astype(str)
    try:
        for index in range(count):
            left = cv2.imread(str(left_frames[index]), cv2.IMREAD_COLOR)[:height]
            right = cv2.imread(str(right_frames[index]), cv2.IMREAD_COLOR)[:height]
            frame = np.concatenate([left, right], axis=1)
            trace_index = int(round(index * (len(steps) - 1) / max(count - 1, 1)))
            lines = [
                'FREE                         RIGHT_HIDDEN_JAM',
                'phase={} step={}'.format(
                    phases[trace_index], int(steps[trace_index])),
                'FORMAL SENSOR gap={:.4f}'.format(
                    float(data['fused_sensor_gap'][trace_index])),
                'ORACLE ONLY jam distance={:.5f} m'.format(
                    float(data['jam_oracle_signed_distance'][trace_index]))]
            for row, line in enumerate(lines):
                cv2.putText(frame, line, (12, 26 + 24 * row),
                            cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return {'video_available': output.exists(), 'output': str(output),
            'frames_written': count, 'error': None,
            'verdict': metrics['verdict']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-dir', required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    print(render_comparison(args.pair_dir, args.report_dir))


if __name__ == '__main__':
    main()
