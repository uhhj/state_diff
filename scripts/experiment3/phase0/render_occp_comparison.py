"""Render OCCP pair diagnostics and a side-by-side comparison video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _plot(path, steps, values, ylabel, phases):
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_axes([0.12, 0.14, 0.83, 0.78])
    axis.plot(steps, values)
    axis.set_xlabel('physics step')
    axis.set_ylabel(ylabel)
    for phase in ('probe', 'test_pull'):
        indices = np.flatnonzero(phases == phase)
        if len(indices):
            axis.axvline(steps[indices[0]], linestyle='--', linewidth=1)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_diagnostic_plots(report_dir):
    report_dir = Path(report_dir)
    with np.load(
            report_dir / 'single_pair_timeseries.npz', allow_pickle=False) as data:
        steps = data['physics_step']
        phases = data['phase'].astype(str)
        _plot(
            report_dir / 'force_timeseries.png', steps,
            data['sensor_gap'], 'formal fused sensor gap', phases)
        _plot(
            report_dir / 'visible_node_distance.png', steps,
            data['visible_gap'], 'visible node RMSE (m)', phases)
        figure = plt.figure(figsize=(8, 4.5))
        axis = figure.add_axes([0.12, 0.14, 0.83, 0.78])
        axis.plot(steps, data['free_endpoint_progress'], label='free')
        axis.plot(steps, data['jam_endpoint_progress'], label='right_hidden_jam')
        axis.set_xlabel('physics step')
        axis.set_ylabel('active endpoint progress (m)')
        axis.legend()
        for phase in ('probe', 'test_pull'):
            indices = np.flatnonzero(phases == phase)
            if len(indices):
                axis.axvline(steps[indices[0]], linestyle='--', linewidth=1)
        figure.savefig(report_dir / 'endpoint_progress.png', dpi=150)
        plt.close(figure)


def render_comparison(pair_dir, output, report_dir=None):
    pair_dir = Path(pair_dir)
    output = Path(output)
    report_dir = Path(report_dir) if report_dir else output.parent
    render_diagnostic_plots(report_dir)
    free_frames = sorted((pair_dir / 'free' / 'frames').glob('*.png'))
    jam_frames = sorted(
        (pair_dir / 'right_hidden_jam' / 'frames').glob('*.png'))
    count = min(len(free_frames), len(jam_frames))
    if count == 0:
        return {'video_available': False, 'error': 'no paired PNG frames'}
    with np.load(
            report_dir / 'single_pair_timeseries.npz', allow_pickle=False) as data:
        phases = data['phase'].astype(str)
        steps = data['physics_step']
        sensor_gap = data['sensor_gap']
        oracle = data['jam_oracle_pin_force']
    first = cv2.imread(str(free_frames[0]), cv2.IMREAD_COLOR)
    second = cv2.imread(str(jam_frames[0]), cv2.IMREAD_COLOR)
    if first is None or second is None:
        return {'video_available': False, 'error': 'failed to read PNG frames'}
    height = min(first.shape[0], second.shape[0])
    width = first.shape[1] + second.shape[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30., (width, height))
    if not writer.isOpened():
        return {'video_available': False, 'error': 'MP4 writer unavailable'}
    try:
        for index in range(count):
            left = cv2.imread(str(free_frames[index]), cv2.IMREAD_COLOR)
            right = cv2.imread(str(jam_frames[index]), cv2.IMREAD_COLOR)
            if left is None or right is None:
                continue
            left = left[:height]
            right = right[:height]
            frame = np.concatenate([left, right], axis=1)
            trace_index = int(round(
                index * (len(steps) - 1) / max(count - 1, 1)))
            lines = [
                'FREE                         RIGHT_HIDDEN_JAM',
                'phase={} step={} sensor_gap={:.4f}'.format(
                    phases[trace_index], int(steps[trace_index]),
                    float(sensor_gap[trace_index])),
                'ORACLE_ONLY pin_force={:.4f}'.format(
                    float(oracle[trace_index])),
            ]
            for row, line in enumerate(lines):
                cv2.putText(
                    frame, line, (12, 26 + 24 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return {
        'video_available': output.exists(),
        'output': str(output),
        'frames_written': count,
        'error': None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-dir')
    args = parser.parse_args()
    print(render_comparison(args.pair_dir, args.output, args.report_dir))


if __name__ == '__main__':
    main()
