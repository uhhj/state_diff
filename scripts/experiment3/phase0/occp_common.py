"""Pure configuration, serialization, and trace helpers for OCCP Phase 0A."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


CONDITIONS = ('free', 'right_hidden_jam')
LEGACY_GEOMETRY_FIELDS = {
    'jam_clearance_scale', 'free_clearance_scale', 'pin_radius_scale',
    'occluder_length_x', 'occluder_width_y'}


def config_schema(config):
    return ('r1' if 'acquisition_speed' in config.get('motion', {})
            else 'legacy_phase0a')


def load_config(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config):
    if config.get('task_name') != 'ccda-occp-audit':
        raise ValueError('task_name must be ccda-occp-audit')
    if config.get('dataset_role') != 'ccda_audit':
        raise ValueError('dataset_role must be ccda_audit')
    if int(config.get('seed', -1)) < 0 or not config.get('pair_id'):
        raise ValueError('seed and pair_id are required')
    execution = config['execution']
    if not execution.get('deterministic'):
        raise ValueError('the Phase 0A pair must be deterministic')
    schema = config_schema(config)
    for name in (
            'hz', 'control_substeps', 'no_action_steps',
            'post_probe_steps', 'post_test_steps'):
        if int(execution[name]) <= 0:
            raise ValueError('{} must be positive'.format(name))
    if int(execution['post_action_settle_steps']) < 0:
        raise ValueError('post_action_settle_steps must be non-negative')
    if float(execution['initial_settle_seconds']) < 0:
        raise ValueError('initial_settle_seconds must be non-negative')
    motion = config['motion']
    speed_key = 'acquisition_speed' if schema == 'r1' else 'speed'
    if float(motion[speed_key]) <= 0 or float(motion['joint_tolerance']) <= 0:
        raise ValueError('motion speed and tolerance must be positive')
    if float(motion['grasp_height_offset']) < 0:
        raise ValueError('grasp_height_offset must be non-negative')
    if int(config['trace']['stride']) <= 0:
        raise ValueError('trace stride must be positive')
    image_size = config['camera']['image_size']
    if len(image_size) != 2 or min(int(value) for value in image_size) <= 0:
        raise ValueError('camera image_size is invalid')
    if float(config['camera']['fps']) <= 0:
        raise ValueError('camera fps must be positive')
    if int(config['camera']['frame_stride']) <= 0:
        raise ValueError('camera frame_stride must be positive')
    geometry = config.get('geometry', {})
    if schema == 'r1':
        legacy = sorted(LEGACY_GEOMETRY_FIELDS.intersection(geometry))
        if legacy:
            raise ValueError(
                'R1 config contains legacy geometry fields: {}'.format(legacy))
        for name in ('pre_snapshot_settle_steps',):
            if int(execution[name]) <= 0:
                raise ValueError('{} must be positive'.format(name))
        for name in ('probe_command_steps', 'test_command_steps'):
            if int(motion[name]) <= 0:
                raise ValueError('{} must be positive'.format(name))
        if int(config['trace']['stride']) != 1:
            raise ValueError('R1 trace stride must be one')


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False) + '\n',
        encoding='utf-8')


def build_action_script(active_position, active_orientation, layout, speed):
    active_position = np.asarray(active_position, dtype=np.float64)
    orientation = [float(value) for value in active_orientation]
    probe = active_position + np.asarray(layout['probe_delta'], dtype=np.float64)
    test = active_position + np.asarray(layout['test_delta'], dtype=np.float64)
    return [
        {
            'name': 'probe',
            'phase': 'probe',
            'target_position': probe.astype(float).tolist(),
            'target_orientation': orientation,
            'speed': float(speed),
        },
        {
            'name': 'test_pull',
            'phase': 'test_pull',
            'target_position': test.astype(float).tolist(),
            'target_orientation': orientation,
            'speed': float(speed),
        },
    ]


def copy_action_script(action_script):
    return copy.deepcopy(action_script)


def trace_to_arrays(trace):
    if not trace:
        raise ValueError('trace is empty')
    fixed_fields = (
        'physics_step', 'phase', 'bead_positions', 'bead_velocities',
        'joint_positions', 'joint_velocities', 'ee_position',
        'ee_orientation', 'ee_linear_velocity', 'ee_angular_velocity',
        'ee_target_position', 'ee_tracking_error',
        'sensor_joint_motor_torque',
        'sensor_joint_reaction_force_torque',
        'sensor_suction_force_xyz', 'sensor_suction_torque_xyz',
        'sensor_grasp_active', 'sensor_constraint_available',
        'oracle_pin_contact_force', 'oracle_pin_contact_count')
    fixed_fields = fixed_fields + (
        'oracle_pin_min_signed_distance', 'visible_mask')
    return {
        field: np.asarray([row[field] for row in trace])
        for field in fixed_fields
    }


def relative_or_absolute(path, root):
    value = Path(path)
    return value if value.is_absolute() else Path(root) / value
