"""Run one exact free/right-hidden-jam OCCP R1 counterfactual pair."""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_ROOT = REPO_ROOT / 'external' / 'deformable-ravens'
for root in (REPO_ROOT, SIM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import pybullet as p  # noqa: E402
from ravens import Environment, tasks  # noqa: E402
from ravens.ccda_video import CCDAVideoRecorder  # noqa: E402
from ravens.tasks.ccda_occp_geometry import (  # noqa: E402
    OCCPGeometryConfig, compute_occp_layout)

from scripts.experiment3.phase0.occp_commands import (  # noqa: E402
    build_fixed_command_script, command_arrays, copy_fixed_command_script,
    execute_fixed_command_phase)
from scripts.experiment3.phase0.occp_common import (  # noqa: E402
    CONDITIONS, config_schema, load_config, trace_to_arrays, write_json)
from scripts.experiment3.phase0.occp_snapshot import (  # noqa: E402
    capture_python_runtime_state, capture_world_state, max_state_difference,
    restore_python_runtime_state)


def camera_config(config):
    image_size = tuple(int(value) for value in config['camera']['image_size'])
    return {
        'image_size': image_size,
        'intrinsics': (450., 0., image_size[1] / 2,
                       0., 450., image_size[0] / 2, 0., 0., 1.),
        'position': np.asarray([0.50, 0.0, 0.75], dtype=np.float64),
        'rotation': p.getQuaternionFromEuler([0., np.pi, -np.pi / 2]),
        'zrange': (0.01, 3.0), 'noise': False}


def grasp_status(env, active_bead_id):
    constraint = getattr(env.ee, 'contact_constraint', None)
    child_matches = False
    if constraint is not None:
        try:
            child_matches = int(p.getConstraintInfo(int(constraint))[2]) == int(
                active_bead_id)
        except Exception:
            child_matches = False
    retained = bool(getattr(env.ee, 'activated', False)
                    and constraint is not None and child_matches)
    return {'activated': bool(getattr(env.ee, 'activated', False)),
            'constraint_available': constraint is not None,
            'constraint_id': None if constraint is None else int(constraint),
            'child_matches_active_bead': bool(child_matches),
            'retained': retained}


def acquire_active_endpoint(env, task, speed, z_offset):
    active_id = int(task.cable_bead_IDs[task._layout['active_endpoint_index']])
    bead_position = np.asarray(
        p.getBasePositionAndOrientation(active_id)[0], dtype=np.float64)
    orientation = np.asarray(env.home_pose[3:], dtype=np.float64)
    approach = bead_position.copy()
    approach[2] = max(0.08, bead_position[2] + 0.06)
    contact = bead_position.copy()
    contact[2] += float(z_offset)
    task.set_ee_target_position(approach)
    approach_ok = env.movep(approach.tolist() + orientation.tolist(), speed=speed)
    task.set_ee_target_position(contact)
    contact_ok = env.movep(contact.tolist() + orientation.tolist(), speed=speed)
    env.step_physics(4)
    env.ee.activate([active_id], [])
    env.step_physics(4)
    status = grasp_status(env, active_id)
    if not (approach_ok and contact_ok and status['retained']):
        raise RuntimeError('active-endpoint grasp failed: {}'.format(status))
    ee_state = p.getLinkState(env.ur5, env.ee_tip_link,
                              computeForwardKinematics=True)
    return {'active_bead_id': active_id,
            'approach_target': approach.astype(float).tolist(),
            'contact_target': contact.astype(float).tolist(),
            'approach_succeeded': bool(approach_ok),
            'contact_succeeded': bool(contact_ok), 'grasp': status,
            'ee_position': [float(value) for value in ee_state[0]],
            'ee_orientation': [float(value) for value in ee_state[1]]}


def _layout_payload(layout):
    result = {}
    for key, value in layout.items():
        result[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def _oracle_summary(arrays):
    contact = np.asarray(arrays['oracle_pin_contact_count']) > 0
    force = np.asarray(arrays['oracle_pin_contact_force']) > 0
    indices = np.flatnonzero(contact | force)
    return {
        'contact_samples': int(np.count_nonzero(contact | force)),
        'first_contact_step': (None if not len(indices) else int(
            arrays['physics_step'][indices[0]])),
        'minimum_signed_distance': float(np.min(
            arrays['oracle_pin_min_signed_distance'])),
        'peak_contact_force': float(np.max(arrays['oracle_pin_contact_force']))}


def run_branch(*, env, task, condition, state_id, base_world_state,
               python_runtime_state, command_script, config, pair_dir,
               active_bead_id, stop_after_probe):
    branch_dir = pair_dir / condition
    branch_dir.mkdir(parents=True, exist_ok=True)
    env.pause()
    with env._ccda_step_lock:
        p.restoreState(stateId=state_id)
        env.reset_ccda_runtime_after_restore()
        restore_python_runtime_state(env, task, python_runtime_state)
        task.reset_branch_runtime(condition)
        arm = task.arm_condition(condition)
        initial_state = capture_world_state(env, task)
    recorder = CCDAVideoRecorder(
        camera_config(config), branch_dir / 'video.mp4',
        fps=float(config['camera']['fps']),
        stride=int(config['camera']['frame_stride']))
    env.set_ccda_video_recorder(recorder)
    records, failures = [], []
    boundaries = {}
    try:
        task.set_ccda_phase('no_action')
        boundaries['no_action_start'] = task.physics_step_count()
        env.step_physics(int(config['execution']['no_action_steps']))
        boundaries['no_action_end'] = task.physics_step_count()
        probe = execute_fixed_command_phase(
            env, task, command_script['phases'][0],
            position_gains=config['motion']['position_gain'])
        records.append(probe)
        task.set_ccda_phase('post_probe')
        boundaries['probe_end'] = task.physics_step_count()
        env.step_physics(int(config['execution']['post_probe_steps']))
        boundaries['post_probe_end'] = task.physics_step_count()
        if not stop_after_probe:
            test = execute_fixed_command_phase(
                env, task, command_script['phases'][1],
                position_gains=config['motion']['position_gain'])
            records.append(test)
            task.set_ccda_phase('post_test')
            boundaries['test_end'] = task.physics_step_count()
            env.step_physics(int(config['execution']['post_test_steps']))
            boundaries['post_test_end'] = task.physics_step_count()
        for record in records:
            if record['actual_physics_steps'] != record['expected_physics_steps']:
                failures.append(record['phase'] + '_physics_step_mismatch')
            if record['endpoint_error'] > config['motion']['max_endpoint_error_m']:
                failures.append(record['phase'] + '_endpoint_error')
            if not record['grasp_retained']:
                failures.append(record['phase'] + '_grasp_lost')
    finally:
        env.set_ccda_video_recorder(None)
        video = recorder.close()
    trace = task.ccda_trace()
    if not trace:
        raise RuntimeError('{} produced an empty trace'.format(condition))
    arrays = trace_to_arrays(trace)
    np.savez_compressed(branch_dir / 'trajectory.npz', **arrays)
    write_json(branch_dir / 'trace.json', trace)
    status = grasp_status(env, active_bead_id)
    if not status['retained']:
        failures.append('final_grasp_lost')
    metadata = {
        'condition': condition, 'arm': arm,
        'initial_full_state_max_abs': max_state_difference(
            base_world_state, initial_state),
        'phase_boundaries': boundaries, 'command_records': records,
        'command_count': int(sum(r['command_count'] for r in records)),
        'command_physics_steps': int(sum(
            r['actual_physics_steps'] for r in records)),
        'execution_failures': failures, 'grasp': status,
        'physics_steps': task.physics_step_count(),
        'trace_samples': len(trace), 'oracle': _oracle_summary(arrays),
        'privileged_state': task.ccda_privileged_state(), 'video': video}
    write_json(branch_dir / 'metadata.json', metadata)
    return metadata, initial_state, arrays


def _geometry_from_config(config):
    payload = dict(config['geometry'])
    payload.update({
        'probe_spacing_scale': config['motion']['probe_spacing_scale'],
        'test_spacing_scale': config['motion']['test_spacing_scale'],
        'test_angle_deg': config['motion']['test_angle_deg']})
    return OCCPGeometryConfig(**payload)


def run_pair(config, stop_after_probe=False):
    if config_schema(config) != 'r1':
        raise RuntimeError(
            'legacy Phase 0A config is preserved evidence and is not interpreted '
            'with R1 diameter-based geometry')
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    pair_id = config['pair_id'] + ('_probe_only' if stop_after_probe else '')
    pair_dir = Path(config['output_root']) / ('pair_' + pair_id)
    pair_dir.mkdir(parents=True, exist_ok=True)
    execution = config['execution']
    env = Environment(
        disp=False, hz=int(execution['hz']),
        deterministic=bool(execution['deterministic']),
        control_substeps=int(execution['control_substeps']),
        post_action_settle_steps=int(execution['post_action_settle_steps']))
    state_id = None
    try:
        geometry = _geometry_from_config(config)
        layout = compute_occp_layout(geometry, 'free')
        task = tasks.names[config['task_name']]()
        task.configure_audit(
            pair_id=pair_id, seed=seed, trace_stride=config['trace']['stride'],
            settle_seconds=execution['initial_settle_seconds'],
            geometry_config=geometry)
        env.reset(task)
        acquisition = acquire_active_endpoint(
            env, task, speed=config['motion']['acquisition_speed'],
            z_offset=config['motion']['grasp_height_offset'])
        task.release_active_endpoint_stabilizer()
        env.step_physics(int(execution['pre_snapshot_settle_steps']))
        post_release_grasp = grasp_status(env, acquisition['active_bead_id'])
        if not post_release_grasp['retained']:
            raise RuntimeError('grasp lost after active stabilizer release')
        task.reset_branch_runtime('free')
        env.pause()
        with env._ccda_step_lock:
            base_world_state = capture_world_state(env, task)
            python_runtime_state = capture_python_runtime_state(env, task)
            state_id = int(p.saveState())
        np.savez_compressed(pair_dir / 'base_state.npz', **base_world_state)
        script = build_fixed_command_script(
            env, start_ee_position=acquisition['ee_position'],
            orientation=acquisition['ee_orientation'], layout=layout,
            probe_command_steps=config['motion']['probe_command_steps'],
            test_command_steps=config['motion']['test_command_steps'],
            control_substeps=execution['control_substeps'])
        jam_script = copy_fixed_command_script(script)
        free_arrays = command_arrays(script)
        jam_arrays = command_arrays(jam_script)
        arrays_equal = all(np.array_equal(free_arrays[k], jam_arrays[k])
                           for k in free_arrays)
        if not arrays_equal:
            raise RuntimeError('paired fixed command arrays differ')
        write_json(pair_dir / 'fixed_command_script.json', script)
        np.savez_compressed(pair_dir / 'fixed_command_arrays.npz', **free_arrays)
        branch_metadata, initial_states, traces = {}, {}, {}
        for condition, commands in zip(CONDITIONS, (script, jam_script)):
            branch_metadata[condition], initial_states[condition], traces[condition] = run_branch(
                env=env, task=task, condition=condition, state_id=state_id,
                base_world_state=base_world_state,
                python_runtime_state=python_runtime_state,
                command_script=commands, config=config, pair_dir=pair_dir,
                active_bead_id=acquisition['active_bead_id'],
                stop_after_probe=stop_after_probe)
        free_trace, jam_trace = traces[CONDITIONS[0]], traces[CONDITIONS[1]]
        step_equal = np.array_equal(
            free_trace['physics_step'], jam_trace['physics_step'])
        phase_equal = np.array_equal(free_trace['phase'], jam_trace['phase'])
        visible = layout['visible_readout_indices']
        metadata = {
            'pair_id': pair_id, 'seed': seed,
            'dataset_role': config['dataset_role'],
            'conditions': list(CONDITIONS), 'stop_after_probe': stop_after_probe,
            'pairing_mode': 'pybullet_save_restore_fixed_low_level_commands',
            'active_stabilizer_released': task.active_endpoint_stabilizer_id is None,
            'pre_snapshot_settle_steps': execution['pre_snapshot_settle_steps'],
            'grasp_after_stabilizer_release': post_release_grasp,
            'fixed_command_arrays_equal': arrays_equal,
            'free_physics_steps': branch_metadata['free']['physics_steps'],
            'jam_physics_steps': branch_metadata['right_hidden_jam']['physics_steps'],
            'physics_step_arrays_equal': bool(step_equal),
            'phase_arrays_equal': bool(phase_equal),
            'probe_command_steps_equal': bool(np.array_equal(
                free_arrays['probe_joint_targets'], jam_arrays['probe_joint_targets'])),
            'test_command_steps_equal': bool(np.array_equal(
                free_arrays['test_joint_targets'], jam_arrays['test_joint_targets'])),
            'base_to_free_max_abs': max_state_difference(
                base_world_state, initial_states['free']),
            'base_to_jam_max_abs': max_state_difference(
                base_world_state, initial_states['right_hidden_jam']),
            'free_to_jam_max_abs': max_state_difference(
                initial_states['free'], initial_states['right_hidden_jam']),
            'initial_visible_bead_rmse': float(np.sqrt(np.mean(np.square(
                initial_states['free']['bead_positions'][visible]
                - initial_states['right_hidden_jam']['bead_positions'][visible])))),
            'initial_robot_state_rmse': float(np.sqrt(np.mean(np.square(
                np.concatenate([initial_states['free']['joint_positions'],
                                initial_states['free']['ee_position']])
                - np.concatenate([initial_states['right_hidden_jam']['joint_positions'],
                                  initial_states['right_hidden_jam']['ee_position']]))))),
            'acquisition': acquisition, 'layout': _layout_payload(layout),
            'execution': execution, 'motion': config['motion'],
            'geometry': config['geometry'], 'analysis': config['analysis'],
            'trace': config['trace'], 'branches': branch_metadata}
        write_json(pair_dir / 'metadata.json', metadata)
        return pair_dir, metadata
    finally:
        if state_id is not None and p.isConnected():
            try:
                p.removeState(stateUniqueId=state_id)
            except Exception:
                pass
        env.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stop-after-probe', action='store_true')
    args = parser.parse_args()
    pair_dir, metadata = run_pair(
        load_config(args.config), stop_after_probe=args.stop_after_probe)
    print('pair_dir={}'.format(pair_dir))
    print('fixed_command_arrays_equal={}'.format(
        metadata['fixed_command_arrays_equal']))


if __name__ == '__main__':
    main()
