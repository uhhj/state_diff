"""Run one exact free/right-hidden-jam OCCP counterfactual pair."""
from __future__ import annotations

import argparse
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

from scripts.experiment3.phase0.occp_common import (  # noqa: E402
    CONDITIONS,
    build_action_script,
    copy_action_script,
    load_config,
    trace_to_arrays,
    write_json,
)
from scripts.experiment3.phase0.occp_snapshot import (  # noqa: E402
    capture_python_runtime_state,
    capture_world_state,
    max_state_difference,
    restore_python_runtime_state,
)


def camera_config(config):
    image_size = tuple(int(value) for value in config['camera']['image_size'])
    return {
        'image_size': image_size,
        'intrinsics': (450., 0., image_size[1] / 2,
                       0., 450., image_size[0] / 2,
                       0., 0., 1.),
        'position': np.asarray([0.50, 0.0, 0.75], dtype=np.float64),
        'rotation': p.getQuaternionFromEuler([0., np.pi, -np.pi / 2]),
        'zrange': (0.01, 3.0),
        'noise': False,
    }


def grasp_status(env, active_bead_id):
    constraint = getattr(env.ee, 'contact_constraint', None)
    child_matches = False
    if constraint is not None:
        try:
            child_matches = int(p.getConstraintInfo(int(constraint))[2]) == int(
                active_bead_id)
        except Exception:
            child_matches = False
    return {
        'activated': bool(getattr(env.ee, 'activated', False)),
        'constraint_available': constraint is not None,
        'constraint_id': None if constraint is None else int(constraint),
        'child_matches_active_bead': bool(child_matches),
        'retained': bool(
            getattr(env.ee, 'activated', False)
            and constraint is not None and child_matches),
    }


def acquire_active_endpoint(env, task, speed, z_offset, settle_seconds):
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
        raise RuntimeError(
            'active-endpoint grasp failed: approach_ok={}, contact_ok={}, status={}'
            .format(approach_ok, contact_ok, status))
    env.settle_for_seconds(float(settle_seconds))
    ee_state = p.getLinkState(
        env.ur5, env.ee_tip_link, computeForwardKinematics=True)
    return {
        'active_bead_id': active_id,
        'approach_target': approach.astype(float).tolist(),
        'contact_target': contact.astype(float).tolist(),
        'approach_succeeded': bool(approach_ok),
        'contact_succeeded': bool(contact_ok),
        'grasp': status,
        'ee_position_after_settle': [float(value) for value in ee_state[0]],
        'ee_orientation_after_settle': [float(value) for value in ee_state[1]],
    }


def _layout_payload(layout):
    payload = {}
    for key, value in layout.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.astype(float).tolist()
        else:
            payload[key] = value
    return payload


def _execute_action(env, task, action, joint_tolerance):
    start_step = task.physics_step_count()
    target_position = np.asarray(action['target_position'], dtype=np.float64)
    target_orientation = np.asarray(
        action['target_orientation'], dtype=np.float64)
    task.set_ccda_phase(action['phase'])
    task.set_ee_target_position(target_position)
    succeeded = env.movep(
        target_position.tolist() + target_orientation.tolist(),
        speed=float(action['speed']),
        joint_tolerance=float(joint_tolerance))
    ee_state = p.getLinkState(
        env.ur5, env.ee_tip_link, computeForwardKinematics=True)
    actual = np.asarray(ee_state[0], dtype=np.float64)
    return {
        'name': action['name'],
        'phase': action['phase'],
        'target_position': target_position.astype(float).tolist(),
        'actual_position': actual.astype(float).tolist(),
        'endpoint_error': float(np.linalg.norm(actual - target_position)),
        'succeeded': bool(succeeded),
        'start_physics_step': int(start_step),
        'end_physics_step': int(task.physics_step_count()),
    }


def run_branch(
        *, env, task, condition, state_id, base_world_state,
        python_runtime_state, action_script, config, pair_dir,
        active_bead_id):
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
        camera_config(config),
        branch_dir / 'video.mp4',
        fps=float(config['camera']['fps']),
        stride=int(config['camera']['frame_stride']))
    env.set_ccda_video_recorder(recorder)
    action_records = []
    execution_failure = None
    try:
        task.set_ccda_phase('no_action')
        env.step_physics(int(config['execution']['no_action_steps']))

        probe_record = _execute_action(
            env, task, action_script[0], config['motion']['joint_tolerance'])
        action_records.append(probe_record)
        if not probe_record['succeeded']:
            execution_failure = 'probe_movep_failed'
        else:
            env.step_physics(int(config['execution']['post_probe_steps']))
            test_record = _execute_action(
                env, task, action_script[1], config['motion']['joint_tolerance'])
            action_records.append(test_record)
            if not test_record['succeeded']:
                execution_failure = 'test_pull_movep_failed'
            else:
                task.set_ccda_phase('post_test')
                env.step_physics(int(config['execution']['post_test_steps']))
    finally:
        env.set_ccda_video_recorder(None)
        video = recorder.close()

    trace = task.ccda_trace()
    if not trace:
        raise RuntimeError('{} branch produced an empty trace'.format(condition))
    arrays = trace_to_arrays(trace)
    np.savez_compressed(branch_dir / 'trajectory.npz', **arrays)
    write_json(branch_dir / 'trace.json', trace)
    metadata = {
        'condition': condition,
        'arm': arm,
        'initial_full_state_max_abs': max_state_difference(
            base_world_state, initial_state),
        'action_script': action_script,
        'action_records': action_records,
        'execution_failure': execution_failure,
        'grasp': grasp_status(env, active_bead_id),
        'physics_steps': task.physics_step_count(),
        'trace_samples': len(trace),
        'privileged_state': task.ccda_privileged_state(),
        'video': video,
    }
    write_json(branch_dir / 'metadata.json', metadata)
    return metadata, initial_state


def run_pair(config):
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    pair_dir = Path(config['output_root']) / ('pair_' + config['pair_id'])
    pair_dir.mkdir(parents=True, exist_ok=True)

    execution = config['execution']
    env = Environment(
        disp=False,
        hz=int(execution['hz']),
        deterministic=bool(execution['deterministic']),
        control_substeps=int(execution['control_substeps']),
        post_action_settle_steps=int(execution['post_action_settle_steps']))
    state_id = None
    try:
        task = tasks.names[config['task_name']]()
        geometry_overrides = config.get('geometry', {})
        geometry = OCCPGeometryConfig(
            probe_spacing_scale=float(config['motion']['probe_spacing_scale']),
            test_spacing_scale=float(config['motion']['test_spacing_scale']),
            test_angle_deg=float(config['motion']['test_angle_deg']),
            jam_clearance_scale=float(
                geometry_overrides.get('jam_clearance_scale', 0.10)),
            pin_from_active_exit_spacing=float(
                geometry_overrides.get('pin_from_active_exit_spacing', 5.0)))
        task.configure_audit(
            pair_id=config['pair_id'],
            seed=seed,
            trace_stride=int(config['trace']['stride']),
            settle_seconds=float(execution['initial_settle_seconds']),
            geometry_config=geometry)
        env.reset(task)
        acquisition = acquire_active_endpoint(
            env, task,
            speed=float(config['motion']['speed']),
            z_offset=float(config['motion']['grasp_height_offset']),
            settle_seconds=float(execution['initial_settle_seconds']))
        task.reset_branch_runtime('free')
        env.pause()
        with env._ccda_step_lock:
            base_world_state = capture_world_state(env, task)
            python_runtime_state = capture_python_runtime_state(env, task)
            state_id = int(p.saveState())
        np.savez_compressed(pair_dir / 'base_state.npz', **base_world_state)

        active_position = np.asarray(
            p.getBasePositionAndOrientation(acquisition['active_bead_id'])[0])
        action_script = build_action_script(
            active_position,
            acquisition['ee_orientation_after_settle'],
            compute_occp_layout(geometry, 'free'),
            config['motion']['speed'])
        jam_action_script = copy_action_script(action_script)
        if action_script != jam_action_script:
            raise RuntimeError('paired action payloads differ')
        write_json(pair_dir / 'action_script.json', action_script)

        branch_metadata = {}
        initial_states = {}
        for condition, actions in zip(
                CONDITIONS, (action_script, jam_action_script)):
            branch_metadata[condition], initial_states[condition] = run_branch(
                env=env,
                task=task,
                condition=condition,
                state_id=state_id,
                base_world_state=base_world_state,
                python_runtime_state=python_runtime_state,
                action_script=actions,
                config=config,
                pair_dir=pair_dir,
                active_bead_id=acquisition['active_bead_id'])

        pair_metadata = {
            'pair_id': config['pair_id'],
            'seed': seed,
            'dataset_role': config['dataset_role'],
            'conditions': list(CONDITIONS),
            'pairing_mode': 'pybullet_save_restore_fixed_step',
            'action_payload_equal': action_script == jam_action_script,
            'base_to_free_max_abs': max_state_difference(
                base_world_state, initial_states['free']),
            'base_to_jam_max_abs': max_state_difference(
                base_world_state, initial_states['right_hidden_jam']),
            'free_to_jam_max_abs': max_state_difference(
                initial_states['free'], initial_states['right_hidden_jam']),
            'initial_visible_bead_rmse': float(np.sqrt(np.mean(np.square(
                initial_states['free']['bead_positions'][
                    compute_occp_layout(geometry, 'free')[
                        'visible_readout_indices']]
                - initial_states['right_hidden_jam']['bead_positions'][
                    compute_occp_layout(geometry, 'free')[
                        'visible_readout_indices']])))),
            'initial_robot_state_rmse': float(np.sqrt(np.mean(np.square(
                np.concatenate([
                    initial_states['free']['joint_positions'],
                    initial_states['free']['ee_position'],
                    initial_states['free']['ee_orientation']])
                - np.concatenate([
                    initial_states['right_hidden_jam']['joint_positions'],
                    initial_states['right_hidden_jam']['ee_position'],
                    initial_states['right_hidden_jam']['ee_orientation']]))))),
            'initial_velocity_rmse': float(np.sqrt(np.mean(np.square(
                np.concatenate([
                    initial_states['free']['bead_linear_velocities'].reshape(-1),
                    initial_states['free']['bead_angular_velocities'].reshape(-1),
                    initial_states['free']['joint_velocities'],
                    initial_states['free']['ee_linear_velocity'],
                    initial_states['free']['ee_angular_velocity']])
                - np.concatenate([
                    initial_states['right_hidden_jam'][
                        'bead_linear_velocities'].reshape(-1),
                    initial_states['right_hidden_jam'][
                        'bead_angular_velocities'].reshape(-1),
                    initial_states['right_hidden_jam']['joint_velocities'],
                    initial_states['right_hidden_jam']['ee_linear_velocity'],
                    initial_states['right_hidden_jam']['ee_angular_velocity']]))))),
            'acquisition': acquisition,
            'layout': _layout_payload(compute_occp_layout(geometry, 'free')),
            'execution': execution,
            'motion': config['motion'],
            'trace': config['trace'],
            'geometry': geometry_overrides,
            'smoke_debug_history': config.get('smoke_debug_history', []),
            'branches': branch_metadata,
        }
        write_json(pair_dir / 'metadata.json', pair_metadata)
        return pair_dir, pair_metadata
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
    args = parser.parse_args()
    pair_dir, metadata = run_pair(load_config(args.config))
    print('pair_dir={}'.format(pair_dir))
    print('action_payload_equal={}'.format(metadata['action_payload_equal']))


if __name__ == '__main__':
    main()
