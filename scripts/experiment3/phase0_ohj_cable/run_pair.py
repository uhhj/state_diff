"""Run the exact FREE/JAM-R/repeat OHJ pair."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for root in (REPO_ROOT, SIM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import pybullet as p  # noqa: E402
from ravens import Environment, tasks  # noqa: E402
from ravens.tasks.ccda_ohj_geometry import OHJGeometryConfig  # noqa: E402

from scripts.experiment3.phase0_ohj_cable.commands import (  # noqa: E402
    build_candidate_scripts, build_probe_script, command_arrays,
    execute_fixed_phase)
from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    load_config, pair_dir, trace_to_arrays, write_json)
from scripts.experiment3.phase0_ohj_cable.snapshot import (  # noqa: E402
    capture_python_runtime_state, capture_world_state,
    restore_python_runtime_state)


def geometry_config(config):
    fields = dict(config["geometry"])
    fields.pop("visible_keypoint_indices")
    return OHJGeometryConfig(**fields)


def grasp_retained(env, active_id):
    constraint = getattr(env.ee, "contact_constraint", None)
    if constraint is None or not getattr(env.ee, "activated", False):
        return False
    try:
        return int(p.getConstraintInfo(int(constraint))[2]) == int(active_id)
    except Exception:
        return False


def acquire_active_endpoint(env, task, motion):
    active_id = task.cable_bead_IDs[task._layout["active_endpoint_index"]]
    bead = np.asarray(p.getBasePositionAndOrientation(active_id)[0])
    orientation = np.asarray(env.home_pose[3:])
    approach = bead + np.array([0.0, 0.0, 0.06])
    contact = bead + np.array([0.0, 0.0,
                               float(motion["grasp_height_offset_m"])])
    task.set_ee_target_position(approach)
    if not env.movep(approach.tolist() + orientation.tolist(),
                     speed=motion["acquisition_speed"]):
        raise RuntimeError("OHJ endpoint approach failed")
    task.set_ee_target_position(contact)
    if not env.movep(contact.tolist() + orientation.tolist(),
                     speed=motion["acquisition_speed"]):
        raise RuntimeError("OHJ endpoint contact failed")
    env.step_physics(4)
    env.ee.activate([active_id], [])
    env.step_physics(4)
    if not grasp_retained(env, active_id):
        raise RuntimeError("OHJ endpoint grasp failed")
    task.release_active_endpoint_stabilizer()
    return active_id


def _save_snapshot(path, world):
    np.savez_compressed(path, **world)


def _run_branch(env, task, condition, actual_condition, state_id,
                runtime_state, reference_passive, probe, straight,
                config, active_id):
    p.restoreState(stateId=state_id)
    env.reset_ccda_runtime_after_restore()
    restore_python_runtime_state(env, task, runtime_state)
    task.reset_branch_runtime(actual_condition)
    arm = task.arm_condition(actual_condition)
    task.reference_passive_xyz = np.asarray(reference_passive).copy()
    initial_state = task.statediff_state().copy()
    task.set_ccda_phase("no_action")
    env.step_physics(config["execution"]["no_action_steps"])
    for phase in probe:
        execute_fixed_phase(
            env, task, phase, config["motion"]["position_gain"])
    task.set_ccda_phase("post_probe")
    env.step_physics(config["execution"]["post_probe_steps"])
    post_probe_state = task.statediff_state().copy()
    post_probe_world = capture_world_state(env, task)
    for phase in straight:
        execute_fixed_phase(
            env, task, phase, config["motion"]["position_gain"])
    task.set_ccda_phase("post_test")
    env.step_physics(config["execution"]["post_test_steps"])
    arrays = trace_to_arrays(task.ccda_trace())
    commands = command_arrays(tuple(probe) + tuple(straight))
    arrays.update({"command_" + key: value for key, value in commands.items()})
    output = pair_dir(config) / condition
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectory.npz", **arrays)
    _save_snapshot(output / "post_probe_snapshot.npz", post_probe_world)
    metadata = {
        "condition": condition,
        "actual_condition": actual_condition,
        "arm": arm,
        "initial_statediff_state": initial_state,
        "post_probe_statediff_state": post_probe_state,
        "grasp_retained": grasp_retained(env, active_id),
        "reference_passive_xyz": reference_passive,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def run_pair(config_path):
    config = load_config(config_path)
    env = Environment(
        disp=False, hz=config["execution"]["hz"],
        deterministic=True,
        control_substeps=config["execution"]["control_substeps"])
    try:
        task = tasks.names[config["task_name"]]()
        task.configure_phase0(
            pair_id=config["pair_id"], seed=config["seed"],
            settle_seconds=config["execution"]["initial_settle_seconds"],
            geometry_config=geometry_config(config))
        env.reset(task)
        active_id = acquire_active_endpoint(env, task, config["motion"])
        env.step_physics(config["execution"]["pre_snapshot_settle_steps"])
        reference_passive = task._bead_positions()[:4].mean(axis=0)
        task.reference_passive_xyz = reference_passive.copy()
        common_world = capture_world_state(env, task)
        runtime_state = capture_python_runtime_state(env, task)
        state_id = p.saveState()
        ee_state = p.getLinkState(
            env.ur5, env.ee_tip_link, computeForwardKinematics=True)
        start = np.asarray(ee_state[0], dtype=np.float64)
        orientation = np.asarray(ee_state[1], dtype=np.float64)
        previous = np.asarray([
            p.getJointState(env.ur5, int(joint))[0] for joint in env.joints])
        probe, _ = build_probe_script(
            env, start, orientation, config["motion"], previous)
        candidates = build_candidate_scripts(
            env, start, orientation, config["motion"], previous)
        branches = {}
        for name, actual in (("free", "free"), ("jam_right", "jam_right"),
                             ("free_repeat", "free")):
            branches[name] = _run_branch(
                env, task, name, actual, state_id, runtime_state,
                reference_passive, probe, candidates["straight"],
                config, active_id)
        commands = command_arrays(tuple(probe) + candidates["straight"])
        write_json(pair_dir(config) / "metadata.json", {
            "config": config,
            "common_world_state": common_world,
            "branches": branches,
            "command_shapes": {key: list(value.shape)
                               for key, value in commands.items()},
            "branch_local_ik": False,
        })
        return {"pair_dir": str(pair_dir(config)), "branches": list(branches)}
    finally:
        env.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_pair(args.config))


if __name__ == "__main__":
    main()
