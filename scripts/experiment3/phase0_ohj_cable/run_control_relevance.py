"""Run the fixed OHJ action library from branch-specific post-probe states."""
from __future__ import annotations

import argparse
import json
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

from scripts.experiment3.phase0_ohj_cable.commands import (  # noqa: E402
    build_candidate_scripts, command_arrays, execute_fixed_phase)
from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    load_config, pair_dir, report_dir, write_json)
from scripts.experiment3.phase0_ohj_cable.run_pair import (  # noqa: E402
    acquire_active_endpoint, geometry_config, grasp_retained)


def _load_snapshot(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _restore_snapshot(env, task, snapshot, condition, reference):
    for index, body in enumerate(task.cable_bead_IDs):
        p.resetBasePositionAndOrientation(
            body, snapshot["bead_positions"][index],
            snapshot["bead_orientations"][index])
        p.resetBaseVelocity(
            body, snapshot["bead_linear_velocities"][index],
            snapshot["bead_angular_velocities"][index])
    for index, joint in enumerate(env.joints):
        p.resetJointState(
            env.ur5, int(joint), snapshot["joint_positions"][index],
            snapshot["joint_velocities"][index])
    env.reset_ccda_runtime_after_restore()
    task.reset_branch_runtime(condition)
    task.arm_condition(condition)
    task.reference_passive_xyz = np.asarray(reference)


def collect_control_matrix(config_path, *, require_pair_complete=True):
    config = load_config(config_path)
    pair = json.loads((report_dir(config) / "pair_metrics.json").read_text(
        encoding="utf-8"))
    if (require_pair_complete
            and pair["verdict"] != "PHASE0D_PAIR_COMPLETE"):
        raise RuntimeError("control relevance requires PHASE0D_PAIR_COMPLETE")
    env = Environment(
        disp=False, hz=config["execution"]["hz"], deterministic=True,
        control_substeps=config["execution"]["control_substeps"])
    try:
        task = tasks.names[config["task_name"]]()
        task.configure_phase0(
            pair_id=config["pair_id"], seed=config["seed"],
            settle_seconds=config["execution"]["initial_settle_seconds"],
            geometry_config=geometry_config(config))
        env.reset(task)
        active_id = acquire_active_endpoint(env, task, config["motion"])
        root = pair_dir(config)
        snapshots = {condition: _load_snapshot(
            root / condition / "post_probe_snapshot.npz")
            for condition in ("free", "jam_right")}
        metadata = {condition: json.loads(
            (root / condition / "metadata.json").read_text(encoding="utf-8"))
            for condition in snapshots}
        free = snapshots["free"]
        scripts = build_candidate_scripts(
            env, free["ee_position"], free["ee_orientation"],
            config["motion"], free["joint_positions"])
        matrix = {}
        command_payload = {}
        for candidate, phases in scripts.items():
            command_payload[candidate] = command_arrays(phases)
        for condition in ("free", "jam_right"):
            matrix[condition] = {}
            reference = metadata[condition]["reference_passive_xyz"]
            for candidate, phases in scripts.items():
                _restore_snapshot(
                    env, task, snapshots[condition], condition, reference)
                for phase in phases:
                    execute_fixed_phase(
                        env, task, phase, config["motion"]["position_gain"])
                task.set_ccda_phase("post_test")
                env.step_physics(config["execution"]["post_test_steps"])
                trace = task.ccda_trace()
                wrench = np.asarray([row["formal_wrench"] for row in trace])
                commands = command_payload[candidate]
                targets = commands["ee_target"]
                path_length = float(np.sum(np.linalg.norm(
                    np.diff(np.vstack([free["ee_position"], targets]), axis=0),
                    axis=1)))
                matrix[condition][candidate] = {
                    "final_progress_m": float(trace[-1]["extraction_progress_m"]),
                    "peak_force_n": float(np.max(np.linalg.norm(
                        wrench[:, :3], axis=1))),
                    "grasp_retained": grasp_retained(env, active_id),
                    "path_length_m": path_length,
                    "command_phase": commands["phase"],
                    "command_ee_target": commands["ee_target"],
                    "command_joint_target": commands["joint_target"],
                }
        return {
            "matrix": matrix,
            "branch_local_ik": False,
            "source_pair_verdict": pair["verdict"],
        }
    finally:
        env.stop()


def run_control_relevance(config_path):
    payload = collect_control_matrix(
        config_path, require_pair_complete=True)
    config = load_config(config_path)
    output = pair_dir(config) / "control_relevance.json"
    write_json(output, {
        "matrix": payload["matrix"],
        "branch_local_ik": payload["branch_local_ik"],
    })
    return {"output": str(output), "matrix": payload["matrix"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("output={}".format(run_control_relevance(args.config)["output"]))


if __name__ == "__main__":
    main()
