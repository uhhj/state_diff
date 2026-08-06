"""Run one strict Soft BlockPush low/high-friction counterfactual pair."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from state_diff.env.block_pushing.soft_block_pushing import (  # noqa: E402
    LOWDIM_LAYOUT, SoftBlockPushEnv)
from scripts.experiment3.phase0_soft_blockpush.common import (  # noqa: E402
    CONDITIONS, load_config, write_json)
from scripts.experiment3.phase0_soft_blockpush.fixed_commands import (  # noqa: E402
    build_fixed_command_script, command_arrays, copy_fixed_command_script,
    execute_fixed_phase)
from scripts.experiment3.phase0_soft_blockpush.snapshot import (  # noqa: E402
    capture_explicit_state, max_state_difference, save_explicit_state)


NPZ_FIELDS = (
    "physics_step", "phase", "node_positions", "node_velocities",
    "visible_keypoints", "node_orientations", "joint_positions",
    "joint_velocities", "joint_motor_torque", "joint_reaction_wrench",
    "ee_position", "ee_velocity", "ee_target_position",
    "ee_tracking_error_xyz", "ee_contact_wrench", "goal_xy",
    "oracle_patch_normal_force", "oracle_patch_tangential_force",
    "oracle_patch_mean_slip_speed", "oracle_patch_stick_ratio",
    "oracle_patch_contact_count")


def trace_arrays(trace):
    """Convert fixed-shape trace fields to non-object NumPy arrays."""
    return {field: np.asarray([row[field] for row in trace])
            for field in NPZ_FIELDS}


class BranchVideo:
    """Small stride-controlled MP4 writer for one deterministic branch."""

    def __init__(self, env, path: Path, config: Dict[str, Any]):
        self.env = env
        self.path = path
        self.stride = int(config["camera"]["frame_stride"])
        height, width = config["camera"]["image_size"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"),
            float(config["camera"]["fps"]), (int(width), int(height)))
        if not self.writer.isOpened():
            raise RuntimeError("unable to open MP4 writer")
        self.frames = 0

    def maybe_record(self) -> None:
        """Record the current frame at the configured physics stride."""
        if self.env._physics_step % self.stride == 0:
            rgb = self.env.render()
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.frames += 1

    def close(self) -> dict:
        """Close the writer and return metadata."""
        self.writer.release()
        return {"output_path": str(self.path), "frames_written": self.frames,
                "video_available": self.path.exists()}


def _hold(env, steps: int, phase: str, joint_target: np.ndarray,
          ee_target: np.ndarray, video: BranchVideo) -> None:
    env.set_phase(phase)
    env.set_ee_target_position(ee_target)
    for _ in range(int(steps)):
        env.set_fixed_joint_target(joint_target)
        env.step_one_physics()
        video.maybe_record()


def _execute_with_video(env, phase: dict, video: BranchVideo) -> dict:
    start = env._physics_step
    actual = []
    for ee_target, joint_target in zip(phase["ee_targets"], phase["joint_targets"]):
        env.set_phase(phase["phase"])
        env.set_ee_target_position(np.asarray(ee_target))
        env.set_fixed_joint_target(np.asarray(joint_target))
        env.step_one_physics()
        video.maybe_record()
        actual.append(env._observation()["ee_position"].tolist())
    target = np.asarray(phase["ee_targets"][-1])
    endpoint = np.asarray(actual[-1])
    return {"phase": phase["phase"], "command_count": len(actual),
            "physics_step_start": int(start),
            "physics_step_end": int(env._physics_step),
            "expected_physics_steps": len(actual),
            "actual_physics_steps": int(env._physics_step - start),
            "final_target_position": target.tolist(),
            "final_actual_position": endpoint.tolist(),
            "endpoint_error_m": float(np.linalg.norm(endpoint - target)),
            "actual_ee_trajectory": actual}


def run_branch(env, condition: str, script: dict, config: Dict[str, Any],
               pair_dir: Path, base_state: dict,
               stop_after_probe: bool):
    """Restore, arm, and execute one exact branch."""
    env.restore_saved_state(env.saved_state_id)
    env.clear_trace()
    arm = env.arm_condition(condition)
    initial = capture_explicit_state(env)
    branch_dir = pair_dir / condition
    branch_dir.mkdir(parents=True, exist_ok=True)
    video = BranchVideo(env, branch_dir / "video.mp4", config)
    records = []
    boundaries = {"no_action_start": 0}
    hold = np.asarray(script["hold_joint_target"], dtype=np.float64)
    start_ee = np.asarray(base_state["ee_target_position"], dtype=np.float64)
    _hold(env, config["execution"]["no_action_steps"], "no_action",
          hold, start_ee, video)
    boundaries["no_action_end"] = int(env._physics_step)
    probe_record = _execute_with_video(env, script["phases"][0], video)
    records.append(probe_record)
    boundaries["probe_end"] = int(env._physics_step)
    probe_hold = np.asarray(script["phases"][0]["joint_targets"][-1])
    probe_ee = np.asarray(script["phases"][0]["ee_targets"][-1])
    _hold(env, config["execution"]["post_probe_steps"], "post_probe",
          probe_hold, probe_ee, video)
    boundaries["post_probe_end"] = int(env._physics_step)
    if not stop_after_probe:
        test_record = _execute_with_video(env, script["phases"][1], video)
        records.append(test_record)
        boundaries["test_end"] = int(env._physics_step)
        test_hold = np.asarray(script["phases"][1]["joint_targets"][-1])
        test_ee = np.asarray(script["phases"][1]["ee_targets"][-1])
        _hold(env, config["execution"]["post_test_steps"], "post_test",
              test_hold, test_ee, video)
        boundaries["post_test_end"] = int(env._physics_step)
    video_meta = video.close()
    trace = env.trace()
    arrays = trace_arrays(trace)
    np.savez_compressed(branch_dir / "trajectory.npz", **arrays)
    write_json(branch_dir / "trace.json", trace)
    failures = []
    for record in records:
        if record["actual_physics_steps"] != record["expected_physics_steps"]:
            failures.append(record["phase"] + "_step_count")
        if record["endpoint_error_m"] > config["robot"]["max_tracking_error_m"]:
            failures.append(record["phase"] + "_tracking_error")
    metadata = {
        "condition": condition, "arm": arm,
        "initial_to_base_max_abs": max_state_difference(base_state, initial),
        "phase_boundaries": boundaries, "command_records": records,
        "test_scripted": True, "test_executed": not stop_after_probe,
        "physics_steps": int(env._physics_step), "trace_samples": len(trace),
        "execution_failures": failures, "video": video_meta}
    write_json(branch_dir / "metadata.json", metadata)
    return metadata, initial, arrays


def run_pair(config: Dict[str, Any], stop_after_probe: bool = False):
    """Run the strict two-condition Pair and write all raw artifacts."""
    pair_id = config["pair_id"] + ("_probe_only" if stop_after_probe else "")
    pair_dir = Path(config["output_root"]) / ("pair_" + pair_id)
    pair_dir.mkdir(parents=True, exist_ok=True)
    env = SoftBlockPushEnv(config)
    try:
        base_state = capture_explicit_state(env)
        save_explicit_state(str(pair_dir / "base_state.npz"), base_state)
        script = build_fixed_command_script(env, config, base_state)
        high_script = copy_fixed_command_script(script)
        free_commands = command_arrays(script)
        high_commands = command_arrays(high_script)
        arrays_equal = all(np.array_equal(free_commands[key], high_commands[key])
                           for key in free_commands)
        if not arrays_equal:
            raise RuntimeError("fixed command arrays differ")
        write_json(pair_dir / "fixed_command_script.json", script)
        np.savez_compressed(pair_dir / "fixed_command_arrays.npz", **free_commands)
        branch_meta, initial_states, traces = {}, {}, {}
        for condition, commands in zip(CONDITIONS, (script, high_script)):
            branch_meta[condition], initial_states[condition], traces[condition] = run_branch(
                env, condition, commands, config, pair_dir, base_state,
                stop_after_probe)
        free, high = traces[CONDITIONS[0]], traces[CONDITIONS[1]]
        step_equal = np.array_equal(free["physics_step"], high["physics_step"])
        phase_equal = np.array_equal(free["phase"], high["phase"])
        edge_meta = env.soft_block.edge_metadata
        metadata = {
            "pair_id": pair_id, "seed": config["seed"],
            "dataset_role": config["dataset_role"],
            "conditions": list(CONDITIONS), "stop_after_probe": stop_after_probe,
            "pairing_mode": "save_restore_fixed_joint_targets",
            "initial_state_max_abs": max_state_difference(
                initial_states[CONDITIONS[0]], initial_states[CONDITIONS[1]]),
            "base_to_free_max_abs": max_state_difference(
                base_state, initial_states[CONDITIONS[0]]),
            "base_to_high_max_abs": max_state_difference(
                base_state, initial_states[CONDITIONS[1]]),
            "fixed_command_arrays_equal": arrays_equal,
            "physics_step_arrays_equal": bool(step_equal),
            "phase_arrays_equal": bool(phase_equal),
            "node_count": len(env.soft_block.body_ids),
            "visible_count": len(env.soft_block.top_indices),
            "top_indices": env.soft_block.top_indices.tolist(),
            "bottom_indices": env.soft_block.bottom_indices.tolist(),
            "lowdim_layout": LOWDIM_LAYOUT,
            "structural_edges": env.soft_block.edges["structural"].tolist(),
            "structural_rest_lengths": [
                edge["rest_length"] for edge in edge_meta["structural"]],
            "shear_edges": env.soft_block.edges["shear"].tolist(),
            "shear_rest_lengths": [edge["rest_length"] for edge in edge_meta["shear"]],
            "initial_pusher_node_signed_distance": (
                env.initial_pusher_node_signed_distance),
            "settle_recenter_xy_offset": env.settle_recenter_xy_offset.tolist(),
            "config": config, "branches": branch_meta}
        write_json(pair_dir / "metadata.json", metadata)
        return pair_dir, metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after-probe", action="store_true")
    args = parser.parse_args()
    pair_dir, metadata = run_pair(
        load_config(args.config), stop_after_probe=args.stop_after_probe)
    print("pair_dir={}".format(pair_dir))
    print("fixed_command_arrays_equal={}".format(
        metadata["fixed_command_arrays_equal"]))


if __name__ == "__main__":
    main()
