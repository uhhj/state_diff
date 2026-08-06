"""Run a strict R1 Kelvin-Voigt Free/High counterfactual Pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush.fixed_commands import (  # noqa: E402
    build_fixed_command_script, command_arrays, copy_fixed_command_script)
from scripts.experiment3.phase0_soft_blockpush.snapshot import (  # noqa: E402
    capture_explicit_state, max_state_difference, save_explicit_state)
from scripts.experiment3.phase0_soft_blockpush_r1.common import (  # noqa: E402
    SPRING_FIELDS, load_config, trace_arrays, with_material_profile, write_json)
from state_diff.env.block_pushing.soft_block_pushing import (  # noqa: E402
    LOWDIM_LAYOUT, SoftBlockPushEnv)
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (  # noqa: E402
    KELVIN_VOIGT_BULLET_SUBSTEPS)


CONDITIONS = ("uniform_low", "right_local_high")
FIELDS = (
    "physics_step", "phase", "node_positions", "node_velocities",
    "visible_keypoints", "node_orientations", "joint_positions",
    "joint_velocities", "joint_motor_torque", "joint_reaction_wrench",
    "ee_position", "ee_velocity", "ee_target_position",
    "ee_tracking_error_xyz", "ee_contact_wrench", "goal_xy",
    "oracle_patch_normal_force", "oracle_patch_tangential_force",
    "oracle_patch_mean_slip_speed", "oracle_patch_stick_ratio",
    "oracle_patch_contact_count", *SPRING_FIELDS)


class BranchVideo:
    """Write one fixed-camera branch video at the configured stride."""

    def __init__(self, env: SoftBlockPushEnv, path: Path, config: dict):
        self.env, self.path = env, path
        self.stride = int(config["camera"]["frame_stride"])
        height, width = config["camera"]["image_size"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"),
            float(config["camera"]["fps"]), (int(width), int(height)))
        if not self.writer.isOpened():
            raise RuntimeError("unable to open branch video")
        self.frames = 0

    def record(self) -> None:
        """Record the current state when the physics stride is due."""
        if self.env._physics_step % self.stride == 0:
            self.writer.write(cv2.cvtColor(
                self.env.render(), cv2.COLOR_RGB2BGR))
            self.frames += 1

    def close(self) -> dict:
        """Release the writer and return evidence metadata."""
        self.writer.release()
        return {"path": str(self.path), "frames": self.frames,
                "available": self.path.exists()}


def _hold(env: SoftBlockPushEnv, count: int, phase: str,
          joint_target: np.ndarray, ee_target: np.ndarray,
          video: BranchVideo) -> None:
    env.set_phase(phase); env.set_ee_target_position(ee_target)
    for _ in range(int(count)):
        env.set_fixed_joint_target(joint_target)
        env.step_one_physics(); video.record()


def _execute(env: SoftBlockPushEnv, phase: dict, video: BranchVideo) -> dict:
    start, actual = env._physics_step, []
    for ee_target, joint_target in zip(phase["ee_targets"], phase["joint_targets"]):
        env.set_phase(phase["phase"])
        env.set_ee_target_position(np.asarray(ee_target))
        env.set_fixed_joint_target(np.asarray(joint_target))
        env.step_one_physics(); video.record()
        actual.append(env._observation()["ee_position"].tolist())
    target, endpoint = np.asarray(phase["ee_targets"][-1]), np.asarray(actual[-1])
    return {"phase": phase["phase"], "command_count": len(actual),
            "physics_step_start": int(start), "physics_step_end": env._physics_step,
            "expected_physics_steps": len(actual),
            "actual_physics_steps": env._physics_step - start,
            "endpoint_error_m": float(np.linalg.norm(endpoint - target)),
            "final_target_position": target.tolist(),
            "final_actual_position": endpoint.tolist()}


def _run_branch(env: SoftBlockPushEnv, condition: str, script: dict,
                config: Dict[str, Any], pair_dir: Path, base_state: dict,
                stop_after_probe: bool) -> tuple[dict, dict, Dict[str, np.ndarray]]:
    env.restore_saved_state(env.saved_state_id); env.clear_trace()
    arm = env.arm_condition(condition)
    initial = capture_explicit_state(env)
    branch_dir = pair_dir / condition; branch_dir.mkdir(parents=True, exist_ok=True)
    video = BranchVideo(env, branch_dir / "video.mp4", config)
    hold = np.asarray(script["hold_joint_target"])
    start_ee = np.asarray(base_state["ee_target_position"])
    boundaries = {"no_action_start": 0}
    records = []
    _hold(env, config["execution"]["no_action_steps"], "no_action",
          hold, start_ee, video)
    boundaries["no_action_end"] = env._physics_step
    records.append(_execute(env, script["phases"][0], video))
    boundaries["probe_end"] = env._physics_step
    probe_joint = np.asarray(script["phases"][0]["joint_targets"][-1])
    probe_ee = np.asarray(script["phases"][0]["ee_targets"][-1])
    _hold(env, config["execution"]["post_probe_steps"], "post_probe",
          probe_joint, probe_ee, video)
    boundaries["post_probe_end"] = env._physics_step
    if not stop_after_probe:
        records.append(_execute(env, script["phases"][1], video))
        boundaries["test_end"] = env._physics_step
        test_joint = np.asarray(script["phases"][1]["joint_targets"][-1])
        test_ee = np.asarray(script["phases"][1]["ee_targets"][-1])
        _hold(env, config["execution"]["post_test_steps"], "post_test",
              test_joint, test_ee, video)
        boundaries["post_test_end"] = env._physics_step
    trace = env.trace(); arrays = trace_arrays(trace, FIELDS)
    np.savez_compressed(branch_dir / "trajectory.npz", **arrays)
    write_json(branch_dir / "trace.json", trace)
    failures = [record["phase"] for record in records
                if (record["actual_physics_steps"] != record["expected_physics_steps"]
                    or record["endpoint_error_m"] >
                    config["robot"]["max_tracking_error_m"])]
    metadata = {
        "condition": condition, "arm": arm,
        "initial_to_base_max_abs": max_state_difference(base_state, initial),
        "phase_boundaries": boundaries, "command_records": records,
        "test_scripted": True, "test_executed": not stop_after_probe,
        "physics_steps": env._physics_step, "trace_samples": len(trace),
        "execution_failures": failures, "video": video.close()}
    write_json(branch_dir / "metadata.json", metadata)
    return metadata, initial, arrays


def run_pair(config_path: str, frozen_material_report: str,
             stop_after_probe: bool = False) -> tuple[Path, dict]:
    """Validate the frozen coupon, run both branches, and save Pair evidence."""
    coupon = json.loads(Path(frozen_material_report).read_text(encoding="utf-8"))
    if coupon.get("verdict") != "PHASE0B_R1_COUPON_COMPLETE":
        raise RuntimeError("material coupon is not complete")
    config = with_material_profile(
        load_config(config_path), coupon["material_profile_path"])
    pair_id = config["pair_id"] + ("_probe_only" if stop_after_probe else "")
    pair_dir = Path(config["output_root"]) / ("pair_" + pair_id)
    pair_dir.mkdir(parents=True, exist_ok=True)
    env = SoftBlockPushEnv(config)
    try:
        base = capture_explicit_state(env)
        bottom = env.soft_block.bottom_indices
        positions = base["node_positions"][bottom, :2]
        x0, x1, y0, y1 = env.floor.bounds["patch"]
        mask = ((positions[:, 0] >= x0) & (positions[:, 0] <= x1)
                & (positions[:, 1] >= y0) & (positions[:, 1] <= y1))
        inside = bottom[mask]; outside = bottom[~mask]
        if not 2 <= len(inside) <= 6:
            raise RuntimeError("R1 geometry requires 2-6 initial patch nodes; got {}".format(
                len(inside)))
        save_explicit_state(str(pair_dir / "base_state.npz"), base)
        script = build_fixed_command_script(env, config, base)
        high_script = copy_fixed_command_script(script)
        commands_free, commands_high = command_arrays(script), command_arrays(high_script)
        arrays_equal = all(np.array_equal(commands_free[key], commands_high[key])
                           for key in commands_free)
        write_json(pair_dir / "fixed_command_script.json", script)
        np.savez_compressed(pair_dir / "fixed_command_arrays.npz", **commands_free)
        branch_meta, initial_states, traces = {}, {}, {}
        for condition, commands in zip(CONDITIONS, (script, high_script)):
            branch_meta[condition], initial_states[condition], traces[condition] = _run_branch(
                env, condition, commands, config, pair_dir, base, stop_after_probe)
        free, high = traces[CONDITIONS[0]], traces[CONDITIONS[1]]
        metadata = {
            "pair_id": pair_id, "seed": config["seed"],
            "stop_after_probe": stop_after_probe,
            "pairing_mode": "save_restore_fixed_joint_targets",
            "initial_state_max_abs": max_state_difference(
                initial_states[CONDITIONS[0]], initial_states[CONDITIONS[1]]),
            "fixed_command_arrays_equal": arrays_equal,
            "physics_step_arrays_equal": bool(np.array_equal(
                free["physics_step"], high["physics_step"])),
            "phase_arrays_equal": bool(np.array_equal(free["phase"], high["phase"])),
            "node_count": len(env.soft_block.body_ids),
            "visible_count": len(env.soft_block.top_indices),
            "top_indices": env.soft_block.top_indices.tolist(),
            "bottom_indices": bottom.tolist(),
            "initial_patch_bottom_indices": inside.tolist(),
            "initial_outside_bottom_indices": outside.tolist(),
            "material_model": env.material_model,
            "material_profile": env.material_profile,
            "material_profile_path": config["soft_block"]["material_profile_path"],
            "coupon_report_path": str(frozen_material_report),
            "coupon_verdict": coupon["verdict"],
            "steps_per_policy": int(config["physics"]["physics_hz"] //
                                    config["physics"]["policy_hz"]),
            "policy_hz": config["physics"]["policy_hz"],
            "physics_hz": config["physics"]["physics_hz"],
            "bullet_internal_substeps": KELVIN_VOIGT_BULLET_SUBSTEPS,
            "lowdim_layout": LOWDIM_LAYOUT,
            "structural_edges": env.soft_block.edges["structural"].tolist(),
            "structural_rest_lengths": [e["rest_length"] for e in
                                        env.soft_block.edge_metadata["structural"]],
            "config": config, "branches": branch_meta}
        write_json(pair_dir / "metadata.json", metadata)
        return pair_dir, metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-material-report", required=True)
    parser.add_argument("--stop-after-probe", action="store_true")
    args = parser.parse_args()
    pair_dir, metadata = run_pair(
        args.config, args.frozen_material_report, args.stop_after_probe)
    print("pair_dir={}".format(pair_dir))
    print("initial_patch_node_count={}".format(
        len(metadata["initial_patch_bottom_indices"])))


if __name__ == "__main__":
    main()
