"""Separated free-space axial/shear R2 material coupon."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    ramp_scale, soft_block_config)
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtSoftBlock, load_material_profile)
from state_diff.env.block_pushing.manual_microstep_integrator import (
    ManualMicrostepConfig, ManualMicrostepIntegrator)
from state_diff.env.block_pushing.soft_block_metrics import rigid_aligned_rmse
from state_diff.env.block_pushing.soft_block_lattice import node_index


COUPON_MODES = ("axial", "shear")


class MaterialCouponR2:
    """Zero-gravity, static-face coupon using true manual microsteps."""

    def __init__(self, config: Dict[str, Any], material_profile_path: str,
                 microsteps_per_outer: int, mode: str):
        if mode not in COUPON_MODES:
            raise ValueError("unknown coupon mode")
        self.config, self.mode = config, mode
        self.material, self.material_profile = load_material_profile(
            material_profile_path)
        self.material_profile_path = material_profile_path
        self.client = bullet_client.BulletClient(pybullet.DIRECT)
        physics = config["physics"]
        self.client.setPhysicsEngineParameter(
            numSolverIterations=physics["solver_iterations"],
            deterministicOverlappingPairs=physics["deterministic_overlapping_pairs"],
            enableConeFriction=physics["enable_cone_friction"])
        self.client.setGravity(0, 0, 0)
        block_config = soft_block_config(config)
        self.anchor_indices = np.asarray([
            node_index(block_config.nx - 1, j, k, block_config)
            for k in range(block_config.nz) for j in range(block_config.ny)], dtype=int)
        self.load_indices = np.asarray([
            node_index(0, j, k, block_config)
            for k in range(block_config.nz) for j in range(block_config.ny)], dtype=int)
        soft = config["soft_block"]
        self.block = KelvinVoigtSoftBlock(
            self.client, block_config, self.material,
            soft["spring_force_cap_n"], tuple(soft["center_xy"]), soft["yaw_deg"],
            static_indices=self.anchor_indices.tolist())
        self.integrator = ManualMicrostepIntegrator(
            self.client, ManualMicrostepConfig(
                physics["outer_timestep_s"], microsteps_per_outer))
        direction_key = mode + "_load_direction_xyz"
        self.direction = np.asarray(config["coupon"][direction_key], dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.trace = []

    def run(self, frame_callback: Optional[Callable[[int, str], None]] = None) -> list:
        """Run no-action, ramp, hold, and recovery at outer-step rate."""
        coupon = self.config["coupon"]
        phases = (("no_action", coupon["no_action_outer_steps"]),
                  ("load_ramp", coupon["load_ramp_outer_steps"]),
                  ("load_hold", coupon["load_hold_outer_steps"]),
                  ("recovery", coupon["recovery_outer_steps"]))
        reference, outer = None, 0
        for phase, count in phases:
            for local in range(1, int(count) + 1):
                outer += 1
                scale = (0.0 if phase in ("no_action", "recovery") else
                         (1.0 if phase == "load_hold" else ramp_scale(local, count)))
                total = (float(coupon["total_load_force_n"])
                         * scale * self.direction)

                def external(_micro: int, _count: int, _dt: float) -> None:
                    positions = self.block.positions(); per = total / len(self.load_indices)
                    for index in self.load_indices:
                        self.client.applyExternalForce(
                            self.block.body_ids[int(index)], -1, per.tolist(),
                            positions[int(index)].tolist(), self.client.WORLD_FRAME)

                stats = self.integrator.advance_outer_step(self.block, external)
                positions = self.block.positions()
                if phase == "no_action" and local == int(count):
                    reference = positions.copy()
                relative = np.zeros(3) if reference is None else (
                    np.mean(positions[self.load_indices] - reference[self.load_indices], axis=0)
                    - np.mean(positions[self.anchor_indices] - reference[self.anchor_indices], axis=0))
                primary = float(relative @ self.direction)
                lateral = float(np.linalg.norm(relative - primary * self.direction))
                rigid = 0.0 if reference is None else rigid_aligned_rmse(reference, positions)
                self.trace.append({
                    "outer_step": outer, "phase": phase,
                    "node_positions": positions.tolist(),
                    "node_velocities": self.block.velocities().tolist(),
                    "visible_keypoints": self.block.visible_positions().tolist(),
                    "external_load_scale": scale,
                    "external_load_force_n": total.tolist(),
                    "primary_face_displacement_m": primary,
                    "lateral_face_displacement_m": lateral,
                    "rigid_aligned_rmse_m": rigid,
                    "spring_energy_structural_j": stats.final_energy_by_kind_j["structural"],
                    "spring_energy_shear_j": stats.final_energy_by_kind_j["shear"],
                    "spring_energy_bending_j": stats.final_energy_by_kind_j["bending"],
                    "spring_energy_total_j": sum(stats.final_energy_by_kind_j.values()),
                    "spring_max_uncapped_edge_force_n": stats.max_uncapped_edge_force_n,
                    "spring_max_applied_edge_force_n": stats.max_applied_edge_force_n,
                    "spring_capped_force_count": stats.capped_force_count,
                    "spring_force_evaluation_count": stats.force_evaluation_count,
                    "spring_net_internal_force_residual_n": stats.max_net_internal_force_residual_n,
                    "spring_min_edge_length_m": stats.min_edge_length_m,
                    "spring_max_edge_length_m": stats.max_edge_length_m,
                    "microsteps_per_outer": stats.microsteps_per_outer})
                if frame_callback is not None:
                    frame_callback(outer, phase)
        return list(self.trace)

    def render(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Render an oblique view without floor or task objects."""
        view = self.client.computeViewMatrix(
            [.52, -.30, .20], [.40, -.15, .01], [0, 0, 1])
        projection = self.client.computeProjectionMatrixFOV(
            55, width / height, .01, 2)
        _, _, rgba, _, _ = self.client.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=projection,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self) -> None:
        """Disconnect the coupon client."""
        self.client.disconnect()
