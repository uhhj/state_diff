"""Independent fixed-face material coupon for Kelvin-Voigt soft blocks."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from scripts.experiment3.phase0_soft_blockpush.common import (
    floor_config, soft_block_config)
from state_diff.env.block_pushing.friction_tiles import FrictionTileFloor
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KELVIN_VOIGT_BULLET_SUBSTEPS, KelvinVoigtSoftBlock,
    load_material_profile)
from state_diff.env.block_pushing.soft_block_lattice import node_index


def coupon_ramp_scale(step: int, total_steps: int) -> float:
    """Return the half-cosine ramp value for a one-based ramp step."""
    if not 1 <= step <= total_steps:
        raise ValueError("ramp step outside one-based interval")
    return float(0.5 - 0.5 * np.cos(np.pi * step / total_steps))


class MaterialCoupon:
    """XArm-free material scene with a fixed and a distributed-load face."""

    def __init__(self, config: Dict[str, Any], material_profile_path: str):
        self.config = config
        self.material_profile_path = str(material_profile_path)
        self.material, self.material_profile = load_material_profile(
            self.material_profile_path)
        self.client = bullet_client.BulletClient(pybullet.DIRECT)
        self.trace = []
        self._setup()

    def _setup(self) -> None:
        physics = self.config["physics"]
        client = self.client
        client.resetSimulation()
        client.setTimeStep(float(physics["fixed_timestep"]))
        client.setPhysicsEngineParameter(
            numSubSteps=KELVIN_VOIGT_BULLET_SUBSTEPS)
        client.setPhysicsEngineParameter(
            numSolverIterations=int(physics["solver_iterations"]),
            deterministicOverlappingPairs=int(
                physics["deterministic_overlapping_pairs"]),
            enableConeFriction=int(physics["enable_cone_friction"]))
        client.setGravity(*self.config["physics"]["gravity"])
        self.floor = FrictionTileFloor(client, floor_config(self.config))
        self.floor.set_condition("uniform_low")
        block_cfg = soft_block_config(self.config)
        soft = self.config["soft_block"]
        self.block = KelvinVoigtSoftBlock(
            client, block_cfg, self.material,
            float(soft["spring_force_cap_n"]), tuple(soft["center_xy"]),
            float(soft["yaw_deg"]))
        self.anchor_indices = np.asarray([
            node_index(block_cfg.nx - 1, j, k, block_cfg)
            for k in range(block_cfg.nz) for j in range(block_cfg.ny)], dtype=np.int64)
        self.load_indices = np.asarray([
            node_index(0, j, k, block_cfg)
            for k in range(block_cfg.nz) for j in range(block_cfg.ny)], dtype=np.int64)
        self.fixture_constraint_ids = []
        for index in self.anchor_indices:
            position = self.block.positions()[int(index)]
            constraint = int(client.createConstraint(
                parentBodyUniqueId=self.block.body_ids[int(index)],
                parentLinkIndex=-1, childBodyUniqueId=-1, childLinkIndex=-1,
                jointType=client.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=position.tolist()))
            self.fixture_constraint_ids.append(constraint)

    def _record(self, step: int, phase: str, scale: float,
                external_force: np.ndarray, stats) -> None:
        positions = self.block.positions()
        self.trace.append({
            "physics_step": int(step), "phase": phase,
            "node_positions": positions.tolist(),
            "node_velocities": self.block.velocities().tolist(),
            "visible_keypoints": positions[self.block.top_indices].tolist(),
            "spring_energy_structural_j": stats.energy_by_kind_j["structural"],
            "spring_energy_shear_j": stats.energy_by_kind_j["shear"],
            "spring_energy_bending_j": stats.energy_by_kind_j["bending"],
            "spring_energy_total_j": sum(stats.energy_by_kind_j.values()),
            "spring_max_abs_force_n": stats.max_abs_force_n,
            "spring_capped_force_count": stats.capped_force_count,
            "spring_force_evaluation_count": stats.force_evaluation_count,
            "external_load_scale": float(scale),
            "external_load_force_n": external_force.tolist(),
        })

    def _step(self, step: int, phase: str, scale: float) -> None:
        stats = self.block.apply_internal_forces()
        coupon = self.config["coupon"]
        direction = np.asarray(coupon["load_direction_xyz"], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        total = float(coupon["total_load_force_n"]) * float(scale) * direction
        per_node = total / len(self.load_indices)
        if scale:
            positions = self.block.positions()
            for index in self.load_indices:
                self.client.applyExternalForce(
                    self.block.body_ids[int(index)], -1, per_node.tolist(),
                    positions[int(index)].tolist(), self.client.WORLD_FRAME)
        self.client.stepSimulation()
        self._record(step, phase, scale, total, stats)

    def run(self, frame_callback: Optional[Callable[[int, str], None]] = None) -> list:
        """Execute no-action, smooth load, hold, and unloaded recovery."""
        self.trace = []
        step = 0
        coupon = self.config["coupon"]
        phases = (
            ("no_action", coupon["no_action_steps"]),
            ("load_ramp", coupon["load_ramp_steps"]),
            ("load_hold", coupon["load_hold_steps"]),
            ("recovery", coupon["recovery_steps"]),
        )
        for phase, count in phases:
            for local in range(1, int(count) + 1):
                step += 1
                if phase == "load_ramp":
                    scale = coupon_ramp_scale(local, int(count))
                elif phase == "load_hold":
                    scale = 1.0
                else:
                    scale = 0.0
                self._step(step, phase, scale)
                if frame_callback is not None:
                    frame_callback(step, phase)
        return list(self.trace)

    def render(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Render a stable top-oblique coupon view."""
        view = self.client.computeViewMatrix(
            cameraEyePosition=[.52, -.30, .20],
            cameraTargetPosition=[.40, -.15, .01], cameraUpVector=[0, 0, 1])
        projection = self.client.computeProjectionMatrixFOV(
            fov=55, aspect=width / height, nearVal=.01, farVal=2.0)
        _, _, rgba, _, _ = self.client.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=projection,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self) -> None:
        """Disconnect the coupon-owned Bullet client."""
        self.client.disconnect()
