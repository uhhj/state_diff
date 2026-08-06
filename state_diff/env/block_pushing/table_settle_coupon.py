"""Gravity-and-floor settle audit for the frozen R2 material candidate."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from scripts.experiment3.phase0_soft_blockpush_r2.common import soft_block_config
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtSoftBlock, load_material_profile)
from state_diff.env.block_pushing.manual_microstep_integrator import (
    ManualMicrostepConfig, ManualMicrostepIntegrator)


class TableSettleCoupon:
    """Settle all 72 dynamic nodes without recentering or velocity resets."""

    def __init__(self, config: Dict[str, Any], material_profile_path: str,
                 microsteps_per_outer: int):
        self.config = config
        self.material, self.material_profile = self._load_profile(
            material_profile_path)
        self.material_profile_path = material_profile_path
        self.client = bullet_client.BulletClient(pybullet.DIRECT)
        physics, settle = config["physics"], config["table_settle"]
        self.client.setPhysicsEngineParameter(
            numSolverIterations=physics["solver_iterations"],
            deterministicOverlappingPairs=physics["deterministic_overlapping_pairs"],
            enableConeFriction=physics["enable_cone_friction"])
        self.client.setGravity(*settle["gravity"])
        thickness, top = (float(settle["floor_thickness_m"]),
                          float(settle["floor_top_z_m"]))
        collision = self.client.createCollisionShape(
            self.client.GEOM_BOX, halfExtents=[1.0, 1.0, thickness / 2])
        visual = self.client.createVisualShape(
            self.client.GEOM_BOX, halfExtents=[1.0, 1.0, thickness / 2],
            rgbaColor=[.52, .52, .56, 1.0])
        self.floor_id = int(self.client.createMultiBody(
            0, collision, visual, basePosition=[0, 0, top - thickness / 2]))
        self.client.changeDynamics(
            self.floor_id, -1,
            lateralFriction=float(settle["floor_lateral_friction"]),
            rollingFriction=0.0, restitution=0.0)
        soft = config["soft_block"]
        self.block = self._create_block(soft_block_config(config), soft)
        self.integrator = ManualMicrostepIntegrator(
            self.client, ManualMicrostepConfig(
                float(physics["outer_timestep_s"]), microsteps_per_outer))
        self.trace = []

    def run(self, frame_callback: Optional[Callable[[int, str], None]] = None) -> list:
        """Run fixed warmup and audit windows and return outer-step telemetry."""
        settle = self.config["table_settle"]
        phases = (("warmup", int(settle["warmup_outer_steps"])),
                  ("audit", int(settle["audit_outer_steps"])))
        outer = 0
        for phase, count in phases:
            for _ in range(count):
                outer += 1
                stats = self.integrator.advance_outer_step(self.block)
                positions, velocities = self.block.positions(), self.block.velocities()
                contacts = sum(len(self.client.getContactPoints(
                    bodyA=body, bodyB=self.floor_id)) for body in self.block.body_ids)
                self.trace.append({
                    "outer_step": outer, "phase": phase,
                    "node_positions": positions.tolist(),
                    "node_velocities": velocities.tolist(),
                    "visible_keypoints": self.block.visible_positions().tolist(),
                    "contact_count": contacts,
                    **self._energy_fields(stats),
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

    def _load_profile(self, path: str):
        return load_material_profile(path)

    def _create_block(self, block_config, soft: dict):
        return KelvinVoigtSoftBlock(
            self.client, block_config, self.material,
            float(soft["spring_force_cap_n"]), tuple(soft["center_xy"]),
            float(soft["yaw_deg"]))

    def _energy_fields(self, stats) -> dict:
        return {
            "spring_energy_structural_j": stats.final_energy_by_kind_j["structural"],
            "spring_energy_shear_j": stats.final_energy_by_kind_j["shear"],
            "spring_energy_bending_j": stats.final_energy_by_kind_j["bending"],
            "spring_energy_total_j": sum(stats.final_energy_by_kind_j.values())}

    def render(self, width: int = 320, height: int = 240) -> np.ndarray:
        """Render an oblique audit view."""
        view = self.client.computeViewMatrix(
            [.52, -.30, .20], [.40, -.15, .01], [0, 0, 1])
        projection = self.client.computeProjectionMatrixFOV(
            55, width / height, .01, 2)
        _, _, rgba, _, _ = self.client.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=projection,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self) -> None:
        self.client.disconnect()
