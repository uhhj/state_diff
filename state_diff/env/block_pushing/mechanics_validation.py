"""Worst-case two-node and 2x2x2 true-microstep validation scenes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtMaterial, KelvinVoigtSoftBlock, SpringStepStats,
    kelvin_voigt_edge_force)
from state_diff.env.block_pushing.manual_microstep_integrator import (
    ManualMicrostepConfig, ManualMicrostepIntegrator)
from state_diff.env.block_pushing.soft_block_metrics import rigid_aligned_rmse
from state_diff.env.block_pushing.soft_block_lattice import node_index


class TwoNodeSpring:
    """Minimal one-static/one-dynamic structural spring interface."""

    def __init__(self, client, material: KelvinVoigtMaterial, node_mass: float,
                 rest_length: float, extension: float, force_cap: float):
        self.client, self.material = client, material
        self.rest_length, self.force_cap = rest_length, force_cap
        self.body_ids = [
            client.createMultiBody(baseMass=0, basePosition=[0, 0, 0]),
            client.createMultiBody(baseMass=node_mass,
                                   basePosition=[rest_length + extension, 0, 0])]

    def apply_internal_forces(self) -> SpringStepStats:
        """Apply the one edge and expose standard spring telemetry."""
        positions = np.asarray([self.client.getBasePositionAndOrientation(b)[0]
                                for b in self.body_ids])
        velocities = np.asarray([self.client.getBaseVelocity(b)[0]
                                 for b in self.body_ids])
        force, capped, energy = kelvin_voigt_edge_force(
            positions[0], velocities[0], positions[1], velocities[1],
            self.rest_length, self.material.structural_stiffness_n_per_m,
            self.material.structural_damping_ns_per_m, self.force_cap)
        for body, applied in zip(self.body_ids, (force, -force)):
            self.client.applyExternalForce(
                body, -1, applied.tolist(), positions[self.body_ids.index(body)].tolist(),
                self.client.WORLD_FRAME)
        norm = float(np.linalg.norm(force))
        length = float(np.linalg.norm(positions[1] - positions[0]))
        return SpringStepStats(
            {"structural": energy, "shear": 0.0, "bending": 0.0},
            norm, norm, int(capped), 1, 0.0, length, length)


def _client(config: dict, microsteps: int):
    client = bullet_client.BulletClient(pybullet.DIRECT)
    physics = config["physics"]
    client.setPhysicsEngineParameter(
        numSolverIterations=int(physics["solver_iterations"]),
        deterministicOverlappingPairs=int(physics["deterministic_overlapping_pairs"]),
        enableConeFriction=int(physics["enable_cone_friction"]))
    client.setGravity(0, 0, 0)
    integrator = ManualMicrostepIntegrator(client, ManualMicrostepConfig(
        float(physics["outer_timestep_s"]), int(microsteps)))
    return client, integrator


def run_two_node_validation(config: dict, material: KelvinVoigtMaterial,
                            microsteps: int) -> Dict[str, np.ndarray]:
    """Run the A8 damped two-node oscillator for one candidate M."""
    client, integrator = _client(config, microsteps)
    settings = config["mechanics_validation"]
    mass = config["soft_block"]["total_mass_kg"] / 72
    spring = TwoNodeSpring(
        client, material, mass, config["soft_block"]["spacing_m"][0],
        settings["two_node_initial_extension_m"],
        config["soft_block"]["spring_force_cap_n"])
    rows = []
    try:
        for outer in range(1, int(settings["two_node_duration_outer_steps"]) + 1):
            stats = integrator.advance_outer_step(spring)
            position = np.asarray(client.getBasePositionAndOrientation(
                spring.body_ids[1])[0])
            velocity = np.asarray(client.getBaseVelocity(spring.body_ids[1])[0])
            length = float(np.linalg.norm(position))
            kinetic = .5 * mass * float(np.dot(velocity, velocity))
            spring_energy = stats.final_energy_by_kind_j["structural"]
            rows.append((outer, position, velocity, length,
                         length - spring.rest_length, kinetic, spring_energy,
                         kinetic + spring_energy, stats.capped_force_count,
                         stats.max_net_internal_force_residual_n))
    finally:
        client.disconnect()
    names = ("outer_step", "dynamic_position", "dynamic_velocity", "edge_length",
             "extension", "kinetic_energy", "spring_energy", "total_energy",
             "capped_force_count", "net_internal_force_residual")
    return {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}


def run_cube_validation(config: dict, material: KelvinVoigtMaterial,
                        microsteps: int, block_config) -> Dict[str, np.ndarray]:
    """Run the static-face 2x2x2 axial load/recovery scene."""
    client, integrator = _client(config, microsteps)
    anchors = np.asarray([node_index(1, j, k, block_config)
                          for k in range(2) for j in range(2)], dtype=np.int64)
    loads = np.asarray([node_index(0, j, k, block_config)
                        for k in range(2) for j in range(2)], dtype=np.int64)
    block = KelvinVoigtSoftBlock(
        client, block_config, material,
        config["soft_block"]["spring_force_cap_n"], (0, 0), 0,
        static_indices=anchors.tolist())
    for body in block.body_ids:
        client.setCollisionFilterGroupMask(body, -1, 0, 0)
    settings = config["mechanics_validation"]
    phases = (("no_action", settings["cube_no_action_outer_steps"]),
              ("load_ramp", settings["cube_ramp_outer_steps"]),
              ("load_hold", settings["cube_hold_outer_steps"]),
              ("recovery", settings["cube_recovery_outer_steps"]))
    reference, rows, outer = None, [], 0
    try:
        for phase, count in phases:
            for local in range(1, int(count) + 1):
                outer += 1
                scale = (0 if phase in ("no_action", "recovery") else
                         (1 if phase == "load_hold" else
                          .5 - .5 * np.cos(np.pi * local / count)))
                total = np.asarray([-settings["cube_total_load_force_n"] * scale, 0, 0])

                def external(_i: int, _m: int, _dt: float) -> None:
                    positions = block.positions(); per = total / len(loads)
                    for index in loads:
                        client.applyExternalForce(
                            block.body_ids[int(index)], -1, per.tolist(),
                            positions[int(index)].tolist(), client.WORLD_FRAME)

                stats = integrator.advance_outer_step(block, external)
                positions = block.positions()
                if phase == "no_action" and local == int(count):
                    reference = positions.copy()
                if reference is None:
                    primary, rigid = 0.0, 0.0
                else:
                    relative = (np.mean(positions[loads] - reference[loads], axis=0)
                                - np.mean(positions[anchors] - reference[anchors], axis=0))
                    primary = float(relative @ np.array([-1., 0, 0]))
                    rigid = rigid_aligned_rmse(reference, positions)
                ratios = block.structural_edge_ratios()
                rows.append((outer, phase, primary, rigid, np.min(ratios),
                             np.max(ratios), stats.capped_force_count,
                             stats.force_evaluation_count,
                             sum(stats.final_energy_by_kind_j.values()),
                             stats.max_net_internal_force_residual_n))
    finally:
        client.disconnect()
    names = ("outer_step", "phase", "primary_face_displacement", "rigid_rmse",
             "edge_ratio_min", "edge_ratio_max", "capped_force_count",
             "force_evaluation_count", "spring_energy",
             "net_internal_force_residual")
    return {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}
