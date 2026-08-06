"""Minimal three-node confirmation for the R3 angle force term."""
from __future__ import annotations

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from state_diff.env.block_pushing.angle_elastic_soft_block import (
    AngleElasticParameters, evaluate_angle_constraints)
from state_diff.env.block_pushing.kelvin_voigt_soft_block import SpringStepStats
from state_diff.env.block_pushing.manual_microstep_integrator import (
    ManualMicrostepConfig, ManualMicrostepIntegrator)


class AngleCell:
    """One static corner and two dynamic arms with no collisions."""

    def __init__(self, client, parameters: AngleElasticParameters,
                 initial_cosine_error: float, arm_length_m: float,
                 node_mass_kg: float):
        self.client, self.parameters = client, parameters
        sine = np.sqrt(1.0 - initial_cosine_error ** 2)
        locations = np.asarray([
            [0.0, 0.0, 0.0], [arm_length_m, 0.0, 0.0],
            [initial_cosine_error * arm_length_m, sine * arm_length_m, 0.0]])
        self.body_ids = [int(client.createMultiBody(
            baseMass=0.0 if index == 0 else node_mass_kg,
            baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
            basePosition=position.tolist()))
            for index, position in enumerate(locations)]
        self.triplets = np.asarray([[0, 1, 2]], dtype=np.int64)
        self.rest_cosines = np.asarray([parameters.rest_cosine])
        self.node_mass_kg = node_mass_kg

    def positions(self):
        return np.asarray([self.client.getBasePositionAndOrientation(body)[0]
                           for body in self.body_ids], dtype=np.float64)

    def velocities(self):
        return np.asarray([self.client.getBaseVelocity(body)[0]
                           for body in self.body_ids], dtype=np.float64)

    def evaluate(self):
        return evaluate_angle_constraints(
            self.positions(), self.velocities(), self.triplets,
            self.rest_cosines, self.parameters.stiffness_n_m,
            self.parameters.damping_n_m_s, self.parameters.force_cap_n)

    def apply_internal_forces(self):
        positions = self.positions()
        evaluation = evaluate_angle_constraints(
            positions, self.velocities(), self.triplets, self.rest_cosines,
            self.parameters.stiffness_n_m, self.parameters.damping_n_m_s,
            self.parameters.force_cap_n)
        forces = np.concatenate([
            evaluation.force_center_n, evaluation.force_first_n,
            evaluation.force_second_n], axis=0)
        for body, force, position in zip(self.body_ids, forces, positions):
            self.client.applyExternalForce(body, -1, force.tolist(),
                                           position.tolist(), self.client.WORLD_FRAME)
        arms = np.linalg.norm(positions[1:] - positions[0], axis=1)
        return SpringStepStats(
            {"angle": float(evaluation.energy_j[0])},
            float(evaluation.max_uncapped_force_n[0]),
            float(max(np.linalg.norm(force) for force in forces)),
            int(evaluation.capped[0]), 1,
            float(np.linalg.norm(np.sum(forces, axis=0))),
            float(np.min(arms)), float(np.max(arms)))


def run_angle_cell_validation(config: dict, parameters: AngleElasticParameters,
                              microsteps_per_outer: int) -> dict:
    """Run one fixed 240 Hz angle-cell trajectory and return arrays."""
    client = bullet_client.BulletClient(pybullet.DIRECT)
    try:
        client.setGravity(0, 0, 0)
        client.setPhysicsEngineParameter(
            numSolverIterations=int(config["physics"]["solver_iterations"]),
            deterministicOverlappingPairs=1, enableConeFriction=1)
        arm = float(config["soft_block"]["spacing_m"][0])
        cell = AngleCell(
            client, parameters,
            float(config["angle_validation"]["initial_cosine_error"]), arm,
            float(config["soft_block"]["total_mass_kg"]) / 72)
        integrator = ManualMicrostepIntegrator(
            client, ManualMicrostepConfig(
                float(config["physics"]["outer_timestep_s"]),
                int(microsteps_per_outer)))
        rows = []
        for outer in range(1, int(config["angle_validation"]["outer_steps"]) + 1):
            stats = integrator.advance_outer_step(cell)
            evaluation = cell.evaluate()
            positions, velocities = cell.positions(), cell.velocities()
            arms = np.linalg.norm(positions[1:] - positions[0], axis=1)
            kinetic = .5 * cell.node_mass_kg * float(np.sum(velocities[1:] ** 2))
            angle_energy = float(evaluation.energy_j[0])
            rows.append((outer, float(evaluation.cosine_error[0]),
                         float(evaluation.cosine_rate[0]), angle_energy, kinetic,
                         angle_energy + kinetic,
                         stats.max_uncapped_edge_force_n,
                         stats.capped_force_count, arms[0], arms[1]))
        values = np.asarray(rows)
        return {
            "outer_step": values[:, 0].astype(np.int64),
            "cosine_error": values[:, 1], "cosine_rate": values[:, 2],
            "angle_energy": values[:, 3], "kinetic_energy": values[:, 4],
            "total_energy": values[:, 5], "max_force": values[:, 6],
            "cap_count": values[:, 7].astype(np.int64),
            "arm_lengths": values[:, 8:10],
            "microsteps_per_outer": np.full(len(values), microsteps_per_outer)}
    finally:
        client.disconnect()
