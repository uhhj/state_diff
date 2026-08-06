"""Explicit Kelvin-Voigt mass-spring mechanics for the 72-node soft block."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from state_diff.env.block_pushing.soft_block_lattice import (
    SoftBlockConfig, build_edge_metadata, initial_node_positions, node_index)


KELVIN_VOIGT_BULLET_SUBSTEPS = 2


@dataclass(frozen=True)
class KelvinVoigtMaterial:
    """Positive axial stiffness and damping coefficients by edge family."""

    structural_stiffness_n_per_m: float
    structural_damping_ns_per_m: float
    shear_stiffness_n_per_m: float
    shear_damping_ns_per_m: float
    bending_stiffness_n_per_m: float
    bending_damping_ns_per_m: float

    def validate(self) -> None:
        """Reject non-physical coefficients."""
        values = (
            self.structural_stiffness_n_per_m,
            self.structural_damping_ns_per_m,
            self.shear_stiffness_n_per_m,
            self.shear_damping_ns_per_m,
            self.bending_stiffness_n_per_m,
            self.bending_damping_ns_per_m,
        )
        if min(values) <= 0:
            raise ValueError("all material coefficients must be positive")

    def coefficients(self, kind: str) -> Tuple[float, float]:
        """Return stiffness and damping for one edge family."""
        if kind not in ("structural", "shear", "bending"):
            raise ValueError("unknown edge kind: {}".format(kind))
        return (float(getattr(self, kind + "_stiffness_n_per_m")),
                float(getattr(self, kind + "_damping_ns_per_m")))


@dataclass(frozen=True)
class SpringStepStats:
    """Auditable aggregate telemetry from one internal-force evaluation."""

    energy_by_kind_j: Dict[str, float]
    max_abs_force_n: float
    capped_force_count: int
    force_evaluation_count: int


def zero_spring_stats() -> SpringStepStats:
    """Return zero telemetry for legacy or not-yet-stepped mechanics."""
    return SpringStepStats(
        {"structural": 0.0, "shear": 0.0, "bending": 0.0}, 0.0, 0, 0)


def material_from_dict(payload: dict) -> KelvinVoigtMaterial:
    """Construct and validate a material from a JSON-compatible mapping."""
    material = KelvinVoigtMaterial(**{
        key: float(payload[key]) for key in (
            "structural_stiffness_n_per_m", "structural_damping_ns_per_m",
            "shear_stiffness_n_per_m", "shear_damping_ns_per_m",
            "bending_stiffness_n_per_m", "bending_damping_ns_per_m")})
    material.validate()
    return material


def load_material_profile(path: str) -> Tuple[KelvinVoigtMaterial, dict]:
    """Load a material profile and return both typed and raw forms."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return material_from_dict(payload), payload


def kelvin_voigt_edge_force(
    position_a: np.ndarray,
    velocity_a: np.ndarray,
    position_b: np.ndarray,
    velocity_b: np.ndarray,
    rest_length_m: float,
    stiffness_n_per_m: float,
    damping_ns_per_m: float,
    force_cap_n: float,
) -> tuple[np.ndarray, bool, float]:
    """Return force on A toward B, cap flag, and spring energy."""
    delta = np.asarray(position_b, dtype=np.float64) - np.asarray(
        position_a, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        raise FloatingPointError("collapsed spring edge")
    direction = delta / length
    relative_velocity = np.asarray(velocity_b) - np.asarray(velocity_a)
    extension = length - float(rest_length_m)
    scalar_force = (float(stiffness_n_per_m) * extension
                    + float(damping_ns_per_m)
                    * float(np.dot(relative_velocity, direction)))
    force = scalar_force * direction
    norm = float(np.linalg.norm(force))
    capped = norm > float(force_cap_n)
    if capped:
        force *= float(force_cap_n) / norm
    energy = 0.5 * float(stiffness_n_per_m) * extension ** 2
    return force, capped, float(energy)


class KelvinVoigtSoftBlock:
    """Dynamic sphere nodes coupled only by explicit Kelvin-Voigt forces."""

    def __init__(self, client, config: SoftBlockConfig,
                 material: KelvinVoigtMaterial, force_cap_n: float,
                 center_xy: Tuple[float, float], yaw_deg: float = 0.0):
        config.validate()
        material.validate()
        if force_cap_n <= 0:
            raise ValueError("force cap must be positive")
        self.client = client
        self.config = config
        self.material = material
        self.force_cap_n = float(force_cap_n)
        self.initial_positions = initial_node_positions(config, center_xy, yaw_deg)
        self.edge_metadata = build_edge_metadata(config)
        self.edges = {kind: np.asarray(
            [[edge["a"], edge["b"]] for edge in records], dtype=np.int64)
            for kind, records in self.edge_metadata.items()}
        self.top_indices = np.asarray([
            node_index(i, j, config.nz - 1, config)
            for j in range(config.ny) for i in range(config.nx)], dtype=np.int64)
        self.bottom_indices = np.asarray([
            node_index(i, j, 0, config)
            for j in range(config.ny) for i in range(config.nx)], dtype=np.int64)
        self.body_ids: List[int] = []
        self.constraint_ids: List[int] = []
        self._create_nodes()

    def _create_nodes(self) -> None:
        collision = self.client.createCollisionShape(
            self.client.GEOM_SPHERE, radius=self.config.node_radius_m)
        visual = self.client.createVisualShape(
            self.client.GEOM_SPHERE, radius=self.config.node_radius_m,
            rgbaColor=[0.20, 0.55, 0.92, 1.0])
        mass = self.config.total_mass_kg / self.config.num_nodes
        for position in self.initial_positions:
            body = int(self.client.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=position.tolist()))
            self.client.changeDynamics(
                body, -1, lateralFriction=self.config.node_lateral_friction,
                rollingFriction=self.config.node_rolling_friction,
                restitution=0.0, linearDamping=self.config.linear_damping,
                angularDamping=self.config.angular_damping)
            self.body_ids.append(body)
        for a, b in self.edges["structural"]:
            self.client.setCollisionFilterPair(
                self.body_ids[int(a)], self.body_ids[int(b)], -1, -1,
                enableCollision=0)

    def positions(self) -> np.ndarray:
        """Return node positions in deterministic order."""
        return np.asarray([self.client.getBasePositionAndOrientation(body)[0]
                           for body in self.body_ids], dtype=np.float64)

    def velocities(self) -> np.ndarray:
        """Return node linear velocities in deterministic order."""
        return np.asarray([self.client.getBaseVelocity(body)[0]
                           for body in self.body_ids], dtype=np.float64)

    def orientations(self) -> np.ndarray:
        """Return node quaternions in deterministic order."""
        return np.asarray([self.client.getBasePositionAndOrientation(body)[1]
                           for body in self.body_ids], dtype=np.float64)

    def visible_positions(self) -> np.ndarray:
        """Return fixed top-layer keypoints."""
        return self.positions()[self.top_indices]

    def center_of_mass(self) -> np.ndarray:
        """Return equal-mass node center of mass."""
        return np.mean(self.positions(), axis=0)

    def edge_lengths(self, kind: str) -> np.ndarray:
        """Return current lengths for one edge family."""
        positions = self.positions()
        pairs = self.edges[kind]
        return np.linalg.norm(
            positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)

    def structural_edge_ratios(self) -> np.ndarray:
        """Return current structural length/rest ratios."""
        rest = np.asarray([edge["rest_length"]
                           for edge in self.edge_metadata["structural"]])
        return self.edge_lengths("structural") / rest

    def recenter_xy(self, center_xy: Tuple[float, float]) -> np.ndarray:
        """Rigidly recenter settled nodes in XY and clear residual velocity."""
        offset = np.asarray(center_xy, dtype=np.float64) - self.center_of_mass()[:2]
        for body in self.body_ids:
            position, orientation = self.client.getBasePositionAndOrientation(body)
            shifted = np.asarray(position, dtype=np.float64)
            shifted[:2] += offset
            self.client.resetBasePositionAndOrientation(
                body, shifted.tolist(), orientation)
            self.client.resetBaseVelocity(
                body, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        return offset

    def apply_internal_forces(self) -> SpringStepStats:
        """Evaluate every edge, aggregate node forces, and apply once per node."""
        positions, velocities = self.positions(), self.velocities()
        node_forces = np.zeros((self.config.num_nodes, 3), dtype=np.float64)
        energies = {"structural": 0.0, "shear": 0.0, "bending": 0.0}
        capped_count, evaluations, maximum = 0, 0, 0.0
        for kind in ("structural", "shear", "bending"):
            stiffness, damping = self.material.coefficients(kind)
            for edge in self.edge_metadata[kind]:
                a, b = int(edge["a"]), int(edge["b"])
                force, capped, energy = kelvin_voigt_edge_force(
                    positions[a], velocities[a], positions[b], velocities[b],
                    edge["rest_length"], stiffness, damping, self.force_cap_n)
                node_forces[a] += force
                node_forces[b] -= force
                energies[kind] += energy
                capped_count += int(capped)
                evaluations += 1
                maximum = max(maximum, float(np.linalg.norm(force)))
        for body, force, position in zip(self.body_ids, node_forces, positions):
            self.client.applyExternalForce(
                objectUniqueId=body, linkIndex=-1, forceObj=force.tolist(),
                posObj=position.tolist(), flags=self.client.WORLD_FRAME)
        return SpringStepStats(energies, maximum, capped_count, evaluations)
