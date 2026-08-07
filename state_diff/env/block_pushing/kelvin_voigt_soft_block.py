"""Explicit Kelvin-Voigt mass-spring mechanics for the 72-node soft block."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from state_diff.env.block_pushing.soft_block_lattice import (
    SoftBlockConfig, build_edge_metadata, initial_node_positions, node_index)


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
    max_uncapped_edge_force_n: float
    max_applied_edge_force_n: float
    capped_force_count: int
    force_evaluation_count: int
    net_internal_force_residual_n: float
    min_edge_length_m: float
    max_edge_length_m: float

    @property
    def max_abs_force_n(self) -> float:
        """Return the legacy name for maximum applied edge force."""
        return self.max_applied_edge_force_n


@dataclass(frozen=True)
class EdgeFamilyEvaluation:
    """Vectorized forces and telemetry for one deterministic edge family."""

    edge_forces_on_a: np.ndarray
    energy_j: np.ndarray
    capped: np.ndarray
    uncapped_force_norm_n: np.ndarray
    applied_force_norm_n: np.ndarray
    edge_lengths_m: np.ndarray


@dataclass(frozen=True)
class InternalForceEvaluation:
    """Aggregated node forces and telemetry before Bullet application."""

    node_forces_n: np.ndarray
    stats: SpringStepStats


def zero_spring_stats() -> SpringStepStats:
    """Return zero telemetry for legacy or not-yet-stepped mechanics."""
    return SpringStepStats(
        {"structural": 0.0, "shear": 0.0, "bending": 0.0},
        0.0, 0.0, 0, 0, 0.0, 0.0, 0.0)


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
    if int(payload.get("profile_version", 1)) == 2:
        multiplier = float(payload["stiffness_multiplier"])
        damping_ratio = float(payload["damping_ratio"])
        reduced_mass = float(payload["node_mass_kg"]) / 2.0
        for kind in ("structural", "shear", "bending"):
            stiffness = float(payload["base_stiffness_n_per_m"][kind]) * multiplier
            damping = 2.0 * damping_ratio * np.sqrt(stiffness * reduced_mass)
            if abs(stiffness - float(payload[kind + "_stiffness_n_per_m"])) > 1e-12:
                raise ValueError("R2 stiffness coefficient mismatch for " + kind)
            if abs(damping - float(payload[kind + "_damping_ns_per_m"])) > 1e-12:
                raise ValueError("R2 damping coefficient mismatch for " + kind)
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


def evaluate_edge_family(
    positions: np.ndarray, velocities: np.ndarray, pairs: np.ndarray,
    rest_lengths_m: np.ndarray, stiffness_n_per_m: float,
    damping_ns_per_m: float, force_cap_n: float,
) -> EdgeFamilyEvaluation:
    """Vectorize deterministic Kelvin-Voigt evaluation for one edge family."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    rest = np.asarray(rest_lengths_m, dtype=np.float64)
    a, b = pairs[:, 0], pairs[:, 1]
    delta = positions[b] - positions[a]
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 1e-12):
        raise FloatingPointError("collapsed spring edge")
    direction = delta / lengths[:, None]
    relative_velocity = velocities[b] - velocities[a]
    extension = lengths - rest
    scalar = (float(stiffness_n_per_m) * extension
              + float(damping_ns_per_m)
              * np.sum(relative_velocity * direction, axis=1))
    uncapped = scalar[:, None] * direction
    uncapped_norm = np.linalg.norm(uncapped, axis=1)
    scale = np.minimum(
        1.0, float(force_cap_n) / np.maximum(uncapped_norm, 1e-30))
    applied = uncapped * scale[:, None]
    return EdgeFamilyEvaluation(
        edge_forces_on_a=applied,
        energy_j=0.5 * float(stiffness_n_per_m) * extension ** 2,
        capped=uncapped_norm > float(force_cap_n),
        uncapped_force_norm_n=uncapped_norm,
        applied_force_norm_n=np.linalg.norm(applied, axis=1),
        edge_lengths_m=lengths)


class KelvinVoigtSoftBlock:
    """Dynamic sphere nodes coupled only by explicit Kelvin-Voigt forces."""

    def __init__(self, client, config: SoftBlockConfig,
                 material: KelvinVoigtMaterial, force_cap_n: float,
                 center_xy: Tuple[float, float], yaw_deg: float = 0.0,
                 static_indices: Optional[Sequence[int]] = None):
        config.validate()
        material.validate()
        if force_cap_n <= 0:
            raise ValueError("force cap must be positive")
        self.client = client
        self.config = config
        self.material = material
        self.force_cap_n = float(force_cap_n)
        self.static_indices = np.asarray(
            sorted(set([] if static_indices is None else static_indices)),
            dtype=np.int64)
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
        static_set = set(self.static_indices.tolist())
        for index, position in enumerate(self.initial_positions):
            body = int(self.client.createMultiBody(
                baseMass=0.0 if index in static_set else mass,
                baseCollisionShapeIndex=collision,
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

    def compute_internal_forces(
        self, positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
    ) -> InternalForceEvaluation:
        """Aggregate all edge forces without mutating Bullet state."""
        positions = self.positions() if positions is None else np.asarray(
            positions, dtype=np.float64)
        velocities = self.velocities() if velocities is None else np.asarray(
            velocities, dtype=np.float64)
        node_forces = np.zeros((self.config.num_nodes, 3), dtype=np.float64)
        energies = {"structural": 0.0, "shear": 0.0, "bending": 0.0}
        capped_count, evaluations = 0, 0
        max_uncapped, max_applied = 0.0, 0.0
        min_length, max_length = float("inf"), 0.0
        for kind in ("structural", "shear", "bending"):
            stiffness, damping = self.material.coefficients(kind)
            pairs = self.edges[kind]
            if len(pairs) == 0:
                continue
            evaluation = evaluate_edge_family(
                positions, velocities, pairs,
                np.asarray([edge["rest_length"]
                            for edge in self.edge_metadata[kind]]),
                stiffness, damping, self.force_cap_n)
            np.add.at(node_forces, pairs[:, 0], evaluation.edge_forces_on_a)
            np.add.at(node_forces, pairs[:, 1], -evaluation.edge_forces_on_a)
            energies[kind] = float(np.sum(evaluation.energy_j))
            capped_count += int(np.sum(evaluation.capped))
            evaluations += len(pairs)
            max_uncapped = max(max_uncapped, float(np.max(
                evaluation.uncapped_force_norm_n)))
            max_applied = max(max_applied, float(np.max(
                evaluation.applied_force_norm_n)))
            min_length = min(min_length, float(np.min(evaluation.edge_lengths_m)))
            max_length = max(max_length, float(np.max(evaluation.edge_lengths_m)))
        residual = float(np.linalg.norm(np.sum(node_forces, axis=0)))
        stats = SpringStepStats(
            energies, max_uncapped, max_applied, capped_count, evaluations,
            residual, min_length, max_length)
        return InternalForceEvaluation(node_forces, stats)

    def apply_node_forces(
        self, node_forces: np.ndarray, positions: np.ndarray,
    ) -> None:
        """Apply one already-aggregated force at each node position."""
        for body, force, position in zip(self.body_ids, node_forces, positions):
            self.client.applyExternalForce(
                objectUniqueId=body, linkIndex=-1, forceObj=force.tolist(),
                posObj=position.tolist(), flags=self.client.WORLD_FRAME)

    def apply_internal_forces(self) -> SpringStepStats:
        """Evaluate every edge, aggregate node forces, and apply once per node."""
        positions, velocities = self.positions(), self.velocities()
        evaluation = self.compute_internal_forces(positions, velocities)
        self.apply_node_forces(evaluation.node_forces_n, positions)
        return evaluation.stats
