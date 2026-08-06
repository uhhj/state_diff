"""Independent orthogonal face-angle elasticity for the soft block."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtMaterial, KelvinVoigtSoftBlock, SpringStepStats,
    material_from_dict)
from state_diff.env.block_pushing.soft_block_lattice import build_angle_triplets


@dataclass(frozen=True)
class AngleElasticParameters:
    stiffness_n_m: float
    damping_n_m_s: float
    force_cap_n: float
    rest_cosine: float = 0.0

    def validate(self) -> None:
        if self.stiffness_n_m <= 0:
            raise ValueError("angle stiffness must be positive")
        if self.damping_n_m_s < 0:
            raise ValueError("angle damping must be non-negative")
        if self.force_cap_n <= 0:
            raise ValueError("angle force cap must be positive")


@dataclass(frozen=True)
class AngleForceEvaluation:
    force_center_n: np.ndarray
    force_first_n: np.ndarray
    force_second_n: np.ndarray
    energy_j: np.ndarray
    cosine_error: np.ndarray
    cosine_rate: np.ndarray
    capped: np.ndarray
    max_uncapped_force_n: np.ndarray
    min_arm_length_m: np.ndarray


def evaluate_angle_constraints(
    positions: np.ndarray, velocities: np.ndarray, triplets: np.ndarray,
    rest_cosines: np.ndarray, stiffness_n_m: float, damping_n_m_s: float,
    force_cap_n: float,
) -> AngleForceEvaluation:
    """Evaluate orthogonal-face angle forces."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    triplets = np.asarray(triplets, dtype=np.int64)
    rest_cosines = np.asarray(rest_cosines, dtype=np.float64)
    center, first, second = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    u = positions[first] - positions[center]
    v = positions[second] - positions[center]
    u_len, v_len = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    if np.any(u_len <= 1e-12) or np.any(v_len <= 1e-12):
        raise FloatingPointError("collapsed angle arm")
    cosine = np.sum(u * v, axis=1) / (u_len * v_len)
    error = cosine - rest_cosines
    grad_u = (v / (u_len * v_len)[:, None]
              - cosine[:, None] * u / (u_len ** 2)[:, None])
    grad_v = (u / (u_len * v_len)[:, None]
              - cosine[:, None] * v / (v_len ** 2)[:, None])
    u_velocity = velocities[first] - velocities[center]
    v_velocity = velocities[second] - velocities[center]
    cosine_rate = (np.sum(grad_u * u_velocity, axis=1)
                   + np.sum(grad_v * v_velocity, axis=1))
    generalized = stiffness_n_m * error + damping_n_m_s * cosine_rate
    force_first = -generalized[:, None] * grad_u
    force_second = -generalized[:, None] * grad_v
    force_center = -(force_first + force_second)
    uncapped_max = np.maximum.reduce([
        np.linalg.norm(force_center, axis=1),
        np.linalg.norm(force_first, axis=1),
        np.linalg.norm(force_second, axis=1)])
    scale = np.minimum(1.0, force_cap_n / np.maximum(uncapped_max, 1e-30))
    force_center = force_center * scale[:, None]
    force_first = force_first * scale[:, None]
    force_second = force_second * scale[:, None]
    return AngleForceEvaluation(
        force_center, force_first, force_second,
        .5 * stiffness_n_m * error ** 2, error, cosine_rate,
        uncapped_max > force_cap_n, uncapped_max,
        np.minimum(u_len, v_len))


def load_angle_material_profile(
    path: str,
) -> Tuple[KelvinVoigtMaterial, AngleElasticParameters, dict]:
    """Load and strictly validate one frozen-edge R3 composite profile."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("profile_version") != 3:
        raise ValueError("R3 profile_version must be 3")
    if payload.get("model") != "kelvin_voigt_edges_plus_orthogonal_face_angles":
        raise ValueError("unexpected R3 model")
    expected_edge = {
        "structural_stiffness_n_per_m": 10.0,
        "structural_damping_ns_per_m": 0.02282177322938192,
        "shear_stiffness_n_per_m": 3.75,
        "shear_damping_ns_per_m": 0.013975424859373685,
        "bending_stiffness_n_per_m": 1.0,
        "bending_damping_ns_per_m": 0.007216878364870322}
    for key, expected in expected_edge.items():
        if abs(float(payload["edge_material"][key]) - expected) > 1e-12:
            raise ValueError("R3 frozen edge coefficient mismatch: " + key)
    if abs(float(payload["node_mass_kg"]) - .03 / 72) > 1e-12:
        raise ValueError("R3 node mass mismatch")
    angle = payload["angle"]
    stiffness = float(angle["stiffness_n_m"])
    if stiffness not in (0.0004, 0.00055, 0.0007):
        raise ValueError("angle stiffness is outside the R3 candidate family")
    if angle["planes"] != ["xy", "xz", "yz"]:
        raise ValueError("R3 angle planes mismatch")
    if int(angle["expected_constraint_count"]) != 121:
        raise ValueError("R3 angle constraint count mismatch")
    damping = 2 * float(angle["damping_ratio"]) * np.sqrt(
        stiffness * ((float(payload["node_mass_kg"]) / 2)
                     * float(angle["reference_arm_length_m"]) ** 2))
    if abs(damping - float(angle["damping_n_m_s"])) > 1e-12:
        raise ValueError("R3 angle damping coefficient mismatch")
    parameters = AngleElasticParameters(
        stiffness, float(angle["damping_n_m_s"]),
        float(angle["force_cap_n"]), float(angle["rest_cosine"]))
    parameters.validate()
    return material_from_dict(payload["edge_material"]), parameters, payload


class AngleElasticSoftBlock(KelvinVoigtSoftBlock):
    """Kelvin-Voigt edge block plus face-angle elasticity."""

    def __init__(self, client, config, material, angle_parameters,
                 force_cap_n, center_xy, yaw_deg=0.0, static_indices=None):
        angle_parameters.validate()
        self.angle_parameters = angle_parameters
        super().__init__(client, config, material, force_cap_n, center_xy,
                         yaw_deg, static_indices)
        records = build_angle_triplets(config)
        self.angle_triplets = np.asarray(
            [[row.center, row.first, row.second] for row in records], dtype=np.int64)
        self.angle_planes = np.asarray([row.plane for row in records])
        self.angle_plane_counts = {
            plane: int(np.sum(self.angle_planes == plane))
            for plane in ("xy", "xz", "yz")}
        p = self.initial_positions
        u = p[self.angle_triplets[:, 1]] - p[self.angle_triplets[:, 0]]
        v = p[self.angle_triplets[:, 2]] - p[self.angle_triplets[:, 0]]
        self.rest_cosines = np.sum(u * v, axis=1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        self.internal_evaluations_per_microstep = sum(
            len(pairs) for pairs in self.edges.values()) + len(self.angle_triplets)

    def apply_internal_forces(self) -> SpringStepStats:
        positions, velocities = self.positions(), self.velocities()
        base = self.compute_internal_forces(positions, velocities)
        angle = evaluate_angle_constraints(
            positions, velocities, self.angle_triplets, self.rest_cosines,
            self.angle_parameters.stiffness_n_m,
            self.angle_parameters.damping_n_m_s,
            self.angle_parameters.force_cap_n)
        angle_node_forces = np.zeros_like(base.node_forces_n)
        np.add.at(angle_node_forces, self.angle_triplets[:, 0], angle.force_center_n)
        np.add.at(angle_node_forces, self.angle_triplets[:, 1], angle.force_first_n)
        np.add.at(angle_node_forces, self.angle_triplets[:, 2], angle.force_second_n)
        node_forces = base.node_forces_n + angle_node_forces
        self.apply_node_forces(node_forces, positions)
        stats = base.stats
        energies = dict(stats.energy_by_kind_j)
        energies["angle"] = float(np.sum(angle.energy_j))
        angle_applied = max(float(np.max(np.linalg.norm(force, axis=1)))
                            for force in (angle.force_center_n,
                                          angle.force_first_n,
                                          angle.force_second_n))
        arms = np.concatenate([
            np.linalg.norm(positions[self.angle_triplets[:, 1]]
                           - positions[self.angle_triplets[:, 0]], axis=1),
            np.linalg.norm(positions[self.angle_triplets[:, 2]]
                           - positions[self.angle_triplets[:, 0]], axis=1)])
        return SpringStepStats(
            energies,
            max(stats.max_uncapped_edge_force_n,
                float(np.max(angle.max_uncapped_force_n))),
            max(stats.max_applied_edge_force_n, angle_applied),
            stats.capped_force_count + int(np.sum(angle.capped)),
            stats.force_evaluation_count + len(self.angle_triplets),
            float(np.linalg.norm(np.sum(node_forces, axis=0))),
            min(stats.min_edge_length_m, float(np.min(arms))),
            max(stats.max_edge_length_m, float(np.max(arms))))
