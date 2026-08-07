"""Deterministic node-and-constraint representation of a small soft block."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SoftBlockConfig:
    """Physical and topological parameters for the fixed soft lattice."""

    nx: int = 6
    ny: int = 4
    nz: int = 3
    node_radius_m: float = 0.005
    spacing_xyz_m: Tuple[float, float, float] = (0.0105, 0.0105, 0.0105)
    total_mass_kg: float = 0.030
    node_lateral_friction: float = 0.80
    node_rolling_friction: float = 0.0001
    linear_damping: float = 0.04
    angular_damping: float = 0.04
    structural_max_force_n: float = 0.80
    shear_max_force_n: float = 0.35
    bending_max_force_n: float = 0.15

    def validate(self) -> None:
        """Validate values that affect topology or simulation semantics."""
        if min(self.nx, self.ny, self.nz) < 2:
            raise ValueError("all grid dimensions must be at least two")
        positive = (
            self.node_radius_m, *self.spacing_xyz_m, self.total_mass_kg,
            self.structural_max_force_n, self.shear_max_force_n,
            self.bending_max_force_n)
        if min(positive) <= 0:
            raise ValueError("radius, spacing, mass, and forces must be positive")
        if 2 * self.node_radius_m > min(self.spacing_xyz_m) * 1.05:
            raise ValueError("node diameter is incompatible with lattice spacing")

    @property
    def num_nodes(self) -> int:
        """Return the deterministic node count."""
        return self.nx * self.ny * self.nz


def node_index(i: int, j: int, k: int, config: SoftBlockConfig) -> int:
    """Map integer grid coordinates to the fixed flat node index."""
    return (k * config.ny + j) * config.nx + i


def initial_node_positions(
    config: SoftBlockConfig,
    center_xy: Tuple[float, float],
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Return deterministic world positions in k/j/i iteration order."""
    config.validate()
    sx, sy, sz = config.spacing_xyz_m
    local = np.zeros((config.num_nodes, 3), dtype=np.float64)
    for k in range(config.nz):
        for j in range(config.ny):
            for i in range(config.nx):
                index = node_index(i, j, k, config)
                local[index] = [
                    (i - (config.nx - 1) / 2.0) * sx,
                    (j - (config.ny - 1) / 2.0) * sy,
                    config.node_radius_m + k * sz,
                ]
    angle = np.deg2rad(float(yaw_deg))
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]], dtype=np.float64)
    local[:, :2] = local[:, :2].dot(rotation.T)
    local[:, :2] += np.asarray(center_xy, dtype=np.float64)
    return local


def build_edge_pairs(config: SoftBlockConfig) -> Dict[str, np.ndarray]:
    """Return sorted duplicate-free [E,2] pairs for each edge family."""
    config.validate()
    pairs = {"structural": set(), "shear": set(), "bending": set()}

    def add(kind: str, first: Tuple[int, int, int], second: Tuple[int, int, int]):
        a = node_index(*first, config)
        b = node_index(*second, config)
        pairs[kind].add(tuple(sorted((a, b))))

    for k in range(config.nz):
        for j in range(config.ny):
            for i in range(config.nx):
                if i + 1 < config.nx:
                    add("structural", (i, j, k), (i + 1, j, k))
                if j + 1 < config.ny:
                    add("structural", (i, j, k), (i, j + 1, k))
                if k + 1 < config.nz:
                    add("structural", (i, j, k), (i, j, k + 1))
                if i + 2 < config.nx:
                    add("bending", (i, j, k), (i + 2, j, k))
                if j + 2 < config.ny:
                    add("bending", (i, j, k), (i, j + 2, k))
                if k + 2 < config.nz:
                    add("bending", (i, j, k), (i, j, k + 2))
    for k in range(config.nz):
        for j in range(config.ny - 1):
            for i in range(config.nx - 1):
                add("shear", (i, j, k), (i + 1, j + 1, k))
                add("shear", (i + 1, j, k), (i, j + 1, k))
    for k in range(config.nz - 1):
        for j in range(config.ny):
            for i in range(config.nx - 1):
                add("shear", (i, j, k), (i + 1, j, k + 1))
                add("shear", (i + 1, j, k), (i, j, k + 1))
    for k in range(config.nz - 1):
        for j in range(config.ny - 1):
            for i in range(config.nx):
                add("shear", (i, j, k), (i, j + 1, k + 1))
                add("shear", (i, j + 1, k), (i, j, k + 1))
    return {kind: np.asarray(sorted(pairs[kind]), dtype=np.int64)
            for kind in ("structural", "shear", "bending")}


def build_edge_metadata(config: SoftBlockConfig) -> Dict[str, List[dict]]:
    """Build legacy edge metadata while preserving order and semantics."""
    positions = initial_node_positions(config, (0.0, 0.0))
    pairs = build_edge_pairs(config)
    forces = {
        "structural": config.structural_max_force_n,
        "shear": config.shear_max_force_n,
        "bending": config.bending_max_force_n}
    result = {}
    for kind in ("structural", "shear", "bending"):
        result[kind] = [
            {"a": int(a), "b": int(b), "kind": kind,
             "rest_length": float(np.linalg.norm(positions[a] - positions[b])),
             "max_force": float(forces[kind])}
            for a, b in pairs[kind]]
    return result


class SoftBlockLattice:
    """PyBullet bodies and midpoint constraints for the fixed soft block."""

    def __init__(self, client, config: SoftBlockConfig,
                 center_xy: Tuple[float, float], yaw_deg: float = 0.0):
        config.validate()
        self.client = client
        self.config = config
        self.initial_positions = initial_node_positions(config, center_xy, yaw_deg)
        self.edge_metadata = build_edge_metadata(config)
        self.edges: Dict[str, np.ndarray] = {
            kind: np.asarray([[edge["a"], edge["b"]] for edge in records],
                             dtype=np.int64)
            for kind, records in self.edge_metadata.items()}
        self.top_indices = np.asarray([
            node_index(i, j, config.nz - 1, config)
            for j in range(config.ny) for i in range(config.nx)], dtype=np.int64)
        self.bottom_indices = np.asarray([
            node_index(i, j, 0, config)
            for j in range(config.ny) for i in range(config.nx)], dtype=np.int64)
        self.body_ids: List[int] = []
        self.constraint_ids: List[int] = []
        self.constraint_ids_by_kind: Dict[str, List[int]] = {
            "structural": [], "shear": [], "bending": []}
        self._create()

    def _create(self) -> None:
        collision = self.client.createCollisionShape(
            self.client.GEOM_SPHERE, radius=self.config.node_radius_m)
        visual = self.client.createVisualShape(
            self.client.GEOM_SPHERE, radius=self.config.node_radius_m,
            rgbaColor=[0.92, 0.24, 0.28, 1.0])
        mass = self.config.total_mass_kg / self.config.num_nodes
        for position in self.initial_positions:
            body_id = int(self.client.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual, basePosition=position.tolist()))
            self.client.changeDynamics(
                body_id, -1,
                lateralFriction=self.config.node_lateral_friction,
                rollingFriction=self.config.node_rolling_friction,
                restitution=0.0, linearDamping=self.config.linear_damping,
                angularDamping=self.config.angular_damping)
            self.body_ids.append(body_id)
        identity = [0.0, 0.0, 0.0, 1.0]
        for kind in ("structural", "shear", "bending"):
            for edge in self.edge_metadata[kind]:
                a, b = edge["a"], edge["b"]
                world_a, quat_a = self.client.getBasePositionAndOrientation(
                    self.body_ids[a])
                world_b, quat_b = self.client.getBasePositionAndOrientation(
                    self.body_ids[b])
                midpoint = 0.5 * (
                    np.asarray(world_a) + np.asarray(world_b))
                inv_a = self.client.invertTransform(world_a, quat_a)
                inv_b = self.client.invertTransform(world_b, quat_b)
                parent_local, _ = self.client.multiplyTransforms(
                    inv_a[0], inv_a[1], midpoint.tolist(), identity)
                child_local, _ = self.client.multiplyTransforms(
                    inv_b[0], inv_b[1], midpoint.tolist(), identity)
                constraint_id = int(self.client.createConstraint(
                    parentBodyUniqueId=self.body_ids[a], parentLinkIndex=-1,
                    childBodyUniqueId=self.body_ids[b], childLinkIndex=-1,
                    jointType=self.client.JOINT_POINT2POINT,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=parent_local,
                    childFramePosition=child_local))
                self.client.changeConstraint(
                    constraint_id, maxForce=edge["max_force"])
                self.constraint_ids.append(constraint_id)
                self.constraint_ids_by_kind[kind].append(constraint_id)
                if kind == "structural":
                    self.client.setCollisionFilterPair(
                        self.body_ids[a], self.body_ids[b], -1, -1,
                        enableCollision=0)

    def positions(self) -> np.ndarray:
        """Return [N, 3] node positions."""
        return np.asarray([self.client.getBasePositionAndOrientation(body)[0]
                           for body in self.body_ids], dtype=np.float64)

    def velocities(self) -> np.ndarray:
        """Return [N, 3] node linear velocities."""
        return np.asarray([self.client.getBaseVelocity(body)[0]
                           for body in self.body_ids], dtype=np.float64)

    def orientations(self) -> np.ndarray:
        """Return [N, 4] node base quaternions."""
        return np.asarray([self.client.getBasePositionAndOrientation(body)[1]
                           for body in self.body_ids], dtype=np.float64)

    def edge_lengths(self, kind: str) -> np.ndarray:
        """Return current lengths for one edge family."""
        if kind not in self.edges:
            raise ValueError("unknown edge kind: {}".format(kind))
        positions = self.positions()
        pairs = self.edges[kind]
        return np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)

    def structural_edge_ratios(self) -> np.ndarray:
        """Return current/rest length ratios for structural edges."""
        rest = np.asarray([edge["rest_length"]
                           for edge in self.edge_metadata["structural"]])
        return self.edge_lengths("structural") / rest

    def center_of_mass(self) -> np.ndarray:
        """Return the equal-mass node center of mass."""
        return np.mean(self.positions(), axis=0)

    def recenter_xy(self, center_xy: Tuple[float, float]) -> np.ndarray:
        """Rigidly recenter a settled lattice in XY and clear residual velocity."""
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

    def visible_positions(self) -> np.ndarray:
        """Return top-layer keypoints in fixed order."""
        return self.positions()[self.top_indices]
