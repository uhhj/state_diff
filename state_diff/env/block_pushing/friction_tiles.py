"""Invisible, coplanar floor tiles with one switchable friction patch."""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


CONDITIONS = ("uniform_low", "right_local_high")


@dataclass(frozen=True)
class FrictionFloorConfig:
    """Geometry and dynamics for the five-tile floor."""

    x_bounds: Tuple[float, float] = (0.15, 0.70)
    y_bounds: Tuple[float, float] = (-0.50, 0.50)
    top_z_m: float = 0.0
    thickness_m: float = 0.010
    outer_lateral_friction: float = 0.20
    patch_free_lateral_friction: float = 0.20
    patch_high_lateral_friction: float = 1.60
    spinning_friction: float = 0.0001
    rolling_friction: float = 0.0
    patch_center_xy: Tuple[float, float] = (0.418, -0.142)
    patch_size_xy: Tuple[float, float] = (0.023, 0.026)

    def validate(self) -> None:
        """Validate floor bounds and patch containment."""
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        cx, cy = self.patch_center_xy
        sx, sy = self.patch_size_xy
        if not xmin < xmax or not ymin < ymax or self.thickness_m <= 0:
            raise ValueError("invalid floor bounds or thickness")
        if min(sx, sy) <= 0:
            raise ValueError("patch size must be positive")
        if not (xmin < cx - sx / 2 < cx + sx / 2 < xmax
                and ymin < cy - sy / 2 < cy + sy / 2 < ymax):
            raise ValueError("patch must be strictly inside floor bounds")
        if min(self.outer_lateral_friction,
               self.patch_free_lateral_friction,
               self.patch_high_lateral_friction) < 0:
            raise ValueError("friction must be non-negative")


def tile_bounds(config: FrictionFloorConfig) -> Dict[str, Tuple[float, ...]]:
    """Return the exact non-overlapping five-rectangle partition."""
    config.validate()
    xmin, xmax = config.x_bounds
    ymin, ymax = config.y_bounds
    cx, cy = config.patch_center_xy
    sx, sy = config.patch_size_xy
    px0, px1 = cx - sx / 2, cx + sx / 2
    py0, py1 = cy - sy / 2, cy + sy / 2
    return {
        "left": (xmin, px0, ymin, ymax),
        "right": (px1, xmax, ymin, ymax),
        "bottom": (px0, px1, ymin, py0),
        "top": (px0, px1, py1, ymax),
        "patch": (px0, px1, py0, py1),
    }


class FrictionTileFloor:
    """Five static collision-only boxes with a switchable center patch."""

    def __init__(self, client, config: FrictionFloorConfig):
        config.validate()
        self.client = client
        self.config = config
        self.bounds = tile_bounds(config)
        self.body_ids: Dict[str, int] = {}
        for name in ("left", "right", "bottom", "top", "patch"):
            x0, x1, y0, y1 = self.bounds[name]
            half = [(x1 - x0) / 2, (y1 - y0) / 2,
                    config.thickness_m / 2]
            center = [(x0 + x1) / 2, (y0 + y1) / 2,
                      config.top_z_m - config.thickness_m / 2]
            collision = client.createCollisionShape(
                client.GEOM_BOX, halfExtents=half)
            body_id = int(client.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=-1, basePosition=center))
            mu = (config.patch_free_lateral_friction if name == "patch"
                  else config.outer_lateral_friction)
            client.changeDynamics(
                body_id, -1, lateralFriction=mu,
                spinningFriction=config.spinning_friction,
                rollingFriction=config.rolling_friction, restitution=0.0)
            self.body_ids[name] = body_id
        self.patch_body_id = self.body_ids["patch"]
        self.condition = "uniform_low"

    def set_condition(self, condition: str) -> None:
        """Change only the patch friction for one counterfactual branch."""
        if condition == "uniform_low":
            mu = self.config.patch_free_lateral_friction
        elif condition == "right_local_high":
            mu = self.config.patch_high_lateral_friction
        else:
            raise ValueError("unknown condition: {}".format(condition))
        self.client.changeDynamics(
            self.patch_body_id, -1, lateralFriction=mu,
            spinningFriction=self.config.spinning_friction,
            rollingFriction=self.config.rolling_friction, restitution=0.0)
        self.condition = condition

    def poses(self) -> Dict[str, np.ndarray]:
        """Return fixed body poses for pairing checks."""
        return {name: np.asarray(
            self.client.getBasePositionAndOrientation(body)[0], dtype=np.float64)
                for name, body in self.body_ids.items()}
