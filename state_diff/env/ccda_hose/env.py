from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

from state_diff.env.ccda_hose.config import (
    FREE_INSERT,
    RIGHT_HIDDEN_JAM,
    HoseEnvConfig,
    SUPPORTED_CONDITIONS,
)
from state_diff.env.ccda_hose.model import make_hose_insert_xml


BRANCH_TO_ID = {
    "success_insert": 0,
    "lateral_jam": 1,
    "s_buckle": 2,
    "half_insert_wrong_angle": 3,
    "failed_insert": 4,
}
ID_TO_BRANCH = {v: k for k, v in BRANCH_TO_ID.items()}


class HiddenJamHoseInsertionEnv:
    """Standalone MuJoCo environment for stage-1 CCDA auditing.

    This is deliberately not a Gym wrapper yet. It is a low-level deterministic
    simulator used to validate whether the hidden lateral jam can create
    contact-conditioned future-state branching.
    """

    def __init__(
        self,
        config: Optional[HoseEnvConfig] = None,
        condition: str = FREE_INSERT,
        seed: int = 0,
        transparent_socket: bool = False,
        show_occluder: bool = True,
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise ImportError(
                "Failed to import mujoco. Install the official mujoco Python package "
                "or activate the coord_bimanual environment."
            ) from _MUJOCO_IMPORT_ERROR
        self.cfg = config or HoseEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.condition = condition
        self.transparent_socket = transparent_socket
        self.show_occluder = show_occluder
        self.model = None
        self.data = None
        self._renderer = None
        self._renderer_size = None
        self._name_cache: Dict[str, Dict[str, int]] = {}
        self.last_action = np.zeros(3, dtype=np.float64)
        self.reset(condition=condition, seed=seed)

    def reset(self, condition: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if condition is not None:
            if condition not in SUPPORTED_CONDITIONS:
                raise ValueError(f"Unsupported condition {condition!r}. Expected {SUPPORTED_CONDITIONS}.")
            self.condition = condition

        xml = make_hose_insert_xml(
            self.cfg,
            self.condition,
            transparent_socket=self.transparent_socket,
            show_occluder=self.show_occluder,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._renderer_size = None
        self._name_cache = {"body": {}, "geom": {}, "camera": {}}

        start = np.array([self.cfg.start_x, 0.0, self.cfg.socket_center_z], dtype=np.float64)
        start[0] += float(self.rng.normal(0.0, self.cfg.pos_noise))
        start[1] += float(self.rng.normal(0.0, self.cfg.y_noise))
        start[2] += float(self.rng.normal(0.0, self.cfg.z_noise))
        self.data.mocap_pos[0] = start
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(3, dtype=np.float64)
        for _ in range(self.cfg.settle_steps):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation()

    def close(self) -> None:
        self._renderer = None
        self._renderer_size = None

    def _body_id(self, name: str) -> int:
        if name not in self._name_cache["body"]:
            self._name_cache["body"][name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._name_cache["body"][name]

    def _geom_id(self, name: str) -> int:
        if name not in self._name_cache["geom"]:
            self._name_cache["geom"][name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self._name_cache["geom"][name]

    def step(self, action_delta: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if action_delta is None:
            action_delta = np.zeros(3, dtype=np.float64)
        action_delta = np.asarray(action_delta, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(action_delta))
        if norm > self.cfg.max_delta_per_env_step:
            action_delta = action_delta / norm * self.cfg.max_delta_per_env_step

        self.data.mocap_pos[0] = self.data.mocap_pos[0] + action_delta
        self.last_action = action_delta.copy()
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self.get_observation()

    def move_mocap_towards(self, target_pos: np.ndarray) -> Dict[str, np.ndarray]:
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        delta = target_pos - self.data.mocap_pos[0]
        return self.step(delta)

    def get_plug_pos(self) -> np.ndarray:
        return self.data.xpos[self._body_id("plug")].copy()

    def get_gripper_pose(self) -> np.ndarray:
        return np.concatenate([self.data.mocap_pos[0].copy(), self.data.mocap_quat[0].copy()])

    def get_hose_keypoints(self) -> np.ndarray:
        pts = [self.get_plug_pos()]
        for i in range(self.cfg.n_segments):
            pts.append(self.data.xpos[self._body_id(f"hose_seg_{i}")].copy())
        return np.asarray(pts, dtype=np.float64)

    def get_visible_keypoints(self) -> np.ndarray:
        pts = self.get_hose_keypoints()
        n = min(self.cfg.visible_segments + 1, len(pts))
        return pts[:n].copy()

    def insertion_depth(self) -> float:
        plug_tip_x = self.get_plug_pos()[0] + self.cfg.plug_length / 2.0
        return float(max(0.0, plug_tip_x - self.cfg.socket_entrance_x))

    def lateral_offset(self) -> float:
        return float(self.get_plug_pos()[1])

    def max_curvature(self) -> float:
        pts = self.get_hose_keypoints()
        if len(pts) < 3:
            return 0.0
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:] - pts[1:-1]
        n1 = np.linalg.norm(v1, axis=-1) + 1e-8
        n2 = np.linalg.norm(v2, axis=-1) + 1e-8
        cosang = np.sum(v1 * v2, axis=-1) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angles = np.arccos(cosang)
        return float(np.max(angles / max(self.cfg.segment_length, 1e-6)))

    def contact_summary(self) -> Dict[str, float]:
        total_force = 0.0
        jam_force = 0.0
        lateral_force = 0.0
        socket_force = 0.0
        n_jam_contacts = 0

        try:
            jam_geom_id = self._geom_id("hidden_jam_block")
        except Exception:
            jam_geom_id = -1

        socket_geom_ids = set()
        for name in [
            "socket_wall_right",
            "socket_wall_left",
            "socket_wall_top",
            "socket_wall_bottom",
        ]:
            try:
                socket_geom_ids.add(self._geom_id(name))
            except Exception:
                pass

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force6)
            force_mag = float(np.linalg.norm(force6[:3]))
            total_force += force_mag

            normal_world = np.asarray(contact.frame[:3], dtype=np.float64)
            world_force = normal_world * force6[0]
            lateral_force += float(abs(world_force[1]))

            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if jam_geom_id >= 0 and (g1 == jam_geom_id or g2 == jam_geom_id):
                jam_force += force_mag
                n_jam_contacts += 1
            if g1 in socket_geom_ids or g2 in socket_geom_ids:
                socket_force += force_mag

        return {
            "total_contact_force": total_force,
            "jam_contact_force": jam_force,
            "lateral_contact_force": lateral_force,
            "socket_contact_force": socket_force,
            "n_contacts": float(self.data.ncon),
            "n_jam_contacts": float(n_jam_contacts),
            "jam_active": float(jam_force > self.cfg.jam_force_threshold),
        }

    def label_branch(self) -> Dict[str, object]:
        depth = self.insertion_depth()
        lat = abs(self.lateral_offset())
        curv = self.max_curvature()
        contacts = self.contact_summary()
        jam_force = contacts["jam_contact_force"]
        lateral_force = contacts["lateral_contact_force"]

        success = (
            depth >= self.cfg.success_insert_depth
            and lat <= self.cfg.lateral_offset_threshold
            and curv <= self.cfg.max_curvature_success_threshold
            and jam_force <= self.cfg.jam_force_threshold
            and lateral_force <= 1.5 * self.cfg.jam_force_threshold
        )

        if success:
            branch = "success_insert"
        elif jam_force > self.cfg.jam_force_threshold or lateral_force > 1.5 * self.cfg.jam_force_threshold:
            branch = "lateral_jam"
        elif curv > self.cfg.max_curvature_buckle_threshold:
            branch = "s_buckle"
        elif depth > self.cfg.half_insert_depth:
            branch = "half_insert_wrong_angle"
        else:
            branch = "failed_insert"

        return {
            "success": bool(success),
            "branch": branch,
            "branch_id": int(BRANCH_TO_ID[branch]),
        }

    def get_observation(self) -> Dict[str, np.ndarray]:
        visible = self.get_visible_keypoints()
        plug = self.get_plug_pos()
        gripper = self.get_gripper_pose()
        socket = np.array([self.cfg.socket_entrance_x, 0.0, self.cfg.socket_center_z], dtype=np.float64)
        contact = self.contact_summary()
        label = self.label_branch()

        visible_state = np.concatenate(
            [
                plug,
                visible.reshape(-1),
                socket,
                gripper,
                self.last_action,
            ]
        ).astype(np.float64)

        privileged_contact = np.array(
            [
                float(self.condition == RIGHT_HIDDEN_JAM),
                contact["total_contact_force"],
                contact["jam_contact_force"],
                contact["lateral_contact_force"],
                contact["socket_contact_force"],
                contact["n_contacts"],
                contact["n_jam_contacts"],
                contact["jam_active"],
            ],
            dtype=np.float64,
        )

        return {
            "visible_state": visible_state,
            "proprio": gripper.astype(np.float64),
            "action": self.last_action.astype(np.float64),
            "hose_keypoints": self.get_hose_keypoints().astype(np.float64),
            "plug_pos": plug.astype(np.float64),
            "insertion_depth": np.array([self.insertion_depth()], dtype=np.float64),
            "lateral_offset": np.array([self.lateral_offset()], dtype=np.float64),
            "max_curvature": np.array([self.max_curvature()], dtype=np.float64),
            "privileged_contact": privileged_contact,
            "success": np.array([float(label["success"])], dtype=np.float64),
            "branch_id": np.array([float(label["branch_id"])], dtype=np.float64),
        }

    def render(self, camera_name: str = "front", width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        width = int(width or self.cfg.render_width)
        height = int(height or self.cfg.render_height)
        if self._renderer is None or self._renderer_size != (width, height):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (width, height)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()


def _append_trace(trace: Dict[str, List[np.ndarray]], obs: Dict[str, np.ndarray]) -> None:
    for k, v in obs.items():
        trace.setdefault(k, []).append(np.asarray(v).copy())


def _stack_trace(trace: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack(v, axis=0) for k, v in trace.items()}


def scripted_rollout(
    condition: str = FREE_INSERT,
    seed: int = 0,
    config: Optional[HoseEnvConfig] = None,
    record_frames: bool = False,
    camera_name: str = "front",
    transparent_socket: bool = False,
    show_occluder: bool = True,
) -> Dict[str, object]:
    """Run a deterministic approach-then-push rollout.

    Returns arrays for stage-1 auditing and optional rendered frames.
    """
    cfg = config or HoseEnvConfig()
    env = HiddenJamHoseInsertionEnv(
        config=cfg,
        condition=condition,
        seed=seed,
        transparent_socket=transparent_socket,
        show_occluder=show_occluder,
    )
    trace: Dict[str, List[np.ndarray]] = {}
    frames: List[np.ndarray] = []

    def record(obs: Dict[str, np.ndarray]) -> None:
        _append_trace(trace, obs)
        if record_frames:
            frames.append(env.render(camera_name=camera_name).copy())

    obs = env.get_observation()
    record(obs)

    target = np.array([cfg.pre_insert_x, 0.0, cfg.socket_center_z], dtype=np.float64)
    for _ in range(cfg.approach_steps):
        obs = env.move_mocap_towards(target)
        record(obs)
    audit_index = len(trace["visible_state"]) - 1

    total_push = cfg.push_distance
    per_step = total_push / max(1, cfg.push_steps)
    push_delta = np.array([per_step, 0.0, 0.0], dtype=np.float64)
    for _ in range(cfg.push_steps):
        obs = env.step(push_delta)
        record(obs)

    stacked = _stack_trace(trace)
    final_label = env.label_branch()
    env.close()

    return {
        "condition": condition,
        "seed": seed,
        "config": asdict(cfg),
        "audit_index": int(audit_index),
        "trace": stacked,
        "final_success": bool(final_label["success"]),
        "final_branch": str(final_label["branch"]),
        "frames": frames,
    }


def summarize_rollouts(rollouts: List[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    for r in rollouts:
        tr = r["trace"]
        rows.append(
            {
                "condition": r["condition"],
                "seed": int(r["seed"]),
                "success": bool(r["final_success"]),
                "branch": r["final_branch"],
                "final_insert_depth": float(tr["insertion_depth"][-1, 0]),
                "final_lateral_offset": float(tr["lateral_offset"][-1, 0]),
                "final_max_curvature": float(tr["max_curvature"][-1, 0]),
                "max_jam_contact_force": float(np.max(tr["privileged_contact"][:, 2])),
                "max_lateral_contact_force": float(np.max(tr["privileged_contact"][:, 3])),
            }
        )

    by_cond = {}
    for cond in SUPPORTED_CONDITIONS:
        cond_rows = [x for x in rows if x["condition"] == cond]
        if not cond_rows:
            continue
        by_cond[cond] = {
            "n": len(cond_rows),
            "success_rate": float(np.mean([x["success"] for x in cond_rows])),
            "mean_final_insert_depth": float(np.mean([x["final_insert_depth"] for x in cond_rows])),
            "mean_final_lateral_offset": float(np.mean([x["final_lateral_offset"] for x in cond_rows])),
            "mean_final_max_curvature": float(np.mean([x["final_max_curvature"] for x in cond_rows])),
            "mean_max_jam_contact_force": float(np.mean([x["max_jam_contact_force"] for x in cond_rows])),
            "mean_max_lateral_contact_force": float(np.mean([x["max_lateral_contact_force"] for x in cond_rows])),
            "branch_counts": {
                b: int(sum(1 for x in cond_rows if x["branch"] == b))
                for b in sorted(BRANCH_TO_ID)
            },
        }

    return {
        "rows": rows,
        "by_condition": by_cond,
    }
