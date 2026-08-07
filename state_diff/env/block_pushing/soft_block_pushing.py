"""Sibling Gym environment for hidden-local-friction soft block pushing."""
from __future__ import annotations

import collections
from typing import Any, Dict, Optional

import gym
from gym import spaces
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
from scipy.spatial import transform

from state_diff.env.block_pushing import block_pushing
from state_diff.env.block_pushing.friction_tiles import (
    CONDITIONS, FrictionTileFloor)
from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtSoftBlock, load_material_profile)
from state_diff.env.block_pushing.manual_microstep_integrator import (
    ManualMicrostepConfig, ManualMicrostepIntegrator,
    OuterStepMechanicsStats)
from state_diff.env.block_pushing.utils import utils_pybullet, xarm_sim_robot
from state_diff.env.block_pushing.utils.pose3d import Pose3d
from state_diff.env.block_pushing.soft_block_task_config import (
    floor_config, soft_block_config, validate_hlf_sbp_config)


STATEDIFF_STATE_LAYOUT = {
    "deformable_keypoints_xyz": (0, 72),
    "ee_xy": (72, 74),
}

CONTACT_SENSOR_LAYOUT = {
    "joint_motor_torque": (0, 6),
    "joint_reaction_wrench": (6, 42),
    "ee_tracking_error_xyz": (42, 45),
}


class SoftBlockPushEnv(gym.Env):
    """Deterministic 72-node soft block and invisible friction-patch task."""

    CONDITIONS = CONDITIONS

    def __init__(self, config: Dict[str, Any], shared_memory: bool = False):
        validate_hlf_sbp_config(config)
        self.config = config
        self._connection_mode = (
            pybullet.SHARED_MEMORY if shared_memory else pybullet.DIRECT)
        self._pybullet_client = bullet_client.BulletClient(self._connection_mode)
        self._rng = np.random.RandomState(int(config["seed"]))
        self._trace = []
        self._physics_step = 0
        self._phase = "setup"
        self._ee_target_position = np.array([
            *config["robot"]["start_xy"], config["robot"]["effector_height"]],
            dtype=np.float64)
        self._saved_state_id: Optional[int] = None
        self.rendered_img = None
        self._setup_scene()
        self.observation_space = self._create_observation_space()
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

    @property
    def pybullet_client(self):
        """Return the environment-owned Bullet client."""
        return self._pybullet_client

    @property
    def robot(self):
        """Return the XArm wrapper."""
        return self._robot

    @property
    def saved_state_id(self) -> int:
        """Return the common low-friction snapshot id."""
        if self._saved_state_id is None:
            raise RuntimeError("common state has not been saved")
        return self._saved_state_id

    @property
    def physics_step(self) -> int:
        """Return the number of completed outer 240 Hz intervals."""
        return self._physics_step

    def _setup_scene(self) -> None:
        client = self._pybullet_client
        physics = self.config["physics"]
        client.resetSimulation()
        client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        client.setPhysicsEngineParameter(enableFileCaching=0)
        try:
            client.setPhysicsEngineParameter(
                numSolverIterations=int(physics["solver_iterations"]),
                deterministicOverlappingPairs=int(
                    physics["deterministic_overlapping_pairs"]),
                enableConeFriction=int(physics["enable_cone_friction"]))
        except TypeError as exc:
            raise RuntimeError(
                "PyBullet lacks required deterministic/friction parameters") from exc
        self._microstep_config = ManualMicrostepConfig(
            outer_timestep_s=float(physics["outer_timestep_s"]),
            microsteps_per_outer=int(physics["microsteps_per_outer"]))
        self._integrator = ManualMicrostepIntegrator(
            client, self._microstep_config)
        client.setGravity(*[float(v) for v in physics["gravity"]])
        self._workspace_uid = utils_pybullet.load_urdf(
            client, block_pushing.WORKSPACE_URDF_PATH,
            basePosition=[0.35, 0.0, 0.0], useFixedBase=True)
        client.setCollisionFilterGroupMask(self._workspace_uid, -1, 0, 0)
        self.floor = FrictionTileFloor(client, floor_config(self.config))
        robot_cfg = self.config["robot"]
        self._robot = xarm_sim_robot.XArmSimRobot(
            client,
            initial_joint_positions=block_pushing.INITIAL_JOINT_POSITIONS,
            end_effector=robot_cfg["effector_type"],
            motor_force_limit=robot_cfg["motor_force_limit"],
            position_gain=robot_cfg["position_gain"],
            velocity_gain=robot_cfg["velocity_gain"])
        if len(self._robot.joint_indices) != 6:
            raise RuntimeError("Phase 0C sensor layout requires exactly six joints")
        goal = np.asarray(self.config["goal"]["center_xy"], dtype=np.float64)
        self._target_id = utils_pybullet.load_urdf(
            client, block_pushing.ZONE_URDF_PATH, useFixedBase=True,
            basePosition=[float(goal[0]), float(goal[1]), 0.0001])
        self._target_pose = Pose3d(
            rotation=transform.Rotation.identity(),
            translation=np.array([goal[0], goal[1], 0.0001]))
        soft_cfg = self.config["soft_block"]
        material_model = soft_cfg.get("material_model")
        self.material_model = material_model
        if material_model != "kelvin_voigt":
            raise ValueError("active HLF-SBP requires kelvin_voigt")
        material, self.material_profile = load_material_profile(
            soft_cfg["material_profile_path"])
        self.soft_block = KelvinVoigtSoftBlock(
            client, soft_block_config(self.config), material,
            force_cap_n=float(soft_cfg["spring_force_cap_n"]),
            center_xy=tuple(soft_cfg["center_xy"]),
            yaw_deg=float(soft_cfg["yaw_deg"]))
        self._robot.enable_joint_force_torque_sensors()
        start_pose = Pose3d(
            rotation=block_pushing.EFFECTOR_DOWN_ROTATION,
            translation=self._ee_target_position.copy())
        rest = self._robot.get_joint_positions()
        start_joints = self._robot.inverse_kinematics_fixed_rest(
            start_pose, rest_joint_positions=rest)
        self._robot.reset_joints(start_joints)
        actual_pose = self._robot.forward_kinematics()
        client.resetBasePositionAndOrientation(
            self._robot.end_effector,
            actual_pose.translation.tolist(), actual_pose.rotation.as_quat().tolist())
        self._robot.set_target_joint_positions(start_joints)
        self._start_joint_positions = np.asarray(start_joints, dtype=np.float64)
        self._target_effector_pose = start_pose
        self._last_mechanics_stats: Optional[OuterStepMechanicsStats] = None
        for _ in range(int(self.config["execution"][
                "pre_snapshot_settle_outer_steps"])):
            self._advance_physics(record_trace=False)
        self.settle_recenter_xy_offset = self.soft_block.recenter_xy(
            tuple(soft_cfg["center_xy"]))
        self.floor.set_condition("uniform_low")
        self.initial_pusher_node_signed_distance = self._minimum_pusher_node_distance()
        self.clear_trace()
        self._saved_state_id = int(client.saveState())

    def _create_observation_space(self):
        box = lambda shape: spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)
        return spaces.Dict(collections.OrderedDict(
            deformable_keypoints=box((24, 3)), joint_position=box((6,)),
            joint_velocity=box((6,)), ee_position=box((3,)),
            ee_velocity=box((3,)), ee_target_position=box((3,)),
            ee_tracking_error=box((3,)), goal_xy=box((2,))))

    def _ee_state(self):
        return self._pybullet_client.getLinkState(
            self._robot.xarm, self._robot.effector_link,
            computeLinkVelocity=1, computeForwardKinematics=True)

    def _observation(self) -> Dict[str, np.ndarray]:
        joints, velocities, _ = self._robot.get_joints_measured()
        ee_state = self._ee_state()
        ee_position = np.asarray(ee_state[0], dtype=np.float64)
        ee_velocity = np.asarray(ee_state[6], dtype=np.float64)
        return collections.OrderedDict(
            deformable_keypoints=self.soft_block.visible_positions(),
            joint_position=np.asarray(joints, dtype=np.float64),
            joint_velocity=np.asarray(velocities, dtype=np.float64),
            ee_position=ee_position, ee_velocity=ee_velocity,
            ee_target_position=self._ee_target_position.copy(),
            ee_tracking_error=ee_position - self._ee_target_position,
            goal_xy=np.asarray(self.config["goal"]["center_xy"], dtype=np.float64))

    def reset(self):
        """Restore the common low-friction snapshot and return observation."""
        self.restore_saved_state(self.saved_state_id)
        self.floor.set_condition("uniform_low")
        self.clear_trace()
        return self._observation()

    def step(self, action):
        """Apply one convenience XY target update for Gym smoke tests."""
        action = np.asarray(action, dtype=np.float64)
        target = self._ee_target_position.copy()
        target[:2] += action
        target[:2] = np.clip(
            target[:2],
            [self.config["floor"]["x_bounds"][0],
             self.config["floor"]["y_bounds"][0]],
            [self.config["floor"]["x_bounds"][1],
             self.config["floor"]["y_bounds"][1]])
        pose = Pose3d(block_pushing.EFFECTOR_DOWN_ROTATION, target)
        joints = self._robot.inverse_kinematics_fixed_rest(
            pose, self._start_joint_positions)
        self.set_ee_target_position(target)
        self.set_fixed_joint_target(joints)
        self.step_one_physics()
        obs = self._observation()
        distance = np.linalg.norm(
            self.soft_block.center_of_mass()[:2] - obs["goal_xy"])
        return obs, -float(distance), bool(distance <= self.config["goal"]["tolerance_m"]), {}

    def arm_condition(self, condition: str) -> dict:
        """Apply the sole branch intervention without advancing physics."""
        before = self.capture_explicit_state()
        self.floor.set_condition(condition)
        after = self.capture_explicit_state()
        differences = []
        for key in before:
            left, right = np.asarray(before[key]), np.asarray(after[key])
            if left.dtype.kind not in "OUS" and left.size:
                differences.append(float(np.max(np.abs(left - right))))
        mu = (self.config["floor"]["patch_free_lateral_friction"]
              if condition == "uniform_low"
              else self.config["floor"]["patch_high_lateral_friction"])
        return {"condition": condition, "patch_lateral_friction": float(mu),
                "explicit_state_max_abs": max(differences or [0.0])}

    def capture_explicit_state(self) -> dict:
        """Capture all explicit branch-equality state except patch dynamics."""
        node_poses = [self._pybullet_client.getBasePositionAndOrientation(body)
                      for body in self.soft_block.body_ids]
        node_velocities = [self._pybullet_client.getBaseVelocity(body)
                           for body in self.soft_block.body_ids]
        joints = self._robot.get_joints_measured()
        ee = self._ee_state()
        tile_names = ("left", "right", "bottom", "top", "patch")
        return {
            "node_positions": np.asarray([v[0] for v in node_poses]),
            "node_orientations": np.asarray([v[1] for v in node_poses]),
            "node_linear_velocities": np.asarray([v[0] for v in node_velocities]),
            "node_angular_velocities": np.asarray([v[1] for v in node_velocities]),
            "joint_positions": np.asarray(joints[0]),
            "joint_velocities": np.asarray(joints[1]),
            "ee_position": np.asarray(ee[0]),
            "ee_orientation": np.asarray(ee[1]),
            "ee_linear_velocity": np.asarray(ee[6]),
            "ee_angular_velocity": np.asarray(ee[7]),
            "ee_target_position": self._ee_target_position.copy(),
            "target_position": np.asarray(
                self._pybullet_client.getBasePositionAndOrientation(
                    self._target_id)[0]),
            "floor_positions": np.asarray([
                self.floor.poses()[name] for name in tile_names]),
            "soft_constraint_ids": np.asarray(
                self.soft_block.constraint_ids, dtype=np.int64),
            "soft_constraint_count": np.asarray(
                [len(self.soft_block.constraint_ids)], dtype=np.int64),
        }

    def restore_saved_state(self, state_id: int) -> None:
        """Restore a Bullet snapshot without stepping simulation."""
        self._pybullet_client.restoreState(stateId=int(state_id))
        self._ee_target_position = np.array([
            *self.config["robot"]["start_xy"],
            self.config["robot"]["effector_height"]], dtype=np.float64)
        self._target_effector_pose = Pose3d(
            block_pushing.EFFECTOR_DOWN_ROTATION,
            self._ee_target_position.copy())

    def set_phase(self, phase: str) -> None:
        """Set the trace phase label."""
        if phase not in ("no_action", "probe", "post_probe", "test", "post_test"):
            raise ValueError("unknown phase: {}".format(phase))
        self._phase = phase

    def set_ee_target_position(self, target: np.ndarray) -> None:
        """Set the formal high-level target recorded in observations."""
        self._ee_target_position = np.asarray(target, dtype=np.float64).copy()

    def set_fixed_joint_target(self, target: np.ndarray) -> None:
        """Apply one precomputed low-level target without IK."""
        self._robot.set_target_joint_positions(np.asarray(target, dtype=np.float64))

    def step_one_physics(self) -> None:
        """Advance exactly one Bullet step and append one stride-selected sample."""
        self._advance_physics(record_trace=True)

    def _advance_physics(self, record_trace: bool) -> None:
        """Advance one outer interval using force-recomputed microsteps."""
        self._last_mechanics_stats = self._integrator.advance_outer_step(
            self.soft_block)
        self._physics_step += 1
        if (record_trace and self._physics_step
                % int(self.config["execution"][
                    "policy_sample_stride_outer_steps"]) == 0):
            self._trace.append(self.trace_sample())

    def _minimum_pusher_node_distance(self) -> float:
        minimum = 0.05
        for body in self.soft_block.body_ids:
            for point in self._pybullet_client.getClosestPoints(
                    self._robot.end_effector, body, distance=0.05):
                minimum = min(minimum, float(point[8]))
        return float(minimum)

    def _oracle_ee_contact_wrench(self) -> np.ndarray:
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        ee_position = np.asarray(self._ee_state()[0], dtype=np.float64)
        for body in self.soft_block.body_ids:
            for point in self._pybullet_client.getContactPoints(
                    bodyA=self._robot.end_effector, bodyB=body):
                normal = float(point[9]) * np.asarray(point[7], dtype=np.float64)
                lateral_one = float(point[10]) * np.asarray(point[11], dtype=np.float64)
                lateral_two = float(point[12]) * np.asarray(point[13], dtype=np.float64)
                contact_force = normal + lateral_one + lateral_two
                force += contact_force
                torque += np.cross(
                    np.asarray(point[5], dtype=np.float64) - ee_position,
                    contact_force)
        return np.concatenate([force, torque])

    def _patch_oracle(self) -> dict:
        indices = []
        normal_force = 0.0
        tangential_force = 0.0
        speeds = []
        for index in self.soft_block.bottom_indices:
            body = self.soft_block.body_ids[int(index)]
            contacts = self._pybullet_client.getContactPoints(
                bodyA=self.floor.patch_body_id, bodyB=body)
            if contacts:
                indices.append(int(index))
                velocity = self._pybullet_client.getBaseVelocity(body)[0]
                speeds.append(float(np.linalg.norm(np.asarray(velocity)[:2])))
            for point in contacts:
                normal_force += float(point[9])
                tangential_force += float(np.hypot(point[10], point[12]))
        stick_ratio = (float(np.mean(np.asarray(speeds) < 0.001))
                       if speeds else 0.0)
        mu = self._pybullet_client.getDynamicsInfo(
            self.floor.patch_body_id, -1)[1]
        return {
            "oracle_patch_contact_node_indices": indices,
            "oracle_patch_normal_force": float(normal_force),
            "oracle_patch_tangential_force": float(tangential_force),
            "oracle_patch_mean_slip_speed": (
                float(np.mean(speeds)) if speeds else 0.0),
            "oracle_patch_stick_ratio": stick_ratio,
            "oracle_patch_mu": float(mu),
            "oracle_bottom_node_positions": self.soft_block.positions()[
                self.soft_block.bottom_indices].tolist(),
        }

    def trace_sample(self) -> dict:
        """Return one fixed-schema formal+oracle physics sample."""
        oracle = self._patch_oracle()
        if self._last_mechanics_stats is None:
            raise RuntimeError("mechanics has not advanced")
        stats = self._last_mechanics_stats
        row = {
            "outer_step": int(self._physics_step), "phase": self._phase,
            "state_diff_state": self.get_statediff_state().tolist(),
            "contact_sensor": self.get_contact_sensor().tolist(),
            "robot_proprio_extended": self.get_extended_proprio().tolist(),
            "node_positions": self.soft_block.positions().tolist(),
            "goal_distance_m": float(np.linalg.norm(
                self.soft_block.center_of_mass()[:2]
                - np.asarray(self.config["goal"]["center_xy"]))),
            "oracle_ee_contact_wrench":
                self._oracle_ee_contact_wrench().tolist(),
            "oracle_patch_contact_count": len(
                oracle["oracle_patch_contact_node_indices"]),
            "spring_energy_structural_j": float(
                stats.final_energy_by_kind_j["structural"]),
            "spring_energy_shear_j": float(
                stats.final_energy_by_kind_j["shear"]),
            "spring_energy_bending_j": float(
                stats.final_energy_by_kind_j["bending"]),
            "spring_energy_total_j": float(sum(
                stats.final_energy_by_kind_j.values())),
            "spring_max_uncapped_edge_force_n": float(
                stats.max_uncapped_edge_force_n),
            "spring_max_applied_edge_force_n": float(
                stats.max_applied_edge_force_n),
            "spring_capped_force_count": int(
                stats.capped_force_count),
            "spring_force_evaluation_count": int(
                stats.force_evaluation_count),
            "spring_net_internal_force_residual_n": float(
                stats.max_net_internal_force_residual_n),
            "microsteps_per_outer": int(stats.microsteps_per_outer),
        }
        row.update(oracle)
        return row

    def clear_trace(self) -> None:
        """Reset branch-local trace and absolute branch step count."""
        self._trace = []
        self._physics_step = 0
        self._phase = "no_action"

    def trace(self):
        """Return a shallow copy of trace rows."""
        return [dict(row) for row in self._trace]

    def get_statediff_state(self) -> np.ndarray:
        """Return the active 74D diffusion state without sensor channels."""
        obs = self._observation()
        state = np.concatenate([
            obs["deformable_keypoints"].reshape(-1), obs["ee_position"][:2],
        ]).astype(np.float64, copy=False)
        if state.shape != (74,):
            raise RuntimeError("unexpected State Diff state shape: " + str(state.shape))
        return state

    def get_contact_sensor(self) -> np.ndarray:
        """Return only the 45D formal force/torque/tracking sensor."""
        _, _, motor_torque = self._robot.get_joints_measured()
        reaction = self._robot.get_joint_reaction_wrenches().reshape(-1)
        tracking = self._observation()["ee_tracking_error"]
        sensor = np.concatenate([
            np.asarray(motor_torque, dtype=np.float64),
            np.asarray(reaction, dtype=np.float64),
            np.asarray(tracking, dtype=np.float64)])
        if sensor.shape != (45,):
            raise RuntimeError("unexpected contact sensor shape: " + str(sensor.shape))
        return sensor

    def get_extended_proprio(self) -> np.ndarray:
        """Return the 24D proprioception-only ablation input."""
        joints, velocities, _ = self._robot.get_joints_measured()
        obs = self._observation()
        return np.concatenate([
            np.asarray(joints), np.asarray(velocities), obs["ee_position"],
            obs["ee_velocity"], obs["ee_target_position"],
            obs["ee_tracking_error"]]).astype(np.float64)

    def render(self, mode="rgb_array") -> np.ndarray:
        """Render the existing BlockPush camera without exposing the patch."""
        del mode
        height, width = [int(v) for v in self.config.get(
            "camera", {"image_size": [240, 320]})["image_size"]]
        front_position = block_pushing.DEFAULT_CAMERA_POSE
        rotation = self._pybullet_client.getQuaternionFromEuler(
            block_pushing.DEFAULT_CAMERA_ORIENTATION)
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        matrix = np.float32(
            self._pybullet_client.getMatrixFromQuaternion(rotation)).reshape(3, 3)
        lookat = np.asarray(front_position) + (matrix @ lookdir).reshape(-1)
        up = (matrix @ updir).reshape(-1)
        view = self._pybullet_client.computeViewMatrix(front_position, lookat, up)
        focal = block_pushing.CAMERA_INTRINSICS[0]
        fov = 180 * np.arctan((height / 2) / focal) * 2 / np.pi
        projection = self._pybullet_client.computeProjectionMatrixFOV(
            fov, width / height, 0.01, 10.0)
        _, _, color, _, _ = self._pybullet_client.getCameraImage(
            width=width, height=height, viewMatrix=view,
            projectionMatrix=projection,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        return np.asarray(color, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self) -> None:
        """Disconnect the environment-owned Bullet client."""
        if self._pybullet_client is not None:
            self._pybullet_client.disconnect()
            self._pybullet_client = None
