"""Phase3.14b-r2.5.5 robot-proxy provenance and migration audit.

This module is intentionally independent of diffusion training.  It defines the
strict target schema and reconstructs that schema from the legacy raw episode
payload without writing migrated data.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "phase314b_r255"
PHASE_NAME = "Phase3.14b-r2.5.5"
BASE_EVIDENCE_COMMIT = "795905d565faaae5244e6c30dce31b8a0dd98066"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
LEGACY_SCHEMA_VERSION = "ccda_state_v2_position_proprio"
TARGET_SCHEMA_VERSION = "ccda_state_v3_controlled_revolute_ee_tip"
TARGET_ROBOT_PROXY_SCHEMA_VERSION = (
    "ccda_robot_proxy_v3_controlled_revolute_ee_tip"
)
FORMAL_TASK_NAME = "ccda-slack-cable-v2"
FORMAL_CONDITIONS = ("free", "hidden_slack_breakaway_pin_v2")
N_BEADS = 24
CABLE_DIM = N_BEADS * 2
CONTROLLED_JOINT_COUNT = 6
ROBOT_PROXY_V3_DIM = CONTROLLED_JOINT_COUNT * 2 + 3 + 4
STATE_V3_DIM = CABLE_DIM + ROBOT_PROXY_V3_DIM
EE_TIP_LINK = 12
TASK_EE = "suction"
ROBOT_URDF_RELATIVE = "assets/ur5/ur5-suction.urdf"


class RobotProxyAuditError(RuntimeError):
    """Raised when legacy robot state cannot be migrated without guessing."""


@dataclass(frozen=True)
class RobotProxyV3Contract:
    schema_version: str = TARGET_ROBOT_PROXY_SCHEMA_VERSION
    state_schema_version: str = TARGET_SCHEMA_VERSION
    task_name: str = FORMAL_TASK_NAME
    task_ee: str = TASK_EE
    robot_urdf_relative: str = ROBOT_URDF_RELATIVE
    controlled_joint_count: int = CONTROLLED_JOINT_COUNT
    ee_tip_link: int = EE_TIP_LINK
    cable_dim: int = CABLE_DIM
    robot_proxy_dim: int = ROBOT_PROXY_V3_DIM
    state_dim: int = STATE_V3_DIM
    missing_value_policy: str = "reject_record"
    quaternion_policy: str = "unit_norm_and_deterministic_hemisphere"
    model_presence_mask: bool = False

    def validate(self) -> None:
        if self.controlled_joint_count != 6:
            raise ValueError("formal UR5 contract requires six controlled joints")
        if self.robot_proxy_dim != 19:
            raise ValueError("robot-proxy v3 dimension must be 19")
        if self.state_dim != 67:
            raise ValueError("state-v3 dimension must be 67")
        if self.missing_value_policy != "reject_record":
            raise ValueError("formal data must reject missing robot state")
        if self.model_presence_mask:
            raise ValueError(
                "presence masks are not model inputs when invalid records are rejected"
            )


@dataclass(frozen=True)
class LegacyRobotRecord:
    source: str
    body_id: Optional[int]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    stored_ee_position: np.ndarray
    stored_ee_quaternion: np.ndarray


@dataclass(frozen=True)
class KinematicsManifest:
    joint_count: int
    controlled_joint_indices: Tuple[int, ...]
    controlled_joint_names: Tuple[str, ...]
    ee_tip_link: int
    ee_tip_link_name: str
    robot_urdf_relative: str
    robot_urdf_sha256: str

    def validate(self) -> None:
        if self.joint_count <= 0:
            raise ValueError("robot joint count must be positive")
        if len(self.controlled_joint_indices) != CONTROLLED_JOINT_COUNT:
            raise ValueError(
                "controlled revolute-joint count is not six: "
                f"{self.controlled_joint_indices}"
            )
        if len(set(self.controlled_joint_indices)) != CONTROLLED_JOINT_COUNT:
            raise ValueError("controlled joint indices are not unique")
        if tuple(sorted(self.controlled_joint_indices)) != (
            self.controlled_joint_indices
        ):
            raise ValueError("controlled joint indices are not ordered")
        if not 0 <= self.ee_tip_link < self.joint_count:
            raise ValueError("official EE tip link is outside the URDF")


@dataclass(frozen=True)
class MigratedRobotRecord:
    values: np.ndarray
    controlled_joint_positions: np.ndarray
    controlled_joint_velocities: np.ndarray
    ee_position: np.ndarray
    ee_quaternion: np.ndarray
    legacy_stored_ee_position_error: float
    legacy_stored_ee_quaternion_error: float


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("non-finite array cannot be serialized")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("non-finite scalar cannot be serialized")
        return value
    raise TypeError(f"unsupported JSON type: {type(value)!r}")


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    text = json.dumps(
        jsonable(dict(payload)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def write_text_once(path: Path, text: str) -> None:
    target = Path(path)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (int(length),):
        raise RobotProxyAuditError(
            f"{name} shape {array.shape} != ({int(length)},)"
        )
    if not np.all(np.isfinite(array)):
        raise RobotProxyAuditError(f"{name} contains NaN or Inf")
    return array


def canonicalize_quaternion(value: Any) -> np.ndarray:
    quaternion = finite_vector(value, name="quaternion", length=4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise RobotProxyAuditError("quaternion has zero or invalid norm")
    quaternion = quaternion / norm

    # Deterministic representative of the q == -q equivalence class.
    flip = bool(quaternion[3] < 0.0)
    if abs(float(quaternion[3])) <= 1.0e-12:
        for component in quaternion[:3]:
            if abs(float(component)) > 1.0e-12:
                flip = bool(component < 0.0)
                break
    if flip:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def quaternion_sign_invariant_error(left: Any, right: Any) -> float:
    lhs = canonicalize_quaternion(left).astype(np.float64)
    rhs = canonicalize_quaternion(right).astype(np.float64)
    return float(min(np.linalg.norm(lhs - rhs), np.linalg.norm(lhs + rhs)))


def extract_legacy_robot_record(
    info: Any,
    *,
    expected_joint_count: int,
) -> LegacyRobotRecord:
    if not isinstance(info, Mapping):
        raise RobotProxyAuditError("episode info is not a mapping")
    extras = info.get("extras")
    if not isinstance(extras, Mapping):
        raise RobotProxyAuditError("episode info has no extras mapping")
    proxy = extras.get("robot_pose_proxy")
    if not isinstance(proxy, Mapping):
        raise RobotProxyAuditError("robot_pose_proxy is missing")

    source = str(proxy.get("source", ""))
    if source != "pybullet_robot_body":
        raise RobotProxyAuditError(
            f"legacy robot source is not migratable: {source!r}"
        )
    body_id_value = proxy.get("body_id")
    body_id = (
        int(body_id_value)
        if isinstance(body_id_value, (int, np.integer))
        else None
    )
    positions = finite_vector(
        proxy.get("joint_positions"),
        name="legacy joint_positions",
        length=expected_joint_count,
    )
    velocities = finite_vector(
        proxy.get("joint_velocities"),
        name="legacy joint_velocities",
        length=expected_joint_count,
    )
    stored_position = finite_vector(
        proxy.get("ee_position"),
        name="legacy ee_position",
        length=3,
    )
    stored_quaternion = canonicalize_quaternion(
        proxy.get("ee_orientation")
    )
    return LegacyRobotRecord(
        source=source,
        body_id=body_id,
        joint_positions=positions,
        joint_velocities=velocities,
        stored_ee_position=stored_position,
        stored_ee_quaternion=stored_quaternion,
    )


def pack_robot_proxy_v3(
    *,
    controlled_joint_positions: Any,
    controlled_joint_velocities: Any,
    ee_position: Any,
    ee_quaternion: Any,
) -> np.ndarray:
    joint_position = finite_vector(
        controlled_joint_positions,
        name="controlled_joint_positions",
        length=CONTROLLED_JOINT_COUNT,
    )
    joint_velocity = finite_vector(
        controlled_joint_velocities,
        name="controlled_joint_velocities",
        length=CONTROLLED_JOINT_COUNT,
    )
    position = finite_vector(ee_position, name="ee_position", length=3)
    quaternion = canonicalize_quaternion(ee_quaternion)
    output = np.concatenate(
        [joint_position, joint_velocity, position, quaternion],
        axis=0,
    ).astype(np.float32)
    if output.shape != (ROBOT_PROXY_V3_DIM,):
        raise AssertionError("robot-proxy v3 packing produced wrong dimension")
    if not np.all(np.isfinite(output)):
        raise RobotProxyAuditError("robot-proxy v3 contains NaN or Inf")
    return output


class PyBulletRobotKinematics:
    """Deterministic FK backend for the pinned formal suction UR5."""

    def __init__(self, submodule_root: Path):
        self.submodule_root = Path(submodule_root).resolve()
        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self._p: Any = None
        self.manifest: Optional[KinematicsManifest] = None

    def __enter__(self) -> "PyBulletRobotKinematics":
        import pybullet as p

        self._p = p
        self.client_id = int(p.connect(p.DIRECT))
        if self.client_id < 0:
            raise RuntimeError("failed to connect to PyBullet DIRECT")
        ravens_root = self.submodule_root / "ravens"
        urdf_path = ravens_root / ROBOT_URDF_RELATIVE
        if not urdf_path.is_file():
            raise FileNotFoundError(urdf_path)
        p.setAdditionalSearchPath(
            str(ravens_root),
            physicsClientId=self.client_id,
        )
        self.robot_id = int(
            p.loadURDF(
                ROBOT_URDF_RELATIVE,
                physicsClientId=self.client_id,
            )
        )
        joint_count = int(
            p.getNumJoints(
                self.robot_id,
                physicsClientId=self.client_id,
            )
        )
        indices: List[int] = []
        names: List[str] = []
        for index in range(joint_count):
            info = p.getJointInfo(
                self.robot_id,
                index,
                physicsClientId=self.client_id,
            )
            if int(info[2]) == int(p.JOINT_REVOLUTE):
                indices.append(int(info[0]))
                raw_name = info[1]
                names.append(
                    raw_name.decode("utf-8")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )
        tip_info = p.getJointInfo(
            self.robot_id,
            EE_TIP_LINK,
            physicsClientId=self.client_id,
        )
        raw_tip_name = tip_info[12]
        tip_name = (
            raw_tip_name.decode("utf-8")
            if isinstance(raw_tip_name, bytes)
            else str(raw_tip_name)
        )
        self.manifest = KinematicsManifest(
            joint_count=joint_count,
            controlled_joint_indices=tuple(indices),
            controlled_joint_names=tuple(names),
            ee_tip_link=EE_TIP_LINK,
            ee_tip_link_name=tip_name,
            robot_urdf_relative=ROBOT_URDF_RELATIVE,
            robot_urdf_sha256=sha256_file(urdf_path),
        )
        self.manifest.validate()
        return self

    def close(self) -> None:
        if self.client_id is not None and self._p is not None:
            try:
                self._p.disconnect(physicsClientId=self.client_id)
            finally:
                self.client_id = None
                self.robot_id = None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def reconstruct(
        self,
        full_joint_positions: Any,
        full_joint_velocities: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (
            self.manifest is None
            or self.client_id is None
            or self.robot_id is None
        ):
            raise RuntimeError("kinematics backend is not open")
        positions = finite_vector(
            full_joint_positions,
            name="full_joint_positions",
            length=self.manifest.joint_count,
        )
        velocities = finite_vector(
            full_joint_velocities,
            name="full_joint_velocities",
            length=self.manifest.joint_count,
        )
        for index in range(self.manifest.joint_count):
            self._p.resetJointState(
                self.robot_id,
                index,
                targetValue=float(positions[index]),
                targetVelocity=float(velocities[index]),
                physicsClientId=self.client_id,
            )
        state = self._p.getLinkState(
            self.robot_id,
            self.manifest.ee_tip_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        position = finite_vector(
            state[0],
            name="reconstructed ee_position",
            length=3,
        ).astype(np.float32)
        quaternion = canonicalize_quaternion(state[1])
        return position, quaternion


def migrate_legacy_record(
    legacy: LegacyRobotRecord,
    *,
    manifest: KinematicsManifest,
    kinematics: Any,
) -> MigratedRobotRecord:
    manifest.validate()
    indices = np.asarray(
        manifest.controlled_joint_indices,
        dtype=np.int64,
    )
    joint_position = legacy.joint_positions[indices].astype(np.float32)
    joint_velocity = legacy.joint_velocities[indices].astype(np.float32)
    ee_position, ee_quaternion = kinematics.reconstruct(
        legacy.joint_positions,
        legacy.joint_velocities,
    )
    ee_position = finite_vector(
        ee_position, name="reconstructed ee_position", length=3
    ).astype(np.float32)
    ee_quaternion = canonicalize_quaternion(ee_quaternion)
    values = pack_robot_proxy_v3(
        controlled_joint_positions=joint_position,
        controlled_joint_velocities=joint_velocity,
        ee_position=ee_position,
        ee_quaternion=ee_quaternion,
    )
    return MigratedRobotRecord(
        values=values,
        controlled_joint_positions=joint_position,
        controlled_joint_velocities=joint_velocity,
        ee_position=ee_position,
        ee_quaternion=ee_quaternion,
        legacy_stored_ee_position_error=float(
            np.linalg.norm(
                legacy.stored_ee_position.astype(np.float64)
                - ee_position.astype(np.float64)
            )
        ),
        legacy_stored_ee_quaternion_error=quaternion_sign_invariant_error(
            legacy.stored_ee_quaternion,
            ee_quaternion,
        ),
    )


def load_episode(path: Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise RobotProxyAuditError(f"episode is not a dict: {path}")
    infos = list(value.get("infos", []))
    actions = list(value.get("actions", []))
    last_info = value.get("last_info")
    if not infos or last_info is None or len(infos) != len(actions):
        raise RobotProxyAuditError(
            f"incomplete episode {path}: "
            f"{len(infos)} infos, {len(actions)} actions"
        )
    return value


def episode_infos(payload: Mapping[str, Any]) -> Iterator[Tuple[int, Any]]:
    infos = list(payload["infos"])
    for index, info in enumerate(infos):
        yield index, info
    yield len(infos), payload["last_info"]


def source_defect_audit(submodule_root: Path) -> Dict[str, Any]:
    task_path = (
        Path(submodule_root)
        / "ravens/tasks/ccda_slack_cable_v2.py"
    )
    environment_path = Path(submodule_root) / "ravens/environment.py"
    task_text = task_path.read_text(encoding="utf-8")
    environment_text = environment_path.read_text(encoding="utf-8")

    checks = {
        "legacy_enumerates_all_urdf_joints": (
            "range(count)" in task_text
            and "p.getJointState(body_id, index)" in task_text
        ),
        "legacy_uses_last_urdf_link": (
            "p.getLinkState(body_id, count - 1)" in task_text
        ),
        "legacy_swallows_broad_exception": (
            "except Exception:" in task_text
            and "missing_zero_proxy" in task_text
        ),
        "environment_defines_suction_tip_link_12": (
            "self.ee_tip_link = 12" in environment_text
        ),
        "environment_filters_revolute_joints": (
            "if j[2] == p.JOINT_REVOLUTE" in environment_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "task_source_sha256": sha256_file(task_path),
        "environment_source_sha256": sha256_file(environment_path),
    }


def update_record_digest(
    digest: "hashlib._Hash",
    *,
    relative_path: str,
    info_index: int,
    values: np.ndarray,
) -> None:
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(struct.pack("<q", int(info_index)))
    digest.update(np.ascontiguousarray(values, dtype=np.float32).tobytes())


def render_markdown(report: Mapping[str, Any]) -> str:
    migration = report["migration"]
    source = report["source_defects"]

    def metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.9g}"

    lines = [
        "# Phase3.14b-r2.5.5 Robot-Proxy Provenance Audit",
        "",
        f"- Audit verdict: `{report['verdict']}`",
        f"- Scientific status: `{report['scientific_status']}`",
        f"- Root cause: `{report['root_cause']}`",
        f"- Migration feasible: `{str(migration['feasible']).lower()}`",
        f"- Required next path: `{migration['required_next_path']}`",
        "",
        "## Confirmed source defects",
        "",
    ]
    for key, value in source["checks"].items():
        lines.append(f"- `{key}`: `{str(bool(value)).lower()}`")
    lines += [
        "",
        "## Raw-data audit",
        "",
        f"- Episodes: `{migration['episode_count']}`",
        f"- Robot records: `{migration['record_count']}`",
        f"- Failures: `{migration['failure_count']}`",
        (
            "- Controlled joint indices: "
            f"`{migration['kinematics']['controlled_joint_indices']}`"
        ),
        (
            "- Legacy EE position delta p95: "
            f"`{metric(migration['legacy_ee_position_error_p95'])}`"
        ),
        (
            "- Legacy EE quaternion delta p95: "
            f"`{metric(migration['legacy_ee_quaternion_error_p95'])}`"
        ),
        "",
        "## Boundary",
        "",
        "- No legacy cache or raw episode was modified.",
        "- No migrated dataset, window NPZ, cache, checkpoint, or prediction tensor was written.",
        "- No diffusion, reverse sampling, IDM, candidate execution, Phase4, or CPS was run.",
        "- `train_only_recommendation` and `selected_configuration` remain `None`.",
    ]
    return "\n".join(lines) + "\n"


def contract_payload() -> Dict[str, Any]:
    contract = RobotProxyV3Contract()
    contract.validate()
    return asdict(contract)
