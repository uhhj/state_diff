from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccda_phase3.phase314b_r255_robot_proxy_provenance import (
    KinematicsManifest,
    LegacyRobotRecord,
    ROBOT_PROXY_V3_DIM,
    RobotProxyAuditError,
    RobotProxyV3Contract,
    canonicalize_quaternion,
    extract_legacy_robot_record,
    migrate_legacy_record,
    pack_robot_proxy_v3,
)


class FakeKinematics:
    def reconstruct(self, positions, velocities):
        del positions, velocities
        return (
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        )


def manifest() -> KinematicsManifest:
    return KinematicsManifest(
        joint_count=14,
        controlled_joint_indices=(1, 2, 3, 4, 5, 6),
        controlled_joint_names=("a", "b", "c", "d", "e", "f"),
        ee_tip_link=12,
        ee_tip_link_name="tip",
        robot_urdf_relative="assets/ur5/ur5-suction.urdf",
        robot_urdf_sha256="0" * 64,
    )


def legacy_info(joint_count=14):
    return {
        "extras": {
            "robot_pose_proxy": {
                "source": "pybullet_robot_body",
                "body_id": 2,
                "joint_positions": list(np.arange(joint_count) / 10.0),
                "joint_velocities": list(np.arange(joint_count) / 100.0),
                "ee_position": [9.0, 9.0, 9.0],
                "ee_orientation": [0.0, 0.0, 0.0, 1.0],
            }
        }
    }


def test_contract_is_semantic_67_dimensional_schema():
    contract = RobotProxyV3Contract()
    contract.validate()
    assert contract.robot_proxy_dim == 19
    assert contract.state_dim == 67
    assert contract.missing_value_policy == "reject_record"
    assert contract.model_presence_mask is False


def test_quaternion_is_normalized_and_sign_canonicalized():
    value = canonicalize_quaternion([0.0, 0.0, 0.0, -2.0])
    np.testing.assert_array_equal(
        value,
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_short_legacy_joint_arrays_are_rejected_not_padded():
    info = legacy_info(joint_count=13)
    with pytest.raises(RobotProxyAuditError):
        extract_legacy_robot_record(info, expected_joint_count=14)


def test_missing_zero_proxy_is_not_migratable():
    info = legacy_info()
    info["extras"]["robot_pose_proxy"]["source"] = "missing_zero_proxy"
    with pytest.raises(RobotProxyAuditError):
        extract_legacy_robot_record(info, expected_joint_count=14)


def test_pack_robot_proxy_v3_has_exact_layout():
    packed = pack_robot_proxy_v3(
        controlled_joint_positions=np.arange(6),
        controlled_joint_velocities=np.arange(6) + 10,
        ee_position=[1, 2, 3],
        ee_quaternion=[0, 0, 0, 1],
    )
    assert packed.shape == (ROBOT_PROXY_V3_DIM,)
    np.testing.assert_array_equal(packed[:6], np.arange(6))
    np.testing.assert_array_equal(packed[6:12], np.arange(6) + 10)


def test_migration_selects_controlled_indices_and_recomputes_tip_pose():
    contract = manifest()
    contract.validate()
    info = legacy_info()
    legacy = extract_legacy_robot_record(info, expected_joint_count=14)
    migrated = migrate_legacy_record(
        legacy,
        manifest=contract,
        kinematics=FakeKinematics(),
    )
    np.testing.assert_allclose(
        migrated.controlled_joint_positions,
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    )
    np.testing.assert_array_equal(
        migrated.ee_quaternion,
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert migrated.values.shape == (19,)
    assert migrated.legacy_stored_ee_position_error > 1.0
