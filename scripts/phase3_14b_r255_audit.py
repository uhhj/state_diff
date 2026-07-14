#!/usr/bin/env python3
"""Read-only robot-proxy provenance and deterministic migration audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r255_robot_proxy_provenance import (
    BASE_EVIDENCE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    FORMAL_CONDITIONS,
    FORMAL_TASK_NAME,
    LEGACY_SCHEMA_VERSION,
    PHASE_NAME,
    PyBulletRobotKinematics,
    RobotProxyAuditError,
    contract_payload,
    episode_infos,
    extract_legacy_robot_record,
    git_output,
    load_episode,
    migrate_legacy_record,
    render_markdown,
    sha256_array,
    sha256_file,
    source_defect_audit,
    update_record_digest,
    write_json_once,
    write_text_once,
)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def assert_repository_contract(root: Path, test_gate: Path) -> Dict[str, Any]:
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("r2.5.5 requires Experiment1")
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_EVIDENCE_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Resume5 evidence commit is not an ancestor")

    submodule = root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("pinned DeformableRavens commit changed")
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise RuntimeError("DeformableRavens worktree is dirty")

    cache = root / "data/phase3_14_cache/phase3_14a_training_cache.npz"
    if sha256_file(cache) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("legacy immutable cache SHA changed")

    status_lines = [
        line
        for line in git_output(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ).splitlines()
        if line.strip()
    ]
    allowed = {test_gate.relative_to(root).as_posix()}
    unexpected = []
    for line in status_lines:
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree paths before r2.5.5 audit: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "head": git_output(root, "rev-parse", "HEAD"),
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
        "legacy_cache_sha256": EXPECTED_CACHE_SHA256,
    }


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def verify_episode_manifest(
    payload: Mapping[str, Any],
    *,
    split: str,
    condition: str,
    source_path: Path,
) -> Dict[str, Any]:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RobotProxyAuditError("episode manifest is missing")
    expected = {
        "split": split,
        "condition": condition,
        "task": FORMAL_TASK_NAME,
        "observation_schema_version": LEGACY_SCHEMA_VERSION,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
        "contains_simulator_bead_velocity": False,
        "input_canonicalization": False,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise RobotProxyAuditError(
                f"{source_path}: manifest {key}="
                f"{manifest.get(key)!r} != {expected_value!r}"
            )
    sidecar = source_path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        raise RobotProxyAuditError(f"missing sidecar: {sidecar}")
    sidecar_value = load_json(sidecar)
    for key, expected_value in expected.items():
        if sidecar_value.get(key) != expected_value:
            raise RobotProxyAuditError(
                f"{sidecar}: {key} mismatch"
            )
    if dict(manifest) != sidecar_value:
        raise RobotProxyAuditError(
            f"pickle/sidecar manifest mismatch: {source_path}"
        )
    return dict(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--formal-root",
        default="data/phase3_state_v2_slack",
    )
    parser.add_argument(
        "--test-gate",
        default="reports/phase3_14b_r255_test_gate_summary.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r255_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r255_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    formal_root = resolve(root, args.formal_root)
    test_gate_path = resolve(root, args.test_gate)
    summary_path = resolve(root, args.summary)
    report_path = resolve(root, args.report)
    if summary_path.exists() or report_path.exists():
        raise RuntimeError("refusing to overwrite r2.5.5 evidence")

    repository = assert_repository_contract(root, test_gate_path)
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("r2.5.5 scoped test gate is not PASS")
    for relative, expected in test_gate["test_manifest_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"test changed after gate: {relative}")

    submodule = root / "external/deformable-ravens"
    source_defects = source_defect_audit(submodule)
    failures: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    pair_members: Dict[Tuple[str, int, str], set] = defaultdict(set)
    action_sha_by_pair: Dict[Tuple[str, int, str], set] = defaultdict(set)
    episode_count = 0
    record_count = 0
    deterministic_count = 0
    position_errors: List[float] = []
    quaternion_errors: List[float] = []
    migrated_digest = hashlib.sha256()
    legacy_digest = hashlib.sha256()

    episode_paths = sorted(
        (formal_root / "raw").glob("*/*/*.pkl")
    )
    if not episode_paths:
        raise RuntimeError("no formal raw episodes found")

    with PyBulletRobotKinematics(submodule) as kinematics:
        assert kinematics.manifest is not None
        manifest = kinematics.manifest
        for source_path in episode_paths:
            relative = source_path.relative_to(formal_root).as_posix()
            try:
                condition = source_path.parent.name
                split = source_path.parent.parent.name
                if condition not in FORMAL_CONDITIONS:
                    raise RobotProxyAuditError(
                        f"unexpected condition directory: {condition}"
                    )
                if split not in {"train", "val", "test"}:
                    raise RobotProxyAuditError(
                        f"unexpected split directory: {split}"
                    )
                payload = load_episode(source_path)
                episode_manifest = verify_episode_manifest(
                    payload,
                    split=split,
                    condition=condition,
                    source_path=source_path,
                )
                seed = int(episode_manifest["visible_seed"])
                pair_group = str(episode_manifest["pair_group"])
                pair_key = (split, seed, pair_group)
                pair_members[pair_key].add(condition)
                action_sha_by_pair[pair_key].add(
                    str(episode_manifest["action_sequence_sha256"])
                )
                episode_count += 1

                for info_index, info in episode_infos(payload):
                    legacy = extract_legacy_robot_record(
                        info,
                        expected_joint_count=manifest.joint_count,
                    )
                    source_counts[legacy.source] += 1
                    migrated = migrate_legacy_record(
                        legacy,
                        manifest=manifest,
                        kinematics=kinematics,
                    )
                    repeated = migrate_legacy_record(
                        legacy,
                        manifest=manifest,
                        kinematics=kinematics,
                    )
                    if not np.array_equal(
                        migrated.values,
                        repeated.values,
                    ):
                        raise RobotProxyAuditError(
                            "same-process FK reconstruction is not exact"
                        )
                    deterministic_count += 1
                    record_count += 1
                    position_errors.append(
                        migrated.legacy_stored_ee_position_error
                    )
                    quaternion_errors.append(
                        migrated.legacy_stored_ee_quaternion_error
                    )
                    update_record_digest(
                        migrated_digest,
                        relative_path=relative,
                        info_index=info_index,
                        values=migrated.values,
                    )
                    legacy_values = np.concatenate(
                        [
                            legacy.joint_positions,
                            legacy.joint_velocities,
                            legacy.stored_ee_position,
                            legacy.stored_ee_quaternion,
                        ]
                    ).astype(np.float32)
                    update_record_digest(
                        legacy_digest,
                        relative_path=relative,
                        info_index=info_index,
                        values=legacy_values,
                    )
            except Exception as exc:
                failures.append(
                    {
                        "source_file": relative,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        incomplete_pairs = {
            "|".join(map(str, key)): sorted(value)
            for key, value in pair_members.items()
            if set(value) != set(FORMAL_CONDITIONS)
        }
        action_mismatch_pairs = {
            "|".join(map(str, key)): sorted(value)
            for key, value in action_sha_by_pair.items()
            if len(value) != 1
        }
        feasible = bool(
            not failures
            and source_defects["all_confirmed"]
            and episode_count == len(episode_paths)
            and record_count > 0
            and deterministic_count == record_count
            and not incomplete_pairs
            and not action_mismatch_pairs
            and source_counts == Counter(
                {"pybullet_robot_body": record_count}
            )
        )

        position_array = np.asarray(position_errors, dtype=np.float64)
        quaternion_array = np.asarray(quaternion_errors, dtype=np.float64)
        migration = {
            "feasible": feasible,
            "required_next_path": (
                "MIGRATE_TO_NEW_WRITE_ONCE_STATE_V3_DATASET"
                if feasible
                else "REGENERATE_WITH_STRICT_V3_LOGGER"
            ),
            "episode_count": episode_count,
            "record_count": record_count,
            "failure_count": len(failures),
            "failures": failures[:50],
            "failure_list_truncated": len(failures) > 50,
            "source_counts": dict(source_counts),
            "same_process_exact_reconstruction_count": deterministic_count,
            "pair_count": len(pair_members),
            "incomplete_pairs": incomplete_pairs,
            "action_mismatch_pairs": action_mismatch_pairs,
            "legacy_payload_sha256": legacy_digest.hexdigest(),
            "migrated_robot_proxy_v3_sha256": (
                migrated_digest.hexdigest()
            ),
            "legacy_ee_position_error_mean": (
                float(np.mean(position_array))
                if position_array.size
                else None
            ),
            "legacy_ee_position_error_p95": (
                float(np.percentile(position_array, 95))
                if position_array.size
                else None
            ),
            "legacy_ee_position_error_max": (
                float(np.max(position_array))
                if position_array.size
                else None
            ),
            "legacy_ee_quaternion_error_mean": (
                float(np.mean(quaternion_array))
                if quaternion_array.size
                else None
            ),
            "legacy_ee_quaternion_error_p95": (
                float(np.percentile(quaternion_array, 95))
                if quaternion_array.size
                else None
            ),
            "legacy_ee_quaternion_error_max": (
                float(np.max(quaternion_array))
                if quaternion_array.size
                else None
            ),
            "kinematics": {
                "joint_count": manifest.joint_count,
                "controlled_joint_indices": list(
                    manifest.controlled_joint_indices
                ),
                "controlled_joint_names": list(
                    manifest.controlled_joint_names
                ),
                "ee_tip_link": manifest.ee_tip_link,
                "ee_tip_link_name": manifest.ee_tip_link_name,
                "robot_urdf_relative": manifest.robot_urdf_relative,
                "robot_urdf_sha256": manifest.robot_urdf_sha256,
            },
        }

    root_cause = (
        "phase314b_r255_legacy_raw_robot_proxy_deterministically_migratable"
        if migration["feasible"]
        else "phase314b_r255_robot_proxy_regeneration_required"
    )
    summary = {
        "phase": PHASE_NAME,
        "phase_id": "phase314b_r255",
        "schema": "phase314b_r255_robot_proxy_provenance_audit_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": root_cause,
        "repository": repository,
        "test_gate_report": test_gate_path.relative_to(root).as_posix(),
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": test_gate,
        "target_contract": contract_payload(),
        "source_defects": source_defects,
        "migration": migration,
        "legacy_cache_modified": False,
        "legacy_raw_modified": False,
        "migrated_dataset_written": False,
        "window_npz_written": False,
        "new_cache_written": False,
        "validation_target_rows_indexed": False,
        "formal_target_rows_indexed": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "robot_proxy_attribution_interpretable": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    write_text_once(report_path, render_markdown(summary))
    write_json_once(summary_path, summary)
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause": root_cause,
                "migration_feasible": migration["feasible"],
                "summary": str(summary_path),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
