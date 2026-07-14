"""Materialize a write-once state-v3 dataset from audited legacy raw episodes.

The implementation is deliberately additive.  It reads the immutable state-v2
raw episodes, reconstructs the official UR5 tool-tip pose with the Stage-A
kinematics contract, writes a standalone pickle-free state-v3 dataset and its
windows, and proves that cable/action semantics are byte-identical to the
historical state-v2 windows.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .data_io import (
    extract_action_template_and_vector,
    extract_bead_xy,
    get_extras,
    load_action_codec_from_template,
)
from .phase314b_r255_robot_proxy_provenance import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    LEGACY_SCHEMA_VERSION,
    PyBulletRobotKinematics,
    RobotProxyAuditError,
    episode_infos,
    extract_legacy_robot_record,
    load_episode,
    migrate_legacy_record,
    sha256_file,
    update_record_digest,
)
from .schema_v3 import (
    ACTION_DIM,
    CABLE_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    FORMAL_CONDITIONS,
    FORMAL_TASK_NAME,
    PAPER_X_DIM,
    SCHEMA_VERSION,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    SchemaV3Manifest,
    build_window,
    state_v3_from_components,
)

BASE_EVIDENCE_COMMIT = "9f9e9f98e678d435ace3c297acb917ac727a0d7d"
EXPECTED_STAGE_A_TEST_GATE_SHA256 = (
    "915244aeeb23177207fb41b0d137c2984fc6cb170ecceccc99139fa95fc237ee"
)
EXPECTED_STAGE_A_SUMMARY_SHA256 = (
    "1306eaf32acda52fcab247ee4caf0517c89e8d7a3cf6313a09f661aafff1ebc5"
)
EXPECTED_STAGE_A_REPORT_SHA256 = (
    "89ba9de727d9dd47ef1ba789f047c3380e761fb884747af85606b7e3d27ca7a1"
)
EXPECTED_EPISODES = 896
EXPECTED_RECORDS = 5152
EXPECTED_PAIRS = 448
EXPECTED_ACTIONS = EXPECTED_RECORDS - EXPECTED_EPISODES
EXPECTED_CONTROLLED_JOINTS = (2, 3, 4, 5, 6, 7)
EXPECTED_EE_TIP_LINK = 12
EXPECTED_EE_TIP_NAME = "tool_tip"
EXPECTED_MIGRATED_ROBOT_DIGEST = (
    "b15be60e00d57c065fd22bd2f389b60c329ecb02b16ce4fe43b9e057c906547c"
)
DATASET_FILE = "migrated/phase3_14b_r255_stageb_dataset.npz"
WINDOW_FILE = "windows/phase3_14b_r255_stageb_windows.npz"
ACTION_TEMPLATE_FILE = "action_template.pkl"
MANIFEST_FILE = "manifest.json"
SOURCE_FILES = (
    "ccda_phase3/schema_v3.py",
    "ccda_phase3/phase314b_r255_stageb_dataset.py",
    "ccda_phase3/phase314b_r255_robot_proxy_provenance.py",
    "ccda_phase3/data_io.py",
    "scripts/phase3_14b_r255_stageb_worker.py",
    "scripts/phase3_14b_r255_stageb_test_gate.py",
    "scripts/phase3_14b_r255_stageb_build.py",
    "scripts/phase3_14b_r255_stageb_blocked.py",
    "scripts/phase3_14b_r255_stageb_run.sh",
    "tests/test_phase3_14b_r255_stageb_dataset.py",
)


class StateV3MigrationError(RuntimeError):
    """Raised when state-v3 materialization would require guessing."""


def _finite(value: Any, *, name: str, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise StateV3MigrationError(f"{name} shape {array.shape} != {shape}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise StateV3MigrationError(f"{name} contains NaN or Inf")
    return array


def optional_int(value: Any, *, missing: int = -1) -> int:
    """Preserve a legitimate zero; only None/empty means missing."""
    if value is None or value == "":
        return int(missing)
    result = int(value)
    return result


def strings(values: Sequence[Any]) -> np.ndarray:
    text = [str(value) for value in values]
    width = max([1] + [len(value) for value in text])
    return np.asarray(text, dtype=f"<U{width}")


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite write-once artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, target)
    directory_fd = os.open(str(target.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_copy_once(source: Path, target: Path) -> None:
    with Path(source).open("rb") as handle:
        atomic_write_once(target, handle.read())


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create byte-reproducible NPZ output with fixed ZIP metadata."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or name.endswith(".npy"):
                raise ValueError(f"invalid NPZ key: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.kind == "O":
                raise ValueError(f"object dtype is forbidden: {name}")
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError(f"non-finite NPZ array: {name}")
            npy = io.BytesIO()
            np.lib.format.write_array(npy, array, allow_pickle=False)
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(
                info,
                npy.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return buffer.getvalue()


def write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write_once(path, deterministic_npz_bytes(arrays))
    with np.load(path, allow_pickle=False) as checked:
        if set(checked.files) != set(arrays):
            raise StateV3MigrationError("saved NPZ key mismatch")
        for key in checked.files:
            expected = np.asarray(arrays[key])
            actual = checked[key]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise StateV3MigrationError(f"saved NPZ layout mismatch: {key}")
            if not np.array_equal(actual, expected, equal_nan=False):
                raise StateV3MigrationError(f"saved NPZ bytes changed: {key}")


def load_npz_strict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        result = {key: loaded[key] for key in loaded.files}
    if any(value.dtype.kind == "O" for value in result.values()):
        raise StateV3MigrationError(f"object dtype in {path}")
    return result


def canonical_source_file(value: Any) -> str:
    parts = Path(str(value)).as_posix().split("/")
    try:
        index = parts.index("raw")
    except ValueError as exc:
        raise StateV3MigrationError(f"source path has no raw component: {value}") from exc
    relative = "/".join(parts[index:])
    if len(relative.split("/")) != 4:
        raise StateV3MigrationError(f"unexpected canonical source path: {relative}")
    return relative


def validate_legacy_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    formal_root: Path,
    split: str,
    condition: str,
) -> Dict[str, Any]:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise StateV3MigrationError(f"missing episode manifest: {source_path}")
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
            raise StateV3MigrationError(
                f"{source_path}: manifest {key}={manifest.get(key)!r} != {expected_value!r}"
            )
    sidecar = source_path.with_suffix(".manifest.json")
    sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
    if dict(manifest) != sidecar_value:
        raise StateV3MigrationError(f"pickle/sidecar manifest mismatch: {source_path}")
    if canonical_source_file(source_path.relative_to(formal_root)) != source_path.relative_to(formal_root).as_posix():
        raise StateV3MigrationError(f"non-canonical raw path: {source_path}")
    return dict(manifest)


def episode_outcome(payload: Mapping[str, Any]) -> Tuple[bool, float, int, int]:
    last_info = payload["last_info"]
    final = get_extras(last_info)
    hidden = final.get("hidden_contact_meta", {})
    if not isinstance(hidden, Mapping):
        hidden = {}
    fraction = final.get("total_rewards", 0.0)
    if final.get("nb_beads"):
        fraction = float(final.get("nb_zone", 0)) / float(final["nb_beads"])
    return (
        bool(final.get("task.done", payload.get("success", False))),
        float(fraction),
        optional_int(hidden.get("slack_engagement_physics_step")),
        optional_int(hidden.get("slack_release_physics_step")),
    )


def pre_engagement_from_info(info: Any) -> bool:
    contact = get_extras(info).get("hidden_contact_meta", {})
    return (
        not isinstance(contact, Mapping)
        or contact.get("slack_engagement_physics_step") is None
    )


def stage_a_robot_digest(
    records: Sequence[Tuple[str, int, np.ndarray]],
) -> str:
    """Reproduce the Stage-A global source-path/info-index digest order."""
    digest = hashlib.sha256()
    for relative, info_index, values in sorted(
        records,
        key=lambda item: (str(item[0]), int(item[1])),
    ):
        update_record_digest(
            digest,
            relative_path=str(relative),
            info_index=int(info_index),
            values=np.asarray(values, dtype=np.float32),
        )
    return digest.hexdigest()


def validate_stage_a_summary(root: Path) -> Dict[str, Any]:
    test_gate = root / "reports/phase3_14b_r255_test_gate_summary.json"
    summary = root / "reports/phase3_14b_r255_summary.json"
    report = root / "reports/phase3_14b_r255_report.md"
    expected = {
        test_gate: EXPECTED_STAGE_A_TEST_GATE_SHA256,
        summary: EXPECTED_STAGE_A_SUMMARY_SHA256,
        report: EXPECTED_STAGE_A_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise StateV3MigrationError(f"Stage-A evidence SHA changed: {path}")
    value = json.loads(summary.read_text(encoding="utf-8"))
    migration = value.get("migration", {})
    kinematics = migration.get("kinematics", {})
    required = {
        "feasible": True,
        "episode_count": EXPECTED_EPISODES,
        "record_count": EXPECTED_RECORDS,
        "pair_count": EXPECTED_PAIRS,
        "failure_count": 0,
        "required_next_path": "MIGRATE_TO_NEW_WRITE_ONCE_STATE_V3_DATASET",
        "migrated_robot_proxy_v3_sha256": EXPECTED_MIGRATED_ROBOT_DIGEST,
    }
    for key, expected_value in required.items():
        if migration.get(key) != expected_value:
            raise StateV3MigrationError(f"Stage-A summary {key} changed")
    if tuple(kinematics.get("controlled_joint_indices", [])) != EXPECTED_CONTROLLED_JOINTS:
        raise StateV3MigrationError("controlled-joint identity changed")
    if int(kinematics.get("ee_tip_link", -1)) != EXPECTED_EE_TIP_LINK:
        raise StateV3MigrationError("EE tip link changed")
    if str(kinematics.get("ee_tip_link_name", "")) != EXPECTED_EE_TIP_NAME:
        raise StateV3MigrationError("EE tip name changed")
    return value


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise StateV3MigrationError(f"comparison shape mismatch: {lhs.shape} != {rhs.shape}")
    return float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0


def compare_legacy_windows(
    legacy: Mapping[str, np.ndarray],
    migrated: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    rows = int(migrated["paper_x"].shape[0])
    if int(legacy["paper_x"].shape[0]) != rows:
        raise StateV3MigrationError("legacy/new window row count mismatch")
    exact_metadata = [
        "condition_name",
        "visible_seed",
        "split_name",
        "pair_group",
        "window_t",
        "success",
        "final_fraction",
        "pre_engagement",
    ]
    metadata_exact: Dict[str, bool] = {}
    for key in exact_metadata:
        metadata_exact[key] = bool(np.array_equal(legacy[key], migrated[key]))
        if not metadata_exact[key]:
            raise StateV3MigrationError(f"legacy/new metadata mismatch: {key}")
    zero_step_corrections: Dict[str, int] = {}
    for key in ("engagement_step", "release_step"):
        old_value = np.asarray(legacy[key], dtype=np.int64)
        new_value = np.asarray(migrated[key], dtype=np.int64)
        exact = old_value == new_value
        permitted_fix = (old_value == -1) & (new_value == 0)
        if not np.all(exact | permitted_fix):
            raise StateV3MigrationError(f"legacy/new metadata mismatch: {key}")
        zero_step_corrections[key] = int(np.sum(permitted_fix))

    old_sources = strings([canonical_source_file(value) for value in legacy["source_file"]])
    source_exact = bool(np.array_equal(old_sources, migrated["source_file"]))
    if not source_exact:
        raise StateV3MigrationError("legacy/new canonical source order mismatch")

    if legacy["paper_x"].shape[1] != DEFAULT_TH * 87:
        raise StateV3MigrationError("legacy paper_x is not the frozen 87-D state-v2 layout")
    if legacy["state_action_x"].shape[1] != DEFAULT_TH * 87 + DEFAULT_TH * ACTION_DIM:
        raise StateV3MigrationError("legacy state_action_x layout changed")
    if legacy["y_state"].shape[1:] != (DEFAULT_TF, 87):
        raise StateV3MigrationError("legacy y_state layout changed")
    old_history = legacy["paper_x"].reshape(rows, DEFAULT_TH, 87)
    new_history = migrated["paper_x"].reshape(rows, DEFAULT_TH, STATE_DIM)
    old_future = legacy["y_state"]
    new_future = migrated["y_state"]
    old_final = legacy["y_final_state"]
    new_final = migrated["y_final_state"]
    cable_history_error = _max_abs(old_history[:, :, :CABLE_DIM], new_history[:, :, :CABLE_DIM])
    cable_future_error = _max_abs(old_future[:, :, :CABLE_DIM], new_future[:, :, :CABLE_DIM])
    cable_final_error = _max_abs(old_final[:, :CABLE_DIM], new_final[:, :CABLE_DIM])
    y_action_error = _max_abs(legacy["y_action"], migrated["y_action"])
    old_action_history = legacy["state_action_x"][:, legacy["paper_x"].shape[1] :]
    new_action_history = migrated["state_action_x"][:, PAPER_X_DIM:]
    action_history_error = _max_abs(old_action_history, new_action_history)
    errors = {
        "cable_history_max_abs": cable_history_error,
        "cable_future_max_abs": cable_future_error,
        "cable_final_max_abs": cable_final_error,
        "action_history_max_abs": action_history_error,
        "target_action_max_abs": y_action_error,
    }
    if any(value != 0.0 for value in errors.values()):
        raise StateV3MigrationError(f"legacy/new deployable content changed: {errors}")
    return {
        "rows": rows,
        "metadata_exact": metadata_exact,
        "canonical_source_exact": source_exact,
        "zero_step_metadata_corrections": zero_step_corrections,
        **errors,
        "cable_and_action_exact": True,
    }


def split_isolation(arrays: Mapping[str, np.ndarray]) -> Dict[str, int]:
    splits = arrays["split_name"].astype(str)
    values_by_name = {
        "visible_seed": arrays["visible_seed"].astype(np.int64),
        "pair_group": arrays["pair_group"].astype(str),
        "source_file": arrays["source_file"].astype(str),
    }
    result: Dict[str, int] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for name, values in values_by_name.items():
            overlap = set(values[splits == left].tolist()).intersection(
                set(values[splits == right].tolist())
            )
            result[f"{left}_{right}_{name}_overlap"] = len(overlap)
    if any(result.values()):
        raise StateV3MigrationError(f"state-v3 split isolation failed: {result}")
    return result


def validate_window_pairs(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    members: MutableMapping[Tuple[str, int, str, int], Counter] = defaultdict(Counter)
    for split, seed, group, time, condition in zip(
        arrays["split_name"].astype(str),
        arrays["visible_seed"].astype(np.int64),
        arrays["pair_group"].astype(str),
        arrays["window_t"].astype(np.int64),
        arrays["condition_name"].astype(str),
    ):
        members[(split, int(seed), group, int(time))][condition] += 1
    expected = Counter({condition: 1 for condition in FORMAL_CONDITIONS})
    invalid = {
        "|".join(map(str, key)): dict(value)
        for key, value in members.items()
        if value != expected
    }
    if invalid:
        raise StateV3MigrationError(
            f"invalid state-v3 window pair multiplicity: "
            f"{list(invalid.items())[:10]}"
        )
    if len(members) * len(FORMAL_CONDITIONS) != int(arrays["paper_x"].shape[0]):
        raise StateV3MigrationError("paired-window row accounting is inconsistent")
    return {
        "pair_window_count": len(members),
        "rows_per_pair": len(FORMAL_CONDITIONS),
        "invalid_pairs": {},
    }


def build_state_v3_artifacts(
    *,
    root: Path,
    output_root: Path,
    formal_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    output_root = Path(output_root).resolve()
    formal_root = (
        root / "data/phase3_state_v2_slack"
        if formal_root is None
        else Path(formal_root).resolve()
    )
    if output_root.exists():
        raise FileExistsError(f"staging output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    stage_a = validate_stage_a_summary(root)
    stage_a_migration = stage_a["migration"]
    stage_a_kinematics = stage_a_migration["kinematics"]
    legacy_manifest_path = formal_root / "manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    legacy_windows_path = Path(legacy_manifest["window_npz"])
    if not legacy_windows_path.is_absolute():
        legacy_windows_path = (legacy_manifest_path.parent / legacy_windows_path).resolve()
    legacy_template_path = Path(legacy_manifest["action_template"])
    if not legacy_template_path.is_absolute():
        legacy_template_path = (legacy_manifest_path.parent / legacy_template_path).resolve()
    if sha256_file(legacy_windows_path) != str(legacy_manifest["window_npz_sha256"]):
        raise StateV3MigrationError("legacy window NPZ SHA changed")
    if sha256_file(legacy_template_path) != str(legacy_manifest["action_template_sha256"]):
        raise StateV3MigrationError("legacy action-template SHA changed")
    codec = load_action_codec_from_template(legacy_template_path)

    state_blocks: List[np.ndarray] = []
    action_blocks: List[np.ndarray] = []
    pre_blocks: List[np.ndarray] = []
    state_offsets = [0]
    action_offsets = [0]
    episode_metadata: Dict[str, List[Any]] = defaultdict(list)
    window_rows: List[Dict[str, Any]] = []
    robot_digest_records: List[Tuple[str, int, np.ndarray]] = []
    pair_members: MutableMapping[Tuple[str, int, str], set] = defaultdict(set)
    paired_action_sha: MutableMapping[Tuple[str, int, str], set] = defaultdict(set)
    robot_sources = Counter()

    episode_paths: List[Path] = []
    for split in ("train", "val", "test"):
        for condition in FORMAL_CONDITIONS:
            episode_paths.extend(sorted((formal_root / "raw" / split / condition).glob("*.pkl")))
    if len(episode_paths) != EXPECTED_EPISODES:
        raise StateV3MigrationError(f"episode count changed: {len(episode_paths)}")

    with PyBulletRobotKinematics(root / "external/deformable-ravens") as kinematics:
        if kinematics.manifest is None:
            raise StateV3MigrationError("kinematics manifest unavailable")
        manifest = kinematics.manifest
        if tuple(manifest.controlled_joint_indices) != EXPECTED_CONTROLLED_JOINTS:
            raise StateV3MigrationError("controlled-joint indices changed")
        if manifest.ee_tip_link != EXPECTED_EE_TIP_LINK or manifest.ee_tip_link_name != EXPECTED_EE_TIP_NAME:
            raise StateV3MigrationError("official EE identity changed")

        for episode_index, source_path in enumerate(episode_paths):
            split = source_path.parent.parent.name
            condition = source_path.parent.name
            payload = load_episode(source_path)
            metadata = validate_legacy_manifest(
                payload,
                source_path=source_path,
                formal_root=formal_root,
                split=split,
                condition=condition,
            )
            infos = list(payload["infos"])
            raw_actions = list(payload["actions"])
            visible_seed = int(metadata["visible_seed"])
            pair_group = str(metadata["pair_group"])
            action_sequence_sha = str(metadata["action_sequence_sha256"])
            pair_key = (split, visible_seed, pair_group)
            pair_members[pair_key].add(condition)
            paired_action_sha[pair_key].add(action_sequence_sha)

            states: List[np.ndarray] = []
            pre_engagement: List[bool] = []
            for info_index, info in episode_infos(payload):
                cable_xy = extract_bead_xy(info)
                if cable_xy.shape != (24, 2):
                    raise StateV3MigrationError(f"invalid ordered cable in {source_path} record {info_index}")
                legacy_robot = extract_legacy_robot_record(
                    info,
                    expected_joint_count=manifest.joint_count,
                )
                migrated_robot = migrate_legacy_record(
                    legacy_robot,
                    manifest=manifest,
                    kinematics=kinematics,
                )
                robot_sources[legacy_robot.source] += 1
                relative = source_path.relative_to(formal_root).as_posix()
                robot_digest_records.append(
                    (relative, int(info_index), migrated_robot.values.copy())
                )
                states.append(state_v3_from_components(cable_xy, migrated_robot.values))
                pre_engagement.append(pre_engagement_from_info(info))

            actions: List[np.ndarray] = []
            for action in raw_actions:
                _, vector, active_codec = extract_action_template_and_vector(action, codec)
                if active_codec.dim() != ACTION_DIM:
                    raise StateV3MigrationError("action codec dimension changed")
                actions.append(np.asarray(vector, dtype=np.float32))
            state_array = np.stack(states).astype(np.float32)
            action_array = np.stack(actions).astype(np.float32)
            pre_array = np.asarray(pre_engagement, dtype=np.bool_)
            if state_array.shape != (len(raw_actions) + 1, STATE_DIM):
                raise StateV3MigrationError(f"state/action count mismatch: {source_path}")
            if action_array.shape != (len(raw_actions), ACTION_DIM):
                raise StateV3MigrationError(f"invalid action array: {source_path}")
            success, final_fraction, engagement_step, release_step = episode_outcome(payload)
            canonical_source = source_path.relative_to(formal_root).as_posix()
            sidecar = source_path.with_suffix(".manifest.json")

            state_blocks.append(state_array)
            action_blocks.append(action_array)
            pre_blocks.append(pre_array)
            state_offsets.append(state_offsets[-1] + state_array.shape[0])
            action_offsets.append(action_offsets[-1] + action_array.shape[0])
            episode_values = {
                "split_name": split,
                "condition_name": condition,
                "visible_seed": visible_seed,
                "pair_group": pair_group,
                "source_file": canonical_source,
                "source_pickle_sha256": sha256_file(source_path),
                "source_sidecar_sha256": sha256_file(sidecar),
                "action_sequence_sha256": action_sequence_sha,
                "success": success,
                "final_fraction": final_fraction,
                "engagement_step": engagement_step,
                "release_step": release_step,
            }
            for key, value in episode_values.items():
                episode_metadata[key].append(value)

            for current in range(action_array.shape[0]):
                window = build_window(
                    states=state_array,
                    action_vectors=action_array,
                    current_index=current,
                )
                window_rows.append(
                    {
                        **window,
                        **episode_values,
                        "episode_index": episode_index,
                        "window_t": current,
                        "pre_engagement": bool(pre_array[current]),
                    }
                )

    # Stage A hashed globally lexicographically sorted raw paths.  Dataset rows
    # intentionally retain the historical train/val/test window order, so the
    # provenance digest is reconstructed separately in the exact Stage-A order.
    robot_digest_value = stage_a_robot_digest(robot_digest_records)
    if robot_digest_value != EXPECTED_MIGRATED_ROBOT_DIGEST:
        raise StateV3MigrationError("materialized robot digest differs from Stage A")
    if robot_sources != Counter({"pybullet_robot_body": EXPECTED_RECORDS}):
        raise StateV3MigrationError(f"unexpected robot sources: {robot_sources}")
    if len(pair_members) != EXPECTED_PAIRS:
        raise StateV3MigrationError("episode pair count changed")
    for key, members in pair_members.items():
        if set(members) != set(FORMAL_CONDITIONS):
            raise StateV3MigrationError(f"incomplete episode pair: {key}")
        if len(paired_action_sha[key]) != 1:
            raise StateV3MigrationError(f"paired action SHA mismatch: {key}")

    states_all = np.concatenate(state_blocks, axis=0)
    actions_all = np.concatenate(action_blocks, axis=0)
    pre_all = np.concatenate(pre_blocks, axis=0)
    if states_all.shape != (EXPECTED_RECORDS, STATE_DIM):
        raise StateV3MigrationError(f"unexpected materialized state shape: {states_all.shape}")
    if actions_all.shape != (EXPECTED_ACTIONS, ACTION_DIM):
        raise StateV3MigrationError(
            f"unexpected materialized action shape: {actions_all.shape}"
        )
    if len(window_rows) != EXPECTED_ACTIONS:
        raise StateV3MigrationError(
            f"unexpected materialized window count: {len(window_rows)}"
        )

    dataset_arrays: Dict[str, np.ndarray] = {
        "states": states_all,
        "actions": actions_all,
        "pre_engagement": pre_all,
        "state_offsets": np.asarray(state_offsets, dtype=np.int64),
        "action_offsets": np.asarray(action_offsets, dtype=np.int64),
        "state_episode_index": np.repeat(
            np.arange(len(state_blocks), dtype=np.int64),
            [block.shape[0] for block in state_blocks],
        ),
        "action_episode_index": np.repeat(
            np.arange(len(action_blocks), dtype=np.int64),
            [block.shape[0] for block in action_blocks],
        ),
        "split_name": strings(episode_metadata["split_name"]),
        "condition_name": strings(episode_metadata["condition_name"]),
        "visible_seed": np.asarray(episode_metadata["visible_seed"], dtype=np.int64),
        "pair_group": strings(episode_metadata["pair_group"]),
        "source_file": strings(episode_metadata["source_file"]),
        "source_pickle_sha256": strings(episode_metadata["source_pickle_sha256"]),
        "source_sidecar_sha256": strings(episode_metadata["source_sidecar_sha256"]),
        "action_sequence_sha256": strings(episode_metadata["action_sequence_sha256"]),
        "success": np.asarray(episode_metadata["success"], dtype=np.bool_),
        "final_fraction": np.asarray(episode_metadata["final_fraction"], dtype=np.float32),
        "engagement_step": np.asarray(episode_metadata["engagement_step"], dtype=np.int64),
        "release_step": np.asarray(episode_metadata["release_step"], dtype=np.int64),
    }

    window_arrays: Dict[str, np.ndarray] = {
        "paper_x": np.stack([row["paper_x"] for row in window_rows]).astype(np.float32),
        "state_action_x": np.stack([row["state_action_x"] for row in window_rows]).astype(np.float32),
        "y_state": np.stack([row["y_state"] for row in window_rows]).astype(np.float32),
        "y_final_state": np.stack([row["y_final_state"] for row in window_rows]).astype(np.float32),
        "y_action": np.stack([row["y_action"] for row in window_rows]).astype(np.float32),
        "state_history_valid_mask": np.stack([row["state_history_valid_mask"] for row in window_rows]),
        "future_valid_mask": np.stack([row["future_valid_mask"] for row in window_rows]),
        "action_history_valid_mask": np.stack([row["action_history_valid_mask"] for row in window_rows]),
        "condition_name": strings([row["condition_name"] for row in window_rows]),
        "visible_seed": np.asarray([row["visible_seed"] for row in window_rows], dtype=np.int64),
        "split_name": strings([row["split_name"] for row in window_rows]),
        "source_file": strings([row["source_file"] for row in window_rows]),
        "source_pickle_sha256": strings([row["source_pickle_sha256"] for row in window_rows]),
        "pair_group": strings([row["pair_group"] for row in window_rows]),
        "episode_index": np.asarray([row["episode_index"] for row in window_rows], dtype=np.int64),
        "window_t": np.asarray([row["window_t"] for row in window_rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in window_rows], dtype=np.bool_),
        "final_fraction": np.asarray([row["final_fraction"] for row in window_rows], dtype=np.float32),
        "engagement_step": np.asarray([row["engagement_step"] for row in window_rows], dtype=np.int64),
        "release_step": np.asarray([row["release_step"] for row in window_rows], dtype=np.int64),
        "pre_engagement": np.asarray([row["pre_engagement"] for row in window_rows], dtype=np.bool_),
    }
    expected_shapes = {
        "paper_x": (PAPER_X_DIM,),
        "state_action_x": (STATE_ACTION_X_DIM,),
        "y_state": (DEFAULT_TF, STATE_DIM),
        "y_final_state": (STATE_DIM,),
        "y_action": (ACTION_DIM,),
    }
    for key, tail in expected_shapes.items():
        value = _finite(window_arrays[key], name=key)
        if value.shape[1:] != tail or value.dtype != np.float32:
            raise StateV3MigrationError(f"invalid state-v3 window array {key}: {value.shape} {value.dtype}")

    legacy_windows = load_npz_strict(legacy_windows_path)
    equivalence = compare_legacy_windows(legacy_windows, window_arrays)
    isolation = split_isolation(window_arrays)
    pair_audit = validate_window_pairs(window_arrays)

    dataset_path = output_root / DATASET_FILE
    windows_path = output_root / WINDOW_FILE
    template_path = output_root / ACTION_TEMPLATE_FILE
    write_npz_once(dataset_path, dataset_arrays)
    write_npz_once(windows_path, window_arrays)
    atomic_copy_once(legacy_template_path, template_path)
    if sha256_file(template_path) != sha256_file(legacy_template_path):
        raise StateV3MigrationError("copied action template changed")

    schema = SchemaV3Manifest()
    schema.validate()
    manifest_payload: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.5 Stage B",
        "schema": "phase314b_r255_stageb_state_v3_dataset_manifest_v1",
        "state_schema": schema.to_dict(),
        "source_state_schema": LEGACY_SCHEMA_VERSION,
        "source_formal_root": "data/phase3_state_v2_slack",
        "source_window_npz_sha256": sha256_file(legacy_windows_path),
        "source_action_template_sha256": sha256_file(legacy_template_path),
        "source_legacy_cache_sha256": EXPECTED_CACHE_SHA256,
        "stage_a_summary_sha256": EXPECTED_STAGE_A_SUMMARY_SHA256,
        "stage_a_migrated_robot_proxy_v3_sha256": EXPECTED_MIGRATED_ROBOT_DIGEST,
        "materialized_robot_proxy_v3_sha256": robot_digest_value,
        "source_code_sha256": {
            relative: sha256_file(root / relative)
            for relative in SOURCE_FILES
        },
        "dataset_file": DATASET_FILE,
        "dataset_sha256": sha256_file(dataset_path),
        "windows_file": WINDOW_FILE,
        "windows_sha256": sha256_file(windows_path),
        "action_template_file": ACTION_TEMPLATE_FILE,
        "action_template_sha256": sha256_file(template_path),
        "episode_count": len(state_blocks),
        "state_record_count": int(states_all.shape[0]),
        "action_record_count": int(actions_all.shape[0]),
        "window_count": len(window_rows),
        "episode_pair_count": len(pair_members),
        "controlled_joint_indices": list(EXPECTED_CONTROLLED_JOINTS),
        "controlled_joint_names": list(
            stage_a_kinematics["controlled_joint_names"]
        ),
        "joint_count": int(stage_a_kinematics["joint_count"]),
        "ee_tip_link": EXPECTED_EE_TIP_LINK,
        "ee_tip_link_name": EXPECTED_EE_TIP_NAME,
        "robot_urdf_relative": str(stage_a_kinematics["robot_urdf_relative"]),
        "robot_urdf_sha256": str(stage_a_kinematics["robot_urdf_sha256"]),
        "stage_a_legacy_payload_sha256": str(
            stage_a_migration["legacy_payload_sha256"]
        ),
        "robot_proxy_source_counts": dict(robot_sources),
        "legacy_equivalence": equivalence,
        "split_isolation": isolation,
        "window_pair_audit": pair_audit,
        "input_canonicalization": False,
        "contains_simulator_bead_velocity": False,
        "missing_value_policy": "reject_record",
        "write_once": True,
        "old_raw_modified": False,
        "old_windows_modified": False,
        "old_cache_modified": False,
    }
    atomic_write_once(output_root / MANIFEST_FILE, stable_json_bytes(manifest_payload))
    return manifest_payload


def directory_manifest(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def compare_build_directories(left: Path, right: Path) -> Dict[str, Any]:
    left_manifest = directory_manifest(left)
    right_manifest = directory_manifest(right)
    if left_manifest != right_manifest:
        names = sorted(set(left_manifest).union(right_manifest))
        differences = [
            name
            for name in names
            if left_manifest.get(name) != right_manifest.get(name)
        ]
        raise StateV3MigrationError(f"independent state-v3 builds differ: {differences[:20]}")
    return {
        "exact": True,
        "file_count": len(left_manifest),
        "file_sha256": left_manifest,
    }
