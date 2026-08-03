"""Phase3.14b-r2.6.0 Stage A: new data-universe genesis and seal.

All r2.5.x raw data, training caches, selection surfaces and probe files are
permanently abandoned.  This stage creates a new, disjoint data universe from
frozen source code and simulator semantics.  It generates paired free/hidden
rollouts, builds one state-v2 window archive per governance role, audits the
new data, creates deterministic export archives and writes a cryptographic
seal.

No model is fitted or evaluated.  Selection holdout, frozen probe and final
evaluation counts remain zero.  The output remains scientifically BLOCKED
until the deterministic export archives are copied to independent persistent
storage and a later stage records that backup attestation.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3.action_codec import ExecutableActionCodec
from ccda_phase3.data_io import load_episode, save_action_template
from ccda_phase3.schema_v2 import (
    ACTION_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    ENVIRONMENT_VERSION,
    FORMAL_CONDITIONS,
    FORMAL_TASK_NAME,
    PAPER_X_DIM,
    SCHEMA_VERSION,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    build_window,
)

PHASE = "Phase3.14b-r2.6.0 Stage A"
SCHEMA = "phase314b_r260_stagea_data_universe_genesis_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"
DATASET_SCHEMA = "phase314b_r260_data_universe_v1"

BASE_HEAD = "912870644734e49b9c0f34b6e87741c7fb63bda5"
BASE_IMPLEMENTATION = "bb08da7a465987585d7c41f37c0dcac983f71730"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"
STAGEJ_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stagej_prevalence_anchored_risk_repair_blocked_summary.json"
)
STAGEJ_BLOCKED_REPORT_SHA256 = (
    "58b97cadf760f9b396fe242d0c3a92d053008b111c6db796acb87be20e80a35a"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.6.0 Stage A: generate and seal new data universe"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.6.0 Stage A data-universe evidence"
)
BLOCKED_SUBJECT = "Record Phase3.14b-r2.6.0 Stage A blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r260_stagea_data_universe_genesis.py"),
    ("A", "scripts/phase3_14b_r260_stagea_worker.py"),
    ("A", "scripts/phase3_14b_r260_stagea_execute.py"),
    ("A", "tests/test_phase3_14b_r260_stagea_data_universe_genesis.py"),
)

DATASET_RELATIVE_ROOT = "data/phase3_14b_r260_data_universe_v1"
RAW_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/raw"
WINDOWS_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/windows"
RECEIPTS_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/receipts"
EXPORTS_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/exports"
ROLE_MANIFEST_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/role_manifests"
SOURCE_LOCK_RELATIVE = DATASET_RELATIVE_ROOT + "/source_lock.json"
INVENTORY_RELATIVE = DATASET_RELATIVE_ROOT + "/artifact_inventory.json"
SEAL_RELATIVE = DATASET_RELATIVE_ROOT + "/seal_manifest.json"
ACTION_TEMPLATE_RELATIVE = WINDOWS_RELATIVE_ROOT + "/action_template.pkl"
ATTEMPT_MARKER_RELATIVE = DATASET_RELATIVE_ROOT + "/attempt_started.json"

SUCCESS_REPORT = "reports/phase3_14b_r260_stagea_data_universe_genesis_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r260_stagea_data_universe_genesis_blocked_summary.json"

WRITE_AHEAD_NAME = ".phase314b_r260_stagea_write_ahead"
LEGACY_STAGEJ_WRITE_AHEAD_NAME = ".phase314b_r259_stagej_write_ahead"

EXPECTED_HZ = 480
EXPECTED_CONDITIONS = tuple(FORMAL_CONDITIONS)
EXPECTED_STATE_DIM = 87
EXPECTED_PAPER_X_DIM = 261
EXPECTED_STATE_ACTION_X_DIM = 303
EXPECTED_FUTURE_SHAPE = (4, 87)
EXPECTED_ACTION_DIM = 14


@dataclass(frozen=True)
class RoleSpec:
    name: str
    seed_start: int
    seed_count: int
    governance_role: str

    @property
    def seed_stop(self) -> int:
        return int(self.seed_start + self.seed_count)

    def seeds(self) -> Tuple[int, ...]:
        return tuple(range(int(self.seed_start), int(self.seed_stop)))


ROLE_SPECS: Tuple[RoleSpec, ...] = (
    RoleSpec("objective_train", 900000, 320, "fit_and_nested_group_oof_only"),
    RoleSpec("selection_holdout", 910000, 96, "one_shot_model_selection_only"),
    RoleSpec("frozen_probe", 920000, 96, "one_shot_pre_final_probe_only"),
    RoleSpec("final_evaluation", 930000, 128, "one_shot_final_evaluation_only"),
)
ROLE_BY_NAME = {spec.name: spec for spec in ROLE_SPECS}
TOTAL_VISIBLE_SEEDS = sum(spec.seed_count for spec in ROLE_SPECS)
TOTAL_EPISODES = 2 * TOTAL_VISIBLE_SEEDS

SOURCE_LOCK_PATHS: Tuple[str, ...] = (
    "ccda_phase3/action_codec.py",
    "ccda_phase3/data_io.py",
    "ccda_phase3/schema_v2.py",
    "scripts/phase3_13_generate_raw.py",
    "scripts/phase3_13_runtime.py",
    "ccda_phase3/phase314b_r260_stagea_data_universe_genesis.py",
    "scripts/phase3_14b_r260_stagea_worker.py",
    "scripts/phase3_14b_r260_stagea_execute.py",
    "external/deformable-ravens/ravens/tasks/ccda_slack_cable_v2.py",
    "external/deformable-ravens/ravens/tasks/__init__.py",
    "external/deformable-ravens/ravens/environment.py",
)

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "legacy_r255_cache_loaded",
    "legacy_r256_cache_loaded",
    "legacy_stagei_data_loaded",
    "risk_fit_run",
    "direction_fit_run",
    "candidate_generation_run",
    "candidate_evaluation_run",
    "selection_holdout_model_evaluation_run",
    "frozen_probe_model_evaluation_run",
    "final_evaluation_model_evaluation_run",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_saved",
)


class StageAError(RuntimeError):
    """Fail-closed r2.6.0 Stage-A error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = compact_json_bytes(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    )
    return sha256_bytes(header + b"\0" + array.tobytes())


def sha256_strings(values: Iterable[str]) -> str:
    return sha256_bytes(compact_json_bytes(sorted(str(value) for value in values)))


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageAError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageAError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    rows: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageAError("unexpected commit record: {!r}".format(line))
        rows.append((fields[0], fields[1]))
    return tuple(sorted(rows))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageAError("{} worktree is dirty".format(label))


def role_contract() -> Mapping[str, Any]:
    all_seeds: List[int] = []
    records = []
    for spec in ROLE_SPECS:
        seeds = list(spec.seeds())
        all_seeds.extend(seeds)
        records.append(
            {
                "name": spec.name,
                "governance_role": spec.governance_role,
                "seed_start": spec.seed_start,
                "seed_stop_exclusive": spec.seed_stop,
                "seed_count": spec.seed_count,
                "seed_list_sha256": sha256_bytes(compact_json_bytes(seeds)),
                "episode_count": 2 * spec.seed_count,
            }
        )
    if len(set(all_seeds)) != len(all_seeds):
        raise StageAError("r2.6.0 seed ranges overlap")
    legacy_ranges = (
        range(400000, 400256),
        range(410000, 410064),
        range(420000, 420128),
        range(430000, 430128),
    )
    if any(set(all_seeds).intersection(value) for value in legacy_ranges):
        raise StageAError("r2.6.0 seed range overlaps an abandoned r2.5.x range")
    return {
        "dataset_schema": DATASET_SCHEMA,
        "roles": records,
        "total_visible_seeds": TOTAL_VISIBLE_SEEDS,
        "total_pair_groups": TOTAL_VISIBLE_SEEDS,
        "total_episodes": TOTAL_EPISODES,
        "all_seed_list_sha256": sha256_bytes(compact_json_bytes(all_seeds)),
        "legacy_seed_overlap": False,
    }


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-J parent": (_git(repo, "rev-parse", BASE_HEAD + "^"), BASE_IMPLEMENTATION),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
        "submodule worktree": (
            _git(repo / "external/deformable-ravens", "rev-parse", "HEAD"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageAError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageAError("Stage-A implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageAError("Stage-A implementation population changed")
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageAError("Stage-A source is missing: {}".format(relative))
        if _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative)) != path.read_bytes():
            raise StageAError("Stage-A source differs from implementation commit")
    blocked_report = repo / STAGEJ_BLOCKED_REPORT
    if not blocked_report.is_file():
        raise StageAError("Stage-J blocked report is missing")
    if sha256_file(blocked_report) != STAGEJ_BLOCKED_REPORT_SHA256:
        raise StageAError("Stage-J blocked report SHA changed")
    if _git_bytes(repo, "show", "{}:{}".format(BASE_HEAD, STAGEJ_BLOCKED_REPORT)) != blocked_report.read_bytes():
        raise StageAError("Stage-J blocked report differs from committed evidence")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(repo / "external/deformable-ravens", "submodule")
    if (repo / DATASET_RELATIVE_ROOT).exists():
        raise StageAError("new r2.6.0 data-universe root already exists")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageAError("Stage-A terminal report already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "abandoned_stagej_implementation": BASE_IMPLEMENTATION,
        "abandoned_stagej_blocked_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def build_source_lock(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    records: Dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageAError("source-lock path missing: {}".format(relative))
        records[relative] = sha256_file(path)
    return {
        "schema": "phase314b_r260_stagea_source_lock_v1",
        "main_commit": implementation_commit,
        "submodule_commit": EXPECTED_SUBMODULE,
        "source_sha256": records,
        "task": FORMAL_TASK_NAME,
        "conditions": list(EXPECTED_CONDITIONS),
        "schema_version": SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "hz": EXPECTED_HZ,
        "role_contract": role_contract(),
        "legacy_r259_data_identity_carried_forward": False,
    }


def _legacy_generation_module(root: Path) -> Any:
    scripts = str(Path(root).resolve() / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("phase3_13_generate_raw")


def _strict_manifest(manifest: Mapping[str, Any], *, role: str, seed: int, implementation_commit: str) -> None:
    expected = {
        "dataset_schema": DATASET_SCHEMA,
        "environment_semantics_version": ENVIRONMENT_VERSION,
        "observation_schema_version": SCHEMA_VERSION,
        "task": FORMAL_TASK_NAME,
        "visible_seed": int(seed),
        "pair_group": "phase314b_r260_{}_seed_{}".format(role, seed),
        "role": role,
        "action_source_condition": "free",
        "main_commit": implementation_commit,
        "submodule_commit": EXPECTED_SUBMODULE,
        "contains_simulator_bead_velocity": False,
        "input_canonicalization": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise StageAError("pair manifest field changed: {}".format(key))
    if manifest.get("condition") not in EXPECTED_CONDITIONS:
        raise StageAError("pair manifest condition changed")
    if int(manifest.get("num_actions", 0)) <= 0:
        raise StageAError("pair manifest contains no actions")


def generate_pair_job(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    root = Path(str(payload["root"])).resolve()
    dataset_root = Path(str(payload["dataset_root"])).resolve()
    write_ahead = Path(str(payload["write_ahead"])).resolve()
    role = str(payload["role"])
    seed = int(payload["seed"])
    implementation_commit = str(payload["implementation_commit"])
    if role not in ROLE_BY_NAME or seed not in ROLE_BY_NAME[role].seeds():
        raise StageAError("generation job outside frozen role contract")
    final_dir = dataset_root / "raw" / role / "seed_{}".format(seed)
    if final_dir.exists():
        raise StageAError("formal seed output already exists: {}".format(final_dir))
    staging_parent = write_ahead / "pair_staging" / role
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / "seed_{}.{}".format(seed, os.getpid())
    if staging.exists():
        raise StageAError("worker staging path already exists")
    staging.mkdir(parents=True)
    generator = _legacy_generation_module(root)
    pair_group = "phase314b_r260_{}_seed_{}".format(role, seed)
    try:
        free_infos, free_actions, free_last, free_success, free_reset = generator.execute_free(
            root, seed, pair_group, EXPECTED_HZ
        )
        hidden_infos, hidden_actions, hidden_last, hidden_success, hidden_reset = generator.execute_replay(
            root, seed, pair_group, EXPECTED_HZ, free_actions
        )
        if len(free_actions) != len(hidden_actions):
            raise StageAError("paired action count mismatch")
        action_digest, vectors = generator.action_sha(free_actions)
        hidden_digest, hidden_vectors = generator.action_sha(hidden_actions)
        if action_digest != hidden_digest:
            raise StageAError("paired action sequence SHA mismatch")
        if not np.array_equal(np.stack(vectors), np.stack(hidden_vectors)):
            raise StageAError("paired executable action vectors differ")
        initial_free = np.asarray(free_infos[0]["extras"]["bead_positions"], dtype=np.float64)
        initial_hidden = np.asarray(hidden_infos[0]["extras"]["bead_positions"], dtype=np.float64)
        if not np.array_equal(initial_free, initial_hidden):
            raise StageAError("paired initial geometry mismatch")
        condition_rows = (
            ("free", free_infos, free_actions, free_last, free_success, free_reset),
            (
                "hidden_slack_breakaway_pin_v2",
                hidden_infos,
                hidden_actions,
                hidden_last,
                hidden_success,
                hidden_reset,
            ),
        )
        files: List[Mapping[str, Any]] = []
        for condition, infos, actions, last_info, success, reset in condition_rows:
            manifest: Dict[str, Any] = {
                "dataset_schema": DATASET_SCHEMA,
                "environment_semantics_version": ENVIRONMENT_VERSION,
                "observation_schema_version": SCHEMA_VERSION,
                "task": FORMAL_TASK_NAME,
                "condition": condition,
                "visible_seed": seed,
                "pair_group": pair_group,
                "role": role,
                "governance_role": ROLE_BY_NAME[role].governance_role,
                "action_source_condition": "free",
                "action_sequence_sha256": action_digest,
                "main_commit": implementation_commit,
                "submodule_commit": EXPECTED_SUBMODULE,
                "contains_simulator_bead_velocity": False,
                "input_canonicalization": False,
                "num_actions": len(actions),
                "settle_steps": reset["settle_steps"],
            }
            _strict_manifest(
                manifest,
                role=role,
                seed=seed,
                implementation_commit=implementation_commit,
            )
            episode = {
                "infos": infos,
                "last_info": last_info,
                "actions": actions,
                "action_vectors": np.stack(vectors[: len(actions)]).astype(np.float32),
                "success": bool(success),
                "manifest": manifest,
            }
            episode_path = staging / "{}.pkl".format(condition)
            with episode_path.open("xb") as handle:
                pickle.dump(episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            manifest_path = staging / "{}.manifest.json".format(condition)
            atomic_write_once(manifest_path, stable_json_bytes(manifest))
            files.extend(
                [
                    {
                        "name": episode_path.name,
                        "size_bytes": episode_path.stat().st_size,
                        "sha256": sha256_file(episode_path),
                    },
                    {
                        "name": manifest_path.name,
                        "size_bytes": manifest_path.stat().st_size,
                        "sha256": sha256_file(manifest_path),
                    },
                ]
            )
        receipt: Dict[str, Any] = {
            "schema": "phase314b_r260_stagea_pair_receipt_v1",
            "role": role,
            "visible_seed": seed,
            "pair_group": pair_group,
            "action_sequence_sha256": action_digest,
            "main_commit": implementation_commit,
            "submodule_commit": EXPECTED_SUBMODULE,
            "files": sorted(files, key=lambda item: str(item["name"])),
        }
        receipt["receipt_sha256"] = sha256_bytes(stable_json_bytes(receipt))
        atomic_write_once(staging / "pair_receipt.json", stable_json_bytes(receipt))
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(staging), str(final_dir))
        descriptor = os.open(str(final_dir.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(str(staging), ignore_errors=True)


def validate_pair_directory(
    pair_dir: Path,
    *,
    role: str,
    seed: int,
    implementation_commit: str,
) -> Mapping[str, Any]:
    directory = Path(pair_dir)
    expected_names = {
        "free.pkl",
        "free.manifest.json",
        "hidden_slack_breakaway_pin_v2.pkl",
        "hidden_slack_breakaway_pin_v2.manifest.json",
        "pair_receipt.json",
    }
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != expected_names:
        raise StageAError("pair file population changed: {}".format(directory))
    receipt = load_json(directory / "pair_receipt.json")
    expected_receipt = sha256_bytes(
        stable_json_bytes({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    )
    if receipt.get("receipt_sha256") != expected_receipt:
        raise StageAError("pair receipt self-hash changed")
    if receipt.get("role") != role or int(receipt.get("visible_seed", -1)) != seed:
        raise StageAError("pair receipt role/seed changed")
    receipt_records = receipt.get("files")
    if not isinstance(receipt_records, Sequence) or len(receipt_records) != 4:
        raise StageAError("pair receipt file population changed")
    expected_record_names = expected_names - {"pair_receipt.json"}
    if {str(record.get("name")) for record in receipt_records if isinstance(record, Mapping)} != expected_record_names:
        raise StageAError("pair receipt filenames changed")
    for record in receipt_records:
        if not isinstance(record, Mapping):
            raise StageAError("pair receipt file record is invalid")
        path = directory / str(record["name"])
        if int(record["size_bytes"]) != path.stat().st_size:
            raise StageAError("pair receipt file size changed")
        if str(record["sha256"]) != sha256_file(path):
            raise StageAError("pair receipt file SHA changed")

    action_hashes = set()
    action_counts = set()
    for condition in EXPECTED_CONDITIONS:
        manifest = load_json(directory / "{}.manifest.json".format(condition))
        _strict_manifest(
            manifest,
            role=role,
            seed=seed,
            implementation_commit=implementation_commit,
        )
        action_hashes.add(str(manifest["action_sequence_sha256"]))
        action_counts.add(int(manifest["num_actions"]))
        if manifest["condition"] != condition:
            raise StageAError("condition manifest filename mismatch")
        with (directory / "{}.pkl".format(condition)).open("rb") as handle:
            episode = pickle.load(handle)
        if not isinstance(episode, Mapping) or episode.get("manifest") != dict(manifest):
            raise StageAError("episode/manifest provenance mismatch")
        vectors = np.asarray(episode.get("action_vectors"), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (int(manifest["num_actions"]), EXPECTED_ACTION_DIM):
            raise StageAError("episode action-vector shape changed")
        vector_sha = hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()
        if vector_sha != str(manifest["action_sequence_sha256"]):
            raise StageAError("episode action-vector SHA changed")
    if len(action_hashes) != 1 or len(action_counts) != 1:
        raise StageAError("paired manifest action identity changed")
    return {
        "role": role,
        "visible_seed": seed,
        "pair_group": receipt["pair_group"],
        "action_sequence_sha256": next(iter(action_hashes)),
        "num_actions": next(iter(action_counts)),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _string_array(values: Iterable[str]) -> np.ndarray:
    rendered = [str(value) for value in values]
    width = max([1] + [len(value) for value in rendered])
    return np.asarray(rendered, dtype="<U{}".format(width))


def _deterministic_npz_payload(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            array = np.asarray(arrays[key])
            if array.dtype.kind == "O":
                raise StageAError("object dtype is forbidden: {}".format(key))
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(
                "{}.npy".format(key), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    payload = _deterministic_npz_payload(arrays)
    atomic_write_once(path, payload)
    return sha256_bytes(payload)


def load_npz_strict(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise StageAError("object dtype found in sealed windows")
    return arrays


def build_role_windows(
    *,
    root: Path,
    role: str,
    codec: Optional[ExecutableActionCodec],
) -> Tuple[Mapping[str, np.ndarray], ExecutableActionCodec, Mapping[str, int]]:
    repo = Path(root).resolve()
    pair_root = repo / RAW_RELATIVE_ROOT / role
    rows: List[Mapping[str, Any]] = []
    source_counts: MutableMapping[str, int] = {}
    active_codec = codec
    for seed in ROLE_BY_NAME[role].seeds():
        pair_dir = pair_root / "seed_{}".format(seed)
        for condition in EXPECTED_CONDITIONS:
            episode = load_episode(
                condition,
                pair_dir / "{}.pkl".format(condition),
                codec=active_codec,
            )
            active_codec = episode["codec"]
            count = min(len(episode["actions"]), len(episode["states"]) - 1)
            if count <= 0:
                raise StageAError("episode has no buildable window")
            for current in range(count):
                window = build_window(
                    states=episode["states"],
                    action_vectors=episode["actions"],
                    current_index=current,
                    th=DEFAULT_TH,
                    tf=DEFAULT_TF,
                )
                for source in episode["robot_pose_proxy_sources"][: current + 1]:
                    source_counts[str(source)] = source_counts.get(str(source), 0) + 1
                rows.append(
                    {
                        **window,
                        "condition_name": condition,
                        "visible_seed": seed,
                        "split_name": role,
                        "source_file": (pair_dir / "{}.pkl".format(condition)).relative_to(repo).as_posix(),
                        "pair_group": episode["pair_group"],
                        "window_t": int(current),
                        "success": bool(episode["success"]),
                        "final_fraction": float(episode["final_fraction"]),
                        "engagement_step": int(episode["engagement_step"]),
                        "release_step": int(episode["release_step"]),
                        "pre_engagement": bool(episode["pre_engagement"][current]),
                    }
                )
    if active_codec is None or not rows:
        raise StageAError("role produced no windows")
    arrays: Dict[str, np.ndarray] = {
        "paper_x": np.stack([row["paper_x"] for row in rows]).astype(np.float32),
        "state_action_x": np.stack([row["state_action_x"] for row in rows]).astype(np.float32),
        "y_state": np.stack([row["y_state"] for row in rows]).astype(np.float32),
        "y_final_state": np.stack([row["y_final_state"] for row in rows]).astype(np.float32),
        "y_action": np.stack([row["y_action"] for row in rows]).astype(np.float32),
        "condition_name": _string_array(row["condition_name"] for row in rows),
        "visible_seed": np.asarray([row["visible_seed"] for row in rows], dtype=np.int64),
        "split_name": _string_array(row["split_name"] for row in rows),
        "source_file": _string_array(row["source_file"] for row in rows),
        "pair_group": _string_array(row["pair_group"] for row in rows),
        "window_t": np.asarray([row["window_t"] for row in rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in rows], dtype=np.bool_),
        "final_fraction": np.asarray([row["final_fraction"] for row in rows], dtype=np.float32),
        "engagement_step": np.asarray([row["engagement_step"] for row in rows], dtype=np.int64),
        "release_step": np.asarray([row["release_step"] for row in rows], dtype=np.int64),
        "pre_engagement": np.asarray([row["pre_engagement"] for row in rows], dtype=np.bool_),
    }
    validate_role_arrays(role, arrays)
    return arrays, active_codec, dict(sorted(source_counts.items()))


def validate_role_arrays(role: str, arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "paper_x",
        "state_action_x",
        "y_state",
        "y_final_state",
        "y_action",
        "condition_name",
        "visible_seed",
        "split_name",
        "source_file",
        "pair_group",
        "window_t",
        "success",
        "final_fraction",
        "engagement_step",
        "release_step",
        "pre_engagement",
    }
    if set(arrays) != required:
        raise StageAError("role array population changed")
    rows = int(arrays["paper_x"].shape[0])
    expected_shapes = {
        "paper_x": (rows, EXPECTED_PAPER_X_DIM),
        "state_action_x": (rows, EXPECTED_STATE_ACTION_X_DIM),
        "y_state": (rows,) + EXPECTED_FUTURE_SHAPE,
        "y_final_state": (rows, EXPECTED_STATE_DIM),
        "y_action": (rows, EXPECTED_ACTION_DIM),
    }
    for key, shape in expected_shapes.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float32:
            raise StageAError("role array shape/dtype changed: {}".format(key))
        if not np.all(np.isfinite(value)):
            raise StageAError("role float array is non-finite: {}".format(key))
    if rows < 2 * ROLE_BY_NAME[role].seed_count:
        raise StageAError("role window count is implausibly small")
    if set(arrays["visible_seed"].astype(np.int64).tolist()) != set(ROLE_BY_NAME[role].seeds()):
        raise StageAError("role window seed coverage changed")
    if set(arrays["condition_name"].astype(str).tolist()) != set(EXPECTED_CONDITIONS):
        raise StageAError("role condition coverage changed")
    if set(arrays["split_name"].astype(str).tolist()) != {role}:
        raise StageAError("role split label changed")
    if len(set(arrays["pair_group"].astype(str).tolist())) != ROLE_BY_NAME[role].seed_count:
        raise StageAError("role pair-group count changed")
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise StageAError("object dtype found in role arrays")


def _row_hash(arrays: Mapping[str, np.ndarray], index: int) -> str:
    digest = hashlib.sha256()
    for key in ("paper_x", "state_action_x", "y_state", "y_action"):
        digest.update(np.ascontiguousarray(arrays[key][index]).tobytes())
    return digest.hexdigest()


def _audit_module(root: Path) -> Any:
    scripts = str(Path(root).resolve() / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("phase3_13_audit_dataset")


def audit_roles(role_arrays: Mapping[str, Mapping[str, np.ndarray]], root: Path) -> Mapping[str, Any]:
    audit = _audit_module(root)
    all_seed_sets = {role: set(arrays["visible_seed"].astype(np.int64).tolist()) for role, arrays in role_arrays.items()}
    all_group_sets = {role: set(arrays["pair_group"].astype(str).tolist()) for role, arrays in role_arrays.items()}
    all_row_hashes = {
        role: {_row_hash(arrays, index) for index in range(arrays["paper_x"].shape[0])}
        for role, arrays in role_arrays.items()
    }
    overlap: Dict[str, int] = {}
    names = [spec.name for spec in ROLE_SPECS]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap["{}_{}_seeds".format(left, right)] = len(all_seed_sets[left] & all_seed_sets[right])
            overlap["{}_{}_groups".format(left, right)] = len(all_group_sets[left] & all_group_sets[right])
            overlap["{}_{}_row_hashes".format(left, right)] = len(all_row_hashes[left] & all_row_hashes[right])
    if any(overlap.values()):
        raise StageAError("r2.6.0 governance-role leakage detected")

    role_results: Dict[str, Any] = {}
    for role, arrays in role_arrays.items():
        conditions = arrays["condition_name"].astype(str)
        seeds = arrays["visible_seed"].astype(np.int64)
        times = arrays["window_t"].astype(np.int64)
        pre = arrays["pre_engagement"].astype(bool)
        index = {
            (str(conditions[i]), int(seeds[i]), int(times[i])): i
            for i in range(len(seeds))
        }
        hiddenness_rows = []
        for feature in ("paper_x", "state_action_x"):
            x_rows: List[np.ndarray] = []
            y_rows: List[int] = []
            group_rows: List[int] = []
            for seed in ROLE_BY_NAME[role].seeds():
                seed_times = sorted(set(times[seeds == seed].tolist()))
                for current in seed_times:
                    free = index.get(("free", seed, current))
                    hidden = index.get(("hidden_slack_breakaway_pin_v2", seed, current))
                    if free is None or hidden is None or not pre[hidden]:
                        continue
                    x_rows.extend([arrays[feature][free], arrays[feature][hidden]])
                    y_rows.extend([0, 1])
                    group_rows.extend([seed, seed])
            if not x_rows:
                raise StageAError("role has no pre-engagement paired rows")
            result = audit.grouped_hiddenness(
                np.asarray(x_rows, dtype=np.float32),
                np.asarray(y_rows, dtype=np.int64),
                np.asarray(group_rows, dtype=np.int64),
            )
            hiddenness_rows.append({"feature": feature, "paired_rows": len(y_rows), **result})
        hiddenness_pass = all(
            row["roc_auc"] <= 0.60
            and row["roc_auc_ci_high"] < 0.65
            and row["balanced_accuracy"] <= 0.60
            for row in hiddenness_rows
        )

        eligible = 0
        diverged = 0
        for seed in ROLE_BY_NAME[role].seeds():
            seed_times = sorted(set(times[seeds == seed].tolist()))
            for current in seed_times:
                free = index.get(("free", seed, current))
                hidden = index.get(("hidden_slack_breakaway_pin_v2", seed, current))
                if free is None or hidden is None:
                    continue
                scale = float(np.std(np.concatenate([arrays["paper_x"][free], arrays["paper_x"][hidden]])))
                prefix = float(
                    np.mean(np.abs(arrays["paper_x"][free] - arrays["paper_x"][hidden]))
                    / max(scale, 1.0e-8)
                )
                final_distance = audit.chamfer(
                    arrays["y_final_state"][free][:48].reshape(24, 2),
                    arrays["y_final_state"][hidden][:48].reshape(24, 2),
                )
                progress = abs(float(arrays["final_fraction"][free] - arrays["final_fraction"][hidden]))
                success_difference = bool(arrays["success"][free] != arrays["success"][hidden])
                row_eligible = prefix <= 0.05
                row_diverged = row_eligible and (
                    final_distance >= 0.003 or progress >= 0.03 or success_difference
                )
                eligible += int(row_eligible)
                diverged += int(row_diverged)
        divergence_rate = diverged / max(1, eligible)
        hidden_episode_engaged = []
        hidden_episode_released = []
        free_episode_engaged = []
        pair_root = Path(root) / RAW_RELATIVE_ROOT / role
        for seed in ROLE_BY_NAME[role].seeds():
            hidden_episode = load_episode(
                "hidden_slack_breakaway_pin_v2",
                pair_root / "seed_{}".format(seed) / "hidden_slack_breakaway_pin_v2.pkl",
            )
            free_episode = load_episode(
                "free", pair_root / "seed_{}".format(seed) / "free.pkl"
            )
            hidden_episode_engaged.append(hidden_episode["engagement_step"] >= 0)
            hidden_episode_released.append(hidden_episode["release_step"] >= 0)
            free_episode_engaged.append(free_episode["engagement_step"] >= 0)
        engagement_rate = float(np.mean(hidden_episode_engaged))
        release_rate = float(np.mean(hidden_episode_released))
        engagement_pass = (
            engagement_rate >= 0.70
            and release_rate >= 0.50
            and not any(free_episode_engaged)
        )
        if not hiddenness_pass:
            raise StageAError("pre-engagement hidden condition is observably leaked: {}".format(role))
        if divergence_rate < 0.30:
            raise StageAError("future branch divergence is insufficient: {}".format(role))
        if not engagement_pass:
            raise StageAError("engagement/release coverage failed: {}".format(role))
        role_results[role] = {
            "window_rows": int(arrays["paper_x"].shape[0]),
            "visible_seeds": len(all_seed_sets[role]),
            "pair_groups": len(all_group_sets[role]),
            "hiddenness": hiddenness_rows,
            "hiddenness_pass": hiddenness_pass,
            "eligible_paired_windows": eligible,
            "diverged_paired_windows": diverged,
            "future_divergence_rate": divergence_rate,
            "future_divergence_pass": True,
            "hidden_engagement_rate": engagement_rate,
            "hidden_release_rate": release_rate,
            "engagement_coverage_pass": engagement_pass,
        }
    return {
        "role_overlap": overlap,
        "role_isolation_pass": True,
        "role_results": role_results,
    }


def build_role_manifest(
    *,
    root: Path,
    role: str,
    arrays: Mapping[str, np.ndarray],
    npz_path: Path,
    robot_sources: Mapping[str, int],
) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    records = []
    for seed in ROLE_BY_NAME[role].seeds():
        pair = validate_pair_directory(
            repo / RAW_RELATIVE_ROOT / role / "seed_{}".format(seed),
            role=role,
            seed=seed,
            implementation_commit=_git(repo, "rev-parse", "HEAD"),
        )
        records.append(pair)
    payload = {
        "schema": "phase314b_r260_stagea_role_manifest_v1",
        "dataset_schema": DATASET_SCHEMA,
        "role": role,
        "governance_role": ROLE_BY_NAME[role].governance_role,
        "seed_start": ROLE_BY_NAME[role].seed_start,
        "seed_stop_exclusive": ROLE_BY_NAME[role].seed_stop,
        "seed_count": ROLE_BY_NAME[role].seed_count,
        "episode_count": 2 * ROLE_BY_NAME[role].seed_count,
        "pair_group_count": ROLE_BY_NAME[role].seed_count,
        "window_row_count": int(arrays["paper_x"].shape[0]),
        "window_npz": npz_path.relative_to(repo).as_posix(),
        "window_npz_sha256": sha256_file(npz_path),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "array_sha256": {key: sha256_array(value) for key, value in arrays.items()},
        "pair_receipt_population_sha256": sha256_strings(
            str(record["receipt_sha256"]) for record in records
        ),
        "action_sequence_population_sha256": sha256_strings(
            str(record["action_sequence_sha256"]) for record in records
        ),
        "robot_proxy_sources": dict(robot_sources),
        "model_fit_count": 0,
        "model_evaluation_count": 0,
    }
    payload["manifest_sha256"] = sha256_bytes(stable_json_bytes(payload))
    return payload


def _inventory_paths(dataset_root: Path) -> List[Path]:
    root = Path(dataset_root).resolve()
    excluded = {
        root / "artifact_inventory.json",
        root / "seal_manifest.json",
    }
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path not in excluded
            and not path.name.endswith(".tmp")
            and "exports" not in path.relative_to(root).parts
        ],
        key=lambda value: value.relative_to(root).as_posix(),
    )


def build_inventory(dataset_root: Path) -> Mapping[str, Any]:
    root = Path(dataset_root).resolve()
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in _inventory_paths(root)
    ]
    if not records:
        raise StageAError("data-universe inventory is empty")
    payload = {
        "schema": "phase314b_r260_stagea_artifact_inventory_v1",
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "records": records,
    }
    payload["inventory_sha256"] = sha256_bytes(stable_json_bytes(payload))
    return payload


def validate_inventory(dataset_root: Path, inventory: Mapping[str, Any]) -> None:
    root = Path(dataset_root).resolve()
    records = inventory.get("records")
    if not isinstance(records, Sequence) or not records:
        raise StageAError("inventory records missing")
    observed_paths = []
    for record in records:
        if not isinstance(record, Mapping):
            raise StageAError("invalid inventory record")
        relative = str(record["path"])
        path = root / relative
        if not path.is_file():
            raise StageAError("inventory file missing: {}".format(relative))
        if int(record["size_bytes"]) != path.stat().st_size:
            raise StageAError("inventory size changed: {}".format(relative))
        if str(record["sha256"]) != sha256_file(path):
            raise StageAError("inventory SHA changed: {}".format(relative))
        observed_paths.append(relative)
    expected_paths = [path.relative_to(root).as_posix() for path in _inventory_paths(root)]
    if observed_paths != expected_paths:
        raise StageAError("inventory path population changed")
    base = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if inventory.get("inventory_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageAError("inventory self-hash changed")


def deterministic_tar_gz(
    *,
    output: Path,
    base_root: Path,
    relative_paths: Sequence[Path],
) -> str:
    target = Path(output)
    if target.exists():
        raise StageAError("export archive already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0, filename="") as gzip_handle:
                with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                    files: List[Path] = []
                    for relative in relative_paths:
                        path = Path(base_root) / relative
                        if path.is_dir():
                            files.extend(item for item in path.rglob("*") if item.is_file())
                        elif path.is_file():
                            files.append(path)
                        else:
                            raise StageAError("export input missing: {}".format(path))
                    unique = sorted(set(files), key=lambda value: value.relative_to(base_root).as_posix())
                    for path in unique:
                        name = path.relative_to(base_root).as_posix()
                        info = tarfile.TarInfo(name=name)
                        info.size = path.stat().st_size
                        info.mode = 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(target)


def build_exports(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    dataset_root = repo / DATASET_RELATIVE_ROOT
    records = []
    for spec in ROLE_SPECS:
        output = repo / EXPORTS_RELATIVE_ROOT / "{}.tar.gz".format(spec.name)
        digest = deterministic_tar_gz(
            output=output,
            base_root=dataset_root,
            relative_paths=(
                Path("raw") / spec.name,
                Path("windows") / "{}.npz".format(spec.name),
                Path("role_manifests") / "{}.json".format(spec.name),
                Path("source_lock.json"),
                Path("windows/action_template.pkl"),
            ),
        )
        records.append(
            {
                "role": spec.name,
                "path": output.relative_to(repo).as_posix(),
                "size_bytes": output.stat().st_size,
                "sha256": digest,
            }
        )
    metadata_output = repo / EXPORTS_RELATIVE_ROOT / "metadata_and_receipts.tar.gz"
    metadata_digest = deterministic_tar_gz(
        output=metadata_output,
        base_root=dataset_root,
        relative_paths=(
            Path("receipts"),
            Path("role_manifests"),
            Path("source_lock.json"),
            Path("artifact_inventory.json"),
            Path("windows/action_template.pkl"),
        ),
    )
    records.append(
        {
            "role": "metadata_and_receipts",
            "path": metadata_output.relative_to(repo).as_posix(),
            "size_bytes": metadata_output.stat().st_size,
            "sha256": metadata_digest,
        }
    )
    return {
        "schema": "phase314b_r260_stagea_export_manifest_v1",
        "records": records,
        "archive_count": len(records),
        "all_archive_identity_sha256": sha256_bytes(stable_json_bytes(records)),
        "external_copy_count_attested": 0,
    }


def build_seal(
    *,
    repository: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    role_manifests: Mapping[str, Mapping[str, Any]],
    audit: Mapping[str, Any],
    inventory: Mapping[str, Any],
    exports: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload: Dict[str, Any] = {
        "schema": "phase314b_r260_stagea_data_universe_seal_v1",
        "dataset_schema": DATASET_SCHEMA,
        "dataset_relative_root": DATASET_RELATIVE_ROOT,
        "repository": dict(repository),
        "source_lock_sha256": sha256_bytes(stable_json_bytes(source_lock)),
        "role_contract": role_contract(),
        "role_manifests": {
            role: {
                "path": ROLE_MANIFEST_RELATIVE_ROOT + "/{}.json".format(role),
                "file_sha256": sha256_file(Path(repository["root"]) / ROLE_MANIFEST_RELATIVE_ROOT / "{}.json".format(role)),
                "self_sha256": value["manifest_sha256"],
                "window_row_count": value["window_row_count"],
                "window_npz_sha256": value["window_npz_sha256"],
            }
            for role, value in role_manifests.items()
        },
        "audit": copy.deepcopy(dict(audit)),
        "inventory_sha256": inventory["inventory_sha256"],
        "inventory_file_count": inventory["file_count"],
        "inventory_total_size_bytes": inventory["total_size_bytes"],
        "exports": copy.deepcopy(dict(exports)),
        "governance": {
            "legacy_r255_r259_data_universe_abandoned": True,
            "legacy_cache_reconstruction_terminated": True,
            "new_role_seed_contract_frozen_before_generation": True,
            "objective_train_model_fit_count": 0,
            "selection_holdout_model_evaluation_count": 0,
            "frozen_probe_model_evaluation_count": 0,
            "final_evaluation_model_evaluation_count": 0,
            "selection_holdout_may_not_be_opened_before_policy_lock": True,
            "frozen_probe_may_not_be_opened_before_selection_lock": True,
            "final_evaluation_may_not_be_opened_before_final_preregistration": True,
            "role_regeneration_forbidden_after_terminal_report": True,
            "external_redundant_backup_required": True,
            "external_redundant_backup_attested": False,
        },
    }
    payload["seal_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_seal(payload)
    return payload


def validate_seal(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "phase314b_r260_stagea_data_universe_seal_v1":
        raise StageAError("seal schema changed")
    if payload.get("dataset_schema") != DATASET_SCHEMA:
        raise StageAError("seal dataset schema changed")
    if payload.get("role_contract") != role_contract():
        raise StageAError("seal role contract changed")
    governance = payload.get("governance")
    if not isinstance(governance, Mapping):
        raise StageAError("seal governance missing")
    true_fields = (
        "legacy_r255_r259_data_universe_abandoned",
        "legacy_cache_reconstruction_terminated",
        "new_role_seed_contract_frozen_before_generation",
        "selection_holdout_may_not_be_opened_before_policy_lock",
        "frozen_probe_may_not_be_opened_before_selection_lock",
        "final_evaluation_may_not_be_opened_before_final_preregistration",
        "role_regeneration_forbidden_after_terminal_report",
        "external_redundant_backup_required",
    )
    if any(governance.get(key) is not True for key in true_fields):
        raise StageAError("seal governance true field changed")
    if governance.get("external_redundant_backup_attested") is not False:
        raise StageAError("backup was incorrectly attested")
    for key in (
        "objective_train_model_fit_count",
        "selection_holdout_model_evaluation_count",
        "frozen_probe_model_evaluation_count",
        "final_evaluation_model_evaluation_count",
    ):
        if governance.get(key) != 0:
            raise StageAError("model access count changed")
    expected = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != "seal_sha256"})
    )
    if payload.get("seal_sha256") != expected:
        raise StageAError("seal self-hash changed")


def build_summary(
    *,
    repository: Mapping[str, Any],
    seal: Mapping[str, Any],
    legacy_write_ahead_before_sha256: str,
    legacy_write_ahead_after_sha256: str,
) -> Mapping[str, Any]:
    validate_seal(seal)
    if legacy_write_ahead_before_sha256 != legacy_write_ahead_after_sha256:
        raise StageAError("legacy Stage-J write-ahead changed")
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "external_redundant_persistence_not_yet_attested",
        "root_cause": (
            "phase314b_r260_stagea_new_data_universe_generated_audited_sealed_"
            "and_exported_but_external_redundant_backup_not_attested"
        ),
        "required_next_path": (
            "ATTEST_TWO_INDEPENDENT_EXTERNAL_COPIES_OF_R260_DATA_UNIVERSE_"
            "BEFORE_ANY_MODEL_FIT_OR_HOLDOUT_ACCESS"
        ),
        "repository": dict(repository),
        "data_universe_seal": copy.deepcopy(dict(seal)),
        "legacy_stagej_write_ahead_before_sha256": legacy_write_ahead_before_sha256,
        "legacy_stagej_write_ahead_after_sha256": legacy_write_ahead_after_sha256,
        "execution_counts": {
            "visible_seed_generation_count": TOTAL_VISIBLE_SEEDS,
            "paired_episode_generation_count": TOTAL_EPISODES,
            "role_window_build_count": len(ROLE_SPECS),
            "role_model_fit_count": 0,
            "selection_holdout_model_evaluation_count": 0,
            "frozen_probe_model_evaluation_count": 0,
            "final_evaluation_model_evaluation_count": 0,
            "deterministic_export_archive_count": len(ROLE_SPECS) + 1,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        "external_backup_attested": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        "external_backup_attested": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageAError("summary field changed: {}".format(key))
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageAError("Stage-A crossed a forbidden boundary")
    seal = payload.get("data_universe_seal")
    if not isinstance(seal, Mapping):
        raise StageAError("summary seal missing")
    validate_seal(seal)
    expected_sha = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != "summary_sha256"})
    )
    if payload.get("summary_sha256") != expected_sha:
        raise StageAError("summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    dataset_root: Path,
    write_ahead: Path,
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r260_stagea_data_universe_generation_or_seal_failed",
        "required_next_path": (
            "AUDIT_R260_STAGEA_PARTIAL_DATA_UNIVERSE_BEFORE_ANY_RESUME_OR_REGENERATION"
        ),
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "dataset_root_present": Path(dataset_root).exists(),
        "write_ahead_present": Path(write_ahead).exists(),
        "partial_artifacts_may_exist": Path(dataset_root).exists(),
        "legacy_data_universe_abandoned": True,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        "external_backup_attested": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
