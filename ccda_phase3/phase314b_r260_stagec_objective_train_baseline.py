"""Phase3.14b-r2.6.0 Stage C objective-train opening and baseline contract.

Stage A generated a new four-role data universe and Stage B attested two
independent external copies.  Stage C is the first stage authorized to parse
one role payload: ``objective_train``.  The other three role payloads remain
sealed.

This stage performs no learned-model fit.  It validates the sealed objective
NPZ against the committed Stage-A identities, freezes deterministic pair-group
folds and feature-normalization statistics, and reports two parameter-free
future-state baselines (persistence and constant velocity).  The resulting
contract is the sole numerical reference for the next objective-train-only
candidate-signal stage.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from . import phase314b_r260_stagea_data_universe_genesis as stagea
from . import phase314b_r260_stageb_external_backup_attestation as stageb
from .schema_v2 import (
    ACTION_DIM,
    DEFAULT_TF,
    DEFAULT_TH,
    FORMAL_CONDITIONS,
    N_BEADS,
    PAPER_X_DIM,
    ROBOT_PROXY_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
)

PHASE = "Phase3.14b-r2.6.0 Stage C"
SCHEMA = "phase314b_r260_stagec_objective_train_baseline_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
CONTRACT_SCHEMA = SCHEMA + "_contract_v1"
MANIFEST_SCHEMA = SCHEMA + "_contract_manifest_v1"

BASE_IMPLEMENTATION = "4b19caa1de2b89ddf700dcad4a4b72928c2a4818"
BASE_HEAD = "5296ecb948b91058ce2f0b502bf50baffac133d8"
STAGEA_IMPLEMENTATION = "b7ca7bc3396c0879f13662fd953881a9f42e40f2"
STAGEA_EVIDENCE = "636a2853db1a972073e2dd254e75a9724a5a1d17"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEA_REPORT = "reports/phase3_14b_r260_stagea_data_universe_genesis_summary.json"
STAGEA_REPORT_SHA256 = "60f9c2bc14d393dae061b2ac7ac3937e5ddf869c9841ba3ad834e300dfefd2c9"
STAGEB_REPORT = "reports/phase3_14b_r260_stageb_external_backup_attestation_summary.json"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.6.0 Stage C: open objective train and freeze baseline statistics"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.6.0 Stage C objective-train baseline evidence"
)
BLOCKED_SUBJECT = "Record Phase3.14b-r2.6.0 Stage C blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r260_stagec_objective_train_baseline.py"),
    ("A", "scripts/phase3_14b_r260_stagec_worker.py"),
    ("A", "scripts/phase3_14b_r260_stagec_execute.py"),
    ("A", "tests/test_phase3_14b_r260_stagec_objective_train_baseline.py"),
)

DATASET_RELATIVE_ROOT = stagea.DATASET_RELATIVE_ROOT
OBJECTIVE_ROLE = "objective_train"
OBJECTIVE_NPZ_RELATIVE = DATASET_RELATIVE_ROOT + "/windows/objective_train.npz"
OBJECTIVE_MANIFEST_RELATIVE = (
    DATASET_RELATIVE_ROOT + "/role_manifests/objective_train.json"
)
SEAL_RELATIVE = stagea.SEAL_RELATIVE
INVENTORY_RELATIVE = stagea.INVENTORY_RELATIVE

OUTPUT_RELATIVE_ROOT = "data/phase3_14b_r260_stagec_objective_train_baseline_v1"
CONTRACT_NPZ_RELATIVE = OUTPUT_RELATIVE_ROOT + "/objective_train_baseline_contract.npz"
CONTRACT_MANIFEST_RELATIVE = OUTPUT_RELATIVE_ROOT + "/contract_manifest.json"

SUCCESS_REPORT = "reports/phase3_14b_r260_stagec_objective_train_baseline_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r260_stagec_objective_train_baseline_blocked_summary.json"

WRITE_AHEAD_SUFFIX = ".phase314b_r260_stagec_write_ahead"
ATTEMPT_MARKER_NAME = "attempt_started.json"
WORKER_EVIDENCE_NAME = "stagec_worker_evidence.json"

EXPECTED_OBJECTIVE_ROWS = 3006
EXPECTED_OBJECTIVE_GROUPS = 320
EXPECTED_OBJECTIVE_SEEDS = 320
EXPECTED_FOLDS = 6
EXPECTED_CONDITIONS = tuple(FORMAL_CONDITIONS)
CABLE_DIM = N_BEADS * 2

Z_95 = 1.959963984540054
Z_80 = 0.8416212335729143
EPSILON = 1.0e-12

FORBIDDEN_ROLE_NPZ_RELATIVES = (
    DATASET_RELATIVE_ROOT + "/windows/selection_holdout.npz",
    DATASET_RELATIVE_ROOT + "/windows/frozen_probe.npz",
    DATASET_RELATIVE_ROOT + "/windows/final_evaluation.npz",
)

FALSE_BOUNDARIES = (
    "selection_holdout_opened",
    "selection_holdout_model_evaluation_run",
    "frozen_probe_opened",
    "frozen_probe_model_evaluation_run",
    "final_evaluation_opened",
    "final_evaluation_model_evaluation_run",
    "risk_fit_run",
    "direction_fit_run",
    "diffusion_model_fit_run",
    "candidate_generation_run",
    "candidate_evaluation_run",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "phase4",
    "cps",
    "weights_saved",
    "checkpoint_saved",
)


class StageCError(RuntimeError):
    """Fail-closed Stage-C error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


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
    header = compact_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return sha256_bytes(header + b"\0" + array.tobytes())


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageCError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageCError("write-once output exists: {}".format(target))
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


def write_ahead_root(root: Path) -> Path:
    repo = Path(root).resolve()
    return repo.with_name(repo.name + WRITE_AHEAD_SUFFIX)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageCError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageCError("{} worktree is dirty".format(label))


def validate_stageb_report(payload: Mapping[str, Any]) -> None:
    stageb.validate_summary(payload)
    expected = {
        "schema": stageb.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "required_next_path": (
            "PREREGISTER_R260_STAGEC_OBJECTIVE_TRAIN_OPENING_AND_BASELINE_STATISTICS"
        ),
        "external_backup_attested": True,
        "independent_external_copy_count": 2,
        "objective_train_access_authorized_by_this_report": True,
        "selection_holdout_access_authorized_by_this_report": False,
        "frozen_probe_access_authorized_by_this_report": False,
        "final_evaluation_access_authorized_by_this_report": False,
        "objective_train_opened": False,
        "role_npz_parsed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageCError("Stage-B authorization field changed: {}".format(key))


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-B evidence parent": (
            _git(repo, "rev-parse", BASE_HEAD + "^"), BASE_IMPLEMENTATION
        ),
        "Stage-A evidence": (
            _git(repo, "rev-parse", BASE_IMPLEMENTATION + "^"), STAGEA_EVIDENCE
        ),
        "Stage-A implementation": (
            _git(repo, "rev-parse", STAGEA_EVIDENCE + "^"), STAGEA_IMPLEMENTATION
        ),
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
            raise StageCError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageCError("Stage-C implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION) != stageb.IMPLEMENTATION_SUBJECT:
        raise StageCError("Stage-B implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_HEAD) != stageb.EVIDENCE_SUBJECT:
        raise StageCError("Stage-B evidence subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageCError("Stage-C implementation population changed")
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageCError("Stage-C source missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageCError("Stage-C source differs from implementation commit")
    stageb_path = repo / STAGEB_REPORT
    if not stageb_path.is_file():
        raise StageCError("Stage-B report missing")
    if _git_bytes(repo, "show", "{}:{}".format(BASE_HEAD, STAGEB_REPORT)) != stageb_path.read_bytes():
        raise StageCError("Stage-B report differs from committed evidence")
    validate_stageb_report(load_json(stageb_path))
    stagea_path = repo / STAGEA_REPORT
    if not stagea_path.is_file() or sha256_file(stagea_path) != STAGEA_REPORT_SHA256:
        raise StageCError("Stage-A report identity changed")
    if _git_bytes(repo, "show", "{}:{}".format(STAGEA_EVIDENCE, STAGEA_REPORT)) != stagea_path.read_bytes():
        raise StageCError("Stage-A report differs from evidence commit")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(repo / "external/deformable-ravens", "submodule")
    if (repo / OUTPUT_RELATIVE_ROOT).exists():
        raise StageCError("Stage-C output root already exists")
    if write_ahead_root(repo).exists():
        raise StageCError("Stage-C write-ahead already exists")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageCError("Stage-C terminal report already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_b_implementation": BASE_IMPLEMENTATION,
        "stage_b_evidence": BASE_HEAD,
        "stage_a_implementation": STAGEA_IMPLEMENTATION,
        "stage_a_evidence": STAGEA_EVIDENCE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def validate_source_metadata(root: Path, *, open_objective_npz: bool) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    stageb_report = load_json(repo / STAGEB_REPORT)
    validate_stageb_report(stageb_report)
    stagea_report = load_json(repo / STAGEA_REPORT)
    stagea.validate_summary(stagea_report)
    if sha256_file(repo / STAGEA_REPORT) != STAGEA_REPORT_SHA256:
        raise StageCError("Stage-A report SHA changed")
    embedded_seal = stagea_report.get("data_universe_seal")
    if not isinstance(embedded_seal, Mapping):
        raise StageCError("Stage-A embedded seal missing")
    stagea.validate_seal(embedded_seal)
    seal_path = repo / SEAL_RELATIVE
    seal = load_json(seal_path)
    stagea.validate_seal(seal)
    if seal != embedded_seal:
        raise StageCError("local seal differs from committed Stage-A evidence")
    objective_entry = seal.get("role_manifests", {}).get(OBJECTIVE_ROLE)
    if not isinstance(objective_entry, Mapping):
        raise StageCError("objective-train seal entry missing")
    manifest_path = repo / OBJECTIVE_MANIFEST_RELATIVE
    manifest = load_json(manifest_path)
    if sha256_file(manifest_path) != objective_entry.get("file_sha256"):
        raise StageCError("objective-train role-manifest file SHA changed")
    if manifest.get("manifest_sha256") != objective_entry.get("self_sha256"):
        raise StageCError("objective-train role-manifest self identity changed")
    base = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != sha256_bytes(stagea.stable_json_bytes(base)):
        raise StageCError("objective-train role-manifest self-hash changed")
    expected_manifest = {
        "schema": "phase314b_r260_stagea_role_manifest_v1",
        "dataset_schema": stagea.DATASET_SCHEMA,
        "role": OBJECTIVE_ROLE,
        "governance_role": "fit_and_nested_group_oof_only",
        "seed_count": EXPECTED_OBJECTIVE_SEEDS,
        "pair_group_count": EXPECTED_OBJECTIVE_GROUPS,
        "window_row_count": EXPECTED_OBJECTIVE_ROWS,
        "episode_count": 2 * EXPECTED_OBJECTIVE_SEEDS,
        "model_fit_count": 0,
        "model_evaluation_count": 0,
        "window_npz": OBJECTIVE_NPZ_RELATIVE,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise StageCError("objective-train role-manifest field changed: {}".format(key))
    if objective_entry.get("window_row_count") != EXPECTED_OBJECTIVE_ROWS:
        raise StageCError("objective-train sealed row count changed")
    if objective_entry.get("window_npz_sha256") != manifest.get("window_npz_sha256"):
        raise StageCError("objective-train NPZ identity differs between seal and manifest")
    npz_path = repo / OBJECTIVE_NPZ_RELATIVE
    if not npz_path.is_file():
        raise StageCError("objective-train NPZ missing")
    inventory_path = repo / INVENTORY_RELATIVE
    inventory = load_json(inventory_path)
    if inventory.get("schema") != "phase314b_r260_stagea_artifact_inventory_v1":
        raise StageCError("Stage-A inventory schema changed")
    inventory_base = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if inventory.get("inventory_sha256") != sha256_bytes(stagea.stable_json_bytes(inventory_base)):
        raise StageCError("Stage-A inventory self-hash changed")
    inventory_records = {
        str(record["path"]): record for record in inventory.get("records", [])
        if isinstance(record, Mapping)
    }
    objective_inventory_key = "windows/objective_train.npz"
    record = inventory_records.get(objective_inventory_key)
    if not isinstance(record, Mapping):
        raise StageCError("objective-train NPZ missing from inventory")
    if record.get("sha256") != manifest.get("window_npz_sha256"):
        raise StageCError("objective-train inventory SHA changed")
    if int(record.get("size_bytes", -1)) != int(npz_path.stat().st_size):
        raise StageCError("objective-train inventory size changed")
    if open_objective_npz and sha256_file(npz_path) != manifest.get("window_npz_sha256"):
        raise StageCError("objective-train NPZ file SHA changed")
    for relative in FORBIDDEN_ROLE_NPZ_RELATIVES:
        if relative == OBJECTIVE_NPZ_RELATIVE:
            raise StageCError("forbidden-role contract is invalid")
    result: Dict[str, Any] = {
        "stage_b_summary_sha256": str(stageb_report["summary_sha256"]),
        "stage_a_report_file_sha256": STAGEA_REPORT_SHA256,
        "stage_a_summary_sha256": str(stagea_report["summary_sha256"]),
        "stage_a_seal_sha256": str(seal["seal_sha256"]),
        "objective_manifest_file_sha256": sha256_file(manifest_path),
        "objective_manifest_self_sha256": str(manifest["manifest_sha256"]),
        "objective_npz_sha256": str(manifest["window_npz_sha256"]),
        "objective_npz_size_bytes": int(npz_path.stat().st_size),
        "objective_row_count": EXPECTED_OBJECTIVE_ROWS,
        "objective_group_count": EXPECTED_OBJECTIVE_GROUPS,
        "objective_seed_count": EXPECTED_OBJECTIVE_SEEDS,
        "forbidden_role_npz_open_count": 0,
    }
    if open_objective_npz:
        arrays = stagea.load_npz_strict(npz_path)
        validate_objective_arrays(arrays, manifest)
        result["arrays"] = arrays
    return result


def validate_objective_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> None:
    stagea.validate_role_arrays(OBJECTIVE_ROLE, arrays)
    if int(arrays["paper_x"].shape[0]) != EXPECTED_OBJECTIVE_ROWS:
        raise StageCError("objective-train row count changed")
    if len(set(arrays["pair_group"].astype(str).tolist())) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageCError("objective-train pair-group count changed")
    if len(set(arrays["visible_seed"].astype(np.int64).tolist())) != EXPECTED_OBJECTIVE_SEEDS:
        raise StageCError("objective-train visible-seed count changed")
    if set(arrays["condition_name"].astype(str).tolist()) != set(EXPECTED_CONDITIONS):
        raise StageCError("objective-train condition population changed")
    if set(arrays["split_name"].astype(str).tolist()) != {OBJECTIVE_ROLE}:
        raise StageCError("objective-train split label changed")
    shapes = manifest.get("array_shapes")
    dtypes = manifest.get("array_dtypes")
    identities = manifest.get("array_sha256")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping) or not isinstance(identities, Mapping):
        raise StageCError("objective-train manifest array contract missing")
    if set(arrays) != set(shapes) or set(arrays) != set(dtypes) or set(arrays) != set(identities):
        raise StageCError("objective-train array population differs from manifest")
    for key, value in arrays.items():
        if list(value.shape) != list(shapes[key]):
            raise StageCError("objective array shape changed: {}".format(key))
        if str(value.dtype) != str(dtypes[key]):
            raise StageCError("objective array dtype changed: {}".format(key))
        if sha256_array(value) != str(identities[key]):
            raise StageCError("objective array SHA changed: {}".format(key))


def deterministic_group_folds(
    pair_groups: np.ndarray, fold_count: int = EXPECTED_FOLDS
) -> Mapping[str, Any]:
    groups = sorted(set(np.asarray(pair_groups).astype(str).tolist()))
    if len(groups) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageCError("unexpected objective pair-group population")
    ordered = sorted(
        groups,
        key=lambda value: (sha256_bytes(value.encode("utf-8")), value),
    )
    group_to_fold = {group: index % int(fold_count) for index, group in enumerate(ordered)}
    group_names = np.asarray(ordered, dtype="<U{}".format(max(len(value) for value in ordered)))
    group_fold_id = np.asarray([group_to_fold[value] for value in ordered], dtype=np.int8)
    row_fold_id = np.asarray(
        [group_to_fold[str(value)] for value in np.asarray(pair_groups).astype(str)],
        dtype=np.int8,
    )
    counts = [int(np.sum(group_fold_id == fold)) for fold in range(int(fold_count))]
    if max(counts) - min(counts) > 1:
        raise StageCError("group-fold population is imbalanced")
    return {
        "group_names": group_names,
        "group_fold_id": group_fold_id,
        "row_fold_id": row_fold_id,
        "group_to_fold": group_to_fold,
        "fold_group_counts": counts,
        "assignment_sha256": sha256_bytes(
            compact_json_bytes([[group, group_to_fold[group]] for group in ordered])
        ),
    }


def _quantiles(values: np.ndarray) -> Mapping[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StageCError("cannot summarize an empty or non-finite array")
    return {
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _safe_relative(value: float, denominator: float) -> float:
    return float(value / max(abs(float(denominator)), EPSILON))


def feature_normalization(arrays: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name in ("paper_x", "state_action_x", "y_state", "y_final_state", "y_action"):
        value = np.asarray(arrays[name], dtype=np.float64)
        mean = np.mean(value, axis=0, dtype=np.float64)
        std = np.std(value, axis=0, dtype=np.float64)
        scale = np.where(std > EPSILON, std, 1.0)
        result[name + "_mean"] = mean.astype(np.float64)
        result[name + "_std"] = std.astype(np.float64)
        result[name + "_scale"] = scale.astype(np.float64)
        result[name + "_constant_mask"] = (std <= EPSILON).astype(np.bool_)
    return result


def _group_means(values: np.ndarray, pair_groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    groups = sorted(set(np.asarray(pair_groups).astype(str).tolist()))
    rendered = np.asarray(pair_groups).astype(str)
    means = np.asarray(
        [float(np.mean(np.asarray(values, dtype=np.float64)[rendered == group])) for group in groups],
        dtype=np.float64,
    )
    names = np.asarray(groups, dtype="<U{}".format(max(len(value) for value in groups)))
    return names, means


def detectability_reference(group_losses: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(group_losses, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise StageCError("invalid group losses for detectability reference")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(values.size)
    ci_half = Z_95 * se
    mde = (Z_95 + Z_80) * se
    return {
        "group_count": int(values.size),
        "mean_loss": mean,
        "group_sd": sd,
        "standard_error": se,
        "normal_95_ci_half_width": ci_half,
        "relative_95_ci_half_width": _safe_relative(ci_half, mean),
        "normal_80_power_two_sided_mde_absolute": mde,
        "normal_80_power_two_sided_mde_relative": _safe_relative(mde, mean),
        "interpretation": (
            "baseline-scale reference only; Stage D must estimate paired "
            "candidate-minus-control difference variance before locking an effect gate"
        ),
    }


def analytic_baselines(arrays: Mapping[str, np.ndarray], row_fold_id: np.ndarray) -> Mapping[str, Any]:
    paper = np.asarray(arrays["paper_x"], dtype=np.float64).reshape(-1, DEFAULT_TH, STATE_DIM)
    future = np.asarray(arrays["y_state"], dtype=np.float64)
    current = paper[:, -1, :]
    previous = paper[:, -2, :]
    horizons = np.arange(1, DEFAULT_TF + 1, dtype=np.float64).reshape(1, DEFAULT_TF, 1)
    persistence = np.repeat(current[:, None, :], DEFAULT_TF, axis=1)
    constant_velocity = current[:, None, :] + horizons * (current - previous)[:, None, :]

    predictions = {
        "persistence": persistence,
        "constant_velocity": constant_velocity,
    }
    result: Dict[str, Any] = {}
    groups = arrays["pair_group"].astype(str)
    conditions = arrays["condition_name"].astype(str)
    for name, prediction in predictions.items():
        error = prediction - future
        full_row = np.mean(error * error, axis=(1, 2), dtype=np.float64)
        cable_row = np.mean(
            error[:, :, :CABLE_DIM] * error[:, :, :CABLE_DIM],
            axis=(1, 2),
            dtype=np.float64,
        )
        final_cable_row = np.mean(
            error[:, -1, :CABLE_DIM] * error[:, -1, :CABLE_DIM],
            axis=1,
            dtype=np.float64,
        )
        horizon_cable = np.mean(
            error[:, :, :CABLE_DIM] * error[:, :, :CABLE_DIM],
            axis=2,
            dtype=np.float64,
        )
        _, group_cable = _group_means(cable_row, groups)
        fold_means = [
            float(np.mean(cable_row[row_fold_id == fold]))
            for fold in range(EXPECTED_FOLDS)
        ]
        result[name] = {
            "full_state_mse": float(np.mean(full_row)),
            "cable_mse": float(np.mean(cable_row)),
            "final_cable_mse": float(np.mean(final_cable_row)),
            "horizon_cable_mse": [float(value) for value in np.mean(horizon_cable, axis=0)],
            "row_cable_mse_distribution": _quantiles(cable_row),
            "group_cable_mse_distribution": _quantiles(group_cable),
            "condition_cable_mse": {
                condition: float(np.mean(cable_row[conditions == condition]))
                for condition in EXPECTED_CONDITIONS
            },
            "fold_cable_mse": fold_means,
            "fold_relative_spread": _safe_relative(max(fold_means) - min(fold_means), np.mean(fold_means)),
            "detectability_reference": detectability_reference(group_cable),
            "row_loss_sha256": sha256_array(cable_row.astype(np.float64)),
            "group_loss_sha256": sha256_array(group_cable.astype(np.float64)),
        }
    return result


def semantic_statistics(arrays: Mapping[str, np.ndarray], row_fold_id: np.ndarray) -> Mapping[str, Any]:
    conditions = arrays["condition_name"].astype(str)
    groups = arrays["pair_group"].astype(str)
    seeds = arrays["visible_seed"].astype(np.int64)
    times = arrays["window_t"].astype(np.int64)
    success = arrays["success"].astype(bool)
    final_fraction = arrays["final_fraction"].astype(np.float64)
    engagement = arrays["engagement_step"].astype(np.int64)
    release = arrays["release_step"].astype(np.int64)
    pre = arrays["pre_engagement"].astype(bool)
    paper = np.asarray(arrays["paper_x"], dtype=np.float64).reshape(-1, DEFAULT_TH, STATE_DIM)
    future = np.asarray(arrays["y_state"], dtype=np.float64)
    current = paper[:, -1, :]
    cable_displacement = np.linalg.norm(
        future[:, :, :CABLE_DIM].reshape(-1, DEFAULT_TF, N_BEADS, 2)
        - current[:, None, :CABLE_DIM].reshape(-1, 1, N_BEADS, 2),
        axis=3,
    )
    robot_displacement = np.linalg.norm(
        future[:, :, CABLE_DIM:] - current[:, None, CABLE_DIM:], axis=2
    )
    result: Dict[str, Any] = {
        "row_count": int(len(conditions)),
        "pair_group_count": int(len(set(groups.tolist()))),
        "visible_seed_count": int(len(set(seeds.tolist()))),
        "fold_count": EXPECTED_FOLDS,
        "fold_row_counts": [int(np.sum(row_fold_id == fold)) for fold in range(EXPECTED_FOLDS)],
        "window_t": _quantiles(times.astype(np.float64)),
        "final_fraction": _quantiles(final_fraction),
        "success_rate": float(np.mean(success)),
        "pre_engagement_rate": float(np.mean(pre)),
        "cable_displacement_per_bead": _quantiles(cable_displacement),
        "robot_proxy_displacement": _quantiles(robot_displacement),
        "target_action_absolute": _quantiles(np.abs(np.asarray(arrays["y_action"], dtype=np.float64))),
        "conditions": {},
    }
    for condition in EXPECTED_CONDITIONS:
        mask = conditions == condition
        result["conditions"][condition] = {
            "row_count": int(np.sum(mask)),
            "pair_group_count": int(len(set(groups[mask].tolist()))),
            "visible_seed_count": int(len(set(seeds[mask].tolist()))),
            "success_rate": float(np.mean(success[mask])),
            "final_fraction": _quantiles(final_fraction[mask]),
            "pre_engagement_rate": float(np.mean(pre[mask])),
            "engagement_step_nonnegative_rate": float(np.mean(engagement[mask] >= 0)),
            "release_step_nonnegative_rate": float(np.mean(release[mask] >= 0)),
            "window_t": _quantiles(times[mask].astype(np.float64)),
        }
    return result


def paired_branch_statistics(arrays: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    conditions = arrays["condition_name"].astype(str)
    seeds = arrays["visible_seed"].astype(np.int64)
    times = arrays["window_t"].astype(np.int64)
    index = {
        (str(conditions[row]), int(seeds[row]), int(times[row])): row
        for row in range(len(seeds))
    }
    prefix_mae: List[float] = []
    final_ordered_mse: List[float] = []
    final_fraction_gap: List[float] = []
    success_mismatch: List[float] = []
    matched = 0
    for seed in sorted(set(seeds.tolist())):
        seed_times = sorted(set(times[seeds == seed].tolist()))
        for current in seed_times:
            free = index.get(("free", seed, current))
            hidden = index.get(("hidden_slack_breakaway_pin_v2", seed, current))
            if free is None or hidden is None:
                continue
            matched += 1
            prefix_mae.append(
                float(np.mean(np.abs(arrays["paper_x"][free] - arrays["paper_x"][hidden])))
            )
            delta = (
                np.asarray(arrays["y_final_state"][free][:CABLE_DIM], dtype=np.float64)
                - np.asarray(arrays["y_final_state"][hidden][:CABLE_DIM], dtype=np.float64)
            )
            final_ordered_mse.append(float(np.mean(delta * delta)))
            final_fraction_gap.append(
                abs(float(arrays["final_fraction"][free] - arrays["final_fraction"][hidden]))
            )
            success_mismatch.append(float(arrays["success"][free] != arrays["success"][hidden]))
    if matched <= 0:
        raise StageCError("objective train has no matched paired windows")
    return {
        "matched_pair_window_count": matched,
        "paper_x_pair_mae": _quantiles(np.asarray(prefix_mae)),
        "final_ordered_cable_mse": _quantiles(np.asarray(final_ordered_mse)),
        "final_fraction_absolute_gap": _quantiles(np.asarray(final_fraction_gap)),
        "success_mismatch_rate": float(np.mean(success_mismatch)),
    }


def float_precision_audit(arrays: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    paper32 = np.asarray(arrays["paper_x"], dtype=np.float32).reshape(-1, DEFAULT_TH, STATE_DIM)
    future32 = np.asarray(arrays["y_state"], dtype=np.float32)
    current32 = paper32[:, -1, :]
    persistence32 = np.repeat(current32[:, None, :], DEFAULT_TF, axis=1)
    mse32 = float(np.mean((persistence32[:, :, :CABLE_DIM] - future32[:, :, :CABLE_DIM]) ** 2, dtype=np.float32))
    paper64 = paper32.astype(np.float64)
    future64 = future32.astype(np.float64)
    current64 = paper64[:, -1, :]
    persistence64 = np.repeat(current64[:, None, :], DEFAULT_TF, axis=1)
    mse64 = float(np.mean((persistence64[:, :, :CABLE_DIM] - future64[:, :, :CABLE_DIM]) ** 2, dtype=np.float64))
    absolute = abs(mse64 - mse32)
    relative = _safe_relative(absolute, mse64)
    return {
        "persistence_cable_mse_float32": mse32,
        "persistence_cable_mse_float64": mse64,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "recommended_future_candidate_numerical_margin": max(1.0e-8, 10.0 * relative),
        "interpretation": "numerical resolution only; not a scientific minimum effect size",
    }


def _deterministic_npz_payload(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            array = np.asarray(arrays[key])
            if array.dtype.kind == "O":
                raise StageCError("object dtype forbidden in Stage-C contract")
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo("{}.npy".format(key), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                member.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def build_contract_arrays(
    arrays: Mapping[str, np.ndarray], folds: Mapping[str, Any]
) -> Mapping[str, np.ndarray]:
    normalization = feature_normalization(arrays)
    output: Dict[str, np.ndarray] = {
        "group_names": np.asarray(folds["group_names"]),
        "group_fold_id": np.asarray(folds["group_fold_id"], dtype=np.int8),
        "row_fold_id": np.asarray(folds["row_fold_id"], dtype=np.int8),
    }
    output.update(normalization)
    return output


def write_contract_once(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    payload = _deterministic_npz_payload(arrays)
    atomic_write_once(path, payload)
    return sha256_bytes(payload)


def validate_contract_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {"group_names", "group_fold_id", "row_fold_id"}
    for name in ("paper_x", "state_action_x", "y_state", "y_final_state", "y_action"):
        required.update(
            {
                name + "_mean",
                name + "_std",
                name + "_scale",
                name + "_constant_mask",
            }
        )
    if set(arrays) != required:
        raise StageCError("Stage-C contract array population changed")
    if arrays["group_names"].shape != (EXPECTED_OBJECTIVE_GROUPS,):
        raise StageCError("Stage-C group-name shape changed")
    if arrays["group_fold_id"].shape != (EXPECTED_OBJECTIVE_GROUPS,):
        raise StageCError("Stage-C group-fold shape changed")
    if arrays["row_fold_id"].shape != (EXPECTED_OBJECTIVE_ROWS,):
        raise StageCError("Stage-C row-fold shape changed")
    if arrays["group_names"].dtype.kind != "U":
        raise StageCError("Stage-C group names must be fixed-width unicode")
    if arrays["group_fold_id"].dtype != np.int8 or arrays["row_fold_id"].dtype != np.int8:
        raise StageCError("Stage-C fold arrays must be int8")
    if set(arrays["group_fold_id"].astype(int).tolist()) != set(range(EXPECTED_FOLDS)):
        raise StageCError("Stage-C fold population changed")
    if set(arrays["row_fold_id"].astype(int).tolist()) != set(range(EXPECTED_FOLDS)):
        raise StageCError("Stage-C row fold population changed")
    expected_feature_shapes = {
        "paper_x": (PAPER_X_DIM,),
        "state_action_x": (STATE_ACTION_X_DIM,),
        "y_state": (DEFAULT_TF, STATE_DIM),
        "y_final_state": (STATE_DIM,),
        "y_action": (ACTION_DIM,),
    }
    for name, shape in expected_feature_shapes.items():
        for suffix in ("_mean", "_std", "_scale"):
            value = arrays[name + suffix]
            if value.shape != shape or value.dtype != np.float64:
                raise StageCError("Stage-C normalization shape/dtype changed: {}{}".format(name, suffix))
        mask = arrays[name + "_constant_mask"]
        if mask.shape != shape or mask.dtype != np.bool_:
            raise StageCError("Stage-C constant-mask shape/dtype changed: {}".format(name))
        if np.any(arrays[name + "_scale"] <= 0.0):
            raise StageCError("Stage-C normalization scale is non-positive: {}".format(name))
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise StageCError("Stage-C contract contains object dtype")
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise StageCError("Stage-C contract contains non-finite values: {}".format(key))


def build_contract_manifest(
    *,
    source: Mapping[str, Any],
    contract_arrays: Mapping[str, np.ndarray],
    contract_file_sha256: str,
    folds: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_contract_arrays(contract_arrays)
    payload: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "source_objective_npz_sha256": source["objective_npz_sha256"],
        "source_objective_manifest_self_sha256": source["objective_manifest_self_sha256"],
        "contract_npz": CONTRACT_NPZ_RELATIVE,
        "contract_npz_sha256": contract_file_sha256,
        "array_shapes": {key: list(value.shape) for key, value in contract_arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in contract_arrays.items()},
        "array_sha256": {key: sha256_array(value) for key, value in contract_arrays.items()},
        "fold_assignment_sha256": folds["assignment_sha256"],
        "fold_count": EXPECTED_FOLDS,
        "fold_group_counts": list(folds["fold_group_counts"]),
        "objective_train_open_count": 1,
        "model_fit_count": 0,
        "model_evaluation_count": 0,
    }
    payload["manifest_sha256"] = sha256_bytes(stable_json_bytes(payload))
    return payload


def validate_contract_manifest(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema": MANIFEST_SCHEMA,
        "contract_npz": CONTRACT_NPZ_RELATIVE,
        "fold_count": EXPECTED_FOLDS,
        "objective_train_open_count": 1,
        "model_fit_count": 0,
        "model_evaluation_count": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageCError("Stage-C contract manifest field changed: {}".format(key))
    counts = payload.get("fold_group_counts")
    if not isinstance(counts, Sequence) or len(counts) != EXPECTED_FOLDS:
        raise StageCError("Stage-C fold counts missing")
    if sum(int(value) for value in counts) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageCError("Stage-C fold group total changed")
    base = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("manifest_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageCError("Stage-C contract manifest self-hash changed")


def run_objective_train_analysis(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    source = dict(validate_source_metadata(repo, open_objective_npz=True))
    arrays = source.pop("arrays")
    folds = deterministic_group_folds(arrays["pair_group"])
    contract_arrays = build_contract_arrays(arrays, folds)
    validate_contract_arrays(contract_arrays)
    output_root = repo / OUTPUT_RELATIVE_ROOT
    output_root.mkdir(parents=True, exist_ok=False)
    contract_path = repo / CONTRACT_NPZ_RELATIVE
    contract_sha = write_contract_once(contract_path, contract_arrays)
    manifest = build_contract_manifest(
        source=source,
        contract_arrays=contract_arrays,
        contract_file_sha256=contract_sha,
        folds=folds,
    )
    atomic_write_once(repo / CONTRACT_MANIFEST_RELATIVE, stable_json_bytes(manifest))
    semantic = semantic_statistics(arrays, folds["row_fold_id"])
    paired = paired_branch_statistics(arrays)
    baselines = analytic_baselines(arrays, folds["row_fold_id"])
    precision = float_precision_audit(arrays)
    checks = {
        "objective_row_count": semantic["row_count"] == EXPECTED_OBJECTIVE_ROWS,
        "objective_group_count": semantic["pair_group_count"] == EXPECTED_OBJECTIVE_GROUPS,
        "objective_seed_count": semantic["visible_seed_count"] == EXPECTED_OBJECTIVE_SEEDS,
        "fold_count": semantic["fold_count"] == EXPECTED_FOLDS,
        "fold_group_balance": max(folds["fold_group_counts"]) - min(folds["fold_group_counts"]) <= 1,
        "paired_windows_present": paired["matched_pair_window_count"] > 0,
        "persistence_positive_loss": baselines["persistence"]["cable_mse"] > 0.0,
        "constant_velocity_finite": math.isfinite(baselines["constant_velocity"]["cable_mse"]),
        "selection_holdout_closed": True,
        "frozen_probe_closed": True,
        "final_evaluation_closed": True,
    }
    if not all(checks.values()):
        raise StageCError("Stage-C objective-train baseline checks failed")
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "source": source,
        "fold_contract": {
            "fold_count": EXPECTED_FOLDS,
            "fold_group_counts": list(folds["fold_group_counts"]),
            "fold_assignment_sha256": folds["assignment_sha256"],
            "group_fold_population_sha256": sha256_array(folds["group_fold_id"]),
            "row_fold_population_sha256": sha256_array(folds["row_fold_id"]),
            "group_assignment": [
                {"pair_group": group, "fold": int(folds["group_to_fold"][group])}
                for group in folds["group_names"].astype(str).tolist()
            ],
            "paired_conditions_never_cross_folds": True,
        },
        "semantic_statistics": semantic,
        "paired_branch_statistics": paired,
        "analytic_baselines": baselines,
        "float_precision_audit": precision,
        "contract_artifact": {
            "contract_npz": CONTRACT_NPZ_RELATIVE,
            "contract_npz_sha256": contract_sha,
            "contract_manifest": CONTRACT_MANIFEST_RELATIVE,
            "contract_manifest_file_sha256": sha256_file(repo / CONTRACT_MANIFEST_RELATIVE),
            "contract_manifest_self_sha256": manifest["manifest_sha256"],
        },
        "checks": checks,
        "execution_counts": {
            "objective_train_logical_open_count": 1,
            "objective_train_file_hash_count": 1,
            "objective_train_npz_parse_count": 1,
            "objective_train_role_manifest_read_count": 1,
            "objective_train_analytic_baseline_count": 2,
            "objective_train_model_fit_count": 0,
            "objective_train_model_evaluation_count": 0,
            "selection_holdout_npz_open_count": 0,
            "frozen_probe_npz_open_count": 0,
            "final_evaluation_npz_open_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["worker_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_worker_result(result, repo)
    return result


def validate_worker_result(payload: Mapping[str, Any], root: Path) -> None:
    expected = {
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageCError("Stage-C worker field changed: {}".format(key))
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCError("Stage-C worker crossed a forbidden boundary")
    counts = payload.get("execution_counts")
    expected_counts = {
        "objective_train_logical_open_count": 1,
        "objective_train_file_hash_count": 1,
        "objective_train_npz_parse_count": 1,
        "objective_train_role_manifest_read_count": 1,
        "objective_train_analytic_baseline_count": 2,
        "objective_train_model_fit_count": 0,
        "objective_train_model_evaluation_count": 0,
        "selection_holdout_npz_open_count": 0,
        "frozen_probe_npz_open_count": 0,
        "final_evaluation_npz_open_count": 0,
    }
    if counts != expected_counts:
        raise StageCError("Stage-C worker execution counts changed")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise StageCError("Stage-C worker checks failed")
    fold = payload.get("fold_contract")
    if not isinstance(fold, Mapping) or fold.get("fold_count") != EXPECTED_FOLDS:
        raise StageCError("Stage-C worker fold contract missing")
    assignments = fold.get("group_assignment")
    if not isinstance(assignments, Sequence) or len(assignments) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageCError("Stage-C worker group assignment changed")
    artifact = payload.get("contract_artifact")
    if not isinstance(artifact, Mapping):
        raise StageCError("Stage-C worker contract artifact missing")
    contract_path = Path(root) / str(artifact["contract_npz"])
    manifest_path = Path(root) / str(artifact["contract_manifest"])
    if not contract_path.is_file() or sha256_file(contract_path) != artifact.get("contract_npz_sha256"):
        raise StageCError("Stage-C contract NPZ identity changed")
    if not manifest_path.is_file() or sha256_file(manifest_path) != artifact.get("contract_manifest_file_sha256"):
        raise StageCError("Stage-C contract manifest identity changed")
    validate_contract_manifest(load_json(manifest_path))
    base = {key: value for key, value in payload.items() if key != "worker_result_sha256"}
    if payload.get("worker_result_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageCError("Stage-C worker self-hash changed")


def build_summary(
    *, repository: Mapping[str, Any], worker: Mapping[str, Any], worker_file_sha256: str
) -> Mapping[str, Any]:
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "primary_failure_locus": "objective_train_baseline_contract_frozen",
        "root_cause": (
            "phase314b_r260_stagec_objective_train_opened_once_and_new_data_"
            "universe_baseline_statistics_fold_contract_and_numeric_scale_frozen"
        ),
        "required_next_path": (
            "PREREGISTER_R260_STAGED_OBJECTIVE_TRAIN_ONLY_CANDIDATE_SIGNAL_"
            "BENCHMARK_AGAINST_FROZEN_STAGEC_BASELINE_CONTRACT"
        ),
        "repository": copy.deepcopy(dict(repository)),
        "worker_evidence": copy.deepcopy(dict(worker)),
        "worker_evidence_file_sha256": worker_file_sha256,
        "objective_train_opened": True,
        "objective_train_access_count_added": 1,
        "cumulative_objective_train_open_count": 1,
        "objective_train_model_fit_count": 0,
        "objective_train_model_evaluation_count": 0,
        "selection_holdout_access_count_added": 0,
        "frozen_probe_access_count_added": 0,
        "final_evaluation_access_count_added": 0,
        "stage_d_effect_gate_policy": {
            "must_use_stagec_pair_group_folds": True,
            "must_report_paired_candidate_minus_control_group_differences": True,
            "must_report_95_percent_group_confidence_interval": True,
            "must_report_all_six_fold_effects": True,
            "must_compare_effect_to_stagec_float_precision_margin": True,
            "must_not_use_baseline_loss_variance_as_candidate_difference_variance": True,
            "minimum_effect_threshold_locked_here": False,
            "reason": (
                "a statistically valid minimum candidate effect requires the variance "
                "of paired candidate-minus-control differences, unavailable before Stage D"
            ),
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "objective_train_opened": True,
        "objective_train_access_count_added": 1,
        "cumulative_objective_train_open_count": 1,
        "objective_train_model_fit_count": 0,
        "objective_train_model_evaluation_count": 0,
        "selection_holdout_access_count_added": 0,
        "frozen_probe_access_count_added": 0,
        "final_evaluation_access_count_added": 0,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageCError("Stage-C summary field changed: {}".format(key))
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCError("Stage-C summary crossed a forbidden boundary")
    worker = payload.get("worker_evidence")
    if not isinstance(worker, Mapping):
        raise StageCError("Stage-C summary worker evidence missing")
    policy = payload.get("stage_d_effect_gate_policy")
    if not isinstance(policy, Mapping):
        raise StageCError("Stage-C effect-gate policy missing")
    required_true = (
        "must_use_stagec_pair_group_folds",
        "must_report_paired_candidate_minus_control_group_differences",
        "must_report_95_percent_group_confidence_interval",
        "must_report_all_six_fold_effects",
        "must_compare_effect_to_stagec_float_precision_margin",
        "must_not_use_baseline_loss_variance_as_candidate_difference_variance",
    )
    if any(policy.get(key) is not True for key in required_true):
        raise StageCError("Stage-C effect-gate policy changed")
    if policy.get("minimum_effect_threshold_locked_here") is not False:
        raise StageCError("Stage-C incorrectly locked a candidate effect threshold")
    base = {key: value for key, value in payload.items() if key != "summary_sha256"}
    if payload.get("summary_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageCError("Stage-C summary self-hash changed")


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException, worker_started: bool
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "objective_train_opening_or_baseline_contract",
        "root_cause": "phase314b_r260_stagec_objective_train_opening_or_baseline_contract_failed",
        "required_next_path": (
            "AUDIT_R260_STAGEC_FAILURE_WITHOUT_SELECTION_FROZEN_OR_FINAL_ACCESS"
        ),
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "worker_started": bool(worker_started),
        "objective_train_open_attempt_started": bool(worker_started),
        "objective_train_access_count_added": None if worker_started else 0,
        "selection_holdout_access_count_added": 0,
        "frozen_probe_access_count_added": 0,
        "final_evaluation_access_count_added": 0,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
