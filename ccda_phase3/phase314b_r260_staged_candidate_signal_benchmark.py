"""Phase3.14b-r2.6.0 Stage D objective-train candidate-signal benchmark.

Stage C opened the new r2.6.0 objective-train role exactly once and froze the
six-fold pair-group assignment, normalization identities, persistence baseline
and numerical-resolution reference.  Stage D asks one narrower question:

    Can a deployable, low-capacity residual surrogate produce six-fold OOF
    future-cable candidates that improve the Stage-C persistence control by a
    scientifically material amount on objective train?

Only ``objective_train`` is parsed.  Selection holdout, frozen probe and final
evaluation remain sealed.  The primary candidate is pre-registered rather than
selected from a matrix: fold-local standardized ``state_action_x`` features,
full-output ridge regression with L2=10, raw ordered-cable trajectory residual,
and fixed candidate scale 1.0.  A fold-local global-mean residual and a
fixed-seed target-permutation ridge are diagnostic controls only.

No risk model, diffusion model, IDM, reverse sampler, Phase4 or CPS component
is trained.  No model weights, candidate tensors or predictions are persisted.
Only bounded metrics and cryptographic identities are written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import phase314b_r260_stagea_data_universe_genesis as stagea
from . import phase314b_r260_stagec_objective_train_baseline as stagec
from .schema_v2 import (
    DEFAULT_TF,
    DEFAULT_TH,
    FORMAL_CONDITIONS,
    N_BEADS,
    STATE_ACTION_X_DIM,
    STATE_DIM,
)

PHASE = "Phase3.14b-r2.6.0 Stage D"
SCHEMA = "phase314b_r260_staged_objective_train_candidate_signal_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_IMPLEMENTATION = "094da963c0f87fda6f70a2e4e92297b50429803a"
BASE_HEAD = "a738b096ffb1fd6b2f44ac618a13743d049dad31"
STAGEB_EVIDENCE = "5296ecb948b91058ce2f0b502bf50baffac133d8"
STAGEA_EVIDENCE = "636a2853db1a972073e2dd254e75a9724a5a1d17"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEC_REPORT = "reports/phase3_14b_r260_stagec_objective_train_baseline_summary.json"
STAGEC_CONTRACT_NPZ = stagec.CONTRACT_NPZ_RELATIVE
STAGEC_CONTRACT_MANIFEST = stagec.CONTRACT_MANIFEST_RELATIVE
OBJECTIVE_NPZ_RELATIVE = stagec.OBJECTIVE_NPZ_RELATIVE
OBJECTIVE_MANIFEST_RELATIVE = stagec.OBJECTIVE_MANIFEST_RELATIVE
FORBIDDEN_ROLE_NPZ_RELATIVES = tuple(stagec.FORBIDDEN_ROLE_NPZ_RELATIVES)

SUCCESS_REPORT = "reports/phase3_14b_r260_staged_objective_train_candidate_signal_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r260_staged_objective_train_candidate_signal_blocked_summary.json"
WRITE_AHEAD_SUFFIX = ".phase314b_r260_staged_write_ahead"
ATTEMPT_MARKER_NAME = "attempt_started.json"
WORKER_EVIDENCE_NAME = "staged_worker_evidence.json"
WORKER_STARTED_NAME = "worker_started.json"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.6.0 Stage D: benchmark objective-train candidate signal"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.6.0 Stage D candidate-signal evidence"
)
BLOCKED_SUBJECT = "Record Phase3.14b-r2.6.0 Stage D blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r260_staged_candidate_signal_benchmark.py"),
    ("A", "scripts/phase3_14b_r260_staged_worker.py"),
    ("A", "scripts/phase3_14b_r260_staged_execute.py"),
    ("A", "tests/test_phase3_14b_r260_staged_candidate_signal_benchmark.py"),
)

EXPECTED_ROWS = stagec.EXPECTED_OBJECTIVE_ROWS
EXPECTED_GROUPS = stagec.EXPECTED_OBJECTIVE_GROUPS
EXPECTED_FOLDS = stagec.EXPECTED_FOLDS
CABLE_DIM = N_BEADS * 2
OUTPUT_DIM = DEFAULT_TF * CABLE_DIM
EPSILON = 1.0e-12
Z_95 = stagec.Z_95

FALSE_BOUNDARIES = (
    "selection_holdout_opened",
    "selection_holdout_model_evaluation_run",
    "frozen_probe_opened",
    "frozen_probe_model_evaluation_run",
    "final_evaluation_opened",
    "final_evaluation_model_evaluation_run",
    "risk_fit_run",
    "diffusion_model_fit_run",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "phase4",
    "cps",
    "weights_saved",
    "checkpoint_saved",
    "candidate_tensor_persisted",
    "prediction_tensor_persisted",
)


class StageDError(RuntimeError):
    """Fail-closed Stage-D error."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str = "state_action_full_ridge_l2_10_scale_1"
    feature_source: str = "state_action_x"
    model: str = "full_output_ridge"
    ridge_l2: float = 10.0
    candidate_scale: float = 1.0
    folds: int = EXPECTED_FOLDS
    permutation_seed: int = 2600314
    standardizer_epsilon: float = 1.0e-12
    ridge_jitter: float = 1.0e-10

    def validate(self) -> None:
        if self.candidate_id != "state_action_full_ridge_l2_10_scale_1":
            raise StageDError("primary candidate ID changed")
        if self.feature_source != "state_action_x":
            raise StageDError("feature source changed")
        if self.model != "full_output_ridge":
            raise StageDError("model family changed")
        if self.ridge_l2 != 10.0 or self.candidate_scale != 1.0:
            raise StageDError("ridge or scale changed")
        if self.folds != EXPECTED_FOLDS:
            raise StageDError("fold count changed")
        if self.permutation_seed != 2600314:
            raise StageDError("permutation seed changed")
        if self.standardizer_epsilon <= 0.0 or self.ridge_jitter <= 0.0:
            raise StageDError("numerical constant invalid")


@dataclass(frozen=True)
class SignalGateSpec:
    aggregate_mse_ratio_max: float = 0.95
    group_relative_improvement_mean_min: float = 0.05
    group_relative_improvement_ci_lower_min: float = 0.01
    positive_group_rate_min: float = 0.60
    positive_fold_support_min: int = 5
    condition_mse_ratio_max: float = 0.98
    physical_valid_rate_min: float = 0.95
    primary_vs_permutation_relative_margin_min: float = 0.03
    primary_vs_global_mean_mse_ratio_margin_min: float = 0.02
    minimum_candidate_movement_rms: float = 1.0e-5

    def validate(self) -> None:
        if self.aggregate_mse_ratio_max != 0.95:
            raise StageDError("aggregate gate changed")
        if self.group_relative_improvement_mean_min != 0.05:
            raise StageDError("mean effect gate changed")
        if self.group_relative_improvement_ci_lower_min != 0.01:
            raise StageDError("CI effect gate changed")
        if self.positive_group_rate_min != 0.60:
            raise StageDError("positive group gate changed")
        if self.positive_fold_support_min != 5:
            raise StageDError("fold support gate changed")
        if self.condition_mse_ratio_max != 0.98:
            raise StageDError("condition gate changed")
        if self.physical_valid_rate_min != 0.95:
            raise StageDError("physical gate changed")
        if self.primary_vs_permutation_relative_margin_min != 0.03:
            raise StageDError("permutation margin changed")
        if self.primary_vs_global_mean_mse_ratio_margin_min != 0.02:
            raise StageDError("global-mean margin changed")
        if self.minimum_candidate_movement_rms != 1.0e-5:
            raise StageDError("movement gate changed")


PRIMARY_SPEC = CandidateSpec()
GATE_SPEC = SignalGateSpec()
PRIMARY_SPEC.validate()
GATE_SPEC.validate()


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
        raise StageDError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageDError("write-once output exists: {}".format(target))
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
            raise StageDError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageDError("{} worktree is dirty".format(label))


def validate_stagec_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    stagec.validate_summary(payload)
    expected = {
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "objective_train_opened": True,
        "objective_train_access_count_added": 1,
        "cumulative_objective_train_open_count": 1,
        "objective_train_model_fit_count": 0,
        "selection_holdout_access_count_added": 0,
        "frozen_probe_access_count_added": 0,
        "final_evaluation_access_count_added": 0,
        "required_next_path": (
            "PREREGISTER_R260_STAGED_OBJECTIVE_TRAIN_ONLY_CANDIDATE_SIGNAL_"
            "BENCHMARK_AGAINST_FROZEN_STAGEC_BASELINE_CONTRACT"
        ),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise StageDError("Stage-C summary field changed: {}".format(key))
    worker = payload.get("worker_evidence")
    if not isinstance(worker, Mapping):
        raise StageDError("Stage-C worker evidence missing")
    baseline = worker.get("analytic_baselines")
    precision = worker.get("float_precision_audit")
    if not isinstance(baseline, Mapping) or not isinstance(precision, Mapping):
        raise StageDError("Stage-C numerical references missing")
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-C evidence parent": (_git(repo, "rev-parse", BASE_HEAD + "^"), BASE_IMPLEMENTATION),
        "Stage-C implementation parent": (
            _git(repo, "rev-parse", BASE_IMPLEMENTATION + "^"),
            STAGEB_EVIDENCE,
        ),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageDError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageDError("Stage-D implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageDError("Stage-D implementation paths changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageDError("submodule worktree identity changed")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(submodule, "submodule")
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageDError("Stage-D source missing: {}".format(relative))
        if _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative)) != path.read_bytes():
            raise StageDError("Stage-D source differs from commit: {}".format(relative))
    stagec_path = repo / STAGEC_REPORT
    if not stagec_path.is_file():
        raise StageDError("Stage-C report missing")
    if _git_bytes(repo, "show", "{}:{}".format(BASE_HEAD, STAGEC_REPORT)) != stagec_path.read_bytes():
        raise StageDError("Stage-C report differs from committed evidence")
    validate_stagec_summary(load_json(stagec_path))
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_c_implementation": BASE_IMPLEMENTATION,
        "stage_c_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def metadata_preflight(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    stagec_summary = validate_stagec_summary(load_json(repo / STAGEC_REPORT))
    contract_manifest = load_json(repo / STAGEC_CONTRACT_MANIFEST)
    stagec.validate_contract_manifest(contract_manifest)
    role_manifest = load_json(repo / OBJECTIVE_MANIFEST_RELATIVE)
    if role_manifest.get("role") != "objective_train":
        raise StageDError("objective role manifest changed")
    objective_path = repo / OBJECTIVE_NPZ_RELATIVE
    contract_path = repo / STAGEC_CONTRACT_NPZ
    if not objective_path.is_file() or not contract_path.is_file():
        raise StageDError("objective or Stage-C contract NPZ missing")
    forbidden_stats = []
    for relative in FORBIDDEN_ROLE_NPZ_RELATIVES:
        path = repo / relative
        if not path.is_file():
            raise StageDError("sealed role NPZ missing: {}".format(relative))
        stat = path.stat()
        forbidden_stats.append(
            {
                "relative_path": relative,
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {
        "stage_c_summary_sha256": sha256_file(repo / STAGEC_REPORT),
        "stage_c_summary_self_sha256": stagec_summary["summary_sha256"],
        "objective_npz_relative": OBJECTIVE_NPZ_RELATIVE,
        "objective_npz_expected_sha256": role_manifest["window_npz_sha256"],
        "objective_npz_size_bytes": int(objective_path.stat().st_size),
        "stage_c_contract_npz_sha256": contract_manifest["contract_npz_sha256"],
        "stage_c_contract_size_bytes": int(contract_path.stat().st_size),
        "sealed_role_stat_only": forbidden_stats,
        "sealed_role_npz_bytes_read": False,
        "objective_npz_bytes_read": False,
    }


def load_npz_strict(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def load_objective_and_contract(root: Path) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], Mapping[str, Any]]:
    repo = Path(root).resolve()
    source_with_arrays = dict(stagec.validate_source_metadata(repo, open_objective_npz=True))
    arrays = source_with_arrays.pop("arrays")
    source = source_with_arrays
    contract = load_npz_strict(repo / STAGEC_CONTRACT_NPZ)
    stagec.validate_contract_arrays(contract)
    if not np.array_equal(contract["row_fold_id"], stagec.deterministic_group_folds(arrays["pair_group"])["row_fold_id"]):
        raise StageDError("Stage-C row-fold assignment no longer matches objective groups")
    if arrays["state_action_x"].shape != (EXPECTED_ROWS, STATE_ACTION_X_DIM):
        raise StageDError("deployable feature shape changed")
    return arrays, contract, source


def persistence_control(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    paper = np.asarray(arrays["paper_x"], dtype=np.float64).reshape(
        EXPECTED_ROWS, DEFAULT_TH, STATE_DIM
    )
    current_cable = paper[:, -1, :CABLE_DIM]
    return np.repeat(current_cable[:, None, :], DEFAULT_TF, axis=1).astype(np.float64)


def cable_target(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    target = np.asarray(arrays["y_state"], dtype=np.float64)[:, :, :CABLE_DIM]
    if target.shape != (EXPECTED_ROWS, DEFAULT_TF, CABLE_DIM):
        raise StageDError("cable target shape changed")
    return target


def fit_standardizer(features: np.ndarray, epsilon: float) -> Mapping[str, np.ndarray]:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise StageDError("standardizer input invalid")
    mean = np.mean(value, axis=0)
    std = np.std(value, axis=0)
    scale = np.where(std > float(epsilon), std, 1.0)
    return {
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "constant_mask": (std <= float(epsilon)).astype(np.bool_),
    }


def apply_standardizer(features: np.ndarray, standardizer: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (
        np.asarray(features, dtype=np.float64)
        - np.asarray(standardizer["mean"], dtype=np.float64)[None]
    ) / np.asarray(standardizer["scale"], dtype=np.float64)[None]
    if not np.all(np.isfinite(result)):
        raise StageDError("standardized features contain non-finite values")
    return result


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float,
    jitter: float,
    epsilon: float,
) -> Mapping[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(features.shape[0], -1)
    standardizer = fit_standardizer(x, epsilon)
    xz = apply_standardizer(x, standardizer)
    y_mean = np.mean(y, axis=0)
    yc = y - y_mean[None]
    gram = xz.T @ xz
    rhs = xz.T @ yc
    coefficient = np.linalg.solve(
        gram + (float(l2) + float(jitter)) * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )
    if not np.all(np.isfinite(coefficient)):
        raise StageDError("ridge coefficient contains non-finite values")
    identity = {
        "feature_mean_sha256": sha256_array(standardizer["mean"]),
        "feature_scale_sha256": sha256_array(standardizer["scale"]),
        "feature_constant_mask_sha256": sha256_array(standardizer["constant_mask"]),
        "target_mean_sha256": sha256_array(y_mean),
        "coefficient_sha256": sha256_array(coefficient),
        "coefficient_norm": float(np.linalg.norm(coefficient)),
    }
    identity["model_sha256"] = sha256_bytes(stable_json_bytes(identity))
    return {
        "standardizer": standardizer,
        "target_mean": y_mean,
        "coefficient": coefficient,
        "identity": identity,
    }


def predict_ridge(model: Mapping[str, Any], features: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    xz = apply_standardizer(features, model["standardizer"])
    flat = np.asarray(model["target_mean"], dtype=np.float64)[None] + xz @ np.asarray(
        model["coefficient"], dtype=np.float64
    )
    result = flat.reshape((features.shape[0],) + target_shape)
    if not np.all(np.isfinite(result)):
        raise StageDError("ridge prediction contains non-finite values")
    return result


def fold_geometry_bounds(target_train: np.ndarray) -> Mapping[str, np.ndarray]:
    points = np.asarray(target_train, dtype=np.float64).reshape(-1, DEFAULT_TF, N_BEADS, 2)
    lengths = np.linalg.norm(points[:, :, 1:, :] - points[:, :, :-1, :], axis=3)
    lower = np.min(lengths, axis=0)
    upper = np.max(lengths, axis=0)
    width = np.maximum(upper - lower, 1.0e-8)
    return {
        "lower": np.maximum(0.0, lower - 0.05 * width),
        "upper": upper + 0.05 * width,
    }


def physical_valid_mask(candidate: np.ndarray, bounds_by_row: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    points = np.asarray(candidate, dtype=np.float64).reshape(-1, DEFAULT_TF, N_BEADS, 2)
    lengths = np.linalg.norm(points[:, :, 1:, :] - points[:, :, :-1, :], axis=3)
    result = np.zeros(points.shape[0], dtype=np.bool_)
    for row, bounds in enumerate(bounds_by_row):
        lower = np.asarray(bounds["lower"], dtype=np.float64)
        upper = np.asarray(bounds["upper"], dtype=np.float64)
        result[row] = bool(
            np.all(np.isfinite(points[row]))
            and np.all(lengths[row] >= lower)
            and np.all(lengths[row] <= upper)
        )
    return result


def oof_predictions(
    *,
    features: np.ndarray,
    residual: np.ndarray,
    row_fold_id: np.ndarray,
    spec: CandidateSpec,
) -> Mapping[str, Any]:
    spec.validate()
    rows = features.shape[0]
    primary = np.empty_like(residual, dtype=np.float64)
    permuted = np.empty_like(residual, dtype=np.float64)
    global_mean = np.empty_like(residual, dtype=np.float64)
    assigned = np.zeros(rows, dtype=np.bool_)
    records = []
    for fold in range(spec.folds):
        test_mask = np.asarray(row_fold_id, dtype=np.int64) == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise StageDError("empty Stage-D OOF fold")
        train_y = np.asarray(residual[train_mask], dtype=np.float64)
        model = fit_ridge(
            features[train_mask],
            train_y,
            l2=spec.ridge_l2,
            jitter=spec.ridge_jitter,
            epsilon=spec.standardizer_epsilon,
        )
        fold_primary = predict_ridge(model, features[test_mask], tuple(residual.shape[1:]))
        rng = np.random.RandomState(spec.permutation_seed + fold * 1009)
        permutation = rng.permutation(int(np.sum(train_mask)))
        perm_model = fit_ridge(
            features[train_mask],
            train_y[permutation],
            l2=spec.ridge_l2,
            jitter=spec.ridge_jitter,
            epsilon=spec.standardizer_epsilon,
        )
        fold_permuted = predict_ridge(perm_model, features[test_mask], tuple(residual.shape[1:]))
        mean_residual = np.mean(train_y, axis=0)
        fold_mean = np.repeat(mean_residual[None], int(np.sum(test_mask)), axis=0)
        primary[test_mask] = fold_primary
        permuted[test_mask] = fold_permuted
        global_mean[test_mask] = fold_mean
        records.append(
            {
                "fold": fold,
                "train_rows": int(np.sum(train_mask)),
                "test_rows": int(np.sum(test_mask)),
                "primary_model_identity": copy.deepcopy(model["identity"]),
                "permutation_model_identity": copy.deepcopy(perm_model["identity"]),
                "permutation_sha256": sha256_array(permutation.astype(np.int64)),
                "primary_prediction_sha256": sha256_array(fold_primary),
                "permutation_prediction_sha256": sha256_array(fold_permuted),
                "global_mean_prediction_sha256": sha256_array(fold_mean),
            }
        )
        assigned[test_mask] = True
    if not np.all(assigned):
        raise StageDError("OOF prediction population incomplete")
    return {
        "primary": primary,
        "permuted": permuted,
        "global_mean": global_mean,
        "records": records,
        "fit_count": spec.folds * 2,
    }


def _group_effects(
    baseline_sse: np.ndarray,
    candidate_sse: np.ndarray,
    groups: np.ndarray,
) -> Mapping[str, Any]:
    names = np.asarray(groups).astype(str)
    unique = np.asarray(sorted(set(names.tolist())))
    effects = []
    ratios = []
    for group in unique:
        mask = names == group
        base = float(np.sum(baseline_sse[mask]))
        cand = float(np.sum(candidate_sse[mask]))
        ratio = cand / max(base, EPSILON)
        ratios.append(ratio)
        effects.append(1.0 - ratio)
    value = np.asarray(effects, dtype=np.float64)
    mean = float(np.mean(value))
    se = float(np.std(value, ddof=1) / math.sqrt(value.size)) if value.size > 1 else 0.0
    lower = mean - Z_95 * se
    upper = mean + Z_95 * se
    return {
        "group_count": int(value.size),
        "relative_improvement_mean": mean,
        "relative_improvement_standard_error": se,
        "relative_improvement_95_ci": [lower, upper],
        "positive_group_rate": float(np.mean(value > 0.0)),
        "effect_sha256": sha256_array(value),
        "ratio_sha256": sha256_array(np.asarray(ratios, dtype=np.float64)),
    }


def evaluate_candidate(
    *,
    name: str,
    control: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    conditions: np.ndarray,
    row_fold_id: np.ndarray,
    physical_mask: np.ndarray,
) -> Mapping[str, Any]:
    baseline_error = np.asarray(control, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    candidate_error = np.asarray(candidate, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    baseline_sse = np.sum(baseline_error * baseline_error, axis=(1, 2), dtype=np.float64)
    candidate_sse = np.sum(candidate_error * candidate_error, axis=(1, 2), dtype=np.float64)
    baseline_mse = float(np.mean(baseline_error * baseline_error))
    candidate_mse = float(np.mean(candidate_error * candidate_error))
    ratio = candidate_mse / max(baseline_mse, EPSILON)
    movement = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    fold_ratios = []
    for fold in range(EXPECTED_FOLDS):
        mask = np.asarray(row_fold_id, dtype=np.int64) == fold
        fold_ratios.append(
            float(np.sum(candidate_sse[mask]) / max(float(np.sum(baseline_sse[mask])), EPSILON))
        )
    condition_ratios = {}
    names = np.asarray(conditions).astype(str)
    for condition in FORMAL_CONDITIONS:
        mask = names == condition
        condition_ratios[condition] = float(
            np.sum(candidate_sse[mask]) / max(float(np.sum(baseline_sse[mask])), EPSILON)
        )
    return {
        "name": name,
        "row_count": int(control.shape[0]),
        "baseline_cable_mse": baseline_mse,
        "candidate_cable_mse": candidate_mse,
        "candidate_mse_ratio": ratio,
        "pooled_relative_improvement": 1.0 - ratio,
        "group_effect": _group_effects(baseline_sse, candidate_sse, groups),
        "fold_mse_ratio": fold_ratios,
        "positive_fold_support": int(sum(value < 1.0 for value in fold_ratios)),
        "condition_mse_ratio": condition_ratios,
        "physical_valid_rate": float(np.mean(np.asarray(physical_mask, dtype=np.bool_))),
        "candidate_movement_rms": float(np.sqrt(np.mean(movement * movement))),
        "candidate_sha256": sha256_array(np.asarray(candidate, dtype=np.float32)),
        "row_candidate_sse_sha256": sha256_array(candidate_sse.astype(np.float64)),
        "physical_mask_sha256": sha256_array(np.asarray(physical_mask, dtype=np.bool_)),
    }


def run_candidate_benchmark(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    arrays, contract, source = load_objective_and_contract(repo)
    stagec_summary = validate_stagec_summary(load_json(repo / STAGEC_REPORT))
    features = np.asarray(arrays["state_action_x"], dtype=np.float64)
    control = persistence_control(arrays)
    target = cable_target(arrays)
    residual = target - control
    row_fold = np.asarray(contract["row_fold_id"], dtype=np.int64)
    groups = np.asarray(arrays["pair_group"]).astype(str)
    conditions = np.asarray(arrays["condition_name"]).astype(str)
    oof = oof_predictions(
        features=features,
        residual=residual,
        row_fold_id=row_fold,
        spec=PRIMARY_SPEC,
    )
    primary_candidate = control + PRIMARY_SPEC.candidate_scale * np.asarray(oof["primary"])
    permutation_candidate = control + PRIMARY_SPEC.candidate_scale * np.asarray(oof["permuted"])
    mean_candidate = control + PRIMARY_SPEC.candidate_scale * np.asarray(oof["global_mean"])

    bounds_by_row: List[Optional[Mapping[str, np.ndarray]]] = [None] * EXPECTED_ROWS
    for fold in range(EXPECTED_FOLDS):
        test_mask = row_fold == fold
        train_mask = ~test_mask
        bounds = fold_geometry_bounds(target[train_mask])
        for row in np.flatnonzero(test_mask):
            bounds_by_row[int(row)] = bounds
    if any(value is None for value in bounds_by_row):
        raise StageDError("geometry bounds population incomplete")
    typed_bounds = [value for value in bounds_by_row if value is not None]
    primary_valid = physical_valid_mask(primary_candidate, typed_bounds)
    permutation_valid = physical_valid_mask(permutation_candidate, typed_bounds)
    mean_valid = physical_valid_mask(mean_candidate, typed_bounds)

    evaluations = {
        "primary": evaluate_candidate(
            name="primary",
            control=control,
            candidate=primary_candidate,
            target=target,
            groups=groups,
            conditions=conditions,
            row_fold_id=row_fold,
            physical_mask=primary_valid,
        ),
        "permutation_control": evaluate_candidate(
            name="permutation_control",
            control=control,
            candidate=permutation_candidate,
            target=target,
            groups=groups,
            conditions=conditions,
            row_fold_id=row_fold,
            physical_mask=permutation_valid,
        ),
        "global_mean_control": evaluate_candidate(
            name="global_mean_control",
            control=control,
            candidate=mean_candidate,
            target=target,
            groups=groups,
            conditions=conditions,
            row_fold_id=row_fold,
            physical_mask=mean_valid,
        ),
    }
    primary = evaluations["primary"]
    permuted = evaluations["permutation_control"]
    global_mean = evaluations["global_mean_control"]
    stagec_margin = float(
        stagec_summary["worker_evidence"]["float_precision_audit"][
            "recommended_future_candidate_numerical_margin"
        ]
    )
    gates = {
        "aggregate_mse_ratio": primary["candidate_mse_ratio"] <= GATE_SPEC.aggregate_mse_ratio_max,
        "group_relative_improvement_mean": (
            primary["group_effect"]["relative_improvement_mean"]
            >= GATE_SPEC.group_relative_improvement_mean_min
        ),
        "group_relative_improvement_ci_lower": (
            primary["group_effect"]["relative_improvement_95_ci"][0]
            >= GATE_SPEC.group_relative_improvement_ci_lower_min
        ),
        "effect_above_stagec_numerical_margin": (
            primary["group_effect"]["relative_improvement_95_ci"][0] > stagec_margin
        ),
        "positive_group_rate": (
            primary["group_effect"]["positive_group_rate"] >= GATE_SPEC.positive_group_rate_min
        ),
        "positive_fold_support": (
            primary["positive_fold_support"] >= GATE_SPEC.positive_fold_support_min
        ),
        "both_conditions_improve": all(
            value <= GATE_SPEC.condition_mse_ratio_max
            for value in primary["condition_mse_ratio"].values()
        ),
        "physical_valid_rate": primary["physical_valid_rate"] >= GATE_SPEC.physical_valid_rate_min,
        "beats_permutation": (
            primary["pooled_relative_improvement"]
            - permuted["pooled_relative_improvement"]
            >= GATE_SPEC.primary_vs_permutation_relative_margin_min
        ),
        "beats_global_mean": (
            global_mean["candidate_mse_ratio"] - primary["candidate_mse_ratio"]
            >= GATE_SPEC.primary_vs_global_mean_mse_ratio_margin_min
        ),
        "candidate_moves": (
            primary["candidate_movement_rms"] >= GATE_SPEC.minimum_candidate_movement_rms
        ),
    }
    gates["all"] = bool(all(gates.values()))
    ready = bool(gates["all"])
    recommendation = None
    if ready:
        recommendation = {
            "candidate_id": PRIMARY_SPEC.candidate_id,
            "feature_source": PRIMARY_SPEC.feature_source,
            "feature_deployability": "observation_and_past_action_only",
            "model": PRIMARY_SPEC.model,
            "ridge_l2": PRIMARY_SPEC.ridge_l2,
            "candidate_scale": PRIMARY_SPEC.candidate_scale,
            "fold_assignment_source": "frozen_r260_stagec_contract",
            "selection_role": "objective_train_oof_candidate_signal_recommendation",
            "selection_holdout_access_authorized": False,
            "risk_model_authorized": False,
        }
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "primary_failure_locus": (
            "objective_train_candidate_signal_confirmed"
            if ready
            else "objective_train_candidate_signal_gate"
        ),
        "root_cause": (
            "phase314b_r260_staged_deployable_ridge_candidate_has_material_group_stable_oof_signal"
            if ready
            else "phase314b_r260_staged_deployable_ridge_candidate_lacks_material_group_stable_oof_signal"
        ),
        "required_next_path": (
            "PREREGISTER_R260_STAGEE_OBJECTIVE_TRAIN_ONLY_PHYSICAL_CANDIDATE_LOCK_AND_RISK_LABEL_AUDIT"
            if ready
            else "AUDIT_R260_STAGED_CANDIDATE_SIGNAL_FAILURE_WITHOUT_HOLDOUT_PROBE_OR_FINAL_ACCESS"
        ),
        "source": source,
        "stage_c_reference": {
            "summary_sha256": sha256_file(repo / STAGEC_REPORT),
            "summary_self_sha256": stagec_summary["summary_sha256"],
            "contract_npz_sha256": sha256_file(repo / STAGEC_CONTRACT_NPZ),
            "fold_assignment_sha256": load_json(repo / STAGEC_CONTRACT_MANIFEST)[
                "fold_assignment_sha256"
            ],
            "persistence_cable_mse": stagec_summary["worker_evidence"]["analytic_baselines"][
                "persistence"
            ]["cable_mse"],
            "numerical_margin": stagec_margin,
        },
        "candidate_spec": asdict(PRIMARY_SPEC),
        "gate_spec": asdict(GATE_SPEC),
        "fit_contract": {
            "fold_local_feature_standardization": True,
            "condition_name_used_as_feature": False,
            "success_or_contact_label_used_as_feature": False,
            "target": "raw_ordered_cable_trajectory_residual_from_persistence",
            "primary_scale_selected_from_data": False,
            "candidate_matrix_search_run": False,
            "permutation_control_is_diagnostic_only": True,
            "global_mean_control_is_diagnostic_only": True,
        },
        "oof": {
            "primary_prediction_sha256": sha256_array(np.asarray(oof["primary"], dtype=np.float64)),
            "permutation_prediction_sha256": sha256_array(np.asarray(oof["permuted"], dtype=np.float64)),
            "global_mean_prediction_sha256": sha256_array(np.asarray(oof["global_mean"], dtype=np.float64)),
            "fold_records": oof["records"],
        },
        "evaluations": evaluations,
        "gates": gates,
        "execution_counts": {
            "objective_train_logical_open_count": 1,
            "objective_train_file_hash_count": 1,
            "objective_train_npz_parse_count": 1,
            "stage_c_contract_npz_parse_count": 1,
            "direction_fit_count": int(oof["fit_count"]),
            "candidate_generation_count": 3,
            "candidate_evaluation_count": 3,
            "risk_fit_count": 0,
            "diffusion_model_fit_count": 0,
            "selection_holdout_npz_open_count": 0,
            "frozen_probe_npz_open_count": 0,
            "final_evaluation_npz_open_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": recommendation,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["direction_fit_run"] = True
    result["candidate_generation_run"] = True
    result["candidate_evaluation_run"] = True
    result["worker_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_worker_result(result)
    return result


def validate_worker_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageDError("Stage-D worker identity changed")
    if payload.get("selected_configuration") is not None:
        raise StageDError("Stage-D selected a configuration")
    for key in FALSE_BOUNDARIES:
        if payload.get(key) is not False:
            raise StageDError("Stage-D crossed forbidden boundary: {}".format(key))
    if payload.get("direction_fit_run") is not True:
        raise StageDError("Stage-D direction fit flag missing")
    if payload.get("candidate_generation_run") is not True or payload.get("candidate_evaluation_run") is not True:
        raise StageDError("Stage-D candidate flags missing")
    counts = payload.get("execution_counts")
    expected_counts = {
        "objective_train_logical_open_count": 1,
        "objective_train_file_hash_count": 1,
        "objective_train_npz_parse_count": 1,
        "stage_c_contract_npz_parse_count": 1,
        "direction_fit_count": 12,
        "candidate_generation_count": 3,
        "candidate_evaluation_count": 3,
        "risk_fit_count": 0,
        "diffusion_model_fit_count": 0,
        "selection_holdout_npz_open_count": 0,
        "frozen_probe_npz_open_count": 0,
        "final_evaluation_npz_open_count": 0,
    }
    if counts != expected_counts:
        raise StageDError("Stage-D execution counts changed")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or "all" not in gates:
        raise StageDError("Stage-D gates missing")
    ready = bool(gates["all"])
    if payload.get("scientific_status") != ("READY" if ready else "BLOCKED"):
        raise StageDError("Stage-D scientific status disagrees with gates")
    if ready != (payload.get("train_only_recommendation") is not None):
        raise StageDError("Stage-D recommendation disagrees with gates")
    base = {key: value for key, value in payload.items() if key != "worker_result_sha256"}
    if payload.get("worker_result_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageDError("Stage-D worker self-hash changed")


def build_summary(
    *,
    repository: Mapping[str, Any],
    worker: Mapping[str, Any],
    worker_file_sha256: str,
) -> Mapping[str, Any]:
    validate_worker_result(worker)
    ready = worker["scientific_status"] == "READY"
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "repository": copy.deepcopy(dict(repository)),
        "worker_evidence": copy.deepcopy(dict(worker)),
        "worker_evidence_file_sha256": worker_file_sha256,
        "objective_train_opened": True,
        "objective_train_access_count_added": 1,
        "cumulative_objective_train_open_count": 2,
        "objective_train_direction_fit_count": 12,
        "objective_train_candidate_evaluation_count": 3,
        "selection_holdout_access_count_added": 0,
        "frozen_probe_access_count_added": 0,
        "final_evaluation_access_count_added": 0,
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(worker["train_only_recommendation"]),
        "candidate_signal_ready": ready,
        "risk_model_authorized_by_this_report": False,
        "selection_holdout_access_authorized_by_this_report": False,
        "frozen_probe_access_authorized_by_this_report": False,
        "final_evaluation_access_authorized_by_this_report": False,
        "rerun_authorized": False,
        "resume_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["direction_fit_run"] = True
    payload["candidate_generation_run"] = True
    payload["candidate_evaluation_run"] = True
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageDError("Stage-D summary identity changed")
    expected = {
        "objective_train_opened": True,
        "objective_train_access_count_added": 1,
        "cumulative_objective_train_open_count": 2,
        "objective_train_direction_fit_count": 12,
        "objective_train_candidate_evaluation_count": 3,
        "selection_holdout_access_count_added": 0,
        "frozen_probe_access_count_added": 0,
        "final_evaluation_access_count_added": 0,
        "selected_configuration": None,
        "risk_model_authorized_by_this_report": False,
        "selection_holdout_access_authorized_by_this_report": False,
        "frozen_probe_access_authorized_by_this_report": False,
        "final_evaluation_access_authorized_by_this_report": False,
        "rerun_authorized": False,
        "resume_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageDError("Stage-D summary field changed: {}".format(key))
    for key in FALSE_BOUNDARIES:
        if payload.get(key) is not False:
            raise StageDError("Stage-D summary crossed forbidden boundary: {}".format(key))
    if payload.get("direction_fit_run") is not True or payload.get("candidate_generation_run") is not True or payload.get("candidate_evaluation_run") is not True:
        raise StageDError("Stage-D summary science flags missing")
    worker = payload.get("worker_evidence")
    if not isinstance(worker, Mapping):
        raise StageDError("Stage-D worker evidence missing")
    validate_worker_result(worker)
    ready = worker["scientific_status"] == "READY"
    if payload.get("candidate_signal_ready") is not ready:
        raise StageDError("Stage-D readiness changed")
    if (payload.get("train_only_recommendation") is not None) != ready:
        raise StageDError("Stage-D recommendation changed")
    base = {key: value for key, value in payload.items() if key != "summary_sha256"}
    if payload.get("summary_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageDError("Stage-D summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    worker_started: bool,
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "objective_train_candidate_signal_execution",
        "root_cause": "phase314b_r260_staged_execution_or_candidate_signal_contract_failed",
        "required_next_path": (
            "AUDIT_R260_STAGED_FAILURE_WITHOUT_SELECTION_FROZEN_OR_FINAL_ACCESS"
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
        "risk_model_authorized_by_this_report": False,
        "rerun_authorized": False,
        "resume_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
