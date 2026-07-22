"""Phase3.14b-r2.5.8 Stage T aligned-gate candidate-matrix confirmation.

Stage-S Resume3 confirmed that the tolerance-aligned upper gate restores
non-zero objective-train OOF support in all 27 backbone/timestep cells and that
this functional result is byte-exact across two cold science workers.  Stage T
freezes that aligned gate as a *shadow scientific contract* and evaluates the
27-cell OOF candidate matrix on objective-train only.

The stage does not write the aligned gate into Stage E, does not access the
selection holdout or frozen probe, and does not select a final configuration.
It records scalar candidate-fidelity evidence and a deterministic candidate
frontier.  Prediction, direction, candidate, mask, callback, and model tensors
are never persisted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.8 Stage T"
SCHEMA = "phase314b_r258_staget_aligned_gate_candidate_matrix_v1"
WORKER_SCHEMA = "phase314b_r258_staget_aligned_gate_candidate_matrix_worker_v1"
CONTRACT_SCHEMA = "phase314b_r258_staget_tolerance_aligned_gate_contract_v1"
BLOCKED_SCHEMA = "phase314b_r258_staget_aligned_gate_candidate_matrix_blocked_v1"

BASE_STAGES_RESUME3_IMPLEMENTATION_COMMIT = "6bcbed8e7d7142aeef0250b9d592cc71d9b280ba"
BASE_STAGES_RESUME3_EVIDENCE_COMMIT = "ccc2be0f872b5ecb6057d4e34c4f5eaa889d4553"
EXPECTED_STAGES_RESUME3_PARENT = "38b096bf5820cdb2f1cadde78149ce389c464352"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_REPORT = (
    "reports/phase3_14b_r258_stages_resume3_"
    "cold_worker_oof_confirmation_summary.json"
)
EXPECTED_BASE_REPORT_SHA256 = "ea1f7d641164d7608435a33dc6cc79eaa72906603d7f03ecb9c721699f4d29a2"
EXPECTED_BASE_SCIENTIFIC_SHA256 = "9316b92cc349a3b689f3336066f6f768e0f6e7022e48368e964ab6211ae2da64"
EXPECTED_BASE_CURRENT_FIT_SHA256 = "cebd647b7748caa1510e924b2b9eca98ebe2273affc467725284c3fbfcf376b4"
EXPECTED_BASE_FUNCTIONAL_SHA256 = "52c2bff55ab15ff9514c5cf349006fd73cd7c5b78bf84e1a08f0aa6453e92fff"
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stages_resume3_tolerance_aligned_oof_support_confirmed"
)
EXPECTED_BASE_NEXT_PATH = (
    "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_OOF_CANDIDATE_MATRIX_"
    "ON_OBJECTIVE_TRAIN_ONLY"
)

SUCCESS_REPORT = "reports/phase3_14b_r258_staget_oof_candidate_matrix_summary.json"
GATE_CONTRACT_REPORT = "reports/phase3_14b_r258_staget_tolerance_aligned_gate_contract.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_staget_oof_candidate_matrix_blocked_summary.json"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage T: freeze aligned gate and confirm OOF candidate matrix"
)
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage T candidate-matrix evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage T blocked evidence"
BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S Resume3: confirm OOF support in cold workers"
)
BASE_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume3 OOF confirmation evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_staget_aligned_gate_candidate_matrix.py"),
    ("A", "scripts/phase3_14b_r258_staget_worker.py"),
    ("A", "scripts/phase3_14b_r258_staget_execute.py"),
    ("A", "tests/test_phase3_14b_r258_staget_aligned_gate_candidate_matrix.py"),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stages_resume3_cold_worker_oof_confirmation.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume3_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume3_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stages_resume3_cold_worker_oof_confirmation.py"),
)
BASE_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (("A", BASE_REPORT),)

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stages_resume3_cold_worker_oof_confirmation.py": (
        "f775f3e0427df383c11de3629517505791fa0b674a3be124ed65ef02e3e27a8c"
    ),
    "scripts/phase3_14b_r258_stages_resume3_worker.py": (
        "420aa7a2c3d0ad6ba14ae22d6028082c1bda12d352b48154d4014e1c7c6eba27"
    ),
    "scripts/phase3_14b_r258_stages_resume3_execute.py": (
        "7260f6fdfb1022ce086e905293fb936e9a354d90d78c1f4012de7cfa84eba931"
    ),
    "tests/test_phase3_14b_r258_stages_resume3_cold_worker_oof_confirmation.py": (
        "1d0ebcfac5c9b096b130559790a837ef82ac6d27e410e417b1b75958c9b01ec8"
    ),
    "ccda_phase3/phase314b_r258_stager_resume1_portable_oof_functional_replay.py": (
        "abc252e0ac8c401c2b1c48d6122f66023061bc07892f4438dce8738d0680d4b5"
    ),
    "ccda_phase3/phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit.py": (
        "9f8da64be049bb9cd9a61e229e4dc20e9a91cccc55261f43511a76f600fb4ed0"
    ),
}

EXPECTED_ENV: Mapping[str, str] = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "selection_holdout_evaluated",
    "selection_holdout_used_for_fit_or_selection",
    "frozen_probe_accessed",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "candidate_execution",
    "deformable_ravens_executed",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_persisted",
    "surrogate_weights_persisted",
    "prediction_tensor_persisted",
    "direction_tensor_persisted",
    "candidate_tensor_persisted",
    "predicate_tensor_persisted",
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)

BASE_FALSE_BOUNDARIES: Tuple[str, ...] = tuple(
    key for key in FALSE_BOUNDARIES if key != "direction_tensor_persisted"
)

EXPECTED_WORKER_COUNT = 2
EXPECTED_CELL_COUNT = 27
EXPECTED_BACKBONE_COUNT = 9
EXPECTED_TIMESTEPS = (10, 25, 50)
EXPECTED_SCIENCE_FITS_PER_WORKER = 27
EXPECTED_REPEAT_FITS_PER_WORKER = 27
EXPECTED_TOTAL_FITS_PER_WORKER = 54
EXPECTED_CALLBACK_PAIRS_PER_WORKER = 27
EXPECTED_SHADOW_ATTEMPTS_PER_WORKER = 189
EXPECTED_MATRIX_RECONSTRUCTION_ATTEMPTS_PER_WORKER = 189
EXPECTED_TOTAL_FITS = 108
EXPECTED_TOTAL_CALLBACK_PAIRS = 54
EXPECTED_TOTAL_SHADOW_ATTEMPTS = 378
EXPECTED_TOTAL_MATRIX_RECONSTRUCTION_ATTEMPTS = 378
EXPECTED_EXTERNAL_MULTIPLIER = 0.25
EXPECTED_TOLERANCE_FACTOR = 5.0


class StageTError(RuntimeError):
    """Fail-closed Stage-T error."""


@dataclass(frozen=True)
class CandidatePolicy:
    relative_epsilon: float = 1.0e-12
    motion_epsilon: float = 1.0e-12
    minimum_positive_reduction_rate: float = 0.5

    def validate(self) -> None:
        if not math.isfinite(self.relative_epsilon) or self.relative_epsilon <= 0.0:
            raise StageTError("relative epsilon is invalid")
        if not math.isfinite(self.motion_epsilon) or self.motion_epsilon <= 0.0:
            raise StageTError("motion epsilon is invalid")
        if not 0.0 < self.minimum_positive_reduction_rate <= 1.0:
            raise StageTError("positive-reduction threshold is invalid")


def stable_json_bytes(value: Any) -> bytes:
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageTError(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageTError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageTError(f"{label} is not a sequence")
    return value


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageTError(f"{label} boundary changed: {key}={mapping.get(key)!r}")


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageTError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _run_git(root: Path, *args: str, text: bool = True) -> Any:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", "replace")
        raise StageTError(f"git {' '.join(args)} failed: {stderr.strip()}")
    return completed.stdout


def _git(root: Path, *args: str) -> str:
    return str(_run_git(root, *args, text=True)).strip()


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageTError("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise StageTError(f"{label} worktree is dirty")


def validate_base_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageTError("Stage-S Resume3 execution did not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageTError("Stage-S Resume3 scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageTError("Stage-S Resume3 root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageTError("Stage-S Resume3 next path changed")
    if report.get("scientific_result_sha256") != EXPECTED_BASE_SCIENTIFIC_SHA256:
        raise StageTError("Stage-S Resume3 scientific SHA changed")
    if report.get("selected_configuration") is not None:
        raise StageTError("Stage-S Resume3 selected a configuration")
    if report.get("train_only_recommendation") is not None:
        raise StageTError("Stage-S Resume3 emitted a recommendation")
    require_false(report, BASE_FALSE_BOUNDARIES, "Stage-S Resume3 report")
    execution = _mapping(report.get("confirmation_execution"), "confirmation execution")
    if execution.get("worker_count") != EXPECTED_WORKER_COUNT:
        raise StageTError("Stage-S Resume3 worker count changed")
    if execution.get("total_oof_fit_count") != EXPECTED_TOTAL_FITS:
        raise StageTError("Stage-S Resume3 OOF fit count changed")
    if execution.get("total_callback_pair_count") != EXPECTED_TOTAL_CALLBACK_PAIRS:
        raise StageTError("Stage-S Resume3 callback count changed")
    if execution.get("total_internal_scale_attempt_count") != EXPECTED_TOTAL_SHADOW_ATTEMPTS:
        raise StageTError("Stage-S Resume3 attempt count changed")
    if execution.get("processes_distinct") is not True:
        raise StageTError("Stage-S Resume3 process separation changed")
    if execution.get("workers_sequential") is not True:
        raise StageTError("Stage-S Resume3 worker ordering changed")
    if report.get("current_fit_projection_sha256") not in (None, EXPECTED_BASE_CURRENT_FIT_SHA256):
        raise StageTError("Stage-S Resume3 current-fit SHA changed")
    if report.get("functional_projection_sha256") not in (None, EXPECTED_BASE_FUNCTIONAL_SHA256):
        raise StageTError("Stage-S Resume3 functional SHA changed")
    summary = _mapping(report.get("confirmation_summary"), "confirmation summary")
    required_summary = {
        "cell_count": EXPECTED_CELL_COUNT,
        "nonzero_support_cell_count": EXPECTED_CELL_COUNT,
        "zero_acceptance_cell_count": 0,
        "oracle_like_cell_count": 7,
        "dominant_discriminator": "direction_retention",
        "dominant_support_count": 2,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise StageTError(f"Stage-S Resume3 summary changed: {key}")
    return report


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageTError("Stage T requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_STAGES_RESUME3_EVIDENCE_COMMIT:
        raise StageTError("Stage-T implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageTError("Stage-T implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageTError("Stage-T implementation paths changed")
    if _git(repo, "rev-parse", f"{BASE_STAGES_RESUME3_EVIDENCE_COMMIT}^") != BASE_STAGES_RESUME3_IMPLEMENTATION_COMMIT:
        raise StageTError("Stage-S Resume3 evidence parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGES_RESUME3_IMPLEMENTATION_COMMIT}^") != EXPECTED_STAGES_RESUME3_PARENT:
        raise StageTError("Stage-S Resume3 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGES_RESUME3_IMPLEMENTATION_COMMIT) != BASE_IMPLEMENTATION_SUBJECT:
        raise StageTError("Stage-S Resume3 implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGES_RESUME3_EVIDENCE_COMMIT) != BASE_EVIDENCE_SUBJECT:
        raise StageTError("Stage-S Resume3 evidence subject changed")
    if commit_name_status(repo, BASE_STAGES_RESUME3_IMPLEMENTATION_COMMIT) != tuple(sorted(BASE_IMPLEMENTATION_PATHS)):
        raise StageTError("Stage-S Resume3 implementation paths changed")
    if commit_name_status(repo, BASE_STAGES_RESUME3_EVIDENCE_COMMIT) != tuple(sorted(BASE_EVIDENCE_PATHS)):
        raise StageTError("Stage-S Resume3 evidence paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageTError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageTError("DeformableRavens commit changed")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(submodule, "DeformableRavens")
    report_path = repo / BASE_REPORT
    if sha256_file(report_path) != EXPECTED_BASE_REPORT_SHA256:
        raise StageTError("Stage-S Resume3 report SHA changed")
    report = validate_base_report(load_json(report_path))
    source_sha: Dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file():
            raise StageTError(f"frozen source missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise StageTError(f"frozen source SHA changed: {relative}")
        source_sha[relative] = actual
    for relative in (SUCCESS_REPORT, GATE_CONTRACT_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageTError(f"Stage-T output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_STAGES_RESUME3_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_SHA256,
        "base_current_fit_projection_sha256": EXPECTED_BASE_CURRENT_FIT_SHA256,
        "base_functional_projection_sha256": EXPECTED_BASE_FUNCTIONAL_SHA256,
        "frozen_source_sha256": source_sha,
        "base_report": copy.deepcopy(dict(report)),
    }




def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise StageTError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise StageTError(f"{label} is non-finite")
    return result


def _required_int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise StageTError(f"{label} is not an integer")
    return int(value)

def scalar_stats(values: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StageTError("cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def zero_stats() -> Mapping[str, Any]:
    return {key: 0 if key == "count" else 0.0 for key in (
        "count", "mean", "std", "min", "p05", "median", "p95", "max"
    )}


def _safe_ratio(numerator: float, denominator: float, epsilon: float) -> float:
    value = float(numerator) / max(float(denominator), float(epsilon))
    if not math.isfinite(value):
        raise StageTError("ratio is non-finite")
    return value


def compute_candidate_fidelity_metrics(
    *,
    control: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    selected_scale: np.ndarray,
    policy: Optional[CandidatePolicy] = None,
) -> Mapping[str, Any]:
    active = CandidatePolicy() if policy is None else policy
    active.validate()
    control_value = np.asarray(control, dtype=np.float64)
    candidate_value = np.asarray(candidate, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    if control_value.shape != candidate_value.shape or control_value.shape != target_value.shape:
        raise StageTError("control/candidate/target shape changed")
    if control_value.ndim < 2 or control_value.shape[0] == 0:
        raise StageTError("candidate population is invalid")
    if not np.all(np.isfinite(control_value)) or not np.all(np.isfinite(candidate_value)) or not np.all(np.isfinite(target_value)):
        raise StageTError("candidate fidelity inputs are non-finite")
    scale = np.asarray(selected_scale, dtype=np.float64).reshape(control_value.shape[0])
    if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
        raise StageTError("selected scale is invalid")
    selected = scale > 0.0
    selected_count = int(np.count_nonzero(selected))
    acceptance = float(np.mean(selected))
    control_delta = (target_value - control_value).reshape(control_value.shape[0], -1)
    candidate_delta = (target_value - candidate_value).reshape(control_value.shape[0], -1)
    motion = (candidate_value - control_value).reshape(control_value.shape[0], -1)
    control_distance = np.sqrt(np.sum(control_delta * control_delta, axis=1))
    candidate_distance = np.sqrt(np.sum(candidate_delta * candidate_delta, axis=1))
    motion_norm = np.sqrt(np.sum(motion * motion, axis=1))
    reduction = control_distance - candidate_distance
    relative_reduction = reduction / np.maximum(control_distance, active.relative_epsilon)
    control_sse = float(np.sum(control_delta * control_delta))
    candidate_sse = float(np.sum(candidate_delta * candidate_delta))
    overall_mse_ratio = _safe_ratio(candidate_sse, control_sse, active.relative_epsilon)
    motion_positive = motion_norm > active.motion_epsilon
    if np.any(selected != motion_positive):
        raise StageTError("selected scale and candidate motion population differ")
    if selected_count:
        accepted_control_sse = float(np.sum(control_delta[selected] ** 2))
        accepted_candidate_sse = float(np.sum(candidate_delta[selected] ** 2))
        accepted_mse_ratio = _safe_ratio(
            accepted_candidate_sse,
            accepted_control_sse,
            active.relative_epsilon,
        )
        accepted_reduction = reduction[selected]
        accepted_relative = relative_reduction[selected]
        accepted_motion = motion_norm[selected]
        positive_rate = float(np.mean(accepted_reduction > 0.0))
        accepted_metrics: Mapping[str, Any] = {
            "control_distance": scalar_stats(control_distance[selected]),
            "candidate_distance": scalar_stats(candidate_distance[selected]),
            "distance_reduction": scalar_stats(accepted_reduction),
            "relative_distance_reduction": scalar_stats(accepted_relative),
            "motion_norm": scalar_stats(accepted_motion),
            "mse_ratio": accepted_mse_ratio,
            "positive_distance_reduction_rate": positive_rate,
        }
    else:
        accepted_mse_ratio = 1.0
        positive_rate = 0.0
        accepted_metrics = {
            "control_distance": zero_stats(),
            "candidate_distance": zero_stats(),
            "distance_reduction": zero_stats(),
            "relative_distance_reduction": zero_stats(),
            "motion_norm": zero_stats(),
            "mse_ratio": 1.0,
            "positive_distance_reduction_rate": 0.0,
        }
    mechanism_eligible = selected_count > 0 and acceptance > 0.0
    fidelity_eligible = (
        mechanism_eligible
        and overall_mse_ratio < 1.0
        and accepted_mse_ratio < 1.0
        and positive_rate > active.minimum_positive_reduction_rate
        and float(accepted_metrics["relative_distance_reduction"]["mean"]) > 0.0
    )
    return {
        "row_count": int(control_value.shape[0]),
        "selected_row_count": selected_count,
        "acceptance_rate": acceptance,
        "overall_control_distance": scalar_stats(control_distance),
        "overall_candidate_distance": scalar_stats(candidate_distance),
        "overall_distance_reduction": scalar_stats(reduction),
        "overall_relative_distance_reduction": scalar_stats(relative_reduction),
        "overall_motion_norm": scalar_stats(motion_norm),
        "overall_mse_ratio": overall_mse_ratio,
        "accepted_rows": accepted_metrics,
        "selected_scale_stats": scalar_stats(scale),
        "candidate_motion_positive_rate": float(np.mean(motion_positive)),
        "mechanism_eligible": mechanism_eligible,
        "fidelity_eligible": fidelity_eligible,
        "eligibility_policy": asdict(active),
    }


def _gate_contract_observation(
    *,
    contract: Mapping[str, Any],
    upper_bound: np.ndarray,
    segment_tolerance: float,
    tolerance_factor: float,
    sha256_array: Any,
) -> Mapping[str, Any]:
    observation: Dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "contract_role": "shadow_frozen_not_written_to_stagee",
        "length_space_rule": "segment_length <= frozen_upper + tolerance_factor * segment_tolerance",
        "log_z_rule": "z <= (log(frozen_upper + tolerance_factor * segment_tolerance) - center_log) / scale_log",
        "tolerance_factor": float(tolerance_factor),
        "segment_tolerance": float(segment_tolerance),
        "length_tolerance": float(contract["length_tolerance"]),
        "upper_bound_sha256": sha256_array(np.asarray(upper_bound, dtype=np.float64)),
        "length_upper_sha256": str(contract["length_upper_sha256"]),
        "position_z_threshold_sha256": str(contract["position_z_threshold_sha256"]),
        "position_shape": list(np.asarray(upper_bound).shape),
        "length_log_z_element_equivalence_required": True,
        "stagee_modified": False,
        "legacy_upper_gate_changed": False,
        "aligned_gate_written_to_stagee": False,
        "objective_train_only": True,
    }
    observation["contract_sha256"] = sha256_bytes(stable_json_bytes(observation))
    return observation


def reconstruct_aligned_candidate_metrics(
    *,
    base_cell: Mapping[str, Any],
    control: np.ndarray,
    base_direction: np.ndarray,
    target: np.ndarray,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    spec: Any,
    policy: Optional[CandidatePolicy] = None,
) -> Mapping[str, Any]:
    """Independently reconstruct the Stage-R aligned candidate without callbacks."""
    stageq = runtime["stageq"]
    stagel = runtime["stagel"]
    stager = runtime["stager"]
    stagee = stagel.stagee258
    stageb = stagee.stageb
    staged = stagee.staged
    stagef = runtime["stagef"]
    integrator_spec = runtime["integrator_spec"]
    definition = stagef.fixed_integrator_definition()
    definition.validate()
    control_raw = np.asarray(control, dtype=np.float32)
    direction = np.asarray(base_direction, dtype=np.float64)
    target_raw = np.asarray(target, dtype=np.float32)
    if control_raw.shape != direction.shape or control_raw.shape != target_raw.shape:
        raise StageTError("Stage-T control/direction/target shape changed")
    proposed_direction = direction * float(spec.external_multiplier)
    expected_order = tuple(float(value) for value in stageq._expected_internal_order(runtime))
    if len(expected_order) != 7:
        raise StageTError("Stage-T internal-scale population changed")
    bounds = stagee.segment_bounds(definition=definition, context=context)
    upper = np.asarray(bounds["upper"], dtype=np.float64)
    lower = np.asarray(bounds["lower"], dtype=np.float64)
    reference = context["stage_d_contract"].reference
    contract = stageq.tolerant_upper_contract(
        upper_bound=upper,
        center_log=np.asarray(reference.center_log, dtype=np.float64),
        scale_log=np.asarray(reference.scale_log, dtype=np.float64),
        segment_tolerance=float(integrator_spec.segment_tolerance),
        tolerance_factor=float(spec.reconstruction_tolerance_factor),
    )
    selected = np.zeros(control_raw.shape[0], dtype=np.bool_)
    selected_scale = np.zeros(control_raw.shape[0], dtype=np.float64)
    output = control_raw.copy()
    mismatch_count = 0
    aligned_upper_failures = 0
    strict_pass_aligned_fail = 0
    attempt_count = 0
    for scale in expected_order:
        attempt_count += 1
        raw_proposal = (
            control_raw.astype(np.float64) + float(scale) * proposed_direction
        ).astype(np.float32)
        reconstruction = stagee.reconstruct_segment_vectors(
            proposed=raw_proposal,
            control=control_raw,
            lower=lower,
            upper=upper,
            coordinate_abs_max=float(context["historical_geometry"].coordinate_abs_max),
            epsilon=float(integrator_spec.segment_tolerance),
        )
        candidate = np.asarray(reconstruction["candidate"], dtype=np.float32)
        bound_pass = np.asarray(reconstruction["bound_pass"], dtype=np.bool_)
        coordinate_possible = np.asarray(
            reconstruction["coordinate_possible"], dtype=np.bool_
        )
        metrics = stagee.observable_row_metrics(
            candidate=candidate,
            raw_proposal=raw_proposal,
            control=control_raw,
            direction=proposed_direction,
            definition=definition,
            context=context,
            spec=integrator_spec,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
            include_topology=False,
        )
        lengths = stageb.segment_lengths(candidate).astype(np.float64)
        scores = staged.segment_scores(candidate, reference)
        aligned = stageq.aligned_upper_masks(
            lengths=lengths,
            z_scores=np.asarray(scores["z"], dtype=np.float64),
            contract=contract,
        )
        aligned_pass = np.asarray(aligned["length_row_pass"], dtype=np.bool_)
        strict_pass = np.asarray(metrics["upper_pass"], dtype=np.bool_)
        mismatch_count += int(aligned["element_mismatch_count"])
        aligned_upper_failures += int(
            np.count_nonzero(~np.asarray(aligned["length_element_pass"], dtype=np.bool_))
        )
        strict_pass_aligned_fail += int(np.count_nonzero(strict_pass & ~aligned_pass))
        masks = stageq._predicate_masks(
            metrics=metrics,
            aligned_upper_pass=aligned_pass,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
        )
        active = ~selected
        fast = active.copy()
        for name in stager.PREDICATE_ORDER[:-1]:
            fast &= np.asarray(masks[name], dtype=np.bool_)
        topology = np.zeros(control_raw.shape[0], dtype=np.bool_)
        checked = np.flatnonzero(fast)
        if checked.size:
            topology[checked] = stageb.physical_validity(
                candidate[checked, None],
                context["historical_geometry"],
            )["topology"][:, 0]
        sequential = stageq.sequential_attempt_summary(
            active=active,
            predicate_masks=masks,
            topology_pass=topology,
        )
        choose = np.asarray(sequential["accepted_mask"], dtype=np.bool_)
        if not np.array_equal(choose, fast & topology):
            raise StageTError("Stage-T acceptance decomposition changed")
        output[choose] = candidate[choose]
        selected_scale[choose] = float(scale)
        selected[choose] = True
    acceptance = float(np.mean(selected))
    candidate_sha = stager.sha256_array(output)
    scale_sha = stager.sha256_array(selected_scale)
    attempted_order = base_cell.get("internal_scale_attempt_order")
    if not isinstance(attempted_order, Sequence) or isinstance(
        attempted_order, (str, bytes, bytearray)
    ):
        raise StageTError("base internal-scale order is not a sequence")
    expected_acceptance = _finite_float(
        base_cell.get("aligned_acceptance_rate"), "base aligned acceptance"
    )
    expected_mismatch = _required_int_value(
        base_cell.get("length_log_z_element_mismatch_count"),
        "base length/log-z mismatch count",
    )
    expected_aligned_failures = _required_int_value(
        base_cell.get("aligned_upper_element_failure_count"),
        "base aligned upper failure count",
    )
    expected_regression = _required_int_value(
        base_cell.get("strict_pass_aligned_fail_row_count"),
        "base strict-pass/aligned-fail count",
    )
    checks = {
        "aligned_candidate_sha256": candidate_sha == base_cell.get("aligned_candidate_sha256"),
        "aligned_selected_scale_sha256": scale_sha == base_cell.get("aligned_selected_scale_sha256"),
        "aligned_acceptance_rate": acceptance == expected_acceptance,
        "internal_scale_order": list(expected_order) == list(attempted_order),
        "length_log_z_mismatch": mismatch_count == expected_mismatch,
        "aligned_upper_failures": aligned_upper_failures == expected_aligned_failures,
        "strict_pass_aligned_fail": strict_pass_aligned_fail == expected_regression,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageTError(f"Stage-T candidate reconstruction differs from Stage-R: {failed}")
    fidelity = dict(
        compute_candidate_fidelity_metrics(
            control=control_raw,
            candidate=output,
            target=target_raw,
            selected_scale=selected_scale,
            policy=policy,
        )
    )
    gate_clean = (
        mismatch_count == 0
        and aligned_upper_failures == 0
        and strict_pass_aligned_fail == 0
    )
    fidelity["mechanism_eligible"] = bool(fidelity["mechanism_eligible"] and gate_clean)
    fidelity["fidelity_eligible"] = bool(fidelity["fidelity_eligible"] and gate_clean)
    return {
        "reconstruction_exact": True,
        "reconstruction_checks": checks,
        "matrix_reconstruction_attempt_count": attempt_count,
        "aligned_candidate_sha256": candidate_sha,
        "aligned_selected_scale_sha256": scale_sha,
        "aligned_selected_scale_histogram": stager.selected_scale_histogram(selected_scale),
        "aligned_acceptance_rate": acceptance,
        "length_log_z_element_mismatch_count": mismatch_count,
        "aligned_upper_element_failure_count": aligned_upper_failures,
        "strict_pass_aligned_fail_row_count": strict_pass_aligned_fail,
        "candidate_fidelity": fidelity,
        "gate_contract_observation": _gate_contract_observation(
            contract=contract,
            upper_bound=upper,
            segment_tolerance=float(integrator_spec.segment_tolerance),
            tolerance_factor=float(spec.reconstruction_tolerance_factor),
            sha256_array=stager.sha256_array,
        ),
        "candidate_tensor_persisted": False,
        "selected_scale_tensor_persisted": False,
        "target_tensor_persisted": False,
    }


@contextmanager
def patched_candidate_matrix_audit(stager: Any):
    original = stager.audit_oof_shadow_cell
    observations: List[Mapping[str, Any]] = []

    def wrapper(**kwargs: Any) -> Mapping[str, Any]:
        base = dict(original(**kwargs))
        runtime = dict(kwargs["runtime"])
        runtime["stager"] = stager
        matrix = reconstruct_aligned_candidate_metrics(
            base_cell=base,
            control=kwargs["control"],
            base_direction=kwargs["base_direction"],
            target=np.asarray(kwargs["context"]["objective_target"], dtype=np.float32),
            context=kwargs["context"],
            runtime=runtime,
            spec=kwargs["spec"],
        )
        base["candidate_matrix_metrics"] = matrix
        observations.append(matrix["gate_contract_observation"])
        return base

    stager.audit_oof_shadow_cell = wrapper
    try:
        yield observations
    finally:
        stager.audit_oof_shadow_cell = original


def _cell_key(cell: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(cell.get("base_direction_id")), int(cell.get("timestep")))


def _rank_cell_key(cell: Mapping[str, Any]) -> Tuple[Any, ...]:
    fidelity = _mapping(cell.get("candidate_fidelity"), "candidate fidelity")
    accepted = _mapping(fidelity.get("accepted_rows"), "accepted rows")
    relative = _mapping(accepted.get("relative_distance_reduction"), "relative reduction")
    return (
        0 if fidelity.get("fidelity_eligible") is True else 1,
        float(fidelity.get("overall_mse_ratio")),
        -float(accepted.get("positive_distance_reduction_rate")),
        -float(relative.get("mean")),
        -float(fidelity.get("acceptance_rate")),
        str(cell.get("base_direction_id")),
        int(cell.get("timestep")),
    )


def pareto_frontier(backbones: Sequence[Mapping[str, Any]]) -> List[str]:
    eligible = [item for item in backbones if item.get("matrix_eligible") is True]
    frontier: List[str] = []
    for item in eligible:
        dominated = False
        for other in eligible:
            if other is item:
                continue
            no_worse = (
                float(other["worst_overall_mse_ratio"]) <= float(item["worst_overall_mse_ratio"])
                and float(other["minimum_acceptance_rate"]) >= float(item["minimum_acceptance_rate"])
                and float(other["minimum_positive_distance_reduction_rate"]) >= float(item["minimum_positive_distance_reduction_rate"])
            )
            strictly_better = (
                float(other["worst_overall_mse_ratio"]) < float(item["worst_overall_mse_ratio"])
                or float(other["minimum_acceptance_rate"]) > float(item["minimum_acceptance_rate"])
                or float(other["minimum_positive_distance_reduction_rate"]) > float(item["minimum_positive_distance_reduction_rate"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(item["base_direction_id"]))
    return sorted(frontier)


def build_candidate_matrix(cells: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(cells) != EXPECTED_CELL_COUNT:
        raise StageTError("Stage-T candidate cell count changed")
    normalized: List[Mapping[str, Any]] = []
    for index, raw in enumerate(cells):
        cell = _mapping(raw, f"candidate cell {index}")
        matrix = _mapping(cell.get("candidate_matrix_metrics"), "candidate matrix metrics")
        fidelity = copy.deepcopy(dict(_mapping(matrix.get("candidate_fidelity"), "candidate fidelity")))
        record = {
            "base_direction_id": str(cell.get("base_direction_id")),
            "timestep": int(cell.get("timestep")),
            "feature_mode": str(cell.get("feature_mode")),
            "aligned_acceptance_rate": float(cell.get("aligned_acceptance_rate")),
            "aligned_candidate_sha256": str(cell.get("aligned_candidate_sha256")),
            "aligned_selected_scale_sha256": str(cell.get("aligned_selected_scale_sha256")),
            "aligned_selected_scale_histogram": copy.deepcopy(cell.get("aligned_selected_scale_histogram")),
            "length_log_z_element_mismatch_count": int(cell.get("length_log_z_element_mismatch_count")),
            "aligned_upper_element_failure_count": int(cell.get("aligned_upper_element_failure_count")),
            "strict_pass_aligned_fail_row_count": int(cell.get("strict_pass_aligned_fail_row_count")),
            "candidate_fidelity": fidelity,
            "matrix_reconstruction_exact": matrix.get("reconstruction_exact") is True,
            "matrix_reconstruction_attempt_count": int(matrix.get("matrix_reconstruction_attempt_count")),
            "legacy_functional_identity_exact": _mapping(cell.get("legacy_identity"), "legacy identity").get("all_functional_exact") is True,
        }
        if not record["matrix_reconstruction_exact"] or not record["legacy_functional_identity_exact"]:
            raise StageTError("candidate functional identity is not exact")
        normalized.append(record)
    normalized.sort(key=_cell_key)
    if len({item["base_direction_id"] for item in normalized}) != EXPECTED_BACKBONE_COUNT:
        raise StageTError("Stage-T backbone population changed")
    if {item["timestep"] for item in normalized} != set(EXPECTED_TIMESTEPS):
        raise StageTError("Stage-T timestep population changed")
    ranked_cells = sorted(normalized, key=_rank_cell_key)
    cell_rank = {
        (item["base_direction_id"], item["timestep"]): rank + 1
        for rank, item in enumerate(ranked_cells)
    }
    for item in normalized:
        item["global_rank"] = cell_rank[(item["base_direction_id"], item["timestep"])]
    per_timestep: Dict[str, List[Mapping[str, Any]]] = {}
    for timestep in EXPECTED_TIMESTEPS:
        values = [item for item in normalized if item["timestep"] == timestep]
        values.sort(key=_rank_cell_key)
        per_timestep[str(timestep)] = [
            {
                "rank": rank + 1,
                "base_direction_id": value["base_direction_id"],
                "fidelity_eligible": value["candidate_fidelity"]["fidelity_eligible"],
                "overall_mse_ratio": value["candidate_fidelity"]["overall_mse_ratio"],
                "acceptance_rate": value["candidate_fidelity"]["acceptance_rate"],
                "positive_distance_reduction_rate": value["candidate_fidelity"]["accepted_rows"]["positive_distance_reduction_rate"],
            }
            for rank, value in enumerate(values)
        ]
    backbone_records: List[Mapping[str, Any]] = []
    for backbone in sorted({item["base_direction_id"] for item in normalized}):
        values = [item for item in normalized if item["base_direction_id"] == backbone]
        if len(values) != len(EXPECTED_TIMESTEPS):
            raise StageTError("backbone timestep population changed")
        fidelities = [item["candidate_fidelity"] for item in values]
        acceptance = [float(value["acceptance_rate"]) for value in fidelities]
        mse_ratio = [float(value["overall_mse_ratio"]) for value in fidelities]
        positive = [float(value["accepted_rows"]["positive_distance_reduction_rate"]) for value in fidelities]
        relative = [float(value["accepted_rows"]["relative_distance_reduction"]["mean"]) for value in fidelities]
        eligible_count = sum(value.get("fidelity_eligible") is True for value in fidelities)
        backbone_records.append({
            "base_direction_id": backbone,
            "fidelity_eligible_timestep_count": eligible_count,
            "matrix_eligible": eligible_count == len(EXPECTED_TIMESTEPS),
            "minimum_acceptance_rate": min(acceptance),
            "mean_acceptance_rate": float(np.mean(acceptance)),
            "worst_overall_mse_ratio": max(mse_ratio),
            "mean_overall_mse_ratio": float(np.mean(mse_ratio)),
            "minimum_positive_distance_reduction_rate": min(positive),
            "mean_positive_distance_reduction_rate": float(np.mean(positive)),
            "minimum_relative_distance_reduction_mean": min(relative),
            "mean_relative_distance_reduction_mean": float(np.mean(relative)),
            "timesteps": [item["timestep"] for item in sorted(values, key=lambda value: value["timestep"])],
        })
    backbone_records.sort(key=lambda item: (
        0 if item["matrix_eligible"] else 1,
        -int(item["fidelity_eligible_timestep_count"]),
        float(item["worst_overall_mse_ratio"]),
        -float(item["minimum_positive_distance_reduction_rate"]),
        -float(item["minimum_acceptance_rate"]),
        str(item["base_direction_id"]),
    ))
    for rank, item in enumerate(backbone_records, 1):
        item["rank"] = rank
    frontier = pareto_frontier(backbone_records)
    mechanism_count = sum(item["candidate_fidelity"]["mechanism_eligible"] is True for item in normalized)
    fidelity_count = sum(item["candidate_fidelity"]["fidelity_eligible"] is True for item in normalized)
    matrix_eligible = [item["base_direction_id"] for item in backbone_records if item["matrix_eligible"]]
    return {
        "cell_count": len(normalized),
        "backbone_count": len(backbone_records),
        "timestep_count": len(EXPECTED_TIMESTEPS),
        "mechanism_eligible_cell_count": mechanism_count,
        "fidelity_eligible_cell_count": fidelity_count,
        "matrix_eligible_backbone_count": len(matrix_eligible),
        "matrix_eligible_backbones": matrix_eligible,
        "pareto_frontier_backbones": frontier,
        "cell_records": normalized,
        "global_cell_ranking": [
            {
                "rank": rank + 1,
                "base_direction_id": item["base_direction_id"],
                "timestep": item["timestep"],
                "fidelity_eligible": item["candidate_fidelity"]["fidelity_eligible"],
                "overall_mse_ratio": item["candidate_fidelity"]["overall_mse_ratio"],
                "acceptance_rate": item["candidate_fidelity"]["acceptance_rate"],
            }
            for rank, item in enumerate(ranked_cells)
        ],
        "per_timestep_ranking": per_timestep,
        "backbone_ranking": backbone_records,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }


def classify_candidate_matrix(matrix: Mapping[str, Any]) -> Mapping[str, str]:
    mechanism = int(matrix.get("mechanism_eligible_cell_count", 0))
    fidelity = int(matrix.get("fidelity_eligible_cell_count", 0))
    matrix_backbones = int(matrix.get("matrix_eligible_backbone_count", 0))
    if mechanism != EXPECTED_CELL_COUNT:
        return {
            "root_cause": "phase314b_r258_staget_confirmed_oof_mechanism_support_regressed",
            "required_next_path": "AUDIT_TOLERANCE_ALIGNED_CANDIDATE_MATRIX_MECHANISM_REGRESSION",
            "primary_failure_locus": "candidate_mechanism_regression",
        }
    if fidelity == 0:
        return {
            "root_cause": "phase314b_r258_staget_oof_support_lacks_objective_train_fidelity",
            "required_next_path": "AUDIT_EXECUTABLE_DIRECTION_MAGNITUDE_AND_TARGET_FIDELITY_ON_OBJECTIVE_TRAIN_ONLY",
            "primary_failure_locus": "candidate_fidelity",
        }
    if matrix_backbones == 0:
        return {
            "root_cause": "phase314b_r258_staget_fidelity_support_is_backbone_or_timestep_stratified",
            "required_next_path": "STRATIFY_FIDELITY_ELIGIBLE_CANDIDATES_BY_BACKBONE_AND_TIMESTEP",
            "primary_failure_locus": "candidate_matrix_stratification",
        }
    return {
        "root_cause": "phase314b_r258_staget_tolerance_aligned_gate_and_candidate_matrix_confirmed",
        "required_next_path": "LOCK_TOLERANCE_ALIGNED_OOF_CANDIDATE_FRONTIER_ON_OBJECTIVE_TRAIN_ONLY",
        "primary_failure_locus": "confirmed_candidate_frontier",
    }


def _load_science_modules() -> Mapping[str, Any]:
    from ccda_phase3 import phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo
    from ccda_phase3 import phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager
    from ccda_phase3 import phase314b_r258_stager_resume1_portable_oof_functional_replay as stager_resume1
    from ccda_phase3 import phase314b_r258_stagel_predicate_assembly_audit as stagel
    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages
    from ccda_phase3 import phase314b_r258_stages_resume1_oracle_rerun_schema_recovery as stages_resume1
    from ccda_phase3 import phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a
    from ccda_phase3 import phase314b_r258_stages_resume3_cold_worker_oof_confirmation as stages_resume3
    return {
        "stageo": stageo,
        "stager": stager,
        "stager_resume1": stager_resume1,
        "stagel": stagel,
        "stages": stages,
        "stages_resume1": stages_resume1,
        "resume2a": resume2a,
        "stages_resume3": stages_resume3,
    }


def worker_payload(
    *,
    worker_id: str,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
) -> Mapping[str, Any]:
    validate_environment_variables()
    modules = _load_science_modules()
    stageo = modules["stageo"]
    stager = modules["stager"]
    stager_resume1 = modules["stager_resume1"]
    stagel = modules["stagel"]
    stages = modules["stages"]
    stages_resume1 = modules["stages_resume1"]
    resume2a = modules["resume2a"]
    stages_resume3 = modules["stages_resume3"]
    environment = stages_resume3.validate_probe_payload_for_science(
        probe_payload,
        resume2a=resume2a,
    )
    runtime_modules = stageo._runtime_modules()
    stagea = runtime_modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment)
    cold = stagea.assert_cold_cuda_context_portable()
    repo = Path(root).resolve()
    repository = {
        "head": str(repository_head),
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
    }
    with stager_resume1.patched_portable_replay(stager, stagel) as registry:
        with patched_candidate_matrix_audit(stager) as gate_observations:
            stage_r_result = stager.run_oof_post_upper_audit(
                root=repo,
                environment=environment,
                repository=repository,
            )
    corrected = stager_resume1._correct_stage_r_result(stage_r_result, registry)
    wrapper: Dict[str, Any] = {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "stage_r_result": corrected,
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }
    functional = stages_resume1.corrected_functional_projection(wrapper, stages=stages)
    functional_sha = stages.sha256_bytes(stages.stable_json_bytes(functional))
    if functional_sha != EXPECTED_BASE_FUNCTIONAL_SHA256:
        raise StageTError("Stage-T functional projection differs from Stage-S Resume3")
    current_fit = stages.current_fit_projection(wrapper)
    current_fit_sha = stages.sha256_bytes(stages.stable_json_bytes(current_fit))
    audit = _mapping(corrected.get("tolerance_aligned_oof_post_upper_audit"), "Stage-R audit")
    cells = _sequence(audit.get("cell_records"), "Stage-R cells")
    matrix = build_candidate_matrix(cells)
    if len(gate_observations) != EXPECTED_CELL_COUNT:
        raise StageTError("gate observation population changed")
    first_gate = gate_observations[0]
    if any(stable_json_bytes(value) != stable_json_bytes(first_gate) for value in gate_observations[1:]):
        raise StageTError("aligned gate contract differs across candidate cells")
    gate_contract = copy.deepcopy(dict(first_gate))
    environment_sha = sha256_bytes(stable_json_bytes(environment))
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "worker_id": str(worker_id),
        "process_id": os.getpid(),
        "execution_verdict": "PASS",
        "repository_head": str(repository_head),
        "environment_sha256": environment_sha,
        "cold_cuda_precheck": copy.deepcopy(dict(cold)),
        "environment_probe_rerun_in_science_process": False,
        "functional_projection_sha256": functional_sha,
        "current_fit_projection_sha256": current_fit_sha,
        "functional_projection_matches_stage_s_resume3": True,
        "candidate_matrix": matrix,
        "candidate_matrix_sha256": sha256_bytes(stable_json_bytes(matrix)),
        "gate_contract": gate_contract,
        "gate_contract_sha256": str(gate_contract["contract_sha256"]),
        "scientific_oof_fit_count": EXPECTED_SCIENCE_FITS_PER_WORKER,
        "repeat_identity_fit_count": EXPECTED_REPEAT_FITS_PER_WORKER,
        "total_oof_fit_count": EXPECTED_TOTAL_FITS_PER_WORKER,
        "callback_pair_count": EXPECTED_CALLBACK_PAIRS_PER_WORKER,
        "shadow_internal_scale_attempt_count": EXPECTED_SHADOW_ATTEMPTS_PER_WORKER,
        "matrix_reconstruction_attempt_count": sum(
            int(_mapping(cell, "cell").get("candidate_matrix_metrics", {}).get("matrix_reconstruction_attempt_count", 0))
            for cell in cells
        ),
        "oracle_callback_rerun_count": 0,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    validate_worker_payload(
        result,
        expected_environment_sha256=environment_sha,
    )
    return result


def validate_worker_payload(
    payload: Mapping[str, Any],
    *,
    expected_environment_sha256: str,
) -> Mapping[str, Any]:
    worker = _mapping(payload, "Stage-T worker")
    if worker.get("schema") != WORKER_SCHEMA or worker.get("execution_verdict") != "PASS":
        raise StageTError("Stage-T worker schema/verdict changed")
    if worker.get("environment_sha256") != expected_environment_sha256:
        raise StageTError("Stage-T worker environment SHA changed")
    cold = _mapping(worker.get("cold_cuda_precheck"), "cold CUDA precheck")
    if cold.get("torch_cuda_is_initialized") is not False:
        raise StageTError("Stage-T worker was not cold")
    if worker.get("environment_probe_rerun_in_science_process") is not False:
        raise StageTError("Stage-T worker reran environment probe")
    required = {
        "functional_projection_sha256": EXPECTED_BASE_FUNCTIONAL_SHA256,
        "current_fit_projection_sha256": EXPECTED_BASE_CURRENT_FIT_SHA256,
        "functional_projection_matches_stage_s_resume3": True,
        "scientific_oof_fit_count": EXPECTED_SCIENCE_FITS_PER_WORKER,
        "repeat_identity_fit_count": EXPECTED_REPEAT_FITS_PER_WORKER,
        "total_oof_fit_count": EXPECTED_TOTAL_FITS_PER_WORKER,
        "callback_pair_count": EXPECTED_CALLBACK_PAIRS_PER_WORKER,
        "shadow_internal_scale_attempt_count": EXPECTED_SHADOW_ATTEMPTS_PER_WORKER,
        "matrix_reconstruction_attempt_count": EXPECTED_MATRIX_RECONSTRUCTION_ATTEMPTS_PER_WORKER,
        "oracle_callback_rerun_count": 0,
    }
    for key, expected in required.items():
        if worker.get(key) != expected:
            raise StageTError(f"Stage-T worker field changed: {key}")
    matrix = _mapping(worker.get("candidate_matrix"), "candidate matrix")
    if matrix.get("cell_count") != EXPECTED_CELL_COUNT:
        raise StageTError("Stage-T matrix cell count changed")
    if worker.get("candidate_matrix_sha256") != sha256_bytes(stable_json_bytes(matrix)):
        raise StageTError("Stage-T candidate matrix SHA is invalid")
    gate = _mapping(worker.get("gate_contract"), "gate contract")
    if gate.get("contract_sha256") != worker.get("gate_contract_sha256"):
        raise StageTError("Stage-T gate contract SHA changed")
    if gate.get("contract_role") != "shadow_frozen_not_written_to_stagee":
        raise StageTError("Stage-T gate deployment role changed")
    require_false(worker, FALSE_BOUNDARIES, "Stage-T worker")
    return worker


def compare_workers(first: Mapping[str, Any], second: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = {
        "environment_sha256": first.get("environment_sha256") == second.get("environment_sha256"),
        "current_fit_projection_sha256": first.get("current_fit_projection_sha256") == second.get("current_fit_projection_sha256"),
        "functional_projection_sha256": first.get("functional_projection_sha256") == second.get("functional_projection_sha256"),
        "candidate_matrix_sha256": first.get("candidate_matrix_sha256") == second.get("candidate_matrix_sha256"),
        "gate_contract_sha256": first.get("gate_contract_sha256") == second.get("gate_contract_sha256"),
        "candidate_matrix": stable_json_bytes(first.get("candidate_matrix")) == stable_json_bytes(second.get("candidate_matrix")),
        "gate_contract": stable_json_bytes(first.get("gate_contract")) == stable_json_bytes(second.get("gate_contract")),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageTError(f"Stage-T workers differ: {failed}")
    return {"all_exact": True, "checks": checks}


def _run_child(command: Sequence[str], *, root: Path, label: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(command),
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
        check=False,
    )
    if completed.returncode != 0:
        raise StageTError(
            f"{label} rc={completed.returncode} stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return completed


def run_confirmation(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    validate_environment_variables()
    repo = Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix="phase314b_r258_staget_") as temporary:
        directory = Path(temporary)
        probe_path = directory / "environment_probe.json"
        probe_command = [
            str(python_bin),
            str(repo / "scripts/phase3_14b_r258_stages_resume2a_worker.py"),
            "--root", str(repo),
            "--mode", "environment-probe",
            "--output", str(probe_path),
        ]
        _run_child(probe_command, root=repo, label="environment probe")
        probe = load_json(probe_path)
        from ccda_phase3 import phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a
        environment = resume2a.validate_probe_payload(probe)
        environment_sha = sha256_bytes(stable_json_bytes(environment))
        workers: List[Mapping[str, Any]] = []
        process_ids = [int(_mapping(probe, "probe").get("process_id"))]
        for index in range(EXPECTED_WORKER_COUNT):
            output = directory / f"worker_{index}.json"
            command = [
                str(python_bin),
                str(repo / "scripts/phase3_14b_r258_staget_worker.py"),
                "--root", str(repo),
                "--worker-id", f"worker-{index + 1}",
                "--probe", str(probe_path),
                "--repository-head", str(repository.get("head")),
                "--output", str(output),
            ]
            _run_child(command, root=repo, label=f"cold science worker {index + 1}")
            worker = validate_worker_payload(
                load_json(output),
                expected_environment_sha256=environment_sha,
            )
            workers.append(worker)
            process_ids.append(int(worker.get("process_id")))
        if len(set(process_ids)) != 3:
            raise StageTError("Stage-T probe/science processes are not distinct")
        comparison = compare_workers(workers[0], workers[1])
    matrix = copy.deepcopy(dict(_mapping(workers[0].get("candidate_matrix"), "candidate matrix")))
    gate = copy.deepcopy(dict(_mapping(workers[0].get("gate_contract"), "gate contract")))
    classification = classify_candidate_matrix(matrix)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": {key: value for key, value in repository.items() if key not in ("base_report",)},
        "immutable_inputs": {
            "base_stage_s_resume3_evidence_commit": BASE_STAGES_RESUME3_EVIDENCE_COMMIT,
            "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "base_scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_SHA256,
            "base_current_fit_projection_sha256": EXPECTED_BASE_CURRENT_FIT_SHA256,
            "base_functional_projection_sha256": EXPECTED_BASE_FUNCTIONAL_SHA256,
            "objective_train_only": True,
            "external_multiplier": EXPECTED_EXTERNAL_MULTIPLIER,
            "tolerance_factor": EXPECTED_TOLERANCE_FACTOR,
            "candidate_policy": asdict(CandidatePolicy()),
        },
        "confirmation_execution": {
            "environment_probe_count": 1,
            "cold_science_worker_count": EXPECTED_WORKER_COUNT,
            "processes_distinct": True,
            "workers_sequential": True,
            "process_ids": {
                "environment_probe": process_ids[0],
                "worker_1": process_ids[1],
                "worker_2": process_ids[2],
            },
            "worker_comparison": comparison,
            "current_fit_projection_sha256": workers[0]["current_fit_projection_sha256"],
            "functional_projection_sha256": workers[0]["functional_projection_sha256"],
            "candidate_matrix_sha256": workers[0]["candidate_matrix_sha256"],
            "gate_contract_sha256": workers[0]["gate_contract_sha256"],
            "total_oof_fit_count": EXPECTED_TOTAL_FITS,
            "total_callback_pair_count": EXPECTED_TOTAL_CALLBACK_PAIRS,
            "total_shadow_internal_scale_attempt_count": EXPECTED_TOTAL_SHADOW_ATTEMPTS,
            "total_matrix_reconstruction_attempt_count": EXPECTED_TOTAL_MATRIX_RECONSTRUCTION_ATTEMPTS,
            "oracle_callback_rerun_count": 0,
            "worker_outputs_persisted": False,
        },
        "candidate_matrix": matrix,
        "candidate_matrix_frontier": copy.deepcopy(matrix.get("pareto_frontier_backbones")),
        "gate_contract_sha256": gate["contract_sha256"],
        "mechanism_boundary": {
            "stagee_modified": False,
            "legacy_upper_gate_changed": False,
            "aligned_upper_gate_changed_from_stage_q": False,
            "aligned_upper_gate_written_to_stagee": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "oof_backbone_population_changed": False,
            "holdout_closed": True,
            "frozen_probe_closed": True,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    gate_report = {
        **gate,
        "phase": PHASE,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_functional_projection_sha256": EXPECTED_BASE_FUNCTIONAL_SHA256,
        "confirmed_by_worker_count": EXPECTED_WORKER_COUNT,
        "worker_contract_byte_exact": True,
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    gate_report["contract_payload_sha256"] = sha256_bytes(stable_json_bytes(gate_report))
    return result, gate_report


def blocked_report(*, repository: Optional[Mapping[str, Any]], error: BaseException) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_staget_candidate_matrix_execution_failed",
        "required_next_path": "RESTORE_STAGET_ALIGNED_GATE_CANDIDATE_MATRIX",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else {
            key: value for key, value in repository.items() if key != "base_report"
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_evidence_preserved": True,
        "full_worker_population_completed": False,
        "full_fit_count_claimed": False,
        "full_callback_count_claimed": False,
        "full_attempt_count_claimed": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
