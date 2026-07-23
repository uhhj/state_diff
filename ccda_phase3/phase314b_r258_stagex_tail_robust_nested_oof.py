"""Phase3.14b-r2.5.8 Stage X objective-train tail-robust nested OOF.

Stage W localized the one-shot selection-holdout failure to accepted-row
fidelity: a majority of accepted rows improved, but a minority adverse side
carried enough squared-error mass to dominate.  Stage X therefore remains
strictly objective-train-only and tests one pre-registered repair family:

* keep the Stage-U backbone, timesteps, external multiplier, internal scale
  search, and tolerance-aligned gate unchanged;
* use six outer group folds for unbiased objective-train evaluation;
* inside each outer training population, use the remaining five group folds to
  select one joint policy across t=10/25/50;
* the policy may only shrink the predicted direction by a predeclared factor
  and abstain to the control when a target-independent adverse-risk model
  exceeds a predeclared threshold;
* labels use objective-train targets only, and outer-fold fidelity targets are excluded from fitting and policy selection;
  their error metrics are computed only after direction, candidate, descriptors,
  risk probability and policy are fixed.

No selection-holdout target, frozen-probe row, formal model, reverse sampler,
IDM, environment candidate execution, checkpoint, tensor, NPZ, cache, image,
or video is written.  Only bounded scalar JSON evidence is persisted.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.8 Stage X"
SCHEMA = "phase314b_r258_stagex_tail_robust_nested_group_oof_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEW_IMPLEMENTATION_COMMIT = "7093a1db40dcd07eea127d985c8d6f6c7f174a4b"
BASE_STAGEW_EVIDENCE_COMMIT = "e0d0756f80415ea36335be6019c86422f043ff83"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_STAGEW_REPORT_SHA256 = "7c52393861498e561f1ff7661c8277ed07e8d1f75e2973e3b4b2364381680a17"
EXPECTED_STAGEU_CONTRACT_SHA256 = "c0bf2c477694cf1bda77f11c7ac407befb77d005afa8422e7e9bae2616714a8c"
EXPECTED_STAGEV_RESUME1_SHA256 = "1b27e2bd20f1022cf2baa8017b995c27e3c9fc486ea72761c7305415990f181b"
EXPECTED_STAGEV_RESULT_SHA256 = "e760df33caa81f4794d739a76e60ccf9c12326547ae14a98f0c4f23e2f3c5946"

STAGEW_REPORT = "reports/phase3_14b_r258_stagew_selection_holdout_transfer_failure_audit_summary.json"
STAGEU_CONTRACT = "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"
STAGEV_REPORT = "reports/phase3_14b_r258_stagev_resume1_locked_selection_holdout_evaluation_summary.json"
SUCCESS_REPORT = "reports/phase3_14b_r258_stagex_tail_robust_nested_group_oof_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stagex_tail_robust_nested_group_oof_blocked_summary.json"

IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage X: calibrate tail-robust nested group OOF"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage X tail-robust OOF evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage X blocked evidence"
STAGEW_IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage W: audit selection-holdout transfer failure"
STAGEW_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage W transfer-failure audit evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagex_tail_robust_nested_oof.py"),
    ("A", "scripts/phase3_14b_r258_stagex_worker.py"),
    ("A", "scripts/phase3_14b_r258_stagex_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagex_tail_robust_nested_oof.py"),
)
STAGEW_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagew_selection_holdout_transfer_failure_audit.py"),
    ("A", "scripts/phase3_14b_r258_stagew_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagew_selection_holdout_transfer_failure_audit.py"),
)
STAGEW_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (("A", STAGEW_REPORT),)

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagev_locked_selection_holdout_evaluation.py": "5aedabe9004630207021f1252f5b99aa6524b8dc9b3077003b20c8b163e7b1b0",
    "ccda_phase3/phase314b_r258_stagew_selection_holdout_transfer_failure_audit.py": "1b80c918ac679570cfba48a033013855d656f134b8fd87533e258a9df480bde4",
    "ccda_phase3/phase314b_r258_stageu_candidate_frontier_lock.py": "a4455727e20ab96658686dc42eb0ab35b7bddadef8e9e99648d2c57371bae657",
}

LOCKED_BACKBONE = "segment_target_rr64_feasible"
LOCKED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_ROWS = 638
EXPECTED_GROUPS = 126
EXPECTED_OUTER_FOLDS = 6
EXPECTED_INNER_FOLDS = 5
EXPECTED_INTERNAL_SCALE_COUNT = 7
DIRECTION_SHRINKAGES: Tuple[float, ...] = (0.50, 0.75, 1.00)
RISK_THRESHOLDS: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

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
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded_for_stagex",
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
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "descriptor_tensor_persisted",
    "risk_probability_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageXError(RuntimeError):
    """Fail-closed Stage-X error."""


@dataclass(frozen=True)
class TailPolicy:
    shrinkage: float
    risk_threshold: float

    @property
    def policy_id(self) -> str:
        return "shrink_{:.2f}__risk_{:.2f}".format(self.shrinkage, self.risk_threshold)

    def validate(self) -> None:
        if float(self.shrinkage) not in DIRECTION_SHRINKAGES:
            raise StageXError("direction shrinkage left predeclared bank")
        if float(self.risk_threshold) not in RISK_THRESHOLDS:
            raise StageXError("risk threshold left predeclared bank")


@dataclass(frozen=True)
class StageXSpec:
    outer_folds: int = 6
    minimum_acceptance_rate: float = 0.50
    minimum_positive_reduction_rate: float = 0.50
    adverse_sse_reduction_fraction: float = 0.10
    group_cvar_fraction: float = 0.20
    modal_outer_fold_minimum: int = 4
    risk_l2: float = 1.0
    risk_max_iterations: int = 80
    risk_tolerance: float = 1.0e-10
    epsilon: float = 1.0e-12

    def validate(self) -> None:
        if int(self.outer_folds) != EXPECTED_OUTER_FOLDS:
            raise StageXError("outer fold count changed")
        if self.minimum_acceptance_rate != 0.50:
            raise StageXError("minimum acceptance changed")
        if self.minimum_positive_reduction_rate != 0.50:
            raise StageXError("positive-reduction threshold changed")
        if self.adverse_sse_reduction_fraction != 0.10:
            raise StageXError("adverse-SSE reduction target changed")
        if self.group_cvar_fraction != 0.20:
            raise StageXError("group CVaR fraction changed")
        if self.modal_outer_fold_minimum != 4:
            raise StageXError("policy stability minimum changed")
        if self.risk_l2 <= 0 or self.risk_max_iterations < 1 or self.risk_tolerance <= 0:
            raise StageXError("risk model contract is invalid")


def policy_population() -> Tuple[TailPolicy, ...]:
    result = tuple(
        TailPolicy(shrinkage=shrinkage, risk_threshold=threshold)
        for shrinkage in DIRECTION_SHRINKAGES
        for threshold in RISK_THRESHOLDS
    )
    for policy in result:
        policy.validate()
    if len(result) != 12 or len({p.policy_id for p in result}) != 12:
        raise StageXError("policy population changed")
    return result


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = stable_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return sha256_bytes(header + b"\n" + array.tobytes(order="C"))


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise StageXError("git {} failed: {}".format(" ".join(args), completed.stderr.strip()))
    return completed.stdout.strip()


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    result = []
    for line in output.splitlines():
        if line.strip():
            fields = line.split("\t")
            result.append((fields[0], fields[-1]))
    return tuple(sorted(result))


def validate_environment_variables() -> Mapping[str, str]:
    observed = {key: os.environ.get(key) for key in EXPECTED_ENV}
    if observed != dict(EXPECTED_ENV):
        raise StageXError("deterministic environment changed: {!r}".format(observed))
    return {key: str(value) for key, value in observed.items()}


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageXError("JSON root is not a mapping: {}".format(path))
    return value


def validate_stagew_report(report: Mapping[str, Any]) -> None:
    if report.get("execution_verdict") != "PASS" or report.get("scientific_status") != "BLOCKED":
        raise StageXError("Stage-W verdict changed")
    if report.get("root_cause") != "phase314b_r258_stagew_aggregate_evidence_localizes_failure_to_accepted_row_fidelity_with_minority_adverse_sse_dominance":
        raise StageXError("Stage-W root changed")
    if report.get("required_next_path") != "DESIGN_OBJECTIVE_TRAIN_ONLY_TAIL_ROBUST_DIRECTION_CALIBRATION_WITH_NESTED_GROUP_OOF_WHILE_KEEPING_FROZEN_PROBE_CLOSED":
        raise StageXError("Stage-W next path changed")
    if report.get("selected_configuration") is not None or report.get("train_only_recommendation") is not None:
        raise StageXError("Stage-W retained a configuration")
    if report.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageXError("holdout evaluation count changed")
    if report.get("rerun_authorized") is not False:
        raise StageXError("Stage-W rerun boundary changed")


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageXError("Stage X requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", head + "^") != BASE_STAGEW_EVIDENCE_COMMIT:
        raise StageXError("Stage-X implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageXError("Stage-X implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageXError("Stage-X implementation paths changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEW_EVIDENCE_COMMIT) != STAGEW_EVIDENCE_SUBJECT:
        raise StageXError("Stage-W evidence subject changed")
    if commit_name_status(repo, BASE_STAGEW_EVIDENCE_COMMIT) != STAGEW_EVIDENCE_PATHS:
        raise StageXError("Stage-W evidence paths changed")
    if _git(repo, "rev-parse", BASE_STAGEW_EVIDENCE_COMMIT + "^") != BASE_STAGEW_IMPLEMENTATION_COMMIT:
        raise StageXError("Stage-W evidence parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEW_IMPLEMENTATION_COMMIT) != STAGEW_IMPLEMENTATION_SUBJECT:
        raise StageXError("Stage-W implementation subject changed")
    if commit_name_status(repo, BASE_STAGEW_IMPLEMENTATION_COMMIT) != tuple(sorted(STAGEW_IMPLEMENTATION_PATHS)):
        raise StageXError("Stage-W implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageXError("origin/Experiment1 changed")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageXError("main worktree is dirty")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageXError("DeformableRavens commit changed")
    if _git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageXError("DeformableRavens worktree is dirty")
    required = {
        STAGEW_REPORT: EXPECTED_STAGEW_REPORT_SHA256,
        STAGEU_CONTRACT: EXPECTED_STAGEU_CONTRACT_SHA256,
        STAGEV_REPORT: EXPECTED_STAGEV_RESUME1_SHA256,
    }
    for relative, expected in required.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageXError("immutable evidence changed: {}".format(relative))
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageXError("frozen source changed: {}".format(relative))
    if (repo / SUCCESS_REPORT).exists() or (repo / BLOCKED_REPORT).exists():
        raise StageXError("Stage-X output exists; rerun is forbidden")
    stagew = _load_json(repo / STAGEW_REPORT)
    validate_stagew_report(stagew)
    stagev = _load_json(repo / STAGEV_REPORT)
    if stagev.get("scientific_result_sha256") != EXPECTED_STAGEV_RESULT_SHA256:
        raise StageXError("Stage-V scientific result changed")
    return {
        "root": str(repo), "head": head, "parent": BASE_STAGEW_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE, "submodule_commit": EXPECTED_SUBMODULE,
        "stagew_report_sha256": EXPECTED_STAGEW_REPORT_SHA256,
        "stageu_contract_sha256": EXPECTED_STAGEU_CONTRACT_SHA256,
        "stagev_resume1_report_sha256": EXPECTED_STAGEV_RESUME1_SHA256,
    }


def validate_objective_population(context: Mapping[str, Any]) -> Mapping[str, Any]:
    groups = np.asarray(context["objective_groups"]).astype(str)
    assignment = np.asarray(context["objective_fold_assignment"], dtype=np.int64)
    target = np.asarray(context["objective_target"])
    if target.shape[0] != EXPECTED_ROWS or groups.shape != (EXPECTED_ROWS,) or assignment.shape != (EXPECTED_ROWS,):
        raise StageXError("objective-train row population changed")
    if len(set(groups.tolist())) != EXPECTED_GROUPS:
        raise StageXError("objective-train group population changed")
    if set(assignment.tolist()) != set(range(EXPECTED_OUTER_FOLDS)):
        raise StageXError("objective fold population changed")
    for group in set(groups.tolist()):
        if len(set(assignment[groups == group].tolist())) != 1:
            raise StageXError("one group crosses outer folds")
    return {
        "row_count": EXPECTED_ROWS, "group_count": EXPECTED_GROUPS,
        "outer_fold_count": EXPECTED_OUTER_FOLDS,
        "group_sha256": sha256_array(groups), "fold_assignment_sha256": sha256_array(assignment),
    }


def compact_risk_descriptors(control: np.ndarray, direction: np.ndarray, candidate: np.ndarray, selected_scale: np.ndarray) -> np.ndarray:
    control_value = np.asarray(control, dtype=np.float64)
    direction_value = np.asarray(direction, dtype=np.float64)
    candidate_value = np.asarray(candidate, dtype=np.float64)
    scale = np.asarray(selected_scale, dtype=np.float64)
    if control_value.shape != direction_value.shape or control_value.shape != candidate_value.shape:
        raise StageXError("descriptor tensor shapes changed")
    if scale.shape != (control_value.shape[0],):
        raise StageXError("descriptor scale shape changed")
    rows = control_value.shape[0]
    c = control_value.reshape(rows, -1)
    d = direction_value.reshape(rows, -1)
    m = (candidate_value - control_value).reshape(rows, -1)
    horizons = control_value.shape[1] if control_value.ndim >= 3 else 1
    d_h = direction_value.reshape(rows, horizons, -1)
    m_h = (candidate_value - control_value).reshape(rows, horizons, -1)
    d_norm = np.linalg.norm(d, axis=1)
    m_norm = np.linalg.norm(m, axis=1)
    cosine = np.sum(d * m, axis=1) / np.maximum(d_norm * m_norm, 1.0e-12)
    columns = [
        scale, (scale > 0).astype(np.float64), d_norm, m_norm,
        np.linalg.norm(c, axis=1), np.max(np.abs(d), axis=1), np.max(np.abs(m), axis=1),
        cosine, m_norm / np.maximum(d_norm, 1.0e-12),
    ]
    columns.extend(np.linalg.norm(d_h, axis=2)[:, index] for index in range(horizons))
    columns.extend(np.linalg.norm(m_h, axis=2)[:, index] for index in range(horizons))
    result = np.stack(columns, axis=1).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise StageXError("risk descriptors are non-finite")
    return result


def _standardize_fit(x: np.ndarray, epsilon: float) -> Mapping[str, np.ndarray]:
    value = np.asarray(x, dtype=np.float64)
    mean = np.mean(value, axis=0)
    raw = np.std(value, axis=0)
    scale = np.where(raw >= epsilon, raw, 1.0)
    return {"mean": mean, "scale": scale}


def _standardize_apply(x: np.ndarray, standardizer: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (np.asarray(x, dtype=np.float64) - standardizer["mean"]) / standardizer["scale"]
    if not np.all(np.isfinite(result)):
        raise StageXError("standardized descriptors are non-finite")
    return result


def fit_risk_model(features: np.ndarray, adverse: np.ndarray, fit_mask: np.ndarray, spec: StageXSpec) -> Mapping[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(adverse, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=np.bool_)
    if x.shape[0] != y.shape[0] or mask.shape != y.shape:
        raise StageXError("risk fit population changed")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return {
            "mode": "constant", "prevalence": 1.0, "fit_row_count": 0,
            "positive_count": 0, "converged": True, "iteration_count": 0,
            "identity": {"fit_mask_sha256": sha256_array(mask), "fit_row_count": 0, "prevalence": 1.0},
        }
    prevalence = float(np.mean(y[indices]))
    prevalence = min(max(prevalence, 1.0e-6), 1.0 - 1.0e-6)
    if indices.size < 8 or len(set(y[indices].tolist())) < 2:
        return {
            "mode": "constant", "prevalence": prevalence, "fit_row_count": int(indices.size),
            "positive_count": int(np.sum(y[indices] > 0.5)), "converged": True, "iteration_count": 0,
            "identity": {"fit_mask_sha256": sha256_array(mask), "fit_row_count": int(indices.size), "prevalence": prevalence},
        }
    standardizer = _standardize_fit(x[indices], spec.epsilon)
    design = _standardize_apply(x[indices], standardizer)
    design = np.concatenate([np.ones((design.shape[0], 1), dtype=np.float64), design], axis=1)
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = math.log(prevalence / (1.0 - prevalence))
    converged = False
    for iteration in range(spec.risk_max_iterations):
        logits = np.clip(design @ coefficient, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        weight = np.maximum(probability * (1.0 - probability), 1.0e-8)
        gradient = design.T @ (probability - y[indices])
        penalty = np.zeros_like(coefficient); penalty[1:] = spec.risk_l2 * coefficient[1:]
        gradient += penalty
        hessian = design.T @ (weight[:, None] * design)
        regularizer = np.eye(coefficient.size, dtype=np.float64) * spec.risk_l2
        regularizer[0, 0] = 1.0e-10
        step = np.linalg.solve(hessian + regularizer, gradient)
        coefficient -= step
        if float(np.max(np.abs(step))) <= spec.risk_tolerance:
            converged = True
            break
    return {
        "coefficient": coefficient, "standardizer": standardizer,
        "prevalence": prevalence, "fit_row_count": int(indices.size),
        "positive_count": int(np.sum(y[indices] > 0.5)), "converged": converged,
        "iteration_count": int(iteration + 1),
        "identity": {
            "fit_mask_sha256": sha256_array(mask), "fit_row_count": int(indices.size),
            "coefficient_sha256": sha256_array(coefficient), "prevalence": prevalence,
        },
    }


def predict_risk(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    if model.get("mode") == "constant":
        return np.full(np.asarray(features).shape[0], float(model["prevalence"]), dtype=np.float64)
    design = _standardize_apply(np.asarray(features, dtype=np.float64), model["standardizer"])
    design = np.concatenate([np.ones((design.shape[0], 1), dtype=np.float64), design], axis=1)
    logits = np.clip(design @ np.asarray(model["coefficient"], dtype=np.float64), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def row_squared_error(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.asarray(value, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return np.sum(difference.reshape(difference.shape[0], -1) ** 2, axis=1)


def apply_policy(control: np.ndarray, candidate: np.ndarray, selected_scale: np.ndarray, adverse_probability: np.ndarray, threshold: float) -> Mapping[str, np.ndarray]:
    control_value = np.asarray(control, dtype=np.float32)
    candidate_value = np.asarray(candidate, dtype=np.float32)
    scale = np.asarray(selected_scale, dtype=np.float64)
    probability = np.asarray(adverse_probability, dtype=np.float64)
    if scale.shape != probability.shape or scale.shape != (control_value.shape[0],):
        raise StageXError("policy vector population changed")
    keep = (scale > 0.0) & (probability <= float(threshold))
    output = control_value.copy(); output[keep] = candidate_value[keep]
    output_scale = np.zeros_like(scale); output_scale[keep] = scale[keep]
    return {"candidate": output, "selected_scale": output_scale, "kept_mask": keep}


def _stats(values: np.ndarray) -> Mapping[str, float]:
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0:
        return {"min": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(value)), "p05": float(np.quantile(value, 0.05)),
        "median": float(np.median(value)), "p95": float(np.quantile(value, 0.95)),
        "max": float(np.max(value)), "mean": float(np.mean(value)),
    }


def group_tail_metrics(control_sse: np.ndarray, candidate_sse: np.ndarray, groups: Sequence[Any], fraction: float) -> Mapping[str, Any]:
    control = np.asarray(control_sse, dtype=np.float64)
    candidate = np.asarray(candidate_sse, dtype=np.float64)
    names = np.asarray(groups).astype(str)
    records = []
    for name in sorted(set(names.tolist())):
        mask = names == name
        denominator = float(np.sum(control[mask]))
        ratio = float(np.sum(candidate[mask]) / max(denominator, 1.0e-12))
        records.append((name, ratio, int(np.sum(mask))))
    ratios = np.asarray([record[1] for record in records], dtype=np.float64)
    tail_count = max(1, int(math.ceil(float(fraction) * ratios.size)))
    order = np.argsort(ratios)[::-1]
    tail = ratios[order[:tail_count]]
    return {
        "group_count": len(records), "tail_count": tail_count,
        "ratio_stats": _stats(ratios), "worst_group_mse_ratio": float(np.max(ratios)),
        "worst_fraction_cvar_mse_ratio": float(np.mean(tail)),
        "group_ratio_sha256": sha256_array(ratios),
    }


def evaluate_policy(control: np.ndarray, raw_candidate: np.ndarray, raw_scale: np.ndarray, adverse_probability: np.ndarray, target: np.ndarray, groups: Sequence[Any], threshold: float, baseline_probability: Optional[np.ndarray], spec: StageXSpec) -> Mapping[str, Any]:
    applied = apply_policy(control, raw_candidate, raw_scale, adverse_probability, threshold)
    candidate = applied["candidate"]
    scale = applied["selected_scale"]
    accepted = scale > 0.0
    control_sse = row_squared_error(control, target)
    candidate_sse = row_squared_error(candidate, target)
    control_distance = np.sqrt(control_sse)
    candidate_distance = np.sqrt(candidate_sse)
    reduction = control_distance - candidate_distance
    relative = reduction / np.maximum(control_distance, spec.epsilon)
    overall_ratio = float(np.sum(candidate_sse) / max(float(np.sum(control_sse)), spec.epsilon))
    if np.any(accepted):
        accepted_ratio = float(np.sum(candidate_sse[accepted]) / max(float(np.sum(control_sse[accepted])), spec.epsilon))
        positive_rate = float(np.mean(reduction[accepted] > 0.0))
        relative_mean = float(np.mean(relative[accepted]))
    else:
        accepted_ratio = 1.0; positive_rate = 0.0; relative_mean = 0.0
    delta = candidate_sse - control_sse
    adverse_mass = float(np.sum(np.maximum(delta, 0.0)) / max(float(np.sum(control_sse)), spec.epsilon))
    beneficial_mass = float(np.sum(np.maximum(-delta, 0.0)) / max(float(np.sum(control_sse)), spec.epsilon))
    raw_accepted = np.asarray(raw_scale, dtype=np.float64) > 0.0
    raw_adverse = (row_squared_error(raw_candidate, target) > control_sse).astype(np.float64)
    if np.any(raw_accepted):
        brier = float(np.mean((np.asarray(adverse_probability)[raw_accepted] - raw_adverse[raw_accepted]) ** 2))
        if baseline_probability is None:
            prevalence = float(np.mean(raw_adverse[raw_accepted]))
            baseline_brier = float(np.mean((prevalence - raw_adverse[raw_accepted]) ** 2))
        else:
            baseline_brier = float(np.mean((np.asarray(baseline_probability)[raw_accepted] - raw_adverse[raw_accepted]) ** 2))
    else:
        brier = baseline_brier = 0.0
    return {
        "row_count": int(control_sse.size), "selected_row_count": int(np.sum(accepted)),
        "acceptance_rate": float(np.mean(accepted)), "overall_mse_ratio": overall_ratio,
        "accepted_row_mse_ratio": accepted_ratio,
        "positive_distance_reduction_rate": positive_rate,
        "relative_distance_reduction_mean": relative_mean,
        "adverse_sse_mass": adverse_mass, "beneficial_sse_mass": beneficial_mass,
        "net_sse_reduction_mass": beneficial_mass - adverse_mass,
        "risk_brier_score": brier, "risk_constant_brier_score": baseline_brier,
        "risk_brier_nonworse": bool(brier <= baseline_brier + 1.0e-12),
        "group_tail": group_tail_metrics(control_sse, candidate_sse, groups, spec.group_cvar_fraction),
        "distance_reduction_stats": _stats(reduction[accepted]),
        "adverse_row_count": int(np.sum(raw_accepted & (raw_adverse > 0.5))),
        "raw_accepted_row_count": int(np.sum(raw_accepted)),
        "output_candidate_sha256": sha256_array(candidate),
        "output_selected_scale_sha256": sha256_array(scale),
    }


def policy_is_eligible(records: Mapping[int, Mapping[str, Any]], baselines: Mapping[int, Mapping[str, Any]], spec: StageXSpec) -> Mapping[str, Any]:
    checks: Dict[str, Dict[str, bool]] = {}
    all_pass = True
    for timestep in LOCKED_TIMESTEPS:
        record = records[int(timestep)]; baseline = baselines[int(timestep)]
        baseline_adverse = float(baseline["adverse_sse_mass"])
        checks[str(timestep)] = {
            "acceptance": float(record["acceptance_rate"]) >= spec.minimum_acceptance_rate,
            "overall_mse": float(record["overall_mse_ratio"]) < 1.0,
            "accepted_mse": float(record["accepted_row_mse_ratio"]) < 1.0,
            "positive_reduction": float(record["positive_distance_reduction_rate"]) > spec.minimum_positive_reduction_rate,
            "relative_reduction": float(record["relative_distance_reduction_mean"]) > 0.0,
            "adverse_sse_reduction": float(record["adverse_sse_mass"]) <= baseline_adverse * (1.0 - spec.adverse_sse_reduction_fraction),
            "group_cvar_improvement": float(record["group_tail"]["worst_fraction_cvar_mse_ratio"]) < float(baseline["group_tail"]["worst_fraction_cvar_mse_ratio"]),
            "risk_brier_nonworse": record["risk_brier_nonworse"] is True,
        }
        all_pass = all_pass and all(checks[str(timestep)].values())
    return {"all_timesteps_pass": bool(all_pass), "checks": checks}


def select_joint_policy(policy_records: Mapping[str, Mapping[int, Mapping[str, Any]]], baseline_records: Mapping[int, Mapping[str, Any]], spec: StageXSpec) -> Mapping[str, Any]:
    population = policy_population()
    ids = {policy.policy_id for policy in population}
    if set(policy_records) != ids:
        raise StageXError("joint policy record population changed")
    eligibility = {policy.policy_id: policy_is_eligible(policy_records[policy.policy_id], baseline_records, spec) for policy in population}
    def score(policy: TailPolicy) -> Tuple[Any, ...]:
        records = policy_records[policy.policy_id]
        eligible = eligibility[policy.policy_id]["all_timesteps_pass"]
        return (
            0 if eligible else 1,
            max(float(records[t]["group_tail"]["worst_fraction_cvar_mse_ratio"]) for t in LOCKED_TIMESTEPS),
            max(float(records[t]["overall_mse_ratio"]) for t in LOCKED_TIMESTEPS),
            max(float(records[t]["adverse_sse_mass"]) for t in LOCKED_TIMESTEPS),
            -min(float(records[t]["acceptance_rate"]) for t in LOCKED_TIMESTEPS),
            DIRECTION_SHRINKAGES.index(policy.shrinkage), RISK_THRESHOLDS.index(policy.risk_threshold),
        )
    ordered = sorted(population, key=score)
    selected = ordered[0]
    return {
        "selected_policy": asdict(selected), "selected_policy_id": selected.policy_id,
        "selected_policy_inner_eligible": eligibility[selected.policy_id]["all_timesteps_pass"],
        "eligible_policy_ids": [policy.policy_id for policy in population if eligibility[policy.policy_id]["all_timesteps_pass"]],
        "policy_eligibility": eligibility,
        "selection_score": list(score(selected)),
        "diagnostic_fallback_used": not eligibility[selected.policy_id]["all_timesteps_pass"],
    }


def modal_policy(selections: Sequence[str]) -> Mapping[str, Any]:
    population = policy_population(); order = {policy.policy_id: index for index, policy in enumerate(population)}
    if len(selections) != EXPECTED_OUTER_FOLDS or any(value not in order for value in selections):
        raise StageXError("outer policy selection population changed")
    counts = {policy.policy_id: selections.count(policy.policy_id) for policy in population}
    selected_id = min(counts, key=lambda key: (-counts[key], order[key]))
    policy = next(item for item in population if item.policy_id == selected_id)
    return {"policy": asdict(policy), "policy_id": selected_id, "support_count": counts[selected_id], "counts": counts}


def classify_stagex(final_records: Mapping[int, Mapping[str, Any]], baseline_records: Mapping[int, Mapping[str, Any]], outer_selections: Sequence[Mapping[str, Any]], final_policy: Mapping[str, Any], spec: StageXSpec) -> Mapping[str, Any]:
    eligibility = policy_is_eligible(final_records, baseline_records, spec)
    every_outer_inner_eligible = all(item["selected_policy_inner_eligible"] is True for item in outer_selections)
    stable = int(final_policy["support_count"]) >= spec.modal_outer_fold_minimum
    ready = eligibility["all_timesteps_pass"] and every_outer_inner_eligible and stable
    if ready:
        root = "phase314b_r258_stagex_tail_robust_nested_oof_controls_adverse_sse_on_objective_train"
        next_path = "FREEZE_STAGE_X_TAIL_ROBUST_POLICY_AND_PREREGISTER_ONE_SHOT_FROZEN_PROBE_EVALUATION"
        locus = "tail_robust_policy_ready_for_freeze"
    elif not every_outer_inner_eligible:
        root = "phase314b_r258_stagex_inner_policy_not_consistently_eligible"
        next_path = "AUDIT_OBJECTIVE_TRAIN_TAIL_POLICY_INNER_FOLD_INSTABILITY_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "inner_selection_instability"
    elif not stable:
        root = "phase314b_r258_stagex_tail_policy_is_not_outer_fold_stable"
        next_path = "AUDIT_OBJECTIVE_TRAIN_TAIL_POLICY_OUTER_FOLD_INSTABILITY_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "outer_policy_instability"
    else:
        root = "phase314b_r258_stagex_tail_robust_policy_fails_nested_oof_contract"
        next_path = "AUDIT_OBJECTIVE_TRAIN_TAIL_RISK_MODEL_OR_POLICY_FAILURE_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "nested_oof_tail_fidelity_failure"
    return {
        "scientific_status": "READY" if ready else "BLOCKED", "root_cause": root,
        "required_next_path": next_path, "primary_failure_locus": locus,
        "final_policy_eligibility": eligibility,
        "all_outer_selections_inner_eligible": every_outer_inner_eligible,
        "final_policy_outer_support_stable": stable,
    }


def _science_modules() -> Mapping[str, Any]:
    from ccda_phase3 import phase314b_r258_stagev_locked_selection_holdout_evaluation as stagev
    modules = dict(stagev._science_modules())
    modules["stagev"] = stagev
    return modules


def _objective_only_context(source: Mapping[str, Any]) -> Mapping[str, Any]:
    forbidden_exact = {
        "holdout_target", "holdout_condition", "holdout_condition_name", "holdout_groups",
        "control_predictions", "selection_holdout_mask", "frozen_probe_target",
    }
    context = {key: value for key, value in source.items() if key not in forbidden_exact and not key.startswith("holdout_")}
    required = (
        "objective_target", "objective_condition", "objective_condition_name", "objective_groups",
        "objective_control_predictions", "objective_fold_assignment", "stage_d_contract",
        "historical_geometry",
    )
    for key in required:
        if key not in context:
            raise StageXError("objective-only context missing {}".format(key))
    context.pop("target", None); context.pop("condition", None); context.pop("groups", None); context.pop("condition_name", None)
    return context


def _fit_direction(*, runtime: Mapping[str, Any], context: Mapping[str, Any], timestep: int, fit_mask: np.ndarray, predict_indices: np.ndarray, oracle: Mapping[str, Any], features: np.ndarray) -> Mapping[str, Any]:
    stagef = runtime["stagef"]; stageg = runtime["stageg"]
    definition = stagef.definition_by_id(LOCKED_BACKBONE)
    model = stageg._fit_direction_model(
        definition=definition, features=features,
        control=np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32),
        oracle_target=oracle, condition_name=context["objective_condition_name"],
        fit_mask=np.asarray(fit_mask, dtype=np.bool_), spec=runtime["stagef_spec"], permutation_seed=None,
    )
    indices = np.asarray(predict_indices, dtype=np.int64)
    control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)[indices]
    test_features = stagef.build_constraint_features(
        condition=np.asarray(context["objective_condition"], dtype=np.float32)[indices],
        control=control, condition_name=np.asarray(context["objective_condition_name"])[indices],
        feature_mode=definition.feature_mode, context=context,
    )
    prediction = stageg._predict_direction(
        definition=definition, model=model, features=test_features, control=control,
        condition_name=np.asarray(context["objective_condition_name"])[indices],
    )
    return {"direction": np.asarray(prediction, dtype=np.float64), "model_identity": copy.deepcopy(model["identity"])}


def _candidate_for_indices(*, runtime: Mapping[str, Any], context: Mapping[str, Any], timestep: int, indices: np.ndarray, direction: np.ndarray, shrinkage: float, reconstruction_spec: Any) -> Mapping[str, Any]:
    control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)[indices]
    result = runtime["stagev"].reconstruct_locked_holdout_candidate(
        control=control, base_direction=np.asarray(direction, dtype=np.float64) * float(shrinkage),
        context=context, runtime=runtime, spec=reconstruction_spec,
    )
    return result


def _risk_labels(control: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (row_squared_error(candidate, target) > row_squared_error(control, target)).astype(np.float64)


def run_nested_oof(*, root: Path, probe_payload: Mapping[str, Any], repository_head: str, stageu_contract: Mapping[str, Any], spec: Optional[StageXSpec] = None) -> Mapping[str, Any]:
    active = StageXSpec() if spec is None else spec; active.validate(); validate_environment_variables()
    modules = _science_modules(); stagev = modules["stagev"]
    environment = modules["stages_resume3"].validate_probe_payload_for_science(probe_payload, resume2a=modules["resume2a"])
    runtime_modules = modules["stageo"]._runtime_modules(); stagea = runtime_modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment); cold = stagea.assert_cold_cuda_context_portable()
    runtime = dict(modules["stageo"]._prepare_runtime(Path(root).resolve(), environment))
    runtime.update({**modules, "stageu_gate": stageu_contract["aligned_gate_lock"]})
    context = _objective_only_context(runtime["context"]); runtime["context"] = context
    population = validate_objective_population(context)
    reconstruction_spec = modules["stager"].StageRSpec(); reconstruction_spec.validate()
    assignment = np.asarray(context["objective_fold_assignment"], dtype=np.int64)
    groups = np.asarray(context["objective_groups"]).astype(str)
    target = np.asarray(context["objective_target"], dtype=np.float32)
    stagef = runtime["stagef"]
    definition = stagef.definition_by_id(LOCKED_BACKBONE)
    features_by_timestep = {}
    oracle_by_timestep = {}
    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        features_by_timestep[int(timestep)] = stagef.build_constraint_features(
            condition=context["objective_condition"], control=control,
            condition_name=context["objective_condition_name"], feature_mode=definition.feature_mode, context=context,
        )
        oracle_by_timestep[int(timestep)] = stagef.generate_projected_oracle_target(
            control=control, target=target, groups=groups, condition_name=context["objective_condition_name"],
            timestep=int(timestep), context=context, direction_spec=runtime["direction_spec"],
            integrator_spec=runtime["integrator_spec"], spec=runtime["stagef_spec"],
        )
    outer_selections: List[Mapping[str, Any]] = []
    outer_outputs: Dict[int, Dict[int, Dict[float, Mapping[str, Any]]]] = {}
    counts = {"direction_fit_count": 0, "candidate_generation_count": 0, "risk_fit_count": 0, "internal_scale_attempt_count": 0, "inner_policy_evaluation_count": 0}
    for outer_fold in range(EXPECTED_OUTER_FOLDS):
        outer_test = assignment == outer_fold; outer_train = ~outer_test
        outer_train_indices = np.flatnonzero(outer_train); outer_test_indices = np.flatnonzero(outer_test)
        inner_candidates: Dict[int, Dict[float, np.ndarray]] = {}
        inner_scales: Dict[int, Dict[float, np.ndarray]] = {}
        inner_directions: Dict[int, np.ndarray] = {}
        for timestep in LOCKED_TIMESTEPS:
            shape = np.asarray(context["objective_control_predictions"][int(timestep)]).shape
            inner_directions[int(timestep)] = np.zeros(shape, dtype=np.float64)
            inner_candidates[int(timestep)] = {s: np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32).copy() for s in DIRECTION_SHRINKAGES}
            inner_scales[int(timestep)] = {s: np.zeros(EXPECTED_ROWS, dtype=np.float64) for s in DIRECTION_SHRINKAGES}
        inner_fold_values = [value for value in range(EXPECTED_OUTER_FOLDS) if value != outer_fold]
        if len(inner_fold_values) != EXPECTED_INNER_FOLDS:
            raise StageXError("inner fold population changed")
        for inner_fold in inner_fold_values:
            validation_mask = outer_train & (assignment == inner_fold)
            fit_mask = outer_train & ~validation_mask
            validation_indices = np.flatnonzero(validation_mask)
            for timestep in LOCKED_TIMESTEPS:
                fitted = _fit_direction(runtime=runtime, context=context, timestep=timestep, fit_mask=fit_mask, predict_indices=validation_indices, oracle=oracle_by_timestep[timestep], features=features_by_timestep[timestep])
                counts["direction_fit_count"] += 1
                inner_directions[timestep][validation_indices] = fitted["direction"]
                for shrinkage in DIRECTION_SHRINKAGES:
                    generated = _candidate_for_indices(runtime=runtime, context=context, timestep=timestep, indices=validation_indices, direction=fitted["direction"], shrinkage=shrinkage, reconstruction_spec=reconstruction_spec)
                    counts["candidate_generation_count"] += 1; counts["internal_scale_attempt_count"] += EXPECTED_INTERNAL_SCALE_COUNT
                    inner_candidates[timestep][shrinkage][validation_indices] = generated["candidate"]
                    inner_scales[timestep][shrinkage][validation_indices] = generated["selected_scale"]
        risk_oof: Dict[int, Dict[float, np.ndarray]] = {}
        risk_constant_oof: Dict[int, Dict[float, np.ndarray]] = {}
        for timestep in LOCKED_TIMESTEPS:
            control_all = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
            risk_oof[timestep] = {}; risk_constant_oof[timestep] = {}
            for shrinkage in DIRECTION_SHRINKAGES:
                descriptors = compact_risk_descriptors(control_all, inner_directions[timestep] * shrinkage, inner_candidates[timestep][shrinkage], inner_scales[timestep][shrinkage])
                labels = np.zeros(EXPECTED_ROWS, dtype=np.float64)
                labels[outer_train_indices] = _risk_labels(
                    control_all[outer_train_indices],
                    inner_candidates[timestep][shrinkage][outer_train_indices],
                    target[outer_train_indices],
                )
                probabilities = np.ones(EXPECTED_ROWS, dtype=np.float64)
                constants = np.ones(EXPECTED_ROWS, dtype=np.float64)
                for inner_fold in inner_fold_values:
                    validation_mask = outer_train & (assignment == inner_fold)
                    fit_mask = outer_train & ~validation_mask & (inner_scales[timestep][shrinkage] > 0.0)
                    model = fit_risk_model(descriptors, labels, fit_mask, active); counts["risk_fit_count"] += 1
                    validation_indices = np.flatnonzero(validation_mask)
                    probabilities[validation_indices] = predict_risk(model, descriptors[validation_indices])
                    constants[validation_indices] = float(model["prevalence"])
                risk_oof[timestep][shrinkage] = probabilities
                risk_constant_oof[timestep][shrinkage] = constants
        baseline_records: Dict[int, Mapping[str, Any]] = {}
        policy_records: Dict[str, Dict[int, Mapping[str, Any]]] = {p.policy_id: {} for p in policy_population()}
        for timestep in LOCKED_TIMESTEPS:
            indices = outer_train_indices
            control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)[indices]
            baseline_records[timestep] = evaluate_policy(
                control, inner_candidates[timestep][1.0][indices], inner_scales[timestep][1.0][indices],
                np.zeros(indices.size, dtype=np.float64), target[indices], groups[indices], 1.0,
                None, active,
            )
            for policy in policy_population():
                policy_records[policy.policy_id][timestep] = evaluate_policy(
                    control, inner_candidates[timestep][policy.shrinkage][indices], inner_scales[timestep][policy.shrinkage][indices],
                    risk_oof[timestep][policy.shrinkage][indices], target[indices], groups[indices], policy.risk_threshold,
                    risk_constant_oof[timestep][policy.shrinkage][indices], active,
                )
                counts["inner_policy_evaluation_count"] += 1
        selection = select_joint_policy(policy_records, baseline_records, active)
        selection["outer_fold"] = outer_fold; selection["inner_train_row_count"] = int(np.sum(outer_train)); selection["outer_test_row_count"] = int(np.sum(outer_test))
        outer_selections.append(selection)
        outer_outputs[outer_fold] = {}
        for timestep in LOCKED_TIMESTEPS:
            fitted = _fit_direction(runtime=runtime, context=context, timestep=timestep, fit_mask=outer_train, predict_indices=outer_test_indices, oracle=oracle_by_timestep[timestep], features=features_by_timestep[timestep])
            counts["direction_fit_count"] += 1
            outer_outputs[outer_fold][timestep] = {}
            control_all = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
            for shrinkage in DIRECTION_SHRINKAGES:
                generated = _candidate_for_indices(runtime=runtime, context=context, timestep=timestep, indices=outer_test_indices, direction=fitted["direction"], shrinkage=shrinkage, reconstruction_spec=reconstruction_spec)
                counts["candidate_generation_count"] += 1; counts["internal_scale_attempt_count"] += EXPECTED_INTERNAL_SCALE_COUNT
                training_descriptors = compact_risk_descriptors(control_all, inner_directions[timestep] * shrinkage, inner_candidates[timestep][shrinkage], inner_scales[timestep][shrinkage])
                training_labels = np.zeros(EXPECTED_ROWS, dtype=np.float64)
                training_labels[outer_train_indices] = _risk_labels(
                    control_all[outer_train_indices],
                    inner_candidates[timestep][shrinkage][outer_train_indices],
                    target[outer_train_indices],
                )
                risk_fit_mask = outer_train & (inner_scales[timestep][shrinkage] > 0.0)
                model = fit_risk_model(training_descriptors, training_labels, risk_fit_mask, active); counts["risk_fit_count"] += 1
                test_control = control_all[outer_test_indices]
                test_descriptors = compact_risk_descriptors(test_control, fitted["direction"] * shrinkage, generated["candidate"], generated["selected_scale"])
                outer_outputs[outer_fold][timestep][shrinkage] = {
                    "indices": outer_test_indices, "control": test_control,
                    "direction": fitted["direction"], "candidate": generated["candidate"],
                    "selected_scale": generated["selected_scale"],
                    "risk_probability": predict_risk(model, test_descriptors),
                    "risk_constant_probability": np.full(outer_test_indices.size, float(model["prevalence"]), dtype=np.float64),
                    "gate_counts": {
                        "length_log_z_element_mismatch_count": generated["length_log_z_element_mismatch_count"],
                        "aligned_upper_element_failure_count": generated["aligned_upper_element_failure_count"],
                        "strict_pass_aligned_fail_row_count": generated["strict_pass_aligned_fail_row_count"],
                    },
                }
    final_policy = modal_policy([item["selected_policy_id"] for item in outer_selections])
    policy = TailPolicy(**final_policy["policy"]); policy.validate()
    final_records: Dict[int, Mapping[str, Any]] = {}; baseline_records = {}; gate_totals = {}
    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        raw_candidate = control.copy(); raw_scale = np.zeros(EXPECTED_ROWS, dtype=np.float64)
        risk = np.ones(EXPECTED_ROWS, dtype=np.float64); constant = np.ones(EXPECTED_ROWS, dtype=np.float64)
        baseline_candidate = control.copy(); baseline_scale = np.zeros(EXPECTED_ROWS, dtype=np.float64)
        gate = {"length_log_z_element_mismatch_count": 0, "aligned_upper_element_failure_count": 0, "strict_pass_aligned_fail_row_count": 0}
        for outer_fold in range(EXPECTED_OUTER_FOLDS):
            selected_output = outer_outputs[outer_fold][timestep][policy.shrinkage]
            indices = selected_output["indices"]
            raw_candidate[indices] = selected_output["candidate"]; raw_scale[indices] = selected_output["selected_scale"]
            risk[indices] = selected_output["risk_probability"]; constant[indices] = selected_output["risk_constant_probability"]
            baseline_output = outer_outputs[outer_fold][timestep][1.0]
            baseline_candidate[indices] = baseline_output["candidate"]; baseline_scale[indices] = baseline_output["selected_scale"]
            for key in gate: gate[key] += int(selected_output["gate_counts"][key])
        final_records[timestep] = evaluate_policy(control, raw_candidate, raw_scale, risk, target, groups, policy.risk_threshold, constant, active)
        baseline_records[timestep] = evaluate_policy(control, baseline_candidate, baseline_scale, np.zeros(EXPECTED_ROWS), target, groups, 1.0, None, active)
        gate_totals[str(timestep)] = gate
        if any(value != 0 for value in gate.values()):
            raise StageXError("tolerance-aligned gate regressed at t={}".format(timestep))
    classification = classify_stagex(final_records, baseline_records, outer_selections, final_policy, active)
    recommendation = None
    if classification["scientific_status"] == "READY":
        recommendation = {
            "backbone_id": LOCKED_BACKBONE, "timesteps": list(LOCKED_TIMESTEPS),
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "direction_shrinkage": policy.shrinkage, "adverse_risk_threshold": policy.risk_threshold,
            "risk_descriptor": "target_independent_compact_direction_candidate_geometry_v1",
            "nested_group_oof": {"outer_folds": 6, "inner_folds_per_outer": 5},
            "aligned_gate_contract_sha256": stageu_contract["aligned_gate_lock"]["inner_gate_contract_sha256"],
        }
    payload: Dict[str, Any] = {
        "phase": PHASE, "schema": WORKER_SCHEMA, "execution_verdict": "PASS", **classification,
        "process_id": os.getpid(), "repository_head": repository_head,
        "environment_sha256": sha256_bytes(stable_json_bytes(environment)), "cold_cuda_precheck": copy.deepcopy(dict(cold)),
        "locked_scientific_base": {"backbone_id": LOCKED_BACKBONE, "timesteps": list(LOCKED_TIMESTEPS), "external_multiplier": 0.25, "tolerance_factor": 5.0, "internal_scale_count": 7},
        "stagex_spec": asdict(active), "policy_population": [asdict(item) for item in policy_population()],
        "population": population, "outer_fold_selections": outer_selections, "final_modal_policy": final_policy,
        "baseline_nested_oof_records": {str(k): v for k, v in baseline_records.items()},
        "tail_robust_nested_oof_records": {str(k): v for k, v in final_records.items()},
        "gate_totals": gate_totals, "execution_counts": counts,
        "selected_configuration": None, "train_only_recommendation": recommendation,
        "selection_holdout_evaluation_count_added": 0, "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluated_in_stagex": False, "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_payload(payload)
    return payload


def validate_worker_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageXError("Stage-X worker schema/verdict changed")
    if payload.get("selection_holdout_evaluation_count_added") != 0 or payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageXError("holdout count changed")
    if payload.get("selected_configuration") is not None or payload.get("rerun_authorized") is not False:
        raise StageXError("selection/rerun boundary changed")
    records = payload.get("tail_robust_nested_oof_records")
    if not isinstance(records, Mapping) or set(records) != {"10", "25", "50"}:
        raise StageXError("Stage-X timestep records changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageXError("a forbidden boundary became true")
    if payload.get("scientific_status") == "READY" and payload.get("train_only_recommendation") is None:
        raise StageXError("READY result lacks train-only recommendation")
    if payload.get("scientific_status") == "BLOCKED" and payload.get("train_only_recommendation") is not None:
        raise StageXError("BLOCKED result retained recommendation")
    expected_sha = sha256_bytes(stable_json_bytes({k: v for k, v in payload.items() if k != "scientific_result_sha256"}))
    if payload.get("scientific_result_sha256") != expected_sha:
        raise StageXError("Stage-X worker self-hash changed")


def blocked_report(repository: Optional[Mapping[str, Any]], error: BaseException) -> Mapping[str, Any]:
    return {
        "phase": PHASE, "schema": BLOCKED_SCHEMA, "execution_verdict": "BLOCKED", "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagex_execution_contract_failed",
        "required_next_path": "DESIGN_ADD_ONLY_STAGEX_EXECUTION_RECOVERY_WITHOUT_HOLDOUT_OR_PROBE_ACCESS",
        "primary_failure_locus": "execution_contract", "selected_configuration": None, "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository), "error_type": type(error).__name__, "error_message": str(error),
        "selection_holdout_evaluation_count_added": 0, "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False, "complete_nested_oof_population_claimed": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageXError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists(): temporary.unlink()
