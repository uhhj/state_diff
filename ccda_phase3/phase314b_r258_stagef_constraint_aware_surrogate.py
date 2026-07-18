"""Phase3.14b-r2.5.8 Stage F constraint-aware direction surrogate.

Stage E proved that its frozen constrained direct-x0 integrator is oracle
reachable at t=10/25/50, while every deployable Stage-D raw-residual surrogate
fails after integration.  Stage F therefore keeps the Stage-E integrator
fixed and changes only the train-only surrogate supervision:

* objective-train ground truth is passed through the frozen oracle-proven
  Stage-E integrator to create a projected-oracle displacement target;
* six-fold grouped OOF models learn that target from deployable observations;
* feature and output transforms are fitted inside each training fold;
* OOF test targets are used only after prediction for evaluation;
* the 236-row selection holdout remains closed until candidate, model,
  parameterization, and gates are locked;
* the 126-row frozen probe remains closed.

No diffusion model is trained and no surrogate weights, coefficients, PCA
bases, random features, predictions, checkpoints, NPZ files, caches, images,
or videos are persisted.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r258_stageb_direct_x0_reachability as stageb258
from ccda_phase3 import phase314b_r258_stagec_balanced_geometry as stagec258
from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged258
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee258

PHASE = "Phase3.14b-r2.5.8 Stage F"
PHASE_ID = "phase314b_r258_stagef"

BASE_EVIDENCE_COMMIT = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_CONTRACT = "reports/phase3_14b_r258_stagee_contract.json"
BASE_WORKER = "reports/phase3_14b_r258_stagee_worker_evidence.json"
BASE_SUMMARY = "reports/phase3_14b_r258_stagee_summary.json"
BASE_REPORT = "reports/phase3_14b_r258_stagee_report.md"
BASE_TEST_GATE = "reports/phase3_14b_r258_stagee_test_gate_summary.json"
BASE_BOUND_FILES = (
    BASE_CONTRACT,
    BASE_WORKER,
    BASE_SUMMARY,
    BASE_REPORT,
    BASE_TEST_GATE,
)

EXPECTED_BASE_WORKER_SHA256 = (
    "e71479a95e23624e8e2adab9f7e1547f96718681be8e3f43b0a2157420147ad1"
)
EXPECTED_BASE_INTERNAL_CONTRACT_SHA256 = (
    "dfa66bc374871521195d6ffca57e047316b0294d37209b57c1f9493b826f8560"
)
EXPECTED_BASE_CONTRACT_FILE_SHA256 = (
    "7a266c9ad1e8a32e749c2aeeba829eb62c0d38d151976d3d5b20a63b28862c4d"
)
EXPECTED_BASE_SELECTION_SHA256 = (
    "068ebecd78e5ac0231d5e3b407855982867432625ef6f5d8f23a37bcd263b94f"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stagee_oracle_valid_but_surrogate_direction_not_integrable"
)
EXPECTED_BASE_NEXT_PATH = (
    "CALIBRATE_CONSTRAINT_AWARE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY"
)
EXPECTED_COMPATIBILITY_SHA256 = (
    "03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec"
)

SOURCE_BOUND_SHA256 = {
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py": (
        "d952e169196916b3ac9ad7a86d1944cfe48281599463f053718a27ea89238f8c"
    ),
    "scripts/phase3_14b_r258_stagee_worker.py": (
        "a918f526ff34878c712e0b0139dc755ebc42fefe026b50e308b8468ffbecf7e8"
    ),
    "scripts/phase3_14b_r258_stagee_run_calibration.py": (
        "4996e41ba3b828759abf925d8ab63b492285a43297d229db029e94937a6be846"
    ),
    "scripts/phase3_14b_r258_stagee_test_gate.py": (
        "af4a7382269e74287427f30cc6eeeecbebf5e6d676d52688f3978ea8be128c85"
    ),
    "scripts/phase3_14b_r258_stagee_blocked.py": (
        "99f9f9c434cb0863016c7f348f1eed7bc08bb76d0630acdb59ae631eb22a2a4f"
    ),
    "scripts/phase3_14b_r258_stagee_run.sh": (
        "39f211712838308fe177d8ca6e4716ff48b6eef184348931484e023778878e8e"
    ),
    "tests/test_phase3_14b_r258_stagee_constrained_integrator.py": (
        "ff3221548e06ec14e8101b047ab855d2f5ee216e50b97d2a2db5baa3d02a65d8"
    ),
    "ccda_phase3/phase314b_r258_staged_direction_surrogate.py": (
        "946fca9d84b54e4affd35ab890970ba1f17154aa8b925f66746994ffe484bbb9"
    ),
}

STAGEF_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py",
    "scripts/phase3_14b_r258_stagef_worker.py",
    "scripts/phase3_14b_r258_stagef_run_calibration.py",
    "scripts/phase3_14b_r258_stagef_test_gate.py",
    "scripts/phase3_14b_r258_stagef_blocked.py",
    "scripts/phase3_14b_r258_stagef_run.sh",
    "tests/test_phase3_14b_r258_stagef_constraint_aware_surrogate.py",
)
IMPLEMENTATION_MESSAGE = (
    "Add Phase3.14b-r2.5.8 constraint-aware projected-oracle direction surrogate"
)
EVIDENCE_MESSAGE = (
    "Record Phase3.14b-r2.5.8 constraint-aware surrogate evidence"
)

CONDITION_VALUES = staged258.CONDITION_VALUES
TIMESTEPS = (10, 25, 50)
ORACLE_INTEGRATOR_ID = "oracle_z4_m2_r25"
CONSTRAINT_Z_CLIP = 4.0
STRUCTURAL_ZERO_Z_FILL = 0.0
CONSTRAINT_SEGMENT_COUNT = stageb.FUTURE_STEPS * (stageb.BEADS - 1)
FULL_CENTERED_CONSTRAINT_DIMENSION = 619
FULL_SEGMENT_CONSTRAINT_DIMENSION = 617


class ConstraintAwareSurrogateError(RuntimeError):
    """Raised when evidence, leakage, fitting, or selection invariants fail."""


class ConstraintZULPAdmissionError(ConstraintAwareSurrogateError):
    """Carry a bounded, JSON-safe ULP-formula diagnosis to the single runner."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: Mapping[str, Any],
        required_next_path: str,
    ) -> None:
        super().__init__(message)
        self.diagnostic = copy.deepcopy(dict(diagnostic))
        self.required_next_path = str(required_next_path)


@dataclass(frozen=True)
class ConstraintAwareDefinition:
    candidate_id: str
    feature_mode: str
    target_mode: str
    model_mode: str
    ridge_alpha: float
    output_rank: int
    rff_dimension: int
    fit_population: str
    role: str

    def validate(self) -> None:
        if self.feature_mode not in (
            "none",
            "condition_only",
            "anchor",
            "full_centered_constraint",
            "full_segment_constraint",
        ):
            raise ValueError("unknown constraint-aware feature mode")
        if self.target_mode not in (
            "raw_reference",
            "projected_point",
            "projected_segment",
            "projected_oracle",
        ):
            raise ValueError("unknown constraint-aware target mode")
        if self.model_mode not in (
            "raw_reference",
            "global_mean",
            "condition_mean",
            "reduced_rank_ridge",
            "rff_ridge",
            "oracle",
        ):
            raise ValueError("unknown constraint-aware model mode")
        if self.fit_population not in ("all", "oracle_feasible"):
            raise ValueError("unknown fit population")
        if self.role not in (
            "negative_control",
            "diagnostic_control",
            "selectable",
            "oracle_control",
        ):
            raise ValueError("unknown candidate role")
        if self.model_mode in ("reduced_rank_ridge", "rff_ridge"):
            if self.ridge_alpha <= 0.0:
                raise ValueError("ridge alpha must be positive")
        elif self.ridge_alpha != 0.0:
            raise ValueError("non-ridge candidate has ridge alpha")
        if self.model_mode == "reduced_rank_ridge":
            if self.output_rank not in (32, 64):
                raise ValueError("reduced output rank changed")
        elif self.output_rank != 0:
            raise ValueError("non-reduced model has output rank")
        if self.model_mode == "rff_ridge":
            if self.rff_dimension != 256:
                raise ValueError("RFF dimension changed")
        elif self.rff_dimension != 0:
            raise ValueError("non-RFF model has RFF dimension")
        if self.role == "selectable":
            if self.feature_mode not in (
                "anchor",
                "full_centered_constraint",
                "full_segment_constraint",
            ):
                raise ValueError("selectable feature is not deployable")
            if self.target_mode not in (
                "projected_point",
                "projected_segment",
            ):
                raise ValueError("selectable target is not constraint-aware")
        if self.model_mode == "condition_mean" and self.role != "diagnostic_control":
            raise ValueError("condition mean is diagnostic only")
        if self.model_mode == "oracle" and self.role != "oracle_control":
            raise ValueError("oracle model is oracle-control only")
        if self.target_mode == "raw_reference" and self.model_mode != "raw_reference":
            raise ValueError("raw-reference target/model mismatch")
        if self.model_mode == "raw_reference" and self.role != "negative_control":
            raise ValueError("raw reference must be a negative control")


CANDIDATE_DEFINITIONS = (
    ConstraintAwareDefinition(
        "raw_centered_a10_reference",
        "anchor",
        "raw_reference",
        "raw_reference",
        0.0,
        0,
        0,
        "all",
        "negative_control",
    ),
    ConstraintAwareDefinition(
        "projected_global_mean",
        "none",
        "projected_point",
        "global_mean",
        0.0,
        0,
        0,
        "all",
        "negative_control",
    ),
    ConstraintAwareDefinition(
        "projected_condition_mean",
        "condition_only",
        "projected_point",
        "condition_mean",
        0.0,
        0,
        0,
        "all",
        "diagnostic_control",
    ),
    ConstraintAwareDefinition(
        "anchor_point_rr32_all",
        "anchor",
        "projected_point",
        "reduced_rank_ridge",
        10.0,
        32,
        0,
        "all",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "anchor_point_rr32_feasible",
        "anchor",
        "projected_point",
        "reduced_rank_ridge",
        10.0,
        32,
        0,
        "oracle_feasible",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "full_point_rr32_all",
        "full_centered_constraint",
        "projected_point",
        "reduced_rank_ridge",
        10.0,
        32,
        0,
        "all",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "full_point_rr32_feasible",
        "full_centered_constraint",
        "projected_point",
        "reduced_rank_ridge",
        10.0,
        32,
        0,
        "oracle_feasible",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "full_point_rr64_feasible",
        "full_centered_constraint",
        "projected_point",
        "reduced_rank_ridge",
        10.0,
        64,
        0,
        "oracle_feasible",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "segment_target_rr32_all",
        "full_segment_constraint",
        "projected_segment",
        "reduced_rank_ridge",
        10.0,
        32,
        0,
        "all",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "segment_target_rr32_feasible",
        "full_segment_constraint",
        "projected_segment",
        "reduced_rank_ridge",
        10.0,
        32,
        0,
        "oracle_feasible",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "segment_target_rr64_feasible",
        "full_segment_constraint",
        "projected_segment",
        "reduced_rank_ridge",
        10.0,
        64,
        0,
        "oracle_feasible",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "full_point_rff256_feasible",
        "full_centered_constraint",
        "projected_point",
        "rff_ridge",
        10.0,
        0,
        256,
        "oracle_feasible",
        "selectable",
    ),
    ConstraintAwareDefinition(
        "projected_oracle_control",
        "none",
        "projected_oracle",
        "oracle",
        0.0,
        0,
        0,
        "all",
        "oracle_control",
    ),
)


@dataclass(frozen=True)
class ConstraintAwareSpec:
    timesteps: Tuple[int, ...] = TIMESTEPS
    grouped_cv_folds: int = 6
    objective_train_noise_seed_offset: int = 8801
    standardizer_epsilon: float = 1.0e-12
    ridge_jitter: float = 1.0e-10
    rff_seed: int = 240319
    permutation_seed: int = 240713
    projected_oracle_feasible_min: float = 0.98
    projected_direction_mean_min: float = 0.30
    projected_direction_nonnegative_min: float = 0.75
    projected_direction_condition_min: float = 0.20
    projected_direction_fold_min: float = 0.10
    projected_global_margin_min: float = 0.10
    projected_condition_margin_min: float = 0.05
    holdout_projected_mean_min: float = 0.20
    holdout_projected_nonnegative_min: float = 0.65
    holdout_projected_condition_min: float = 0.15
    permutation_cosine_max: float = 0.10
    permutation_margin_min: float = 0.10
    translation_offset_xy: Tuple[float, float] = (0.375, -0.625)
    translation_tolerance: float = 2.0e-5
    translation_structural_tolerance: float = 1.0e-10
    translation_float32_z_ulp_factor: float = 8.0
    translation_float32_z_bound_max: float = 1.0e-2
    constraint_z_clip: float = CONSTRAINT_Z_CLIP

    def validate(self) -> None:
        if self.timesteps != TIMESTEPS:
            raise ValueError("timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("grouped CV fold count changed")
        if self.objective_train_noise_seed_offset != 8801:
            raise ValueError("objective-train noise seed changed")
        if tuple(definition.candidate_id for definition in CANDIDATE_DEFINITIONS) != (
            "raw_centered_a10_reference",
            "projected_global_mean",
            "projected_condition_mean",
            "anchor_point_rr32_all",
            "anchor_point_rr32_feasible",
            "full_point_rr32_all",
            "full_point_rr32_feasible",
            "full_point_rr64_feasible",
            "segment_target_rr32_all",
            "segment_target_rr32_feasible",
            "segment_target_rr64_feasible",
            "full_point_rff256_feasible",
            "projected_oracle_control",
        ):
            raise ValueError("candidate order changed")
        for definition in CANDIDATE_DEFINITIONS:
            definition.validate()
        for value in (
            self.standardizer_epsilon,
            self.ridge_jitter,
            self.projected_oracle_feasible_min,
            self.projected_direction_mean_min,
            self.projected_direction_nonnegative_min,
            self.projected_direction_condition_min,
            self.projected_global_margin_min,
            self.projected_condition_margin_min,
            self.holdout_projected_mean_min,
            self.holdout_projected_nonnegative_min,
            self.holdout_projected_condition_min,
            self.permutation_margin_min,
            self.translation_tolerance,
            self.translation_structural_tolerance,
            self.translation_float32_z_ulp_factor,
            self.translation_float32_z_bound_max,
            self.constraint_z_clip,
        ):
            if float(value) <= 0.0:
                raise ValueError("positive Stage-F threshold is invalid")
        if not 0.0 <= self.permutation_cosine_max < 1.0:
            raise ValueError("permutation cosine threshold is invalid")
        if self.constraint_z_clip != CONSTRAINT_Z_CLIP:
            raise ValueError("constraint z clip changed")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be serialized")
        return value
    raise TypeError("unsupported JSON type: {!r}".format(type(value)))


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


def atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError("refusing to overwrite write-once output: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / ".{}.{}.tmp".format(target.name, os.getpid())
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConstraintAwareSurrogateError("JSON root is not an object")
    return value


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def assert_commit_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(root),
    )
    if completed.returncode != 0:
        raise ConstraintAwareSurrogateError(
            "commit {} is not an ancestor of {}".format(ancestor, descendant)
        )


def assert_file_bound_to_commit(root: Path, relative: str, commit: str) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        ["git", "show", "{}:{}".format(commit, relative)],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise ConstraintAwareSurrogateError(
            "file differs from immutable commit: {}".format(relative)
        )
    return sha256_bytes(observed)


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    head = git_output(repository_root, "rev-parse", "HEAD")
    assert_commit_ancestor(repository_root, BASE_EVIDENCE_COMMIT, head)
    file_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            BASE_EVIDENCE_COMMIT,
        )
        for relative in BASE_BOUND_FILES
    }
    if file_sha[BASE_CONTRACT] != EXPECTED_BASE_CONTRACT_FILE_SHA256:
        raise ConstraintAwareSurrogateError("Stage-E contract file SHA changed")
    source_sha = {}
    for relative, expected in SOURCE_BOUND_SHA256.items():
        actual = assert_file_bound_to_commit(
            repository_root,
            relative,
            BASE_EVIDENCE_COMMIT,
        )
        if actual != expected:
            raise ConstraintAwareSurrogateError(
                "Stage-E source SHA changed: {}".format(relative)
            )
        source_sha[relative] = actual

    contract = load_json(repository_root / BASE_CONTRACT)
    worker = load_json(repository_root / BASE_WORKER)
    summary = load_json(repository_root / BASE_SUMMARY)
    test_gate = load_json(repository_root / BASE_TEST_GATE)
    if contract.get("contract_sha256") != EXPECTED_BASE_INTERNAL_CONTRACT_SHA256:
        raise ConstraintAwareSurrogateError("Stage-E internal contract SHA changed")
    if contract.get("selection_sha256") != EXPECTED_BASE_SELECTION_SHA256:
        raise ConstraintAwareSurrogateError("Stage-E selection SHA changed")
    if summary.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise ConstraintAwareSurrogateError("Stage-E root cause changed")
    if summary.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise ConstraintAwareSurrogateError("Stage-E next path changed")
    if summary.get("scientific_status") != "BLOCKED":
        raise ConstraintAwareSurrogateError("Stage-E status changed")
    if summary.get("selected_configuration") is not None:
        raise ConstraintAwareSurrogateError("Stage-E selected a configuration")
    if summary.get("train_only_recommendation") is not None:
        raise ConstraintAwareSurrogateError("Stage-E emitted a recommendation")
    comparison = worker.get("comparison", {})
    if worker.get("workers_exact") is not True or comparison.get("exact") is not True:
        raise ConstraintAwareSurrogateError("Stage-E workers are not exact")
    if comparison.get("left_sha256") != EXPECTED_BASE_WORKER_SHA256:
        raise ConstraintAwareSurrogateError("Stage-E worker SHA changed")
    if comparison.get("right_sha256") != EXPECTED_BASE_WORKER_SHA256:
        raise ConstraintAwareSurrogateError("Stage-E worker SHA mismatch")
    if test_gate.get("verdict") != "PASS":
        raise ConstraintAwareSurrogateError("Stage-E test gate is not PASS")
    if int(test_gate.get("test_file_count", -1)) != 55:
        raise ConstraintAwareSurrogateError("Stage-E test-file count changed")
    if int(test_gate.get("passed_test_count", -1)) != 1324:
        raise ConstraintAwareSurrogateError("Stage-E pass count changed")
    worker_result = worker.get("worker_result", {})
    classification = worker_result.get("classification", {})
    if classification.get("oracle_all_timesteps") is not True:
        raise ConstraintAwareSurrogateError("Stage-E oracle no longer passes")
    if classification.get("eligible_candidate_ids") != []:
        raise ConstraintAwareSurrogateError("Stage-E unexpectedly has eligible candidates")
    forbidden = (
        "frozen_probe_accessed",
        "new_diffusion_model_candidate_trained",
        "reverse_sampling_run",
        "formal_training_run",
        "idm_run",
        "candidate_execution",
        "deformable_ravens_executed",
        "phase4",
        "cps",
    )
    for key in forbidden:
        if worker_result.get(key) is not False:
            raise ConstraintAwareSurrogateError(
                "Stage-E forbidden boundary changed: {}".format(key)
            )
    return {
        "head": head,
        "file_sha256": file_sha,
        "source_sha256": source_sha,
        "contract": contract,
        "worker": worker,
        "summary": summary,
        "test_gate": test_gate,
    }


def definition_by_id(candidate_id: str) -> ConstraintAwareDefinition:
    for definition in CANDIDATE_DEFINITIONS:
        if definition.candidate_id == candidate_id:
            return definition
    raise KeyError(candidate_id)


def _derived_integrator_definition(
    *,
    candidate_id: str,
    direction_source_id: str,
    role: str,
) -> stagee258.IntegratorDefinition:
    oracle = oracle_integrator_definition()
    definition = stagee258.IntegratorDefinition(
        candidate_id=candidate_id,
        direction_source_id=direction_source_id,
        integration_mode=oracle.integration_mode,
        bound_mode=oracle.bound_mode,
        lower_z=oracle.lower_z,
        upper_z=oracle.upper_z,
        retention_min=oracle.retention_min,
        maximum_scale=oracle.maximum_scale,
        role=role,
    )
    definition.validate()
    return definition


def fixed_integrator_definition() -> stagee258.IntegratorDefinition:
    return _derived_integrator_definition(
        candidate_id="stagef_fixed_z4_m2_r25",
        direction_source_id="centered_ridge_a10",
        role="selectable",
    )


def projection_only_definition() -> stagee258.IntegratorDefinition:
    return _derived_integrator_definition(
        candidate_id="stagef_projection_only_z4_m2_r25",
        direction_source_id="zero",
        role="diagnostic_control",
    )


def _condition_parts(condition: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(condition, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != stageb.CONDITION_DIM:
        raise ValueError("condition shape changed")
    history = value[:, : stageb.HISTORY_STEPS * stageb.STATE_DIM].reshape(
        value.shape[0], stageb.HISTORY_STEPS, stageb.STATE_DIM
    )
    cable = history[:, :, : stageb.CABLE_DIM].reshape(
        value.shape[0], stageb.HISTORY_STEPS, stageb.BEADS, 2
    )
    robot = history[:, :, stageb.CABLE_DIM :]
    actions = value[:, stageb.HISTORY_STEPS * stageb.STATE_DIM :]
    return cable, robot, actions


def _control_points(control: np.ndarray) -> np.ndarray:
    value = np.asarray(control, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (stageb.FUTURE_STEPS, stageb.CABLE_DIM):
        raise ValueError("control output shape changed")
    return value.reshape(value.shape[0], stageb.FUTURE_STEPS, stageb.BEADS, 2)


def stable_center_points(value: np.ndarray) -> np.ndarray:
    """Return centroid-relative points after cancelling translation first.

    This is algebraically identical to ``points - mean(points)`` but avoids
    subtracting a large global centroid from every point.  The first bead is
    used only as a numerical origin; the returned coordinates remain centered
    on the full ordered cable centroid.
    """
    points = np.asarray(value, dtype=np.float64)
    if points.shape[-2:] != (stageb.BEADS, 2):
        raise ValueError("cable point shape changed")
    relative = points - points[..., :1, :]
    centered = relative - np.mean(relative, axis=-2, keepdims=True)
    if not np.all(np.isfinite(centered)):
        raise ConstraintAwareSurrogateError("stable centered points are non-finite")
    return centered


def stable_segment_vectors(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape[-2:] != (stageb.BEADS, 2):
        raise ValueError("cable point shape changed")
    result = points[..., 1:, :] - points[..., :-1, :]
    if not np.all(np.isfinite(result)):
        raise ConstraintAwareSurrogateError("stable segment vectors are non-finite")
    return result


def stable_relative_robot_features(
    *,
    cable: np.ndarray,
    robot: np.ndarray,
) -> np.ndarray:
    cable_points = np.asarray(cable, dtype=np.float64)
    robot_value = np.asarray(robot, dtype=np.float64)
    expected = (
        cable_points.shape[0],
        stageb.HISTORY_STEPS,
        staged258.schema_v3.ROBOT_PROXY_DIM,
    )
    if robot_value.shape != expected:
        raise ValueError("robot-proxy shape changed: {}".format(robot_value.shape))
    relative = cable_points - cable_points[:, :, :1, :]
    centroid_from_anchor = (
        cable_points[:, :, 0, :]
        + np.mean(relative, axis=2)
    )
    joint_position = robot_value[
        :, :, staged258.schema_v3.ROBOT_JOINT_POSITION_SLICE
    ]
    joint_velocity = robot_value[
        :, :, staged258.schema_v3.ROBOT_JOINT_VELOCITY_SLICE
    ]
    ee_position = robot_value[
        :, :, staged258.schema_v3.ROBOT_EE_POSITION_SLICE
    ]
    quaternion = robot_value[
        :, :, staged258.schema_v3.ROBOT_EE_QUATERNION_SLICE
    ]
    result = np.concatenate(
        [
            joint_position,
            joint_velocity,
            ee_position[:, :, :2] - centroid_from_anchor,
            ee_position[:, :, 2:3],
            quaternion,
        ],
        axis=2,
    )
    if result.shape != expected:
        raise ConstraintAwareSurrogateError(
            "stable relative robot feature layout changed"
        )
    if not np.all(np.isfinite(result)):
        raise ConstraintAwareSurrogateError(
            "stable relative robot features are non-finite"
        )
    return result.astype(np.float64)


def exact_translate_cable_inputs(
    *,
    condition: np.ndarray,
    control: np.ndarray,
    offset_xy: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate in float64 without introducing a quantization round trip."""
    translated_condition = np.asarray(condition, dtype=np.float64).copy()
    translated_control = np.asarray(control, dtype=np.float64).copy()
    offset = np.asarray(offset_xy, dtype=np.float64)
    if offset.shape != (2,):
        raise ValueError("translation offset shape changed")
    history = translated_condition[
        :, : stageb.HISTORY_STEPS * stageb.STATE_DIM
    ].reshape(
        translated_condition.shape[0],
        stageb.HISTORY_STEPS,
        stageb.STATE_DIM,
    )
    cable = history[:, :, : stageb.CABLE_DIM].reshape(
        translated_condition.shape[0],
        stageb.HISTORY_STEPS,
        stageb.BEADS,
        2,
    )
    cable += offset[None, None, None, :]
    robot = history[:, :, stageb.CABLE_DIM :]
    ee_position = robot[
        :, :, staged258.schema_v3.ROBOT_EE_POSITION_SLICE
    ]
    ee_position[:, :, :2] += offset[None, None, :]
    translated_control.reshape(
        translated_control.shape[0],
        stageb.FUTURE_STEPS,
        stageb.BEADS,
        2,
    )[:] += offset[None, None, None, :]
    return translated_condition, translated_control


def constraint_z_features(
    control: np.ndarray,
    context: Mapping[str, Any],
) -> np.ndarray:
    """Return the frozen Stage-E saturated log-z state.

    This compatibility function intentionally retains the historical Stage-F
    behavior: an exact zero-length segment maps to the lower saturated value.
    Deployable constraint-aware features must call ``constraint_state_features``
    instead, which separates that discrete structural state from continuous z.
    """
    points = _control_points(control)
    lengths = np.linalg.norm(stable_segment_vectors(points), axis=-1)
    reference = context["stage_d_contract"].reference
    reference.validate()
    center = np.asarray(reference.center_log, dtype=np.float64)
    scale = np.asarray(reference.scale_log, dtype=np.float64)
    if center.shape != (stageb.FUTURE_STEPS, stageb.BEADS - 1):
        raise ConstraintAwareSurrogateError("constraint center shape changed")
    if scale.shape != center.shape or np.any(scale <= 0.0):
        raise ConstraintAwareSurrogateError("constraint scale changed")
    lower_log = center - CONSTRAINT_Z_CLIP * scale
    upper_log = center + CONSTRAINT_Z_CLIP * scale
    log_length = np.log(np.maximum(lengths, np.finfo(np.float64).tiny))
    clipped_log = np.minimum(np.maximum(log_length, lower_log[None]), upper_log[None])
    z = (clipped_log - center[None]) / scale[None]
    z = np.clip(z, -CONSTRAINT_Z_CLIP, CONSTRAINT_Z_CLIP)
    if not np.all(np.isfinite(z)):
        raise ConstraintAwareSurrogateError("constraint z features are non-finite")
    if float(np.max(np.abs(z))) > CONSTRAINT_Z_CLIP + 1.0e-12:
        raise ConstraintAwareSurrogateError("constraint z saturation failed")
    return z.astype(np.float64)


def structural_zero_mask(control: np.ndarray) -> np.ndarray:
    """Return the exact discrete mask for duplicated adjacent cable beads.

    The mask uses componentwise equality of adjacent segment vectors.  V4
    established that resolvable segments survive the frozen float32 translation
    round trip, so the discrete state remains separate from continuous z.
    """
    vectors = stable_segment_vectors(_control_points(control))
    mask = np.all(vectors == 0.0, axis=-1)
    expected = (
        np.asarray(control).shape[0],
        stageb.FUTURE_STEPS,
        stageb.BEADS - 1,
    )
    if mask.shape != expected:
        raise ConstraintAwareSurrogateError(
            "structural-zero mask shape changed: {}".format(mask.shape)
        )
    return mask


def constraint_state_features(
    control: np.ndarray,
    context: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Separate continuous resolvable z from exact structural-zero state."""
    raw_z = constraint_z_features(control, context)
    zero_mask_bool = structural_zero_mask(control)
    continuous_z = np.where(
        zero_mask_bool,
        np.float64(STRUCTURAL_ZERO_Z_FILL),
        raw_z,
    ).astype(np.float64)
    zero_mask = zero_mask_bool.astype(np.float64)
    if continuous_z.shape != zero_mask.shape:
        raise ConstraintAwareSurrogateError("constraint-state shape mismatch")
    if not np.all(np.isfinite(continuous_z)):
        raise ConstraintAwareSurrogateError(
            "resolvable constraint z features are non-finite"
        )
    if not np.all((zero_mask == 0.0) | (zero_mask == 1.0)):
        raise ConstraintAwareSurrogateError("structural-zero mask is not binary")
    if np.any(continuous_z[zero_mask_bool] != STRUCTURAL_ZERO_Z_FILL):
        raise ConstraintAwareSurrogateError("structural-zero z fill changed")
    return {
        "constraint_z_resolvable": continuous_z,
        "structural_zero_mask": zero_mask,
    }


def _feature_block_slices(feature_mode: str) -> Tuple[Tuple[str, slice, str], ...]:
    if feature_mode == "anchor":
        return (("anchor", slice(0, 211), "ordinary"),)
    if feature_mode == "full_centered_constraint":
        return (
            ("history_centered", slice(0, 144), "ordinary"),
            ("relative_robot", slice(144, 201), "ordinary"),
            ("actions", slice(201, 243), "ordinary"),
            ("future_centered", slice(243, 435), "ordinary"),
            ("constraint_z_resolvable", slice(435, 527), "constraint_z"),
            (
                "structural_zero_mask",
                slice(527, FULL_CENTERED_CONSTRAINT_DIMENSION),
                "structural_zero_mask",
            ),
        )
    if feature_mode == "full_segment_constraint":
        return (
            ("history_segments", slice(0, 138), "ordinary"),
            ("relative_robot", slice(138, 195), "ordinary"),
            ("actions", slice(195, 237), "ordinary"),
            ("future_segments", slice(237, 421), "ordinary"),
            ("constraint_z_resolvable", slice(421, 513), "constraint_z"),
            ("structural_zero_mask", slice(513, 605), "structural_zero_mask"),
            ("history_centroid_velocity", slice(605, 609), "ordinary"),
            (
                "future_centroid_increment",
                slice(609, FULL_SEGMENT_CONSTRAINT_DIMENSION),
                "ordinary",
            ),
        )
    raise ValueError("unknown feature mode: {}".format(feature_mode))


def _constraint_z_from_lengths(
    *,
    lengths: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    clip: float,
) -> np.ndarray:
    value = np.asarray(lengths, dtype=np.float64)
    lower_log = center - float(clip) * scale
    upper_log = center + float(clip) * scale
    log_length = np.log(np.maximum(value, np.finfo(np.float64).tiny))
    clipped_log = np.minimum(
        np.maximum(log_length, lower_log[None]),
        upper_log[None],
    )
    result = (clipped_log - center[None]) / scale[None]
    return np.clip(result, -float(clip), float(clip)).astype(np.float64)


def _worst_ulp_formula_records(
    *,
    resolvable: np.ndarray,
    source_lengths: np.ndarray,
    rounded_lengths: np.ndarray,
    scale: np.ndarray,
    source_z: np.ndarray,
    rounded_z: np.ndarray,
    observed_z_difference: np.ndarray,
    observed_length_error: np.ndarray,
    full_endpoint_ulp_norm: np.ndarray,
    half_endpoint_ulp_norm: np.ndarray,
    safety_length_error: np.ndarray,
    safety_relative_error: np.ndarray,
    clipping_aware_bound: np.ndarray,
    limit: float,
    count: int = 24,
) -> List[Dict[str, Any]]:
    score = np.maximum(
        clipping_aware_bound / float(limit),
        observed_z_difference / float(limit),
    )
    score = np.where(resolvable, score, -np.inf)
    flat_order = np.argsort(score.reshape(-1))[::-1]
    records: List[Dict[str, Any]] = []
    for flat_index in flat_order:
        if len(records) >= int(count):
            break
        index = tuple(int(value) for value in np.unravel_index(flat_index, score.shape))
        if not bool(resolvable[index]):
            continue
        horizon = index[1]
        segment = index[2]
        records.append(
            {
                "row": index[0],
                "horizon_index": horizon,
                "segment_index": segment,
                "source_length": float(source_lengths[index]),
                "rounded_length": float(rounded_lengths[index]),
                "scale_log": float(scale[horizon, segment]),
                "source_z": float(source_z[index]),
                "rounded_z": float(rounded_z[index]),
                "observed_z_difference": float(observed_z_difference[index]),
                "observed_length_error": float(observed_length_error[index]),
                "full_endpoint_ulp_norm": float(full_endpoint_ulp_norm[index]),
                "half_endpoint_ulp_norm": float(half_endpoint_ulp_norm[index]),
                "safety_length_error": float(safety_length_error[index]),
                "safety_relative_error": float(safety_relative_error[index]),
                "clipping_aware_z_bound": float(clipping_aware_bound[index]),
                "source_lower_clipped": bool(source_z[index] <= -CONSTRAINT_Z_CLIP),
                "source_upper_clipped": bool(source_z[index] >= CONSTRAINT_Z_CLIP),
            }
        )
    return records


def _float32_constraint_z_bound(
    *,
    control: np.ndarray,
    offset_xy: Sequence[float],
    context: Mapping[str, Any],
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    """Apply a segmentwise, clipping-aware float32 ULP propagation bound.

    V5 separated exact structural zeros, but its admission implementation still
    multiplied the *observed* z drift by the frozen ULP safety factor.  That is
    not an ULP error-propagation formula.  This function applies the unchanged
    factor to the round-to-nearest endpoint uncertainty, propagates the length
    interval through the exact clipped log-z transform, and checks the existing
    1e-2 maximum.  No tolerance, split, candidate, or holdout policy changes.
    """
    control_value = np.asarray(control)
    source_points = _control_points(control_value)
    offset = np.asarray(offset_xy, dtype=np.float64)
    if offset.shape != (2,):
        raise ValueError("translation offset shape changed")
    exact_translated = source_points + offset[None, None, None, :]
    rounded_float32 = exact_translated.astype(np.float32)
    rounded_points = rounded_float32.astype(np.float64)
    rounded_control = rounded_points.reshape(
        control_value.shape[0],
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    )

    source_state = constraint_state_features(control_value, context)
    rounded_state = constraint_state_features(rounded_control, context)
    source_mask = source_state["structural_zero_mask"].astype(np.bool_)
    rounded_mask = rounded_state["structural_zero_mask"].astype(np.bool_)
    if not np.array_equal(source_mask, rounded_mask):
        changed = int(np.count_nonzero(source_mask != rounded_mask))
        raise ConstraintAwareSurrogateError(
            "structural-zero mask changed under float32 translation: {}".format(
                changed
            )
        )

    resolvable = ~source_mask
    resolvable_count = int(np.count_nonzero(resolvable))
    structural_zero_count = int(np.count_nonzero(source_mask))
    if resolvable_count <= 0:
        raise ConstraintAwareSurrogateError(
            "constraint-z audit has no resolvable segments"
        )

    source_vectors = stable_segment_vectors(source_points)
    rounded_vectors = stable_segment_vectors(rounded_points)
    source_lengths = np.linalg.norm(source_vectors, axis=-1)
    rounded_lengths = np.linalg.norm(rounded_vectors, axis=-1)
    cast_collapse = resolvable & (rounded_lengths == 0.0)
    if np.any(cast_collapse):
        raise ConstraintAwareSurrogateError(
            "float32 translation collapsed a resolvable segment"
        )

    reference = context["stage_d_contract"].reference
    reference.validate()
    center = np.asarray(reference.center_log, dtype=np.float64)
    scale = np.asarray(reference.scale_log, dtype=np.float64)
    if center.shape != (stageb.FUTURE_STEPS, stageb.BEADS - 1):
        raise ConstraintAwareSurrogateError("constraint center shape changed")
    if scale.shape != center.shape or np.any(scale <= 0.0):
        raise ConstraintAwareSurrogateError("constraint scale changed")

    source_z = source_state["constraint_z_resolvable"]
    rounded_z = rounded_state["constraint_z_resolvable"]
    observed_z_difference = np.abs(source_z - rounded_z)
    maximum_observed = float(np.max(observed_z_difference[resolvable]))
    mean_observed = float(np.mean(observed_z_difference[resolvable]))

    # IEEE round-to-nearest error is at most one half ULP per endpoint
    # coordinate.  Summing the two endpoint coordinate bounds and taking the
    # Euclidean norm gives a segment-length perturbation bound.  The historical
    # factor=8 is then applied unchanged to that uncertainty, not to observed z.
    spacing = np.abs(np.spacing(rounded_float32)).astype(np.float64)
    full_endpoint_ulp_vector = spacing[..., 1:, :] + spacing[..., :-1, :]
    full_endpoint_ulp_norm = np.linalg.norm(full_endpoint_ulp_vector, axis=-1)
    half_endpoint_ulp_norm = 0.5 * full_endpoint_ulp_norm
    observed_length_error = np.abs(rounded_lengths - source_lengths)
    length_error_excess = observed_length_error - half_endpoint_ulp_norm
    numerical_slack = 64.0 * np.finfo(np.float64).eps
    maximum_length_error_excess = float(
        np.max(length_error_excess[resolvable])
    )
    rounding_model_covers_length = bool(
        maximum_length_error_excess <= numerical_slack
    )
    safety_length_error = (
        float(spec.translation_float32_z_ulp_factor)
        * half_endpoint_ulp_norm
    )
    safety_relative_error = np.zeros_like(source_lengths, dtype=np.float64)
    safety_relative_error[resolvable] = (
        safety_length_error[resolvable] / source_lengths[resolvable]
    )

    lower_lengths = np.maximum(
        source_lengths - safety_length_error,
        np.finfo(np.float64).tiny,
    )
    upper_lengths = source_lengths + safety_length_error
    lower_z = _constraint_z_from_lengths(
        lengths=lower_lengths,
        center=center,
        scale=scale,
        clip=spec.constraint_z_clip,
    )
    upper_z = _constraint_z_from_lengths(
        lengths=upper_lengths,
        center=center,
        scale=scale,
        clip=spec.constraint_z_clip,
    )
    clipping_aware_bound = np.maximum(
        np.abs(lower_z - source_z),
        np.abs(upper_z - source_z),
    )
    clipping_aware_bound = np.where(resolvable, clipping_aware_bound, 0.0)
    maximum_formula_bound = float(np.max(clipping_aware_bound[resolvable]))
    mean_formula_bound = float(np.mean(clipping_aware_bound[resolvable]))

    observed_excess = observed_z_difference - clipping_aware_bound
    maximum_observed_excess = float(np.max(observed_excess[resolvable]))
    formula_covers_observed = bool(maximum_observed_excess <= numerical_slack)

    lower_length = np.exp(center - spec.constraint_z_clip * scale)
    minimum_lower_length = float(np.min(lower_length))
    minimum_scale = float(np.min(scale))
    maximum_spacing = float(np.max(spacing))
    historical_global_bound = (
        float(spec.translation_float32_z_ulp_factor)
        * math.sqrt(2.0)
        * maximum_spacing
        / (minimum_lower_length * minimum_scale)
    )
    v5_empirical_bound = (
        float(spec.translation_float32_z_ulp_factor) * maximum_observed
    )
    applied_bound = max(
        float(spec.translation_tolerance),
        float(maximum_formula_bound),
    )
    maximum_allowed = float(spec.translation_float32_z_bound_max)

    diagnostic: Dict[str, Any] = {
        "schema": "phase314b_r258_stagef_constraint_z_ulp_formula_v1",
        "admission_formula": (
            "segmentwise_round_to_nearest_endpoint_half_ulp_"
            "times_frozen_factor_then_exact_clipped_log_z_interval"
        ),
        "structural_zero_policy": "exact_adjacent_bead_equality",
        "structural_zero_count": structural_zero_count,
        "resolvable_segment_count": resolvable_count,
        "mask_roundtrip_exact": True,
        "float32_cast_collapse_count": 0,
        "frozen_parameters": {
            "ulp_factor": float(spec.translation_float32_z_ulp_factor),
            "ordinary_translation_tolerance": float(spec.translation_tolerance),
            "maximum_allowed_bound": maximum_allowed,
            "constraint_z_clip": float(spec.constraint_z_clip),
        },
        "coordinate_and_geometry": {
            "maximum_coordinate_spacing": maximum_spacing,
            "minimum_clipped_segment_length": minimum_lower_length,
            "minimum_resolvable_segment_length": float(
                np.min(source_lengths[resolvable])
            ),
            "minimum_log_scale": minimum_scale,
            "maximum_full_endpoint_ulp_norm_over_source_length": float(
                np.max(
                    full_endpoint_ulp_norm[resolvable]
                    / source_lengths[resolvable]
                )
            ),
            "maximum_half_endpoint_ulp_norm_over_source_length": float(
                np.max(
                    half_endpoint_ulp_norm[resolvable]
                    / source_lengths[resolvable]
                )
            ),
            "maximum_safety_relative_length_error": float(
                np.max(safety_relative_error[resolvable])
            ),
        },
        "observed_roundtrip": {
            "maximum_resolvable_length_difference": float(
                np.max(observed_length_error[resolvable])
            ),
            "maximum_length_error_minus_half_ulp_bound": (
                maximum_length_error_excess
            ),
            "round_to_nearest_half_ulp_model_covers_length": (
                rounding_model_covers_length
            ),
            "maximum_resolvable_z_difference": maximum_observed,
            "mean_resolvable_z_difference": mean_observed,
        },
        "formula_result": {
            "maximum_clipping_aware_z_bound": maximum_formula_bound,
            "mean_clipping_aware_z_bound": mean_formula_bound,
            "applied_bound": float(applied_bound),
            "maximum_allowed_bound": maximum_allowed,
            "formula_covers_observed": formula_covers_observed,
            "maximum_observed_minus_formula_bound": maximum_observed_excess,
        },
        "comparison_with_prior_implementations": {
            "historical_global_extrema_bound": float(historical_global_bound),
            "v5_empirical_drift_times_factor_bound": float(v5_empirical_bound),
            "historical_global_bound_pass": bool(
                historical_global_bound <= maximum_allowed
            ),
            "v5_empirical_bound_pass": bool(
                max(float(spec.translation_tolerance), v5_empirical_bound)
                <= maximum_allowed
            ),
            "segmentwise_clipping_aware_bound_pass": bool(
                applied_bound <= maximum_allowed
            ),
        },
        "worst_segments": _worst_ulp_formula_records(
            resolvable=resolvable,
            source_lengths=source_lengths,
            rounded_lengths=rounded_lengths,
            scale=scale,
            source_z=source_z,
            rounded_z=rounded_z,
            observed_z_difference=observed_z_difference,
            observed_length_error=observed_length_error,
            full_endpoint_ulp_norm=full_endpoint_ulp_norm,
            half_endpoint_ulp_norm=half_endpoint_ulp_norm,
            safety_length_error=safety_length_error,
            safety_relative_error=safety_relative_error,
            clipping_aware_bound=clipping_aware_bound,
            limit=maximum_allowed,
        ),
        "policy": {
            "exact_gate_relaxed": False,
            "ulp_factor_changed": False,
            "maximum_allowed_bound_changed": False,
            "holdout_used": False,
            "frozen_probe_used": False,
            "automatic_tolerance_inferred": False,
        },
    }

    if not rounding_model_covers_length:
        raise ConstraintZULPAdmissionError(
            "observed segment-length drift exceeds the half-ULP rounding model",
            diagnostic=diagnostic,
            required_next_path=(
                "AUDIT_FLOAT32_ENDPOINT_ROUNDING_MODEL_BEFORE_ANY_BOUND_CHANGE"
            ),
        )
    if not formula_covers_observed:
        raise ConstraintZULPAdmissionError(
            "observed float32 z drift exceeds the segmentwise ULP formula",
            diagnostic=diagnostic,
            required_next_path=(
                "AUDIT_FLOAT32_ROUNDING_ERROR_MODEL_BEFORE_ANY_BOUND_CHANGE"
            ),
        )
    if applied_bound > maximum_allowed:
        if maximum_observed <= maximum_allowed:
            next_path = (
                "REVIEW_FROZEN_ULP_FACTOR_AND_BOUND_MAX_WITH_"
                "PREDECLARED_OBJECTIVE_TRAIN_ONLY_POLICY"
            )
        else:
            next_path = (
                "REDESIGN_CONSTRAINT_Z_TRANSLATION_NUMERICS_ON_"
                "OBJECTIVE_TRAIN_ONLY"
            )
        raise ConstraintZULPAdmissionError(
            "segmentwise clipping-aware ULP bound exceeds frozen maximum",
            diagnostic=diagnostic,
            required_next_path=next_path,
        )

    return {
        "structural_zero_policy": "exact_adjacent_bead_equality",
        "structural_zero_count": structural_zero_count,
        "resolvable_segment_count": resolvable_count,
        "mask_roundtrip_exact": True,
        "float32_cast_collapse_count": 0,
        "maximum_coordinate_spacing": maximum_spacing,
        "minimum_clipped_segment_length": minimum_lower_length,
        "minimum_resolvable_segment_length": float(
            np.min(source_lengths[resolvable])
        ),
        "maximum_endpoint_ulp_norm_over_source_length": float(
            np.max(
                full_endpoint_ulp_norm[resolvable]
                / source_lengths[resolvable]
            )
        ),
        "maximum_observed_resolvable_z_difference": maximum_observed,
        "mean_observed_resolvable_z_difference": mean_observed,
        "ulp_factor": float(spec.translation_float32_z_ulp_factor),
        "derived_bound": float(maximum_formula_bound),
        "applied_bound": float(applied_bound),
        "maximum_allowed_bound": maximum_allowed,
        "admission_formula": diagnostic["admission_formula"],
        "formula_covers_observed": True,
        "formula_diagnostic": diagnostic,
    }


def build_constraint_features(
    *,
    condition: np.ndarray,
    control: np.ndarray,
    condition_name: Sequence[Any],
    feature_mode: str,
    context: Mapping[str, Any],
) -> np.ndarray:
    if feature_mode == "none":
        return np.zeros((np.asarray(condition).shape[0], 0), dtype=np.float64)
    if feature_mode == "condition_only":
        return staged258.condition_one_hot(condition_name)
    if feature_mode == "anchor":
        return staged258.build_surrogate_features(
            condition=condition,
            control=control,
            condition_name=condition_name,
            feature_mode="centered_anchor",
        )
    cable, robot, actions = _condition_parts(condition)
    future = _control_points(control)
    relative_robot = stable_relative_robot_features(cable=cable, robot=robot)
    constraint_state = constraint_state_features(control, context)
    z = constraint_state["constraint_z_resolvable"]
    structural_zero = constraint_state["structural_zero_mask"]
    if feature_mode == "full_centered_constraint":
        parts = (
            stable_center_points(cable).reshape(cable.shape[0], -1),
            relative_robot.reshape(cable.shape[0], -1),
            actions,
            stable_center_points(future).reshape(cable.shape[0], -1),
            z.reshape(cable.shape[0], -1),
            structural_zero.reshape(cable.shape[0], -1),
        )
        expected_dimension = FULL_CENTERED_CONSTRAINT_DIMENSION
    elif feature_mode == "full_segment_constraint":
        history_centroid = np.mean(cable, axis=2)
        future_centroid = np.mean(future, axis=2)
        history_velocity = np.diff(history_centroid, axis=1)
        future_chain = np.concatenate(
            [history_centroid[:, -1:, :], future_centroid], axis=1
        )
        future_increment = np.diff(future_chain, axis=1)
        parts = (
            stable_segment_vectors(cable).reshape(cable.shape[0], -1),
            relative_robot.reshape(cable.shape[0], -1),
            actions,
            stable_segment_vectors(future).reshape(cable.shape[0], -1),
            z.reshape(cable.shape[0], -1),
            structural_zero.reshape(cable.shape[0], -1),
            history_velocity.reshape(cable.shape[0], -1),
            future_increment.reshape(cable.shape[0], -1),
        )
        expected_dimension = FULL_SEGMENT_CONSTRAINT_DIMENSION
    else:
        raise ValueError("unknown constraint-aware feature mode: {}".format(feature_mode))
    result = np.concatenate(parts, axis=1).astype(np.float64)
    if result.shape[1] != expected_dimension:
        raise ConstraintAwareSurrogateError(
            "constraint-aware feature dimension changed: {}".format(result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise ConstraintAwareSurrogateError("constraint-aware features are non-finite")
    return result


def translation_invariance_audit(
    *,
    condition: np.ndarray,
    control: np.ndarray,
    condition_name: Sequence[Any],
    context: Mapping[str, Any],
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    """Separate structural invariance from float32 round-trip sensitivity."""
    spec.validate()
    exact_condition, exact_control = exact_translate_cable_inputs(
        condition=condition,
        control=control,
        offset_xy=spec.translation_offset_xy,
    )
    rounded_condition, rounded_control = staged258.translate_cable_inputs(
        condition=condition,
        control=control,
        offset_xy=spec.translation_offset_xy,
    )
    z_bound = _float32_constraint_z_bound(
        control=control,
        offset_xy=spec.translation_offset_xy,
        context=context,
        spec=spec,
    )
    records: Dict[str, Any] = {}
    for mode in ("anchor", "full_centered_constraint", "full_segment_constraint"):
        original = build_constraint_features(
            condition=condition,
            control=control,
            condition_name=condition_name,
            feature_mode=mode,
            context=context,
        )
        exact_translated = build_constraint_features(
            condition=exact_condition,
            control=exact_control,
            condition_name=condition_name,
            feature_mode=mode,
            context=context,
        )
        rounded_translated = build_constraint_features(
            condition=rounded_condition,
            control=rounded_control,
            condition_name=condition_name,
            feature_mode=mode,
            context=context,
        )
        structural_difference = np.abs(original - exact_translated)
        structural_maximum = float(np.max(structural_difference))
        if structural_maximum > spec.translation_structural_tolerance:
            raise ConstraintAwareSurrogateError(
                "feature mode is structurally translation variant: {}".format(mode)
            )
        roundtrip_difference = np.abs(original - rounded_translated)
        block_records = {}
        for name, block_slice, block_kind in _feature_block_slices(mode):
            values = roundtrip_difference[:, block_slice]
            maximum = float(np.max(values)) if values.size else 0.0
            mean = float(np.mean(values)) if values.size else 0.0
            if block_kind == "constraint_z":
                tolerance = float(z_bound["applied_bound"])
            elif block_kind == "structural_zero_mask":
                tolerance = 0.0
                if not np.array_equal(
                    original[:, block_slice], rounded_translated[:, block_slice]
                ):
                    raise ConstraintAwareSurrogateError(
                        "structural-zero mask is not float32 translation invariant: {}".format(
                            mode
                        )
                    )
            else:
                tolerance = float(spec.translation_tolerance)
            if maximum > tolerance:
                raise ConstraintAwareSurrogateError(
                    "float32 translation drift exceeds {} block envelope: {}".format(
                        name, mode
                    )
                )
            block_records[name] = {
                "kind": block_kind,
                "start": int(block_slice.start),
                "stop": int(block_slice.stop),
                "maximum_absolute_difference": maximum,
                "mean_absolute_difference": mean,
                "tolerance": tolerance,
                "pass": True,
            }
        records[mode] = {
            "structural": {
                "maximum_absolute_difference": structural_maximum,
                "mean_absolute_difference": float(np.mean(structural_difference)),
                "tolerance": float(spec.translation_structural_tolerance),
                "pass": True,
            },
            "float32_roundtrip": {
                "maximum_absolute_difference": float(np.max(roundtrip_difference)),
                "mean_absolute_difference": float(np.mean(roundtrip_difference)),
                "blocks": block_records,
                "pass": True,
            },
            "original_sha256": sha256_array(original),
            "exact_translated_sha256": sha256_array(exact_translated),
            "rounded_translated_sha256": sha256_array(rounded_translated),
        }
    return {
        "offset_xy": list(spec.translation_offset_xy),
        "structural_tolerance": float(spec.translation_structural_tolerance),
        "ordinary_float32_tolerance": float(spec.translation_tolerance),
        "constraint_z_clip": float(spec.constraint_z_clip),
        "constraint_z_quantization_envelope": z_bound,
        "records": records,
        "all_selectable_modes_invariant": True,
        "audit_interpretation": (
            "structural symmetry and the exact structural-zero mask are hard "
            "gates; float32 drift is bounded only on resolvable continuous z"
        ),
    }


def encode_projected_target(
    *,
    control: np.ndarray,
    oracle_candidate: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    control_raw = np.asarray(control, dtype=np.float64)
    candidate_raw = np.asarray(oracle_candidate, dtype=np.float64)
    if control_raw.shape != candidate_raw.shape:
        raise ValueError("projected target shape mismatch")
    if target_mode == "projected_point":
        return (candidate_raw - control_raw).reshape(control_raw.shape[0], -1)
    if target_mode == "projected_segment":
        control_points = _control_points(control_raw)
        candidate_points = _control_points(candidate_raw)
        centroid_delta = np.mean(candidate_points, axis=2) - np.mean(control_points, axis=2)
        segment_delta = staged258.segment_vectors(candidate_points) - staged258.segment_vectors(
            control_points
        )
        result = np.concatenate(
            [centroid_delta.reshape(control_raw.shape[0], -1), segment_delta.reshape(control_raw.shape[0], -1)],
            axis=1,
        )
        if result.shape[1] != stageb.FUTURE_STEPS * stageb.CABLE_DIM:
            raise ConstraintAwareSurrogateError("segment target dimension changed")
        return result.astype(np.float64)
    raise ValueError("target mode cannot be encoded: {}".format(target_mode))


def decode_projected_prediction(
    *,
    control: np.ndarray,
    encoded: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    control_raw = np.asarray(control, dtype=np.float64)
    flat = np.asarray(encoded, dtype=np.float64)
    rows = control_raw.shape[0]
    if flat.shape != (rows, stageb.FUTURE_STEPS * stageb.CABLE_DIM):
        raise ValueError("encoded prediction shape changed")
    if target_mode == "projected_point":
        return flat.reshape(control_raw.shape).astype(np.float64)
    if target_mode == "projected_segment":
        control_points = _control_points(control_raw)
        centroid_count = stageb.FUTURE_STEPS * 2
        centroid_delta = flat[:, :centroid_count].reshape(rows, stageb.FUTURE_STEPS, 2)
        segment_delta = flat[:, centroid_count:].reshape(
            rows, stageb.FUTURE_STEPS, stageb.BEADS - 1, 2
        )
        segment_vectors = staged258.segment_vectors(control_points) + segment_delta
        reconstructed = np.zeros_like(control_points)
        reconstructed[:, :, 1:, :] = np.cumsum(segment_vectors, axis=2)
        target_centroid = np.mean(control_points, axis=2) + centroid_delta
        reconstructed += (
            target_centroid - np.mean(reconstructed, axis=2)
        )[:, :, None, :]
        return (reconstructed - control_points).reshape(control_raw.shape).astype(np.float64)
    raise ValueError("target mode cannot be decoded: {}".format(target_mode))


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def oracle_integrator_definition() -> stagee258.IntegratorDefinition:
    """Resolve the fixed Stage-E integrator from its typed registry."""
    matches = [
        definition
        for definition in stagee258.INTEGRATOR_DEFINITIONS
        if definition.candidate_id == ORACLE_INTEGRATOR_ID
    ]
    if len(matches) != 1:
        raise ConstraintAwareSurrogateError(
            "oracle integrator definition is not unique"
        )
    definition = matches[0]
    definition.validate()
    return definition


def generate_projected_oracle_target(
    *,
    control: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    timestep: int,
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    oracle_direction = np.asarray(target, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    oracle_definition = oracle_integrator_definition()
    integration = stagee258.integrate_rowwise(
        control=control,
        direction=oracle_direction,
        definition=oracle_definition,
        context=context,
        spec=integrator_spec,
    )
    candidate = np.asarray(integration["candidate"], dtype=np.float32)
    projected_delta = candidate.astype(np.float64) - np.asarray(control, dtype=np.float64)
    feasible = np.asarray(integration["selected_scale"], dtype=np.float64) > 0.0
    evaluated = stagee258.evaluate_integrated_candidate(
        integration=integration,
        control=control,
        target=target,
        groups=groups,
        condition_name=condition_name,
        timestep=int(timestep),
        context=context,
        direction_spec=direction_spec,
        integrator_spec=integrator_spec,
    )
    if integration["observable_feasible_rate"] < spec.projected_oracle_feasible_min:
        raise ConstraintAwareSurrogateError("projected-oracle feasible rate fell below contract")
    if not evaluated["scientific_pass"]:
        raise ConstraintAwareSurrogateError("projected-oracle target is not scientific-valid")
    return {
        "candidate": candidate,
        "projected_delta": projected_delta,
        "feasible_mask": feasible,
        "candidate_sha256": sha256_array(candidate),
        "projected_delta_sha256": sha256_array(projected_delta),
        "feasible_mask_sha256": sha256_array(feasible),
        "feasible_rate": float(np.mean(feasible)),
        "projected_norm": _safe_stats(
            np.linalg.norm(projected_delta.reshape(projected_delta.shape[0], -1), axis=1)
        ),
        "integration": {key: value for key, value in integration.items() if key != "candidate"},
        "evaluation": evaluated,
        "target_used_for_supervision": True,
        "target_used_for_candidate_selection": False,
    }


def _ridge_solve(x: np.ndarray, y: np.ndarray, *, alpha: float, jitter: float) -> np.ndarray:
    design = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    gram = design.T @ design
    rhs = design.T @ target
    coefficient = np.linalg.solve(
        gram + (float(alpha) + float(jitter)) * np.eye(gram.shape[0], dtype=np.float64),
        rhs,
    )
    if not np.all(np.isfinite(coefficient)):
        raise ConstraintAwareSurrogateError("ridge coefficient is non-finite")
    return coefficient.astype(np.float64)


def _candidate_seed(candidate_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**31 - 1))


def fit_constraint_model(
    *,
    definition: ConstraintAwareDefinition,
    features: np.ndarray,
    encoded_target: np.ndarray,
    condition_name: Sequence[Any],
    fit_mask: np.ndarray,
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    definition.validate()
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(encoded_target, dtype=np.float64)
    names = np.asarray(condition_name).astype(str)
    mask = np.asarray(fit_mask, dtype=np.bool_)
    if x.shape[0] != y.shape[0] or names.shape[0] != y.shape[0] or mask.shape != (y.shape[0],):
        raise ValueError("constraint model row population changed")
    if int(np.sum(mask)) < 32:
        raise ConstraintAwareSurrogateError("constraint model fit population is too small")
    x_fit = x[mask]
    y_fit = y[mask]
    names_fit = names[mask]
    global_mean = np.mean(y_fit, axis=0).astype(np.float64)
    if definition.model_mode == "global_mean":
        model = {"mode": "global_mean", "global_mean": global_mean}
    elif definition.model_mode == "condition_mean":
        condition_means = {}
        for name in CONDITION_VALUES:
            class_mask = names_fit == name
            if not np.any(class_mask):
                raise ConstraintAwareSurrogateError("condition mean has an empty class")
            condition_means[name] = np.mean(y_fit[class_mask], axis=0).astype(np.float64)
        model = {
            "mode": "condition_mean",
            "global_mean": global_mean,
            "condition_means": condition_means,
        }
    else:
        standardizer = staged258.fit_feature_standardizer(
            x_fit, epsilon=spec.standardizer_epsilon
        )
        xz = staged258.apply_feature_standardizer(x_fit, standardizer)
        y_centered = y_fit - global_mean[None]
        if definition.model_mode == "reduced_rank_ridge":
            _, singular, vt = np.linalg.svd(y_centered, full_matrices=False)
            rank = min(int(definition.output_rank), int(vt.shape[0]))
            basis = vt[:rank].astype(np.float64)
            scores = y_centered @ basis.T
            coefficient = _ridge_solve(
                xz,
                scores,
                alpha=definition.ridge_alpha,
                jitter=spec.ridge_jitter,
            )
            model = {
                "mode": "reduced_rank_ridge",
                "global_mean": global_mean,
                "feature_standardizer": standardizer,
                "basis": basis,
                "coefficient": coefficient,
                "singular_values": singular.astype(np.float64),
            }
        elif definition.model_mode == "rff_ridge":
            seed = _candidate_seed(definition.candidate_id, spec.rff_seed)
            random = np.random.RandomState(seed)
            input_dimension = int(xz.shape[1])
            random_weight = random.normal(
                size=(input_dimension, int(definition.rff_dimension))
            ).astype(np.float64)
            random_phase = random.uniform(
                0.0,
                2.0 * np.pi,
                size=(int(definition.rff_dimension),),
            ).astype(np.float64)
            gamma = 1.0 / float(max(input_dimension, 1))
            phi = math.sqrt(2.0 / float(definition.rff_dimension)) * np.cos(
                math.sqrt(2.0 * gamma) * (xz @ random_weight) + random_phase[None]
            )
            coefficient = _ridge_solve(
                phi,
                y_centered,
                alpha=definition.ridge_alpha,
                jitter=spec.ridge_jitter,
            )
            model = {
                "mode": "rff_ridge",
                "global_mean": global_mean,
                "feature_standardizer": standardizer,
                "random_weight": random_weight,
                "random_phase": random_phase,
                "gamma": float(gamma),
                "coefficient": coefficient,
                "seed": int(seed),
            }
        else:
            raise ValueError("model mode cannot be fitted")
    identity = {
        "definition": asdict(definition),
        "fit_row_count": int(np.sum(mask)),
        "fit_mask_sha256": sha256_array(mask),
        "global_mean_sha256": sha256_array(global_mean),
    }
    if "feature_standardizer" in model:
        identity.update(
            {
                "feature_mean_sha256": sha256_array(model["feature_standardizer"]["mean"]),
                "feature_scale_sha256": sha256_array(model["feature_standardizer"]["scale"]),
                "feature_active_sha256": sha256_array(model["feature_standardizer"]["active"]),
            }
        )
    for key in ("basis", "coefficient", "singular_values", "random_weight", "random_phase"):
        if key in model:
            identity["{}_sha256".format(key)] = sha256_array(model[key])
    if "gamma" in model:
        identity["gamma"] = float(model["gamma"])
        identity["seed"] = int(model["seed"])
    identity["model_sha256"] = sha256_bytes(stable_json_bytes(identity))
    model["identity"] = identity
    return model


def predict_constraint_model(
    *,
    definition: ConstraintAwareDefinition,
    model: Mapping[str, Any],
    features: np.ndarray,
    condition_name: Sequence[Any],
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    names = np.asarray(condition_name).astype(str)
    rows = x.shape[0]
    mode = model["mode"]
    if mode == "global_mean":
        result = np.repeat(np.asarray(model["global_mean"])[None], rows, axis=0)
    elif mode == "condition_mean":
        result = np.empty((rows, np.asarray(model["global_mean"]).shape[0]), dtype=np.float64)
        for index, name in enumerate(names):
            result[index] = model["condition_means"].get(name, model["global_mean"])
    else:
        xz = staged258.apply_feature_standardizer(x, model["feature_standardizer"])
        if mode == "reduced_rank_ridge":
            centered = xz @ np.asarray(model["coefficient"], dtype=np.float64)
            centered = centered @ np.asarray(model["basis"], dtype=np.float64)
        elif mode == "rff_ridge":
            phi = math.sqrt(2.0 / float(definition.rff_dimension)) * np.cos(
                math.sqrt(2.0 * float(model["gamma"]))
                * (xz @ np.asarray(model["random_weight"], dtype=np.float64))
                + np.asarray(model["random_phase"], dtype=np.float64)[None]
            )
            centered = phi @ np.asarray(model["coefficient"], dtype=np.float64)
        else:
            raise ValueError("unknown fitted model mode")
        result = np.asarray(model["global_mean"], dtype=np.float64)[None] + centered
    if not np.all(np.isfinite(result)):
        raise ConstraintAwareSurrogateError("constraint-aware prediction is non-finite")
    return result.astype(np.float64)


def _raw_reference_definition() -> staged258.SurrogateDefinition:
    for definition in staged258.SURROGATE_DEFINITIONS:
        if definition.candidate_id == "centered_ridge_a10":
            return definition
    raise KeyError("centered_ridge_a10")


def fit_oof_candidate(
    *,
    definition: ConstraintAwareDefinition,
    features: np.ndarray,
    control: np.ndarray,
    raw_target: np.ndarray,
    oracle_target: Mapping[str, Any],
    condition_name: Sequence[Any],
    fold_assignment: np.ndarray,
    spec: ConstraintAwareSpec,
    direction_spec: staged258.DirectionSurrogateSpec,
    permutation_seed: Optional[int] = None,
) -> Dict[str, Any]:
    definition.validate()
    x = np.asarray(features, dtype=np.float64)
    control_raw = np.asarray(control, dtype=np.float32)
    target_raw = np.asarray(raw_target, dtype=np.float32)
    names = np.asarray(condition_name).astype(str)
    assignment = np.asarray(fold_assignment, dtype=np.int64)
    oracle_candidate = np.asarray(oracle_target["candidate"], dtype=np.float32)
    oracle_delta = np.asarray(oracle_target["projected_delta"], dtype=np.float64)
    oracle_feasible = np.asarray(oracle_target["feasible_mask"], dtype=np.bool_)
    rows = control_raw.shape[0]
    prediction = np.empty_like(oracle_delta, dtype=np.float64)
    covered = np.zeros(rows, dtype=np.bool_)
    fold_records = []
    encoded_full = None
    if definition.target_mode in ("projected_point", "projected_segment"):
        encoded_full = encode_projected_target(
            control=control_raw,
            oracle_candidate=oracle_candidate,
            target_mode=definition.target_mode,
        )
    raw_residual = target_raw.astype(np.float64) - control_raw.astype(np.float64)
    for fold in range(spec.grouped_cv_folds):
        test_mask = assignment == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise ConstraintAwareSurrogateError("OOF fold population is empty")
        fit_mask = train_mask.copy()
        if definition.fit_population == "oracle_feasible":
            fit_mask &= oracle_feasible
        if definition.model_mode == "oracle":
            fold_prediction = oracle_delta[test_mask]
            model_identity = {
                "mode": "oracle",
                "target_used_for_prediction": True,
            }
        elif definition.model_mode == "raw_reference":
            raw_definition = _raw_reference_definition()
            raw_features = x
            model = staged258.fit_surrogate_model(
                definition=raw_definition,
                features=raw_features[train_mask],
                target_residual=raw_residual[train_mask],
                condition_name=names[train_mask],
                spec=direction_spec,
            )
            fold_prediction = staged258.predict_surrogate_model(
                definition=raw_definition,
                model=model,
                features=raw_features[test_mask],
                condition_name=names[test_mask],
                target_shape=tuple(control_raw.shape[1:]),
            )
            model_identity = copy.deepcopy(model["identity"])
        else:
            if encoded_full is None:
                raise ConstraintAwareSurrogateError("encoded projected target is missing")
            training_target = encoded_full.copy()
            if permutation_seed is not None:
                fit_indices = np.flatnonzero(fit_mask)
                random = np.random.RandomState(
                    int(permutation_seed) + 1009 * int(fold)
                )
                permuted = fit_indices[random.permutation(fit_indices.shape[0])]
                training_target[fit_indices] = training_target[permuted]
            model = fit_constraint_model(
                definition=definition,
                features=x,
                encoded_target=training_target,
                condition_name=names,
                fit_mask=fit_mask,
                spec=spec,
            )
            encoded_prediction = predict_constraint_model(
                definition=definition,
                model=model,
                features=x[test_mask],
                condition_name=names[test_mask],
            )
            fold_prediction = decode_projected_prediction(
                control=control_raw[test_mask],
                encoded=encoded_prediction,
                target_mode=definition.target_mode,
            )
            model_identity = copy.deepcopy(model["identity"])
        prediction[test_mask] = fold_prediction
        covered[test_mask] = True
        fold_records.append(
            {
                "fold": int(fold),
                "train_rows": int(np.sum(train_mask)),
                "fit_rows": int(np.sum(fit_mask)),
                "test_rows": int(np.sum(test_mask)),
                "train_mask_sha256": sha256_array(train_mask),
                "fit_mask_sha256": sha256_array(fit_mask),
                "test_mask_sha256": sha256_array(test_mask),
                "model_identity": model_identity,
                "prediction_sha256": sha256_array(fold_prediction),
                "permuted_training_target": bool(permutation_seed is not None),
                "test_target_used_for_fit": False,
            }
        )
    if not np.all(covered):
        raise ConstraintAwareSurrogateError("OOF prediction population is incomplete")
    return {
        "prediction": prediction,
        "prediction_sha256": sha256_array(prediction),
        "fold_records": fold_records,
        "permuted_training_targets": bool(permutation_seed is not None),
        "oracle_test_target_used_for_prediction": bool(definition.model_mode == "oracle"),
    }


def projected_direction_gate(
    *,
    metrics: Mapping[str, Any],
    global_mean: float,
    condition_mean: float,
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    condition_means = [
        float(metrics["by_condition"][name]["cosine"]["mean"])
        for name in CONDITION_VALUES
    ]
    fold_means = [
        float(record["cosine"]["mean"])
        for record in metrics.get("by_fold", {}).values()
    ]
    gates = {
        "mean_cosine": float(metrics["cosine"]["mean"]) >= spec.projected_direction_mean_min,
        "nonnegative_rate": float(metrics["nonnegative_rate"]) >= spec.projected_direction_nonnegative_min,
        "minimum_condition": min(condition_means) >= spec.projected_direction_condition_min,
        "minimum_fold": bool(fold_means) and min(fold_means) >= spec.projected_direction_fold_min,
        "global_margin": (
            float(metrics["cosine"]["mean"]) - float(global_mean)
            >= spec.projected_global_margin_min
        ),
        "condition_margin": (
            float(metrics["cosine"]["mean"]) - float(condition_mean)
            >= spec.projected_condition_margin_min
        ),
    }
    return {
        **gates,
        "all": bool(all(gates.values())),
        "minimum_condition_cosine": float(min(condition_means)),
        "minimum_fold_cosine": float(min(fold_means)) if fold_means else None,
        "global_baseline_mean": float(global_mean),
        "condition_baseline_mean": float(condition_mean),
        "global_margin_value": float(
            metrics["cosine"]["mean"]
            - float(global_mean)
        ),
        "condition_margin_value": float(
            metrics["cosine"]["mean"]
            - float(condition_mean)
        ),
    }


def holdout_projected_direction_gate(
    metrics: Mapping[str, Any], *, spec: ConstraintAwareSpec
) -> Dict[str, Any]:
    condition_means = [
        float(metrics["by_condition"][name]["cosine"]["mean"])
        for name in CONDITION_VALUES
    ]
    gates = {
        "mean_cosine": float(metrics["cosine"]["mean"]) >= spec.holdout_projected_mean_min,
        "nonnegative_rate": float(metrics["nonnegative_rate"]) >= spec.holdout_projected_nonnegative_min,
        "minimum_condition": min(condition_means) >= spec.holdout_projected_condition_min,
    }
    return {
        **gates,
        "all": bool(all(gates.values())),
        "minimum_condition_cosine": float(min(condition_means)),
    }


def _fixed_integration_record(
    *,
    direction: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    timestep: int,
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
) -> Dict[str, Any]:
    integration = stagee258.integrate_rowwise(
        control=control,
        direction=direction,
        definition=fixed_integrator_definition(),
        context=context,
        spec=integrator_spec,
    )
    observable = stagee258.integrator_observable_gates(
        integration,
        spec=integrator_spec,
    )
    evaluated = stagee258.evaluate_integrated_candidate(
        integration=integration,
        control=control,
        target=target,
        groups=groups,
        condition_name=condition_name,
        timestep=int(timestep),
        context=context,
        direction_spec=direction_spec,
        integrator_spec=integrator_spec,
    )
    return {
        "integration": {key: value for key, value in integration.items() if key != "candidate"},
        "observable_gates": observable,
        "evaluation": evaluated,
        "scientific_pass": bool(observable["all"] and evaluated["scientific_pass"]),
    }


def candidate_record(
    *,
    definition: ConstraintAwareDefinition,
    context: Mapping[str, Any],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    global_baselines: Mapping[int, float],
    condition_baselines: Mapping[int, float],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: ConstraintAwareSpec,
    permutation_seed: Optional[int] = None,
) -> Dict[str, Any]:
    timestep_records = {}
    for timestep in spec.timesteps:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        target = np.asarray(context["objective_target"], dtype=np.float32)
        oracle_target = oracle_targets[int(timestep)]
        features = build_constraint_features(
            condition=context["objective_condition"],
            control=control,
            condition_name=context["objective_condition_name"],
            feature_mode=definition.feature_mode,
            context=context,
        )
        oof = fit_oof_candidate(
            definition=definition,
            features=features,
            control=control,
            raw_target=target,
            oracle_target=oracle_target,
            condition_name=context["objective_condition_name"],
            fold_assignment=context["objective_fold_assignment"],
            spec=spec,
            direction_spec=direction_spec,
            permutation_seed=permutation_seed,
        )
        metrics = staged258.row_direction_metrics(
            predicted_residual=oof["prediction"],
            target_residual=oracle_target["projected_delta"],
            condition_name=context["objective_condition_name"],
            fold_assignment=context["objective_fold_assignment"],
            epsilon=spec.standardizer_epsilon,
        )
        if definition.role in ("negative_control", "oracle_control"):
            direction_gates = {"all": False, "control": True}
        else:
            direction_gates = projected_direction_gate(
                metrics=metrics,
                global_mean=float(global_baselines[int(timestep)]),
                condition_mean=float(condition_baselines[int(timestep)]),
                spec=spec,
            )
        integrated = _fixed_integration_record(
            direction=oof["prediction"],
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
        )
        timestep_records[str(timestep)] = {
            "timestep": int(timestep),
            "feature_sha256": sha256_array(features),
            "projected_target_sha256": oracle_target["projected_delta_sha256"],
            "oof": {key: value for key, value in oof.items() if key != "prediction"},
            "projected_direction_metrics": metrics,
            "projected_direction_gates": direction_gates,
            **integrated,
        }
    eligible = bool(
        definition.role == "selectable"
        and all(
            record["projected_direction_gates"]["all"]
            and record["scientific_pass"]
            for record in timestep_records.values()
        )
    )
    return {
        "candidate_id": definition.candidate_id,
        "definition": asdict(definition),
        "role": definition.role,
        "timestep_records": timestep_records,
        "eligible": eligible,
        "permuted_training_targets": bool(permutation_seed is not None),
        "holdout_used": False,
    }


def _selection_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    timestep_records = list(record["timestep_records"].values())
    minimum_reduction = min(
        float(item["evaluation"]["target_distance_reduction_fraction"]["mean"])
        for item in timestep_records
    )
    maximum_nmse = max(
        float(item["evaluation"]["normalized_mse_ratio"])
        for item in timestep_records
    )
    minimum_physical = min(
        float(item["evaluation"]["historical_physical_row_any_rate"])
        for item in timestep_records
    )
    minimum_cosine = min(
        float(item["projected_direction_metrics"]["cosine"]["mean"])
        for item in timestep_records
    )
    definition = record["definition"]
    model_priority = {
        "reduced_rank_ridge": 0,
        "rff_ridge": 1,
    }.get(definition["model_mode"], 2)
    return (
        -minimum_reduction,
        maximum_nmse,
        -minimum_physical,
        -minimum_cosine,
        model_priority,
        int(definition["output_rank"] or definition["rff_dimension"]),
        record["candidate_id"],
    )


def select_candidate(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        return None
    return copy.deepcopy(min(eligible, key=_selection_key))


def permutation_control(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    global_baselines: Mapping[int, float],
    condition_baselines: Mapping[int, float],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    definition = ConstraintAwareDefinition(**selected["definition"])
    permuted = candidate_record(
        definition=definition,
        context=context,
        oracle_targets=oracle_targets,
        global_baselines=global_baselines,
        condition_baselines=condition_baselines,
        direction_spec=direction_spec,
        integrator_spec=integrator_spec,
        spec=spec,
        permutation_seed=spec.permutation_seed,
    )
    records = {}
    passes = []
    for timestep in spec.timesteps:
        original_mean = float(
            selected["timestep_records"][str(timestep)]["projected_direction_metrics"]["cosine"]["mean"]
        )
        permuted_record = permuted["timestep_records"][str(timestep)]
        permuted_mean = float(permuted_record["projected_direction_metrics"]["cosine"]["mean"])
        gates = {
            "permuted_cosine": permuted_mean <= spec.permutation_cosine_max,
            "margin": original_mean - permuted_mean >= spec.permutation_margin_min,
            "state_not_scientific": not bool(permuted_record["scientific_pass"]),
        }
        records[str(timestep)] = {
            "original_mean_cosine": original_mean,
            "permuted_mean_cosine": permuted_mean,
            "margin": original_mean - permuted_mean,
            "gates": {**gates, "all": bool(all(gates.values()))},
            "permuted_record": permuted_record,
        }
        passes.append(all(gates.values()))
    return {
        "candidate_id": selected["candidate_id"],
        "records": records,
        "all_pass": bool(all(passes)),
        "holdout_used": False,
    }


def projection_only_control(
    *,
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    records = {}
    scientific = []
    for timestep in spec.timesteps:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        target = np.asarray(context["objective_target"], dtype=np.float32)
        direction = np.zeros_like(control, dtype=np.float64)
        integration = stagee258.integrate_rowwise(
            control=control,
            direction=direction,
            definition=projection_only_definition(),
            context=context,
            spec=integrator_spec,
        )
        evaluated = stagee258.evaluate_integrated_candidate(
            integration=integration,
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
        )
        records[str(timestep)] = {
            "integration": {key: value for key, value in integration.items() if key != "candidate"},
            "evaluation": evaluated,
            "scientific_pass": bool(evaluated["scientific_pass"]),
        }
        scientific.append(bool(evaluated["scientific_pass"]))
    return {
        "records": records,
        "all_timesteps_scientific": bool(all(scientific)),
        "pass": bool(not all(scientific)),
        "holdout_used": False,
    }


def fit_full_and_predict(
    *,
    definition: ConstraintAwareDefinition,
    train_features: np.ndarray,
    train_control: np.ndarray,
    train_target: np.ndarray,
    train_oracle: Mapping[str, Any],
    train_names: Sequence[Any],
    test_features: np.ndarray,
    test_control: np.ndarray,
    test_names: Sequence[Any],
    spec: ConstraintAwareSpec,
    direction_spec: staged258.DirectionSurrogateSpec,
) -> Dict[str, Any]:
    definition.validate()
    if definition.model_mode == "raw_reference":
        raw_definition = _raw_reference_definition()
        raw_residual = np.asarray(train_target, dtype=np.float64) - np.asarray(train_control, dtype=np.float64)
        model = staged258.fit_surrogate_model(
            definition=raw_definition,
            features=train_features,
            target_residual=raw_residual,
            condition_name=train_names,
            spec=direction_spec,
        )
        prediction = staged258.predict_surrogate_model(
            definition=raw_definition,
            model=model,
            features=test_features,
            condition_name=test_names,
            target_shape=tuple(np.asarray(test_control).shape[1:]),
        )
        return {"prediction": prediction, "model_identity": model["identity"]}
    encoded = encode_projected_target(
        control=train_control,
        oracle_candidate=train_oracle["candidate"],
        target_mode=definition.target_mode,
    )
    fit_mask = np.ones(encoded.shape[0], dtype=np.bool_)
    if definition.fit_population == "oracle_feasible":
        fit_mask &= np.asarray(train_oracle["feasible_mask"], dtype=np.bool_)
    model = fit_constraint_model(
        definition=definition,
        features=train_features,
        encoded_target=encoded,
        condition_name=train_names,
        fit_mask=fit_mask,
        spec=spec,
    )
    encoded_prediction = predict_constraint_model(
        definition=definition,
        model=model,
        features=test_features,
        condition_name=test_names,
    )
    prediction = decode_projected_prediction(
        control=test_control,
        encoded=encoded_prediction,
        target_mode=definition.target_mode,
    )
    return {"prediction": prediction, "model_identity": model["identity"]}


def locked_holdout_evaluation(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    objective_oracle_targets: Mapping[int, Mapping[str, Any]],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: stagee258.ConstrainedIntegratorSpec,
    spec: ConstraintAwareSpec,
) -> Dict[str, Any]:
    definition = ConstraintAwareDefinition(**selected["definition"])
    records = {}
    passes = []
    for timestep in spec.timesteps:
        train_control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        train_target = np.asarray(context["objective_target"], dtype=np.float32)
        holdout_control = np.asarray(context["control_predictions"][int(timestep)], dtype=np.float32)
        holdout_target = np.asarray(context["holdout_target"], dtype=np.float32)
        train_features = build_constraint_features(
            condition=context["objective_condition"],
            control=train_control,
            condition_name=context["objective_condition_name"],
            feature_mode=definition.feature_mode,
            context=context,
        )
        holdout_features = build_constraint_features(
            condition=context["holdout_condition"],
            control=holdout_control,
            condition_name=context["holdout_condition_name"],
            feature_mode=definition.feature_mode,
            context=context,
        )
        fitted = fit_full_and_predict(
            definition=definition,
            train_features=train_features,
            train_control=train_control,
            train_target=train_target,
            train_oracle=objective_oracle_targets[int(timestep)],
            train_names=context["objective_condition_name"],
            test_features=holdout_features,
            test_control=holdout_control,
            test_names=context["holdout_condition_name"],
            spec=spec,
            direction_spec=direction_spec,
        )
        predicted = np.asarray(fitted["prediction"], dtype=np.float64)
        integrated = _fixed_integration_record(
            direction=predicted,
            control=holdout_control,
            target=holdout_target,
            groups=context["holdout_groups"],
            condition_name=context["holdout_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
        )
        # Holdout target is opened only after the locked prediction and candidate exist.
        holdout_oracle = generate_projected_oracle_target(
            control=holdout_control,
            target=holdout_target,
            groups=context["holdout_groups"],
            condition_name=context["holdout_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
            spec=spec,
        )
        metrics = staged258.row_direction_metrics(
            predicted_residual=predicted,
            target_residual=holdout_oracle["projected_delta"],
            condition_name=context["holdout_condition_name"],
            fold_assignment=None,
            epsilon=spec.standardizer_epsilon,
        )
        direction_gates = holdout_projected_direction_gate(metrics, spec=spec)
        scientific_pass = bool(direction_gates["all"] and integrated["scientific_pass"])
        records[str(timestep)] = {
            "model_identity": fitted["model_identity"],
            "train_feature_sha256": sha256_array(train_features),
            "holdout_feature_sha256": sha256_array(holdout_features),
            "prediction_sha256": sha256_array(predicted),
            "holdout_target_used_for_fit": False,
            "holdout_target_used_for_prediction": False,
            "holdout_target_used_for_candidate_generation": False,
            "holdout_target_used_after_locked_prediction_for_evaluation": True,
            "holdout_oracle": {key: value for key, value in holdout_oracle.items() if key not in ("candidate", "projected_delta", "feasible_mask")},
            "projected_direction_metrics": metrics,
            "projected_direction_gates": direction_gates,
            **integrated,
            "scientific_pass": scientific_pass,
        }
        passes.append(scientific_pass)
    return {
        "candidate_id": selected["candidate_id"],
        "definition": copy.deepcopy(selected["definition"]),
        "timestep_records": records,
        "scientific_pass": bool(all(passes)),
        "selection_changed_after_holdout": False,
        "holdout_used_for_fit": False,
        "holdout_used_for_selection": False,
        "holdout_evaluated": True,
    }


def classify(
    *,
    records: Sequence[Mapping[str, Any]],
    oracle_targets: Mapping[int, Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    permutation: Optional[Mapping[str, Any]],
    projection_only: Optional[Mapping[str, Any]],
    holdout: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    selectable = [record for record in records if record["role"] == "selectable"]
    direction_all = [
        record["candidate_id"]
        for record in selectable
        if all(
            item["projected_direction_gates"]["all"]
            for item in record["timestep_records"].values()
        )
    ]
    state_all = [
        record["candidate_id"]
        for record in selectable
        if all(item["scientific_pass"] for item in record["timestep_records"].values())
    ]
    eligible = [record["candidate_id"] for record in selectable if record["eligible"]]
    oracle_all = all(
        bool(oracle_targets[int(timestep)]["evaluation"]["scientific_pass"])
        for timestep in TIMESTEPS
    )
    if not oracle_all:
        root = "phase314b_r258_stagef_projected_oracle_target_not_stable"
        next_path = "AUDIT_PROJECTED_ORACLE_SUPERVISION_CONTRACT"
        locus = "projected_oracle_target"
    elif selected is None:
        if not direction_all:
            root = "phase314b_r258_stagef_constraint_aware_target_has_no_grouped_cv_signal"
            next_path = "AUDIT_LOCAL_CONSTRAINT_FEATURE_IDENTIFIABILITY"
            locus = "constraint_aware_direction_prediction"
        elif not state_all:
            root = "phase314b_r258_stagef_constraint_aware_direction_still_not_integrable"
            next_path = "CALIBRATE_JOINT_DIRECTION_AND_FEASIBILITY_SURROGATE"
            locus = "post_prediction_integrability"
        else:
            root = "phase314b_r258_stagef_state_witness_fails_selection_policy"
            next_path = "AUDIT_CONSTRAINT_AWARE_SELECTION_POLICY"
            locus = "selection_policy"
    elif permutation is None or not permutation["all_pass"]:
        root = "phase314b_r258_stagef_permutation_control_indicates_target_shortcut"
        next_path = "AUDIT_CONSTRAINT_AWARE_SURROGATE_LEAKAGE"
        locus = "permutation_shortcut"
    elif projection_only is None or not projection_only["pass"]:
        root = "phase314b_r258_stagef_projection_only_control_explains_witness"
        next_path = "AUDIT_CONSTRAINT_ONLY_SHORTCUT"
        locus = "projection_only_shortcut"
    elif holdout is None:
        raise ConstraintAwareSurrogateError("selected candidate lacks locked holdout")
    elif not holdout["scientific_pass"]:
        root = "phase314b_r258_stagef_constraint_aware_surrogate_does_not_generalize"
        next_path = "AUDIT_NONLINEAR_SEQUENCE_CONSTRAINT_AWARE_SURROGATE"
        locus = "holdout_generalization"
    else:
        root = "phase314b_r258_stagef_constraint_aware_surrogate_selected_and_holdout_validated"
        next_path = "CALIBRATE_SELECTED_CONSTRAINT_AWARE_SURROGATE_IN_TRAIN_ONLY_DENOISER"
        locus = "constraint_aware_surrogate_validated"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "selectable_candidate_count": len(selectable),
        "direction_all_candidate_ids": direction_all,
        "state_all_candidate_ids": state_all,
        "eligible_candidate_ids": eligible,
        "projected_oracle_all_timesteps": bool(oracle_all),
        "selected_configuration": copy.deepcopy(selected),
        "permutation_pass": None if permutation is None else bool(permutation["all_pass"]),
        "projection_only_pass": None if projection_only is None else bool(projection_only["pass"]),
        "holdout_scientific_pass": None if holdout is None else bool(holdout["scientific_pass"]),
    }


def build_context(
    *,
    root: Path,
    captured: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
) -> Dict[str, Any]:
    return staged258.build_direction_context(
        root=Path(root).resolve(),
        captured=captured,
        spec=direction_spec,
    )


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[ConstraintAwareSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
    integrator_spec: Optional[stagee258.ConstrainedIntegratorSpec] = None,
) -> Dict[str, Any]:
    active = ConstraintAwareSpec() if spec is None else spec
    active_direction = staged258.DirectionSurrogateSpec() if direction_spec is None else direction_spec
    active_integrator = (
        stagee258.ConstrainedIntegratorSpec() if integrator_spec is None else integrator_spec
    )
    active.validate()
    active_direction.validate()
    active_integrator.validate()
    repository_root = Path(root).resolve()
    base = validate_base_evidence(repository_root)
    stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != EXPECTED_COMPATIBILITY_SHA256:
        raise ConstraintAwareSurrogateError("portable compatibility SHA changed")
    cold = stagec258.stagea258.assert_cold_cuda_context_portable()
    captured = stageb258.capture_portable_control_model(root=repository_root)
    context = build_context(
        root=repository_root,
        captured=captured,
        direction_spec=active_direction,
    )
    translation = translation_invariance_audit(
        condition=context["objective_condition"],
        control=context["objective_control_predictions"][10],
        condition_name=context["objective_condition_name"],
        context=context,
        spec=active,
    )
    oracle_targets = {}
    for timestep in active.timesteps:
        oracle_targets[int(timestep)] = generate_projected_oracle_target(
            control=context["objective_control_predictions"][int(timestep)],
            target=context["objective_target"],
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active,
        )
    placeholder_baselines = {
        int(timestep): -1.0
        for timestep in active.timesteps
    }
    global_record = candidate_record(
        definition=definition_by_id("projected_global_mean"),
        context=context,
        oracle_targets=oracle_targets,
        global_baselines=placeholder_baselines,
        condition_baselines=placeholder_baselines,
        direction_spec=active_direction,
        integrator_spec=active_integrator,
        spec=active,
    )
    condition_record = candidate_record(
        definition=definition_by_id("projected_condition_mean"),
        context=context,
        oracle_targets=oracle_targets,
        global_baselines=placeholder_baselines,
        condition_baselines=placeholder_baselines,
        direction_spec=active_direction,
        integrator_spec=active_integrator,
        spec=active,
    )
    global_baselines = {
        int(timestep): float(
            global_record["timestep_records"][str(timestep)]
            ["projected_direction_metrics"]["cosine"]["mean"]
        )
        for timestep in active.timesteps
    }
    condition_baselines = {
        int(timestep): float(
            condition_record["timestep_records"][str(timestep)]
            ["projected_direction_metrics"]["cosine"]["mean"]
        )
        for timestep in active.timesteps
    }
    records = [global_record, condition_record]
    for definition in CANDIDATE_DEFINITIONS:
        if definition.candidate_id in (
            "projected_global_mean",
            "projected_condition_mean",
        ):
            continue
        records.append(
            candidate_record(
                definition=definition,
                context=context,
                oracle_targets=oracle_targets,
                global_baselines=global_baselines,
                condition_baselines=condition_baselines,
                direction_spec=active_direction,
                integrator_spec=active_integrator,
                spec=active,
            )
        )
    order = {definition.candidate_id: index for index, definition in enumerate(CANDIDATE_DEFINITIONS)}
    records.sort(key=lambda record: order[record["candidate_id"]])
    selected = select_candidate(records)
    permutation = None
    projection = None
    holdout = None
    if selected is not None:
        permutation = permutation_control(
            selected=selected,
            context=context,
            oracle_targets=oracle_targets,
            global_baselines=global_baselines,
            condition_baselines=condition_baselines,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active,
        )
        projection = projection_only_control(
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
            spec=active,
        )
        if permutation["all_pass"] and projection["pass"]:
            holdout = locked_holdout_evaluation(
                selected=selected,
                context=context,
                objective_oracle_targets=oracle_targets,
                direction_spec=active_direction,
                integrator_spec=active_integrator,
                spec=active,
            )
    classification = classify(
        records=records,
        oracle_targets=oracle_targets,
        selected=selected,
        permutation=permutation,
        projection_only=projection,
        holdout=holdout,
    )
    validated = (
        copy.deepcopy(selected)
        if (
            selected is not None
            and permutation is not None
            and permutation["all_pass"]
            and projection is not None
            and projection["pass"]
            and holdout is not None
            and holdout["scientific_pass"]
        )
        else None
    )
    contract = {
        "schema": "phase314b_r258_stagef_constraint_aware_contract_v2_structural_zero",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_identity": {
            "worker_sha256": EXPECTED_BASE_WORKER_SHA256,
            "internal_contract_sha256": EXPECTED_BASE_INTERNAL_CONTRACT_SHA256,
            "contract_file_sha256": EXPECTED_BASE_CONTRACT_FILE_SHA256,
            "selection_sha256": EXPECTED_BASE_SELECTION_SHA256,
            "root_cause": EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": EXPECTED_BASE_NEXT_PATH,
        },
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "constraint_aware_spec": asdict(active),
        "direction_spec": asdict(active_direction),
        "integrator_spec": asdict(active_integrator),
        "fixed_integrator": asdict(fixed_integrator_definition()),
        "candidate_definitions": [asdict(definition) for definition in CANDIDATE_DEFINITIONS],
        "supervision_contract": {
            "objective_train_target": "frozen Stage-E projected-oracle displacement",
            "oracle_target_uses_ground_truth": True,
            "test_fold_target_used_for_model_fit": False,
            "feature_standardizer_fit": "outer training fold only",
            "output_basis_fit": "outer training fold only",
            "RFF_map_fit": "fixed seed; feature standardizer outer training fold only",
            "target_standardizer_used": False,
            "selection_holdout_used_for_fit_or_selection": False,
            "frozen_probe_accessed": False,
        },
        "feature_contract": {
            "condition_name_in_selectable_features": False,
            "group_pair_seed_window_metadata_in_features": False,
            "translation_invariant": True,
            "anchor_dimension": 211,
            "full_centered_constraint_dimension": FULL_CENTERED_CONSTRAINT_DIMENSION,
            "full_segment_constraint_dimension": FULL_SEGMENT_CONSTRAINT_DIMENSION,
            "constraint_segment_count": CONSTRAINT_SEGMENT_COUNT,
            "constraint_state_representation": {
                "continuous": "clipped z on resolvable segments; zero-filled on structural zeros",
                "discrete": "exact structural-zero mask",
                "zero_definition": "adjacent ordered bead coordinates are exactly equal",
                "threshold_learned": False,
                "holdout_or_frozen_probe_used_to_define_mask": False,
            },
        },
        "source_sha256": base["source_sha256"],
        "diffusion_model_candidate_trained": False,
        "model_architecture_changed": False,
        "surrogate_weights_persisted": False,
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    selection_payload = {
        "schema": "phase314b_r258_stagef_selection_v2_structural_zero",
        "phase": PHASE,
        "selected_configuration": copy.deepcopy(selected),
        "validated_train_only_recommendation": copy.deepcopy(validated),
        "classification": classification,
        "candidate_ids": [record["candidate_id"] for record in records],
        "permutation_pass": None if permutation is None else permutation["all_pass"],
        "projection_only_pass": None if projection is None else projection["pass"],
        "holdout_scientific_pass": None if holdout is None else holdout["scientific_pass"],
    }
    selection_payload["selection_sha256"] = sha256_bytes(stable_json_bytes(selection_payload))
    ready = validated is not None
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagef_result_v2_structural_zero",
        "verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "immutable_inputs": {
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "base_worker_sha256": EXPECTED_BASE_WORKER_SHA256,
            "base_internal_contract_sha256": EXPECTED_BASE_INTERNAL_CONTRACT_SHA256,
            "base_contract_file_sha256": EXPECTED_BASE_CONTRACT_FILE_SHA256,
            "base_selection_sha256": EXPECTED_BASE_SELECTION_SHA256,
            "base_file_sha256": base["file_sha256"],
            "source_sha256": base["source_sha256"],
        },
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "split": {
            "objective_train_rows": int(context["objective_target"].shape[0]),
            "objective_train_groups": int(len(set(context["objective_groups"].tolist()))),
            "selection_holdout_rows": int(context["holdout_target"].shape[0]),
            "frozen_probe_rows": int(np.sum(context["frozen_probe_mask"])),
            "fold_assignment_sha256": sha256_array(context["objective_fold_assignment"]),
        },
        "translation_invariance": translation,
        "projected_oracle_targets": {
            str(timestep): {
                key: value
                for key, value in oracle_targets[int(timestep)].items()
                if key not in ("candidate", "projected_delta", "feasible_mask")
            }
            for timestep in active.timesteps
        },
        "global_projected_baselines": {
            str(key): value
            for key, value in global_baselines.items()
        },
        "condition_projected_baselines": {
            str(key): value
            for key, value in condition_baselines.items()
        },
        "candidate_records": records,
        "objective_train_selected_configuration": copy.deepcopy(selected),
        "permutation_control": permutation,
        "projection_only_control": projection,
        "locked_holdout_evaluation": holdout,
        "classification": classification,
        "constraint_aware_contract": contract,
        "selection": selection_payload,
        "selected_configuration": copy.deepcopy(validated),
        "train_only_recommendation": copy.deepcopy(validated),
        "selection_holdout_evaluated": bool(holdout is not None),
        "selection_holdout_used_for_fit": False,
        "selection_holdout_used_for_selection": False,
        "frozen_probe_accessed": False,
        "new_diffusion_model_candidate_trained": False,
        "reverse_sampling_run": False,
        "formal_training_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "surrogate_weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }


def identity_projection(result: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(result))


def compare_worker_results(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    left_identity = identity_projection(left)
    right_identity = identity_projection(right)
    left_bytes = stable_json_bytes(left_identity)
    right_bytes = stable_json_bytes(right_identity)
    return {
        "exact": left_bytes == right_bytes,
        "left_sha256": sha256_bytes(left_bytes),
        "right_sha256": sha256_bytes(right_bytes),
    }


def status_paths(root: Path) -> Tuple[str, ...]:
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value)
    return tuple(sorted(paths))


def validate_initial_worktree(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(repository_root, "branch", "--show-current") != "Experiment1":
        raise ConstraintAwareSurrogateError("Stage F requires Experiment1")
    head = git_output(repository_root, "rev-parse", "HEAD")
    remote = git_output(repository_root, "rev-parse", "origin/Experiment1")
    if head != BASE_EVIDENCE_COMMIT or remote != BASE_EVIDENCE_COMMIT:
        raise ConstraintAwareSurrogateError("Stage-F initial local/remote HEAD changed")
    validate_base_evidence(repository_root)
    expected = tuple(sorted(STAGEF_IMPLEMENTATION_PATHS))
    if status_paths(repository_root) != expected:
        raise ConstraintAwareSurrogateError(
            "unexpected initial Stage-F worktree paths: {}".format(status_paths(repository_root))
        )
    submodule = repository_root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise ConstraintAwareSurrogateError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise ConstraintAwareSurrogateError("DeformableRavens worktree is dirty")
    return {
        "branch": "Experiment1",
        "head": head,
        "origin_experiment1": remote,
        "expected_untracked_paths": list(expected),
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def commit_parent(root: Path, commit: str) -> str:
    return git_output(root, "rev-parse", "{}^".format(commit))


def commit_subject(root: Path, commit: str) -> str:
    return git_output(root, "show", "-s", "--format=%s", commit)


def commit_paths(root: Path, commit: str) -> Tuple[str, ...]:
    output = git_output(root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    return tuple(sorted(line.strip() for line in output.splitlines() if line.strip()))


def validate_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    if commit_parent(repository_root, commit) != BASE_EVIDENCE_COMMIT:
        raise ConstraintAwareSurrogateError("Stage-F implementation parent changed")
    if commit_subject(repository_root, commit) != IMPLEMENTATION_MESSAGE:
        raise ConstraintAwareSurrogateError("Stage-F implementation subject changed")
    if commit_paths(repository_root, commit) != tuple(sorted(STAGEF_IMPLEMENTATION_PATHS)):
        raise ConstraintAwareSurrogateError("Stage-F implementation paths changed")
    for relative in STAGEF_IMPLEMENTATION_PATHS:
        assert_file_bound_to_commit(repository_root, relative, commit)
    return {
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "implementation_commit": commit,
    }
