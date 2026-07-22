"""Phase3.14b-r2.5.8 Stage E constrained direct-x0 integrator.

Stage D established that deployable grouped-OOF direction surrogates predict the
raw ordered-cable XY residual with useful cosine at t=10/25/50, but the
unconstrained state update ``control + scale * direction`` never forms a full
scientific state witness.  This stage therefore does not train a diffusion
model and does not redesign the direction surrogate.  It calibrates a
*target-independent* direct-x0 integration operator that:

* moves along an OOF surrogate direction;
* clips ordered segment-vector lengths to frozen train-only bounds;
* preserves the proposed cable centroid and segment orientations;
* uses only candidate/control values and frozen geometry contracts when
  selecting a per-row scale;
* uses ground truth only after candidate generation for scientific scoring.

Candidate and policy selection use only the 638-row objective-training
population through six-fold grouped OOF predictions.  The 236-row selection
holdout remains closed until one integrator candidate, direction source, and
all policy parameters are locked.  The 126-row frozen probe remains closed.
No surrogate weights, prediction tensors, checkpoints, NPZ files, caches,
images, or videos are persisted.
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
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r258_stageb_direct_x0_reachability as stageb258
from ccda_phase3 import phase314b_r258_stagec_balanced_geometry as stagec258
from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged258

PHASE = "Phase3.14b-r2.5.8 Stage E"
PHASE_ID = "phase314b_r258_stagee"

BASE_EVIDENCE_COMMIT = "61be1c377eeda6da4c25e21399e160beb9a33249"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_CONTRACT = "reports/phase3_14b_r258_staged_resume2_contract.json"
BASE_WORKER = "reports/phase3_14b_r258_staged_resume2_worker_evidence.json"
BASE_SUMMARY = "reports/phase3_14b_r258_staged_resume2_summary.json"
BASE_REPORT = "reports/phase3_14b_r258_staged_resume2_report.md"
BASE_TEST_GATE = "reports/phase3_14b_r258_staged_resume2_test_gate_summary.json"
BASE_BOUND_FILES = (
    BASE_CONTRACT,
    BASE_WORKER,
    BASE_SUMMARY,
    BASE_REPORT,
    BASE_TEST_GATE,
)

EXPECTED_BASE_WORKER_SHA256 = "0d9cd32cb8724d9e90c7c29a826e2e12e66937efccb7123320b5fbf037430c4f"
EXPECTED_BASE_CONTRACT_SHA256 = "3bfb7f19a1f2d37ee8e7f2f202c2a304396ee1b611932ca2012d841b4540f6e4"
EXPECTED_BASE_SELECTION_SHA256 = "3f78083f8038b283648b03717a95ee00b5851abf69ab3ee256959b180b0f8dba"
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_staged_grouped_cv_direction_signal_"
    "does_not_form_valid_state_transition"
)
EXPECTED_BASE_NEXT_PATH = "CALIBRATE_SURROGATE_GUIDED_CONSTRAINED_DIRECT_X0_INTEGRATOR"
EXPECTED_COMPATIBILITY_SHA256 = "03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec"

SOURCE_BOUND_SHA256 = {
    "ccda_phase3/phase314b_r258_staged_direction_surrogate.py": (
        "946fca9d84b54e4affd35ab890970ba1f17154aa8b925f66746994ffe484bbb9"
    ),
    "ccda_phase3/phase314b_r258_stageb_direct_x0_reachability.py": (
        "42e4a382a22baffdff6ea9778ec89772ab25a53f2044c90c6d0870dd4e6fd3dd"
    ),
    "ccda_phase3/phase314b_r258_stagec_balanced_geometry.py": (
        "65e3f41a1788e90dcd87de1bfd0f373d67fa34bc3c2b6f7fa595f5f2968b77c6"
    ),
    "ccda_phase3/phase314b_r258_staged_resume2_temporal_views.py": (
        "4830b09eafcab374de197b770532d879f63472d8c0d9f9a9ca430f9b110257d8"
    ),
    "scripts/phase3_14b_r258_staged_resume2_test_gate.py": (
        "66a09d2366fd093a5420f59caf14c0619e49a675ef3ec450d90eb82574c2c39b"
    ),
    "scripts/phase3_14b_r258_staged_resume2_run_calibration.py": (
        "bd0183d93073d07e548bbe8bea80af837e825cc81e3ac99bfe2eb1a3d0597679"
    ),
    "tests/test_phase3_14b_r258_staged_resume2_temporal_views.py": (
        "559486e146264fd091d3c772c16f6ab93657b3a138097f9a5787abf45b328cb3"
    ),
}

DIRECTION_SOURCE_IDS = (
    "centered_ridge_a10",
    "centered_rridge_k32_a10",
    "segment_rridge_k32_a10",
)


class ConstrainedIntegratorError(RuntimeError):
    """Raised when evidence, integrator, leakage, or selection invariants fail."""


@dataclass(frozen=True)
class IntegratorDefinition:
    candidate_id: str
    direction_source_id: str
    integration_mode: str
    bound_mode: str
    lower_z: float
    upper_z: float
    retention_min: float
    maximum_scale: float
    role: str

    def validate(self) -> None:
        if self.integration_mode not in ("linear", "segment_reconstruct"):
            raise ValueError("unknown integration mode")
        if self.bound_mode not in ("none", "historical", "robust_intersection"):
            raise ValueError("unknown bound mode")
        if self.role not in (
            "negative_control",
            "diagnostic_control",
            "oracle_control",
            "selectable",
        ):
            raise ValueError("unknown integrator role")
        if self.direction_source_id not in (*DIRECTION_SOURCE_IDS, "zero", "oracle"):
            raise ValueError("unknown direction source")
        if self.integration_mode == "linear" and self.bound_mode != "none":
            raise ValueError("linear integrator cannot use reconstruction bounds")
        if self.integration_mode == "segment_reconstruct" and self.bound_mode == "none":
            raise ValueError("reconstruction requires segment bounds")
        if self.bound_mode == "robust_intersection":
            if self.lower_z <= 0.0 or self.upper_z <= 0.0:
                raise ValueError("robust z bounds must be positive")
        elif self.lower_z != 0.0 or self.upper_z != 0.0:
            raise ValueError("non-robust candidate has z bounds")
        if not 0.0 <= self.retention_min <= 1.0:
            raise ValueError("direction-retention threshold is invalid")
        if self.maximum_scale <= 0.0:
            raise ValueError("maximum scale must be positive")
        if self.role == "selectable" and self.direction_source_id not in DIRECTION_SOURCE_IDS:
            raise ValueError("selectable integrator must use a deployable direction source")
        if self.role == "oracle_control" and self.direction_source_id != "oracle":
            raise ValueError("oracle control must use oracle direction")
        if self.direction_source_id == "zero" and self.role != "diagnostic_control":
            raise ValueError("zero direction is diagnostic only")


@dataclass(frozen=True)
class ConstrainedIntegratorSpec:
    timesteps: Tuple[int, ...] = (10, 25, 50)
    scale_grid: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
    grouped_cv_folds: int = 6
    objective_train_noise_seed_offset: int = 8801
    permutation_seed: int = 151117
    standardizer_epsilon: float = 1.0e-12
    segment_tolerance: float = 2.5e-6
    coordinate_tolerance: float = 1.0e-6
    observable_feasible_rate_min: float = 0.90
    fallback_rate_max: float = 0.10
    mean_retention_min: float = 0.35
    correction_ratio_p95_max: float = 2.0
    permutation_cosine_max: float = 0.10
    permutation_margin_min: float = 0.10
    projection_only_all_timestep_forbidden: bool = True

    def validate(self) -> None:
        if self.timesteps != (10, 25, 50):
            raise ValueError("timestep population changed")
        if self.scale_grid != tuple(sorted(set(self.scale_grid))):
            raise ValueError("scale grid must be sorted and unique")
        if any(float(value) <= 0.0 for value in self.scale_grid):
            raise ValueError("scale grid must be positive")
        if self.grouped_cv_folds != 6:
            raise ValueError("grouped CV fold count changed")
        if self.objective_train_noise_seed_offset != 8801:
            raise ValueError("objective-train noise seed changed")
        for value in (
            self.standardizer_epsilon,
            self.segment_tolerance,
            self.coordinate_tolerance,
            self.observable_feasible_rate_min,
            self.mean_retention_min,
            self.correction_ratio_p95_max,
            self.permutation_margin_min,
        ):
            if float(value) <= 0.0:
                raise ValueError("positive integrator threshold is invalid")
        if not 0.0 <= self.fallback_rate_max < 1.0:
            raise ValueError("fallback threshold is invalid")
        if not 0.0 <= self.permutation_cosine_max < 1.0:
            raise ValueError("permutation cosine threshold is invalid")
        if tuple(definition.candidate_id for definition in INTEGRATOR_DEFINITIONS) != (
            "linear_centered_a10_m2",
            "centered_a10_hist_m1_r25",
            "centered_a10_z4_m1_r25",
            "centered_a10_z3_m1_r50",
            "centered_a10_z4_m2_r25",
            "centered_rr32_z4_m1_r25",
            "centered_rr32_z3_m1_r50",
            "centered_rr32_z4_m2_r25",
            "segment_rr32_z4_m1_r25",
            "segment_rr32_z3_m1_r50",
            "segment_rr32_z4_m2_r25",
            "zero_z4_projection_only",
            "oracle_z4_m2_r25",
        ):
            raise ValueError("integrator candidate order changed")
        for definition in INTEGRATOR_DEFINITIONS:
            definition.validate()


INTEGRATOR_DEFINITIONS = (
    IntegratorDefinition(
        "linear_centered_a10_m2",
        "centered_ridge_a10",
        "linear",
        "none",
        0.0,
        0.0,
        0.95,
        2.0,
        "negative_control",
    ),
    IntegratorDefinition(
        "centered_a10_hist_m1_r25",
        "centered_ridge_a10",
        "segment_reconstruct",
        "historical",
        0.0,
        0.0,
        0.25,
        1.0,
        "diagnostic_control",
    ),
    IntegratorDefinition(
        "centered_a10_z4_m1_r25",
        "centered_ridge_a10",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        1.0,
        "selectable",
    ),
    IntegratorDefinition(
        "centered_a10_z3_m1_r50",
        "centered_ridge_a10",
        "segment_reconstruct",
        "robust_intersection",
        3.0,
        3.0,
        0.50,
        1.0,
        "selectable",
    ),
    IntegratorDefinition(
        "centered_a10_z4_m2_r25",
        "centered_ridge_a10",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        2.0,
        "selectable",
    ),
    IntegratorDefinition(
        "centered_rr32_z4_m1_r25",
        "centered_rridge_k32_a10",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        1.0,
        "selectable",
    ),
    IntegratorDefinition(
        "centered_rr32_z3_m1_r50",
        "centered_rridge_k32_a10",
        "segment_reconstruct",
        "robust_intersection",
        3.0,
        3.0,
        0.50,
        1.0,
        "selectable",
    ),
    IntegratorDefinition(
        "centered_rr32_z4_m2_r25",
        "centered_rridge_k32_a10",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        2.0,
        "selectable",
    ),
    IntegratorDefinition(
        "segment_rr32_z4_m1_r25",
        "segment_rridge_k32_a10",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        1.0,
        "selectable",
    ),
    IntegratorDefinition(
        "segment_rr32_z3_m1_r50",
        "segment_rridge_k32_a10",
        "segment_reconstruct",
        "robust_intersection",
        3.0,
        3.0,
        0.50,
        1.0,
        "selectable",
    ),
    IntegratorDefinition(
        "segment_rr32_z4_m2_r25",
        "segment_rridge_k32_a10",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        2.0,
        "selectable",
    ),
    IntegratorDefinition(
        "zero_z4_projection_only",
        "zero",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.0,
        1.0,
        "diagnostic_control",
    ),
    IntegratorDefinition(
        "oracle_z4_m2_r25",
        "oracle",
        "segment_reconstruct",
        "robust_intersection",
        4.0,
        4.0,
        0.25,
        2.0,
        "oracle_control",
    ),
)


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
        raise ConstrainedIntegratorError("JSON root is not an object: {}".format(path))
    return value


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def assert_commit_ancestor(root: Path, commit: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ConstrainedIntegratorError("base evidence commit is not an ancestor")


def assert_file_bound_to_commit(root: Path, relative: str, commit: str) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        ["git", "show", "{}:{}".format(commit, relative)],
        cwd=str(root),
    )
    current = path.read_bytes()
    if committed != current:
        raise ConstrainedIntegratorError("file differs from bound commit: {}".format(relative))
    return sha256_bytes(current)


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_ancestor(repository_root, BASE_EVIDENCE_COMMIT)
    file_sha = {
        relative: assert_file_bound_to_commit(repository_root, relative, BASE_EVIDENCE_COMMIT)
        for relative in BASE_BOUND_FILES
    }
    source_sha = {
        relative: assert_file_bound_to_commit(repository_root, relative, BASE_EVIDENCE_COMMIT)
        for relative in SOURCE_BOUND_SHA256
    }
    for relative, expected in SOURCE_BOUND_SHA256.items():
        if source_sha[relative] != expected:
            raise ConstrainedIntegratorError("frozen source SHA changed: {}".format(relative))

    contract = load_json(repository_root / BASE_CONTRACT)
    worker = load_json(repository_root / BASE_WORKER)
    summary = load_json(repository_root / BASE_SUMMARY)
    test_gate = load_json(repository_root / BASE_TEST_GATE)
    if worker.get("workers_exact") is not True or int(worker.get("worker_count", -1)) != 2:
        raise ConstrainedIntegratorError("Stage-D Resume2 workers are not exact")
    comparison = worker.get("comparison", {})
    if comparison.get("left_sha256") != EXPECTED_BASE_WORKER_SHA256:
        raise ConstrainedIntegratorError("Stage-D Resume2 worker SHA changed")
    if comparison.get("right_sha256") != EXPECTED_BASE_WORKER_SHA256:
        raise ConstrainedIntegratorError("Stage-D Resume2 worker pair changed")
    result = worker.get("worker_result", {})
    if result.get("scientific_status") != "BLOCKED":
        raise ConstrainedIntegratorError("Stage-D Resume2 scientific status changed")
    if result.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise ConstrainedIntegratorError("Stage-D Resume2 root cause changed")
    if result.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise ConstrainedIntegratorError("Stage-D Resume2 next path changed")
    if result.get("selected_configuration") is not None:
        raise ConstrainedIntegratorError("Stage-D Resume2 selected a configuration")
    if result.get("train_only_recommendation") is not None:
        raise ConstrainedIntegratorError("Stage-D Resume2 emitted a recommendation")
    result_contract = result.get("direction_surrogate_contract", {})
    if result_contract.get("contract_sha256") != EXPECTED_BASE_CONTRACT_SHA256:
        raise ConstrainedIntegratorError("Stage-D Resume2 internal contract changed")
    if result.get("selection", {}).get("selection_sha256") != EXPECTED_BASE_SELECTION_SHA256:
        raise ConstrainedIntegratorError("Stage-D Resume2 selection changed")
    records = result.get("candidate_records")
    if not isinstance(records, list) or len(records) != 9:
        raise ConstrainedIntegratorError("Stage-D candidate population changed")
    selectable = [record for record in records if record.get("role") == "selectable"]
    if len(selectable) != 6:
        raise ConstrainedIntegratorError("Stage-D selectable population changed")
    for record in selectable:
        if record.get("eligible") is not False:
            raise ConstrainedIntegratorError("Stage-D candidate unexpectedly eligible")
        timestep_records = record.get("timestep_records", {})
        if tuple(sorted(int(key) for key in timestep_records)) != (10, 25, 50):
            raise ConstrainedIntegratorError("Stage-D timestep record changed")
        if not all(
            item.get("direction_gates", {}).get("all") is True
            for item in timestep_records.values()
        ):
            raise ConstrainedIntegratorError("Stage-D direction gate no longer passes")
        if any(
            item.get("scale_sweep", {}).get("scientific_reachable") is True
            for item in timestep_records.values()
        ):
            raise ConstrainedIntegratorError("Stage-D state witness unexpectedly exists")
    if summary.get("verdict") != "PASS" or summary.get("scientific_status") != "BLOCKED":
        raise ConstrainedIntegratorError("Stage-D Resume2 summary changed")
    if contract.get("contract_sha256") != EXPECTED_BASE_CONTRACT_SHA256:
        raise ConstrainedIntegratorError("Stage-D Resume2 contract record changed")
    if contract.get("selection_sha256") != EXPECTED_BASE_SELECTION_SHA256:
        raise ConstrainedIntegratorError("Stage-D Resume2 contract selection changed")
    if test_gate.get("verdict") != "PASS":
        raise ConstrainedIntegratorError("Stage-D Resume2 test gate is not PASS")
    if int(test_gate.get("test_file_count", -1)) != 54:
        raise ConstrainedIntegratorError("Stage-D Resume2 test-file count changed")
    if int(test_gate.get("passed_test_count", -1)) != 1243:
        raise ConstrainedIntegratorError("Stage-D Resume2 pass count changed")
    return {
        "file_sha256": file_sha,
        "source_sha256": source_sha,
        "contract": contract,
        "worker": worker,
        "worker_result": result,
        "summary": summary,
        "test_gate": test_gate,
    }


def definition_by_id(candidate_id: str) -> staged258.SurrogateDefinition:
    matches = [
        definition
        for definition in staged258.SURROGATE_DEFINITIONS
        if definition.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise ConstrainedIntegratorError("direction source definition changed: {}".format(candidate_id))
    definition = matches[0]
    definition.validate()
    return definition


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    if not np.all(np.isfinite(array)):
        raise ValueError("stats contain NaN or Inf")
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


def _row_cosine(left: np.ndarray, right: np.ndarray, *, epsilon: float) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64).reshape(left.shape[0], -1)
    b = np.asarray(right, dtype=np.float64).reshape(right.shape[0], -1)
    numerator = np.sum(a * b, axis=1)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return numerator / np.maximum(denominator, float(epsilon))


def segment_bounds(
    *,
    definition: IntegratorDefinition,
    context: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    definition.validate()
    historical = context["historical_geometry"]
    historical.validate()
    lower = np.asarray(historical.segment_lower, dtype=np.float64).copy()
    upper = np.asarray(historical.segment_upper, dtype=np.float64).copy()
    if definition.bound_mode == "robust_intersection":
        reference = context["stage_d_contract"].reference
        reference.validate()
        robust_lower = np.exp(
            np.asarray(reference.center_log, dtype=np.float64)
            - float(definition.lower_z) * np.asarray(reference.scale_log, dtype=np.float64)
        )
        upper_z = min(
            float(definition.upper_z),
            float(context["upper_gate"].upper_threshold),
        )
        robust_upper = np.exp(
            np.asarray(reference.center_log, dtype=np.float64)
            + upper_z * np.asarray(reference.scale_log, dtype=np.float64)
        )
        lower = np.maximum(lower, robust_lower)
        upper = np.minimum(upper, robust_upper)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ConstrainedIntegratorError("integrator segment bounds are non-finite")
    if np.any(lower <= 0.0) or np.any(upper <= lower):
        raise ConstrainedIntegratorError("integrator segment bounds are empty")
    return {
        "lower": lower.astype(np.float64),
        "upper": upper.astype(np.float64),
        "lower_sha256": sha256_array(lower.astype(np.float64)),
        "upper_sha256": sha256_array(upper.astype(np.float64)),
    }


def _coordinate_recenter(points: np.ndarray, *, absolute_limit: float) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(points, dtype=np.float64).copy()
    minimum = np.min(value, axis=2)
    maximum = np.max(value, axis=2)
    span = maximum - minimum
    possible = np.all(span <= 2.0 * float(absolute_limit), axis=(1, 2))
    lower_shift = -float(absolute_limit) - minimum
    upper_shift = float(absolute_limit) - maximum
    zero = np.zeros_like(lower_shift)
    shift = np.minimum(np.maximum(zero, lower_shift), upper_shift)
    value += shift[:, :, None, :]
    return value, possible


def reconstruct_segment_vectors(
    *,
    proposed: np.ndarray,
    control: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coordinate_abs_max: float,
    epsilon: float,
) -> Dict[str, Any]:
    proposed_raw = np.asarray(proposed, dtype=np.float64)
    control_raw = np.asarray(control, dtype=np.float64)
    if proposed_raw.shape != control_raw.shape or proposed_raw.ndim != 3:
        raise ValueError("proposed/control shape mismatch")
    if proposed_raw.shape[1:] != (stageb.FUTURE_STEPS, stageb.CABLE_DIM):
        raise ValueError("cable output shape changed")
    proposed_points = stageb.cable_points(proposed_raw).astype(np.float64)
    control_points = stageb.cable_points(control_raw).astype(np.float64)
    vectors = proposed_points[:, :, 1:, :] - proposed_points[:, :, :-1, :]
    control_vectors = control_points[:, :, 1:, :] - control_points[:, :, :-1, :]
    norms = np.linalg.norm(vectors, axis=-1)
    control_norms = np.linalg.norm(control_vectors, axis=-1)
    direction = np.empty_like(vectors)
    regular = norms > float(epsilon)
    direction[regular] = vectors[regular] / norms[regular, None]
    fallback = (~regular) & (control_norms > float(epsilon))
    direction[fallback] = control_vectors[fallback] / control_norms[fallback, None]
    degenerate = ~(regular | fallback)
    if np.any(degenerate):
        direction[degenerate] = np.asarray([1.0, 0.0], dtype=np.float64)
    clipped = np.clip(norms, lower[None], upper[None])
    clipped_vectors = direction * clipped[..., None]
    reconstructed = np.zeros_like(proposed_points)
    reconstructed[:, :, 1:, :] = np.cumsum(clipped_vectors, axis=2)
    proposed_centroid = np.mean(proposed_points, axis=2)
    reconstructed_centroid = np.mean(reconstructed, axis=2)
    reconstructed += (proposed_centroid - reconstructed_centroid)[:, :, None, :]
    reconstructed, coordinate_possible = _coordinate_recenter(
        reconstructed,
        absolute_limit=float(coordinate_abs_max),
    )
    result = reconstructed.reshape(proposed_raw.shape).astype(np.float32)
    lengths = stageb.segment_lengths(result).astype(np.float64)
    bound_pass = np.all(
        (lengths >= lower[None] - 5.0 * float(epsilon))
        & (lengths <= upper[None] + 5.0 * float(epsilon)),
        axis=(1, 2),
    )
    if not np.all(np.isfinite(result)):
        raise ConstrainedIntegratorError("reconstructed candidate is non-finite")
    return {
        "candidate": result,
        "coordinate_possible": coordinate_possible.astype(np.bool_),
        "bound_pass": bound_pass.astype(np.bool_),
        "degenerate_segment_rate": float(np.mean(degenerate)),
        "clipped_segment_rate": float(
            np.mean((norms < lower[None]) | (norms > upper[None]))
        ),
    }


def observable_row_metrics(
    *,
    candidate: np.ndarray,
    raw_proposal: np.ndarray,
    control: np.ndarray,
    direction: np.ndarray,
    definition: IntegratorDefinition,
    context: Mapping[str, Any],
    spec: ConstrainedIntegratorSpec,
    bound_pass: Optional[np.ndarray] = None,
    coordinate_possible: Optional[np.ndarray] = None,
    include_topology: bool = True,
) -> Dict[str, Any]:
    value = np.asarray(candidate, dtype=np.float32)
    control_raw = np.asarray(control, dtype=np.float32)
    proposal = np.asarray(raw_proposal, dtype=np.float32)
    direction_raw = np.asarray(direction, dtype=np.float64)
    rows = value.shape[0]
    scores = staged.segment_scores(value, context["stage_d_contract"].reference)
    upper_pass = scores["upper"][:, 0] <= float(context["upper_gate"].upper_threshold)
    lower_pass = scores["lower"][:, 0] <= float(context["stage_d_contract"].lower_threshold)
    historical = context["historical_geometry"]
    finite = np.all(np.isfinite(value), axis=(1, 2))
    coordinate_physical = (
        np.max(np.abs(value), axis=(1, 2))
        <= float(historical.coordinate_abs_max)
    )
    lengths = stageb.segment_lengths(value).astype(np.float64)
    segment_physical = np.all(
        (lengths >= np.asarray(historical.segment_lower, dtype=np.float64)[None])
        & (lengths <= np.asarray(historical.segment_upper, dtype=np.float64)[None]),
        axis=(1, 2),
    )
    if include_topology:
        topology_physical = stageb.physical_validity(
            value[:, None],
            historical,
        )["topology"][:, 0]
    else:
        topology_physical = np.ones(rows, dtype=np.bool_)
    physical_pass = (
        finite
        & coordinate_physical
        & segment_physical
        & topology_physical
    )
    actual_move = value.astype(np.float64) - control_raw.astype(np.float64)
    proposed_move = proposal.astype(np.float64) - control_raw.astype(np.float64)
    retention = _row_cosine(actual_move, direction_raw, epsilon=spec.standardizer_epsilon)
    move_norm = np.sqrt(np.sum(actual_move.reshape(rows, -1) ** 2, axis=1))
    proposed_norm = np.sqrt(np.sum(proposed_move.reshape(rows, -1) ** 2, axis=1))
    correction_norm = np.sqrt(
        np.sum((value.astype(np.float64) - proposal.astype(np.float64)).reshape(rows, -1) ** 2, axis=1)
    )
    correction_ratio = correction_norm / np.maximum(proposed_norm, spec.standardizer_epsilon)
    if definition.direction_source_id == "zero":
        retention_pass = np.ones(rows, dtype=np.bool_)
    else:
        retention_pass = retention >= float(definition.retention_min)
    nonzero = move_norm > float(spec.standardizer_epsilon)
    if definition.direction_source_id == "zero":
        nonzero = np.ones(rows, dtype=np.bool_)
    bound = (
        np.ones(rows, dtype=np.bool_)
        if bound_pass is None
        else np.asarray(bound_pass, dtype=np.bool_)
    )
    coordinate = (
        np.ones(rows, dtype=np.bool_)
        if coordinate_possible is None
        else np.asarray(coordinate_possible, dtype=np.bool_)
    )
    feasible = (
        finite
        & upper_pass
        & lower_pass
        & physical_pass
        & retention_pass
        & nonzero
        & bound
        & coordinate
    )
    return {
        "feasible": feasible,
        "finite": finite,
        "upper_pass": upper_pass,
        "lower_pass": lower_pass,
        "physical_pass": physical_pass,
        "coordinate_pass": coordinate_physical & coordinate,
        "segment_pass": segment_physical & bound,
        "topology_pass": topology_physical,
        "topology_evaluated": bool(include_topology),
        "retention_pass": retention_pass,
        "nonzero_move": nonzero,
        "direction_retention": retention,
        "move_norm": move_norm,
        "proposed_norm": proposed_norm,
        "correction_ratio": correction_ratio,
    }



PREDICATE_CALLBACK_SCHEMA = "phase314b_r258_stagek_integrator_predicate_event_v1"
PREDICATE_ORDER: Tuple[str, ...] = (
    "finite_state",
    "upper_segment_geometry",
    "lower_segment_geometry",
    "coordinate_recenter",
    "coordinate_geometry",
    "reconstruction_bounds",
    "segment_geometry",
    "direction_retention",
    "displacement",
    "topology",
)
PredicateTelemetryCallback = Callable[[Mapping[str, Any]], None]


def _freeze_predicate_telemetry_value(value: Any) -> Any:
    """Return a recursively immutable scalar-only callback payload."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_predicate_telemetry_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_predicate_telemetry_value(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze_predicate_telemetry_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConstrainedIntegratorError("non-finite predicate telemetry scalar")
        return value
    raise TypeError(
        "predicate telemetry contains a non-scalar value: {}".format(type(value))
    )


def _predicate_rate(mask: np.ndarray, active: np.ndarray) -> float:
    active_mask = np.asarray(active, dtype=np.bool_)
    count = int(np.count_nonzero(active_mask))
    if count == 0:
        return 1.0
    values = np.asarray(mask, dtype=np.bool_)
    return float(np.mean(values[active_mask]))


def _predicate_telemetry_attempt_event(
    *,
    definition: IntegratorDefinition,
    scale: float,
    selected_before: np.ndarray,
    metrics: Mapping[str, Any],
    bound_pass: Optional[np.ndarray],
    coordinate_possible: Optional[np.ndarray],
    topology_candidates: np.ndarray,
    topology_pass: np.ndarray,
    choose: np.ndarray,
) -> Mapping[str, Any]:
    selected_mask = np.asarray(selected_before, dtype=np.bool_)
    active = ~selected_mask
    rows = int(active.shape[0])
    coordinate_recenter = (
        np.ones(rows, dtype=np.bool_)
        if coordinate_possible is None
        else np.asarray(coordinate_possible, dtype=np.bool_)
    )
    reconstruction_bounds = (
        np.ones(rows, dtype=np.bool_)
        if bound_pass is None
        else np.asarray(bound_pass, dtype=np.bool_)
    )
    coordinate_geometry = np.where(
        coordinate_recenter,
        np.asarray(metrics["coordinate_pass"], dtype=np.bool_),
        True,
    )
    segment_geometry = np.where(
        reconstruction_bounds,
        np.asarray(metrics["segment_pass"], dtype=np.bool_),
        True,
    )
    predicate_masks: Dict[str, np.ndarray] = {
        "finite_state": np.asarray(metrics["finite"], dtype=np.bool_),
        "upper_segment_geometry": np.asarray(metrics["upper_pass"], dtype=np.bool_),
        "lower_segment_geometry": np.asarray(metrics["lower_pass"], dtype=np.bool_),
        "coordinate_recenter": coordinate_recenter,
        "coordinate_geometry": coordinate_geometry,
        "reconstruction_bounds": reconstruction_bounds,
        "segment_geometry": segment_geometry,
        "direction_retention": np.asarray(metrics["retention_pass"], dtype=np.bool_),
        "displacement": np.asarray(metrics["nonzero_move"], dtype=np.bool_),
    }
    expected_fast = active.copy()
    for name in PREDICATE_ORDER[:-1]:
        expected_fast &= predicate_masks[name]
    actual_fast = np.asarray(topology_candidates, dtype=np.bool_)
    if not np.array_equal(expected_fast, actual_fast):
        raise ConstrainedIntegratorError(
            "predicate telemetry fast-feasible decomposition changed"
        )
    topology_values = np.asarray(topology_pass, dtype=np.bool_)
    accepted = np.asarray(choose, dtype=np.bool_)
    if not np.array_equal(actual_fast & topology_values, accepted):
        raise ConstrainedIntegratorError(
            "predicate telemetry topology/acceptance decomposition changed"
        )

    unresolved = active.copy()
    first_failed_counts: Dict[str, int] = {}
    first_failed_rates: Dict[str, float] = {}
    active_count = int(np.count_nonzero(active))
    for name in PREDICATE_ORDER[:-1]:
        failures = unresolved & ~predicate_masks[name]
        count = int(np.count_nonzero(failures))
        first_failed_counts[name] = count
        first_failed_rates[name] = 0.0 if active_count == 0 else float(count / active_count)
        unresolved &= predicate_masks[name]
    topology_failures = unresolved & ~topology_values
    topology_count = int(np.count_nonzero(topology_failures))
    first_failed_counts["topology"] = topology_count
    first_failed_rates["topology"] = (
        0.0 if active_count == 0 else float(topology_count / active_count)
    )
    accepted_count = int(np.count_nonzero(accepted))
    if sum(first_failed_counts.values()) + accepted_count != active_count:
        raise ConstrainedIntegratorError(
            "predicate telemetry first-failure population does not close"
        )

    pass_counts = {
        name: int(np.count_nonzero(np.asarray(mask, dtype=np.bool_) & active))
        for name, mask in predicate_masks.items()
    }
    pass_rates = {
        name: _predicate_rate(mask, active)
        for name, mask in predicate_masks.items()
    }
    topology_checked_count = int(np.count_nonzero(actual_fast))
    topology_pass_count = int(np.count_nonzero(topology_values & actual_fast))
    pass_counts["topology"] = topology_pass_count
    pass_rates["topology"] = (
        1.0
        if topology_checked_count == 0
        else float(np.mean(topology_values[actual_fast]))
    )
    event = {
        "schema": PREDICATE_CALLBACK_SCHEMA,
        "event_type": "scale_attempt",
        "candidate_id": definition.candidate_id,
        "direction_source_id": definition.direction_source_id,
        "integration_mode": definition.integration_mode,
        "bound_mode": definition.bound_mode,
        "attempted_scale": float(scale),
        "row_count": rows,
        "active_row_count": active_count,
        "already_selected_count": int(np.count_nonzero(selected_mask)),
        "fast_feasible_count": topology_checked_count,
        "topology_checked_count": topology_checked_count,
        "accepted_count": accepted_count,
        "accepted_rate_over_active": (
            0.0 if active_count == 0 else float(accepted_count / active_count)
        ),
        "predicate_order": PREDICATE_ORDER,
        "predicate_pass_counts": pass_counts,
        "predicate_pass_rates": pass_rates,
        "first_failed_counts": first_failed_counts,
        "first_failed_rates": first_failed_rates,
        "all_active_rows_accounted_for": True,
        "target_used": False,
    }
    return _freeze_predicate_telemetry_value(event)


def _predicate_telemetry_final_event(
    *,
    definition: IntegratorDefinition,
    selected: np.ndarray,
    selected_scale: np.ndarray,
    final_metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected_mask = np.asarray(selected, dtype=np.bool_)
    scales = np.asarray(selected_scale, dtype=np.float64)
    event = {
        "schema": PREDICATE_CALLBACK_SCHEMA,
        "event_type": "final_summary",
        "candidate_id": definition.candidate_id,
        "direction_source_id": definition.direction_source_id,
        "integration_mode": definition.integration_mode,
        "bound_mode": definition.bound_mode,
        "row_count": int(selected_mask.shape[0]),
        "selected_count": int(np.count_nonzero(selected_mask)),
        "fallback_count": int(np.count_nonzero(~selected_mask)),
        "selected_nonzero_rate": float(np.mean(selected_mask)),
        "selected_scale_minimum": float(np.min(scales)),
        "selected_scale_mean": float(np.mean(scales)),
        "selected_scale_maximum": float(np.max(scales)),
        "final_upper_pass_rate": float(np.mean(final_metrics["upper_pass"])),
        "final_lower_pass_rate": float(np.mean(final_metrics["lower_pass"])),
        "final_physical_pass_rate": float(np.mean(final_metrics["physical_pass"])),
        "final_topology_pass_rate": float(np.mean(final_metrics["topology_pass"])),
        "target_used": False,
    }
    return _freeze_predicate_telemetry_value(event)


def _emit_predicate_telemetry(
    callback: PredicateTelemetryCallback,
    event: Mapping[str, Any],
) -> None:
    callback(event)


def integrate_rowwise(
    *,
    control: np.ndarray,
    direction: np.ndarray,
    definition: IntegratorDefinition,
    context: Mapping[str, Any],
    spec: ConstrainedIntegratorSpec,
    predicate_callback: Optional[PredicateTelemetryCallback] = None,
) -> Dict[str, Any]:
    definition.validate()
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(direction, dtype=np.float64)
    if direction_raw.shape != control_raw.shape:
        raise ValueError("direction/control shape mismatch")
    rows = control_raw.shape[0]
    selected = np.zeros(rows, dtype=np.bool_)
    selected_scale = np.zeros(rows, dtype=np.float64)
    output = control_raw.copy()
    selected_raw_proposal = control_raw.copy()
    selected_clipped_rate = np.zeros(rows, dtype=np.float64)
    selected_degenerate_rate = np.zeros(rows, dtype=np.float64)
    bounds = (
        None
        if definition.bound_mode == "none"
        else segment_bounds(definition=definition, context=context)
    )
    candidate_records = []
    scales = [
        float(scale)
        for scale in spec.scale_grid
        if float(scale) <= float(definition.maximum_scale) + spec.standardizer_epsilon
    ]
    for scale in sorted(scales, reverse=True):
        raw_proposal = (control_raw.astype(np.float64) + scale * direction_raw).astype(np.float32)
        if definition.integration_mode == "linear":
            candidate = raw_proposal
            bound_pass = None
            coordinate_possible = None
            clipped_rate = 0.0
            degenerate_rate = 0.0
        else:
            reconstruction = reconstruct_segment_vectors(
                proposed=raw_proposal,
                control=control_raw,
                lower=bounds["lower"],
                upper=bounds["upper"],
                coordinate_abs_max=float(context["historical_geometry"].coordinate_abs_max),
                epsilon=spec.segment_tolerance,
            )
            candidate = reconstruction["candidate"]
            bound_pass = reconstruction["bound_pass"]
            coordinate_possible = reconstruction["coordinate_possible"]
            clipped_rate = float(reconstruction["clipped_segment_rate"])
            degenerate_rate = float(reconstruction["degenerate_segment_rate"])
        metrics = observable_row_metrics(
            candidate=candidate,
            raw_proposal=raw_proposal,
            control=control_raw,
            direction=direction_raw,
            definition=definition,
            context=context,
            spec=spec,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
            include_topology=False,
        )
        topology_candidates = (
            (~selected)
            & np.asarray(metrics["feasible"], dtype=np.bool_)
        )
        topology_pass = np.zeros(rows, dtype=np.bool_)
        topology_checked = np.flatnonzero(topology_candidates)
        if topology_checked.size:
            topology_pass[topology_checked] = stageb.physical_validity(
                candidate[topology_checked, None],
                context["historical_geometry"],
            )["topology"][:, 0]
        choose = topology_candidates & topology_pass
        if predicate_callback is not None:
            _emit_predicate_telemetry(
                predicate_callback,
                _predicate_telemetry_attempt_event(
                    definition=definition,
                    scale=scale,
                    selected_before=selected,
                    metrics=metrics,
                    bound_pass=bound_pass,
                    coordinate_possible=coordinate_possible,
                    topology_candidates=topology_candidates,
                    topology_pass=topology_pass,
                    choose=choose,
                ),
            )
        output[choose] = candidate[choose]
        selected_raw_proposal[choose] = raw_proposal[choose]
        selected_scale[choose] = scale
        selected_clipped_rate[choose] = clipped_rate
        selected_degenerate_rate[choose] = degenerate_rate
        selected[choose] = True
        candidate_records.append(
            {
                "scale": scale,
                "feasible_rate": float(np.mean(metrics["feasible"])),
                "upper_pass_rate": float(np.mean(metrics["upper_pass"])),
                "fast_physical_pass_rate": float(np.mean(metrics["physical_pass"])),
                "topology_checked_rate": float(np.mean(topology_candidates)),
                "topology_pass_rate_among_checked": (
                    1.0
                    if topology_checked.size == 0
                    else float(np.mean(topology_pass[topology_checked]))
                ),
                "accepted_rate": float(np.mean(choose)),
                "retention_pass_rate": float(np.mean(metrics["retention_pass"])),
                "candidate_sha256": sha256_array(candidate),
            }
        )
    final_metrics = observable_row_metrics(
        candidate=output,
        raw_proposal=selected_raw_proposal,
        control=control_raw,
        direction=direction_raw,
        definition=definition,
        context=context,
        spec=spec,
        include_topology=True,
    )
    fallback = ~selected
    if predicate_callback is not None:
        _emit_predicate_telemetry(
            predicate_callback,
            _predicate_telemetry_final_event(
                definition=definition,
                selected=selected,
                selected_scale=selected_scale,
                final_metrics=final_metrics,
            ),
        )
    return {
        "candidate": output,
        "candidate_sha256": sha256_array(output),
        "candidate_id": definition.candidate_id,
        "direction_source_id": definition.direction_source_id,
        "candidate_role": definition.role,
        "integration_mode": definition.integration_mode,
        "bound_mode": definition.bound_mode,
        "selected_scale": selected_scale,
        "selected_scale_sha256": sha256_array(selected_scale),
        "selected_scale_stats": _safe_stats(selected_scale),
        "selected_nonzero_rate": float(np.mean(selected)),
        "fallback_rate": float(np.mean(fallback)),
        "observable_feasible_rate": float(np.mean(selected)),
        "final_upper_pass_rate": float(np.mean(final_metrics["upper_pass"])),
        "final_lower_pass_rate": float(np.mean(final_metrics["lower_pass"])),
        "final_physical_pass_rate": float(np.mean(final_metrics["physical_pass"])),
        "final_topology_pass_rate": float(np.mean(final_metrics["topology_pass"])),
        "direction_retention": _safe_stats(final_metrics["direction_retention"]),
        "correction_ratio": _safe_stats(final_metrics["correction_ratio"]),
        "clipped_segment_rate": _safe_stats(selected_clipped_rate),
        "degenerate_segment_rate": _safe_stats(selected_degenerate_rate),
        "scale_candidates": candidate_records,
        "bounds": (
            None
            if bounds is None
            else {
                key: value
                for key, value in bounds.items()
                if key.endswith("sha256")
            }
        ),
        "target_used_for_generation": False,
    }


def integrator_observable_gates(
    integration: Mapping[str, Any],
    *,
    spec: ConstrainedIntegratorSpec,
) -> Dict[str, bool]:
    gates = {
        "feasible_rate": bool(
            float(integration["observable_feasible_rate"])
            >= spec.observable_feasible_rate_min
        ),
        "fallback_rate": bool(
            float(integration["fallback_rate"])
            <= spec.fallback_rate_max
        ),
        "retention": bool(
            integration.get("direction_source_id") == "zero"
            or float(integration["direction_retention"]["mean"])
            >= spec.mean_retention_min
        ),
        "correction_ratio": bool(
            float(integration["correction_ratio"]["p95"])
            <= spec.correction_ratio_p95_max
        ),
        "upper": bool(float(integration["final_upper_pass_rate"]) >= 0.90),
        "physical": bool(float(integration["final_physical_pass_rate"]) >= 0.90),
    }
    gates["all"] = bool(all(gates.values()))
    return gates


def evaluate_integrated_candidate(
    *,
    integration: Mapping[str, Any],
    control: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    timestep: int,
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: ConstrainedIntegratorSpec,
) -> Dict[str, Any]:
    evaluation = stageb258.evaluate_output_population(
        np.asarray(integration["candidate"], dtype=np.float32),
        control=control,
        target=target,
        groups=groups,
        condition_name=condition_name,
        target_standardizer=context["target_standardizer"],
        objective_contract=context["objective_contract"],
        upper_gate=context["upper_gate"],
        stage_d_contract=context["stage_d_contract"],
        historical_geometry=context["historical_geometry"],
        top_k=16,
        epsilon=integrator_spec.standardizer_epsilon,
    )
    state_gates = staged258.state_scientific_gates(
        timestep=int(timestep),
        evaluation=evaluation,
        spec=direction_spec,
    )
    observable_gates = integrator_observable_gates(integration, spec=integrator_spec)
    return {
        "evaluation": evaluation,
        "state_gates": state_gates,
        "observable_gates": observable_gates,
        "scientific_pass": bool(state_gates["all"] and observable_gates["all"]),
    }


def fit_direction_oof(
    *,
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
) -> Dict[str, Any]:
    baseline_records: Dict[str, Dict[str, Any]] = {}
    baseline_predictions: Dict[str, Dict[int, np.ndarray]] = {}
    for candidate_id in ("global_mean", "condition_mean"):
        definition = definition_by_id(candidate_id)
        baseline_records[candidate_id] = {}
        baseline_predictions[candidate_id] = {}
        for timestep in direction_spec.timesteps:
            control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
            target = np.asarray(context["objective_target"], dtype=np.float32)
            residual = staged258.active_residual(
                control=control,
                target=target,
                target_standardizer=context["target_standardizer"],
            )
            features = staged258.build_surrogate_features(
                condition=context["objective_condition"],
                control=control,
                condition_name=context["objective_condition_name"],
                feature_mode=definition.feature_mode,
            )
            oof = staged258.fit_oof_predictions(
                definition=definition,
                features=features,
                target_residual=residual,
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                spec=direction_spec,
            )
            metrics = staged258.row_direction_metrics(
                predicted_residual=oof["prediction"],
                target_residual=residual,
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                epsilon=direction_spec.standardizer_epsilon,
            )
            baseline_predictions[candidate_id][int(timestep)] = oof["prediction"]
            baseline_records[candidate_id][str(timestep)] = {
                "prediction_sha256": oof["prediction_sha256"],
                "fold_records": oof["fold_records"],
                "direction_metrics": metrics,
            }
    baseline_means = {
        candidate_id: {
            timestep: float(record["direction_metrics"]["cosine"]["mean"])
            for timestep, record in timestep_records.items()
        }
        for candidate_id, timestep_records in baseline_records.items()
    }

    predictions: Dict[str, Dict[int, np.ndarray]] = {}
    records: Dict[str, Dict[str, Any]] = {}
    for candidate_id in DIRECTION_SOURCE_IDS:
        definition = definition_by_id(candidate_id)
        predictions[candidate_id] = {}
        records[candidate_id] = {
            "candidate_id": candidate_id,
            "definition": asdict(definition),
            "timestep_records": {},
        }
        for timestep in direction_spec.timesteps:
            control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
            target = np.asarray(context["objective_target"], dtype=np.float32)
            residual = staged258.active_residual(
                control=control,
                target=target,
                target_standardizer=context["target_standardizer"],
            )
            features = staged258.build_surrogate_features(
                condition=context["objective_condition"],
                control=control,
                condition_name=context["objective_condition_name"],
                feature_mode=definition.feature_mode,
            )
            oof = staged258.fit_oof_predictions(
                definition=definition,
                features=features,
                target_residual=residual,
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                spec=direction_spec,
            )
            metrics = staged258.row_direction_metrics(
                predicted_residual=oof["prediction"],
                target_residual=residual,
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                epsilon=direction_spec.standardizer_epsilon,
            )
            gates = staged258.direction_gate(
                metrics=metrics,
                global_baseline_mean=float(baseline_means["global_mean"][str(timestep)]),
                condition_baseline_mean=float(baseline_means["condition_mean"][str(timestep)]),
                spec=direction_spec,
            )
            if gates["all"] is not True:
                raise ConstrainedIntegratorError(
                    "Stage-E direction source no longer passes: {} t{}".format(candidate_id, timestep)
                )
            predictions[candidate_id][int(timestep)] = oof["prediction"]
            records[candidate_id]["timestep_records"][str(timestep)] = {
                "feature_sha256": sha256_array(features),
                "target_residual_sha256": sha256_array(residual),
                "oof": {key: value for key, value in oof.items() if key != "prediction"},
                "direction_metrics": metrics,
                "direction_gates": gates,
            }
    return {
        "baseline_records": baseline_records,
        "baseline_means": baseline_means,
        "direction_records": records,
        "predictions": predictions,
    }


def _direction_for_definition(
    *,
    definition: IntegratorDefinition,
    timestep: int,
    direction_bundle: Mapping[str, Any],
    control: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    if definition.direction_source_id == "zero":
        return np.zeros_like(control, dtype=np.float64)
    if definition.direction_source_id == "oracle":
        return np.asarray(target, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    return np.asarray(
        direction_bundle["predictions"][definition.direction_source_id][int(timestep)],
        dtype=np.float64,
    )


def integrator_candidate_record(
    *,
    definition: IntegratorDefinition,
    context: Mapping[str, Any],
    direction_bundle: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: ConstrainedIntegratorSpec,
) -> Dict[str, Any]:
    definition.validate()
    timestep_records = {}
    for timestep in integrator_spec.timesteps:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        target = np.asarray(context["objective_target"], dtype=np.float32)
        direction = _direction_for_definition(
            definition=definition,
            timestep=int(timestep),
            direction_bundle=direction_bundle,
            control=control,
            target=target,
        )
        integration = integrate_rowwise(
            control=control,
            direction=direction,
            definition=definition,
            context=context,
            spec=integrator_spec,
        )
        evaluated = evaluate_integrated_candidate(
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
        direction_record = (
            None
            if definition.direction_source_id in ("zero", "oracle")
            else copy.deepcopy(
                direction_bundle["direction_records"][definition.direction_source_id]["timestep_records"][str(timestep)]
            )
        )
        timestep_records[str(timestep)] = {
            "timestep": int(timestep),
            "direction_source_record": direction_record,
            "direction_sha256": sha256_array(direction),
            "integration": {key: value for key, value in integration.items() if key != "candidate"},
            **evaluated,
        }
    eligible = bool(
        definition.role == "selectable"
        and all(
            record["scientific_pass"]
            and record["direction_source_record"]["direction_gates"]["all"]
            for record in timestep_records.values()
        )
    )
    return {
        "candidate_id": definition.candidate_id,
        "definition": asdict(definition),
        "role": definition.role,
        "direction_uses_target": bool(definition.direction_source_id == "oracle"),
        "timestep_records": timestep_records,
        "eligible": eligible,
        "holdout_used": False,
    }


def _selection_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    items = [record["timestep_records"][str(timestep)] for timestep in (10, 25, 50)]
    if not record.get("eligible"):
        raise ConstrainedIntegratorError("selection key requires an eligible candidate")
    reductions = [
        float(item["evaluation"]["target_distance_reduction_fraction"]["mean"])
        for item in items
    ]
    physical = [float(item["evaluation"]["historical_physical_row_any_rate"]) for item in items]
    nmse = [float(item["evaluation"]["normalized_mse_ratio"]) for item in items]
    fallback = [float(item["integration"]["fallback_rate"]) for item in items]
    movement = [float(item["evaluation"]["target_direction_cosine"]["mean"]) for item in items]
    source = str(record["definition"]["direction_source_id"])
    reduced_rank = 0 if "rr32" in record["candidate_id"] else 1
    return (
        -min(reductions),
        -min(physical),
        max(nmse),
        max(fallback),
        -min(movement),
        reduced_rank,
        source,
        str(record["candidate_id"]),
    )


def select_integrator(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    eligible = [
        record
        for record in records
        if record.get("eligible") and record.get("role") == "selectable"
    ]
    if not eligible:
        return None
    selected = min(eligible, key=_selection_key)
    return {
        "candidate_id": selected["candidate_id"],
        "definition": copy.deepcopy(selected["definition"]),
        "selection_key": list(_selection_key(selected)),
        "selection_population": "six-fold grouped OOF objective train only",
        "holdout_used": False,
    }


def permutation_control(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    direction_bundle: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: ConstrainedIntegratorSpec,
) -> Dict[str, Any]:
    definition = IntegratorDefinition(**selected["definition"])
    source = definition_by_id(definition.direction_source_id)
    records = {}
    all_pass = []
    selected_record = next(
        record
        for record in context["integrator_candidate_records"]
        if record["candidate_id"] == selected["candidate_id"]
    )
    for timestep in integrator_spec.timesteps:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        target = np.asarray(context["objective_target"], dtype=np.float32)
        residual = staged258.active_residual(
            control=control,
            target=target,
            target_standardizer=context["target_standardizer"],
        )
        features = staged258.build_surrogate_features(
            condition=context["objective_condition"],
            control=control,
            condition_name=context["objective_condition_name"],
            feature_mode=source.feature_mode,
        )
        oof = staged258.fit_oof_predictions(
            definition=source,
            features=features,
            target_residual=residual,
            condition_name=context["objective_condition_name"],
            fold_assignment=context["objective_fold_assignment"],
            spec=direction_spec,
            permutation_seed=integrator_spec.permutation_seed + int(timestep) * 97,
        )
        metrics = staged258.row_direction_metrics(
            predicted_residual=oof["prediction"],
            target_residual=residual,
            condition_name=context["objective_condition_name"],
            fold_assignment=context["objective_fold_assignment"],
            epsilon=integrator_spec.standardizer_epsilon,
        )
        integration = integrate_rowwise(
            control=control,
            direction=oof["prediction"],
            definition=definition,
            context=context,
            spec=integrator_spec,
        )
        evaluated = evaluate_integrated_candidate(
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
        selected_mean = float(
            selected_record["timestep_records"][str(timestep)]["direction_source_record"]["direction_metrics"]["cosine"]["mean"]
        )
        permuted_mean = float(metrics["cosine"]["mean"])
        passed = bool(
            permuted_mean <= integrator_spec.permutation_cosine_max
            and selected_mean - permuted_mean >= integrator_spec.permutation_margin_min
            and not evaluated["scientific_pass"]
        )
        all_pass.append(passed)
        records[str(timestep)] = {
            "timestep": int(timestep),
            "oof": {key: value for key, value in oof.items() if key != "prediction"},
            "direction_metrics": metrics,
            "selected_mean_cosine": selected_mean,
            "permutation_mean_cosine": permuted_mean,
            "margin": float(selected_mean - permuted_mean),
            "integration": {key: value for key, value in integration.items() if key != "candidate"},
            "evaluation": evaluated,
            "pass": passed,
        }
    return {
        "candidate_id": selected["candidate_id"],
        "records": records,
        "all_pass": bool(all(all_pass)),
        "holdout_used": False,
    }


def projection_only_control(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: ConstrainedIntegratorSpec,
) -> Dict[str, Any]:
    selected_definition = IntegratorDefinition(**selected["definition"])
    definition = IntegratorDefinition(
        candidate_id="projection_only_locked_policy",
        direction_source_id="zero",
        integration_mode=selected_definition.integration_mode,
        bound_mode=selected_definition.bound_mode,
        lower_z=selected_definition.lower_z,
        upper_z=selected_definition.upper_z,
        retention_min=0.0,
        maximum_scale=selected_definition.maximum_scale,
        role="diagnostic_control",
    )
    records = {}
    passes = []
    for timestep in integrator_spec.timesteps:
        control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        target = np.asarray(context["objective_target"], dtype=np.float32)
        direction = np.zeros_like(control, dtype=np.float64)
        integration = integrate_rowwise(
            control=control,
            direction=direction,
            definition=definition,
            context=context,
            spec=integrator_spec,
        )
        evaluated = evaluate_integrated_candidate(
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
        passes.append(evaluated["scientific_pass"])
        records[str(timestep)] = {
            "integration": {key: value for key, value in integration.items() if key != "candidate"},
            **evaluated,
        }
    all_timesteps = bool(all(passes))
    return {
        "records": records,
        "all_timesteps_scientific_pass": all_timesteps,
        "pass": bool(not all_timesteps),
        "holdout_used": False,
    }


def fit_selected_and_evaluate_holdout(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    direction_spec: staged258.DirectionSurrogateSpec,
    integrator_spec: ConstrainedIntegratorSpec,
) -> Dict[str, Any]:
    integrator = IntegratorDefinition(**selected["definition"])
    source = definition_by_id(integrator.direction_source_id)
    records = {}
    direction_passes = []
    state_passes = []
    for timestep in integrator_spec.timesteps:
        objective_control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
        objective_target = np.asarray(context["objective_target"], dtype=np.float32)
        objective_residual = staged258.active_residual(
            control=objective_control,
            target=objective_target,
            target_standardizer=context["target_standardizer"],
        )
        train_features = staged258.build_surrogate_features(
            condition=context["objective_condition"],
            control=objective_control,
            condition_name=context["objective_condition_name"],
            feature_mode=source.feature_mode,
        )
        model = staged258.fit_surrogate_model(
            definition=source,
            features=train_features,
            target_residual=objective_residual,
            condition_name=context["objective_condition_name"],
            spec=direction_spec,
        )
        holdout_control = np.asarray(context["control_predictions"][int(timestep)], dtype=np.float32)
        holdout_target = np.asarray(context["holdout_target"], dtype=np.float32)
        holdout_residual = staged258.active_residual(
            control=holdout_control,
            target=holdout_target,
            target_standardizer=context["target_standardizer"],
        )
        holdout_features = staged258.build_surrogate_features(
            condition=context["holdout_condition"],
            control=holdout_control,
            condition_name=context["holdout_condition_name"],
            feature_mode=source.feature_mode,
        )
        prediction = staged258.predict_surrogate_model(
            definition=source,
            model=model,
            features=holdout_features,
            condition_name=context["holdout_condition_name"],
            target_shape=tuple(holdout_residual.shape[1:]),
        )
        direction_metrics = staged258.row_direction_metrics(
            predicted_residual=prediction,
            target_residual=holdout_residual,
            condition_name=context["holdout_condition_name"],
            fold_assignment=None,
            epsilon=integrator_spec.standardizer_epsilon,
        )
        direction_gates = staged258.holdout_direction_gate(
            metrics=direction_metrics,
            spec=direction_spec,
        )
        integration = integrate_rowwise(
            control=holdout_control,
            direction=prediction,
            definition=integrator,
            context=context,
            spec=integrator_spec,
        )
        evaluated = evaluate_integrated_candidate(
            integration=integration,
            control=holdout_control,
            target=holdout_target,
            groups=context["holdout_groups"],
            condition_name=context["holdout_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=direction_spec,
            integrator_spec=integrator_spec,
        )
        direction_passes.append(direction_gates["all"])
        state_passes.append(evaluated["scientific_pass"])
        records[str(timestep)] = {
            "timestep": int(timestep),
            "model_identity": copy.deepcopy(model["identity"]),
            "holdout_feature_sha256": sha256_array(holdout_features),
            "holdout_target_residual_sha256": sha256_array(holdout_residual),
            "prediction_sha256": sha256_array(prediction),
            "direction_metrics": direction_metrics,
            "direction_gates": direction_gates,
            "integration": {key: value for key, value in integration.items() if key != "candidate"},
            **evaluated,
        }
    return {
        "candidate_id": selected["candidate_id"],
        "definition": copy.deepcopy(selected["definition"]),
        "timestep_records": records,
        "direction_all_timesteps": bool(all(direction_passes)),
        "state_all_timesteps": bool(all(state_passes)),
        "scientific_pass": bool(all(direction_passes) and all(state_passes)),
        "selection_changed_after_holdout": False,
        "holdout_used_for_fit": False,
        "holdout_used_for_selection": False,
        "holdout_evaluated": True,
        "surrogate_weights_persisted": False,
        "prediction_tensor_persisted": False,
    }


def classify_integrator(
    *,
    records: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    permutation: Optional[Mapping[str, Any]],
    projection_only: Optional[Mapping[str, Any]],
    holdout: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    selectable = [record for record in records if record["role"] == "selectable"]
    oracle = next(record for record in records if record["role"] == "oracle_control")
    oracle_all = bool(
        all(item["scientific_pass"] for item in oracle["timestep_records"].values())
    )
    state_all_ids = [
        record["candidate_id"]
        for record in selectable
        if all(item["scientific_pass"] for item in record["timestep_records"].values())
    ]
    observable_all_ids = [
        record["candidate_id"]
        for record in selectable
        if all(item["observable_gates"]["all"] for item in record["timestep_records"].values())
    ]
    eligible_ids = [record["candidate_id"] for record in selectable if record["eligible"]]
    if selected is None:
        if not oracle_all:
            root = "phase314b_r258_stagee_constrained_integrator_fails_oracle_direction"
            next_path = "REDESIGN_NONLINEAR_CABLE_MANIFOLD_PROJECTION"
            locus = "integrator_family_not_oracle_reachable"
        elif not state_all_ids:
            root = "phase314b_r258_stagee_oracle_valid_but_surrogate_direction_not_integrable"
            next_path = "CALIBRATE_CONSTRAINT_AWARE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY"
            locus = "surrogate_precision_under_constraints"
        else:
            root = "phase314b_r258_stagee_state_reachability_requires_non_deployable_policy"
            next_path = "REDESIGN_TARGET_FREE_INTEGRATOR_SELECTION_POLICY"
            locus = "observable_policy_gate"
    elif permutation is None or not permutation["all_pass"]:
        root = "phase314b_r258_stagee_direction_permutation_control_indicates_shortcut"
        next_path = "AUDIT_CONSTRAINED_INTEGRATOR_DIRECTION_LEAKAGE"
        locus = "direction_permutation_shortcut"
    elif projection_only is None or not projection_only["pass"]:
        root = "phase314b_r258_stagee_projection_only_control_explains_state_witness"
        next_path = "AUDIT_CONSTRAINT_ONLY_STATE_SHORTCUT"
        locus = "projection_only_shortcut"
    elif holdout is None:
        raise ConstrainedIntegratorError("selected integrator lacks locked holdout evaluation")
    elif not holdout["direction_all_timesteps"]:
        root = "phase314b_r258_stagee_locked_direction_does_not_generalize_to_holdout"
        next_path = "AUDIT_NONLINEAR_SEQUENCE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY"
        locus = "holdout_direction_generalization"
    elif not holdout["state_all_timesteps"]:
        root = "phase314b_r258_stagee_constrained_integrator_does_not_generalize_to_holdout"
        next_path = "CALIBRATE_CONSTRAINT_AWARE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY"
        locus = "holdout_state_generalization"
    else:
        root = "phase314b_r258_stagee_constrained_integrator_selected_and_holdout_validated"
        next_path = "CALIBRATE_SELECTED_CONSTRAINED_INTEGRATOR_IN_TRAIN_ONLY_DENOISER"
        locus = "constrained_integrator_validated"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "selectable_candidate_count": len(selectable),
        "eligible_candidate_ids": eligible_ids,
        "state_all_candidate_ids": state_all_ids,
        "observable_all_candidate_ids": observable_all_ids,
        "oracle_all_timesteps": oracle_all,
        "selected_configuration": copy.deepcopy(selected),
        "permutation_pass": None if permutation is None else bool(permutation["all_pass"]),
        "projection_only_pass": None if projection_only is None else bool(projection_only["pass"]),
        "holdout_scientific_pass": None if holdout is None else bool(holdout["scientific_pass"]),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    integrator_spec: Optional[ConstrainedIntegratorSpec] = None,
    direction_spec: Optional[staged258.DirectionSurrogateSpec] = None,
) -> Dict[str, Any]:
    active_integrator = ConstrainedIntegratorSpec() if integrator_spec is None else integrator_spec
    active_direction = staged258.DirectionSurrogateSpec() if direction_spec is None else direction_spec
    active_integrator.validate()
    active_direction.validate()
    repository_root = Path(root).resolve()
    base = validate_base_evidence(repository_root)
    stagec258.stagea258.validate_environment_payload(environment)
    if environment.get("compatibility_sha256") != EXPECTED_COMPATIBILITY_SHA256:
        raise ConstrainedIntegratorError("portable compatibility SHA changed")
    cold = stagec258.stagea258.assert_cold_cuda_context_portable()
    captured = stageb258.capture_portable_control_model(root=repository_root)
    context = staged258.build_direction_context(
        root=repository_root,
        captured=captured,
        spec=active_direction,
    )
    direction_bundle = fit_direction_oof(context=context, direction_spec=active_direction)
    candidate_records = [
        integrator_candidate_record(
            definition=definition,
            context=context,
            direction_bundle=direction_bundle,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
        )
        for definition in INTEGRATOR_DEFINITIONS
    ]
    selected = select_integrator(candidate_records)
    context["integrator_candidate_records"] = candidate_records
    permutation = None
    projection_only = None
    holdout = None
    if selected is not None:
        permutation = permutation_control(
            selected=selected,
            context=context,
            direction_bundle=direction_bundle,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
        )
        projection_only = projection_only_control(
            selected=selected,
            context=context,
            direction_spec=active_direction,
            integrator_spec=active_integrator,
        )
        if permutation["all_pass"] and projection_only["pass"]:
            holdout = fit_selected_and_evaluate_holdout(
                selected=selected,
                context=context,
                direction_spec=active_direction,
                integrator_spec=active_integrator,
            )
    classification = classify_integrator(
        records=candidate_records,
        selected=selected,
        permutation=permutation,
        projection_only=projection_only,
        holdout=holdout,
    )
    validated = (
        copy.deepcopy(selected)
        if (
            selected is not None
            and permutation is not None
            and permutation["all_pass"]
            and projection_only is not None
            and projection_only["pass"]
            and holdout is not None
            and holdout["scientific_pass"]
        )
        else None
    )
    contract = {
        "schema": "phase314b_r258_stagee_constrained_integrator_contract_v1",
        "phase": PHASE,
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "base_identity": {
            "worker_sha256": EXPECTED_BASE_WORKER_SHA256,
            "contract_sha256": EXPECTED_BASE_CONTRACT_SHA256,
            "selection_sha256": EXPECTED_BASE_SELECTION_SHA256,
            "root_cause": EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": EXPECTED_BASE_NEXT_PATH,
        },
        "environment": dict(environment),
        "cold_main_worker_context": cold,
        "control_capture": captured["control_identity"],
        "direction_surrogate_spec": asdict(active_direction),
        "constrained_integrator_spec": asdict(active_integrator),
        "integrator_definitions": [asdict(definition) for definition in INTEGRATOR_DEFINITIONS],
        "direction_source_ids": list(DIRECTION_SOURCE_IDS),
        "generation_contract": {
            "selectable_candidates_use_ground_truth_target": False,
            "oracle_control_uses_ground_truth_target": True,
            "oracle_control_selectable": False,
            "uses_selection_holdout": False,
            "uses_frozen_probe": False,
            "row_scale_policy": "largest observable-feasible pre-registered scale",
            "segment_projection": (
                "clip proposed ordered segment-vector lengths to frozen bounds, "
                "preserve segment directions and proposed horizon centroids"
            ),
            "coordinate_policy": "minimum translation required to enter frozen coordinate box",
            "topology_policy": (
                "evaluate topology only for rows passing vectorized observable gates; "
                "retry smaller scales after topology rejection"
            ),
            "fallback": "unchanged diffusion control",
            "zero_direction_control": (
                "direction-retention aggregate gate exempt; all geometry, physical, "
                "fidelity, and target-relative scientific gates remain active"
            ),
        },
        "evaluation_contract": {
            "target_relative_collapse_stretch": "evaluation only",
            "target_direction_cosine": "evaluation only",
            "target_distance_reduction": "evaluation only",
            "candidate_selection_population": "six-fold grouped OOF objective train only",
            "holdout_access": "one locked final evaluation only",
        },
        "source_sha256": base["source_sha256"],
        "diffusion_model_candidate_trained": False,
        "model_architecture_changed": False,
        "frozen_probe_accessed": False,
        "selection_holdout_changes_selection": False,
    }
    contract["contract_sha256"] = sha256_bytes(stable_json_bytes(contract))
    selection_payload = {
        "schema": "phase314b_r258_stagee_selection_v1",
        "phase": PHASE,
        "selected_configuration": copy.deepcopy(selected),
        "validated_train_only_recommendation": copy.deepcopy(validated),
        "classification": classification,
        "candidate_ids": [record["candidate_id"] for record in candidate_records],
        "permutation_pass": None if permutation is None else permutation["all_pass"],
        "projection_only_pass": None if projection_only is None else projection_only["pass"],
        "holdout_scientific_pass": None if holdout is None else holdout["scientific_pass"],
    }
    selection_payload["selection_sha256"] = sha256_bytes(stable_json_bytes(selection_payload))
    ready = validated is not None
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagee_result_v1",
        "verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "immutable_inputs": {
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "base_worker_sha256": EXPECTED_BASE_WORKER_SHA256,
            "base_contract_sha256": EXPECTED_BASE_CONTRACT_SHA256,
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
        "direction_bundle": {
            "baseline_records": direction_bundle["baseline_records"],
            "baseline_means": direction_bundle["baseline_means"],
            "direction_records": direction_bundle["direction_records"],
        },
        "integrator_candidate_records": candidate_records,
        "objective_train_selected_configuration": copy.deepcopy(selected),
        "permutation_control": permutation,
        "projection_only_control": projection_only,
        "locked_holdout_evaluation": holdout,
        "classification": classification,
        "constrained_integrator_contract": contract,
        "selection": selection_payload,
        "selected_configuration": copy.deepcopy(validated),
        "train_only_recommendation": copy.deepcopy(validated),
        "selection_holdout_evaluated": bool(holdout is not None),
        "selectable_generation_uses_ground_truth_target": False,
        "oracle_control_uses_ground_truth_target": True,
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
    """Return the complete deterministic worker identity."""
    return copy.deepcopy(dict(result))


def compare_worker_results(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    left_bytes = stable_json_bytes(identity_projection(left))
    right_bytes = stable_json_bytes(identity_projection(right))
    return {
        "exact": bool(left_bytes == right_bytes),
        "left_sha256": sha256_bytes(left_bytes),
        "right_sha256": sha256_bytes(right_bytes),
        "left_size": len(left_bytes),
        "right_size": len(right_bytes),
    }


STAGEE_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py",
    "scripts/phase3_14b_r258_stagee_worker.py",
    "scripts/phase3_14b_r258_stagee_run_calibration.py",
    "scripts/phase3_14b_r258_stagee_test_gate.py",
    "scripts/phase3_14b_r258_stagee_blocked.py",
    "scripts/phase3_14b_r258_stagee_run.sh",
    "tests/test_phase3_14b_r258_stagee_constrained_integrator.py",
)


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
        raise ConstrainedIntegratorError("Stage E requires Experiment1")
    head = git_output(repository_root, "rev-parse", "HEAD")
    remote = git_output(repository_root, "rev-parse", "origin/Experiment1")
    if head != BASE_EVIDENCE_COMMIT or remote != BASE_EVIDENCE_COMMIT:
        raise ConstrainedIntegratorError("Stage-E initial local/remote binding changed")
    base = validate_base_evidence(repository_root)
    observed = status_paths(repository_root)
    expected = tuple(sorted(STAGEE_IMPLEMENTATION_PATHS))
    if observed != expected:
        raise ConstrainedIntegratorError("unexpected initial Stage-E paths: {}".format(observed))
    submodule = repository_root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise ConstrainedIntegratorError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise ConstrainedIntegratorError("DeformableRavens worktree is dirty")
    return {
        "head": head,
        "origin_experiment1": remote,
        "base_worker_sha256": base["worker"]["comparison"]["left_sha256"],
        "base_test_count": base["test_gate"]["passed_test_count"],
        "expected_untracked_paths": list(expected),
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def commit_parent(root: Path, commit: str) -> str:
    return git_output(root, "rev-parse", "{}^".format(commit))


def commit_subject(root: Path, commit: str) -> str:
    return git_output(root, "show", "-s", "--format=%s", commit)


def commit_paths(root: Path, commit: str) -> Tuple[str, ...]:
    output = git_output(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line.strip() for line in output.splitlines() if line.strip()))


def validate_implementation_commit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    commit = git_output(repository_root, "rev-parse", "HEAD")
    if commit_parent(repository_root, commit) != BASE_EVIDENCE_COMMIT:
        raise ConstrainedIntegratorError("Stage-E implementation parent changed")
    if commit_subject(repository_root, commit) != (
        "Add Phase3.14b-r2.5.8 surrogate-guided constrained direct-x0 integrator"
    ):
        raise ConstrainedIntegratorError("Stage-E implementation subject changed")
    if commit_paths(repository_root, commit) != tuple(sorted(STAGEE_IMPLEMENTATION_PATHS)):
        raise ConstrainedIntegratorError("Stage-E implementation paths changed")
    validate_base_evidence(repository_root)
    if git_output(repository_root, "rev-parse", "origin/Experiment1") != BASE_EVIDENCE_COMMIT:
        raise ConstrainedIntegratorError("origin/Experiment1 changed before Stage-E push")
    submodule = repository_root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise ConstrainedIntegratorError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise ConstrainedIntegratorError("DeformableRavens worktree is dirty")
    return {
        "implementation_commit": commit,
        "parent": BASE_EVIDENCE_COMMIT,
        "origin_experiment1": BASE_EVIDENCE_COMMIT,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }
