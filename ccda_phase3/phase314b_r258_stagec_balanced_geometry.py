"""Phase3.14b-r2.5.8 Stage C balanced segment objective calibration.

Stage B proved that the frozen one-sided upper-only K16 objective can reach the
upper/fidelity region directly in x0 space, but does so through lower-segment
violations, severe historical-physical invalidity, and measurable collapse at
all audited timesteps.  The target endpoint itself is upper-valid and
physically valid, and the registered direct-x0 tangent lower bound is high.
The next causal question is therefore objective design, not model capacity or
another optimizer variant.

This stage fits lower and distribution-anchor boundaries exclusively on the
638-row objective-training population.  It then compares a small,
pre-registered family of balanced lower/upper objectives by optimizing only
temporary direct-x0 tensors on the 236-row grouped selection holdout.

The holdout target is evaluation-only.  It is never used to fit a boundary,
weight, candidate, radius, or optimizer setting.  The 126-row frozen probe,
model training, reverse sampling, formal training, IDM, candidate execution,
DeformableRavens execution, Phase4, and CPS remain closed.  No checkpoint,
weights, prediction tensor, oracle tensor, NPZ, cache, image, or video is
persisted.
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
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r258_stagea_conflict_projected_k16 as stagea258
from ccda_phase3 import phase314b_r258_stageb_direct_x0_reachability as stageb258

PHASE = "Phase3.14b-r2.5.8 Stage C"
PHASE_ID = "phase314b_r258_stagec"

BASE_EVIDENCE_COMMIT = (
    "174d4428f835bb7fa76d9bdd497c9ff63452122f"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

BASE_CONTRACT = (
    "reports/phase3_14b_r258_stageb_contract.json"
)
BASE_WORKER = (
    "reports/phase3_14b_r258_stageb_worker_evidence.json"
)
BASE_SUMMARY = (
    "reports/phase3_14b_r258_stageb_summary.json"
)
BASE_REPORT = (
    "reports/phase3_14b_r258_stageb_report.md"
)
BASE_TEST_GATE = (
    "reports/phase3_14b_r258_stageb_test_gate_summary.json"
)
BASE_BOUND_FILES = (
    BASE_CONTRACT,
    BASE_WORKER,
    BASE_SUMMARY,
    BASE_REPORT,
    BASE_TEST_GATE,
)

EXPECTED_BASE_WORKER_SHA256 = (
    "f61f2cb9226904d360a1e9199b6d9434"
    "f35b499d4d15cec7048d62e6d4b45d96"
)
EXPECTED_BASE_CONTRACT_SHA256 = (
    "2322d276e57deae59135e0170920d6c9b"
    "2df3aeab040dbe58daf73db2adde684"
)
EXPECTED_BASE_SELECTION_SHA256 = (
    "ed716f7fca55c2b222f6363908f50f89"
    "a981a821751f2bddbb7f24ea9df6a443"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stageb_upper_only_direct_x0_"
    "oracle_uses_nonphysical_or_fidelity_shortcut"
)
EXPECTED_BASE_NEXT_PATH = (
    "REDESIGN_BALANCED_LOWER_UPPER_SEGMENT_"
    "GEOMETRY_OBJECTIVE_BEFORE_MODEL_REPAIR"
)
EXPECTED_STAGEC_CALIBRATION_SHA256 = (
    "fb91d29c84cedccf8f72458f08c2f213"
    "bbbc542663593094c0ab651701a052f0"
)
EXPECTED_STAGEC_CONTROL_SHA256 = (
    "dee61b8e31455eeb0e7e0eceae5c1500"
    "e3ce57ca8bd42989378d9f76fd779ae7"
)

TIMESTEP_UPPER_MIN = {
    10: 0.75,
    25: 0.60,
    50: 0.50,
}


class BalancedGeometryError(RuntimeError):
    """Raised when immutable evidence or an objective invariant fails."""


@dataclass(frozen=True)
class BalancedCandidateDefinition:
    candidate_id: str
    lower_quantile: float
    lower_weight: float
    anchor_weight: float
    role: str

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate id is empty")
        if self.role not in ("negative_control", "selectable"):
            raise ValueError("candidate role changed")
        if not 0.0 < float(self.lower_quantile) < 0.5:
            raise ValueError("lower quantile is invalid")
        if float(self.lower_weight) < 0.0:
            raise ValueError("lower weight is negative")
        if float(self.anchor_weight) < 0.0:
            raise ValueError("anchor weight is negative")
        if self.role == "negative_control":
            if self.lower_weight != 0.0 or self.anchor_weight != 0.0:
                raise ValueError("negative control changed")
        elif self.lower_weight <= 0.0:
            raise ValueError("selectable candidate lacks a lower term")


CANDIDATE_DEFINITIONS = (
    BalancedCandidateDefinition(
        "upper_only_reference",
        0.05,
        0.0,
        0.0,
        "negative_control",
    ),
    BalancedCandidateDefinition(
        "band_q01_w1",
        0.01,
        1.0,
        0.0,
        "selectable",
    ),
    BalancedCandidateDefinition(
        "band_q05_w1",
        0.05,
        1.0,
        0.0,
        "selectable",
    ),
    BalancedCandidateDefinition(
        "band_q05_w2",
        0.05,
        2.0,
        0.0,
        "selectable",
    ),
    BalancedCandidateDefinition(
        "band_q10_w1",
        0.10,
        1.0,
        0.0,
        "selectable",
    ),
    BalancedCandidateDefinition(
        "band_q05_w1_a0p25",
        0.05,
        1.0,
        0.25,
        "selectable",
    ),
    BalancedCandidateDefinition(
        "band_q05_w1_a1p00",
        0.05,
        1.0,
        1.00,
        "selectable",
    ),
)


@dataclass(frozen=True)
class BalancedGeometrySpec:
    timesteps: Tuple[int, ...] = (10, 25, 50)
    top_k: int = 16
    lower_quantiles: Tuple[float, ...] = (
        0.01,
        0.05,
        0.10,
    )
    anchor_quantiles: Tuple[float, float] = (
        0.25,
        0.75,
    )
    minimum_log_band_width: float = 1.0e-4

    oracle_radii: Tuple[float, ...] = (
        0.025,
        0.050,
        0.100,
        0.200,
        0.400,
        0.800,
    )
    oracle_steps: int = 256
    oracle_learning_rate_scale: float = 0.25
    oracle_min_learning_rate: float = 1.0e-3
    oracle_checkpoints: Tuple[int, ...] = (
        0,
        1,
        8,
        32,
        128,
        256,
    )

    fidelity_ratio_max: float = 1.20
    lower_pass_min: float = 0.90
    physical_valid_min: float = 0.90
    collapse_fraction_max: float = 0.05
    stretch_fraction_max: float = 0.05
    movement_target_cosine_min: float = 0.30
    target_distance_reduction_min: float = 0.0

    gradient_target_cosine_min: float = 0.25
    gradient_nonnegative_rate_min: float = 0.70

    norm_epsilon: float = 1.0e-12
    numerical_tolerance: float = 1.0e-6

    def validate(self) -> None:
        if self.timesteps != tuple(sorted(TIMESTEP_UPPER_MIN)):
            raise ValueError("timestep population changed")
        if self.top_k != 16:
            raise ValueError("K16 contract changed")
        if self.lower_quantiles != tuple(sorted(set(self.lower_quantiles))):
            raise ValueError("lower quantiles changed")
        if self.lower_quantiles != (0.01, 0.05, 0.10):
            raise ValueError("registered lower quantiles changed")
        if self.anchor_quantiles != (0.25, 0.75):
            raise ValueError("anchor quantiles changed")
        if self.minimum_log_band_width <= 0.0:
            raise ValueError("minimum band width is invalid")
        if self.oracle_radii != tuple(sorted(set(self.oracle_radii))):
            raise ValueError("oracle radii changed")
        if self.oracle_steps <= 0:
            raise ValueError("oracle steps are invalid")
        if self.oracle_checkpoints[0] != 0:
            raise ValueError("oracle checkpoints lack zero")
        if self.oracle_checkpoints[-1] != self.oracle_steps:
            raise ValueError("oracle final checkpoint changed")
        if tuple(sorted(set(self.oracle_checkpoints))) != self.oracle_checkpoints:
            raise ValueError("oracle checkpoints are not ordered")
        for definition in CANDIDATE_DEFINITIONS:
            definition.validate()
        if tuple(
            definition.candidate_id
            for definition in CANDIDATE_DEFINITIONS
        ) != (
            "upper_only_reference",
            "band_q01_w1",
            "band_q05_w1",
            "band_q05_w2",
            "band_q10_w1",
            "band_q05_w1_a0p25",
            "band_q05_w1_a1p00",
        ):
            raise ValueError("candidate order changed")
        for value in (
            self.fidelity_ratio_max,
            self.lower_pass_min,
            self.physical_valid_min,
            self.movement_target_cosine_min,
            self.gradient_target_cosine_min,
            self.gradient_nonnegative_rate_min,
        ):
            if float(value) <= 0.0:
                raise ValueError("positive threshold is invalid")
        if self.collapse_fraction_max < 0.0:
            raise ValueError("collapse threshold is negative")
        if self.stretch_fraction_max < 0.0:
            raise ValueError("stretch threshold is negative")
        if self.norm_epsilon <= 0.0 or self.numerical_tolerance <= 0.0:
            raise ValueError("numerical constant is invalid")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_strings(values: Sequence[Any]) -> str:
    return sha256_bytes(
        "\n".join(
            str(value)
            for value in values
        ).encode("utf-8")
    )


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
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
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


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            "refusing to overwrite write-once output: {}".format(target)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (
        ".{}.{}.tmp".format(target.name, os.getpid())
    )
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
        raise BalancedGeometryError(
            "JSON root is not an object: {}".format(path)
        )
    return value


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def assert_commit_ancestor(root: Path, commit: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise BalancedGeometryError(
            "required commit is not an ancestor: {}".format(commit)
        )


def assert_file_bound_to_commit(
    root: Path,
    relative: str,
    commit: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        ["git", "show", "{}:{}".format(commit, relative)],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise BalancedGeometryError(
            "base-bound file differs: {}".format(relative)
        )
    return sha256_bytes(observed)


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_ancestor(
        repository_root,
        BASE_EVIDENCE_COMMIT,
    )
    file_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            BASE_EVIDENCE_COMMIT,
        )
        for relative in BASE_BOUND_FILES
    }
    contract = load_json(
        repository_root / BASE_CONTRACT
    )
    worker = load_json(
        repository_root / BASE_WORKER
    )
    summary = load_json(
        repository_root / BASE_SUMMARY
    )
    test_gate = load_json(
        repository_root / BASE_TEST_GATE
    )

    if summary.get("verdict") != "PASS":
        raise BalancedGeometryError(
            "Stage-B verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise BalancedGeometryError(
            "Stage-B scientific status changed"
        )
    if summary.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise BalancedGeometryError(
            "Stage-B root cause changed"
        )
    if summary.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise BalancedGeometryError(
            "Stage-B next path changed"
        )
    if summary.get("selected_configuration") is not None:
        raise BalancedGeometryError(
            "Stage-B selected a configuration"
        )
    if summary.get("train_only_recommendation") is not None:
        raise BalancedGeometryError(
            "Stage-B emitted a train-only recommendation"
        )
    if summary.get("workers_exact") is not True:
        raise BalancedGeometryError(
            "Stage-B workers are not exact"
        )

    comparison = worker.get("comparison")
    if not isinstance(comparison, dict):
        raise BalancedGeometryError(
            "Stage-B worker comparison is missing"
        )
    if comparison.get("exact") is not True:
        raise BalancedGeometryError(
            "Stage-B worker comparison changed"
        )
    if (
        comparison.get("left_sha256")
        != EXPECTED_BASE_WORKER_SHA256
        or comparison.get("right_sha256")
        != EXPECTED_BASE_WORKER_SHA256
    ):
        raise BalancedGeometryError(
            "Stage-B worker identity changed"
        )
    result = worker.get("worker_result")
    if not isinstance(result, dict):
        raise BalancedGeometryError(
            "Stage-B worker result is missing"
        )
    if result.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise BalancedGeometryError(
            "Stage-B worker root cause changed"
        )
    if result.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise BalancedGeometryError(
            "Stage-B worker next path changed"
        )
    if (
        result.get("reachability_contract", {}).get("contract_sha256")
        != EXPECTED_BASE_CONTRACT_SHA256
    ):
        raise BalancedGeometryError(
            "Stage-B internal contract changed"
        )
    if (
        result.get("selection", {}).get("selection_sha256")
        != EXPECTED_BASE_SELECTION_SHA256
    ):
        raise BalancedGeometryError(
            "Stage-B selection changed"
        )
    if contract.get("contract_sha256") != EXPECTED_BASE_CONTRACT_SHA256:
        raise BalancedGeometryError(
            "Stage-B contract record changed"
        )
    if contract.get("selection_sha256") != EXPECTED_BASE_SELECTION_SHA256:
        raise BalancedGeometryError(
            "Stage-B contract selection changed"
        )

    control = result.get("control_capture", {})
    if control.get("reference_equivalence_pass") is not True:
        raise BalancedGeometryError(
            "Stage-B control reference equivalence failed"
        )
    if (
        control.get("observed_calibration_sha256")
        != EXPECTED_STAGEC_CALIBRATION_SHA256
    ):
        raise BalancedGeometryError(
            "Stage-B calibration replay changed"
        )
    if (
        control.get("observed_control_sha256")
        != EXPECTED_STAGEC_CONTROL_SHA256
    ):
        raise BalancedGeometryError(
            "Stage-B control replay changed"
        )
    if control.get("stagec_nonzero_candidate_training_count") != 0:
        raise BalancedGeometryError(
            "Stage-B replay trained a Stage-C candidate"
        )

    classification = result.get("classification", {})
    if classification.get("primary_failure_locus") != (
        "one_sided_objective_shortcut"
    ):
        raise BalancedGeometryError(
            "Stage-B failure locus changed"
        )
    if classification.get("target_anchor_all_pass") is not True:
        raise BalancedGeometryError(
            "Stage-B target anchors changed"
        )
    if classification.get("target_line_all_reachable") is not True:
        raise BalancedGeometryError(
            "Stage-B target line changed"
        )
    if int(
        classification.get("oracle_upper_fidelity_count", -1)
    ) != 3:
        raise BalancedGeometryError(
            "Stage-B upper/fidelity count changed"
        )
    if int(
        classification.get("oracle_valid_count", -1)
    ) != 0:
        raise BalancedGeometryError(
            "Stage-B cable-valid count changed"
        )
    if int(
        classification.get("nonphysical_shortcut_count", -1)
    ) != 3:
        raise BalancedGeometryError(
            "Stage-B shortcut count changed"
        )

    timestep_records = result.get("timestep_records")
    if not isinstance(timestep_records, dict):
        raise BalancedGeometryError(
            "Stage-B timestep records are missing"
        )
    for timestep in ("10", "25", "50"):
        record = timestep_records.get(timestep)
        if not isinstance(record, dict):
            raise BalancedGeometryError(
                "Stage-B timestep record is missing: {}".format(timestep)
            )
        oracle = record.get("direct_x0_oracle", {})
        if oracle.get("upper_fidelity_reachable") is not True:
            raise BalancedGeometryError(
                "Stage-B upper/fidelity reference changed"
            )
        if oracle.get("cable_valid_reachable") is not False:
            raise BalancedGeometryError(
                "Stage-B cable-valid reference changed"
            )

    if test_gate.get("verdict") != "PASS":
        raise BalancedGeometryError(
            "Stage-B test gate is not PASS"
        )
    if int(test_gate.get("test_file_count", -1)) != 49:
        raise BalancedGeometryError(
            "Stage-B test-file count changed"
        )
    if int(test_gate.get("passed_test_count", -1)) != 959:
        raise BalancedGeometryError(
            "Stage-B pass count changed"
        )
    if int(test_gate.get("stageb_new_passed", -1)) != 48:
        raise BalancedGeometryError(
            "Stage-B new-test count changed"
        )

    for key in (
        "frozen_probe_accessed",
        "reverse_sampling_run",
        "full_stageb_repaired_model_trained",
        "formal_pilot_run",
        "checkpoint_saved",
        "weights_persisted",
        "prediction_tensor_persisted",
        "oracle_tensor_persisted",
        "candidate_tensor_persisted",
        "npz_saved",
        "cache_saved",
        "formal_diffusion_training",
        "formal_reverse_sampling",
        "formal_idm_training",
        "action_diverse_data_collection",
        "candidate_execution",
        "deformable_ravens_executed",
        "phase4",
        "cps",
    ):
        if summary.get(key) is not False:
            raise BalancedGeometryError(
                "Stage-B boundary changed: {}".format(key)
            )

    return {
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "file_sha256": file_sha,
        "contract": contract,
        "worker": worker,
        "worker_result": result,
        "summary": summary,
        "test_gate": test_gate,
    }


def linear_quantile_axis0(
    value: np.ndarray,
    quantile: float,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] <= 0:
        raise ValueError(
            "quantile input must have a nonempty row axis"
        )
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile is outside [0,1]")
    ordered = np.sort(array, axis=0)
    position = q * float(ordered.shape[0] - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    weight = float(position - lower_index)
    result = (
        (1.0 - weight) * ordered[lower_index]
        + weight * ordered[upper_index]
    )
    if not np.all(np.isfinite(result)):
        raise BalancedGeometryError(
            "quantile result contains NaN or Inf"
        )
    return result.astype(np.float32)


def _stats(value: np.ndarray) -> Dict[str, float]:
    return stageb258._safe_stats(
        np.asarray(value, dtype=np.float64)
    )


def fit_balanced_reference(
    *,
    context: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    target = np.asarray(
        context["target"][
            context["objective_train_mask"]
        ],
        dtype=np.float32,
    )
    groups = np.asarray(
        context["groups"][
            context["objective_train_mask"]
        ]
    ).astype(str)
    lengths = stageb.segment_lengths(
        target
    ).astype(np.float64)
    if np.any(lengths <= 0.0):
        raise BalancedGeometryError(
            "objective-train target contains nonpositive segment length"
        )
    log_lengths = np.log(lengths)
    quantiles = {
        "q01": linear_quantile_axis0(
            log_lengths,
            0.01,
        ),
        "q05": linear_quantile_axis0(
            log_lengths,
            0.05,
        ),
        "q10": linear_quantile_axis0(
            log_lengths,
            0.10,
        ),
        "q25": linear_quantile_axis0(
            log_lengths,
            0.25,
        ),
        "q50": linear_quantile_axis0(
            log_lengths,
            0.50,
        ),
        "q75": linear_quantile_axis0(
            log_lengths,
            0.75,
        ),
    }
    upper = np.asarray(
        context[
            "objective_contract"
        ].allowed_upper_log_length,
        dtype=np.float32,
    )
    expected_shape = (
        stageb.FUTURE_STEPS,
        stageb.BEADS - 1,
    )
    if upper.shape != expected_shape:
        raise BalancedGeometryError(
            "frozen upper-bound shape changed"
        )
    for name, array in quantiles.items():
        if array.shape != expected_shape:
            raise BalancedGeometryError(
                "{} shape changed".format(name)
            )

    clipped_lower = {}
    for name in ("q01", "q05", "q10"):
        clipped_lower[name] = np.minimum(
            quantiles[name],
            upper
            - float(spec.minimum_log_band_width),
        ).astype(np.float32)
    anchor_lower = np.maximum(
        quantiles["q25"],
        clipped_lower["q05"],
    ).astype(np.float32)
    anchor_upper = np.minimum(
        quantiles["q75"],
        upper,
    ).astype(np.float32)
    if np.any(
        anchor_upper - anchor_lower
        < -float(spec.numerical_tolerance)
    ):
        raise BalancedGeometryError(
            "objective-train IQR anchor is inconsistent with frozen upper bound"
        )
    anchor_upper = np.maximum(
        anchor_upper,
        anchor_lower,
    ).astype(np.float32)

    lower_coverage = {
        name: float(
            np.mean(
                log_lengths
                >= array[None]
            )
        )
        for name, array in clipped_lower.items()
    }
    upper_coverage = float(
        np.mean(
            log_lengths
            <= upper[None]
        )
    )
    anchor_coverage = float(
        np.mean(
            (
                log_lengths
                >= anchor_lower[None]
            )
            & (
                log_lengths
                <= anchor_upper[None]
            )
        )
    )
    return {
        "schema":
            "phase314b_r258_stagec_balanced_reference_v1",
        "fit_rows": int(target.shape[0]),
        "fit_group_count": int(
            len(set(groups.tolist()))
        ),
        "fit_group_sha256":
            sha256_strings(
                sorted(set(groups.tolist()))
            ),
        "fit_target_sha256":
            sha256_array(target),
        "fit_log_length_sha256":
            sha256_array(
                log_lengths.astype(np.float64)
            ),
        "segment_length":
            _stats(lengths),
        "quantiles": quantiles,
        "clipped_lower": clipped_lower,
        "frozen_upper": upper,
        "anchor_lower": anchor_lower,
        "anchor_upper": anchor_upper,
        "lower_element_coverage":
            lower_coverage,
        "upper_element_coverage":
            upper_coverage,
        "anchor_element_coverage":
            anchor_coverage,
        "array_sha256": {
            **{
                name: sha256_array(array)
                for name, array
                in quantiles.items()
            },
            **{
                "lower_{}".format(name):
                    sha256_array(array)
                for name, array
                in clipped_lower.items()
            },
            "frozen_upper":
                sha256_array(upper),
            "anchor_lower":
                sha256_array(anchor_lower),
            "anchor_upper":
                sha256_array(anchor_upper),
        },
        "uses_objective_train_target":
            True,
        "uses_selection_holdout_target":
            False,
        "uses_frozen_probe":
            False,
    }


def lower_key(quantile: float) -> str:
    mapping = {
        0.01: "q01",
        0.05: "q05",
        0.10: "q10",
    }
    key = mapping.get(round(float(quantile), 2))
    if key is None:
        raise ValueError(
            "unregistered lower quantile"
        )
    return key


def candidate_contract(
    *,
    definition: BalancedCandidateDefinition,
    reference: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    definition.validate()
    key = lower_key(
        definition.lower_quantile
    )
    lower = np.asarray(
        reference["clipped_lower"][key],
        dtype=np.float32,
    )
    upper = np.asarray(
        reference["frozen_upper"],
        dtype=np.float32,
    )
    anchor_lower = np.asarray(
        reference["anchor_lower"],
        dtype=np.float32,
    )
    anchor_upper = np.asarray(
        reference["anchor_upper"],
        dtype=np.float32,
    )
    if np.any(
        lower
        > upper
        - float(spec.minimum_log_band_width)
        + float(spec.numerical_tolerance)
    ):
        raise BalancedGeometryError(
            "candidate lower boundary is not below upper boundary"
        )
    payload = {
        "schema":
            "phase314b_r258_stagec_balanced_candidate_contract_v1",
        "candidate":
            asdict(definition),
        "top_k": int(spec.top_k),
        "lower_log_length":
            lower,
        "upper_log_length":
            upper,
        "anchor_lower_log_length":
            anchor_lower,
        "anchor_upper_log_length":
            anchor_upper,
        "lower_log_length_sha256":
            sha256_array(lower),
        "upper_log_length_sha256":
            sha256_array(upper),
        "anchor_lower_log_length_sha256":
            sha256_array(anchor_lower),
        "anchor_upper_log_length_sha256":
            sha256_array(anchor_upper),
        "reference_fit_target_sha256":
            reference["fit_target_sha256"],
        "reference_fit_group_sha256":
            reference["fit_group_sha256"],
        "uses_selection_holdout_target":
            False,
        "uses_frozen_probe":
            False,
    }
    payload["contract_sha256"] = (
        sha256_bytes(
            stable_json_bytes(payload)
        )
    )
    return payload


def _torch_log_lengths(
    predicted_x0_z: Any,
    *,
    target_standardizer: stageb.ArrayStandardizer,
    epsilon: float,
) -> Any:
    torch, _ = stageb._torch_imports()
    mean = torch.as_tensor(
        target_standardizer.mean,
        dtype=predicted_x0_z.dtype,
        device=predicted_x0_z.device,
    )
    scale = torch.as_tensor(
        target_standardizer.scale,
        dtype=predicted_x0_z.dtype,
        device=predicted_x0_z.device,
    )
    raw = (
        predicted_x0_z
        * scale[None]
        + mean[None]
    )
    points = raw.reshape(
        raw.shape[0],
        stageb.FUTURE_STEPS,
        stageb.BEADS,
        2,
    )
    delta = (
        points[:, :, 1:, :]
        - points[:, :, :-1, :]
    )
    lengths = torch.linalg.vector_norm(
        delta,
        dim=-1,
    )
    return torch.log(
        torch.clamp(
            lengths,
            min=float(epsilon),
        )
    )


def _topk_mean_square(
    excess: Any,
    *,
    top_k: int,
) -> Tuple[Any, Any]:
    torch, _ = stageb._torch_imports()
    flattened = excess.reshape(
        excess.shape[0],
        -1,
    )
    actual_k = min(
        int(top_k),
        int(flattened.shape[1]),
    )
    selected, indices = torch.topk(
        flattened,
        k=actual_k,
        dim=1,
        largest=True,
        sorted=True,
    )
    return (
        torch.mean(
            selected * selected,
            dim=1,
        ),
        indices,
    )


def balanced_objective_terms_torch(
    predicted_x0_z: Any,
    *,
    target_standardizer: stageb.ArrayStandardizer,
    candidate: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    definition = BalancedCandidateDefinition(
        **candidate["candidate"]
    )
    definition.validate()
    log_lengths = _torch_log_lengths(
        predicted_x0_z,
        target_standardizer=
            target_standardizer,
        epsilon=spec.norm_epsilon,
    )
    lower = torch.as_tensor(
        candidate["lower_log_length"],
        dtype=predicted_x0_z.dtype,
        device=predicted_x0_z.device,
    )
    upper = torch.as_tensor(
        candidate["upper_log_length"],
        dtype=predicted_x0_z.dtype,
        device=predicted_x0_z.device,
    )
    anchor_lower = torch.as_tensor(
        candidate[
            "anchor_lower_log_length"
        ],
        dtype=predicted_x0_z.dtype,
        device=predicted_x0_z.device,
    )
    anchor_upper = torch.as_tensor(
        candidate[
            "anchor_upper_log_length"
        ],
        dtype=predicted_x0_z.dtype,
        device=predicted_x0_z.device,
    )

    upper_excess = torch.relu(
        log_lengths - upper[None]
    )
    lower_excess = torch.relu(
        lower[None] - log_lengths
    )
    anchor_excess = torch.maximum(
        torch.relu(
            anchor_lower[None]
            - log_lengths
        ),
        torch.relu(
            log_lengths
            - anchor_upper[None]
        ),
    )
    upper_row, upper_indices = (
        _topk_mean_square(
            upper_excess,
            top_k=spec.top_k,
        )
    )
    lower_row, lower_indices = (
        _topk_mean_square(
            lower_excess,
            top_k=spec.top_k,
        )
    )
    anchor_row, anchor_indices = (
        _topk_mean_square(
            anchor_excess,
            top_k=spec.top_k,
        )
    )
    row_total = (
        upper_row
        + float(definition.lower_weight)
        * lower_row
        + float(definition.anchor_weight)
        * anchor_row
    )
    return {
        "total":
            torch.mean(row_total),
        "row_total":
            row_total,
        "upper_total":
            torch.mean(upper_row),
        "lower_total":
            torch.mean(lower_row),
        "anchor_total":
            torch.mean(anchor_row),
        "upper_row":
            upper_row,
        "lower_row":
            lower_row,
        "anchor_row":
            anchor_row,
        "upper_excess":
            upper_excess,
        "lower_excess":
            lower_excess,
        "anchor_excess":
            anchor_excess,
        "upper_indices":
            upper_indices,
        "lower_indices":
            lower_indices,
        "anchor_indices":
            anchor_indices,
        "log_lengths":
            log_lengths,
    }


def _numpy_topk_square(
    excess: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    flattened = np.asarray(
        excess,
        dtype=np.float64,
    ).reshape(
        excess.shape[0],
        -1,
    )
    actual_k = min(
        int(top_k),
        int(flattened.shape[1]),
    )
    selected = np.partition(
        flattened,
        kth=flattened.shape[1] - actual_k,
        axis=1,
    )[:, -actual_k:]
    return np.mean(
        selected * selected,
        axis=1,
    )


def balanced_objective_profile_numpy(
    value: np.ndarray,
    *,
    candidate: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    definition = BalancedCandidateDefinition(
        **candidate["candidate"]
    )
    definition.validate()
    raw = np.asarray(
        value,
        dtype=np.float32,
    )
    lengths = stageb.segment_lengths(
        raw
    ).astype(np.float64)
    log_lengths = np.log(
        np.maximum(
            lengths,
            float(spec.norm_epsilon),
        )
    )
    lower = np.asarray(
        candidate["lower_log_length"],
        dtype=np.float64,
    )
    upper = np.asarray(
        candidate["upper_log_length"],
        dtype=np.float64,
    )
    anchor_lower = np.asarray(
        candidate[
            "anchor_lower_log_length"
        ],
        dtype=np.float64,
    )
    anchor_upper = np.asarray(
        candidate[
            "anchor_upper_log_length"
        ],
        dtype=np.float64,
    )
    upper_excess = np.maximum(
        log_lengths
        - upper[None],
        0.0,
    )
    lower_excess = np.maximum(
        lower[None]
        - log_lengths,
        0.0,
    )
    anchor_excess = np.maximum(
        np.maximum(
            anchor_lower[None]
            - log_lengths,
            0.0,
        ),
        np.maximum(
            log_lengths
            - anchor_upper[None],
            0.0,
        ),
    )
    upper_row = _numpy_topk_square(
        upper_excess,
        top_k=spec.top_k,
    )
    lower_row = _numpy_topk_square(
        lower_excess,
        top_k=spec.top_k,
    )
    anchor_row = _numpy_topk_square(
        anchor_excess,
        top_k=spec.top_k,
    )
    row_total = (
        upper_row
        + float(definition.lower_weight)
        * lower_row
        + float(definition.anchor_weight)
        * anchor_row
    )
    return {
        "total":
            float(np.mean(row_total)),
        "upper_total":
            float(np.mean(upper_row)),
        "lower_total":
            float(np.mean(lower_row)),
        "anchor_total":
            float(np.mean(anchor_row)),
        "row_total":
            _stats(row_total),
        "row_total_sha256":
            sha256_array(
                row_total.astype(np.float64)
            ),
        "upper_positive_element_rate":
            float(
                np.mean(
                    upper_excess > 0.0
                )
            ),
        "lower_positive_element_rate":
            float(
                np.mean(
                    lower_excess > 0.0
                )
            ),
        "anchor_positive_element_rate":
            float(
                np.mean(
                    anchor_excess > 0.0
                )
            ),
        "maximum_upper_excess":
            float(np.max(upper_excess)),
        "maximum_lower_excess":
            float(np.max(lower_excess)),
        "maximum_anchor_excess":
            float(np.max(anchor_excess)),
    }


def objective_gradient_alignment(
    *,
    control: np.ndarray,
    target: np.ndarray,
    context: Mapping[str, Any],
    candidate: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    device = torch.device("cuda:0")
    control_z_np = (
        context["target_standardizer"]
        .normalize(control)
        .astype(np.float32)
    )
    target_z_np = (
        context["target_standardizer"]
        .normalize(target)
        .astype(np.float32)
    )
    value = torch.as_tensor(
        control_z_np,
        dtype=torch.float32,
        device=device,
    ).detach().clone()
    value.requires_grad_(True)
    terms = balanced_objective_terms_torch(
        value,
        target_standardizer=
            context["target_standardizer"],
        candidate=candidate,
        spec=spec,
    )
    gradient = torch.autograd.grad(
        terms["total"],
        value,
        retain_graph=False,
        create_graph=False,
    )[0]
    descent = (
        -gradient.detach().cpu().numpy()
    )
    target_direction = (
        target_z_np - control_z_np
    )
    active = np.asarray(
        context[
            "target_standardizer"
        ].active,
        dtype=np.bool_,
    ).reshape(-1)
    descent_flat = descent.reshape(
        descent.shape[0],
        -1,
    )
    target_flat = target_direction.reshape(
        target_direction.shape[0],
        -1,
    )
    if np.any(active):
        descent_flat = descent_flat[:, active]
        target_flat = target_flat[:, active]
    cosine = stageb258._row_cosine(
        descent_flat,
        target_flat,
        epsilon=spec.norm_epsilon,
    )
    descent_norm = np.sqrt(
        np.sum(
            descent_flat
            * descent_flat,
            axis=1,
        )
    )
    target_norm = np.sqrt(
        np.sum(
            target_flat
            * target_flat,
            axis=1,
        )
    )
    record = {
        "objective_total":
            float(
                terms["total"]
                .detach()
                .cpu()
            ),
        "upper_total":
            float(
                terms["upper_total"]
                .detach()
                .cpu()
            ),
        "lower_total":
            float(
                terms["lower_total"]
                .detach()
                .cpu()
            ),
        "anchor_total":
            float(
                terms["anchor_total"]
                .detach()
                .cpu()
            ),
        "gradient_sha256":
            sha256_array(
                gradient.detach()
                .cpu()
                .numpy()
            ),
        "descent_target_cosine":
            _stats(cosine),
        "nonnegative_alignment_rate":
            float(
                np.mean(
                    cosine >= 0.0
                )
            ),
        "descent_norm":
            _stats(descent_norm),
        "target_direction_norm":
            _stats(target_norm),
        "zero_descent_row_rate":
            float(
                np.mean(
                    descent_norm
                    <= spec.norm_epsilon
                )
            ),
    }
    record["direction_pass"] = bool(
        record[
            "descent_target_cosine"
        ]["mean"]
        >= spec.gradient_target_cosine_min
        and record[
            "nonnegative_alignment_rate"
        ]
        >= spec.gradient_nonnegative_rate_min
    )
    return record


def _snapshot_oracle(
    *,
    value: Any,
    control: np.ndarray,
    target: np.ndarray,
    context: Mapping[str, Any],
    candidate: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    raw = (
        context[
            "target_standardizer"
        ]
        .denormalize(
            value.detach()
            .cpu()
            .numpy()
        )
        .astype(np.float32)
    )
    evaluation = (
        stageb258
        .evaluate_output_population(
            raw,
            control=control,
            target=target,
            groups=
                context[
                    "holdout_groups"
                ],
            condition_name=
                context[
                    "holdout_condition_name"
                ],
            target_standardizer=
                context[
                    "target_standardizer"
                ],
            objective_contract=
                context[
                    "objective_contract"
                ],
            upper_gate=
                context[
                    "upper_gate"
                ],
            stage_d_contract=
                context[
                    "stage_d_contract"
                ],
            historical_geometry=
                context[
                    "historical_geometry"
                ],
            top_k=spec.top_k,
            epsilon=spec.norm_epsilon,
        )
    )
    return {
        "evaluation": evaluation,
        "balanced_objective":
            balanced_objective_profile_numpy(
                raw,
                candidate=candidate,
                spec=spec,
            ),
    }


def optimize_balanced_oracle(
    *,
    control: np.ndarray,
    target: np.ndarray,
    radius: float,
    context: Mapping[str, Any],
    candidate: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    device = torch.device("cuda:0")
    control_z_np = (
        context["target_standardizer"]
        .normalize(control)
        .astype(np.float32)
    )
    control_z = torch.as_tensor(
        control_z_np,
        dtype=torch.float32,
        device=device,
    )
    value = torch.nn.Parameter(
        control_z.detach().clone()
    )
    learning_rate = max(
        float(radius)
        * spec.oracle_learning_rate_scale,
        spec.oracle_min_learning_rate,
    )
    optimizer = torch.optim.Adam(
        [value],
        lr=learning_rate,
    )
    active_flat = torch.as_tensor(
        np.asarray(
            context[
                "target_standardizer"
            ].active,
            dtype=np.bool_,
        ).reshape(-1),
        dtype=torch.bool,
        device=device,
    )
    checkpoints: MutableMapping[
        str,
        Any,
    ] = {}
    total_history: List[float] = []
    upper_history: List[float] = []
    lower_history: List[float] = []
    anchor_history: List[float] = []

    checkpoints["0"] = {
        "step": 0,
        **_snapshot_oracle(
            value=value,
            control=control,
            target=target,
            context=context,
            candidate=candidate,
            spec=spec,
        ),
    }
    for step in range(
        1,
        spec.oracle_steps + 1,
    ):
        optimizer.zero_grad(
            set_to_none=True
        )
        terms = (
            balanced_objective_terms_torch(
                value,
                target_standardizer=
                    context[
                        "target_standardizer"
                    ],
                candidate=candidate,
                spec=spec,
            )
        )
        loss = terms["total"]
        if not bool(
            torch.isfinite(loss)
            .detach()
            .cpu()
        ):
            raise BalancedGeometryError(
                "balanced oracle loss is not finite"
            )
        loss.backward()
        if value.grad is None:
            raise BalancedGeometryError(
                "balanced oracle gradient is missing"
            )
        if not bool(
            torch.all(
                torch.isfinite(
                    value.grad
                )
            )
            .detach()
            .cpu()
        ):
            raise BalancedGeometryError(
                "balanced oracle gradient is not finite"
            )
        optimizer.step()
        stageb258._project_normalized_trust_region(
            value=value,
            control=control_z,
            active_flat=active_flat,
            radius=float(radius),
            epsilon=spec.norm_epsilon,
        )
        total_history.append(
            float(
                terms["total"]
                .detach()
                .cpu()
            )
        )
        upper_history.append(
            float(
                terms["upper_total"]
                .detach()
                .cpu()
            )
        )
        lower_history.append(
            float(
                terms["lower_total"]
                .detach()
                .cpu()
            )
        )
        anchor_history.append(
            float(
                terms["anchor_total"]
                .detach()
                .cpu()
            )
        )
        if step in spec.oracle_checkpoints:
            checkpoints[str(step)] = {
                "step": int(step),
                **_snapshot_oracle(
                    value=value,
                    control=control,
                    target=target,
                    context=context,
                    candidate=candidate,
                    spec=spec,
                ),
            }

    final = checkpoints[
        str(spec.oracle_steps)
    ]
    return {
        "radius": float(radius),
        "steps": int(
            spec.oracle_steps
        ),
        "learning_rate":
            float(learning_rate),
        "loss_history": {
            "total_first":
                float(total_history[0]),
            "total_final":
                float(total_history[-1]),
            "upper_first":
                float(upper_history[0]),
            "upper_final":
                float(upper_history[-1]),
            "lower_first":
                float(lower_history[0]),
            "lower_final":
                float(lower_history[-1]),
            "anchor_first":
                float(anchor_history[0]),
            "anchor_final":
                float(anchor_history[-1]),
            "total_sha256":
                sha256_array(
                    np.asarray(
                        total_history,
                        dtype=np.float64,
                    )
                ),
            "upper_sha256":
                sha256_array(
                    np.asarray(
                        upper_history,
                        dtype=np.float64,
                    )
                ),
            "lower_sha256":
                sha256_array(
                    np.asarray(
                        lower_history,
                        dtype=np.float64,
                    )
                ),
            "anchor_sha256":
                sha256_array(
                    np.asarray(
                        anchor_history,
                        dtype=np.float64,
                    )
                ),
            "finite": bool(
                np.all(
                    np.isfinite(
                        np.asarray(
                            total_history
                            + upper_history
                            + lower_history
                            + anchor_history,
                            dtype=np.float64,
                        )
                    )
                )
            ),
        },
        "checkpoints":
            dict(checkpoints),
        "final":
            final,
        "oracle_tensor_persisted":
            False,
    }


def scientific_gates(
    *,
    timestep: int,
    record: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, bool]:
    evaluation = record[
        "final"
    ]["evaluation"]
    movement_cosine = float(
        evaluation[
            "target_direction_cosine"
        ]["mean"]
    )
    distance_reduction = float(
        evaluation[
            "target_distance_reduction_fraction"
        ]["mean"]
    )
    gates = {
        "upper": bool(
            evaluation[
                "upper_row_pass_rate"
            ]
            >= TIMESTEP_UPPER_MIN[
                int(timestep)
            ]
        ),
        "fidelity": bool(
            evaluation[
                "normalized_mse_ratio"
            ]
            <= spec.fidelity_ratio_max
        ),
        "lower": bool(
            evaluation[
                "lower_row_pass_rate"
            ]
            >= spec.lower_pass_min
        ),
        "physical": bool(
            evaluation[
                "historical_physical_row_any_rate"
            ]
            >= spec.physical_valid_min
        ),
        "collapse": bool(
            evaluation[
                "segment_length"
            ]["collapse_fraction"]
            <= spec.collapse_fraction_max
        ),
        "stretch": bool(
            evaluation[
                "segment_length"
            ]["stretch_fraction"]
            <= spec.stretch_fraction_max
        ),
        "movement_alignment": bool(
            movement_cosine
            >= spec.movement_target_cosine_min
        ),
        "target_distance_reduction": bool(
            distance_reduction
            > spec.target_distance_reduction_min
        ),
        "loss_finite": bool(
            record["loss_history"]["finite"]
        ),
    }
    gates["all"] = bool(
        all(gates.values())
    )
    return gates


def balanced_oracle_sweep(
    *,
    control: np.ndarray,
    target: np.ndarray,
    timestep: int,
    context: Mapping[str, Any],
    candidate: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    records = []
    for radius in spec.oracle_radii:
        record = optimize_balanced_oracle(
            control=control,
            target=target,
            radius=float(radius),
            context=context,
            candidate=candidate,
            spec=spec,
        )
        record["gates"] = scientific_gates(
            timestep=int(timestep),
            record=record,
            spec=spec,
        )
        records.append(record)

    valid = [
        record
        for record in records
        if record["gates"]["all"]
    ]
    upper_fidelity = [
        record
        for record in records
        if (
            record["gates"]["upper"]
            and record["gates"]["fidelity"]
        )
    ]
    lower_physical = [
        record
        for record in records
        if (
            record["gates"]["lower"]
            and record["gates"]["physical"]
            and record["gates"]["collapse"]
        )
    ]
    witness = (
        min(
            valid,
            key=lambda item: (
                float(item["radius"]),
                float(
                    item["final"][
                        "evaluation"
                    ][
                        "normalized_mse_ratio"
                    ]
                ),
                -float(
                    item["final"][
                        "evaluation"
                    ][
                        "historical_physical_row_any_rate"
                    ]
                ),
            ),
        )
        if valid
        else None
    )
    best_upper_fidelity = (
        min(
            upper_fidelity,
            key=lambda item: (
                float(item["radius"]),
                float(
                    item["final"][
                        "evaluation"
                    ][
                        "normalized_mse_ratio"
                    ]
                ),
            ),
        )
        if upper_fidelity
        else None
    )
    best_lower_physical = (
        min(
            lower_physical,
            key=lambda item: (
                float(item["radius"]),
                -float(
                    item["final"][
                        "evaluation"
                    ][
                        "historical_physical_row_any_rate"
                    ]
                ),
            ),
        )
        if lower_physical
        else None
    )
    return {
        "timestep":
            int(timestep),
        "upper_threshold":
            float(
                TIMESTEP_UPPER_MIN[
                    int(timestep)
                ]
            ),
        "records":
            records,
        "scientific_witness":
            witness,
        "scientific_reachable":
            bool(witness is not None),
        "upper_fidelity_reachable":
            bool(
                best_upper_fidelity
                is not None
            ),
        "lower_physical_reachable":
            bool(
                best_lower_physical
                is not None
            ),
        "best_upper_fidelity":
            best_upper_fidelity,
        "best_lower_physical":
            best_lower_physical,
    }


def historical_negative_control(
    base: Mapping[str, Any],
) -> Dict[str, Any]:
    result = base["worker_result"]
    records = {}
    for timestep in ("10", "25", "50"):
        source = result[
            "timestep_records"
        ][timestep]
        records[timestep] = {
            "timestep": int(timestep),
            "geometry_gradient_alignment":
                copy.deepcopy(
                    source[
                        "geometry_gradient_alignment"
                    ]
                ),
            "direct_x0_oracle":
                copy.deepcopy(
                    source[
                        "direct_x0_oracle"
                    ]
                ),
        }
    return {
        "candidate_id":
            "upper_only_reference",
        "definition":
            asdict(
                CANDIDATE_DEFINITIONS[0]
            ),
        "source":
            "immutable_stageb_evidence",
        "candidate_run":
            False,
        "timestep_records":
            records,
        "eligible":
            False,
        "negative_control":
            True,
        "base_root_cause":
            result["root_cause"],
        "base_contract_sha256":
            EXPECTED_BASE_CONTRACT_SHA256,
        "base_selection_sha256":
            EXPECTED_BASE_SELECTION_SHA256,
    }


def candidate_record(
    *,
    definition: BalancedCandidateDefinition,
    contract: Mapping[str, Any],
    context: Mapping[str, Any],
    spec: BalancedGeometrySpec,
) -> Dict[str, Any]:
    if definition.role != "selectable":
        raise ValueError(
            "negative control cannot be executed as a selectable candidate"
        )
    timesteps: MutableMapping[
        str,
        Any,
    ] = {}
    for timestep in spec.timesteps:
        control = context[
            "control_predictions"
        ][int(timestep)]
        target = context[
            "holdout_target"
        ]
        alignment = (
            objective_gradient_alignment(
                control=control,
                target=target,
                context=context,
                candidate=contract,
                spec=spec,
            )
        )
        sweep = balanced_oracle_sweep(
            control=control,
            target=target,
            timestep=int(timestep),
            context=context,
            candidate=contract,
            spec=spec,
        )
        timesteps[str(timestep)] = {
            "timestep": int(timestep),
            "control_prediction_sha256":
                sha256_array(control),
            "target_evaluation_sha256":
                sha256_array(target),
            "gradient_alignment":
                alignment,
            "oracle":
                sweep,
        }
    direction_all = bool(
        all(
            record[
                "gradient_alignment"
            ]["direction_pass"]
            for record
            in timesteps.values()
        )
    )
    witness_all = bool(
        all(
            record[
                "oracle"
            ]["scientific_reachable"]
            for record
            in timesteps.values()
        )
    )
    eligible = bool(
        witness_all
    )
    witness_radii = {
        key: (
            float(
                record[
                    "oracle"
                ][
                    "scientific_witness"
                ]["radius"]
            )
            if record[
                "oracle"
            ][
                "scientific_witness"
            ]
            is not None
            else None
        )
        for key, record
        in timesteps.items()
    }
    return {
        "candidate_id":
            definition.candidate_id,
        "definition":
            asdict(definition),
        "candidate_contract":
            copy.deepcopy(contract),
        "source":
            "current_worker_direct_x0_oracle",
        "candidate_run":
            True,
        "timestep_records":
            dict(timesteps),
        "direction_all_pass":
            direction_all,
        "scientific_witness_all_timesteps":
            witness_all,
        "witness_radii":
            witness_radii,
        "eligible":
            eligible,
        "negative_control":
            False,
    }


def _candidate_selection_key(
    record: Mapping[str, Any],
) -> Tuple[Any, ...]:
    timestep_records = record[
        "timestep_records"
    ]
    gradient_cosines = [
        float(
            timestep_records[
                str(timestep)
            ][
                "gradient_alignment"
            ][
                "descent_target_cosine"
            ]["mean"]
        )
        for timestep in (10, 25, 50)
    ]
    witnesses = [
        timestep_records[
            str(timestep)
        ]["oracle"][
            "scientific_witness"
        ]
        for timestep in (10, 25, 50)
    ]
    physical = [
        float(
            witness["final"][
                "evaluation"
            ][
                "historical_physical_row_any_rate"
            ]
        )
        for witness in witnesses
    ]
    nmse = [
        float(
            witness["final"][
                "evaluation"
            ][
                "normalized_mse_ratio"
            ]
        )
        for witness in witnesses
    ]
    collapse = [
        float(
            witness["final"][
                "evaluation"
            ][
                "segment_length"
            ]["collapse_fraction"]
        )
        for witness in witnesses
    ]
    movement_cosine = [
        float(
            witness["final"][
                "evaluation"
            ][
                "target_direction_cosine"
            ]["mean"]
        )
        for witness in witnesses
    ]
    distance_reduction = [
        float(
            witness["final"][
                "evaluation"
            ][
                "target_distance_reduction_fraction"
            ]["mean"]
        )
        for witness in witnesses
    ]
    definition = record["definition"]
    return (
        -min(physical),
        max(nmse),
        max(collapse),
        -min(movement_cosine),
        -min(distance_reduction),
        float(definition["anchor_weight"]),
        float(definition["lower_weight"]),
        float(definition["lower_quantile"]),
        -min(gradient_cosines),
        str(record["candidate_id"]),
    )


def select_objective(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    eligible = [
        record
        for record in records
        if (
            record.get("eligible")
            and not record.get(
                "negative_control"
            )
        )
    ]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=_candidate_selection_key,
    )
    return {
        "candidate_id":
            selected["candidate_id"],
        "definition":
            copy.deepcopy(
                selected["definition"]
            ),
        "candidate_contract_sha256":
            selected[
                "candidate_contract"
            ]["contract_sha256"],
        "witness_radii":
            copy.deepcopy(
                selected["witness_radii"]
            ),
        "selection_key":
            list(
                _candidate_selection_key(
                    selected
                )
            ),
    }


def classify_objectives(
    *,
    records: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    selectable = [
        record
        for record in records
        if not record.get(
            "negative_control"
        )
    ]
    eligible = [
        record
        for record in selectable
        if record["eligible"]
    ]
    all_upper_fidelity = []
    all_lower_physical = []
    any_scientific_timestep = []
    direction_all = []
    for record in selectable:
        timestep_records = record[
            "timestep_records"
        ]
        all_upper_fidelity.append(
            all(
                item["oracle"][
                    "upper_fidelity_reachable"
                ]
                for item
                in timestep_records.values()
            )
        )
        all_lower_physical.append(
            all(
                item["oracle"][
                    "lower_physical_reachable"
                ]
                for item
                in timestep_records.values()
            )
        )
        any_scientific_timestep.append(
            any(
                item["oracle"][
                    "scientific_reachable"
                ]
                for item
                in timestep_records.values()
            )
        )
        direction_all.append(
            bool(
                record[
                    "direction_all_pass"
                ]
            )
        )

    anchor_eligible = [
        record
        for record in eligible
        if float(
            record[
                "definition"
            ]["anchor_weight"]
        )
        > 0.0
    ]
    plain_eligible = [
        record
        for record in eligible
        if float(
            record[
                "definition"
            ]["anchor_weight"]
        )
        == 0.0
    ]

    if selected is not None:
        selected_anchor = float(
            selected[
                "definition"
            ]["anchor_weight"]
        )
        if (
            selected_anchor > 0.0
            and not plain_eligible
        ):
            root = (
                "phase314b_r258_stagec_distribution_anchor_"
                "required_for_balanced_objective_reachability"
            )
            next_path = (
                "CALIBRATE_SELECTED_ANCHORED_BALANCED_"
                "OBJECTIVE_IN_TRAIN_ONLY_DENOISER"
            )
            locus = "distribution_anchor_required"
        else:
            root = (
                "phase314b_r258_stagec_balanced_segment_"
                "objective_train_only_oracle_selected"
            )
            next_path = (
                "CALIBRATE_SELECTED_BALANCED_SEGMENT_"
                "OBJECTIVE_IN_TRAIN_ONLY_DENOISER"
            )
            locus = "balanced_objective_selected"
    elif any(
        upper and not lower
        for upper, lower
        in zip(
            all_upper_fidelity,
            all_lower_physical,
        )
    ):
        root = (
            "phase314b_r258_stagec_balanced_objectives_"
            "still_use_nonphysical_or_collapse_shortcut"
        )
        next_path = (
            "ADD_TOPOLOGY_AND_PAIRWISE_DISTANCE_BARRIERS_"
            "TO_DIRECT_X0_OBJECTIVE_ORACLE"
        )
        locus = "balanced_objective_nonphysical_shortcut"
    elif any(
        lower and not upper
        for upper, lower
        in zip(
            all_upper_fidelity,
            all_lower_physical,
        )
    ):
        root = (
            "phase314b_r258_stagec_lower_band_preserves_"
            "validity_but_blocks_upper_correction"
        )
        next_path = (
            "CALIBRATE_POSITIONWISE_ASYMMETRIC_BAND_"
            "FROM_OBJECTIVE_TRAIN_ONLY"
        )
        locus = "lower_band_overconstraint"
    elif any(any_scientific_timestep):
        root = (
            "phase314b_r258_stagec_balanced_objective_"
            "reachability_depends_on_timestep"
        )
        next_path = (
            "AUDIT_TIMESTEP_CONDITIONAL_BALANCED_"
            "OBJECTIVE_BEFORE_MODEL_CALIBRATION"
        )
        locus = "timestep_conditional"
    elif any(
        upper and direction
        for upper, direction
        in zip(
            all_upper_fidelity,
            direction_all,
        )
    ):
        root = (
            "phase314b_r258_stagec_balanced_objective_"
            "reaches_upper_fidelity_without_valid_cable_region"
        )
        next_path = (
            "ADD_TOPOLOGY_AND_PAIRWISE_DISTANCE_BARRIERS_"
            "TO_DIRECT_X0_OBJECTIVE_ORACLE"
        )
        locus = "missing_nonlocal_validity_terms"
    elif any(all_upper_fidelity) and not any(direction_all):
        root = (
            "phase314b_r258_stagec_valid_endpoint_reachable_"
            "but_balanced_local_descent_misaligned"
        )
        next_path = (
            "CALIBRATE_OBJECTIVE_TRAIN_DISTRIBUTION_"
            "DIRECTION_SURROGATE_WITHOUT_HOLDOUT_LEAKAGE"
        )
        locus = "local_descent_misalignment"
    else:
        root = (
            "phase314b_r258_stagec_registered_balanced_"
            "objectives_have_no_joint_reachability_solution"
        )
        next_path = (
            "AUDIT_NONLOCAL_CABLE_VALIDITY_TERMS_AND_"
            "POSITIONWISE_BAND_DESIGN"
        )
        locus = "no_joint_solution"

    return {
        "root_cause":
            root,
        "required_next_path":
            next_path,
        "primary_failure_locus":
            locus,
        "selectable_candidate_count":
            len(selectable),
        "eligible_candidate_count":
            len(eligible),
        "eligible_candidate_ids": [
            record["candidate_id"]
            for record in eligible
        ],
        "plain_eligible_candidate_ids": [
            record["candidate_id"]
            for record in plain_eligible
        ],
        "anchor_eligible_candidate_ids": [
            record["candidate_id"]
            for record in anchor_eligible
        ],
        "all_upper_fidelity":
            all_upper_fidelity,
        "all_lower_physical":
            all_lower_physical,
        "any_scientific_timestep":
            any_scientific_timestep,
        "direction_all":
            direction_all,
        "selected_configuration":
            copy.deepcopy(selected),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[BalancedGeometrySpec] = None,
) -> Dict[str, Any]:
    active_spec = (
        BalancedGeometrySpec()
        if spec is None
        else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    base = validate_base_evidence(
        repository_root
    )
    stagea258.validate_environment_payload(
        environment
    )
    cold = (
        stagea258
        .assert_cold_cuda_context_portable()
    )
    captured = (
        stageb258
        .capture_portable_control_model(
            root=repository_root
        )
    )
    context = (
        stageb258.build_audit_context(
            root=repository_root,
            captured=captured,
        )
    )
    reference = fit_balanced_reference(
        context=context,
        spec=active_spec,
    )

    records: List[Dict[str, Any]] = [
        historical_negative_control(
            base
        )
    ]
    candidate_contracts = {}
    for definition in CANDIDATE_DEFINITIONS[1:]:
        contract = candidate_contract(
            definition=definition,
            reference=reference,
            spec=active_spec,
        )
        candidate_contracts[
            definition.candidate_id
        ] = contract
        records.append(
            candidate_record(
                definition=definition,
                contract=contract,
                context=context,
                spec=active_spec,
            )
        )

    selected = select_objective(
        records
    )
    classification = classify_objectives(
        records=records,
        selected=selected,
    )
    contract = {
        "schema":
            "phase314b_r258_stagec_balanced_geometry_contract_v1",
        "phase": PHASE,
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "base_identity": {
            "worker_sha256":
                EXPECTED_BASE_WORKER_SHA256,
            "contract_sha256":
                EXPECTED_BASE_CONTRACT_SHA256,
            "selection_sha256":
                EXPECTED_BASE_SELECTION_SHA256,
            "root_cause":
                EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path":
                EXPECTED_BASE_NEXT_PATH,
        },
        "environment":
            dict(environment),
        "cold_main_worker_context":
            cold,
        "control_capture":
            captured[
                "control_identity"
            ],
        "balanced_geometry_spec":
            asdict(active_spec),
        "candidate_definitions": [
            asdict(definition)
            for definition
            in CANDIDATE_DEFINITIONS
        ],
        "balanced_reference":
            reference,
        "candidate_contracts":
            candidate_contracts,
        "negative_control_rule": (
            "upper_only_reference is loaded from immutable Stage-B "
            "evidence and cannot be selected"
        ),
        "objective_fit_rule": (
            "lower and anchor boundaries use only the 638-row "
            "objective-training target population"
        ),
        "selection_evaluation_rule": (
            "236-row grouped holdout target is evaluation-only and "
            "does not fit any objective boundary or weight"
        ),
        "model_architecture_changed":
            False,
        "model_candidate_trained":
            False,
        "uses_frozen_probe":
            False,
    }
    contract["contract_sha256"] = (
        sha256_bytes(
            stable_json_bytes(contract)
        )
    )
    selection_payload = {
        "selected_configuration":
            selected,
        "train_only_recommendation":
            selected,
        "classification":
            classification,
        "candidate_records":
            records,
        "uses_selection_holdout_target_for_fit":
            False,
        "uses_selection_holdout_target_for_evaluation":
            True,
        "uses_frozen_probe":
            False,
    }
    selection_payload[
        "selection_sha256"
    ] = sha256_bytes(
        stable_json_bytes(
            selection_payload
        )
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema":
            "phase314b_r258_stagec_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": (
            "READY_FOR_TRAIN_ONLY_MODEL_CALIBRATION"
            if selected is not None
            else "BLOCKED"
        ),
        "root_cause":
            classification[
                "root_cause"
            ],
        "required_next_path":
            classification[
                "required_next_path"
            ],
        "immutable_inputs": {
            "base_evidence_commit":
                BASE_EVIDENCE_COMMIT,
            "base_worker_sha256":
                EXPECTED_BASE_WORKER_SHA256,
            "base_contract_sha256":
                EXPECTED_BASE_CONTRACT_SHA256,
            "base_selection_sha256":
                EXPECTED_BASE_SELECTION_SHA256,
        },
        "environment":
            dict(environment),
        "cold_main_worker_context":
            cold,
        "control_capture": {
            **captured[
                "control_identity"
            ],
            "train_candidate_call_count":
                captured[
                    "train_candidate_call_count"
                ],
            "capture_wrapper_restored":
                captured[
                    "capture_wrapper_restored"
                ],
        },
        "split": {
            "stageb_training_rows":
                int(
                    np.sum(
                        context[
                            "stageb_train_mask"
                        ]
                    )
                ),
            "objective_train_rows":
                int(
                    np.sum(
                        context[
                            "objective_train_mask"
                        ]
                    )
                ),
            "selection_holdout_rows":
                int(
                    np.sum(
                        context[
                            "selection_holdout_mask"
                        ]
                    )
                ),
            "frozen_probe_rows":
                int(
                    np.sum(
                        context[
                            "frozen_probe_mask"
                        ]
                    )
                ),
            **context[
                "selection_split"
            ],
            "frozen_probe_accessed":
                False,
        },
        "balanced_reference":
            reference,
        "candidate_records":
            records,
        "classification":
            classification,
        "balanced_geometry_contract":
            contract,
        "selection":
            selection_payload,
        "selected_configuration":
            selected,
        "train_only_recommendation":
            selected,
        "new_model_candidate_trained":
            False,
        "direct_x0_tensor_optimization_run":
            True,
        "balanced_objective_calibration_run":
            True,
        "control_replay_exact":
            bool(
                captured[
                    "control_identity"
                ][
                    "reference_equivalence_pass"
                ]
            ),
        "frozen_probe_accessed":
            False,
        "reverse_sampling_run":
            False,
        "full_stageb_repaired_model_trained":
            False,
        "formal_pilot_run":
            False,
        "checkpoint_saved":
            False,
        "weights_persisted":
            False,
        "prediction_tensor_persisted":
            False,
        "oracle_tensor_persisted":
            False,
        "candidate_tensor_persisted":
            False,
        "npz_saved":
            False,
        "cache_saved":
            False,
        "formal_diffusion_training":
            False,
        "formal_reverse_sampling":
            False,
        "formal_idm_training":
            False,
        "action_diverse_data_collection":
            False,
        "candidate_execution":
            False,
        "deformable_ravens_executed":
            False,
        "phase4":
            False,
        "cps":
            False,
    }


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause":
            result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "scientific_status":
            result["scientific_status"],
        "environment":
            result["environment"],
        "cold_main_worker_context":
            result[
                "cold_main_worker_context"
            ],
        "control_capture":
            result["control_capture"],
        "split":
            result["split"],
        "balanced_reference":
            result["balanced_reference"],
        "candidate_records":
            result["candidate_records"],
        "classification":
            result["classification"],
        "balanced_geometry_contract":
            result[
                "balanced_geometry_contract"
            ],
        "selection":
            result["selection"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(
        identity_projection(left)
    )
    right_payload = stable_json_bytes(
        identity_projection(right)
    )
    return {
        "exact":
            left_payload
            == right_payload,
        "left_sha256":
            sha256_bytes(
                left_payload
            ),
        "right_sha256":
            sha256_bytes(
                right_payload
            ),
        "environment_exact": (
            left["environment"]
            == right["environment"]
        ),
        "control_capture_exact": (
            left["control_capture"]
            == right["control_capture"]
        ),
        "reference_exact": (
            left["balanced_reference"]
            == right["balanced_reference"]
        ),
        "candidate_records_exact": (
            left["candidate_records"]
            == right["candidate_records"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
        "contract_exact": (
            left[
                "balanced_geometry_contract"
            ]
            == right[
                "balanced_geometry_contract"
            ]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
    }
