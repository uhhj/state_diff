"""Phase3.14b-r2.5.8 Stage D direction-surrogate calibration.

Stage C established that six lower/upper segment-length objectives eliminate
neither the missing spatial direction nor the lack of a scientific direct-x0
witness.  Their initial target-direction cosine is only approximately
0.019--0.044.  Segment lengths describe local scale, but do not identify the
conditioned spatial displacement, curvature direction, or target configuration.

This stage asks whether the 638-row objective-training population contains a
group-generalizing direction signal.  It trains only small deterministic
CPU ridge/reduced-rank ridge surrogates.  Candidate selection uses six-fold
grouped out-of-fold predictions inside objective train.  The 236-row grouped
selection holdout is evaluated exactly once after the candidate and per-timestep
scale have been frozen.  The 126-row frozen probe remains closed.

No diffusion-model candidate is trained.  The frozen direct-x0 control is
replayed exactly and retained only in memory.  No surrogate weights, prediction
tensor, checkpoint, NPZ, cache, image, or video is persisted.
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

from ccda_phase3 import schema_v3
from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r258_stageb_direct_x0_reachability as stageb258
from ccda_phase3 import phase314b_r258_stagec_balanced_geometry as stagec258
from ccda_phase3 import phase314b_r258_stagec_resume1_commit_recovery as stagec_resume1

PHASE = "Phase3.14b-r2.5.8 Stage D"
PHASE_ID = "phase314b_r258_staged"

BASE_EVIDENCE_COMMIT = (
    "6866507c42b9bc9d2d271becd1a9423f61710405"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

BASE_CONTRACT = (
    "reports/phase3_14b_r258_stagec_resume1_contract.json"
)
BASE_WORKER = (
    "reports/phase3_14b_r258_stagec_resume1_worker_evidence.json"
)
BASE_SUMMARY = (
    "reports/phase3_14b_r258_stagec_resume1_summary.json"
)
BASE_REPORT = (
    "reports/phase3_14b_r258_stagec_resume1_report.md"
)
BASE_TEST_GATE = (
    "reports/phase3_14b_r258_stagec_resume1_test_gate_summary.json"
)
BASE_BOUND_FILES = (
    BASE_CONTRACT,
    BASE_WORKER,
    BASE_SUMMARY,
    BASE_REPORT,
    BASE_TEST_GATE,
)
SOURCE_BOUND_FILES = (
    "ccda_phase3/schema_v3.py",
    "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py",
    "ccda_phase3/phase314b_r258_stageb_direct_x0_reachability.py",
    "ccda_phase3/phase314b_r258_stagec_balanced_geometry.py",
    "ccda_phase3/phase314b_r258_stagec_resume1_commit_recovery.py",
)

EXPECTED_BASE_WORKER_SHA256 = (
    "8af4981760c2959667379221215c067fc"
    "d838b3e73349ab2af5517e197806913"
)
EXPECTED_BASE_CONTRACT_SHA256 = (
    "355e3a12e13533b4a6f283344cfda1be"
    "20c4d6a4bef2fdd28f2ba03b3397b696"
)
EXPECTED_BASE_SELECTION_SHA256 = (
    "613dd19b4611fddd24e5a5a71c29054b"
    "8a668a868d40198caccb42c82f4b3f50"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stagec_valid_endpoint_reachable_"
    "but_balanced_local_descent_misaligned"
)
EXPECTED_BASE_NEXT_PATH = (
    "CALIBRATE_OBJECTIVE_TRAIN_DISTRIBUTION_DIRECTION_"
    "SURROGATE_WITHOUT_HOLDOUT_LEAKAGE"
)
EXPECTED_STAGEC_CALIBRATION_SHA256 = (
    "fb91d29c84cedccf8f72458f08c2f213"
    "bbbc542663593094c0ab651701a052f0"
)
EXPECTED_STAGEC_CONTROL_SHA256 = (
    "dee61b8e31455eeb0e7e0eceae5c1500"
    "e3ce57ca8bd42989378d9f76fd779ae7"
)
EXPECTED_OBJECTIVE_FIT_TARGET_SHA256 = (
    "db5e699a29a342e09075b824b7532aff"
    "234bd24d60909079d871b9ab8f309afa"
)
EXPECTED_OBJECTIVE_FIT_GROUP_SHA256 = (
    "ba071d0d3dfae273a9b74e64bb4623a6"
    "abf8b897e8cbe6ccd85dee29ce2506cf"
)

TIMESTEP_UPPER_MIN = {
    10: 0.75,
    25: 0.60,
    50: 0.50,
}
CONDITION_VALUES = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)
CABLE_ANCHOR_INDICES = (
    0,
    3,
    6,
    10,
    13,
    17,
    20,
    23,
)
SEGMENT_ANCHOR_INDICES = (
    0,
    3,
    6,
    9,
    13,
    16,
    19,
    22,
)


class DirectionSurrogateError(RuntimeError):
    """Raised when evidence, split, model, or leakage invariants fail."""


@dataclass(frozen=True)
class SurrogateDefinition:
    candidate_id: str
    feature_mode: str
    model_mode: str
    ridge_alpha: float
    output_rank: int
    role: str

    def validate(self) -> None:
        if self.feature_mode not in (
            "none",
            "condition_only",
            "centered_anchor",
            "segment_anchor",
            "absolute_anchor",
        ):
            raise ValueError("unknown feature mode")
        if self.model_mode not in (
            "global_mean",
            "condition_mean",
            "ridge",
            "reduced_rank_ridge",
        ):
            raise ValueError("unknown model mode")
        if self.role not in (
            "negative_control",
            "diagnostic_control",
            "selectable",
        ):
            raise ValueError("unknown candidate role")
        if self.model_mode in (
            "ridge",
            "reduced_rank_ridge",
        ):
            if self.ridge_alpha <= 0.0:
                raise ValueError("ridge alpha must be positive")
        if self.model_mode == "reduced_rank_ridge":
            if self.output_rank not in (16, 32):
                raise ValueError("reduced rank changed")
        elif self.output_rank != 0:
            raise ValueError("non-reduced model has output rank")
        if self.role == "selectable" and self.feature_mode not in (
            "centered_anchor",
            "segment_anchor",
        ):
            raise ValueError(
                "selectable candidate must use translation-invariant features"
            )
        if self.role == "negative_control" and self.model_mode not in (
            "global_mean",
            "condition_mean",
        ):
            raise ValueError("negative control model changed")
        if self.role == "diagnostic_control" and self.feature_mode != (
            "absolute_anchor"
        ):
            raise ValueError("diagnostic control must use absolute coordinates")


SURROGATE_DEFINITIONS = (
    SurrogateDefinition(
        "global_mean",
        "none",
        "global_mean",
        0.0,
        0,
        "negative_control",
    ),
    SurrogateDefinition(
        "condition_mean",
        "condition_only",
        "condition_mean",
        0.0,
        0,
        "negative_control",
    ),
    SurrogateDefinition(
        "absolute_rridge_k32_a10",
        "absolute_anchor",
        "reduced_rank_ridge",
        10.0,
        32,
        "diagnostic_control",
    ),
    SurrogateDefinition(
        "centered_ridge_a1",
        "centered_anchor",
        "ridge",
        1.0,
        0,
        "selectable",
    ),
    SurrogateDefinition(
        "centered_ridge_a10",
        "centered_anchor",
        "ridge",
        10.0,
        0,
        "selectable",
    ),
    SurrogateDefinition(
        "centered_rridge_k16_a10",
        "centered_anchor",
        "reduced_rank_ridge",
        10.0,
        16,
        "selectable",
    ),
    SurrogateDefinition(
        "centered_rridge_k32_a10",
        "centered_anchor",
        "reduced_rank_ridge",
        10.0,
        32,
        "selectable",
    ),
    SurrogateDefinition(
        "segment_rridge_k16_a10",
        "segment_anchor",
        "reduced_rank_ridge",
        10.0,
        16,
        "selectable",
    ),
    SurrogateDefinition(
        "segment_rridge_k32_a10",
        "segment_anchor",
        "reduced_rank_ridge",
        10.0,
        32,
        "selectable",
    ),
)


@dataclass(frozen=True)
class DirectionSurrogateSpec:
    timesteps: Tuple[int, ...] = (10, 25, 50)
    grouped_cv_folds: int = 6
    objective_train_noise_seed_offset: int = 8801
    scale_grid: Tuple[float, ...] = (
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
    )

    oof_mean_cosine_min: float = 0.25
    oof_nonnegative_rate_min: float = 0.70
    oof_condition_cosine_min: float = 0.20
    oof_fold_cosine_min: float = 0.10
    global_baseline_margin_min: float = 0.10
    condition_baseline_margin_min: float = 0.05
    permutation_cosine_max: float = 0.10
    permutation_margin_min: float = 0.10

    state_fidelity_ratio_max: float = 0.95
    state_lower_pass_min: float = 0.90
    state_physical_valid_min: float = 0.90
    state_collapse_fraction_max: float = 0.05
    state_stretch_fraction_max: float = 0.05
    state_movement_cosine_min: float = 0.25
    state_target_distance_reduction_min: float = 0.05

    holdout_mean_cosine_min: float = 0.20
    holdout_nonnegative_rate_min: float = 0.65
    holdout_condition_cosine_min: float = 0.15

    translation_offset_xy: Tuple[float, float] = (
        0.375,
        -0.625,
    )
    translation_tolerance: float = 1.0e-6
    standardizer_epsilon: float = 1.0e-12
    ridge_jitter: float = 1.0e-10
    permutation_seed: int = 99173

    def validate(self) -> None:
        if self.timesteps != tuple(sorted(TIMESTEP_UPPER_MIN)):
            raise ValueError("timestep population changed")
        if self.grouped_cv_folds != 6:
            raise ValueError("grouped CV fold count changed")
        if self.objective_train_noise_seed_offset != 8801:
            raise ValueError("objective-train noise offset changed")
        if self.scale_grid != tuple(sorted(set(self.scale_grid))):
            raise ValueError("scale grid must be ordered and unique")
        if any(float(value) <= 0.0 for value in self.scale_grid):
            raise ValueError("scale must be positive")
        if CABLE_ANCHOR_INDICES != tuple(sorted(set(CABLE_ANCHOR_INDICES))):
            raise ValueError("cable anchor population changed")
        if SEGMENT_ANCHOR_INDICES != tuple(
            sorted(set(SEGMENT_ANCHOR_INDICES))
        ):
            raise ValueError("segment anchor population changed")
        if CABLE_ANCHOR_INDICES[-1] >= stageb.BEADS:
            raise ValueError("cable anchor outside bead population")
        if SEGMENT_ANCHOR_INDICES[-1] >= stageb.BEADS - 1:
            raise ValueError("segment anchor outside segment population")
        for definition in SURROGATE_DEFINITIONS:
            definition.validate()
        if tuple(
            definition.candidate_id
            for definition in SURROGATE_DEFINITIONS
        ) != (
            "global_mean",
            "condition_mean",
            "absolute_rridge_k32_a10",
            "centered_ridge_a1",
            "centered_ridge_a10",
            "centered_rridge_k16_a10",
            "centered_rridge_k32_a10",
            "segment_rridge_k16_a10",
            "segment_rridge_k32_a10",
        ):
            raise ValueError("surrogate candidate order changed")
        for value in (
            self.oof_mean_cosine_min,
            self.oof_nonnegative_rate_min,
            self.oof_condition_cosine_min,
            self.global_baseline_margin_min,
            self.condition_baseline_margin_min,
            self.permutation_margin_min,
            self.state_fidelity_ratio_max,
            self.state_lower_pass_min,
            self.state_physical_valid_min,
            self.state_movement_cosine_min,
            self.state_target_distance_reduction_min,
            self.holdout_mean_cosine_min,
            self.holdout_nonnegative_rate_min,
            self.holdout_condition_cosine_min,
        ):
            if float(value) <= 0.0:
                raise ValueError("positive threshold is invalid")
        if self.translation_tolerance <= 0.0:
            raise ValueError("translation tolerance is invalid")
        if self.standardizer_epsilon <= 0.0:
            raise ValueError("standardizer epsilon is invalid")
        if self.ridge_jitter <= 0.0:
            raise ValueError("ridge jitter is invalid")


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


def sha256_strings(values: Sequence[Any]) -> str:
    return sha256_bytes(
        "\n".join(str(value) for value in values).encode("utf-8")
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
        raise DirectionSurrogateError(
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
        raise DirectionSurrogateError(
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
        raise DirectionSurrogateError(
            "base-bound file differs: {}".format(relative)
        )
    return sha256_bytes(observed)


def validate_source_binding(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    source_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            BASE_EVIDENCE_COMMIT,
        )
        for relative in SOURCE_BOUND_FILES
    }
    schema_v3.SchemaV3Manifest().validate()
    if schema_v3.ROBOT_PROXY_DIM != 19:
        raise DirectionSurrogateError(
            "robot-proxy dimension changed"
        )
    if schema_v3.ROBOT_EE_POSITION_SLICE != slice(12, 15):
        raise DirectionSurrogateError(
            "EE position slice changed"
        )
    if schema_v3.ROBOT_EE_QUATERNION_SLICE != slice(15, 19):
        raise DirectionSurrogateError(
            "EE quaternion slice changed"
        )
    if stageb.CONDITION_DIM != 243:
        raise DirectionSurrogateError(
            "diffusion condition dimension changed"
        )
    if stageb.CABLE_DIM != 48 or stageb.FUTURE_STEPS != 4:
        raise DirectionSurrogateError(
            "cable target layout changed"
        )
    return {
        "source_sha256":
            source_sha,
        "robot_proxy_dim":
            int(
                schema_v3.ROBOT_PROXY_DIM
            ),
        "robot_joint_position_slice":
            [
                schema_v3
                .ROBOT_JOINT_POSITION_SLICE
                .start,
                schema_v3
                .ROBOT_JOINT_POSITION_SLICE
                .stop,
            ],
        "robot_joint_velocity_slice":
            [
                schema_v3
                .ROBOT_JOINT_VELOCITY_SLICE
                .start,
                schema_v3
                .ROBOT_JOINT_VELOCITY_SLICE
                .stop,
            ],
        "robot_ee_position_slice":
            [
                schema_v3
                .ROBOT_EE_POSITION_SLICE
                .start,
                schema_v3
                .ROBOT_EE_POSITION_SLICE
                .stop,
            ],
        "robot_ee_quaternion_slice":
            [
                schema_v3
                .ROBOT_EE_QUATERNION_SLICE
                .start,
                schema_v3
                .ROBOT_EE_QUATERNION_SLICE
                .stop,
            ],
        "condition_dim":
            int(stageb.CONDITION_DIM),
        "target_shape": [
            int(stageb.FUTURE_STEPS),
            int(stageb.CABLE_DIM),
        ],
    }


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
    source_binding = validate_source_binding(
        repository_root
    )
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
        raise DirectionSurrogateError(
            "Stage-C Resume1 verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise DirectionSurrogateError(
            "Stage-C Resume1 scientific status changed"
        )
    if summary.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise DirectionSurrogateError(
            "Stage-C Resume1 root cause changed"
        )
    if summary.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise DirectionSurrogateError(
            "Stage-C Resume1 next path changed"
        )
    if summary.get("selected_configuration") is not None:
        raise DirectionSurrogateError(
            "Stage-C Resume1 selected a configuration"
        )
    if summary.get("train_only_recommendation") is not None:
        raise DirectionSurrogateError(
            "Stage-C Resume1 emitted a recommendation"
        )
    if summary.get("workers_exact") is not True:
        raise DirectionSurrogateError(
            "Stage-C Resume1 workers are not exact"
        )

    comparison = worker.get("comparison")
    if not isinstance(comparison, dict):
        raise DirectionSurrogateError(
            "Stage-C Resume1 worker comparison is missing"
        )
    if comparison.get("exact") is not True:
        raise DirectionSurrogateError(
            "Stage-C Resume1 worker comparison changed"
        )
    if (
        comparison.get("left_sha256")
        != EXPECTED_BASE_WORKER_SHA256
        or comparison.get("right_sha256")
        != EXPECTED_BASE_WORKER_SHA256
    ):
        raise DirectionSurrogateError(
            "Stage-C Resume1 worker identity changed"
        )
    result = worker.get("worker_result")
    if not isinstance(result, dict):
        raise DirectionSurrogateError(
            "Stage-C Resume1 worker result is missing"
        )
    if result.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise DirectionSurrogateError(
            "Stage-C Resume1 worker root cause changed"
        )
    if result.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise DirectionSurrogateError(
            "Stage-C Resume1 worker next path changed"
        )
    if (
        result.get(
            "balanced_geometry_contract",
            {},
        ).get("contract_sha256")
        != EXPECTED_BASE_CONTRACT_SHA256
    ):
        raise DirectionSurrogateError(
            "Stage-C Resume1 internal contract changed"
        )
    if (
        result.get("selection", {}).get("selection_sha256")
        != EXPECTED_BASE_SELECTION_SHA256
    ):
        raise DirectionSurrogateError(
            "Stage-C Resume1 selection changed"
        )
    if contract.get("contract_sha256") != EXPECTED_BASE_CONTRACT_SHA256:
        raise DirectionSurrogateError(
            "Stage-C Resume1 contract record changed"
        )
    if contract.get("selection_sha256") != EXPECTED_BASE_SELECTION_SHA256:
        raise DirectionSurrogateError(
            "Stage-C Resume1 contract selection changed"
        )

    control = result.get("control_capture", {})
    if control.get("reference_equivalence_pass") is not True:
        raise DirectionSurrogateError(
            "Stage-C Resume1 control reference equivalence failed"
        )
    if (
        control.get("observed_calibration_sha256")
        != EXPECTED_STAGEC_CALIBRATION_SHA256
    ):
        raise DirectionSurrogateError(
            "Stage-C calibration replay changed"
        )
    if (
        control.get("observed_control_sha256")
        != EXPECTED_STAGEC_CONTROL_SHA256
    ):
        raise DirectionSurrogateError(
            "Stage-C control replay changed"
        )
    if control.get("stagec_nonzero_candidate_training_count") != 0:
        raise DirectionSurrogateError(
            "Stage-C replay trained a nonzero candidate"
        )

    reference = result.get("balanced_reference", {})
    if reference.get("fit_rows") != 638:
        raise DirectionSurrogateError(
            "objective-fit row count changed"
        )
    if reference.get("fit_group_count") != 126:
        raise DirectionSurrogateError(
            "objective-fit group count changed"
        )
    if (
        reference.get("fit_target_sha256")
        != EXPECTED_OBJECTIVE_FIT_TARGET_SHA256
    ):
        raise DirectionSurrogateError(
            "objective-fit target identity changed"
        )
    if (
        reference.get("fit_group_sha256")
        != EXPECTED_OBJECTIVE_FIT_GROUP_SHA256
    ):
        raise DirectionSurrogateError(
            "objective-fit group identity changed"
        )
    if reference.get("uses_selection_holdout_target") is not False:
        raise DirectionSurrogateError(
            "Stage-C objective fit used selection holdout"
        )
    if reference.get("uses_frozen_probe") is not False:
        raise DirectionSurrogateError(
            "Stage-C objective fit used frozen probe"
        )

    candidates = result.get("candidate_records")
    if not isinstance(candidates, list) or len(candidates) != 7:
        raise DirectionSurrogateError(
            "Stage-C candidate population changed"
        )
    if any(
        bool(record.get("eligible"))
        for record in candidates
    ):
        raise DirectionSurrogateError(
            "Stage-C unexpectedly has an eligible objective"
        )

    if test_gate.get("verdict") != "PASS":
        raise DirectionSurrogateError(
            "Stage-C Resume1 test gate is not PASS"
        )
    if int(test_gate.get("test_file_count", -1)) != 51:
        raise DirectionSurrogateError(
            "Stage-C Resume1 test-file count changed"
        )
    if int(test_gate.get("passed_test_count", -1)) != 1064:
        raise DirectionSurrogateError(
            "Stage-C Resume1 pass count changed"
        )
    if int(test_gate.get("resume1_new_passed", -1)) != 41:
        raise DirectionSurrogateError(
            "Stage-C Resume1 new-test count changed"
        )

    for key in (
        "new_model_candidate_trained",
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
            raise DirectionSurrogateError(
                "Stage-C Resume1 boundary changed: {}".format(key)
            )

    return {
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "file_sha256": file_sha,
        "source_binding":
            source_binding,
        "contract": contract,
        "worker": worker,
        "worker_result": result,
        "summary": summary,
        "test_gate": test_gate,
    }


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    return stageb258._safe_stats(
        np.asarray(value, dtype=np.float64)
    )


def deterministic_group_folds(
    groups: Sequence[Any],
    *,
    folds: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    values = np.asarray(groups).astype(str)
    unique = sorted(set(values.tolist()))
    if len(unique) < int(folds):
        raise DirectionSurrogateError(
            "group population is smaller than fold count"
        )
    mapping = {
        group: index % int(folds)
        for index, group in enumerate(unique)
    }
    assignment = np.asarray(
        [mapping[group] for group in values],
        dtype=np.int64,
    )
    for group in unique:
        observed = set(
            assignment[
                values == group
            ].tolist()
        )
        if len(observed) != 1:
            raise AssertionError(
                "group crossed direction-surrogate folds"
            )
    counts = [
        int(np.sum(assignment == fold))
        for fold in range(int(folds))
    ]
    if any(count <= 0 for count in counts):
        raise DirectionSurrogateError(
            "direction-surrogate fold is empty"
        )
    return assignment, mapping


def condition_one_hot(
    condition_name: Sequence[Any],
) -> np.ndarray:
    names = np.asarray(
        condition_name
    ).astype(str)
    unknown = sorted(
        set(names.tolist())
        - set(CONDITION_VALUES)
    )
    if unknown:
        raise DirectionSurrogateError(
            "unknown condition values: {}".format(unknown)
        )
    return np.stack(
        [
            names == CONDITION_VALUES[0],
            names == CONDITION_VALUES[1],
        ],
        axis=1,
    ).astype(np.float64)


def _condition_parts(
    condition: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(
        condition,
        dtype=np.float64,
    )
    if value.ndim != 2 or value.shape[1] != stageb.CONDITION_DIM:
        raise ValueError(
            "condition shape changed: {}".format(value.shape)
        )
    history_flat = value[:, : stageb.HISTORY_STEPS * stageb.STATE_DIM]
    actions = value[:, stageb.HISTORY_STEPS * stageb.STATE_DIM :]
    history = history_flat.reshape(
        value.shape[0],
        stageb.HISTORY_STEPS,
        stageb.STATE_DIM,
    )
    cable = history[:, :, : stageb.CABLE_DIM].reshape(
        value.shape[0],
        stageb.HISTORY_STEPS,
        stageb.BEADS,
        2,
    )
    robot = history[:, :, stageb.CABLE_DIM :]
    if actions.shape[1] != stageb.HISTORY_STEPS * stageb.ACTION_DIM:
        raise ValueError("action-history shape changed")
    if robot.shape[2] != stageb.STATE_DIM - stageb.CABLE_DIM:
        raise ValueError("robot-proxy shape changed")
    return cable, robot, actions


def _control_points(
    control: np.ndarray,
) -> np.ndarray:
    value = np.asarray(
        control,
        dtype=np.float64,
    )
    expected = (
        value.shape[0],
        stageb.FUTURE_STEPS,
        stageb.CABLE_DIM,
    )
    if value.shape != expected:
        raise ValueError(
            "control shape changed: {}".format(value.shape)
        )
    return value.reshape(
        value.shape[0],
        stageb.FUTURE_STEPS,
        stageb.BEADS,
        2,
    )


def center_points(
    value: np.ndarray,
) -> np.ndarray:
    points = np.asarray(
        value,
        dtype=np.float64,
    )
    return (
        points
        - np.mean(
            points,
            axis=-2,
            keepdims=True,
        )
    )


def segment_vectors(
    value: np.ndarray,
) -> np.ndarray:
    points = np.asarray(
        value,
        dtype=np.float64,
    )
    return (
        points[..., 1:, :]
        - points[..., :-1, :]
    )


def relative_robot_features(
    *,
    cable: np.ndarray,
    robot: np.ndarray,
) -> np.ndarray:
    cable_points = np.asarray(
        cable,
        dtype=np.float64,
    )
    robot_value = np.asarray(
        robot,
        dtype=np.float64,
    )
    expected_robot = (
        cable_points.shape[0],
        stageb.HISTORY_STEPS,
        schema_v3.ROBOT_PROXY_DIM,
    )
    if robot_value.shape != expected_robot:
        raise ValueError(
            "robot-proxy shape changed: {}".format(robot_value.shape)
        )
    centroid = np.mean(
        cable_points,
        axis=2,
    )
    joint_position = robot_value[
        :,
        :,
        schema_v3.ROBOT_JOINT_POSITION_SLICE,
    ]
    joint_velocity = robot_value[
        :,
        :,
        schema_v3.ROBOT_JOINT_VELOCITY_SLICE,
    ]
    ee_position = robot_value[
        :,
        :,
        schema_v3.ROBOT_EE_POSITION_SLICE,
    ]
    quaternion = robot_value[
        :,
        :,
        schema_v3.ROBOT_EE_QUATERNION_SLICE,
    ]
    ee_relative_xy = (
        ee_position[:, :, :2]
        - centroid
    )
    ee_z = ee_position[:, :, 2:3]
    result = np.concatenate(
        [
            joint_position,
            joint_velocity,
            ee_relative_xy,
            ee_z,
            quaternion,
        ],
        axis=2,
    )
    if result.shape != expected_robot:
        raise DirectionSurrogateError(
            "relative robot feature layout changed"
        )
    return result.astype(np.float64)


def build_surrogate_features(
    *,
    condition: np.ndarray,
    control: np.ndarray,
    condition_name: Sequence[Any],
    feature_mode: str,
) -> np.ndarray:
    cable, robot, actions = (
        _condition_parts(condition)
    )
    future = _control_points(control)
    if feature_mode == "none":
        return np.zeros(
            (condition.shape[0], 0),
            dtype=np.float64,
        )
    if feature_mode == "condition_only":
        return condition_one_hot(
            condition_name
        )
    if feature_mode == "centered_anchor":
        history_part = center_points(cable)[
            :,
            :,
            CABLE_ANCHOR_INDICES,
            :,
        ].reshape(condition.shape[0], -1)
        future_part = center_points(future)[
            :,
            :,
            CABLE_ANCHOR_INDICES,
            :,
        ].reshape(condition.shape[0], -1)
        robot_part = relative_robot_features(
            cable=cable,
            robot=robot,
        )
    elif feature_mode == "segment_anchor":
        history_part = segment_vectors(cable)[
            :,
            :,
            SEGMENT_ANCHOR_INDICES,
            :,
        ].reshape(condition.shape[0], -1)
        future_part = segment_vectors(future)[
            :,
            :,
            SEGMENT_ANCHOR_INDICES,
            :,
        ].reshape(condition.shape[0], -1)
        robot_part = relative_robot_features(
            cable=cable,
            robot=robot,
        )
    elif feature_mode == "absolute_anchor":
        history_part = cable[
            :,
            :,
            CABLE_ANCHOR_INDICES,
            :,
        ].reshape(condition.shape[0], -1)
        future_part = future[
            :,
            :,
            CABLE_ANCHOR_INDICES,
            :,
        ].reshape(condition.shape[0], -1)
        robot_part = robot
    else:
        raise ValueError(
            "unknown feature mode: {}".format(feature_mode)
        )
    result = np.concatenate(
        [
            history_part,
            robot_part.reshape(
                condition.shape[0],
                -1,
            ),
            actions,
            future_part,
        ],
        axis=1,
    ).astype(np.float64)
    if result.shape[1] != 211:
        raise DirectionSurrogateError(
            "surrogate feature dimension changed: {}".format(result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise DirectionSurrogateError(
            "surrogate features contain NaN or Inf"
        )
    return result


def translate_cable_inputs(
    *,
    condition: np.ndarray,
    control: np.ndarray,
    offset_xy: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    translated_condition = np.asarray(
        condition,
        dtype=np.float64,
    ).copy()
    translated_control = np.asarray(
        control,
        dtype=np.float64,
    ).copy()
    offset = np.asarray(
        offset_xy,
        dtype=np.float64,
    )
    if offset.shape != (2,):
        raise ValueError("translation offset shape changed")
    history = translated_condition[
        :,
        : stageb.HISTORY_STEPS * stageb.STATE_DIM,
    ].reshape(
        translated_condition.shape[0],
        stageb.HISTORY_STEPS,
        stageb.STATE_DIM,
    )
    cable = history[
        :,
        :,
        : stageb.CABLE_DIM,
    ].reshape(
        translated_condition.shape[0],
        stageb.HISTORY_STEPS,
        stageb.BEADS,
        2,
    )
    cable += offset[None, None, None, :]
    robot = history[
        :,
        :,
        stageb.CABLE_DIM :,
    ]
    robot[
        :,
        :,
        schema_v3.ROBOT_EE_POSITION_SLICE,
    ][:, :, :2] += offset[None, None, :]
    translated_control.reshape(
        translated_control.shape[0],
        stageb.FUTURE_STEPS,
        stageb.BEADS,
        2,
    )[:] += offset[None, None, None, :]
    return (
        translated_condition.astype(np.float32),
        translated_control.astype(np.float32),
    )


def translation_invariance_audit(
    *,
    condition: np.ndarray,
    control: np.ndarray,
    condition_name: Sequence[Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, Any]:
    translated_condition, translated_control = (
        translate_cable_inputs(
            condition=condition,
            control=control,
            offset_xy=spec.translation_offset_xy,
        )
    )
    records = {}
    for mode in (
        "centered_anchor",
        "segment_anchor",
        "absolute_anchor",
    ):
        original = build_surrogate_features(
            condition=condition,
            control=control,
            condition_name=condition_name,
            feature_mode=mode,
        )
        translated = build_surrogate_features(
            condition=translated_condition,
            control=translated_control,
            condition_name=condition_name,
            feature_mode=mode,
        )
        difference = np.abs(
            translated - original
        )
        records[mode] = {
            "maximum_absolute_difference":
                float(np.max(difference)),
            "mean_absolute_difference":
                float(np.mean(difference)),
            "original_sha256":
                sha256_array(original),
            "translated_sha256":
                sha256_array(translated),
        }
    if records["centered_anchor"][
        "maximum_absolute_difference"
    ] > spec.translation_tolerance:
        raise DirectionSurrogateError(
            "centered-anchor features are not translation invariant"
        )
    if records["segment_anchor"][
        "maximum_absolute_difference"
    ] > spec.translation_tolerance:
        raise DirectionSurrogateError(
            "segment-anchor features are not translation invariant"
        )
    if records["absolute_anchor"][
        "maximum_absolute_difference"
    ] <= spec.translation_tolerance:
        raise DirectionSurrogateError(
            "absolute-coordinate control did not respond to translation"
        )
    return {
        "offset_xy":
            list(spec.translation_offset_xy),
        "tolerance":
            float(spec.translation_tolerance),
        "records":
            records,
        "selectable_modes_invariant":
            True,
        "absolute_control_sensitive":
            True,
    }


def fit_feature_standardizer(
    value: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, np.ndarray]:
    array = np.asarray(
        value,
        dtype=np.float64,
    )
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(
            "feature standardizer requires at least two rows"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "feature standardizer input is non-finite"
        )
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    active = std > float(epsilon)
    scale = np.where(
        active,
        std,
        1.0,
    )
    return {
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "active": active.astype(np.bool_),
    }


def apply_feature_standardizer(
    value: np.ndarray,
    standardizer: Mapping[str, np.ndarray],
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )
    mean = np.asarray(
        standardizer["mean"],
        dtype=np.float64,
    )
    scale = np.asarray(
        standardizer["scale"],
        dtype=np.float64,
    )
    if array.shape[1:] != mean.shape:
        raise ValueError(
            "feature standardizer shape mismatch"
        )
    result = (
        array - mean[None]
    ) / scale[None]
    if not np.all(np.isfinite(result)):
        raise DirectionSurrogateError(
            "standardized features contain NaN or Inf"
        )
    return result.astype(np.float64)


def _ridge_solve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    jitter: float,
) -> np.ndarray:
    design = np.asarray(
        x,
        dtype=np.float64,
    )
    target = np.asarray(
        y,
        dtype=np.float64,
    )
    if design.ndim != 2 or target.ndim != 2:
        raise ValueError("ridge input rank changed")
    if design.shape[0] != target.shape[0]:
        raise ValueError("ridge row count mismatch")
    gram = design.T @ design
    rhs = design.T @ target
    regularized = gram + (
        float(alpha) + float(jitter)
    ) * np.eye(
        gram.shape[0],
        dtype=np.float64,
    )
    coefficient = np.linalg.solve(
        regularized,
        rhs,
    )
    if not np.all(np.isfinite(coefficient)):
        raise DirectionSurrogateError(
            "ridge coefficient contains NaN or Inf"
        )
    return coefficient.astype(np.float64)


def fit_surrogate_model(
    *,
    definition: SurrogateDefinition,
    features: np.ndarray,
    target_residual: np.ndarray,
    condition_name: Sequence[Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, Any]:
    definition.validate()
    x = np.asarray(
        features,
        dtype=np.float64,
    )
    y = np.asarray(
        target_residual,
        dtype=np.float64,
    ).reshape(
        target_residual.shape[0],
        -1,
    )
    names = np.asarray(
        condition_name
    ).astype(str)
    if x.shape[0] != y.shape[0] or names.shape[0] != y.shape[0]:
        raise ValueError(
            "surrogate fit row count mismatch"
        )
    if not np.all(np.isfinite(y)):
        raise DirectionSurrogateError(
            "surrogate target contains NaN or Inf"
        )
    global_mean = np.mean(
        y,
        axis=0,
    ).astype(np.float64)

    if definition.model_mode == "global_mean":
        model = {
            "model_mode": "global_mean",
            "global_mean": global_mean,
        }
    elif definition.model_mode == "condition_mean":
        condition_means = {}
        for name in CONDITION_VALUES:
            mask = names == name
            if not np.any(mask):
                raise DirectionSurrogateError(
                    "condition mean has an empty class"
                )
            condition_means[name] = np.mean(
                y[mask],
                axis=0,
            ).astype(np.float64)
        model = {
            "model_mode": "condition_mean",
            "global_mean": global_mean,
            "condition_means": condition_means,
        }
    else:
        standardizer = fit_feature_standardizer(
            x,
            epsilon=spec.standardizer_epsilon,
        )
        xz = apply_feature_standardizer(
            x,
            standardizer,
        )
        y_centered = y - global_mean[None]
        if definition.model_mode == "ridge":
            coefficient = _ridge_solve(
                xz,
                y_centered,
                alpha=definition.ridge_alpha,
                jitter=spec.ridge_jitter,
            )
            model = {
                "model_mode": "ridge",
                "feature_standardizer":
                    standardizer,
                "global_mean": global_mean,
                "coefficient": coefficient,
            }
        elif definition.model_mode == "reduced_rank_ridge":
            _, singular, vt = np.linalg.svd(
                y_centered,
                full_matrices=False,
            )
            rank = min(
                int(definition.output_rank),
                int(vt.shape[0]),
            )
            basis = vt[:rank].astype(
                np.float64
            )
            scores = y_centered @ basis.T
            coefficient = _ridge_solve(
                xz,
                scores,
                alpha=definition.ridge_alpha,
                jitter=spec.ridge_jitter,
            )
            model = {
                "model_mode":
                    "reduced_rank_ridge",
                "feature_standardizer":
                    standardizer,
                "global_mean":
                    global_mean,
                "basis":
                    basis,
                "coefficient":
                    coefficient,
                "singular_values":
                    singular.astype(
                        np.float64
                    ),
            }
        else:
            raise ValueError(
                "unknown model mode"
            )

    identity_payload = {
        "definition": asdict(definition),
        "global_mean_sha256":
            sha256_array(global_mean),
    }
    if "feature_standardizer" in model:
        identity_payload[
            "feature_mean_sha256"
        ] = sha256_array(
            model[
                "feature_standardizer"
            ]["mean"]
        )
        identity_payload[
            "feature_scale_sha256"
        ] = sha256_array(
            model[
                "feature_standardizer"
            ]["scale"]
        )
        identity_payload[
            "feature_active_sha256"
        ] = sha256_array(
            model[
                "feature_standardizer"
            ]["active"]
        )
    if "condition_means" in model:
        identity_payload[
            "condition_mean_sha256"
        ] = {
            name: sha256_array(
                model[
                    "condition_means"
                ][name]
            )
            for name in CONDITION_VALUES
        }
    if "coefficient" in model:
        identity_payload[
            "coefficient_sha256"
        ] = sha256_array(
            model["coefficient"]
        )
        identity_payload[
            "coefficient_norm"
        ] = float(
            np.linalg.norm(
                model["coefficient"]
            )
        )
    if "basis" in model:
        identity_payload[
            "basis_sha256"
        ] = sha256_array(
            model["basis"]
        )
        identity_payload[
            "basis_orthogonality_error"
        ] = float(
            np.max(
                np.abs(
                    model["basis"]
                    @ model["basis"].T
                    - np.eye(
                        model["basis"].shape[0],
                        dtype=np.float64,
                    )
                )
            )
        )
        identity_payload[
            "singular_values_sha256"
        ] = sha256_array(
            model["singular_values"]
        )
    identity_payload[
        "model_sha256"
    ] = sha256_bytes(
        stable_json_bytes(
            identity_payload
        )
    )
    model["identity"] = identity_payload
    return model


def predict_surrogate_model(
    *,
    definition: SurrogateDefinition,
    model: Mapping[str, Any],
    features: np.ndarray,
    condition_name: Sequence[Any],
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    definition.validate()
    x = np.asarray(
        features,
        dtype=np.float64,
    )
    names = np.asarray(
        condition_name
    ).astype(str)
    rows = x.shape[0]
    mode = model["model_mode"]
    if mode == "global_mean":
        flat = np.repeat(
            np.asarray(
                model["global_mean"],
                dtype=np.float64,
            )[None],
            rows,
            axis=0,
        )
    elif mode == "condition_mean":
        flat = np.empty(
            (
                rows,
                np.asarray(
                    model["global_mean"]
                ).shape[0],
            ),
            dtype=np.float64,
        )
        for index, name in enumerate(names):
            value = model[
                "condition_means"
            ].get(name)
            if value is None:
                value = model[
                    "global_mean"
                ]
            flat[index] = value
    else:
        xz = apply_feature_standardizer(
            x,
            model[
                "feature_standardizer"
            ],
        )
        predicted = xz @ np.asarray(
            model["coefficient"],
            dtype=np.float64,
        )
        if mode == "reduced_rank_ridge":
            predicted = predicted @ np.asarray(
                model["basis"],
                dtype=np.float64,
            )
        flat = (
            np.asarray(
                model["global_mean"],
                dtype=np.float64,
            )[None]
            + predicted
        )
    result = flat.reshape(
        (rows,) + tuple(target_shape)
    )
    if not np.all(np.isfinite(result)):
        raise DirectionSurrogateError(
            "surrogate prediction contains NaN or Inf"
        )
    return result.astype(np.float64)


def active_residual(
    *,
    control: np.ndarray,
    target: np.ndarray,
    target_standardizer: stageb.ArrayStandardizer,
) -> np.ndarray:
    """Return raw ordered-cable XY residual without fold-target normalization.

    The frozen target standardizer is accepted only to preserve the surrounding
    call contract.  Its mean, scale, and active mask are not used.  Every raw XY
    output dimension remains in the surrogate target, preventing grouped OOF
    folds from inheriting any target statistic fitted on the complete
    objective-training population.
    """
    control_raw = np.asarray(
        control,
        dtype=np.float64,
    )
    target_raw = np.asarray(
        target,
        dtype=np.float64,
    )
    if control_raw.shape != target_raw.shape:
        raise ValueError(
            "raw residual shape mismatch"
        )
    _ = target_standardizer
    residual = target_raw - control_raw
    return residual.astype(np.float64)


def row_direction_metrics(
    *,
    predicted_residual: np.ndarray,
    target_residual: np.ndarray,
    condition_name: Sequence[Any],
    fold_assignment: Optional[np.ndarray],
    epsilon: float,
) -> Dict[str, Any]:
    predicted = np.asarray(
        predicted_residual,
        dtype=np.float64,
    ).reshape(
        predicted_residual.shape[0],
        -1,
    )
    target = np.asarray(
        target_residual,
        dtype=np.float64,
    ).reshape(
        target_residual.shape[0],
        -1,
    )
    cosine = stageb258._row_cosine(
        predicted,
        target,
        epsilon=epsilon,
    )
    predicted_norm = np.sqrt(
        np.sum(
            predicted * predicted,
            axis=1,
        )
    )
    target_norm = np.sqrt(
        np.sum(
            target * target,
            axis=1,
        )
    )
    names = np.asarray(
        condition_name
    ).astype(str)
    by_condition = {
        name: {
            "rows": int(
                np.sum(names == name)
            ),
            "cosine":
                _safe_stats(
                    cosine[names == name]
                ),
            "nonnegative_rate":
                float(
                    np.mean(
                        cosine[
                            names == name
                        ]
                        >= 0.0
                    )
                ),
        }
        for name in CONDITION_VALUES
    }
    by_fold = {}
    if fold_assignment is not None:
        folds = np.asarray(
            fold_assignment,
            dtype=np.int64,
        )
        for fold in sorted(
            set(folds.tolist())
        ):
            mask = folds == fold
            by_fold[str(fold)] = {
                "rows": int(
                    np.sum(mask)
                ),
                "cosine":
                    _safe_stats(
                        cosine[mask]
                    ),
                "nonnegative_rate":
                    float(
                        np.mean(
                            cosine[mask]
                            >= 0.0
                        )
                    ),
            }
    return {
        "cosine":
            _safe_stats(cosine),
        "nonnegative_rate":
            float(
                np.mean(
                    cosine >= 0.0
                )
            ),
        "positive_quarter_rate":
            float(
                np.mean(
                    cosine >= 0.25
                )
            ),
        "predicted_norm":
            _safe_stats(predicted_norm),
        "target_norm":
            _safe_stats(target_norm),
        "zero_prediction_rate":
            float(
                np.mean(
                    predicted_norm
                    <= epsilon
                )
            ),
        "by_condition":
            by_condition,
        "by_fold":
            by_fold,
        "cosine_sha256":
            sha256_array(
                cosine.astype(np.float64)
            ),
        "prediction_sha256":
            sha256_array(
                np.asarray(
                    predicted_residual,
                    dtype=np.float64,
                )
            ),
    }


def direction_gate(
    *,
    metrics: Mapping[str, Any],
    global_baseline_mean: float,
    condition_baseline_mean: float,
    spec: DirectionSurrogateSpec,
) -> Dict[str, bool]:
    condition_means = [
        float(
            metrics[
                "by_condition"
            ][name]["cosine"]["mean"]
        )
        for name in CONDITION_VALUES
    ]
    fold_means = [
        float(
            record["cosine"]["mean"]
        )
        for record in metrics[
            "by_fold"
        ].values()
    ]
    gates = {
        "mean_cosine": bool(
            metrics["cosine"]["mean"]
            >= spec.oof_mean_cosine_min
        ),
        "nonnegative_rate": bool(
            metrics["nonnegative_rate"]
            >= spec.oof_nonnegative_rate_min
        ),
        "condition_minimum": bool(
            min(condition_means)
            >= spec.oof_condition_cosine_min
        ),
        "fold_minimum": bool(
            min(fold_means)
            >= spec.oof_fold_cosine_min
        ),
        "global_margin": bool(
            metrics["cosine"]["mean"]
            - float(global_baseline_mean)
            >= spec.global_baseline_margin_min
        ),
        "condition_margin": bool(
            metrics["cosine"]["mean"]
            - float(condition_baseline_mean)
            >= spec.condition_baseline_margin_min
        ),
    }
    gates["all"] = bool(
        all(gates.values())
    )
    return gates


def holdout_direction_gate(
    *,
    metrics: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, bool]:
    condition_means = [
        float(
            metrics[
                "by_condition"
            ][name]["cosine"]["mean"]
        )
        for name in CONDITION_VALUES
    ]
    gates = {
        "mean_cosine": bool(
            metrics["cosine"]["mean"]
            >= spec.holdout_mean_cosine_min
        ),
        "nonnegative_rate": bool(
            metrics["nonnegative_rate"]
            >= spec.holdout_nonnegative_rate_min
        ),
        "condition_minimum": bool(
            min(condition_means)
            >= spec.holdout_condition_cosine_min
        ),
    }
    gates["all"] = bool(
        all(gates.values())
    )
    return gates


def build_direction_context(
    *,
    root: Path,
    captured: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, Any]:
    context = stageb258.build_audit_context(
        root=Path(root).resolve(),
        captured=captured,
    )
    objective_mask = np.asarray(
        context["objective_train_mask"],
        dtype=np.bool_,
    )
    objective_condition = np.asarray(
        context["condition"][objective_mask],
        dtype=np.float32,
    )
    objective_target = np.asarray(
        context["target"][objective_mask],
        dtype=np.float32,
    )
    objective_groups = np.asarray(
        context["groups"][objective_mask]
    ).astype(str)
    objective_condition_name = np.asarray(
        context["condition_name"][
            objective_mask
        ]
    ).astype(str)
    predictions, prediction_sha = (
        stageb258.stagec.one_step_predictions(
            model=context["model"],
            condition=objective_condition,
            target=objective_target,
            spec=context["stageb_spec"],
            condition_standardizer=
                context[
                    "condition_standardizer"
                ],
            target_standardizer=
                context[
                    "target_standardizer"
                ],
            noise_seed=(
                context[
                    "stageb_spec"
                ].seed
                + int(
                    spec.objective_train_noise_seed_offset
                )
            ),
        )
    )
    assignment, mapping = (
        deterministic_group_folds(
            objective_groups,
            folds=spec.grouped_cv_folds,
        )
    )
    if len(set(objective_groups.tolist())) != 126:
        raise DirectionSurrogateError(
            "objective-train group population changed"
        )
    if objective_target.shape[0] != 638:
        raise DirectionSurrogateError(
            "objective-train row population changed"
        )
    if np.any(
        context["frozen_probe_mask"]
        & context["objective_train_mask"]
    ):
        raise DirectionSurrogateError(
            "objective train crossed frozen probe"
        )
    return {
        **context,
        "objective_condition":
            objective_condition,
        "objective_target":
            objective_target,
        "objective_groups":
            objective_groups,
        "objective_condition_name":
            objective_condition_name,
        "objective_control_predictions":
            predictions,
        "objective_control_prediction_sha256":
            prediction_sha,
        "objective_fold_assignment":
            assignment,
        "objective_fold_mapping":
            mapping,
    }


def fit_oof_predictions(
    *,
    definition: SurrogateDefinition,
    features: np.ndarray,
    target_residual: np.ndarray,
    condition_name: Sequence[Any],
    fold_assignment: np.ndarray,
    spec: DirectionSurrogateSpec,
    permutation_seed: Optional[int] = None,
) -> Dict[str, Any]:
    definition.validate()
    rows = target_residual.shape[0]
    predictions = np.empty_like(
        np.asarray(
            target_residual,
            dtype=np.float64,
        )
    )
    assigned = np.zeros(
        rows,
        dtype=np.bool_,
    )
    fold_records = []
    for fold in range(
        spec.grouped_cv_folds
    ):
        test_mask = (
            np.asarray(
                fold_assignment,
                dtype=np.int64,
            )
            == fold
        )
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise DirectionSurrogateError(
                "OOF fold is empty"
            )
        train_target = np.asarray(
            target_residual[train_mask],
            dtype=np.float64,
        )
        permutation_sha = None
        if permutation_seed is not None:
            rng = np.random.RandomState(
                int(permutation_seed)
                + int(fold) * 1009
            )
            permutation = rng.permutation(
                train_target.shape[0]
            )
            train_target = train_target[
                permutation
            ]
            permutation_sha = sha256_array(
                permutation.astype(
                    np.int64
                )
            )
        model = fit_surrogate_model(
            definition=definition,
            features=np.asarray(
                features[train_mask],
                dtype=np.float64,
            ),
            target_residual=train_target,
            condition_name=np.asarray(
                condition_name
            )[train_mask],
            spec=spec,
        )
        fold_prediction = (
            predict_surrogate_model(
                definition=definition,
                model=model,
                features=np.asarray(
                    features[test_mask],
                    dtype=np.float64,
                ),
                condition_name=np.asarray(
                    condition_name
                )[test_mask],
                target_shape=tuple(
                    target_residual.shape[1:]
                ),
            )
        )
        predictions[test_mask] = (
            fold_prediction
        )
        assigned[test_mask] = True
        fold_records.append(
            {
                "fold": int(fold),
                "train_rows":
                    int(np.sum(train_mask)),
                "test_rows":
                    int(np.sum(test_mask)),
                "model_identity":
                    copy.deepcopy(
                        model["identity"]
                    ),
                "prediction_sha256":
                    sha256_array(
                        fold_prediction
                    ),
                "permutation_sha256":
                    permutation_sha,
            }
        )
    if not np.all(assigned):
        raise DirectionSurrogateError(
            "OOF prediction population is incomplete"
        )
    return {
        "prediction":
            predictions,
        "prediction_sha256":
            sha256_array(predictions),
        "fold_records":
            fold_records,
        "permuted_training_targets":
            bool(
                permutation_seed is not None
            ),
    }


def state_candidate_from_residual(
    *,
    control: np.ndarray,
    predicted_residual: np.ndarray,
    scale: float,
    target_standardizer: stageb.ArrayStandardizer,
) -> np.ndarray:
    control_raw = np.asarray(
        control,
        dtype=np.float64,
    )
    residual = np.asarray(
        predicted_residual,
        dtype=np.float64,
    ).copy()
    if residual.shape != control_raw.shape:
        raise ValueError(
            "predicted raw residual shape changed"
        )
    _ = target_standardizer
    candidate = (
        control_raw
        + float(scale) * residual
    ).astype(np.float32)
    if not np.all(np.isfinite(candidate)):
        raise DirectionSurrogateError(
            "state candidate contains NaN or Inf"
        )
    return candidate


def state_scientific_gates(
    *,
    timestep: int,
    evaluation: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, bool]:
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
            <= spec.state_fidelity_ratio_max
        ),
        "lower": bool(
            evaluation[
                "lower_row_pass_rate"
            ]
            >= spec.state_lower_pass_min
        ),
        "physical": bool(
            evaluation[
                "historical_physical_row_any_rate"
            ]
            >= spec.state_physical_valid_min
        ),
        "collapse": bool(
            evaluation[
                "segment_length"
            ]["collapse_fraction"]
            <= spec.state_collapse_fraction_max
        ),
        "stretch": bool(
            evaluation[
                "segment_length"
            ]["stretch_fraction"]
            <= spec.state_stretch_fraction_max
        ),
        "movement_alignment": bool(
            evaluation[
                "target_direction_cosine"
            ]["mean"]
            >= spec.state_movement_cosine_min
        ),
        "target_distance_reduction": bool(
            evaluation[
                "target_distance_reduction_fraction"
            ]["mean"]
            >= spec.state_target_distance_reduction_min
        ),
    }
    gates["all"] = bool(
        all(gates.values())
    )
    return gates


def evaluate_scale_grid(
    *,
    predicted_residual: np.ndarray,
    control: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    timestep: int,
    context: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, Any]:
    records = []
    for scale in spec.scale_grid:
        candidate = state_candidate_from_residual(
            control=control,
            predicted_residual=
                predicted_residual,
            scale=float(scale),
            target_standardizer=
                context[
                    "target_standardizer"
                ],
        )
        evaluation = (
            stageb258
            .evaluate_output_population(
                candidate,
                control=control,
                target=target,
                groups=groups,
                condition_name=
                    condition_name,
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
                top_k=16,
                epsilon=
                    spec.standardizer_epsilon,
            )
        )
        gates = state_scientific_gates(
            timestep=int(timestep),
            evaluation=evaluation,
            spec=spec,
        )
        records.append(
            {
                "scale":
                    float(scale),
                "candidate_sha256":
                    sha256_array(
                        candidate
                    ),
                "evaluation":
                    evaluation,
                "gates":
                    gates,
            }
        )
    witnesses = [
        record
        for record in records
        if record["gates"]["all"]
    ]
    witness = (
        min(
            witnesses,
            key=lambda record: (
                float(
                    record[
                        "evaluation"
                    ][
                        "normalized_mse_ratio"
                    ]
                ),
                -float(
                    record[
                        "evaluation"
                    ][
                        "historical_physical_row_any_rate"
                    ]
                ),
                float(
                    record[
                        "evaluation"
                    ][
                        "segment_length"
                    ]["collapse_fraction"]
                ),
                float(record["scale"]),
            ),
        )
        if witnesses
        else None
    )
    best_direction = max(
        records,
        key=lambda record: (
            float(
                record[
                    "evaluation"
                ][
                    "target_direction_cosine"
                ]["mean"]
            ),
            float(
                record[
                    "evaluation"
                ][
                    "target_distance_reduction_fraction"
                ]["mean"]
            ),
            -float(
                record[
                    "evaluation"
                ][
                    "normalized_mse_ratio"
                ]
            ),
        ),
    )
    best_physical = max(
        records,
        key=lambda record: (
            float(
                record[
                    "evaluation"
                ][
                    "historical_physical_row_any_rate"
                ]
            ),
            float(
                record[
                    "evaluation"
                ][
                    "lower_row_pass_rate"
                ]
            ),
            -float(
                record[
                    "evaluation"
                ][
                    "segment_length"
                ]["collapse_fraction"]
            ),
        ),
    )
    return {
        "timestep":
            int(timestep),
        "records":
            records,
        "scientific_witness":
            witness,
        "scientific_reachable":
            bool(witness is not None),
        "best_direction_record":
            best_direction,
        "best_physical_record":
            best_physical,
    }


def candidate_oof_record(
    *,
    definition: SurrogateDefinition,
    context: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
    baseline_means: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    definition.validate()
    timestep_records = {}
    for timestep in spec.timesteps:
        control = np.asarray(
            context[
                "objective_control_predictions"
            ][int(timestep)],
            dtype=np.float32,
        )
        target = np.asarray(
            context[
                "objective_target"
            ],
            dtype=np.float32,
        )
        residual = active_residual(
            control=control,
            target=target,
            target_standardizer=
                context[
                    "target_standardizer"
                ],
        )
        features = build_surrogate_features(
            condition=
                context[
                    "objective_condition"
                ],
            control=control,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            feature_mode=
                definition.feature_mode,
        )
        oof = fit_oof_predictions(
            definition=definition,
            features=features,
            target_residual=residual,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            fold_assignment=
                context[
                    "objective_fold_assignment"
                ],
            spec=spec,
        )
        metrics = row_direction_metrics(
            predicted_residual=
                oof["prediction"],
            target_residual=residual,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            fold_assignment=
                context[
                    "objective_fold_assignment"
                ],
            epsilon=
                spec.standardizer_epsilon,
        )
        global_mean = float(
            baseline_means.get(
                "global_mean",
                {},
            ).get(
                str(timestep),
                metrics["cosine"]["mean"],
            )
        )
        condition_mean = float(
            baseline_means.get(
                "condition_mean",
                {},
            ).get(
                str(timestep),
                metrics["cosine"]["mean"],
            )
        )
        gates = (
            direction_gate(
                metrics=metrics,
                global_baseline_mean=
                    global_mean,
                condition_baseline_mean=
                    condition_mean,
                spec=spec,
            )
            if definition.role
            not in (
                "negative_control",
            )
            else {
                "all": False,
                "baseline": True,
            }
        )
        scale_sweep = evaluate_scale_grid(
            predicted_residual=
                oof["prediction"],
            control=control,
            target=target,
            groups=
                context[
                    "objective_groups"
                ],
            condition_name=
                context[
                    "objective_condition_name"
                ],
            timestep=int(timestep),
            context=context,
            spec=spec,
        )
        timestep_records[
            str(timestep)
        ] = {
            "timestep":
                int(timestep),
            "feature_sha256":
                sha256_array(features),
            "target_residual_sha256":
                sha256_array(residual),
            "oof":
                {
                    key: value
                    for key, value
                    in oof.items()
                    if key != "prediction"
                },
            "direction_metrics":
                metrics,
            "direction_gates":
                gates,
            "scale_sweep":
                scale_sweep,
        }
    eligible = bool(
        definition.role == "selectable"
        and all(
            record[
                "direction_gates"
            ]["all"]
            and record[
                "scale_sweep"
            ]["scientific_reachable"]
            for record
            in timestep_records.values()
        )
    )
    return {
        "candidate_id":
            definition.candidate_id,
        "definition":
            asdict(definition),
        "timestep_records":
            timestep_records,
        "role":
            definition.role,
        "eligible":
            eligible,
        "holdout_used":
            False,
    }


def collect_baseline_means(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    output = {}
    for candidate_id in (
        "global_mean",
        "condition_mean",
    ):
        record = next(
            (
                item
                for item in records
                if item["candidate_id"]
                == candidate_id
            ),
            None,
        )
        if record is None:
            raise DirectionSurrogateError(
                "baseline candidate is missing"
            )
        output[candidate_id] = {
            timestep: float(
                value[
                    "direction_metrics"
                ]["cosine"]["mean"]
            )
            for timestep, value
            in record[
                "timestep_records"
            ].items()
        }
    return output


def _selection_key(
    record: Mapping[str, Any],
) -> Tuple[Any, ...]:
    timestep_records = record[
        "timestep_records"
    ]
    direction = [
        float(
            timestep_records[
                str(timestep)
            ][
                "direction_metrics"
            ]["cosine"]["mean"]
        )
        for timestep in (10, 25, 50)
    ]
    witnesses = [
        timestep_records[
            str(timestep)
        ][
            "scale_sweep"
        ][
            "scientific_witness"
        ]
        for timestep in (10, 25, 50)
    ]
    if any(
        witness is None
        for witness in witnesses
    ):
        raise DirectionSurrogateError(
            "eligible candidate lacks a scientific witness"
        )
    physical = [
        float(
            witness[
                "evaluation"
            ][
                "historical_physical_row_any_rate"
            ]
        )
        for witness in witnesses
    ]
    nmse = [
        float(
            witness[
                "evaluation"
            ][
                "normalized_mse_ratio"
            ]
        )
        for witness in witnesses
    ]
    collapse = [
        float(
            witness[
                "evaluation"
            ][
                "segment_length"
            ]["collapse_fraction"]
        )
        for witness in witnesses
    ]
    definition = record[
        "definition"
    ]
    complexity = (
        0
        if definition[
            "model_mode"
        ] == "reduced_rank_ridge"
        else 1
    )
    return (
        -min(direction),
        -min(physical),
        max(nmse),
        max(collapse),
        complexity,
        int(
            definition[
                "output_rank"
            ]
        ),
        float(
            definition[
                "ridge_alpha"
            ]
        ),
        str(
            record[
                "candidate_id"
            ]
        ),
    )


def select_oof_candidate(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    eligible = [
        record
        for record in records
        if record.get("eligible")
        and record.get("role")
        == "selectable"
    ]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=_selection_key,
    )
    locked_scales = {
        timestep: float(
            record[
                "scale_sweep"
            ][
                "scientific_witness"
            ]["scale"]
        )
        for timestep, record
        in selected[
            "timestep_records"
        ].items()
    }
    return {
        "candidate_id":
            selected[
                "candidate_id"
            ],
        "definition":
            copy.deepcopy(
                selected[
                    "definition"
                ]
            ),
        "locked_scales":
            locked_scales,
        "selection_key":
            list(
                _selection_key(
                    selected
                )
            ),
        "selection_source":
            "objective_train_grouped_oof_only",
        "selection_holdout_used":
            False,
    }


def permutation_control(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, Any]:
    definition = SurrogateDefinition(
        **selected["definition"]
    )
    records = {}
    pass_by_timestep = []
    for timestep in spec.timesteps:
        control = np.asarray(
            context[
                "objective_control_predictions"
            ][int(timestep)],
            dtype=np.float32,
        )
        target = np.asarray(
            context[
                "objective_target"
            ],
            dtype=np.float32,
        )
        residual = active_residual(
            control=control,
            target=target,
            target_standardizer=
                context[
                    "target_standardizer"
                ],
        )
        features = build_surrogate_features(
            condition=
                context[
                    "objective_condition"
                ],
            control=control,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            feature_mode=
                definition.feature_mode,
        )
        oof = fit_oof_predictions(
            definition=definition,
            features=features,
            target_residual=residual,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            fold_assignment=
                context[
                    "objective_fold_assignment"
                ],
            spec=spec,
            permutation_seed=(
                spec.permutation_seed
                + int(timestep) * 10000
            ),
        )
        metrics = row_direction_metrics(
            predicted_residual=
                oof["prediction"],
            target_residual=residual,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            fold_assignment=
                context[
                    "objective_fold_assignment"
                ],
            epsilon=
                spec.standardizer_epsilon,
        )
        selected_record = next(
            record
            for record
            in context[
                "candidate_records"
            ]
            if record[
                "candidate_id"
            ]
            == selected[
                "candidate_id"
            ]
        )
        selected_mean = float(
            selected_record[
                "timestep_records"
            ][str(timestep)][
                "direction_metrics"
            ]["cosine"]["mean"]
        )
        permutation_mean = float(
            metrics[
                "cosine"
            ]["mean"]
        )
        passed = bool(
            permutation_mean
            <= spec.permutation_cosine_max
            and selected_mean
            - permutation_mean
            >= spec.permutation_margin_min
        )
        pass_by_timestep.append(
            passed
        )
        records[str(timestep)] = {
            "timestep":
                int(timestep),
            "oof": {
                key: value
                for key, value
                in oof.items()
                if key != "prediction"
            },
            "direction_metrics":
                metrics,
            "selected_mean_cosine":
                selected_mean,
            "permutation_mean_cosine":
                permutation_mean,
            "margin":
                float(
                    selected_mean
                    - permutation_mean
                ),
            "pass":
                passed,
        }
    return {
        "candidate_id":
            selected[
                "candidate_id"
            ],
        "records":
            records,
        "all_pass":
            bool(
                all(pass_by_timestep)
            ),
        "holdout_used":
            False,
    }


def fit_selected_and_evaluate_holdout(
    *,
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    spec: DirectionSurrogateSpec,
) -> Dict[str, Any]:
    definition = SurrogateDefinition(
        **selected["definition"]
    )
    timestep_records = {}
    all_direction = []
    all_state = []
    for timestep in spec.timesteps:
        objective_control = np.asarray(
            context[
                "objective_control_predictions"
            ][int(timestep)],
            dtype=np.float32,
        )
        objective_target = np.asarray(
            context[
                "objective_target"
            ],
            dtype=np.float32,
        )
        objective_residual = (
            active_residual(
                control=objective_control,
                target=objective_target,
                target_standardizer=
                    context[
                        "target_standardizer"
                    ],
            )
        )
        train_features = (
            build_surrogate_features(
                condition=
                    context[
                        "objective_condition"
                    ],
                control=
                    objective_control,
                condition_name=
                    context[
                        "objective_condition_name"
                    ],
                feature_mode=
                    definition.feature_mode,
            )
        )
        model = fit_surrogate_model(
            definition=definition,
            features=train_features,
            target_residual=
                objective_residual,
            condition_name=
                context[
                    "objective_condition_name"
                ],
            spec=spec,
        )

        holdout_control = np.asarray(
            context[
                "control_predictions"
            ][int(timestep)],
            dtype=np.float32,
        )
        holdout_target = np.asarray(
            context[
                "holdout_target"
            ],
            dtype=np.float32,
        )
        holdout_residual = (
            active_residual(
                control=holdout_control,
                target=holdout_target,
                target_standardizer=
                    context[
                        "target_standardizer"
                    ],
            )
        )
        holdout_features = (
            build_surrogate_features(
                condition=
                    context[
                        "holdout_condition"
                    ],
                control=
                    holdout_control,
                condition_name=
                    context[
                        "holdout_condition_name"
                    ],
                feature_mode=
                    definition.feature_mode,
            )
        )
        prediction = predict_surrogate_model(
            definition=definition,
            model=model,
            features=holdout_features,
            condition_name=
                context[
                    "holdout_condition_name"
                ],
            target_shape=tuple(
                holdout_residual.shape[1:]
            ),
        )
        direction_metrics = (
            row_direction_metrics(
                predicted_residual=
                    prediction,
                target_residual=
                    holdout_residual,
                condition_name=
                    context[
                        "holdout_condition_name"
                    ],
                fold_assignment=None,
                epsilon=
                    spec.standardizer_epsilon,
            )
        )
        direction_gates = (
            holdout_direction_gate(
                metrics=
                    direction_metrics,
                spec=spec,
            )
        )
        scale = float(
            selected[
                "locked_scales"
            ][str(timestep)]
        )
        candidate = (
            state_candidate_from_residual(
                control=holdout_control,
                predicted_residual=
                    prediction,
                scale=scale,
                target_standardizer=
                    context[
                        "target_standardizer"
                    ],
            )
        )
        evaluation = (
            stageb258
            .evaluate_output_population(
                candidate,
                control=
                    holdout_control,
                target=
                    holdout_target,
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
                top_k=16,
                epsilon=
                    spec.standardizer_epsilon,
            )
        )
        state_gates = (
            state_scientific_gates(
                timestep=int(timestep),
                evaluation=evaluation,
                spec=spec,
            )
        )
        all_direction.append(
            direction_gates["all"]
        )
        all_state.append(
            state_gates["all"]
        )
        timestep_records[
            str(timestep)
        ] = {
            "timestep":
                int(timestep),
            "locked_scale":
                scale,
            "model_identity":
                copy.deepcopy(
                    model["identity"]
                ),
            "holdout_feature_sha256":
                sha256_array(
                    holdout_features
                ),
            "holdout_target_residual_sha256":
                sha256_array(
                    holdout_residual
                ),
            "prediction_sha256":
                sha256_array(
                    prediction
                ),
            "direction_metrics":
                direction_metrics,
            "direction_gates":
                direction_gates,
            "candidate_sha256":
                sha256_array(
                    candidate
                ),
            "state_evaluation":
                evaluation,
            "state_gates":
                state_gates,
        }
    return {
        "candidate_id":
            selected[
                "candidate_id"
            ],
        "definition":
            copy.deepcopy(
                selected[
                    "definition"
                ]
            ),
        "locked_scales":
            copy.deepcopy(
                selected[
                    "locked_scales"
                ]
            ),
        "timestep_records":
            timestep_records,
        "direction_all_timesteps":
            bool(all(all_direction)),
        "state_all_timesteps":
            bool(all(all_state)),
        "scientific_pass":
            bool(
                all(all_direction)
                and all(all_state)
            ),
        "selection_changed_after_holdout":
            False,
        "holdout_used_for_fit":
            False,
        "holdout_used_for_selection":
            False,
        "holdout_evaluated":
            True,
        "surrogate_weights_persisted":
            False,
        "prediction_tensor_persisted":
            False,
    }


def classify_direction_surrogate(
    *,
    candidate_records: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    permutation: Optional[Mapping[str, Any]],
    holdout: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    selectable = [
        record
        for record in candidate_records
        if record["role"] == "selectable"
    ]
    diagnostic = next(
        record
        for record in candidate_records
        if record["role"]
        == "diagnostic_control"
    )
    eligible = [
        record
        for record in selectable
        if record["eligible"]
    ]
    any_direction_all = [
        record
        for record in selectable
        if all(
            item[
                "direction_gates"
            ]["all"]
            for item
            in record[
                "timestep_records"
            ].values()
        )
    ]
    any_scale_all = [
        record
        for record in selectable
        if all(
            item[
                "scale_sweep"
            ]["scientific_reachable"]
            for item
            in record[
                "timestep_records"
            ].values()
        )
    ]
    diagnostic_direction_all = bool(
        all(
            item[
                "direction_gates"
            ]["all"]
            for item
            in diagnostic[
                "timestep_records"
            ].values()
        )
    )
    diagnostic_scale_all = bool(
        all(
            item[
                "scale_sweep"
            ]["scientific_reachable"]
            for item
            in diagnostic[
                "timestep_records"
            ].values()
        )
    )

    if selected is None:
        if (
            diagnostic_direction_all
            and diagnostic_scale_all
        ):
            root = (
                "phase314b_r258_staged_direction_signal_"
                "depends_on_absolute_position_features"
            )
            next_path = (
                "AUDIT_ABSOLUTE_POSITION_DEPENDENCE_WITH_"
                "TRANSLATION_EQUIVARIANT_SURROGATE"
            )
            locus = "absolute_position_dependence"
        elif any_direction_all and any_scale_all:
            root = (
                "phase314b_r258_staged_direction_and_state_"
                "reachability_do_not_coincide_in_one_surrogate"
            )
            next_path = (
                "CALIBRATE_DIRECTION_SAFETY_COMPOSITION_"
                "ON_OBJECTIVE_TRAIN_ONLY"
            )
            locus = "direction_state_candidate_mismatch"
        elif any_direction_all and not any_scale_all:
            root = (
                "phase314b_r258_staged_grouped_cv_direction_"
                "signal_does_not_form_valid_state_transition"
            )
            next_path = (
                "CALIBRATE_SURROGATE_GUIDED_CONSTRAINED_"
                "DIRECT_X0_INTEGRATOR"
            )
            locus = "direction_without_state_reachability"
        else:
            root = (
                "phase314b_r258_staged_objective_train_"
                "direction_surrogate_has_no_grouped_cv_signal"
            )
            next_path = (
                "AUDIT_CONDITION_REPRESENTATION_AND_TARGET_"
                "RESIDUAL_IDENTIFIABILITY"
            )
            locus = "no_grouped_cv_direction_signal"
    elif permutation is None or not permutation[
        "all_pass"
    ]:
        root = (
            "phase314b_r258_staged_direction_surrogate_"
            "permutation_control_indicates_leakage"
        )
        next_path = (
            "AUDIT_DIRECTION_SURROGATE_LEAKAGE_AND_GROUP_SPLIT"
        )
        locus = "permutation_leakage"
    elif holdout is None:
        raise DirectionSurrogateError(
            "selected surrogate lacks locked holdout evaluation"
        )
    elif not holdout[
        "direction_all_timesteps"
    ]:
        root = (
            "phase314b_r258_staged_objective_train_cv_"
            "direction_does_not_generalize_to_selection_holdout"
        )
        next_path = (
            "AUDIT_NONLINEAR_OR_SEQUENCE_DIRECTION_SURROGATE_"
            "ON_OBJECTIVE_TRAIN_ONLY"
        )
        locus = "holdout_direction_generalization"
    elif not holdout[
        "state_all_timesteps"
    ]:
        root = (
            "phase314b_r258_staged_direction_generalizes_"
            "but_needs_constrained_geometry_integrator"
        )
        next_path = (
            "CALIBRATE_SURROGATE_GUIDED_CONSTRAINED_"
            "DIRECT_X0_INTEGRATOR"
        )
        locus = "direction_without_holdout_state_reachability"
    else:
        root = (
            "phase314b_r258_staged_objective_train_direction_"
            "surrogate_selected_and_holdout_validated"
        )
        next_path = (
            "CALIBRATE_SELECTED_DIRECTION_SURROGATE_"
            "GUIDANCE_IN_TRAIN_ONLY_DENOISER"
        )
        locus = "direction_surrogate_validated"

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
            record[
                "candidate_id"
            ]
            for record in eligible
        ],
        "direction_all_candidate_ids": [
            record[
                "candidate_id"
            ]
            for record
            in any_direction_all
        ],
        "state_reachable_candidate_ids": [
            record[
                "candidate_id"
            ]
            for record
            in any_scale_all
        ],
        "absolute_diagnostic_direction_all":
            diagnostic_direction_all,
        "absolute_diagnostic_state_all":
            diagnostic_scale_all,
        "selected_configuration":
            copy.deepcopy(selected),
        "permutation_pass": (
            None
            if permutation is None
            else bool(
                permutation["all_pass"]
            )
        ),
        "holdout_scientific_pass": (
            None
            if holdout is None
            else bool(
                holdout[
                    "scientific_pass"
                ]
            )
        ),
    }


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[DirectionSurrogateSpec] = None,
) -> Dict[str, Any]:
    active_spec = (
        DirectionSurrogateSpec()
        if spec is None
        else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    base = validate_base_evidence(
        repository_root
    )
    source_binding = base[
        "source_binding"
    ]
    stagec258.stagea258.validate_environment_payload(
        environment
    )
    cold = (
        stagec258.stagea258
        .assert_cold_cuda_context_portable()
    )
    captured = (
        stageb258
        .capture_portable_control_model(
            root=repository_root
        )
    )
    context = build_direction_context(
        root=repository_root,
        captured=captured,
        spec=active_spec,
    )

    translation_records = {}
    for timestep in active_spec.timesteps:
        translation_records[
            str(timestep)
        ] = translation_invariance_audit(
            condition=
                context[
                    "objective_condition"
                ],
            control=
                context[
                    "objective_control_predictions"
                ][int(timestep)],
            condition_name=
                context[
                    "objective_condition_name"
                ],
            spec=active_spec,
        )

    baseline_records = []
    for definition in SURROGATE_DEFINITIONS[:2]:
        baseline_records.append(
            candidate_oof_record(
                definition=definition,
                context=context,
                spec=active_spec,
                baseline_means={},
            )
        )
    baseline_means = collect_baseline_means(
        baseline_records
    )
    candidate_records = list(
        baseline_records
    )
    for definition in SURROGATE_DEFINITIONS[2:]:
        candidate_records.append(
            candidate_oof_record(
                definition=definition,
                context=context,
                spec=active_spec,
                baseline_means=
                    baseline_means,
            )
        )

    selected_oof = (
        select_oof_candidate(
            candidate_records
        )
    )
    context[
        "candidate_records"
    ] = candidate_records
    permutation = None
    holdout = None
    if selected_oof is not None:
        permutation = permutation_control(
            selected=selected_oof,
            context=context,
            spec=active_spec,
        )
        if permutation["all_pass"]:
            holdout = (
                fit_selected_and_evaluate_holdout(
                    selected=selected_oof,
                    context=context,
                    spec=active_spec,
                )
            )

    classification = (
        classify_direction_surrogate(
            candidate_records=
                candidate_records,
            selected=selected_oof,
            permutation=permutation,
            holdout=holdout,
        )
    )
    validated_recommendation = (
        copy.deepcopy(
            selected_oof
        )
        if (
            selected_oof is not None
            and permutation is not None
            and permutation["all_pass"]
            and holdout is not None
            and holdout[
                "scientific_pass"
            ]
        )
        else None
    )

    contract = {
        "schema":
            "phase314b_r258_staged_direction_surrogate_contract_v1",
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
            "objective_fit_target_sha256":
                EXPECTED_OBJECTIVE_FIT_TARGET_SHA256,
            "objective_fit_group_sha256":
                EXPECTED_OBJECTIVE_FIT_GROUP_SHA256,
        },
        "environment":
            dict(environment),
        "cold_main_worker_context":
            cold,
        "control_capture":
            captured[
                "control_identity"
            ],
        "direction_surrogate_spec":
            asdict(active_spec),
        "surrogate_definitions": [
            asdict(definition)
            for definition
            in SURROGATE_DEFINITIONS
        ],
        "source_binding":
            source_binding,
        "feature_contract": {
            "cable_anchor_indices":
                list(
                    CABLE_ANCHOR_INDICES
                ),
            "segment_anchor_indices":
                list(
                    SEGMENT_ANCHOR_INDICES
                ),
            "feature_dimension":
                211,
            "condition_name_in_selectable_features":
                False,
            "condition_name_in_absolute_diagnostic_features":
                False,
            "condition_name_use":
                "nonselectable condition-mean shortcut baseline only",
            "condition_name_argument_ignored_for_deployable_modes":
                True,
            "robot_proxy_layout":
                "joint_position6,joint_velocity6,ee_position3,quaternion4",
            "selectable_robot_position":
                "ee_xy relative to same-step cable centroid; ee_z retained",
            "group_id_in_features":
                False,
            "pair_key_in_features":
                False,
            "visible_seed_in_features":
                False,
            "window_index_in_features":
                False,
            "selectable_features_translation_invariant":
                True,
            "absolute_features_selectable":
                False,
        },
        "fit_selection_contract": {
            "surrogate_target_space":
                "raw ordered-cable XY residual",
            "surrogate_target_standardizer":
                None,
            "fold_target_pca_fit":
                "training fold only",
            "surrogate_fit_population":
                "638-row objective train only",
            "candidate_selection_population":
                "six-fold grouped OOF objective train only",
            "selection_holdout_access":
                (
                    "locked final evaluation only after candidate "
                    "and scales are selected"
                ),
            "selection_holdout_changes_selection":
                False,
            "frozen_probe_accessed":
                False,
        },
        "translation_invariance":
            translation_records,
        "model_architecture_changed":
            False,
        "diffusion_model_candidate_trained":
            False,
        "direction_surrogate_trained":
            True,
        "surrogate_weights_persisted":
            False,
        "uses_frozen_probe":
            False,
    }
    contract[
        "contract_sha256"
    ] = sha256_bytes(
        stable_json_bytes(
            contract
        )
    )

    selection_payload = {
        "objective_train_selected_configuration":
            selected_oof,
        "permutation_control":
            permutation,
        "locked_holdout_evaluation":
            holdout,
        "validated_train_only_recommendation":
            validated_recommendation,
        "classification":
            classification,
        "candidate_records":
            candidate_records,
        "baseline_means":
            baseline_means,
        "selection_holdout_used_for_fit":
            False,
        "selection_holdout_used_for_candidate_selection":
            False,
        "selection_holdout_evaluated":
            bool(holdout is not None),
        "frozen_probe_accessed":
            False,
    }
    selection_payload[
        "selection_sha256"
    ] = sha256_bytes(
        stable_json_bytes(
            selection_payload
        )
    )

    scientific_status = (
        "READY_FOR_TRAIN_ONLY_DENOISER_GUIDANCE_CALIBRATION"
        if validated_recommendation
        is not None
        else "BLOCKED"
    )
    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema":
            "phase314b_r258_staged_worker_result_v1",
        "verdict": "PASS",
        "scientific_status":
            scientific_status,
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
            "objective_train_groups":
                int(
                    len(
                        set(
                            context[
                                "objective_groups"
                            ].tolist()
                        )
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
            "objective_fold_assignment_sha256":
                sha256_array(
                    context[
                        "objective_fold_assignment"
                    ]
                ),
            "objective_fold_mapping_sha256":
                sha256_bytes(
                    stable_json_bytes(
                        {
                            "mapping":
                                context[
                                    "objective_fold_mapping"
                                ]
                        }
                    )
                ),
            "frozen_probe_accessed":
                False,
        },
        "source_binding":
            source_binding,
        "objective_control_prediction_sha256":
            context[
                "objective_control_prediction_sha256"
            ],
        "translation_invariance":
            translation_records,
        "baseline_means":
            baseline_means,
        "candidate_records":
            candidate_records,
        "objective_train_selected_configuration":
            selected_oof,
        "permutation_control":
            permutation,
        "locked_holdout_evaluation":
            holdout,
        "classification":
            classification,
        "direction_surrogate_contract":
            contract,
        "selection":
            selection_payload,
        "selected_configuration":
            validated_recommendation,
        "train_only_recommendation":
            validated_recommendation,
        "direction_surrogate_trained":
            True,
        "new_diffusion_model_candidate_trained":
            False,
        "direct_x0_tensor_optimization_run":
            False,
        "balanced_objective_calibration_run":
            False,
        "control_replay_exact":
            bool(
                captured[
                    "control_identity"
                ][
                    "reference_equivalence_pass"
                ]
            ),
        "selection_holdout_evaluated":
            bool(holdout is not None),
        "selection_holdout_used_for_fit":
            False,
        "selection_holdout_used_for_selection":
            False,
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
        "surrogate_weights_persisted":
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
        "source_binding":
            result[
                "source_binding"
            ],
        "objective_control_prediction_sha256":
            result[
                "objective_control_prediction_sha256"
            ],
        "translation_invariance":
            result[
                "translation_invariance"
            ],
        "baseline_means":
            result["baseline_means"],
        "candidate_records":
            result["candidate_records"],
        "objective_train_selected_configuration":
            result[
                "objective_train_selected_configuration"
            ],
        "permutation_control":
            result[
                "permutation_control"
            ],
        "locked_holdout_evaluation":
            result[
                "locked_holdout_evaluation"
            ],
        "classification":
            result["classification"],
        "direction_surrogate_contract":
            result[
                "direction_surrogate_contract"
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
            left_payload == right_payload,
        "left_sha256":
            sha256_bytes(left_payload),
        "right_sha256":
            sha256_bytes(right_payload),
        "environment_exact": (
            left["environment"]
            == right["environment"]
        ),
        "control_capture_exact": (
            left["control_capture"]
            == right["control_capture"]
        ),
        "split_exact": (
            left["split"]
            == right["split"]
        ),
        "translation_exact": (
            left[
                "translation_invariance"
            ]
            == right[
                "translation_invariance"
            ]
        ),
        "candidate_records_exact": (
            left["candidate_records"]
            == right["candidate_records"]
        ),
        "permutation_exact": (
            left[
                "permutation_control"
            ]
            == right[
                "permutation_control"
            ]
        ),
        "holdout_exact": (
            left[
                "locked_holdout_evaluation"
            ]
            == right[
                "locked_holdout_evaluation"
            ]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
        "contract_exact": (
            left[
                "direction_surrogate_contract"
            ]
            == right[
                "direction_surrogate_contract"
            ]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
    }


STAGED_IMPLEMENTATION_PATHS = (
    "ccda_phase3/phase314b_r258_staged_direction_surrogate.py",
    "scripts/phase3_14b_r258_staged_worker.py",
    "scripts/phase3_14b_r258_staged_run_calibration.py",
    "scripts/phase3_14b_r258_staged_test_gate.py",
    "scripts/phase3_14b_r258_staged_blocked.py",
    "scripts/phase3_14b_r258_staged_run.sh",
    "tests/test_phase3_14b_r258_staged_direction_surrogate.py",
)


def status_paths(root: Path) -> Tuple[str, ...]:
    output = git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    paths = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(
                " -> ",
                1,
            )[1]
        paths.append(value)
    return tuple(sorted(paths))


def validate_initial_worktree(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(
        repository_root,
        "branch",
        "--show-current",
    ) != "Experiment1":
        raise DirectionSurrogateError(
            "Stage D requires Experiment1"
        )
    head = git_output(
        repository_root,
        "rev-parse",
        "HEAD",
    )
    remote = git_output(
        repository_root,
        "rev-parse",
        "origin/Experiment1",
    )
    if head != BASE_EVIDENCE_COMMIT:
        raise DirectionSurrogateError(
            "initial local HEAD changed: {}".format(head)
        )
    if remote != BASE_EVIDENCE_COMMIT:
        raise DirectionSurrogateError(
            "initial origin/Experiment1 changed: {}".format(remote)
        )
    base = validate_base_evidence(
        repository_root
    )
    observed = status_paths(
        repository_root
    )
    expected = tuple(
        sorted(
            STAGED_IMPLEMENTATION_PATHS
        )
    )
    if observed != expected:
        raise DirectionSurrogateError(
            "unexpected initial Stage-D worktree paths: {}".format(observed)
        )
    submodule = (
        repository_root
        / "external/deformable-ravens"
    )
    if git_output(
        submodule,
        "rev-parse",
        "HEAD",
    ) != EXPECTED_SUBMODULE_COMMIT:
        raise DirectionSurrogateError(
            "DeformableRavens commit changed"
        )
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise DirectionSurrogateError(
            "DeformableRavens worktree is dirty"
        )
    return {
        "head": head,
        "origin_experiment1": remote,
        "base_worker_sha256":
            base[
                "worker"
            ]["comparison"][
                "left_sha256"
            ],
        "base_test_count":
            base[
                "test_gate"
            ]["passed_test_count"],
        "expected_untracked_paths":
            list(expected),
        "submodule_commit":
            EXPECTED_SUBMODULE_COMMIT,
    }
