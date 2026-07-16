"""Phase3.14b-r2.5.8 Stage-A portable conflict-projected K16 calibration.

The frozen r2.5.7 Resume4 evidence remains the historical numerical reference,
but no physical GPU model, GPU UUID, PCI bus, compute capability, driver, or
full environment-observation SHA is a prerequisite.  Any CUDA GPU may execute
this stage when the frozen software contract is present and a disposable child
process successfully exercises the exact forward, dual-autograd, blockwise
projection, optimizer-step, and 100-row checkpoint operation family.

Before candidate training, the current hardware replays the frozen Stage-C
calibration and diffusion-only control using frozen helper functions.  Stable
structure, strings, booleans, integer counts, initial model/optimizer identity,
and source exposure remain exact.  Hardware-sensitive floating-point metrics
must satisfy pre-registered numerical tolerances; hardware-sensitive model,
optimizer, prediction, and history SHA values are retained as observations but
are not cross-hardware admission gates.  The current-hardware control is then
used for all candidate/control ratios.

Five train-only candidates compare global projection as a negative control
against blockwise projection over the noisy, condition, time, four residual,
output-normalization, and output-head blocks.  Candidate-selection thresholds
are unchanged.  Two isolated workers on the same rented instance must still be
byte-exact, proving within-instance determinism.

The frozen 126-row probe, full repaired model, reverse sampler, formal training,
IDM, environment execution, Phase4, and CPS remain closed.  No checkpoint,
weights, prediction tensor, candidate tensor, NPZ, cache, image, or video is
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_stagec_reverse_attribution as stagec
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3 import phase314b_r257_stageb_mechanism_audit as stageb_mechanism
from ccda_phase3 import phase314b_r257_stagec_topk_quadratic as stagec_topk
from ccda_phase3 import phase314b_r257_staged_timestep_gate as staged
from ccda_phase3 import phase314b_r257_staged_resume3_calibration_trace as resume3
from ccda_phase3 import phase314b_r257_staged_resume4_3090 as resume4

PHASE = "Phase3.14b-r2.5.8 Stage A"
PHASE_ID = "phase314b_r258_stagea"

BASE_EVIDENCE_COMMIT = (
    "f0a5bec1f89f625e74150e55e2da884413d72eb9"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

RESUME4_CONTRACT = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_contract.json"
)
RESUME4_WORKER = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_worker_evidence.json"
)
RESUME4_SUMMARY = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_summary.json"
)
RESUME4_REPORT = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_report.md"
)
RESUME4_TEST_GATE = (
    "reports/"
    "phase3_14b_r257_staged_resume4_3090_test_gate_summary.json"
)

EXPECTED_RESUME4_WORKER_SHA256 = (
    "0e87be2ae637ad1dd3680535e40ab8b29f7d7e54bf8c6555817d4da4617cac63"
)
EXPECTED_RESUME4_ENVIRONMENT_SHA256 = (
    "7b51113fb013a7f24716049fe4e11464261af62b0b3c4950cdf454fc21aad260"
)
EXPECTED_STAGEC_CALIBRATION_SHA256 = (
    "fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0"
)
EXPECTED_STAGEC_CONTROL_SHA256 = (
    "dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7"
)
EXPECTED_RESUME4_SELECTION_SHA256 = (
    "136d4575add0452759106f4f141837dc78ff72f042eb0d08ea37cb0c96334fb6"
)
EXPECTED_RESUME4_CONTRACT_SHA256 = (
    "cc2cb6f87947cceb070104a7f50a04e1f40d9074612007f4790688bb77111c50"
)
EXPECTED_RESUME4_CONTRACT_FILE_SHA256 = (
    "16bccdbb3ed803ab1fd7e5a78d8a80ae931d4238b0b21220c8cd21db1cd1e936"
)
EXPECTED_STAGEC_INITIAL_MODEL_SHA256 = (
    "4f1c102d9f39ba59544f5565b4c01fe03e073fca596aefdc822b707cd2f2c7fd"
)

CONTROL_CANDIDATE_ID = stagec_topk.CONTROL_CANDIDATE_ID
PROJECTION_MODES = ("global", "blockwise")
EXPECTED_BLOCK_ORDER = (
    "noisy_projection",
    "condition_projection",
    "time_projection",
    "residual_block_0",
    "residual_block_1",
    "residual_block_2",
    "residual_block_3",
    "output_norm",
    "output_head",
)


class ConflictProjectionError(RuntimeError):
    """Raised when an immutable identity or projection contract fails."""


@dataclass(frozen=True)
class ProjectionCandidateDefinition:
    projection_mode: str
    timestep_cutoff: int
    target_projected_ratio: float

    def validate(self) -> None:
        if self.projection_mode not in PROJECTION_MODES:
            raise ValueError("unknown projection mode")
        if self.timestep_cutoff not in (25, 50):
            raise ValueError("unregistered timestep cutoff")
        if not 0.0 < float(self.target_projected_ratio) <= 0.50:
            raise ValueError("projected gradient ratio is outside (0,0.5]")


@dataclass(frozen=True)
class ConflictProjectionSpec:
    """Pre-registered conflict-projection calibration contract."""

    top_k: int = 16
    candidate_definitions: Tuple[ProjectionCandidateDefinition, ...] = (
        ProjectionCandidateDefinition("global", 50, 0.25),
        ProjectionCandidateDefinition("blockwise", 25, 0.25),
        ProjectionCandidateDefinition("blockwise", 50, 0.10),
        ProjectionCandidateDefinition("blockwise", 50, 0.25),
        ProjectionCandidateDefinition("blockwise", 50, 0.50),
    )
    calibration_batch_size: int = 100
    calibration_seed_offset: int = 6301
    holdout_noise_seed_offset: int = 4101
    completed_step_checkpoints: Tuple[int, ...] = (
        1,
        100,
        500,
        2000,
        8000,
    )
    projection_epsilon: float = 1.0e-12
    post_projection_dot_tolerance: float = 1.0e-6
    row_loss_epsilon: float = 1.0e-12

    t10_mean_excess_reduction_min: float = 0.20
    t25_mean_excess_reduction_min: float = 0.15
    t50_mean_excess_reduction_min: float = 0.10
    top8_mean_excess_reduction_min: float = 0.0
    p95_excess_ratio_max: float = 1.0
    positive_segment_count_ratio_max: float = 1.0
    train_control_nmse_ratio_max: float = 1.20
    one_step_nmse_ratio_max: float = 1.20
    binary_t10_upper_row_min: float = 0.75
    binary_t25_upper_row_min: float = 0.60
    binary_t50_upper_row_min: float = 0.50
    gradient_clip_frequency_max: float = 0.25
    largest_row_contribution_max: float = 0.25
    top_five_percent_contribution_max: float = 0.60
    active_fraction_tolerance: float = 0.03
    post_projection_violation_frequency_max: float = 0.0
    diffusion_gradient_change_count_max: int = 0

    inactive_projection_trigger_frequency_max: float = 0.01
    inactive_removed_norm_fraction_max: float = 0.005

    def validate(self) -> None:
        maximum_positions = (
            stageb.FUTURE_STEPS * (stageb.BEADS - 1)
        )
        if not 0 < int(self.top_k) <= maximum_positions:
            raise ValueError("top-k is outside ordered geometry")
        if self.calibration_batch_size != stageb.DiagnosticSpec().train_timesteps:
            raise ValueError(
                "calibration batch must contain every timestep once"
            )
        if (
            tuple(sorted(set(self.completed_step_checkpoints)))
            != self.completed_step_checkpoints
        ):
            raise ValueError("checkpoints must be ordered and unique")
        if (
            self.completed_step_checkpoints[-1]
            != stageb.DiagnosticSpec().train_steps
        ):
            raise ValueError(
                "final checkpoint must equal frozen training steps"
            )
        if len(self.candidate_definitions) != 5:
            raise ValueError("candidate population changed")
        expected = (
            ("global", 50, 0.25),
            ("blockwise", 25, 0.25),
            ("blockwise", 50, 0.10),
            ("blockwise", 50, 0.25),
            ("blockwise", 50, 0.50),
        )
        observed = tuple(
            (
                item.projection_mode,
                int(item.timestep_cutoff),
                float(item.target_projected_ratio),
            )
            for item in self.candidate_definitions
        )
        if observed != expected:
            raise ValueError("candidate definitions changed")
        for item in self.candidate_definitions:
            item.validate()
        for value in (
            self.projection_epsilon,
            self.post_projection_dot_tolerance,
            self.row_loss_epsilon,
            self.t10_mean_excess_reduction_min,
            self.t25_mean_excess_reduction_min,
            self.t50_mean_excess_reduction_min,
            self.p95_excess_ratio_max,
            self.positive_segment_count_ratio_max,
            self.train_control_nmse_ratio_max,
            self.one_step_nmse_ratio_max,
            self.binary_t10_upper_row_min,
            self.binary_t25_upper_row_min,
            self.binary_t50_upper_row_min,
            self.gradient_clip_frequency_max,
            self.largest_row_contribution_max,
            self.top_five_percent_contribution_max,
            self.active_fraction_tolerance,
        ):
            if float(value) <= 0.0:
                raise ValueError(
                    "registered positive threshold is invalid"
                )
        if self.top8_mean_excess_reduction_min < 0.0:
            raise ValueError("top-8 threshold is negative")
        if self.post_projection_violation_frequency_max != 0.0:
            raise ValueError(
                "post-projection violations are not permitted"
            )
        if self.diffusion_gradient_change_count_max != 0:
            raise ValueError(
                "diffusion-gradient mutation is not permitted"
            )


@dataclass(frozen=True)
class ProjectedCandidate:
    candidate_id: str
    projection_mode: str
    timestep_cutoff: int
    top_k: int
    target_gradient_ratio: float
    lambda_upper: float
    initial_model_sha256: str
    diffusion_gradient_norm: float
    geometry_gradient_norm: float
    projected_geometry_gradient_norm: float
    pre_projection_cosine: float
    post_projection_cosine: float
    projection_triggered_block_fraction: float
    removed_geometry_norm_fraction: float

    def validate(self) -> None:
        if self.projection_mode not in PROJECTION_MODES:
            raise ValueError("candidate projection mode is invalid")
        if not self.candidate_id:
            raise ValueError("candidate id is empty")
        if int(self.top_k) != 16:
            raise ValueError("candidate top-k changed")
        if int(self.timestep_cutoff) not in (25, 50):
            raise ValueError("candidate cutoff changed")
        for value in (
            self.target_gradient_ratio,
            self.lambda_upper,
            self.diffusion_gradient_norm,
            self.geometry_gradient_norm,
            self.projected_geometry_gradient_norm,
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    "candidate positive scalar is invalid"
                )
        for value in (
            self.pre_projection_cosine,
            self.post_projection_cosine,
        ):
            if not -1.000001 <= float(value) <= 1.000001:
                raise ValueError("candidate cosine is invalid")
        for value in (
            self.projection_triggered_block_fraction,
            self.removed_geometry_norm_fraction,
        ):
            if not 0.0 <= float(value) <= 1.000001:
                raise ValueError(
                    "candidate projection fraction is invalid"
                )
        if len(self.initial_model_sha256) != 64:
            raise ValueError("candidate model SHA is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(
        np.ascontiguousarray(array).tobytes()
    )
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if (
            value.dtype.kind in "fc"
            and not np.all(np.isfinite(value))
        ):
            raise ValueError(
                "non-finite array cannot be serialized"
            )
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
    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(
                "non-finite scalar cannot be serialized"
            )
        return value
    raise TypeError(
        "unsupported JSON value: {!r}".format(
            type(value)
        )
    )


def stable_json_bytes(
    payload: Mapping[str, Any],
) -> bytes:
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
            "refusing to overwrite write-once output: "
            "{}".format(target)
        )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = target.parent / (
        ".{}.{}.tmp".format(
            target.name,
            os.getpid(),
        )
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
        directory_fd = os.open(
            str(target.parent),
            os.O_RDONLY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise ConflictProjectionError(
            "JSON root is not an object: {}".format(
                path
            )
        )
    return value


def assert_commit_ancestor(
    root: Path,
    commit: str,
) -> None:
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ConflictProjectionError(
            "required commit is not an ancestor: "
            "{}".format(commit)
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
        [
            "git",
            "show",
            "{}:{}".format(
                commit,
                relative,
            ),
        ],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if observed != committed:
        raise ConflictProjectionError(
            "immutable Resume4 file changed: "
            "{}".format(relative)
        )
    return sha256_bytes(observed)


def validate_resume4_evidence(
    root: Path,
) -> Dict[str, Any]:
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
        for relative in (
            RESUME4_CONTRACT,
            RESUME4_WORKER,
            RESUME4_SUMMARY,
            RESUME4_REPORT,
            RESUME4_TEST_GATE,
        )
    }
    if (
        file_sha[RESUME4_CONTRACT]
        != EXPECTED_RESUME4_CONTRACT_FILE_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 contract-file SHA changed"
        )

    summary = load_json(
        repository_root / RESUME4_SUMMARY
    )
    worker = load_json(
        repository_root / RESUME4_WORKER
    )
    contract = load_json(
        repository_root / RESUME4_CONTRACT
    )
    test_gate = load_json(
        repository_root / RESUME4_TEST_GATE
    )

    if summary.get("verdict") != "PASS":
        raise ConflictProjectionError(
            "Resume4 verdict changed"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise ConflictProjectionError(
            "Resume4 scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r257_staged_timestep_gated_k16_"
        "no_geometry_or_fidelity_solution"
    ):
        raise ConflictProjectionError(
            "Resume4 root cause changed"
        )
    if summary.get("required_next_path") != (
        "CALIBRATE_CONFLICT_PROJECTED_LOW_NOISE_K16_"
        "OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
    ):
        raise ConflictProjectionError(
            "Resume4 next path changed"
        )
    if summary.get("selected_configuration") is not None:
        raise ConflictProjectionError(
            "Resume4 selection is no longer empty"
        )
    if summary.get("train_only_recommendation") is not None:
        raise ConflictProjectionError(
            "Resume4 recommendation is no longer empty"
        )
    if summary.get("workers_exact") is not True:
        raise ConflictProjectionError(
            "Resume4 workers are not exact"
        )

    comparison = worker.get("comparison", {})
    if comparison.get("exact") is not True:
        raise ConflictProjectionError(
            "Resume4 worker comparison changed"
        )
    if (
        comparison.get("left_sha256")
        != EXPECTED_RESUME4_WORKER_SHA256
        or comparison.get("right_sha256")
        != EXPECTED_RESUME4_WORKER_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 worker identity changed"
        )

    result = worker.get("worker_result")
    if not isinstance(result, dict):
        raise ConflictProjectionError(
            "Resume4 worker result is missing"
        )
    environment = (
        result.get("resume4_3090_environment", {})
        .get("environment_probe", {})
    )
    if (
        environment.get("environment_sha256")
        != EXPECTED_RESUME4_ENVIRONMENT_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 environment identity changed"
        )
    attribution = result.get(
        "resume3_calibration_attribution",
        {},
    )
    capture = attribution.get(
        "stagec_entrypoint_capture",
        {},
    )
    if (
        capture.get("calibration_sha256")
        != EXPECTED_STAGEC_CALIBRATION_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 Stage-C calibration changed"
        )
    if (
        capture.get("control_record_sha256")
        != EXPECTED_STAGEC_CONTROL_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 Stage-C control changed"
        )
    if capture.get("calibration_exact") is not True:
        raise ConflictProjectionError(
            "Resume4 calibration is not exact"
        )
    if capture.get("control_record_exact") is not True:
        raise ConflictProjectionError(
            "Resume4 control is not exact"
        )
    if (
        int(
            capture.get(
                "stagec_nonzero_candidate_training_count",
                -1,
            )
        )
        != 0
    ):
        raise ConflictProjectionError(
            "Resume4 Stage-C candidate count changed"
        )
    selection = result.get("selection", {})
    if (
        selection.get("selection_sha256")
        != EXPECTED_RESUME4_SELECTION_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 selection identity changed"
        )
    result_contract = result.get(
        "calibration_contract",
        {},
    )
    if (
        result_contract.get("contract_sha256")
        != EXPECTED_RESUME4_CONTRACT_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 internal contract changed"
        )
    if (
        contract.get("contract_sha256")
        != EXPECTED_RESUME4_CONTRACT_SHA256
    ):
        raise ConflictProjectionError(
            "Resume4 contract record changed"
        )

    if test_gate.get("verdict") != "PASS":
        raise ConflictProjectionError(
            "Resume4 test gate is not PASS"
        )
    if int(
        test_gate.get("test_file_count", -1)
    ) != 46:
        raise ConflictProjectionError(
            "Resume4 test-file count changed"
        )
    if int(
        test_gate.get("passed_test_count", -1)
    ) != 837:
        raise ConflictProjectionError(
            "Resume4 pass count changed"
        )
    if int(
        test_gate.get(
            "r257_staged_resume4_3090_new_passed",
            -1,
        )
    ) != 25:
        raise ConflictProjectionError(
            "Resume4 new-test count changed"
        )
    for key in (
        "frozen_probe_accessed",
        "reverse_sampling_run",
        "full_stageb_repaired_model_trained",
        "formal_pilot_run",
        "candidate_execution",
        "deformable_ravens_executed",
        "phase4",
        "cps",
    ):
        if summary.get(key) is not False:
            raise ConflictProjectionError(
                "Resume4 boundary changed: {}".format(
                    key
                )
            )

    return {
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "file_sha256": file_sha,
        "summary": summary,
        "worker": worker,
        "worker_result": result,
        "contract": contract,
        "test_gate": test_gate,
        "worker_identity_sha256":
            EXPECTED_RESUME4_WORKER_SHA256,
        "environment_sha256":
            EXPECTED_RESUME4_ENVIRONMENT_SHA256,
        "stagec_calibration_sha256":
            EXPECTED_STAGEC_CALIBRATION_SHA256,
        "stagec_control_sha256":
            EXPECTED_STAGEC_CONTROL_SHA256,
        "selection_sha256":
            EXPECTED_RESUME4_SELECTION_SHA256,
        "contract_sha256":
            EXPECTED_RESUME4_CONTRACT_SHA256,
        "contract_file_sha256":
            EXPECTED_RESUME4_CONTRACT_FILE_SHA256,
    }


def source_logic_audit(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageb_path = (
        repository_root
        / "ccda_phase3/"
        "phase314b_r256_stageb_cable_diffusion.py"
    )
    staged_path = (
        repository_root
        / "ccda_phase3/"
        "phase314b_r257_staged_timestep_gate.py"
    )
    stageb_text = stageb_path.read_text(
        encoding="utf-8"
    )
    staged_text = staged_path.read_text(
        encoding="utf-8"
    )
    checks = {
        "direct_x0_residual_model": (
            "predicted = flat + self.output"
            in stageb_text
        ),
        "four_residual_blocks": (
            "residual_blocks: int = 4"
            in stageb_text
        ),
        "noisy_projection_present": (
            "self.noisy_projection"
            in stageb_text
        ),
        "condition_projection_present": (
            "self.condition_projection"
            in stageb_text
        ),
        "time_projection_present": (
            "self.time_projection"
            in stageb_text
        ),
        "output_norm_present": (
            "self.output_norm"
            in stageb_text
        ),
        "hard_low_noise_gate_preserved": (
            "active iff sampled diffusion timestep <= cutoff"
            in staged_text
            or "timestep <= int(cutoff)"
            in staged_text
        ),
        "active_row_conditional_mean_preserved": (
            "torch.sum(base[\"row_loss\"] * weights) / denominator"
            in staged_text
        ),
        "topk_quadratic_preserved": (
            "gated_topk_quadratic_terms_torch"
            in staged_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(
            all(checks.values())
        ),
        "source_sha256": {
            stageb_path.relative_to(
                repository_root
            ).as_posix():
                sha256_file(stageb_path),
            staged_path.relative_to(
                repository_root
            ).as_posix():
                sha256_file(staged_path),
        },
        "model_parameterization_changed": False,
        "geometry_target_changed": False,
        "optimizer_update_rule_changed": True,
    }


def ratio_token(value: float) -> str:
    return (
        "{:.2f}".format(float(value))
        .replace(".", "p")
        .replace("-", "m")
    )


def candidate_id(
    definition: ProjectionCandidateDefinition,
) -> str:
    mode = (
        "global"
        if definition.projection_mode == "global"
        else "block"
    )
    return (
        "{}_t{}_r{}".format(
            mode,
            int(definition.timestep_cutoff),
            ratio_token(
                definition.target_projected_ratio
            ),
        )
    )


def parameter_block_name(name: str) -> str:
    if name.startswith("noisy_projection."):
        return "noisy_projection"
    if name.startswith("condition_projection."):
        return "condition_projection"
    if name.startswith("time_projection."):
        return "time_projection"
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) < 3:
            raise ConflictProjectionError(
                "residual parameter name is malformed: "
                "{}".format(name)
            )
        index = int(parts[1])
        return "residual_block_{}".format(index)
    if name.startswith("output_norm."):
        return "output_norm"
    if name.startswith("output."):
        return "output_head"
    raise ConflictProjectionError(
        "unregistered model parameter: {}".format(
            name
        )
    )


def parameter_block_map(
    named_parameters: Sequence[Tuple[str, Any]],
) -> Dict[str, List[int]]:
    blocks: MutableMapping[str, List[int]] = {}
    for index, (name, _parameter) in enumerate(
        named_parameters
    ):
        block = parameter_block_name(name)
        blocks.setdefault(block, []).append(index)
    if tuple(blocks.keys()) != EXPECTED_BLOCK_ORDER:
        raise ConflictProjectionError(
            "model block order changed: {}".format(
                tuple(blocks.keys())
            )
        )
    if any(not indices for indices in blocks.values()):
        raise ConflictProjectionError(
            "empty parameter block"
        )
    covered = sorted(
        index
        for indices in blocks.values()
        for index in indices
    )
    if covered != list(range(len(named_parameters))):
        raise ConflictProjectionError(
            "parameter block coverage is incomplete"
        )
    return {
        key: list(value)
        for key, value in blocks.items()
    }


def gradient_sequence_sha256(
    named_parameters: Sequence[Tuple[str, Any]],
    gradients: Sequence[Any],
) -> str:
    torch, _ = stageb._torch_imports()
    digest = hashlib.sha256()
    for (name, parameter), gradient in zip(
        named_parameters,
        gradients,
    ):
        value = (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient
        )
        tensor = (
            value.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
        )
        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tuple(tensor.shape)).encode(
                "utf-8"
            )
        )
        digest.update(
            tensor.numpy().tobytes()
        )
    return digest.hexdigest()


def _safe_gradient(
    parameter: Any,
    gradient: Optional[Any],
) -> Any:
    torch, _ = stageb._torch_imports()
    return (
        torch.zeros_like(parameter)
        if gradient is None
        else gradient
    )


def _group_metrics(
    diffusion: Sequence[Any],
    geometry: Sequence[Any],
    projected: Sequence[Any],
    indices: Sequence[int],
    *,
    epsilon: float,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    d_sq = torch.zeros(
        (),
        device=diffusion[indices[0]].device,
        dtype=torch.float32,
    )
    g_sq = torch.zeros_like(d_sq)
    p_sq = torch.zeros_like(d_sq)
    dot_pre = torch.zeros_like(d_sq)
    dot_post = torch.zeros_like(d_sq)
    removed_sq = torch.zeros_like(d_sq)
    elements = 0
    active_elements = 0
    conflict_elements = 0
    for index in indices:
        d = diffusion[index]
        g = geometry[index]
        p = projected[index]
        d_sq = d_sq + torch.sum(d * d)
        g_sq = g_sq + torch.sum(g * g)
        p_sq = p_sq + torch.sum(p * p)
        dot_pre = dot_pre + torch.sum(d * g)
        dot_post = dot_post + torch.sum(d * p)
        removed = g - p
        removed_sq = (
            removed_sq
            + torch.sum(removed * removed)
        )
        elements += int(g.numel())
        active = (d != 0.0) & (g != 0.0)
        active_elements += int(
            torch.sum(active).detach().cpu()
        )
        conflict_elements += int(
            torch.sum(
                (d * g < 0.0) & active
            ).detach().cpu()
        )

    d_sq_f = float(d_sq.detach().cpu())
    g_sq_f = float(g_sq.detach().cpu())
    p_sq_f = float(p_sq.detach().cpu())
    dot_pre_f = float(dot_pre.detach().cpu())
    dot_post_f = float(dot_post.detach().cpu())
    removed_sq_f = float(
        removed_sq.detach().cpu()
    )
    d_norm = math.sqrt(max(d_sq_f, 0.0))
    g_norm = math.sqrt(max(g_sq_f, 0.0))
    p_norm = math.sqrt(max(p_sq_f, 0.0))
    removed_norm = math.sqrt(
        max(removed_sq_f, 0.0)
    )
    pre_denominator = max(
        d_norm * g_norm,
        float(epsilon),
    )
    post_denominator = max(
        d_norm * p_norm,
        float(epsilon),
    )
    return {
        "element_count": int(elements),
        "diffusion_norm": d_norm,
        "geometry_norm": g_norm,
        "projected_geometry_norm": p_norm,
        "removed_geometry_norm": removed_norm,
        "pre_dot": dot_pre_f,
        "post_dot": dot_post_f,
        "pre_cosine":
            dot_pre_f / pre_denominator,
        "post_cosine":
            dot_post_f / post_denominator,
        "active_element_count":
            int(active_elements),
        "conflict_fraction_among_active": (
            float(conflict_elements)
            / float(active_elements)
            if active_elements
            else 0.0
        ),
        "retained_norm_fraction": (
            p_norm / g_norm
            if g_norm > 0.0
            else 0.0
        ),
        "removed_norm_fraction": (
            removed_norm / g_norm
            if g_norm > 0.0
            else 0.0
        ),
    }


def project_gradient_pair(
    named_parameters: Sequence[Tuple[str, Any]],
    diffusion_gradients: Sequence[Optional[Any]],
    geometry_gradients: Sequence[Optional[Any]],
    *,
    projection_mode: str,
    lambda_upper: float,
    epsilon: float,
    post_dot_tolerance: float,
    include_per_parameter: bool,
) -> Tuple[List[Any], Dict[str, Any]]:
    torch, _ = stageb._torch_imports()
    if projection_mode not in PROJECTION_MODES:
        raise ValueError("unknown projection mode")
    if len(named_parameters) != len(diffusion_gradients):
        raise ValueError("diffusion gradient length mismatch")
    if len(named_parameters) != len(geometry_gradients):
        raise ValueError("geometry gradient length mismatch")

    diffusion = [
        _safe_gradient(parameter, gradient)
        for (_name, parameter), gradient in zip(
            named_parameters,
            diffusion_gradients,
        )
    ]
    geometry = [
        _safe_gradient(parameter, gradient)
        for (_name, parameter), gradient in zip(
            named_parameters,
            geometry_gradients,
        )
    ]
    diffusion_sha_before = (
        gradient_sequence_sha256(
            named_parameters,
            diffusion,
        )
    )

    block_map = parameter_block_map(
        named_parameters
    )
    groups = (
        {"global": list(range(len(named_parameters)))}
        if projection_mode == "global"
        else block_map
    )
    projected: List[Any] = [
        gradient.detach().clone()
        for gradient in geometry
    ]
    group_records: MutableMapping[str, Any] = {}
    triggered_count = 0
    post_violation_count = 0

    for group_name, indices in groups.items():
        dot = torch.zeros(
            (),
            device=diffusion[indices[0]].device,
            dtype=torch.float32,
        )
        d_sq = torch.zeros_like(dot)
        for index in indices:
            dot = (
                dot
                + torch.sum(
                    diffusion[index] * geometry[index]
                )
            )
            d_sq = (
                d_sq
                + torch.sum(
                    diffusion[index]
                    * diffusion[index]
                )
            )
        dot_float = float(dot.detach().cpu())
        d_sq_float = float(d_sq.detach().cpu())
        triggered = bool(
            dot_float < 0.0
            and d_sq_float > float(epsilon)
        )
        if triggered:
            coefficient = (
                dot
                / (
                    d_sq
                    + torch.as_tensor(
                        float(epsilon),
                        device=d_sq.device,
                        dtype=d_sq.dtype,
                    )
                )
            )
            for index in indices:
                projected[index] = (
                    geometry[index]
                    - coefficient * diffusion[index]
                ).detach()
            triggered_count += 1

        metrics = _group_metrics(
            diffusion,
            geometry,
            projected,
            indices,
            epsilon=float(epsilon),
        )
        tolerance = (
            float(post_dot_tolerance)
            * max(
                metrics["diffusion_norm"]
                * metrics[
                    "projected_geometry_norm"
                ],
                1.0,
            )
        )
        post_violation = bool(
            metrics["post_dot"] < -tolerance
        )
        if post_violation:
            post_violation_count += 1
        group_records[group_name] = {
            **metrics,
            "projection_triggered": triggered,
            "post_dot_tolerance": tolerance,
            "post_projection_violation":
                post_violation,
            "parameter_names": [
                named_parameters[index][0]
                for index in indices
            ],
        }

    diffusion_sha_after = (
        gradient_sequence_sha256(
            named_parameters,
            diffusion,
        )
    )
    if diffusion_sha_after != diffusion_sha_before:
        raise ConflictProjectionError(
            "projection mutated the diffusion gradient"
        )

    all_indices = list(
        range(len(named_parameters))
    )
    global_metrics = _group_metrics(
        diffusion,
        geometry,
        projected,
        all_indices,
        epsilon=float(epsilon),
    )
    combined = [
        (
            d
            + float(lambda_upper) * p
        ).detach()
        for d, p in zip(
            diffusion,
            projected,
        )
    ]
    combined_sq = 0.0
    for value in combined:
        combined_sq += float(
            torch.sum(value * value)
            .detach()
            .cpu()
        )
    combined_norm = math.sqrt(
        max(combined_sq, 0.0)
    )
    global_metrics.update(
        {
            "scaled_projected_geometry_norm":
                float(lambda_upper)
                * global_metrics[
                    "projected_geometry_norm"
                ],
            "scaled_projected_to_diffusion_ratio": (
                float(lambda_upper)
                * global_metrics[
                    "projected_geometry_norm"
                ]
                / global_metrics["diffusion_norm"]
                if global_metrics[
                    "diffusion_norm"
                ]
                > 0.0
                else 0.0
            ),
            "combined_norm": combined_norm,
            "combined_to_diffusion_ratio": (
                combined_norm
                / global_metrics["diffusion_norm"]
                if global_metrics[
                    "diffusion_norm"
                ]
                > 0.0
                else 0.0
            ),
            "projection_triggered_group_count":
                int(triggered_count),
            "projection_group_count":
                int(len(groups)),
            "projection_triggered_group_fraction": (
                float(triggered_count)
                / float(len(groups))
            ),
            "post_projection_violation_count":
                int(post_violation_count),
            "post_projection_violation_fraction": (
                float(post_violation_count)
                / float(len(groups))
            ),
            "diffusion_gradient_sha256":
                diffusion_sha_before,
            "diffusion_gradient_unchanged":
                True,
            "geometry_gradient_sha256":
                gradient_sequence_sha256(
                    named_parameters,
                    geometry,
                ),
            "projected_geometry_gradient_sha256":
                gradient_sequence_sha256(
                    named_parameters,
                    projected,
                ),
            "combined_gradient_sha256":
                gradient_sequence_sha256(
                    named_parameters,
                    combined,
                ),
        }
    )

    per_parameter: MutableMapping[str, Any] = {}
    if include_per_parameter:
        for index, (name, parameter) in enumerate(
            named_parameters
        ):
            metrics = _group_metrics(
                diffusion,
                geometry,
                projected,
                [index],
                epsilon=float(epsilon),
            )
            tolerance = (
                float(post_dot_tolerance)
                * max(
                    metrics["diffusion_norm"]
                    * metrics[
                        "projected_geometry_norm"
                    ],
                    1.0,
                )
            )
            per_parameter[name] = {
                **metrics,
                "shape": list(parameter.shape),
                "block":
                    parameter_block_name(name),
                "post_dot_tolerance":
                    tolerance,
                "post_projection_violation":
                    bool(
                        metrics["post_dot"]
                        < -tolerance
                    ),
            }

    return combined, {
        "projection_mode": projection_mode,
        "global": global_metrics,
        "per_group": dict(group_records),
        "per_parameter": dict(per_parameter),
    }


def fixed_batch_projected_gradients(
    *,
    model: Any,
    calibration_batch: Mapping[str, Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: int,
    timestep_cutoff: int,
    projection_mode: str,
    lambda_upper: float,
    spec: ConflictProjectionSpec,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    device = torch.device("cuda:0")
    noisy = torch.as_tensor(
        calibration_batch["noisy_z"],
        dtype=torch.float32,
        device=device,
    )
    timestep = torch.as_tensor(
        calibration_batch["timesteps"],
        dtype=torch.long,
        device=device,
    )
    condition = torch.as_tensor(
        calibration_batch["condition_z"],
        dtype=torch.float32,
        device=device,
    )
    clean = torch.as_tensor(
        calibration_batch["target_z"],
        dtype=torch.float32,
        device=device,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [
        parameter
        for _name, parameter
        in named_parameters
    ]

    predicted = model(
        noisy,
        timestep,
        condition,
    )
    diffusion_loss = torch.mean(
        (predicted - clean) ** 2
    )
    geometry = (
        staged.gated_topk_quadratic_terms_torch(
            predicted,
            timestep,
            target_standardizer=
                target_standardizer,
            objective_contract=
                objective_contract,
            top_k=int(top_k),
            timestep_cutoff=
                int(timestep_cutoff),
        )
    )
    diffusion_gradients = torch.autograd.grad(
        diffusion_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    geometry_gradients = torch.autograd.grad(
        geometry["total"],
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    _combined, projection = (
        project_gradient_pair(
            named_parameters,
            diffusion_gradients,
            geometry_gradients,
            projection_mode=projection_mode,
            lambda_upper=float(lambda_upper),
            epsilon=spec.projection_epsilon,
            post_dot_tolerance=
                spec.post_projection_dot_tolerance,
            include_per_parameter=True,
        )
    )
    row_loss = (
        geometry["row_loss"]
        .detach()
        .cpu()
        .numpy()
    )
    selected_indices = (
        geometry["selected_indices"]
        .detach()
        .cpu()
        .numpy()
    )
    selected_excess = (
        geometry["selected_excess"]
        .detach()
        .cpu()
        .numpy()
    )
    model.zero_grad(set_to_none=True)
    return {
        "diffusion_loss":
            float(diffusion_loss.detach().cpu()),
        "geometry_loss":
            float(geometry["total"].detach().cpu()),
        "ungated_geometry_loss":
            float(
                geometry["ungated_total"]
                .detach()
                .cpu()
            ),
        "gate": {
            "timestep_cutoff":
                int(timestep_cutoff),
            "active_count":
                int(
                    geometry["active_count"]
                    .detach()
                    .cpu()
                ),
            "active_fraction":
                float(
                    geometry["active_fraction"]
                    .detach()
                    .cpu()
                ),
            "zero_active":
                bool(
                    geometry["zero_active"]
                    .detach()
                    .cpu()
                ),
        },
        "gradient":
            projection["global"],
        "projection": projection,
        "row_contribution":
            stagec_topk.row_contribution_profile(
                row_loss,
                epsilon=
                    float(spec.row_loss_epsilon),
            ),
        "selected_positions":
            stagec_topk.selected_position_profile(
                selected_indices
            ),
        "selected_excess":
            stagec_topk.safe_stats(
                selected_excess
            ),
        "selected_excess_sha256":
            sha256_array(selected_excess),
        "predicted_x0_sha256":
            sha256_array(
                predicted.detach().cpu().numpy()
            ),
    }


def calibration_definition_key(
    definition: ProjectionCandidateDefinition,
) -> str:
    return "{}_t{}".format(
        definition.projection_mode,
        int(definition.timestep_cutoff),
    )


def calibrate_candidates(
    *,
    stageb_spec: stageb.DiagnosticSpec,
    calibration_batch: Mapping[str, Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    spec: ConflictProjectionSpec,
) -> Tuple[List[ProjectedCandidate], Dict[str, Any]]:
    stageb.set_deterministic_runtime(
        stageb_spec.seed
    )
    model = stageb.make_model(
        stageb_spec
    ).to("cuda:0")
    initial_model_sha = (
        stageb.tensor_state_sha256(model)
    )
    if (
        initial_model_sha
        != EXPECTED_STAGEC_INITIAL_MODEL_SHA256
    ):
        raise ConflictProjectionError(
            "common initial model SHA changed"
        )

    raw_records: MutableMapping[str, Any] = {}
    candidates: List[ProjectedCandidate] = []
    for definition in spec.candidate_definitions:
        key = calibration_definition_key(
            definition
        )
        if key not in raw_records:
            raw_records[key] = (
                fixed_batch_projected_gradients(
                    model=model,
                    calibration_batch=
                        calibration_batch,
                    target_standardizer=
                        target_standardizer,
                    objective_contract=
                        objective_contract,
                    top_k=spec.top_k,
                    timestep_cutoff=
                        definition.timestep_cutoff,
                    projection_mode=
                        definition.projection_mode,
                    lambda_upper=1.0,
                    spec=spec,
                )
            )
        raw = raw_records[key]
        gradient = raw["gradient"]
        diffusion_norm = float(
            gradient["diffusion_norm"]
        )
        geometry_norm = float(
            gradient["geometry_norm"]
        )
        projected_norm = float(
            gradient[
                "projected_geometry_norm"
            ]
        )
        if (
            not np.isfinite(diffusion_norm)
            or not np.isfinite(geometry_norm)
            or not np.isfinite(projected_norm)
            or diffusion_norm <= 0.0
            or geometry_norm <= 0.0
            or projected_norm <= 0.0
        ):
            raise ConflictProjectionError(
                "invalid projected calibration norms: "
                "{}".format(key)
            )
        lambda_upper = (
            float(
                definition.target_projected_ratio
            )
            * diffusion_norm
            / projected_norm
        )
        candidate = ProjectedCandidate(
            candidate_id=candidate_id(
                definition
            ),
            projection_mode=
                definition.projection_mode,
            timestep_cutoff=
                int(
                    definition.timestep_cutoff
                ),
            top_k=int(spec.top_k),
            target_gradient_ratio=float(
                definition.target_projected_ratio
            ),
            lambda_upper=float(lambda_upper),
            initial_model_sha256=
                initial_model_sha,
            diffusion_gradient_norm=
                diffusion_norm,
            geometry_gradient_norm=
                geometry_norm,
            projected_geometry_gradient_norm=
                projected_norm,
            pre_projection_cosine=float(
                gradient["pre_cosine"]
            ),
            post_projection_cosine=float(
                gradient["post_cosine"]
            ),
            projection_triggered_block_fraction=
                float(
                    gradient[
                        "projection_triggered_group_fraction"
                    ]
                ),
            removed_geometry_norm_fraction=
                float(
                    gradient[
                        "removed_norm_fraction"
                    ]
                ),
        )
        candidate.validate()
        candidates.append(candidate)

    payload = {
        "initial_model_sha256":
            initial_model_sha,
        "block_order":
            list(EXPECTED_BLOCK_ORDER),
        "raw_projection_records":
            dict(raw_records),
        "candidate_count":
            len(candidates),
        "candidate_order": [
            candidate.candidate_id
            for candidate in candidates
        ],
        "candidates": [
            candidate.to_dict()
            for candidate in candidates
        ],
    }
    del model
    return candidates, {
        **payload,
        "calibration_sha256":
            sha256_bytes(
                stable_json_bytes(payload)
            ),
    }


def _append_block_trigger_counts(
    counts: MutableMapping[str, int],
    projection: Mapping[str, Any],
) -> None:
    for block, record in projection[
        "per_group"
    ].items():
        counts.setdefault(block, 0)
        counts[block] += int(
            record["projection_triggered"]
        )


def train_projected_candidate(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    objective_contract: stagea.UpperObjectiveContract,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    candidate: ProjectedCandidate,
    calibration_batch: Mapping[str, Any],
    spec: ConflictProjectionSpec,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    torch, _ = stageb._torch_imports()
    candidate.validate()
    device = torch.device("cuda:0")
    model = stageb.make_model(
        stageb_spec
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stageb_spec.learning_rate,
        weight_decay=
            stageb_spec.weight_decay,
    )
    scheduler = stageb.scheduler_arrays(
        stageb_spec
    )
    alpha_bar = torch.as_tensor(
        scheduler["alpha_bar"],
        dtype=torch.float32,
        device=device,
    )
    condition_z = torch.as_tensor(
        condition_standardizer.normalize(
            condition
        ),
        dtype=torch.float32,
        device=device,
    )
    target_z = torch.as_tensor(
        target_standardizer.normalize(target),
        dtype=torch.float32,
        device=device,
    )
    initial_model_sha = (
        stageb.tensor_state_sha256(model)
    )
    initial_optimizer_sha = (
        stageb.optimizer_state_sha256(
            optimizer
        )
    )
    if (
        initial_model_sha
        != candidate.initial_model_sha256
    ):
        raise ConflictProjectionError(
            "candidate initial model SHA changed"
        )

    named_parameters = [
        (name, parameter)
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]
    parameter_block_map(named_parameters)
    parameters = [
        parameter
        for _name, parameter
        in named_parameters
    ]

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        stageb_spec.seed + 17
    )

    reported_loss_values: List[float] = []
    diffusion_values: List[float] = []
    geometry_values: List[float] = []
    gradient_values: List[float] = []
    clipping_values: List[bool] = []
    active_counts: List[int] = []
    active_fractions: List[float] = []
    projection_trigger_values: List[float] = []
    post_violation_values: List[float] = []
    removed_fraction_values: List[float] = []
    retained_fraction_values: List[float] = []
    combined_ratio_values: List[float] = []
    diffusion_change_values: List[int] = []
    violation_rate_values: List[float] = []
    maximum_excess_values: List[float] = []
    exposure = np.zeros(
        condition.shape[0],
        dtype=np.int64,
    )
    timestep_exposure = np.zeros(
        stageb_spec.train_timesteps,
        dtype=np.int64,
    )
    checkpoint_records: MutableMapping[
        str,
        Any,
    ] = {}
    block_trigger_counts: MutableMapping[
        str,
        int,
    ] = {}

    model.train()
    for step_index in range(
        stageb_spec.train_steps
    ):
        indices = torch.randint(
            low=0,
            high=condition.shape[0],
            size=(stageb_spec.batch_size,),
            generator=generator,
            device=device,
        )
        timestep = torch.randint(
            low=0,
            high=stageb_spec.train_timesteps,
            size=(stageb_spec.batch_size,),
            generator=generator,
            device=device,
        )
        noise = torch.randn(
            (
                stageb_spec.batch_size,
                stageb.FUTURE_STEPS,
                stageb.CABLE_DIM,
            ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        clean = target_z[indices]
        alpha = alpha_bar[timestep].reshape(
            -1,
            1,
            1,
        )
        noisy = (
            torch.sqrt(alpha) * clean
            + torch.sqrt(1.0 - alpha)
            * noise
        )
        optimizer.zero_grad(
            set_to_none=True
        )
        predicted = model(
            noisy,
            timestep,
            condition_z[indices],
        )
        diffusion_loss = torch.mean(
            (predicted - clean) ** 2
        )
        geometry = (
            staged.gated_topk_quadratic_terms_torch(
                predicted,
                timestep,
                target_standardizer=
                    target_standardizer,
                objective_contract=
                    objective_contract,
                top_k=candidate.top_k,
                timestep_cutoff=
                    candidate.timestep_cutoff,
            )
        )
        diffusion_gradients = (
            torch.autograd.grad(
                diffusion_loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        )
        geometry_gradients = (
            torch.autograd.grad(
                geometry["total"],
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        )
        combined_gradients, projection = (
            project_gradient_pair(
                named_parameters,
                diffusion_gradients,
                geometry_gradients,
                projection_mode=
                    candidate.projection_mode,
                lambda_upper=
                    candidate.lambda_upper,
                epsilon=
                    spec.projection_epsilon,
                post_dot_tolerance=
                    spec.post_projection_dot_tolerance,
                include_per_parameter=False,
            )
        )
        gradient = projection["global"]
        if (
            gradient[
                "post_projection_violation_count"
            ]
            != 0
        ):
            raise ConflictProjectionError(
                "post-projection block dot is negative"
            )
        if (
            gradient[
                "diffusion_gradient_unchanged"
            ]
            is not True
        ):
            raise ConflictProjectionError(
                "diffusion gradient changed"
            )
        for (
            (_name, parameter),
            combined,
        ) in zip(
            named_parameters,
            combined_gradients,
        ):
            parameter.grad = (
                combined.detach().clone()
            )
        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=10.0,
            )
        )
        optimizer.step()

        reported_loss = (
            diffusion_loss
            + candidate.lambda_upper
            * geometry["total"]
        )
        gradient_float = float(
            gradient_norm.detach().cpu()
        )
        active_count = int(
            geometry["active_count"]
            .detach()
            .cpu()
        )
        active_fraction = float(
            geometry["active_fraction"]
            .detach()
            .cpu()
        )
        reported_loss_values.append(
            float(
                reported_loss.detach().cpu()
            )
        )
        diffusion_values.append(
            float(
                diffusion_loss.detach().cpu()
            )
        )
        geometry_values.append(
            float(
                geometry["total"]
                .detach()
                .cpu()
            )
        )
        gradient_values.append(
            gradient_float
        )
        clipping_values.append(
            bool(gradient_float > 10.0)
        )
        active_counts.append(active_count)
        active_fractions.append(
            active_fraction
        )
        projection_trigger_values.append(
            float(
                gradient[
                    "projection_triggered_group_fraction"
                ]
            )
        )
        post_violation_values.append(
            float(
                gradient[
                    "post_projection_violation_count"
                ]
                > 0
            )
        )
        removed_fraction_values.append(
            float(
                gradient[
                    "removed_norm_fraction"
                ]
            )
        )
        retained_fraction_values.append(
            float(
                gradient[
                    "retained_norm_fraction"
                ]
            )
        )
        combined_ratio_values.append(
            float(
                gradient[
                    "combined_to_diffusion_ratio"
                ]
            )
        )
        diffusion_change_values.append(0)
        violation_rate_values.append(
            float(
                geometry[
                    "row_violation_rate"
                ]
                .detach()
                .cpu()
            )
        )
        maximum_excess_values.append(
            float(
                geometry["maximum_excess"]
                .detach()
                .cpu()
            )
        )
        _append_block_trigger_counts(
            block_trigger_counts,
            projection,
        )

        index_np = (
            indices.detach().cpu().numpy()
        )
        timestep_np = (
            timestep.detach().cpu().numpy()
        )
        np.add.at(
            exposure,
            index_np,
            1,
        )
        np.add.at(
            timestep_exposure,
            timestep_np,
            1,
        )

        completed_step = step_index + 1
        if (
            completed_step
            in spec.completed_step_checkpoints
        ):
            checkpoint_records[
                str(completed_step)
            ] = (
                fixed_batch_projected_gradients(
                    model=model,
                    calibration_batch=
                        calibration_batch,
                    target_standardizer=
                        target_standardizer,
                    objective_contract=
                        objective_contract,
                    top_k=candidate.top_k,
                    timestep_cutoff=
                        candidate.timestep_cutoff,
                    projection_mode=
                        candidate.projection_mode,
                    lambda_upper=
                        candidate.lambda_upper,
                    spec=spec,
                )
            )

    model.eval()
    reported_array = np.asarray(
        reported_loss_values,
        dtype=np.float64,
    )
    diffusion_array = np.asarray(
        diffusion_values,
        dtype=np.float64,
    )
    geometry_array = np.asarray(
        geometry_values,
        dtype=np.float64,
    )
    gradient_array = np.asarray(
        gradient_values,
        dtype=np.float64,
    )
    trigger_array = np.asarray(
        projection_trigger_values,
        dtype=np.float64,
    )
    post_violation_array = np.asarray(
        post_violation_values,
        dtype=np.float64,
    )
    removed_array = np.asarray(
        removed_fraction_values,
        dtype=np.float64,
    )
    retained_array = np.asarray(
        retained_fraction_values,
        dtype=np.float64,
    )
    combined_ratio_array = np.asarray(
        combined_ratio_values,
        dtype=np.float64,
    )
    diffusion_change_array = np.asarray(
        diffusion_change_values,
        dtype=np.int64,
    )
    exposure_array = np.asarray(
        exposure,
        dtype=np.int64,
    )
    timestep_exposure_array = np.asarray(
        timestep_exposure,
        dtype=np.int64,
    )
    active_count_array = np.asarray(
        active_counts,
        dtype=np.int64,
    )
    active_fraction_array = np.asarray(
        active_fractions,
        dtype=np.float64,
    )
    violation_rate_array = np.asarray(
        violation_rate_values,
        dtype=np.float64,
    )
    maximum_excess_array = np.asarray(
        maximum_excess_values,
        dtype=np.float64,
    )

    tail = slice(-100, None)
    expected_active_fraction = (
        float(candidate.timestep_cutoff + 1)
        / float(
            stageb_spec.train_timesteps
        )
    )
    total_projection_groups = (
        1
        if candidate.projection_mode == "global"
        else len(EXPECTED_BLOCK_ORDER)
    )
    training_record = {
        "candidate_id":
            candidate.candidate_id,
        "projection_mode":
            candidate.projection_mode,
        "lambda_upper":
            float(candidate.lambda_upper),
        "target_gradient_ratio":
            float(
                candidate.target_gradient_ratio
            ),
        "timestep_cutoff":
            int(candidate.timestep_cutoff),
        "top_k":
            int(candidate.top_k),
        "initial_model_sha256":
            initial_model_sha,
        "final_model_sha256":
            stageb.tensor_state_sha256(model),
        "initial_optimizer_sha256":
            initial_optimizer_sha,
        "final_optimizer_sha256":
            stageb.optimizer_state_sha256(
                optimizer
            ),
        "reported_loss_history_sha256":
            sha256_array(reported_array),
        "diffusion_loss_history_sha256":
            sha256_array(diffusion_array),
        "geometry_loss_history_sha256":
            sha256_array(geometry_array),
        "gradient_history_sha256":
            sha256_array(gradient_array),
        "projection_trigger_history_sha256":
            sha256_array(trigger_array),
        "post_projection_violation_history_sha256":
            sha256_array(
                post_violation_array
            ),
        "removed_norm_fraction_history_sha256":
            sha256_array(removed_array),
        "retained_norm_fraction_history_sha256":
            sha256_array(retained_array),
        "combined_ratio_history_sha256":
            sha256_array(
                combined_ratio_array
            ),
        "diffusion_change_history_sha256":
            sha256_array(
                diffusion_change_array
            ),
        "source_exposure_sha256":
            sha256_array(exposure_array),
        "timestep_exposure_sha256":
            sha256_array(
                timestep_exposure_array
            ),
        "active_count_history_sha256":
            sha256_array(active_count_array),
        "active_fraction_history_sha256":
            sha256_array(
                active_fraction_array
            ),
        "reported_loss_first":
            float(reported_array[0]),
        "reported_loss_final":
            float(reported_array[-1]),
        "reported_loss_tail_mean":
            float(
                np.mean(reported_array[tail])
            ),
        "diffusion_loss_tail_mean":
            float(
                np.mean(diffusion_array[tail])
            ),
        "geometry_loss_tail_mean":
            float(
                np.mean(geometry_array[tail])
            ),
        "gradient_tail_mean":
            float(
                np.mean(gradient_array[tail])
            ),
        "projection_trigger_fraction_tail_mean":
            float(
                np.mean(trigger_array[tail])
            ),
        "removed_norm_fraction_tail_mean":
            float(
                np.mean(removed_array[tail])
            ),
        "retained_norm_fraction_tail_mean":
            float(
                np.mean(retained_array[tail])
            ),
        "combined_to_diffusion_ratio_tail_mean":
            float(
                np.mean(
                    combined_ratio_array[tail]
                )
            ),
        "batch_row_violation_rate_tail_mean":
            float(
                np.mean(
                    violation_rate_array[tail]
                )
            ),
        "maximum_log_excess_tail_max":
            float(
                np.max(
                    maximum_excess_array[tail]
                )
            ),
        "loss_finite":
            bool(
                np.all(
                    np.isfinite(reported_array)
                )
                and np.all(
                    np.isfinite(
                        diffusion_array
                    )
                )
                and np.all(
                    np.isfinite(
                        geometry_array
                    )
                )
            ),
        "gradient_finite":
            bool(
                np.all(
                    np.isfinite(gradient_array)
                )
            ),
        "projection_finite":
            bool(
                np.all(
                    np.isfinite(trigger_array)
                )
                and np.all(
                    np.isfinite(removed_array)
                )
                and np.all(
                    np.isfinite(
                        combined_ratio_array
                    )
                )
            ),
        "training_rows":
            int(condition.shape[0]),
        "training_steps":
            int(stageb_spec.train_steps),
    }
    diagnostics = {
        "projection_mode":
            candidate.projection_mode,
        "block_order":
            list(EXPECTED_BLOCK_ORDER),
        "projection_group_count":
            int(total_projection_groups),
        "checkpoint_records":
            dict(checkpoint_records),
        "gradient_clip_frequency":
            float(
                np.mean(
                    np.asarray(
                        clipping_values,
                        dtype=np.float64,
                    )
                )
            ),
        "projection_trigger_frequency":
            float(
                np.mean(trigger_array > 0.0)
            ),
        "mean_triggered_group_fraction":
            float(np.mean(trigger_array)),
        "post_projection_violation_frequency":
            float(
                np.mean(
                    post_violation_array
                )
            ),
        "diffusion_gradient_change_count":
            int(
                np.sum(
                    diffusion_change_array
                )
            ),
        "removed_norm_fraction_mean":
            float(np.mean(removed_array)),
        "retained_norm_fraction_mean":
            float(np.mean(retained_array)),
        "combined_to_diffusion_ratio_mean":
            float(
                np.mean(
                    combined_ratio_array
                )
            ),
        "per_block_projection_trigger_frequency": {
            block: (
                float(count)
                / float(stageb_spec.train_steps)
            )
            for block, count
            in sorted(
                block_trigger_counts.items()
            )
        },
        "expected_gate_active_fraction":
            expected_active_fraction,
        "observed_global_active_fraction":
            float(
                np.sum(active_count_array)
                / (
                    stageb_spec.train_steps
                    * stageb_spec.batch_size
                )
            ),
        "active_count_stats":
            stagec_topk.safe_stats(
                active_count_array.astype(
                    np.float64
                )
            ),
        "active_fraction_stats":
            stagec_topk.safe_stats(
                active_fraction_array
            ),
    }
    return model, training_record, diagnostics


def final_checkpoint(
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    records = diagnostics.get(
        "checkpoint_records"
    )
    if not isinstance(records, Mapping):
        raise ConflictProjectionError(
            "checkpoint records are missing"
        )
    key = str(
        stageb.DiagnosticSpec().train_steps
    )
    if key not in records:
        raise ConflictProjectionError(
            "final checkpoint is missing"
        )
    value = records[key]
    if not isinstance(value, Mapping):
        raise ConflictProjectionError(
            "final checkpoint is malformed"
        )
    return value


def projection_contract_gates(
    record: Mapping[str, Any],
    *,
    spec: ConflictProjectionSpec,
) -> Dict[str, bool]:
    diagnostics = record[
        "training_diagnostics"
    ]
    checkpoint = final_checkpoint(
        diagnostics
    )
    return {
        "post_projection_training":
            float(
                diagnostics[
                    "post_projection_violation_frequency"
                ]
            )
            <= spec.post_projection_violation_frequency_max,
        "diffusion_gradient_training":
            int(
                diagnostics[
                    "diffusion_gradient_change_count"
                ]
            )
            <= spec.diffusion_gradient_change_count_max,
        "post_projection_checkpoint": (
            int(
                checkpoint["gradient"][
                    "post_projection_violation_count"
                ]
            )
            == 0
        ),
        "diffusion_gradient_checkpoint": (
            checkpoint["gradient"][
                "diffusion_gradient_unchanged"
            ]
            is True
        ),
        "projected_gradient_finite": (
            record["training"][
                "projection_finite"
            ]
            is True
        ),
    }


def make_selection_record(
    *,
    candidate: ProjectedCandidate,
    training: Mapping[str, Any],
    training_diagnostics: Mapping[str, Any],
    train_control: Mapping[str, Any],
    one_step: Mapping[str, Any],
    profiles: Mapping[str, Any],
    control: Mapping[str, Any],
    spec: ConflictProjectionSpec,
) -> Dict[str, Any]:
    decision_spec = staged.TimestepGateSpec()
    base = staged.selection_record(
        candidate=candidate,
        training=training,
        training_diagnostics=
            training_diagnostics,
        train_control=train_control,
        one_step=one_step,
        profiles=profiles,
        control=control,
        spec=decision_spec,
    )
    projection_gates = (
        projection_contract_gates(
            base,
            spec=spec,
        )
    )
    is_primary = (
        candidate.projection_mode
        == "blockwise"
    )
    eligible = bool(
        base["eligible_for_selection"]
        and all(projection_gates.values())
        and is_primary
    )
    return {
        **base,
        "projection_contract_gates":
            projection_gates,
        "projection_contract_pass":
            bool(
                all(
                    projection_gates.values()
                )
            ),
        "primary_mechanism_candidate":
            is_primary,
        "negative_control":
            bool(
                candidate.projection_mode
                == "global"
            ),
        "eligible_for_selection":
            eligible,
    }


def select_configuration(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    eligible = [
        record
        for record in records
        if record.get(
            "eligible_for_selection"
        )
        is True
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda record: (
            -float(
                record["binary_rate"]["10"]
            ),
            float(
                record["holdout_profiles"]["10"][
                    "row_best_max_log_excess"
                ]["mean"]
            ),
            float(
                record["relative_to_control"][
                    "train_control_nmse_ratio"
                ]
            ),
            int(
                record["candidate"][
                    "timestep_cutoff"
                ]
            ),
            float(
                record["candidate"][
                    "target_gradient_ratio"
                ]
            ),
            str(record["candidate_id"]),
        ),
    )[0]
    return {
        "candidate_id":
            selected["candidate_id"],
        "projection_mode":
            selected["candidate"][
                "projection_mode"
            ],
        "timestep_cutoff":
            int(
                selected["candidate"][
                    "timestep_cutoff"
                ]
            ),
        "top_k":
            int(
                selected["candidate"][
                    "top_k"
                ]
            ),
        "target_gradient_ratio":
            float(
                selected["candidate"][
                    "target_gradient_ratio"
                ]
            ),
        "lambda_upper":
            float(
                selected["candidate"][
                    "lambda_upper"
                ]
            ),
        "pilot_model_sha256":
            selected["training"][
                "final_model_sha256"
            ],
        "selection_rule": (
            "blockwise candidates only; all geometry, "
            "fidelity, binary, stability and projection "
            "contracts pass; then highest t10 binary rate, "
            "lowest t10 excess, lowest fidelity ratio, "
            "narrowest cutoff and lowest projected ratio"
        ),
        "selection_record_sha256":
            sha256_bytes(
                stable_json_bytes(selected)
            ),
    }


def _best_by_t10(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return max(
        records,
        key=lambda record: (
            float(
                record["continuous_response"][
                    "t10_mean_excess_reduction"
                ]
            ),
            -float(
                record["relative_to_control"][
                    "train_control_nmse_ratio"
                ]
            ),
            str(record["candidate_id"]),
        ),
    )


def classify_selection(
    *,
    records: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]],
    spec: ConflictProjectionSpec,
) -> Dict[str, Any]:
    if not records:
        raise ValueError(
            "candidate records are empty"
        )
    global_records = [
        record
        for record in records
        if record["candidate"][
            "projection_mode"
        ]
        == "global"
    ]
    block_records = [
        record
        for record in records
        if record["candidate"][
            "projection_mode"
        ]
        == "blockwise"
    ]
    if len(global_records) != 1:
        raise ConflictProjectionError(
            "global negative-control population changed"
        )
    if len(block_records) != 4:
        raise ConflictProjectionError(
            "blockwise candidate population changed"
        )
    global_record = global_records[0]
    best_block = _best_by_t10(
        block_records
    )
    blockwise_advantage = (
        float(
            best_block[
                "continuous_response"
            ][
                "t10_mean_excess_reduction"
            ]
        )
        - float(
            global_record[
                "continuous_response"
            ][
                "t10_mean_excess_reduction"
            ]
        )
    )

    if selected is not None:
        root = (
            "phase314b_r258_stagea_conflict_projected_"
            "k16_train_only_configuration_selected"
        )
        next_path = (
            "RETRAIN_SELECTED_CONFLICT_PROJECTED_K16_ON_"
            "FULL_STAGEB_TRAIN_AND_OPEN_FROZEN_PROBE"
        )
        locus = "none"
    else:
        continuous = [
            record
            for record in block_records
            if record[
                "continuous_geometry_pass"
            ]
        ]
        fidelity = [
            record
            for record in block_records
            if record["fidelity_pass"]
        ]
        continuous_fidelity = [
            record
            for record in block_records
            if (
                record[
                    "continuous_geometry_pass"
                ]
                and record["fidelity_pass"]
            )
        ]
        stable = [
            record
            for record in continuous_fidelity
            if (
                record["stability_pass"]
                and record[
                    "projection_contract_pass"
                ]
            )
        ]

        trigger_frequency = float(
            best_block[
                "training_diagnostics"
            ][
                "projection_trigger_frequency"
            ]
        )
        removed_fraction = float(
            best_block[
                "training_diagnostics"
            ][
                "removed_norm_fraction_mean"
            ]
        )
        if (
            trigger_frequency
            <= spec.inactive_projection_trigger_frequency_max
            and removed_fraction
            <= spec.inactive_removed_norm_fraction_max
        ):
            root = (
                "phase314b_r258_stagea_blockwise_"
                "conflict_projection_inactive"
            )
            next_path = (
                "AUDIT_BLOCKWISE_GRADIENT_CONFLICT_"
                "LOCALIZATION_BEFORE_FURTHER_CALIBRATION"
            )
            locus = "projection_inactive"
        elif not continuous and not fidelity:
            root = (
                "phase314b_r258_stagea_conflict_projected_"
                "k16_no_geometry_or_fidelity_solution"
            )
            next_path = (
                "AUDIT_DIRECT_X0_MODEL_PARAMETERIZATION_AND_"
                "UPPER_GEOMETRY_TARGET_REACHABILITY"
            )
            locus = "joint_tradeoff"
        elif fidelity and not continuous:
            root = (
                "phase314b_r258_stagea_conflict_projection_"
                "restored_fidelity_but_geometry_underpowered"
            )
            next_path = (
                "CALIBRATE_CONFLICT_PROJECTED_K16_CURRICULUM_"
                "ON_TRAIN_ONLY_SPLIT"
            )
            locus = "geometry_underpowered"
        elif continuous and not continuous_fidelity:
            root = (
                "phase314b_r258_stagea_conflict_projected_"
                "geometry_response_with_persistent_fidelity_tradeoff"
            )
            next_path = (
                "CALIBRATE_DIFFUSION_TRUST_REGION_PROJECTED_"
                "K16_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
            )
            locus = "fidelity_tradeoff"
        elif continuous_fidelity and not stable:
            root = (
                "phase314b_r258_stagea_conflict_projected_"
                "projection_or_outlier_instability"
            )
            next_path = (
                "CALIBRATE_CLIPPED_OR_TRUST_REGION_PROJECTED_"
                "K16_OBJECTIVE_ON_TRAIN_ONLY_SPLIT"
            )
            locus = "stability"
        else:
            root = (
                "phase314b_r258_stagea_conflict_projected_"
                "continuous_and_fidelity_response_without_gate_crossing"
            )
            next_path = (
                "CALIBRATE_PROJECTED_K16_CURRICULUM_FOR_"
                "BINARY_GATE_CROSSING_ON_TRAIN_ONLY_SPLIT"
            )
            locus = "binary_gate_crossing"

    best_fidelity = min(
        block_records,
        key=lambda record: (
            float(
                record["relative_to_control"][
                    "train_control_nmse_ratio"
                ]
            ),
            -float(
                record["continuous_response"][
                    "t10_mean_excess_reduction"
                ]
            ),
            str(record["candidate_id"]),
        ),
    )
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "global_negative_control":
            global_record["candidate_id"],
        "best_blockwise_candidate":
            best_block["candidate_id"],
        "blockwise_advantage_over_global_t10":
            float(blockwise_advantage),
        "best_t10_mean_excess_reduction":
            float(
                best_block[
                    "continuous_response"
                ][
                    "t10_mean_excess_reduction"
                ]
            ),
        "best_t10_binary_upper_row_any":
            float(
                best_block[
                    "binary_rate"
                ]["10"]
            ),
        "best_fidelity_candidate":
            best_fidelity["candidate_id"],
        "best_train_control_nmse_ratio":
            float(
                best_fidelity[
                    "relative_to_control"
                ][
                    "train_control_nmse_ratio"
                ]
            ),
        "selected_configuration":
            selected,
    }


# The hardware-portable validate_environment_payload implementation is
# defined after the portable compatibility helpers at the end of this module.


def run_calibration(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[
        ConflictProjectionSpec
    ] = None,
) -> Dict[str, Any]:
    active_spec = (
        ConflictProjectionSpec()
        if spec is None
        else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_resume4_evidence(
        repository_root
    )
    validate_environment_payload(
        environment
    )
    logic = source_logic_audit(
        repository_root
    )
    if not logic["all_confirmed"]:
        raise ConflictProjectionError(
            "source assumptions changed"
        )

    cold_context = assert_cold_cuda_context_portable()
    captured = replay_stagec_control_portable(
        root=repository_root,
        equivalence_spec=PortableNumericalEquivalenceSpec(),
    )
    control_bundle = captured["control_bundle"]
    control_identity = captured["control_identity"]
    if control_identity["reference_equivalence_pass"] is not True:
        raise ConflictProjectionError(
            "current Stage-C control is not numerically equivalent "
            "to the frozen reference"
        )
    if (
        control_identity[
            "stagec_nonzero_candidate_training_count"
        ]
        != 0
    ):
        raise ConflictProjectionError(
            "portable Stage-C replay ran a nonzero candidate"
        )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    upstream = (
        stagec_topk.validate_immutable_inputs(
            repository_root
        )
    )
    objective_contract = (
        stageb_mechanism.load_objective_contract(
            upstream[
                "upstream_immutable"
            ]["stagea_contract"]
        )
    )
    upper_gate = (
        stagea.load_upper_gate_contract(
            upstream[
                "upstream_immutable"
            ][
                "upstream_immutable"
            ]["staged3_contract"]
        )
    )
    stage_d_contract, _ = (
        staged1.load_stage_d_gate(
            repository_root
            / staged3.STAGE_D_GATE
        )
    )

    arrays = stageb.load_npz_strict(
        repository_root
        / stageb.STAGEA_TRAIN_VIEW,
        required_keys=
            stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = (
        stageb.validate_train_view(arrays)
    )
    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(
        arrays["episode_group_key"]
    ).astype(str)
    condition_name = np.asarray(
        arrays["condition_name"]
    ).astype(str)

    stageb_train, frozen_probe, _ = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=
                stageb_spec.probe_fold,
        )
    )
    objective_train, selection_holdout, split = (
        stagea.deterministic_selection_split(
            groups,
            stageb_train,
            folds=(
                stagea.UpperObjectiveSpec()
                .selection_group_folds
            ),
            holdout_fold=(
                stagea.UpperObjectiveSpec()
                .selection_holdout_fold
            ),
        )
    )
    resume4_split = immutable[
        "worker_result"
    ]["split"]
    for key in (
        "objective_train_rows",
        "selection_holdout_rows",
        "objective_train_groups",
        "selection_holdout_groups",
        "objective_train_group_sha256",
        "selection_holdout_group_sha256",
        "group_overlap",
        "row_overlap",
    ):
        if split[key] != resume4_split[key]:
            raise ConflictProjectionError(
                "train-only split changed: "
                "{}".format(key)
            )
    if np.any(
        frozen_probe
        & (
            objective_train
            | selection_holdout
        )
    ):
        raise AssertionError(
            "frozen probe crossed train-only split"
        )

    condition_standardizer = (
        stageb.fit_standardizer(
            condition[objective_train]
        )
    )
    target_standardizer = (
        stageb.fit_standardizer(
            target[objective_train]
        )
    )
    calibration_batch = (
        staged.stratified_calibration_batch(
            condition=
                condition[objective_train],
            target=
                target[objective_train],
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
            stageb_spec=stageb_spec,
            spec=active_spec,
        )
    )
    candidates, calibration = (
        calibrate_candidates(
            stageb_spec=stageb_spec,
            calibration_batch=
                calibration_batch,
            target_standardizer=
                target_standardizer,
            objective_contract=
                objective_contract,
            spec=active_spec,
        )
    )

    records: List[Dict[str, Any]] = []
    for candidate in candidates:
        stageb.set_deterministic_runtime(
            stageb_spec.seed
        )
        model, training, diagnostics = (
            train_projected_candidate(
                condition=
                    condition[objective_train],
                target=
                    target[objective_train],
                stageb_spec=stageb_spec,
                objective_contract=
                    objective_contract,
                condition_standardizer=
                    condition_standardizer,
                target_standardizer=
                    target_standardizer,
                candidate=candidate,
                calibration_batch=
                    calibration_batch,
                spec=active_spec,
            )
        )
        train_control = (
            stagea.train_control_audit(
                model=model,
                condition=
                    condition[objective_train],
                target=
                    target[objective_train],
                stageb_spec=stageb_spec,
                condition_standardizer=
                    condition_standardizer,
                target_standardizer=
                    target_standardizer,
            )
        )
        one_step = stagea.one_step_audit(
            model=model,
            condition=
                condition[selection_holdout],
            target=
                target[selection_holdout],
            groups=
                groups[selection_holdout],
            condition_name=
                condition_name[
                    selection_holdout
                ],
            stageb_spec=stageb_spec,
            upper_gate=upper_gate,
            stage_d_contract=
                stage_d_contract,
            historical_geometry=
                stageb.fit_geometry_contract(
                    target[stageb_train]
                ),
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
            noise_seed=(
                stageb_spec.seed
                + active_spec
                .holdout_noise_seed_offset
            ),
        )
        predictions, prediction_sha = (
            stagec.one_step_predictions(
                model=model,
                condition=
                    condition[
                        selection_holdout
                    ],
                target=
                    target[
                        selection_holdout
                    ],
                spec=stageb_spec,
                condition_standardizer=
                    condition_standardizer,
                target_standardizer=
                    target_standardizer,
                noise_seed=(
                    stageb_spec.seed
                    + active_spec
                    .holdout_noise_seed_offset
                ),
            )
        )
        if (
            prediction_sha
            != one_step[
                "prediction_sha256"
            ]
        ):
            raise ConflictProjectionError(
                "candidate prediction SHA differs"
            )
        profiles = (
            stagec_topk.holdout_profiles(
                predictions,
                upper_gate=upper_gate,
                objective_contract=
                    objective_contract,
            )
        )
        records.append(
            make_selection_record(
                candidate=candidate,
                training=training,
                training_diagnostics=
                    diagnostics,
                train_control=train_control,
                one_step=one_step,
                profiles=profiles,
                control=control_bundle,
                spec=active_spec,
            )
        )
        del model

    selected = select_configuration(
        records
    )
    classification = (
        classify_selection(
            records=records,
            selected=selected,
            spec=active_spec,
        )
    )
    selection_payload = {
        "control": control_bundle,
        "candidate_records": records,
        "selected_configuration":
            selected,
        "selection_uses_frozen_probe":
            False,
        "selection_rule": (
            "global projection is a negative control; "
            "only blockwise candidates may be selected; "
            "all frozen train-only geometry, fidelity, "
            "binary, stability and projection gates apply"
        ),
    }
    selection_sha = sha256_bytes(
        stable_json_bytes(
            selection_payload
        )
    )

    contract = {
        "schema":
            "phase314b_r258_stagea_portable_conflict_projection_contract_v2",
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "base_resume4_identity": {
            "worker_sha256":
                EXPECTED_RESUME4_WORKER_SHA256,
            "environment_sha256":
                EXPECTED_RESUME4_ENVIRONMENT_SHA256,
            "stagec_calibration_sha256":
                EXPECTED_STAGEC_CALIBRATION_SHA256,
            "stagec_control_sha256":
                EXPECTED_STAGEC_CONTROL_SHA256,
            "selection_sha256":
                EXPECTED_RESUME4_SELECTION_SHA256,
            "contract_sha256":
                EXPECTED_RESUME4_CONTRACT_SHA256,
            "contract_file_sha256":
                EXPECTED_RESUME4_CONTRACT_FILE_SHA256,
        },
        "environment": dict(environment),
        "hardware_identity_is_observation_only": True,
        "gpu_model_restricted": False,
        "portable_numerical_equivalence_spec":
            asdict(PortableNumericalEquivalenceSpec()),
        "cold_main_worker_context":
            cold_context,
        "control_capture":
            control_identity,
        "objective_variant": (
            "hard_low_noise_k16_quadratic_with_"
            "global_or_blockwise_conflict_projection"
        ),
        "model_parameterization_changed":
            False,
        "geometry_target_changed":
            False,
        "optimizer_update_rule_changed":
            True,
        "block_order":
            list(EXPECTED_BLOCK_ORDER),
        "candidate_definitions": [
            asdict(definition)
            for definition
            in active_spec.candidate_definitions
        ],
        "projection_rule": (
            "if group dot(g_geometry,g_diffusion)<0, "
            "g_geometry <- g_geometry - "
            "dot/||g_diffusion||^2 * g_diffusion"
        ),
        "diffusion_gradient_mutation":
            "forbidden",
        "lambda_calibration": (
            "lambda = target_ratio * ||g_diffusion|| / "
            "||projected_g_geometry|| at the common "
            "initial model on a deterministic t=0..99 batch"
        ),
        "calibration_batch_sha256": {
            key: value
            for key, value
            in calibration_batch.items()
            if key.endswith("_sha256")
        },
        "calibration": calibration,
        "selection_thresholds": {
            key: value
            for key, value
            in asdict(active_spec).items()
            if (
                key.endswith("_min")
                or key.endswith("_max")
                or key.endswith("_tolerance")
            )
        },
        "uses_frozen_probe": False,
        "runs_reverse_sampling": False,
        "trains_full_stageb_model":
            False,
        "global_projection_is_negative_control":
            True,
    }
    contract["contract_sha256"] = (
        sha256_bytes(
            stable_json_bytes(contract)
        )
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema":
            "phase314b_r258_stagea_portable_worker_result_v2",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause":
            classification["root_cause"],
        "required_next_path":
            classification[
                "required_next_path"
            ],
        "projection_spec":
            asdict(active_spec),
        "immutable_inputs": {
            "base_evidence_commit":
                BASE_EVIDENCE_COMMIT,
            "resume4_worker_sha256":
                immutable[
                    "worker_identity_sha256"
                ],
            "resume4_environment_sha256":
                immutable[
                    "environment_sha256"
                ],
            "resume4_selection_sha256":
                immutable[
                    "selection_sha256"
                ],
            "resume4_contract_sha256":
                immutable[
                    "contract_sha256"
                ],
            "resume4_contract_file_sha256":
                immutable[
                    "contract_file_sha256"
                ],
        },
        "source_logic_audit": logic,
        "environment": dict(environment),
        "hardware_identity_is_observation_only": True,
        "gpu_model_restricted": False,
        "portable_numerical_equivalence_spec":
            asdict(PortableNumericalEquivalenceSpec()),
        "cold_main_worker_context":
            cold_context,
        "control_capture":
            control_identity,
        "train_view_validation":
            validation,
        "split": {
            "stageb_training_rows":
                int(np.sum(stageb_train)),
            "objective_train_rows":
                int(np.sum(objective_train)),
            "selection_holdout_rows":
                int(
                    np.sum(
                        selection_holdout
                    )
                ),
            "frozen_probe_rows":
                int(np.sum(frozen_probe)),
            **split,
            "frozen_probe_accessed":
                False,
        },
        "calibration_contract":
            contract,
        "selection": {
            **selection_payload,
            "selection_sha256":
                selection_sha,
        },
        "classification":
            classification,
        "selected_configuration":
            selected,
        "train_only_recommendation":
            selected,
        "new_hyperparameter_candidate_run":
            True,
        "new_objective_variant_run":
            True,
        "control_replay_exact":
            bool(control_identity["byte_exact_to_reference"]),
        "control_reference_equivalent": True,
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
        "candidate_tensor_persisted":
            False,
        "npz_saved": False,
        "cache_saved": False,
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
        "phase4": False,
        "cps": False,
    }


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause":
            result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "environment":
            result["environment"],
        "cold_main_worker_context":
            result[
                "cold_main_worker_context"
            ],
        "control_capture":
            result["control_capture"],
        "split": result["split"],
        "calibration_contract":
            result["calibration_contract"],
        "selection": result["selection"],
        "classification":
            result["classification"],
        "selected_configuration":
            result[
                "selected_configuration"
            ],
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
        "calibration_contract_exact": (
            left["calibration_contract"]
            == right[
                "calibration_contract"
            ]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
    }

# ---------------------------------------------------------------------------
# Hardware-portable execution contract (revised before first repository run)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PortableEnvironmentSpec:
    python_version: Tuple[int, int, int] = (3, 9, 15)
    numpy_version: str = "1.23.3"
    torch_version: str = "1.12.1.post200"
    torch_cuda_version: str = "11.2"
    cublas_workspace_config: str = ":4096:8"
    python_hash_seed: str = "0"
    required_operation_schema: str = (
        "r258_stagea_forward_two_grad_block_projection_"
        "optimizer_step_and_100row_checkpoint_v1"
    )
    dry_run_batch_size: int = 64
    dry_run_cutoff: int = 50
    dry_run_top_k: int = 16
    dry_run_lambda: float = 0.10

    def validate(self) -> None:
        if self.python_version != (3, 9, 15):
            raise ValueError("portable Python contract changed")
        if self.numpy_version != "1.23.3":
            raise ValueError("portable NumPy contract changed")
        if self.torch_version != "1.12.1.post200":
            raise ValueError("portable PyTorch contract changed")
        if self.torch_cuda_version != "11.2":
            raise ValueError("portable Torch-CUDA contract changed")
        if self.cublas_workspace_config != ":4096:8":
            raise ValueError("portable cuBLAS workspace contract changed")
        if self.python_hash_seed != "0":
            raise ValueError("portable hash-seed contract changed")
        if self.dry_run_batch_size != 64:
            raise ValueError("portable dry-run batch changed")
        if self.dry_run_cutoff != 50 or self.dry_run_top_k != 16:
            raise ValueError("portable dry-run objective changed")
        if not 0.0 < self.dry_run_lambda <= 0.50:
            raise ValueError("portable dry-run lambda is invalid")


@dataclass(frozen=True)
class PortableNumericalEquivalenceSpec:
    """Pre-registered reference tolerances for cross-hardware replay.

    These tolerances are applied only to the frozen Stage-C calibration and
    diffusion-only control replay.  Candidate selection thresholds are not
    relaxed.  Dictionary structure, strings, booleans, integer counts, initial
    model identity, initial optimizer identity, and source-exposure identity
    remain exact.
    """

    calibration_rtol: float = 5.0e-3
    calibration_atol: float = 1.0e-6
    control_rtol: float = 1.0e-2
    control_atol: float = 1.0e-6
    maximum_difference_records: int = 256

    def validate(self) -> None:
        for value in (
            self.calibration_rtol,
            self.calibration_atol,
            self.control_rtol,
            self.control_atol,
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("portable numerical tolerance is invalid")
        if not 1 <= int(self.maximum_difference_records) <= 4096:
            raise ValueError("portable difference-record limit is invalid")


PORTABLE_EXACT_SHA_KEYS = frozenset(
    {
        "initial_model_sha256",
        "initial_optimizer_sha256",
        "source_exposure_sha256",
    }
)

PORTABLE_IGNORED_SHA_KEYS = frozenset(
    {
        "final_model_sha256",
        "final_optimizer_sha256",
        "total_loss_history_sha256",
        "reported_loss_history_sha256",
        "diffusion_loss_history_sha256",
        "geometry_loss_history_sha256",
        "gradient_history_sha256",
        "checkpoint_record_sha256",
        "prediction_sha256",
        "predicted_x0_sha256",
        "selected_excess_sha256",
        "selected_index_sha256",
        "row_loss_sha256",
        "calibration_sha256",
        "control_record_sha256",
        "control_training_sha256",
        "control_model_sha256",
        "control_optimizer_sha256",
        "control_loss_sha256",
        "control_gradient_sha256",
        "control_exposure_sha256",
    }
)


def _nvidia_smi_observation() -> Dict[str, Any]:
    query = (
        "name,uuid,pci.bus_id,driver_version,memory.total"
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=" + query,
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
        }
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        return {
            "available": False,
            "error_type": "unexpected_row_count",
            "row_count": len(rows),
        }
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 5:
        return {
            "available": False,
            "error_type": "unexpected_column_count",
            "column_count": len(parts),
        }
    return {
        "available": True,
        "name": parts[0],
        "uuid": parts[1],
        "pci_bus_id": parts[2],
        "driver_version": parts[3],
        "memory_total_mib": parts[4],
    }


def _portable_objective_context(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    upstream = stagec_topk.validate_immutable_inputs(repository_root)
    objective_contract = stageb_mechanism.load_objective_contract(
        upstream["upstream_immutable"]["stagea_contract"]
    )
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(arrays["episode_group_key"]).astype(str)
    stageb_train, frozen_probe, _ = stageb.deterministic_group_split(
        groups,
        folds=stageb_spec.group_folds,
        probe_fold=stageb_spec.probe_fold,
    )
    objective_train, selection_holdout, split = (
        stagea.deterministic_selection_split(
            groups,
            stageb_train,
            folds=stagea.UpperObjectiveSpec().selection_group_folds,
            holdout_fold=stagea.UpperObjectiveSpec().selection_holdout_fold,
        )
    )
    if np.any(frozen_probe & (objective_train | selection_holdout)):
        raise ConflictProjectionError(
            "portable context crossed the frozen probe"
        )
    condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )
    return {
        "repository_root": repository_root,
        "stageb_spec": stageb_spec,
        "objective_contract": objective_contract,
        "condition": condition,
        "target": target,
        "groups": groups,
        "stageb_train": stageb_train,
        "objective_train": objective_train,
        "selection_holdout": selection_holdout,
        "frozen_probe": frozen_probe,
        "split": split,
        "condition_standardizer": condition_standardizer,
        "target_standardizer": target_standardizer,
    }


def required_operation_dry_run(
    *,
    root: Path,
    environment_spec: Optional[PortableEnvironmentSpec] = None,
) -> Dict[str, Any]:
    """Exercise the exact operation family required by Stage A.

    This runs in a disposable child process.  Passing the dry run replaces
    fixed GPU-family and fixed-memory requirements.
    """
    active = (
        PortableEnvironmentSpec()
        if environment_spec is None
        else environment_spec
    )
    active.validate()
    torch, _ = stageb._torch_imports()
    context = _portable_objective_context(root)
    stageb_spec = context["stageb_spec"]
    condition = context["condition"][context["objective_train"]]
    target = context["target"][context["objective_train"]]
    condition_standardizer = context["condition_standardizer"]
    target_standardizer = context["target_standardizer"]
    objective_contract = context["objective_contract"]

    stageb.set_deterministic_runtime(stageb_spec.seed)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    model = stageb.make_model(stageb_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stageb_spec.learning_rate,
        weight_decay=stageb_spec.weight_decay,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameter_block_map(named_parameters)
    parameters = [parameter for _name, parameter in named_parameters]

    rows = int(active.dry_run_batch_size)
    condition_z = torch.as_tensor(
        condition_standardizer.normalize(condition[:rows]),
        dtype=torch.float32,
        device=device,
    )
    clean = torch.as_tensor(
        target_standardizer.normalize(target[:rows]),
        dtype=torch.float32,
        device=device,
    )
    timesteps_np = np.arange(rows, dtype=np.int64) % stageb_spec.train_timesteps
    noise_np = stageb._fixed_noise(
        target[:rows].shape,
        seed=stageb_spec.seed + 9901,
    )
    scheduler = stageb.scheduler_arrays(stageb_spec)
    noisy_np = stageb.q_sample_numpy(
        target_standardizer.normalize(target[:rows]),
        noise_np,
        timesteps_np,
        scheduler["alpha_bar"],
    )
    noisy = torch.as_tensor(
        noisy_np,
        dtype=torch.float32,
        device=device,
    )
    timesteps = torch.as_tensor(
        timesteps_np,
        dtype=torch.long,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    predicted = model(noisy, timesteps, condition_z)
    diffusion_loss = torch.mean((predicted - clean) ** 2)
    geometry = staged.gated_topk_quadratic_terms_torch(
        predicted,
        timesteps,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        top_k=active.dry_run_top_k,
        timestep_cutoff=active.dry_run_cutoff,
    )
    diffusion_gradients = torch.autograd.grad(
        diffusion_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    geometry_gradients = torch.autograd.grad(
        geometry["total"],
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    combined, projection = project_gradient_pair(
        named_parameters,
        diffusion_gradients,
        geometry_gradients,
        projection_mode="blockwise",
        lambda_upper=active.dry_run_lambda,
        epsilon=ConflictProjectionSpec().projection_epsilon,
        post_dot_tolerance=(
            ConflictProjectionSpec().post_projection_dot_tolerance
        ),
        include_per_parameter=False,
    )
    for (_name, parameter), gradient in zip(named_parameters, combined):
        parameter.grad = gradient.detach().clone()
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    optimizer.step()

    calibration_batch = staged.stratified_calibration_batch(
        condition=condition,
        target=target,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=ConflictProjectionSpec(),
    )
    checkpoint = fixed_batch_projected_gradients(
        model=model,
        calibration_batch=calibration_batch,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        top_k=active.dry_run_top_k,
        timestep_cutoff=active.dry_run_cutoff,
        projection_mode="blockwise",
        lambda_upper=active.dry_run_lambda,
        spec=ConflictProjectionSpec(),
    )
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finite = bool(
        np.isfinite(float(diffusion_loss.detach().cpu()))
        and np.isfinite(float(geometry["total"].detach().cpu()))
        and np.isfinite(float(gradient_norm.detach().cpu()))
        and checkpoint["gradient"]["post_projection_violation_count"] == 0
        and checkpoint["gradient"]["diffusion_gradient_unchanged"] is True
    )
    result = {
        "schema": active.required_operation_schema,
        "pass": finite,
        "batch_size": rows,
        "calibration_batch_size": int(calibration_batch["timesteps"].shape[0]),
        "top_k": active.dry_run_top_k,
        "timestep_cutoff": active.dry_run_cutoff,
        "projection_mode": "blockwise",
        "diffusion_loss": float(diffusion_loss.detach().cpu()),
        "geometry_loss": float(geometry["total"].detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "post_projection_violation_count": int(
            projection["global"]["post_projection_violation_count"]
        ),
        "diffusion_gradient_unchanged": bool(
            projection["global"]["diffusion_gradient_unchanged"]
        ),
        "checkpoint_post_projection_violation_count": int(
            checkpoint["gradient"]["post_projection_violation_count"]
        ),
        "checkpoint_diffusion_gradient_unchanged": bool(
            checkpoint["gradient"]["diffusion_gradient_unchanged"]
        ),
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "initial_model_sha256": EXPECTED_STAGEC_INITIAL_MODEL_SHA256,
        "frozen_probe_accessed": False,
        "artifact_persisted": False,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    if result["pass"] is not True:
        raise ConflictProjectionError(
            "required CUDA operation dry run did not pass"
        )
    return result


def probe_portable_environment(
    *,
    root: Path,
    environment_spec: Optional[PortableEnvironmentSpec] = None,
) -> Dict[str, Any]:
    import platform
    import socket
    import sys

    active = (
        PortableEnvironmentSpec()
        if environment_spec is None
        else environment_spec
    )
    active.validate()
    validate_resume4_evidence(Path(root).resolve())
    torch, _ = stageb._torch_imports()
    python_version = tuple(sys.version_info[:3])
    if python_version != active.python_version:
        raise ConflictProjectionError(
            "Python contract changed: {}".format(python_version)
        )
    if np.__version__ != active.numpy_version:
        raise ConflictProjectionError(
            "NumPy contract changed: {}".format(np.__version__)
        )
    if torch.__version__ != active.torch_version:
        raise ConflictProjectionError(
            "PyTorch contract changed: {}".format(torch.__version__)
        )
    if torch.version.cuda != active.torch_cuda_version:
        raise ConflictProjectionError(
            "Torch CUDA contract changed: {}".format(torch.version.cuda)
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != active.cublas_workspace_config:
        raise ConflictProjectionError("CUBLAS_WORKSPACE_CONFIG changed")
    if os.environ.get("PYTHONHASHSEED") != active.python_hash_seed:
        raise ConflictProjectionError("PYTHONHASHSEED changed")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            raise ConflictProjectionError(
                "thread-count environment changed: {}".format(variable)
            )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ConflictProjectionError("no CUDA device is available")

    device_index = 0
    properties = torch.cuda.get_device_properties(device_index)
    stageb.set_deterministic_runtime(stageb.DiagnosticSpec().seed)
    deterministic_payload = {
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "thread_counts": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    if deterministic_payload != {
        "deterministic_algorithms_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": active.cublas_workspace_config,
        "python_hash_seed": active.python_hash_seed,
        "thread_counts": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }:
        raise ConflictProjectionError("deterministic runtime contract changed")

    dry_run = required_operation_dry_run(root=root, environment_spec=active)
    compatibility = {
        "schema": "phase314b_r258_stagea_portable_compatibility_v1",
        "python_version": list(python_version),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "required_operation_schema": active.required_operation_schema,
        "required_operation_pass": bool(dry_run["pass"]),
        "deterministic_runtime": deterministic_payload,
    }
    observation = {
        "schema": "phase314b_r258_stagea_hardware_observation_v1",
        "hostname": socket.gethostname(),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_index": device_index,
        "torch_device_name": str(torch.cuda.get_device_name(device_index)),
        "compute_capability": list(
            torch.cuda.get_device_capability(device_index)
        ),
        "total_memory_bytes": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _nvidia_smi_observation(),
    }
    result = {
        "schema": "phase314b_r258_stagea_portable_environment_v1",
        "compatibility": compatibility,
        "hardware_observation": observation,
        "required_operation_dry_run": dry_run,
        "compatibility_pass": True,
    }
    result["compatibility_sha256"] = sha256_bytes(
        stable_json_bytes(compatibility)
    )
    result["observation_sha256"] = sha256_bytes(
        stable_json_bytes(
            {
                "hardware_observation": observation,
                "required_operation_dry_run": dry_run,
            }
        )
    )
    return result


def _semanticize_portable(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_string = str(key)
            if key_string == "top_positions":
                continue
            if key_string.endswith("_sha256"):
                if key_string in PORTABLE_EXACT_SHA_KEYS:
                    result[key_string] = _semanticize_portable(item)
                continue
            result[key_string] = _semanticize_portable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_semanticize_portable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numeric_tree_compare(
    expected: Any,
    observed: Any,
    *,
    rtol: float,
    atol: float,
    maximum_records: int,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    total = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0

    def add(path: str, kind: str, left: Any, right: Any, **extra: Any) -> None:
        nonlocal total
        total += 1
        if len(records) < maximum_records:
            records.append(
                {
                    "path": path,
                    "kind": kind,
                    "expected": left,
                    "observed": right,
                    **extra,
                }
            )

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal maximum_absolute, maximum_relative
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys = set(left)
            right_keys = set(right)
            for key in sorted(left_keys - right_keys):
                add(path + "." + str(key), "missing_observed_key", left[key], None)
            for key in sorted(right_keys - left_keys):
                add(path + "." + str(key), "unexpected_observed_key", None, right[key])
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], path + "." + str(key))
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                add(path, "sequence_length", len(left), len(right))
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, "{}[{}]".format(path, index))
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                add(path, "boolean_mismatch", left, right)
            return
        if left is None or right is None:
            if left is not right:
                add(path, "none_mismatch", left, right)
            return
        if isinstance(left, str) or isinstance(right, str):
            if type(left) is not type(right) or left != right:
                add(path, "string_mismatch", left, right)
            return
        if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
            if int(left) != int(right):
                add(path, "integer_mismatch", int(left), int(right))
            return
        if isinstance(left, (int, float, np.number)) and isinstance(
            right, (int, float, np.number)
        ):
            left_float = float(left)
            right_float = float(right)
            if not np.isfinite(left_float) or not np.isfinite(right_float):
                if left_float != right_float:
                    add(path, "nonfinite_mismatch", left_float, right_float)
                return
            absolute = abs(left_float - right_float)
            relative = absolute / max(abs(left_float), abs(right_float), 1.0e-30)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            if not math.isclose(left_float, right_float, rel_tol=rtol, abs_tol=atol):
                add(
                    path,
                    "numeric_tolerance",
                    left_float,
                    right_float,
                    absolute_difference=absolute,
                    relative_difference=relative,
                    rtol=float(rtol),
                    atol=float(atol),
                )
            return
        if type(left) is not type(right) or left != right:
            add(path, "value_mismatch", repr(left), repr(right))

    visit(expected, observed, "$")
    return {
        "pass": total == 0,
        "difference_count": total,
        "recorded_difference_count": len(records),
        "records_truncated": total > len(records),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "rtol": float(rtol),
        "atol": float(atol),
        "differences": records,
    }


def replay_stagec_control_portable(
    *,
    root: Path,
    equivalence_spec: Optional[PortableNumericalEquivalenceSpec] = None,
) -> Dict[str, Any]:
    active = (
        PortableNumericalEquivalenceSpec()
        if equivalence_spec is None
        else equivalence_spec
    )
    active.validate()
    repository_root = Path(root).resolve()
    immutable = staged.validate_immutable_inputs(repository_root)
    expected_calibration = immutable["stagec_contract"]["calibration"]
    expected_control = immutable["control_record"]

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    upstream = stagec_topk.validate_immutable_inputs(repository_root)
    objective_contract = stageb_mechanism.load_objective_contract(
        upstream["upstream_immutable"]["stagea_contract"]
    )
    upper_gate = stagea.load_upper_gate_contract(
        upstream["upstream_immutable"]["upstream_immutable"]["staged3_contract"]
    )
    stage_d_contract, _ = staged1.load_stage_d_gate(
        repository_root / staged3.STAGE_D_GATE
    )
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    condition = np.asarray(arrays["diffusion_condition_x"], dtype=np.float32)
    target = np.asarray(arrays["diffusion_target_cable"], dtype=np.float32)
    groups = np.asarray(arrays["episode_group_key"]).astype(str)
    condition_name = np.asarray(arrays["condition_name"]).astype(str)
    stageb_train, frozen_probe, _ = stageb.deterministic_group_split(
        groups,
        folds=stageb_spec.group_folds,
        probe_fold=stageb_spec.probe_fold,
    )
    objective_train, selection_holdout, split = stagea.deterministic_selection_split(
        groups,
        stageb_train,
        folds=stagea.UpperObjectiveSpec().selection_group_folds,
        holdout_fold=stagea.UpperObjectiveSpec().selection_holdout_fold,
    )
    stagec_split = immutable["stagec_worker_result"]["split"]
    for key in (
        "objective_train_rows",
        "selection_holdout_rows",
        "objective_train_groups",
        "selection_holdout_groups",
        "objective_train_group_sha256",
        "selection_holdout_group_sha256",
        "group_overlap",
        "row_overlap",
    ):
        if split[key] != stagec_split[key]:
            raise ConflictProjectionError(
                "portable Stage-C split changed: {}".format(key)
            )
    if np.any(frozen_probe & (objective_train | selection_holdout)):
        raise ConflictProjectionError("portable Stage-C replay crossed frozen probe")

    condition_standardizer = stageb.fit_standardizer(condition[objective_train])
    target_standardizer = stageb.fit_standardizer(target[objective_train])
    diagnostic_batch = stageb_mechanism.fixed_diagnostic_batch(
        condition=condition[objective_train],
        target=target[objective_train],
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=stageb_mechanism.MechanismAuditSpec(),
    )
    expected_batch_sha = immutable["upstream_immutable"]["stageb_contract"]["diagnostic_batch_sha256"]
    observed_batch_sha = {
        key: value
        for key, value in diagnostic_batch.items()
        if key.endswith("_sha256")
    }
    if observed_batch_sha != expected_batch_sha:
        raise ConflictProjectionError("portable Stage-C diagnostic batch changed")

    candidates, observed_calibration = stagec_topk.calibrate_candidates(
        stageb_spec=stageb_spec,
        diagnostic_batch=diagnostic_batch,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    stageb.set_deterministic_runtime(stageb_spec.seed)
    model, training, diagnostics = stagec_topk.train_candidate(
        condition=condition[objective_train],
        target=target[objective_train],
        stageb_spec=stageb_spec,
        objective_contract=objective_contract,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        candidate=None,
        diagnostic_batch=diagnostic_batch,
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    train_control = stagea.train_control_audit(
        model=model,
        condition=condition[objective_train],
        target=target[objective_train],
        stageb_spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    one_step = stagea.one_step_audit(
        model=model,
        condition=condition[selection_holdout],
        target=target[selection_holdout],
        groups=groups[selection_holdout],
        condition_name=condition_name[selection_holdout],
        stageb_spec=stageb_spec,
        upper_gate=upper_gate,
        stage_d_contract=stage_d_contract,
        historical_geometry=stageb.fit_geometry_contract(target[stageb_train]),
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=(
            stageb_spec.seed
            + stagec_topk.TopKCalibrationSpec().holdout_noise_seed_offset
        ),
    )
    predictions, prediction_sha = stagec.one_step_predictions(
        model=model,
        condition=condition[selection_holdout],
        target=target[selection_holdout],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=(
            stageb_spec.seed
            + stagec_topk.TopKCalibrationSpec().holdout_noise_seed_offset
        ),
    )
    if prediction_sha != one_step["prediction_sha256"]:
        raise ConflictProjectionError("portable one-step aggregate SHA is inconsistent")
    profiles = stagec_topk.holdout_profiles(
        predictions,
        upper_gate=upper_gate,
        objective_contract=objective_contract,
    )
    observed_control = stagec_topk.selection_record(
        candidate=None,
        training=training,
        training_diagnostics=diagnostics,
        train_control=train_control,
        one_step=one_step,
        profiles=profiles,
        control=None,
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    del model

    calibration_semantic_expected = _semanticize_portable(expected_calibration)
    calibration_semantic_observed = _semanticize_portable(observed_calibration)
    control_semantic_expected = _semanticize_portable(expected_control)
    control_semantic_observed = _semanticize_portable(observed_control)
    calibration_comparison = _numeric_tree_compare(
        calibration_semantic_expected,
        calibration_semantic_observed,
        rtol=active.calibration_rtol,
        atol=active.calibration_atol,
        maximum_records=active.maximum_difference_records,
    )
    control_comparison = _numeric_tree_compare(
        control_semantic_expected,
        control_semantic_observed,
        rtol=active.control_rtol,
        atol=active.control_atol,
        maximum_records=active.maximum_difference_records,
    )
    byte_exact_calibration = (
        stable_json_bytes(expected_calibration)
        == stable_json_bytes(observed_calibration)
    )
    byte_exact_control = (
        stable_json_bytes(expected_control)
        == stable_json_bytes(observed_control)
    )
    pass_gate = bool(
        calibration_comparison["pass"]
        and control_comparison["pass"]
    )
    identity = {
        "capture_method": "frozen_stagec_helpers_portable_numeric_replay",
        "reference_environment": "Resume4 RTX-3090 evidence",
        "reference_calibration_sha256": EXPECTED_STAGEC_CALIBRATION_SHA256,
        "reference_control_sha256": EXPECTED_STAGEC_CONTROL_SHA256,
        "observed_calibration_sha256": observed_calibration["calibration_sha256"],
        "observed_control_sha256": sha256_bytes(
            stable_json_bytes(observed_control)
        ),
        "candidate_order": [candidate.candidate_id for candidate in candidates],
        "stagec_nonzero_candidate_training_count": 0,
        "calibration_byte_exact": byte_exact_calibration,
        "control_byte_exact": byte_exact_control,
        "byte_exact_to_reference": bool(
            byte_exact_calibration and byte_exact_control
        ),
        "calibration_numerical_equivalence": calibration_comparison,
        "control_numerical_equivalence": control_comparison,
        "reference_equivalence_pass": pass_gate,
        "hardware_sensitive_sha_fields_are_observations": True,
        "candidate_selection_thresholds_relaxed": False,
        "frozen_probe_accessed": False,
    }
    if not pass_gate:
        paths = [
            item["path"]
            for item in (
                calibration_comparison["differences"]
                + control_comparison["differences"]
            )[:16]
        ]
        raise ConflictProjectionError(
            "portable Stage-C numerical-equivalence gate failed: {}".format(paths)
        )
    return {
        "control_bundle": observed_control,
        "control_identity": identity,
    }


def validate_environment_payload(
    environment: Mapping[str, Any],
) -> None:
    active = PortableEnvironmentSpec()
    active.validate()
    if environment.get("compatibility_pass") is not True:
        raise ConflictProjectionError("portable compatibility gate did not pass")
    compatibility = environment.get("compatibility")
    observation = environment.get("hardware_observation")
    dry_run = environment.get("required_operation_dry_run")
    if not isinstance(compatibility, Mapping):
        raise ConflictProjectionError("portable compatibility payload is missing")
    if not isinstance(observation, Mapping):
        raise ConflictProjectionError("hardware observation is missing")
    if not isinstance(dry_run, Mapping) or dry_run.get("pass") is not True:
        raise ConflictProjectionError("required-operation dry run did not pass")
    expected_compatibility_sha = sha256_bytes(
        stable_json_bytes(compatibility)
    )
    if environment.get("compatibility_sha256") != expected_compatibility_sha:
        raise ConflictProjectionError("portable compatibility SHA is invalid")
    expected_observation_sha = sha256_bytes(
        stable_json_bytes(
            {
                "hardware_observation": observation,
                "required_operation_dry_run": dry_run,
            }
        )
    )
    if environment.get("observation_sha256") != expected_observation_sha:
        raise ConflictProjectionError("hardware observation SHA is invalid")
    if tuple(compatibility.get("python_version", ())) != active.python_version:
        raise ConflictProjectionError("portable Python compatibility changed")
    if compatibility.get("numpy_version") != active.numpy_version:
        raise ConflictProjectionError("portable NumPy compatibility changed")
    if compatibility.get("torch_version") != active.torch_version:
        raise ConflictProjectionError("portable PyTorch compatibility changed")
    if compatibility.get("torch_cuda_version") != active.torch_cuda_version:
        raise ConflictProjectionError("portable Torch-CUDA compatibility changed")
    if compatibility.get("required_operation_schema") != active.required_operation_schema:
        raise ConflictProjectionError("portable required-operation schema changed")
    if compatibility.get("required_operation_pass") is not True:
        raise ConflictProjectionError("portable required-operation contract failed")
    # Deliberately no GPU name, UUID, PCI bus, compute capability, driver, or
    # total-memory equality requirement.  Those fields are audit observations.


def assert_cold_cuda_context_portable() -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    initialized = bool(torch.cuda.is_initialized())
    if initialized:
        raise ConflictProjectionError(
            "main Stage-A worker initialized CUDA before portable control replay"
        )
    return {
        "checked": True,
        "torch_cuda_is_initialized": False,
        "purpose": (
            "environment and operation probes run in disposable child processes"
        ),
    }
