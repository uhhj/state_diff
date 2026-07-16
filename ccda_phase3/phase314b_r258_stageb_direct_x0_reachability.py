"""Phase3.14b-r2.5.8 Stage B direct-x0 reachability audit.

The r2.5.8 Stage-A portable conflict-projection experiment established that
blockwise gradient conflict exists but is too small to explain the persistent
geometry/fidelity failure.  This stage therefore does not train another model
candidate and does not change the geometry loss.

It audits three distinct questions on the frozen train-only selection holdout:

1. Target-direction reachability:
   Does interpolation from the exact diffusion-only control prediction toward
   the train-only ground truth cross the frozen one-sided upper gate while
   improving fidelity and retaining cable validity?

2. Direct-x0 objective reachability:
   When the predicted x0 tensor itself is optimized with the frozen K16
   upper-only objective inside pre-registered normalized trust regions, does it
   approach the target, or does it exploit the one-sided objective by shrinking
   ordered segments and degrading lower/physical validity?

3. Model tangent reachability:
   Can the local tangent space of the direct-x0 denoiser express the valid
   target-directed displacement?  A VJP supplies the steepest parameter
   direction and deterministic central finite differences estimate the output
   response globally and for each frozen model block.  A separate exact linear
   least-squares audit measures the capacity of the output head with frozen
   hidden features.

The current hardware uses the portable compatibility contract.  The frozen
Stage-C control is replayed through the r2.5.8 portable numerical-equivalence
gate and retained in memory through a read-only capture wrapper.  Two workers
on the current rented instance must be byte-exact.

The 126-row frozen probe, full repaired-model training, reverse sampling,
formal training, IDM, candidate execution, DeformableRavens execution, Phase4,
and CPS remain closed.  No checkpoint, weight, prediction tensor, oracle tensor,
NPZ, cache, image, or video is persisted.
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
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3 import phase314b_r257_stageb_mechanism_audit as stageb_mechanism
from ccda_phase3 import phase314b_r257_stagec_topk_quadratic as stagec_topk
from ccda_phase3 import phase314b_r258_stagea_conflict_projected_k16 as stagea258

PHASE = "Phase3.14b-r2.5.8 Stage B"
PHASE_ID = "phase314b_r258_stageb"

BASE_EVIDENCE_COMMIT = (
    "4d5773d070b99f0eccdb97c1bc9fa7c5cb3bc583"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

BASE_CONTRACT = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_contract.json"
)
BASE_WORKER = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_worker_evidence.json"
)
BASE_SUMMARY = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_summary.json"
)
BASE_REPORT = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_report.md"
)
BASE_TEST_GATE = (
    "reports/"
    "phase3_14b_r258_stagea_resume1_test_gate_summary.json"
)

EXPECTED_BASE_WORKER_SHA256 = (
    "c71590b7557da309cc02b3fc9e6c4387"
    "f815eb7800d4307f76ce84d238c13149"
)
EXPECTED_COMPATIBILITY_SHA256 = (
    "03fef5cfcfac7933bca0dfc72e67156b"
    "b260d50ae7f2f1f6a6375a34c90ad7ec"
)
EXPECTED_OBSERVATION_SHA256 = (
    "a20b4f68cbbe7ab1dd6f04d35887e999"
    "71706658b6113fd8dfd97d1e9849707e"
)
EXPECTED_STAGEC_CALIBRATION_SHA256 = (
    "fb91d29c84cedccf8f72458f08c2f213"
    "bbbc542663593094c0ab651701a052f0"
)
EXPECTED_STAGEC_CONTROL_SHA256 = (
    "dee61b8e31455eeb0e7e0eceae5c1500"
    "e3ce57ca8bd42989378d9f76fd779ae7"
)
EXPECTED_BASE_CALIBRATION_SHA256 = (
    "2460d4bbb4faf379c3021b6ed87b51757"
    "5f1b7e1c33773440d7348f817de27db"
)
EXPECTED_BASE_CONTRACT_SHA256 = (
    "73cb8412498aa5c65c01fc123c35ee72"
    "eb8da9c6c543d31227118dd8bffc693b"
)
EXPECTED_BASE_CONTRACT_FILE_SHA256 = (
    "f995aca77f7a9e19bbf02dc658d180ef"
    "c9a7622c59751327e047de2d893b4aa5"
)
EXPECTED_BASE_SELECTION_SHA256 = (
    "62e2800808620dc48858dec5105f97fca"
    "71ac25a5ca987242ae5dc2a9cc71a4f"
)
EXPECTED_CONTROL_INITIAL_MODEL_SHA256 = (
    "4f1c102d9f39ba59544f5565b4c01fe0"
    "3e073fca596aefdc822b707cd2f2c7fd"
)

BASE_BOUND_FILES = (
    BASE_CONTRACT,
    BASE_WORKER,
    BASE_SUMMARY,
    BASE_REPORT,
    BASE_TEST_GATE,
)

TIMESTEP_UPPER_MIN = {
    10: 0.75,
    25: 0.60,
    50: 0.50,
}


class ReachabilityAuditError(RuntimeError):
    """Raised when immutable evidence or an audit invariant fails."""


@dataclass(frozen=True)
class ReachabilitySpec:
    """Pre-registered train-only reachability audit contract."""

    timesteps: Tuple[int, ...] = (10, 25, 50)
    top_k: int = 16

    target_line_grid_points: int = 101
    target_line_physical_valid_min: float = 0.90

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
    oracle_fidelity_ratio_max: float = 1.20
    oracle_lower_pass_min: float = 0.90
    oracle_historical_valid_min: float = 0.90
    oracle_collapse_fraction_max: float = 0.05

    tangent_rows: int = 32
    tangent_relative_epsilons: Tuple[float, ...] = (
        1.0e-4,
        3.0e-4,
        1.0e-3,
    )
    tangent_krylov_iterations: int = 8
    tangent_krylov_min_improvement: float = 1.0e-4
    tangent_response_stability_cosine_min: float = 0.95
    tangent_explained_low: float = 0.25
    tangent_explained_high: float = 0.75
    output_head_explained_high: float = 0.90
    output_head_relative_update_max: float = 1.00
    output_head_rcond: float = 1.0e-10

    geometry_gradient_target_cosine_nonnegative: float = 0.0
    norm_epsilon: float = 1.0e-12
    numerical_tolerance: float = 1.0e-6

    def validate(self) -> None:
        if self.timesteps != tuple(sorted(TIMESTEP_UPPER_MIN)):
            raise ValueError("registered timestep population changed")
        if self.top_k != 16:
            raise ValueError("registered K16 objective changed")
        if self.target_line_grid_points != 101:
            raise ValueError("target-line grid changed")
        if self.target_line_grid_points < 3:
            raise ValueError("target-line grid is too small")
        if self.oracle_radii != tuple(sorted(set(self.oracle_radii))):
            raise ValueError("oracle radii must be ordered and unique")
        if any(float(value) <= 0.0 for value in self.oracle_radii):
            raise ValueError("oracle radius must be positive")
        if self.oracle_steps <= 0:
            raise ValueError("oracle steps must be positive")
        if self.oracle_checkpoints[0] != 0:
            raise ValueError("oracle checkpoints must include step zero")
        if self.oracle_checkpoints[-1] != self.oracle_steps:
            raise ValueError("final oracle checkpoint changed")
        if tuple(sorted(set(self.oracle_checkpoints))) != self.oracle_checkpoints:
            raise ValueError("oracle checkpoints must be ordered and unique")
        if self.tangent_rows <= 0:
            raise ValueError("tangent row count must be positive")
        if self.tangent_krylov_iterations <= 0:
            raise ValueError("tangent Krylov iteration count must be positive")
        if self.tangent_krylov_min_improvement <= 0.0:
            raise ValueError("tangent Krylov improvement threshold is invalid")
        if self.tangent_relative_epsilons != tuple(
            sorted(set(self.tangent_relative_epsilons))
        ):
            raise ValueError("tangent epsilons must be ordered and unique")
        for value in (
            self.target_line_physical_valid_min,
            self.oracle_fidelity_ratio_max,
            self.oracle_lower_pass_min,
            self.oracle_historical_valid_min,
            self.tangent_response_stability_cosine_min,
            self.tangent_explained_low,
            self.tangent_explained_high,
            self.output_head_explained_high,
            self.output_head_relative_update_max,
        ):
            if float(value) <= 0.0:
                raise ValueError("registered positive threshold is invalid")
        if self.tangent_explained_low >= self.tangent_explained_high:
            raise ValueError("tangent thresholds are not ordered")
        if self.oracle_collapse_fraction_max < 0.0:
            raise ValueError("collapse threshold is negative")
        if self.norm_epsilon <= 0.0 or self.numerical_tolerance <= 0.0:
            raise ValueError("numerical constants must be positive")


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
    raise TypeError("unsupported JSON value: {!r}".format(type(value)))


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
        raise ReachabilityAuditError(
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
        raise ReachabilityAuditError(
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
        raise ReachabilityAuditError(
            "base-bound file differs: {}".format(relative)
        )
    return sha256_bytes(observed)


def validate_base_evidence(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_ancestor(repository_root, BASE_EVIDENCE_COMMIT)
    file_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            BASE_EVIDENCE_COMMIT,
        )
        for relative in BASE_BOUND_FILES
    }
    if file_sha[BASE_CONTRACT] != EXPECTED_BASE_CONTRACT_FILE_SHA256:
        raise ReachabilityAuditError("base contract-file SHA changed")

    summary = load_json(repository_root / BASE_SUMMARY)
    worker = load_json(repository_root / BASE_WORKER)
    contract = load_json(repository_root / BASE_CONTRACT)
    test_gate = load_json(repository_root / BASE_TEST_GATE)

    if summary.get("verdict") != "PASS":
        raise ReachabilityAuditError("base verdict is not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise ReachabilityAuditError("base scientific status changed")
    if summary.get("root_cause") != (
        "phase314b_r258_stagea_conflict_projected_k16_"
        "no_geometry_or_fidelity_solution"
    ):
        raise ReachabilityAuditError("base root cause changed")
    if summary.get("required_next_path") != (
        "AUDIT_DIRECT_X0_MODEL_PARAMETERIZATION_AND_"
        "UPPER_GEOMETRY_TARGET_REACHABILITY"
    ):
        raise ReachabilityAuditError("base next path changed")
    if summary.get("selected_configuration") is not None:
        raise ReachabilityAuditError("base selected a configuration")
    if summary.get("workers_exact") is not True:
        raise ReachabilityAuditError("base workers are not exact")

    comparison = worker.get("comparison", {})
    if comparison.get("exact") is not True:
        raise ReachabilityAuditError("base worker comparison changed")
    if (
        comparison.get("left_sha256") != EXPECTED_BASE_WORKER_SHA256
        or comparison.get("right_sha256") != EXPECTED_BASE_WORKER_SHA256
    ):
        raise ReachabilityAuditError("base worker identity changed")

    result = worker.get("worker_result")
    if not isinstance(result, dict):
        raise ReachabilityAuditError("base worker result is missing")
    environment = result.get("environment", {})
    if environment.get("compatibility_sha256") != EXPECTED_COMPATIBILITY_SHA256:
        raise ReachabilityAuditError("base compatibility identity changed")
    if environment.get("observation_sha256") != EXPECTED_OBSERVATION_SHA256:
        raise ReachabilityAuditError("base observation identity changed")
    if result.get("control_replay_exact") is not True:
        raise ReachabilityAuditError("base control replay is not exact")
    control = result.get("control_capture", {})
    if control.get("observed_calibration_sha256") != (
        EXPECTED_STAGEC_CALIBRATION_SHA256
    ):
        raise ReachabilityAuditError("base Stage-C calibration changed")
    if control.get("observed_control_sha256") != EXPECTED_STAGEC_CONTROL_SHA256:
        raise ReachabilityAuditError("base Stage-C control changed")
    if control.get("stagec_nonzero_candidate_training_count") != 0:
        raise ReachabilityAuditError("base Stage-C candidate count changed")
    calibration = result.get("calibration_contract", {}).get("calibration", {})
    if calibration.get("calibration_sha256") != EXPECTED_BASE_CALIBRATION_SHA256:
        raise ReachabilityAuditError("base projection calibration changed")
    if result.get("calibration_contract", {}).get("contract_sha256") != (
        EXPECTED_BASE_CONTRACT_SHA256
    ):
        raise ReachabilityAuditError("base internal contract changed")
    if result.get("selection", {}).get("selection_sha256") != (
        EXPECTED_BASE_SELECTION_SHA256
    ):
        raise ReachabilityAuditError("base selection changed")
    if contract.get("contract_sha256") != EXPECTED_BASE_CONTRACT_SHA256:
        raise ReachabilityAuditError("base contract record changed")

    if test_gate.get("verdict") != "PASS":
        raise ReachabilityAuditError("base test gate is not PASS")
    if int(test_gate.get("test_file_count", -1)) != 48:
        raise ReachabilityAuditError("base test-file count changed")
    if int(test_gate.get("passed_test_count", -1)) != 911:
        raise ReachabilityAuditError("base pass count changed")
    if int(test_gate.get("resume1_new_passed", -1)) != 12:
        raise ReachabilityAuditError("base Resume1 test count changed")

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
            raise ReachabilityAuditError(
                "base forbidden-stage boundary changed: {}".format(key)
            )

    return {
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "file_sha256": file_sha,
        "summary": summary,
        "worker": worker,
        "worker_result": result,
        "contract": contract,
        "test_gate": test_gate,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageb_text = (
        repository_root
        / "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py"
    ).read_text(encoding="utf-8")
    stagea_text = (
        repository_root
        / "ccda_phase3/phase314b_r257_stagea_upper_objective.py"
    ).read_text(encoding="utf-8")
    topk_text = (
        repository_root
        / "ccda_phase3/phase314b_r257_stagec_topk_quadratic.py"
    ).read_text(encoding="utf-8")
    checks = {
        "direct_x0_residual_parameterization": (
            "predicted = flat + self.output(self.output_norm(hidden))"
            in stageb_text
        ),
        "upper_objective_uses_raw_ordered_xy": (
            "predicted_raw = predicted_x0_z * scale + mean"
            in stagea_text
            and "delta_xy = points[:, :, 1:, :] - points[:, :, :-1, :]"
            in stagea_text
        ),
        "upper_objective_is_one_sided": (
            "excess = torch.relu(log_lengths - allowed)"
            in stagea_text
        ),
        "upper_objective_has_no_lower_penalty": (
            '"uses_lower_xy_constraint": False'
            in stagea_text
        ),
        "topk_quadratic_is_k16_compatible": (
            "torch.topk(" in topk_text
            and "row_loss = torch.mean(selected * selected, dim=1)"
            in topk_text
        ),
        "historical_physical_validity_tracks_lower_and_upper": (
            "(lengths >= contract.segment_lower[None, None])"
            in stageb_text
            and "(lengths <= contract.segment_upper[None, None])"
            in stageb_text
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "source_sha256": {
            "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py":
                sha256_file(
                    repository_root
                    / "ccda_phase3/phase314b_r256_stageb_cable_diffusion.py"
                ),
            "ccda_phase3/phase314b_r257_stagea_upper_objective.py":
                sha256_file(
                    repository_root
                    / "ccda_phase3/phase314b_r257_stagea_upper_objective.py"
                ),
            "ccda_phase3/phase314b_r257_stagec_topk_quadratic.py":
                sha256_file(
                    repository_root
                    / "ccda_phase3/phase314b_r257_stagec_topk_quadratic.py"
                ),
        },
        "identified_risk": (
            "one-sided upper-only geometry descent can shorten segments "
            "without an explicit lower/collapse constraint"
        ),
        "model_architecture_changed": False,
        "geometry_objective_changed": False,
        "new_model_candidate_trained": False,
    }


def capture_portable_control_model(
    *,
    root: Path,
) -> Dict[str, Any]:
    """Run the frozen portable replay while retaining its control model."""
    captured: Dict[str, Any] = {}
    original = stagec_topk.train_candidate
    call_count = 0

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        result = original(*args, **kwargs)
        if call_count != 1:
            raise ReachabilityAuditError(
                "portable replay trained more than one model"
            )
        captured["model"] = result[0]
        captured["training"] = result[1]
        captured["training_diagnostics"] = result[2]
        return result

    stagec_topk.train_candidate = wrapper
    try:
        replay = stagea258.replay_stagec_control_portable(
            root=Path(root).resolve()
        )
    finally:
        stagec_topk.train_candidate = original

    if call_count != 1 or "model" not in captured:
        raise ReachabilityAuditError(
            "portable replay did not expose exactly one control model"
        )
    identity = replay["control_identity"]
    if identity.get("reference_equivalence_pass") is not True:
        raise ReachabilityAuditError(
            "portable control reference equivalence failed"
        )
    if identity.get("stagec_nonzero_candidate_training_count") != 0:
        raise ReachabilityAuditError(
            "portable replay trained a nonzero Stage-C candidate"
        )
    if identity.get("observed_calibration_sha256") != (
        EXPECTED_STAGEC_CALIBRATION_SHA256
    ):
        raise ReachabilityAuditError(
            "portable calibration identity changed"
        )
    if identity.get("observed_control_sha256") != (
        EXPECTED_STAGEC_CONTROL_SHA256
    ):
        raise ReachabilityAuditError(
            "portable control identity changed"
        )
    if captured["training"].get("initial_model_sha256") != (
        EXPECTED_CONTROL_INITIAL_MODEL_SHA256
    ):
        raise ReachabilityAuditError(
            "portable control initial model changed"
        )
    return {
        **captured,
        "control_bundle": replay["control_bundle"],
        "control_identity": identity,
        "train_candidate_call_count": call_count,
        "capture_wrapper_restored": (
            stagec_topk.train_candidate is original
        ),
    }


def build_audit_context(
    *,
    root: Path,
    captured: Mapping[str, Any],
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()

    upstream = stagec_topk.validate_immutable_inputs(
        repository_root
    )
    objective_contract = (
        stageb_mechanism.load_objective_contract(
            upstream["upstream_immutable"][
                "stagea_contract"
            ]
        )
    )
    upper_gate = stagea.load_upper_gate_contract(
        upstream["upstream_immutable"][
            "upstream_immutable"
        ]["staged3_contract"]
    )
    stage_d_contract, stage_d_payload = (
        staged1.load_stage_d_gate(
            repository_root
            / staged3.STAGE_D_GATE
        )
    )
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    validation = stageb.validate_train_view(arrays)
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

    stageb_train, frozen_probe, stageb_split = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
        )
    )
    objective_train, selection_holdout, selection_split = (
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
    if np.any(
        frozen_probe
        & (
            objective_train
            | selection_holdout
        )
    ):
        raise ReachabilityAuditError(
            "audit context crossed the frozen probe"
        )

    condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )
    historical_geometry = stageb.fit_geometry_contract(
        target[stageb_train]
    )

    holdout_condition = condition[selection_holdout]
    holdout_target = target[selection_holdout]
    holdout_groups = groups[selection_holdout]
    holdout_condition_name = condition_name[
        selection_holdout
    ]
    predictions, prediction_sha = (
        stagec.one_step_predictions(
            model=captured["model"],
            condition=holdout_condition,
            target=holdout_target,
            spec=stageb_spec,
            condition_standardizer=
                condition_standardizer,
            target_standardizer=
                target_standardizer,
            noise_seed=(
                stageb_spec.seed
                + stagec_topk.TopKCalibrationSpec()
                .holdout_noise_seed_offset
            ),
        )
    )
    expected_prediction_sha = (
        captured["control_bundle"][
            "one_step"
        ]["prediction_sha256"]
    )
    if prediction_sha != expected_prediction_sha:
        raise ReachabilityAuditError(
            "retained control model prediction differs "
            "from portable replay"
        )

    return {
        "repository_root": repository_root,
        "stageb_spec": stageb_spec,
        "objective_contract": objective_contract,
        "upper_gate": upper_gate,
        "stage_d_contract": stage_d_contract,
        "stage_d_payload": stage_d_payload,
        "arrays_validation": validation,
        "condition": condition,
        "target": target,
        "groups": groups,
        "condition_name": condition_name,
        "stageb_train_mask": stageb_train,
        "frozen_probe_mask": frozen_probe,
        "objective_train_mask": objective_train,
        "selection_holdout_mask": selection_holdout,
        "stageb_split": stageb_split,
        "selection_split": selection_split,
        "condition_standardizer":
            condition_standardizer,
        "target_standardizer":
            target_standardizer,
        "historical_geometry":
            historical_geometry,
        "holdout_condition":
            holdout_condition,
        "holdout_target":
            holdout_target,
        "holdout_groups":
            holdout_groups,
        "holdout_condition_name":
            holdout_condition_name,
        "control_predictions":
            predictions,
        "control_prediction_sha256":
            prediction_sha,
        "model": captured["model"],
        "control_bundle":
            captured["control_bundle"],
        "control_identity":
            captured["control_identity"],
    }


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


def _row_cosine(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64).reshape(
        left.shape[0],
        -1,
    )
    b = np.asarray(right, dtype=np.float64).reshape(
        right.shape[0],
        -1,
    )
    numerator = np.sum(a * b, axis=1)
    denominator = np.sqrt(
        np.sum(a * a, axis=1)
        * np.sum(b * b, axis=1)
    )
    return numerator / np.maximum(
        denominator,
        float(epsilon),
    )


def topk_upper_profile_numpy(
    value: np.ndarray,
    *,
    objective_contract: stagea.UpperObjectiveContract,
    top_k: int,
) -> Dict[str, Any]:
    raw = np.asarray(value, dtype=np.float32)
    lengths = stageb.segment_lengths(raw).astype(
        np.float64
    )
    log_lengths = np.log(
        np.maximum(
            lengths,
            float(
                objective_contract
                .segment_length_epsilon
            ),
        )
    )
    allowed = np.asarray(
        objective_contract.allowed_upper_log_length,
        dtype=np.float64,
    )
    excess = np.maximum(
        log_lengths - allowed[None],
        0.0,
    )
    flattened = excess.reshape(
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
    row_loss = np.mean(
        selected * selected,
        axis=1,
    )
    return {
        "total": float(np.mean(row_loss)),
        "row_loss": _safe_stats(row_loss),
        "row_loss_sha256":
            sha256_array(row_loss.astype(np.float64)),
        "maximum_excess":
            float(np.max(excess)),
        "positive_element_rate":
            float(np.mean(excess > 0.0)),
        "row_violation_rate":
            float(
                np.mean(
                    np.max(excess, axis=(1, 2))
                    > 0.0
                )
            ),
    }


def evaluate_output_population(
    value: np.ndarray,
    *,
    control: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    upper_gate: staged3.OneSidedUpperContract,
    stage_d_contract: staged.SegmentGateContract,
    historical_geometry: stageb.GeometryContract,
    top_k: int,
    epsilon: float,
) -> Dict[str, Any]:
    candidate = np.asarray(
        value,
        dtype=np.float32,
    )
    control_raw = np.asarray(
        control,
        dtype=np.float32,
    )
    target_raw = np.asarray(
        target,
        dtype=np.float32,
    )
    if (
        candidate.shape
        != control_raw.shape
        or candidate.shape
        != target_raw.shape
    ):
        raise ValueError(
            "output/control/target shapes differ"
        )

    scores = staged.segment_scores(
        candidate,
        stage_d_contract.reference,
    )
    upper_pass = (
        scores["upper"][:, 0]
        <= float(upper_gate.upper_threshold)
    )
    lower_pass = (
        scores["lower"][:, 0]
        <= float(stage_d_contract.lower_threshold)
    )
    joint_pass = (
        scores["joint"][:, 0]
        <= float(stage_d_contract.joint_threshold)
    )
    physical = stageb.physical_validity(
        candidate[:, None],
        historical_geometry,
    )
    upper_decomposition = (
        staged3.upper_only_decomposition(
            value=candidate,
            contract=upper_gate,
            stage_d_contract=
                stage_d_contract,
            historical_geometry=
                historical_geometry,
            groups=groups,
            condition_name=condition_name,
        )
    )

    candidate_z = target_standardizer.normalize(
        candidate
    )
    control_z = target_standardizer.normalize(
        control_raw
    )
    target_z = target_standardizer.normalize(
        target_raw
    )
    active = np.asarray(
        target_standardizer.active,
        dtype=np.bool_,
    ).reshape(-1)
    delta_control = (
        candidate_z - control_z
    ).reshape(candidate.shape[0], -1)
    target_direction = (
        target_z - control_z
    ).reshape(candidate.shape[0], -1)
    residual_to_target = (
        target_z - candidate_z
    ).reshape(candidate.shape[0], -1)
    if np.any(active):
        delta_active = delta_control[:, active]
        target_active = target_direction[:, active]
        residual_active = residual_to_target[:, active]
    else:
        delta_active = delta_control
        target_active = target_direction
        residual_active = residual_to_target

    control_nmse = stageb.normalized_mse(
        control_raw,
        target_raw,
        target_standardizer,
    )
    candidate_nmse = stageb.normalized_mse(
        candidate,
        target_raw,
        target_standardizer,
    )
    control_lengths = stageb.segment_lengths(
        control_raw
    ).astype(np.float64)
    target_lengths = stageb.segment_lengths(
        target_raw
    ).astype(np.float64)
    candidate_lengths = stageb.segment_lengths(
        candidate
    ).astype(np.float64)
    collapse = candidate_lengths < (
        0.50 * np.maximum(
            target_lengths,
            float(epsilon),
        )
    )
    stretch = candidate_lengths > (
        1.50 * np.maximum(
            target_lengths,
            float(epsilon),
        )
    )

    return {
        "rows": int(candidate.shape[0]),
        "normalized_mse": float(candidate_nmse),
        "control_normalized_mse":
            float(control_nmse),
        "normalized_mse_ratio": (
            float(candidate_nmse / control_nmse)
            if control_nmse > 0.0
            else (
                1.0
                if candidate_nmse == 0.0
                else float("inf")
            )
        ),
        "raw_rmse":
            stageb.rmse(candidate, target_raw),
        "topk_upper":
            topk_upper_profile_numpy(
                candidate,
                objective_contract=
                    objective_contract,
                top_k=top_k,
            ),
        "upper_row_pass_rate":
            float(np.mean(upper_pass)),
        "lower_row_pass_rate":
            float(np.mean(lower_pass)),
        "joint_row_pass_rate":
            float(np.mean(joint_pass)),
        "historical_physical_candidate_rate":
            float(physical["candidate_rate"]),
        "historical_physical_row_any_rate":
            float(physical["row_any_rate"]),
        "upper_combined_row_any_rate":
            float(
                upper_decomposition[
                    "combined"
                ]["row_any_rate"]
            ),
        "coordinate_pass_rate":
            float(np.mean(physical["coordinate"])),
        "topology_pass_rate":
            float(np.mean(physical["topology"])),
        "historical_segment_pass_rate":
            float(np.mean(physical["segment"])),
        "normalized_movement_rms":
            _safe_stats(
                np.sqrt(
                    np.mean(
                        delta_active * delta_active,
                        axis=1,
                    )
                )
            ),
        "normalized_target_residual_rms":
            _safe_stats(
                np.sqrt(
                    np.mean(
                        residual_active
                        * residual_active,
                        axis=1,
                    )
                )
            ),
        "target_direction_cosine":
            _safe_stats(
                _row_cosine(
                    delta_active,
                    target_active,
                    epsilon=epsilon,
                )
            ),
        "target_distance_reduction_fraction":
            _safe_stats(
                1.0
                - (
                    np.sqrt(
                        np.sum(
                            residual_active
                            * residual_active,
                            axis=1,
                        )
                    )
                    / np.maximum(
                        np.sqrt(
                            np.sum(
                                target_active
                                * target_active,
                                axis=1,
                            )
                        ),
                        float(epsilon),
                    )
                )
            ),
        "segment_length": {
            "candidate":
                _safe_stats(candidate_lengths),
            "control":
                _safe_stats(control_lengths),
            "target":
                _safe_stats(target_lengths),
            "candidate_to_target_ratio":
                _safe_stats(
                    candidate_lengths
                    / np.maximum(
                        target_lengths,
                        float(epsilon),
                    )
                ),
            "candidate_to_control_ratio":
                _safe_stats(
                    candidate_lengths
                    / np.maximum(
                        control_lengths,
                        float(epsilon),
                    )
                ),
            "collapse_fraction":
                float(np.mean(collapse)),
            "stretch_fraction":
                float(np.mean(stretch)),
        },
        "output_sha256":
            sha256_array(candidate),
    }


def target_line_audit(
    *,
    control: np.ndarray,
    target: np.ndarray,
    timestep: int,
    context: Mapping[str, Any],
    spec: ReachabilitySpec,
) -> Dict[str, Any]:
    upper_min = float(
        TIMESTEP_UPPER_MIN[int(timestep)]
    )
    alphas = np.linspace(
        0.0,
        1.0,
        int(spec.target_line_grid_points),
        dtype=np.float64,
    )
    records: List[Dict[str, Any]] = []
    row_pass_history: List[np.ndarray] = []
    row_valid_history: List[np.ndarray] = []
    first_aggregate: Optional[Dict[str, Any]] = None
    for alpha in alphas.tolist():
        candidate = (
            np.asarray(control, dtype=np.float64)
            + float(alpha)
            * (
                np.asarray(target, dtype=np.float64)
                - np.asarray(control, dtype=np.float64)
            )
        ).astype(np.float32)
        evaluation = evaluate_output_population(
            candidate,
            control=control,
            target=target,
            groups=context["holdout_groups"],
            condition_name=
                context["holdout_condition_name"],
            target_standardizer=
                context["target_standardizer"],
            objective_contract=
                context["objective_contract"],
            upper_gate=context["upper_gate"],
            stage_d_contract=
                context["stage_d_contract"],
            historical_geometry=
                context["historical_geometry"],
            top_k=spec.top_k,
            epsilon=spec.norm_epsilon,
        )
        scores = staged.segment_scores(
            candidate,
            context["stage_d_contract"].reference,
        )
        row_pass = (
            scores["upper"][:, 0]
            <= float(
                context[
                    "upper_gate"
                ].upper_threshold
            )
        )
        physical = stageb.physical_validity(
            candidate[:, None],
            context["historical_geometry"],
        )
        row_valid = (
            row_pass
            & np.asarray(
                physical["valid"][:, 0],
                dtype=np.bool_,
            )
        )
        row_pass_history.append(row_pass)
        row_valid_history.append(row_valid)
        record = {
            "alpha": float(alpha),
            "upper_row_pass_rate":
                evaluation[
                    "upper_row_pass_rate"
                ],
            "lower_row_pass_rate":
                evaluation[
                    "lower_row_pass_rate"
                ],
            "historical_physical_row_any_rate":
                evaluation[
                    "historical_physical_row_any_rate"
                ],
            "normalized_mse":
                evaluation[
                    "normalized_mse"
                ],
            "normalized_mse_ratio":
                evaluation[
                    "normalized_mse_ratio"
                ],
            "topk_geometry_total":
                evaluation[
                    "topk_upper"
                ]["total"],
            "collapse_fraction":
                evaluation[
                    "segment_length"
                ]["collapse_fraction"],
            "output_sha256":
                evaluation["output_sha256"],
        }
        record["aggregate_upper_fidelity_pass"] = bool(
            record["upper_row_pass_rate"]
            >= upper_min
            and record["normalized_mse_ratio"]
            <= spec.oracle_fidelity_ratio_max
        )
        record["aggregate_valid_pass"] = bool(
            record["aggregate_upper_fidelity_pass"]
            and record[
                "historical_physical_row_any_rate"
            ]
            >= spec.target_line_physical_valid_min
        )
        records.append(record)
        if (
            first_aggregate is None
            and record["aggregate_valid_pass"]
        ):
            first_aggregate = copy.deepcopy(record)

    upper_history = np.stack(
        row_pass_history,
        axis=1,
    )
    valid_history = np.stack(
        row_valid_history,
        axis=1,
    )
    first_upper_alpha = np.full(
        upper_history.shape[0],
        np.nan,
        dtype=np.float64,
    )
    first_alpha = np.full(
        valid_history.shape[0],
        np.nan,
        dtype=np.float64,
    )
    for row in range(valid_history.shape[0]):
        upper_passed = np.flatnonzero(upper_history[row])
        if upper_passed.size:
            first_upper_alpha[row] = alphas[int(upper_passed[0])]
        valid_passed = np.flatnonzero(valid_history[row])
        if valid_passed.size:
            first_alpha[row] = alphas[int(valid_passed[0])]
    pass_to_fail = (
        valid_history[:, :-1]
        & ~valid_history[:, 1:]
    )
    monotonic_violation_rows = np.any(
        pass_to_fail,
        axis=1,
    )
    finite_alpha = first_alpha[
        np.isfinite(first_alpha)
    ]
    return {
        "timestep": int(timestep),
        "upper_pass_threshold": upper_min,
        "grid_points": int(alphas.size),
        "alpha_step":
            float(alphas[1] - alphas[0]),
        "records": records,
        "first_aggregate_valid":
            first_aggregate,
        "aggregate_line_reachable":
            bool(first_aggregate is not None),
        "row_upper_reachable_rate":
            float(np.mean(np.isfinite(first_upper_alpha))),
        "row_valid_reachable_rate":
            float(np.mean(np.isfinite(first_alpha))),
        "row_reachable_rate":
            float(np.mean(np.isfinite(first_alpha))),
        "row_first_upper_alpha":
            _safe_stats(
                first_upper_alpha[np.isfinite(first_upper_alpha)]
            ),
        "row_first_upper_alpha_sha256":
            sha256_array(first_upper_alpha),
        "row_first_valid_alpha":
            _safe_stats(finite_alpha),
        "row_first_pass_alpha":
            _safe_stats(finite_alpha),
        "row_first_pass_alpha_sha256":
            sha256_array(first_alpha),
        "monotonic_pass_to_fail_row_rate":
            float(
                np.mean(monotonic_violation_rows)
            ),
        "control_anchor":
            records[0],
        "target_anchor":
            records[-1],
        "_row_first_alpha":
            first_alpha,
    }


def direct_x0_geometry_gradient_alignment(
    *,
    control: np.ndarray,
    target: np.ndarray,
    context: Mapping[str, Any],
    spec: ReachabilitySpec,
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
    terms = stagec_topk.topk_quadratic_terms_torch(
        value,
        target_standardizer=
            context["target_standardizer"],
        objective_contract=
            context["objective_contract"],
        top_k=spec.top_k,
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
        context["target_standardizer"].active,
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
    cosine = _row_cosine(
        descent_flat,
        target_flat,
        epsilon=spec.norm_epsilon,
    )
    descent_norm = np.sqrt(
        np.sum(
            descent_flat * descent_flat,
            axis=1,
        )
    )
    target_norm = np.sqrt(
        np.sum(
            target_flat * target_flat,
            axis=1,
        )
    )
    return {
        "geometry_loss":
            float(terms["total"].detach().cpu()),
        "geometry_gradient_sha256":
            sha256_array(
                gradient.detach().cpu().numpy()
            ),
        "descent_target_cosine":
            _safe_stats(cosine),
        "nonnegative_alignment_rate":
            float(
                np.mean(
                    cosine
                    >= spec
                    .geometry_gradient_target_cosine_nonnegative
                )
            ),
        "descent_norm":
            _safe_stats(descent_norm),
        "target_direction_norm":
            _safe_stats(target_norm),
        "zero_descent_row_rate":
            float(
                np.mean(
                    descent_norm
                    <= spec.norm_epsilon
                )
            ),
    }


def _project_normalized_trust_region(
    *,
    value: Any,
    control: Any,
    active_flat: Any,
    radius: float,
    epsilon: float,
) -> None:
    torch, _ = stageb._torch_imports()
    with torch.no_grad():
        delta = (
            value - control
        ).reshape(value.shape[0], -1)
        active = active_flat.to(
            device=value.device
        )
        if bool(torch.any(active)):
            active_delta = delta[:, active]
            rms = torch.sqrt(
                torch.mean(
                    active_delta
                    * active_delta,
                    dim=1,
                )
                + float(epsilon)
            )
            scale = torch.clamp(
                float(radius) / rms,
                max=1.0,
            )
            delta[:, active] = (
                active_delta
                * scale[:, None]
            )
            delta[:, ~active] = 0.0
        else:
            rms = torch.sqrt(
                torch.mean(
                    delta * delta,
                    dim=1,
                )
                + float(epsilon)
            )
            scale = torch.clamp(
                float(radius) / rms,
                max=1.0,
            )
            delta[:] = (
                delta
                * scale[:, None]
            )
        value.copy_(
            control
            + delta.reshape_as(value)
        )


def optimize_direct_x0_oracle(
    *,
    control: np.ndarray,
    target: np.ndarray,
    radius: float,
    context: Mapping[str, Any],
    spec: ReachabilitySpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    optimizer = torch.optim.Adam(
        [value],
        lr=max(
            float(radius)
            * spec.oracle_learning_rate_scale,
            spec.oracle_min_learning_rate,
        ),
    )
    active_flat = torch.as_tensor(
        np.asarray(
            context["target_standardizer"].active,
            dtype=np.bool_,
        ).reshape(-1),
        dtype=torch.bool,
        device=device,
    )
    checkpoint_records: MutableMapping[
        str,
        Any,
    ] = {}
    loss_history: List[float] = []

    def snapshot(step: int) -> None:
        raw = (
            context["target_standardizer"]
            .denormalize(
                value.detach().cpu().numpy()
            )
            .astype(np.float32)
        )
        evaluation = evaluate_output_population(
            raw,
            control=control,
            target=target,
            groups=context["holdout_groups"],
            condition_name=
                context["holdout_condition_name"],
            target_standardizer=
                context["target_standardizer"],
            objective_contract=
                context["objective_contract"],
            upper_gate=context["upper_gate"],
            stage_d_contract=
                context["stage_d_contract"],
            historical_geometry=
                context["historical_geometry"],
            top_k=spec.top_k,
            epsilon=spec.norm_epsilon,
        )
        checkpoint_records[str(step)] = {
            "step": int(step),
            "evaluation": evaluation,
        }

    snapshot(0)
    for step in range(1, spec.oracle_steps + 1):
        optimizer.zero_grad(
            set_to_none=True
        )
        terms = (
            stagec_topk
            .topk_quadratic_terms_torch(
                value,
                target_standardizer=
                    context[
                        "target_standardizer"
                    ],
                objective_contract=
                    context[
                        "objective_contract"
                    ],
                top_k=spec.top_k,
            )
        )
        loss = terms["total"]
        if not bool(
            torch.isfinite(loss)
            .detach()
            .cpu()
        ):
            raise ReachabilityAuditError(
                "direct-x0 oracle loss is not finite"
            )
        loss.backward()
        gradient = value.grad
        if gradient is None or not bool(
            torch.all(torch.isfinite(gradient))
            .detach()
            .cpu()
        ):
            raise ReachabilityAuditError(
                "direct-x0 oracle gradient is not finite"
            )
        optimizer.step()
        _project_normalized_trust_region(
            value=value,
            control=control_z,
            active_flat=active_flat,
            radius=float(radius),
            epsilon=spec.norm_epsilon,
        )
        loss_history.append(
            float(loss.detach().cpu())
        )
        if step in spec.oracle_checkpoints:
            snapshot(step)

    final_raw = (
        context["target_standardizer"]
        .denormalize(
            value.detach().cpu().numpy()
        )
        .astype(np.float32)
    )
    final = checkpoint_records[
        str(spec.oracle_steps)
    ]["evaluation"]
    return final_raw, {
        "radius": float(radius),
        "steps": int(spec.oracle_steps),
        "learning_rate": float(
            max(
                float(radius)
                * spec.oracle_learning_rate_scale,
                spec.oracle_min_learning_rate,
            )
        ),
        "loss_history_sha256":
            sha256_array(
                np.asarray(
                    loss_history,
                    dtype=np.float64,
                )
            ),
        "loss_first":
            float(loss_history[0]),
        "loss_final":
            float(loss_history[-1]),
        "loss_finite":
            bool(
                np.all(
                    np.isfinite(
                        np.asarray(loss_history)
                    )
                )
            ),
        "checkpoints":
            dict(checkpoint_records),
        "final": final,
        "oracle_tensor_persisted": False,
    }


def direct_x0_oracle_sweep(
    *,
    control: np.ndarray,
    target: np.ndarray,
    timestep: int,
    context: Mapping[str, Any],
    spec: ReachabilitySpec,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    internal_outputs: Dict[str, np.ndarray] = {}
    upper_min = float(
        TIMESTEP_UPPER_MIN[int(timestep)]
    )
    for radius in spec.oracle_radii:
        output, record = (
            optimize_direct_x0_oracle(
                control=control,
                target=target,
                radius=float(radius),
                context=context,
                spec=spec,
            )
        )
        final = record["final"]
        record["upper_fidelity_pass"] = bool(
            final["upper_row_pass_rate"]
            >= upper_min
            and final["normalized_mse_ratio"]
            <= spec.oracle_fidelity_ratio_max
        )
        record["cable_valid_pass"] = bool(
            record["upper_fidelity_pass"]
            and final["lower_row_pass_rate"]
            >= spec.oracle_lower_pass_min
            and final[
                "historical_physical_row_any_rate"
            ]
            >= spec.oracle_historical_valid_min
            and final["segment_length"][
                "collapse_fraction"
            ]
            <= spec.oracle_collapse_fraction_max
        )
        key = "{:.3f}".format(float(radius))
        internal_outputs[key] = output
        records.append(record)

    best_geometry = min(
        records,
        key=lambda item: (
            float(
                item["final"][
                    "topk_upper"
                ]["total"]
            ),
            float(
                item["final"][
                    "normalized_mse_ratio"
                ]
            ),
            float(item["radius"]),
        ),
    )
    valid = [
        item
        for item in records
        if item["cable_valid_pass"]
    ]
    best_valid = (
        min(
            valid,
            key=lambda item: (
                float(item["radius"]),
                float(
                    item["final"][
                        "normalized_mse_ratio"
                    ]
                ),
            ),
        )
        if valid
        else None
    )
    upper_fidelity = [
        item
        for item in records
        if item["upper_fidelity_pass"]
    ]
    best_upper_fidelity = (
        min(
            upper_fidelity,
            key=lambda item: (
                float(item["radius"]),
                -float(
                    item["final"][
                        "lower_row_pass_rate"
                    ]
                ),
            ),
        )
        if upper_fidelity
        else None
    )
    return {
        "timestep": int(timestep),
        "upper_pass_threshold":
            upper_min,
        "records": records,
        "best_geometry_radius":
            float(best_geometry["radius"]),
        "best_geometry_record":
            best_geometry,
        "best_upper_fidelity_record":
            best_upper_fidelity,
        "best_cable_valid_record":
            best_valid,
        "upper_fidelity_reachable":
            bool(best_upper_fidelity is not None),
        "cable_valid_reachable":
            bool(best_valid is not None),
        "_outputs": internal_outputs,
    }


def deterministic_tangent_rows(
    *,
    eligible: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    count: int,
) -> np.ndarray:
    mask = np.asarray(eligible, dtype=np.bool_)
    group_values = np.asarray(groups).astype(str)
    conditions = np.asarray(condition_name).astype(str)
    if mask.shape != group_values.shape or mask.shape != conditions.shape:
        raise ValueError("tangent row metadata shape changed")
    by_condition: MutableMapping[str, List[int]] = {}
    for name in sorted(set(conditions[mask].tolist())):
        indices = np.flatnonzero(mask & (conditions == name))
        ordered = sorted(
            indices.tolist(),
            key=lambda index: (
                group_values[index],
                int(index),
            ),
        )
        by_condition[name] = ordered
    selected: List[int] = []
    names = sorted(by_condition)
    while len(selected) < int(count):
        changed = False
        for name in names:
            if by_condition[name]:
                selected.append(
                    by_condition[name].pop(0)
                )
                changed = True
                if len(selected) >= int(count):
                    break
        if not changed:
            break
    if not selected:
        raise ReachabilityAuditError(
            "no target-reachable control-failure rows "
            "were available for tangent audit"
        )
    return np.asarray(
        selected,
        dtype=np.int64,
    )


def _cosine_flat(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = math.sqrt(
        float(np.dot(a, a))
        * float(np.dot(b, b))
    )
    return float(
        np.dot(a, b)
        / max(denominator, float(epsilon))
    )


def _explained_ratio_along(
    response: np.ndarray,
    desired: np.ndarray,
    *,
    epsilon: float,
) -> Dict[str, float]:
    v = np.asarray(
        response,
        dtype=np.float64,
    ).reshape(-1)
    d = np.asarray(
        desired,
        dtype=np.float64,
    ).reshape(-1)
    v_sq = float(np.dot(v, v))
    d_sq = float(np.dot(d, d))
    if d_sq <= float(epsilon):
        return {
            "desired_norm": 0.0,
            "response_norm":
                math.sqrt(max(v_sq, 0.0)),
            "cosine": 0.0,
            "optimal_scale": 0.0,
            "explained_ratio": 0.0,
            "residual_norm": 0.0,
        }
    scale = (
        float(np.dot(v, d))
        / max(v_sq, float(epsilon))
    )
    residual = d - scale * v
    residual_sq = float(
        np.dot(residual, residual)
    )
    explained = 1.0 - residual_sq / d_sq
    return {
        "desired_norm":
            math.sqrt(max(d_sq, 0.0)),
        "response_norm":
            math.sqrt(max(v_sq, 0.0)),
        "cosine":
            _cosine_flat(
                v,
                d,
                epsilon=epsilon,
            ),
        "optimal_scale":
            float(scale),
        "explained_ratio":
            float(explained),
        "residual_norm":
            math.sqrt(max(residual_sq, 0.0)),
    }


def _parameter_group_response(
    *,
    model: Any,
    named_parameters: Sequence[Tuple[str, Any]],
    gradients: Sequence[Any],
    parameter_indices: Sequence[int],
    forward_fn: Any,
    desired_selected: np.ndarray,
    selected_rows: np.ndarray,
    active_flat: np.ndarray,
    relative_epsilons: Sequence[float],
    epsilon: float,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    indices = list(parameter_indices)
    if not indices:
        raise ValueError("parameter group is empty")
    parameter_sq = 0.0
    gradient_sq = 0.0
    for index in indices:
        parameter = named_parameters[index][1]
        gradient = gradients[index]
        parameter_sq += float(
            torch.sum(
                parameter.detach()
                * parameter.detach()
            ).cpu()
        )
        gradient_sq += float(
            torch.sum(
                gradient.detach()
                * gradient.detach()
            ).cpu()
        )
    parameter_norm = math.sqrt(
        max(parameter_sq, 0.0)
    )
    gradient_norm = math.sqrt(
        max(gradient_sq, 0.0)
    )
    if gradient_norm <= float(epsilon):
        return {
            "parameter_count": int(
                sum(
                    named_parameters[index][1].numel()
                    for index in indices
                )
            ),
            "parameter_norm":
                parameter_norm,
            "vjp_gradient_norm": 0.0,
            "finite_difference_records": [],
            "authoritative_record": None,
            "response_stability_min_cosine": 0.0,
            "zero_vjp_gradient": True,
        }

    originals = {
        index: (
            named_parameters[index][1]
            .detach()
            .clone()
        )
        for index in indices
    }
    records: List[Dict[str, Any]] = []
    responses: List[np.ndarray] = []
    try:
        for relative in relative_epsilons:
            step = (
                float(relative)
                * max(parameter_norm, 1.0)
                / max(
                    gradient_norm,
                    float(epsilon),
                )
            )
            with torch.no_grad():
                for index in indices:
                    named_parameters[index][1].copy_(
                        originals[index]
                        + step * gradients[index]
                    )
            with torch.no_grad():
                plus = (
                    forward_fn()
                    .detach()
                    .cpu()
                    .numpy()
                )
            with torch.no_grad():
                for index in indices:
                    named_parameters[index][1].copy_(
                        originals[index]
                        - step * gradients[index]
                    )
            with torch.no_grad():
                minus = (
                    forward_fn()
                    .detach()
                    .cpu()
                    .numpy()
                )
            with torch.no_grad():
                for index in indices:
                    named_parameters[index][1].copy_(
                        originals[index]
                    )
            response = (
                plus - minus
            ) / (2.0 * step)
            response_selected = response[
                selected_rows
            ].reshape(
                selected_rows.shape[0],
                -1,
            )
            response_active = response_selected[
                :,
                active_flat,
            ]
            metrics = _explained_ratio_along(
                response_active,
                desired_selected,
                epsilon=epsilon,
            )
            responses.append(
                response_active.astype(
                    np.float64
                )
            )
            records.append(
                {
                    "relative_epsilon":
                        float(relative),
                    "parameter_step":
                        float(step),
                    "response_sha256":
                        sha256_array(
                            response_active.astype(
                                np.float64
                            )
                        ),
                    **metrics,
                }
            )
    finally:
        with torch.no_grad():
            for index in indices:
                named_parameters[index][1].copy_(
                    originals[index]
                )

    pairwise: List[float] = []
    for left in range(len(responses)):
        for right in range(left + 1, len(responses)):
            pairwise.append(
                _cosine_flat(
                    responses[left],
                    responses[right],
                    epsilon=epsilon,
                )
            )
    authoritative_index = len(records) // 2
    return {
        "parameter_count": int(
            sum(
                named_parameters[index][1].numel()
                for index in indices
            )
        ),
        "parameter_norm":
            parameter_norm,
        "vjp_gradient_norm":
            gradient_norm,
        "finite_difference_records":
            records,
        "authoritative_record":
            records[authoritative_index],
        "response_stability_min_cosine": (
            float(min(pairwise))
            if pairwise
            else 1.0
        ),
        "zero_vjp_gradient": False,
    }


def _capture_output_hidden(
    *,
    model: Any,
    forward_fn: Any,
) -> Tuple[Any, np.ndarray]:
    captured: Dict[str, Any] = {}

    def hook(
        _module: Any,
        inputs: Tuple[Any, ...],
    ) -> None:
        if not inputs:
            raise ReachabilityAuditError(
                "output-head hook received no input"
            )
        captured["hidden"] = (
            inputs[0].detach().cpu().numpy()
        )

    handle = model.output.register_forward_pre_hook(
        hook
    )
    try:
        output = forward_fn()
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise ReachabilityAuditError(
            "output-head hidden activation was not captured"
        )
    return output, np.asarray(
        captured["hidden"],
        dtype=np.float32,
    )


def output_head_linear_reachability(
    *,
    model: Any,
    forward_fn: Any,
    control_z: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    desired_z: np.ndarray,
    selected_rows: np.ndarray,
    context: Mapping[str, Any],
    spec: ReachabilitySpec,
) -> Dict[str, Any]:
    """Estimate output-head capacity with grouped cross-validation.

    A same-row least-squares fit is not an admissible capacity test because the
    hidden dimension exceeds the selected-row count.  The authoritative metric
    is therefore a deterministic four-fold group-held-out prediction.  The
    full-data fit is retained only as an in-sample upper bound.
    """
    output, hidden = _capture_output_hidden(
        model=model,
        forward_fn=forward_fn,
    )
    observed = output.detach().cpu().numpy()
    if observed.shape != control_z.shape:
        raise ReachabilityAuditError(
            "output-head audit control shape changed"
        )
    selected_hidden = hidden[selected_rows].astype(np.float64)
    selected_desired = desired_z[selected_rows].reshape(
        selected_rows.shape[0],
        -1,
    ).astype(np.float64)
    design = np.concatenate(
        [
            selected_hidden,
            np.ones(
                (selected_hidden.shape[0], 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )
    selected_groups = np.asarray(
        context["holdout_groups"]
    ).astype(str)[selected_rows]
    unique_groups = sorted(set(selected_groups.tolist()))
    if len(unique_groups) < 4:
        raise ReachabilityAuditError(
            "output-head grouped cross-validation has fewer than four groups"
        )
    fold_map = {
        group: index % 4
        for index, group in enumerate(unique_groups)
    }
    fold_assignment = np.asarray(
        [fold_map[group] for group in selected_groups],
        dtype=np.int64,
    )
    cross_validated = np.zeros_like(selected_desired)
    assigned = np.zeros(selected_rows.size, dtype=np.bool_)
    fold_records: List[Dict[str, Any]] = []
    relative_updates: List[float] = []
    current_weight = (
        model.output.weight.detach()
        .cpu().numpy().astype(np.float64)
    )
    current_bias = (
        model.output.bias.detach()
        .cpu().numpy().astype(np.float64)
    )
    current_norm = math.sqrt(
        float(
            np.sum(current_weight * current_weight)
            + np.sum(current_bias * current_bias)
        )
    )
    for fold in range(4):
        test_mask = fold_assignment == fold
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise ReachabilityAuditError(
                "output-head grouped cross-validation fold is empty"
            )
        solution, residuals, rank, singular = np.linalg.lstsq(
            design[train_mask],
            selected_desired[train_mask],
            rcond=spec.output_head_rcond,
        )
        predicted_test = design[test_mask] @ solution
        cross_validated[test_mask] = predicted_test
        assigned[test_mask] = True
        weight_delta = solution[:-1].T
        bias_delta = solution[-1]
        delta_norm = math.sqrt(
            float(
                np.sum(weight_delta * weight_delta)
                + np.sum(bias_delta * bias_delta)
            )
        )
        relative_update = delta_norm / max(
            current_norm,
            spec.norm_epsilon,
        )
        relative_updates.append(relative_update)
        fold_metrics = _explained_ratio_along(
            predicted_test,
            selected_desired[test_mask],
            epsilon=spec.norm_epsilon,
        )
        fold_records.append(
            {
                "fold": int(fold),
                "train_rows": int(np.sum(train_mask)),
                "test_rows": int(np.sum(test_mask)),
                "train_group_count": int(
                    len(set(selected_groups[train_mask].tolist()))
                ),
                "test_group_count": int(
                    len(set(selected_groups[test_mask].tolist()))
                ),
                "design_rank": int(rank),
                "singular_values": _safe_stats(singular),
                "least_squares_residual_sum": (
                    float(np.sum(residuals))
                    if residuals.size
                    else 0.0
                ),
                "relative_parameter_update_norm": float(relative_update),
                "predicted_displacement_sha256": sha256_array(
                    predicted_test
                ),
                **fold_metrics,
            }
        )
    if not np.all(assigned):
        raise ReachabilityAuditError(
            "output-head cross-validation did not assign every row"
        )
    cross_validated_metrics = _explained_ratio_along(
        cross_validated,
        selected_desired,
        epsilon=spec.norm_epsilon,
    )

    full_solution, full_residuals, full_rank, full_singular = np.linalg.lstsq(
        design,
        selected_desired,
        rcond=spec.output_head_rcond,
    )
    full_predicted = design @ full_solution
    in_sample_metrics = _explained_ratio_along(
        full_predicted,
        selected_desired,
        epsilon=spec.norm_epsilon,
    )

    candidate_z = np.asarray(control_z, dtype=np.float32).copy()
    candidate_z[selected_rows] = (
        candidate_z[selected_rows]
        + cross_validated.reshape(
            selected_rows.shape[0],
            stageb.FUTURE_STEPS,
            stageb.CABLE_DIM,
        ).astype(np.float32)
    )
    candidate_raw = (
        context["target_standardizer"]
        .denormalize(candidate_z)
        .astype(np.float32)
    )
    evaluation = evaluate_output_population(
        candidate_raw,
        control=control,
        target=target,
        groups=context["holdout_groups"],
        condition_name=context["holdout_condition_name"],
        target_standardizer=context["target_standardizer"],
        objective_contract=context["objective_contract"],
        upper_gate=context["upper_gate"],
        stage_d_contract=context["stage_d_contract"],
        historical_geometry=context["historical_geometry"],
        top_k=spec.top_k,
        epsilon=spec.norm_epsilon,
    )
    return {
        "selected_rows": int(selected_rows.size),
        "selected_group_count": int(len(unique_groups)),
        "cross_validation_folds": 4,
        "fold_assignment_sha256": sha256_array(fold_assignment),
        "fold_records": fold_records,
        "design_shape": list(design.shape),
        "desired_shape": list(selected_desired.shape),
        "cross_validated_predicted_displacement_sha256": sha256_array(
            cross_validated
        ),
        "predicted_displacement_sha256": sha256_array(cross_validated),
        "relative_parameter_update_norm": float(max(relative_updates)),
        "relative_parameter_update_norm_stats": _safe_stats(
            np.asarray(relative_updates, dtype=np.float64)
        ),
        "in_sample_upper_bound": {
            "design_rank": int(full_rank),
            "singular_values": _safe_stats(full_singular),
            "least_squares_residual_sum": (
                float(np.sum(full_residuals))
                if full_residuals.size
                else 0.0
            ),
            "predicted_displacement_sha256": sha256_array(full_predicted),
            **in_sample_metrics,
        },
        **cross_validated_metrics,
        "evaluation": evaluation,
        "model_weights_modified": False,
        "authoritative_metric": "four-fold grouped cross-validated explained ratio",
    }

def iterative_global_tangent_reachability(
    *,
    model: Any,
    named_parameters: Sequence[Tuple[str, Any]],
    forward_fn: Any,
    desired_z: np.ndarray,
    selected_rows: np.ndarray,
    active_flat: np.ndarray,
    spec: ReachabilitySpec,
) -> Dict[str, Any]:
    """Greedy Krylov lower bound on local tangent-space reachability.

    Each iteration forms a VJP from the current output residual, then estimates
    the corresponding J(J^T residual) response by central finite differences.
    Orthonormalized responses define a monotonically expanding output subspace.
    The final explained ratio is a lower bound, not a proof of full-Jacobian
    rank or unreachable capacity.
    """
    torch, _ = stageb._torch_imports()
    device = next(model.parameters()).device
    parameters = [
        parameter
        for _name, parameter in named_parameters
    ]
    desired_selected = desired_z[selected_rows].reshape(
        selected_rows.size,
        -1,
    )[:, active_flat].astype(np.float64)
    desired_vector = desired_selected.reshape(-1)
    desired_sq = float(np.dot(desired_vector, desired_vector))
    if desired_sq <= spec.norm_epsilon:
        return {
            "iterations_requested": int(spec.tangent_krylov_iterations),
            "iterations_completed": 0,
            "records": [],
            "final_explained_ratio": 0.0,
            "final_residual_norm": 0.0,
            "desired_norm": 0.0,
            "stopped_reason": "zero_desired_displacement",
            "model_restored_exact": True,
        }

    parameter_sq = sum(
        float(
            torch.sum(
                parameter.detach() * parameter.detach()
            ).cpu()
        )
        for parameter in parameters
    )
    parameter_norm = math.sqrt(max(parameter_sq, 0.0))
    relative_epsilon = spec.tangent_relative_epsilons[
        len(spec.tangent_relative_epsilons) // 2
    ]
    basis: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []
    projection = np.zeros_like(desired_vector)
    previous_explained = 0.0
    model_sha_before = stageb.tensor_state_sha256(model)
    stopped_reason = "iteration_limit"

    for iteration in range(1, spec.tangent_krylov_iterations + 1):
        residual = desired_vector - projection
        residual_selected = residual.reshape(desired_selected.shape)
        residual_full = np.zeros_like(desired_z, dtype=np.float32)
        residual_full_flat = residual_full.reshape(residual_full.shape[0], -1)
        active_indices = np.flatnonzero(active_flat)
        residual_full_flat[np.ix_(selected_rows, active_indices)] = (
            residual_selected.astype(np.float32)
        )
        residual_tensor = torch.as_tensor(
            residual_full,
            dtype=torch.float32,
            device=device,
        )
        output = forward_fn()
        inner = torch.sum(output * residual_tensor)
        gradients_raw = torch.autograd.grad(
            inner,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        gradients = [
            (
                torch.zeros_like(parameter)
                if gradient is None
                else gradient.detach()
            )
            for parameter, gradient
            in zip(parameters, gradients_raw)
        ]
        gradient_sq = sum(
            float(
                torch.sum(gradient * gradient).cpu()
            )
            for gradient in gradients
        )
        gradient_norm = math.sqrt(max(gradient_sq, 0.0))
        if gradient_norm <= spec.norm_epsilon:
            stopped_reason = "zero_vjp_gradient"
            break
        step = (
            float(relative_epsilon)
            * max(parameter_norm, 1.0)
            / max(gradient_norm, spec.norm_epsilon)
        )
        originals = [
            parameter.detach().clone()
            for parameter in parameters
        ]
        try:
            with torch.no_grad():
                for parameter, original, gradient in zip(
                    parameters,
                    originals,
                    gradients,
                ):
                    parameter.copy_(original + step * gradient)
                plus = forward_fn().detach().cpu().numpy()
                for parameter, original, gradient in zip(
                    parameters,
                    originals,
                    gradients,
                ):
                    parameter.copy_(original - step * gradient)
                minus = forward_fn().detach().cpu().numpy()
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
        response = (plus - minus) / (2.0 * step)
        response_vector = response[selected_rows].reshape(
            selected_rows.size,
            -1,
        )[:, active_flat].astype(np.float64).reshape(-1)
        raw_response_norm = math.sqrt(
            max(float(np.dot(response_vector, response_vector)), 0.0)
        )
        orthogonal = response_vector.copy()
        for vector in basis:
            orthogonal -= float(np.dot(vector, orthogonal)) * vector
        orthogonal_norm = math.sqrt(
            max(float(np.dot(orthogonal, orthogonal)), 0.0)
        )
        if orthogonal_norm <= spec.norm_epsilon:
            stopped_reason = "dependent_response_direction"
            break
        basis.append(orthogonal / orthogonal_norm)
        projection = sum(
            float(np.dot(vector, desired_vector)) * vector
            for vector in basis
        )
        residual_after = desired_vector - projection
        residual_sq = float(np.dot(residual_after, residual_after))
        explained = float(1.0 - residual_sq / desired_sq)
        improvement = explained - previous_explained
        records.append(
            {
                "iteration": int(iteration),
                "relative_epsilon": float(relative_epsilon),
                "parameter_step": float(step),
                "vjp_gradient_norm": float(gradient_norm),
                "raw_response_norm": float(raw_response_norm),
                "orthogonal_response_norm": float(orthogonal_norm),
                "response_target_cosine": _cosine_flat(
                    response_vector,
                    desired_vector,
                    epsilon=spec.norm_epsilon,
                ),
                "response_residual_cosine": _cosine_flat(
                    response_vector,
                    residual,
                    epsilon=spec.norm_epsilon,
                ),
                "explained_ratio": explained,
                "explained_ratio_improvement": float(improvement),
                "residual_norm": math.sqrt(max(residual_sq, 0.0)),
                "response_sha256": sha256_array(response_vector),
                "basis_vector_sha256": sha256_array(basis[-1]),
            }
        )
        previous_explained = explained
        if improvement < spec.tangent_krylov_min_improvement:
            stopped_reason = "minimum_improvement"
            break

    model_sha_after = stageb.tensor_state_sha256(model)
    if model_sha_after != model_sha_before:
        raise ReachabilityAuditError(
            "iterative tangent audit modified model weights"
        )
    final_residual = desired_vector - projection
    final_residual_sq = float(np.dot(final_residual, final_residual))
    return {
        "iterations_requested": int(spec.tangent_krylov_iterations),
        "iterations_completed": int(len(records)),
        "relative_epsilon": float(relative_epsilon),
        "records": records,
        "final_explained_ratio": float(
            1.0 - final_residual_sq / desired_sq
        ),
        "final_residual_norm": math.sqrt(max(final_residual_sq, 0.0)),
        "desired_norm": math.sqrt(max(desired_sq, 0.0)),
        "stopped_reason": stopped_reason,
        "model_sha256_before": model_sha_before,
        "model_sha256_after": model_sha_after,
        "model_restored_exact": True,
        "interpretation": (
            "greedy VJP-finite-difference Krylov lower bound; "
            "not a full Jacobian rank proof"
        ),
    }


def tangent_reachability_audit(
    *,
    timestep: int,
    control: np.ndarray,
    target: np.ndarray,
    target_line: Mapping[str, Any],
    context: Mapping[str, Any],
    spec: ReachabilitySpec,
) -> Dict[str, Any]:
    torch, _ = stageb._torch_imports()
    model = context["model"]
    device = torch.device("cuda:0")
    control_z = (
        context["target_standardizer"]
        .normalize(control)
        .astype(np.float32)
    )
    target_z = (
        context["target_standardizer"]
        .normalize(target)
        .astype(np.float32)
    )
    first_alpha = np.asarray(
        target_line["_row_first_alpha"],
        dtype=np.float64,
    )
    control_scores = staged.segment_scores(
        control,
        context["stage_d_contract"].reference,
    )
    target_scores = staged.segment_scores(
        target,
        context["stage_d_contract"].reference,
    )
    eligible = (
        (
            control_scores["upper"][:, 0]
            > float(
                context[
                    "upper_gate"
                ].upper_threshold
            )
        )
        & (
            target_scores["upper"][:, 0]
            <= float(
                context[
                    "upper_gate"
                ].upper_threshold
            )
        )
        & np.isfinite(first_alpha)
    )
    selected_rows = deterministic_tangent_rows(
        eligible=eligible,
        groups=context["holdout_groups"],
        condition_name=
            context["holdout_condition_name"],
        count=spec.tangent_rows,
    )
    desired_z = np.zeros_like(
        control_z,
        dtype=np.float32,
    )
    alpha_selected = first_alpha[
        selected_rows
    ].astype(np.float32)
    desired_z[selected_rows] = (
        alpha_selected[:, None, None]
        * (
            target_z[selected_rows]
            - control_z[selected_rows]
        )
    )

    stageb_spec = context["stageb_spec"]
    target_holdout_z = (
        context["target_standardizer"]
        .normalize(
            context["holdout_target"]
        )
        .astype(np.float32)
    )
    noise = stageb._fixed_noise(
        context["holdout_target"].shape,
        seed=(
            stageb_spec.seed
            + stagec_topk.TopKCalibrationSpec()
            .holdout_noise_seed_offset
        ),
    )
    timestep_array = np.full(
        context["holdout_target"].shape[0],
        int(timestep),
        dtype=np.int64,
    )
    scheduler = stageb.scheduler_arrays(
        stageb_spec
    )
    noisy_z = stageb.q_sample_numpy(
        target_holdout_z,
        noise,
        timestep_array,
        scheduler["alpha_bar"],
    )
    noisy_raw = (
        context["target_standardizer"]
        .denormalize(noisy_z)
        .astype(np.float32)
    )
    noisy_z = (
        context["target_standardizer"]
        .normalize(noisy_raw)
        .astype(np.float32)
    )
    condition_z = (
        context["condition_standardizer"]
        .normalize(
            context["holdout_condition"]
        )
        .astype(np.float32)
    )
    noisy_tensor = torch.as_tensor(
        noisy_z,
        dtype=torch.float32,
        device=device,
    )
    timestep_tensor = torch.as_tensor(
        timestep_array,
        dtype=torch.long,
        device=device,
    )
    condition_tensor = torch.as_tensor(
        condition_z,
        dtype=torch.float32,
        device=device,
    )

    def forward_fn() -> Any:
        return model(
            noisy_tensor,
            timestep_tensor,
            condition_tensor,
        )

    model.eval()
    model_sha_before = stageb.tensor_state_sha256(
        model
    )
    base_output = forward_fn()
    base_np = (
        base_output.detach().cpu().numpy()
    )
    base_raw = (
        context["target_standardizer"]
        .denormalize(base_np)
        .astype(np.float32)
    )
    maximum_control_difference = float(
        np.max(
            np.abs(
                base_raw
                - np.asarray(
                    control,
                    dtype=np.float32,
                )
            )
        )
    )
    if maximum_control_difference > 5.0e-5:
        raise ReachabilityAuditError(
            "tangent forward differs from frozen control"
        )

    active = np.asarray(
        context["target_standardizer"].active,
        dtype=np.bool_,
    ).reshape(-1)
    desired_selected = desired_z[
        selected_rows
    ].reshape(
        selected_rows.size,
        -1,
    )
    desired_active = desired_selected[
        :,
        active,
    ]
    desired_tensor = torch.as_tensor(
        desired_z,
        dtype=torch.float32,
        device=device,
    )
    active_tensor = torch.as_tensor(
        active,
        dtype=torch.bool,
        device=device,
    )
    inner = torch.sum(
        base_output.reshape(
            base_output.shape[0],
            -1,
        )[:, active_tensor]
        * desired_tensor.reshape(
            desired_tensor.shape[0],
            -1,
        )[:, active_tensor]
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
    gradients_raw = torch.autograd.grad(
        inner,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    gradients = [
        (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient.detach()
        )
        for parameter, gradient
        in zip(
            parameters,
            gradients_raw,
        )
    ]
    block_map = stagea258.parameter_block_map(
        named_parameters
    )
    groups: MutableMapping[
        str,
        List[int],
    ] = {
        "global": list(
            range(len(named_parameters))
        ),
        **block_map,
    }
    response_records: MutableMapping[
        str,
        Any,
    ] = {}
    total_gradient_sq = sum(
        float(
            torch.sum(
                gradient * gradient
            ).cpu()
        )
        for gradient in gradients
    )
    for name, indices in groups.items():
        record = _parameter_group_response(
            model=model,
            named_parameters=
                named_parameters,
            gradients=gradients,
            parameter_indices=indices,
            forward_fn=forward_fn,
            desired_selected=
                desired_active,
            selected_rows=
                selected_rows,
            active_flat=active,
            relative_epsilons=
                spec.tangent_relative_epsilons,
            epsilon=spec.norm_epsilon,
        )
        group_gradient_sq = sum(
            float(
                torch.sum(
                    gradients[index]
                    * gradients[index]
                ).cpu()
            )
            for index in indices
        )
        record["vjp_gradient_energy_fraction"] = (
            group_gradient_sq
            / max(
                total_gradient_sq,
                spec.norm_epsilon,
            )
        )
        record["finite_difference_stable"] = bool(
            record[
                "response_stability_min_cosine"
            ]
            >= spec
            .tangent_response_stability_cosine_min
            or record["zero_vjp_gradient"]
        )
        response_records[name] = record

    iterative_global = iterative_global_tangent_reachability(
        model=model,
        named_parameters=named_parameters,
        forward_fn=forward_fn,
        desired_z=desired_z,
        selected_rows=selected_rows,
        active_flat=active,
        spec=spec,
    )
    head = output_head_linear_reachability(
        model=model,
        forward_fn=forward_fn,
        control_z=control_z,
        target=target,
        control=control,
        desired_z=desired_z,
        selected_rows=selected_rows,
        context=context,
        spec=spec,
    )
    model_sha_after = stageb.tensor_state_sha256(
        model
    )
    if model_sha_after != model_sha_before:
        raise ReachabilityAuditError(
            "finite-difference tangent audit modified model weights"
        )
    return {
        "timestep": int(timestep),
        "eligible_row_count":
            int(np.sum(eligible)),
        "selected_row_count":
            int(selected_rows.size),
        "selected_row_indices":
            selected_rows.tolist(),
        "selected_row_index_sha256":
            sha256_array(selected_rows),
        "selected_group_sha256":
            stagea.sha256_strings(
                context["holdout_groups"][
                    selected_rows
                ].tolist()
            ),
        "selected_condition_counts": {
            name: int(
                np.sum(
                    context[
                        "holdout_condition_name"
                    ][selected_rows]
                    == name
                )
            )
            for name in sorted(
                set(
                    context[
                        "holdout_condition_name"
                    ][selected_rows].tolist()
                )
            )
        },
        "selected_first_alpha":
            _safe_stats(alpha_selected),
        "desired_displacement_sha256":
            sha256_array(
                desired_z[selected_rows]
            ),
        "base_control_max_abs_difference":
            maximum_control_difference,
        "model_sha256_before":
            model_sha_before,
        "model_sha256_after":
            model_sha_after,
        "model_restored_exact":
            model_sha_before
            == model_sha_after,
        "vjp_gradient_sha256":
            stagea258.gradient_sequence_sha256(
                named_parameters,
                gradients,
            ),
        "parameter_groups":
            dict(response_records),
        "iterative_global_tangent_lower_bound":
            iterative_global,
        "output_head_linear":
            head,
        "model_weights_persisted": False,
    }


def classify_reachability(
    *,
    timestep_records: Mapping[str, Mapping[str, Any]],
    spec: ReachabilitySpec,
) -> Dict[str, Any]:
    ordered = [
        timestep_records[str(value)]
        for value in spec.timesteps
    ]
    target_anchor_pass = [
        bool(
            item["target_line"][
                "target_anchor"
            ]["upper_row_pass_rate"]
            >= TIMESTEP_UPPER_MIN[
                int(item["timestep"])
            ]
            and item["target_line"][
                "target_anchor"
            ][
                "historical_physical_row_any_rate"
            ]
            >= spec.target_line_physical_valid_min
        )
        for item in ordered
    ]
    line_reachable = [
        bool(
            item["target_line"][
                "aggregate_line_reachable"
            ]
        )
        for item in ordered
    ]
    oracle_upper_fidelity = [
        bool(
            item["direct_x0_oracle"][
                "upper_fidelity_reachable"
            ]
        )
        for item in ordered
    ]
    oracle_valid = [
        bool(
            item["direct_x0_oracle"][
                "cable_valid_reachable"
            ]
        )
        for item in ordered
    ]
    nonphysical_shortcut = [
        bool(
            upper and not valid
        )
        for upper, valid
        in zip(
            oracle_upper_fidelity,
            oracle_valid,
        )
    ]
    gradient_cosines = [
        float(
            item[
                "geometry_gradient_alignment"
            ][
                "descent_target_cosine"
            ]["mean"]
        )
        for item in ordered
    ]
    global_explained = []
    head_explained = []
    head_relative_updates = []
    tangent_stable = []
    for item in ordered:
        global_record = item[
            "tangent_reachability"
        ]["parameter_groups"]["global"]
        iterative = item[
            "tangent_reachability"
        ][
            "iterative_global_tangent_lower_bound"
        ]
        global_explained.append(
            float(
                iterative[
                    "final_explained_ratio"
                ]
            )
        )
        head_explained.append(
            float(
                item[
                    "tangent_reachability"
                ][
                    "output_head_linear"
                ][
                    "explained_ratio"
                ]
            )
        )
        head_relative_updates.append(
            float(
                item[
                    "tangent_reachability"
                ][
                    "output_head_linear"
                ][
                    "relative_parameter_update_norm"
                ]
            )
        )
        tangent_stable.append(
            bool(
                global_record[
                    "finite_difference_stable"
                ]
            )
        )

    target_anchor_all = bool(
        all(target_anchor_pass)
    )
    line_all = bool(all(line_reachable))
    valid_count = int(sum(oracle_valid))
    upper_fidelity_count = int(
        sum(oracle_upper_fidelity)
    )
    nonphysical_count = int(
        sum(nonphysical_shortcut)
    )
    mean_gradient_cosine = float(
        np.mean(
            np.asarray(
                gradient_cosines,
                dtype=np.float64,
            )
        )
    )
    tangent_low_all = bool(
        all(
            value < spec.tangent_explained_low
            for value in global_explained
        )
    )
    tangent_high_all = bool(
        all(
            value >= spec.tangent_explained_high
            for value in global_explained
        )
    )
    head_high_all = bool(
        all(
            value >= spec.output_head_explained_high
            for value in head_explained
        )
    )
    head_update_all = bool(
        all(
            value <= spec.output_head_relative_update_max
            for value in head_relative_updates
        )
    )
    head_pass_all = bool(
        head_high_all and head_update_all
    )
    tangent_stable_all = bool(all(tangent_stable))

    if not target_anchor_all:
        root = (
            "phase314b_r258_stageb_train_only_target_"
            "fails_frozen_upper_or_physical_gate"
        )
        next_path = (
            "RECALIBRATE_OR_REDEFINE_UPPER_GEOMETRY_"
            "TARGET_BEFORE_MODEL_REPAIR"
        )
        locus = "target_or_gate"
    elif not line_all:
        root = (
            "phase314b_r258_stageb_control_to_target_"
            "line_not_upper_reachable"
        )
        next_path = (
            "AUDIT_UPPER_GATE_TARGET_CONSISTENCY_AND_"
            "NONLINEAR_OUTPUT_REACHABILITY"
        )
        locus = "target_line_reachability"
    elif nonphysical_count > 0:
        root = (
            "phase314b_r258_stageb_upper_only_direct_x0_"
            "oracle_uses_nonphysical_or_fidelity_shortcut"
        )
        next_path = (
            "REDESIGN_BALANCED_LOWER_UPPER_SEGMENT_"
            "GEOMETRY_OBJECTIVE_BEFORE_MODEL_REPAIR"
        )
        locus = "one_sided_objective_shortcut"
    elif valid_count == 0:
        root = (
            "phase314b_r258_stageb_upper_only_direct_x0_"
            "objective_does_not_reach_valid_target_region"
        )
        next_path = (
            "REDESIGN_BALANCED_LOWER_UPPER_SEGMENT_"
            "GEOMETRY_OBJECTIVE_BEFORE_MODEL_REPAIR"
        )
        locus = "objective_direction_or_conditioning"
    elif valid_count == len(ordered):
        if not tangent_stable_all:
            root = (
                "phase314b_r258_stageb_tangent_finite_difference_"
                "numerically_unstable"
            )
            next_path = (
                "CALIBRATE_TANGENT_FINITE_DIFFERENCE_SCALE_"
                "BEFORE_REACHABILITY_CLASSIFICATION"
            )
            locus = "tangent_numerical_stability"
        elif tangent_low_all and not head_pass_all:
            root = (
                "phase314b_r258_stageb_registered_tangent_"
                "krylov_lower_bound_low"
            )
            next_path = (
                "RUN_HIGHER_RANK_JACOBIAN_REACHABILITY_"
                "AUDIT_BEFORE_ARCHITECTURE_CHANGE"
            )
            locus = "registered_tangent_lower_bound"
        elif tangent_high_all and head_pass_all:
            root = (
                "phase314b_r258_stageb_output_and_model_"
                "tangent_reachable_but_training_trajectory_failed"
            )
            next_path = (
                "CALIBRATE_OUTPUT_SPACE_TRUST_REGION_ORACLE_"
                "DISTILLATION_ON_TRAIN_ONLY_SPLIT"
            )
            locus = "optimization_trajectory"
        else:
            root = (
                "phase314b_r258_stageb_valid_output_reachable_"
                "with_mixed_tangent_support"
            )
            next_path = (
                "AUDIT_BLOCKWISE_TANGENT_BOTTLENECKS_BEFORE_"
                "SELECTING_MODEL_OR_OPTIMIZER_REPAIR"
            )
            locus = "mixed_tangent_support"
    else:
        root = (
            "phase314b_r258_stageb_upper_reachability_"
            "depends_on_timestep"
        )
        next_path = (
            "AUDIT_TIMESTEP_CONDITIONAL_TARGET_AND_TANGENT_"
            "REACHABILITY_BEFORE_REPAIR"
        )
        locus = "timestep_conditional"

    return {
        "root_cause": root,
        "required_next_path":
            next_path,
        "primary_failure_locus":
            locus,
        "target_anchor_pass_by_timestep":
            target_anchor_pass,
        "target_anchor_all_pass":
            target_anchor_all,
        "target_line_reachable_by_timestep":
            line_reachable,
        "target_line_all_reachable":
            line_all,
        "oracle_upper_fidelity_by_timestep":
            oracle_upper_fidelity,
        "oracle_valid_by_timestep":
            oracle_valid,
        "oracle_upper_fidelity_count":
            upper_fidelity_count,
        "oracle_valid_count":
            valid_count,
        "nonphysical_shortcut_count":
            nonphysical_count,
        "geometry_descent_target_cosine":
            gradient_cosines,
        "geometry_descent_target_cosine_mean":
            mean_gradient_cosine,
        "global_tangent_explained_ratio":
            global_explained,
        "output_head_explained_ratio":
            head_explained,
        "output_head_relative_update_norm":
            head_relative_updates,
        "global_tangent_stable":
            tangent_stable,
        "global_tangent_stable_all":
            tangent_stable_all,
        "tangent_low_all":
            tangent_low_all,
        "tangent_high_all":
            tangent_high_all,
        "output_head_high_all":
            head_high_all,
        "output_head_update_all":
            head_update_all,
        "output_head_pass_all":
            head_pass_all,
        "selected_configuration": None,
    }


def _public_target_line(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def _public_oracle(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def run_audit(
    *,
    root: Path,
    environment: Mapping[str, Any],
    spec: Optional[ReachabilitySpec] = None,
) -> Dict[str, Any]:
    active_spec = (
        ReachabilitySpec()
        if spec is None
        else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    base = validate_base_evidence(
        repository_root
    )
    source = source_logic_audit(
        repository_root
    )
    if not source["all_confirmed"]:
        raise ReachabilityAuditError(
            "source logic assumptions changed"
        )
    stagea258.validate_environment_payload(
        environment
    )
    cold = (
        stagea258
        .assert_cold_cuda_context_portable()
    )
    captured = capture_portable_control_model(
        root=repository_root
    )
    context = build_audit_context(
        root=repository_root,
        captured=captured,
    )

    timestep_records: MutableMapping[
        str,
        Any,
    ] = {}
    for timestep in active_spec.timesteps:
        control = context[
            "control_predictions"
        ][int(timestep)]
        target = context["holdout_target"]
        target_line_internal = (
            target_line_audit(
                control=control,
                target=target,
                timestep=int(timestep),
                context=context,
                spec=active_spec,
            )
        )
        gradient_alignment = (
            direct_x0_geometry_gradient_alignment(
                control=control,
                target=target,
                context=context,
                spec=active_spec,
            )
        )
        oracle_internal = (
            direct_x0_oracle_sweep(
                control=control,
                target=target,
                timestep=int(timestep),
                context=context,
                spec=active_spec,
            )
        )
        tangent = tangent_reachability_audit(
            timestep=int(timestep),
            control=control,
            target=target,
            target_line=
                target_line_internal,
            context=context,
            spec=active_spec,
        )
        timestep_records[str(timestep)] = {
            "timestep": int(timestep),
            "control_prediction_sha256":
                sha256_array(control),
            "target_sha256":
                sha256_array(target),
            "target_line":
                _public_target_line(
                    target_line_internal
                ),
            "geometry_gradient_alignment":
                gradient_alignment,
            "direct_x0_oracle":
                _public_oracle(
                    oracle_internal
                ),
            "tangent_reachability":
                tangent,
        }

    classification = classify_reachability(
        timestep_records=timestep_records,
        spec=active_spec,
    )
    contract = {
        "schema":
            "phase314b_r258_stageb_reachability_contract_v1",
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "base_identity": {
            "worker_sha256":
                EXPECTED_BASE_WORKER_SHA256,
            "compatibility_sha256":
                EXPECTED_COMPATIBILITY_SHA256,
            "observation_sha256":
                EXPECTED_OBSERVATION_SHA256,
            "stagec_calibration_sha256":
                EXPECTED_STAGEC_CALIBRATION_SHA256,
            "stagec_control_sha256":
                EXPECTED_STAGEC_CONTROL_SHA256,
            "stagea_calibration_sha256":
                EXPECTED_BASE_CALIBRATION_SHA256,
            "stagea_contract_sha256":
                EXPECTED_BASE_CONTRACT_SHA256,
            "stagea_contract_file_sha256":
                EXPECTED_BASE_CONTRACT_FILE_SHA256,
            "stagea_selection_sha256":
                EXPECTED_BASE_SELECTION_SHA256,
        },
        "environment": dict(environment),
        "cold_main_worker_context":
            cold,
        "control_capture":
            captured["control_identity"],
        "source_logic_audit":
            source,
        "reachability_spec":
            asdict(active_spec),
        "target_line_rule": (
            "x(alpha)=control+alpha*(ground_truth-control), "
            "alpha in 101-point [0,1] grid; target is train-only "
            "diagnostic evidence and is never used for model selection"
        ),
        "direct_x0_oracle_rule": (
            "optimize only frozen K16 upper quadratic loss on the "
            "predicted normalized x0 tensor; project every row into "
            "a fixed normalized RMS ball around the control; target "
            "is evaluation-only"
        ),
        "tangent_rule": (
            "VJP direction J^T d followed by deterministic central "
            "finite-difference output response for global and nine "
            "frozen parameter blocks"
        ),
        "output_head_rule": (
            "frozen-hidden exact linear least-squares displacement "
            "fit; model weights are not modified"
        ),
        "model_architecture_changed":
            False,
        "geometry_objective_changed":
            False,
        "new_model_candidate_trained":
            False,
        "uses_frozen_probe": False,
        "runs_reverse_sampling": False,
        "trains_full_stageb_model":
            False,
    }
    contract["contract_sha256"] = (
        sha256_bytes(
            stable_json_bytes(contract)
        )
    )
    selection_payload = {
        "selected_configuration": None,
        "train_only_recommendation": None,
        "classification":
            classification,
        "timestep_records":
            timestep_records,
        "uses_frozen_probe": False,
        "uses_target_for_model_selection":
            False,
        "uses_target_for_diagnostic_reachability":
            True,
    }
    selection_sha = sha256_bytes(
        stable_json_bytes(
            selection_payload
        )
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema":
            "phase314b_r258_stageb_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause":
            classification["root_cause"],
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
        "environment": dict(environment),
        "cold_main_worker_context":
            cold,
        "source_logic_audit":
            source,
        "control_capture": {
            **captured["control_identity"],
            "train_candidate_call_count":
                captured[
                    "train_candidate_call_count"
                ],
            "capture_wrapper_restored":
                captured[
                    "capture_wrapper_restored"
                ],
        },
        "train_view_validation":
            context["arrays_validation"],
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
        "reachability_contract":
            contract,
        "timestep_records":
            dict(timestep_records),
        "classification":
            classification,
        "selection": {
            **selection_payload,
            "selection_sha256":
                selection_sha,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "new_model_candidate_trained":
            False,
        "direct_x0_tensor_optimization_run":
            True,
        "tangent_audit_run": True,
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
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted":
            False,
        "oracle_tensor_persisted":
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
        "candidate_execution": False,
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
        "reachability_contract":
            result[
                "reachability_contract"
            ],
        "timestep_records":
            result["timestep_records"],
        "classification":
            result["classification"],
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
        "target_line_exact": (
            {
                key: value["target_line"]
                for key, value
                in left[
                    "timestep_records"
                ].items()
            }
            == {
                key: value["target_line"]
                for key, value
                in right[
                    "timestep_records"
                ].items()
            }
        ),
        "oracle_exact": (
            {
                key: value[
                    "direct_x0_oracle"
                ]
                for key, value
                in left[
                    "timestep_records"
                ].items()
            }
            == {
                key: value[
                    "direct_x0_oracle"
                ]
                for key, value
                in right[
                    "timestep_records"
                ].items()
            }
        ),
        "tangent_exact": (
            {
                key: value[
                    "tangent_reachability"
                ]
                for key, value
                in left[
                    "timestep_records"
                ].items()
            }
            == {
                key: value[
                    "tangent_reachability"
                ]
                for key, value
                in right[
                    "timestep_records"
                ].items()
            }
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
    }
