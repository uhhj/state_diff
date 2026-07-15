"""Phase3.14b-r2.5.7 Stage-D Resume1 control-trace correction.

The original Stage D stopped before any gated candidate because its exact
control differed from the frozen Stage-C control.  Source attribution shows
that the failed Stage-D worker changed the pre-control CUDA/autograd execution
trace:

* Stage C performed its frozen 64-row ungated K8/K16 lambda calibration before
  the control replay.
* Failed Stage D performed a new 100-row timestep-gated calibration before the
  control replay.

Resetting Python/NumPy/Torch RNG state does not make those two execution traces
the same byte-exact replay contract.  Resume1 therefore reproduces the complete
Stage-C pre-control calibration trace and verifies its calibration SHA before
running the exact control.  Only after the control training, train-control,
one-step, and continuous profiles are exact does Resume1 construct the new
Stage-D stratified calibration batch and run the six registered gated
candidates.

The failed implementation, failed test gate, and failed blocked summary remain
immutable.  The frozen probe, full-874-row model, reverse sampler, formal pilot,
IDM, environment, and CPS remain closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_stagec_reverse_attribution as stagec
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3 import phase314b_r257_stageb_mechanism_audit as stageb_mechanism
from ccda_phase3 import phase314b_r257_stagec_topk_quadratic as stagec_topk
from ccda_phase3 import phase314b_r257_staged_timestep_gate as failed_staged

PHASE = "Phase3.14b-r2.5.7 Stage D Resume1"
PHASE_ID = "phase314b_r257_staged_resume1"

FAILED_IMPLEMENTATION_COMMIT = (
    "ccff30947a85028a2c7974edb5b008f3b3de2757"
)
STAGEC_EVIDENCE_COMMIT = (
    "a002246558e66029bdd6279e09c95e1303e2356c"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

FAILED_BLOCKED = (
    "reports/phase3_14b_r257_staged_blocked_summary.json"
)
FAILED_TEST_GATE = (
    "reports/phase3_14b_r257_staged_test_gate_summary.json"
)
EXPECTED_FAILED_BLOCKED_SHA256 = (
    "bd9f3f63df92605cf7425ebecdcd80cb62c4cc33c9909ee3d51680909057da4f"
)

FAILED_IMPLEMENTATION_FILES = (
    "ccda_phase3/phase314b_r257_staged_timestep_gate.py",
    "scripts/phase3_14b_r257_staged_worker.py",
    "scripts/phase3_14b_r257_staged_run_calibration.py",
    "scripts/phase3_14b_r257_staged_test_gate.py",
    "scripts/phase3_14b_r257_staged_blocked.py",
    "scripts/phase3_14b_r257_staged_run.sh",
    "tests/test_phase3_14b_r257_staged_timestep_gate.py",
)

CONTROL_CANDIDATE_ID = failed_staged.CONTROL_CANDIDATE_ID


class Resume1ControlTraceError(RuntimeError):
    """Raised when the failed provenance or corrected trace changes."""


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
            raise ValueError("non-finite scalar cannot be serialized")
        return value
    raise TypeError(
        "unsupported JSON value: {!r}".format(type(value))
    )


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
            "refusing to overwrite write-once output: {}".format(
                target
            )
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
        raise Resume1ControlTraceError(
            "JSON root is not an object: {}".format(path)
        )
    return value


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
        raise Resume1ControlTraceError(
            "required commit is not an ancestor: {}".format(
                commit
            )
        )


def assert_failed_implementation_bound(
    root: Path,
    relative: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                FAILED_IMPLEMENTATION_COMMIT,
                relative,
            ),
        ],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise Resume1ControlTraceError(
            "failed Stage-D implementation changed: {}".format(
                relative
            )
        )
    return sha256_bytes(observed)


def assert_tracked_current_blob(
    root: Path,
    relative: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    current = subprocess.check_output(
        ["git", "show", "HEAD:{}".format(relative)],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if current != observed:
        raise Resume1ControlTraceError(
            "tracked failed artifact differs from HEAD: {}".format(
                relative
            )
        )
    return sha256_bytes(observed)


def validate_failed_attempt(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    assert_commit_ancestor(
        repository_root,
        STAGEC_EVIDENCE_COMMIT,
    )
    assert_commit_ancestor(
        repository_root,
        FAILED_IMPLEMENTATION_COMMIT,
    )

    implementation_sha = {
        relative: assert_failed_implementation_bound(
            repository_root,
            relative,
        )
        for relative in FAILED_IMPLEMENTATION_FILES
    }
    blocked_sha = assert_tracked_current_blob(
        repository_root,
        FAILED_BLOCKED,
    )
    test_gate_sha = assert_tracked_current_blob(
        repository_root,
        FAILED_TEST_GATE,
    )
    if blocked_sha != EXPECTED_FAILED_BLOCKED_SHA256:
        raise Resume1ControlTraceError(
            "failed blocked-summary SHA changed"
        )

    blocked = load_json(repository_root / FAILED_BLOCKED)
    if blocked.get("verdict") != "BLOCKED":
        raise Resume1ControlTraceError(
            "failed verdict is no longer BLOCKED"
        )
    if blocked.get("scientific_status") != "BLOCKED":
        raise Resume1ControlTraceError(
            "failed scientific status changed"
        )
    if blocked.get("root_cause") != (
        "phase314b_r257_staged_execution_failed_before_completion"
    ):
        raise Resume1ControlTraceError(
            "failed root cause changed"
        )
    if blocked.get("control_replay_exact") is not False:
        raise Resume1ControlTraceError(
            "failed control replay status changed"
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
        if blocked.get(key) is not False:
            raise Resume1ControlTraceError(
                "failed boundary changed: {}".format(key)
            )

    gate = load_json(repository_root / FAILED_TEST_GATE)
    if gate.get("verdict") != "PASS":
        raise Resume1ControlTraceError(
            "failed Stage-D test gate is not PASS"
        )
    if int(gate.get("passed_test_count", -1)) != 749:
        raise Resume1ControlTraceError(
            "failed Stage-D pass count changed"
        )
    if int(gate.get("test_file_count", -1)) != 42:
        raise Resume1ControlTraceError(
            "failed Stage-D test-file count changed"
        )
    if int(gate.get("r257_staged_new_passed", -1)) != 43:
        raise Resume1ControlTraceError(
            "failed Stage-D new-test count changed"
        )

    return {
        "failed_implementation_commit":
            FAILED_IMPLEMENTATION_COMMIT,
        "stagec_evidence_commit":
            STAGEC_EVIDENCE_COMMIT,
        "failed_implementation_sha256":
            implementation_sha,
        "failed_blocked_path": FAILED_BLOCKED,
        "failed_blocked_sha256": blocked_sha,
        "failed_test_gate_path": FAILED_TEST_GATE,
        "failed_test_gate_sha256": test_gate_sha,
        "failed_blocked": blocked,
        "failed_test_gate": gate,
    }


def source_trace_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    failed_text = (
        repository_root
        / "ccda_phase3/"
        "phase314b_r257_staged_timestep_gate.py"
    ).read_text(encoding="utf-8")
    stagec_text = (
        repository_root
        / "ccda_phase3/"
        "phase314b_r257_stagec_topk_quadratic.py"
    ).read_text(encoding="utf-8")

    failed_calibration = failed_text.find(
        "candidates, calibration = calibrate_candidates("
    )
    failed_control = failed_text.find(
        "control_model, control_training, control_diagnostics"
    )
    stagec_calibration = stagec_text.find(
        "candidates, calibration = calibrate_candidates("
    )
    stagec_control = stagec_text.find(
        "model, training, diagnostics = train_candidate("
    )
    checks = {
        "failed_stage_d_calibration_before_control": (
            failed_calibration >= 0
            and failed_control >= 0
            and failed_calibration < failed_control
        ),
        "stage_c_calibration_before_control": (
            stagec_calibration >= 0
            and stagec_control >= 0
            and stagec_calibration < stagec_control
        ),
        "failed_stage_d_uses_stratified_100_row_calibration": (
            "stratified_calibration_batch(" in failed_text
            and "calibration_batch_size: int = 100" in failed_text
        ),
        "stage_c_uses_frozen_64_row_diagnostic_calibration": (
            "stageb_mechanism.fixed_diagnostic_batch(" in stagec_text
            and "calibrate_candidates(" in stagec_text
        ),
        "failed_exact_control_calls_stagec_train_candidate": (
            "stagec_topk.train_candidate(" in failed_text
        ),
        "failed_files_remain_unmodified": True,
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "failed_precontrol_trace":
            "new 100-row gated calibration",
        "frozen_stagec_precontrol_trace":
            "original 64-row ungated K8/K16 calibration",
        "correction":
            "replay frozen Stage-C calibration trace and verify "
            "its calibration identity before exact control; "
            "defer new gated calibration until control exact",
        "causal_claim_scope":
            "execution-trace mismatch established; the exact "
            "differing training field is not inferred without "
            "rerunning the failed contract",
    }


def stagec_control_preamble(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    stagec_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    diagnostic_batch = stageb_mechanism.fixed_diagnostic_batch(
        condition=condition,
        target=target,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=stageb_mechanism.MechanismAuditSpec(),
    )
    candidates, calibration = stagec_topk.calibrate_candidates(
        stageb_spec=stageb_spec,
        diagnostic_batch=diagnostic_batch,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    expected = stagec_contract.get("calibration")
    if calibration != expected:
        raise Resume1ControlTraceError(
            "Stage-C pre-control calibration replay differs"
        )
    return {
        "diagnostic_batch": diagnostic_batch,
        "candidate_count": len(candidates),
        "candidate_order": [
            candidate.candidate_id
            for candidate in candidates
        ],
        "calibration": calibration,
        "calibration_exact": True,
        "calibration_sha256":
            calibration["calibration_sha256"],
    }


def exact_control_replay(
    *,
    condition: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    condition_name: np.ndarray,
    stageb_train: np.ndarray,
    objective_train: np.ndarray,
    selection_holdout: np.ndarray,
    stageb_spec: stageb.DiagnosticSpec,
    condition_standardizer: stageb.ArrayStandardizer,
    target_standardizer: stageb.ArrayStandardizer,
    objective_contract: stagea.UpperObjectiveContract,
    upper_gate: Any,
    stage_d_contract: Any,
    control_expected: Mapping[str, Any],
    stagec_contract: Mapping[str, Any],
    holdout_noise_seed_offset: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    preamble = stagec_control_preamble(
        condition=condition[objective_train],
        target=target[objective_train],
        stageb_spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        stagec_contract=stagec_contract,
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
        diagnostic_batch=preamble["diagnostic_batch"],
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    observed_compatible = (
        stagec_topk.compatible_training_record(training)
    )
    expected_training = control_expected["training"]
    differing_training_keys = sorted(
        key
        for key in set(observed_compatible) | set(expected_training)
        if observed_compatible.get(key) != expected_training.get(key)
    )
    if differing_training_keys:
        raise Resume1ControlTraceError(
            "Resume1 exact-control training differs after "
            "frozen Stage-C preamble: {}".format(
                differing_training_keys
            )
        )

    train_control = stagea.train_control_audit(
        model=model,
        condition=condition[objective_train],
        target=target[objective_train],
        stageb_spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
    )
    if train_control != control_expected["train_control"]:
        raise Resume1ControlTraceError(
            "Resume1 exact-control train-control differs"
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
        historical_geometry=stageb.fit_geometry_contract(
            target[stageb_train]
        ),
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=(
            stageb_spec.seed + int(holdout_noise_seed_offset)
        ),
    )
    if one_step != control_expected["one_step"]:
        raise Resume1ControlTraceError(
            "Resume1 exact-control one-step differs"
        )

    predictions, prediction_sha = stagec.one_step_predictions(
        model=model,
        condition=condition[selection_holdout],
        target=target[selection_holdout],
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=(
            stageb_spec.seed + int(holdout_noise_seed_offset)
        ),
    )
    if prediction_sha != one_step["prediction_sha256"]:
        raise Resume1ControlTraceError(
            "Resume1 control prediction aggregate SHA differs"
        )
    profiles = stagec_topk.holdout_profiles(
        predictions,
        upper_gate=upper_gate,
        objective_contract=objective_contract,
    )
    if profiles != control_expected["holdout_profiles"]:
        raise Resume1ControlTraceError(
            "Resume1 exact-control continuous profiles differ"
        )

    bundle = {
        "candidate_id": CONTROL_CANDIDATE_ID,
        "candidate": None,
        "training": expected_training,
        "training_diagnostics": diagnostics,
        "train_control": train_control,
        "one_step": one_step,
        "holdout_profiles": profiles,
    }
    identity = {
        "precontrol_trace":
            "frozen_stagec_64row_ungated_calibration",
        "preamble_calibration_exact": True,
        "preamble_calibration_sha256":
            preamble["calibration_sha256"],
        "preamble_candidate_order":
            preamble["candidate_order"],
        "training_exact": True,
        "train_control_exact": True,
        "one_step_exact": True,
        "profiles_exact": True,
        "control_model_sha256":
            expected_training["final_model_sha256"],
        "control_optimizer_sha256":
            expected_training["final_optimizer_sha256"],
        "control_loss_sha256":
            expected_training["total_loss_history_sha256"],
        "control_gradient_sha256":
            expected_training["gradient_history_sha256"],
        "control_exposure_sha256":
            expected_training["source_exposure_sha256"],
    }
    del model
    return bundle, identity


def run_resume1(
    *,
    root: Path,
    spec: Optional[failed_staged.TimestepGateSpec] = None,
) -> Dict[str, Any]:
    active_spec = (
        failed_staged.TimestepGateSpec()
        if spec is None
        else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    failed = validate_failed_attempt(repository_root)
    trace = source_trace_audit(repository_root)
    if not trace["all_confirmed"]:
        raise Resume1ControlTraceError(
            "Resume1 source-trace assumptions changed"
        )

    immutable = failed_staged.validate_immutable_inputs(
        repository_root
    )
    stagec_worker_result = immutable["stagec_worker_result"]
    stagec_contract = immutable["stagec_contract"]
    control_expected = immutable["control_record"]
    upstream = stagec_topk.validate_immutable_inputs(
        repository_root
    )
    objective_contract = (
        stageb_mechanism.load_objective_contract(
            upstream["stagea_contract"]
        )
    )
    upper_gate = stagea.load_upper_gate_contract(
        upstream["upstream_immutable"][
            "staged3_contract"
        ]
    )
    stage_d_contract, _ = staged1.load_stage_d_gate(
        repository_root / staged3.STAGE_D_GATE
    )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
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

    stageb_train, frozen_probe, _ = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
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
    expected_split = stagec_worker_result["split"]
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
        if split[key] != expected_split[key]:
            raise Resume1ControlTraceError(
                "train-only split changed: {}".format(key)
            )
    if np.any(
        frozen_probe
        & (objective_train | selection_holdout)
    ):
        raise AssertionError(
            "frozen probe crossed train-only split"
        )

    condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )

    control_bundle, control_identity = exact_control_replay(
        condition=condition,
        target=target,
        groups=groups,
        condition_name=condition_name,
        stageb_train=stageb_train,
        objective_train=objective_train,
        selection_holdout=selection_holdout,
        stageb_spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        upper_gate=upper_gate,
        stage_d_contract=stage_d_contract,
        control_expected=control_expected,
        stagec_contract=stagec_contract,
        holdout_noise_seed_offset=(
            active_spec.holdout_noise_seed_offset
        ),
    )

    # New Stage-D work begins only after the exact control passes.
    calibration_batch = (
        failed_staged.stratified_calibration_batch(
            condition=condition[objective_train],
            target=target[objective_train],
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            stageb_spec=stageb_spec,
            spec=active_spec,
        )
    )
    candidates, calibration = (
        failed_staged.calibrate_candidates(
            stageb_spec=stageb_spec,
            calibration_batch=calibration_batch,
            target_standardizer=target_standardizer,
            objective_contract=objective_contract,
            spec=active_spec,
        )
    )

    records: List[Dict[str, Any]] = []
    for candidate in candidates:
        stageb.set_deterministic_runtime(stageb_spec.seed)
        model, training, diagnostics = (
            failed_staged.train_gated_candidate(
                condition=condition[objective_train],
                target=target[objective_train],
                stageb_spec=stageb_spec,
                objective_contract=objective_contract,
                condition_standardizer=
                    condition_standardizer,
                target_standardizer=
                    target_standardizer,
                candidate=candidate,
                calibration_batch=calibration_batch,
                spec=active_spec,
            )
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
            historical_geometry=stageb.fit_geometry_contract(
                target[stageb_train]
            ),
            condition_standardizer=condition_standardizer,
            target_standardizer=target_standardizer,
            noise_seed=(
                stageb_spec.seed
                + active_spec.holdout_noise_seed_offset
            ),
        )
        predictions, prediction_sha = (
            stagec.one_step_predictions(
                model=model,
                condition=condition[selection_holdout],
                target=target[selection_holdout],
                spec=stageb_spec,
                condition_standardizer=
                    condition_standardizer,
                target_standardizer=
                    target_standardizer,
                noise_seed=(
                    stageb_spec.seed
                    + active_spec.holdout_noise_seed_offset
                ),
            )
        )
        if prediction_sha != one_step["prediction_sha256"]:
            raise Resume1ControlTraceError(
                "candidate prediction aggregate SHA differs"
            )
        profiles = stagec_topk.holdout_profiles(
            predictions,
            upper_gate=upper_gate,
            objective_contract=objective_contract,
        )
        records.append(
            failed_staged.selection_record(
                candidate=candidate,
                training=training,
                training_diagnostics=diagnostics,
                train_control=train_control,
                one_step=one_step,
                profiles=profiles,
                control=control_bundle,
                spec=active_spec,
            )
        )
        del model

    selected = failed_staged.select_configuration(records)
    classification = failed_staged.classify_selection(
        records=records,
        selected=selected,
        spec=active_spec,
    )
    selection_payload = {
        "control": control_bundle,
        "candidate_records": records,
        "selected_configuration": selected,
        "selection_uses_frozen_probe": False,
        "selection_rule": (
            "continuous geometry, fidelity, stability and "
            "binary gates on the frozen train-only holdout"
        ),
    }
    selection_sha = sha256_bytes(
        stable_json_bytes(selection_payload)
    )

    contract = {
        "schema":
            "phase314b_r257_staged_resume1_contract_v1",
        "failed_implementation_commit":
            FAILED_IMPLEMENTATION_COMMIT,
        "failed_blocked_sha256":
            EXPECTED_FAILED_BLOCKED_SHA256,
        "correction_scope":
            "pre-control execution trace only",
        "old_precontrol_trace":
            "new 100-row gated calibration",
        "corrected_precontrol_trace":
            "frozen Stage-C 64-row ungated K8/K16 "
            "calibration, exact calibration identity, "
            "then exact control",
        "control_identity": control_identity,
        "objective_variant":
            "hard_timestep_gated_k16_quadratic_raw_log_excess",
        "top_k": int(active_spec.top_k),
        "timestep_cutoffs":
            list(active_spec.timestep_cutoffs),
        "target_gradient_ratios":
            list(active_spec.target_gradient_ratios),
        "calibration_batch_sha256": {
            key: value
            for key, value in calibration_batch.items()
            if key.endswith("_sha256")
        },
        "calibration": calibration,
        "selection_thresholds": {
            key: value
            for key, value in asdict(active_spec).items()
            if (
                key.endswith("_min")
                or key.endswith("_max")
                or key.endswith("_tolerance")
            )
        },
        "uses_frozen_probe": False,
        "runs_reverse_sampling": False,
        "trains_full_stageb_model": False,
        "failed_files_modified": False,
    }
    contract["contract_sha256"] = sha256_bytes(
        stable_json_bytes(contract)
    )

    return {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema":
            "phase314b_r257_staged_resume1_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path":
            classification["required_next_path"],
        "failed_attempt": failed,
        "source_trace_audit": trace,
        "immutable_stagec": {
            "stagec_worker_identity_sha256":
                immutable["worker_identity_sha256"],
            "stagec_contract_sha256":
                failed_staged.EXPECTED_STAGEC_CONTRACT_SHA256,
            "stagec_contract_file_sha256":
                failed_staged.EXPECTED_STAGEC_CONTRACT_FILE_SHA256,
            "stagec_selection_sha256":
                failed_staged.EXPECTED_STAGEC_SELECTION_SHA256,
        },
        "train_view_validation": validation,
        "split": {
            "stageb_training_rows": int(np.sum(stageb_train)),
            "objective_train_rows":
                int(np.sum(objective_train)),
            "selection_holdout_rows":
                int(np.sum(selection_holdout)),
            "frozen_probe_rows": int(np.sum(frozen_probe)),
            **split,
            "frozen_probe_accessed": False,
        },
        "control_trace_correction": control_identity,
        "calibration_contract": contract,
        "selection": {
            **selection_payload,
            "selection_sha256": selection_sha,
        },
        "classification": classification,
        "selected_configuration": selected,
        "train_only_recommendation": selected,
        "resume1_correction_applied": True,
        "failed_files_modified": False,
        "control_replay_exact": True,
        "new_hyperparameter_candidate_run": True,
        "new_objective_variant_run": True,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "formal_diffusion_training": False,
        "formal_reverse_sampling": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "failed_attempt": {
            "failed_implementation_commit":
                result["failed_attempt"][
                    "failed_implementation_commit"
                ],
            "failed_blocked_sha256":
                result["failed_attempt"][
                    "failed_blocked_sha256"
                ],
            "failed_test_gate_sha256":
                result["failed_attempt"][
                    "failed_test_gate_sha256"
                ],
        },
        "source_trace_audit":
            result["source_trace_audit"],
        "split": result["split"],
        "control_trace_correction":
            result["control_trace_correction"],
        "calibration_contract":
            result["calibration_contract"],
        "selection": result["selection"],
        "classification": result["classification"],
        "selected_configuration":
            result["selected_configuration"],
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
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "control_trace_exact": (
            left["control_trace_correction"]
            == right["control_trace_correction"]
        ),
        "calibration_contract_exact": (
            left["calibration_contract"]
            == right["calibration_contract"]
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
