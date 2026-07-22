"""Phase3.14b-r2.5.8 Stage S Resume2A cold-CUDA process-boundary smoke.

This stage is intentionally execution-only and intentionally small.  Stage-S
Resume1 correctly repaired the historical oracle-rerun field, but its first
confirmation worker performed the CUDA environment probe and the frozen
control replay in the same Python process.  The probe's required-operation
CUDA dry-run initialized that process, so the subsequent portable control
replay correctly failed its cold-CUDA gate.

Resume2A does not run OOF fitting, callbacks, internal-scale attempts, model
training, holdout, reverse, IDM, candidate execution, Phase4, or CPS.  It only
proves the required process topology:

1. a disposable environment-probe subprocess may initialize CUDA and writes a
   temporary environment payload;
2. a new cold subprocess reads that payload and calls the unchanged Stage-O
   ``_prepare_runtime`` entry point, which validates the payload, asserts the
   cold CUDA context, and performs the frozen portable control replay;
3. both subprocesses exit and all temporary payloads are deleted.

Only after this smoke passes may a later stage wire the same boundary into the
two full Stage-S science workers.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage S Resume2A"
SCHEMA = "phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke_v1"
BLOCKED_SCHEMA = (
    "phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke_blocked_v1"
)

BASE_RESUME1_IMPLEMENTATION_COMMIT = "70045b5b304b75ef770eee6702bd55610eefd2d7"
BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT = "28cfba64980ed311810ca314dcc249a6a5681fb3"
BASE_RESUME1_PARENT = "fda74c4b3496bff630c74c60c14685f7d1fdc60d"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

RESUME1_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S Resume1: restore oracle-rerun field schema"
)
RESUME1_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage S Resume1 blocked evidence"
RESUME1_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stages_resume1_oof_support_confirmation_"
    "blocked_summary.json"
)
EXPECTED_RESUME1_BLOCKED_REPORT_SHA256 = (
    "55f509a6e5320e42b2d0e4b5004c5cdc402a06174e073dc3c812132e278ad9b3"
)

RESUME1_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stages_resume1_"
        "oracle_rerun_schema_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r258_stages_resume1_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume1_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stages_resume1_"
        "oracle_rerun_schema_recovery.py",
    ),
)
RESUME1_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", RESUME1_BLOCKED_REPORT),
)
RESUME1_SOURCE_SHA256: Mapping[str, str] = {
    (
        "ccda_phase3/phase314b_r258_stages_resume1_"
        "oracle_rerun_schema_recovery.py"
    ): "1d6c89358ea9b0f6087f90c683384a3adf82e9b10c513b9d0634799ed97ea3d9",
    "scripts/phase3_14b_r258_stages_resume1_worker.py": (
        "9b5bf8e684ccdef081c9db334fc945e524a13f25eca3b48f62d3fd1c8b699c64"
    ),
    "scripts/phase3_14b_r258_stages_resume1_execute.py": (
        "a8a5d56519dec2a7a2334e2ca5ec4074e16794ba37202db9461b9a5cb2d050d5"
    ),
    (
        "tests/test_phase3_14b_r258_stages_resume1_"
        "oracle_rerun_schema_recovery.py"
    ): "eeb43946b252b7e8b86ffcc8e0f943980e93baf482767281868af8567b8fe735",
}

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stages_resume2a_cold_cuda_process_boundary_"
    "smoke_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stages_resume2a_cold_cuda_process_boundary_"
    "smoke_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S Resume2A: smoke cold-CUDA process boundary"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume2A cold-CUDA smoke evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume2A blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py",
    ),
    ("A", "scripts/phase3_14b_r258_stages_resume2a_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume2a_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py",
    ),
)

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
    "candidate_tensor_persisted",
    "predicate_tensor_persisted",
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageSResume2AError(RuntimeError):
    """Fail-closed Resume2A execution error."""


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
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageSResume2AError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise StageSResume2AError(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageSResume2AError(f"{label} is not a mapping")
    return value


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageSResume2AError(
            "deterministic environment mismatch: {}".format(mismatch)
        )
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
        raise StageSResume2AError(
            "git {} failed: {}".format(" ".join(args), str(stderr).strip())
        )
    return completed.stdout


def _git(root: Path, *args: str) -> str:
    return str(_run_git(root, *args, text=True)).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return bytes(_run_git(root, *args, text=False))


def assert_clean_worktree(root: Path, label: str) -> None:
    raw = _git_bytes(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if raw:
        raise StageSResume2AError(f"{label} worktree is not clean")


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageSResume2AError("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageSResume2AError("Stage-S Resume2A requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT:
        raise StageSResume2AError("Resume2A implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageSResume2AError("Resume2A implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageSResume2AError("Resume2A implementation paths changed")

    if _git(repo, "rev-parse", f"{BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_RESUME1_IMPLEMENTATION_COMMIT
    ):
        raise StageSResume2AError("Resume1 blocked provenance changed")
    if _git(repo, "rev-parse", f"{BASE_RESUME1_IMPLEMENTATION_COMMIT}^") != (
        BASE_RESUME1_PARENT
    ):
        raise StageSResume2AError("Resume1 implementation provenance changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_RESUME1_IMPLEMENTATION_COMMIT,
    ) != RESUME1_IMPLEMENTATION_SUBJECT:
        raise StageSResume2AError("Resume1 implementation subject changed")
    if commit_name_status(repo, BASE_RESUME1_IMPLEMENTATION_COMMIT) != tuple(
        sorted(RESUME1_IMPLEMENTATION_PATHS)
    ):
        raise StageSResume2AError("Resume1 implementation paths changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT,
    ) != RESUME1_BLOCKED_SUBJECT:
        raise StageSResume2AError("Resume1 blocked-evidence subject changed")
    if commit_name_status(repo, BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(RESUME1_BLOCKED_PATHS)
    ):
        raise StageSResume2AError("Resume1 blocked-evidence paths changed")

    for relative, expected_sha in RESUME1_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageSResume2AError(f"Resume1 source changed: {relative}")
        committed = _git_bytes(
            repo,
            "show",
            f"{BASE_RESUME1_IMPLEMENTATION_COMMIT}:{relative}",
        )
        if committed != path.read_bytes():
            raise StageSResume2AError(
                f"Resume1 source differs from committed blob: {relative}"
            )

    blocked = repo / RESUME1_BLOCKED_REPORT
    if not blocked.is_file():
        raise StageSResume2AError("Resume1 blocked report is missing")
    if sha256_file(blocked) != EXPECTED_RESUME1_BLOCKED_REPORT_SHA256:
        raise StageSResume2AError("Resume1 blocked report SHA changed")
    committed_blocked = _git_bytes(
        repo,
        "show",
        f"{BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT}:{RESUME1_BLOCKED_REPORT}",
    )
    if committed_blocked != blocked.read_bytes():
        raise StageSResume2AError("Resume1 blocked report differs from committed blob")

    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageSResume2AError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageSResume2AError("DeformableRavens commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Resume2A")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageSResume2AError(f"Resume2A output already exists: {relative}")

    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "resume1_implementation_commit": BASE_RESUME1_IMPLEMENTATION_COMMIT,
        "resume1_blocked_evidence_commit": BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT,
        "resume1_blocked_report_sha256": EXPECTED_RESUME1_BLOCKED_REPORT_SHA256,
        "resume1_source_sha256": dict(RESUME1_SOURCE_SHA256),
    }


def validate_probe_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    wrapper = _mapping(payload, "environment probe wrapper")
    if wrapper.get("mode") != "environment_probe":
        raise StageSResume2AError("environment probe mode changed")
    if wrapper.get("execution_verdict") != "PASS":
        raise StageSResume2AError("environment probe did not PASS")
    environment = _mapping(wrapper.get("environment"), "environment payload")
    if environment.get("compatibility_pass") is not True:
        raise StageSResume2AError("portable compatibility did not pass")
    compatibility = _mapping(environment.get("compatibility"), "compatibility")
    dry_run = _mapping(
        environment.get("required_operation_dry_run"),
        "required-operation dry run",
    )
    if compatibility.get("required_operation_pass") is not True:
        raise StageSResume2AError("required-operation compatibility did not pass")
    if dry_run.get("pass") is not True:
        raise StageSResume2AError("required-operation dry run did not pass")
    expected_sha = sha256_bytes(stable_json_bytes(environment))
    if wrapper.get("environment_sha256") != expected_sha:
        raise StageSResume2AError("environment wrapper SHA is invalid")
    return environment


def environment_probe_payload(root: Path) -> Mapping[str, Any]:
    """Run only inside a disposable probe subprocess."""
    validate_environment_variables()
    from ccda_phase3 import (
        phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager,
    )

    environment = stager.probe_environment(Path(root).resolve())
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": "phase314b_r258_stages_resume2a_environment_probe_v1",
        "mode": "environment_probe",
        "execution_verdict": "PASS",
        "process_id": os.getpid(),
        "environment": environment,
        "environment_sha256": sha256_bytes(stable_json_bytes(environment)),
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
    }
    validate_probe_payload(payload)
    return payload


def validate_smoke_payload(
    payload: Mapping[str, Any],
    *,
    expected_environment_sha256: str,
) -> Mapping[str, Any]:
    wrapper = _mapping(payload, "cold-control smoke wrapper")
    if wrapper.get("mode") != "cold_control_smoke":
        raise StageSResume2AError("cold-control smoke mode changed")
    if wrapper.get("execution_verdict") != "PASS":
        raise StageSResume2AError("cold-control smoke did not PASS")
    if wrapper.get("environment_sha256") != expected_environment_sha256:
        raise StageSResume2AError("smoke environment SHA differs from probe")
    cold = _mapping(wrapper.get("cold_cuda_precheck"), "cold CUDA precheck")
    runtime_cold = _mapping(
        wrapper.get("runtime_cold_cuda_contract"),
        "runtime cold CUDA contract",
    )
    if cold.get("torch_cuda_is_initialized") is not False:
        raise StageSResume2AError("smoke process was not cold before replay")
    if runtime_cold.get("torch_cuda_is_initialized") is not False:
        raise StageSResume2AError("Stage-O runtime cold contract changed")
    if wrapper.get("frozen_control_replay_completed") is not True:
        raise StageSResume2AError("frozen control replay did not complete")
    for count_key in (
        "oof_fit_count",
        "callback_pair_count",
        "internal_scale_attempt_count",
        "oracle_callback_rerun_count",
    ):
        if wrapper.get(count_key) != 0:
            raise StageSResume2AError(f"smoke unexpectedly ran science: {count_key}")
    if wrapper.get("control_capture_sha256") != sha256_bytes(
        stable_json_bytes(_mapping(wrapper.get("control_capture"), "control capture"))
    ):
        raise StageSResume2AError("control-capture SHA is invalid")
    if wrapper.get("ulp_policy_sha256") != sha256_bytes(
        stable_json_bytes(_mapping(wrapper.get("ulp_policy"), "ULP policy"))
    ):
        raise StageSResume2AError("ULP-policy SHA is invalid")
    return wrapper


def cold_control_smoke_payload(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run only inside a new process that has not executed the CUDA probe."""
    validate_environment_variables()
    environment = validate_probe_payload(probe_payload)

    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )

    modules = stageo._runtime_modules()
    stagea = modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment)
    cold_precheck = stagea.assert_cold_cuda_context_portable()
    runtime = stageo._prepare_runtime(Path(root).resolve(), environment)
    runtime_cold = _mapping(
        runtime.get("cold_main_worker_context"),
        "Stage-O cold-main-worker context",
    )
    control_capture = dict(
        _mapping(runtime.get("control_capture"), "Stage-O control capture")
    )
    ulp_policy = dict(_mapping(runtime.get("ulp_policy"), "Stage-O ULP policy"))

    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": "phase314b_r258_stages_resume2a_cold_control_smoke_v1",
        "mode": "cold_control_smoke",
        "execution_verdict": "PASS",
        "process_id": os.getpid(),
        "environment_sha256": sha256_bytes(stable_json_bytes(environment)),
        "cold_cuda_precheck": dict(cold_precheck),
        "runtime_cold_cuda_contract": dict(runtime_cold),
        "frozen_control_replay_completed": True,
        "control_capture": control_capture,
        "control_capture_sha256": sha256_bytes(stable_json_bytes(control_capture)),
        "ulp_policy": ulp_policy,
        "ulp_policy_sha256": sha256_bytes(stable_json_bytes(ulp_policy)),
        "oof_fit_count": 0,
        "callback_pair_count": 0,
        "internal_scale_attempt_count": 0,
        "oracle_callback_rerun_count": 0,
        "selection_holdout_evaluated": False,
        "frozen_probe_accessed": False,
        "formal_training_run": False,
        "reverse_sampling_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "worker_output_persisted": False,
        "environment_probe_rerun_in_smoke_process": False,
    }
    validate_smoke_payload(
        payload,
        expected_environment_sha256=payload["environment_sha256"],
    )
    return payload


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
        raise StageSResume2AError(
            f"{label} rc={completed.returncode} "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return completed


def run_process_boundary_smoke(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
) -> Mapping[str, Any]:
    validate_environment_variables()
    repo = Path(root).resolve()
    worker_script = repo / "scripts/phase3_14b_r258_stages_resume2a_worker.py"
    with tempfile.TemporaryDirectory(
        prefix="phase314b_r258_stages_resume2a_"
    ) as temporary:
        directory = Path(temporary)
        probe_path = directory / "environment_probe.json"
        smoke_path = directory / "cold_control_smoke.json"

        probe_command = [
            str(python_bin),
            str(worker_script),
            "--mode",
            "environment-probe",
            "--root",
            str(repo),
            "--output",
            str(probe_path),
        ]
        _run_child(probe_command, root=repo, label="environment probe")
        probe_payload = load_json(probe_path)
        environment = validate_probe_payload(probe_payload)

        smoke_command = [
            str(python_bin),
            str(worker_script),
            "--mode",
            "cold-control-smoke",
            "--root",
            str(repo),
            "--probe",
            str(probe_path),
            "--output",
            str(smoke_path),
        ]
        _run_child(smoke_command, root=repo, label="cold-control smoke")
        smoke_payload = validate_smoke_payload(
            load_json(smoke_path),
            expected_environment_sha256=sha256_bytes(stable_json_bytes(environment)),
        )

        probe_pid = probe_payload.get("process_id")
        smoke_pid = smoke_payload.get("process_id")
        if not isinstance(probe_pid, int) or not isinstance(smoke_pid, int):
            raise StageSResume2AError("subprocess PID evidence is invalid")
        if probe_pid == smoke_pid:
            raise StageSResume2AError("probe and smoke ran in the same process")

        probe_sha = sha256_bytes(stable_json_bytes(probe_payload))
        smoke_sha = sha256_bytes(stable_json_bytes(smoke_payload))

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stages_resume2a_cold_cuda_process_boundary_confirmed"
        ),
        "required_next_path": (
            "RUN_STAGES_RESUME3_CONFIRMATION_WITH_DISPOSABLE_PROBE_"
            "AND_COLD_SCIENCE_WORKERS"
        ),
        "primary_failure_locus": "process_boundary_confirmed",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": dict(repository),
        "immutable_inputs": {
            "resume1_implementation_commit": BASE_RESUME1_IMPLEMENTATION_COMMIT,
            "resume1_blocked_evidence_commit": BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT,
            "resume1_blocked_report_sha256": EXPECTED_RESUME1_BLOCKED_REPORT_SHA256,
            "origin_experiment1": EXPECTED_REMOTE,
            "submodule_commit": EXPECTED_SUBMODULE,
        },
        "process_boundary_smoke": {
            "environment_probe_process_count": 1,
            "cold_control_smoke_process_count": 1,
            "processes_distinct": True,
            "environment_probe_payload_sha256": probe_sha,
            "cold_control_smoke_payload_sha256": smoke_sha,
            "environment_sha256": smoke_payload["environment_sha256"],
            "cold_cuda_before_control_replay": True,
            "frozen_control_replay_completed": True,
            "control_capture_sha256": smoke_payload["control_capture_sha256"],
            "ulp_policy_sha256": smoke_payload["ulp_policy_sha256"],
            "temporary_payloads_deleted": True,
            "oof_fit_count": 0,
            "callback_pair_count": 0,
            "internal_scale_attempt_count": 0,
            "oracle_callback_rerun_count": 0,
        },
        "mechanism_boundary": {
            "stage_s_modified": False,
            "stage_s_resume1_modified": False,
            "stage_r_resume1_modified": False,
            "stage_o_modified": False,
            "stage_a_portable_contract_modified": False,
            "cold_cuda_gate_removed": False,
            "environment_probe_skipped": False,
            "required_operation_dry_run_skipped": False,
            "science_worker_started": False,
            "oof_replay_started": False,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stages_resume2a_cold_cuda_process_boundary_failed"
        ),
        "required_next_path": (
            "RESTORE_STAGES_RESUME2A_COLD_CUDA_PROCESS_BOUNDARY_SMOKE"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "resume1_blocked_report_sha256": EXPECTED_RESUME1_BLOCKED_REPORT_SHA256,
        "science_worker_started": False,
        "oof_fit_count_claimed": False,
        "callback_pair_count_claimed": False,
        "internal_scale_attempt_count_claimed": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
