"""Stage-V Resume1: recover a CUDA-unavailable failure proven pre-holdout.

The original Stage-V attempt stopped in its first disposable environment-probe
process because no CUDA device was available.  The exact original source and
blocked report prove that the cold science worker never started and therefore
no objective-train fit, candidate generation, holdout-target read, or holdout
evaluation occurred.

Resume1 is add-only.  It preserves the original Stage-V implementation and
blocked evidence, authorizes exactly one recovered call to the unchanged
``run_evaluation`` function on a CUDA-capable host, and writes distinct
write-once Resume1 evidence.  It does not weaken the Stage-U frontier, gate,
metric, leakage, or no-fallback contracts.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage V Resume1"
SCHEMA = "phase314b_r258_stagev_resume1_preholdout_cuda_recovery_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEV_IMPLEMENTATION_COMMIT = "5bb1140e013f48e0a5a2357fcfa8ec79e9a7d177"
BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT = "e02d8131b49c0b012e188b79a4e46b5c2f9cfbba"
EXPECTED_STAGEV_IMPLEMENTATION_PARENT = "959a4a77cbd3af40e0458d8c4dcc49a404246f0f"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

ORIGINAL_STAGEV_SOURCE = (
    "ccda_phase3/phase314b_r258_stagev_locked_selection_holdout_evaluation.py"
)
ORIGINAL_STAGEV_WORKER = "scripts/phase3_14b_r258_stagev_worker.py"
ORIGINAL_STAGEV_EXECUTE = "scripts/phase3_14b_r258_stagev_execute.py"
ORIGINAL_STAGEV_TEST = (
    "tests/test_phase3_14b_r258_stagev_locked_selection_holdout_evaluation.py"
)
ORIGINAL_STAGEV_SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagev_locked_selection_holdout_evaluation_summary.json"
)
ORIGINAL_STAGEV_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagev_locked_selection_holdout_evaluation_"
    "blocked_summary.json"
)
EXPECTED_ORIGINAL_BLOCKED_REPORT_SHA256 = (
    "f9facbe30f6292256451ec94ceaaf87654b5ea5cd95892956e4433ba90bb033c"
)

RESUME1_SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagev_resume1_locked_selection_holdout_"
    "evaluation_summary.json"
)
RESUME1_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagev_resume1_locked_selection_holdout_"
    "evaluation_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage V Resume1: recover pre-holdout CUDA execution"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage V Resume1 locked holdout evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage V Resume1 blocked evidence"
)
ORIGINAL_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage V: evaluate locked frontier on selection holdout once"
)
ORIGINAL_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage V blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stagev_resume1_preholdout_cuda_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r258_stagev_resume1_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stagev_resume1_preholdout_cuda_recovery.py",
    ),
)
ORIGINAL_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", ORIGINAL_STAGEV_SOURCE),
    ("A", ORIGINAL_STAGEV_WORKER),
    ("A", ORIGINAL_STAGEV_EXECUTE),
    ("A", ORIGINAL_STAGEV_TEST),
)
ORIGINAL_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", ORIGINAL_STAGEV_BLOCKED_REPORT),
)

ORIGINAL_SOURCE_SHA256: Mapping[str, str] = {
    ORIGINAL_STAGEV_SOURCE: (
        "5aedabe9004630207021f1252f5b99aa6524b8dc9b3077003b20c8b163e7b1b0"
    ),
    ORIGINAL_STAGEV_WORKER: (
        "be3e364926765c6911bb3f8a1028dde832847eb8cb8322175e65b0487275efa4"
    ),
    ORIGINAL_STAGEV_EXECUTE: (
        "75dc6a64995127f71e598ef62443a88dfa187af83bf47501efcd272956b762ef"
    ),
    ORIGINAL_STAGEV_TEST: (
        "7391e1a9d6605188f13be9ba98f7a5610a4756cd724a729871a79205fa65018b"
    ),
    "scripts/phase3_14b_r258_stages_resume2a_worker.py": (
        "096ffbedea6e4504ad01f77b637d812fb8a69c2cb8437c1a3aaa4cccf66a97d2"
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

ORIGINAL_FALSE_BOUNDARIES: Tuple[str, ...] = (
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


class StageVResume1Error(RuntimeError):
    """Fail-closed Stage-V Resume1 execution-contract error."""


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageVResume1Error(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageVResume1Error(f"{label} is not a mapping")
    return value


def require_false(payload: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if payload.get(key) is not False:
            raise StageVResume1Error(
                f"{label} boundary changed: {key}={payload.get(key)!r}"
            )


def validate_environment_variables() -> Mapping[str, str]:
    observed = {key: os.environ.get(key) for key in EXPECTED_ENV}
    if observed != dict(EXPECTED_ENV):
        raise StageVResume1Error(
            f"deterministic environment changed: {observed!r}"
        )
    return dict(EXPECTED_ENV)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageVResume1Error(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageVResume1Error(
            f"git {' '.join(args)} failed: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def status_paths(root: Path) -> Tuple[str, ...]:
    output = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return tuple(line for line in output.splitlines() if line.strip())


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        records.append((parts[0], parts[-1]))
    return tuple(sorted(records))


def prove_original_control_flow(source_path: Path) -> Mapping[str, Any]:
    """Prove the failed labelled child is before every science/holdout action."""
    path = Path(source_path)
    if sha256_file(path) != ORIGINAL_SOURCE_SHA256[ORIGINAL_STAGEV_SOURCE]:
        raise StageVResume1Error("original Stage-V source SHA changed")
    text = path.read_text(encoding="utf-8")
    function_start = text.find("def run_evaluation(")
    function_end = text.find("\ndef blocked_report(", function_start)
    if function_start < 0 or function_end < 0:
        raise StageVResume1Error("original Stage-V run_evaluation layout changed")
    body = text[function_start:function_end]
    probe_index = body.find('label="environment probe"')
    worker_index = body.find('label="single cold science worker"')
    worker_load_index = body.find("validate_worker_payload(load_json(worker_path))")
    if not (0 <= probe_index < worker_index < worker_load_index):
        raise StageVResume1Error("original Stage-V child-process order changed")
    if "evaluate_timestep(" in body:
        raise StageVResume1Error(
            "original controller unexpectedly evaluates holdout in-process"
        )
    return {
        "original_stagev_source_sha256": ORIGINAL_SOURCE_SHA256[
            ORIGINAL_STAGEV_SOURCE
        ],
        "environment_probe_precedes_science_worker": True,
        "science_worker_precedes_worker_payload_load": True,
        "controller_has_no_in_process_holdout_evaluation": True,
    }


def validate_original_blocked_payload(
    payload: Mapping[str, Any],
    *,
    control_flow: Mapping[str, Any],
) -> Mapping[str, Any]:
    if payload.get("phase") != "Phase3.14b-r2.5.8 Stage V":
        raise StageVResume1Error("original Stage-V blocked phase changed")
    if payload.get("schema") != (
        "phase314b_r258_stagev_locked_selection_holdout_evaluation_v1_blocked_v1"
    ):
        raise StageVResume1Error("original Stage-V blocked schema changed")
    expected = {
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagev_execution_contract_failed",
        "required_next_path": (
            "DESIGN_ADD_ONLY_STAGEV_EXECUTION_CONTRACT_CORRECTION_"
            "WITHOUT_REACCESSING_HOLDOUT_IF_ACCESS_OCCURRED"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_access_state": (
            "unknown_if_failure_after_worker_start"
        ),
        "complete_worker_population_claimed": False,
        "complete_fit_count_claimed": False,
        "complete_candidate_count_claimed": False,
        "complete_holdout_evaluation_count_claimed": False,
        "rerun_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageVResume1Error(
                f"original Stage-V blocked field changed: {key}"
            )
    repository = _mapping(payload.get("repository"), "original repository")
    if repository.get("head") != BASE_STAGEV_IMPLEMENTATION_COMMIT:
        raise StageVResume1Error("original blocked report implementation changed")
    if repository.get("parent") != EXPECTED_STAGEV_IMPLEMENTATION_PARENT:
        raise StageVResume1Error("original blocked report parent changed")
    if repository.get("origin_experiment1") != EXPECTED_REMOTE:
        raise StageVResume1Error("original blocked report remote changed")
    if repository.get("submodule_commit") != EXPECTED_SUBMODULE:
        raise StageVResume1Error("original blocked report submodule changed")
    if payload.get("error_type") != "StageVError":
        raise StageVResume1Error("original Stage-V blocked error type changed")
    message = payload.get("error_message")
    if not isinstance(message, str):
        raise StageVResume1Error("original Stage-V blocked message is missing")
    if not re.match(r"^environment probe rc=[1-9][0-9]* ", message):
        raise StageVResume1Error(
            "original failure was not labelled as the environment probe"
        )
    if "no CUDA device is available" not in message:
        raise StageVResume1Error("original CUDA-unavailable cause changed")
    if "single cold science worker" in message:
        raise StageVResume1Error("original message ambiguously names science worker")
    require_false(payload, ORIGINAL_FALSE_BOUNDARIES, "original Stage-V blocked")
    if control_flow.get("environment_probe_precedes_science_worker") is not True:
        raise StageVResume1Error("original control-flow proof is incomplete")
    return {
        "failure_stage": "environment_probe",
        "failure_cause": "cuda_unavailable",
        "environment_probe_started": True,
        "environment_probe_completed": False,
        "environment_probe_pid_persisted": False,
        "science_worker_started": False,
        "objective_train_fit_count": 0,
        "candidate_generation_count": 0,
        "selection_holdout_target_access_count": 0,
        "selection_holdout_evaluation_count": 0,
        "holdout_access_before_resume1_proven_false": True,
        "proof_basis": {
            "blocked_report_exact_sha_required": True,
            "failure_label": "environment probe",
            "original_control_flow": dict(control_flow),
        },
    }


def validate_original_blocked_report(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    report_path = repo / ORIGINAL_STAGEV_BLOCKED_REPORT
    if not report_path.is_file():
        raise StageVResume1Error("original Stage-V blocked report is missing")
    if sha256_file(report_path) != EXPECTED_ORIGINAL_BLOCKED_REPORT_SHA256:
        raise StageVResume1Error("original Stage-V blocked report SHA changed")
    committed = _git_bytes(
        repo,
        "show",
        f"{BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT}:{ORIGINAL_STAGEV_BLOCKED_REPORT}",
    )
    if committed != report_path.read_bytes():
        raise StageVResume1Error(
            "original Stage-V blocked report differs from committed blob"
        )
    control_flow = prove_original_control_flow(repo / ORIGINAL_STAGEV_SOURCE)
    proof = validate_original_blocked_payload(
        load_json(report_path), control_flow=control_flow
    )
    return {
        "blocked_evidence_commit": BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT,
        "blocked_report": ORIGINAL_STAGEV_BLOCKED_REPORT,
        "blocked_report_sha256": EXPECTED_ORIGINAL_BLOCKED_REPORT_SHA256,
        **proof,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageVResume1Error("Stage-V Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT:
        raise StageVResume1Error("Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageVResume1Error("Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageVResume1Error("Resume1 implementation paths changed")

    if _git(repo, "rev-parse", f"{BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGEV_IMPLEMENTATION_COMMIT
    ):
        raise StageVResume1Error("Stage-V blocked evidence parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGEV_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_STAGEV_IMPLEMENTATION_PARENT
    ):
        raise StageVResume1Error("Stage-V implementation parent changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGEV_IMPLEMENTATION_COMMIT,
    ) != ORIGINAL_IMPLEMENTATION_SUBJECT:
        raise StageVResume1Error("original Stage-V implementation subject changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT,
    ) != ORIGINAL_BLOCKED_SUBJECT:
        raise StageVResume1Error("original Stage-V blocked subject changed")
    if commit_name_status(repo, BASE_STAGEV_IMPLEMENTATION_COMMIT) != tuple(
        sorted(ORIGINAL_IMPLEMENTATION_PATHS)
    ):
        raise StageVResume1Error("original Stage-V implementation paths changed")
    if commit_name_status(repo, BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(ORIGINAL_BLOCKED_PATHS)
    ):
        raise StageVResume1Error("original Stage-V blocked paths changed")

    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageVResume1Error("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageVResume1Error("DeformableRavens commit changed")
    if status_paths(repo) or status_paths(submodule):
        raise StageVResume1Error("repository or submodule worktree is dirty")

    for relative, expected_sha in ORIGINAL_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageVResume1Error(f"frozen source changed: {relative}")
        if relative in {
            ORIGINAL_STAGEV_SOURCE,
            ORIGINAL_STAGEV_WORKER,
            ORIGINAL_STAGEV_EXECUTE,
            ORIGINAL_STAGEV_TEST,
        }:
            committed = _git_bytes(
                repo,
                "show",
                f"{BASE_STAGEV_IMPLEMENTATION_COMMIT}:{relative}",
            )
            if committed != path.read_bytes():
                raise StageVResume1Error(
                    f"original Stage-V source differs from committed blob: {relative}"
                )

    if (repo / ORIGINAL_STAGEV_SUCCESS_REPORT).exists():
        raise StageVResume1Error("original Stage-V success report unexpectedly exists")
    for relative in (RESUME1_SUCCESS_REPORT, RESUME1_BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageVResume1Error(f"Resume1 output already exists: {relative}")

    proof = validate_original_blocked_report(repo)
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "original_stagev_implementation_commit": BASE_STAGEV_IMPLEMENTATION_COMMIT,
        "original_stagev_blocked_evidence_commit": (
            BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT
        ),
        "original_failure_proof": proof,
    }


def validate_cuda_admission() -> Mapping[str, Any]:
    """Runtime admission; the wrapper also runs this before repository mutation."""
    try:
        import torch
    except BaseException as error:  # pragma: no cover - target environment check
        raise StageVResume1Error(f"PyTorch import failed: {error}") from error
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    if not available or count <= 0:
        raise StageVResume1Error(
            "Resume1 requires a CUDA-capable host before consuming its one attempt"
        )
    return {
        "torch_version": str(torch.__version__),
        "cuda_available": available,
        "cuda_device_count": count,
        "cuda_initialized_in_controller": bool(torch.cuda.is_initialized()),
    }


def _original_stagev_module() -> Any:
    from ccda_phase3 import (
        phase314b_r258_stagev_locked_selection_holdout_evaluation as stagev,
    )

    if sha256_file(Path(stagev.__file__)) != ORIGINAL_SOURCE_SHA256[
        ORIGINAL_STAGEV_SOURCE
    ]:
        raise StageVResume1Error("imported original Stage-V source changed")
    signature = inspect.signature(stagev.run_evaluation)
    if tuple(signature.parameters) != ("root", "repository", "python_bin"):
        raise StageVResume1Error("original Stage-V run_evaluation signature changed")
    return stagev


def validate_recovered_summary(
    summary: Mapping[str, Any], *, stagev: Any
) -> Mapping[str, Any]:
    if summary.get("phase") != stagev.PHASE:
        raise StageVResume1Error("recovered Stage-V phase changed")
    if summary.get("schema") != stagev.SCHEMA:
        raise StageVResume1Error("recovered Stage-V schema changed")
    if summary.get("execution_verdict") != "PASS":
        raise StageVResume1Error("recovered Stage-V execution did not pass")
    if summary.get("selection_holdout_evaluated") is not True:
        raise StageVResume1Error("recovered Stage-V did not evaluate holdout")
    if summary.get("selection_holdout_used_for_fit_or_selection") is not False:
        raise StageVResume1Error("recovered Stage-V leaked holdout into fit/selection")
    if summary.get("evaluation_count") != 1:
        raise StageVResume1Error("recovered Stage-V evaluation count changed")
    if summary.get("rerun_authorized") is not False:
        raise StageVResume1Error("recovered Stage-V authorized a rerun")
    execution = _mapping(summary.get("stagev_execution"), "Stage-V execution")
    expected_execution = {
        "environment_probe_count": 1,
        "cold_science_worker_count": 1,
        "objective_full_fit_count": 6,
        "holdout_direction_prediction_count": 6,
        "candidate_generation_count": 3,
        "joint_selection_holdout_evaluation_count": 1,
        "holdout_target_metric_access_count": 3,
        "internal_scale_attempt_count": 21,
        "oof_fit_count": 0,
        "callback_pair_count": 0,
        "worker_output_persisted": False,
    }
    if dict(execution) != expected_execution:
        raise StageVResume1Error("recovered Stage-V execution counts changed")
    worker_view = copy.deepcopy(dict(summary))
    worker_view["schema"] = stagev.WORKER_SCHEMA
    stagev.validate_worker_payload(worker_view)
    return summary


def run_recovery(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
    cuda_admission: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_environment_variables()
    if cuda_admission.get("cuda_available") is not True:
        raise StageVResume1Error("CUDA admission was not satisfied")
    proof = _mapping(
        repository.get("original_failure_proof"), "original failure proof"
    )
    if proof.get("holdout_access_before_resume1_proven_false") is not True:
        raise StageVResume1Error("pre-Resume1 holdout closure was not proven")

    stagev = _original_stagev_module()
    repository_view = {
        "root": str(Path(root).resolve()),
        "head": str(repository["head"]),
        "parent": str(repository["parent"]),
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
    }
    recovered = stagev.run_evaluation(
        root=Path(root).resolve(),
        repository=repository_view,
        python_bin=str(python_bin),
    )
    recovered = validate_recovered_summary(recovered, stagev=stagev)
    recovered_sha = sha256_bytes(stable_json_bytes(recovered))

    result: Dict[str, Any] = copy.deepcopy(dict(recovered))
    result.update(
        {
            "phase": PHASE,
            "schema": SCHEMA,
            "resume1_execution_verdict": "PASS",
            "recovered_stagev_result_sha256": recovered_sha,
            "original_stagev_blocked_report_preserved": True,
            "original_stagev_blocked_report_sha256": (
                EXPECTED_ORIGINAL_BLOCKED_REPORT_SHA256
            ),
            "original_stagev_source_modified": False,
            "original_stagev_worker_modified": False,
            "original_stagev_execute_modified": False,
            "original_stagev_test_modified": False,
            "holdout_access_before_resume1": False,
            "holdout_access_before_resume1_proven": True,
            "prior_selection_holdout_evaluation_count": 0,
            "resume1_selection_holdout_evaluation_count": 1,
            "cumulative_selection_holdout_evaluation_count": 1,
            "resume1_rerun_authorized": False,
            "recovery_scope": {
                "execution_contract_only": True,
                "cuda_unavailable_failure_recovered": True,
                "science_code_changed": False,
                "frontier_changed": False,
                "timestep_policy_changed": False,
                "gate_changed": False,
                "threshold_changed": False,
                "fallback_opened": False,
                "holdout_reaccessed": False,
            },
            "pre_science_cuda_admission": dict(cuda_admission),
            "original_failure_proof": copy.deepcopy(dict(proof)),
        }
    )
    result.pop("scientific_result_sha256", None)
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def classify_failure_stage(error: BaseException) -> Mapping[str, Any]:
    message = str(error)
    if message.startswith("environment probe rc="):
        return {
            "failure_stage": "environment_probe",
            "holdout_access_state": "proven_not_accessed",
            "science_worker_started": False,
        }
    if message.startswith("single cold science worker rc="):
        return {
            "failure_stage": "cold_science_worker",
            "holdout_access_state": "unknown_after_worker_start",
            "science_worker_started": True,
        }
    return {
        "failure_stage": "resume1_controller_or_validation",
        "holdout_access_state": "unknown_unless_controller_failed_pre_science",
        "science_worker_started": None,
    }


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    failure = classify_failure_stage(error)
    proof = None
    if repository is not None:
        proof = repository.get("original_failure_proof")
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagev_resume1_execution_contract_failed",
        "required_next_path": (
            "AUDIT_STAGEV_RESUME1_FAILURE_WITHOUT_RERUN_OR_HOLDOUT_REACCESS"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None
        if repository is None
        else {
            key: value
            for key, value in repository.items()
            if key != "original_failure_proof"
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        **failure,
        "original_stagev_blocked_report_preserved": True,
        "original_stagev_blocked_report_sha256": (
            EXPECTED_ORIGINAL_BLOCKED_REPORT_SHA256
        ),
        "original_holdout_access_before_resume1": False,
        "original_holdout_access_before_resume1_proof": proof,
        "complete_resume1_worker_population_claimed": False,
        "complete_resume1_fit_count_claimed": False,
        "complete_resume1_candidate_count_claimed": False,
        "complete_resume1_holdout_evaluation_count_claimed": False,
        "rerun_authorized": False,
        "resume1_rerun_authorized": False,
        **{key: False for key in ORIGINAL_FALSE_BOUNDARIES},
    }


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageVResume1Error(f"write-once output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
