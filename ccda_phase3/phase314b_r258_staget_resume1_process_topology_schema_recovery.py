"""Stage-T Resume1: recover the frozen Stage-S Resume3 process-topology schema.

The original Stage-T implementation stopped before launching any Stage-T worker
because its historical validator looked for ``processes_distinct`` and
``workers_sequential`` inside ``confirmation_execution``.  The immutable
Stage-S Resume3 report records those facts under ``process_topology`` as
``probe_distinct_from_each_science_worker`` and
``workers_launched_sequentially``.

Resume1 changes only this read boundary.  It validates the real schema and PID
population strictly, creates a temporary deep-copied compatibility view for the
unchanged Stage-T validator, and then runs the unchanged Stage-T confirmation
exactly once.  The original Stage-T source, workers, candidate reconstruction,
eligibility policy, ranking, Pareto frontier, gate contract, and blocked report
remain immutable.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage T Resume1"
SCHEMA = "phase314b_r258_staget_resume1_process_topology_schema_recovery_v1"
GATE_SCHEMA = "phase314b_r258_staget_resume1_tolerance_aligned_gate_contract_v1"
BLOCKED_SCHEMA = "phase314b_r258_staget_resume1_process_topology_schema_recovery_blocked_v1"

BASE_STAGET_IMPLEMENTATION_COMMIT = "fb2b43f7da97a47a92753e936d89422a479e4ba6"
BASE_STAGET_BLOCKED_EVIDENCE_COMMIT = "92d6ff422f32dcca9a75e2ea082b4763d28c5767"
EXPECTED_STAGET_IMPLEMENTATION_PARENT = "ccc2be0f872b5ecb6057d4e34c4f5eaa889d4553"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_STAGET_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_staget_oof_candidate_matrix_blocked_summary.json"
)
EXPECTED_STAGET_BLOCKED_REPORT_SHA256 = (
    "3dd1135d391b41a28a20fd259a0a928fcf072c5b37464e3c50ec5d23cc4e3892"
)
BASE_STAGES_RESUME3_REPORT = (
    "reports/phase3_14b_r258_stages_resume3_"
    "cold_worker_oof_confirmation_summary.json"
)
EXPECTED_STAGES_RESUME3_REPORT_SHA256 = (
    "ea1f7d641164d7608435a33dc6cc79eaa72906603d7f03ecb9c721699f4d29a2"
)

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_staget_resume1_oof_candidate_matrix_summary.json"
)
GATE_CONTRACT_REPORT = (
    "reports/phase3_14b_r258_staget_resume1_tolerance_aligned_gate_contract.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_staget_resume1_"
    "process_topology_schema_recovery_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage T Resume1: restore process-topology schema"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage T Resume1 candidate-matrix evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage T Resume1 blocked evidence"
)
BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage T: freeze aligned gate and confirm OOF candidate matrix"
)
BASE_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage T blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/"
        "phase314b_r258_staget_resume1_process_topology_schema_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r258_staget_resume1_execute.py"),
    (
        "A",
        "tests/"
        "test_phase3_14b_r258_staget_resume1_process_topology_schema_recovery.py",
    ),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_staget_aligned_gate_candidate_matrix.py"),
    ("A", "scripts/phase3_14b_r258_staget_worker.py"),
    ("A", "scripts/phase3_14b_r258_staget_execute.py"),
    ("A", "tests/test_phase3_14b_r258_staget_aligned_gate_candidate_matrix.py"),
)
BASE_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", BASE_STAGET_BLOCKED_REPORT),
)
BASE_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_staget_aligned_gate_candidate_matrix.py": (
        "e59961e9312666e759b64adae7c3ae776ee701bccb902e165c5cecd85704ec8c"
    ),
    "scripts/phase3_14b_r258_staget_worker.py": (
        "69173d6a087b3c02fe46c467fc0fb48f198f94dc752612e006e0cd4103245eda"
    ),
    "scripts/phase3_14b_r258_staget_execute.py": (
        "6be787d071f38b71873f409f7c196a54fa12d911a47fd1b2646b1d0dee7878c8"
    ),
    "tests/test_phase3_14b_r258_staget_aligned_gate_candidate_matrix.py": (
        "b3f57e956e5fbe2c32f887ca9be74a59f0353e8a3690d1189ac61f8ce2935ab0"
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
    "direction_tensor_persisted",
    "candidate_tensor_persisted",
    "predicate_tensor_persisted",
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageTResume1Error(RuntimeError):
    """Fail-closed Stage-T Resume1 error."""


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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageTResume1Error(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageTResume1Error(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise StageTResume1Error(f"{label} is not a sequence")
    return value


def _required_bool(mapping: Mapping[str, Any], key: str, label: str) -> bool:
    if key not in mapping:
        raise StageTResume1Error(f"{label} is missing: {key}")
    value = mapping[key]
    if not isinstance(value, bool):
        raise StageTResume1Error(f"{label} is not boolean: {key}={value!r}")
    return value


def _required_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    if key not in mapping:
        raise StageTResume1Error(f"{label} is missing: {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StageTResume1Error(
            f"{label} is not a non-boolean integer: {key}={value!r}"
        )
    return int(value)


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageTResume1Error(
                f"{label} boundary changed: {key}={mapping.get(key)!r}"
            )


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageTResume1Error(
            f"deterministic environment mismatch: {mismatch}"
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
        stderr = (
            completed.stderr
            if text
            else completed.stderr.decode("utf-8", "replace")
        )
        raise StageTResume1Error(
            f"git {' '.join(args)} failed: {stderr.strip()}"
        )
    return completed.stdout


def _git(root: Path, *args: str) -> str:
    return str(_run_git(root, *args, text=True)).strip()


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        commit,
    )
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageTResume1Error("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise StageTResume1Error(f"{label} worktree is dirty")


def extract_process_topology(
    report: Mapping[str, Any],
    *,
    expected_worker_count: int,
) -> Mapping[str, Any]:
    """Validate the real immutable Stage-S Resume3 topology schema."""
    topology = _mapping(report.get("process_topology"), "process topology")
    execution = _mapping(
        report.get("confirmation_execution"), "confirmation execution"
    )

    # Reject the invented aliases.  They were never part of the frozen report.
    for alias in ("processes_distinct", "workers_sequential"):
        if alias in execution:
            raise StageTResume1Error(
                f"unexpected invented topology alias in confirmation execution: {alias}"
            )

    probe_count = _required_int(
        topology,
        "environment_probe_process_count",
        "environment probe process count",
    )
    worker_count = _required_int(
        topology,
        "cold_science_worker_process_count",
        "cold science worker process count",
    )
    probe_pid = _required_int(
        topology,
        "probe_process_id",
        "environment probe process ID",
    )
    worker_pid_values = _sequence(
        topology.get("science_worker_process_ids"),
        "science worker process IDs",
    )
    worker_pids = tuple(
        _required_int(
            {"pid": value},
            "pid",
            f"science worker process ID {index}",
        )
        for index, value in enumerate(worker_pid_values)
    )
    probe_distinct = _required_bool(
        topology,
        "probe_distinct_from_each_science_worker",
        "probe/science process separation",
    )
    workers_sequential = _required_bool(
        topology,
        "workers_launched_sequentially",
        "science worker ordering",
    )
    probe_rerun = _required_bool(
        topology,
        "environment_probe_rerun_in_science_workers",
        "science-worker environment-probe rerun flag",
    )
    temporary_deleted = _required_bool(
        topology,
        "temporary_payloads_deleted",
        "temporary payload deletion flag",
    )

    if probe_count != 1:
        raise StageTResume1Error(
            f"environment probe process count changed: {probe_count}"
        )
    if worker_count != int(expected_worker_count):
        raise StageTResume1Error(
            f"cold science worker process count changed: {worker_count}"
        )
    if len(worker_pids) != int(expected_worker_count):
        raise StageTResume1Error(
            f"science worker PID population changed: {len(worker_pids)}"
        )
    if not probe_distinct:
        raise StageTResume1Error(
            "Stage-S Resume3 probe/science-worker isolation changed"
        )
    if not workers_sequential:
        raise StageTResume1Error(
            "Stage-S Resume3 workers were not launched sequentially"
        )
    if probe_rerun:
        raise StageTResume1Error(
            "Stage-S Resume3 science workers reran the environment probe"
        )
    if not temporary_deleted:
        raise StageTResume1Error(
            "Stage-S Resume3 temporary payloads were not deleted"
        )
    all_pids = (probe_pid, *worker_pids)
    if len(set(all_pids)) != 1 + int(expected_worker_count):
        raise StageTResume1Error(
            f"Stage-S Resume3 process IDs are not all distinct: {all_pids}"
        )

    return {
        "environment_probe_process_count": probe_count,
        "cold_science_worker_process_count": worker_count,
        "probe_process_id": probe_pid,
        "science_worker_process_ids": list(worker_pids),
        "probe_distinct_from_each_science_worker": probe_distinct,
        "workers_launched_sequentially": workers_sequential,
        "environment_probe_rerun_in_science_workers": probe_rerun,
        "temporary_payloads_deleted": temporary_deleted,
        "all_process_ids_distinct": True,
    }


def corrected_validate_base_report(
    report: Mapping[str, Any],
    *,
    staget: Any,
    original_validator: Optional[Any] = None,
) -> Mapping[str, Any]:
    """Call the unchanged Stage-T validator through one strict schema adapter."""
    validator = staget.validate_base_report if original_validator is None else original_validator
    topology = extract_process_topology(
        report,
        expected_worker_count=int(staget.EXPECTED_WORKER_COUNT),
    )
    normalized = copy.deepcopy(dict(report))
    execution = dict(
        _mapping(
            normalized.get("confirmation_execution"),
            "normalized confirmation execution",
        )
    )
    execution["processes_distinct"] = bool(
        topology["probe_distinct_from_each_science_worker"]
    )
    execution["workers_sequential"] = bool(
        topology["workers_launched_sequentially"]
    )
    normalized["confirmation_execution"] = execution
    validator(normalized)
    return report


@contextmanager
def patched_base_report_validator(staget: Any) -> Iterator[None]:
    """Temporarily patch only the Stage-T historical input boundary."""
    original = staget.validate_base_report

    def adapter(report: Mapping[str, Any]) -> Mapping[str, Any]:
        return corrected_validate_base_report(
            report,
            staget=staget,
            original_validator=original,
        )

    staget.validate_base_report = adapter
    try:
        yield
    finally:
        staget.validate_base_report = original


def _validate_base_provenance(repo: Path) -> Mapping[str, Any]:
    if _git(repo, "rev-parse", f"{BASE_STAGET_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGET_IMPLEMENTATION_COMMIT
    ):
        raise StageTResume1Error("Stage-T blocked evidence parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGET_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_STAGET_IMPLEMENTATION_PARENT
    ):
        raise StageTResume1Error("Stage-T implementation parent changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGET_IMPLEMENTATION_COMMIT,
    ) != BASE_IMPLEMENTATION_SUBJECT:
        raise StageTResume1Error("Stage-T implementation subject changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGET_BLOCKED_EVIDENCE_COMMIT,
    ) != BASE_BLOCKED_SUBJECT:
        raise StageTResume1Error("Stage-T blocked evidence subject changed")
    if commit_name_status(
        repo, BASE_STAGET_IMPLEMENTATION_COMMIT
    ) != tuple(sorted(BASE_IMPLEMENTATION_PATHS)):
        raise StageTResume1Error("Stage-T implementation paths changed")
    if commit_name_status(
        repo, BASE_STAGET_BLOCKED_EVIDENCE_COMMIT
    ) != tuple(sorted(BASE_BLOCKED_PATHS)):
        raise StageTResume1Error("Stage-T blocked evidence paths changed")

    blocked_path = repo / BASE_STAGET_BLOCKED_REPORT
    if not blocked_path.is_file():
        raise StageTResume1Error("Stage-T blocked report is missing")
    if sha256_file(blocked_path) != EXPECTED_STAGET_BLOCKED_REPORT_SHA256:
        raise StageTResume1Error("Stage-T blocked report SHA changed")

    source_shas: Dict[str, str] = {}
    for relative, expected in BASE_SOURCE_SHA256.items():
        actual = sha256_file(repo / relative)
        source_shas[relative] = actual
        if actual != expected:
            raise StageTResume1Error(
                f"Stage-T frozen source changed: {relative}"
            )
    return {
        "implementation_commit": BASE_STAGET_IMPLEMENTATION_COMMIT,
        "blocked_evidence_commit": BASE_STAGET_BLOCKED_EVIDENCE_COMMIT,
        "blocked_report_sha256": EXPECTED_STAGET_BLOCKED_REPORT_SHA256,
        "source_sha256": source_shas,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageTResume1Error("Stage-T Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_STAGET_BLOCKED_EVIDENCE_COMMIT:
        raise StageTResume1Error("Stage-T Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageTResume1Error("Stage-T Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageTResume1Error("Stage-T Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageTResume1Error("origin/Experiment1 changed")

    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageTResume1Error("DeformableRavens commit changed")
    assert_clean_worktree(repo, "Stage-T Resume1")
    assert_clean_worktree(submodule, "DeformableRavens")
    base = _validate_base_provenance(repo)

    for relative in (SUCCESS_REPORT, GATE_CONTRACT_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageTResume1Error(
                f"Stage-T Resume1 output already exists: {relative}"
            )

    base_report_path = repo / BASE_STAGES_RESUME3_REPORT
    if not base_report_path.is_file():
        raise StageTResume1Error("Stage-S Resume3 report is missing")
    if sha256_file(base_report_path) != EXPECTED_STAGES_RESUME3_REPORT_SHA256:
        raise StageTResume1Error("Stage-S Resume3 report SHA changed")

    from ccda_phase3 import (
        phase314b_r258_staget_aligned_gate_candidate_matrix as staget,
    )

    base_report = load_json(base_report_path)
    corrected_validate_base_report(base_report, staget=staget)
    topology = extract_process_topology(
        base_report,
        expected_worker_count=int(staget.EXPECTED_WORKER_COUNT),
    )
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_STAGET_BLOCKED_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "base_stage_t": base,
        "base_stage_s_resume3_report_sha256": (
            EXPECTED_STAGES_RESUME3_REPORT_SHA256
        ),
        "historical_process_topology": topology,
    }


def _validate_recovered_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("execution_verdict") != "PASS":
        raise StageTResume1Error("recovered Stage-T execution did not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise StageTResume1Error("recovered Stage-T scientific status changed")
    if summary.get("selected_configuration") is not None:
        raise StageTResume1Error("recovered Stage-T selected a configuration")
    if summary.get("train_only_recommendation") is not None:
        raise StageTResume1Error("recovered Stage-T emitted a recommendation")
    require_false(summary, FALSE_BOUNDARIES, "recovered Stage-T summary")


def _validate_recovered_gate(gate: Mapping[str, Any]) -> None:
    if gate.get("deployment_authorized") is not False:
        raise StageTResume1Error("recovered gate authorized deployment")
    if gate.get("selected_configuration") is not None:
        raise StageTResume1Error("recovered gate selected a configuration")
    if gate.get("train_only_recommendation") is not None:
        raise StageTResume1Error("recovered gate emitted a recommendation")
    require_false(gate, FALSE_BOUNDARIES, "recovered gate contract")


def execute_recovery(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    validate_environment_variables()
    repo = Path(root).resolve()
    from ccda_phase3 import (
        phase314b_r258_staget_aligned_gate_candidate_matrix as staget,
    )

    base_report = load_json(repo / BASE_STAGES_RESUME3_REPORT)
    corrected_validate_base_report(base_report, staget=staget)
    with patched_base_report_validator(staget):
        recovered_summary, recovered_gate = staget.run_confirmation(
            root=repo,
            repository=repository,
            python_bin=str(python_bin),
        )
    _validate_recovered_summary(recovered_summary)
    _validate_recovered_gate(recovered_gate)

    recovered_summary_sha = sha256_bytes(stable_json_bytes(recovered_summary))
    recovered_gate_sha = sha256_bytes(stable_json_bytes(recovered_gate))
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": recovered_summary["scientific_status"],
        "root_cause": recovered_summary["root_cause"],
        "required_next_path": recovered_summary["required_next_path"],
        "primary_failure_locus": recovered_summary["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": dict(repository),
        "immutable_inputs": {
            "base_stage_t_implementation_commit": (
                BASE_STAGET_IMPLEMENTATION_COMMIT
            ),
            "base_stage_t_blocked_evidence_commit": (
                BASE_STAGET_BLOCKED_EVIDENCE_COMMIT
            ),
            "base_stage_t_blocked_report_sha256": (
                EXPECTED_STAGET_BLOCKED_REPORT_SHA256
            ),
            "base_stage_s_resume3_report_sha256": (
                EXPECTED_STAGES_RESUME3_REPORT_SHA256
            ),
            "schema_recovery_only": True,
            "original_stage_t_science_modified": False,
            "original_stage_t_worker_modified": False,
        },
        "historical_process_topology": dict(
            _mapping(
                repository.get("historical_process_topology"),
                "repository historical process topology",
            )
        ),
        "recovered_stage_t_result_sha256": recovered_summary_sha,
        "recovered_stage_t_result": recovered_summary,
        "recovered_stage_t_gate_contract_sha256": recovered_gate_sha,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(
        stable_json_bytes(result)
    )

    gate_wrapper: Dict[str, Any] = {
        "phase": PHASE,
        "schema": GATE_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "base_stage_t_blocked_report_sha256": (
            EXPECTED_STAGET_BLOCKED_REPORT_SHA256
        ),
        "base_stage_s_resume3_report_sha256": (
            EXPECTED_STAGES_RESUME3_REPORT_SHA256
        ),
        "recovered_stage_t_gate_contract_sha256": recovered_gate_sha,
        "recovered_stage_t_gate_contract": recovered_gate,
        "schema_recovery_only": True,
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    gate_wrapper["contract_payload_sha256"] = sha256_bytes(
        stable_json_bytes(gate_wrapper)
    )
    return result, gate_wrapper


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
            "phase314b_r258_staget_resume1_"
            "process_topology_schema_recovery_failed"
        ),
        "required_next_path": (
            "RESTORE_STAGET_RESUME1_PROCESS_TOPOLOGY_SCHEMA_RECOVERY"
        ),
        "primary_failure_locus": "execution_contract",
        "error_type": type(error).__name__,
        "error": str(error),
        "repository": None if repository is None else dict(repository),
        "base_stage_t_implementation_commit": BASE_STAGET_IMPLEMENTATION_COMMIT,
        "base_stage_t_blocked_evidence_commit": (
            BASE_STAGET_BLOCKED_EVIDENCE_COMMIT
        ),
        "base_stage_t_blocked_report_sha256": (
            EXPECTED_STAGET_BLOCKED_REPORT_SHA256
        ),
        "base_stage_s_resume3_report_sha256": (
            EXPECTED_STAGES_RESUME3_REPORT_SHA256
        ),
        "stage_t_worker_population_started": None,
        "completed_oof_fit_count_claimed": None,
        "completed_callback_pair_count_claimed": None,
        "completed_matrix_reconstruction_attempt_count_claimed": None,
        "execution_counts_not_claimed_without_success_report": True,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
