"""Stage-X Resume2: evidence-only lost-worker-result adjudication.

Stage-X Resume1 launched one disposable CUDA environment probe and one cold
objective-train science worker.  The worker returned successfully and its
payload passed the committed ``validate_worker_payload`` gate.  The controller
then read the environment-probe PID through an invented nested path
``probe[\"probe\"][\"process_id\"]`` even though the frozen probe worker writes
``process_id`` at the payload top level.  That KeyError occurred while both JSON
payloads lived inside ``TemporaryDirectory``; normal exception unwinding removed
them before any repository evidence could preserve the worker result.

This stage is deliberately evidence-only.  It validates immutable Git/report
provenance and the exact controller control flow, records that the science
worker completed but its result is not recoverable from committed evidence,
freezes the correct top-level probe PID schema for future controllers, and
forbids a Stage-X Resume1 science replay.  It does not import NumPy, Torch, CUDA,
scientific runtime modules, datasets, selection holdout, or frozen probe data.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage X Resume2"
SCHEMA = "phase314b_r258_stagex_resume2_lost_worker_result_adjudication_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_RESUME1_IMPLEMENTATION_COMMIT = "703aa39901df3d65fc3a87b7f68a1a7d2168e940"
BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT = "be66026ac6b4befb751f621f03f738ade05027d4"
BASE_RESUME1_PARENT = "b5955b7863d56d7a2c719a603816ef952aee91c3"
BASE_STAGEX_PARENT = "e0d0756f80415ea36335be6019c86422f043ff83"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

RESUME1_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage X Resume1: recover import and crossfit aggregation"
)
RESUME1_BLOCKED_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage X Resume1 blocked evidence"
)
ORIGINAL_STAGEX_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage X: calibrate tail-robust nested group OOF"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage X Resume2: adjudicate lost worker result"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage X Resume2 lost-result evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage X Resume2 blocked evidence"
)

RESUME1_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py"),
    ("A", "scripts/phase3_14b_r258_stagex_resume1_worker.py"),
    ("A", "scripts/phase3_14b_r258_stagex_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagex_resume1_import_crossfit_recovery.py"),
)
RESUME1_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "reports/phase3_14b_r258_stagex_resume1_tail_robust_nested_group_oof_blocked_summary.json",
    ),
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagex_resume2_lost_worker_result_adjudication.py"),
    ("A", "scripts/phase3_14b_r258_stagex_resume2_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagex_resume2_lost_worker_result_adjudication.py"),
)

RESUME1_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py": (
        "66195d9f4a6e01fc3168501e65e3d0d57aed82ba182bf3b728645c271d448cff"
    ),
    "scripts/phase3_14b_r258_stagex_resume1_worker.py": (
        "436e72b4c397ca61de64a1ad9026a7d1ef57f3677f33829fcfde8cf979accd87"
    ),
    "scripts/phase3_14b_r258_stagex_resume1_execute.py": (
        "a4e4cf0c9ebd0700b777559fb6d5758dda449c93a624b9d9896eee51b34e589c"
    ),
    "tests/test_phase3_14b_r258_stagex_resume1_import_crossfit_recovery.py": (
        "fc93b10869d73dcec73b3679dfde618fa8c9cea9b111970769f6e73946255f79"
    ),
}
PROBE_WORKER_PATH = "scripts/phase3_14b_r258_stages_resume2a_worker.py"
EXPECTED_PROBE_WORKER_SHA256 = (
    "096ffbedea6e4504ad01f77b637d812fb8a69c2cb8437c1a3aaa4cccf66a97d2"
)
RESUME1_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagex_resume1_tail_robust_nested_group_oof_"
    "blocked_summary.json"
)
EXPECTED_RESUME1_BLOCKED_REPORT_SHA256 = (
    "d6b7e02e71fb3041c228559a9bb089333e540db8cc49e0cffb0dc3b3c5c4f877"
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagex_resume2_lost_worker_result_"
    "adjudication_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagex_resume2_lost_worker_result_"
    "adjudication_blocked_summary.json"
)

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "environment_probe_run",
    "science_worker_run",
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded",
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
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "descriptor_tensor_persisted",
    "risk_probability_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageXResume2Error(RuntimeError):
    """Fail-closed Stage-X Resume2 error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
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
        raise StageXResume2Error("JSON root is not a mapping: {}".format(path))
    return value


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageXResume2Error("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageXResume2Error(
            "git {} failed: {}".format(" ".join(args), completed.stderr.strip())
        )
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageXResume2Error(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace").strip()
            )
        )
    return bytes(completed.stdout)


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    rows = []
    for line in output.splitlines():
        if line.strip():
            fields = line.split("\t")
            rows.append((fields[0], fields[-1]))
    return tuple(sorted(rows))


def extract_probe_process_id(probe_payload: Mapping[str, Any]) -> int:
    """Frozen correction for future controllers; never launches a probe."""
    if "probe" in probe_payload:
        raise StageXResume2Error("invented nested probe wrapper is forbidden")
    process_id = probe_payload.get("process_id")
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise StageXResume2Error("top-level probe process_id is invalid")
    return process_id


def _subscript_path(node: ast.AST) -> Optional[Tuple[Any, ...]]:
    values = []
    current = node
    while isinstance(current, ast.Subscript):
        slice_node = current.slice
        if isinstance(slice_node, ast.Constant):
            values.append(slice_node.value)
        else:
            return None
        current = current.value
    if isinstance(current, ast.Name):
        values.append(current.id)
        return tuple(reversed(values))
    return None


def controller_control_flow_audit(source: str) -> Mapping[str, Any]:
    tree = ast.parse(source)
    worker_child_line: Optional[int] = None
    worker_read_line: Optional[int] = None
    worker_validate_line: Optional[int] = None
    probe_pid_line: Optional[int] = None
    probe_pid_path: Optional[Tuple[Any, ...]] = None
    temporary_directory_line: Optional[int] = None

    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    if call.func.attr == "TemporaryDirectory":
                        temporary_directory_line = int(node.lineno)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "_child":
                segment = ast.get_source_segment(source, node) or ""
                if "cold Stage-X Resume1 worker" in segment:
                    worker_child_line = int(node.lineno)
            if isinstance(node.func, ast.Name) and node.func.id == "validate_worker_payload":
                if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "worker":
                    worker_validate_line = int(node.lineno)
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "worker" in targets:
                segment = ast.get_source_segment(source, node.value) or ""
                if "worker_path" in segment and "json.loads" in segment:
                    worker_read_line = int(node.lineno)
            if "probe_pid" in targets:
                probe_pid_line = int(node.lineno)
                value = node.value
                if isinstance(value, ast.Call) and value.args:
                    probe_pid_path = _subscript_path(value.args[0])
                else:
                    probe_pid_path = _subscript_path(value)

    required = {
        "temporary_directory_line": temporary_directory_line,
        "worker_child_line": worker_child_line,
        "worker_read_line": worker_read_line,
        "worker_validate_line": worker_validate_line,
        "probe_pid_line": probe_pid_line,
    }
    if any(value is None for value in required.values()):
        raise StageXResume2Error("Resume1 controller control-flow landmarks changed")
    ordered = [
        int(worker_child_line),
        int(worker_read_line),
        int(worker_validate_line),
        int(probe_pid_line),
    ]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise StageXResume2Error("Resume1 controller operation order changed")
    if probe_pid_path != ("probe", "probe", "process_id"):
        raise StageXResume2Error("Resume1 probe PID defect no longer matches evidence")
    return {
        **{key: int(value) for key, value in required.items()},
        "operation_order": (
            "cold_worker_return",
            "worker_payload_read",
            "worker_payload_validated",
            "invalid_probe_pid_read",
        ),
        "invalid_probe_pid_path": list(probe_pid_path),
        "correct_probe_pid_path": ["probe", "process_id"],
        "worker_completed_before_failure": True,
        "worker_payload_validated_before_failure": True,
        "failure_inside_temporary_directory": True,
    }


def probe_worker_schema_audit(source: str) -> Mapping[str, Any]:
    tree = ast.parse(source)
    process_id_output_line: Optional[int] = None
    nested_probe_literal = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "probe":
            nested_probe_literal = True
        if isinstance(node, ast.Subscript):
            path = _subscript_path(node)
            if path == ("result", "process_id"):
                process_id_output_line = int(node.lineno)
    if process_id_output_line is None:
        raise StageXResume2Error("frozen probe worker top-level process_id output changed")
    # The worker may use the word probe as an argument/mode; only the output path matters.
    return {
        "top_level_process_id_output_line": process_id_output_line,
        "probe_pid_schema": "top_level_process_id",
        "correct_controller_expression": "probe['process_id']",
        "nested_probe_wrapper_required": False,
        "source_contains_probe_token": nested_probe_literal,
    }


def validate_blocked_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "phase": "Phase3.14b-r2.5.8 Stage X Resume1",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagex_resume1_execution_contract_failed",
        "primary_failure_locus": "execution_contract",
        "error_type": "KeyError",
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "complete_nested_oof_population_claimed": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise StageXResume2Error("Resume1 blocked report field changed: {}".format(key))
    message = str(report.get("error_message", ""))
    if message not in {"'probe'", "probe"}:
        raise StageXResume2Error("Resume1 blocked error message changed")
    if report.get("selected_configuration") is not None:
        raise StageXResume2Error("Resume1 blocked report selected a configuration")
    if report.get("train_only_recommendation") is not None:
        raise StageXResume2Error("Resume1 blocked report retained a recommendation")
    next_path = str(report.get("required_next_path", ""))
    if not next_path.endswith("IF_SCIENCE_DID_NOT_START"):
        raise StageXResume2Error("Resume1 conditional recovery next path changed")
    return {
        "error_type": "KeyError",
        "error_message": message,
        "conditional_recovery_clause": "science_did_not_start",
        "conditional_recovery_clause_satisfied": False,
        "rerun_authorized": False,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageXResume2Error("Stage-X Resume2 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", head + "^") != BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT:
        raise StageXResume2Error("Stage-X Resume2 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageXResume2Error("Stage-X Resume2 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageXResume2Error("Stage-X Resume2 implementation paths changed")
    if _git(repo, "rev-parse", BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT + "^") != BASE_RESUME1_IMPLEMENTATION_COMMIT:
        raise StageXResume2Error("Resume1 blocked evidence parent changed")
    if _git(repo, "rev-parse", BASE_RESUME1_IMPLEMENTATION_COMMIT + "^") != BASE_RESUME1_PARENT:
        raise StageXResume2Error("Resume1 implementation parent changed")
    if _git(repo, "rev-parse", BASE_RESUME1_PARENT + "^") != BASE_STAGEX_PARENT:
        raise StageXResume2Error("original Stage-X parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_RESUME1_IMPLEMENTATION_COMMIT) != RESUME1_IMPLEMENTATION_SUBJECT:
        raise StageXResume2Error("Resume1 implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT) != RESUME1_BLOCKED_SUBJECT:
        raise StageXResume2Error("Resume1 blocked evidence subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_RESUME1_PARENT) != ORIGINAL_STAGEX_SUBJECT:
        raise StageXResume2Error("original Stage-X subject changed")
    if commit_name_status(repo, BASE_RESUME1_IMPLEMENTATION_COMMIT) != tuple(sorted(RESUME1_IMPLEMENTATION_PATHS)):
        raise StageXResume2Error("Resume1 implementation path population changed")
    if commit_name_status(repo, BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT) != tuple(sorted(RESUME1_EVIDENCE_PATHS)):
        raise StageXResume2Error("Resume1 evidence path population changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageXResume2Error("origin/Experiment1 changed")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageXResume2Error("main worktree is dirty")
    submodule = repo / "external/deformable-ravens"
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageXResume2Error("DeformableRavens gitlink changed")
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageXResume2Error("DeformableRavens worktree commit changed")
    if _git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageXResume2Error("DeformableRavens worktree is dirty")

    for relative, expected_sha in RESUME1_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageXResume2Error("Resume1 source changed: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(BASE_RESUME1_IMPLEMENTATION_COMMIT, relative))
        if committed != path.read_bytes():
            raise StageXResume2Error("Resume1 source differs from committed blob: {}".format(relative))

    probe_worker = repo / PROBE_WORKER_PATH
    if not probe_worker.is_file() or sha256_file(probe_worker) != EXPECTED_PROBE_WORKER_SHA256:
        raise StageXResume2Error("frozen probe worker source changed")

    blocked_path = repo / RESUME1_BLOCKED_REPORT
    if not blocked_path.is_file() or sha256_file(blocked_path) != EXPECTED_RESUME1_BLOCKED_REPORT_SHA256:
        raise StageXResume2Error("Resume1 blocked report SHA changed")
    committed_report = _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT, RESUME1_BLOCKED_REPORT),
    )
    if committed_report != blocked_path.read_bytes():
        raise StageXResume2Error("Resume1 blocked report differs from committed blob")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageXResume2Error("Stage-X Resume2 output already exists")

    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "resume1_implementation_commit": BASE_RESUME1_IMPLEMENTATION_COMMIT,
        "resume1_blocked_evidence_commit": BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT,
        "resume1_blocked_report_sha256": EXPECTED_RESUME1_BLOCKED_REPORT_SHA256,
    }


def build_adjudication(root: Path, repository: Mapping[str, Any]) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    report = load_json(repo / RESUME1_BLOCKED_REPORT)
    report_audit = validate_blocked_report(report)
    controller_source = (repo / "scripts/phase3_14b_r258_stagex_resume1_execute.py").read_text(encoding="utf-8")
    controller = controller_control_flow_audit(controller_source)
    probe_source = (repo / PROBE_WORKER_PATH).read_text(encoding="utf-8")
    probe_schema = probe_worker_schema_audit(probe_source)

    science_completed = bool(
        controller["worker_completed_before_failure"]
        and controller["worker_payload_validated_before_failure"]
        and report_audit["error_type"] == "KeyError"
    )
    if not science_completed:
        raise StageXResume2Error("cannot prove Resume1 science worker completion")

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagex_resume2_worker_completed_but_result_"
            "irrecoverably_lost_before_repository_evidence"
        ),
        "required_next_path": (
            "PREREGISTER_NEW_OBJECTIVE_TRAIN_ONLY_STAGE_WITH_DURABLE_"
            "WRITE_AHEAD_WORKER_EVIDENCE_AND_NO_HOLDOUT_OR_FROZEN_PROBE_ACCESS"
        ),
        "primary_failure_locus": "completed_worker_result_not_persisted",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": dict(repository),
        "immutable_resume1_blocked_report": {
            "path": RESUME1_BLOCKED_REPORT,
            "sha256": EXPECTED_RESUME1_BLOCKED_REPORT_SHA256,
            **report_audit,
        },
        "controller_control_flow_audit": controller,
        "probe_pid_schema_audit": probe_schema,
        "execution_loss_adjudication": {
            "environment_probe_started": True,
            "science_worker_started": True,
            "science_worker_completed": True,
            "worker_payload_read": True,
            "worker_payload_validated": True,
            "worker_payload_repository_evidence_persisted": False,
            "worker_payload_recoverable_from_committed_repository": False,
            "actual_probe_pid_recoverable_from_committed_repository": False,
            "actual_worker_pid_recoverable_from_committed_repository": False,
            "actual_outer_fold_policies_recoverable_from_committed_repository": False,
            "actual_metrics_recoverable_from_committed_repository": False,
            "actual_execution_counts_recoverable_from_committed_repository": False,
            "scientific_ready_or_blocked_classification_recoverable": False,
            "human_observation_of_lost_payload_provable_from_repository": False,
            "temporary_payload_cleanup_expected_from_source": True,
        },
        "replay_governance": {
            "stagex_resume1_science_replay_authorized": False,
            "environment_probe_reaccess_authorized": False,
            "science_worker_reexecution_authorized": False,
            "conditional_recovery_clause": "science_did_not_start",
            "conditional_recovery_clause_satisfied": False,
            "probe_pid_schema_correction_changes_science_authorization": False,
            "reason": (
                "Resume1 worker completed and validated before the controller failure; "
                "the blocked report and worker contract both freeze rerun_authorized=false."
            ),
        },
        "future_controller_contract": {
            "probe_pid_expression": "probe['process_id']",
            "nested_probe_pid_expression_forbidden": "probe['probe']['process_id']",
            "durable_worker_result_before_controller_decoration_required": True,
            "write_ahead_result_must_be_write_once": True,
            "worker_result_sha_must_be_recorded_before_temporary_cleanup": True,
            "controller_summary_may_reference_but_not_recompute_worker_science": True,
            "fresh_stage_must_be_preregistered_before_any_new_science_process": True,
        },
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["adjudication_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def validate_adjudication(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageXResume2Error("Stage-X Resume2 schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageXResume2Error("lost-result adjudication cannot be READY")
    if payload.get("selected_configuration") is not None:
        raise StageXResume2Error("lost-result adjudication selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageXResume2Error("lost-result adjudication retained a recommendation")
    if payload.get("rerun_authorized") is not False:
        raise StageXResume2Error("lost-result adjudication authorized rerun")
    governance = payload.get("replay_governance")
    if not isinstance(governance, Mapping):
        raise StageXResume2Error("replay governance missing")
    if any(governance.get(key) is not False for key in (
        "stagex_resume1_science_replay_authorized",
        "environment_probe_reaccess_authorized",
        "science_worker_reexecution_authorized",
        "conditional_recovery_clause_satisfied",
        "probe_pid_schema_correction_changes_science_authorization",
    )):
        raise StageXResume2Error("a replay boundary became true")
    loss = payload.get("execution_loss_adjudication")
    if not isinstance(loss, Mapping):
        raise StageXResume2Error("execution-loss adjudication missing")
    for key in (
        "environment_probe_started",
        "science_worker_started",
        "science_worker_completed",
        "worker_payload_read",
        "worker_payload_validated",
    ):
        if loss.get(key) is not True:
            raise StageXResume2Error("completed execution fact changed: {}".format(key))
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageXResume2Error("Stage-X Resume2 crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != "adjudication_sha256"})
    )
    if payload.get("adjudication_sha256") != expected:
        raise StageXResume2Error("Stage-X Resume2 self-hash changed")


def blocked_report(repository: Optional[Mapping[str, Any]], error: BaseException) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagex_resume2_adjudication_execution_failed",
        "required_next_path": "AUDIT_STAGEX_RESUME2_EVIDENCE_ONLY_ADJUDICATION_WITHOUT_SCIENCE_RERUN",
        "primary_failure_locus": "adjudication_execution",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
