"""Phase3.14b-r2.5.9 Stage B Resume1 post-commit evidence-audit recovery.

The original Stage-B wrapper correctly created its add-only implementation
commit and then launched the evidence-only controller.  The controller's
repository validator incorrectly required HEAD to remain at the Stage-A
evidence commit, so it failed before reading any Stage-A evidence.  This
Resume1 is add-only and evidence-only: it binds the original Stage-B
implementation and blocked evidence, validates the current Resume1
implementation commit supplied by the wrapper, and executes only the immutable
aggregate audit functions from the original Stage-B module.

No CUDA context, dataset, environment probe, science worker, fit, candidate,
selection holdout, frozen probe, DeformableRavens execution, training, reverse
sampling, IDM, Phase4, or CPS is permitted.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ccda_phase3 import phase314b_r259_stageb_outer_crossfit_tail_failure_audit as stageb

PHASE = "Phase3.14b-r2.5.9 Stage B Resume1"
SCHEMA = "phase314b_r259_stageb_resume1_postcommit_evidence_audit_recovery_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEB_IMPLEMENTATION_COMMIT = "aa1bfb4a4f719e299613badf4f9853bc59c3bd85"
BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT = "2d9568888c5583e82fb8adafad1cb867f65b7971"
BASE_STAGEB_PARENT = "3100a877f3664936f713b089e1489768975f3fa4"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_STAGEB_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage B: audit outer-crossfit tail failure"
)
BASE_STAGEB_BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage B blocked evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage B Resume1: restore post-commit evidence audit"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage B Resume1 outer-crossfit audit evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage B Resume1 blocked evidence"
)

BASE_STAGEB_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stageb_outer_crossfit_tail_failure_audit.py"),
    ("A", "scripts/phase3_14b_r259_stageb_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stageb_outer_crossfit_tail_failure_audit.py"),
)
BASE_STAGEB_IMPLEMENTATION_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r259_stageb_outer_crossfit_tail_failure_audit.py": (
        "53f52a27efa540a2d981c8c9b7aba61ed99ae1ec7307e2bb6011a6594b136895"
    ),
    "scripts/phase3_14b_r259_stageb_execute.py": (
        "fe6bbe95aa78840f158a5841dc86802008ac2127b1a2adecf448bc5725308d84"
    ),
    "tests/test_phase3_14b_r259_stageb_outer_crossfit_tail_failure_audit.py": (
        "a643a52aca63ed6997dc950f77a160b4ff9caa4c260a36055b858785744cfc9f"
    ),
}
BASE_STAGEB_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stageb_outer_crossfit_tail_failure_audit_blocked_summary.json"
)
BASE_STAGEB_BLOCKED_REPORT_SHA256 = (
    "15d226ee1eb606931af9d5ba5b4e4830b0b7c72454339c00529fe4dbb479f36f"
)
BASE_STAGEB_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", BASE_STAGEB_BLOCKED_REPORT),
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r259_stageb_resume1_postcommit_evidence_audit_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r259_stageb_resume1_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r259_stageb_resume1_postcommit_evidence_audit_recovery.py",
    ),
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_stageb_resume1_outer_crossfit_tail_failure_audit_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stageb_resume1_outer_crossfit_tail_failure_audit_blocked_summary.json"
)

FALSE_BOUNDARIES = stageb.FALSE_BOUNDARIES


class StageBResume1Error(RuntimeError):
    """Fail-closed Stage-B Resume1 recovery error."""


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
        raise StageBResume1Error("JSON root is not a mapping: {}".format(path))
    return value


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageBResume1Error("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    lines = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records = []
    for line in lines.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageBResume1Error("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise StageBResume1Error("{} worktree is dirty".format(label))


def validate_original_blocked_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise StageBResume1Error("original Stage-B blocked report is missing")
    if sha256_file(report_path) != BASE_STAGEB_BLOCKED_REPORT_SHA256:
        raise StageBResume1Error("original Stage-B blocked report SHA changed")
    payload = load_json(report_path)
    expected = {
        "phase": stageb.PHASE,
        "schema": stageb.BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageb_evidence_audit_execution_failed",
        "required_next_path": (
            "RESTORE_R259_STAGEB_EVIDENCE_ONLY_OUTER_CROSSFIT_AUDIT_WITHOUT_"
            "SCIENCE_REEXECUTION"
        ),
        "primary_failure_locus": "evidence_audit_execution_contract",
        "error_type": "StageBError",
        "error_message": "Stage-B starting HEAD changed",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageBResume1Error(
                "original Stage-B blocked field changed: {}".format(key)
            )
    if payload.get("repository") is not None:
        raise StageBResume1Error("original failure unexpectedly resolved repository")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageBResume1Error("original Stage-B blocked report crossed a boundary")
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise StageBResume1Error("Resume1 implementation commit is invalid")
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageBResume1Error("branch changed")
    if _git(repo, "rev-parse", "HEAD") != implementation_commit:
        raise StageBResume1Error("Resume1 implementation HEAD changed")
    if _git(repo, "rev-parse", "HEAD^") != BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT:
        raise StageBResume1Error("Resume1 implementation parent changed")
    if _git(repo, "rev-parse", BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT + "^") != (
        BASE_STAGEB_IMPLEMENTATION_COMMIT
    ):
        raise StageBResume1Error("original blocked evidence parent changed")
    if _git(repo, "rev-parse", BASE_STAGEB_IMPLEMENTATION_COMMIT + "^") != (
        BASE_STAGEB_PARENT
    ):
        raise StageBResume1Error("original Stage-B implementation parent changed")
    subject_checks = (
        (BASE_STAGEB_IMPLEMENTATION_COMMIT, BASE_STAGEB_IMPLEMENTATION_SUBJECT),
        (BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT, BASE_STAGEB_BLOCKED_EVIDENCE_SUBJECT),
        (implementation_commit, IMPLEMENTATION_SUBJECT),
    )
    for commit, expected_subject in subject_checks:
        if _git(repo, "show", "-s", "--format=%s", commit) != expected_subject:
            raise StageBResume1Error("commit subject changed: {}".format(commit))
    if commit_name_status(repo, BASE_STAGEB_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEB_IMPLEMENTATION_PATHS)
    ):
        raise StageBResume1Error("original Stage-B implementation paths changed")
    if commit_name_status(repo, BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_STAGEB_BLOCKED_PATHS)
    ):
        raise StageBResume1Error("original Stage-B blocked paths changed")
    if commit_name_status(repo, implementation_commit) != tuple(
        sorted(IMPLEMENTATION_PATHS)
    ):
        raise StageBResume1Error("Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageBResume1Error("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageBResume1Error("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageBResume1Error("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-B Resume1")

    for relative, expected_sha in BASE_STAGEB_IMPLEMENTATION_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageBResume1Error("original Stage-B source changed: {}".format(relative))
        committed = _git_bytes(
            repo, "show", "{}:{}".format(BASE_STAGEB_IMPLEMENTATION_COMMIT, relative)
        )
        if committed != path.read_bytes():
            raise StageBResume1Error(
                "original Stage-B source differs from committed blob: {}".format(relative)
            )

    for relative, expected_sha in stageb.BASE_STAGEA_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageBResume1Error("Stage-A source changed: {}".format(relative))
        committed = _git_bytes(
            repo,
            "show",
            "{}:{}".format(stageb.BASE_STAGEA_IMPLEMENTATION_COMMIT, relative),
        )
        if committed != path.read_bytes():
            raise StageBResume1Error(
                "Stage-A source differs from committed blob: {}".format(relative)
            )

    stagea_evidence_hashes = {
        stageb.PROBE_EVIDENCE: stageb.EXPECTED_PROBE_FILE_SHA256,
        stageb.WORKER_EVIDENCE: stageb.EXPECTED_WORKER_FILE_SHA256,
        stageb.STAGEA_SUMMARY: stageb.EXPECTED_SUMMARY_FILE_SHA256,
    }
    for relative, expected_sha in stagea_evidence_hashes.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageBResume1Error("Stage-A evidence changed: {}".format(relative))
        committed = _git_bytes(
            repo,
            "show",
            "{}:{}".format(stageb.BASE_STAGEA_EVIDENCE_COMMIT, relative),
        )
        if committed != path.read_bytes():
            raise StageBResume1Error(
                "Stage-A evidence differs from committed blob: {}".format(relative)
            )

    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageBResume1Error("Resume1 source missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageBResume1Error(
                "Resume1 source differs from committed blob: {}".format(relative)
            )

    original_blocked = repo / BASE_STAGEB_BLOCKED_REPORT
    committed_blocked = _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT, BASE_STAGEB_BLOCKED_REPORT),
    )
    if committed_blocked != original_blocked.read_bytes():
        raise StageBResume1Error("original blocked report differs from committed blob")
    validate_original_blocked_report(original_blocked)

    if (repo / stageb.SUCCESS_REPORT).exists():
        raise StageBResume1Error("original Stage-B success report unexpectedly exists")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageBResume1Error("Resume1 output already exists: {}".format(relative))

    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT,
        "original_stageb_implementation_commit": BASE_STAGEB_IMPLEMENTATION_COMMIT,
        "original_stageb_blocked_evidence_commit": BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT,
        "stagea_evidence_commit": BASE_STAGEB_PARENT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
    }


def _validate_stagea_inputs(repo: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    probe = stageb.load_json(repo / stageb.PROBE_EVIDENCE)
    worker = stageb.load_json(repo / stageb.WORKER_EVIDENCE)
    summary = stageb.load_json(repo / stageb.STAGEA_SUMMARY)
    stageb.validate_stagea_summary(summary)
    stageb.validate_worker_evidence(worker)
    protocol = summary.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageBResume1Error("Stage-A durable protocol is invalid")
    if protocol.get("probe_evidence_file_sha256") != stageb.EXPECTED_PROBE_FILE_SHA256:
        raise StageBResume1Error("summary probe evidence SHA changed")
    if protocol.get("worker_evidence_file_sha256") != stageb.EXPECTED_WORKER_FILE_SHA256:
        raise StageBResume1Error("summary worker evidence SHA changed")
    if protocol.get("worker_result_sha256") != stageb.EXPECTED_WORKER_RESULT_SHA256:
        raise StageBResume1Error("summary worker result SHA changed")
    if probe.get("execution_verdict") != "PASS":
        raise StageBResume1Error("Stage-A probe evidence changed")
    return probe, worker, summary


def run_recovery(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    repository = validate_repository(repo, implementation_commit)
    _, worker, _ = _validate_stagea_inputs(repo)

    gates = stageb.reconstruct_outer_procedure_gates(worker)
    selections = stageb.audit_outer_selections(worker)
    surface = stageb.audit_attribution_surface(worker)
    classification = stageb.classify_audit(gates, selections, surface)

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        **classification,
        "repository": repository,
        "immutable_inputs": {
            "original_stageb_implementation_commit": BASE_STAGEB_IMPLEMENTATION_COMMIT,
            "original_stageb_blocked_evidence_commit": BASE_STAGEB_BLOCKED_EVIDENCE_COMMIT,
            "original_stageb_blocked_report": BASE_STAGEB_BLOCKED_REPORT,
            "original_stageb_blocked_report_sha256": BASE_STAGEB_BLOCKED_REPORT_SHA256,
            "resume1_implementation_commit": implementation_commit,
            "stagea_implementation_commit": stageb.BASE_STAGEA_IMPLEMENTATION_COMMIT,
            "stagea_evidence_commit": stageb.BASE_STAGEA_EVIDENCE_COMMIT,
            "probe_evidence_path": stageb.PROBE_EVIDENCE,
            "probe_evidence_file_sha256": stageb.EXPECTED_PROBE_FILE_SHA256,
            "worker_evidence_path": stageb.WORKER_EVIDENCE,
            "worker_evidence_file_sha256": stageb.EXPECTED_WORKER_FILE_SHA256,
            "worker_result_sha256": stageb.EXPECTED_WORKER_RESULT_SHA256,
            "stagea_summary_path": stageb.STAGEA_SUMMARY,
            "stagea_summary_file_sha256": stageb.EXPECTED_SUMMARY_FILE_SHA256,
            "stagea_summary_self_sha256": stageb.EXPECTED_SUMMARY_SELF_SHA256,
        },
        "recovery_contract": {
            "evidence_only": True,
            "original_stageb_failure_before_stagea_evidence_read": True,
            "original_stageb_scientific_audit_completed": False,
            "stagea_science_replayed": False,
            "stagea_policy_or_threshold_changed": False,
            "dynamic_resume1_implementation_commit_bound": True,
            "repository_validator_invocation_count": 1,
            "original_stageb_controller_reexecuted": False,
            "new_fit_count": 0,
            "new_candidate_generation_count": 0,
            "new_risk_fit_count": 0,
            "new_internal_scale_attempt_count": 0,
            "new_policy_evaluation_count": 0,
        },
        "frozen_outer_procedure_gate_reconstruction": gates,
        "outer_selection_audit": selections,
        "attribution_surface_audit": surface,
        "aggregate_interpretation": {
            "t10_failure_is_coverage_not_accepted_subset_fidelity": True,
            "t25_and_t50_pass_all_frozen_procedure_checks": True,
            "two_inner_selection_fallback_folds_exist": True,
            "modal_policy_has_four_of_six_support": True,
            "full_objective_oof_policy_is_eligible": True,
            "full_objective_oof_eligibility_cannot_replace_outer_crossfit": True,
            "fold_1_or_4_outer_test_causality_claimed": False,
            "threshold_relaxation_authorized": False,
            "single_timestep_selection_authorized": False,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["audit_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_recovery_result(result, implementation_commit)
    return result


def validate_recovery_result(payload: Mapping[str, Any], implementation_commit: str) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageBResume1Error("Resume1 audit schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageBResume1Error("Resume1 audit scientific status changed")
    if payload.get("root_cause") != (
        "phase314b_r259_stageb_aggregate_evidence_localizes_t10_"
        "acceptance_coverage_failure_but_outer_test_fold_attribution_is_unavailable"
    ):
        raise StageBResume1Error("Resume1 audit root cause changed")
    if payload.get("required_next_path") != (
        "PREREGISTER_R259_STAGEC_OBJECTIVE_TRAIN_ONLY_FOLD_RESOLVED_"
        "TAIL_ATTRIBUTION_WITH_DURABLE_AGGREGATE_EVIDENCE"
    ):
        raise StageBResume1Error("Resume1 audit next path changed")
    if payload.get("primary_failure_locus") != (
        "t10_acceptance_coverage_with_fold_resolved_attribution_unavailable"
    ):
        raise StageBResume1Error("Resume1 audit failure locus changed")
    repository = payload.get("repository")
    if not isinstance(repository, Mapping) or repository.get("head") != implementation_commit:
        raise StageBResume1Error("Resume1 implementation identity changed")
    recovery = payload.get("recovery_contract")
    if not isinstance(recovery, Mapping):
        raise StageBResume1Error("Resume1 recovery contract is invalid")
    required_true = (
        "evidence_only",
        "original_stageb_failure_before_stagea_evidence_read",
        "dynamic_resume1_implementation_commit_bound",
    )
    if any(recovery.get(key) is not True for key in required_true):
        raise StageBResume1Error("Resume1 recovery truth changed")
    for key in (
        "new_fit_count",
        "new_candidate_generation_count",
        "new_risk_fit_count",
        "new_internal_scale_attempt_count",
        "new_policy_evaluation_count",
    ):
        if recovery.get(key) != 0:
            raise StageBResume1Error("Resume1 performed science: {}".format(key))
    if recovery.get("repository_validator_invocation_count") != 1:
        raise StageBResume1Error("Resume1 repository validation count changed")
    if payload.get("selected_configuration") is not None:
        raise StageBResume1Error("Resume1 selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageBResume1Error("Resume1 retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageBResume1Error("Resume1 re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageBResume1Error("Resume1 cumulative holdout count changed")
    if payload.get("rerun_authorized") is not False:
        raise StageBResume1Error("Resume1 authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageBResume1Error("Resume1 crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != "audit_sha256"})
    )
    if payload.get("audit_sha256") != expected:
        raise StageBResume1Error("Resume1 audit self-hash changed")


def blocked_report(
    repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageb_resume1_evidence_audit_recovery_failed",
        "required_next_path": (
            "RESTORE_R259_STAGEB_RESUME1_POSTCOMMIT_EVIDENCE_AUDIT_WITHOUT_"
            "SCIENCE_REEXECUTION"
        ),
        "primary_failure_locus": "evidence_audit_recovery_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
