"""Phase3.14b-r2.5.9 Stage A durable tail-robust nested-group OOF calibration.

Stage-X Resume2 established that the Stage-X Resume1 science worker completed
and returned a payload that passed validation, but the controller attempted to
read an invented nested probe PID field before persisting the worker result.
The worker payload then disappeared with its temporary directory.  The lost
result is not used here and Stage-X Resume1 is never replayed.

This is a newly preregistered objective-train-only stage.  It reuses the frozen,
leakage-correct Stage-X Resume1 scientific kernel without changing the backbone,
candidate bank, nested group folds, risk model, policy population, thresholds,
or scientific gates.  The new contribution is an evidence protocol:

1. a disposable environment probe writes a durable write-once file in a sibling
   write-ahead directory outside the Git worktree, preserving clean-worktree gates;
2. a fresh cold science worker computes the nested-OOF result, validates it, and
   writes the complete aggregate payload to the same external durable directory;
3. only after that file is closed, fsynced, atomically renamed and hashed may the
   controller promote byte-exact copies into repository reports, add process-topology
   metadata and write the final summary;
4. a controller failure after worker completion preserves the complete worker
   result for evidence-only finalization and never authorizes science replay.

No selection holdout or frozen probe is accessed.  No DeformableRavens task,
formal training, reverse sampling, IDM, candidate execution, Phase4 or CPS is
run.  No model weights, predictions, candidates, descriptors, probabilities,
NPZ files, caches, images or videos are persisted.  Only aggregate JSON evidence
is written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from ccda_phase3 import phase314b_r258_stagex_resume1_import_crossfit_recovery as kernel
from ccda_phase3 import phase314b_r258_stagex_tail_robust_nested_oof as stagex
from ccda_phase3 import phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a

PHASE = "Phase3.14b-r2.5.9 Stage A"
SCHEMA = "phase314b_r259_stagea_durable_tail_robust_nested_group_oof_v1"
PROBE_SCHEMA = SCHEMA + "_environment_probe_v1"
WORKER_SCHEMA = SCHEMA + "_worker_evidence_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_RESUME2_IMPLEMENTATION_COMMIT = "35e57957bb8b07a56e2744cd1a1cf52c40a9a81c"
BASE_RESUME2_EVIDENCE_COMMIT = "34011592bb762b350d4c7489d598a22ee9175042"
BASE_RESUME2_PARENT = "be66026ac6b4befb751f621f03f738ade05027d4"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_RESUME2_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage X Resume2: adjudicate lost worker result"
)
BASE_RESUME2_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage X Resume2 lost-result evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage A: add durable tail-robust nested OOF"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage A durable tail-robust OOF evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage A blocked evidence"
)

BASE_RESUME2_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagex_resume2_lost_worker_result_adjudication.py"),
    ("A", "scripts/phase3_14b_r258_stagex_resume2_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagex_resume2_lost_worker_result_adjudication.py"),
)
BASE_RESUME2_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "reports/phase3_14b_r258_stagex_resume2_lost_worker_result_adjudication_summary.json",
    ),
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagea_durable_tail_robust_nested_oof.py"),
    ("A", "scripts/phase3_14b_r259_stagea_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagea_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagea_durable_tail_robust_nested_oof.py"),
)

BASE_RESUME2_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagex_resume2_lost_worker_result_adjudication.py": (
        "f0cd55fb22f3216204d849fdcb96031a302dafc2a2066f8306dcf88632f3c85c"
    ),
    "scripts/phase3_14b_r258_stagex_resume2_execute.py": (
        "4828bca0f611b38a13610d25c0ae06e089955450a08bf3a638af52f8e34a92d7"
    ),
    "tests/test_phase3_14b_r258_stagex_resume2_lost_worker_result_adjudication.py": (
        "29bd5346f454a9f735e97320173e4f5739f4279d756db980699fd5818b93817b"
    ),
}
FROZEN_SCIENCE_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagex_tail_robust_nested_oof.py": (
        "289da5e41b3d8dde85b429ea3147c5f005af97f0c7d156b1b58e9389f02a9025"
    ),
    "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py": (
        "66195d9f4a6e01fc3168501e65e3d0d57aed82ba182bf3b728645c271d448cff"
    ),
}

BASE_RESUME2_REPORT = (
    "reports/phase3_14b_r258_stagex_resume2_lost_worker_result_"
    "adjudication_summary.json"
)
EXPECTED_BASE_RESUME2_REPORT_SHA256 = (
    "862bafe0a446450dcf8ec1b5b5f0a8d48075e52f9ba068b603505d2aef37da35"
)
EXPECTED_BASE_ADJUDICATION_SHA256 = (
    "f5af7ac15df498f52f9ef12d2e974c9aee41d9e857850f022f8e1d5f26c8aaa1"
)

PROBE_EVIDENCE = "reports/phase3_14b_r259_stagea_environment_probe_evidence.json"
WORKER_EVIDENCE = (
    "reports/phase3_14b_r259_stagea_tail_robust_nested_group_oof_worker_evidence.json"
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_stagea_durable_tail_robust_nested_group_oof_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stagea_durable_tail_robust_nested_group_oof_blocked_summary.json"
)
OUTPUT_PATHS: Tuple[str, ...] = (
    PROBE_EVIDENCE,
    WORKER_EVIDENCE,
    SUCCESS_REPORT,
    BLOCKED_REPORT,
)

EXPECTED_EXECUTION_COUNTS: Mapping[str, int] = {
    "direction_fit_count": 108,
    "candidate_generation_count": 324,
    "risk_fit_count": 324,
    "internal_scale_attempt_count": 2268,
    "inner_policy_evaluation_count": 216,
    "full_objective_oof_policy_evaluation_count": 36,
    "nonconverged_risk_fit_count": 0,
}

FALSE_BOUNDARIES: Tuple[str, ...] = tuple(stagex.FALSE_BOUNDARIES)


class StageAError(RuntimeError):
    """Fail-closed Phase3.14b-r2.5.9 Stage-A error."""


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
        raise StageAError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    """Write, fsync, atomically rename, and fsync the parent directory."""
    target = Path(path)
    if target.exists():
        raise StageAError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()



def promote_write_ahead(source: Path, target: Path, validator: Any) -> str:
    """Validate external durable evidence and copy exact bytes into the repository."""
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise StageAError("write-ahead source is missing: {}".format(source_path))
    payload = load_json(source_path)
    validator(payload)
    source_bytes = source_path.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    if target_path.exists():
        if target_path.read_bytes() != source_bytes:
            raise StageAError("repository evidence differs from write-ahead source")
        return source_sha
    atomic_write_once(target_path, source_bytes)
    if target_path.read_bytes() != source_bytes:
        raise StageAError("repository promotion is not byte-exact")
    return source_sha

def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageAError(
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
        raise StageAError(
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


def child_environment(root: Path) -> Mapping[str, str]:
    env = dict(os.environ)
    root_text = str(Path(root).resolve())
    prior = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join(
        [root_text] + [item for item in prior if item != root_text]
    )
    return env


def extract_probe_process_id(payload: Mapping[str, Any]) -> int:
    if "probe" in payload:
        raise StageAError("nested probe wrapper is forbidden")
    process_id = payload.get("process_id")
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise StageAError("top-level probe process_id is invalid")
    return process_id


def validate_base_adjudication(report: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema": "phase314b_r258_stagex_resume2_lost_worker_result_adjudication_v1",
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "completed_worker_result_not_persisted",
        "root_cause": (
            "phase314b_r258_stagex_resume2_worker_completed_but_result_"
            "irrecoverably_lost_before_repository_evidence"
        ),
        "required_next_path": (
            "PREREGISTER_NEW_OBJECTIVE_TRAIN_ONLY_STAGE_WITH_DURABLE_WRITE_AHEAD_"
            "WORKER_EVIDENCE_AND_NO_HOLDOUT_OR_FROZEN_PROBE_ACCESS"
        ),
        "adjudication_sha256": EXPECTED_BASE_ADJUDICATION_SHA256,
        "rerun_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise StageAError("base adjudication field changed: {}".format(key))
    governance = report.get("replay_governance")
    if not isinstance(governance, Mapping):
        raise StageAError("base replay governance missing")
    for key in (
        "stagex_resume1_science_replay_authorized",
        "environment_probe_reaccess_authorized",
        "science_worker_reexecution_authorized",
    ):
        if governance.get(key) is not False:
            raise StageAError("base replay governance changed: {}".format(key))
    future = report.get("future_controller_contract")
    if not isinstance(future, Mapping):
        raise StageAError("future controller contract missing")
    for key in (
        "durable_worker_result_before_controller_decoration_required",
        "worker_result_sha_must_be_recorded_before_temporary_cleanup",
        "write_ahead_result_must_be_write_once",
        "fresh_stage_must_be_preregistered_before_any_new_science_process",
    ):
        if future.get(key) is not True:
            raise StageAError("future controller contract changed: {}".format(key))
    expected_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in report.items() if key != "adjudication_sha256"}
        )
    )
    if report.get("adjudication_sha256") != expected_hash:
        raise StageAError("base adjudication self-hash changed")
    return report


def validate_repository(
    root: Path, *, allow_existing_write_ahead: bool = False
) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageAError("Stage-A requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", head + "^") != BASE_RESUME2_EVIDENCE_COMMIT:
        raise StageAError("Stage-A implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageAError("Stage-A implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageAError("Stage-A implementation paths changed")
    if _git(repo, "rev-parse", BASE_RESUME2_EVIDENCE_COMMIT + "^") != BASE_RESUME2_IMPLEMENTATION_COMMIT:
        raise StageAError("Resume2 evidence parent changed")
    if _git(repo, "rev-parse", BASE_RESUME2_IMPLEMENTATION_COMMIT + "^") != BASE_RESUME2_PARENT:
        raise StageAError("Resume2 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_RESUME2_IMPLEMENTATION_COMMIT) != BASE_RESUME2_IMPLEMENTATION_SUBJECT:
        raise StageAError("Resume2 implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_RESUME2_EVIDENCE_COMMIT) != BASE_RESUME2_EVIDENCE_SUBJECT:
        raise StageAError("Resume2 evidence subject changed")
    if commit_name_status(repo, BASE_RESUME2_IMPLEMENTATION_COMMIT) != tuple(sorted(BASE_RESUME2_IMPLEMENTATION_PATHS)):
        raise StageAError("Resume2 implementation paths changed")
    if commit_name_status(repo, BASE_RESUME2_EVIDENCE_COMMIT) != tuple(sorted(BASE_RESUME2_EVIDENCE_PATHS)):
        raise StageAError("Resume2 evidence paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageAError("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageAError("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageAError("DeformableRavens worktree commit changed")
    if _git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageAError("DeformableRavens worktree is not clean")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageAError("main worktree is not clean")

    for relative, expected_sha in BASE_RESUME2_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageAError("Resume2 source changed: {}".format(relative))
        committed = _git_bytes(
            repo, "show", "{}:{}".format(BASE_RESUME2_IMPLEMENTATION_COMMIT, relative)
        )
        if sha256_bytes(committed) != expected_sha or committed != path.read_bytes():
            raise StageAError("Resume2 source differs from committed blob: {}".format(relative))
    for relative, expected_sha in FROZEN_SCIENCE_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageAError("frozen science source changed: {}".format(relative))

    report_path = repo / BASE_RESUME2_REPORT
    if not report_path.is_file() or sha256_file(report_path) != EXPECTED_BASE_RESUME2_REPORT_SHA256:
        raise StageAError("Resume2 adjudication report changed")
    committed_report = _git_bytes(
        repo, "show", "{}:{}".format(BASE_RESUME2_EVIDENCE_COMMIT, BASE_RESUME2_REPORT)
    )
    if committed_report != report_path.read_bytes():
        raise StageAError("Resume2 adjudication differs from committed blob")
    validate_base_adjudication(load_json(report_path))
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageAError("Stage-A terminal output already exists: {}".format(relative))
    if not allow_existing_write_ahead:
        for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE):
            if (repo / relative).exists():
                raise StageAError("Stage-A write-ahead evidence already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_RESUME2_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "base_resume2_implementation_commit": BASE_RESUME2_IMPLEMENTATION_COMMIT,
        "base_resume2_evidence_commit": BASE_RESUME2_EVIDENCE_COMMIT,
        "base_resume2_report_sha256": EXPECTED_BASE_RESUME2_REPORT_SHA256,
        "base_adjudication_sha256": EXPECTED_BASE_ADJUDICATION_SHA256,
    }


def make_probe_evidence(root: Path) -> Mapping[str, Any]:
    """Run only inside the disposable probe subprocess."""
    payload = dict(resume2a.environment_probe_payload(Path(root).resolve()))
    process_id = extract_probe_process_id(payload)
    environment = resume2a.validate_probe_payload(payload)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": process_id,
        "source_probe_schema": payload.get("schema"),
        "environment": copy.deepcopy(environment),
        "environment_sha256": str(payload["environment_sha256"]),
        "source_probe_payload_sha256": sha256_bytes(stable_json_bytes(payload)),
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    result["probe_evidence_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_probe_evidence(result)
    return result


def validate_probe_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != PROBE_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageAError("Stage-A probe schema/verdict changed")
    extract_probe_process_id(payload)
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise StageAError("Stage-A environment payload missing")
    expected_environment_sha = resume2a.sha256_bytes(resume2a.stable_json_bytes(environment))
    if payload.get("environment_sha256") != expected_environment_sha:
        raise StageAError("Stage-A environment SHA changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "probe_evidence_sha256"}
        )
    )
    if payload.get("probe_evidence_sha256") != expected:
        raise StageAError("Stage-A probe self-hash changed")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageAError("Stage-A probe is not durable evidence")
    if payload.get("write_ahead_location_outside_git_worktree") is not True:
        raise StageAError("Stage-A probe write-ahead location changed")
    return environment


def _new_classification(kernel_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    status = kernel_payload.get("scientific_status")
    locus = kernel_payload.get("primary_failure_locus")
    if status == "READY":
        return {
            "scientific_status": "READY",
            "root_cause": "phase314b_r259_stagea_durable_tail_robust_nested_oof_ready",
            "required_next_path": (
                "FREEZE_R259_STAGEA_TRAIN_ONLY_POLICY_AND_PREREGISTER_ONE_SHOT_"
                "FROZEN_PROBE_EVALUATION"
            ),
            "primary_failure_locus": "durable_tail_robust_nested_oof_ready",
        }
    next_by_locus = {
        "outer_crossfit_tail_fidelity_failure": (
            "AUDIT_R259_STAGEA_OUTER_CROSSFIT_TAIL_FAILURE_ON_OBJECTIVE_TRAIN_ONLY"
        ),
        "inner_selection_instability": (
            "AUDIT_R259_STAGEA_INNER_SELECTION_INSTABILITY_ON_OBJECTIVE_TRAIN_ONLY"
        ),
        "policy_stability_failure": (
            "AUDIT_R259_STAGEA_POLICY_STABILITY_ON_OBJECTIVE_TRAIN_ONLY"
        ),
        "full_objective_oof_policy_failure": (
            "AUDIT_R259_STAGEA_FIXED_POLICY_TUNING_SURFACE_ON_OBJECTIVE_TRAIN_ONLY"
        ),
    }
    return {
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagea_durable_tail_robust_nested_oof_blocked",
        "required_next_path": next_by_locus.get(
            str(locus),
            "AUDIT_R259_STAGEA_OBJECTIVE_TRAIN_ONLY_TAIL_ROBUST_FAILURE",
        ),
        "primary_failure_locus": locus,
    }


def make_worker_evidence(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run the frozen science kernel in one fresh cold process."""
    validate_probe_evidence(probe_payload)
    kernel_payload = kernel.run_nested_oof(
        root=Path(root).resolve(),
        probe_payload={
            "mode": "environment_probe",
            "execution_verdict": "PASS",
            "process_id": extract_probe_process_id(probe_payload),
            "environment": copy.deepcopy(probe_payload["environment"]),
            "environment_sha256": probe_payload["environment_sha256"],
            "process_disposable": True,
            "cuda_initialization_allowed_in_this_process": True,
        },
        repository_head=str(repository_head),
        stageu_contract=stageu_contract,
    )
    kernel.validate_worker_payload(kernel_payload)
    classification = _new_classification(kernel_payload)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        **classification,
        "process_id": int(kernel_payload["process_id"]),
        "repository_head": str(repository_head),
        "environment_sha256": str(kernel_payload["environment_sha256"]),
        "cold_cuda_precheck": copy.deepcopy(kernel_payload["cold_cuda_precheck"]),
        "preregistration_contract": {
            "new_stage_identifier": PHASE,
            "stagex_resume1_stage_reexecuted": False,
            "frozen_scientific_kernel_reused": True,
            "new_stage_science_process": True,
            "lost_stagex_resume1_result_used": False,
            "new_science_process_authorized_by": EXPECTED_BASE_ADJUDICATION_SHA256,
            "durable_worker_evidence_required_before_controller_decoration": True,
            "worker_evidence_path": WORKER_EVIDENCE,
            "selection_holdout_remains_closed": True,
            "frozen_probe_remains_closed": True,
        },
        "scientific_kernel": {
            "module": "phase314b_r258_stagex_resume1_import_crossfit_recovery",
            "source_sha256": FROZEN_SCIENCE_SOURCE_SHA256[
                "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py"
            ],
            "kernel_scientific_result_sha256": kernel_payload[
                "scientific_result_sha256"
            ],
            "kernel_root_cause": kernel_payload["root_cause"],
            "kernel_required_next_path": kernel_payload["required_next_path"],
            "kernel_primary_failure_locus": kernel_payload["primary_failure_locus"],
        },
        "locked_scientific_base": copy.deepcopy(kernel_payload["locked_scientific_base"]),
        "stagex_spec": copy.deepcopy(kernel_payload["stagex_spec"]),
        "policy_population": copy.deepcopy(kernel_payload["policy_population"]),
        "population": copy.deepcopy(kernel_payload["population"]),
        "outer_fold_selections": copy.deepcopy(kernel_payload["outer_fold_selections"]),
        "inner_selection_modal_policy_diagnostic": copy.deepcopy(
            kernel_payload["inner_selection_modal_policy_diagnostic"]
        ),
        "outer_crossfit_fold_selected_policy_records": copy.deepcopy(
            kernel_payload["outer_crossfit_fold_selected_policy_records"]
        ),
        "outer_crossfit_baseline_records": copy.deepcopy(
            kernel_payload["outer_crossfit_baseline_records"]
        ),
        "full_objective_oof_fixed_policy_selection": copy.deepcopy(
            kernel_payload["full_objective_oof_fixed_policy_selection"]
        ),
        "full_objective_oof_fixed_policy_records": copy.deepcopy(
            kernel_payload["full_objective_oof_fixed_policy_records"]
        ),
        "procedure_gate_totals": copy.deepcopy(kernel_payload["procedure_gate_totals"]),
        "execution_counts": copy.deepcopy(kernel_payload["execution_counts"]),
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(
            kernel_payload.get("train_only_recommendation")
        ),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluated_in_r259_stagea": False,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["worker_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_worker_evidence(result)
    return result


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageAError("Stage-A worker schema/verdict changed")
    if payload.get("repository_head") is None:
        raise StageAError("Stage-A worker lacks repository HEAD")
    if payload.get("selected_configuration") is not None:
        raise StageAError("Stage-A selected a deployment configuration")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageAError("Stage-A re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageAError("selection-holdout cumulative count changed")
    if payload.get("rerun_authorized") is not False:
        raise StageAError("Stage-A authorized rerun")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageAError("Stage-A worker is not durable evidence")
    if payload.get("write_ahead_location_outside_git_worktree") is not True:
        raise StageAError("Stage-A worker write-ahead location changed")
    if payload.get("execution_counts") != EXPECTED_EXECUTION_COUNTS:
        raise StageAError("Stage-A execution counts changed")
    records = payload.get("outer_crossfit_fold_selected_policy_records")
    if not isinstance(records, Mapping) or set(records) != {"10", "25", "50"}:
        raise StageAError("Stage-A outer crossfit record population changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageAError("Stage-A crossed a forbidden boundary")
    if payload.get("scientific_status") == "READY":
        if payload.get("train_only_recommendation") is None:
            raise StageAError("READY worker lacks train-only recommendation")
    elif payload.get("scientific_status") == "BLOCKED":
        if payload.get("train_only_recommendation") is not None:
            raise StageAError("BLOCKED worker retained recommendation")
    else:
        raise StageAError("Stage-A scientific status changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected:
        raise StageAError("Stage-A worker self-hash changed")


def build_summary(
    *,
    repository: Mapping[str, Any],
    probe_path: Path,
    worker_path: Path,
) -> Mapping[str, Any]:
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    validate_probe_evidence(probe)
    validate_worker_evidence(worker)
    probe_pid = extract_probe_process_id(probe)
    worker_pid = worker.get("process_id")
    if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid <= 0:
        raise StageAError("Stage-A worker PID is invalid")
    if probe_pid == worker_pid:
        raise StageAError("probe and science worker PIDs are not distinct")
    probe_sha = sha256_file(probe_path)
    worker_sha = sha256_file(worker_path)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": dict(repository),
        "durable_evidence_protocol": {
            "probe_evidence_path": PROBE_EVIDENCE,
            "probe_evidence_file_sha256": probe_sha,
            "probe_evidence_self_sha256": probe["probe_evidence_sha256"],
            "worker_evidence_path": WORKER_EVIDENCE,
            "worker_evidence_file_sha256": worker_sha,
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_result_persisted_before_controller_decoration": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_evidence_promoted_byte_exact": True,
            "worker_result_sha_recorded_before_cleanup": True,
            "temporary_worker_payload_used": False,
            "controller_recomputed_worker_science": False,
            "write_ahead_files_write_once": True,
        },
        "process_topology": {
            "environment_probe_process_count": 1,
            "cold_science_worker_process_count": 1,
            "probe_process_id": probe_pid,
            "science_worker_process_id": worker_pid,
            "processes_distinct": True,
            "probe_pid_schema": "top_level_process_id",
            "child_pythonpath_explicit": True,
        },
        "worker_scientific_result": {
            "scientific_status": worker["scientific_status"],
            "root_cause": worker["root_cause"],
            "required_next_path": worker["required_next_path"],
            "primary_failure_locus": worker["primary_failure_locus"],
            "execution_counts": copy.deepcopy(worker["execution_counts"]),
            "train_only_recommendation": copy.deepcopy(
                worker.get("train_only_recommendation")
            ),
        },
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(
            worker.get("train_only_recommendation")
        ),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["summary_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_summary(result)
    return result


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageAError("Stage-A summary schema/verdict changed")
    protocol = payload.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageAError("Stage-A durable evidence protocol missing")
    for key in (
        "worker_result_persisted_before_controller_decoration",
        "worker_result_sha_recorded_before_cleanup",
        "write_ahead_files_write_once",
        "external_write_ahead_outside_git_worktree",
        "repository_evidence_promoted_byte_exact",
    ):
        if protocol.get(key) is not True:
            raise StageAError("Stage-A durable protocol changed: {}".format(key))
    if protocol.get("temporary_worker_payload_used") is not False:
        raise StageAError("Stage-A used a temporary worker payload")
    if protocol.get("controller_recomputed_worker_science") is not False:
        raise StageAError("Stage-A controller recomputed science")
    topology = payload.get("process_topology")
    if not isinstance(topology, Mapping) or topology.get("processes_distinct") is not True:
        raise StageAError("Stage-A process topology changed")
    if payload.get("selected_configuration") is not None:
        raise StageAError("Stage-A summary selected a deployment configuration")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageAError("Stage-A summary re-accessed selection holdout")
    if payload.get("rerun_authorized") is not False:
        raise StageAError("Stage-A summary authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageAError("Stage-A summary crossed a forbidden boundary")
    if payload.get("scientific_status") == "READY":
        if payload.get("train_only_recommendation") is None:
            raise StageAError("READY summary lacks recommendation")
    elif payload.get("scientific_status") == "BLOCKED":
        if payload.get("train_only_recommendation") is not None:
            raise StageAError("BLOCKED summary retained recommendation")
    else:
        raise StageAError("Stage-A summary scientific status changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageAError("Stage-A summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Path,
    worker_path: Path,
) -> Mapping[str, Any]:
    probe_reference: Optional[Mapping[str, Any]] = None
    worker_reference: Optional[Mapping[str, Any]] = None
    durable_worker_preserved = False
    try:
        if probe_path.is_file():
            probe = load_json(probe_path)
            validate_probe_evidence(probe)
            probe_reference = {
                "path": PROBE_EVIDENCE,
                "file_sha256": sha256_file(probe_path),
                "probe_evidence_sha256": probe["probe_evidence_sha256"],
                "process_id": extract_probe_process_id(probe),
            }
    except BaseException as probe_error:
        probe_reference = {"validation_error": repr(probe_error)}
    try:
        if worker_path.is_file():
            worker = load_json(worker_path)
            validate_worker_evidence(worker)
            durable_worker_preserved = True
            worker_reference = {
                "path": WORKER_EVIDENCE,
                "file_sha256": sha256_file(worker_path),
                "worker_result_sha256": worker["worker_result_sha256"],
                "scientific_status": worker["scientific_status"],
                "root_cause": worker["root_cause"],
                "required_next_path": worker["required_next_path"],
                "primary_failure_locus": worker["primary_failure_locus"],
                "train_only_recommendation": copy.deepcopy(
                    worker.get("train_only_recommendation")
                ),
                "execution_counts": copy.deepcopy(worker["execution_counts"]),
            }
    except BaseException as worker_error:
        worker_reference = {"validation_error": repr(worker_error)}
    if durable_worker_preserved:
        next_path = (
            "FINALIZE_R259_STAGEA_DURABLE_WORKER_EVIDENCE_WITHOUT_SCIENCE_REEXECUTION"
        )
        locus = "controller_failed_after_durable_worker_persistence"
    else:
        next_path = (
            "DESIGN_ADD_ONLY_R259_STAGEA_EXECUTION_RECOVERY_ONLY_IF_SCIENCE_DID_NOT_COMPLETE"
        )
        locus = "execution_contract_before_durable_worker_persistence"
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagea_execution_contract_failed",
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "durable_probe_evidence": probe_reference,
        "durable_worker_evidence": worker_reference,
        "durable_worker_result_preserved": durable_worker_preserved,
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["blocked_report_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result
