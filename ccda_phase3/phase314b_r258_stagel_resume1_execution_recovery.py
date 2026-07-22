"""Stage-L Resume1 deterministic-environment execution recovery.

The original Stage-L implementation and all nine test gates completed, but the
write-once consolidated entrypoint stopped in ``validate_environment`` before
``run_once`` because the launching command omitted the deterministic process
environment.  This module does not modify or reimplement Stage-L science.  It:

* binds the original Stage-L implementation commit;
* binds and preserves the write-once deterministic-environment blocked report;
* validates an add-only Resume1 commit;
* imports the unchanged Stage-L consolidated module by path;
* calls its unchanged ``run_once`` exactly once in a correctly seeded process;
* writes one independent Resume1 success or blocked report.

The existing Stage-L test gates are not replayed.  The only new test population
is the focused Resume1 execution-contract test file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage L Resume1"
PHASE_ID = "phase314b_r258_stagel_resume1"
SCHEMA = "phase314b_r258_stagel_resume1_execution_recovery_v1"
BLOCKED_SCHEMA = "phase314b_r258_stagel_resume1_execution_recovery_blocked_v1"

EXPECTED_BRANCH = "Experiment1"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEK_EVIDENCE_COMMIT = "70f55e62aea557194737528f942bfde697f984b5"
STAGEL_IMPLEMENTATION_COMMIT = "cd7fe38b1013b13ca84086bedd64dfa5eaacff47"
STAGEL_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage L: audit explicit callback predicate assembly"
)
STAGEL_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagel_predicate_assembly_audit.py"),
    ("M", "scripts/phase3_14b_r258_stagef_consolidated_e2e.py"),
    ("A", "tests/test_phase3_14b_r258_stagel_predicate_assembly_audit.py"),
)

PROVENANCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage L deterministic-environment blocked provenance"
)
RESUME1_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage L Resume1: recover deterministic execution"
)
RESUME1_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagel_resume1_execution_recovery.py"),
    ("A", "scripts/phase3_14b_r258_stagel_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagel_resume1_execution_recovery.py"),
)

STAGEL_SCIENTIFIC_MODULE = (
    "ccda_phase3/phase314b_r258_stagel_predicate_assembly_audit.py"
)
STAGEL_CONSOLIDATED_RUNNER = "scripts/phase3_14b_r258_stagef_consolidated_e2e.py"
STAGEL_TEST = "tests/test_phase3_14b_r258_stagel_predicate_assembly_audit.py"
STAGEL_BOUND_PATHS = (
    STAGEL_SCIENTIFIC_MODULE,
    STAGEL_CONSOLIDATED_RUNNER,
    STAGEL_TEST,
)

STAGEL_SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagel_predicate_assembly_audit_summary.json"
)
STAGEL_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagel_predicate_assembly_audit_blocked_summary.json"
)
EXPECTED_STAGEL_BLOCKED_SHA256 = (
    "eaa9aedcc801fbf96175a68e0e0d958b6481a91e51ea2e5bbe610b212dc3f088"
)

RESUME1_SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagel_resume1_predicate_assembly_audit_summary.json"
)
RESUME1_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagel_resume1_predicate_assembly_audit_blocked_summary.json"
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

PRIOR_TEST_COUNTS: Tuple[int, ...] = (28, 29, 81, 24, 21, 18, 15, 15, 138)
FALSE_BOUNDARIES: Tuple[str, ...] = (
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
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class StageLResume1Error(RuntimeError):
    """Raised when Resume1 provenance or execution invariants fail."""


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageLResume1Error("JSON root is not an object: {}".format(path))
    return value


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise StageLResume1Error("refusing to overwrite write-once output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.{}".format(os.getpid()))
    if temporary.exists():
        raise StageLResume1Error("temporary output already exists: {}".format(temporary))
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _run_git(root: Path, *args: str, binary: bool = False) -> Any:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageLResume1Error(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace").strip()
            )
        )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", "strict").strip()


def git(root: Path, *args: str) -> str:
    return str(_run_git(root, *args, binary=False))


def git_bytes(root: Path, *args: str) -> bytes:
    return bytes(_run_git(root, *args, binary=True))


def commit_parent(root: Path, commit: str) -> str:
    parents = git(root, "show", "-s", "--format=%P", commit).split()
    if len(parents) != 1:
        raise StageLResume1Error(
            "commit must have exactly one parent: {} -> {!r}".format(commit, parents)
        )
    return parents[0]


def commit_subject(root: Path, commit: str) -> str:
    return git(root, "show", "-s", "--format=%s", commit)


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    raw = git_bytes(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        "--no-renames",
        commit,
    ).split(b"\0")
    if raw and raw[-1] == b"":
        raw.pop()
    if len(raw) % 2:
        raise StageLResume1Error("incomplete diff-tree record for {}".format(commit))
    return tuple(
        (
            raw[index].decode("ascii", "strict"),
            raw[index + 1].decode("utf-8", "strict"),
        )
        for index in range(0, len(raw), 2)
    )


def require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise StageLResume1Error(
            "{} changed: expected={!r}, actual={!r}".format(label, expected, actual)
        )


def require_clean(root: Path, label: str) -> None:
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise StageLResume1Error("{} worktree is not clean: {}".format(label, status))


def require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise StageLResume1Error(
            "required ancestor missing: {} !<= {}".format(ancestor, descendant)
        )


def validate_environment(environment: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    observed = os.environ if environment is None else environment
    mismatch = {
        key: {"expected": expected, "actual": observed.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if observed.get(key) != expected
    }
    if mismatch:
        raise StageLResume1Error(
            "deterministic environment changed: "
            + json.dumps(mismatch, sort_keys=True)
        )
    return dict(EXPECTED_ENV)


def validate_original_blocked_report(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise StageLResume1Error("Stage-L blocked report is missing")
    actual_sha = sha256_path(path)
    require_equal(
        "Stage-L blocked report SHA256",
        actual_sha,
        EXPECTED_STAGEL_BLOCKED_SHA256,
    )
    report = load_json(path)
    require_equal("blocked execution verdict", report.get("execution_verdict"), "BLOCKED")
    require_equal("blocked scientific status", report.get("scientific_status"), "BLOCKED")
    require_equal(
        "blocked root cause",
        report.get("root_cause"),
        "phase314b_r258_stagel_consolidated_execution_failed",
    )
    error = report.get("error")
    if not isinstance(error, str) or "deterministic environment changed" not in error:
        raise StageLResume1Error("blocked report is not the deterministic-environment failure")
    if report.get("selected_configuration") is not None:
        raise StageLResume1Error("blocked report selected a configuration")
    if report.get("train_only_recommendation") is not None:
        raise StageLResume1Error("blocked report emitted a recommendation")
    for key in FALSE_BOUNDARIES:
        if key in report and report.get(key) is not False:
            raise StageLResume1Error("blocked report boundary changed: {}".format(key))
    if "scientific_result" in report:
        raise StageLResume1Error("blocked report unexpectedly contains a scientific result")
    return report


def _require_blob_equal(root: Path, commit: str, relative: str) -> str:
    committed = git_bytes(root, "show", "{}:{}".format(commit, relative))
    current = (root / relative).read_bytes()
    if current != committed:
        raise StageLResume1Error(
            "current file differs from bound commit {}: {}".format(commit, relative)
        )
    return sha256_bytes(current)


def validate_repository(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    require_equal(
        "repository root",
        git(repository_root, "rev-parse", "--show-toplevel"),
        str(repository_root),
    )
    require_equal(
        "branch",
        git(repository_root, "branch", "--show-current"),
        EXPECTED_BRANCH,
    )

    head = git(repository_root, "rev-parse", "HEAD")
    provenance = commit_parent(repository_root, head)
    require_equal(
        "Resume1 implementation subject",
        commit_subject(repository_root, head),
        RESUME1_IMPLEMENTATION_SUBJECT,
    )
    require_equal(
        "Resume1 implementation paths",
        commit_name_status(repository_root, head),
        RESUME1_IMPLEMENTATION_PATHS,
    )
    require_equal(
        "provenance subject",
        commit_subject(repository_root, provenance),
        PROVENANCE_SUBJECT,
    )
    require_equal(
        "provenance paths",
        commit_name_status(repository_root, provenance),
        (("A", STAGEL_BLOCKED_REPORT),),
    )
    require_equal(
        "provenance parent",
        commit_parent(repository_root, provenance),
        STAGEL_IMPLEMENTATION_COMMIT,
    )

    require_equal(
        "Stage-L implementation parent",
        commit_parent(repository_root, STAGEL_IMPLEMENTATION_COMMIT),
        STAGEK_EVIDENCE_COMMIT,
    )
    require_equal(
        "Stage-L implementation subject",
        commit_subject(repository_root, STAGEL_IMPLEMENTATION_COMMIT),
        STAGEL_IMPLEMENTATION_SUBJECT,
    )
    require_equal(
        "Stage-L implementation paths",
        commit_name_status(repository_root, STAGEL_IMPLEMENTATION_COMMIT),
        STAGEL_IMPLEMENTATION_PATHS,
    )
    require_ancestor(repository_root, STAGEK_EVIDENCE_COMMIT, head)
    require_ancestor(repository_root, STAGEL_IMPLEMENTATION_COMMIT, head)

    remote = git(repository_root, "rev-parse", "refs/remotes/origin/Experiment1")
    require_equal("origin/Experiment1", remote, EXPECTED_REMOTE_HEAD)
    gitlink = git(repository_root, "rev-parse", "HEAD:external/deformable-ravens")
    require_equal("submodule gitlink", gitlink, EXPECTED_SUBMODULE)
    submodule = repository_root / "external/deformable-ravens"
    require_equal("submodule worktree", git(submodule, "rev-parse", "HEAD"), EXPECTED_SUBMODULE)
    require_clean(submodule, "DeformableRavens")
    require_clean(repository_root, "main")

    if (repository_root / STAGEL_SUCCESS_REPORT).exists():
        raise StageLResume1Error("original Stage-L success report unexpectedly exists")
    blocked = validate_original_blocked_report(repository_root / STAGEL_BLOCKED_REPORT)
    blocked_blob = git_bytes(
        repository_root,
        "show",
        "{}:{}".format(provenance, STAGEL_BLOCKED_REPORT),
    )
    require_equal(
        "blocked report committed blob SHA256",
        sha256_bytes(blocked_blob),
        EXPECTED_STAGEL_BLOCKED_SHA256,
    )
    if blocked_blob != (repository_root / STAGEL_BLOCKED_REPORT).read_bytes():
        raise StageLResume1Error("blocked report differs from provenance commit")

    for relative in (RESUME1_SUCCESS_REPORT, RESUME1_BLOCKED_REPORT):
        if (repository_root / relative).exists():
            raise StageLResume1Error("Resume1 output already exists: {}".format(relative))

    stage_l_hashes = {
        relative: _require_blob_equal(
            repository_root, STAGEL_IMPLEMENTATION_COMMIT, relative
        )
        for relative in STAGEL_BOUND_PATHS
    }
    resume1_hashes = {
        relative: sha256_path(repository_root / relative)
        for _, relative in RESUME1_IMPLEMENTATION_PATHS
    }
    return {
        "branch": EXPECTED_BRANCH,
        "head": head,
        "resume1_implementation_parent": provenance,
        "resume1_implementation_paths": [list(item) for item in RESUME1_IMPLEMENTATION_PATHS],
        "stage_l_implementation_commit": STAGEL_IMPLEMENTATION_COMMIT,
        "stage_l_implementation_parent": STAGEK_EVIDENCE_COMMIT,
        "stage_l_implementation_paths": [list(item) for item in STAGEL_IMPLEMENTATION_PATHS],
        "stage_l_blocked_provenance_commit": provenance,
        "stage_l_blocked_report": STAGEL_BLOCKED_REPORT,
        "stage_l_blocked_report_sha256": EXPECTED_STAGEL_BLOCKED_SHA256,
        "stage_l_blocked_error": blocked.get("error"),
        "stage_l_source_sha256": stage_l_hashes,
        "resume1_source_sha256": resume1_hashes,
        "remote_head": remote,
        "submodule": EXPECTED_SUBMODULE,
        "prior_test_counts": list(PRIOR_TEST_COUNTS),
        "prior_test_counts_source": "operator-attested; not rerun by Resume1",
    }


def load_stage_l_runner(root: Path) -> ModuleType:
    path = (Path(root).resolve() / STAGEL_CONSOLIDATED_RUNNER).resolve()
    expected = (Path(root).resolve() / STAGEL_CONSOLIDATED_RUNNER).resolve()
    require_equal("Stage-L runner path", path, expected)
    name = "_phase314b_r258_stagel_resume1_bound_runner"
    specification = importlib.util.spec_from_file_location(name, str(path))
    if specification is None or specification.loader is None:
        raise StageLResume1Error("cannot load Stage-L consolidated runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for attribute in ("run_once", "stable_json_bytes", "sha256_bytes"):
        if not callable(getattr(module, attribute, None)):
            raise StageLResume1Error("Stage-L runner lacks callable {}".format(attribute))
    return module


def build_stage_l_repository_mapping(
    root: Path,
    repository: Mapping[str, Any],
    runner: ModuleType,
) -> Dict[str, Any]:
    stagee_immutable = tuple(getattr(runner, "STAGEE_IMMUTABLE_PATHS"))
    stagee_anchor = str(getattr(runner, "STAGEE_ANCHOR"))
    stagee_scientific = str(getattr(runner, "STAGEE_SCIENTIFIC_PATH"))
    stagee_hashes: Dict[str, str] = {}
    for relative in stagee_immutable:
        committed = git_bytes(root, "show", "{}:{}".format(stagee_anchor, relative))
        current = (root / relative).read_bytes()
        if current != committed:
            raise StageLResume1Error("Stage-E immutable input changed: {}".format(relative))
        stagee_hashes[relative] = sha256_bytes(current)
    stagee_hashes[stagee_scientific] = sha256_path(root / stagee_scientific)
    stagee_hashes[stagee_scientific + ":stagel_blob"] = sha256_bytes(
        git_bytes(root, "show", "{}:{}".format(STAGEL_IMPLEMENTATION_COMMIT, stagee_scientific))
    )
    value = dict(repository)
    value["stagee_input_sha256"] = stagee_hashes
    value["root_argument_transport"] = {
        "raw_repr": repr(str(root)),
        "normalized": str(root),
        "changed": False,
        "removed_suffix_codepoints": [],
        "removed_suffix_length": 0,
        "general_whitespace_stripped": False,
        "interior_line_terminators_allowed": False,
    }
    value["resolved_root"] = str(root)
    return value


def _require_false(mapping: Mapping[str, Any], keys: Iterable[str], prefix: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageLResume1Error(
                "{} boundary changed: {}={!r}".format(prefix, key, mapping.get(key))
            )


def validate_stage_l_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    require_equal("Stage-L execution verdict", summary.get("execution_verdict"), "PASS")
    require_equal("Stage-L scientific status", summary.get("scientific_status"), "BLOCKED")
    if summary.get("selected_configuration") is not None:
        raise StageLResume1Error("Stage-L selected a configuration")
    if summary.get("train_only_recommendation") is not None:
        raise StageLResume1Error("Stage-L emitted a recommendation")
    execution = summary.get("execution")
    if not isinstance(execution, Mapping):
        raise StageLResume1Error("Stage-L summary lacks execution mapping")
    single_run = execution.get("single_run_result_sha256")
    if not isinstance(single_run, str) or HEX64.fullmatch(single_run) is None:
        raise StageLResume1Error("Stage-L single-run SHA is invalid")
    require_equal("Stage-L Python process count", execution.get("python_process_count"), 1)
    require_equal("Stage-L child Python count", execution.get("child_python_process_count"), 0)

    scientific = summary.get("scientific_result")
    if not isinstance(scientific, Mapping):
        raise StageLResume1Error("Stage-L summary lacks scientific_result")
    require_equal(
        "predicate assembly completed",
        scientific.get("predicate_assembly_audit_completed"),
        True,
    )
    audit = scientific.get("predicate_assembly_audit")
    if not isinstance(audit, Mapping):
        raise StageLResume1Error("Stage-L scientific result lacks assembly audit")
    require_equal("callback pair count", audit.get("callback_off_on_pair_count"), 132)
    require_equal("callback bit exact", audit.get("all_callback_results_bit_exact"), True)
    require_equal("callback events persisted", audit.get("callback_events_persisted"), False)
    _require_false(scientific, FALSE_BOUNDARIES, "Stage-L scientific")
    boundaries = summary.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise StageLResume1Error("Stage-L summary lacks boundaries")
    _require_false(boundaries, FALSE_BOUNDARIES, "Stage-L summary")
    return {
        "single_run_result_sha256": single_run,
        "root_cause": summary.get("root_cause"),
        "required_next_path": summary.get("required_next_path"),
        "primary_failure_locus": scientific.get("primary_failure_locus"),
        "callback_pair_count": 132,
    }


def run_resume1(root: Path, repository: Mapping[str, Any]) -> Mapping[str, Any]:
    runner = load_stage_l_runner(root)
    underlying_repository = build_stage_l_repository_mapping(root, repository, runner)
    stage_l_summary = runner.run_once(root, underlying_repository)
    if not isinstance(stage_l_summary, Mapping):
        raise StageLResume1Error("Stage-L run_once returned a non-mapping")
    identity = validate_stage_l_summary(stage_l_summary)
    stage_l_bytes = stable_json_bytes(stage_l_summary)
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": stage_l_summary["scientific_status"],
        "root_cause": stage_l_summary["root_cause"],
        "required_next_path": stage_l_summary["required_next_path"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": dict(repository),
        "recovery_contract": {
            "schema": "phase314b_r258_stagel_resume1_recovery_contract_v1",
            "original_stage_l_runner_attempt_count": 1,
            "original_stage_l_scientific_execution_count": 0,
            "resume1_runner_attempt_count": 1,
            "resume1_scientific_execution_count": 1,
            "original_failure_locus": "deterministic_environment_preflight",
            "original_blocked_report_preserved": True,
            "original_blocked_report_sha256": EXPECTED_STAGEL_BLOCKED_SHA256,
            "stage_l_science_modified": False,
            "stage_l_consolidated_run_once_modified": False,
            "existing_test_gates_rerun": False,
            "prior_test_counts": list(PRIOR_TEST_COUNTS),
            "prior_test_counts_source": "operator-attested; not rerun by Resume1",
            "deterministic_environment": dict(EXPECTED_ENV),
            "underlying_stage_l_summary_sha256": sha256_bytes(stage_l_bytes),
            "underlying_stage_l_single_run_result_sha256": identity[
                "single_run_result_sha256"
            ],
            "underlying_callback_pair_count": identity["callback_pair_count"],
        },
        "stage_l_result": stage_l_summary,
        "boundaries": {
            "selection_holdout_evaluated": False,
            "selection_holdout_used_for_fit_or_selection": False,
            **{key: False for key in FALSE_BOUNDARIES},
        },
    }


def blocked_payload(
    error: BaseException,
    repository: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagel_resume1_execution_recovery_failed",
        "required_next_path": "INSPECT_STAGEL_RESUME1_WITHOUT_RERUNNING_STAGE_L",
        "error_type": "{}.{}".format(type(error).__module__, type(error).__qualname__),
        "error": str(error),
        "traceback": traceback.format_exc(),
        "repository": repository,
        "original_stage_l_runner_attempt_count": 1,
        "original_stage_l_scientific_execution_count": 0,
        "resume1_runner_attempt_count": 1,
        "resume1_scientific_execution_count": "unknown_after_failure",
        "existing_test_gates_rerun": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": False,
        "selection_holdout_used_for_fit_or_selection": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve(strict=False)
    success = root / RESUME1_SUCCESS_REPORT
    blocked = root / RESUME1_BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-L Resume1 output already exists", file=sys.stderr)
        return 2

    repository: Optional[Mapping[str, Any]] = {"resolved_root": str(root)}
    try:
        validate_environment()
        repository = validate_repository(root)
        result = run_resume1(root, repository)
        payload = stable_json_bytes(result)
        write_once(success, payload)
        print(
            json.dumps(
                {
                    "execution_verdict": "PASS",
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "selected_configuration": None,
                    "single_run_result_sha256": result["recovery_contract"][
                        "underlying_stage_l_single_run_result_sha256"
                    ],
                    "summary": str(success),
                    "summary_sha256": sha256_bytes(payload),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload_value = blocked_payload(error, repository)
        try:
            payload = stable_json_bytes(payload_value)
            write_once(blocked, payload)
            print(
                json.dumps(
                    {
                        "execution_verdict": "BLOCKED",
                        "error_type": payload_value["error_type"],
                        "error": payload_value["error"],
                        "blocked_report": str(blocked),
                        "blocked_report_sha256": sha256_bytes(payload),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        except BaseException as report_error:
            print(
                "BLOCKED REPORT WRITE FAILED: {}: {}".format(
                    type(report_error).__name__, report_error
                ),
                file=sys.stderr,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
