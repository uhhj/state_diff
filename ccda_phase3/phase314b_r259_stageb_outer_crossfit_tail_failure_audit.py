"""Phase3.14b-r2.5.9 Stage B outer-crossfit tail-failure audit.

Stage A completed one newly preregistered objective-train-only nested-group OOF
calibration with durable probe, worker, and summary evidence.  Execution passed,
but the unbiased outer-crossfit procedure remained scientifically blocked.

This stage is evidence-only.  It reads the immutable Stage-A durable worker
payload and final summary, reconstructs every frozen policy gate from persisted
aggregate scalars, audits the six inner-selection records, and determines what
causal attribution is and is not supported by the committed evidence surface.
It never imports Torch or NumPy, starts CUDA, launches a probe or worker, reads a
dataset, fits a model, generates a candidate, or accesses the selection holdout
or frozen probe.

The audit is intentionally fail-closed.  It requires the exact Stage-A commit
chain, source hashes, evidence hashes, execution counts, policy population,
three-timestep gate pattern, fallback-fold population, modal policy, and clean
aligned-gate totals.  Its expected conclusion is aggregate localization only:
the persisted evidence proves a t=10 acceptance-coverage failure and inner-fold
selection instability, but it does not contain outer-test fold metrics, row or
group identities, or condition/seed attribution.  A later newly preregistered
objective-train-only stage is therefore required to produce durable fold-resolved
aggregate evidence; Stage A itself is never replayed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.9 Stage B"
SCHEMA = "phase314b_r259_stageb_outer_crossfit_tail_failure_audit_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEA_IMPLEMENTATION_COMMIT = "507461048754a5a03f287cd1ef7017aacd53afbb"
BASE_STAGEA_EVIDENCE_COMMIT = "3100a877f3664936f713b089e1489768975f3fa4"
BASE_STAGEA_PARENT = "34011592bb762b350d4c7489d598a22ee9175042"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_STAGEA_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage A: add durable tail-robust nested OOF"
)
BASE_STAGEA_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage A durable tail-robust OOF evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage B: audit outer-crossfit tail failure"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage B outer-crossfit audit evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage B blocked evidence"
)

BASE_STAGEA_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagea_durable_tail_robust_nested_oof.py"),
    ("A", "scripts/phase3_14b_r259_stagea_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagea_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagea_durable_tail_robust_nested_oof.py"),
)
BASE_STAGEA_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "reports/phase3_14b_r259_stagea_environment_probe_evidence.json"),
    (
        "A",
        "reports/phase3_14b_r259_stagea_tail_robust_nested_group_oof_worker_evidence.json",
    ),
    (
        "A",
        "reports/phase3_14b_r259_stagea_durable_tail_robust_nested_group_oof_summary.json",
    ),
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stageb_outer_crossfit_tail_failure_audit.py"),
    ("A", "scripts/phase3_14b_r259_stageb_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stageb_outer_crossfit_tail_failure_audit.py"),
)

BASE_STAGEA_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r259_stagea_durable_tail_robust_nested_oof.py": (
        "7e653977a5edc15fe84ccba6e2d7548456a31a3a1d3de8cb7c82e1ce37c7b18e"
    ),
    "scripts/phase3_14b_r259_stagea_worker.py": (
        "c191af24e8b99e3d7445af203807bb5b2fb65986905d565dfea3484332fcdd1e"
    ),
    "scripts/phase3_14b_r259_stagea_execute.py": (
        "167c78310facb79d87d6c3ab87308df7c549c62f38864ce567986a8f1cbb51ed"
    ),
    "tests/test_phase3_14b_r259_stagea_durable_tail_robust_nested_oof.py": (
        "e037deee38cd2d626a458f4ddb62c8ded594290f32176b5198c7457ff5e15809"
    ),
}

PROBE_EVIDENCE = "reports/phase3_14b_r259_stagea_environment_probe_evidence.json"
WORKER_EVIDENCE = (
    "reports/phase3_14b_r259_stagea_tail_robust_nested_group_oof_worker_evidence.json"
)
STAGEA_SUMMARY = (
    "reports/phase3_14b_r259_stagea_durable_tail_robust_nested_group_oof_summary.json"
)
EXPECTED_PROBE_FILE_SHA256 = (
    "840dc98ca2b37179920e1b95183a86eb4160ff1668f8b1d95df0d9fd1cbde7a1"
)
EXPECTED_WORKER_FILE_SHA256 = (
    "f9524bf1cc08a24ebc547e1e05fdad42274902a33dd03cb24fc8f2cf6c32bd92"
)
EXPECTED_SUMMARY_FILE_SHA256 = (
    "a433fe674b56bfbb2c09ff6ca836da941f4c9b32f69a9a05c06d22fdd3b9e8c0"
)
EXPECTED_WORKER_RESULT_SHA256 = (
    "3d3631c6765caee1c19f83d961afef8ccdb1676256371bd444612cf211231cca"
)
EXPECTED_SUMMARY_SELF_SHA256 = (
    "30295f799e1db737208c8afbc80a12243ee64c45c3f294ba0e3d83a5f5c63f37"
)

SUCCESS_REPORT = (
    "reports/phase3_14b_r259_stageb_outer_crossfit_tail_failure_audit_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stageb_outer_crossfit_tail_failure_audit_blocked_summary.json"
)

EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_OUTER_FOLDS = 6
EXPECTED_MODAL_POLICY = "shrink_1.00__risk_0.50"
EXPECTED_MODAL_SUPPORT = 4
EXPECTED_FALLBACK_FOLDS: Tuple[int, ...] = (1, 4)
EXPECTED_EXECUTION_COUNTS: Mapping[str, int] = {
    "direction_fit_count": 108,
    "candidate_generation_count": 324,
    "risk_fit_count": 324,
    "internal_scale_attempt_count": 2268,
    "inner_policy_evaluation_count": 216,
    "full_objective_oof_policy_evaluation_count": 36,
    "nonconverged_risk_fit_count": 0,
}
EXPECTED_STAGEA_ROOT = "phase314b_r259_stagea_durable_tail_robust_nested_oof_blocked"
EXPECTED_STAGEA_LOCUS = "outer_crossfit_tail_fidelity_failure"
EXPECTED_STAGEA_NEXT = (
    "AUDIT_R259_STAGEA_OUTER_CROSSFIT_TAIL_FAILURE_ON_OBJECTIVE_TRAIN_ONLY"
)

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "environment_probe_run",
    "science_worker_run",
    "cuda_initialized",
    "dataset_loaded",
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


class StageBError(RuntimeError):
    """Fail-closed Phase3.14b-r2.5.9 Stage-B error."""


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
        raise StageBError("JSON root is not a mapping: {}".format(path))
    return value


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageBError("write-once output exists: {}".format(target))
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
        ["git", *args], cwd=str(root), text=True
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    result: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise StageBError("unexpected name-status line: {!r}".format(line))
        result.append((parts[0], parts[1]))
    return tuple(sorted(result))


def assert_clean_worktree(root: Path, label: str) -> None:
    output = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if output:
        raise StageBError("{} worktree is dirty".format(label))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBError("{} is not a mapping".format(label))
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageBError("{} is not a sequence".format(label))
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StageBError("{} is boolean".format(label))
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise StageBError("{} is not numeric".format(label)) from error
    if not math.isfinite(result):
        raise StageBError("{} is nonfinite".format(label))
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageBError("{} is not an integer".format(label))
    return value


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageBError("branch changed")
    if _git(repo, "rev-parse", "HEAD") != BASE_STAGEA_EVIDENCE_COMMIT:
        raise StageBError("Stage-B starting HEAD changed")
    if _git(repo, "rev-parse", "HEAD^") != BASE_STAGEA_IMPLEMENTATION_COMMIT:
        raise StageBError("Stage-A evidence parent changed")
    if _git(repo, "rev-parse", BASE_STAGEA_IMPLEMENTATION_COMMIT + "^") != BASE_STAGEA_PARENT:
        raise StageBError("Stage-A implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEA_IMPLEMENTATION_COMMIT) != (
        BASE_STAGEA_IMPLEMENTATION_SUBJECT
    ):
        raise StageBError("Stage-A implementation subject changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEA_EVIDENCE_COMMIT) != (
        BASE_STAGEA_EVIDENCE_SUBJECT
    ):
        raise StageBError("Stage-A evidence subject changed")
    if commit_name_status(repo, BASE_STAGEA_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEA_IMPLEMENTATION_PATHS)
    ):
        raise StageBError("Stage-A implementation path population changed")
    if commit_name_status(repo, BASE_STAGEA_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_STAGEA_EVIDENCE_PATHS)
    ):
        raise StageBError("Stage-A evidence path population changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageBError("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageBError("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageBError("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-B")

    for relative, expected_sha in BASE_STAGEA_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageBError("Stage-A source changed: {}".format(relative))
        committed = _git_bytes(
            repo, "show", "{}:{}".format(BASE_STAGEA_IMPLEMENTATION_COMMIT, relative)
        )
        if committed != path.read_bytes():
            raise StageBError("Stage-A source differs from committed blob: {}".format(relative))

    evidence_hashes = {
        PROBE_EVIDENCE: EXPECTED_PROBE_FILE_SHA256,
        WORKER_EVIDENCE: EXPECTED_WORKER_FILE_SHA256,
        STAGEA_SUMMARY: EXPECTED_SUMMARY_FILE_SHA256,
    }
    for relative, expected_sha in evidence_hashes.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageBError("Stage-A evidence changed: {}".format(relative))
        committed = _git_bytes(
            repo, "show", "{}:{}".format(BASE_STAGEA_EVIDENCE_COMMIT, relative)
        )
        if committed != path.read_bytes():
            raise StageBError("Stage-A evidence differs from committed blob: {}".format(relative))

    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageBError("Stage-B output already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": BASE_STAGEA_EVIDENCE_COMMIT,
        "parent": BASE_STAGEA_IMPLEMENTATION_COMMIT,
        "stagea_parent": BASE_STAGEA_PARENT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "stagea_implementation_commit": BASE_STAGEA_IMPLEMENTATION_COMMIT,
        "stagea_evidence_commit": BASE_STAGEA_EVIDENCE_COMMIT,
    }


def validate_stagea_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("execution_verdict") != "PASS":
        raise StageBError("Stage-A execution verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageBError("Stage-A scientific status changed")
    if payload.get("root_cause") != EXPECTED_STAGEA_ROOT:
        raise StageBError("Stage-A root cause changed")
    if payload.get("primary_failure_locus") != EXPECTED_STAGEA_LOCUS:
        raise StageBError("Stage-A failure locus changed")
    if payload.get("required_next_path") != EXPECTED_STAGEA_NEXT:
        raise StageBError("Stage-A next path changed")
    if payload.get("selected_configuration") is not None:
        raise StageBError("Stage-A selected configuration changed")
    if payload.get("train_only_recommendation") is not None:
        raise StageBError("Stage-A retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageBError("Stage-A re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageBError("Stage-A cumulative holdout count changed")
    if payload.get("rerun_authorized") is not False:
        raise StageBError("Stage-A rerun authorization changed")
    if payload.get("summary_sha256") != EXPECTED_SUMMARY_SELF_SHA256:
        raise StageBError("Stage-A summary self-hash changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if expected != EXPECTED_SUMMARY_SELF_SHA256:
        raise StageBError("Stage-A summary self-hash is invalid")
    protocol = _mapping(payload.get("durable_evidence_protocol"), "durable protocol")
    if protocol.get("worker_result_persisted_before_controller_decoration") is not True:
        raise StageBError("Stage-A worker was not durably persisted")
    if protocol.get("repository_evidence_promoted_byte_exact") is not True:
        raise StageBError("Stage-A evidence was not byte-exact promoted")
    if protocol.get("worker_result_sha256") != EXPECTED_WORKER_RESULT_SHA256:
        raise StageBError("Stage-A summary worker-result SHA changed")


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("execution_verdict") != "PASS":
        raise StageBError("Stage-A worker verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageBError("Stage-A worker scientific status changed")
    if payload.get("root_cause") != EXPECTED_STAGEA_ROOT:
        raise StageBError("Stage-A worker root changed")
    if payload.get("primary_failure_locus") != EXPECTED_STAGEA_LOCUS:
        raise StageBError("Stage-A worker locus changed")
    if payload.get("required_next_path") != EXPECTED_STAGEA_NEXT:
        raise StageBError("Stage-A worker next path changed")
    if payload.get("worker_result_sha256") != EXPECTED_WORKER_RESULT_SHA256:
        raise StageBError("Stage-A worker-result SHA changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if expected != EXPECTED_WORKER_RESULT_SHA256:
        raise StageBError("Stage-A worker self-hash is invalid")
    if payload.get("execution_counts") != EXPECTED_EXECUTION_COUNTS:
        raise StageBError("Stage-A execution counts changed")
    if payload.get("selected_configuration") is not None:
        raise StageBError("Stage-A worker selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageBError("Stage-A worker retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageBError("Stage-A worker re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageBError("Stage-A worker cumulative holdout count changed")
    if payload.get("rerun_authorized") is not False:
        raise StageBError("Stage-A worker rerun authorization changed")


def _gate_checks(
    record: Mapping[str, Any], baseline: Mapping[str, Any], spec: Mapping[str, Any]
) -> Mapping[str, bool]:
    minimum_acceptance = _finite(spec.get("minimum_acceptance_rate"), "minimum acceptance")
    minimum_positive = _finite(
        spec.get("minimum_positive_reduction_rate"), "minimum positive reduction"
    )
    adverse_fraction = _finite(
        spec.get("adverse_sse_reduction_fraction"), "adverse reduction fraction"
    )
    group_tail = _mapping(record.get("group_tail"), "record group tail")
    baseline_group_tail = _mapping(baseline.get("group_tail"), "baseline group tail")
    return {
        "acceptance": _finite(record.get("acceptance_rate"), "acceptance")
        >= minimum_acceptance,
        "overall_mse": _finite(record.get("overall_mse_ratio"), "overall MSE") < 1.0,
        "accepted_mse": _finite(
            record.get("accepted_row_mse_ratio"), "accepted-row MSE"
        )
        < 1.0,
        "positive_reduction": _finite(
            record.get("positive_distance_reduction_rate"), "positive reduction"
        )
        > minimum_positive,
        "relative_reduction": _finite(
            record.get("relative_distance_reduction_mean"), "relative reduction"
        )
        > 0.0,
        "adverse_sse_reduction": _finite(
            record.get("adverse_sse_mass"), "adverse SSE mass"
        )
        <= _finite(baseline.get("adverse_sse_mass"), "baseline adverse SSE mass")
        * (1.0 - adverse_fraction),
        "group_cvar_improvement": _finite(
            group_tail.get("worst_fraction_cvar_mse_ratio"), "group CVaR"
        )
        < _finite(
            baseline_group_tail.get("worst_fraction_cvar_mse_ratio"),
            "baseline group CVaR",
        ),
        "risk_brier_nonworse": record.get("risk_brier_nonworse") is True,
    }


def reconstruct_outer_procedure_gates(worker: Mapping[str, Any]) -> Mapping[str, Any]:
    records = _mapping(
        worker.get("outer_crossfit_fold_selected_policy_records"),
        "outer crossfit records",
    )
    baselines = _mapping(worker.get("outer_crossfit_baseline_records"), "baseline records")
    if set(records) != {str(value) for value in EXPECTED_TIMESTEPS}:
        raise StageBError("outer crossfit timestep population changed")
    if set(baselines) != {str(value) for value in EXPECTED_TIMESTEPS}:
        raise StageBError("baseline timestep population changed")
    spec = _mapping(worker.get("stagex_spec"), "Stage-X spec")
    if _finite(spec.get("minimum_acceptance_rate"), "minimum acceptance") != 0.5:
        raise StageBError("frozen acceptance threshold changed")

    by_timestep: Dict[str, Mapping[str, Any]] = {}
    failed_pattern: Dict[str, List[str]] = {}
    for timestep in EXPECTED_TIMESTEPS:
        key = str(timestep)
        record = _mapping(records[key], "outer record {}".format(key))
        baseline = _mapping(baselines[key], "baseline record {}".format(key))
        checks = _gate_checks(record, baseline, spec)
        failed = sorted(name for name, passed in checks.items() if not passed)
        failed_pattern[key] = failed
        acceptance = _finite(record.get("acceptance_rate"), "acceptance")
        overall = _finite(record.get("overall_mse_ratio"), "overall MSE")
        baseline_overall = _finite(
            baseline.get("overall_mse_ratio"), "baseline overall MSE"
        )
        accepted_ratio = _finite(
            record.get("accepted_row_mse_ratio"), "accepted-row MSE"
        )
        group_tail = _mapping(record.get("group_tail"), "group tail")
        baseline_tail = _mapping(baseline.get("group_tail"), "baseline group tail")
        by_timestep[key] = {
            "row_count": _integer(record.get("row_count"), "row count"),
            "selected_row_count": _integer(
                record.get("selected_row_count"), "selected row count"
            ),
            "frozen_checks": checks,
            "failed_checks": failed,
            "all_checks_pass": not failed,
            "acceptance_rate": acceptance,
            "acceptance_threshold": 0.5,
            "acceptance_deficit": max(0.0, 0.5 - acceptance),
            "overall_mse_ratio": overall,
            "baseline_overall_mse_ratio": baseline_overall,
            "robust_minus_baseline_overall_mse": overall - baseline_overall,
            "accepted_row_mse_ratio": accepted_ratio,
            "accepted_subset_fidelity_gain": 1.0 - accepted_ratio,
            "positive_distance_reduction_rate": _finite(
                record.get("positive_distance_reduction_rate"),
                "positive reduction",
            ),
            "relative_distance_reduction_mean": _finite(
                record.get("relative_distance_reduction_mean"),
                "relative reduction",
            ),
            "adverse_sse_mass": _finite(
                record.get("adverse_sse_mass"), "adverse SSE mass"
            ),
            "baseline_adverse_sse_mass": _finite(
                baseline.get("adverse_sse_mass"), "baseline adverse SSE mass"
            ),
            "worst_fraction_cvar_mse_ratio": _finite(
                group_tail.get("worst_fraction_cvar_mse_ratio"), "group CVaR"
            ),
            "baseline_worst_fraction_cvar_mse_ratio": _finite(
                baseline_tail.get("worst_fraction_cvar_mse_ratio"),
                "baseline group CVaR",
            ),
            "risk_brier_score": _finite(
                record.get("risk_brier_score"), "risk Brier score"
            ),
            "risk_constant_brier_score": _finite(
                record.get("risk_constant_brier_score"),
                "constant Brier score",
            ),
        }

    expected_pattern = {"10": ["acceptance"], "25": [], "50": []}
    if failed_pattern != expected_pattern:
        raise StageBError(
            "Stage-A frozen gate pattern changed: {!r}".format(failed_pattern)
        )
    return {
        "frozen_gate_pattern": failed_pattern,
        "single_failed_timestep": 10,
        "single_failed_gate": "acceptance",
        "all_other_frozen_checks_pass": True,
        "by_timestep": by_timestep,
    }


def audit_outer_selections(worker: Mapping[str, Any]) -> Mapping[str, Any]:
    selections = _sequence(worker.get("outer_fold_selections"), "outer selections")
    if len(selections) != EXPECTED_OUTER_FOLDS:
        raise StageBError("outer selection count changed")
    records: List[Mapping[str, Any]] = []
    fallback_folds: List[int] = []
    selected_ids: List[str] = []
    modal_inner_eligibility: Dict[str, bool] = {}
    for index, raw in enumerate(selections):
        selection = _mapping(raw, "outer selection {}".format(index))
        fold = _integer(selection.get("outer_fold"), "outer fold")
        if fold != index:
            raise StageBError("outer fold order changed")
        selected_id = selection.get("selected_policy_id")
        if not isinstance(selected_id, str):
            raise StageBError("selected policy ID is invalid")
        selected_ids.append(selected_id)
        fallback = selection.get("diagnostic_fallback_used") is True
        eligible = selection.get("selected_policy_inner_eligible") is True
        if fallback == eligible:
            raise StageBError("fallback/eligibility contract changed")
        if fallback:
            fallback_folds.append(fold)
        policy_eligibility = _mapping(
            selection.get("policy_eligibility"), "policy eligibility"
        )
        if selected_id not in policy_eligibility:
            raise StageBError("selected policy lacks eligibility record")
        selected_eligibility = _mapping(
            policy_eligibility[selected_id], "selected policy eligibility"
        )
        checks_by_timestep = _mapping(
            selected_eligibility.get("checks"), "selected checks"
        )
        failed_checks: Dict[str, List[str]] = {}
        for timestep in EXPECTED_TIMESTEPS:
            checks = _mapping(
                checks_by_timestep.get(str(timestep)),
                "selected checks t{}".format(timestep),
            )
            failed_checks[str(timestep)] = sorted(
                key for key, value in checks.items() if value is not True
            )
        modal_record = _mapping(
            policy_eligibility.get(EXPECTED_MODAL_POLICY),
            "modal policy eligibility",
        )
        modal_inner_eligibility[str(fold)] = (
            modal_record.get("all_timesteps_pass") is True
        )
        records.append(
            {
                "outer_fold": fold,
                "selected_policy_id": selected_id,
                "selected_policy_inner_eligible": eligible,
                "diagnostic_fallback_used": fallback,
                "eligible_policy_count": len(
                    _sequence(selection.get("eligible_policy_ids"), "eligible policy IDs")
                ),
                "selected_policy_failed_checks": failed_checks,
                "modal_policy_inner_eligible": modal_inner_eligibility[str(fold)],
                "inner_train_row_count": _integer(
                    selection.get("inner_train_row_count"), "inner train rows"
                ),
                "outer_test_row_count": _integer(
                    selection.get("outer_test_row_count"), "outer test rows"
                ),
            }
        )
    if tuple(fallback_folds) != EXPECTED_FALLBACK_FOLDS:
        raise StageBError("fallback-fold population changed")

    modal = _mapping(
        worker.get("inner_selection_modal_policy_diagnostic"), "modal diagnostic"
    )
    if modal.get("policy_id") != EXPECTED_MODAL_POLICY:
        raise StageBError("modal policy changed")
    if modal.get("support_count") != EXPECTED_MODAL_SUPPORT:
        raise StageBError("modal support changed")
    counts = _mapping(modal.get("counts"), "modal counts")
    if counts.get(EXPECTED_MODAL_POLICY) != EXPECTED_MODAL_SUPPORT:
        raise StageBError("modal count mapping changed")
    full_selection = _mapping(
        worker.get("full_objective_oof_fixed_policy_selection"),
        "full-objective selection",
    )
    if full_selection.get("selected_policy_id") != EXPECTED_MODAL_POLICY:
        raise StageBError("full-objective selected policy changed")
    if full_selection.get("selected_policy_inner_eligible") is not True:
        raise StageBError("full-objective policy is no longer eligible")
    return {
        "outer_fold_count": EXPECTED_OUTER_FOLDS,
        "records": records,
        "selected_policy_ids": selected_ids,
        "fallback_folds": fallback_folds,
        "fallback_fold_count": len(fallback_folds),
        "eligible_selection_fold_count": EXPECTED_OUTER_FOLDS - len(fallback_folds),
        "modal_policy_id": EXPECTED_MODAL_POLICY,
        "modal_policy_support_count": EXPECTED_MODAL_SUPPORT,
        "modal_policy_support_fraction": EXPECTED_MODAL_SUPPORT / EXPECTED_OUTER_FOLDS,
        "modal_policy_inner_eligibility_by_fold": modal_inner_eligibility,
        "full_objective_oof_selected_policy_id": full_selection[
            "selected_policy_id"
        ],
        "full_objective_oof_selected_policy_eligible": True,
        "full_objective_surface_role": (
            "tuning_only_not_unbiased_outer_test_attribution"
        ),
    }


def audit_attribution_surface(worker: Mapping[str, Any]) -> Mapping[str, Any]:
    top_level = set(worker)
    forbidden_fold_metric_keys = (
        "outer_fold_test_records",
        "outer_fold_test_metrics",
        "outer_test_metrics_by_fold",
        "fold_resolved_outer_metrics",
    )
    present = sorted(key for key in forbidden_fold_metric_keys if key in top_level)
    if present:
        raise StageBError("unexpected fold-resolved metric surface appeared: {}".format(present))
    records = _mapping(
        worker.get("outer_crossfit_fold_selected_policy_records"),
        "outer procedure records",
    )
    group_identity_persisted = False
    for timestep in EXPECTED_TIMESTEPS:
        group_tail = _mapping(
            _mapping(records[str(timestep)], "record").get("group_tail"),
            "group tail",
        )
        if any(
            key in group_tail
            for key in (
                "group_records",
                "group_ids",
                "worst_group_ids",
                "group_ratios",
            )
        ):
            group_identity_persisted = True
    if group_identity_persisted:
        raise StageBError("unexpected group identity surface appeared")
    return {
        "aggregate_timestep_metrics_available": True,
        "inner_selection_by_outer_fold_available": True,
        "full_objective_fixed_policy_surface_available": True,
        "outer_test_metrics_by_fold_available": False,
        "outer_test_candidate_counts_by_fold_available": False,
        "outer_test_risk_calibration_by_fold_available": False,
        "row_level_metrics_available": False,
        "group_identity_or_ratio_table_available": False,
        "condition_attribution_available": False,
        "visible_seed_attribution_available": False,
        "pair_group_attribution_available": False,
        "future_horizon_attribution_available": False,
        "segment_attribution_available": False,
        "fallback_folds_can_be_identified_from_inner_selection": True,
        "fallback_folds_can_be_claimed_as_outer_test_failure_causes": False,
        "reason": (
            "Stage-A persisted fold identities and inner-selection records, but "
            "stitched the outer-test outputs before metric aggregation and persisted "
            "only global per-timestep records and anonymous group-tail summaries."
        ),
    }


def classify_audit(
    gates: Mapping[str, Any], selections: Mapping[str, Any], surface: Mapping[str, Any]
) -> Mapping[str, str]:
    if gates.get("single_failed_timestep") != 10:
        raise StageBError("single failed timestep changed")
    if gates.get("single_failed_gate") != "acceptance":
        raise StageBError("single failed gate changed")
    if selections.get("fallback_folds") != [1, 4]:
        raise StageBError("fallback folds changed")
    if surface.get("outer_test_metrics_by_fold_available") is not False:
        raise StageBError("fold-resolved surface classification changed")
    return {
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r259_stageb_aggregate_evidence_localizes_t10_"
            "acceptance_coverage_failure_but_outer_test_fold_attribution_is_unavailable"
        ),
        "required_next_path": (
            "PREREGISTER_R259_STAGEC_OBJECTIVE_TRAIN_ONLY_FOLD_RESOLVED_"
            "TAIL_ATTRIBUTION_WITH_DURABLE_AGGREGATE_EVIDENCE"
        ),
        "primary_failure_locus": (
            "t10_acceptance_coverage_with_fold_resolved_attribution_unavailable"
        ),
    }


def run_audit(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    repository = validate_repository(repo)
    probe = load_json(repo / PROBE_EVIDENCE)
    worker = load_json(repo / WORKER_EVIDENCE)
    summary = load_json(repo / STAGEA_SUMMARY)
    validate_stagea_summary(summary)
    validate_worker_evidence(worker)
    if summary["durable_evidence_protocol"]["probe_evidence_file_sha256"] != (
        EXPECTED_PROBE_FILE_SHA256
    ):
        raise StageBError("summary probe-evidence SHA changed")
    if summary["durable_evidence_protocol"]["worker_evidence_file_sha256"] != (
        EXPECTED_WORKER_FILE_SHA256
    ):
        raise StageBError("summary worker-evidence SHA changed")
    if summary["durable_evidence_protocol"]["worker_result_sha256"] != (
        EXPECTED_WORKER_RESULT_SHA256
    ):
        raise StageBError("summary worker-result SHA changed")
    if probe.get("execution_verdict") != "PASS":
        raise StageBError("Stage-A probe evidence changed")

    gates = reconstruct_outer_procedure_gates(worker)
    selections = audit_outer_selections(worker)
    surface = audit_attribution_surface(worker)
    classification = classify_audit(gates, selections, surface)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        **classification,
        "repository": repository,
        "immutable_inputs": {
            "stagea_implementation_commit": BASE_STAGEA_IMPLEMENTATION_COMMIT,
            "stagea_evidence_commit": BASE_STAGEA_EVIDENCE_COMMIT,
            "probe_evidence_path": PROBE_EVIDENCE,
            "probe_evidence_file_sha256": EXPECTED_PROBE_FILE_SHA256,
            "worker_evidence_path": WORKER_EVIDENCE,
            "worker_evidence_file_sha256": EXPECTED_WORKER_FILE_SHA256,
            "worker_result_sha256": EXPECTED_WORKER_RESULT_SHA256,
            "stagea_summary_path": STAGEA_SUMMARY,
            "stagea_summary_file_sha256": EXPECTED_SUMMARY_FILE_SHA256,
            "stagea_summary_self_sha256": EXPECTED_SUMMARY_SELF_SHA256,
        },
        "audit_scope": {
            "evidence_only": True,
            "stagea_science_replayed": False,
            "stagea_policy_or_threshold_changed": False,
            "new_fit_count": 0,
            "new_candidate_generation_count": 0,
            "new_risk_fit_count": 0,
            "new_internal_scale_attempt_count": 0,
            "new_policy_evaluation_count": 0,
            "controller_recomputed_science": False,
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
    validate_audit(result)
    return result


def validate_audit(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageBError("Stage-B audit schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageBError("Stage-B audit status changed")
    if payload.get("selected_configuration") is not None:
        raise StageBError("Stage-B selected configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageBError("Stage-B retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageBError("Stage-B re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageBError("Stage-B cumulative holdout count changed")
    if payload.get("rerun_authorized") is not False:
        raise StageBError("Stage-B authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageBError("Stage-B crossed forbidden boundary")
    scope = _mapping(payload.get("audit_scope"), "audit scope")
    for key in (
        "new_fit_count",
        "new_candidate_generation_count",
        "new_risk_fit_count",
        "new_internal_scale_attempt_count",
        "new_policy_evaluation_count",
    ):
        if scope.get(key) != 0:
            raise StageBError("Stage-B audit performed science: {}".format(key))
    if scope.get("evidence_only") is not True:
        raise StageBError("Stage-B is not evidence-only")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "audit_sha256"}
        )
    )
    if payload.get("audit_sha256") != expected:
        raise StageBError("Stage-B audit self-hash changed")


def blocked_report(
    repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageb_evidence_audit_execution_failed",
        "required_next_path": (
            "RESTORE_R259_STAGEB_EVIDENCE_ONLY_OUTER_CROSSFIT_AUDIT_WITHOUT_"
            "SCIENCE_REEXECUTION"
        ),
        "primary_failure_locus": "evidence_audit_execution_contract",
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
