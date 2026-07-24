"""Phase3.14b-r2.5.9 Stage D Resume3 call-order recovery.

Resume2 repaired the cable tensor contract to [N, 4, 48].  Its cold worker then
reached the original Stage-D procedure stitching code and stopped before durable
worker evidence with::

    variable-threshold threshold population changed

The threshold population did not change.  The original Stage-D call supplies
nine positional arguments in the wrong order.  The frozen kernel signature is::

    control, candidate, scale, risk, thresholds, target, groups, constant, spec

while Stage-D supplies::

    control, candidate, scale, risk, target, groups, thresholds, constant, spec

Consequently the [N,4,48] target tensor is validated as the threshold vector.
This add-only recovery installs two temporary input-boundary adapters while the
unchanged original ``run_stage_d`` executes:

* Resume2's correct 48D / 24-bead descriptor adapter;
* a call-order adapter that reorders only the threshold/target/groups arguments.

The 360 recipes, folds, fits, candidate mechanism, thresholds, result tree and
scientific validators are unchanged.  Execution remains one portable probe,
one cold science worker, and one terminal report.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import (
    phase314b_r259_staged_resume2_cable_state_schema_recovery as resume2,
)

PHASE = "Phase3.14b-r2.5.9 Stage D Resume3"
SCHEMA = "phase314b_r259_staged_resume3_variable_threshold_call_order_recovery_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_HEAD = "78e819074335a00ba142944b4dabb347ba8982ea"
BASE_RESUME2_IMPLEMENTATION = "5baa02794b259db5964a6f245a17c94d851b5a48"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage D Resume3: repair threshold call order"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D Resume3 t10 risk repair evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D Resume3 blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r259_staged_resume3_variable_threshold_call_order_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r259_staged_resume3_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r259_staged_resume3_variable_threshold_call_order_recovery.py",
    ),
)

RESUME2_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_resume2_"
    "cable_state_schema_recovery_blocked_summary.json"
)
RESUME2_BLOCKED_REPORT_SHA256 = (
    "f13bde1a6e7adf3fb18d05901ab58ce36f811897c505f440214f6f5767017f58"
)
RESUME2_PROBE_EVIDENCE = (
    "reports/phase3_14b_r259_staged_resume2_environment_probe_evidence.json"
)
RESUME2_PROBE_SHA256 = (
    "a7c8d5493fc4cc020af51c335a22b8878224b38595260d6503df49eedfb002dc"
)
STAGE_D_SOURCE = (
    "ccda_phase3/phase314b_r259_staged_t10_risk_descriptor_calibration_repair.py"
)
STAGE_D_SOURCE_SHA256 = (
    "0be595499b12e1782c005ca873e59559e102725e9b5141c5397c76d917ce6261"
)
RESUME2_SOURCE = (
    "ccda_phase3/phase314b_r259_staged_resume2_cable_state_schema_recovery.py"
)
RESUME2_SOURCE_SHA256 = (
    "28f412aba309a17e3b2210fa7416a0696c584430fe37666e8eac79697674ae70"
)

PROBE_EVIDENCE = (
    "reports/phase3_14b_r259_staged_resume3_environment_probe_evidence.json"
)
WORKER_EVIDENCE = (
    "reports/phase3_14b_r259_staged_resume3_t10_risk_repair_worker_evidence.json"
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_staged_resume3_variable_threshold_call_order_recovery_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_resume3_variable_threshold_call_order_recovery_blocked_summary.json"
)


class StageDResume3Error(RuntimeError):
    """Fail-closed Resume3 error."""


def _base() -> Any:
    from ccda_phase3 import (
        phase314b_r259_staged_t10_risk_descriptor_calibration_repair as staged,
    )

    return staged


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
        raise StageDResume3Error("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageDResume3Error("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def promote_write_ahead(source: Path, destination: Path, validator: Any) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise StageDResume3Error("write-ahead source missing: {}".format(source_path))
    payload = source_path.read_bytes()
    validator(json.loads(payload.decode("utf-8")))
    atomic_write_once(destination_path, payload)
    if destination_path.read_bytes() != payload:
        raise StageDResume3Error("write-ahead promotion is not byte-exact")
    return sha256_bytes(payload)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageDResume3Error("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageDResume3Error("{} worktree is dirty".format(label))


def validate_resume2_blocked_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise StageDResume3Error("Resume2 blocked report is missing")
    if sha256_file(report_path) != RESUME2_BLOCKED_REPORT_SHA256:
        raise StageDResume3Error("Resume2 blocked report SHA changed")
    payload = load_json(report_path)
    expected = {
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "staged_resume2_execution_contract",
        "worker_evidence_present": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageDResume3Error("Resume2 blocked field changed: {}".format(key))
    if "variable-threshold threshold population changed" not in str(
        payload.get("error_message", "")
    ):
        raise StageDResume3Error("Resume2 failure message changed")
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    """Small preflight: exact parent, branch, sources, submodule and clean trees."""
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageDResume3Error(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != (
        IMPLEMENTATION_SUBJECT
    ):
        raise StageDResume3Error("Resume3 implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(
        sorted(IMPLEMENTATION_PATHS)
    ):
        raise StageDResume3Error("Resume3 implementation paths changed")

    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageDResume3Error("DeformableRavens worktree commit changed")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(submodule, "DeformableRavens")

    validate_resume2_blocked_report(repo / RESUME2_BLOCKED_REPORT)
    if sha256_file(repo / RESUME2_PROBE_EVIDENCE) != RESUME2_PROBE_SHA256:
        raise StageDResume3Error("Resume2 probe evidence SHA changed")
    if sha256_file(repo / STAGE_D_SOURCE) != STAGE_D_SOURCE_SHA256:
        raise StageDResume3Error("original Stage-D source changed")
    if sha256_file(repo / RESUME2_SOURCE) != RESUME2_SOURCE_SHA256:
        raise StageDResume3Error("Resume2 source changed")

    for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageDResume3Error("Resume3 output already exists: {}".format(relative))

    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "resume2_blocked_evidence_commit": BASE_HEAD,
        "resume2_implementation_commit": BASE_RESUME2_IMPLEMENTATION,
    }


def evaluate_stage_d_variable_threshold_call(
    control: np.ndarray,
    raw_candidate: np.ndarray,
    raw_scale: np.ndarray,
    adverse_probability: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    thresholds: np.ndarray,
    baseline_probability: np.ndarray,
    spec: Any,
) -> Mapping[str, Any]:
    """Accept Stage-D's historical positional order and call the frozen kernel correctly."""
    staged = _base()
    rows = int(np.asarray(control).shape[0])
    threshold_value = np.asarray(thresholds, dtype=np.float64)
    target_value = np.asarray(target)
    groups_value = np.asarray(groups)
    if threshold_value.shape != (rows,):
        raise StageDResume3Error("Stage-D threshold vector must be [N]")
    if target_value.shape != np.asarray(control).shape:
        raise StageDResume3Error("Stage-D target tensor shape changed")
    if groups_value.shape != (rows,):
        raise StageDResume3Error("Stage-D group population changed")
    return staged.kernel_original_variable_threshold_policy(
        control,
        raw_candidate,
        raw_scale,
        adverse_probability,
        threshold_value,
        target_value,
        groups_value,
        baseline_probability,
        spec,
    )


@contextmanager
def installed_resume3_adapters() -> Iterator[None]:
    staged = _base()
    kernel = staged.kernel
    original_descriptor = staged.build_risk_descriptors
    original_policy = kernel.evaluate_variable_threshold_policy
    if hasattr(staged, "kernel_original_variable_threshold_policy"):
        raise StageDResume3Error("Resume3 policy adapter is already installed")
    staged.kernel_original_variable_threshold_policy = original_policy
    staged.build_risk_descriptors = resume2.build_risk_descriptors_48
    kernel.evaluate_variable_threshold_policy = evaluate_stage_d_variable_threshold_call
    try:
        yield
    finally:
        kernel.evaluate_variable_threshold_policy = original_policy
        staged.build_risk_descriptors = original_descriptor
        delattr(staged, "kernel_original_variable_threshold_policy")


def make_probe_evidence(root: Path) -> Mapping[str, Any]:
    return resume2.make_probe_evidence(Path(root).resolve())


def validate_probe_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return resume2.validate_probe_evidence(payload)


def run_science_worker(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    staged = _base()
    validate_probe_evidence(probe_payload)
    with installed_resume3_adapters():
        payload = dict(
            staged.run_stage_d(
                root=Path(root).resolve(),
                probe_payload=probe_payload,
                repository_head=str(repository_head),
                stageu_contract=stageu_contract,
            )
        )
    payload.pop("worker_result_sha256", None)
    payload["resume3_call_order_recovery"] = {
        "cable_tensor_contract": ["N", 4, 48],
        "descriptor_adapter": "resume2_48d_24_bead",
        "historical_stage_d_argument_order": [
            "control",
            "candidate",
            "scale",
            "risk",
            "target",
            "groups",
            "thresholds",
            "constant",
            "spec",
        ],
        "frozen_kernel_argument_order": [
            "control",
            "candidate",
            "scale",
            "risk",
            "thresholds",
            "target",
            "groups",
            "constant",
            "spec",
        ],
        "threshold_population_changed": False,
        "recipe_bank_changed": False,
        "threshold_values_changed": False,
        "science_kernel_copied": False,
        "single_cold_science_worker": True,
    }
    payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_evidence(payload)
    return payload


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    staged = _base()
    staged.validate_worker_evidence(payload)
    recovery = payload.get("resume3_call_order_recovery")
    expected = {
        "cable_tensor_contract": ["N", 4, 48],
        "descriptor_adapter": "resume2_48d_24_bead",
        "historical_stage_d_argument_order": [
            "control",
            "candidate",
            "scale",
            "risk",
            "target",
            "groups",
            "thresholds",
            "constant",
            "spec",
        ],
        "frozen_kernel_argument_order": [
            "control",
            "candidate",
            "scale",
            "risk",
            "thresholds",
            "target",
            "groups",
            "constant",
            "spec",
        ],
        "threshold_population_changed": False,
        "recipe_bank_changed": False,
        "threshold_values_changed": False,
        "science_kernel_copied": False,
        "single_cold_science_worker": True,
    }
    if recovery != expected:
        raise StageDResume3Error("Resume3 call-order recovery contract changed")
    expected_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected_hash:
        raise StageDResume3Error("Resume3 worker self-hash changed")


def build_summary(
    *, repository: Mapping[str, Any], probe_path: Path, worker_path: Path
) -> Mapping[str, Any]:
    staged = _base()
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    validate_probe_evidence(probe)
    validate_worker_evidence(worker)
    if int(probe.get("process_id", -1)) == int(worker.get("process_id", -1)):
        raise StageDResume3Error("probe and worker process IDs are not distinct")

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": copy.deepcopy(dict(repository)),
        "call_order_recovery": copy.deepcopy(worker["resume3_call_order_recovery"]),
        "portable_environment_contract": copy.deepcopy(
            probe["portable_environment_audit"]
        ),
        "population": copy.deepcopy(worker["population"]),
        "legacy_stagec_control_contract": copy.deepcopy(
            worker["legacy_stagec_control_contract"]
        ),
        "recipe_population": copy.deepcopy(worker["recipe_population"]),
        "outer_fold_selections": copy.deepcopy(worker["outer_fold_selections"]),
        "outer_fold_selected_recipe_metrics": copy.deepcopy(
            worker["outer_fold_selected_recipe_metrics"]
        ),
        "outer_selection_modal_recipe_diagnostic": copy.deepcopy(
            worker["outer_selection_modal_recipe_diagnostic"]
        ),
        "outer_crossfit_fold_selected_recipe_record": copy.deepcopy(
            worker["outer_crossfit_fold_selected_recipe_record"]
        ),
        "outer_crossfit_no_abstention_baseline_record": copy.deepcopy(
            worker["outer_crossfit_no_abstention_baseline_record"]
        ),
        "outer_crossfit_procedure_eligibility": copy.deepcopy(
            worker["outer_crossfit_procedure_eligibility"]
        ),
        "full_objective_oof_fixed_recipe_selection": copy.deepcopy(
            worker["full_objective_oof_fixed_recipe_selection"]
        ),
        "procedure_gate_totals": copy.deepcopy(worker["procedure_gate_totals"]),
        "execution_counts": copy.deepcopy(worker["execution_counts"]),
        "durable_evidence": {
            "probe_path": PROBE_EVIDENCE,
            "probe_sha256": sha256_file(probe_path),
            "worker_path": WORKER_EVIDENCE,
            "worker_sha256": sha256_file(worker_path),
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_persisted_before_summary": True,
            "promotion_byte_exact": True,
        },
        "process_topology": {
            "probe_process_id": int(probe["process_id"]),
            "worker_process_id": int(worker["process_id"]),
            "processes_distinct": True,
            "cold_science_worker_count": 1,
        },
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(
            worker["train_only_recommendation"]
        ),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in staged.FALSE_BOUNDARIES},
    }
    result["summary_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_summary(result)
    return result


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageDResume3Error("Resume3 summary schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageDResume3Error("Resume3 scientific status changed")
    recovery = payload.get("call_order_recovery")
    if not isinstance(recovery, Mapping):
        raise StageDResume3Error("Resume3 call-order recovery missing")
    if recovery.get("threshold_population_changed") is not False:
        raise StageDResume3Error("Resume3 changed threshold population")
    if payload.get("selected_configuration") is not None:
        raise StageDResume3Error("Resume3 selected a configuration")
    recommendation = payload.get("train_only_recommendation")
    if payload.get("scientific_status") == "READY":
        if not isinstance(recommendation, Mapping):
            raise StageDResume3Error("READY Resume3 lacks train-only recommendation")
    elif recommendation is not None:
        raise StageDResume3Error("BLOCKED Resume3 retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageDResume3Error("Resume3 re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageDResume3Error("Resume3 holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageDResume3Error("Resume3 accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageDResume3Error("Resume3 authorized rerun")
    if any(payload.get(key) is not False for key in _base().FALSE_BOUNDARIES):
        raise StageDResume3Error("Resume3 crossed a forbidden boundary")
    expected_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected_hash:
        raise StageDResume3Error("Resume3 summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Path,
    worker_path: Path,
) -> Mapping[str, Any]:
    probe_present = Path(probe_path).is_file()
    worker_present = Path(worker_path).is_file()
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_staged_resume3_call_order_recovery_failed",
        "required_next_path": (
            "FINALIZE_R259_STAGED_RESUME3_EVIDENCE_IF_WORKER_PERSISTED_"
            "OTHERWISE_AUDIT_EXECUTION_CONTRACT"
        ),
        "primary_failure_locus": "staged_resume3_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "probe_evidence_present": probe_present,
        "probe_evidence_sha256": sha256_file(probe_path) if probe_present else None,
        "worker_evidence_present": worker_present,
        "worker_evidence_sha256": sha256_file(worker_path) if worker_present else None,
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in _base().FALSE_BOUNDARIES},
    }
