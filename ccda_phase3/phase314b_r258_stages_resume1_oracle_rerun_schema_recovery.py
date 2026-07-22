"""Phase3.14b-r2.5.8 Stage S Resume1 oracle-rerun schema recovery.

The original Stage-S run stopped before either confirmation worker completed.
Stage-S projected the frozen Stage-R Resume1 payload using the invented key
``oracle_callback_rerun_count``.  The frozen Stage-R schema actually exposes
``oracle_callback_pairs_rerun``.  Calling ``int()`` on the missing key produced
``TypeError: int(None)`` before confirmation science.

Resume1 changes only this historical input boundary.  The frozen Stage-R input
key is required exactly, validated as a non-boolean integer, and normalized to
the unchanged Stage-S projection key ``oracle_callback_rerun_count``.  The
original Stage-S worker count, sequential execution, current-fit identity,
functional projection, aligned-gate evidence, classification tree, leakage
boundaries, and report structure remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage S Resume1"
SCHEMA = "phase314b_r258_stages_resume1_oracle_rerun_schema_recovery_v1"
BLOCKED_SCHEMA = "phase314b_r258_stages_resume1_oracle_rerun_schema_recovery_blocked_v1"

BASE_STAGES_IMPLEMENTATION_COMMIT = "b00c0d8822bec46241dc0d39c5bd29316233f444"
BASE_STAGES_BLOCKED_EVIDENCE_COMMIT = "fda74c4b3496bff630c74c60c14685f7d1fdc60d"
EXPECTED_STAGES_IMPLEMENTATION_PARENT = "35285e99304a926723379a3cf2a22acd17f9c0ac"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGES_SOURCE = "ccda_phase3/phase314b_r258_stages_oof_support_confirmation.py"
STAGES_WORKER = "scripts/phase3_14b_r258_stages_worker.py"
STAGES_EXECUTE = "scripts/phase3_14b_r258_stages_execute.py"
STAGES_TEST = "tests/test_phase3_14b_r258_stages_oof_support_confirmation.py"
ORIGINAL_SOURCE_SHA256: Mapping[str, str] = {
    STAGES_SOURCE: "0af6cdf7d3051c2a3f0364a3ece648c2653b8397f02ba038faeb537bcb795e0d",
    STAGES_WORKER: "5b435f12d1d7686d2d3a122d5fd97e52e3672fa73ffaf981b098d2a3d157309b",
    STAGES_EXECUTE: "2c0dbff1e427f17b39b8ca5f3d6654eb1ae2cfcdbd9572e9ef2e09fb7cfc1e4e",
    STAGES_TEST: "48a1f300f3f2c01ccd8d825215e5a4e3d90775b36c9614d6d21177fad974034d",
}
STAGES_BLOCKED_REPORT = "reports/phase3_14b_r258_stages_oof_support_confirmation_blocked_summary.json"
EXPECTED_STAGES_BLOCKED_REPORT_SHA256 = "5ec3341ece368fd9565e988bc48d26506bfbd6e43d6f9a9c29334005805befe5"

SUCCESS_REPORT = "reports/phase3_14b_r258_stages_resume1_oof_support_confirmation_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stages_resume1_oof_support_confirmation_blocked_summary.json"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S Resume1: restore oracle-rerun field schema"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume1 OOF confirmation evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume1 blocked evidence"
)
ORIGINAL_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S: confirm tolerance-aligned OOF support"
)
ORIGINAL_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage S blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stages_resume1_oracle_rerun_schema_recovery.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume1_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stages_resume1_oracle_rerun_schema_recovery.py"),
)
ORIGINAL_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGES_SOURCE),
    ("A", STAGES_WORKER),
    ("A", STAGES_EXECUTE),
    ("A", STAGES_TEST),
)
ORIGINAL_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGES_BLOCKED_REPORT),
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


class StageSResume1Error(RuntimeError):
    """Fail-closed Stage-S Resume1 error."""


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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageSResume1Error(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageSResume1Error(f"{label} is not a sequence")
    return value


def _required_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    """Read one frozen-schema integer without aliases or implicit coercion."""
    if key not in mapping:
        raise StageSResume1Error(f"{label} is missing: {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StageSResume1Error(
            f"{label} is not a non-boolean integer: {key}={value!r}"
        )
    return int(value)


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
        raise StageSResume1Error(
            "git {} failed: {}".format(" ".join(args), completed.stderr.strip())
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
        raise StageSResume1Error(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace")
            )
        )
    return completed.stdout


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageSResume1Error("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def status_paths(root: Path) -> Tuple[str, ...]:
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    values: List[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        values.append(value)
    return tuple(sorted(values))


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageSResume1Error(
            "deterministic environment mismatch: {}".format(mismatch)
        )
    return dict(EXPECTED_ENV)


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageSResume1Error(
                f"{label} boundary changed: {key}={mapping.get(key)!r}"
            )


def _validate_original_provenance(repo: Path) -> Mapping[str, Any]:
    if _git(repo, "rev-parse", f"{BASE_STAGES_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGES_IMPLEMENTATION_COMMIT
    ):
        raise StageSResume1Error("Stage-S blocked provenance parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGES_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_STAGES_IMPLEMENTATION_PARENT
    ):
        raise StageSResume1Error("Stage-S implementation parent changed")
    if _git(
        repo, "show", "-s", "--format=%s", BASE_STAGES_IMPLEMENTATION_COMMIT
    ) != ORIGINAL_IMPLEMENTATION_SUBJECT:
        raise StageSResume1Error("Stage-S implementation subject changed")
    if commit_name_status(repo, BASE_STAGES_IMPLEMENTATION_COMMIT) != tuple(
        sorted(ORIGINAL_IMPLEMENTATION_PATHS)
    ):
        raise StageSResume1Error("Stage-S implementation paths changed")
    if _git(
        repo, "show", "-s", "--format=%s", BASE_STAGES_BLOCKED_EVIDENCE_COMMIT
    ) != ORIGINAL_BLOCKED_SUBJECT:
        raise StageSResume1Error("Stage-S blocked-evidence subject changed")
    if commit_name_status(repo, BASE_STAGES_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(ORIGINAL_BLOCKED_PATHS)
    ):
        raise StageSResume1Error("Stage-S blocked-evidence paths changed")

    for relative, expected_sha in ORIGINAL_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file():
            raise StageSResume1Error(f"original Stage-S file is missing: {relative}")
        if sha256_file(path) != expected_sha:
            raise StageSResume1Error(f"original Stage-S file SHA changed: {relative}")
        if _git_bytes(
            repo, "show", f"{BASE_STAGES_IMPLEMENTATION_COMMIT}:{relative}"
        ) != path.read_bytes():
            raise StageSResume1Error(
                f"original Stage-S file differs from committed blob: {relative}"
            )

    blocked = repo / STAGES_BLOCKED_REPORT
    if not blocked.is_file():
        raise StageSResume1Error("original Stage-S blocked report is missing")
    if sha256_file(blocked) != EXPECTED_STAGES_BLOCKED_REPORT_SHA256:
        raise StageSResume1Error("original Stage-S blocked report SHA changed")
    if _git_bytes(
        repo,
        "show",
        f"{BASE_STAGES_BLOCKED_EVIDENCE_COMMIT}:{STAGES_BLOCKED_REPORT}",
    ) != blocked.read_bytes():
        raise StageSResume1Error(
            "original Stage-S blocked report differs from committed blob"
        )
    return {
        "stage_s_implementation_commit": BASE_STAGES_IMPLEMENTATION_COMMIT,
        "stage_s_blocked_evidence_commit": BASE_STAGES_BLOCKED_EVIDENCE_COMMIT,
        "stage_s_blocked_report_sha256": EXPECTED_STAGES_BLOCKED_REPORT_SHA256,
        "stage_s_source_sha256": dict(ORIGINAL_SOURCE_SHA256),
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageSResume1Error("Stage-S Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_STAGES_BLOCKED_EVIDENCE_COMMIT:
        raise StageSResume1Error("Stage-S Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageSResume1Error("Stage-S Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageSResume1Error("Stage-S Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageSResume1Error("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageSResume1Error("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageSResume1Error("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageSResume1Error("Stage-S Resume1 worktree must be clean")
    original = _validate_original_provenance(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageSResume1Error(f"Stage-S Resume1 output already exists: {relative}")

    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages

    base_report = stages.validate_base_report(repo)
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_STAGES_BLOCKED_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "original_stage_s": original,
        "base_report_sha256": stages.EXPECTED_BASE_REPORT_SHA256,
        "base_report": base_report,
    }


def corrected_functional_projection(
    stage_r_wrapper: Mapping[str, Any],
    *,
    stages: Any,
) -> Mapping[str, Any]:
    """Original Stage-S projection with one strict historical-key correction."""
    if stage_r_wrapper.get("execution_verdict") != "PASS":
        raise StageSResume1Error("worker Stage-R Resume1 execution did not PASS")
    if stage_r_wrapper.get("scientific_status") != "BLOCKED":
        raise StageSResume1Error("worker Stage-R Resume1 scientific status changed")
    if stage_r_wrapper.get("selected_configuration") is not None:
        raise StageSResume1Error("worker selected a configuration")
    if stage_r_wrapper.get("train_only_recommendation") is not None:
        raise StageSResume1Error("worker emitted a recommendation")
    require_false(stage_r_wrapper, stages.FALSE_BOUNDARIES, "worker Stage-R wrapper")
    stage_r = _mapping(stage_r_wrapper.get("stage_r_result"), "Stage-R payload")
    audit = _mapping(
        stage_r.get("tolerance_aligned_oof_post_upper_audit"), "Stage-R audit"
    )
    classification = copy.deepcopy(
        dict(_mapping(audit.get("classification"), "Stage-R classification"))
    )
    cells = sorted(
        (
            _mapping(value, f"cell {index}")
            for index, value in enumerate(
                _sequence(audit.get("cell_records"), "Stage-R cell records")
            )
        ),
        key=lambda cell: (
            str(cell.get("base_direction_id")),
            int(cell.get("timestep")),
        ),
    )
    records: List[Mapping[str, Any]] = []
    for cell in cells:
        identity = _mapping(cell.get("legacy_identity"), "legacy identity")
        if identity.get("all_functional_exact") is not True:
            raise StageSResume1Error("legacy functional replay is not exact")
        records.append(
            {
                "base_direction_id": str(cell.get("base_direction_id")),
                "timestep": int(cell.get("timestep")),
                "feature_mode": str(cell.get("feature_mode")),
                "feature_sha256": str(cell.get("feature_sha256")),
                "aligned_acceptance_rate": stages._float(
                    cell.get("aligned_acceptance_rate"), "aligned acceptance"
                ),
                "aligned_acceptance_locus": str(
                    cell.get("aligned_acceptance_locus")
                ),
                "aligned_selected_scale_sha256": str(
                    cell.get("aligned_selected_scale_sha256")
                ),
                "aligned_selected_scale_histogram": copy.deepcopy(
                    cell.get("aligned_selected_scale_histogram")
                ),
                "aligned_candidate_sha256": str(
                    cell.get("aligned_candidate_sha256")
                ),
                "internal_scale_attempt_order": copy.deepcopy(
                    cell.get("internal_scale_attempt_order")
                ),
                "aligned_assembly": copy.deepcopy(cell.get("aligned_assembly")),
                "raw_oracle_control": copy.deepcopy(cell.get("raw_oracle_control")),
                "projected_oracle_control": copy.deepcopy(
                    cell.get("projected_oracle_control")
                ),
                "comparator_source": cell.get("comparator_source"),
                "comparator_discriminator_predicate": cell.get(
                    "comparator_discriminator_predicate"
                ),
                "dual_oracle_discriminator_predicate": cell.get(
                    "dual_oracle_discriminator_predicate"
                ),
                "strict_pass_aligned_fail_row_count": int(
                    cell.get("strict_pass_aligned_fail_row_count")
                ),
                "aligned_upper_element_failure_count": int(
                    cell.get("aligned_upper_element_failure_count")
                ),
                "length_log_z_element_mismatch_count": int(
                    cell.get("length_log_z_element_mismatch_count")
                ),
                "legacy_functional_identity_exact": True,
            }
        )
    if len(records) != stages.EXPECTED_CELL_COUNT:
        raise StageSResume1Error("functional projection cell count changed")

    # Frozen Stage-R input key -> unchanged Stage-S normalized output key.
    oracle_reruns = _required_int(
        audit,
        "oracle_callback_pairs_rerun",
        "oracle callback pair rerun count",
    )
    projection = {
        "root_cause": stage_r.get("root_cause"),
        "required_next_path": stage_r.get("required_next_path"),
        "primary_failure_locus": stage_r.get("primary_failure_locus"),
        "cell_count": len(records),
        "scientific_fit_count": int(audit.get("scientific_oof_fit_count")),
        "repeat_fit_count": int(audit.get("portable_repeat_fit_count")),
        "total_fit_count": int(audit.get("total_oof_fit_count")),
        "callback_pair_count": int(audit.get("legacy_callback_off_on_pair_count")),
        "oracle_callback_rerun_count": oracle_reruns,
        "historical_prediction_sha_exact_cell_count": int(
            audit.get("historical_prediction_sha_exact_cell_count")
        ),
        "historical_prediction_sha_mismatch_cell_count": int(
            audit.get("historical_prediction_sha_mismatch_cell_count")
        ),
        "classification": classification,
        "cell_records": records,
    }
    return projection


def corrected_worker_payload(
    *,
    worker_id: str,
    root: Path,
    repository: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_environment_variables()
    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages
    from ccda_phase3 import (
        phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager,
    )
    from ccda_phase3 import (
        phase314b_r258_stager_resume1_portable_oof_functional_replay as resume1,
    )

    base_report = stages.validate_base_report(Path(root).resolve())
    base_functional = corrected_functional_projection(base_report, stages=stages)
    current_environment = stager.probe_environment(Path(root).resolve())
    replay = resume1.execute_recovery(
        root=Path(root).resolve(),
        environment=current_environment,
        repository=_mapping(base_report.get("repository"), "base repository"),
    )
    functional = corrected_functional_projection(replay, stages=stages)
    fit_identity = stages.current_fit_projection(replay)
    base_functional_sha = stages.sha256_bytes(stages.stable_json_bytes(base_functional))
    functional_sha = stages.sha256_bytes(stages.stable_json_bytes(functional))
    if functional_sha != base_functional_sha:
        raise StageSResume1Error(
            "worker functional projection differs from Stage-R Resume1"
        )
    summary = stages.validate_confirmation_projection(functional)
    return {
        "phase": stages.PHASE,
        "schema": stages.WORKER_SCHEMA,
        "worker_id": str(worker_id),
        "execution_verdict": "PASS",
        "repository_head": repository.get("head"),
        "environment": current_environment,
        "environment_sha256": stages.sha256_bytes(
            stages.stable_json_bytes(current_environment)
        ),
        "base_functional_projection_sha256": base_functional_sha,
        "functional_projection_sha256": functional_sha,
        "current_fit_projection_sha256": stages.sha256_bytes(
            stages.stable_json_bytes(fit_identity)
        ),
        "functional_projection_matches_base": True,
        "functional_projection": functional,
        "current_fit_projection": fit_identity,
        "confirmation_summary": summary,
        "scientific_oof_fit_count": stages.EXPECTED_SCIENCE_FITS_PER_WORKER,
        "repeat_identity_fit_count": stages.EXPECTED_REPEAT_FITS_PER_WORKER,
        "total_oof_fit_count": stages.EXPECTED_FITS_PER_WORKER,
        "callback_pair_count": stages.EXPECTED_CALLBACK_PAIRS_PER_WORKER,
        "oracle_callback_rerun_count": 0,
        "internal_scale_attempt_count": stages.EXPECTED_INTERNAL_ATTEMPTS_PER_WORKER,
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }


def run_recovered_confirmation(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
) -> Mapping[str, Any]:
    """Run the unchanged Stage-S confirmation with the corrected input boundary."""
    validate_environment_variables()
    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages

    repo = Path(root).resolve()
    base_report = _mapping(repository.get("base_report"), "base report")
    base_projection = corrected_functional_projection(base_report, stages=stages)
    base_projection_sha = stages.sha256_bytes(stages.stable_json_bytes(base_projection))
    with tempfile.TemporaryDirectory(
        prefix="phase314b_r258_stages_resume1_"
    ) as temporary:
        directory = Path(temporary)
        outputs: List[Path] = []
        for index in range(stages.EXPECTED_WORKER_COUNT):
            output = directory / f"worker_{index}.json"
            outputs.append(output)
            command = [
                str(python_bin),
                str(repo / "scripts/phase3_14b_r258_stages_resume1_worker.py"),
                "--root",
                str(repo),
                "--worker-id",
                f"worker-{index + 1}",
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=str(repo),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ),
                check=False,
            )
            if completed.returncode != 0:
                raise StageSResume1Error(
                    f"worker {index} rc={completed.returncode} "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
        workers = [stages.load_json(path) for path in outputs]

    comparison = stages.compare_workers(workers[0], workers[1])
    if workers[0].get("functional_projection_sha256") != base_projection_sha:
        raise StageSResume1Error(
            "confirmed projection differs from frozen Stage-R Resume1"
        )
    summary = _mapping(workers[0].get("confirmation_summary"), "confirmation summary")
    classification = stages.classify_confirmation(summary)
    worker_summaries: List[Mapping[str, Any]] = []
    for worker in workers:
        worker_summaries.append(
            {
                "worker_id": worker.get("worker_id"),
                "environment_sha256": worker.get("environment_sha256"),
                "functional_projection_sha256": worker.get(
                    "functional_projection_sha256"
                ),
                "current_fit_projection_sha256": worker.get(
                    "current_fit_projection_sha256"
                ),
                "functional_projection_matches_base": True,
                "scientific_oof_fit_count": worker.get("scientific_oof_fit_count"),
                "repeat_identity_fit_count": worker.get("repeat_identity_fit_count"),
                "total_oof_fit_count": worker.get("total_oof_fit_count"),
                "callback_pair_count": worker.get("callback_pair_count"),
                "oracle_callback_rerun_count": worker.get(
                    "oracle_callback_rerun_count"
                ),
                "internal_scale_attempt_count": worker.get(
                    "internal_scale_attempt_count"
                ),
            }
        )
    result: Dict[str, Any] = {
        "phase": stages.PHASE,
        "schema": stages.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": {
            key: value for key, value in repository.items() if key != "base_report"
        },
        "immutable_inputs": {
            "base_evidence_commit": stages.BASE_EVIDENCE_COMMIT,
            "base_report_sha256": stages.EXPECTED_BASE_REPORT_SHA256,
            "base_stage_r_payload_sha256": stages.EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256,
            "base_scientific_result_sha256": stages.EXPECTED_BASE_SCIENTIFIC_SHA256,
            "stage_r_resume1_source_sha256": stages.EXPECTED_STAGER_RESUME1_SOURCE_SHA256,
            "objective_train_only": True,
            "oof_cell_count": stages.EXPECTED_CELL_COUNT,
            "backbone_count": stages.EXPECTED_BACKBONE_COUNT,
            "timestep_count": stages.EXPECTED_TIMESTEP_COUNT,
            "worker_count": stages.EXPECTED_WORKER_COUNT,
        },
        "confirmation_execution": {
            "worker_count": stages.EXPECTED_WORKER_COUNT,
            "workers": worker_summaries,
            "worker_comparison": comparison,
            "workers_functional_projection_matches_base": True,
            "workers_current_fit_identity_byte_exact": True,
            "total_scientific_oof_fit_count": stages.EXPECTED_SCIENCE_FITS_PER_WORKER
            * stages.EXPECTED_WORKER_COUNT,
            "total_repeat_identity_fit_count": stages.EXPECTED_REPEAT_FITS_PER_WORKER
            * stages.EXPECTED_WORKER_COUNT,
            "total_oof_fit_count": stages.EXPECTED_TOTAL_FITS,
            "total_callback_pair_count": stages.EXPECTED_TOTAL_CALLBACK_PAIRS,
            "oracle_callback_rerun_count": 0,
            "total_internal_scale_attempt_count": stages.EXPECTED_TOTAL_INTERNAL_ATTEMPTS,
            "worker_outputs_persisted": False,
        },
        "confirmation_summary": copy.deepcopy(dict(summary)),
        "confirmed_functional_projection_sha256": base_projection_sha,
        "confirmed_functional_projection": workers[0]["functional_projection"],
        "mechanism_boundary": {
            "stagee_modified": False,
            "stageh_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stageq_modified": False,
            "stageq_resume1_modified": False,
            "stager_modified": False,
            "stager_resume1_modified": False,
            "stages_modified": False,
            "legacy_upper_gate_changed": False,
            "aligned_upper_gate_changed": False,
            "aligned_upper_gate_written_to_stagee": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "oof_backbone_population_changed": False,
            "historical_prediction_sha_required": False,
            "cross_worker_current_fit_identity_required": True,
            "cross_worker_functional_identity_required": True,
        },
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = stages.sha256_bytes(
        stages.stable_json_bytes(result)
    )
    return result


def execute_recovery(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
) -> Mapping[str, Any]:
    validate_environment_variables()
    recovered = run_recovered_confirmation(
        root=Path(root).resolve(),
        repository=repository,
        python_bin=python_bin,
    )
    recovered_sha = sha256_bytes(stable_json_bytes(recovered))
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": recovered["scientific_status"],
        "root_cause": recovered["root_cause"],
        "required_next_path": recovered["required_next_path"],
        "primary_failure_locus": recovered["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": {
            key: value for key, value in repository.items() if key != "base_report"
        },
        "immutable_inputs": {
            "base_stage_s_implementation_commit": BASE_STAGES_IMPLEMENTATION_COMMIT,
            "base_stage_s_blocked_evidence_commit": BASE_STAGES_BLOCKED_EVIDENCE_COMMIT,
            "base_stage_s_blocked_report_sha256": EXPECTED_STAGES_BLOCKED_REPORT_SHA256,
            "original_stage_s_source_sha256": dict(ORIGINAL_SOURCE_SHA256),
        },
        "recovery_contract": {
            "failure_message": (
                "int() argument must be a string, a bytes-like object or a number, "
                "not 'NoneType'"
            ),
            "historical_input_key": "oracle_callback_pairs_rerun",
            "normalized_projection_key": "oracle_callback_rerun_count",
            "historical_input_key_required_exactly": True,
            "incorrect_alias_accepted": False,
            "implicit_integer_coercion_used": False,
            "original_stage_s_files_modified": False,
            "original_stage_s_blocked_report_preserved": True,
            "worker_count": 2,
            "workers_run_sequentially": True,
            "worker_script_recovery_only": True,
        },
        "recovered_stage_s_result_sha256": recovered_sha,
        "recovered_stage_s_scientific_result_sha256": recovered[
            "scientific_result_sha256"
        ],
        "recovered_stage_s_result": recovered,
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
        "root_cause": "phase314b_r258_stages_resume1_schema_recovery_failed",
        "required_next_path": "RESTORE_STAGES_RESUME1_ORACLE_RERUN_SCHEMA_RECOVERY",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None
        if repository is None
        else {key: value for key, value in repository.items() if key != "base_report"},
        "error_type": type(error).__name__,
        "error_message": str(error),
        "base_stage_s_blocked_report_sha256": EXPECTED_STAGES_BLOCKED_REPORT_SHA256,
        "base_stage_s_blocked_evidence_preserved": True,
        **{key: False for key in FALSE_BOUNDARIES},
    }
