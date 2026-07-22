"""Phase3.14b-r2.5.8 Stage S OOF support confirmation.

Stage-R Resume1 established broad tolerance-aligned OOF support on all 27
objective-train cells, while historical prediction tensors were not portable
across runs. Stage S changes no mechanism. It runs two isolated confirmations
of the unchanged Stage-R Resume1 functional replay and separates:

* current-fit identity, which must be byte-exact across the two new workers;
* functional/scientific identity, which must match both workers and the frozen
  Stage-R Resume1 evidence exactly.

No holdout, frozen probe, formal training, reverse, IDM, candidate execution,
DeformableRavens, Phase4, or CPS is run. Worker outputs are temporary and only
aggregate evidence and hashes are persisted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage S"
SCHEMA = "phase314b_r258_stages_oof_support_confirmation_v1"
BLOCKED_SCHEMA = "phase314b_r258_stages_oof_support_confirmation_blocked_v1"
WORKER_SCHEMA = "phase314b_r258_stages_oof_support_confirmation_worker_v1"

BASE_IMPLEMENTATION_COMMIT = "5668d053ef267bbf8cc5c12d7a70656fe09fe697"
BASE_EVIDENCE_COMMIT = "35285e99304a926723379a3cf2a22acd17f9c0ac"
EXPECTED_BASE_IMPLEMENTATION_PARENT = "61985677dde41f48396331c8c446f97f1d626d30"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_REPORT = "reports/phase3_14b_r258_stager_resume1_portable_oof_functional_replay_summary.json"
EXPECTED_BASE_REPORT_SHA256 = "4900dacf43affd7f9aa097d71978d481b9cb944ea5b19711a26f196a87a87d54"
EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256 = "8386edb35cbd50ae9f3d6ff53be235b9f2c4e8c6abb82cd33c94c161bb82c662"
EXPECTED_BASE_SCIENTIFIC_SHA256 = "04cd8feaeed1d8db16b8c4c8d3ba51242e0691e2cc139bc5b8749440d3f47fd3"
STAGER_RESUME1_SOURCE = "ccda_phase3/phase314b_r258_stager_resume1_portable_oof_functional_replay.py"
EXPECTED_STAGER_RESUME1_SOURCE_SHA256 = "abc252e0ac8c401c2b1c48d6122f66023061bc07892f4438dce8738d0680d4b5"

SUCCESS_REPORT = "reports/phase3_14b_r258_stages_oof_support_confirmation_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stages_oof_support_confirmation_blocked_summary.json"

IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage S: confirm tolerance-aligned OOF support"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage S OOF support confirmation evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage S blocked evidence"
BASE_IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage R Resume1: restore portable OOF functional replay"
BASE_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage R Resume1 OOF functional evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stages_oof_support_confirmation.py"),
    ("A", "scripts/phase3_14b_r258_stages_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stages_oof_support_confirmation.py"),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGER_RESUME1_SOURCE),
    ("A", "scripts/phase3_14b_r258_stager_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stager_resume1_portable_oof_functional_replay.py"),
)
BASE_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (("A", BASE_REPORT),)

EXPECTED_WORKER_COUNT = 2
EXPECTED_CELL_COUNT = 27
EXPECTED_BACKBONE_COUNT = 9
EXPECTED_TIMESTEP_COUNT = 3
EXPECTED_FITS_PER_WORKER = 54
EXPECTED_SCIENCE_FITS_PER_WORKER = 27
EXPECTED_REPEAT_FITS_PER_WORKER = 27
EXPECTED_CALLBACK_PAIRS_PER_WORKER = 27
EXPECTED_INTERNAL_ATTEMPTS_PER_WORKER = 189
EXPECTED_TOTAL_FITS = 108
EXPECTED_TOTAL_CALLBACK_PAIRS = 54
EXPECTED_TOTAL_INTERNAL_ATTEMPTS = 378
EXPECTED_NONZERO_SUPPORT_CELLS = 27
EXPECTED_ZERO_ACCEPTANCE_CELLS = 0
EXPECTED_ORACLE_LIKE_CELLS = 7
EXPECTED_DOMINANT_DISCRIMINATOR = "direction_retention"
EXPECTED_DOMINANT_SUPPORT = 2
EXPECTED_DOMINANT_BACKBONE_COVERAGE = 1
EXPECTED_DOMINANT_TIMESTEP_COVERAGE = 2
STABLE_DISCRIMINATOR_MIN = 24

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


class StageSError(RuntimeError):
    """Fail-closed Stage-S error."""


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
        raise StageSError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageSError(f"{label} is not a sequence")
    return value


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
        raise StageSError(
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
        raise StageSError(
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
            raise StageSError("unexpected diff-tree record")
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
        raise StageSError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageSError(f"{label} boundary changed: {key}={mapping.get(key)!r}")


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageSError(f"JSON root is not a mapping: {path}")
    return value


def _validate_base_report_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageSError("Stage-R Resume1 base execution verdict changed")
    if report.get("scientific_status") != "BLOCKED":
        raise StageSError("Stage-R Resume1 base scientific status changed")
    if report.get("root_cause") != (
        "phase314b_r258_stager_tolerance_alignment_restores_broad_oof_support"
    ):
        raise StageSError("Stage-R Resume1 base root cause changed")
    if report.get("required_next_path") != (
        "CONFIRM_TOLERANCE_ALIGNED_OOF_SUPPORT_ON_OBJECTIVE_TRAIN_ONLY"
    ):
        raise StageSError("Stage-R Resume1 base next path changed")
    if report.get("selected_configuration") is not None:
        raise StageSError("Stage-R Resume1 selected a configuration")
    if report.get("train_only_recommendation") is not None:
        raise StageSError("Stage-R Resume1 emitted a recommendation")
    if report.get("stage_r_result_sha256") != EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256:
        raise StageSError("Stage-R Resume1 recovered payload SHA changed")
    stage_r = _mapping(report.get("stage_r_result"), "Stage-R Resume1 payload")
    if stage_r.get("scientific_result_sha256") != EXPECTED_BASE_SCIENTIFIC_SHA256:
        raise StageSError("Stage-R Resume1 scientific SHA changed")
    require_false(report, FALSE_BOUNDARIES, "Stage-R Resume1 report")
    return report


def validate_base_report(root: Path) -> Mapping[str, Any]:
    path = Path(root) / BASE_REPORT
    if not path.is_file():
        raise StageSError("Stage-R Resume1 base report is missing")
    if sha256_file(path) != EXPECTED_BASE_REPORT_SHA256:
        raise StageSError("Stage-R Resume1 base report SHA changed")
    committed = _git_bytes(root, "show", f"{BASE_EVIDENCE_COMMIT}:{BASE_REPORT}")
    if committed != path.read_bytes():
        raise StageSError("Stage-R Resume1 report differs from committed blob")
    return _validate_base_report_payload(load_json(path))


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageSError("Stage-S requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_EVIDENCE_COMMIT:
        raise StageSError("Stage-S implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageSError("Stage-S implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageSError("Stage-S implementation paths changed")
    if _git(repo, "rev-parse", f"{BASE_EVIDENCE_COMMIT}^") != BASE_IMPLEMENTATION_COMMIT:
        raise StageSError("Stage-R Resume1 evidence parent changed")
    if _git(repo, "rev-parse", f"{BASE_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_BASE_IMPLEMENTATION_PARENT
    ):
        raise StageSError("Stage-R Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION_COMMIT) != (
        BASE_IMPLEMENTATION_SUBJECT
    ):
        raise StageSError("Stage-R Resume1 implementation subject changed")
    if commit_name_status(repo, BASE_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_IMPLEMENTATION_PATHS)
    ):
        raise StageSError("Stage-R Resume1 implementation paths changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_EVIDENCE_COMMIT) != (
        BASE_EVIDENCE_SUBJECT
    ):
        raise StageSError("Stage-R Resume1 evidence subject changed")
    if commit_name_status(repo, BASE_EVIDENCE_COMMIT) != tuple(sorted(BASE_EVIDENCE_PATHS)):
        raise StageSError("Stage-R Resume1 evidence paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageSError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageSError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageSError("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageSError("Stage-S worktree must be clean")
    source = repo / STAGER_RESUME1_SOURCE
    if sha256_file(source) != EXPECTED_STAGER_RESUME1_SOURCE_SHA256:
        raise StageSError("Stage-R Resume1 source SHA changed")
    if _git_bytes(
        repo, "show", f"{BASE_IMPLEMENTATION_COMMIT}:{STAGER_RESUME1_SOURCE}"
    ) != source.read_bytes():
        raise StageSError("Stage-R Resume1 source differs from committed blob")
    base = validate_base_report(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageSError(f"Stage-S output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "stage_r_resume1_source_sha256": EXPECTED_STAGER_RESUME1_SOURCE_SHA256,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_stage_r_payload_sha256": EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256,
        "base_scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_SHA256,
        "base_report": base,
    }


def _float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise StageSError(f"{label} is non-finite")
    return result


def _cell_key(cell: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(cell.get("base_direction_id")), int(cell.get("timestep")))


def current_fit_projection(stage_r_wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    stage_r = _mapping(stage_r_wrapper.get("stage_r_result"), "Stage-R payload")
    audit = _mapping(
        stage_r.get("tolerance_aligned_oof_post_upper_audit"), "Stage-R audit"
    )
    cells = sorted(
        (_mapping(value, f"cell {index}") for index, value in enumerate(
            _sequence(audit.get("cell_records"), "Stage-R cell records")
        )),
        key=_cell_key,
    )
    records: List[Mapping[str, Any]] = []
    for cell in cells:
        identity = _mapping(cell.get("legacy_identity"), "legacy identity")
        repeat = _mapping(identity.get("portable_repeat_fit"), "portable repeat fit")
        if repeat.get("repeat_prediction_byte_exact") is not True:
            raise StageSError("worker current repeat prediction is not exact")
        records.append({
            "base_direction_id": str(cell.get("base_direction_id")),
            "timestep": int(cell.get("timestep")),
            "current_prediction_sha256": str(repeat.get("current_prediction_sha256")),
            "repeat_prediction_sha256": str(repeat.get("repeat_prediction_sha256")),
            "fold_records_sha256": str(repeat.get("fold_records_sha256")),
            "model_identities_sha256": str(repeat.get("model_identities_sha256")),
            "fold_count": int(repeat.get("fold_count")),
        })
    if len(records) != EXPECTED_CELL_COUNT:
        raise StageSError("current-fit projection cell count changed")
    return {"cell_count": len(records), "records": records}


def functional_projection(stage_r_wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    if stage_r_wrapper.get("execution_verdict") != "PASS":
        raise StageSError("worker Stage-R Resume1 execution did not PASS")
    if stage_r_wrapper.get("scientific_status") != "BLOCKED":
        raise StageSError("worker Stage-R Resume1 scientific status changed")
    if stage_r_wrapper.get("selected_configuration") is not None:
        raise StageSError("worker selected a configuration")
    if stage_r_wrapper.get("train_only_recommendation") is not None:
        raise StageSError("worker emitted a recommendation")
    require_false(stage_r_wrapper, FALSE_BOUNDARIES, "worker Stage-R wrapper")
    stage_r = _mapping(stage_r_wrapper.get("stage_r_result"), "Stage-R payload")
    audit = _mapping(
        stage_r.get("tolerance_aligned_oof_post_upper_audit"), "Stage-R audit"
    )
    classification = copy.deepcopy(
        dict(_mapping(audit.get("classification"), "Stage-R classification"))
    )
    cells = sorted(
        (_mapping(value, f"cell {index}") for index, value in enumerate(
            _sequence(audit.get("cell_records"), "Stage-R cell records")
        )),
        key=_cell_key,
    )
    records: List[Mapping[str, Any]] = []
    for cell in cells:
        identity = _mapping(cell.get("legacy_identity"), "legacy identity")
        if identity.get("all_functional_exact") is not True:
            raise StageSError("legacy functional replay is not exact")
        records.append({
            "base_direction_id": str(cell.get("base_direction_id")),
            "timestep": int(cell.get("timestep")),
            "feature_mode": str(cell.get("feature_mode")),
            "feature_sha256": str(cell.get("feature_sha256")),
            "aligned_acceptance_rate": _float(
                cell.get("aligned_acceptance_rate"), "aligned acceptance"
            ),
            "aligned_acceptance_locus": str(cell.get("aligned_acceptance_locus")),
            "aligned_selected_scale_sha256": str(
                cell.get("aligned_selected_scale_sha256")
            ),
            "aligned_selected_scale_histogram": copy.deepcopy(
                cell.get("aligned_selected_scale_histogram")
            ),
            "aligned_candidate_sha256": str(cell.get("aligned_candidate_sha256")),
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
        })
    if len(records) != EXPECTED_CELL_COUNT:
        raise StageSError("functional projection cell count changed")
    projection = {
        "root_cause": stage_r.get("root_cause"),
        "required_next_path": stage_r.get("required_next_path"),
        "primary_failure_locus": stage_r.get("primary_failure_locus"),
        "cell_count": len(records),
        "scientific_fit_count": int(audit.get("scientific_oof_fit_count")),
        "repeat_fit_count": int(audit.get("portable_repeat_fit_count")),
        "total_fit_count": int(audit.get("total_oof_fit_count")),
        "callback_pair_count": int(audit.get("legacy_callback_off_on_pair_count")),
        "oracle_callback_rerun_count": int(audit.get("oracle_callback_rerun_count")),
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


def validate_confirmation_projection(projection: Mapping[str, Any]) -> Mapping[str, Any]:
    if int(projection.get("cell_count")) != EXPECTED_CELL_COUNT:
        raise StageSError("confirmation cell count changed")
    if int(projection.get("scientific_fit_count")) != EXPECTED_SCIENCE_FITS_PER_WORKER:
        raise StageSError("confirmation science-fit count changed")
    if int(projection.get("repeat_fit_count")) != EXPECTED_REPEAT_FITS_PER_WORKER:
        raise StageSError("confirmation repeat-fit count changed")
    if int(projection.get("total_fit_count")) != EXPECTED_FITS_PER_WORKER:
        raise StageSError("confirmation total-fit count changed")
    if int(projection.get("callback_pair_count")) != EXPECTED_CALLBACK_PAIRS_PER_WORKER:
        raise StageSError("confirmation callback-pair count changed")
    if int(projection.get("oracle_callback_rerun_count")) != 0:
        raise StageSError("oracle controls were rerun")
    classification = _mapping(projection.get("classification"), "classification")
    required = {
        "meaningful_support_cell_count": EXPECTED_NONZERO_SUPPORT_CELLS,
        "zero_acceptance_cell_count": EXPECTED_ZERO_ACCEPTANCE_CELLS,
        "oracle_like_admission_cell_count": EXPECTED_ORACLE_LIKE_CELLS,
        "dominant_dual_oracle_discriminator": EXPECTED_DOMINANT_DISCRIMINATOR,
        "dominant_discriminator_support_count": EXPECTED_DOMINANT_SUPPORT,
        "dominant_discriminator_backbone_coverage": EXPECTED_DOMINANT_BACKBONE_COVERAGE,
        "dominant_discriminator_timestep_coverage": EXPECTED_DOMINANT_TIMESTEP_COVERAGE,
    }
    for key, expected in required.items():
        if classification.get(key) != expected:
            raise StageSError(
                f"confirmation classification changed: {key}="
                f"{classification.get(key)!r} expected={expected!r}"
            )
    if int(classification.get("length_log_z_element_mismatch_count")) != 0:
        raise StageSError("length/log-z mismatch reappeared")
    if int(classification.get("strict_pass_aligned_fail_row_count")) != 0:
        raise StageSError("aligned upper gate introduced a reverse regression")
    if int(classification.get("aligned_upper_element_failure_count")) != 0:
        raise StageSError("aligned upper element failures reappeared")
    records = _sequence(projection.get("cell_records"), "confirmation cells")
    acceptances = [_float(_mapping(value, "cell").get("aligned_acceptance_rate"), "acceptance") for value in records]
    nonzero = sum(value > 0.0 for value in acceptances)
    zero = sum(value == 0.0 for value in acceptances)
    if nonzero != EXPECTED_NONZERO_SUPPORT_CELLS or zero != EXPECTED_ZERO_ACCEPTANCE_CELLS:
        raise StageSError("confirmation acceptance support changed")
    return {
        "cell_count": len(records),
        "nonzero_support_cell_count": nonzero,
        "zero_acceptance_cell_count": zero,
        "acceptance_min": min(acceptances),
        "acceptance_mean": float(sum(acceptances) / len(acceptances)),
        "acceptance_max": max(acceptances),
        "oracle_like_cell_count": sum(value >= 0.95 for value in acceptances),
        "dominant_discriminator": classification.get(
            "dominant_dual_oracle_discriminator"
        ),
        "dominant_support_count": int(
            classification.get("dominant_discriminator_support_count")
        ),
        "dominant_backbone_coverage": int(
            classification.get("dominant_discriminator_backbone_coverage")
        ),
        "dominant_timestep_coverage": int(
            classification.get("dominant_discriminator_timestep_coverage")
        ),
    }


def worker_payload(
    *, worker_id: str, root: Path, repository: Mapping[str, Any]
) -> Mapping[str, Any]:
    validate_environment_variables()
    from ccda_phase3 import (
        phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager,
    )
    from ccda_phase3 import (
        phase314b_r258_stager_resume1_portable_oof_functional_replay as resume1,
    )

    base_report = validate_base_report(Path(root).resolve())
    base_functional = functional_projection(base_report)
    current_environment = stager.probe_environment(Path(root).resolve())
    replay = resume1.execute_recovery(
        root=Path(root).resolve(),
        environment=current_environment,
        repository=_mapping(base_report.get("repository"), "base repository"),
    )
    functional = functional_projection(replay)
    fit_identity = current_fit_projection(replay)
    base_functional_sha = sha256_bytes(stable_json_bytes(base_functional))
    functional_sha = sha256_bytes(stable_json_bytes(functional))
    if functional_sha != base_functional_sha:
        raise StageSError("worker functional projection differs from Stage-R Resume1")
    summary = validate_confirmation_projection(functional)
    return {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "worker_id": str(worker_id),
        "execution_verdict": "PASS",
        "repository_head": repository.get("head"),
        "environment": current_environment,
        "environment_sha256": sha256_bytes(stable_json_bytes(current_environment)),
        "base_functional_projection_sha256": base_functional_sha,
        "functional_projection_sha256": functional_sha,
        "current_fit_projection_sha256": sha256_bytes(
            stable_json_bytes(fit_identity)
        ),
        "functional_projection_matches_base": True,
        "functional_projection": functional,
        "current_fit_projection": fit_identity,
        "confirmation_summary": summary,
        "scientific_oof_fit_count": EXPECTED_SCIENCE_FITS_PER_WORKER,
        "repeat_identity_fit_count": EXPECTED_REPEAT_FITS_PER_WORKER,
        "total_oof_fit_count": EXPECTED_FITS_PER_WORKER,
        "callback_pair_count": EXPECTED_CALLBACK_PAIRS_PER_WORKER,
        "oracle_callback_rerun_count": 0,
        "internal_scale_attempt_count": EXPECTED_INTERNAL_ATTEMPTS_PER_WORKER,
        **{key: False for key in FALSE_BOUNDARIES},
    }


def compare_workers(first: Mapping[str, Any], second: Mapping[str, Any]) -> Mapping[str, Any]:
    for index, worker in enumerate((first, second)):
        if worker.get("execution_verdict") != "PASS":
            raise StageSError(f"worker {index} did not PASS")
        if worker.get("functional_projection_matches_base") is not True:
            raise StageSError(f"worker {index} does not match the base projection")
        require_false(worker, FALSE_BOUNDARIES, f"worker {index}")
    checks = {
        "functional_projection_sha256": first.get("functional_projection_sha256")
        == second.get("functional_projection_sha256"),
        "current_fit_projection_sha256": first.get("current_fit_projection_sha256")
        == second.get("current_fit_projection_sha256"),
        "environment_sha256": first.get("environment_sha256")
        == second.get("environment_sha256"),
        "confirmation_summary": stable_json_bytes(first.get("confirmation_summary"))
        == stable_json_bytes(second.get("confirmation_summary")),
        "functional_projection": stable_json_bytes(first.get("functional_projection"))
        == stable_json_bytes(second.get("functional_projection")),
        "current_fit_projection": stable_json_bytes(first.get("current_fit_projection"))
        == stable_json_bytes(second.get("current_fit_projection")),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageSError(f"isolated Stage-S workers differ: {failed}")
    return {"all_exact": True, "checks": checks}


def classify_confirmation(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    nonzero = int(summary.get("nonzero_support_cell_count"))
    zero = int(summary.get("zero_acceptance_cell_count"))
    dominant_support = int(summary.get("dominant_support_count"))
    backbone_coverage = int(summary.get("dominant_backbone_coverage"))
    timestep_coverage = int(summary.get("dominant_timestep_coverage"))
    if nonzero == EXPECTED_CELL_COUNT and zero == 0:
        if (
            dominant_support >= STABLE_DISCRIMINATOR_MIN
            and backbone_coverage >= 8
            and timestep_coverage == EXPECTED_TIMESTEP_COUNT
        ):
            root = "phase314b_r258_stages_stable_post_upper_discriminator_confirmed"
            next_path = "REPAIR_CONFIRMED_TOLERANCE_ALIGNED_POST_UPPER_DISCRIMINATOR"
            locus = "stable_post_upper_discriminator"
        else:
            root = "phase314b_r258_stages_tolerance_aligned_oof_support_confirmed"
            next_path = (
                "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_OOF_CANDIDATE_MATRIX_"
                "ON_OBJECTIVE_TRAIN_ONLY"
            )
            locus = "confirmed_broad_oof_support"
    elif nonzero >= 24:
        root = "phase314b_r258_stages_tolerance_aligned_oof_support_partially_confirmed"
        next_path = "STRATIFY_CONFIRMED_OOF_SUPPORT_BY_BACKBONE_AND_TIMESTEP"
        locus = "partial_oof_support"
    else:
        root = "phase314b_r258_stages_tolerance_aligned_oof_support_not_confirmed"
        next_path = "AUDIT_TOLERANCE_ALIGNED_OOF_CONFIRMATION_INSTABILITY"
        locus = "oof_support_confirmation_failure"
    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
    }


def run_confirmation(
    *, root: Path, repository: Mapping[str, Any], python_bin: str
) -> Mapping[str, Any]:
    validate_environment_variables()
    repo = Path(root).resolve()
    base_report = _mapping(repository.get("base_report"), "base report")
    base_projection = functional_projection(base_report)
    base_projection_sha = sha256_bytes(stable_json_bytes(base_projection))
    with tempfile.TemporaryDirectory(prefix="phase314b_r258_stages_") as temporary:
        directory = Path(temporary)
        outputs: List[Path] = []
        for index in range(EXPECTED_WORKER_COUNT):
            output = directory / f"worker_{index}.json"
            outputs.append(output)
            command = [
                str(python_bin),
                str(repo / "scripts/phase3_14b_r258_stages_worker.py"),
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
                raise StageSError(
                    f"worker {index} rc={completed.returncode} "
                    f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
                )
        workers = [load_json(path) for path in outputs]
    comparison = compare_workers(workers[0], workers[1])
    if workers[0].get("functional_projection_sha256") != base_projection_sha:
        raise StageSError("confirmed projection differs from frozen Stage-R Resume1")
    summary = _mapping(workers[0].get("confirmation_summary"), "confirmation summary")
    classification = classify_confirmation(summary)
    worker_summaries = []
    for worker in workers:
        worker_summaries.append({
            "worker_id": worker.get("worker_id"),
            "environment_sha256": worker.get("environment_sha256"),
            "functional_projection_sha256": worker.get("functional_projection_sha256"),
            "current_fit_projection_sha256": worker.get("current_fit_projection_sha256"),
            "functional_projection_matches_base": True,
            "scientific_oof_fit_count": worker.get("scientific_oof_fit_count"),
            "repeat_identity_fit_count": worker.get("repeat_identity_fit_count"),
            "total_oof_fit_count": worker.get("total_oof_fit_count"),
            "callback_pair_count": worker.get("callback_pair_count"),
            "oracle_callback_rerun_count": worker.get("oracle_callback_rerun_count"),
            "internal_scale_attempt_count": worker.get("internal_scale_attempt_count"),
        })
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": {key: value for key, value in repository.items() if key != "base_report"},
        "immutable_inputs": {
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "base_stage_r_payload_sha256": EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256,
            "base_scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_SHA256,
            "stage_r_resume1_source_sha256": EXPECTED_STAGER_RESUME1_SOURCE_SHA256,
            "objective_train_only": True,
            "oof_cell_count": EXPECTED_CELL_COUNT,
            "backbone_count": EXPECTED_BACKBONE_COUNT,
            "timestep_count": EXPECTED_TIMESTEP_COUNT,
            "worker_count": EXPECTED_WORKER_COUNT,
        },
        "confirmation_execution": {
            "worker_count": EXPECTED_WORKER_COUNT,
            "workers": worker_summaries,
            "worker_comparison": comparison,
            "workers_functional_projection_matches_base": True,
            "workers_current_fit_identity_byte_exact": True,
            "total_scientific_oof_fit_count": EXPECTED_SCIENCE_FITS_PER_WORKER
            * EXPECTED_WORKER_COUNT,
            "total_repeat_identity_fit_count": EXPECTED_REPEAT_FITS_PER_WORKER
            * EXPECTED_WORKER_COUNT,
            "total_oof_fit_count": EXPECTED_TOTAL_FITS,
            "total_callback_pair_count": EXPECTED_TOTAL_CALLBACK_PAIRS,
            "oracle_callback_rerun_count": 0,
            "total_internal_scale_attempt_count": EXPECTED_TOTAL_INTERNAL_ATTEMPTS,
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
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stages_oof_support_confirmation_failed",
        "required_next_path": "RESTORE_STAGES_OOF_SUPPORT_CONFIRMATION",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else {
            key: value for key, value in repository.items() if key != "base_report"
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_evidence_preserved": True,
        **{key: False for key in FALSE_BOUNDARIES},
    }
