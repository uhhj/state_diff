"""Phase3.14b-r2.5.8 Stage W evidence-only transfer-failure audit.

Stage V Resume1 completed the single authorised selection-holdout evaluation.
All three locked timesteps retained mechanism/gate legality but failed both
pre-registered MSE-ratio criteria.  Stage W is deliberately evidence-only:
it reads the immutable Stage-U frontier contract, the immutable Stage-V
Resume1 evidence, the original pre-holdout blocked report, and Git provenance.
It does not import scientific runtime modules, load datasets, fit models,
generate candidates, read a holdout target, or access the frozen probe.

The audit localises the failure as far as the persisted aggregates permit.  In
particular, when accepted-row MSE exceeds the baseline while more than half of
accepted rows reduce Euclidean distance, a minority non-improving side must
carry enough adverse squared-error mass to dominate the accepted-row total.
The exact rows, conditions, seeds, horizons, and segments remain unidentifiable
because row-level tensors and metadata were intentionally not persisted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage W"
SCHEMA = "phase314b_r258_stagew_selection_holdout_transfer_failure_audit_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"
CONTRACT_SCHEMA = SCHEMA + "_negative_result_contract_v1"

BASE_STAGEV_RESUME1_IMPLEMENTATION_COMMIT = "33b11b7871bea37b8078b2c7f1ce21bd618368be"
BASE_STAGEV_RESUME1_EVIDENCE_COMMIT = "55e3cfdcbc4ed8d3392f9c307f793fdd1799de6c"
EXPECTED_STAGEV_RESUME1_PARENT = "e02d8131b49c0b012e188b79a4e46b5c2f9cfbba"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEU_CONTRACT = "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"
STAGEV_RESUME1_SUMMARY = (
    "reports/phase3_14b_r258_stagev_resume1_locked_selection_holdout_"
    "evaluation_summary.json"
)
ORIGINAL_STAGEV_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagev_locked_selection_holdout_evaluation_"
    "blocked_summary.json"
)
OUTPUT_REPORT = (
    "reports/phase3_14b_r258_stagew_selection_holdout_transfer_failure_"
    "audit_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagew_selection_holdout_transfer_failure_"
    "audit_blocked_summary.json"
)

EXPECTED_STAGEU_CONTRACT_SHA256 = (
    "c0bf2c477694cf1bda77f11c7ac407befb77d005afa8422e7e9bae2616714a8c"
)
EXPECTED_STAGEU_CONTRACT_PAYLOAD_SHA256 = (
    "ab2011731f9cd6c560d7721673aa33d09ebab05fa17fbf9a1d64cb35f1cce985"
)
EXPECTED_STAGEU_LOCKED_CELLS_SHA256 = (
    "a2367bd378b9ddba84f839b0409cc658541b69ba1a2aa1c9bd4ce9c1d2063bd1"
)
EXPECTED_STAGEV_RESUME1_SUMMARY_SHA256 = (
    "1b27e2bd20f1022cf2baa8017b995c27e3c9fc486ea72761c7305415990f181b"
)
EXPECTED_STAGEV_RESUME1_SCIENTIFIC_RESULT_SHA256 = (
    "e760df33caa81f4794d739a76e60ccf9c12326547ae14a98f0c4f23e2f3c5946"
)
EXPECTED_ORIGINAL_STAGEV_BLOCKED_REPORT_SHA256 = (
    "f9facbe30f6292256451ec94ceaaf87654b5ea5cd95892956e4433ba90bb033c"
)

LOCKED_BACKBONE = "segment_target_rr64_feasible"
LOCKED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_OBJECTIVE_ROWS = 638
EXPECTED_HOLDOUT_ROWS = 236
EXPECTED_CUMULATIVE_HOLDOUT_EVALUATIONS = 1

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage W: audit selection-holdout transfer failure"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage W transfer-failure audit evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage W blocked audit evidence"
)
BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage V Resume1: recover pre-holdout CUDA execution"
)
BASE_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage V Resume1 locked holdout evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stagew_selection_holdout_transfer_failure_audit.py",
    ),
    ("A", "scripts/phase3_14b_r258_stagew_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stagew_selection_holdout_transfer_failure_audit.py",
    ),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
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
BASE_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGEV_RESUME1_SUMMARY),
)

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stageu_candidate_frontier_lock.py": (
        "a4455727e20ab96658686dc42eb0ab35b7bddadef8e9e99648d2c57371bae657"
    ),
    "ccda_phase3/phase314b_r258_stagev_locked_selection_holdout_evaluation.py": (
        "5aedabe9004630207021f1252f5b99aa6524b8dc9b3077003b20c8b163e7b1b0"
    ),
    "ccda_phase3/phase314b_r258_stagev_resume1_preholdout_cuda_recovery.py": (
        "083e11939797af7247355f9e04ac5db09ab069c0ab89f85e9b09eae99516c002"
    ),
    "scripts/phase3_14b_r258_stagev_resume1_execute.py": (
        "5d36695024aa111b5c2ff786d6bc58cb434e06fa6757804a9dd773b84c9492a9"
    ),
    "tests/test_phase3_14b_r258_stagev_resume1_preholdout_cuda_recovery.py": (
        "25b3cfdfbaee3c8d72faccac85867d0a644504e218494c258657ac2041135f30"
    ),
}

EXPECTED_ENV: Mapping[str, str] = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded",
    "selection_holdout_candidate_regenerated",
    "selection_holdout_metric_recomputed",
    "objective_train_refit",
    "model_fit_run",
    "candidate_generation_run",
    "internal_scale_search_run",
    "backbone_search_reopened",
    "timestep_search_reopened",
    "threshold_changed",
    "fallback_backbone_used",
    "fallback_timestep_used",
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
    "direction_tensor_persisted",
    "candidate_tensor_persisted",
    "holdout_target_tensor_persisted",
    "row_level_metric_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageWError(RuntimeError):
    """Fail-closed Stage-W evidence-audit error."""


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
        raise StageWError(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageWError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageWError(f"{label} is not a sequence")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise StageWError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise StageWError(f"{label} is non-finite")
    return result


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StageWError(f"{label} is boolean, not integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise StageWError(f"{label} is not an integer: {value!r}") from error
    if result != value:
        raise StageWError(f"{label} is not exact integer-valued: {value!r}")
    return result


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
        raise StageWError(
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
        raise StageWError(
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


def validate_environment_variables() -> Mapping[str, str]:
    observed = {key: os.environ.get(key) for key in EXPECTED_ENV}
    if observed != dict(EXPECTED_ENV):
        raise StageWError(f"deterministic environment changed: {observed!r}")
    return dict(EXPECTED_ENV)


def require_false(payload: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if payload.get(key) is not False:
            raise StageWError(f"{label} boundary changed: {key}={payload.get(key)!r}")


def verify_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: Optional[str] = None,
    label: str,
) -> str:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise StageWError(f"{label} {field} is invalid")
    copy_value = copy.deepcopy(dict(payload))
    copy_value.pop(field, None)
    calculated = sha256_bytes(stable_json_bytes(copy_value))
    if calculated != observed:
        raise StageWError(
            f"{label} self-hash changed: calculated={calculated} observed={observed}"
        )
    if expected is not None and observed != expected:
        raise StageWError(f"{label} expected {field} changed")
    return observed


def validate_scalar_stats(value: Any, label: str) -> Mapping[str, Any]:
    stats = _mapping(value, label)
    expected_keys = ("count", "mean", "std", "min", "p05", "median", "p95", "max")
    if set(stats) != set(expected_keys):
        raise StageWError(f"{label} keys changed: {sorted(stats)!r}")
    count = _exact_int(stats["count"], f"{label}.count")
    if count < 0:
        raise StageWError(f"{label}.count is negative")
    numbers = {
        key: _finite_float(stats[key], f"{label}.{key}")
        for key in expected_keys
        if key != "count"
    }
    if numbers["std"] < 0.0:
        raise StageWError(f"{label}.std is negative")
    ordered = [
        numbers["min"],
        numbers["p05"],
        numbers["median"],
        numbers["p95"],
        numbers["max"],
    ]
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        raise StageWError(f"{label} quantiles are not ordered")
    return {"count": count, **numbers}


def validate_stageu_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    if contract.get("phase") != "Phase3.14b-r2.5.8 Stage U":
        raise StageWError("Stage-U phase changed")
    if contract.get("schema") != "phase314b_r258_stageu_candidate_frontier_contract_v1":
        raise StageWError("Stage-U schema changed")
    verify_self_hash(
        contract,
        field="contract_payload_sha256",
        expected=EXPECTED_STAGEU_CONTRACT_PAYLOAD_SHA256,
        label="Stage-U contract",
    )
    frontier = _mapping(contract.get("frontier_lock"), "Stage-U frontier")
    exact_frontier = {
        "backbone_id": LOCKED_BACKBONE,
        "timesteps": list(LOCKED_TIMESTEPS),
        "timestep_policy": "joint_all_timesteps_no_cherry_pick",
        "backbone_search_reopened": False,
        "timestep_search_reopened": False,
        "fallback_backbone_allowed": False,
        "fallback_timestep_allowed": False,
        "frontier_is_unique": True,
        "all_locked_timesteps_fidelity_eligible": True,
        "locked_cells_sha256": EXPECTED_STAGEU_LOCKED_CELLS_SHA256,
    }
    for key, expected in exact_frontier.items():
        if frontier.get(key) != expected:
            raise StageWError(f"Stage-U frontier changed: {key}")
    cells = _sequence(frontier.get("locked_cells"), "Stage-U locked cells")
    if sha256_bytes(stable_json_bytes(cells)) != EXPECTED_STAGEU_LOCKED_CELLS_SHA256:
        raise StageWError("Stage-U locked cells payload changed")
    if [int(_mapping(cell, "Stage-U cell")["timestep"]) for cell in cells] != list(
        LOCKED_TIMESTEPS
    ):
        raise StageWError("Stage-U locked timestep order changed")
    validated_cells = []
    for item in cells:
        cell = _mapping(item, "Stage-U cell")
        timestep = _exact_int(cell.get("timestep"), "Stage-U timestep")
        if cell.get("base_direction_id") != LOCKED_BACKBONE:
            raise StageWError(f"Stage-U backbone changed at t={timestep}")
        if _exact_int(cell.get("row_count"), f"Stage-U row count t={timestep}") != EXPECTED_OBJECTIVE_ROWS:
            raise StageWError(f"Stage-U row count changed at t={timestep}")
        if cell.get("mechanism_eligible") is not True or cell.get("fidelity_eligible") is not True:
            raise StageWError(f"Stage-U eligibility changed at t={timestep}")
        if any(
            _exact_int(cell.get(key), f"Stage-U {key} t={timestep}") != 0
            for key in (
                "length_log_z_element_mismatch_count",
                "aligned_upper_element_failure_count",
                "strict_pass_aligned_fail_row_count",
            )
        ):
            raise StageWError(f"Stage-U gate cleanliness changed at t={timestep}")
        selected_count = _exact_int(
            cell.get("selected_row_count"), f"Stage-U selected count t={timestep}"
        )
        acceptance = _finite_float(
            cell.get("aligned_acceptance_rate"), f"Stage-U acceptance t={timestep}"
        )
        if abs(acceptance - selected_count / EXPECTED_OBJECTIVE_ROWS) > 1.0e-15:
            raise StageWError(f"Stage-U acceptance/count mismatch at t={timestep}")
        metrics = {
            "timestep": timestep,
            "row_count": EXPECTED_OBJECTIVE_ROWS,
            "selected_row_count": selected_count,
            "acceptance_rate": acceptance,
            "overall_mse_ratio": _finite_float(
                cell.get("overall_mse_ratio"), f"Stage-U overall ratio t={timestep}"
            ),
            "accepted_row_mse_ratio": _finite_float(
                cell.get("accepted_row_mse_ratio"),
                f"Stage-U accepted ratio t={timestep}",
            ),
            "positive_distance_reduction_rate": _finite_float(
                cell.get("positive_distance_reduction_rate"),
                f"Stage-U positive rate t={timestep}",
            ),
            "relative_distance_reduction_mean": _finite_float(
                cell.get("relative_distance_reduction_mean"),
                f"Stage-U relative mean t={timestep}",
            ),
            "selected_scale_histogram": copy.deepcopy(
                dict(_mapping(cell.get("aligned_selected_scale_histogram"), "Stage-U histogram"))
            ),
        }
        if not (
            metrics["overall_mse_ratio"] < 1.0
            and metrics["accepted_row_mse_ratio"] < 1.0
            and metrics["positive_distance_reduction_rate"] > 0.5
            and metrics["relative_distance_reduction_mean"] > 0.0
        ):
            raise StageWError(f"Stage-U fidelity lock changed at t={timestep}")
        validated_cells.append(metrics)
    policy = _mapping(contract.get("next_stage_holdout_policy"), "Stage-U holdout policy")
    exact_policy = {
        "evaluation_count": 1,
        "population": "locked_backbone_all_three_timesteps",
        "backbone_id": LOCKED_BACKBONE,
        "timesteps": list(LOCKED_TIMESTEPS),
        "all_timesteps_must_pass": True,
        "post_holdout_backbone_selection_forbidden": True,
        "post_holdout_timestep_selection_forbidden": True,
        "threshold_changes_after_holdout_forbidden": True,
        "rerun_after_failure_forbidden": True,
        "aggregate_metrics_are_diagnostic_only": True,
        "frozen_probe_remains_closed": True,
    }
    for key, expected in exact_policy.items():
        if policy.get(key) != expected:
            raise StageWError(f"Stage-U holdout policy changed: {key}")
    return {
        "locked_cells": validated_cells,
        "aligned_gate_lock": copy.deepcopy(
            dict(_mapping(contract.get("aligned_gate_lock"), "Stage-U gate lock"))
        ),
    }


def _validate_histogram(value: Any, *, row_count: int, label: str) -> Mapping[str, int]:
    histogram = _mapping(value, label)
    result: Dict[str, int] = {}
    for key, count_value in histogram.items():
        try:
            float(str(key))
        except ValueError as error:
            raise StageWError(f"{label} contains nonnumeric scale key: {key!r}") from error
        count = _exact_int(count_value, f"{label}[{key!r}]")
        if count < 0:
            raise StageWError(f"{label} contains negative count")
        result[str(key)] = count
    if sum(result.values()) != row_count:
        raise StageWError(f"{label} population changed: {sum(result.values())} != {row_count}")
    return dict(sorted(result.items(), key=lambda item: float(item[0])))


def _validate_timestep_record(record_value: Any, timestep: int) -> Mapping[str, Any]:
    record = _mapping(record_value, f"Stage-V t={timestep} record")
    if _exact_int(record.get("timestep"), f"Stage-V timestep t={timestep}") != timestep:
        raise StageWError(f"Stage-V timestep order changed at t={timestep}")
    if record.get("base_direction_id") != LOCKED_BACKBONE:
        raise StageWError(f"Stage-V backbone changed at t={timestep}")
    if record.get("feature_mode") != "full_segment_constraint":
        raise StageWError(f"Stage-V feature mode changed at t={timestep}")
    if _exact_int(
        record.get("selection_holdout_rows"), f"Stage-V holdout rows t={timestep}"
    ) != EXPECTED_HOLDOUT_ROWS:
        raise StageWError(f"Stage-V holdout row count changed at t={timestep}")
    if _exact_int(record.get("full_fit_repeat_count"), f"Stage-V fits t={timestep}") != 2:
        raise StageWError(f"Stage-V fit repeat count changed at t={timestep}")
    if record.get("candidate_finalized_before_holdout_target_access") is not True:
        raise StageWError(f"Stage-V candidate/target order changed at t={timestep}")
    if record.get("holdout_target_used_for_fit") is not False:
        raise StageWError(f"Stage-V holdout used for fit at t={timestep}")
    if record.get("holdout_target_used_for_selection") is not False:
        raise StageWError(f"Stage-V holdout used for selection at t={timestep}")
    if record.get("holdout_target_used_only_for_final_metrics") is not True:
        raise StageWError(f"Stage-V holdout role changed at t={timestep}")
    if _exact_int(
        record.get("holdout_target_access_count"), f"Stage-V target access t={timestep}"
    ) != 1:
        raise StageWError(f"Stage-V target access count changed at t={timestep}")
    for key in (
        "candidate_tensor_persisted",
        "selected_scale_tensor_persisted",
        "direction_tensor_persisted",
        "holdout_target_tensor_persisted",
    ):
        if record.get(key) is not False:
            raise StageWError(f"Stage-V tensor boundary changed at t={timestep}: {key}")
    for key in (
        "length_log_z_element_mismatch_count",
        "aligned_upper_element_failure_count",
        "strict_pass_aligned_fail_row_count",
    ):
        if _exact_int(record.get(key), f"Stage-V {key} t={timestep}") != 0:
            raise StageWError(f"Stage-V gate regression at t={timestep}: {key}")
    fidelity = _mapping(record.get("candidate_fidelity"), f"Stage-V fidelity t={timestep}")
    row_count = _exact_int(fidelity.get("row_count"), f"Stage-V row count t={timestep}")
    if row_count != EXPECTED_HOLDOUT_ROWS:
        raise StageWError(f"Stage-V fidelity row count changed at t={timestep}")
    selected_count = _exact_int(
        fidelity.get("selected_row_count"), f"Stage-V selected count t={timestep}"
    )
    if not 0 < selected_count < row_count:
        raise StageWError(f"Stage-V selected population invalid at t={timestep}")
    acceptance = _finite_float(
        fidelity.get("acceptance_rate"), f"Stage-V acceptance t={timestep}"
    )
    if abs(acceptance - selected_count / row_count) > 1.0e-15:
        raise StageWError(f"Stage-V acceptance/count mismatch at t={timestep}")
    record_acceptance = _finite_float(
        record.get("acceptance_rate"), f"Stage-V record acceptance t={timestep}"
    )
    if record_acceptance != acceptance:
        raise StageWError(f"Stage-V acceptance duplication changed at t={timestep}")
    histogram = _validate_histogram(
        record.get("selected_scale_histogram"),
        row_count=row_count,
        label=f"Stage-V histogram t={timestep}",
    )
    if histogram.get("0", 0) != row_count - selected_count:
        raise StageWError(f"Stage-V zero-scale population changed at t={timestep}")
    overall_ratio = _finite_float(
        fidelity.get("overall_mse_ratio"), f"Stage-V overall ratio t={timestep}"
    )
    accepted = _mapping(fidelity.get("accepted_rows"), f"Stage-V accepted rows t={timestep}")
    accepted_ratio = _finite_float(
        accepted.get("mse_ratio"), f"Stage-V accepted ratio t={timestep}"
    )
    positive_rate = _finite_float(
        accepted.get("positive_distance_reduction_rate"),
        f"Stage-V positive rate t={timestep}",
    )
    relative_stats = validate_scalar_stats(
        accepted.get("relative_distance_reduction"),
        f"Stage-V accepted relative reduction t={timestep}",
    )
    if relative_stats["count"] != selected_count:
        raise StageWError(f"Stage-V accepted relative count changed at t={timestep}")
    accepted_control_stats = validate_scalar_stats(
        accepted.get("control_distance"),
        f"Stage-V accepted control distance t={timestep}",
    )
    accepted_candidate_stats = validate_scalar_stats(
        accepted.get("candidate_distance"),
        f"Stage-V accepted candidate distance t={timestep}",
    )
    accepted_reduction_stats = validate_scalar_stats(
        accepted.get("distance_reduction"),
        f"Stage-V accepted distance reduction t={timestep}",
    )
    accepted_motion_stats = validate_scalar_stats(
        accepted.get("motion_norm"),
        f"Stage-V accepted motion t={timestep}",
    )
    for stats_label, stats in (
        ("control", accepted_control_stats),
        ("candidate", accepted_candidate_stats),
        ("reduction", accepted_reduction_stats),
        ("motion", accepted_motion_stats),
    ):
        if stats["count"] != selected_count:
            raise StageWError(
                f"Stage-V accepted {stats_label} count changed at t={timestep}"
            )
    overall_control_stats = validate_scalar_stats(
        fidelity.get("overall_control_distance"),
        f"Stage-V overall control distance t={timestep}",
    )
    overall_candidate_stats = validate_scalar_stats(
        fidelity.get("overall_candidate_distance"),
        f"Stage-V overall candidate distance t={timestep}",
    )
    overall_reduction_stats = validate_scalar_stats(
        fidelity.get("overall_distance_reduction"),
        f"Stage-V overall distance reduction t={timestep}",
    )
    overall_relative_stats = validate_scalar_stats(
        fidelity.get("overall_relative_distance_reduction"),
        f"Stage-V overall relative reduction t={timestep}",
    )
    overall_motion_stats = validate_scalar_stats(
        fidelity.get("overall_motion_norm"),
        f"Stage-V overall motion t={timestep}",
    )
    for stats_label, stats in (
        ("control", overall_control_stats),
        ("candidate", overall_candidate_stats),
        ("reduction", overall_reduction_stats),
        ("relative", overall_relative_stats),
        ("motion", overall_motion_stats),
    ):
        if stats["count"] != row_count:
            raise StageWError(f"Stage-V overall {stats_label} count changed at t={timestep}")
    checks = _mapping(record.get("pass_checks"), f"Stage-V checks t={timestep}")
    expected_checks = {
        "mechanism_eligible": True,
        "overall_mse_ratio": False,
        "accepted_row_mse_ratio": False,
        "positive_distance_reduction_rate": True,
        "relative_distance_reduction_mean": True,
        "length_log_z_element_mismatch_count": True,
        "aligned_upper_element_failure_count": True,
        "strict_pass_aligned_fail_row_count": True,
        "all": False,
    }
    if dict(checks) != expected_checks:
        raise StageWError(f"Stage-V failed-check signature changed at t={timestep}")
    if fidelity.get("mechanism_eligible") is not True:
        raise StageWError(f"Stage-V mechanism eligibility changed at t={timestep}")
    if fidelity.get("fidelity_eligible") is not False:
        raise StageWError(f"Stage-V fidelity eligibility changed at t={timestep}")
    if record.get("scientific_pass") is not False:
        raise StageWError(f"Stage-V timestep pass changed at t={timestep}")
    if not (
        overall_ratio > 1.0
        and accepted_ratio > 1.0
        and positive_rate > 0.5
        and relative_stats["mean"] > 0.0
    ):
        raise StageWError(f"Stage-V transfer-failure signature changed at t={timestep}")
    return {
        "timestep": timestep,
        "row_count": row_count,
        "selected_row_count": selected_count,
        "acceptance_rate": acceptance,
        "overall_mse_ratio": overall_ratio,
        "accepted_row_mse_ratio": accepted_ratio,
        "positive_distance_reduction_rate": positive_rate,
        "relative_distance_reduction_mean": relative_stats["mean"],
        "selected_scale_histogram": histogram,
        "accepted_distribution": {
            "control_distance": accepted_control_stats,
            "candidate_distance": accepted_candidate_stats,
            "distance_reduction": accepted_reduction_stats,
            "relative_distance_reduction": relative_stats,
            "motion_norm": accepted_motion_stats,
        },
        "overall_distribution": {
            "control_distance": overall_control_stats,
            "candidate_distance": overall_candidate_stats,
            "distance_reduction": overall_reduction_stats,
            "relative_distance_reduction": overall_relative_stats,
            "motion_norm": overall_motion_stats,
        },
        "pass_checks": dict(checks),
    }


def validate_stagev_resume1_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if summary.get("phase") != "Phase3.14b-r2.5.8 Stage V Resume1":
        raise StageWError("Stage-V Resume1 phase changed")
    if summary.get("schema") != "phase314b_r258_stagev_resume1_preholdout_cuda_recovery_v1":
        raise StageWError("Stage-V Resume1 schema changed")
    verify_self_hash(
        summary,
        field="scientific_result_sha256",
        expected=EXPECTED_STAGEV_RESUME1_SCIENTIFIC_RESULT_SHA256,
        label="Stage-V Resume1 summary",
    )
    exact = {
        "execution_verdict": "PASS",
        "resume1_execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagev_locked_frontier_fails_preregistered_"
            "selection_holdout_contract"
        ),
        "required_next_path": (
            "AUDIT_SELECTION_HOLDOUT_TRANSFER_FAILURE_WITHOUT_RERUN_"
            "RETUNING_OR_FALLBACK"
        ),
        "primary_failure_locus": "all_timestep_fidelity_transfer_failure",
        "passed_timesteps": [],
        "failed_timesteps": list(LOCKED_TIMESTEPS),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": True,
        "selection_holdout_used_for_fit_or_selection": False,
        "evaluation_count": 1,
        "rerun_authorized": False,
        "resume1_rerun_authorized": False,
        "holdout_access_before_resume1": False,
        "holdout_access_before_resume1_proven": True,
        "prior_selection_holdout_evaluation_count": 0,
        "resume1_selection_holdout_evaluation_count": 1,
        "cumulative_selection_holdout_evaluation_count": 1,
        "original_stagev_blocked_report_preserved": True,
        "original_stagev_blocked_report_sha256": (
            EXPECTED_ORIGINAL_STAGEV_BLOCKED_REPORT_SHA256
        ),
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise StageWError(f"Stage-V Resume1 summary changed: {key}")
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
        raise StageWError("Stage-V execution counts changed")
    topology = _mapping(summary.get("process_topology"), "Stage-V process topology")
    expected_topology = {
        "environment_probe_process_count": 1,
        "cold_science_worker_process_count": 1,
        "processes_distinct": True,
        "workers_launched_sequentially": True,
        "temporary_payloads_deleted": True,
    }
    for key, expected in expected_topology.items():
        if topology.get(key) != expected:
            raise StageWError(f"Stage-V process topology changed: {key}")
    if _exact_int(topology.get("probe_process_id"), "Stage-V probe PID") == _exact_int(
        topology.get("science_worker_process_id"), "Stage-V science PID"
    ):
        raise StageWError("Stage-V process IDs are not distinct")
    recovery_scope = _mapping(summary.get("recovery_scope"), "Stage-V recovery scope")
    expected_scope = {
        "execution_contract_only": True,
        "cuda_unavailable_failure_recovered": True,
        "science_code_changed": False,
        "frontier_changed": False,
        "timestep_policy_changed": False,
        "gate_changed": False,
        "threshold_changed": False,
        "fallback_opened": False,
        "holdout_reaccessed": False,
    }
    if dict(recovery_scope) != expected_scope:
        raise StageWError("Stage-V recovery scope changed")
    records = _sequence(summary.get("timestep_records"), "Stage-V timestep records")
    if len(records) != len(LOCKED_TIMESTEPS):
        raise StageWError("Stage-V timestep population changed")
    validated_records = [
        _validate_timestep_record(record, timestep)
        for record, timestep in zip(records, LOCKED_TIMESTEPS)
    ]
    for key in (
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
    ):
        if summary.get(key) is not False:
            raise StageWError(f"Stage-V boundary changed: {key}")
    return {
        "records": validated_records,
        "execution": dict(execution),
        "process_topology": dict(topology),
    }


def normalize_histogram(histogram: Mapping[str, int]) -> Mapping[str, float]:
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        raise StageWError("cannot normalize empty histogram")
    return {key: int(value) / total for key, value in histogram.items()}


def histogram_total_variation(
    left: Mapping[str, int], right: Mapping[str, int]
) -> float:
    left_normalized = normalize_histogram(left)
    right_normalized = normalize_histogram(right)
    keys = sorted(set(left_normalized) | set(right_normalized), key=float)
    return 0.5 * sum(
        abs(left_normalized.get(key, 0.0) - right_normalized.get(key, 0.0))
        for key in keys
    )


def integer_positive_count(rate: float, selected_count: int, label: str) -> int:
    if selected_count <= 0:
        raise StageWError(f"{label} selected count is not positive")
    estimate = int(round(float(rate) * selected_count))
    if not 0 <= estimate <= selected_count:
        raise StageWError(f"{label} inferred positive count is invalid")
    if abs(float(rate) - estimate / selected_count) > 1.0e-12:
        raise StageWError(f"{label} positive-rate/count identity changed")
    return estimate


def accepted_baseline_sse_share(overall_ratio: float, accepted_ratio: float) -> float:
    numerator = float(overall_ratio) - 1.0
    denominator = float(accepted_ratio) - 1.0
    if abs(denominator) <= 1.0e-15:
        raise StageWError("accepted MSE ratio is too close to one for SSE-share recovery")
    share = numerator / denominator
    if not math.isfinite(share) or not -1.0e-12 <= share <= 1.0 + 1.0e-12:
        raise StageWError(f"recovered accepted baseline SSE share is invalid: {share}")
    share = min(1.0, max(0.0, share))
    reconstructed = 1.0 + share * (float(accepted_ratio) - 1.0)
    if abs(reconstructed - float(overall_ratio)) > 5.0e-13:
        raise StageWError("accepted baseline SSE-share identity does not reconstruct")
    return share


def build_timestep_diagnostic(
    train: Mapping[str, Any], holdout: Mapping[str, Any]
) -> Mapping[str, Any]:
    timestep = int(train["timestep"])
    if int(holdout["timestep"]) != timestep:
        raise StageWError("train/holdout timestep mismatch")
    selected = int(holdout["selected_row_count"])
    positive = integer_positive_count(
        float(holdout["positive_distance_reduction_rate"]),
        selected,
        f"t={timestep}",
    )
    nonpositive = selected - positive
    accepted_ratio = float(holdout["accepted_row_mse_ratio"])
    overall_ratio = float(holdout["overall_mse_ratio"])
    train_accepted_share = accepted_baseline_sse_share(
        float(train["overall_mse_ratio"]),
        float(train["accepted_row_mse_ratio"]),
    )
    holdout_accepted_share = accepted_baseline_sse_share(
        overall_ratio,
        accepted_ratio,
    )
    minority_signature = positive > nonpositive and accepted_ratio > 1.0
    if not minority_signature:
        raise StageWError(f"minority adverse SSE signature changed at t={timestep}")
    distribution = _mapping(holdout["accepted_distribution"], "accepted distribution")
    reduction_stats = _mapping(distribution["distance_reduction"], "distance reduction")
    relative_stats = _mapping(
        distribution["relative_distance_reduction"], "relative reduction"
    )
    if float(reduction_stats["min"]) >= 0.0:
        raise StageWError(f"accepted-row adverse tail disappeared at t={timestep}")
    return {
        "timestep": timestep,
        "train": copy.deepcopy(dict(train)),
        "selection_holdout": copy.deepcopy(dict(holdout)),
        "transfer_gaps": {
            "acceptance_rate": float(holdout["acceptance_rate"])
            - float(train["acceptance_rate"]),
            "overall_mse_ratio": overall_ratio - float(train["overall_mse_ratio"]),
            "accepted_row_mse_ratio": accepted_ratio
            - float(train["accepted_row_mse_ratio"]),
            "positive_distance_reduction_rate": float(
                holdout["positive_distance_reduction_rate"]
            )
            - float(train["positive_distance_reduction_rate"]),
            "relative_distance_reduction_mean": float(
                holdout["relative_distance_reduction_mean"]
            )
            - float(train["relative_distance_reduction_mean"]),
            "accepted_baseline_sse_share": holdout_accepted_share
            - train_accepted_share,
        },
        "contract_failure_margins": {
            "overall_mse_ratio_minus_one": overall_ratio - 1.0,
            "accepted_row_mse_ratio_minus_one": accepted_ratio - 1.0,
            "positive_reduction_rate_minus_half": float(
                holdout["positive_distance_reduction_rate"]
            )
            - 0.5,
            "relative_reduction_mean_minus_zero": float(
                holdout["relative_distance_reduction_mean"]
            ),
        },
        "accepted_population_counts": {
            "selected_rows": selected,
            "strictly_improving_rows": positive,
            "non_improving_rows": nonpositive,
            "strictly_improving_rows_are_majority": positive > nonpositive,
            "adverse_row_count_lower_bound": 1,
            "adverse_row_count_upper_bound": nonpositive,
        },
        "sse_identity": {
            "train_accepted_baseline_sse_share": train_accepted_share,
            "holdout_accepted_baseline_sse_share": holdout_accepted_share,
            "holdout_accepted_sse_excess_relative_to_accepted_baseline": (
                accepted_ratio - 1.0
            ),
            "holdout_total_sse_excess_relative_to_total_baseline": (
                overall_ratio - 1.0
            ),
            "unselected_candidate_equals_control": True,
            "minority_non_improving_side_dominates_net_accepted_sse": True,
        },
        "scale_population": {
            "train_histogram": copy.deepcopy(dict(train["selected_scale_histogram"])),
            "holdout_histogram": copy.deepcopy(
                dict(holdout["selected_scale_histogram"])
            ),
            "normalized_total_variation": histogram_total_variation(
                _mapping(train["selected_scale_histogram"], "train histogram"),
                _mapping(holdout["selected_scale_histogram"], "holdout histogram"),
            ),
        },
        "persisted_distribution_signature": {
            "accepted_distance_reduction_min": float(reduction_stats["min"]),
            "accepted_distance_reduction_p05": float(reduction_stats["p05"]),
            "accepted_distance_reduction_median": float(reduction_stats["median"]),
            "accepted_distance_reduction_p95": float(reduction_stats["p95"]),
            "accepted_distance_reduction_max": float(reduction_stats["max"]),
            "accepted_relative_reduction_min": float(relative_stats["min"]),
            "accepted_relative_reduction_p05": float(relative_stats["p05"]),
            "accepted_relative_reduction_median": float(relative_stats["median"]),
            "accepted_relative_reduction_p95": float(relative_stats["p95"]),
            "accepted_relative_reduction_max": float(relative_stats["max"]),
            "row_identifiers_available": False,
        },
        "failure_signature": {
            "mechanism_and_aligned_gate_pass": True,
            "overall_mse_contract_pass": False,
            "accepted_row_mse_contract_pass": False,
            "majority_distance_improvement_pass": True,
            "mean_relative_improvement_pass": True,
            "accepted_row_fidelity_transfer_failure": True,
        },
    }


def build_cross_timestep_diagnostic(
    diagnostics: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if [int(value["timestep"]) for value in diagnostics] != list(LOCKED_TIMESTEPS):
        raise StageWError("diagnostic timestep order changed")
    margin_by_timestep: Dict[str, float] = {}
    for value in diagnostics:
        margins = _mapping(value["contract_failure_margins"], "failure margins")
        margin_by_timestep[str(value["timestep"])] = max(
            float(margins["overall_mse_ratio_minus_one"]),
            float(margins["accepted_row_mse_ratio_minus_one"]),
        )
    closest_timestep = min(
        LOCKED_TIMESTEPS, key=lambda timestep: margin_by_timestep[str(timestep)]
    )
    all_acceptance_decreased = all(
        float(_mapping(item["transfer_gaps"], "transfer gaps")["acceptance_rate"])
        < 0.0
        for item in diagnostics
    )
    all_sse_share_decreased = all(
        float(
            _mapping(item["transfer_gaps"], "transfer gaps")[
                "accepted_baseline_sse_share"
            ]
        )
        < 0.0
        for item in diagnostics
    )
    return {
        "all_locked_timesteps_audited": True,
        "all_timesteps_mechanism_and_gate_clean": True,
        "all_timesteps_fail_same_two_mse_checks": True,
        "all_timesteps_majority_distance_improving": True,
        "all_timesteps_positive_mean_relative_reduction": True,
        "all_timesteps_minority_non_improving_side_dominates_net_accepted_sse": True,
        "all_timesteps_acceptance_decreased_from_objective_train": all_acceptance_decreased,
        "all_timesteps_accepted_baseline_sse_share_decreased_from_objective_train": (
            all_sse_share_decreased
        ),
        "failure_is_cross_timestep_systematic": True,
        "failure_is_not_an_isolated_timestep_or_gate_regression": True,
        "diagnostic_closest_to_threshold_timestep": int(closest_timestep),
        "diagnostic_failure_margin_by_timestep": margin_by_timestep,
        "closest_timestep_selection_authorized": False,
        "aggregate_metrics_override_preregistered_checks": False,
    }


def build_evidence_sufficiency() -> Mapping[str, Any]:
    return {
        "available": {
            "objective_train_locked_aggregate_metrics": True,
            "selection_holdout_aggregate_metrics": True,
            "selection_holdout_scalar_distribution_summaries": True,
            "selection_holdout_selected_scale_histograms": True,
            "candidate_and_direction_identity_hashes": True,
            "process_and_access_counts": True,
        },
        "unavailable": {
            "row_level_control_distance": True,
            "row_level_candidate_distance": True,
            "row_level_squared_error_delta": True,
            "condition_attribution": True,
            "visible_seed_attribution": True,
            "pair_group_attribution": True,
            "future_horizon_attribution": True,
            "segment_index_attribution": True,
            "baseline_error_quantile_membership": True,
            "gate_margin_attribution": True,
            "row_identifier_recovery": True,
        },
        "reason_unavailable": (
            "Stage-V persisted aggregate metrics, scalar summaries, hashes, and "
            "histograms but explicitly did not persist candidate, direction, "
            "holdout-target, selected-scale, or row-level metric tensors."
        ),
        "row_level_attribution_requires_forbidden_holdout_reaccess_or_recomputation": True,
        "aggregate_localisation_complete": True,
        "row_level_causal_localisation_claimed": False,
    }


def build_negative_result_contract() -> Mapping[str, Any]:
    contract: Dict[str, Any] = {
        "phase": PHASE,
        "schema": CONTRACT_SCHEMA,
        "contract_role": "freeze_stagev_negative_result_and_next_research_boundary",
        "stagev_negative_result_final": True,
        "selection_holdout_evaluation_count": 1,
        "selection_holdout_rerun_authorized": False,
        "locked_backbone": LOCKED_BACKBONE,
        "locked_timesteps": list(LOCKED_TIMESTEPS),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "scientific_localisation": {
            "mechanism_and_geometry_transfer": "PASS",
            "accepted_row_fidelity_transfer": "FAIL",
            "cross_timestep_pattern": "SYSTEMATIC",
            "minority_non_improving_sse_dominance": "PROVEN_FROM_AGGREGATES",
            "row_level_cause": "UNIDENTIFIABLE_FROM_PERSISTED_EVIDENCE",
        },
        "forbidden": {
            "rerun_stagev_or_resume1": True,
            "reaccess_selection_holdout": True,
            "regenerate_selection_holdout_candidates": True,
            "recompute_selection_holdout_metrics": True,
            "retune_using_selection_holdout": True,
            "choose_t25_as_near_pass": True,
            "relax_mse_threshold_after_result": True,
            "reopen_backbone_or_timestep_search_on_holdout": True,
            "use_fallback_backbone_or_timestep": True,
            "access_frozen_probe_during_failure_audit": True,
            "run_formal_training_reverse_idm_or_candidate_execution": True,
        },
        "allowed_next_research": {
            "new_hypothesis_must_be_objective_train_only": True,
            "nested_group_oof_required": True,
            "tail_robust_or_risk_sensitive_objective_may_be_studied": True,
            "all_new_hyperparameters_must_be_fixed_without_holdout": True,
            "stagev_metrics_may_define_failure_mode_but_not_numeric_tuning_targets": True,
            "frozen_probe_remains_closed_until_a_separate_preregistered_final_test": True,
        },
        "required_next_path": (
            "DESIGN_OBJECTIVE_TRAIN_ONLY_TAIL_ROBUST_DIRECTION_CALIBRATION_"
            "WITH_NESTED_GROUP_OOF_WHILE_KEEPING_FROZEN_PROBE_CLOSED"
        ),
        **{key: False for key in FALSE_BOUNDARIES},
    }
    contract["contract_payload_sha256"] = sha256_bytes(stable_json_bytes(contract))
    return contract


def prove_frozen_source_semantics(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageWError(f"frozen source changed: {relative}")
    stagev_source = (
        repo
        / "ccda_phase3/phase314b_r258_stagev_locked_selection_holdout_evaluation.py"
    ).read_text(encoding="utf-8")
    required_fragments = (
        "output = control_raw.copy()",
        "output[choose], selected_scale[choose], selected[choose] = candidate[choose], float(scale), True",
        '"candidate_tensor_persisted": False',
        '"selected_scale_tensor_persisted": False',
        '"direction_tensor_persisted": False',
        '"holdout_target_tensor_persisted": False',
        '"worker_output_persisted": False',
    )
    missing = [fragment for fragment in required_fragments if fragment not in stagev_source]
    if missing:
        raise StageWError(f"frozen Stage-V source semantics changed: {missing!r}")
    return {
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "candidate_initialised_as_control": True,
        "only_accepted_rows_are_replaced": True,
        "unselected_candidate_equals_control": True,
        "candidate_tensor_persisted": False,
        "selected_scale_tensor_persisted": False,
        "direction_tensor_persisted": False,
        "holdout_target_tensor_persisted": False,
        "worker_payload_persisted": False,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise StageWError("root is not the canonical repository top level")
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageWError("current branch is not Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", "HEAD^")
    if parent != BASE_STAGEV_RESUME1_EVIDENCE_COMMIT:
        raise StageWError(f"Stage-W implementation parent changed: {parent}")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageWError("Stage-W implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageWError("Stage-W implementation path set changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEV_RESUME1_EVIDENCE_COMMIT) != BASE_EVIDENCE_SUBJECT:
        raise StageWError("Stage-V Resume1 evidence subject changed")
    if commit_name_status(repo, BASE_STAGEV_RESUME1_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_EVIDENCE_PATHS)
    ):
        raise StageWError("Stage-V Resume1 evidence path set changed")
    if _git(repo, "rev-parse", f"{BASE_STAGEV_RESUME1_EVIDENCE_COMMIT}^") != BASE_STAGEV_RESUME1_IMPLEMENTATION_COMMIT:
        raise StageWError("Stage-V Resume1 evidence parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEV_RESUME1_IMPLEMENTATION_COMMIT) != BASE_IMPLEMENTATION_SUBJECT:
        raise StageWError("Stage-V Resume1 implementation subject changed")
    if commit_name_status(repo, BASE_STAGEV_RESUME1_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_IMPLEMENTATION_PATHS)
    ):
        raise StageWError("Stage-V Resume1 implementation path set changed")
    if _git(repo, "rev-parse", f"{BASE_STAGEV_RESUME1_IMPLEMENTATION_COMMIT}^") != EXPECTED_STAGEV_RESUME1_PARENT:
        raise StageWError("Stage-V Resume1 implementation parent changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageWError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageWError("DeformableRavens commit changed")
    if status_paths(repo) or status_paths(submodule):
        raise StageWError("repository or submodule worktree is dirty")
    for status, relative in IMPLEMENTATION_PATHS:
        if status != "A":
            raise StageWError("Stage-W implementation path status changed")
        working = repo / relative
        committed = _git_bytes(repo, "show", f"{head}:{relative}")
        if not working.is_file() or working.read_bytes() != committed:
            raise StageWError(f"Stage-W committed blob mismatch: {relative}")
    source_semantics = prove_frozen_source_semantics(repo)
    exact_files = {
        STAGEU_CONTRACT: EXPECTED_STAGEU_CONTRACT_SHA256,
        STAGEV_RESUME1_SUMMARY: EXPECTED_STAGEV_RESUME1_SUMMARY_SHA256,
        ORIGINAL_STAGEV_BLOCKED_REPORT: EXPECTED_ORIGINAL_STAGEV_BLOCKED_REPORT_SHA256,
    }
    for relative, expected_sha in exact_files.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageWError(f"immutable evidence changed: {relative}")
    for relative in (OUTPUT_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageWError(f"Stage-W output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "base_stagev_resume1_implementation_commit": (
            BASE_STAGEV_RESUME1_IMPLEMENTATION_COMMIT
        ),
        "base_stagev_resume1_evidence_commit": BASE_STAGEV_RESUME1_EVIDENCE_COMMIT,
        "source_semantics": source_semantics,
    }


def validate_output_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("phase") != PHASE or payload.get("schema") != SCHEMA:
        raise StageWError("Stage-W output phase/schema changed")
    exact = {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagew_aggregate_evidence_localizes_failure_to_"
            "accepted_row_fidelity_with_minority_adverse_sse_dominance"
        ),
        "required_next_path": (
            "DESIGN_OBJECTIVE_TRAIN_ONLY_TAIL_ROBUST_DIRECTION_CALIBRATION_"
            "WITH_NESTED_GROUP_OOF_WHILE_KEEPING_FROZEN_PROBE_CLOSED"
        ),
        "primary_failure_locus": "accepted_row_fidelity_transfer_failure",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise StageWError(f"Stage-W output changed: {key}")
    diagnostics = _sequence(payload.get("timestep_diagnostics"), "Stage-W diagnostics")
    if [int(item["timestep"]) for item in diagnostics] != list(LOCKED_TIMESTEPS):
        raise StageWError("Stage-W output timestep order changed")
    cross = _mapping(payload.get("cross_timestep_diagnostic"), "Stage-W cross diagnostic")
    if cross.get("failure_is_cross_timestep_systematic") is not True:
        raise StageWError("Stage-W cross-timestep conclusion changed")
    sufficiency = _mapping(payload.get("evidence_sufficiency"), "Stage-W evidence sufficiency")
    if sufficiency.get("row_level_causal_localisation_claimed") is not False:
        raise StageWError("Stage-W overclaims row-level causality")
    contract = _mapping(payload.get("negative_result_contract"), "Stage-W contract")
    verify_self_hash(
        contract,
        field="contract_payload_sha256",
        label="Stage-W negative-result contract",
    )
    require_false(payload, FALSE_BOUNDARIES, "Stage-W output")
    verify_self_hash(payload, field="audit_result_sha256", label="Stage-W output")
    return payload


def execute_audit(
    *,
    root: Path,
    repository: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_environment_variables()
    repo = Path(root).resolve()
    stageu_contract = load_json(repo / STAGEU_CONTRACT)
    stagev_summary = load_json(repo / STAGEV_RESUME1_SUMMARY)
    stageu = validate_stageu_contract(stageu_contract)
    stagev = validate_stagev_resume1_summary(stagev_summary)
    train_cells = _sequence(stageu["locked_cells"], "validated Stage-U cells")
    holdout_records = _sequence(stagev["records"], "validated Stage-V records")
    diagnostics = [
        build_timestep_diagnostic(train, holdout)
        for train, holdout in zip(train_cells, holdout_records)
    ]
    cross = build_cross_timestep_diagnostic(diagnostics)
    sufficiency = build_evidence_sufficiency()
    contract = build_negative_result_contract()
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagew_aggregate_evidence_localizes_failure_to_"
            "accepted_row_fidelity_with_minority_adverse_sse_dominance"
        ),
        "required_next_path": contract["required_next_path"],
        "primary_failure_locus": "accepted_row_fidelity_transfer_failure",
        "repository": {
            key: copy.deepcopy(value)
            for key, value in repository.items()
            if key != "source_semantics"
        },
        "source_evidence": {
            "stageu_contract_path": STAGEU_CONTRACT,
            "stageu_contract_sha256": EXPECTED_STAGEU_CONTRACT_SHA256,
            "stageu_contract_payload_sha256": (
                EXPECTED_STAGEU_CONTRACT_PAYLOAD_SHA256
            ),
            "stagev_resume1_summary_path": STAGEV_RESUME1_SUMMARY,
            "stagev_resume1_summary_sha256": EXPECTED_STAGEV_RESUME1_SUMMARY_SHA256,
            "stagev_resume1_scientific_result_sha256": (
                EXPECTED_STAGEV_RESUME1_SCIENTIFIC_RESULT_SHA256
            ),
            "original_stagev_blocked_report_path": ORIGINAL_STAGEV_BLOCKED_REPORT,
            "original_stagev_blocked_report_sha256": (
                EXPECTED_ORIGINAL_STAGEV_BLOCKED_REPORT_SHA256
            ),
            "source_semantics": copy.deepcopy(repository["source_semantics"]),
        },
        "audit_scope": {
            "evidence_only": True,
            "json_files_read": [
                STAGEU_CONTRACT,
                STAGEV_RESUME1_SUMMARY,
                ORIGINAL_STAGEV_BLOCKED_REPORT,
            ],
            "gpu_required": False,
            "torch_imported": False,
            "numpy_imported": False,
            "dataset_loaded": False,
            "model_loaded": False,
            "holdout_target_loaded": False,
            "candidate_regenerated": False,
            "metrics_recomputed_from_tensors": False,
        },
        "timestep_diagnostics": diagnostics,
        "cross_timestep_diagnostic": cross,
        "evidence_sufficiency": sufficiency,
        "negative_result_contract": contract,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": (
            EXPECTED_CUMULATIVE_HOLDOUT_EVALUATIONS
        ),
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["audit_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_output_payload(payload)
    return payload


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagew_evidence_audit_execution_failed",
        "required_next_path": (
            "DESIGN_ADD_ONLY_STAGEW_EVIDENCE_AUDIT_CORRECTION_WITHOUT_"
            "HOLDOUT_REACCESS"
        ),
        "primary_failure_locus": "evidence_audit_execution_contract",
        "repository": None
        if repository is None
        else {
            key: copy.deepcopy(value)
            for key, value in repository.items()
            if key != "source_semantics"
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageWError(f"write-once output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
