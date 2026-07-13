#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5.1 Resume2.

Resume2 preserves the original and Resume1 blocked evidence, proves that the
latest block was caused only by treating JSON-object key order as a semantic
objective-matrix contract, and restarts the complete train-only pilot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    BASE_BLOCKED_REPORT_COMMIT,
    BASE_REPORT_COMMIT,
    BASE_RESUME1_BLOCKED_REPORT_COMMIT,
    BASE_RESUME1_CORRECTION_COMMIT,
    CALIBRATED_GEOMETRY_OBJECTIVES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_R25_RECOMMENDATION,
    EXPECTED_R25_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    OBJECTIVE_MATRIX_SCHEMA,
    PAIRED_INVERSION_SCHEMA,
    PHASE,
    assert_canonical_paired_row_contract,
    calibrated_objective_names,
    source_sha256,
)

ORIGINAL_PREFLIGHT = "reports/phase3_14b_r251_preflight_summary.json"
ORIGINAL_BLOCKED_JSON = "reports/phase3_14b_r251_blocked_summary.json"
ORIGINAL_BLOCKED_MD = "reports/phase3_14b_r251_blocked_report.md"
RESUME1_PREFLIGHT = "reports/phase3_14b_r251_resume_preflight_summary.json"
RESUME1_BLOCKED_JSON = "reports/phase3_14b_r251_resume_blocked_summary.json"
RESUME1_BLOCKED_MD = "reports/phase3_14b_r251_resume_blocked_report.md"
RESUME2_PREFLIGHT = "reports/phase3_14b_r251_resume2_preflight_summary.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def git_success(root: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def require_mapping_boundary(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    name: str,
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(
                f"{name} boundary mismatch for {key}: "
                f"{value.get(key)!r} != {expected_value!r}"
            )


def require_unchanged_at_commit(
    root: Path,
    commit: str,
    paths: tuple[str, ...],
    *,
    name: str,
) -> None:
    if not git_success(root, "diff", "--quiet", commit, "--", *paths):
        raise RuntimeError(f"{name} changed after {commit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--resume-generation", type=int, default=2)
    parser.add_argument("--output", default=RESUME2_PREFLIGHT)
    args = parser.parse_args()

    if int(args.resume_generation) != 2:
        raise RuntimeError("this corrected source authorizes Resume2 only")

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=True)
    if repository["branch"] != "Experiment1":
        raise RuntimeError("r2.5.1 Resume2 requires Experiment1")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("immutable cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    for commit, label in (
        (BASE_REPORT_COMMIT, "r2.5 final report"),
        (BASE_BLOCKED_REPORT_COMMIT, "r2.5.1 original blocked report"),
        (BASE_RESUME1_CORRECTION_COMMIT, "r2.5.1 Resume1 correction"),
        (BASE_RESUME1_BLOCKED_REPORT_COMMIT, "r2.5.1 Resume1 blocked report"),
    ):
        if not git_success(root, "merge-base", "--is-ancestor", commit, "HEAD"):
            raise RuntimeError(f"{label} commit is not an ancestor")

    historical_r25 = (
        "ccda_phase3/phase314b_r25_ordered_geometry.py",
        "scripts/phase3_14b_r25_preflight.py",
        "scripts/phase3_14b_r25_run_pilot.py",
        "scripts/phase3_14b_r25_finalize.py",
        "scripts/phase3_14b_r25_run.sh",
        "tests/test_phase314b_r25_ordered_geometry.py",
        "reports/phase3_14b_r25_preflight_summary.json",
        "reports/phase3_14b_r25_pilot_summary.json",
        "reports/phase3_14b_r25_summary.json",
        "reports/phase3_14b_r25_report.md",
    )
    require_unchanged_at_commit(
        root,
        BASE_REPORT_COMMIT,
        historical_r25,
        name="historical r2.5 evidence",
    )

    original_evidence = (
        ORIGINAL_PREFLIGHT,
        ORIGINAL_BLOCKED_JSON,
        ORIGINAL_BLOCKED_MD,
    )
    require_unchanged_at_commit(
        root,
        BASE_BLOCKED_REPORT_COMMIT,
        original_evidence,
        name="original r2.5.1 blocked evidence",
    )

    resume1_evidence = (
        RESUME1_PREFLIGHT,
        RESUME1_BLOCKED_JSON,
        RESUME1_BLOCKED_MD,
    )
    require_unchanged_at_commit(
        root,
        BASE_RESUME1_BLOCKED_REPORT_COMMIT,
        resume1_evidence,
        name="Resume1 blocked evidence",
    )

    r25 = load_json(root / "reports/phase3_14b_r25_summary.json")
    require_mapping_boundary(
        r25,
        {
            "verdict": "PASS",
            "root_cause": EXPECTED_R25_ROOT_CAUSE,
            "train_only_recommendation": EXPECTED_R25_RECOMMENDATION,
            "selected_configuration": None,
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "checkpoint_saved": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        },
        name="r2.5 summary",
    )

    original_preflight = load_json(root / ORIGINAL_PREFLIGHT)
    if original_preflight.get("verdict") != "PASS":
        raise RuntimeError("original r2.5.1 preflight did not pass")

    original_blocked = load_json(root / ORIGINAL_BLOCKED_JSON)
    require_mapping_boundary(
        original_blocked,
        {
            "phase": PHASE,
            "verdict": "BLOCKED",
            "root_cause": "phase314b_r251_execution_failed",
            "stage": "pilot",
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "candidate_eligible": False,
            "selected_configuration": None,
            "checkpoint_saved": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        },
        name="original r2.5.1 blocked report",
    )
    original_exception = str(original_blocked.get("exact_exception_tail", ""))
    for fragment in (
        "paired_reverse_pool_metrics_with_inversion",
        "nearest_index_metrics",
        "future must be finite [N,4,87]",
    ):
        if fragment not in original_exception:
            raise RuntimeError(
                f"original blocked exception contract missing fragment: {fragment}"
            )

    resume1_preflight = load_json(root / RESUME1_PREFLIGHT)
    resume1 = resume1_preflight.get("resume")
    if not isinstance(resume1, Mapping) or resume1.get("generation") != 1:
        raise RuntimeError("Resume1 preflight provenance is invalid")
    if resume1_preflight.get("paired_inversion_contract", {}).get("schema") != PAIRED_INVERSION_SCHEMA:
        raise RuntimeError("Resume1 paired inversion schema mismatch")

    resume1_blocked = load_json(root / RESUME1_BLOCKED_JSON)
    require_mapping_boundary(
        resume1_blocked,
        {
            "phase": PHASE,
            "verdict": "BLOCKED",
            "root_cause": "phase314b_r251_execution_failed",
            "resume_generation": 1,
            "stage": "finalize",
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "candidate_eligible": False,
            "selected_configuration": None,
            "checkpoint_saved": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        },
        name="Resume1 blocked report",
    )
    resume1_exception = str(resume1_blocked.get("exact_exception_tail", ""))
    required_resume1_fragments = (
        "phase3_14b_r251_finalize.py",
        "unique objective matrix/order changed",
    )
    missing = [value for value in required_resume1_fragments if value not in resume1_exception]
    if missing:
        raise RuntimeError(
            "Resume1 blocked exception contract mismatch: " + ", ".join(missing)
        )

    # The Resume1 pilot summary was not committed. A clean worktree means there
    # is no cryptographically retained runtime artifact that can be finalized.
    # Resume2 therefore restarts from pilot_start rather than reconstructing a
    # report from logs or user-provided tables.
    forbidden_existing = (
        "reports/phase3_14b_r251_pilot_summary.json",
        "reports/phase3_14b_r251_summary.json",
        "reports/phase3_14b_r251_report.md",
        "reports/phase3_14b_r251_resume2_blocked_summary.json",
        "reports/phase3_14b_r251_resume2_blocked_report.md",
    )
    existing = [value for value in forbidden_existing if (root / value).exists()]
    if existing:
        raise RuntimeError(
            "unexpected r2.5.1 Resume2 artifacts already exist: " + ", ".join(existing)
        )

    arrays, manifest, _, _, train, fit, calibration, validation = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train rows contain another split")
    paired_rows = select_balanced_paired_rows(arrays, train, len(EXPECTED_PAIRED_ROWS))
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        output_relative = output.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError("preflight output must remain inside repository") from exc
    if output_relative != RESUME2_PREFLIGHT:
        raise RuntimeError(
            f"Resume2 preflight must use {RESUME2_PREFLIGHT}, got {output_relative}"
        )

    objective_order = list(calibrated_objective_names())
    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "Resume1 JSON-object ordering defect verified; Resume2 binds the "
            "objective matrix to an explicit ordered array and exact key membership"
        ),
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "base_blocked_report_commit": BASE_BLOCKED_REPORT_COMMIT,
        "base_resume1_correction_commit": BASE_RESUME1_CORRECTION_COMMIT,
        "base_resume1_blocked_report_commit": BASE_RESUME1_BLOCKED_REPORT_COMMIT,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "resume": {
            "enabled": True,
            "generation": 2,
            "blocked_report_commit": BASE_RESUME1_BLOCKED_REPORT_COMMIT,
            "blocked_report": RESUME1_BLOCKED_JSON,
            "correction": "json_object_order_not_semantic_objective_contract",
            "restart_from": "pilot_start",
            "partial_runtime_results_reused": False,
        },
        "objective_matrix_contract": {
            "schema": OBJECTIVE_MATRIX_SCHEMA,
            "objective_order": objective_order,
            "mapping_membership_must_match": True,
            "json_object_iteration_order_semantic": False,
            "explicit_json_array_order_required": True,
            "sort_keys_serialization_supported": True,
        },
        "paired_inversion_contract": {
            "schema": PAIRED_INVERSION_SCHEMA,
            "canonical_metric_input": "[N,4,87]",
            "vectorized_candidate_query_batch": True,
        },
        "paired_row_contract": paired_contract,
        "r251_source_sha256": source_sha256(root),
        "calibrated_objectives": [
            {
                "name": objective.name,
                "family": objective.family,
                "target_gradient_ratio": objective.target_gradient_ratio,
                "selectable": objective.selectable,
            }
            for objective in CALIBRATED_GEOMETRY_OBJECTIVES
        ],
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "fit_rows": int(len(fit)),
            "calibration_rows": int(len(calibration)),
            "validation_rows_verified_but_not_used": int(len(validation)),
        },
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "selected_configuration": None,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
