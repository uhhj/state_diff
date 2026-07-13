#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5.1 Resume1.

Resume1 preserves the original blocked evidence and verifies the corrected
batched nearest-index contract before any train-only GPU execution.
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
    CALIBRATED_GEOMETRY_OBJECTIVES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_R25_RECOMMENDATION,
    EXPECTED_R25_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    PAIRED_INVERSION_SCHEMA,
    PHASE,
    assert_canonical_paired_row_contract,
    source_sha256,
)

ORIGINAL_PREFLIGHT = "reports/phase3_14b_r251_preflight_summary.json"
ORIGINAL_BLOCKED_JSON = "reports/phase3_14b_r251_blocked_summary.json"
ORIGINAL_BLOCKED_MD = "reports/phase3_14b_r251_blocked_report.md"
RESUME_PREFLIGHT = "reports/phase3_14b_r251_resume_preflight_summary.json"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--resume-generation", type=int, default=1)
    parser.add_argument("--output", default=RESUME_PREFLIGHT)
    args = parser.parse_args()

    if int(args.resume_generation) != 1:
        raise RuntimeError("this corrected source authorizes Resume1 only")

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=True)
    if repository["branch"] != "Experiment1":
        raise RuntimeError("r2.5.1 Resume1 requires Experiment1")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("immutable cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    for commit, label in (
        (BASE_REPORT_COMMIT, "r2.5 final report"),
        (BASE_BLOCKED_REPORT_COMMIT, "r2.5.1 blocked report"),
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
    if not git_success(root, "diff", "--quiet", BASE_REPORT_COMMIT, "--", *historical_r25):
        raise RuntimeError("historical r2.5 evidence changed after final report")

    blocked_evidence = (
        ORIGINAL_PREFLIGHT,
        ORIGINAL_BLOCKED_JSON,
        ORIGINAL_BLOCKED_MD,
    )
    if not git_success(
        root,
        "diff",
        "--quiet",
        BASE_BLOCKED_REPORT_COMMIT,
        "--",
        *blocked_evidence,
    ):
        raise RuntimeError("original r2.5.1 blocked evidence changed")

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

    blocked = load_json(root / ORIGINAL_BLOCKED_JSON)
    require_mapping_boundary(
        blocked,
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
        name="r2.5.1 blocked report",
    )
    exception = str(blocked.get("exact_exception_tail", ""))
    required_fragments = (
        "paired_reverse_pool_metrics_with_inversion",
        "nearest_index_metrics",
        "future must be finite [N,4,87]",
    )
    missing_fragments = [value for value in required_fragments if value not in exception]
    if missing_fragments:
        raise RuntimeError(
            "original blocked exception contract mismatch: "
            + ", ".join(missing_fragments)
        )

    forbidden_existing = (
        "reports/phase3_14b_r251_pilot_summary.json",
        "reports/phase3_14b_r251_summary.json",
        "reports/phase3_14b_r251_report.md",
        "reports/phase3_14b_r251_resume_blocked_summary.json",
        "reports/phase3_14b_r251_resume_blocked_report.md",
    )
    existing = [value for value in forbidden_existing if (root / value).exists()]
    if existing:
        raise RuntimeError("unexpected r2.5.1 Resume1 artifacts already exist: " + ", ".join(existing))

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
    if output_relative != RESUME_PREFLIGHT:
        raise RuntimeError(
            f"Resume1 preflight must use {RESUME_PREFLIGHT}, got {output_relative}"
        )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "r2.5.1 singleton nearest-index rank defect verified and corrected; "
            "train-only boundaries preserved"
        ),
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "base_blocked_report_commit": BASE_BLOCKED_REPORT_COMMIT,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "resume": {
            "enabled": True,
            "generation": 1,
            "blocked_report_commit": BASE_BLOCKED_REPORT_COMMIT,
            "blocked_report": ORIGINAL_BLOCKED_JSON,
            "correction": "paired_reverse_singleton_batch_contract",
            "restart_from": "pilot_start",
            "partial_runtime_results_reused": False,
        },
        "paired_inversion_contract": {
            "schema": PAIRED_INVERSION_SCHEMA,
            "canonical_metric_input": "[N,4,87]",
            "singleton_input_forbidden": True,
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
