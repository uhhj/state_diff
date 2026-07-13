#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5.2 paired transport attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    assert_canonical_paired_row_contract,
)
from ccda_phase3.phase314b_r252_transport_attribution import (
    BASE_REPORT_COMMIT,
    DEPENDENCY_PATHS,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_R251_RECOMMENDATION,
    EXPECTED_R251_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    ATTRIBUTION_SCHEMA,
    TRAJECTORY_SCHEMA,
    PER_TIMESTEP_GRADIENT_SCHEMA,
    assert_only_allowed_worktree_paths,
    dependency_sha256,
    diagnostic_objectives,
    git_output,
    source_sha256,
)

HISTORICAL_PATHS = (
    "ccda_phase3/phase314b_r251_gradient_calibration.py",
    "scripts/phase3_14b_r251_preflight.py",
    "scripts/phase3_14b_r251_run_pilot.py",
    "scripts/phase3_14b_r251_finalize.py",
    "scripts/phase3_14b_r251_run.sh",
    "tests/test_phase314b_r251_gradient_calibration.py",
    "reports/phase3_14b_r251_pilot_summary.json",
    "reports/phase3_14b_r251_summary.json",
    "reports/phase3_14b_r251_report.md",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def assert_historical_paths_unchanged(root: Path, paths: Sequence[str]) -> None:
    for path in paths:
        code = 0
        try:
            git_output(root, "diff", "--quiet", BASE_REPORT_COMMIT, "--", path)
        except Exception as exc:
            code = getattr(exc, "returncode", 1)
        if code != 0:
            raise RuntimeError(f"historical r2.5.1 path changed after base report: {path}")


def historical_endpoint_contract(summary: Mapping[str, Any]) -> Dict[str, Any]:
    if summary.get("verdict") != "PASS":
        raise RuntimeError("r2.5.1 summary verdict is not PASS")
    if summary.get("root_cause") != EXPECTED_R251_ROOT_CAUSE:
        raise RuntimeError("r2.5.1 root cause changed")
    if summary.get("train_only_recommendation") is not EXPECTED_R251_RECOMMENDATION:
        raise RuntimeError("r2.5.1 recommendation changed")
    if summary.get("selected_configuration") is not None:
        raise RuntimeError("r2.5.1 unexpectedly selected a configuration")
    unique = summary.get("unique_free_variants")
    paired = summary.get("paired_variants")
    if not isinstance(unique, Mapping) or not isinstance(paired, Mapping):
        raise RuntimeError("r2.5.1 compact variant mappings are missing")
    expected: Dict[str, Any] = {}
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        if name not in unique or name not in paired:
            raise RuntimeError(f"r2.5.1 diagnostic objective missing: {name}")
        unique_value = unique[name]
        paired_value = paired[name]
        if not bool(unique_value.get("pass")):
            raise RuntimeError(f"diagnostic objective did not pass unique gate: {name}")
        if bool(paired_value.get("one_step_pass")):
            raise RuntimeError(f"diagnostic objective unexpectedly passed paired one-step: {name}")
        required = (
            "reverse_sample_validity_rate",
            "reverse_query_has_valid_candidate_rate",
            "reverse_best_ordered_rmse_mean",
            "reverse_nearest_inversion_p95",
            "both_branch_support_rate",
        )
        missing = [field for field in required if paired_value.get(field) is None]
        if missing:
            raise RuntimeError(f"r2.5.1 reverse fields missing for {name}: {missing}")
        if float(paired_value["reverse_query_has_valid_candidate_rate"]) != 1.0:
            raise RuntimeError(f"diagnostic objective valid-query contract changed: {name}")
        expected[name] = {
            field: float(paired_value[field])
            for field in required
        }
        expected[name]["one_step_pass"] = False
        expected[name]["one_step_branch_pass"] = bool(
            paired_value.get("one_step_branch_pass", False)
        )
        expected[name]["comparison_to_control_pass"] = bool(
            paired_value.get("comparison_to_control_pass", False)
        )
    if not bool(expected["ordered_mean_raw_g100"]["comparison_to_control_pass"]):
        raise RuntimeError("mean g100 historical reverse comparison changed")
    if not bool(expected["ordered_cvar_contract_g010"]["comparison_to_control_pass"]):
        raise RuntimeError("contract g010 historical reverse comparison changed")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r252_preflight_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output_relative = output.relative_to(root).as_posix()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preflight: {output}")

    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, ())
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("r2.5.2 requires Experiment1 branch")
    try:
        git_output(root, "merge-base", "--is-ancestor", BASE_REPORT_COMMIT, "HEAD")
    except Exception as exc:
        raise RuntimeError("r2.5.1 report commit is not an ancestor") from exc
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    assert_historical_paths_unchanged(root, HISTORICAL_PATHS)

    summary_path = root / "reports/phase3_14b_r251_summary.json"
    r251_summary = load_json(summary_path)
    expected_endpoints = historical_endpoint_contract(r251_summary)
    diagnostic = diagnostic_objectives()

    arrays, manifest, _, _, train, _, _, validation = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train indices contain non-train rows")
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only paired one-step and reverse-trajectory attribution preflight",
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "source_sha256": source_sha256(root),
        "dependency_sha256": dependency_sha256(root),
        "historical_paths_unchanged": list(HISTORICAL_PATHS),
        "r251_contract": {
            "root_cause": r251_summary["root_cause"],
            "train_only_recommendation": r251_summary["train_only_recommendation"],
            "selected_configuration": r251_summary["selected_configuration"],
            "expected_endpoints": expected_endpoints,
        },
        "diagnostic_objective_order": [value.name for value in diagnostic],
        "schemas": {
            "attribution": ATTRIBUTION_SCHEMA,
            "trajectory": TRAJECTORY_SCHEMA,
            "per_timestep_gradient": PER_TIMESTEP_GRADIENT_SCHEMA,
        },
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_row_count": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "paired_rows": paired_rows.tolist(),
            "paired_row_contract": paired_contract,
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
    print(json.dumps({"verdict": "PASS", "output": output_relative}, sort_keys=True))


if __name__ == "__main__":
    main()
