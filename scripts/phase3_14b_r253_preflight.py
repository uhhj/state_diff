#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5.3 gate-separation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    assert_canonical_paired_row_contract,
)
from ccda_phase3.phase314b_r253_gate_separation import (
    BASE_REPORT_COMMIT,
    DEPENDENCY_PATHS,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_R252_MECHANISMS,
    EXPECTED_R252_RECOMMENDATION,
    EXPECTED_R252_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    GATE_SEPARATION_SCHEMA,
    GateSeparationSpec,
    HISTORICAL_REPRODUCTION_SCHEMA,
    PHASE,
    RECONSTRUCTION_DECOMPOSITION_SCHEMA,
    SYNTHETIC_CONTROL_SCHEMA,
    assert_only_allowed_worktree_paths,
    dependency_sha256,
    git_output,
    source_sha256,
)

HISTORICAL_PATHS = (
    "ccda_phase3/phase314b_r252_transport_attribution.py",
    "scripts/phase3_14b_r252_preflight.py",
    "scripts/phase3_14b_r252_run_pilot.py",
    "scripts/phase3_14b_r252_finalize.py",
    "scripts/phase3_14b_r252_run.sh",
    "tests/test_phase314b_r252_transport_attribution.py",
    "reports/phase3_14b_r252_preflight_summary.json",
    "reports/phase3_14b_r252_pilot_summary.json",
    "reports/phase3_14b_r252_summary.json",
    "reports/phase3_14b_r252_report.md",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def assert_historical_paths_unchanged(root: Path, paths: Sequence[str]) -> None:
    for path in paths:
        try:
            git_output(root, "diff", "--quiet", BASE_REPORT_COMMIT, "--", path)
        except Exception as exc:
            raise RuntimeError(
                f"historical r2.5.2 path changed after base report: {path}"
            ) from exc


def historical_r252_contract(summary: Mapping[str, Any]) -> Dict[str, Any]:
    if summary.get("verdict") != "PASS":
        raise RuntimeError("r2.5.2 summary verdict is not PASS")
    if summary.get("root_cause") != EXPECTED_R252_ROOT_CAUSE:
        raise RuntimeError("r2.5.2 root cause changed")
    mechanisms = summary.get("supported_mechanisms")
    if not isinstance(mechanisms, list):
        raise RuntimeError("r2.5.2 supported mechanisms are missing")
    if set(mechanisms) != set(EXPECTED_R252_MECHANISMS):
        raise RuntimeError("r2.5.2 supported mechanisms changed")
    if summary.get("train_only_recommendation") is not EXPECTED_R252_RECOMMENDATION:
        raise RuntimeError("r2.5.2 recommendation changed")
    if summary.get("selected_configuration") is not None:
        raise RuntimeError("r2.5.2 selected a configuration unexpectedly")
    variants = summary.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("r2.5.2 diagnostic variant matrix changed")

    expected: Dict[str, Any] = {}
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        value = variants[name]
        if not isinstance(value, Mapping):
            raise RuntimeError(f"r2.5.2 variant payload missing: {name}")
        if not bool(value.get("training_completed")):
            raise RuntimeError(f"r2.5.2 training incomplete: {name}")
        if bool(value.get("composite_one_step_pass")):
            raise RuntimeError(f"r2.5.2 composite one-step unexpectedly passed: {name}")
        if not bool(value.get("branch_audit_pass")):
            raise RuntimeError(f"r2.5.2 branch audit did not pass: {name}")
        endpoint = value.get("endpoint", {})
        required_endpoint = (
            "sample_validity_rate",
            "valid_query_rate",
            "best_ordered_rmse_mean",
            "nearest_inversion_p95",
            "both_branch_support_rate",
        )
        missing = [key for key in required_endpoint if endpoint.get(key) is None]
        if missing:
            raise RuntimeError(f"r2.5.2 endpoint fields missing for {name}: {missing}")
        expected[name] = {
            "reconstruction_aggregate_gate_pass": bool(
                value.get("reconstruction_aggregate_gate_pass", False)
            ),
            "reconstruction_all_source_gate_pass": bool(
                value.get("reconstruction_all_source_gate_pass", False)
            ),
            "denoising_result_pass": bool(value.get("denoising_result_pass", False)),
            "branch_audit_pass": bool(value.get("branch_audit_pass", False)),
            "composite_one_step_pass": bool(
                value.get("composite_one_step_pass", False)
            ),
            "legacy_own_target_closer_fraction": float(
                value.get("legacy_own_target_closer_fraction", 0.0)
            ),
            "legacy_separation_ratio_p50": float(
                value.get("legacy_separation_ratio_p50", 0.0)
            ),
            "legacy_delta_cosine_p50": float(
                value.get("legacy_delta_cosine_p50", 0.0)
            ),
            "endpoint": {key: float(endpoint[key]) for key in required_endpoint},
        }
    shared_prior = summary.get("shared_prior", {})
    prior_sha = shared_prior.get("prior_state_sha256")
    if not isinstance(prior_sha, str) or len(prior_sha) != 64:
        raise RuntimeError("r2.5.2 shared-prior SHA is missing")
    return {
        "root_cause": summary["root_cause"],
        "supported_mechanisms": list(mechanisms),
        "train_only_recommendation": summary["train_only_recommendation"],
        "selected_configuration": summary["selected_configuration"],
        "shared_prior_sha256": prior_sha,
        "variants": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r253_preflight_summary.json",
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
        raise RuntimeError("r2.5.3 requires Experiment1 branch")
    try:
        git_output(root, "merge-base", "--is-ancestor", BASE_REPORT_COMMIT, "HEAD")
    except Exception as exc:
        raise RuntimeError("r2.5.2 report commit is not an ancestor") from exc
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    assert_historical_paths_unchanged(root, HISTORICAL_PATHS)
    r252_summary = load_json(root / "reports/phase3_14b_r252_summary.json")
    historical = historical_r252_contract(r252_summary)
    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    prediction_reference = frozen.get("prediction_reference_contract", {})
    inversion_threshold = float(
        prediction_reference.get("nearest_index_inversion_threshold", -1.0)
    )
    expected_inversion = GateSeparationSpec().topology_inversion_p95_max
    if abs(inversion_threshold - expected_inversion) > 1.0e-12:
        raise RuntimeError("frozen nearest-index inversion threshold changed")

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
        "meaning": (
            "train-only exact-reconstruction / branch-transport gate-separation "
            "preflight"
        ),
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "source_sha256": source_sha256(root),
        "dependency_sha256": dependency_sha256(root),
        "historical_paths_unchanged": list(HISTORICAL_PATHS),
        "r252_contract": historical,
        "diagnostic_objective_order": list(DIAGNOSTIC_OBJECTIVE_NAMES),
        "frozen_prediction_reference_contract": {
            "reference": prediction_reference.get("reference"),
            "nearest_index_inversion_threshold": inversion_threshold,
        },
        "schemas": {
            "gate_separation": GATE_SEPARATION_SCHEMA,
            "reconstruction_decomposition": RECONSTRUCTION_DECOMPOSITION_SCHEMA,
            "synthetic_controls": SYNTHETIC_CONTROL_SCHEMA,
            "historical_reproduction": HISTORICAL_REPRODUCTION_SCHEMA,
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
