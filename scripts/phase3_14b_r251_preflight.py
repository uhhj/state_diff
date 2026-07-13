#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5.1 geometry-gradient calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    BASE_REPORT_COMMIT,
    CALIBRATED_GEOMETRY_OBJECTIVES,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R25_RECOMMENDATION,
    EXPECTED_R25_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    source_sha256,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r251_preflight_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=True)
    if repository["branch"] != "Experiment1":
        raise RuntimeError("r2.5.1 requires Experiment1")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("immutable cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    if not git_success(root, "merge-base", "--is-ancestor", BASE_REPORT_COMMIT, "HEAD"):
        raise RuntimeError("r2.5 final report commit is not an ancestor")

    historical_paths = (
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
    if not git_success(root, "diff", "--quiet", BASE_REPORT_COMMIT, "--", *historical_paths):
        raise RuntimeError("historical r2.5 evidence changed after final report")

    r25 = load_json(root / "reports/phase3_14b_r25_summary.json")
    expected = {
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
    }
    for key, value in expected.items():
        if r25.get(key) != value:
            raise RuntimeError(f"r2.5 summary mismatch for {key}: {r25.get(key)!r}")
    active_names = (
        "ordered_mean_raw",
        "ordered_cvar_raw",
        "ordered_cvar_contract",
    )
    active = r25.get("unique_free_variants", {})
    ratios = {}
    for name in active_names:
        value = active.get(name, {})
        ratio = float(value.get("geometry_to_v_gradient_ratio_median", 0.0))
        if bool(value.get("geometry_gradient_gate")) or ratio <= 5.0:
            raise RuntimeError(f"r2.5 gradient-scaling evidence missing for {name}")
        ratios[name] = ratio
    paired_control = r25.get("paired_variants", {}).get("v_only_frozen_control", {})
    if "reverse_nearest_inversion_p95" in paired_control:
        raise RuntimeError("r2.5 summary unexpectedly already contains paired inversion")

    arrays, manifest, _, _, train, fit, calibration, validation = load_verified_inputs(root)
    split = arrays["split_name"].astype(str)
    if not (split[train] == "train").all():
        raise RuntimeError("train rows contain another split")
    if not (split[validation] == "val").all():
        raise RuntimeError("validation index contract mismatch")

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("preflight output must remain inside repository") from exc

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "r2.5 gradient-scaling failure and train-only boundaries verified",
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "r25_root_cause": EXPECTED_R25_ROOT_CAUSE,
        "r25_train_only_recommendation": EXPECTED_R25_RECOMMENDATION,
        "r25_active_gradient_ratios": ratios,
        "r25_paired_nearest_inversion_instrumentation_gap": True,
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
