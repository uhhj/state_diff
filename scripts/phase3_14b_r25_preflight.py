#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.5 train-only geometry repair."""

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
from ccda_phase3.phase314b_r25_ordered_geometry import (
    BASE_REPORT_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R242_RECOMMENDATION,
    EXPECTED_R242_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    GEOMETRY_OBJECTIVES,
    PHASE,
    source_sha256,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


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
        default="reports/phase3_14b_r25_preflight_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=True)
    if repository["branch"] != "Experiment1":
        raise RuntimeError("r2.5 requires Experiment1")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("immutable cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    if not git_success(root, "merge-base", "--is-ancestor", BASE_REPORT_COMMIT, "HEAD"):
        raise RuntimeError("r2.4.2 final report commit is not an ancestor")

    historical_paths = (
        "ccda_phase3/phase314b_r242_frozen_prior.py",
        "scripts/phase3_14b_r242_preflight.py",
        "scripts/phase3_14b_r242_run_pilot.py",
        "scripts/phase3_14b_r242_finalize.py",
        "scripts/phase3_14b_r242_run.sh",
        "tests/test_phase314b_r242_frozen_prior.py",
        "reports/phase3_14b_r242_pilot_summary.json",
        "reports/phase3_14b_r242_summary.json",
        "reports/phase3_14b_r242_report.md",
    )
    if not git_success(root, "diff", "--quiet", BASE_REPORT_COMMIT, "--", *historical_paths):
        raise RuntimeError("historical r2.4.2 evidence changed after final report")

    r242 = load_json(root / "reports/phase3_14b_r242_summary.json")
    expected = {
        "verdict": "PASS",
        "root_cause": EXPECTED_R242_ROOT_CAUSE,
        "train_only_recommendation": EXPECTED_R242_RECOMMENDATION,
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
        if r242.get(key) != value:
            raise RuntimeError(f"r2.4.2 summary mismatch for {key}: {r242.get(key)!r}")
    unique = r242.get("unique_free_variants", {})
    paired = r242.get("paired_low_mid_variants", {})
    recommended_unique = unique.get(EXPECTED_R242_RECOMMENDATION, {})
    recommended_paired = paired.get(EXPECTED_R242_RECOMMENDATION, {})
    if not bool(recommended_unique.get("pass")):
        raise RuntimeError("recommended r2.4.2 unique-free control did not pass")
    if not bool(recommended_paired.get("combined_pass")):
        raise RuntimeError("recommended r2.4.2 paired low/mid control did not pass")
    if float(recommended_unique.get("prior_drift_ratio", float("inf"))) != 1.0:
        raise RuntimeError("recommended r2.4.2 prior was not strictly frozen")

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
        "meaning": "r2.4.2 frozen-p512/r512 evidence and train-only r2.5 boundary verified",
        "repository": repository,
        "base_report_commit": BASE_REPORT_COMMIT,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "r242_root_cause": EXPECTED_R242_ROOT_CAUSE,
        "r242_train_only_recommendation": EXPECTED_R242_RECOMMENDATION,
        "r242_recommended_unique": recommended_unique,
        "r242_recommended_paired": recommended_paired,
        "r25_source_sha256": source_sha256(root),
        "geometry_objectives": [objective.name for objective in GEOMETRY_OBJECTIVES],
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
