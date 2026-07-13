#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.3.1 tiny-control correction."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from ccda_phase3.phase314a_contract import sha256_file, strict_json_load
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import (
    BASE_R23_COMMIT,
    EXPECTED_R23_ROOT_CAUSE,
    PHASE,
    source_sha256,
)


def git_ok(root: Path, *args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r231_preflight_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=True)
    if not git_ok(
        root,
        "merge-base",
        "--is-ancestor",
        BASE_R23_COMMIT,
        "HEAD",
    ):
        raise RuntimeError("r2.3 final-report commit is not an ancestor")

    r23 = strict_json_load(root / "reports/phase3_14b_r23_summary.json")
    if (
        r23.get("verdict"),
        r23.get("root_cause"),
        r23.get("formal_test_read"),
    ) != ("PASS", EXPECTED_R23_ROOT_CAUSE, False):
        raise RuntimeError("unexpected r2.3 final state")

    old_tiny_path = root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
    if not old_tiny_path.is_file():
        raise RuntimeError("missing r2.3 tiny-overfit report")

    forbidden_outputs = (
        root / "reports/phase3_14b_r231_controls_summary.json",
        root / "reports/phase3_14b_r231_summary.json",
        root / "reports/phase3_14b_r231_report.md",
        root / "checkpoints/phase3_14b_r231",
    )
    existing = [str(path) for path in forbidden_outputs if path.exists()]
    if existing:
        raise RuntimeError(f"r2.3.1 outputs already exist: {existing}")

    arrays, manifest, _, _, train, fit, calibration, validation = (
        load_verified_inputs(root)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("r2.3.1 controls require CUDA")

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r231_preflight_passed",
        "repository": repository,
        "r23": {
            "commit": BASE_R23_COMMIT,
            "verdict": r23["verdict"],
            "root_cause": r23["root_cause"],
            "old_tiny_report_sha256": sha256_file(old_tiny_path),
            "old_fixed_control_contract_error": (
                "fixed-noise training was gated with newly sampled "
                "evaluation noise"
            ),
        },
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "fit_rows": int(len(fit)),
            "calibration_rows": int(len(calibration)),
            "validation_rows_present_but_not_used": int(len(validation)),
        },
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "source_sha256": source_sha256(root),
        "formal_test_read": False,
        "validation_read_by_controls": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
