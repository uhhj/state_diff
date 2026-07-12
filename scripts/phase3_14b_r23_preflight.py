#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.3 validation-only diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

from ccda_phase3.phase314b_r23_diagnostics import (
    PHASE,
    load_pilot_runs,
    load_verified_inputs,
    require_repository_state,
    source_sha256,
    verify_checkpoint,
    write_json_once,
)
from ccda_phase3.phase314b_r22_contract import load_self_hashed_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    repository = require_repository_state(root, require_clean=True)
    arrays, manifest, _, _, train, fit, calibration, validation = (
        load_verified_inputs(root)
    )
    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    runs = load_pilot_runs(root)
    checkpoints = {}
    for name, run in runs.items():
        payload = verify_checkpoint(root, run, frozen)
        checkpoints[name] = {
            "path": run["checkpoint"],
            "sha256": run["checkpoint_sha256"],
            "best_epoch": int(payload["best_epoch"]),
            "training_seed": int(payload["training_seed"]),
        }

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r23_preflight_passed",
        "repository": repository,
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "fit_rows": int(len(fit)),
            "calibration_rows": int(len(calibration)),
            "validation_rows": int(len(validation)),
            "train_visible_seed_count": int(
                len(set(arrays["visible_seed"][train].tolist()))
            ),
            "validation_visible_seed_count": int(
                len(set(arrays["visible_seed"][validation].tolist()))
            ),
        },
        "checkpoints": checkpoints,
        "source_sha256": source_sha256(root),
        "formal_test_read": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(
        root / "reports/phase3_14b_r23_preflight_summary.json",
        report,
    )


if __name__ == "__main__":
    main()
