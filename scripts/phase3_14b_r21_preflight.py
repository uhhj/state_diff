#!/usr/bin/env python3
"""Preflight Phase3.14b-r2.1 validity and ordered-geometry audit."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
)
from ccda_phase3.phase314b_contract import CACHE_SHA256
from ccda_phase3.phase314b_r21_contract import (
    BASE_MAIN_COMMIT,
    EXPECTED_RUN_COUNT,
    SUBMODULE_COMMIT,
    fixed_validation_rows,
    load_r2_inputs,
    require_base_reports,
    source_sha256,
    train_indices,
    train_visible_seed_partition,
    verify_r2_checkpoints,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r21_preflight_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r21_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    reports = require_base_reports(root)
    arrays, manifest, x_raw, x_std = load_r2_inputs(root)
    train = train_indices(arrays)
    fit_rows, calibration_rows = train_visible_seed_partition(
        arrays,
        train,
    )
    validation = fixed_validation_rows(arrays)

    checkpoint_verification = verify_r2_checkpoints(
        root,
        reports["training"],
    )
    if checkpoint_verification["count"] != EXPECTED_RUN_COUNT:
        raise RuntimeError("unexpected checkpoint count")

    # Verify the checkpoint-bound source snapshot for every run.
    bound_sources = {}
    for run in reports["training"]["runs"]:
        checkpoint_path = root / str(run["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("phase") != "phase3_14b_r2":
            raise RuntimeError("non-r2 checkpoint in training summary")
        if checkpoint.get("cache_sha256") != CACHE_SHA256:
            raise RuntimeError("checkpoint cache SHA256 mismatch")
        for relative, expected in checkpoint["source_sha256"].items():
            actual = sha256_file(root / relative)
            if actual != expected:
                raise RuntimeError(
                    f"r2 checkpoint-bound source changed: {relative}"
                )
            bound_sources[relative] = actual

    main_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=str(root),
        text=True,
    ).strip()
    if main_status:
        raise RuntimeError(
            "tracked main worktree must be clean before r2.1 preflight"
        )
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_MAIN_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
    )
    if ancestry.returncode != 0:
        raise RuntimeError("HEAD is not a descendant of Phase3.14b-r2")

    submodule_root = root / "external/deformable-ravens"
    submodule_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(submodule_root),
        text=True,
    ).strip()
    submodule_status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=str(submodule_root),
        text=True,
    ).strip()
    if submodule_head != SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit changed")
    if submodule_status:
        raise RuntimeError("submodule worktree is not clean")

    forbidden_test_artifacts = [
        root / "reports/phase3_14b_r2_test_summary.json",
        root
        / "data/phase3_14b_r2_eval/selected_test_sample_pool.npz",
    ]
    present = [
        str(path.relative_to(root))
        for path in forbidden_test_artifacts
        if path.exists()
    ]
    if present:
        raise RuntimeError(
            f"formal r2 test artifacts unexpectedly exist: {present}"
        )

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r21_preflight_supported",
        "cache_sha256": CACHE_SHA256,
        "cache_manifest_sha256": manifest["cache_sha256"],
        "train_rows": int(train.size),
        "fit_rows": int(fit_rows.size),
        "calibration_rows": int(calibration_rows.size),
        "validation_rows": int(validation.size),
        "checkpoint_verification": checkpoint_verification,
        "checkpoint_bound_source_sha256": bound_sources,
        "r21_source_sha256": source_sha256(root),
        "main_tracked_worktree_clean_before_report": True,
        "submodule_commit": submodule_head,
        "submodule_clean": True,
        "formal_test_artifacts_present": False,
        "training": False,
        "test_read": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r2.1 Preflight",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314b_r21_preflight_supported`",
        f"- Cache SHA256: `{CACHE_SHA256}`",
        f"- R2 checkpoints verified: `{checkpoint_verification['count']}`",
        f"- Fit rows: `{fit_rows.size}`",
        f"- Calibration rows: `{calibration_rows.size}`",
        f"- Validation rows: `{validation.size}`",
        "- Formal test artifacts present: `False`",
        "- No training, test read, IDM, execution, Phase4, or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
