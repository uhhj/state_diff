#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.4 train-only noisy-skip pilot."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ccda_phase3.phase314a_contract import strict_json_load
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r24_noisy_skip import (
    BASE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R232_ROOT_CAUSE,
    EXPECTED_R23_TINY_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    assert_only_allowed_worktree_paths,
    dependency_sha256,
    git_output,
    sha256_file,
    source_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    repository = require_repository_state(root, require_clean=False)
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("wrong branch")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
        cwd=root,
    ).returncode
    if ancestor != 0:
        raise RuntimeError("required r2.3.2 report commit is not an ancestor")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    allowed = ("reports/phase3_14b_r24_preflight_summary.json",)
    assert_only_allowed_worktree_paths(root, allowed)

    r232 = strict_json_load(root / "reports/phase3_14b_r232_summary.json")
    if r232.get("root_cause") != EXPECTED_R232_ROOT_CAUSE:
        raise RuntimeError("r2.3.2 root cause mismatch")
    if r232.get("formal_test_read") is not False:
        raise RuntimeError("formal test was accessed")
    if r232.get("selected_configuration") not in (None, "None"):
        raise RuntimeError("r2.3.2 unexpectedly selected a configuration")

    historical = root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
    historical_sha = sha256_file(historical)
    if historical_sha != EXPECTED_R23_TINY_SHA256:
        raise RuntimeError("historical r2.3 tiny report changed")

    forbidden = (
        root / "reports/phase3_14b_r24_pilot_summary.json",
        root / "reports/phase3_14b_r24_summary.json",
        root / "reports/phase3_14b_r24_report.md",
        root / "checkpoints/phase3_14b_r24",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError("r2.4 output already exists: " + ", ".join(existing))

    arrays, manifest, _, _, train, _, _, validation = load_verified_inputs(root)
    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r24_preflight_passed",
        "repository": repository,
        "required_base_commit": BASE_COMMIT,
        "head": git_output(root, "rev-parse", "HEAD"),
        "submodule_commit": repository["submodule_commit"],
        "cache_sha256": repository["cache_sha256"],
        "frozen_contract_sha256": repository["frozen_contract_sha256"],
        "historical_r23_tiny_sha256": historical_sha,
        "r232_root_cause": r232["root_cause"],
        "dataset": {
            "schema_version": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "total_cache_rows": int(len(arrays["split_name"])),
        },
        "scope": {
            "train_only_pilot": True,
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "formal_candidate_selected": False,
            "checkpoint_saved": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        },
        "source_sha256": source_sha256(root),
        "dependency_sha256": dependency_sha256(root),
    }
    output = root / "reports/phase3_14b_r24_preflight_summary.json"
    write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
