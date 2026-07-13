#!/usr/bin/env python3
"""Preflight for Phase3.14b-r2.4.1 multirow/source-batching audit."""
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
from ccda_phase3.phase314b_r241_multirow import (
    BASE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R23_TINY_SHA256,
    EXPECTED_R24_MODULE_SHA256,
    EXPECTED_R24_ROOT_CAUSE,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    assert_only_allowed_worktree_paths,
    dependency_sha256,
    git_output,
    sha256_file,
    source_sha256,
)


def require_unchanged_from_base(
    root: Path,
    paths,
) -> None:
    for path in paths:
        result = subprocess.run(
            ["git", "diff", "--quiet", BASE_COMMIT, "HEAD", "--", path],
            cwd=root,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"historical r2.4 file changed after {BASE_COMMIT}: {path}"
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
        raise RuntimeError("required r2.4 report commit is not an ancestor")
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    allowed = ("reports/phase3_14b_r241_preflight_summary.json",)
    assert_only_allowed_worktree_paths(root, allowed)

    require_unchanged_from_base(
        root,
        (
            "reports/phase3_14b_r24_pilot_summary.json",
            "reports/phase3_14b_r24_summary.json",
            "reports/phase3_14b_r24_report.md",
            "ccda_phase3/phase314b_r24_noisy_skip.py",
            "scripts/phase3_14b_r24_run_pilot.py",
        ),
    )
    r24 = strict_json_load(root / "reports/phase3_14b_r24_summary.json")
    if r24.get("root_cause") != EXPECTED_R24_ROOT_CAUSE:
        raise RuntimeError("r2.4 root cause mismatch")
    if r24.get("formal_test_read") is not False:
        raise RuntimeError("formal test was accessed")
    if r24.get("selected_configuration") not in (None, "None"):
        raise RuntimeError("r2.4 selected a formal configuration")
    if r24.get("train_only_recommendation") not in (None, "None"):
        raise RuntimeError("r2.4 unexpectedly produced a recommendation")

    r24_module_sha = sha256_file(
        root / "ccda_phase3/phase314b_r24_noisy_skip.py"
    )
    if r24_module_sha != EXPECTED_R24_MODULE_SHA256:
        raise RuntimeError("r2.4 module SHA mismatch")
    historical_sha = sha256_file(
        root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
    )
    if historical_sha != EXPECTED_R23_TINY_SHA256:
        raise RuntimeError("historical r2.3 tiny report changed")

    forbidden = (
        root / "reports/phase3_14b_r241_audit_summary.json",
        root / "reports/phase3_14b_r241_summary.json",
        root / "reports/phase3_14b_r241_report.md",
        root / "checkpoints/phase3_14b_r241",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "r2.4.1 output already exists: " + ", ".join(existing)
        )

    arrays, manifest, _, _, train, _, _, validation = load_verified_inputs(
        root
    )
    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r241_preflight_passed",
        "repository": repository,
        "required_base_commit": BASE_COMMIT,
        "head": git_output(root, "rev-parse", "HEAD"),
        "submodule_commit": repository["submodule_commit"],
        "cache_sha256": repository["cache_sha256"],
        "frozen_contract_sha256": repository["frozen_contract_sha256"],
        "historical_r23_tiny_sha256": historical_sha,
        "r24_module_sha256": r24_module_sha,
        "r24_root_cause": r24["root_cause"],
        "dataset": {
            "schema_version": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "total_cache_rows": int(len(arrays["split_name"])),
        },
        "scope": {
            "train_only_audit": True,
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "candidate_selected": False,
            "checkpoint_saved": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        },
        "source_sha256": source_sha256(root),
        "dependency_sha256": dependency_sha256(root),
    }
    output = root / "reports/phase3_14b_r241_preflight_summary.json"
    write_json_once(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
