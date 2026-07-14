#!/usr/bin/env python3
"""Write-once preflight for the Resume4 committed-evidence audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume4_reproduction_audit import (
    BASE_BLOCKED_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PHASE3_R2_PASS_COUNT,
    EXPECTED_R253_PILOT_SHA256,
    EXPECTED_R253_SUMMARY_SHA256,
    EXPECTED_RESUME3_BLOCKED_SHA256,
    EXPECTED_RESUME3_PILOT_SHA256,
    EXPECTED_RESUME3_PREFLIGHT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    R253_PILOT_PATH,
    R253_SUMMARY_PATH,
    RESUME3_BLOCKED_PATH,
    RESUME3_PILOT_PATH,
    RESUME3_PREFLIGHT_PATH,
    SUBMODULE_ENVIRONMENT_PATH,
    SUBMODULE_TASK_PATH,
    ReproductionAuditSpec,
    assert_only_allowed_worktree_paths,
    git_output,
    load_json,
    phase3_r2_test_manifest,
    sha256_file,
    source_sha256,
    write_json_once,
)


def require_sha(root: Path, relative: str, expected: str) -> str:
    observed = sha256_file(root / relative)
    if observed != expected:
        raise RuntimeError(f"immutable evidence SHA mismatch: {relative}")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--test-gate-report",
        default="reports/phase3_14b_r254_resume4_test_gate_summary.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume4_preflight_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    test_gate_path = Path(args.test_gate_report)
    if not test_gate_path.is_absolute():
        test_gate_path = root / test_gate_path
    test_gate_path = test_gate_path.resolve()
    test_gate_relative = test_gate_path.relative_to(root).as_posix()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preflight: {output}")

    assert_only_allowed_worktree_paths(root, (test_gate_relative,))
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Resume4 requires Experiment1")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", BASE_BLOCKED_COMMIT, "HEAD"],
            cwd=root,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Resume3 blocked commit is not an ancestor") from exc
    submodule_commit = git_output(
        root / "external/deformable-ravens", "rev-parse", "HEAD"
    )
    if submodule_commit != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if git_output(root / "external/deformable-ravens", "status", "--porcelain"):
        raise RuntimeError("DeformableRavens worktree is not clean")

    if sha256_file(root / "data/phase3_14_cache/phase3_14a_training_cache.npz") != EXPECTED_CACHE_SHA256:
        raise RuntimeError("formal cache SHA changed")
    contract = load_json(root / "reports/phase3_14b_r22_frozen_contract.json")
    if contract.get("artifact_sha256") != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen validity-contract SHA changed")

    evidence_sha = {
        R253_PILOT_PATH: require_sha(root, R253_PILOT_PATH, EXPECTED_R253_PILOT_SHA256),
        R253_SUMMARY_PATH: require_sha(root, R253_SUMMARY_PATH, EXPECTED_R253_SUMMARY_SHA256),
        RESUME3_PREFLIGHT_PATH: require_sha(
            root, RESUME3_PREFLIGHT_PATH, EXPECTED_RESUME3_PREFLIGHT_SHA256
        ),
        RESUME3_PILOT_PATH: require_sha(
            root, RESUME3_PILOT_PATH, EXPECTED_RESUME3_PILOT_SHA256
        ),
        RESUME3_BLOCKED_PATH: require_sha(
            root, RESUME3_BLOCKED_PATH, EXPECTED_RESUME3_BLOCKED_SHA256
        ),
    }
    for relative in (SUBMODULE_TASK_PATH, SUBMODULE_ENVIRONMENT_PATH):
        if not (root / relative).is_file():
            raise RuntimeError(f"required pinned source is missing: {relative}")

    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Resume4 scoped test gate did not pass")
    if test_gate.get("schema") != (
        "phase314b_r254_resume4_scoped_static_test_gate_v1"
    ):
        raise RuntimeError("Resume4 scoped test-gate schema changed")
    if test_gate.get("passed_test_count") != EXPECTED_PHASE3_R2_PASS_COUNT:
        raise RuntimeError("Resume4 scoped test count changed")
    if test_gate.get("test_manifest_sha256") != phase3_r2_test_manifest(root):
        raise RuntimeError("Resume4 scoped test manifest changed after execution")
    if test_gate.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume4 source changed after scoped tests")
    if test_gate.get("full_repository_suite_required") is not False:
        raise RuntimeError("Resume4 must not require full-repository pytest")
    for key in (
        "selection_uses_ignore",
        "selection_uses_k_expression",
        "selection_uses_deselection",
    ):
        if test_gate.get(key) is not False:
            raise RuntimeError(f"forbidden test selection mechanism: {key}")

    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume4",
        "phase_id": PHASE,
        "verdict": "PASS",
        "meaning": "committed-evidence-only reproduction-mismatch preflight",
        "repository": {
            "branch": "Experiment1",
            "head": git_output(root, "rev-parse", "HEAD"),
            "base_blocked_commit": BASE_BLOCKED_COMMIT,
            "submodule_commit": submodule_commit,
            "cache_sha256": EXPECTED_CACHE_SHA256,
            "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        },
        "source_sha256": source_sha256(root),
        "evidence_sha256": evidence_sha,
        "test_gate_report": test_gate_relative,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": {
            "schema": test_gate["schema"],
            "test_file_count": test_gate["test_file_count"],
            "passed_test_count": test_gate["passed_test_count"],
            "test_manifest_sha256": test_gate["test_manifest_sha256"],
            "full_repository_suite_required": False,
            "out_of_scope_collection_baseline": test_gate[
                "out_of_scope_collection_baseline"
            ],
        },
        "spec": ReproductionAuditSpec().__dict__,
        "allowed_inputs": [test_gate_relative] + list(evidence_sha) + [
            SUBMODULE_TASK_PATH,
            SUBMODULE_ENVIRONMENT_PATH,
        ],
        "validation_targets_used": False,
        "formal_test_read": False,
        "retraining_performed": False,
        "reverse_sampling_rerun": False,
        "formal_training": False,
        "checkpoint_saved": False,
        "prediction_tensor_persisted": False,
        "candidate_execution": False,
        "idm": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    write_json_once(output, report)
    print(json.dumps({"verdict": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
