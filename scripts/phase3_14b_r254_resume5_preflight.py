#!/usr/bin/env python3
"""Write-once preflight for Resume5 isolated same-device retraining."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    BASE_EVIDENCE_COMMIT,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_RESUME3_PILOT_SHA256,
    EXPECTED_RESUME4_AUDIT_SHA256,
    EXPECTED_RESUME4_MARKDOWN_SHA256,
    EXPECTED_RESUME4_PREFLIGHT_SHA256,
    EXPECTED_RESUME4_SUMMARY_SHA256,
    EXPECTED_RESUME4_TEST_GATE_SHA256,
    EXPECTED_SCOPED_TEST_COUNT,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    RESUME3_PILOT_PATH,
    RESUME4_AUDIT_PATH,
    RESUME4_MARKDOWN_PATH,
    RESUME4_PREFLIGHT_PATH,
    RESUME4_SUMMARY_PATH,
    RESUME4_TEST_GATE_PATH,
    Resume5Spec,
    assert_only_allowed_worktree_paths,
    git_output,
    load_json,
    sha256_file,
    source_sha256,
    write_json_once,
)


def require_sha(root: Path, relative: str, expected: str) -> str:
    observed = sha256_file(root / relative)
    if observed != expected:
        raise RuntimeError(f"immutable evidence SHA changed: {relative}")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--test-gate-report",
        default="reports/phase3_14b_r254_resume5_test_gate_summary.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume5_preflight_summary.json",
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
        raise RuntimeError(f"refusing to overwrite Resume5 preflight: {output}")
    assert_only_allowed_worktree_paths(root, (test_gate_relative,))

    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Resume5 requires Experiment1")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_EVIDENCE_COMMIT, "HEAD"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("Resume4 evidence commit is not an ancestor")
    submodule_root = root / "external/deformable-ravens"
    if git_output(submodule_root, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if git_output(submodule_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("DeformableRavens worktree is dirty")
    if sha256_file(root / "data/phase3_14_cache/phase3_14a_training_cache.npz") != EXPECTED_CACHE_SHA256:
        raise RuntimeError("formal cache SHA changed")
    frozen = load_json(root / "reports/phase3_14b_r22_frozen_contract.json")
    if frozen.get("artifact_sha256") != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA changed")

    evidence = {
        RESUME4_TEST_GATE_PATH: require_sha(
            root, RESUME4_TEST_GATE_PATH, EXPECTED_RESUME4_TEST_GATE_SHA256
        ),
        RESUME4_PREFLIGHT_PATH: require_sha(
            root, RESUME4_PREFLIGHT_PATH, EXPECTED_RESUME4_PREFLIGHT_SHA256
        ),
        RESUME4_AUDIT_PATH: require_sha(
            root, RESUME4_AUDIT_PATH, EXPECTED_RESUME4_AUDIT_SHA256
        ),
        RESUME4_SUMMARY_PATH: require_sha(
            root, RESUME4_SUMMARY_PATH, EXPECTED_RESUME4_SUMMARY_SHA256
        ),
        RESUME4_MARKDOWN_PATH: require_sha(
            root, RESUME4_MARKDOWN_PATH, EXPECTED_RESUME4_MARKDOWN_SHA256
        ),
        RESUME3_PILOT_PATH: require_sha(
            root, RESUME3_PILOT_PATH, EXPECTED_RESUME3_PILOT_SHA256
        ),
    }
    test_gate = load_json(test_gate_path)
    if test_gate.get("verdict") != "PASS":
        raise RuntimeError("Resume5 scoped test gate did not pass")
    if test_gate.get("schema") != "phase314b_r254_resume5_scoped_static_test_gate_v1":
        raise RuntimeError("Resume5 test-gate schema changed")
    if test_gate.get("passed_test_count") != EXPECTED_SCOPED_TEST_COUNT:
        raise RuntimeError("Resume5 scoped test count changed")
    if test_gate.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume5 source changed after static tests")
    for relative, expected in test_gate.get("test_manifest_sha256", {}).items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"test changed after gate: {relative}")
    for key in (
        "selection_uses_ignore",
        "selection_uses_k_expression",
        "selection_uses_deselection",
    ):
        if test_gate.get(key) is not False:
            raise RuntimeError(f"forbidden test selection mechanism: {key}")
    spec = Resume5Spec()
    spec.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("Resume5 CUDA device is unavailable")
    if torch.cuda.get_device_name(0) != spec.expected_gpu_name:
        raise RuntimeError("Resume5 requires the fixed RTX 4090")

    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume5",
        "phase_id": PHASE,
        "schema": "phase314b_r254_resume5_preflight_v1",
        "verdict": "PASS",
        "repository": {
            "branch": "Experiment1",
            "head": git_output(root, "rev-parse", "HEAD"),
            "base_evidence_commit": BASE_EVIDENCE_COMMIT,
            "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
            "cache_sha256": EXPECTED_CACHE_SHA256,
            "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        },
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "spec": asdict(spec),
        "source_sha256": source_sha256(root),
        "immutable_evidence_sha256": evidence,
        "test_gate_report": test_gate_relative,
        "test_gate_sha256": sha256_file(test_gate_path),
        "test_gate": {
            "passed_test_count": test_gate["passed_test_count"],
            "test_manifest_sha256": test_gate["test_manifest_sha256"],
            "full_repository_suite_required": False,
            "out_of_scope_collection_baseline": test_gate[
                "out_of_scope_collection_baseline"
            ],
        },
        "execution_contract": {
            "isolated_worker_processes": 3,
            "worker_persists_files": False,
            "same_device_identity_uses_exact_sha": True,
            "functional_contract_adds_numeric_tolerance": False,
            "robot_proxy_metrics_are_gate_inputs": False,
            "legacy_cache_loader_eagerly_materializes_full_npz": True,
            "validation_target_rows_may_not_be_indexed": True,
            "formal_target_rows_may_not_be_indexed": True,
        },
        "validation_targets_used": False,
        "formal_test_read": False,
        "reverse_sampling_rerun": False,
        "formal_training": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
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
