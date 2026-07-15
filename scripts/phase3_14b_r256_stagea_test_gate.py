#!/usr/bin/env python3
"""Manifest-driven test gate for Phase3.14b-r2.5.6 Stage A."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_stagea_contract import (
    BASE_EVIDENCE_COMMIT,
    EXPECTED_STAGEC_TEST_GATE_SHA256,
    atomic_write_once,
    git_output,
    load_json,
    sha256_file,
    stable_json_bytes,
)

PREVIOUS_GATE = "reports/phase3_14b_r255_stagec_test_gate_summary.json"
FROZEN_BASELINE_COMMIT = "795905d565faaae5244e6c30dce31b8a0dd98066"
STAGE_A_R255_TEST = "tests/test_phase3_14b_r255_robot_proxy_provenance.py"
STAGE_B_R255_TEST = "tests/test_phase3_14b_r255_stageb_dataset.py"
STAGE_C_R255_TEST = "tests/test_phase3_14b_r255_stagec_cache_attribution.py"
STAGE_A_R256_TEST = "tests/test_phase3_14b_r256_stagea_contract.py"
EXPECTED_HISTORICAL_PASSED = 425
EXPECTED_STAGE_A_R255_PASSED = 6
EXPECTED_STAGE_B_R255_PASSED = 14
EXPECTED_STAGE_C_R255_PASSED = 18
EXPECTED_STAGE_A_R256_PASSED = 9
EXPECTED_TOTAL_PASSED = 472
EXPECTED_TEST_FILES = 31
NEW_PYTHON_FILES = (
    "ccda_phase3/phase314b_r256_stagea_contract.py",
    "scripts/phase3_14b_r256_stagea_worker.py",
    "scripts/phase3_14b_r256_stagea_build_and_audit.py",
    "scripts/phase3_14b_r256_stagea_test_gate.py",
    "scripts/phase3_14b_r256_stagea_blocked.py",
)


def run_pytest(
    *,
    python_bin: str,
    cwd: Path,
    files: Sequence[str],
) -> Tuple[int, str]:
    if not files:
        raise RuntimeError("pytest file list is empty")
    completed = subprocess.run(
        [python_bin, "-m", "pytest", "-q"] + list(files),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"pytest failed with code {completed.returncode}: {files}"
        )
    matches = re.findall(r"(\d+)\s+passed", completed.stdout)
    if not matches:
        raise RuntimeError("could not parse pytest pass count")
    return int(matches[-1]), completed.stdout


def verify_manifest(root: Path, manifest: Mapping[str, str]) -> None:
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen test changed: {relative}: {observed} != {expected}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r256_stagea_test_gate_summary.json",
    )
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite test evidence: {output}")
    if git_output(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage A requires Experiment1")
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_EVIDENCE_COMMIT,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Stage C Resume1 evidence is not an ancestor")

    previous_path = root / PREVIOUS_GATE
    if sha256_file(previous_path) != EXPECTED_STAGEC_TEST_GATE_SHA256:
        raise RuntimeError("Stage-C test gate SHA changed")
    previous = load_json(previous_path)
    if previous.get("verdict") != "PASS":
        raise RuntimeError("Stage-C test gate is not PASS")
    if int(previous.get("passed_test_count", -1)) != 463:
        raise RuntimeError("Stage-C pass count changed")
    frozen_manifest = {
        str(key): str(value)
        for key, value in previous["test_manifest_sha256"].items()
    }
    if len(frozen_manifest) != 30:
        raise RuntimeError("Stage-C frozen test file count changed")
    verify_manifest(root, frozen_manifest)
    for required in (
        STAGE_A_R255_TEST,
        STAGE_B_R255_TEST,
        STAGE_C_R255_TEST,
    ):
        if required not in frozen_manifest:
            raise RuntimeError(f"frozen manifest is missing {required}")
    if not (root / STAGE_A_R256_TEST).is_file():
        raise FileNotFoundError(root / STAGE_A_R256_TEST)

    subprocess.run(
        [args.python_bin, "-m", "py_compile"]
        + [str(root / relative) for relative in NEW_PYTHON_FILES],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        ["bash", "-n", str(root / "scripts/phase3_14b_r256_stagea_run.sh")],
        cwd=str(root),
        check=True,
    )

    historical = sorted(
        relative
        for relative in frozen_manifest
        if relative not in {
            STAGE_A_R255_TEST,
            STAGE_B_R255_TEST,
            STAGE_C_R255_TEST,
        }
    )
    worktree = root.parent / f".phase314b_r256_stagea_baseline_{os.getpid()}"
    if worktree.exists():
        raise RuntimeError(f"detached worktree already exists: {worktree}")
    historical_passed = -1
    try:
        subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                str(worktree),
                FROZEN_BASELINE_COMMIT,
            ],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        historical_passed, _ = run_pytest(
            python_bin=args.python_bin,
            cwd=worktree,
            files=historical,
        )
    finally:
        if worktree.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=str(root),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shutil.rmtree(worktree, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(root),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    stage_a_r255_passed, _ = run_pytest(
        python_bin=args.python_bin,
        cwd=root,
        files=[STAGE_A_R255_TEST],
    )
    stage_b_r255_passed, _ = run_pytest(
        python_bin=args.python_bin,
        cwd=root,
        files=[STAGE_B_R255_TEST],
    )
    stage_c_r255_passed, _ = run_pytest(
        python_bin=args.python_bin,
        cwd=root,
        files=[STAGE_C_R255_TEST],
    )
    stage_a_r256_passed, _ = run_pytest(
        python_bin=args.python_bin,
        cwd=root,
        files=[STAGE_A_R256_TEST],
    )
    total = (
        historical_passed
        + stage_a_r255_passed
        + stage_b_r255_passed
        + stage_c_r255_passed
        + stage_a_r256_passed
    )
    observed = (
        historical_passed,
        stage_a_r255_passed,
        stage_b_r255_passed,
        stage_c_r255_passed,
        stage_a_r256_passed,
        total,
    )
    expected = (
        EXPECTED_HISTORICAL_PASSED,
        EXPECTED_STAGE_A_R255_PASSED,
        EXPECTED_STAGE_B_R255_PASSED,
        EXPECTED_STAGE_C_R255_PASSED,
        EXPECTED_STAGE_A_R256_PASSED,
        EXPECTED_TOTAL_PASSED,
    )
    if observed != expected:
        raise RuntimeError(
            f"test pass-count contract changed: {observed} != {expected}"
        )

    current_manifest: Dict[str, str] = dict(frozen_manifest)
    current_manifest[STAGE_A_R256_TEST] = sha256_file(
        root / STAGE_A_R256_TEST
    )
    if len(current_manifest) != EXPECTED_TEST_FILES:
        raise RuntimeError("Stage-A test file count changed")
    payload: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.6 Stage A",
        "schema": "phase314b_r256_stagea_manifest_test_gate_v1",
        "verdict": "PASS",
        "previous_test_gate_sha256": EXPECTED_STAGEC_TEST_GATE_SHA256,
        "detached_frozen_baseline_commit": FROZEN_BASELINE_COMMIT,
        "historical_closure_execution": "detached_frozen_baseline_worktree",
        "historical_passed_test_count": historical_passed,
        "stage_a_r255_passed_test_count": stage_a_r255_passed,
        "stage_b_r255_passed_test_count": stage_b_r255_passed,
        "stage_c_r255_passed_test_count": stage_c_r255_passed,
        "stage_a_r256_passed_test_count": stage_a_r256_passed,
        "passed_test_count": total,
        "expected_passed_test_count": EXPECTED_TOTAL_PASSED,
        "test_file_count": len(current_manifest),
        "pytest_invocation_count": 5,
        "single_pytest_invocation": False,
        "selection_uses_glob": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "selection_temporarily_hides_files": False,
        "historical_tests_modified": False,
        "test_manifest_sha256": current_manifest,
    }
    atomic_write_once(output, stable_json_bytes(payload))
    print(json.dumps({"verdict": "PASS", "passed": total}, sort_keys=True))


if __name__ == "__main__":
    main()
