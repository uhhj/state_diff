#!/usr/bin/env python3
"""Manifest-driven scoped test gate without glob collisions or file hiding."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccda_phase3.phase314b_r255_robot_proxy_provenance import sha256_file, write_json_once
from ccda_phase3.phase314b_r255_stageb_dataset import EXPECTED_STAGE_A_TEST_GATE_SHA256

NEW_TEST = "tests/test_phase3_14b_r255_stageb_dataset.py"
STAGE_A_NEW_TEST = "tests/test_phase3_14b_r255_robot_proxy_provenance.py"
EXPECTED_NEW_TESTS = 14
EXPECTED_STAGE_A_NEW_TESTS = 6
FROZEN_BASELINE_COMMIT = "795905d565faaae5244e6c30dce31b8a0dd98066"


def run_pytest(
    *,
    cwd: Path,
    tests: List[str],
    expected_passed: int,
) -> int:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(cwd)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"] + tests,
        cwd=str(cwd),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"scoped test gate failed: {completed.returncode}")
    if re.search(
        r"\b(failed|error|skipped|xfailed|xpassed|deselected)\b",
        completed.stdout,
        flags=re.IGNORECASE,
    ):
        raise RuntimeError("scoped test gate reported a non-pass status")
    matches = re.findall(r"(\d+)\s+passed", completed.stdout)
    if not matches:
        raise RuntimeError("could not parse pytest pass count")
    passed = int(matches[-1])
    if passed != expected_passed:
        raise RuntimeError(
            f"scoped test count {passed} != expected {expected_passed}"
        )
    return passed


def prepare_frozen_worktree(
    *,
    root: Path,
    worktree: Path,
    historical_tests: List[str],
    frozen: Dict[str, str],
) -> None:
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
        stdout=subprocess.DEVNULL,
    )
    for relative in historical_tests:
        path = worktree / relative
        if not path.is_file() or sha256_file(path) != frozen[relative]:
            raise RuntimeError(
                f"detached frozen test differs from manifest: {relative}"
            )
    for name in ("data", "checkpoints"):
        source = root / name
        target = worktree / name
        if source.exists() and not target.exists():
            target.symlink_to(source, target_is_directory=True)
    external = worktree / "external/deformable-ravens"
    if external.is_dir() and not external.is_symlink():
        external.rmdir()
    if not external.exists():
        external.parent.mkdir(parents=True, exist_ok=True)
        external.symlink_to(
            root / "external/deformable-ravens",
            target_is_directory=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--stage-a-gate",
        default="reports/phase3_14b_r255_test_gate_summary.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r255_stageb_test_gate_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    stage_a_path = (root / args.stage_a_gate).resolve()
    output = (root / args.output).resolve()
    if sha256_file(stage_a_path) != EXPECTED_STAGE_A_TEST_GATE_SHA256:
        raise RuntimeError("Stage-A frozen test-gate SHA changed")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    frozen: Dict[str, str] = dict(stage_a["test_manifest_sha256"])
    for relative, expected in frozen.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen test changed: {relative}")
    new_test = root / NEW_TEST
    if not new_test.is_file():
        raise FileNotFoundError(new_test)
    new_test_sha_before = sha256_file(new_test)
    tests: List[str] = list(frozen.keys()) + [NEW_TEST]
    if len(tests) != len(set(tests)):
        raise RuntimeError("test manifest contains duplicate paths")

    compile_targets = [
        "ccda_phase3/schema_v3.py",
        "ccda_phase3/phase314b_r255_stageb_dataset.py",
        "scripts/phase3_14b_r255_stageb_worker.py",
        "scripts/phase3_14b_r255_stageb_test_gate.py",
        "scripts/phase3_14b_r255_stageb_build.py",
        "scripts/phase3_14b_r255_stageb_blocked.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "py_compile"] + compile_targets,
        cwd=str(root),
        check=True,
    )
    expected_passed = int(stage_a["passed_test_count"]) + EXPECTED_NEW_TESTS
    historical_tests = [
        relative
        for relative in frozen
        if relative != STAGE_A_NEW_TEST
    ]
    if (
        STAGE_A_NEW_TEST not in frozen
        or len(historical_tests) + 1 != len(frozen)
    ):
        raise RuntimeError("Stage-A manifest split is not the fixed 27+1")
    expected_historical = (
        int(stage_a["passed_test_count"]) - EXPECTED_STAGE_A_NEW_TESTS
    )
    temporary_parent = Path(
        tempfile.mkdtemp(prefix="phase314b_r255_stageb_gate_")
    )
    worktree = temporary_parent / "frozen"
    try:
        prepare_frozen_worktree(
            root=root,
            worktree=worktree,
            historical_tests=historical_tests,
            frozen=frozen,
        )
        historical_passed = run_pytest(
            cwd=worktree,
            tests=historical_tests,
            expected_passed=expected_historical,
        )
        stage_a_passed = run_pytest(
            cwd=root,
            tests=[STAGE_A_NEW_TEST],
            expected_passed=EXPECTED_STAGE_A_NEW_TESTS,
        )
        stage_b_passed = run_pytest(
            cwd=root,
            tests=[NEW_TEST],
            expected_passed=EXPECTED_NEW_TESTS,
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(temporary_parent, ignore_errors=True)
    passed = historical_passed + stage_a_passed + stage_b_passed
    if passed != expected_passed:
        raise RuntimeError(f"combined pass count {passed} != {expected_passed}")
    new_test_sha_after = sha256_file(new_test)
    if new_test_sha_after != new_test_sha_before:
        raise RuntimeError("new Stage-B test changed during the test gate")
    manifest = dict(frozen)
    manifest[NEW_TEST] = new_test_sha_after
    report = {
        "phase": "Phase3.14b-r2.5.5 Stage B",
        "schema": "phase314b_r255_stageb_manifest_driven_test_gate_v1",
        "verdict": "PASS",
        "passed_test_count": passed,
        "expected_passed_test_count": expected_passed,
        "test_file_count": len(tests),
        "test_manifest_sha256": manifest,
        "stage_a_test_gate_sha256": EXPECTED_STAGE_A_TEST_GATE_SHA256,
        "pytest_invocation_count": 3,
        "historical_closure_execution": "detached_frozen_baseline_worktree",
        "detached_frozen_baseline_commit": FROZEN_BASELINE_COMMIT,
        "historical_passed_test_count": historical_passed,
        "stage_a_new_passed_test_count": stage_a_passed,
        "stage_b_new_passed_test_count": stage_b_passed,
        "single_pytest_invocation": False,
        "selection_uses_glob": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "selection_temporarily_hides_files": False,
        "historical_tests_modified": False,
    }
    write_json_once(output, report)
    print(json.dumps({"verdict": "PASS", "passed": passed}))


if __name__ == "__main__":
    main()
