#!/usr/bin/env python3
"""Test gate for Stage-C Resume1 commit-outcome recovery."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagec_resume1_commit_recovery import (
    FIRST_TEST_GATE,
    IMPLEMENTATION_COMMIT,
    REMOTE_BASE_COMMIT,
    RESUME_TEST_GATE,
    atomic_write_once,
    load_json,
    sha256_file,
    stable_json_bytes,
    validate_implementation_commit,
    validate_provenance_commit,
)

BASE_GATE_SCRIPT = (
    "scripts/"
    "phase3_14b_r258_stagec_test_gate.py"
)
NEW_TEST = (
    "tests/"
    "test_phase3_14b_r258_stagec_resume1_commit_recovery.py"
)


def parse_pass_count(output: str) -> int:
    matches = re.findall(
        r"(\d+)\s+passed",
        output,
    )
    if not matches:
        raise RuntimeError(
            "could not parse pytest pass count"
        )
    return int(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/state_diff2",
    )
    parser.add_argument(
        "--output",
        default=RESUME_TEST_GATE,
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite Resume1 test gate: {}".format(output)
        )

    implementation = (
        validate_implementation_commit(
            root
        )
    )
    provenance = (
        validate_provenance_commit(
            root
        )
    )
    first_path = root / FIRST_TEST_GATE
    first = load_json(first_path)
    if first.get("verdict") != "PASS":
        raise RuntimeError(
            "first Stage-C test gate is not PASS"
        )
    if int(
        first.get("passed_test_count", -1)
    ) != 1023:
        raise RuntimeError(
            "first Stage-C pass count changed"
        )
    if int(
        first.get("test_file_count", -1)
    ) != 50:
        raise RuntimeError(
            "first Stage-C test-file count changed"
        )
    if int(
        first.get("stagec_new_passed", -1)
    ) != 64:
        raise RuntimeError(
            "first Stage-C new-test count changed"
        )
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                provenance[
                    "provenance_commit"
                ],
                FIRST_TEST_GATE,
            ),
        ],
        cwd=str(root),
    )
    if committed != first_path.read_bytes():
        raise RuntimeError(
            "first Stage-C test gate differs from provenance commit"
        )
    manifest = first.get(
        "test_manifest_sha256"
    )
    if (
        not isinstance(manifest, dict)
        or len(manifest) != 50
    ):
        raise RuntimeError(
            "first Stage-C test manifest changed"
        )
    for relative, expected in manifest.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(
                "frozen test changed: {}".format(relative)
            )

    compile_targets = [
        root
        / (
            "ccda_phase3/"
            "phase314b_r258_stagec_resume1_commit_recovery.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_resume1_worker.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_resume1_run_calibration.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_resume1_test_gate.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_resume1_blocked.py"
        ),
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
        ]
        + [
            str(path)
            for path in compile_targets
        ],
        cwd=str(root),
        check=True,
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_stagec_resume1_gate_",
            dir="/tmp",
        )
    )
    replay_output = (
        temporary_root
        / "stagec_test_gate.json"
    )
    if replay_output.exists():
        raise FileExistsError(
            "temporary Stage-C gate output exists"
        )
    try:
        replay = subprocess.run(
            [
                sys.executable,
                str(root / BASE_GATE_SCRIPT),
                "--root",
                str(root),
                "--output",
                str(replay_output),
            ],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(replay.stdout, end="")
        if replay.returncode != 0:
            raise RuntimeError(
                "first Stage-C test-gate replay failed"
            )
        replay_report = load_json(
            replay_output
        )
        if (
            replay_report.get("verdict")
            != "PASS"
            or int(
                replay_report.get(
                    "passed_test_count",
                    -1,
                )
            )
            != 1023
            or int(
                replay_report.get(
                    "test_file_count",
                    -1,
                )
            )
            != 50
            or int(
                replay_report.get(
                    "stagec_new_passed",
                    -1,
                )
            )
            != 64
        ):
            raise RuntimeError(
                "first Stage-C test-gate replay changed"
            )

        new_tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(root / NEW_TEST),
            ],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(
            new_tests.stdout,
            end="",
        )
        if new_tests.returncode != 0:
            raise RuntimeError(
                "Resume1 tests failed"
            )
        new_passed = parse_pass_count(
            new_tests.stdout
        )
        if new_passed != 41:
            raise RuntimeError(
                "Resume1 new-test count changed: {}".format(new_passed)
            )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    total_passed = 1023 + new_passed
    if total_passed != 1064:
        raise RuntimeError(
            "Resume1 total pass count changed"
        )
    test_manifest: Dict[str, str] = dict(
        manifest
    )
    test_manifest[NEW_TEST] = (
        sha256_file(root / NEW_TEST)
    )
    report: Dict[str, Any] = {
        "phase":
            "Phase3.14b-r2.5.8 Stage C Resume1",
        "schema":
            "phase314b_r258_stagec_resume1_test_gate_v1",
        "verdict": "PASS",
        "remote_base_commit":
            REMOTE_BASE_COMMIT,
        "implementation_commit":
            IMPLEMENTATION_COMMIT,
        "provenance_commit":
            provenance[
                "provenance_commit"
            ],
        "first_gate_path":
            FIRST_TEST_GATE,
        "first_gate_sha256":
            sha256_file(
                first_path
            ),
        "first_gate_replayed_in_tmp":
            True,
        "first_gate_output_persisted":
            False,
        "base_passed": 1023,
        "resume1_new_passed":
            new_passed,
        "passed_test_count":
            total_passed,
        "test_file_count": 51,
        "test_manifest_sha256":
            test_manifest,
        "implementation_provenance":
            implementation,
        "failure_provenance":
            provenance,
        "commit_exit_code_141_recovered":
            True,
        "scientific_calibration_started_before_resume":
            False,
        "selection_uses_glob": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression":
            False,
        "selection_uses_deselection":
            False,
        "selection_temporarily_hides_files":
            False,
    }
    atomic_write_once(
        output,
        stable_json_bytes(report),
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "passed": total_passed,
                "new_passed": new_passed,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
