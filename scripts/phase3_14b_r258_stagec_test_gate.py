#!/usr/bin/env python3
"""Manifest-driven test gate for Phase3.14b-r2.5.8 Stage C."""
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

from ccda_phase3.phase314b_r258_stagec_balanced_geometry import (
    BASE_EVIDENCE_COMMIT,
    BASE_TEST_GATE,
    atomic_write_once,
    load_json,
    sha256_file,
    stable_json_bytes,
    validate_base_evidence,
)

BASE_GATE_SCRIPT = (
    "scripts/"
    "phase3_14b_r258_stageb_test_gate.py"
)
NEW_TEST = (
    "tests/"
    "test_phase3_14b_r258_stagec_balanced_geometry.py"
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
        default=(
            "reports/"
            "phase3_14b_r258_stagec_"
            "test_gate_summary.json"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite Stage-C test gate: {}".format(output)
        )

    provenance = validate_base_evidence(root)
    base_path = root / BASE_TEST_GATE
    base = load_json(base_path)
    if base.get("verdict") != "PASS":
        raise RuntimeError(
            "base test gate is not PASS"
        )
    if int(
        base.get("passed_test_count", -1)
    ) != 959:
        raise RuntimeError(
            "base pass count changed"
        )
    if int(
        base.get("test_file_count", -1)
    ) != 49:
        raise RuntimeError(
            "base test-file count changed"
        )
    if int(
        base.get("stageb_new_passed", -1)
    ) != 48:
        raise RuntimeError(
            "base Stage-B new-test count changed"
        )
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(
                BASE_EVIDENCE_COMMIT,
                BASE_TEST_GATE,
            ),
        ],
        cwd=str(root),
    )
    if committed != base_path.read_bytes():
        raise RuntimeError(
            "base test gate differs from evidence commit"
        )
    manifest = base.get(
        "test_manifest_sha256"
    )
    if (
        not isinstance(manifest, dict)
        or len(manifest) != 49
    ):
        raise RuntimeError(
            "base test manifest changed"
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
            "phase314b_r258_stagec_balanced_geometry.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_worker.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_run_calibration.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_test_gate.py"
        ),
        root
        / (
            "scripts/"
            "phase3_14b_r258_stagec_blocked.py"
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
            prefix="phase314b_r258_stagec_gate_",
            dir="/tmp",
        )
    )
    base_replay_output = (
        temporary_root
        / "base_test_gate.json"
    )
    if base_replay_output.exists():
        raise FileExistsError(
            "temporary base-gate output exists"
        )
    try:
        replay = subprocess.run(
            [
                sys.executable,
                str(root / BASE_GATE_SCRIPT),
                "--root",
                str(root),
                "--output",
                str(base_replay_output),
            ],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(replay.stdout, end="")
        if replay.returncode != 0:
            raise RuntimeError(
                "base Stage-B test-gate replay failed"
            )
        replay_report = load_json(
            base_replay_output
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
            != 959
            or int(
                replay_report.get(
                    "test_file_count",
                    -1,
                )
            )
            != 49
        ):
            raise RuntimeError(
                "base test-gate replay changed"
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
                "Stage-C tests failed"
            )
        new_passed = parse_pass_count(
            new_tests.stdout
        )
        if new_passed != 64:
            raise RuntimeError(
                "Stage-C new-test count changed: {}".format(new_passed)
            )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    total_passed = 959 + new_passed
    if total_passed != 1023:
        raise RuntimeError(
            "Stage-C total pass count changed"
        )
    test_manifest: Dict[str, str] = dict(
        manifest
    )
    test_manifest[NEW_TEST] = (
        sha256_file(root / NEW_TEST)
    )
    report: Dict[str, Any] = {
        "phase":
            "Phase3.14b-r2.5.8 Stage C",
        "schema":
            "phase314b_r258_stagec_test_gate_v1",
        "verdict": "PASS",
        "base_evidence_commit":
            BASE_EVIDENCE_COMMIT,
        "base_gate_path":
            BASE_TEST_GATE,
        "base_gate_replayed_in_tmp":
            True,
        "base_gate_output_persisted":
            False,
        "base_passed": 959,
        "stagec_new_passed":
            new_passed,
        "passed_test_count":
            total_passed,
        "test_file_count": 50,
        "test_manifest_sha256":
            test_manifest,
        "base_provenance": {
            "base_evidence_commit":
                provenance[
                    "base_evidence_commit"
                ],
            "base_worker_sha256":
                provenance[
                    "worker"
                ]["comparison"][
                    "left_sha256"
                ],
            "base_contract_sha256":
                provenance[
                    "worker_result"
                ][
                    "reachability_contract"
                ]["contract_sha256"],
            "base_selection_sha256":
                provenance[
                    "worker_result"
                ]["selection"][
                    "selection_sha256"
                ],
        },
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
