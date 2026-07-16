#!/usr/bin/env python3
"""Test gate for Stage-A portable Resume1 output-lifecycle correction."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagea_resume1_output_lifecycle import (
    RESUME1_NAMESPACE,
    atomic_write_once,
    load_json,
    make_absent_child,
    sha256_file,
    stable_json_bytes,
    validate_failed_files,
)

NEW_TEST = (
    "tests/"
    "test_phase3_14b_r258_stagea_resume1_output_lifecycle.py"
)
ORIGINAL_GATE_SCRIPT = (
    "scripts/"
    "phase3_14b_r258_stagea_test_gate.py"
)


def parse_pass_count(output: str) -> int:
    matches = re.findall(r"(\d+)\s+passed", output)
    if not matches:
        raise RuntimeError("could not parse pytest pass count")
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
            "phase3_14b_r258_stagea_resume1_"
            "test_gate_summary.json"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite Resume1 test gate: {}".format(output)
        )

    provenance = validate_failed_files(
        root,
        require_tracked=True,
    )

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_stagea_resume1_gate_",
            dir="/tmp",
        )
    )
    original_gate_path = make_absent_child(
        temporary_root,
        "original_portable_test_gate.json",
    )
    try:
        original = subprocess.run(
            [
                sys.executable,
                str(root / ORIGINAL_GATE_SCRIPT),
                "--root",
                str(root),
                "--output",
                str(original_gate_path),
            ],
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(original.stdout, end="")
        if original.returncode != 0:
            raise RuntimeError(
                "original portable test gate failed"
            )
        original_report = load_json(original_gate_path)
        if original_report.get("verdict") != "PASS":
            raise RuntimeError(
                "original portable test gate is not PASS"
            )
        if int(
            original_report.get("passed_test_count", -1)
        ) != 899:
            raise RuntimeError(
                "original portable pass count changed"
            )
        if int(
            original_report.get("test_file_count", -1)
        ) != 47:
            raise RuntimeError(
                "original portable test-file count changed"
            )
        if int(
            original_report.get("r258_stagea_new_passed", -1)
        ) != 62:
            raise RuntimeError(
                "original portable new-test count changed"
            )

        resume1 = subprocess.run(
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
        print(resume1.stdout, end="")
        if resume1.returncode != 0:
            raise RuntimeError("Resume1 tests failed")
        resume1_passed = parse_pass_count(
            resume1.stdout
        )
        if resume1_passed != 12:
            raise RuntimeError(
                "Resume1 new-test count changed: {}".format(
                    resume1_passed
                )
            )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    manifest = dict(
        original_report["test_manifest_sha256"]
    )
    manifest[NEW_TEST] = sha256_file(
        root / NEW_TEST
    )
    report = {
        "phase":
            "Phase3.14b-r2.5.8 Stage A Portable Resume1",
        "schema":
            "phase314b_r258_stagea_resume1_test_gate_v1",
        "verdict": "PASS",
        "namespace": RESUME1_NAMESPACE,
        "failed_attempt_provenance":
            provenance,
        "original_portable_passed": 899,
        "resume1_new_passed": 12,
        "passed_test_count": 911,
        "test_file_count": 48,
        "test_manifest_sha256": manifest,
        "original_gate_replayed_in_tmp": True,
        "original_gate_output_persisted": False,
        "environment_probe_output_precreated": False,
        "selection_uses_glob": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "selection_temporarily_hides_files": False,
    }
    atomic_write_once(
        output,
        stable_json_bytes(report),
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "passed": 911,
                "new_passed": 12,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
