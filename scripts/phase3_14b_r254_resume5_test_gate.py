#!/usr/bin/env python3
"""Run the frozen 425-test base plus the explicitly scoped Resume5 tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    EXPECTED_BASE_TEST_COUNT,
    EXPECTED_RESUME4_TEST_GATE_SHA256,
    EXPECTED_RESUME5_TEST_COUNT,
    EXPECTED_SCOPED_TEST_COUNT,
    PHASE,
    RESUME4_TEST_GATE_PATH,
    RESUME5_TEST_PATH,
    assert_only_allowed_worktree_paths,
    load_json,
    sha256_file,
    source_sha256,
    write_json_once,
)


def parse_pass_count(output: str) -> int:
    rows = []
    for line in str(output).splitlines():
        match = re.search(r"(?:^|\s)([0-9]+) passed(?:,|\s|$)", line)
        if match:
            rows.append((line, int(match.group(1))))
    if len(rows) != 1:
        raise ValueError("expected exactly one pytest pass summary")
    line, count = rows[0]
    forbidden = (
        " failed", " error", " skipped", " xfailed", " xpassed", " deselected"
    )
    if any(value in line for value in forbidden):
        raise ValueError(f"scoped pytest was not all-pass: {line}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume5_test_gate_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Resume5 test gate: {output}")
    assert_only_allowed_worktree_paths(root, ())

    base_path = root / RESUME4_TEST_GATE_PATH
    if sha256_file(base_path) != EXPECTED_RESUME4_TEST_GATE_SHA256:
        raise RuntimeError("Resume4 test-gate evidence changed")
    base = load_json(base_path)
    if base.get("verdict") != "PASS" or base.get("passed_test_count") != EXPECTED_BASE_TEST_COUNT:
        raise RuntimeError("Resume4 frozen test gate is not the expected 425-pass base")
    manifest = base.get("test_manifest_sha256")
    if not isinstance(manifest, dict) or len(manifest) != 27:
        raise RuntimeError("Resume4 frozen test manifest changed")
    for relative, expected in manifest.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"frozen base test changed: {relative}")
    if RESUME5_TEST_PATH.startswith("tests/test_phase314b_r2"):
        raise RuntimeError("Resume5 test name would enter the closed Resume4 glob")
    resume5_test_sha = sha256_file(root / RESUME5_TEST_PATH)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *sorted(manifest),
        RESUME5_TEST_PATH,
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    if completed.returncode != 0:
        raise RuntimeError(f"Resume5 scoped pytest failed: {completed.returncode}")
    passed = parse_pass_count(completed.stdout)
    if passed != EXPECTED_SCOPED_TEST_COUNT:
        raise RuntimeError(
            f"Resume5 pass count changed: {passed} != {EXPECTED_SCOPED_TEST_COUNT}"
        )
    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume5",
        "phase_id": PHASE,
        "schema": "phase314b_r254_resume5_scoped_static_test_gate_v1",
        "verdict": "PASS",
        "command": command,
        "base_test_file_count": len(manifest),
        "base_passed_test_count": EXPECTED_BASE_TEST_COUNT,
        "resume5_test_file_count": 1,
        "resume5_passed_test_count": EXPECTED_RESUME5_TEST_COUNT,
        "passed_test_count": passed,
        "test_manifest_sha256": {**manifest, RESUME5_TEST_PATH: resume5_test_sha},
        "resume4_test_gate_sha256": EXPECTED_RESUME4_TEST_GATE_SHA256,
        "full_repository_suite_required": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "out_of_scope_collection_baseline": base.get(
            "out_of_scope_collection_baseline", []
        ),
        "source_sha256": source_sha256(root),
        "validation_targets_used": False,
        "formal_test_read": False,
        "retraining_performed": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    write_json_once(output, report)
    print(json.dumps({"verdict": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
