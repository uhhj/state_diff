#!/usr/bin/env python3
"""Run and persist the exact Phase3.14b r2.x static-test gate."""
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
    EXPECTED_PHASE3_R2_PASS_COUNT,
    OUT_OF_SCOPE_COLLECTION_BASELINE,
    PHASE,
    assert_only_allowed_worktree_paths,
    parse_pytest_pass_count,
    phase3_r2_test_manifest,
    source_sha256,
    write_json_once,
)


TEST_GATE_SCHEMA = "phase314b_r254_resume4_scoped_static_test_gate_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume4_test_gate_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite test-gate evidence: {output}")

    assert_only_allowed_worktree_paths(root, ())
    manifest = phase3_r2_test_manifest(root)
    command = [sys.executable, "-m", "pytest", "-q", *manifest.keys()]
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
        raise RuntimeError(
            f"scoped Phase3 r2.x pytest failed with exit {completed.returncode}"
        )
    passed = parse_pytest_pass_count(completed.stdout)
    if passed != EXPECTED_PHASE3_R2_PASS_COUNT:
        raise RuntimeError(
            f"Phase3 r2.x pass count changed: {passed} != "
            f"{EXPECTED_PHASE3_R2_PASS_COUNT}"
        )

    report: Dict[str, Any] = {
        "phase": "Phase3.14b-r2.5.4 Resume4",
        "phase_id": PHASE,
        "schema": TEST_GATE_SCHEMA,
        "verdict": "PASS",
        "meaning": "exact scoped Phase3.14b r2.x regression gate",
        "python_executable": sys.executable,
        "command": command,
        "test_file_count": len(manifest),
        "passed_test_count": passed,
        "test_manifest_sha256": manifest,
        "scope_globs": [
            "tests/test_phase3_14b_r2*.py",
            "tests/test_phase314b_r2*.py",
        ],
        "full_repository_suite_required": False,
        "selection_uses_ignore": False,
        "selection_uses_k_expression": False,
        "selection_uses_deselection": False,
        "out_of_scope_collection_baseline": [
            dict(item) for item in OUT_OF_SCOPE_COLLECTION_BASELINE
        ],
        "source_sha256": source_sha256(root),
        "retraining_performed": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
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
