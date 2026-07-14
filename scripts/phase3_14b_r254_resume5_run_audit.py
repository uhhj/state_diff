#!/usr/bin/env python3
"""Run three isolated workers and persist only the combined Resume5 audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    RESUME4_AUDIT_PATH,
    Resume5Spec,
    assert_only_allowed_worktree_paths,
    build_resume5_audit,
    git_output,
    load_json,
    parse_worker_stdout,
    sha256_file,
    source_sha256,
    write_json_once,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--test-gate-report",
        default="reports/phase3_14b_r254_resume5_test_gate_summary.json",
    )
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r254_resume5_preflight_summary.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume5_determinism_audit_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    test_gate_path = resolve(args.test_gate_report)
    preflight_path = resolve(args.preflight_report)
    output = resolve(args.output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Resume5 audit: {output}")
    test_relative = test_gate_path.relative_to(root).as_posix()
    preflight_relative = preflight_path.relative_to(root).as_posix()
    allowed = (test_relative, preflight_relative)
    assert_only_allowed_worktree_paths(root, allowed)
    test_gate = load_json(test_gate_path)
    preflight = load_json(preflight_path)
    if test_gate.get("verdict") != "PASS" or preflight.get("verdict") != "PASS":
        raise RuntimeError("Resume5 test gate/preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume5 source changed after preflight")
    if preflight.get("test_gate_sha256") != sha256_file(test_gate_path):
        raise RuntimeError("Resume5 test gate changed after preflight")
    for relative, expected in preflight.get("immutable_evidence_sha256", {}).items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"immutable evidence changed: {relative}")

    worker = root / "scripts/phase3_14b_r254_resume5_worker.py"
    repeats: List[Dict[str, Any]] = []
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    for repeat_index in range(Resume5Spec().repeat_count):
        command = [
            sys.executable,
            str(worker),
            "--root",
            str(root),
            "--repeat-index",
            str(repeat_index),
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            raise RuntimeError(
                f"Resume5 worker {repeat_index} failed with "
                f"{completed.returncode}:\n{tail}"
            )
        repeats.append(parse_worker_stdout(completed.stdout))
        assert_only_allowed_worktree_paths(root, allowed)
        print(json.dumps({
            "worker": repeat_index,
            "verdict": "PASS",
            "worker_summary_sha256": repeats[-1]["worker_summary_sha256"],
        }, sort_keys=True), flush=True)

    report = build_resume5_audit(
        repeats=repeats,
        resume4_audit=load_json(root / RESUME4_AUDIT_PATH),
    )
    report["repository"] = {
        "branch": git_output(root, "branch", "--show-current"),
        "head": git_output(root, "rev-parse", "HEAD"),
        "submodule_commit": git_output(
            root / "external/deformable-ravens", "rev-parse", "HEAD"
        ),
    }
    report["source_sha256"] = source_sha256(root)
    report["test_gate_report"] = test_relative
    report["test_gate_sha256"] = sha256_file(test_gate_path)
    report["preflight_report"] = preflight_relative
    report["preflight_sha256"] = sha256_file(preflight_path)
    report["immutable_evidence_sha256"] = dict(
        preflight["immutable_evidence_sha256"]
    )
    report["worker_transport"] = {
        "method": "captured_stdout_single_json_sentinel",
        "temporary_worker_files": False,
        "raw_stdout_persisted": False,
    }
    write_json_once(output, report)
    print(json.dumps({
        "verdict": report["verdict"],
        "scientific_status": report["scientific_status"],
        "root_cause": report["root_cause"],
        "same_device_residual_determinism_pass": report[
            "same_device_residual_determinism_pass"
        ],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
