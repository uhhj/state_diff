#!/usr/bin/env python3
"""Run the Resume4 report-only reproduction decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume4_reproduction_audit import (
    R253_PILOT_PATH,
    R253_SUMMARY_PATH,
    RESUME3_BLOCKED_PATH,
    RESUME3_PILOT_PATH,
    SUBMODULE_ENVIRONMENT_PATH,
    SUBMODULE_TASK_PATH,
    assert_only_allowed_worktree_paths,
    audit_robot_proxy_sources,
    build_reproduction_audit,
    git_output,
    load_json,
    sha256_file,
    source_sha256,
    write_json_once,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--test-gate-report",
        default="reports/phase3_14b_r254_resume4_test_gate_summary.json",
    )
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r254_resume4_preflight_summary.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume4_reproduction_audit_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    test_gate_path = Path(args.test_gate_report)
    if not test_gate_path.is_absolute():
        test_gate_path = root / test_gate_path
    test_gate_path = test_gate_path.resolve()
    preflight_path = Path(args.preflight_report)
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    preflight_path = preflight_path.resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite audit: {output}")

    preflight_relative = preflight_path.relative_to(root).as_posix()
    test_gate_relative = test_gate_path.relative_to(root).as_posix()
    assert_only_allowed_worktree_paths(
        root, (test_gate_relative, preflight_relative)
    )
    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Resume4 preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume4 source changed after preflight")
    if preflight.get("test_gate_report") != test_gate_relative:
        raise RuntimeError("Resume4 test-gate path changed after preflight")
    if preflight.get("test_gate_sha256") != sha256_file(test_gate_path):
        raise RuntimeError("Resume4 test-gate evidence changed after preflight")
    for relative, expected in preflight.get("evidence_sha256", {}).items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"evidence changed after preflight: {relative}")

    static_source_audit = audit_robot_proxy_sources(
        (root / SUBMODULE_TASK_PATH).read_text(encoding="utf-8"),
        (root / SUBMODULE_ENVIRONMENT_PATH).read_text(encoding="utf-8"),
    )
    if not static_source_audit["source_patterns_recognized"]:
        raise RuntimeError("pinned robot-proxy source pattern changed")
    report = build_reproduction_audit(
        r253_pilot=load_json(root / R253_PILOT_PATH),
        r253_summary=load_json(root / R253_SUMMARY_PATH),
        resume3_pilot=load_json(root / RESUME3_PILOT_PATH),
        resume3_blocked=load_json(root / RESUME3_BLOCKED_PATH),
        robot_proxy_source_audit=static_source_audit,
    )
    report["repository"] = {
        "branch": git_output(root, "branch", "--show-current"),
        "head": git_output(root, "rev-parse", "HEAD"),
        "submodule_commit": git_output(
            root / "external/deformable-ravens", "rev-parse", "HEAD"
        ),
    }
    report["preflight_report"] = preflight_relative
    report["preflight_sha256"] = sha256_file(preflight_path)
    report["test_gate_report"] = test_gate_relative
    report["test_gate_sha256"] = sha256_file(test_gate_path)
    report["test_gate"] = dict(preflight["test_gate"])
    report["source_sha256"] = source_sha256(root)
    report["evidence_sha256"] = dict(preflight["evidence_sha256"])
    write_json_once(output, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "scientific_status": report["scientific_status"],
                "root_cause": report["root_cause"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
