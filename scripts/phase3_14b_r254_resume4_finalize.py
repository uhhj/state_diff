#!/usr/bin/env python3
"""Finalize the successful Resume4 audit without unblocking research gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume4_reproduction_audit import (
    EXPECTED_PHASE3_R2_PASS_COUNT,
    assert_only_allowed_worktree_paths,
    load_json,
    render_markdown,
    sha256_file,
    source_sha256,
    write_json_once,
    write_text_once,
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
        "--audit-report",
        default="reports/phase3_14b_r254_resume4_reproduction_audit_summary.json",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/phase3_14b_r254_resume4_summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/phase3_14b_r254_resume4_report.md",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    test_gate_path = resolve(args.test_gate_report)
    preflight_path = resolve(args.preflight_report)
    audit_path = resolve(args.audit_report)
    summary_path = resolve(args.summary_output)
    markdown_path = resolve(args.markdown_output)
    if summary_path.exists() or markdown_path.exists():
        raise RuntimeError("refusing to overwrite Resume4 final evidence")
    allowed = (
        test_gate_path.relative_to(root).as_posix(),
        preflight_path.relative_to(root).as_posix(),
        audit_path.relative_to(root).as_posix(),
    )
    assert_only_allowed_worktree_paths(root, allowed)
    test_gate = load_json(test_gate_path)
    preflight = load_json(preflight_path)
    audit = load_json(audit_path)
    if (
        test_gate.get("verdict") != "PASS"
        or preflight.get("verdict") != "PASS"
        or audit.get("verdict") != "PASS"
    ):
        raise RuntimeError("Resume4 test gate/preflight/audit did not pass")
    if test_gate.get("passed_test_count") != EXPECTED_PHASE3_R2_PASS_COUNT:
        raise RuntimeError("Resume4 scoped test count changed")
    if preflight.get("test_gate_sha256") != sha256_file(test_gate_path):
        raise RuntimeError("Resume4 test-gate evidence changed")
    if audit.get("test_gate_sha256") != sha256_file(test_gate_path):
        raise RuntimeError("Resume4 audit references different test-gate evidence")
    if audit.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume4 source changed after audit")
    if audit.get("scientific_status") != "BLOCKED":
        raise RuntimeError("unexpected Resume4 scientific status")
    if audit.get("robot_proxy_attribution_interpretable") is not False:
        raise RuntimeError("Resume4 must not interpret robot-proxy attribution")
    if audit.get("train_only_recommendation") is not None:
        raise RuntimeError("Resume4 must not recommend a model")
    if audit.get("selected_configuration") is not None:
        raise RuntimeError("Resume4 must not select a configuration")

    summary = dict(audit)
    summary.update(
        {
            "phase": "Phase3.14b-r2.5.4 Resume4",
            "meaning": "committed-evidence reproduction mismatch decomposed",
            "static_test_count": int(test_gate["passed_test_count"]),
            "static_test_file_count": int(test_gate["test_file_count"]),
            "static_test_manifest_sha256": dict(
                test_gate["test_manifest_sha256"]
            ),
            "full_repository_suite_required": False,
            "out_of_scope_collection_baseline": list(
                test_gate["out_of_scope_collection_baseline"]
            ),
            "test_gate_report": test_gate_path.relative_to(root).as_posix(),
            "test_gate_sha256": sha256_file(test_gate_path),
            "preflight_report": preflight_path.relative_to(root).as_posix(),
            "preflight_sha256": sha256_file(preflight_path),
            "audit_report": audit_path.relative_to(root).as_posix(),
            "audit_sha256": sha256_file(audit_path),
        }
    )
    markdown = render_markdown(summary, int(test_gate["passed_test_count"]))
    write_text_once(markdown_path, markdown)
    write_json_once(summary_path, summary)
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause": summary["root_cause"],
                "summary": str(summary_path),
                "report": str(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
