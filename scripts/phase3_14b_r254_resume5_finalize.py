#!/usr/bin/env python3
"""Finalize a completed Resume5 audit without selecting a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    EXPECTED_SCOPED_TEST_COUNT,
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
        default="reports/phase3_14b_r254_resume5_test_gate_summary.json",
    )
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r254_resume5_preflight_summary.json",
    )
    parser.add_argument(
        "--audit-report",
        default="reports/phase3_14b_r254_resume5_determinism_audit_summary.json",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/phase3_14b_r254_resume5_summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/phase3_14b_r254_resume5_report.md",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    test_path = resolve(args.test_gate_report)
    preflight_path = resolve(args.preflight_report)
    audit_path = resolve(args.audit_report)
    summary_path = resolve(args.summary_output)
    markdown_path = resolve(args.markdown_output)
    if summary_path.exists() or markdown_path.exists():
        raise RuntimeError("refusing to overwrite Resume5 final evidence")
    allowed = tuple(
        path.relative_to(root).as_posix()
        for path in (test_path, preflight_path, audit_path)
    )
    assert_only_allowed_worktree_paths(root, allowed)
    test_gate = load_json(test_path)
    preflight = load_json(preflight_path)
    audit = load_json(audit_path)
    if any(item.get("verdict") != "PASS" for item in (test_gate, preflight, audit)):
        raise RuntimeError("Resume5 gate/preflight/audit did not complete")
    if test_gate.get("passed_test_count") != EXPECTED_SCOPED_TEST_COUNT:
        raise RuntimeError("Resume5 static-test count changed")
    if preflight.get("test_gate_sha256") != sha256_file(test_path):
        raise RuntimeError("Resume5 test gate changed")
    if audit.get("preflight_sha256") != sha256_file(preflight_path):
        raise RuntimeError("Resume5 preflight changed")
    if audit.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume5 source changed after runtime audit")
    required_false = (
        "robot_proxy_attribution_interpretable",
        "robot_proxy_metrics_used_for_gate",
        "validation_target_rows_indexed",
        "formal_target_rows_indexed",
        "validation_targets_used",
        "formal_test_read",
        "reverse_sampling_rerun",
        "formal_training",
        "checkpoint_saved",
        "weights_persisted",
        "prediction_tensor_persisted",
        "candidate_execution",
        "idm",
        "phase4",
        "cps",
    )
    for key in required_false:
        if audit.get(key) is not False:
            raise RuntimeError(f"Resume5 boundary changed: {key}")
    if audit.get("legacy_cache_loader_eagerly_materializes_full_npz") is not True:
        raise RuntimeError("Resume5 must disclose legacy eager NPZ materialization")
    if audit.get("train_only_recommendation") is not None:
        raise RuntimeError("Resume5 must not recommend a model")
    if audit.get("selected_configuration") is not None:
        raise RuntimeError("Resume5 must not select a model")

    summary = dict(audit)
    summary.update({
        "phase": "Phase3.14b-r2.5.4 Resume5",
        "meaning": "same-device residual determinism and cable functional non-regression audited",
        "static_test_count": int(test_gate["passed_test_count"]),
        "static_test_file_count": len(test_gate["test_manifest_sha256"]),
        "test_gate_report": test_path.relative_to(root).as_posix(),
        "test_gate_sha256": sha256_file(test_path),
        "preflight_report": preflight_path.relative_to(root).as_posix(),
        "preflight_sha256": sha256_file(preflight_path),
        "audit_report": audit_path.relative_to(root).as_posix(),
        "audit_sha256": sha256_file(audit_path),
    })
    markdown = render_markdown(summary, int(test_gate["passed_test_count"]))
    write_text_once(markdown_path, markdown)
    write_json_once(summary_path, summary)
    print(json.dumps({
        "verdict": "PASS",
        "scientific_status": summary["scientific_status"],
        "root_cause": summary["root_cause"],
        "summary": str(summary_path),
        "report": str(markdown_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
