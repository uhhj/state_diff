#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.4.1 multirow/source-batching audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ccda_phase3.phase314a_contract import strict_json_load
from ccda_phase3.phase314b_r23_diagnostics import (
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r241_multirow import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R23_TINY_SHA256,
    PHASE,
    assert_only_allowed_worktree_paths,
    classify_audit,
    sha256_file,
)


def _direct_table(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        name: {
            "pass": bool(value["pass"]),
            "aggregate_gate": bool(value["aggregate"]["gate_pass"]),
            "all_source_gate": bool(value["all_source_gate_pass"]),
            "z_mse": float(value["aggregate"]["metrics"]["z_mse"]),
            "ordered_rmse_p95": float(
                value["aggregate"]["metrics"]["ordered_rmse"]["p95"]
            ),
        }
        for name, value in audit["direct_condition_controls"].items()
    }


def _variant_table(audit: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {}
    for name, value in audit["diffusion_variants"].items():
        true_eval = value["evaluations"]["true"]
        result[name] = {
            "pass": bool(value["pass"]),
            "aggregate_gate": bool(true_eval["gate_pass"]),
            "all_source_gate": bool(
                true_eval["all_source_gate_pass"]
            ),
            "source_pass_fraction": float(
                true_eval["source_pass_fraction"]
            ),
            "prior_drift_ratio": float(value["prior_drift_ratio"]),
            "sampling_relative_range": float(
                value["sampling"]["relative_range"]
            ),
            "condition_effect": bool(
                value["high_timestep_condition_effect"][
                    "condition_effect_supported"
                ]
            ),
            "z_mse": float(
                true_eval["aggregate"]["metrics"]["z_mse"]
            ),
            "ordered_rmse_p95": float(
                true_eval["aggregate"]["metrics"]["ordered_rmse"]["p95"]
            ),
        }
    return result


def _markdown(
    summary: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    lines = [
        "# Phase3.14b-r2.4.1 Conditioned Multirow and Source-Batching Audit",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS` (diagnosis completed, not model repair)",
        f"- Root cause: `{summary['root_cause']}`",
        (
            "- Train-only debug recommendation: "
            f"`{summary['train_only_debug_recommendation']}`"
        ),
        "- Selected configuration: `None`",
        f"- Next: `{summary['next_stage']}`",
        "",
        "## Alignment and identifiability",
        "",
        (
            "- Source-row alignment: "
            f"`{audit['source_alignment']['pass']}`"
        ),
        (
            "- Condition identifiability: "
            f"`{audit['condition_identifiability']['pass']}`"
        ),
        (
            "- Exact duplicate conflict: "
            f"`{audit['condition_identifiability']['exact_duplicate_conflict']}`"
        ),
        "",
        "## Direct condition controls",
        "",
    ]
    for name, value in _direct_table(audit).items():
        lines.append(f"- `{name}`: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(["", "## Diffusion variants", ""])
    if not audit["diffusion_variants"]:
        lines.append("- Not run because the direct condition-capacity gate failed.")
    else:
        for name, value in _variant_table(audit).items():
            lines.append(
                f"- `{name}`: `{json.dumps(value, sort_keys=True)}`"
            )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Validation targets used: `False`",
            "- Formal test read: `False`",
            "- Formal training: `False`",
            "- Candidate selected: `False`",
            "- Checkpoint saved: `False`",
            "- IDM / candidate execution: `False`",
            "- Phase4 / CPS: `False`",
            "",
            (
                "Any recommendation is train-only and does not authorize "
                "formal validation."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    repository = require_repository_state(root, require_clean=False)
    allowed = (
        "reports/phase3_14b_r241_preflight_summary.json",
        "reports/phase3_14b_r241_audit_summary.json",
    )
    assert_only_allowed_worktree_paths(root, allowed)

    preflight = strict_json_load(
        root / "reports/phase3_14b_r241_preflight_summary.json"
    )
    audit = strict_json_load(
        root / "reports/phase3_14b_r241_audit_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("preflight did not pass")
    if audit.get("verdict") != "PASS":
        raise RuntimeError("audit did not complete")
    if preflight["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache hash mismatch")
    if preflight["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("contract hash mismatch")
    if (
        preflight["historical_r23_tiny_sha256"]
        != EXPECTED_R23_TINY_SHA256
    ):
        raise RuntimeError("historical tiny report hash mismatch")
    if audit.get("source_sha256") != preflight.get("source_sha256"):
        raise RuntimeError("r2.4.1 source changed after preflight")
    if audit.get("dependency_sha256") != preflight.get(
        "dependency_sha256"
    ):
        raise RuntimeError("dependency changed after preflight")

    for key in (
        "validation_targets_used",
        "formal_test_read",
        "formal_training",
        "candidate_eligible",
        "checkpoint_saved",
        "idm",
        "candidate_execution",
        "phase4",
        "cps",
    ):
        if audit.get(key) is not False:
            raise RuntimeError(f"forbidden boundary changed: {key}")
    if audit.get("selected_configuration") not in (None, "None"):
        raise RuntimeError("formal configuration was selected")
    if (root / "checkpoints/phase3_14b_r241").exists():
        raise RuntimeError("forbidden checkpoint directory exists")

    classification = classify_audit(audit)
    for key in (
        "root_cause",
        "next_stage",
        "train_only_debug_recommendation",
    ):
        if audit.get(key) != classification.get(key):
            raise RuntimeError(f"classification mismatch: {key}")

    summary = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only conditioned multirow/source-batching diagnosis "
            "completed; no formal model repair"
        ),
        "root_cause": classification["root_cause"],
        "next_stage": classification["next_stage"],
        "train_only_debug_recommendation": classification[
            "train_only_debug_recommendation"
        ],
        "selected_configuration": None,
        "repository": repository,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "historical_r23_tiny_sha256": sha256_file(
            root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
        ),
        "source_alignment": audit["source_alignment"],
        "condition_identifiability": audit[
            "condition_identifiability"
        ],
        "direct_controls": _direct_table(audit),
        "diffusion_variants": _variant_table(audit),
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    summary_path = root / "reports/phase3_14b_r241_summary.json"
    write_json_once(summary_path, summary)

    report_path = root / "reports/phase3_14b_r241_report.md"
    if report_path.exists():
        raise RuntimeError(f"refusing to replace {report_path}")
    report_path.write_text(_markdown(summary, audit))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
