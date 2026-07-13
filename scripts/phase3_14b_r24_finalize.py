#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.4 train-only noisy-skip pilot."""
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
from ccda_phase3.phase314b_r24_noisy_skip import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R23_TINY_SHA256,
    PHASE,
    assert_only_allowed_worktree_paths,
    classify_pilot,
    sha256_file,
)


def _gate_table(pilot: Mapping[str, Any]) -> Mapping[str, Any]:
    output = {}
    for stage, runs in pilot["stages"].items():
        output[stage] = {}
        for name, run in runs.items():
            output[stage][name] = {
                key: bool(value["gate_pass"])
                for key, value in run["evaluations"].items()
            }
            output[stage][name]["jvp_t50"] = bool(
                run["jvp"]["t50_gate_pass"]
            )
    return output


def _markdown(summary: Mapping[str, Any], pilot: Mapping[str, Any]) -> str:
    lines = [
        "# Phase3.14b-r2.4 Train-Only Timestep-Conditioned Noisy-Skip Pilot",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS` (train-only pilot completed, not formal model repair)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Train-only recommendation: `{summary['train_only_recommendation']}`",
        "- Selected configuration: `None`",
        f"- Next: `{summary['next_stage']}`",
        "",
        "## Stage gates",
        "",
    ]
    for stage, runs in _gate_table(pilot).items():
        lines.append(f"### {stage}")
        lines.append("")
        for name, gates in runs.items():
            lines.append(f"- `{name}`: `{json.dumps(gates, sort_keys=True)}`")
        lines.append("")
    lines.extend(
        [
            "## Boundaries",
            "",
            "- Validation targets used: `False`",
            "- Formal test read: `False`",
            "- Formal training: `False`",
            "- Formal candidate selected: `False`",
            "- Checkpoint saved: `False`",
            "- IDM / candidate execution: `False`",
            "- Phase4 / CPS: `False`",
            "",
            "The recommendation, when present, is train-only and authorizes only a separate train-only ordered-geometry pilot.",
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
        "reports/phase3_14b_r24_preflight_summary.json",
        "reports/phase3_14b_r24_pilot_summary.json",
    )
    assert_only_allowed_worktree_paths(root, allowed)

    preflight = strict_json_load(
        root / "reports/phase3_14b_r24_preflight_summary.json"
    )
    pilot = strict_json_load(
        root / "reports/phase3_14b_r24_pilot_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("preflight did not pass")
    if pilot.get("verdict") != "PASS":
        raise RuntimeError("pilot did not complete")
    if preflight["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache hash mismatch")
    if preflight["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("contract hash mismatch")
    if preflight["historical_r23_tiny_sha256"] != EXPECTED_R23_TINY_SHA256:
        raise RuntimeError("historical report hash mismatch")
    if pilot.get("source_sha256") != preflight.get("source_sha256"):
        raise RuntimeError("r2.4 source changed after preflight")
    if pilot.get("dependency_sha256") != preflight.get("dependency_sha256"):
        raise RuntimeError("r2.4 dependency changed after preflight")
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
        if pilot.get(key) is not False:
            raise RuntimeError(f"forbidden boundary changed: {key}")
    if pilot.get("selected_configuration") not in (None, "None"):
        raise RuntimeError("formal configuration was selected")
    if (root / "checkpoints/phase3_14b_r24").exists():
        raise RuntimeError("forbidden r2.4 checkpoint directory exists")

    classification = classify_pilot(pilot)
    for key in (
        "root_cause",
        "next_stage",
        "train_only_recommendation",
    ):
        if pilot.get(key) != classification.get(key):
            raise RuntimeError(f"pilot classification mismatch: {key}")

    summary = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only noisy-skip pilot completed; no formal model repair "
            "and no formal candidate selection"
        ),
        "root_cause": classification["root_cause"],
        "next_stage": classification["next_stage"],
        "train_only_recommendation": classification[
            "train_only_recommendation"
        ],
        "selected_configuration": None,
        "repository": repository,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "historical_r23_tiny_sha256": sha256_file(
            root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
        ),
        "analytic_formula_oracle": pilot["analytic_formula_oracle"],
        "dependency_sha256": pilot["dependency_sha256"],
        "gates": _gate_table(pilot),
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
    summary_path = root / "reports/phase3_14b_r24_summary.json"
    write_json_once(summary_path, summary)
    report_path = root / "reports/phase3_14b_r24_report.md"
    if report_path.exists():
        raise RuntimeError(f"refusing to replace {report_path}")
    report_path.write_text(_markdown(summary, pilot))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
