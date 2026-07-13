#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.3.2 train-only denoiser isolation."""
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
from ccda_phase3.phase314b_r232_controls import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_R23_TINY_SHA256,
    PHASE,
    assert_only_allowed_worktree_paths,
    classify_controls,
    sha256_file,
)


def _gate_map(controls: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {}
    for name, run in controls["runs"].items():
        if run["control_kind"] == "random_stream_minibatch":
            result[name] = {
                "heldout": bool(run["heldout"]["gate_pass"]),
            }
        else:
            result[name] = {
                key: bool(value["gate_pass"])
                for key, value in run["evaluations"].items()
            }
    return result


def _report_markdown(summary: Mapping[str, Any], controls: Mapping[str, Any]) -> str:
    gates = _gate_map(controls)
    lines = [
        "# Phase3.14b-r2.3.2 Single-Row Random-Noise Denoiser Isolation",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{summary['verdict']}` (diagnosis completed, not model repair)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Selected configuration: `{summary['selected_configuration']}`",
        f"- Next: `{summary['next_stage']}`",
        "",
        "## Core checks",
        "",
        f"- Oracle v/x0 parity: `{controls['oracle_parity']['pass']}`",
        f"- Timestep embedding audit: `{controls['timestep_embedding_audit']['pass']}`",
        f"- Oracle v max abs: `{controls['oracle_parity']['v_target_max_abs']}`",
        f"- Oracle x0 max abs: `{controls['oracle_parity']['x0_reconstruction_max_abs']}`",
        "",
        "## Control gates",
        "",
    ]
    for name, values in gates.items():
        lines.append(f"- `{name}`: `{json.dumps(values, sort_keys=True)}`")
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
            "A PASS verdict means only that the train-only diagnostic evidence chain completed.",
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
        "reports/phase3_14b_r232_preflight_summary.json",
        "reports/phase3_14b_r232_controls_summary.json",
    )
    assert_only_allowed_worktree_paths(root, allowed)
    preflight = strict_json_load(
        root / "reports/phase3_14b_r232_preflight_summary.json"
    )
    controls = strict_json_load(
        root / "reports/phase3_14b_r232_controls_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("preflight did not pass")
    if controls.get("verdict") != "PASS":
        raise RuntimeError("controls did not complete")
    if preflight["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("preflight cache hash mismatch")
    if preflight["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("preflight contract hash mismatch")
    if preflight["historical_r23_tiny_sha256"] != EXPECTED_R23_TINY_SHA256:
        raise RuntimeError("historical report hash mismatch")
    if controls.get("source_sha256") != preflight.get("source_sha256"):
        raise RuntimeError("r2.3.2 source hashes changed after preflight")
    if controls.get("formal_test_read") is not False:
        raise RuntimeError("formal test was accessed")
    if controls.get("validation_targets_used") is not False:
        raise RuntimeError("validation targets were used")
    if controls.get("checkpoint_saved") is not False:
        raise RuntimeError("r2.3.2 checkpoint was saved")
    if (root / "checkpoints/phase3_14b_r232").exists():
        raise RuntimeError("forbidden r2.3.2 checkpoint directory exists")

    classification = classify_controls(controls)
    if controls.get("root_cause") != classification["root_cause"]:
        raise RuntimeError("control classification mismatch")
    if controls.get("next_stage") != classification["next_stage"]:
        raise RuntimeError("control next-stage mismatch")

    summary = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only diagnostic completed; no model repair",
        "root_cause": classification["root_cause"],
        "next_stage": classification["next_stage"],
        "selected_configuration": None,
        "repository": repository,
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "historical_r23_tiny_sha256": sha256_file(
            root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
        ),
        "oracle_parity": controls["oracle_parity"],
        "timestep_embedding_audit": controls["timestep_embedding_audit"],
        "gates": _gate_map(controls),
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
    summary_path = root / "reports/phase3_14b_r232_summary.json"
    write_json_once(summary_path, summary)
    report_path = root / "reports/phase3_14b_r232_report.md"
    if report_path.exists():
        raise RuntimeError(f"refusing to replace {report_path}")
    report_path.write_text(_report_markdown(summary, controls))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
