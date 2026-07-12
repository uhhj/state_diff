#!/usr/bin/env python3
"""Finalize the r2.3 diagnosis without selecting or training a candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

from ccda_phase3.phase314a_contract import strict_json_load
from ccda_phase3.phase314b_r23_diagnostics import (
    PHASE,
    classify_diagnosis,
    require_repository_state,
    source_sha256,
    write_json_once,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=False)

    resume_preflight_path = (
        root / "reports/phase3_14b_r23_resume_preflight_summary.json"
    )
    original_preflight_path = (
        root / "reports/phase3_14b_r23_preflight_summary.json"
    )
    preflight_path = (
        resume_preflight_path
        if resume_preflight_path.is_file()
        else original_preflight_path
    )
    preflight = strict_json_load(preflight_path)
    checkpoint = strict_json_load(
        root / "reports/phase3_14b_r23_checkpoint_diagnosis_summary.json"
    )
    tiny = strict_json_load(
        root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
    )
    if any(
        report.get("formal_test_read") is not False
        for report in (preflight, checkpoint, tiny)
    ):
        raise RuntimeError("formal test access detected")
    classification = classify_diagnosis(checkpoint, tiny)

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning_of_pass": (
            "diagnostic evidence chain completed; candidate geometry remains "
            "unrepaired and no configuration is selected"
        ),
        "root_cause": classification["root_cause"],
        "next_stage": classification["next_stage"],
        "preflight_report": str(preflight_path.relative_to(root)),
        "sampling_contract": checkpoint["sampling_contract"],
        "resumed_after_blocked_sampling_contract": (
            preflight_path == resume_preflight_path
        ),
        "repository": repository,
        "evidence": {
            "scheduler_parity_max_abs": checkpoint[
                "scheduler_parity_max_abs"
            ],
            "ordered_weighted_to_v_gradient_ratio_median": checkpoint[
                "ordered_weighted_to_v_gradient_ratio_median"
            ],
            "familywise_tail_signature": checkpoint[
                "familywise_tail_signature"
            ],
            "reverse_accumulation_signature": checkpoint[
                "reverse_accumulation_signature"
            ],
            "tiny_overfit_gates": {
                name: run["gate_pass"]
                for name, run in tiny["runs"].items()
            },
        },
        "selected_configuration": None,
        "formal_test_read": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "source_sha256": source_sha256(root),
    }
    write_json_once(
        root / "reports/phase3_14b_r23_summary.json",
        report,
    )
    markdown = f"""# Phase3.14b-r2.3 Ordered-Geometry Failure Diagnosis

- Verdict: `PASS` (diagnosis completed, not model repair)
- Root cause: `{report['root_cause']}`
- Selected configuration: `None`
- Next: `{report['next_stage']}`
- Formal test: `UNREAD / BLOCKED`
- Formal training: `BLOCKED`
- IDM / candidate execution: `BLOCKED`
- Phase4 / CPS: `BLOCKED`
"""
    target = root / "reports/phase3_14b_r23_report.md"
    if target.exists():
        raise RuntimeError(f"refusing to replace {target}")
    target.write_text(markdown)


if __name__ == "__main__":
    main()
