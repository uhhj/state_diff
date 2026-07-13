#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.3.1 train-only control correction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ccda_phase3.phase314a_contract import sha256_file, strict_json_load
from ccda_phase3.phase314b_r23_diagnostics import (
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import (
    PHASE,
    assert_only_allowed_worktree_paths,
    classify_controls,
    source_sha256,
)


def write_text_once(path: Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"refusing to replace {target}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(
        root,
        (
            "reports/phase3_14b_r231_preflight_summary.json",
            "reports/phase3_14b_r231_controls_summary.json",
        ),
    )

    preflight_path = root / "reports/phase3_14b_r231_preflight_summary.json"
    controls_path = root / "reports/phase3_14b_r231_controls_summary.json"
    preflight = strict_json_load(preflight_path)
    controls = strict_json_load(controls_path)

    if preflight.get("phase") != PHASE or controls.get("phase") != PHASE:
        raise RuntimeError("r2.3.1 phase mismatch")
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.3.1 preflight did not pass")
    if controls.get("verdict") != "PASS":
        raise RuntimeError("r2.3.1 controls did not complete")
    if controls.get("formal_test_read") is not False:
        raise RuntimeError("formal test was accessed")
    if controls.get("validation_rows_used_by_controls") is not False:
        raise RuntimeError("validation rows were used by controls")
    if controls.get("checkpoint_saved") is not False:
        raise RuntimeError("a diagnostic checkpoint was saved")
    if (root / "checkpoints/phase3_14b_r231").exists():
        raise RuntimeError("unexpected r2.3.1 checkpoint directory")

    current_hashes = source_sha256(root)
    if controls.get("source_sha256") != current_hashes:
        raise RuntimeError("control report source hashes are stale")
    if preflight.get("source_sha256") != current_hashes:
        raise RuntimeError("preflight source hashes are stale")

    old_tiny_path = root / "reports/phase3_14b_r23_tiny_overfit_summary.json"
    expected_old_sha = preflight["r23"]["old_tiny_report_sha256"]
    if sha256_file(old_tiny_path) != expected_old_sha:
        raise RuntimeError("historical r2.3 tiny report changed")

    classification = classify_controls(controls)
    runs = controls["runs"]
    any_fixed_replay_pass = any(
        runs[name]["exact_replay_gate_pass"]
        for name in (
            "fixed_pairs_v_only_adamw",
            "fixed_pairs_v_only_adam",
            "fixed_pairs_v_only_wide",
        )
    )
    prior_capacity_conclusion_invalidated = bool(
        runs["fixed_one_row_v_only"]["exact_replay_gate_pass"]
        and any_fixed_replay_pass
    )

    summary = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "tiny-control evaluation contract corrected and train-only "
            "isolation completed; no model repair"
        ),
        "root_cause": classification["root_cause"],
        "next_stage": classification["next_stage"],
        "selected_configuration": None,
        "repository": repository,
        "historical_r23": {
            "root_cause": preflight["r23"]["root_cause"],
            "fixed_control_contract_error": (
                preflight["r23"]["old_fixed_control_contract_error"]
            ),
            "capacity_conclusion_invalidated": (
                prior_capacity_conclusion_invalidated
            ),
        },
        "control_gates": {
            "direct_one_row": runs["direct_one_row"]["gate_pass"],
            "direct_unique_free_16": (
                runs["direct_unique_free_16"]["gate_pass"]
            ),
            "fixed_one_row_v_only_exact_replay": (
                runs["fixed_one_row_v_only"]["exact_replay_gate_pass"]
            ),
            "fixed_pairs_v_only_adamw_exact_replay": (
                runs["fixed_pairs_v_only_adamw"]["exact_replay_gate_pass"]
            ),
            "fixed_pairs_v_only_adam_exact_replay": (
                runs["fixed_pairs_v_only_adam"]["exact_replay_gate_pass"]
            ),
            "fixed_pairs_v_only_wide_exact_replay": (
                runs["fixed_pairs_v_only_wide"]["exact_replay_gate_pass"]
            ),
            "fixed_pairs_r22_geometry_exact_replay": (
                runs["fixed_pairs_r22_geometry"]["exact_replay_gate_pass"]
            ),
            "random_one_row_t50": (
                runs["random_one_row"]["single_branch_t50_gate_pass"]
            ),
            "random_unique_free_16_t50": (
                runs["random_unique_free_16"][
                    "single_branch_t50_gate_pass"
                ]
            ),
            "random_paired_16_t50_diagnostic_only": (
                runs["random_paired_16"]["single_branch_t50_gate_pass"]
            ),
        },
        "pair_ambiguity": controls["pair_ambiguity"],
        "source_sha256": current_hashes,
        "formal_test_read": False,
        "validation_rows_used_by_controls": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "checkpoint_saved": False,
    }
    write_json_once(
        root / "reports/phase3_14b_r231_summary.json",
        summary,
    )

    gate_lines = "\n".join(
        f"- {name}: `{value}`"
        for name, value in summary["control_gates"].items()
    )
    report = f"""# Phase3.14b-r2.3.1 Tiny-Control Contract Correction

## Verdict

- Verdict: `PASS` (diagnostic correction completed, not model repair)
- Root cause: `{summary["root_cause"]}`
- Selected configuration: `None`
- Next: `{summary["next_stage"]}`

## Corrected Contract

The historical r2.3 fixed-noise control trained on one fixed noisy tuple but
evaluated its memorization gate with newly sampled noise. The corrected gate
uses the exact training `x_t`, timestep and noise. Fresh-noise behavior is
reported separately.

Overfit gates are target-relative and do not require every training target to
pass the population physical-validity contract.

## Control Gates

{gate_lines}

## Interpretation

- Historical r2.3 capacity conclusion invalidated:
  `{prior_capacity_conclusion_invalidated}`
- Pair input max-absolute median:
  `{controls["pair_ambiguity"]["input_max_abs"]["median"]}`
- Pair target ordered-RMSE median:
  `{controls["pair_ambiguity"]["target_ordered_rmse"]["median"]}`

## Boundaries

- Validation target rows used by controls: `False`
- Formal test read: `False`
- Formal training: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`
"""
    write_text_once(
        root / "reports/phase3_14b_r231_report.md",
        report,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
