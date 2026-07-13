#!/usr/bin/env python3
"""Finalize Phase3.14b-r2.4.2 train-only frozen-prior pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ccda_phase3.phase314b_r23_diagnostics import write_json_once
from ccda_phase3.phase314b_r242_frozen_prior import PHASE


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r242_preflight_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    preflight_path = Path(args.preflight_report)
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    preflight_path = preflight_path.resolve()
    try:
        preflight_relative = preflight_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError("preflight report must remain inside repository") from exc
    pilot_path = root / "reports/phase3_14b_r242_pilot_summary.json"
    preflight = load_json(preflight_path)
    pilot = load_json(pilot_path)

    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.4.2 preflight did not pass")
    if pilot.get("verdict") != "PASS":
        raise RuntimeError("r2.4.2 pilot did not complete")
    if pilot.get("preflight_report") != preflight_relative:
        raise RuntimeError("pilot/preflight provenance mismatch")
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
        if bool(pilot.get(key)):
            raise RuntimeError(f"forbidden boundary flag is true: {key}")
    if pilot.get("selected_configuration") is not None:
        raise RuntimeError("r2.4.2 selected a formal configuration")

    unique_compact = {}
    for name, value in pilot["unique_free_variants"].items():
        unique_compact[name] = {
            "pass": bool(value["pass"]),
            "prior_policy": value["variant"]["prior_policy"],
            "prior_hidden_dim": int(
                value["variant"]["prior_hidden_dim"]
            ),
            "residual_hidden_dim": int(
                value["variant"]["residual_hidden_dim"]
            ),
            "prior_drift_ratio": float(value["prior_drift_ratio"]),
            "source_pass_fraction": float(
                value["evaluations"]["true"]["source_pass_fraction"]
            ),
            "z_mse": float(
                value["evaluations"]["true"]["aggregate"]["metrics"]["z_mse"]
            ),
            "ordered_rmse_p95": float(
                value["evaluations"]["true"]["aggregate"]["metrics"][
                    "ordered_rmse_p95"
                ]
            ),
            "condition_effect": bool(
                value["condition_effect"]["condition_effect_supported"]
            ),
        }

    paired_compact = {}
    for name, value in pilot["paired_low_mid_variants"].items():
        paired_compact[name] = {
            "denoising_pass": bool(value["pass"]),
            "branch_audit_pass": bool(
                value["paired_branch_audit"]["pass"]
            ),
            "combined_pass": bool(
                value["pass"]
                and value["paired_branch_audit"]["pass"]
            ),
            "prior_drift_ratio": float(value["prior_drift_ratio"]),
            "z_mse": float(
                value["evaluations"]["true"]["aggregate"]["metrics"]["z_mse"]
            ),
            "ordered_rmse_p95": float(
                value["evaluations"]["true"]["aggregate"]["metrics"][
                    "ordered_rmse_p95"
                ]
            ),
            "own_target_closer_fraction": float(
                value["paired_branch_audit"][
                    "own_target_closer_fraction"
                ]
            ),
            "branch_delta_cosine_p50": float(
                value["paired_branch_audit"][
                    "branch_delta_cosine"
                ]["p50"]
            ),
            "high_timestep_diagnostic_gate": bool(
                value["high_timestep_diagnostic"]["gate_pass"]
            ),
        }

    summary = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only factorized frozen-prior/width pilot completed; "
            "PASS does not mean formal model repair"
        ),
        "root_cause": pilot["root_cause"],
        "secondary_mechanism": pilot.get("secondary_mechanism"),
        "next_stage": pilot["next_stage"],
        "train_only_recommendation": pilot.get(
            "train_only_recommendation"
        ),
        "selected_configuration": None,
        "preflight_report": preflight_relative,
        "resumed_after_schema_block": bool(
            pilot.get("resumed_after_schema_block")
        ),
        "result_schema_contract": preflight.get(
            "result_schema_contract"
        ),
        "r241_corrected_interpretation": pilot[
            "r241_corrected_interpretation"
        ],
        "direct_width_seed_stability": pilot[
            "direct_width_seed_stability"
        ],
        "unique_free_variants": unique_compact,
        "unique_advancing_variants": pilot[
            "unique_advancing_variants"
        ],
        "paired_low_mid_variants": paired_compact,
        "dataset": pilot["dataset"],
        "repository": pilot["repository"],
        "cache_sha256": pilot["repository"]["cache_sha256"],
        "frozen_contract_sha256": pilot["repository"][
            "frozen_contract_sha256"
        ],
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
    write_json_once(
        root / "reports/phase3_14b_r242_summary.json",
        summary,
    )

    lines = [
        "# Phase3.14b-r2.4.2 Frozen-Prior / Factorized-Width Pilot",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS` (diagnostic pilot completed, not formal repair)",
        f"- Root cause: `{summary['root_cause']}`",
        (
            "- Secondary mechanism: "
            f"`{summary.get('secondary_mechanism')}`"
        ),
        (
            "- Train-only recommendation: "
            f"`{summary.get('train_only_recommendation')}`"
        ),
        "- Selected configuration: `None`",
        f"- Next: `{summary['next_stage']}`",
        "",
        "## Corrected r2.4.1 interpretation",
        "",
        (
            "- Classifier precedence bug supported: "
            f"`{summary['r241_corrected_interpretation']['classifier_precedence_bug_supported']}`"
        ),
        (
            "- Corrected primary hypothesis: "
            f"`{summary['r241_corrected_interpretation']['corrected_primary_hypothesis']}`"
        ),
        "",
        "## Direct prior width stability",
        "",
    ]
    for width, value in summary["direct_width_seed_stability"].items():
        lines.append(
            f"- Width {width}: pass seeds "
            f"`{value['pass_count']}/{value['seed_count']}`, "
            f"stable 2/3 = `{value['stable_2_of_3']}`"
        )
    lines.extend(
        [
            "",
            "## Unique-free variants",
            "",
        ]
    )
    for name, value in unique_compact.items():
        lines.append(
            f"- `{name}`: pass=`{value['pass']}`, "
            f"drift=`{value['prior_drift_ratio']:.6g}`, "
            f"source fraction=`{value['source_pass_fraction']:.3f}`, "
            f"z MSE=`{value['z_mse']:.6g}`"
        )
    lines.extend(
        [
            "",
            "## Paired low/mid-noise variants",
            "",
        ]
    )
    if not paired_compact:
        lines.append("- Not started because no unique-free variant advanced.")
    else:
        for name, value in paired_compact.items():
            lines.append(
                f"- `{name}`: combined pass=`{value['combined_pass']}`, "
                f"own-target closer=`{value['own_target_closer_fraction']:.3f}`, "
                f"branch cosine p50=`{value['branch_delta_cosine_p50']:.3f}`"
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
                "A PASS verdict means only that the train-only diagnostic "
                "pilot completed."
            ),
        ]
    )
    report_path = root / "reports/phase3_14b_r242_report.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
