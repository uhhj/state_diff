#!/usr/bin/env python3
"""Assign a scientific root cause from the Phase3.14b-r1 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccda_phase3.phase314a_contract import (
    strict_json_dump,
    strict_json_load,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--audit",
        default="reports/phase3_14b_r1_audit_summary.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r1_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r1_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    audit = strict_json_load(root / args.audit)
    if audit.get("verdict") != "PASS":
        raise RuntimeError("diagnostic audit did not complete")
    gates = audit["gates"]

    roundtrip = float(gates["standardization_roundtrip_max_abs"])
    scheduler_oracle = float(
        gates["scheduler_true_epsilon_x0_max_abs"]
    )
    manual_equivalence = float(
        gates["manual_vs_diffusers_prev_max_abs"]
    )
    amplification = float(
        gates["terminal_x0_error_amplification"]
    )
    high_low_ratio = float(
        gates["ema_high_to_low_x0_mse_ratio"]
    )
    t10 = float(gates["ema_partial_t10_chamfer"])
    t99 = float(gates["ema_partial_t99_chamfer"])
    stochastic = float(
        gates["ema_stochastic_final_chamfer"]
    )
    posterior_mean = float(
        gates["ema_posterior_mean_final_chamfer"]
    )
    raw_stochastic = float(
        gates["raw_stochastic_final_chamfer"]
    )
    ema_high = float(gates["ema_high_bin_pred_x0_mse"])
    raw_high = float(
        audit["ema_vs_raw"]["raw_high_x0_mse"]
    )

    evidence = {
        "standardization_contract_pass": roundtrip <= 1e-6,
        "scheduler_true_epsilon_pass": scheduler_oracle <= 5e-3,
        "manual_scheduler_equivalence_pass": (
            manual_equivalence <= 5e-5
        ),
        "terminal_amplification_large": amplification >= 500.0,
        "high_timestep_error_dominates": high_low_ratio >= 100.0,
        "partial_high_snr_better_than_terminal": (
            t99 >= max(5.0 * t10, t10 + 0.5)
        ),
        "posterior_noise_primary": (
            stochastic >= 2.0 * max(posterior_mean, 1e-12)
            and posterior_mean <= 1.0
        ),
        "ema_only_failure": (
            ema_high >= 10.0 * max(raw_high, 1e-12)
            and raw_stochastic <= 1.0
        ),
        "low_timestep_denoising_failed": t10 > 0.5,
    }

    if not evidence["standardization_contract_pass"]:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r1_standardization_contract_failed"
        )
        next_step = (
            "Repair the derived future normalization contract and rebuild "
            "only the derived Phase3.14 cache after explicit provenance review."
        )
    elif (
        not evidence["scheduler_true_epsilon_pass"]
        or not evidence["manual_scheduler_equivalence_pass"]
    ):
        verdict = "FAIL"
        root_cause = "phase314b_r1_scheduler_implementation_failed"
        next_step = (
            "Repair the scheduler/step implementation and rerun this audit "
            "with the existing checkpoint before any retraining."
        )
    elif evidence["ema_only_failure"]:
        verdict = "FAIL"
        root_cause = "phase314b_r1_ema_checkpoint_path_failed"
        next_step = (
            "Repair EMA update/loading, then rerun selected-checkpoint "
            "sampling without retraining the full matrix."
        )
    elif (
        evidence["terminal_amplification_large"]
        and evidence["high_timestep_error_dominates"]
        and evidence["partial_high_snr_better_than_terminal"]
    ):
        verdict = "PASS"
        root_cause = (
            "phase314b_r1_cosine_epsilon_terminal_snr_instability_supported"
        )
        next_step = (
            "Phase3.14b-r2 targeted objective/schedule repair on the MLP "
            "paper-state model only; do not rerun the full 12-run matrix yet."
        )
    elif evidence["posterior_noise_primary"]:
        verdict = "PASS"
        root_cause = (
            "phase314b_r1_reverse_variance_instability_supported"
        )
        next_step = (
            "Phase3.14b-r2 deterministic DDIM/posterior-mean sampling "
            "comparison before objective retraining."
        )
    elif evidence["low_timestep_denoising_failed"]:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r1_model_denoising_failed_across_snr"
        )
        next_step = (
            "Audit training loss, target construction, and model capacity; "
            "do not proceed to IDM."
        )
    else:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r1_sampling_scale_root_cause_inconclusive"
        )
        next_step = (
            "Expand diagnostic queries and inspect per-dimension traces; "
            "do not retrain or proceed to IDM."
        )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "evidence": evidence,
        "gates": gates,
        "next_step": next_step,
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r1 Final Report",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Next step: `{next_step}`",
        "",
        "| Evidence | Result |",
        "|---|---:|",
    ]
    for key, value in evidence.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "No retraining, IDM, candidate action execution, Phase4, or CPS "
        "was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (
        root
        / "reports/phase3_14b_r1_no_phase4_confirmation.md"
    ).write_text(
        "# Phase3.14b-r1 No Phase4 Confirmation\n\n"
        "- DDPM retraining: `False`\n"
        "- IDM training: `False`\n"
        "- Candidate action execution: `False`\n"
        "- Phase4: `False`\n"
        "- CPS: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
