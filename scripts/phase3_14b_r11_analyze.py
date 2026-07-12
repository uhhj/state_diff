#!/usr/bin/env python3
"""Assign the final Phase3.14b-r1 root cause after exact scheduler repair."""
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
        default="reports/phase3_14b_r11_audit_summary.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r11_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r11_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    audit = strict_json_load(root / args.audit)
    if audit.get("verdict") != "PASS":
        raise RuntimeError("exact scheduler audit did not complete")
    gates = audit["gates"]

    exact_equivalence = bool(
        gates["exact_prev_allclose"]
        and gates["exact_pred_x0_allclose"]
        and float(gates["exact_prev_max_relative_rms"]) <= 1e-6
    )
    standardization = (
        float(gates["standardization_roundtrip_max_abs"]) <= 1e-6
    )
    true_epsilon = (
        float(gates["scheduler_true_epsilon_x0_max_abs"]) <= 5e-3
    )
    amplification = (
        float(gates["terminal_x0_error_amplification"]) >= 500.0
    )
    high_low = (
        float(gates["ema_high_to_low_x0_mse_ratio"]) >= 100.0
    )
    t10 = float(gates["ema_partial_t10_chamfer"])
    t99 = float(gates["ema_partial_t99_chamfer"])
    partial_separation = t99 >= max(5.0 * t10, t10 + 0.5)
    stochastic = float(gates["ema_stochastic_final_chamfer"])
    posterior_mean = float(
        gates["ema_posterior_mean_final_chamfer"]
    )
    raw_stochastic = float(gates["raw_stochastic_final_chamfer"])
    ema_raw_both_fail = (
        stochastic > 1.0 and raw_stochastic > 1.0
    )
    posterior_noise_not_primary = (
        stochastic <= 2.0 * max(posterior_mean, 1e-12)
    )

    evidence = {
        "standardization_contract_pass": standardization,
        "true_epsilon_oracle_pass": true_epsilon,
        "exact_scheduler_equivalence_pass": exact_equivalence,
        "terminal_amplification_large": amplification,
        "high_timestep_x0_error_dominates": high_low,
        "partial_t99_much_worse_than_t10": partial_separation,
        "ema_and_raw_both_fail": ema_raw_both_fail,
        "posterior_noise_not_primary": posterior_noise_not_primary,
        "legacy_algebraic_difference_nonzero": (
            float(gates["legacy_prev_max_abs"]) > 0.0
        ),
    }

    if not standardization:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r11_standardization_contract_failed"
        )
        next_step = "Repair the derived normalization contract."
    elif not true_epsilon:
        verdict = "FAIL"
        root_cause = "phase314b_r11_true_epsilon_oracle_failed"
        next_step = "Inspect the installed scheduler and add-noise contract."
    elif not exact_equivalence:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r11_exact_scheduler_equivalence_failed"
        )
        next_step = (
            "Inspect device, dtype, variance random stream, and installed "
            "Diffusers source before any retraining."
        )
    elif (
        amplification
        and high_low
        and partial_separation
        and ema_raw_both_fail
        and posterior_noise_not_primary
    ):
        verdict = "PASS"
        root_cause = (
            "phase314b_r1_cosine_epsilon_terminal_snr_instability_supported"
        )
        next_step = (
            "Phase3.14b-r2 targeted objective/schedule repair using only "
            "MLP-DDPM paper_state before any full-matrix rerun."
        )
    else:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r11_sampling_scale_root_cause_inconclusive"
        )
        next_step = (
            "Expand exact per-step diagnostics; keep retraining and later "
            "phases blocked."
        )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "evidence": evidence,
        "gates": gates,
        "next_step": next_step,
        "ddpm_retraining": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r1.1 Final Report",
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
        "No DDPM retraining, IDM, candidate action execution, Phase4, "
        "or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (
        root
        / "reports/phase3_14b_r11_no_phase4_confirmation.md"
    ).write_text(
        "# Phase3.14b-r1.1 No Phase4 Confirmation\n\n"
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
