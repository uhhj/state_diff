#!/usr/bin/env python3
"""Assign the Phase3.14b-r2.1 validity/geometry root cause."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ccda_phase3.phase314a_contract import (
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_r21_contract import (
    CALIBRATED_GT_VAL_MIN,
    ORIGINAL_GT_TRAIN_MIN,
    ORIGINAL_GT_VAL_MIN,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--audit",
        default="reports/phase3_14b_r21_audit_summary.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r21_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r21_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    audit = strict_json_load(root / args.audit)
    if audit.get("verdict") != "PASS":
        raise RuntimeError("r2.1 audit did not complete")
    if audit.get("formal_test_read") is not False:
        raise RuntimeError("formal test was read during r2.1")

    references = audit["reference_metrics"]
    original_train = float(
        references["train_gt"]["original"]["sample_validity_rate"]
    )
    original_val = float(
        references["validation_gt"]["original"][
            "sample_validity_rate"
        ]
    )
    original_repeat = float(
        references["last_repeat_validation"]["original"][
            "sample_validity_rate"
        ]
    )
    original_deterministic = float(
        references["deterministic_validation"]["original"][
            "sample_validity_rate"
        ]
    )
    calibrated_val = float(
        references["validation_gt"]["calibrated"][
            "sample_validity_rate"
        ]
    )
    calibrated_repeat = float(
        references["last_repeat_validation"]["calibrated"][
            "sample_validity_rate"
        ]
    )
    calibrated_deterministic = float(
        references["deterministic_validation"]["calibrated"][
            "sample_validity_rate"
        ]
    )

    original_contract_miscalibrated = bool(
        original_train < ORIGINAL_GT_TRAIN_MIN
        or original_val < ORIGINAL_GT_VAL_MIN
        or original_repeat < ORIGINAL_GT_VAL_MIN
    )
    calibrated_contract_supported = bool(
        calibrated_val >= CALIBRATED_GT_VAL_MIN
        and calibrated_repeat >= CALIBRATED_GT_VAL_MIN
    )

    v_runs = [
        run
        for run in audit["runs"]
        if run["repair_config"] == "v_prediction_cosine"
    ]
    if len(v_runs) != 3:
        raise RuntimeError("expected three v-prediction runs")
    v_calibrated_validity = np.asarray(
        [
            run["calibrated_validity"]["sample_validity_rate"]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_coordinate_validity = np.asarray(
        [
            run["calibrated_validity"][
                "coordinate_validity_rate"
            ]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_segment_validity = np.asarray(
        [
            run["calibrated_validity"][
                "segment_score_validity_rate"
            ]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_ordered_rmse = np.asarray(
        [
            run["geometry"]["final_ordered_rmse_mean"]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_chamfer = np.asarray(
        [
            run["geometry"]["final_chamfer_mean"]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_permutation_gap = np.asarray(
        [
            run["geometry"]["permutation_gap_mean"]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_stretch_p95 = np.asarray(
        [
            run["geometry"][
                "max_segment_ratio_to_train_center_p95"
            ]
            for run in v_runs
        ],
        dtype=np.float64,
    )
    v_inversion = np.asarray(
        [
            run["geometry"][
                "nearest_index_inversion_rate_mean"
            ]
            for run in v_runs
        ],
        dtype=np.float64,
    )

    ordered_geometry_failure = bool(
        np.median(v_calibrated_validity) < 0.50
        or np.median(v_segment_validity) < 0.50
        or np.median(v_stretch_p95) >= 4.0
        or np.median(v_permutation_gap) >= 0.02
    )
    pointset_ordering_gap_supported = bool(
        np.median(v_ordered_rmse)
        >= 1.5 * max(np.median(v_chamfer), 1e-12)
        and np.median(v_permutation_gap) >= 0.02
    )

    evidence = {
        "original_contract_miscalibrated": original_contract_miscalibrated,
        "calibrated_contract_supported": calibrated_contract_supported,
        "ordered_geometry_failure": ordered_geometry_failure,
        "pointset_ordering_gap_supported": (
            pointset_ordering_gap_supported
        ),
        "original_train_gt_validity": original_train,
        "original_validation_gt_validity": original_val,
        "original_last_repeat_validity": original_repeat,
        "original_deterministic_validity": original_deterministic,
        "calibrated_validation_gt_validity": calibrated_val,
        "calibrated_last_repeat_validity": calibrated_repeat,
        "calibrated_deterministic_validity": calibrated_deterministic,
        "v_prediction_calibrated_validity_median": float(
            np.median(v_calibrated_validity)
        ),
        "v_prediction_coordinate_validity_median": float(
            np.median(v_coordinate_validity)
        ),
        "v_prediction_segment_validity_median": float(
            np.median(v_segment_validity)
        ),
        "v_prediction_ordered_rmse_median": float(
            np.median(v_ordered_rmse)
        ),
        "v_prediction_chamfer_median": float(
            np.median(v_chamfer)
        ),
        "v_prediction_permutation_gap_median": float(
            np.median(v_permutation_gap)
        ),
        "v_prediction_segment_stretch_p95_median": float(
            np.median(v_stretch_p95)
        ),
        "v_prediction_nearest_inversion_median": float(
            np.median(v_inversion)
        ),
    }

    if (
        original_contract_miscalibrated
        and calibrated_contract_supported
        and ordered_geometry_failure
    ):
        verdict = "PASS"
        root_cause = (
            "phase314b_r21_contract_miscalibration_and_ordered_"
            "geometry_failure_supported"
        )
        next_step = (
            "Phase3.14b-r2.2: freeze a train-only family-wise validity "
            "contract and add ordered-cable geometry repair to "
            "v_prediction_cosine; keep test and IDM blocked."
        )
    elif (
        original_contract_miscalibrated
        and calibrated_contract_supported
        and not ordered_geometry_failure
    ):
        verdict = "PASS"
        root_cause = (
            "phase314b_r21_validity_contract_miscalibration_supported"
        )
        next_step = (
            "Phase3.14b-r2.2: re-adjudicate the existing r2 checkpoints "
            "under the train-only calibrated contract before any retraining "
            "or formal test."
        )
    elif (
        not original_contract_miscalibrated
        and ordered_geometry_failure
    ):
        verdict = "PASS"
        root_cause = (
            "phase314b_r21_ordered_cable_geometry_failure_supported"
        )
        next_step = (
            "Phase3.14b-r2.2: targeted v-prediction ordered-geometry "
            "repair using validation only; do not loosen the validity gate."
        )
    else:
        verdict = "FAIL"
        root_cause = (
            "phase314b_r21_validity_geometry_root_cause_inconclusive"
        )
        next_step = (
            "Expand validation-only candidate diagnostics; keep retraining, "
            "formal test, IDM, and Phase4 blocked."
        )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "evidence": evidence,
        "next_step": next_step,
        "formal_test_read": False,
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r2.1 Final Report",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Next step: `{next_step}`",
        "",
        "| Evidence | Value |",
        "|---|---:|",
    ]
    for key, value in evidence.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "- Formal test read: `False`",
        "- DDPM retraining: `False`",
        "- IDM: `False`",
        "- Candidate action execution: `False`",
        "- Phase4/CPS: `False`",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (
        root / "reports/phase3_14b_r21_no_phase4_confirmation.md"
    ).write_text(
        "# Phase3.14b-r2.1 No Phase4 Confirmation\n\n"
        "- DDPM retraining: `False`\n"
        "- Formal test read: `False`\n"
        "- IDM training: `False`\n"
        "- Candidate action execution: `False`\n"
        "- Phase4: `False`\n"
        "- CPS: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
