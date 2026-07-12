#!/usr/bin/env python3
"""Diagnose existing r2.2 pilot checkpoints without new candidate training."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.phase314b_r22_contract import GEOMETRY_CONFIGS, load_self_hashed_json
from ccda_phase3.phase314b_r22_geometry import (
    contract_from_json,
    normalizers_from_json,
)
from ccda_phase3.phase314b_r23_diagnostics import (
    DIRECT_TIMESTEPS,
    EXPECTED_PAIRED_CONDITIONS,
    GRADIENT_TIMESTEPS,
    PHASE,
    PILOT_CONFIGS,
    TRACE_TIMESTEPS,
    direct_x0_prediction,
    failure_decomposition,
    gradient_audit,
    load_ema_model,
    load_pilot_runs,
    load_verified_inputs,
    paired_selection_metadata,
    posterior_mean_trace,
    require_repository_state,
    scheduler_x0_parity,
    select_balanced_paired_rows,
    source_sha256,
    verify_checkpoint,
    write_csv_once,
    write_json_once,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--validation-rows", type=int, default=64)
    parser.add_argument("--trace-rows", type=int, default=16)
    parser.add_argument("--gradient-rows", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    require_repository_state(root, require_clean=False)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("r2.3 checkpoint diagnosis requires CUDA")
    device = torch.device("cuda")
    seed_all(31460)

    arrays, _, x_raw, x_standardizer, train, _, _, validation = (
        load_verified_inputs(root)
    )
    validation_rows = select_balanced_paired_rows(
        arrays,
        validation,
        args.validation_rows,
    )
    trace_rows = select_balanced_paired_rows(
        arrays,
        validation_rows,
        args.trace_rows,
    )
    gradient_rows_index = select_balanced_paired_rows(
        arrays,
        train,
        args.gradient_rows,
    )
    if len(validation_rows) != args.validation_rows:
        raise RuntimeError("validation row request was not satisfied exactly")
    if len(trace_rows) != args.trace_rows:
        raise RuntimeError("trace row request was not satisfied exactly")
    if len(gradient_rows_index) != args.gradient_rows:
        raise RuntimeError("gradient row request was not satisfied exactly")

    x_z = x_standardizer.transform(x_raw)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_standardizer = future_standardizer(arrays)
    y_z = y_standardizer.transform(y_raw)
    active_np = np.asarray(arrays["future_active"], dtype=bool)
    if active_np.shape != (4, 87) or not active_np[:, :48].all():
        raise RuntimeError("active mask does not preserve all ordered bead XY")

    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    physical = contract_from_json(frozen["physical_contract"])
    normalizers = normalizers_from_json(frozen["geometry_normalizers"])
    pilot_runs = load_pilot_runs(root)
    repair_config = REPAIR_CONFIGS["v_prediction_cosine"]

    future_mean = torch.from_numpy(
        np.asarray(arrays["future_mean"], dtype=np.float32)
    ).to(device)
    future_scale = torch.from_numpy(
        np.asarray(arrays["future_scale"], dtype=np.float32)
    ).to(device)
    active = torch.from_numpy(active_np).to(device)

    condition_validation = torch.from_numpy(
        x_z[validation_rows]
    ).float().to(device)
    clean_validation = torch.from_numpy(
        y_z[validation_rows]
    ).float().to(device)
    target_validation = y_raw[validation_rows]

    condition_trace = torch.from_numpy(x_z[trace_rows]).float().to(device)
    clean_trace = torch.from_numpy(y_z[trace_rows]).float().to(device)
    target_trace = y_raw[trace_rows]

    condition_gradient = torch.from_numpy(
        x_z[gradient_rows_index]
    ).float().to(device)
    clean_gradient = torch.from_numpy(
        y_z[gradient_rows_index]
    ).float().to(device)
    raw_gradient = torch.from_numpy(
        y_raw[gradient_rows_index]
    ).float().to(device)

    direct_rows = []
    trace_rows_csv = []
    gradient_rows_csv = []
    config_reports = {}
    parity_values = []

    for config_index, config_name in enumerate(PILOT_CONFIGS):
        run = pilot_runs[config_name]
        payload = verify_checkpoint(root, run, frozen)
        model = load_ema_model(payload, device=device)
        geometry_config = GEOMETRY_CONFIGS[config_name]
        scheduler = make_repair_scheduler(repair_config)

        config_report = {
            "checkpoint": run["checkpoint"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "direct_x0": {},
            "truth_noised_trace": {},
            "pure_noise_trace": {},
        }

        for timestep in DIRECT_TIMESTEPS:
            parity = scheduler_x0_parity(
                scheduler=scheduler,
                repair_config=repair_config,
                device=device,
                timestep=timestep,
                seed=31500 + config_index * 100 + timestep,
            )
            parity_values.append(parity)
            _, predicted_raw = direct_x0_prediction(
                model=model,
                scheduler=scheduler,
                repair_config=repair_config,
                condition_z=condition_validation,
                clean_z=clean_validation,
                active_mask=active,
                timestep=timestep,
                seed=31600 + timestep,
                future_mean=future_mean,
                future_scale=future_scale,
            )
            metrics = failure_decomposition(
                predicted_raw[None],
                target_validation,
                physical,
            )
            config_report["direct_x0"][str(timestep)] = metrics
            direct_rows.append(
                {
                    "config": config_name,
                    "timestep": timestep,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float, bool))
                    },
                }
            )

        for start_mode in ("truth_noised", "pure_noise"):
            captures = posterior_mean_trace(
                model=model,
                scheduler=scheduler,
                repair_config=repair_config,
                condition_z=condition_trace,
                clean_z=clean_trace,
                active_mask=active,
                start_timestep=99,
                seed=31700 + config_index,
                future_mean=future_mean,
                future_scale=future_scale,
                capture_timesteps=TRACE_TIMESTEPS,
                start_mode=start_mode,
            )
            mode_report = {}
            for timestep in TRACE_TIMESTEPS:
                metrics = failure_decomposition(
                    captures[timestep][None],
                    target_trace,
                    physical,
                )
                mode_report[str(timestep)] = metrics
                trace_rows_csv.append(
                    {
                        "config": config_name,
                        "start_mode": start_mode,
                        "timestep": timestep,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if isinstance(value, (int, float, bool))
                        },
                    }
                )
            config_report[f"{start_mode}_trace"] = mode_report

        for timestep in GRADIENT_TIMESTEPS:
            rows = gradient_audit(
                model=model,
                scheduler=scheduler,
                repair_config=repair_config,
                geometry_config=geometry_config,
                condition_z=condition_gradient,
                clean_z=clean_gradient,
                clean_raw=raw_gradient,
                active_mask=active,
                future_mean=future_mean,
                future_scale=future_scale,
                normalizers=normalizers,
                timestep=timestep,
                seed=31800 + timestep,
            )
            for row in rows:
                gradient_rows_csv.append({"config": config_name, **row})

        config_reports[config_name] = config_report
        del model
        torch.cuda.empty_cache()

    ordered_rows = [
        row
        for row in gradient_rows_csv
        if row["component"] == "ordered_xy"
        and row["config"] in {"ordered_edge", "ordered_edge_temporal"}
    ]
    ordered_ratio_median = float(
        np.median(
            [row["weighted_to_v_gradient_ratio"] for row in ordered_rows]
        )
    )

    candidate_direct = config_reports["ordered_edge"]["direct_x0"]
    segment_element_p95 = float(
        candidate_direct["50"]["segment_element_score_p95"]
    )
    segment_family_p95 = float(
        candidate_direct["50"]["segment_family_score_p95"]
    )
    familywise_tail_supported = bool(
        segment_family_p95
        > max(
            5.0 * segment_element_p95,
            float(physical.segment_score_threshold),
        )
    )

    truth_start = config_reports["ordered_edge"]["truth_noised_trace"]["99"]
    truth_end = config_reports["ordered_edge"]["truth_noised_trace"]["0"]
    reverse_accumulation_supported = bool(
        truth_start["segment_family_failure_rate"] < 0.50
        and truth_end["segment_family_failure_rate"]
        > truth_start["segment_family_failure_rate"] + 0.25
    )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r23_checkpoint_diagnosis_completed",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "sampling_contract": {
            "argument_semantics": "actual_rows",
            "selection_unit": "complete_visible_seed_pair",
            "expected_conditions": list(EXPECTED_PAIRED_CONDITIONS),
            "validation": paired_selection_metadata(
                arrays,
                validation_rows,
            ),
            "trace": paired_selection_metadata(
                arrays,
                trace_rows,
            ),
            "gradient": paired_selection_metadata(
                arrays,
                gradient_rows_index,
            ),
        },
        "validation_row_count": int(len(validation_rows)),
        "trace_row_count": int(len(trace_rows)),
        "gradient_row_count": int(len(gradient_rows_index)),
        "direct_timesteps": list(DIRECT_TIMESTEPS),
        "trace_timesteps": list(TRACE_TIMESTEPS),
        "gradient_timesteps": list(GRADIENT_TIMESTEPS),
        "scheduler_parity_max_abs": float(max(parity_values)),
        "ordered_weighted_to_v_gradient_ratio_median": ordered_ratio_median,
        "familywise_tail_signature": {
            "supported": familywise_tail_supported,
            "segment_element_score_p95_t50": segment_element_p95,
            "segment_family_score_p95_t50": segment_family_p95,
        },
        "reverse_accumulation_signature": {
            "supported": reverse_accumulation_supported,
            "truth_noised_t99_segment_failure": truth_start[
                "segment_family_failure_rate"
            ],
            "truth_noised_t0_segment_failure": truth_end[
                "segment_family_failure_rate"
            ],
        },
        "configs": config_reports,
        "gradient_rows": gradient_rows_csv,
        "source_sha256": source_sha256(root),
        "formal_test_read": False,
        "new_candidate_training": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(
        root / "reports/phase3_14b_r23_checkpoint_diagnosis_summary.json",
        report,
    )
    write_csv_once(
        root / "reports/phase3_14b_r23_direct_x0.csv",
        direct_rows,
    )
    write_csv_once(
        root / "reports/phase3_14b_r23_reverse_trace.csv",
        trace_rows_csv,
    )
    write_csv_once(
        root / "reports/phase3_14b_r23_gradient_audit.csv",
        gradient_rows_csv,
    )
    print(
        json.dumps(
            {
                "scheduler_parity_max_abs": report[
                    "scheduler_parity_max_abs"
                ],
                "ordered_gradient_ratio": ordered_ratio_median,
                "familywise_tail": familywise_tail_supported,
                "reverse_accumulation": reverse_accumulation_supported,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
