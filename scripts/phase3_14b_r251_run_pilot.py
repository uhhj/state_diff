#!/usr/bin/env python3
"""Run Phase3.14b-r2.5.1 train-only gradient-calibration pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
from ccda_phase3.phase314b_r22_geometry import contract_from_json
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import select_unique_condition_rows
from ccda_phase3.phase314b_r241_multirow import build_labeled_bank
from ccda_phase3.phase314b_r242_frozen_prior import paired_branch_audit
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryTrainSpec,
    ReverseSamplingSpec,
    fit_geometry_scales,
    paired_full_reverse_branch_support,
    reverse_sample_pool,
    torch_contract_from_payload,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    CALIBRATED_GEOMETRY_OBJECTIVES,
    COMMON_PAIRED_PRIOR_SEED,
    COMMON_PAIRED_TRAINING_SEED,
    COMMON_REVERSE_SEED,
    COMMON_UNIQUE_PRIOR_SEED,
    COMMON_UNIQUE_TRAINING_SEED,
    EXPECTED_UNIQUE_ROWS,
    FULL_REVERSE_K,
    PAIRED_EVAL_TIMESTEPS,
    PAIRED_GEOMETRY_TIMESTEP_MAX,
    PHASE,
    PAIRED_INVERSION_SCHEMA,
    UNIQUE_EVAL_TIMESTEPS,
    GradientCalibrationSpec,
    assert_canonical_paired_row_contract,
    assert_only_allowed_worktree_paths,
    classify_pilot,
    compare_calibrated_to_control,
    fit_shared_prior_snapshot,
    paired_reverse_pool_metrics_with_inversion,
    source_sha256,
    strip_runtime_objects,
    train_calibrated_geometry_variant,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def subset(
    rows: np.ndarray,
    *,
    condition_all: torch.Tensor,
    clean_z_all: torch.Tensor,
    clean_raw_all: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    selected = np.asarray(rows, dtype=np.int64)
    index = torch.from_numpy(selected).to(condition_all.device)
    return {
        "condition_z": condition_all.index_select(0, index),
        "clean_z": clean_z_all.index_select(0, index),
        "clean_raw": clean_raw_all.index_select(0, index),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r251_preflight_summary.json",
    )
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--residual-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--evaluation-noises", type=int, default=8)
    parser.add_argument("--reverse-samples", type=int, default=FULL_REVERSE_K)
    parser.add_argument("--calibration-batches", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight_path = Path(args.preflight_report)
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    preflight_path = preflight_path.resolve()
    preflight_relative = preflight_path.relative_to(root).as_posix()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, (preflight_relative,))
    if not torch.cuda.is_available():
        raise RuntimeError("r2.5.1 pilot requires CUDA")
    device = torch.device("cuda")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.5.1 preflight did not pass")
    if preflight.get("r251_source_sha256") != source_sha256(root):
        raise RuntimeError("r2.5.1 source changed after preflight")
    resume = preflight.get("resume")
    if not isinstance(resume, dict) or resume.get("generation") != 1:
        raise RuntimeError("r2.5.1 Resume1 provenance is missing")
    if resume.get("correction") != "paired_reverse_singleton_batch_contract":
        raise RuntimeError("r2.5.1 Resume1 correction contract mismatch")

    arrays, manifest, x_raw, x_standardizer, train, fit, _, validation = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train indices contain non-train rows")
    unique_rows = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=16,
    )
    if tuple(int(value) for value in unique_rows.tolist()) != EXPECTED_UNIQUE_ROWS:
        raise RuntimeError(
            "unique-free row identity mismatch: "
            f"{unique_rows.tolist()} != {list(EXPECTED_UNIQUE_ROWS)}"
        )
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_row_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if preflight.get("paired_row_contract") != paired_row_contract:
        raise RuntimeError("paired-row contract changed after preflight")
    condition_names = np.asarray(arrays["condition_name"]).astype(str)[paired_rows]

    x_z = x_standardizer.transform(x_raw).astype(np.float32)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_standardizer = future_standardizer(arrays)
    y_z = y_standardizer.transform(y_raw).astype(np.float32)
    condition_all = torch.from_numpy(x_z).float().to(device)
    clean_z_all = torch.from_numpy(y_z).float().to(device)
    clean_raw_all = torch.from_numpy(y_raw).float().to(device)
    active = torch.from_numpy(np.asarray(arrays["future_active"], dtype=bool)).to(device)
    future_mean = torch.from_numpy(np.asarray(arrays["future_mean"], dtype=np.float32)).to(device)
    future_scale = torch.from_numpy(np.asarray(arrays["future_scale"], dtype=np.float32)).to(device)
    unique_data = subset(
        unique_rows,
        condition_all=condition_all,
        clean_z_all=clean_z_all,
        clean_raw_all=clean_raw_all,
    )
    paired_data = subset(
        paired_rows,
        condition_all=condition_all,
        clean_z_all=clean_z_all,
        clean_raw_all=clean_raw_all,
    )

    frozen = load_self_hashed_json(root / "reports/phase3_14b_r22_frozen_contract.json")
    if frozen["artifact_sha256"] != repository["frozen_contract_sha256"]:
        raise RuntimeError("frozen contract changed after repository gate")
    physical_contract = contract_from_json(frozen["physical_contract"])
    torch_contract = torch_contract_from_payload(frozen["physical_contract"])
    scales = fit_geometry_scales(y_raw[fit])
    reference_threshold = float(
        frozen["prediction_reference_contract"]["final_ordered_rmse_threshold"]
    )

    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
    noise_seeds = tuple(99000 + index for index in range(int(args.evaluation_noises)))
    unique_bank = build_labeled_bank(
        scheduler=scheduler,
        **unique_data,
        active_mask=active,
        timesteps=UNIQUE_EVAL_TIMESTEPS,
        noise_seeds=noise_seeds,
        matched_noise_across_sources=True,
    )
    paired_bank = build_labeled_bank(
        scheduler=scheduler,
        **paired_data,
        active_mask=active,
        timesteps=PAIRED_EVAL_TIMESTEPS,
        noise_seeds=noise_seeds,
        matched_noise_across_sources=True,
    )
    train_spec = GeometryTrainSpec(
        prior_steps=int(args.prior_steps),
        residual_steps=int(args.residual_steps),
        batch_size=int(args.batch_size),
        prior_learning_rate=float(args.prior_learning_rate),
        residual_learning_rate=float(args.residual_learning_rate),
    )
    calibration_spec = GradientCalibrationSpec(
        batch_count=int(args.calibration_batches),
    )

    unique_prior_snapshot = fit_shared_prior_snapshot(
        scheduler=scheduler,
        condition_z=unique_data["condition_z"],
        clean_z=unique_data["clean_z"],
        active_mask=active,
        train_spec=train_spec,
        seed=COMMON_UNIQUE_PRIOR_SEED,
    )
    unique_runtime: Dict[str, Any] = {}
    for objective in CALIBRATED_GEOMETRY_OBJECTIVES:
        unique_runtime[objective.name] = train_calibrated_geometry_variant(
            scheduler=scheduler,
            **unique_data,
            evaluation_bank=unique_bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
            torch_contract=torch_contract,
            scales=scales,
            objective=objective,
            train_spec=train_spec,
            calibration_spec=calibration_spec,
            train_timesteps=tuple(range(100)),
            geometry_timestep_max=99,
            shared_prior_snapshot=unique_prior_snapshot,
            seed=COMMON_UNIQUE_TRAINING_SEED,
        )

    advancing_names = [
        objective.name
        for objective in CALIBRATED_GEOMETRY_OBJECTIVES
        if bool(unique_runtime[objective.name].get("pass"))
    ]
    pair_ids = [value for pair in range(8) for value in (pair, pair)]
    paired_runtime: Dict[str, Any] = {}
    query_condition = paired_data["condition_z"][0::2]
    paired_targets = paired_data["clean_raw"].reshape(8, 2, 4, 87)
    pair_input_difference = float(
        torch.max(
            torch.abs(
                paired_data["condition_z"][0::2]
                - paired_data["condition_z"][1::2]
            )
        ).cpu()
    )
    if pair_input_difference > 1.0e-6:
        raise RuntimeError(f"paired deployable inputs differ: {pair_input_difference}")

    paired_prior_snapshot = None
    if advancing_names:
        paired_prior_snapshot = fit_shared_prior_snapshot(
            scheduler=scheduler,
            condition_z=paired_data["condition_z"],
            clean_z=paired_data["clean_z"],
            active_mask=active,
            train_spec=train_spec,
            seed=COMMON_PAIRED_PRIOR_SEED,
        )
    objective_by_name = {
        objective.name: objective
        for objective in CALIBRATED_GEOMETRY_OBJECTIVES
    }
    for name in advancing_names:
        if paired_prior_snapshot is None:
            raise RuntimeError("paired prior snapshot was not created")
        result = train_calibrated_geometry_variant(
            scheduler=scheduler,
            **paired_data,
            evaluation_bank=paired_bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
            torch_contract=torch_contract,
            scales=scales,
            objective=objective_by_name[name],
            train_spec=train_spec,
            calibration_spec=calibration_spec,
            train_timesteps=tuple(range(100)),
            geometry_timestep_max=PAIRED_GEOMETRY_TIMESTEP_MAX,
            shared_prior_snapshot=paired_prior_snapshot,
            seed=COMMON_PAIRED_TRAINING_SEED,
        )
        if "_true_prediction_z" not in result:
            result["one_step_branch_audit"] = {
                "pass": False,
                "failure": result.get("failure", "paired_training_incomplete"),
            }
            result["one_step_pass"] = False
            result["reverse_metrics"] = {}
            result["full_reverse_branch_support"] = {
                "pass": False,
                "failure": "paired_training_incomplete",
            }
            result["comparison_to_control"] = {
                "pass": False,
                "failure": "paired_training_incomplete",
            }
            paired_runtime[name] = result
            continue
        branch_audit = paired_branch_audit(
            predicted_z=result["_true_prediction_z"],
            bank=paired_bank,
            source_pair_ids=pair_ids,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        reverse_scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
        reverse_spec = ReverseSamplingSpec(
            sample_count=int(args.reverse_samples),
            inference_steps=100,
            seed=COMMON_REVERSE_SEED,
        )
        pool_z = reverse_sample_pool(
            model=result["_model"],
            scheduler=reverse_scheduler,
            condition_z=query_condition,
            active_mask=active,
            spec=reverse_spec,
            matched_initial_noise_across_rows=True,
        )
        reverse_metrics_raw = paired_reverse_pool_metrics_with_inversion(
            pool_z=pool_z,
            paired_target_raw=paired_targets,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
        )
        branch_support = paired_full_reverse_branch_support(
            pool_raw=reverse_metrics_raw["_pool_raw"],
            paired_target_raw=paired_data["clean_raw"].detach().cpu().numpy(),
            reference_threshold=reference_threshold,
        )
        result["one_step_branch_audit"] = branch_audit
        result["one_step_pass"] = bool(result["pass"] and branch_audit["pass"])
        result["reverse_metrics"] = {
            key: value
            for key, value in reverse_metrics_raw.items()
            if not key.startswith("_")
        }
        result["full_reverse_branch_support"] = branch_support
        paired_runtime[name] = result

    control_name = "v_only_frozen_control"
    if control_name in paired_runtime:
        control = paired_runtime[control_name]
        for name, result in paired_runtime.items():
            if name == control_name:
                result["comparison_to_control"] = {
                    "segment_score_p95_ratio": 1.0,
                    "sample_validity_delta": 0.0,
                    "best_ordered_rmse_ratio": 1.0,
                    "nearest_inversion_p95_ratio": 1.0,
                    "nearest_inversion_limit": float(
                        result["reverse_metrics"]["nearest_inversion_p95"]
                    ),
                    "tail_improved": False,
                    "ordered_preserved": True,
                    "inversion_preserved": True,
                    "pass": True,
                }
            elif not result.get("reverse_metrics"):
                result["comparison_to_control"] = {
                    "pass": False,
                    "failure": "paired_reverse_metrics_missing",
                }
            else:
                result["comparison_to_control"] = compare_calibrated_to_control(
                    candidate={"reverse_metrics": result["reverse_metrics"]},
                    control={"reverse_metrics": control["reverse_metrics"]},
                )

    unique_results = {
        name: strip_runtime_objects(value)
        for name, value in unique_runtime.items()
    }
    paired_results = {
        name: strip_runtime_objects(value)
        for name, value in paired_runtime.items()
    }
    visible_seed = np.asarray(arrays["visible_seed"]).astype(np.int64)
    pair_key = np.asarray(arrays["pair_key"]).astype(str)
    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only actual-gradient geometry calibration completed; no formal selection",
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "preflight_report": preflight_relative,
        "resume": resume,
        "source_sha256": source_sha256(root),
        "frozen_contract_sha256": frozen["artifact_sha256"],
        "geometry_scales": scales.to_json(),
        "prediction_reference_threshold": reference_threshold,
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "unique_free_rows": unique_rows.tolist(),
            "unique_free_visible_seeds": visible_seed[unique_rows].tolist(),
            "paired_rows": paired_rows.tolist(),
            "paired_visible_seeds": visible_seed[paired_rows].tolist(),
            "paired_conditions": condition_names.tolist(),
            "paired_pair_keys": pair_key[paired_rows].tolist(),
            "paired_input_max_abs_difference": pair_input_difference,
            "paired_row_contract": paired_row_contract,
        },
        "evaluation": {
            "noise_seeds": list(noise_seeds),
            "common_unique_prior_seed": COMMON_UNIQUE_PRIOR_SEED,
            "common_unique_training_seed": COMMON_UNIQUE_TRAINING_SEED,
            "common_paired_prior_seed": COMMON_PAIRED_PRIOR_SEED,
            "common_paired_training_seed": COMMON_PAIRED_TRAINING_SEED,
            "common_reverse_seed": COMMON_REVERSE_SEED,
            "unique_one_step_timesteps": list(UNIQUE_EVAL_TIMESTEPS),
            "paired_one_step_timesteps": list(PAIRED_EVAL_TIMESTEPS),
            "paired_geometry_timestep_max": PAIRED_GEOMETRY_TIMESTEP_MAX,
            "reverse_sample_count": int(args.reverse_samples),
            "reverse_inference_steps": 100,
            "high_noise_exact_branch_gate": False,
        },
        "fixed_model_contract": {
            "architecture": "factorized_analytic_x0_skip_mlp",
            "prior_hidden_dim": 512,
            "residual_hidden_dim": 512,
            "prior_policy": "strict_frozen_shared_snapshot",
            "batching": "balanced",
            "prediction_type": "v_prediction",
        },
        "gradient_calibration_contract": {
            "method": "fixed multiplier from actual residual-gradient norms",
            "formula": "lambda = target_ratio / median(||grad Lgeo|| / ||grad Lv||)",
            "batch_count": int(args.calibration_batches),
            "target_ratios": [0.10, 0.50, 1.00],
            "safe_ratio_interval": [0.02, 5.0],
            "target_tracking_factor": 4.0,
            "loss_magnitude_balancing_used": False,
        },
        "unique_shared_prior": strip_runtime_objects(unique_prior_snapshot),
        "paired_shared_prior": (
            strip_runtime_objects(paired_prior_snapshot)
            if paired_prior_snapshot is not None
            else None
        ),
        "unique_free_variants": unique_results,
        "unique_advancing_variants": advancing_names,
        "paired_variants": paired_results,
        "paired_nearest_inversion_instrumented": True,
        "paired_nearest_inversion_schema": PAIRED_INVERSION_SCHEMA,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "selected_configuration": None,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    report.update(classify_pilot(report))
    write_json_once(root / "reports/phase3_14b_r251_pilot_summary.json", report)
    print(
        json.dumps(
            {
                "root_cause": report["root_cause"],
                "train_only_recommendation": report["train_only_recommendation"],
                "unique_pass": {
                    key: bool(value.get("pass"))
                    for key, value in unique_results.items()
                },
                "paired_one_step_pass": {
                    key: bool(value.get("one_step_pass"))
                    for key, value in paired_results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
