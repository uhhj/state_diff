#!/usr/bin/env python3
"""Run Phase3.14b-r2.5.2 train-only transport-attribution pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

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
from ccda_phase3.phase314b_r241_multirow import build_labeled_bank
from ccda_phase3.phase314b_r251_gradient_calibration import (
    GradientCalibrationSpec,
    assert_canonical_paired_row_contract,
    fit_shared_prior_snapshot,
    instantiate_snapshot_model,
    train_calibrated_geometry_variant,
)
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryTrainSpec,
    fit_geometry_scales,
    torch_contract_from_payload,
)
from ccda_phase3.phase314b_r252_transport_attribution import (
    ATTRIBUTION_SCHEMA,
    COMMON_PAIRED_PRIOR_SEED,
    COMMON_PAIRED_TRAINING_SEED,
    COMMON_REVERSE_SEED,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EVALUATION_NOISE_SEEDS,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SUBMODULE_COMMIT,
    FULL_REVERSE_K,
    PAIR_TIMESTEPS,
    PER_TIMESTEP_GRADIENT_SCHEMA,
    PHASE,
    TRAJECTORY_SCHEMA,
    assert_only_allowed_worktree_paths,
    classify_attribution,
    compare_endpoint_to_r251,
    diagnostic_objectives,
    evaluate_reverse_trajectory,
    paired_one_step_attribution,
    per_timestep_gradient_audit,
    reverse_predicted_x0_trajectory,
    source_sha256,
    strip_runtime_objects,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


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


def gate_decomposition(
    result: Mapping[str, Any],
    attribution: Mapping[str, Any],
) -> Dict[str, Any]:
    evaluations = result.get("evaluations", {})
    true_value = evaluations.get("true", {}) if isinstance(evaluations, Mapping) else {}
    aggregate = true_value.get("aggregate", {}) if isinstance(true_value, Mapping) else {}
    branch_pass = bool(
        attribution.get("legacy_model_audit", {}).get("pass", False)
    )
    denoising_pass = bool(result.get("pass", False))
    return {
        "calibration_pass": bool(result.get("calibration", {}).get("pass", False)),
        "gradient_tracking_pass": bool(
            result.get("gradient_tracking", {}).get("pass", False)
        ),
        "reconstruction_aggregate_gate_pass": bool(
            aggregate.get("gate_pass", False)
        ),
        "reconstruction_all_source_gate_pass": bool(
            true_value.get("all_source_gate_pass", False)
        ),
        "reconstruction_combined_gate_pass": bool(
            true_value.get("gate_pass", False)
        ),
        "condition_effect_pass": bool(
            result.get("condition_effect", {}).get(
                "condition_effect_supported", False
            )
        ),
        "prior_drift_pass": bool(
            float(result.get("prior_drift_ratio", float("inf"))) <= 1.000001
        ),
        "denoising_result_pass": denoising_pass,
        "branch_audit_pass": branch_pass,
        "composite_one_step_pass": bool(denoising_pass and branch_pass),
    }



def compare_to_control(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    candidate_segment = float(candidate_metrics["segment_score_p95"])
    control_segment = float(control_metrics["segment_score_p95"])
    candidate_validity = float(candidate_metrics["sample_validity_rate"])
    control_validity = float(control_metrics["sample_validity_rate"])
    candidate_best = float(candidate_metrics["best_ordered_rmse_mean"])
    control_best = float(control_metrics["best_ordered_rmse_mean"])
    candidate_inversion = float(candidate_metrics["nearest_inversion_p95"])
    control_inversion = float(control_metrics["nearest_inversion_p95"])
    tail_improved = bool(
        candidate_segment <= 0.90 * max(control_segment, 1.0e-12)
        or candidate_validity >= control_validity + 0.05
    )
    ordered_preserved = bool(candidate_best <= 1.10 * max(control_best, 1.0e-12))
    inversion_limit = max(control_inversion * 1.10, control_inversion + 1.0e-6)
    inversion_preserved = bool(candidate_inversion <= inversion_limit)
    return {
        "segment_score_p95_ratio": float(
            candidate_segment / max(control_segment, 1.0e-12)
        ),
        "sample_validity_delta": float(candidate_validity - control_validity),
        "best_ordered_rmse_ratio": float(
            candidate_best / max(control_best, 1.0e-12)
        ),
        "nearest_inversion_p95_ratio": float(
            candidate_inversion / max(control_inversion, 1.0e-12)
        ),
        "tail_improved": tail_improved,
        "ordered_preserved": ordered_preserved,
        "inversion_preserved": inversion_preserved,
        "pass": bool(tail_improved and ordered_preserved and inversion_preserved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r252_preflight_summary.json",
    )
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--residual-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--reverse-samples", type=int, default=FULL_REVERSE_K)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight_path = Path(args.preflight_report)
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    preflight_path = preflight_path.resolve()
    preflight_relative = preflight_path.relative_to(root).as_posix()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, (preflight_relative,))
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("r2.5.2 pilot requires CUDA")
    device = torch.device("cuda")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.5.2 preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("r2.5.2 source changed after preflight")
    if preflight.get("diagnostic_objective_order") != list(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("diagnostic objective order changed")
    schemas = preflight.get("schemas", {})
    if schemas.get("attribution") != ATTRIBUTION_SCHEMA:
        raise RuntimeError("attribution schema mismatch")
    if schemas.get("trajectory") != TRAJECTORY_SCHEMA:
        raise RuntimeError("trajectory schema mismatch")
    if schemas.get("per_timestep_gradient") != PER_TIMESTEP_GRADIENT_SCHEMA:
        raise RuntimeError("per-timestep gradient schema mismatch")

    arrays, manifest, x_raw, x_standardizer, train, fit, _, validation = (
        load_verified_inputs(root)
    )
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train indices contain non-train rows")
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_contract != preflight.get("dataset", {}).get("paired_row_contract"):
        raise RuntimeError("paired-row contract changed after preflight")
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    x_z = x_standardizer.transform(x_raw).astype(np.float32)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_standardizer = future_standardizer(arrays)
    y_z = y_standardizer.transform(y_raw).astype(np.float32)
    condition_all = torch.from_numpy(x_z).float().to(device)
    clean_z_all = torch.from_numpy(y_z).float().to(device)
    clean_raw_all = torch.from_numpy(y_raw).float().to(device)
    active = torch.from_numpy(np.asarray(arrays["future_active"], dtype=bool)).to(
        device
    )
    future_mean = torch.from_numpy(
        np.asarray(arrays["future_mean"], dtype=np.float32)
    ).to(device)
    future_scale = torch.from_numpy(
        np.asarray(arrays["future_scale"], dtype=np.float32)
    ).to(device)
    paired_data = subset(
        paired_rows,
        condition_all=condition_all,
        clean_z_all=clean_z_all,
        clean_raw_all=clean_raw_all,
    )
    query_condition = paired_data["condition_z"][0::2]
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
    paired_targets = paired_data["clean_raw"].reshape(8, 2, DEFAULT_TF, STATE_DIM)

    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    if frozen["artifact_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract payload changed")
    physical_contract = contract_from_json(frozen["physical_contract"])
    torch_contract = torch_contract_from_payload(frozen["physical_contract"])
    scales = fit_geometry_scales(y_raw[fit])
    reference_threshold = float(
        frozen["prediction_reference_contract"]["final_ordered_rmse_threshold"]
    )

    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
    paired_bank = build_labeled_bank(
        scheduler=scheduler,
        **paired_data,
        active_mask=active,
        timesteps=PAIR_TIMESTEPS,
        noise_seeds=EVALUATION_NOISE_SEEDS,
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
        batch_count=int(args.calibration_batches)
    )
    shared_prior = fit_shared_prior_snapshot(
        scheduler=scheduler,
        condition_z=paired_data["condition_z"],
        clean_z=paired_data["clean_z"],
        active_mask=active,
        train_spec=train_spec,
        seed=COMMON_PAIRED_PRIOR_SEED,
    )
    source_pair_ids = [value for pair in range(8) for value in (pair, pair)]
    expected_endpoints = preflight["r251_contract"]["expected_endpoints"]
    runtime: Dict[str, Any] = {}

    for objective in diagnostic_objectives():
        initial_model = instantiate_snapshot_model(
            scheduler=scheduler,
            condition_dim=int(paired_data["condition_z"].shape[1]),
            device=device,
            snapshot=shared_prior,
            seed=COMMON_PAIRED_TRAINING_SEED,
        )
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
            objective=objective,
            train_spec=train_spec,
            calibration_spec=calibration_spec,
            train_timesteps=tuple(range(100)),
            geometry_timestep_max=50,
            shared_prior_snapshot=shared_prior,
            seed=COMMON_PAIRED_TRAINING_SEED,
        )
        model = result.get("_model")
        if model is None or "_true_prediction_z" not in result:
            runtime[objective.name] = {
                "geometry_objective": asdict(objective),
                "training_completed": False,
                "failure": result.get("failure", "paired_training_incomplete"),
                "training_result": strip_runtime_objects(result),
            }
            continue
        attribution = paired_one_step_attribution(
            model=model,
            scheduler=scheduler,
            bank=paired_bank,
            source_pair_ids=source_pair_ids,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        multiplier = float(result.get("calibration", {}).get("multiplier", 0.0))
        gradient = per_timestep_gradient_audit(
            initial_model=initial_model,
            final_model=model,
            scheduler=scheduler,
            condition_z=paired_data["condition_z"],
            clean_z=paired_data["clean_z"],
            clean_raw=paired_data["clean_raw"],
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            torch_contract=torch_contract,
            scales=scales,
            objective=objective,
            multiplier=multiplier,
            batch_size=int(args.batch_size),
        )
        reverse_scheduler = make_repair_scheduler(
            REPAIR_CONFIGS["v_prediction_cosine"]
        )
        trajectory_runtime = reverse_predicted_x0_trajectory(
            model=model,
            scheduler=reverse_scheduler,
            condition_z=query_condition,
            active_mask=active,
            sample_count=int(args.reverse_samples),
            inference_steps=100,
            seed=COMMON_REVERSE_SEED,
        )
        trajectory = evaluate_reverse_trajectory(
            trajectory=trajectory_runtime,
            paired_target_raw=paired_targets,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
            reference_threshold=reference_threshold,
        )
        reproduction = compare_endpoint_to_r251(
            observed=trajectory["endpoint"],
            expected=expected_endpoints[objective.name],
        )
        runtime[objective.name] = {
            "geometry_objective": asdict(objective),
            "geometry_objective_selectable": bool(objective.selectable),
            "training_completed": True,
            "training_result": strip_runtime_objects(result),
            "one_step_attribution": strip_runtime_objects(attribution),
            "one_step_gate_decomposition": gate_decomposition(
                result,
                attribution,
            ),
            "per_timestep_gradient": gradient,
            "trajectory": trajectory,
            "r251_endpoint_reproduction": reproduction,
            "historical_r251_one_step_pass": bool(
                expected_endpoints[objective.name]["one_step_pass"]
            ),
            "historical_r251_one_step_branch_pass": bool(
                expected_endpoints[objective.name]["one_step_branch_pass"]
            ),
        }

    if set(runtime) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic runtime matrix is incomplete")
    control_endpoint = runtime["v_only_frozen_control"].get("trajectory", {}).get(
        "endpoint"
    )
    if isinstance(control_endpoint, Mapping):
        for name, value in runtime.items():
            endpoint = value.get("trajectory", {}).get("endpoint")
            if not isinstance(endpoint, Mapping):
                value["endpoint_comparison_to_control"] = {
                    "pass": False,
                    "failure": "endpoint_missing",
                }
            elif name == "v_only_frozen_control":
                value["endpoint_comparison_to_control"] = {
                    "tail_improved": False,
                    "ordered_preserved": True,
                    "inversion_preserved": True,
                    "pass": True,
                }
            else:
                value["endpoint_comparison_to_control"] = compare_to_control(
                    endpoint,
                    control_endpoint,
                )

    serialized = {
        name: strip_runtime_objects(runtime[name])
        for name in DIAGNOSTIC_OBJECTIVE_NAMES
    }
    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only paired one-step and reverse-trajectory attribution completed",
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "preflight_report": preflight_relative,
        "source_sha256": source_sha256(root),
        "schemas": {
            "attribution": ATTRIBUTION_SCHEMA,
            "trajectory": TRAJECTORY_SCHEMA,
            "per_timestep_gradient": PER_TIMESTEP_GRADIENT_SCHEMA,
        },
        "fixed_contract": {
            "model": "frozen_p512_r512",
            "diagnostic_objective_order": list(DIAGNOSTIC_OBJECTIVE_NAMES),
            "paired_timesteps": list(PAIR_TIMESTEPS),
            "evaluation_noise_seeds": list(EVALUATION_NOISE_SEEDS),
            "paired_prior_seed": COMMON_PAIRED_PRIOR_SEED,
            "paired_training_seed": COMMON_PAIRED_TRAINING_SEED,
            "reverse_seed": COMMON_REVERSE_SEED,
            "reverse_steps": 100,
            "reverse_k": int(args.reverse_samples),
            "trajectory_timesteps": [99, 90, 75, 50, 25, 10, 0],
        },
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "paired_rows": paired_rows.tolist(),
            "paired_row_contract": paired_contract,
            "paired_input_max_abs_difference": pair_input_difference,
        },
        "geometry_scales": scales.to_json(),
        "shared_prior": strip_runtime_objects(shared_prior),
        "variants": serialized,
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
    report.update(classify_attribution(report))
    write_json_once(root / "reports/phase3_14b_r252_pilot_summary.json", report)
    print(
        json.dumps(
            {
                "root_cause": report["root_cause"],
                "supported_mechanisms": report["supported_mechanisms"],
                "train_only_recommendation": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
