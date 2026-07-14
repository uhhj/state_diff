#!/usr/bin/env python3
"""Run the Phase3.14b-r2.5.4 train-only robot-proxy attribution pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
from ccda_phase3.phase314b_r22_geometry import (
    contract_from_json,
    torch_inverse_standardize,
)
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
    train_calibrated_geometry_variant,
)
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryTrainSpec,
    fit_geometry_scales,
    torch_contract_from_payload,
)
from ccda_phase3.phase314b_r252_transport_attribution import diagnostic_objectives
from ccda_phase3.phase314b_r253_gate_separation import (
    GateSeparationSpec,
    compare_reconstruction_to_training_result,
    evaluate_prediction_contract,
)
from ccda_phase3.phase314b_r254_robot_proxy_attribution import (
    CABLE_DIM,
    COMMON_PAIRED_PRIOR_SEED,
    COMMON_PAIRED_TRAINING_SEED,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EVALUATION_NOISE_SEEDS,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SHARED_PAIRED_PRIOR_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PAIR_TIMESTEPS,
    PHASE,
    ROBOT_ATTRIBUTION_SCHEMA,
    RobotProxyAuditSpec,
    action_sensitivity_probe,
    assert_only_allowed_worktree_paths,
    classify_robot_proxy_attribution,
    evaluate_synthetic_controls,
    hybrid_counterfactual_audit,
    model_robot_error_audit,
    robot_predictability_baselines,
    robot_proxy_schema_audit,
    source_sha256,
    strip_runtime_objects,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.schema_v2 import ROBOT_PROXY_DIM, STATE_DIM


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
    index = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(condition_all.device)
    return {
        "condition_z": condition_all.index_select(0, index),
        "clean_z": clean_z_all.index_select(0, index),
        "clean_raw": clean_raw_all.index_select(0, index),
    }


def visible_seed_array(arrays: Mapping[str, Any]) -> np.ndarray:
    for key in ("visible_seed", "seed", "episode_seed"):
        if key in arrays:
            value = np.asarray(arrays[key])
            if value.ndim == 1:
                return value
    raise RuntimeError("visible-seed field is missing")


def aggregate_prediction_by_source(
    prediction_z: torch.Tensor,
    source_ids: torch.Tensor,
    source_count: int,
) -> np.ndarray:
    prediction = prediction_z.detach().cpu().numpy().astype(np.float64)
    source = source_ids.detach().cpu().numpy().astype(np.int64)
    output = []
    for local in range(int(source_count)):
        mask = source == local
        if int(np.count_nonzero(mask)) == 0:
            raise RuntimeError(f"prediction bank missing source {local}")
        output.append(np.mean(prediction[mask], axis=0))
    return np.stack(output, axis=0)


def compare_to_r253(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    observed_contracts = observed.get("contracts", {})
    expected_contracts = expected.get("contracts", {})
    for key in (
        "historical_exact_reconstruction_pass",
        "branch_transport_pass",
        "one_step_physical_pass",
        "ordered_topology_pass",
        "final_horizon_cable_geometry_pass",
        "all_earlier_horizon_cable_geometry_pass",
    ):
        current = bool(observed_contracts.get(key))
        target = bool(expected_contracts.get(key))
        checks[f"contract.{key}"] = {
            "observed": current,
            "expected": target,
            "pass": current is target,
        }
    observed_group = observed.get("state_group_z_metrics", {})
    expected_group = expected.get("state_group_z_metrics", {})
    for key in (
        "full_z_mse",
        "cable_z_mse",
        "robot_proxy_z_mse",
        "cable_error_fraction",
        "robot_proxy_error_fraction",
    ):
        current = float(observed_group.get(key, float("inf")))
        target = float(expected_group.get(key, float("nan")))
        error = abs(current - target)
        checks[f"state_group_z_metrics.{key}"] = {
            "observed": current,
            "expected": target,
            "absolute_error": error,
            "tolerance": float(tolerance),
            "pass": bool(error <= tolerance),
        }
    return {
        "checks": checks,
        "pass": bool(all(item["pass"] for item in checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report", default="reports/phase3_14b_r254_preflight_summary.json"
    )
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--residual-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
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
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("r2.5.4 deterministic retraining requires CUDA")
    device = torch.device("cuda")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.5.4 preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("r2.5.4 source changed after preflight")
    if preflight.get("diagnostic_objective_order") != list(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic objective order changed")

    arrays, manifest, x_raw, x_standardizer, train, fit, _, validation = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train indices contain non-train rows")
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_contract != preflight.get("dataset", {}).get("paired_row_contract"):
        raise RuntimeError("paired-row contract changed after preflight")
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    x_raw_array = np.asarray(x_raw, dtype=np.float32)
    history_raw = x_raw_array.reshape(x_raw_array.shape[0], -1, STATE_DIM)
    x_z = x_standardizer.transform(x_raw_array).astype(np.float32)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_standardizer = future_standardizer(arrays)
    y_z = y_standardizer.transform(y_raw).astype(np.float32)
    y_action = np.asarray(arrays["y_action"], dtype=np.float32)
    groups = visible_seed_array(arrays)
    active_np = np.asarray(arrays["future_active"], dtype=bool)
    mean_np = np.asarray(arrays["future_mean"], dtype=np.float32)
    scale_np = np.asarray(arrays["future_scale"], dtype=np.float32)

    condition_all = torch.from_numpy(x_z).float().to(device)
    clean_z_all = torch.from_numpy(y_z).float().to(device)
    clean_raw_all = torch.from_numpy(y_raw).float().to(device)
    active = torch.from_numpy(active_np).to(device)
    future_mean = torch.from_numpy(mean_np).to(device)
    future_scale = torch.from_numpy(scale_np).to(device)
    paired_data = subset(
        paired_rows,
        condition_all=condition_all,
        clean_z_all=clean_z_all,
        clean_raw_all=clean_raw_all,
    )
    input_difference = float(torch.max(torch.abs(
        paired_data["condition_z"][0::2] - paired_data["condition_z"][1::2]
    )).cpu())
    if input_difference > 1.0e-6:
        raise RuntimeError("paired deployable inputs differ")

    frozen = load_self_hashed_json(root / "reports/phase3_14b_r22_frozen_contract.json")
    if frozen["artifact_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract payload changed")
    physical_contract = contract_from_json(frozen["physical_contract"])
    torch_contract = torch_contract_from_payload(frozen["physical_contract"])
    gate_spec = GateSeparationSpec()
    spec = RobotProxyAuditSpec()
    scales = fit_geometry_scales(y_raw[fit])
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
    calibration_spec = GradientCalibrationSpec(batch_count=int(args.calibration_batches))
    shared_prior = fit_shared_prior_snapshot(
        scheduler=scheduler,
        condition_z=paired_data["condition_z"],
        clean_z=paired_data["clean_z"],
        active_mask=active,
        train_spec=train_spec,
        seed=COMMON_PAIRED_PRIOR_SEED,
    )
    if shared_prior.get("prior_state_sha256") != EXPECTED_SHARED_PAIRED_PRIOR_SHA256:
        raise RuntimeError("shared paired-prior SHA did not reproduce")

    source_pair_ids = [value for pair in range(8) for value in (pair, pair)]
    schema_audit = robot_proxy_schema_audit(
        history_raw=history_raw[train],
        future_raw=y_raw[train],
        future_active=active_np,
        spec=spec,
    )
    synthetic = evaluate_synthetic_controls(spec)
    baselines_runtime = robot_predictability_baselines(
        history_raw=history_raw,
        condition_features=x_z,
        future_raw=y_raw,
        future_z=y_z,
        future_mean=mean_np,
        future_scale=scale_np,
        future_active=active_np,
        action_target=y_action,
        groups=groups,
        fit_rows=train,
        evaluation_rows=paired_rows,
        spec=spec,
    )
    baseline_name = baselines_runtime["best_deployable_baseline"]
    baseline_robot_z = baselines_runtime["_prediction_z"][baseline_name]

    expected_variants = preflight["r253_contract"]["variants"]
    runtime: Dict[str, Any] = {}
    for objective in diagnostic_objectives():
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
        predicted_z_tensor = result.get("_true_prediction_z")
        if model is None or predicted_z_tensor is None:
            runtime[objective.name] = {
                "geometry_objective": asdict(objective),
                "training_completed": False,
                "failure": result.get("failure", "paired_training_incomplete"),
            }
            continue

        separation = evaluate_prediction_contract(
            predicted_z=predicted_z_tensor,
            bank=paired_bank,
            source_pair_ids=source_pair_ids,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
            spec=gate_spec,
        )
        metric_reproduction = compare_reconstruction_to_training_result(
            observed=separation,
            training_result=result,
            tolerance=spec.reproduction_absolute_tolerance,
        )
        historical_reproduction = compare_to_r253(
            separation,
            expected_variants[objective.name],
            spec.reproduction_absolute_tolerance,
        )
        historical_reproduction["training_metrics"] = metric_reproduction
        historical_reproduction["pass"] = bool(
            historical_reproduction["pass"] and metric_reproduction["pass"]
        )

        predicted_z = predicted_z_tensor.detach().cpu().numpy().astype(np.float64)
        predicted_raw = torch_inverse_standardize(
            predicted_z_tensor, future_mean, future_scale
        ).detach().cpu().numpy().astype(np.float64)
        target_z_bank = paired_bank.clean_z.detach().cpu().numpy().astype(np.float64)
        target_raw_bank = paired_bank.clean_raw.detach().cpu().numpy().astype(np.float64)
        bank_history = history_raw[paired_rows][
            paired_bank.source_ids.detach().cpu().numpy().astype(np.int64)
        ]
        bank_robot_error = model_robot_error_audit(
            prediction_z=predicted_z,
            prediction_raw=predicted_raw,
            target_z=target_z_bank,
            target_raw=target_raw_bank,
            future_active=active_np,
            history_raw=bank_history,
        )

        source_mean_z = aggregate_prediction_by_source(
            predicted_z_tensor, paired_bank.source_ids, len(paired_rows)
        )
        source_mean_raw = (
            source_mean_z * scale_np[None, ...] + mean_np[None, ...]
        ).astype(np.float64)
        source_robot_error = model_robot_error_audit(
            prediction_z=source_mean_z,
            prediction_raw=source_mean_raw,
            target_z=y_z[paired_rows],
            target_raw=y_raw[paired_rows],
            future_active=active_np,
            history_raw=history_raw[paired_rows],
        )
        hybrid = hybrid_counterfactual_audit(
            prediction_z=source_mean_z,
            target_z=y_z[paired_rows],
            baseline_robot_z=baseline_robot_z,
            future_active=active_np,
            z_mse_threshold=float(
                separation["historical_reconstruction_gate"]["z_mse_max"]
            ),
        )
        action_probe = action_sensitivity_probe(
            condition_features=x_z,
            oracle_future_z=y_z,
            model_future_z_for_evaluation=source_mean_z,
            action_target=y_action,
            groups=groups,
            fit_rows=train,
            evaluation_rows=paired_rows,
            spec=spec,
        )
        runtime[objective.name] = {
            "geometry_objective": asdict(objective),
            "training_completed": True,
            "r253_reproduction": historical_reproduction,
            "r253_reproduction_pass": bool(historical_reproduction["pass"]),
            "fixed_cable_contract": {
                "branch_transport_pass": bool(
                    separation["contracts"]["branch_transport_pass"]
                ),
                "ordered_topology_pass": bool(
                    separation["contracts"]["ordered_topology_pass"]
                ),
                "one_step_physical_pass": bool(
                    separation["contracts"]["one_step_physical_pass"]
                ),
                "historical_exact_reconstruction_pass": bool(
                    separation["contracts"]["historical_exact_reconstruction_pass"]
                ),
            },
            "schema_audit": schema_audit,
            "robot_error": bank_robot_error,
            "source_mean_robot_error": source_robot_error,
            "predictability_baselines": baselines_runtime,
            "hybrid_counterfactual": hybrid,
            "action_sensitivity": action_probe,
        }

    if set(runtime) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic runtime matrix is incomplete")
    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only robot-proxy schema, predictability and action-sensitivity attribution completed",
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "preflight_report": preflight_relative,
        "source_sha256": source_sha256(root),
        "schema": ROBOT_ATTRIBUTION_SCHEMA,
        "spec": asdict(spec),
        "fixed_contract": {
            "model": "frozen_p512_r512",
            "diagnostic_objective_order": list(DIAGNOSTIC_OBJECTIVE_NAMES),
            "paired_timesteps": list(PAIR_TIMESTEPS),
            "evaluation_noise_seeds": list(EVALUATION_NOISE_SEEDS),
            "paired_prior_seed": COMMON_PAIRED_PRIOR_SEED,
            "paired_training_seed": COMMON_PAIRED_TRAINING_SEED,
            "training_changed": False,
            "cable_branch_contract_changed": False,
            "ordered_topology_contract_changed": False,
            "reverse_sampling_rerun": False,
            "action_probe_is_formal_idm": False,
        },
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "paired_rows": paired_rows.tolist(),
            "paired_row_contract": paired_contract,
            "paired_input_max_abs_difference": input_difference,
        },
        "geometry_scales": scales.to_json(),
        "shared_prior": strip_runtime_objects(shared_prior),
        "schema_audit": strip_runtime_objects(schema_audit),
        "synthetic_controls": strip_runtime_objects(synthetic),
        "predictability_baselines": strip_runtime_objects(baselines_runtime),
        "variants": strip_runtime_objects(runtime),
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
    report.update(classify_robot_proxy_attribution(report))
    write_json_once(root / "reports/phase3_14b_r254_pilot_summary.json", report)
    print(json.dumps({
        "verdict": "PASS",
        "root_cause": report["root_cause"],
        "supported_mechanisms": report["supported_mechanisms"],
        "train_only_recommendation": None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
