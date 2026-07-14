#!/usr/bin/env python3
"""Run one isolated, summary-only Resume5 residual-training repeat."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
from ccda_phase3.phase314b_r22_geometry import contract_from_json
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    select_balanced_paired_rows,
)
from ccda_phase3.phase314b_r241_multirow import build_labeled_bank
import ccda_phase3.phase314b_r251_gradient_calibration as r251
from ccda_phase3.phase314b_r252_transport_attribution import diagnostic_objectives
from ccda_phase3.phase314b_r253_gate_separation import (
    GateSeparationSpec,
    compare_reconstruction_to_training_result,
    evaluate_prediction_contract,
)
from ccda_phase3.phase314b_r254_resume3_prediction_adapter import (
    reconstruct_prior_prediction,
)
from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    CONTRACT_KEYS,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256,
    EXPECTED_CURRENT_PRIOR_STATE_SHA256,
    NUMERIC_KEYS,
    Resume5Spec,
    assert_json_summary_safe,
    assert_only_allowed_worktree_paths,
    canonical_sha256,
    instrument_residual_training,
)
from ccda_phase3.phase314b_r254_robot_proxy_attribution import (
    EVALUATION_NOISE_SEEDS,
    EXPECTED_PAIRED_ROWS,
    PAIR_TIMESTEPS,
    RobotProxyAuditSpec,
)
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryTrainSpec,
    fit_geometry_scales,
    torch_contract_from_payload,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.schema_v2 import STATE_DIM


def subset_rows(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[np.asarray(rows, dtype=np.int64)]


def runtime_contract() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Resume5 worker requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": [int(properties.major), int(properties.minor)],
        "gpu_total_memory": int(properties.total_memory),
        "gpu_multiprocessor_count": int(properties.multi_processor_count),
        "backend_flags": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "PYTHONNOUSERSITE",
            )
        },
    }


def build_training_inputs(root: Path, device: torch.device) -> Dict[str, Any]:
    arrays, _, x_raw, x_standardizer, train, fit, _, _ = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train") or np.any(split[fit] != "train"):
        raise RuntimeError("train/fit indices contain non-train rows")
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = r251.assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_rows.tolist() != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired row identity changed")

    # Transform only the fixed train rows used by this audit.  Validation and
    # formal targets are never indexed or passed to a model/metric.
    condition_np = x_standardizer.transform(
        subset_rows(np.asarray(x_raw, dtype=np.float32), paired_rows)
    ).astype(np.float32)
    y_raw_all = np.asarray(arrays["y_state"], dtype=np.float32)
    paired_raw_np = subset_rows(y_raw_all, paired_rows)
    y_standardizer = future_standardizer(arrays)
    paired_z_np = y_standardizer.transform(paired_raw_np).astype(np.float32)
    active_np = np.asarray(arrays["future_active"], dtype=np.bool_)
    mean_np = np.asarray(arrays["future_mean"], dtype=np.float32)
    scale_np = np.asarray(arrays["future_scale"], dtype=np.float32)
    if paired_raw_np.shape != (16, 4, STATE_DIM):
        raise RuntimeError("paired future shape changed")

    condition = torch.from_numpy(condition_np).to(device=device, dtype=torch.float32)
    clean_z = torch.from_numpy(paired_z_np).to(device=device, dtype=torch.float32)
    clean_raw = torch.from_numpy(paired_raw_np).to(device=device, dtype=torch.float32)
    active = torch.from_numpy(active_np).to(device=device, dtype=torch.bool)
    future_mean = torch.from_numpy(mean_np).to(device=device, dtype=torch.float32)
    future_scale = torch.from_numpy(scale_np).to(device=device, dtype=torch.float32)
    return {
        "paired_rows": paired_rows,
        "paired_contract": paired_contract,
        "condition_z": condition,
        "clean_z": clean_z,
        "clean_raw": clean_raw,
        "active_mask": active,
        "future_mean": future_mean,
        "future_scale": future_scale,
        "geometry_scales": fit_geometry_scales(y_raw_all[np.asarray(fit, dtype=np.int64)]),
    }


def run_repeat(root: Path, repeat_index: int, spec: Resume5Spec) -> Dict[str, Any]:
    spec.validate()
    allowed = (
        "reports/phase3_14b_r254_resume5_test_gate_summary.json",
        "reports/phase3_14b_r254_resume5_preflight_summary.json",
    )
    assert_only_allowed_worktree_paths(root, allowed)
    runtime = runtime_contract()
    if runtime["gpu_name"] != spec.expected_gpu_name:
        raise RuntimeError(
            f"Resume5 GPU changed: {runtime['gpu_name']} != {spec.expected_gpu_name}"
        )
    device = torch.device("cuda:0")
    data = build_training_inputs(root, device)

    frozen = load_self_hashed_json(root / "reports/phase3_14b_r22_frozen_contract.json")
    if frozen.get("artifact_sha256") != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen validity contract changed")
    physical_contract = contract_from_json(frozen["physical_contract"])
    torch_contract = torch_contract_from_payload(frozen["physical_contract"])
    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
    evaluation_bank = build_labeled_bank(
        scheduler=scheduler,
        condition_z=data["condition_z"],
        clean_z=data["clean_z"],
        clean_raw=data["clean_raw"],
        active_mask=data["active_mask"],
        timesteps=PAIR_TIMESTEPS,
        noise_seeds=EVALUATION_NOISE_SEEDS,
        matched_noise_across_sources=True,
    )
    train_spec = GeometryTrainSpec(
        prior_steps=spec.prior_steps,
        residual_steps=spec.residual_steps,
        batch_size=spec.batch_size,
        prior_learning_rate=spec.prior_learning_rate,
        residual_learning_rate=spec.residual_learning_rate,
    )
    calibration_spec = r251.GradientCalibrationSpec(
        batch_count=spec.calibration_batches
    )
    shared_prior = r251.fit_shared_prior_snapshot(
        scheduler=scheduler,
        condition_z=data["condition_z"],
        clean_z=data["clean_z"],
        active_mask=data["active_mask"],
        train_spec=train_spec,
        seed=spec.common_prior_seed,
    )
    state_sha = str(shared_prior.get("prior_state_sha256"))
    if state_sha != EXPECTED_CURRENT_PRIOR_STATE_SHA256:
        raise RuntimeError("same-device prior state SHA did not reproduce")
    prediction_sha, prediction_shape, prediction_dtype = reconstruct_prior_prediction(
        shared_prior,
        scheduler=scheduler,
        condition_z=data["condition_z"],
        seed=spec.common_prior_seed,
    )
    if prediction_sha != EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256:
        raise RuntimeError("same-device prior prediction SHA did not reproduce")
    prior_history = shared_prior.get("prior_history", [])
    prior_summary = {
        "state_sha256": state_sha,
        "prediction_sha256": prediction_sha,
        "prediction_shape": [int(value) for value in prediction_shape],
        "prediction_dtype": prediction_dtype,
        "prior_z_mse": float(shared_prior["prior_z_mse"]),
        "prior_history": prior_history,
        "prior_history_sha256": canonical_sha256(prior_history),
    }

    variants: Dict[str, Any] = {}
    source_pair_ids = [value for pair in range(8) for value in (pair, pair)]
    gate_spec = GateSeparationSpec()
    reproduction_tolerance = RobotProxyAuditSpec().reproduction_absolute_tolerance
    objectives = diagnostic_objectives()
    if tuple(item.name for item in objectives) != DIAGNOSTIC_OBJECTIVE_NAMES:
        raise RuntimeError("diagnostic objective order changed")
    for objective in objectives:
        with instrument_residual_training(r251) as recorder:
            result = r251.train_calibrated_geometry_variant(
                scheduler=scheduler,
                condition_z=data["condition_z"],
                clean_z=data["clean_z"],
                clean_raw=data["clean_raw"],
                evaluation_bank=evaluation_bank,
                active_mask=data["active_mask"],
                future_mean=data["future_mean"],
                future_scale=data["future_scale"],
                physical_contract=physical_contract,
                torch_contract=torch_contract,
                scales=data["geometry_scales"],
                objective=objective,
                train_spec=train_spec,
                calibration_spec=calibration_spec,
                train_timesteps=tuple(range(100)),
                geometry_timestep_max=50,
                shared_prior_snapshot=shared_prior,
                seed=spec.common_training_seed,
            )
            identity = recorder.finish(result)
        prediction = result.get("_true_prediction_z")
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError(f"variant {objective.name} returned no prediction")
        separation = evaluate_prediction_contract(
            predicted_z=prediction,
            bank=evaluation_bank,
            source_pair_ids=source_pair_ids,
            active_mask=data["active_mask"],
            future_mean=data["future_mean"],
            future_scale=data["future_scale"],
            physical_contract=physical_contract,
            spec=gate_spec,
        )
        internal = compare_reconstruction_to_training_result(
            observed=separation,
            training_result=result,
            tolerance=reproduction_tolerance,
        )
        if not internal.get("pass"):
            raise RuntimeError(f"variant {objective.name} failed internal metric binding")
        stripped_result = r251.strip_runtime_objects(result)
        payload: Dict[str, Any] = {
            "geometry_objective": asdict(objective),
            "identity": identity,
            "contracts": {
                key: bool(separation["contracts"][key]) for key in CONTRACT_KEYS
            },
            "state_group_z_metrics": {
                key: float(separation["state_group_z_metrics"][key])
                for key in NUMERIC_KEYS
            },
            "internal_metric_reproduction": internal,
            "training_pass": bool(result.get("pass")),
            "training_result_summary_sha256": canonical_sha256(stripped_result),
            "separation_summary_sha256": canonical_sha256(separation),
            "calibration_sha256": canonical_sha256(result.get("calibration", {})),
            "gradient_tracking_sha256": canonical_sha256(
                result.get("gradient_tracking", {})
            ),
            "condition_effect_sha256": canonical_sha256(
                result.get("condition_effect", {})
            ),
        }
        payload["variant_fingerprint_sha256"] = canonical_sha256(payload)
        assert_json_summary_safe(payload)
        variants[objective.name] = payload
        del result
        del prediction
        torch.cuda.synchronize(device)

    output: Dict[str, Any] = {
        "schema": "phase314b_r254_resume5_worker_summary_v1",
        "repeat_index": int(repeat_index),
        "runtime": runtime,
        "fixed_contract": {
            "paired_rows": data["paired_rows"].tolist(),
            "paired_row_contract": data["paired_contract"],
            "paired_timesteps": list(PAIR_TIMESTEPS),
            "evaluation_noise_seeds": list(EVALUATION_NOISE_SEEDS),
            "objective_order": list(DIAGNOSTIC_OBJECTIVE_NAMES),
            "training_spec": asdict(train_spec),
            "calibration_spec": asdict(calibration_spec),
            "prior_seed": spec.common_prior_seed,
            "training_seed": spec.common_training_seed,
        },
        "prior": prior_summary,
        "variants": variants,
        "legacy_cache_loader_eagerly_materializes_full_npz": True,
        "validation_target_rows_indexed": False,
        "formal_target_rows_indexed": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "reverse_sampling_rerun": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "worker_file_output": False,
    }
    output["worker_summary_sha256"] = canonical_sha256(output)
    assert_json_summary_safe(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--repeat-index", type=int, required=True)
    args = parser.parse_args()
    if args.repeat_index not in range(3):
        raise ValueError("repeat-index must be 0, 1, or 2")
    report = run_repeat(Path(args.root).resolve(), args.repeat_index, Resume5Spec())
    print("RESUME5_RESULT_JSON=" + json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
