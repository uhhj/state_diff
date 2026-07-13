#!/usr/bin/env python3
"""Run Phase3.14b-r2.5.3 train-only gate-separation pilot."""

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
    train_calibrated_geometry_variant,
)
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryTrainSpec,
    fit_geometry_scales,
    torch_contract_from_payload,
)
from ccda_phase3.phase314b_r252_transport_attribution import (
    paired_one_step_attribution,
)
from ccda_phase3.phase314b_r253_gate_separation import (
    COMMON_PAIRED_PRIOR_SEED,
    COMMON_PAIRED_TRAINING_SEED,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    EVALUATION_NOISE_SEEDS,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PAIRED_ROWS,
    EXPECTED_SUBMODULE_COMMIT,
    GATE_SEPARATION_SCHEMA,
    HISTORICAL_REPRODUCTION_SCHEMA,
    PAIR_TIMESTEPS,
    PHASE,
    RECONSTRUCTION_DECOMPOSITION_SCHEMA,
    SYNTHETIC_CONTROL_SCHEMA,
    GateSeparationSpec,
    assert_only_allowed_worktree_paths,
    classify_gate_separation,
    compare_reconstruction_to_training_result,
    evaluate_prediction_contract,
    evaluate_synthetic_controls,
    source_sha256,
    strip_runtime_objects,
)
from ccda_phase3.phase314b_r252_transport_attribution import diagnostic_objectives
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


def compare_to_r252(
    *,
    gate_separation: Mapping[str, Any],
    attribution: Mapping[str, Any],
    expected: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    branch = gate_separation.get("branch_transport", {})
    contracts = gate_separation.get("contracts", {})
    checks: Dict[str, Any] = {}

    def boolean(name: str, observed: bool, target: bool) -> None:
        checks[name] = {
            "observed": bool(observed),
            "expected": bool(target),
            "pass": bool(observed) is bool(target),
        }

    def number(name: str, observed: float, target: float) -> None:
        error = abs(float(observed) - float(target))
        checks[name] = {
            "observed": float(observed),
            "expected": float(target),
            "absolute_error": float(error),
            "tolerance": float(tolerance),
            "pass": bool(error <= tolerance),
        }

    boolean(
        "branch_audit_pass",
        bool(contracts.get("branch_transport_pass")),
        bool(expected["branch_audit_pass"]),
    )
    boolean(
        "composite_one_step_pass",
        bool(contracts.get("historical_composite_pass")),
        bool(expected["composite_one_step_pass"]),
    )
    number(
        "own_target_closer_fraction",
        float(branch.get("own_target_closer_fraction", 0.0)),
        float(expected["legacy_own_target_closer_fraction"]),
    )
    number(
        "separation_ratio_p50",
        float(branch.get("separation_ratio", {}).get("p50", 0.0)),
        float(expected["legacy_separation_ratio_p50"]),
    )
    number(
        "branch_delta_cosine_p50",
        float(branch.get("branch_delta_cosine", {}).get("p50", 0.0)),
        float(expected["legacy_delta_cosine_p50"]),
    )
    oracle = attribution.get("exact_v_oracle_audit", {})
    checks["exact_v_oracle_pass"] = {
        "observed": bool(oracle.get("pass", False)),
        "expected": True,
        "pass": bool(oracle.get("pass", False)),
    }
    pipeline = attribution.get("pipeline_controls", {})
    checks["pipeline_controls_pass"] = {
        "observed": bool(pipeline.get("pass", False)),
        "expected": True,
        "pass": bool(pipeline.get("pass", False)),
    }
    return {
        "schema": HISTORICAL_REPRODUCTION_SCHEMA,
        "checks": checks,
        "pass": bool(all(item["pass"] for item in checks.values())),
    }


def reverse_contract_from_history(
    expected: Mapping[str, Any],
    spec: GateSeparationSpec,
) -> Dict[str, Any]:
    endpoint = expected["endpoint"]
    sample_validity = float(endpoint["sample_validity_rate"])
    valid_query = float(endpoint["valid_query_rate"])
    both = float(endpoint["both_branch_support_rate"])
    return {
        "source": "immutable_r252_final_summary",
        "sample_validity_rate": sample_validity,
        "valid_query_rate": valid_query,
        "best_ordered_rmse_mean": float(endpoint["best_ordered_rmse_mean"]),
        "nearest_inversion_p95": float(endpoint["nearest_inversion_p95"]),
        "both_branch_support_rate": both,
        "sample_validity_pass": bool(
            sample_validity >= spec.reverse_sample_validity_min
        ),
        "valid_query_pass": bool(valid_query >= spec.reverse_valid_query_min),
        "both_branch_support_pass": bool(
            both >= spec.reverse_both_branch_support_min
        ),
        "candidate_quality_gate_pass": bool(
            sample_validity >= spec.reverse_sample_validity_min
            and valid_query >= spec.reverse_valid_query_min
            and both >= spec.reverse_both_branch_support_min
        ),
        "candidate_selected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r253_preflight_summary.json",
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
        raise RuntimeError("r2.5.3 pilot requires CUDA")
    device = torch.device("cuda")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.5.3 preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("r2.5.3 source changed after preflight")
    if preflight.get("diagnostic_objective_order") != list(
        DIAGNOSTIC_OBJECTIVE_NAMES
    ):
        raise RuntimeError("diagnostic objective order changed")
    schemas = preflight.get("schemas", {})
    if schemas.get("gate_separation") != GATE_SEPARATION_SCHEMA:
        raise RuntimeError("gate-separation schema mismatch")
    if schemas.get("reconstruction_decomposition") != RECONSTRUCTION_DECOMPOSITION_SCHEMA:
        raise RuntimeError("reconstruction-decomposition schema mismatch")
    if schemas.get("synthetic_controls") != SYNTHETIC_CONTROL_SCHEMA:
        raise RuntimeError("synthetic-control schema mismatch")

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
    input_difference = float(
        torch.max(
            torch.abs(
                paired_data["condition_z"][0::2]
                - paired_data["condition_z"][1::2]
            )
        ).cpu()
    )
    if input_difference > 1.0e-6:
        raise RuntimeError("paired deployable inputs differ")

    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    if frozen["artifact_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract payload changed")
    physical_contract = contract_from_json(frozen["physical_contract"])
    torch_contract = torch_contract_from_payload(frozen["physical_contract"])
    spec = GateSeparationSpec()
    frozen_inversion_threshold = float(
        frozen["prediction_reference_contract"][
            "nearest_index_inversion_threshold"
        ]
    )
    if abs(
        frozen_inversion_threshold - spec.topology_inversion_p95_max
    ) > 1.0e-12:
        raise RuntimeError("frozen nearest-index inversion threshold changed")
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
    expected_prior_sha = preflight["r252_contract"]["shared_prior_sha256"]
    if shared_prior.get("prior_state_sha256") != expected_prior_sha:
        raise RuntimeError("shared paired-prior SHA did not reproduce r2.5.2")

    source_pair_ids = [value for pair in range(8) for value in (pair, pair)]
    synthetic = evaluate_synthetic_controls(
        bank=paired_bank,
        source_pair_ids=source_pair_ids,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        physical_contract=physical_contract,
        spec=spec,
    )

    runtime: Dict[str, Any] = {}
    expected_variants = preflight["r252_contract"]["variants"]
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
        predicted_z = result.get("_true_prediction_z")
        if model is None or predicted_z is None:
            runtime[objective.name] = {
                "geometry_objective": asdict(objective),
                "training_completed": False,
                "failure": result.get("failure", "paired_training_incomplete"),
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
        separation = evaluate_prediction_contract(
            predicted_z=predicted_z,
            bank=paired_bank,
            source_pair_ids=source_pair_ids,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
            spec=spec,
        )
        metric_reproduction = compare_reconstruction_to_training_result(
            observed=separation,
            training_result=result,
            tolerance=spec.reproduction_absolute_tolerance,
        )
        r252_reproduction = compare_to_r252(
            gate_separation=separation,
            attribution=attribution,
            expected=expected_variants[objective.name],
            tolerance=spec.reproduction_absolute_tolerance,
        )
        r252_reproduction["reconstruction_metrics"] = metric_reproduction
        r252_reproduction["pass"] = bool(
            r252_reproduction["pass"] and metric_reproduction["pass"]
        )
        runtime[objective.name] = {
            "geometry_objective": asdict(objective),
            "training_completed": True,
            "training_contract": {
                "calibration_pass": bool(
                    result.get("calibration", {}).get("pass", False)
                ),
                "gradient_tracking_pass": bool(
                    result.get("gradient_tracking", {}).get("pass", False)
                ),
                "prior_drift_ratio": float(
                    result.get("prior_drift_ratio", float("inf"))
                ),
                "condition_effect_pass": bool(
                    result.get("condition_effect", {}).get(
                        "condition_effect_supported", False
                    )
                ),
            },
            "gate_separation": separation,
            "one_step_attribution_controls": {
                "exact_v_oracle_audit": attribution.get(
                    "exact_v_oracle_audit", {}
                ),
                "pipeline_controls": attribution.get("pipeline_controls", {}),
            },
            "r252_one_step_reproduction": r252_reproduction,
            "immutable_reverse_contract": reverse_contract_from_history(
                expected_variants[objective.name], spec
            ),
        }

    if set(runtime) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("diagnostic runtime matrix is incomplete")

    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only exact-reconstruction / branch-transport gate-separation "
            "audit completed"
        ),
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "preflight_report": preflight_relative,
        "source_sha256": source_sha256(root),
        "schemas": {
            "gate_separation": GATE_SEPARATION_SCHEMA,
            "reconstruction_decomposition": RECONSTRUCTION_DECOMPOSITION_SCHEMA,
            "synthetic_controls": SYNTHETIC_CONTROL_SCHEMA,
            "historical_reproduction": HISTORICAL_REPRODUCTION_SCHEMA,
        },
        "fixed_contract": {
            "model": "frozen_p512_r512",
            "diagnostic_objective_order": list(DIAGNOSTIC_OBJECTIVE_NAMES),
            "paired_timesteps": list(PAIR_TIMESTEPS),
            "evaluation_noise_seeds": list(EVALUATION_NOISE_SEEDS),
            "paired_prior_seed": COMMON_PAIRED_PRIOR_SEED,
            "paired_training_seed": COMMON_PAIRED_TRAINING_SEED,
            "training_changed": False,
            "historical_reconstruction_gate_changed": False,
            "branch_gate_changed": False,
            "reverse_metrics_recomputed": False,
            "ordered_topology_threshold_source": (
                "frozen_prediction_reference_contract"
            ),
            "nearest_index_inversion_p95_max": (
                frozen_inversion_threshold
            ),
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
        "synthetic_controls": strip_runtime_objects(synthetic),
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
    report.update(classify_gate_separation(report))
    write_json_once(root / "reports/phase3_14b_r253_pilot_summary.json", report)
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "root_cause": report["root_cause"],
                "supported_mechanisms": report["supported_mechanisms"],
                "train_only_recommendation": None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
