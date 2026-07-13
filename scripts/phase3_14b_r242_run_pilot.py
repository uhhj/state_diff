#!/usr/bin/env python3
"""Run Phase3.14b-r2.4.2 train-only factorized frozen-prior pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import (
    select_unique_condition_rows,
)
from ccda_phase3.phase314b_r241_multirow import build_labeled_bank
from ccda_phase3.phase314b_r242_frozen_prior import (
    EVAL_TIMESTEPS,
    EXPECTED_PAIRED_CONDITIONS,
    FACTORIAL_VARIANTS,
    HIGH_TIMESTEPS,
    LOW_MID_TIMESTEPS,
    PHASE,
    PriorFitSpec,
    ResidualTrainSpec,
    assert_only_allowed_worktree_paths,
    classify_pilot,
    corrected_r241_interpretation,
    evaluate_factorized_bank,
    paired_branch_audit,
    source_sha256,
    strip_runtime_objects,
    summarize_seed_stability,
    train_factorized_variant,
    train_prior_seed_control,
    variant_passes_unique,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def subset(
    rows: np.ndarray,
    *,
    condition_all: torch.Tensor,
    clean_z_all: torch.Tensor,
    clean_raw_all: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size <= 0:
        raise ValueError("rows must be nonempty one-dimensional")
    index = torch.from_numpy(selected).to(condition_all.device)
    return {
        "condition_z": condition_all.index_select(0, index),
        "clean_z": clean_z_all.index_select(0, index),
        "clean_raw": clean_raw_all.index_select(0, index),
    }


def pair_ids_for_rows(
    arrays: Dict[str, np.ndarray],
    rows: np.ndarray,
) -> Sequence[int]:
    selected = np.asarray(rows, dtype=np.int64)
    pair_key = np.asarray(arrays["pair_key"]).astype(str)[selected]
    condition = np.asarray(arrays["condition_name"]).astype(str)[selected]
    mapping: Dict[str, int] = {}
    pair_ids = []
    for key in pair_key.tolist():
        if key not in mapping:
            mapping[key] = len(mapping)
        pair_ids.append(mapping[key])
    for key, pair_id in mapping.items():
        local = np.flatnonzero(pair_key == key)
        if local.size != 2:
            raise RuntimeError(f"pair {key} does not contain two rows")
        observed = set(condition[local].tolist())
        if observed != set(EXPECTED_PAIRED_CONDITIONS):
            raise RuntimeError(
                f"pair {key} has unexpected conditions: {observed}"
            )
        if set(np.asarray(pair_ids)[local].tolist()) != {pair_id}:
            raise RuntimeError("pair-id construction failed")
    return [int(value) for value in pair_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--residual-steps", type=int, default=8000)
    parser.add_argument("--paired-residual-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--decoupled-prior-learning-rate",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--evaluation-noises", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(
        root,
        ("reports/phase3_14b_r242_preflight_summary.json",),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("r2.4.2 pilot requires CUDA")
    device = torch.device("cuda")

    preflight = load_json(
        root / "reports/phase3_14b_r242_preflight_summary.json"
    )
    corrected = preflight["r241_corrected_interpretation"]
    if not corrected["classifier_precedence_bug_supported"]:
        raise RuntimeError("preflight interpretation gate failed")

    (
        arrays,
        manifest,
        x_raw,
        x_standardizer,
        train,
        _,
        _,
        validation,
    ) = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train index includes non-train rows")

    unique_rows = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=16,
    )
    paired_rows = select_balanced_paired_rows(
        arrays,
        train,
        16,
    )

    x_z = x_standardizer.transform(x_raw).astype(np.float32)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_standardizer = future_standardizer(arrays)
    y_z = y_standardizer.transform(y_raw).astype(np.float32)

    condition_all = torch.from_numpy(x_z).float().to(device)
    clean_z_all = torch.from_numpy(y_z).float().to(device)
    clean_raw_all = torch.from_numpy(y_raw).float().to(device)
    active = torch.from_numpy(
        np.asarray(arrays["future_active"], dtype=bool)
    ).to(device)
    future_mean = torch.from_numpy(
        np.asarray(arrays["future_mean"], dtype=np.float32)
    ).to(device)
    future_scale = torch.from_numpy(
        np.asarray(arrays["future_scale"], dtype=np.float32)
    ).to(device)

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

    scheduler = make_repair_scheduler(
        REPAIR_CONFIGS["v_prediction_cosine"]
    )
    evaluation_noise_seeds = tuple(
        97000 + index
        for index in range(int(args.evaluation_noises))
    )
    unique_bank = build_labeled_bank(
        scheduler=scheduler,
        **unique_data,
        active_mask=active,
        timesteps=EVAL_TIMESTEPS,
        noise_seeds=evaluation_noise_seeds,
        matched_noise_across_sources=True,
    )
    paired_low_mid_bank = build_labeled_bank(
        scheduler=scheduler,
        **paired_data,
        active_mask=active,
        timesteps=LOW_MID_TIMESTEPS,
        noise_seeds=evaluation_noise_seeds,
        matched_noise_across_sources=True,
    )
    paired_high_bank = build_labeled_bank(
        scheduler=scheduler,
        **paired_data,
        active_mask=active,
        timesteps=HIGH_TIMESTEPS,
        noise_seeds=evaluation_noise_seeds,
        matched_noise_across_sources=True,
    )

    prior_spec = PriorFitSpec(
        steps=int(args.prior_steps),
        learning_rate=float(args.prior_learning_rate),
    )
    prior_seed_values = (95101, 95102, 95103)
    direct_width_runs: Dict[str, Any] = {}
    direct_width_stability: Dict[str, Any] = {}
    for width in (512, 1024):
        runs = [
            train_prior_seed_control(
                scheduler=scheduler,
                **unique_data,
                active_mask=active,
                future_mean=future_mean,
                future_scale=future_scale,
                hidden_dim=int(width),
                spec=prior_spec,
                seed=int(seed),
            )
            for seed in prior_seed_values
        ]
        direct_width_runs[str(width)] = runs
        direct_width_stability[str(width)] = summarize_seed_stability(runs)

    residual_spec = ResidualTrainSpec(
        steps=int(args.residual_steps),
        batch_size=int(args.batch_size),
        residual_learning_rate=float(args.residual_learning_rate),
        prior_learning_rate=float(
            args.decoupled_prior_learning_rate
        ),
    )
    unique_results_runtime: Dict[str, Any] = {}
    for index, variant in enumerate(FACTORIAL_VARIANTS):
        unique_results_runtime[variant.name] = train_factorized_variant(
            scheduler=scheduler,
            **unique_data,
            evaluation_bank=unique_bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            variant=variant,
            prior_spec=prior_spec,
            residual_spec=residual_spec,
            train_timesteps=tuple(range(100)),
            seed=96100 + index,
        )

    advancing_names = [
        name
        for name, value in unique_results_runtime.items()
        if variant_passes_unique(value)
        and name != "joint_p1024_r512_control"
    ]

    paired_results_runtime: Dict[str, Any] = {}
    source_pair_ids = pair_ids_for_rows(arrays, paired_rows)
    paired_spec = ResidualTrainSpec(
        steps=int(args.paired_residual_steps),
        batch_size=int(args.batch_size),
        residual_learning_rate=float(args.residual_learning_rate),
        prior_learning_rate=float(
            args.decoupled_prior_learning_rate
        ),
    )
    variant_by_name = {
        variant.name: variant for variant in FACTORIAL_VARIANTS
    }
    for index, name in enumerate(advancing_names):
        variant = variant_by_name[name]
        result = train_factorized_variant(
            scheduler=scheduler,
            **paired_data,
            evaluation_bank=paired_low_mid_bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            variant=variant,
            prior_spec=prior_spec,
            residual_spec=paired_spec,
            train_timesteps=LOW_MID_TIMESTEPS,
            seed=97100 + index,
        )
        model = result["_model"]
        high_evaluation = evaluate_factorized_bank(
            model=model,
            scheduler=scheduler,
            bank=paired_high_bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        result["high_timestep_diagnostic"] = {
            key: value
            for key, value in high_evaluation.items()
            if not key.startswith("_")
        }
        result["paired_branch_audit"] = paired_branch_audit(
            predicted_z=result["_true_prediction_z"],
            bank=paired_low_mid_bank,
            source_pair_ids=source_pair_ids,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        paired_results_runtime[name] = result

    unique_results = {
        name: strip_runtime_objects(value)
        for name, value in unique_results_runtime.items()
    }
    paired_results = {
        name: strip_runtime_objects(value)
        for name, value in paired_results_runtime.items()
    }

    visible_seed = np.asarray(arrays["visible_seed"]).astype(np.int64)
    condition_name = np.asarray(arrays["condition_name"]).astype(str)
    pair_key = np.asarray(arrays["pair_key"]).astype(str)

    report: Dict[str, Any] = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only factorized width/frozen-prior pilot completed; "
            "no formal model selection"
        ),
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "unique_free_rows": unique_rows.tolist(),
            "unique_free_visible_seeds": (
                visible_seed[unique_rows].tolist()
            ),
            "paired_rows": paired_rows.tolist(),
            "paired_visible_seeds": (
                visible_seed[paired_rows].tolist()
            ),
            "paired_conditions": (
                condition_name[paired_rows].tolist()
            ),
            "paired_pair_keys": pair_key[paired_rows].tolist(),
            "source_pair_ids": list(source_pair_ids),
            "unique_paired_row_overlap_count": int(
                np.intersect1d(unique_rows, paired_rows).size
            ),
        },
        "evaluation": {
            "noise_seeds": list(evaluation_noise_seeds),
            "unique_timesteps": list(EVAL_TIMESTEPS),
            "paired_gate_timesteps": list(LOW_MID_TIMESTEPS),
            "paired_diagnostic_timesteps": list(HIGH_TIMESTEPS),
        },
        "r241_corrected_interpretation": corrected_r241_interpretation(
            load_json(root / "reports/phase3_14b_r241_summary.json")
        ),
        "direct_width_seed_runs": direct_width_runs,
        "direct_width_seed_stability": direct_width_stability,
        "unique_free_variants": unique_results,
        "unique_advancing_variants": advancing_names,
        "paired_low_mid_variants": paired_results,
        "source_sha256": source_sha256(root),
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

    output = root / "reports/phase3_14b_r242_pilot_summary.json"
    write_json_once(output, report)
    compact = {
        "root_cause": report["root_cause"],
        "next_stage": report["next_stage"],
        "train_only_recommendation": report[
            "train_only_recommendation"
        ],
        "direct_width_seed_stability": direct_width_stability,
        "unique_free_pass": {
            key: bool(value["pass"])
            for key, value in unique_results.items()
        },
        "paired_low_mid_pass": {
            key: bool(
                value["pass"]
                and value["paired_branch_audit"]["pass"]
            )
            for key, value in paired_results.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
