#!/usr/bin/env python3
"""Run Phase3.14b-r2.4 train-only timestep-conditioned noisy-skip pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    paired_selection_metadata,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import select_unique_condition_rows
from ccda_phase3.phase314b_r24_noisy_skip import (
    EVAL_TIMESTEPS,
    HIGH_TIMESTEPS,
    JVP_TIMESTEPS,
    LOW_MID_TIMESTEPS,
    PHASE,
    PILOT_MODELS,
    PilotTrainSpec,
    analytic_formula_oracle,
    assert_only_allowed_worktree_paths,
    classify_pilot,
    dependency_sha256,
    make_multirow_bank,
    source_sha256,
    stage_pass,
    train_online_pilot,
)


def subset_rows(
    rows: np.ndarray,
    *,
    condition_all: torch.Tensor,
    clean_z_all: torch.Tensor,
    clean_raw_all: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or selected.size <= 0:
        raise ValueError("rows must be nonempty one-dimensional indices")
    index = torch.from_numpy(selected).to(condition_all.device)
    return {
        "condition_z": condition_all.index_select(0, index),
        "clean_z": clean_z_all.index_select(0, index),
        "clean_raw": clean_raw_all.index_select(0, index),
    }


def build_banks(
    *,
    scheduler,
    data: Dict[str, torch.Tensor],
    active: torch.Tensor,
    seed_offset: int,
):
    full = make_multirow_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=EVAL_TIMESTEPS,
        noise_seeds=(seed_offset + 1, seed_offset + 2),
    )
    low_mid = make_multirow_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=LOW_MID_TIMESTEPS,
        noise_seeds=(seed_offset + 1, seed_offset + 2),
    )
    high = make_multirow_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=HIGH_TIMESTEPS,
        noise_seeds=(seed_offset + 1, seed_offset + 2),
    )
    jvp = make_multirow_bank(
        scheduler=scheduler,
        condition_z=data["condition_z"][:1],
        clean_z=data["clean_z"][:1],
        clean_raw=data["clean_raw"][:1],
        active_mask=active,
        timesteps=JVP_TIMESTEPS,
        noise_seeds=(seed_offset + 3,),
    )
    return full, low_mid, high, jvp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--one-row-steps", type=int, default=5000)
    parser.add_argument("--unique-steps", type=int, default=7000)
    parser.add_argument("--paired-steps", type=int, default=9000)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(
        root,
        ("reports/phase3_14b_r24_preflight_summary.json",),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("r2.4 pilot requires CUDA")
    device = torch.device("cuda")

    arrays, manifest, x_raw, x_standardizer, train, _, _, validation = (
        load_verified_inputs(root)
    )
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train index includes non-train rows")

    one_row = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=1,
    )
    unique_free = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=16,
    )
    paired = select_balanced_paired_rows(arrays, train, 16)

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

    source_data = {
        "one_row": subset_rows(
            one_row,
            condition_all=condition_all,
            clean_z_all=clean_z_all,
            clean_raw_all=clean_raw_all,
        ),
        "unique_free_16": subset_rows(
            unique_free,
            condition_all=condition_all,
            clean_z_all=clean_z_all,
            clean_raw_all=clean_raw_all,
        ),
        "paired_16": subset_rows(
            paired,
            condition_all=condition_all,
            clean_z_all=clean_z_all,
            clean_raw_all=clean_raw_all,
        ),
    }

    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
    bank_sets = {}
    for stage_index, (stage, data) in enumerate(source_data.items()):
        full, low_mid, high, jvp = build_banks(
            scheduler=scheduler,
            data=data,
            active=active,
            seed_offset=70000 + 100 * stage_index,
        )
        bank_sets[stage] = {
            "full": full,
            "low_mid": low_mid,
            "high": high,
            "jvp": jvp,
        }

    analytic_oracle = analytic_formula_oracle(
        scheduler=scheduler,
        bank=bank_sets["one_row"]["full"],
        active_mask=active,
    )
    if not analytic_oracle["pass"]:
        raise RuntimeError("analytic noisy-skip oracle parity failed")

    stages = {
        "one_row": {},
        "unique_free_16": {},
        "paired_16": {},
    }

    one_spec = PilotTrainSpec(
        warmup_steps=int(args.warmup_steps),
        diffusion_steps=int(args.one_row_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )
    unique_spec = PilotTrainSpec(
        warmup_steps=int(args.warmup_steps),
        diffusion_steps=int(args.unique_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )
    paired_spec = PilotTrainSpec(
        warmup_steps=int(args.warmup_steps),
        diffusion_steps=int(args.paired_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )

    for model_index, model_spec in enumerate(PILOT_MODELS):
        # Learned time-affine has no x0 prior and therefore cannot run warmup.
        effective_one = one_spec
        if model_spec.kind == "learned_time_affine":
            effective_one = PilotTrainSpec(
                warmup_steps=0,
                diffusion_steps=one_spec.diffusion_steps,
                batch_size=one_spec.batch_size,
                learning_rate=one_spec.learning_rate,
            )
        stages["one_row"][model_spec.name] = train_online_pilot(
            name=f"one_row_{model_spec.name}",
            scheduler=scheduler,
            **source_data["one_row"],
            evaluation_banks={
                "fresh_noise_all_t": bank_sets["one_row"]["full"],
            },
            jvp_bank=bank_sets["one_row"]["jvp"],
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model_spec,
            train_spec=effective_one,
            seed=82000 + model_index,
        )

    advancing = [
        spec
        for spec in PILOT_MODELS
        if stage_pass(
            stages["one_row"][spec.name],
            "fresh_noise_all_t",
        )
    ]

    for model_index, model_spec in enumerate(advancing):
        effective_unique = unique_spec
        if model_spec.kind == "learned_time_affine":
            effective_unique = PilotTrainSpec(
                warmup_steps=0,
                diffusion_steps=unique_spec.diffusion_steps,
                batch_size=unique_spec.batch_size,
                learning_rate=unique_spec.learning_rate,
            )
        stages["unique_free_16"][model_spec.name] = train_online_pilot(
            name=f"unique_free_16_{model_spec.name}",
            scheduler=scheduler,
            **source_data["unique_free_16"],
            evaluation_banks={
                "fresh_noise_all_t": bank_sets["unique_free_16"]["full"],
            },
            jvp_bank=bank_sets["unique_free_16"]["jvp"],
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model_spec,
            train_spec=effective_unique,
            seed=83000 + model_index,
        )

    paired_advancing = [
        spec
        for spec in advancing
        if spec.name in stages["unique_free_16"]
        and stage_pass(
            stages["unique_free_16"][spec.name],
            "fresh_noise_all_t",
        )
    ]

    for model_index, model_spec in enumerate(paired_advancing):
        effective_paired = paired_spec
        if model_spec.kind == "learned_time_affine":
            effective_paired = PilotTrainSpec(
                warmup_steps=0,
                diffusion_steps=paired_spec.diffusion_steps,
                batch_size=paired_spec.batch_size,
                learning_rate=paired_spec.learning_rate,
            )
        stages["paired_16"][model_spec.name] = train_online_pilot(
            name=f"paired_16_{model_spec.name}",
            scheduler=scheduler,
            **source_data["paired_16"],
            evaluation_banks={
                "fresh_noise_low_mid": bank_sets["paired_16"]["low_mid"],
                "fresh_noise_high_diagnostic": bank_sets["paired_16"]["high"],
            },
            jvp_bank=bank_sets["paired_16"]["jvp"],
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model_spec,
            train_spec=effective_paired,
            seed=84000 + model_index,
        )

    pair_inputs = x_z[paired].reshape(8, 2, -1)
    pair_targets = y_raw[paired, :, :48].reshape(8, 2, -1)
    pair_input_max_abs = np.max(
        np.abs(pair_inputs[:, 0] - pair_inputs[:, 1]), axis=1
    )
    pair_target_rmse = np.sqrt(
        np.mean(
            (pair_targets[:, 0] - pair_targets[:, 1]) ** 2,
            axis=1,
        )
    )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only noisy-skip architecture pilot completed; no formal "
            "model selection and no ordered-geometry repair"
        ),
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "one_row": int(one_row[0]),
            "one_row_condition": str(arrays["condition_name"][one_row[0]]),
            "one_row_visible_seed": int(arrays["visible_seed"][one_row[0]]),
            "unique_free_rows": unique_free.tolist(),
            "paired_rows": paired.tolist(),
            "paired_sampling": paired_selection_metadata(arrays, paired),
            "paired_input_max_abs_p50": float(
                np.percentile(pair_input_max_abs, 50)
            ),
            "paired_input_exact_fraction": float(
                np.mean(pair_input_max_abs == 0.0)
            ),
            "paired_target_ordered_rmse_p50": float(
                np.percentile(pair_target_rmse, 50)
            ),
        },
        "analytic_formula_oracle": analytic_oracle,
        "stages": stages,
        "source_sha256": source_sha256(root),
        "dependency_sha256": dependency_sha256(root),
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
    output = root / "reports/phase3_14b_r24_pilot_summary.json"
    write_json_once(output, report)

    compact = {
        "root_cause": report["root_cause"],
        "next_stage": report["next_stage"],
        "train_only_recommendation": report["train_only_recommendation"],
        "stage_models": {
            stage: sorted(runs) for stage, runs in stages.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
