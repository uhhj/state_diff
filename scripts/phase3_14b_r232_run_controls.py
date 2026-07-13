#!/usr/bin/env python3
"""Run Phase3.14b-r2.3.2 one-row noise/timestep isolation controls."""
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
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import select_unique_condition_rows
from ccda_phase3.phase314b_r232_controls import (
    CARTESIAN_TIMESTEPS,
    CONTROL_TIMESTEPS,
    FULL_TIMESTEPS,
    ModelSpec,
    PHASE,
    TrainSpec,
    assert_only_allowed_worktree_paths,
    classify_controls,
    make_tuple_bank,
    oracle_parity,
    source_sha256,
    timestep_embedding_audit,
    train_bank_control,
    train_random_stream_control,
)


def subset_one(
    row: np.ndarray,
    *,
    condition_all: torch.Tensor,
    clean_z_all: torch.Tensor,
    clean_raw_all: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if np.asarray(row).shape != (1,):
        raise ValueError("one-row control requires one index")
    index = torch.from_numpy(np.asarray(row, dtype=np.int64)).to(
        condition_all.device
    )
    return {
        "condition_z": condition_all.index_select(0, index),
        "clean_z": clean_z_all.index_select(0, index),
        "clean_raw": clean_raw_all.index_select(0, index),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--bank-steps", type=int, default=4000)
    parser.add_argument("--cartesian-steps", type=int, default=6000)
    parser.add_argument("--stream-steps", type=int, default=6000)
    parser.add_argument("--bank-size", type=int, default=64)
    parser.add_argument("--heldout-bank-size", type=int, default=64)
    parser.add_argument("--stream-batch-size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.root).resolve()

    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(
        root,
        ("reports/phase3_14b_r232_preflight_summary.json",),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("r2.3.2 controls require CUDA")
    device = torch.device("cuda")

    arrays, manifest, x_raw, x_standardizer, train, _, _, validation = (
        load_verified_inputs(root)
    )
    if np.any(np.asarray(arrays["split_name"])[train].astype(str) != "train"):
        raise RuntimeError("train split contains non-train rows")
    one_row = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=1,
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
    data = subset_one(
        one_row,
        condition_all=condition_all,
        clean_z_all=clean_z_all,
        clean_raw_all=clean_raw_all,
    )
    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])

    training_seeds = tuple(41000 + value for value in range(args.bank_size))
    heldout_seeds = tuple(
        51000 + value for value in range(args.heldout_bank_size)
    )
    cartesian_train_seeds = tuple(61000 + value for value in range(16))
    cartesian_heldout_seeds = tuple(71000 + value for value in range(16))

    fixed_t_train = make_tuple_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=(50,),
        noise_seeds=training_seeds,
        mode="fixed_timestep",
    )
    fixed_t_seen = fixed_t_train
    fixed_t_heldout = make_tuple_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=(50,),
        noise_seeds=heldout_seeds,
        mode="fixed_timestep",
    )
    fixed_noise_all_t = make_tuple_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=FULL_TIMESTEPS,
        noise_seeds=(62001,),
        mode="fixed_noise",
    )
    cartesian_train = make_tuple_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=CARTESIAN_TIMESTEPS,
        noise_seeds=cartesian_train_seeds,
        mode="cartesian",
    )
    cartesian_heldout = make_tuple_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=CONTROL_TIMESTEPS,
        noise_seeds=cartesian_heldout_seeds,
        mode="cartesian",
    )

    oracle_bank = make_tuple_bank(
        scheduler=scheduler,
        **data,
        active_mask=active,
        timesteps=CONTROL_TIMESTEPS,
        noise_seeds=(81001, 81002, 81003),
        mode="cartesian",
    )
    oracle = oracle_parity(
        scheduler=scheduler,
        bank=oracle_bank,
        active_mask=active,
    )
    time_audit = timestep_embedding_audit(time_dim=128, device=device)

    base_model = ModelSpec("baseline_mlp", hidden_dim=512, time_dim=128)
    residual_model = ModelSpec("residual_mlp", hidden_dim=512, time_dim=128)
    affine_model = ModelSpec("time_affine", hidden_dim=512, time_dim=128)
    bank_spec = TrainSpec(
        steps=args.bank_steps,
        batch_size=min(64, args.bank_size),
        learning_rate=1.0e-3,
    )
    multi_t_spec = TrainSpec(
        steps=args.bank_steps,
        batch_size=64,
        learning_rate=1.0e-3,
    )
    cartesian_spec = TrainSpec(
        steps=args.cartesian_steps,
        batch_size=64,
        learning_rate=1.0e-3,
    )
    stream_spec = TrainSpec(
        steps=args.stream_steps,
        batch_size=args.stream_batch_size,
        learning_rate=1.0e-3,
    )

    runs = {}
    fixed_evaluations = {
        "seen_noise_bank": fixed_t_seen,
        "heldout_noise_bank": fixed_t_heldout,
    }
    for name, model, seed in (
        ("fixed_t50_baseline", base_model, 42001),
        ("fixed_t50_residual", residual_model, 42002),
        ("fixed_t50_time_affine", affine_model, 42003),
    ):
        runs[name] = train_bank_control(
            name=name,
            scheduler=scheduler,
            train_bank=fixed_t_train,
            evaluation_banks=fixed_evaluations,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model,
            train_spec=bank_spec,
            seed=seed,
        )

    for name, model, seed in (
        ("fixed_noise_all_t_baseline", base_model, 43001),
        ("fixed_noise_all_t_residual", residual_model, 43002),
    ):
        runs[name] = train_bank_control(
            name=name,
            scheduler=scheduler,
            train_bank=fixed_noise_all_t,
            evaluation_banks={"seen_timestep_bank": fixed_noise_all_t},
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model,
            train_spec=multi_t_spec,
            seed=seed,
        )

    for name, model, seed in (
        ("cartesian_baseline", base_model, 44001),
        ("cartesian_residual", residual_model, 44002),
    ):
        runs[name] = train_bank_control(
            name=name,
            scheduler=scheduler,
            train_bank=cartesian_train,
            evaluation_banks={
                "seen_cartesian": cartesian_train,
                "heldout_cartesian": cartesian_heldout,
            },
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model,
            train_spec=cartesian_spec,
            seed=seed,
        )

    for name, model, seed in (
        ("stream_baseline", base_model, 45001),
        ("stream_residual", residual_model, 45002),
    ):
        runs[name] = train_random_stream_control(
            name=name,
            scheduler=scheduler,
            **data,
            heldout_bank=cartesian_heldout,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            model_spec=model,
            train_spec=stream_spec,
            seed=seed,
        )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": "train-only denoiser isolation completed; no model repair",
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "selected_train_row": int(one_row[0]),
            "selected_condition": str(arrays["condition_name"][one_row[0]]),
            "selected_visible_seed": int(arrays["visible_seed"][one_row[0]]),
            "validation_rows_verified_but_not_used": int(len(validation)),
        },
        "oracle_parity": oracle,
        "timestep_embedding_audit": time_audit,
        "banks": {
            "fixed_t50_train": fixed_t_train.row_count,
            "fixed_t50_heldout": fixed_t_heldout.row_count,
            "fixed_noise_all_t": fixed_noise_all_t.row_count,
            "cartesian_train": cartesian_train.row_count,
            "cartesian_heldout": cartesian_heldout.row_count,
        },
        "runs": runs,
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
    report.update(classify_controls(report))
    output = root / "reports/phase3_14b_r232_controls_summary.json"
    write_json_once(output, report)
    compact = {
        "root_cause": report["root_cause"],
        "next_stage": report["next_stage"],
        "oracle_pass": oracle["pass"],
        "time_embedding_pass": time_audit["pass"],
        "gates": {
            name: (
                value.get("heldout", {}).get("gate_pass")
                if value["control_kind"] == "random_stream_minibatch"
                else {
                    key: item["gate_pass"]
                    for key, item in value["evaluations"].items()
                }
            )
            for name, value in runs.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
