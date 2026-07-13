#!/usr/bin/env python3
"""Run train-only tiny controls for Phase3.14b-r2.3.1.

No model is checkpointed or candidate-eligible. Validation and formal-test rows
are not loaded into any control tensor.
"""
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
from ccda_phase3.phase314b_r22_contract import load_self_hashed_json
from ccda_phase3.phase314b_r22_geometry import normalizers_from_json
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    paired_selection_metadata,
    require_repository_state,
    select_balanced_paired_rows,
    write_json_once,
)
from ccda_phase3.phase314b_r231_controls import (
    DenoiserSpec,
    assert_only_allowed_worktree_paths,
    OptimizerSpec,
    PHASE,
    pair_ambiguity_metrics,
    select_unique_condition_rows,
    source_sha256,
    train_direct_control,
    train_fixed_tuple_control,
    train_random_noise_control,
)


def subset_tensors(
    rows: np.ndarray,
    *,
    condition_all: torch.Tensor,
    clean_z_all: torch.Tensor,
    clean_raw_all: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    index = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(
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
    parser.add_argument("--direct-one-steps", type=int, default=2000)
    parser.add_argument("--direct-multi-steps", type=int, default=5000)
    parser.add_argument("--fixed-one-steps", type=int, default=3000)
    parser.add_argument("--fixed-pair-steps", type=int, default=5000)
    parser.add_argument("--random-one-steps", type=int, default=6000)
    parser.add_argument("--random-multi-steps", type=int, default=8000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(
        root,
        ("reports/phase3_14b_r231_preflight_summary.json",),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("r2.3.1 controls require CUDA")
    device = torch.device("cuda")

    arrays, manifest, x_raw, x_standardizer, train, _, _, validation = (
        load_verified_inputs(root)
    )
    # load_verified_inputs verifies validation provenance, but control selection
    # and tensors below use train rows only.
    if np.any(np.asarray(arrays["split_name"][train]).astype(str) != "train"):
        raise RuntimeError("train split contains non-train rows")

    one_row = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=1,
    )
    unique_free_16 = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=16,
    )
    paired_16 = select_balanced_paired_rows(arrays, train, 16)

    selected_union = np.unique(
        np.concatenate([one_row, unique_free_16, paired_16])
    )
    if np.any(
        np.asarray(arrays["split_name"][selected_union]).astype(str) != "train"
    ):
        raise RuntimeError("a control selected a non-train row")

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

    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    normalizers = normalizers_from_json(frozen["geometry_normalizers"])
    scheduler = make_repair_scheduler(
        REPAIR_CONFIGS["v_prediction_cosine"]
    )

    data = {
        "one": subset_tensors(
            one_row,
            condition_all=condition_all,
            clean_z_all=clean_z_all,
            clean_raw_all=clean_raw_all,
        ),
        "unique": subset_tensors(
            unique_free_16,
            condition_all=condition_all,
            clean_z_all=clean_z_all,
            clean_raw_all=clean_raw_all,
        ),
        "paired": subset_tensors(
            paired_16,
            condition_all=condition_all,
            clean_z_all=clean_z_all,
            clean_raw_all=clean_raw_all,
        ),
    }

    adam_no_clip = OptimizerSpec(
        name="adam",
        learning_rate=1.0e-3,
        weight_decay=0.0,
        clip_grad_norm=None,
    )
    adamw_baseline = OptimizerSpec(
        name="adamw",
        learning_rate=1.0e-3,
        weight_decay=0.0,
        clip_grad_norm=10.0,
    )
    adam_wide = OptimizerSpec(
        name="adam",
        learning_rate=5.0e-4,
        weight_decay=0.0,
        clip_grad_norm=None,
    )
    base_model = DenoiserSpec(hidden_dim=512, time_dim=128)
    wide_model = DenoiserSpec(hidden_dim=1024, time_dim=128)

    common = {
        "active_mask": active,
        "future_mean": future_mean,
        "future_scale": future_scale,
    }
    runs = {}

    runs["direct_one_row"] = train_direct_control(
        name="direct_one_row",
        **data["one"],
        **common,
        optimizer_spec=adam_no_clip,
        hidden_dim=512,
        max_steps=args.direct_one_steps,
        seed=32101,
    )
    runs["direct_unique_free_16"] = train_direct_control(
        name="direct_unique_free_16",
        **data["unique"],
        **common,
        optimizer_spec=adam_no_clip,
        hidden_dim=512,
        max_steps=args.direct_multi_steps,
        seed=32102,
    )

    runs["fixed_one_row_v_only"] = train_fixed_tuple_control(
        name="fixed_one_row_v_only",
        scheduler=scheduler,
        **data["one"],
        **common,
        normalizers=normalizers,
        denoiser_spec=base_model,
        optimizer_spec=adamw_baseline,
        loss_mode="v_only",
        timestep=50,
        noise_seed=33101,
        fresh_noise_seed=34101,
        max_steps=args.fixed_one_steps,
        seed=32103,
    )
    runs["fixed_pairs_v_only_adamw"] = train_fixed_tuple_control(
        name="fixed_pairs_v_only_adamw",
        scheduler=scheduler,
        **data["paired"],
        **common,
        normalizers=normalizers,
        denoiser_spec=base_model,
        optimizer_spec=adamw_baseline,
        loss_mode="v_only",
        timestep=50,
        noise_seed=33102,
        fresh_noise_seed=34102,
        max_steps=args.fixed_pair_steps,
        seed=32104,
    )
    runs["fixed_pairs_v_only_adam"] = train_fixed_tuple_control(
        name="fixed_pairs_v_only_adam",
        scheduler=scheduler,
        **data["paired"],
        **common,
        normalizers=normalizers,
        denoiser_spec=base_model,
        optimizer_spec=adam_no_clip,
        loss_mode="v_only",
        timestep=50,
        noise_seed=33102,
        fresh_noise_seed=34102,
        max_steps=args.fixed_pair_steps,
        seed=32105,
    )
    runs["fixed_pairs_v_only_wide"] = train_fixed_tuple_control(
        name="fixed_pairs_v_only_wide",
        scheduler=scheduler,
        **data["paired"],
        **common,
        normalizers=normalizers,
        denoiser_spec=wide_model,
        optimizer_spec=adam_wide,
        loss_mode="v_only",
        timestep=50,
        noise_seed=33102,
        fresh_noise_seed=34102,
        max_steps=args.fixed_pair_steps,
        seed=32106,
    )
    runs["fixed_pairs_r22_geometry"] = train_fixed_tuple_control(
        name="fixed_pairs_r22_geometry",
        scheduler=scheduler,
        **data["paired"],
        **common,
        normalizers=normalizers,
        denoiser_spec=base_model,
        optimizer_spec=adamw_baseline,
        loss_mode="r22_geometry",
        timestep=50,
        noise_seed=33102,
        fresh_noise_seed=34102,
        max_steps=args.fixed_pair_steps,
        seed=32107,
    )

    runs["random_one_row"] = train_random_noise_control(
        name="random_one_row",
        scheduler=scheduler,
        **data["one"],
        **common,
        denoiser_spec=base_model,
        optimizer_spec=adam_no_clip,
        max_steps=args.random_one_steps,
        seed=32108,
    )
    runs["random_unique_free_16"] = train_random_noise_control(
        name="random_unique_free_16",
        scheduler=scheduler,
        **data["unique"],
        **common,
        denoiser_spec=base_model,
        optimizer_spec=adam_no_clip,
        max_steps=args.random_multi_steps,
        seed=32109,
    )
    runs["random_paired_16"] = train_random_noise_control(
        name="random_paired_16",
        scheduler=scheduler,
        **data["paired"],
        **common,
        denoiser_spec=base_model,
        optimizer_spec=adam_no_clip,
        max_steps=args.random_multi_steps,
        seed=32110,
    )

    pair_ambiguity = pair_ambiguity_metrics(
        arrays,
        paired_16,
        x_z,
        y_raw,
    )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r231_controls_completed",
        "meaning": "train-only diagnostic controls completed; no model repair",
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_present_but_not_used": int(len(validation)),
        },
        "selections": {
            "one_row": one_row.tolist(),
            "unique_free_16": unique_free_16.tolist(),
            "paired_16": paired_16.tolist(),
            "paired_metadata": paired_selection_metadata(
                arrays,
                paired_16,
            ),
        },
        "pair_ambiguity": pair_ambiguity,
        "runs": runs,
        "source_sha256": source_sha256(root),
        "validation_rows_used_by_controls": False,
        "formal_test_read": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    output = root / "reports/phase3_14b_r231_controls_summary.json"
    write_json_once(output, report)

    compact = {
        name: {
            "gate_pass": run.get(
                "gate_pass",
                run.get(
                    "exact_replay_gate_pass",
                    run.get("single_branch_t50_gate_pass"),
                ),
            ),
            "kind": run["control_kind"],
        }
        for name, run in runs.items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
