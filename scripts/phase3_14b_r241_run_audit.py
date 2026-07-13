#!/usr/bin/env python3
"""Run Phase3.14b-r2.4.1 train-only multirow/source-batching audit."""
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
from ccda_phase3.phase314b_r231_controls import (
    select_unique_condition_rows,
)
from ccda_phase3.phase314b_r241_multirow import (
    AUDIT_TIMESTEPS,
    MULTIROW_VARIANTS,
    PHASE,
    DiffusionTrainSpec,
    PriorTrainSpec,
    assert_only_allowed_worktree_paths,
    build_labeled_bank,
    classify_audit,
    condition_identifiability_audit,
    dependency_sha256,
    source_alignment_audit,
    source_sha256,
    train_condition_prior,
    train_multirow_variant,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=2500)
    parser.add_argument("--diffusion-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--evaluation-noises", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(
        root,
        ("reports/phase3_14b_r241_preflight_summary.json",),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("r2.4.1 audit requires CUDA")
    device = torch.device("cuda")

    arrays, manifest, x_raw, x_standardizer, train, _, _, validation = (
        load_verified_inputs(root)
    )
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train index includes non-train rows")

    unique_rows = select_unique_condition_rows(
        arrays,
        train,
        condition="free",
        row_count=16,
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

    data16 = subset(
        unique_rows,
        condition_all=condition_all,
        clean_z_all=clean_z_all,
        clean_raw_all=clean_raw_all,
    )
    scheduler = make_repair_scheduler(
        REPAIR_CONFIGS["v_prediction_cosine"]
    )
    evaluation_noise_seeds = tuple(
        92000 + index for index in range(int(args.evaluation_noises))
    )
    evaluation_bank = build_labeled_bank(
        scheduler=scheduler,
        **data16,
        active_mask=active,
        timesteps=AUDIT_TIMESTEPS,
        noise_seeds=evaluation_noise_seeds,
        matched_noise_across_sources=True,
    )
    alignment = source_alignment_audit(
        bank=evaluation_bank,
        source_condition=data16["condition_z"],
        source_clean_z=data16["clean_z"],
        source_clean_raw=data16["clean_raw"],
    )
    if not alignment["pass"]:
        raise RuntimeError("source-row alignment audit failed")

    identifiability = condition_identifiability_audit(
        data16["condition_z"],
        data16["clean_z"],
        data16["clean_raw"],
    )

    direct_controls = {}
    for index, row_count in enumerate((2, 4, 8, 16)):
        local = {
            key: value[:row_count]
            for key, value in data16.items()
        }
        key = f"rows_{row_count}_width_512"
        direct_controls[key] = train_condition_prior(
            **local,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            spec=PriorTrainSpec(
                steps=int(args.prior_steps),
                learning_rate=float(args.learning_rate),
                hidden_dim=512,
            ),
            seed=93000 + index,
        )

    direct_controls["rows_16_width_1024"] = train_condition_prior(
        **data16,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        spec=PriorTrainSpec(
            steps=int(args.prior_steps),
            learning_rate=float(args.learning_rate),
            hidden_dim=1024,
        ),
        seed=93010,
    )

    direct_capacity_supported = bool(
        direct_controls["rows_16_width_512"]["pass"]
        or direct_controls["rows_16_width_1024"]["pass"]
    )
    variants = {}
    if alignment["pass"] and direct_capacity_supported:
        train_spec = DiffusionTrainSpec(
            warmup_steps=int(args.warmup_steps),
            diffusion_steps=int(args.diffusion_steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
        )
        for index, variant in enumerate(MULTIROW_VARIANTS):
            variants[variant.name] = train_multirow_variant(
                scheduler=scheduler,
                **data16,
                evaluation_bank=evaluation_bank,
                active_mask=active,
                future_mean=future_mean,
                future_scale=future_scale,
                variant=variant,
                train_spec=train_spec,
                seed=94000 + index,
            )

    visible_seed = np.asarray(arrays["visible_seed"]).astype(np.int64)
    condition_name = np.asarray(arrays["condition_name"]).astype(str)
    pair_key = np.asarray(arrays["pair_key"]).astype(str)
    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "meaning": (
            "train-only conditioned multirow/source-batching audit "
            "completed; no formal model selection"
        ),
        "repository": repository,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dataset": {
            "manifest_schema": manifest.get("schema_version"),
            "train_rows": int(len(train)),
            "validation_rows_verified_but_not_used": int(len(validation)),
            "selected_rows": unique_rows.tolist(),
            "selected_visible_seeds": visible_seed[unique_rows].tolist(),
            "selected_conditions": condition_name[unique_rows].tolist(),
            "selected_pair_keys": pair_key[unique_rows].tolist(),
            "distinct_visible_seed_count": int(
                len(set(visible_seed[unique_rows].tolist()))
            ),
        },
        "evaluation_bank": {
            "timesteps": list(AUDIT_TIMESTEPS),
            "noise_seeds": list(evaluation_noise_seeds),
            "row_count": int(evaluation_bank.row_count),
            "matched_noise_across_sources": True,
        },
        "source_alignment": alignment,
        "condition_identifiability": identifiability,
        "direct_condition_controls": direct_controls,
        "diffusion_variants": variants,
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
    report.update(classify_audit(report))
    output = root / "reports/phase3_14b_r241_audit_summary.json"
    write_json_once(output, report)

    compact = {
        "root_cause": report["root_cause"],
        "next_stage": report["next_stage"],
        "train_only_debug_recommendation": report[
            "train_only_debug_recommendation"
        ],
        "direct_controls": {
            key: bool(value["pass"])
            for key, value in direct_controls.items()
        },
        "diffusion_variants": {
            key: bool(value["pass"])
            for key, value in variants.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
