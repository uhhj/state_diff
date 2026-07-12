#!/usr/bin/env python3
"""Evaluate selected DDPM on test candidate support without IDM/execution."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    K_VALUES,
    future_standardizer,
    input_values_and_standardizer,
    load_locked_cache,
    paired_key_index,
    source_sha256,
    split_mask,
)
from ccda_phase3.phase314b_diffusion import (
    make_ddpm_scheduler,
    sample_future_z,
)
from ccda_phase3.phase314b_metrics import (
    branch_support_for_pair,
    candidate_final_chamfer,
    candidate_physical_validity,
    fit_segment_bounds,
    improvement_bootstrap,
    nested_best_of_k,
    pairwise_pool_diversity,
    sample_mean_future,
)
from ccda_phase3.phase314b_models import build_denoiser


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_selected_deterministic(
    *,
    root: Path,
    arrays: Dict[str, np.ndarray],
    row_indices: np.ndarray,
    future_std,
    device: torch.device,
) -> np.ndarray:
    selected = strict_json_load(
        root / "reports/phase3_14a_selected_deterministic_model.json"
    )
    checkpoint_path = root / str(selected["checkpoint"])
    if sha256_file(checkpoint_path) != selected["checkpoint_sha256"]:
        raise RuntimeError("deterministic checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("phase") != "phase3_14a":
        raise RuntimeError("not a Phase3.14a deterministic checkpoint")
    variant = str(checkpoint["input_variant"])
    x_raw, x_std = input_values_and_standardizer(arrays, variant)
    x_z = x_std.transform(x_raw[row_indices])

    model = torch.nn.Sequential(
        torch.nn.Linear(int(checkpoint["input_dim"]), 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 4 * 87),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, x_z.shape[0], 512):
            batch = torch.from_numpy(x_z[start:start + 512]).to(device)
            outputs.append(
                model(batch)
                .reshape(-1, 4, 87)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return future_std.inverse(np.concatenate(outputs, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--cache",
        default=(
            "data/phase3_14_cache/"
            "phase3_14a_training_cache.npz"
        ),
    )
    parser.add_argument(
        "--cache-manifest",
        default=(
            "data/phase3_14_cache/"
            "phase3_14a_training_cache_manifest.json"
        ),
    )
    parser.add_argument(
        "--selection",
        default="reports/phase3_14b_selected_model.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_candidate_summary.json",
    )
    parser.add_argument(
        "--query-csv",
        default="reports/phase3_14b_candidate_queries.csv",
    )
    parser.add_argument(
        "--k-csv",
        default="reports/phase3_14b_best_of_k.csv",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_candidate_report.md",
    )
    parser.add_argument(
        "--sample-output",
        default=(
            "data/phase3_14b_eval/"
            "selected_test_sample_pool.npz"
        ),
    )
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--sample-seed", type=int, default=350000)
    parser.add_argument("--allow-cpu-eval", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PHASE314B_ALLOW_OFFLINE_SUPPORT") != "1":
        raise SystemExit("PHASE314B_ALLOW_OFFLINE_SUPPORT must be 1")
    if os.environ.get("PHASE314B_OFFLINE_SUPPORT_CONFIRMED") != "1":
        raise SystemExit(
            "PHASE314B_OFFLINE_SUPPORT_CONFIRMED must be 1"
        )
    if int(args.max_k) != 32:
        raise ValueError("formal Phase3.14b evaluation requires Kmax=32")

    root = Path(args.root).resolve()
    arrays, cache_manifest = load_locked_cache(
        root / args.cache,
        root / args.cache_manifest,
    )
    selection = strict_json_load(root / args.selection)
    if selection.get("test_used_for_selection") is not False:
        raise RuntimeError("test split was used during model selection")
    checkpoint_path = root / str(selection["checkpoint"])
    if sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("selected DDPM checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("phase") != "phase3_14b":
        raise RuntimeError("selected checkpoint is not Phase3.14b")
    if checkpoint.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("selected checkpoint used another cache")
    if checkpoint.get("clip_sample") is not False:
        raise RuntimeError("selected checkpoint used clip_sample")
    if checkpoint.get("thresholding") is not False:
        raise RuntimeError("selected checkpoint used thresholding")
    for relative, expected in checkpoint["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(
                f"selected checkpoint source changed: {relative}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu_eval:
        raise SystemExit(
            "K=32 test sampling should use a GPU. "
            "Pass --allow-cpu-eval only when runtime is accepted."
        )

    selected_mask = split_mask(
        arrays,
        "test",
        full_horizon_only=True,
        pre_engagement_only=True,
    )
    pairs = paired_key_index(arrays, selected_mask)
    if not pairs:
        raise RuntimeError("no eligible paired test rows")
    pair_keys = sorted(pairs)
    row_indices = np.asarray(
        [
            index
            for key in pair_keys
            for index in pairs[key]
        ],
        dtype=np.int64,
    )
    row_lookup = {
        int(original): local
        for local, original in enumerate(row_indices.tolist())
    }

    variant = str(selection["input_variant"])
    x_raw, x_std = input_values_and_standardizer(arrays, variant)
    condition_z = x_std.transform(x_raw[row_indices])
    future_std = future_standardizer(arrays)
    future_active = np.asarray(
        arrays["future_active"],
        dtype=np.bool_,
    )

    model = build_denoiser(
        str(selection["model_family"]),
        condition_dim=condition_z.shape[1],
    ).to(device)
    model.load_state_dict(
        checkpoint["ema_state_dict"]["shadow"],
        strict=True,
    )
    model.eval()

    samples_z = sample_future_z(
        model=model,
        scheduler=make_ddpm_scheduler(),
        condition_z=torch.from_numpy(condition_z).to(device),
        active_mask=torch.from_numpy(future_active).to(device),
        num_samples=32,
        seed=int(args.sample_seed),
        num_inference_steps=100,
        sample_batch_size=128,
    )
    sample_pool = future_std.inverse(
        samples_z.numpy().astype(np.float32)
    )
    target = np.asarray(
        arrays["y_state"][row_indices],
        dtype=np.float32,
    )
    distances = candidate_final_chamfer(sample_pool, target)
    nested = nested_best_of_k(distances, K_VALUES)
    mean_future = sample_mean_future(sample_pool)
    mean_distance = candidate_final_chamfer(
        mean_future[None, ...],
        target,
    )[0]
    diversity = pairwise_pool_diversity(sample_pool)

    deterministic = load_selected_deterministic(
        root=root,
        arrays=arrays,
        row_indices=row_indices,
        future_std=future_std,
        device=device,
    )
    deterministic_distance = candidate_final_chamfer(
        deterministic[None, ...],
        target,
    )[0]

    train_full = split_mask(
        arrays,
        "train",
        full_horizon_only=True,
        pre_engagement_only=False,
    )
    lower, upper = fit_segment_bounds(
        arrays["y_state"][train_full]
    )
    physical = candidate_physical_validity(
        sample_pool[:16],
        lower,
        upper,
    )

    conditions = arrays["condition_name"][row_indices].astype(str)
    visible_seeds = arrays["visible_seed"][row_indices].astype(np.int64)
    pair_key_values = arrays["pair_key"][row_indices].astype(str)
    query_rows: List[Dict[str, Any]] = []
    support16 = []
    support32 = []

    for key in pair_keys:
        free_original, hidden_original = pairs[key]
        free_local = row_lookup[free_original]
        hidden_local = row_lookup[hidden_original]
        free_final = arrays["y_state"][free_original, -1]
        hidden_final = arrays["y_state"][hidden_original, -1]
        for local in (free_local, hidden_local):
            result16 = branch_support_for_pair(
                sample_pool[:16, local, -1, :],
                free_final=free_final,
                hidden_final=hidden_final,
            )
            result32 = branch_support_for_pair(
                sample_pool[:32, local, -1, :],
                free_final=free_final,
                hidden_final=hidden_final,
            )
            support16.append(result16)
            support32.append(result32)
            query_rows.append(
                {
                    "pair_key": key,
                    "visible_seed": int(visible_seeds[local]),
                    "condition": conditions[local],
                    "branch_eligible": result16["eligible"],
                    "branch_separation": result16[
                        "branch_separation"
                    ],
                    "k1_error": float(nested[1][local]),
                    "best4_error": float(nested[4][local]),
                    "best8_error": float(nested[8][local]),
                    "best16_error": float(nested[16][local]),
                    "best32_error": float(nested[32][local]),
                    "deterministic_error": float(
                        deterministic_distance[local]
                    ),
                    "sample_mean_error": float(mean_distance[local]),
                    "diversity": float(diversity[local]),
                    "both_supported_k16": (
                        bool(result16["both_supported"])
                        if result16["eligible"]
                        else False
                    ),
                    "both_supported_k32": (
                        bool(result32["both_supported"])
                        if result32["eligible"]
                        else False
                    ),
                    "physical_valid_candidate_k16": bool(
                        np.any(
                            physical["valid_mask"][:, local]
                        )
                    ),
                }
            )

    eligible16 = [item for item in support16 if item["eligible"]]
    eligible32 = [item for item in support32 if item["eligible"]]
    if not eligible16:
        raise RuntimeError("no branch-separated test queries")

    improvement_vs_deterministic = improvement_bootstrap(
        deterministic_distance,
        nested[16],
        visible_seeds,
        iterations=int(args.bootstraps),
        seed=351000,
    )
    improvement_vs_k1 = improvement_bootstrap(
        nested[1],
        nested[16],
        visible_seeds,
        iterations=int(args.bootstraps),
        seed=351001,
    )
    both16 = float(
        np.mean([item["both_supported"] for item in eligible16])
    )
    both32 = float(
        np.mean([item["both_supported"] for item in eligible32])
    )

    k_rows = []
    for k in K_VALUES:
        k_rows.append(
            {
                "k": k,
                "mean_best_of_k_final_chamfer": float(
                    np.mean(nested[k])
                ),
                "median_best_of_k_final_chamfer": float(
                    np.median(nested[k])
                ),
            }
        )

    finite = bool(np.all(np.isfinite(sample_pool)))
    gate_values = {
        "best16_vs_deterministic_ci_low": improvement_vs_deterministic[
            "ci_low"
        ],
        "best16_vs_k1_ci_low": improvement_vs_k1["ci_low"],
        "both_branch_support_rate_k16": both16,
        "both_branch_support_rate_k32": both32,
        "physical_sample_validity_rate_k16": physical[
            "sample_validity_rate"
        ],
        "query_has_valid_candidate_rate_k16": physical[
            "query_has_valid_candidate_rate"
        ],
        "finite_samples": finite,
    }

    if not finite:
        verdict = "FAIL"
        root_cause = "phase314b_nonfinite_sampling_failed"
    elif physical["sample_validity_rate"] < 0.90:
        verdict = "FAIL"
        root_cause = "phase314b_candidate_physical_validity_failed"
    elif (
        improvement_vs_deterministic["ci_low"] <= 0.0
        or improvement_vs_k1["ci_low"] <= 0.0
    ):
        verdict = "FAIL"
        root_cause = (
            "phase314b_ddpm_not_better_than_deterministic_baseline"
        )
    elif both16 < 0.20:
        verdict = "FAIL"
        root_cause = "phase314b_candidate_branch_support_insufficient"
    else:
        verdict = "PASS"
        root_cause = "phase314b_ddpm_candidate_support_supported"

    sample_output = root / args.sample_output
    if sample_output.exists():
        if os.environ.get("PHASE314B_ALLOW_SAMPLE_REPLACE") != "1":
            raise SystemExit(
                "Phase3.14b sample pool already exists; replacement is blocked"
            )
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = sample_output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_pool=sample_pool.astype(np.float32),
        row_indices=row_indices,
        pair_key=np.asarray(pair_key_values, dtype="<U64"),
        condition_name=np.asarray(conditions, dtype="<U64"),
        visible_seed=visible_seeds,
        checkpoint_sha256=np.asarray(
            [selection["checkpoint_sha256"]],
            dtype="<U64",
        ),
        sample_seed=np.asarray([args.sample_seed], dtype=np.int64),
    )
    os.replace(temporary, sample_output)

    summary = {
        "verdict": verdict,
        "root_cause": root_cause,
        "cache_sha256": CACHE_SHA256,
        "selected_model": selection,
        "test_pair_keys": len(pair_keys),
        "test_query_rows": int(row_indices.size),
        "branch_eligible_queries": len(eligible16),
        "sample_pool_shape": list(sample_pool.shape),
        "sample_pool_file": str(sample_output.relative_to(root)),
        "sample_pool_sha256": sha256_file(sample_output),
        "k_metrics": k_rows,
        "sample_mean_error": float(np.mean(mean_distance)),
        "deterministic_error": float(
            np.mean(deterministic_distance)
        ),
        "diversity": float(np.mean(diversity)),
        "improvement_vs_deterministic": improvement_vs_deterministic,
        "improvement_vs_k1": improvement_vs_k1,
        "gate_values": gate_values,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, summary)
    write_csv(root / args.query_csv, query_rows)
    write_csv(root / args.k_csv, k_rows)

    lines = [
        "# Phase3.14b DDPM Candidate Support",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Selected model: `{selection['model_family']}`",
        f"- Input: `{selection['input_variant']}`",
        f"- Seed: `{selection['training_seed']}`",
        f"- Test pair keys: `{len(pair_keys)}`",
        f"- Branch-eligible queries: `{len(eligible16)}`",
        "",
        "| Gate | Value |",
        "|---|---:|",
    ]
    for key, value in gate_values.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "- IDM and query-local execution were not run.",
        "- Phase4 and CPS remain blocked.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
