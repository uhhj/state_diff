#!/usr/bin/env python3
"""Evaluate the selected stable Phase3.14b-r2 checkpoint on formal test."""
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
from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_metrics import (
    candidate_final_chamfer,
    nested_best_of_k,
    pairwise_pool_diversity,
    sample_mean_future,
)
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.phase314b_r2_contract import (
    CACHE_SHA256,
    K_VALUES,
    REPAIR_CONFIGS,
    TEST_MIN_BOTH_BRANCH_SUPPORT_K16,
    TEST_MIN_PHYSICAL_VALIDITY,
    TEST_MIN_QUERY_VALIDITY,
    load_r2_inputs,
    test_indices_and_pairs,
    train_indices,
)
from ccda_phase3.phase314b_r2_diffusion import (
    make_repair_scheduler,
    sample_future_z,
)
from ccda_phase3.phase314b_r2_metrics import (
    ValidityContract,
    branch_support_summary,
    evaluate_pool,
    fit_validity_contract,
    improvement_bootstrap,
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def load_deterministic_prediction(
    *,
    root: Path,
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    future_std,
    device: torch.device,
) -> np.ndarray:
    selection = strict_json_load(
        root / "reports/phase3_14a_selected_deterministic_model.json"
    )
    checkpoint_path = root / str(selection["checkpoint"])
    if sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("deterministic checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("phase") != "phase3_14a":
        raise RuntimeError("not a Phase3.14a checkpoint")
    if checkpoint.get("input_variant") != "state_action":
        raise RuntimeError("unexpected deterministic input variant")

    from ccda_phase3.phase314b_contract import (
        input_values_and_standardizer,
    )

    x_raw, x_std = input_values_and_standardizer(
        arrays,
        "state_action",
    )
    x_z = x_std.transform(x_raw[indices])
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
            batch = torch.from_numpy(
                x_z[start:start + 512]
            ).to(device)
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
        "--selection",
        default="reports/phase3_14b_r2_selected_model.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r2_test_summary.json",
    )
    parser.add_argument(
        "--query-csv",
        default="reports/phase3_14b_r2_test_queries.csv",
    )
    parser.add_argument(
        "--k-csv",
        default="reports/phase3_14b_r2_best_of_k.csv",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r2_test_report.md",
    )
    parser.add_argument(
        "--sample-output",
        default=(
            "data/phase3_14b_r2_eval/"
            "selected_test_sample_pool.npz"
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=385000)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--allow-cpu-eval", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PHASE314B_R2_ALLOW_TEST") != "1":
        raise SystemExit("PHASE314B_R2_ALLOW_TEST must be 1")
    if os.environ.get("PHASE314B_R2_TEST_CONFIRMED") != "1":
        raise SystemExit("PHASE314B_R2_TEST_CONFIRMED must be 1")

    root = Path(args.root).resolve()
    selection_gate = strict_json_load(
        root / "reports/phase3_14b_r2_selection_summary.json"
    )
    if selection_gate.get("verdict") != "PASS":
        raise RuntimeError("r2 validation selection did not pass")
    if selection_gate.get("test_evaluation_allowed") is not True:
        raise RuntimeError("test evaluation is not authorized")

    selection = strict_json_load(root / args.selection)
    if selection.get("test_used_for_selection") is not False:
        raise RuntimeError("test was used during r2 selection")
    checkpoint_path = root / str(selection["checkpoint"])
    if sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("r2 selected checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("phase") != "phase3_14b_r2":
        raise RuntimeError("not an r2 checkpoint")
    if checkpoint.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("r2 checkpoint used another cache")
    for relative, expected in checkpoint["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(
                f"r2 checkpoint-bound source changed: {relative}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu_eval:
        raise SystemExit(
            "Formal K=32 test should use CUDA; pass --allow-cpu-eval "
            "only after accepting runtime cost"
        )

    arrays, _, x_raw, x_std = load_r2_inputs(root)
    test_indices, pair_map = test_indices_and_pairs(arrays)
    train = train_indices(arrays)
    x_z = x_std.transform(x_raw)
    future_std = future_standardizer(arrays)
    future_active = np.asarray(
        arrays["future_active"],
        dtype=np.bool_,
    )
    validity = fit_validity_contract(arrays["y_state"][train])

    config_name = str(selection["repair_config"])
    config = REPAIR_CONFIGS[config_name]
    model = build_denoiser(
        "mlp_ddpm",
        condition_dim=x_z.shape[1],
    ).to(device)
    model.load_state_dict(
        checkpoint["ema_state_dict"]["shadow"],
        strict=True,
    )
    model.eval()

    sample_z_tensor = sample_future_z(
        model=model,
        scheduler=make_repair_scheduler(config),
        config=config,
        condition_z=torch.from_numpy(
            x_z[test_indices]
        ).to(device),
        active_mask=torch.from_numpy(future_active).to(device),
        num_samples=32,
        seed=int(args.sample_seed),
        num_inference_steps=100,
        row_batch_size=128,
    )
    sample_z = sample_z_tensor.numpy().astype(np.float32)
    sample_raw = future_std.inverse(sample_z)
    target = np.asarray(
        arrays["y_state"][test_indices],
        dtype=np.float32,
    )
    metrics = evaluate_pool(
        sample_pool_z=sample_z,
        sample_pool_raw=sample_raw,
        target_raw=target,
        active_mask=future_active,
        validity_contract=validity,
        k_values=K_VALUES,
    )

    deterministic = load_deterministic_prediction(
        root=root,
        arrays=arrays,
        indices=test_indices,
        future_std=future_std,
        device=device,
    )
    deterministic_error = candidate_final_chamfer(
        deterministic[None, ...],
        target,
    )[0]
    nested = metrics["nested"]
    visible_seed = np.asarray(
        arrays["visible_seed"][test_indices],
        dtype=np.int64,
    )
    improvement_vs_deterministic = improvement_bootstrap(
        deterministic_error,
        nested[16],
        visible_seed,
        iterations=int(args.bootstraps),
        seed=386000,
    )
    improvement_vs_k1 = improvement_bootstrap(
        nested[1],
        nested[16],
        visible_seed,
        iterations=int(args.bootstraps),
        seed=386001,
    )
    support16 = branch_support_summary(
        sample_pool=sample_raw,
        row_indices=test_indices,
        pair_map=pair_map,
        arrays=arrays,
        k=16,
    )
    support32 = branch_support_summary(
        sample_pool=sample_raw,
        row_indices=test_indices,
        pair_map=pair_map,
        arrays=arrays,
        k=32,
    )

    physical = metrics["physical_validity"]
    finite = bool(
        np.all(np.isfinite(sample_z))
        and np.all(np.isfinite(sample_raw))
    )
    if not finite:
        verdict = "FAIL"
        root_cause = "phase314b_r2_nonfinite_sampling_failed"
    elif (
        physical["sample_validity_rate"]
        < TEST_MIN_PHYSICAL_VALIDITY
        or physical["query_has_valid_candidate_rate"]
        < TEST_MIN_QUERY_VALIDITY
    ):
        verdict = "FAIL"
        root_cause = "phase314b_r2_physical_validity_failed"
    elif (
        improvement_vs_deterministic["ci_low"] <= 0.0
        or improvement_vs_k1["ci_low"] <= 0.0
    ):
        verdict = "FAIL"
        root_cause = (
            "phase314b_r2_candidate_headroom_not_supported"
        )
    elif (
        support16["both_supported_rate"]
        < TEST_MIN_BOTH_BRANCH_SUPPORT_K16
    ):
        verdict = "FAIL"
        root_cause = (
            "phase314b_r2_candidate_branch_support_insufficient"
        )
    else:
        verdict = "PASS"
        root_cause = "phase314b_r2_candidate_support_repaired"

    sample_output = root / args.sample_output
    if sample_output.exists():
        if os.environ.get(
            "PHASE314B_R2_ALLOW_SAMPLE_REPLACE"
        ) != "1":
            raise SystemExit("r2 sample pool already exists")
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = sample_output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_pool_z=sample_z,
        sample_pool_raw=sample_raw,
        row_indices=test_indices,
        pair_key=np.asarray(
            arrays["pair_key"][test_indices],
            dtype="<U64",
        ),
        condition_name=np.asarray(
            arrays["condition_name"][test_indices],
            dtype="<U64",
        ),
        visible_seed=visible_seed,
        checkpoint_sha256=np.asarray(
            [selection["checkpoint_sha256"]],
            dtype="<U64",
        ),
        sample_seed=np.asarray([args.sample_seed], dtype=np.int64),
    )
    os.replace(temporary, sample_output)

    query_rows = []
    validity_mask = metrics["sample_valid_mask"]
    for local, original in enumerate(test_indices.tolist()):
        query_rows.append(
            {
                "row_index": int(original),
                "pair_key": str(arrays["pair_key"][original]),
                "visible_seed": int(arrays["visible_seed"][original]),
                "condition": str(arrays["condition_name"][original]),
                "deterministic_error": float(
                    deterministic_error[local]
                ),
                "k1_error": float(nested[1][local]),
                "best4_error": float(nested[4][local]),
                "best8_error": float(nested[8][local]),
                "best16_error": float(nested[16][local]),
                "best32_error": float(nested[32][local]),
                "valid_candidate_k16": bool(
                    np.any(validity_mask[:16, local])
                ),
                "valid_candidate_k32": bool(
                    np.any(validity_mask[:32, local])
                ),
            }
        )
    k_rows = [
        {
            "k": int(k),
            "mean_final_chamfer": metrics["k_metrics"][
                str(k)
            ]["mean"],
            "median_final_chamfer": metrics["k_metrics"][
                str(k)
            ]["median"],
        }
        for k in K_VALUES
    ]

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "cache_sha256": CACHE_SHA256,
        "selected_model": selection,
        "device": str(device),
        "test_pair_keys": len(pair_map),
        "test_query_rows": int(test_indices.size),
        "sample_pool_shape": list(sample_raw.shape),
        "sample_pool_file": str(sample_output.relative_to(root)),
        "sample_pool_sha256": sha256_file(sample_output),
        "k_metrics": k_rows,
        "deterministic_error_mean": float(
            np.mean(deterministic_error)
        ),
        "sample_mean_error": metrics["sample_mean_error"],
        "pool_diversity_mean": metrics["pool_diversity_mean"],
        "physical_validity": physical,
        "z_stats": metrics["z_stats"],
        "improvement_vs_deterministic": improvement_vs_deterministic,
        "improvement_vs_k1": improvement_vs_k1,
        "branch_support_k16": {
            key: value
            for key, value in support16.items()
            if key != "rows"
        },
        "branch_support_k32": {
            key: value
            for key, value in support32.items()
            if key != "rows"
        },
        "test_used_for_selection": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    write_csv(root / args.query_csv, query_rows)
    write_csv(root / args.k_csv, k_rows)

    lines = [
        "# Phase3.14b-r2 Formal Test",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Selected config: `{config_name}`",
        f"- Selected seed: `{selection['training_seed']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        (
            "| Physical validity | "
            f"{physical['sample_validity_rate']:.8f} |"
        ),
        (
            "| Query validity | "
            f"{physical['query_has_valid_candidate_rate']:.8f} |"
        ),
        (
            "| Best16 vs deterministic CI low | "
            f"{improvement_vs_deterministic['ci_low']:.8f} |"
        ),
        (
            "| Best16 vs K1 CI low | "
            f"{improvement_vs_k1['ci_low']:.8f} |"
        ),
        (
            "| K16 both-branch support | "
            f"{support16['both_supported_rate']:.8f} |"
        ),
        (
            "| K32 both-branch support | "
            f"{support32['both_supported_rate']:.8f} |"
        ),
        "",
        "- IDM, candidate action execution, Phase4, and CPS were not run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
