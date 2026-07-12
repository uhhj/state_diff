#!/usr/bin/env python3
"""Rerun Phase3.14b-r1 with exact Diffusers-0.11.1 step arithmetic."""
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
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    fixed_balanced_eval_indices,
    future_standardizer,
    load_locked_cache,
)
from ccda_phase3.phase314b_diffusion import make_ddpm_scheduler
from ccda_phase3.phase314b_r1_diagnostics import (
    SELECTED_CHECKPOINT_SHA256,
    compare_weight_sets,
    load_selected_model,
    scheduler_coefficients,
    standardization_audit,
    timestep_model_audit,
    true_epsilon_scheduler_oracle,
)
from ccda_phase3.phase314b_r11_exact_scheduler import (
    exact_partial_denoise_audit,
    exact_reverse_chain_trace,
    exact_step_equivalence,
    final_result_stats,
    scheduler_source_contract,
    summarize_equivalence,
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


def flatten_equivalence_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    flattened = []
    for row in rows:
        base = {
            "device": row["device"],
            "dtype": row["dtype"],
            "timestep": row["timestep"],
            "exact_beta": row["exact_beta"],
            "exact_alpha": row["exact_alpha"],
            "exact_variance": row["exact_variance"],
        }
        for component in (
            "pred_x0",
            "posterior_mean_plus_variance",
            "variance_term",
            "legacy_prev_vs_official",
            "legacy_pred_x0_vs_official",
        ):
            for key, value in row[component].items():
                if key == "max_index":
                    value = json.dumps(value)
                base[f"{component}_{key}"] = value
        flattened.append(base)
    return flattened


def add_label(rows: List[Dict[str, Any]], **labels: Any):
    return [{**labels, **row} for row in rows]


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
    parser.add_argument("--max-val-pair-keys", type=int, default=8)
    parser.add_argument("--rows-per-timestep-bin", type=int, default=64)
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r11_audit_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r11_audit_report.md",
    )
    parser.add_argument(
        "--equivalence-csv",
        default="reports/phase3_14b_r11_equivalence.csv",
    )
    parser.add_argument(
        "--bins-csv",
        default="reports/phase3_14b_r11_timestep_bins.csv",
    )
    parser.add_argument(
        "--trace-csv",
        default="reports/phase3_14b_r11_reverse_trace.csv",
    )
    parser.add_argument(
        "--partial-csv",
        default="reports/phase3_14b_r11_partial_denoise.csv",
    )
    args = parser.parse_args()

    if os.environ.get("PHASE314B_R11_ALLOW_DIAGNOSTICS") != "1":
        raise SystemExit(
            "PHASE314B_R11_ALLOW_DIAGNOSTICS must be 1"
        )
    if os.environ.get("PHASE314B_R11_DIAGNOSTICS_CONFIRMED") != "1":
        raise SystemExit(
            "PHASE314B_R11_DIAGNOSTICS_CONFIRMED must be 1"
        )

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_r11_preflight_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14b-r1.1 preflight is not PASS")

    arrays, cache_manifest = load_locked_cache(
        root / args.cache,
        root / args.cache_manifest,
    )
    indices = fixed_balanced_eval_indices(
        arrays,
        split="val",
        max_pair_keys=int(args.max_val_pair_keys),
    )
    if indices.size < 4:
        raise RuntimeError("diagnostic subset is too small")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    future_std = future_standardizer(arrays)
    target_raw = np.asarray(
        arrays["y_state"][indices],
        dtype=np.float32,
    )
    x0_z_np = future_std.transform(target_raw)
    active_np = np.asarray(
        arrays["future_active"],
        dtype=np.bool_,
    )
    x0_z = torch.from_numpy(x0_z_np).to(device)
    active = torch.from_numpy(active_np).to(device)

    source_contract = scheduler_source_contract()
    standardization = standardization_audit(arrays)
    scheduler = make_ddpm_scheduler()
    coefficients = scheduler_coefficients(scheduler)

    oracle_noise = torch.randn(
        x0_z.shape,
        generator=torch.Generator(device=device).manual_seed(980000),
        device=device,
        dtype=x0_z.dtype,
    )
    oracle_noise = torch.where(
        active[None, :, :],
        oracle_noise,
        torch.zeros_like(oracle_noise),
    )
    oracle_rows = true_epsilon_scheduler_oracle(
        scheduler,
        x0=x0_z,
        noise=oracle_noise,
    )

    equivalence_by_device = {}
    all_equivalence_rows = []
    for equivalence_device in (
        [torch.device("cpu")]
        + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    ):
        rows = exact_step_equivalence(
            device=equivalence_device,
            shape=(4, 4, 87),
            seed=981000,
        )
        equivalence_by_device[str(equivalence_device)] = {
            "summary": summarize_equivalence(rows),
            "rows": rows,
        }
        all_equivalence_rows.extend(rows)

    weight_results: Dict[str, Any] = {}
    timestep_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    partial_rows: List[Dict[str, Any]] = []

    for use_ema in (True, False):
        label = "ema" if use_ema else "raw"
        model, condition_all, metadata = load_selected_model(
            root=root,
            arrays=arrays,
            device=device,
            use_ema=use_ema,
        )
        condition = torch.from_numpy(
            condition_all[indices]
        ).to(device)

        bins = timestep_model_audit(
            model=model,
            scheduler=make_ddpm_scheduler(),
            x0_z=x0_z,
            condition_z=condition,
            active_mask=active,
            rows_per_timestep=int(args.rows_per_timestep_bin),
            seed=982000 if use_ema else 982001,
        )
        timestep_rows.extend(add_label(bins, weights=label))

        stochastic_final, stochastic_trace = exact_reverse_chain_trace(
            model=model,
            scheduler=make_ddpm_scheduler(),
            condition_z=condition,
            active_mask=active,
            future_std=future_std,
            seed=983000,
            posterior_noise=True,
        )
        mean_final, mean_trace = exact_reverse_chain_trace(
            model=model,
            scheduler=make_ddpm_scheduler(),
            condition_z=condition,
            active_mask=active,
            future_std=future_std,
            seed=983000,
            posterior_noise=False,
        )
        trace_rows.extend(
            add_label(
                stochastic_trace,
                weights=label,
                reverse_mode="stochastic_ddpm",
            )
        )
        trace_rows.extend(
            add_label(
                mean_trace,
                weights=label,
                reverse_mode="posterior_mean",
            )
        )

        partial = exact_partial_denoise_audit(
            model=model,
            scheduler=make_ddpm_scheduler(),
            x0_z=x0_z,
            target_raw=target_raw,
            condition_z=condition,
            active_mask=active,
            future_std=future_std,
            seed=984000,
        )
        partial_rows.extend(add_label(partial, weights=label))

        weight_results[label] = {
            "metadata": metadata,
            "timestep_bins": bins,
            "stochastic_ddpm": final_result_stats(
                final_z=stochastic_final,
                future_std=future_std,
                target_raw=target_raw,
                active_mask=active_np,
            ),
            "posterior_mean": final_result_stats(
                final_z=mean_final,
                future_std=future_std,
                target_raw=target_raw,
                active_mask=active_np,
            ),
            "partial_denoise": partial,
        }

    comparison = compare_weight_sets(
        weight_results["ema"],
        weight_results["raw"],
    )
    ema_bins = {
        row["bin"]: row
        for row in weight_results["ema"]["timestep_bins"]
    }
    ema_partial = {
        row["start_timestep"]: row
        for row in weight_results["ema"]["partial_denoise"]
    }

    runtime_equivalence = equivalence_by_device[str(device)]["summary"]
    gates = {
        "standardization_roundtrip_max_abs": standardization[
            "roundtrip_max_abs"
        ],
        "scheduler_true_epsilon_x0_max_abs": max(
            row["diffusers_x0_max_abs_error"]
            for row in oracle_rows
        ),
        "exact_prev_allclose": runtime_equivalence[
            "all_exact_prev_close"
        ],
        "exact_pred_x0_allclose": runtime_equivalence[
            "all_pred_x0_close"
        ],
        "exact_prev_max_abs": runtime_equivalence[
            "exact_prev_max_abs"
        ],
        "exact_prev_max_relative_rms": runtime_equivalence[
            "exact_prev_max_relative_rms"
        ],
        "legacy_prev_max_abs": runtime_equivalence[
            "legacy_prev_max_abs"
        ],
        "terminal_x0_error_amplification": coefficients[-1][
            "x0_error_amplification"
        ],
        "ema_high_bin_pred_x0_mse": ema_bins["90-99"][
            "pred_x0_mse"
        ],
        "ema_low_bin_pred_x0_mse": ema_bins["0-9"][
            "pred_x0_mse"
        ],
        "ema_high_to_low_x0_mse_ratio": comparison[
            "ema_high_to_low_ratio"
        ],
        "ema_partial_t10_chamfer": ema_partial[10][
            "final_chamfer_mean"
        ],
        "ema_partial_t99_chamfer": ema_partial[99][
            "final_chamfer_mean"
        ],
        "ema_stochastic_final_chamfer": weight_results["ema"][
            "stochastic_ddpm"
        ]["final_chamfer_mean"],
        "ema_posterior_mean_final_chamfer": weight_results["ema"][
            "posterior_mean"
        ]["final_chamfer_mean"],
        "raw_stochastic_final_chamfer": weight_results["raw"][
            "stochastic_ddpm"
        ]["final_chamfer_mean"],
        "raw_posterior_mean_final_chamfer": weight_results["raw"][
            "posterior_mean"
        ]["final_chamfer_mean"],
    }

    previous = strict_json_load(
        root / "reports/phase3_14b_r1_audit_summary.json"
    )
    summary = {
        "verdict": "PASS",
        "root_cause": "phase314b_r11_exact_scheduler_audit_completed",
        "cache_sha256": CACHE_SHA256,
        "checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "device": str(device),
        "diagnostic_indices": indices.tolist(),
        "diagnostic_rows": int(indices.size),
        "scheduler_source_contract": source_contract,
        "standardization": standardization,
        "scheduler_coefficients": {
            "first": coefficients[0],
            "last": coefficients[-1],
            "selected": [
                coefficients[index]
                for index in (0, 10, 25, 50, 75, 90, 99)
            ],
        },
        "true_epsilon_oracle": oracle_rows,
        "equivalence_by_device": equivalence_by_device,
        "weights": weight_results,
        "ema_vs_raw": comparison,
        "gates": gates,
        "previous_r1_gates": previous["gates"],
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, summary)
    write_csv(
        root / args.equivalence_csv,
        flatten_equivalence_rows(all_equivalence_rows),
    )
    write_csv(root / args.bins_csv, timestep_rows)
    write_csv(root / args.trace_csv, trace_rows)
    write_csv(root / args.partial_csv, partial_rows)

    lines = [
        "# Phase3.14b-r1.1 Exact Scheduler Audit",
        "",
        "- Verdict: `PASS` (diagnostic execution completed)",
        (
            "- Root cause: "
            "`phase314b_r11_exact_scheduler_audit_completed`"
        ),
        f"- Runtime device: `{device}`",
        f"- Diagnostic rows: `{indices.size}`",
        "",
        "| Gate | Value |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "The separate analyzer assigns the scientific root cause.",
        "No retraining, IDM, candidate execution, Phase4, or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "root_cause": (
                    "phase314b_r11_exact_scheduler_audit_completed"
                ),
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
