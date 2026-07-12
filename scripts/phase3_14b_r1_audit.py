#!/usr/bin/env python3
"""Run the Phase3.14b-r1 sampling-scale root-cause audit."""
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
    final_chamfer_rows,
    finite_stats,
    load_selected_model,
    manual_step_equivalence,
    partial_denoise_audit,
    reverse_chain_trace,
    scheduler_coefficients,
    standardization_audit,
    timestep_model_audit,
    true_epsilon_scheduler_oracle,
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


def add_label(
    rows: List[Dict[str, Any]],
    **labels: Any,
) -> List[Dict[str, Any]]:
    return [{**labels, **row} for row in rows]


def final_result_stats(
    *,
    final_z: torch.Tensor,
    future_std,
    target_raw: np.ndarray,
    active_mask: np.ndarray,
) -> Dict[str, Any]:
    active = torch.from_numpy(active_mask).to(
        device=final_z.device,
        dtype=torch.bool,
    )
    active_values = final_z[
        active[None, :, :].expand_as(final_z)
    ]
    raw = future_std.inverse(
        final_z.detach().cpu().numpy().astype(np.float32)
    )
    chamfer = final_chamfer_rows(raw, target_raw)
    return {
        "z": finite_stats(active_values.detach().cpu().numpy()),
        "raw_xy": finite_stats(raw[..., :48]),
        "final_chamfer_mean": float(np.mean(chamfer)),
        "final_chamfer_median": float(np.median(chamfer)),
        "final_chamfer_max": float(np.max(chamfer)),
    }


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
        default="reports/phase3_14b_r1_audit_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r1_audit_report.md",
    )
    parser.add_argument(
        "--scheduler-csv",
        default="reports/phase3_14b_r1_scheduler_coefficients.csv",
    )
    parser.add_argument(
        "--oracle-csv",
        default="reports/phase3_14b_r1_scheduler_oracle.csv",
    )
    parser.add_argument(
        "--bins-csv",
        default="reports/phase3_14b_r1_timestep_bins.csv",
    )
    parser.add_argument(
        "--trace-csv",
        default="reports/phase3_14b_r1_reverse_trace.csv",
    )
    parser.add_argument(
        "--partial-csv",
        default="reports/phase3_14b_r1_partial_denoise.csv",
    )
    args = parser.parse_args()

    if os.environ.get("PHASE314B_R1_ALLOW_DIAGNOSTICS") != "1":
        raise SystemExit("PHASE314B_R1_ALLOW_DIAGNOSTICS must be 1")
    if os.environ.get("PHASE314B_R1_DIAGNOSTICS_CONFIRMED") != "1":
        raise SystemExit(
            "PHASE314B_R1_DIAGNOSTICS_CONFIRMED must be 1"
        )

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_r1_preflight_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14b-r1 preflight is not PASS")

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
        raise RuntimeError("diagnostic validation subset is too small")

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

    standardization = standardization_audit(arrays)
    scheduler = make_ddpm_scheduler()
    coefficients = scheduler_coefficients(scheduler)

    generator = torch.Generator(device=device).manual_seed(930000)
    oracle_noise = torch.randn(
        x0_z.shape,
        generator=generator,
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
    equivalence_rows = manual_step_equivalence(
        scheduler,
        epsilon=oracle_noise,
        sample=torch.randn(
            x0_z.shape,
            generator=torch.Generator(
                device=device
            ).manual_seed(930001),
            device=device,
            dtype=x0_z.dtype,
        ),
    )
    scheduler_oracle_rows = []
    for oracle, equivalence in zip(oracle_rows, equivalence_rows):
        scheduler_oracle_rows.append(
            {
                **oracle,
                **{
                    f"manual_{key}": value
                    for key, value in equivalence.items()
                    if key != "timestep"
                },
            }
        )

    weight_results: Dict[str, Any] = {}
    timestep_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    partial_rows: List[Dict[str, Any]] = []

    for use_ema in (True, False):
        label = "ema" if use_ema else "raw"
        model, condition_all, model_meta = load_selected_model(
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
            seed=940000 if use_ema else 940001,
        )
        timestep_rows.extend(add_label(bins, weights=label))

        posterior_final, posterior_trace = reverse_chain_trace(
            model=model,
            scheduler=make_ddpm_scheduler(),
            condition_z=condition,
            active_mask=active,
            future_std=future_std,
            seed=950000,
            posterior_noise=True,
        )
        mean_final, mean_trace = reverse_chain_trace(
            model=model,
            scheduler=make_ddpm_scheduler(),
            condition_z=condition,
            active_mask=active,
            future_std=future_std,
            seed=950000,
            posterior_noise=False,
        )
        trace_rows.extend(
            add_label(
                posterior_trace,
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

        partial = partial_denoise_audit(
            model=model,
            scheduler=make_ddpm_scheduler(),
            x0_z=x0_z,
            target_raw=target_raw,
            condition_z=condition,
            active_mask=active,
            future_std=future_std,
            seed=960000,
        )
        partial_rows.extend(add_label(partial, weights=label))

        weight_results[label] = {
            "metadata": model_meta,
            "timestep_bins": bins,
            "stochastic_ddpm": final_result_stats(
                final_z=posterior_final,
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

    scheduler_max_oracle = max(
        row["diffusers_x0_max_abs_error"]
        for row in scheduler_oracle_rows
    )
    manual_max = max(
        row["manual_prev_sample_max_abs"]
        for row in scheduler_oracle_rows
    )
    high_amplification = coefficients[-1][
        "x0_error_amplification"
    ]
    ema_bins = {
        row["bin"]: row
        for row in weight_results["ema"]["timestep_bins"]
    }
    ema_partial = {
        row["start_timestep"]: row
        for row in weight_results["ema"]["partial_denoise"]
    }

    gates = {
        "standardization_roundtrip_max_abs": standardization[
            "roundtrip_max_abs"
        ],
        "scheduler_true_epsilon_x0_max_abs": scheduler_max_oracle,
        "manual_vs_diffusers_prev_max_abs": manual_max,
        "terminal_x0_error_amplification": high_amplification,
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

    summary = {
        "verdict": "PASS",
        "root_cause": "phase314b_r1_sampling_scale_audit_completed",
        "cache_sha256": CACHE_SHA256,
        "checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "device": str(device),
        "diagnostic_indices": indices.tolist(),
        "diagnostic_rows": int(indices.size),
        "standardization": standardization,
        "scheduler_coefficients": {
            "first": coefficients[0],
            "last": coefficients[-1],
            "selected": [
                coefficients[index]
                for index in (0, 10, 25, 50, 75, 90, 99)
            ],
        },
        "scheduler_oracle": scheduler_oracle_rows,
        "weights": weight_results,
        "ema_vs_raw": comparison,
        "gates": gates,
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, summary)
    write_csv(root / args.scheduler_csv, coefficients)
    write_csv(root / args.oracle_csv, scheduler_oracle_rows)
    write_csv(root / args.bins_csv, timestep_rows)
    write_csv(root / args.trace_csv, trace_rows)
    write_csv(root / args.partial_csv, partial_rows)

    lines = [
        "# Phase3.14b-r1 Sampling-Scale Audit",
        "",
        "- Verdict: `PASS` (diagnostic execution completed)",
        (
            "- Root cause: "
            "`phase314b_r1_sampling_scale_audit_completed`"
        ),
        f"- Diagnostic rows: `{indices.size}`",
        f"- Device: `{device}`",
        "",
        "| Diagnostic | Value |",
        "|---|---:|",
    ]
    for key, value in gates.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "This report does not authorize retraining, IDM, candidate "
        "execution, Phase4, or CPS. The separate analyzer assigns the "
        "scientific root cause.",
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
                    "phase314b_r1_sampling_scale_audit_completed"
                ),
                "diagnostic_rows": int(indices.size),
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
