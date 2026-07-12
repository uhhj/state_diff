#!/usr/bin/env python3
"""Audit the r2 validity contract and ordered-cable geometry on validation only."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    future_standardizer,
    input_values_and_standardizer,
)
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.phase314b_r2_contract import (
    REPAIR_CONFIGS,
    load_r2_inputs,
    train_indices,
)
from ccda_phase3.phase314b_r2_diffusion import (
    make_repair_scheduler,
    sample_future_z,
)
from ccda_phase3.phase314b_r2_metrics import (
    fit_validity_contract,
)
from ccda_phase3.phase314b_r21_contract import (
    K_AUDIT,
    fixed_validation_rows,
    observed_last_repeat,
    require_base_reports,
    strict_npz_save,
    train_visible_seed_partition,
)
from ccda_phase3.phase314b_r21_geometry import (
    aggregate_geometry_rows,
    candidate_geometry_rows,
    calibrated_validity,
    evaluate_original_and_calibrated,
    fit_calibrated_geometry_contract,
    per_edge_violation_summary,
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
    arrays: Mapping[str, np.ndarray],
    rows: np.ndarray,
    future_std,
    device: torch.device,
) -> np.ndarray:
    selection = strict_json_load(
        root / "reports/phase3_14a_selected_deterministic_model.json"
    )
    checkpoint_path = root / str(selection["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("deterministic checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("phase") != "phase3_14a":
        raise RuntimeError("not a Phase3.14a deterministic checkpoint")
    variant = str(checkpoint["input_variant"])
    x_raw, x_std = input_values_and_standardizer(arrays, variant)
    x_z = x_std.transform(x_raw[rows])

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
                x_z[start : start + 512]
            ).to(device)
            outputs.append(
                model(batch)
                .reshape(-1, 4, 87)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return future_std.inverse(np.concatenate(outputs, axis=0))


def edge_rows(
    *,
    label: str,
    config: str,
    seed: int,
    summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    under = np.asarray(
        summary["under_rate_by_horizon_edge"],
        dtype=np.float64,
    )
    over = np.asarray(
        summary["over_rate_by_horizon_edge"],
        dtype=np.float64,
    )
    total = np.asarray(
        summary["violation_rate_by_horizon_edge"],
        dtype=np.float64,
    )
    rows = []
    for horizon in range(under.shape[0]):
        for edge in range(under.shape[1]):
            rows.append(
                {
                    "label": label,
                    "repair_config": config,
                    "training_seed": int(seed),
                    "horizon": int(horizon),
                    "edge_index": int(edge),
                    "under_rate": float(under[horizon, edge]),
                    "over_rate": float(over[horizon, edge]),
                    "violation_rate": float(total[horizon, edge]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--pool-root",
        default="data/phase3_14b_r21_eval",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r21_audit_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r21_audit_report.md",
    )
    parser.add_argument(
        "--run-csv",
        default="reports/phase3_14b_r21_run_summary.csv",
    )
    parser.add_argument(
        "--geometry-csv",
        default="reports/phase3_14b_r21_candidate_geometry.csv",
    )
    parser.add_argument(
        "--edge-csv",
        default="reports/phase3_14b_r21_edge_violations.csv",
    )
    args = parser.parse_args()

    if os.environ.get("PHASE314B_R21_ALLOW_DIAGNOSTICS") != "1":
        raise SystemExit(
            "PHASE314B_R21_ALLOW_DIAGNOSTICS must be 1"
        )
    if os.environ.get(
        "PHASE314B_R21_DIAGNOSTICS_CONFIRMED"
    ) != "1":
        raise SystemExit(
            "PHASE314B_R21_DIAGNOSTICS_CONFIRMED must be 1"
        )

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_r21_preflight_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2.1 preflight is not PASS")

    reports = require_base_reports(root)
    arrays, manifest, paper_x, paper_std = load_r2_inputs(root)
    train = train_indices(arrays)
    fit_rows, calibration_rows = train_visible_seed_partition(
        arrays,
        train,
    )
    validation = fixed_validation_rows(arrays)
    future_std = future_standardizer(arrays)
    future_active = np.asarray(
        arrays["future_active"],
        dtype=np.bool_,
    )
    target_val = np.asarray(
        arrays["y_state"][validation],
        dtype=np.float32,
    )

    original_contract = fit_validity_contract(
        arrays["y_state"][train]
    )
    calibrated_contract = fit_calibrated_geometry_contract(
        fit_future=arrays["y_state"][fit_rows],
        calibration_future=arrays["y_state"][calibration_rows],
    )

    reference_states = {
        "train_gt": np.asarray(arrays["y_state"][train], dtype=np.float32),
        "fit_gt": np.asarray(arrays["y_state"][fit_rows], dtype=np.float32),
        "calibration_gt": np.asarray(
            arrays["y_state"][calibration_rows],
            dtype=np.float32,
        ),
        "validation_gt": target_val,
        "last_repeat_validation": observed_last_repeat(
            paper_x,
            validation,
        ),
    }
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    reference_states["deterministic_validation"] = (
        load_deterministic_prediction(
            root=root,
            arrays=arrays,
            rows=validation,
            future_std=future_std,
            device=device,
        )
    )

    reference_metrics = {
        name: evaluate_original_and_calibrated(
            states,
            original_contract=original_contract,
            calibrated_contract=calibrated_contract,
        )
        for name, states in reference_states.items()
    }

    x_z = paper_std.transform(paper_x)
    pool_root = root / args.pool_root
    pool_root.mkdir(parents=True, exist_ok=True)
    run_summaries: List[Dict[str, Any]] = []
    run_csv_rows: List[Dict[str, Any]] = []
    geometry_csv_rows: List[Dict[str, Any]] = []
    edge_csv_rows: List[Dict[str, Any]] = []

    for run in reports["training"]["runs"]:
        config_name = str(run["repair_config"])
        seed = int(run["training_seed"])
        config = REPAIR_CONFIGS[config_name]
        checkpoint_path = root / str(run["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_denoiser(
            "mlp_ddpm",
            condition_dim=x_z.shape[1],
        ).to(device)
        model.load_state_dict(
            checkpoint["ema_state_dict"]["shadow"],
            strict=True,
        )
        model.eval()

        sample_seed = 371000 + seed
        sample_z_tensor = sample_future_z(
            model=model,
            scheduler=make_repair_scheduler(config),
            config=config,
            condition_z=torch.from_numpy(
                x_z[validation]
            ).to(device),
            active_mask=torch.from_numpy(
                future_active
            ).to(device),
            num_samples=K_AUDIT,
            seed=sample_seed,
            num_inference_steps=100,
            row_batch_size=128,
        )
        sample_z = sample_z_tensor.numpy().astype(np.float32)
        sample_raw = future_std.inverse(sample_z)
        pool_path = (
            pool_root
            / f"{config_name}_seed_{seed}_val_k8.npz"
        )
        strict_npz_save(
            pool_path,
            sample_pool_z=sample_z,
            sample_pool_raw=sample_raw,
            row_indices=validation,
            pair_key=np.asarray(
                arrays["pair_key"][validation],
                dtype="<U128",
            ),
            condition_name=np.asarray(
                arrays["condition_name"][validation],
                dtype="<U64",
            ),
            visible_seed=np.asarray(
                arrays["visible_seed"][validation],
                dtype=np.int64,
            ),
            checkpoint_sha256=np.asarray(
                [run["checkpoint_sha256"]],
                dtype="<U64",
            ),
            sample_seed=np.asarray([sample_seed], dtype=np.int64),
        )

        validity = evaluate_original_and_calibrated(
            sample_raw,
            original_contract=original_contract,
            calibrated_contract=calibrated_contract,
        )
        edge_summary = per_edge_violation_summary(
            sample_raw,
            original_contract,
        )
        candidate_rows = candidate_geometry_rows(
            sample_pool=sample_raw,
            target_future=target_val,
            row_indices=validation,
            pair_keys=np.asarray(
                arrays["pair_key"][validation]
            ).astype(str),
            conditions=np.asarray(
                arrays["condition_name"][validation]
            ).astype(str),
            visible_seeds=np.asarray(
                arrays["visible_seed"][validation],
                dtype=np.int64,
            ),
            segment_center=calibrated_contract.segment_center,
        )
        aggregate = aggregate_geometry_rows(candidate_rows)

        summary = {
            "repair_config": config_name,
            "training_seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(root)),
            "checkpoint_sha256": run["checkpoint_sha256"],
            "sample_seed": sample_seed,
            "pool_file": str(pool_path.relative_to(root)),
            "pool_sha256": sha256_file(pool_path),
            "original_validity": validity["original"],
            "calibrated_validity": validity["calibrated"],
            "edge_violations": edge_summary,
            "geometry": aggregate,
            "test_used": False,
        }
        run_summaries.append(summary)

        run_csv_rows.append(
            {
                "repair_config": config_name,
                "training_seed": seed,
                "original_sample_validity": validity[
                    "original"
                ]["sample_validity_rate"],
                "original_coordinate_validity": validity[
                    "original"
                ]["coordinate_validity_rate"],
                "original_segment_validity": validity[
                    "original"
                ]["segment_validity_rate"],
                "original_quaternion_validity": validity[
                    "original"
                ]["quaternion_validity_rate"],
                "calibrated_sample_validity": validity[
                    "calibrated"
                ]["sample_validity_rate"],
                "calibrated_coordinate_validity": validity[
                    "calibrated"
                ]["coordinate_validity_rate"],
                "calibrated_segment_validity": validity[
                    "calibrated"
                ]["segment_score_validity_rate"],
                "calibrated_chain_validity": validity[
                    "calibrated"
                ]["chain_score_validity_rate"],
                "gross_stretch_fraction_mean": aggregate[
                    "gross_stretch_fraction_mean"
                ],
                "gross_compression_fraction_mean": aggregate[
                    "gross_compression_fraction_mean"
                ],
                "ordered_rmse_mean": aggregate[
                    "final_ordered_rmse_mean"
                ],
                "chamfer_mean": aggregate[
                    "final_chamfer_mean"
                ],
                "permutation_gap_mean": aggregate[
                    "permutation_gap_mean"
                ],
                "nearest_inversion_rate_mean": aggregate[
                    "nearest_index_inversion_rate_mean"
                ],
                "max_segment_ratio_to_train_center_p95": aggregate[
                    "max_segment_ratio_to_train_center_p95"
                ],
            }
        )
        for row in candidate_rows:
            geometry_csv_rows.append(
                {
                    "repair_config": config_name,
                    "training_seed": seed,
                    **row,
                }
            )
        edge_csv_rows.extend(
            edge_rows(
                label="candidate",
                config=config_name,
                seed=seed,
                summary=edge_summary,
            )
        )

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r21_validity_geometry_audit_completed",
        "cache_sha256": CACHE_SHA256,
        "device": str(device),
        "train_rows": int(train.size),
        "fit_rows": int(fit_rows.size),
        "calibration_rows": int(calibration_rows.size),
        "validation_rows": int(validation.size),
        "original_contract": original_contract.to_json(),
        "calibrated_contract": calibrated_contract.to_json(),
        "reference_metrics": reference_metrics,
        "runs": run_summaries,
        "formal_test_read": False,
        "training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    write_csv(root / args.run_csv, run_csv_rows)
    write_csv(root / args.geometry_csv, geometry_csv_rows)
    write_csv(root / args.edge_csv, edge_csv_rows)

    lines = [
        "# Phase3.14b-r2.1 Validity and Ordered-Geometry Audit",
        "",
        "- Verdict: `PASS` (diagnostic execution completed)",
        (
            "- Root cause: "
            "`phase314b_r21_validity_geometry_audit_completed`"
        ),
        f"- Device: `{device}`",
        f"- Validation rows: `{validation.size}`",
        "- Formal test read: `False`",
        "",
        "## Reference calibration",
        "",
        "| Reference | Original validity | Calibrated validity |",
        "|---|---:|---:|",
    ]
    for name, item in reference_metrics.items():
        lines.append(
            f"| `{name}` | "
            f"{item['original']['sample_validity_rate']:.6f} | "
            f"{item['calibrated']['sample_validity_rate']:.6f} |"
        )
    lines += [
        "",
        "## R2 checkpoints",
        "",
        "| Config | Seed | Original valid | Calibrated valid | "
        "Ordered RMSE | Chamfer | Permutation gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_csv_rows:
        lines.append(
            f"| `{row['repair_config']}` | "
            f"{row['training_seed']} | "
            f"{row['original_sample_validity']:.6f} | "
            f"{row['calibrated_sample_validity']:.6f} | "
            f"{row['ordered_rmse_mean']:.6f} | "
            f"{row['chamfer_mean']:.6f} | "
            f"{row['permutation_gap_mean']:.6f} |"
        )
    lines += [
        "",
        "The separate analyzer assigns the scientific root cause.",
        "No retraining, formal test read, IDM, execution, Phase4, or CPS "
        "was run.",
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
                    "phase314b_r21_validity_geometry_audit_completed"
                ),
                "reference_metrics": reference_metrics,
                "run_count": len(run_summaries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
