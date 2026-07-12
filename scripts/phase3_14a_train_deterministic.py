#!/usr/bin/env python3
"""Train and evaluate Phase3.14a deterministic future baselines."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import (
    BEAD_XY_DIM,
    CACHE_ROWS,
    DEFAULT_TF,
    DEFAULT_TH,
    PAPER_X_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    Standardizer,
    current_state_from_paper_x,
    load_npz_no_pickle,
    previous_state_from_paper_x,
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314a_metrics import (
    evaluate_future_prediction,
    final_chamfer_rows,
)


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


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_standardizer(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
) -> Standardizer:
    return Standardizer(
        mean=arrays[f"{prefix}_mean"],
        scale=arrays[f"{prefix}_scale"],
        active=arrays[f"{prefix}_active"],
        raw_std=arrays[f"{prefix}_raw_std"],
    )


def subset_metrics_without_rows(result: Dict[str, Any]) -> Dict[str, Any]:
    return {"subsets": result["subsets"]}


def metric_rows(
    *,
    model_name: str,
    input_variant: str,
    training_seed: str,
    split: str,
    result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for subset, values in result["subsets"].items():
        rows.append(
            {
                "model": model_name,
                "input_variant": input_variant,
                "training_seed": training_seed,
                "split": split,
                "subset": subset,
                "rows": values["rows"],
                "visible_seeds": values["visible_seeds"],
                "full_state_mse": values["full_state_mse"],
                "bead_xy_mse": values["bead_xy_mse"],
                "robot_proxy_mse": values["robot_proxy_mse"],
                "final_xy_chamfer": values[
                    "final_xy_chamfer"
                ]["mean"],
                "final_xy_chamfer_ci_low": values[
                    "final_xy_chamfer"
                ]["ci_low"],
                "final_xy_chamfer_ci_high": values[
                    "final_xy_chamfer"
                ]["ci_high"],
                "trajectory_xy_chamfer": values[
                    "trajectory_xy_chamfer"
                ]["mean"],
                "trajectory_xy_chamfer_ci_low": values[
                    "trajectory_xy_chamfer"
                ]["ci_low"],
                "trajectory_xy_chamfer_ci_high": values[
                    "trajectory_xy_chamfer"
                ]["ci_high"],
            }
        )
    return rows


def repeat_prediction(paper_x: np.ndarray) -> np.ndarray:
    current = current_state_from_paper_x(paper_x)
    return np.repeat(
        current[:, None, :],
        DEFAULT_TF,
        axis=1,
    ).astype(np.float32)


def constant_velocity_prediction(
    paper_x: np.ndarray,
    state_history_mask: np.ndarray,
) -> np.ndarray:
    current = current_state_from_paper_x(paper_x)
    previous = previous_state_from_paper_x(
        paper_x,
        state_history_mask,
    )
    delta_xy = current[:, :BEAD_XY_DIM] - previous[:, :BEAD_XY_DIM]
    prediction = np.repeat(
        current[:, None, :],
        DEFAULT_TF,
        axis=1,
    )
    for horizon in range(DEFAULT_TF):
        prediction[:, horizon, :BEAD_XY_DIM] = (
            current[:, :BEAD_XY_DIM]
            + float(horizon + 1) * delta_xy
        )
    return prediction.astype(np.float32)


def fit_ridge(
    x_train_z: np.ndarray,
    y_train_z: np.ndarray,
    future_mask_train: np.ndarray,
    *,
    l2: float,
) -> np.ndarray:
    weights = np.zeros(
        (DEFAULT_TF, x_train_z.shape[1] + 1, STATE_DIM),
        dtype=np.float64,
    )
    for horizon in range(DEFAULT_TF):
        selected = future_mask_train[:, horizon]
        x = x_train_z[selected].astype(np.float64)
        y = y_train_z[selected, horizon].astype(np.float64)
        design = np.concatenate(
            [x, np.ones((x.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        regularizer = np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        lhs = design.T @ design + float(l2) * regularizer
        rhs = design.T @ y
        try:
            weights[horizon] = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights[horizon] = np.linalg.pinv(lhs) @ rhs
    return weights


def predict_ridge(
    x_z: np.ndarray,
    weights: np.ndarray,
    future_standardizer: Standardizer,
) -> np.ndarray:
    design = np.concatenate(
        [
            x_z.astype(np.float64),
            np.ones((x_z.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    prediction_z = np.stack(
        [design @ weights[horizon] for horizon in range(DEFAULT_TF)],
        axis=1,
    ).astype(np.float32)
    return future_standardizer.inverse(prediction_z)


class FutureMLP:
    @staticmethod
    def create(input_dim: int):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, DEFAULT_TF * STATE_DIM),
        )


def masked_standardized_loss(
    prediction_z,
    target_z,
    valid_mask,
):
    import torch

    pred = prediction_z.reshape(-1, DEFAULT_TF, STATE_DIM)
    target = target_z.reshape(-1, DEFAULT_TF, STATE_DIM)
    mask = valid_mask[:, :, None].expand_as(pred)
    squared = (pred - target) ** 2
    return squared[mask].mean()


def predict_mlp(
    model,
    x_z: np.ndarray,
    future_standardizer: Standardizer,
    *,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_z.shape[0], int(batch_size)):
            batch = torch.from_numpy(
                x_z[start:start + batch_size]
            ).to(device)
            result = model(batch).reshape(
                -1,
                DEFAULT_TF,
                STATE_DIM,
            )
            outputs.append(result.cpu().numpy().astype(np.float32))
    prediction_z = np.concatenate(outputs, axis=0)
    return future_standardizer.inverse(prediction_z)


def quaternion_norm_error(prediction: np.ndarray) -> float:
    quaternion = np.asarray(
        prediction,
        dtype=np.float32,
    )[..., 83:87]
    norms = np.linalg.norm(quaternion, axis=-1)
    return float(np.max(np.abs(norms - 1.0)))


def evaluate_split(
    prediction: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    return evaluate_future_prediction(
        prediction,
        arrays["y_state"][indices],
        arrays["future_valid_mask"][indices],
        visible_seed=arrays["visible_seed"][indices],
        condition_name=arrays["condition_name"][indices],
        pre_engagement=arrays["pre_engagement"][indices],
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )


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
        "--checkpoint-dir",
        default="checkpoints/phase3_14a/deterministic",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14a_deterministic_summary.json",
    )
    parser.add_argument(
        "--metrics-csv",
        default="reports/phase3_14a_deterministic_metrics.csv",
    )
    parser.add_argument(
        "--training-csv",
        default="reports/phase3_14a_training_history.csv",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14a_deterministic_report.md",
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[31401, 31402, 31403],
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--patience-evals", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--bootstraps", type=int, default=10000)
    args = parser.parse_args()

    if os.environ.get("PHASE314A_ALLOW_DETERMINISTIC_TRAINING") != "1":
        raise SystemExit(
            "PHASE314A_ALLOW_DETERMINISTIC_TRAINING must be 1"
        )
    if os.environ.get(
        "PHASE314A_DETERMINISTIC_TRAINING_CONFIRMED"
    ) != "1":
        raise SystemExit(
            "PHASE314A_DETERMINISTIC_TRAINING_CONFIRMED must be 1"
        )

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14a_preflight_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14a preflight is not PASS")

    cache_path = root / args.cache
    cache_manifest = strict_json_load(root / args.cache_manifest)
    if sha256_file(cache_path) != cache_manifest["cache_sha256"]:
        raise RuntimeError("cache changed after preflight")
    arrays = load_npz_no_pickle(cache_path)
    if arrays["paper_x"].shape[0] != CACHE_ROWS:
        raise RuntimeError("cache row count changed")

    split_values = arrays["split_name"].astype(str)
    indices = {
        split: np.flatnonzero(split_values == split).astype(np.int64)
        for split in ("train", "val", "test")
    }
    if any(value.size == 0 for value in indices.values()):
        raise RuntimeError("empty formal split")

    x_variants = {
        "paper_state": (
            arrays["paper_x"],
            load_standardizer(arrays, "paper_x"),
        ),
        "state_action": (
            arrays["state_action_x"],
            load_standardizer(arrays, "state_action_x"),
        ),
    }
    future_standardizer = load_standardizer(arrays, "future")
    target_z = future_standardizer.transform(arrays["y_state"])

    checkpoint_dir = root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_table: List[Dict[str, Any]] = []
    training_history: List[Dict[str, Any]] = []
    model_summaries: List[Dict[str, Any]] = []
    prediction_store: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}

    # Input-independent baselines.
    for model_name, prediction_all in (
        (
            "last_state_repeat",
            repeat_prediction(arrays["paper_x"]),
        ),
        (
            "xy_constant_velocity_robot_hold",
            constant_velocity_prediction(
                arrays["paper_x"],
                arrays["state_history_valid_mask"],
            ),
        ),
    ):
        split_results = {}
        for split_offset, split in enumerate(("val", "test")):
            selected = indices[split]
            result = evaluate_split(
                prediction_all[selected],
                arrays,
                selected,
                bootstrap_iterations=args.bootstraps,
                bootstrap_seed=314100 + split_offset,
            )
            split_results[split] = subset_metrics_without_rows(result)
            metric_table.extend(
                metric_rows(
                    model_name=model_name,
                    input_variant="paper_state",
                    training_seed="none",
                    split=split,
                    result=result,
                )
            )
        model_summaries.append(
            {
                "model": model_name,
                "input_variant": "paper_state",
                "training_seed": None,
                "selection_source": "not_trained",
                "quaternion_max_norm_error": quaternion_norm_error(
                    prediction_all
                ),
                "metrics": split_results,
            }
        )

    # Ridge for both formal input variants.
    for variant, (x_all, x_standardizer) in x_variants.items():
        x_z = x_standardizer.transform(x_all)
        weights = fit_ridge(
            x_z[indices["train"]],
            target_z[indices["train"]],
            arrays["future_valid_mask"][indices["train"]],
            l2=args.ridge_l2,
        )
        prediction_all = predict_ridge(
            x_z,
            weights,
            future_standardizer,
        )
        checkpoint_path = (
            checkpoint_dir / f"ridge_{variant}.npz"
        )
        temporary = checkpoint_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            weights=weights.astype(np.float32),
            cache_sha256=np.asarray(
                [cache_manifest["cache_sha256"]],
                dtype="<U64",
            ),
            input_variant=np.asarray([variant], dtype="<U32"),
            ridge_l2=np.asarray([args.ridge_l2], dtype=np.float64),
        )
        os.replace(temporary, checkpoint_path)
        strict_json_dump(
            checkpoint_path.with_suffix(".checkpoint.json"),
            {
                "phase": "phase3_14a",
                "model_family": "ridge",
                "input_variant": variant,
                "ridge_l2": float(args.ridge_l2),
                "selection_split": "val",
                "cache_sha256": cache_manifest["cache_sha256"],
                "phase314a_source_sha256": cache_manifest[
                    "phase314a_source_sha256"
                ],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "ddpm": False,
                "idm": False,
                "phase4": False,
                "cps": False,
            },
        )

        split_results = {}
        for split_offset, split in enumerate(("val", "test")):
            selected = indices[split]
            result = evaluate_split(
                prediction_all[selected],
                arrays,
                selected,
                bootstrap_iterations=args.bootstraps,
                bootstrap_seed=314200 + split_offset,
            )
            split_results[split] = subset_metrics_without_rows(result)
            metric_table.extend(
                metric_rows(
                    model_name="ridge",
                    input_variant=variant,
                    training_seed="none",
                    split=split,
                    result=result,
                )
            )
        model_summaries.append(
            {
                "model": "ridge",
                "input_variant": variant,
                "training_seed": None,
                "selection_source": "fixed_l2_validation_reported",
                "checkpoint": str(
                    checkpoint_path.relative_to(root)
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "quaternion_max_norm_error": quaternion_norm_error(
                    prediction_all
                ),
                "metrics": split_results,
            }
        )

    # MLP runs. Test is evaluated only after each run's best epoch is frozen.
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    for variant, (x_all, x_standardizer) in x_variants.items():
        x_z = x_standardizer.transform(x_all)
        for seed in args.training_seeds:
            set_seed(seed)
            model = FutureMLP.create(x_z.shape[1]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
            )
            x_train = torch.from_numpy(
                x_z[indices["train"]]
            )
            y_train = torch.from_numpy(
                target_z[indices["train"]]
            )
            mask_train = torch.from_numpy(
                arrays["future_valid_mask"][indices["train"]]
            )
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            best_state = None
            best_epoch = -1
            best_val = float("inf")
            stale_evals = 0

            for epoch in range(1, int(args.max_epochs) + 1):
                model.train()
                permutation = torch.randperm(
                    x_train.shape[0],
                    generator=generator,
                )
                losses = []
                for start in range(
                    0,
                    x_train.shape[0],
                    int(args.batch_size),
                ):
                    batch_index = permutation[
                        start:start + int(args.batch_size)
                    ]
                    xb = x_train[batch_index].to(device)
                    yb = y_train[batch_index].to(device)
                    mb = mask_train[batch_index].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(xb)
                    loss = masked_standardized_loss(
                        output,
                        yb,
                        mb,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"non-finite MLP loss for {variant}/{seed}"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))

                if epoch % int(args.eval_interval) != 0:
                    continue
                val_selected = indices["val"]
                val_prediction = predict_mlp(
                    model,
                    x_z[val_selected],
                    future_standardizer,
                    device=device,
                )
                val_final_rows = final_chamfer_rows(
                    val_prediction,
                    arrays["y_state"][val_selected],
                    arrays["future_valid_mask"][val_selected],
                )
                val_pre = arrays["pre_engagement"][
                    val_selected
                ].astype(np.bool_)
                if not np.any(val_pre):
                    raise RuntimeError(
                        "validation pre-engagement subset is empty"
                    )
                val_metric = float(np.mean(val_final_rows[val_pre]))
                training_history.append(
                    {
                        "model": "deterministic_mlp",
                        "input_variant": variant,
                        "training_seed": seed,
                        "epoch": epoch,
                        "train_masked_standardized_mse": float(
                            np.mean(losses)
                        ),
                        "val_pre_engagement_final_xy_chamfer": (
                            val_metric
                        ),
                    }
                )
                if val_metric < best_val - 1e-10:
                    best_val = float(val_metric)
                    best_epoch = int(epoch)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_evals = 0
                else:
                    stale_evals += 1
                if stale_evals >= int(args.patience_evals):
                    break

            if best_state is None:
                raise RuntimeError(
                    f"MLP did not produce a checkpoint: {variant}/{seed}"
                )
            model.load_state_dict(best_state)
            checkpoint_path = (
                checkpoint_dir / f"mlp_{variant}_seed{seed}.pt"
            )
            checkpoint = {
                "phase": "phase3_14a",
                "model_family": "deterministic_mlp",
                "input_variant": variant,
                "training_seed": int(seed),
                "best_epoch": int(best_epoch),
                "selection_split": "val",
                "selection_subset": "pre_engagement",
                "selection_metric": "final_xy_chamfer",
                "selection_value": float(best_val),
                "cache_sha256": cache_manifest["cache_sha256"],
                "phase314a_source_sha256": cache_manifest[
                    "phase314a_source_sha256"
                ],
                "input_dim": int(x_z.shape[1]),
                "future_shape": [DEFAULT_TF, STATE_DIM],
                "hidden_dims": [512, 512],
                "state_dict": best_state,
                "ddpm": False,
                "idm": False,
                "phase4": False,
                "cps": False,
            }
            temporary = checkpoint_path.with_suffix(".tmp.pt")
            torch.save(checkpoint, temporary)
            os.replace(temporary, checkpoint_path)
            strict_json_dump(
                checkpoint_path.with_suffix(".checkpoint.json"),
                {
                    "phase": "phase3_14a",
                    "model_family": "deterministic_mlp",
                    "input_variant": variant,
                    "training_seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "selection_split": "val",
                    "selection_subset": "pre_engagement",
                    "selection_metric": "final_xy_chamfer",
                    "selection_value": float(best_val),
                    "cache_sha256": cache_manifest["cache_sha256"],
                    "phase314a_source_sha256": cache_manifest[
                        "phase314a_source_sha256"
                    ],
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "ddpm": False,
                    "idm": False,
                    "phase4": False,
                    "cps": False,
                },
            )

            split_results = {}
            for split_offset, split in enumerate(("val", "test")):
                selected = indices[split]
                prediction = predict_mlp(
                    model,
                    x_z[selected],
                    future_standardizer,
                    device=device,
                )
                result = evaluate_split(
                    prediction,
                    arrays,
                    selected,
                    bootstrap_iterations=args.bootstraps,
                    bootstrap_seed=314400 + seed + split_offset,
                )
                split_results[split] = subset_metrics_without_rows(
                    result
                )
                metric_table.extend(
                    metric_rows(
                        model_name="deterministic_mlp",
                        input_variant=variant,
                        training_seed=str(seed),
                        split=split,
                        result=result,
                    )
                )
            model_summaries.append(
                {
                    "model": "deterministic_mlp",
                    "input_variant": variant,
                    "training_seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "selection_source": (
                        "validation_pre_engagement_final_xy_chamfer"
                    ),
                    "selection_value": float(best_val),
                    "checkpoint": str(
                        checkpoint_path.relative_to(root)
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "quaternion_max_norm_error": quaternion_norm_error(
                        predict_mlp(
                            model,
                            x_z[indices["val"]],
                            future_standardizer,
                            device=device,
                        )
                    ),
                    "metrics": split_results,
                }
            )

    # Freeze the selection using validation only.
    repeat_summary = next(
        item
        for item in model_summaries
        if item["model"] == "last_state_repeat"
    )
    repeat_primary = repeat_summary["metrics"]["val"]["subsets"][
        "pre_engagement"
    ]["final_xy_chamfer"]["mean"]

    learned = [
        item
        for item in model_summaries
        if item["model"] in {"ridge", "deterministic_mlp"}
    ]
    best = min(
        learned,
        key=lambda item: item["metrics"]["val"]["subsets"][
            "pre_engagement"
        ]["final_xy_chamfer"]["mean"],
    )
    best_primary = best["metrics"]["val"]["subsets"][
        "pre_engagement"
    ]["final_xy_chamfer"]["mean"]
    improvement = (
        (float(repeat_primary) - float(best_primary))
        / max(float(repeat_primary), 1e-12)
    )
    repeat_all = repeat_summary["metrics"]["val"]["subsets"][
        "all"
    ]["final_xy_chamfer"]["mean"]
    best_all = best["metrics"]["val"]["subsets"]["all"][
        "final_xy_chamfer"
    ]["mean"]
    overall_non_degrading = best_all <= repeat_all * 1.02

    supported = bool(
        improvement >= 0.05 and overall_non_degrading
    )
    verdict = "PASS" if supported else "FAIL"
    root_cause = (
        "phase314a_state_v2_future_learnability_supported"
        if supported
        else "phase314a_future_model_not_learnable"
    )

    selected_path = root / (
        "reports/phase3_14a_selected_deterministic_model.json"
    )
    strict_json_dump(
        selected_path,
        {
            "selection_version": "phase3_14a_val_selection_v1",
            "selected_model": best["model"],
            "input_variant": best["input_variant"],
            "training_seed": best["training_seed"],
            "checkpoint": best.get("checkpoint"),
            "checkpoint_sha256": best.get("checkpoint_sha256"),
            "selection_split": "val",
            "selection_subset": "pre_engagement",
            "selection_metric": "final_xy_chamfer",
            "selection_value": best_primary,
            "repeat_reference": repeat_primary,
            "relative_improvement": improvement,
            "overall_non_degrading": overall_non_degrading,
            "test_not_used_for_selection": True,
            "cache_sha256": cache_manifest["cache_sha256"],
        },
    )

    summary = {
        "verdict": verdict,
        "root_cause": root_cause,
        "cache_sha256": cache_manifest["cache_sha256"],
        "device": device,
        "training_seeds": [int(value) for value in args.training_seeds],
        "models": model_summaries,
        "selection": strict_json_load(selected_path),
        "learnability_gate": {
            "required_pre_engagement_improvement": 0.05,
            "actual_pre_engagement_improvement": improvement,
            "overall_non_degrading": overall_non_degrading,
            "supported": supported,
        },
        "ddpm_training": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, summary)
    write_csv(root / args.metrics_csv, metric_table)
    write_csv(root / args.training_csv, training_history)

    lines = [
        "# Phase3.14a Deterministic Future Learnability",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        (
            "- Selected model: "
            f"`{best['model']} / {best['input_variant']} / "
            f"seed={best['training_seed']}`"
        ),
        (
            "- Validation pre-engagement improvement over repeat: "
            f"`{improvement:.6f}`"
        ),
        f"- Overall non-degrading: `{overall_non_degrading}`",
        "",
        "| Model | Input | Seed | Val pre-engagement Chamfer |",
        "|---|---|---:|---:|",
    ]
    for item in model_summaries:
        value = item["metrics"]["val"]["subsets"][
            "pre_engagement"
        ]["final_xy_chamfer"]["mean"]
        lines.append(
            f"| `{item['model']}` | `{item['input_variant']}` | "
            f"`{item['training_seed']}` | {value:.8f} |"
        )
    lines += [
        "",
        "- DDPM/IDM/candidate execution: not run.",
        "- Phase4/CPS remain blocked.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "root_cause": root_cause,
                "selected_model": best["model"],
                "input_variant": best["input_variant"],
                "training_seed": best["training_seed"],
                "relative_improvement": improvement,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if not supported:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
