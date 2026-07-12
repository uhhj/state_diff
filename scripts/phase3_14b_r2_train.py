#!/usr/bin/env python3
"""Train the Phase3.14b-r2 targeted MLP objective/schedule matrix."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_diffusion import ExponentialMovingAverage
from ccda_phase3.phase314b_metrics import candidate_final_chamfer
from ccda_phase3.phase314b_models import (
    build_denoiser,
    parameter_count,
)
from ccda_phase3.phase314b_r2_contract import (
    CACHE_SHA256,
    FORMAL_SEEDS,
    REPAIR_CONFIGS,
    R2_SOURCE_PATHS,
    load_r2_inputs,
    source_sha256,
    strict_checkpoint_save,
    train_indices,
    validation_indices,
)
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    make_repair_scheduler,
    partial_denoise,
    sample_future_z,
    scheduler_contract,
    training_target,
)
from ccda_phase3.phase314b_r2_metrics import (
    evaluate_pool,
    fit_validity_contract,
    run_stability_gate,
)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def selection_key(metrics: Dict[str, Any]) -> Tuple[Any, ...]:
    physical = float(
        metrics["physical_validity"]["sample_validity_rate"]
    )
    query_valid = float(
        metrics["physical_validity"][
            "query_has_valid_candidate_rate"
        ]
    )
    z_p99 = float(metrics["z_stats"]["abs_p99"])
    z_max = float(metrics["z_stats"]["abs_max"])
    best4 = float(metrics["k_metrics"]["4"]["mean"])
    k1 = float(metrics["k_metrics"]["1"]["mean"])
    hard_stable = (
        physical >= 0.90
        and query_valid >= 0.95
        and z_p99 <= 10.0
        and z_max <= 50.0
        and k1 <= 0.25
        and best4 <= 0.12
    )
    if hard_stable:
        return (0, best4, k1, -physical, -query_valid)
    return (
        1,
        -physical,
        -query_valid,
        z_p99,
        z_max,
        best4,
        k1,
    )


def evaluate_validation(
    *,
    model: torch.nn.Module,
    config,
    x_z: np.ndarray,
    target_raw: np.ndarray,
    indices: np.ndarray,
    future_std,
    future_active: np.ndarray,
    validity_contract,
    device: torch.device,
    k: int,
    seed: int,
) -> Dict[str, Any]:
    samples_z_tensor = sample_future_z(
        model=model,
        scheduler=make_repair_scheduler(config),
        config=config,
        condition_z=torch.from_numpy(x_z[indices]).to(device),
        active_mask=torch.from_numpy(future_active).to(device),
        num_samples=int(k),
        seed=int(seed),
        num_inference_steps=100,
        row_batch_size=128,
    )
    samples_z = samples_z_tensor.numpy().astype(np.float32)
    samples_raw = future_std.inverse(samples_z)
    return evaluate_pool(
        sample_pool_z=samples_z,
        sample_pool_raw=samples_raw,
        target_raw=target_raw[indices],
        active_mask=future_active,
        validity_contract=validity_contract,
        k_values=(1, 4) if int(k) == 4 else (1, 4, 8),
    )


def partial_chamfer(
    *,
    model: torch.nn.Module,
    config,
    x_z: np.ndarray,
    y_z: np.ndarray,
    target_raw: np.ndarray,
    indices: np.ndarray,
    future_std,
    future_active: np.ndarray,
    device: torch.device,
    timestep: int,
    seed: int,
) -> float:
    prediction_z = partial_denoise(
        model=model,
        scheduler=make_repair_scheduler(config),
        config=config,
        clean_z=torch.from_numpy(y_z[indices]).to(device),
        condition_z=torch.from_numpy(x_z[indices]).to(device),
        active_mask=torch.from_numpy(future_active).to(device),
        start_timestep=int(timestep),
        seed=int(seed),
    )
    prediction_raw = future_std.inverse(
        prediction_z.cpu().numpy().astype(np.float32)
    )
    distances = candidate_final_chamfer(
        prediction_raw[None, ...],
        target_raw[indices],
    )[0]
    return float(np.mean(distances))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/phase3_14b_r2",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r2_training_summary.json",
    )
    parser.add_argument(
        "--history",
        default="reports/phase3_14b_r2_training_history.csv",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r2_training_report.md",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(REPAIR_CONFIGS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(FORMAL_SEEDS),
    )
    parser.add_argument("--max-epochs", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--allow-cpu-full", action="store_true")
    parser.add_argument("--replace-checkpoints", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PHASE314B_R2_ALLOW_TRAINING") != "1":
        raise SystemExit("PHASE314B_R2_ALLOW_TRAINING must be 1")
    if os.environ.get("PHASE314B_R2_TRAINING_CONFIRMED") != "1":
        raise SystemExit(
            "PHASE314B_R2_TRAINING_CONFIRMED must be 1"
        )

    root = Path(args.root).resolve()
    smoke = strict_json_load(
        root / "reports/phase3_14b_r2_smoke_summary.json"
    )
    if smoke.get("verdict") != "PASS":
        raise RuntimeError("r2 runtime smoke is not PASS")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu_full:
        raise SystemExit(
            "Formal 9-run r2 matrix should use CUDA; "
            "pass --allow-cpu-full only after accepting runtime cost"
        )

    arrays, cache_manifest, x_raw, x_std = load_r2_inputs(root)
    train = train_indices(arrays)
    validation = validation_indices(arrays)
    x_z = x_std.transform(x_raw)
    future_std = future_standardizer(arrays)
    y_z = future_std.transform(arrays["y_state"])
    future_active = np.asarray(
        arrays["future_active"],
        dtype=np.bool_,
    )
    active_tensor = torch.from_numpy(future_active).to(device)
    validity_contract = fit_validity_contract(
        arrays["y_state"][train]
    )
    source_hashes = source_sha256(root)

    checkpoint_root = root / args.checkpoint_root
    existing = list(checkpoint_root.rglob("best.pt")) \
        if checkpoint_root.exists() else []
    if existing:
        if not args.replace_checkpoints:
            raise SystemExit(
                "r2 checkpoints already exist; replacement is blocked"
            )
        if os.environ.get(
            "PHASE314B_R2_ALLOW_CHECKPOINT_REPLACE"
        ) != "1":
            raise SystemExit(
                "PHASE314B_R2_ALLOW_CHECKPOINT_REPLACE must be 1"
            )
        shutil.rmtree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    x_train = torch.from_numpy(x_z[train]).float()
    y_train = torch.from_numpy(y_z[train]).float()

    for config_name in args.configs:
        if config_name not in REPAIR_CONFIGS:
            raise ValueError(f"unknown repair config: {config_name}")
        config = REPAIR_CONFIGS[config_name]
        for seed in args.seeds:
            set_seed(seed)
            scheduler = make_repair_scheduler(config)
            model = build_denoiser(
                "mlp_ddpm",
                condition_dim=x_z.shape[1],
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
            )
            ema = ExponentialMovingAverage(
                model,
                decay=float(args.ema_decay),
            )
            permutation_generator = torch.Generator()
            permutation_generator.manual_seed(int(seed))
            device_generator = torch.Generator(device=device)
            device_generator.manual_seed(int(seed) + 100000)

            best_key = None
            best_epoch = -1
            best_payload = None
            stale = 0
            started = time.time()

            for epoch in range(1, int(args.max_epochs) + 1):
                model.train()
                permutation = torch.randperm(
                    train.size,
                    generator=permutation_generator,
                )
                epoch_losses = []
                for start in range(
                    0,
                    train.size,
                    int(args.batch_size),
                ):
                    batch_index = permutation[
                        start:start + int(args.batch_size)
                    ]
                    condition = x_train[batch_index].to(device)
                    clean = y_train[batch_index].to(device)
                    timesteps = torch.randint(
                        0,
                        config.num_train_timesteps,
                        (clean.shape[0],),
                        generator=device_generator,
                        device=device,
                        dtype=torch.long,
                    )
                    noise = torch.randn(
                        clean.shape,
                        generator=device_generator,
                        device=device,
                        dtype=clean.dtype,
                    )
                    noise = torch.where(
                        active_tensor[None, :, :],
                        noise,
                        torch.zeros_like(noise),
                    )
                    noisy = scheduler.add_noise(
                        clean,
                        noise,
                        timesteps,
                    )
                    noisy = torch.where(
                        active_tensor[None, :, :],
                        noisy,
                        torch.zeros_like(noisy),
                    )
                    prediction = model(
                        noisy,
                        timesteps,
                        condition,
                    )
                    prediction = torch.where(
                        active_tensor[None, :, :],
                        prediction,
                        torch.zeros_like(prediction),
                    )
                    target = training_target(
                        scheduler=scheduler,
                        config=config,
                        clean_sample=clean,
                        noise=noise,
                        timesteps=timesteps,
                    )
                    target = torch.where(
                        active_tensor[None, :, :],
                        target,
                        torch.zeros_like(target),
                    )
                    loss = active_mse(
                        prediction,
                        target,
                        active_tensor,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"non-finite loss: {config_name}/{seed}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )
                    optimizer.step()
                    ema.update(model)
                    epoch_losses.append(float(loss.detach().cpu()))

                if epoch % int(args.eval_interval) != 0:
                    continue

                ema_model = ema.averaged_model(
                    model,
                    device=device,
                )
                metrics = evaluate_validation(
                    model=ema_model,
                    config=config,
                    x_z=x_z,
                    target_raw=arrays["y_state"],
                    indices=validation,
                    future_std=future_std,
                    future_active=future_active,
                    validity_contract=validity_contract,
                    device=device,
                    k=4,
                    seed=370000 + int(seed),
                )
                key = selection_key(metrics)
                history.append(
                    {
                        "config": config_name,
                        "prediction_type": config.prediction_type,
                        "schedule_kind": config.schedule_kind,
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "train_objective_mse": float(
                            np.mean(epoch_losses)
                        ),
                        "val_k1": metrics["k_metrics"]["1"]["mean"],
                        "val_best4": metrics["k_metrics"]["4"]["mean"],
                        "val_physical_validity": metrics[
                            "physical_validity"
                        ]["sample_validity_rate"],
                        "val_query_validity": metrics[
                            "physical_validity"
                        ]["query_has_valid_candidate_rate"],
                        "val_z_abs_p99": metrics["z_stats"]["abs_p99"],
                        "val_z_abs_max": metrics["z_stats"]["abs_max"],
                        "elapsed_seconds": float(
                            time.time() - started
                        ),
                    }
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_epoch = int(epoch)
                    best_payload = {
                        "model_state_dict": {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        "ema_state_dict": ema.state_dict(),
                        "selection_metrics_k4": {
                            key_name: value
                            for key_name, value in metrics.items()
                            if key_name not in {
                                "distances",
                                "nested",
                                "sample_valid_mask",
                            }
                        },
                    }
                    stale = 0
                else:
                    stale += 1
                if stale >= int(args.patience_evals):
                    break

            if best_payload is None:
                raise RuntimeError(
                    f"no selected checkpoint: {config_name}/{seed}"
                )

            model.load_state_dict(
                best_payload["model_state_dict"],
                strict=True,
            )
            ema_best = ExponentialMovingAverage.from_state_dict(
                model,
                best_payload["ema_state_dict"],
            ).averaged_model(model, device=device)

            final_metrics = evaluate_validation(
                model=ema_best,
                config=config,
                x_z=x_z,
                target_raw=arrays["y_state"],
                indices=validation,
                future_std=future_std,
                future_active=future_active,
                validity_contract=validity_contract,
                device=device,
                k=8,
                seed=371000 + int(seed),
            )
            partial_t10 = partial_chamfer(
                model=ema_best,
                config=config,
                x_z=x_z,
                y_z=y_z,
                target_raw=arrays["y_state"],
                indices=validation[:16],
                future_std=future_std,
                future_active=future_active,
                device=device,
                timestep=10,
                seed=372000 + int(seed),
            )
            partial_t99 = partial_chamfer(
                model=ema_best,
                config=config,
                x_z=x_z,
                y_z=y_z,
                target_raw=arrays["y_state"],
                indices=validation[:16],
                future_std=future_std,
                future_active=future_active,
                device=device,
                timestep=99,
                seed=373000 + int(seed),
            )
            stability = run_stability_gate(
                pool_metrics=final_metrics,
                partial_t10_chamfer=partial_t10,
                partial_t99_chamfer=partial_t99,
            )

            run_dir = (
                checkpoint_root
                / config_name
                / f"seed_{seed}"
            )
            checkpoint_path = run_dir / "best.pt"
            contract = {
                "phase": "phase3_14b_r2",
                "model_family": "mlp_ddpm",
                "input_variant": "paper_state",
                "repair_config": config_name,
                "prediction_type": config.prediction_type,
                "schedule_kind": config.schedule_kind,
                "training_seed": int(seed),
                "best_epoch": int(best_epoch),
                "cache_sha256": CACHE_SHA256,
                "source_sha256": source_hashes,
                "scheduler": scheduler_contract(
                    make_repair_scheduler(config),
                    config,
                ),
                "selection_split": "val",
                "selection_subset": (
                    "full_horizon_pre_engagement_paired"
                ),
                "selection_k": 4,
                "final_validation_k": 8,
                "test_used_for_selection": False,
                "future_active_mask_used": True,
                "full_horizon_training_only": True,
                "ema_decay": float(args.ema_decay),
                "parameter_count": parameter_count(model),
                "stability_gate": stability,
                "idm": False,
                "phase4": False,
                "cps": False,
            }
            strict_checkpoint_save(
                checkpoint_path,
                {
                    **contract,
                    **best_payload,
                },
            )
            checkpoint_hash = sha256_file(checkpoint_path)
            strict_json_dump(
                checkpoint_path.with_suffix(".checkpoint.json"),
                {
                    **contract,
                    "checkpoint_sha256": checkpoint_hash,
                },
            )
            runs.append(
                {
                    **contract,
                    "checkpoint": str(
                        checkpoint_path.relative_to(root)
                    ),
                    "checkpoint_sha256": checkpoint_hash,
                    "validation_metrics": {
                        key: value
                        for key, value in final_metrics.items()
                        if key not in {
                            "distances",
                            "nested",
                            "sample_valid_mask",
                        }
                    },
                    "partial_t10_chamfer": partial_t10,
                    "partial_t99_chamfer": partial_t99,
                }
            )

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r2_training_completed",
        "cache_sha256": CACHE_SHA256,
        "device": str(device),
        "torch_version": torch.__version__,
        "configs": list(args.configs),
        "seeds": [int(value) for value in args.seeds],
        "run_count": len(runs),
        "runs": runs,
        "validity_contract": validity_contract.to_json(),
        "test_used": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    write_csv(root / args.history, history)

    lines = [
        "# Phase3.14b-r2 Targeted Training",
        "",
        "- Verdict: `PASS` (training completed)",
        "- Root cause: `phase314b_r2_training_completed`",
        f"- Device: `{device}`",
        f"- Runs: `{len(runs)}`",
        "",
        "| Config | Seed | Epoch | Stable | K1 | Best-8 | Physical | t99 |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        values = run["stability_gate"]["values"]
        lines.append(
            f"| `{run['repair_config']}` | {run['training_seed']} | "
            f"{run['best_epoch']} | "
            f"`{run['stability_gate']['stable']}` | "
            f"{values['k1']:.8f} | {values['best8']:.8f} | "
            f"{values['physical_validity']:.6f} | "
            f"{values['partial_t99_chamfer']:.8f} |"
        )
    lines += [
        "",
        "- Test was not used.",
        "- IDM, action execution, Phase4, and CPS were not run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "run_count": len(runs),
                "stable_runs": int(
                    sum(
                        bool(run["stability_gate"]["stable"])
                        for run in runs
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
