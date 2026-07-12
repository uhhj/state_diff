#!/usr/bin/env python3
"""Train Phase3.14b MLP-DDPM and Temporal-U-Net-DDPM models."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    FAMILIES,
    INPUT_VARIANTS,
    TRAINING_SEEDS,
    atomic_torch_save,
    fixed_balanced_eval_indices,
    future_standardizer,
    input_values_and_standardizer,
    load_locked_cache,
    source_sha256,
    split_mask,
)
from ccda_phase3.phase314b_diffusion import (
    ExponentialMovingAverage,
    active_noise_loss,
    make_ddpm_scheduler,
    sample_future_z,
    scheduler_contract,
)
from ccda_phase3.phase314b_metrics import (
    candidate_final_chamfer,
    nested_best_of_k,
    pairwise_pool_diversity,
)
from ccda_phase3.phase314b_models import (
    build_denoiser,
    parameter_count,
)


SOURCE_PATHS = (
    "ccda_phase3/phase314b_contract.py",
    "ccda_phase3/phase314b_models.py",
    "ccda_phase3/phase314b_diffusion.py",
    "ccda_phase3/phase314b_metrics.py",
    "scripts/phase3_14b_preflight.py",
    "scripts/phase3_14b_train_ddpm.py",
    "scripts/phase3_14b_eval_ddpm.py",
    "scripts/phase3_14b_analyze.py",
    "scripts/phase3_14b_run.sh",
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def val_sample_metrics(
    *,
    model: torch.nn.Module,
    scheduler,
    condition_z: np.ndarray,
    target_raw: np.ndarray,
    future_std,
    future_active: np.ndarray,
    device: torch.device,
    seed: int,
    k: int = 8,
) -> Dict[str, float]:
    condition_tensor = torch.from_numpy(
        np.asarray(condition_z, dtype=np.float32)
    ).to(device)
    active_tensor = torch.from_numpy(
        np.asarray(future_active, dtype=np.bool_)
    ).to(device)
    samples_z = sample_future_z(
        model=model,
        scheduler=scheduler,
        condition_z=condition_tensor,
        active_mask=active_tensor,
        num_samples=int(k),
        seed=int(seed),
        num_inference_steps=100,
        sample_batch_size=128,
    )
    samples_raw = future_std.inverse(
        samples_z.numpy().astype(np.float32)
    )
    distances = candidate_final_chamfer(
        samples_raw,
        target_raw,
    )
    nested = nested_best_of_k(distances, (1, int(k)))
    diversity = pairwise_pool_diversity(samples_raw)
    return {
        "k1_final_chamfer": float(np.mean(nested[1])),
        "best_of_k_final_chamfer": float(
            np.mean(nested[int(k)])
        ),
        "pool_diversity": float(np.mean(diversity)),
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
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/phase3_14b/ddpm",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_training_summary.json",
    )
    parser.add_argument(
        "--history",
        default="reports/phase3_14b_training_history.csv",
    )
    parser.add_argument(
        "--selection",
        default="reports/phase3_14b_selected_model.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_training_report.md",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILIES),
    )
    parser.add_argument(
        "--input-variants",
        nargs="+",
        default=list(INPUT_VARIANTS),
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=list(TRAINING_SEEDS),
    )
    parser.add_argument("--max-epochs", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-val-pair-keys", type=int, default=64)
    parser.add_argument("--allow-cpu-full", action="store_true")
    parser.add_argument("--replace-checkpoints", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PHASE314B_ALLOW_DDPM_TRAINING") != "1":
        raise SystemExit("PHASE314B_ALLOW_DDPM_TRAINING must be 1")
    if os.environ.get("PHASE314B_DDPM_TRAINING_CONFIRMED") != "1":
        raise SystemExit(
            "PHASE314B_DDPM_TRAINING_CONFIRMED must be 1"
        )

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_preflight_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14b preflight is not PASS")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda" and not args.allow_cpu_full:
        raise SystemExit(
            "Full 12-run Phase3.14b training should use a GPU. "
            "Pass --allow-cpu-full only when the runtime cost is accepted."
        )

    arrays, cache_manifest = load_locked_cache(
        root / args.cache,
        root / args.cache_manifest,
    )
    train_indices = np.flatnonzero(
        split_mask(
            arrays,
            "train",
            full_horizon_only=True,
            pre_engagement_only=False,
        )
    ).astype(np.int64)
    val_indices = fixed_balanced_eval_indices(
        arrays,
        split="val",
        max_pair_keys=int(args.max_val_pair_keys),
    )
    if not np.all(arrays["future_valid_mask"][train_indices]):
        raise RuntimeError("DDPM training includes padded future rows")
    if not np.all(arrays["future_valid_mask"][val_indices]):
        raise RuntimeError("DDPM validation includes padded future rows")

    future_std = future_standardizer(arrays)
    target_z_all = future_std.transform(arrays["y_state"])
    future_active = np.asarray(
        arrays["future_active"],
        dtype=np.bool_,
    )
    active_tensor = torch.from_numpy(future_active).to(device)

    source_hashes = source_sha256(root, SOURCE_PATHS)
    checkpoint_root = root / args.checkpoint_root
    existing_checkpoints = list(checkpoint_root.rglob("best.pt"))         if checkpoint_root.exists() else []
    if existing_checkpoints:
        if not args.replace_checkpoints:
            raise SystemExit(
                "Phase3.14b checkpoints already exist; replacement is blocked"
            )
        if os.environ.get("PHASE314B_ALLOW_CHECKPOINT_REPLACE") != "1":
            raise SystemExit(
                "PHASE314B_ALLOW_CHECKPOINT_REPLACE must be 1"
            )
        import shutil
        shutil.rmtree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []

    for family in args.families:
        if family not in FAMILIES:
            raise ValueError(f"unsupported family: {family}")
        for variant in args.input_variants:
            if variant not in INPUT_VARIANTS:
                raise ValueError(
                    f"unsupported input variant: {variant}"
                )
            x_raw, x_std = input_values_and_standardizer(
                arrays,
                variant,
            )
            x_z_all = x_std.transform(x_raw)
            x_train = torch.from_numpy(
                x_z_all[train_indices]
            ).float()
            y_train = torch.from_numpy(
                target_z_all[train_indices]
            ).float()

            for seed in args.training_seeds:
                set_seed(seed)
                scheduler = make_ddpm_scheduler()
                model = build_denoiser(
                    family,
                    condition_dim=x_z_all.shape[1],
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
                generator = torch.Generator()
                generator.manual_seed(int(seed))

                best_metric = float("inf")
                best_epoch = -1
                best_payload = None
                stale_evals = 0
                start_time = time.time()

                for epoch in range(1, int(args.max_epochs) + 1):
                    model.train()
                    permutation = torch.randperm(
                        train_indices.size,
                        generator=generator,
                    )
                    epoch_losses = []
                    for start in range(
                        0,
                        train_indices.size,
                        int(args.batch_size),
                    ):
                        batch_index = permutation[
                            start:start + int(args.batch_size)
                        ]
                        condition = x_train[batch_index].to(device)
                        target = y_train[batch_index].to(device)
                        batch = target.shape[0]
                        timesteps = torch.randint(
                            0,
                            scheduler.config.num_train_timesteps,
                            (batch,),
                            device=device,
                            dtype=torch.long,
                        )
                        noise = torch.randn_like(target)
                        noise = torch.where(
                            active_tensor[None, :, :],
                            noise,
                            torch.zeros_like(noise),
                        )
                        noisy = scheduler.add_noise(
                            target,
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
                        loss = active_noise_loss(
                            prediction,
                            noise,
                            active_tensor,
                        )
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"non-finite loss: "
                                f"{family}/{variant}/{seed}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            1.0,
                        )
                        optimizer.step()
                        ema.update(model)
                        epoch_losses.append(
                            float(loss.detach().cpu())
                        )

                    if epoch % int(args.eval_interval) != 0:
                        continue

                    ema_model = ema.averaged_model(
                        model,
                        device=device,
                    )
                    val_metrics = val_sample_metrics(
                        model=ema_model,
                        scheduler=make_ddpm_scheduler(),
                        condition_z=x_z_all[val_indices],
                        target_raw=arrays["y_state"][val_indices],
                        future_std=future_std,
                        future_active=future_active,
                        device=device,
                        seed=340000 + int(seed),
                        k=8,
                    )
                    record = {
                        "family": family,
                        "input_variant": variant,
                        "training_seed": int(seed),
                        "epoch": int(epoch),
                        "train_noise_mse": float(
                            np.mean(epoch_losses)
                        ),
                        **val_metrics,
                        "elapsed_seconds": float(
                            time.time() - start_time
                        ),
                    }
                    history.append(record)

                    selection_metric = val_metrics[
                        "best_of_k_final_chamfer"
                    ]
                    if selection_metric < best_metric - 1e-10:
                        best_metric = float(selection_metric)
                        best_epoch = int(epoch)
                        best_payload = {
                            "model_state_dict": {
                                key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()
                            },
                            "ema_state_dict": ema.state_dict(),
                            "val_metrics": dict(val_metrics),
                        }
                        stale_evals = 0
                    else:
                        stale_evals += 1
                    if stale_evals >= int(args.patience_evals):
                        break

                if best_payload is None:
                    raise RuntimeError(
                        f"no checkpoint selected for "
                        f"{family}/{variant}/{seed}"
                    )

                run_dir = (
                    checkpoint_root
                    / family
                    / variant
                    / f"seed_{seed}"
                )
                checkpoint_path = run_dir / "best.pt"
                contract = {
                    "phase": "phase3_14b",
                    "model_family": family,
                    "input_variant": variant,
                    "training_seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "selection_split": "val",
                    "selection_subset": (
                        "full_horizon_pre_engagement_paired"
                    ),
                    "selection_metric": (
                        "best_of_8_final_xy_chamfer"
                    ),
                    "selection_value": float(best_metric),
                    "cache_sha256": CACHE_SHA256,
                    "source_sha256": source_hashes,
                    "condition_dim": int(x_z_all.shape[1]),
                    "future_shape": [4, 87],
                    "parameter_count": parameter_count(model),
                    "scheduler": scheduler_contract(
                        make_ddpm_scheduler()
                    ),
                    "ema_decay": float(args.ema_decay),
                    "full_horizon_training_only": True,
                    "future_active_mask_used": True,
                    "clip_sample": False,
                    "thresholding": False,
                    "test_used_for_selection": False,
                    "idm_training": False,
                    "phase4": False,
                    "cps": False,
                }
                atomic_torch_save(
                    checkpoint_path,
                    {
                        **contract,
                        **best_payload,
                    },
                )
                checkpoint_hash = sha256_file(checkpoint_path)
                strict_json_dump(
                    checkpoint_path.with_suffix(
                        ".checkpoint.json"
                    ),
                    {
                        **contract,
                        "checkpoint_sha256": checkpoint_hash,
                    },
                )
                run_summaries.append(
                    {
                        **contract,
                        "checkpoint": str(
                            checkpoint_path.relative_to(root)
                        ),
                        "checkpoint_sha256": checkpoint_hash,
                        "val_metrics": best_payload["val_metrics"],
                    }
                )

    selected = min(
        run_summaries,
        key=lambda item: (
            item["selection_value"],
            item["val_metrics"]["k1_final_chamfer"],
            item["model_family"],
            item["input_variant"],
            item["training_seed"],
        ),
    )
    strict_json_dump(
        root / args.selection,
        {
            "selection_version": "phase3_14b_val_selection_v1",
            **selected,
            "all_run_count": len(run_summaries),
            "test_used_for_selection": False,
        },
    )
    summary = {
        "verdict": "PASS",
        "root_cause": "phase314b_ddpm_training_supported",
        "cache_sha256": CACHE_SHA256,
        "device": str(device),
        "torch_version": torch.__version__,
        "runs": run_summaries,
        "selected": selected,
        "test_evaluation": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, summary)
    write_csv(root / args.history, history)
    lines = [
        "# Phase3.14b DDPM Training",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314b_ddpm_training_supported`",
        f"- Runs: `{len(run_summaries)}`",
        f"- Device: `{device}`",
        (
            "- Selected: "
            f"`{selected['model_family']} / "
            f"{selected['input_variant']} / "
            f"seed={selected['training_seed']}`"
        ),
        (
            "- Validation best-of-8 final Chamfer: "
            f"`{selected['selection_value']:.8f}`"
        ),
        "",
        "| Family | Input | Seed | Epoch | K1 | Best-of-8 | Diversity |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in run_summaries:
        metrics = item["val_metrics"]
        lines.append(
            f"| `{item['model_family']}` | "
            f"`{item['input_variant']}` | "
            f"{item['training_seed']} | {item['best_epoch']} | "
            f"{metrics['k1_final_chamfer']:.8f} | "
            f"{metrics['best_of_k_final_chamfer']:.8f} | "
            f"{metrics['pool_diversity']:.8f} |"
        )
    lines += [
        "",
        "- Test split was not used for selection.",
        "- IDM, candidate execution, Phase4, and CPS were not run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "selected_family": selected["model_family"],
                "selected_input": selected["input_variant"],
                "selected_seed": selected["training_seed"],
                "selected_val_best_of_8": selected[
                    "selection_value"
                ],
                "runs": len(run_summaries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
