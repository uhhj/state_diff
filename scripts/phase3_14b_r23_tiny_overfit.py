#!/usr/bin/env python3
"""Train-only tiny-set overfit controls for r2.3 diagnosis.

The four runs are diagnostic controls only. They are never checkpointed,
never selected, and never evaluated on validation or formal test rows.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_diffusion import ExponentialMovingAverage
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler
from ccda_phase3.phase314b_r22_contract import GEOMETRY_CONFIGS, load_self_hashed_json
from ccda_phase3.phase314b_r22_geometry import (
    contract_from_json,
    normalizers_from_json,
)
from ccda_phase3.phase314b_r22_loss import geometry_aware_v_loss
from ccda_phase3.phase314b_r23_diagnostics import (
    PHASE,
    diagnostic_tail_training_loss,
    direct_x0_prediction,
    failure_decomposition,
    load_verified_inputs,
    paired_selection_metadata,
    require_repository_state,
    select_balanced_paired_rows,
    source_sha256,
    write_json_once,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_control(
    *,
    name: str,
    use_tail_control: bool,
    random_timesteps: bool,
    condition: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    normalizers,
    physical,
    target_raw: np.ndarray,
    device: torch.device,
    max_steps: int,
    learning_rate: float,
    seed: int,
):
    seed_all(seed)
    model = build_denoiser("mlp_ddpm", condition_dim=261).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
    geometry_config = GEOMETRY_CONFIGS["ordered_edge_temporal"]
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    fixed_timestep = torch.full(
        (clean_z.shape[0],),
        50,
        device=device,
        dtype=torch.long,
    )
    fixed_noise = torch.randn(
        clean_z.shape,
        generator=generator,
        device=device,
        dtype=clean_z.dtype,
    )
    fixed_noise = torch.where(
        active[None],
        fixed_noise,
        torch.zeros_like(fixed_noise),
    )

    history = []
    final_loss = None
    for step in range(1, max_steps + 1):
        if random_timesteps:
            timesteps = torch.randint(
                0,
                100,
                (clean_z.shape[0],),
                generator=generator,
                device=device,
            )
            noise = torch.randn(
                clean_z.shape,
                generator=generator,
                device=device,
                dtype=clean_z.dtype,
            )
            noise = torch.where(
                active[None],
                noise,
                torch.zeros_like(noise),
            )
        else:
            timesteps = fixed_timestep
            noise = fixed_noise

        noisy = scheduler.add_noise(clean_z, noise, timesteps)
        noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
        output = model(noisy, timesteps, condition)
        output = torch.where(active[None], output, torch.zeros_like(output))

        if use_tail_control:
            losses = diagnostic_tail_training_loss(
                model_output=output,
                noisy_sample=noisy,
                clean_z=clean_z,
                clean_raw=clean_raw,
                noise=noise,
                timesteps=timesteps,
                scheduler=scheduler,
                repair_config=REPAIR_CONFIGS["v_prediction_cosine"],
                active_mask=active,
                future_mean=future_mean,
                future_scale=future_scale,
                normalizers=normalizers,
            )
        else:
            losses = geometry_aware_v_loss(
                model_output=output,
                noisy_sample=noisy,
                clean_z=clean_z,
                clean_raw=clean_raw,
                noise=noise,
                timesteps=timesteps,
                scheduler=scheduler,
                repair_config=REPAIR_CONFIGS["v_prediction_cosine"],
                active_mask=active,
                future_mean=future_mean,
                future_scale=future_scale,
                normalizers=normalizers,
                geometry_config=geometry_config,
                epoch=100,
            )
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        )
        optimizer.step()
        final_loss = float(losses["total_loss"].detach().cpu())
        if step == 1 or step % 100 == 0 or step == max_steps:
            history.append(
                {
                    "step": step,
                    "total_loss": final_loss,
                    "gradient_norm_before_clip": gradient_norm,
                }
            )
        if final_loss <= 1e-6:
            break

    evaluations = {}
    model.eval()
    for timestep in (10, 50, 99):
        _, predicted_raw = direct_x0_prediction(
            model=model,
            scheduler=scheduler,
            repair_config=REPAIR_CONFIGS["v_prediction_cosine"],
            condition_z=condition,
            clean_z=clean_z,
            active_mask=active,
            timestep=timestep,
            seed=seed + 2000 + timestep,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        evaluations[str(timestep)] = failure_decomposition(
            predicted_raw[None],
            target_raw,
            physical,
        )

    t50 = evaluations["50"]
    fixed_gate = bool(
        t50["trajectory_ordered_rmse_p95"] <= 0.005
        and t50["segment_family_failure_rate"] <= 0.05
        and t50["chain_family_failure_rate"] <= 0.05
        and t50["gross_stretch_fraction_center"] <= 0.01
    )
    t99 = evaluations["99"]
    random_gate = bool(
        t99["trajectory_ordered_rmse_p95"] <= 0.03
        and t99["segment_family_failure_rate"] <= 0.20
        and t99["chain_family_failure_rate"] <= 0.20
        and t99["gross_stretch_fraction_center"] <= 0.02
    )
    gate_pass = random_gate if random_timesteps else fixed_gate
    del model
    torch.cuda.empty_cache()
    return {
        "name": name,
        "tail_control": use_tail_control,
        "random_timesteps": random_timesteps,
        "steps_completed": int(history[-1]["step"]),
        "final_loss": final_loss,
        "history": history,
        "evaluations": evaluations,
        "fixed_gate": fixed_gate,
        "random_gate": random_gate,
        "gate_pass": gate_pass,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--tiny-rows", type=int, default=16)
    parser.add_argument("--fixed-steps", type=int, default=3000)
    parser.add_argument("--random-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    require_repository_state(root, require_clean=False)
    if not torch.cuda.is_available():
        raise RuntimeError("tiny-overfit diagnosis requires CUDA")
    device = torch.device("cuda")
    arrays, _, x_raw, x_standardizer, train, _, _, _ = (
        load_verified_inputs(root)
    )
    tiny_rows = select_balanced_paired_rows(
        arrays,
        train,
        args.tiny_rows,
    )
    if np.any(np.asarray(arrays["split_name"][tiny_rows]).astype(str) != "train"):
        raise RuntimeError("tiny overfit received non-train rows")

    x_z = x_standardizer.transform(x_raw)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_standardizer = future_standardizer(arrays)
    y_z = y_standardizer.transform(y_raw)
    active = torch.from_numpy(
        np.asarray(arrays["future_active"], dtype=bool)
    ).to(device)
    condition = torch.from_numpy(x_z[tiny_rows]).float().to(device)
    clean_z = torch.from_numpy(y_z[tiny_rows]).float().to(device)
    clean_raw = torch.from_numpy(y_raw[tiny_rows]).float().to(device)
    future_mean = torch.from_numpy(
        np.asarray(arrays["future_mean"], dtype=np.float32)
    ).to(device)
    future_scale = torch.from_numpy(
        np.asarray(arrays["future_scale"], dtype=np.float32)
    ).to(device)

    frozen = load_self_hashed_json(
        root / "reports/phase3_14b_r22_frozen_contract.json"
    )
    physical = contract_from_json(frozen["physical_contract"])
    normalizers = normalizers_from_json(frozen["geometry_normalizers"])

    specifications = (
        ("r22_exact_fixed", False, False, args.fixed_steps, 31901),
        ("tail_control_fixed", True, False, args.fixed_steps, 31902),
        ("r22_exact_random", False, True, args.random_steps, 31903),
        ("tail_control_random", True, True, args.random_steps, 31904),
    )
    runs = {}
    for name, tail, random_t, max_steps, seed in specifications:
        runs[name] = run_control(
            name=name,
            use_tail_control=tail,
            random_timesteps=random_t,
            condition=condition,
            clean_z=clean_z,
            clean_raw=clean_raw,
            active=active,
            future_mean=future_mean,
            future_scale=future_scale,
            normalizers=normalizers,
            physical=physical,
            target_raw=y_raw[tiny_rows],
            device=device,
            max_steps=max_steps,
            learning_rate=args.learning_rate,
            seed=seed,
        )

    report = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": "phase314b_r23_tiny_overfit_controls_completed",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "sampling_contract": {
            "argument_semantics": "actual_rows",
            "selection_unit": "complete_visible_seed_pair",
            "tiny": paired_selection_metadata(arrays, tiny_rows),
        },
        "tiny_row_count": int(len(tiny_rows)),
        "tiny_rows": tiny_rows.tolist(),
        "visible_seeds": arrays["visible_seed"][tiny_rows].tolist(),
        "condition_names": (
            np.asarray(arrays["condition_name"])
            .astype(str)[tiny_rows]
            .tolist()
        ),
        "pair_keys": (
            np.asarray(arrays["pair_key"]).astype(str)[tiny_rows].tolist()
        ),
        "split_names": arrays["split_name"][tiny_rows].astype(str).tolist(),
        "runs": runs,
        "source_sha256": source_sha256(root),
        "formal_test_read": False,
        "validation_read": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(
        root / "reports/phase3_14b_r23_tiny_overfit_summary.json",
        report,
    )
    print(
        json.dumps(
            {name: run["gate_pass"] for name, run in runs.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
