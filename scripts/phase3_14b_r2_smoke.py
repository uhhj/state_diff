#!/usr/bin/env python3
"""Two-epoch Phase3.14b-r2 runtime smoke for all repair configurations."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from ccda_phase3.phase314a_contract import strict_json_dump, strict_json_load
from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.phase314b_r2_contract import (
    REPAIR_CONFIGS,
    SMOKE_SEED,
    load_r2_inputs,
    train_indices,
    validation_indices,
)
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    make_repair_scheduler,
    sample_future_z,
    training_target,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r2_smoke_summary.json",
    )
    args = parser.parse_args()

    if os.environ.get("PHASE314B_R2_ALLOW_SMOKE") != "1":
        raise SystemExit("PHASE314B_R2_ALLOW_SMOKE must be 1")

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_r2_preflight_summary.json"
    )
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("r2 preflight is not PASS")

    random.seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED)
    torch.manual_seed(SMOKE_SEED)
    arrays, _, x_raw, x_std = load_r2_inputs(root)
    train = train_indices(arrays)[:128]
    validation = validation_indices(arrays)[:4]
    future_std = future_standardizer(arrays)
    y_z = future_std.transform(arrays["y_state"])
    x_z = x_std.transform(x_raw)
    active = torch.from_numpy(
        arrays["future_active"].astype(np.bool_)
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    results = {}

    for offset, (name, config) in enumerate(REPAIR_CONFIGS.items()):
        torch.manual_seed(SMOKE_SEED + offset)
        scheduler = make_repair_scheduler(config)
        model = build_denoiser(
            "mlp_ddpm",
            condition_dim=x_z.shape[1],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        x_train = torch.from_numpy(x_z[train]).float()
        y_train = torch.from_numpy(y_z[train]).float()
        losses = []
        for _epoch in range(2):
            model.train()
            condition = x_train.to(device)
            clean = y_train.to(device)
            timesteps = torch.randint(
                0,
                config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
            )
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, timesteps)
            prediction = model(noisy, timesteps, condition)
            target = training_target(
                scheduler=scheduler,
                config=config,
                clean_sample=clean,
                noise=noise,
                timesteps=timesteps,
            )
            loss = active_mse(
                prediction,
                target,
                active.to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite smoke loss: {name}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        samples = sample_future_z(
            model=model,
            scheduler=make_repair_scheduler(config),
            config=config,
            condition_z=torch.from_numpy(
                x_z[validation]
            ).to(device),
            active_mask=active.to(device),
            num_samples=2,
            seed=365000 + offset,
            num_inference_steps=100,
            row_batch_size=4,
        )
        if samples.shape != (2, 4, 4, 87):
            raise RuntimeError(
                f"smoke sample shape mismatch: {samples.shape}"
            )
        if not torch.all(torch.isfinite(samples)):
            raise RuntimeError(f"non-finite smoke samples: {name}")
        results[name] = {
            "losses": losses,
            "sample_shape": list(samples.shape),
            "sample_abs_max": float(
                torch.max(torch.abs(samples)).item()
            ),
        }

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r2_runtime_smoke_supported",
        "device": str(device),
        "configs": results,
        "formal_training": False,
        "test_used": False,
        "idm": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
