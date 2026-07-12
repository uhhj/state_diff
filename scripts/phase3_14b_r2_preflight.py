#!/usr/bin/env python3
"""Phase3.14b-r2 preflight for targeted objective/schedule repair."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r2_contract import (
    CACHE_SHA256,
    DIFFUSERS_VERSION,
    OLD_FAILED_CHECKPOINT_SHA256,
    REPAIR_CONFIGS,
    SUBMODULE_COMMIT,
    default_linear_100_stats,
    load_r2_inputs,
    require_r11_supported,
    schedule_stats_from_betas,
    source_sha256,
    train_indices,
    validation_indices,
)
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    make_repair_scheduler,
    require_diffusers_0111,
    scheduler_contract,
    training_target,
)
from ccda_phase3.phase314b_r2_metrics import (
    fit_validity_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r2_preflight_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r2_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    r11 = require_r11_supported(root)
    arrays, cache_manifest, x_raw, x_std = load_r2_inputs(root)
    train = train_indices(arrays)
    validation = validation_indices(arrays)

    selected_old = strict_json_load(
        root / "reports/phase3_14b_selected_model.json"
    )
    old_checkpoint = root / str(selected_old["checkpoint"])
    if not old_checkpoint.is_file():
        raise FileNotFoundError(old_checkpoint)
    old_hash = sha256_file(old_checkpoint)
    if old_hash != OLD_FAILED_CHECKPOINT_SHA256:
        raise RuntimeError("old failed checkpoint SHA256 changed")
    old_payload = torch.load(old_checkpoint, map_location="cpu")
    for relative, expected in old_payload["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(
                f"old checkpoint-bound source changed: {relative}"
            )

    deterministic_selection = strict_json_load(
        root / "reports/phase3_14a_selected_deterministic_model.json"
    )
    deterministic_checkpoint = (
        root / str(deterministic_selection["checkpoint"])
    )
    if not deterministic_checkpoint.is_file():
        raise FileNotFoundError(deterministic_checkpoint)
    if (
        sha256_file(deterministic_checkpoint)
        != deterministic_selection["checkpoint_sha256"]
    ):
        raise RuntimeError("deterministic checkpoint SHA256 changed")

    main_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=str(root),
        text=True,
    ).strip()
    if main_status:
        raise RuntimeError(
            "tracked main worktree must be clean before r2 preflight"
        )
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "dc18d252d25c677582e412fc8b98b3b28fe343c0",
            "HEAD",
        ],
        cwd=str(root),
    )
    if ancestry.returncode != 0:
        raise RuntimeError("current HEAD is not a descendant of dc18d252")

    submodule_root = root / "external/deformable-ravens"
    submodule_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(submodule_root),
        text=True,
    ).strip()
    submodule_status = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=str(submodule_root),
        text=True,
    ).strip()
    if submodule_head != SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit changed")
    if submodule_status:
        raise RuntimeError("submodule worktree is not clean")

    version = require_diffusers_0111()
    future_std = future_standardizer(arrays)
    target_z = future_std.transform(
        arrays["y_state"][validation[:2]]
    )
    condition_z = x_std.transform(x_raw[validation[:2]])
    active = torch.from_numpy(
        np.asarray(arrays["future_active"], dtype=np.bool_)
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    shape_smoke = {}
    schedule_contracts = {}

    for name, config in REPAIR_CONFIGS.items():
        scheduler = make_repair_scheduler(config)
        schedule_contracts[name] = scheduler_contract(
            scheduler,
            config,
        )
        model = build_denoiser(
            "mlp_ddpm",
            condition_dim=condition_z.shape[1],
        ).to(device)
        clean = torch.from_numpy(target_z).to(device)
        condition = torch.from_numpy(condition_z).to(device)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor(
            [0, config.num_train_timesteps - 1],
            device=device,
            dtype=torch.long,
        )
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
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke loss for {name}")
        shape_smoke[name] = {
            "prediction_type": config.prediction_type,
            "schedule_kind": config.schedule_kind,
            "input_shape": list(noisy.shape),
            "condition_shape": list(condition.shape),
            "target_shape": list(target.shape),
            "output_shape": list(prediction.shape),
            "loss": float(loss.detach().cpu()),
        }

    validity = fit_validity_contract(
        arrays["y_state"][train]
    )
    source_hashes = source_sha256(root)
    linear = default_linear_100_stats()
    if linear["terminal_signal_coefficient"] <= 0.5:
        raise RuntimeError(
            "default linear exclusion assumption changed"
        )

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r2_preflight_supported",
        "cache_sha256": CACHE_SHA256,
        "old_failed_checkpoint_sha256": old_hash,
        "deterministic_checkpoint_sha256": (
            deterministic_selection["checkpoint_sha256"]
        ),
        "r11_root_cause": r11["root_cause"],
        "main_tracked_worktree_clean_before_report": True,
        "diffusers_version": version,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "train_rows": int(train.size),
        "validation_rows": int(validation.size),
        "repair_configs": schedule_contracts,
        "shape_smoke": shape_smoke,
        "validity_contract": validity.to_json(),
        "default_linear_100_exclusion": linear,
        "r2_source_sha256": source_hashes,
        "submodule_commit": submodule_head,
        "submodule_clean": True,
        "test_used": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r2 Preflight",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314b_r2_preflight_supported`",
        f"- Device: `{device}`",
        f"- Train rows: `{train.size}`",
        f"- Validation rows: `{validation.size}`",
        "",
        "| Config | Objective | Schedule | Terminal alpha-bar | Amplification |",
        "|---|---|---|---:|---:|",
    ]
    for name, item in schedule_contracts.items():
        lines.append(
            f"| `{name}` | `{item['prediction_type']}` | "
            f"`{item['schedule_kind']}` | "
            f"{item['terminal_alpha_bar']:.8e} | "
            f"{item['epsilon_x0_error_amplification']:.4f} |"
        )
    lines += [
        "",
        (
            "- Default 100-step linear schedule is excluded because its "
            f"terminal signal coefficient is "
            f"`{linear['terminal_signal_coefficient']:.6f}`."
        ),
        "- No training, test evaluation, IDM, Phase4, or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
