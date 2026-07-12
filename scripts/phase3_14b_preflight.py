#!/usr/bin/env python3
"""Phase3.14b preflight: immutable cache, baseline, scheduler, and shapes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ccda_phase3.phase314a_contract import strict_json_dump, strict_json_load
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    FAMILIES,
    INPUT_VARIANTS,
    fixed_balanced_eval_indices,
    full_horizon_mask,
    input_values_and_standardizer,
    load_locked_cache,
    split_mask,
    validate_training_population,
)
from ccda_phase3.phase314b_diffusion import (
    make_ddpm_scheduler,
    scheduler_contract,
)
from ccda_phase3.phase314b_models import (
    build_denoiser,
    parameter_count,
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
        "--summary",
        default="reports/phase3_14b_preflight_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # Refresh strict formal provenance before Phase3.14b creates reports.
    subprocess.run(
        [
            sys.executable,
            "scripts/phase3_14_provenance_gate.py",
            "--root",
            str(root),
        ],
        cwd=str(root),
        check=True,
    )
    provenance = strict_json_load(
        root / "reports/phase3_14_provenance_gate_summary.json"
    )
    if provenance.get("verdict") != "PASS":
        raise RuntimeError("formal Phase3.14 provenance is not PASS")

    cache_path = root / args.cache
    manifest_path = root / args.cache_manifest

    phase314a = strict_json_load(
        root / "reports/phase3_14a_summary.json"
    )
    if phase314a.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14a is not PASS")
    if (
        phase314a.get("root_cause")
        != "phase314a_state_v2_future_learnability_supported"
    ):
        raise RuntimeError("unexpected Phase3.14a root cause")

    selected = strict_json_load(
        root / "reports/phase3_14a_selected_deterministic_model.json"
    )
    if selected.get("selected_model") != "deterministic_mlp":
        raise RuntimeError("selected Phase3.14a model is not an MLP")
    selected_checkpoint = root / str(selected["checkpoint"])
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(selected_checkpoint)
    from ccda_phase3.phase314a_contract import sha256_file
    if (
        sha256_file(selected_checkpoint)
        != selected["checkpoint_sha256"]
    ):
        raise RuntimeError("selected deterministic checkpoint hash mismatch")
    if selected["cache_sha256"] != CACHE_SHA256:
        raise RuntimeError("selected deterministic model used another cache")

    arrays, cache_manifest = load_locked_cache(
        cache_path,
        manifest_path,
    )
    population = validate_training_population(arrays)
    val_indices = fixed_balanced_eval_indices(
        arrays,
        split="val",
        max_pair_keys=64,
    )

    scheduler = make_ddpm_scheduler()
    scheduler_info = scheduler_contract(scheduler)
    shape_smoke: Dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    future_active = torch.from_numpy(
        np.asarray(arrays["future_active"], dtype=np.bool_)
    ).to(device)

    for variant in INPUT_VARIANTS:
        values, standardizer = input_values_and_standardizer(
            arrays,
            variant,
        )
        condition = torch.from_numpy(
            standardizer.transform(values[val_indices[:2]])
        ).to(device)
        for family in FAMILIES:
            model = build_denoiser(
                family,
                condition_dim=condition.shape[1],
            ).to(device)
            sample = torch.randn(
                (2, 4, 87),
                device=device,
                requires_grad=True,
            )
            timestep = torch.tensor(
                [1, 50],
                device=device,
                dtype=torch.long,
            )
            output = model(sample, timestep, condition)
            if output.shape != sample.shape:
                raise RuntimeError(
                    f"{family}/{variant} shape mismatch"
                )
            output = torch.where(
                future_active[None, :, :],
                output,
                torch.zeros_like(output),
            )
            loss = output.square().mean()
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{family}/{variant} backward is non-finite"
                )
            shape_smoke[f"{family}:{variant}"] = {
                "input_shape": list(sample.shape),
                "condition_shape": list(condition.shape),
                "output_shape": list(output.shape),
                "parameters": parameter_count(model),
                "loss": float(loss.detach().cpu()),
            }

    submodule_status = subprocess.check_output(
        [
            "git",
            "-C",
            "external/deformable-ravens",
            "status",
            "--short",
        ],
        cwd=str(root),
        text=True,
    ).strip()
    if submodule_status:
        raise RuntimeError("submodule is not clean")

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_preflight_supported",
        "cache_sha256": CACHE_SHA256,
        "cache_rows": int(arrays["paper_x"].shape[0]),
        "population": population,
        "fixed_val_rows": int(val_indices.size),
        "scheduler": scheduler_info,
        "shape_smoke": shape_smoke,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "selected_deterministic": selected,
        "submodule_clean": True,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    lines = [
        "# Phase3.14b Preflight",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314b_preflight_supported`",
        f"- Cache SHA256: `{CACHE_SHA256}`",
        f"- CUDA available: `{payload['cuda_available']}`",
        f"- Fixed validation rows: `{val_indices.size}`",
        "",
        "| Model | Parameters | Output |",
        "|---|---:|---|",
    ]
    for name, item in shape_smoke.items():
        lines.append(
            f"| `{name}` | {item['parameters']} | "
            f"`{item['output_shape']}` |"
        )
    lines += [
        "",
        "- No DDPM training was run by preflight.",
        "- IDM, candidate execution, Phase4, and CPS remain blocked.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
