#!/usr/bin/env python3
"""Fit one r2.5.4 shared prior and return an in-memory comparison payload.

This worker writes one JSON object to stdout.  It never writes a model,
checkpoint, candidate pool, NPZ, or dataset artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from ccda_phase3.phase314b_contract import future_standardizer
from ccda_phase3.phase314b_r23_diagnostics import (
    load_verified_inputs,
    select_balanced_paired_rows,
)
from ccda_phase3.phase314b_r251_gradient_calibration import (
    assert_canonical_paired_row_contract,
    fit_shared_prior_snapshot,
    instantiate_snapshot_model,
)
from ccda_phase3.phase314b_r254_resume1_prior_determinism import (
    COMMON_PAIRED_PRIOR_SEED,
    EXPECTED_PAIRED_ROWS,
    WORKER_SCHEMA,
    encode_prediction,
    encode_tensor_state,
    tensor_state_summary,
)
from ccda_phase3.phase314b_r25_ordered_geometry import GeometryTrainSpec
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import make_repair_scheduler


def subset(rows: np.ndarray, *, condition: torch.Tensor, clean: torch.Tensor) -> Dict[str, torch.Tensor]:
    index = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(condition.device)
    return {
        "condition_z": condition.index_select(0, index),
        "clean_z": clean.index_select(0, index),
    }


def environment_report() -> Dict[str, Any]:
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_index": int(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": [int(capability[0]), int(capability[1])],
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("shared-prior repeat worker requires CUDA")
    device = torch.device("cuda")

    arrays, _, x_raw, x_standardizer, train, _, _, _ = load_verified_inputs(root)
    split = np.asarray(arrays["split_name"]).astype(str)
    if np.any(split[train] != "train"):
        raise RuntimeError("train indices contain non-train rows")
    paired_rows = select_balanced_paired_rows(arrays, train, 16)
    paired_contract = assert_canonical_paired_row_contract(arrays, paired_rows)
    if paired_contract.get("observed_rows") != list(EXPECTED_PAIRED_ROWS):
        raise RuntimeError("paired-row identity changed")

    x_z = x_standardizer.transform(np.asarray(x_raw, dtype=np.float32)).astype(np.float32)
    y_raw = np.asarray(arrays["y_state"], dtype=np.float32)
    y_z = future_standardizer(arrays).transform(y_raw).astype(np.float32)
    condition_all = torch.from_numpy(x_z).float().to(device)
    clean_all = torch.from_numpy(y_z).float().to(device)
    active = torch.from_numpy(np.asarray(arrays["future_active"], dtype=bool)).to(device)
    paired = subset(paired_rows, condition=condition_all, clean=clean_all)

    scheduler = make_repair_scheduler(REPAIR_CONFIGS["v_prediction_cosine"])
    train_spec = GeometryTrainSpec(
        prior_steps=int(args.prior_steps),
        residual_steps=8000,
        batch_size=int(args.batch_size),
        prior_learning_rate=float(args.prior_learning_rate),
        residual_learning_rate=1.0e-3,
    )
    before = environment_report()
    snapshot = fit_shared_prior_snapshot(
        scheduler=scheduler,
        condition_z=paired["condition_z"],
        clean_z=paired["clean_z"],
        active_mask=active,
        train_spec=train_spec,
        seed=COMMON_PAIRED_PRIOR_SEED,
    )
    torch.cuda.synchronize()
    state = snapshot.get("_prior_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("shared prior snapshot did not retain state")
    model = instantiate_snapshot_model(
        scheduler=scheduler,
        condition_dim=int(paired["condition_z"].shape[1]),
        device=device,
        snapshot=snapshot,
        seed=COMMON_PAIRED_PRIOR_SEED,
    )
    model.eval()
    with torch.no_grad():
        prediction = model.predict_base_x0(paired["condition_z"])
    torch.cuda.synchronize()
    prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
    prediction_sha = hashlib.sha256(
        np.ascontiguousarray(prediction_np).tobytes(order="C")
    ).hexdigest()

    report = {
        "schema": WORKER_SCHEMA,
        "run_id": str(args.run_id),
        "root": str(root),
        "seed": COMMON_PAIRED_PRIOR_SEED,
        "train_spec": {
            "prior_steps": int(args.prior_steps),
            "batch_size": int(args.batch_size),
            "prior_learning_rate": float(args.prior_learning_rate),
        },
        "paired_row_contract": paired_contract,
        "environment_before_fit": before,
        "environment_after_fit": environment_report(),
        "prior_state_sha256": str(snapshot["prior_state_sha256"]),
        "prior_prediction_sha256": prediction_sha,
        "prior_z_mse": float(snapshot["prior_z_mse"]),
        "prior_history": snapshot["prior_history"],
        "state_summary": tensor_state_summary(state),
        "_state_payload": encode_tensor_state(state),
        "_prediction_payload": encode_prediction(prediction_np),
    }
    sys.stdout.write(json.dumps(report, separators=(",", ":"), sort_keys=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
