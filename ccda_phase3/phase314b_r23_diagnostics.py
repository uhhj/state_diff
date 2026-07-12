"""Validation-only ordered-geometry failure diagnostics for Phase3.14b-r2.3.

This module is deliberately diagnostic-only:
- it reads train/validation rows and frozen r2.2 artifacts;
- it never loads formal test rows;
- it never projects or repairs generated candidates;
- tiny-set models created by the companion script are not candidate models.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ccda_phase3.phase314a_contract import sha256_file, strict_json_load
from ccda_phase3.phase314b_contract import CACHE_SHA256, future_standardizer
from ccda_phase3.phase314b_diffusion import ExponentialMovingAverage
from ccda_phase3.phase314b_models import build_denoiser
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    make_repair_scheduler,
    predict_original_sample,
    training_target,
)
from ccda_phase3.phase314b_r21_geometry import calibrated_validity
from ccda_phase3.phase314b_r22_contract import (
    GEOMETRY_CONFIGS,
    SUBMODULE_COMMIT,
    assert_no_test_access,
    load_self_hashed_json,
    load_train_validation_rows,
)
from ccda_phase3.phase314b_r22_geometry import (
    GeometryNormalizers,
    contract_from_json,
    normalizers_from_json,
    ordered_geometry_components,
    torch_inverse_standardize,
    torch_ordered_xy,
)

PHASE = "phase3_14b_r23"
BASE_FAILURE_COMMIT = "a54cdd20d88f5eb00ebe25e8d4a20c28fe7d6e7f"
EXPECTED_CACHE_SHA256 = "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
EXPECTED_FROZEN_CONTRACT_SHA256 = "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
EXPECTED_R22_ROOT_CAUSE = "phase314b_r22_no_pilot_geometry_repair"
PILOT_CONFIGS = (
    "v_baseline_replay",
    "edge_length",
    "ordered_edge",
    "ordered_edge_temporal",
)
DIRECT_TIMESTEPS = (0, 10, 25, 50, 75, 90, 99)
TRACE_TIMESTEPS = (99, 90, 75, 50, 25, 10, 0)
GRADIENT_TIMESTEPS = (10, 50, 90, 99)

R23_SOURCE_PATHS = (
    "ccda_phase3/phase314b_r23_diagnostics.py",
    "scripts/phase3_14b_r23_preflight.py",
    "scripts/phase3_14b_r23_checkpoint_diagnosis.py",
    "scripts/phase3_14b_r23_tiny_overfit.py",
    "scripts/phase3_14b_r23_finalize.py",
    "scripts/phase3_14b_r23_run.sh",
    "tests/test_phase314b_r23_diagnostics.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("non-finite ndarray")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
        return _jsonable(array)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"refusing to replace {target}")
    data = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(data)
    os.replace(temporary, target)


def write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"refusing to replace {target}")
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = sorted({str(key) for row in rows for key in row})
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([_jsonable(row) for row in rows])
    os.replace(temporary, target)


def source_sha256(root: Path) -> Dict[str, str]:
    return {path: sha256_file(Path(root) / path) for path in R23_SOURCE_PATHS}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def _git_ok(root: Path, *args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def require_repository_state(root: Path, *, require_clean: bool = True) -> Dict[str, Any]:
    root = Path(root).resolve()
    branch = _git(root, "branch", "--show-current")
    if branch != "Experiment1":
        raise RuntimeError(f"wrong branch: {branch}")
    if not _git_ok(root, "merge-base", "--is-ancestor", BASE_FAILURE_COMMIT, "HEAD"):
        raise RuntimeError("r2.2 failure commit is not an ancestor of HEAD")
    if require_clean:
        status = _git(root, "status", "--porcelain", "--untracked-files=all")
        if status:
            raise RuntimeError(f"worktree is not clean:\n{status}")

    submodule_root = root / "external/deformable-ravens"
    submodule_commit = _git(submodule_root, "rev-parse", "HEAD")
    if submodule_commit != SUBMODULE_COMMIT:
        raise RuntimeError(f"submodule commit mismatch: {submodule_commit}")
    if _git(submodule_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("deformable-ravens submodule is dirty")
    if not _git_ok(root, "diff", "--quiet", "--", "external/deformable-ravens"):
        raise RuntimeError("submodule pointer is modified")

    cache_path = root / "data/phase3_14_cache/phase3_14a_training_cache.npz"
    cache_sha = sha256_file(cache_path)
    if cache_sha != EXPECTED_CACHE_SHA256 or CACHE_SHA256 != EXPECTED_CACHE_SHA256:
        raise RuntimeError(f"cache SHA mismatch: {cache_sha}")

    frozen_path = root / "reports/phase3_14b_r22_frozen_contract.json"
    frozen = load_self_hashed_json(frozen_path)
    if frozen.get("artifact_sha256") != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    final_summary = strict_json_load(root / "reports/phase3_14b_r22_summary.json")
    if (
        final_summary.get("verdict"),
        final_summary.get("root_cause"),
        final_summary.get("formal_test_read"),
    ) != ("FAIL", EXPECTED_R22_ROOT_CAUSE, False):
        raise RuntimeError("r2.2 final-summary state mismatch")

    selection = strict_json_load(
        root / "reports/phase3_14b_r22_pilot_selection_summary.json"
    )
    if selection.get("selected_config") is not None:
        raise RuntimeError("r2.2 unexpectedly selected a pilot configuration")
    if selection.get("formal_test_read") is not False:
        raise RuntimeError("formal test was accessed")

    return {
        "branch": branch,
        "main_commit": _git(root, "rev-parse", "HEAD"),
        "base_failure_commit": BASE_FAILURE_COMMIT,
        "submodule_commit": submodule_commit,
        "cache_sha256": cache_sha,
        "frozen_contract_sha256": frozen["artifact_sha256"],
        "r22_verdict": final_summary["verdict"],
        "r22_root_cause": final_summary["root_cause"],
        "formal_test_read": False,
    }


def load_verified_inputs(root: Path):
    root = Path(root).resolve()
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        arrays, manifest, x_raw, x_std, train, fit, calibration, validation = (
            load_train_validation_rows(root)
        )
        assert_no_test_access(arrays, train, validation)
    finally:
        os.chdir(previous_cwd)
    if np.any(np.asarray(arrays["split_name"][validation]).astype(str) != "val"):
        raise RuntimeError("validation rows contain non-validation data")
    return arrays, manifest, x_raw, x_std, train, fit, calibration, validation


def load_pilot_runs(root: Path) -> Dict[str, Mapping[str, Any]]:
    summary = strict_json_load(
        Path(root) / "reports/phase3_14b_r22_pilot_training_summary.json"
    )
    if summary.get("formal_test_read") is not False:
        raise RuntimeError("pilot summary says test was read")
    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != len(PILOT_CONFIGS):
        raise RuntimeError("unexpected pilot run count")
    by_name: Dict[str, Mapping[str, Any]] = {}
    for run in runs:
        name = str(run["geometry_config"])
        if name in by_name:
            raise RuntimeError(f"duplicate run: {name}")
        by_name[name] = run
    if set(by_name) != set(PILOT_CONFIGS):
        raise RuntimeError(f"pilot config mismatch: {sorted(by_name)}")
    return by_name


def verify_checkpoint(
    root: Path,
    run: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> Mapping[str, Any]:
    checkpoint_path = Path(root) / str(run["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != str(run["checkpoint_sha256"]):
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    expected = {
        "phase": "phase3_14b_r22",
        "mode": "pilot",
        "cache_sha256": EXPECTED_CACHE_SHA256,
        "frozen_contract_sha256": EXPECTED_FROZEN_CONTRACT_SHA256,
        "formal_test_read": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"checkpoint {checkpoint_path} field {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    if payload.get("geometry_config") != run.get("geometry_config"):
        raise RuntimeError("checkpoint geometry config mismatch")
    frozen_sources = frozen.get("checkpoint_bound_source_sha256")
    if frozen_sources is not None and payload.get("source_sha256") != frozen_sources:
        raise RuntimeError("checkpoint source hash does not match frozen contract")
    return payload


def load_ema_model(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    model = build_denoiser("mlp_ddpm", condition_dim=261).to(device)
    model.load_state_dict(payload["model_state_dict"])
    averaged = ExponentialMovingAverage.from_state_dict(
        model,
        payload["ema_state_dict"],
    ).averaged_model(model, device=device)
    averaged.eval()
    return averaged


def _future(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (4, 87):
        raise ValueError(f"{name} must be [N,4,87], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _pool(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or array.shape[2:] != (4, 87):
        raise ValueError(f"{name} must be [K,N,4,87], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def ordered_xy_np(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-2:] != (4, 87):
        raise ValueError(f"future tail shape mismatch: {array.shape}")
    return array[..., :48].reshape(*array.shape[:-2], 4, 24, 2)


def edges_np(value: np.ndarray) -> np.ndarray:
    xy = ordered_xy_np(value)
    return xy[..., 1:, :] - xy[..., :-1, :]


def segment_lengths_np(value: np.ndarray) -> np.ndarray:
    return np.linalg.norm(edges_np(value), axis=-1)


def chain_lengths_np(value: np.ndarray) -> np.ndarray:
    return segment_lengths_np(value).sum(axis=-1)


def _quantiles(value: np.ndarray, prefix: str) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"invalid quantile input: {prefix}")
    return {
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_p99": float(np.percentile(array, 99)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_mean": float(np.mean(array)),
    }


def failure_decomposition(
    sample_pool: np.ndarray,
    target_future: np.ndarray,
    physical_contract,
) -> Dict[str, Any]:
    samples = _pool(sample_pool, name="sample_pool")
    target = _future(target_future, name="target_future")
    if samples.shape[1] != target.shape[0]:
        raise ValueError("pool/target row mismatch")

    validity = calibrated_validity(samples, physical_contract)
    sample_valid_mask = np.asarray(validity["sample_valid_mask"], dtype=bool)

    sample_xy = ordered_xy_np(samples)
    target_xy = ordered_xy_np(target)
    sample_edges = sample_xy[..., 1:, :] - sample_xy[..., :-1, :]
    target_edges = target_xy[..., 1:, :] - target_xy[..., :-1, :]
    sample_segments = np.linalg.norm(sample_edges, axis=-1)
    target_segments = np.linalg.norm(target_edges, axis=-1)
    sample_chain = sample_segments.sum(axis=-1)
    target_chain = target_segments.sum(axis=-1)

    segment_center = np.asarray(physical_contract.segment_center, dtype=np.float32)
    segment_scale = np.asarray(physical_contract.segment_scale, dtype=np.float32)
    chain_center = np.asarray(physical_contract.chain_center, dtype=np.float32)
    chain_scale = np.asarray(physical_contract.chain_scale, dtype=np.float32)

    segment_score_element = np.abs(
        (sample_segments - segment_center[None, None]) /
        segment_scale[None, None]
    )
    segment_bad = (
        segment_score_element > float(physical_contract.segment_score_threshold)
    )
    segment_family_score = segment_score_element.max(axis=(-2, -1))
    bad_constraint_count = segment_bad.sum(axis=(-2, -1))

    chain_score_element = np.abs(
        (sample_chain - chain_center[None, None]) /
        chain_scale[None, None]
    )
    chain_family_score = chain_score_element.max(axis=-1)
    chain_bad = (
        chain_family_score > float(physical_contract.chain_score_threshold)
    )

    coordinate_lower = np.asarray(
        physical_contract.coordinate_lower,
        dtype=np.float32,
    )
    coordinate_upper = np.asarray(
        physical_contract.coordinate_upper,
        dtype=np.float32,
    )
    coordinate_bad = np.any(
        (sample_xy < coordinate_lower[None, None, None, None, :])
        | (sample_xy > coordinate_upper[None, None, None, None, :]),
        axis=(2, 3, 4),
    )

    quaternion_norm = np.linalg.norm(samples[..., 83:87], axis=-1)
    quaternion_bad = np.any(
        (quaternion_norm < float(physical_contract.quaternion_norm_lower))
        | (quaternion_norm > float(physical_contract.quaternion_norm_upper)),
        axis=2,
    )

    segment_ratio_center = sample_segments / np.maximum(
        segment_center[None, None],
        1e-8,
    )
    segment_ratio_target = sample_segments / np.maximum(
        target_segments[None],
        1e-8,
    )
    chain_ratio_target = sample_chain / np.maximum(target_chain[None], 1e-8)

    trajectory_rmse = np.sqrt(
        np.mean(
            (sample_xy - target_xy[None]) ** 2,
            axis=(-3, -2, -1),
        )
    )
    final_rmse = np.sqrt(
        np.mean(
            (sample_xy[..., -1, :, :] - target_xy[None, ..., -1, :, :]) ** 2,
            axis=(-2, -1),
        )
    )
    edge_vector_rmse = np.sqrt(
        np.mean(
            (sample_edges - target_edges[None]) ** 2,
            axis=(-3, -2, -1),
        )
    )

    first_failure_counter: Counter[str] = Counter()
    for sample_index, row_index in np.argwhere(np.any(segment_bad, axis=(-2, -1))):
        flat_index = int(np.argmax(segment_bad[sample_index, row_index]))
        horizon, edge = np.unravel_index(flat_index, segment_bad.shape[-2:])
        first_failure_counter[f"h{horizon}_e{edge}"] += 1

    category_count = (
        coordinate_bad.astype(np.int64)
        + np.any(segment_bad, axis=(-2, -1)).astype(np.int64)
        + chain_bad.astype(np.int64)
        + quaternion_bad.astype(np.int64)
    )

    output: Dict[str, Any] = {
        "sample_count": int(samples.shape[0] * samples.shape[1]),
        "query_count": int(samples.shape[1]),
        "sample_validity_rate": float(np.mean(sample_valid_mask)),
        "query_has_valid_candidate_rate": float(
            np.mean(np.any(sample_valid_mask, axis=0))
        ),
        "coordinate_failure_rate": float(np.mean(coordinate_bad)),
        "segment_family_failure_rate": float(
            np.mean(np.any(segment_bad, axis=(-2, -1)))
        ),
        "chain_family_failure_rate": float(np.mean(chain_bad)),
        "quaternion_failure_rate": float(np.mean(quaternion_bad)),
        "gross_stretch_fraction_center": float(
            np.mean(segment_ratio_center >= 4.0)
        ),
        "gross_compression_fraction_center": float(
            np.mean(segment_ratio_center <= 0.25)
        ),
        "first_segment_failure_histogram": dict(
            first_failure_counter.most_common()
        ),
        **_quantiles(segment_score_element, "segment_element_score"),
        **_quantiles(segment_family_score, "segment_family_score"),
        **_quantiles(bad_constraint_count, "bad_segment_constraint_count"),
        **_quantiles(chain_family_score, "chain_family_score"),
        **_quantiles(category_count, "failed_category_count"),
        **_quantiles(segment_ratio_center, "segment_ratio_center"),
        **_quantiles(segment_ratio_target, "segment_ratio_target"),
        **_quantiles(chain_ratio_target, "chain_ratio_target"),
        **_quantiles(trajectory_rmse, "trajectory_ordered_rmse"),
        **_quantiles(final_rmse, "final_ordered_rmse"),
        **_quantiles(edge_vector_rmse, "edge_vector_rmse"),
        **_quantiles(quaternion_norm, "quaternion_norm"),
    }
    return output


def direct_x0_prediction(
    *,
    model: torch.nn.Module,
    scheduler,
    repair_config,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
    timestep: int,
    seed: int,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    device = clean_z.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        clean_z.shape,
        generator=generator,
        device=device,
        dtype=clean_z.dtype,
    )
    active = active_mask.to(device=device, dtype=torch.bool)
    noise = torch.where(active[None], noise, torch.zeros_like(noise))
    timesteps = torch.full(
        (clean_z.shape[0],),
        int(timestep),
        device=device,
        dtype=torch.long,
    )
    noisy = scheduler.add_noise(clean_z, noise, timesteps)
    noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
    with torch.no_grad():
        model_output = model(noisy, timesteps, condition_z)
        model_output = torch.where(
            active[None],
            model_output,
            torch.zeros_like(model_output),
        )
        predicted_z = predict_original_sample(
            scheduler=scheduler,
            config=repair_config,
            sample=noisy,
            model_output=model_output,
            timesteps=timesteps,
        )
        predicted_z = torch.where(
            active[None],
            predicted_z,
            torch.zeros_like(predicted_z),
        )
        predicted_raw = torch_inverse_standardize(
            predicted_z,
            future_mean,
            future_scale,
        )
    return (
        predicted_z.detach().cpu().numpy().astype(np.float32),
        predicted_raw.detach().cpu().numpy().astype(np.float32),
    )


def scheduler_x0_parity(
    *,
    scheduler,
    repair_config,
    device: torch.device,
    timestep: int,
    seed: int,
) -> float:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sample = torch.randn(
        (3, 4, 87),
        generator=generator,
        device=device,
    )
    model_output = torch.randn(
        sample.shape,
        generator=generator,
        device=device,
    )
    timesteps = torch.full(
        (sample.shape[0],),
        int(timestep),
        device=device,
        dtype=torch.long,
    )
    expected = predict_original_sample(
        scheduler=scheduler,
        config=repair_config,
        sample=sample,
        model_output=model_output,
        timesteps=timesteps,
    )
    step_generator = torch.Generator(device=device).manual_seed(int(seed) + 1)
    result = scheduler.step(
        model_output,
        int(timestep),
        sample,
        generator=step_generator,
    )
    actual = result.pred_original_sample
    return float(torch.max(torch.abs(expected - actual)).detach().cpu())


def posterior_mean_trace(
    *,
    model: torch.nn.Module,
    scheduler,
    repair_config,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
    start_timestep: int,
    seed: int,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    capture_timesteps: Sequence[int],
    start_mode: str,
) -> Dict[int, np.ndarray]:
    if start_mode not in {"truth_noised", "pure_noise"}:
        raise ValueError(start_mode)
    device = clean_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        clean_z.shape,
        generator=generator,
        device=device,
        dtype=clean_z.dtype,
    )
    noise = torch.where(active[None], noise, torch.zeros_like(noise))
    if start_mode == "truth_noised":
        batch_t = torch.full(
            (clean_z.shape[0],),
            int(start_timestep),
            device=device,
            dtype=torch.long,
        )
        current = scheduler.add_noise(clean_z, noise, batch_t)
    else:
        current = noise
    current = torch.where(active[None], current, torch.zeros_like(current))

    captures: Dict[int, np.ndarray] = {}
    wanted = {int(item) for item in capture_timesteps}
    with torch.no_grad():
        for timestep in range(int(start_timestep), -1, -1):
            current_t = torch.full(
                (clean_z.shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )
            model_output = model(current, current_t, condition_z)
            model_output = torch.where(
                active[None],
                model_output,
                torch.zeros_like(model_output),
            )
            predicted_z = predict_original_sample(
                scheduler=scheduler,
                config=repair_config,
                sample=current,
                model_output=model_output,
                timesteps=current_t,
            )
            predicted_z = torch.where(
                active[None],
                predicted_z,
                torch.zeros_like(predicted_z),
            )
            if timestep in wanted:
                predicted_raw = torch_inverse_standardize(
                    predicted_z,
                    future_mean,
                    future_scale,
                )
                captures[timestep] = (
                    predicted_raw.detach().cpu().numpy().astype(np.float32)
                )

            alpha_prod_t = scheduler.alphas_cumprod[timestep].to(
                device=device,
                dtype=current.dtype,
            )
            alpha_prod_prev = (
                scheduler.alphas_cumprod[timestep - 1].to(
                    device=device,
                    dtype=current.dtype,
                )
                if timestep > 0
                else scheduler.one.to(device=device, dtype=current.dtype)
            )
            beta_prod_t = 1.0 - alpha_prod_t
            beta_prod_prev = 1.0 - alpha_prod_prev
            pred_x0_coeff = (
                torch.sqrt(alpha_prod_prev)
                * scheduler.betas[timestep].to(
                    device=device,
                    dtype=current.dtype,
                )
                / beta_prod_t
            )
            current_coeff = (
                torch.sqrt(
                    scheduler.alphas[timestep].to(
                        device=device,
                        dtype=current.dtype,
                    )
                )
                * beta_prod_prev
                / beta_prod_t
            )
            current = pred_x0_coeff * predicted_z + current_coeff * current
            current = torch.where(
                active[None],
                current,
                torch.zeros_like(current),
            )
    missing = wanted - set(captures)
    if missing:
        raise RuntimeError(f"trace did not capture timesteps: {sorted(missing)}")
    return captures


def _gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _gradient_norm(grads: Sequence[torch.Tensor | None]) -> float:
    total = None
    for grad in grads:
        if grad is None:
            continue
        term = torch.sum(grad.detach() * grad.detach())
        total = term if total is None else total + term
    return 0.0 if total is None else float(torch.sqrt(total).cpu())


def _gradient_cosine(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> float:
    dot = None
    left_norm = None
    right_norm = None
    for left_grad, right_grad in zip(left, right):
        if left_grad is None or right_grad is None:
            continue
        left_value = left_grad.detach()
        right_value = right_grad.detach()
        current_dot = torch.sum(left_value * right_value)
        current_left = torch.sum(left_value * left_value)
        current_right = torch.sum(right_value * right_value)
        dot = current_dot if dot is None else dot + current_dot
        left_norm = current_left if left_norm is None else left_norm + current_left
        right_norm = (
            current_right if right_norm is None else right_norm + current_right
        )
    if dot is None or left_norm is None or right_norm is None:
        return 0.0
    denominator = torch.sqrt(left_norm * right_norm).clamp_min(1e-30)
    return float((dot / denominator).cpu())


def diagnostic_tail_components(
    predicted_raw: torch.Tensor,
    target_raw: torch.Tensor,
    normalizers: GeometryNormalizers,
    *,
    tail_fraction: float = 0.10,
) -> Dict[str, torch.Tensor]:
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0,1]")
    predicted_xy = torch_ordered_xy(predicted_raw)
    target_xy = torch_ordered_xy(target_raw)
    predicted_edges = predicted_xy[..., 1:, :] - predicted_xy[..., :-1, :]
    target_edges = target_xy[..., 1:, :] - target_xy[..., :-1, :]
    predicted_segments = torch.linalg.norm(predicted_edges, dim=-1)
    target_segments = torch.linalg.norm(target_edges, dim=-1)
    predicted_chain = predicted_segments.sum(dim=-1)
    target_chain = target_segments.sum(dim=-1)

    ordered_abs = torch.abs(
        (predicted_xy - target_xy) / normalizers.edge_vector_scale
    ).flatten(1)
    edge_abs = torch.abs(
        (predicted_edges - target_edges) / normalizers.edge_vector_scale
    ).flatten(1)
    segment_abs = torch.abs(
        (predicted_segments - target_segments)
        / normalizers.segment_length_scale
    ).flatten(1)
    chain_abs = torch.abs(
        (predicted_chain - target_chain) / normalizers.chain_length_scale
    ).flatten(1)

    def cvar(value: torch.Tensor) -> torch.Tensor:
        count = max(1, int(math.ceil(value.shape[1] * tail_fraction)))
        return torch.topk(value, k=count, dim=1, largest=True).values.mean(dim=1)

    return {
        "tail_ordered": cvar(ordered_abs),
        "tail_edge_vector": cvar(edge_abs),
        "tail_segment": cvar(segment_abs),
        "tail_chain": cvar(chain_abs),
    }


def gradient_audit(
    *,
    model: torch.nn.Module,
    scheduler,
    repair_config,
    geometry_config,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    normalizers: GeometryNormalizers,
    timestep: int,
    seed: int,
) -> list[Dict[str, Any]]:
    device = clean_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        clean_z.shape,
        generator=generator,
        device=device,
        dtype=clean_z.dtype,
    )
    noise = torch.where(active[None], noise, torch.zeros_like(noise))
    timesteps = torch.full(
        (clean_z.shape[0],),
        int(timestep),
        device=device,
        dtype=torch.long,
    )
    noisy = scheduler.add_noise(clean_z, noise, timesteps)
    noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
    model_output = model(noisy, timesteps, condition_z)
    model_output = torch.where(
        active[None],
        model_output,
        torch.zeros_like(model_output),
    )
    target_v = training_target(
        scheduler=scheduler,
        config=repair_config,
        clean_sample=clean_z,
        noise=noise,
        timesteps=timesteps,
    )
    v_loss = active_mse(model_output, target_v, active)
    predicted_z = predict_original_sample(
        scheduler=scheduler,
        config=repair_config,
        sample=noisy,
        model_output=model_output,
        timesteps=timesteps,
    )
    predicted_z = torch.where(
        active[None],
        predicted_z,
        torch.zeros_like(predicted_z),
    )
    predicted_raw = torch_inverse_standardize(
        predicted_z,
        future_mean,
        future_scale,
    )
    components = ordered_geometry_components(
        predicted_raw,
        clean_raw,
        normalizers,
    )
    tail = diagnostic_tail_components(
        predicted_raw,
        clean_raw,
        normalizers,
    )

    alpha_bar = scheduler.alphas_cumprod[timesteps].to(
        device=device,
        dtype=clean_z.dtype,
    )
    timestep_weight = 0.5 + 0.5 * torch.sqrt(alpha_bar)
    scalar_losses: MutableMapping[str, torch.Tensor] = {"v_loss": v_loss}
    for name, per_sample in {**components, **tail}.items():
        scalar_losses[name] = (timestep_weight * per_sample).mean()

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    base_grad = _gradients(v_loss, parameters, retain_graph=True)
    base_norm = _gradient_norm(base_grad)
    component_weights = {
        "ordered_xy": float(geometry_config.ordered_weight),
        "edge_vector": float(geometry_config.edge_vector_weight),
        "segment_length": float(geometry_config.segment_length_weight),
        "chain_length": float(geometry_config.chain_length_weight),
        "temporal_edge": float(geometry_config.temporal_edge_weight),
        "tail_ordered": 0.0,
        "tail_edge_vector": 0.0,
        "tail_segment": 0.0,
        "tail_chain": 0.0,
    }
    rows: list[Dict[str, Any]] = []
    for index, (name, loss) in enumerate(scalar_losses.items()):
        grads = base_grad if name == "v_loss" else _gradients(
            loss,
            parameters,
            retain_graph=index < len(scalar_losses) - 1,
        )
        norm = _gradient_norm(grads)
        component_weight = component_weights.get(name, 0.0)
        effective_weight = (
            1.0
            if name == "v_loss"
            else float(geometry_config.geometry_outer_weight) * component_weight
        )
        weighted_norm = norm * effective_weight
        rows.append(
            {
                "timestep": int(timestep),
                "component": name,
                "loss": float(loss.detach().cpu()),
                "gradient_norm": norm,
                "cosine_with_v": (
                    1.0
                    if name == "v_loss"
                    else _gradient_cosine(grads, base_grad)
                ),
                "configured_component_weight": component_weight,
                "effective_weight": effective_weight,
                "weighted_gradient_norm": weighted_norm,
                "weighted_to_v_gradient_ratio": (
                    weighted_norm / max(base_norm, 1e-30)
                ),
                "zero_gradient": bool(norm <= 1e-20),
            }
        )
    return rows


def diagnostic_tail_training_loss(
    *,
    model_output: torch.Tensor,
    noisy_sample: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    repair_config,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    normalizers: GeometryNormalizers,
) -> Dict[str, torch.Tensor]:
    target_v = training_target(
        scheduler=scheduler,
        config=repair_config,
        clean_sample=clean_z,
        noise=noise,
        timesteps=timesteps,
    )
    v_loss = active_mse(model_output, target_v, active_mask)
    predicted_z = predict_original_sample(
        scheduler=scheduler,
        config=repair_config,
        sample=noisy_sample,
        model_output=model_output,
        timesteps=timesteps,
    )
    predicted_z = torch.where(
        active_mask[None].expand_as(predicted_z),
        predicted_z,
        torch.zeros_like(predicted_z),
    )
    predicted_raw = torch_inverse_standardize(
        predicted_z,
        future_mean,
        future_scale,
    )
    tail = diagnostic_tail_components(
        predicted_raw,
        clean_raw,
        normalizers,
    )
    geometry_loss = (
        tail["tail_ordered"].mean()
        + tail["tail_edge_vector"].mean()
        + tail["tail_segment"].mean()
        + tail["tail_chain"].mean()
    )
    total = v_loss + 0.30 * geometry_loss
    return {
        "total_loss": total,
        "v_loss": v_loss,
        "geometry_loss": geometry_loss,
    }


def classify_diagnosis(
    checkpoint_report: Mapping[str, Any],
    tiny_report: Mapping[str, Any],
) -> Dict[str, str]:
    parity = float(checkpoint_report["scheduler_parity_max_abs"])
    if parity > 1e-6:
        return {
            "root_cause": "phase314b_r23_v_prediction_scheduler_parity_failed",
            "next_stage": "repair v-prediction/scheduler equivalence before training",
        }

    gradient_rows = checkpoint_report["gradient_rows"]
    disconnected = [
        row
        for row in gradient_rows
        if row["component"] in {
            "ordered_xy",
            "edge_vector",
            "segment_length",
            "chain_length",
        }
        and row["zero_gradient"]
    ]
    if disconnected:
        return {
            "root_cause": "phase314b_r23_geometry_gradient_disconnect_supported",
            "next_stage": "repair differentiable geometry path before training",
        }

    exact_fixed = tiny_report["runs"]["r22_exact_fixed"]["gate_pass"]
    tail_fixed = tiny_report["runs"]["tail_control_fixed"]["gate_pass"]
    exact_random = tiny_report["runs"]["r22_exact_random"]["gate_pass"]
    tail_random = tiny_report["runs"]["tail_control_random"]["gate_pass"]

    ordered_rows = [
        row
        for row in gradient_rows
        if row["component"] == "ordered_xy"
        and row["config"] in {"ordered_edge", "ordered_edge_temporal"}
    ]
    ordered_ratio = float(
        np.median(
            [row["weighted_to_v_gradient_ratio"] for row in ordered_rows]
        )
    )
    family_tail = bool(
        checkpoint_report["familywise_tail_signature"]["supported"]
    )
    reverse_signature = bool(
        checkpoint_report["reverse_accumulation_signature"]["supported"]
    )

    if (not exact_fixed) and tail_fixed and ordered_ratio < 1e-2:
        return {
            "root_cause": (
                "phase314b_r23_ordered_scale_and_mean_reduction_failure_supported"
            ),
            "next_stage": (
                "Phase3.14b-r2.4 diagnostic tail-aware geometry pilot; "
                "formal remains blocked"
            ),
        }
    if exact_fixed and exact_random and reverse_signature:
        return {
            "root_cause": "phase314b_r23_reverse_process_accumulation_supported",
            "next_stage": (
                "Phase3.14b-r2.4 reverse-process stability repair; "
                "formal remains blocked"
            ),
        }
    if family_tail and tail_random and not exact_random:
        return {
            "root_cause": (
                "phase314b_r23_familywise_tail_objective_mismatch_supported"
            ),
            "next_stage": (
                "Phase3.14b-r2.4 CVaR/family-wise geometry pilot; "
                "formal remains blocked"
            ),
        }
    if not tail_fixed:
        return {
            "root_cause": (
                "phase314b_r23_tiny_overfit_capacity_or_implementation_failure"
            ),
            "next_stage": "debug model/optimizer on train-only tiny set",
        }
    return {
        "root_cause": "phase314b_r23_mixed_ordered_geometry_failure_supported",
        "next_stage": (
            "design a targeted r2.4 pilot from the checkpoint and tiny-overfit "
            "diagnostics; formal remains blocked"
        ),
    }
