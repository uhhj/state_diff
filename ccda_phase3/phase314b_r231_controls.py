"""Train-only tiny-control correction for Phase3.14b-r2.3.1.

This module corrects the r2.3 tiny-overfit evaluation contract:

* fixed-tuple controls are evaluated on the exact timestep/noise used in training;
* fresh-noise evaluation is reported separately and is never used as the
  fixed-tuple memorization gate;
* overfit gates compare predictions directly with their training targets,
  rather than requiring every target to satisfy a population physical contract;
* single-row, unique-condition and paired-condition controls are separated.

No function in this module reads validation or formal-test rows.
"""
from __future__ import annotations

import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ccda_phase3.phase314a_contract import sha256_file
from ccda_phase3.phase314b_models import MLPFutureDenoiser
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    predict_original_sample,
    training_target,
)
from ccda_phase3.phase314b_r22_geometry import (
    torch_inverse_standardize,
)
from ccda_phase3.phase314b_r22_loss import geometry_aware_v_loss
from ccda_phase3.phase314b_r22_contract import GEOMETRY_CONFIGS
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS

PHASE = "phase3_14b_r231"
BASE_R23_COMMIT = "19e6d4eececd1abc51264449bd32db70df899b1d"
EXPECTED_R23_ROOT_CAUSE = (
    "phase314b_r23_tiny_overfit_capacity_or_implementation_failure"
)
EXPECTED_CONDITIONS = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)

R231_SOURCE_PATHS = (
    "ccda_phase3/phase314b_r231_controls.py",
    "scripts/phase3_14b_r231_preflight.py",
    "scripts/phase3_14b_r231_run_controls.py",
    "scripts/phase3_14b_r231_finalize.py",
    "scripts/phase3_14b_r231_run.sh",
    "tests/test_phase314b_r231_controls.py",
)


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    learning_rate: float
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None

    def validate(self) -> None:
        if self.name not in {"adam", "adamw"}:
            raise ValueError(f"unsupported optimizer: {self.name}")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.clip_grad_norm is not None:
            if (
                not math.isfinite(self.clip_grad_norm)
                or self.clip_grad_norm <= 0
            ):
                raise ValueError("clip_grad_norm must be finite and positive")


@dataclass(frozen=True)
class DenoiserSpec:
    hidden_dim: int = 512
    time_dim: int = 128

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.time_dim <= 0:
            raise ValueError("denoiser dimensions must be positive")


@dataclass(frozen=True)
class ReconstructionGate:
    z_mse_max: float
    ordered_rmse_p95_max: float
    segment_relative_error_p95_max: float
    chain_relative_error_p95_max: float

    def validate(self) -> None:
        for value in asdict(self).values():
            if not math.isfinite(value) or value < 0:
                raise ValueError("gate thresholds must be finite/nonnegative")


EXACT_REPLAY_GATE = ReconstructionGate(
    z_mse_max=1.0e-4,
    ordered_rmse_p95_max=2.0e-3,
    segment_relative_error_p95_max=0.10,
    chain_relative_error_p95_max=0.05,
)
DIRECT_REGRESSION_GATE = ReconstructionGate(
    z_mse_max=1.0e-4,
    ordered_rmse_p95_max=2.0e-3,
    segment_relative_error_p95_max=0.10,
    chain_relative_error_p95_max=0.05,
)
RANDOM_SINGLE_BRANCH_GATE = ReconstructionGate(
    z_mse_max=2.5e-3,
    ordered_rmse_p95_max=1.0e-2,
    segment_relative_error_p95_max=0.30,
    chain_relative_error_p95_max=0.15,
)

for _gate in (
    EXACT_REPLAY_GATE,
    DIRECT_REGRESSION_GATE,
    RANDOM_SINGLE_BRANCH_GATE,
):
    _gate.validate()


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {
        path: sha256_file(base / path)
        for path in R231_SOURCE_PATHS
    }



def assert_only_allowed_worktree_paths(
    root: Path,
    allowed_paths: Sequence[str],
) -> None:
    """Reject any tracked/untracked change outside an explicit allow-list."""
    allowed = {str(Path(path).as_posix()) for path in allowed_paths}
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=Path(root),
        text=True,
    )
    observed = set()
    for line in output.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        observed.add(str(Path(path).as_posix()))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree changes: " + ", ".join(unexpected)
        )


def select_unique_condition_rows(
    arrays: Mapping[str, np.ndarray],
    candidate_rows: np.ndarray,
    *,
    condition: str,
    row_count: int,
) -> np.ndarray:
    """Select one deterministic row per distinct visible seed.

    This is a capacity/optimizer control, not a paired CCDA evaluation.
    Selecting one branch per seed prevents an intentionally ambiguous
    free/hidden target pair from invalidating a deterministic reconstruction
    gate.
    """
    requested = int(row_count)
    if requested <= 0:
        raise ValueError("row_count must be positive")
    if condition not in EXPECTED_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")

    rows = np.asarray(candidate_rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("candidate_rows must be one-dimensional")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("candidate_rows contains duplicates")

    conditions = np.asarray(arrays["condition_name"]).astype(str)
    seeds = np.asarray(arrays["visible_seed"]).astype(np.int64)
    pair_keys = np.asarray(arrays["pair_key"]).astype(str)

    eligible = sorted(
        (
            int(seeds[row]),
            str(pair_keys[row]),
            int(row),
        )
        for row in rows.tolist()
        if str(conditions[row]) == condition
    )

    selected = []
    used_seeds = set()
    for seed, _, row in eligible:
        if seed in used_seeds:
            continue
        used_seeds.add(seed)
        selected.append(row)
        if len(selected) == requested:
            break

    if len(selected) != requested:
        raise RuntimeError(
            "not enough unique-condition rows: "
            f"condition={condition}, requested={requested}, "
            f"available={len(selected)}"
        )
    result = np.asarray(selected, dtype=np.int64)
    if len(set(seeds[result].tolist())) != requested:
        raise RuntimeError("visible-seed uniqueness invariant failed")
    if np.any(conditions[result] != condition):
        raise RuntimeError("condition-selection invariant failed")
    return result


def pair_ambiguity_metrics(
    arrays: Mapping[str, np.ndarray],
    paired_rows: np.ndarray,
    condition_z: np.ndarray,
    target_raw: np.ndarray,
) -> Dict[str, Any]:
    """Measure train-only input similarity and target divergence per pair."""
    rows = np.asarray(paired_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) % 2:
        raise ValueError("paired_rows must be a one-dimensional even array")

    conditions = np.asarray(arrays["condition_name"]).astype(str)
    seeds = np.asarray(arrays["visible_seed"]).astype(np.int64)
    pair_keys = np.asarray(arrays["pair_key"]).astype(str)

    input_max_abs = []
    input_l2 = []
    target_ordered_rmse = []
    exact_input_equal = []

    for start in range(0, len(rows), 2):
        pair = rows[start:start + 2]
        pair_conditions = tuple(conditions[pair].tolist())
        if pair_conditions != EXPECTED_CONDITIONS:
            raise RuntimeError(
                f"unexpected pair condition order: {pair_conditions}"
            )
        if len(set(seeds[pair].tolist())) != 1:
            raise RuntimeError("pair visible-seed mismatch")
        if len(set(pair_keys[pair].tolist())) != 1:
            raise RuntimeError("pair-key mismatch")

        x0 = np.asarray(condition_z[pair[0]], dtype=np.float64)
        x1 = np.asarray(condition_z[pair[1]], dtype=np.float64)
        delta = x0 - x1
        input_max_abs.append(float(np.max(np.abs(delta))))
        input_l2.append(float(np.linalg.norm(delta)))
        exact_input_equal.append(bool(np.array_equal(x0, x1)))

        y0 = np.asarray(target_raw[pair[0], :, :48], dtype=np.float64)
        y1 = np.asarray(target_raw[pair[1], :, :48], dtype=np.float64)
        target_ordered_rmse.append(
            float(np.sqrt(np.mean((y0 - y1) ** 2)))
        )

    def stats(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    return {
        "pair_count": int(len(rows) // 2),
        "input_max_abs": stats(input_max_abs),
        "input_l2": stats(input_l2),
        "target_ordered_rmse": stats(target_ordered_rmse),
        "exact_input_equal_fraction": float(np.mean(exact_input_equal)),
    }


def make_optimizer(
    parameters,
    spec: OptimizerSpec,
) -> torch.optim.Optimizer:
    spec.validate()
    kwargs = {
        "lr": float(spec.learning_rate),
        "weight_decay": float(spec.weight_decay),
    }
    if spec.name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    return torch.optim.AdamW(parameters, **kwargs)


def build_denoiser(
    *,
    condition_dim: int,
    spec: DenoiserSpec,
) -> nn.Module:
    spec.validate()
    return MLPFutureDenoiser(
        condition_dim=int(condition_dim),
        hidden_dim=int(spec.hidden_dim),
        time_dim=int(spec.time_dim),
    )


class DirectFutureRegressor(nn.Module):
    """Condition-to-future MLP used only as a train-only capacity control."""

    def __init__(
        self,
        *,
        condition_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 4 * 87,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(condition_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, condition_z: torch.Tensor) -> torch.Tensor:
        if condition_z.ndim != 2:
            raise ValueError("condition_z must be [B,C]")
        return self.net(condition_z).reshape(condition_z.shape[0], 4, 87)


def masked_target_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    return active_mse(prediction, target, active_mask)


def make_noise(
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=clean_z.device).manual_seed(int(seed))
    noise = torch.randn(
        clean_z.shape,
        generator=generator,
        device=clean_z.device,
        dtype=clean_z.dtype,
    )
    active = active_mask.to(device=clean_z.device, dtype=torch.bool)
    return torch.where(active[None], noise, torch.zeros_like(noise))


def make_fixed_tuple(
    *,
    scheduler,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
    timestep: int,
    noise_seed: int,
) -> Dict[str, torch.Tensor]:
    if not 0 <= int(timestep) < 100:
        raise ValueError("timestep out of range")
    active = active_mask.to(device=clean_z.device, dtype=torch.bool)
    noise = make_noise(clean_z, active, seed=noise_seed)
    timesteps = torch.full(
        (clean_z.shape[0],),
        int(timestep),
        device=clean_z.device,
        dtype=torch.long,
    )
    noisy = scheduler.add_noise(clean_z, noise, timesteps)
    noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
    return {
        "noise": noise,
        "timesteps": timesteps,
        "noisy": noisy,
    }


@torch.no_grad()
def predict_x0_from_tuple(
    *,
    model: nn.Module,
    scheduler,
    repair_config,
    condition_z: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = active_mask.to(device=noisy.device, dtype=torch.bool)
    output = model(noisy, timesteps, condition_z)
    output = torch.where(active[None], output, torch.zeros_like(output))
    predicted_z = predict_original_sample(
        scheduler=scheduler,
        config=repair_config,
        sample=noisy,
        model_output=output,
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
    return predicted_z, predicted_raw


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("metric values must be finite/nonempty")
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def target_reconstruction_metrics(
    *,
    predicted_z: torch.Tensor | np.ndarray,
    target_z: torch.Tensor | np.ndarray,
    predicted_raw: torch.Tensor | np.ndarray,
    target_raw: torch.Tensor | np.ndarray,
    active_mask: torch.Tensor | np.ndarray,
) -> Dict[str, Any]:
    """Target-relative metrics for a memorization/capacity control."""
    pz = np.asarray(
        predicted_z.detach().cpu().numpy()
        if torch.is_tensor(predicted_z)
        else predicted_z,
        dtype=np.float64,
    )
    tz = np.asarray(
        target_z.detach().cpu().numpy()
        if torch.is_tensor(target_z)
        else target_z,
        dtype=np.float64,
    )
    pr = np.asarray(
        predicted_raw.detach().cpu().numpy()
        if torch.is_tensor(predicted_raw)
        else predicted_raw,
        dtype=np.float64,
    )
    tr = np.asarray(
        target_raw.detach().cpu().numpy()
        if torch.is_tensor(target_raw)
        else target_raw,
        dtype=np.float64,
    )
    active = np.asarray(
        active_mask.detach().cpu().numpy()
        if torch.is_tensor(active_mask)
        else active_mask,
        dtype=bool,
    )

    if pz.shape != tz.shape or pr.shape != tr.shape:
        raise ValueError("prediction/target shape mismatch")
    if pz.ndim != 3 or pz.shape[1:] != (4, 87):
        raise ValueError("future arrays must be [B,4,87]")
    if active.shape != (4, 87):
        raise ValueError("active mask must be [4,87]")
    if not all(np.isfinite(x).all() for x in (pz, tz, pr, tr)):
        raise ValueError("non-finite reconstruction arrays")

    expanded = np.broadcast_to(active[None], pz.shape)
    z_error = pz - tz
    z_mse = float(np.mean(np.square(z_error[expanded])))
    z_max_abs = float(np.max(np.abs(z_error[expanded])))

    pxy = pr[..., :48].reshape(pr.shape[0], 4, 24, 2)
    txy = tr[..., :48].reshape(tr.shape[0], 4, 24, 2)
    ordered_rmse = np.sqrt(np.mean((pxy - txy) ** 2, axis=(1, 2, 3)))

    pedge = pxy[:, :, 1:, :] - pxy[:, :, :-1, :]
    tedge = txy[:, :, 1:, :] - txy[:, :, :-1, :]
    edge_rmse = np.sqrt(np.mean((pedge - tedge) ** 2, axis=(1, 2, 3)))

    psegment = np.linalg.norm(pedge, axis=-1)
    tsegment = np.linalg.norm(tedge, axis=-1)
    segment_abs = np.abs(psegment - tsegment)
    segment_relative = segment_abs / np.maximum(tsegment, 1.0e-6)

    pchain = psegment.sum(axis=-1)
    tchain = tsegment.sum(axis=-1)
    chain_abs = np.abs(pchain - tchain)
    chain_relative = chain_abs / np.maximum(tchain, 1.0e-6)

    return {
        "row_count": int(pz.shape[0]),
        "z_mse": z_mse,
        "z_max_abs": z_max_abs,
        "ordered_rmse": _quantiles(ordered_rmse),
        "edge_vector_rmse": _quantiles(edge_rmse),
        "segment_absolute_error": _quantiles(segment_abs.reshape(-1)),
        "segment_relative_error": _quantiles(segment_relative.reshape(-1)),
        "chain_absolute_error": _quantiles(chain_abs.reshape(-1)),
        "chain_relative_error": _quantiles(chain_relative.reshape(-1)),
    }


def gate_reconstruction(
    metrics: Mapping[str, Any],
    gate: ReconstructionGate,
) -> bool:
    gate.validate()
    return bool(
        float(metrics["z_mse"]) <= gate.z_mse_max
        and float(metrics["ordered_rmse"]["p95"])
        <= gate.ordered_rmse_p95_max
        and float(metrics["segment_relative_error"]["p95"])
        <= gate.segment_relative_error_p95_max
        and float(metrics["chain_relative_error"]["p95"])
        <= gate.chain_relative_error_p95_max
    )


def _history_row(
    *,
    step: int,
    total_loss: torch.Tensor,
    gradient_norm: float,
) -> Dict[str, float | int]:
    return {
        "step": int(step),
        "total_loss": float(total_loss.detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm),
    }


def _gradient_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    clip_grad_norm: Optional[float],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad_norm is None:
        squared = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared += float(
                    torch.sum(parameter.grad.detach() ** 2).cpu()
                )
        gradient_norm = math.sqrt(squared)
    else:
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(clip_grad_norm),
            )
        )
    optimizer.step()
    return float(gradient_norm)


def train_direct_control(
    *,
    name: str,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    optimizer_spec: OptimizerSpec,
    hidden_dim: int,
    max_steps: int,
    seed: int,
    log_interval: int = 100,
) -> Dict[str, Any]:
    seed_all(seed)
    model = DirectFutureRegressor(
        condition_dim=condition_z.shape[1],
        hidden_dim=hidden_dim,
    ).to(condition_z.device)
    optimizer = make_optimizer(model.parameters(), optimizer_spec)
    history = []

    model.train()
    for step in range(1, int(max_steps) + 1):
        output = model(condition_z)
        output = torch.where(
            active_mask[None],
            output,
            torch.zeros_like(output),
        )
        loss = masked_target_mse(output, clean_z, active_mask)
        gradient_norm = _gradient_step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            clip_grad_norm=optimizer_spec.clip_grad_norm,
        )
        if step == 1 or step % log_interval == 0 or step == max_steps:
            history.append(
                _history_row(
                    step=step,
                    total_loss=loss,
                    gradient_norm=gradient_norm,
                )
            )
        if float(loss.detach().cpu()) <= 1.0e-7:
            break

    model.eval()
    with torch.no_grad():
        predicted_z = model(condition_z)
        predicted_z = torch.where(
            active_mask[None],
            predicted_z,
            torch.zeros_like(predicted_z),
        )
        predicted_raw = torch_inverse_standardize(
            predicted_z,
            future_mean,
            future_scale,
        )
    metrics = target_reconstruction_metrics(
        predicted_z=predicted_z,
        target_z=clean_z,
        predicted_raw=predicted_raw,
        target_raw=clean_raw,
        active_mask=active_mask,
    )
    passed = gate_reconstruction(metrics, DIRECT_REGRESSION_GATE)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "control_kind": "direct_regression",
        "row_count": int(condition_z.shape[0]),
        "optimizer": asdict(optimizer_spec),
        "hidden_dim": int(hidden_dim),
        "steps_completed": int(history[-1]["step"]),
        "history": history,
        "metrics": metrics,
        "gate": asdict(DIRECT_REGRESSION_GATE),
        "gate_pass": passed,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }


def _loss_for_mode(
    *,
    loss_mode: str,
    model_output: torch.Tensor,
    noisy_sample: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    normalizers,
) -> torch.Tensor:
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    if loss_mode == "v_only":
        target = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_z,
            noise=noise,
            timesteps=timesteps,
        )
        return active_mse(model_output, target, active_mask)
    if loss_mode == "r22_geometry":
        losses = geometry_aware_v_loss(
            model_output=model_output,
            noisy_sample=noisy_sample,
            clean_z=clean_z,
            clean_raw=clean_raw,
            noise=noise,
            timesteps=timesteps,
            scheduler=scheduler,
            repair_config=repair,
            active_mask=active_mask,
            future_mean=future_mean,
            future_scale=future_scale,
            normalizers=normalizers,
            geometry_config=GEOMETRY_CONFIGS["ordered_edge_temporal"],
            epoch=100,
        )
        return losses["total_loss"]
    raise ValueError(f"unsupported loss_mode: {loss_mode}")


def train_fixed_tuple_control(
    *,
    name: str,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    normalizers,
    denoiser_spec: DenoiserSpec,
    optimizer_spec: OptimizerSpec,
    loss_mode: str,
    timestep: int,
    noise_seed: int,
    max_steps: int,
    seed: int,
    fresh_noise_seed: int,
    log_interval: int = 100,
) -> Dict[str, Any]:
    """Train and evaluate a fixed (condition, x_t, t, target-v) tuple.

    The exact-replay gate uses the identical `noisy` tensor and timestep used
    during training. Fresh-noise results are diagnostic-only.
    """
    seed_all(seed)
    model = build_denoiser(
        condition_dim=condition_z.shape[1],
        spec=denoiser_spec,
    ).to(condition_z.device)
    optimizer = make_optimizer(model.parameters(), optimizer_spec)
    fixed = make_fixed_tuple(
        scheduler=scheduler,
        clean_z=clean_z,
        active_mask=active_mask,
        timestep=timestep,
        noise_seed=noise_seed,
    )
    history = []

    model.train()
    for step in range(1, int(max_steps) + 1):
        output = model(
            fixed["noisy"],
            fixed["timesteps"],
            condition_z,
        )
        output = torch.where(
            active_mask[None],
            output,
            torch.zeros_like(output),
        )
        loss = _loss_for_mode(
            loss_mode=loss_mode,
            model_output=output,
            noisy_sample=fixed["noisy"],
            clean_z=clean_z,
            clean_raw=clean_raw,
            noise=fixed["noise"],
            timesteps=fixed["timesteps"],
            scheduler=scheduler,
            active_mask=active_mask,
            future_mean=future_mean,
            future_scale=future_scale,
            normalizers=normalizers,
        )
        gradient_norm = _gradient_step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            clip_grad_norm=optimizer_spec.clip_grad_norm,
        )
        if step == 1 or step % log_interval == 0 or step == max_steps:
            history.append(
                _history_row(
                    step=step,
                    total_loss=loss,
                    gradient_norm=gradient_norm,
                )
            )
        if float(loss.detach().cpu()) <= 1.0e-7:
            break

    model.eval()
    replay_z, replay_raw = predict_x0_from_tuple(
        model=model,
        scheduler=scheduler,
        repair_config=REPAIR_CONFIGS["v_prediction_cosine"],
        condition_z=condition_z,
        noisy=fixed["noisy"],
        timesteps=fixed["timesteps"],
        active_mask=active_mask,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    replay_metrics = target_reconstruction_metrics(
        predicted_z=replay_z,
        target_z=clean_z,
        predicted_raw=replay_raw,
        target_raw=clean_raw,
        active_mask=active_mask,
    )

    fresh = make_fixed_tuple(
        scheduler=scheduler,
        clean_z=clean_z,
        active_mask=active_mask,
        timestep=timestep,
        noise_seed=fresh_noise_seed,
    )
    fresh_z, fresh_raw = predict_x0_from_tuple(
        model=model,
        scheduler=scheduler,
        repair_config=REPAIR_CONFIGS["v_prediction_cosine"],
        condition_z=condition_z,
        noisy=fresh["noisy"],
        timesteps=fresh["timesteps"],
        active_mask=active_mask,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    fresh_metrics = target_reconstruction_metrics(
        predicted_z=fresh_z,
        target_z=clean_z,
        predicted_raw=fresh_raw,
        target_raw=clean_raw,
        active_mask=active_mask,
    )

    replay_pass = gate_reconstruction(replay_metrics, EXACT_REPLAY_GATE)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "control_kind": "fixed_tuple_v_prediction",
        "row_count": int(condition_z.shape[0]),
        "loss_mode": loss_mode,
        "timestep": int(timestep),
        "training_noise_seed": int(noise_seed),
        "fresh_noise_seed": int(fresh_noise_seed),
        "denoiser": asdict(denoiser_spec),
        "optimizer": asdict(optimizer_spec),
        "steps_completed": int(history[-1]["step"]),
        "history": history,
        "exact_replay_metrics": replay_metrics,
        "fresh_noise_metrics": fresh_metrics,
        "exact_replay_gate": asdict(EXACT_REPLAY_GATE),
        "exact_replay_gate_pass": replay_pass,
        "fresh_noise_is_gate": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }


def evaluate_fresh_noise_grid(
    *,
    model: nn.Module,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    timesteps: Sequence[int],
    base_seed: int,
) -> Dict[str, Any]:
    result = {}
    for timestep in timesteps:
        fixed = make_fixed_tuple(
            scheduler=scheduler,
            clean_z=clean_z,
            active_mask=active_mask,
            timestep=int(timestep),
            noise_seed=int(base_seed) + int(timestep),
        )
        predicted_z, predicted_raw = predict_x0_from_tuple(
            model=model,
            scheduler=scheduler,
            repair_config=REPAIR_CONFIGS["v_prediction_cosine"],
            condition_z=condition_z,
            noisy=fixed["noisy"],
            timesteps=fixed["timesteps"],
            active_mask=active_mask,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        result[str(timestep)] = target_reconstruction_metrics(
            predicted_z=predicted_z,
            target_z=clean_z,
            predicted_raw=predicted_raw,
            target_raw=clean_raw,
            active_mask=active_mask,
        )
    return result


def train_random_noise_control(
    *,
    name: str,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    denoiser_spec: DenoiserSpec,
    optimizer_spec: OptimizerSpec,
    max_steps: int,
    seed: int,
    log_interval: int = 200,
) -> Dict[str, Any]:
    """Train v-prediction with fresh random t/noise; v-only isolation loss."""
    seed_all(seed)
    model = build_denoiser(
        condition_dim=condition_z.shape[1],
        spec=denoiser_spec,
    ).to(condition_z.device)
    optimizer = make_optimizer(model.parameters(), optimizer_spec)
    generator = torch.Generator(device=condition_z.device).manual_seed(
        int(seed) + 1000
    )
    history = []
    active = active_mask.to(device=condition_z.device, dtype=torch.bool)
    repair = REPAIR_CONFIGS["v_prediction_cosine"]

    model.train()
    for step in range(1, int(max_steps) + 1):
        timesteps = torch.randint(
            0,
            100,
            (clean_z.shape[0],),
            generator=generator,
            device=condition_z.device,
            dtype=torch.long,
        )
        noise = torch.randn(
            clean_z.shape,
            generator=generator,
            device=condition_z.device,
            dtype=clean_z.dtype,
        )
        noise = torch.where(active[None], noise, torch.zeros_like(noise))
        noisy = scheduler.add_noise(clean_z, noise, timesteps)
        noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))

        output = model(noisy, timesteps, condition_z)
        output = torch.where(active[None], output, torch.zeros_like(output))
        target = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_z,
            noise=noise,
            timesteps=timesteps,
        )
        loss = active_mse(output, target, active)
        gradient_norm = _gradient_step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            clip_grad_norm=optimizer_spec.clip_grad_norm,
        )
        if step == 1 or step % log_interval == 0 or step == max_steps:
            history.append(
                _history_row(
                    step=step,
                    total_loss=loss,
                    gradient_norm=gradient_norm,
                )
            )

    model.eval()
    evaluation = evaluate_fresh_noise_grid(
        model=model,
        scheduler=scheduler,
        condition_z=condition_z,
        clean_z=clean_z,
        clean_raw=clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        timesteps=(10, 50, 90, 99),
        base_seed=int(seed) + 2000,
    )

    # Capacity/optimizer gate uses t=50. High-timestep paired results are
    # diagnostic because branch-identical conditions can make per-row x0
    # reconstruction non-identifiable.
    t50_pass = gate_reconstruction(
        evaluation["50"],
        RANDOM_SINGLE_BRANCH_GATE,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "control_kind": "random_noise_v_prediction",
        "row_count": int(condition_z.shape[0]),
        "loss_mode": "v_only",
        "denoiser": asdict(denoiser_spec),
        "optimizer": asdict(optimizer_spec),
        "steps_completed": int(history[-1]["step"]),
        "history": history,
        "evaluation": evaluation,
        "single_branch_t50_gate": asdict(RANDOM_SINGLE_BRANCH_GATE),
        "single_branch_t50_gate_pass": t50_pass,
        "paired_result_is_capacity_gate": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }


def classify_controls(report: Mapping[str, Any]) -> Dict[str, str]:
    runs = report["runs"]

    if not runs["direct_one_row"]["gate_pass"]:
        return {
            "root_cause": (
                "phase314b_r231_direct_one_row_optimizer_or_capacity_failure"
            ),
            "next_stage": (
                "inspect tensor targets, optimizer updates and model output "
                "masking on one train row"
            ),
        }
    if not runs["direct_unique_free_16"]["gate_pass"]:
        return {
            "root_cause": (
                "phase314b_r231_direct_multirow_capacity_failure"
            ),
            "next_stage": (
                "isolate direct-regression width/optimizer before diffusion"
            ),
        }
    if not runs["fixed_one_row_v_only"]["exact_replay_gate_pass"]:
        return {
            "root_cause": (
                "phase314b_r231_fixed_v_target_or_replay_failure"
            ),
            "next_stage": (
                "inspect v targets, active mask and exact replay tensors"
            ),
        }

    fixed_candidates = (
        "fixed_pairs_v_only_adamw",
        "fixed_pairs_v_only_adam",
        "fixed_pairs_v_only_wide",
    )
    fixed_pass = any(
        runs[name]["exact_replay_gate_pass"]
        for name in fixed_candidates
    )
    if not fixed_pass:
        return {
            "root_cause": (
                "phase314b_r231_multirow_denoiser_capacity_or_optimizer_failure"
            ),
            "next_stage": (
                "redesign the MLP denoiser or optimizer using train-only "
                "fixed tuples"
            ),
        }

    if (
        runs["fixed_pairs_v_only_adamw"]["exact_replay_gate_pass"]
        and not runs["fixed_pairs_r22_geometry"]["exact_replay_gate_pass"]
    ):
        return {
            "root_cause": (
                "phase314b_r231_r22_geometry_loss_interference_supported"
            ),
            "next_stage": (
                "repair geometry-loss scaling/reduction on fixed train tuples"
            ),
        }

    if not runs["random_one_row"]["single_branch_t50_gate_pass"]:
        return {
            "root_cause": (
                "phase314b_r231_random_noise_single_row_optimization_failure"
            ),
            "next_stage": (
                "debug timestep/noise coverage and denoiser conditioning on "
                "one train row"
            ),
        }
    if not runs["random_unique_free_16"]["single_branch_t50_gate_pass"]:
        return {
            "root_cause": (
                "phase314b_r231_random_noise_multirow_capacity_failure"
            ),
            "next_stage": (
                "increase denoiser capacity or revise optimization before "
                "ordered-geometry repair"
            ),
        }

    if not runs["random_paired_16"]["single_branch_t50_gate_pass"]:
        ambiguity = report["pair_ambiguity"]
        if (
            float(ambiguity["input_max_abs"]["median"]) <= 1.0e-3
            and float(ambiguity["target_ordered_rmse"]["median"]) >= 1.0e-2
        ):
            return {
                "root_cause": (
                    "phase314b_r231_paired_branch_ambiguity_not_capacity_supported"
                ),
                "next_stage": (
                    "design r2.4 as a distributional ordered-geometry pilot; "
                    "do not require per-row high-noise reconstruction"
                ),
            }
        return {
            "root_cause": (
                "phase314b_r231_paired_random_noise_mixed_failure"
            ),
            "next_stage": (
                "inspect paired input distances and stochastic target "
                "identifiability before r2.4"
            ),
        }

    return {
        "root_cause": "phase314b_r231_tiny_model_optimizer_controls_supported",
        "next_stage": (
            "Phase3.14b-r2.4 train-only ordered-geometry objective pilot; "
            "formal validation remains blocked"
        ),
    }
