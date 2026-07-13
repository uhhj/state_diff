"""Train-only frozen-prior ordered-geometry repair for Phase3.14b-r2.5.

This module is additive.  It fixes the model contract to the r2.4.2
train-only recommendation ``frozen_p512_r512`` and varies only the geometry
objective.  It never reads validation targets or formal-test rows, never
persists model weights, and never modifies the immutable cache, frozen
contract, dataset, or DeformableRavens submodule.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ccda_phase3.phase314b_r21_geometry import calibrated_validity
from ccda_phase3.phase314b_r22_geometry import (
    contract_from_json,
    nearest_index_metrics,
    torch_inverse_standardize,
)
from ccda_phase3.phase314b_r231_controls import (
    RANDOM_SINGLE_BRANCH_GATE,
)
from ccda_phase3.phase314b_r241_multirow import (
    LabeledTupleBank,
    SourceBatchSampler,
    condition_ablation_banks,
    sample_aligned_source_batch,
    seed_everything,
)
from ccda_phase3.phase314b_r242_frozen_prior import (
    FactorizedAnalyticX0SkipDenoiser,
    evaluate_factorized_bank,
    evaluation_result_metrics,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    predict_original_sample,
    training_target,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


PHASE = "phase3_14b_r25"
BASE_REPORT_COMMIT = "80cb0ae3d6103503fe7100ab56587dfca7d366c5"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R242_ROOT_CAUSE = (
    "phase314b_r242_prior_gradient_isolation_supported"
)
EXPECTED_R242_RECOMMENDATION = "frozen_p512_r512"
EXPECTED_PAIRED_CONDITIONS = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)
UNIQUE_EVAL_TIMESTEPS = (10, 25, 50, 75, 90, 99)
PAIRED_GEOMETRY_TIMESTEP_MAX = 50
FULL_REVERSE_STEPS = 100
FULL_REVERSE_K = 16
COMMON_UNIQUE_TRAINING_SEED = 101000
COMMON_PAIRED_TRAINING_SEED = 102000
COMMON_REVERSE_SEED = 103000

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r25_ordered_geometry.py",
    "scripts/phase3_14b_r25_preflight.py",
    "scripts/phase3_14b_r25_run_pilot.py",
    "scripts/phase3_14b_r25_finalize.py",
    "scripts/phase3_14b_r25_run.sh",
    "tests/test_phase314b_r25_ordered_geometry.py",
)


@dataclass(frozen=True)
class GeometryObjectiveConfig:
    """One geometry objective while all model/training factors stay fixed."""

    name: str
    outer_weight: float
    mean_weight: float
    cvar_weight: float
    contract_weight: float
    tail_fraction: float = 0.10
    loss_balance_min: float = 0.05
    loss_balance_max: float = 20.0
    selectable: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("geometry objective name is required")
        for name, value in (
            ("outer_weight", self.outer_weight),
            ("mean_weight", self.mean_weight),
            ("cvar_weight", self.cvar_weight),
            ("contract_weight", self.contract_weight),
            ("tail_fraction", self.tail_fraction),
            ("loss_balance_min", self.loss_balance_min),
            ("loss_balance_max", self.loss_balance_max),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.outer_weight < 0:
            raise ValueError("outer_weight must be nonnegative")
        if min(self.mean_weight, self.cvar_weight, self.contract_weight) < 0:
            raise ValueError("geometry component weights must be nonnegative")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must lie in (0,1]")
        if self.loss_balance_min <= 0:
            raise ValueError("loss_balance_min must be positive")
        if self.loss_balance_max < self.loss_balance_min:
            raise ValueError("invalid loss-balance range")
        active = self.mean_weight + self.cvar_weight + self.contract_weight
        if self.selectable and (self.outer_weight <= 0 or active <= 0):
            raise ValueError("selectable geometry objective must be active")
        if not self.selectable and (self.outer_weight != 0 or active != 0):
            raise ValueError("control objective must have zero geometry weight")


GEOMETRY_OBJECTIVES: Tuple[GeometryObjectiveConfig, ...] = (
    GeometryObjectiveConfig(
        name="v_only_frozen_control",
        outer_weight=0.0,
        mean_weight=0.0,
        cvar_weight=0.0,
        contract_weight=0.0,
        selectable=False,
    ),
    GeometryObjectiveConfig(
        name="ordered_mean_raw",
        outer_weight=0.10,
        mean_weight=1.0,
        cvar_weight=0.0,
        contract_weight=0.0,
    ),
    GeometryObjectiveConfig(
        name="ordered_cvar_raw",
        outer_weight=0.10,
        mean_weight=0.50,
        cvar_weight=1.0,
        contract_weight=0.0,
    ),
    GeometryObjectiveConfig(
        name="ordered_cvar_contract",
        outer_weight=0.10,
        mean_weight=0.50,
        cvar_weight=1.0,
        contract_weight=0.50,
    ),
)
for _objective in GEOMETRY_OBJECTIVES:
    _objective.validate()


@dataclass(frozen=True)
class GeometryScales:
    """Dimensionally appropriate train-only geometry scales.

    Unlike r2.2, the ordered-coordinate scale is not a total chain length.
    It is the already-frozen random-denoising ordered-RMSE gate (1 cm).
    Edge and temporal-edge errors are normalized by the train-fit median
    physical segment length.  Relative segment and chain errors are already
    dimensionless.
    """

    ordered_xy_m: float
    edge_vector_m: float
    segment_floor_m: float
    chain_floor_m: float
    charbonnier_eps: float = 1.0e-3

    def validate(self) -> None:
        for value in asdict(self).values():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("geometry scales must be finite and positive")

    def to_json(self) -> Dict[str, float]:
        self.validate()
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class TorchGeometryContract:
    coordinate_lower: torch.Tensor
    coordinate_upper: torch.Tensor
    segment_center: torch.Tensor
    segment_scale: torch.Tensor
    segment_score_threshold: float
    chain_center: torch.Tensor
    chain_scale: torch.Tensor
    chain_score_threshold: float

    def to(self, device: torch.device, dtype: torch.dtype) -> "TorchGeometryContract":
        return TorchGeometryContract(
            coordinate_lower=self.coordinate_lower.to(device=device, dtype=dtype),
            coordinate_upper=self.coordinate_upper.to(device=device, dtype=dtype),
            segment_center=self.segment_center.to(device=device, dtype=dtype),
            segment_scale=self.segment_scale.to(device=device, dtype=dtype),
            segment_score_threshold=float(self.segment_score_threshold),
            chain_center=self.chain_center.to(device=device, dtype=dtype),
            chain_scale=self.chain_scale.to(device=device, dtype=dtype),
            chain_score_threshold=float(self.chain_score_threshold),
        )


@dataclass(frozen=True)
class GeometryTrainSpec:
    prior_steps: int = 5000
    residual_steps: int = 8000
    batch_size: int = 64
    prior_learning_rate: float = 1.0e-3
    residual_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None

    def validate(self) -> None:
        if self.prior_steps <= 0 or self.residual_steps <= 0:
            raise ValueError("training steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for value in (self.prior_learning_rate, self.residual_learning_rate):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("learning rates must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.clip_grad_norm is not None:
            if not math.isfinite(self.clip_grad_norm) or self.clip_grad_norm <= 0:
                raise ValueError("clip_grad_norm must be positive")


@dataclass(frozen=True)
class ReverseSamplingSpec:
    sample_count: int = FULL_REVERSE_K
    inference_steps: int = FULL_REVERSE_STEPS
    seed: int = 99501

    def validate(self) -> None:
        if self.sample_count <= 0 or self.inference_steps <= 0:
            raise ValueError("reverse sampling sizes must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=Path(root), text=True).strip()


def assert_only_allowed_worktree_paths(root: Path, allowed_paths: Sequence[str]) -> None:
    allowed = {str(Path(value).as_posix()) for value in allowed_paths}
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
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
        raise RuntimeError("unexpected worktree changes: " + ", ".join(unexpected))


def fit_geometry_scales(fit_future_raw: np.ndarray) -> GeometryScales:
    future = np.asarray(fit_future_raw, dtype=np.float32)
    if future.ndim != 3 or future.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("fit_future_raw must be [N,4,87]")
    if not np.isfinite(future).all():
        raise ValueError("fit_future_raw contains non-finite values")
    xy = future[..., :48].reshape(-1, DEFAULT_TF, 24, 2)
    edge = xy[..., 1:, :] - xy[..., :-1, :]
    segment = np.linalg.norm(edge, axis=-1)
    chain = segment.sum(axis=-1)
    scales = GeometryScales(
        ordered_xy_m=float(RANDOM_SINGLE_BRANCH_GATE.ordered_rmse_p95_max),
        edge_vector_m=max(float(np.median(segment)), 1.0e-3),
        segment_floor_m=max(float(np.percentile(segment, 5)), 1.0e-3),
        chain_floor_m=max(float(np.percentile(chain, 5)), 5.0e-2),
    )
    scales.validate()
    return scales


def torch_contract_from_payload(payload: Mapping[str, Any]) -> TorchGeometryContract:
    required = (
        "coordinate_lower",
        "coordinate_upper",
        "segment_center",
        "segment_scale",
        "segment_score_threshold",
        "chain_center",
        "chain_scale",
        "chain_score_threshold",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("physical contract missing: " + ", ".join(missing))
    result = TorchGeometryContract(
        coordinate_lower=torch.as_tensor(payload["coordinate_lower"], dtype=torch.float32),
        coordinate_upper=torch.as_tensor(payload["coordinate_upper"], dtype=torch.float32),
        segment_center=torch.as_tensor(payload["segment_center"], dtype=torch.float32),
        segment_scale=torch.as_tensor(payload["segment_scale"], dtype=torch.float32),
        segment_score_threshold=float(payload["segment_score_threshold"]),
        chain_center=torch.as_tensor(payload["chain_center"], dtype=torch.float32),
        chain_scale=torch.as_tensor(payload["chain_scale"], dtype=torch.float32),
        chain_score_threshold=float(payload["chain_score_threshold"]),
    )
    if result.coordinate_lower.shape != (2,) or result.coordinate_upper.shape != (2,):
        raise ValueError("coordinate contract shape mismatch")
    if result.segment_center.shape != (DEFAULT_TF, 23):
        raise ValueError("segment-center shape mismatch")
    if result.segment_scale.shape != (DEFAULT_TF, 23):
        raise ValueError("segment-scale shape mismatch")
    if result.chain_center.shape != (DEFAULT_TF,) or result.chain_scale.shape != (DEFAULT_TF,):
        raise ValueError("chain contract shape mismatch")
    tensors = (
        result.coordinate_lower,
        result.coordinate_upper,
        result.segment_center,
        result.segment_scale,
        result.chain_center,
        result.chain_scale,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("physical contract contains non-finite values")
    if bool(torch.any(result.segment_scale <= 0)) or bool(torch.any(result.chain_scale <= 0)):
        raise ValueError("physical contract scales must be positive")
    if result.segment_score_threshold <= 0 or result.chain_score_threshold <= 0:
        raise ValueError("physical contract thresholds must be positive")
    return result


def _charbonnier(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(value.square() + float(epsilon) ** 2) - float(epsilon)


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("tail values must be [B,M]")
    if values.shape[1] <= 0:
        raise ValueError("tail values are empty")
    count = max(1, int(math.ceil(values.shape[1] * float(fraction))))
    return torch.topk(values, k=count, dim=1, largest=True, sorted=False).values.mean(dim=1)


def geometry_timestep_mask(timesteps: torch.Tensor, max_timestep: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional")
    if max_timestep < 0 or max_timestep >= FULL_REVERSE_STEPS:
        raise ValueError("max_timestep out of range")
    return timesteps.long() <= int(max_timestep)


def ordered_geometry_loss_components(
    *,
    predicted_raw: torch.Tensor,
    target_raw: torch.Tensor,
    scales: GeometryScales,
    contract: TorchGeometryContract,
    tail_fraction: float,
) -> Dict[str, torch.Tensor]:
    """Return per-sample target-relative and family-wise geometry losses."""
    scales.validate()
    if predicted_raw.shape != target_raw.shape:
        raise ValueError("predicted/target raw shapes differ")
    if predicted_raw.ndim != 3 or predicted_raw.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("raw future must be [B,4,87]")
    if not bool(torch.isfinite(predicted_raw).all()) or not bool(torch.isfinite(target_raw).all()):
        raise ValueError("raw future contains non-finite values")

    batch = int(predicted_raw.shape[0])
    device = predicted_raw.device
    dtype = predicted_raw.dtype
    physical = contract.to(device, dtype)

    predicted_xy = predicted_raw[..., :48].reshape(batch, DEFAULT_TF, 24, 2)
    target_xy = target_raw[..., :48].reshape(batch, DEFAULT_TF, 24, 2)
    predicted_edge = predicted_xy[..., 1:, :] - predicted_xy[..., :-1, :]
    target_edge = target_xy[..., 1:, :] - target_xy[..., :-1, :]
    predicted_segment = torch.linalg.norm(predicted_edge, dim=-1)
    target_segment = torch.linalg.norm(target_edge, dim=-1)
    predicted_chain = predicted_segment.sum(dim=-1)
    target_chain = target_segment.sum(dim=-1)
    predicted_temporal = predicted_edge[:, 1:] - predicted_edge[:, :-1]
    target_temporal = target_edge[:, 1:] - target_edge[:, :-1]

    ordered_error = torch.linalg.norm(predicted_xy - target_xy, dim=-1) / float(scales.ordered_xy_m)
    edge_error = torch.linalg.norm(predicted_edge - target_edge, dim=-1) / float(scales.edge_vector_m)
    segment_relative = torch.abs(predicted_segment - target_segment) / torch.clamp(
        target_segment, min=float(scales.segment_floor_m)
    )
    chain_relative = torch.abs(predicted_chain - target_chain) / torch.clamp(
        target_chain, min=float(scales.chain_floor_m)
    )
    temporal_error = torch.linalg.norm(predicted_temporal - target_temporal, dim=-1) / float(scales.edge_vector_m)

    ordered_penalty = _charbonnier(ordered_error, scales.charbonnier_eps)
    edge_penalty = _charbonnier(edge_error, scales.charbonnier_eps)
    segment_penalty = _charbonnier(segment_relative, scales.charbonnier_eps)
    chain_penalty = _charbonnier(chain_relative, scales.charbonnier_eps)
    temporal_penalty = _charbonnier(temporal_error, scales.charbonnier_eps)

    mean_loss = (
        0.25 * ordered_penalty.flatten(1).mean(1)
        + 1.00 * edge_penalty.flatten(1).mean(1)
        + 1.00 * segment_penalty.flatten(1).mean(1)
        + 0.25 * chain_penalty.flatten(1).mean(1)
        + 0.25 * temporal_penalty.flatten(1).mean(1)
    )
    tail_values = torch.cat(
        (
            0.25 * ordered_penalty.flatten(1),
            1.00 * edge_penalty.flatten(1),
            1.00 * segment_penalty.flatten(1),
            0.25 * chain_penalty.flatten(1),
            0.25 * temporal_penalty.flatten(1),
        ),
        dim=1,
    )
    cvar_loss = _top_fraction_mean(tail_values, tail_fraction)

    segment_score = torch.abs(
        (predicted_segment - physical.segment_center[None])
        / physical.segment_scale[None]
    )
    chain_score = torch.abs(
        (predicted_chain - physical.chain_center[None])
        / physical.chain_scale[None]
    )
    segment_excess = torch.relu(
        segment_score.amax(dim=(1, 2)) / float(physical.segment_score_threshold) - 1.0
    )
    chain_excess = torch.relu(
        chain_score.amax(dim=1) / float(physical.chain_score_threshold) - 1.0
    )
    lower_excess = torch.relu(physical.coordinate_lower[None, None, None] - predicted_xy)
    upper_excess = torch.relu(predicted_xy - physical.coordinate_upper[None, None, None])
    coordinate_excess = torch.maximum(lower_excess, upper_excess).amax(dim=(1, 2, 3)) / float(scales.ordered_xy_m)
    contract_loss = (
        _charbonnier(segment_excess, scales.charbonnier_eps)
        + _charbonnier(chain_excess, scales.charbonnier_eps)
        + 0.25 * _charbonnier(coordinate_excess, scales.charbonnier_eps)
    )

    output = {
        "mean": mean_loss,
        "cvar": cvar_loss,
        "contract": contract_loss,
        "ordered_mean": ordered_penalty.flatten(1).mean(1),
        "edge_mean": edge_penalty.flatten(1).mean(1),
        "segment_relative_mean": segment_relative.flatten(1).mean(1),
        "segment_relative_max": segment_relative.flatten(1).amax(1),
        "chain_relative_mean": chain_relative.flatten(1).mean(1),
        "chain_relative_max": chain_relative.flatten(1).amax(1),
        "segment_contract_excess": segment_excess,
        "chain_contract_excess": chain_excess,
        "coordinate_contract_excess": coordinate_excess,
    }
    if not all(bool(torch.isfinite(value).all()) for value in output.values()):
        raise RuntimeError("non-finite ordered geometry component")
    return output


def combined_geometry_loss(
    *,
    components: Mapping[str, torch.Tensor],
    config: GeometryObjectiveConfig,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    config.validate()
    mask = sample_mask.to(dtype=torch.bool)
    if mask.ndim != 1:
        raise ValueError("sample_mask must be one-dimensional")
    if not bool(mask.any()):
        reference = components["mean"]
        return reference.new_zeros(())
    per_sample = (
        float(config.mean_weight) * components["mean"]
        + float(config.cvar_weight) * components["cvar"]
        + float(config.contract_weight) * components["contract"]
    )
    return per_sample[mask].mean()


def loss_balance_factor(
    v_loss: torch.Tensor,
    geometry_loss: torch.Tensor,
    config: GeometryObjectiveConfig,
) -> torch.Tensor:
    if config.outer_weight == 0:
        return v_loss.new_zeros(())
    ratio = v_loss.detach() / torch.clamp(geometry_loss.detach(), min=1.0e-8)
    return torch.clamp(
        ratio,
        min=float(config.loss_balance_min),
        max=float(config.loss_balance_max),
    )


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter], retain_graph: bool) -> float:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().square().sum()
    return float(torch.sqrt(total).cpu())


def _optimizer(parameters: Iterable[nn.Parameter], learning_rate: float, weight_decay: float):
    values = [parameter for parameter in parameters if parameter.requires_grad]
    if not values:
        raise ValueError("optimizer received no trainable parameters")
    return torch.optim.AdamW(values, lr=float(learning_rate), weight_decay=float(weight_decay))


def _gradient_step(
    *,
    optimizer,
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    clip_grad_norm: Optional[float],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameter_list = [
        value for value in parameters
        if value.requires_grad and value.grad is not None
    ]
    if not parameter_list:
        raise RuntimeError("loss produced no gradients")
    if clip_grad_norm is None:
        squared = sum(
            float(torch.sum(value.grad.detach().square()).cpu())
            for value in parameter_list
        )
        norm = math.sqrt(squared)
    else:
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameter_list,
                max_norm=float(clip_grad_norm),
            )
        )
    if not math.isfinite(norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return float(norm)


def _sample_timesteps(
    *,
    batch_size: int,
    values: Sequence[int],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    candidates = torch.tensor([int(value) for value in values], device=device, dtype=torch.long)
    if candidates.numel() == 0:
        raise ValueError("timestep values are empty")
    index = torch.randint(
        0,
        int(candidates.numel()),
        (int(batch_size),),
        generator=generator,
        device=device,
    )
    return candidates.index_select(0, index)


def fit_frozen_prior(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
    spec: GeometryTrainSpec,
) -> List[Dict[str, float]]:
    spec.validate()
    optimizer = _optimizer(
        model.prior_parameters(),
        learning_rate=spec.prior_learning_rate,
        weight_decay=spec.weight_decay,
    )
    active = active_mask.to(device=condition_z.device, dtype=torch.bool)
    history: List[Dict[str, float]] = []
    for step in range(1, int(spec.prior_steps) + 1):
        prediction = model.predict_base_x0(condition_z)
        loss = active_mse(prediction, clean_z, active)
        norm = _gradient_step(
            optimizer=optimizer,
            loss=loss,
            parameters=model.prior_parameters(),
            clip_grad_norm=spec.clip_grad_norm,
        )
        if step == 1 or step % 250 == 0 or step == int(spec.prior_steps):
            history.append({"step": int(step), "loss": float(loss.detach().cpu()), "gradient_norm": norm})
    for parameter in model.prior_parameters():
        parameter.requires_grad_(False)
    return history


def _prior_z_mse(
    model: FactorizedAnalyticX0SkipDenoiser,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
) -> float:
    with torch.no_grad():
        prediction = model.predict_base_x0(condition_z)
        return float(active_mse(prediction, clean_z, active_mask).cpu())


def _condition_effect(evaluations: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    required = {"true", "zero", "permuted"}
    if set(evaluations) != required:
        raise ValueError("condition ablation must contain true/zero/permuted")
    true_value = float(evaluation_result_metrics(evaluations["true"])["z_mse"])
    zero_value = float(evaluation_result_metrics(evaluations["zero"])["z_mse"])
    permuted_value = float(evaluation_result_metrics(evaluations["permuted"])["z_mse"])
    denominator = max(true_value, 1.0e-12)
    return {
        "true_z_mse": true_value,
        "zero_z_mse": zero_value,
        "permuted_z_mse": permuted_value,
        "zero_to_true_ratio": float(zero_value / denominator),
        "permuted_to_true_ratio": float(permuted_value / denominator),
        "condition_effect_supported": bool(
            zero_value >= 1.5 * denominator and permuted_value >= 1.5 * denominator
        ),
    }


def prediction_physical_metrics(
    *,
    predicted_z: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
) -> Dict[str, Any]:
    predicted_raw = torch_inverse_standardize(predicted_z, future_mean, future_scale)
    raw = predicted_raw.detach().cpu().numpy().astype(np.float32)
    validity = calibrated_validity(raw[None], physical_contract)
    return {
        key: value
        for key, value in validity.items()
        if key != "sample_valid_mask"
    }


def train_geometry_variant(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    evaluation_bank: LabeledTupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
    torch_contract: TorchGeometryContract,
    scales: GeometryScales,
    objective: GeometryObjectiveConfig,
    train_spec: GeometryTrainSpec,
    train_timesteps: Sequence[int],
    geometry_timestep_max: int,
    seed: int,
) -> Dict[str, Any]:
    """Train one strictly frozen p512/r512 model from scratch."""
    objective.validate()
    train_spec.validate()
    seed_everything(seed)
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    model = FactorizedAnalyticX0SkipDenoiser(
        condition_dim=int(condition_z.shape[1]),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        prior_hidden_dim=512,
        residual_hidden_dim=512,
    ).to(device)
    prior_history = fit_frozen_prior(
        model=model,
        condition_z=condition_z,
        clean_z=clean_z,
        active_mask=active,
        spec=train_spec,
    )
    prior_before = _prior_z_mse(model, condition_z, clean_z, active)
    residual_parameters = model.residual_parameters()
    optimizer = _optimizer(
        residual_parameters,
        learning_rate=train_spec.residual_learning_rate,
        weight_decay=train_spec.weight_decay,
    )
    sampler = SourceBatchSampler(
        source_count=int(condition_z.shape[0]),
        batch_size=int(train_spec.batch_size),
        mode="balanced",
        device=device,
        seed=int(seed) + 1000,
    )
    generator = torch.Generator(device=device).manual_seed(int(seed) + 2000)
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    history: List[Dict[str, Any]] = []
    gradient_audit: List[Dict[str, float]] = []

    for step in range(1, int(train_spec.residual_steps) + 1):
        _, condition_batch, clean_batch, raw_batch = sample_aligned_source_batch(
            condition_z=condition_z,
            clean_z=clean_z,
            clean_raw=clean_raw,
            sampler=sampler,
        )
        batch_size = int(condition_batch.shape[0])
        timesteps = _sample_timesteps(
            batch_size=batch_size,
            values=train_timesteps,
            generator=generator,
            device=device,
        )
        noise = torch.randn(
            clean_batch.shape,
            generator=generator,
            device=device,
            dtype=clean_batch.dtype,
        )
        noise = torch.where(active[None], noise, torch.zeros_like(noise))
        noisy = scheduler.add_noise(clean_batch, noise, timesteps)
        noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
        target_v = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_batch,
            noise=noise,
            timesteps=timesteps,
        )
        output = model(noisy, timesteps, condition_batch, detach_prior_for_v=True)
        output = torch.where(active[None], output, torch.zeros_like(output))
        v_loss = active_mse(output, target_v, active)

        if objective.outer_weight > 0:
            predicted_z = predict_original_sample(
                scheduler=scheduler,
                config=repair,
                sample=noisy,
                model_output=output,
                timesteps=timesteps,
            )
            predicted_z = torch.where(active[None], predicted_z, torch.zeros_like(predicted_z))
            predicted_raw = torch_inverse_standardize(predicted_z, future_mean, future_scale)
            components = ordered_geometry_loss_components(
                predicted_raw=predicted_raw,
                target_raw=raw_batch,
                scales=scales,
                contract=torch_contract,
                tail_fraction=objective.tail_fraction,
            )
            sample_mask = geometry_timestep_mask(timesteps, geometry_timestep_max)
            geometry_loss = combined_geometry_loss(
                components=components,
                config=objective,
                sample_mask=sample_mask,
            )
            balance = loss_balance_factor(v_loss, geometry_loss, objective)
            weighted_geometry = float(objective.outer_weight) * balance * geometry_loss
        else:
            components = {}
            geometry_loss = v_loss.new_zeros(())
            balance = v_loss.new_zeros(())
            weighted_geometry = v_loss.new_zeros(())
        total = v_loss + weighted_geometry

        if objective.outer_weight > 0 and (step == 1 or step == int(train_spec.residual_steps)):
            v_norm = _gradient_norm(v_loss, residual_parameters, retain_graph=True)
            geometry_norm = _gradient_norm(weighted_geometry, residual_parameters, retain_graph=True)
            gradient_audit.append(
                {
                    "step": int(step),
                    "v_gradient_norm": v_norm,
                    "weighted_geometry_gradient_norm": geometry_norm,
                    "geometry_to_v_gradient_ratio": float(geometry_norm / max(v_norm, 1.0e-12)),
                }
            )

        norm = _gradient_step(
            optimizer=optimizer,
            loss=total,
            parameters=residual_parameters,
            clip_grad_norm=train_spec.clip_grad_norm,
        )
        if step == 1 or step % 250 == 0 or step == int(train_spec.residual_steps):
            item: Dict[str, Any] = {
                "step": int(step),
                "total_loss": float(total.detach().cpu()),
                "v_loss": float(v_loss.detach().cpu()),
                "geometry_loss": float(geometry_loss.detach().cpu()),
                "weighted_geometry_loss": float(weighted_geometry.detach().cpu()),
                "loss_balance_factor": float(balance.detach().cpu()),
                "gradient_norm": norm,
                "geometry_sample_fraction": float(
                    geometry_timestep_mask(timesteps, geometry_timestep_max).float().mean().cpu()
                ),
            }
            for key in (
                "ordered_mean",
                "edge_mean",
                "segment_relative_mean",
                "segment_relative_max",
                "chain_relative_mean",
                "chain_relative_max",
                "segment_contract_excess",
                "chain_contract_excess",
            ):
                if key in components:
                    item[key] = float(components[key].detach().mean().cpu())
            history.append(item)

    prior_after = _prior_z_mse(model, condition_z, clean_z, active)
    prior_drift = float(prior_after / max(prior_before, 1.0e-12))
    ablation_banks = condition_ablation_banks(evaluation_bank, seed=int(seed) + 3000)
    evaluations_raw = {
        name: evaluate_factorized_bank(
            model=model,
            scheduler=scheduler,
            bank=bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        for name, bank in ablation_banks.items()
    }
    true_evaluation = evaluations_raw["true"]
    condition_effect = _condition_effect(evaluations_raw)
    physical = prediction_physical_metrics(
        predicted_z=true_evaluation["_predicted_z"],
        future_mean=future_mean,
        future_scale=future_scale,
        physical_contract=physical_contract,
    )
    geometry_gradient_ratio = (
        float(np.median([item["geometry_to_v_gradient_ratio"] for item in gradient_audit]))
        if gradient_audit
        else 0.0
    )
    gradient_gate = bool(
        objective.outer_weight == 0
        or (0.02 <= geometry_gradient_ratio <= 5.0)
    )
    result: Dict[str, Any] = {
        "architecture": {
            "family": model.family,
            "prior_hidden_dim": 512,
            "residual_hidden_dim": 512,
            "prior_policy": "strict_frozen",
        },
        "geometry_objective": asdict(objective),
        "train_spec": asdict(train_spec),
        "train_timesteps": [int(value) for value in train_timesteps],
        "geometry_timestep_max": int(geometry_timestep_max),
        "prior_history": prior_history,
        "residual_history": history,
        "gradient_audit": gradient_audit,
        "geometry_to_v_gradient_ratio_median": geometry_gradient_ratio,
        "geometry_gradient_gate": gradient_gate,
        "source_exposure": sampler.count_report(),
        "prior_z_mse_before_residual": prior_before,
        "prior_z_mse_after_residual": prior_after,
        "prior_drift_ratio": prior_drift,
        "evaluations": {
            name: {key: value for key, value in evaluation.items() if not key.startswith("_")}
            for name, evaluation in evaluations_raw.items()
        },
        "condition_effect": condition_effect,
        "one_step_physical": physical,
        "pass": bool(
            true_evaluation["gate_pass"]
            and condition_effect["condition_effect_supported"]
            and prior_drift <= 1.000001
            and gradient_gate
        ),
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "_model": model,
        "_true_prediction_z": true_evaluation["_predicted_z"],
    }
    return result


def strip_runtime_objects(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def _set_scheduler_timesteps(scheduler, count: int, device: torch.device) -> None:
    signature = inspect.signature(scheduler.set_timesteps)
    if "device" in signature.parameters:
        scheduler.set_timesteps(int(count), device=device)
    else:
        scheduler.set_timesteps(int(count))


def reverse_sample_pool(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    condition_z: torch.Tensor,
    active_mask: torch.Tensor,
    spec: ReverseSamplingSpec,
    matched_initial_noise_across_rows: bool,
) -> torch.Tensor:
    """Run the official 100-step scheduler and return [K,N,4,87] z samples."""
    spec.validate()
    model.eval()
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    source_count = int(condition_z.shape[0])
    generator = torch.Generator(device=device).manual_seed(int(spec.seed))
    if matched_initial_noise_across_rows:
        base = torch.randn(
            (int(spec.sample_count), 1, DEFAULT_TF, STATE_DIM),
            generator=generator,
            device=device,
            dtype=condition_z.dtype,
        )
        sample = base.expand(-1, source_count, -1, -1).clone()
    else:
        sample = torch.randn(
            (int(spec.sample_count), source_count, DEFAULT_TF, STATE_DIM),
            generator=generator,
            device=device,
            dtype=condition_z.dtype,
        )
    sample = torch.where(active[None, None], sample, torch.zeros_like(sample))
    flat_condition = condition_z[None].expand(int(spec.sample_count), -1, -1).reshape(
        int(spec.sample_count) * source_count, -1
    )
    flat = sample.reshape(int(spec.sample_count) * source_count, DEFAULT_TF, STATE_DIM)
    _set_scheduler_timesteps(scheduler, spec.inference_steps, device)
    step_signature = inspect.signature(scheduler.step)
    use_generator = "generator" in step_signature.parameters

    with torch.no_grad():
        for value in scheduler.timesteps:
            timestep_value = int(value.item()) if torch.is_tensor(value) else int(value)
            timesteps = torch.full(
                (flat.shape[0],),
                timestep_value,
                device=device,
                dtype=torch.long,
            )
            output = model(flat, timesteps, flat_condition, detach_prior_for_v=True)
            output = torch.where(active[None], output, torch.zeros_like(output))
            kwargs = {"generator": generator} if use_generator else {}
            flat = scheduler.step(output, timestep_value, flat, **kwargs).prev_sample
            flat = torch.where(active[None], flat, torch.zeros_like(flat))
            if not bool(torch.isfinite(flat).all()):
                raise RuntimeError(f"non-finite reverse sample at timestep {timestep_value}")
    return flat.reshape(int(spec.sample_count), source_count, DEFAULT_TF, STATE_DIM)


def reverse_pool_metrics(
    *,
    pool_z: torch.Tensor,
    target_raw: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
) -> Dict[str, Any]:
    if pool_z.ndim != 4:
        raise ValueError("pool_z must be [K,N,4,87]")
    pool_raw = torch_inverse_standardize(pool_z, future_mean, future_scale)
    raw = pool_raw.detach().cpu().numpy().astype(np.float32)
    target = target_raw.detach().cpu().numpy().astype(np.float32)
    validity = calibrated_validity(raw, physical_contract)
    prediction_xy = raw[..., -1, :48].reshape(raw.shape[0], raw.shape[1], 24, 2)
    target_xy = target[:, -1, :48].reshape(target.shape[0], 24, 2)
    ordered = np.sqrt(np.mean((prediction_xy - target_xy[None]) ** 2, axis=(-2, -1)))
    best = ordered.min(axis=0)
    inversion = np.stack(
        [nearest_index_metrics(sample, target)["nearest_inversion"] for sample in raw],
        axis=0,
    )
    output = {
        "sample_count": int(raw.shape[0]),
        "query_count": int(raw.shape[1]),
        "finite": bool(np.isfinite(raw).all()),
        "best_ordered_rmse_mean": float(np.mean(best)),
        "best_ordered_rmse_p95": float(np.percentile(best, 95)),
        "k1_ordered_rmse_mean": float(np.mean(ordered[0])),
        "nearest_inversion_p95": float(np.percentile(inversion, 95)),
        "pool_diversity": float(np.mean(np.std(raw[..., :48], axis=0))),
        "calibrated": {
            key: value
            for key, value in validity.items()
            if key != "sample_valid_mask"
        },
        "_pool_raw": raw,
        "_ordered_error": ordered,
        "_valid_mask": validity["sample_valid_mask"],
    }
    return output


def paired_reverse_pool_metrics(
    *,
    pool_z: torch.Tensor,
    paired_target_raw: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
) -> Dict[str, Any]:
    """Evaluate a pool against the nearer of two train-only paired targets."""
    if pool_z.ndim != 4:
        raise ValueError("pool_z must be [K,P,4,87]")
    targets = paired_target_raw
    if targets.ndim != 4 or targets.shape[1:] != (2, DEFAULT_TF, STATE_DIM):
        raise ValueError("paired_target_raw must be [P,2,4,87]")
    if pool_z.shape[1] != targets.shape[0]:
        raise ValueError("pool query count and pair count differ")
    pool_raw_tensor = torch_inverse_standardize(pool_z, future_mean, future_scale)
    raw = pool_raw_tensor.detach().cpu().numpy().astype(np.float32)
    target = targets.detach().cpu().numpy().astype(np.float32)
    validity = calibrated_validity(raw, physical_contract)
    prediction_xy = raw[..., -1, :48].reshape(raw.shape[0], raw.shape[1], 24, 2)
    target_xy = target[..., -1, :48].reshape(target.shape[0], 2, 24, 2)
    ordered = np.sqrt(
        np.mean(
            (prediction_xy[:, :, None] - target_xy[None]) ** 2,
            axis=(-2, -1),
        )
    )
    nearest_branch_error = np.min(ordered, axis=2)
    best = np.min(nearest_branch_error, axis=0)
    return {
        "sample_count": int(raw.shape[0]),
        "query_count": int(raw.shape[1]),
        "finite": bool(np.isfinite(raw).all()),
        "best_ordered_rmse_mean": float(np.mean(best)),
        "best_ordered_rmse_p95": float(np.percentile(best, 95)),
        "k1_ordered_rmse_mean": float(np.mean(nearest_branch_error[0])),
        "pool_diversity": float(np.mean(np.std(raw[..., :48], axis=0))),
        "calibrated": {
            key: value
            for key, value in validity.items()
            if key != "sample_valid_mask"
        },
        "_pool_raw": raw,
        "_ordered_error_by_branch": ordered,
        "_valid_mask": validity["sample_valid_mask"],
    }


def paired_full_reverse_branch_support(
    *,
    pool_raw: np.ndarray,
    paired_target_raw: np.ndarray,
    reference_threshold: float,
) -> Dict[str, Any]:
    """Measure whether one identical-condition pool covers both paired targets."""
    pool = np.asarray(pool_raw, dtype=np.float32)
    targets = np.asarray(paired_target_raw, dtype=np.float32)
    if pool.ndim != 4 or targets.ndim != 3:
        raise ValueError("invalid paired branch-support shapes")
    if targets.shape[0] % 2 != 0:
        raise ValueError("paired targets must contain complete pairs")
    pair_count = targets.shape[0] // 2
    if pool.shape[1] != pair_count:
        raise ValueError("pool query count must equal pair count")
    pool_xy = pool[..., -1, :48].reshape(pool.shape[0], pair_count, 24, 2)
    target_xy = targets[..., -1, :48].reshape(pair_count, 2, 24, 2)
    both_support: List[bool] = []
    occupancy_support: List[bool] = []
    separation_values: List[float] = []
    threshold_values: List[float] = []
    best_a_values: List[float] = []
    best_b_values: List[float] = []
    for pair in range(pair_count):
        target_a = target_xy[pair, 0]
        target_b = target_xy[pair, 1]
        separation = float(np.sqrt(np.mean((target_a - target_b) ** 2)))
        threshold = min(float(reference_threshold), max(1.0e-3, 0.45 * separation))
        errors_a = np.sqrt(np.mean((pool_xy[:, pair] - target_a[None]) ** 2, axis=(-2, -1)))
        errors_b = np.sqrt(np.mean((pool_xy[:, pair] - target_b[None]) ** 2, axis=(-2, -1)))
        best_a = float(np.min(errors_a))
        best_b = float(np.min(errors_b))
        assignment = errors_a < errors_b
        occupancy = bool(np.any(assignment) and np.any(~assignment))
        supported = bool(best_a <= threshold and best_b <= threshold)
        both_support.append(supported)
        occupancy_support.append(occupancy)
        separation_values.append(separation)
        threshold_values.append(threshold)
        best_a_values.append(best_a)
        best_b_values.append(best_b)
    return {
        "pair_count": int(pair_count),
        "both_branch_support_rate": float(np.mean(both_support)),
        "two_branch_occupancy_rate": float(np.mean(occupancy_support)),
        "target_separation_p50": float(np.percentile(separation_values, 50)),
        "support_threshold_p50": float(np.percentile(threshold_values, 50)),
        "best_branch_a_error_p95": float(np.percentile(best_a_values, 95)),
        "best_branch_b_error_p95": float(np.percentile(best_b_values, 95)),
        "pass": bool(
            np.mean(both_support) >= 0.75
            and np.mean(occupancy_support) >= 0.75
        ),
    }


def paired_target_rows_to_queries(
    condition_z: torch.Tensor,
    target_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if condition_z.shape[0] % 2 != 0 or target_raw.shape[0] != condition_z.shape[0]:
        raise ValueError("paired tensors must contain complete aligned pairs")
    condition_a = condition_z[0::2]
    condition_b = condition_z[1::2]
    difference = float(torch.max(torch.abs(condition_a - condition_b)).detach().cpu())
    if difference > 1.0e-6:
        raise RuntimeError(f"paired deployable inputs are not identical: {difference}")
    paired_targets = target_raw.reshape(-1, 2, DEFAULT_TF, STATE_DIM)
    return condition_a, paired_targets.reshape(-1, DEFAULT_TF, STATE_DIM)


def compare_geometry_to_control(
    *,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_reverse = candidate["reverse_metrics"]
    control_reverse = control["reverse_metrics"]
    candidate_calibrated = candidate_reverse["calibrated"]
    control_calibrated = control_reverse["calibrated"]
    candidate_segment = float(candidate_calibrated["segment_score_p95"])
    control_segment = float(control_calibrated["segment_score_p95"])
    candidate_validity = float(candidate_calibrated["sample_validity_rate"])
    control_validity = float(control_calibrated["sample_validity_rate"])
    candidate_best = float(candidate_reverse["best_ordered_rmse_mean"])
    control_best = float(control_reverse["best_ordered_rmse_mean"])
    tail_improved = bool(
        candidate_segment <= 0.90 * max(control_segment, 1.0e-12)
        or candidate_validity >= control_validity + 0.05
    )
    ordered_preserved = bool(candidate_best <= 1.10 * max(control_best, 1.0e-12))
    return {
        "segment_score_p95_ratio": float(candidate_segment / max(control_segment, 1.0e-12)),
        "sample_validity_delta": float(candidate_validity - control_validity),
        "best_ordered_rmse_ratio": float(candidate_best / max(control_best, 1.0e-12)),
        "tail_improved": tail_improved,
        "ordered_preserved": ordered_preserved,
        "pass": bool(tail_improved and ordered_preserved),
    }


def classify_pilot(report: Mapping[str, Any]) -> Dict[str, Any]:
    unique = report.get("unique_free_variants", {})
    paired = report.get("paired_variants", {})
    control_name = "v_only_frozen_control"
    if control_name not in unique or not bool(unique[control_name].get("pass")):
        return {
            "root_cause": "phase314b_r25_frozen_baseline_integration_failed",
            "next_stage": "repair the r2.4.2 frozen-p512/r512 integration before geometry training",
            "train_only_recommendation": None,
        }
    geometry_names = [
        objective.name for objective in GEOMETRY_OBJECTIVES if objective.selectable
    ]
    gradient_active = [
        name for name in geometry_names
        if bool(unique.get(name, {}).get("geometry_gradient_gate"))
    ]
    if not gradient_active:
        return {
            "root_cause": "phase314b_r25_geometry_gradient_scaling_failed",
            "next_stage": "repair train-only geometry scaling before another pilot",
            "train_only_recommendation": None,
        }
    unique_pass = [name for name in gradient_active if bool(unique.get(name, {}).get("pass"))]
    if not unique_pass:
        return {
            "root_cause": "phase314b_r25_geometry_objective_destabilized_unique_denoising",
            "next_stage": "reduce or reformulate the train-only geometry objective",
            "train_only_recommendation": None,
        }
    paired_pass = [
        name for name in unique_pass
        if bool(paired.get(name, {}).get("one_step_pass"))
    ]
    if not paired_pass:
        return {
            "root_cause": "phase314b_r25_paired_low_mid_geometry_transport_failed",
            "next_stage": "debug paired low/mid ordered-geometry transport with frozen prior",
            "train_only_recommendation": None,
        }
    control = paired.get(control_name)
    if not isinstance(control, Mapping):
        return {
            "root_cause": "phase314b_r25_missing_reverse_control",
            "next_stage": "repair train-only reverse-sampling control evidence",
            "train_only_recommendation": None,
        }
    supported: List[Tuple[str, float]] = []
    mode_collapse = []
    for name in paired_pass:
        value = paired[name]
        branch = value.get("full_reverse_branch_support", {})
        comparison = value.get("comparison_to_control", {})
        reverse = value.get("reverse_metrics", {})
        calibrated = reverse.get("calibrated", {})
        support_ok = bool(branch.get("pass"))
        valid_query_ok = float(calibrated.get("query_has_valid_candidate_rate", 0.0)) >= 0.875
        if not support_ok:
            mode_collapse.append(name)
            continue
        if valid_query_ok and bool(comparison.get("pass")):
            score = (
                float(branch.get("both_branch_support_rate", 0.0))
                + float(calibrated.get("sample_validity_rate", 0.0))
                - float(reverse.get("best_ordered_rmse_mean", 1.0))
            )
            supported.append((name, score))
    if supported:
        supported.sort(key=lambda item: (-item[1], item[0]))
        return {
            "root_cause": "phase314b_r25_train_only_frozen_prior_geometry_repair_supported",
            "next_stage": "repeat the recommended geometry objective across three train-only seeds before any validation access",
            "train_only_recommendation": supported[0][0],
        }
    if mode_collapse:
        return {
            "root_cause": "phase314b_r25_geometry_induced_branch_support_collapse",
            "next_stage": "repair geometry-tail weighting without collapsing paired candidate support",
            "train_only_recommendation": None,
        }
    return {
        "root_cause": "phase314b_r25_reverse_ordered_geometry_failure_persists",
        "next_stage": "diagnose full reverse geometry accumulation under the frozen-prior model",
        "train_only_recommendation": None,
    }
