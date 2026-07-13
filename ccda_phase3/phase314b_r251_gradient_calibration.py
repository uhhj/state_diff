"""Train-only geometry-gradient calibration for Phase3.14b-r2.5.1.

This additive module fixes the model contract to the r2.4.2/r2.5 frozen
p512/r512 recommendation and changes only the geometry-gradient multiplier.
It never reads validation targets or formal-test rows, never persists model
weights, and never modifies the immutable cache, frozen contract, dataset, or
DeformableRavens submodule.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ccda_phase3.phase314b_r21_geometry import calibrated_validity
from ccda_phase3.phase314b_r22_geometry import nearest_index_metrics, torch_inverse_standardize
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
from ccda_phase3.phase314b_r25_ordered_geometry import (
    GeometryObjectiveConfig,
    GeometryScales,
    GeometryTrainSpec,
    ReverseSamplingSpec,
    TorchGeometryContract,
    combined_geometry_loss,
    fit_frozen_prior,
    geometry_timestep_mask,
    ordered_geometry_loss_components,
    paired_full_reverse_branch_support,
    reverse_sample_pool,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import active_mse, predict_original_sample, training_target
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM

PHASE = "phase3_14b_r251"
BASE_REPORT_COMMIT = "9d40e0264fc2bc1f1d5576b9d9833aedd426ea98"
BASE_IMPLEMENTATION_COMMIT = "88de2de2b5b88729c7e2d126493a3c3677f9b141"
BASE_BLOCKED_REPORT_COMMIT = "d396e9e23ad429c9b6a7a1db874107657f91d65e"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R25_ROOT_CAUSE = "phase314b_r25_geometry_gradient_scaling_failed"
EXPECTED_R25_RECOMMENDATION = None
EXPECTED_PAIRED_CONDITIONS = ("free", "hidden_slack_breakaway_pin_v2")
EXPECTED_UNIQUE_ROWS = (6, 10, 19, 25, 31, 36, 43, 49, 55, 59, 65, 69, 74, 81, 88, 92)
EXPECTED_PAIRED_ROWS = (
    6, 1226,
    10, 1230,
    19, 1239,
    25, 1245,
    31, 1251,
    36, 1256,
    43, 1263,
    49, 1269,
)
PAIRED_INVERSION_SCHEMA = "phase314b_r251_paired_reverse_batched_inversion_v2"

COMMON_UNIQUE_TRAINING_SEED = 101000
COMMON_PAIRED_TRAINING_SEED = 102000
COMMON_REVERSE_SEED = 103000
COMMON_UNIQUE_PRIOR_SEED = COMMON_UNIQUE_TRAINING_SEED
COMMON_PAIRED_PRIOR_SEED = COMMON_PAIRED_TRAINING_SEED
CALIBRATION_SEED_OFFSET = 5000
TRAINING_SAMPLER_SEED_OFFSET = 1000
TRAINING_NOISE_SEED_OFFSET = 2000
CONDITION_ABLATION_SEED_OFFSET = 3000

UNIQUE_EVAL_TIMESTEPS = (10, 25, 50, 75, 90, 99)
PAIRED_EVAL_TIMESTEPS = (10, 25, 50)
PAIRED_GEOMETRY_TIMESTEP_MAX = 50
FULL_REVERSE_STEPS = 100
FULL_REVERSE_K = 16
TARGET_GRADIENT_RATIOS = (0.10, 0.50, 1.00)
CALIBRATION_BATCHES = 8
CALIBRATION_MULTIPLIER_MIN = 1.0e-10
CALIBRATION_MULTIPLIER_MAX = 100.0
GRADIENT_SAFE_MIN = 0.02
GRADIENT_SAFE_MAX = 5.0
TARGET_TRACK_FACTOR = 4.0

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r251_gradient_calibration.py",
    "scripts/phase3_14b_r251_preflight.py",
    "scripts/phase3_14b_r251_run_pilot.py",
    "scripts/phase3_14b_r251_finalize.py",
    "scripts/phase3_14b_r251_run.sh",
    "tests/test_phase314b_r251_gradient_calibration.py",
)


@dataclass(frozen=True)
class CalibratedGeometryObjective:
    """One geometry family at one frozen target gradient ratio."""

    name: str
    family: str
    target_gradient_ratio: float
    mean_weight: float
    cvar_weight: float
    contract_weight: float
    tail_fraction: float = 0.10
    selectable: bool = True

    def validate(self) -> None:
        if not self.name or not self.family:
            raise ValueError("geometry objective name and family are required")
        values = (
            self.target_gradient_ratio,
            self.mean_weight,
            self.cvar_weight,
            self.contract_weight,
            self.tail_fraction,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("geometry objective values must be finite")
        if min(self.mean_weight, self.cvar_weight, self.contract_weight) < 0:
            raise ValueError("geometry component weights must be nonnegative")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must lie in (0,1]")
        active = self.mean_weight + self.cvar_weight + self.contract_weight
        if self.selectable:
            if self.target_gradient_ratio <= 0 or active <= 0:
                raise ValueError("selectable calibrated objective must be active")
        else:
            if self.target_gradient_ratio != 0 or active != 0:
                raise ValueError("control must have zero geometry and target ratio")

    def r25_config(self) -> GeometryObjectiveConfig:
        """Adapter for the already-tested r2.5 component aggregator.

        The returned outer weight is one only because r2.5's
        ``combined_geometry_loss`` validates a complete objective. r2.5.1 does
        not use r2.5's loss-magnitude balancing or outer weighting.
        """
        self.validate()
        return GeometryObjectiveConfig(
            name=self.name,
            outer_weight=1.0 if self.selectable else 0.0,
            mean_weight=float(self.mean_weight),
            cvar_weight=float(self.cvar_weight),
            contract_weight=float(self.contract_weight),
            tail_fraction=float(self.tail_fraction),
            selectable=bool(self.selectable),
        )


def _ratio_suffix(value: float) -> str:
    return f"g{int(round(float(value) * 100)):03d}"


def _objective_matrix() -> Tuple[CalibratedGeometryObjective, ...]:
    values: List[CalibratedGeometryObjective] = [
        CalibratedGeometryObjective(
            name="v_only_frozen_control",
            family="v_only",
            target_gradient_ratio=0.0,
            mean_weight=0.0,
            cvar_weight=0.0,
            contract_weight=0.0,
            selectable=False,
        )
    ]
    families = (
        ("ordered_mean_raw", 1.0, 0.0, 0.0),
        ("ordered_cvar_raw", 0.5, 1.0, 0.0),
        ("ordered_cvar_contract", 0.5, 1.0, 0.5),
    )
    for family, mean_weight, cvar_weight, contract_weight in families:
        for target in TARGET_GRADIENT_RATIOS:
            values.append(
                CalibratedGeometryObjective(
                    name=f"{family}_{_ratio_suffix(target)}",
                    family=family,
                    target_gradient_ratio=float(target),
                    mean_weight=float(mean_weight),
                    cvar_weight=float(cvar_weight),
                    contract_weight=float(contract_weight),
                )
            )
    result = tuple(values)
    for objective in result:
        objective.validate()
    return result


CALIBRATED_GEOMETRY_OBJECTIVES = _objective_matrix()


@dataclass(frozen=True)
class GradientCalibrationSpec:
    batch_count: int = CALIBRATION_BATCHES
    multiplier_min: float = CALIBRATION_MULTIPLIER_MIN
    multiplier_max: float = CALIBRATION_MULTIPLIER_MAX
    safe_min: float = GRADIENT_SAFE_MIN
    safe_max: float = GRADIENT_SAFE_MAX
    target_track_factor: float = TARGET_TRACK_FACTOR

    def validate(self) -> None:
        if self.batch_count <= 0:
            raise ValueError("calibration batch_count must be positive")
        values = (
            self.multiplier_min,
            self.multiplier_max,
            self.safe_min,
            self.safe_max,
            self.target_track_factor,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("calibration bounds must be finite and positive")
        if self.multiplier_max < self.multiplier_min:
            raise ValueError("invalid calibration multiplier bounds")
        if self.safe_max < self.safe_min:
            raise ValueError("invalid gradient safe bounds")
        if self.target_track_factor < 1:
            raise ValueError("target_track_factor must be at least one")


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


def _optimizer(parameters: Iterable[nn.Parameter], learning_rate: float, weight_decay: float):
    values = [parameter for parameter in parameters if parameter.requires_grad]
    if not values:
        raise ValueError("optimizer received no trainable parameters")
    return torch.optim.AdamW(values, lr=float(learning_rate), weight_decay=float(weight_decay))


def _gradient_norm(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    retain_graph: bool,
) -> float:
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
    value = float(torch.sqrt(total).cpu())
    if not math.isfinite(value):
        raise RuntimeError("non-finite gradient norm")
    return value


def _gradient_step(
    *,
    optimizer,
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    clip_grad_norm: Optional[float],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    values = [
        parameter
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not values:
        raise RuntimeError("loss produced no gradients")
    if clip_grad_norm is None:
        norm = math.sqrt(
            sum(float(torch.sum(value.grad.detach().square()).cpu()) for value in values)
        )
    else:
        norm = float(
            torch.nn.utils.clip_grad_norm_(values, max_norm=float(clip_grad_norm))
        )
    if not math.isfinite(norm):
        raise RuntimeError("non-finite optimizer gradient norm")
    optimizer.step()
    return float(norm)


def _sample_timesteps(
    *,
    batch_size: int,
    values: Sequence[int],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    candidates = torch.tensor(
        [int(value) for value in values],
        device=device,
        dtype=torch.long,
    )
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


def _prior_z_mse(
    model: FactorizedAnalyticX0SkipDenoiser,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
) -> float:
    with torch.no_grad():
        return float(
            active_mse(
                model.predict_base_x0(condition_z),
                clean_z,
                active_mask,
            ).cpu()
        )


def _condition_effect(evaluations: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if set(evaluations) != {"true", "zero", "permuted"}:
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
            zero_value >= 1.5 * denominator
            and permuted_value >= 1.5 * denominator
        ),
    }


def _prediction_physical_metrics(
    *,
    predicted_z: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
) -> Dict[str, Any]:
    predicted_raw = torch_inverse_standardize(predicted_z, future_mean, future_scale)
    raw = predicted_raw.detach().cpu().numpy().astype(np.float32)
    validity = calibrated_validity(raw[None], physical_contract)
    return {key: value for key, value in validity.items() if key != "sample_valid_mask"}


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = torch.as_tensor(state[key]).detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fit_shared_prior_snapshot(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    active_mask: torch.Tensor,
    train_spec: GeometryTrainSpec,
    seed: int,
) -> Dict[str, Any]:
    """Fit one prior per dataset stage and keep its state in memory only."""
    train_spec.validate()
    seed_everything(int(seed))
    model = FactorizedAnalyticX0SkipDenoiser(
        condition_dim=int(condition_z.shape[1]),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        prior_hidden_dim=512,
        residual_hidden_dim=512,
    ).to(condition_z.device)
    history = fit_frozen_prior(
        model=model,
        condition_z=condition_z,
        clean_z=clean_z,
        active_mask=active_mask,
        spec=train_spec,
    )
    prior_mse = _prior_z_mse(model, condition_z, clean_z, active_mask)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.x0_prior.state_dict().items()
    }
    return {
        "seed": int(seed),
        "prior_history": history,
        "prior_z_mse": float(prior_mse),
        "state_tensor_count": int(len(state)),
        "prior_state_sha256": tensor_state_sha256(state),
        "_prior_state": state,
    }


def instantiate_snapshot_model(
    *,
    scheduler,
    condition_dim: int,
    device: torch.device,
    snapshot: Mapping[str, Any],
    seed: int,
) -> FactorizedAnalyticX0SkipDenoiser:
    state = snapshot.get("_prior_state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("shared prior snapshot state is missing")
    seed_everything(int(seed))
    model = FactorizedAnalyticX0SkipDenoiser(
        condition_dim=int(condition_dim),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        prior_hidden_dim=512,
        residual_hidden_dim=512,
    ).to(device)
    model.x0_prior.load_state_dict(state, strict=True)
    loaded_state = {
        key: value.detach().cpu()
        for key, value in model.x0_prior.state_dict().items()
    }
    if tensor_state_sha256(loaded_state) != snapshot.get("prior_state_sha256"):
        raise RuntimeError("loaded prior snapshot hash mismatch")
    for parameter in model.prior_parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.prior_parameters()):
        raise RuntimeError("prior snapshot was not frozen")
    return model


def geometry_loss_for_objective(
    *,
    components: Mapping[str, torch.Tensor],
    objective: CalibratedGeometryObjective,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    if not objective.selectable:
        return components["mean"].new_zeros(())
    return combined_geometry_loss(
        components=components,
        config=objective.r25_config(),
        sample_mask=sample_mask,
    )


def calibrated_multiplier_from_gradient_norms(
    *,
    target_ratio: float,
    v_gradient_norms: Sequence[float],
    geometry_gradient_norms: Sequence[float],
    spec: GradientCalibrationSpec,
) -> Dict[str, Any]:
    """Compute one frozen multiplier from actual residual-gradient norms."""
    spec.validate()
    if not math.isfinite(target_ratio) or target_ratio <= 0:
        raise ValueError("target_ratio must be finite and positive")
    v = np.asarray(v_gradient_norms, dtype=np.float64)
    geometry = np.asarray(geometry_gradient_norms, dtype=np.float64)
    if v.ndim != 1 or geometry.ndim != 1 or v.shape != geometry.shape:
        raise ValueError("gradient norm arrays must be aligned one-dimensional values")
    if v.size != int(spec.batch_count):
        raise ValueError("gradient norm count does not match calibration batch_count")
    if not np.isfinite(v).all() or not np.isfinite(geometry).all():
        raise ValueError("calibration gradient norms contain non-finite values")
    if np.any(v <= 0) or np.any(geometry <= 0):
        raise ValueError("calibration gradient norms must be positive")
    raw_ratio = geometry / v
    raw_median = float(np.median(raw_ratio))
    multiplier_candidates = float(target_ratio) * v / geometry
    multiplier = float(target_ratio / raw_median)
    weighted = multiplier * raw_ratio
    weighted_median = float(np.median(weighted))
    multiplier_gate = bool(
        spec.multiplier_min <= multiplier <= spec.multiplier_max
    )
    target_gate = bool(
        abs(weighted_median - float(target_ratio))
        <= max(1.0e-8, 1.0e-6 * float(target_ratio))
    )
    return {
        "schema": "phase314b_r251_fixed_gradient_multiplier_v1",
        "target_gradient_ratio": float(target_ratio),
        "batch_count": int(v.size),
        "v_gradient_norms": v.tolist(),
        "geometry_gradient_norms": geometry.tolist(),
        "raw_geometry_to_v_ratios": raw_ratio.tolist(),
        "raw_ratio_p05": float(np.percentile(raw_ratio, 5)),
        "raw_ratio_median": raw_median,
        "raw_ratio_p95": float(np.percentile(raw_ratio, 95)),
        "multiplier_candidates": multiplier_candidates.tolist(),
        "multiplier": multiplier,
        "weighted_ratios": weighted.tolist(),
        "weighted_ratio_p05": float(np.percentile(weighted, 5)),
        "weighted_ratio_median": weighted_median,
        "weighted_ratio_p95": float(np.percentile(weighted, 95)),
        "multiplier_gate": multiplier_gate,
        "target_gate": target_gate,
        "pass": bool(multiplier_gate and target_gate),
    }


def gradient_tracking_gate(
    *,
    target_ratio: float,
    observed_ratios: Sequence[float],
    spec: GradientCalibrationSpec,
) -> Dict[str, Any]:
    spec.validate()
    values = np.asarray(observed_ratios, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observed gradient ratios must be non-empty")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("observed gradient ratios must be finite and nonnegative")
    median = float(np.median(values))
    p95 = float(np.percentile(values, 95))
    lower = max(float(spec.safe_min), float(target_ratio) / float(spec.target_track_factor))
    upper = min(float(spec.safe_max), float(target_ratio) * float(spec.target_track_factor))
    target_tracked = bool(lower <= median <= upper)
    safe = bool(median >= spec.safe_min and p95 <= spec.safe_max)
    return {
        "target_gradient_ratio": float(target_ratio),
        "observed_ratios": values.tolist(),
        "observed_ratio_p05": float(np.percentile(values, 5)),
        "observed_ratio_median": median,
        "observed_ratio_p95": p95,
        "observed_ratio_max": float(np.max(values)),
        "target_tracking_lower": lower,
        "target_tracking_upper": upper,
        "target_tracked": target_tracked,
        "safe_gradient_gate": safe,
        "pass": bool(target_tracked and safe),
    }


def _forward_losses(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    repair,
    condition_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    raw_batch: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    torch_contract: TorchGeometryContract,
    scales: GeometryScales,
    objective: CalibratedGeometryObjective,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    geometry_timestep_max: int,
) -> Dict[str, Any]:
    active = active_mask.to(device=clean_batch.device, dtype=torch.bool)
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
    if not objective.selectable:
        return {
            "v_loss": v_loss,
            "geometry_loss": v_loss.new_zeros(()),
            "components": {},
            "geometry_mask": torch.zeros_like(timesteps, dtype=torch.bool),
        }
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
    mask = geometry_timestep_mask(timesteps, geometry_timestep_max)
    if not bool(mask.any()):
        raise RuntimeError("calibrated geometry batch contains no active timestep")
    geometry_loss = geometry_loss_for_objective(
        components=components,
        objective=objective,
        sample_mask=mask,
    )
    return {
        "v_loss": v_loss,
        "geometry_loss": geometry_loss,
        "components": components,
        "geometry_mask": mask,
    }


def calibrate_geometry_multiplier(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    torch_contract: TorchGeometryContract,
    scales: GeometryScales,
    objective: CalibratedGeometryObjective,
    train_timesteps: Sequence[int],
    geometry_timestep_max: int,
    batch_size: int,
    seed: int,
    spec: GradientCalibrationSpec,
) -> Dict[str, Any]:
    objective.validate()
    spec.validate()
    if not objective.selectable:
        return {
            "schema": "phase314b_r251_control_no_geometry_calibration_v1",
            "target_gradient_ratio": 0.0,
            "multiplier": 0.0,
            "pass": True,
        }
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    sampler = SourceBatchSampler(
        source_count=int(condition_z.shape[0]),
        batch_size=int(batch_size),
        mode="balanced",
        device=device,
        seed=int(seed) + CALIBRATION_SEED_OFFSET,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(seed) + CALIBRATION_SEED_OFFSET + 1
    )
    residual_parameters = model.residual_parameters()
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    v_norms: List[float] = []
    geometry_norms: List[float] = []
    for _ in range(int(spec.batch_count)):
        _, condition_batch, clean_batch, raw_batch = sample_aligned_source_batch(
            condition_z=condition_z,
            clean_z=clean_z,
            clean_raw=clean_raw,
            sampler=sampler,
        )
        timesteps = _sample_timesteps(
            batch_size=int(condition_batch.shape[0]),
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
        losses = _forward_losses(
            model=model,
            scheduler=scheduler,
            repair=repair,
            condition_batch=condition_batch,
            clean_batch=clean_batch,
            raw_batch=raw_batch,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            torch_contract=torch_contract,
            scales=scales,
            objective=objective,
            timesteps=timesteps,
            noise=noise,
            geometry_timestep_max=geometry_timestep_max,
        )
        v_norms.append(
            _gradient_norm(losses["v_loss"], residual_parameters, retain_graph=True)
        )
        geometry_norms.append(
            _gradient_norm(
                losses["geometry_loss"],
                residual_parameters,
                retain_graph=False,
            )
        )
    result = calibrated_multiplier_from_gradient_norms(
        target_ratio=objective.target_gradient_ratio,
        v_gradient_norms=v_norms,
        geometry_gradient_norms=geometry_norms,
        spec=spec,
    )
    result["source_exposure"] = sampler.count_report()
    return result


def train_calibrated_geometry_variant(
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
    objective: CalibratedGeometryObjective,
    train_spec: GeometryTrainSpec,
    calibration_spec: GradientCalibrationSpec,
    train_timesteps: Sequence[int],
    geometry_timestep_max: int,
    shared_prior_snapshot: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Train one calibrated residual from an identical frozen prior snapshot."""
    objective.validate()
    train_spec.validate()
    calibration_spec.validate()
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    model = instantiate_snapshot_model(
        scheduler=scheduler,
        condition_dim=int(condition_z.shape[1]),
        device=device,
        snapshot=shared_prior_snapshot,
        seed=int(seed),
    )
    prior_before = _prior_z_mse(model, condition_z, clean_z, active)
    snapshot_mse = float(shared_prior_snapshot["prior_z_mse"])
    if abs(prior_before - snapshot_mse) > max(1.0e-12, 1.0e-6 * snapshot_mse):
        raise RuntimeError("loaded prior snapshot MSE mismatch")

    calibration = calibrate_geometry_multiplier(
        model=model,
        scheduler=scheduler,
        condition_z=condition_z,
        clean_z=clean_z,
        clean_raw=clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        torch_contract=torch_contract,
        scales=scales,
        objective=objective,
        train_timesteps=train_timesteps,
        geometry_timestep_max=geometry_timestep_max,
        batch_size=int(train_spec.batch_size),
        seed=int(seed),
        spec=calibration_spec,
    )
    if not bool(calibration["pass"]):
        prior_after = _prior_z_mse(model, condition_z, clean_z, active)
        return {
            "architecture": {
                "family": model.family,
                "prior_hidden_dim": 512,
                "residual_hidden_dim": 512,
                "prior_policy": "strict_frozen_shared_snapshot",
            },
            "geometry_objective": asdict(objective),
            "train_spec": asdict(train_spec),
            "calibration_spec": asdict(calibration_spec),
            "calibration": calibration,
            "gradient_audit": [],
            "gradient_tracking": {
                "target_gradient_ratio": float(objective.target_gradient_ratio),
                "observed_ratios": [],
                "observed_ratio_median": 0.0,
                "observed_ratio_p95": 0.0,
                "target_tracked": False,
                "safe_gradient_gate": False,
                "pass": False,
            },
            "source_exposure": calibration.get("source_exposure", {}),
            "prior_z_mse_before_residual": float(prior_before),
            "prior_z_mse_after_residual": float(prior_after),
            "prior_drift_ratio": float(prior_after / max(prior_before, 1.0e-12)),
            "evaluations": {},
            "condition_effect": {"condition_effect_supported": False},
            "one_step_physical": {},
            "pass": False,
            "failure": "gradient_calibration_failed",
            "candidate_eligible": False,
            "checkpoint_saved": False,
            "_model": model,
        }

    multiplier = float(calibration["multiplier"])
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
        seed=int(seed) + TRAINING_SAMPLER_SEED_OFFSET,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(seed) + TRAINING_NOISE_SEED_OFFSET
    )
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    history: List[Dict[str, Any]] = []
    gradient_audit: List[Dict[str, float]] = []
    audit_steps = {
        1,
        max(2, int(train_spec.residual_steps) // 4),
        max(2, int(train_spec.residual_steps) // 2),
        int(train_spec.residual_steps),
    }

    for step in range(1, int(train_spec.residual_steps) + 1):
        _, condition_batch, clean_batch, raw_batch = sample_aligned_source_batch(
            condition_z=condition_z,
            clean_z=clean_z,
            clean_raw=clean_raw,
            sampler=sampler,
        )
        timesteps = _sample_timesteps(
            batch_size=int(condition_batch.shape[0]),
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
        losses = _forward_losses(
            model=model,
            scheduler=scheduler,
            repair=repair,
            condition_batch=condition_batch,
            clean_batch=clean_batch,
            raw_batch=raw_batch,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            torch_contract=torch_contract,
            scales=scales,
            objective=objective,
            timesteps=timesteps,
            noise=noise,
            geometry_timestep_max=geometry_timestep_max,
        )
        v_loss = losses["v_loss"]
        geometry_loss = losses["geometry_loss"]
        weighted_geometry = float(multiplier) * geometry_loss
        total = v_loss + weighted_geometry

        if objective.selectable and step in audit_steps:
            v_norm = _gradient_norm(v_loss, residual_parameters, retain_graph=True)
            geometry_norm = _gradient_norm(
                geometry_loss,
                residual_parameters,
                retain_graph=True,
            )
            weighted_ratio = float(multiplier * geometry_norm / max(v_norm, 1.0e-12))
            gradient_audit.append(
                {
                    "step": int(step),
                    "v_gradient_norm": float(v_norm),
                    "unweighted_geometry_gradient_norm": float(geometry_norm),
                    "gradient_multiplier": float(multiplier),
                    "weighted_geometry_to_v_gradient_ratio": weighted_ratio,
                }
            )

        optimizer_norm = _gradient_step(
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
                "gradient_multiplier": float(multiplier),
                "optimizer_gradient_norm": float(optimizer_norm),
                "geometry_sample_fraction": float(
                    losses["geometry_mask"].float().mean().detach().cpu()
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
                if key in losses["components"]:
                    item[key] = float(losses["components"][key].detach().mean().cpu())
            history.append(item)

    if objective.selectable:
        tracking = gradient_tracking_gate(
            target_ratio=objective.target_gradient_ratio,
            observed_ratios=[
                item["weighted_geometry_to_v_gradient_ratio"]
                for item in gradient_audit
            ],
            spec=calibration_spec,
        )
    else:
        tracking = {
            "target_gradient_ratio": 0.0,
            "observed_ratios": [],
            "observed_ratio_median": 0.0,
            "observed_ratio_p95": 0.0,
            "target_tracked": True,
            "safe_gradient_gate": True,
            "pass": True,
        }

    prior_after = _prior_z_mse(model, condition_z, clean_z, active)
    prior_drift = float(prior_after / max(prior_before, 1.0e-12))
    ablation_banks = condition_ablation_banks(
        evaluation_bank,
        seed=int(seed) + CONDITION_ABLATION_SEED_OFFSET,
    )
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
    physical = _prediction_physical_metrics(
        predicted_z=true_evaluation["_predicted_z"],
        future_mean=future_mean,
        future_scale=future_scale,
        physical_contract=physical_contract,
    )
    result: Dict[str, Any] = {
        "architecture": {
            "family": model.family,
            "prior_hidden_dim": 512,
            "residual_hidden_dim": 512,
            "prior_policy": "strict_frozen_shared_snapshot",
        },
        "geometry_objective": asdict(objective),
        "train_spec": asdict(train_spec),
        "calibration_spec": asdict(calibration_spec),
        "train_timesteps": [int(value) for value in train_timesteps],
        "geometry_timestep_max": int(geometry_timestep_max),
        "calibration": calibration,
        "gradient_audit": gradient_audit,
        "gradient_tracking": tracking,
        "residual_history": history,
        "source_exposure": sampler.count_report(),
        "prior_z_mse_before_residual": float(prior_before),
        "prior_z_mse_after_residual": float(prior_after),
        "prior_drift_ratio": prior_drift,
        "evaluations": {
            name: {
                key: value
                for key, value in evaluation.items()
                if not key.startswith("_")
            }
            for name, evaluation in evaluations_raw.items()
        },
        "condition_effect": condition_effect,
        "one_step_physical": physical,
        "pass": bool(
            true_evaluation["gate_pass"]
            and condition_effect["condition_effect_supported"]
            and prior_drift <= 1.000001
            and calibration["pass"]
            and tracking["pass"]
        ),
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "_model": model,
        "_true_prediction_z": true_evaluation["_predicted_z"],
    }
    return result


def assert_canonical_paired_row_contract(
    arrays: Mapping[str, np.ndarray],
    paired_rows: Sequence[int],
) -> Dict[str, Any]:
    """Verify the immutable paired-row identity and metadata contract.

    Row identifiers are audit metadata only. They are never exposed to the
    model. The exact assertion prevents a copied/reporting typo from silently
    becoming a different train-only dataset selection.
    """
    rows = np.asarray(paired_rows, dtype=np.int64)
    expected_rows = np.asarray(EXPECTED_PAIRED_ROWS, dtype=np.int64)
    if rows.shape != expected_rows.shape:
        raise RuntimeError(
            f"paired-row shape mismatch: {rows.shape} != {expected_rows.shape}"
        )
    if not np.array_equal(rows, expected_rows):
        raise RuntimeError(
            "paired-row identity mismatch: "
            f"observed={rows.tolist()} expected={expected_rows.tolist()}"
        )
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("paired-row contract contains duplicate row indices")

    condition = np.asarray(arrays["condition_name"]).astype(str)[rows]
    visible_seed = np.asarray(arrays["visible_seed"]).astype(np.int64)[rows]
    pair_key = np.asarray(arrays["pair_key"]).astype(str)[rows]
    expected_conditions = np.asarray(EXPECTED_PAIRED_CONDITIONS * 8, dtype=str)
    if not np.array_equal(condition, expected_conditions):
        raise RuntimeError(
            "paired-row condition order mismatch: "
            f"observed={condition.tolist()} expected={expected_conditions.tolist()}"
        )

    pair_records: List[Dict[str, Any]] = []
    for pair_index in range(8):
        begin = 2 * pair_index
        row_pair = rows[begin : begin + 2]
        seed_pair = visible_seed[begin : begin + 2]
        key_pair = pair_key[begin : begin + 2]
        if seed_pair[0] != seed_pair[1]:
            raise RuntimeError(
                f"paired visible-seed mismatch at pair {pair_index}: {seed_pair.tolist()}"
            )
        if key_pair[0] != key_pair[1]:
            raise RuntimeError(
                f"paired pair-key mismatch at pair {pair_index}: {key_pair.tolist()}"
            )
        pair_records.append(
            {
                "pair_index": int(pair_index),
                "rows": row_pair.astype(int).tolist(),
                "visible_seed": int(seed_pair[0]),
                "pair_key": str(key_pair[0]),
                "conditions": condition[begin : begin + 2].tolist(),
            }
        )

    pair_seeds = [record["visible_seed"] for record in pair_records]
    pair_keys = [record["pair_key"] for record in pair_records]
    if len(set(pair_seeds)) != 8:
        raise RuntimeError("paired-row contract does not contain eight distinct seeds")
    if len(set(pair_keys)) != 8:
        raise RuntimeError("paired-row contract does not contain eight distinct pair keys")

    return {
        "pass": True,
        "expected_rows": expected_rows.astype(int).tolist(),
        "observed_rows": rows.astype(int).tolist(),
        "distinct_row_count": int(len(np.unique(rows))),
        "distinct_visible_seed_count": int(len(set(pair_seeds))),
        "distinct_pair_key_count": int(len(set(pair_keys))),
        "pairs": pair_records,
        "model_input_use": False,
    }


def _as_future_batch(value: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize one future or a future batch to finite [N,4,87]."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError(
            f"{name} must be [4,87] or [N,4,87], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def paired_nearest_index_audit(
    *,
    pool_raw: np.ndarray,
    paired_target_raw: np.ndarray,
    nearest_branch: np.ndarray,
) -> Dict[str, Any]:
    """Compute nearest-index metrics in one explicit batched call.

    ``nearest_index_metrics`` has a strict [N,4,87] contract. The previous
    r2.5.1 implementation iterated over singleton [4,87] tensors and therefore
    failed at runtime. This adapter constructs the selected target batch and
    calls the canonical metric exactly once.
    """
    raw = np.asarray(pool_raw, dtype=np.float32)
    target = np.asarray(paired_target_raw, dtype=np.float32)
    branch = np.asarray(nearest_branch, dtype=np.int64)
    if raw.ndim != 4 or raw.shape[2:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("pool_raw must be [K,P,4,87]")
    if target.ndim != 4 or target.shape[1:] != (2, DEFAULT_TF, STATE_DIM):
        raise ValueError("paired_target_raw must be [P,2,4,87]")
    if branch.shape != raw.shape[:2]:
        raise ValueError(
            f"nearest_branch must be [K,P], got {branch.shape} for {raw.shape[:2]}"
        )
    if raw.shape[1] != target.shape[0]:
        raise ValueError("pool query count and pair count differ")
    if not np.isfinite(raw).all() or not np.isfinite(target).all():
        raise ValueError("paired inversion inputs must be finite")
    if np.any((branch < 0) | (branch > 1)):
        raise ValueError("nearest_branch values must be zero or one")

    sample_count, query_count = branch.shape
    query_index = np.broadcast_to(
        np.arange(query_count, dtype=np.int64)[None, :],
        branch.shape,
    )
    selected_target = target[query_index, branch]
    prediction_batch = _as_future_batch(
        raw.reshape(sample_count * query_count, DEFAULT_TF, STATE_DIM),
        name="paired prediction batch",
    )
    target_batch = _as_future_batch(
        selected_target.reshape(sample_count * query_count, DEFAULT_TF, STATE_DIM),
        name="paired selected-target batch",
    )
    metric = nearest_index_metrics(prediction_batch, target_batch)
    inversion = np.asarray(metric["nearest_inversion"], dtype=np.float64).reshape(-1)
    unique_fraction = np.asarray(
        metric["nearest_unique_fraction"], dtype=np.float64
    ).reshape(-1)
    expected_count = sample_count * query_count
    if inversion.shape != (expected_count,):
        raise RuntimeError(
            "nearest inversion result shape mismatch: "
            f"{inversion.shape} != {(expected_count,)}"
        )
    if unique_fraction.shape != (expected_count,):
        raise RuntimeError(
            "nearest unique-fraction result shape mismatch: "
            f"{unique_fraction.shape} != {(expected_count,)}"
        )
    if not np.isfinite(inversion).all() or not np.isfinite(unique_fraction).all():
        raise RuntimeError("paired nearest-index metrics are non-finite")
    if np.any((inversion < 0) | (inversion > 1)):
        raise RuntimeError("paired nearest inversion lies outside [0,1]")
    if np.any((unique_fraction < 0) | (unique_fraction > 1)):
        raise RuntimeError("paired nearest unique fraction lies outside [0,1]")

    inversion = inversion.reshape(sample_count, query_count)
    unique_fraction = unique_fraction.reshape(sample_count, query_count)
    return {
        "schema": PAIRED_INVERSION_SCHEMA,
        "batched_item_count": int(expected_count),
        "nearest_inversion_mean": float(np.mean(inversion)),
        "nearest_inversion_p95": float(np.percentile(inversion, 95)),
        "nearest_inversion_max": float(np.max(inversion)),
        "nearest_unique_fraction_mean": float(np.mean(unique_fraction)),
        "nearest_unique_fraction_p05": float(np.percentile(unique_fraction, 5)),
        "nearest_unique_fraction_min": float(np.min(unique_fraction)),
        "_nearest_inversion": inversion,
        "_nearest_unique_fraction": unique_fraction,
        "_selected_target_batch": selected_target,
    }


def paired_reverse_pool_metrics_with_inversion(
    *,
    pool_z: torch.Tensor,
    paired_target_raw: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract,
) -> Dict[str, Any]:
    """Evaluate K paired-query samples with batched nearest-index metrics."""
    if pool_z.ndim != 4 or tuple(pool_z.shape[2:]) != (DEFAULT_TF, STATE_DIM):
        raise ValueError("pool_z must be [K,P,4,87]")
    targets = paired_target_raw
    if targets.ndim != 4 or tuple(targets.shape[1:]) != (
        2,
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("paired_target_raw must be [P,2,4,87]")
    if pool_z.shape[1] != targets.shape[0]:
        raise ValueError("pool query count and pair count differ")
    if not bool(torch.isfinite(pool_z).all()):
        raise ValueError("pool_z must be finite")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("paired_target_raw must be finite")

    pool_raw_tensor = torch_inverse_standardize(pool_z, future_mean, future_scale)
    raw = pool_raw_tensor.detach().cpu().numpy().astype(np.float32)
    target = targets.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(raw).all():
        raise RuntimeError("inverse-standardized reverse pool is non-finite")

    validity = calibrated_validity(raw, physical_contract)
    prediction_xy = raw[..., -1, :48].reshape(raw.shape[0], raw.shape[1], 24, 2)
    target_xy = target[..., -1, :48].reshape(target.shape[0], 2, 24, 2)
    ordered = np.sqrt(
        np.mean(
            (prediction_xy[:, :, None] - target_xy[None]) ** 2,
            axis=(-2, -1),
        )
    )
    nearest_branch = np.argmin(ordered, axis=2)
    nearest_branch_error = np.min(ordered, axis=2)
    inversion_audit = paired_nearest_index_audit(
        pool_raw=raw,
        paired_target_raw=target,
        nearest_branch=nearest_branch,
    )
    best = np.min(nearest_branch_error, axis=0)
    result = {
        "sample_count": int(raw.shape[0]),
        "query_count": int(raw.shape[1]),
        "finite": True,
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
        "_nearest_branch": nearest_branch,
        "_valid_mask": validity["sample_valid_mask"],
    }
    for key, value in inversion_audit.items():
        if key == "_selected_target_batch":
            continue
        result[key] = value
    return result


def compare_calibrated_to_control(
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
    candidate_inversion = float(candidate_reverse["nearest_inversion_p95"])
    control_inversion = float(control_reverse["nearest_inversion_p95"])
    tail_improved = bool(
        candidate_segment <= 0.90 * max(control_segment, 1.0e-12)
        or candidate_validity >= control_validity + 0.05
    )
    ordered_preserved = bool(candidate_best <= 1.10 * max(control_best, 1.0e-12))
    inversion_limit = max(
        control_inversion * 1.10,
        control_inversion + 1.0e-6,
    )
    inversion_preserved = bool(candidate_inversion <= inversion_limit)
    return {
        "segment_score_p95_ratio": float(
            candidate_segment / max(control_segment, 1.0e-12)
        ),
        "sample_validity_delta": float(candidate_validity - control_validity),
        "best_ordered_rmse_ratio": float(
            candidate_best / max(control_best, 1.0e-12)
        ),
        "nearest_inversion_p95_ratio": float(
            candidate_inversion / max(control_inversion, 1.0e-12)
        ),
        "nearest_inversion_limit": float(inversion_limit),
        "tail_improved": tail_improved,
        "ordered_preserved": ordered_preserved,
        "inversion_preserved": inversion_preserved,
        "pass": bool(tail_improved and ordered_preserved and inversion_preserved),
    }


def strip_runtime_objects(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: strip_runtime_objects(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, tuple):
        return [strip_runtime_objects(item) for item in value]
    if isinstance(value, list):
        return [strip_runtime_objects(item) for item in value]
    return value


def classify_pilot(report: Mapping[str, Any]) -> Dict[str, Any]:
    unique = report.get("unique_free_variants", {})
    paired = report.get("paired_variants", {})
    control_name = "v_only_frozen_control"
    control_unique = unique.get(control_name, {})
    if not bool(control_unique.get("pass")):
        return {
            "root_cause": "phase314b_r251_frozen_baseline_integration_failed",
            "next_stage": "repair the frozen-p512/r512 baseline integration",
            "train_only_recommendation": None,
        }
    selectable = [
        objective.name
        for objective in CALIBRATED_GEOMETRY_OBJECTIVES
        if objective.selectable
    ]
    calibrated = [
        name
        for name in selectable
        if bool(unique.get(name, {}).get("calibration", {}).get("pass"))
    ]
    if not calibrated:
        return {
            "root_cause": "phase314b_r251_gradient_calibration_contract_failed",
            "next_stage": "repair actual-gradient calibration without changing geometry objectives",
            "train_only_recommendation": None,
        }
    tracking = [
        name
        for name in calibrated
        if bool(unique.get(name, {}).get("gradient_tracking", {}).get("pass"))
    ]
    if not tracking:
        return {
            "root_cause": "phase314b_r251_geometry_gradient_nonstationarity_supported",
            "next_stage": "diagnose gradient-ratio drift under fixed calibrated multipliers",
            "train_only_recommendation": None,
        }
    unique_pass = [name for name in tracking if bool(unique.get(name, {}).get("pass"))]
    if not unique_pass:
        return {
            "root_cause": "phase314b_r251_calibrated_geometry_destabilized_unique_denoising",
            "next_stage": "reformulate geometry loss after successful gradient calibration",
            "train_only_recommendation": None,
        }
    paired_pass = [
        name
        for name in unique_pass
        if bool(paired.get(name, {}).get("one_step_pass"))
    ]
    if not paired_pass:
        return {
            "root_cause": "phase314b_r251_paired_low_mid_geometry_transport_failed",
            "next_stage": "diagnose paired low/mid transport under calibrated geometry gradients",
            "train_only_recommendation": None,
        }
    control = paired.get(control_name)
    if not isinstance(control, Mapping):
        return {
            "root_cause": "phase314b_r251_missing_reverse_control",
            "next_stage": "repair calibrated reverse-control evidence",
            "train_only_recommendation": None,
        }
    supported: List[Tuple[str, float]] = []
    mode_collapse: List[str] = []
    branch_preserving: List[str] = []
    for name in paired_pass:
        value = paired[name]
        branch = value.get("full_reverse_branch_support", {})
        comparison = value.get("comparison_to_control", {})
        reverse = value.get("reverse_metrics", {})
        calibrated_metrics = reverse.get("calibrated", {})
        if not bool(branch.get("pass")):
            mode_collapse.append(name)
            continue
        branch_preserving.append(name)
        valid_query = float(
            calibrated_metrics.get("query_has_valid_candidate_rate", 0.0)
        ) >= 0.875
        if valid_query and bool(comparison.get("pass")):
            score = (
                2.0 * float(calibrated_metrics.get("sample_validity_rate", 0.0))
                + float(branch.get("both_branch_support_rate", 0.0))
                + float(branch.get("two_branch_occupancy_rate", 0.0))
                - float(reverse.get("best_ordered_rmse_mean", 1.0))
                - 0.01 * float(reverse.get("nearest_inversion_p95", 1.0))
            )
            supported.append((name, score))
    if supported:
        supported.sort(key=lambda item: (-item[1], item[0]))
        return {
            "root_cause": "phase314b_r251_train_only_gradient_calibration_supported",
            "next_stage": "replicate the recommended calibrated objective across three train-only seeds",
            "train_only_recommendation": supported[0][0],
        }
    if mode_collapse and not branch_preserving:
        return {
            "root_cause": "phase314b_r251_calibrated_geometry_branch_support_collapse",
            "next_stage": "repair geometry gain without collapsing paired branch support",
            "train_only_recommendation": None,
        }
    return {
        "root_cause": "phase314b_r251_reverse_geometry_gain_not_supported",
        "next_stage": "diagnose reverse-step geometry accumulation under calibrated gradients",
        "train_only_recommendation": None,
    }
