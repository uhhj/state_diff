"""Train-only denoiser-isolation controls for Phase3.14b-r2.3.2.

This module diagnoses why the baseline MLP can memorize exact fixed tuples but
fails fresh random-noise reconstruction on one training row.  It never reads
validation targets or formal-test rows and never saves a model checkpoint.
"""
from __future__ import annotations

import hashlib
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ccda_phase3.phase314b_models import (
    MLPFutureDenoiser,
    SinusoidalTimeEmbedding,
)
from ccda_phase3.phase314b_r2_contract import REPAIR_CONFIGS
from ccda_phase3.phase314b_r2_diffusion import (
    active_mse,
    predict_original_sample,
    training_target,
)
from ccda_phase3.phase314b_r22_geometry import torch_inverse_standardize
from ccda_phase3.phase314b_r231_controls import (
    RANDOM_SINGLE_BRANCH_GATE,
    ReconstructionGate,
    gate_reconstruction,
    target_reconstruction_metrics,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM

PHASE = "phase3_14b_r232"
BASE_COMMIT = "5749624abea6dedbd33ef0731c93eab676aac3de"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R23_TINY_SHA256 = (
    "d370bef927af6dd0f3f5f94171482889dbadbbf5cf28c7f410567e5b25c886b2"
)
EXPECTED_R231_ROOT_CAUSE = (
    "phase314b_r231_random_noise_single_row_optimization_failure"
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r232_controls.py",
    "scripts/phase3_14b_r232_preflight.py",
    "scripts/phase3_14b_r232_run_controls.py",
    "scripts/phase3_14b_r232_finalize.py",
    "scripts/phase3_14b_r232_run.sh",
    "tests/test_phase314b_r232_controls.py",
)

CONTROL_TIMESTEPS = (10, 25, 50, 75, 90, 99)
FULL_TIMESTEPS = tuple(range(100))
CARTESIAN_TIMESTEPS = (10, 30, 50, 70, 90, 99)


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
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
class ModelSpec:
    kind: str
    hidden_dim: int = 512
    time_dim: int = 128

    def validate(self) -> None:
        if self.kind not in {"baseline_mlp", "residual_mlp", "time_affine"}:
            raise ValueError(f"unsupported model kind: {self.kind}")
        if self.hidden_dim <= 0 or self.time_dim <= 0:
            raise ValueError("model dimensions must be positive")


@dataclass
class TupleBank:
    condition_z: torch.Tensor
    clean_z: torch.Tensor
    clean_raw: torch.Tensor
    noisy: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    noise_ids: torch.Tensor

    def validate(self) -> None:
        count = int(self.noisy.shape[0])
        expected_future = (DEFAULT_TF, STATE_DIM)
        for name in ("clean_z", "clean_raw", "noisy", "noise"):
            value = getattr(self, name)
            if value.shape != (count,) + expected_future:
                raise ValueError(f"{name} has invalid shape: {value.shape}")
        if self.condition_z.ndim != 2 or self.condition_z.shape[0] != count:
            raise ValueError("condition_z has invalid shape")
        if self.timesteps.shape != (count,):
            raise ValueError("timesteps has invalid shape")
        if self.noise_ids.shape != (count,):
            raise ValueError("noise_ids has invalid shape")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                self.condition_z,
                self.clean_z,
                self.clean_raw,
                self.noisy,
                self.noise,
            )
        ):
            raise ValueError("tuple bank contains non-finite tensors")

    @property
    def row_count(self) -> int:
        return int(self.noisy.shape[0])


class ResidualMLPFutureDenoiser(nn.Module):
    """The baseline MLP with an explicit identity path from x_t to v."""

    family = "residual_mlp_diagnostic"

    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.residual = MLPFutureDenoiser(
            condition_dim=int(condition_dim),
            hidden_dim=int(hidden_dim),
            time_dim=int(time_dim),
        )

    def forward(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
    ) -> torch.Tensor:
        return noisy_future_z + self.residual(
            noisy_future_z,
            timestep,
            condition_z,
        )


class TimeAffineFutureDenoiser(nn.Module):
    """Diagnostic architecture with an explicit time-gated x_t pathway.

    For a fixed clean target x0, the exact v target is affine in x_t for every
    timestep.  This architecture therefore tests whether the baseline failure
    is caused by having to learn a high-dimensional identity/gain pathway
    implicitly through dense hidden layers.
    """

    family = "time_affine_diagnostic"

    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time = SinusoidalTimeEmbedding(int(time_dim))
        self.log_gain = nn.Sequential(
            nn.Linear(int(time_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.bias = nn.Sequential(
            nn.Linear(int(condition_dim) + int(time_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), DEFAULT_TF * STATE_DIM),
        )
        nn.init.zeros_(self.log_gain[-1].weight)
        nn.init.zeros_(self.log_gain[-1].bias)

    def forward(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_future_z.ndim != 3:
            raise ValueError("future sample must be [B,T,D]")
        batch = int(noisy_future_z.shape[0])
        time_z = self.time(timestep)
        gain = torch.exp(
            torch.clamp(self.log_gain(time_z), min=-6.0, max=6.0)
        ).reshape(batch, 1, 1)
        bias = self.bias(torch.cat([condition_z, time_z], dim=-1)).reshape(
            batch,
            DEFAULT_TF,
            STATE_DIM,
        )
        return gain * noisy_future_z + bias


def build_model(condition_dim: int, spec: ModelSpec) -> nn.Module:
    spec.validate()
    if spec.kind == "baseline_mlp":
        return MLPFutureDenoiser(
            condition_dim=int(condition_dim),
            hidden_dim=int(spec.hidden_dim),
            time_dim=int(spec.time_dim),
        )
    if spec.kind == "residual_mlp":
        return ResidualMLPFutureDenoiser(
            condition_dim=int(condition_dim),
            hidden_dim=int(spec.hidden_dim),
            time_dim=int(spec.time_dim),
        )
    return TimeAffineFutureDenoiser(
        condition_dim=int(condition_dim),
        hidden_dim=int(spec.hidden_dim),
        time_dim=int(spec.time_dim),
    )


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return subprocess.check_output(
        ["git", *args], cwd=Path(root), text=True
    ).strip()


def assert_only_allowed_worktree_paths(
    root: Path,
    allowed_paths: Sequence[str],
) -> None:
    allowed = {str(Path(path).as_posix()) for path in allowed_paths}
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
        raise RuntimeError(
            "unexpected worktree changes: " + ", ".join(unexpected)
        )


def _active_noise(
    shape: Sequence[int],
    active_mask: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        tuple(int(value) for value in shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    active = active_mask.to(device=device, dtype=torch.bool)
    return torch.where(active[None], noise, torch.zeros_like(noise))


def make_tuple_bank(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    timesteps: Sequence[int],
    noise_seeds: Sequence[int],
    mode: str,
) -> TupleBank:
    """Create a deterministic one-target tuple bank.

    Modes:
      fixed_timestep: one timestep, one tuple per noise seed;
      fixed_noise: one noise seed, one tuple per timestep;
      cartesian: every timestep/noise-seed combination.
    """
    if condition_z.shape[0] != 1 or clean_z.shape[0] != 1:
        raise ValueError("tuple-bank controls require exactly one source row")
    if clean_raw.shape[0] != 1:
        raise ValueError("clean_raw must contain exactly one source row")
    if mode not in {"fixed_timestep", "fixed_noise", "cartesian"}:
        raise ValueError(f"unsupported tuple-bank mode: {mode}")
    timestep_values = tuple(int(value) for value in timesteps)
    seed_values = tuple(int(value) for value in noise_seeds)
    if not timestep_values or not seed_values:
        raise ValueError("timesteps and noise_seeds must be nonempty")
    if any(value < 0 or value >= 100 for value in timestep_values):
        raise ValueError("timestep out of range")
    if mode == "fixed_timestep" and len(timestep_values) != 1:
        raise ValueError("fixed_timestep mode requires one timestep")
    if mode == "fixed_noise" and len(seed_values) != 1:
        raise ValueError("fixed_noise mode requires one noise seed")

    pairs: Iterable[Tuple[int, int]]
    if mode == "fixed_timestep":
        pairs = ((timestep_values[0], seed) for seed in seed_values)
    elif mode == "fixed_noise":
        pairs = ((timestep, seed_values[0]) for timestep in timestep_values)
    else:
        pairs = (
            (timestep, seed)
            for timestep in timestep_values
            for seed in seed_values
        )

    noisy_rows = []
    noise_rows = []
    timestep_rows = []
    noise_ids = []
    active = active_mask.to(device=clean_z.device, dtype=torch.bool)
    for timestep, noise_seed in pairs:
        noise = _active_noise(
            clean_z.shape,
            active,
            seed=noise_seed,
            device=clean_z.device,
            dtype=clean_z.dtype,
        )
        time_tensor = torch.full(
            (1,),
            int(timestep),
            device=clean_z.device,
            dtype=torch.long,
        )
        noisy = scheduler.add_noise(clean_z, noise, time_tensor)
        noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
        noisy_rows.append(noisy)
        noise_rows.append(noise)
        timestep_rows.append(int(timestep))
        noise_ids.append(int(noise_seed))

    count = len(noisy_rows)
    bank = TupleBank(
        condition_z=condition_z.repeat(count, 1),
        clean_z=clean_z.repeat(count, 1, 1),
        clean_raw=clean_raw.repeat(count, 1, 1),
        noisy=torch.cat(noisy_rows, dim=0),
        noise=torch.cat(noise_rows, dim=0),
        timesteps=torch.tensor(
            timestep_rows,
            device=clean_z.device,
            dtype=torch.long,
        ),
        noise_ids=torch.tensor(
            noise_ids,
            device=clean_z.device,
            dtype=torch.long,
        ),
    )
    bank.validate()
    return bank


def oracle_v_from_x0(
    *,
    scheduler,
    noisy: torch.Tensor,
    clean_z: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = scheduler.alphas_cumprod.to(
        device=noisy.device,
        dtype=noisy.dtype,
    )[timesteps]
    alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
    sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1.0e-12)).reshape(
        -1, 1, 1
    )
    return (alpha * noisy - clean_z) / sigma


def oracle_parity(
    *,
    scheduler,
    bank: TupleBank,
    active_mask: torch.Tensor,
) -> Dict[str, float | bool]:
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    target = training_target(
        scheduler=scheduler,
        config=repair,
        clean_sample=bank.clean_z,
        noise=bank.noise,
        timesteps=bank.timesteps,
    )
    oracle = oracle_v_from_x0(
        scheduler=scheduler,
        noisy=bank.noisy,
        clean_z=bank.clean_z,
        timesteps=bank.timesteps,
    )
    active = active_mask.to(device=bank.noisy.device, dtype=torch.bool)
    target = torch.where(active[None], target, torch.zeros_like(target))
    oracle = torch.where(active[None], oracle, torch.zeros_like(oracle))
    reconstructed = predict_original_sample(
        scheduler=scheduler,
        config=repair,
        sample=bank.noisy,
        model_output=oracle,
        timesteps=bank.timesteps,
    )
    reconstructed = torch.where(
        active[None], reconstructed, torch.zeros_like(reconstructed)
    )
    target_max_abs = float(torch.max(torch.abs(target - oracle)).cpu())
    x0_max_abs = float(
        torch.max(torch.abs(reconstructed - bank.clean_z)).cpu()
    )
    return {
        "v_target_max_abs": target_max_abs,
        "x0_reconstruction_max_abs": x0_max_abs,
        "pass": bool(target_max_abs <= 2.0e-5 and x0_max_abs <= 2.0e-5),
    }


def timestep_embedding_audit(
    *,
    time_dim: int = 128,
    device: torch.device,
) -> Dict[str, Any]:
    module = SinusoidalTimeEmbedding(int(time_dim)).to(device)
    timesteps = torch.arange(100, device=device, dtype=torch.long)
    with torch.no_grad():
        embedding = module(timesteps).double()
    distances = torch.cdist(embedding, embedding)
    distances.fill_diagonal_(float("inf"))
    normalized = embedding / torch.clamp(
        torch.linalg.norm(embedding, dim=1, keepdim=True), min=1.0e-12
    )
    cosine = normalized @ normalized.T
    cosine.fill_diagonal_(float("-inf"))
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered).cpu())
    min_distance = float(torch.min(distances).cpu())
    max_cosine = float(torch.max(cosine).cpu())
    return {
        "time_dim": int(time_dim),
        "timestep_count": 100,
        "centered_rank": rank,
        "minimum_pairwise_l2": min_distance,
        "maximum_off_diagonal_cosine": max_cosine,
        "exact_duplicate": bool(min_distance == 0.0),
        "pass": bool(min_distance > 1.0e-6 and rank >= 20),
    }


def _make_optimizer(model: nn.Module, spec: TrainSpec):
    spec.validate()
    return torch.optim.Adam(
        model.parameters(),
        lr=float(spec.learning_rate),
        weight_decay=float(spec.weight_decay),
    )


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
                squared += float(torch.sum(parameter.grad.detach() ** 2).cpu())
        norm = math.sqrt(squared)
    else:
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_grad_norm)
            )
        )
    optimizer.step()
    return float(norm)


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be finite and nonempty")
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


@torch.no_grad()
def evaluate_bank(
    *,
    model: nn.Module,
    scheduler,
    bank: TupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    gate: ReconstructionGate = RANDOM_SINGLE_BRANCH_GATE,
) -> Dict[str, Any]:
    model.eval()
    active = active_mask.to(device=bank.noisy.device, dtype=torch.bool)
    output = model(bank.noisy, bank.timesteps, bank.condition_z)
    output = torch.where(active[None], output, torch.zeros_like(output))
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    target_v = training_target(
        scheduler=scheduler,
        config=repair,
        clean_sample=bank.clean_z,
        noise=bank.noise,
        timesteps=bank.timesteps,
    )
    v_mse = float(active_mse(output, target_v, active).cpu())
    predicted_z = predict_original_sample(
        scheduler=scheduler,
        config=repair,
        sample=bank.noisy,
        model_output=output,
        timesteps=bank.timesteps,
    )
    predicted_z = torch.where(
        active[None], predicted_z, torch.zeros_like(predicted_z)
    )
    predicted_raw = torch_inverse_standardize(
        predicted_z, future_mean, future_scale
    )
    aggregate = target_reconstruction_metrics(
        predicted_z=predicted_z,
        target_z=bank.clean_z,
        predicted_raw=predicted_raw,
        target_raw=bank.clean_raw,
        active_mask=active,
    )
    by_timestep: Dict[str, Any] = {}
    for timestep in sorted(set(bank.timesteps.detach().cpu().tolist())):
        index = torch.nonzero(bank.timesteps == int(timestep), as_tuple=False).reshape(-1)
        metrics = target_reconstruction_metrics(
            predicted_z=predicted_z.index_select(0, index),
            target_z=bank.clean_z.index_select(0, index),
            predicted_raw=predicted_raw.index_select(0, index),
            target_raw=bank.clean_raw.index_select(0, index),
            active_mask=active,
        )
        by_timestep[str(int(timestep))] = {
            "metrics": metrics,
            "gate_pass": gate_reconstruction(metrics, gate),
            "row_count": int(index.numel()),
        }
    return {
        "row_count": bank.row_count,
        "v_target_mse": v_mse,
        "metrics": aggregate,
        "gate": asdict(gate),
        "gate_pass": gate_reconstruction(aggregate, gate),
        "by_timestep": by_timestep,
    }


def noisy_input_jvp_audit(
    *,
    model: nn.Module,
    scheduler,
    bank: TupleBank,
    active_mask: torch.Tensor,
    sample_index: int = 0,
    direction_seed: int = 7001,
) -> Dict[str, float]:
    model.eval()
    index = int(sample_index)
    noisy = bank.noisy[index:index + 1].detach().requires_grad_(True)
    timestep = bank.timesteps[index:index + 1]
    condition = bank.condition_z[index:index + 1]
    active = active_mask.to(device=noisy.device, dtype=torch.bool)
    direction = _active_noise(
        noisy.shape,
        active,
        seed=direction_seed,
        device=noisy.device,
        dtype=noisy.dtype,
    )

    def function(value: torch.Tensor) -> torch.Tensor:
        output = model(value, timestep, condition)
        return torch.where(active[None], output, torch.zeros_like(output))

    _, jvp = torch.autograd.functional.jvp(
        function,
        noisy,
        direction,
        create_graph=False,
        strict=True,
    )
    alpha_bar = scheduler.alphas_cumprod.to(
        device=noisy.device,
        dtype=noisy.dtype,
    )[timestep]
    expected_gain = torch.sqrt(alpha_bar / torch.clamp(1.0 - alpha_bar, min=1.0e-12))
    expected = expected_gain.reshape(-1, 1, 1) * direction
    observed_flat = jvp[:, active].reshape(-1)
    expected_flat = expected[:, active].reshape(-1)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            observed_flat[None], expected_flat[None], dim=1
        ).cpu()
    )
    relative_rmse = float(
        (
            torch.sqrt(torch.mean((observed_flat - expected_flat) ** 2))
            / torch.clamp(torch.sqrt(torch.mean(expected_flat ** 2)), min=1.0e-12)
        ).cpu()
    )
    observed_gain = float(
        (torch.dot(observed_flat, direction[:, active].reshape(-1))
         / torch.clamp(
             torch.dot(direction[:, active].reshape(-1), direction[:, active].reshape(-1)),
             min=1.0e-12,
         )).cpu()
    )
    return {
        "timestep": int(timestep.item()),
        "expected_scalar_gain": float(expected_gain.item()),
        "projected_observed_gain": observed_gain,
        "jvp_expected_cosine": cosine,
        "jvp_relative_rmse": relative_rmse,
    }


def train_bank_control(
    *,
    name: str,
    scheduler,
    train_bank: TupleBank,
    evaluation_banks: Mapping[str, TupleBank],
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    model_spec: ModelSpec,
    train_spec: TrainSpec,
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    seed_all(seed)
    model = build_model(train_bank.condition_z.shape[1], model_spec).to(
        train_bank.noisy.device
    )
    optimizer = _make_optimizer(model, train_spec)
    active = active_mask.to(device=train_bank.noisy.device, dtype=torch.bool)
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    generator = torch.Generator(device=train_bank.noisy.device).manual_seed(
        int(seed) + 1000
    )
    history = []
    model.train()
    for step in range(1, int(train_spec.steps) + 1):
        indices = torch.randint(
            0,
            train_bank.row_count,
            (int(train_spec.batch_size),),
            generator=generator,
            device=train_bank.noisy.device,
            dtype=torch.long,
        )
        noisy = train_bank.noisy.index_select(0, indices)
        timesteps = train_bank.timesteps.index_select(0, indices)
        condition = train_bank.condition_z.index_select(0, indices)
        clean_z = train_bank.clean_z.index_select(0, indices)
        noise = train_bank.noise.index_select(0, indices)
        output = model(noisy, timesteps, condition)
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
            clip_grad_norm=train_spec.clip_grad_norm,
        )
        if step == 1 or step % int(log_interval) == 0 or step == train_spec.steps:
            history.append(
                {
                    "step": int(step),
                    "sampled_v_loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )

    evaluations = {
        key: evaluate_bank(
            model=model,
            scheduler=scheduler,
            bank=value,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        for key, value in evaluation_banks.items()
    }
    jvp = noisy_input_jvp_audit(
        model=model,
        scheduler=scheduler,
        bank=next(iter(evaluation_banks.values())),
        active_mask=active,
        direction_seed=int(seed) + 9000,
    )
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "control_kind": "finite_tuple_bank",
        "model": asdict(model_spec),
        "train": asdict(train_spec),
        "parameter_count": parameter_count,
        "train_bank_rows": train_bank.row_count,
        "history": history,
        "evaluations": evaluations,
        "jvp": jvp,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }


def train_random_stream_control(
    *,
    name: str,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    heldout_bank: TupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    model_spec: ModelSpec,
    train_spec: TrainSpec,
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    if condition_z.shape[0] != 1 or clean_z.shape[0] != 1:
        raise ValueError("random-stream control requires one source row")
    seed_all(seed)
    model = build_model(condition_z.shape[1], model_spec).to(condition_z.device)
    optimizer = _make_optimizer(model, train_spec)
    active = active_mask.to(device=condition_z.device, dtype=torch.bool)
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    generator = torch.Generator(device=condition_z.device).manual_seed(
        int(seed) + 1000
    )
    history = []
    model.train()
    batch_size = int(train_spec.batch_size)
    condition_batch = condition_z.repeat(batch_size, 1)
    clean_batch = clean_z.repeat(batch_size, 1, 1)
    for step in range(1, int(train_spec.steps) + 1):
        timesteps = torch.randint(
            0,
            100,
            (batch_size,),
            generator=generator,
            device=condition_z.device,
            dtype=torch.long,
        )
        noise = torch.randn(
            clean_batch.shape,
            generator=generator,
            device=condition_z.device,
            dtype=clean_z.dtype,
        )
        noise = torch.where(active[None], noise, torch.zeros_like(noise))
        noisy = scheduler.add_noise(clean_batch, noise, timesteps)
        noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
        output = model(noisy, timesteps, condition_batch)
        output = torch.where(active[None], output, torch.zeros_like(output))
        target = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_batch,
            noise=noise,
            timesteps=timesteps,
        )
        loss = active_mse(output, target, active)
        gradient_norm = _gradient_step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            clip_grad_norm=train_spec.clip_grad_norm,
        )
        if step == 1 or step % int(log_interval) == 0 or step == train_spec.steps:
            history.append(
                {
                    "step": int(step),
                    "batch_v_loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )

    heldout = evaluate_bank(
        model=model,
        scheduler=scheduler,
        bank=heldout_bank,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    jvp = noisy_input_jvp_audit(
        model=model,
        scheduler=scheduler,
        bank=heldout_bank,
        active_mask=active,
        direction_seed=int(seed) + 9000,
    )
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "control_kind": "random_stream_minibatch",
        "model": asdict(model_spec),
        "train": asdict(train_spec),
        "parameter_count": parameter_count,
        "history": history,
        "heldout": heldout,
        "jvp": jvp,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }


def _evaluation_pass(run: Mapping[str, Any], key: str) -> bool:
    return bool(run["evaluations"][key]["gate_pass"])


def classify_controls(report: Mapping[str, Any]) -> Dict[str, str]:
    if not bool(report["oracle_parity"]["pass"]):
        return {
            "root_cause": "phase314b_r232_v_target_or_x0_oracle_parity_failed",
            "next_stage": "repair scheduler/target algebra before any training",
        }
    if not bool(report["timestep_embedding_audit"]["pass"]):
        return {
            "root_cause": "phase314b_r232_timestep_embedding_collision_supported",
            "next_stage": "replace timestep embedding and repeat train-only controls",
        }

    runs = report["runs"]
    fixed_standard_seen = _evaluation_pass(
        runs["fixed_t50_baseline"], "seen_noise_bank"
    )
    fixed_residual_seen = _evaluation_pass(
        runs["fixed_t50_residual"], "seen_noise_bank"
    )
    fixed_affine_seen = _evaluation_pass(
        runs["fixed_t50_time_affine"], "seen_noise_bank"
    )
    fixed_standard_fresh = _evaluation_pass(
        runs["fixed_t50_baseline"], "heldout_noise_bank"
    )
    fixed_residual_fresh = _evaluation_pass(
        runs["fixed_t50_residual"], "heldout_noise_bank"
    )
    fixed_affine_fresh = _evaluation_pass(
        runs["fixed_t50_time_affine"], "heldout_noise_bank"
    )

    if not fixed_standard_seen and (fixed_residual_seen or fixed_affine_seen):
        return {
            "root_cause": "phase314b_r232_noisy_input_skip_path_deficiency_supported",
            "next_stage": (
                "design a train-only residual/noisy-skip denoiser pilot; "
                "formal validation remains blocked"
            ),
        }
    if not fixed_standard_seen:
        return {
            "root_cause": "phase314b_r232_finite_noise_bank_optimization_failed",
            "next_stage": "inspect finite-bank optimizer convergence on one row",
        }
    if not fixed_standard_fresh and (fixed_residual_fresh or fixed_affine_fresh):
        return {
            "root_cause": "phase314b_r232_noisy_input_skip_path_deficiency_supported",
            "next_stage": (
                "design a train-only residual/noisy-skip denoiser pilot; "
                "formal validation remains blocked"
            ),
        }
    if not fixed_standard_fresh and not fixed_residual_fresh and not fixed_affine_fresh:
        return {
            "root_cause": "phase314b_r232_fixed_timestep_noise_generalization_failed",
            "next_stage": "increase finite noise-bank coverage and inspect target scaling",
        }

    multi_standard = _evaluation_pass(
        runs["fixed_noise_all_t_baseline"], "seen_timestep_bank"
    )
    multi_residual = _evaluation_pass(
        runs["fixed_noise_all_t_residual"], "seen_timestep_bank"
    )
    if not multi_standard and multi_residual:
        return {
            "root_cause": "phase314b_r232_timestep_conditioning_residual_path_supported",
            "next_stage": "retain explicit x_t residual path for the next train-only pilot",
        }
    if not multi_standard and not multi_residual:
        return {
            "root_cause": "phase314b_r232_multitimestep_optimization_failed",
            "next_stage": "debug timestep-conditioned optimization on a fixed noise vector",
        }

    cart_standard = _evaluation_pass(
        runs["cartesian_baseline"], "heldout_cartesian"
    )
    cart_residual = _evaluation_pass(
        runs["cartesian_residual"], "heldout_cartesian"
    )
    if not cart_standard and cart_residual:
        return {
            "root_cause": "phase314b_r232_cartesian_noisy_skip_deficiency_supported",
            "next_stage": "run a train-only residual-denoiser objective pilot",
        }

    stream_standard = bool(runs["stream_baseline"]["heldout"]["gate_pass"])
    stream_residual = bool(runs["stream_residual"]["heldout"]["gate_pass"])
    if stream_residual and not stream_standard:
        return {
            "root_cause": "phase314b_r232_random_stream_noisy_skip_deficiency_supported",
            "next_stage": "run a train-only residual-denoiser pilot; no formal validation",
        }
    if stream_standard and stream_residual:
        return {
            "root_cause": "phase314b_r232_r231_batch1_coverage_failure_supported",
            "next_stage": (
                "repeat ordered-geometry repair as a train-only minibatch pilot "
                "with no candidate selection"
            ),
        }
    if cart_standard or cart_residual:
        return {
            "root_cause": "phase314b_r232_online_random_stream_optimization_failure_supported",
            "next_stage": "replace online batch-1 noise sampling with controlled bank/minibatch training",
        }
    return {
        "root_cause": "phase314b_r232_noise_timestep_coverage_failure_unresolved",
        "next_stage": "expand train-only finite-bank controls before model repair",
    }
