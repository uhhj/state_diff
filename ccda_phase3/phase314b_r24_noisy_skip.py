"""Train-only timestep-conditioned noisy-skip pilot for Phase3.14b-r2.4.

This module is additive.  It does not read validation targets or formal-test
rows, does not save checkpoints, and does not modify the immutable cache,
contract, or DeformableRavens submodule.
"""
from __future__ import annotations

import hashlib
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ccda_phase3.phase314b_models import SinusoidalTimeEmbedding
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
from ccda_phase3.phase314b_r232_controls import (
    TimeAffineFutureDenoiser,
    TupleBank,
    noisy_input_jvp_audit,
    seed_all,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM

PHASE = "phase3_14b_r24"
BASE_COMMIT = "2c525b315426e5b7b05c87f4f79307de890d3837"
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
EXPECTED_R232_ROOT_CAUSE = (
    "phase314b_r232_noisy_input_skip_path_deficiency_supported"
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r24_noisy_skip.py",
    "scripts/phase3_14b_r24_preflight.py",
    "scripts/phase3_14b_r24_run_pilot.py",
    "scripts/phase3_14b_r24_finalize.py",
    "scripts/phase3_14b_r24_run.sh",
    "tests/test_phase314b_r24_noisy_skip.py",
)

DEPENDENCY_SHA256 = {
    "ccda_phase3/phase314a_contract.py":
        "0bcc573005455e1ad31eb1c031112b556c3181bfc0b4f07169129975d3e351b1",
    "ccda_phase3/phase314b_contract.py":
        "6174809a04ee581403ccd26aeb42ab49168cf662504d2318a37c111198dec519",
    "ccda_phase3/phase314b_models.py":
        "db8ff101114934854f6e9cf32aeab6f7ae0063949fc95d00a674544cde9041f6",
    "ccda_phase3/phase314b_r2_contract.py":
        "aa92c4a851c3183e5aa4758169bd1b83412ef6e60c67541cce9deb1274b33283",
    "ccda_phase3/phase314b_r2_diffusion.py":
        "417af4ebc17517ec0f4c403e45fcc69b01ffe76f8d85f3cf090479d73a5ea3ba",
    "ccda_phase3/phase314b_r22_geometry.py":
        "e2d3fcb5630677910642561a9f604623ca6c0ea3663943cd19becfc483919371",
    "ccda_phase3/phase314b_r23_diagnostics.py":
        "a56f7914c2b3b8cdbea7d11c1b6b3db906abf1bba862dac5ca627e27fa63ba68",
    "ccda_phase3/phase314b_r231_controls.py":
        "c16bfbfa3ae0b7b27c25ee79cf298c76c3d5511a545af228a898869d96098d35",
    "ccda_phase3/phase314b_r232_controls.py":
        "7d661167115249c77e52b6e8646cb05d2b28f8ce334f01f775078d7fd84dd27c",
    "ccda_phase3/schema_v2.py":
        "e1594460477e4b96e3899509d4b145d36691c4a26818ad938b530372a2588c2b",
}

EVAL_TIMESTEPS = (10, 25, 50, 75, 90, 99)
LOW_MID_TIMESTEPS = (10, 25, 50)
HIGH_TIMESTEPS = (75, 90, 99)
JVP_TIMESTEPS = (10, 50, 90)
EXPECTED_CONDITIONS = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)


@dataclass(frozen=True)
class PilotModelSpec:
    name: str
    kind: str
    hidden_dim: int = 512
    time_dim: int = 128
    use_residual: bool = False
    x0_aux_weight: float = 0.0
    base_x0_weight: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("model name is required")
        if self.kind not in {
            "learned_time_affine",
            "analytic_x0_skip",
        }:
            raise ValueError(f"unsupported model kind: {self.kind}")
        if self.kind == "learned_time_affine" and self.use_residual:
            raise ValueError("learned_time_affine does not use analytic residual")
        if self.hidden_dim <= 0 or self.time_dim <= 0:
            raise ValueError("model dimensions must be positive")
        for value in (self.x0_aux_weight, self.base_x0_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("loss weights must be finite and nonnegative")


@dataclass(frozen=True)
class PilotTrainSpec:
    warmup_steps: int
    diffusion_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if self.diffusion_steps <= 0 or self.batch_size <= 0:
            raise ValueError("diffusion_steps and batch_size must be positive")
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


PILOT_MODELS = (
    PilotModelSpec(
        name="learned_time_affine_v",
        kind="learned_time_affine",
    ),
    PilotModelSpec(
        name="analytic_x0_skip_v",
        kind="analytic_x0_skip",
        use_residual=False,
    ),
    PilotModelSpec(
        name="analytic_x0_skip_residual_v",
        kind="analytic_x0_skip",
        use_residual=True,
    ),
    PilotModelSpec(
        name="analytic_x0_skip_residual_v_x0",
        kind="analytic_x0_skip",
        use_residual=True,
        x0_aux_weight=0.10,
    ),
)
for _model_spec in PILOT_MODELS:
    _model_spec.validate()


class AnalyticX0SkipDenoiser(nn.Module):
    """v-prediction denoiser with an analytic noisy-input transport path.

    For alpha=sqrt(alpha_bar), sigma=sqrt(1-alpha_bar):

        v = (alpha * x_t - x0) / sigma.

    The network predicts a stable-scale standardized x0 prior from condition,
    while the analytic path transports x_t with the exact coefficient.  An
    optional zero-initialized residual-v network supplies x_t-dependent
    corrections needed for multimodal or imperfectly conditioned futures.
    """

    family = "analytic_x0_skip_mlp"

    def __init__(
        self,
        condition_dim: int,
        alpha_bar: torch.Tensor,
        hidden_dim: int = 512,
        time_dim: int = 128,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        alpha_bar_value = torch.as_tensor(alpha_bar, dtype=torch.float32)
        if alpha_bar_value.ndim != 1 or alpha_bar_value.numel() != 100:
            raise ValueError("alpha_bar must contain 100 timesteps")
        if not bool(torch.isfinite(alpha_bar_value).all()):
            raise ValueError("alpha_bar contains non-finite values")
        if bool(torch.any(alpha_bar_value <= 0)) or bool(
            torch.any(alpha_bar_value >= 1)
        ):
            raise ValueError("alpha_bar values must lie in (0,1)")

        alpha = torch.sqrt(alpha_bar_value)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar_value, min=1.0e-12))
        self.register_buffer("alpha_table", alpha)
        self.register_buffer("sigma_table", sigma)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.use_residual = bool(use_residual)
        self.time = SinusoidalTimeEmbedding(self.time_dim)

        flat_future = DEFAULT_TF * STATE_DIM
        self.x0_prior = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, flat_future),
        )

        if self.use_residual:
            self.residual_v = nn.Sequential(
                nn.Linear(
                    self.condition_dim + flat_future + self.time_dim,
                    self.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, flat_future),
            )
            nn.init.zeros_(self.residual_v[-1].weight)
            nn.init.zeros_(self.residual_v[-1].bias)
        else:
            self.residual_v = None

    def predict_base_x0(self, condition_z: torch.Tensor) -> torch.Tensor:
        if condition_z.ndim != 2 or condition_z.shape[1] != self.condition_dim:
            raise ValueError("condition_z has invalid shape")
        batch = int(condition_z.shape[0])
        return self.x0_prior(condition_z).reshape(
            batch,
            DEFAULT_TF,
            STATE_DIM,
        )

    def coefficients(
        self,
        timestep: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        index = timestep.long().reshape(-1)
        if bool(torch.any(index < 0)) or bool(torch.any(index >= 100)):
            raise ValueError("timestep out of range")
        alpha = self.alpha_table[index].to(dtype=dtype).reshape(-1, 1, 1)
        sigma = self.sigma_table[index].to(dtype=dtype).reshape(-1, 1, 1)
        return alpha, sigma

    def forward(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_future_z.ndim != 3:
            raise ValueError("future sample must be [B,T,D]")
        if noisy_future_z.shape[1:] != (DEFAULT_TF, STATE_DIM):
            raise ValueError("future sample has invalid trailing shape")
        if condition_z.shape[0] != noisy_future_z.shape[0]:
            raise ValueError("condition batch mismatch")
        alpha, sigma = self.coefficients(timestep, noisy_future_z.dtype)
        base_x0 = self.predict_base_x0(condition_z)
        output = (alpha * noisy_future_z - base_x0) / sigma
        if self.residual_v is not None:
            batch = int(noisy_future_z.shape[0])
            joined = torch.cat(
                [
                    condition_z,
                    noisy_future_z.reshape(batch, -1),
                    self.time(timestep),
                ],
                dim=-1,
            )
            output = output + self.residual_v(joined).reshape(
                batch,
                DEFAULT_TF,
                STATE_DIM,
            )
        return output


class LearnedTimeAffineWrapper(TimeAffineFutureDenoiser):
    """Name-stable wrapper around the r2.3.2 diagnostic architecture."""

    family = "learned_time_affine_mlp"



def build_pilot_model(
    *,
    condition_dim: int,
    scheduler,
    spec: PilotModelSpec,
) -> nn.Module:
    spec.validate()
    if spec.kind == "learned_time_affine":
        return LearnedTimeAffineWrapper(
            condition_dim=int(condition_dim),
            hidden_dim=int(spec.hidden_dim),
            time_dim=int(spec.time_dim),
        )
    return AnalyticX0SkipDenoiser(
        condition_dim=int(condition_dim),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        hidden_dim=int(spec.hidden_dim),
        time_dim=int(spec.time_dim),
        use_residual=bool(spec.use_residual),
    )



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}



def dependency_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    observed = {path: sha256_file(base / path) for path in DEPENDENCY_SHA256}
    mismatched = {
        path: {"expected": DEPENDENCY_SHA256[path], "observed": value}
        for path, value in observed.items()
        if value != DEPENDENCY_SHA256[path]
    }
    if mismatched:
        raise RuntimeError(f"r2.4 dependency hash mismatch: {mismatched}")
    return observed


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
    value = torch.randn(
        tuple(int(item) for item in shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    active = active_mask.to(device=device, dtype=torch.bool)
    return torch.where(active[None], value, torch.zeros_like(value))



def make_multirow_bank(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    timesteps: Sequence[int],
    noise_seeds: Sequence[int],
) -> TupleBank:
    """Create a deterministic row x timestep x noise bank.

    The same noise seed produces the same noise tensor for every source row.
    This is intentional: free/hidden paired rows are compared under matched
    perturbations.
    """
    if condition_z.ndim != 2:
        raise ValueError("condition_z must be [N,C]")
    row_count = int(condition_z.shape[0])
    if row_count <= 0:
        raise ValueError("at least one source row is required")
    expected = (row_count, DEFAULT_TF, STATE_DIM)
    for name, value in (
        ("clean_z", clean_z),
        ("clean_raw", clean_raw),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} has invalid shape: {value.shape}")
    timestep_values = tuple(int(value) for value in timesteps)
    seed_values = tuple(int(value) for value in noise_seeds)
    if not timestep_values or not seed_values:
        raise ValueError("timesteps and noise_seeds must be nonempty")
    if any(value < 0 or value >= 100 for value in timestep_values):
        raise ValueError("timestep out of range")

    active = active_mask.to(device=clean_z.device, dtype=torch.bool)
    condition_rows = []
    clean_rows = []
    raw_rows = []
    noisy_rows = []
    noise_rows = []
    timestep_rows = []
    noise_ids = []
    for row in range(row_count):
        for timestep in timestep_values:
            for noise_seed in seed_values:
                noise = _active_noise(
                    (1, DEFAULT_TF, STATE_DIM),
                    active,
                    seed=int(noise_seed),
                    device=clean_z.device,
                    dtype=clean_z.dtype,
                )
                time_tensor = torch.full(
                    (1,),
                    int(timestep),
                    device=clean_z.device,
                    dtype=torch.long,
                )
                clean_one = clean_z[row:row + 1]
                noisy = scheduler.add_noise(clean_one, noise, time_tensor)
                noisy = torch.where(
                    active[None], noisy, torch.zeros_like(noisy)
                )
                condition_rows.append(condition_z[row:row + 1])
                clean_rows.append(clean_one)
                raw_rows.append(clean_raw[row:row + 1])
                noisy_rows.append(noisy)
                noise_rows.append(noise)
                timestep_rows.append(int(timestep))
                noise_ids.append(int(noise_seed))

    bank = TupleBank(
        condition_z=torch.cat(condition_rows, dim=0),
        clean_z=torch.cat(clean_rows, dim=0),
        clean_raw=torch.cat(raw_rows, dim=0),
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



def analytic_formula_oracle(
    *,
    scheduler,
    bank: TupleBank,
    active_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Verify the analytic x_t/x0 decomposition against Diffusers v target."""
    active = active_mask.to(device=bank.noisy.device, dtype=torch.bool)
    alpha_bar = scheduler.alphas_cumprod.to(
        device=bank.noisy.device,
        dtype=bank.noisy.dtype,
    )[bank.timesteps]
    alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
    sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1.0e-12)).reshape(
        -1, 1, 1
    )
    analytic_v = (alpha * bank.noisy - bank.clean_z) / sigma
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    target_v = training_target(
        scheduler=scheduler,
        config=repair,
        clean_sample=bank.clean_z,
        noise=bank.noise,
        timesteps=bank.timesteps,
    )
    analytic_v = torch.where(
        active[None], analytic_v, torch.zeros_like(analytic_v)
    )
    target_v = torch.where(
        active[None], target_v, torch.zeros_like(target_v)
    )
    reconstructed = predict_original_sample(
        scheduler=scheduler,
        config=repair,
        sample=bank.noisy,
        model_output=analytic_v,
        timesteps=bank.timesteps,
    )
    reconstructed = torch.where(
        active[None], reconstructed, torch.zeros_like(reconstructed)
    )
    v_max_abs = float(torch.max(torch.abs(analytic_v - target_v)).cpu())
    x0_max_abs = float(
        torch.max(torch.abs(reconstructed - bank.clean_z)).cpu()
    )
    gains = {}
    for timestep in JVP_TIMESTEPS:
        value = scheduler.alphas_cumprod[int(timestep)].to(
            device=bank.noisy.device, dtype=bank.noisy.dtype
        )
        gains[str(int(timestep))] = float(
            torch.sqrt(value / torch.clamp(1.0 - value, min=1.0e-12)).cpu()
        )
    return {
        "v_target_max_abs": v_max_abs,
        "x0_reconstruction_max_abs": x0_max_abs,
        "expected_noisy_input_gain": gains,
        "pass": bool(v_max_abs <= 2.0e-5 and x0_max_abs <= 2.0e-5),
    }


def _optimizer(model: nn.Module, spec: PilotTrainSpec):
    spec.validate()
    return torch.optim.AdamW(
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
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                total += float(torch.sum(parameter.grad.detach() ** 2).cpu())
        gradient_norm = math.sqrt(total)
    else:
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_grad_norm)
            )
        )
    if not math.isfinite(gradient_norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return gradient_norm



def _sample_source_batch(
    *,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        0,
        int(condition_z.shape[0]),
        (int(batch_size),),
        generator=generator,
        device=condition_z.device,
        dtype=torch.long,
    )
    return (
        condition_z.index_select(0, indices),
        clean_z.index_select(0, indices),
        clean_raw.index_select(0, indices),
    )


@torch.no_grad()
def evaluate_pilot_bank(
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
    by_timestep = {}
    for timestep in sorted(set(bank.timesteps.detach().cpu().tolist())):
        index = torch.nonzero(
            bank.timesteps == int(timestep), as_tuple=False
        ).reshape(-1)
        metrics = target_reconstruction_metrics(
            predicted_z=predicted_z.index_select(0, index),
            target_z=bank.clean_z.index_select(0, index),
            predicted_raw=predicted_raw.index_select(0, index),
            target_raw=bank.clean_raw.index_select(0, index),
            active_mask=active,
        )
        by_timestep[str(int(timestep))] = {
            "row_count": int(index.numel()),
            "metrics": metrics,
            "gate_pass": bool(gate_reconstruction(metrics, gate)),
        }
    return {
        "row_count": int(bank.row_count),
        "v_target_mse": v_mse,
        "metrics": aggregate,
        "gate": asdict(gate),
        "gate_pass": bool(gate_reconstruction(aggregate, gate)),
        "by_timestep": by_timestep,
    }



def jvp_sweep(
    *,
    model: nn.Module,
    scheduler,
    bank: TupleBank,
    active_mask: torch.Tensor,
) -> Dict[str, Any]:
    values = {}
    for timestep in JVP_TIMESTEPS:
        matching = torch.nonzero(
            bank.timesteps == int(timestep), as_tuple=False
        ).reshape(-1)
        if matching.numel() == 0:
            raise RuntimeError(f"JVP bank lacks timestep {timestep}")
        values[str(int(timestep))] = noisy_input_jvp_audit(
            model=model,
            scheduler=scheduler,
            bank=bank,
            active_mask=active_mask,
            sample_index=int(matching[0].item()),
            direction_seed=90000 + int(timestep),
        )
    middle = values["50"]
    return {
        "by_timestep": values,
        "t50_gate_pass": bool(
            float(middle["jvp_expected_cosine"]) >= 0.95
            and float(middle["jvp_relative_rmse"]) <= 0.20
        ),
    }



def train_online_pilot(
    *,
    name: str,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    evaluation_banks: Mapping[str, TupleBank],
    jvp_bank: TupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    model_spec: PilotModelSpec,
    train_spec: PilotTrainSpec,
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    if condition_z.ndim != 2 or condition_z.shape[0] <= 0:
        raise ValueError("source conditions must be nonempty [N,C]")
    if clean_z.shape != (
        condition_z.shape[0],
        DEFAULT_TF,
        STATE_DIM,
    ):
        raise ValueError("clean_z source shape mismatch")
    if clean_raw.shape != clean_z.shape:
        raise ValueError("clean_raw source shape mismatch")
    model_spec.validate()
    train_spec.validate()
    seed_all(int(seed))
    device = condition_z.device
    model = build_pilot_model(
        condition_dim=int(condition_z.shape[1]),
        scheduler=scheduler,
        spec=model_spec,
    ).to(device)
    optimizer = _optimizer(model, train_spec)
    active = active_mask.to(device=device, dtype=torch.bool)
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    generator = torch.Generator(device=device).manual_seed(int(seed) + 1000)
    history = []

    if train_spec.warmup_steps > 0:
        if not hasattr(model, "predict_base_x0"):
            raise ValueError("warmup requires predict_base_x0")
        model.train()
        for step in range(1, int(train_spec.warmup_steps) + 1):
            condition_batch, clean_batch, _ = _sample_source_batch(
                condition_z=condition_z,
                clean_z=clean_z,
                clean_raw=clean_raw,
                batch_size=int(train_spec.batch_size),
                generator=generator,
            )
            predicted_base = model.predict_base_x0(condition_batch)
            warmup_loss = active_mse(predicted_base, clean_batch, active)
            gradient_norm = _gradient_step(
                model=model,
                optimizer=optimizer,
                loss=warmup_loss,
                clip_grad_norm=train_spec.clip_grad_norm,
            )
            if (
                step == 1
                or step % int(log_interval) == 0
                or step == train_spec.warmup_steps
            ):
                history.append(
                    {
                        "phase": "x0_prior_warmup",
                        "step": int(step),
                        "loss": float(warmup_loss.detach().cpu()),
                        "gradient_norm_before_clip": float(gradient_norm),
                    }
                )

    model.train()
    for step in range(1, int(train_spec.diffusion_steps) + 1):
        condition_batch, clean_batch, _ = _sample_source_batch(
            condition_z=condition_z,
            clean_z=clean_z,
            clean_raw=clean_raw,
            batch_size=int(train_spec.batch_size),
            generator=generator,
        )
        batch_size = int(condition_batch.shape[0])
        timesteps = torch.randint(
            0,
            100,
            (batch_size,),
            generator=generator,
            device=device,
            dtype=torch.long,
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
        output = model(noisy, timesteps, condition_batch)
        output = torch.where(active[None], output, torch.zeros_like(output))
        target_v = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_batch,
            noise=noise,
            timesteps=timesteps,
        )
        v_loss = active_mse(output, target_v, active)
        predicted_x0 = predict_original_sample(
            scheduler=scheduler,
            config=repair,
            sample=noisy,
            model_output=output,
            timesteps=timesteps,
        )
        predicted_x0 = torch.where(
            active[None], predicted_x0, torch.zeros_like(predicted_x0)
        )
        x0_loss = active_mse(predicted_x0, clean_batch, active)
        base_loss = torch.zeros((), device=device, dtype=v_loss.dtype)
        if hasattr(model, "predict_base_x0"):
            base_loss = active_mse(
                model.predict_base_x0(condition_batch),
                clean_batch,
                active,
            )
        total = (
            v_loss
            + float(model_spec.x0_aux_weight) * x0_loss
            + float(model_spec.base_x0_weight) * base_loss
        )
        gradient_norm = _gradient_step(
            model=model,
            optimizer=optimizer,
            loss=total,
            clip_grad_norm=train_spec.clip_grad_norm,
        )
        if (
            step == 1
            or step % int(log_interval) == 0
            or step == train_spec.diffusion_steps
        ):
            history.append(
                {
                    "phase": "diffusion",
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "v_loss": float(v_loss.detach().cpu()),
                    "x0_loss": float(x0_loss.detach().cpu()),
                    "base_x0_loss": float(base_loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )

    evaluations = {
        key: evaluate_pilot_bank(
            model=model,
            scheduler=scheduler,
            bank=bank,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        for key, bank in evaluation_banks.items()
    }
    jvp = jvp_sweep(
        model=model,
        scheduler=scheduler,
        bank=jvp_bank,
        active_mask=active,
    )
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    result = {
        "name": str(name),
        "model": asdict(model_spec),
        "train": asdict(train_spec),
        "source_row_count": int(condition_z.shape[0]),
        "parameter_count": parameter_count,
        "history": history,
        "evaluations": evaluations,
        "jvp": jvp,
        "candidate_eligible": False,
        "checkpoint_saved": False,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result



def stage_pass(
    run: Mapping[str, Any],
    evaluation_key: str,
    *,
    require_jvp: bool = True,
) -> bool:
    evaluation = run["evaluations"][evaluation_key]
    return bool(
        evaluation["gate_pass"]
        and (
            (not require_jvp)
            or bool(run["jvp"]["t50_gate_pass"])
        )
    )



def select_train_only_recommendation(
    stage_runs: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Select a train-only architecture recommendation by a frozen priority.

    This is not a formal model selection.  Every recommended architecture must
    pass one-row fresh-noise, unique-free fresh-noise, and paired low/mid-noise
    target-relative gates.
    """
    priority = (
        "analytic_x0_skip_v",
        "analytic_x0_skip_residual_v",
        "analytic_x0_skip_residual_v_x0",
        "learned_time_affine_v",
    )
    one = stage_runs.get("one_row", {})
    unique = stage_runs.get("unique_free_16", {})
    paired = stage_runs.get("paired_16", {})
    for name in priority:
        if name not in one or name not in unique or name not in paired:
            continue
        if not stage_pass(one[name], "fresh_noise_all_t"):
            continue
        if not stage_pass(unique[name], "fresh_noise_all_t"):
            continue
        if not stage_pass(
            paired[name], "fresh_noise_low_mid", require_jvp=False
        ):
            continue
        return name
    return None



def classify_pilot(report: Mapping[str, Any]) -> Dict[str, Any]:
    oracle = report.get("analytic_formula_oracle", {"pass": True})
    if not bool(oracle.get("pass")):
        return {
            "root_cause": "phase314b_r24_analytic_skip_oracle_parity_failed",
            "next_stage": "repair analytic noisy-skip algebra before training",
            "train_only_recommendation": None,
        }
    stages = report["stages"]
    one = stages.get("one_row", {})
    unique = stages.get("unique_free_16", {})
    paired = stages.get("paired_16", {})

    one_pass = {
        name
        for name, run in one.items()
        if stage_pass(run, "fresh_noise_all_t")
    }
    if not one_pass:
        return {
            "root_cause": "phase314b_r24_random_stream_skip_pilot_failed",
            "next_stage": (
                "debug analytic-skip numerical conditioning on one train row"
            ),
            "train_only_recommendation": None,
        }

    unique_pass = {
        name
        for name, run in unique.items()
        if stage_pass(run, "fresh_noise_all_t")
    }
    if not unique_pass:
        return {
            "root_cause": "phase314b_r24_conditioned_multirow_generalization_failed",
            "next_stage": (
                "debug condition capacity and source-row batching on unique-free train rows"
            ),
            "train_only_recommendation": None,
        }

    paired_pass = {
        name
        for name, run in paired.items()
        if stage_pass(
            run, "fresh_noise_low_mid", require_jvp=False
        )
    }
    recommendation = select_train_only_recommendation(stages)
    if not paired_pass or recommendation is None:
        return {
            "root_cause": "phase314b_r24_paired_low_noise_branch_transport_failed",
            "next_stage": (
                "debug paired low-noise branch transport before geometry repair"
            ),
            "train_only_recommendation": None,
        }

    if recommendation.startswith("analytic_x0_skip"):
        root = "phase314b_r24_train_only_analytic_noisy_skip_supported"
    else:
        root = "phase314b_r24_train_only_learned_affine_skip_supported"
    return {
        "root_cause": root,
        "next_stage": (
            "run a separate train-only ordered-geometry pilot using the frozen "
            "recommended architecture; formal validation remains blocked"
        ),
        "train_only_recommendation": recommendation,
    }
