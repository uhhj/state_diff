"""Train-only frozen-prior and factorized-width pilot for Phase3.14b-r2.4.2.

This module is additive and diagnostic. It does not read validation targets or
formal-test rows, does not save checkpoints, and does not modify the immutable
cache, validity contract, dataset, or DeformableRavens submodule.
"""

from __future__ import annotations

import hashlib
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    DIRECT_REGRESSION_GATE,
    RANDOM_SINGLE_BRANCH_GATE,
    ReconstructionGate,
    gate_reconstruction,
    target_reconstruction_metrics,
)
from ccda_phase3.phase314b_r241_multirow import (
    LabeledTupleBank,
    SourceBatchSampler,
    build_labeled_bank,
    condition_ablation_banks,
    evaluate_denoiser_bank,
    sample_aligned_source_batch,
    seed_everything,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


PHASE = "phase3_14b_r242"
BASE_COMMIT = "17eb4c51fb97534d49a2211ae43f97d489418254"
R241_IMPLEMENTATION_COMMIT = "08e7bb57a5e43111d8aa257fe1a890c812fae27e"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R241_ROOT_CAUSE = (
    "phase314b_r241_condition_encoder_width_limit_supported"
)
EXPECTED_R241_MODULE_SHA256 = (
    "883a7eb8637294a9613911310cc89774c16ca1085d3edfe7570e797b9082422a"
)
EXPECTED_R24_MODULE_SHA256 = (
    "f8ccac7ef82297c12fe152a48532fcb5a938e12e0eb0faca6914a27a6e37d307"
)
EXPECTED_R23_TINY_SHA256 = (
    "d370bef927af6dd0f3f5f94171482889dbadbbf5cf28c7f410567e5b25c886b2"
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r242_frozen_prior.py",
    "scripts/phase3_14b_r242_preflight.py",
    "scripts/phase3_14b_r242_run_pilot.py",
    "scripts/phase3_14b_r242_finalize.py",
    "scripts/phase3_14b_r242_run.sh",
    "tests/test_phase314b_r242_frozen_prior.py",
)

EVAL_TIMESTEPS = (10, 25, 50, 75, 90, 99)
LOW_MID_TIMESTEPS = (10, 25, 50)
HIGH_TIMESTEPS = (75, 90, 99)
EXPECTED_PAIRED_CONDITIONS = (
    "free",
    "hidden_slack_breakaway_pin_v2",
)


@dataclass(frozen=True)
class PriorFitSpec:
    steps: int = 5000
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("prior fit steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("prior learning rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("prior weight decay must be nonnegative")


@dataclass(frozen=True)
class ResidualTrainSpec:
    steps: int = 8000
    batch_size: int = 64
    residual_learning_rate: float = 1.0e-3
    prior_learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("residual steps and batch size must be positive")
        for name, value in (
            ("residual_learning_rate", self.residual_learning_rate),
            ("prior_learning_rate", self.prior_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be nonnegative")
        if self.clip_grad_norm is not None:
            if (
                not math.isfinite(self.clip_grad_norm)
                or self.clip_grad_norm <= 0
            ):
                raise ValueError("clip_grad_norm must be positive")


@dataclass(frozen=True)
class FactorizedVariant:
    name: str
    prior_hidden_dim: int
    residual_hidden_dim: int
    prior_policy: str
    prior_anchor_weight: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("variant name is required")
        if self.prior_hidden_dim <= 0 or self.residual_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.prior_policy not in {"frozen", "decoupled", "joint"}:
            raise ValueError("unsupported prior policy")
        if (
            not math.isfinite(self.prior_anchor_weight)
            or self.prior_anchor_weight < 0
        ):
            raise ValueError("prior_anchor_weight must be nonnegative")
        if self.prior_policy == "frozen" and self.prior_anchor_weight != 0:
            raise ValueError("frozen prior cannot use an anchor")


FACTORIAL_VARIANTS = (
    FactorizedVariant(
        name="frozen_p512_r512",
        prior_hidden_dim=512,
        residual_hidden_dim=512,
        prior_policy="frozen",
    ),
    FactorizedVariant(
        name="frozen_p1024_r512",
        prior_hidden_dim=1024,
        residual_hidden_dim=512,
        prior_policy="frozen",
    ),
    FactorizedVariant(
        name="frozen_p512_r1024",
        prior_hidden_dim=512,
        residual_hidden_dim=1024,
        prior_policy="frozen",
    ),
    FactorizedVariant(
        name="frozen_p1024_r1024",
        prior_hidden_dim=1024,
        residual_hidden_dim=1024,
        prior_policy="frozen",
    ),
    FactorizedVariant(
        name="decoupled_p1024_r512",
        prior_hidden_dim=1024,
        residual_hidden_dim=512,
        prior_policy="decoupled",
    ),
    FactorizedVariant(
        name="joint_p1024_r512_control",
        prior_hidden_dim=1024,
        residual_hidden_dim=512,
        prior_policy="joint",
        prior_anchor_weight=1.0,
    ),
)
for _variant in FACTORIAL_VARIANTS:
    _variant.validate()


class FactorizedAnalyticX0SkipDenoiser(nn.Module):
    """Analytic v-prediction skip with independently sized prior and residual.

    The prior predicts standardized x0 from deployable conditioning. The
    residual predicts corrections to the analytic v mapping. Separating the
    two widths is necessary because the r2.4 implementation used one hidden
    width for both paths and therefore could not attribute a gain specifically
    to condition-prior capacity.
    """

    family = "factorized_analytic_x0_skip_mlp"

    def __init__(
        self,
        *,
        condition_dim: int,
        alpha_bar: torch.Tensor,
        prior_hidden_dim: int,
        residual_hidden_dim: int,
        time_dim: int = 128,
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
            raise ValueError("alpha_bar must lie in (0,1)")

        self.condition_dim = int(condition_dim)
        self.prior_hidden_dim = int(prior_hidden_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.time_dim = int(time_dim)

        alpha = torch.sqrt(alpha_bar_value)
        sigma = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_value, min=1.0e-12)
        )
        self.register_buffer("alpha_table", alpha)
        self.register_buffer("sigma_table", sigma)

        flat = DEFAULT_TF * STATE_DIM
        self.time = SinusoidalTimeEmbedding(self.time_dim)
        self.x0_prior = nn.Sequential(
            nn.Linear(self.condition_dim, self.prior_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.prior_hidden_dim, self.prior_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.prior_hidden_dim, flat),
        )
        self.residual_v = nn.Sequential(
            nn.Linear(
                self.condition_dim + flat + self.time_dim,
                self.residual_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.residual_hidden_dim,
                self.residual_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.residual_hidden_dim, flat),
        )
        nn.init.zeros_(self.residual_v[-1].weight)
        nn.init.zeros_(self.residual_v[-1].bias)

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

    def predict_base_x0(self, condition_z: torch.Tensor) -> torch.Tensor:
        if condition_z.ndim != 2:
            raise ValueError("condition_z must be [B,C]")
        if condition_z.shape[1] != self.condition_dim:
            raise ValueError("condition dimension mismatch")
        batch = int(condition_z.shape[0])
        return self.x0_prior(condition_z).reshape(
            batch,
            DEFAULT_TF,
            STATE_DIM,
        )

    def predict_residual_v(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_future_z.ndim != 3:
            raise ValueError("noisy future must be [B,T,D]")
        if noisy_future_z.shape[1:] != (DEFAULT_TF, STATE_DIM):
            raise ValueError("future trailing shape mismatch")
        if condition_z.shape[0] != noisy_future_z.shape[0]:
            raise ValueError("condition batch mismatch")
        batch = int(noisy_future_z.shape[0])
        joined = torch.cat(
            (
                condition_z,
                noisy_future_z.reshape(batch, -1),
                self.time(timestep),
            ),
            dim=-1,
        )
        return self.residual_v(joined).reshape(
            batch,
            DEFAULT_TF,
            STATE_DIM,
        )

    def forward(
        self,
        noisy_future_z: torch.Tensor,
        timestep: torch.Tensor,
        condition_z: torch.Tensor,
        *,
        detach_prior_for_v: bool = False,
    ) -> torch.Tensor:
        alpha, sigma = self.coefficients(
            timestep,
            noisy_future_z.dtype,
        )
        base_x0 = self.predict_base_x0(condition_z)
        if detach_prior_for_v:
            base_x0 = base_x0.detach()
        analytic = (alpha * noisy_future_z - base_x0) / sigma
        return analytic + self.predict_residual_v(
            noisy_future_z,
            timestep,
            condition_z,
        )

    def prior_parameters(self) -> List[nn.Parameter]:
        return list(self.x0_prior.parameters())

    def residual_parameters(self) -> List[nn.Parameter]:
        return list(self.residual_v.parameters())


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
        ["git", *args],
        cwd=Path(root),
        text=True,
    ).strip()


def assert_only_allowed_worktree_paths(
    root: Path,
    allowed_paths: Sequence[str],
) -> None:
    allowed = {str(Path(path).as_posix()) for path in allowed_paths}
    output = git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    observed = set()
    for line in output.splitlines():
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        observed.add(str(Path(value).as_posix()))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree changes: " + ", ".join(unexpected)
        )


def corrected_r241_interpretation(
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Correct the r2.4.1 classifier-precedence ambiguity without rewriting it.

    The historical classifier returned the width result before it inspected
    diffusion variants. When a frozen-prior diffusion variant passed and all
    joint-prior variants drifted, the diffusion evidence is stronger than a
    one-seed direct-regression miss by one source.
    """
    direct = summary["direct_controls"]
    diffusion = summary["diffusion_variants"]

    width_512 = bool(direct["rows_16_width_512"]["pass"])
    width_1024 = bool(direct["rows_16_width_1024"]["pass"])
    frozen = diffusion["balanced_frozen_prior"]
    joint_names = (
        "current_joint_random_reused_optimizer",
        "balanced_joint_fresh_optimizer",
        "balanced_anchor_prior",
        "random_anchor_prior",
    )
    joint_failed = all(
        not bool(diffusion[name]["pass"])
        for name in joint_names
    )
    severe_joint_drift = all(
        float(diffusion[name]["prior_drift_ratio"]) >= 100.0
        for name in joint_names
    )

    precedence_bug_supported = bool(
        (not width_512)
        and width_1024
        and bool(frozen["pass"])
        and joint_failed
        and severe_joint_drift
    )
    return {
        "historical_root_cause": summary.get("root_cause"),
        "width_512_direct_pass": width_512,
        "width_1024_direct_pass": width_1024,
        "balanced_frozen_prior_pass": bool(frozen["pass"]),
        "balanced_frozen_prior_drift_ratio": float(
            frozen["prior_drift_ratio"]
        ),
        "joint_variants_failed": joint_failed,
        "joint_variants_severe_drift": severe_joint_drift,
        "classifier_precedence_bug_supported": precedence_bug_supported,
        "corrected_primary_hypothesis": (
            "x0_prior_gradient_coupling_and_drift"
            if precedence_bug_supported
            else "not_determined"
        ),
        "secondary_hypothesis": (
            "single_seed_condition_width_sensitivity"
            if (not width_512 and width_1024)
            else None
        ),
    }


def _optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    values = [value for value in parameters if value.requires_grad]
    if not values:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(
        values,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _gradient_step(
    *,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    clip_grad_norm: Optional[float],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameter_list = [
        value
        for value in parameters
        if value.requires_grad and value.grad is not None
    ]
    if not parameter_list:
        raise RuntimeError("loss produced no gradients")
    if clip_grad_norm is None:
        squared = sum(
            float(torch.sum(value.grad.detach() ** 2).cpu())
            for value in parameter_list
        )
        norm = math.sqrt(squared)
    else:
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameter_list,
                float(clip_grad_norm),
            )
        )
    if not math.isfinite(norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()
    return norm


def _prediction_metrics(
    *,
    predicted_z: torch.Tensor,
    target_z: torch.Tensor,
    target_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    gate: ReconstructionGate,
) -> Dict[str, Any]:
    active = active_mask.to(
        device=predicted_z.device,
        dtype=torch.bool,
    )
    prediction = torch.where(
        active[None],
        predicted_z,
        torch.zeros_like(predicted_z),
    )
    predicted_raw = torch_inverse_standardize(
        prediction,
        future_mean,
        future_scale,
    )
    metrics = target_reconstruction_metrics(
        predicted_z=prediction,
        target_z=target_z,
        predicted_raw=predicted_raw,
        target_raw=target_raw,
        active_mask=active,
    )
    return {
        "metrics": metrics,
        "gate": asdict(gate),
        "gate_pass": bool(gate_reconstruction(metrics, gate)),
    }


def _by_source_metrics(
    *,
    predicted_z: torch.Tensor,
    bank: LabeledTupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    gate: ReconstructionGate,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    passes: List[bool] = []
    source_values = sorted(
        set(bank.source_ids.detach().cpu().tolist())
    )
    for source_id in source_values:
        index = torch.nonzero(
            bank.source_ids == int(source_id),
            as_tuple=False,
        ).reshape(-1)
        item = _prediction_metrics(
            predicted_z=predicted_z.index_select(0, index),
            target_z=bank.clean_z.index_select(0, index),
            target_raw=bank.clean_raw.index_select(0, index),
            active_mask=active_mask,
            future_mean=future_mean,
            future_scale=future_scale,
            gate=gate,
        )
        results[str(int(source_id))] = {
            "row_count": int(index.numel()),
            **item,
        }
        passes.append(bool(item["gate_pass"]))
    return {
        "by_source": results,
        "all_source_gate_pass": bool(all(passes)),
        "source_pass_fraction": (
            float(np.mean(np.asarray(passes, dtype=np.float64)))
            if passes
            else 0.0
        ),
    }


@torch.no_grad()
def evaluate_factorized_bank(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    scheduler,
    bank: LabeledTupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    gate: ReconstructionGate = RANDOM_SINGLE_BRANCH_GATE,
) -> Dict[str, Any]:
    model.eval()
    active = active_mask.to(device=bank.noisy.device, dtype=torch.bool)
    output = model(
        bank.noisy,
        bank.timesteps,
        bank.condition_z,
    )
    output = torch.where(
        active[None],
        output,
        torch.zeros_like(output),
    )
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    target_v = training_target(
        scheduler=scheduler,
        config=repair,
        clean_sample=bank.clean_z,
        noise=bank.noise,
        timesteps=bank.timesteps,
    )
    v_target_mse = float(active_mse(output, target_v, active).cpu())
    predicted_z = predict_original_sample(
        scheduler=scheduler,
        config=repair,
        sample=bank.noisy,
        model_output=output,
        timesteps=bank.timesteps,
    )
    predicted_z = torch.where(
        active[None],
        predicted_z,
        torch.zeros_like(predicted_z),
    )
    aggregate = _prediction_metrics(
        predicted_z=predicted_z,
        target_z=bank.clean_z,
        target_raw=bank.clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=gate,
    )
    by_source = _by_source_metrics(
        predicted_z=predicted_z,
        bank=bank,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=gate,
    )
    by_timestep: Dict[str, Any] = {}
    for timestep in sorted(
        set(bank.timesteps.detach().cpu().tolist())
    ):
        index = torch.nonzero(
            bank.timesteps == int(timestep),
            as_tuple=False,
        ).reshape(-1)
        item = _prediction_metrics(
            predicted_z=predicted_z.index_select(0, index),
            target_z=bank.clean_z.index_select(0, index),
            target_raw=bank.clean_raw.index_select(0, index),
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
            gate=gate,
        )
        by_timestep[str(int(timestep))] = {
            "row_count": int(index.numel()),
            **item,
        }
    return {
        "row_count": int(bank.row_count),
        "v_target_mse": v_target_mse,
        "aggregate": aggregate,
        **by_source,
        "by_timestep": by_timestep,
        "gate_pass": bool(
            aggregate["gate_pass"]
            and by_source["all_source_gate_pass"]
        ),
        "_predicted_z": predicted_z,
    }


def _strip_private_tensors(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("_")
    }


def _prior_metrics(
    *,
    model: FactorizedAnalyticX0SkipDenoiser,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
) -> Dict[str, Any]:
    with torch.no_grad():
        predicted_z = model.predict_base_x0(condition_z)
    aggregate = _prediction_metrics(
        predicted_z=predicted_z,
        target_z=clean_z,
        target_raw=clean_raw,
        active_mask=active_mask,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=DIRECT_REGRESSION_GATE,
    )
    source_ids = torch.arange(
        condition_z.shape[0],
        device=condition_z.device,
        dtype=torch.long,
    )
    bank = LabeledTupleBank(
        condition_z=condition_z,
        clean_z=clean_z,
        clean_raw=clean_raw,
        noisy=clean_z,
        noise=torch.zeros_like(clean_z),
        timesteps=torch.zeros_like(source_ids),
        noise_ids=torch.zeros_like(source_ids),
        source_ids=source_ids,
    )
    by_source = _by_source_metrics(
        predicted_z=predicted_z,
        bank=bank,
        active_mask=active_mask,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=DIRECT_REGRESSION_GATE,
    )
    return {
        **aggregate,
        **by_source,
        "pass": bool(
            aggregate["gate_pass"]
            and by_source["all_source_gate_pass"]
        ),
    }


def train_prior_seed_control(
    *,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    scheduler,
    hidden_dim: int,
    spec: PriorFitSpec,
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    spec.validate()
    seed_everything(seed)
    model = FactorizedAnalyticX0SkipDenoiser(
        condition_dim=int(condition_z.shape[1]),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        prior_hidden_dim=int(hidden_dim),
        residual_hidden_dim=512,
    ).to(condition_z.device)
    optimizer = _optimizer(
        model.prior_parameters(),
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    active = active_mask.to(
        device=condition_z.device,
        dtype=torch.bool,
    )
    history = []
    for step in range(1, int(spec.steps) + 1):
        prediction = model.predict_base_x0(condition_z)
        loss = active_mse(prediction, clean_z, active)
        norm = _gradient_step(
            optimizer=optimizer,
            loss=loss,
            parameters=model.prior_parameters(),
            clip_grad_norm=None,
        )
        if (
            step == 1
            or step % int(log_interval) == 0
            or step == int(spec.steps)
        ):
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(norm),
                }
            )
    result = _prior_metrics(
        model=model,
        condition_z=condition_z,
        clean_z=clean_z,
        clean_raw=clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    return {
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "spec": asdict(spec),
        "history": history,
        **result,
        "parameter_count": int(
            sum(value.numel() for value in model.x0_prior.parameters())
        ),
    }


def summarize_seed_stability(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    passes = [bool(run["pass"]) for run in runs]
    z_values = [
        float(run["aggregate"]["metrics"]["z_mse"])
        for run in runs
    ]
    ordered_values = [
        float(run["aggregate"]["metrics"]["ordered_rmse_p95"])
        for run in runs
    ]
    return {
        "seed_count": int(len(runs)),
        "pass_count": int(sum(passes)),
        "stable_2_of_3": bool(sum(passes) >= 2),
        "z_mse_median": float(np.median(z_values)),
        "z_mse_max": float(np.max(z_values)),
        "ordered_rmse_p95_median": float(np.median(ordered_values)),
        "ordered_rmse_p95_max": float(np.max(ordered_values)),
    }


def _condition_effect(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    true_value = float(
        evaluations["true"]["aggregate"]["metrics"]["z_mse"]
    )
    zero_value = float(
        evaluations["zero"]["aggregate"]["metrics"]["z_mse"]
    )
    permuted_value = float(
        evaluations["permuted"]["aggregate"]["metrics"]["z_mse"]
    )
    denominator = max(true_value, 1.0e-12)
    return {
        "true_z_mse": true_value,
        "zero_z_mse": zero_value,
        "permuted_z_mse": permuted_value,
        "zero_to_true_ratio": float(zero_value / denominator),
        "permuted_to_true_ratio": float(
            permuted_value / denominator
        ),
        "condition_effect_supported": bool(
            zero_value >= 1.5 * denominator
            and permuted_value >= 1.5 * denominator
        ),
    }


def _sample_timesteps(
    *,
    batch_size: int,
    timestep_values: Sequence[int],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    values = torch.tensor(
        [int(item) for item in timestep_values],
        device=device,
        dtype=torch.long,
    )
    if values.numel() == 0:
        raise ValueError("timestep_values must be nonempty")
    choice = torch.randint(
        0,
        int(values.numel()),
        (int(batch_size),),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    return values.index_select(0, choice)


def train_factorized_variant(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    evaluation_bank: LabeledTupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    variant: FactorizedVariant,
    prior_spec: PriorFitSpec,
    residual_spec: ResidualTrainSpec,
    train_timesteps: Sequence[int],
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    variant.validate()
    prior_spec.validate()
    residual_spec.validate()
    seed_everything(seed)
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    model = FactorizedAnalyticX0SkipDenoiser(
        condition_dim=int(condition_z.shape[1]),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        prior_hidden_dim=int(variant.prior_hidden_dim),
        residual_hidden_dim=int(variant.residual_hidden_dim),
    ).to(device)

    prior_optimizer = _optimizer(
        model.prior_parameters(),
        learning_rate=prior_spec.learning_rate,
        weight_decay=prior_spec.weight_decay,
    )
    warmup_history = []
    for step in range(1, int(prior_spec.steps) + 1):
        prediction = model.predict_base_x0(condition_z)
        loss = active_mse(prediction, clean_z, active)
        norm = _gradient_step(
            optimizer=prior_optimizer,
            loss=loss,
            parameters=model.prior_parameters(),
            clip_grad_norm=None,
        )
        if (
            step == 1
            or step % int(log_interval) == 0
            or step == int(prior_spec.steps)
        ):
            warmup_history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(norm),
                }
            )

    prior_after_warmup = _prior_metrics(
        model=model,
        condition_z=condition_z,
        clean_z=clean_z,
        clean_raw=clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
    )

    if variant.prior_policy == "frozen":
        for parameter in model.prior_parameters():
            parameter.requires_grad_(False)
        residual_optimizer = _optimizer(
            model.residual_parameters(),
            learning_rate=residual_spec.residual_learning_rate,
            weight_decay=residual_spec.weight_decay,
        )
        separate_prior_optimizer = None
        joint_optimizer = None
    elif variant.prior_policy == "decoupled":
        residual_optimizer = _optimizer(
            model.residual_parameters(),
            learning_rate=residual_spec.residual_learning_rate,
            weight_decay=residual_spec.weight_decay,
        )
        separate_prior_optimizer = _optimizer(
            model.prior_parameters(),
            learning_rate=residual_spec.prior_learning_rate,
            weight_decay=residual_spec.weight_decay,
        )
        joint_optimizer = None
    else:
        residual_optimizer = None
        separate_prior_optimizer = None
        joint_optimizer = _optimizer(
            model.parameters(),
            learning_rate=residual_spec.residual_learning_rate,
            weight_decay=residual_spec.weight_decay,
        )

    sampler = SourceBatchSampler(
        source_count=int(condition_z.shape[0]),
        batch_size=int(residual_spec.batch_size),
        mode="balanced",
        device=device,
        seed=int(seed) + 1000,
    )
    noise_generator = torch.Generator(device=device).manual_seed(
        int(seed) + 2000
    )
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    history = []

    for step in range(1, int(residual_spec.steps) + 1):
        (
            _,
            condition_batch,
            clean_batch,
            _,
        ) = sample_aligned_source_batch(
            condition_z=condition_z,
            clean_z=clean_z,
            clean_raw=clean_raw,
            sampler=sampler,
        )
        batch_size = int(condition_batch.shape[0])
        timestep = _sample_timesteps(
            batch_size=batch_size,
            timestep_values=train_timesteps,
            generator=noise_generator,
            device=device,
        )
        noise = torch.randn(
            clean_batch.shape,
            generator=noise_generator,
            device=device,
            dtype=clean_batch.dtype,
        )
        noise = torch.where(
            active[None],
            noise,
            torch.zeros_like(noise),
        )
        noisy = scheduler.add_noise(clean_batch, noise, timestep)
        noisy = torch.where(
            active[None],
            noisy,
            torch.zeros_like(noisy),
        )
        target_v = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_batch,
            noise=noise,
            timesteps=timestep,
        )

        prior_loss_value = 0.0
        if variant.prior_policy == "decoupled":
            assert separate_prior_optimizer is not None
            prior_prediction = model.predict_base_x0(condition_batch)
            prior_loss = active_mse(
                prior_prediction,
                clean_batch,
                active,
            )
            _gradient_step(
                optimizer=separate_prior_optimizer,
                loss=prior_loss,
                parameters=model.prior_parameters(),
                clip_grad_norm=residual_spec.clip_grad_norm,
            )
            prior_loss_value = float(prior_loss.detach().cpu())

        if variant.prior_policy in {"frozen", "decoupled"}:
            assert residual_optimizer is not None
            output = model(
                noisy,
                timestep,
                condition_batch,
                detach_prior_for_v=True,
            )
            output = torch.where(
                active[None],
                output,
                torch.zeros_like(output),
            )
            v_loss = active_mse(output, target_v, active)
            norm = _gradient_step(
                optimizer=residual_optimizer,
                loss=v_loss,
                parameters=model.residual_parameters(),
                clip_grad_norm=residual_spec.clip_grad_norm,
            )
            total_value = float(v_loss.detach().cpu())
        else:
            assert joint_optimizer is not None
            output = model(
                noisy,
                timestep,
                condition_batch,
                detach_prior_for_v=False,
            )
            output = torch.where(
                active[None],
                output,
                torch.zeros_like(output),
            )
            v_loss = active_mse(output, target_v, active)
            prior_loss = active_mse(
                model.predict_base_x0(condition_batch),
                clean_batch,
                active,
            )
            total = (
                v_loss
                + float(variant.prior_anchor_weight) * prior_loss
            )
            norm = _gradient_step(
                optimizer=joint_optimizer,
                loss=total,
                parameters=model.parameters(),
                clip_grad_norm=residual_spec.clip_grad_norm,
            )
            total_value = float(total.detach().cpu())
            prior_loss_value = float(prior_loss.detach().cpu())

        if (
            step == 1
            or step % int(log_interval) == 0
            or step == int(residual_spec.steps)
        ):
            history.append(
                {
                    "step": int(step),
                    "total_loss": total_value,
                    "v_loss": float(v_loss.detach().cpu()),
                    "prior_loss": prior_loss_value,
                    "gradient_norm": float(norm),
                }
            )

    prior_after_diffusion = _prior_metrics(
        model=model,
        condition_z=condition_z,
        clean_z=clean_z,
        clean_raw=clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    ablation_banks = condition_ablation_banks(
        evaluation_bank,
        seed=int(seed) + 3000,
    )
    evaluations_raw = {
        key: evaluate_factorized_bank(
            model=model,
            scheduler=scheduler,
            bank=value,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        for key, value in ablation_banks.items()
    }
    condition_effect = _condition_effect(evaluations_raw)

    warmup_z = float(
        prior_after_warmup["aggregate"]["metrics"]["z_mse"]
    )
    post_z = float(
        prior_after_diffusion["aggregate"]["metrics"]["z_mse"]
    )
    prior_drift_ratio = float(
        post_z / max(warmup_z, 1.0e-12)
    )
    true_evaluation = evaluations_raw["true"]
    result = {
        "variant": asdict(variant),
        "prior_fit": asdict(prior_spec),
        "residual_train": asdict(residual_spec),
        "train_timesteps": [int(value) for value in train_timesteps],
        "source_row_count": int(condition_z.shape[0]),
        "warmup_history": warmup_history,
        "diffusion_history": history,
        "source_exposure": sampler.count_report(),
        "prior_after_warmup": prior_after_warmup,
        "prior_after_diffusion": prior_after_diffusion,
        "prior_drift_ratio": prior_drift_ratio,
        "evaluations": {
            key: _strip_private_tensors(value)
            for key, value in evaluations_raw.items()
        },
        "condition_effect": condition_effect,
        "pass": bool(
            true_evaluation["gate_pass"]
            and condition_effect["condition_effect_supported"]
            and (
                variant.prior_policy == "joint"
                or prior_drift_ratio <= 2.0
            )
        ),
        "parameter_count": int(
            sum(value.numel() for value in model.parameters())
        ),
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "_model": model,
        "_true_prediction_z": true_evaluation["_predicted_z"],
    }
    return result


def strip_runtime_objects(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
    }


def paired_branch_audit(
    *,
    predicted_z: torch.Tensor,
    bank: LabeledTupleBank,
    source_pair_ids: Sequence[int],
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
) -> Dict[str, Any]:
    """Measure whether matched-noise pair predictions preserve branch identity."""
    pair_ids = np.asarray(source_pair_ids, dtype=np.int64)
    source_count = int(torch.max(bank.source_ids).item()) + 1
    if pair_ids.shape != (source_count,):
        raise ValueError("source_pair_ids shape mismatch")
    counts = {
        int(value): int(np.sum(pair_ids == value))
        for value in np.unique(pair_ids)
    }
    if any(value != 2 for value in counts.values()):
        raise ValueError("every pair id must map to exactly two sources")

    predicted_raw = torch_inverse_standardize(
        predicted_z,
        future_mean,
        future_scale,
    )
    pred_xy = (
        predicted_raw[:, :, :48]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    target_xy = (
        bank.clean_raw[:, :, :48]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    source = bank.source_ids.detach().cpu().numpy().astype(np.int64)
    timestep = bank.timesteps.detach().cpu().numpy().astype(np.int64)
    noise_id = bank.noise_ids.detach().cpu().numpy().astype(np.int64)

    own_closer: List[float] = []
    target_separation: List[float] = []
    predicted_separation: List[float] = []
    separation_ratio: List[float] = []
    delta_cosine: List[float] = []

    for pair_id in sorted(counts):
        source_members = np.flatnonzero(pair_ids == int(pair_id))
        for local_timestep in sorted(set(timestep.tolist())):
            for local_noise in sorted(set(noise_id.tolist())):
                selected = np.flatnonzero(
                    np.isin(source, source_members)
                    & (timestep == int(local_timestep))
                    & (noise_id == int(local_noise))
                )
                if selected.size != 2:
                    raise RuntimeError(
                        "paired bank is incomplete for "
                        f"pair={pair_id}, t={local_timestep}, "
                        f"noise={local_noise}"
                    )
                a, b = int(selected[0]), int(selected[1])
                own_a = float(
                    np.sqrt(np.mean((pred_xy[a] - target_xy[a]) ** 2))
                )
                own_b = float(
                    np.sqrt(np.mean((pred_xy[b] - target_xy[b]) ** 2))
                )
                cross_a = float(
                    np.sqrt(np.mean((pred_xy[a] - target_xy[b]) ** 2))
                )
                cross_b = float(
                    np.sqrt(np.mean((pred_xy[b] - target_xy[a]) ** 2))
                )
                own_closer.extend(
                    [
                        float(own_a < cross_a),
                        float(own_b < cross_b),
                    ]
                )
                target_delta = (target_xy[a] - target_xy[b]).reshape(-1)
                predicted_delta = (pred_xy[a] - pred_xy[b]).reshape(-1)
                target_norm = float(np.linalg.norm(target_delta))
                predicted_norm = float(np.linalg.norm(predicted_delta))
                target_rmse = float(
                    np.sqrt(np.mean(target_delta ** 2))
                )
                predicted_rmse = float(
                    np.sqrt(np.mean(predicted_delta ** 2))
                )
                target_separation.append(target_rmse)
                predicted_separation.append(predicted_rmse)
                separation_ratio.append(
                    predicted_rmse / max(target_rmse, 1.0e-12)
                )
                delta_cosine.append(
                    float(
                        np.dot(target_delta, predicted_delta)
                        / max(target_norm * predicted_norm, 1.0e-12)
                    )
                )

    def stats(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(np.min(array)),
            "p05": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    own_fraction = float(np.mean(np.asarray(own_closer, dtype=np.float64)))
    ratio_stats = stats(separation_ratio)
    cosine_stats = stats(delta_cosine)
    return {
        "comparison_count": int(len(target_separation)),
        "own_target_closer_fraction": own_fraction,
        "target_ordered_separation": stats(target_separation),
        "predicted_ordered_separation": stats(predicted_separation),
        "separation_ratio": ratio_stats,
        "branch_delta_cosine": cosine_stats,
        "pass": bool(
            own_fraction >= 0.90
            and ratio_stats["p50"] >= 0.50
            and cosine_stats["p50"] >= 0.50
        ),
    }


def variant_passes_unique(result: Mapping[str, Any]) -> bool:
    return bool(result.get("pass"))


def variant_passes_paired(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("pass")
        and result.get("paired_branch_audit", {}).get("pass")
    )


def classify_pilot(report: Mapping[str, Any]) -> Dict[str, Any]:
    corrected = report["r241_corrected_interpretation"]
    if not bool(corrected["classifier_precedence_bug_supported"]):
        return {
            "root_cause": (
                "phase314b_r242_r241_evidence_contract_mismatch"
            ),
            "next_stage": "repair r2.4.1 evidence interpretation first",
            "train_only_recommendation": None,
        }

    width = report["direct_width_seed_stability"]
    width_512 = bool(width["512"]["stable_2_of_3"])
    width_1024 = bool(width["1024"]["stable_2_of_3"])

    unique = report.get("unique_free_variants", {})
    paired = report.get("paired_low_mid_variants", {})
    unique_pass = {
        name for name, value in unique.items()
        if variant_passes_unique(value)
    }
    paired_pass = {
        name for name, value in paired.items()
        if variant_passes_paired(value)
    }

    frozen_pass = {
        name for name in unique_pass
        if name.startswith("frozen_")
    }
    decoupled_pass = "decoupled_p1024_r512" in unique_pass
    joint_pass = "joint_p1024_r512_control" in unique_pass

    if not width_512 and width_1024 and not frozen_pass:
        return {
            "root_cause": (
                "phase314b_r242_condition_width_limit_replicated"
            ),
            "next_stage": (
                "retain width 1024 and continue train-only frozen-prior "
                "optimization"
            ),
            "train_only_recommendation": "prior_width_1024",
        }

    if not unique_pass:
        return {
            "root_cause": (
                "phase314b_r242_frozen_prior_residual_optimization_failed"
            ),
            "next_stage": (
                "debug residual-v optimization with a frozen prior on the "
                "same 16 train rows"
            ),
            "train_only_recommendation": None,
        }

    if frozen_pass and not joint_pass:
        if decoupled_pass:
            primary = (
                "phase314b_r242_prior_gradient_isolation_supported"
            )
            debug_recommendation = "decoupled_or_frozen_prior"
        else:
            primary = (
                "phase314b_r242_strict_frozen_prior_required"
            )
            debug_recommendation = "frozen_prior"
    else:
        primary = "phase314b_r242_factorized_multirow_supported"
        debug_recommendation = sorted(unique_pass)[0]

    if not paired_pass:
        return {
            "root_cause": (
                "phase314b_r242_paired_low_mid_branch_transport_failed"
            ),
            "secondary_mechanism": primary,
            "next_stage": (
                "debug paired low/mid-noise residual branch transport with "
                "the passing frozen-prior recipe"
            ),
            "train_only_recommendation": debug_recommendation,
        }

    priority = (
        "frozen_p512_r512",
        "frozen_p1024_r512",
        "decoupled_p1024_r512",
        "frozen_p512_r1024",
        "frozen_p1024_r1024",
    )
    recommendation = next(
        name for name in priority if name in paired_pass
    )
    return {
        "root_cause": primary,
        "next_stage": (
            "run a separate train-only ordered-geometry pilot using the "
            "factorized frozen/decoupled prior recipe; formal validation "
            "remains blocked"
        ),
        "train_only_recommendation": recommendation,
    }
