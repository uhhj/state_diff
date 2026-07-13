"""Train-only conditioned multirow and source-batching audit.

Phase3.14b-r2.4.1 is additive and diagnostic.  It uses only train rows,
does not select a formal configuration, does not save model weights, and does
not read formal-test targets.
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
from ccda_phase3.phase314b_r232_controls import seed_all
from ccda_phase3.phase314b_r24_noisy_skip import (
    AnalyticX0SkipDenoiser,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM

PHASE = "phase3_14b_r241"
BASE_COMMIT = "ba6f913add4a9508179977c4adc52912de35533b"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R24_ROOT_CAUSE = (
    "phase314b_r24_conditioned_multirow_generalization_failed"
)
EXPECTED_R24_MODULE_SHA256 = (
    "f8ccac7ef82297c12fe152a48532fcb5a938e12e0eb0faca6914a27a6e37d307"
)
EXPECTED_R23_TINY_SHA256 = (
    "d370bef927af6dd0f3f5f94171482889dbadbbf5cf28c7f410567e5b25c886b2"
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r241_multirow.py",
    "scripts/phase3_14b_r241_preflight.py",
    "scripts/phase3_14b_r241_run_audit.py",
    "scripts/phase3_14b_r241_finalize.py",
    "scripts/phase3_14b_r241_run.sh",
    "tests/test_phase314b_r241_multirow.py",
)

DEPENDENCY_SHA256 = {
    "ccda_phase3/phase314b_models.py":
        "db8ff101114934854f6e9cf32aeab6f7ae0063949fc95d00a674544cde9041f6",
    "ccda_phase3/phase314b_r2_contract.py":
        "aa92c4a851c3183e5aa4758169bd1b83412ef6e60c67541cce9deb1274b33283",
    "ccda_phase3/phase314b_r2_diffusion.py":
        "417af4ebc17517ec0f4c403e45fcc69b01ffe76f8d85f3cf090479d73a5ea3ba",
    "ccda_phase3/phase314b_r22_geometry.py":
        "e2d3fcb5630677910642561a9f604623ca6c0ea3663943cd19becfc483919371",
    "ccda_phase3/phase314b_r231_controls.py":
        "c16bfbfa3ae0b7b27c25ee79cf298c76c3d5511a545af228a898869d96098d35",
    "ccda_phase3/phase314b_r232_controls.py":
        "7d661167115249c77e52b6e8646cb05d2b28f8ce334f01f775078d7fd84dd27c",
    "ccda_phase3/phase314b_r24_noisy_skip.py":
        EXPECTED_R24_MODULE_SHA256,
    "ccda_phase3/schema_v2.py":
        "e1594460477e4b96e3899509d4b145d36691c4a26818ad938b530372a2588c2b",
}

AUDIT_TIMESTEPS = (10, 25, 50, 75, 90, 99)
HIGH_TIMESTEPS = (75, 90, 99)


@dataclass(frozen=True)
class PriorTrainSpec:
    steps: int = 5000
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    hidden_dim: int = 512

    def validate(self) -> None:
        if self.steps <= 0 or self.hidden_dim <= 0:
            raise ValueError("prior steps/hidden_dim must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("prior learning rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("prior weight decay must be nonnegative")


@dataclass(frozen=True)
class DiffusionTrainSpec:
    warmup_steps: int = 2500
    diffusion_steps: int = 8000
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    clip_grad_norm: Optional[float] = None

    def validate(self) -> None:
        if self.warmup_steps <= 0 or self.diffusion_steps <= 0:
            raise ValueError("training steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be nonnegative")
        if self.clip_grad_norm is not None:
            if (
                not math.isfinite(self.clip_grad_norm)
                or self.clip_grad_norm <= 0
            ):
                raise ValueError("clip grad norm must be positive")


@dataclass(frozen=True)
class MultirowVariant:
    name: str
    sampling_mode: str
    freeze_prior: bool
    base_anchor_weight: float
    reuse_warmup_optimizer: bool
    hidden_dim: int = 512
    time_dim: int = 128

    def validate(self) -> None:
        if not self.name:
            raise ValueError("variant name is required")
        if self.sampling_mode not in {"random", "balanced"}:
            raise ValueError("unsupported sampling mode")
        if not math.isfinite(self.base_anchor_weight):
            raise ValueError("base anchor must be finite")
        if self.base_anchor_weight < 0:
            raise ValueError("base anchor must be nonnegative")
        if self.hidden_dim <= 0 or self.time_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.freeze_prior and self.base_anchor_weight != 0:
            raise ValueError("frozen prior cannot use a base anchor")


MULTIROW_VARIANTS = (
    MultirowVariant(
        name="current_joint_random_reused_optimizer",
        sampling_mode="random",
        freeze_prior=False,
        base_anchor_weight=0.0,
        reuse_warmup_optimizer=True,
    ),
    MultirowVariant(
        name="balanced_joint_fresh_optimizer",
        sampling_mode="balanced",
        freeze_prior=False,
        base_anchor_weight=0.0,
        reuse_warmup_optimizer=False,
    ),
    MultirowVariant(
        name="balanced_frozen_prior",
        sampling_mode="balanced",
        freeze_prior=True,
        base_anchor_weight=0.0,
        reuse_warmup_optimizer=False,
    ),
    MultirowVariant(
        name="balanced_anchor_prior",
        sampling_mode="balanced",
        freeze_prior=False,
        base_anchor_weight=1.0,
        reuse_warmup_optimizer=False,
    ),
    MultirowVariant(
        name="random_anchor_prior",
        sampling_mode="random",
        freeze_prior=False,
        base_anchor_weight=1.0,
        reuse_warmup_optimizer=False,
    ),
)
for _variant in MULTIROW_VARIANTS:
    _variant.validate()


@dataclass
class LabeledTupleBank:
    condition_z: torch.Tensor
    clean_z: torch.Tensor
    clean_raw: torch.Tensor
    noisy: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    noise_ids: torch.Tensor
    source_ids: torch.Tensor

    def validate(self) -> None:
        count = int(self.noisy.shape[0])
        expected = (count, DEFAULT_TF, STATE_DIM)
        for name in ("clean_z", "clean_raw", "noisy", "noise"):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} has invalid shape: {value.shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if self.condition_z.ndim != 2 or self.condition_z.shape[0] != count:
            raise ValueError("condition_z has invalid shape")
        if not bool(torch.isfinite(self.condition_z).all()):
            raise ValueError("condition_z contains non-finite values")
        for name in ("timesteps", "noise_ids", "source_ids"):
            value = getattr(self, name)
            if tuple(value.shape) != (count,):
                raise ValueError(f"{name} has invalid shape")
        if bool(torch.any(self.source_ids < 0)):
            raise ValueError("source_ids must be nonnegative")

    @property
    def row_count(self) -> int:
        return int(self.noisy.shape[0])

    def index_select(self, index: torch.Tensor) -> "LabeledTupleBank":
        selected = index.long().reshape(-1)
        result = LabeledTupleBank(
            condition_z=self.condition_z.index_select(0, selected),
            clean_z=self.clean_z.index_select(0, selected),
            clean_raw=self.clean_raw.index_select(0, selected),
            noisy=self.noisy.index_select(0, selected),
            noise=self.noise.index_select(0, selected),
            timesteps=self.timesteps.index_select(0, selected),
            noise_ids=self.noise_ids.index_select(0, selected),
            source_ids=self.source_ids.index_select(0, selected),
        )
        result.validate()
        return result


class ConditionX0Prior(nn.Module):
    """Condition-only standardized future regressor."""

    def __init__(self, condition_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        flat = DEFAULT_TF * STATE_DIM
        self.net = nn.Sequential(
            nn.Linear(self.condition_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), flat),
        )

    def forward(self, condition_z: torch.Tensor) -> torch.Tensor:
        if condition_z.ndim != 2:
            raise ValueError("condition_z must be [B,C]")
        if condition_z.shape[1] != self.condition_dim:
            raise ValueError("condition dimension mismatch")
        return self.net(condition_z).reshape(
            condition_z.shape[0],
            DEFAULT_TF,
            STATE_DIM,
        )


class SourceBatchSampler:
    """Return source indices while preserving exact condition/target alignment."""

    def __init__(
        self,
        source_count: int,
        batch_size: int,
        mode: str,
        *,
        device: torch.device,
        seed: int,
    ) -> None:
        self.source_count = int(source_count)
        self.batch_size = int(batch_size)
        self.mode = str(mode)
        self.device = device
        if self.source_count <= 0 or self.batch_size <= 0:
            raise ValueError("source_count and batch_size must be positive")
        if self.mode not in {"random", "balanced"}:
            raise ValueError("unsupported sampling mode")
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.counts = torch.zeros(
            self.source_count,
            device=device,
            dtype=torch.long,
        )
        self._order = torch.empty(0, device=device, dtype=torch.long)
        self._cursor = 0

    def _next_balanced(self) -> torch.Tensor:
        parts = []
        remaining = self.batch_size
        while remaining > 0:
            if self._cursor >= int(self._order.numel()):
                self._order = torch.randperm(
                    self.source_count,
                    generator=self.generator,
                    device=self.device,
                )
                self._cursor = 0
            take = min(
                remaining,
                int(self._order.numel()) - self._cursor,
            )
            parts.append(self._order[self._cursor:self._cursor + take])
            self._cursor += take
            remaining -= take
        return torch.cat(parts, dim=0)

    def next_indices(self) -> torch.Tensor:
        if self.mode == "random":
            index = torch.randint(
                0,
                self.source_count,
                (self.batch_size,),
                generator=self.generator,
                device=self.device,
                dtype=torch.long,
            )
        else:
            index = self._next_balanced()
        self.counts += torch.bincount(
            index,
            minlength=self.source_count,
        )
        return index

    def count_report(self) -> Dict[str, Any]:
        value = self.counts.detach().cpu().numpy().astype(np.int64)
        mean = float(np.mean(value))
        return {
            "counts": value.tolist(),
            "min": int(np.min(value)),
            "max": int(np.max(value)),
            "mean": mean,
            "max_minus_min": int(np.max(value) - np.min(value)),
            "relative_range": (
                float((np.max(value) - np.min(value)) / mean)
                if mean > 0 else float("inf")
            ),
        }


def seed_everything(seed: int) -> None:
    seed_all(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


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
    observed = {
        path: sha256_file(base / path)
        for path in DEPENDENCY_SHA256
    }
    mismatch = {
        path: {
            "expected": DEPENDENCY_SHA256[path],
            "observed": observed[path],
        }
        for path in DEPENDENCY_SHA256
        if observed[path] != DEPENDENCY_SHA256[path]
    }
    if mismatch:
        raise RuntimeError(f"r2.4.1 dependency hash mismatch: {mismatch}")
    return observed


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
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
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


def build_labeled_bank(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    timesteps: Sequence[int],
    noise_seeds: Sequence[int],
    matched_noise_across_sources: bool = True,
) -> LabeledTupleBank:
    """Build source-tagged deterministic tuples.

    Source tags make condition/target alignment directly auditable.  Matched
    noises isolate condition effects; unmatched noises are available for
    robustness tests.
    """
    if condition_z.ndim != 2:
        raise ValueError("condition_z must be [N,C]")
    source_count = int(condition_z.shape[0])
    expected = (source_count, DEFAULT_TF, STATE_DIM)
    if tuple(clean_z.shape) != expected:
        raise ValueError("clean_z shape mismatch")
    if tuple(clean_raw.shape) != expected:
        raise ValueError("clean_raw shape mismatch")
    times = tuple(int(value) for value in timesteps)
    seeds = tuple(int(value) for value in noise_seeds)
    if not times or not seeds:
        raise ValueError("timesteps/noise seeds must be nonempty")
    if any(value < 0 or value >= 100 for value in times):
        raise ValueError("timestep out of range")

    active = active_mask.to(device=clean_z.device, dtype=torch.bool)
    condition_items = []
    clean_items = []
    raw_items = []
    noisy_items = []
    noise_items = []
    time_items = []
    noise_id_items = []
    source_items = []

    for source_id in range(source_count):
        for timestep in times:
            for noise_id in seeds:
                effective_seed = int(noise_id)
                if not matched_noise_across_sources:
                    effective_seed += 1000003 * int(source_id)
                noise = _active_noise(
                    (1, DEFAULT_TF, STATE_DIM),
                    active,
                    seed=effective_seed,
                    device=clean_z.device,
                    dtype=clean_z.dtype,
                )
                time = torch.full(
                    (1,),
                    int(timestep),
                    device=clean_z.device,
                    dtype=torch.long,
                )
                clean_one = clean_z[source_id:source_id + 1]
                noisy = scheduler.add_noise(clean_one, noise, time)
                noisy = torch.where(
                    active[None],
                    noisy,
                    torch.zeros_like(noisy),
                )
                condition_items.append(
                    condition_z[source_id:source_id + 1]
                )
                clean_items.append(clean_one)
                raw_items.append(clean_raw[source_id:source_id + 1])
                noisy_items.append(noisy)
                noise_items.append(noise)
                time_items.append(int(timestep))
                noise_id_items.append(int(noise_id))
                source_items.append(int(source_id))

    result = LabeledTupleBank(
        condition_z=torch.cat(condition_items, dim=0),
        clean_z=torch.cat(clean_items, dim=0),
        clean_raw=torch.cat(raw_items, dim=0),
        noisy=torch.cat(noisy_items, dim=0),
        noise=torch.cat(noise_items, dim=0),
        timesteps=torch.tensor(
            time_items,
            device=clean_z.device,
            dtype=torch.long,
        ),
        noise_ids=torch.tensor(
            noise_id_items,
            device=clean_z.device,
            dtype=torch.long,
        ),
        source_ids=torch.tensor(
            source_items,
            device=clean_z.device,
            dtype=torch.long,
        ),
    )
    result.validate()
    return result


def source_alignment_audit(
    *,
    bank: LabeledTupleBank,
    source_condition: torch.Tensor,
    source_clean_z: torch.Tensor,
    source_clean_raw: torch.Tensor,
) -> Dict[str, Any]:
    bank.validate()
    source_ids = bank.source_ids.long()
    expected_condition = source_condition.index_select(0, source_ids)
    expected_clean = source_clean_z.index_select(0, source_ids)
    expected_raw = source_clean_raw.index_select(0, source_ids)
    condition_max = float(
        torch.max(torch.abs(bank.condition_z - expected_condition)).cpu()
    )
    clean_max = float(
        torch.max(torch.abs(bank.clean_z - expected_clean)).cpu()
    )
    raw_max = float(
        torch.max(torch.abs(bank.clean_raw - expected_raw)).cpu()
    )
    return {
        "condition_max_abs": condition_max,
        "clean_z_max_abs": clean_max,
        "clean_raw_max_abs": raw_max,
        "pass": bool(
            condition_max == 0.0
            and clean_max == 0.0
            and raw_max == 0.0
        ),
    }


def sample_aligned_source_batch(
    *,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    sampler: SourceBatchSampler,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = sampler.next_indices()
    return (
        index,
        condition_z.index_select(0, index),
        clean_z.index_select(0, index),
        clean_raw.index_select(0, index),
    )


def _pairwise_upper(value: np.ndarray) -> np.ndarray:
    count = int(value.shape[0])
    if count < 2:
        return np.zeros((0,), dtype=np.float64)
    distance = np.sqrt(
        np.sum(
            (value[:, None, :] - value[None, :, :]) ** 2,
            axis=-1,
        )
    )
    upper = np.triu_indices(count, k=1)
    return distance[upper].astype(np.float64)


def condition_identifiability_audit(
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
) -> Dict[str, Any]:
    condition = (
        condition_z.detach().cpu().numpy().astype(np.float64)
    )
    target_z = clean_z.detach().cpu().numpy().reshape(
        condition.shape[0], -1
    ).astype(np.float64)
    target_xy = clean_raw[:, :, :48].detach().cpu().numpy().reshape(
        condition.shape[0], -1
    ).astype(np.float64)
    count = int(condition.shape[0])
    centered = condition - np.mean(condition, axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered, tol=1.0e-8))

    exact_groups: Dict[bytes, List[int]] = {}
    for index, row in enumerate(condition):
        exact_groups.setdefault(row.tobytes(), []).append(index)
    duplicates = [
        indices for indices in exact_groups.values()
        if len(indices) > 1
    ]
    duplicate_conflicts = []
    for indices in duplicates:
        local = target_xy[np.asarray(indices, dtype=np.int64)]
        pair = _pairwise_upper(local)
        duplicate_conflicts.append(
            {
                "indices": [int(value) for value in indices],
                "target_ordered_rmse_max": (
                    float(np.max(pair) / math.sqrt(local.shape[1]))
                    if pair.size else 0.0
                ),
            }
        )

    condition_distance = _pairwise_upper(condition)
    target_distance = _pairwise_upper(target_z)
    correlation = 0.0
    if condition_distance.size >= 2:
        condition_std = float(np.std(condition_distance))
        target_std = float(np.std(target_distance))
        if condition_std > 0 and target_std > 0:
            correlation = float(
                np.corrcoef(condition_distance, target_distance)[0, 1]
            )

    nearest_target = []
    nearest_condition = []
    for index in range(count):
        delta = np.sqrt(
            np.sum((condition - condition[index:index + 1]) ** 2, axis=1)
        )
        delta[index] = np.inf
        nearest = int(np.argmin(delta))
        nearest_condition.append(float(delta[nearest]))
        nearest_target.append(
            float(
                np.sqrt(
                    np.mean(
                        (target_xy[index] - target_xy[nearest]) ** 2
                    )
                )
            )
        )

    def stats(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {
                "min": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "max": 0.0,
            }
        return {
            "min": float(np.min(array)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    conflict = any(
        item["target_ordered_rmse_max"] > 1.0e-6
        for item in duplicate_conflicts
    )
    return {
        "row_count": count,
        "condition_dim": int(condition.shape[1]),
        "centered_rank": rank,
        "exact_duplicate_group_count": int(len(duplicates)),
        "duplicate_target_conflicts": duplicate_conflicts,
        "exact_duplicate_conflict": bool(conflict),
        "pairwise_condition_l2": stats(condition_distance.tolist()),
        "nearest_condition_l2": stats(nearest_condition),
        "nearest_target_ordered_rmse": stats(nearest_target),
        "condition_target_pairwise_distance_correlation": correlation,
        "pass": bool(not conflict and (count <= 1 or rank > 0)),
    }


def _optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    trainable = [value for value in parameters if value.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _gradient_step(
    *,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    clip_grad_norm: Optional[float],
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameter_list = [
        value for value in parameters
        if value.requires_grad and value.grad is not None
    ]
    if clip_grad_norm is None:
        total = sum(
            float(torch.sum(value.grad.detach() ** 2).cpu())
            for value in parameter_list
        )
        norm = math.sqrt(total)
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
    active = active_mask.to(device=predicted_z.device, dtype=torch.bool)
    prediction = torch.where(
        active[None],
        predicted_z,
        torch.zeros_like(predicted_z),
    )
    raw = torch_inverse_standardize(
        prediction,
        future_mean,
        future_scale,
    )
    metrics = target_reconstruction_metrics(
        predicted_z=prediction,
        target_z=target_z,
        predicted_raw=raw,
        target_raw=target_raw,
        active_mask=active,
    )
    return {
        "metrics": metrics,
        "gate": asdict(gate),
        "gate_pass": bool(gate_reconstruction(metrics, gate)),
    }


def _metrics_by_source(
    *,
    predicted_z: torch.Tensor,
    bank: LabeledTupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    gate: ReconstructionGate,
) -> Dict[str, Any]:
    values = {}
    passes = []
    for source_id in sorted(
        set(bank.source_ids.detach().cpu().tolist())
    ):
        index = torch.nonzero(
            bank.source_ids == int(source_id),
            as_tuple=False,
        ).reshape(-1)
        result = _prediction_metrics(
            predicted_z=predicted_z.index_select(0, index),
            target_z=bank.clean_z.index_select(0, index),
            target_raw=bank.clean_raw.index_select(0, index),
            active_mask=active_mask,
            future_mean=future_mean,
            future_scale=future_scale,
            gate=gate,
        )
        values[str(int(source_id))] = {
            "row_count": int(index.numel()),
            **result,
        }
        passes.append(bool(result["gate_pass"]))
    return {
        "by_source": values,
        "all_source_gate_pass": bool(all(passes)),
        "source_pass_fraction": (
            float(np.mean(np.asarray(passes, dtype=np.float64)))
            if passes else 0.0
        ),
    }


def train_condition_prior(
    *,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    spec: PriorTrainSpec,
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    spec.validate()
    seed_everything(seed)
    model = ConditionX0Prior(
        condition_dim=int(condition_z.shape[1]),
        hidden_dim=int(spec.hidden_dim),
    ).to(condition_z.device)
    optimizer = _optimizer(
        model.parameters(),
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    active = active_mask.to(device=condition_z.device, dtype=torch.bool)
    history = []
    for step in range(1, int(spec.steps) + 1):
        prediction = model(condition_z)
        loss = active_mse(prediction, clean_z, active)
        norm = _gradient_step(
            optimizer=optimizer,
            loss=loss,
            parameters=model.parameters(),
            clip_grad_norm=None,
        )
        if (
            step == 1
            or step % int(log_interval) == 0
            or step == spec.steps
        ):
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(norm),
                }
            )
    model.eval()
    with torch.no_grad():
        prediction = model(condition_z)
    aggregate = _prediction_metrics(
        predicted_z=prediction,
        target_z=clean_z,
        target_raw=clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=DIRECT_REGRESSION_GATE,
    )
    # Treat each source row as its own bank for worst-row gating.
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
    by_source = _metrics_by_source(
        predicted_z=prediction,
        bank=bank,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=DIRECT_REGRESSION_GATE,
    )
    return {
        "spec": asdict(spec),
        "history": history,
        "aggregate": aggregate,
        **by_source,
        "pass": bool(
            aggregate["gate_pass"]
            and by_source["all_source_gate_pass"]
        ),
        "parameter_count": int(
            sum(value.numel() for value in model.parameters())
        ),
    }


@torch.no_grad()
def evaluate_denoiser_bank(
    *,
    model: AnalyticX0SkipDenoiser,
    scheduler,
    bank: LabeledTupleBank,
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
    prediction = predict_original_sample(
        scheduler=scheduler,
        config=repair,
        sample=bank.noisy,
        model_output=output,
        timesteps=bank.timesteps,
    )
    prediction = torch.where(
        active[None],
        prediction,
        torch.zeros_like(prediction),
    )
    aggregate = _prediction_metrics(
        predicted_z=prediction,
        target_z=bank.clean_z,
        target_raw=bank.clean_raw,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=gate,
    )
    by_source = _metrics_by_source(
        predicted_z=prediction,
        bank=bank,
        active_mask=active,
        future_mean=future_mean,
        future_scale=future_scale,
        gate=gate,
    )
    by_timestep = {}
    for timestep in sorted(
        set(bank.timesteps.detach().cpu().tolist())
    ):
        index = torch.nonzero(
            bank.timesteps == int(timestep),
            as_tuple=False,
        ).reshape(-1)
        item = _prediction_metrics(
            predicted_z=prediction.index_select(0, index),
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
        "v_target_mse": v_mse,
        "aggregate": aggregate,
        **by_source,
        "by_timestep": by_timestep,
        "gate_pass": bool(
            aggregate["gate_pass"]
            and by_source["all_source_gate_pass"]
        ),
    }


def _replace_bank_condition(
    bank: LabeledTupleBank,
    condition_z: torch.Tensor,
) -> LabeledTupleBank:
    if condition_z.shape != bank.condition_z.shape:
        raise ValueError("replacement condition shape mismatch")
    result = LabeledTupleBank(
        condition_z=condition_z,
        clean_z=bank.clean_z,
        clean_raw=bank.clean_raw,
        noisy=bank.noisy,
        noise=bank.noise,
        timesteps=bank.timesteps,
        noise_ids=bank.noise_ids,
        source_ids=bank.source_ids,
    )
    result.validate()
    return result


def condition_ablation_banks(
    bank: LabeledTupleBank,
    *,
    seed: int,
) -> Dict[str, LabeledTupleBank]:
    source_count = int(torch.max(bank.source_ids).item()) + 1
    source_condition = torch.empty(
        source_count,
        bank.condition_z.shape[1],
        device=bank.condition_z.device,
        dtype=bank.condition_z.dtype,
    )
    for source_id in range(source_count):
        first = torch.nonzero(
            bank.source_ids == source_id,
            as_tuple=False,
        ).reshape(-1)[0]
        source_condition[source_id] = bank.condition_z[first]

    # A cyclic shift is a deterministic derangement for source_count > 1;
    # unlike a random permutation it cannot accidentally leave rows fixed.
    if source_count <= 1:
        permutation = torch.arange(
            source_count,
            device=bank.condition_z.device,
            dtype=torch.long,
        )
    else:
        shift = 1 + (int(seed) % (source_count - 1))
        permutation = torch.roll(
            torch.arange(
                source_count,
                device=bank.condition_z.device,
                dtype=torch.long,
            ),
            shifts=shift,
        )
    permuted_source = source_condition.index_select(0, permutation)
    permuted = permuted_source.index_select(0, bank.source_ids.long())
    zero = torch.zeros_like(bank.condition_z)
    return {
        "true": bank,
        "permuted": _replace_bank_condition(bank, permuted),
        "zero": _replace_bank_condition(bank, zero),
    }


def _prior_metrics(
    *,
    model: AnalyticX0SkipDenoiser,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
) -> Dict[str, Any]:
    with torch.no_grad():
        prediction = model.predict_base_x0(condition_z)
    aggregate = _prediction_metrics(
        predicted_z=prediction,
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
    by_source = _metrics_by_source(
        predicted_z=prediction,
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
        "permuted_to_true_ratio": float(permuted_value / denominator),
        "condition_effect_supported": bool(
            zero_value >= 1.5 * denominator
            and permuted_value >= 1.5 * denominator
        ),
    }


def train_multirow_variant(
    *,
    scheduler,
    condition_z: torch.Tensor,
    clean_z: torch.Tensor,
    clean_raw: torch.Tensor,
    evaluation_bank: LabeledTupleBank,
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    variant: MultirowVariant,
    train_spec: DiffusionTrainSpec,
    seed: int,
    log_interval: int = 250,
) -> Dict[str, Any]:
    variant.validate()
    train_spec.validate()
    seed_everything(seed)
    device = condition_z.device
    active = active_mask.to(device=device, dtype=torch.bool)
    model = AnalyticX0SkipDenoiser(
        condition_dim=int(condition_z.shape[1]),
        alpha_bar=scheduler.alphas_cumprod.detach().cpu(),
        hidden_dim=int(variant.hidden_dim),
        time_dim=int(variant.time_dim),
        use_residual=True,
    ).to(device)

    warmup_optimizer = _optimizer(
        model.parameters(),
        learning_rate=train_spec.learning_rate,
        weight_decay=train_spec.weight_decay,
    )
    warmup_history = []
    for step in range(1, int(train_spec.warmup_steps) + 1):
        prediction = model.predict_base_x0(condition_z)
        loss = active_mse(prediction, clean_z, active)
        norm = _gradient_step(
            optimizer=warmup_optimizer,
            loss=loss,
            parameters=model.parameters(),
            clip_grad_norm=train_spec.clip_grad_norm,
        )
        if (
            step == 1
            or step % int(log_interval) == 0
            or step == train_spec.warmup_steps
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

    if variant.freeze_prior:
        for parameter in model.x0_prior.parameters():
            parameter.requires_grad_(False)

    if variant.reuse_warmup_optimizer:
        diffusion_optimizer = warmup_optimizer
    else:
        diffusion_optimizer = _optimizer(
            model.parameters(),
            learning_rate=train_spec.learning_rate,
            weight_decay=train_spec.weight_decay,
        )

    sampler = SourceBatchSampler(
        source_count=int(condition_z.shape[0]),
        batch_size=int(train_spec.batch_size),
        mode=variant.sampling_mode,
        device=device,
        seed=int(seed) + 1000,
    )
    noise_generator = torch.Generator(device=device).manual_seed(
        int(seed) + 2000
    )
    repair = REPAIR_CONFIGS["v_prediction_cosine"]
    history = []

    for step in range(1, int(train_spec.diffusion_steps) + 1):
        source_index, condition_batch, clean_batch, _ = (
            sample_aligned_source_batch(
                condition_z=condition_z,
                clean_z=clean_z,
                clean_raw=clean_raw,
                sampler=sampler,
            )
        )
        batch_size = int(source_index.numel())
        timestep = torch.randint(
            0,
            100,
            (batch_size,),
            generator=noise_generator,
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn(
            clean_batch.shape,
            generator=noise_generator,
            device=device,
            dtype=clean_batch.dtype,
        )
        noise = torch.where(active[None], noise, torch.zeros_like(noise))
        noisy = scheduler.add_noise(clean_batch, noise, timestep)
        noisy = torch.where(active[None], noisy, torch.zeros_like(noisy))
        output = model(noisy, timestep, condition_batch)
        output = torch.where(active[None], output, torch.zeros_like(output))
        target = training_target(
            scheduler=scheduler,
            config=repair,
            clean_sample=clean_batch,
            noise=noise,
            timesteps=timestep,
        )
        v_loss = active_mse(output, target, active)
        base_loss = active_mse(
            model.predict_base_x0(condition_batch),
            clean_batch,
            active,
        )
        total = (
            v_loss
            + float(variant.base_anchor_weight) * base_loss
        )
        norm = _gradient_step(
            optimizer=diffusion_optimizer,
            loss=total,
            parameters=model.parameters(),
            clip_grad_norm=train_spec.clip_grad_norm,
        )
        if (
            step == 1
            or step % int(log_interval) == 0
            or step == train_spec.diffusion_steps
        ):
            history.append(
                {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "v_loss": float(v_loss.detach().cpu()),
                    "base_x0_loss": float(base_loss.detach().cpu()),
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
    ablations = condition_ablation_banks(
        evaluation_bank,
        seed=int(seed) + 3000,
    )
    evaluations = {
        key: evaluate_denoiser_bank(
            model=model,
            scheduler=scheduler,
            bank=value,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        for key, value in ablations.items()
    }

    high_mask = torch.zeros_like(
        evaluation_bank.timesteps,
        dtype=torch.bool,
    )
    for high_timestep in HIGH_TIMESTEPS:
        high_mask = high_mask | (
            evaluation_bank.timesteps == int(high_timestep)
        )
    high_index = torch.nonzero(
        high_mask,
        as_tuple=False,
    ).reshape(-1)
    high_bank = evaluation_bank.index_select(high_index)
    high_ablations = condition_ablation_banks(
        high_bank,
        seed=int(seed) + 4000,
    )
    high_evaluations = {
        key: evaluate_denoiser_bank(
            model=model,
            scheduler=scheduler,
            bank=value,
            active_mask=active,
            future_mean=future_mean,
            future_scale=future_scale,
        )
        for key, value in high_ablations.items()
    }

    warmup_mse = float(
        prior_after_warmup["metrics"]["z_mse"]
    )
    post_mse = float(
        prior_after_diffusion["metrics"]["z_mse"]
    )
    prior_drift_ratio = float(
        post_mse / max(warmup_mse, 1.0e-12)
    )
    result = {
        "variant": asdict(variant),
        "train": asdict(train_spec),
        "source_row_count": int(condition_z.shape[0]),
        "warmup_history": warmup_history,
        "diffusion_history": history,
        "sampling": sampler.count_report(),
        "prior_after_warmup": prior_after_warmup,
        "prior_after_diffusion": prior_after_diffusion,
        "prior_drift_ratio": prior_drift_ratio,
        "evaluations": evaluations,
        "condition_effect": _condition_effect(evaluations),
        "high_timestep_evaluations": high_evaluations,
        "high_timestep_condition_effect": _condition_effect(
            high_evaluations
        ),
        "pass": bool(
            evaluations["true"]["gate_pass"]
            and evaluations["true"]["all_source_gate_pass"]
            and _condition_effect(high_evaluations)[
                "condition_effect_supported"
            ]
        ),
        "candidate_eligible": False,
        "selected_configuration": None,
        "checkpoint_saved": False,
        "parameter_count": int(
            sum(value.numel() for value in model.parameters())
        ),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _direct_pass(
    direct_controls: Mapping[str, Any],
    key: str,
) -> bool:
    return bool(direct_controls.get(key, {}).get("pass", False))


def classify_audit(report: Mapping[str, Any]) -> Dict[str, Any]:
    alignment = report["source_alignment"]
    if not bool(alignment.get("pass")):
        return {
            "root_cause": "phase314b_r241_source_row_alignment_bug_supported",
            "next_stage": "repair source-row alignment before any model audit",
            "train_only_debug_recommendation": None,
        }

    identifiability = report["condition_identifiability"]
    if bool(identifiability.get("exact_duplicate_conflict")):
        return {
            "root_cause": (
                "phase314b_r241_condition_input_nonidentifiability_supported"
            ),
            "next_stage": (
                "revise the observable condition contract or target grouping; "
                "do not increase model capacity"
            ),
            "train_only_debug_recommendation": None,
        }

    direct = report["direct_condition_controls"]
    direct_16 = _direct_pass(direct, "rows_16_width_512")
    wide_16 = _direct_pass(direct, "rows_16_width_1024")
    if not direct_16 and not wide_16:
        return {
            "root_cause": (
                "phase314b_r241_direct_condition_multirow_capacity_failed"
            ),
            "next_stage": (
                "isolate condition normalization/encoder capacity on the "
                "16 train rows before diffusion"
            ),
            "train_only_debug_recommendation": None,
        }
    if not direct_16 and wide_16:
        return {
            "root_cause": (
                "phase314b_r241_condition_encoder_width_limit_supported"
            ),
            "next_stage": (
                "run an additive train-only wider condition-encoder pilot"
            ),
            "train_only_debug_recommendation": "condition_width_1024",
        }

    runs = report.get("diffusion_variants", {})
    passing = [
        name for name, value in runs.items()
        if bool(value.get("pass"))
    ]
    current = runs.get("current_joint_random_reused_optimizer")
    balanced = runs.get("balanced_joint_fresh_optimizer")
    frozen = runs.get("balanced_frozen_prior")
    anchor = runs.get("balanced_anchor_prior")
    random_anchor = runs.get("random_anchor_prior")

    if passing:
        priority = (
            "balanced_frozen_prior",
            "balanced_anchor_prior",
            "random_anchor_prior",
            "balanced_joint_fresh_optimizer",
            "current_joint_random_reused_optimizer",
        )
        recommendation = next(
            name for name in priority if name in passing
        )
        if (
            current is not None
            and not bool(current.get("pass"))
            and balanced is not None
            and bool(balanced.get("pass"))
        ):
            root = (
                "phase314b_r241_random_source_batch_or_optimizer_state_"
                "failure_supported"
            )
        elif (
            current is not None
            and not bool(current.get("pass"))
            and (
                (frozen is not None and bool(frozen.get("pass")))
                or (anchor is not None and bool(anchor.get("pass")))
                or (
                    random_anchor is not None
                    and bool(random_anchor.get("pass"))
                )
            )
        ):
            root = (
                "phase314b_r241_x0_prior_drift_under_joint_diffusion_"
                "supported"
            )
        else:
            root = "phase314b_r241_train_only_conditioned_multirow_supported"
        return {
            "root_cause": root,
            "next_stage": (
                "run a separate train-only paired low/mid-noise audit using "
                "the frozen debug recipe; formal validation remains blocked"
            ),
            "train_only_debug_recommendation": recommendation,
        }

    effects = [
        bool(
            value.get("high_timestep_condition_effect", {}).get(
                "condition_effect_supported"
            )
        )
        for value in runs.values()
    ]
    if runs and not any(effects):
        return {
            "root_cause": (
                "phase314b_r241_condition_path_underutilization_supported"
            ),
            "next_stage": (
                "add explicit condition modulation and repeat train-only "
                "unique-free controls"
            ),
            "train_only_debug_recommendation": None,
        }

    drift = [
        float(value.get("prior_drift_ratio", 1.0))
        for value in runs.values()
    ]
    if drift and max(drift) >= 100.0:
        return {
            "root_cause": (
                "phase314b_r241_x0_prior_drift_without_recovery_supported"
            ),
            "next_stage": (
                "freeze or strongly anchor the condition x0 prior and debug "
                "residual optimization"
            ),
            "train_only_debug_recommendation": None,
        }

    return {
        "root_cause": (
            "phase314b_r241_conditioned_residual_multirow_optimization_failed"
        ),
        "next_stage": (
            "debug residual-v optimization and condition modulation on the "
            "same 16 train rows"
        ),
        "train_only_debug_recommendation": None,
    }
