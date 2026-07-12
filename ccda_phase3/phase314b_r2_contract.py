"""Phase3.14b-r2 targeted objective/schedule repair contract."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import (
    BEAD_XY_DIM,
    CACHE_ROWS,
    N_BEADS,
    Standardizer,
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    fixed_balanced_eval_indices,
    future_standardizer,
    input_values_and_standardizer,
    load_locked_cache,
    paired_key_index,
    split_mask,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, PAPER_X_DIM, STATE_DIM


PHASE = "phase3_14b_r2"
SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
OLD_FAILED_CHECKPOINT_SHA256 = (
    "28d876028d46d9efc71ae70c16b44a6933806654784538a366b95c89275d8e2a"
)
DIFFUSERS_VERSION = "0.11.1"

FORMAL_SEEDS = (31431, 31432, 31433)
SMOKE_SEED = 31430
K_VALUES = (1, 4, 8, 16, 32)

DETERMINISTIC_VAL_REFERENCE = 0.060045
VAL_K1_LIMIT = max(0.25, 4.0 * DETERMINISTIC_VAL_REFERENCE)
VAL_BEST8_LIMIT = max(0.12, 2.0 * DETERMINISTIC_VAL_REFERENCE)
PARTIAL_T10_LIMIT = max(0.06, DETERMINISTIC_VAL_REFERENCE)
PARTIAL_T99_LIMIT = max(0.25, 4.0 * DETERMINISTIC_VAL_REFERENCE)

MIN_RUN_PHYSICAL_VALIDITY = 0.90
MIN_RUN_QUERY_VALIDITY = 0.95
MAX_RUN_Z_ABS_P99 = 10.0
MAX_RUN_Z_ABS_MAX = 50.0
MIN_STABLE_SEEDS_PER_CONFIG = 2

TEST_MIN_PHYSICAL_VALIDITY = 0.90
TEST_MIN_QUERY_VALIDITY = 0.95
TEST_MIN_BOTH_BRANCH_SUPPORT_K16 = 0.20

R2_SOURCE_PATHS = (
    "ccda_phase3/phase314b_r2_contract.py",
    "ccda_phase3/phase314b_r2_diffusion.py",
    "ccda_phase3/phase314b_r2_metrics.py",
    "scripts/phase3_14b_r2_preflight.py",
    "scripts/phase3_14b_r2_smoke.py",
    "scripts/phase3_14b_r2_train.py",
    "scripts/phase3_14b_r2_select.py",
    "scripts/phase3_14b_r2_eval.py",
    "scripts/phase3_14b_r2_analyze.py",
    "scripts/phase3_14b_r2_run.sh",
)


@dataclass(frozen=True)
class RepairConfig:
    name: str
    prediction_type: str
    schedule_kind: str
    num_train_timesteps: int = 100
    max_beta: Optional[float] = None

    def validate(self) -> None:
        if self.prediction_type not in {
            "epsilon",
            "sample",
            "v_prediction",
        }:
            raise ValueError(
                f"unsupported prediction_type: {self.prediction_type}"
            )
        if self.schedule_kind not in {
            "cosine_original",
            "cosine_capped",
        }:
            raise ValueError(
                f"unsupported schedule_kind: {self.schedule_kind}"
            )
        if self.schedule_kind == "cosine_capped":
            if self.max_beta is None:
                raise ValueError("capped cosine requires max_beta")
            if not 0.0 < float(self.max_beta) < 0.999:
                raise ValueError("capped max_beta must lie in (0,0.999)")
        elif self.max_beta is not None:
            raise ValueError("original cosine must not set max_beta")


REPAIR_CONFIGS = {
    "epsilon_cosine_cap_0p5": RepairConfig(
        name="epsilon_cosine_cap_0p5",
        prediction_type="epsilon",
        schedule_kind="cosine_capped",
        max_beta=0.5,
    ),
    "sample_cosine": RepairConfig(
        name="sample_cosine",
        prediction_type="sample",
        schedule_kind="cosine_original",
    ),
    "v_prediction_cosine": RepairConfig(
        name="v_prediction_cosine",
        prediction_type="v_prediction",
        schedule_kind="cosine_original",
    ),
}
for _config in REPAIR_CONFIGS.values():
    _config.validate()


def cosine_betas(
    num_train_timesteps: int,
    *,
    max_beta: float,
) -> np.ndarray:
    def alpha_bar(value: float) -> float:
        return math.cos(
            (value + 0.008) / 1.008 * math.pi / 2.0
        ) ** 2

    betas = []
    count = int(num_train_timesteps)
    for index in range(count):
        t1 = index / count
        t2 = (index + 1) / count
        betas.append(
            min(
                1.0 - alpha_bar(t2) / alpha_bar(t1),
                float(max_beta),
            )
        )
    value = np.asarray(betas, dtype=np.float32)
    if value.shape != (count,):
        raise RuntimeError("cosine beta shape mismatch")
    if not np.all(np.isfinite(value)):
        raise RuntimeError("cosine betas are non-finite")
    if np.any(value <= 0.0) or np.any(value >= 1.0):
        raise RuntimeError("cosine betas must lie in (0,1)")
    return value


def schedule_stats_from_betas(betas: np.ndarray) -> Dict[str, Any]:
    value = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - value
    alpha_bar = np.cumprod(alphas)
    terminal = float(alpha_bar[-1])
    return {
        "num_train_timesteps": int(value.size),
        "beta_min": float(np.min(value)),
        "beta_max": float(np.max(value)),
        "terminal_alpha_bar": terminal,
        "terminal_signal_coefficient": float(math.sqrt(terminal)),
        "terminal_noise_coefficient": float(
            math.sqrt(max(0.0, 1.0 - terminal))
        ),
        "epsilon_x0_error_amplification": float(
            1.0 / math.sqrt(max(terminal, np.finfo(np.float64).tiny))
        ),
        "betas_sha256": hashlib.sha256(
            np.ascontiguousarray(value.astype(np.float32)).tobytes()
        ).hexdigest(),
    }


def default_linear_100_stats() -> Dict[str, Any]:
    betas = np.linspace(
        0.0001,
        0.02,
        100,
        dtype=np.float64,
    )
    result = schedule_stats_from_betas(betas)
    result["excluded_from_matrix"] = True
    result["reason"] = (
        "terminal signal coefficient is too large for sampling from "
        "an unconditional N(0,I) terminal prior"
    )
    return result


def source_sha256(
    root: Path,
    relative_paths: Sequence[str] = R2_SOURCE_PATHS,
) -> Dict[str, str]:
    output = {}
    for relative in relative_paths:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = sha256_file(path)
    return output


def load_r2_inputs(
    root: Path,
    *,
    cache_relative: str = (
        "data/phase3_14_cache/phase3_14a_training_cache.npz"
    ),
    manifest_relative: str = (
        "data/phase3_14_cache/"
        "phase3_14a_training_cache_manifest.json"
    ),
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], np.ndarray, Standardizer]:
    arrays, manifest = load_locked_cache(
        Path(root) / cache_relative,
        Path(root) / manifest_relative,
    )
    x_raw, x_std = input_values_and_standardizer(
        arrays,
        "paper_state",
    )
    if x_raw.shape != (CACHE_ROWS, PAPER_X_DIM):
        raise RuntimeError("paper_state input shape changed")
    return arrays, manifest, x_raw, x_std


def train_indices(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    selected = split_mask(
        arrays,
        "train",
        full_horizon_only=True,
        pre_engagement_only=False,
    )
    indices = np.flatnonzero(selected).astype(np.int64)
    if indices.size != 1000:
        raise RuntimeError(
            f"formal full-horizon train row count {indices.size} != 1000"
        )
    return indices


def validation_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    max_pair_keys: int = 64,
) -> np.ndarray:
    indices = fixed_balanced_eval_indices(
        arrays,
        split="val",
        max_pair_keys=int(max_pair_keys),
    )
    if indices.size != 102:
        raise RuntimeError(
            f"formal validation row count {indices.size} != 102"
        )
    return indices


def test_indices_and_pairs(
    arrays: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    selected = split_mask(
        arrays,
        "test",
        full_horizon_only=True,
        pre_engagement_only=True,
    )
    pairs = paired_key_index(arrays, selected)
    if len(pairs) != 103:
        raise RuntimeError(
            f"formal test pair count {len(pairs)} != 103"
        )
    indices = np.asarray(
        [
            index
            for key in sorted(pairs)
            for index in pairs[key]
        ],
        dtype=np.int64,
    )
    if indices.size != 206:
        raise RuntimeError("formal test query row count != 206")
    return indices, pairs


def strict_checkpoint_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def require_r11_supported(root: Path) -> Dict[str, Any]:
    summary = strict_json_load(
        Path(root) / "reports/phase3_14b_r11_summary.json"
    )
    if summary.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14b-r1.1 is not PASS")
    if (
        summary.get("root_cause")
        != "phase314b_r1_cosine_epsilon_terminal_snr_instability_supported"
    ):
        raise RuntimeError("unexpected r1.1 root cause")
    return summary
