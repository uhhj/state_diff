"""Phase3.14b immutable contracts for DDPM training and support audit."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import (
    CACHE_ROWS,
    CACHE_VERSION,
    DEFAULT_TF,
    PAPER_X_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    Standardizer,
    load_npz_no_pickle,
    sha256_array,
    sha256_file,
    strict_json_dump,
    strict_json_load,
    validate_formal_windows,
)


PHASE = "phase3_14b"
CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
FAMILIES = ("mlp_ddpm", "temporal_unet_ddpm")
INPUT_VARIANTS = ("paper_state", "state_action")
TRAINING_SEEDS = (31421, 31422, 31423)
K_VALUES = (1, 4, 8, 16, 32)
FULL_HORIZON = DEFAULT_TF
BRANCH_SEPARATION_MIN_METERS = 0.003
BRANCH_SUPPORT_RADIUS_FRACTION = 0.5

MIN_FULL_TRAIN_ROWS = 512
MIN_FULL_VAL_PRE_ROWS = 64
MIN_ELIGIBLE_TEST_PAIR_KEYS = 32


@dataclass(frozen=True)
class InputSpec:
    name: str
    key: str
    mean_key: str
    scale_key: str
    active_key: str
    dimension: int


INPUT_SPECS = {
    "paper_state": InputSpec(
        name="paper_state",
        key="paper_x",
        mean_key="paper_x_mean",
        scale_key="paper_x_scale",
        active_key="paper_x_active",
        dimension=PAPER_X_DIM,
    ),
    "state_action": InputSpec(
        name="state_action",
        key="state_action_x",
        mean_key="state_action_x_mean",
        scale_key="state_action_x_scale",
        active_key="state_action_x_active",
        dimension=STATE_ACTION_X_DIM,
    ),
}


def source_sha256(root: Path, relative_paths: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in relative_paths:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def load_locked_cache(
    cache_path: Path,
    manifest_path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    cache = Path(cache_path)
    manifest = strict_json_load(Path(manifest_path))
    if manifest.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("unexpected Phase3.14a cache version")
    if manifest.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError(
            "cache manifest hash differs from the approved Phase3.14a hash"
        )
    actual_hash = sha256_file(cache)
    if actual_hash != CACHE_SHA256:
        raise RuntimeError(
            f"immutable cache changed: {actual_hash} != {CACHE_SHA256}"
        )
    arrays = load_npz_no_pickle(cache)
    validation = validate_formal_windows(arrays)
    if validation["rows"] != CACHE_ROWS:
        raise RuntimeError("immutable cache row count changed")
    for key, expected in manifest["array_sha256"].items():
        if key not in arrays:
            raise RuntimeError(f"cache is missing locked array {key}")
        if sha256_array(arrays[key]) != expected:
            raise RuntimeError(f"cache array hash mismatch: {key}")
    required = {
        "future_valid_mask": (CACHE_ROWS, DEFAULT_TF),
        "pair_key": (CACHE_ROWS,),
        "current_state": (CACHE_ROWS, STATE_DIM),
        "future_mean": (DEFAULT_TF, STATE_DIM),
        "future_scale": (DEFAULT_TF, STATE_DIM),
        "future_active": (DEFAULT_TF, STATE_DIM),
    }
    for key, shape in required.items():
        if key not in arrays or arrays[key].shape != shape:
            raise RuntimeError(
                f"cache array {key} shape is "
                f"{None if key not in arrays else arrays[key].shape}, "
                f"expected {shape}"
            )
    return arrays, manifest


def input_values_and_standardizer(
    arrays: Mapping[str, np.ndarray],
    variant: str,
) -> Tuple[np.ndarray, Standardizer]:
    if variant not in INPUT_SPECS:
        raise ValueError(f"unsupported input variant: {variant}")
    spec = INPUT_SPECS[variant]
    values = np.asarray(arrays[spec.key], dtype=np.float32)
    if values.shape != (CACHE_ROWS, spec.dimension):
        raise ValueError(f"{variant} shape mismatch: {values.shape}")
    standardizer = Standardizer(
        mean=np.asarray(arrays[spec.mean_key], dtype=np.float32),
        scale=np.asarray(arrays[spec.scale_key], dtype=np.float32),
        active=np.asarray(arrays[spec.active_key], dtype=np.bool_),
        raw_std=np.asarray(
            arrays[spec.mean_key.replace("_mean", "_raw_std")],
            dtype=np.float32,
        ),
    )
    return values, standardizer


def future_standardizer(
    arrays: Mapping[str, np.ndarray],
) -> Standardizer:
    return Standardizer(
        mean=np.asarray(arrays["future_mean"], dtype=np.float32),
        scale=np.asarray(arrays["future_scale"], dtype=np.float32),
        active=np.asarray(arrays["future_active"], dtype=np.bool_),
        raw_std=np.asarray(arrays["future_raw_std"], dtype=np.float32),
    )


def full_horizon_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    mask = np.asarray(arrays["future_valid_mask"], dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[1] != DEFAULT_TF:
        raise ValueError("future_valid_mask must be [N,4]")
    return np.all(mask, axis=1)


def split_mask(
    arrays: Mapping[str, np.ndarray],
    split: str,
    *,
    full_horizon_only: bool = True,
    pre_engagement_only: bool = False,
) -> np.ndarray:
    selected = np.asarray(arrays["split_name"]).astype(str) == str(split)
    if full_horizon_only:
        selected &= full_horizon_mask(arrays)
    if pre_engagement_only:
        selected &= np.asarray(arrays["pre_engagement"], dtype=np.bool_)
    return selected


def paired_key_index(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> Dict[str, Tuple[int, int]]:
    conditions = np.asarray(arrays["condition_name"]).astype(str)
    keys = np.asarray(arrays["pair_key"]).astype(str)
    groups: Dict[str, Dict[str, int]] = {}
    for index in np.flatnonzero(selected):
        groups.setdefault(keys[index], {})[conditions[index]] = int(index)
    expected = {"free", "hidden_slack_breakaway_pin_v2"}
    result: Dict[str, Tuple[int, int]] = {}
    for key, value in groups.items():
        if set(value) != expected:
            continue
        result[key] = (
            value["free"],
            value["hidden_slack_breakaway_pin_v2"],
        )
    return result


def fixed_balanced_eval_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    split: str,
    max_pair_keys: int,
) -> np.ndarray:
    selected = split_mask(
        arrays,
        split,
        full_horizon_only=True,
        pre_engagement_only=True,
    )
    pairs = paired_key_index(arrays, selected)
    keys = sorted(pairs)[: int(max_pair_keys)]
    indices = []
    for key in keys:
        indices.extend(pairs[key])
    if not indices:
        raise RuntimeError(f"no paired full-horizon rows in {split}")
    return np.asarray(indices, dtype=np.int64)


def validate_training_population(
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, int]:
    train_full = int(np.sum(split_mask(arrays, "train")))
    val_pre = int(
        np.sum(
            split_mask(
                arrays,
                "val",
                full_horizon_only=True,
                pre_engagement_only=True,
            )
        )
    )
    test_selected = split_mask(
        arrays,
        "test",
        full_horizon_only=True,
        pre_engagement_only=True,
    )
    test_pairs = len(paired_key_index(arrays, test_selected))
    if train_full < MIN_FULL_TRAIN_ROWS:
        raise RuntimeError(
            f"only {train_full} full-horizon training rows; "
            f"minimum is {MIN_FULL_TRAIN_ROWS}"
        )
    if val_pre < MIN_FULL_VAL_PRE_ROWS:
        raise RuntimeError(
            f"only {val_pre} full-horizon pre-engagement validation rows"
        )
    if test_pairs < MIN_ELIGIBLE_TEST_PAIR_KEYS:
        raise RuntimeError(
            f"only {test_pairs} eligible test pair keys"
        )
    return {
        "train_full_horizon_rows": train_full,
        "val_full_horizon_pre_engagement_rows": val_pre,
        "test_full_horizon_pre_engagement_pair_keys": test_pairs,
    }


def assert_no_forbidden_condition_features(
    arrays: Mapping[str, np.ndarray],
) -> None:
    # Formal feature dimensions are fixed by the Phase3.13 schema.  This
    # assertion catches accidental concatenation in derived Phase3.14b code.
    if arrays["paper_x"].shape[1] != PAPER_X_DIM:
        raise RuntimeError("paper_x dimension changed")
    if arrays["state_action_x"].shape[1] != STATE_ACTION_X_DIM:
        raise RuntimeError("state_action_x dimension changed")


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)
