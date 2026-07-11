#!/usr/bin/env python3
"""Strict serialization, hashing, and grouped-split helpers for Phase3.12d-r2.2."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np


def np_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return np_scalar(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_text(value: Any) -> str:
    value = np_scalar(value)
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _jsonable(value: Any) -> Any:
    value = np_scalar(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported strict JSON value: {type(value).__name__}")


def ensure_finite_tree(value: Any, path: str = "root") -> None:
    value = np_scalar(value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite numeric array at {path}")
        for index, item in enumerate(value.tolist()):
            ensure_finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {path}: {value}")


def strict_json_dump(path: Path, payload: Any) -> None:
    ensure_finite_tree(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def stable_episode_key(condition: Any, visible_seed: Any, source_file: Any) -> str:
    return "|".join(
        [decode_text(condition), str(int(np_scalar(visible_seed))), decode_text(source_file)]
    )


def stable_window_hash(
    paper_x: np.ndarray,
    state_action_x: np.ndarray,
    y_state: np.ndarray,
    y_action: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (paper_x, state_action_x, y_state, y_action):
        digest.update(sha256_array(np.asarray(value)).encode("ascii"))
    return digest.hexdigest()


def group_bootstrap_indices(
    groups: Sequence[Any], n_bootstrap: int, seed: int
) -> List[np.ndarray]:
    group_array = np.asarray(groups)
    unique = np.unique(group_array)
    if unique.size == 0:
        raise ValueError("cannot bootstrap empty groups")
    rng = np.random.default_rng(int(seed))
    output: List[np.ndarray] = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        pieces = [np.flatnonzero(group_array == group) for group in sampled]
        output.append(np.concatenate(pieces).astype(np.int64))
    return output


def assert_disjoint_groups(train_groups: Iterable[Any], test_groups: Iterable[Any]) -> None:
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        raise AssertionError(f"train/test group overlap: {sorted(overlap)[:10]}")


def grouped_fold_indices(groups: Sequence[int], folds: int, repeat: int) -> List[tuple]:
    group_array = np.asarray(groups, dtype=np.int64)
    unique = np.unique(group_array)
    if int(folds) < 2 or unique.size < int(folds):
        raise ValueError("not enough unique groups for grouped folds")
    rng = np.random.default_rng(312200 + int(repeat))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, int(folds))
    result = []
    for test_groups in chunks:
        test = np.flatnonzero(np.isin(group_array, test_groups))
        train = np.flatnonzero(~np.isin(group_array, test_groups))
        assert_disjoint_groups(group_array[train], group_array[test])
        result.append((train, test))
    return result


def binary_roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    positive = s[y == 1]
    negative = s[y == 0]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
