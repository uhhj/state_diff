"""Phase3.14b-r2.1 validity calibration and geometry-audit contract."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from ccda_phase3.phase314a_contract import (
    BEAD_XY_DIM,
    CACHE_ROWS,
    N_BEADS,
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_contract import (
    CACHE_SHA256,
    future_standardizer,
)
from ccda_phase3.phase314b_r2_contract import (
    REPAIR_CONFIGS,
    SUBMODULE_COMMIT,
    load_r2_inputs,
    train_indices,
    validation_indices,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, PAPER_X_DIM, STATE_DIM


PHASE = "phase3_14b_r21"
BASE_MAIN_COMMIT = "52fbd36a390b8fdf898a08ad17229180f8bd15aa"
R2_IMPLEMENTATION_COMMIT = "329744bc3fa27f8d48d1afe60651e78a2aaf87bf"

R2_FAILURE_ROOT = "phase314b_r2_no_stable_configuration"
R11_SUPPORTED_ROOT = (
    "phase314b_r1_cosine_epsilon_terminal_snr_instability_supported"
)

EXPECTED_RUN_COUNT = 9
EXPECTED_VAL_ROWS = 102
EXPECTED_TRAIN_ROWS = 1000
EXPECTED_VAL_PAIR_KEYS = 51
K_AUDIT = 8

CALIBRATION_ALPHA = 0.01
ROBUST_SCALE_FLOOR_METERS = 1e-4
COORDINATE_SCALE_FLOOR_METERS = 1e-3
SEGMENT_GROSS_STRETCH_RATIO = 4.0
SEGMENT_GROSS_COMPRESSION_RATIO = 0.25

ORIGINAL_GT_TRAIN_MIN = 0.95
ORIGINAL_GT_VAL_MIN = 0.90
CALIBRATED_GT_VAL_MIN = 0.90

R21_SOURCE_PATHS = (
    "ccda_phase3/phase314b_r21_contract.py",
    "ccda_phase3/phase314b_r21_geometry.py",
    "scripts/phase3_14b_r21_preflight.py",
    "scripts/phase3_14b_r21_audit.py",
    "scripts/phase3_14b_r21_analyze.py",
    "scripts/phase3_14b_r21_run.sh",
)


def source_sha256(
    root: Path,
    relative_paths: Sequence[str] = R21_SOURCE_PATHS,
) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for relative in relative_paths:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = sha256_file(path)
    return output


def require_base_reports(root: Path) -> Dict[str, Any]:
    root = Path(root)
    r2 = strict_json_load(root / "reports/phase3_14b_r2_summary.json")
    if r2.get("verdict") != "FAIL":
        raise RuntimeError("Phase3.14b-r2 must be closed as FAIL")
    if r2.get("root_cause") != R2_FAILURE_ROOT:
        raise RuntimeError("unexpected Phase3.14b-r2 root cause")

    selection = strict_json_load(
        root / "reports/phase3_14b_r2_selection_summary.json"
    )
    if selection.get("verdict") != "FAIL":
        raise RuntimeError("r2 selection must remain FAIL")
    if selection.get("selected_model") is not None:
        raise RuntimeError("r2 must not have a selected model")
    if selection.get("test_evaluation_allowed") is not False:
        raise RuntimeError("formal test must remain blocked")

    r11 = strict_json_load(root / "reports/phase3_14b_r11_summary.json")
    if r11.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14b-r1.1 is not PASS")
    if r11.get("root_cause") != R11_SUPPORTED_ROOT:
        raise RuntimeError("unexpected Phase3.14b-r1.1 root cause")

    training = strict_json_load(
        root / "reports/phase3_14b_r2_training_summary.json"
    )
    if training.get("verdict") != "PASS":
        raise RuntimeError("r2 training did not complete")
    if int(training.get("run_count", -1)) != EXPECTED_RUN_COUNT:
        raise RuntimeError("r2 run count changed")
    if training.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("r2 training used another cache")

    return {
        "r2": r2,
        "r2_selection": selection,
        "r11": r11,
        "training": training,
    }


def verify_r2_checkpoints(
    root: Path,
    training_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    root = Path(root)
    verified = []
    for run in training_summary["runs"]:
        checkpoint = root / str(run["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual = sha256_file(checkpoint)
        if actual != run["checkpoint_sha256"]:
            raise RuntimeError(
                f"r2 checkpoint SHA256 mismatch: {checkpoint}"
            )
        verified.append(
            {
                "repair_config": str(run["repair_config"]),
                "training_seed": int(run["training_seed"]),
                "checkpoint": str(checkpoint.relative_to(root)),
                "checkpoint_sha256": actual,
            }
        )
    return {
        "count": len(verified),
        "runs": verified,
    }


def train_visible_seed_partition(
    arrays: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split train rows by visible seed, never by individual window."""
    seeds = np.asarray(arrays["visible_seed"][train_rows], dtype=np.int64)
    unique = np.unique(seeds)
    if unique.size < 20:
        raise RuntimeError("not enough visible seeds for calibration split")
    unique = np.sort(unique)
    fit_seeds = unique[::2]
    calibration_seeds = unique[1::2]
    fit_mask = np.isin(seeds, fit_seeds)
    calibration_mask = np.isin(seeds, calibration_seeds)
    if np.any(fit_mask & calibration_mask):
        raise RuntimeError("fit/calibration seed overlap")
    if not np.all(fit_mask | calibration_mask):
        raise RuntimeError("unassigned train rows")
    return train_rows[fit_mask], train_rows[calibration_mask]


def fixed_validation_rows(
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    rows = validation_indices(arrays)
    if rows.size != EXPECTED_VAL_ROWS:
        raise RuntimeError("formal validation row count changed")
    pair_keys = np.unique(np.asarray(arrays["pair_key"][rows]).astype(str))
    if pair_keys.size != EXPECTED_VAL_PAIR_KEYS:
        raise RuntimeError("formal validation pair count changed")
    return rows


def observed_last_repeat(
    paper_x: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    values = np.asarray(paper_x, dtype=np.float32)
    if values.shape != (CACHE_ROWS, PAPER_X_DIM):
        raise RuntimeError("paper_x shape changed")
    history = values[rows].reshape(-1, 3, STATE_DIM)
    last = history[:, -1, :]
    return np.repeat(last[:, None, :], DEFAULT_TF, axis=1)


def conformal_quantile(
    values: np.ndarray,
    *,
    alpha: float,
) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("conformal quantile requires finite values")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    rank = int(math.ceil((array.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), array.size)
    return float(np.sort(array)[rank - 1])


def strict_npz_save(path: Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and os.environ.get(
        "PHASE314B_R21_ALLOW_POOL_REPLACE"
    ) != "1":
        raise RuntimeError(f"refusing to replace existing pool: {target}")
    temporary = target.with_suffix(".tmp.npz")
    checked = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object:
            raise ValueError(f"object array forbidden: {name}")
        if np.issubdtype(array.dtype, np.number):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"non-finite array: {name}")
        checked[name] = array
    np.savez_compressed(temporary, **checked)
    os.replace(temporary, target)
