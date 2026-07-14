"""Shared-prior determinism and functional-equivalence audit for r2.5.4.

The original r2.5.4 runner treated a byte-identical SHA256 of freshly trained
floating-point parameters as a cross-run and cross-device scientific gate.
This module separates three contracts:

1. exact in-memory/snapshot integrity;
2. repeated same-device functional reproducibility;
3. historical cross-device functional equivalence.

No robot-proxy attribution is interpreted here.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import itertools
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import zlib

import numpy as np
import torch

PHASE = "phase3_14b_r254_resume1"
SCHEMA = "phase314b_r254_resume1_prior_determinism_v1"
WORKER_SCHEMA = "phase314b_r254_resume1_prior_worker_v1"
BASE_BLOCKED_REPORT_COMMIT = "d75988089ae6ba5e3fe65e44fae28eadcf02cfdc"
BASE_IMPLEMENTATION_COMMIT = "d315ac52f193eb6c0d5fc195b2cfa229dc7391e1"
BASE_R253_REPORT_COMMIT = "c35c374f61f9b4dc2ca4cdaa76b12ddda71be994"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R253_PILOT_SHA256 = (
    "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
)
EXPECTED_HISTORICAL_PRIOR_SHA256 = (
    "083278d6b0710ddc3863d822978973f29842bc757edf73ff03a93bb154f5665d"
)
EXPECTED_PAIRED_ROWS = (
    6, 1226, 10, 1230, 19, 1239, 25, 1245,
    31, 1251, 36, 1256, 43, 1263, 49, 1269,
)
COMMON_PAIRED_PRIOR_SEED = 102000
REPEAT_RUN_IDS = ("repeat_a", "repeat_b", "repeat_c")
EXPECTED_PYTHON = "/miniforge3/envs/coord_bimanual/bin/python"

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r254_resume1_prior_determinism.py",
    "scripts/phase3_14b_r254_resume1_prior_worker.py",
    "scripts/phase3_14b_r254_resume1_preflight.py",
    "scripts/phase3_14b_r254_resume1_run_audit.py",
    "scripts/phase3_14b_r254_resume1_finalize.py",
    "scripts/phase3_14b_r254_resume1_run.sh",
    "tests/test_phase314b_r254_resume1_prior_determinism.py",
)

IMMUTABLE_R254_PATHS = (
    "ccda_phase3/phase314b_r254_robot_proxy_attribution.py",
    "scripts/phase3_14b_r254_preflight.py",
    "scripts/phase3_14b_r254_run_pilot.py",
    "scripts/phase3_14b_r254_finalize.py",
    "scripts/phase3_14b_r254_run.sh",
    "tests/test_phase314b_r254_robot_proxy_attribution.py",
    "reports/phase3_14b_r254_preflight_summary.json",
    "reports/phase3_14b_r254_blocked_summary.json",
    "reports/phase3_14b_r254_blocked_report.md",
)


@dataclass(frozen=True)
class PriorEquivalenceSpec:
    """Frozen diagnostic tolerances for the prior audit.

    Same-device prediction tolerances are intentionally strict because all
    repeated fits use the same code, seed, process-level environment and GPU.
    Historical tolerances are looser because the committed prior was produced
    on a different GPU model and only scalar/history fingerprints were saved.
    """

    repeat_count: int = 3
    same_device_prediction_max_abs: float = 1.0e-6
    same_device_prediction_rmse: float = 1.0e-7
    same_device_prior_z_mse_abs: float = 1.0e-8
    same_device_prior_z_mse_relative: float = 1.0e-6
    same_device_parameter_max_abs: float = 1.0e-6
    same_device_parameter_rmse: float = 1.0e-7
    historical_prior_z_mse_relative: float = 5.0e-3
    historical_final_loss_relative: float = 5.0e-3
    historical_loss_curve_p95_relative: float = 1.0e-2

    def validate(self) -> None:
        if self.repeat_count < 2:
            raise ValueError("repeat_count must be at least two")
        for key, value in asdict(self).items():
            if key == "repeat_count":
                continue
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"invalid tolerance {key}: {value}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    return {path: sha256_file(Path(root) / path) for path in SOURCE_PATHS}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=Path(root),
        stderr=subprocess.STDOUT,
    )


def assert_paths_match_commit(
    root: Path,
    *,
    commit: str,
    paths: Sequence[str],
) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    mismatch: Dict[str, Dict[str, str]] = {}
    for name in paths:
        path = Path(root) / name
        if not path.is_file():
            raise RuntimeError(f"immutable path is missing: {name}")
        current = path.read_bytes()
        historical = git_blob_bytes(root, commit, name)
        current_sha = hashlib.sha256(current).hexdigest()
        historical_sha = hashlib.sha256(historical).hexdigest()
        observed[name] = current_sha
        if current_sha != historical_sha:
            mismatch[name] = {
                "current": current_sha,
                "historical": historical_sha,
            }
    if mismatch:
        raise RuntimeError(f"historical r2.5.4 evidence changed: {mismatch}")
    return observed


def assert_only_allowed_worktree_paths(root: Path, allowed_paths: Sequence[str]) -> None:
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
        raise RuntimeError("unexpected worktree changes: " + ", ".join(unexpected))


def _array_payload(array: np.ndarray) -> Dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array))
    compressed = zlib.compress(value.tobytes(order="C"), level=9)
    return {
        "dtype": value.dtype.str,
        "shape": [int(item) for item in value.shape],
        "data_b64_zlib": base64.b64encode(compressed).decode("ascii"),
        "raw_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _array_from_payload(payload: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(str(payload["dtype"]))
    shape = tuple(int(item) for item in payload["shape"])
    compressed = base64.b64decode(str(payload["data_b64_zlib"]).encode("ascii"))
    raw = zlib.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != payload.get("raw_sha256"):
        raise ValueError("array payload SHA mismatch")
    expected_size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    if len(raw) != expected_size:
        raise ValueError("array payload byte length mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def encode_tensor_state(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for key in sorted(state):
        value = torch.as_tensor(state[key]).detach().cpu().contiguous().numpy()
        tensors[str(key)] = _array_payload(value)
    return {"keys": list(tensors), "tensors": tensors}


def decode_tensor_state(payload: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    keys = [str(value) for value in payload.get("keys", [])]
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or keys != sorted(tensors):
        raise ValueError("invalid tensor-state payload keys")
    return {key: _array_from_payload(tensors[key]) for key in keys}


def encode_prediction(array: np.ndarray) -> Dict[str, Any]:
    return _array_payload(np.asarray(array, dtype=np.float32))


def decode_prediction(payload: Mapping[str, Any]) -> np.ndarray:
    return _array_from_payload(payload).astype(np.float32, copy=False)


def tensor_state_summary(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    rows = []
    total_count = 0
    total_squared = 0.0
    for key in sorted(state):
        value = torch.as_tensor(state[key]).detach().cpu().to(torch.float64)
        count = int(value.numel())
        total_count += count
        total_squared += float(torch.sum(value.square()))
        rows.append({
            "key": str(key),
            "shape": [int(item) for item in value.shape],
            "dtype": str(state[key].dtype),
            "count": count,
            "mean": float(torch.mean(value)) if count else 0.0,
            "std": float(torch.std(value, unbiased=False)) if count else 0.0,
            "minimum": float(torch.min(value)) if count else 0.0,
            "maximum": float(torch.max(value)) if count else 0.0,
            "l2_norm": float(torch.linalg.vector_norm(value)) if count else 0.0,
        })
    return {
        "tensor_count": len(rows),
        "parameter_count": total_count,
        "l2_norm": math.sqrt(total_squared),
        "tensors": rows,
    }


def _finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def compare_tensor_states(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    spec: PriorEquivalenceSpec,
) -> Dict[str, Any]:
    spec.validate()
    if set(first) != set(second):
        return {
            "key_match": False,
            "missing_from_first": sorted(set(second) - set(first)),
            "missing_from_second": sorted(set(first) - set(second)),
            "exact_equal": False,
            "functional_parameter_bound_pass": False,
        }
    squared_sum = 0.0
    reference_squared_sum = 0.0
    count = 0
    max_abs = 0.0
    exact = True
    per_tensor = []
    for key in sorted(first):
        a = _finite_array(first[key], name=f"first[{key}]")
        b = _finite_array(second[key], name=f"second[{key}]")
        if a.shape != b.shape:
            raise ValueError(f"state shape mismatch for {key}: {a.shape} != {b.shape}")
        diff = a - b
        item_max = float(np.max(np.abs(diff))) if diff.size else 0.0
        item_rmse = float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0
        max_abs = max(max_abs, item_max)
        squared_sum += float(np.sum(np.square(diff)))
        reference_squared_sum += float(np.sum(np.square(a)))
        count += int(diff.size)
        exact = bool(exact and np.array_equal(a, b))
        per_tensor.append({
            "key": key,
            "max_abs": item_max,
            "rmse": item_rmse,
        })
    rmse = math.sqrt(squared_sum / max(count, 1))
    relative_l2 = math.sqrt(squared_sum) / max(math.sqrt(reference_squared_sum), 1.0e-12)
    per_tensor.sort(key=lambda item: item["max_abs"], reverse=True)
    return {
        "key_match": True,
        "exact_equal": exact,
        "parameter_count": count,
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "largest_tensor_differences": per_tensor[:10],
        "functional_parameter_bound_pass": bool(
            max_abs <= spec.same_device_parameter_max_abs
            and rmse <= spec.same_device_parameter_rmse
        ),
    }


def compare_predictions(
    first: np.ndarray,
    second: np.ndarray,
    *,
    spec: PriorEquivalenceSpec,
) -> Dict[str, Any]:
    spec.validate()
    a = _finite_array(first, name="first prediction")
    b = _finite_array(second, name="second prediction")
    if a.shape != b.shape:
        raise ValueError(f"prediction shape mismatch: {a.shape} != {b.shape}")
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    return {
        "shape": [int(item) for item in a.shape],
        "exact_equal": bool(np.array_equal(a, b)),
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_l2": float(
            np.linalg.norm(diff.reshape(-1))
            / max(np.linalg.norm(a.reshape(-1)), 1.0e-12)
        ),
        "pass": bool(
            max_abs <= spec.same_device_prediction_max_abs
            and rmse <= spec.same_device_prediction_rmse
        ),
    }


def symmetric_relative_error(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(
        abs(float(observed)), abs(float(expected)), 1.0e-12
    )


def compare_prior_z_mse(
    first: float,
    second: float,
    *,
    spec: PriorEquivalenceSpec,
) -> Dict[str, Any]:
    absolute = abs(float(first) - float(second))
    relative = symmetric_relative_error(first, second)
    return {
        "first": float(first),
        "second": float(second),
        "absolute_error": absolute,
        "relative_error": relative,
        "pass": bool(
            absolute <= spec.same_device_prior_z_mse_abs
            or relative <= spec.same_device_prior_z_mse_relative
        ),
    }


def _history_by_step(history: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    result: Dict[int, Mapping[str, Any]] = {}
    for row in history:
        step = int(row["step"])
        if step in result:
            raise ValueError(f"duplicate prior-history step: {step}")
        loss = float(row["loss"])
        gradient = float(row.get("gradient_norm", float("nan")))
        if not math.isfinite(loss) or not math.isfinite(gradient):
            raise ValueError("prior history contains non-finite values")
        result[step] = row
    if not result:
        raise ValueError("prior history is empty")
    return result


def compare_histories(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    a = _history_by_step(first)
    b = _history_by_step(second)
    if set(a) != set(b):
        return {
            "step_match": False,
            "first_steps": sorted(a),
            "second_steps": sorted(b),
            "pass": False,
        }
    rows = []
    for step in sorted(a):
        rows.append({
            "step": step,
            "loss_relative_error": symmetric_relative_error(
                float(a[step]["loss"]), float(b[step]["loss"])
            ),
            "gradient_relative_error": symmetric_relative_error(
                float(a[step]["gradient_norm"]),
                float(b[step]["gradient_norm"]),
            ),
        })
    loss_errors = np.asarray([row["loss_relative_error"] for row in rows])
    gradient_errors = np.asarray([row["gradient_relative_error"] for row in rows])
    return {
        "step_match": True,
        "steps": [row["step"] for row in rows],
        "loss_relative_error_p50": float(np.percentile(loss_errors, 50)),
        "loss_relative_error_p95": float(np.percentile(loss_errors, 95)),
        "loss_relative_error_max": float(np.max(loss_errors)),
        "final_loss_relative_error": float(rows[-1]["loss_relative_error"]),
        "gradient_relative_error_p50": float(np.percentile(gradient_errors, 50)),
        "gradient_relative_error_p95": float(np.percentile(gradient_errors, 95)),
        "gradient_relative_error_max": float(np.max(gradient_errors)),
        "rows": rows,
    }


def _pair_name(first: str, second: str) -> str:
    return f"{first}__vs__{second}"


def pairwise_repeat_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    spec: PriorEquivalenceSpec,
) -> Dict[str, Any]:
    spec.validate()
    if len(records) != spec.repeat_count:
        raise ValueError(
            f"expected {spec.repeat_count} repeat records, got {len(records)}"
        )
    ids = [str(record["run_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("repeat run IDs are not unique")
    comparisons: Dict[str, Any] = {}
    for first_index, second_index in itertools.combinations(range(len(records)), 2):
        first = records[first_index]
        second = records[second_index]
        name = _pair_name(str(first["run_id"]), str(second["run_id"]))
        state = compare_tensor_states(
            decode_tensor_state(first["_state_payload"]),
            decode_tensor_state(second["_state_payload"]),
            spec=spec,
        )
        prediction = compare_predictions(
            decode_prediction(first["_prediction_payload"]),
            decode_prediction(second["_prediction_payload"]),
            spec=spec,
        )
        prior_z = compare_prior_z_mse(
            float(first["prior_z_mse"]),
            float(second["prior_z_mse"]),
            spec=spec,
        )
        history = compare_histories(first["prior_history"], second["prior_history"])
        history_pass = bool(
            history.get("step_match")
            and float(history["loss_relative_error_p95"])
            <= spec.historical_loss_curve_p95_relative
        )
        comparisons[name] = {
            "state": state,
            "prediction": prediction,
            "prior_z_mse": prior_z,
            "history": history,
            "history_pass": history_pass,
            "exact_state_sha_match": bool(
                first["prior_state_sha256"] == second["prior_state_sha256"]
            ),
            "pass": bool(
                prediction["pass"]
                and prior_z["pass"]
                and history_pass
                and state.get("functional_parameter_bound_pass", False)
            ),
        }
    hashes = [str(record["prior_state_sha256"]) for record in records]
    prediction_hashes = [str(record["prior_prediction_sha256"]) for record in records]
    return {
        "run_ids": ids,
        "observed_state_sha256": hashes,
        "unique_state_sha256_count": len(set(hashes)),
        "all_state_sha_exact": len(set(hashes)) == 1,
        "observed_prediction_sha256": prediction_hashes,
        "unique_prediction_sha256_count": len(set(prediction_hashes)),
        "all_prediction_sha_exact": len(set(prediction_hashes)) == 1,
        "pairwise": comparisons,
        "functional_equivalence_pass": bool(
            comparisons and all(value["pass"] for value in comparisons.values())
        ),
    }


def historical_functional_fingerprint(
    records: Sequence[Mapping[str, Any]],
    historical: Mapping[str, Any],
    *,
    spec: PriorEquivalenceSpec,
) -> Dict[str, Any]:
    spec.validate()
    expected_mse = float(historical["prior_z_mse"])
    expected_history = historical["prior_history"]
    current_mse_values = np.asarray(
        [float(record["prior_z_mse"]) for record in records], dtype=np.float64
    )
    current_mse = float(np.median(current_mse_values))
    mse_relative = symmetric_relative_error(current_mse, expected_mse)

    expected_by_step = _history_by_step(expected_history)
    current_by_run = [_history_by_step(record["prior_history"]) for record in records]
    if any(set(value) != set(expected_by_step) for value in current_by_run):
        return {
            "history_step_match": False,
            "prior_z_mse_expected": expected_mse,
            "prior_z_mse_observed_median": current_mse,
            "prior_z_mse_relative_error": mse_relative,
            "functional_fingerprint_pass": False,
        }
    rows = []
    for step in sorted(expected_by_step):
        current_losses = [float(value[step]["loss"]) for value in current_by_run]
        current_gradients = [
            float(value[step]["gradient_norm"]) for value in current_by_run
        ]
        observed_loss = float(np.median(current_losses))
        observed_gradient = float(np.median(current_gradients))
        rows.append({
            "step": step,
            "expected_loss": float(expected_by_step[step]["loss"]),
            "observed_loss_median": observed_loss,
            "loss_relative_error": symmetric_relative_error(
                observed_loss, float(expected_by_step[step]["loss"])
            ),
            "expected_gradient_norm": float(
                expected_by_step[step]["gradient_norm"]
            ),
            "observed_gradient_norm_median": observed_gradient,
            "gradient_relative_error": symmetric_relative_error(
                observed_gradient,
                float(expected_by_step[step]["gradient_norm"]),
            ),
        })
    loss_errors = np.asarray([row["loss_relative_error"] for row in rows])
    final_error = float(rows[-1]["loss_relative_error"])
    p95_error = float(np.percentile(loss_errors, 95))
    exact_match_count = sum(
        str(record["prior_state_sha256"])
        == str(historical["prior_state_sha256"])
        for record in records
    )
    return {
        "history_step_match": True,
        "expected_prior_state_sha256": str(historical["prior_state_sha256"]),
        "observed_state_sha256": [
            str(record["prior_state_sha256"]) for record in records
        ],
        "exact_state_sha_match_count": int(exact_match_count),
        "prior_z_mse_expected": expected_mse,
        "prior_z_mse_observed_values": current_mse_values.tolist(),
        "prior_z_mse_observed_median": current_mse,
        "prior_z_mse_relative_error": mse_relative,
        "loss_curve_relative_error_p50": float(np.percentile(loss_errors, 50)),
        "loss_curve_relative_error_p95": p95_error,
        "loss_curve_relative_error_max": float(np.max(loss_errors)),
        "final_loss_relative_error": final_error,
        "rows": rows,
        "functional_fingerprint_pass": bool(
            mse_relative <= spec.historical_prior_z_mse_relative
            and final_error <= spec.historical_final_loss_relative
            and p95_error <= spec.historical_loss_curve_p95_relative
        ),
    }


def strip_worker_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"_state_payload", "_prediction_payload"}
    }


def classify_prior_audit(report: Mapping[str, Any]) -> Dict[str, Any]:
    repeat = report.get("same_device_repeats", {})
    historical = report.get("historical_functional_fingerprint", {})
    records = report.get("runs", [])
    if not isinstance(records, Sequence) or len(records) != len(REPEAT_RUN_IDS):
        root = "phase314b_r254_resume1_prior_repeat_matrix_incomplete"
        next_stage = "repair_prior_repeat_execution"
        functional = False
    elif not bool(repeat.get("functional_equivalence_pass")):
        root = "phase314b_r254_resume1_same_device_prior_functional_nondeterminism"
        next_stage = "strict_deterministic_prior_training_audit"
        functional = False
    elif not bool(historical.get("functional_fingerprint_pass")):
        root = "phase314b_r254_resume1_prior_functional_equivalence_not_supported"
        next_stage = "historical_prior_reproduction_debug"
        functional = False
    else:
        functional = True
        expected = str(historical.get("expected_prior_state_sha256"))
        observed = [str(value) for value in repeat.get("observed_state_sha256", [])]
        if expected in observed:
            root = "phase314b_r254_resume1_shared_prior_exact_reproduction_supported"
        elif bool(repeat.get("all_state_sha_exact")):
            current_gpu = str(report.get("environment", {}).get("gpu_name"))
            historical_gpu = str(
                report.get("historical_prior", {}).get("gpu_name")
            )
            if current_gpu and historical_gpu and current_gpu != historical_gpu:
                root = (
                    "phase314b_r254_resume1_cross_device_bitwise_sha_contract_overstrict"
                )
            else:
                root = (
                    "phase314b_r254_resume1_historical_bitwise_sha_contract_overstrict"
                )
        else:
            root = (
                "phase314b_r254_resume1_bitwise_nondeterminism_functionally_bounded"
            )
        next_stage = (
            "phase3_14b_r254_resume2_robot_proxy_attribution_"
            "functional_prior_contract"
        )
    return {
        "root_cause": root,
        "next_stage": next_stage,
        "functional_prior_contract_supported": functional,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def tensor_state_sha256_numpy(state: Mapping[str, np.ndarray]) -> str:
    """Recompute the canonical r2.5.1 tensor-state hash from NumPy arrays."""
    digest = hashlib.sha256()
    for key in sorted(state):
        array = np.ascontiguousarray(np.asarray(state[key]))
        tensor_dtype = str(torch.from_numpy(array).dtype)
        digest.update(str(key).encode("utf-8"))
        digest.update(tensor_dtype.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
