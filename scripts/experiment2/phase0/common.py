"""Pure metrics and serialization helpers for Phase 0 paired rollouts."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import numpy as np


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resample_indices(length: int, target_length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if target_length == 1:
        return np.array([0], dtype=np.int64)
    indices = np.rint(np.linspace(0, length - 1, target_length)).astype(np.int64)
    return np.maximum.accumulate(np.clip(indices, 0, length - 1))


def phase_slice(trace: Dict[str, np.ndarray], phase: str) -> Dict[str, np.ndarray]:
    if "phase" not in trace:
        raise KeyError("trace has no phase array")
    phases = np.asarray(trace["phase"]).astype(str)
    mask = phases == phase
    return {
        key: np.asarray(value)[mask]
        for key, value in trace.items()
        if np.asarray(value).shape and np.asarray(value).shape[0] == phases.shape[0]
    }


def ordered_bead_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"expected matching [T,N,3] arrays, got {first.shape} and {second.shape}")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("bead arrays contain NaN or Inf")
    return np.mean(np.linalg.norm(first[..., :2] - second[..., :2], axis=-1), axis=-1)


def no_action_drift(beads: np.ndarray) -> float:
    values = np.asarray(beads, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[0] == 0:
        raise ValueError(f"expected non-empty [T,N,3] beads, got {values.shape}")
    displacement = np.linalg.norm(values[..., :2] - values[0:1, ..., :2], axis=-1)
    return float(np.max(displacement))


def _require_phase(trace: Dict[str, np.ndarray], phase: str) -> Dict[str, np.ndarray]:
    sliced = phase_slice(trace, phase)
    if "bead_positions" not in sliced or sliced["bead_positions"].shape[0] == 0:
        raise ValueError(f"trace has no bead frames for phase {phase!r}")
    return sliced


def _aligned_beads(
    free_phase: Dict[str, np.ndarray],
    hidden_phase: Dict[str, np.ndarray],
    limit: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    free = np.asarray(free_phase["bead_positions"], dtype=np.float64)
    hidden = np.asarray(hidden_phase["bead_positions"], dtype=np.float64)
    target = min(free.shape[0], hidden.shape[0], limit)
    if target <= 0:
        raise ValueError("cannot align empty phase traces")
    return free[resample_indices(free.shape[0], target)], hidden[
        resample_indices(hidden.shape[0], target)
    ]


def compute_pair_metrics(
    free_trace: Dict[str, np.ndarray],
    hidden_trace: Dict[str, np.ndarray],
    free_meta: Dict[str, Any],
    hidden_meta: Dict[str, Any],
    hz: float,
    trace_stride: int,
) -> Dict[str, Any]:
    if float(hz) <= 0 or int(trace_stride) <= 0:
        raise ValueError("hz and trace_stride must be positive")
    free_hash = free_meta.get("action_hash")
    hidden_hash = hidden_meta.get("action_hash")
    if not free_hash or free_hash != hidden_hash:
        raise ValueError("paired rollout action hashes do not match")

    free_initial = np.asarray(free_trace["bead_positions"], dtype=np.float64)[0]
    hidden_initial = np.asarray(hidden_trace["bead_positions"], dtype=np.float64)[0]
    if free_initial.shape != hidden_initial.shape:
        raise ValueError("paired rollout bead shapes do not match")

    free_no_action = _require_phase(free_trace, "no_action")
    hidden_no_action = _require_phase(hidden_trace, "no_action")
    free_preload = _require_phase(free_trace, "preload")
    hidden_preload = _require_phase(hidden_trace, "preload")
    free_main = _require_phase(free_trace, "main_pull")
    hidden_main = _require_phase(hidden_trace, "main_pull")
    free_post = _require_phase(free_trace, "post_main")
    hidden_post = _require_phase(hidden_trace, "post_main")

    free_main_beads, hidden_main_beads = _aligned_beads(free_main, hidden_main)
    main_distance = ordered_bead_distance(free_main_beads, hidden_main_beads)
    free_post_beads, hidden_post_beads = _aligned_beads(free_post, hidden_post)
    post_distance = ordered_bead_distance(free_post_beads, hidden_post_beads)
    dt = float(trace_stride) / float(hz)
    free_impulse = float(np.sum(np.asarray(free_preload["contact_force_norm"], dtype=np.float64)) * dt)
    hidden_impulse = float(
        np.sum(np.asarray(hidden_preload["contact_force_norm"], dtype=np.float64)) * dt
    )

    free_privileged = free_meta.get("privileged_state", {})
    hidden_privileged = hidden_meta.get("privileged_state", {})
    return {
        "action_hash_match": True,
        "initial_max_abs_xy": float(np.max(np.abs(free_initial[:, :2] - hidden_initial[:, :2]))),
        "preload_end_max_abs_xy": float(
            np.max(
                np.abs(
                    np.asarray(free_preload["bead_positions"][-1])[:, :2]
                    - np.asarray(hidden_preload["bead_positions"][-1])[:, :2]
                )
            )
        ),
        "free_no_action_drift": no_action_drift(free_no_action["bead_positions"]),
        "hidden_no_action_drift": no_action_drift(hidden_no_action["bead_positions"]),
        "free_preload_contact_impulse": free_impulse,
        "hidden_preload_contact_impulse": hidden_impulse,
        "contact_impulse_gap": hidden_impulse - free_impulse,
        "main_branch_ade": float(np.mean(main_distance)),
        "main_branch_fde": float(main_distance[-1]),
        "post_main_branch_fde": float(post_distance[-1]),
        "free_arm_max_abs_jump": float(free_privileged.get("arm_max_abs_jump", np.nan)),
        "hidden_arm_max_abs_jump": float(hidden_privileged.get("arm_max_abs_jump", np.nan)),
    }
