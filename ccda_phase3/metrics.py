from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def finite_mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    if not vals:
        return None
    return float(np.mean(vals))


def finite_std(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    if len(vals) <= 1:
        return 0.0 if vals else None
    return float(np.std(vals, ddof=1))


def bead_xy_from_state(state: np.ndarray, n_beads: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    return state[: n_beads * 2].reshape(n_beads, 2)


def chamfer_xy(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    bb = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    if aa.size == 0 or bb.size == 0:
        return float("nan")
    d = np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=-1)
    return float(np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))) / 2.0


def state_chamfer(a: np.ndarray, b: np.ndarray, n_beads: int) -> float:
    return chamfer_xy(bead_xy_from_state(a, n_beads), bead_xy_from_state(b, n_beads))


def curve_metric_from_state(state: np.ndarray, n_beads: int) -> float:
    xy = bead_xy_from_state(state, n_beads)
    if len(xy) < 3:
        return 0.0
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    chord = float(np.linalg.norm(xy[-1] - xy[0]))
    return float(np.sum(seg) - chord)


def physical_violation_length(state: np.ndarray, n_beads: int, min_len: float = 0.001, max_len: float = 0.08) -> float:
    xy = bead_xy_from_state(state, n_beads)
    if len(xy) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    low = np.maximum(0.0, min_len - seg)
    high = np.maximum(0.0, seg - max_len)
    return float(np.sum(low + high))


def branch_stats(samples_final: np.ndarray, true_final: np.ndarray, free_final: np.ndarray, pin_final: np.ndarray, true_condition: str, n_beads: int) -> Dict[str, float]:
    samples = np.asarray(samples_final, dtype=np.float32)
    d_true = np.asarray([state_chamfer(s, true_final, n_beads) for s in samples], dtype=np.float32)
    d_free = np.asarray([state_chamfer(s, free_final, n_beads) for s in samples], dtype=np.float32)
    d_pin = np.asarray([state_chamfer(s, pin_final, n_beads) for s in samples], dtype=np.float32)
    closer_pin = d_pin < d_free
    p_pin = float(np.mean(closer_pin)) if len(closer_pin) else 0.0
    p_free = 1.0 - p_pin
    correct_pin = true_condition == "hidden_pin"
    if true_condition not in ("free", "hidden_pin"):
        wrong = float("nan")
        acc = float("nan")
        majority_wrong = False
    else:
        wrong = p_free if correct_pin else p_pin
        acc = 1.0 - wrong
        majority_wrong = bool((p_pin >= 0.5) != correct_pin)
    probs = np.asarray([max(p_free, 1e-8), max(p_pin, 1e-8)], dtype=np.float32)
    entropy = float(-np.sum(probs * np.log(probs)))
    mean_final = np.mean(samples, axis=0)
    midpoint = 0.5 * (np.asarray(free_final) + np.asarray(pin_final))
    d_mean_free = state_chamfer(mean_final, free_final, n_beads)
    d_mean_pin = state_chamfer(mean_final, pin_final, n_beads)
    d_mean_mid = state_chamfer(mean_final, midpoint, n_beads)
    return {
        "future_chamfer_to_true": float(np.mean(d_true)),
        "final_chamfer_to_true": float(d_true[0]) if len(d_true) == 1 else float(np.mean(d_true)),
        "min_sample_chamfer_to_true": float(np.min(d_true)) if len(d_true) else float("nan"),
        "mean_prediction_chamfer_to_true": state_chamfer(mean_final, true_final, n_beads),
        "p_free_branch": p_free,
        "p_pin_branch": p_pin,
        "sample_wrong_branch_rate": float(wrong),
        "majority_wrong_branch": bool(majority_wrong),
        "branch_entropy": entropy,
        "branch_accuracy": float(acc),
        "d_mean_to_free": float(d_mean_free),
        "d_mean_to_pin": float(d_mean_pin),
        "d_mean_to_midpoint": float(d_mean_mid),
        "averaging_score": float(min(d_mean_free, d_mean_pin) - d_mean_mid),
        "physical_violation_length": physical_violation_length(mean_final, n_beads),
        "curve_error": abs(curve_metric_from_state(mean_final, n_beads) - curve_metric_from_state(true_final, n_beads)),
    }
