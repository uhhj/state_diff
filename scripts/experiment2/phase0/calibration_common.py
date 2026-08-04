"""Pure helpers for Hidden-Friction Cable parameter calibration."""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List

import numpy as np


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_candidate_config(
    base_config: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    override = {
        key: value
        for key, value in candidate.items()
        if key in {"action", "friction"}
    }
    return deep_update(base_config, override)


def _median(rows: List[Dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def _maximum(rows: List[Dict[str, Any]], key: str) -> float:
    return float(np.max([float(row[key]) for row in rows]))


def summarize_candidate(
    candidate_id: str,
    candidate: Dict[str, Any],
    pair_rows: List[Dict[str, Any]],
    targets: Dict[str, float],
) -> Dict[str, Any]:
    if not pair_rows:
        raise ValueError("candidate has no valid pair rows")

    median_preload = _median(pair_rows, "preload_end_max_abs_xy")
    median_contact = _median(pair_rows, "contact_impulse_gap")
    median_ade = _median(pair_rows, "main_branch_ade")
    median_fde = _median(pair_rows, "main_branch_fde")
    median_drift = max(
        _median(pair_rows, "free_no_action_drift"),
        _median(pair_rows, "hidden_no_action_drift"),
    )
    amplification_values = [
        float(row["main_branch_fde"])
        / max(float(row["preload_end_max_abs_xy"]), 1e-9)
        for row in pair_rows
    ]
    median_amplification = float(np.median(amplification_values))

    exact_pairing = all(
        bool(row["action_hash_match"])
        and bool(row["free_base_state_hash_match"])
        and bool(row["hidden_base_state_hash_match"])
        and bool(row["free_hidden_initial_hash_match"])
        for row in pair_rows
    )

    checks = {
        "exact_pairing": exact_pairing,
        "initial_match": (
            _maximum(pair_rows, "max_initial_state_difference")
            <= float(targets["max_initial_abs_xy"])
        ),
        "arm_jump": max(
            _maximum(pair_rows, "free_arm_max_abs_jump"),
            _maximum(pair_rows, "hidden_arm_max_abs_jump"),
        )
        <= float(targets["max_arm_jump"]),
        "no_action_drift": median_drift
        <= float(targets["max_median_no_action_drift"]),
        "preload_visibility": median_preload
        <= float(targets["max_median_preload_visible_difference"]),
        "contact_informativeness": median_contact
        >= float(targets["min_median_contact_impulse_gap"]),
        "main_ade": median_ade
        >= float(targets["min_median_main_branch_ade"]),
        "main_fde": median_fde
        >= float(targets["min_median_main_branch_fde"]),
        "branch_amplification": median_amplification
        >= float(targets["min_median_branch_amplification"]),
    }
    eligible = all(checks.values())

    def ratio(value: float, target: float) -> float:
        return min(max(value / max(target, 1e-12), 0.0), 2.0)

    preload_limit = float(
        targets["max_median_preload_visible_difference"]
    )
    drift_limit = float(targets["max_median_no_action_drift"])
    score = (
        2.0 * ratio(
            median_contact,
            float(targets["min_median_contact_impulse_gap"]),
        )
        + ratio(median_ade, float(targets["min_median_main_branch_ade"]))
        + 2.0
        * ratio(median_fde, float(targets["min_median_main_branch_fde"]))
        + ratio(
            median_amplification,
            float(targets["min_median_branch_amplification"]),
        )
        + max(0.0, 1.0 - median_preload / preload_limit)
        + max(0.0, 1.0 - median_drift / drift_limit)
    )
    if not exact_pairing:
        score = -1.0e30

    return {
        "candidate_id": candidate_id,
        "candidate": candidate,
        "valid_pairs": len(pair_rows),
        "eligible": bool(eligible),
        "checks": checks,
        "score": float(score),
        "median": {
            "preload_end_max_abs_xy": median_preload,
            "contact_impulse_gap": median_contact,
            "main_branch_ade": median_ade,
            "main_branch_fde": median_fde,
            "max_no_action_drift": median_drift,
            "branch_amplification": median_amplification,
        },
        "maximum": {
            "initial_state_difference": _maximum(
                pair_rows, "max_initial_state_difference"
            ),
            "free_arm_max_abs_jump": _maximum(
                pair_rows, "free_arm_max_abs_jump"
            ),
            "hidden_arm_max_abs_jump": _maximum(
                pair_rows, "hidden_arm_max_abs_jump"
            ),
        },
        "pairs": pair_rows,
    }


def rank_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(rows),
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["score"]),
            float(row["median"]["preload_end_max_abs_xy"]),
            -float(row["median"]["main_branch_fde"]),
            str(row["candidate_id"]),
        ),
    )
