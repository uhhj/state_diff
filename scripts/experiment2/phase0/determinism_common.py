"""Pure comparison helpers for fixed-step deterministic replay."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def compare_traces(
    reference: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    numeric_atol: float,
) -> Dict[str, Any]:
    numeric_atol = float(numeric_atol)
    if numeric_atol < 0:
        raise ValueError("numeric_atol must be non-negative")

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    result: Dict[str, Any] = {
        "keys_match": reference_keys == candidate_keys,
        "fields": {},
    }
    for key in sorted(reference_keys | candidate_keys):
        if key not in reference or key not in candidate:
            result["fields"][key] = {
                "present_in_both": False,
                "shape_match": False,
                "exact": False,
                "max_abs": None,
                "within_atol": False,
            }
            continue

        first = np.asarray(reference[key])
        second = np.asarray(candidate[key])
        shape_match = first.shape == second.shape
        field: Dict[str, Any] = {
            "present_in_both": True,
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
            "shape_match": shape_match,
        }
        if not shape_match:
            field.update(
                {
                    "exact": False,
                    "max_abs": None,
                    "within_atol": False,
                }
            )
        elif first.dtype.kind in {"U", "S", "O"}:
            exact = bool(np.array_equal(first, second))
            field.update(
                {
                    "exact": exact,
                    "max_abs": None,
                    "within_atol": exact,
                }
            )
        else:
            exact = bool(np.array_equal(first, second))
            max_abs = (
                float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
                if first.size
                else 0.0
            )
            field.update(
                {
                    "exact": exact,
                    "max_abs": max_abs,
                    "within_atol": max_abs <= numeric_atol,
                }
            )
        result["fields"][key] = field

    result["shape_match"] = bool(
        result["keys_match"]
        and all(field["shape_match"] for field in result["fields"].values())
    )
    result["phase_exact"] = bool(
        result["fields"].get("phase", {}).get("exact", False)
    )
    result["numeric_within_atol"] = bool(
        all(field["within_atol"] for field in result["fields"].values())
    )
    result["byte_exact"] = bool(
        result["keys_match"]
        and all(field["exact"] for field in result["fields"].values())
    )
    result["passed"] = bool(
        result["shape_match"]
        and result["phase_exact"]
        and result["numeric_within_atol"]
    )
    return result


def compare_repeat_metadata(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("repeat metadata is empty")

    def all_equal(key: str) -> bool:
        return len({str(row[key]) for row in rows}) == 1

    free_events = [row["free_events"] for row in rows]
    hidden_events = [row["hidden_events"] for row in rows]
    checks = {
        "base_state_hash": all_equal("base_state_hash"),
        "action_hash": all_equal("action_hash"),
        "free_trace_length": all_equal("free_trace_length"),
        "hidden_trace_length": all_equal("hidden_trace_length"),
        "free_events": all(value == free_events[0] for value in free_events),
        "hidden_events": all(value == hidden_events[0] for value in hidden_events),
        "pair_action_match": all(bool(row["action_hash_match"]) for row in rows),
        "pair_initial_match": all(
            bool(row["free_base_state_hash_match"])
            and bool(row["hidden_base_state_hash_match"])
            and bool(row["free_hidden_initial_hash_match"])
            and float(row["max_initial_state_difference"]) == 0.0
            for row in rows
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def max_numeric_error(comparisons: Iterable[Dict[str, Any]]) -> float:
    maximum = 0.0
    for comparison in comparisons:
        for field in comparison["fields"].values():
            value = field.get("max_abs")
            if value is not None:
                maximum = max(maximum, float(value))
    return maximum
