#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from typing import Any, Dict

import numpy as np

import phase3_12d_r21_paired_horizon_audit as audit


def make_record(
    *,
    pre_xy: np.ndarray,
    post_xy: np.ndarray,
    immediate_xy: np.ndarray,
    armed: bool,
    constraint_count: int,
) -> Dict[str, Any]:
    n = int(pre_xy.shape[0])
    zeros = np.zeros((n, 3), dtype=np.float64)
    goal = np.stack(
        [
            np.linspace(0.0, 0.2, n),
            np.linspace(0.0, 0.0, n),
        ],
        axis=1,
    )
    return {
        "goal_xy": goal.tolist(),
        "radius": 0.005,
        "hidden_constraint_count": int(constraint_count),
        "hidden_body_count": 0,
        "hidden_contact_applied": bool(armed),
        "meta": {
            "settle_steps_used": 600,
            "pre_arm_xy": np.asarray(pre_xy, dtype=float).tolist(),
            "immediate_post_arm_xy": np.asarray(immediate_xy, dtype=float).tolist(),
            "post_arm_xy": np.asarray(post_xy, dtype=float).tolist(),
            "pre_arm_velocity": zeros.tolist(),
            "immediate_post_arm_velocity": zeros.tolist(),
            "post_arm_velocity": zeros.tolist(),
            "arm_anchor_error": 0.0,
            "hook_error": "None",
            "breakaway_released": False,
        },
    }


def assert_close(value: float, target: float = 0.0, atol: float = 1e-12) -> None:
    if not abs(float(value) - float(target)) <= float(atol):
        raise AssertionError(f"{value} is not close to {target} within {atol}")


def main() -> None:
    base = np.asarray(
        [
            [0.00, 0.00],
            [0.05, 0.00],
            [0.10, 0.00],
            [0.15, 0.00],
        ],
        dtype=np.float64,
    )
    shared_motion = np.asarray(
        [
            [2.0e-4, 0.0],
            [1.5e-4, 0.0],
            [1.0e-4, 0.0],
            [0.5e-4, 0.0],
        ],
        dtype=np.float64,
    )

    free = make_record(
        pre_xy=base,
        immediate_xy=base,
        post_xy=base + shared_motion,
        armed=False,
        constraint_count=0,
    )
    free_rep = copy.deepcopy(free)
    unarmed = copy.deepcopy(free)
    armed_same = make_record(
        pre_xy=base,
        immediate_xy=base,
        post_xy=base + shared_motion,
        armed=True,
        constraint_count=1,
    )

    result = audit.build_horizon_comparison(
        horizon=20,
        free_a=free,
        free_b=free_rep,
        hidden_unarmed=unarmed,
        hidden_armed=armed_same,
    )

    # The historical absolute drift is non-zero, but condition-specific paired
    # and excess-motion differences must be exactly zero.
    if result["absolute_no_action_drift"]["hidden_armed"]["max_abs"] <= 1e-4:
        raise AssertionError("synthetic absolute drift did not exceed the old gate")
    assert_close(result["same_horizon"]["paired_visible_difference"]["max_abs"])
    assert_close(result["same_horizon"]["excess_motion_difference"]["max_abs"])
    assert_close(result["same_horizon"]["common_evolution_parity"]["max_abs"])
    assert_close(result["same_horizon"]["free_reproducibility"]["max_abs"])

    # Add a latent-condition-only motion. Both same-horizon and excess-motion
    # comparisons must detect it.
    extra = np.zeros_like(base)
    extra[2, 0] = 2.5e-4
    armed_leaky = make_record(
        pre_xy=base,
        immediate_xy=base,
        post_xy=base + shared_motion + extra,
        armed=True,
        constraint_count=1,
    )
    leaky = audit.build_horizon_comparison(
        horizon=20,
        free_a=free,
        free_b=free_rep,
        hidden_unarmed=unarmed,
        hidden_armed=armed_leaky,
    )
    assert_close(
        leaky["same_horizon"]["paired_visible_difference"]["max_abs"],
        2.5e-4,
    )
    assert_close(
        leaky["same_horizon"]["excess_motion_difference"]["max_abs"],
        2.5e-4,
    )

    # A free replicate discrepancy must be attributed to simulator
    # reproducibility, not to the hidden condition.
    free_bad = copy.deepcopy(free)
    bad_post = np.asarray(free_bad["meta"]["post_arm_xy"], dtype=np.float64)
    bad_post[0, 1] += 3.0e-5
    free_bad["meta"]["post_arm_xy"] = bad_post.tolist()
    reproducibility = audit.build_horizon_comparison(
        horizon=20,
        free_a=free,
        free_b=free_bad,
        hidden_unarmed=unarmed,
        hidden_armed=armed_same,
    )
    assert_close(
        reproducibility["same_horizon"]["free_reproducibility"]["max_abs"],
        3.0e-5,
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "shared_absolute_drift": result["absolute_no_action_drift"][
                    "hidden_armed"
                ]["max_abs"],
                "shared_paired_difference": result["same_horizon"][
                    "paired_visible_difference"
                ]["max_abs"],
                "leak_detected": leaky["same_horizon"][
                    "paired_visible_difference"
                ]["max_abs"],
                "reproducibility_detected": reproducibility["same_horizon"][
                    "free_reproducibility"
                ]["max_abs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
