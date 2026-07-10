#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SELECTORS = [
    "ddpm_mean",
    "condition_nearest_upper",
    "proxy_state_motion_nn",
    "proxy_combined_topk_action_geom",
]

CONDITIONS = [
    "free",
    "hidden_breakaway_pin",
    "hidden_high_friction",
]

VISIBLE_SEEDS = [
    312000,
    312001,
    312002,
    312003,
    312500,
    312501,
    312502,
    312503,
]

FORBIDDEN_PREFIXES = [
    "tensorflow",
    "ravens.agents",
    "ravens.models",
    "ravens.datasets",
]


def add_repo_paths(root: Path) -> None:
    paths = [
        root,
        root / "scripts",
        root / "external/deformable-ravens",
    ]
    for path in paths:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def assert_no_forbidden_modules(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(
            f"[Phase3.12c][FAIL] forbidden modules loaded at {stage}: "
            f"{sorted(set(bad))[:40]}"
        )


def seed_cohort(seed: int) -> str:
    seed = int(seed)
    if 312000 <= seed <= 312003:
        return "phase312_seed_block"
    if 312500 <= seed <= 312503:
        return "phase312b_seed_block"
    return "other"


def seed_everything(seed: int) -> None:
    """Seed all RNGs used by the current runtime.

    PYTHONHASHSEED must also be set in the parent subprocess environment,
    because changing it after interpreter startup is not sufficient.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        # Torch import is separately checked by preflight.
        pass


def set_ccda_environment(condition: str, visible_seed: int) -> str:
    """Set selector-independent reset metadata.

    Pair group intentionally excludes selector and condition. The hidden
    condition remains controlled by CCDA_HIDDEN_CONDITION.
    """
    visible_seed = int(visible_seed)
    pair_group = f"phase3_12c_seed_{visible_seed}"

    os.environ["CCDA_HIDDEN_CONDITION"] = str(condition)
    os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
    os.environ["CCDA_PAIR_GROUP"] = pair_group

    return pair_group


def scalar_from_npz(data: Any, key: str) -> Any:
    raw = data[key]
    if getattr(raw, "shape", ()) == ():
        return raw.item()
    flat = raw.reshape(-1)
    if len(flat) != 1:
        raise ValueError(
            f"Expected scalar-like NPZ entry for {key}, got shape={raw.shape}"
        )
    value = flat[0]
    return value.item() if hasattr(value, "item") else value


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True)
    )
    tmp.replace(path)


def write_csv_row(
    path: Path,
    row: Dict[str, Any],
    fields: Sequence[str],
    write_header: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()

        clean: Dict[str, Any] = {}
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, str):
                value = value.replace("\r", "\\r").replace("\n", "\\n")
            clean[key] = value

        writer.writerow(clean)
        file.flush()
        os.fsync(file.fileno())


def deterministic_reset(
    env: Any,
    task: Any,
    *,
    min_settle_steps: int,
    max_settle_steps: int,
    static_checks_required: int,
    static_check_interval: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reset without wall-clock-driven physics stepping.

    DeformableRavens normally:
    - starts a background simulation thread,
    - sleeps for a wall-clock duration inside cable reset,
    - starts the background thread again at the end of Environment.reset,
    - calls Environment.step(None).

    For reset integrity we replace those operations during reset, then
    perform a deterministic number of direct PyBullet steps.

    The original Environment methods are restored before rollout actions.
    """
    import pybullet as p

    min_settle_steps = int(min_settle_steps)
    max_settle_steps = int(max_settle_steps)
    static_checks_required = int(static_checks_required)
    static_check_interval = int(static_check_interval)

    if min_settle_steps < 0:
        raise ValueError("min_settle_steps must be >= 0")
    if max_settle_steps < min_settle_steps:
        raise ValueError(
            "max_settle_steps must be >= min_settle_steps"
        )
    if static_checks_required < 1:
        raise ValueError("static_checks_required must be >= 1")
    if static_check_interval < 1:
        raise ValueError("static_check_interval must be >= 1")

    # Remove wall-clock sleeps from the two cable reset layers.
    if hasattr(task, "_settle_secs"):
        task._settle_secs = 0.0
    os.environ["CCDA_SETTLE_SECONDS"] = "0"

    original_start = env.start
    original_step = env.step

    def no_background_start() -> None:
        env.running = False

    def no_action_step(act=None):
        # Environment.reset only consumes the first tuple element.
        return {}, 0.0, False, {}

    env.pause()
    env.start = no_background_start
    env.step = no_action_step

    try:
        env.reset(task)
    finally:
        env.start = original_start
        env.step = original_step
        env.pause()

    try:
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1
        )
    except Exception:
        pass

    p.setTimeStep(1.0 / float(env.hz))

    steps_used = 0
    stable_count = 0
    settled_static = False

    while steps_used < max_settle_steps:
        p.stepSimulation()

        end_effector = getattr(env, "ee", None)
        if end_effector is not None:
            try:
                end_effector.step()
            except Exception:
                pass

        steps_used += 1

        if steps_used < min_settle_steps:
            continue

        if steps_used % static_check_interval != 0:
            continue

        try:
            currently_static = bool(env.is_static())
        except Exception:
            currently_static = False

        if currently_static:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= static_checks_required:
            settled_static = True
            break

    env.pause()

    # Environment.reset normally performs one reward/info query through
    # Environment.step(None). Recreate that query after deterministic settle.
    _, reward_extras = task.reward()
    reward_extras["task.done"] = bool(task.done())

    info = env.info
    info["extras"] = reward_extras

    metadata = {
        "settle_steps_used": int(steps_used),
        "settle_static": bool(settled_static),
        "static_checks_observed": int(stable_count),
        "min_settle_steps": int(min_settle_steps),
        "max_settle_steps": int(max_settle_steps),
        "static_checks_required": int(static_checks_required),
        "static_check_interval": int(static_check_interval),
    }
    return info, metadata


def initial_snapshot(
    info: Dict[str, Any],
    *,
    n_beads: int,
    round_decimals: int = 7,
) -> Dict[str, Any]:
    from ccda_phase3.metrics import curve_metric_from_state
    from ccda_phase3.rollout import (
        final_fraction_from_info,
        state_from_live_info,
    )

    state = state_from_live_info(info, prev_xy=None).astype(np.float32)
    rounded = np.round(
        state.astype(np.float64),
        decimals=int(round_decimals),
    ).astype(np.float32)

    xy_dim = int(n_beads) * 2
    bead_xy = state[:xy_dim]
    bead_vel = state[xy_dim:xy_dim * 2]
    robot_proxy = state[xy_dim * 2:]

    extras = info.get("extras", {})
    if not isinstance(extras, dict):
        extras = {}

    hidden_meta = extras.get("hidden_contact_meta", {})
    if not isinstance(hidden_meta, dict):
        hidden_meta = {}

    breakaway_released = hidden_meta.get(
        "breakaway_released",
        extras.get("breakaway_released", False),
    )

    return {
        "initial_fraction": float(final_fraction_from_info(info)),
        "initial_curve": float(
            curve_metric_from_state(state, int(n_beads))
        ),
        "initial_state_dim": int(state.size),
        "initial_state_sha256": hashlib.sha256(
            state.tobytes()
        ).hexdigest(),
        "initial_state_round_sha256": hashlib.sha256(
            rounded.tobytes()
        ).hexdigest(),
        "initial_state_json": json.dumps(
            [float(x) for x in state],
            separators=(",", ":"),
        ),
        "initial_bead_xy_json": json.dumps(
            [float(x) for x in bead_xy],
            separators=(",", ":"),
        ),
        "initial_bead_velocity_json": json.dumps(
            [float(x) for x in bead_vel],
            separators=(",", ":"),
        ),
        "initial_robot_proxy_json": json.dumps(
            [float(x) for x in robot_proxy],
            separators=(",", ":"),
        ),
        "initial_condition_logged": str(
            extras.get("hidden_condition", "")
        ),
        "initial_pair_group_logged": str(
            extras.get("ccda_pair_group", "")
        ),
        "initial_visible_seed_logged": str(
            extras.get("ccda_visible_seed", "")
        ),
        "initial_breakaway_released": int(
            bool(breakaway_released)
        ),
    }


def parse_state_json(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Missing initial_state_json")
    text = str(value)
    parsed = json.loads(text)
    state = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if state.size == 0:
        raise ValueError("Empty initial_state_json")
    if not np.all(np.isfinite(state)):
        raise ValueError("Non-finite value in initial_state_json")
    return state


def finite_mean(values: Iterable[float]) -> float:
    items = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return float(np.mean(items)) if items else float("nan")


def finite_median(values: Iterable[float]) -> float:
    items = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return float(np.median(items)) if items else float("nan")
