from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .data_io import ROBOT_PROXY_DIM, extract_bead_xy, extract_bead_vel_xy, extract_robot_pose_proxy, get_extras


def state_from_live_info(info: Dict[str, Any], prev_xy=None) -> np.ndarray:
    xy = extract_bead_xy(info)
    vel = extract_bead_vel_xy(info)
    if vel.shape != xy.shape:
        vel = np.zeros_like(xy) if prev_xy is None else xy - prev_xy
    robot, _ = extract_robot_pose_proxy(info)
    return np.concatenate([xy.reshape(-1), vel.reshape(-1), robot], axis=0).astype(np.float32)


def pad_history(items: List[np.ndarray], length: int) -> np.ndarray:
    if not items:
        raise ValueError("empty history")
    arr = np.asarray(items, dtype=np.float32)
    if len(arr) >= length:
        return arr[-length:]
    pad = np.repeat(arr[:1], length - len(arr), axis=0)
    return np.concatenate([pad, arr], axis=0)


def final_fraction_from_info(info: Dict[str, Any]) -> float:
    ex = get_extras(info)
    nb = ex.get("nb_beads", None)
    nz = ex.get("nb_zone", None)
    try:
        return float(nz) / float(nb)
    except Exception:
        try:
            return float(ex.get("total_rewards", np.nan))
        except Exception:
            return float("nan")
