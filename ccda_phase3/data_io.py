import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .action_codec import ActionCodec

CONDITIONS = ["free", "hidden_pin", "hidden_high_friction"]
CONDITION_TO_ID = {name: i for i, name in enumerate(CONDITIONS)}
ROBOT_PROXY_MAX_JOINTS = 16
ROBOT_PROXY_DIM = ROBOT_PROXY_MAX_JOINTS * 2 + 3 + 4


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def get_extras(info: Any) -> Dict[str, Any]:
    if isinstance(info, dict):
        ex = info.get("extras", {})
        return ex if isinstance(ex, dict) else {}
    return {}


def extract_bead_xy(info: Any) -> np.ndarray:
    ex = get_extras(info)
    pos = ex.get("bead_positions", [])
    arr = np.asarray(pos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32, copy=True)


def extract_bead_vel_xy(info: Any) -> np.ndarray:
    ex = get_extras(info)
    vel = ex.get("bead_velocities", [])
    arr = np.asarray(vel, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32, copy=True)


def extract_robot_pose_proxy(info: Any) -> Tuple[np.ndarray, str]:
    proxy = get_extras(info).get("robot_pose_proxy", {})
    source = "missing_zero_proxy"
    out = np.zeros((ROBOT_PROXY_DIM,), dtype=np.float32)
    if isinstance(proxy, dict):
        source = str(proxy.get("source", source))
        jp = np.asarray(proxy.get("joint_positions", []), dtype=np.float32).reshape(-1)[:ROBOT_PROXY_MAX_JOINTS]
        jv = np.asarray(proxy.get("joint_velocities", []), dtype=np.float32).reshape(-1)[:ROBOT_PROXY_MAX_JOINTS]
        ee = np.asarray(proxy.get("ee_position", [0, 0, 0]), dtype=np.float32).reshape(-1)[:3]
        orn = np.asarray(proxy.get("ee_orientation", [0, 0, 0, 1]), dtype=np.float32).reshape(-1)[:4]
        out[: jp.size] = jp
        out[ROBOT_PROXY_MAX_JOINTS : ROBOT_PROXY_MAX_JOINTS + jv.size] = jv
        out[ROBOT_PROXY_MAX_JOINTS * 2 : ROBOT_PROXY_MAX_JOINTS * 2 + ee.size] = ee
        out[ROBOT_PROXY_MAX_JOINTS * 2 + 3 : ROBOT_PROXY_MAX_JOINTS * 2 + 3 + orn.size] = orn
    return out, source


def state_from_info(info: Any, prev_xy: Optional[np.ndarray] = None, dt: float = 1.0) -> Tuple[np.ndarray, str, int]:
    xy = extract_bead_xy(info)
    vel = extract_bead_vel_xy(info)
    if xy.size == 0:
        raise ValueError("missing bead_positions")
    if vel.shape != xy.shape or not np.any(np.isfinite(vel)):
        if prev_xy is not None and prev_xy.shape == xy.shape:
            vel = (xy - prev_xy) / max(float(dt), 1e-6)
        else:
            vel = np.zeros_like(xy)
    robot, source = extract_robot_pose_proxy(info)
    state = np.concatenate([xy.reshape(-1), vel.reshape(-1), robot], axis=0).astype(np.float32)
    return state, source, int(xy.shape[0])


def extract_action_template_and_vector(action: Any, codec: Optional[ActionCodec] = None) -> Tuple[Any, np.ndarray, ActionCodec]:
    if codec is None:
        codec = ActionCodec(action)
    vec = codec.encode(action)
    return codec.template, vec, codec


def parse_ep_len(file_name: str) -> int:
    try:
        return int(Path(file_name).stem.split("-")[-1])
    except Exception:
        return -1


def visible_seed_from_extras(ex: Dict[str, Any]) -> int:
    value = ex.get("ccda_visible_seed", -1)
    try:
        return int(value)
    except Exception:
        group = str(ex.get("ccda_pair_group", ""))
        digits = "".join(ch for ch in group if ch.isdigit())
        return int(digits) if digits else -1


def load_episode(condition: str, condition_dir: Path, file_name: str, codec: Optional[ActionCodec] = None) -> Dict[str, Any]:
    condition_dir = Path(condition_dir)
    info_list = load_pickle(condition_dir / "info" / file_name)
    last_info = load_pickle(condition_dir / "last_info" / file_name)
    actions = load_pickle(condition_dir / "action" / file_name)
    if not isinstance(info_list, list):
        info_list = []
    state_infos = list(info_list) + [last_info]
    states = []
    sources = []
    prev_xy = None
    n_beads = 0
    for info in state_infos:
        st, source, nb = state_from_info(info, prev_xy=prev_xy)
        states.append(st)
        sources.append(source)
        prev_xy = extract_bead_xy(info)
        n_beads = max(n_beads, nb)
    action_vecs = []
    template = None
    for action in actions:
        template, vec, codec = extract_action_template_and_vector(action, codec)
        action_vecs.append(vec)
    first_ex = get_extras(info_list[0] if info_list else last_info)
    last_ex = get_extras(last_info)
    nb_beads = last_ex.get("nb_beads", n_beads)
    nb_zone = last_ex.get("nb_zone", None)
    final_fraction = None
    if nb_beads:
        try:
            final_fraction = float(nb_zone) / float(nb_beads)
        except Exception:
            final_fraction = None
    if final_fraction is None:
        final_fraction = last_ex.get("total_rewards", None)
    return {
        "condition": condition,
        "condition_id": CONDITION_TO_ID[condition],
        "file_name": file_name,
        "episode_len": parse_ep_len(file_name),
        "visible_seed": visible_seed_from_extras(first_ex or last_ex),
        "pair_group": str(first_ex.get("ccda_pair_group", last_ex.get("ccda_pair_group", ""))),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(action_vecs, dtype=np.float32),
        "action_template": template,
        "codec": codec,
        "success": bool(last_ex.get("task.done", False)),
        "final_fraction": float(final_fraction) if final_fraction is not None else np.nan,
        "robot_pose_proxy_sources": sources,
        "n_beads": n_beads,
    }


def resample_or_pad_history(history: np.ndarray, th: int) -> np.ndarray:
    arr = np.asarray(history, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        raise ValueError("empty history")
    if len(arr) >= th:
        return arr[-th:]
    pad = np.repeat(arr[:1], th - len(arr), axis=0)
    return np.concatenate([pad, arr], axis=0)


def resample_or_pad_future(future: np.ndarray, tf: int) -> np.ndarray:
    arr = np.asarray(future, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if len(arr) == 0:
        raise ValueError("empty future")
    if len(arr) >= tf:
        return arr[:tf]
    pad = np.repeat(arr[-1:], tf - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def condition_files(condition_dir: Path) -> List[str]:
    color_dir = Path(condition_dir) / "color"
    if not color_dir.exists():
        return []
    return [p.name for p in sorted(color_dir.glob("*.pkl"))]


def build_windows_from_dataset(split_name: str, data_root: Path, th: int, tf: int, codec: Optional[ActionCodec] = None, max_windows_per_episode: int = 1) -> Tuple[List[Dict[str, Any]], Optional[ActionCodec], Counter]:
    data_root = Path(data_root)
    windows: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    for condition in CONDITIONS:
        cond_dir = data_root / condition
        for file_name in condition_files(cond_dir):
            ep = load_episode(condition, cond_dir, file_name, codec=codec)
            codec = ep["codec"]
            states = ep["states"]
            actions = ep["actions"]
            if len(actions) == 0 or len(states) < 2:
                continue
            max_t = len(actions)
            if max_windows_per_episode > 0:
                max_t = min(max_t, max_windows_per_episode)
            for t in range(max_t):
                hist_states = resample_or_pad_history(states[: t + 1], th)
                future = resample_or_pad_future(states[t + 1 :], tf)
                past_actions = actions[max(0, t - th) : t]
                if len(past_actions) == 0:
                    past_actions = np.zeros((1, actions.shape[1]), dtype=np.float32)
                hist_actions = resample_or_pad_history(past_actions, th)
                source_counts.update(ep["robot_pose_proxy_sources"][: t + 1])
                windows.append(
                    {
                        "paper_x": hist_states.reshape(-1),
                        "state_action_x": np.concatenate([hist_states.reshape(-1), hist_actions.reshape(-1)]).astype(np.float32),
                        "y_state": future.astype(np.float32),
                        "y_final_state": future[-1].astype(np.float32),
                        "y_action": actions[t].astype(np.float32),
                        "condition_id": ep["condition_id"],
                        "condition_name": condition,
                        "visible_seed": ep["visible_seed"],
                        "split_name": split_name,
                        "source_file": file_name,
                        "window_t": int(t),
                        "success": ep["success"],
                        "final_fraction": ep["final_fraction"],
                        "n_beads": ep["n_beads"],
                        "robot_pose_proxy_source": ep["robot_pose_proxy_sources"][min(t, len(ep["robot_pose_proxy_sources"]) - 1)],
                    }
                )
    return windows, codec, source_counts


def save_action_template(path: Path, codec: ActionCodec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"template": codec.template, "paths": codec.paths}, f)


def load_action_codec_from_template(path: Path) -> ActionCodec:
    obj = load_pickle(path)
    codec = ActionCodec(obj["template"])
    return codec
