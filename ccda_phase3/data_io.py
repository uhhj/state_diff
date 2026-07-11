import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .action_codec import ActionCodec, ExecutableActionCodec

DEFAULT_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
CONDITIONS = list(DEFAULT_CONDITIONS)
FORBIDDEN_METADATA_NOT_IN_X = [
    "hidden_condition",
    "hidden_contact_meta",
    "recoverability_params",
    "recoverability_class_candidate",
    "breakaway_released",
    "breakaway_release_step",
    "breakaway_step",
    "breakaway_threshold",
    "breakaway_max_disp_seen",
    "condition_id",
    "condition_name",
    "condition_label",
    "condition_id_onehot",
    "success",
    "final_fraction",
    "ccda_pair_group",
    "pair_group",
    "source_file",
    "visible_seed",
    "ccda_visible_seed",
]
ROBOT_PROXY_MAX_JOINTS = 16
ROBOT_PROXY_DIM = ROBOT_PROXY_MAX_JOINTS * 2 + 3 + 4
_VISIBLE_SEED_PATTERN = re.compile(r"(?:^|_)seed_(?P<seed>[0-9]+)$")


def normalize_conditions(conditions: Optional[Iterable[str]] = None) -> List[str]:
    if conditions is None:
        return list(DEFAULT_CONDITIONS)
    out: List[str] = []
    for item in conditions:
        for part in str(item).replace(",", " ").split():
            part = part.strip()
            if part and part not in out:
                out.append(part)
    if "free" not in out:
        raise ValueError("Phase3 condition list must include free")
    return out


def condition_to_id_map(conditions: Optional[Iterable[str]] = None) -> Dict[str, int]:
    return {name: i for i, name in enumerate(normalize_conditions(conditions))}


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


def finite_velocity_or_difference(
    xy: np.ndarray,
    velocity: np.ndarray,
    *,
    prev_xy: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Return finite XY velocity with identical offline/live semantics."""
    xy_array = np.asarray(xy, dtype=np.float32)
    if (
        xy_array.ndim != 2
        or xy_array.shape[1] != 2
        or not np.all(np.isfinite(xy_array))
    ):
        raise ValueError(f"xy must be finite [N,2], got shape={xy_array.shape}")

    velocity_array = np.asarray(velocity, dtype=np.float32)
    if (
        velocity_array.shape == xy_array.shape
        and np.all(np.isfinite(velocity_array))
    ):
        return velocity_array.astype(np.float32, copy=True)

    if prev_xy is not None:
        previous = np.asarray(prev_xy, dtype=np.float32)
        if previous.shape == xy_array.shape and np.all(np.isfinite(previous)):
            result = (xy_array - previous) / max(float(dt), 1e-6)
            if np.all(np.isfinite(result)):
                return result.astype(np.float32, copy=False)

    return np.zeros_like(xy_array, dtype=np.float32)


def state_from_info(
    info: Any,
    prev_xy: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Tuple[np.ndarray, str, int]:
    xy = extract_bead_xy(info)
    if xy.size == 0:
        raise ValueError("missing bead_positions")

    vel = finite_velocity_or_difference(
        xy,
        extract_bead_vel_xy(info),
        prev_xy=prev_xy,
        dt=dt,
    )
    robot, source = extract_robot_pose_proxy(info)
    if not np.all(np.isfinite(robot)):
        raise ValueError("non-finite robot_pose_proxy")

    state = np.concatenate(
        [xy.reshape(-1), vel.reshape(-1), robot], axis=0
    ).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError("state_from_info produced non-finite state")
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
    explicit = ex.get("ccda_visible_seed", None)
    if explicit not in (None, ""):
        try:
            seed = int(explicit)
        except (TypeError, ValueError, OverflowError):
            seed = -1
        if seed >= 0:
            return seed

    group = str(ex.get("ccda_pair_group", "")).strip()
    match = _VISIBLE_SEED_PATTERN.search(group)
    if match is None:
        return -1
    try:
        return int(match.group("seed"))
    except (TypeError, ValueError, OverflowError):
        return -1


def load_episode(condition: str, condition_dir: Path, file_name: str, codec: Optional[ActionCodec] = None, condition_to_id: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
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
        "condition_id": (condition_to_id or condition_to_id_map())[condition],
        "file_name": file_name,
        "episode_len": parse_ep_len(file_name),
        "visible_seed": visible_seed_from_extras(first_ex or last_ex),
        "pair_group": str(first_ex.get("ccda_pair_group", last_ex.get("ccda_pair_group", ""))),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(action_vecs, dtype=np.float32),
        "raw_actions": actions,
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


def build_action_history(actions: List[Any], t: int, th: int, codec: ExecutableActionCodec) -> np.ndarray:
    """Past-action history for the state_action baseline.

    Returns actions from [t-th, ..., t-1], left-padded with zeros. The current
    target action actions[t] is intentionally excluded.
    """
    action_dim = codec.dim()
    hist = []
    for k in range(t - th, t):
        if k < 0:
            hist.append(np.zeros(action_dim, dtype=np.float32))
        else:
            hist.append(codec.encode(actions[k]).astype(np.float32))
    if not hist:
        hist = [np.zeros(action_dim, dtype=np.float32) for _ in range(th)]
    return np.stack(hist, axis=0).reshape(-1).astype(np.float32)


def condition_files(condition_dir: Path) -> List[str]:
    color_dir = Path(condition_dir) / "color"
    if not color_dir.exists():
        return []
    return [p.name for p in sorted(color_dir.glob("*.pkl"))]


def build_windows_from_dataset(split_name: str, data_root: Path, th: int, tf: int, codec: Optional[ActionCodec] = None, max_windows_per_episode: int = 0, conditions: Optional[Iterable[str]] = None) -> Tuple[List[Dict[str, Any]], Optional[ActionCodec], Counter]:
    data_root = Path(data_root)
    windows: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    active_conditions = normalize_conditions(conditions)
    condition_to_id = condition_to_id_map(active_conditions)
    for condition in active_conditions:
        cond_dir = data_root / condition
        for file_name in condition_files(cond_dir):
            ep = load_episode(condition, cond_dir, file_name, codec=codec, condition_to_id=condition_to_id)
            codec = ep["codec"]
            states = ep["states"]
            raw_actions = ep.get("raw_actions", [])
            if len(raw_actions) == 0 or len(states) < 2:
                continue
            max_t = min(len(raw_actions), len(states) - 1)
            if max_windows_per_episode > 0:
                max_t = min(max_t, max_windows_per_episode)
            for t in range(max_t):
                if t >= len(raw_actions) or t + 1 >= len(states):
                    continue
                hist_states = resample_or_pad_history(states[: t + 1], th)
                future = resample_or_pad_future(states[t + 1 : t + 1 + tf], tf)
                hist_actions = build_action_history(raw_actions, t, th, codec)
                y_action_t = codec.encode(raw_actions[t]).astype(np.float32)
                source_counts.update(ep["robot_pose_proxy_sources"][: t + 1])
                windows.append(
                    {
                        "paper_x": hist_states.reshape(-1),
                        "state_action_x": np.concatenate([hist_states.reshape(-1), hist_actions.reshape(-1)]).astype(np.float32),
                        "y_state": future.astype(np.float32),
                        "y_final_state": future[-1].astype(np.float32),
                        "y_action": y_action_t,
                        "condition_id": ep["condition_id"],
                        "condition_name": condition,
                        "visible_seed": ep["visible_seed"],
                        "split_name": split_name,
                        "source_file": file_name,
                        "window_t": int(t),
                        "episode_action_len": int(len(raw_actions)),
                        "success": ep["success"],
                        "final_fraction": ep["final_fraction"],
                        "n_beads": ep["n_beads"],
                        "robot_pose_proxy_source": ep["robot_pose_proxy_sources"][min(t, len(ep["robot_pose_proxy_sources"]) - 1)],
                    }
                )
    return windows, codec, source_counts


def save_action_template(path: Path, codec: ActionCodec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(codec, ExecutableActionCodec):
        codec = ExecutableActionCodec(codec.template)
    summary = codec.summary()
    if codec.dim() != 14 or summary.get("num_camera_config_paths", 0) != 0:
        raise ValueError(f"Invalid executable action codec: {summary}")
    codec.save(path)


def load_action_codec_from_template(path: Path) -> ActionCodec:
    return ExecutableActionCodec.load(path)
