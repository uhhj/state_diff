"""Formal state-v2 raw episode and window I/O."""
from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .action_codec import ActionCodec, ExecutableActionCodec

ROBOT_PROXY_MAX_JOINTS = 16
ROBOT_PROXY_DIM = ROBOT_PROXY_MAX_JOINTS * 2 + 3 + 4
_VISIBLE_SEED_PATTERN = re.compile(r"(?:^|_)seed_(?P<seed>[0-9]+)$")


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def get_extras(info: Any) -> Dict[str, Any]:
    if isinstance(info, dict):
        extras = info.get("extras", {})
        return extras if isinstance(extras, dict) else {}
    return {}


def extract_bead_xy(info: Any) -> np.ndarray:
    positions = get_extras(info).get("bead_positions", [])
    array = np.asarray(positions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    result = array[:, :2].astype(np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("bead positions contain NaN or Inf")
    return result


def extract_robot_pose_proxy(info: Any) -> Tuple[np.ndarray, str]:
    proxy = get_extras(info).get("robot_pose_proxy", {})
    source = "missing_zero_proxy"
    output = np.zeros(ROBOT_PROXY_DIM, dtype=np.float32)
    if isinstance(proxy, dict):
        source = str(proxy.get("source", source))
        positions = np.asarray(proxy.get("joint_positions", []), dtype=np.float32).reshape(-1)[:ROBOT_PROXY_MAX_JOINTS]
        velocities = np.asarray(proxy.get("joint_velocities", []), dtype=np.float32).reshape(-1)[:ROBOT_PROXY_MAX_JOINTS]
        ee_position = np.asarray(proxy.get("ee_position", [0, 0, 0]), dtype=np.float32).reshape(-1)[:3]
        ee_orientation = np.asarray(proxy.get("ee_orientation", [0, 0, 0, 1]), dtype=np.float32).reshape(-1)[:4]
        output[: positions.size] = positions
        output[ROBOT_PROXY_MAX_JOINTS : ROBOT_PROXY_MAX_JOINTS + velocities.size] = velocities
        output[ROBOT_PROXY_MAX_JOINTS * 2 : ROBOT_PROXY_MAX_JOINTS * 2 + ee_position.size] = ee_position
        start = ROBOT_PROXY_MAX_JOINTS * 2 + 3
        output[start : start + ee_orientation.size] = ee_orientation
    if not np.all(np.isfinite(output)):
        raise ValueError("robot proxy contains NaN or Inf")
    return output, source


from .schema_v2 import (  # noqa: E402
    FORMAL_CONDITIONS,
    ROBOT_PROXY_DIM as SCHEMA_ROBOT_PROXY_DIM,
    STATE_DIM,
    build_window,
    state_v2_from_info,
)

if ROBOT_PROXY_DIM != SCHEMA_ROBOT_PROXY_DIM:
    raise RuntimeError("robot proxy dimension disagrees with formal schema")

DEFAULT_CONDITIONS = list(FORMAL_CONDITIONS)
CONDITIONS = list(FORMAL_CONDITIONS)


def normalize_conditions(conditions: Optional[Iterable[str]] = None) -> List[str]:
    values = list(FORMAL_CONDITIONS) if conditions is None else [part for item in conditions for part in str(item).replace(",", " ").split()]
    output: List[str] = []
    for value in values:
        if value not in FORMAL_CONDITIONS:
            raise ValueError(f"unsupported formal condition: {value}")
        if value not in output:
            output.append(value)
    if set(output) != set(FORMAL_CONDITIONS):
        raise ValueError(f"formal dataset requires both conditions: {FORMAL_CONDITIONS}")
    return output


def condition_to_id_map(conditions: Optional[Iterable[str]] = None) -> Dict[str, int]:
    return {name: index for index, name in enumerate(normalize_conditions(conditions))}


def visible_seed_from_extras(extras: Dict[str, Any]) -> int:
    explicit = extras.get("ccda_visible_seed")
    try:
        if explicit not in (None, "") and int(explicit) >= 0:
            return int(explicit)
    except (TypeError, ValueError, OverflowError):
        pass
    match = _VISIBLE_SEED_PATTERN.search(str(extras.get("ccda_pair_group", "")))
    return int(match.group("seed")) if match else -1


def extract_action_template_and_vector(action: Any, codec: Optional[ActionCodec] = None) -> Tuple[Any, np.ndarray, ActionCodec]:
    active = ExecutableActionCodec(action) if codec is None else codec
    return active.template, active.encode(action).astype(np.float32), active


def load_episode(condition: str, episode_path: Path, codec: Optional[ActionCodec] = None) -> Dict[str, Any]:
    payload = load_pickle(episode_path)
    infos = list(payload.get("infos", []))
    last_info = payload.get("last_info")
    raw_actions = list(payload.get("actions", []))
    if last_info is None or not infos or not raw_actions:
        raise ValueError(f"incomplete raw episode: {episode_path}")
    states, sources = [], []
    pre_engagement = []
    for info in infos + [last_info]:
        state, source, _ = state_v2_from_info(info)
        states.append(state)
        sources.append(source)
        contact = get_extras(info).get("hidden_contact_meta", {})
        pre_engagement.append(
            not isinstance(contact, dict)
            or contact.get("slack_engagement_physics_step") is None
        )
    action_vectors = []
    for action in raw_actions:
        _, vector, codec = extract_action_template_and_vector(action, codec)
        action_vectors.append(vector)
    states_array = np.asarray(states, dtype=np.float32)
    actions_array = np.asarray(action_vectors, dtype=np.float32)
    if states_array.shape[1] != STATE_DIM or actions_array.shape[1] != 14:
        raise ValueError("formal raw episode dimensions are invalid")
    metadata = dict(payload.get("manifest", {}))
    first = get_extras(infos[0])
    final = get_extras(last_info)
    hidden = final.get("hidden_contact_meta", {}) if isinstance(final.get("hidden_contact_meta", {}), dict) else {}
    fraction = final.get("total_rewards", 0.0)
    if final.get("nb_beads"):
        fraction = float(final.get("nb_zone", 0)) / float(final["nb_beads"])
    return {
        "condition": condition,
        "file_name": episode_path.name,
        "source_path": str(episode_path),
        "states": states_array,
        "actions": actions_array,
        "raw_actions": raw_actions,
        "codec": codec,
        "visible_seed": int(metadata.get("visible_seed", visible_seed_from_extras(first))),
        "pair_group": str(metadata.get("pair_group", first.get("ccda_pair_group", ""))),
        "success": bool(final.get("task.done", payload.get("success", False))),
        "final_fraction": float(fraction),
        "engagement_step": int(hidden.get("slack_engagement_physics_step") or -1),
        "release_step": int(hidden.get("slack_release_physics_step") or -1),
        "robot_pose_proxy_sources": sources,
        "pre_engagement": np.asarray(pre_engagement, dtype=np.bool_),
        "manifest": metadata,
    }


def condition_files(condition_dir: Path) -> List[Path]:
    return sorted(Path(condition_dir).glob("*.pkl"))


def build_windows_from_dataset(
    split_name: str,
    data_root: Path,
    th: int,
    tf: int,
    codec: Optional[ActionCodec] = None,
    max_windows_per_episode: int = 0,
    conditions: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[ActionCodec], Counter]:
    windows: List[Dict[str, Any]] = []
    sources: Counter = Counter()
    for condition in normalize_conditions(conditions):
        for episode_path in condition_files(Path(data_root) / condition):
            episode = load_episode(condition, episode_path, codec=codec)
            codec = episode["codec"]
            count = min(len(episode["actions"]), len(episode["states"]) - 1)
            if max_windows_per_episode > 0:
                count = min(count, int(max_windows_per_episode))
            for current in range(count):
                window = build_window(
                    states=episode["states"],
                    action_vectors=episode["actions"],
                    current_index=current,
                    th=th,
                    tf=tf,
                )
                sources.update(episode["robot_pose_proxy_sources"][: current + 1])
                windows.append(
                    {
                        **window,
                        "condition_name": condition,
                        "visible_seed": episode["visible_seed"],
                        "split_name": split_name,
                        "source_file": str(episode_path),
                        "pair_group": episode["pair_group"],
                        "window_t": int(current),
                        "success": episode["success"],
                        "final_fraction": episode["final_fraction"],
                        "engagement_step": episode["engagement_step"],
                        "release_step": episode["release_step"],
                        "pre_engagement": bool(episode["pre_engagement"][current]),
                    }
                )
    return windows, codec, sources


def save_action_template(path: Path, codec: ActionCodec) -> None:
    active = codec if isinstance(codec, ExecutableActionCodec) else ExecutableActionCodec(codec.template)
    summary = active.summary()
    if active.dim() != 14 or summary.get("num_camera_config_paths") != 0:
        raise ValueError(f"invalid executable codec: {summary}")
    active.save(path)


def load_action_codec_from_template(path: Path) -> ActionCodec:
    return ExecutableActionCodec.load(path)
