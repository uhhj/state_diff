import copy
import math
from typing import Any, List, Tuple

import numpy as np


class ActionCodec:
    """Flatten and reconstruct DeformableRavens action dicts.

    The template preserves non-numeric fields such as primitive name. Numeric
    leaves are replaced by values from the predicted vector. Tuple containers
    are decoded as lists, which PyBullet and DeformableRavens accept for poses.
    """

    def __init__(self, template: Any):
        self.template = copy.deepcopy(template)
        self.paths: List[Tuple[Any, ...]] = []
        self._collect_paths(self.template, [])

    def _is_number(self, x: Any) -> bool:
        return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))

    def _collect_paths(self, obj: Any, path: List[Any]) -> None:
        if self._is_number(obj):
            self.paths.append(tuple(path))
            return
        if isinstance(obj, np.ndarray):
            flat = obj.reshape(-1)
            for i in range(flat.size):
                self.paths.append(tuple(path + [("ndarray", i, obj.shape)]))
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._collect_paths(v, path + [i])
            return
        if isinstance(obj, dict):
            for k in sorted(obj.keys(), key=lambda z: str(z)):
                if str(k) == "primitive":
                    continue
                self._collect_paths(obj[k], path + [k])

    def dim(self) -> int:
        return len(self.paths)

    def encode(self, obj: Any) -> np.ndarray:
        vals = []
        for p in self.paths:
            vals.append(float(self._get(obj, p)))
        return np.asarray(vals, dtype=np.float32)

    def decode(self, vec: Any) -> Any:
        out = self._mutable_copy(self.template)
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.size < len(self.paths):
            vec = np.pad(vec, (0, len(self.paths) - vec.size), mode="constant")
        for val, p in zip(vec, self.paths):
            self._set(out, p, float(val))
        return self._restore_action_types(out)

    def _restore_action_types(self, obj: Any) -> Any:
        """Restore schema details that DeformableRavens expects."""
        if isinstance(obj, dict) and isinstance(obj.get("camera_config"), list):
            for cfg in obj["camera_config"]:
                if isinstance(cfg, dict) and "image_size" in cfg:
                    cfg["image_size"] = tuple(int(round(float(x))) for x in cfg["image_size"])
        return obj

    def _mutable_copy(self, obj: Any) -> Any:
        if isinstance(obj, tuple):
            return [self._mutable_copy(x) for x in obj]
        if isinstance(obj, list):
            return [self._mutable_copy(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._mutable_copy(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return np.array(obj, copy=True)
        return copy.deepcopy(obj)

    def _get(self, obj: Any, path: Tuple[Any, ...]) -> Any:
        cur = obj
        for step in path:
            if isinstance(step, tuple) and step[0] == "ndarray":
                return cur.reshape(-1)[step[1]]
            cur = cur[step]
        return cur

    def _set(self, obj: Any, path: Tuple[Any, ...], val: float) -> Any:
        cur = obj
        for step in path[:-1]:
            cur = cur[step]
        last = path[-1]
        if isinstance(last, tuple) and last[0] == "ndarray":
            flat = cur.reshape(-1)
            flat[last[1]] = val
        else:
            cur[last] = val
        return obj
