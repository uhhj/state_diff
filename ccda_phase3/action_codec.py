#!/usr/bin/env python3
"""Executable action codec for Phase3 CCDA experiments.

The DeformableRavens action dict may contain observation metadata such as
camera_config. Those fields are not executable robot actions and must never
be used as inverse-dynamics targets.

This codec encodes only executable pick-place primitive parameters:
  - params.pose0 position + quaternion
  - params.pose1 position + quaternion

Expected vector length:
  pose0: xyz(3) + quat(4) = 7
  pose1: xyz(3) + quat(4) = 7
  total = 14

The codec preserves the full original action template for decoding, but only
replaces executable numeric leaves.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


PathType = Tuple[Any, ...]


class ExecutableActionCodec:
    """Encode/decode only executable action params.

    It intentionally ignores:
      - camera_config
      - observation config
      - seed / metadata
      - hidden condition
      - reward / success
      - object ids

    It supports action structures commonly seen in DeformableRavens:
      action["params"]["pose0"]
      action["params"]["pose1"]

    Fallback:
      action["pose0"]
      action["pose1"]
    """

    def __init__(self, template: Dict[str, Any]):
        self.template = copy.deepcopy(template)
        self.paths: List[PathType] = []
        self.path_strings: List[str] = []
        self._discover_paths()

        if not self.paths:
            raise ValueError(
                "ExecutableActionCodec found no executable pose paths. "
                "Expected params/pose0 and params/pose1 or pose0/pose1."
            )

        bad = [p for p in self.path_strings if "camera_config" in p]
        if bad:
            raise ValueError(f"ExecutableActionCodec must not encode camera_config paths: {bad[:5]}")

    def dim(self) -> int:
        return len(self.paths)

    def _discover_paths(self) -> None:
        candidate_roots = []

        if isinstance(self.template, dict):
            if "params" in self.template and isinstance(self.template["params"], dict):
                params = self.template["params"]
                if "pose0" in params:
                    candidate_roots.append(("params", "pose0"))
                if "pose1" in params:
                    candidate_roots.append(("params", "pose1"))

            if "pose0" in self.template:
                candidate_roots.append(("pose0",))
            if "pose1" in self.template:
                candidate_roots.append(("pose1",))

        for root in candidate_roots:
            obj = self._get_by_path(self.template, root)
            leaves = self._numeric_leaf_paths(obj, root)
            self.paths.extend(leaves)

        self.paths = list(dict.fromkeys(self.paths))
        self.path_strings = [self.path_to_str(p) for p in self.paths]

    def _numeric_leaf_paths(self, obj: Any, prefix: PathType) -> List[PathType]:
        out: List[PathType] = []

        if self._is_number(obj):
            out.append(prefix)
            return out

        if isinstance(obj, np.ndarray):
            flat = obj.reshape(-1)
            for i in range(flat.size):
                out.append(prefix + (("__ndarray_flat__", i, tuple(obj.shape)),))
            return out

        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                out.extend(self._numeric_leaf_paths(v, prefix + (i,)))
            return out

        if isinstance(obj, dict):
            for k in sorted(obj.keys(), key=lambda x: str(x)):
                out.extend(self._numeric_leaf_paths(obj[k], prefix + (k,)))
            return out

        return out

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))

    def encode(self, action: Dict[str, Any]) -> np.ndarray:
        vals = []
        for p in self.paths:
            vals.append(float(self._get_by_path(action, p)))
        return np.asarray(vals, dtype=np.float32)

    def decode(self, vec: Iterable[float]) -> Dict[str, Any]:
        action = self._mutable_copy(self.template)
        arr = np.asarray(list(vec), dtype=np.float32).reshape(-1)

        if arr.size != len(self.paths):
            raise ValueError(f"decode expected dim={len(self.paths)}, got dim={arr.size}")

        for val, p in zip(arr, self.paths):
            self._set_by_path(action, p, float(val))

        return action

    def roundtrip_error(self, action: Dict[str, Any]) -> float:
        v0 = self.encode(action)
        decoded = self.decode(v0)
        v1 = self.encode(decoded)
        return float(np.max(np.abs(v0 - v1))) if v0.size else 0.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ExecutableActionCodec":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def path_to_str(path: PathType) -> str:
        parts = []
        for p in path:
            if isinstance(p, tuple) and p and p[0] == "__ndarray_flat__":
                parts.append(f"ndarray_flat_{p[1]}")
            else:
                parts.append(str(p))
        return "/".join(parts)


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

    def _get_by_path(self, obj: Any, path: PathType) -> Any:
        cur = obj
        for step in path:
            if isinstance(step, tuple) and step and step[0] == "__ndarray_flat__":
                return cur.reshape(-1)[step[1]]
            cur = cur[step]
        return cur

    def _set_by_path(self, obj: Any, path: PathType, val: float) -> None:
        cur = obj
        for step in path[:-1]:
            cur = cur[step]

        last = path[-1]
        if isinstance(last, tuple) and last and last[0] == "__ndarray_flat__":
            flat = cur.reshape(-1)
            flat[last[1]] = val
        else:
            cur[last] = val

    def summary(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "dim": self.dim(),
            "paths": self.path_strings,
            "num_camera_config_paths": sum("camera_config" in p for p in self.path_strings),
            "num_param_paths": sum(p.startswith("params/") for p in self.path_strings),
        }


ActionCodec = ExecutableActionCodec
