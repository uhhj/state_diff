#!/usr/bin/env python3
"""Executable action codec for Phase3 CCDA experiments."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PathType = Tuple[Any, ...]


class ExecutableActionCodec:
    """Encode/decode only executable DeformableRavens pick-place params.

    Encoded fields:
      - params/pose0: xyz + quat
      - params/pose1: xyz + quat

    Ignored fields:
      - camera_config
      - observation config
      - reward/success
      - hidden condition
      - object ids
      - seeds and metadata
    """

    def __init__(self, template: Dict[str, Any]):
        self.template = copy.deepcopy(template)
        self.paths: List[PathType] = []
        self.path_strings: List[str] = []
        self._discover_paths()
        if not self.paths:
            raise ValueError("No executable pose paths found. Expected params/pose0 and params/pose1.")
        bad = [p for p in self.path_strings if "camera_config" in p]
        if bad:
            raise ValueError(f"camera_config must not be encoded: {bad[:5]}")
        if self.dim() != 14:
            raise ValueError(f"Expected executable action dim 14, got {self.dim()} with paths {self.path_strings}")

    def _discover_paths(self) -> None:
        roots = []
        if isinstance(self.template, dict):
            params = self.template.get("params", None)
            if isinstance(params, dict):
                if "pose0" in params:
                    roots.append(("params", "pose0"))
                if "pose1" in params:
                    roots.append(("params", "pose1"))
            if "pose0" in self.template:
                roots.append(("pose0",))
            if "pose1" in self.template:
                roots.append(("pose1",))

        for root in roots:
            self.paths.extend(self._numeric_leaf_paths(self._get_by_path(self.template, root), root))

        self.paths = list(dict.fromkeys(self.paths))
        self.path_strings = [self.path_to_str(p) for p in self.paths]

    def _numeric_leaf_paths(self, obj: Any, prefix: PathType) -> List[PathType]:
        out: List[PathType] = []
        if isinstance(obj, (int, float, np.integer, np.floating)) and np.isfinite(float(obj)):
            out.append(prefix)
        elif isinstance(obj, np.ndarray):
            for i in range(obj.reshape(-1).size):
                out.append(prefix + (("__ndarray_flat__", i, tuple(obj.shape)),))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                out.extend(self._numeric_leaf_paths(v, prefix + (i,)))
        elif isinstance(obj, dict):
            for k in sorted(obj.keys(), key=lambda z: str(z)):
                out.extend(self._numeric_leaf_paths(obj[k], prefix + (k,)))
        return out

    def dim(self) -> int:
        return len(self.paths)

    def encode(self, action: Dict[str, Any]) -> np.ndarray:
        return np.asarray([float(self._get_by_path(action, p)) for p in self.paths], dtype=np.float32)

    def decode(self, vec: Iterable[float]) -> Dict[str, Any]:
        arr = np.asarray(list(vec), dtype=np.float32).reshape(-1)
        if arr.size != self.dim():
            raise ValueError(f"decode expected dim={self.dim()}, got {arr.size}")
        action = self._mutable_copy(self.template)
        for val, path in zip(arr, self.paths):
            self._set_by_path(action, path, float(val))
        return action

    def roundtrip_error(self, action: Dict[str, Any]) -> float:
        v0 = self.encode(action)
        v1 = self.encode(self.decode(v0))
        return float(np.max(np.abs(v0 - v1))) if v0.size else 0.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ExecutableActionCodec":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ExecutableActionCodec):
            raise TypeError(f"Expected ExecutableActionCodec pickle, got {type(obj).__name__}")
        if obj.dim() != 14 or obj.summary().get("num_camera_config_paths", 0) != 0:
            raise ValueError(f"Invalid ExecutableActionCodec summary: {obj.summary()}")
        return obj

    @staticmethod
    def path_to_str(path: PathType) -> str:
        parts = []
        for p in path:
            if isinstance(p, tuple) and p[0] == "__ndarray_flat__":
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
            if isinstance(step, tuple) and step[0] == "__ndarray_flat__":
                return cur.reshape(-1)[step[1]]
            cur = cur[step]
        return cur

    def _set_by_path(self, obj: Any, path: PathType, value: float) -> None:
        cur = obj
        for step in path[:-1]:
            cur = cur[step]
        last = path[-1]
        if isinstance(last, tuple) and last[0] == "__ndarray_flat__":
            cur.reshape(-1)[last[1]] = value
        else:
            cur[last] = value

    def summary(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "dim": self.dim(),
            "paths": self.path_strings,
            "num_paths": len(self.path_strings),
            "num_camera_config_paths": sum("camera_config" in p for p in self.path_strings),
            "num_param_paths": sum(p.startswith("params/") for p in self.path_strings),
        }


ActionCodec = ExecutableActionCodec
