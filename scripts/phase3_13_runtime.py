#!/usr/bin/env python3
"""TensorFlow-free runtime helpers for Phase3.13 data generation."""
from __future__ import annotations

import contextlib
import importlib
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import pybullet as p


@contextlib.contextmanager
def temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    previous: Dict[str, Optional[str]] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def install_minimal_ravens(root: Path):
    submodule = Path(root) / "external/deformable-ravens"
    if str(submodule) not in sys.path:
        sys.path.insert(0, str(submodule))
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    package = types.ModuleType("ravens")
    package.__path__ = [str(submodule / "ravens")]
    package.__package__ = "ravens"
    sys.modules["ravens"] = package
    try:
        import pkg_resources
        original = pkg_resources.get_distribution
        try:
            original("pybullet")
        except Exception:
            class Distribution:
                version = "3.0.4"
            pkg_resources.get_distribution = lambda name: Distribution() if str(name) == "pybullet" else original(name)
    except Exception:
        pass
    tasks = importlib.import_module("ravens.tasks")
    environment = importlib.import_module("ravens.environment").Environment
    if "tensorflow" in sys.modules:
        raise RuntimeError("TensorFlow loaded in formal data runtime")
    return tasks, environment


def direct_physics_step(env: Any, task: Any, *, dispatch_hooks: bool = True) -> None:
    lock = getattr(env, "_ccda_step_lock", None)
    context = lock if lock is not None else contextlib.nullcontext()
    with context:
        if dispatch_hooks:
            hook = getattr(task, "physics_pre_step_hook", None)
            if callable(hook):
                hook()
        p.stepSimulation()
        if getattr(env, "ee", None) is not None:
            env.ee.step()
        if dispatch_hooks:
            hook = getattr(task, "physics_step_hook", None)
            if callable(hook):
                hook()


def reward_info(env: Any, task: Any) -> Dict[str, Any]:
    _, extras = task.reward()
    extras["task.done"] = bool(task.done())
    info = env.info
    info["extras"] = extras
    return info


def deterministic_reset(env: Any, task: Any) -> Dict[str, Any]:
    if hasattr(task, "_settle_secs"):
        task._settle_secs = 0.0
    original_start, original_step = env.start, env.step
    env.pause()
    env.start = lambda: setattr(env, "running", False)
    env.step = lambda act=None: ({}, 0.0, False, {})
    with temporary_environment(
        {
            "CCDA_SETTLE_SECONDS": "0",
            "CCDA_POST_ARM_SETTLE_SECONDS": "0",
            "CCDA_DEFER_HIDDEN_CONTACT_ARMING": "1",
        }
    ):
        try:
            env.reset(task)
        finally:
            env.start, env.step = original_start, original_step
            env.pause()
        p.setTimeStep(1.0 / float(env.hz))
        for _ in range(540):
            direct_physics_step(env, task, dispatch_hooks=False)
        for bead in task.cable_bead_IDs:
            p.resetBaseVelocity(int(bead), linearVelocity=(0, 0, 0), angularVelocity=(0, 0, 0))
        if hasattr(task, "_ccda_physics_step_count"):
            task._ccda_physics_step_count = 0
        arm = task.arm_hidden_contact_after_settle(env)
        info = reward_info(env, task)
    return {"info": info, "arm": arm, "settle_steps": 540}


def close_env(env: Any) -> None:
    try:
        env.pause()
        env.running = False
        env.ee = None
        time.sleep(0.02)
        env.stop()
    except Exception:
        pass
