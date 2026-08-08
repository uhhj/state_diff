"""Config loading for the DHR control-only smoke."""
from __future__ import annotations

import json
from pathlib import Path


def load_config(path):
    config = json.loads(
        Path(path).read_text(
            encoding="utf-8"))

    if config.get("task_name") != "ccda-dhr-cable-smoke":
        raise ValueError("unexpected DHR task")

    if tuple(config.get("conditions", ())) != (
            "free", "jam_right"):
        raise ValueError(
            "DHR smoke requires FREE and JAM-R only")

    if not config["execution"]["deterministic"]:
        raise ValueError(
            "DHR smoke must be deterministic")

    return config
