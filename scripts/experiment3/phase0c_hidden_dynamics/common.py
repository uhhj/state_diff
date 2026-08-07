"""Active HLF-SBP Phase 0C helpers."""
from __future__ import annotations

from pathlib import Path

from state_diff.env.block_pushing.soft_block_task_config import (
    load_hlf_sbp_config)

CONDITIONS = ("uniform_low", "right_local_high")


def load_config(path: str) -> dict:
    config = load_hlf_sbp_config(path)
    if config["phase_name"] != "phase0c-hidden-dynamics-validity":
        raise ValueError("not a Phase 0C config")
    return config


def external_pair_dir(config: dict, condition: str) -> Path:
    return Path(config["output_root"]) / config["pair_id"] / condition


def report_pair_dir(config: dict) -> Path:
    return Path(config["report_root"]) / config["pair_id"]
