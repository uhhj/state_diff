from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
DLO_ROOT = REPO_ROOT / "external" / "dlo-lab"
DLO_EXPERIMENTS = DLO_ROOT / "experiments"


def add_dlolab_to_path():
    path = str(DLO_EXPERIMENTS)

    if path not in sys.path:
        sys.path.insert(
            0,
            path,
        )


def required_asset_path():
    return (
        DLO_ROOT
        / "genesis"
        / "assets"
        / "dlo-lab"
        / "target_pos"
        / "wiring_post_finalpos.npy"
    )


def official_log_dir():
    return (
        DLO_EXPERIMENTS
        / "logs"
        / "wiring_post"
        / "cmaes-01"
    )
