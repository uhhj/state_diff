#!/usr/bin/env python3
"""Isolated worker for one deterministic state-v3 materialization pass."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccda_phase3.phase314b_r255_stageb_dataset import build_state_v3_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = build_state_v3_artifacts(
        root=Path(args.root),
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "dataset_sha256": report["dataset_sha256"],
                "windows_sha256": report["windows_sha256"],
                "episode_count": report["episode_count"],
                "state_record_count": report["state_record_count"],
                "window_count": report["window_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
