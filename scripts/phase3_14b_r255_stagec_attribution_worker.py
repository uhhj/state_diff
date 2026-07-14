#!/usr/bin/env python3
"""Independent train-only attribution worker for Stage C."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r255_stagec_attribution import (
    run_train_only_attribution,
)
from ccda_phase3.phase314b_r255_stagec_cache import (
    atomic_write_once,
    load_npz_strict,
    stable_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    arrays = load_npz_strict(Path(args.train_view))
    report = run_train_only_attribution(arrays)
    atomic_write_once(Path(args.output), stable_json_bytes(report))
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "root_cause": report["root_cause"],
                "required_next_path": report["required_next_path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
