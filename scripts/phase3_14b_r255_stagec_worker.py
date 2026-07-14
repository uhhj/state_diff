#!/usr/bin/env python3
"""Independent worker for the Stage-C state-v3 cache build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r255_stagec_cache import build_cache_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    payload = build_cache_artifacts(
        root=Path(args.root),
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "cache_sha256": payload["cache_sha256"],
                "train_view_sha256": payload["train_attribution_view_sha256"],
                "rows": payload["rows"],
                "train_rows": payload["train_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
