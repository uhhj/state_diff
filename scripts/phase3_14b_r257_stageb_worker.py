#!/usr/bin/env python3
"""Isolated r2.5.7 Stage-B mechanism-attribution worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r257_stageb_mechanism_audit import (
    atomic_write_once,
    run_mechanism_audit,
    stable_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_mechanism_audit(root=Path(args.root))
    output = Path(args.output).resolve()
    atomic_write_once(output, stable_json_bytes(result))
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "root_cause": result["root_cause"],
                "required_next_path": result["required_next_path"],
                "mechanism_contract_sha256":
                    result["mechanism_contract"][
                        "contract_sha256"
                    ],
                "best_lambda_by_continuous_response":
                    result["mechanism_summary"][
                        "best_lambda_by_t10_mean_excess"
                    ],
                "best_t10_mean_excess_reduction":
                    result["mechanism_summary"][
                        "best_t10_mean_excess_reduction"
                    ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
