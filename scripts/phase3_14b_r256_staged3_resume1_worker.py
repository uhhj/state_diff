#!/usr/bin/env python3
"""Isolated Stage-D.3 Resume1 one-sided upper-gate worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_staged3_resume1_upper_gate_freeze import (
    atomic_write_once,
    run_freeze_and_reaudit,
    stable_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_freeze_and_reaudit(root=Path(args.root))
    output = Path(args.output).resolve()
    atomic_write_once(output, stable_json_bytes(result))
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "root_cause": result["root_cause"],
                "required_next_path": result["required_next_path"],
                "blocked_evidence_commit":
                    result["immutable_inputs"]["blocked_attempt"][
                        "blocked_evidence_commit"
                    ],
                "upper_contract_sha256":
                    result["upper_gate_contract"]["contract_sha256"],
                "upper_threshold":
                    result["upper_gate_contract"]["upper_threshold"],
                "reverse_candidate_sha256":
                    result["frozen_replay"]["identity"][
                        "reverse_candidate_sha256"
                    ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
