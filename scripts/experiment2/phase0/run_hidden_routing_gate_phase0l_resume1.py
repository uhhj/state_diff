#!/usr/bin/env python3
"""Run Phase 0L Resume1 after fixed geometry provenance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
)
from scripts.experiment2.phase0.run_hidden_routing_gate_phase0l import (
    main as phase0l_main,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/experiment2/phase0/"
            "hidden_routing_gate_phase0l_"
            "resume1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/"
            "phase0_hidden_routing_gate/"
            "phase0l_resume1"
        ),
    )
    args = parser.parse_args()

    config_path = (
        REPO_ROOT / args.config
    ).resolve()
    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )
    provenance_path = (
        REPO_ROOT
        / config[
            "required_provenance_report"
        ]
    ).resolve()
    if not provenance_path.is_file():
        raise RuntimeError(
            "required geometry provenance "
            "report is missing"
        )
    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )
    if not bool(provenance.get("passed")):
        raise RuntimeError(
            "geometry provenance did not pass"
        )
    if provenance[
        "resume_config_hash"
    ] != canonical_json_sha256(config):
        raise RuntimeError(
            "provenance was produced for "
            "a different Resume1 config"
        )

    original_argv = list(sys.argv)
    try:
        sys.argv = [
            original_argv[0],
            "--config",
            str(
                config_path.relative_to(
                    REPO_ROOT
                )
            ),
            "--output",
            str(args.output),
        ]
        phase0l_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

