#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ccda_phase3.provenance_v2 import (
    build_source_lock,
    sha256_file,
    strict_json_dump,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--output",
        default="reports/phase3_13_r1_source_lock.json",
    )
    parser.add_argument(
        "--sha-output",
        default="reports/phase3_13_r1_source_lock.sha256",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / args.output
    sha_output = root / args.sha_output

    lock = build_source_lock(root)
    strict_json_dump(output, lock)
    digest = sha256_file(output)
    sha_output.parent.mkdir(parents=True, exist_ok=True)
    sha_output.write_text(digest + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_lock": str(output),
                "source_lock_sha256": digest,
                "main_commit": lock["main_commit"],
                "submodule_commit": lock["submodule_commit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
