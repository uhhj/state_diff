#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap(root: Path) -> None:
    value = str(Path(root).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--repository-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    _bootstrap(args.root)
    from ccda_phase3 import phase314b_r259_stagej_prevalence_anchored_risk_repair as stagej

    if args.import_smoke:
        print(stagej.SCHEMA)
        return 0
    probe = stagej.load_json(args.probe)
    payload = stagej.run_worker(
        root=args.root,
        probe_payload=probe,
        repository_head=args.repository_head,
    )
    stagej.atomic_write_once(args.output, stagej.stable_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
