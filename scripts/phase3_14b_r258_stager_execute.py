#!/usr/bin/env python3
"""Execute Stage-R tolerance-aligned OOF post-upper audit exactly once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit import (
    BLOCKED_REPORT,
    SUCCESS_REPORT,
    StageRError,
    blocked_report,
    run_oof_post_upper_audit,
    stable_json_bytes,
    validate_environment_variables,
    validate_repository,
)


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageRError("environment payload is not a mapping")
    return value


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageRError(f"write-once output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--environment-json", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-R output already exists", file=sys.stderr)
        return 2

    repository = None
    try:
        validate_environment_variables()
        repository = validate_repository(root)
        environment = load_json(Path(args.environment_json).resolve())
        result = run_oof_post_upper_audit(
            root=root,
            environment=environment,
            repository=repository,
        )
        write_once(success, stable_json_bytes(result))
        audit = result["tolerance_aligned_oof_post_upper_audit"]
        classification = audit["classification"]
        print(
            json.dumps(
                {
                    "execution_verdict": result["execution_verdict"],
                    "scientific_status": result["scientific_status"],
                    "root_cause": result["root_cause"],
                    "required_next_path": result["required_next_path"],
                    "scientific_result_sha256": result[
                        "scientific_result_sha256"
                    ],
                    "oof_cell_count": audit["oof_cell_count"],
                    "callback_pair_count": audit[
                        "legacy_callback_off_on_pair_count"
                    ],
                    "meaningful_support_cell_count": classification[
                        "meaningful_support_cell_count"
                    ],
                    "dominant_discriminator": classification[
                        "dominant_dual_oracle_discriminator"
                    ],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = blocked_report(repository=repository, error=error)
        try:
            write_once(blocked, stable_json_bytes(payload))
        except BaseException as write_error:
            print(
                f"BLOCKED: {error}; additionally failed to write blocked "
                f"evidence: {write_error}",
                file=sys.stderr,
            )
            return 2
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
