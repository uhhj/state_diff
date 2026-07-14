#!/usr/bin/env python3
"""Run original r2.5.4 attribution under the committed functional-prior contract."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Dict, Mapping, MutableMapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r254_resume2_functional_prior import (
    CURRENT_DEVICE_PRIOR_STATE_SHA256,
    ORIGINAL_R254_PREFLIGHT_PATH,
    RESUME1_AUDIT_PATH,
    RESUME1_EVIDENCE_PATH,
    RESUME1_SUMMARY_PATH,
    RESUME2_PILOT_PATH,
    RESUME2_PREFLIGHT_PATH,
    augment_pilot_payload,
    load_json,
    source_sha256,
    validate_fresh_prior_snapshot,
    verify_resume1_functional_contract,
)


def load_original_runner(root: Path) -> ModuleType:
    path = root / "scripts/phase3_14b_r254_run_pilot.py"
    spec = importlib.util.spec_from_file_location("phase314b_r254_original_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load original r2.5.4 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--preflight-report", default=RESUME2_PREFLIGHT_PATH)
    parser.add_argument("--output", default=RESUME2_PILOT_PATH)
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--residual-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--residual-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--calibration-batches", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight_path = Path(args.preflight_report)
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite Resume2 pilot")
    if (root / "reports/phase3_14b_r254_pilot_summary.json").exists():
        raise RuntimeError("unexpected standard r2.5.4 pilot already exists")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Resume2 preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume2 source changed after preflight")

    functional_contract = verify_resume1_functional_contract(
        load_json(root / RESUME1_SUMMARY_PATH),
        load_json(root / RESUME1_AUDIT_PATH),
        load_json(root / RESUME1_EVIDENCE_PATH),
    )

    module = load_original_runner(root)
    module.EXPECTED_SHARED_PAIRED_PRIOR_SHA256 = CURRENT_DEVICE_PRIOR_STATE_SHA256

    original_fit = module.fit_shared_prior_snapshot
    fresh_validation: Dict[str, Any] = {}

    def validated_fit(*fit_args: Any, **fit_kwargs: Any) -> MutableMapping[str, Any]:
        snapshot = original_fit(*fit_args, **fit_kwargs)
        if not isinstance(snapshot, MutableMapping):
            raise RuntimeError("original prior fit returned a non-mapping snapshot")
        validation = validate_fresh_prior_snapshot(snapshot)
        fresh_validation.clear()
        fresh_validation.update(validation)
        return snapshot

    module.fit_shared_prior_snapshot = validated_fit

    original_assert = module.assert_only_allowed_worktree_paths

    def allowed_worktree(root_arg: Path, allowed: Any) -> None:
        merged = tuple(allowed) + (preflight_path.relative_to(root).as_posix(),)
        original_assert(root_arg, merged)

    module.assert_only_allowed_worktree_paths = allowed_worktree

    original_write = module.write_json_once
    original_target = (root / "reports/phase3_14b_r254_pilot_summary.json").resolve()

    def redirected_write(path: Path, payload: Mapping[str, Any]) -> None:
        resolved = Path(path).resolve()
        if resolved != original_target:
            original_write(path, payload)
            return
        if not fresh_validation:
            raise RuntimeError("fresh-prior validation was not recorded before pilot write")
        augmented = augment_pilot_payload(
            payload,
            functional_contract=functional_contract,
            fresh_prior_validation=fresh_validation,
            resume2_source_hashes=source_sha256(root),
            resume2_preflight_path=preflight_path.relative_to(root).as_posix(),
        )
        original_write(output, augmented)

    module.write_json_once = redirected_write

    old_argv = list(sys.argv)
    try:
        sys.argv = [
            str(root / "scripts/phase3_14b_r254_run_pilot.py"),
            "--root",
            str(root),
            "--preflight-report",
            ORIGINAL_R254_PREFLIGHT_PATH,
            "--prior-steps",
            str(args.prior_steps),
            "--residual-steps",
            str(args.residual_steps),
            "--batch-size",
            str(args.batch_size),
            "--prior-learning-rate",
            str(args.prior_learning_rate),
            "--residual-learning-rate",
            str(args.residual_learning_rate),
            "--calibration-batches",
            str(args.calibration_batches),
        ]
        module.main()
    finally:
        sys.argv = old_argv

    if not output.is_file():
        raise RuntimeError("Resume2 pilot output was not created")
    if original_target.exists():
        raise RuntimeError("adapter wrote the forbidden standard pilot path")
    print(json.dumps({"verdict": "PASS", "pilot": output.relative_to(root).as_posix()}, sort_keys=True))


if __name__ == "__main__":
    main()
