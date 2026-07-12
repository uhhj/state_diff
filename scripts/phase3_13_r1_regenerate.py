#!/usr/bin/env python3
"""Regenerate the formal Phase3.13 dataset under an exact source lock."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ccda_phase3.provenance_v2 import (
    ATTESTATION_VERSION,
    FORMAL_SPLITS,
    sha256_file,
    strict_json_dump,
    strict_json_load,
    tree_merkle_root,
    verify_source_lock,
)


def run(
    root: Path,
    args: Sequence[str],
    *,
    env: Mapping[str, str],
) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(
        list(args),
        cwd=str(root),
        env=dict(env),
        check=True,
    )


def validate_raw_manifests(
    raw_root: Path,
    lock: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_conditions = (
        "free",
        "hidden_slack_breakaway_pin_v2",
    )
    manifests = sorted(raw_root.glob("*/*/*.manifest.json"))
    pickles = sorted(raw_root.glob("*/*/*.pkl"))
    expected_episodes = 2 * sum(
        int(spec["num_seeds"])
        for spec in FORMAL_SPLITS.values()
    )
    if len(manifests) != expected_episodes:
        raise RuntimeError(
            "manifest count %d != %d"
            % (len(manifests), expected_episodes)
        )
    if len(pickles) != expected_episodes:
        raise RuntimeError(
            "episode count %d != %d"
            % (len(pickles), expected_episodes)
        )

    observed: Dict[str, Dict[str, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    pair_actions: Dict[tuple, set] = defaultdict(set)
    for path in manifests:
        value = strict_json_load(path)
        split = str(value["split"])
        condition = str(value["condition"])
        seed = int(value["visible_seed"])
        expected = FORMAL_SPLITS[split]
        start = int(expected["seed_start"])
        stop = start + int(expected["num_seeds"])
        if not start <= seed < stop:
            raise RuntimeError(
                "seed %d is outside locked %s range" % (seed, split)
            )
        if condition not in expected_conditions:
            raise RuntimeError("unexpected condition %s" % condition)
        if value["main_commit"] != lock["main_commit"]:
            raise RuntimeError("episode main commit mismatch: %s" % path)
        if value["submodule_commit"] != lock["submodule_commit"]:
            raise RuntimeError(
                "episode submodule commit mismatch: %s" % path
            )
        contract = lock["formal_contract"]
        required = {
            "environment_semantics_version": contract[
                "environment_version"
            ],
            "observation_schema_version": contract["schema_version"],
            "task": contract["task"],
            "action_source_condition": "free",
            "contains_simulator_bead_velocity": False,
            "input_canonicalization": False,
        }
        for key, expected_value in required.items():
            if value.get(key) != expected_value:
                raise RuntimeError(
                    "%s mismatch in %s: %r != %r"
                    % (key, path, value.get(key), expected_value)
                )
        observed[split][condition].add(seed)
        pair_actions[(split, seed)].add(
            str(value["action_sequence_sha256"])
        )

    for split, spec in FORMAL_SPLITS.items():
        expected_seeds = set(
            range(
                int(spec["seed_start"]),
                int(spec["seed_start"])
                + int(spec["num_seeds"]),
            )
        )
        for condition in expected_conditions:
            if observed[split][condition] != expected_seeds:
                raise RuntimeError(
                    "raw seed coverage mismatch for %s/%s"
                    % (split, condition)
                )
    mismatched_actions = {
        "%s:%d" % key: sorted(values)
        for key, values in pair_actions.items()
        if len(values) != 1
    }
    if mismatched_actions:
        raise RuntimeError(
            "paired action hashes differ: %s" % mismatched_actions
        )

    return {
        "episode_count": len(pickles),
        "manifest_count": len(manifests),
        "pair_count": len(pair_actions),
        "paired_action_hashes_match": True,
        "split_seed_coverage": {
            split: {
                condition: len(observed[split][condition])
                for condition in expected_conditions
            }
            for split in FORMAL_SPLITS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--source-lock",
        default="reports/phase3_13_r1_source_lock.json",
    )
    parser.add_argument(
        "--staging-root",
        default="data/phase3_state_v2_slack_r1_staging",
    )
    parser.add_argument(
        "--audit-prefix",
        default="reports/phase3_13_r1",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if os.environ.get("PHASE313_R1_ALLOW_REGENERATION") != "1":
        raise SystemExit("PHASE313_R1_ALLOW_REGENERATION must be 1")
    if os.environ.get("PHASE313_R1_REGENERATION_CONFIRMED") != "1":
        raise SystemExit("PHASE313_R1_REGENERATION_CONFIRMED must be 1")

    root = Path(args.root).resolve()
    lock_path = root / args.source_lock
    lock = strict_json_load(lock_path)
    verify_source_lock(
        root,
        lock,
        require_exact_heads=True,
        require_clean_tracked_worktree=True,
    )

    staging = root / args.staging_root
    if staging.exists():
        if not args.fresh:
            raise SystemExit(
                "staging exists; use --fresh with the explicit replacement "
                "gate to restart"
            )
        if (
            os.environ.get("PHASE313_R1_ALLOW_REPLACE_STAGING")
            != "1"
        ):
            raise SystemExit(
                "PHASE313_R1_ALLOW_REPLACE_STAGING must be 1"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root),
            str(root / "scripts"),
            str(root / "external/deformable-ravens"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PHASE313_ALLOW_DATA_GENERATION"] = "1"
    env["PHASE313_DATA_GENERATION_CONFIRMED"] = "1"

    raw_relative = str(
        (staging / "raw").relative_to(root)
    )
    for split in ("train", "val", "test"):
        verify_source_lock(
            root,
            lock,
            require_exact_heads=True,
            require_clean_tracked_worktree=True,
        )
        spec = FORMAL_SPLITS[split]
        run(
            root,
            [
                sys.executable,
                "scripts/phase3_13_generate_raw.py",
                "--root",
                str(root),
                "--split",
                split,
                "--seed-start",
                str(spec["seed_start"]),
                "--num-seeds",
                str(spec["num_seeds"]),
                "--output-root",
                raw_relative,
                "--workers",
                str(max(1, int(args.workers))),
                "--fresh",
            ],
            env=env,
        )
        verify_source_lock(
            root,
            lock,
            require_exact_heads=True,
            require_clean_tracked_worktree=True,
        )

    raw_validation = validate_raw_manifests(staging / "raw", lock)

    verify_source_lock(
        root,
        lock,
        require_exact_heads=True,
        require_clean_tracked_worktree=True,
    )
    run(
        root,
        [
            sys.executable,
            "scripts/phase3_13_build_windows.py",
            "--root",
            str(root),
            "--raw-root",
            str((staging / "raw").relative_to(root)),
            "--output-dir",
            str((staging / "windows").relative_to(root)),
            "--manifest",
            str((staging / "manifest.json").relative_to(root)),
        ],
        env=env,
    )

    manifest = strict_json_load(staging / "manifest.json")
    if Path(str(manifest["window_npz"])).is_absolute():
        raise RuntimeError("manifest window_npz must be relative")
    if Path(str(manifest["action_template"])).is_absolute():
        raise RuntimeError("manifest action_template must be relative")
    if manifest["main_commit"] != lock["main_commit"]:
        raise RuntimeError("window manifest main commit mismatch")
    if manifest["submodule_commit"] != lock["submodule_commit"]:
        raise RuntimeError("window manifest submodule commit mismatch")

    for relative, expected_hash in manifest[
        "source_code_sha256"
    ].items():
        locked_hash = lock["main_source_sha256"].get(relative)
        if locked_hash is None:
            raise RuntimeError(
                "builder recorded an unlocked source path: %s" % relative
            )
        if expected_hash != locked_hash:
            raise RuntimeError(
                "builder/lock source hash mismatch for %s" % relative
            )

    verify_source_lock(
        root,
        lock,
        require_exact_heads=True,
        require_clean_tracked_worktree=True,
    )
    run(
        root,
        [
            sys.executable,
            "scripts/phase3_13_audit_dataset.py",
            "--root",
            str(root),
            "--raw-root",
            str((staging / "raw").relative_to(root)),
            "--windows",
            str(
                (
                    staging
                    / "windows/phase3_13_windows.npz"
                ).relative_to(root)
            ),
            "--template",
            str(
                (
                    staging
                    / "windows/action_template.pkl"
                ).relative_to(root)
            ),
            "--out-prefix",
            args.audit_prefix,
        ],
        env=env,
    )

    audit_summary = root / (
        args.audit_prefix + "_dataset_audit_summary.json"
    )
    audit = strict_json_load(audit_summary)
    if audit.get("verdict") != "PASS":
        raise RuntimeError("staging Phase3.13 audit did not pass")
    if (
        audit.get("root_cause")
        != "phase313_state_v2_dataset_supported"
    ):
        raise RuntimeError("unexpected staging audit root cause")

    raw_merkle, raw_files = tree_merkle_root(
        staging / "raw",
        suffixes=(".pkl", ".json"),
    )
    attestation = {
        "attestation_version": ATTESTATION_VERSION,
        "source_lock_path": str(
            Path(args.source_lock).as_posix()
        ),
        "source_lock_sha256": sha256_file(lock_path),
        "generator_main_commit": lock["main_commit"],
        "generator_submodule_commit": lock["submodule_commit"],
        "main_source_sha256": lock["main_source_sha256"],
        "submodule_source_sha256": lock[
            "submodule_source_sha256"
        ],
        "raw_merkle_root": raw_merkle,
        "raw_file_count": len(raw_files),
        "raw_validation": raw_validation,
        "windows_sha256": sha256_file(
            staging / "windows/phase3_13_windows.npz"
        ),
        "action_template_sha256": sha256_file(
            staging / "windows/action_template.pkl"
        ),
        "manifest_sha256": sha256_file(
            staging / "manifest.json"
        ),
        "audit_summary_sha256": sha256_file(audit_summary),
        "audit_verdict": audit["verdict"],
        "audit_root_cause": audit["root_cause"],
        "formal_training_allowed": True,
        "phase4_allowed": False,
    }
    strict_json_dump(
        staging / "provenance_attestation.json",
        attestation,
    )
    verify_source_lock(
        root,
        lock,
        require_exact_heads=True,
        require_clean_tracked_worktree=True,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "staging_root": str(staging),
                "raw_merkle_root": raw_merkle,
                "windows_sha256": attestation["windows_sha256"],
                "source_lock_sha256": attestation[
                    "source_lock_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
