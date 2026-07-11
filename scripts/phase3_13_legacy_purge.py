#!/usr/bin/env python3
"""Inventory and remove deprecated CCDA assets from the current branch tip.

This script deliberately does NOT rewrite Git history. Tracked files removed
with ``git rm`` remain recoverable from earlier commits. Untracked local
artifacts are permanently deleted only in ``--apply`` mode and only after the
manifest has been written.

Run from the main repository root.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CONFIRMATION = "DELETE_PHASE3_LEGACY_ASSETS"
SUBMODULE_REL = Path("external/deformable-ravens")

# Tracked main-repository assets superseded by Phase3.13.
MAIN_TRACKED_DELETE = (
    "scripts/phase0_*",
    "scripts/phase1_*",
    "scripts/phase2_*",
    "scripts/phase2_5*",
    "scripts/phase3_*",
    "tests/test_phase3_12d_*",
    "reports/*",
)

# Phase3.13 files are created before the purge and must survive broad patterns.
MAIN_TRACKED_KEEP = (
    "scripts/phase3_13_*",
    "tests/test_phase3_13_*",
    "reports/phase3_13_*",
    "reports/README.md",
    "GOALS.md",
    "LEGACY_ASSET_REMOVAL.md",
    "data/phase3_state_v2_slack*",
)

# Local, normally ignored artifacts known to be legacy.
MAIN_UNTRACKED_DELETE = (
    "ccda_phase3/__pycache__",
    "scripts/__pycache__",
    "tests/__pycache__",
    "scripts/phase0_*",
    "scripts/phase1_*",
    "scripts/phase2_*",
    "scripts/phase2_5*",
    "scripts/phase3_*",
    "reports/.ipynb_checkpoints",
    "checkpoints/phase3*",
    "checkpoints/ccda*",
    "data/phase1*",
    "data/phase2*",
    "data/phase3_*",
    "data/ccda*",
    "reports/*workers*",
    "reports/phase0_*",
    "reports/phase1_*",
    "reports/phase2_*",
    "reports/phase3_*",
)

# Files superseded by the clean v2-only submodule surface.
SUBMODULE_TRACKED_DELETE = (
    "ccda_continuous_video.py",
    "ccda_generate_hidden_contact.py",
    "ccda_record_continuous_rollout.py",
    "ccda_recoverability_audit.py",
    "ravens/tasks/ccda_hidden_contact_cable.py",
)

SUBMODULE_UNTRACKED_DELETE = (
    "__pycache__",
    "ravens/__pycache__",
    "ravens/tasks/__pycache__",
    "data/hidden-contact-cable-line*",
    "data/phase1*",
    "data/phase2*",
    "data/phase3*",
    "data/ccda*",
    "logs/*hidden-contact*",
    "logs/*ccda*",
    "goals/*hidden-contact*",
)

REQUIRED_NEW_MAIN = (
    "ccda_phase3/schema_v2.py",
    "scripts/phase3_13_generate_raw.py",
    "scripts/phase3_13_build_windows.py",
    "scripts/phase3_13_audit_dataset.py",
    "scripts/phase3_13_runtime.py",
    "scripts/phase3_13_run.sh",
    "tests/test_ccda_slack_breakaway.py",
    "tests/test_ccda_state_v2.py",
    "tests/test_ccda_dataset_windows.py",
    "tests/test_phase3_13_legacy_purge.py",
)

REQUIRED_NEW_SUBMODULE = (
    "ravens/tasks/ccda_slack_cable_v2.py",
    "ravens/tasks/ccda_slack_breakaway.py",
)


@dataclass(frozen=True)
class AssetRecord:
    repository: str
    path: str
    tracked: bool
    size_bytes: int
    sha256: str
    reason: str
    action: str


def run(
    cwd: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git_files(root: Path, *, tracked: bool) -> List[str]:
    if tracked:
        command = ["git", "ls-files", "-z"]
    else:
        command = [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    result = subprocess.run(
        command,
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        fnmatch.fnmatchcase(normalized, pattern)
        for pattern in patterns
    )


def covered_by_record(path: str, records: Sequence[AssetRecord]) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        normalized == record.path
        or normalized.startswith(record.path.rstrip("/") + "/")
        for record in records
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_for(
    repository: str,
    root: Path,
    relative: str,
    *,
    tracked: bool,
    reason: str,
) -> AssetRecord:
    path = root / relative
    if path.is_file() or path.is_symlink():
        size = path.stat().st_size
        checksum = sha256_file(path) if path.is_file() else "symlink"
    else:
        size = 0
        checksum = "directory"
    return AssetRecord(
        repository=repository,
        path=relative,
        tracked=tracked,
        size_bytes=int(size),
        sha256=checksum,
        reason=reason,
        action="git_rm" if tracked else "delete_local",
    )


def expand_untracked_patterns(
    root: Path,
    patterns: Sequence[str],
) -> List[str]:
    results: set[str] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if ".git" in path.parts:
                continue
            results.add(path.relative_to(root).as_posix())
    return sorted(results)


def strict_dump(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assert_required(root: Path, relative_paths: Sequence[str]) -> None:
    missing = [
        relative for relative in relative_paths
        if not (root / relative).exists()
    ]
    if missing:
        raise RuntimeError(
            "New replacement assets must exist before purge: "
            + ", ".join(missing)
        )


def classify_main(root: Path) -> Tuple[List[AssetRecord], List[str]]:
    records: List[AssetRecord] = []
    unknown: List[str] = []

    for relative in git_files(root, tracked=True):
        if matches_any(relative, MAIN_TRACKED_KEEP):
            continue
        if matches_any(relative, MAIN_TRACKED_DELETE):
            records.append(
                record_for(
                    "main",
                    root,
                    relative,
                    tracked=True,
                    reason=(
                        "superseded phase-specific script/report/test; "
                        "preserved in Git history"
                    ),
                )
            )

    for relative in expand_untracked_patterns(
        root,
        MAIN_UNTRACKED_DELETE,
    ):
        if matches_any(relative, MAIN_TRACKED_KEEP):
            continue
        records.append(
            record_for(
                "main",
                root,
                relative,
                tracked=False,
                reason="legacy local CCDA data/checkpoint/report artifact",
            )
        )

    # Detect suspicious legacy-looking paths not covered by the allow-list.
    for relative in git_files(root, tracked=False):
        normalized = relative.replace("\\", "/")
        looks_legacy = (
            normalized.startswith(("data/", "checkpoints/", "reports/"))
            and any(
                token in normalized.lower()
                for token in (
                    "phase0",
                    "phase1",
                    "phase2",
                    "phase3_",
                    "phase3-",
                    "ccda",
                    "hidden-contact",
                    "worker",
                    "candidate",
                )
            )
        )
        if (
            looks_legacy
            and not matches_any(normalized, MAIN_TRACKED_KEEP)
            and not covered_by_record(normalized, records)
        ):
            unknown.append(normalized)

    return records, sorted(set(unknown))


def classify_submodule(root: Path) -> Tuple[List[AssetRecord], List[str]]:
    records: List[AssetRecord] = []
    unknown: List[str] = []

    for relative in git_files(root, tracked=True):
        if matches_any(relative, SUBMODULE_TRACKED_DELETE):
            records.append(
                record_for(
                    "submodule",
                    root,
                    relative,
                    tracked=True,
                    reason=(
                        "legacy rigid/diagnostic CCDA surface replaced "
                        "by v2-only task"
                    ),
                )
            )

    for relative in expand_untracked_patterns(
        root,
        SUBMODULE_UNTRACKED_DELETE,
    ):
        records.append(
            record_for(
                "submodule",
                root,
                relative,
                tracked=False,
                reason="legacy local DeformableRavens CCDA artifact",
            )
        )

    for relative in git_files(root, tracked=False):
        normalized = relative.replace("\\", "/")
        looks_legacy = (
            normalized.startswith(("data/", "logs/", "goals/"))
            and any(
                token in normalized.lower()
                for token in (
                    "phase",
                    "ccda",
                    "hidden-contact",
                    "breakaway-pin",
                )
            )
        )
        if (
            looks_legacy
            and not covered_by_record(normalized, records)
        ):
            unknown.append(normalized)

    return records, sorted(set(unknown))


def delete_untracked(root: Path, relative: str) -> None:
    path = root / relative
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def git_rm(root: Path, paths: Sequence[str]) -> None:
    if not paths:
        return
    # Chunk to avoid shell/argument limits.
    for start in range(0, len(paths), 100):
        chunk = list(paths[start:start + 100])
        run(
            root,
            ["git", "rm", "-r", "--ignore-unmatch", "--", *chunk],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--manifest",
        default="reports/phase3_13_legacy_purge_manifest.json",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    submodule = root / SUBMODULE_REL

    if run(root, ["git", "branch", "--show-current"]).stdout.strip() != (
        "Experiment1"
    ):
        raise SystemExit("main branch must be Experiment1")
    if run(
        submodule,
        ["git", "branch", "--show-current"],
    ).stdout.strip() != "ccda-cable":
        raise SystemExit("submodule branch must be ccda-cable")

    assert_required(root, REQUIRED_NEW_MAIN)
    assert_required(submodule, REQUIRED_NEW_SUBMODULE)

    main_records, main_unknown = classify_main(root)
    sub_records, sub_unknown = classify_submodule(submodule)

    unknown = {
        "main": main_unknown,
        "submodule": sub_unknown,
    }
    if main_unknown or sub_unknown:
        strict_dump(
            root / args.manifest,
            {
                "verdict": "BLOCKED_UNKNOWN_LEGACY_ASSETS",
                "unknown": unknown,
                "records": [
                    asdict(record)
                    for record in [*main_records, *sub_records]
                ],
            },
        )
        raise SystemExit(
            "Unknown legacy-looking assets require explicit classification; "
            "see purge manifest"
        )

    payload: Dict[str, object] = {
        "verdict": "DRY_RUN" if not args.apply else "APPLY_REQUESTED",
        "history_rewrite": False,
        "tracked_assets_recoverable_from_git_history": True,
        "untracked_assets_permanently_deleted_on_apply": bool(args.apply),
        "main_head_before": run(
            root,
            ["git", "rev-parse", "HEAD"],
        ).stdout.strip(),
        "submodule_head_before": run(
            submodule,
            ["git", "rev-parse", "HEAD"],
        ).stdout.strip(),
        "records": [
            asdict(record)
            for record in [*main_records, *sub_records]
        ],
        "counts": {
            "main_tracked": sum(
                1 for record in main_records if record.tracked
            ),
            "main_untracked": sum(
                1 for record in main_records if not record.tracked
            ),
            "submodule_tracked": sum(
                1 for record in sub_records if record.tracked
            ),
            "submodule_untracked": sum(
                1 for record in sub_records if not record.tracked
            ),
            "bytes_total": sum(
                record.size_bytes for record in [*main_records, *sub_records]
            ),
        },
        "unknown": unknown,
    }
    strict_dump(root / args.manifest, payload)

    if not args.apply:
        print(json.dumps(payload["counts"], indent=2, sort_keys=True))
        print("Dry run only. No files deleted.")
        return

    if os.environ.get("PHASE313_LEGACY_PURGE_CONFIRM", "") != CONFIRMATION:
        raise SystemExit(
            "Set PHASE313_LEGACY_PURGE_CONFIRM="
            + CONFIRMATION
            + " to execute deletion"
        )

    git_rm(
        submodule,
        [record.path for record in sub_records if record.tracked],
    )
    for record in sub_records:
        if not record.tracked:
            delete_untracked(submodule, record.path)

    git_rm(
        root,
        [record.path for record in main_records if record.tracked],
    )
    for record in main_records:
        if not record.tracked:
            delete_untracked(root, record.path)

    remaining_main, remaining_main_unknown = classify_main(root)
    remaining_sub, remaining_sub_unknown = classify_submodule(submodule)
    if (
        remaining_main
        or remaining_sub
        or remaining_main_unknown
        or remaining_sub_unknown
    ):
        raise RuntimeError(
            "Purge incomplete; remaining legacy assets detected"
        )

    payload["verdict"] = "APPLIED"
    payload["post_checks"] = {
        "legacy_main_remaining": 0,
        "legacy_submodule_remaining": 0,
        "unknown_remaining": 0,
    }
    strict_dump(root / args.manifest, payload)
    print("Legacy asset purge applied successfully.")


if __name__ == "__main__":
    main()
