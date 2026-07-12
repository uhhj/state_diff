#!/usr/bin/env python3
"""Compare invalid-provenance and regenerated Phase3.13 datasets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from ccda_phase3.provenance_v2 import (
    compare_npz,
    sha256_file,
    strict_json_dump,
    strict_json_load,
    tree_merkle_root,
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def dataset_summary(root: Path) -> Dict[str, Any]:
    manifest_path = root / "manifest.json"
    windows = root / "windows/phase3_13_windows.npz"
    template = root / "windows/action_template.pkl"
    raw_merkle, raw_files = tree_merkle_root(
        root / "raw",
        suffixes=(".pkl", ".json"),
    )
    manifest = strict_json_load(manifest_path)
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "windows_sha256": sha256_file(windows),
        "action_template_sha256": sha256_file(template),
        "raw_merkle_root": raw_merkle,
        "raw_file_count": len(raw_files),
        "main_commit": manifest.get("main_commit"),
        "submodule_commit": manifest.get("submodule_commit"),
        "source_code_sha256": manifest.get(
            "source_code_sha256",
            {},
        ),
        "split_window_counts": manifest.get(
            "split_window_counts",
            {},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--old-root",
        default="data/phase3_state_v2_slack",
    )
    parser.add_argument(
        "--new-root",
        default="data/phase3_state_v2_slack_r1_staging",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_13_r1_dataset_comparison.json",
    )
    parser.add_argument(
        "--csv",
        default="reports/phase3_13_r1_array_comparison.csv",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_13_r1_dataset_comparison.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    old_root = root / args.old_root
    new_root = root / args.new_root
    old_summary = dataset_summary(old_root)
    new_summary = dataset_summary(new_root)
    arrays = compare_npz(
        old_root / "windows/phase3_13_windows.npz",
        new_root / "windows/phase3_13_windows.npz",
    )

    payload = {
        "old": old_summary,
        "new": new_summary,
        "array_comparison": arrays,
        "new_dataset_preferred": True,
        "reason": (
            "The old artifact lacks strict source provenance. "
            "The new artifact is authoritative whenever its full "
            "Phase3.13 audit and provenance gate pass."
        ),
    }
    strict_json_dump(root / args.output, payload)
    write_csv(root / args.csv, arrays["rows"])

    changed = [
        row["key"] for row in arrays["rows"]
        if not row["exact_equal"]
    ]
    lines = [
        "# Phase3.13-r1 Dataset Comparison",
        "",
        f"- All shapes match: `{arrays['all_shapes_match']}`",
        f"- All dtypes match: `{arrays['all_dtypes_match']}`",
        f"- All arrays exact: `{arrays['all_arrays_exact']}`",
        f"- Changed arrays: `{', '.join(changed) if changed else 'none'}`",
        "",
        "The regenerated dataset is authoritative because it is tied to "
        "the frozen source lock. Exact old/new equality is diagnostic, "
        "not a promotion requirement.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
