#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def transform(src: Path, dst: Path) -> None:
    text = src.read_text()
    required = [
        "phase3_12d_r1_common",
        "p12c.deterministic_reset",
        "phase3_12d_r1",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"Unexpected r1 query script; missing tokens: {missing}")

    text = text.replace("phase3_12d_r1_common", "phase3_12d_r2_common")
    text = text.replace(
        "p12c.deterministic_reset(",
        "common.deterministic_reset_deferred_arm(",
    )
    text = text.replace("phase3_12d_r1", "phase3_12d_r2")

    # The r2 reset helper accepts all r1 deterministic-reset keywords and adds
    # safe defaults arm_after_settle=True/post_arm_steps=0. No candidate or
    # snapshot logic is otherwise changed.
    forbidden = [
        "phase3_12d_r1_common",
        "p12c.deterministic_reset(",
        "phase3_12d_r1_",
    ]
    leftovers = [token for token in forbidden if token in text]
    if leftovers:
        raise RuntimeError(f"Incomplete r2 query transform: {leftovers}")

    dst.write_text(text)
    print(f"[r2 prepare] wrote {dst}")


def copy_analyzer(src: Path, dst: Path) -> None:
    text = src.read_text()
    if "phase3_12d_r1" not in text:
        raise RuntimeError(f"Unexpected r1 analyzer: {src}")
    text = text.replace("phase3_12d_r1", "phase3_12d_r2")
    dst.write_text(text)
    print(f"[r2 prepare] wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scripts = root / "scripts"
    transform(
        scripts / "phase3_12d_r1_query_local_snapshot.py",
        scripts / "phase3_12d_r2_query_local_snapshot.py",
    )
    copy_analyzer(
        scripts / "phase3_12d_r1_analyze.py",
        scripts / "phase3_12d_r2_analyze.py",
    )


if __name__ == "__main__":
    main()
