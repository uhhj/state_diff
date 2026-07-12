#!/usr/bin/env python3
"""Build the immutable Phase3.14a training cache from formal Phase3.13-r1."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ccda_phase3.data_io import load_action_codec_from_template
from ccda_phase3.phase314a_contract import (
    ACTION_DIM,
    CACHE_ROWS,
    CACHE_VERSION,
    DEFAULT_TF,
    DEFAULT_TH,
    PAPER_X_DIM,
    STATE_ACTION_X_DIM,
    STATE_DIM,
    action_history_valid_mask,
    canonical_source_path,
    current_state_from_paper_x,
    fit_masked_future_standardizer,
    fit_standardizer,
    future_valid_mask,
    load_npz_no_pickle,
    load_raw_action_count,
    sha256_array,
    sha256_file,
    state_history_valid_mask,
    strict_json_dump,
    strict_json_load,
    validate_formal_windows,
    validate_state_quaternions,
    verify_source_sidecar,
)
from ccda_phase3.provenance_v2 import (
    git_head,
    resolve_manifest_artifact,
)


def run_provenance_gate(root: Path) -> Dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/phase3_14_provenance_gate.py",
            "--root",
            str(root),
        ],
        cwd=str(root),
        check=True,
    )
    path = root / "reports/phase3_14_provenance_gate_summary.json"
    value = strict_json_load(path)
    if value.get("verdict") != "PASS":
        raise RuntimeError("Phase3.14 provenance gate is not PASS")
    if value.get("root_cause") != "phase314_data_provenance_supported":
        raise RuntimeError("unexpected Phase3.14 provenance root cause")
    if value.get("formal_training_allowed") is not True:
        raise RuntimeError("formal training is not allowed")
    return value


def action_quaternion_indices(codec_summary: Dict[str, Any]) -> Dict[str, list]:
    paths = [str(value) for value in codec_summary.get("paths", [])]
    result = {
        "pose0": [
            index
            for index, path in enumerate(paths)
            if path.startswith("params/pose0/1/")
        ],
        "pose1": [
            index
            for index, path in enumerate(paths)
            if path.startswith("params/pose1/1/")
        ],
    }
    for key, indices in result.items():
        if len(indices) != 4:
            raise ValueError(
                f"{key} quaternion path count is {len(indices)}, not 4"
            )
        if indices != sorted(indices):
            raise ValueError(f"{key} quaternion indices are not ordered")
    return result


def quaternion_stats(value: np.ndarray, indices: list) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float32)
    quaternion = array[:, indices]
    norms = np.linalg.norm(quaternion, axis=1)
    if not np.all(np.isfinite(norms)):
        raise ValueError("action quaternion norms are non-finite")
    error = float(np.max(np.abs(norms - 1.0)))
    if error > 1e-3:
        raise ValueError(
            f"action quaternion max norm error {error} exceeds 1e-3"
        )
    return {
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "max_abs_norm_error": error,
    }


def split_isolation(arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    splits = arrays["split_name"].astype(str)
    seeds = arrays["visible_seed"].astype(np.int64)
    groups = arrays["pair_group"].astype(str)
    sources = arrays["canonical_source_file"].astype(str)
    result: Dict[str, int] = {}
    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        left_mask = splits == left
        right_mask = splits == right
        for name, values in (
            ("seed", seeds),
            ("pair_group", groups),
            ("source_file", sources),
        ):
            overlap = set(values[left_mask].tolist()).intersection(
                set(values[right_mask].tolist())
            )
            result[f"{left}_{right}_{name}_overlap"] = len(overlap)
    if any(result.values()):
        raise ValueError(f"cache split isolation failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--formal-root",
        default="data/phase3_state_v2_slack",
    )
    parser.add_argument(
        "--output-dir",
        default="data/phase3_14_cache",
    )
    parser.add_argument(
        "--cache-name",
        default="phase3_14a_training_cache.npz",
    )
    parser.add_argument(
        "--manifest-name",
        default="phase3_14a_training_cache_manifest.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14a_cache_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14a_cache_report.md",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    formal_root = (root / args.formal_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    cache_path = output_dir / args.cache_name
    cache_manifest_path = output_dir / args.manifest_name

    if cache_path.exists() or cache_manifest_path.exists():
        if not args.replace:
            raise SystemExit(
                "training cache already exists; replacement is blocked"
            )
        if os.environ.get("PHASE314A_ALLOW_CACHE_REPLACE") != "1":
            raise SystemExit("PHASE314A_ALLOW_CACHE_REPLACE must be 1")

    provenance = run_provenance_gate(root)
    formal_manifest_path = formal_root / "manifest.json"
    formal_attestation_path = formal_root / "provenance_attestation.json"
    formal_manifest = strict_json_load(formal_manifest_path)
    formal_attestation = strict_json_load(formal_attestation_path)

    windows_path = resolve_manifest_artifact(
        formal_manifest_path,
        formal_manifest["window_npz"],
    )
    template_path = resolve_manifest_artifact(
        formal_manifest_path,
        formal_manifest["action_template"],
    )
    arrays = load_npz_no_pickle(windows_path)
    formal_validation = validate_formal_windows(arrays)

    rows = formal_validation["rows"]
    splits = arrays["split_name"].astype(str)
    conditions = arrays["condition_name"].astype(str)
    recorded_sources = arrays["source_file"].astype(str)
    seeds = arrays["visible_seed"].astype(np.int64)
    groups = arrays["pair_group"].astype(str)
    window_t = arrays["window_t"].astype(np.int64)

    future_masks = np.zeros((rows, DEFAULT_TF), dtype=np.bool_)
    state_history_masks = np.zeros(
        (rows, DEFAULT_TH),
        dtype=np.bool_,
    )
    action_history_masks = np.zeros(
        (rows, DEFAULT_TH),
        dtype=np.bool_,
    )
    raw_action_counts = np.zeros(rows, dtype=np.int64)
    canonical_sources = []
    source_cache: Dict[Tuple[str, str, str], Tuple[Path, int, bool]] = {}
    relinked_rows = 0
    relinked_unique_sources = set()

    for index in range(rows):
        key = (
            splits[index],
            conditions[index],
            recorded_sources[index],
        )
        if key not in source_cache:
            source_path, relinked = canonical_source_path(
                formal_root,
                split=splits[index],
                condition=conditions[index],
                recorded_source=recorded_sources[index],
            )
            verify_source_sidecar(
                source_path,
                split=splits[index],
                condition=conditions[index],
                visible_seed=int(seeds[index]),
                pair_group=groups[index],
            )
            action_count = load_raw_action_count(source_path)
            source_cache[key] = (
                source_path,
                action_count,
                relinked,
            )
        source_path, action_count, relinked = source_cache[key]
        # Verify seed/group for each row even when a source path is cached.
        verify_source_sidecar(
            source_path,
            split=splits[index],
            condition=conditions[index],
            visible_seed=int(seeds[index]),
            pair_group=groups[index],
        )
        raw_action_counts[index] = action_count
        future_masks[index] = future_valid_mask(
            action_count,
            int(window_t[index]),
        )
        state_history_masks[index] = state_history_valid_mask(
            int(window_t[index])
        )
        action_history_masks[index] = action_history_valid_mask(
            int(window_t[index])
        )
        canonical_relative = source_path.relative_to(
            formal_root
        ).as_posix()
        canonical_sources.append(canonical_relative)
        if relinked:
            relinked_rows += 1
            relinked_unique_sources.add(canonical_relative)

    current_state = current_state_from_paper_x(arrays["paper_x"])
    next_state = arrays["y_state"][:, 0, :].copy()
    idm_pair_x = np.concatenate(
        [current_state, next_state],
        axis=1,
    ).astype(np.float32)
    idm_trajectory_x = np.concatenate(
        [
            arrays["paper_x"],
            arrays["y_state"].reshape(rows, -1),
        ],
        axis=1,
    ).astype(np.float32)

    pair_key = np.asarray(
        [
            f"{split}|{int(seed)}|{int(time)}"
            for split, seed, time in zip(splits, seeds, window_t)
        ],
        dtype="<U64",
    )
    pair_counts = Counter(pair_key.tolist())
    invalid_pair_counts = {
        key: count for key, count in pair_counts.items() if count != 2
    }
    if invalid_pair_counts:
        raise ValueError(
            "formal paired windows are incomplete: "
            f"{list(invalid_pair_counts.items())[:10]}"
        )

    output_arrays: Dict[str, np.ndarray] = {
        **{key: np.asarray(value) for key, value in arrays.items()},
        "row_index": np.arange(rows, dtype=np.int64),
        "future_valid_mask": future_masks,
        "state_history_valid_mask": state_history_masks,
        "action_history_valid_mask": action_history_masks,
        "raw_action_count": raw_action_counts,
        "canonical_source_file": np.asarray(
            canonical_sources,
            dtype="<U256",
        ),
        "pair_key": pair_key,
        "current_state": current_state.astype(np.float32),
        "next_state": next_state.astype(np.float32),
        "idm_pair_x": idm_pair_x,
        "idm_trajectory_x": idm_trajectory_x,
        "direct_action_x": arrays["state_action_x"].copy(),
    }

    train = splits == "train"
    if int(np.sum(train)) != 2440:
        raise ValueError(
            f"train row count {np.sum(train)} != 2440"
        )
    paper_standardizer = fit_standardizer(
        arrays["paper_x"][train]
    )
    state_action_standardizer = fit_standardizer(
        arrays["state_action_x"][train]
    )
    future_standardizer = fit_masked_future_standardizer(
        arrays["y_state"][train],
        future_masks[train],
    )
    output_arrays.update(paper_standardizer.to_npz("paper_x"))
    output_arrays.update(
        state_action_standardizer.to_npz("state_action_x")
    )
    output_arrays.update(future_standardizer.to_npz("future"))

    state_quaternion = validate_state_quaternions(
        np.concatenate(
            [
                arrays["paper_x"].reshape(
                    rows * DEFAULT_TH,
                    STATE_DIM,
                ),
                arrays["y_state"].reshape(
                    rows * DEFAULT_TF,
                    STATE_DIM,
                ),
            ],
            axis=0,
        )
    )
    codec = load_action_codec_from_template(template_path)
    codec_summary = codec.summary()
    if codec.dim() != ACTION_DIM:
        raise ValueError("action codec dimension is not 14")
    if codec_summary.get("num_camera_config_paths") != 0:
        raise ValueError("camera_config paths are present")
    quaternion_indices = action_quaternion_indices(codec_summary)
    action_quaternion = {
        key: quaternion_stats(arrays["y_action"], indices)
        for key, indices in quaternion_indices.items()
    }

    cache_validation = validate_formal_windows(output_arrays)
    isolation = split_isolation(output_arrays)
    if any(value.dtype.kind == "O" for value in output_arrays.values()):
        raise ValueError("cache contains object arrays")
    for key, value in output_arrays.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"cache array {key} is non-finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".{args.cache_name}.tmp.npz"
    if temporary.exists():
        temporary.unlink()
    np.savez_compressed(temporary, **output_arrays)
    with np.load(temporary, allow_pickle=False) as checked:
        if set(checked.files) != set(output_arrays):
            raise RuntimeError("temporary cache keys changed")
        if checked["paper_x"].shape[0] != CACHE_ROWS:
            raise RuntimeError("temporary cache row count changed")
    os.replace(temporary, cache_path)

    main_commit = git_head(root)
    submodule_commit = git_head(
        root / "external/deformable-ravens"
    )
    phase314a_source_paths = (
        "ccda_phase3/phase314a_contract.py",
        "ccda_phase3/phase314a_metrics.py",
        "scripts/phase3_14a_build_training_cache.py",
        "scripts/phase3_14a_preflight.py",
        "scripts/phase3_14a_train_deterministic.py",
        "scripts/phase3_14a_analyze.py",
        "scripts/phase3_14a_run.sh",
    )
    cache_manifest = {
        "cache_version": CACHE_VERSION,
        "cache_file": cache_path.name,
        "cache_sha256": sha256_file(cache_path),
        "cache_size_bytes": int(cache_path.stat().st_size),
        "formal_root": str(Path(args.formal_root).as_posix()),
        "formal_windows_sha256": sha256_file(windows_path),
        "formal_manifest_sha256": sha256_file(
            formal_manifest_path
        ),
        "formal_attestation_sha256": sha256_file(
            formal_attestation_path
        ),
        "formal_action_template_sha256": sha256_file(
            template_path
        ),
        "formal_raw_merkle_root": formal_attestation[
            "raw_merkle_root"
        ],
        "source_lock_sha256": formal_attestation[
            "source_lock_sha256"
        ],
        "generator_main_commit": formal_attestation[
            "generator_main_commit"
        ],
        "generator_submodule_commit": formal_attestation[
            "generator_submodule_commit"
        ],
        "cache_builder_main_commit": main_commit,
        "cache_builder_submodule_commit": submodule_commit,
        "phase314a_source_sha256": {
            relative: sha256_file(root / relative)
            for relative in phase314a_source_paths
        },
        "phase314_provenance_root_cause": provenance[
            "root_cause"
        ],
        "formal_validation": formal_validation,
        "cache_validation": cache_validation,
        "split_isolation": isolation,
        "rows": rows,
        "split_rows": {
            split: int(np.sum(splits == split))
            for split in ("train", "val", "test")
        },
        "split_visible_seeds": {
            split: int(np.unique(seeds[splits == split]).size)
            for split in ("train", "val", "test")
        },
        "future_valid_counts": {
            str(index + 1): int(
                np.sum(np.sum(future_masks, axis=1) == index + 1)
            )
            for index in range(DEFAULT_TF)
        },
        "future_padding_fraction": float(
            1.0 - np.mean(future_masks.astype(np.float64))
        ),
        "state_history_padding_fraction": float(
            1.0 - np.mean(
                state_history_masks.astype(np.float64)
            )
        ),
        "action_history_padding_fraction": float(
            1.0 - np.mean(
                action_history_masks.astype(np.float64)
            )
        ),
        "source_path_relink": {
            "rows": relinked_rows,
            "unique_sources": len(relinked_unique_sources),
            "policy": (
                "formal_root/raw/<split>/<condition>/<basename>"
            ),
            "sidecars_verified": True,
        },
        "pair_keys": len(pair_counts),
        "pair_rows_per_key": 2,
        "codec_summary": codec_summary,
        "action_quaternion_indices": quaternion_indices,
        "state_quaternion": state_quaternion,
        "action_quaternion": action_quaternion,
        "array_sha256": {
            key: sha256_array(value)
            for key, value in output_arrays.items()
        },
        "training_cache_immutable": True,
        "test_split_selection_allowed": False,
        "ddpm_training": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(cache_manifest_path, cache_manifest)
    strict_json_dump(root / args.summary, {
        "verdict": "PASS",
        "root_cause": "phase314a_immutable_training_cache_supported",
        **cache_manifest,
    })

    lines = [
        "# Phase3.14a Immutable Training Cache",
        "",
        "- Verdict: `PASS`",
        (
            "- Root cause: "
            "`phase314a_immutable_training_cache_supported`"
        ),
        f"- Rows: `{rows}`",
        f"- Cache SHA256: `{cache_manifest['cache_sha256']}`",
        (
            "- Source rows relinked after staging promotion: "
            f"`{relinked_rows}`"
        ),
        "",
        "| Split | Rows | Visible seeds |",
        "|---|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| `{split}` | "
            f"{cache_manifest['split_rows'][split]} | "
            f"{cache_manifest['split_visible_seeds'][split]} |"
        )
    lines += [
        "",
        f"- Future padding fraction: "
        f"`{cache_manifest['future_padding_fraction']:.8f}`",
        "- No model training or candidate execution was run.",
        "- Phase4 and CPS remain blocked.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "verdict": "PASS",
                "root_cause": (
                    "phase314a_immutable_training_cache_supported"
                ),
                "cache": str(cache_path),
                "cache_sha256": cache_manifest["cache_sha256"],
                "rows": rows,
                "relinked_rows": relinked_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
