"""Strict provenance helpers for Phase3.13-r1 and Phase3.14.

The formal dataset is generated under a frozen source lock.  A later
report-only commit is allowed only when the locked generator commit remains an
ancestor and every locked source file is byte-identical.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LOCK_VERSION = "phase313_r1_source_lock_v1"
ATTESTATION_VERSION = "phase313_r1_dataset_attestation_v1"

MAIN_SOURCE_PATHS = (
    "ccda_phase3/schema_v2.py",
    "ccda_phase3/data_io.py",
    "ccda_phase3/action_codec.py",
    "scripts/phase3_13_runtime.py",
    "scripts/phase3_13_generate_raw.py",
    "scripts/phase3_13_build_windows.py",
    "scripts/phase3_13_audit_dataset.py",
    "ccda_phase3/provenance_v2.py",
    "scripts/phase3_13_r1_create_source_lock.py",
    "scripts/phase3_13_r1_regenerate.py",
    "scripts/phase3_13_r1_verify.py",
)

SUBMODULE_SOURCE_PATHS = (
    "ravens/environment.py",
    "ravens/tasks/__init__.py",
    "ravens/tasks/ccda_slack_breakaway.py",
    "ravens/tasks/ccda_slack_cable_v2.py",
)

FORMAL_SPLITS = {
    "train": {"seed_start": 400000, "num_seeds": 256},
    "val": {"seed_start": 410000, "num_seeds": 64},
    "test": {"seed_start": 420000, "num_seeds": 128},
}


def _run(
    cwd: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def git_output(root: Path, *args: str) -> str:
    return _run(root, ["git", *args]).stdout.strip()


def git_head(root: Path) -> str:
    return git_output(root, "rev-parse", "HEAD")


def git_branch(root: Path) -> str:
    return git_output(root, "branch", "--show-current")


def tracked_worktree_clean(root: Path) -> bool:
    unstaged = _run(root, ["git", "diff", "--quiet"], check=False).returncode
    staged = _run(
        root,
        ["git", "diff", "--cached", "--quiet"],
        check=False,
    ).returncode
    return unstaged == 0 and staged == 0


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return (
        _run(
            root,
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
        ).returncode
        == 0
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def merkle_root(
    root: Path,
    relative_paths: Iterable[str],
) -> str:
    """Hash sorted relative path names and file hashes."""
    digest = hashlib.sha256()
    for relative in sorted(set(str(item) for item in relative_paths)):
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        name = relative.replace("\\", "/").encode("utf-8")
        file_hash = bytes.fromhex(sha256_file(path))
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(file_hash)
    return digest.hexdigest()


def tree_merkle_root(
    root: Path,
    *,
    suffixes: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    allowed = None if suffixes is None else set(suffixes)
    paths = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if allowed is not None and path.suffix not in allowed:
            continue
        paths.append(path.relative_to(base).as_posix())
    if not paths:
        raise ValueError("no files for Merkle root under %s" % base)
    return merkle_root(base, paths), sorted(paths)


def _jsonable(value: Any, path: str = "root") -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value), path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("%s contains NaN or Inf" % path)
        return _jsonable(value.tolist(), path)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), path)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, "%s.%s" % (path, key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item, "%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("%s is non-finite" % path)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("%s has unsupported type %s" % (path, type(value)))


def strict_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _jsonable(dict(payload)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def strict_json_load(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    _jsonable(value)
    return value


def source_hashes(root: Path, paths: Sequence[str]) -> Dict[str, str]:
    result = {}
    for relative in paths:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(relative)] = sha256_file(path)
    return result


def runtime_versions() -> Dict[str, Any]:
    import sys

    result: Dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
    }
    for name in ("numpy", "torch", "pybullet", "scipy"):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            result[name] = {"error": repr(exc)}
    return result


def build_source_lock(root: Path) -> Dict[str, Any]:
    main = Path(root).resolve()
    submodule = main / "external/deformable-ravens"
    if git_branch(main) != "Experiment1":
        raise RuntimeError("main branch must be Experiment1")
    if git_branch(submodule) != "ccda-cable":
        raise RuntimeError("submodule branch must be ccda-cable")
    if not tracked_worktree_clean(main):
        raise RuntimeError("main tracked worktree must be clean")
    if not tracked_worktree_clean(submodule):
        raise RuntimeError("submodule tracked worktree must be clean")

    from ccda_phase3.schema_v2 import (
        ACTION_DIM,
        DEFAULT_TF,
        DEFAULT_TH,
        ENVIRONMENT_VERSION,
        FORMAL_CONDITIONS,
        FORMAL_TASK_NAME,
        PAPER_X_DIM,
        SCHEMA_VERSION,
        STATE_ACTION_X_DIM,
        STATE_DIM,
    )

    return {
        "lock_version": LOCK_VERSION,
        "main_branch": "Experiment1",
        "submodule_branch": "ccda-cable",
        "main_commit": git_head(main),
        "submodule_commit": git_head(submodule),
        "main_source_sha256": source_hashes(main, MAIN_SOURCE_PATHS),
        "submodule_source_sha256": source_hashes(
            submodule,
            SUBMODULE_SOURCE_PATHS,
        ),
        "runtime": runtime_versions(),
        "formal_contract": {
            "schema_version": SCHEMA_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "task": FORMAL_TASK_NAME,
            "conditions": list(FORMAL_CONDITIONS),
            "state_dim": int(STATE_DIM),
            "paper_x_dim": int(PAPER_X_DIM),
            "state_action_x_dim": int(STATE_ACTION_X_DIM),
            "action_dim": int(ACTION_DIM),
            "th": int(DEFAULT_TH),
            "tf": int(DEFAULT_TF),
            "contains_simulator_bead_velocity": False,
            "input_canonicalization": False,
        },
        "splits": FORMAL_SPLITS,
        "exact_heads_required_during_generation": True,
        "report_only_descendant_commit_allowed_after_generation": True,
    }


def verify_source_lock(
    root: Path,
    lock: Mapping[str, Any],
    *,
    require_exact_heads: bool,
    require_clean_tracked_worktree: bool = True,
) -> Dict[str, Any]:
    main = Path(root).resolve()
    submodule = main / "external/deformable-ravens"
    if lock.get("lock_version") != LOCK_VERSION:
        raise ValueError("unsupported source lock version")

    current_main = git_head(main)
    current_submodule = git_head(submodule)
    locked_main = str(lock["main_commit"])
    locked_submodule = str(lock["submodule_commit"])

    if require_exact_heads:
        if current_main != locked_main:
            raise RuntimeError(
                "main HEAD changed during generation: %s != %s"
                % (current_main, locked_main)
            )
        if current_submodule != locked_submodule:
            raise RuntimeError(
                "submodule HEAD changed during generation"
            )
    else:
        if not is_ancestor(main, locked_main, current_main):
            raise RuntimeError(
                "locked generator commit is not an ancestor of current HEAD"
            )
        if current_submodule != locked_submodule:
            raise RuntimeError(
                "formal submodule commit changed after generation"
            )

    if require_clean_tracked_worktree:
        if not tracked_worktree_clean(main):
            raise RuntimeError("main tracked source has local modifications")
        if not tracked_worktree_clean(submodule):
            raise RuntimeError(
                "submodule tracked source has local modifications"
            )

    actual_main = source_hashes(
        main,
        tuple(lock["main_source_sha256"].keys()),
    )
    actual_submodule = source_hashes(
        submodule,
        tuple(lock["submodule_source_sha256"].keys()),
    )
    main_mismatch = {
        key: {
            "expected": lock["main_source_sha256"][key],
            "actual": actual_main[key],
        }
        for key in actual_main
        if actual_main[key] != lock["main_source_sha256"][key]
    }
    submodule_mismatch = {
        key: {
            "expected": lock["submodule_source_sha256"][key],
            "actual": actual_submodule[key],
        }
        for key in actual_submodule
        if actual_submodule[key]
        != lock["submodule_source_sha256"][key]
    }
    if main_mismatch or submodule_mismatch:
        raise RuntimeError(
            "locked source hash mismatch: main=%s submodule=%s"
            % (main_mismatch, submodule_mismatch)
        )

    return {
        "main_commit_locked": locked_main,
        "main_commit_current": current_main,
        "submodule_commit": current_submodule,
        "exact_heads": current_main == locked_main,
        "main_source_hashes_match": True,
        "submodule_source_hashes_match": True,
        "tracked_worktrees_clean": True,
    }


def resolve_manifest_artifact(
    manifest_path: Path,
    value: str,
) -> Path:
    """Resolve an artifact path relative to the manifest parent.

    Absolute paths are rejected for the regenerated formal dataset.
    """
    raw = Path(str(value))
    if raw.is_absolute():
        raise ValueError(
            "absolute artifact path is not relocatable: %s" % raw
        )
    candidate = (Path(manifest_path).parent / raw).resolve()
    parent = Path(manifest_path).parent.resolve()
    if parent != candidate and parent not in candidate.parents:
        raise ValueError("artifact path escapes dataset root")
    return candidate


def load_npz_no_pickle(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as loaded:
        result = {key: loaded[key] for key in loaded.files}
    if any(value.dtype.kind == "O" for value in result.values()):
        raise ValueError("object array found in formal windows")
    return result


def compare_npz(
    old_path: Path,
    new_path: Path,
) -> Dict[str, Any]:
    old = load_npz_no_pickle(old_path)
    new = load_npz_no_pickle(new_path)
    keys = sorted(set(old).union(new))
    rows: List[Dict[str, Any]] = []
    all_shapes_match = True
    all_dtypes_match = True
    for key in keys:
        if key not in old or key not in new:
            rows.append(
                {
                    "key": key,
                    "present_old": key in old,
                    "present_new": key in new,
                    "shape_match": False,
                    "dtype_match": False,
                    "exact_equal": False,
                    "max_abs": None,
                    "old_sha256": (
                        sha256_array(old[key]) if key in old else None
                    ),
                    "new_sha256": (
                        sha256_array(new[key]) if key in new else None
                    ),
                }
            )
            all_shapes_match = False
            all_dtypes_match = False
            continue
        left = old[key]
        right = new[key]
        shape_match = left.shape == right.shape
        dtype_match = left.dtype == right.dtype
        exact = bool(
            shape_match
            and dtype_match
            and np.array_equal(left, right)
        )
        max_abs: Optional[float] = None
        if (
            shape_match
            and left.dtype.kind in "fiu"
            and right.dtype.kind in "fiu"
        ):
            max_abs = float(
                np.max(
                    np.abs(
                        left.astype(np.float64)
                        - right.astype(np.float64)
                    )
                )
            )
        rows.append(
            {
                "key": key,
                "present_old": True,
                "present_new": True,
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "exact_equal": exact,
                "max_abs": max_abs,
                "old_sha256": sha256_array(left),
                "new_sha256": sha256_array(right),
            }
        )
        all_shapes_match = all_shapes_match and shape_match
        all_dtypes_match = all_dtypes_match and dtype_match

    return {
        "all_shapes_match": all_shapes_match,
        "all_dtypes_match": all_dtypes_match,
        "all_arrays_exact": all(row["exact_equal"] for row in rows),
        "rows": rows,
    }
