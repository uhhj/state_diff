#!/usr/bin/env python3
"""Run the Stage-F base gate and calibration through one Python entrypoint.

The entrypoint preserves the historical evidence chain but does not delegate to
legacy Resume wrappers.  If the portable Stage-C numerical-equivalence gate
passes, it continues to one real Stage-F calibration in the same process.  If the portable base gate reports a numerical mismatch, it captures the existing
base-gate diagnosis. If real calibration reaches the constraint-z float32
precision admission and stops, it captures the live failing frame, coordinate
round-trip error, segment-length collapse, ULP-to-length ratios, translation
pairs, and runtime settings. It never infers a tolerance from the failure and
never persists prediction tensors.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


sys.dont_write_bytecode = True
SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PHASE = "Phase3.14b-r2.5.8 Stage F Structural-Zero Feature Correction"
SCHEMA = "phase314b_r258_stagef_consolidated_e2e_v5_structural_zero"
DEFAULT_ROOT = Path("/data/state_diff2")

EXPECTED_BRANCH = "Experiment1"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_PARENT = "14af8bb3aabee09ded040b126c869f97e989362a"
EXPECTED_SUBJECT = "Phase3.14b-r2.5.8 Stage F: separate structural-zero constraint state"
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("M", "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py"),
    ("M", "scripts/phase3_14b_r258_stagef_consolidated_e2e.py"),
    ("M", "tests/test_phase3_14b_r258_stagef_constraint_aware_surrogate.py"),
    ("M", "tests/test_phase3_14b_r258_stagef_resume1_translation_fix.py"),
    ("A", "tests/test_phase3_14b_r258_stagef_structural_zero_features.py"),
)

STAGEF_ANCHOR = "f41a2364b7283b1eb963d305823804374ddfc8d6"
STAGEE_ANCHOR = "6758ea7ad800667a436b0243d3b1f6c63256d854"
STAGEF_MODULE = "ccda_phase3.phase314b_r258_stagef_constraint_aware_surrogate"
STAGEF_SCIENTIFIC_PATH = (
    "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py"
)
STAGEF_IMMUTABLE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagef_resume1_translation_fix.py",
    "scripts/phase3_14b_r258_stagef_resume1_worker.py",
)
STAGEE_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py",
    "reports/phase3_14b_r258_stagee_contract.json",
    "reports/phase3_14b_r258_stagee_summary.json",
    "reports/phase3_14b_r258_stagee_worker_evidence.json",
)
STAGEE_WORKER_EVIDENCE = "reports/phase3_14b_r258_stagee_worker_evidence.json"
CANONICAL_WORKER = "scripts/phase3_14b_r258_stagef_resume1_worker.py"

SUCCESS_REPORT = "reports/phase3_14b_r258_stagef_consolidated_v5_structural_zero_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stagef_consolidated_v5_structural_zero_blocked_summary.json"

EXPECTED_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

# Original write-once evidence with externally frozen hashes.
IMMUTABLE_SHA256: Mapping[str, str] = {
    "reports/phase3_14b_r258_stagef_test_gate_summary.json":
        "7d3eadbd063e2248a17c0d26fcd3d47897c46dc434b98a6692c17c214a9a5bd2",
    "reports/phase3_14b_r258_stagef_blocked_summary.json":
        "3a970c1c36998da92312eb78833e2f21401e8556414c87f4502be253a69f5886",
    "reports/phase3_14b_r258_stagef_resume1_blocked_summary.json":
        "b8a11c88aefd9e70f65d2a0d2afbf88b8feabad5f6c97784e41f4d813513c80f",
    "reports/phase3_14b_r258_stagef_resume2_blocked_summary.json":
        "a143a48cba9ccec0987989d5b589d8bd18960d15ee09aa107ef7525ee41cdb70",
}

# Later blocked reports are bound to the commits that recorded them.
COMMIT_BOUND_REPORTS: Mapping[str, str] = {
    "reports/phase3_14b_r258_stagef_resume3_blocked_summary.json":
        "2cab275171bdcc671f40f86092f19a357226dd18",
    "reports/phase3_14b_r258_stagef_resume4_blocked_summary.json":
        "23c49349c842a96aefadb91a2c9eac4ea78c131f",
    "reports/phase3_14b_r258_stagef_resume5_blocked_summary.json":
        "e2fdbeca10df7fbf1925a3c6ec02b8a8a864c851",
    "reports/phase3_14b_r258_stagef_consolidated_blocked_summary.json":
        "bb21870c9b12c699a16b2e6a8501a8c163e9550b",
    "reports/phase3_14b_r258_stagef_consolidated_v2_blocked_summary.json":
        "0e405929ed547a57c1d0ab69131efd11cdb658ff",
    "reports/phase3_14b_r258_stagef_consolidated_v3_blocked_summary.json":
        "84bccbc897f5e9736f487b7110d136c24d7b4be6",
    "reports/phase3_14b_r258_stagef_consolidated_v4_precision_summary.json":
        "14af8bb3aabee09ded040b126c869f97e989362a",
}

REQUIRED_ANCESTORS: Tuple[str, ...] = (
    "519531f411c43b17a668c3c6c1a43b46a94f40a7",
    "f41a2364b7283b1eb963d305823804374ddfc8d6",
    "578708d9cd10d1abd1b052ee338dae7ee9cae011",
    "abe6b8020bfea9d72c1e54bf2e866ee5c285d632",
    "2cab275171bdcc671f40f86092f19a357226dd18",
    "ce88ac14159c7073ab65a4004e49c748d51fab8d",
    "23c49349c842a96aefadb91a2c9eac4ea78c131f",
    "3cd48e4d9aabd97a397e482ca8d7fe9eb89a3f90",
    "e2fdbeca10df7fbf1925a3c6ec02b8a8a864c851",
    "d4c8f5a7f359e1aad7b1298874a2de8ba9d66c02",
    "bb21870c9b12c699a16b2e6a8501a8c163e9550b",
    "4f1a8a0e9683a691e5bf23f16bdefee074c86ea4",
    "0e405929ed547a57c1d0ab69131efd11cdb658ff",
    "053679777b8362d19fbf8e5e3637dc56dcc5d432",
    "84bccbc897f5e9736f487b7110d136c24d7b4be6",
    "02ee4afa59acadc3344ac318c372d527b905581d",
    "14af8bb3aabee09ded040b126c869f97e989362a",
)

FORBIDDEN_TRUE_FIELDS: Tuple[str, ...] = (
    "frozen_probe_accessed",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "candidate_execution",
    "deformable_ravens_executed",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_persisted",
    "surrogate_weights_persisted",
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class ExecutionError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_value(value: Any) -> Any:
    """Convert small controller values; scientific results use module JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExecutionError(f"non-finite evidence float: {value!r}")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExecutionError(f"non-string evidence key: {key!r}")
            result[key] = json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return json_value(value.item())
    raise ExecutionError(f"unsupported evidence value: {type(value)!r}")


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def git_bytes(repo: Path, *args: str, check: bool = True) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and process.returncode != 0:
        raise ExecutionError(
            f"git {' '.join(args)} failed ({process.returncode}): "
            f"{process.stderr.decode('utf-8', 'replace')}"
        )
    return process.stdout


def git(repo: Path, *args: str, check: bool = True) -> str:
    return git_bytes(repo, *args, check=check).decode(
        "utf-8", "strict"
    ).rstrip("\n")


def require_equal(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ExecutionError(
            f"{label} changed: expected={expected!r}, actual={actual!r}"
        )


def require_clean(repo: Path, label: str) -> None:
    status = git_bytes(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status:
        raise ExecutionError(
            f"{label} is not clean: {status.decode('utf-8', 'replace')!r}"
        )


def require_ancestor(repo: Path, ancestor: str, descendant: str) -> None:
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise ExecutionError(f"required ancestor missing: {ancestor}")


def require_blob_equal(repo: Path, commit: str, relpath: str) -> str:
    path = repo / relpath
    if not path.is_file():
        raise ExecutionError(f"immutable path missing: {relpath}")
    committed = git_bytes(repo, "show", f"{commit}:{relpath}")
    current = path.read_bytes()
    if committed != current:
        raise ExecutionError(
            f"immutable path differs from {commit}: {relpath}"
        )
    return sha256_bytes(current)


def validate_environment() -> None:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise ExecutionError(
            "deterministic environment changed: "
            + json.dumps(mismatch, sort_keys=True)
        )


def validate_repository(repo: Path) -> Mapping[str, Any]:
    require_equal(
        "repository root",
        git(repo, "rev-parse", "--show-toplevel"),
        str(repo.resolve()),
    )
    branch = git(repo, "branch", "--show-current")
    require_equal("branch", branch, EXPECTED_BRANCH)

    head = git(repo, "rev-parse", "HEAD")
    parents = git(repo, "show", "-s", "--format=%P", head).split()
    if parents != [EXPECTED_PARENT]:
        raise ExecutionError(
            f"implementation parent changed: expected {[EXPECTED_PARENT]!r}, "
            f"actual={parents!r}"
        )
    subject = git(repo, "show", "-s", "--format=%s", head)
    require_equal("implementation subject", subject, EXPECTED_SUBJECT)

    raw_changed = git_bytes(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        "--no-renames",
        head,
    ).split(b"\0")
    if raw_changed and raw_changed[-1] == b"":
        raw_changed.pop()
    if len(raw_changed) % 2 != 0:
        raise ExecutionError(
            "implementation diff-tree has an incomplete status/path record"
        )
    changed = tuple(
        (
            raw_changed[index].decode("ascii", "strict"),
            raw_changed[index + 1].decode("utf-8", "strict"),
        )
        for index in range(0, len(raw_changed), 2)
    )
    if changed != IMPLEMENTATION_PATHS:
        raise ExecutionError(
            "structural-zero implementation path population changed: "
            f"expected={IMPLEMENTATION_PATHS!r}, actual={changed!r}"
        )

    for ancestor in REQUIRED_ANCESTORS:
        require_ancestor(repo, ancestor, head)

    remote = git(repo, "rev-parse", "refs/remotes/origin/Experiment1")
    require_equal("origin/Experiment1", remote, EXPECTED_REMOTE_HEAD)

    gitlink = git(repo, "rev-parse", "HEAD:external/deformable-ravens")
    require_equal("submodule gitlink", gitlink, EXPECTED_SUBMODULE)
    submodule = repo / "external/deformable-ravens"
    require_equal(
        "submodule worktree",
        git(submodule, "rev-parse", "HEAD"),
        EXPECTED_SUBMODULE,
    )
    require_clean(submodule, "DeformableRavens")
    require_clean(repo, "main worktree")

    immutable_hashes: Dict[str, str] = {}
    for relpath, expected in IMMUTABLE_SHA256.items():
        actual = sha256_path(repo / relpath)
        if actual != expected:
            raise ExecutionError(
                f"immutable report SHA changed: {relpath}: {actual} != {expected}"
            )
        immutable_hashes[relpath] = actual

    commit_bound = {
        relpath: require_blob_equal(repo, commit, relpath)
        for relpath, commit in COMMIT_BOUND_REPORTS.items()
    }
    stagef_hashes = {
        relpath: require_blob_equal(repo, STAGEF_ANCHOR, relpath)
        for relpath in STAGEF_IMMUTABLE_PATHS
    }
    stagef_hashes[STAGEF_SCIENTIFIC_PATH] = sha256_path(
        repo / STAGEF_SCIENTIFIC_PATH
    )
    stagee_hashes = {
        relpath: require_blob_equal(repo, STAGEE_ANCHOR, relpath)
        for relpath in STAGEE_PATHS
    }

    if (repo / SUCCESS_REPORT).exists() or (repo / BLOCKED_REPORT).exists():
        raise ExecutionError(
            "consolidated output already exists; this execution is write-once"
        )

    return {
        "branch": branch,
        "head": head,
        "parent": EXPECTED_PARENT,
        "subject": subject,
        "remote_head": remote,
        "submodule": EXPECTED_SUBMODULE,
        "implementation_paths": [list(item) for item in IMPLEMENTATION_PATHS],
        "immutable_report_sha256": immutable_hashes,
        "commit_bound_report_sha256": commit_bound,
        "stagef_source_sha256": stagef_hashes,
        "stagee_input_sha256": stagee_hashes,
    }


def load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExecutionError(f"JSON root is not an object: {path}")
    return payload


def load_frozen_environment(repo: Path) -> Mapping[str, Any]:
    evidence = load_json(repo / STAGEE_WORKER_EVIDENCE)
    worker_result = evidence.get("worker_result")
    if not isinstance(worker_result, dict):
        raise ExecutionError("Stage-E worker evidence lacks worker_result")
    environment = worker_result.get("environment")
    if not isinstance(environment, dict):
        raise ExecutionError("Stage-E worker evidence lacks environment")
    if environment.get("compatibility_pass") is not True:
        raise ExecutionError("frozen Stage-E compatibility is not PASS")
    dry_run = environment.get("required_operation_dry_run")
    if not isinstance(dry_run, dict) or dry_run.get("pass") is not True:
        raise ExecutionError("frozen Stage-E required-operation gate is not PASS")
    return copy.deepcopy(environment)


def required_callable(module: Any, name: str) -> Callable[..., Any]:
    value = getattr(module, name, None)
    if not callable(value):
        raise ExecutionError(f"missing callable: {STAGEF_MODULE}.{name}")
    return value


def call_validate_base(function: Callable[..., Any], repo: Path) -> Any:
    signature = inspect.signature(function)
    try:
        signature.bind(root=repo)
    except TypeError:
        try:
            signature.bind(repo)
        except TypeError as exc:
            raise ExecutionError(
                f"validate_base_evidence signature changed: {signature}"
            ) from exc
        return function(repo)
    return function(root=repo)


def call_calibration(
    function: Callable[..., Any],
    repo: Path,
    environment: Mapping[str, Any],
) -> Any:
    signature = inspect.signature(function)
    try:
        signature.bind(root=repo, environment=environment)
    except TypeError as exc:
        raise ExecutionError(
            f"run_calibration signature changed: {signature}"
        ) from exc
    return function(root=repo, environment=environment)


def is_python_command(command: Any, executable: Any = None) -> bool:
    value = executable
    if value is None:
        if isinstance(command, (list, tuple)) and command:
            value = command[0]
        elif isinstance(command, (str, bytes)):
            text = (
                command.decode("utf-8", "replace")
                if isinstance(command, bytes)
                else command
            )
            value = text.strip().split()[0] if text.strip() else None
    if value is None:
        return False
    text = os.fsdecode(value)
    base = Path(text).name.lower()
    if base.startswith("python") or base in {"py", "pypy", "pypy3"}:
        return True
    try:
        return Path(text).resolve() == Path(sys.executable).resolve()
    except OSError:
        return False


class OneProcessGuard:
    """Reject child Python, multiprocessing, fork, and os.system."""

    def __init__(self) -> None:
        self.commands: List[Mapping[str, Any]] = []
        self._popen = subprocess.Popen
        self._process_start = multiprocessing.Process.start
        self._fork = getattr(os, "fork", None)
        self._system = os.system

    def __enter__(self) -> "OneProcessGuard":
        guard = self
        real_popen = self._popen

        class GuardedPopen(real_popen):  # type: ignore[misc, valid-type]
            def __init__(self, args: Any, *pos: Any, **kwargs: Any) -> None:
                executable = kwargs.get("executable")
                guard.commands.append(
                    {
                        "command": repr(args),
                        "executable": None
                        if executable is None
                        else os.fsdecode(executable),
                        "shell": bool(kwargs.get("shell", False)),
                    }
                )
                if kwargs.get("shell", False):
                    raise ExecutionError(
                        f"shell subprocess is forbidden during science: {args!r}"
                    )
                candidates = (
                    list(args)
                    if isinstance(args, (list, tuple))
                    else [args]
                )
                if is_python_command(args, executable) or any(
                    is_python_command([candidate]) for candidate in candidates
                ):
                    raise ExecutionError(
                        f"child Python process is forbidden: {args!r}"
                    )
                super().__init__(args, *pos, **kwargs)

        def blocked_process_start(_process: Any) -> None:
            raise ExecutionError("multiprocessing is forbidden")

        def blocked_fork() -> int:
            raise ExecutionError("fork is forbidden")

        def blocked_system(_command: Any) -> int:
            raise ExecutionError("os.system is forbidden")

        subprocess.Popen = GuardedPopen  # type: ignore[assignment]
        multiprocessing.Process.start = blocked_process_start  # type: ignore[assignment]
        if self._fork is not None:
            os.fork = blocked_fork  # type: ignore[attr-defined, assignment]
        os.system = blocked_system  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc: Any) -> None:
        subprocess.Popen = self._popen  # type: ignore[assignment]
        multiprocessing.Process.start = self._process_start  # type: ignore[assignment]
        if self._fork is not None:
            os.fork = self._fork  # type: ignore[attr-defined, assignment]
        os.system = self._system  # type: ignore[assignment]


def normalize_scientific_result(module: Any, result: Any) -> Mapping[str, Any]:
    serializer = getattr(module, "stable_json_bytes", None)
    if not callable(serializer):
        raise ExecutionError("Stage-F module lacks stable_json_bytes")
    raw = serializer(result)
    if not isinstance(raw, (bytes, bytearray)):
        raise ExecutionError("Stage-F stable_json_bytes did not return bytes")
    normalized = json.loads(bytes(raw).decode("utf-8"))
    if not isinstance(normalized, dict):
        raise ExecutionError("Stage-F result is not a JSON object")
    return normalized


def validate_scientific_result(result: Mapping[str, Any]) -> None:
    if result.get("verdict") != "PASS":
        raise ExecutionError(
            f"Stage-F audit did not complete: verdict={result.get('verdict')!r}"
        )
    status = result.get("scientific_status")
    if status not in {"READY", "BLOCKED"}:
        raise ExecutionError(f"invalid scientific_status: {status!r}")
    for field in ("root_cause", "required_next_path"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ExecutionError(f"Stage-F result lacks {field}")
    for field in FORBIDDEN_TRUE_FIELDS:
        if result.get(field) is True:
            raise ExecutionError(f"forbidden boundary crossed: {field}=true")

    cold = result.get("cold_main_worker_context")
    if isinstance(cold, dict):
        if cold.get("torch_cuda_is_initialized") is not False:
            raise ExecutionError("Stage-F result does not preserve cold CUDA entry")

    if status == "READY":
        if result.get("selected_configuration") is None:
            raise ExecutionError("READY result lacks selected_configuration")
        if result.get("train_only_recommendation") is None:
            raise ExecutionError("READY result lacks train_only_recommendation")
        if result.get("selection_holdout_evaluated") is not True:
            raise ExecutionError("READY result lacks locked holdout evaluation")


def torch_observation(torch: Any) -> Mapping[str, Any]:
    initialized = bool(torch.cuda.is_initialized())
    result: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_initialized": initialized,
    }
    if initialized:
        device = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "device_index": device,
                "device_name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        )
    return result



NUMERICAL_EQUIVALENCE_MARKER = (
    "portable Stage-C numerical-equivalence gate failed"
)
DEFAULT_NUMERICAL_DIFF_PATHS: Tuple[str, ...] = (
    "$.holdout_profiles.10.huber_region.positive_element_count",
    "$.one_step.timesteps.10.upper.sha256",
    "$.one_step.timesteps.25.upper.sha256",
    "$.one_step.timesteps.50.upper.sha256",
)
EXPECTED_NAME_HINTS: Tuple[str, ...] = (
    "expected",
    "reference",
    "frozen",
    "recorded",
    "canonical",
    "baseline",
)
ACTUAL_NAME_HINTS: Tuple[str, ...] = (
    "actual",
    "current",
    "observed",
    "replay",
    "portable",
    "generated",
    "candidate",
)
MAX_DIAGNOSTIC_ITEMS = 192
MAX_DIAGNOSTIC_DEPTH = 10
MAX_FRAME_LOCALS = 48


def _is_array_like(value: Any) -> bool:
    module = type(value).__module__.split(".", 1)[0]
    name = type(value).__name__
    return (
        module in {"numpy", "torch"}
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and name not in {"dtype"}
    )


def _array_numpy(value: Any) -> Any:
    import numpy as np  # type: ignore

    if type(value).__module__.split(".", 1)[0] == "torch":
        value = value.detach()
        if hasattr(value, "is_sparse") and bool(value.is_sparse):
            value = value.to_dense()
        value = value.cpu().contiguous().numpy()
    return np.asarray(value)


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


def array_summary(value: Any) -> Mapping[str, Any]:
    import numpy as np  # type: ignore

    array = _array_numpy(value)
    contiguous = np.ascontiguousarray(array)
    summary: Dict[str, Any] = {
        "python_type": (
            f"{type(value).__module__}.{type(value).__qualname__}"
        ),
        "shape": [int(item) for item in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "ndim": int(contiguous.ndim),
        "element_count": int(contiguous.size),
        "nbytes": int(contiguous.nbytes),
        "c_contiguous": bool(contiguous.flags.c_contiguous),
        "raw_c_bytes_sha256": sha256_bytes(contiguous.tobytes(order="C")),
    }
    if contiguous.size == 0:
        return summary

    if np.issubdtype(contiguous.dtype, np.number):
        numeric = contiguous.astype(np.float64, copy=False)
        finite = np.isfinite(numeric)
        finite_values = numeric[finite]
        summary.update(
            {
                "finite_count": int(finite.sum()),
                "nan_count": int(np.isnan(numeric).sum()),
                "positive_infinity_count": int(np.isposinf(numeric).sum()),
                "negative_infinity_count": int(np.isneginf(numeric).sum()),
            }
        )
        if finite_values.size:
            summary.update(
                {
                    "minimum": float(finite_values.min()),
                    "maximum": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "absolute_maximum": float(np.abs(finite_values).max()),
                    "positive_element_count": int((finite_values > 0.0).sum()),
                    "negative_element_count": int((finite_values < 0.0).sum()),
                    "zero_element_count": int((finite_values == 0.0).sum()),
                    "near_zero_counts": {
                        "abs_le_1e-15": int((np.abs(finite_values) <= 1e-15).sum()),
                        "abs_le_1e-12": int((np.abs(finite_values) <= 1e-12).sum()),
                        "abs_le_1e-10": int((np.abs(finite_values) <= 1e-10).sum()),
                        "abs_le_1e-8": int((np.abs(finite_values) <= 1e-8).sum()),
                        "abs_le_1e-6": int((np.abs(finite_values) <= 1e-6).sum()),
                    },
                }
            )
    return summary


def diagnostic_snapshot(
    value: Any,
    *,
    depth: int = 0,
    seen: Optional[set] = None,
) -> Any:
    """Create a bounded JSON snapshot without persisting tensor contents."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"non_finite_float": "nan"}
        if math.isinf(value):
            return {
                "non_finite_float": "positive_infinity"
                if value > 0
                else "negative_infinity"
            }
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if _is_array_like(value):
        try:
            return {"array_summary": array_summary(value)}
        except BaseException as error:
            return {
                "array_summary_error": (
                    f"{type(error).__module__}.{type(error).__qualname__}: "
                    f"{error}"
                ),
                "python_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
            }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            value = dataclasses.asdict(value)
        except BaseException:
            return {
                "python_type": (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ),
                "representation": repr(value)[:512],
            }
    if depth >= MAX_DIAGNOSTIC_DEPTH:
        return {
            "truncated": "maximum_depth",
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        }

    identity = id(value)
    if identity in seen:
        return {"cycle": True}
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            result: Dict[str, Any] = {}
            entries = list(value.items())
            for index, (key, item) in enumerate(entries):
                if index >= MAX_DIAGNOSTIC_ITEMS:
                    result["__truncated_items__"] = len(entries) - index
                    break
                result[str(key)] = diagnostic_snapshot(
                    item,
                    depth=depth + 1,
                    seen=seen,
                )
            return result
        if isinstance(value, (list, tuple)):
            result_list: List[Any] = []
            for index, item in enumerate(value):
                if index >= MAX_DIAGNOSTIC_ITEMS:
                    result_list.append(
                        {"truncated_items": len(value) - index}
                    )
                    break
                result_list.append(
                    diagnostic_snapshot(item, depth=depth + 1, seen=seen)
                )
            return result_list
        if isinstance(value, (set, frozenset)):
            snapshots = [
                diagnostic_snapshot(item, depth=depth + 1, seen=seen)
                for item in list(value)[:MAX_DIAGNOSTIC_ITEMS]
            ]
            return sorted(
                snapshots,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
        if hasattr(value, "item") and callable(value.item):
            try:
                return diagnostic_snapshot(
                    value.item(), depth=depth + 1, seen=seen
                )
            except BaseException:
                pass
        return {
            "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "representation": repr(value)[:512],
        }
    finally:
        seen.discard(identity)


def parse_numerical_diff_paths(message: str) -> Tuple[str, ...]:
    found = tuple(
        dict.fromkeys(
            re.findall(r"(?m)^\s*-\s+(\$\.[^\s]+)\s*$", message)
        )
    )
    return found or DEFAULT_NUMERICAL_DIFF_PATHS


def path_tokens(path: str) -> Tuple[str, ...]:
    if path == "$":
        return ()
    if not path.startswith("$."):
        raise ValueError(f"unsupported diagnostic path: {path!r}")
    return tuple(token for token in path[2:].split(".") if token)


def resolve_path(value: Any, path: str) -> Tuple[bool, Any]:
    current = value
    for token in path_tokens(path):
        if isinstance(current, Mapping):
            if token in current:
                current = current[token]
                continue
            try:
                integer_key = int(token)
            except ValueError:
                return False, None
            if integer_key in current:
                current = current[integer_key]
                continue
            return False, None
        if isinstance(current, (list, tuple)):
            try:
                index = int(token)
            except ValueError:
                return False, None
            if not 0 <= index < len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def scalar_leaf_map(value: Any, prefix: str = "$") -> Mapping[str, Any]:
    leaves: Dict[str, Any] = {}

    def visit(item: Any, path: str, depth: int) -> None:
        if depth > MAX_DIAGNOSTIC_DEPTH:
            return
        if item is None or isinstance(item, (str, bool, int)):
            leaves[path] = item
            return
        if isinstance(item, float):
            leaves[path] = diagnostic_snapshot(item)
            return
        if isinstance(item, Path):
            leaves[path] = item.as_posix()
            return
        if _is_array_like(item):
            summary = array_summary(item)
            visit(summary, path + ".array_summary", depth + 1)
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            try:
                item = dataclasses.asdict(item)
            except BaseException:
                leaves[path] = repr(item)[:512]
                return
        if isinstance(item, Mapping):
            for index, (key, child) in enumerate(item.items()):
                if index >= MAX_DIAGNOSTIC_ITEMS:
                    break
                visit(child, path + "." + str(key), depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item[:MAX_DIAGNOSTIC_ITEMS]):
                visit(child, path + "." + str(index), depth + 1)
            return
        if hasattr(item, "item") and callable(item.item):
            try:
                visit(item.item(), path, depth + 1)
                return
            except BaseException:
                pass
        leaves[path] = repr(item)[:512]

    visit(value, prefix, 0)
    return leaves


def leaf_differences(left: Any, right: Any) -> Mapping[str, Mapping[str, Any]]:
    left_leaves = scalar_leaf_map(left)
    right_leaves = scalar_leaf_map(right)
    result: Dict[str, Mapping[str, Any]] = {}
    for path in sorted(set(left_leaves) | set(right_leaves)):
        left_present = path in left_leaves
        right_present = path in right_leaves
        left_value = left_leaves.get(path)
        right_value = right_leaves.get(path)
        if left_present and right_present and left_value == right_value:
            continue
        result[path] = {
            "left_present": left_present,
            "right_present": right_present,
            "left": diagnostic_snapshot(left_value),
            "right": diagnostic_snapshot(right_value),
        }
    return result


def name_hint_score(label: str, hints: Sequence[str]) -> int:
    lowered = label.lower()
    return sum(1 for hint in hints if hint in lowered)


def collect_mapping_candidates(
    error: BaseException,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    candidates: List[Mapping[str, Any]] = []
    frame_summaries: List[Mapping[str, Any]] = []
    traceback_cursor = error.__traceback__
    frame_index = 0
    while traceback_cursor is not None:
        frame = traceback_cursor.tb_frame
        module_name = str(frame.f_globals.get("__name__", ""))
        code_name = frame.f_code.co_name
        local_items = list(frame.f_locals.items())
        selected_locals: Dict[str, Any] = {}
        for local_index, (name, value) in enumerate(local_items):
            if local_index >= MAX_FRAME_LOCALS:
                selected_locals["__truncated_locals__"] = (
                    len(local_items) - local_index
                )
                break
            lowered = name.lower()
            should_capture = (
                isinstance(value, Mapping)
                or _is_array_like(value)
                or any(
                    token in lowered
                    for token in (
                        "expected",
                        "actual",
                        "current",
                        "reference",
                        "frozen",
                        "diff",
                        "mismatch",
                        "profile",
                        "one_step",
                        "upper",
                    )
                )
            )
            if should_capture:
                selected_locals[name] = diagnostic_snapshot(value)

            def collect_nested(
                item: Any,
                label: str,
                depth: int,
                seen: set,
            ) -> None:
                if depth > 3:
                    return
                identity = id(item)
                if identity in seen:
                    return
                if dataclasses.is_dataclass(item) and not isinstance(item, type):
                    try:
                        item = dataclasses.asdict(item)
                    except BaseException:
                        return
                if isinstance(item, Mapping):
                    candidates.append(
                        {
                            "label": label,
                            "frame_index": frame_index,
                            "module": module_name,
                            "function": code_name,
                            "raw": item,
                        }
                    )
                    seen.add(identity)
                    try:
                        for nested_index, (key, child) in enumerate(item.items()):
                            if nested_index >= 32:
                                break
                            if isinstance(child, Mapping) or dataclasses.is_dataclass(child):
                                collect_nested(
                                    child,
                                    label + "." + str(key),
                                    depth + 1,
                                    seen,
                                )
                    finally:
                        seen.discard(identity)

            collect_nested(
                value,
                f"frame[{frame_index}].{module_name}.{code_name}.{name}",
                0,
                set(),
            )

        frame_summaries.append(
            {
                "frame_index": frame_index,
                "module": module_name,
                "function": code_name,
                "filename": frame.f_code.co_filename,
                "line_number": int(traceback_cursor.tb_lineno),
                "selected_locals": selected_locals,
            }
        )
        traceback_cursor = traceback_cursor.tb_next
        frame_index += 1
    return candidates, frame_summaries


def orient_candidate_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any], int]:
    left_label = str(left["label"])
    right_label = str(right["label"])
    left_expected = name_hint_score(left_label, EXPECTED_NAME_HINTS)
    left_actual = name_hint_score(left_label, ACTUAL_NAME_HINTS)
    right_expected = name_hint_score(right_label, EXPECTED_NAME_HINTS)
    right_actual = name_hint_score(right_label, ACTUAL_NAME_HINTS)
    normal = left_expected + right_actual
    reversed_score = right_expected + left_actual
    if reversed_score > normal:
        return right, left, reversed_score
    return left, right, normal


def select_expected_actual_pair(
    candidates: Sequence[Mapping[str, Any]],
    mismatch_paths: Sequence[str],
) -> Optional[Mapping[str, Any]]:
    best: Optional[Mapping[str, Any]] = None
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            expected, actual, orientation = orient_candidate_pair(left, right)
            expected_raw = expected["raw"]
            actual_raw = actual["raw"]
            target_rows: Dict[str, Mapping[str, Any]] = {}
            coverage = 0
            for path in mismatch_paths:
                expected_present, expected_value = resolve_path(expected_raw, path)
                actual_present, actual_value = resolve_path(actual_raw, path)
                differs = (
                    expected_present
                    and actual_present
                    and diagnostic_snapshot(expected_value)
                    != diagnostic_snapshot(actual_value)
                )
                if differs:
                    coverage += 1
                target_rows[path] = {
                    "expected_present": expected_present,
                    "actual_present": actual_present,
                    "expected": diagnostic_snapshot(expected_value)
                    if expected_present
                    else None,
                    "actual": diagnostic_snapshot(actual_value)
                    if actual_present
                    else None,
                    "different": differs,
                }
            if coverage == 0:
                continue
            differences = leaf_differences(expected_raw, actual_raw)
            extra_count = len(
                [path for path in differences if path not in mismatch_paths]
            )
            score = coverage * 100000 + orientation * 1000 - extra_count
            candidate_result: Mapping[str, Any] = {
                "score": score,
                "coverage": coverage,
                "orientation_hint_score": orientation,
                "expected_label": expected["label"],
                "actual_label": actual["label"],
                "expected_raw": expected_raw,
                "actual_raw": actual_raw,
                "target_rows": target_rows,
                "all_differences": differences,
                "extra_difference_count": extra_count,
            }
            if best is None or int(candidate_result["score"]) > int(best["score"]):
                best = candidate_result
    return best


def parent_path(path: str) -> str:
    if "." not in path[2:]:
        return "$"
    return path.rsplit(".", 1)[0]


def find_arrays(value: Any, prefix: str = "$") -> Mapping[str, Any]:
    result: Dict[str, Any] = {}
    seen: set = set()

    def visit(item: Any, path: str, depth: int) -> None:
        if depth > 5 or len(result) >= 64:
            return
        if _is_array_like(item):
            result[path] = item
            return
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            try:
                item = dataclasses.asdict(item)
            except BaseException:
                return
        identity = id(item)
        if identity in seen:
            return
        if isinstance(item, Mapping):
            seen.add(identity)
            try:
                for index, (key, child) in enumerate(item.items()):
                    if index >= 64:
                        break
                    visit(child, path + "." + str(key), depth + 1)
            finally:
                seen.discard(identity)
        elif isinstance(item, (list, tuple)):
            seen.add(identity)
            try:
                for index, child in enumerate(item[:64]):
                    visit(child, path + "." + str(index), depth + 1)
            finally:
                seen.discard(identity)

    visit(value, prefix, 0)
    return result


def compare_array_values(expected: Any, actual: Any) -> Mapping[str, Any]:
    import numpy as np  # type: ignore

    expected_array = _array_numpy(expected)
    actual_array = _array_numpy(actual)
    result: Dict[str, Any] = {
        "expected": array_summary(expected),
        "actual": array_summary(actual),
        "shape_equal": expected_array.shape == actual_array.shape,
        "dtype_equal": expected_array.dtype == actual_array.dtype,
    }
    if expected_array.shape != actual_array.shape:
        return result
    expected_numeric = expected_array.astype(np.float64, copy=False)
    actual_numeric = actual_array.astype(np.float64, copy=False)
    finite_pair = np.isfinite(expected_numeric) & np.isfinite(actual_numeric)
    result["finite_pair_count"] = int(finite_pair.sum())
    if not finite_pair.any():
        return result
    expected_finite = expected_numeric[finite_pair]
    actual_finite = actual_numeric[finite_pair]
    difference = actual_finite - expected_finite
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(expected_finite), np.finfo(np.float64).tiny)
    result.update(
        {
            "exactly_different_count": int(
                np.not_equal(actual_finite, expected_finite).sum()
            ),
            "sign_change_count": int(
                np.not_equal(
                    np.signbit(actual_finite),
                    np.signbit(expected_finite),
                ).sum()
            ),
            "max_abs_diff": float(absolute.max()),
            "mean_abs_diff": float(absolute.mean()),
            "max_rel_diff": float((absolute / denominator).max()),
            "allclose": {
                "rtol_0_atol_0": bool(
                    np.allclose(actual_finite, expected_finite, rtol=0.0, atol=0.0)
                ),
                "rtol_1e-9_atol_1e-12": bool(
                    np.allclose(actual_finite, expected_finite, rtol=1e-9, atol=1e-12)
                ),
                "rtol_1e-8_atol_1e-10": bool(
                    np.allclose(actual_finite, expected_finite, rtol=1e-8, atol=1e-10)
                ),
                "rtol_1e-6_atol_1e-8": bool(
                    np.allclose(actual_finite, expected_finite, rtol=1e-6, atol=1e-8)
                ),
            },
        }
    )
    return result


def pair_array_diagnostics(
    expected_value: Any,
    actual_value: Any,
) -> Mapping[str, Any]:
    expected_arrays = find_arrays(expected_value)
    actual_arrays = find_arrays(actual_value)
    result: Dict[str, Any] = {
        "expected_array_paths": sorted(expected_arrays),
        "actual_array_paths": sorted(actual_arrays),
        "comparisons": {},
    }
    comparisons: Dict[str, Any] = {}
    for path in sorted(set(expected_arrays) & set(actual_arrays)):
        try:
            comparisons[path] = compare_array_values(
                expected_arrays[path], actual_arrays[path]
            )
        except BaseException as error:
            comparisons[path] = {
                "comparison_error": (
                    f"{type(error).__module__}.{type(error).__qualname__}: "
                    f"{error}"
                )
            }
    result["comparisons"] = comparisons
    result["reference_numeric_arrays_available"] = bool(comparisons)
    return result


def runtime_numerical_observation(torch: Any) -> Mapping[str, Any]:
    observation: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    if callable(get_precision):
        observation["float32_matmul_precision"] = str(get_precision())
    try:
        import numpy as np  # type: ignore

        observation["numpy_version"] = str(np.__version__)
    except BaseException as error:
        observation["numpy_import_error"] = str(error)
    if torch.cuda.is_initialized():
        device = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        observation["cuda_device"] = {
            "index": device,
            "name": str(properties.name),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory_bytes": int(properties.total_memory),
        }
    return observation


def numerical_equivalence_diagnostic(
    error: BaseException,
    torch: Any,
) -> Mapping[str, Any]:
    message = str(error)
    mismatch_paths = parse_numerical_diff_paths(message)
    candidates, frames = collect_mapping_candidates(error)
    pair = select_expected_actual_pair(candidates, mismatch_paths)

    diagnostic: Dict[str, Any] = {
        "gate_marker": NUMERICAL_EQUIVALENCE_MARKER,
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error": message,
        "mismatch_paths": list(mismatch_paths),
        "traceback_frames": frames,
        "mapping_candidate_count": len(candidates),
        "runtime": runtime_numerical_observation(torch),
        "policy": {
            "exact_gate_relaxed": False,
            "tolerance_inferred_from_current_failure": False,
            "scientific_calibration_started": False,
            "reason": (
                "A SHA mismatch does not encode numerical magnitude. The exact "
                "gate must not be relaxed unless expected and current numerical "
                "values are both available or a separately justified semantic "
                "contract is established."
            ),
        },
    }

    if pair is None:
        diagnostic.update(
            {
                "capture_status": "PARTIAL",
                "expected_actual_pair_found": False,
                "decision": "KEEP_BLOCKED_DIAGNOSTIC_PAIR_NOT_RECOVERED",
                "required_next_path": (
                    "INSPECT_THE_V3_TRACEBACK_LOCAL_SNAPSHOTS_IN_THE_SAME_"
                    "CONSOLIDATED_ENTRYPOINT"
                ),
            }
        )
        return diagnostic

    expected_raw = pair["expected_raw"]
    actual_raw = pair["actual_raw"]
    differences = pair["all_differences"]
    mismatch_set = set(mismatch_paths)
    additional_paths = sorted(path for path in differences if path not in mismatch_set)
    missing_target_paths = sorted(
        path
        for path, row in pair["target_rows"].items()
        if not bool(row["expected_present"]) or not bool(row["actual_present"])
    )

    parent_diagnostics: Dict[str, Any] = {}
    reference_arrays_available = True
    for path in mismatch_paths:
        parent = parent_path(path)
        expected_present, expected_parent = resolve_path(expected_raw, parent)
        actual_present, actual_parent = resolve_path(actual_raw, parent)
        row: Dict[str, Any] = {
            "parent_path": parent,
            "expected_present": expected_present,
            "actual_present": actual_present,
            "expected_snapshot": diagnostic_snapshot(expected_parent)
            if expected_present
            else None,
            "actual_snapshot": diagnostic_snapshot(actual_parent)
            if actual_present
            else None,
        }
        if expected_present and actual_present:
            array_diagnostic = pair_array_diagnostics(
                expected_parent, actual_parent
            )
            row["array_diagnostic"] = array_diagnostic
            if path.endswith(".sha256") and not bool(
                array_diagnostic["reference_numeric_arrays_available"]
            ):
                reference_arrays_available = False
        elif path.endswith(".sha256"):
            reference_arrays_available = False
        parent_diagnostics[path] = row

    positive_path = (
        "$.holdout_profiles.10.huber_region.positive_element_count"
    )
    positive_delta: Optional[int] = None
    positive_row = pair["target_rows"].get(positive_path)
    if positive_row:
        expected_count = positive_row.get("expected")
        actual_count = positive_row.get("actual")
        if (
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and isinstance(actual_count, int)
            and not isinstance(actual_count, bool)
        ):
            positive_delta = actual_count - expected_count

    exact_target_only = (
        not additional_paths
        and not missing_target_paths
        and int(pair["coverage"]) == len(mismatch_paths)
    )
    if reference_arrays_available:
        decision = "KEEP_BLOCKED_PENDING_REVIEW_OF_CAPTURED_NUMERICAL_DIFFS"
        next_path = (
            "REVIEW_CAPTURED_REFERENCE_AND_CURRENT_ARRAY_DIFFERENCES_BEFORE_"
            "CHANGING_THE_SAME_CONSOLIDATED_GATE"
        )
    else:
        decision = "KEEP_BLOCKED_REFERENCE_NUMERICAL_VALUES_NOT_AVAILABLE"
        next_path = (
            "ESTABLISH_A_REFERENCE_NUMERICAL_WITNESS_OR_AN_INDEPENDENT_"
            "SEMANTIC_EQUIVALENCE_CONTRACT_BEFORE_RELAXING_THE_GATE"
        )

    diagnostic.update(
        {
            "capture_status": "COMPLETE" if not missing_target_paths else "PARTIAL",
            "expected_actual_pair_found": True,
            "expected_candidate_label": pair["expected_label"],
            "actual_candidate_label": pair["actual_label"],
            "orientation_hint_score": pair["orientation_hint_score"],
            "target_path_coverage": int(pair["coverage"]),
            "target_differences": pair["target_rows"],
            "all_leaf_difference_count": len(differences),
            "additional_difference_paths": additional_paths,
            "missing_target_paths": missing_target_paths,
            "exactly_reported_paths_differ": exact_target_only,
            "positive_element_count_delta": positive_delta,
            "parent_subtree_diagnostics": parent_diagnostics,
            "reference_numerical_arrays_available_for_sha_mismatches": (
                reference_arrays_available
            ),
            "decision": decision,
            "required_next_path": next_path,
        }
    )
    return diagnostic


def diagnostic_summary(
    repository: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> Mapping[str, Any]:
    diagnostic_sha = sha256_bytes(stable_json_bytes(diagnostic))
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagef_portable_numerical_equivalence_unresolved"
        ),
        "required_next_path": diagnostic["required_next_path"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": repository,
        "execution": {
            "mode": "one_python_process_one_base_gate_diagnosis",
            "pid": os.getpid(),
            "python_process_count": 1,
            "child_python_process_count": 0,
            "legacy_resume_wrappers_executed": False,
            "duplicate_workers_executed": False,
            "historical_temporal_gate_replayed": False,
            "scientific_calibration_started": False,
            "single_run_result_sha256": diagnostic_sha,
        },
        "base_evidence_validation": {
            "completed_without_exception": False,
            "failure_classified_as_portable_numerical_equivalence": True,
        },
        "numerical_equivalence_diagnosis": diagnostic,
        "boundaries": {
            "frozen_probe_accessed": False,
            "selection_holdout_evaluated": False,
            "formal_training_run": False,
            "reverse_sampling_run": False,
            "idm_run": False,
            "candidate_execution": False,
            "deformable_ravens_executed": False,
            "phase4": False,
            "cps": False,
            "checkpoint_saved": False,
            "weights_persisted": False,
            "surrogate_weights_persisted": False,
            "prediction_tensor_persisted": False,
            "candidate_tensor_persisted": False,
            "npz_saved": False,
            "cache_saved": False,
            "image_saved": False,
            "video_saved": False,
        },
    }



CONSTRAINT_Z_PRECISION_MARKERS: Tuple[str, ...] = (
    "float32 coordinate resolution is insufficient for constraint-z audit",
    "float32 coordinate resolution is insufficient for resolvable constraint-z audit",
)
CONSTRAINT_Z_FUNCTION = "_float32_constraint_z_bound"
MAX_PRECISION_ARRAYS = 48
MAX_PRECISION_PAIRS = 64
MAX_PRECISION_LOCAL_ITEMS = 128


def _traceback_frames(error: BaseException) -> List[Any]:
    frames: List[Any] = []
    current = error.__traceback__
    while current is not None:
        frames.append(current)
        current = current.tb_next
    return frames


def _source_excerpt(frame: Any, radius: int = 4) -> Mapping[str, Any]:
    filename = Path(frame.tb_frame.f_code.co_filename)
    lineno = int(frame.tb_lineno)
    result: Dict[str, Any] = {
        "filename": str(filename),
        "function": frame.tb_frame.f_code.co_name,
        "lineno": lineno,
    }
    try:
        lines = filename.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        result["source_error"] = f"{type(error).__name__}: {error}"
        return result
    start = max(1, lineno - radius)
    end = min(len(lines), lineno + radius)
    excerpt = [
        {"lineno": number, "text": lines[number - 1]}
        for number in range(start, end + 1)
    ]
    result.update(
        {
            "file_sha256": sha256_path(filename),
            "excerpt": excerpt,
        }
    )
    return result


def _bounded_local_snapshot(frame: Any) -> Mapping[str, Any]:
    snapshots: Dict[str, Any] = {}
    omitted: List[str] = []
    for index, name in enumerate(sorted(frame.tb_frame.f_locals)):
        if index >= MAX_PRECISION_LOCAL_ITEMS:
            omitted.extend(sorted(frame.tb_frame.f_locals)[index:])
            break
        value = frame.tb_frame.f_locals[name]
        try:
            snapshots[name] = diagnostic_snapshot(value)
        except BaseException as error:
            snapshots[name] = {
                "snapshot_error": (
                    f"{type(error).__module__}.{type(error).__qualname__}: "
                    f"{error}"
                )
            }
    return {
        "locals": snapshots,
        "omitted_local_names": omitted,
    }


def _collect_array_candidates(
    value: Any,
    path: str,
    output: List[Tuple[str, Any]],
    *,
    depth: int = 0,
    seen: Optional[set] = None,
) -> None:
    if seen is None:
        seen = set()
    if len(output) >= MAX_PRECISION_ARRAYS or depth > 3:
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if _is_array_like(value):
        output.append((path, value))
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            if len(output) >= MAX_PRECISION_ARRAYS:
                return
            try:
                item = value[key]
            except BaseException:
                continue
            _collect_array_candidates(
                item,
                f"{path}.{key}",
                output,
                depth=depth + 1,
                seen=seen,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:32]):
            if len(output) >= MAX_PRECISION_ARRAYS:
                return
            _collect_array_candidates(
                item,
                f"{path}[{index}]",
                output,
                depth=depth + 1,
                seen=seen,
            )


def _quantiles(values: Any) -> Mapping[str, Optional[float]]:
    import numpy as np  # type: ignore

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {name: None for name in ("q0", "q01", "q05", "q50", "q95", "q99", "q100")}
    probabilities = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    names = ["q0", "q01", "q05", "q50", "q95", "q99", "q100"]
    computed = np.quantile(array, probabilities)
    return {name: float(value) for name, value in zip(names, computed)}


def coordinate_precision_summary(value: Any) -> Mapping[str, Any]:
    import numpy as np  # type: ignore

    raw = _array_numpy(value)
    array = np.asarray(raw)
    summary: Dict[str, Any] = dict(array_summary(value))
    summary["coordinate_candidate"] = False
    if array.ndim < 2 or array.shape[-1] not in (2, 3):
        summary["coordinate_rejection"] = "last dimension is not XY/XYZ"
        return summary
    if not np.issubdtype(array.dtype, np.floating):
        summary["coordinate_rejection"] = "dtype is not floating"
        return summary

    summary["coordinate_candidate"] = True
    source64 = np.asarray(array, dtype=np.float64)
    cast32 = np.asarray(array, dtype=np.float32)
    roundtrip64 = cast32.astype(np.float64)
    finite = np.isfinite(source64)
    finite_values = source64[finite]
    summary["source_absolute_coordinate_quantiles"] = _quantiles(np.abs(finite_values))

    error = np.abs(roundtrip64 - source64)
    finite_error = error[np.isfinite(error)]
    summary["float32_roundtrip"] = {
        "max_abs_error": float(finite_error.max()) if finite_error.size else None,
        "mean_abs_error": float(finite_error.mean()) if finite_error.size else None,
        "nonzero_error_count": int((finite_error > 0.0).sum()),
        "error_quantiles": _quantiles(finite_error),
    }

    spacing = np.abs(np.spacing(cast32).astype(np.float64))
    finite_spacing = spacing[np.isfinite(spacing)]
    positive_spacing = finite_spacing[finite_spacing > 0.0]
    summary["float32_spacing"] = {
        "quantiles": _quantiles(positive_spacing),
        "maximum": float(positive_spacing.max()) if positive_spacing.size else None,
        "minimum_positive": float(positive_spacing.min()) if positive_spacing.size else None,
    }

    if array.shape[-2] < 2:
        summary["segment_rejection"] = "bead axis has fewer than two points"
        return summary

    source_delta = np.diff(source64, axis=-2)
    cast_delta = np.diff(roundtrip64, axis=-2)
    source_length = np.linalg.norm(source_delta, axis=-1)
    cast_length = np.linalg.norm(cast_delta, axis=-1)
    source_positive = source_length[source_length > 0.0]
    cast_positive = cast_length[cast_length > 0.0]
    source_zero = source_length == 0.0
    cast_zero = cast_length == 0.0
    collapsed_by_cast = (source_length > 0.0) & cast_zero

    endpoint_spacing = np.maximum(spacing[..., 1:, :], spacing[..., :-1, :])
    endpoint_spacing_norm = np.linalg.norm(endpoint_spacing, axis=-1)
    ratio = np.full(source_length.shape, np.nan, dtype=np.float64)
    positive_mask = source_length > 0.0
    ratio[positive_mask] = endpoint_spacing_norm[positive_mask] / source_length[positive_mask]
    finite_ratio = ratio[np.isfinite(ratio)]

    length_error = np.abs(cast_length - source_length)
    finite_length_error = length_error[np.isfinite(length_error)]
    summary["segments"] = {
        "segment_count": int(source_length.size),
        "source_zero_count": int(source_zero.sum()),
        "cast_float32_zero_count": int(cast_zero.sum()),
        "collapsed_by_float32_cast_count": int(collapsed_by_cast.sum()),
        "source_length_quantiles": _quantiles(source_positive),
        "cast_float32_length_quantiles": _quantiles(cast_positive),
        "length_roundtrip_error_quantiles": _quantiles(finite_length_error),
        "length_roundtrip_max_abs_error": (
            float(finite_length_error.max()) if finite_length_error.size else None
        ),
        "endpoint_ulp_norm_over_source_length_quantiles": _quantiles(finite_ratio),
        "endpoint_ulp_norm_over_source_length_maximum": (
            float(finite_ratio.max()) if finite_ratio.size else None
        ),
    }

    if collapsed_by_cast.any():
        indices = np.argwhere(collapsed_by_cast)
        summary["segments"]["first_collapsed_indices"] = [
            [int(item) for item in row]
            for row in indices[:16]
        ]
    return summary


def _translation_pair_summary(
    left_name: str,
    left_value: Any,
    right_name: str,
    right_value: Any,
) -> Optional[Mapping[str, Any]]:
    import numpy as np  # type: ignore

    left = np.asarray(_array_numpy(left_value), dtype=np.float64)
    right = np.asarray(_array_numpy(right_value), dtype=np.float64)
    if left.shape != right.shape or left.ndim < 2 or left.shape[-1] not in (2, 3):
        return None
    if left.size == 0:
        return None
    delta = right - left
    finite = np.isfinite(delta)
    if not finite.all():
        return {
            "left": left_name,
            "right": right_name,
            "shape": [int(item) for item in left.shape],
            "non_finite_delta_count": int((~finite).sum()),
        }
    flattened = delta.reshape(-1, delta.shape[-1])
    median_translation = np.median(flattened, axis=0)
    translation_residual = flattened - median_translation
    centered_left = left - np.take(left, [0], axis=-2)
    centered_right = right - np.take(right, [0], axis=-2)
    centered_difference = np.abs(centered_right - centered_left)
    return {
        "left": left_name,
        "right": right_name,
        "shape": [int(item) for item in left.shape],
        "median_translation": [float(item) for item in median_translation],
        "translation_residual_max_abs": float(np.abs(translation_residual).max()),
        "centered_difference_max_abs": float(centered_difference.max()),
        "centered_difference_mean_abs": float(centered_difference.mean()),
        "exact_after_first_point_centering": bool(np.array_equal(centered_left, centered_right)),
        "allclose_after_first_point_centering_1e_12": bool(
            np.allclose(centered_left, centered_right, rtol=1e-12, atol=1e-12)
        ),
        "allclose_after_first_point_centering_1e_9": bool(
            np.allclose(centered_left, centered_right, rtol=1e-9, atol=1e-9)
        ),
    }


def constraint_z_precision_diagnostic(
    error: BaseException,
    stagef: Any,
    torch: Any,
) -> Mapping[str, Any]:
    frames = _traceback_frames(error)
    frame_trace = [
        {
            "filename": frame.tb_frame.f_code.co_filename,
            "function": frame.tb_frame.f_code.co_name,
            "lineno": int(frame.tb_lineno),
        }
        for frame in frames
    ]
    target = None
    for frame in reversed(frames):
        if frame.tb_frame.f_code.co_name == CONSTRAINT_Z_FUNCTION:
            target = frame
            break
    if target is None:
        for frame in reversed(frames):
            if "constraint_z" in frame.tb_frame.f_code.co_name.lower():
                target = frame
                break
    if target is None:
        raise ExecutionError(
            "constraint-z precision marker was raised without a constraint-z traceback frame"
        )

    arrays: List[Tuple[str, Any]] = []
    for name, value in sorted(target.tb_frame.f_locals.items()):
        _collect_array_candidates(value, f"local.{name}", arrays)
    array_diagnostics: Dict[str, Any] = {}
    coordinate_arrays: List[Tuple[str, Any]] = []
    for path, value in arrays:
        try:
            summary = coordinate_precision_summary(value)
        except BaseException as capture_error:
            summary = {
                "capture_error": (
                    f"{type(capture_error).__module__}."
                    f"{type(capture_error).__qualname__}: {capture_error}"
                )
            }
        array_diagnostics[path] = summary
        if summary.get("coordinate_candidate") is True:
            coordinate_arrays.append((path, value))

    pairs: List[Mapping[str, Any]] = []
    for left_index, (left_name, left_value) in enumerate(coordinate_arrays):
        for right_name, right_value in coordinate_arrays[left_index + 1:]:
            if len(pairs) >= MAX_PRECISION_PAIRS:
                break
            pair = _translation_pair_summary(
                left_name,
                left_value,
                right_name,
                right_value,
            )
            if pair is not None:
                pairs.append(pair)
        if len(pairs) >= MAX_PRECISION_PAIRS:
            break

    source_function = getattr(stagef, CONSTRAINT_Z_FUNCTION, None)
    function_identity: Dict[str, Any] = {
        "module_has_named_callable": callable(source_function),
    }
    if callable(source_function):
        try:
            source = inspect.getsource(source_function)
            function_identity.update(
                {
                    "signature": str(inspect.signature(source_function)),
                    "source_sha256": sha256_bytes(source.encode("utf-8")),
                    "source_line_count": len(source.splitlines()),
                }
            )
        except (OSError, TypeError) as source_error:
            function_identity["source_error"] = (
                f"{type(source_error).__name__}: {source_error}"
            )

    structural_zero = 0
    cast_collapse = 0
    maximum_ratio: Optional[float] = None
    for summary in array_diagnostics.values():
        if not isinstance(summary, Mapping):
            continue
        segments = summary.get("segments")
        if not isinstance(segments, Mapping):
            continue
        structural_zero += int(segments.get("source_zero_count") or 0)
        cast_collapse += int(segments.get("collapsed_by_float32_cast_count") or 0)
        value = _finite_float(
            segments.get("endpoint_ulp_norm_over_source_length_maximum")
        )
        if value is not None:
            maximum_ratio = value if maximum_ratio is None else max(maximum_ratio, value)

    if cast_collapse > 0:
        interpretation = "FLOAT32_CAST_COLLAPSES_NONZERO_SEGMENTS"
        required_next = "DESIGN_A_FROZEN_TRAIN_ONLY_LENGTH_FLOOR_OR_NONLOG_FEATURE_FROM_CAPTURED_ROWS"
    elif structural_zero > 0:
        interpretation = "SOURCE_COORDINATES_CONTAIN_STRUCTURAL_ZERO_SEGMENTS"
        required_next = "SEPARATE_STRUCTURAL_ZERO_MASK_FROM_RESOLVABLE_SEGMENT_FEATURES"
    elif coordinate_arrays:
        interpretation = "NO_CAST_COLLAPSE_OBSERVED_BOUND_FORMULA_OR_THRESHOLD_REQUIRES_REVIEW"
        required_next = "COMPARE_CAPTURED_ULP_TO_SEGMENT_RATIOS_WITH_THE_FROZEN_BOUND_FORMULA"
    else:
        interpretation = "NO_COORDINATE_ARRAY_WAS_AVAILABLE_IN_THE_FAILING_FRAME"
        required_next = "EXPOSE_THE_BOUND_INPUTS_WITHIN_THE_EXISTING_SINGLE_ENTRYPOINT"

    return {
        "diagnosis_kind": "constraint_z_float32_precision_admission",
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "exception": str(error),
        "traceback_frames": frame_trace,
        "failing_frame": _source_excerpt(target),
        "failing_frame_locals": _bounded_local_snapshot(target),
        "function_identity": function_identity,
        "array_candidate_count": len(arrays),
        "coordinate_candidate_count": len(coordinate_arrays),
        "array_diagnostics": array_diagnostics,
        "translation_pair_diagnostics": pairs,
        "aggregate": {
            "source_zero_segment_count_across_candidates": structural_zero,
            "float32_cast_collapse_count_across_candidates": cast_collapse,
            "maximum_endpoint_ulp_norm_over_source_length": maximum_ratio,
        },
        "runtime_numerical_settings": runtime_numerical_observation(torch),
        "exact_gate_relaxed": False,
        "stagef_scientific_source_modified": True,
        "automatic_tolerance_inferred": False,
        "interpretation": interpretation,
        "required_next_path": required_next,
    }


def precision_diagnostic_summary(
    repository: Mapping[str, Any],
    base_result: Any,
    diagnostic: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    diagnostic_sha = sha256_bytes(stable_json_bytes(diagnostic))
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagef_structural_zero_feature_correction_not_admitted",
        "required_next_path": diagnostic["required_next_path"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": repository,
        "execution": {
            "mode": "one_python_process_one_real_calibration_precision_diagnosis",
            "pid": os.getpid(),
            "python_process_count": 1,
            "child_python_process_count": 0,
            "observed_non_python_child_commands": list(commands),
            "legacy_resume_wrappers_executed": False,
            "duplicate_workers_executed": False,
            "historical_temporal_gate_replayed": False,
            "scientific_calibration_started": True,
            "scientific_calibration_completed": False,
            "single_run_result_sha256": diagnostic_sha,
        },
        "base_evidence_validation": {
            "completed_without_exception": True,
            "return_type": (
                f"{type(base_result).__module__}.{type(base_result).__qualname__}"
            ),
        },
        "constraint_z_precision_diagnosis": diagnostic,
        "boundaries": {
            "frozen_probe_accessed": False,
            "selection_holdout_evaluated": "not_reached_before_precision_admission_failure",
            "formal_training_run": False,
            "reverse_sampling_run": False,
            "idm_run": False,
            "candidate_execution": False,
            "deformable_ravens_executed": False,
            "phase4": False,
            "cps": False,
            "checkpoint_saved": False,
            "weights_persisted": False,
            "surrogate_weights_persisted": False,
            "prediction_tensor_persisted": False,
            "candidate_tensor_persisted": False,
            "npz_saved": False,
            "cache_saved": False,
            "image_saved": False,
            "video_saved": False,
        },
    }

def run_once(repo: Path, repository: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen_environment = load_frozen_environment(repo)

    # The historical worker is retained and blob-bound as evidence, but it is not
    # part of the consolidated execution interface.  Source-token searches are
    # intentionally forbidden here: imports may be aliased, validation may be
    # internal to run_calibration, and comments are not an API contract.
    import torch  # type: ignore

    if torch.cuda.is_initialized():
        raise ExecutionError("CUDA was initialized before Stage-F import")

    stagef = importlib.import_module(STAGEF_MODULE)
    expected_stagef_path = (
        repo / "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py"
    ).resolve()
    actual_stagef_file = getattr(stagef, "__file__", None)
    if actual_stagef_file is None:
        raise ExecutionError("Stage-F module has no __file__ identity")
    actual_stagef_path = Path(actual_stagef_file).resolve()
    require_equal(
        "imported Stage-F module path",
        str(actual_stagef_path),
        str(expected_stagef_path),
    )
    validate_base = required_callable(stagef, "validate_base_evidence")
    run_calibration = required_callable(stagef, "run_calibration")

    if torch.cuda.is_initialized():
        raise ExecutionError("Stage-F import initialized CUDA")

    try:
        base_result = call_validate_base(validate_base, repo)
    except BaseException as error:
        if NUMERICAL_EQUIVALENCE_MARKER not in str(error):
            raise
        diagnostic = numerical_equivalence_diagnostic(error, torch)
        require_clean(repo, "main worktree after numerical diagnosis")
        require_clean(
            repo / "external/deformable-ravens",
            "submodule after numerical diagnosis",
        )
        return diagnostic_summary(repository, diagnostic)
    if torch.cuda.is_initialized():
        raise ExecutionError("base-evidence validation initialized CUDA")

    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r258_stagef_one_process_", dir="/tmp")
    )
    previous_cache_env = {
        key: os.environ.get(key)
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME")
    }
    os.environ["TMPDIR"] = str(temporary_root)
    os.environ["XDG_CACHE_HOME"] = str(temporary_root / "xdg")
    os.environ["TORCH_HOME"] = str(temporary_root / "torch")

    precision_diagnostic: Optional[Mapping[str, Any]] = None
    precision_commands: Sequence[Mapping[str, Any]] = ()
    try:
        try:
            with OneProcessGuard() as guard:
                raw_result = call_calibration(
                    run_calibration,
                    repo,
                    frozen_environment,
                )
        except BaseException as error:
            if not any(marker in str(error) for marker in CONSTRAINT_Z_PRECISION_MARKERS):
                raise
            precision_commands = list(guard.commands)
            precision_diagnostic = constraint_z_precision_diagnostic(
                error,
                stagef,
                torch,
            )
    finally:
        for key, previous in previous_cache_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        shutil.rmtree(temporary_root, ignore_errors=True)

    if precision_diagnostic is not None:
        require_clean(repo, "main worktree after precision diagnosis")
        require_clean(
            repo / "external/deformable-ravens",
            "submodule after precision diagnosis",
        )
        return precision_diagnostic_summary(
            repository,
            base_result,
            precision_diagnostic,
            precision_commands,
        )

    result = normalize_scientific_result(stagef, raw_result)
    validate_scientific_result(result)
    require_clean(repo, "main worktree after calibration")
    require_clean(repo / "external/deformable-ravens", "submodule after calibration")

    runtime = torch_observation(torch)
    if runtime["cuda_initialized"] is not True:
        raise ExecutionError("real Stage-F calibration did not initialize CUDA")

    result_sha = sha256_bytes(stable_json_bytes(result))
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": result["scientific_status"],
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "selected_configuration": result.get("selected_configuration"),
        "train_only_recommendation": result.get("train_only_recommendation"),
        "repository": repository,
        "execution": {
            "mode": "one_python_process_one_real_calibration",
            "pid": os.getpid(),
            "python_process_count": 1,
            "child_python_process_count": 0,
            "observed_non_python_child_commands": guard.commands,
            "legacy_resume_wrappers_executed": False,
            "duplicate_workers_executed": False,
            "historical_temporal_gate_replayed": False,
            "historical_temporal_gate_policy": (
                "retired as an execution prerequisite; immutable commits, "
                "science sources, and prior reports remain bound"
            ),
            "actual_calibration_is_required_operation_gate": True,
            "single_run_result_sha256": result_sha,
        },
        "environment": {
            "frozen_portable_contract_source": STAGEE_WORKER_EVIDENCE,
            "frozen_portable_contract_sha256": repository[
                "stagee_input_sha256"
            ][STAGEE_WORKER_EVIDENCE],
            "frozen_contract_passed_to_current_stagef": True,
            "stagef_scientific_source_modified": True,
            "cold_cuda_before_import": True,
            "cold_cuda_before_calibration": True,
            "current_runtime_after_calibration": runtime,
            "interpretation": (
                "The frozen Stage-E environment object supplies the unchanged "
                "Stage-F API contract. Current hardware is observed after the "
                "real run. Successful calibration is the current operation gate."
            ),
        },
        "base_evidence_validation": {
            "completed_without_exception": True,
            "return_type": (
                f"{type(base_result).__module__}.{type(base_result).__qualname__}"
            ),
        },
        "scientific_result": result,
        "boundaries": {
            "frozen_probe_accessed": False,
            "formal_training_run": False,
            "reverse_sampling_run": False,
            "idm_run": False,
            "candidate_execution": False,
            "deformable_ravens_executed": False,
            "phase4": False,
            "cps": False,
            "checkpoint_saved": False,
            "weights_persisted": False,
            "surrogate_weights_persisted": False,
            "prediction_tensor_persisted": False,
            "candidate_tensor_persisted": False,
            "npz_saved": False,
            "cache_saved": False,
            "image_saved": False,
            "video_saved": False,
        },
    }


def blocked_payload(error: BaseException, repository: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": "phase314b_r258_stagef_consolidated_blocked_v5_structural_zero",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagef_consolidated_execution_failed",
        "required_next_path": (
            "INSPECT_THE_SINGLE_CONSOLIDATED_ENTRYPOINT_WITHOUT_ADDING_A_WRAPPER"
        ),
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error": str(error),
        "traceback": traceback.format_exc(),
        "repository": repository,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "frozen_probe_accessed": False,
        "holdout_evaluated": "unknown_after_execution_failure",
        "formal_training_run": False,
        "reverse_sampling_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo = args.root.resolve()
    success_path = repo / SUCCESS_REPORT
    blocked_path = repo / BLOCKED_REPORT

    if success_path.exists() or blocked_path.exists():
        print("BLOCKED: consolidated output already exists", file=sys.stderr)
        return 2

    repository: Optional[Mapping[str, Any]] = None
    try:
        validate_environment()
        repository = validate_repository(repo)
        summary = run_once(repo, repository)
        data = stable_json_bytes(summary)
        write_once(success_path, data)
        print(
            json.dumps(
                {
                    "execution_verdict": "PASS",
                    "scientific_status": summary["scientific_status"],
                    "root_cause": summary["root_cause"],
                    "required_next_path": summary["required_next_path"],
                    "selected_configuration": summary[
                        "selected_configuration"
                    ],
                    "single_run_result_sha256": summary["execution"][
                        "single_run_result_sha256"
                    ],
                    "summary": str(success_path),
                    "summary_sha256": sha256_bytes(data),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = blocked_payload(error, repository)
        try:
            data = stable_json_bytes(payload)
            write_once(blocked_path, data)
            print(
                json.dumps(
                    {
                        "execution_verdict": "BLOCKED",
                        "error_type": payload["error_type"],
                        "error": payload["error"],
                        "blocked_report": str(blocked_path),
                        "blocked_report_sha256": sha256_bytes(data),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        except BaseException as report_error:
            print(
                "BLOCKED REPORT WRITE FAILED: "
                f"{type(report_error).__name__}: {report_error}",
                file=sys.stderr,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
