#!/usr/bin/env python3
"""Run Stage F once, end to end, in one Python process.

This is the terminal replacement for the Stage-F Resume wrapper chain.  It
preserves prior commits and reports as immutable evidence, binds the unchanged
Stage-F science to its implementation commit, starts with a cold CUDA context,
runs one real calibration directly through the scientific module, and writes
one write-once JSON result.

A completed calibration may legitimately return scientific_status READY or
BLOCKED.  Either is a valid scientific result.  An execution error writes one
blocked report and must not be retried in place.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import multiprocessing
import os
import platform
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

PHASE = "Phase3.14b-r2.5.8 Stage F Execution Consolidation"
SCHEMA = "phase314b_r258_stagef_consolidated_e2e_v1"
DEFAULT_ROOT = Path("/data/state_diff2")

EXPECTED_BRANCH = "Experiment1"
EXPECTED_REMOTE_HEAD = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_PARENT = "e2fdbeca10df7fbf1925a3c6ec02b8a8a864c851"
EXPECTED_SUBJECT = "Phase3.14b-r2.5.8 Stage F: consolidate execution path"
IMPLEMENTATION_PATH = "scripts/phase3_14b_r258_stagef_consolidated_e2e.py"

STAGEF_ANCHOR = "f41a2364b7283b1eb963d305823804374ddfc8d6"
STAGEE_ANCHOR = "6758ea7ad800667a436b0243d3b1f6c63256d854"
STAGEF_MODULE = "ccda_phase3.phase314b_r258_stagef_constraint_aware_surrogate"
STAGEF_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r258_stagef_constraint_aware_surrogate.py",
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

SUCCESS_REPORT = "reports/phase3_14b_r258_stagef_consolidated_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stagef_consolidated_blocked_summary.json"

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

    changed = git_bytes(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        "--no-renames",
        head,
    ).split(b"\0")
    if changed and changed[-1] == b"":
        changed.pop()
    expected_change = [b"A", IMPLEMENTATION_PATH.encode("utf-8")]
    if changed != expected_change:
        raise ExecutionError(
            "implementation commit must add only the consolidated runner: "
            f"expected={expected_change!r}, actual={changed!r}"
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
        for relpath in STAGEF_PATHS
    }
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
        "implementation_path": IMPLEMENTATION_PATH,
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


def run_once(repo: Path, repository: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen_environment = load_frozen_environment(repo)

    canonical_worker = (repo / CANONICAL_WORKER).read_text(encoding="utf-8")
    for token in ("validate_base_evidence", "run_calibration"):
        if token not in canonical_worker:
            raise ExecutionError(
                f"canonical corrected worker no longer references {token}"
            )

    import torch  # type: ignore

    if torch.cuda.is_initialized():
        raise ExecutionError("CUDA was initialized before Stage-F import")

    stagef = importlib.import_module(STAGEF_MODULE)
    validate_base = required_callable(stagef, "validate_base_evidence")
    run_calibration = required_callable(stagef, "run_calibration")

    if torch.cuda.is_initialized():
        raise ExecutionError("Stage-F import initialized CUDA")

    base_result = call_validate_base(validate_base, repo)
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

    try:
        with OneProcessGuard() as guard:
            raw_result = call_calibration(
                run_calibration,
                repo,
                frozen_environment,
            )
    finally:
        for key, previous in previous_cache_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        shutil.rmtree(temporary_root, ignore_errors=True)

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
            "frozen_contract_passed_to_unchanged_stagef": True,
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
        "schema": "phase314b_r258_stagef_consolidated_blocked_v1",
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
