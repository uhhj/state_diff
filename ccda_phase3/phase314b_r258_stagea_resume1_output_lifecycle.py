"""Phase3.14b-r2.5.8 Stage-A Portable Resume1 output-lifecycle correction.

The first portable Stage-A attempt stopped before tests, implementation commit,
control replay, or candidate training because the shell preflight used
``mktemp FILE_TEMPLATE``.  That command created the file, while the worker's
``atomic_write_once`` contract correctly required a non-existent output path.

Resume1 preserves the original seven portable files and the first blocked
summary as immutable provenance.  It creates a temporary directory with
``mktemp -d`` and passes an absent ``environment.json`` child path to the
unchanged worker.  All new test/evidence/blocked outputs use a Resume1
write-once namespace.

No scientific, environment-compatibility, numerical-equivalence, candidate,
training, selection, or forbidden-stage rule is changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

BASE_EVIDENCE_COMMIT = (
    "f0a5bec1f89f625e74150e55e2da884413d72eb9"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)
FAILED_BLOCKED_PATH = (
    "reports/"
    "phase3_14b_r258_stagea_blocked_summary.json"
)
EXPECTED_FAILED_BLOCKED_SHA256 = (
    "832626d9cf6d146fe43cd76c7efada14b86658369ef1b49171cbe2d78a672768"
)

ORIGINAL_PORTABLE_FILES = {
    (
        "ccda_phase3/"
        "phase314b_r258_stagea_conflict_projected_k16.py"
    ): (
        "bc7b999386965621145728c6583b9126d"
        "c066fba3c1c8fc39badb513644db40c"
    ),
    "scripts/phase3_14b_r258_stagea_blocked.py": (
        "5c193b5f253b181eaae53f290ad5adb19"
        "8980cbbeb078830b508a565d08f0572"
    ),
    "scripts/phase3_14b_r258_stagea_run.sh": (
        "5bb70651f4cd58dda89657eb364a936a1"
        "507b532f16105ff34ab8789f5bb3e9f"
    ),
    (
        "scripts/"
        "phase3_14b_r258_stagea_run_calibration.py"
    ): (
        "07a605f4913a379210c537c66ebb65692"
        "28048c5c5a69bc23f839afb744f2f71"
    ),
    "scripts/phase3_14b_r258_stagea_test_gate.py": (
        "c7e85712a714de74c32b625b72062d33"
        "eb5c7d969be08f3f427bd34b78a9ed5a"
    ),
    "scripts/phase3_14b_r258_stagea_worker.py": (
        "326bbbd6ffd323b41a24a581c2bfe003"
        "3d664ec8b06317cf9582972e6c3df1c5"
    ),
    (
        "tests/"
        "test_phase3_14b_r258_stagea_conflict_projected_k16.py"
    ): (
        "1fd94be14cb0a3d5a542b836da771b25"
        "b32a6958ce583b3016ec846bcf06c65e"
    ),
}

RESUME1_NAMESPACE = "phase3_14b_r258_stagea_resume1"


class Resume1LifecycleError(RuntimeError):
    """Raised when failed provenance or output lifecycle changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            "refusing to overwrite write-once output: {}".format(target)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (
        ".{}.{}.tmp".format(target.name, os.getpid())
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Resume1LifecycleError(
            "JSON root is not an object: {}".format(path)
        )
    return value


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def make_absent_child(directory: Path, filename: str) -> Path:
    parent = Path(directory)
    if not parent.is_dir():
        raise Resume1LifecycleError(
            "temporary parent is not a directory: {}".format(parent)
        )
    if not filename or Path(filename).name != filename:
        raise Resume1LifecycleError("temporary filename is invalid")
    output = parent / filename
    if output.exists():
        raise FileExistsError(
            "temporary output must not exist: {}".format(output)
        )
    return output


def validate_failed_blocked_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stagea_"
            "execution_failed_before_completion"
        ),
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "control_replay_exact": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise Resume1LifecycleError(
                "failed blocked field changed: {}".format(key)
            )
    command = str(payload.get("failed_command", ""))
    if "phase3_14b_r258_stagea_worker.py" not in command:
        raise Resume1LifecycleError(
            "failed command does not identify environment worker"
        )


def validate_failed_files(root: Path, *, require_tracked: bool) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    if git_output(repository_root, "branch", "--show-current") != "Experiment1":
        raise Resume1LifecycleError("Experiment1 branch is required")
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_EVIDENCE_COMMIT,
            "HEAD",
        ],
        cwd=str(repository_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise Resume1LifecycleError(
            "Resume4 evidence is not an ancestor"
        )

    observed = {}
    for relative, expected_sha in ORIGINAL_PORTABLE_FILES.items():
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise Resume1LifecycleError(
                "original portable file changed: {}".format(relative)
            )
        observed[relative] = actual
        if require_tracked:
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=str(repository_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            committed = subprocess.check_output(
                ["git", "show", "HEAD:{}".format(relative)],
                cwd=str(repository_root),
            )
            if committed != path.read_bytes():
                raise Resume1LifecycleError(
                    "tracked original file differs from HEAD: {}".format(
                        relative
                    )
                )

    blocked_path = repository_root / FAILED_BLOCKED_PATH
    if not blocked_path.is_file():
        raise FileNotFoundError(blocked_path)
    blocked_sha = sha256_file(blocked_path)
    if blocked_sha != EXPECTED_FAILED_BLOCKED_SHA256:
        raise Resume1LifecycleError("failed blocked SHA changed")
    validate_failed_blocked_payload(load_json(blocked_path))
    if require_tracked:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", FAILED_BLOCKED_PATH],
            cwd=str(repository_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        committed = subprocess.check_output(
            ["git", "show", "HEAD:{}".format(FAILED_BLOCKED_PATH)],
            cwd=str(repository_root),
        )
        if committed != blocked_path.read_bytes():
            raise Resume1LifecycleError(
                "tracked failed blocked report differs from HEAD"
            )

    submodule = repository_root / "external/deformable-ravens"
    if git_output(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise Resume1LifecycleError("submodule commit changed")
    if git_output(
        submodule,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise Resume1LifecycleError("submodule worktree is dirty")

    return {
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "original_portable_file_sha256": observed,
        "failed_blocked_path": FAILED_BLOCKED_PATH,
        "failed_blocked_sha256": blocked_sha,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
        "require_tracked": bool(require_tracked),
    }


def correction_record(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema":
            "phase314b_r258_stagea_resume1_output_lifecycle_v1",
        "namespace": RESUME1_NAMESPACE,
        "provenance": dict(provenance),
        "failure_locus": "precreated_environment_probe_output",
        "old_lifecycle": (
            "mktemp FILE_TEMPLATE created the output file before "
            "atomic_write_once"
        ),
        "new_lifecycle": (
            "mktemp -d creates only a directory; environment.json "
            "must not exist before worker invocation"
        ),
        "scientific_contract_changed": False,
        "environment_contract_changed": False,
        "control_reference_contract_changed": False,
        "candidate_population_changed": False,
        "selection_contract_changed": False,
    }
