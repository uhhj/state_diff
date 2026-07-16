"""Phase3.14b-r2.5.7 Stage-D Resume4 RTX-3090 replay namespace.

Resume3 correctly established two facts on the prior RTX-4090 execution host:

* the hand-reconstructed Stage-C calibration differed in 220 fields; and
* the frozen Stage-C ``run_calibration`` entry point itself could not reproduce
  the committed Stage-C control identity on that host.

The Resume3 contract is write-once and must not be deleted, overwritten, or
rerun.  Resume4 creates an independent write-once namespace for an explicitly
selected RTX 3090 replay host.

The hardware/software probe runs in a disposable child process.  The manual
calibration difference probe also runs in a disposable child process.  The main
worker must still have a cold CUDA context immediately before delegating to the
frozen Resume3 execution path.  Resume3 then remains responsible for:

* frozen Stage-C entry-point calibration/control capture;
* stopping before the first nonzero Stage-C candidate;
* exact calibration and control comparison;
* the existing Resume2 schema adapter;
* the existing Resume1/Stage-D gated calibration;
* all six registered gated candidates;
* train-only selection and scientific classification; and
* all forbidden-stage boundaries.

Resume4 does not relax any SHA, floating-point, control, selection, or worker
identity requirement.  It only introduces a new immutable environment identity
and a new write-once output namespace.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np

from ccda_phase3 import phase314b_r257_staged_resume3_calibration_trace as resume3

PHASE = "Phase3.14b-r2.5.7 Stage D Resume4 RTX 3090"
PHASE_ID = "phase314b_r257_staged_resume4_3090"

STAGED_IMPLEMENTATION_COMMIT = (
    "ccff30947a85028a2c7974edb5b008f3b3de2757"
)
STAGED_BLOCKED_EVIDENCE_COMMIT = (
    "630cff532d811c889702bb236766b659bdfd83d4"
)
RESUME1_IMPLEMENTATION_COMMIT = (
    "d34838bfebbde2376d128dff398dadcc97877191"
)
RESUME1_BLOCKED_EVIDENCE_COMMIT = (
    "85ec59d45c6445e9cd61e91b7e6d6d94fe729ecf"
)
RESUME2_IMPLEMENTATION_COMMIT = (
    "e68d069a8ad5135a10d8e4ab3a07f46049d596aa"
)
RESUME2_BLOCKED_EVIDENCE_COMMIT = (
    "18bad1f2d542875b788e1959e048717ec39b5a49"
)
RESUME3_IMPLEMENTATION_COMMIT = (
    "ef5eebb964ac7a8198506962646e048ccf0f6605"
)
STAGEC_EVIDENCE_COMMIT = (
    "a002246558e66029bdd6279e09c95e1303e2356c"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

RESUME3_BLOCKED = (
    "reports/"
    "phase3_14b_r257_staged_resume3_blocked_summary.json"
)
RESUME3_TEST_GATE = (
    "reports/"
    "phase3_14b_r257_staged_resume3_test_gate_summary.json"
)
EXPECTED_RESUME3_BLOCKED_SHA256 = (
    "c769fa1f0af69dfe24612cf9bdb7f43c8ce03fb93f039c069b84b07d088a89cd"
)
EXPECTED_RESUME3_TEST_GATE_SHA256 = (
    "c1cf8558bc0d5a44018812b034e4ce5d3498c290c68c20d0ef8a249938b05adc"
)

EXPECTED_PYTHON = (3, 9, 15)
EXPECTED_NUMPY = "1.23.3"
EXPECTED_TORCH = "1.12.1.post200"
EXPECTED_TORCH_CUDA = "11.2"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_GPU_NAME_TOKEN = "RTX 3090"
MINIMUM_TOTAL_MEMORY_BYTES = 23 * 1024 ** 3

RESUME3_IMPLEMENTATION_FILES = (
    (
        "ccda_phase3/"
        "phase314b_r257_staged_resume3_calibration_trace.py"
    ),
    "scripts/phase3_14b_r257_staged_resume3_worker.py",
    "scripts/phase3_14b_r257_staged_resume3_run.py",
    "scripts/phase3_14b_r257_staged_resume3_test_gate.py",
    "scripts/phase3_14b_r257_staged_resume3_blocked.py",
    "scripts/phase3_14b_r257_staged_resume3_run.sh",
    (
        "tests/"
        "test_phase3_14b_r257_staged_resume3_calibration_trace.py"
    ),
)


class Resume4EnvironmentError(RuntimeError):
    """Raised when provenance or the RTX-3090 replay contract changes."""


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("non-finite array cannot be serialized")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite scalar cannot be serialized")
        return value
    raise TypeError(
        "unsupported JSON value: {!r}".format(type(value))
    )


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            "refusing to overwrite write-once output: {}".format(
                target
            )
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
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Resume4EnvironmentError(
            "JSON root is not an object: {}".format(path)
        )
    return value


def assert_commit_ancestor(root: Path, commit: str) -> None:
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise Resume4EnvironmentError(
            "required commit is not an ancestor: {}".format(
                commit
            )
        )


def assert_file_bound_to_commit(
    root: Path,
    relative: str,
    commit: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(commit, relative),
        ],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise Resume4EnvironmentError(
            "bound file changed: {}".format(relative)
        )
    return sha256_bytes(observed)


def assert_tracked_current_blob(
    root: Path,
    relative: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    committed = subprocess.check_output(
        ["git", "show", "HEAD:{}".format(relative)],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise Resume4EnvironmentError(
            "tracked report differs from HEAD: {}".format(
                relative
            )
        )
    return sha256_bytes(observed)


def validate_resume3_failure(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    for commit in (
        STAGEC_EVIDENCE_COMMIT,
        STAGED_IMPLEMENTATION_COMMIT,
        STAGED_BLOCKED_EVIDENCE_COMMIT,
        RESUME1_IMPLEMENTATION_COMMIT,
        RESUME1_BLOCKED_EVIDENCE_COMMIT,
        RESUME2_IMPLEMENTATION_COMMIT,
        RESUME2_BLOCKED_EVIDENCE_COMMIT,
        RESUME3_IMPLEMENTATION_COMMIT,
    ):
        assert_commit_ancestor(repository_root, commit)

    implementation_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            RESUME3_IMPLEMENTATION_COMMIT,
        )
        for relative in RESUME3_IMPLEMENTATION_FILES
    }

    blocked_sha = assert_tracked_current_blob(
        repository_root,
        RESUME3_BLOCKED,
    )
    gate_sha = assert_tracked_current_blob(
        repository_root,
        RESUME3_TEST_GATE,
    )
    if blocked_sha != EXPECTED_RESUME3_BLOCKED_SHA256:
        raise Resume4EnvironmentError(
            "Resume3 blocked-summary SHA changed"
        )
    if gate_sha != EXPECTED_RESUME3_TEST_GATE_SHA256:
        raise Resume4EnvironmentError(
            "Resume3 test-gate SHA changed"
        )

    blocked = load_json(repository_root / RESUME3_BLOCKED)
    expected_blocked = {
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r257_staged_resume3_"
            "execution_failed_before_completion"
        ),
        "resume3_correction_applied": False,
        "control_replay_exact": False,
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }
    for key, expected in expected_blocked.items():
        if blocked.get(key) != expected:
            raise Resume4EnvironmentError(
                "Resume3 blocked field changed: {}".format(key)
            )

    gate = load_json(repository_root / RESUME3_TEST_GATE)
    if gate.get("verdict") != "PASS":
        raise Resume4EnvironmentError(
            "Resume3 test gate is not PASS"
        )
    if int(gate.get("passed_test_count", -1)) != 812:
        raise Resume4EnvironmentError(
            "Resume3 pass count changed"
        )
    if int(gate.get("test_file_count", -1)) != 45:
        raise Resume4EnvironmentError(
            "Resume3 test-file count changed"
        )
    if int(
        gate.get("r257_staged_resume3_new_passed", -1)
    ) != 28:
        raise Resume4EnvironmentError(
            "Resume3 new-test count changed"
        )

    return {
        "resume3_implementation_commit":
            RESUME3_IMPLEMENTATION_COMMIT,
        "resume3_implementation_sha256":
            implementation_sha,
        "resume3_blocked_path": RESUME3_BLOCKED,
        "resume3_blocked_sha256": blocked_sha,
        "resume3_test_gate_path": RESUME3_TEST_GATE,
        "resume3_test_gate_sha256": gate_sha,
        "resume3_blocked": blocked,
        "resume3_test_gate": gate,
    }


def normalize_gpu_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def parse_nvidia_smi_row(row: str) -> Dict[str, str]:
    parts = [part.strip() for part in str(row).split(",")]
    if len(parts) != 5:
        raise Resume4EnvironmentError(
            "unexpected nvidia-smi row: {!r}".format(row)
        )
    return {
        "name": parts[0],
        "uuid": parts[1],
        "pci_bus_id": parts[2],
        "driver_version": parts[3],
        "memory_total_mib": parts[4],
    }


def environment_probe() -> Dict[str, Any]:
    """Probe RTX-3090 identity in a disposable process."""
    import torch

    python_version = tuple(sys.version_info[:3])
    if python_version != EXPECTED_PYTHON:
        raise Resume4EnvironmentError(
            "Python contract changed: {}".format(
                python_version
            )
        )
    if np.__version__ != EXPECTED_NUMPY:
        raise Resume4EnvironmentError(
            "NumPy contract changed: {}".format(
                np.__version__
            )
        )
    if torch.__version__ != EXPECTED_TORCH:
        raise Resume4EnvironmentError(
            "PyTorch contract changed: {}".format(
                torch.__version__
            )
        )
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise Resume4EnvironmentError(
            "Torch CUDA contract changed: {}".format(
                torch.version.cuda
            )
        )
    if not torch.cuda.is_available():
        raise Resume4EnvironmentError(
            "CUDA is unavailable"
        )
    if torch.cuda.device_count() < 1:
        raise Resume4EnvironmentError(
            "no CUDA device is visible"
        )

    device_index = 0
    torch_name = normalize_gpu_name(
        torch.cuda.get_device_name(device_index)
    )
    properties = torch.cuda.get_device_properties(
        device_index
    )
    capability = tuple(
        torch.cuda.get_device_capability(device_index)
    )
    if EXPECTED_GPU_NAME_TOKEN not in torch_name.upper():
        # EXPECTED_GPU_NAME_TOKEN is already uppercase except for spacing.
        if EXPECTED_GPU_NAME_TOKEN.upper() not in torch_name.upper():
            raise Resume4EnvironmentError(
                "cuda:0 is not an RTX 3090: {}".format(
                    torch_name
                )
            )
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise Resume4EnvironmentError(
            "RTX-3090 compute capability changed: {}".format(
                capability
            )
        )
    if int(properties.total_memory) < MINIMUM_TOTAL_MEMORY_BYTES:
        raise Resume4EnvironmentError(
            "visible GPU memory is below RTX-3090 contract"
        )

    smi_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu="
            "name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    ).strip()
    if "\n" in smi_output:
        raise Resume4EnvironmentError(
            "nvidia-smi returned multiple rows for device 0"
        )
    smi = parse_nvidia_smi_row(smi_output)
    smi_name = normalize_gpu_name(smi["name"])
    if EXPECTED_GPU_NAME_TOKEN.upper() not in smi_name.upper():
        raise Resume4EnvironmentError(
            "nvidia-smi device 0 is not RTX 3090: {}".format(
                smi_name
            )
        )

    probe = {
        "schema":
            "phase314b_r257_staged_resume4_3090_environment_v1",
        "python_version": list(python_version),
        "python_implementation":
            platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_index": int(device_index),
        "torch_device_name": torch_name,
        "compute_capability": list(capability),
        "total_memory_bytes":
            int(properties.total_memory),
        "multi_processor_count":
            int(properties.multi_processor_count),
        "nvidia_smi": smi,
        "cuda_visible_devices":
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expected_gpu_name_token":
            EXPECTED_GPU_NAME_TOKEN,
        "expected_compute_capability":
            list(EXPECTED_COMPUTE_CAPABILITY),
        "contract_pass": True,
    }
    probe["environment_sha256"] = sha256_bytes(
        stable_json_bytes(probe)
    )
    return probe


def assert_cold_cuda_context() -> Dict[str, Any]:
    """Require no CUDA context in the main replay worker."""
    import torch

    initialized = bool(torch.cuda.is_initialized())
    if initialized:
        raise Resume4EnvironmentError(
            "main Resume4 worker initialized CUDA before "
            "the frozen Stage-C entry point"
        )
    return {
        "checked": True,
        "torch_cuda_is_initialized": False,
        "purpose": (
            "keep environment/manual probes out of the "
            "main Stage-C replay CUDA context"
        ),
    }


def run_resume4(
    *,
    root: Path,
    environment: Mapping[str, Any],
    manual_diff: Mapping[str, Any],
    spec: Optional[Any] = None,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    failure = validate_resume3_failure(
        repository_root
    )

    if environment.get("contract_pass") is not True:
        raise Resume4EnvironmentError(
            "RTX-3090 environment probe did not pass"
        )
    if (
        EXPECTED_GPU_NAME_TOKEN.upper()
        not in str(
            environment.get("torch_device_name", "")
        ).upper()
    ):
        raise Resume4EnvironmentError(
            "environment payload does not identify RTX 3090"
        )
    if tuple(
        environment.get("compute_capability", ())
    ) != EXPECTED_COMPUTE_CAPABILITY:
        raise Resume4EnvironmentError(
            "environment payload compute capability changed"
        )
    expected_environment_sha = sha256_bytes(
        stable_json_bytes(
            {
                key: value
                for key, value in environment.items()
                if key != "environment_sha256"
            }
        )
    )
    if (
        environment.get("environment_sha256")
        != expected_environment_sha
    ):
        raise Resume4EnvironmentError(
            "environment fingerprint SHA is invalid"
        )
    if manual_diff.get("completed") is not True:
        raise Resume4EnvironmentError(
            "manual calibration diff did not complete"
        )
    if manual_diff.get("training_performed") is not False:
        raise Resume4EnvironmentError(
            "manual calibration diff unexpectedly trained"
        )
    if manual_diff.get("frozen_probe_accessed") is not False:
        raise Resume4EnvironmentError(
            "manual calibration diff accessed frozen probe"
        )

    cold_context = assert_cold_cuda_context()
    result = resume3.run_resume3(
        root=repository_root,
        manual_diff=manual_diff,
        spec=spec,
    )

    result = copy.deepcopy(result)
    result["phase"] = PHASE
    result["phase_id"] = PHASE_ID
    result["schema"] = (
        "phase314b_r257_staged_resume4_3090_worker_result_v1"
    )
    result["resume4_3090_environment"] = {
        "environment_probe": copy.deepcopy(
            dict(environment)
        ),
        "cold_main_worker_context":
            cold_context,
        "resume3_implementation_commit":
            RESUME3_IMPLEMENTATION_COMMIT,
        "resume3_blocked_sha256":
            EXPECTED_RESUME3_BLOCKED_SHA256,
        "resume3_test_gate_sha256":
            EXPECTED_RESUME3_TEST_GATE_SHA256,
        "new_write_once_namespace":
            "phase3_14b_r257_staged_resume4_3090",
        "resume3_files_modified": False,
        "resume3_reports_modified": False,
        "correction_scope": (
            "re-execute the frozen Resume3 authority on an "
            "operator-selected RTX 3090 in a new write-once "
            "namespace; no identity or scientific gate relaxed"
        ),
    }
    result["resume3_failed_attempt"] = failure
    result["resume4_3090_correction_applied"] = True
    result["resume3_files_modified"] = False
    result["resume3_blocked_modified"] = False
    result["resume3_test_gate_modified"] = False

    contract = result.get("calibration_contract")
    if not isinstance(contract, MutableMapping):
        raise Resume4EnvironmentError(
            "Resume3 calibration contract is missing"
        )
    contract = copy.deepcopy(dict(contract))
    contract["resume4_3090_environment"] = (
        result["resume4_3090_environment"]
    )
    contract["correction_scope"] = (
        "new RTX-3090 write-once replay namespace; frozen "
        "Stage-C entrypoint and Resume1/Stage-D execution "
        "remain authoritative"
    )
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = sha256_bytes(
        stable_json_bytes(contract)
    )
    result["calibration_contract"] = contract
    return result


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "resume4_3090_environment":
            result["resume4_3090_environment"],
        "resume3_calibration_attribution":
            result["resume3_calibration_attribution"],
        "control_trace_correction":
            result["control_trace_correction"],
        "split": result["split"],
        "calibration_contract":
            result["calibration_contract"],
        "selection": result["selection"],
        "classification": result["classification"],
        "selected_configuration":
            result["selected_configuration"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(
        identity_projection(left)
    )
    right_payload = stable_json_bytes(
        identity_projection(right)
    )
    return {
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "environment_exact": (
            left["resume4_3090_environment"]
            == right["resume4_3090_environment"]
        ),
        "manual_diff_exact": (
            left["resume3_calibration_attribution"][
                "manual_calibration_diff"
            ]
            == right["resume3_calibration_attribution"][
                "manual_calibration_diff"
            ]
        ),
        "stagec_capture_exact": (
            left["resume3_calibration_attribution"][
                "stagec_entrypoint_capture"
            ]
            == right["resume3_calibration_attribution"][
                "stagec_entrypoint_capture"
            ]
        ),
        "control_trace_exact": (
            left["control_trace_correction"]
            == right["control_trace_correction"]
        ),
        "calibration_contract_exact": (
            left["calibration_contract"]
            == right["calibration_contract"]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
    }
