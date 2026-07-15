"""Phase3.14b-r2.5.7 Stage-D Resume2 schema correction.

Stage-D Resume1 stopped before training because it treated the return value of
``stagec_topk.validate_immutable_inputs`` as though the Stage-A contract were a
top-level field.  The frozen validator schema is nested:

    Stage-C validation
      ["upstream_immutable"] -> Stage-B mechanism validation
        ["stagea_contract"] -> Stage-A objective contract
        ["upstream_immutable"] -> Stage-A validation
          ["staged3_contract"] -> frozen one-sided upper gate

Resume1 attempted:

    upstream["stagea_contract"]
    upstream["upstream_immutable"]["staged3_contract"]

Both are one level too shallow.  Resume2 adds a read-only compatibility adapter
that:

* validates the canonical nested schema;
* deep-copies only the mapping spine needed by Resume1;
* exposes ``stagea_contract`` as a top-level compatibility alias;
* exposes ``staged3_contract`` inside the first upstream layer;
* verifies aliases are equal to the canonical values;
* temporarily installs the adapted validator only while calling the immutable
  Resume1 execution function; and
* restores the original validator on success or failure.

No Resume1 or Stage-D file is modified.  The original failed reports remain
immutable.  Once the schema correction permits Resume1 to enter its normal
execution, all existing Resume1 exact-control, candidate, selection, scientific
classification, write-once, and forbidden-stage contracts remain authoritative.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, MutableMapping, Optional

import numpy as np

from ccda_phase3 import phase314b_r257_stagec_topk_quadratic as stagec_topk
from ccda_phase3 import phase314b_r257_staged_resume1_control_trace as resume1

PHASE = "Phase3.14b-r2.5.7 Stage D Resume2"
PHASE_ID = "phase314b_r257_staged_resume2"

STAGED_IMPLEMENTATION_COMMIT = (
    "ccff30947a85028a2c7974edb5b008f3b3de2757"
)
STAGED_BLOCKED_EVIDENCE_COMMIT = (
    "630cff532d811c889702bb236766b659bdfd83d4"
)
RESUME1_IMPLEMENTATION_COMMIT = (
    "d34838bfebbde2376d128dff398dadcc97877191"
)
STAGEC_EVIDENCE_COMMIT = (
    "a002246558e66029bdd6279e09c95e1303e2356c"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

RESUME1_BLOCKED = (
    "reports/"
    "phase3_14b_r257_staged_resume1_blocked_summary.json"
)
RESUME1_TEST_GATE = (
    "reports/"
    "phase3_14b_r257_staged_resume1_test_gate_summary.json"
)
EXPECTED_RESUME1_BLOCKED_SHA256 = (
    "ab1ec6cd75c7164b29da29e296925fb9ed9897bcc552fd835060e0b1e1c4eb24"
)
EXPECTED_RESUME1_TEST_GATE_SHA256 = (
    "d31a882153134304b37f3fd2b16b853fdec8c177ef49c3ee879813a43ea06298"
)

RESUME1_IMPLEMENTATION_FILES = (
    (
        "ccda_phase3/"
        "phase314b_r257_staged_resume1_control_trace.py"
    ),
    "scripts/phase3_14b_r257_staged_resume1_worker.py",
    "scripts/phase3_14b_r257_staged_resume1_run.py",
    "scripts/phase3_14b_r257_staged_resume1_test_gate.py",
    "scripts/phase3_14b_r257_staged_resume1_blocked.py",
    "scripts/phase3_14b_r257_staged_resume1_run.sh",
    (
        "tests/"
        "test_phase3_14b_r257_staged_resume1_control_trace.py"
    ),
)


class Resume2SchemaError(RuntimeError):
    """Raised when the frozen schema or failure provenance changes."""


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
        raise Resume2SchemaError(
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
        raise Resume2SchemaError(
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
        raise Resume2SchemaError(
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
        raise Resume2SchemaError(
            "tracked report differs from HEAD: {}".format(
                relative
            )
        )
    return sha256_bytes(observed)


def validate_resume1_failure(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    for commit in (
        STAGEC_EVIDENCE_COMMIT,
        STAGED_IMPLEMENTATION_COMMIT,
        STAGED_BLOCKED_EVIDENCE_COMMIT,
        RESUME1_IMPLEMENTATION_COMMIT,
    ):
        assert_commit_ancestor(repository_root, commit)

    implementation_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            RESUME1_IMPLEMENTATION_COMMIT,
        )
        for relative in RESUME1_IMPLEMENTATION_FILES
    }

    blocked_sha = assert_tracked_current_blob(
        repository_root,
        RESUME1_BLOCKED,
    )
    gate_sha = assert_tracked_current_blob(
        repository_root,
        RESUME1_TEST_GATE,
    )
    if blocked_sha != EXPECTED_RESUME1_BLOCKED_SHA256:
        raise Resume2SchemaError(
            "Resume1 blocked-summary SHA changed"
        )
    if gate_sha != EXPECTED_RESUME1_TEST_GATE_SHA256:
        raise Resume2SchemaError(
            "Resume1 test-gate SHA changed"
        )

    blocked = load_json(repository_root / RESUME1_BLOCKED)
    expected_blocked = {
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r257_staged_resume1_"
            "execution_failed_before_completion"
        ),
        "resume1_correction_applied": False,
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
            raise Resume2SchemaError(
                "Resume1 blocked field changed: {}".format(key)
            )

    gate = load_json(repository_root / RESUME1_TEST_GATE)
    if gate.get("verdict") != "PASS":
        raise Resume2SchemaError(
            "Resume1 test gate is not PASS"
        )
    if int(gate.get("passed_test_count", -1)) != 763:
        raise Resume2SchemaError(
            "Resume1 pass count changed"
        )
    if int(gate.get("test_file_count", -1)) != 43:
        raise Resume2SchemaError(
            "Resume1 test-file count changed"
        )
    if int(
        gate.get("r257_staged_resume1_new_passed", -1)
    ) != 14:
        raise Resume2SchemaError(
            "Resume1 new-test count changed"
        )

    return {
        "staged_implementation_commit":
            STAGED_IMPLEMENTATION_COMMIT,
        "staged_blocked_evidence_commit":
            STAGED_BLOCKED_EVIDENCE_COMMIT,
        "resume1_implementation_commit":
            RESUME1_IMPLEMENTATION_COMMIT,
        "resume1_implementation_sha256":
            implementation_sha,
        "resume1_blocked_path": RESUME1_BLOCKED,
        "resume1_blocked_sha256": blocked_sha,
        "resume1_test_gate_path": RESUME1_TEST_GATE,
        "resume1_test_gate_sha256": gate_sha,
        "resume1_blocked": blocked,
        "resume1_test_gate": gate,
    }


def require_mapping(
    value: Any,
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Resume2SchemaError(
            "schema path is not a mapping: {}".format(path)
        )
    return value


def resolve_canonical_stagec_schema(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve and validate the frozen Stage-C→B→A schema."""
    stagec_layer = require_mapping(
        payload,
        path="stagec",
    )
    stageb_layer = require_mapping(
        stagec_layer.get("upstream_immutable"),
        path="stagec.upstream_immutable",
    )
    stagea_contract = require_mapping(
        stageb_layer.get("stagea_contract"),
        path=(
            "stagec.upstream_immutable."
            "stagea_contract"
        ),
    )
    stagea_layer = require_mapping(
        stageb_layer.get("upstream_immutable"),
        path=(
            "stagec.upstream_immutable."
            "upstream_immutable"
        ),
    )
    staged3_contract = require_mapping(
        stagea_layer.get("staged3_contract"),
        path=(
            "stagec.upstream_immutable."
            "upstream_immutable.staged3_contract"
        ),
    )
    return {
        "stagec_layer": stagec_layer,
        "stageb_layer": stageb_layer,
        "stagea_layer": stagea_layer,
        "stagea_contract": stagea_contract,
        "staged3_contract": staged3_contract,
        "canonical_stagea_contract_path": (
            "upstream_immutable.stagea_contract"
        ),
        "canonical_staged3_contract_path": (
            "upstream_immutable."
            "upstream_immutable.staged3_contract"
        ),
    }


def adapt_stagec_schema_for_resume1(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a non-mutating compatibility view for frozen Resume1."""
    resolved = resolve_canonical_stagec_schema(payload)
    adapted = copy.deepcopy(dict(payload))

    adapted_stageb = require_mapping(
        adapted.get("upstream_immutable"),
        path="adapted.upstream_immutable",
    )
    adapted_stageb_mutable: MutableMapping[str, Any] = dict(
        adapted_stageb
    )
    adapted["upstream_immutable"] = adapted_stageb_mutable

    canonical_stagea_contract = copy.deepcopy(
        resolved["stagea_contract"]
    )
    canonical_staged3_contract = copy.deepcopy(
        resolved["staged3_contract"]
    )

    top_existing = adapted.get("stagea_contract")
    if (
        top_existing is not None
        and top_existing != canonical_stagea_contract
    ):
        raise Resume2SchemaError(
            "existing top-level stagea_contract alias "
            "conflicts with canonical schema"
        )
    upstream_existing = adapted_stageb_mutable.get(
        "staged3_contract"
    )
    if (
        upstream_existing is not None
        and upstream_existing != canonical_staged3_contract
    ):
        raise Resume2SchemaError(
            "existing staged3_contract alias conflicts "
            "with canonical schema"
        )

    adapted["stagea_contract"] = (
        canonical_stagea_contract
    )
    adapted_stageb_mutable["staged3_contract"] = (
        canonical_staged3_contract
    )

    if adapted["stagea_contract"] != (
        resolved["stagea_contract"]
    ):
        raise AssertionError(
            "Stage-A compatibility alias is not exact"
        )
    if adapted["upstream_immutable"][
        "staged3_contract"
    ] != resolved["staged3_contract"]:
        raise AssertionError(
            "Stage-D.3 compatibility alias is not exact"
        )
    return adapted


def schema_audit(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = resolve_canonical_stagec_schema(payload)
    adapted = adapt_stagec_schema_for_resume1(payload)
    return {
        "canonical_stagea_contract_path":
            resolved["canonical_stagea_contract_path"],
        "canonical_staged3_contract_path":
            resolved["canonical_staged3_contract_path"],
        "resume1_stagea_contract_path":
            "stagea_contract",
        "resume1_staged3_contract_path":
            "upstream_immutable.staged3_contract",
        "stagea_contract_alias_exact": (
            adapted["stagea_contract"]
            == resolved["stagea_contract"]
        ),
        "staged3_contract_alias_exact": (
            adapted["upstream_immutable"][
                "staged3_contract"
            ]
            == resolved["staged3_contract"]
        ),
        "source_payload_unchanged": (
            payload
            == resolved["stagec_layer"]
        ),
        "correction_scope":
            "schema compatibility aliases only",
    }


@contextlib.contextmanager
def adapted_stagec_validator() -> Iterator[None]:
    """Temporarily adapt the validator used by the frozen Resume1 module."""
    original: Callable[[Path], Dict[str, Any]] = (
        resume1.stagec_topk.validate_immutable_inputs
    )

    def adapted(root: Path) -> Dict[str, Any]:
        canonical = original(root)
        return adapt_stagec_schema_for_resume1(canonical)

    resume1.stagec_topk.validate_immutable_inputs = adapted
    try:
        yield
    finally:
        resume1.stagec_topk.validate_immutable_inputs = original


def run_resume2(
    *,
    root: Path,
    spec: Optional[Any] = None,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    failed = validate_resume1_failure(repository_root)

    canonical_payload = (
        stagec_topk.validate_immutable_inputs(
            repository_root
        )
    )
    audit = schema_audit(canonical_payload)
    if not (
        audit["stagea_contract_alias_exact"]
        and audit["staged3_contract_alias_exact"]
        and audit["source_payload_unchanged"]
    ):
        raise Resume2SchemaError(
            "Resume2 schema audit failed"
        )

    before = (
        resume1.stagec_topk.validate_immutable_inputs
    )
    with adapted_stagec_validator():
        result = resume1.run_resume1(
            root=repository_root,
            spec=spec,
        )
    after = (
        resume1.stagec_topk.validate_immutable_inputs
    )
    if after is not before:
        raise Resume2SchemaError(
            "Stage-C validator was not restored"
        )

    result = copy.deepcopy(result)
    result["phase"] = PHASE
    result["phase_id"] = PHASE_ID
    result["schema"] = (
        "phase314b_r257_staged_resume2_worker_result_v1"
    )
    result["resume2_schema_correction"] = {
        **audit,
        "resume1_implementation_commit":
            RESUME1_IMPLEMENTATION_COMMIT,
        "resume1_blocked_sha256":
            EXPECTED_RESUME1_BLOCKED_SHA256,
        "resume1_test_gate_sha256":
            EXPECTED_RESUME1_TEST_GATE_SHA256,
        "validator_restored": True,
        "resume1_files_modified": False,
        "schema_aliases_installed_temporarily": True,
    }
    result["resume1_failed_attempt"] = failed
    result["resume2_correction_applied"] = True
    result["resume1_files_modified"] = False
    result["resume1_blocked_modified"] = False
    result["resume1_test_gate_modified"] = False

    contract = result.get("calibration_contract")
    if not isinstance(contract, MutableMapping):
        raise Resume2SchemaError(
            "Resume1 result calibration contract is missing"
        )
    contract = copy.deepcopy(dict(contract))
    contract["resume2_schema_correction"] = (
        result["resume2_schema_correction"]
    )
    contract["correction_scope"] = (
        "Resume1 immutable-schema compatibility only; "
        "all Resume1 control, candidate, selection and "
        "classification logic unchanged"
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
        "resume2_schema_correction":
            result["resume2_schema_correction"],
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
        "schema_correction_exact": (
            left["resume2_schema_correction"]
            == right["resume2_schema_correction"]
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
