"""Stage-C Resume1 portable-GPU execution recovery.

The original r2.5.9 Stage C was correctly preregistered as a new
objective-train-only fold-resolved attribution run.  Its first execution
stopped before science because Stage C compared the complete Stage-A RTX-3090
environment payload byte-for-byte with the current RTX-4080-SUPER payload.
That equality included hardware observations which the already-frozen
``probe_portable_environment`` contract deliberately treats as provenance,
not as scientific admission criteria.

This add-only recovery preserves the original Stage-C scientific kernel and
read-only stitch instrumentation.  It changes only the execution adapter:

* software, deterministic runtime, portable compatibility SHA, and required
  CUDA-operation dry run remain exact;
* GPU model, compute capability, SM count, memory size, hostname, driver
  observation, and complete environment SHA are recorded but are not required
  to equal Stage A;
* Stage-A aggregate replay is checked by exact discrete decisions/counts plus
  preregistered cross-device numerical tolerances;
* two independent cold workers on the current GPU must have byte-exact
  scientific projections;
* durable write-ahead evidence is preserved before controller decoration.

No selection holdout, frozen probe, formal training, reverse sampling, IDM,
candidate execution, DeformableRavens execution, Phase4, or CPS is permitted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ccda_phase3 import phase314b_r259_stagec_fold_resolved_tail_attribution as base

PHASE = "Phase3.14b-r2.5.9 Stage C Resume1"
SCHEMA = "phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery_v1"
PROBE_SCHEMA = SCHEMA + "_environment_probe_v1"
WORKER_SCHEMA = SCHEMA + "_science_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEC_IMPLEMENTATION_COMMIT = "c8e99921b746e9d5c4758ccc4d451765d90fc51c"
BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT = "042ab9b4a3187d44d85eb61dbc438ccb5cef266f"
BASE_STAGEC_PARENT = "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5"
EXPECTED_REMOTE = BASE_STAGEC_PARENT
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_STAGEC_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage C: add fold-resolved tail attribution"
)
BASE_STAGEC_BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage C blocked evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage C Resume1: restore portable GPU compatibility"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage C Resume1 fold-resolved attribution evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage C Resume1 blocked evidence"
)

BASE_STAGEC_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagec_fold_resolved_tail_attribution.py"),
    ("A", "scripts/phase3_14b_r259_stagec_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagec_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagec_fold_resolved_tail_attribution.py"),
)
BASE_STAGEC_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r259_stagec_fold_resolved_tail_attribution.py": (
        "a8a96337c836270b32a4711b4e1ea6f326b08afd185743b36e0511c7e03e6e71"
    ),
    "scripts/phase3_14b_r259_stagec_worker.py": (
        "b9748cac94bc0eda2e6d38ec78e5657dbb0cb67d5bdd4fb992e65b855d9dfc61"
    ),
    "scripts/phase3_14b_r259_stagec_execute.py": (
        "a397fee1f76b64bffd03cff773ee903612612a2c91f490cfab170a9fd19c6733"
    ),
    "tests/test_phase3_14b_r259_stagec_fold_resolved_tail_attribution.py": (
        "492c5835195ab6e442eb2c31ebe67531486c7389f8251c1ce8be5dd280c9a2a6"
    ),
}
BASE_STAGEC_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stagec_fold_resolved_tail_attribution_blocked_summary.json"
)
BASE_STAGEC_BLOCKED_REPORT_SHA256 = (
    "6d0d455c328c1a9f9ef7dded701dbac13d76a259a8c6da47c1cfd2a235e2e27d"
)
BASE_STAGEC_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", BASE_STAGEC_BLOCKED_REPORT),
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r259_stagec_resume1_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagec_resume1_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r259_stagec_resume1_portable_gpu_compatibility_recovery.py",
    ),
)

PROBE_EVIDENCE = "reports/phase3_14b_r259_stagec_resume1_environment_probe_evidence.json"
WORKER_A_EVIDENCE = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_worker_a_evidence.json"
)
WORKER_B_EVIDENCE = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_worker_b_evidence.json"
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_blocked_summary.json"
)

FALSE_BOUNDARIES = base.FALSE_BOUNDARIES
MIN_TOTAL_MEMORY_BYTES = 15 * 1024**3
MIN_COMPUTE_CAPABILITY_MAJOR = 8


@dataclass(frozen=True)
class PortableAggregateSpec:
    """Frozen cross-device tolerance for aggregate scalar evidence only."""

    rtol: float = 5.0e-3
    atol: float = 1.0e-6
    maximum_difference_records: int = 256

    def validate(self) -> None:
        if not math.isfinite(self.rtol) or self.rtol <= 0.0 or self.rtol > 5.0e-3:
            raise ValueError("portable aggregate rtol changed")
        if not math.isfinite(self.atol) or self.atol <= 0.0 or self.atol > 1.0e-6:
            raise ValueError("portable aggregate atol changed")
        if self.maximum_difference_records != 256:
            raise ValueError("portable difference-record limit changed")


class StageCResume1Error(RuntimeError):
    """Fail-closed Stage-C Resume1 error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageCResume1Error("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    """Durable write-once file creation with file and directory fsync."""
    target = Path(path)
    if target.exists():
        raise StageCResume1Error("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        directory_fd = os.open(str(target.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageCResume1Error("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageCResume1Error("{} worktree is dirty".format(label))


def validate_original_blocked_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise StageCResume1Error("original Stage-C blocked report is missing")
    if sha256_file(report_path) != BASE_STAGEC_BLOCKED_REPORT_SHA256:
        raise StageCResume1Error("original Stage-C blocked report SHA changed")
    payload = load_json(report_path)
    expected = {
        "phase": base.PHASE,
        "schema": base.BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagec_execution_contract_failed",
        "required_next_path": (
            "DESIGN_ADD_ONLY_R259_STAGEC_EXECUTION_RECOVERY_BEFORE_SCIENCE_IF_"
            "WORKER_NOT_STARTED"
        ),
        "primary_failure_locus": "stagec_execution_contract",
        "durable_probe_evidence_available": False,
        "durable_worker_evidence_available": False,
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageCResume1Error("original Stage-C blocked field changed: {}".format(key))
    if "Stage-C environment differs from Stage-A environment" not in str(
        payload.get("error_message", "")
    ):
        raise StageCResume1Error("original Stage-C failure message changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCResume1Error("original Stage-C blocked report crossed a boundary")
    return payload


def validate_repository(
    root: Path,
    implementation_commit: str,
    *,
    allow_promoted_evidence: bool = False,
) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise StageCResume1Error("Resume1 implementation commit is invalid")
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageCResume1Error("Stage-C Resume1 requires Experiment1")
    if _git(repo, "rev-parse", "HEAD") != implementation_commit:
        raise StageCResume1Error("Resume1 implementation HEAD changed")
    if _git(repo, "rev-parse", "HEAD^") != BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT:
        raise StageCResume1Error("Resume1 implementation parent changed")
    if _git(repo, "rev-parse", BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT + "^") != (
        BASE_STAGEC_IMPLEMENTATION_COMMIT
    ):
        raise StageCResume1Error("original Stage-C blocked parent changed")
    if _git(repo, "rev-parse", BASE_STAGEC_IMPLEMENTATION_COMMIT + "^") != BASE_STAGEC_PARENT:
        raise StageCResume1Error("original Stage-C implementation parent changed")
    subjects = (
        (BASE_STAGEC_IMPLEMENTATION_COMMIT, BASE_STAGEC_IMPLEMENTATION_SUBJECT),
        (BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT, BASE_STAGEC_BLOCKED_EVIDENCE_SUBJECT),
        (implementation_commit, IMPLEMENTATION_SUBJECT),
    )
    for commit, expected in subjects:
        if _git(repo, "show", "-s", "--format=%s", commit) != expected:
            raise StageCResume1Error("commit subject changed: {}".format(commit))
    if commit_name_status(repo, BASE_STAGEC_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEC_IMPLEMENTATION_PATHS)
    ):
        raise StageCResume1Error("original Stage-C implementation paths changed")
    if commit_name_status(repo, BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_STAGEC_BLOCKED_PATHS)
    ):
        raise StageCResume1Error("original Stage-C blocked paths changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageCResume1Error("Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageCResume1Error("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageCResume1Error("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageCResume1Error("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-C Resume1")

    for relative, expected_sha in BASE_STAGEC_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageCResume1Error("original Stage-C source changed: {}".format(relative))
        committed = _git_bytes(
            repo, "show", "{}:{}".format(BASE_STAGEC_IMPLEMENTATION_COMMIT, relative)
        )
        if committed != path.read_bytes():
            raise StageCResume1Error(
                "original Stage-C source differs from committed blob: {}".format(relative)
            )
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageCResume1Error("Resume1 source missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageCResume1Error(
                "Resume1 source differs from committed blob: {}".format(relative)
            )

    blocked_path = repo / BASE_STAGEC_BLOCKED_REPORT
    committed_blocked = _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT, BASE_STAGEC_BLOCKED_REPORT),
    )
    if committed_blocked != blocked_path.read_bytes():
        raise StageCResume1Error("original Stage-C blocked report differs from committed blob")
    validate_original_blocked_report(blocked_path)

    # Re-run the original immutable Stage-A/Stage-B report validators without
    # asking them to validate the current HEAD.  Every source report is also
    # bound by file SHA and by its committed blob.
    immutable_reports = (
        (
            base.STAGEB_RESUME1_REPORT,
            base.EXPECTED_STAGEB_RESUME1_REPORT_SHA256,
            base.BASE_STAGEB_RESUME1_EVIDENCE_COMMIT,
        ),
        (base.STAGEA_PROBE, base.EXPECTED_STAGEA_PROBE_SHA256, base.BASE_STAGEA_EVIDENCE_COMMIT),
        (base.STAGEA_WORKER, base.EXPECTED_STAGEA_WORKER_SHA256, base.BASE_STAGEA_EVIDENCE_COMMIT),
        (base.STAGEA_SUMMARY, base.EXPECTED_STAGEA_SUMMARY_SHA256, base.BASE_STAGEA_EVIDENCE_COMMIT),
    )
    for relative, expected_sha, commit in immutable_reports:
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageCResume1Error("immutable source report changed: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(commit, relative))
        if committed != path.read_bytes():
            raise StageCResume1Error(
                "immutable report differs from committed blob: {}".format(relative)
            )

    base.validate_stageb_resume1_report(load_json(repo / base.STAGEB_RESUME1_REPORT))
    base.stagea.validate_probe_evidence(load_json(repo / base.STAGEA_PROBE))
    stagea_worker = load_json(repo / base.STAGEA_WORKER)
    base.stagea.validate_worker_evidence(stagea_worker)
    if stagea_worker.get("worker_result_sha256") != base.EXPECTED_STAGEA_WORKER_RESULT_SHA256:
        raise StageCResume1Error("Stage-A worker result SHA changed")
    stagea_summary = load_json(repo / base.STAGEA_SUMMARY)
    base.stagea.validate_summary(stagea_summary)
    if stagea_summary.get("summary_sha256") != base.EXPECTED_STAGEA_SUMMARY_SELF_SHA256:
        raise StageCResume1Error("Stage-A summary self-hash changed")

    outputs = (PROBE_EVIDENCE, WORKER_A_EVIDENCE, WORKER_B_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT)
    for relative in outputs:
        if (repo / relative).exists() and not allow_promoted_evidence:
            raise StageCResume1Error("Resume1 output already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "original_stagec_implementation_commit": BASE_STAGEC_IMPLEMENTATION_COMMIT,
        "original_stagec_blocked_evidence_commit": BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT,
        "original_stagec_blocked_report_sha256": BASE_STAGEC_BLOCKED_REPORT_SHA256,
        "stagea_worker_result_sha256": base.EXPECTED_STAGEA_WORKER_RESULT_SHA256,
    }


def extract_process_id(payload: Mapping[str, Any], label: str) -> int:
    value = payload.get("process_id")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StageCResume1Error("{} process_id is invalid".format(label))
    return value


def _hardware_observation(environment: Mapping[str, Any]) -> Mapping[str, Any]:
    value = environment.get("hardware_observation")
    if not isinstance(value, Mapping):
        raise StageCResume1Error("hardware observation is missing")
    return value


def portable_environment_audit(
    current: Mapping[str, Any], reference: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate portable compatibility while treating hardware as provenance."""
    # This frozen validator deliberately has no GPU-name/UUID/CC/memory equality
    # requirement and includes the required-operation dry run.
    base.stagea.resume2a.validate_probe_payload(
        {
            "mode": "environment_probe",
            "execution_verdict": "PASS",
            "environment": current,
            "environment_sha256": base.stagea.resume2a.sha256_bytes(
                base.stagea.resume2a.stable_json_bytes(current)
            ),
        }
    )
    base.stagea.resume2a.validate_probe_payload(
        {
            "mode": "environment_probe",
            "execution_verdict": "PASS",
            "environment": reference,
            "environment_sha256": base.stagea.resume2a.sha256_bytes(
                base.stagea.resume2a.stable_json_bytes(reference)
            ),
        }
    )
    current_compatibility = current.get("compatibility")
    reference_compatibility = reference.get("compatibility")
    if not isinstance(current_compatibility, Mapping) or not isinstance(
        reference_compatibility, Mapping
    ):
        raise StageCResume1Error("portable compatibility payload missing")
    current_compatibility_sha = str(current.get("compatibility_sha256"))
    reference_compatibility_sha = str(reference.get("compatibility_sha256"))
    if current_compatibility_sha != reference_compatibility_sha:
        raise StageCResume1Error("portable software/operation compatibility differs from Stage A")
    current_hw = _hardware_observation(current)
    reference_hw = _hardware_observation(reference)
    capability = current_hw.get("compute_capability")
    if not isinstance(capability, Sequence) or len(capability) != 2:
        raise StageCResume1Error("current compute capability is invalid")
    if int(capability[0]) < MIN_COMPUTE_CAPABILITY_MAJOR:
        raise StageCResume1Error("current GPU compute capability is below the frozen minimum")
    total_memory = int(current_hw.get("total_memory_bytes", 0))
    if total_memory < MIN_TOTAL_MEMORY_BYTES:
        raise StageCResume1Error("current GPU memory is below the frozen Stage-C minimum")
    if int(current_hw.get("cuda_device_count", 0)) != 1:
        raise StageCResume1Error("Stage-C Resume1 requires exactly one visible CUDA device")
    hardware_keys = (
        "torch_device_name",
        "compute_capability",
        "total_memory_bytes",
        "multi_processor_count",
        "cudnn_version",
    )
    differences = {
        key: {"stagea": reference_hw.get(key), "current": current_hw.get(key)}
        for key in hardware_keys
        if reference_hw.get(key) != current_hw.get(key)
    }
    return {
        "portable_contract_passed": True,
        "compatibility_sha256": current_compatibility_sha,
        "stagea_compatibility_sha256": reference_compatibility_sha,
        "software_and_required_operation_exact": True,
        "hardware_identity_required": False,
        "hardware_identity_exact": not bool(differences),
        "hardware_differences": differences,
        "minimum_total_memory_bytes": MIN_TOTAL_MEMORY_BYTES,
        "minimum_compute_capability_major": MIN_COMPUTE_CAPABILITY_MAJOR,
        "current_required_operation_pass": bool(
            current.get("required_operation_dry_run", {}).get("pass")
        ),
        "stagea_required_operation_pass": bool(
            reference.get("required_operation_dry_run", {}).get("pass")
        ),
    }


def make_probe_evidence(root: Path) -> Mapping[str, Any]:
    source = dict(base.stagea.resume2a.environment_probe_payload(Path(root).resolve()))
    process_id = int(source["process_id"])
    environment = base.stagea.resume2a.validate_probe_payload(source)
    stagea_probe = load_json(Path(root).resolve() / base.STAGEA_PROBE)
    stagea_environment = base.stagea.validate_probe_evidence(stagea_probe)
    audit = portable_environment_audit(environment, stagea_environment)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": process_id,
        "source_probe_schema": source.get("schema"),
        "environment": copy.deepcopy(dict(environment)),
        "environment_sha256": str(source["environment_sha256"]),
        "source_probe_payload_sha256": sha256_bytes(compact_json_bytes(source)),
        "stagea_environment_file_sha256": base.EXPECTED_STAGEA_PROBE_SHA256,
        "stagea_environment_sha256": stagea_probe.get("environment_sha256"),
        "stagea_environment_exact": compact_json_bytes(environment)
        == compact_json_bytes(stagea_environment),
        "portable_environment_audit": audit,
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    result["probe_evidence_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_probe_evidence(result)
    return result


def validate_probe_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != PROBE_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageCResume1Error("Resume1 probe schema/verdict changed")
    extract_process_id(payload, "probe")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise StageCResume1Error("Resume1 probe environment missing")
    expected_environment_sha = base.stagea.resume2a.sha256_bytes(
        base.stagea.resume2a.stable_json_bytes(environment)
    )
    if payload.get("environment_sha256") != expected_environment_sha:
        raise StageCResume1Error("Resume1 environment SHA is invalid")
    audit = payload.get("portable_environment_audit")
    if not isinstance(audit, Mapping) or audit.get("portable_contract_passed") is not True:
        raise StageCResume1Error("portable environment contract did not pass")
    if audit.get("software_and_required_operation_exact") is not True:
        raise StageCResume1Error("portable software/operation contract changed")
    if audit.get("hardware_identity_required") is not False:
        raise StageCResume1Error("hardware equality was reintroduced")
    if payload.get("stagea_environment_file_sha256") != base.EXPECTED_STAGEA_PROBE_SHA256:
        raise StageCResume1Error("Stage-A environment provenance changed")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageCResume1Error("Resume1 probe is not durable")
    if payload.get("write_ahead_location_outside_git_worktree") is not True:
        raise StageCResume1Error("Resume1 probe write-ahead location changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "probe_evidence_sha256"}
        )
    )
    if payload.get("probe_evidence_sha256") != expected:
        raise StageCResume1Error("Resume1 probe self-hash changed")
    return environment


OUTPUT_SHA_TOKENS = (
    "prediction",
    "candidate",
    "direction",
    "model",
    "optimizer",
    "loss",
    "gradient",
    "risk_probability",
    "callback",
    "outer_outputs",
)
EXACT_SHA_TOKENS = (
    "contract",
    "population",
    "assignment",
    "feature",
    "input",
    "source_exposure",
    "selected_scale",
    "mask",
)


def _semanticize_portable(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.endswith("_sha256"):
                lower = name.lower()
                if any(token in lower for token in EXACT_SHA_TOKENS):
                    output[name] = _semanticize_portable(item)
                elif any(token in lower for token in OUTPUT_SHA_TOKENS):
                    continue
                else:
                    # Unknown output hashes are provenance-only across devices.
                    continue
            else:
                output[name] = _semanticize_portable(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_semanticize_portable(item) for item in value]
    return value


def _numeric_tree_compare(
    current: Any,
    reference: Any,
    *,
    spec: PortableAggregateSpec,
    path: str = "$",
    differences: Optional[List[Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    spec.validate()
    records: List[Mapping[str, Any]] = [] if differences is None else differences
    start = len(records)

    def add(kind: str, here: str, left: Any, right: Any) -> None:
        if len(records) < spec.maximum_difference_records:
            records.append(
                {"path": here, "kind": kind, "current": left, "stagea": right}
            )

    def walk(left: Any, right: Any, here: str) -> None:
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                add("type", here, type(left).__name__, type(right).__name__)
                return
            if set(left) != set(right):
                add("keys", here, sorted(left), sorted(right))
                return
            for key in sorted(left):
                walk(left[key], right[key], here + "." + str(key))
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                add("type", here, type(left).__name__, type(right).__name__)
                return
            if len(left) != len(right):
                add("length", here, len(left), len(right))
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(left_item, right_item, "{}[{}]".format(here, index))
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if left is not right:
                add("boolean", here, left, right)
            return
        if isinstance(left, int) or isinstance(right, int):
            if not isinstance(left, int) or not isinstance(right, int) or left != right:
                add("integer", here, left, right)
            return
        if isinstance(left, float) or isinstance(right, float):
            try:
                a = float(left)
                b = float(right)
            except (TypeError, ValueError):
                add("numeric_type", here, left, right)
                return
            if not math.isfinite(a) or not math.isfinite(b):
                if a != b:
                    add("nonfinite", here, a, b)
                return
            tolerance = max(spec.atol, spec.rtol * max(abs(a), abs(b)))
            if abs(a - b) > tolerance:
                add("numeric", here, a, b)
            return
        if left != right:
            add("exact", here, left, right)

    walk(current, reference, path)
    new = records[start:]
    return {
        "pass": not bool(new),
        "difference_count": len(new),
        "differences": copy.deepcopy(new),
        "rtol": spec.rtol,
        "atol": spec.atol,
    }


def _selection_projection(values: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    result = []
    for item in values:
        result.append(
            {
                "outer_fold": int(item["outer_fold"]),
                "selected_policy": copy.deepcopy(item.get("selected_policy")),
                "selected_policy_id": item.get("selected_policy_id"),
                "selected_policy_inner_eligible": bool(
                    item.get("selected_policy_inner_eligible")
                ),
                "diagnostic_fallback_used": bool(item.get("diagnostic_fallback_used")),
            }
        )
    return result


def _decision_projection(value: Any) -> Any:
    """Keep strings, booleans, integer counts, policy mappings and gate decisions."""
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lower = name.lower()
            if isinstance(item, bool) or isinstance(item, int) or isinstance(item, str) or item is None:
                output[name] = item
            elif isinstance(item, Mapping) and (
                "policy" in lower or "gate" in lower or "selection" in lower
            ):
                output[name] = _decision_projection(item)
            elif isinstance(item, (list, tuple)) and (
                "policy" in lower or "fold" in lower or "eligible" in lower
            ):
                output[name] = _decision_projection(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_decision_projection(item) for item in value]
    return value


def portable_compare_stagea_reproduction(
    kernel_payload: Mapping[str, Any], stagea_worker: Mapping[str, Any],
    *, spec: Optional[PortableAggregateSpec] = None,
) -> Mapping[str, Any]:
    active = PortableAggregateSpec() if spec is None else spec
    active.validate()
    exact_fields = (
        "locked_scientific_base",
        "stagex_spec",
        "policy_population",
        "population",
        "procedure_gate_totals",
        "execution_counts",
    )
    exact_checks: Dict[str, Any] = {}
    exact_pass = True
    for field in exact_fields:
        current_sha = sha256_bytes(compact_json_bytes(kernel_payload[field]))
        reference_sha = sha256_bytes(compact_json_bytes(stagea_worker[field]))
        passed = current_sha == reference_sha
        exact_checks[field] = {
            "current_sha256": current_sha,
            "stagea_sha256": reference_sha,
            "exact": passed,
        }
        exact_pass = exact_pass and passed

    current_selections = _selection_projection(kernel_payload["outer_fold_selections"])
    reference_selections = _selection_projection(stagea_worker["outer_fold_selections"])
    selection_exact = compact_json_bytes(current_selections) == compact_json_bytes(
        reference_selections
    )
    decision_pairs = {
        "inner_selection_modal_policy_diagnostic": (
            kernel_payload["inner_selection_modal_policy_diagnostic"],
            stagea_worker["inner_selection_modal_policy_diagnostic"],
        ),
        "full_objective_oof_fixed_policy_selection": (
            kernel_payload["full_objective_oof_fixed_policy_selection"],
            stagea_worker["full_objective_oof_fixed_policy_selection"],
        ),
    }
    decision_checks: Dict[str, bool] = {
        "outer_fold_selections": selection_exact,
        "scientific_status": kernel_payload.get("scientific_status")
        == stagea_worker.get("scientific_status"),
        "primary_failure_locus": kernel_payload.get("primary_failure_locus")
        == stagea_worker.get("scientific_kernel", {}).get("kernel_primary_failure_locus"),
    }
    for field, (current, reference) in decision_pairs.items():
        decision_checks[field] = compact_json_bytes(_decision_projection(current)) == (
            compact_json_bytes(_decision_projection(reference))
        )

    numeric_fields = (
        "outer_crossfit_fold_selected_policy_records",
        "outer_crossfit_baseline_records",
        "full_objective_oof_fixed_policy_records",
    )
    numeric_checks: Dict[str, Any] = {}
    numeric_pass = True
    for field in numeric_fields:
        comparison = _numeric_tree_compare(
            _semanticize_portable(kernel_payload[field]),
            _semanticize_portable(stagea_worker[field]),
            spec=active,
            path="$.{}".format(field),
        )
        numeric_checks[field] = comparison
        numeric_pass = numeric_pass and bool(comparison["pass"])

    historical_bitwise = {
        field: sha256_bytes(compact_json_bytes(kernel_payload[field]))
        == sha256_bytes(compact_json_bytes(stagea_worker[field]))
        for field in (
            "outer_fold_selections",
            "inner_selection_modal_policy_diagnostic",
            *numeric_fields,
            "full_objective_oof_fixed_policy_selection",
        )
    }
    passed = exact_pass and all(decision_checks.values()) and numeric_pass
    result = {
        # Temporary compatibility with the original Stage-C validator.  The
        # Resume1 worker replaces this object before persistence.
        "all_exact": passed,
        "portable_contract_passed": passed,
        "contract_mode": "cross_gpu_functional_equivalence",
        "aggregate_tolerance": asdict(active),
        "exact_field_checks": exact_checks,
        "exact_decision_checks": decision_checks,
        "numerical_checks": numeric_checks,
        "historical_bitwise_identity": historical_bitwise,
        "historical_all_bitwise_exact": all(historical_bitwise.values()),
        "hardware_identity_required": False,
    }
    if not passed:
        failed_exact = sorted(
            key for key, value in exact_checks.items() if value["exact"] is not True
        )
        failed_decisions = sorted(
            key for key, value in decision_checks.items() if value is not True
        )
        failed_numeric = sorted(
            key for key, value in numeric_checks.items() if value["pass"] is not True
        )
        raise StageCResume1Error(
            "portable Stage-A aggregate equivalence failed: exact={} decisions={} numeric={}".format(
                failed_exact, failed_decisions, failed_numeric
            )
        )
    return result


def make_worker_evidence(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
    worker_slot: str,
) -> Mapping[str, Any]:
    validate_probe_evidence(probe_payload)
    if worker_slot not in ("A", "B"):
        raise StageCResume1Error("worker slot changed")
    original_probe_validator = base.validate_probe_evidence
    original_compare = base.compare_stagea_reproduction
    base.validate_probe_evidence = validate_probe_evidence
    base.compare_stagea_reproduction = portable_compare_stagea_reproduction
    try:
        original = dict(
            base.make_worker_evidence(
                root=Path(root).resolve(),
                probe_payload=probe_payload,
                repository_head=str(repository_head),
                stageu_contract=stageu_contract,
            )
        )
    finally:
        base.validate_probe_evidence = original_probe_validator
        base.compare_stagea_reproduction = original_compare

    temporary_identity = original.pop("stagea_reproduction_identity")
    original.pop("worker_result_sha256", None)
    if temporary_identity.get("portable_contract_passed") is not True:
        raise StageCResume1Error("portable Stage-A identity did not pass")
    original.update(
        {
            "phase": PHASE,
            "schema": WORKER_SCHEMA,
            "worker_slot": worker_slot,
            "portable_stagea_reproduction_identity": copy.deepcopy(temporary_identity),
            "portable_gpu_contract": {
                "exact_hardware_identity_required": False,
                "stagea_gpu_identity_used_as_provenance_only": True,
                "software_and_required_operation_exact": True,
                "same_device_double_worker_identity_required": True,
                "historical_cross_device_bitwise_output_sha_required": False,
                "exact_discrete_decisions_and_counts_required": True,
                "continuous_aggregate_tolerance": asdict(PortableAggregateSpec()),
            },
        }
    )
    prereg = dict(original.get("preregistration_contract", {}))
    prereg.update(
        {
            "stagec_original_science_worker_started": False,
            "stagec_resume1_new_science_stage": True,
            "portable_gpu_execution_recovery_only": True,
            "hardware_identity_gate_relaxed": True,
            "stagea_global_result_must_reproduce_exactly": False,
            "stagea_global_result_must_pass_portable_functional_contract": True,
            "scientific_policy_or_threshold_changed": False,
        }
    )
    original["preregistration_contract"] = prereg
    original["worker_result_sha256"] = sha256_bytes(stable_json_bytes(original))
    validate_worker_evidence(original)
    return original


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageCResume1Error("Resume1 worker schema/verdict changed")
    if payload.get("worker_slot") not in ("A", "B"):
        raise StageCResume1Error("Resume1 worker slot changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageCResume1Error("Stage-C attribution cannot be READY")
    portable = payload.get("portable_stagea_reproduction_identity")
    if not isinstance(portable, Mapping) or portable.get("portable_contract_passed") is not True:
        raise StageCResume1Error("portable Stage-A reproduction identity failed")
    if portable.get("hardware_identity_required") is not False:
        raise StageCResume1Error("worker reintroduced hardware equality")
    if payload.get("execution_counts") != base.EXPECTED_EXECUTION_COUNTS:
        raise StageCResume1Error("scientific execution counts changed")
    instrumentation = payload.get("instrumentation_identity")
    if not isinstance(instrumentation, Mapping):
        raise StageCResume1Error("instrumentation identity missing")
    if instrumentation.get("stitch_call_count") != 1:
        raise StageCResume1Error("stitch call count changed")
    if instrumentation.get("outer_outputs_unchanged") is not True:
        raise StageCResume1Error("instrumentation mutated outer outputs")
    diagnostics = payload.get("fold_resolved_attribution")
    if not isinstance(diagnostics, Mapping):
        raise StageCResume1Error("fold diagnostics missing")
    if diagnostics.get("fold_timestep_record_count") != 18:
        raise StageCResume1Error("fold/timestep record count changed")
    if payload.get("selected_configuration") is not None:
        raise StageCResume1Error("Resume1 selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageCResume1Error("Resume1 retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageCResume1Error("Resume1 re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageCResume1Error("Resume1 cumulative holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageCResume1Error("Resume1 accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageCResume1Error("Resume1 authorized rerun")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageCResume1Error("Resume1 worker is not durable")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCResume1Error("Resume1 worker crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected:
        raise StageCResume1Error("Resume1 worker self-hash changed")


def worker_scientific_projection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_worker_evidence(payload)
    result = copy.deepcopy(dict(payload))
    for key in ("process_id", "worker_result_sha256", "worker_slot"):
        result.pop(key, None)
    return result


def compare_same_device_workers(
    worker_a: Mapping[str, Any], worker_b: Mapping[str, Any]
) -> Mapping[str, Any]:
    projection_a = worker_scientific_projection(worker_a)
    projection_b = worker_scientific_projection(worker_b)
    bytes_a = compact_json_bytes(projection_a)
    bytes_b = compact_json_bytes(projection_b)
    exact = bytes_a == bytes_b
    result = {
        "same_device_scientific_projection_byte_exact": exact,
        "worker_a_projection_sha256": sha256_bytes(bytes_a),
        "worker_b_projection_sha256": sha256_bytes(bytes_b),
        "worker_a_result_sha256": worker_a["worker_result_sha256"],
        "worker_b_result_sha256": worker_b["worker_result_sha256"],
    }
    if not exact:
        raise StageCResume1Error("current-GPU cold workers are not byte-exact")
    return result


def promote_write_ahead(source: Path, destination: Path, validator: Any) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise StageCResume1Error("write-ahead source missing: {}".format(source_path))
    validator(load_json(source_path))
    atomic_write_once(destination_path, source_path.read_bytes())
    if destination_path.read_bytes() != source_path.read_bytes():
        raise StageCResume1Error("write-ahead promotion is not byte-exact")


def build_summary(
    *,
    repository: Mapping[str, Any],
    probe_path: Path,
    worker_a_path: Path,
    worker_b_path: Path,
) -> Mapping[str, Any]:
    probe = load_json(probe_path)
    worker_a = load_json(worker_a_path)
    worker_b = load_json(worker_b_path)
    validate_probe_evidence(probe)
    validate_worker_evidence(worker_a)
    validate_worker_evidence(worker_b)
    probe_pid = extract_process_id(probe, "probe")
    worker_a_pid = extract_process_id(worker_a, "worker A")
    worker_b_pid = extract_process_id(worker_b, "worker B")
    if len({probe_pid, worker_a_pid, worker_b_pid}) != 3:
        raise StageCResume1Error("probe and worker PIDs are not distinct")
    identity = compare_same_device_workers(worker_a, worker_b)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker_a["scientific_status"],
        "root_cause": worker_a["root_cause"],
        "required_next_path": worker_a["required_next_path"],
        "primary_failure_locus": worker_a["primary_failure_locus"],
        "repository": copy.deepcopy(dict(repository)),
        "original_stagec_blocked_provenance": {
            "implementation_commit": BASE_STAGEC_IMPLEMENTATION_COMMIT,
            "blocked_evidence_commit": BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT,
            "blocked_report_sha256": BASE_STAGEC_BLOCKED_REPORT_SHA256,
            "original_science_worker_started": False,
            "original_fold_metrics_computed": False,
        },
        "portable_environment_contract": copy.deepcopy(
            probe["portable_environment_audit"]
        ),
        "portable_stagea_reproduction_identity": copy.deepcopy(
            worker_a["portable_stagea_reproduction_identity"]
        ),
        "same_device_double_worker_identity": identity,
        "fold_resolved_attribution": copy.deepcopy(worker_a["fold_resolved_attribution"]),
        "outer_fold_selections": copy.deepcopy(worker_a["outer_fold_selections"]),
        "inner_selection_modal_policy_diagnostic": copy.deepcopy(
            worker_a["inner_selection_modal_policy_diagnostic"]
        ),
        "global_outer_crossfit_fold_selected_policy_records": copy.deepcopy(
            worker_a["global_outer_crossfit_fold_selected_policy_records"]
        ),
        "global_outer_crossfit_baseline_records": copy.deepcopy(
            worker_a["global_outer_crossfit_baseline_records"]
        ),
        "full_objective_oof_fixed_policy_selection": copy.deepcopy(
            worker_a["full_objective_oof_fixed_policy_selection"]
        ),
        "procedure_gate_totals": copy.deepcopy(worker_a["procedure_gate_totals"]),
        "execution_counts": {
            "per_worker": copy.deepcopy(worker_a["execution_counts"]),
            "cold_science_worker_count": 2,
            "total_direction_fit_count": 2
            * int(worker_a["execution_counts"]["direction_fit_count"]),
            "total_candidate_generation_count": 2
            * int(worker_a["execution_counts"]["candidate_generation_count"]),
            "total_risk_fit_count": 2
            * int(worker_a["execution_counts"]["risk_fit_count"]),
            "total_internal_scale_attempt_count": 2
            * int(worker_a["execution_counts"]["internal_scale_attempt_count"]),
        },
        "durable_evidence_protocol": {
            "probe_evidence_path": PROBE_EVIDENCE,
            "probe_evidence_file_sha256": sha256_file(probe_path),
            "probe_evidence_self_sha256": probe["probe_evidence_sha256"],
            "worker_a_evidence_path": WORKER_A_EVIDENCE,
            "worker_a_evidence_file_sha256": sha256_file(worker_a_path),
            "worker_a_result_sha256": worker_a["worker_result_sha256"],
            "worker_b_evidence_path": WORKER_B_EVIDENCE,
            "worker_b_evidence_file_sha256": sha256_file(worker_b_path),
            "worker_b_result_sha256": worker_b["worker_result_sha256"],
            "workers_persisted_before_controller_decoration": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_evidence_promoted_byte_exact": True,
            "write_ahead_files_write_once": True,
            "controller_recomputed_worker_science": False,
        },
        "process_topology": {
            "environment_probe_process_count": 1,
            "cold_science_worker_count": 2,
            "probe_process_id": probe_pid,
            "worker_a_process_id": worker_a_pid,
            "worker_b_process_id": worker_b_pid,
            "processes_distinct": True,
            "workers_sequential": True,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["summary_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_summary(result)
    return result


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageCResume1Error("Resume1 summary schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageCResume1Error("Resume1 attribution cannot be READY")
    environment = payload.get("portable_environment_contract")
    if not isinstance(environment, Mapping) or environment.get("portable_contract_passed") is not True:
        raise StageCResume1Error("summary portable environment contract failed")
    reproduction = payload.get("portable_stagea_reproduction_identity")
    if not isinstance(reproduction, Mapping) or reproduction.get("portable_contract_passed") is not True:
        raise StageCResume1Error("summary Stage-A functional identity failed")
    worker_identity = payload.get("same_device_double_worker_identity")
    if not isinstance(worker_identity, Mapping) or worker_identity.get(
        "same_device_scientific_projection_byte_exact"
    ) is not True:
        raise StageCResume1Error("summary same-device worker identity failed")
    topology = payload.get("process_topology")
    if not isinstance(topology, Mapping) or topology.get("processes_distinct") is not True:
        raise StageCResume1Error("summary process topology changed")
    protocol = payload.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageCResume1Error("summary durable protocol missing")
    for key in (
        "workers_persisted_before_controller_decoration",
        "external_write_ahead_outside_git_worktree",
        "repository_evidence_promoted_byte_exact",
        "write_ahead_files_write_once",
    ):
        if protocol.get(key) is not True:
            raise StageCResume1Error("summary durable protocol changed: {}".format(key))
    if payload.get("selected_configuration") is not None:
        raise StageCResume1Error("summary selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageCResume1Error("summary retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageCResume1Error("summary re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageCResume1Error("summary cumulative holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageCResume1Error("summary accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageCResume1Error("summary authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCResume1Error("summary crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageCResume1Error("summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Optional[Path] = None,
    worker_a_path: Optional[Path] = None,
    worker_b_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    paths = {
        "probe": probe_path,
        "worker_a": worker_a_path,
        "worker_b": worker_b_path,
    }
    available = {
        key: bool(path is not None and Path(path).is_file())
        for key, path in paths.items()
    }
    any_worker = available["worker_a"] or available["worker_b"]
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagec_resume1_portable_gpu_recovery_failed",
        "required_next_path": (
            "FINALIZE_R259_STAGEC_RESUME1_DURABLE_WORKER_EVIDENCE_WITHOUT_SCIENCE_REEXECUTION"
            if any_worker
            else "RESTORE_R259_STAGEC_RESUME1_PORTABLE_GPU_RECOVERY_BEFORE_SCIENCE"
        ),
        "primary_failure_locus": (
            "controller_after_durable_worker_evidence"
            if any_worker
            else "portable_gpu_execution_contract"
        ),
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "durable_probe_evidence_available": available["probe"],
        "durable_worker_a_evidence_available": available["worker_a"],
        "durable_worker_b_evidence_available": available["worker_b"],
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    validators = {
        "probe": validate_probe_evidence,
        "worker_a": validate_worker_evidence,
        "worker_b": validate_worker_evidence,
    }
    for key, path in paths.items():
        if available[key]:
            path_obj = Path(path)  # type: ignore[arg-type]
            payload["{}_evidence_file_sha256".format(key)] = sha256_file(path_obj)
            try:
                value = load_json(path_obj)
                validators[key](value)
                payload["{}_evidence_validated".format(key)] = True
            except BaseException as validation_error:
                payload["{}_evidence_validated".format(key)] = False
                payload["{}_validation_error".format(key)] = str(validation_error)
    return payload
