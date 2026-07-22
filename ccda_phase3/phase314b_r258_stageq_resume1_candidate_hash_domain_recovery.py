"""Phase3.14b-r2.5.8 Stage Q Resume1 candidate hash-domain recovery.

The original Stage-Q shadow audit stopped before scientific classification
because its replay identity validator compared hashes from incompatible frozen
array-hash domains:

* Stage-E hashes: ``dtype + JSON-list(shape) + bytes``;
* Stage-O/Stage-P hashes: ``dtype + '|' + repr(tuple(shape)) + '|' + bytes``.

The candidate bytes can therefore be identical while the SHA strings differ.
Resume1 does not modify Stage-Q, Stage-E, Stage-P, or the shadow-gate science.
For each of the same six oracle cells it:

1. performs the single frozen callback-off/on integration;
2. independently validates final and per-scale candidates in both hash domains;
3. supplies that cached integration to the unchanged Stage-Q cell audit;
4. temporarily dispatches Stage-O hashing to the Stage-E domain only for the
   final float32 candidate tensor expected by the original validator;
5. replaces only copied Stage-P per-scale candidate-hash labels with the
   already-validated Stage-E-domain equivalents consumed by the original code;
6. restores every temporary function replacement in ``finally``.

No callback pair is duplicated.  No threshold, tolerance, scale bank,
reconstruction, predicate, topology, classification, or data boundary changes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


PHASE = "Phase3.14b-r2.5.8 Stage Q Resume1"
SCHEMA = "phase314b_r258_stageq_resume1_candidate_hash_domain_recovery_v1"
BLOCKED_SCHEMA = "phase314b_r258_stageq_resume1_candidate_hash_domain_recovery_blocked_v1"

BASE_STAGEQ_IMPLEMENTATION_COMMIT = "8597b2da8c5a1270887d4bd5c7bd0816770ec759"
BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT = "d328122fb299167f9bc0c453ff7d45405d6467e7"
EXPECTED_STAGEQ_IMPLEMENTATION_PARENT = "d0a06346e33e4638e98e129a052b8f42f4da92bb"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEQ_BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow_"
    "blocked_summary.json"
)
EXPECTED_STAGEQ_BLOCKED_REPORT_SHA256 = (
    "378f97da7e46aad3cb35d3eef404314974c06199c4a43ed25dfc0fab5838584c"
)
STAGEQ_SOURCE = (
    "ccda_phase3/phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow.py"
)
EXPECTED_STAGEQ_SOURCE_SHA256 = (
    "42f7577ec15f38e5222221595fa424c3276336ac383796c67e88db382df78884"
)
STAGEQ_RUNNER = "scripts/phase3_14b_r258_stageq_execute.py"
EXPECTED_STAGEQ_RUNNER_SHA256 = (
    "99c636b312ab0643601622ac7263207a877c419781e4460b5438cfba12b05a3a"
)
STAGEQ_TEST = (
    "tests/test_phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow.py"
)
EXPECTED_STAGEQ_TEST_SHA256 = (
    "67f728675f8161f11a5ad25b2ea2109ad9b4b5c97862d080a3280e4b7a4f37f8"
)

STAGEP_REPORT = (
    "reports/phase3_14b_r258_stagep_oracle_upper_segment_gate_provenance_"
    "summary.json"
)
EXPECTED_STAGEP_REPORT_SHA256 = (
    "e4a7aba0715a0bb6b8da72654f0bd55d9bda0c67e52f532620f26d6f4f68898c"
)
EXPECTED_STAGEP_SCIENTIFIC_RESULT_SHA256 = (
    "1663c33b2430c97ef3f408a3c5de1244e15ee435f02bcd47480e9b919f646031"
)

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stageq_resume1_tolerance_aligned_upper_gate_"
    "shadow_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stageq_resume1_tolerance_aligned_upper_gate_"
    "shadow_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage Q Resume1: restore candidate hash domains"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage Q Resume1 tolerance-aligned evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage Q Resume1 blocked evidence"
)
ORIGINAL_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage Q: audit tolerance-aligned upper gate"
)
ORIGINAL_BLOCKED_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage Q blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stageq_resume1_candidate_hash_domain_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r258_stageq_resume1_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stageq_resume1_candidate_hash_domain_recovery.py",
    ),
)
ORIGINAL_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGEQ_SOURCE),
    ("A", STAGEQ_RUNNER),
    ("A", STAGEQ_TEST),
)
ORIGINAL_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGEQ_BLOCKED_REPORT),
)

EXPECTED_ENV: Mapping[str, str] = {
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

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "selection_holdout_evaluated",
    "selection_holdout_used_for_fit_or_selection",
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
    "predicate_tensor_persisted",
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)

EXPECTED_PAIR_COUNT = 6
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
ORACLE_SOURCES: Tuple[str, ...] = ("raw_oracle", "projected_oracle")
EXTERNAL_MULTIPLIER = 0.25


class StageQResume1Error(RuntimeError):
    """Fail-closed Stage-Q Resume1 error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stageop_array_sha_contract(value: np.ndarray) -> str:
    """Frozen Stage-O/Stage-P/Stage-Q local array hash domain."""
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode("ascii")
        + b"|"
        + repr(tuple(array.shape)).encode("ascii")
        + b"|"
        + array.tobytes(order="C")
    )
    return sha256_bytes(payload)


def stagee_array_sha_contract(value: np.ndarray) -> str:
    """Frozen Stage-E array hash domain."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageQResume1Error(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageQResume1Error(f"{label} is not a sequence")
    return value


def _scale_key(value: float) -> str:
    return format(float(value), ".17g")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageQResume1Error(
            "git {} failed: {}".format(" ".join(args), completed.stderr.strip())
        )
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageQResume1Error(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace")
            )
        )
    return completed.stdout


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageQResume1Error("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def status_paths(root: Path) -> Tuple[str, ...]:
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    values: List[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        values.append(value)
    return tuple(sorted(values))


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageQResume1Error(
                f"{label} boundary changed: {key}={mapping.get(key)!r}"
            )


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageQResume1Error(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _validate_original_provenance(repo: Path) -> Mapping[str, Any]:
    if _git(repo, "rev-parse", f"{BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGEQ_IMPLEMENTATION_COMMIT
    ):
        raise StageQResume1Error("Stage-Q blocked provenance parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGEQ_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_STAGEQ_IMPLEMENTATION_PARENT
    ):
        raise StageQResume1Error("Stage-Q implementation parent changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGEQ_IMPLEMENTATION_COMMIT,
    ) != ORIGINAL_IMPLEMENTATION_SUBJECT:
        raise StageQResume1Error("Stage-Q implementation subject changed")
    if commit_name_status(repo, BASE_STAGEQ_IMPLEMENTATION_COMMIT) != tuple(
        sorted(ORIGINAL_IMPLEMENTATION_PATHS)
    ):
        raise StageQResume1Error("Stage-Q implementation paths changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT,
    ) != ORIGINAL_BLOCKED_SUBJECT:
        raise StageQResume1Error("Stage-Q blocked-evidence subject changed")
    if commit_name_status(repo, BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(ORIGINAL_BLOCKED_PATHS)
    ):
        raise StageQResume1Error("Stage-Q blocked-evidence paths changed")

    blocked_path = repo / STAGEQ_BLOCKED_REPORT
    if not blocked_path.is_file():
        raise StageQResume1Error("Stage-Q blocked report is missing")
    if sha256_file(blocked_path) != EXPECTED_STAGEQ_BLOCKED_REPORT_SHA256:
        raise StageQResume1Error("Stage-Q blocked report SHA changed")
    committed = _git_bytes(
        repo,
        "show",
        f"{BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT}:{STAGEQ_BLOCKED_REPORT}",
    )
    if committed != blocked_path.read_bytes():
        raise StageQResume1Error("Stage-Q blocked report differs from committed blob")
    payload = json.loads(blocked_path.read_text(encoding="utf-8"))
    if payload.get("execution_verdict") != "BLOCKED":
        raise StageQResume1Error("Stage-Q blocked verdict changed")
    if payload.get("root_cause") != (
        "phase314b_r258_stageq_shadow_alignment_execution_failed"
    ):
        raise StageQResume1Error("Stage-Q blocked root cause changed")
    if payload.get("required_next_path") != (
        "RESTORE_STAGEQ_TOLERANCE_ALIGNED_UPPER_GATE_SHADOW"
    ):
        raise StageQResume1Error("Stage-Q blocked next path changed")
    if payload.get("primary_failure_locus") != "execution_contract":
        raise StageQResume1Error("Stage-Q blocked primary locus changed")
    if payload.get("error_message") != (
        "Stage-Q replay differs from Stage-P: ['candidate_sha256']"
    ):
        raise StageQResume1Error("Stage-Q blocked error message changed")
    require_false(payload, FALSE_BOUNDARIES, "Stage-Q blocked report")
    return {
        "stage_q_implementation_commit": BASE_STAGEQ_IMPLEMENTATION_COMMIT,
        "stage_q_blocked_evidence_commit": BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT,
        "stage_q_blocked_report_sha256": EXPECTED_STAGEQ_BLOCKED_REPORT_SHA256,
        "original_failure_message": payload.get("error_message"),
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageQResume1Error("Stage-Q Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT:
        raise StageQResume1Error("Stage-Q Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageQResume1Error("Stage-Q Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageQResume1Error("Stage-Q Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageQResume1Error("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageQResume1Error("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageQResume1Error("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageQResume1Error("Stage-Q Resume1 worktree must be clean")

    source_checks = {
        STAGEQ_SOURCE: EXPECTED_STAGEQ_SOURCE_SHA256,
        STAGEQ_RUNNER: EXPECTED_STAGEQ_RUNNER_SHA256,
        STAGEQ_TEST: EXPECTED_STAGEQ_TEST_SHA256,
    }
    for relative, expected in source_checks.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageQResume1Error(f"frozen Stage-Q blob changed: {relative}")
        if _git_bytes(
            repo,
            "show",
            f"{BASE_STAGEQ_IMPLEMENTATION_COMMIT}:{relative}",
        ) != path.read_bytes():
            raise StageQResume1Error(
                f"Stage-Q path differs from implementation blob: {relative}"
            )

    stagep_path = repo / STAGEP_REPORT
    if not stagep_path.is_file() or sha256_file(stagep_path) != EXPECTED_STAGEP_REPORT_SHA256:
        raise StageQResume1Error("Stage-P report SHA changed")
    stagep_payload = json.loads(stagep_path.read_text(encoding="utf-8"))
    if stagep_payload.get("scientific_result_sha256") != (
        EXPECTED_STAGEP_SCIENTIFIC_RESULT_SHA256
    ):
        raise StageQResume1Error("Stage-P scientific-result SHA changed")

    provenance = _validate_original_provenance(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageQResume1Error(
                f"Stage-Q Resume1 output already exists: {relative}"
            )
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "stage_q_source_sha256": EXPECTED_STAGEQ_SOURCE_SHA256,
        "stage_q_runner_sha256": EXPECTED_STAGEQ_RUNNER_SHA256,
        "stage_q_test_sha256": EXPECTED_STAGEQ_TEST_SHA256,
        "stage_p_report_sha256": EXPECTED_STAGEP_REPORT_SHA256,
        "blocked_provenance": provenance,
    }


def _legacy_scale_candidate_map(
    integration: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    records = _sequence(integration.get("scale_candidates"), "scale candidates")
    output: Dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"scale candidate {index}")
        key = _scale_key(float(record.get("scale")))
        if key in output:
            raise StageQResume1Error("duplicate scale candidate")
        output[key] = record
    return output


def _candidate_tensor(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 3:
        raise StageQResume1Error(
            f"{label} is not the frozen float32 candidate tensor"
        )
    if not np.all(np.isfinite(array)):
        raise StageQResume1Error(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def _selected_scale_vector(value: Any, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (rows,) or not np.all(np.isfinite(array)):
        raise StageQResume1Error("selected-scale vector changed")
    return np.ascontiguousarray(array)


def _expected_internal_order(runtime: Mapping[str, Any]) -> Tuple[float, ...]:
    stageq = runtime["stageq"]
    return tuple(float(value) for value in stageq._expected_internal_order(runtime))


def prepare_hash_domain_recovery(
    *,
    source_id: str,
    timestep: int,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    base_cell: Mapping[str, Any],
    spec: Any,
    integration: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate both historical hash domains and build a copied adapter cell."""
    stageq = runtime["stageq"]
    stagep = runtime["stagep"]
    stagee = runtime["stagel"].stagee258
    stagef = runtime["stagef"]
    integrator_spec = runtime["integrator_spec"]
    definition = stagef.fixed_integrator_definition()
    definition.validate()

    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if control_raw.shape != direction_raw.shape:
        raise StageQResume1Error("control/oracle direction shape changed")
    if float(spec.external_multiplier) != EXTERNAL_MULTIPLIER:
        raise StageQResume1Error("Stage-Q external multiplier changed")
    proposed_direction = direction_raw * float(spec.external_multiplier)

    selected = _selected_scale_vector(
        integration.get("selected_scale"), int(control_raw.shape[0])
    )
    candidate = _candidate_tensor(integration.get("candidate"), "final candidate")
    stagee_final_sha = stagee.sha256_array(candidate)
    stagee_contract_final_sha = stagee_array_sha_contract(candidate)
    if stagee_final_sha != stagee_contract_final_sha:
        raise StageQResume1Error("runtime Stage-E candidate hash contract changed")
    stagep_final_sha = stagep.sha256_array(candidate)
    stageop_contract_final_sha = stageop_array_sha_contract(candidate)
    if stagep_final_sha != stageop_contract_final_sha:
        raise StageQResume1Error("runtime Stage-P candidate hash contract changed")
    if stagee_final_sha == stagep_final_sha:
        raise StageQResume1Error("Stage-E and Stage-P candidate hash domains collapsed")

    final_checks = {
        "source_id": str(base_cell.get("source_id")) == str(source_id),
        "timestep": int(base_cell.get("timestep")) == int(timestep),
        "selected_scale_stageop_sha": stageq.sha256_array(selected)
        == base_cell.get("final_selected_scale_sha256"),
        "integration_declared_stagee_candidate_sha": str(
            integration.get("candidate_sha256")
        )
        == stagee_final_sha,
        "stagep_final_field_is_stagee_domain": str(
            base_cell.get("final_candidate_sha256")
        )
        == stagee_final_sha,
        "callback_capture_sha": str(capture.get("events_sha256"))
        == str(base_cell.get("callback_capture_sha256")),
        "callback_result_bit_exact": capture.get("returned_result_bit_exact") is True,
    }
    if not all(final_checks.values()):
        failed = sorted(key for key, value in final_checks.items() if not value)
        raise StageQResume1Error(f"final hash-domain preflight failed: {failed}")

    attempts = [
        _mapping(value, f"callback event {index}")
        for index, value in enumerate(
            _sequence(capture.get("events"), "callback events")
        )
        if isinstance(value, Mapping) and value.get("event_type") == "scale_attempt"
    ]
    expected_order = _expected_internal_order(runtime)
    observed_order = tuple(float(item.get("attempted_scale")) for item in attempts)
    if observed_order != expected_order:
        raise StageQResume1Error("callback attempt order changed during preflight")
    base_attempts = _sequence(base_cell.get("attempt_records"), "Stage-P attempts")
    if len(base_attempts) != len(expected_order):
        raise StageQResume1Error("Stage-P attempt population changed")
    legacy_candidates = _legacy_scale_candidate_map(integration)
    bounds = stagee.segment_bounds(definition=definition, context=context)
    lower = np.asarray(bounds["lower"], dtype=np.float64)
    upper = np.asarray(bounds["upper"], dtype=np.float64)

    corrected_cell: MutableMapping[str, Any] = copy.deepcopy(dict(base_cell))
    corrected_attempts = _sequence(
        corrected_cell.get("attempt_records"), "copied Stage-P attempts"
    )
    per_scale_records: List[Mapping[str, Any]] = []
    for index, (scale, raw_base) in enumerate(zip(expected_order, base_attempts)):
        base_attempt = _mapping(raw_base, f"Stage-P attempt {index}")
        if float(base_attempt.get("internal_scale")) != float(scale):
            raise StageQResume1Error("Stage-P attempt order changed")
        raw_proposal = (
            control_raw.astype(np.float64) + float(scale) * proposed_direction
        ).astype(np.float32)
        reconstruction = stagee.reconstruct_segment_vectors(
            proposed=raw_proposal,
            control=control_raw,
            lower=lower,
            upper=upper,
            coordinate_abs_max=float(
                context["historical_geometry"].coordinate_abs_max
            ),
            epsilon=float(integrator_spec.segment_tolerance),
        )
        scale_candidate = _candidate_tensor(
            reconstruction.get("candidate"), f"candidate at scale {scale}"
        )
        stagee_candidate_sha = stagee.sha256_array(scale_candidate)
        stagep_candidate_sha = stagep.sha256_array(scale_candidate)
        raw_stagep_sha = stagep.sha256_array(raw_proposal)
        checks = {
            "runtime_stagee_contract": stagee_candidate_sha
            == stagee_array_sha_contract(scale_candidate),
            "runtime_stagep_contract": stagep_candidate_sha
            == stageop_array_sha_contract(scale_candidate),
            "legacy_stagee_scale_candidate_sha": stagee_candidate_sha
            == _mapping(
                legacy_candidates.get(_scale_key(scale)),
                f"legacy candidate {scale}",
            ).get("candidate_sha256"),
            "stagep_candidate_sha": stagep_candidate_sha
            == base_attempt.get("reconstructed_candidate_sha256"),
            "stagep_raw_proposal_sha": raw_stagep_sha
            == base_attempt.get("raw_proposal_sha256"),
        }
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            raise StageQResume1Error(
                f"per-scale hash-domain preflight failed at {scale}: {failed}"
            )
        copied_attempt = corrected_attempts[index]
        if not isinstance(copied_attempt, MutableMapping):
            raise StageQResume1Error("copied Stage-P attempt is not mutable")
        copied_attempt["reconstructed_candidate_sha256"] = stagee_candidate_sha
        per_scale_records.append(
            {
                "internal_scale": float(scale),
                "stagee_candidate_sha256": stagee_candidate_sha,
                "stagep_candidate_sha256": stagep_candidate_sha,
                "stagep_raw_proposal_sha256": raw_stagep_sha,
                "hash_domains_distinct": stagee_candidate_sha
                != stagep_candidate_sha,
                "all_checks_pass": True,
            }
        )

    return {
        "corrected_base_cell": corrected_cell,
        "expected_internal_order": expected_order,
        "final_candidate_stagee_sha256": stagee_final_sha,
        "final_candidate_stagep_sha256": stagep_final_sha,
        "final_checks": final_checks,
        "per_scale_records": per_scale_records,
        "all_hash_domains_validated": True,
        "candidate_bytes_modified": False,
        "base_report_modified": False,
    }


class _SingleUseCachedCallback:
    def __init__(
        self,
        *,
        integration: Mapping[str, Any],
        capture: Mapping[str, Any],
        expected_control: np.ndarray,
        expected_direction: np.ndarray,
        expected_context: Mapping[str, Any],
        expected_integrator_spec: Any,
        expected_callback_spec: Any,
    ) -> None:
        self.integration = integration
        self.capture = capture
        self.expected_control = np.asarray(expected_control)
        self.expected_direction = np.asarray(expected_direction)
        self.expected_context = expected_context
        self.expected_integrator_spec = expected_integrator_spec
        self.expected_callback_spec = expected_callback_spec
        self.call_count = 0

    def __call__(self, **kwargs: Any) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        self.call_count += 1
        if self.call_count != 1:
            raise StageQResume1Error("cached callback integration called more than once")
        if not np.array_equal(np.asarray(kwargs.get("control")), self.expected_control):
            raise StageQResume1Error("cached callback control changed")
        if not np.array_equal(np.asarray(kwargs.get("direction")), self.expected_direction):
            raise StageQResume1Error("cached callback direction changed")
        if kwargs.get("context") is not self.expected_context:
            raise StageQResume1Error("cached callback context changed")
        if kwargs.get("integrator_spec") is not self.expected_integrator_spec:
            raise StageQResume1Error("cached integrator spec changed")
        if kwargs.get("callback_spec") is not self.expected_callback_spec:
            raise StageQResume1Error("cached callback spec changed")
        return self.integration, self.capture


@contextmanager
def _patched_runtime_hash_and_callback(
    *,
    runtime: Mapping[str, Any],
    candidate_stagee_sha: str,
    cached_callback: _SingleUseCachedCallback,
):
    stageo = runtime["stageo"]
    stagek = runtime["stagek"]
    stagee = runtime["stagel"].stagee258
    original_stageo_sha = stageo.sha256_array
    original_callback = stagek.callback_integrate_rowwise

    def domain_dispatch(value: np.ndarray) -> str:
        array = np.asarray(value)
        if array.dtype == np.float32 and array.ndim == 3:
            result = stagee.sha256_array(array)
            if result != candidate_stagee_sha:
                raise StageQResume1Error(
                    "unexpected float32 candidate passed to hash dispatcher"
                )
            return result
        return original_stageo_sha(value)

    stageo.sha256_array = domain_dispatch
    stagek.callback_integrate_rowwise = cached_callback
    try:
        yield
    finally:
        stageo.sha256_array = original_stageo_sha
        stagek.callback_integrate_rowwise = original_callback


def recovered_audit_shadow_cell(
    *,
    original_audit: Callable[..., Mapping[str, Any]],
    source_id: str,
    timestep: int,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    base_cell: Mapping[str, Any],
    spec: Any,
) -> Mapping[str, Any]:
    stagek = runtime["stagek"]
    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    proposed_direction = direction_raw * float(spec.external_multiplier)
    integration, capture = stagek.callback_integrate_rowwise(
        control=control_raw,
        direction=proposed_direction,
        context=context,
        integrator_spec=runtime["integrator_spec"],
        callback_spec=runtime["callback_spec"],
    )
    recovery = prepare_hash_domain_recovery(
        source_id=source_id,
        timestep=timestep,
        control=control_raw,
        base_direction=direction_raw,
        context=context,
        runtime=runtime,
        base_cell=base_cell,
        spec=spec,
        integration=integration,
        capture=capture,
    )
    cached = _SingleUseCachedCallback(
        integration=integration,
        capture=capture,
        expected_control=control_raw,
        expected_direction=proposed_direction,
        expected_context=context,
        expected_integrator_spec=runtime["integrator_spec"],
        expected_callback_spec=runtime["callback_spec"],
    )
    with _patched_runtime_hash_and_callback(
        runtime=runtime,
        candidate_stagee_sha=str(
            recovery["final_candidate_stagee_sha256"]
        ),
        cached_callback=cached,
    ):
        cell = original_audit(
            source_id=source_id,
            timestep=timestep,
            control=control_raw,
            base_direction=direction_raw,
            context=context,
            runtime=runtime,
            base_cell=_mapping(
                recovery["corrected_base_cell"], "corrected base cell"
            ),
            spec=spec,
        )
    if cached.call_count != 1:
        raise StageQResume1Error("unchanged Stage-Q audit did not consume cached replay")
    result = copy.deepcopy(dict(cell))
    result["hash_domain_recovery"] = {
        "final_candidate_stagee_sha256": recovery[
            "final_candidate_stagee_sha256"
        ],
        "final_candidate_stagep_sha256": recovery[
            "final_candidate_stagep_sha256"
        ],
        "final_hash_domains_distinct": recovery[
            "final_candidate_stagee_sha256"
        ]
        != recovery["final_candidate_stagep_sha256"],
        "final_checks": recovery["final_checks"],
        "per_scale_records": recovery["per_scale_records"],
        "callback_pair_run_count": 1,
        "callback_pair_reused_by_original_stageq_audit": True,
        "candidate_bytes_modified": False,
        "base_report_modified": False,
        "original_stageq_logic_called": True,
    }
    return result


@contextmanager
def patched_stageq_cell_audit(stageq: Any):
    original = stageq.audit_shadow_cell

    def recovered(**kwargs: Any) -> Mapping[str, Any]:
        runtime = dict(kwargs["runtime"])
        runtime["stageq"] = stageq
        kwargs = dict(kwargs)
        kwargs["runtime"] = runtime
        return recovered_audit_shadow_cell(original_audit=original, **kwargs)

    stageq.audit_shadow_cell = recovered
    try:
        yield
    finally:
        stageq.audit_shadow_cell = original


def execute_recovery(
    *,
    root: Path,
    environment: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_environment_variables()
    from ccda_phase3 import (
        phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow as stageq,
    )

    if sha256_file(Path(stageq.__file__).resolve()) != EXPECTED_STAGEQ_SOURCE_SHA256:
        raise StageQResume1Error("imported Stage-Q source SHA changed")
    original_audit = stageq.audit_shadow_cell
    with patched_stageq_cell_audit(stageq):
        stageq_result = stageq.run_shadow_alignment(
            root=Path(root).resolve(),
            environment=environment,
            repository=repository,
        )
    if stageq.audit_shadow_cell is not original_audit:
        raise StageQResume1Error("original Stage-Q cell audit was not restored")

    if stageq_result.get("execution_verdict") != "PASS":
        raise StageQResume1Error("recovered Stage-Q execution did not PASS")
    if stageq_result.get("scientific_status") != "BLOCKED":
        raise StageQResume1Error("recovered Stage-Q scientific status changed")
    if stageq_result.get("selected_configuration") is not None:
        raise StageQResume1Error("recovered Stage-Q selected a configuration")
    if stageq_result.get("train_only_recommendation") is not None:
        raise StageQResume1Error("recovered Stage-Q emitted a recommendation")
    shadow = _mapping(
        stageq_result.get("tolerance_aligned_upper_gate_shadow"),
        "Stage-Q shadow result",
    )
    cells = _sequence(shadow.get("cell_records"), "Stage-Q cell records")
    if len(cells) != EXPECTED_PAIR_COUNT:
        raise StageQResume1Error("recovered Stage-Q cell population changed")
    if int(shadow.get("legacy_callback_off_on_pair_count", -1)) != EXPECTED_PAIR_COUNT:
        raise StageQResume1Error("recovered Stage-Q callback pair count changed")
    for index, raw in enumerate(cells):
        cell = _mapping(raw, f"recovered Stage-Q cell {index}")
        recovery = _mapping(
            cell.get("hash_domain_recovery"), f"hash recovery {index}"
        )
        if recovery.get("callback_pair_run_count") != 1:
            raise StageQResume1Error("a Stage-Q cell duplicated its callback pair")
        if recovery.get("candidate_bytes_modified") is not False:
            raise StageQResume1Error("candidate bytes changed during recovery")
        if recovery.get("original_stageq_logic_called") is not True:
            raise StageQResume1Error("original Stage-Q cell logic was bypassed")
    require_false(stageq_result, FALSE_BOUNDARIES, "recovered Stage-Q result")

    stageq_result_sha = sha256_bytes(stable_json_bytes(stageq_result))
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": stageq_result["scientific_status"],
        "root_cause": stageq_result["root_cause"],
        "required_next_path": stageq_result["required_next_path"],
        "primary_failure_locus": stageq_result["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": copy.deepcopy(dict(repository)),
        "environment": copy.deepcopy(dict(environment)),
        "recovery_contract": {
            "original_stage_q_runner_attempt_count": 1,
            "original_stage_q_success_report_created": False,
            "original_stage_q_blocked_report_preserved": True,
            "original_stage_q_blocked_report_sha256": (
                EXPECTED_STAGEQ_BLOCKED_REPORT_SHA256
            ),
            "original_failure_message": (
                "Stage-Q replay differs from Stage-P: ['candidate_sha256']"
            ),
            "repair_scope": "candidate_hash_domain_identity_only",
            "stagee_hash_domain": "dtype+json_list_shape+bytes",
            "stageop_hash_domain": "dtype+pipe+repr_tuple_shape+pipe+bytes",
            "final_and_per_scale_domains_validated": True,
            "original_stage_q_source_modified": False,
            "temporary_function_replacements_restored": True,
            "resume1_runner_attempt_count": 1,
            "resume1_targeted_callback_pair_count": EXPECTED_PAIR_COUNT,
            "callback_pairs_duplicated": False,
        },
        "stage_q_result_sha256": stageq_result_sha,
        "stage_q_scientific_result_sha256": stageq_result.get(
            "scientific_result_sha256"
        ),
        "stage_q_result": stageq_result,
        "mechanism_boundary": {
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stagem_modified": False,
            "stagen_modified": False,
            "stageo_modified": False,
            "stageo_resume1_modified": False,
            "stagep_modified": False,
            "original_stageq_modified": False,
            "legacy_upper_gate_changed": False,
            "aligned_upper_gate_changed": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "classification_tree_changed": False,
            "candidate_bytes_changed": False,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }


def blocked_report(
    *, repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stageq_resume1_hash_domain_recovery_failed",
        "required_next_path": "RESTORE_STAGEQ_RESUME1_CANDIDATE_HASH_DOMAIN_RECOVERY",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "original_stage_q_blocked_report_preserved": True,
        "original_stage_q_blocked_report_sha256": (
            EXPECTED_STAGEQ_BLOCKED_REPORT_SHA256
        ),
        "original_stage_q_source_modified": False,
        "candidate_bytes_modified": False,
        "callback_pairs_duplicated": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
