"""Phase3.14b-r2.5.9 Stage D Resume1 state-dimension contract recovery.

The original Stage-D implementation was scientifically preregistered and its
focused test gate passed, but Worker A stopped before durable science evidence
because the new descriptor adapter incorrectly required a 67-dimensional state.
The current Phase3 cable-state contract is [N, 4, 87]; the first 46 coordinates
remain the ordered 23-bead XY cable geometry used by the preregistered
``compact_geometry_v2`` descriptor.

This add-only recovery changes only that entry contract.  It invokes the
original Stage-D ``run_stage_d`` under a temporary descriptor adapter, restores
the original function in ``finally``, and keeps the recipe bank, nested group
OOF, direction/candidate mechanisms, thresholds, execution counts and
classification tree unchanged.

One portable probe and one cold science worker are persisted through an
external write-ahead directory.  Stage-C Resume1 already established current-
device byte-exact reproducibility and RTX-3090/RTX-4080-SUPER functional
compatibility, so this narrow recovery does not repeat two complete 360-recipe
workers.  No selection holdout, frozen probe, formal training, reverse sampling,
IDM, candidate execution, DeformableRavens execution, Phase4 or CPS is allowed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.9 Stage D Resume1"
SCHEMA = "phase314b_r259_staged_resume1_state_dimension_contract_recovery_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGE_D_IMPLEMENTATION_COMMIT = "865a61e1f72494bab0e9c1976f911ff87d638d5f"
BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT = "8e87a28cb98a7529a0e58c30ce8a0d774dc4471f"
BASE_STAGE_D_PARENT = "591d1fd0814f7bd50b8e580274462908512a745f"
EXPECTED_REMOTE = "591d1fd0814f7bd50b8e580274462908512a745f"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage D: repair t10 risk descriptors and calibration"
)
BASE_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.9 Stage D blocked evidence"
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage D Resume1: restore 87D state contract"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D Resume1 t10 risk repair evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D Resume1 blocked evidence"
)

BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_staged_t10_risk_descriptor_calibration_repair.py"),
    ("A", "scripts/phase3_14b_r259_staged_execute.py"),
    ("A", "scripts/phase3_14b_r259_staged_worker.py"),
    ("A", "tests/test_phase3_14b_r259_staged_t10_risk_descriptor_calibration_repair.py"),
)
BASE_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r259_staged_t10_risk_descriptor_calibration_repair.py": (
        "0be595499b12e1782c005ca873e59559e102725e9b5141c5397c76d917ce6261"
    ),
    "scripts/phase3_14b_r259_staged_execute.py": (
        "975cf72f6aa94397f8bb24f5a50238e66cb6d70121b9ac79a537b526b6bffc45"
    ),
    "scripts/phase3_14b_r259_staged_worker.py": (
        "1ed5a80d6ff47e9af115ae2efc6a3e6baa6e9ebb8e0c1637609a493c32a3d6aa"
    ),
    "tests/test_phase3_14b_r259_staged_t10_risk_descriptor_calibration_repair.py": (
        "92fce742ca002b655d08b4f4217d321e634a4c22536f9ec0045e0c073d42e180"
    ),
}

BASE_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_t10_risk_descriptor_calibration_repair_blocked_summary.json"
)
BASE_BLOCKED_REPORT_SHA256 = (
    "9323a89eb99ecf74fb0b4b2bce9154723624f903bc2ea4c0ab76684552199c26"
)
BASE_PROBE_EVIDENCE = "reports/phase3_14b_r259_staged_environment_probe_evidence.json"
BASE_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = tuple(
    sorted((("A", BASE_BLOCKED_REPORT), ("A", BASE_PROBE_EVIDENCE)))
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r259_staged_resume1_state_dimension_contract_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r259_staged_resume1_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r259_staged_resume1_state_dimension_contract_recovery.py",
    ),
)

PROBE_EVIDENCE = "reports/phase3_14b_r259_staged_resume1_environment_probe_evidence.json"
WORKER_EVIDENCE = "reports/phase3_14b_r259_staged_resume1_t10_risk_repair_worker_evidence.json"
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_staged_resume1_state_dimension_contract_recovery_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_resume1_state_dimension_contract_recovery_blocked_summary.json"
)

EXPECTED_HORIZONS = 4
EXPECTED_STATE_DIM = 87
CABLE_XY_DIM = 46
CABLE_BEADS = 23
EXPECTED_LEGACY_DESCRIPTOR_DIM = 17
EXPECTED_GEOMETRY_DESCRIPTOR_DIM = 49
EXPECTED_COMBINED_DESCRIPTOR_DIM = 66


class StageDResume1Error(RuntimeError):
    """Fail-closed Stage-D Resume1 error."""


def _base() -> Any:
    from ccda_phase3 import (
        phase314b_r259_staged_t10_risk_descriptor_calibration_repair as staged,
    )

    return staged


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageDResume1Error("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageDResume1Error("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        descriptor = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def promote_write_ahead(source: Path, destination: Path, validator: Any) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise StageDResume1Error("write-ahead source is missing: {}".format(source_path))
    payload = source_path.read_bytes()
    validator(json.loads(payload.decode("utf-8")))
    atomic_write_once(destination_path, payload)
    if destination_path.read_bytes() != payload:
        raise StageDResume1Error("write-ahead promotion is not byte-exact")
    return sha256_bytes(payload)


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
            raise StageDResume1Error("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageDResume1Error("{} worktree is dirty".format(label))


def validate_original_blocked_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise StageDResume1Error("original Stage-D blocked report is missing")
    if sha256_file(report_path) != BASE_BLOCKED_REPORT_SHA256:
        raise StageDResume1Error("original Stage-D blocked report SHA changed")
    payload = load_json(report_path)
    required = {
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "staged_execution_contract",
        "required_next_path": "DESIGN_ADD_ONLY_R259_STAGED_EXECUTION_RECOVERY_BEFORE_SCIENCE",
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise StageDResume1Error("original blocked report field changed: {}".format(key))
    message = str(payload.get("error_message", ""))
    if "risk descriptor tensor shape must be [N,4,67]" not in message:
        raise StageDResume1Error("original Stage-D failure message changed")
    if payload.get("science_reexecution_authorized") is not False:
        raise StageDResume1Error("original Stage-D report authorized rerun")
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if len(str(implementation_commit)) != 40:
        raise StageDResume1Error("implementation commit is invalid")
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT),
        "blocked parent": (
            _git(repo, "rev-parse", BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT + "^"),
            BASE_STAGE_D_IMPLEMENTATION_COMMIT,
        ),
        "implementation parent": (
            _git(repo, "rev-parse", BASE_STAGE_D_IMPLEMENTATION_COMMIT + "^"),
            BASE_STAGE_D_PARENT,
        ),
        "origin": (_git(repo, "rev-parse", "origin/Experiment1"), EXPECTED_REMOTE),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageDResume1Error(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    subjects = (
        (BASE_STAGE_D_IMPLEMENTATION_COMMIT, BASE_IMPLEMENTATION_SUBJECT),
        (BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT, BASE_BLOCKED_SUBJECT),
        (implementation_commit, IMPLEMENTATION_SUBJECT),
    )
    for commit, expected in subjects:
        if _git(repo, "show", "-s", "--format=%s", commit) != expected:
            raise StageDResume1Error("commit subject changed: {}".format(commit))
    if commit_name_status(repo, BASE_STAGE_D_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_IMPLEMENTATION_PATHS)
    ):
        raise StageDResume1Error("original Stage-D implementation paths changed")
    if commit_name_status(repo, BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_BLOCKED_PATHS)
    ):
        raise StageDResume1Error("original Stage-D blocked evidence paths changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageDResume1Error("Resume1 implementation paths changed")

    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageDResume1Error("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-D Resume1")

    for relative, expected_sha in BASE_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageDResume1Error("original Stage-D source changed: {}".format(relative))
        if _git_bytes(
            repo,
            "show",
            "{}:{}".format(BASE_STAGE_D_IMPLEMENTATION_COMMIT, relative),
        ) != path.read_bytes():
            raise StageDResume1Error("original Stage-D source/commit differs: {}".format(relative))

    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageDResume1Error("Resume1 source missing: {}".format(relative))
        if _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative)) != path.read_bytes():
            raise StageDResume1Error("Resume1 source/commit differs: {}".format(relative))

    blocked = repo / BASE_BLOCKED_REPORT
    if _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT, BASE_BLOCKED_REPORT),
    ) != blocked.read_bytes():
        raise StageDResume1Error("original blocked report differs from committed blob")
    validate_original_blocked_report(blocked)

    original_probe = repo / BASE_PROBE_EVIDENCE
    if not original_probe.is_file():
        raise StageDResume1Error("original Stage-D probe evidence is missing")
    if _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT, BASE_PROBE_EVIDENCE),
    ) != original_probe.read_bytes():
        raise StageDResume1Error("original Stage-D probe differs from committed blob")
    _base().validate_probe_evidence(load_json(original_probe))

    for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageDResume1Error("Resume1 output already exists: {}".format(relative))

    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT,
        "original_stage_d_implementation_commit": BASE_STAGE_D_IMPLEMENTATION_COMMIT,
        "original_stage_d_blocked_evidence_commit": BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT,
        "stage_c_resume1_evidence_commit": BASE_STAGE_D_PARENT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
    }


def _row_mean(value: np.ndarray) -> np.ndarray:
    return np.mean(value.reshape(value.shape[0], -1), axis=1)


def _row_max(value: np.ndarray) -> np.ndarray:
    return np.max(value.reshape(value.shape[0], -1), axis=1)


def build_risk_descriptors_87(
    descriptor_id: str,
    control: np.ndarray,
    direction: np.ndarray,
    candidate: np.ndarray,
    selected_scale: np.ndarray,
) -> np.ndarray:
    """Build the frozen Stage-D descriptors for the actual [N,4,87] state."""
    staged = _base()
    control_value = np.asarray(control, dtype=np.float64)
    direction_value = np.asarray(direction, dtype=np.float64)
    candidate_value = np.asarray(candidate, dtype=np.float64)
    scale_value = np.asarray(selected_scale, dtype=np.float64)
    if control_value.shape != direction_value.shape or control_value.shape != candidate_value.shape:
        raise StageDResume1Error("risk descriptor tensor shapes differ")
    if control_value.ndim != 3 or control_value.shape[1:] != (
        EXPECTED_HORIZONS,
        EXPECTED_STATE_DIM,
    ):
        raise StageDResume1Error("risk descriptor tensor shape must be [N,4,87]")
    rows = int(control_value.shape[0])
    if scale_value.shape != (rows,):
        raise StageDResume1Error("selected-scale shape must be [N]")

    legacy = staged.stagex.compact_risk_descriptors(
        control_value,
        direction_value,
        candidate_value,
        scale_value,
    )
    if legacy.shape != (rows, EXPECTED_LEGACY_DESCRIPTOR_DIM):
        raise StageDResume1Error("legacy descriptor shape changed")
    if descriptor_id == "compact_v1":
        result = np.asarray(legacy, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise StageDResume1Error("legacy descriptors contain NaN or Inf")
        return result
    if descriptor_id != "compact_geometry_v2":
        raise StageDResume1Error("descriptor ID changed")

    control_xy = control_value[..., :CABLE_XY_DIM].reshape(
        rows, EXPECTED_HORIZONS, CABLE_BEADS, 2
    )
    candidate_xy = candidate_value[..., :CABLE_XY_DIM].reshape(
        rows, EXPECTED_HORIZONS, CABLE_BEADS, 2
    )
    direction_xy = direction_value[..., :CABLE_XY_DIM].reshape(
        rows, EXPECTED_HORIZONS, CABLE_BEADS, 2
    )
    movement_xy = candidate_xy - control_xy
    control_segments = np.linalg.norm(np.diff(control_xy, axis=2), axis=3)
    candidate_segments = np.linalg.norm(np.diff(candidate_xy, axis=2), axis=3)
    segment_delta = candidate_segments - control_segments
    control_chain = np.sum(control_segments, axis=2)
    candidate_chain = np.sum(candidate_segments, axis=2)
    chain_delta = candidate_chain - control_chain
    control_endpoint = np.linalg.norm(
        control_xy[:, :, -1] - control_xy[:, :, 0], axis=2
    )
    candidate_endpoint = np.linalg.norm(
        candidate_xy[:, :, -1] - candidate_xy[:, :, 0], axis=2
    )
    centroid_motion = np.linalg.norm(
        np.mean(candidate_xy, axis=2) - np.mean(control_xy, axis=2), axis=2
    )
    bead_motion = np.linalg.norm(movement_xy, axis=3)
    control_bend = np.linalg.norm(
        control_xy[:, :, 2:]
        - 2.0 * control_xy[:, :, 1:-1]
        + control_xy[:, :, :-2],
        axis=3,
    )
    candidate_bend = np.linalg.norm(
        candidate_xy[:, :, 2:]
        - 2.0 * candidate_xy[:, :, 1:-1]
        + candidate_xy[:, :, :-2],
        axis=3,
    )
    bend_delta = candidate_bend - control_bend
    flat_direction = direction_xy.reshape(rows, -1)
    flat_movement = movement_xy.reshape(rows, -1)
    direction_norm = np.linalg.norm(flat_direction, axis=1)
    movement_norm = np.linalg.norm(flat_movement, axis=1)
    cosine = np.sum(flat_direction * flat_movement, axis=1) / np.maximum(
        direction_norm * movement_norm, 1.0e-12
    )

    columns: List[np.ndarray] = [
        _row_mean(control_segments),
        _row_max(control_segments),
        np.std(control_segments.reshape(rows, -1), axis=1),
        _row_mean(candidate_segments),
        _row_max(candidate_segments),
        np.std(candidate_segments.reshape(rows, -1), axis=1),
        _row_mean(np.abs(segment_delta)),
        _row_max(np.abs(segment_delta)),
        _row_mean(segment_delta),
        _row_max(candidate_segments) / np.maximum(_row_max(control_segments), 1.0e-12),
        np.mean(control_chain, axis=1),
        np.mean(candidate_chain, axis=1),
        np.mean(np.abs(chain_delta), axis=1),
        np.max(np.abs(chain_delta), axis=1),
        np.mean(candidate_chain, axis=1)
        / np.maximum(np.mean(control_chain, axis=1), 1.0e-12),
        np.mean(control_endpoint, axis=1),
        np.mean(candidate_endpoint, axis=1),
        np.mean(np.abs(candidate_endpoint - control_endpoint), axis=1),
        np.mean(centroid_motion, axis=1),
        np.max(centroid_motion, axis=1),
        _row_mean(bead_motion),
        _row_max(bead_motion),
        _row_mean(control_bend),
        _row_mean(candidate_bend),
        _row_mean(np.abs(bend_delta)),
        direction_norm,
        movement_norm,
        cosine,
        movement_norm / np.maximum(direction_norm, 1.0e-12),
    ]
    columns.extend(candidate_segments.max(axis=2)[:, horizon] for horizon in range(4))
    columns.extend(np.abs(segment_delta).max(axis=2)[:, horizon] for horizon in range(4))
    columns.extend(np.abs(chain_delta)[:, horizon] for horizon in range(4))
    columns.extend(bead_motion.max(axis=2)[:, horizon] for horizon in range(4))
    columns.extend(np.abs(bend_delta).mean(axis=2)[:, horizon] for horizon in range(4))
    geometry = np.stack(columns, axis=1).astype(np.float64)
    if geometry.shape != (rows, EXPECTED_GEOMETRY_DESCRIPTOR_DIM):
        raise StageDResume1Error("geometry descriptor width changed")
    result = np.concatenate([legacy, geometry], axis=1).astype(np.float64)
    if result.shape != (rows, EXPECTED_COMBINED_DESCRIPTOR_DIM):
        raise StageDResume1Error("combined descriptor width changed")
    if not np.all(np.isfinite(result)):
        raise StageDResume1Error("risk descriptors contain NaN or Inf")
    return result


@contextmanager
def installed_state_dimension_adapter() -> Iterator[None]:
    staged = _base()
    original = staged.build_risk_descriptors
    staged.build_risk_descriptors = build_risk_descriptors_87
    try:
        yield
    finally:
        staged.build_risk_descriptors = original


def make_probe_evidence(root: Path) -> Mapping[str, Any]:
    return _base().make_probe_evidence(Path(root).resolve())


def validate_probe_evidence(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _base().validate_probe_evidence(payload)


def run_science_worker(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    staged = _base()
    validate_probe_evidence(probe_payload)
    with installed_state_dimension_adapter():
        payload = dict(
            staged.run_stage_d(
                root=Path(root).resolve(),
                probe_payload=probe_payload,
                repository_head=str(repository_head),
                stageu_contract=stageu_contract,
            )
        )
    payload.pop("worker_result_sha256", None)
    payload["resume1_state_dimension_recovery"] = {
        "original_stage_d_failure_before_durable_science_evidence": True,
        "original_required_state_dim": 67,
        "actual_frozen_state_dim": EXPECTED_STATE_DIM,
        "horizon_count": EXPECTED_HORIZONS,
        "ordered_cable_xy_dim": CABLE_XY_DIM,
        "ordered_cable_bead_count": CABLE_BEADS,
        "legacy_descriptor_uses_full_87d_state": True,
        "geometry_descriptor_uses_first_46_ordered_xy_values": True,
        "recipe_bank_changed": False,
        "nested_group_oof_changed": False,
        "direction_or_candidate_mechanism_changed": False,
        "thresholds_changed": False,
        "single_cold_science_worker": True,
    }
    payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_evidence(payload)
    return payload


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    staged = _base()
    staged.validate_worker_evidence(payload)
    recovery = payload.get("resume1_state_dimension_recovery")
    if not isinstance(recovery, Mapping):
        raise StageDResume1Error("Resume1 state-dimension recovery contract is missing")
    expected = {
        "original_stage_d_failure_before_durable_science_evidence": True,
        "original_required_state_dim": 67,
        "actual_frozen_state_dim": EXPECTED_STATE_DIM,
        "horizon_count": EXPECTED_HORIZONS,
        "ordered_cable_xy_dim": CABLE_XY_DIM,
        "ordered_cable_bead_count": CABLE_BEADS,
        "legacy_descriptor_uses_full_87d_state": True,
        "geometry_descriptor_uses_first_46_ordered_xy_values": True,
        "recipe_bank_changed": False,
        "nested_group_oof_changed": False,
        "direction_or_candidate_mechanism_changed": False,
        "thresholds_changed": False,
        "single_cold_science_worker": True,
    }
    if dict(recovery) != expected:
        raise StageDResume1Error("Resume1 state-dimension recovery contract changed")
    recomputed = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != recomputed:
        raise StageDResume1Error("Resume1 worker self-hash changed")


def build_summary(
    *,
    repository: Mapping[str, Any],
    probe_path: Path,
    worker_path: Path,
) -> Mapping[str, Any]:
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    validate_probe_evidence(probe)
    validate_worker_evidence(worker)
    if int(probe.get("process_id", -1)) == int(worker.get("process_id", -1)):
        raise StageDResume1Error("probe and science worker process IDs are not distinct")
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": copy.deepcopy(dict(repository)),
        "recovery_contract": copy.deepcopy(worker["resume1_state_dimension_recovery"]),
        "portable_environment_contract": copy.deepcopy(
            probe["portable_environment_audit"]
        ),
        "original_stage_d_blocked_evidence": {
            "implementation_commit": BASE_STAGE_D_IMPLEMENTATION_COMMIT,
            "blocked_evidence_commit": BASE_STAGE_D_BLOCKED_EVIDENCE_COMMIT,
            "blocked_report": BASE_BLOCKED_REPORT,
            "blocked_report_sha256": BASE_BLOCKED_REPORT_SHA256,
            "worker_a_started": True,
            "worker_a_durable_science_evidence_generated": False,
            "worker_b_started": False,
        },
        "population": copy.deepcopy(worker["population"]),
        "legacy_stagec_control_contract": copy.deepcopy(
            worker["legacy_stagec_control_contract"]
        ),
        "recipe_population": copy.deepcopy(worker["recipe_population"]),
        "outer_fold_selections": copy.deepcopy(worker["outer_fold_selections"]),
        "outer_fold_selected_recipe_metrics": copy.deepcopy(
            worker["outer_fold_selected_recipe_metrics"]
        ),
        "outer_selection_modal_recipe_diagnostic": copy.deepcopy(
            worker["outer_selection_modal_recipe_diagnostic"]
        ),
        "outer_crossfit_fold_selected_recipe_record": copy.deepcopy(
            worker["outer_crossfit_fold_selected_recipe_record"]
        ),
        "outer_crossfit_no_abstention_baseline_record": copy.deepcopy(
            worker["outer_crossfit_no_abstention_baseline_record"]
        ),
        "outer_crossfit_procedure_eligibility": copy.deepcopy(
            worker["outer_crossfit_procedure_eligibility"]
        ),
        "all_outer_inner_selections_eligible": worker[
            "all_outer_inner_selections_eligible"
        ],
        "outer_recipe_modal_support_stable": worker[
            "outer_recipe_modal_support_stable"
        ],
        "full_objective_oof_fixed_recipe_selection": copy.deepcopy(
            worker["full_objective_oof_fixed_recipe_selection"]
        ),
        "full_objective_oof_selected_recipe_eligible": worker[
            "full_objective_oof_selected_recipe_eligible"
        ],
        "procedure_gate_totals": copy.deepcopy(worker["procedure_gate_totals"]),
        "execution_counts": copy.deepcopy(worker["execution_counts"]),
        "durable_evidence_protocol": {
            "probe_evidence_path": PROBE_EVIDENCE,
            "probe_evidence_file_sha256": sha256_file(probe_path),
            "worker_evidence_path": WORKER_EVIDENCE,
            "worker_evidence_file_sha256": sha256_file(worker_path),
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_persisted_before_controller_decoration": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_evidence_promoted_byte_exact": True,
            "write_ahead_files_write_once": True,
            "controller_recomputed_worker_science": False,
        },
        "process_topology": {
            "environment_probe_process_count": 1,
            "cold_science_worker_count": 1,
            "probe_process_id": int(probe["process_id"]),
            "science_worker_process_id": int(worker["process_id"]),
            "processes_distinct": True,
        },
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(
            worker["train_only_recommendation"]
        ),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
    }
    staged = _base()
    for key in staged.FALSE_BOUNDARIES:
        result[key] = False
    result["summary_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_summary(result)
    return result


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageDResume1Error("Resume1 summary schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageDResume1Error("Resume1 scientific status changed")
    recovery = payload.get("recovery_contract")
    if not isinstance(recovery, Mapping) or recovery.get("actual_frozen_state_dim") != 87:
        raise StageDResume1Error("Resume1 summary state dimension changed")
    protocol = payload.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageDResume1Error("Resume1 durable protocol missing")
    for key in (
        "worker_persisted_before_controller_decoration",
        "external_write_ahead_outside_git_worktree",
        "repository_evidence_promoted_byte_exact",
        "write_ahead_files_write_once",
    ):
        if protocol.get(key) is not True:
            raise StageDResume1Error("Resume1 durable protocol changed: {}".format(key))
    if protocol.get("controller_recomputed_worker_science") is not False:
        raise StageDResume1Error("Resume1 controller recomputed science")
    if payload.get("selected_configuration") is not None:
        raise StageDResume1Error("Resume1 selected a configuration")
    if payload.get("scientific_status") == "READY":
        if not isinstance(payload.get("train_only_recommendation"), Mapping):
            raise StageDResume1Error("READY summary lacks train-only recommendation")
    elif payload.get("train_only_recommendation") is not None:
        raise StageDResume1Error("BLOCKED summary retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageDResume1Error("Resume1 re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageDResume1Error("Resume1 holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageDResume1Error("Resume1 accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageDResume1Error("Resume1 authorized rerun")
    if any(payload.get(key) is not False for key in _base().FALSE_BOUNDARIES):
        raise StageDResume1Error("Resume1 crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageDResume1Error("Resume1 summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Path,
    worker_path: Path,
) -> Mapping[str, Any]:
    probe_present = Path(probe_path).is_file()
    worker_present = Path(worker_path).is_file()
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_staged_resume1_state_dimension_recovery_execution_failed",
        "required_next_path": (
            "RESTORE_R259_STAGED_RESUME1_EVIDENCE_FINALIZATION_IF_WORKER_PERSISTED_"
            "OTHERWISE_AUDIT_EXECUTION_CONTRACT"
        ),
        "primary_failure_locus": "staged_resume1_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "original_stage_d_blocked_report_sha256": BASE_BLOCKED_REPORT_SHA256,
        "probe_evidence_present": probe_present,
        "probe_evidence_sha256": sha256_file(probe_path) if probe_present else None,
        "worker_evidence_present": worker_present,
        "worker_evidence_sha256": sha256_file(worker_path) if worker_present else None,
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in _base().FALSE_BOUNDARIES},
    }
