"""Phase3.14b-r2.5.9 Stage D Resume2 cable-state schema recovery.

The Stage-D risk-repair experiment operates on ``diffusion_target_cable``.
That array is not the complete 87-dimensional state-v3 tensor: Stage-A creates
it with ``full_future[..., :CABLE_DIM]``.  The frozen state-v3 schema has 24
cable beads, therefore ``CABLE_DIM == 48`` and the scientific tensors are
``[N, 4, 48]``.

Stage-D originally required 67 dimensions.  Resume1 changed the requirement to
87 dimensions and then used only the first 46 values as 23 beads, confusing the
23 cable segments with the 24 cable beads.  Both contracts fail before durable
science evidence.

This add-only recovery changes only the descriptor input adapter:

* require ``[N, 4, 48]``;
* use the frozen compact descriptor over all 48 cable coordinates;
* reshape all 48 coordinates into 24 ordered XY beads;
* compute the same 49 geometry features over 23 adjacent segments;
* preserve the original 360-recipe bank, nested group OOF, thresholds,
  candidate mechanism, execution counts and result tree.

The execution topology is intentionally small: one portable probe, one cold
science worker and one controller summary.  No holdout or frozen-probe access,
formal training, reverse sampling, IDM, candidate execution, DeformableRavens,
Phase4 or CPS is allowed.
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

from ccda_phase3.schema_v3 import CABLE_DIM, DEFAULT_TF, N_BEADS

PHASE = "Phase3.14b-r2.5.9 Stage D Resume2"
SCHEMA = "phase314b_r259_staged_resume2_cable_state_schema_recovery_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_HEAD = "ee57ed875c7fb9d5eefbc06dc455a1b49097f5ae"
BASE_RESUME1_IMPLEMENTATION = "00802d2196ddb1fd6326316eecbc3f8071b388fd"
BASE_STAGE_D_BLOCKED_EVIDENCE = "8e87a28cb98a7529a0e58c30ce8a0d774dc4471f"
BASE_STAGE_D_IMPLEMENTATION = "865a61e1f72494bab0e9c1976f911ff87d638d5f"
BASE_STAGE_C_EVIDENCE = "591d1fd0814f7bd50b8e580274462908512a745f"
EXPECTED_REMOTE = BASE_HEAD
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage D Resume2: restore 48D cable schema"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D Resume2 t10 risk repair evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D Resume2 blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r259_staged_resume2_cable_state_schema_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r259_staged_resume2_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r259_staged_resume2_cable_state_schema_recovery.py",
    ),
)

RESUME1_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_resume1_"
    "state_dimension_contract_recovery_blocked_summary.json"
)
RESUME1_BLOCKED_REPORT_SHA256 = (
    "1c270b74574665754d0f24da09c6464d4ba134872744f8731d2e6362829ac9d3"
)
RESUME1_PROBE_EVIDENCE = (
    "reports/phase3_14b_r259_staged_resume1_environment_probe_evidence.json"
)
RESUME1_PROBE_SHA256 = (
    "e1f2eab4cfd42cd60b9ea21873463c63143b1ba907dd29c7321f574a1f889d69"
)
ORIGINAL_BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_"
    "t10_risk_descriptor_calibration_repair_blocked_summary.json"
)
ORIGINAL_BLOCKED_REPORT_SHA256 = (
    "9323a89eb99ecf74fb0b4b2bce9154723624f903bc2ea4c0ab76684552199c26"
)

PROBE_EVIDENCE = (
    "reports/phase3_14b_r259_staged_resume2_environment_probe_evidence.json"
)
WORKER_EVIDENCE = (
    "reports/phase3_14b_r259_staged_resume2_t10_risk_repair_worker_evidence.json"
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_staged_resume2_cable_state_schema_recovery_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_staged_resume2_cable_state_schema_recovery_blocked_summary.json"
)

LEGACY_DESCRIPTOR_DIM = 17
GEOMETRY_DESCRIPTOR_DIM = 49
COMBINED_DESCRIPTOR_DIM = 66
SEGMENT_COUNT = N_BEADS - 1


class StageDResume2Error(RuntimeError):
    """Fail-closed Stage-D Resume2 error."""


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
        raise StageDResume2Error("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageDResume2Error("write-once output exists: {}".format(target))
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
        raise StageDResume2Error("write-ahead source missing: {}".format(source_path))
    payload = source_path.read_bytes()
    validator(json.loads(payload.decode("utf-8")))
    atomic_write_once(destination_path, payload)
    if destination_path.read_bytes() != payload:
        raise StageDResume2Error("write-ahead promotion is not byte-exact")
    return sha256_bytes(payload)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageDResume2Error("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageDResume2Error("{} worktree is dirty".format(label))


def validate_resume1_blocked_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise StageDResume2Error("Resume1 blocked report is missing")
    if sha256_file(report_path) != RESUME1_BLOCKED_REPORT_SHA256:
        raise StageDResume2Error("Resume1 blocked report SHA changed")
    payload = load_json(report_path)
    required = {
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "staged_resume1_execution_contract",
        "worker_evidence_present": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise StageDResume2Error("Resume1 blocked field changed: {}".format(key))
    if "risk descriptor tensor shape must be [N,4,87]" not in str(
        payload.get("error_message", "")
    ):
        raise StageDResume2Error("Resume1 failure message changed")
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "origin": (_git(repo, "rev-parse", "origin/Experiment1"), EXPECTED_REMOTE),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageDResume2Error(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != (
        IMPLEMENTATION_SUBJECT
    ):
        raise StageDResume2Error("Resume2 implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(
        sorted(IMPLEMENTATION_PATHS)
    ):
        raise StageDResume2Error("Resume2 implementation paths changed")

    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageDResume2Error("DeformableRavens worktree commit changed")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(submodule, "DeformableRavens")

    validate_resume1_blocked_report(repo / RESUME1_BLOCKED_REPORT)
    if sha256_file(repo / RESUME1_PROBE_EVIDENCE) != RESUME1_PROBE_SHA256:
        raise StageDResume2Error("Resume1 probe evidence SHA changed")
    if sha256_file(repo / ORIGINAL_BLOCKED_REPORT) != ORIGINAL_BLOCKED_REPORT_SHA256:
        raise StageDResume2Error("original Stage-D blocked report SHA changed")

    for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageDResume2Error("Resume2 output already exists: {}".format(relative))

    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "resume1_blocked_evidence_commit": BASE_HEAD,
        "resume1_implementation_commit": BASE_RESUME1_IMPLEMENTATION,
        "stage_d_implementation_commit": BASE_STAGE_D_IMPLEMENTATION,
        "stage_c_evidence_commit": BASE_STAGE_C_EVIDENCE,
    }


def _row_mean(value: np.ndarray) -> np.ndarray:
    return np.mean(value.reshape(value.shape[0], -1), axis=1)


def _row_max(value: np.ndarray) -> np.ndarray:
    return np.max(value.reshape(value.shape[0], -1), axis=1)


def build_risk_descriptors_48(
    descriptor_id: str,
    control: np.ndarray,
    direction: np.ndarray,
    candidate: np.ndarray,
    selected_scale: np.ndarray,
) -> np.ndarray:
    """Build the frozen Stage-D descriptors for cable-only [N,4,48] tensors."""
    staged = _base()
    control_value = np.asarray(control, dtype=np.float64)
    direction_value = np.asarray(direction, dtype=np.float64)
    candidate_value = np.asarray(candidate, dtype=np.float64)
    scale_value = np.asarray(selected_scale, dtype=np.float64)

    if control_value.shape != direction_value.shape or control_value.shape != candidate_value.shape:
        raise StageDResume2Error("risk descriptor tensor shapes differ")
    if control_value.ndim != 3 or control_value.shape[1:] != (DEFAULT_TF, CABLE_DIM):
        raise StageDResume2Error(
            f"risk descriptor tensor shape must be [N,{DEFAULT_TF},{CABLE_DIM}]"
        )
    rows = int(control_value.shape[0])
    if scale_value.shape != (rows,):
        raise StageDResume2Error("selected-scale shape must be [N]")

    legacy = staged.stagex.compact_risk_descriptors(
        control_value,
        direction_value,
        candidate_value,
        scale_value,
    )
    if legacy.shape != (rows, LEGACY_DESCRIPTOR_DIM):
        raise StageDResume2Error("legacy descriptor shape changed")
    if descriptor_id == "compact_v1":
        result = np.asarray(legacy, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise StageDResume2Error("legacy descriptors contain NaN or Inf")
        return result
    if descriptor_id != "compact_geometry_v2":
        raise StageDResume2Error("descriptor ID changed")

    control_xy = control_value.reshape(rows, DEFAULT_TF, N_BEADS, 2)
    candidate_xy = candidate_value.reshape(rows, DEFAULT_TF, N_BEADS, 2)
    direction_xy = direction_value.reshape(rows, DEFAULT_TF, N_BEADS, 2)
    movement_xy = candidate_xy - control_xy

    control_segments = np.linalg.norm(np.diff(control_xy, axis=2), axis=3)
    candidate_segments = np.linalg.norm(np.diff(candidate_xy, axis=2), axis=3)
    if control_segments.shape[2] != SEGMENT_COUNT:
        raise StageDResume2Error("ordered segment count changed")
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
    columns.extend(candidate_segments.max(axis=2)[:, horizon] for horizon in range(DEFAULT_TF))
    columns.extend(np.abs(segment_delta).max(axis=2)[:, horizon] for horizon in range(DEFAULT_TF))
    columns.extend(np.abs(chain_delta)[:, horizon] for horizon in range(DEFAULT_TF))
    columns.extend(bead_motion.max(axis=2)[:, horizon] for horizon in range(DEFAULT_TF))
    columns.extend(np.abs(bend_delta).mean(axis=2)[:, horizon] for horizon in range(DEFAULT_TF))

    geometry = np.stack(columns, axis=1).astype(np.float64)
    if geometry.shape != (rows, GEOMETRY_DESCRIPTOR_DIM):
        raise StageDResume2Error("geometry descriptor width changed")
    result = np.concatenate([legacy, geometry], axis=1).astype(np.float64)
    if result.shape != (rows, COMBINED_DESCRIPTOR_DIM):
        raise StageDResume2Error("combined descriptor width changed")
    if not np.all(np.isfinite(result)):
        raise StageDResume2Error("risk descriptors contain NaN or Inf")
    return result


@contextmanager
def installed_cable_schema_adapter() -> Iterator[None]:
    staged = _base()
    original = staged.build_risk_descriptors
    staged.build_risk_descriptors = build_risk_descriptors_48
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
    with installed_cable_schema_adapter():
        payload = dict(
            staged.run_stage_d(
                root=Path(root).resolve(),
                probe_payload=probe_payload,
                repository_head=str(repository_head),
                stageu_contract=stageu_contract,
            )
        )
    payload.pop("worker_result_sha256", None)
    payload["resume2_cable_schema_recovery"] = {
        "actual_tensor_contract": ["N", DEFAULT_TF, CABLE_DIM],
        "state_v3_bead_count": N_BEADS,
        "ordered_segment_count": SEGMENT_COUNT,
        "legacy_descriptor_uses_all_cable_coordinates": True,
        "geometry_descriptor_uses_all_ordered_beads": True,
        "discarded_cable_coordinate_count": 0,
        "recipe_bank_changed": False,
        "nested_group_oof_changed": False,
        "candidate_mechanism_changed": False,
        "thresholds_changed": False,
        "single_cold_science_worker": True,
    }
    payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_evidence(payload)
    return payload


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    staged = _base()
    staged.validate_worker_evidence(payload)
    recovery = payload.get("resume2_cable_schema_recovery")
    expected = {
        "actual_tensor_contract": ["N", DEFAULT_TF, CABLE_DIM],
        "state_v3_bead_count": N_BEADS,
        "ordered_segment_count": SEGMENT_COUNT,
        "legacy_descriptor_uses_all_cable_coordinates": True,
        "geometry_descriptor_uses_all_ordered_beads": True,
        "discarded_cable_coordinate_count": 0,
        "recipe_bank_changed": False,
        "nested_group_oof_changed": False,
        "candidate_mechanism_changed": False,
        "thresholds_changed": False,
        "single_cold_science_worker": True,
    }
    if recovery != expected:
        raise StageDResume2Error("Resume2 cable-schema recovery contract changed")
    expected_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected_hash:
        raise StageDResume2Error("Resume2 worker self-hash changed")


def build_summary(
    *, repository: Mapping[str, Any], probe_path: Path, worker_path: Path
) -> Mapping[str, Any]:
    staged = _base()
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    validate_probe_evidence(probe)
    validate_worker_evidence(worker)
    if int(probe.get("process_id", -1)) == int(worker.get("process_id", -1)):
        raise StageDResume2Error("probe and worker process IDs are not distinct")

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": copy.deepcopy(dict(repository)),
        "cable_schema_recovery": copy.deepcopy(worker["resume2_cable_schema_recovery"]),
        "portable_environment_contract": copy.deepcopy(
            probe["portable_environment_audit"]
        ),
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
        "full_objective_oof_fixed_recipe_selection": copy.deepcopy(
            worker["full_objective_oof_fixed_recipe_selection"]
        ),
        "procedure_gate_totals": copy.deepcopy(worker["procedure_gate_totals"]),
        "execution_counts": copy.deepcopy(worker["execution_counts"]),
        "durable_evidence": {
            "probe_path": PROBE_EVIDENCE,
            "probe_sha256": sha256_file(probe_path),
            "worker_path": WORKER_EVIDENCE,
            "worker_sha256": sha256_file(worker_path),
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_persisted_before_summary": True,
            "promotion_byte_exact": True,
        },
        "process_topology": {
            "probe_process_id": int(probe["process_id"]),
            "worker_process_id": int(worker["process_id"]),
            "processes_distinct": True,
            "cold_science_worker_count": 1,
        },
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(
            worker["train_only_recommendation"]
        ),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in staged.FALSE_BOUNDARIES},
    }
    result["summary_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_summary(result)
    return result


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageDResume2Error("Resume2 summary schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageDResume2Error("Resume2 scientific status changed")
    recovery = payload.get("cable_schema_recovery")
    if not isinstance(recovery, Mapping) or recovery.get("actual_tensor_contract") != [
        "N",
        DEFAULT_TF,
        CABLE_DIM,
    ]:
        raise StageDResume2Error("Resume2 summary cable schema changed")
    if payload.get("selected_configuration") is not None:
        raise StageDResume2Error("Resume2 selected a configuration")
    recommendation = payload.get("train_only_recommendation")
    if payload.get("scientific_status") == "READY":
        if not isinstance(recommendation, Mapping):
            raise StageDResume2Error("READY Resume2 lacks train-only recommendation")
    elif recommendation is not None:
        raise StageDResume2Error("BLOCKED Resume2 retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageDResume2Error("Resume2 re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageDResume2Error("Resume2 holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageDResume2Error("Resume2 accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageDResume2Error("Resume2 authorized rerun")
    if any(payload.get(key) is not False for key in _base().FALSE_BOUNDARIES):
        raise StageDResume2Error("Resume2 crossed a forbidden boundary")
    expected_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected_hash:
        raise StageDResume2Error("Resume2 summary self-hash changed")


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
        "root_cause": "phase314b_r259_staged_resume2_cable_schema_recovery_failed",
        "required_next_path": (
            "FINALIZE_R259_STAGED_RESUME2_EVIDENCE_IF_WORKER_PERSISTED_"
            "OTHERWISE_AUDIT_EXECUTION_CONTRACT"
        ),
        "primary_failure_locus": "staged_resume2_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
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
