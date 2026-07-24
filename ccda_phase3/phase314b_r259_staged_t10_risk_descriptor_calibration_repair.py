"""Phase3.14b-r2.5.9 Stage D t=10 risk descriptor/calibration repair.

This is a new objective-train-only nested-group-OOF stage.  It does not replay
selection holdout or frozen probe evidence.  The Stage-C Resume1 result proved
that the remaining failure is t=10 risk abstention with mixed candidate quality,
while t=25 and t=50 pass the frozen procedure gates.  Stage D therefore keeps
those two timesteps as immutable controls and changes only the target-independent
risk descriptor/calibration layer at t=10.

The direction backbone, direction supervision, candidate generator, aligned
geometry gate, shrinkage bank, risk-threshold bank, group folds and all physical
mechanism code remain frozen.  Two independent cold workers must be byte-exact
on the same device.  No row-level tensor, model, weight, probability vector,
candidate, checkpoint, NPZ, image or video is persisted.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r258_stagex_tail_robust_nested_oof as stagex
from ccda_phase3 import phase314b_r258_stagex_resume1_import_crossfit_recovery as kernel
from ccda_phase3 import phase314b_r259_stagec_fold_resolved_tail_attribution as stagec
from ccda_phase3 import phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery as stagec_resume1

PHASE = "Phase3.14b-r2.5.9 Stage D"
SCHEMA = "phase314b_r259_staged_t10_risk_descriptor_calibration_repair_v1"
PROBE_SCHEMA = SCHEMA + "_environment_probe_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT = "b6b35467fbd7736c89ed9746c181e61c22adcd05"
BASE_STAGEC_RESUME1_EVIDENCE_COMMIT = "591d1fd0814f7bd50b8e580274462908512a745f"
BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT = "042ab9b4a3187d44d85eb61dbc438ccb5cef266f"
EXPECTED_REMOTE_HEADS = (
    "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5",
    BASE_STAGEC_RESUME1_EVIDENCE_COMMIT,
)
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_STAGEC_RESUME1_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage C Resume1: restore portable GPU compatibility"
)
BASE_STAGEC_RESUME1_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage C Resume1 fold-resolved attribution evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage D: repair t10 risk descriptors and calibration"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D t10 risk repair evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage D blocked evidence"
)

BASE_STAGEC_RESUME1_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery.py"),
    ("A", "scripts/phase3_14b_r259_stagec_resume1_execute.py"),
    ("A", "scripts/phase3_14b_r259_stagec_resume1_worker.py"),
    ("A", "tests/test_phase3_14b_r259_stagec_resume1_portable_gpu_compatibility_recovery.py"),
)
BASE_STAGEC_RESUME1_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery.py": (
        "2547fa56b0d2a5464df92151521d84d7ef905dea58566dedb3be8445bae5926e"
    ),
    "scripts/phase3_14b_r259_stagec_resume1_execute.py": (
        "a7e31bf18289ef98e313fbb90d9bbdc5083b3eb565ecde3352d6e8a0d2fe317d"
    ),
    "scripts/phase3_14b_r259_stagec_resume1_worker.py": (
        "af838a215863d89e685c062de0e9afeec89e27c401155e96f9c66da91af5f4d3"
    ),
    "tests/test_phase3_14b_r259_stagec_resume1_portable_gpu_compatibility_recovery.py": (
        "2caf242b72b37bcc45b6db7fb66b8b309299fa7d259445f53a0a20fa23129907"
    ),
}
BASE_STAGEC_RESUME1_PROBE = (
    "reports/phase3_14b_r259_stagec_resume1_environment_probe_evidence.json"
)
BASE_STAGEC_RESUME1_WORKER_A = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_worker_a_evidence.json"
)
BASE_STAGEC_RESUME1_WORKER_B = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_worker_b_evidence.json"
)
BASE_STAGEC_RESUME1_SUMMARY = (
    "reports/phase3_14b_r259_stagec_resume1_fold_resolved_tail_attribution_summary.json"
)
BASE_STAGEC_RESUME1_SUMMARY_SELF_SHA256 = (
    "c77dda0c3a4e7a3f1396cec22e81b77cbf27de25d39af9040244d67b1b2eca38"
)
BASE_STAGEC_RESUME1_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = tuple(
    sorted(
        (
            ("A", BASE_STAGEC_RESUME1_PROBE),
            ("A", BASE_STAGEC_RESUME1_WORKER_A),
            ("A", BASE_STAGEC_RESUME1_WORKER_B),
            ("A", BASE_STAGEC_RESUME1_SUMMARY),
        )
    )
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_staged_t10_risk_descriptor_calibration_repair.py"),
    ("A", "scripts/phase3_14b_r259_staged_execute.py"),
    ("A", "scripts/phase3_14b_r259_staged_worker.py"),
    ("A", "tests/test_phase3_14b_r259_staged_t10_risk_descriptor_calibration_repair.py"),
)

PROBE_EVIDENCE = "reports/phase3_14b_r259_staged_environment_probe_evidence.json"
WORKER_A_EVIDENCE = "reports/phase3_14b_r259_staged_t10_risk_repair_worker_a_evidence.json"
WORKER_B_EVIDENCE = "reports/phase3_14b_r259_staged_t10_risk_repair_worker_b_evidence.json"
SUCCESS_REPORT = "reports/phase3_14b_r259_staged_t10_risk_descriptor_calibration_repair_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_staged_t10_risk_descriptor_calibration_repair_blocked_summary.json"

LOCKED_TIMESTEP = 10
CONTROL_TIMESTEPS = (25, 50)
OUTER_FOLDS = 6
INNER_FOLDS = 5
EXPECTED_ROWS = 638
EXPECTED_GROUPS = 126
DIRECTION_SHRINKAGES = (0.50, 0.75, 1.00)
RISK_THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
DESCRIPTOR_IDS = ("compact_v1", "compact_geometry_v2")
RISK_L2_VALUES = (0.25, 1.00, 4.00)
TEMPERATURES = (0.75, 1.00, 1.25, 1.50, 2.00)
EXPECTED_RECIPE_COUNT = 360
EXPECTED_EXECUTION_COUNTS: Mapping[str, int] = {
    "direction_fit_count": 36,
    "candidate_generation_count": 108,
    "risk_fit_count": 648,
    "internal_scale_attempt_count": 756,
    "inner_recipe_evaluation_count": 2160,
    "full_objective_recipe_evaluation_count": 360,
    "nonconverged_risk_fit_count": 0,
    "descriptor_build_count": 72,
}

FALSE_BOUNDARIES = stagec_resume1.FALSE_BOUNDARIES


class StageDError(RuntimeError):
    """Fail-closed Stage-D error."""


@dataclass(frozen=True)
class RiskRecipe:
    descriptor_id: str
    risk_l2: float
    temperature: float
    shrinkage: float
    risk_threshold: float

    @property
    def recipe_id(self) -> str:
        return (
            "desc_{}__l2_{:.2f}__temp_{:.2f}__shrink_{:.2f}__risk_{:.2f}".format(
                self.descriptor_id,
                self.risk_l2,
                self.temperature,
                self.shrinkage,
                self.risk_threshold,
            )
        )

    def validate(self) -> None:
        if self.descriptor_id not in DESCRIPTOR_IDS:
            raise StageDError("descriptor left predeclared bank")
        if float(self.risk_l2) not in RISK_L2_VALUES:
            raise StageDError("risk L2 left predeclared bank")
        if float(self.temperature) not in TEMPERATURES:
            raise StageDError("temperature left predeclared bank")
        if float(self.shrinkage) not in DIRECTION_SHRINKAGES:
            raise StageDError("shrinkage left predeclared bank")
        if float(self.risk_threshold) not in RISK_THRESHOLDS:
            raise StageDError("threshold left predeclared bank")


@dataclass(frozen=True)
class StageDSpec:
    minimum_acceptance_rate: float = 0.50
    minimum_positive_reduction_rate: float = 0.50
    adverse_sse_reduction_fraction: float = 0.10
    group_cvar_fraction: float = 0.20
    modal_outer_fold_minimum: int = 4
    epsilon: float = 1.0e-12

    def validate(self) -> None:
        if self.minimum_acceptance_rate != 0.50:
            raise StageDError("acceptance gate changed")
        if self.minimum_positive_reduction_rate != 0.50:
            raise StageDError("positive-reduction gate changed")
        if self.adverse_sse_reduction_fraction != 0.10:
            raise StageDError("adverse-SSE gate changed")
        if self.group_cvar_fraction != 0.20:
            raise StageDError("CVaR fraction changed")
        if self.modal_outer_fold_minimum != 4:
            raise StageDError("modal support gate changed")
        if self.epsilon != 1.0e-12:
            raise StageDError("numerical epsilon changed")


def recipe_population() -> Tuple[RiskRecipe, ...]:
    values = tuple(
        RiskRecipe(
            descriptor_id=descriptor_id,
            risk_l2=l2,
            temperature=temperature,
            shrinkage=shrinkage,
            risk_threshold=threshold,
        )
        for descriptor_id in DESCRIPTOR_IDS
        for l2 in RISK_L2_VALUES
        for temperature in TEMPERATURES
        for shrinkage in DIRECTION_SHRINKAGES
        for threshold in RISK_THRESHOLDS
    )
    for value in values:
        value.validate()
    if len(values) != EXPECTED_RECIPE_COUNT:
        raise StageDError("recipe population changed")
    if len({item.recipe_id for item in values}) != len(values):
        raise StageDError("recipe IDs are not unique")
    return values


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
        raise StageDError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageDError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def promote_write_ahead(source: Path, destination: Path, validator: Any) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise StageDError("write-ahead source missing: {}".format(source_path))
    if destination_path.exists():
        raise StageDError("repository evidence already exists: {}".format(destination_path))
    payload = source_path.read_bytes()
    validator(json.loads(payload.decode("utf-8")))
    atomic_write_once(destination_path, payload)
    if destination_path.read_bytes() != payload:
        raise StageDError("write-ahead promotion is not byte-exact")
    return sha256_bytes(payload)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageDError(
            "git {} failed: {}".format(" ".join(args), completed.stderr.strip())
        )
    return completed.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageDError("git bytes {} failed".format(" ".join(args)))
    return completed.stdout


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageDError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageDError("{} worktree is dirty".format(label))


def validate_stagec_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    stagec_resume1.validate_summary(summary)
    if summary.get("summary_sha256") != BASE_STAGEC_RESUME1_SUMMARY_SELF_SHA256:
        raise StageDError("Stage-C Resume1 summary self-hash changed")
    if summary.get("primary_failure_locus") != (
        "t10_risk_abstention_limited_with_mixed_candidate_quality"
    ):
        raise StageDError("Stage-C Resume1 failure locus changed")
    if summary.get("required_next_path") != (
        "DESIGN_R259_STAGED_OBJECTIVE_TRAIN_ONLY_T10_RISK_CALIBRATION_AND_"
        "DESCRIPTOR_REPAIR_WITH_NESTED_GROUP_OOF"
    ):
        raise StageDError("Stage-C Resume1 next path changed")
    if summary.get("scientific_status") != "BLOCKED":
        raise StageDError("Stage-C Resume1 scientific status changed")
    if summary.get("selected_configuration") is not None:
        raise StageDError("Stage-C Resume1 selected a configuration")
    if summary.get("train_only_recommendation") is not None:
        raise StageDError("Stage-C Resume1 retained a recommendation")
    identity = summary.get("same_device_double_worker_identity")
    if not isinstance(identity, Mapping) or identity.get(
        "same_device_scientific_projection_byte_exact"
    ) is not True:
        raise StageDError("Stage-C Resume1 worker identity changed")
    portable = summary.get("portable_stagea_reproduction_identity")
    if not isinstance(portable, Mapping) or portable.get("portable_contract_passed") is not True:
        raise StageDError("Stage-C Resume1 portable identity changed")
    return summary


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageDError("Stage D requires Experiment1")
    if _git(repo, "rev-parse", "HEAD") != implementation_commit:
        raise StageDError("Stage-D implementation HEAD changed")
    if _git(repo, "rev-parse", "HEAD^") != BASE_STAGEC_RESUME1_EVIDENCE_COMMIT:
        raise StageDError("Stage-D implementation parent changed")
    if _git(repo, "rev-parse", BASE_STAGEC_RESUME1_EVIDENCE_COMMIT + "^") != (
        BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT
    ):
        raise StageDError("Stage-C Resume1 evidence parent changed")
    if _git(repo, "rev-parse", BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT + "^") != (
        BASE_STAGEC_BLOCKED_EVIDENCE_COMMIT
    ):
        raise StageDError("Stage-C Resume1 implementation parent changed")
    subjects = (
        (
            BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT,
            BASE_STAGEC_RESUME1_IMPLEMENTATION_SUBJECT,
        ),
        (BASE_STAGEC_RESUME1_EVIDENCE_COMMIT, BASE_STAGEC_RESUME1_EVIDENCE_SUBJECT),
        (implementation_commit, IMPLEMENTATION_SUBJECT),
    )
    for commit, subject in subjects:
        if _git(repo, "show", "-s", "--format=%s", commit) != subject:
            raise StageDError("commit subject changed: {}".format(commit))
    if commit_name_status(repo, BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEC_RESUME1_IMPLEMENTATION_PATHS)
    ):
        raise StageDError("Stage-C Resume1 implementation paths changed")
    if commit_name_status(repo, BASE_STAGEC_RESUME1_EVIDENCE_COMMIT) != (
        BASE_STAGEC_RESUME1_EVIDENCE_PATHS
    ):
        raise StageDError("Stage-C Resume1 evidence paths changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageDError("Stage-D implementation paths changed")
    remote = _git(repo, "rev-parse", "origin/Experiment1")
    if remote not in EXPECTED_REMOTE_HEADS:
        raise StageDError("origin/Experiment1 is outside the predeclared remote states")
    _git(repo, "merge-base", "--is-ancestor", remote, BASE_STAGEC_RESUME1_EVIDENCE_COMMIT)
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageDError("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageDError("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage D")

    for relative, expected_sha in BASE_STAGEC_RESUME1_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageDError("Stage-C Resume1 source changed: {}".format(relative))
        committed = _git_bytes(
            repo,
            "show",
            "{}:{}".format(BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT, relative),
        )
        if committed != path.read_bytes():
            raise StageDError("Stage-C Resume1 source differs from committed blob")
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageDError("Stage-D implementation source missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageDError("Stage-D source differs from committed blob: {}".format(relative))

    summary_path = repo / BASE_STAGEC_RESUME1_SUMMARY
    if not summary_path.is_file():
        raise StageDError("Stage-C Resume1 summary is missing")
    committed_summary = _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_STAGEC_RESUME1_EVIDENCE_COMMIT, BASE_STAGEC_RESUME1_SUMMARY),
    )
    if committed_summary != summary_path.read_bytes():
        raise StageDError("Stage-C Resume1 summary differs from committed blob")
    validate_stagec_summary(load_json(summary_path))
    for relative in (PROBE_EVIDENCE, WORKER_A_EVIDENCE, WORKER_B_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageDError("Stage-D output already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_STAGEC_RESUME1_EVIDENCE_COMMIT,
        "origin_experiment1": remote,
        "submodule_commit": EXPECTED_SUBMODULE,
        "stagec_resume1_implementation_commit": BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT,
        "stagec_resume1_evidence_commit": BASE_STAGEC_RESUME1_EVIDENCE_COMMIT,
        "stagec_resume1_summary_self_sha256": BASE_STAGEC_RESUME1_SUMMARY_SELF_SHA256,
    }


def make_probe_evidence(root: Path) -> Mapping[str, Any]:
    source = stagec_resume1.make_probe_evidence(Path(root).resolve())
    environment = stagec_resume1.validate_probe_evidence(source)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": int(source["process_id"]),
        "environment": copy.deepcopy(dict(environment)),
        "environment_sha256": str(source["environment_sha256"]),
        "portable_environment_audit": copy.deepcopy(source["portable_environment_audit"]),
        "source_stagec_resume1_probe_schema": source.get("schema"),
        "source_stagec_resume1_probe_sha256": source.get("probe_evidence_sha256"),
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
        raise StageDError("Stage-D probe schema/verdict changed")
    process_id = payload.get("process_id")
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise StageDError("Stage-D probe PID is invalid")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise StageDError("Stage-D probe environment missing")
    if payload.get("environment_sha256") != stagec_resume1.base.stagea.resume2a.sha256_bytes(
        stagec_resume1.base.stagea.resume2a.stable_json_bytes(environment)
    ):
        raise StageDError("Stage-D environment SHA is invalid")
    audit = payload.get("portable_environment_audit")
    if not isinstance(audit, Mapping) or audit.get("portable_contract_passed") is not True:
        raise StageDError("Stage-D portable environment contract failed")
    if audit.get("hardware_identity_required") is not False:
        raise StageDError("Stage-D reintroduced exact hardware equality")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageDError("Stage-D probe is not durable")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "probe_evidence_sha256"}
        )
    )
    if payload.get("probe_evidence_sha256") != expected:
        raise StageDError("Stage-D probe self-hash changed")
    return environment


LEGACY_DESCRIPTOR_NAMES = (
    "selected_scale",
    "has_selected_scale",
    "direction_norm",
    "movement_norm",
    "control_norm",
    "direction_abs_max",
    "movement_abs_max",
    "direction_movement_cosine",
    "movement_direction_ratio",
    "direction_h0_norm",
    "direction_h1_norm",
    "direction_h2_norm",
    "direction_h3_norm",
    "movement_h0_norm",
    "movement_h1_norm",
    "movement_h2_norm",
    "movement_h3_norm",
)
GEOMETRY_DESCRIPTOR_NAMES = (
    "control_segment_mean",
    "control_segment_max",
    "control_segment_std",
    "candidate_segment_mean",
    "candidate_segment_max",
    "candidate_segment_std",
    "segment_abs_delta_mean",
    "segment_abs_delta_max",
    "segment_signed_delta_mean",
    "candidate_control_max_segment_ratio",
    "control_chain_mean",
    "candidate_chain_mean",
    "chain_abs_delta_mean",
    "chain_abs_delta_max",
    "candidate_control_chain_ratio",
    "control_endpoint_mean",
    "candidate_endpoint_mean",
    "endpoint_abs_delta_mean",
    "centroid_motion_mean",
    "centroid_motion_max",
    "bead_motion_mean",
    "bead_motion_max",
    "control_bend_mean",
    "candidate_bend_mean",
    "bend_abs_delta_mean",
    "direction_xy_norm",
    "movement_xy_norm",
    "direction_movement_cosine_xy",
    "movement_direction_ratio_xy",
    "candidate_segment_max_h0",
    "candidate_segment_max_h1",
    "candidate_segment_max_h2",
    "candidate_segment_max_h3",
    "segment_abs_delta_max_h0",
    "segment_abs_delta_max_h1",
    "segment_abs_delta_max_h2",
    "segment_abs_delta_max_h3",
    "chain_abs_delta_h0",
    "chain_abs_delta_h1",
    "chain_abs_delta_h2",
    "chain_abs_delta_h3",
    "bead_motion_max_h0",
    "bead_motion_max_h1",
    "bead_motion_max_h2",
    "bead_motion_max_h3",
    "bend_abs_delta_mean_h0",
    "bend_abs_delta_mean_h1",
    "bend_abs_delta_mean_h2",
    "bend_abs_delta_mean_h3",
)


def descriptor_names(descriptor_id: str) -> Tuple[str, ...]:
    if descriptor_id == "compact_v1":
        return LEGACY_DESCRIPTOR_NAMES
    if descriptor_id == "compact_geometry_v2":
        return LEGACY_DESCRIPTOR_NAMES + GEOMETRY_DESCRIPTOR_NAMES
    raise StageDError("unknown descriptor ID")


def _row_mean(value: np.ndarray) -> np.ndarray:
    return np.mean(value.reshape(value.shape[0], -1), axis=1)


def _row_max(value: np.ndarray) -> np.ndarray:
    return np.max(value.reshape(value.shape[0], -1), axis=1)


def build_risk_descriptors(
    descriptor_id: str,
    control: np.ndarray,
    direction: np.ndarray,
    candidate: np.ndarray,
    selected_scale: np.ndarray,
) -> np.ndarray:
    control_value = np.asarray(control, dtype=np.float64)
    direction_value = np.asarray(direction, dtype=np.float64)
    candidate_value = np.asarray(candidate, dtype=np.float64)
    scale_value = np.asarray(selected_scale, dtype=np.float64)
    if control_value.shape != direction_value.shape or control_value.shape != candidate_value.shape:
        raise StageDError("risk descriptor tensor shape changed")
    if control_value.ndim != 3 or control_value.shape[1:] != (4, 67):
        raise StageDError("risk descriptor tensor shape must be [N,4,67]")
    rows = control_value.shape[0]
    if scale_value.shape != (rows,):
        raise StageDError("selected-scale shape must be [N]")
    legacy = stagex.compact_risk_descriptors(
        control_value, direction_value, candidate_value, scale_value
    )
    if legacy.shape != (rows, len(LEGACY_DESCRIPTOR_NAMES)):
        raise StageDError("legacy descriptor shape changed")
    if descriptor_id == "compact_v1":
        return np.asarray(legacy, dtype=np.float64)
    if descriptor_id != "compact_geometry_v2":
        raise StageDError("descriptor ID changed")

    control_xy = control_value[..., :46].reshape(rows, 4, 23, 2)
    candidate_xy = candidate_value[..., :46].reshape(rows, 4, 23, 2)
    direction_xy = direction_value[..., :46].reshape(rows, 4, 23, 2)
    movement_xy = candidate_xy - control_xy
    control_segments = np.linalg.norm(np.diff(control_xy, axis=2), axis=3)
    candidate_segments = np.linalg.norm(np.diff(candidate_xy, axis=2), axis=3)
    segment_delta = candidate_segments - control_segments
    control_chain = np.sum(control_segments, axis=2)
    candidate_chain = np.sum(candidate_segments, axis=2)
    chain_delta = candidate_chain - control_chain
    control_endpoint = np.linalg.norm(control_xy[:, :, -1] - control_xy[:, :, 0], axis=2)
    candidate_endpoint = np.linalg.norm(candidate_xy[:, :, -1] - candidate_xy[:, :, 0], axis=2)
    centroid_motion = np.linalg.norm(
        np.mean(candidate_xy, axis=2) - np.mean(control_xy, axis=2), axis=2
    )
    bead_motion = np.linalg.norm(movement_xy, axis=3)
    control_bend = np.linalg.norm(
        control_xy[:, :, 2:] - 2.0 * control_xy[:, :, 1:-1] + control_xy[:, :, :-2],
        axis=3,
    )
    candidate_bend = np.linalg.norm(
        candidate_xy[:, :, 2:] - 2.0 * candidate_xy[:, :, 1:-1] + candidate_xy[:, :, :-2],
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
        _row_max(candidate_segments)
        / np.maximum(_row_max(control_segments), 1.0e-12),
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
    if geometry.shape[1] != len(GEOMETRY_DESCRIPTOR_NAMES):
        raise StageDError("geometry descriptor width changed")
    result = np.concatenate([legacy, geometry], axis=1).astype(np.float64)
    if result.shape[1] != len(descriptor_names(descriptor_id)):
        raise StageDError("combined descriptor width changed")
    if not np.all(np.isfinite(result)):
        raise StageDError("risk descriptors contain NaN or Inf")
    return result


def risk_spec(l2: float) -> Any:
    spec = stagex.StageXSpec(risk_l2=float(l2))
    spec.validate()
    return spec


def fit_risk_model_checked(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    l2: float,
) -> Mapping[str, Any]:
    model = stagex.fit_risk_model(features, labels, fit_mask, risk_spec(l2))
    if model.get("mode") != "constant" and model.get("converged") is not True:
        raise StageDError("non-constant risk IRLS did not converge")
    return model


def prevalence_centered_temperature(
    probability: np.ndarray,
    prevalence: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if float(temperature) not in TEMPERATURES:
        raise StageDError("temperature changed")
    value = np.asarray(probability, dtype=np.float64)
    prior = np.asarray(prevalence, dtype=np.float64)
    if prior.ndim == 0:
        prior = np.full(value.shape, float(prior), dtype=np.float64)
    if prior.shape != value.shape:
        raise StageDError("prevalence population changed")
    clipped = np.clip(value, 1.0e-6, 1.0 - 1.0e-6)
    clipped_prior = np.clip(prior, 1.0e-6, 1.0 - 1.0e-6)
    logit = np.log(clipped / (1.0 - clipped))
    prior_logit = np.log(clipped_prior / (1.0 - clipped_prior))
    calibrated_logit = prior_logit + (logit - prior_logit) / float(temperature)
    calibrated_logit = np.clip(calibrated_logit, -40.0, 40.0)
    result = 1.0 / (1.0 + np.exp(-calibrated_logit))
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise StageDError("calibrated risk is invalid")
    return result.astype(np.float64)


def t10_checks(
    record: Mapping[str, Any], baseline: Mapping[str, Any], spec: StageDSpec
) -> Mapping[str, bool]:
    baseline_adverse = float(baseline["adverse_sse_mass"])
    return {
        "acceptance": float(record["acceptance_rate"]) >= spec.minimum_acceptance_rate,
        "overall_mse": float(record["overall_mse_ratio"]) < 1.0,
        "accepted_mse": float(record["accepted_row_mse_ratio"]) < 1.0,
        "positive_reduction": float(record["positive_distance_reduction_rate"])
        > spec.minimum_positive_reduction_rate,
        "relative_reduction": float(record["relative_distance_reduction_mean"]) > 0.0,
        "adverse_sse_reduction": float(record["adverse_sse_mass"])
        <= baseline_adverse * (1.0 - spec.adverse_sse_reduction_fraction),
        "group_cvar_improvement": float(
            record["group_tail"]["worst_fraction_cvar_mse_ratio"]
        )
        < float(baseline["group_tail"]["worst_fraction_cvar_mse_ratio"]),
        "risk_brier_nonworse": record["risk_brier_nonworse"] is True,
    }


def recipe_is_eligible(
    record: Mapping[str, Any], baseline: Mapping[str, Any], spec: StageDSpec
) -> Mapping[str, Any]:
    checks = t10_checks(record, baseline, spec)
    return {"pass": bool(all(checks.values())), "checks": checks}


def _recipe_order(recipe: RiskRecipe) -> Tuple[int, int, int, int, int]:
    return (
        DESCRIPTOR_IDS.index(recipe.descriptor_id),
        RISK_L2_VALUES.index(recipe.risk_l2),
        TEMPERATURES.index(recipe.temperature),
        DIRECTION_SHRINKAGES.index(recipe.shrinkage),
        RISK_THRESHOLDS.index(recipe.risk_threshold),
    )


def select_recipe(
    records: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    spec: StageDSpec,
) -> Mapping[str, Any]:
    population = recipe_population()
    if set(records) != {item.recipe_id for item in population}:
        raise StageDError("recipe record population changed")
    eligibility = {
        item.recipe_id: recipe_is_eligible(records[item.recipe_id], baseline, spec)
        for item in population
    }

    def score(recipe: RiskRecipe) -> Tuple[Any, ...]:
        record = records[recipe.recipe_id]
        return (
            0 if eligibility[recipe.recipe_id]["pass"] else 1,
            float(record["group_tail"]["worst_fraction_cvar_mse_ratio"]),
            float(record["overall_mse_ratio"]),
            float(record["adverse_sse_mass"]),
            -float(record["acceptance_rate"]),
            float(record["risk_brier_score"]),
            *_recipe_order(recipe),
        )

    selected = min(population, key=score)
    return {
        "selected_recipe": asdict(selected),
        "selected_recipe_id": selected.recipe_id,
        "selected_recipe_inner_eligible": eligibility[selected.recipe_id]["pass"],
        "diagnostic_fallback_used": not eligibility[selected.recipe_id]["pass"],
        "eligible_recipe_ids": [
            item.recipe_id for item in population if eligibility[item.recipe_id]["pass"]
        ],
        "recipe_eligibility": eligibility,
        "selection_score": list(score(selected)),
    }


def modal_recipe(selections: Sequence[str]) -> Mapping[str, Any]:
    population = recipe_population()
    order = {item.recipe_id: index for index, item in enumerate(population)}
    if len(selections) != OUTER_FOLDS or any(value not in order for value in selections):
        raise StageDError("outer recipe selection population changed")
    counts = {item.recipe_id: selections.count(item.recipe_id) for item in population}
    selected_id = min(counts, key=lambda key: (-counts[key], order[key]))
    recipe = next(item for item in population if item.recipe_id == selected_id)
    return {
        "recipe": asdict(recipe),
        "recipe_id": selected_id,
        "support_count": counts[selected_id],
        "counts": counts,
    }


def _science_context(
    *, root: Path, probe_payload: Mapping[str, Any], stageu_contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    stagex.validate_environment_variables()
    environment = validate_probe_evidence(probe_payload)
    modules = stagex._science_modules()
    runtime_modules = modules["stageo"]._runtime_modules()
    stagea = runtime_modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment)
    cold = stagea.assert_cold_cuda_context_portable()
    runtime = dict(modules["stageo"]._prepare_runtime(Path(root).resolve(), environment))
    runtime.update({**modules, "stageu_gate": stageu_contract["aligned_gate_lock"]})
    context = stagex._objective_only_context(runtime["context"])
    runtime["context"] = context
    population = stagex.validate_objective_population(context)
    if population["row_count"] != EXPECTED_ROWS or population["group_count"] != EXPECTED_GROUPS:
        raise StageDError("objective-train population changed")
    return {
        "environment": environment,
        "cold": cold,
        "runtime": runtime,
        "context": context,
        "population": population,
        "modules": modules,
    }


def _legacy_control_contract(summary: Mapping[str, Any], spec: StageDSpec) -> Mapping[str, Any]:
    records = summary.get("global_outer_crossfit_fold_selected_policy_records")
    baselines = summary.get("global_outer_crossfit_baseline_records")
    if not isinstance(records, Mapping) or not isinstance(baselines, Mapping):
        raise StageDError("Stage-C global records are missing")
    result: Dict[str, Any] = {}
    for timestep in (10, 25, 50):
        record = records[str(timestep)]
        baseline = baselines[str(timestep)]
        checks = t10_checks(record, baseline, spec)
        result[str(timestep)] = {
            "record": copy.deepcopy(record),
            "baseline": copy.deepcopy(baseline),
            "checks": checks,
            "all_pass": bool(all(checks.values())),
            "failed_checks": sorted(key for key, value in checks.items() if not value),
        }
    if result["10"]["failed_checks"] != ["acceptance"]:
        raise StageDError("Stage-C t10 failure pattern changed")
    for timestep in CONTROL_TIMESTEPS:
        if result[str(timestep)]["all_pass"] is not True:
            raise StageDError("Stage-C frozen control timestep no longer passes")
    return result


def _evaluate_recipe(
    *,
    recipe: RiskRecipe,
    control: np.ndarray,
    candidates: Mapping[float, np.ndarray],
    scales: Mapping[float, np.ndarray],
    probabilities: Mapping[Tuple[str, float, float], np.ndarray],
    prevalences: Mapping[Tuple[str, float, float], np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    stagex_spec: Any,
) -> Mapping[str, Any]:
    key = (recipe.descriptor_id, recipe.risk_l2, recipe.shrinkage)
    calibrated = prevalence_centered_temperature(
        probabilities[key][indices], prevalences[key][indices], recipe.temperature
    )
    return stagex.evaluate_policy(
        control[indices],
        candidates[recipe.shrinkage][indices],
        scales[recipe.shrinkage][indices],
        calibrated,
        target[indices],
        groups[indices],
        recipe.risk_threshold,
        prevalences[key][indices],
        stagex_spec,
    )


def run_stage_d(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
    spec: Optional[StageDSpec] = None,
) -> Mapping[str, Any]:
    active = StageDSpec() if spec is None else spec
    active.validate()
    stagec_summary = validate_stagec_summary(
        load_json(Path(root).resolve() / BASE_STAGEC_RESUME1_SUMMARY)
    )
    legacy_controls = _legacy_control_contract(stagec_summary, active)
    prepared = _science_context(
        root=Path(root).resolve(), probe_payload=probe_payload, stageu_contract=stageu_contract
    )
    environment = prepared["environment"]
    cold = prepared["cold"]
    runtime = dict(prepared["runtime"])
    context = prepared["context"]
    population = prepared["population"]
    modules = prepared["modules"]
    assignment = np.asarray(context["objective_fold_assignment"], dtype=np.int64)
    groups = np.asarray(context["objective_groups"]).astype(str)
    target = np.asarray(context["objective_target"], dtype=np.float32)
    conditions = np.asarray(context["objective_condition_name"]).astype(str)
    control_full = np.asarray(
        context["objective_control_predictions"][LOCKED_TIMESTEP], dtype=np.float32
    )
    stagef = runtime["stagef"]
    definition = stagef.definition_by_id(stagex.LOCKED_BACKBONE)
    features = stagef.build_constraint_features(
        condition=context["objective_condition"],
        control=control_full,
        condition_name=context["objective_condition_name"],
        feature_mode=definition.feature_mode,
        context=context,
    )
    oracle = stagef.generate_projected_oracle_target(
        control=control_full,
        target=target,
        groups=groups,
        condition_name=context["objective_condition_name"],
        timestep=LOCKED_TIMESTEP,
        context=context,
        direction_spec=runtime["direction_spec"],
        integrator_spec=runtime["integrator_spec"],
        spec=runtime["stagef_spec"],
    )
    reconstruction_spec = modules["stager"].StageRSpec()
    reconstruction_spec.validate()
    stagex_spec = stagex.StageXSpec()
    stagex_spec.validate()

    counts = {
        "direction_fit_count": 0,
        "candidate_generation_count": 0,
        "risk_fit_count": 0,
        "internal_scale_attempt_count": 0,
        "inner_recipe_evaluation_count": 0,
        "full_objective_recipe_evaluation_count": 0,
        "nonconverged_risk_fit_count": 0,
        "descriptor_build_count": 0,
    }
    outer_selections: List[Mapping[str, Any]] = []
    outer_outputs: Dict[int, Dict[Tuple[str, float, float], Mapping[str, Any]]] = {}
    fold_diagnostics: List[Mapping[str, Any]] = []

    for outer_fold in range(OUTER_FOLDS):
        outer_test = assignment == outer_fold
        outer_train = ~outer_test
        outer_train_indices = np.flatnonzero(outer_train)
        outer_test_indices = np.flatnonzero(outer_test)
        inner_folds = [value for value in range(OUTER_FOLDS) if value != outer_fold]
        if len(inner_folds) != INNER_FOLDS:
            raise StageDError("inner fold population changed")
        inner_direction = np.zeros(control_full.shape, dtype=np.float64)
        inner_candidates = {
            shrinkage: control_full.copy() for shrinkage in DIRECTION_SHRINKAGES
        }
        inner_scales = {
            shrinkage: np.zeros(EXPECTED_ROWS, dtype=np.float64)
            for shrinkage in DIRECTION_SHRINKAGES
        }
        for inner_fold in inner_folds:
            validation_mask = outer_train & (assignment == inner_fold)
            fit_mask = outer_train & ~validation_mask
            validation_indices = np.flatnonzero(validation_mask)
            fitted = stagex._fit_direction(
                runtime=runtime,
                context=context,
                timestep=LOCKED_TIMESTEP,
                fit_mask=fit_mask,
                predict_indices=validation_indices,
                oracle=oracle,
                features=features,
            )
            counts["direction_fit_count"] += 1
            inner_direction[validation_indices] = fitted["direction"]
            for shrinkage in DIRECTION_SHRINKAGES:
                generated = stagex._candidate_for_indices(
                    runtime=runtime,
                    context=context,
                    timestep=LOCKED_TIMESTEP,
                    indices=validation_indices,
                    direction=fitted["direction"],
                    shrinkage=shrinkage,
                    reconstruction_spec=reconstruction_spec,
                )
                kernel.require_clean_generation(
                    generated,
                    label="stageD outer{} inner{} shrink{}".format(
                        outer_fold, inner_fold, shrinkage
                    ),
                )
                counts["candidate_generation_count"] += 1
                counts["internal_scale_attempt_count"] += stagex.EXPECTED_INTERNAL_SCALE_COUNT
                inner_candidates[shrinkage][validation_indices] = generated["candidate"]
                inner_scales[shrinkage][validation_indices] = generated["selected_scale"]

        descriptors: Dict[Tuple[str, float], np.ndarray] = {}
        labels_by_shrinkage: Dict[float, np.ndarray] = {}
        probabilities: Dict[Tuple[str, float, float], np.ndarray] = {}
        prevalences: Dict[Tuple[str, float, float], np.ndarray] = {}
        for shrinkage in DIRECTION_SHRINKAGES:
            labels = np.zeros(EXPECTED_ROWS, dtype=np.float64)
            labels[outer_train_indices] = stagex._risk_labels(
                control_full[outer_train_indices],
                inner_candidates[shrinkage][outer_train_indices],
                target[outer_train_indices],
            )
            labels_by_shrinkage[shrinkage] = labels
            for descriptor_id in DESCRIPTOR_IDS:
                descriptor = build_risk_descriptors(
                    descriptor_id,
                    control_full,
                    inner_direction * shrinkage,
                    inner_candidates[shrinkage],
                    inner_scales[shrinkage],
                )
                descriptors[(descriptor_id, shrinkage)] = descriptor
                counts["descriptor_build_count"] += 1
                for l2 in RISK_L2_VALUES:
                    key = (descriptor_id, l2, shrinkage)
                    probability = np.ones(EXPECTED_ROWS, dtype=np.float64)
                    prevalence = np.ones(EXPECTED_ROWS, dtype=np.float64)
                    for inner_fold in inner_folds:
                        validation_mask = outer_train & (assignment == inner_fold)
                        fit_mask = (
                            outer_train
                            & ~validation_mask
                            & (inner_scales[shrinkage] > 0.0)
                        )
                        model = fit_risk_model_checked(
                            descriptor, labels, fit_mask, l2
                        )
                        counts["risk_fit_count"] += 1
                        validation_indices = np.flatnonzero(validation_mask)
                        probability[validation_indices] = stagex.predict_risk(
                            model, descriptor[validation_indices]
                        )
                        prevalence[validation_indices] = float(model["prevalence"])
                    probabilities[key] = probability
                    prevalences[key] = prevalence

        baseline = stagex.evaluate_policy(
            control_full[outer_train_indices],
            inner_candidates[1.0][outer_train_indices],
            inner_scales[1.0][outer_train_indices],
            np.zeros(outer_train_indices.size, dtype=np.float64),
            target[outer_train_indices],
            groups[outer_train_indices],
            1.0,
            None,
            stagex_spec,
        )
        recipe_records: Dict[str, Mapping[str, Any]] = {}
        for recipe in recipe_population():
            recipe_records[recipe.recipe_id] = _evaluate_recipe(
                recipe=recipe,
                control=control_full,
                candidates=inner_candidates,
                scales=inner_scales,
                probabilities=probabilities,
                prevalences=prevalences,
                target=target,
                groups=groups,
                indices=outer_train_indices,
                stagex_spec=stagex_spec,
            )
            counts["inner_recipe_evaluation_count"] += 1
        selection = dict(select_recipe(recipe_records, baseline, active))
        selection.update(
            {
                "outer_fold": outer_fold,
                "inner_train_row_count": int(outer_train_indices.size),
                "outer_test_row_count": int(outer_test_indices.size),
            }
        )
        outer_selections.append(selection)

        fitted_outer = stagex._fit_direction(
            runtime=runtime,
            context=context,
            timestep=LOCKED_TIMESTEP,
            fit_mask=outer_train,
            predict_indices=outer_test_indices,
            oracle=oracle,
            features=features,
        )
        counts["direction_fit_count"] += 1
        outer_outputs[outer_fold] = {}
        for shrinkage in DIRECTION_SHRINKAGES:
            generated = stagex._candidate_for_indices(
                runtime=runtime,
                context=context,
                timestep=LOCKED_TIMESTEP,
                indices=outer_test_indices,
                direction=fitted_outer["direction"],
                shrinkage=shrinkage,
                reconstruction_spec=reconstruction_spec,
            )
            kernel.require_clean_generation(
                generated,
                label="stageD outer{} test shrink{}".format(outer_fold, shrinkage),
            )
            counts["candidate_generation_count"] += 1
            counts["internal_scale_attempt_count"] += stagex.EXPECTED_INTERNAL_SCALE_COUNT
            training_labels = labels_by_shrinkage[shrinkage]
            risk_fit_mask = outer_train & (inner_scales[shrinkage] > 0.0)
            for descriptor_id in DESCRIPTOR_IDS:
                training_descriptor = descriptors[(descriptor_id, shrinkage)]
                test_descriptor = build_risk_descriptors(
                    descriptor_id,
                    control_full[outer_test_indices],
                    fitted_outer["direction"] * shrinkage,
                    generated["candidate"],
                    generated["selected_scale"],
                )
                counts["descriptor_build_count"] += 1
                for l2 in RISK_L2_VALUES:
                    model = fit_risk_model_checked(
                        training_descriptor, training_labels, risk_fit_mask, l2
                    )
                    counts["risk_fit_count"] += 1
                    key = (descriptor_id, l2, shrinkage)
                    outer_outputs[outer_fold][key] = {
                        "indices": outer_test_indices,
                        "candidate": np.asarray(generated["candidate"], dtype=np.float32),
                        "selected_scale": np.asarray(
                            generated["selected_scale"], dtype=np.float64
                        ),
                        "base_risk_probability": stagex.predict_risk(
                            model, test_descriptor
                        ),
                        "risk_prevalence": np.full(
                            outer_test_indices.size,
                            float(model["prevalence"]),
                            dtype=np.float64,
                        ),
                        "descriptor_sha256": stagex.sha256_array(test_descriptor),
                        "gate_counts": {
                            "length_log_z_element_mismatch_count": int(
                                generated["length_log_z_element_mismatch_count"]
                            ),
                            "aligned_upper_element_failure_count": int(
                                generated["aligned_upper_element_failure_count"]
                            ),
                            "strict_pass_aligned_fail_row_count": int(
                                generated["strict_pass_aligned_fail_row_count"]
                            ),
                        },
                    }
        selected_recipe = RiskRecipe(**selection["selected_recipe"])
        selected_key = (
            selected_recipe.descriptor_id,
            selected_recipe.risk_l2,
            selected_recipe.shrinkage,
        )
        selected_output = outer_outputs[outer_fold][selected_key]
        selected_probability = prevalence_centered_temperature(
            selected_output["base_risk_probability"],
            selected_output["risk_prevalence"],
            selected_recipe.temperature,
        )
        outer_metric = stagex.evaluate_policy(
            control_full[outer_test_indices],
            selected_output["candidate"],
            selected_output["selected_scale"],
            selected_probability,
            target[outer_test_indices],
            groups[outer_test_indices],
            selected_recipe.risk_threshold,
            selected_output["risk_prevalence"],
            stagex_spec,
        )
        fold_diagnostics.append(
            {
                "outer_fold": outer_fold,
                "selected_recipe_id": selected_recipe.recipe_id,
                "selected_recipe_inner_eligible": selection[
                    "selected_recipe_inner_eligible"
                ],
                "diagnostic_fallback_used": selection["diagnostic_fallback_used"],
                "outer_test_metric": outer_metric,
                "outer_test_condition_counts": {
                    condition: int(np.sum(conditions[outer_test_indices] == condition))
                    for condition in sorted(set(conditions[outer_test_indices].tolist()))
                },
            }
        )

    # Unbiased outer-procedure estimate: every row uses the recipe selected by
    # its own outer fold's inner OOF surface.
    procedure_candidate = control_full.copy()
    procedure_scale = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    procedure_risk = np.ones(EXPECTED_ROWS, dtype=np.float64)
    procedure_constant = np.ones(EXPECTED_ROWS, dtype=np.float64)
    procedure_threshold = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    seen = np.zeros(EXPECTED_ROWS, dtype=np.bool_)
    gate_totals = {
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
    }
    for selection in outer_selections:
        outer_fold = int(selection["outer_fold"])
        recipe = RiskRecipe(**selection["selected_recipe"])
        key = (recipe.descriptor_id, recipe.risk_l2, recipe.shrinkage)
        output = outer_outputs[outer_fold][key]
        indices = np.asarray(output["indices"], dtype=np.int64)
        if np.any(seen[indices]):
            raise StageDError("outer procedure rows overlap")
        seen[indices] = True
        procedure_candidate[indices] = output["candidate"]
        procedure_scale[indices] = output["selected_scale"]
        procedure_risk[indices] = prevalence_centered_temperature(
            output["base_risk_probability"],
            output["risk_prevalence"],
            recipe.temperature,
        )
        procedure_constant[indices] = output["risk_prevalence"]
        procedure_threshold[indices] = recipe.risk_threshold
        for name in gate_totals:
            gate_totals[name] += int(output["gate_counts"][name])
    if not np.all(seen):
        raise StageDError("outer procedure does not cover all rows")
    procedure_record = kernel.evaluate_variable_threshold_policy(
        control_full,
        procedure_candidate,
        procedure_scale,
        procedure_risk,
        target,
        groups,
        procedure_threshold,
        procedure_constant,
        stagex_spec,
    )

    baseline_candidate = control_full.copy()
    baseline_scale = np.zeros(EXPECTED_ROWS, dtype=np.float64)
    for outer_fold in range(OUTER_FOLDS):
        output = outer_outputs[outer_fold][("compact_v1", 1.0, 1.0)]
        indices = np.asarray(output["indices"], dtype=np.int64)
        baseline_candidate[indices] = output["candidate"]
        baseline_scale[indices] = output["selected_scale"]
    baseline_record = stagex.evaluate_policy(
        control_full,
        baseline_candidate,
        baseline_scale,
        np.zeros(EXPECTED_ROWS, dtype=np.float64),
        target,
        groups,
        1.0,
        None,
        stagex_spec,
    )

    full_records: Dict[str, Mapping[str, Any]] = {}
    for recipe in recipe_population():
        candidate = control_full.copy()
        scale = np.zeros(EXPECTED_ROWS, dtype=np.float64)
        risk = np.ones(EXPECTED_ROWS, dtype=np.float64)
        constant = np.ones(EXPECTED_ROWS, dtype=np.float64)
        for outer_fold in range(OUTER_FOLDS):
            key = (recipe.descriptor_id, recipe.risk_l2, recipe.shrinkage)
            output = outer_outputs[outer_fold][key]
            indices = np.asarray(output["indices"], dtype=np.int64)
            candidate[indices] = output["candidate"]
            scale[indices] = output["selected_scale"]
            risk[indices] = prevalence_centered_temperature(
                output["base_risk_probability"],
                output["risk_prevalence"],
                recipe.temperature,
            )
            constant[indices] = output["risk_prevalence"]
        full_records[recipe.recipe_id] = stagex.evaluate_policy(
            control_full,
            candidate,
            scale,
            risk,
            target,
            groups,
            recipe.risk_threshold,
            constant,
            stagex_spec,
        )
        counts["full_objective_recipe_evaluation_count"] += 1
    full_selection = select_recipe(full_records, baseline_record, active)
    modal = modal_recipe([item["selected_recipe_id"] for item in outer_selections])
    procedure_eligibility = recipe_is_eligible(procedure_record, baseline_record, active)
    all_inner_eligible = all(
        item["selected_recipe_inner_eligible"] is True for item in outer_selections
    )
    stable = int(modal["support_count"]) >= active.modal_outer_fold_minimum
    full_eligible = full_selection["selected_recipe_inner_eligible"] is True
    controls_pass = all(
        legacy_controls[str(timestep)]["all_pass"] is True
        for timestep in CONTROL_TIMESTEPS
    )
    ready = bool(
        procedure_eligibility["pass"]
        and all_inner_eligible
        and stable
        and full_eligible
        and controls_pass
    )
    failed_checks = sorted(
        key for key, value in procedure_eligibility["checks"].items() if not value
    )
    if ready:
        root_cause = (
            "phase314b_r259_staged_t10_descriptor_calibration_repair_passes_"
            "unbiased_nested_group_oof_with_frozen_t25_t50_controls"
        )
        next_path = (
            "PREREGISTER_ONE_SHOT_FROZEN_PROBE_EVALUATION_OF_R259_T10_RISK_REPAIR"
        )
        locus = "t10_risk_repair_ready_for_frozen_probe"
    elif failed_checks == ["acceptance"]:
        root_cause = (
            "phase314b_r259_staged_t10_risk_repair_does_not_restore_frozen_coverage"
        )
        next_path = (
            "AUDIT_R259_STAGED_T10_RISK_REPAIR_RESIDUAL_COVERAGE_FAILURE_ON_OBJECTIVE_TRAIN_ONLY"
        )
        locus = "t10_risk_repair_residual_coverage_failure"
    elif not procedure_eligibility["pass"]:
        root_cause = (
            "phase314b_r259_staged_t10_risk_repair_trades_coverage_for_tail_or_fidelity_regression"
        )
        next_path = (
            "AUDIT_R259_STAGED_T10_RISK_REPAIR_FIDELITY_AND_TAIL_TRADEOFF_ON_OBJECTIVE_TRAIN_ONLY"
        )
        locus = "t10_risk_repair_tail_fidelity_tradeoff"
    elif not all_inner_eligible:
        root_cause = "phase314b_r259_staged_t10_risk_recipe_inner_selection_is_unstable"
        next_path = "AUDIT_R259_STAGED_T10_RISK_RECIPE_INNER_FOLD_INSTABILITY"
        locus = "inner_recipe_instability"
    elif not stable:
        root_cause = "phase314b_r259_staged_t10_risk_recipe_is_not_outer_fold_stable"
        next_path = "AUDIT_R259_STAGED_T10_RISK_RECIPE_OUTER_FOLD_INSTABILITY"
        locus = "outer_recipe_instability"
    else:
        root_cause = "phase314b_r259_staged_full_objective_t10_risk_recipe_is_not_eligible"
        next_path = "AUDIT_R259_STAGED_T10_FULL_OBJECTIVE_RECIPE_FAILURE"
        locus = "full_objective_recipe_failure"

    recommendation = None
    if ready:
        selected = RiskRecipe(**full_selection["selected_recipe"])
        recommendation = {
            "backbone_id": stagex.LOCKED_BACKBONE,
            "timesteps": [10, 25, 50],
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "t10_risk_adapter_only": True,
            "t10_descriptor_id": selected.descriptor_id,
            "t10_descriptor_dimension": len(descriptor_names(selected.descriptor_id)),
            "t10_risk_l2": selected.risk_l2,
            "t10_prevalence_centered_temperature": selected.temperature,
            "t10_direction_shrinkage": selected.shrinkage,
            "t10_adverse_risk_threshold": selected.risk_threshold,
            "t25_t50_risk_path": "frozen_stagec_resume1_legacy_control",
            "selection_role": (
                "full_objective_oof_tuning_after_unbiased_outer_procedure_pass"
            ),
            "aligned_gate_contract_sha256": stageu_contract["aligned_gate_lock"][
                "inner_gate_contract_sha256"
            ],
        }

    if counts != dict(EXPECTED_EXECUTION_COUNTS):
        raise StageDError("Stage-D execution counts changed: {!r}".format(counts))
    if any(value != 0 for value in gate_totals.values()):
        raise StageDError("Stage-D tolerance-aligned gate regressed")

    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": root_cause,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "process_id": os.getpid(),
        "repository_head": repository_head,
        "environment_sha256": sha256_bytes(stable_json_bytes(environment)),
        "cold_cuda_precheck": copy.deepcopy(dict(cold)),
        "preregistration_contract": {
            "objective_train_only": True,
            "t10_risk_descriptor_and_calibration_only": True,
            "direction_backbone_changed": False,
            "candidate_generator_changed": False,
            "aligned_gate_changed": False,
            "direction_shrinkage_bank_changed": False,
            "risk_threshold_bank_changed": False,
            "t25_t50_science_rerun": False,
            "t25_t50_stagec_controls_frozen": True,
            "selection_holdout_access_added": 0,
            "frozen_probe_accessed": False,
        },
        "descriptor_contract": {
            "descriptor_ids": list(DESCRIPTOR_IDS),
            "descriptor_dimensions": {
                value: len(descriptor_names(value)) for value in DESCRIPTOR_IDS
            },
            "descriptor_feature_names": {
                value: list(descriptor_names(value)) for value in DESCRIPTOR_IDS
            },
            "target_independent": True,
            "raw_descriptor_tensors_persisted": False,
        },
        "calibration_contract": {
            "method": "prevalence_centered_temperature_scaling",
            "temperature_bank": list(TEMPERATURES),
            "risk_l2_bank": list(RISK_L2_VALUES),
            "posthoc_parameters_fitted": False,
        },
        "recipe_population": [asdict(item) for item in recipe_population()],
        "population": population,
        "legacy_stagec_control_contract": legacy_controls,
        "outer_fold_selections": outer_selections,
        "outer_fold_selected_recipe_metrics": fold_diagnostics,
        "outer_selection_modal_recipe_diagnostic": modal,
        "outer_crossfit_fold_selected_recipe_record": procedure_record,
        "outer_crossfit_no_abstention_baseline_record": baseline_record,
        "outer_crossfit_procedure_eligibility": procedure_eligibility,
        "all_outer_inner_selections_eligible": all_inner_eligible,
        "outer_recipe_modal_support_stable": stable,
        "full_objective_oof_fixed_recipe_selection": full_selection,
        "full_objective_oof_fixed_recipe_records": full_records,
        "full_objective_oof_selected_recipe_eligible": full_eligible,
        "procedure_gate_totals": gate_totals,
        "execution_counts": counts,
        "selected_configuration": None,
        "train_only_recommendation": recommendation,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_evidence(payload)
    return payload


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageDError("Stage-D worker schema/verdict changed")
    status = payload.get("scientific_status")
    if status not in ("READY", "BLOCKED"):
        raise StageDError("Stage-D scientific status changed")
    if payload.get("execution_counts") != EXPECTED_EXECUTION_COUNTS:
        raise StageDError("Stage-D execution counts changed")
    expected_recipes = [asdict(item) for item in recipe_population()]
    if payload.get("recipe_population") != expected_recipes:
        raise StageDError("Stage-D recipe population changed")
    objective_population = payload.get("population")
    if not isinstance(objective_population, Mapping):
        raise StageDError("Stage-D objective population missing")
    if int(objective_population.get("row_count", -1)) != EXPECTED_ROWS:
        raise StageDError("Stage-D objective row population changed")
    if int(objective_population.get("group_count", -1)) != EXPECTED_GROUPS:
        raise StageDError("Stage-D objective group population changed")
    selections = payload.get("outer_fold_selections")
    if not isinstance(selections, list) or len(selections) != OUTER_FOLDS:
        raise StageDError("Stage-D outer selection population changed")
    recipe_by_id = {item.recipe_id: item for item in recipe_population()}
    observed_folds = []
    for selection in selections:
        if not isinstance(selection, Mapping):
            raise StageDError("Stage-D outer selection is invalid")
        outer_fold = int(selection.get("outer_fold", -1))
        observed_folds.append(outer_fold)
        recipe_id = selection.get("selected_recipe_id")
        if recipe_id not in recipe_by_id:
            raise StageDError("Stage-D selected recipe left preregistered bank")
        selected_recipe = selection.get("selected_recipe")
        if selected_recipe != asdict(recipe_by_id[str(recipe_id)]):
            raise StageDError("Stage-D selected recipe ID/payload mismatch")
        if selection.get("selected_recipe_inner_eligible") not in (True, False):
            raise StageDError("Stage-D inner eligibility is invalid")
        if selection.get("diagnostic_fallback_used") is not (
            not selection.get("selected_recipe_inner_eligible")
        ):
            raise StageDError("Stage-D fallback/eligibility contract changed")
    if sorted(observed_folds) != list(range(OUTER_FOLDS)):
        raise StageDError("Stage-D outer fold IDs changed")
    controls = payload.get("legacy_stagec_control_contract")
    if not isinstance(controls, Mapping):
        raise StageDError("Stage-D legacy control contract missing")
    if controls.get("10", {}).get("failed_checks") != ["acceptance"]:
        raise StageDError("Stage-D frozen t10 baseline pattern changed")
    if any(controls.get(str(timestep), {}).get("all_pass") is not True for timestep in CONTROL_TIMESTEPS):
        raise StageDError("Stage-D frozen t25/t50 controls failed")
    gates = payload.get("procedure_gate_totals")
    if not isinstance(gates, Mapping) or set(gates) != {
        "length_log_z_element_mismatch_count",
        "aligned_upper_element_failure_count",
        "strict_pass_aligned_fail_row_count",
    }:
        raise StageDError("Stage-D gate-total schema changed")
    if any(int(value) != 0 for value in gates.values()):
        raise StageDError("Stage-D tolerance-aligned gate regressed")
    procedure = payload.get("outer_crossfit_procedure_eligibility")
    if not isinstance(procedure, Mapping) or procedure.get("pass") not in (True, False):
        raise StageDError("Stage-D procedure eligibility missing")
    modal = payload.get("outer_selection_modal_recipe_diagnostic")
    if not isinstance(modal, Mapping) or modal.get("recipe_id") not in recipe_by_id:
        raise StageDError("Stage-D modal recipe diagnostic changed")
    if int(modal.get("support_count", -1)) < 1 or int(modal.get("support_count", -1)) > OUTER_FOLDS:
        raise StageDError("Stage-D modal support changed")
    full_selection = payload.get("full_objective_oof_fixed_recipe_selection")
    if not isinstance(full_selection, Mapping):
        raise StageDError("Stage-D full-objective recipe selection missing")
    if full_selection.get("selected_recipe_id") not in recipe_by_id:
        raise StageDError("Stage-D full-objective recipe left bank")
    if payload.get("selected_configuration") is not None:
        raise StageDError("Stage-D selected a configuration")
    recommendation = payload.get("train_only_recommendation")
    if status == "READY":
        if not isinstance(recommendation, Mapping):
            raise StageDError("READY Stage-D lacks train-only recommendation")
        if procedure.get("pass") is not True:
            raise StageDError("READY Stage-D procedure did not pass")
        if not all(item["selected_recipe_inner_eligible"] is True for item in selections):
            raise StageDError("READY Stage-D contains fallback outer selection")
        if int(modal["support_count"]) < StageDSpec().modal_outer_fold_minimum:
            raise StageDError("READY Stage-D modal recipe is unstable")
        if full_selection.get("selected_recipe_inner_eligible") is not True:
            raise StageDError("READY Stage-D full-objective recipe is ineligible")
    elif recommendation is not None:
        raise StageDError("BLOCKED Stage-D retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageDError("Stage-D re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageDError("Stage-D cumulative holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageDError("Stage-D accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageDError("Stage-D authorized rerun")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageDError("Stage-D worker is not durable")
    if payload.get("write_ahead_location_outside_git_worktree") is not True:
        raise StageDError("Stage-D write-ahead location changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageDError("Stage-D crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected:
        raise StageDError("Stage-D worker self-hash changed")


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
    result = {
        "same_device_scientific_projection_byte_exact": bytes_a == bytes_b,
        "worker_a_scientific_projection_sha256": sha256_bytes(bytes_a),
        "worker_b_scientific_projection_sha256": sha256_bytes(bytes_b),
    }
    if result["same_device_scientific_projection_byte_exact"] is not True:
        raise StageDError("Stage-D same-device workers differ")
    return result


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
    pids = [probe["process_id"], worker_a["process_id"], worker_b["process_id"]]
    if len(set(int(value) for value in pids)) != 3:
        raise StageDError("Stage-D process IDs are not distinct")
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
        "stagec_resume1_provenance": {
            "implementation_commit": BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT,
            "evidence_commit": BASE_STAGEC_RESUME1_EVIDENCE_COMMIT,
            "summary_path": BASE_STAGEC_RESUME1_SUMMARY,
            "summary_self_sha256": BASE_STAGEC_RESUME1_SUMMARY_SELF_SHA256,
        },
        "portable_environment_contract": copy.deepcopy(
            probe["portable_environment_audit"]
        ),
        "same_device_double_worker_identity": identity,
        "legacy_stagec_control_contract": copy.deepcopy(
            worker_a["legacy_stagec_control_contract"]
        ),
        "outer_fold_selections": copy.deepcopy(worker_a["outer_fold_selections"]),
        "outer_fold_selected_recipe_metrics": copy.deepcopy(
            worker_a["outer_fold_selected_recipe_metrics"]
        ),
        "outer_selection_modal_recipe_diagnostic": copy.deepcopy(
            worker_a["outer_selection_modal_recipe_diagnostic"]
        ),
        "outer_crossfit_fold_selected_recipe_record": copy.deepcopy(
            worker_a["outer_crossfit_fold_selected_recipe_record"]
        ),
        "outer_crossfit_no_abstention_baseline_record": copy.deepcopy(
            worker_a["outer_crossfit_no_abstention_baseline_record"]
        ),
        "outer_crossfit_procedure_eligibility": copy.deepcopy(
            worker_a["outer_crossfit_procedure_eligibility"]
        ),
        "full_objective_oof_fixed_recipe_selection": copy.deepcopy(
            worker_a["full_objective_oof_fixed_recipe_selection"]
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
            "probe_process_id": int(probe["process_id"]),
            "worker_a_process_id": int(worker_a["process_id"]),
            "worker_b_process_id": int(worker_b["process_id"]),
            "processes_distinct": True,
            "workers_sequential": True,
        },
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(
            worker_a["train_only_recommendation"]
        ),
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
        raise StageDError("Stage-D summary schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageDError("Stage-D summary status changed")
    identity = payload.get("same_device_double_worker_identity")
    if not isinstance(identity, Mapping) or identity.get(
        "same_device_scientific_projection_byte_exact"
    ) is not True:
        raise StageDError("Stage-D summary worker identity failed")
    protocol = payload.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageDError("Stage-D durable protocol missing")
    for key in (
        "workers_persisted_before_controller_decoration",
        "external_write_ahead_outside_git_worktree",
        "repository_evidence_promoted_byte_exact",
        "write_ahead_files_write_once",
    ):
        if protocol.get(key) is not True:
            raise StageDError("Stage-D durable protocol changed: {}".format(key))
    if protocol.get("controller_recomputed_worker_science") is not False:
        raise StageDError("Stage-D controller recomputed science")
    if payload.get("selected_configuration") is not None:
        raise StageDError("Stage-D summary selected a configuration")
    if payload.get("scientific_status") == "READY":
        if not isinstance(payload.get("train_only_recommendation"), Mapping):
            raise StageDError("READY summary lacks recommendation")
    elif payload.get("train_only_recommendation") is not None:
        raise StageDError("BLOCKED summary retained recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageDError("Stage-D summary re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageDError("Stage-D summary holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageDError("Stage-D summary accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageDError("Stage-D summary authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageDError("Stage-D summary crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageDError("Stage-D summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Optional[Path] = None,
    worker_a_path: Optional[Path] = None,
    worker_b_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    probe_available = bool(probe_path is not None and Path(probe_path).is_file())
    worker_a_available = bool(worker_a_path is not None and Path(worker_a_path).is_file())
    worker_b_available = bool(worker_b_path is not None and Path(worker_b_path).is_file())
    any_worker = worker_a_available or worker_b_available
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_staged_execution_contract_failed",
        "required_next_path": (
            "FINALIZE_R259_STAGED_DURABLE_WORKER_EVIDENCE_WITHOUT_SCIENCE_REEXECUTION"
            if any_worker
            else "DESIGN_ADD_ONLY_R259_STAGED_EXECUTION_RECOVERY_BEFORE_SCIENCE"
        ),
        "primary_failure_locus": (
            "controller_after_durable_worker_evidence"
            if any_worker
            else "staged_execution_contract"
        ),
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "durable_probe_evidence_available": probe_available,
        "durable_worker_a_evidence_available": worker_a_available,
        "durable_worker_b_evidence_available": worker_b_available,
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
