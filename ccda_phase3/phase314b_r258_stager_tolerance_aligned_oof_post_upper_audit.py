"""Phase3.14b-r2.5.8 Stage R tolerance-aligned OOF post-upper audit.

Stage-Q Resume1 proved that the legacy strict upper-segment gate rejected a
reconstruction-tolerance population already present in the control states.  A
shadow gate expressed in the same ``upper + 5 * segment_tolerance`` length
contract cleared all aligned upper violations and restored admission for all
six raw/projected oracle controls.

Stage R reopens the previously blocked question for the deployable grouped-OOF
direction bank.  It refits exactly the frozen nine OOF backbones on the 638-row
objective-training population, verifies every feature and prediction hash
against the immutable Stage-L evidence, replays only external multiplier 0.25,
and verifies the legacy callback result against Stage-L before applying the
unchanged Stage-Q tolerance-aligned shadow gate.

No holdout or frozen probe is accessed.  No formal diffusion model, reverse
sampler, inverse dynamics model, candidate execution, DeformableRavens task,
Phase4, or CPS run is performed.  OOF models and predictions exist only in
memory and are not persisted.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PHASE = "Phase3.14b-r2.5.8 Stage R"
SCHEMA = "phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit_v1"
BLOCKED_SCHEMA = (
    "phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit_blocked_v1"
)

BASE_EVIDENCE_COMMIT = "b47d2d848ce1a4f4109f5937d9f224897e1301cb"
BASE_IMPLEMENTATION_COMMIT = "e38b7c0c0a6c468505076222fb5d61c02ed07f2b"
BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT = "d328122fb299167f9bc0c453ff7d45405d6467e7"
BASE_STAGEQ_IMPLEMENTATION_COMMIT = "8597b2da8c5a1270887d4bd5c7bd0816770ec759"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_REPORT = (
    "reports/phase3_14b_r258_stageq_resume1_tolerance_aligned_upper_gate_"
    "shadow_summary.json"
)
EXPECTED_BASE_REPORT_SHA256 = (
    "0f5383ba53f415525f69875591165c0cbfafe26469c3aba846b4a8b3e8c60f24"
)
EXPECTED_BASE_STAGEQ_PAYLOAD_SHA256 = (
    "b25e110386445a6ec62252a370ec8cc1806a6edd331da8f11bcca9f5c30463b8"
)
EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256 = (
    "41497509dda7d4025701b83c6e3c32ade62ff8bbca3343f98d72bd291e7773cb"
)
EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_stageq_tolerance_alignment_restores_dual_oracle_admission"
)
EXPECTED_BASE_NEXT_PATH = (
    "REOPEN_LOWER_MULTIPLIER_OOF_POST_UPPER_AUDIT_WITH_TOLERANCE_ALIGNED_GATE"
)
EXPECTED_BASE_STRICT_FAILURES = 619626
EXPECTED_BASE_ALIGNED_FAILURES = 0
EXPECTED_BASE_STRICT_FAIL_ALIGNED_PASS_ROWS = 26796
EXPECTED_BASE_MASK_MISMATCHES = 0
EXPECTED_BASE_ADMITTED_ORACLE_CELLS = 6

STAGEL_REPORT = (
    "reports/phase3_14b_r258_stagel_resume1_predicate_assembly_audit_summary.json"
)
EXPECTED_STAGEL_REPORT_SHA256 = (
    "120cf31f649342589e74e3a5b8ce97bb50f4aac33f865fae54ebebef02f79eed"
)
EXPECTED_STAGEL_SINGLE_RUN_SHA256 = (
    "193a6514fd0e3fac76e3cb913eaa10be90b86d400183fe348190056c30080beb"
)
EXPECTED_STAGEL_CALLBACK_PAIR_COUNT = 132

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stager_tolerance_aligned_oof_post_upper_"
    "audit_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stager_tolerance_aligned_oof_post_upper_"
    "audit_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage R: audit tolerance-aligned OOF post-upper rejection"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage R tolerance-aligned OOF evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage R blocked evidence"
)
BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage Q Resume1: restore candidate hash domains"
)
BASE_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage Q Resume1 tolerance-aligned evidence"
)
BASE_STAGEQ_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage Q: audit tolerance-aligned upper gate"
)
BASE_STAGEQ_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage Q blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit.py",
    ),
    ("A", "scripts/phase3_14b_r258_stager_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stager_tolerance_aligned_oof_post_upper_audit.py",
    ),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
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
BASE_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (("A", BASE_REPORT),)
BASE_STAGEQ_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow.py",
    ),
    ("A", "scripts/phase3_14b_r258_stageq_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow.py",
    ),
)
BASE_STAGEQ_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "reports/phase3_14b_r258_stageq_tolerance_aligned_upper_gate_shadow_"
        "blocked_summary.json",
    ),
)

LOWER_MULTIPLIER = 0.25
EXPECTED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_BACKBONE_COUNT = 9
EXPECTED_CELL_COUNT = 27
EXPECTED_OOF_FIT_COUNT = 27
EXPECTED_CALLBACK_PAIR_COUNT = 27
EXPECTED_INTERNAL_SCALE_COUNT = 7
EXPECTED_INTERNAL_ATTEMPT_COUNT = EXPECTED_CELL_COUNT * EXPECTED_INTERNAL_SCALE_COUNT
ORACLE_SOURCES: Tuple[str, ...] = ("raw_oracle", "projected_oracle")

PREDICATE_ORDER: Tuple[str, ...] = (
    "finite_state",
    "upper_segment_geometry",
    "lower_segment_geometry",
    "coordinate_recenter",
    "coordinate_geometry",
    "reconstruction_bounds",
    "segment_geometry",
    "direction_retention",
    "displacement",
    "topology",
)

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagee_constrained_integrator.py": (
        "37a9567391a308bfe804b95b4aa488ada40e82b2eb94bd03f2060be962cc9eb1"
    ),
    "ccda_phase3/phase314b_r258_stageh_candidate_descriptor_identifiability.py": (
        "5312332c8f77c7ff1c1735f60467ddd66810246188a0494940e7fc6bca9c0d78"
    ),
    "ccda_phase3/phase314b_r258_stagek_explicit_integrator_predicate_callback.py": (
        "403b4242c6bf7c018e6220bf4e3920e26a9cd945e6bf3871792d88bd7e1f3139"
    ),
    "ccda_phase3/phase314b_r258_stagel_predicate_assembly_audit.py": (
        "b6d5c3ae42ced096a715e40de7c692eb4a313c7c4a5e31ad3807773d7963fcbd"
    ),
    "ccda_phase3/phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow.py": (
        "42f7577ec15f38e5222221595fa424c3276336ac383796c67e88db382df78884"
    ),
    "ccda_phase3/phase314b_r258_stageq_resume1_candidate_hash_domain_recovery.py": (
        "e0eeb56948971decd51aeb69d33c544074ee93ccd66737cf17ce214b95578db3"
    ),
}

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


class StageRError(RuntimeError):
    """Fail-closed Stage-R error."""


@dataclass(frozen=True)
class StageRSpec:
    external_multiplier: float = LOWER_MULTIPLIER
    grouped_cv_folds: int = 6
    oracle_acceptance_rate_min: float = 0.95
    oof_rejection_rate_max: float = 0.05
    oracle_conditional_pass_rate_min: float = 0.90
    oof_conditional_pass_rate_max: float = 0.10
    oof_rejection_mass_min: float = 0.50
    oracle_rejection_mass_max: float = 0.10
    stable_cell_support_min: int = 24
    sparse_other_cell_support_max: int = 3
    stable_backbone_coverage_min: int = 8
    reconstruction_tolerance_factor: float = 5.0
    binary_tolerance: float = 1.0e-12
    candidate_motion_epsilon: float = 1.0e-12

    def validate(self) -> None:
        if float(self.external_multiplier) != LOWER_MULTIPLIER:
            raise StageRError("Stage-R external multiplier changed")
        if int(self.grouped_cv_folds) != 6:
            raise StageRError("Stage-R grouped fold count changed")
        if int(self.stable_cell_support_min) != 24:
            raise StageRError("Stage-R stable support changed")
        if int(self.sparse_other_cell_support_max) != 3:
            raise StageRError("Stage-R sparse support changed")
        if int(self.stable_backbone_coverage_min) != 8:
            raise StageRError("Stage-R backbone coverage changed")
        if float(self.reconstruction_tolerance_factor) != 5.0:
            raise StageRError("Stage-R reconstruction tolerance factor changed")
        for value in (
            self.oracle_acceptance_rate_min,
            self.oof_rejection_rate_max,
            self.oracle_conditional_pass_rate_min,
            self.oof_conditional_pass_rate_max,
            self.oof_rejection_mass_min,
            self.oracle_rejection_mass_max,
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise StageRError("Stage-R rate threshold is outside [0,1]")
        if self.binary_tolerance <= 0.0 or self.candidate_motion_epsilon <= 0.0:
            raise StageRError("Stage-R numerical tolerance is invalid")


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


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode("ascii")
        + b"|"
        + repr(tuple(array.shape)).encode("ascii")
        + b"|"
        + array.tobytes(order="C")
    )
    return sha256_bytes(payload)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageRError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageRError(f"{label} is not a sequence")
    return value


def _finite_rate(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not (0.0 <= result <= 1.0):
        raise StageRError(f"{label} is not a finite rate")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise StageRError(f"{label} is boolean")
    result = int(value)
    if result < 0 or float(result) != float(value):
        raise StageRError(f"{label} is not a nonnegative integer")
    return result


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
        raise StageRError(
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
        raise StageRError(
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
            raise StageRError("unexpected diff-tree record")
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
            raise StageRError(f"{label} boundary changed: {key}={mapping.get(key)!r}")


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageRError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageRError(f"cannot load JSON {path}: {error}") from error
    return _mapping(value, str(path))


def _validate_blob(
    root: Path,
    relative: str,
    expected_sha: str,
    commit: Optional[str] = None,
) -> Mapping[str, Any]:
    path = Path(root) / relative
    if not path.is_file():
        raise StageRError(f"required report is missing: {relative}")
    if sha256_file(path) != expected_sha:
        raise StageRError(f"report SHA changed: {relative}")
    blob_commit = "HEAD" if commit is None else commit
    if _git_bytes(root, "show", f"{blob_commit}:{relative}") != path.read_bytes():
        raise StageRError(f"report differs from committed blob: {relative}")
    return _load_json(path)


def validate_base_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageRError("Stage-Q Resume1 execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageRError("Stage-Q Resume1 scientific status changed")
    if report.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageRError("Stage-Q Resume1 root cause changed")
    if report.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageRError("Stage-Q Resume1 next path changed")
    if report.get("stage_q_result_sha256") != EXPECTED_BASE_STAGEQ_PAYLOAD_SHA256:
        raise StageRError("recovered Stage-Q payload SHA changed")
    if report.get("stage_q_scientific_result_sha256") != (
        EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256
    ):
        raise StageRError("Stage-Q scientific-result SHA changed")
    stageq_result = _mapping(report.get("stage_q_result"), "Stage-Q result")
    if sha256_bytes(stable_json_bytes(stageq_result)) != (
        EXPECTED_BASE_STAGEQ_PAYLOAD_SHA256
    ):
        raise StageRError("Stage-Q payload bytes differ from frozen SHA")
    if stageq_result.get("scientific_result_sha256") != (
        EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256
    ):
        raise StageRError("nested Stage-Q scientific SHA changed")
    if stageq_result.get("selected_configuration") is not None:
        raise StageRError("Stage-Q selected configuration is not null")
    if stageq_result.get("train_only_recommendation") is not None:
        raise StageRError("Stage-Q recommendation is not null")
    shadow = _mapping(
        stageq_result.get("tolerance_aligned_upper_gate_shadow"),
        "Stage-Q shadow",
    )
    classification = _mapping(shadow.get("classification"), "Stage-Q classification")
    checks = {
        "cell_count": int(classification.get("cell_count", -1))
        == EXPECTED_BASE_ADMITTED_ORACLE_CELLS,
        "strict_failures": int(
            classification.get("strict_upper_element_failure_count", -1)
        )
        == EXPECTED_BASE_STRICT_FAILURES,
        "aligned_failures": int(
            classification.get("aligned_upper_element_failure_count", -1)
        )
        == EXPECTED_BASE_ALIGNED_FAILURES,
        "strict_fail_aligned_pass": int(
            classification.get("strict_fail_aligned_pass_row_count", -1)
        )
        == EXPECTED_BASE_STRICT_FAIL_ALIGNED_PASS_ROWS,
        "strict_pass_aligned_fail": int(
            classification.get("strict_pass_aligned_fail_row_count", -1)
        )
        == 0,
        "mask_mismatch": int(
            classification.get("length_log_z_element_mismatch_count", -1)
        )
        == EXPECTED_BASE_MASK_MISMATCHES,
        "oracle_admitted": int(
            classification.get("aligned_oracle_admitted_cell_count", -1)
        )
        == EXPECTED_BASE_ADMITTED_ORACLE_CELLS,
        "callback_pairs": int(shadow.get("legacy_callback_off_on_pair_count", -1))
        == EXPECTED_BASE_ADMITTED_ORACLE_CELLS,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageRError(f"Stage-Q frozen result changed: {failed}")
    require_false(stageq_result, FALSE_BOUNDARIES, "Stage-Q result")
    return stageq_result


def extract_stagel_scientific_result(
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageRError("Stage-L Resume1 execution is not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageRError("Stage-L Resume1 scientific status changed")
    recovery = _mapping(report.get("recovery_contract"), "Stage-L recovery contract")
    if recovery.get("underlying_stage_l_single_run_result_sha256") != (
        EXPECTED_STAGEL_SINGLE_RUN_SHA256
    ):
        raise StageRError("Stage-L single-run SHA changed")
    if int(recovery.get("underlying_callback_pair_count", -1)) != (
        EXPECTED_STAGEL_CALLBACK_PAIR_COUNT
    ):
        raise StageRError("Stage-L callback-pair count changed")
    if recovery.get("stage_l_science_modified") is not False:
        raise StageRError("Stage-L science was modified")
    stage_l_result = _mapping(report.get("stage_l_result"), "Stage-L result")
    scientific = _mapping(stage_l_result.get("scientific_result"), "Stage-L science")
    audit = _mapping(scientific.get("predicate_assembly_audit"), "Stage-L audit")
    if int(audit.get("callback_off_on_pair_count", -1)) != (
        EXPECTED_STAGEL_CALLBACK_PAIR_COUNT
    ):
        raise StageRError("Stage-L callback population changed")
    if audit.get("all_callback_results_bit_exact") is not True:
        raise StageRError("Stage-L callback identity changed")
    records = _sequence(audit.get("oof_records"), "Stage-L OOF records")
    if len(records) != EXPECTED_CELL_COUNT:
        raise StageRError("Stage-L OOF record population changed")
    require_false(scientific, FALSE_BOUNDARIES, "Stage-L scientific result")
    return scientific


def oracle_control_map(
    stageq_result: Mapping[str, Any],
) -> Mapping[Tuple[str, int], Mapping[str, Any]]:
    shadow = _mapping(
        stageq_result.get("tolerance_aligned_upper_gate_shadow"),
        "Stage-Q shadow",
    )
    raw_cells = _sequence(shadow.get("cell_records"), "Stage-Q cells")
    output: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_cells):
        cell = _mapping(raw, f"Stage-Q cell {index}")
        key = (str(cell.get("source_id")), int(cell.get("timestep")))
        if key in output:
            raise StageRError("duplicate Stage-Q oracle cell")
        if cell.get("aligned_oracle_admitted") is not True:
            raise StageRError("Stage-Q oracle control is not admitted")
        if _finite_rate(
            cell.get("aligned_oracle_acceptance_rate"),
            "oracle acceptance rate",
        ) < 0.95:
            raise StageRError("Stage-Q oracle acceptance fell below 0.95")
        output[key] = cell
    expected = {
        (source, timestep)
        for source in ORACLE_SOURCES
        for timestep in EXPECTED_TIMESTEPS
    }
    if set(output) != expected:
        raise StageRError("Stage-Q oracle cell population changed")
    return output


def stagel_oof_record_map(
    scientific: Mapping[str, Any],
) -> Mapping[Tuple[str, int], Mapping[str, Any]]:
    audit = _mapping(scientific.get("predicate_assembly_audit"), "Stage-L audit")
    raw_records = _sequence(audit.get("oof_records"), "Stage-L OOF records")
    output: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    backbones: set[str] = set()
    for index, raw in enumerate(raw_records):
        record = _mapping(raw, f"Stage-L OOF record {index}")
        key = (str(record.get("base_direction_id")), int(record.get("timestep")))
        if key in output:
            raise StageRError("duplicate Stage-L OOF record")
        backbones.add(key[0])
        output[key] = record
    if len(output) != EXPECTED_CELL_COUNT or len(backbones) != EXPECTED_BACKBONE_COUNT:
        raise StageRError("Stage-L backbone/timestep population changed")
    if {key[1] for key in output} != set(EXPECTED_TIMESTEPS):
        raise StageRError("Stage-L timestep population changed")
    return output


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageRError("Stage-R requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_EVIDENCE_COMMIT:
        raise StageRError("Stage-R implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageRError("Stage-R implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageRError("Stage-R implementation paths changed")

    if _git(repo, "rev-parse", f"{BASE_EVIDENCE_COMMIT}^") != BASE_IMPLEMENTATION_COMMIT:
        raise StageRError("Stage-Q Resume1 evidence parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION_COMMIT) != (
        BASE_IMPLEMENTATION_SUBJECT
    ):
        raise StageRError("Stage-Q Resume1 implementation subject changed")
    if commit_name_status(repo, BASE_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_IMPLEMENTATION_PATHS)
    ):
        raise StageRError("Stage-Q Resume1 implementation paths changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_EVIDENCE_COMMIT) != (
        BASE_EVIDENCE_SUBJECT
    ):
        raise StageRError("Stage-Q Resume1 evidence subject changed")
    if commit_name_status(repo, BASE_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_EVIDENCE_PATHS)
    ):
        raise StageRError("Stage-Q Resume1 evidence paths changed")

    if _git(repo, "rev-parse", f"{BASE_IMPLEMENTATION_COMMIT}^") != (
        BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT
    ):
        raise StageRError("Stage-Q Resume1 provenance changed")
    if _git(repo, "rev-parse", f"{BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGEQ_IMPLEMENTATION_COMMIT
    ):
        raise StageRError("Stage-Q blocked provenance changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGEQ_IMPLEMENTATION_COMMIT,
    ) != BASE_STAGEQ_IMPLEMENTATION_SUBJECT:
        raise StageRError("Stage-Q implementation subject changed")
    if commit_name_status(repo, BASE_STAGEQ_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEQ_IMPLEMENTATION_PATHS)
    ):
        raise StageRError("Stage-Q implementation paths changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT,
    ) != BASE_STAGEQ_BLOCKED_SUBJECT:
        raise StageRError("Stage-Q blocked evidence subject changed")
    if commit_name_status(repo, BASE_STAGEQ_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_STAGEQ_BLOCKED_PATHS)
    ):
        raise StageRError("Stage-Q blocked evidence paths changed")

    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageRError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageRError("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageRError("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageRError("Stage-R worktree must be clean before execution")

    source_sha: Dict[str, str] = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file():
            raise StageRError(f"frozen source is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise StageRError(f"frozen source SHA changed: {relative}")
        source_sha[relative] = actual

    base_report = _validate_blob(
        repo,
        BASE_REPORT,
        EXPECTED_BASE_REPORT_SHA256,
        BASE_EVIDENCE_COMMIT,
    )
    stageq_result = validate_base_report(base_report)
    stagel_report = _validate_blob(
        repo,
        STAGEL_REPORT,
        EXPECTED_STAGEL_REPORT_SHA256,
    )
    stagel_scientific = extract_stagel_scientific_result(stagel_report)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageRError(f"Stage-R output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "frozen_source_sha256": source_sha,
        "base_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "base_stageq_payload_sha256": EXPECTED_BASE_STAGEQ_PAYLOAD_SHA256,
        "base_scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256,
        "stagel_report_sha256": EXPECTED_STAGEL_REPORT_SHA256,
        "oracle_control_cell_count": len(oracle_control_map(stageq_result)),
        "stagel_oof_record_count": len(stagel_oof_record_map(stagel_scientific)),
    }


def selected_scale_histogram(selected: np.ndarray) -> Mapping[str, int]:
    array = np.asarray(selected, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise StageRError("selected scale contains invalid values")
    output: Dict[str, int] = {}
    for value in array:
        key = _scale_key(float(value))
        output[key] = output.get(key, 0) + 1
    if sum(output.values()) != int(array.size):
        raise StageRError("selected-scale histogram does not close")
    return dict(sorted(output.items(), key=lambda item: float(item[0])))


def aggregate_attempt_records(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not records:
        raise StageRError("cannot aggregate an empty attempt population")
    reached = {name: 0 for name in PREDICATE_ORDER}
    passed = {name: 0 for name in PREDICATE_ORDER}
    failed = {name: 0 for name in PREDICATE_ORDER}
    active_total = 0
    accepted_total = 0
    for index, raw in enumerate(records):
        record = _mapping(raw, f"attempt record {index}")
        active = _nonnegative_int(
            record.get("shadow_active_row_count"),
            "shadow active rows",
        )
        accepted = _nonnegative_int(
            record.get("shadow_accepted_count"),
            "shadow accepted rows",
        )
        failures = _mapping(
            record.get("shadow_first_failed_counts"),
            "shadow first-failure counts",
        )
        if accepted > active:
            raise StageRError("accepted rows exceed active rows")
        active_total += active
        accepted_total += accepted
        unresolved = active
        for predicate in PREDICATE_ORDER:
            failure = _nonnegative_int(
                failures.get(predicate),
                f"first failure {predicate}",
            )
            if failure > unresolved:
                raise StageRError("first-failure count exceeds sequential reach")
            reached[predicate] += unresolved
            failed[predicate] += failure
            unresolved -= failure
            passed[predicate] += unresolved
        if unresolved != accepted:
            raise StageRError("attempt population does not close to accepted rows")
    rejected_total = active_total - accepted_total
    conditional_pass = {
        key: 1.0 if reached[key] == 0 else float(passed[key] / reached[key])
        for key in PREDICATE_ORDER
    }
    conditional_failure = {
        key: 0.0 if reached[key] == 0 else float(failed[key] / reached[key])
        for key in PREDICATE_ORDER
    }
    rejection_mass = {
        key: 0.0 if rejected_total == 0 else float(failed[key] / rejected_total)
        for key in PREDICATE_ORDER
    }
    dominant = max(PREDICATE_ORDER, key=lambda key: failed[key])
    dominant_count = int(failed[dominant])
    return {
        "attempt_count": len(records),
        "active_row_attempt_count": active_total,
        "accepted_row_attempt_count": accepted_total,
        "rejected_row_attempt_count": rejected_total,
        "sequential_reach_counts": reached,
        "sequential_pass_counts": passed,
        "first_failed_counts": failed,
        "conditional_pass_rates": conditional_pass,
        "conditional_failure_rates": conditional_failure,
        "rejection_mass_rates": rejection_mass,
        "dominant_failure_predicate": None if dominant_count == 0 else dominant,
        "dominant_failure_count": dominant_count,
        "dominant_failure_mass": (
            0.0 if rejected_total == 0 else float(dominant_count / rejected_total)
        ),
        "sequential_population_closed": True,
    }


def compare_with_oracle(
    *,
    oof: Mapping[str, Any],
    oracle: Mapping[str, Any],
    spec: StageRSpec,
) -> Mapping[str, Any]:
    comparisons: List[Mapping[str, Any]] = []
    discriminator: Optional[str] = None
    discriminator_mode: Optional[str] = None
    for predicate in PREDICATE_ORDER:
        oof_conditional = _finite_rate(
            _mapping(oof.get("conditional_pass_rates"), "OOF conditional rates").get(
                predicate
            ),
            f"OOF conditional pass {predicate}",
        )
        oracle_conditional = _finite_rate(
            _mapping(
                oracle.get("conditional_pass_rates"),
                "oracle conditional rates",
            ).get(predicate),
            f"oracle conditional pass {predicate}",
        )
        oof_mass = _finite_rate(
            _mapping(oof.get("rejection_mass_rates"), "OOF rejection mass").get(
                predicate
            ),
            f"OOF rejection mass {predicate}",
        )
        oracle_mass = _finite_rate(
            _mapping(
                oracle.get("rejection_mass_rates"),
                "oracle rejection mass",
            ).get(predicate),
            f"oracle rejection mass {predicate}",
        )
        conditional_discriminator = (
            oracle_conditional >= spec.oracle_conditional_pass_rate_min
            and oof_conditional <= spec.oof_conditional_pass_rate_max
        )
        rejection_mass_discriminator = (
            oof_mass >= spec.oof_rejection_mass_min
            and oracle_mass <= spec.oracle_rejection_mass_max
        )
        comparisons.append(
            {
                "predicate": predicate,
                "oof_conditional_pass_rate": oof_conditional,
                "oracle_conditional_pass_rate": oracle_conditional,
                "oof_rejection_mass_rate": oof_mass,
                "oracle_rejection_mass_rate": oracle_mass,
                "conditional_discriminator": conditional_discriminator,
                "rejection_mass_discriminator": rejection_mass_discriminator,
            }
        )
        if discriminator is None and (
            conditional_discriminator or rejection_mass_discriminator
        ):
            discriminator = predicate
            if conditional_discriminator and rejection_mass_discriminator:
                discriminator_mode = "conditional_and_rejection_mass"
            elif conditional_discriminator:
                discriminator_mode = "conditional_pass"
            else:
                discriminator_mode = "rejection_mass"
    return {
        "discriminator_predicate": discriminator,
        "discriminator_mode": discriminator_mode,
        "predicate_comparisons": comparisons,
    }


def _persisted_multiplier_record(
    audit: Mapping[str, Any],
    multiplier: float,
) -> Mapping[str, Any]:
    values = _sequence(audit.get("multiplier_records"), "multiplier records")
    matches = [
        _mapping(value, f"multiplier record {index}")
        for index, value in enumerate(values)
        if float(_mapping(value, f"multiplier record {index}").get("scale_multiplier"))
        == float(multiplier)
    ]
    if len(matches) != 1:
        raise StageRError("persisted multiplier-record population changed")
    return matches[0]


def validate_legacy_oof_identity(
    *,
    base_direction_id: str,
    timestep: int,
    feature_sha256: str,
    prediction_sha256: str,
    proposed_direction: np.ndarray,
    integration: Mapping[str, Any],
    capture: Mapping[str, Any],
    persisted_record: Mapping[str, Any],
    stagel: Any,
    stagek: Any,
    spec: StageRSpec,
    control: np.ndarray,
) -> Mapping[str, Any]:
    if str(persisted_record.get("base_direction_id")) != base_direction_id:
        raise StageRError("Stage-L backbone identity changed")
    if int(persisted_record.get("timestep")) != int(timestep):
        raise StageRError("Stage-L timestep identity changed")
    if str(persisted_record.get("feature_sha256")) != feature_sha256:
        raise StageRError("Stage-R feature hash differs from Stage-L")
    if str(persisted_record.get("oof_prediction_sha256")) != prediction_sha256:
        raise StageRError("Stage-R OOF prediction differs from Stage-L")

    persisted_audit = _mapping(
        persisted_record.get("oof_predicate_assembly"),
        "persisted OOF predicate assembly",
    )
    expected = _persisted_multiplier_record(persisted_audit, LOWER_MULTIPLIER)
    selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(
        np.asarray(control).shape[0]
    )
    candidate = np.asarray(integration["candidate"], dtype=np.float32)
    motion = stagek._row_norm(
        candidate.astype(np.float64) - np.asarray(control, dtype=np.float32).astype(np.float64)
    )
    selected_positive = selected > spec.binary_tolerance
    motion_positive = motion > spec.candidate_motion_epsilon
    if np.any(selected_positive != motion_positive):
        raise StageRError("selected-scale and candidate-motion observables disagree")
    attempts = [
        event
        for event in _sequence(capture.get("events"), "callback events")
        if isinstance(event, Mapping) and event.get("event_type") == "scale_attempt"
    ]
    if not attempts:
        raise StageRError("callback capture lacks scale attempts")
    observed_assembly = stagel.aggregate_sequential_attempts(attempts)
    checks = {
        "feature_sha256": True,
        "prediction_sha256": True,
        "proposed_direction_sha256": stagel.sha256_array(proposed_direction)
        == expected.get("proposed_direction_sha256"),
        "selected_scale_positive_rate": float(np.mean(selected_positive))
        == float(expected.get("selected_scale_positive_rate")),
        "selected_scale_sha256": stagel.sha256_array(selected)
        == expected.get("selected_scale_sha256"),
        "candidate_motion_positive_rate": float(np.mean(motion_positive))
        == float(expected.get("candidate_motion_positive_rate")),
        "candidate_sha256": stagel.sha256_array(candidate)
        == expected.get("candidate_sha256"),
        "callback_capture_sha256": str(capture.get("events_sha256"))
        == str(expected.get("callback_capture_sha256")),
        "callback_result_bit_exact": capture.get("returned_result_bit_exact") is True
        and expected.get("callback_result_bit_exact") is True,
        "assembly": stable_json_bytes(observed_assembly)
        == stable_json_bytes(expected.get("assembly")),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise StageRError(f"Stage-R legacy replay differs from Stage-L: {failed}")
    return {
        "all_exact": True,
        "checks": checks,
        "persisted_multiplier": LOWER_MULTIPLIER,
        "persisted_callback_capture_sha256": expected.get("callback_capture_sha256"),
        "persisted_candidate_sha256": expected.get("candidate_sha256"),
        "persisted_selected_scale_sha256": expected.get("selected_scale_sha256"),
        "legacy_assembly": observed_assembly,
    }


def audit_oof_shadow_cell(
    *,
    base_direction_id: str,
    timestep: int,
    feature_mode: str,
    feature_sha256: str,
    prediction_sha256: str,
    control: np.ndarray,
    base_direction: np.ndarray,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    persisted_record: Mapping[str, Any],
    oracle_cells: Mapping[Tuple[str, int], Mapping[str, Any]],
    spec: StageRSpec,
) -> Mapping[str, Any]:
    stageq = runtime["stageq"]
    stagel = runtime["stagel"]
    stagek = runtime["stagek"]
    stagee = stagel.stagee258
    stageb = stagee.stageb
    staged = stagee.staged
    stagef = runtime["stagef"]
    integrator_spec = runtime["integrator_spec"]
    callback_spec = runtime["callback_spec"]
    definition = stagef.fixed_integrator_definition()
    definition.validate()

    control_raw = np.asarray(control, dtype=np.float32)
    direction_raw = np.asarray(base_direction, dtype=np.float64)
    if control_raw.shape != direction_raw.shape:
        raise StageRError("control/OOF direction shape changed")
    proposed_direction = direction_raw * float(spec.external_multiplier)
    integration, capture = stagek.callback_integrate_rowwise(
        control=control_raw,
        direction=proposed_direction,
        context=context,
        integrator_spec=integrator_spec,
        callback_spec=callback_spec,
    )
    legacy_identity = validate_legacy_oof_identity(
        base_direction_id=base_direction_id,
        timestep=timestep,
        feature_sha256=feature_sha256,
        prediction_sha256=prediction_sha256,
        proposed_direction=proposed_direction,
        integration=integration,
        capture=capture,
        persisted_record=persisted_record,
        stagel=stagel,
        stagek=stagek,
        spec=spec,
        control=control_raw,
    )

    attempts = [
        _mapping(value, f"callback event {index}")
        for index, value in enumerate(
            _sequence(capture.get("events"), "callback events")
        )
        if isinstance(value, Mapping) and value.get("event_type") == "scale_attempt"
    ]
    expected_order = stageq._expected_internal_order(runtime)
    observed_order = tuple(float(item.get("attempted_scale")) for item in attempts)
    if observed_order != expected_order:
        raise StageRError("callback attempt order differs from frozen Stage-E order")
    if len(expected_order) != EXPECTED_INTERNAL_SCALE_COUNT:
        raise StageRError("internal-scale population changed")

    legacy_selected_scale = np.asarray(
        integration["selected_scale"], dtype=np.float64
    ).reshape(control_raw.shape[0])
    legacy_scale_candidates = stageq._legacy_scale_candidate_map(integration)
    bounds = stagee.segment_bounds(definition=definition, context=context)
    upper_bound = np.asarray(bounds["upper"], dtype=np.float64)
    lower_bound = np.asarray(bounds["lower"], dtype=np.float64)
    reference = context["stage_d_contract"].reference
    tolerance_contract = stageq.tolerant_upper_contract(
        upper_bound=upper_bound,
        center_log=np.asarray(reference.center_log, dtype=np.float64),
        scale_log=np.asarray(reference.scale_log, dtype=np.float64),
        segment_tolerance=float(integrator_spec.segment_tolerance),
        tolerance_factor=float(spec.reconstruction_tolerance_factor),
    )
    strict_threshold = float(context["upper_gate"].upper_threshold)

    control_lengths = stageb.segment_lengths(control_raw).astype(np.float64)
    control_scores = staged.segment_scores(control_raw, reference)
    control_aligned = stageq.aligned_upper_masks(
        lengths=control_lengths,
        z_scores=np.asarray(control_scores["z"], dtype=np.float64),
        contract=tolerance_contract,
    )
    control_strict = (
        np.asarray(control_scores["upper"], dtype=np.float64)[:, 0]
        <= strict_threshold
    )

    shadow_selected = np.zeros(control_raw.shape[0], dtype=np.bool_)
    shadow_selected_scale = np.zeros(control_raw.shape[0], dtype=np.float64)
    shadow_output = control_raw.copy()
    attempt_records: List[Mapping[str, Any]] = []
    strict_element_failures = 0
    aligned_element_failures = 0
    mismatch_count = 0
    strict_pass_aligned_fail_rows = 0
    strict_fail_aligned_pass_rows = 0

    for event, scale in zip(attempts, expected_order):
        raw_proposal = (
            control_raw.astype(np.float64) + float(scale) * proposed_direction
        ).astype(np.float32)
        reconstruction = stagee.reconstruct_segment_vectors(
            proposed=raw_proposal,
            control=control_raw,
            lower=lower_bound,
            upper=upper_bound,
            coordinate_abs_max=float(
                context["historical_geometry"].coordinate_abs_max
            ),
            epsilon=float(integrator_spec.segment_tolerance),
        )
        candidate = np.asarray(reconstruction["candidate"], dtype=np.float32)
        bound_pass = np.asarray(reconstruction["bound_pass"], dtype=np.bool_)
        coordinate_possible = np.asarray(
            reconstruction["coordinate_possible"], dtype=np.bool_
        )
        stagee_candidate_sha = stagee.sha256_array(candidate)
        scale_key = _scale_key(scale)
        legacy_scale = _mapping(
            legacy_scale_candidates.get(scale_key),
            f"legacy scale candidate {scale_key}",
        )
        if stagee_candidate_sha != legacy_scale.get("candidate_sha256"):
            raise StageRError("candidate generation differs from frozen Stage-E replay")

        metrics = stagee.observable_row_metrics(
            candidate=candidate,
            raw_proposal=raw_proposal,
            control=control_raw,
            direction=proposed_direction,
            definition=definition,
            context=context,
            spec=integrator_spec,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
            include_topology=False,
        )
        candidate_lengths = stageb.segment_lengths(candidate).astype(np.float64)
        candidate_scores = staged.segment_scores(candidate, reference)
        aligned = stageq.aligned_upper_masks(
            lengths=candidate_lengths,
            z_scores=np.asarray(candidate_scores["z"], dtype=np.float64),
            contract=tolerance_contract,
        )
        aligned_upper_pass = np.asarray(
            aligned["length_row_pass"], dtype=np.bool_
        )
        strict_upper_pass = np.asarray(metrics["upper_pass"], dtype=np.bool_)
        strict_element_pass = (
            np.asarray(candidate_scores["z"], dtype=np.float64)[:, 0]
            <= strict_threshold
        )
        aligned_element_pass = np.asarray(
            aligned["length_element_pass"], dtype=np.bool_
        )
        strict_element_failure_count = int(np.count_nonzero(~strict_element_pass))
        aligned_element_failure_count = int(np.count_nonzero(~aligned_element_pass))
        strict_element_failures += strict_element_failure_count
        aligned_element_failures += aligned_element_failure_count
        mismatch_count += int(aligned["element_mismatch_count"])
        strict_pass_aligned_fail = strict_upper_pass & ~aligned_upper_pass
        strict_fail_aligned_pass = ~strict_upper_pass & aligned_upper_pass
        strict_pass_aligned_fail_rows += int(np.count_nonzero(strict_pass_aligned_fail))
        strict_fail_aligned_pass_rows += int(np.count_nonzero(strict_fail_aligned_pass))

        legacy_selected_before = legacy_selected_scale > float(scale)
        callback_failures = _mapping(
            event.get("first_failed_counts"), "callback first-failed counts"
        )
        legacy_direct_upper_fail = (
            (~legacy_selected_before)
            & np.asarray(metrics["finite"], dtype=np.bool_)
            & ~strict_upper_pass
        )
        if int(callback_failures.get("upper_segment_geometry", -1)) != int(
            np.count_nonzero(legacy_direct_upper_fail)
        ):
            raise StageRError("legacy callback upper first-failure count changed")

        active = ~shadow_selected
        masks = stageq._predicate_masks(
            metrics=metrics,
            aligned_upper_pass=aligned_upper_pass,
            bound_pass=bound_pass,
            coordinate_possible=coordinate_possible,
        )
        fast = active.copy()
        for name in PREDICATE_ORDER[:-1]:
            fast &= np.asarray(masks[name], dtype=np.bool_)
        topology_pass = np.zeros(control_raw.shape[0], dtype=np.bool_)
        checked = np.flatnonzero(fast)
        if checked.size:
            topology_pass[checked] = stageb.physical_validity(
                candidate[checked, None],
                context["historical_geometry"],
            )["topology"][:, 0]
        sequential = stageq.sequential_attempt_summary(
            active=active,
            predicate_masks=masks,
            topology_pass=topology_pass,
        )
        choose = np.asarray(sequential["accepted_mask"], dtype=np.bool_)
        if not np.array_equal(choose, fast & topology_pass):
            raise StageRError("shadow topology/acceptance decomposition changed")
        shadow_output[choose] = candidate[choose]
        shadow_selected_scale[choose] = float(scale)
        shadow_selected[choose] = True

        attempt_records.append(
            {
                "internal_scale": float(scale),
                "effective_multiplier": float(scale * spec.external_multiplier),
                "stagee_candidate_sha256": stagee_candidate_sha,
                "legacy_active_row_count": int(
                    np.count_nonzero(~legacy_selected_before)
                ),
                "shadow_active_row_count": int(np.count_nonzero(active)),
                "legacy_strict_upper_row_pass_count": int(
                    np.count_nonzero(strict_upper_pass & ~legacy_selected_before)
                ),
                "shadow_aligned_upper_row_pass_count": int(
                    np.count_nonzero(aligned_upper_pass & active)
                ),
                "strict_upper_element_failure_count": strict_element_failure_count,
                "aligned_upper_element_failure_count": aligned_element_failure_count,
                "strict_fail_aligned_pass_row_count": int(
                    np.count_nonzero(strict_fail_aligned_pass)
                ),
                "strict_pass_aligned_fail_row_count": int(
                    np.count_nonzero(strict_pass_aligned_fail)
                ),
                "length_log_z_element_equivalence": aligned["element_equivalence"],
                "length_log_z_row_equivalence": aligned["row_equivalence"],
                "length_log_z_element_mismatch_count": aligned[
                    "element_mismatch_count"
                ],
                "length_log_z_row_mismatch_count": aligned["row_mismatch_count"],
                "shadow_topology_checked_count": int(checked.size),
                "shadow_accepted_count": int(np.count_nonzero(choose)),
                "shadow_first_failed_counts": dict(
                    sequential["first_failed_counts"]
                ),
                "shadow_population_closed": True,
                "callback_events_persisted": False,
                "candidate_tensor_persisted": False,
                "predicate_mask_persisted": False,
            }
        )

    assembly = aggregate_attempt_records(attempt_records)
    aligned_acceptance = float(np.mean(shadow_selected))
    raw_oracle = oracle_cells[("raw_oracle", int(timestep))]
    projected_oracle = oracle_cells[("projected_oracle", int(timestep))]
    raw_acceptance = _finite_rate(
        raw_oracle.get("aligned_oracle_acceptance_rate"),
        "raw oracle acceptance",
    )
    projected_acceptance = _finite_rate(
        projected_oracle.get("aligned_oracle_acceptance_rate"),
        "projected oracle acceptance",
    )
    raw_assembly = aggregate_attempt_records(
        _sequence(raw_oracle.get("attempt_records"), "raw oracle attempts")
    )
    projected_assembly = aggregate_attempt_records(
        _sequence(
            projected_oracle.get("attempt_records"),
            "projected oracle attempts",
        )
    )
    raw_comparison = compare_with_oracle(oof=assembly, oracle=raw_assembly, spec=spec)
    projected_comparison = compare_with_oracle(
        oof=assembly,
        oracle=projected_assembly,
        spec=spec,
    )
    comparator_source = (
        "raw_oracle" if raw_acceptance >= projected_acceptance else "projected_oracle"
    )
    comparator_comparison = (
        raw_comparison
        if comparator_source == "raw_oracle"
        else projected_comparison
    )
    raw_discriminator = raw_comparison.get("discriminator_predicate")
    projected_discriminator = projected_comparison.get("discriminator_predicate")
    dual_discriminator = (
        raw_discriminator
        if isinstance(raw_discriminator, str)
        and raw_discriminator == projected_discriminator
        else None
    )

    if aligned_acceptance >= spec.oracle_acceptance_rate_min:
        acceptance_locus = "oracle_like_admission"
    elif aligned_acceptance > spec.oof_rejection_rate_max:
        acceptance_locus = "partial_admission"
    else:
        acceptance_locus = "rejected"

    return {
        "base_direction_id": base_direction_id,
        "timestep": int(timestep),
        "feature_mode": feature_mode,
        "external_multiplier": float(spec.external_multiplier),
        "objective_train_rows": int(control_raw.shape[0]),
        "feature_sha256": feature_sha256,
        "oof_prediction_sha256": prediction_sha256,
        "legacy_identity": legacy_identity,
        "legacy_callback_result_bit_exact": True,
        "legacy_acceptance_rate": float(
            np.mean(legacy_selected_scale > spec.binary_tolerance)
        ),
        "aligned_acceptance_rate": aligned_acceptance,
        "aligned_acceptance_locus": acceptance_locus,
        "aligned_selected_scale_sha256": sha256_array(shadow_selected_scale),
        "aligned_selected_scale_histogram": selected_scale_histogram(
            shadow_selected_scale
        ),
        "aligned_candidate_sha256": sha256_array(shadow_output),
        "internal_scale_attempt_order": list(expected_order),
        "control_gate_comparison": {
            "strict_upper_failure_count": int(np.count_nonzero(~control_strict)),
            "aligned_upper_failure_count": int(
                np.count_nonzero(~control_aligned["length_row_pass"])
            ),
            "strict_fail_aligned_pass_count": int(
                np.count_nonzero((~control_strict) & control_aligned["length_row_pass"])
            ),
            "strict_pass_aligned_fail_count": int(
                np.count_nonzero(control_strict & ~control_aligned["length_row_pass"])
            ),
            "length_log_z_element_mismatch_count": int(
                control_aligned["element_mismatch_count"]
            ),
        },
        "strict_upper_element_failure_count": strict_element_failures,
        "aligned_upper_element_failure_count": aligned_element_failures,
        "strict_fail_aligned_pass_row_count": strict_fail_aligned_pass_rows,
        "strict_pass_aligned_fail_row_count": strict_pass_aligned_fail_rows,
        "length_log_z_element_mismatch_count": mismatch_count,
        "aligned_assembly": assembly,
        "raw_oracle_control": {
            "aligned_acceptance_rate": raw_acceptance,
            "assembly": raw_assembly,
            "comparison": raw_comparison,
        },
        "projected_oracle_control": {
            "aligned_acceptance_rate": projected_acceptance,
            "assembly": projected_assembly,
            "comparison": projected_comparison,
        },
        "comparator_source": comparator_source,
        "comparator_discriminator_predicate": comparator_comparison.get(
            "discriminator_predicate"
        ),
        "dual_oracle_discriminator_predicate": dual_discriminator,
        "oracle_discriminator_agreement": (
            isinstance(raw_discriminator, str)
            and raw_discriminator == projected_discriminator
        ),
        "attempt_records": attempt_records,
        "callback_events_persisted": False,
        "row_identity_persisted": False,
        "direction_tensor_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "predicate_mask_persisted": False,
    }


def classify_oof_post_upper(
    cells: Sequence[Mapping[str, Any]],
    spec: Optional[StageRSpec] = None,
) -> Mapping[str, Any]:
    active = StageRSpec() if spec is None else spec
    active.validate()
    if len(cells) != EXPECTED_CELL_COUNT:
        raise StageRError("Stage-R cell population changed")

    backbones = {str(cell.get("base_direction_id")) for cell in cells}
    timesteps = {int(cell.get("timestep")) for cell in cells}
    if len(backbones) != EXPECTED_BACKBONE_COUNT or timesteps != set(
        EXPECTED_TIMESTEPS
    ):
        raise StageRError("Stage-R backbone/timestep population changed")

    mask_mismatches = 0
    strict_pass_aligned_fail = 0
    aligned_upper_failures = 0
    oracle_like = 0
    meaningful_support = 0
    rejected = 0
    zero_acceptance = 0
    dual_counts: Dict[str, int] = {}
    comparator_counts: Dict[str, int] = {}
    source_disagreement_cells = 0
    predicate_backbones: Dict[str, set[str]] = {}
    predicate_timesteps: Dict[str, set[int]] = {}
    by_backbone: Dict[str, Dict[str, int]] = {key: {} for key in backbones}
    by_timestep: Dict[str, Dict[str, int]] = {
        str(value): {} for value in EXPECTED_TIMESTEPS
    }

    for cell in cells:
        backbone = str(cell.get("base_direction_id"))
        timestep = int(cell.get("timestep"))
        acceptance = _finite_rate(
            cell.get("aligned_acceptance_rate"),
            "aligned OOF acceptance",
        )
        if acceptance >= active.oracle_acceptance_rate_min:
            oracle_like += 1
        if acceptance > active.oof_rejection_rate_max:
            meaningful_support += 1
        else:
            rejected += 1
        if acceptance == 0.0:
            zero_acceptance += 1
        mask_mismatches += _nonnegative_int(
            cell.get("length_log_z_element_mismatch_count"),
            "length/log-z mismatches",
        )
        strict_pass_aligned_fail += _nonnegative_int(
            cell.get("strict_pass_aligned_fail_row_count"),
            "strict-pass aligned-fail rows",
        )
        aligned_upper_failures += _nonnegative_int(
            cell.get("aligned_upper_element_failure_count"),
            "aligned upper failures",
        )

        comparator = cell.get("comparator_discriminator_predicate")
        if isinstance(comparator, str):
            comparator_counts[comparator] = comparator_counts.get(comparator, 0) + 1
        dual = cell.get("dual_oracle_discriminator_predicate")
        if isinstance(dual, str):
            dual_counts[dual] = dual_counts.get(dual, 0) + 1
            predicate_backbones.setdefault(dual, set()).add(backbone)
            predicate_timesteps.setdefault(dual, set()).add(timestep)
            by_backbone[backbone][dual] = by_backbone[backbone].get(dual, 0) + 1
            by_timestep[str(timestep)][dual] = (
                by_timestep[str(timestep)].get(dual, 0) + 1
            )
        raw_predicate = _mapping(
            _mapping(cell.get("raw_oracle_control"), "raw oracle control").get(
                "comparison"
            ),
            "raw comparison",
        ).get("discriminator_predicate")
        projected_predicate = _mapping(
            _mapping(
                cell.get("projected_oracle_control"),
                "projected oracle control",
            ).get("comparison"),
            "projected comparison",
        ).get("discriminator_predicate")
        if raw_predicate != projected_predicate:
            source_disagreement_cells += 1

    dominant: Optional[str] = None
    dominant_support = 0
    other_max = 0
    backbone_coverage = 0
    timestep_coverage = 0
    if dual_counts:
        dominant = max(
            dual_counts,
            key=lambda key: (dual_counts[key], -PREDICATE_ORDER.index(key)),
        )
        dominant_support = int(dual_counts[dominant])
        other_max = max(
            (count for key, count in dual_counts.items() if key != dominant),
            default=0,
        )
        backbone_coverage = len(predicate_backbones.get(dominant, set()))
        timestep_coverage = len(predicate_timesteps.get(dominant, set()))

    predicate_outcomes: Mapping[str, Tuple[str, str]] = {
        "finite_state": (
            "phase314b_r258_stager_oof_rejected_by_nonfinite_state_after_upper_alignment",
            "AUDIT_OOF_FINITE_STATE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "lower_segment_geometry": (
            "phase314b_r258_stager_oof_rejected_by_lower_segment_geometry_after_upper_alignment",
            "AUDIT_OOF_LOWER_SEGMENT_GATE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "coordinate_recenter": (
            "phase314b_r258_stager_oof_rejected_by_coordinate_recenter_after_upper_alignment",
            "AUDIT_OOF_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "coordinate_geometry": (
            "phase314b_r258_stager_oof_rejected_by_coordinate_geometry_after_upper_alignment",
            "AUDIT_OOF_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "reconstruction_bounds": (
            "phase314b_r258_stager_oof_rejected_by_reconstruction_bounds_after_upper_alignment",
            "AUDIT_OOF_RECONSTRUCTION_BOUND_CLOSURE_AFTER_UPPER_TOLERANCE_ALIGNMENT",
        ),
        "segment_geometry": (
            "phase314b_r258_stager_oof_rejected_by_strict_historical_segment_geometry",
            "ALIGN_HISTORICAL_SEGMENT_BOUND_TOLERANCE_WITH_RECONSTRUCTION_CONTRACT",
        ),
        "direction_retention": (
            "phase314b_r258_stager_oof_rejected_by_direction_retention_after_upper_alignment",
            "CALIBRATE_INTEGRATOR_COMPATIBLE_DIRECTION_BASIS_ON_OBJECTIVE_TRAIN_ONLY",
        ),
        "displacement": (
            "phase314b_r258_stager_oof_rejected_by_displacement_after_upper_alignment",
            "CALIBRATE_EXECUTABLE_DIRECTION_MAGNITUDE_PARAMETERIZATION",
        ),
        "topology": (
            "phase314b_r258_stager_oof_rejected_by_topology_after_upper_alignment",
            "CALIBRATE_TOPOLOGY_COMPATIBLE_DIRECTION_SURROGATE_ON_OBJECTIVE_TRAIN_ONLY",
        ),
    }

    if mask_mismatches > 0:
        root = "phase314b_r258_stager_length_and_log_z_masks_not_equivalent_on_oof"
        next_path = "AUDIT_LENGTH_LOG_Z_UPPER_TOLERANCE_NUMERICS_ON_OOF"
        primary = "length_log_z_boolean_mismatch"
    elif strict_pass_aligned_fail > 0:
        root = "phase314b_r258_stager_tolerance_alignment_introduces_oof_upper_regression"
        next_path = "AUDIT_TOLERANCE_ALIGNED_UPPER_GATE_REGRESSION_ON_OOF"
        primary = "strict_pass_aligned_fail_regression"
    elif aligned_upper_failures > 0:
        root = "phase314b_r258_stager_aligned_upper_gate_does_not_close_oof_reconstruction"
        next_path = "AUDIT_TOLERANCE_ALIGNED_UPPER_GATE_IMPLEMENTATION_ON_OOF"
        primary = "aligned_upper_violation_remains"
    elif oracle_like == EXPECTED_CELL_COUNT:
        root = "phase314b_r258_stager_tolerance_alignment_restores_all_oof_oracle_like_admission"
        next_path = "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_OOF_CANDIDATE_MATRIX"
        primary = "all_oof_cells_oracle_like"
    elif meaningful_support >= active.stable_cell_support_min and rejected <= (
        EXPECTED_CELL_COUNT - active.stable_cell_support_min
    ):
        root = "phase314b_r258_stager_tolerance_alignment_restores_broad_oof_support"
        next_path = "CONFIRM_TOLERANCE_ALIGNED_OOF_SUPPORT_ON_OBJECTIVE_TRAIN_ONLY"
        primary = "broad_oof_support_restored"
    elif (
        dominant is not None
        and dominant_support >= active.stable_cell_support_min
        and other_max <= active.sparse_other_cell_support_max
        and backbone_coverage >= active.stable_backbone_coverage_min
        and timestep_coverage == len(EXPECTED_TIMESTEPS)
    ):
        root, next_path = predicate_outcomes[dominant]
        primary = f"post_upper_rejection::{dominant}"
    elif source_disagreement_cells > active.sparse_other_cell_support_max:
        root = "phase314b_r258_stager_post_upper_discriminator_is_oracle_source_dependent"
        next_path = "STRATIFY_TOLERANCE_ALIGNED_OOF_REJECTION_BY_ORACLE_SOURCE"
        primary = "oracle_source_dependent_discriminator"
    elif dual_counts or comparator_counts:
        root = "phase314b_r258_stager_post_upper_rejection_is_backbone_or_timestep_dependent"
        next_path = "STRATIFY_TOLERANCE_ALIGNED_OOF_REJECTION_BY_BACKBONE_TIMESTEP_AND_PREDICATE"
        primary = "heterogeneous_post_upper_rejection"
    else:
        root = "phase314b_r258_stager_scalar_post_upper_surface_insufficient"
        next_path = "ADD_READ_ONLY_ROW_COHORT_HASHES_FOR_TOLERANCE_ALIGNED_OOF_REJECTION"
        primary = "post_upper_scalar_surface_insufficient"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": primary,
        "cell_count": len(cells),
        "backbone_count": len(backbones),
        "timestep_count": len(timesteps),
        "oracle_like_admission_cell_count": oracle_like,
        "meaningful_support_cell_count": meaningful_support,
        "rejected_cell_count": rejected,
        "zero_acceptance_cell_count": zero_acceptance,
        "length_log_z_element_mismatch_count": mask_mismatches,
        "strict_pass_aligned_fail_row_count": strict_pass_aligned_fail,
        "aligned_upper_element_failure_count": aligned_upper_failures,
        "dual_oracle_discriminator_counts": dict(sorted(dual_counts.items())),
        "comparator_discriminator_counts": dict(
            sorted(comparator_counts.items())
        ),
        "oracle_source_disagreement_cell_count": source_disagreement_cells,
        "dominant_dual_oracle_discriminator": dominant,
        "dominant_discriminator_support_count": dominant_support,
        "maximum_other_discriminator_support_count": other_max,
        "dominant_discriminator_backbone_coverage": backbone_coverage,
        "dominant_discriminator_timestep_coverage": timestep_coverage,
        "discriminator_counts_by_backbone": {
            key: dict(sorted(value.items())) for key, value in sorted(by_backbone.items())
        },
        "discriminator_counts_by_timestep": {
            key: dict(sorted(value.items())) for key, value in sorted(by_timestep.items())
        },
        "classification_spec": asdict(active),
    }


def probe_environment(root: Path) -> Mapping[str, Any]:
    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )

    return stageo.probe_environment(Path(root).resolve())


def run_oof_post_upper_audit(
    *,
    root: Path,
    environment: Mapping[str, Any],
    repository: Mapping[str, Any],
    spec: Optional[StageRSpec] = None,
) -> Mapping[str, Any]:
    active = StageRSpec() if spec is None else spec
    active.validate()
    validate_environment_variables()
    repo = Path(root).resolve()

    base_report = _load_json(repo / BASE_REPORT)
    stageq_result = validate_base_report(base_report)
    oracle_cells = oracle_control_map(stageq_result)
    stagel_report = _load_json(repo / STAGEL_REPORT)
    stagel_scientific = extract_stagel_scientific_result(stagel_report)
    persisted_records = stagel_oof_record_map(stagel_scientific)

    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )
    from ccda_phase3 import (
        phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow as stageq,
    )

    runtime = dict(stageo._prepare_runtime(repo, environment))
    runtime["stageo"] = stageo
    runtime["stageq"] = stageq
    stagel = runtime["stagel"]
    stageh = stagel.stageh
    stagef = runtime["stagef"]
    context = runtime["context"]

    base_direction_ids = tuple(stagel.BASE_DIRECTION_IDS)
    if len(base_direction_ids) != EXPECTED_BACKBONE_COUNT:
        raise StageRError("Stage-R backbone count changed")
    if set(base_direction_ids) != {key[0] for key in persisted_records}:
        raise StageRError("Stage-R/Stage-L backbone population differs")

    oracle_targets: Dict[int, Mapping[str, Any]] = {}
    for timestep in EXPECTED_TIMESTEPS:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)],
            dtype=np.float32,
        )
        target = np.asarray(context["objective_target"], dtype=np.float32)
        oracle_targets[int(timestep)] = stagef.generate_projected_oracle_target(
            control=control,
            target=target,
            groups=context["objective_groups"],
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=runtime["direction_spec"],
            integrator_spec=runtime["integrator_spec"],
            spec=runtime["stagef_spec"],
        )

    cells: List[Mapping[str, Any]] = []
    for base_direction_id in base_direction_ids:
        definition = stagef.definition_by_id(base_direction_id)
        if definition.role != "selectable":
            raise StageRError("Stage-R backbone is not selectable")
        for timestep in EXPECTED_TIMESTEPS:
            t = int(timestep)
            control = np.asarray(
                context["objective_control_predictions"][t],
                dtype=np.float32,
            )
            features = stagef.build_constraint_features(
                condition=context["objective_condition"],
                control=control,
                condition_name=context["objective_condition_name"],
                feature_mode=definition.feature_mode,
                context=context,
            )
            feature_sha = stagel.sha256_array(features)
            oof = stageh.fit_oof_base_direction(
                base_direction_id=base_direction_id,
                features=features,
                control=control,
                oracle_target=oracle_targets[t],
                condition_name=context["objective_condition_name"],
                fold_assignment=context["objective_fold_assignment"],
                stagef_spec=runtime["stagef_spec"],
                grouped_cv_folds=active.grouped_cv_folds,
            )
            prediction = np.asarray(oof["prediction"], dtype=np.float64)
            prediction_sha = str(oof["prediction_sha256"])
            fold_records = _sequence(oof.get("fold_records"), "OOF fold records")
            if len(fold_records) != active.grouped_cv_folds:
                raise StageRError("OOF fold-record population changed")
            if any(
                _mapping(record, f"OOF fold {index}").get(
                    "test_target_used_for_fit"
                )
                is not False
                for index, record in enumerate(fold_records)
            ):
                raise StageRError("OOF test target leaked into fit")
            key = (base_direction_id, t)
            cell = dict(
                audit_oof_shadow_cell(
                    base_direction_id=base_direction_id,
                    timestep=t,
                    feature_mode=definition.feature_mode,
                    feature_sha256=feature_sha,
                    prediction_sha256=prediction_sha,
                    control=control,
                    base_direction=prediction,
                    context=context,
                    runtime=runtime,
                    persisted_record=persisted_records[key],
                    oracle_cells=oracle_cells,
                    spec=active,
                )
            )
            cell["oof_fold_count"] = len(fold_records)
            cell["oof_fold_records_sha256"] = sha256_bytes(
                stable_json_bytes(fold_records)
            )
            cell["all_test_targets_excluded_from_fit"] = True
            cells.append(cell)

    if len(cells) != EXPECTED_CELL_COUNT:
        raise StageRError("Stage-R OOF cell count changed")
    if sum(len(_sequence(cell.get("attempt_records"), "cell attempts")) for cell in cells) != (
        EXPECTED_INTERNAL_ATTEMPT_COUNT
    ):
        raise StageRError("Stage-R internal-attempt population changed")
    classification = classify_oof_post_upper(cells, active)

    oracle_summary = {
        f"{source}::t{timestep}": {
            "aligned_acceptance_rate": _finite_rate(
                oracle_cells[(source, timestep)].get("aligned_oracle_acceptance_rate"),
                "oracle acceptance",
            ),
            "aligned_oracle_admitted": oracle_cells[(source, timestep)].get(
                "aligned_oracle_admitted"
            )
            is True,
            "assembly": aggregate_attempt_records(
                _sequence(
                    oracle_cells[(source, timestep)].get("attempt_records"),
                    "oracle attempts",
                )
            ),
        }
        for source in ORACLE_SOURCES
        for timestep in EXPECTED_TIMESTEPS
    }

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": classification["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": copy.deepcopy(dict(repository)),
        "environment": copy.deepcopy(dict(environment)),
        "audit_spec": asdict(active),
        "base_stageq_resume1": {
            "report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "recovered_stageq_payload_sha256": EXPECTED_BASE_STAGEQ_PAYLOAD_SHA256,
            "scientific_result_sha256": EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256,
            "root_cause": EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": EXPECTED_BASE_NEXT_PATH,
            "strict_upper_element_failure_count": EXPECTED_BASE_STRICT_FAILURES,
            "aligned_upper_element_failure_count": EXPECTED_BASE_ALIGNED_FAILURES,
            "aligned_oracle_admitted_cell_count": EXPECTED_BASE_ADMITTED_ORACLE_CELLS,
        },
        "tolerance_aligned_oof_post_upper_audit": {
            "external_multiplier": LOWER_MULTIPLIER,
            "base_direction_ids": list(base_direction_ids),
            "timesteps": list(EXPECTED_TIMESTEPS),
            "grouped_cv_folds": active.grouped_cv_folds,
            "oof_fit_count": EXPECTED_OOF_FIT_COUNT,
            "oof_cell_count": len(cells),
            "legacy_callback_off_on_pair_count": len(cells),
            "internal_scale_attempt_count": EXPECTED_INTERNAL_ATTEMPT_COUNT,
            "oracle_controls_reused_from_stageq_resume1": True,
            "oracle_callback_pairs_rerun": 0,
            "all_oof_predictions_match_stagel": all(
                _mapping(
                    _mapping(cell.get("legacy_identity"), "legacy identity").get(
                        "checks"
                    ),
                    "legacy identity checks",
                ).get("feature_sha256")
                is True
                and _mapping(
                    _mapping(cell.get("legacy_identity"), "legacy identity").get(
                        "checks"
                    ),
                    "legacy identity checks",
                ).get("prediction_sha256")
                is True
                for cell in cells
            ),
            "all_legacy_025_integrations_match_stagel": all(
                _mapping(cell.get("legacy_identity"), "legacy identity").get(
                    "all_exact"
                )
                is True
                for cell in cells
            ),
            "oracle_controls": oracle_summary,
            "cell_records": cells,
            "classification": classification,
            "row_level_arrays_persisted": False,
            "callback_events_persisted": False,
            "feature_tensors_persisted": False,
            "prediction_tensors_persisted": False,
            "direction_tensors_persisted": False,
            "proposal_tensors_persisted": False,
            "candidate_tensors_persisted": False,
            "predicate_masks_persisted": False,
        },
        "immutable_inputs": {
            "stageq_resume1_report_sha256": EXPECTED_BASE_REPORT_SHA256,
            "stageq_resume1_scientific_result_sha256": (
                EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256
            ),
            "stagel_resume1_report_sha256": EXPECTED_STAGEL_REPORT_SHA256,
            "stagel_single_run_sha256": EXPECTED_STAGEL_SINGLE_RUN_SHA256,
            "stage_l_132_callback_pairs_rerun": False,
            "oof_surrogate_refit_for_identity_replay": True,
            "oof_surrogate_fit_count": EXPECTED_OOF_FIT_COUNT,
            "oof_prediction_identity_checked_against_stagel": True,
            "oracle_callback_pairs_rerun": 0,
            "oof_callback_pairs_run": EXPECTED_CALLBACK_PAIR_COUNT,
        },
        "mechanism_boundary": {
            "stagee_modified": False,
            "stageh_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stageq_modified": False,
            "stageq_resume1_modified": False,
            "legacy_upper_gate_changed": False,
            "aligned_upper_gate_changed": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "oof_backbone_population_changed": False,
            "oof_features_changed": False,
            "oof_predictions_changed": False,
            "objective_train_only_oof_replay": True,
            "formal_model_training": False,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stager_oof_post_upper_execution_failed",
        "required_next_path": "RESTORE_STAGER_TOLERANCE_ALIGNED_OOF_POST_UPPER_AUDIT",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stageq_resume1_report_preserved": True,
        "stageq_resume1_report_sha256": EXPECTED_BASE_REPORT_SHA256,
        "stagel_report_preserved": True,
        "stagel_report_sha256": EXPECTED_STAGEL_REPORT_SHA256,
        "oof_surrogate_weights_persisted": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
