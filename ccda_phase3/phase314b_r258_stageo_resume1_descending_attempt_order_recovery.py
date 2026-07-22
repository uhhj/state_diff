"""Phase3.14b-r2.5.8 Stage O Resume1 descending attempt-order recovery.

Repairs only the Stage-O validator that incorrectly required increasing
internal scales. Frozen Stage-E/Stage-K emit scale attempts in descending order.
The corrected function is installed only around one unchanged Stage-O
``run_targeted_replay`` call and is restored in ``finally``.
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage O Resume1"
SCHEMA = "phase314b_r258_stageo_resume1_descending_attempt_order_recovery_v1"
BLOCKED_SCHEMA = "phase314b_r258_stageo_resume1_descending_attempt_order_recovery_blocked_v1"

BASE_STAGEO_IMPLEMENTATION_COMMIT = "ff4b4e1b94a095bf84e914af3515a3fd66eff8a1"
BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT = "8dd4a129708db0141d3d328e39270f9b4b8d8ebc"
EXPECTED_STAGEO_IMPLEMENTATION_PARENT = "9fb03b578bce6f680043f0ff38d0c20592713e6a"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEO_BLOCKED_REPORT = "reports/phase3_14b_r258_stageo_dual_oracle_lower_multiplier_admission_blocked_summary.json"
EXPECTED_STAGEO_BLOCKED_REPORT_SHA256 = "9d3368304d3217e420b17564cc7aa103884ef970ba20916937b4f397c29167ad"
STAGEO_SOURCE = "ccda_phase3/phase314b_r258_stageo_dual_oracle_lower_multiplier_admission.py"
EXPECTED_STAGEO_SOURCE_SHA256 = "7cdceb909eb599ccd04a7e907940a2ed5b32fd38ae115682ef1bac85f2aa411e"

SUCCESS_REPORT = "reports/phase3_14b_r258_stageo_resume1_dual_oracle_lower_multiplier_admission_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stageo_resume1_dual_oracle_lower_multiplier_admission_blocked_summary.json"

IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage O Resume1: restore descending attempt order"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage O Resume1 dual-oracle evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage O Resume1 blocked evidence"
ORIGINAL_IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage O: audit dual-oracle lower-multiplier admission"
ORIGINAL_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage O blocked evidence"

IMPLEMENTATION_PATHS = (
    ("A", "ccda_phase3/phase314b_r258_stageo_resume1_descending_attempt_order_recovery.py"),
    ("A", "scripts/phase3_14b_r258_stageo_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stageo_resume1_descending_attempt_order_recovery.py"),
)
ORIGINAL_IMPLEMENTATION_PATHS = (
    ("A", STAGEO_SOURCE),
    ("A", "scripts/phase3_14b_r258_stageo_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stageo_dual_oracle_lower_multiplier_admission.py"),
)
ORIGINAL_BLOCKED_PATHS = (("A", STAGEO_BLOCKED_REPORT),)

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

FALSE_BOUNDARIES = (
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

class StageOResume1Error(RuntimeError):
    pass

def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False
    ).encode("utf-8")

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        raise StageOResume1Error(
            "git {} failed: {}".format(" ".join(args), completed.stderr.strip())
        )
    return completed.stdout.strip()

def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=str(root),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if completed.returncode != 0:
        raise StageOResume1Error(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace")
            )
        )
    return completed.stdout

def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageOResume1Error("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))

def status_paths(root: Path) -> Tuple[str, ...]:
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    values = []
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
            raise StageOResume1Error(
                "{} boundary changed: {}={!r}".format(label, key, mapping.get(key))
            )

def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageOResume1Error(
            "deterministic environment mismatch: {}".format(mismatch)
        )
    return dict(EXPECTED_ENV)

def expected_internal_scale_attempt_order(
    *, scale_grid: Sequence[float], maximum_scale: float,
    standardizer_epsilon: float
) -> Tuple[float, ...]:
    maximum = float(maximum_scale)
    epsilon = float(standardizer_epsilon)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise StageOResume1Error("fixed integrator maximum scale is invalid")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise StageOResume1Error("integrator standardizer epsilon is invalid")
    eligible: List[float] = []
    for raw in scale_grid:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise StageOResume1Error("integrator scale grid contains an invalid value")
        if value <= maximum + epsilon:
            eligible.append(value)
    if not eligible:
        raise StageOResume1Error("fixed integrator has no eligible internal scale")
    if len(set(eligible)) != len(eligible):
        raise StageOResume1Error("integrator scale grid contains duplicate values")
    return tuple(sorted(eligible, reverse=True))

def group_attempts_in_exact_descending_order(
    events: Sequence[Mapping[str, Any]], *,
    expected_order: Sequence[float],
    scale_key: Callable[[float], str],
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    expected = tuple(float(value) for value in expected_order)
    if not expected:
        raise StageOResume1Error("expected internal-scale order is empty")
    observed: List[float] = []
    for event in events:
        value = float(event.get("attempted_scale"))
        if not math.isfinite(value) or value <= 0.0:
            raise StageOResume1Error("invalid attempted internal scale")
        observed.append(value)
    if tuple(observed) != expected:
        raise StageOResume1Error(
            "internal scale attempt order differs from frozen Stage-E execution order: "
            "expected={!r} observed={!r}".format(expected, tuple(observed))
        )
    output: Dict[str, List[Mapping[str, Any]]] = {}
    for event, scale in zip(events, observed):
        output.setdefault(scale_key(scale), []).append(event)
    if tuple(float(key) for key in output) != expected:
        raise StageOResume1Error("grouping changed callback emission order")
    return {key: tuple(value) for key, value in output.items()}

def _validate_original_provenance(repo: Path) -> Mapping[str, Any]:
    if _git(repo, "rev-parse", f"{BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGEO_IMPLEMENTATION_COMMIT
    ):
        raise StageOResume1Error("Stage-O blocked provenance parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGEO_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_STAGEO_IMPLEMENTATION_PARENT
    ):
        raise StageOResume1Error("Stage-O implementation parent changed")
    if _git(
        repo, "show", "-s", "--format=%s", BASE_STAGEO_IMPLEMENTATION_COMMIT
    ) != ORIGINAL_IMPLEMENTATION_SUBJECT:
        raise StageOResume1Error("Stage-O implementation subject changed")
    if commit_name_status(repo, BASE_STAGEO_IMPLEMENTATION_COMMIT) != tuple(
        sorted(ORIGINAL_IMPLEMENTATION_PATHS)
    ):
        raise StageOResume1Error("Stage-O implementation paths changed")
    if _git(
        repo, "show", "-s", "--format=%s", BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT
    ) != ORIGINAL_BLOCKED_SUBJECT:
        raise StageOResume1Error("Stage-O blocked-evidence subject changed")
    if commit_name_status(repo, BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(ORIGINAL_BLOCKED_PATHS)
    ):
        raise StageOResume1Error("Stage-O blocked-evidence paths changed")

    blocked_path = repo / STAGEO_BLOCKED_REPORT
    if not blocked_path.is_file():
        raise StageOResume1Error("Stage-O blocked report is missing")
    if sha256_file(blocked_path) != EXPECTED_STAGEO_BLOCKED_REPORT_SHA256:
        raise StageOResume1Error("Stage-O blocked report SHA changed")
    committed = _git_bytes(
        repo, "show",
        f"{BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT}:{STAGEO_BLOCKED_REPORT}"
    )
    if committed != blocked_path.read_bytes():
        raise StageOResume1Error(
            "Stage-O blocked report differs from committed blob"
        )

    payload = json.loads(blocked_path.read_text(encoding="utf-8"))
    if payload.get("execution_verdict") != "BLOCKED":
        raise StageOResume1Error("Stage-O blocked verdict changed")
    if payload.get("root_cause") != (
        "phase314b_r258_stageo_targeted_oracle_replay_failed"
    ):
        raise StageOResume1Error("Stage-O blocked root cause changed")
    if payload.get("required_next_path") != "RESTORE_STAGEO_TARGETED_ORACLE_REPLAY":
        raise StageOResume1Error("Stage-O blocked next path changed")
    if payload.get("error_message") != "internal scale attempt order changed":
        raise StageOResume1Error("Stage-O blocked failure message changed")
    require_false(payload, FALSE_BOUNDARIES, "Stage-O blocked report")
    return {
        "stage_o_implementation_commit": BASE_STAGEO_IMPLEMENTATION_COMMIT,
        "stage_o_blocked_evidence_commit": BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT,
        "stage_o_blocked_report_sha256": EXPECTED_STAGEO_BLOCKED_REPORT_SHA256,
        "original_failure_message": payload.get("error_message"),
    }

def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageOResume1Error("Stage-O Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT:
        raise StageOResume1Error("Stage-O Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageOResume1Error("Stage-O Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageOResume1Error("Stage-O Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageOResume1Error("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageOResume1Error("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageOResume1Error("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageOResume1Error(
            "Stage-O Resume1 worktree must be clean before execution"
        )

    stageo_source = repo / STAGEO_SOURCE
    if sha256_file(stageo_source) != EXPECTED_STAGEO_SOURCE_SHA256:
        raise StageOResume1Error("original Stage-O source changed")
    if _git_bytes(
        repo, "show",
        f"{BASE_STAGEO_IMPLEMENTATION_COMMIT}:{STAGEO_SOURCE}"
    ) != stageo_source.read_bytes():
        raise StageOResume1Error(
            "original Stage-O source differs from implementation blob"
        )
    provenance = _validate_original_provenance(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageOResume1Error(
                f"Stage-O Resume1 output already exists: {relative}"
            )
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "original_stage_o_source_sha256": EXPECTED_STAGEO_SOURCE_SHA256,
        "blocked_provenance": provenance,
    }

def _runtime_attempt_order(stageo: Any) -> Tuple[float, ...]:
    modules = stageo._runtime_modules()
    stagel = modules["stagel"]
    stagek = modules["stagek"]
    integrator_spec = stagel.stagee258.ConstrainedIntegratorSpec()
    definition = stagek.stagef.fixed_integrator_definition()
    integrator_spec.validate()
    definition.validate()
    return expected_internal_scale_attempt_order(
        scale_grid=integrator_spec.scale_grid,
        maximum_scale=definition.maximum_scale,
        standardizer_epsilon=integrator_spec.standardizer_epsilon,
    )

@contextmanager
def patched_descending_grouping(stageo: Any, expected_order: Sequence[float]):
    original = stageo.group_attempts_by_internal_scale

    def corrected(
        events: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        return group_attempts_in_exact_descending_order(
            events,
            expected_order=expected_order,
            scale_key=stageo._scale_key,
        )

    stageo.group_attempts_by_internal_scale = corrected
    try:
        yield
    finally:
        stageo.group_attempts_by_internal_scale = original

def execute_recovery(
    *, root: Path, environment: Mapping[str, Any],
    repository: Mapping[str, Any]
) -> Mapping[str, Any]:
    validate_environment_variables()
    from ccda_phase3 import (
        phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo,
    )

    if sha256_file(Path(stageo.__file__).resolve()) != EXPECTED_STAGEO_SOURCE_SHA256:
        raise StageOResume1Error("imported Stage-O source SHA changed")

    expected_order = _runtime_attempt_order(stageo)
    original_grouping = stageo.group_attempts_by_internal_scale
    with patched_descending_grouping(stageo, expected_order):
        stage_o_result = stageo.run_targeted_replay(
            root=Path(root).resolve(),
            environment=environment,
            repository=repository,
        )
    if stageo.group_attempts_by_internal_scale is not original_grouping:
        raise StageOResume1Error(
            "original Stage-O grouping function was not restored"
        )

    if stage_o_result.get("execution_verdict") != "PASS":
        raise StageOResume1Error("recovered Stage-O execution did not PASS")
    if stage_o_result.get("scientific_status") != "BLOCKED":
        raise StageOResume1Error("recovered Stage-O scientific status changed")
    if stage_o_result.get("selected_configuration") is not None:
        raise StageOResume1Error("recovered Stage-O selected a configuration")
    if stage_o_result.get("train_only_recommendation") is not None:
        raise StageOResume1Error("recovered Stage-O emitted a recommendation")

    targeted = stage_o_result.get("targeted_replay")
    if not isinstance(targeted, Mapping):
        raise StageOResume1Error("recovered Stage-O targeted replay is missing")
    if int(targeted.get("callback_off_on_pair_count", -1)) != 6:
        raise StageOResume1Error("recovered Stage-O replay pair count changed")
    if targeted.get("all_callback_results_bit_exact") is not True:
        raise StageOResume1Error("recovered callback replay is not bit-exact")
    if targeted.get("all_cells_match_stage_l_025_identity") is not True:
        raise StageOResume1Error(
            "recovered replay differs from Stage-L identity"
        )
    require_false(stage_o_result, FALSE_BOUNDARIES, "recovered Stage-O result")

    stage_o_result_sha = sha256_bytes(stable_json_bytes(stage_o_result))
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": stage_o_result["scientific_status"],
        "root_cause": stage_o_result["root_cause"],
        "required_next_path": stage_o_result["required_next_path"],
        "primary_failure_locus": stage_o_result["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": copy.deepcopy(dict(repository)),
        "environment": copy.deepcopy(dict(environment)),
        "recovery_contract": {
            "original_stage_o_runner_attempt_count": 1,
            "original_stage_o_success_report_created": False,
            "original_stage_o_blocked_report_preserved": True,
            "original_stage_o_blocked_report_sha256": (
                EXPECTED_STAGEO_BLOCKED_REPORT_SHA256
            ),
            "original_failure_message": "internal scale attempt order changed",
            "repair_scope": (
                "descending_internal_scale_attempt_order_validator_only"
            ),
            "expected_internal_scale_attempt_order": list(expected_order),
            "attempt_order_derived_from_frozen_stagee_contract": True,
            "callback_events_sorted_or_reordered": False,
            "original_stage_o_source_modified": False,
            "temporary_patch_restored_after_call": True,
            "resume1_runner_attempt_count": 1,
            "resume1_targeted_callback_pair_count": 6,
        },
        "stage_o_result_sha256": stage_o_result_sha,
        "stage_o_result": stage_o_result,
        "mechanism_boundary": {
            "stagee_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stagem_modified": False,
            "stagen_modified": False,
            "original_stageo_modified": False,
            "integrator_scale_search_changed": False,
            "integrator_thresholds_changed": False,
            "external_scale_bank_changed": False,
            "oracle_source_population_changed": False,
            "timestep_population_changed": False,
            "replay_population_changed": False,
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
        "root_cause": "phase314b_r258_stageo_resume1_recovery_failed",
        "required_next_path": (
            "RESTORE_STAGEO_RESUME1_DESCENDING_ATTEMPT_ORDER_RECOVERY"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "original_stage_o_blocked_report_preserved": True,
        "original_stage_o_blocked_report_sha256": (
            EXPECTED_STAGEO_BLOCKED_REPORT_SHA256
        ),
        "original_stage_o_source_modified": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
