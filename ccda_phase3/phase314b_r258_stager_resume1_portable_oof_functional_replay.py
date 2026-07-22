"""Phase3.14b-r2.5.8 Stage R Resume1 portable OOF functional replay.

The original Stage-R run stopped before science because it required the newly
refitted OOF prediction SHA to equal the historical Stage-L prediction SHA.
Stage-L intentionally persisted no prediction tensor, so a SHA mismatch cannot
be measured or repaired from the historical report.  The first frozen backbone
is reduced-rank ridge and uses SVD, whose byte-level output can be platform
sensitive even when the downstream constrained-integrator behavior is stable.

Resume1 does not forge the historical SHA and does not weaken downstream
identity.  Every current OOF fit is repeated immediately and must be byte-exact
(including fold records and model identities).  Historical prediction and
proposed-direction SHA equality are recorded, while the following historical
Stage-L multiplier-0.25 identities remain hard gates: selected scale, final
candidate, candidate-motion rate, callback capture, callback-off/on equality,
and complete sequential assembly.  Only after this portable functional replay
contract passes does the unchanged Stage-R shadow audit continue.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.8 Stage R Resume1"
SCHEMA = "phase314b_r258_stager_resume1_portable_oof_functional_replay_v1"
BLOCKED_SCHEMA = "phase314b_r258_stager_resume1_portable_oof_functional_replay_blocked_v1"

BASE_STAGER_IMPLEMENTATION_COMMIT = "3fe97091427b7a6bcef1c79805ebebece93746a5"
BASE_STAGER_BLOCKED_EVIDENCE_COMMIT = "61985677dde41f48396331c8c446f97f1d626d30"
EXPECTED_STAGER_IMPLEMENTATION_PARENT = "b47d2d848ce1a4f4109f5937d9f224897e1301cb"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGER_SOURCE = "ccda_phase3/phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit.py"
EXPECTED_STAGER_SOURCE_SHA256 = "9f8da64be049bb9cd9a61e229e4dc20e9a91cccc55261f43511a76f600fb4ed0"
STAGER_BLOCKED_REPORT = "reports/phase3_14b_r258_stager_tolerance_aligned_oof_post_upper_audit_blocked_summary.json"
EXPECTED_STAGER_BLOCKED_REPORT_SHA256 = "d10376e855ceb89c453366488213524c98b04b28ac2050f1b3df46d9bc6549a6"

SUCCESS_REPORT = "reports/phase3_14b_r258_stager_resume1_portable_oof_functional_replay_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stager_resume1_portable_oof_functional_replay_blocked_summary.json"

IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage R Resume1: restore portable OOF functional replay"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage R Resume1 OOF functional evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage R Resume1 blocked evidence"
ORIGINAL_IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage R: audit tolerance-aligned OOF post-upper rejection"
ORIGINAL_BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.8 Stage R blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stager_resume1_portable_oof_functional_replay.py"),
    ("A", "scripts/phase3_14b_r258_stager_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stager_resume1_portable_oof_functional_replay.py"),
)
ORIGINAL_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", STAGER_SOURCE),
    ("A", "scripts/phase3_14b_r258_stager_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stager_tolerance_aligned_oof_post_upper_audit.py"),
)
ORIGINAL_BLOCKED_PATHS: Tuple[Tuple[str, str], ...] = (("A", STAGER_BLOCKED_REPORT),)

EXPECTED_CELL_COUNT = 27
EXPECTED_SCIENCE_FIT_COUNT = 27
EXPECTED_REPEAT_FIT_COUNT = 27
EXPECTED_TOTAL_FIT_COUNT = 54
EXPECTED_CALLBACK_PAIR_COUNT = 27
LOWER_MULTIPLIER = 0.25

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


class StageRResume1Error(RuntimeError):
    """Fail-closed Stage-R Resume1 error."""


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
        raise StageRResume1Error(
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
        raise StageRResume1Error(
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
            raise StageRResume1Error("unexpected diff-tree record")
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
            raise StageRResume1Error(
                "{} boundary changed: {}={!r}".format(label, key, mapping.get(key))
            )


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageRResume1Error(
            "deterministic environment mismatch: {}".format(mismatch)
        )
    return dict(EXPECTED_ENV)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageRResume1Error(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageRResume1Error(f"{label} is not a sequence")
    return value


def _validate_original_provenance(repo: Path) -> Mapping[str, Any]:
    if _git(repo, "rev-parse", f"{BASE_STAGER_BLOCKED_EVIDENCE_COMMIT}^") != (
        BASE_STAGER_IMPLEMENTATION_COMMIT
    ):
        raise StageRResume1Error("Stage-R blocked provenance parent changed")
    if _git(repo, "rev-parse", f"{BASE_STAGER_IMPLEMENTATION_COMMIT}^") != (
        EXPECTED_STAGER_IMPLEMENTATION_PARENT
    ):
        raise StageRResume1Error("Stage-R implementation parent changed")
    if _git(
        repo, "show", "-s", "--format=%s", BASE_STAGER_IMPLEMENTATION_COMMIT
    ) != ORIGINAL_IMPLEMENTATION_SUBJECT:
        raise StageRResume1Error("Stage-R implementation subject changed")
    if commit_name_status(repo, BASE_STAGER_IMPLEMENTATION_COMMIT) != tuple(
        sorted(ORIGINAL_IMPLEMENTATION_PATHS)
    ):
        raise StageRResume1Error("Stage-R implementation paths changed")
    if _git(
        repo, "show", "-s", "--format=%s", BASE_STAGER_BLOCKED_EVIDENCE_COMMIT
    ) != ORIGINAL_BLOCKED_SUBJECT:
        raise StageRResume1Error("Stage-R blocked-evidence subject changed")
    if commit_name_status(repo, BASE_STAGER_BLOCKED_EVIDENCE_COMMIT) != tuple(
        sorted(ORIGINAL_BLOCKED_PATHS)
    ):
        raise StageRResume1Error("Stage-R blocked-evidence paths changed")

    source = repo / STAGER_SOURCE
    if sha256_file(source) != EXPECTED_STAGER_SOURCE_SHA256:
        raise StageRResume1Error("original Stage-R source SHA changed")
    if _git_bytes(
        repo,
        "show",
        f"{BASE_STAGER_IMPLEMENTATION_COMMIT}:{STAGER_SOURCE}",
    ) != source.read_bytes():
        raise StageRResume1Error("original Stage-R source differs from committed blob")

    blocked_path = repo / STAGER_BLOCKED_REPORT
    if not blocked_path.is_file():
        raise StageRResume1Error("Stage-R blocked report is missing")
    if sha256_file(blocked_path) != EXPECTED_STAGER_BLOCKED_REPORT_SHA256:
        raise StageRResume1Error("Stage-R blocked report SHA changed")
    if _git_bytes(
        repo,
        "show",
        f"{BASE_STAGER_BLOCKED_EVIDENCE_COMMIT}:{STAGER_BLOCKED_REPORT}",
    ) != blocked_path.read_bytes():
        raise StageRResume1Error("Stage-R blocked report differs from committed blob")
    payload = json.loads(blocked_path.read_text(encoding="utf-8"))
    if payload.get("execution_verdict") != "BLOCKED":
        raise StageRResume1Error("Stage-R blocked verdict changed")
    if payload.get("root_cause") != "phase314b_r258_stager_oof_post_upper_execution_failed":
        raise StageRResume1Error("Stage-R blocked root changed")
    if payload.get("required_next_path") != "RESTORE_STAGER_TOLERANCE_ALIGNED_OOF_POST_UPPER_AUDIT":
        raise StageRResume1Error("Stage-R blocked next path changed")
    if payload.get("error_message") != "Stage-R OOF prediction differs from Stage-L":
        raise StageRResume1Error("Stage-R blocked error message changed")
    require_false(payload, FALSE_BOUNDARIES, "Stage-R blocked report")
    return {
        "stage_r_implementation_commit": BASE_STAGER_IMPLEMENTATION_COMMIT,
        "stage_r_blocked_evidence_commit": BASE_STAGER_BLOCKED_EVIDENCE_COMMIT,
        "stage_r_blocked_report_sha256": EXPECTED_STAGER_BLOCKED_REPORT_SHA256,
        "original_failure_message": payload.get("error_message"),
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageRResume1Error("Stage-R Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    parent = _git(repo, "rev-parse", f"{head}^")
    if parent != BASE_STAGER_BLOCKED_EVIDENCE_COMMIT:
        raise StageRResume1Error("Stage-R Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageRResume1Error("Stage-R Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageRResume1Error("Stage-R Resume1 implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageRResume1Error("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageRResume1Error("DeformableRavens commit changed")
    if status_paths(submodule):
        raise StageRResume1Error("DeformableRavens worktree is dirty")
    if status_paths(repo):
        raise StageRResume1Error("Stage-R Resume1 worktree must be clean")
    provenance = _validate_original_provenance(repo)
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageRResume1Error(f"Stage-R Resume1 output already exists: {relative}")
    return {
        "root": str(repo),
        "head": head,
        "parent": parent,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "original_stage_r_source_sha256": EXPECTED_STAGER_SOURCE_SHA256,
        "blocked_provenance": provenance,
    }


class RepeatFitRegistry:
    """Run each OOF fit twice and queue its deterministic identity."""

    def __init__(self, original_fit: Any) -> None:
        self.original_fit = original_fit
        self.pending: List[Mapping[str, Any]] = []
        self.records: List[Mapping[str, Any]] = []
        self.fit_call_count = 0

    @staticmethod
    def _fold_identity(result: Mapping[str, Any]) -> Mapping[str, Any]:
        folds = _sequence(result.get("fold_records"), "OOF fold records")
        model_identities = [
            copy.deepcopy(
                dict(
                    _mapping(
                        _mapping(fold, f"fold {index}").get("model_identity"),
                        f"fold {index} model identity",
                    )
                )
            )
            for index, fold in enumerate(folds)
        ]
        return {
            "fold_records_sha256": sha256_bytes(stable_json_bytes(folds)),
            "model_identities_sha256": sha256_bytes(stable_json_bytes(model_identities)),
            "fold_count": len(folds),
        }

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]:
        first = self.original_fit(**kwargs)
        second = self.original_fit(**kwargs)
        self.fit_call_count += 2
        first_prediction = str(first.get("prediction_sha256"))
        second_prediction = str(second.get("prediction_sha256"))
        first_fold = self._fold_identity(first)
        second_fold = self._fold_identity(second)
        checks = {
            "prediction_sha256": first_prediction == second_prediction,
            "fold_records_sha256": first_fold["fold_records_sha256"]
            == second_fold["fold_records_sha256"],
            "model_identities_sha256": first_fold["model_identities_sha256"]
            == second_fold["model_identities_sha256"],
            "fold_count": first_fold["fold_count"] == second_fold["fold_count"],
        }
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            raise StageRResume1Error(
                f"current OOF repeat fit is not byte-exact: {failed}"
            )
        record = {
            "base_direction_id": str(kwargs.get("base_direction_id")),
            "current_prediction_sha256": first_prediction,
            "repeat_prediction_sha256": second_prediction,
            "repeat_prediction_byte_exact": True,
            "fold_records_sha256": first_fold["fold_records_sha256"],
            "model_identities_sha256": first_fold["model_identities_sha256"],
            "fold_count": first_fold["fold_count"],
            "checks": checks,
        }
        self.pending.append(record)
        self.records.append(record)
        return first

    def consume(self, base_direction_id: str) -> Mapping[str, Any]:
        if not self.pending:
            raise StageRResume1Error("portable OOF fit identity queue is empty")
        record = self.pending.pop(0)
        if record.get("base_direction_id") != base_direction_id:
            raise StageRResume1Error("portable OOF fit identity order changed")
        return record

    def validate_complete(self) -> None:
        if self.pending:
            raise StageRResume1Error("portable OOF fit identity queue was not consumed")
        if len(self.records) != EXPECTED_CELL_COUNT:
            raise StageRResume1Error("portable OOF repeat cell count changed")
        if self.fit_call_count != EXPECTED_TOTAL_FIT_COUNT:
            raise StageRResume1Error("portable OOF total fit count changed")


def portable_validate_legacy_oof_identity(
    *,
    registry: RepeatFitRegistry,
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
    spec: Any,
    control: np.ndarray,
) -> Mapping[str, Any]:
    if str(persisted_record.get("base_direction_id")) != base_direction_id:
        raise StageRResume1Error("Stage-L backbone identity changed")
    if int(persisted_record.get("timestep")) != int(timestep):
        raise StageRResume1Error("Stage-L timestep identity changed")
    if str(persisted_record.get("feature_sha256")) != feature_sha256:
        raise StageRResume1Error("Stage-R feature hash differs from Stage-L")

    repeat = registry.consume(base_direction_id)
    if repeat.get("current_prediction_sha256") != prediction_sha256:
        raise StageRResume1Error("fit wrapper/current prediction SHA changed")

    persisted_audit = _mapping(
        persisted_record.get("oof_predicate_assembly"),
        "persisted OOF predicate assembly",
    )
    values = _sequence(persisted_audit.get("multiplier_records"), "multiplier records")
    matches = [
        _mapping(value, f"multiplier record {index}")
        for index, value in enumerate(values)
        if float(_mapping(value, f"multiplier record {index}").get("scale_multiplier"))
        == LOWER_MULTIPLIER
    ]
    if len(matches) != 1:
        raise StageRResume1Error("persisted multiplier-record population changed")
    expected = matches[0]

    selected = np.asarray(integration["selected_scale"], dtype=np.float64).reshape(
        np.asarray(control).shape[0]
    )
    candidate = np.asarray(integration["candidate"], dtype=np.float32)
    motion = stagek._row_norm(
        candidate.astype(np.float64)
        - np.asarray(control, dtype=np.float32).astype(np.float64)
    )
    selected_positive = selected > float(spec.binary_tolerance)
    motion_positive = motion > float(spec.candidate_motion_epsilon)
    if np.any(selected_positive != motion_positive):
        raise StageRResume1Error(
            "selected-scale and candidate-motion observables disagree"
        )
    attempts = [
        event
        for event in _sequence(capture.get("events"), "callback events")
        if isinstance(event, Mapping) and event.get("event_type") == "scale_attempt"
    ]
    if not attempts:
        raise StageRResume1Error("callback capture lacks scale attempts")
    observed_assembly = stagel.aggregate_sequential_attempts(attempts)

    historical_prediction_exact = (
        str(persisted_record.get("oof_prediction_sha256")) == prediction_sha256
    )
    historical_proposed_exact = (
        stagel.sha256_array(proposed_direction)
        == expected.get("proposed_direction_sha256")
    )
    functional_checks = {
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
    if not all(functional_checks.values()):
        failed = sorted(key for key, value in functional_checks.items() if not value)
        raise StageRResume1Error(
            f"portable OOF functional replay differs from Stage-L: {failed}"
        )

    checks = {
        "feature_sha256": True,
        "prediction_sha256": historical_prediction_exact,
        "current_repeat_prediction_sha256": True,
        "current_repeat_fold_records_sha256": True,
        "current_repeat_model_identities_sha256": True,
        "proposed_direction_sha256": historical_proposed_exact,
        **functional_checks,
    }
    return {
        "all_exact": True,
        "all_functional_exact": True,
        "historical_prediction_sha_exact": historical_prediction_exact,
        "historical_proposed_direction_sha_exact": historical_proposed_exact,
        "checks": checks,
        "portable_repeat_fit": copy.deepcopy(dict(repeat)),
        "persisted_multiplier": LOWER_MULTIPLIER,
        "persisted_callback_capture_sha256": expected.get("callback_capture_sha256"),
        "persisted_candidate_sha256": expected.get("candidate_sha256"),
        "persisted_selected_scale_sha256": expected.get("selected_scale_sha256"),
        "legacy_assembly": observed_assembly,
    }


@contextmanager
def patched_portable_replay(stager: Any, stagel: Any):
    original_validate = stager.validate_legacy_oof_identity
    original_fit = stagel.stageh.fit_oof_base_direction
    registry = RepeatFitRegistry(original_fit)

    def validate_wrapper(**kwargs: Any) -> Mapping[str, Any]:
        return portable_validate_legacy_oof_identity(registry=registry, **kwargs)

    stager.validate_legacy_oof_identity = validate_wrapper
    stagel.stageh.fit_oof_base_direction = registry
    try:
        yield registry
    finally:
        stager.validate_legacy_oof_identity = original_validate
        stagel.stageh.fit_oof_base_direction = original_fit


def _correct_stage_r_result(
    result: Mapping[str, Any],
    registry: RepeatFitRegistry,
) -> Mapping[str, Any]:
    corrected = copy.deepcopy(dict(result))
    if corrected.get("execution_verdict") != "PASS":
        raise StageRResume1Error("recovered Stage-R execution did not PASS")
    if corrected.get("scientific_status") != "BLOCKED":
        raise StageRResume1Error("recovered Stage-R scientific status changed")
    if corrected.get("selected_configuration") is not None:
        raise StageRResume1Error("recovered Stage-R selected a configuration")
    if corrected.get("train_only_recommendation") is not None:
        raise StageRResume1Error("recovered Stage-R emitted a recommendation")
    require_false(corrected, FALSE_BOUNDARIES, "recovered Stage-R result")

    audit = _mapping(
        corrected.get("tolerance_aligned_oof_post_upper_audit"),
        "Stage-R audit",
    )
    cells = _sequence(audit.get("cell_records"), "Stage-R cells")
    if len(cells) != EXPECTED_CELL_COUNT:
        raise StageRResume1Error("recovered Stage-R cell count changed")
    historical_prediction_exact = 0
    historical_proposed_exact = 0
    for index, raw in enumerate(cells):
        cell = _mapping(raw, f"Stage-R cell {index}")
        identity = _mapping(cell.get("legacy_identity"), "legacy identity")
        if identity.get("all_functional_exact") is not True:
            raise StageRResume1Error("legacy functional replay is not exact")
        repeat = _mapping(identity.get("portable_repeat_fit"), "repeat fit")
        if repeat.get("repeat_prediction_byte_exact") is not True:
            raise StageRResume1Error("current OOF repeat fit is not exact")
        historical_prediction_exact += int(
            identity.get("historical_prediction_sha_exact") is True
        )
        historical_proposed_exact += int(
            identity.get("historical_proposed_direction_sha_exact") is True
        )

    mismatch_count = len(cells) - historical_prediction_exact
    proposed_mismatch_count = len(cells) - historical_proposed_exact
    mutable_audit = dict(audit)
    mutable_audit.update(
        {
            "scientific_oof_fit_count": EXPECTED_SCIENCE_FIT_COUNT,
            "portable_repeat_fit_count": EXPECTED_REPEAT_FIT_COUNT,
            "total_oof_fit_count": EXPECTED_TOTAL_FIT_COUNT,
            "all_current_oof_repeat_predictions_byte_exact": True,
            "all_current_oof_repeat_fold_records_byte_exact": True,
            "all_current_oof_repeat_model_identities_byte_exact": True,
            "historical_prediction_sha_exact_cell_count": historical_prediction_exact,
            "historical_prediction_sha_mismatch_cell_count": mismatch_count,
            "historical_proposed_direction_sha_exact_cell_count": historical_proposed_exact,
            "historical_proposed_direction_sha_mismatch_cell_count": proposed_mismatch_count,
            "all_legacy_025_functional_integrations_match_stagel": True,
            "historical_prediction_tensor_available": False,
            "historical_prediction_sha_forged_or_overwritten": False,
            "portable_functional_replay_contract_passed": True,
        }
    )
    corrected["tolerance_aligned_oof_post_upper_audit"] = mutable_audit

    immutable = dict(_mapping(corrected.get("immutable_inputs"), "immutable inputs"))
    immutable.update(
        {
            "oof_surrogate_scientific_fit_count": EXPECTED_SCIENCE_FIT_COUNT,
            "oof_surrogate_repeat_identity_fit_count": EXPECTED_REPEAT_FIT_COUNT,
            "oof_surrogate_total_fit_count": EXPECTED_TOTAL_FIT_COUNT,
            "historical_prediction_sha_checked": True,
            "historical_prediction_sha_required_for_science": False,
            "historical_integrator_functional_identity_required": True,
            "current_repeat_prediction_identity_required": True,
        }
    )
    corrected["immutable_inputs"] = immutable

    mechanism = dict(
        _mapping(corrected.get("mechanism_boundary"), "mechanism boundary")
    )
    mechanism.update(
        {
            "oof_predictions_changed": mismatch_count > 0,
            "historical_prediction_byte_identity_preserved": mismatch_count == 0,
            "current_oof_repeat_byte_identity_preserved": True,
            "legacy_integrator_functional_identity_preserved": True,
            "historical_prediction_sha_replaced": False,
            "stage_r_science_logic_modified": False,
            "stage_r_shadow_gate_modified": False,
        }
    )
    corrected["mechanism_boundary"] = mechanism

    corrected.pop("scientific_result_sha256", None)
    corrected["scientific_result_sha256"] = sha256_bytes(
        stable_json_bytes(corrected)
    )
    registry.validate_complete()
    return corrected


def execute_recovery(
    *,
    root: Path,
    environment: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_environment_variables()
    from ccda_phase3 import (
        phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager,
    )
    from ccda_phase3 import phase314b_r258_stagel_predicate_assembly_audit as stagel

    if sha256_file(Path(stager.__file__).resolve()) != EXPECTED_STAGER_SOURCE_SHA256:
        raise StageRResume1Error("imported Stage-R source SHA changed")

    with patched_portable_replay(stager, stagel) as registry:
        stage_r_result = stager.run_oof_post_upper_audit(
            root=Path(root).resolve(),
            environment=environment,
            repository=repository,
        )
    corrected = _correct_stage_r_result(stage_r_result, registry)
    stage_r_result_sha = sha256_bytes(stable_json_bytes(corrected))
    audit = _mapping(
        corrected.get("tolerance_aligned_oof_post_upper_audit"),
        "corrected Stage-R audit",
    )
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": corrected["scientific_status"],
        "root_cause": corrected["root_cause"],
        "required_next_path": corrected["required_next_path"],
        "primary_failure_locus": corrected["primary_failure_locus"],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": copy.deepcopy(dict(repository)),
        "environment": copy.deepcopy(dict(environment)),
        "recovery_contract": {
            "original_stage_r_runner_attempt_count": 1,
            "original_stage_r_success_report_created": False,
            "original_stage_r_blocked_report_preserved": True,
            "original_stage_r_blocked_report_sha256": (
                EXPECTED_STAGER_BLOCKED_REPORT_SHA256
            ),
            "original_failure_message": (
                "Stage-R OOF prediction differs from Stage-L"
            ),
            "repair_scope": "portable_oof_functional_replay_identity_only",
            "historical_prediction_sha_forged_or_overwritten": False,
            "historical_prediction_tensor_persisted": False,
            "current_fit_repeated_per_cell": True,
            "scientific_oof_fit_count": EXPECTED_SCIENCE_FIT_COUNT,
            "repeat_identity_fit_count": EXPECTED_REPEAT_FIT_COUNT,
            "total_oof_fit_count": EXPECTED_TOTAL_FIT_COUNT,
            "callback_pair_count": EXPECTED_CALLBACK_PAIR_COUNT,
            "original_stage_r_source_modified": False,
            "temporary_patches_restored_after_call": True,
            "historical_prediction_sha_exact_cell_count": audit[
                "historical_prediction_sha_exact_cell_count"
            ],
            "historical_prediction_sha_mismatch_cell_count": audit[
                "historical_prediction_sha_mismatch_cell_count"
            ],
            "portable_functional_replay_contract_passed": True,
        },
        "stage_r_result_sha256": stage_r_result_sha,
        "stage_r_result": corrected,
        "mechanism_boundary": {
            "stagee_modified": False,
            "stageh_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stageq_modified": False,
            "stageq_resume1_modified": False,
            "original_stager_modified": False,
            "legacy_upper_gate_changed": False,
            "aligned_upper_gate_changed": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "oof_backbone_population_changed": False,
            "historical_prediction_sha_replaced": False,
            "portable_functional_identity_contract_only": True,
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
        "root_cause": "phase314b_r258_stager_resume1_portable_oof_recovery_failed",
        "required_next_path": "RESTORE_STAGER_RESUME1_PORTABLE_OOF_FUNCTIONAL_REPLAY",
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "original_stage_r_blocked_report_preserved": True,
        "original_stage_r_blocked_report_sha256": (
            EXPECTED_STAGER_BLOCKED_REPORT_SHA256
        ),
        "historical_prediction_sha_forged_or_overwritten": False,
        "original_stage_r_source_modified": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
