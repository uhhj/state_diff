"""Phase3.14b-r2.5.9 Stage C fold-resolved tail attribution.

Stage-B Resume1 established, from immutable Stage-A durable aggregate evidence,
that the only failed outer-crossfit gate was ``t=10`` acceptance coverage:
accepted-subset fidelity and every frozen gate at ``t=25`` and ``t=50`` passed.
That evidence did not persist outer-test metrics by fold or anonymous group, so
it could not determine whether candidate support or adverse-risk abstention
caused the coverage collapse.

Stage C is a newly preregistered objective-train-only diagnostic stage.  It does
not change the frozen backbone, timesteps, nested grouped folds, direction
model, candidate bank, aligned gate, shrinkage bank, risk model, thresholds, or
eligibility gates.  Instead it installs a read-only instrumentation adapter at
the frozen Stage-X Resume1 ``stitch_fold_selected_procedure`` boundary.  The
adapter receives the already-generated outer-fold outputs, computes only
fold/timestep/condition/anonymous-group aggregates, verifies that the full
outer-output fingerprint is unchanged, and returns the original stitch result.

The complete Stage-A global result is reproduced and compared field-by-field to
the committed Stage-A durable worker evidence.  Fold-resolved diagnostics are
valid only if that identity check passes.  No selection holdout or frozen probe
is accessed.  No row-level tensor, model, candidate, descriptor, probability,
raw group identifier, checkpoint, NPZ, cache, image, or video is persisted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r258_stagex_tail_robust_nested_oof as stagex
from ccda_phase3 import phase314b_r258_stagex_resume1_import_crossfit_recovery as kernel
from ccda_phase3 import phase314b_r259_stagea_durable_tail_robust_nested_oof as stagea
from ccda_phase3 import phase314b_r259_stageb_resume1_postcommit_evidence_audit_recovery as stageb_resume1

PHASE = "Phase3.14b-r2.5.9 Stage C"
SCHEMA = "phase314b_r259_stagec_fold_resolved_tail_attribution_v1"
PROBE_SCHEMA = SCHEMA + "_environment_probe_v1"
WORKER_SCHEMA = SCHEMA + "_worker_evidence_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT = "6b1506b8b48c19ff31cdfed519a07d8561569f91"
BASE_STAGEB_RESUME1_EVIDENCE_COMMIT = "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5"
BASE_STAGEB_RESUME1_PARENT = "2d9568888c5583e82fb8adafad1cb867f65b7971"
BASE_STAGEA_IMPLEMENTATION_COMMIT = "507461048754a5a03f287cd1ef7017aacd53afbb"
BASE_STAGEA_EVIDENCE_COMMIT = "3100a877f3664936f713b089e1489768975f3fa4"
EXPECTED_REMOTE = "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_STAGEB_RESUME1_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage B Resume1: restore post-commit evidence audit"
)
BASE_STAGEB_RESUME1_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage B Resume1 outer-crossfit audit evidence"
)
BASE_STAGEA_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage A: add durable tail-robust nested OOF"
)
BASE_STAGEA_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage A durable tail-robust OOF evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage C: add fold-resolved tail attribution"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage C fold-resolved attribution evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage C blocked evidence"
)

BASE_STAGEB_RESUME1_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stageb_resume1_postcommit_evidence_audit_recovery.py"),
    ("A", "scripts/phase3_14b_r259_stageb_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stageb_resume1_postcommit_evidence_audit_recovery.py"),
)
BASE_STAGEB_RESUME1_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "reports/phase3_14b_r259_stageb_resume1_outer_crossfit_tail_failure_audit_summary.json"),
)
BASE_STAGEA_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagea_durable_tail_robust_nested_oof.py"),
    ("A", "scripts/phase3_14b_r259_stagea_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagea_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagea_durable_tail_robust_nested_oof.py"),
)
BASE_STAGEA_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "reports/phase3_14b_r259_stagea_environment_probe_evidence.json"),
    ("A", "reports/phase3_14b_r259_stagea_tail_robust_nested_group_oof_worker_evidence.json"),
    ("A", "reports/phase3_14b_r259_stagea_durable_tail_robust_nested_group_oof_summary.json"),
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagec_fold_resolved_tail_attribution.py"),
    ("A", "scripts/phase3_14b_r259_stagec_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagec_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagec_fold_resolved_tail_attribution.py"),
)

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagex_tail_robust_nested_oof.py": (
        "289da5e41b3d8dde85b429ea3147c5f005af97f0c7d156b1b58e9389f02a9025"
    ),
    "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py": (
        "66195d9f4a6e01fc3168501e65e3d0d57aed82ba182bf3b728645c271d448cff"
    ),
    "ccda_phase3/phase314b_r259_stagea_durable_tail_robust_nested_oof.py": (
        "7e653977a5edc15fe84ccba6e2d7548456a31a3a1d3de8cb7c82e1ce37c7b18e"
    ),
    "scripts/phase3_14b_r259_stagea_worker.py": (
        "c191af24e8b99e3d7445af203807bb5b2fb65986905d565dfea3484332fcdd1e"
    ),
    "scripts/phase3_14b_r259_stagea_execute.py": (
        "167c78310facb79d87d6c3ab87308df7c549c62f38864ce567986a8f1cbb51ed"
    ),
    "tests/test_phase3_14b_r259_stagea_durable_tail_robust_nested_oof.py": (
        "e037deee38cd2d626a458f4ddb62c8ded594290f32176b5198c7457ff5e15809"
    ),
    "ccda_phase3/phase314b_r259_stageb_resume1_postcommit_evidence_audit_recovery.py": (
        "30edce80cec4fe461aa5c2ca69a8a2fe4a31592dae9d200972ea6b69d693d8b8"
    ),
    "scripts/phase3_14b_r259_stageb_resume1_execute.py": (
        "c6559ea4824f1c21dae47b4777c98ee658d4b6d7c43ec3d08e3f5d99f47bfccd"
    ),
    "tests/test_phase3_14b_r259_stageb_resume1_postcommit_evidence_audit_recovery.py": (
        "b4bc65b8bd51179bb783ea72a8f30fcf071bf7f59d994f055baa49a82988e0cf"
    ),
}

STAGEB_RESUME1_REPORT = (
    "reports/phase3_14b_r259_stageb_resume1_outer_crossfit_tail_failure_audit_summary.json"
)
EXPECTED_STAGEB_RESUME1_REPORT_SHA256 = (
    "b1a4729b9186ebde0c56d3d1a11322598980b44eda3425eaadd14e94ae0a94f9"
)
EXPECTED_STAGEB_RESUME1_AUDIT_SHA256 = (
    "574f1abc3162c2dd5f5d38c3f9509a63e59064679a42061a4cfbde3a50b31362"
)

STAGEA_PROBE = "reports/phase3_14b_r259_stagea_environment_probe_evidence.json"
STAGEA_WORKER = (
    "reports/phase3_14b_r259_stagea_tail_robust_nested_group_oof_worker_evidence.json"
)
STAGEA_SUMMARY = (
    "reports/phase3_14b_r259_stagea_durable_tail_robust_nested_group_oof_summary.json"
)
EXPECTED_STAGEA_PROBE_SHA256 = (
    "840dc98ca2b37179920e1b95183a86eb4160ff1668f8b1d95df0d9fd1cbde7a1"
)
EXPECTED_STAGEA_WORKER_SHA256 = (
    "f9524bf1cc08a24ebc547e1e05fdad42274902a33dd03cb24fc8f2cf6c32bd92"
)
EXPECTED_STAGEA_SUMMARY_SHA256 = (
    "a433fe674b56bfbb2c09ff6ca836da941f4c9b32f69a9a05c06d22fdd3b9e8c0"
)
EXPECTED_STAGEA_WORKER_RESULT_SHA256 = (
    "3d3631c6765caee1c19f83d961afef8ccdb1676256371bd444612cf211231cca"
)
EXPECTED_STAGEA_SUMMARY_SELF_SHA256 = (
    "30295f799e1db737208c8afbc80a12243ee64c45c3f294ba0e3d83a5f5c63f37"
)

PROBE_EVIDENCE = "reports/phase3_14b_r259_stagec_environment_probe_evidence.json"
WORKER_EVIDENCE = (
    "reports/phase3_14b_r259_stagec_fold_resolved_tail_attribution_worker_evidence.json"
)
SUCCESS_REPORT = (
    "reports/phase3_14b_r259_stagec_fold_resolved_tail_attribution_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r259_stagec_fold_resolved_tail_attribution_blocked_summary.json"
)

LOCKED_TIMESTEPS = tuple(int(value) for value in kernel.LOCKED_TIMESTEPS)
DIRECTION_SHRINKAGES = tuple(float(value) for value in kernel.DIRECTION_SHRINKAGES)
EXPECTED_OUTER_FOLDS = int(kernel.EXPECTED_OUTER_FOLDS)
EXPECTED_ROWS = int(kernel.EXPECTED_ROWS)
EXPECTED_EXECUTION_COUNTS: Mapping[str, int] = {
    "direction_fit_count": 108,
    "candidate_generation_count": 324,
    "risk_fit_count": 324,
    "internal_scale_attempt_count": 2268,
    "inner_policy_evaluation_count": 216,
    "full_objective_oof_policy_evaluation_count": 36,
    "nonconverged_risk_fit_count": 0,
}
EXPECTED_DIAGNOSTIC_COUNTS: Mapping[str, int] = {
    "fold_timestep_record_count": 18,
    "fold_resolved_policy_metric_evaluation_count": 36,
    "risk_calibration_bin_count": 90,
}
CALIBRATION_BIN_EDGES: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
GROUP_HASH_NAMESPACE = "phase314b-r259-stagec-anonymous-group-v1"
FALSE_BOUNDARIES = tuple(stagex.FALSE_BOUNDARIES)


class StageCError(RuntimeError):
    """Fail-closed Stage-C error."""


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
        raise StageCError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageCError("write-once output exists: {}".format(target))
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


def promote_write_ahead(source: Path, target: Path, validator: Any) -> str:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise StageCError("write-ahead source missing: {}".format(source_path))
    payload = source_path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise StageCError("write-ahead JSON root changed")
    validator(value)
    atomic_write_once(target_path, payload)
    if target_path.read_bytes() != payload:
        raise StageCError("write-ahead promotion is not byte-exact")
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
        raise StageCError(
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
        raise StageCError(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace").strip()
            )
        )
    return bytes(completed.stdout)


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if line.strip():
            fields = line.split("\t")
            records.append((fields[0], fields[-1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise StageCError("{} worktree is dirty".format(label))


def child_environment(root: Path) -> Mapping[str, str]:
    env = dict(os.environ)
    root_text = str(Path(root).resolve())
    prior = env.get("PYTHONPATH", "")
    entries = [item for item in prior.split(os.pathsep) if item and item != root_text]
    env["PYTHONPATH"] = os.pathsep.join([root_text] + entries)
    return env


def anonymous_group_hash(group: Any) -> str:
    return sha256_bytes(
        (GROUP_HASH_NAMESPACE + "|" + str(group)).encode("utf-8")
    )


def _safe_ratio(numerator: float, denominator: float, epsilon: float) -> float:
    return float(numerator / max(float(denominator), float(epsilon)))


def _array_stats(values: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def calibration_bins(
    probabilities: np.ndarray,
    adverse: np.ndarray,
    support_mask: np.ndarray,
) -> Sequence[Mapping[str, Any]]:
    risk = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(adverse, dtype=np.bool_).reshape(-1)
    support = np.asarray(support_mask, dtype=np.bool_).reshape(-1)
    if risk.shape != labels.shape or risk.shape != support.shape:
        raise StageCError("calibration population shape changed")
    if np.any(~np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
        raise StageCError("risk probability is outside [0,1]")
    records: List[Mapping[str, Any]] = []
    for index in range(len(CALIBRATION_BIN_EDGES) - 1):
        lower = CALIBRATION_BIN_EDGES[index]
        upper = CALIBRATION_BIN_EDGES[index + 1]
        if index == len(CALIBRATION_BIN_EDGES) - 2:
            mask = support & (risk >= lower) & (risk <= upper)
        else:
            mask = support & (risk >= lower) & (risk < upper)
        records.append(
            {
                "bin_index": index,
                "lower_inclusive": lower,
                "upper_inclusive": bool(index == len(CALIBRATION_BIN_EDGES) - 2),
                "upper": upper,
                "count": int(np.sum(mask)),
                "mean_predicted_risk": float(np.mean(risk[mask])) if np.any(mask) else 0.0,
                "observed_adverse_rate": float(np.mean(labels[mask])) if np.any(mask) else 0.0,
            }
        )
    if sum(int(item["count"]) for item in records) != int(np.sum(support)):
        raise StageCError("calibration bins do not cover supported rows")
    return records


def _subset_record(
    *,
    mask: np.ndarray,
    control_sse: np.ndarray,
    raw_sse: np.ndarray,
    final_sse: np.ndarray,
    raw_supported: np.ndarray,
    final_accepted: np.ndarray,
    improving: np.ndarray,
    adverse: np.ndarray,
    risk: np.ndarray,
    epsilon: float,
) -> Mapping[str, Any]:
    subset = np.asarray(mask, dtype=np.bool_)
    if subset.shape != control_sse.shape:
        raise StageCError("subset population shape changed")
    count = int(np.sum(subset))
    denominator = float(np.sum(control_sse[subset]))
    supported = subset & raw_supported
    accepted = subset & final_accepted
    rejected = supported & ~final_accepted
    neutral = raw_supported & ~improving & ~adverse
    accepted_control = float(np.sum(control_sse[accepted]))
    return {
        "row_count": count,
        "raw_support_count": int(np.sum(supported)),
        "raw_support_rate": float(np.mean(raw_supported[subset])) if count else 0.0,
        "final_acceptance_count": int(np.sum(accepted)),
        "final_acceptance_rate": float(np.mean(final_accepted[subset])) if count else 0.0,
        "raw_overall_mse_ratio": _safe_ratio(
            float(np.sum(raw_sse[subset])), denominator, epsilon
        ) if count else 1.0,
        "final_overall_mse_ratio": _safe_ratio(
            float(np.sum(final_sse[subset])), denominator, epsilon
        ) if count else 1.0,
        "accepted_mse_ratio": _safe_ratio(
            float(np.sum(final_sse[accepted])), accepted_control, epsilon
        ) if np.any(accepted) else 1.0,
        "raw_improving_count": int(np.sum(supported & improving)),
        "raw_adverse_count": int(np.sum(supported & adverse)),
        "raw_neutral_count": int(np.sum(subset & neutral)),
        "accepted_improving_count": int(np.sum(accepted & improving)),
        "accepted_adverse_count": int(np.sum(accepted & adverse)),
        "accepted_neutral_count": int(np.sum(accepted & neutral)),
        "rejected_would_improve_count": int(np.sum(rejected & improving)),
        "rejected_adverse_count": int(np.sum(rejected & adverse)),
        "rejected_neutral_count": int(np.sum(rejected & neutral)),
        "mean_predicted_risk_on_raw_support": (
            float(np.mean(risk[supported])) if np.any(supported) else 0.0
        ),
        "observed_adverse_rate_on_raw_support": (
            float(np.mean(adverse[supported])) if np.any(supported) else 0.0
        ),
    }


def fold_policy_attribution(
    *,
    control: np.ndarray,
    raw_candidate: np.ndarray,
    raw_scale: np.ndarray,
    risk_probability: np.ndarray,
    risk_constant_probability: np.ndarray,
    threshold: float,
    target: np.ndarray,
    groups: Sequence[Any],
    conditions: Sequence[Any],
    spec: Any,
    policy_id: str,
    outer_fold: int,
    selected_policy_inner_eligible: bool,
    diagnostic_fallback_used: bool,
) -> Mapping[str, Any]:
    control_value = np.asarray(control, dtype=np.float32)
    candidate_value = np.asarray(raw_candidate, dtype=np.float32)
    scale = np.asarray(raw_scale, dtype=np.float64).reshape(-1)
    risk = np.asarray(risk_probability, dtype=np.float64).reshape(-1)
    constant = np.asarray(risk_constant_probability, dtype=np.float64).reshape(-1)
    target_value = np.asarray(target, dtype=np.float32)
    group_values = np.asarray(groups).astype(str).reshape(-1)
    condition_values = np.asarray(conditions).astype(str).reshape(-1)
    rows = control_value.shape[0]
    if candidate_value.shape != control_value.shape or target_value.shape != control_value.shape:
        raise StageCError("fold attribution tensor shape changed")
    for value, label in (
        (scale, "scale"),
        (risk, "risk"),
        (constant, "constant"),
        (group_values, "groups"),
        (condition_values, "conditions"),
    ):
        if value.shape != (rows,):
            raise StageCError("fold attribution {} population changed".format(label))
    if not (0.0 <= float(threshold) <= 1.0):
        raise StageCError("risk threshold changed")

    applied = stagex.apply_policy(
        control_value, candidate_value, scale, risk, float(threshold)
    )
    final_candidate = np.asarray(applied["candidate"], dtype=np.float32)
    final_scale = np.asarray(applied["selected_scale"], dtype=np.float64)
    raw_supported = scale > 0.0
    final_accepted = final_scale > 0.0
    control_sse = stagex.row_squared_error(control_value, target_value)
    raw_sse = stagex.row_squared_error(candidate_value, target_value)
    final_sse = stagex.row_squared_error(final_candidate, target_value)
    raw_delta = raw_sse - control_sse
    improving = raw_delta < 0.0
    adverse = raw_delta > 0.0
    neutral = raw_supported & ~improving & ~adverse
    rejected = raw_supported & ~final_accepted
    denominator = max(float(np.sum(control_sse)), float(spec.epsilon))

    selected_metrics = stagex.evaluate_policy(
        control_value,
        candidate_value,
        scale,
        risk,
        target_value,
        group_values,
        float(threshold),
        constant,
        spec,
    )
    baseline_metrics = stagex.evaluate_policy(
        control_value,
        candidate_value,
        scale,
        np.zeros(rows, dtype=np.float64),
        target_value,
        group_values,
        1.0,
        None,
        spec,
    )

    supported_count = int(np.sum(raw_supported))
    calibration = calibration_bins(risk, adverse, raw_supported)
    group_hashes = [anonymous_group_hash(value) for value in sorted(set(group_values))]
    group_records: List[Mapping[str, Any]] = []
    for raw_group in sorted(set(group_values)):
        mask = group_values == raw_group
        record = dict(
            _subset_record(
                mask=mask,
                control_sse=control_sse,
                raw_sse=raw_sse,
                final_sse=final_sse,
                raw_supported=raw_supported,
                final_accepted=final_accepted,
                improving=improving,
                adverse=adverse,
                risk=risk,
                epsilon=float(spec.epsilon),
            )
        )
        record["anonymous_group_sha256"] = anonymous_group_hash(raw_group)
        group_records.append(record)
    group_records.sort(key=lambda item: str(item["anonymous_group_sha256"]))

    condition_records: List[Mapping[str, Any]] = []
    for condition in sorted(set(condition_values)):
        mask = condition_values == condition
        record = dict(
            _subset_record(
                mask=mask,
                control_sse=control_sse,
                raw_sse=raw_sse,
                final_sse=final_sse,
                raw_supported=raw_supported,
                final_accepted=final_accepted,
                improving=improving,
                adverse=adverse,
                risk=risk,
                epsilon=float(spec.epsilon),
            )
        )
        record["condition_name"] = condition
        condition_records.append(record)

    accepted = final_accepted
    result: Dict[str, Any] = {
        "outer_fold": int(outer_fold),
        "timestep": None,
        "selected_policy_id": str(policy_id),
        "selected_policy_inner_eligible": bool(selected_policy_inner_eligible),
        "diagnostic_fallback_used": bool(diagnostic_fallback_used),
        "risk_threshold": float(threshold),
        "row_count": rows,
        "group_count": len(group_records),
        "condition_count": len(condition_records),
        "anonymous_group_population_sha256": sha256_bytes(
            compact_json_bytes(sorted(group_hashes))
        ),
        "raw_candidate": {
            "support_count": supported_count,
            "support_rate": float(np.mean(raw_supported)),
            "overall_mse_ratio": _safe_ratio(float(np.sum(raw_sse)), denominator, float(spec.epsilon)),
            "improving_count": int(np.sum(raw_supported & improving)),
            "adverse_count": int(np.sum(raw_supported & adverse)),
            "neutral_count": int(np.sum(neutral)),
            "adverse_prevalence": (
                float(np.mean(adverse[raw_supported])) if supported_count else 0.0
            ),
        },
        "final_policy_metrics": copy.deepcopy(selected_metrics),
        "no_abstention_baseline_metrics": copy.deepcopy(baseline_metrics),
        "decision_partition": {
            "accepted_improving_count": int(np.sum(accepted & improving)),
            "accepted_adverse_count": int(np.sum(accepted & adverse)),
            "accepted_neutral_count": int(np.sum(accepted & neutral)),
            "rejected_would_improve_count": int(np.sum(rejected & improving)),
            "rejected_adverse_count": int(np.sum(rejected & adverse)),
            "rejected_neutral_count": int(np.sum(rejected & neutral)),
            "accepted_beneficial_sse_mass": float(
                np.sum(np.maximum(-raw_delta[accepted], 0.0)) / denominator
            ),
            "accepted_adverse_sse_mass": float(
                np.sum(np.maximum(raw_delta[accepted], 0.0)) / denominator
            ),
            "rejected_lost_beneficial_sse_mass": float(
                np.sum(np.maximum(-raw_delta[rejected], 0.0)) / denominator
            ),
            "rejected_avoided_adverse_sse_mass": float(
                np.sum(np.maximum(raw_delta[rejected], 0.0)) / denominator
            ),
        },
        "risk_calibration": {
            "raw_support_count": supported_count,
            "mean_predicted_risk": float(np.mean(risk[raw_supported])) if supported_count else 0.0,
            "observed_adverse_prevalence": float(np.mean(adverse[raw_supported])) if supported_count else 0.0,
            "signed_calibration_gap": (
                float(np.mean(risk[raw_supported]) - np.mean(adverse[raw_supported]))
                if supported_count
                else 0.0
            ),
            "brier_score": float(selected_metrics["risk_brier_score"]),
            "constant_brier_score": float(selected_metrics["risk_constant_brier_score"]),
            "brier_nonworse": bool(selected_metrics["risk_brier_nonworse"]),
            "threshold_margin_stats": _array_stats(
                float(threshold) - risk[raw_supported]
            ),
            "bins": calibration,
        },
        "condition_aggregates": condition_records,
        "anonymous_group_aggregates": group_records,
        "provenance_hashes": {
            "raw_candidate_sha256": stagex.sha256_array(candidate_value),
            "raw_selected_scale_sha256": stagex.sha256_array(scale),
            "risk_probability_sha256": stagex.sha256_array(risk),
            "final_candidate_sha256": stagex.sha256_array(final_candidate),
            "final_selected_scale_sha256": stagex.sha256_array(final_scale),
        },
    }
    partition_count = sum(
        int(result["decision_partition"][key])
        for key in (
            "accepted_improving_count",
            "accepted_adverse_count",
            "accepted_neutral_count",
            "rejected_would_improve_count",
            "rejected_adverse_count",
            "rejected_neutral_count",
        )
    )
    if partition_count != supported_count:
        raise StageCError("decision partition does not cover raw-supported rows")
    if sum(int(item["row_count"]) for item in group_records) != rows:
        raise StageCError("anonymous group aggregates do not cover fold rows")
    if sum(int(item["row_count"]) for item in condition_records) != rows:
        raise StageCError("condition aggregates do not cover fold rows")
    return result


def outer_outputs_fingerprint(
    outer_outputs: Mapping[int, Mapping[int, Mapping[float, Mapping[str, Any]]]]
) -> str:
    records: List[Mapping[str, Any]] = []
    for fold in sorted(outer_outputs):
        for timestep in sorted(outer_outputs[fold]):
            for shrinkage in sorted(outer_outputs[fold][timestep]):
                value = outer_outputs[fold][timestep][shrinkage]
                records.append(
                    {
                        "outer_fold": int(fold),
                        "timestep": int(timestep),
                        "shrinkage": float(shrinkage),
                        "indices_sha256": stagex.sha256_array(value["indices"]),
                        "control_sha256": stagex.sha256_array(value["control"]),
                        "direction_sha256": stagex.sha256_array(value["direction"]),
                        "candidate_sha256": stagex.sha256_array(value["candidate"]),
                        "selected_scale_sha256": stagex.sha256_array(value["selected_scale"]),
                        "risk_probability_sha256": stagex.sha256_array(value["risk_probability"]),
                        "risk_constant_probability_sha256": stagex.sha256_array(
                            value["risk_constant_probability"]
                        ),
                        "gate_counts": copy.deepcopy(dict(value["gate_counts"])),
                    }
                )
    return sha256_bytes(compact_json_bytes(records))


def build_fold_resolved_records(
    *,
    context: Mapping[str, Any],
    outer_outputs: Mapping[int, Mapping[int, Mapping[float, Mapping[str, Any]]]],
    outer_selections: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    groups: np.ndarray,
    spec: Any,
) -> Mapping[str, Any]:
    if len(outer_selections) != EXPECTED_OUTER_FOLDS:
        raise StageCError("outer selection population changed")
    conditions = np.asarray(context["objective_condition_name"]).astype(str)
    fold_records: List[Mapping[str, Any]] = []
    seen_rows = {int(timestep): np.zeros(EXPECTED_ROWS, dtype=np.bool_) for timestep in LOCKED_TIMESTEPS}
    group_record_count = 0
    for fold in range(EXPECTED_OUTER_FOLDS):
        selection = outer_selections[fold]
        if int(selection.get("outer_fold", -1)) != fold:
            raise StageCError("outer selection order changed")
        policy = kernel.TailPolicy(**selection["selected_policy"])
        policy.validate()
        for timestep in LOCKED_TIMESTEPS:
            output = outer_outputs[fold][int(timestep)][float(policy.shrinkage)]
            indices = np.asarray(output["indices"], dtype=np.int64)
            if np.any(seen_rows[int(timestep)][indices]):
                raise StageCError("outer fold rows overlap")
            seen_rows[int(timestep)][indices] = True
            record = dict(
                fold_policy_attribution(
                    control=output["control"],
                    raw_candidate=output["candidate"],
                    raw_scale=output["selected_scale"],
                    risk_probability=output["risk_probability"],
                    risk_constant_probability=output["risk_constant_probability"],
                    threshold=float(policy.risk_threshold),
                    target=np.asarray(target)[indices],
                    groups=np.asarray(groups)[indices],
                    conditions=conditions[indices],
                    spec=spec,
                    policy_id=policy.policy_id,
                    outer_fold=fold,
                    selected_policy_inner_eligible=bool(
                        selection["selected_policy_inner_eligible"]
                    ),
                    diagnostic_fallback_used=bool(
                        selection["diagnostic_fallback_used"]
                    ),
                )
            )
            record["timestep"] = int(timestep)
            group_record_count += len(record["anonymous_group_aggregates"])
            fold_records.append(record)
    if any(not np.all(value) for value in seen_rows.values()):
        raise StageCError("fold-resolved records do not cover objective rows")
    fold_records.sort(key=lambda item: (int(item["outer_fold"]), int(item["timestep"])))
    if len(fold_records) != EXPECTED_DIAGNOSTIC_COUNTS["fold_timestep_record_count"]:
        raise StageCError("fold/timestep diagnostic population changed")
    if sum(len(item["risk_calibration"]["bins"]) for item in fold_records) != (
        EXPECTED_DIAGNOSTIC_COUNTS["risk_calibration_bin_count"]
    ):
        raise StageCError("risk calibration bin population changed")
    return {
        "fold_timestep_records": fold_records,
        "anonymous_group_aggregate_record_count": int(group_record_count),
        "fold_timestep_record_count": len(fold_records),
        "fold_resolved_policy_metric_evaluation_count": (
            EXPECTED_DIAGNOSTIC_COUNTS["fold_resolved_policy_metric_evaluation_count"]
        ),
        "risk_calibration_bin_count": (
            EXPECTED_DIAGNOSTIC_COUNTS["risk_calibration_bin_count"]
        ),
    }


def compare_stagea_reproduction(
    kernel_payload: Mapping[str, Any], stagea_worker: Mapping[str, Any]
) -> Mapping[str, Any]:
    field_pairs = {
        "locked_scientific_base": "locked_scientific_base",
        "stagex_spec": "stagex_spec",
        "policy_population": "policy_population",
        "population": "population",
        "outer_fold_selections": "outer_fold_selections",
        "inner_selection_modal_policy_diagnostic": "inner_selection_modal_policy_diagnostic",
        "outer_crossfit_fold_selected_policy_records": "outer_crossfit_fold_selected_policy_records",
        "outer_crossfit_baseline_records": "outer_crossfit_baseline_records",
        "full_objective_oof_fixed_policy_selection": "full_objective_oof_fixed_policy_selection",
        "full_objective_oof_fixed_policy_records": "full_objective_oof_fixed_policy_records",
        "procedure_gate_totals": "procedure_gate_totals",
        "execution_counts": "execution_counts",
    }
    identities: Dict[str, Any] = {}
    all_exact = True
    for kernel_field, worker_field in field_pairs.items():
        left = kernel_payload[kernel_field]
        right = stagea_worker[worker_field]
        left_sha = sha256_bytes(compact_json_bytes(left))
        right_sha = sha256_bytes(compact_json_bytes(right))
        exact = left_sha == right_sha and compact_json_bytes(left) == compact_json_bytes(right)
        identities[kernel_field] = {
            "current_sha256": left_sha,
            "stagea_sha256": right_sha,
            "exact": bool(exact),
        }
        all_exact = all_exact and bool(exact)
    if kernel_payload.get("scientific_status") != stagea_worker.get("scientific_status"):
        all_exact = False
    if kernel_payload.get("primary_failure_locus") != stagea_worker.get("scientific_kernel", {}).get(
        "kernel_primary_failure_locus"
    ):
        all_exact = False
    result = {
        "all_exact": bool(all_exact),
        "field_identities": identities,
        "current_kernel_scientific_status": kernel_payload.get("scientific_status"),
        "stagea_scientific_status": stagea_worker.get("scientific_status"),
        "current_kernel_primary_failure_locus": kernel_payload.get("primary_failure_locus"),
        "stagea_kernel_primary_failure_locus": stagea_worker.get("scientific_kernel", {}).get(
            "kernel_primary_failure_locus"
        ),
    }
    if result["all_exact"] is not True:
        raise StageCError("Stage-C does not exactly reproduce Stage-A aggregate evidence")
    return result


def classify_t10_fold_attribution(
    fold_records: Sequence[Mapping[str, Any]],
    fallback_folds: Sequence[int],
) -> Mapping[str, Any]:
    t10 = [item for item in fold_records if int(item["timestep"]) == 10]
    if len(t10) != EXPECTED_OUTER_FOLDS:
        raise StageCError("t10 fold record population changed")
    failed = [
        int(item["outer_fold"])
        for item in t10
        if float(item["final_policy_metrics"]["acceptance_rate"]) < 0.50
    ]
    if not failed:
        raise StageCError("Stage-C did not reproduce any t10 failing outer fold")
    categories: Dict[str, str] = {}
    beneficial_over_rejection: Dict[str, bool] = {}
    for item in t10:
        fold = int(item["outer_fold"])
        if fold not in failed:
            categories[str(fold)] = "passes_frozen_acceptance"
            beneficial_over_rejection[str(fold)] = False
            continue
        raw_support = float(item["raw_candidate"]["support_rate"])
        decision = item["decision_partition"]
        if raw_support < 0.50:
            categories[str(fold)] = "candidate_support_limited"
            beneficial_over_rejection[str(fold)] = False
        else:
            count_dominance = int(decision["rejected_would_improve_count"]) > int(
                decision["rejected_adverse_count"]
            )
            mass_dominance = float(
                decision["rejected_lost_beneficial_sse_mass"]
            ) > float(decision["rejected_avoided_adverse_sse_mass"])
            beneficial = bool(count_dominance and mass_dominance)
            categories[str(fold)] = (
                "risk_abstention_overrejects_beneficial_candidates"
                if beneficial
                else "risk_abstention_limited_mixed_quality"
            )
            beneficial_over_rejection[str(fold)] = beneficial
    failed_categories = {categories[str(fold)] for fold in failed}
    if failed_categories == {"candidate_support_limited"}:
        locus = "t10_candidate_support_limited_by_raw_candidate_availability"
        root = "phase314b_r259_stagec_fold_resolved_evidence_localizes_t10_coverage_failure_to_raw_candidate_support"
        next_path = "DESIGN_R259_STAGED_OBJECTIVE_TRAIN_ONLY_T10_CANDIDATE_SUPPORT_REPAIR_WITH_NESTED_GROUP_OOF_AND_DURABLE_EVIDENCE"
    elif failed_categories == {"risk_abstention_overrejects_beneficial_candidates"}:
        locus = "t10_risk_abstention_overrejects_beneficial_candidates"
        root = "phase314b_r259_stagec_fold_resolved_evidence_localizes_t10_coverage_failure_to_beneficial_candidate_overrejection"
        next_path = "DESIGN_R259_STAGED_OBJECTIVE_TRAIN_ONLY_T10_RISK_CALIBRATION_REPAIR_WITH_FOLD_RESOLVED_NESTED_GROUP_OOF"
    elif failed_categories <= {
        "risk_abstention_overrejects_beneficial_candidates",
        "risk_abstention_limited_mixed_quality",
    }:
        locus = "t10_risk_abstention_limited_with_mixed_candidate_quality"
        root = "phase314b_r259_stagec_fold_resolved_evidence_localizes_t10_coverage_failure_to_risk_abstention_with_mixed_quality"
        next_path = "DESIGN_R259_STAGED_OBJECTIVE_TRAIN_ONLY_T10_RISK_CALIBRATION_AND_DESCRIPTOR_REPAIR_WITH_NESTED_GROUP_OOF"
    else:
        locus = "t10_mixed_candidate_support_and_risk_abstention_failure"
        root = "phase314b_r259_stagec_fold_resolved_evidence_localizes_t10_coverage_failure_to_mixed_support_and_abstention"
        next_path = "DESIGN_R259_STAGED_OBJECTIVE_TRAIN_ONLY_T10_JOINT_SUPPORT_AND_RISK_REPAIR_WITH_FOLD_RESOLVED_EVIDENCE"

    fallback_set = sorted(int(value) for value in fallback_folds)
    overlap = sorted(set(failed).intersection(fallback_set))
    return {
        "scientific_status": "BLOCKED",
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "t10_failed_outer_folds": sorted(failed),
        "fold_categories": categories,
        "beneficial_over_rejection": beneficial_over_rejection,
        "stagea_inner_fallback_folds": fallback_set,
        "failed_fold_fallback_overlap": overlap,
        "all_failed_folds_are_fallback_folds": set(failed).issubset(set(fallback_set)),
        "all_fallback_folds_fail_outer_acceptance": set(fallback_set).issubset(set(failed)),
        "threshold_relaxation_authorized": False,
        "policy_reselection_authorized": False,
        "single_timestep_selection_authorized": False,
    }


def validate_stageb_resume1_report(report: Mapping[str, Any]) -> None:
    stageb_resume1.validate_recovery_result(
        report, BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT
    )
    if report.get("audit_sha256") != EXPECTED_STAGEB_RESUME1_AUDIT_SHA256:
        raise StageCError("Stage-B Resume1 audit SHA changed")
    if report.get("root_cause") != (
        "phase314b_r259_stageb_aggregate_evidence_localizes_t10_"
        "acceptance_coverage_failure_but_outer_test_fold_attribution_is_unavailable"
    ):
        raise StageCError("Stage-B Resume1 root cause changed")
    if report.get("required_next_path") != (
        "PREREGISTER_R259_STAGEC_OBJECTIVE_TRAIN_ONLY_FOLD_RESOLVED_"
        "TAIL_ATTRIBUTION_WITH_DURABLE_AGGREGATE_EVIDENCE"
    ):
        raise StageCError("Stage-B Resume1 next path changed")


def validate_repository(
    root: Path,
    implementation_commit: str,
    *,
    allow_promoted_evidence: bool = False,
) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise StageCError("Stage-C implementation commit is invalid")
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageCError("Stage-C requires Experiment1")
    if _git(repo, "rev-parse", "HEAD") != implementation_commit:
        raise StageCError("Stage-C implementation HEAD changed")
    if _git(repo, "rev-parse", implementation_commit + "^") != BASE_STAGEB_RESUME1_EVIDENCE_COMMIT:
        raise StageCError("Stage-C implementation parent changed")
    if _git(repo, "rev-parse", BASE_STAGEB_RESUME1_EVIDENCE_COMMIT + "^") != BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT:
        raise StageCError("Stage-B Resume1 evidence parent changed")
    if _git(repo, "rev-parse", BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT + "^") != BASE_STAGEB_RESUME1_PARENT:
        raise StageCError("Stage-B Resume1 implementation parent changed")
    subjects = (
        (BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT, BASE_STAGEB_RESUME1_IMPLEMENTATION_SUBJECT),
        (BASE_STAGEB_RESUME1_EVIDENCE_COMMIT, BASE_STAGEB_RESUME1_EVIDENCE_SUBJECT),
        (BASE_STAGEA_IMPLEMENTATION_COMMIT, BASE_STAGEA_IMPLEMENTATION_SUBJECT),
        (BASE_STAGEA_EVIDENCE_COMMIT, BASE_STAGEA_EVIDENCE_SUBJECT),
        (implementation_commit, IMPLEMENTATION_SUBJECT),
    )
    for commit, expected in subjects:
        if _git(repo, "show", "-s", "--format=%s", commit) != expected:
            raise StageCError("commit subject changed: {}".format(commit))
    if commit_name_status(repo, BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEB_RESUME1_IMPLEMENTATION_PATHS)
    ):
        raise StageCError("Stage-B Resume1 implementation paths changed")
    if commit_name_status(repo, BASE_STAGEB_RESUME1_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_STAGEB_RESUME1_EVIDENCE_PATHS)
    ):
        raise StageCError("Stage-B Resume1 evidence paths changed")
    if commit_name_status(repo, BASE_STAGEA_IMPLEMENTATION_COMMIT) != tuple(
        sorted(BASE_STAGEA_IMPLEMENTATION_PATHS)
    ):
        raise StageCError("Stage-A implementation paths changed")
    if commit_name_status(repo, BASE_STAGEA_EVIDENCE_COMMIT) != tuple(
        sorted(BASE_STAGEA_EVIDENCE_PATHS)
    ):
        raise StageCError("Stage-A evidence paths changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageCError("Stage-C implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageCError("origin/Experiment1 changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageCError("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageCError("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-C")

    for relative, expected_sha in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageCError("frozen source changed: {}".format(relative))
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageCError("Stage-C source missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageCError("Stage-C source differs from committed blob: {}".format(relative))

    report_path = repo / STAGEB_RESUME1_REPORT
    if not report_path.is_file() or sha256_file(report_path) != EXPECTED_STAGEB_RESUME1_REPORT_SHA256:
        raise StageCError("Stage-B Resume1 report changed")
    committed_report = _git_bytes(
        repo,
        "show",
        "{}:{}".format(BASE_STAGEB_RESUME1_EVIDENCE_COMMIT, STAGEB_RESUME1_REPORT),
    )
    if committed_report != report_path.read_bytes():
        raise StageCError("Stage-B Resume1 report differs from committed blob")
    validate_stageb_resume1_report(load_json(report_path))

    stagea_paths = (
        (STAGEA_PROBE, EXPECTED_STAGEA_PROBE_SHA256),
        (STAGEA_WORKER, EXPECTED_STAGEA_WORKER_SHA256),
        (STAGEA_SUMMARY, EXPECTED_STAGEA_SUMMARY_SHA256),
    )
    for relative, expected_sha in stagea_paths:
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageCError("Stage-A evidence changed: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(BASE_STAGEA_EVIDENCE_COMMIT, relative))
        if committed != path.read_bytes():
            raise StageCError("Stage-A evidence differs from committed blob: {}".format(relative))
    stagea.validate_probe_evidence(load_json(repo / STAGEA_PROBE))
    stagea_worker = load_json(repo / STAGEA_WORKER)
    stagea.validate_worker_evidence(stagea_worker)
    if stagea_worker.get("worker_result_sha256") != EXPECTED_STAGEA_WORKER_RESULT_SHA256:
        raise StageCError("Stage-A worker result SHA changed")
    stagea_summary = load_json(repo / STAGEA_SUMMARY)
    stagea.validate_summary(stagea_summary)
    if stagea_summary.get("summary_sha256") != EXPECTED_STAGEA_SUMMARY_SELF_SHA256:
        raise StageCError("Stage-A summary self-hash changed")

    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageCError("Stage-C terminal output already exists: {}".format(relative))
    if not allow_promoted_evidence:
        for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE):
            if (repo / relative).exists():
                raise StageCError("Stage-C promoted evidence already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_STAGEB_RESUME1_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "stageb_resume1_implementation_commit": BASE_STAGEB_RESUME1_IMPLEMENTATION_COMMIT,
        "stageb_resume1_evidence_commit": BASE_STAGEB_RESUME1_EVIDENCE_COMMIT,
        "stageb_resume1_report_sha256": EXPECTED_STAGEB_RESUME1_REPORT_SHA256,
        "stageb_resume1_audit_sha256": EXPECTED_STAGEB_RESUME1_AUDIT_SHA256,
        "stagea_implementation_commit": BASE_STAGEA_IMPLEMENTATION_COMMIT,
        "stagea_evidence_commit": BASE_STAGEA_EVIDENCE_COMMIT,
        "stagea_worker_result_sha256": EXPECTED_STAGEA_WORKER_RESULT_SHA256,
    }


def extract_probe_process_id(payload: Mapping[str, Any]) -> int:
    if "probe" in payload:
        raise StageCError("nested probe PID schema is forbidden")
    process_id = payload.get("process_id")
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
        raise StageCError("Stage-C probe process_id is invalid")
    return process_id


def make_probe_evidence(root: Path) -> Mapping[str, Any]:
    source = dict(stagea.resume2a.environment_probe_payload(Path(root).resolve()))
    process_id = int(source["process_id"])
    environment = stagea.resume2a.validate_probe_payload(source)
    stagea_probe = load_json(Path(root).resolve() / STAGEA_PROBE)
    stagea_environment = stagea.validate_probe_evidence(stagea_probe)
    if compact_json_bytes(environment) != compact_json_bytes(stagea_environment):
        raise StageCError("Stage-C environment differs from Stage-A environment")
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": process_id,
        "source_probe_schema": source.get("schema"),
        "environment": copy.deepcopy(dict(environment)),
        "environment_sha256": str(source["environment_sha256"]),
        "source_probe_payload_sha256": sha256_bytes(compact_json_bytes(source)),
        "stagea_environment_file_sha256": EXPECTED_STAGEA_PROBE_SHA256,
        "stagea_environment_exact": True,
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
        raise StageCError("Stage-C probe schema/verdict changed")
    extract_probe_process_id(payload)
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise StageCError("Stage-C probe environment missing")
    expected_environment_sha = stagea.resume2a.sha256_bytes(
        stagea.resume2a.stable_json_bytes(environment)
    )
    if payload.get("environment_sha256") != expected_environment_sha:
        raise StageCError("Stage-C environment SHA changed")
    if payload.get("stagea_environment_exact") is not True:
        raise StageCError("Stage-C environment identity changed")
    if payload.get("stagea_environment_file_sha256") != EXPECTED_STAGEA_PROBE_SHA256:
        raise StageCError("Stage-A environment provenance changed")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageCError("Stage-C probe is not durable")
    if payload.get("write_ahead_location_outside_git_worktree") is not True:
        raise StageCError("Stage-C probe write-ahead location changed")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "probe_evidence_sha256"}
        )
    )
    if payload.get("probe_evidence_sha256") != expected:
        raise StageCError("Stage-C probe self-hash changed")
    return environment


def make_worker_evidence(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_probe_evidence(probe_payload)
    stagea_worker = load_json(Path(root).resolve() / STAGEA_WORKER)
    stagea.validate_worker_evidence(stagea_worker)
    captured: MutableMapping[str, Any] = {
        "call_count": 0,
        "before_fingerprint": None,
        "after_fingerprint": None,
        "diagnostics": None,
    }
    original_stitch = kernel.stitch_fold_selected_procedure

    def instrumented_stitch(
        *,
        context: Mapping[str, Any],
        outer_outputs: Mapping[int, Mapping[int, Mapping[float, Mapping[str, Any]]]],
        outer_selections: Sequence[Mapping[str, Any]],
        target: np.ndarray,
        groups: np.ndarray,
        spec: Any,
    ) -> Any:
        captured["call_count"] = int(captured["call_count"]) + 1
        if captured["call_count"] != 1:
            raise StageCError("stitch instrumentation invoked more than once")
        before = outer_outputs_fingerprint(outer_outputs)
        original_result = original_stitch(
            context=context,
            outer_outputs=outer_outputs,
            outer_selections=outer_selections,
            target=target,
            groups=groups,
            spec=spec,
        )
        diagnostics = build_fold_resolved_records(
            context=context,
            outer_outputs=outer_outputs,
            outer_selections=outer_selections,
            target=target,
            groups=groups,
            spec=spec,
        )
        after = outer_outputs_fingerprint(outer_outputs)
        if before != after:
            raise StageCError("read-only instrumentation mutated outer outputs")
        captured["before_fingerprint"] = before
        captured["after_fingerprint"] = after
        captured["diagnostics"] = diagnostics
        return original_result

    kernel.stitch_fold_selected_procedure = instrumented_stitch
    try:
        kernel_payload = kernel.run_nested_oof(
            root=Path(root).resolve(),
            probe_payload={
                "mode": "environment_probe",
                "execution_verdict": "PASS",
                "process_id": extract_probe_process_id(probe_payload),
                "environment": copy.deepcopy(probe_payload["environment"]),
                "environment_sha256": probe_payload["environment_sha256"],
                "process_disposable": True,
                "cuda_initialization_allowed_in_this_process": True,
            },
            repository_head=str(repository_head),
            stageu_contract=stageu_contract,
        )
    finally:
        kernel.stitch_fold_selected_procedure = original_stitch
    kernel.validate_worker_payload(kernel_payload)
    if captured["call_count"] != 1 or not isinstance(captured["diagnostics"], Mapping):
        raise StageCError("fold-resolved instrumentation did not complete")
    if captured["before_fingerprint"] != captured["after_fingerprint"]:
        raise StageCError("outer-output identity changed")
    reproduction = compare_stagea_reproduction(kernel_payload, stagea_worker)
    diagnostics = captured["diagnostics"]
    fallback_folds = [
        int(item["outer_fold"])
        for item in kernel_payload["outer_fold_selections"]
        if item["diagnostic_fallback_used"] is True
    ]
    classification = classify_t10_fold_attribution(
        diagnostics["fold_timestep_records"], fallback_folds
    )
    diagnostic_counts = {
        "fold_timestep_record_count": int(diagnostics["fold_timestep_record_count"]),
        "fold_resolved_policy_metric_evaluation_count": int(
            diagnostics["fold_resolved_policy_metric_evaluation_count"]
        ),
        "risk_calibration_bin_count": int(diagnostics["risk_calibration_bin_count"]),
        "anonymous_group_aggregate_record_count": int(
            diagnostics["anonymous_group_aggregate_record_count"]
        ),
    }
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        **classification,
        "process_id": int(kernel_payload["process_id"]),
        "repository_head": str(repository_head),
        "environment_sha256": str(kernel_payload["environment_sha256"]),
        "cold_cuda_precheck": copy.deepcopy(kernel_payload["cold_cuda_precheck"]),
        "preregistration_contract": {
            "new_stage_identifier": PHASE,
            "stagea_execution_replayed_as_stagea": False,
            "frozen_nested_oof_mechanism_recomputed_in_new_stage": True,
            "stagea_terminal_result_used_for_policy_selection": False,
            "stagea_global_result_must_reproduce_exactly": True,
            "read_only_stitch_instrumentation": True,
            "outer_output_mutation_forbidden": True,
            "fold_selected_policy_only": True,
            "policy_bank_changed": False,
            "threshold_changed": False,
            "acceptance_threshold_changed": False,
            "selection_holdout_remains_closed": True,
            "frozen_probe_remains_closed": True,
            "durable_worker_evidence_required_before_controller_decoration": True,
            "aggregate_only_evidence": True,
        },
        "frozen_scientific_base": copy.deepcopy(kernel_payload["locked_scientific_base"]),
        "stagex_spec": copy.deepcopy(kernel_payload["stagex_spec"]),
        "policy_population": copy.deepcopy(kernel_payload["policy_population"]),
        "population": copy.deepcopy(kernel_payload["population"]),
        "stagea_reproduction_identity": reproduction,
        "instrumentation_identity": {
            "stitch_call_count": int(captured["call_count"]),
            "outer_outputs_before_sha256": captured["before_fingerprint"],
            "outer_outputs_after_sha256": captured["after_fingerprint"],
            "outer_outputs_unchanged": True,
        },
        "outer_fold_selections": copy.deepcopy(kernel_payload["outer_fold_selections"]),
        "inner_selection_modal_policy_diagnostic": copy.deepcopy(
            kernel_payload["inner_selection_modal_policy_diagnostic"]
        ),
        "global_outer_crossfit_fold_selected_policy_records": copy.deepcopy(
            kernel_payload["outer_crossfit_fold_selected_policy_records"]
        ),
        "global_outer_crossfit_baseline_records": copy.deepcopy(
            kernel_payload["outer_crossfit_baseline_records"]
        ),
        "full_objective_oof_fixed_policy_selection": copy.deepcopy(
            kernel_payload["full_objective_oof_fixed_policy_selection"]
        ),
        "procedure_gate_totals": copy.deepcopy(kernel_payload["procedure_gate_totals"]),
        "fold_resolved_attribution": copy.deepcopy(diagnostics),
        "execution_counts": copy.deepcopy(kernel_payload["execution_counts"]),
        "diagnostic_counts": diagnostic_counts,
        "evidence_privacy_contract": {
            "raw_group_identifiers_persisted": False,
            "row_indices_persisted": False,
            "row_level_metrics_persisted": False,
            "candidate_tensor_persisted": False,
            "risk_probability_tensor_persisted": False,
            "anonymous_group_hash_namespace": GROUP_HASH_NAMESPACE,
            "condition_level_aggregates_persisted": True,
            "anonymous_group_level_aggregates_persisted": True,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["worker_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_worker_evidence(result)
    return result


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageCError("Stage-C worker schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageCError("Stage-C attribution cannot be READY")
    if payload.get("selected_configuration") is not None:
        raise StageCError("Stage-C selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageCError("Stage-C retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageCError("Stage-C re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageCError("Stage-C cumulative holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageCError("Stage-C accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageCError("Stage-C authorized rerun")
    if payload.get("execution_counts") != EXPECTED_EXECUTION_COUNTS:
        raise StageCError("Stage-C scientific execution counts changed")
    identity = payload.get("stagea_reproduction_identity")
    if not isinstance(identity, Mapping) or identity.get("all_exact") is not True:
        raise StageCError("Stage-C Stage-A reproduction identity failed")
    instrumentation = payload.get("instrumentation_identity")
    if not isinstance(instrumentation, Mapping):
        raise StageCError("Stage-C instrumentation identity missing")
    if instrumentation.get("stitch_call_count") != 1:
        raise StageCError("Stage-C stitch call count changed")
    if instrumentation.get("outer_outputs_unchanged") is not True:
        raise StageCError("Stage-C mutated outer outputs")
    if instrumentation.get("outer_outputs_before_sha256") != instrumentation.get(
        "outer_outputs_after_sha256"
    ):
        raise StageCError("Stage-C outer-output fingerprints differ")
    diagnostics = payload.get("fold_resolved_attribution")
    if not isinstance(diagnostics, Mapping):
        raise StageCError("Stage-C fold diagnostics missing")
    if diagnostics.get("fold_timestep_record_count") != 18:
        raise StageCError("Stage-C fold record count changed")
    records = diagnostics.get("fold_timestep_records")
    if not isinstance(records, Sequence) or len(records) != 18:
        raise StageCError("Stage-C fold record population changed")
    if sum(len(item["risk_calibration"]["bins"]) for item in records) != 90:
        raise StageCError("Stage-C calibration bin count changed")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageCError("Stage-C worker is not durable")
    if payload.get("write_ahead_location_outside_git_worktree") is not True:
        raise StageCError("Stage-C write-ahead location changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCError("Stage-C crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected:
        raise StageCError("Stage-C worker self-hash changed")


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
    probe_pid = extract_probe_process_id(probe)
    worker_pid = worker.get("process_id")
    if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid <= 0:
        raise StageCError("Stage-C worker PID is invalid")
    if probe_pid == worker_pid:
        raise StageCError("Stage-C probe and worker PIDs are not distinct")
    probe_sha = sha256_file(probe_path)
    worker_sha = sha256_file(worker_path)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": dict(repository),
        "durable_evidence_protocol": {
            "probe_evidence_path": PROBE_EVIDENCE,
            "probe_evidence_file_sha256": probe_sha,
            "probe_evidence_self_sha256": probe["probe_evidence_sha256"],
            "worker_evidence_path": WORKER_EVIDENCE,
            "worker_evidence_file_sha256": worker_sha,
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_result_persisted_before_controller_decoration": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_evidence_promoted_byte_exact": True,
            "worker_result_sha_recorded_before_cleanup": True,
            "controller_recomputed_worker_science": False,
            "write_ahead_files_write_once": True,
        },
        "process_topology": {
            "environment_probe_process_count": 1,
            "cold_science_worker_process_count": 1,
            "probe_process_id": probe_pid,
            "science_worker_process_id": worker_pid,
            "processes_distinct": True,
            "probe_pid_schema": "top_level_process_id",
            "child_pythonpath_explicit": True,
        },
        "stagea_reproduction_identity": copy.deepcopy(
            worker["stagea_reproduction_identity"]
        ),
        "instrumentation_identity": copy.deepcopy(worker["instrumentation_identity"]),
        "t10_fold_attribution": {
            "failed_outer_folds": copy.deepcopy(worker["t10_failed_outer_folds"]),
            "fold_categories": copy.deepcopy(worker["fold_categories"]),
            "stagea_inner_fallback_folds": copy.deepcopy(
                worker["stagea_inner_fallback_folds"]
            ),
            "failed_fold_fallback_overlap": copy.deepcopy(
                worker["failed_fold_fallback_overlap"]
            ),
            "all_failed_folds_are_fallback_folds": worker[
                "all_failed_folds_are_fallback_folds"
            ],
            "all_fallback_folds_fail_outer_acceptance": worker[
                "all_fallback_folds_fail_outer_acceptance"
            ],
        },
        "execution_counts": copy.deepcopy(worker["execution_counts"]),
        "diagnostic_counts": copy.deepcopy(worker["diagnostic_counts"]),
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
        raise StageCError("Stage-C summary schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageCError("Stage-C summary cannot be READY")
    if payload.get("selected_configuration") is not None:
        raise StageCError("Stage-C summary selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageCError("Stage-C summary retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageCError("Stage-C summary re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageCError("Stage-C summary holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageCError("Stage-C summary accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageCError("Stage-C summary authorized rerun")
    protocol = payload.get("durable_evidence_protocol")
    if not isinstance(protocol, Mapping):
        raise StageCError("Stage-C durable protocol missing")
    for key in (
        "worker_result_persisted_before_controller_decoration",
        "external_write_ahead_outside_git_worktree",
        "repository_evidence_promoted_byte_exact",
        "worker_result_sha_recorded_before_cleanup",
        "write_ahead_files_write_once",
    ):
        if protocol.get(key) is not True:
            raise StageCError("Stage-C durable protocol changed: {}".format(key))
    if protocol.get("controller_recomputed_worker_science") is not False:
        raise StageCError("Stage-C controller recomputed science")
    topology = payload.get("process_topology")
    if not isinstance(topology, Mapping) or topology.get("processes_distinct") is not True:
        raise StageCError("Stage-C process topology changed")
    identity = payload.get("stagea_reproduction_identity")
    if not isinstance(identity, Mapping) or identity.get("all_exact") is not True:
        raise StageCError("Stage-C summary reproduction identity failed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageCError("Stage-C summary crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageCError("Stage-C summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Optional[Path] = None,
    worker_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    probe_available = bool(probe_path is not None and Path(probe_path).is_file())
    worker_available = bool(worker_path is not None and Path(worker_path).is_file())
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagec_execution_contract_failed",
        "required_next_path": (
            "FINALIZE_R259_STAGEC_DURABLE_WORKER_EVIDENCE_WITHOUT_SCIENCE_"
            "REEXECUTION"
            if worker_available
            else "DESIGN_ADD_ONLY_R259_STAGEC_EXECUTION_RECOVERY_BEFORE_SCIENCE_IF_WORKER_NOT_STARTED"
        ),
        "primary_failure_locus": (
            "controller_after_durable_worker_evidence"
            if worker_available
            else "stagec_execution_contract"
        ),
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "durable_probe_evidence_available": probe_available,
        "durable_worker_evidence_available": worker_available,
        "science_reexecution_authorized": False if worker_available else False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    if probe_available:
        payload["probe_evidence_sha256"] = sha256_file(Path(probe_path))
    if worker_available:
        payload["worker_evidence_sha256"] = sha256_file(Path(worker_path))
        try:
            worker = load_json(Path(worker_path))
            validate_worker_evidence(worker)
            payload["worker_result_sha256"] = worker["worker_result_sha256"]
            payload["durable_worker_evidence_validated"] = True
        except BaseException as validation_error:
            payload["durable_worker_evidence_validated"] = False
            payload["worker_validation_error"] = str(validation_error)
    return payload
