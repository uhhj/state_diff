"""Functional-prior adapter for Phase3.14b-r2.5.4 Resume2.

This module does not train a residual model and does not alter the historical
r2.5.4 implementation.  It verifies the committed Resume1 prior audit and
validates the single fresh RTX-4090 prior produced by the original r2.5.4
runner before that prior is allowed to enter the robot-proxy attribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

PHASE = "Phase3.14b-r2.5.4-Resume2"
RESUME_GENERATION = 2
RESUME2_SCHEMA = "phase314b_r254_resume2_functional_prior_contract_v1"

EXPECTED_BASELINE_COMMIT = "2ae075b20be4a0e960287d05264a46882665db15"
EXPECTED_R254_IMPLEMENTATION_COMMIT = "d315ac52f193eb6c0d5fc195b2cfa229dc7391e1"
EXPECTED_R254_BLOCKED_COMMIT = "d75988089ae6ba5e3fe65e44fae28eadcf02cfdc"
EXPECTED_R253_FINAL_COMMIT = "c35c374f61f9b4dc2ca4cdaa76b12ddda71be994"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
EXPECTED_CONTRACT_SHA256 = "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
EXPECTED_R253_PILOT_SHA256 = "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"

HISTORICAL_PRIOR_STATE_SHA256 = "083278d6b0710ddc3863d822978973f29842bc757edf73ff03a93bb154f5665d"
CURRENT_DEVICE_PRIOR_STATE_SHA256 = "8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904"
CURRENT_DEVICE_PRIOR_PREDICTION_SHA256 = "70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9"
CURRENT_DEVICE_PRIOR_Z_MSE = 0.17507688701152802
EXPECTED_CURRENT_GPU = "NVIDIA GeForce RTX 4090"
EXPECTED_HISTORICAL_GPU = "NVIDIA GeForce RTX 4080 SUPER"
EXPECTED_RESUME1_ROOT_CAUSE = "phase314b_r254_resume1_cross_device_bitwise_sha_contract_overstrict"
EXPECTED_NEXT_STAGE = "phase3_14b_r254_resume2_robot_proxy_attribution_functional_prior_contract"

RESUME1_SUMMARY_PATH = "reports/phase3_14b_r254_resume1_summary.json"
RESUME1_AUDIT_PATH = "reports/phase3_14b_r254_resume1_prior_audit_summary.json"
RESUME1_EVIDENCE_PATH = "reports/phase3_14b_r254_resume1_prior_repeat_evidence.json"
R253_PILOT_PATH = "reports/phase3_14b_r253_pilot_summary.json"
ORIGINAL_R254_PREFLIGHT_PATH = "reports/phase3_14b_r254_preflight_summary.json"
ORIGINAL_R254_BLOCKED_SUMMARY_PATH = "reports/phase3_14b_r254_blocked_summary.json"
ORIGINAL_R254_BLOCKED_REPORT_PATH = "reports/phase3_14b_r254_blocked_report.md"
RESUME2_PREFLIGHT_PATH = "reports/phase3_14b_r254_resume2_preflight_summary.json"
RESUME2_PILOT_PATH = "reports/phase3_14b_r254_resume2_pilot_summary.json"
RESUME2_BLOCKED_SUMMARY_PATH = "reports/phase3_14b_r254_resume2_blocked_summary.json"
RESUME2_BLOCKED_REPORT_PATH = "reports/phase3_14b_r254_resume2_blocked_report.md"
FINAL_SUMMARY_PATH = "reports/phase3_14b_r254_summary.json"
FINAL_REPORT_PATH = "reports/phase3_14b_r254_report.md"

IMMUTABLE_PATHS_AT_BASELINE = (
    "ccda_phase3/phase314b_r254_robot_proxy_attribution.py",
    "scripts/phase3_14b_r254_preflight.py",
    "scripts/phase3_14b_r254_run_pilot.py",
    "scripts/phase3_14b_r254_finalize.py",
    "scripts/phase3_14b_r254_run.sh",
    "tests/test_phase314b_r254_robot_proxy_attribution.py",
    "ccda_phase3/phase314b_r254_resume1_prior_determinism.py",
    "scripts/phase3_14b_r254_resume1_prior_worker.py",
    "scripts/phase3_14b_r254_resume1_preflight.py",
    "scripts/phase3_14b_r254_resume1_run_audit.py",
    "scripts/phase3_14b_r254_resume1_finalize.py",
    "scripts/phase3_14b_r254_resume1_run.sh",
    "tests/test_phase314b_r254_resume1_prior_determinism.py",
    RESUME1_SUMMARY_PATH,
    RESUME1_AUDIT_PATH,
    RESUME1_EVIDENCE_PATH,
    ORIGINAL_R254_PREFLIGHT_PATH,
    ORIGINAL_R254_BLOCKED_SUMMARY_PATH,
    ORIGINAL_R254_BLOCKED_REPORT_PATH,
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r254_resume2_functional_prior.py",
    "scripts/phase3_14b_r254_resume2_preflight.py",
    "scripts/phase3_14b_r254_resume2_run_pilot.py",
    "scripts/phase3_14b_r254_resume2_finalize.py",
    "scripts/phase3_14b_r254_resume2_blocked.py",
    "scripts/phase3_14b_r254_resume2_run.sh",
    "tests/test_phase314b_r254_resume2_functional_prior.py",
)


@dataclass(frozen=True)
class FunctionalPriorSpec:
    """Frozen acceptance bounds for the already-audited RTX-4090 prior."""

    prior_z_mse_absolute_tolerance: float = 1.0e-10
    prior_z_mse_relative_tolerance: float = 1.0e-8
    repeat_count: int = 3

    def validate(self) -> None:
        if self.repeat_count != 3:
            raise ValueError("Resume1 contract requires exactly three repeats")
        for key, value in asdict(self).items():
            if key == "repeat_count":
                continue
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"invalid functional-prior tolerance {key}: {value}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prediction_sha256(value: Any) -> str:
    """Resume1 prediction hash: contiguous float32 raw bytes."""

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return sha256_bytes(array.tobytes(order="C"))


def source_sha256(root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=Path(root),
        stderr=subprocess.STDOUT,
    )


def require_ancestor(root: Path, commit: str) -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=Path(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def assert_paths_match_commit(
    root: Path, *, commit: str, paths: Sequence[str]
) -> Dict[str, str]:
    observed: Dict[str, str] = {}
    mismatch: Dict[str, Dict[str, str]] = {}
    for relative in paths:
        current_path = Path(root) / relative
        if not current_path.is_file():
            raise RuntimeError(f"immutable path missing: {relative}")
        current = current_path.read_bytes()
        historical = git_blob_bytes(root, commit, relative)
        current_sha = sha256_bytes(current)
        historical_sha = sha256_bytes(historical)
        observed[relative] = current_sha
        if current_sha != historical_sha:
            mismatch[relative] = {
                "current_sha256": current_sha,
                "baseline_sha256": historical_sha,
            }
    if mismatch:
        raise RuntimeError(f"immutable r2.5.4/Resume1 evidence changed: {mismatch}")
    return observed


def assert_only_allowed_worktree_paths(
    root: Path, allowed_paths: Sequence[str]
) -> None:
    allowed = {str(Path(value).as_posix()) for value in allowed_paths}
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    observed = set()
    for line in output.splitlines():
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        observed.add(str(Path(value).as_posix()))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError("unexpected worktree paths: " + ", ".join(unexpected))


def _walk_key(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for current_key, current_value in value.items():
            if str(current_key) == key:
                yield current_value
            yield from _walk_key(current_value, key)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk_key(item, key)


def _bool_candidates(value: Mapping[str, Any], keys: Sequence[str]) -> Sequence[bool]:
    result = []
    for key in keys:
        for item in _walk_key(value, key):
            if isinstance(item, (bool, np.bool_)):
                result.append(bool(item))
    return result


def _string_candidates(value: Mapping[str, Any], keys: Sequence[str]) -> Sequence[str]:
    result = []
    for key in keys:
        for item in _walk_key(value, key):
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                result.extend(str(entry) for entry in item if isinstance(entry, str))
    return result


def _float_candidates(value: Mapping[str, Any], keys: Sequence[str]) -> Sequence[float]:
    result = []
    for key in keys:
        for item in _walk_key(value, key):
            if isinstance(item, (int, float, np.number)) and not isinstance(item, bool):
                number = float(item)
                if math.isfinite(number):
                    result.append(number)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                for entry in item:
                    if isinstance(entry, (int, float, np.number)) and not isinstance(entry, bool):
                        number = float(entry)
                        if math.isfinite(number):
                            result.append(number)
    return result


def symmetric_relative_error(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(
        abs(float(observed)), abs(float(expected)), 1.0e-12
    )


def verify_resume1_functional_contract(
    summary: Mapping[str, Any],
    audit: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    spec: FunctionalPriorSpec = FunctionalPriorSpec(),
) -> Dict[str, Any]:
    """Validate the committed Resume1 contract without depending on one schema path."""

    spec.validate()
    documents = {"summary": summary, "audit": audit, "evidence": evidence}

    root_values = _string_candidates(summary, ("root_cause",))
    if EXPECTED_RESUME1_ROOT_CAUSE not in root_values:
        raise RuntimeError(f"Resume1 root cause mismatch: {root_values}")

    next_values = _string_candidates(summary, ("next_stage", "next"))
    if EXPECTED_NEXT_STAGE not in next_values:
        raise RuntimeError(f"Resume1 next-stage mismatch: {next_values}")

    verdicts = _string_candidates(summary, ("verdict",))
    if "PASS" not in verdicts:
        raise RuntimeError(f"Resume1 verdict is not PASS: {verdicts}")

    bool_checks = {
        "functional_prior_contract_supported": (
            "functional_prior_contract_supported",
        ),
        "same_device_functional_equivalence": (
            "same_device_functional_equivalence",
            "functional_equivalence_pass",
        ),
        "same_device_exact_state_sha": (
            "same_device_exact_sha",
            "all_state_sha_exact",
        ),
        "same_device_exact_prediction_sha": (
            "all_prediction_sha_exact",
        ),
        "historical_functional_fingerprint": (
            "historical_functional_fingerprint",
            "functional_fingerprint_pass",
        ),
    }
    resolved_bools: Dict[str, bool] = {}
    combined = {"summary": summary, "audit": audit, "evidence": evidence}
    for name, keys in bool_checks.items():
        candidates = list(_bool_candidates(combined, keys))
        if not candidates or not any(candidates):
            raise RuntimeError(f"Resume1 Boolean contract missing/false: {name}={candidates}")
        resolved_bools[name] = True

    state_hashes = _string_candidates(
        combined,
        (
            "observed_state_sha256",
            "observed_state_sha256_values",
            "prior_state_sha256",
        ),
    )
    current_state_count = sum(value == CURRENT_DEVICE_PRIOR_STATE_SHA256 for value in state_hashes)
    if current_state_count < spec.repeat_count:
        raise RuntimeError(
            "Resume1 current-device state SHA matrix incomplete: "
            f"count={current_state_count}, values={state_hashes}"
        )

    prediction_hashes = _string_candidates(
        combined,
        (
            "observed_prediction_sha256",
            "prior_prediction_sha256",
            "prediction_sha256",
        ),
    )
    current_prediction_count = sum(
        value == CURRENT_DEVICE_PRIOR_PREDICTION_SHA256 for value in prediction_hashes
    )
    if current_prediction_count < spec.repeat_count:
        raise RuntimeError(
            "Resume1 current-device prediction SHA matrix incomplete: "
            f"count={current_prediction_count}, values={prediction_hashes}"
        )

    mse_values = _float_candidates(
        combined,
        (
            "prior_z_mse_observed_values",
            "prior_z_mse",
            "prior_z_mse_observed_median",
        ),
    )
    matching_mse = [
        value
        for value in mse_values
        if abs(value - CURRENT_DEVICE_PRIOR_Z_MSE)
        <= spec.prior_z_mse_absolute_tolerance
        or symmetric_relative_error(value, CURRENT_DEVICE_PRIOR_Z_MSE)
        <= spec.prior_z_mse_relative_tolerance
    ]
    if len(matching_mse) < spec.repeat_count:
        raise RuntimeError(
            "Resume1 current-device prior z-MSE matrix incomplete: "
            f"matches={matching_mse}, all={mse_values}"
        )

    historical_hashes = _string_candidates(
        combined,
        ("expected_prior_state_sha256", "historical_prior_sha256"),
    )
    if HISTORICAL_PRIOR_STATE_SHA256 not in historical_hashes:
        raise RuntimeError("historical prior SHA is missing from Resume1 evidence")

    return {
        "schema": RESUME2_SCHEMA,
        "resume1_verdict": "PASS",
        "resume1_root_cause": EXPECTED_RESUME1_ROOT_CAUSE,
        "resume1_next_stage": EXPECTED_NEXT_STAGE,
        "functional_prior_contract_supported": True,
        "same_device_exact_state_sha": True,
        "same_device_exact_prediction_sha": True,
        "same_device_functional_equivalence": True,
        "historical_functional_fingerprint": True,
        "historical_prior_state_sha256": HISTORICAL_PRIOR_STATE_SHA256,
        "current_device_prior_state_sha256": CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "current_device_prior_prediction_sha256": CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
        "current_device_prior_z_mse": CURRENT_DEVICE_PRIOR_Z_MSE,
        "state_sha_evidence_count": current_state_count,
        "prediction_sha_evidence_count": current_prediction_count,
        "prior_z_mse_evidence_count": len(matching_mse),
        "resolved_boolean_contract": resolved_bools,
        "source_documents": list(documents),
    }


def validate_fresh_prior_snapshot(
    snapshot: MutableMapping[str, Any],
    *,
    spec: FunctionalPriorSpec = FunctionalPriorSpec(),
) -> Dict[str, Any]:
    """Require exact same-device reproduction before residual training starts."""

    spec.validate()
    observed_state_sha = str(snapshot.get("prior_state_sha256", ""))
    if observed_state_sha != CURRENT_DEVICE_PRIOR_STATE_SHA256:
        raise RuntimeError(
            "fresh prior did not reproduce the Resume1 RTX-4090 state SHA: "
            f"observed={observed_state_sha}"
        )

    prediction = snapshot.get("_prior_prediction_z")
    if prediction is None:
        raise RuntimeError("fresh prior snapshot lacks _prior_prediction_z")
    observed_prediction_sha = prediction_sha256(prediction)
    if observed_prediction_sha != CURRENT_DEVICE_PRIOR_PREDICTION_SHA256:
        raise RuntimeError(
            "fresh prior did not reproduce the Resume1 prediction SHA: "
            f"observed={observed_prediction_sha}"
        )

    observed_mse = float(snapshot.get("prior_z_mse", float("nan")))
    if not math.isfinite(observed_mse):
        raise RuntimeError("fresh prior z-MSE is non-finite")
    absolute_error = abs(observed_mse - CURRENT_DEVICE_PRIOR_Z_MSE)
    relative_error = symmetric_relative_error(observed_mse, CURRENT_DEVICE_PRIOR_Z_MSE)
    mse_pass = bool(
        absolute_error <= spec.prior_z_mse_absolute_tolerance
        or relative_error <= spec.prior_z_mse_relative_tolerance
    )
    if not mse_pass:
        raise RuntimeError(
            "fresh prior z-MSE did not reproduce Resume1: "
            f"observed={observed_mse}, expected={CURRENT_DEVICE_PRIOR_Z_MSE}"
        )

    result = {
        "schema": RESUME2_SCHEMA,
        "historical_cross_device_state_sha256": HISTORICAL_PRIOR_STATE_SHA256,
        "same_device_expected_state_sha256": CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "observed_state_sha256": observed_state_sha,
        "state_sha_exact": True,
        "same_device_expected_prediction_sha256": CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
        "observed_prediction_sha256": observed_prediction_sha,
        "prediction_sha_exact": True,
        "expected_prior_z_mse": CURRENT_DEVICE_PRIOR_Z_MSE,
        "observed_prior_z_mse": observed_mse,
        "prior_z_mse_absolute_error": absolute_error,
        "prior_z_mse_relative_error": relative_error,
        "prior_z_mse_pass": True,
        "historical_cross_device_functional_fingerprint_required": True,
        "functional_prior_contract_pass": True,
    }
    snapshot["prior_prediction_sha256"] = observed_prediction_sha
    snapshot["resume2_functional_prior_validation"] = result
    return result


def augment_pilot_payload(
    payload: Mapping[str, Any],
    *,
    functional_contract: Mapping[str, Any],
    fresh_prior_validation: Mapping[str, Any],
    resume2_source_hashes: Mapping[str, str],
    resume2_preflight_path: str = RESUME2_PREFLIGHT_PATH,
) -> Dict[str, Any]:
    result = dict(payload)
    result["resume_generation"] = RESUME_GENERATION
    result["resume2_schema"] = RESUME2_SCHEMA
    result["resume2_preflight_report"] = resume2_preflight_path
    result["resume2_source_sha256"] = dict(resume2_source_hashes)
    result["functional_prior_contract"] = dict(functional_contract)
    result["fresh_prior_validation"] = dict(fresh_prior_validation)
    result["historical_exact_prior_sha_required_for_cross_device"] = False
    result["same_device_exact_prior_sha_required"] = True
    result["robot_proxy_attribution_run"] = True
    result["reverse_sampling_rerun"] = False
    result["train_only_recommendation"] = None
    result["selected_configuration"] = None
    return result


def validate_resume2_pilot_contract(payload: Mapping[str, Any]) -> None:
    if int(payload.get("resume_generation", -1)) != RESUME_GENERATION:
        raise RuntimeError("Resume2 pilot generation mismatch")
    if payload.get("resume2_schema") != RESUME2_SCHEMA:
        raise RuntimeError("Resume2 pilot schema mismatch")
    contract = payload.get("functional_prior_contract")
    if not isinstance(contract, Mapping) or not bool(
        contract.get("functional_prior_contract_supported")
    ):
        raise RuntimeError("Resume2 pilot lacks the committed functional-prior contract")
    fresh = payload.get("fresh_prior_validation")
    if not isinstance(fresh, Mapping) or not bool(fresh.get("functional_prior_contract_pass")):
        raise RuntimeError("Resume2 pilot fresh-prior validation failed")
    shared = payload.get("shared_prior")
    if not isinstance(shared, Mapping):
        raise RuntimeError("Resume2 pilot shared_prior is missing")
    if str(shared.get("prior_state_sha256")) != CURRENT_DEVICE_PRIOR_STATE_SHA256:
        raise RuntimeError("Resume2 pilot shared-prior state SHA mismatch")
    if str(shared.get("prior_prediction_sha256")) != CURRENT_DEVICE_PRIOR_PREDICTION_SHA256:
        raise RuntimeError("Resume2 pilot shared-prior prediction SHA mismatch")
    if payload.get("train_only_recommendation") is not None:
        raise RuntimeError("Resume2 must not recommend a configuration")
    if payload.get("selected_configuration") is not None:
        raise RuntimeError("Resume2 must not select a configuration")
    if bool(payload.get("reverse_sampling_rerun")):
        raise RuntimeError("Resume2 must not rerun reverse sampling")


def compact_functional_prior_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "resume_generation": RESUME_GENERATION,
        "schema": RESUME2_SCHEMA,
        "resume1_root_cause": payload.get("resume1_root_cause"),
        "functional_prior_contract_supported": bool(
            payload.get("functional_prior_contract_supported")
        ),
        "historical_prior_state_sha256": payload.get(
            "historical_prior_state_sha256"
        ),
        "current_device_prior_state_sha256": payload.get(
            "current_device_prior_state_sha256"
        ),
        "current_device_prior_prediction_sha256": payload.get(
            "current_device_prior_prediction_sha256"
        ),
        "current_device_prior_z_mse": float(
            payload.get("current_device_prior_z_mse", float("nan"))
        ),
        "same_device_exact_state_sha": bool(
            payload.get("same_device_exact_state_sha")
        ),
        "same_device_exact_prediction_sha": bool(
            payload.get("same_device_exact_prediction_sha")
        ),
        "same_device_functional_equivalence": bool(
            payload.get("same_device_functional_equivalence")
        ),
        "historical_functional_fingerprint": bool(
            payload.get("historical_functional_fingerprint")
        ),
    }
