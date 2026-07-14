"""Resume1 helpers for Phase3.14b-r2.5.3 finalizer-only correction.

The original r2.5.3 pilot asked ``paired_one_step_attribution`` for the
nonexistent key ``exact_v_oracle_audit``.  The r2.5.2 producer actually emits
``legacy_oracle_audit`` and persists its result inside
``pipeline_controls.oracle_legacy_gate_pass``.  This module repairs only that
adapter contract in memory.  It never edits or regenerates the committed pilot.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

from ccda_phase3.phase314b_r253_gate_separation import (
    DIAGNOSTIC_OBJECTIVE_NAMES,
    HISTORICAL_REPRODUCTION_SCHEMA,
    classify_gate_separation,
    variant_supported_mechanisms,
)

PHASE = "Phase3.14b-r2.5.3-Resume1"
BASE_BLOCKED_COMMIT = "2a221b16c4588feaf6d8af75ce24b7109fe48433"
PILOT_IMPLEMENTATION_COMMIT = "7a628a85e015b26e354703a64ae0b9792dfe85ca"
PILOT_SUMMARY_RELATIVE = "reports/phase3_14b_r253_pilot_summary.json"
PILOT_SUMMARY_SHA256 = (
    "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
)
ORIGINAL_PREFLIGHT_RELATIVE = "reports/phase3_14b_r253_preflight_summary.json"
ORIGINAL_BLOCKED_SUMMARY_RELATIVE = "reports/phase3_14b_r253_blocked_summary.json"
ORIGINAL_BLOCKED_REPORT_RELATIVE = "reports/phase3_14b_r253_blocked_report.md"
RESUME_PREFLIGHT_RELATIVE = "reports/phase3_14b_r253_resume_preflight_summary.json"
RESUME_BLOCKED_SUMMARY_RELATIVE = "reports/phase3_14b_r253_resume_blocked_summary.json"
RESUME_BLOCKED_REPORT_RELATIVE = "reports/phase3_14b_r253_resume_blocked_report.md"
FINAL_SUMMARY_RELATIVE = "reports/phase3_14b_r253_summary.json"
FINAL_REPORT_RELATIVE = "reports/phase3_14b_r253_report.md"

CORRECTION_SCHEMA = "phase314b_r253_oracle_adapter_resume_v2"
RESUME_PROVENANCE_SCHEMA = "phase314b_r253_finalizer_only_resume_v1"

ORIGINAL_R253_PATHS: Tuple[str, ...] = (
    "ccda_phase3/phase314b_r253_gate_separation.py",
    "scripts/phase3_14b_r253_preflight.py",
    "scripts/phase3_14b_r253_run_pilot.py",
    "scripts/phase3_14b_r253_finalize.py",
    "scripts/phase3_14b_r253_run.sh",
    "tests/test_phase314b_r253_gate_separation.py",
    ORIGINAL_PREFLIGHT_RELATIVE,
    ORIGINAL_BLOCKED_SUMMARY_RELATIVE,
    ORIGINAL_BLOCKED_REPORT_RELATIVE,
    PILOT_SUMMARY_RELATIVE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def assert_finite_json(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def run_git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def git_text(root: Path, *args: str) -> str:
    return run_git(root, *args).decode("utf-8").strip()


def assert_commit_is_ancestor(root: Path, commit: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def committed_blob(root: Path, commit: str, relative: str) -> bytes:
    return run_git(root, "show", f"{commit}:{relative}")


def assert_path_matches_commit(root: Path, commit: str, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"required historical path is missing: {relative}")
    expected = committed_blob(root, commit, relative)
    actual = path.read_bytes()
    if actual != expected:
        raise RuntimeError(f"historical path changed after {commit}: {relative}")
    return sha256_bytes(actual)


def historical_evidence_sha256(root: Path) -> Dict[str, str]:
    return {
        relative: assert_path_matches_commit(root, BASE_BLOCKED_COMMIT, relative)
        for relative in ORIGINAL_R253_PATHS
    }


def tracked_or_untracked_paths(root: Path) -> list[str]:
    output = git_text(root, "status", "--porcelain", "--untracked-files=all")
    paths: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw)
    return sorted(paths)


def assert_only_allowed_paths(root: Path, allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    observed = set(tracked_or_untracked_paths(root))
    unexpected = sorted(observed - allowed_set)
    if unexpected:
        raise RuntimeError(f"unexpected worktree paths: {unexpected}")


def validate_committed_pilot(root: Path, pilot: Mapping[str, Any]) -> Dict[str, Any]:
    pilot_path = root / PILOT_SUMMARY_RELATIVE
    actual_sha = sha256_file(pilot_path)
    if actual_sha != PILOT_SUMMARY_SHA256:
        raise RuntimeError(
            "pilot summary SHA mismatch: "
            f"expected={PILOT_SUMMARY_SHA256} observed={actual_sha}"
        )
    if pilot.get("verdict") != "PASS":
        raise RuntimeError("committed pilot verdict is not PASS")
    variants = pilot.get("variants")
    if not isinstance(variants, Mapping):
        raise RuntimeError("pilot variants must be a mapping")
    if set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise RuntimeError("committed pilot diagnostic matrix changed")
    return {
        "pilot_summary": PILOT_SUMMARY_RELATIVE,
        "pilot_summary_sha256": actual_sha,
        "pilot_implementation_commit": PILOT_IMPLEMENTATION_COMMIT,
        "blocked_report_commit": BASE_BLOCKED_COMMIT,
    }


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must be a mapping")
    return value


def _check_is_pass(item: Mapping[str, Any], path: str) -> bool:
    if "pass" not in item:
        raise RuntimeError(f"{path}.pass is missing")
    return bool(item["pass"])


def repair_r252_reproduction(
    value: Mapping[str, Any],
    *,
    objective_name: str,
) -> Dict[str, Any]:
    """Correct the one known oracle-key adapter defect in a persisted variant.

    The correction is deliberately narrow.  It is accepted only when:

    * the original reproduction schema is present;
    * every non-oracle historical comparison passed;
    * the exact-v check is the known missing-key signature
      (observed=False, expected=True, pass=False);
    * the persisted current pipeline proves the oracle legacy gate and complete
      pipeline passed;
    * reconstruction metric reproduction passed.
    """

    corrected = copy.deepcopy(dict(value))
    reproduction = _require_mapping(
        corrected.get("r252_one_step_reproduction"),
        f"variants.{objective_name}.r252_one_step_reproduction",
    )
    if reproduction.get("schema") != HISTORICAL_REPRODUCTION_SCHEMA:
        raise RuntimeError(f"historical reproduction schema mismatch: {objective_name}")
    checks = _require_mapping(
        reproduction.get("checks"),
        f"variants.{objective_name}.r252_one_step_reproduction.checks",
    )
    oracle_check = _require_mapping(
        checks.get("exact_v_oracle_pass"),
        f"variants.{objective_name}.checks.exact_v_oracle_pass",
    )
    known_signature = (
        set(oracle_check) == {"observed", "expected", "pass"}
        and oracle_check.get("observed") is False
        and oracle_check.get("expected") is True
        and oracle_check.get("pass") is False
    )
    if not known_signature:
        raise RuntimeError(
            f"unexpected exact-v mismatch signature for {objective_name}: "
            f"{dict(oracle_check)}"
        )

    failed_non_oracle = [
        str(name)
        for name, item in checks.items()
        if name != "exact_v_oracle_pass"
        and not _check_is_pass(
            _require_mapping(item, f"variants.{objective_name}.checks.{name}"),
            f"variants.{objective_name}.checks.{name}",
        )
    ]
    if failed_non_oracle:
        raise RuntimeError(
            f"non-oracle r2.5.2 reproduction checks failed for {objective_name}: "
            f"{failed_non_oracle}"
        )

    reconstruction = _require_mapping(
        reproduction.get("reconstruction_metrics"),
        f"variants.{objective_name}.reconstruction_metrics",
    )
    if not _check_is_pass(reconstruction, f"variants.{objective_name}.reconstruction_metrics"):
        raise RuntimeError(f"reconstruction reproduction failed for {objective_name}")

    controls = _require_mapping(
        corrected.get("one_step_attribution_controls"),
        f"variants.{objective_name}.one_step_attribution_controls",
    )
    pipeline = _require_mapping(
        controls.get("pipeline_controls"),
        f"variants.{objective_name}.one_step_attribution_controls.pipeline_controls",
    )
    if pipeline.get("oracle_legacy_gate_pass") is not True:
        raise RuntimeError(
            f"persisted oracle legacy gate did not pass for {objective_name}"
        )
    if pipeline.get("pass") is not True:
        raise RuntimeError(f"persisted pipeline controls did not pass for {objective_name}")

    corrected_checks = copy.deepcopy(dict(checks))
    corrected_checks["exact_v_oracle_pass"] = {
        "observed": True,
        "expected": True,
        "pass": True,
        "source": (
            "one_step_attribution_controls.pipeline_controls."
            "oracle_legacy_gate_pass"
        ),
        "original_adapter_check": dict(oracle_check),
    }
    corrected_reproduction = copy.deepcopy(dict(reproduction))
    corrected_reproduction.update(
        {
            "schema": CORRECTION_SCHEMA,
            "original_schema": HISTORICAL_REPRODUCTION_SCHEMA,
            "checks": corrected_checks,
            "pass": True,
            "resume1_correction": {
                "producer_key": "legacy_oracle_audit",
                "persisted_proof_path": (
                    "one_step_attribution_controls.pipeline_controls."
                    "oracle_legacy_gate_pass"
                ),
                "faulty_consumer_key": "exact_v_oracle_audit",
                "training_rerun": False,
                "pilot_mutated_on_disk": False,
            },
        }
    )
    corrected["r252_one_step_reproduction"] = corrected_reproduction
    return corrected


def corrected_pilot_view(pilot: Mapping[str, Any]) -> Dict[str, Any]:
    corrected = copy.deepcopy(dict(pilot))
    variants = _require_mapping(corrected.get("variants"), "pilot.variants")
    corrected_variants: Dict[str, Any] = {}
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        corrected_variants[name] = repair_r252_reproduction(
            _require_mapping(variants.get(name), f"pilot.variants.{name}"),
            objective_name=name,
        )
    corrected["variants"] = corrected_variants

    classification = classify_gate_separation(
        {
            "variants": corrected_variants,
            "synthetic_controls": corrected.get("synthetic_controls", {}),
        }
    )
    if classification.get("root_cause") == (
        "phase314b_r253_r252_one_step_reproduction_failed"
    ):
        raise RuntimeError("corrected classifier still reports reproduction failure")
    for key in (
        "root_cause",
        "supported_mechanisms",
        "next_stage",
        "train_only_recommendation",
    ):
        corrected[key] = classification.get(key)
    corrected["mechanisms_by_variant"] = {
        name: variant_supported_mechanisms(value)
        for name, value in corrected_variants.items()
    }
    if corrected.get("train_only_recommendation") is not None:
        raise RuntimeError("r2.5.3 Resume1 cannot recommend a configuration")
    corrected["resume1_finalizer_correction"] = {
        "schema": CORRECTION_SCHEMA,
        "finalizer_only": True,
        "gpu_pilot_rerun": False,
        "original_pilot_sha256": PILOT_SUMMARY_SHA256,
    }
    return corrected


def resume_source_sha256(root: Path) -> Dict[str, str]:
    paths = (
        "ccda_phase3/phase314b_r253_resume1_finalizer.py",
        "scripts/phase3_14b_r253_resume1_preflight.py",
        "scripts/phase3_14b_r253_resume1_finalize.py",
        "scripts/phase3_14b_r253_resume1_run.sh",
        "tests/test_phase314b_r253_resume1_finalizer.py",
    )
    output: Dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Resume1 source file missing: {relative}")
        output[relative] = sha256_file(path)
    return output


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite JSON: {path}")
    assert_finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
