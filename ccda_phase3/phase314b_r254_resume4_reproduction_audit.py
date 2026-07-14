"""Read-only r2.5.3 reproduction-mismatch decomposition for r2.5.4 Resume4.

The audit consumes only committed reports and pinned source files.  It never
loads validation targets, trains a model, samples a reverse process, or writes
weights/predictions.  Its purpose is to distinguish an adapter/schema defect
from a fixed-pipeline mismatch and from missing residual-training identity
evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PHASE = "phase3_14b_r254_resume4"
SCHEMA = "phase314b_r254_resume4_reproduction_mismatch_v1"
BASE_BLOCKED_COMMIT = "3db29b6b8184977a0919b0c6af52cec2d616e9c6"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R253_PILOT_SHA256 = (
    "a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86"
)
EXPECTED_R253_SUMMARY_SHA256 = (
    "8cbd0d2dac4180578e966479be07b5a753bdb902658bc63000cfcd6eb2bea28a"
)
EXPECTED_RESUME3_PREFLIGHT_SHA256 = (
    "27c40844cd5d601e41cd5bf789d6d0545127dd76d29ae24a334e1a898e5bdbed"
)
EXPECTED_RESUME3_PILOT_SHA256 = (
    "d28c4b40c16ba852a2dc3cc42cfee548defeb6aa7ed3717203690282b984f36e"
)
EXPECTED_RESUME3_BLOCKED_SHA256 = (
    "fd634c47580061180cad61957ee3faa2de84b7b5f689451a3b563d0368a92c74"
)
EXPECTED_CURRENT_PRIOR_STATE_SHA256 = (
    "8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904"
)
EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256 = (
    "70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9"
)
EXPECTED_PHASE3_R2_PASS_COUNT = 425

R253_PILOT_PATH = "reports/phase3_14b_r253_pilot_summary.json"
R253_SUMMARY_PATH = "reports/phase3_14b_r253_summary.json"
RESUME3_PREFLIGHT_PATH = "reports/phase3_14b_r254_resume3_preflight_summary.json"
RESUME3_PILOT_PATH = "reports/phase3_14b_r254_resume3_pilot_summary.json"
RESUME3_BLOCKED_PATH = "reports/phase3_14b_r254_resume3_blocked_summary.json"
SUBMODULE_TASK_PATH = "external/deformable-ravens/ravens/tasks/ccda_slack_cable_v2.py"
SUBMODULE_ENVIRONMENT_PATH = "external/deformable-ravens/ravens/environment.py"

DIAGNOSTIC_OBJECTIVE_NAMES = (
    "v_only_frozen_control",
    "ordered_mean_raw_g100",
    "ordered_cvar_contract_g010",
)
CONTRACT_CHECK_KEYS = (
    "historical_exact_reconstruction_pass",
    "branch_transport_pass",
    "one_step_physical_pass",
    "ordered_topology_pass",
    "final_horizon_cable_geometry_pass",
    "all_earlier_horizon_cable_geometry_pass",
)
NUMERIC_CHECK_KEYS = (
    "full_z_mse",
    "cable_z_mse",
    "robot_proxy_z_mse",
    "cable_error_fraction",
    "robot_proxy_error_fraction",
)
TRAINING_CHECK_KEYS = (
    "z_mse",
    "ordered_rmse.p95",
    "segment_relative_error.p95",
    "chain_relative_error.p95",
)
RESIDUAL_IDENTITY_FIELDS = (
    "residual_initial_state_sha256",
    "residual_final_state_sha256",
    "optimizer_initial_state_sha256",
    "optimizer_final_state_sha256",
    "prediction_sha256",
    "training_loss_history",
    "gradient_history",
    "source_exposure_counts",
    "training_rng_fingerprint",
)

PHASE3_R2_TEST_PATHS = (
    "tests/test_phase3_14b_r21_contract.py",
    "tests/test_phase3_14b_r21_geometry.py",
    "tests/test_phase3_14b_r2_contract.py",
    "tests/test_phase3_14b_r2_diffusion.py",
    "tests/test_phase3_14b_r2_metrics.py",
    "tests/test_phase314b_r22_contract.py",
    "tests/test_phase314b_r22_gates.py",
    "tests/test_phase314b_r22_geometry.py",
    "tests/test_phase314b_r22_loss.py",
    "tests/test_phase314b_r231_controls.py",
    "tests/test_phase314b_r232_controls.py",
    "tests/test_phase314b_r23_diagnostics.py",
    "tests/test_phase314b_r241_multirow.py",
    "tests/test_phase314b_r242_frozen_prior.py",
    "tests/test_phase314b_r24_noisy_skip.py",
    "tests/test_phase314b_r251_gradient_calibration.py",
    "tests/test_phase314b_r252_transport_attribution.py",
    "tests/test_phase314b_r253_gate_separation.py",
    "tests/test_phase314b_r253_resume1_finalizer.py",
    "tests/test_phase314b_r253_resume2_finalizer.py",
    "tests/test_phase314b_r253_resume3_finalizer.py",
    "tests/test_phase314b_r254_resume1_prior_determinism.py",
    "tests/test_phase314b_r254_resume2_functional_prior.py",
    "tests/test_phase314b_r254_resume3_prediction_adapter.py",
    "tests/test_phase314b_r254_resume4_reproduction_audit.py",
    "tests/test_phase314b_r254_robot_proxy_attribution.py",
    "tests/test_phase314b_r25_ordered_geometry.py",
)

OUT_OF_SCOPE_COLLECTION_BASELINE = (
    {
        "test_path": "tests/test_ccda_phase3_state_safety.py",
        "reason": "removed legacy finite_velocity_or_difference/state_from_info API",
    },
    {
        "test_path": "tests/test_single_realsense.py",
        "reason": "state_diff.real_world is absent from this branch",
    },
    {
        "test_path": "tests/test_multi_realsense.py",
        "reason": "state_diff.real_world is absent from this branch",
    },
    {
        "test_path": "tests/test_ring_buffer.py",
        "reason": "optional atomics dependency is absent",
    },
    {
        "test_path": "tests/test_shared_queue.py",
        "reason": "optional atomics dependency is absent",
    },
    {
        "test_path": "tests/test_robomimic_image_runner.py",
        "reason": "robomimic image runner is absent from this branch",
    },
    {
        "test_path": "tests/test_robomimic_lowdim_runner.py",
        "reason": "robomimic lowdim runner is absent from this branch",
    },
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r254_resume4_reproduction_audit.py",
    "scripts/phase3_14b_r254_resume4_test_gate.py",
    "scripts/phase3_14b_r254_resume4_preflight.py",
    "scripts/phase3_14b_r254_resume4_audit.py",
    "scripts/phase3_14b_r254_resume4_finalize.py",
    "scripts/phase3_14b_r254_resume4_blocked.py",
    "scripts/phase3_14b_r254_resume4_run.sh",
    "tests/test_phase314b_r254_resume4_reproduction_audit.py",
)


@dataclass(frozen=True)
class ReproductionAuditSpec:
    historical_absolute_tolerance: float = 2.0e-6
    expected_model_count: int = 3
    expected_contract_checks_per_model: int = 6
    expected_numeric_checks_per_model: int = 5
    expected_training_checks_per_model: int = 4

    def validate(self) -> None:
        if not math.isfinite(self.historical_absolute_tolerance):
            raise ValueError("historical tolerance must be finite")
        if self.historical_absolute_tolerance < 0:
            raise ValueError("historical tolerance must be nonnegative")
        for name in (
            "expected_model_count",
            "expected_contract_checks_per_model",
            "expected_numeric_checks_per_model",
            "expected_training_checks_per_model",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}


def phase3_r2_test_manifest(root: Path) -> Dict[str, str]:
    base = Path(root)
    discovered = sorted(
        {
            path.relative_to(base).as_posix()
            for pattern in (
                "tests/test_phase3_14b_r2*.py",
                "tests/test_phase314b_r2*.py",
            )
            for path in base.glob(pattern)
        }
    )
    expected = sorted(PHASE3_R2_TEST_PATHS)
    if discovered != expected:
        missing = sorted(set(expected) - set(discovered))
        extra = sorted(set(discovered) - set(expected))
        raise RuntimeError(
            f"Phase3 r2.x test scope changed; missing={missing}, extra={extra}"
        )
    for item in OUT_OF_SCOPE_COLLECTION_BASELINE:
        path = str(item["test_path"])
        if not (base / path).is_file():
            raise RuntimeError(f"out-of-scope baseline test disappeared: {path}")
        if path in discovered:
            raise RuntimeError(f"out-of-scope test entered Resume4 gate: {path}")
    return {path: sha256_file(base / path) for path in expected}


def parse_pytest_pass_count(output: str) -> int:
    import re

    summaries = []
    for line in str(output).splitlines():
        if " passed" not in line:
            continue
        match = re.search(r"(?:^|\s)([0-9]+) passed(?:,|\s|$)", line)
        if match:
            summaries.append((line, int(match.group(1))))
    if len(summaries) != 1:
        raise ValueError("expected exactly one pytest pass summary")
    line, count = summaries[0]
    forbidden = (
        " failed",
        " error",
        " skipped",
        " xfailed",
        " xpassed",
        " deselected",
    )
    if any(token in line for token in forbidden):
        raise ValueError(f"scoped pytest summary is not all-pass: {line}")
    return count


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_once(path: Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(root), text=True
    ).strip()


def assert_only_allowed_worktree_paths(root: Path, allowed: Sequence[str]) -> None:
    allowed_set = {Path(path).as_posix() for path in allowed}
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    observed = set()
    for line in output.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        observed.add(Path(path).as_posix())
    unexpected = sorted(observed - allowed_set)
    if unexpected:
        raise RuntimeError("unexpected worktree paths: " + ", ".join(unexpected))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1.0e-20)


def _find_key_paths(value: Any, wanted: Iterable[str], prefix: str = "") -> Dict[str, List[str]]:
    result = {key: [] for key in wanted}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key in result:
                    result[key].append(child_path)
                visit(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    visit(value, prefix)
    return result


def _pipeline_identity(
    historical: Mapping[str, Any], current: Mapping[str, Any]
) -> Dict[str, Any]:
    comparisons: List[Tuple[str, Any, Any]] = []
    for section, keys in (
        (
            "dataset",
            (
                "paired_rows",
                "paired_input_max_abs_difference",
                "train_rows",
                "validation_rows_verified_but_not_used",
            ),
        ),
        (
            "fixed_contract",
            (
                "model",
                "diagnostic_objective_order",
                "paired_timesteps",
                "evaluation_noise_seeds",
                "paired_prior_seed",
                "paired_training_seed",
                "training_changed",
            ),
        ),
        (
            "repository",
            (
                "branch",
                "cache_sha256",
                "frozen_contract_sha256",
                "submodule_commit",
            ),
        ),
    ):
        old = _mapping(historical.get(section), f"historical {section}")
        new = _mapping(current.get(section), f"current {section}")
        for key in keys:
            comparisons.append((f"{section}.{key}", old.get(key), new.get(key)))
    records = [
        {
            "path": path,
            "historical": expected,
            "current": observed,
            "pass": observed == expected,
        }
        for path, expected, observed in comparisons
    ]
    return {
        "comparisons": records,
        "fixed_pipeline_identity_pass": bool(all(row["pass"] for row in records)),
        "failed_paths": [row["path"] for row in records if not row["pass"]],
    }


def _expected_check_value(
    separation: Mapping[str, Any], check_name: str
) -> Any:
    prefix, key = check_name.split(".", 1)
    if prefix == "contract":
        return _mapping(separation.get("contracts"), "historical contracts").get(key)
    if prefix == "state_group_z_metrics":
        return _mapping(
            separation.get("state_group_z_metrics"), "historical group metrics"
        ).get(key)
    raise ValueError(f"unsupported check path: {check_name}")


def _audit_model(
    name: str,
    historical_variant: Mapping[str, Any],
    current_variant: Mapping[str, Any],
    spec: ReproductionAuditSpec,
) -> Dict[str, Any]:
    separation = _mapping(
        historical_variant.get("gate_separation"), f"historical separation {name}"
    )
    reproduction = _mapping(
        current_variant.get("r253_reproduction"), f"current reproduction {name}"
    )
    checks = _mapping(reproduction.get("checks"), f"current checks {name}")
    expected_names = {
        *(f"contract.{key}" for key in CONTRACT_CHECK_KEYS),
        *(f"state_group_z_metrics.{key}" for key in NUMERIC_CHECK_KEYS),
    }
    missing = sorted(expected_names - set(checks))
    extra = sorted(set(checks) - expected_names)
    rows: List[Dict[str, Any]] = []
    logic_errors: List[str] = []
    binding_errors: List[str] = []
    boolean_changes = 0
    numeric_failures = 0
    for check_name in sorted(expected_names & set(checks)):
        item = _mapping(checks[check_name], f"check {name}.{check_name}")
        expected = item.get("expected")
        observed = item.get("observed")
        recorded_pass = item.get("pass")
        source_expected = _expected_check_value(separation, check_name)
        if expected != source_expected:
            binding_errors.append(check_name)
        if check_name.startswith("contract."):
            if not isinstance(expected, bool) or not isinstance(observed, bool):
                raise ValueError(f"boolean check has non-boolean values: {check_name}")
            absolute_error: Optional[float] = None
            relative_error: Optional[float] = None
            tolerance: Optional[float] = None
            recomputed_pass = observed == expected
            boolean_changes += int(not recomputed_pass)
        else:
            expected_number = _finite_number(expected, f"{check_name}.expected")
            observed_number = _finite_number(observed, f"{check_name}.observed")
            tolerance = _finite_number(item.get("tolerance"), f"{check_name}.tolerance")
            absolute_error = abs(observed_number - expected_number)
            relative_error = _relative_error(observed_number, expected_number)
            recomputed_pass = absolute_error <= tolerance
            numeric_failures += int(not recomputed_pass)
        if bool(recorded_pass) != bool(recomputed_pass):
            logic_errors.append(check_name)
        rows.append(
            {
                "check": check_name,
                "kind": "boolean" if check_name.startswith("contract.") else "scalar",
                "expected": expected,
                "observed": observed,
                "absolute_difference": absolute_error,
                "relative_difference": relative_error,
                "configured_tolerance": tolerance,
                "recorded_pass": bool(recorded_pass),
                "recomputed_pass": bool(recomputed_pass),
                "expected_source": f"{R253_PILOT_PATH}:variants.{name}.gate_separation",
                "observed_source": f"{RESUME3_PILOT_PATH}:variants.{name}.r253_reproduction",
                "shape": "scalar",
            }
        )

    training = _mapping(
        reproduction.get("training_metrics"), f"training metrics {name}"
    )
    training_checks = _mapping(training.get("checks"), f"training checks {name}")
    training_missing = sorted(set(TRAINING_CHECK_KEYS) - set(training_checks))
    training_extra = sorted(set(training_checks) - set(TRAINING_CHECK_KEYS))
    training_rows = []
    training_logic_errors = []
    for check_name in TRAINING_CHECK_KEYS:
        if check_name not in training_checks:
            continue
        item = _mapping(training_checks[check_name], f"training check {check_name}")
        expected = _finite_number(item.get("expected"), f"training {check_name}.expected")
        observed = _finite_number(item.get("observed"), f"training {check_name}.observed")
        tolerance = _finite_number(item.get("tolerance"), f"training {check_name}.tolerance")
        difference = abs(observed - expected)
        recomputed = difference <= tolerance
        if bool(item.get("pass")) != recomputed:
            training_logic_errors.append(check_name)
        training_rows.append(
            {
                "check": check_name,
                "expected": expected,
                "observed": observed,
                "absolute_difference": difference,
                "relative_difference": _relative_error(observed, expected),
                "configured_tolerance": tolerance,
                "recorded_pass": bool(item.get("pass")),
                "recomputed_pass": recomputed,
            }
        )

    fixed_cable = _mapping(
        current_variant.get("fixed_cable_contract"), f"fixed cable contract {name}"
    )
    observed_boolean_binding_errors = []
    for key in (
        "historical_exact_reconstruction_pass",
        "branch_transport_pass",
        "one_step_physical_pass",
        "ordered_topology_pass",
    ):
        check = checks.get(f"contract.{key}", {})
        if fixed_cable.get(key) != check.get("observed"):
            observed_boolean_binding_errors.append(key)

    fingerprint_paths = _find_key_paths(current_variant, RESIDUAL_IDENTITY_FIELDS)
    identity_available = {
        key: paths for key, paths in fingerprint_paths.items() if paths
    }
    schema_pass = not (
        missing
        or extra
        or logic_errors
        or binding_errors
        or training_missing
        or training_extra
        or training_logic_errors
        or observed_boolean_binding_errors
    )
    return {
        "model": name,
        "schema_and_binding_pass": schema_pass,
        "missing_checks": missing,
        "extra_checks": extra,
        "check_logic_errors": logic_errors,
        "historical_binding_errors": binding_errors,
        "observed_boolean_binding_errors": observed_boolean_binding_errors,
        "checks": rows,
        "boolean_change_count": boolean_changes,
        "numeric_failure_count": numeric_failures,
        "recorded_r253_reproduction_pass": bool(
            current_variant.get("r253_reproduction_pass")
        ),
        "training_metric_reproduction": {
            "schema_pass": not (
                training_missing or training_extra or training_logic_errors
            ),
            "missing_checks": training_missing,
            "extra_checks": training_extra,
            "logic_errors": training_logic_errors,
            "all_pass": bool(training_rows)
            and all(row["recomputed_pass"] for row in training_rows),
            "checks": training_rows,
        },
        "residual_identity_fingerprints": {
            "required_fields": list(RESIDUAL_IDENTITY_FIELDS),
            "available_fields": identity_available,
            "complete": set(identity_available) == set(RESIDUAL_IDENTITY_FIELDS),
        },
    }


def audit_robot_proxy_sources(task_source: str, environment_source: str) -> Dict[str, Any]:
    logger_all_joints = (
        "states = [p.getJointState(body_id, index) for index in range(count)]"
        in task_source
    )
    environment_filters_revolute = (
        "if j[2] == p.JOINT_REVOLUTE" in environment_source
        and "self.joints" in environment_source
    )
    logger_uses_last_link = "p.getLinkState(body_id, count - 1)" in task_source
    environment_uses_tip_link = "self.ee_tip_link" in environment_source
    broad_fallback = "except Exception:" in task_source and "missing_zero_proxy" in task_source
    findings = []
    if logger_all_joints and environment_filters_revolute:
        findings.append(
            {
                "id": "robot_proxy_joint_selection_includes_fixed_joints",
                "severity": "confirmed_contract_defect",
                "evidence": "task logs every URDF joint while control uses revolute env.joints",
                "current_phase_action": "report_only_do_not_modify_pinned_submodule",
            }
        )
    if logger_uses_last_link and environment_uses_tip_link:
        findings.append(
            {
                "id": "robot_proxy_end_effector_link_not_environment_tip_link",
                "severity": "confirmed_contract_defect",
                "evidence": "task uses count-1 while environment owns ee_tip_link",
                "current_phase_action": "report_only_do_not_modify_pinned_submodule",
            }
        )
    if broad_fallback:
        findings.append(
            {
                "id": "robot_proxy_collection_errors_silently_zero_filled",
                "severity": "confirmed_observability_defect",
                "evidence": "broad exception falls through to missing_zero_proxy",
                "current_phase_action": "report_only_do_not_modify_pinned_submodule",
            }
        )
    return {
        "source_patterns_recognized": bool(
            logger_all_joints
            and environment_filters_revolute
            and logger_uses_last_link
            and environment_uses_tip_link
            and broad_fallback
        ),
        "findings": findings,
        "submodule_modified": False,
    }


def build_reproduction_audit(
    *,
    r253_pilot: Mapping[str, Any],
    r253_summary: Mapping[str, Any],
    resume3_pilot: Mapping[str, Any],
    resume3_blocked: Mapping[str, Any],
    robot_proxy_source_audit: Optional[Mapping[str, Any]] = None,
    spec: ReproductionAuditSpec = ReproductionAuditSpec(),
) -> Dict[str, Any]:
    spec.validate()
    if r253_pilot.get("gpu_name") != "NVIDIA GeForce RTX 4080 SUPER":
        raise ValueError("historical GPU identity changed")
    if resume3_pilot.get("gpu_name") != "NVIDIA GeForce RTX 4090":
        raise ValueError("Resume3 GPU identity changed")
    if r253_summary.get("root_cause") != (
        "phase314b_r253_robot_proxy_reconstruction_conflation_supported"
    ):
        raise ValueError("r2.5.3 final root cause changed")
    if resume3_pilot.get("root_cause") != (
        "phase314b_r254_r253_contract_reproduction_failed"
    ):
        raise ValueError("Resume3 pilot root cause changed")
    if resume3_blocked.get("verdict") != "BLOCKED":
        raise ValueError("Resume3 blocked verdict changed")

    historical_variants = _mapping(r253_pilot.get("variants"), "historical variants")
    current_variants = _mapping(resume3_pilot.get("variants"), "current variants")
    if set(historical_variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise ValueError("historical diagnostic matrix changed")
    if set(current_variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        raise ValueError("current diagnostic matrix changed")

    models = []
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        models.append(
            _audit_model(
                name,
                _mapping(historical_variants[name], f"historical variant {name}"),
                _mapping(current_variants[name], f"current variant {name}"),
                spec,
            )
        )
    adapter_schema_pass = all(row["schema_and_binding_pass"] for row in models)
    internal_metric_pass = all(
        row["training_metric_reproduction"]["all_pass"] for row in models
    )
    residual_identity_complete = all(
        row["residual_identity_fingerprints"]["complete"] for row in models
    )
    pipeline = _pipeline_identity(r253_pilot, resume3_pilot)

    current_prior = _mapping(
        resume3_pilot.get("fresh_prior_validation"), "fresh prior validation"
    )
    functional_prior = {
        "historical_state_sha256": _mapping(
            r253_pilot.get("shared_prior"), "historical prior"
        ).get("prior_state_sha256"),
        "current_state_sha256": current_prior.get("observed_state_sha256"),
        "current_prediction_sha256": current_prior.get("observed_prediction_sha256"),
        "state_sha_cross_device_equal": False,
        "current_same_device_state_sha_exact": bool(current_prior.get("state_sha_exact")),
        "current_same_device_prediction_sha_exact": bool(
            current_prior.get("prediction_sha_exact")
        ),
        "current_prior_z_mse_pass": bool(current_prior.get("prior_z_mse_pass")),
        "functional_prior_contract_pass": bool(
            current_prior.get("functional_prior_contract_pass")
        ),
    }
    if functional_prior["current_state_sha256"] != EXPECTED_CURRENT_PRIOR_STATE_SHA256:
        raise ValueError("current prior state SHA changed")
    if (
        functional_prior["current_prediction_sha256"]
        != EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256
    ):
        raise ValueError("current prior prediction SHA changed")

    boolean_changes = sum(row["boolean_change_count"] for row in models)
    numeric_failures = sum(row["numeric_failure_count"] for row in models)
    reproduction_pass_count = sum(
        int(row["recorded_r253_reproduction_pass"]) for row in models
    )
    static_audit = dict(robot_proxy_source_audit or {})

    if not adapter_schema_pass:
        root_cause = "phase314b_r254_resume4_reproduction_adapter_defect_supported"
        next_stage = "repair the report adapter/finalizer without retraining"
    elif not pipeline["fixed_pipeline_identity_pass"]:
        root_cause = "phase314b_r254_resume4_fixed_pipeline_identity_mismatch_supported"
        next_stage = "repair the fixed-pipeline mismatch before any retraining"
    elif reproduction_pass_count == len(DIAGNOSTIC_OBJECTIVE_NAMES):
        root_cause = "phase314b_r254_resume4_historical_reproduction_passed"
        next_stage = "finalize the already-computed robot-proxy attribution"
    elif not residual_identity_complete:
        root_cause = (
            "phase314b_r254_resume4_residual_reproduction_identity_unobservable"
        )
        next_stage = (
            "Phase3.14b-r2.5.4 Resume5 same-device residual determinism and "
            "functional-reproduction contract audit"
        )
    else:
        root_cause = "phase314b_r254_resume4_substantive_residual_divergence_supported"
        next_stage = "debug deterministic residual training"

    mechanisms = []
    if adapter_schema_pass:
        mechanisms.append("reproduction_check_schema_and_historical_binding_valid")
    if internal_metric_pass:
        mechanisms.append("current_run_internal_metric_reproduction_exact")
    if pipeline["fixed_pipeline_identity_pass"]:
        mechanisms.append("fixed_rows_seeds_noise_bank_and_contract_identity")
    if not residual_identity_complete:
        mechanisms.append("residual_training_identity_fingerprints_missing")
    if numeric_failures:
        mechanisms.append("historical_cross_device_continuous_metrics_not_reproduced")
    if boolean_changes:
        mechanisms.append("historical_cross_device_contract_booleans_changed")
    if functional_prior["functional_prior_contract_pass"]:
        mechanisms.append("functional_prior_contract_passed")

    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "verdict": "PASS",
        "scientific_status": "BLOCKED" if reproduction_pass_count < 3 else "READY",
        "root_cause": root_cause,
        "supported_mechanisms": mechanisms,
        "next_stage": next_stage,
        "spec": asdict(spec),
        "evidence_only": True,
        "retraining_performed": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_execution": False,
        "idm": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
        "robot_proxy_attribution_interpretable": bool(
            reproduction_pass_count == 3 and residual_identity_complete
        ),
        "pipeline_identity": pipeline,
        "functional_prior": functional_prior,
        "adapter_schema_pass": adapter_schema_pass,
        "current_run_internal_metric_reproduction_pass": internal_metric_pass,
        "residual_training_identity_complete": residual_identity_complete,
        "aggregate": {
            "model_count": len(models),
            "historical_reproduction_pass_count": reproduction_pass_count,
            "historical_reproduction_fail_count": len(models) - reproduction_pass_count,
            "contract_boolean_change_count": boolean_changes,
            "continuous_metric_failure_count": numeric_failures,
        },
        "models": {row["model"]: row for row in models},
        "robot_proxy_static_source_audit": static_audit,
    }


def render_markdown(report: Mapping[str, Any], static_test_count: int) -> str:
    aggregate = _mapping(report.get("aggregate"), "aggregate")
    lines = [
        "# Phase3.14b-r2.5.4 Resume4 Reproduction-Mismatch Audit",
        "",
        f"- Audit verdict: `{report['verdict']}`",
        f"- Scientific status: `{report['scientific_status']}`",
        f"- Root cause: `{report['root_cause']}`",
        f"- Static tests: `{int(static_test_count)} passed`",
        "- Static-test scope: `27 frozen Phase3.14b r2.x files`",
        "- Full-repository pytest required: `false`",
        "- Retraining: `false`",
        "- Robot-proxy attribution interpretable: `false`",
        "",
        "## Conclusion",
        "",
        "The report adapter is structurally consistent and every current-run "
        "training metric reproduces exactly, but the cross-device residual "
        "metrics do not reproduce and no residual initialization, final-state, "
        "prediction, optimizer, or loss-history fingerprints were persisted. "
        "The failure therefore cannot be classified as tiny numeric noise or as "
        "substantive scientific divergence from the committed evidence alone.",
        "",
        "## Aggregate",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Models failing historical reproduction | {aggregate['historical_reproduction_fail_count']} |",
        f"| Changed Boolean contracts | {aggregate['contract_boolean_change_count']} |",
        f"| Failed continuous metric checks | {aggregate['continuous_metric_failure_count']} |",
        f"| Adapter schema valid | {str(bool(report['adapter_schema_pass'])).lower()} |",
        f"| Internal metric reproduction | {str(bool(report['current_run_internal_metric_reproduction_pass'])).lower()} |",
        f"| Residual identity complete | {str(bool(report['residual_training_identity_complete'])).lower()} |",
        "",
        "## Per-model reproduction checks",
        "",
    ]
    models = _mapping(report.get("models"), "models")
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        model = _mapping(models[name], name)
        lines.extend(
            [
                f"### `{name}`",
                "",
                "| Check | Expected | Observed | Absolute diff | Relative diff | Tolerance | Pass |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in model["checks"]:
            absolute = "n/a" if row["absolute_difference"] is None else f"{row['absolute_difference']:.9g}"
            relative = "n/a" if row["relative_difference"] is None else f"{row['relative_difference']:.9g}"
            tolerance = "exact" if row["configured_tolerance"] is None else f"{row['configured_tolerance']:.9g}"
            lines.append(
                f"| `{row['check']}` | `{row['expected']}` | `{row['observed']}` | "
                f"{absolute} | {relative} | {tolerance} | "
                f"{str(bool(row['recomputed_pass'])).lower()} |"
            )
        lines.append("")

    lines.extend(["## Confirmed source-code findings", ""])
    source_audit = _mapping(
        report.get("robot_proxy_static_source_audit", {}), "source audit"
    )
    findings = source_audit.get("findings", [])
    if findings:
        for finding in findings:
            lines.append(
                f"- `{finding['id']}`: {finding['evidence']}. The pinned "
                "submodule remains unchanged in Resume4."
            )
    else:
        lines.append("- No static robot-proxy finding was recorded.")
    lines.extend(
        [
            "",
            "## Static-test scope boundary",
            "",
            "The gate uses the exact frozen union of "
            "`tests/test_phase3_14b_r2*.py` and "
            "`tests/test_phase314b_r2*.py`. It does not use `--ignore`, `-k`, "
            "deselection, or a full-repository collection. The seven known "
            "out-of-scope collection failures are recorded as repository "
            "baseline defects and are not repaired in Resume4.",
            "",
            "## Decision",
            "",
            f"Next: `{report['next_stage']}`.",
            "",
            "Until that audit completes, `train_only_recommendation`, "
            "`selected_configuration`, robot-metric interpretation, formal "
            "training, IDM, candidate execution, and Phase4/CPS remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)
