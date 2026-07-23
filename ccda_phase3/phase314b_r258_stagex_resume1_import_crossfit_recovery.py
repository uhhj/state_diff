"""Stage-X Resume1: import-safe and leakage-correct nested-OOF recovery.

The original Stage-X implementation commit was created, but its controller was
executed by file path without the repository root on ``sys.path``.  The top-level
``ccda_phase3`` import therefore failed before controller ``main()``, before the
CUDA environment probe, and before the Stage-X worker.

Static review also found a scientific aggregation defect that must be corrected
before the first science execution: the original code took the modal policy
among the six outer-fold inner selections and then evaluated that globally
selected policy on the same pooled outer-OOF predictions.  Each row is an outer
holdout in one fold but participates in policy selection through the other five
folds, so that pooled modal-policy metric is not a strictly unbiased nested-OOF
performance estimate.

Resume1 is add-only.  It keeps the original direction model, projected-oracle
supervision, candidate reconstruction, tolerance-aligned gate, policy bank,
risk descriptors, thresholds and all holdout/probe boundaries unchanged.  It:

* bootstraps every executable entry point before project imports;
* passes an explicit repository-root ``PYTHONPATH`` to every child process;
* requires every generated candidate bank member to preserve the aligned gate;
* fails closed on a non-converged non-constant risk fit;
* evaluates the tuning procedure by stitching each outer fold with that fold's
  independently selected inner policy;
* uses the pooled fixed-policy OOF surface only to tune the final train-only
  policy after the unbiased procedure estimate has passed;
* never re-accesses the selection holdout or the frozen probe.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r258_stagex_tail_robust_nested_oof as stagex

PHASE = "Phase3.14b-r2.5.8 Stage X Resume1"
SCHEMA = "phase314b_r258_stagex_resume1_import_crossfit_recovery_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEX_IMPLEMENTATION_COMMIT = "b5955b7863d56d7a2c719a603816ef952aee91c3"
BASE_STAGEX_PARENT = "e0d0756f80415ea36335be6019c86422f043ff83"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

ORIGINAL_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage X: calibrate tail-robust nested group OOF"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage X Resume1: recover import and crossfit aggregation"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage X Resume1 tail-robust OOF evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage X Resume1 blocked evidence"
)

ORIGINAL_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagex_tail_robust_nested_oof.py"),
    ("A", "scripts/phase3_14b_r258_stagex_worker.py"),
    ("A", "scripts/phase3_14b_r258_stagex_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagex_tail_robust_nested_oof.py"),
)
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagex_resume1_import_crossfit_recovery.py"),
    ("A", "scripts/phase3_14b_r258_stagex_resume1_worker.py"),
    ("A", "scripts/phase3_14b_r258_stagex_resume1_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagex_resume1_import_crossfit_recovery.py"),
)
ORIGINAL_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stagex_tail_robust_nested_oof.py": (
        "289da5e41b3d8dde85b429ea3147c5f005af97f0c7d156b1b58e9389f02a9025"
    ),
    "scripts/phase3_14b_r258_stagex_worker.py": (
        "a8b6bc4e4f2950b65679cfabcdd15b335a7e6cba8120bad12b9fc76f67f0da7b"
    ),
    "scripts/phase3_14b_r258_stagex_execute.py": (
        "d21da7e6e0bfaff1a81eda6ede92c4eeaa71c422d387149e6353b4338598adf6"
    ),
    "tests/test_phase3_14b_r258_stagex_tail_robust_nested_oof.py": (
        "4d552da67fd44626f69a7754ddf16b936d14205f1833dba733dd523b9b701216"
    ),
}

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stagex_resume1_tail_robust_nested_group_oof_summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stagex_resume1_tail_robust_nested_group_oof_blocked_summary.json"
)

LOCKED_TIMESTEPS = stagex.LOCKED_TIMESTEPS
DIRECTION_SHRINKAGES = stagex.DIRECTION_SHRINKAGES
EXPECTED_OUTER_FOLDS = stagex.EXPECTED_OUTER_FOLDS
EXPECTED_ROWS = stagex.EXPECTED_ROWS
EXPECTED_INTERNAL_SCALE_COUNT = stagex.EXPECTED_INTERNAL_SCALE_COUNT
FALSE_BOUNDARIES = stagex.FALSE_BOUNDARIES
StageXError = stagex.StageXError
StageXSpec = stagex.StageXSpec
TailPolicy = stagex.TailPolicy
stable_json_bytes = stagex.stable_json_bytes
sha256_bytes = stagex.sha256_bytes
sha256_file = stagex.sha256_file
sha256_array = stagex.sha256_array


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageXError(
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
        raise StageXError(
            "git {} failed: {}".format(
                " ".join(args), completed.stderr.decode("utf-8", "replace").strip()
            )
        )
    return bytes(completed.stdout)


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    values = []
    for line in output.splitlines():
        if line.strip():
            fields = line.split("\t")
            values.append((fields[0], fields[-1]))
    return tuple(sorted(values))


def child_environment(root: Path) -> Mapping[str, str]:
    """Return a child environment with the repository root first on PYTHONPATH."""
    env = dict(os.environ)
    root_text = str(Path(root).resolve())
    prior = env.get("PYTHONPATH", "")
    entries = [item for item in prior.split(os.pathsep) if item and item != root_text]
    env["PYTHONPATH"] = os.pathsep.join([root_text] + entries)
    return env


def entrypoint_import_defect(source: str) -> Mapping[str, Any]:
    """Prove the original controller imports ccda_phase3 before path bootstrap."""
    tree = ast.parse(source)
    project_import_lines: List[int] = []
    sys_path_mutation_lines: List[int] = []
    main_call_lines: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "ccda_phase3" or str(node.module).startswith("ccda_phase3.")
        ):
            project_import_lines.append(int(node.lineno))
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"insert", "append"}:
                text = ast.unparse(func.value) if hasattr(ast, "unparse") else ""
                if "sys.path" in text:
                    sys_path_mutation_lines.append(int(node.lineno))
        if isinstance(node, ast.If):
            segment = ast.get_source_segment(source, node.test) or ""
            if "__name__" in segment and "__main__" in segment:
                main_call_lines.append(int(node.lineno))
    if not project_import_lines:
        raise StageXError("original Stage-X controller lacks expected project import")
    first_import = min(project_import_lines)
    path_before_import = any(line < first_import for line in sys_path_mutation_lines)
    if path_before_import:
        raise StageXError("original Stage-X controller unexpectedly bootstraps sys.path")
    if not main_call_lines or min(main_call_lines) <= first_import:
        raise StageXError("original Stage-X controller control flow changed")
    return {
        "project_import_line": first_import,
        "sys_path_bootstrap_before_import": False,
        "main_guard_line": min(main_call_lines),
        "failure_precedes_controller_main": True,
        "failure_precedes_environment_probe_call": True,
        "failure_precedes_science_worker_call": True,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageXError("Stage-X Resume1 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", head + "^") != BASE_STAGEX_IMPLEMENTATION_COMMIT:
        raise StageXError("Resume1 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageXError("Resume1 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageXError("Resume1 implementation paths changed")
    if _git(repo, "rev-parse", BASE_STAGEX_IMPLEMENTATION_COMMIT + "^") != BASE_STAGEX_PARENT:
        raise StageXError("original Stage-X implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", BASE_STAGEX_IMPLEMENTATION_COMMIT) != ORIGINAL_IMPLEMENTATION_SUBJECT:
        raise StageXError("original Stage-X implementation subject changed")
    if commit_name_status(repo, BASE_STAGEX_IMPLEMENTATION_COMMIT) != tuple(sorted(ORIGINAL_IMPLEMENTATION_PATHS)):
        raise StageXError("original Stage-X implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageXError("origin/Experiment1 changed")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageXError("main worktree is dirty")
    submodule = repo / "external/deformable-ravens"
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageXError("DeformableRavens gitlink changed")
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageXError("DeformableRavens worktree commit changed")
    if _git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageXError("DeformableRavens worktree is dirty")
    for relative, expected in ORIGINAL_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageXError("original Stage-X source changed: {}".format(relative))
        if _git_bytes(repo, "show", "{}:{}".format(BASE_STAGEX_IMPLEMENTATION_COMMIT, relative)) != path.read_bytes():
            raise StageXError("original Stage-X worktree differs from implementation blob: {}".format(relative))
    for relative in (
        stagex.SUCCESS_REPORT,
        stagex.BLOCKED_REPORT,
        SUCCESS_REPORT,
        BLOCKED_REPORT,
    ):
        if (repo / relative).exists():
            raise StageXError("Stage-X output exists; recovery rerun is forbidden: {}".format(relative))
    prior = entrypoint_import_defect(
        (repo / "scripts/phase3_14b_r258_stagex_execute.py").read_text(encoding="utf-8")
    )
    for relative, expected in {
        stagex.STAGEW_REPORT: stagex.EXPECTED_STAGEW_REPORT_SHA256,
        stagex.STAGEU_CONTRACT: stagex.EXPECTED_STAGEU_CONTRACT_SHA256,
        stagex.STAGEV_REPORT: stagex.EXPECTED_STAGEV_RESUME1_SHA256,
    }.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageXError("immutable evidence changed: {}".format(relative))
    stagex.validate_stagew_report(stagex._load_json(repo / stagex.STAGEW_REPORT))
    stagev = stagex._load_json(repo / stagex.STAGEV_REPORT)
    if stagev.get("scientific_result_sha256") != stagex.EXPECTED_STAGEV_RESULT_SHA256:
        raise StageXError("Stage-V scientific result changed")
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_STAGEX_IMPLEMENTATION_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "original_stagex_implementation_commit": BASE_STAGEX_IMPLEMENTATION_COMMIT,
        "original_stagex_source_sha256": dict(ORIGINAL_SOURCE_SHA256),
        "prior_attempt_provenance": {
            **prior,
            "operator_observed_error": "ModuleNotFoundError: No module named 'ccda_phase3'",
            "operator_reported_environment_probe_count": 0,
            "operator_reported_science_worker_count": 0,
            "operator_reported_selection_holdout_access_added": 0,
            "claim_source": "operator trace plus static controller control-flow proof",
        },
    }


def require_clean_generation(result: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "length_log_z_element_mismatch_count",
        "aligned_upper_element_failure_count",
        "strict_pass_aligned_fail_row_count",
    ):
        if int(result.get(key, -1)) != 0:
            raise StageXError("{} violates aligned gate: {}".format(label, key))
    if result.get("candidate_finalized_without_holdout_target") is not True:
        raise StageXError("{} lacks target-independent finalization proof".format(label))
    if int(result.get("internal_scale_attempt_count", -1)) != EXPECTED_INTERNAL_SCALE_COUNT:
        raise StageXError("{} internal-scale population changed".format(label))


def fit_risk_model_checked(
    features: np.ndarray,
    adverse: np.ndarray,
    fit_mask: np.ndarray,
    spec: StageXSpec,
) -> Mapping[str, Any]:
    model = stagex.fit_risk_model(features, adverse, fit_mask, spec)
    if model.get("mode") != "constant" and model.get("converged") is not True:
        raise StageXError("non-constant adverse-risk IRLS did not converge")
    return model


def evaluate_variable_threshold_policy(
    control: np.ndarray,
    raw_candidate: np.ndarray,
    raw_scale: np.ndarray,
    adverse_probability: np.ndarray,
    thresholds: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    baseline_probability: np.ndarray,
    spec: StageXSpec,
) -> Mapping[str, Any]:
    control_value = np.asarray(control, dtype=np.float32)
    candidate_value = np.asarray(raw_candidate, dtype=np.float32)
    scale = np.asarray(raw_scale, dtype=np.float64)
    risk = np.asarray(adverse_probability, dtype=np.float64)
    threshold = np.asarray(thresholds, dtype=np.float64)
    constant = np.asarray(baseline_probability, dtype=np.float64)
    rows = control_value.shape[0]
    for value, label in ((scale, "scale"), (risk, "risk"), (threshold, "threshold"), (constant, "constant")):
        if value.shape != (rows,):
            raise StageXError("variable-threshold {} population changed".format(label))
    keep = (scale > 0.0) & (risk <= threshold)
    output = control_value.copy()
    output[keep] = candidate_value[keep]
    output_scale = np.zeros_like(scale)
    output_scale[keep] = scale[keep]

    control_sse = stagex.row_squared_error(control_value, target)
    candidate_sse = stagex.row_squared_error(output, target)
    control_distance = np.sqrt(control_sse)
    candidate_distance = np.sqrt(candidate_sse)
    reduction = control_distance - candidate_distance
    relative = reduction / np.maximum(control_distance, spec.epsilon)
    accepted = output_scale > 0.0
    overall_ratio = float(np.sum(candidate_sse) / max(float(np.sum(control_sse)), spec.epsilon))
    if np.any(accepted):
        accepted_ratio = float(
            np.sum(candidate_sse[accepted])
            / max(float(np.sum(control_sse[accepted])), spec.epsilon)
        )
        positive_rate = float(np.mean(reduction[accepted] > 0.0))
        relative_mean = float(np.mean(relative[accepted]))
    else:
        accepted_ratio = 1.0
        positive_rate = 0.0
        relative_mean = 0.0
    delta = candidate_sse - control_sse
    adverse_mass = float(
        np.sum(np.maximum(delta, 0.0)) / max(float(np.sum(control_sse)), spec.epsilon)
    )
    beneficial_mass = float(
        np.sum(np.maximum(-delta, 0.0)) / max(float(np.sum(control_sse)), spec.epsilon)
    )
    raw_accepted = scale > 0.0
    raw_adverse = (
        stagex.row_squared_error(candidate_value, target) > control_sse
    ).astype(np.float64)
    if np.any(raw_accepted):
        brier = float(np.mean((risk[raw_accepted] - raw_adverse[raw_accepted]) ** 2))
        constant_brier = float(
            np.mean((constant[raw_accepted] - raw_adverse[raw_accepted]) ** 2)
        )
    else:
        brier = constant_brier = 0.0
    return {
        "row_count": int(rows),
        "selected_row_count": int(np.sum(accepted)),
        "acceptance_rate": float(np.mean(accepted)),
        "overall_mse_ratio": overall_ratio,
        "accepted_row_mse_ratio": accepted_ratio,
        "positive_distance_reduction_rate": positive_rate,
        "relative_distance_reduction_mean": relative_mean,
        "adverse_sse_mass": adverse_mass,
        "beneficial_sse_mass": beneficial_mass,
        "net_sse_reduction_mass": beneficial_mass - adverse_mass,
        "risk_brier_score": brier,
        "risk_constant_brier_score": constant_brier,
        "risk_brier_nonworse": bool(brier <= constant_brier + 1.0e-12),
        "group_tail": stagex.group_tail_metrics(
            control_sse, candidate_sse, groups, spec.group_cvar_fraction
        ),
        "distance_reduction_stats": stagex._stats(reduction[accepted]),
        "adverse_row_count": int(np.sum(raw_accepted & (raw_adverse > 0.5))),
        "raw_accepted_row_count": int(np.sum(raw_accepted)),
        "output_candidate_sha256": sha256_array(output),
        "output_selected_scale_sha256": sha256_array(output_scale),
        "threshold_population_sha256": sha256_array(threshold),
    }


def _empty_stitch(context: Mapping[str, Any], timestep: int) -> MutableMapping[str, np.ndarray]:
    control = np.asarray(
        context["objective_control_predictions"][int(timestep)], dtype=np.float32
    )
    return {
        "control": control,
        "candidate": control.copy(),
        "scale": np.zeros(control.shape[0], dtype=np.float64),
        "risk": np.ones(control.shape[0], dtype=np.float64),
        "constant": np.ones(control.shape[0], dtype=np.float64),
        "threshold": np.zeros(control.shape[0], dtype=np.float64),
    }


def stitch_fold_selected_procedure(
    *,
    context: Mapping[str, Any],
    outer_outputs: Mapping[int, Mapping[int, Mapping[float, Mapping[str, Any]]]],
    outer_selections: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    groups: np.ndarray,
    spec: StageXSpec,
) -> Tuple[Mapping[int, Mapping[str, Any]], Mapping[str, Mapping[str, int]]]:
    if len(outer_selections) != EXPECTED_OUTER_FOLDS:
        raise StageXError("outer selection population changed")
    records: Dict[int, Mapping[str, Any]] = {}
    gate_totals: Dict[str, Mapping[str, int]] = {}
    for timestep in LOCKED_TIMESTEPS:
        stitched = _empty_stitch(context, int(timestep))
        gate = {
            "length_log_z_element_mismatch_count": 0,
            "aligned_upper_element_failure_count": 0,
            "strict_pass_aligned_fail_row_count": 0,
        }
        assigned = np.zeros(stitched["control"].shape[0], dtype=np.bool_)
        for outer_fold in range(EXPECTED_OUTER_FOLDS):
            selected = TailPolicy(**outer_selections[outer_fold]["selected_policy"])
            selected.validate()
            output = outer_outputs[outer_fold][int(timestep)][selected.shrinkage]
            indices = np.asarray(output["indices"], dtype=np.int64)
            if np.any(assigned[indices]):
                raise StageXError("outer procedure rows overlap")
            assigned[indices] = True
            stitched["candidate"][indices] = output["candidate"]
            stitched["scale"][indices] = output["selected_scale"]
            stitched["risk"][indices] = output["risk_probability"]
            stitched["constant"][indices] = output["risk_constant_probability"]
            stitched["threshold"][indices] = selected.risk_threshold
            for key in gate:
                gate[key] += int(output["gate_counts"][key])
        if not np.all(assigned):
            raise StageXError("outer procedure does not cover every objective row")
        if any(gate.values()):
            raise StageXError("fold-selected procedure contains aligned-gate failure")
        records[int(timestep)] = evaluate_variable_threshold_policy(
            stitched["control"],
            stitched["candidate"],
            stitched["scale"],
            stitched["risk"],
            stitched["threshold"],
            target,
            groups,
            stitched["constant"],
            spec,
        )
        gate_totals[str(timestep)] = gate
    return records, gate_totals


def fixed_policy_oof_surface(
    *,
    context: Mapping[str, Any],
    outer_outputs: Mapping[int, Mapping[int, Mapping[float, Mapping[str, Any]]]],
    target: np.ndarray,
    groups: np.ndarray,
    spec: StageXSpec,
) -> Tuple[
    Mapping[str, Mapping[int, Mapping[str, Any]]],
    Mapping[int, Mapping[str, Any]],
]:
    baseline_records: Dict[int, Mapping[str, Any]] = {}
    policy_records: Dict[str, Dict[int, Mapping[str, Any]]] = {
        policy.policy_id: {} for policy in stagex.policy_population()
    }
    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)], dtype=np.float32
        )
        baseline_candidate = control.copy()
        baseline_scale = np.zeros(control.shape[0], dtype=np.float64)
        fixed: Dict[float, MutableMapping[str, np.ndarray]] = {
            shrinkage: _empty_stitch(context, int(timestep))
            for shrinkage in DIRECTION_SHRINKAGES
        }
        assigned = np.zeros(control.shape[0], dtype=np.bool_)
        for outer_fold in range(EXPECTED_OUTER_FOLDS):
            indices = np.asarray(
                outer_outputs[outer_fold][int(timestep)][1.0]["indices"], dtype=np.int64
            )
            if np.any(assigned[indices]):
                raise StageXError("fixed-policy OOF rows overlap")
            assigned[indices] = True
            baseline = outer_outputs[outer_fold][int(timestep)][1.0]
            baseline_candidate[indices] = baseline["candidate"]
            baseline_scale[indices] = baseline["selected_scale"]
            for shrinkage in DIRECTION_SHRINKAGES:
                output = outer_outputs[outer_fold][int(timestep)][shrinkage]
                fixed[shrinkage]["candidate"][indices] = output["candidate"]
                fixed[shrinkage]["scale"][indices] = output["selected_scale"]
                fixed[shrinkage]["risk"][indices] = output["risk_probability"]
                fixed[shrinkage]["constant"][indices] = output[
                    "risk_constant_probability"
                ]
                if any(int(value) != 0 for value in output["gate_counts"].values()):
                    raise StageXError("fixed-policy OOF surface contains gate failure")
        if not np.all(assigned):
            raise StageXError("fixed-policy OOF surface lacks row coverage")
        baseline_records[int(timestep)] = stagex.evaluate_policy(
            control,
            baseline_candidate,
            baseline_scale,
            np.zeros(control.shape[0], dtype=np.float64),
            target,
            groups,
            1.0,
            None,
            spec,
        )
        for policy in stagex.policy_population():
            stitched = fixed[policy.shrinkage]
            policy_records[policy.policy_id][int(timestep)] = stagex.evaluate_policy(
                control,
                stitched["candidate"],
                stitched["scale"],
                stitched["risk"],
                target,
                groups,
                policy.risk_threshold,
                stitched["constant"],
                spec,
            )
    return policy_records, baseline_records


def classify_resume1(
    *,
    procedure_records: Mapping[int, Mapping[str, Any]],
    baseline_records: Mapping[int, Mapping[str, Any]],
    outer_selections: Sequence[Mapping[str, Any]],
    modal: Mapping[str, Any],
    full_selection: Mapping[str, Any],
    spec: StageXSpec,
) -> Mapping[str, Any]:
    procedure_eligibility = stagex.policy_is_eligible(
        procedure_records, baseline_records, spec
    )
    every_inner_eligible = all(
        item.get("selected_policy_inner_eligible") is True for item in outer_selections
    )
    stable = int(modal["support_count"]) >= spec.modal_outer_fold_minimum
    final_fixed_eligible = full_selection.get("selected_policy_inner_eligible") is True
    ready = bool(
        procedure_eligibility["all_timesteps_pass"]
        and every_inner_eligible
        and stable
        and final_fixed_eligible
    )
    if ready:
        root = (
            "phase314b_r258_stagex_resume1_unbiased_outer_procedure_and_"
            "full_objective_oof_fixed_policy_control_adverse_sse"
        )
        next_path = (
            "FREEZE_STAGE_X_RESUME1_FULL_OBJECTIVE_TRAIN_POLICY_AND_"
            "PREREGISTER_ONE_SHOT_FROZEN_PROBE_EVALUATION"
        )
        locus = "tail_robust_nested_selection_procedure_ready"
    elif not procedure_eligibility["all_timesteps_pass"]:
        root = "phase314b_r258_stagex_resume1_outer_crossfit_procedure_fails_tail_contract"
        next_path = "AUDIT_OBJECTIVE_TRAIN_OUTER_CROSSFIT_TAIL_FAILURE_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "outer_crossfit_tail_fidelity_failure"
    elif not every_inner_eligible:
        root = "phase314b_r258_stagex_resume1_inner_policy_not_consistently_eligible"
        next_path = "AUDIT_OBJECTIVE_TRAIN_INNER_POLICY_INSTABILITY_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "inner_selection_instability"
    elif not stable:
        root = "phase314b_r258_stagex_resume1_inner_selected_policy_is_not_fold_stable"
        next_path = "AUDIT_OBJECTIVE_TRAIN_POLICY_STABILITY_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "policy_stability_failure"
    else:
        root = "phase314b_r258_stagex_resume1_full_objective_oof_fixed_policy_not_eligible"
        next_path = "AUDIT_OBJECTIVE_TRAIN_FIXED_POLICY_TUNING_SURFACE_WITHOUT_HOLDOUT_OR_PROBE"
        locus = "full_objective_oof_policy_failure"
    return {
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "outer_crossfit_procedure_eligibility": procedure_eligibility,
        "all_outer_inner_selections_eligible": every_inner_eligible,
        "inner_selection_modal_support_stable": stable,
        "full_objective_oof_fixed_policy_eligible": final_fixed_eligible,
    }


def run_nested_oof(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    stageu_contract: Mapping[str, Any],
    spec: Optional[StageXSpec] = None,
) -> Mapping[str, Any]:
    active = StageXSpec() if spec is None else spec
    active.validate()
    stagex.validate_environment_variables()
    modules = stagex._science_modules()
    environment = modules["stages_resume3"].validate_probe_payload_for_science(
        probe_payload, resume2a=modules["resume2a"]
    )
    runtime_modules = modules["stageo"]._runtime_modules()
    stagea = runtime_modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment)
    cold = stagea.assert_cold_cuda_context_portable()
    runtime = dict(modules["stageo"]._prepare_runtime(Path(root).resolve(), environment))
    runtime.update({**modules, "stageu_gate": stageu_contract["aligned_gate_lock"]})
    context = stagex._objective_only_context(runtime["context"])
    runtime["context"] = context
    population = stagex.validate_objective_population(context)
    reconstruction_spec = modules["stager"].StageRSpec()
    reconstruction_spec.validate()
    assignment = np.asarray(context["objective_fold_assignment"], dtype=np.int64)
    groups = np.asarray(context["objective_groups"]).astype(str)
    target = np.asarray(context["objective_target"], dtype=np.float32)
    stagef = runtime["stagef"]
    definition = stagef.definition_by_id(stagex.LOCKED_BACKBONE)
    features_by_timestep: Dict[int, np.ndarray] = {}
    oracle_by_timestep: Dict[int, Mapping[str, Any]] = {}
    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)], dtype=np.float32
        )
        features_by_timestep[int(timestep)] = stagef.build_constraint_features(
            condition=context["objective_condition"],
            control=control,
            condition_name=context["objective_condition_name"],
            feature_mode=definition.feature_mode,
            context=context,
        )
        oracle_by_timestep[int(timestep)] = stagef.generate_projected_oracle_target(
            control=control,
            target=target,
            groups=groups,
            condition_name=context["objective_condition_name"],
            timestep=int(timestep),
            context=context,
            direction_spec=runtime["direction_spec"],
            integrator_spec=runtime["integrator_spec"],
            spec=runtime["stagef_spec"],
        )

    outer_selections: List[Mapping[str, Any]] = []
    outer_outputs: Dict[int, Dict[int, Dict[float, Mapping[str, Any]]]] = {}
    counts = {
        "direction_fit_count": 0,
        "candidate_generation_count": 0,
        "risk_fit_count": 0,
        "internal_scale_attempt_count": 0,
        "inner_policy_evaluation_count": 0,
        "full_objective_oof_policy_evaluation_count": 0,
        "nonconverged_risk_fit_count": 0,
    }

    for outer_fold in range(EXPECTED_OUTER_FOLDS):
        outer_test = assignment == outer_fold
        outer_train = ~outer_test
        outer_train_indices = np.flatnonzero(outer_train)
        outer_test_indices = np.flatnonzero(outer_test)
        inner_candidates: Dict[int, Dict[float, np.ndarray]] = {}
        inner_scales: Dict[int, Dict[float, np.ndarray]] = {}
        inner_directions: Dict[int, np.ndarray] = {}
        for timestep in LOCKED_TIMESTEPS:
            control_full = np.asarray(
                context["objective_control_predictions"][int(timestep)], dtype=np.float32
            )
            inner_directions[int(timestep)] = np.zeros(control_full.shape, dtype=np.float64)
            inner_candidates[int(timestep)] = {
                shrinkage: control_full.copy() for shrinkage in DIRECTION_SHRINKAGES
            }
            inner_scales[int(timestep)] = {
                shrinkage: np.zeros(EXPECTED_ROWS, dtype=np.float64)
                for shrinkage in DIRECTION_SHRINKAGES
            }
        inner_fold_values = [
            value for value in range(EXPECTED_OUTER_FOLDS) if value != outer_fold
        ]
        if len(inner_fold_values) != stagex.EXPECTED_INNER_FOLDS:
            raise StageXError("inner fold population changed")

        for inner_fold in inner_fold_values:
            validation_mask = outer_train & (assignment == inner_fold)
            fit_mask = outer_train & ~validation_mask
            validation_indices = np.flatnonzero(validation_mask)
            for timestep in LOCKED_TIMESTEPS:
                fitted = stagex._fit_direction(
                    runtime=runtime,
                    context=context,
                    timestep=timestep,
                    fit_mask=fit_mask,
                    predict_indices=validation_indices,
                    oracle=oracle_by_timestep[timestep],
                    features=features_by_timestep[timestep],
                )
                counts["direction_fit_count"] += 1
                inner_directions[timestep][validation_indices] = fitted["direction"]
                for shrinkage in DIRECTION_SHRINKAGES:
                    generated = stagex._candidate_for_indices(
                        runtime=runtime,
                        context=context,
                        timestep=timestep,
                        indices=validation_indices,
                        direction=fitted["direction"],
                        shrinkage=shrinkage,
                        reconstruction_spec=reconstruction_spec,
                    )
                    require_clean_generation(
                        generated,
                        label="outer{} inner{} t{} shrink{}".format(
                            outer_fold, inner_fold, timestep, shrinkage
                        ),
                    )
                    counts["candidate_generation_count"] += 1
                    counts["internal_scale_attempt_count"] += EXPECTED_INTERNAL_SCALE_COUNT
                    inner_candidates[timestep][shrinkage][validation_indices] = generated[
                        "candidate"
                    ]
                    inner_scales[timestep][shrinkage][validation_indices] = generated[
                        "selected_scale"
                    ]

        risk_oof: Dict[int, Dict[float, np.ndarray]] = {}
        risk_constant_oof: Dict[int, Dict[float, np.ndarray]] = {}
        for timestep in LOCKED_TIMESTEPS:
            control_all = np.asarray(
                context["objective_control_predictions"][int(timestep)], dtype=np.float32
            )
            risk_oof[timestep] = {}
            risk_constant_oof[timestep] = {}
            for shrinkage in DIRECTION_SHRINKAGES:
                descriptors = stagex.compact_risk_descriptors(
                    control_all,
                    inner_directions[timestep] * shrinkage,
                    inner_candidates[timestep][shrinkage],
                    inner_scales[timestep][shrinkage],
                )
                labels = np.zeros(EXPECTED_ROWS, dtype=np.float64)
                labels[outer_train_indices] = stagex._risk_labels(
                    control_all[outer_train_indices],
                    inner_candidates[timestep][shrinkage][outer_train_indices],
                    target[outer_train_indices],
                )
                probabilities = np.ones(EXPECTED_ROWS, dtype=np.float64)
                constants = np.ones(EXPECTED_ROWS, dtype=np.float64)
                for inner_fold in inner_fold_values:
                    validation_mask = outer_train & (assignment == inner_fold)
                    fit_mask = (
                        outer_train
                        & ~validation_mask
                        & (inner_scales[timestep][shrinkage] > 0.0)
                    )
                    model = fit_risk_model_checked(descriptors, labels, fit_mask, active)
                    counts["risk_fit_count"] += 1
                    validation_indices = np.flatnonzero(validation_mask)
                    probabilities[validation_indices] = stagex.predict_risk(
                        model, descriptors[validation_indices]
                    )
                    constants[validation_indices] = float(model["prevalence"])
                risk_oof[timestep][shrinkage] = probabilities
                risk_constant_oof[timestep][shrinkage] = constants

        baseline_records: Dict[int, Mapping[str, Any]] = {}
        policy_records: Dict[str, Dict[int, Mapping[str, Any]]] = {
            policy.policy_id: {} for policy in stagex.policy_population()
        }
        for timestep in LOCKED_TIMESTEPS:
            indices = outer_train_indices
            control = np.asarray(
                context["objective_control_predictions"][int(timestep)], dtype=np.float32
            )[indices]
            baseline_records[timestep] = stagex.evaluate_policy(
                control,
                inner_candidates[timestep][1.0][indices],
                inner_scales[timestep][1.0][indices],
                np.zeros(indices.size, dtype=np.float64),
                target[indices],
                groups[indices],
                1.0,
                None,
                active,
            )
            for policy in stagex.policy_population():
                policy_records[policy.policy_id][timestep] = stagex.evaluate_policy(
                    control,
                    inner_candidates[timestep][policy.shrinkage][indices],
                    inner_scales[timestep][policy.shrinkage][indices],
                    risk_oof[timestep][policy.shrinkage][indices],
                    target[indices],
                    groups[indices],
                    policy.risk_threshold,
                    risk_constant_oof[timestep][policy.shrinkage][indices],
                    active,
                )
                counts["inner_policy_evaluation_count"] += 1
        selection = dict(stagex.select_joint_policy(policy_records, baseline_records, active))
        selection.update(
            {
                "outer_fold": outer_fold,
                "inner_train_row_count": int(np.sum(outer_train)),
                "outer_test_row_count": int(np.sum(outer_test)),
            }
        )
        outer_selections.append(selection)

        outer_outputs[outer_fold] = {}
        for timestep in LOCKED_TIMESTEPS:
            fitted = stagex._fit_direction(
                runtime=runtime,
                context=context,
                timestep=timestep,
                fit_mask=outer_train,
                predict_indices=outer_test_indices,
                oracle=oracle_by_timestep[timestep],
                features=features_by_timestep[timestep],
            )
            counts["direction_fit_count"] += 1
            outer_outputs[outer_fold][timestep] = {}
            control_all = np.asarray(
                context["objective_control_predictions"][int(timestep)], dtype=np.float32
            )
            for shrinkage in DIRECTION_SHRINKAGES:
                generated = stagex._candidate_for_indices(
                    runtime=runtime,
                    context=context,
                    timestep=timestep,
                    indices=outer_test_indices,
                    direction=fitted["direction"],
                    shrinkage=shrinkage,
                    reconstruction_spec=reconstruction_spec,
                )
                require_clean_generation(
                    generated,
                    label="outer{} test t{} shrink{}".format(
                        outer_fold, timestep, shrinkage
                    ),
                )
                counts["candidate_generation_count"] += 1
                counts["internal_scale_attempt_count"] += EXPECTED_INTERNAL_SCALE_COUNT
                training_descriptors = stagex.compact_risk_descriptors(
                    control_all,
                    inner_directions[timestep] * shrinkage,
                    inner_candidates[timestep][shrinkage],
                    inner_scales[timestep][shrinkage],
                )
                training_labels = np.zeros(EXPECTED_ROWS, dtype=np.float64)
                training_labels[outer_train_indices] = stagex._risk_labels(
                    control_all[outer_train_indices],
                    inner_candidates[timestep][shrinkage][outer_train_indices],
                    target[outer_train_indices],
                )
                risk_fit_mask = outer_train & (
                    inner_scales[timestep][shrinkage] > 0.0
                )
                model = fit_risk_model_checked(
                    training_descriptors, training_labels, risk_fit_mask, active
                )
                counts["risk_fit_count"] += 1
                test_control = control_all[outer_test_indices]
                test_descriptors = stagex.compact_risk_descriptors(
                    test_control,
                    fitted["direction"] * shrinkage,
                    generated["candidate"],
                    generated["selected_scale"],
                )
                outer_outputs[outer_fold][timestep][shrinkage] = {
                    "indices": outer_test_indices,
                    "control": test_control,
                    "direction": fitted["direction"],
                    "candidate": generated["candidate"],
                    "selected_scale": generated["selected_scale"],
                    "risk_probability": stagex.predict_risk(model, test_descriptors),
                    "risk_constant_probability": np.full(
                        outer_test_indices.size,
                        float(model["prevalence"]),
                        dtype=np.float64,
                    ),
                    "gate_counts": {
                        "length_log_z_element_mismatch_count": generated[
                            "length_log_z_element_mismatch_count"
                        ],
                        "aligned_upper_element_failure_count": generated[
                            "aligned_upper_element_failure_count"
                        ],
                        "strict_pass_aligned_fail_row_count": generated[
                            "strict_pass_aligned_fail_row_count"
                        ],
                    },
                }

    modal = stagex.modal_policy(
        [item["selected_policy_id"] for item in outer_selections]
    )
    procedure_records, procedure_gate_totals = stitch_fold_selected_procedure(
        context=context,
        outer_outputs=outer_outputs,
        outer_selections=outer_selections,
        target=target,
        groups=groups,
        spec=active,
    )
    fixed_policy_records, baseline_records = fixed_policy_oof_surface(
        context=context,
        outer_outputs=outer_outputs,
        target=target,
        groups=groups,
        spec=active,
    )
    counts["full_objective_oof_policy_evaluation_count"] = (
        len(stagex.policy_population()) * len(LOCKED_TIMESTEPS)
    )
    full_selection = stagex.select_joint_policy(
        fixed_policy_records, baseline_records, active
    )
    classification = classify_resume1(
        procedure_records=procedure_records,
        baseline_records=baseline_records,
        outer_selections=outer_selections,
        modal=modal,
        full_selection=full_selection,
        spec=active,
    )
    recommendation = None
    if classification["scientific_status"] == "READY":
        selected_policy = TailPolicy(**full_selection["selected_policy"])
        selected_policy.validate()
        recommendation = {
            "backbone_id": stagex.LOCKED_BACKBONE,
            "timesteps": list(LOCKED_TIMESTEPS),
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "direction_shrinkage": selected_policy.shrinkage,
            "adverse_risk_threshold": selected_policy.risk_threshold,
            "selection_role": (
                "full_objective_oof_tuning_after_unbiased_outer_crossfit_"
                "selection_procedure_pass"
            ),
            "risk_descriptor": (
                "target_independent_compact_direction_candidate_geometry_v1"
            ),
            "aligned_gate_contract_sha256": stageu_contract["aligned_gate_lock"][
                "inner_gate_contract_sha256"
            ],
        }

    expected_counts = {
        "direction_fit_count": 108,
        "candidate_generation_count": 324,
        "risk_fit_count": 324,
        "internal_scale_attempt_count": 2268,
        "inner_policy_evaluation_count": 216,
        "full_objective_oof_policy_evaluation_count": 36,
        "nonconverged_risk_fit_count": 0,
    }
    if counts != expected_counts:
        raise StageXError(
            "Stage-X Resume1 execution counts changed: {!r}".format(counts)
        )

    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        **classification,
        "process_id": os.getpid(),
        "repository_head": repository_head,
        "environment_sha256": sha256_bytes(stable_json_bytes(environment)),
        "cold_cuda_precheck": copy.deepcopy(dict(cold)),
        "recovery_contract": {
            "original_stagex_implementation_commit": BASE_STAGEX_IMPLEMENTATION_COMMIT,
            "original_science_worker_started": False,
            "original_environment_probe_started": False,
            "original_selection_holdout_access_added": 0,
            "import_bootstrap_added_before_project_import": True,
            "child_pythonpath_explicit": True,
            "original_modal_policy_outer_reuse_forbidden": True,
            "outer_primary_metric_uses_fold_selected_policy": True,
            "full_objective_fixed_policy_surface_role": "tuning_only_after_outer_procedure_estimate",
        },
        "locked_scientific_base": {
            "backbone_id": stagex.LOCKED_BACKBONE,
            "timesteps": list(LOCKED_TIMESTEPS),
            "external_multiplier": 0.25,
            "tolerance_factor": 5.0,
            "internal_scale_count": 7,
        },
        "stagex_spec": asdict(active),
        "policy_population": [asdict(item) for item in stagex.policy_population()],
        "population": population,
        "outer_fold_selections": outer_selections,
        "inner_selection_modal_policy_diagnostic": modal,
        "outer_crossfit_fold_selected_policy_records": {
            str(key): value for key, value in procedure_records.items()
        },
        "outer_crossfit_baseline_records": {
            str(key): value for key, value in baseline_records.items()
        },
        "full_objective_oof_fixed_policy_selection": full_selection,
        "full_objective_oof_fixed_policy_records": {
            policy_id: {str(key): value for key, value in records.items()}
            for policy_id, records in fixed_policy_records.items()
        },
        "procedure_gate_totals": procedure_gate_totals,
        "execution_counts": counts,
        "selected_configuration": None,
        "train_only_recommendation": recommendation,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluated_in_stagex_resume1": False,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_payload(payload)
    return payload


def validate_worker_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageXError("Stage-X Resume1 worker schema/verdict changed")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageXError("Stage-X Resume1 re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageXError("cumulative selection-holdout count changed")
    if payload.get("selected_configuration") is not None:
        raise StageXError("Stage-X Resume1 selected a deployment configuration")
    if payload.get("rerun_authorized") is not False:
        raise StageXError("Stage-X Resume1 authorized rerun")
    records = payload.get("outer_crossfit_fold_selected_policy_records")
    if not isinstance(records, Mapping) or set(records) != {"10", "25", "50"}:
        raise StageXError("outer crossfit record population changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageXError("a forbidden boundary became true")
    if payload.get("scientific_status") == "READY" and payload.get("train_only_recommendation") is None:
        raise StageXError("READY result lacks train-only recommendation")
    if payload.get("scientific_status") == "BLOCKED" and payload.get("train_only_recommendation") is not None:
        raise StageXError("BLOCKED result retained recommendation")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "scientific_result_sha256"}
        )
    )
    if payload.get("scientific_result_sha256") != expected:
        raise StageXError("Stage-X Resume1 worker self-hash changed")


def blocked_report(
    repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagex_resume1_execution_contract_failed",
        "required_next_path": (
            "DESIGN_ADD_ONLY_STAGEX_RESUME1_EXECUTION_RECOVERY_WITHOUT_"
            "HOLDOUT_OR_PROBE_REACCESS_IF_SCIENCE_DID_NOT_START"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "complete_nested_oof_population_claimed": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }


write_once = stagex.write_once
