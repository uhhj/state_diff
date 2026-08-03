"""Read-only Stage-E audit of Stage-D Resume3 outer-fold recipe instability.

This stage performs no CUDA work, no dataset loading, no fitting, no candidate
reconstruction and no holdout/frozen-probe access.  It reads the committed
Stage-D Resume3 worker evidence and summary, validates their frozen identities,
and audits whether the failed exact-recipe modal-support gate reflects:

1. instability in operational axes (shrinkage / risk threshold),
2. instability in the eligible recipe family, or
3. secondary calibration variation inside an operationally stable family.

The source evidence does not persist every recipe's inner selection score or
row-level accept masks.  Rank margins and mask Hamming distances are therefore
reported as unavailable rather than reconstructed or guessed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.9 Stage E"
SCHEMA = "phase314b_r259_stagee_recipe_instability_audit_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_HEAD = "6c1de2f5af23f2525414139a223d41b73037d74b"
BASE_IMPLEMENTATION = "190ce56a8e579425ed074ced94b1151e933f0c85"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage E: audit t10 outer-fold recipe instability"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage E recipe-instability audit"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage E blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagee_recipe_instability_audit.py"),
    ("A", "scripts/phase3_14b_r259_stagee_recipe_instability_audit.py"),
    ("A", "tests/test_phase3_14b_r259_stagee_recipe_instability_audit.py"),
)

SOURCE_WORKER = (
    "reports/phase3_14b_r259_staged_resume3_t10_risk_repair_worker_evidence.json"
)
SOURCE_WORKER_SHA256 = (
    "879a4cc1a9b00ec9d82bac7ee1fbaad36f2d2c23532d048994fee408edb2a408"
)
SOURCE_WORKER_RESULT_SHA256 = (
    "85863ca30410882a65ecf59b2aa123646cbdef9d9dbcef3a3ca5fd69febd56ae"
)
SOURCE_SUMMARY = (
    "reports/phase3_14b_r259_staged_resume3_variable_threshold_call_order_recovery_summary.json"
)
SOURCE_SUMMARY_SHA256 = (
    "95f15edd666a7fc56d374e8892f1964bbc68b64d564ddf57b9ba8e08f3488466"
)
SOURCE_SUMMARY_SELF_SHA256 = (
    "488ab5de981564372d27e31511442a1a71afc13b3ee4f8cb94078c01bce13158"
)

SUCCESS_REPORT = "reports/phase3_14b_r259_stagee_recipe_instability_audit_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_stagee_recipe_instability_audit_blocked_summary.json"

EXPECTED_OUTER_FOLDS = 6
EXPECTED_RECIPE_COUNT = 360
EXPECTED_EXACT_MODAL_SUPPORT = 3
REQUIRED_EXACT_MODAL_SUPPORT = 4
OPERATIONAL_AXES = ("shrinkage", "risk_threshold")
CALIBRATION_AXES = ("descriptor_id", "risk_l2", "temperature")
DESCRIPTOR_IDS = ("compact_v1", "compact_geometry_v2")
RISK_L2_VALUES = (0.25, 1.0, 4.0)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
SHRINKAGES = (0.5, 0.75, 1.0)
RISK_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)
SOURCE_WORKER_SCHEMA = "phase314b_r259_staged_t10_risk_descriptor_calibration_repair_v1_worker_v1"
SOURCE_SUMMARY_SCHEMA = "phase314b_r259_staged_resume3_variable_threshold_call_order_recovery_v1"

FALSE_BOUNDARIES = (
    "environment_probe_run",
    "science_worker_run",
    "cuda_used",
    "dataset_loaded",
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded",
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
    "npz_saved",
    "descriptor_tensor_persisted",
    "risk_probability_tensor_persisted",
    "candidate_tensor_persisted",
    "prediction_tensor_persisted",
    "image_saved",
    "video_saved",
    "cache_saved",
)


class StageEAuditError(RuntimeError):
    """Fail-closed Stage-E audit error."""


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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise StageEAuditError("JSON root is not a mapping: {}".format(path))
    return payload


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageEAuditError("write-once output exists: {}".format(target))
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
            raise StageEAuditError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageEAuditError("{} worktree is dirty".format(label))


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageEAuditError("branch must be Experiment1")
    if _git(repo, "rev-parse", "HEAD") != implementation_commit:
        raise StageEAuditError("HEAD is not the Stage-E implementation commit")
    if _git(repo, "rev-parse", "HEAD^") != BASE_HEAD:
        raise StageEAuditError("Stage-E implementation parent changed")
    if _git(repo, "rev-parse", BASE_HEAD + "^") != BASE_IMPLEMENTATION:
        raise StageEAuditError("Stage-D Resume3 evidence parent changed")
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageEAuditError("Stage-E implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageEAuditError("Stage-E implementation population changed")
    if _git(repo, "rev-parse", "HEAD:external/deformable-ravens") != EXPECTED_SUBMODULE:
        raise StageEAuditError("DeformableRavens gitlink changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageEAuditError("DeformableRavens worktree commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-E")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageEAuditError("Stage-E terminal report already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_d_resume3_implementation": BASE_IMPLEMENTATION,
        "stage_d_resume3_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def validate_source_evidence(root: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    repo = Path(root).resolve()
    worker_path = repo / SOURCE_WORKER
    summary_path = repo / SOURCE_SUMMARY
    if sha256_file(worker_path) != SOURCE_WORKER_SHA256:
        raise StageEAuditError("Stage-D Resume3 worker file SHA changed")
    if sha256_file(summary_path) != SOURCE_SUMMARY_SHA256:
        raise StageEAuditError("Stage-D Resume3 summary file SHA changed")
    worker = load_json(worker_path)
    summary = load_json(summary_path)
    if worker.get("schema") != SOURCE_WORKER_SCHEMA or worker.get("execution_verdict") != "PASS":
        raise StageEAuditError("Stage-D Resume3 worker schema/verdict changed")
    worker_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in worker.items() if key != "worker_result_sha256"}
        )
    )
    if worker.get("worker_result_sha256") != SOURCE_WORKER_RESULT_SHA256 or worker_hash != SOURCE_WORKER_RESULT_SHA256:
        raise StageEAuditError("Stage-D Resume3 worker-result SHA changed")
    if summary.get("schema") != SOURCE_SUMMARY_SCHEMA or summary.get("execution_verdict") != "PASS":
        raise StageEAuditError("Stage-D Resume3 summary schema/verdict changed")
    summary_hash = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in summary.items() if key != "summary_sha256"}
        )
    )
    if summary.get("summary_sha256") != SOURCE_SUMMARY_SELF_SHA256 or summary_hash != SOURCE_SUMMARY_SELF_SHA256:
        raise StageEAuditError("Stage-D Resume3 summary self-hash changed")
    expected = {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_staged_t10_risk_recipe_is_not_outer_fold_stable",
        "required_next_path": "AUDIT_R259_STAGED_T10_RISK_RECIPE_OUTER_FOLD_INSTABILITY",
        "primary_failure_locus": "outer_recipe_instability",
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise StageEAuditError("Stage-D Resume3 summary field changed: {}".format(key))
    if summary.get("outer_fold_selections") != worker.get("outer_fold_selections"):
        raise StageEAuditError("summary/worker outer selections differ")
    if summary.get("recipe_population") != worker.get("recipe_population"):
        raise StageEAuditError("summary/worker recipe population differs")
    return worker, summary


def recipe_id(recipe: Mapping[str, Any]) -> str:
    descriptor_id = str(recipe["descriptor_id"])
    risk_l2 = float(recipe["risk_l2"])
    temperature = float(recipe["temperature"])
    shrinkage = float(recipe["shrinkage"])
    risk_threshold = float(recipe["risk_threshold"])
    if descriptor_id not in DESCRIPTOR_IDS:
        raise StageEAuditError("descriptor left frozen bank")
    if risk_l2 not in RISK_L2_VALUES:
        raise StageEAuditError("risk L2 left frozen bank")
    if temperature not in TEMPERATURES:
        raise StageEAuditError("temperature left frozen bank")
    if shrinkage not in SHRINKAGES:
        raise StageEAuditError("shrinkage left frozen bank")
    if risk_threshold not in RISK_THRESHOLDS:
        raise StageEAuditError("risk threshold left frozen bank")
    return "desc_{}__l2_{:.2f}__temp_{:.2f}__shrink_{:.2f}__risk_{:.2f}".format(
        descriptor_id, risk_l2, temperature, shrinkage, risk_threshold
    )


def operational_family_id(recipe: Mapping[str, Any]) -> str:
    return "shrink_{:.2f}__risk_{:.2f}".format(
        float(recipe["shrinkage"]), float(recipe["risk_threshold"])
    )


def calibration_family_id(recipe: Mapping[str, Any]) -> str:
    return "desc_{}__l2_{:.2f}__temp_{:.2f}".format(
        recipe["descriptor_id"], float(recipe["risk_l2"]), float(recipe["temperature"])
    )


def _counter_report(values: Iterable[str]) -> Mapping[str, Any]:
    counts = collections.Counter(values)
    if not counts:
        raise StageEAuditError("cannot summarize empty support population")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "counts": {key: value for key, value in sorted(counts.items())},
        "modal_value": ordered[0][0],
        "modal_support": ordered[0][1],
        "population": sum(counts.values()),
        "unique_count": len(counts),
    }


def _metric_spread(records: Sequence[Mapping[str, Any]], path: Tuple[str, ...]) -> Mapping[str, float]:
    values: List[float] = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise StageEAuditError("missing common-surface metric: {}".format(".".join(path)))
            value = value[key]
        values.append(float(value))
    return {
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def audit_recipe_instability(worker: Mapping[str, Any], summary: Mapping[str, Any]) -> Mapping[str, Any]:
    population = worker.get("recipe_population")
    selections = worker.get("outer_fold_selections")
    full_records = worker.get("full_objective_oof_fixed_recipe_records")
    full_selection = worker.get("full_objective_oof_fixed_recipe_selection")
    if not isinstance(population, list) or len(population) != EXPECTED_RECIPE_COUNT:
        raise StageEAuditError("recipe population is not exactly 360")
    if not isinstance(selections, list) or len(selections) != EXPECTED_OUTER_FOLDS:
        raise StageEAuditError("outer-fold selection population is not exactly 6")
    if not isinstance(full_records, Mapping) or len(full_records) != EXPECTED_RECIPE_COUNT:
        raise StageEAuditError("full-objective recipe record population changed")
    if not isinstance(full_selection, Mapping):
        raise StageEAuditError("full-objective selection is missing")

    recipe_map: Dict[str, Mapping[str, Any]] = {}
    for recipe in population:
        if not isinstance(recipe, Mapping):
            raise StageEAuditError("recipe record is not a mapping")
        rid = recipe_id(recipe)
        if rid in recipe_map:
            raise StageEAuditError("duplicate recipe ID")
        recipe_map[rid] = recipe
    if set(recipe_map) != set(full_records):
        raise StageEAuditError("recipe population/full-objective record IDs differ")

    selected_ids: List[str] = []
    eligible_sets: List[set[str]] = []
    selected_recipes: List[Mapping[str, Any]] = []
    eligible_counts: List[int] = []
    for expected_fold, selection in enumerate(selections):
        if not isinstance(selection, Mapping) or selection.get("outer_fold") != expected_fold:
            raise StageEAuditError("outer folds are not ordered 0..5")
        if selection.get("diagnostic_fallback_used") is not False:
            raise StageEAuditError("Stage-D Resume3 used a diagnostic fallback")
        if selection.get("selected_recipe_inner_eligible") is not True:
            raise StageEAuditError("selected recipe was not inner eligible")
        rid = str(selection.get("selected_recipe_id"))
        selected = selection.get("selected_recipe")
        if not isinstance(selected, Mapping) or recipe_id(selected) != rid:
            raise StageEAuditError("selected recipe ID/payload mismatch")
        eligible = selection.get("eligible_recipe_ids")
        if not isinstance(eligible, list) or not eligible:
            raise StageEAuditError("eligible recipe population is missing")
        eligible_set = set(map(str, eligible))
        if rid not in eligible_set or not eligible_set.issubset(recipe_map):
            raise StageEAuditError("selected/eligible recipe identity mismatch")
        selected_ids.append(rid)
        selected_recipes.append(selected)
        eligible_sets.append(eligible_set)
        eligible_counts.append(len(eligible_set))

    exact_support = _counter_report(selected_ids)
    if exact_support["modal_support"] != EXPECTED_EXACT_MODAL_SUPPORT:
        raise StageEAuditError("Stage-D exact modal support changed")

    factor_support = {
        "descriptor_id": _counter_report(str(item["descriptor_id"]) for item in selected_recipes),
        "risk_l2": _counter_report("{:.2f}".format(float(item["risk_l2"])) for item in selected_recipes),
        "temperature": _counter_report("{:.2f}".format(float(item["temperature"])) for item in selected_recipes),
        "shrinkage": _counter_report("{:.2f}".format(float(item["shrinkage"])) for item in selected_recipes),
        "risk_threshold": _counter_report("{:.2f}".format(float(item["risk_threshold"])) for item in selected_recipes),
    }
    operational_support = _counter_report(operational_family_id(item) for item in selected_recipes)
    calibration_support = _counter_report(calibration_family_id(item) for item in selected_recipes)

    intersection = set.intersection(*eligible_sets)
    union = set.union(*eligible_sets)
    eligibility_support = collections.Counter(rid for values in eligible_sets for rid in values)
    support_histogram = collections.Counter(eligibility_support.values())

    selected_operational_family = str(operational_support["modal_value"])
    family_population = {
        rid for rid, recipe in recipe_map.items()
        if operational_family_id(recipe) == selected_operational_family
    }
    family_eligible_counts = [len(values & family_population) for values in eligible_sets]
    family_fully_eligible_each_fold = all(
        count == len(family_population) for count in family_eligible_counts
    )

    unique_selected_ids = sorted(set(selected_ids))
    common_surface_records: List[Mapping[str, Any]] = []
    selected_full_eligibility: Dict[str, Any] = {}
    full_eligibility = full_selection.get("recipe_eligibility")
    if not isinstance(full_eligibility, Mapping):
        raise StageEAuditError("full-objective eligibility surface is missing")
    for rid in unique_selected_ids:
        record = full_records.get(rid)
        eligibility = full_eligibility.get(rid)
        if not isinstance(record, Mapping) or not isinstance(eligibility, Mapping):
            raise StageEAuditError("selected recipe missing from common surface")
        common_surface_records.append(record)
        selected_full_eligibility[rid] = eligibility
    all_selected_full_objective_eligible = all(
        value.get("pass") is True for value in selected_full_eligibility.values()
    )

    metric_paths = {
        "acceptance_rate": ("acceptance_rate",),
        "overall_mse_ratio": ("overall_mse_ratio",),
        "accepted_row_mse_ratio": ("accepted_row_mse_ratio",),
        "positive_distance_reduction_rate": ("positive_distance_reduction_rate",),
        "relative_distance_reduction_mean": ("relative_distance_reduction_mean",),
        "risk_brier_score": ("risk_brier_score",),
        "worst_fraction_cvar_mse_ratio": ("group_tail", "worst_fraction_cvar_mse_ratio"),
        "adverse_sse_mass": ("adverse_sse_mass",),
        "net_sse_reduction_mass": ("net_sse_reduction_mass",),
    }
    common_surface_spreads = {
        name: _metric_spread(common_surface_records, path)
        for name, path in metric_paths.items()
    }
    selected_scale_shas = {
        str(record["output_selected_scale_sha256"]) for record in common_surface_records
    }
    candidate_shas = {
        str(record["output_candidate_sha256"]) for record in common_surface_records
    }

    operational_stable = operational_support["modal_support"] == EXPECTED_OUTER_FOLDS
    exact_stable = exact_support["modal_support"] >= REQUIRED_EXACT_MODAL_SUPPORT
    if exact_stable:
        classification = "exact_recipe_stable_source_contract_inconsistent"
        root_cause = "phase314b_r259_stagee_stage_d_instability_claim_conflicts_with_exact_support"
        next_path = "RECONCILE_R259_STAGED_RECIPE_STABILITY_EVIDENCE"
    elif not operational_stable:
        classification = "operational_policy_axes_are_outer_fold_unstable"
        root_cause = "phase314b_r259_stagee_t10_operational_risk_policy_is_outer_fold_unstable"
        next_path = "REDESIGN_R259_STAGED_T10_OPERATIONAL_RISK_POLICY"
    elif not family_fully_eligible_each_fold:
        classification = "operational_family_selected_but_eligibility_is_fold_dependent"
        root_cause = "phase314b_r259_stagee_t10_operational_family_eligibility_is_outer_fold_unstable"
        next_path = "REDESIGN_R259_STAGED_T10_CALIBRATION_TRANSFER"
    elif not all_selected_full_objective_eligible:
        classification = "selected_calibration_recipes_do_not_transfer_to_common_surface"
        root_cause = "phase314b_r259_stagee_t10_selected_calibration_recipes_fail_common_surface_transfer"
        next_path = "REDESIGN_R259_STAGED_T10_CALIBRATION_TRANSFER"
    else:
        classification = "operational_family_stable_calibration_recipe_realizations_differ"
        root_cause = "phase314b_r259_stagee_t10_exact_recipe_instability_is_secondary_within_fully_eligible_operational_family"
        next_path = "DESIGN_R259_STAGED_T10_OPERATIONAL_FAMILY_LOCK_WITH_NESTED_CALIBRATION_CANONICALIZATION"

    return {
        "classification": classification,
        "root_cause": root_cause,
        "required_next_path": next_path,
        "exact_recipe_support": exact_support,
        "required_exact_modal_support": REQUIRED_EXACT_MODAL_SUPPORT,
        "factor_support": factor_support,
        "operational_family_axes": list(OPERATIONAL_AXES),
        "operational_family_support": operational_support,
        "calibration_axes": list(CALIBRATION_AXES),
        "calibration_family_support": calibration_support,
        "eligible_surface": {
            "eligible_recipe_count_by_fold": eligible_counts,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "intersection_recipe_ids_sha256": sha256_bytes(
                stable_json_bytes(sorted(intersection))
            ),
            "union_recipe_ids_sha256": sha256_bytes(stable_json_bytes(sorted(union))),
            "eligibility_support_histogram": {
                str(key): value for key, value in sorted(support_histogram.items())
            },
            "recipes_eligible_in_all_six_folds": sum(
                count == EXPECTED_OUTER_FOLDS for count in eligibility_support.values()
            ),
        },
        "selected_operational_family": {
            "family_id": selected_operational_family,
            "recipe_population": len(family_population),
            "eligible_recipe_count_by_fold": family_eligible_counts,
            "all_family_recipes_eligible_in_every_fold": family_fully_eligible_each_fold,
            "family_recipe_ids_sha256": sha256_bytes(
                stable_json_bytes(sorted(family_population))
            ),
        },
        "common_full_objective_surface": {
            "unique_selected_recipe_count": len(unique_selected_ids),
            "unique_selected_recipe_ids": unique_selected_ids,
            "selected_recipe_eligibility": selected_full_eligibility,
            "all_selected_recipes_eligible": all_selected_full_objective_eligible,
            "metric_spreads": common_surface_spreads,
            "unique_selected_scale_sha_count": len(selected_scale_shas),
            "unique_candidate_sha_count": len(candidate_shas),
            "exact_output_identity": len(selected_scale_shas) == 1 and len(candidate_shas) == 1,
        },
        "source_evidence_limitations": {
            "per_recipe_inner_selection_scores_persisted": False,
            "near_tie_rank_margin_computable": False,
            "row_level_accept_masks_persisted": False,
            "accept_mask_hamming_distance_computable": False,
            "reason": (
                "outer-fold evidence stores the selected score and all recipe eligibility, "
                "but not every recipe's inner score or row-level masks"
            ),
        },
    }


def build_summary(repository: Mapping[str, Any], worker: Mapping[str, Any], source_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = audit_recipe_instability(worker, source_summary)
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "outer_recipe_instability_audit",
        "root_cause": audit["root_cause"],
        "required_next_path": audit["required_next_path"],
        "repository": dict(repository),
        "source_evidence": {
            "worker_path": SOURCE_WORKER,
            "worker_file_sha256": SOURCE_WORKER_SHA256,
            "worker_result_sha256": SOURCE_WORKER_RESULT_SHA256,
            "summary_path": SOURCE_SUMMARY,
            "summary_file_sha256": SOURCE_SUMMARY_SHA256,
            "summary_self_sha256": SOURCE_SUMMARY_SELF_SHA256,
        },
        "audit": audit,
        "execution_counts": {
            "environment_probe_count": 0,
            "science_worker_count": 0,
            "cuda_operation_count": 0,
            "dataset_load_count": 0,
            "direction_fit_count": 0,
            "candidate_generation_count": 0,
            "risk_fit_count": 0,
            "recipe_evaluation_count": 0,
            "report_read_count": 2,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
    }
    for key in FALSE_BOUNDARIES:
        result[key] = False
    result["summary_sha256"] = sha256_bytes(stable_json_bytes(result))
    validate_summary(result)
    return result


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageEAuditError("Stage-E summary schema/verdict changed")
    if payload.get("scientific_status") != "BLOCKED":
        raise StageEAuditError("Stage-E audit must remain scientifically blocked")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise StageEAuditError("Stage-E audit payload missing")
    if audit.get("exact_recipe_support", {}).get("modal_support") != EXPECTED_EXACT_MODAL_SUPPORT:
        raise StageEAuditError("exact recipe modal support changed")
    if payload.get("selected_configuration") is not None:
        raise StageEAuditError("Stage-E selected a configuration")
    if payload.get("train_only_recommendation") is not None:
        raise StageEAuditError("Stage-E created a train-only recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageEAuditError("Stage-E re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageEAuditError("Stage-E holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageEAuditError("Stage-E accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageEAuditError("Stage-E authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageEAuditError("Stage-E crossed a forbidden boundary")
    counts = payload.get("execution_counts")
    if not isinstance(counts, Mapping):
        raise StageEAuditError("Stage-E execution counts missing")
    for key, value in counts.items():
        if key == "report_read_count":
            if value != 2:
                raise StageEAuditError("Stage-E report read count changed")
        elif value != 0:
            raise StageEAuditError("Stage-E performed scientific execution")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageEAuditError("Stage-E summary self-hash changed")


def blocked_report(repository: Optional[Mapping[str, Any]], error: BaseException) -> Mapping[str, Any]:
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "stagee_execution_contract",
        "root_cause": "phase314b_r259_stagee_recipe_instability_audit_execution_failed",
        "required_next_path": "AUDIT_STAGEE_INPUT_OR_SCHEMA_BEFORE_ANY_FURTHER_SCIENCE",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
    }
    for key in FALSE_BOUNDARIES:
        result[key] = False
    return result
