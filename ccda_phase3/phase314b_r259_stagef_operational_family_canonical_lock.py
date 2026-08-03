"""Stage F: lock the stable t10 operational family and canonical calibration.

This stage performs no CUDA work, dataset loading, fitting, candidate generation,
or recipe evaluation. It reads the immutable Stage-D Resume3 worker evidence and
Stage-E audit summary. Stage E proved that the operational axes
``shrinkage=1.00`` and ``risk_threshold=0.50`` have 6/6 outer-fold support and
that all 30 calibration recipes inside that family are eligible in every fold.

Stage F applies a preregistered structural canonicalization that is independent
of performance metrics:

1. restrict to the locked operational family;
2. prefer the smallest descriptor representation;
3. prefer identity temperature (least probability transformation);
4. prefer the strongest preregistered L2 regularization;
5. use recipe ID only as a final deterministic tie-break.

The resulting canonical recipe is then checked against the already-persisted
nested outer-OOF eligibility and fixed-recipe outer-OOF record. No historical
science is recomputed or altered.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.9 Stage F"
SCHEMA = "phase314b_r259_stagef_operational_family_canonical_lock_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_HEAD = "3457df7143f8585bb381fa1acb07452daf2567ef"
BASE_IMPLEMENTATION = "dbd38ee9a4389ecddd86bf673b1d6b1794fead99"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage F: lock t10 operational family and canonical calibration"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage F canonical calibration lock"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage F blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagef_operational_family_canonical_lock.py"),
    ("A", "scripts/phase3_14b_r259_stagef_operational_family_canonical_lock.py"),
    ("A", "tests/test_phase3_14b_r259_stagef_operational_family_canonical_lock.py"),
)

SOURCE_STAGEE = "reports/phase3_14b_r259_stagee_recipe_instability_audit_summary.json"
SOURCE_STAGEE_SHA256 = "026343261f32b806114d272e818818833e7c40fa9edeca21a4af510e642d3cea"
SOURCE_STAGEE_SELF_SHA256 = "9da37bc108e33525a9e4ff708689a4258b8ea2f76eb9122324310ec740c553af"
SOURCE_STAGEE_SCHEMA = "phase314b_r259_stagee_recipe_instability_audit_v1"

SOURCE_WORKER = "reports/phase3_14b_r259_staged_resume3_t10_risk_repair_worker_evidence.json"
SOURCE_WORKER_SHA256 = "879a4cc1a9b00ec9d82bac7ee1fbaad36f2d2c23532d048994fee408edb2a408"
SOURCE_WORKER_RESULT_SHA256 = "85863ca30410882a65ecf59b2aa123646cbdef9d9dbcef3a3ca5fd69febd56ae"
SOURCE_WORKER_SCHEMA = "phase314b_r259_staged_t10_risk_descriptor_calibration_repair_v1_worker_v1"

SUCCESS_REPORT = "reports/phase3_14b_r259_stagef_operational_family_canonical_lock_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_stagef_operational_family_canonical_lock_blocked_summary.json"

EXPECTED_RECIPE_COUNT = 360
EXPECTED_OUTER_FOLDS = 6
LOCKED_SHRINKAGE = 1.0
LOCKED_RISK_THRESHOLD = 0.5
LOCKED_FAMILY_ID = "shrink_1.00__risk_0.50"
EXPECTED_FAMILY_RECIPE_COUNT = 30
EXPECTED_FAMILY_RECIPE_IDS_SHA256 = "cb9f85387bc1cf9fce8f1b090b5c2044cfbef038a7814d280f537b5435573776"

DESCRIPTOR_DIMENSIONS = {
    "compact_v1": 17,
    "compact_geometry_v2": 66,
}
RISK_L2_VALUES = (0.25, 1.0, 4.0)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
SHRINKAGES = (0.5, 0.75, 1.0)
RISK_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)

CANONICAL_RECIPE_ID = (
    "desc_compact_v1__l2_4.00__temp_1.00__shrink_1.00__risk_0.50"
)

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


class StageFError(RuntimeError):
    """Fail-closed Stage-F error."""


@dataclass(frozen=True)
class Recipe:
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
        raise StageFError("JSON root is not a mapping: {}".format(path))
    return payload


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageFError("write-once output exists: {}".format(target))
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
            raise StageFError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageFError("{} worktree is dirty".format(label))


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-E evidence parent": (
            _git(repo, "rev-parse", BASE_HEAD + "^"),
            BASE_IMPLEMENTATION,
        ),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageFError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageFError("Stage-F implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageFError("Stage-F implementation population changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageFError("DeformableRavens worktree commit changed")
    assert_clean_worktree(repo, "Stage-F")
    assert_clean_worktree(submodule, "DeformableRavens")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageFError("Stage-F terminal report already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_e_implementation": BASE_IMPLEMENTATION,
        "stage_e_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != field})
    )


def validate_source_evidence(root: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    repo = Path(root).resolve()
    stagee_path = repo / SOURCE_STAGEE
    worker_path = repo / SOURCE_WORKER
    if not stagee_path.is_file() or sha256_file(stagee_path) != SOURCE_STAGEE_SHA256:
        raise StageFError("Stage-E summary file SHA changed")
    if not worker_path.is_file() or sha256_file(worker_path) != SOURCE_WORKER_SHA256:
        raise StageFError("Stage-D Resume3 worker file SHA changed")
    stagee = load_json(stagee_path)
    worker = load_json(worker_path)
    if stagee.get("schema") != SOURCE_STAGEE_SCHEMA or stagee.get("execution_verdict") != "PASS":
        raise StageFError("Stage-E summary schema/verdict changed")
    if stagee.get("summary_sha256") != SOURCE_STAGEE_SELF_SHA256:
        raise StageFError("Stage-E summary self-hash identity changed")
    if _self_hash(stagee, "summary_sha256") != SOURCE_STAGEE_SELF_SHA256:
        raise StageFError("Stage-E summary self-hash recomputation failed")
    audit = stagee.get("audit")
    if not isinstance(audit, Mapping):
        raise StageFError("Stage-E audit payload missing")
    if audit.get("classification") != "operational_family_stable_calibration_recipe_realizations_differ":
        raise StageFError("Stage-E classification changed")
    if audit.get("required_next_path") != (
        "DESIGN_R259_STAGED_T10_OPERATIONAL_FAMILY_LOCK_WITH_NESTED_CALIBRATION_CANONICALIZATION"
    ):
        raise StageFError("Stage-E next path changed")
    family = audit.get("selected_operational_family")
    if not isinstance(family, Mapping):
        raise StageFError("Stage-E operational family missing")
    expected_family = {
        "family_id": LOCKED_FAMILY_ID,
        "recipe_population": EXPECTED_FAMILY_RECIPE_COUNT,
        "eligible_recipe_count_by_fold": [30] * EXPECTED_OUTER_FOLDS,
        "all_family_recipes_eligible_in_every_fold": True,
        "family_recipe_ids_sha256": EXPECTED_FAMILY_RECIPE_IDS_SHA256,
    }
    if dict(family) != expected_family:
        raise StageFError("Stage-E operational family contract changed")
    if worker.get("schema") != SOURCE_WORKER_SCHEMA or worker.get("execution_verdict") != "PASS":
        raise StageFError("Stage-D worker schema/verdict changed")
    if worker.get("worker_result_sha256") != SOURCE_WORKER_RESULT_SHA256:
        raise StageFError("Stage-D worker-result identity changed")
    if _self_hash(worker, "worker_result_sha256") != SOURCE_WORKER_RESULT_SHA256:
        raise StageFError("Stage-D worker-result self-hash recomputation failed")
    return stagee, worker


def recipe_from_mapping(value: Mapping[str, Any]) -> Recipe:
    try:
        result = Recipe(
            descriptor_id=str(value["descriptor_id"]),
            risk_l2=float(value["risk_l2"]),
            temperature=float(value["temperature"]),
            shrinkage=float(value["shrinkage"]),
            risk_threshold=float(value["risk_threshold"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StageFError("invalid recipe payload") from error
    if result.descriptor_id not in DESCRIPTOR_DIMENSIONS:
        raise StageFError("recipe descriptor left frozen bank")
    if result.risk_l2 not in RISK_L2_VALUES:
        raise StageFError("recipe L2 left frozen bank")
    if result.temperature not in TEMPERATURES:
        raise StageFError("recipe temperature left frozen bank")
    if result.shrinkage not in SHRINKAGES:
        raise StageFError("recipe shrinkage left frozen bank")
    if result.risk_threshold not in RISK_THRESHOLDS:
        raise StageFError("recipe risk threshold left frozen bank")
    return result


def operational_family_id(recipe: Recipe) -> str:
    return "shrink_{:.2f}__risk_{:.2f}".format(
        recipe.shrinkage, recipe.risk_threshold
    )


def canonical_order(recipe: Recipe) -> Tuple[Any, ...]:
    """Structural order only; no observed performance metric is used."""
    if operational_family_id(recipe) != LOCKED_FAMILY_ID:
        raise StageFError("canonical order received recipe outside locked family")
    return (
        DESCRIPTOR_DIMENSIONS[recipe.descriptor_id],
        abs(math.log(recipe.temperature)),
        -recipe.risk_l2,
        recipe.recipe_id,
    )


def canonicalize_family(population: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(population) != EXPECTED_RECIPE_COUNT:
        raise StageFError("recipe population is not exactly 360")
    recipes = [recipe_from_mapping(value) for value in population]
    ids = [recipe.recipe_id for recipe in recipes]
    if len(set(ids)) != EXPECTED_RECIPE_COUNT:
        raise StageFError("recipe IDs are not unique")
    family = [
        recipe for recipe in recipes if operational_family_id(recipe) == LOCKED_FAMILY_ID
    ]
    if len(family) != EXPECTED_FAMILY_RECIPE_COUNT:
        raise StageFError("locked family population is not exactly 30")
    family_ids = sorted(recipe.recipe_id for recipe in family)
    if sha256_bytes(stable_json_bytes(family_ids)) != EXPECTED_FAMILY_RECIPE_IDS_SHA256:
        raise StageFError("locked family identity changed")
    ordered = sorted(family, key=canonical_order)
    selected = ordered[0]
    if selected.recipe_id != CANONICAL_RECIPE_ID:
        raise StageFError("canonical recipe identity changed")
    return {
        "rule_id": "minimum_descriptor__identity_temperature__maximum_l2__recipe_id",
        "performance_metrics_used": False,
        "operational_family_id": LOCKED_FAMILY_ID,
        "family_recipe_count": len(family),
        "family_recipe_ids_sha256": EXPECTED_FAMILY_RECIPE_IDS_SHA256,
        "ordering_axes": [
            "descriptor_dimension_ascending",
            "absolute_log_temperature_ascending",
            "risk_l2_descending",
            "recipe_id_ascending",
        ],
        "canonical_recipe": asdict(selected),
        "canonical_recipe_id": selected.recipe_id,
        "canonical_order_key": list(canonical_order(selected)),
    }


def _require_all_checks(eligibility: Mapping[str, Any], label: str) -> None:
    if eligibility.get("pass") is not True:
        raise StageFError("{} is not eligible".format(label))
    checks = eligibility.get("checks")
    expected = {
        "acceptance",
        "accepted_mse",
        "adverse_sse_reduction",
        "group_cvar_improvement",
        "overall_mse",
        "positive_reduction",
        "relative_reduction",
        "risk_brier_nonworse",
    }
    if not isinstance(checks, Mapping) or set(checks) != expected:
        raise StageFError("{} eligibility checks changed".format(label))
    if any(value is not True for value in checks.values()):
        raise StageFError("{} eligibility contains failed checks".format(label))


def lock_canonical_recipe(stagee: Mapping[str, Any], worker: Mapping[str, Any]) -> Mapping[str, Any]:
    population = worker.get("recipe_population")
    if not isinstance(population, list):
        raise StageFError("worker recipe population missing")
    canonical = canonicalize_family(population)
    rid = str(canonical["canonical_recipe_id"])

    selections = worker.get("outer_fold_selections")
    if not isinstance(selections, list) or len(selections) != EXPECTED_OUTER_FOLDS:
        raise StageFError("outer-fold selection population changed")
    fold_support: List[Mapping[str, Any]] = []
    for expected_fold, selection in enumerate(selections):
        if not isinstance(selection, Mapping) or selection.get("outer_fold") != expected_fold:
            raise StageFError("outer folds are not ordered 0..5")
        eligible = selection.get("eligible_recipe_ids")
        eligibility = selection.get("recipe_eligibility")
        if not isinstance(eligible, list) or not isinstance(eligibility, Mapping):
            raise StageFError("outer-fold eligibility surface missing")
        current = eligibility.get(rid)
        if rid not in eligible or not isinstance(current, Mapping):
            raise StageFError("canonical recipe missing from outer-fold eligible set")
        _require_all_checks(current, "outer fold {} canonical recipe".format(expected_fold))
        fold_support.append(
            {
                "outer_fold": expected_fold,
                "canonical_recipe_eligible": True,
                "eligible_recipe_count": len(eligible),
            }
        )

    records = worker.get("full_objective_oof_fixed_recipe_records")
    full_selection = worker.get("full_objective_oof_fixed_recipe_selection")
    if not isinstance(records, Mapping) or len(records) != EXPECTED_RECIPE_COUNT:
        raise StageFError("fixed-recipe OOF record population changed")
    if not isinstance(full_selection, Mapping):
        raise StageFError("fixed-recipe OOF eligibility surface missing")
    eligibility_surface = full_selection.get("recipe_eligibility")
    if not isinstance(eligibility_surface, Mapping):
        raise StageFError("fixed-recipe eligibility surface missing")
    record = records.get(rid)
    eligibility = eligibility_surface.get(rid)
    if not isinstance(record, Mapping) or not isinstance(eligibility, Mapping):
        raise StageFError("canonical fixed-recipe OOF evidence missing")
    _require_all_checks(eligibility, "canonical fixed-recipe OOF")

    controls = worker.get("legacy_stagec_control_contract")
    if not isinstance(controls, Mapping):
        raise StageFError("legacy t25/t50 controls missing")
    for timestep in (25, 50):
        control = controls.get(str(timestep))
        if not isinstance(control, Mapping) or control.get("all_pass") is not True:
            raise StageFError("frozen t{} control is not passing".format(timestep))

    expected_metrics = {
        "row_count": 638,
        "acceptance_rate": 0.6316614420062696,
        "overall_mse_ratio": 0.9211018929745592,
        "accepted_row_mse_ratio": 0.8605000982262854,
        "positive_distance_reduction_rate": 0.7146401985111662,
        "relative_distance_reduction_mean": 0.07480610639671968,
        "risk_brier_score": 0.2025145146622673,
        "risk_constant_brier_score": 0.24297551501533382,
        "output_candidate_sha256": "04dec9f03c82c4f4c97f3b7c9a8017f90741d001876ede872d53ec3a01d4ea05",
        "output_selected_scale_sha256": "967407ceab68cf55856cb9f94ff59abdd498afe8476e66470acf612d6b8e66bf",
    }
    for key, expected in expected_metrics.items():
        observed = record.get(key)
        if isinstance(expected, float):
            if float(observed) != expected:
                raise StageFError("canonical metric changed: {}".format(key))
        elif observed != expected:
            raise StageFError("canonical identity changed: {}".format(key))

    audit = stagee["audit"]
    return {
        "canonicalization": canonical,
        "outer_fold_nested_eligibility": {
            "all_six_folds_eligible": True,
            "fold_records": fold_support,
        },
        "fixed_recipe_outer_oof": {
            "recipe_id": rid,
            "eligibility": eligibility,
            "record": record,
        },
        "frozen_control_timesteps": {
            "25": {"all_pass": True},
            "50": {"all_pass": True},
        },
        "stagee_operational_family_audit_sha256": sha256_bytes(
            stable_json_bytes(audit["selected_operational_family"])
        ),
        "canonical_lock_ready": True,
    }


def build_summary(
    repository: Mapping[str, Any], stagee: Mapping[str, Any], worker: Mapping[str, Any]
) -> Mapping[str, Any]:
    lock = lock_canonical_recipe(stagee, worker)
    canonical = lock["canonicalization"]["canonical_recipe"]
    recommendation = {
        "backbone_id": "segment_target_rr64_feasible",
        "timesteps": [10, 25, 50],
        "timestep_policy": "joint_all_timesteps_no_cherry_pick",
        "t10_risk_adapter_only": True,
        "t10_descriptor_id": canonical["descriptor_id"],
        "t10_descriptor_dimension": DESCRIPTOR_DIMENSIONS[canonical["descriptor_id"]],
        "t10_risk_l2": canonical["risk_l2"],
        "t10_prevalence_centered_temperature": canonical["temperature"],
        "t10_direction_shrinkage": canonical["shrinkage"],
        "t10_adverse_risk_threshold": canonical["risk_threshold"],
        "t25_t50_risk_path": "frozen_stagec_resume1_legacy_control",
        "selection_role": "structural_canonicalization_of_six_fold_eligible_operational_family",
        "performance_metrics_used_for_canonicalization": False,
        "frozen_probe_access_authorized_by_this_report": False,
    }
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "primary_failure_locus": "t10_canonical_risk_repair_ready_for_frozen_probe_preregistration",
        "root_cause": (
            "phase314b_r259_stagef_t10_operational_family_and_structurally_canonicalized_"
            "recipe_pass_existing_nested_group_oof"
        ),
        "required_next_path": (
            "PREREGISTER_ONE_SHOT_FROZEN_PROBE_EVALUATION_OF_R259_T10_CANONICAL_RISK_REPAIR"
        ),
        "repository": dict(repository),
        "source_evidence": {
            "stagee_path": SOURCE_STAGEE,
            "stagee_file_sha256": SOURCE_STAGEE_SHA256,
            "stagee_self_sha256": SOURCE_STAGEE_SELF_SHA256,
            "stage_d_worker_path": SOURCE_WORKER,
            "stage_d_worker_file_sha256": SOURCE_WORKER_SHA256,
            "stage_d_worker_result_sha256": SOURCE_WORKER_RESULT_SHA256,
        },
        "lock": lock,
        "execution_counts": {
            "report_read_count": 2,
            "environment_probe_count": 0,
            "science_worker_count": 0,
            "cuda_operation_count": 0,
            "dataset_load_count": 0,
            "direction_fit_count": 0,
            "candidate_generation_count": 0,
            "risk_fit_count": 0,
            "recipe_evaluation_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": recommendation,
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
        raise StageFError("Stage-F summary schema/verdict changed")
    if payload.get("scientific_status") != "READY":
        raise StageFError("Stage-F canonical lock is not READY")
    lock = payload.get("lock")
    if not isinstance(lock, Mapping) or lock.get("canonical_lock_ready") is not True:
        raise StageFError("Stage-F canonical lock payload missing")
    canonical = lock.get("canonicalization")
    if not isinstance(canonical, Mapping) or canonical.get("canonical_recipe_id") != CANONICAL_RECIPE_ID:
        raise StageFError("Stage-F canonical recipe changed")
    if canonical.get("performance_metrics_used") is not False:
        raise StageFError("Stage-F canonicalization used performance metrics")
    recommendation = payload.get("train_only_recommendation")
    if not isinstance(recommendation, Mapping):
        raise StageFError("Stage-F READY report lacks recommendation")
    if recommendation.get("t10_descriptor_id") != "compact_v1":
        raise StageFError("Stage-F recommendation descriptor changed")
    if recommendation.get("t10_risk_l2") != 4.0:
        raise StageFError("Stage-F recommendation L2 changed")
    if recommendation.get("t10_prevalence_centered_temperature") != 1.0:
        raise StageFError("Stage-F recommendation temperature changed")
    if recommendation.get("t10_direction_shrinkage") != LOCKED_SHRINKAGE:
        raise StageFError("Stage-F recommendation shrinkage changed")
    if recommendation.get("t10_adverse_risk_threshold") != LOCKED_RISK_THRESHOLD:
        raise StageFError("Stage-F recommendation threshold changed")
    if payload.get("selected_configuration") is not None:
        raise StageFError("Stage-F selected a final configuration")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageFError("Stage-F re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageFError("Stage-F cumulative holdout count changed")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageFError("Stage-F accessed frozen probe")
    if payload.get("rerun_authorized") is not False:
        raise StageFError("Stage-F authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageFError("Stage-F crossed a forbidden boundary")
    counts = payload.get("execution_counts")
    if not isinstance(counts, Mapping) or counts.get("report_read_count") != 2:
        raise StageFError("Stage-F report-read count changed")
    if any(value != 0 for key, value in counts.items() if key != "report_read_count"):
        raise StageFError("Stage-F performed scientific execution")
    expected = _self_hash(payload, "summary_sha256")
    if payload.get("summary_sha256") != expected:
        raise StageFError("Stage-F summary self-hash changed")


def blocked_report(repository: Optional[Mapping[str, Any]], error: BaseException) -> Mapping[str, Any]:
    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "stagef_execution_contract",
        "root_cause": "phase314b_r259_stagef_operational_family_lock_execution_failed",
        "required_next_path": "AUDIT_STAGEF_INPUT_OR_CANONICALIZATION_CONTRACT",
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
