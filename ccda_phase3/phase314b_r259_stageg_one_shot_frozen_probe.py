"""Phase3.14b-r2.5.9 Stage G one-shot frozen-probe evaluation.

Stage F structurally canonicalized the stable t=10 operational family without
using performance metrics.  Stage G performs the first and only frozen-probe
access for the resulting joint three-timestep policy.

The scientific procedure is fixed before probe access:

* fit direction surrogates and adverse-risk models on objective-train only;
* risk training uses six-fold grouped OOF direction/candidate descriptors;
* fit one final direction model per timestep on all objective-train rows;
* evaluate the frozen probe once and jointly at timesteps 10, 25 and 50;
* t=10 uses the Stage-F canonical risk recipe;
* t=25 and t=50 use the frozen Stage-C legacy risk path;
* no refit, threshold change, timestep drop, recipe fallback or rerun is
  permitted after the worker starts.

Only bounded JSON evidence is persisted.  No checkpoint, weight, tensor, NPZ,
cache, image or video is written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import (
    phase314b_r259_staged_resume2_cable_state_schema_recovery as resume2,
)
from ccda_phase3 import (
    phase314b_r259_staged_resume3_variable_threshold_call_order_recovery as resume3,
)

PHASE = "Phase3.14b-r2.5.9 Stage G"
SCHEMA = "phase314b_r259_stageg_one_shot_frozen_probe_v1"
PROBE_SCHEMA = SCHEMA + "_environment_probe_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEF_IMPLEMENTATION = "aaffdd796772ef9fadbeb3bae670aa8bc0f9b7e2"
BASE_STAGEF_EVIDENCE = "b324b38d655b11e13138675c6cfb304f43f9fbba"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage G: evaluate canonical risk repair on frozen probe"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage G one-shot frozen-probe evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage G blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stageg_one_shot_frozen_probe.py"),
    ("A", "scripts/phase3_14b_r259_stageg_worker.py"),
    ("A", "scripts/phase3_14b_r259_stageg_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stageg_one_shot_frozen_probe.py"),
)

STAGEF_REPORT = (
    "reports/phase3_14b_r259_stagef_operational_family_canonical_lock_summary.json"
)
STAGEF_REPORT_SHA256 = (
    "d40930f5129f7c71b8d86c9533500c4b5f55a71feed10e72cf244537b28b4b39"
)
STAGEF_SELF_SHA256 = (
    "d9c754282722bb7f4c2312f8981ec935e2e9d74131a892b941ec827e21c13ca1"
)
STAGEU_CONTRACT = "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"
STAGEU_CONTRACT_SHA256 = (
    "c0bf2c477694cf1bda77f11c7ac407befb77d005afa8422e7e9bae2616714a8c"
)

PROBE_EVIDENCE = "reports/phase3_14b_r259_stageg_environment_probe_evidence.json"
WORKER_EVIDENCE = "reports/phase3_14b_r259_stageg_frozen_probe_worker_evidence.json"
SUCCESS_REPORT = "reports/phase3_14b_r259_stageg_one_shot_frozen_probe_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_stageg_one_shot_frozen_probe_blocked_summary.json"

LOCKED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
OUTER_FOLDS = 6
EXPECTED_OBJECTIVE_ROWS = 638
EXPECTED_OBJECTIVE_GROUPS = 126
EXPECTED_FROZEN_PROBE_ROWS = 126
EXPECTED_INTERNAL_SCALE_COUNT = 7
FROZEN_PROBE_NOISE_SEED_OFFSET = 9901

CANONICAL_RECIPE_ID = (
    "desc_compact_v1__l2_4.00__temp_1.00__shrink_1.00__risk_0.50"
)
CANONICAL_DESCRIPTOR = "compact_v1"
CANONICAL_T10_RISK_L2 = 4.0
LEGACY_CONTROL_RISK_L2 = 1.0
LOCKED_SHRINKAGE = 1.0
LOCKED_RISK_THRESHOLD = 0.5
LOCKED_TEMPERATURE = 1.0

EXPECTED_EXECUTION_COUNTS: Mapping[str, int] = {
    "objective_direction_fit_count": 21,
    "candidate_generation_count": 21,
    "objective_risk_fit_count": 3,
    "descriptor_build_count": 6,
    "internal_scale_attempt_count": 147,
    "frozen_probe_control_prediction_count": 1,
    "frozen_probe_joint_evaluation_count": 1,
}

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded",
    "selection_holdout_used_for_fit_or_selection",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "candidate_execution",
    "deformable_ravens_executed",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_persisted",
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "descriptor_tensor_persisted",
    "risk_probability_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageGError(RuntimeError):
    """Fail-closed Stage-G error."""


@dataclass(frozen=True)
class FrozenProbeSpec:
    timesteps: Tuple[int, ...] = LOCKED_TIMESTEPS
    frozen_probe_noise_seed_offset: int = FROZEN_PROBE_NOISE_SEED_OFFSET
    required_acceptance_rate: float = 0.50
    required_positive_reduction_rate: float = 0.50
    adverse_sse_reduction_fraction: float = 0.10
    group_cvar_fraction: float = 0.20

    def validate(self) -> None:
        if tuple(self.timesteps) != LOCKED_TIMESTEPS:
            raise StageGError("timestep population changed")
        if int(self.frozen_probe_noise_seed_offset) != FROZEN_PROBE_NOISE_SEED_OFFSET:
            raise StageGError("frozen-probe noise seed offset changed")
        if self.required_acceptance_rate != 0.50:
            raise StageGError("acceptance threshold changed")
        if self.required_positive_reduction_rate != 0.50:
            raise StageGError("positive-reduction threshold changed")
        if self.adverse_sse_reduction_fraction != 0.10:
            raise StageGError("adverse-SSE threshold changed")
        if self.group_cvar_fraction != 0.20:
            raise StageGError("group-CVaR fraction changed")


def _staged() -> Any:
    return resume3._base()


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


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageGError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageGError("write-once output exists: {}".format(target))
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


def promote_write_ahead(source: Path, destination: Path, validator: Any) -> str:
    source_path = Path(source)
    if not source_path.is_file():
        raise StageGError("write-ahead evidence is missing: {}".format(source_path))
    payload = source_path.read_bytes()
    validator(json.loads(payload.decode("utf-8")))
    atomic_write_once(Path(destination), payload)
    if Path(destination).read_bytes() != payload:
        raise StageGError("write-ahead promotion is not byte-exact")
    return sha256_bytes(payload)


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
            raise StageGError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def validate_stagef_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file() or sha256_file(report_path) != STAGEF_REPORT_SHA256:
        raise StageGError("Stage-F report SHA changed")
    payload = load_json(report_path)
    if payload.get("schema") != "phase314b_r259_stagef_operational_family_canonical_lock_v1":
        raise StageGError("Stage-F schema changed")
    if payload.get("execution_verdict") != "PASS" or payload.get("scientific_status") != "READY":
        raise StageGError("Stage-F is not READY")
    if payload.get("summary_sha256") != STAGEF_SELF_SHA256:
        raise StageGError("Stage-F self-hash changed")
    lock = payload.get("lock")
    if not isinstance(lock, Mapping) or lock.get("canonical_lock_ready") is not True:
        raise StageGError("Stage-F canonical lock changed")
    canonical = lock.get("canonicalization")
    if not isinstance(canonical, Mapping):
        raise StageGError("Stage-F canonicalization is missing")
    if canonical.get("canonical_recipe_id") != CANONICAL_RECIPE_ID:
        raise StageGError("canonical recipe changed")
    if canonical.get("performance_metrics_used") is not False:
        raise StageGError("canonicalization used performance metrics")
    recommendation = payload.get("train_only_recommendation")
    if not isinstance(recommendation, Mapping):
        raise StageGError("Stage-F recommendation is missing")
    expected = {
        "backbone_id": "segment_target_rr64_feasible",
        "timesteps": [10, 25, 50],
        "timestep_policy": "joint_all_timesteps_no_cherry_pick",
        "t10_descriptor_id": CANONICAL_DESCRIPTOR,
        "t10_descriptor_dimension": 17,
        "t10_risk_l2": CANONICAL_T10_RISK_L2,
        "t10_prevalence_centered_temperature": LOCKED_TEMPERATURE,
        "t10_direction_shrinkage": LOCKED_SHRINKAGE,
        "t10_adverse_risk_threshold": LOCKED_RISK_THRESHOLD,
        "t25_t50_risk_path": "frozen_stagec_resume1_legacy_control",
    }
    for key, value in expected.items():
        if recommendation.get(key) != value:
            raise StageGError("Stage-F recommendation changed: {}".format(key))
    if recommendation.get("frozen_probe_access_authorized_by_this_report") is not False:
        raise StageGError("Stage-F directly authorized frozen-probe access")
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_STAGEF_EVIDENCE),
        "Stage-F parent": (
            _git(repo, "rev-parse", BASE_STAGEF_EVIDENCE + "^"),
            BASE_STAGEF_IMPLEMENTATION,
        ),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise StageGError(
                "{} changed: expected={} actual={}".format(label, expected, actual)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageGError("Stage-G implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageGError("Stage-G implementation paths changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageGError("submodule worktree commit changed")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageGError("main worktree is dirty")
    if _git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageGError("submodule worktree is dirty")
    validate_stagef_report(repo / STAGEF_REPORT)
    if sha256_file(repo / STAGEU_CONTRACT) != STAGEU_CONTRACT_SHA256:
        raise StageGError("Stage-U contract SHA changed")
    for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageGError("Stage-G output already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_STAGEF_EVIDENCE,
        "stage_f_evidence": BASE_STAGEF_EVIDENCE,
        "stage_f_implementation": BASE_STAGEF_IMPLEMENTATION,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def make_environment_probe(root: Path) -> Mapping[str, Any]:
    source = resume3.make_probe_evidence(Path(root).resolve())
    environment = resume3.validate_probe_evidence(source)
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": os.getpid(),
        "environment": copy.deepcopy(dict(environment)),
        "portable_environment_audit": copy.deepcopy(
            source.get("portable_environment_audit", {})
        ),
        "required_operation_dry_run": copy.deepcopy(
            environment.get("required_operation_dry_run", {})
        ),
        "durable_write_ahead_evidence": True,
        "frozen_probe_accessed": False,
    }
    payload["probe_evidence_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_environment_probe(payload)
    return payload


def validate_environment_probe(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != PROBE_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageGError("environment-probe schema/verdict changed")
    if int(payload.get("process_id", -1)) <= 0:
        raise StageGError("environment-probe PID is invalid")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise StageGError("environment payload is missing")
    compatibility = environment.get("compatibility")
    if environment.get("compatibility_pass") is not True or not isinstance(compatibility, Mapping):
        raise StageGError("portable environment compatibility failed")
    if compatibility.get("required_operation_pass") is not True:
        raise StageGError("required-operation compatibility failed")
    dry_run = payload.get("required_operation_dry_run")
    if not isinstance(dry_run, Mapping) or dry_run.get("pass") is not True:
        raise StageGError("required-operation dry run failed")
    audit = payload.get("portable_environment_audit")
    if not isinstance(audit, Mapping) or audit.get("portable_contract_passed") is not True:
        raise StageGError("portable environment audit failed")
    if payload.get("durable_write_ahead_evidence") is not True:
        raise StageGError("environment probe is not durable")
    if payload.get("frozen_probe_accessed") is not False:
        raise StageGError("environment probe accessed frozen probe")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "probe_evidence_sha256"}
        )
    )
    if payload.get("probe_evidence_sha256") != expected:
        raise StageGError("environment-probe self-hash changed")
    return environment


def _risk_l2_for_timestep(timestep: int) -> float:
    return CANONICAL_T10_RISK_L2 if int(timestep) == 10 else LEGACY_CONTROL_RISK_L2


def _training_models(
    *, runtime: Mapping[str, Any], context: Mapping[str, Any]
) -> Tuple[Mapping[int, Mapping[str, Any]], Mapping[str, int], Mapping[int, Mapping[str, Any]]]:
    staged = _staged()
    stagex = staged.stagex
    stagef = runtime["stagef"]
    stageg = runtime["stageg"]
    definition = stagef.definition_by_id(stagex.LOCKED_BACKBONE)
    assignment = np.asarray(context["objective_fold_assignment"], dtype=np.int64)
    groups = np.asarray(context["objective_groups"]).astype(str)
    target = np.asarray(context["objective_target"], dtype=np.float32)
    if target.shape[0] != EXPECTED_OBJECTIVE_ROWS or len(set(groups.tolist())) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageGError("objective-train population changed")
    if sorted(set(assignment.tolist())) != list(range(OUTER_FOLDS)):
        raise StageGError("objective fold assignment changed")
    reconstruction_spec = runtime["stager"].StageRSpec()
    reconstruction_spec.validate()
    models: Dict[int, Mapping[str, Any]] = {}
    diagnostics: Dict[int, Mapping[str, Any]] = {}
    counts = dict(EXPECTED_EXECUTION_COUNTS)
    for key in counts:
        counts[key] = 0

    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(
            context["objective_control_predictions"][int(timestep)], dtype=np.float32
        )
        features = stagef.build_constraint_features(
            condition=context["objective_condition"],
            control=control,
            condition_name=context["objective_condition_name"],
            feature_mode=definition.feature_mode,
            context=context,
        )
        oracle = stagef.generate_projected_oracle_target(
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
        oof_direction = np.zeros(control.shape, dtype=np.float64)
        oof_candidate = control.copy()
        oof_scale = np.zeros(control.shape[0], dtype=np.float64)
        seen = np.zeros(control.shape[0], dtype=np.bool_)
        fold_records: List[Mapping[str, Any]] = []
        for fold in range(OUTER_FOLDS):
            test_mask = assignment == fold
            fit_mask = ~test_mask
            indices = np.flatnonzero(test_mask)
            fitted = stagex._fit_direction(
                runtime=runtime,
                context=context,
                timestep=int(timestep),
                fit_mask=fit_mask,
                predict_indices=indices,
                oracle=oracle,
                features=features,
            )
            counts["objective_direction_fit_count"] += 1
            generated = stagex._candidate_for_indices(
                runtime=runtime,
                context=context,
                timestep=int(timestep),
                indices=indices,
                direction=fitted["direction"],
                shrinkage=LOCKED_SHRINKAGE,
                reconstruction_spec=reconstruction_spec,
            )
            staged.kernel.require_clean_generation(
                generated, label="Stage-G objective OOF t{} fold{}".format(timestep, fold)
            )
            counts["candidate_generation_count"] += 1
            counts["internal_scale_attempt_count"] += EXPECTED_INTERNAL_SCALE_COUNT
            oof_direction[indices] = fitted["direction"]
            oof_candidate[indices] = generated["candidate"]
            oof_scale[indices] = generated["selected_scale"]
            seen[indices] = True
            fold_records.append(
                {
                    "fold": fold,
                    "row_count": int(indices.size),
                    "direction_model_identity": copy.deepcopy(fitted["model_identity"]),
                    "candidate_sha256": sha256_array(generated["candidate"]),
                    "selected_scale_sha256": sha256_array(generated["selected_scale"]),
                }
            )
        if not np.all(seen):
            raise StageGError("objective OOF coverage is incomplete")
        descriptor = resume2.build_risk_descriptors_48(
            CANONICAL_DESCRIPTOR,
            control,
            oof_direction,
            oof_candidate,
            oof_scale,
        )
        counts["descriptor_build_count"] += 1
        labels = stagex._risk_labels(control, oof_candidate, target)
        risk_fit_mask = oof_scale > 0.0
        risk_model = staged.fit_risk_model_checked(
            descriptor,
            labels,
            risk_fit_mask,
            _risk_l2_for_timestep(timestep),
        )
        counts["objective_risk_fit_count"] += 1
        full_mask = np.ones(control.shape[0], dtype=np.bool_)
        direction_model = stageg._fit_direction_model(
            definition=definition,
            features=features,
            control=control,
            oracle_target=oracle,
            condition_name=context["objective_condition_name"],
            fit_mask=full_mask,
            spec=runtime["stagef_spec"],
            permutation_seed=None,
        )
        counts["objective_direction_fit_count"] += 1
        models[int(timestep)] = {
            "direction_model": direction_model,
            "risk_model": risk_model,
            "definition": definition,
        }
        diagnostics[int(timestep)] = {
            "timestep": int(timestep),
            "risk_l2": _risk_l2_for_timestep(timestep),
            "descriptor_id": CANONICAL_DESCRIPTOR,
            "oof_descriptor_sha256": sha256_array(descriptor),
            "oof_candidate_sha256": sha256_array(oof_candidate),
            "oof_selected_scale_sha256": sha256_array(oof_scale),
            "adverse_label_rate": float(np.mean(labels)),
            "risk_fit_row_count": int(np.sum(risk_fit_mask)),
            "risk_prevalence": float(risk_model["prevalence"]),
            "fold_records": fold_records,
            "final_direction_model_identity": copy.deepcopy(direction_model["identity"]),
        }
    if counts["objective_direction_fit_count"] != 21:
        raise StageGError("direction-fit count changed")
    return models, counts, diagnostics


def _mark_frozen_probe_access(path: Path) -> None:
    atomic_write_once(
        Path(path),
        stable_json_bytes(
            {
                "phase": PHASE,
                "process_id": os.getpid(),
                "frozen_probe_access_begun": True,
                "rerun_authorized": False,
            }
        ),
    )


def run_frozen_probe_worker(
    *,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
    access_marker: Path,
    spec: Optional[FrozenProbeSpec] = None,
) -> Mapping[str, Any]:
    active = FrozenProbeSpec() if spec is None else spec
    active.validate()
    environment = validate_environment_probe(probe_payload)
    repo = Path(root).resolve()
    stagef_report = validate_stagef_report(repo / STAGEF_REPORT)
    stageu_contract = load_json(repo / STAGEU_CONTRACT)
    staged = _staged()
    stagex = staged.stagex
    modules = stagex._science_modules()
    runtime = dict(modules["stageo"]._prepare_runtime(repo, environment))
    runtime.update({**modules, "stageu_gate": stageu_contract["aligned_gate_lock"]})
    full_context = runtime["context"]
    objective_context = stagex._objective_only_context(full_context)
    runtime["context"] = objective_context
    models, counts, training_diagnostics = _training_models(
        runtime=runtime, context=objective_context
    )

    # This marker is written only after every fit is complete and immediately
    # before the first frozen-probe target slice.  A worker start or access
    # marker consumes the one-shot attempt; no rerun is authorized.
    _mark_frozen_probe_access(Path(access_marker))
    frozen_mask = np.asarray(full_context["frozen_probe_mask"], dtype=np.bool_)
    selection_mask = np.asarray(full_context["selection_holdout_mask"], dtype=np.bool_)
    objective_mask = np.asarray(full_context["objective_train_mask"], dtype=np.bool_)
    if np.any(frozen_mask & (selection_mask | objective_mask)):
        raise StageGError("frozen-probe split overlaps train/selection populations")
    if int(np.sum(frozen_mask)) != EXPECTED_FROZEN_PROBE_ROWS:
        raise StageGError("frozen-probe row population changed")
    probe_condition = np.asarray(full_context["condition"], dtype=np.float32)[frozen_mask]
    probe_target = np.asarray(full_context["target"], dtype=np.float32)[frozen_mask]
    probe_groups = np.asarray(full_context["groups"])[frozen_mask].astype(str)
    probe_condition_name = np.asarray(full_context["condition_name"])[frozen_mask].astype(str)
    stageb258 = runtime["stagef"].stageb258
    controls, control_sha = stageb258.stagec.one_step_predictions(
        model=full_context["model"],
        condition=probe_condition,
        target=probe_target,
        spec=full_context["stageb_spec"],
        condition_standardizer=full_context["condition_standardizer"],
        target_standardizer=full_context["target_standardizer"],
        noise_seed=(
            int(full_context["stageb_spec"].seed)
            + int(active.frozen_probe_noise_seed_offset)
        ),
    )
    counts["frozen_probe_control_prediction_count"] += 1
    reconstruction_spec = runtime["stager"].StageRSpec()
    reconstruction_spec.validate()
    stagex_spec = stagex.StageXSpec()
    stagex_spec.validate()
    stage_d_spec = staged.StageDSpec()
    stage_d_spec.validate()
    timestep_records: Dict[str, Mapping[str, Any]] = {}
    gate_totals = {
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
    }
    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(controls[int(timestep)], dtype=np.float32)
        model_bundle = models[int(timestep)]
        definition = model_bundle["definition"]
        features = runtime["stagef"].build_constraint_features(
            condition=probe_condition,
            control=control,
            condition_name=probe_condition_name,
            feature_mode=definition.feature_mode,
            context=full_context,
        )
        direction = runtime["stageg"]._predict_direction(
            definition=definition,
            model=model_bundle["direction_model"],
            features=features,
            control=control,
            condition_name=probe_condition_name,
        )
        generated = runtime["stagev"].reconstruct_locked_holdout_candidate(
            control=control,
            base_direction=np.asarray(direction, dtype=np.float64) * LOCKED_SHRINKAGE,
            context=full_context,
            runtime=runtime,
            spec=reconstruction_spec,
        )
        staged.kernel.require_clean_generation(
            generated, label="Stage-G frozen probe t{}".format(timestep)
        )
        counts["candidate_generation_count"] += 1
        counts["internal_scale_attempt_count"] += EXPECTED_INTERNAL_SCALE_COUNT
        descriptor = resume2.build_risk_descriptors_48(
            CANONICAL_DESCRIPTOR,
            control,
            np.asarray(direction, dtype=np.float64) * LOCKED_SHRINKAGE,
            generated["candidate"],
            generated["selected_scale"],
        )
        counts["descriptor_build_count"] += 1
        risk_probability = stagex.predict_risk(
            model_bundle["risk_model"], descriptor
        )
        prevalence = np.full(
            control.shape[0],
            float(model_bundle["risk_model"]["prevalence"]),
            dtype=np.float64,
        )
        record = stagex.evaluate_policy(
            control,
            generated["candidate"],
            generated["selected_scale"],
            risk_probability,
            probe_target,
            probe_groups,
            LOCKED_RISK_THRESHOLD,
            prevalence,
            stagex_spec,
        )
        baseline = stagex.evaluate_policy(
            control,
            generated["candidate"],
            generated["selected_scale"],
            np.zeros(control.shape[0], dtype=np.float64),
            probe_target,
            probe_groups,
            1.0,
            None,
            stagex_spec,
        )
        eligibility = staged.recipe_is_eligible(record, baseline, stage_d_spec)
        gate_counts = {
            key: int(generated[key]) for key in gate_totals
        }
        for key, value in gate_counts.items():
            gate_totals[key] += value
        timestep_records[str(timestep)] = {
            "timestep": int(timestep),
            "risk_path": (
                "stagef_canonical_t10" if int(timestep) == 10
                else "frozen_stagec_resume1_legacy_control"
            ),
            "descriptor_id": CANONICAL_DESCRIPTOR,
            "risk_l2": _risk_l2_for_timestep(timestep),
            "temperature": LOCKED_TEMPERATURE,
            "direction_shrinkage": LOCKED_SHRINKAGE,
            "risk_threshold": LOCKED_RISK_THRESHOLD,
            "record": record,
            "no_abstention_baseline": baseline,
            "eligibility": eligibility,
            "gate_counts": gate_counts,
            "direction_sha256": sha256_array(np.asarray(direction, dtype=np.float64)),
            "candidate_sha256": sha256_array(generated["candidate"]),
            "selected_scale_sha256": sha256_array(generated["selected_scale"]),
            "descriptor_sha256": sha256_array(descriptor),
            "risk_probability_sha256": sha256_array(risk_probability),
        }
    counts["frozen_probe_joint_evaluation_count"] += 1
    if counts != dict(EXPECTED_EXECUTION_COUNTS):
        raise StageGError("Stage-G execution counts changed: {!r}".format(counts))
    all_metric_eligibility_pass = all(
        timestep_records[str(timestep)]["eligibility"]["pass"] is True
        for timestep in LOCKED_TIMESTEPS
    )
    gates_pass = all(value == 0 for value in gate_totals.values())
    ready = bool(all_metric_eligibility_pass and gates_pass)
    selected = None
    if ready:
        selected = copy.deepcopy(stagef_report["train_only_recommendation"])
        selected.pop("frozen_probe_access_authorized_by_this_report", None)
        selected["stagef_frozen_probe_access_authorized"] = False
        selected["selection_role"] = "one_shot_frozen_probe_validated_joint_policy"
        selected["frozen_probe_evaluated_by_stageg"] = True
        selected["frozen_probe_validation_passed"] = True
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": (
            "phase314b_r259_stageg_canonical_joint_policy_transfers_to_one_shot_frozen_probe"
            if ready else
            "phase314b_r259_stageg_canonical_joint_policy_fails_one_shot_frozen_probe"
        ),
        "required_next_path": (
            "PREREGISTER_R259_FORMAL_BASELINE_TRAINING_WITH_FROZEN_CANONICAL_RISK_POLICY"
            if ready else
            "AUDIT_R259_FROZEN_PROBE_TRANSFER_FAILURE_WITHOUT_REACCESS_OR_RETUNING"
        ),
        "primary_failure_locus": (
            "canonical_joint_policy_frozen_probe_pass"
            if ready else "canonical_joint_policy_frozen_probe_failure"
        ),
        "process_id": os.getpid(),
        "repository_head": str(repository_head),
        "preregistration_contract": {
            "one_shot_frozen_probe": True,
            "joint_all_timesteps_no_cherry_pick": True,
            "canonical_recipe_id": CANONICAL_RECIPE_ID,
            "t10_risk_l2": CANONICAL_T10_RISK_L2,
            "t25_t50_risk_l2": LEGACY_CONTROL_RISK_L2,
            "descriptor_id": CANONICAL_DESCRIPTOR,
            "temperature": LOCKED_TEMPERATURE,
            "direction_shrinkage": LOCKED_SHRINKAGE,
            "risk_threshold": LOCKED_RISK_THRESHOLD,
            "frozen_probe_noise_seed_offset": active.frozen_probe_noise_seed_offset,
            "fallback_forbidden": True,
            "post_probe_refit_forbidden": True,
            "post_probe_threshold_change_forbidden": True,
            "rerun_forbidden": True,
        },
        "population": {
            "objective_rows": EXPECTED_OBJECTIVE_ROWS,
            "objective_groups": EXPECTED_OBJECTIVE_GROUPS,
            "frozen_probe_rows": int(probe_target.shape[0]),
            "frozen_probe_groups": int(len(set(probe_groups.tolist()))),
            "frozen_probe_mask_sha256": sha256_array(frozen_mask),
            "frozen_probe_target_sha256": sha256_array(probe_target),
            "frozen_probe_condition_sha256": sha256_array(probe_condition),
        },
        "control_prediction_sha256": control_sha,
        "training_diagnostics": training_diagnostics,
        "frozen_probe_timestep_records": timestep_records,
        "all_timestep_metric_eligibility_pass": all_metric_eligibility_pass,
        "all_timesteps_pass": ready,
        "procedure_gate_totals": gate_totals,
        "execution_counts": counts,
        "selected_configuration": selected,
        "train_only_recommendation": selected,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_evidence(payload)
    return payload


def validate_worker_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageGError("worker schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageGError("worker scientific status changed")
    contract = payload.get("preregistration_contract")
    if not isinstance(contract, Mapping):
        raise StageGError("worker preregistration contract missing")
    required_true = (
        "one_shot_frozen_probe",
        "joint_all_timesteps_no_cherry_pick",
        "fallback_forbidden",
        "post_probe_refit_forbidden",
        "post_probe_threshold_change_forbidden",
        "rerun_forbidden",
    )
    if any(contract.get(key) is not True for key in required_true):
        raise StageGError("worker preregistration boundary changed")
    if payload.get("execution_counts") != dict(EXPECTED_EXECUTION_COUNTS):
        raise StageGError("worker execution counts changed")
    records = payload.get("frozen_probe_timestep_records")
    if not isinstance(records, Mapping) or sorted(records) != ["10", "25", "50"]:
        raise StageGError("worker timestep record population changed")
    all_pass = all(records[str(t)]["eligibility"]["pass"] is True for t in LOCKED_TIMESTEPS)
    gate_totals = payload.get("procedure_gate_totals")
    if not isinstance(gate_totals, Mapping):
        raise StageGError("worker gate totals missing")
    ready = bool(all_pass and all(int(value) == 0 for value in gate_totals.values()))
    if (payload.get("scientific_status") == "READY") != ready:
        raise StageGError("worker READY classification changed")
    if payload.get("all_timestep_metric_eligibility_pass") is not all_pass:
        raise StageGError("worker metric-eligibility classification changed")
    if payload.get("all_timesteps_pass") is not ready:
        raise StageGError("worker joint all-timestep classification changed")
    if ready:
        if not isinstance(payload.get("selected_configuration"), Mapping):
            raise StageGError("READY worker lacks selected configuration")
        if not isinstance(payload.get("train_only_recommendation"), Mapping):
            raise StageGError("READY worker lacks recommendation")
    else:
        if payload.get("selected_configuration") is not None:
            raise StageGError("BLOCKED worker selected a configuration")
        if payload.get("train_only_recommendation") is not None:
            raise StageGError("BLOCKED worker retained a recommendation")
    if payload.get("selection_holdout_evaluation_count_added") != 0:
        raise StageGError("worker re-accessed selection holdout")
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageGError("worker selection-holdout count changed")
    if payload.get("frozen_probe_evaluation_count_added") != 1:
        raise StageGError("worker frozen-probe count changed")
    if payload.get("cumulative_frozen_probe_evaluation_count") != 1:
        raise StageGError("worker cumulative frozen-probe count changed")
    if payload.get("frozen_probe_accessed") is not True:
        raise StageGError("worker did not record frozen-probe access")
    if payload.get("rerun_authorized") is not False:
        raise StageGError("worker authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageGError("worker crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "worker_result_sha256"}
        )
    )
    if payload.get("worker_result_sha256") != expected:
        raise StageGError("worker self-hash changed")


def build_summary(
    *, repository: Mapping[str, Any], probe_path: Path, worker_path: Path
) -> Mapping[str, Any]:
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    validate_environment_probe(probe)
    validate_worker_evidence(worker)
    if int(probe["process_id"]) == int(worker["process_id"]):
        raise StageGError("environment probe and science worker PIDs are not distinct")
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": copy.deepcopy(dict(repository)),
        "preregistration_contract": copy.deepcopy(worker["preregistration_contract"]),
        "population": copy.deepcopy(worker["population"]),
        "control_prediction_sha256": worker["control_prediction_sha256"],
        "training_diagnostics": copy.deepcopy(worker["training_diagnostics"]),
        "frozen_probe_timestep_records": copy.deepcopy(
            worker["frozen_probe_timestep_records"]
        ),
        "all_timesteps_pass": worker["all_timesteps_pass"],
        "procedure_gate_totals": copy.deepcopy(worker["procedure_gate_totals"]),
        "execution_counts": copy.deepcopy(worker["execution_counts"]),
        "process_topology": {
            "environment_probe_count": 1,
            "cold_science_worker_count": 1,
            "probe_process_id": int(probe["process_id"]),
            "worker_process_id": int(worker["process_id"]),
            "processes_distinct": True,
        },
        "durable_evidence_protocol": {
            "probe_path": PROBE_EVIDENCE,
            "probe_file_sha256": sha256_file(probe_path),
            "worker_path": WORKER_EVIDENCE,
            "worker_file_sha256": sha256_file(worker_path),
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_persisted_before_controller_summary": True,
            "external_write_ahead_outside_git_worktree": True,
            "repository_promotion_byte_exact": True,
            "controller_recomputed_science": False,
        },
        "selected_configuration": copy.deepcopy(worker["selected_configuration"]),
        "train_only_recommendation": copy.deepcopy(worker["train_only_recommendation"]),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageGError("summary schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageGError("summary scientific status changed")
    records = payload.get("frozen_probe_timestep_records")
    if not isinstance(records, Mapping) or sorted(records) != ["10", "25", "50"]:
        raise StageGError("summary timestep records changed")
    ready = payload.get("scientific_status") == "READY"
    if ready != bool(payload.get("all_timesteps_pass")):
        raise StageGError("summary joint result changed")
    if ready and not isinstance(payload.get("selected_configuration"), Mapping):
        raise StageGError("READY summary lacks selected configuration")
    if not ready and payload.get("selected_configuration") is not None:
        raise StageGError("BLOCKED summary selected a configuration")
    if payload.get("frozen_probe_evaluation_count_added") != 1:
        raise StageGError("summary frozen-probe count changed")
    if payload.get("cumulative_frozen_probe_evaluation_count") != 1:
        raise StageGError("summary cumulative frozen-probe count changed")
    if payload.get("frozen_probe_accessed") is not True:
        raise StageGError("summary did not record frozen-probe access")
    if payload.get("rerun_authorized") is not False:
        raise StageGError("summary authorized rerun")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageGError("summary crossed a forbidden boundary")
    expected = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if payload.get("summary_sha256") != expected:
        raise StageGError("summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    probe_path: Path,
    worker_path: Path,
    worker_started_path: Path,
    access_marker_path: Path,
) -> Mapping[str, Any]:
    worker_started = Path(worker_started_path).is_file()
    access_begun = Path(access_marker_path).is_file()
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageg_one_shot_frozen_probe_execution_failed",
        "required_next_path": (
            "FINALIZE_R259_STAGEG_DURABLE_EVIDENCE_WITHOUT_SCIENCE_REEXECUTION"
            if Path(worker_path).is_file()
            else "AUDIT_R259_STAGEG_ONE_SHOT_EXECUTION_FAILURE_WITHOUT_REACCESS"
        ),
        "primary_failure_locus": "stageg_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "environment_probe_evidence_present": Path(probe_path).is_file(),
        "worker_started": worker_started,
        "frozen_probe_access_begun": access_begun,
        "worker_evidence_present": Path(worker_path).is_file(),
        "one_shot_attempt_consumed": worker_started,
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 1 if access_begun else 0,
        "cumulative_frozen_probe_evaluation_count": 1 if access_begun else 0,
        "frozen_probe_accessed": access_begun,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
