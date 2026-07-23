"""Phase3.14b-r2.5.8 Stage V locked selection-holdout evaluation.

This stage evaluates exactly one pre-registered frontier:
``segment_target_rr64_feasible`` at timesteps 10/25/50.  The model is fit only
on objective-train.  The selection holdout target is guarded until the
candidate and selected-scale arrays have been finalized.  No backbone,
timestep, scale, threshold, or gate is selected after holdout access.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.8 Stage V"
SCHEMA = "phase314b_r258_stagev_locked_selection_holdout_evaluation_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_STAGEU_IMPLEMENTATION_COMMIT = "67c8ffc53dc38557b6f081bf9e0e958a3c53c444"
BASE_STAGEU_EVIDENCE_COMMIT = "959a4a77cbd3af40e0458d8c4dcc49a404246f0f"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_SUMMARY = "reports/phase3_14b_r258_stageu_candidate_frontier_lock_summary.json"
BASE_CONTRACT = "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"
EXPECTED_BASE_SUMMARY_SHA256 = "67225b09b5ecbcb75e9c9024a739b389c891a94f00735a409cb87f7f29b8663d"
EXPECTED_BASE_CONTRACT_SHA256 = "c0bf2c477694cf1bda77f11c7ac407befb77d005afa8422e7e9bae2616714a8c"
EXPECTED_CONTRACT_PAYLOAD_SHA256 = "ab2011731f9cd6c560d7721673aa33d09ebab05fa17fbf9a1d64cb35f1cce985"
EXPECTED_LOCKED_CELLS_SHA256 = "a2367bd378b9ddba84f839b0409cc658541b69ba1a2aa1c9bd4ce9c1d2063bd1"

SUCCESS_REPORT = "reports/phase3_14b_r258_stagev_locked_selection_holdout_evaluation_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r258_stagev_locked_selection_holdout_evaluation_blocked_summary.json"
IMPLEMENTATION_SUBJECT = "Phase3.14b-r2.5.8 Stage V: evaluate locked frontier on selection holdout once"
EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage V locked holdout evidence"
BLOCKED_EVIDENCE_SUBJECT = "Record Phase3.14b-r2.5.8 Stage V blocked evidence"
IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stagev_locked_selection_holdout_evaluation.py"),
    ("A", "scripts/phase3_14b_r258_stagev_worker.py"),
    ("A", "scripts/phase3_14b_r258_stagev_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stagev_locked_selection_holdout_evaluation.py"),
)

LOCKED_BACKBONE = "segment_target_rr64_feasible"
LOCKED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
EXPECTED_OBJECTIVE_ROWS = 638
EXPECTED_OBJECTIVE_GROUPS = 126
EXPECTED_HOLDOUT_ROWS = 236
EXPECTED_HOLDOUT_GROUPS = 42
EXPECTED_FROZEN_PROBE_ROWS = 126
EXPECTED_EXTERNAL_MULTIPLIER = 0.25
EXPECTED_TOLERANCE_FACTOR = 5.0
EXPECTED_INTERNAL_SCALE_COUNT = 7

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

FROZEN_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_stageu_candidate_frontier_lock.py": "a4455727e20ab96658686dc42eb0ab35b7bddadef8e9e99648d2c57371bae657",
    "ccda_phase3/phase314b_r258_staget_aligned_gate_candidate_matrix.py": "e59961e9312666e759b64adae7c3ae776ee701bccb902e165c5cecd85704ec8c",
    "ccda_phase3/phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit.py": "9f8da64be049bb9cd9a61e229e4dc20e9a91cccc55261f43511a76f600fb4ed0",
    "ccda_phase3/phase314b_r258_stageh_candidate_descriptor_identifiability.py": "5312332c8f77c7ff1c1735f60467ddd66810246188a0494940e7fc6bca9c0d78",
    "ccda_phase3/phase314b_r258_stageg_joint_direction_feasibility.py": "c61e74c5436e936a70be3deccc9c9d0127883c98894b00d2cd65373742d1d697",
    "ccda_phase3/phase314b_r258_stages_resume3_cold_worker_oof_confirmation.py": "f775f3e0427df383c11de3629517505791fa0b674a3be124ed65ef02e3e27a8c",
}

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "frozen_probe_accessed", "formal_training_run", "reverse_sampling_run",
    "idm_run", "candidate_execution", "deformable_ravens_executed", "phase4",
    "cps", "checkpoint_saved", "weights_persisted", "surrogate_weights_persisted",
    "prediction_tensor_persisted", "direction_tensor_persisted",
    "candidate_tensor_persisted", "predicate_tensor_persisted",
    "callback_event_persisted", "npz_saved", "cache_saved", "image_saved",
    "video_saved",
)

class StageVError(RuntimeError):
    """Fail-closed Stage-V execution-contract error."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = stable_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageVError(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageVError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageVError(f"{label} is not a sequence")
    return value


def require_false(payload: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if payload.get(key) is not False:
            raise StageVError(f"{label} boundary changed: {key}={payload.get(key)!r}")


def validate_environment_variables() -> Mapping[str, str]:
    observed = {key: os.environ.get(key) for key in EXPECTED_ENV}
    if observed != dict(EXPECTED_ENV):
        raise StageVError(f"deterministic environment changed: {observed!r}")
    return dict(EXPECTED_ENV)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise StageVError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def status_paths(root: Path) -> Tuple[str, ...]:
    output = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return tuple(line for line in output.splitlines() if line.strip())


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "show", "--format=", "--name-status", commit)
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        records.append((parts[0], parts[-1]))
    return tuple(sorted(records))


def validate_stageu_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    if contract.get("phase") != "Phase3.14b-r2.5.8 Stage U":
        raise StageVError("Stage-U phase changed")
    if contract.get("schema") != "phase314b_r258_stageu_candidate_frontier_contract_v1":
        raise StageVError("Stage-U contract schema changed")
    if contract.get("contract_payload_sha256") != EXPECTED_CONTRACT_PAYLOAD_SHA256:
        raise StageVError("Stage-U contract payload SHA changed")
    payload = dict(contract)
    payload.pop("contract_payload_sha256", None)
    if sha256_bytes(stable_json_bytes(payload)) != EXPECTED_CONTRACT_PAYLOAD_SHA256:
        raise StageVError("Stage-U contract payload is not self-consistent")
    frontier = _mapping(contract.get("frontier_lock"), "frontier lock")
    exact_frontier = {
        "backbone_id": LOCKED_BACKBONE,
        "timesteps": list(LOCKED_TIMESTEPS),
        "timestep_policy": "joint_all_timesteps_no_cherry_pick",
        "backbone_search_reopened": False,
        "timestep_search_reopened": False,
        "fallback_backbone_allowed": False,
        "fallback_timestep_allowed": False,
        "frontier_is_unique": True,
        "all_locked_timesteps_fidelity_eligible": True,
        "locked_cells_sha256": EXPECTED_LOCKED_CELLS_SHA256,
    }
    for key, expected in exact_frontier.items():
        if frontier.get(key) != expected:
            raise StageVError(f"Stage-U frontier changed: {key}")
    cells = _sequence(frontier.get("locked_cells"), "locked cells")
    if sha256_bytes(stable_json_bytes(cells)) != EXPECTED_LOCKED_CELLS_SHA256:
        raise StageVError("locked-cell payload changed")
    policy = _mapping(contract.get("next_stage_holdout_policy"), "holdout policy")
    required_policy = {
        "evaluation_count": 1,
        "population": "locked_backbone_all_three_timesteps",
        "backbone_id": LOCKED_BACKBONE,
        "timesteps": list(LOCKED_TIMESTEPS),
        "all_timesteps_must_pass": True,
        "post_holdout_backbone_selection_forbidden": True,
        "post_holdout_timestep_selection_forbidden": True,
        "threshold_changes_after_holdout_forbidden": True,
        "rerun_after_failure_forbidden": True,
        "aggregate_metrics_are_diagnostic_only": True,
        "frozen_probe_remains_closed": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise StageVError(f"Stage-U holdout policy changed: {key}")
    criteria = _mapping(policy.get("per_timestep_pass_criteria"), "pass criteria")
    if criteria != {
        "mechanism_eligible": True,
        "overall_mse_ratio_strictly_less_than": 1.0,
        "accepted_row_mse_ratio_strictly_less_than": 1.0,
        "positive_distance_reduction_rate_strictly_greater_than": 0.5,
        "relative_distance_reduction_mean_strictly_greater_than": 0.0,
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
    }:
        raise StageVError("Stage-U per-timestep criteria changed")
    gate = _mapping(contract.get("aligned_gate_lock"), "aligned gate lock")
    if gate.get("tolerance_factor") != 5.0 or gate.get("segment_tolerance") != 2.5e-6:
        raise StageVError("aligned-gate tolerance changed")
    if gate.get("position_shape") != [4, 23]:
        raise StageVError("aligned-gate shape changed")
    if gate.get("deployment_authorized") is not False or gate.get("aligned_gate_written_to_stagee") is not False:
        raise StageVError("aligned gate was deployed")
    require_false(contract, ("selection_holdout_evaluated", "selection_holdout_used_for_fit_or_selection", *FALSE_BOUNDARIES), "Stage-U contract")
    return {"frontier": frontier, "policy": policy, "gate": gate}


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageVError("Stage V requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_STAGEU_EVIDENCE_COMMIT:
        raise StageVError("Stage-V implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageVError("Stage-V implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageVError("Stage-V implementation paths changed")
    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageVError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageVError("DeformableRavens commit changed")
    if status_paths(repo) or status_paths(submodule):
        raise StageVError("repository or submodule worktree is dirty")
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageVError(f"frozen source changed: {relative}")
    for relative, expected in ((BASE_SUMMARY, EXPECTED_BASE_SUMMARY_SHA256), (BASE_CONTRACT, EXPECTED_BASE_CONTRACT_SHA256)):
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise StageVError(f"Stage-U evidence changed: {relative}")
    if (repo / SUCCESS_REPORT).exists() or (repo / BLOCKED_REPORT).exists():
        raise StageVError("Stage-V output already exists; rerun is forbidden")
    contract = load_json(repo / BASE_CONTRACT)
    validated = validate_stageu_contract(contract)
    return {"root": str(repo), "head": head, "parent": BASE_STAGEU_EVIDENCE_COMMIT, "origin_experiment1": EXPECTED_REMOTE, "submodule_commit": EXPECTED_SUBMODULE, "stageu_contract": contract, "stageu_validated": validated}


class HoldoutTargetGuard(Mapping[str, Any]):
    """Context proxy that forbids holdout-target access until explicitly unlocked."""
    def __init__(self, source: Mapping[str, Any]):
        self._source = source
        self._unlocked = False
        self.access_count = 0
    def unlock(self) -> None:
        if self._unlocked:
            raise StageVError("holdout target guard unlocked twice")
        self._unlocked = True
    def __getitem__(self, key: str) -> Any:
        if key == "holdout_target":
            if not self._unlocked:
                raise StageVError("holdout target accessed before candidate finalization")
            self.access_count += 1
        return self._source[key]
    def __iter__(self) -> Iterator[str]: return iter(self._source)
    def __len__(self) -> int: return len(self._source)
    def get(self, key: str, default: Any = None) -> Any:
        try: return self[key]
        except KeyError: return default


def _science_modules() -> Mapping[str, Any]:
    from ccda_phase3 import phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo
    from ccda_phase3 import phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow as stageq
    from ccda_phase3 import phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit as stager
    from ccda_phase3 import phase314b_r258_stageh_candidate_descriptor_identifiability as stageh
    from ccda_phase3 import phase314b_r258_stageg_joint_direction_feasibility as stageg
    from ccda_phase3 import phase314b_r258_staget_aligned_gate_candidate_matrix as staget
    from ccda_phase3 import phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a
    from ccda_phase3 import phase314b_r258_stages_resume3_cold_worker_oof_confirmation as stages_resume3
    return {"stageo": stageo, "stageq": stageq, "stager": stager, "stageh": stageh, "stageg": stageg, "staget": staget, "resume2a": resume2a, "stages_resume3": stages_resume3}


def validate_population(context: Mapping[str, Any]) -> Mapping[str, int]:
    objective_groups = np.asarray(context["objective_groups"]).astype(str)
    holdout_groups = np.asarray(context["holdout_groups"]).astype(str)
    if objective_groups.shape[0] != EXPECTED_OBJECTIVE_ROWS or len(set(objective_groups.tolist())) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageVError("objective-train population changed")
    if holdout_groups.shape[0] != EXPECTED_HOLDOUT_ROWS or len(set(holdout_groups.tolist())) != EXPECTED_HOLDOUT_GROUPS:
        raise StageVError("selection-holdout population changed")
    frozen = np.asarray(context["frozen_probe_mask"], dtype=np.bool_)
    if int(np.count_nonzero(frozen)) != EXPECTED_FROZEN_PROBE_ROWS:
        raise StageVError("frozen-probe population changed")
    return {"objective_rows": EXPECTED_OBJECTIVE_ROWS, "objective_groups": EXPECTED_OBJECTIVE_GROUPS, "holdout_rows": EXPECTED_HOLDOUT_ROWS, "holdout_groups": EXPECTED_HOLDOUT_GROUPS, "frozen_probe_rows": EXPECTED_FROZEN_PROBE_ROWS}


def fit_locked_direction_twice(*, timestep: int, context: Mapping[str, Any], runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    stagef, stageg = runtime["stagef"], runtime["stageg"]
    definition = stagef.definition_by_id(LOCKED_BACKBONE)
    if definition.role != "selectable" or definition.feature_mode != "full_segment_constraint":
        raise StageVError("locked backbone definition changed")
    train_control = np.asarray(context["objective_control_predictions"][int(timestep)], dtype=np.float32)
    holdout_control = np.asarray(context["control_predictions"][int(timestep)], dtype=np.float32)
    train_target = np.asarray(context["objective_target"], dtype=np.float32)
    train_oracle = stagef.generate_projected_oracle_target(
        control=train_control, target=train_target, groups=context["objective_groups"],
        condition_name=context["objective_condition_name"], timestep=int(timestep),
        context=context, direction_spec=runtime["direction_spec"],
        integrator_spec=runtime["integrator_spec"], spec=runtime["stagef_spec"],
    )
    train_features = stagef.build_constraint_features(
        condition=context["objective_condition"], control=train_control,
        condition_name=context["objective_condition_name"],
        feature_mode=definition.feature_mode, context=context,
    )
    holdout_features = stagef.build_constraint_features(
        condition=context["holdout_condition"], control=holdout_control,
        condition_name=context["holdout_condition_name"],
        feature_mode=definition.feature_mode, context=context,
    )
    fit_mask = np.ones(train_control.shape[0], dtype=np.bool_)
    if definition.fit_population == "oracle_feasible":
        fit_mask &= np.asarray(train_oracle["feasible_mask"], dtype=np.bool_)
    outputs = []
    for _ in range(2):
        model = stageg._fit_direction_model(
            definition=definition, features=train_features, control=train_control,
            oracle_target=train_oracle, condition_name=context["objective_condition_name"],
            fit_mask=fit_mask, spec=runtime["stagef_spec"], permutation_seed=None,
        )
        prediction = stageg._predict_direction(
            definition=definition, model=model, features=holdout_features,
            control=holdout_control, condition_name=context["holdout_condition_name"],
        )
        outputs.append({"model_identity": copy.deepcopy(model["identity"]), "prediction": np.asarray(prediction, dtype=np.float64), "prediction_sha256": sha256_array(prediction)})
    if stable_json_bytes(outputs[0]["model_identity"]) != stable_json_bytes(outputs[1]["model_identity"]):
        raise StageVError(f"full-fit model identity is not exact at t={timestep}")
    if outputs[0]["prediction_sha256"] != outputs[1]["prediction_sha256"]:
        raise StageVError(f"holdout direction prediction is not exact at t={timestep}")
    return {"control": holdout_control, "direction": outputs[0]["prediction"], "feature_sha256": sha256_array(holdout_features), "prediction_sha256": outputs[0]["prediction_sha256"], "model_identity": outputs[0]["model_identity"], "fit_rows": int(np.count_nonzero(fit_mask)), "full_fit_repeat_count": 2}


def reconstruct_locked_holdout_candidate(*, control: np.ndarray, base_direction: np.ndarray, context: Mapping[str, Any], runtime: Mapping[str, Any], spec: Any) -> Mapping[str, Any]:
    """Target-independent aligned-gate candidate reconstruction for holdout."""
    stageq, stager, stagel = runtime["stageq"], runtime["stager"], runtime["stagel"]
    stagee = stagel.stagee258
    stageb, staged, stagef = stagee.stageb, stagee.staged, runtime["stagef"]
    integrator_spec = runtime["integrator_spec"]
    definition = stagef.fixed_integrator_definition(); definition.validate()
    control_raw = np.asarray(control, dtype=np.float32)
    direction = np.asarray(base_direction, dtype=np.float64)
    if control_raw.shape != direction.shape:
        raise StageVError("holdout control/direction shape changed")
    if float(spec.external_multiplier) != EXPECTED_EXTERNAL_MULTIPLIER or float(spec.reconstruction_tolerance_factor) != EXPECTED_TOLERANCE_FACTOR:
        raise StageVError("locked scale/tolerance changed")
    proposed_direction = direction * float(spec.external_multiplier)
    expected_order = tuple(float(x) for x in stageq._expected_internal_order(runtime))
    if len(expected_order) != EXPECTED_INTERNAL_SCALE_COUNT:
        raise StageVError("internal-scale population changed")
    bounds = stagee.segment_bounds(definition=definition, context=context)
    upper, lower = np.asarray(bounds["upper"], dtype=np.float64), np.asarray(bounds["lower"], dtype=np.float64)
    reference = context["stage_d_contract"].reference
    contract = stageq.tolerant_upper_contract(
        upper_bound=upper, center_log=np.asarray(reference.center_log, dtype=np.float64),
        scale_log=np.asarray(reference.scale_log, dtype=np.float64),
        segment_tolerance=float(integrator_spec.segment_tolerance),
        tolerance_factor=float(spec.reconstruction_tolerance_factor),
    )
    selected = np.zeros(control_raw.shape[0], dtype=np.bool_)
    selected_scale = np.zeros(control_raw.shape[0], dtype=np.float64)
    output = control_raw.copy()
    mismatch_count = aligned_upper_failures = strict_pass_aligned_fail = 0
    for scale in expected_order:
        raw_proposal = (control_raw.astype(np.float64) + float(scale) * proposed_direction).astype(np.float32)
        reconstruction = stagee.reconstruct_segment_vectors(
            proposed=raw_proposal, control=control_raw, lower=lower, upper=upper,
            coordinate_abs_max=float(context["historical_geometry"].coordinate_abs_max),
            epsilon=float(integrator_spec.segment_tolerance),
        )
        candidate = np.asarray(reconstruction["candidate"], dtype=np.float32)
        bound_pass = np.asarray(reconstruction["bound_pass"], dtype=np.bool_)
        coordinate_possible = np.asarray(reconstruction["coordinate_possible"], dtype=np.bool_)
        metrics = stagee.observable_row_metrics(
            candidate=candidate, raw_proposal=raw_proposal, control=control_raw,
            direction=proposed_direction, definition=definition, context=context,
            spec=integrator_spec, bound_pass=bound_pass,
            coordinate_possible=coordinate_possible, include_topology=False,
        )
        lengths = stageb.segment_lengths(candidate).astype(np.float64)
        scores = staged.segment_scores(candidate, reference)
        aligned = stageq.aligned_upper_masks(lengths=lengths, z_scores=np.asarray(scores["z"], dtype=np.float64), contract=contract)
        aligned_pass = np.asarray(aligned["length_row_pass"], dtype=np.bool_)
        strict_pass = np.asarray(metrics["upper_pass"], dtype=np.bool_)
        mismatch_count += int(aligned["element_mismatch_count"])
        aligned_upper_failures += int(np.count_nonzero(~np.asarray(aligned["length_element_pass"], dtype=np.bool_)))
        strict_pass_aligned_fail += int(np.count_nonzero(strict_pass & ~aligned_pass))
        masks = stageq._predicate_masks(metrics=metrics, aligned_upper_pass=aligned_pass, bound_pass=bound_pass, coordinate_possible=coordinate_possible)
        active = ~selected
        fast = active.copy()
        for name in stager.PREDICATE_ORDER[:-1]:
            fast &= np.asarray(masks[name], dtype=np.bool_)
        topology = np.zeros(control_raw.shape[0], dtype=np.bool_)
        checked = np.flatnonzero(fast)
        if checked.size:
            topology[checked] = stageb.physical_validity(candidate[checked, None], context["historical_geometry"])["topology"][:, 0]
        sequential = stageq.sequential_attempt_summary(active=active, predicate_masks=masks, topology_pass=topology)
        choose = np.asarray(sequential["accepted_mask"], dtype=np.bool_)
        if not np.array_equal(choose, fast & topology):
            raise StageVError("acceptance decomposition changed")
        output[choose], selected_scale[choose], selected[choose] = candidate[choose], float(scale), True
    observation = runtime["staget"]._gate_contract_observation(
        contract=contract, upper_bound=upper,
        segment_tolerance=float(integrator_spec.segment_tolerance),
        tolerance_factor=float(spec.reconstruction_tolerance_factor),
        sha256_array=stager.sha256_array,
    )
    gate_lock = runtime["stageu_gate"]
    exact_gate = {
        "contract_sha256": gate_lock["inner_gate_contract_sha256"],
        "tolerance_factor": gate_lock["tolerance_factor"],
        "segment_tolerance": gate_lock["segment_tolerance"],
        "length_tolerance": gate_lock["length_tolerance"],
        "position_shape": gate_lock["position_shape"],
        "upper_bound_sha256": gate_lock["upper_bound_sha256"],
        "length_upper_sha256": gate_lock["length_upper_sha256"],
        "position_z_threshold_sha256": gate_lock["position_z_threshold_sha256"],
    }
    for key, expected in exact_gate.items():
        if observation.get(key) != expected:
            raise StageVError(f"aligned-gate identity changed: {key}")
    return {
        "candidate": output, "selected_scale": selected_scale,
        "candidate_sha256": stager.sha256_array(output),
        "selected_scale_sha256": stager.sha256_array(selected_scale),
        "selected_scale_histogram": stager.selected_scale_histogram(selected_scale),
        "acceptance_rate": float(np.mean(selected)),
        "length_log_z_element_mismatch_count": mismatch_count,
        "aligned_upper_element_failure_count": aligned_upper_failures,
        "strict_pass_aligned_fail_row_count": strict_pass_aligned_fail,
        "internal_scale_attempt_order": list(expected_order),
        "internal_scale_attempt_count": len(expected_order),
        "gate_contract_observation": observation,
        "candidate_finalized_without_holdout_target": True,
    }


def per_timestep_checks(metrics: Mapping[str, Any], candidate: Mapping[str, Any]) -> Mapping[str, bool]:
    accepted = _mapping(metrics.get("accepted_rows"), "accepted rows")
    relative = _mapping(accepted.get("relative_distance_reduction"), "relative reduction")
    checks = {
        "mechanism_eligible": metrics.get("mechanism_eligible") is True,
        "overall_mse_ratio": float(metrics["overall_mse_ratio"]) < 1.0,
        "accepted_row_mse_ratio": float(accepted["mse_ratio"]) < 1.0,
        "positive_distance_reduction_rate": float(accepted["positive_distance_reduction_rate"]) > 0.5,
        "relative_distance_reduction_mean": float(relative["mean"]) > 0.0,
        "length_log_z_element_mismatch_count": int(candidate["length_log_z_element_mismatch_count"]) == 0,
        "aligned_upper_element_failure_count": int(candidate["aligned_upper_element_failure_count"]) == 0,
        "strict_pass_aligned_fail_row_count": int(candidate["strict_pass_aligned_fail_row_count"]) == 0,
    }
    return {**checks, "all": bool(all(checks.values()))}


def evaluate_timestep(*, timestep: int, guard: HoldoutTargetGuard, runtime: MutableMapping[str, Any], spec: Any) -> Mapping[str, Any]:
    fitted = fit_locked_direction_twice(timestep=timestep, context=guard, runtime=runtime)
    candidate = reconstruct_locked_holdout_candidate(control=fitted["control"], base_direction=fitted["direction"], context=guard, runtime=runtime, spec=spec)
    # Unlock only after direction fit, scale selection, and candidate generation are complete.
    guard.unlock()
    holdout_target = np.asarray(guard["holdout_target"], dtype=np.float32)
    fidelity = dict(runtime["staget"].compute_candidate_fidelity_metrics(
        control=fitted["control"], candidate=candidate["candidate"], target=holdout_target,
        selected_scale=candidate["selected_scale"], policy=runtime["staget"].CandidatePolicy(),
    ))
    gate_clean = all(int(candidate[key]) == 0 for key in (
        "length_log_z_element_mismatch_count", "aligned_upper_element_failure_count", "strict_pass_aligned_fail_row_count"))
    fidelity["mechanism_eligible"] = bool(fidelity["mechanism_eligible"] and gate_clean)
    fidelity["fidelity_eligible"] = bool(fidelity["fidelity_eligible"] and gate_clean)
    checks = per_timestep_checks(fidelity, candidate)
    return {
        "base_direction_id": LOCKED_BACKBONE, "timestep": int(timestep),
        "feature_mode": "full_segment_constraint", "selection_holdout_rows": int(fitted["control"].shape[0]),
        "objective_fit_rows": fitted["fit_rows"], "full_fit_repeat_count": 2,
        "model_identity": fitted["model_identity"], "holdout_feature_sha256": fitted["feature_sha256"],
        "holdout_direction_sha256": fitted["prediction_sha256"],
        "candidate_sha256": candidate["candidate_sha256"],
        "selected_scale_sha256": candidate["selected_scale_sha256"],
        "selected_scale_histogram": candidate["selected_scale_histogram"],
        "acceptance_rate": candidate["acceptance_rate"],
        "internal_scale_attempt_order": candidate["internal_scale_attempt_order"],
        "internal_scale_attempt_count": candidate["internal_scale_attempt_count"],
        "length_log_z_element_mismatch_count": candidate["length_log_z_element_mismatch_count"],
        "aligned_upper_element_failure_count": candidate["aligned_upper_element_failure_count"],
        "strict_pass_aligned_fail_row_count": candidate["strict_pass_aligned_fail_row_count"],
        "candidate_finalized_before_holdout_target_access": True,
        "holdout_target_used_for_fit": False, "holdout_target_used_for_selection": False,
        "holdout_target_used_only_for_final_metrics": True,
        "holdout_target_access_count": guard.access_count,
        "candidate_fidelity": fidelity, "pass_checks": checks, "scientific_pass": checks["all"],
        "candidate_tensor_persisted": False, "selected_scale_tensor_persisted": False,
        "direction_tensor_persisted": False, "holdout_target_tensor_persisted": False,
    }


def classify(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    passed = [int(r["timestep"]) for r in records if r.get("scientific_pass") is True]
    failed = [int(r["timestep"]) for r in records if r.get("scientific_pass") is not True]
    if not failed:
        return {"scientific_status": "READY", "root_cause": "phase314b_r258_stagev_locked_frontier_transfers_to_selection_holdout", "required_next_path": "FREEZE_SELECTION_HOLDOUT_VALIDATED_FRONTIER_AS_FINAL_BASELINE_MECHANISM_CONTRACT", "primary_failure_locus": "all_locked_timesteps_transfer", "passed_timesteps": passed, "failed_timesteps": failed}
    gate_keys = {"length_log_z_element_mismatch_count", "aligned_upper_element_failure_count", "strict_pass_aligned_fail_row_count"}
    gate_failure = any(not all(bool(r["pass_checks"].get(k)) for k in gate_keys) for r in records)
    mechanism_failure = any(r["pass_checks"].get("mechanism_eligible") is not True for r in records)
    if gate_failure:
        locus = "mechanism_gate_regression"
    elif mechanism_failure:
        locus = "admission_mechanism_transfer_failure"
    elif len(failed) == len(LOCKED_TIMESTEPS):
        locus = "all_timestep_fidelity_transfer_failure"
    else:
        locus = "timestep_specific_fidelity_distribution_shift"
    return {"scientific_status": "BLOCKED", "root_cause": "phase314b_r258_stagev_locked_frontier_fails_preregistered_selection_holdout_contract", "required_next_path": "AUDIT_SELECTION_HOLDOUT_TRANSFER_FAILURE_WITHOUT_RERUN_RETUNING_OR_FALLBACK", "primary_failure_locus": locus, "passed_timesteps": passed, "failed_timesteps": failed}


def worker_payload(*, root: Path, probe_payload: Mapping[str, Any], repository_head: str, stageu_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_environment_variables()
    modules = _science_modules()
    environment = modules["stages_resume3"].validate_probe_payload_for_science(probe_payload, resume2a=modules["resume2a"])
    runtime_modules = modules["stageo"]._runtime_modules()
    stagea = runtime_modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment)
    cold = stagea.assert_cold_cuda_context_portable()
    runtime = dict(modules["stageo"]._prepare_runtime(Path(root).resolve(), environment))
    runtime.update({"stageo": modules["stageo"], "stageq": modules["stageq"], "stager": modules["stager"], "stageh": modules["stageh"], "stageg": modules["stageg"], "staget": modules["staget"], "stageu_gate": _mapping(stageu_contract["aligned_gate_lock"], "Stage-U gate")})
    population = validate_population(runtime["context"])
    spec = modules["stager"].StageRSpec(); spec.validate()
    records = []
    for timestep in LOCKED_TIMESTEPS:
        # Each timestep receives a fresh locked guard; this is one joint evaluation,
        # not three adaptive selections or reruns.
        guard = HoldoutTargetGuard(runtime["context"])
        records.append(evaluate_timestep(timestep=timestep, guard=guard, runtime=runtime, spec=spec))
    result = classify(records)
    selected = None
    if result["scientific_status"] == "READY":
        selected = {"backbone_id": LOCKED_BACKBONE, "timesteps": list(LOCKED_TIMESTEPS), "timestep_policy": "joint_all_timesteps_no_cherry_pick", "aligned_gate_contract_sha256": stageu_contract["aligned_gate_lock"]["inner_gate_contract_sha256"]}
    payload: Dict[str, Any] = {
        "phase": PHASE, "schema": WORKER_SCHEMA, "execution_verdict": "PASS",
        **result, "process_id": os.getpid(), "repository_head": str(repository_head),
        "environment_sha256": sha256_bytes(stable_json_bytes(environment)),
        "cold_cuda_precheck": copy.deepcopy(dict(cold)),
        "environment_probe_rerun_in_science_process": False,
        "locked_frontier": {"backbone_id": LOCKED_BACKBONE, "timesteps": list(LOCKED_TIMESTEPS), "timestep_policy": "joint_all_timesteps_no_cherry_pick"},
        "population": population, "timestep_records": records,
        "selected_configuration": selected, "train_only_recommendation": None,
        "selection_holdout_evaluated": True,
        "selection_holdout_used_for_fit_or_selection": False,
        "post_holdout_backbone_selection_performed": False,
        "post_holdout_timestep_selection_performed": False,
        "threshold_changed_after_holdout": False,
        "fallback_backbone_used": False, "fallback_timestep_used": False,
        "stagev_execution": {"environment_probe_count": 1, "cold_science_worker_count": 1, "objective_full_fit_count": 6, "holdout_direction_prediction_count": 6, "candidate_generation_count": 3, "joint_selection_holdout_evaluation_count": 1, "holdout_target_metric_access_count": 3, "internal_scale_attempt_count": 21, "oof_fit_count": 0, "callback_pair_count": 0, "worker_output_persisted": False},
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker_payload(payload)
    return payload


def validate_worker_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageVError("worker schema/verdict changed")
    if payload.get("selection_holdout_evaluated") is not True or payload.get("selection_holdout_used_for_fit_or_selection") is not False:
        raise StageVError("holdout boundary changed")
    records = _sequence(payload.get("timestep_records"), "timestep records")
    if [int(r["timestep"]) for r in records] != list(LOCKED_TIMESTEPS):
        raise StageVError("timestep population/order changed")
    if any(r.get("holdout_target_access_count") != 1 for r in records):
        raise StageVError("holdout target access count changed")
    if any(r.get("full_fit_repeat_count") != 2 for r in records):
        raise StageVError("full-fit repeat count changed")
    execution = _mapping(payload.get("stagev_execution"), "Stage-V execution")
    expected = {"environment_probe_count": 1, "cold_science_worker_count": 1, "objective_full_fit_count": 6, "holdout_direction_prediction_count": 6, "candidate_generation_count": 3, "joint_selection_holdout_evaluation_count": 1, "holdout_target_metric_access_count": 3, "internal_scale_attempt_count": 21, "oof_fit_count": 0, "callback_pair_count": 0, "worker_output_persisted": False}
    if dict(execution) != expected:
        raise StageVError("Stage-V execution counts changed")
    if payload.get("scientific_status") == "READY":
        if payload.get("selected_configuration") is None or not all(r.get("scientific_pass") is True for r in records):
            raise StageVError("READY payload is inconsistent")
    else:
        if payload.get("selected_configuration") is not None:
            raise StageVError("failed holdout retained a configuration")
    require_false(payload, FALSE_BOUNDARIES, "Stage-V worker")
    return payload


def _run_child(command: Sequence[str], *, root: Path, label: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(list(command), cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ), check=False)
    if completed.returncode != 0:
        raise StageVError(f"{label} rc={completed.returncode} stdout={completed.stdout!r} stderr={completed.stderr!r}")
    return completed


def run_evaluation(*, root: Path, repository: Mapping[str, Any], python_bin: str) -> Mapping[str, Any]:
    validate_environment_variables()
    repo = Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix="phase314b_r258_stagev_") as temporary:
        directory = Path(temporary)
        probe_path, worker_path = directory / "environment_probe.json", directory / "worker.json"
        _run_child([str(python_bin), str(repo / "scripts/phase3_14b_r258_stages_resume2a_worker.py"), "--root", str(repo), "--mode", "environment-probe", "--output", str(probe_path)], root=repo, label="environment probe")
        probe = load_json(probe_path)
        _run_child([str(python_bin), str(repo / "scripts/phase3_14b_r258_stagev_worker.py"), "--root", str(repo), "--probe", str(probe_path), "--repository-head", str(repository["head"]), "--output", str(worker_path)], root=repo, label="single cold science worker")
        worker = validate_worker_payload(load_json(worker_path))
        probe_pid = int(_mapping(probe, "probe")["process_id"])
        worker_pid = int(worker["process_id"])
        if probe_pid == worker_pid:
            raise StageVError("probe and science worker processes are not distinct")
    summary = copy.deepcopy(dict(worker))
    summary.update({"schema": SCHEMA, "repository": {k: v for k, v in repository.items() if k not in ("stageu_contract", "stageu_validated")}, "process_topology": {"environment_probe_process_count": 1, "cold_science_worker_process_count": 1, "probe_process_id": probe_pid, "science_worker_process_id": worker_pid, "processes_distinct": True, "workers_launched_sequentially": True, "temporary_payloads_deleted": True}, "stageu_summary_sha256": EXPECTED_BASE_SUMMARY_SHA256, "stageu_contract_sha256": EXPECTED_BASE_CONTRACT_SHA256, "stageu_contract_payload_sha256": EXPECTED_CONTRACT_PAYLOAD_SHA256, "stageu_contract_read_only": True, "evaluation_count": 1, "rerun_authorized": False})
    summary.pop("scientific_result_sha256", None)
    summary["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(summary))
    return summary


def blocked_report(*, repository: Optional[Mapping[str, Any]], error: BaseException) -> Mapping[str, Any]:
    return {"phase": PHASE, "schema": BLOCKED_SCHEMA, "execution_verdict": "BLOCKED", "scientific_status": "BLOCKED", "root_cause": "phase314b_r258_stagev_execution_contract_failed", "required_next_path": "DESIGN_ADD_ONLY_STAGEV_EXECUTION_CONTRACT_CORRECTION_WITHOUT_REACCESSING_HOLDOUT_IF_ACCESS_OCCURRED", "primary_failure_locus": "execution_contract", "selected_configuration": None, "train_only_recommendation": None, "repository": None if repository is None else {k: v for k, v in repository.items() if k not in ("stageu_contract", "stageu_validated")}, "error_type": type(error).__name__, "error_message": str(error), "selection_holdout_access_state": "unknown_if_failure_after_worker_start", "complete_worker_population_claimed": False, "complete_fit_count_claimed": False, "complete_candidate_count_claimed": False, "complete_holdout_evaluation_count_claimed": False, "rerun_authorized": False, **{key: False for key in FALSE_BOUNDARIES}}


def write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageVError(f"write-once output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
