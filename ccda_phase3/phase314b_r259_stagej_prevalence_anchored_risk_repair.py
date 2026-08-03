"""Stage J objective-train-only prevalence-anchored risk repair.

Stage H established that the fixed raw candidates preserve mean MSE improvement
on the consumed frozen probe, while all three adverse-risk models are worse
than their objective-train prevalence constants. Stage I then generated and
sealed a new untouched evaluation surface before any repair search.

Stage J does not open the Stage-I NPZ or raw episodes. It reads only the Stage-I
tracked report and JSON seal metadata. Scientific computation uses the
historical objective-train rows only.

The repair keeps the candidate mechanism and the raw logistic risk model fixed.
It adds one deterministic calibration operation:

    p_repaired = q + alpha * (p_raw - q)

where q is the prevalence of the risk model's fit population. Alpha is the
closed-form Brier reliability slope clipped to [0, 1]. In each nested outer
fold, alpha is the minimum of the five inner held-out-fold slopes. The final
train-only alpha is the minimum of the six full cross-fit fold slopes. There is
no alpha grid, threshold search, descriptor search, L2 search, recipe fallback,
or timestep cherry-picking.

Before risk research, a new objective-only runtime must reproduce the immutable
Stage-G objective OOF candidate, selected-scale and descriptor identities. This
prevents the no-holdout adapter from silently changing the locked candidate
mechanism.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r259_stageg_one_shot_frozen_probe as stageg259
from ccda_phase3 import phase314b_r259_staged_resume2_cable_state_schema_recovery as resume2

PHASE = "Phase3.14b-r2.5.9 Stage J"
SCHEMA = "phase314b_r259_stagej_prevalence_anchored_risk_repair_v1"
PROBE_SCHEMA = SCHEMA + "_environment_probe_v1"
WORKER_SCHEMA = SCHEMA + "_worker_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

BASE_HEAD = "ca5bbe74154e0ba239f5d499c8a96d15b860b902"
BASE_IMPLEMENTATION = "8def374d4398786b068992ec1830c9b2945d04e2"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage J: repair risk transfer with prevalence anchoring"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage J prevalence-anchored risk evidence"
)
BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.9 Stage J blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagej_prevalence_anchored_risk_repair.py"),
    ("A", "scripts/phase3_14b_r259_stagej_worker.py"),
    ("A", "scripts/phase3_14b_r259_stagej_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagej_prevalence_anchored_risk_repair.py"),
)

STAGEI_REPORT = "reports/phase3_14b_r259_stagei_fresh_eval_seal_summary.json"
STAGEI_REPORT_SHA256 = "5943a506ca5ccaf0045e8c2ec75df83aca3c434920cda16cbc4c21a4401de63f"
STAGEI_SELF_SHA256 = "9a2ad6a69f2c6c68a1c50b3786979d69965da6a7fdf7cbe7bcd2c07cd3e09bf8"
STAGEI_DATA_ROOT = "data/phase3_14b_r259_fresh_eval_v1"
STAGEI_SEAL = STAGEI_DATA_ROOT + "/seal_manifest.json"
STAGEI_WINDOW_MANIFEST = STAGEI_DATA_ROOT + "/window_manifest.json"
STAGEI_INVENTORY = STAGEI_DATA_ROOT + "/artifact_inventory.json"
STAGEI_SEAL_INTERNAL_SHA256 = "31bc307dbe62431a9e4c602ce10a1cea384b66ef3cd12dba80a4d34692e2b083"
STAGEI_WINDOW_MANIFEST_SHA256 = "ec2925334c947e1a728a522386d7f507646acc2f520254408564e2b3c1e8255f"
STAGEI_WINDOW_NPZ_SHA256 = "b355febefee93de7065e71daedb4ed7f5c6c74948743b727c7675557560a586d"
STAGEI_WINDOW_ROWS = 1270
STAGEI_PAIR_GROUPS = 128

R258_OBJECTIVE_EVIDENCE = "reports/phase3_14b_r258_staged_resume2_worker_evidence.json"
R258_OBJECTIVE_EVIDENCE_SHA256 = "63ee24b7531491f785ae57f36c8cf0a30c10bec508a919df4d4a6a98b1c95ae2"
EXPECTED_OBJECTIVE_CONTROL_SHA256 = "b38316f817810f85cf7baa570333ce8cc60aa90d6aeb2261206bfd66e5eff9dc"

STAGEG_WORKER = "reports/phase3_14b_r259_stageg_frozen_probe_worker_evidence.json"
STAGEG_WORKER_FILE_SHA256 = "e6334481d75bcc48028eb5624d237c219c08212c8533a8ca05bedcecba87f48c"
STAGEG_WORKER_RESULT_SHA256 = "73b4ee84b6a7adcb6cc458cc5691e85cc1cc1c30e2cff9fdbf8e6cb5b88c99f0"

STAGEU_CONTRACT = "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"
STAGEU_CONTRACT_SHA256 = "c0bf2c477694cf1bda77f11c7ac407befb77d005afa8422e7e9bae2616714a8c"

PROBE_EVIDENCE = "reports/phase3_14b_r259_stagej_environment_probe_evidence.json"
WORKER_EVIDENCE = "reports/phase3_14b_r259_stagej_prevalence_anchored_worker_evidence.json"
SUCCESS_REPORT = "reports/phase3_14b_r259_stagej_prevalence_anchored_risk_repair_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_stagej_prevalence_anchored_risk_repair_blocked_summary.json"

LOCKED_TIMESTEPS: Tuple[int, ...] = (10, 25, 50)
OUTER_FOLDS = 6
EXPECTED_OBJECTIVE_ROWS = 638
EXPECTED_OBJECTIVE_GROUPS = 126
EXPECTED_CONDITIONS = ("free", "hidden_slack_breakaway_pin_v2")
LOCKED_DESCRIPTOR = "compact_v1"
LOCKED_SHRINKAGE = 1.0
LOCKED_THRESHOLD = 0.5
RISK_L2_BY_TIMESTEP = {10: 4.0, 25: 1.0, 50: 1.0}
MINIMUM_BRIER_FOLD_SUPPORT = 5
MINIMUM_POLICY_FOLD_SUPPORT = 4
SLOPE_EPSILON = 1.0e-12

EXPECTED_EXECUTION_COUNTS: Mapping[str, int] = {
    "historical_train_view_load_count": 1,
    "objective_control_training_count": 1,
    "objective_direction_fit_count": 18,
    "objective_candidate_generation_count": 18,
    "objective_risk_fit_count": 66,
    "descriptor_build_count": 3,
    "fresh_metadata_report_read_count": 4,
    "fresh_target_read_count": 0,
}

EXPECTED_STAGEG_IDENTITIES: Mapping[int, Mapping[str, str]] = {
    10: {
        "candidate": "40ace8308ea92a95ca4136951091cbec6d3ae53f33a0d1f54da9f1ae704050ae",
        "selected_scale": "a901f124123ed07931cc6a5e01ecd44a52643d62c421d4d9e56c16d47ea8490d",
        "descriptor": "3cbc1289e49266386b946adeed5df75875e04fb3177ebe3b7665f3661f80a244",
    },
    25: {
        "candidate": "80eeb99a73791657ef25ee1b7e1bc1a18acbb07a1c8ddfa6103df5759904dc8c",
        "selected_scale": "3fbfebd48d241cffe4fe7964393f1ea9cfe33b906e8d07a6b8f9e875414aacff",
        "descriptor": "9186cdb6231be28ded8bb741ff0fdf4d027f4cfaccfd88ab95f6cdc13255fbb2",
    },
    50: {
        "candidate": "7ec5aca1366ea14b55b947d426148622c372c6e4f1fbe99c0098dd00b7a3f54f",
        "selected_scale": "cc3c194ad93d1f1b3f0d1fcfa2369c33980ec672217cbf57820cfe89c97f4b7c",
        "descriptor": "e73bd3583be839feb6fbc52d431eafc06289208307a1a2f70ad88ba8ee3f3908",
    },
}

FALSE_BOUNDARIES = (
    "selection_holdout_target_indexed",
    "selection_holdout_evaluated",
    "frozen_probe_target_indexed",
    "frozen_probe_evaluated",
    "fresh_window_npz_opened",
    "fresh_raw_episode_opened",
    "fresh_target_indexed",
    "fresh_model_evaluated",
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
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "descriptor_tensor_persisted",
    "risk_probability_tensor_persisted",
)


class StageJError(RuntimeError):
    """Fail-closed Stage-J error."""


@dataclass(frozen=True)
class RepairSpec:
    minimum_brier_fold_support: int = MINIMUM_BRIER_FOLD_SUPPORT
    minimum_policy_fold_support: int = MINIMUM_POLICY_FOLD_SUPPORT
    minimum_acceptance_rate: float = 0.50
    minimum_positive_reduction_rate: float = 0.50
    adverse_sse_reduction_fraction: float = 0.10
    group_cvar_fraction: float = 0.20

    def validate(self) -> None:
        if self.minimum_brier_fold_support != 5:
            raise StageJError("Brier fold support changed")
        if self.minimum_policy_fold_support != 4:
            raise StageJError("policy fold support changed")
        if self.minimum_acceptance_rate != 0.50:
            raise StageJError("acceptance threshold changed")
        if self.minimum_positive_reduction_rate != 0.50:
            raise StageJError("positive reduction threshold changed")
        if self.adverse_sse_reduction_fraction != 0.10:
            raise StageJError("adverse SSE gate changed")
        if self.group_cvar_fraction != 0.20:
            raise StageJError("group CVaR gate changed")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
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
    return stageg259.sha256_array(np.asarray(value))


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageJError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageJError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
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
        raise StageJError("write-ahead evidence missing: {}".format(source_path))
    payload = source_path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    validator(value)
    atomic_write_once(Path(destination), payload)
    if Path(destination).read_bytes() != payload:
        raise StageJError("write-ahead promotion is not byte-exact")
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
            raise StageJError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != field})
    )


def validate_stagei_contract(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    report_path = repo / STAGEI_REPORT
    if not report_path.is_file() or sha256_file(report_path) != STAGEI_REPORT_SHA256:
        raise StageJError("Stage-I report identity changed")
    report = load_json(report_path)
    required = {
        "schema": "phase314b_r259_stagei_fresh_untouched_evaluation_seal_v1",
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "summary_sha256": STAGEI_SELF_SHA256,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise StageJError("Stage-I report field changed: {}".format(key))
    if _self_hash(report, "summary_sha256") != STAGEI_SELF_SHA256:
        raise StageJError("Stage-I report self-hash changed")
    seal_summary = report.get("fresh_evaluation_seal")
    if not isinstance(seal_summary, Mapping):
        raise StageJError("Stage-I seal summary missing")
    expected_seal = {
        "dataset_relative_root": STAGEI_DATA_ROOT,
        "seal_sha256": STAGEI_SEAL_INTERNAL_SHA256,
        "window_manifest_sha256": STAGEI_WINDOW_MANIFEST_SHA256,
        "window_npz_sha256": STAGEI_WINDOW_NPZ_SHA256,
        "window_row_count": STAGEI_WINDOW_ROWS,
        "window_pair_group_count": STAGEI_PAIR_GROUPS,
    }
    for key, expected in expected_seal.items():
        if seal_summary.get(key) != expected:
            raise StageJError("Stage-I seal summary changed: {}".format(key))
    governance = seal_summary.get("governance")
    if not isinstance(governance, Mapping):
        raise StageJError("Stage-I governance missing")
    for key in (
        "generated_before_new_risk_repair_search",
        "targets_not_used_for_fit_selection_or_evaluation",
        "future_evaluation_must_be_one_shot",
        "future_evaluation_requires_independently_locked_policy",
        "post_evaluation_retuning_forbidden",
        "fallback_after_evaluation_forbidden",
    ):
        if governance.get(key) is not True:
            raise StageJError("Stage-I governance changed: {}".format(key))
    if governance.get("future_evaluation_count") != 0:
        raise StageJError("Stage-I fresh set was already evaluated")

    # Metadata only: never open or hash the NPZ/raw episodes in Stage J.
    seal_path = repo / STAGEI_SEAL
    window_manifest_path = repo / STAGEI_WINDOW_MANIFEST
    inventory_path = repo / STAGEI_INVENTORY
    for path in (seal_path, window_manifest_path, inventory_path):
        if not path.is_file():
            raise StageJError("Stage-I metadata missing: {}".format(path))
    seal = load_json(seal_path)
    if seal.get("seal_sha256") != STAGEI_SEAL_INTERNAL_SHA256:
        raise StageJError("Stage-I on-disk seal identity changed")
    if _self_hash(seal, "seal_sha256") != STAGEI_SEAL_INTERNAL_SHA256:
        raise StageJError("Stage-I on-disk seal self-hash changed")
    if sha256_file(window_manifest_path) != STAGEI_WINDOW_MANIFEST_SHA256:
        raise StageJError("Stage-I window manifest identity changed")
    window_manifest = load_json(window_manifest_path)
    if window_manifest.get("row_count") != STAGEI_WINDOW_ROWS:
        raise StageJError("Stage-I window row count changed")
    if window_manifest.get("window_npz_sha256") != STAGEI_WINDOW_NPZ_SHA256:
        raise StageJError("Stage-I NPZ identity changed in metadata")
    inventory = load_json(inventory_path)
    if inventory.get("inventory_sha256") != seal.get("inventory_sha256"):
        raise StageJError("Stage-I inventory identity changed")
    return {
        "report": report,
        "seal": seal,
        "window_manifest": window_manifest,
        "inventory_identity": inventory.get("inventory_sha256"),
        "fresh_npz_opened": False,
        "fresh_raw_episode_opened": False,
    }


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-I implementation": (_git(repo, "rev-parse", BASE_HEAD + "^"), BASE_IMPLEMENTATION),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageJError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageJError("Stage-J implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageJError("Stage-J implementation population changed")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageJError("main worktree is dirty")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageJError("submodule worktree changed")
    if _git(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageJError("submodule is dirty")
    validate_stagei_contract(repo)
    if sha256_file(repo / STAGEG_WORKER) != STAGEG_WORKER_FILE_SHA256:
        raise StageJError("Stage-G worker file identity changed")
    stageg_worker = load_json(repo / STAGEG_WORKER)
    if stageg_worker.get("worker_result_sha256") != STAGEG_WORKER_RESULT_SHA256:
        raise StageJError("Stage-G worker result identity changed")
    if sha256_file(repo / R258_OBJECTIVE_EVIDENCE) != R258_OBJECTIVE_EVIDENCE_SHA256:
        raise StageJError("objective control evidence identity changed")
    if sha256_file(repo / STAGEU_CONTRACT) != STAGEU_CONTRACT_SHA256:
        raise StageJError("Stage-U aligned-gate contract identity changed")
    for relative in (PROBE_EVIDENCE, WORKER_EVIDENCE, SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageJError("Stage-J terminal artifact already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_i_implementation": BASE_IMPLEMENTATION,
        "stage_i_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def make_environment_probe(root: Path) -> Mapping[str, Any]:
    source = stageg259.make_environment_probe(Path(root).resolve())
    payload = dict(source)
    payload["phase"] = PHASE
    payload["schema"] = PROBE_SCHEMA
    payload.pop("probe_evidence_sha256", None)
    payload["selection_holdout_target_indexed"] = False
    payload["frozen_probe_target_indexed"] = False
    payload["fresh_target_indexed"] = False
    payload["probe_evidence_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_environment_probe(payload)
    return payload


def validate_environment_probe(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != PROBE_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageJError("environment probe schema/verdict changed")
    if payload.get("selection_holdout_target_indexed") is not False:
        raise StageJError("environment probe indexed selection holdout")
    if payload.get("frozen_probe_target_indexed") is not False:
        raise StageJError("environment probe indexed frozen probe")
    if payload.get("fresh_target_indexed") is not False:
        raise StageJError("environment probe indexed fresh targets")
    if payload.get("probe_evidence_sha256") != _self_hash(payload, "probe_evidence_sha256"):
        raise StageJError("environment probe self-hash changed")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise StageJError("environment payload missing")
    return environment


def _modules() -> Mapping[str, Any]:
    staged = stageg259._staged()
    runtime = staged.stagex._science_modules()
    return {"staged": staged, **runtime}


def _objective_only_control_and_context(
    root: Path, environment: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Build the runtime without calling build_audit_context or portable replay.

    The historical train-view container is eagerly materialized by the legacy
    NPZ loader, but only objective-train target rows are indexed. Selection and
    frozen masks are computed from metadata and their target rows are never
    sliced, evaluated, or passed to any model function.
    """
    modules = _modules()
    staged = modules["staged"]
    stagex = staged.stagex
    stageo = modules["stageo"]
    runtime_modules = stageo._runtime_modules()
    stagel = runtime_modules["stagel"]
    stagek = runtime_modules["stagek"]
    stagef = stagel.stagef
    staged258 = stagel.staged258
    stageb258 = stagef.stageb258
    stageb = stageb258.stageb
    stagec = stageb258.stagec
    stagea = stageb258.stagea
    stagea258 = stageb258.stagea258
    stageb_mechanism = stageb258.stageb_mechanism
    stagec_topk = stageb258.stagec_topk

    stagef.stagec258.stagea258.validate_environment_payload(environment)
    cold = stagef.stagec258.stagea258.assert_cold_cuda_context_portable()
    repo = Path(root).resolve()
    immutable = stageb258.staged.validate_immutable_inputs(repo)
    upstream = stagec_topk.validate_immutable_inputs(repo)
    objective_contract = stageb_mechanism.load_objective_contract(
        upstream["upstream_immutable"]["stagea_contract"]
    )
    upper_gate = stagea.load_upper_gate_contract(
        upstream["upstream_immutable"]["upstream_immutable"]["staged3_contract"]
    )
    stage_d_contract, stage_d_payload = stageb258.staged1.load_stage_d_gate(
        repo / stageb258.staged3.STAGE_D_GATE
    )
    arrays = stageb.load_npz_strict(
        repo / stageb.STAGEA_TRAIN_VIEW, required_keys=stageb.REQUIRED_TRAIN_KEYS
    )
    validation = stageb.validate_train_view(arrays)
    condition_all = np.asarray(arrays["diffusion_condition_x"], dtype=np.float32)
    target_container = np.asarray(arrays["diffusion_target_cable"], dtype=np.float32)
    groups_all = np.asarray(arrays["episode_group_key"]).astype(str)
    names_all = np.asarray(arrays["condition_name"]).astype(str)
    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    stageb_train, frozen_mask, stageb_split = stageb.deterministic_group_split(
        groups_all, folds=stageb_spec.group_folds, probe_fold=stageb_spec.probe_fold
    )
    objective_mask, selection_mask, selection_split = stagea.deterministic_selection_split(
        groups_all,
        stageb_train,
        folds=stagea.UpperObjectiveSpec().selection_group_folds,
        holdout_fold=stagea.UpperObjectiveSpec().selection_holdout_fold,
    )
    if np.any(objective_mask & (selection_mask | frozen_mask)):
        raise StageJError("objective split overlaps closed populations")
    if int(np.sum(objective_mask)) != EXPECTED_OBJECTIVE_ROWS:
        raise StageJError("objective row population changed")

    # Only this target indexing operation is allowed in Stage J.
    objective_condition = condition_all[objective_mask].copy()
    objective_target = target_container[objective_mask].copy()
    objective_groups = groups_all[objective_mask].copy()
    objective_names = names_all[objective_mask].copy()
    del target_container

    condition_standardizer = stageb.fit_standardizer(objective_condition)
    target_standardizer = stageb.fit_standardizer(objective_target)
    diagnostic_batch = stageb_mechanism.fixed_diagnostic_batch(
        condition=objective_condition,
        target=objective_target,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=stageb_mechanism.MechanismAuditSpec(),
    )
    expected_batch = immutable["upstream_immutable"]["stageb_contract"]["diagnostic_batch_sha256"]
    observed_batch = {key: value for key, value in diagnostic_batch.items() if key.endswith("_sha256")}
    if observed_batch != expected_batch:
        raise StageJError("objective diagnostic batch changed")
    _, calibration = stagec_topk.calibrate_candidates(
        stageb_spec=stageb_spec,
        diagnostic_batch=diagnostic_batch,
        target_standardizer=target_standardizer,
        objective_contract=objective_contract,
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    if calibration.get("calibration_sha256") != stageb258.EXPECTED_STAGEC_CALIBRATION_SHA256:
        raise StageJError("objective-only calibration identity changed")
    stageb.set_deterministic_runtime(stageb_spec.seed)
    model, training, diagnostics = stagec_topk.train_candidate(
        condition=objective_condition,
        target=objective_target,
        stageb_spec=stageb_spec,
        objective_contract=objective_contract,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        candidate=None,
        diagnostic_batch=diagnostic_batch,
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    if training.get("initial_model_sha256") != stageb258.EXPECTED_CONTROL_INITIAL_MODEL_SHA256:
        raise StageJError("objective-only control initialization changed")
    direction_spec = staged258.DirectionSurrogateSpec()
    direction_spec.validate()
    controls, control_sha = stagec.one_step_predictions(
        model=model,
        condition=objective_condition,
        target=objective_target,
        spec=stageb_spec,
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        noise_seed=stageb_spec.seed + int(direction_spec.objective_train_noise_seed_offset),
    )
    if control_sha != EXPECTED_OBJECTIVE_CONTROL_SHA256:
        raise StageJError("objective control prediction identity changed")
    assignment, mapping = staged258.deterministic_group_folds(
        objective_groups, folds=direction_spec.grouped_cv_folds
    )
    if len(set(objective_groups.tolist())) != EXPECTED_OBJECTIVE_GROUPS:
        raise StageJError("objective group population changed")

    # Objective-only geometry is permitted only if candidate identities later
    # prove it is functionally identical to the immutable Stage-G mechanism.
    historical_geometry = stageb.fit_geometry_contract(objective_target)
    objective_context: Dict[str, Any] = {
        "repository_root": repo,
        "stageb_spec": stageb_spec,
        "objective_contract": objective_contract,
        "upper_gate": upper_gate,
        "stage_d_contract": stage_d_contract,
        "stage_d_payload": stage_d_payload,
        "arrays_validation": validation,
        "condition": objective_condition,
        "target": objective_target,
        "groups": objective_groups,
        "condition_name": objective_names,
        "stageb_train_mask": np.ones(EXPECTED_OBJECTIVE_ROWS, dtype=np.bool_),
        "objective_train_mask": np.ones(EXPECTED_OBJECTIVE_ROWS, dtype=np.bool_),
        "selection_holdout_mask": np.zeros(EXPECTED_OBJECTIVE_ROWS, dtype=np.bool_),
        "frozen_probe_mask": np.zeros(EXPECTED_OBJECTIVE_ROWS, dtype=np.bool_),
        "stageb_split": stageb_split,
        "selection_split": selection_split,
        "condition_standardizer": condition_standardizer,
        "target_standardizer": target_standardizer,
        "historical_geometry": historical_geometry,
        "model": model,
        "objective_condition": objective_condition,
        "objective_target": objective_target,
        "objective_groups": objective_groups,
        "objective_condition_name": objective_names,
        "objective_control_predictions": controls,
        "objective_control_prediction_sha256": control_sha,
        "objective_fold_assignment": assignment,
        "objective_fold_mapping": mapping,
    }
    stagef_spec = stagef.ConstraintAwareSpec()
    stagef_spec.validate()
    ulp = stagef.calibrate_float32_constraint_z_policy(
        controls_by_timestep={int(t): controls[int(t)] for t in LOCKED_TIMESTEPS},
        context=objective_context,
        spec=stagef_spec,
    )
    if float(ulp["selected_factor"]) != stageo.EXPECTED_SELECTED_ULP_FACTOR:
        raise StageJError("objective-only ULP policy changed")
    stagef_spec = stagef.replace(
        stagef_spec, translation_float32_z_ulp_factor=stageo.EXPECTED_SELECTED_ULP_FACTOR
    )
    integrator_spec = stagel.stagee258.ConstrainedIntegratorSpec()
    integrator_spec.validate()
    stageu_contract = load_json(repo / STAGEU_CONTRACT)
    runtime = {
        **modules,
        **runtime_modules,
        "stagef": stagef,
        "stageu_gate": stageu_contract["aligned_gate_lock"],
        "direction_spec": direction_spec,
        "stagef_spec": stagef_spec,
        "integrator_spec": integrator_spec,
        "callback_spec": stagek.ExplicitPredicateCallbackSpec(),
        "context": objective_context,
        "cold_main_worker_context": cold,
        "ulp_policy": ulp,
    }
    audit = {
        "historical_train_view_eagerly_materialized": True,
        "objective_target_rows_indexed": EXPECTED_OBJECTIVE_ROWS,
        "selection_holdout_target_rows_indexed": 0,
        "frozen_probe_target_rows_indexed": 0,
        "fresh_target_rows_indexed": 0,
        "build_audit_context_called": False,
        "portable_control_replay_called": False,
        "objective_control_prediction_sha256": control_sha,
        "calibration_sha256": calibration["calibration_sha256"],
        "control_initial_model_sha256": training["initial_model_sha256"],
        "control_training_diagnostics_sha256": sha256_bytes(stable_json_bytes(diagnostics)),
    }
    return runtime, audit


def anchored_probability(raw: np.ndarray, anchor: float, alpha: float) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    q = float(anchor)
    a = float(alpha)
    if not (0.0 <= q <= 1.0) or not (0.0 <= a <= 1.0):
        raise StageJError("anchor or alpha outside [0,1]")
    result = np.clip(q + a * (value - q), 0.0, 1.0)
    if not np.all(np.isfinite(result)):
        raise StageJError("anchored probability is non-finite")
    return result


def reliability_slope(
    raw: np.ndarray, labels: np.ndarray, anchor: float, mask: np.ndarray
) -> Mapping[str, Any]:
    probability = np.asarray(raw, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    use = np.asarray(mask, dtype=np.bool_)
    if probability.shape != target.shape or use.shape != target.shape:
        raise StageJError("reliability slope population changed")
    if not np.any(use):
        raise StageJError("reliability slope has no rows")
    delta = probability[use] - float(anchor)
    numerator = float(np.sum(delta * (target[use] - float(anchor))))
    denominator = float(np.sum(delta * delta))
    unconstrained = 0.0 if denominator <= SLOPE_EPSILON else numerator / denominator
    alpha = float(np.clip(unconstrained, 0.0, 1.0))
    return {
        "alpha": alpha,
        "unconstrained_alpha": unconstrained,
        "numerator": numerator,
        "denominator": denominator,
        "row_count": int(np.sum(use)),
        "anchor": float(anchor),
    }


def brier(probability: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    use = np.asarray(mask, dtype=np.bool_)
    if p.shape != y.shape or use.shape != y.shape or not np.any(use):
        raise StageJError("Brier population changed")
    return float(np.mean((p[use] - y[use]) ** 2))


def _condition_brier(
    probability: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    condition_name: Sequence[Any],
    anchor_by_row: np.ndarray,
) -> Mapping[str, Any]:
    names = np.asarray(condition_name).astype(str)
    result: Dict[str, Any] = {}
    for name in EXPECTED_CONDITIONS:
        use = np.asarray(mask, dtype=np.bool_) & (names == name)
        if not np.any(use):
            raise StageJError("condition Brier population missing: {}".format(name))
        repaired = brier(probability, labels, use)
        constant = brier(anchor_by_row, labels, use)
        result[name] = {
            "row_count": int(np.sum(use)),
            "repaired_brier": repaired,
            "constant_brier": constant,
            "nonworse": bool(repaired <= constant + 1.0e-12),
        }
    return result


def _fit_model(
    staged: Any,
    descriptor: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    l2: float,
) -> Mapping[str, Any]:
    return staged.fit_risk_model_checked(descriptor, labels, fit_mask, l2)


def nested_repair_for_timestep(
    *,
    timestep: int,
    staged: Any,
    descriptor: np.ndarray,
    labels: np.ndarray,
    risk_fit_mask: np.ndarray,
    folds: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    selected_scale: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    condition_name: Sequence[Any],
    spec: RepairSpec,
) -> Mapping[str, Any]:
    spec.validate()
    rows = descriptor.shape[0]
    if rows != EXPECTED_OBJECTIVE_ROWS:
        raise StageJError("nested repair row population changed")
    raw_oof = np.zeros(rows, dtype=np.float64)
    repaired_oof = np.zeros(rows, dtype=np.float64)
    anchor_oof = np.zeros(rows, dtype=np.float64)
    covered = np.zeros(rows, dtype=np.bool_)
    model_cache: Dict[Tuple[int, ...], Mapping[str, Any]] = {}
    fit_count = 0

    def model_excluding(excluded: Tuple[int, ...]) -> Mapping[str, Any]:
        nonlocal fit_count
        key = tuple(sorted(excluded))
        if key not in model_cache:
            fit_mask = np.asarray(risk_fit_mask, dtype=np.bool_).copy()
            for fold in key:
                fit_mask &= folds != int(fold)
            model_cache[key] = _fit_model(
                staged, descriptor, labels, fit_mask, RISK_L2_BY_TIMESTEP[int(timestep)]
            )
            fit_count += 1
        return model_cache[key]

    outer_records: List[Mapping[str, Any]] = []
    brier_support = 0
    policy_support = 0
    for outer in range(OUTER_FOLDS):
        inner_slopes = []
        for inner in range(OUTER_FOLDS):
            if inner == outer:
                continue
            model = model_excluding((outer, inner))
            inner_mask = (folds == inner) & risk_fit_mask
            prediction = staged.stagex.predict_risk(model, descriptor)
            slope = reliability_slope(
                prediction, labels, float(model["prevalence"]), inner_mask
            )
            inner_slopes.append({"inner_fold": inner, **slope})
        alpha = min(float(item["alpha"]) for item in inner_slopes)
        outer_model = model_excluding((outer,))
        outer_rows = folds == outer
        outer_raw = staged.stagex.predict_risk(outer_model, descriptor[outer_rows])
        outer_anchor = float(outer_model["prevalence"])
        outer_repaired = anchored_probability(outer_raw, outer_anchor, alpha)
        raw_oof[outer_rows] = outer_raw
        repaired_oof[outer_rows] = outer_repaired
        anchor_oof[outer_rows] = outer_anchor
        covered[outer_rows] = True
        outer_risk = outer_rows & risk_fit_mask
        raw_score = brier(raw_oof, labels, outer_risk)
        repaired_score = brier(repaired_oof, labels, outer_risk)
        constant_score = brier(anchor_oof, labels, outer_risk)
        brier_pass = bool(
            repaired_score <= constant_score + 1.0e-12
            and repaired_score < raw_score - 1.0e-12
        )
        brier_support += int(brier_pass)
        record = staged.stagex.evaluate_policy(
            control[outer_rows],
            candidate[outer_rows],
            selected_scale[outer_rows],
            repaired_oof[outer_rows],
            target[outer_rows],
            np.asarray(groups)[outer_rows],
            LOCKED_THRESHOLD,
            anchor_oof[outer_rows],
            staged.stagex.StageXSpec(),
        )
        baseline = staged.stagex.evaluate_policy(
            control[outer_rows],
            candidate[outer_rows],
            selected_scale[outer_rows],
            np.zeros(int(np.sum(outer_rows)), dtype=np.float64),
            target[outer_rows],
            np.asarray(groups)[outer_rows],
            1.0,
            None,
            staged.stagex.StageXSpec(),
        )
        eligibility = staged.recipe_is_eligible(record, baseline, staged.StageDSpec())
        policy_support += int(eligibility["pass"] is True)
        outer_records.append(
            {
                "outer_fold": outer,
                "alpha": alpha,
                "inner_slopes": inner_slopes,
                "risk_fit_prevalence": outer_anchor,
                "raw_brier": raw_score,
                "repaired_brier": repaired_score,
                "constant_brier": constant_score,
                "brier_pass": brier_pass,
                "record": record,
                "no_abstention_baseline": baseline,
                "eligibility": eligibility,
                "raw_probability_sha256": sha256_array(outer_raw),
                "repaired_probability_sha256": sha256_array(outer_repaired),
            }
        )
    if not np.all(covered):
        raise StageJError("nested outer OOF coverage incomplete")

    final_fold_slopes = []
    for fold in range(OUTER_FOLDS):
        model = model_excluding((fold,))
        fold_mask = (folds == fold) & risk_fit_mask
        prediction = staged.stagex.predict_risk(model, descriptor)
        final_fold_slopes.append(
            {"fold": fold, **reliability_slope(prediction, labels, float(model["prevalence"]), fold_mask)}
        )
    final_alpha = min(float(item["alpha"]) for item in final_fold_slopes)
    full_model = _fit_model(
        staged,
        descriptor,
        labels,
        np.asarray(risk_fit_mask, dtype=np.bool_),
        RISK_L2_BY_TIMESTEP[int(timestep)],
    )
    fit_count += 1

    aggregate_raw_brier = brier(raw_oof, labels, risk_fit_mask)
    aggregate_repaired_brier = brier(repaired_oof, labels, risk_fit_mask)
    aggregate_constant_brier = brier(anchor_oof, labels, risk_fit_mask)
    combined_record = staged.stagex.evaluate_policy(
        control,
        candidate,
        selected_scale,
        repaired_oof,
        target,
        groups,
        LOCKED_THRESHOLD,
        anchor_oof,
        staged.stagex.StageXSpec(),
    )
    combined_baseline = staged.stagex.evaluate_policy(
        control,
        candidate,
        selected_scale,
        np.zeros(rows, dtype=np.float64),
        target,
        groups,
        1.0,
        None,
        staged.stagex.StageXSpec(),
    )
    combined_eligibility = staged.recipe_is_eligible(
        combined_record, combined_baseline, staged.StageDSpec()
    )
    condition_records = _condition_brier(
        repaired_oof, labels, risk_fit_mask, condition_name, anchor_oof
    )
    gates = {
        "positive_final_alpha": final_alpha > 0.0,
        "aggregate_brier_nonworse_than_constant": (
            aggregate_repaired_brier <= aggregate_constant_brier + 1.0e-12
        ),
        "aggregate_brier_improves_raw": (
            aggregate_repaired_brier < aggregate_raw_brier - 1.0e-12
        ),
        "brier_fold_support": brier_support >= spec.minimum_brier_fold_support,
        "policy_fold_support": policy_support >= spec.minimum_policy_fold_support,
        "all_conditions_brier_nonworse": all(
            item["nonworse"] is True for item in condition_records.values()
        ),
        "combined_policy_eligible": combined_eligibility["pass"] is True,
    }
    gates["all"] = bool(all(gates.values()))
    return {
        "timestep": int(timestep),
        "risk_l2": RISK_L2_BY_TIMESTEP[int(timestep)],
        "repair_rule": "minimum_inner_fold_closed_form_linear_reliability_slope",
        "outer_records": outer_records,
        "brier_fold_support": brier_support,
        "policy_fold_support": policy_support,
        "final_fold_slopes": final_fold_slopes,
        "final_alpha": final_alpha,
        "aggregate": {
            "raw_brier": aggregate_raw_brier,
            "repaired_brier": aggregate_repaired_brier,
            "constant_brier": aggregate_constant_brier,
            "brier_improvement_vs_raw": aggregate_raw_brier - aggregate_repaired_brier,
            "brier_skill_vs_constant": (
                1.0 - aggregate_repaired_brier / aggregate_constant_brier
                if aggregate_constant_brier > 0.0
                else None
            ),
            "condition_brier": condition_records,
            "record": combined_record,
            "no_abstention_baseline": combined_baseline,
            "eligibility": combined_eligibility,
        },
        "gates": gates,
        "fit_count": fit_count,
        "raw_oof_probability_sha256": sha256_array(raw_oof),
        "repaired_oof_probability_sha256": sha256_array(repaired_oof),
        "anchor_oof_sha256": sha256_array(anchor_oof),
        "full_model_identity": copy.deepcopy(full_model["identity"]),
        "full_model_prevalence": float(full_model["prevalence"]),
    }


def run_worker(
    *, root: Path, probe_payload: Mapping[str, Any], repository_head: str
) -> Mapping[str, Any]:
    environment = validate_environment_probe(probe_payload)
    repo = Path(root).resolve()
    stagei = validate_stagei_contract(repo)
    runtime, objective_access_audit = _objective_only_control_and_context(repo, environment)
    staged = stageg259._staged()
    stagex = staged.stagex
    context = runtime["context"]
    stagef = runtime["stagef"]
    definition = stagef.definition_by_id(stagex.LOCKED_BACKBONE)
    assignment = np.asarray(context["objective_fold_assignment"], dtype=np.int64)
    target = np.asarray(context["objective_target"], dtype=np.float32)
    groups = np.asarray(context["objective_groups"]).astype(str)
    names = np.asarray(context["objective_condition_name"]).astype(str)
    reconstruction_spec = runtime["stager"].StageRSpec()
    reconstruction_spec.validate()
    timestep_records: Dict[str, Mapping[str, Any]] = {}
    execution_counts = {
        "historical_train_view_load_count": 1,
        "objective_control_training_count": 1,
        "objective_direction_fit_count": 0,
        "objective_candidate_generation_count": 0,
        "objective_risk_fit_count": 0,
        "descriptor_build_count": 0,
        "fresh_metadata_report_read_count": 4,
        "fresh_target_read_count": 0,
    }
    all_candidate_identity = True
    for timestep in LOCKED_TIMESTEPS:
        control = np.asarray(context["objective_control_predictions"][timestep], dtype=np.float32)
        features = stagef.build_constraint_features(
            condition=context["objective_condition"],
            control=control,
            condition_name=names,
            feature_mode=definition.feature_mode,
            context=context,
        )
        oracle = stagef.generate_projected_oracle_target(
            control=control,
            target=target,
            groups=groups,
            condition_name=names,
            timestep=timestep,
            context=context,
            direction_spec=runtime["direction_spec"],
            integrator_spec=runtime["integrator_spec"],
            spec=runtime["stagef_spec"],
        )
        oof_direction = np.zeros(control.shape, dtype=np.float64)
        oof_candidate = control.copy()
        oof_scale = np.zeros(control.shape[0], dtype=np.float64)
        for fold in range(OUTER_FOLDS):
            test_mask = assignment == fold
            indices = np.flatnonzero(test_mask)
            fitted = stagex._fit_direction(
                runtime=runtime,
                context=context,
                timestep=timestep,
                fit_mask=~test_mask,
                predict_indices=indices,
                oracle=oracle,
                features=features,
            )
            execution_counts["objective_direction_fit_count"] += 1
            generated = stagex._candidate_for_indices(
                runtime=runtime,
                context=context,
                timestep=timestep,
                indices=indices,
                direction=fitted["direction"],
                shrinkage=LOCKED_SHRINKAGE,
                reconstruction_spec=reconstruction_spec,
            )
            staged.kernel.require_clean_generation(
                generated, label="Stage-J objective OOF t{} fold{}".format(timestep, fold)
            )
            execution_counts["objective_candidate_generation_count"] += 1
            oof_direction[indices] = fitted["direction"]
            oof_candidate[indices] = generated["candidate"]
            oof_scale[indices] = generated["selected_scale"]
        descriptor = resume2.build_risk_descriptors_48(
            LOCKED_DESCRIPTOR, control, oof_direction, oof_candidate, oof_scale
        )
        execution_counts["descriptor_build_count"] += 1
        observed_identity = {
            "candidate": sha256_array(oof_candidate),
            "selected_scale": sha256_array(oof_scale),
            "descriptor": sha256_array(descriptor),
        }
        identity_pass = observed_identity == dict(EXPECTED_STAGEG_IDENTITIES[timestep])
        all_candidate_identity = bool(all_candidate_identity and identity_pass)
        if not identity_pass:
            raise StageJError(
                "objective-only runtime changed locked Stage-G identity at t={}: {!r}".format(
                    timestep, observed_identity
                )
            )
        labels = stagex._risk_labels(control, oof_candidate, target)
        risk_fit_mask = oof_scale > 0.0
        repair = nested_repair_for_timestep(
            timestep=timestep,
            staged=staged,
            descriptor=descriptor,
            labels=labels,
            risk_fit_mask=risk_fit_mask,
            folds=assignment,
            control=control,
            candidate=oof_candidate,
            selected_scale=oof_scale,
            target=target,
            groups=groups,
            condition_name=names,
            spec=RepairSpec(),
        )
        execution_counts["objective_risk_fit_count"] += int(repair["fit_count"])
        timestep_records[str(timestep)] = {
            "locked_identity": {
                "expected": dict(EXPECTED_STAGEG_IDENTITIES[timestep]),
                "observed": observed_identity,
                "exact": identity_pass,
            },
            "adverse_label_rate": float(np.mean(labels)),
            "risk_fit_row_count": int(np.sum(risk_fit_mask)),
            **repair,
        }
    all_repair_pass = all(
        timestep_records[str(t)]["gates"]["all"] is True for t in LOCKED_TIMESTEPS
    )
    ready = bool(all_candidate_identity and all_repair_pass)
    recommendation = None
    if ready:
        recommendation = {
            "backbone_id": stagex.LOCKED_BACKBONE,
            "timesteps": list(LOCKED_TIMESTEPS),
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "descriptor_id": LOCKED_DESCRIPTOR,
            "direction_shrinkage": LOCKED_SHRINKAGE,
            "risk_threshold": LOCKED_THRESHOLD,
            "risk_l2_by_timestep": {str(k): v for k, v in RISK_L2_BY_TIMESTEP.items()},
            "calibration_rule": "linear_prevalence_anchor_with_minimum_crossfit_fold_reliability_slope",
            "alpha_by_timestep": {
                str(t): float(timestep_records[str(t)]["final_alpha"])
                for t in LOCKED_TIMESTEPS
            },
            "full_risk_model_identity_by_timestep": {
                str(t): timestep_records[str(t)]["full_model_identity"]
                for t in LOCKED_TIMESTEPS
            },
            "fresh_evaluation_access_authorized_by_this_report": False,
            "selection_role": "objective_train_nested_group_oof_repair_recommendation",
        }
    if execution_counts != dict(EXPECTED_EXECUTION_COUNTS):
        raise StageJError("Stage-J execution counts changed: {!r}".format(execution_counts))

    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY" if ready else "BLOCKED",
        "root_cause": (
            "phase314b_r259_stagej_prevalence_anchored_risk_repair_passes_objective_train_nested_group_oof"
            if ready
            else "phase314b_r259_stagej_prevalence_anchored_risk_repair_fails_objective_train_nested_group_oof"
        ),
        "required_next_path": (
            "LOCK_R259_PREVALENCE_ANCHORED_RISK_REPAIR_BEFORE_ONE_SHOT_SEALED_FRESH_EVALUATION"
            if ready
            else "AUDIT_R259_PREVALENCE_ANCHORED_RISK_REPAIR_OUTER_FOLD_FAILURE_WITHOUT_FRESH_SET_ACCESS"
        ),
        "primary_failure_locus": (
            "prevalence_anchored_risk_repair_ready_for_structural_lock"
            if ready
            else "prevalence_anchored_risk_repair_outer_fold_gate"
        ),
        "process_id": os.getpid(),
        "repository_head": repository_head,
        "stagei_seal_identity": {
            "seal_sha256": STAGEI_SEAL_INTERNAL_SHA256,
            "window_npz_sha256": STAGEI_WINDOW_NPZ_SHA256,
            "window_rows": STAGEI_WINDOW_ROWS,
            "fresh_npz_opened": stagei["fresh_npz_opened"],
            "fresh_raw_episode_opened": stagei["fresh_raw_episode_opened"],
        },
        "objective_access_audit": objective_access_audit,
        "locked_candidate_identity_exact_all_timesteps": all_candidate_identity,
        "timestep_records": timestep_records,
        "execution_counts": execution_counts,
        "selected_configuration": None,
        "train_only_recommendation": recommendation,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_worker(payload)
    return payload


def validate_worker(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKER_SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageJError("worker schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageJError("worker scientific status changed")
    records = payload.get("timestep_records")
    if not isinstance(records, Mapping) or sorted(records) != ["10", "25", "50"]:
        raise StageJError("worker timestep population changed")
    if payload.get("execution_counts") != dict(EXPECTED_EXECUTION_COUNTS):
        raise StageJError("worker execution counts changed")
    ready = all(records[str(t)]["gates"]["all"] is True for t in LOCKED_TIMESTEPS)
    if (payload.get("scientific_status") == "READY") != ready:
        raise StageJError("worker classification changed")
    if ready and not isinstance(payload.get("train_only_recommendation"), Mapping):
        raise StageJError("READY worker lacks recommendation")
    if not ready and payload.get("train_only_recommendation") is not None:
        raise StageJError("BLOCKED worker retained recommendation")
    if payload.get("selected_configuration") is not None:
        raise StageJError("Stage J selected a final configuration")
    for key in FALSE_BOUNDARIES:
        if payload.get(key) is not False:
            raise StageJError("worker crossed forbidden boundary: {}".format(key))
    for key in (
        "selection_holdout_evaluation_count_added",
        "frozen_probe_evaluation_count_added",
        "fresh_evaluation_count_added",
        "cumulative_fresh_evaluation_count",
    ):
        if payload.get(key) != 0:
            raise StageJError("worker evaluation count changed: {}".format(key))
    if payload.get("cumulative_selection_holdout_evaluation_count") != 1:
        raise StageJError("selection holdout cumulative count changed")
    if payload.get("cumulative_frozen_probe_evaluation_count") != 1:
        raise StageJError("frozen probe cumulative count changed")
    if payload.get("rerun_authorized") is not False:
        raise StageJError("worker authorized rerun")
    if payload.get("worker_result_sha256") != _self_hash(payload, "worker_result_sha256"):
        raise StageJError("worker self-hash changed")


def build_summary(
    *, repository: Mapping[str, Any], probe_path: Path, worker_path: Path
) -> Mapping[str, Any]:
    probe = load_json(probe_path)
    worker = load_json(worker_path)
    validate_environment_probe(probe)
    validate_worker(worker)
    if int(probe["process_id"]) == int(worker["process_id"]):
        raise StageJError("probe and worker process IDs are not distinct")
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": worker["scientific_status"],
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "primary_failure_locus": worker["primary_failure_locus"],
        "repository": dict(repository),
        "stagei_seal_identity": copy.deepcopy(worker["stagei_seal_identity"]),
        "objective_access_audit": copy.deepcopy(worker["objective_access_audit"]),
        "locked_candidate_identity_exact_all_timesteps": worker[
            "locked_candidate_identity_exact_all_timesteps"
        ],
        "timestep_records": copy.deepcopy(worker["timestep_records"]),
        "execution_counts": copy.deepcopy(worker["execution_counts"]),
        "process_topology": {
            "environment_probe_count": 1,
            "cold_science_worker_count": 1,
            "probe_process_id": int(probe["process_id"]),
            "worker_process_id": int(worker["process_id"]),
            "processes_distinct": True,
        },
        "durable_evidence": {
            "probe_path": PROBE_EVIDENCE,
            "probe_file_sha256": sha256_file(probe_path),
            "worker_path": WORKER_EVIDENCE,
            "worker_file_sha256": sha256_file(worker_path),
            "worker_result_sha256": worker["worker_result_sha256"],
            "controller_recomputed_science": False,
        },
        "selected_configuration": None,
        "train_only_recommendation": copy.deepcopy(worker["train_only_recommendation"]),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("execution_verdict") != "PASS":
        raise StageJError("summary schema/verdict changed")
    if payload.get("scientific_status") not in ("READY", "BLOCKED"):
        raise StageJError("summary scientific status changed")
    if payload.get("selected_configuration") is not None:
        raise StageJError("summary selected final configuration")
    if payload.get("fresh_evaluation_count_added") != 0:
        raise StageJError("summary accessed fresh set")
    if payload.get("cumulative_fresh_evaluation_count") != 0:
        raise StageJError("summary fresh evaluation count changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageJError("summary crossed forbidden boundary")
    if payload.get("summary_sha256") != _self_hash(payload, "summary_sha256"):
        raise StageJError("summary self-hash changed")


def blocked_report(
    repository: Optional[Mapping[str, Any]], error: BaseException, worker_started: bool
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagej_execution_or_objective_only_identity_gate_failed",
        "required_next_path": "AUDIT_R259_STAGEJ_FAILURE_WITHOUT_SELECTION_FROZEN_OR_FRESH_ACCESS",
        "primary_failure_locus": "stagej_execution_contract",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "worker_started": bool(worker_started),
        "science_reexecution_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
