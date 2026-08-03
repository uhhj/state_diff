"""Phase3.14b-r2.5.9 Stage I fresh untouched evaluation-set seal.

Stage H proved that raw state-correction candidates retain mean improvement on
all three timesteps while the adverse-risk probabilities fail to transfer
beyond what can be explained by prevalence shift.  The consumed selection
holdout and frozen probe must never be reused for model repair.

This stage establishes the next independent evaluation surface *before* any
new risk-repair search.  It generates paired formal-task episodes at a
predeclared, non-overlapping visible-seed range, builds the standard state-v2
windows, audits all identities and writes a cryptographic seal.  It does not
fit or evaluate a model and it does not load the historical dataset, selection
holdout or frozen probe.

The generated dataset lives outside Git under
``data/phase3_14b_r259_fresh_eval_v1``.  Git evidence contains only the seal
summary.  Future stages may train only on the historical objective-train split;
the sealed dataset may be opened exactly once only after a new risk-repair
policy is independently locked.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PHASE = "Phase3.14b-r2.5.9 Stage I"
SCHEMA = "phase314b_r259_stagei_fresh_untouched_evaluation_seal_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"

ORIGINAL_STAGE_H_IMPLEMENTATION = "05dbc4c045799f0534d7a48654c82db18e55a56a"
ORIGINAL_STAGE_H_EVIDENCE = "bd5bc7d7d71ade7c7a219063fd062ac4e2a5f9d2"
STAGE_H_IMPLEMENTATION = "449a7425ee8e1b5369a863a436918668f6b980eb"
STAGE_H_EVIDENCE = "edb1908f0cfa8e90625172098ad542e139682b9d"
STAGE_H_REPORT = (
    "reports/phase3_14b_r259_stageh_frozen_probe_transfer_failure_audit_summary.json"
)
STAGE_H_REPORT_SHA256 = (
    "cb3504172a223b51a206191fb46c2c8874a9d84170d4c6985779fcdeed8e0dfb"
)
STAGE_H_SELF_SHA256 = (
    "21e0836da0417a9c40e8cc66e85beddc7062a15b636e79f66be9f9f63897fcbc"
)
RECOVERY_MANIFEST = (
    "reports/phase3_14b_r259_stageeh_lost_instance_recovery_manifest.json"
)
RECOVERY_MANIFEST_SHA256 = (
    "dd8d453ce22db4d132020f04cb093031b02e7106b5543e099273f8c4b8a838be"
)
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.9 Stage I: seal fresh untouched risk-transfer evaluation set"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.9 Stage I fresh evaluation seal"
)
BLOCKED_SUBJECT = "Record Phase3.14b-r2.5.9 Stage I blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r259_stagei_fresh_eval_seal.py"),
    ("A", "scripts/phase3_14b_r259_stagei_fresh_eval_execute.py"),
    ("A", "tests/test_phase3_14b_r259_stagei_fresh_eval_seal.py"),
)

DATASET_RELATIVE_ROOT = "data/phase3_14b_r259_fresh_eval_v1"
RAW_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/raw"
WINDOWS_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/windows"
WINDOW_NPZ_RELATIVE = WINDOWS_RELATIVE_ROOT + "/fresh_eval_windows.npz"
ACTION_TEMPLATE_RELATIVE = WINDOWS_RELATIVE_ROOT + "/action_template.pkl"
WINDOW_MANIFEST_RELATIVE = DATASET_RELATIVE_ROOT + "/window_manifest.json"
INVENTORY_RELATIVE = DATASET_RELATIVE_ROOT + "/artifact_inventory.json"
SEAL_RELATIVE = DATASET_RELATIVE_ROOT + "/seal_manifest.json"
ATTEMPT_MARKER_RELATIVE = DATASET_RELATIVE_ROOT + "/attempt_started.json"

SUCCESS_REPORT = "reports/phase3_14b_r259_stagei_fresh_eval_seal_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r259_stagei_fresh_eval_seal_blocked_summary.json"

FRESH_SPLIT_NAME = "test"
FRESH_VISIBLE_SEED_START = 430000
FRESH_VISIBLE_SEED_COUNT = 128
FRESH_VISIBLE_SEED_STOP = FRESH_VISIBLE_SEED_START + FRESH_VISIBLE_SEED_COUNT
EXPECTED_EPISODE_COUNT = 2 * FRESH_VISIBLE_SEED_COUNT
EXPECTED_PAIR_GROUP_COUNT = FRESH_VISIBLE_SEED_COUNT
EXPECTED_CONDITIONS = ("free", "hidden_slack_breakaway_pin_v2")
EXPECTED_TASK = "ccda-slack-cable-v2"
EXPECTED_STATE_DIM = 87
EXPECTED_PAPER_X_DIM = 261
EXPECTED_STATE_ACTION_X_DIM = 303
EXPECTED_FUTURE_HORIZON = 4
EXPECTED_ACTION_DIM = 14
EXPECTED_HZ = 480

FALSE_BOUNDARIES = (
    "historical_dataset_loaded",
    "selection_holdout_reaccessed",
    "frozen_probe_reaccessed",
    "risk_fit_run",
    "direction_fit_run",
    "recipe_evaluation_run",
    "candidate_evaluation_run",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "phase4",
    "cps",
)


class StageIError(RuntimeError):
    """Fail-closed Stage-I error."""


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


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = compact_json_bytes(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    )
    return sha256_bytes(header + b"\0" + array.tobytes())


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageIError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageIError("write-once output exists: {}".format(target))
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


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=str(root))


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageIError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageIError("{} worktree is dirty".format(label))


def validate_stageh_report(path: Path) -> Mapping[str, Any]:
    report_path = Path(path)
    if not report_path.is_file():
        raise StageIError("Stage-H report is missing")
    if sha256_file(report_path) != STAGE_H_REPORT_SHA256:
        raise StageIError("Stage-H report file SHA changed")
    payload = load_json(report_path)
    required = {
        "schema": "phase314b_r259_stageh_frozen_probe_transfer_failure_audit_v1",
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r259_stageh_risk_probability_transfer_fails_beyond_"
            "prevalence_shift_while_raw_candidates_preserve_mean_improvement"
        ),
        "required_next_path": (
            "DESIGN_R259_RISK_TRANSFER_REPAIR_ON_OBJECTIVE_TRAIN_WITH_FRESH_"
            "UNTOUCHED_EVALUATION_SET"
        ),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "summary_sha256": STAGE_H_SELF_SHA256,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "frozen_probe_reaccessed": False,
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluation_count_added": 0,
        "selection_holdout_reaccessed": False,
        "rerun_authorized": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise StageIError("Stage-H report field changed: {}".format(key))
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise StageIError("Stage-H audit is missing")
    if audit.get("classification") != (
        "risk_probability_transfer_failure_beyond_base_rate_shift_with_raw_"
        "candidate_mean_improvement_preserved"
    ):
        raise StageIError("Stage-H classification changed")
    governance = audit.get("governance_conclusion")
    if not isinstance(governance, Mapping):
        raise StageIError("Stage-H governance conclusion is missing")
    expected_governance = {
        "frozen_probe_access_consumed": True,
        "frozen_probe_reaccess_authorized": False,
        "future_final_evaluation_requires_fresh_untouched_data": True,
        "recipe_fallback_authorized": False,
        "retuning_against_frozen_probe_authorized": False,
        "timestep_cherry_pick_authorized": False,
    }
    for key, expected in expected_governance.items():
        if governance.get(key) != expected:
            raise StageIError("Stage-H governance field changed: {}".format(key))
    recomputed = sha256_bytes(
        stable_json_bytes(
            {key: value for key, value in payload.items() if key != "summary_sha256"}
        )
    )
    if recomputed != STAGE_H_SELF_SHA256:
        raise StageIError("Stage-H summary self-hash changed")
    return payload


def validate_recovery_manifest(path: Path) -> Mapping[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise StageIError("Stage E-H recovery manifest is missing")
    if sha256_file(manifest_path) != RECOVERY_MANIFEST_SHA256:
        raise StageIError("Stage E-H recovery manifest SHA changed")
    payload = load_json(manifest_path)
    if payload.get("schema") != "phase314b_r259_stageeh_lost_instance_recovery_manifest_v1":
        raise StageIError("Stage E-H recovery manifest schema changed")
    if payload.get("recovery_kind") != "exact_files_new_git_identity":
        raise StageIError("Stage E-H recovery kind changed")
    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping):
        raise StageIError("Stage E-H recovery baseline is missing")
    if baseline.get("main_commit") != "6c1de2f5af23f2525414139a223d41b73037d74b":
        raise StageIError("Stage E-H recovery baseline commit changed")
    if baseline.get("submodule_commit") != EXPECTED_SUBMODULE:
        raise StageIError("Stage E-H recovery submodule changed")
    original = payload.get("original_lost_commits")
    recovered = payload.get("recovered_commits")
    if not isinstance(original, Mapping) or not isinstance(recovered, Mapping):
        raise StageIError("Stage E-H recovery commit mapping is missing")
    if original.get("stage_h_implementation") != ORIGINAL_STAGE_H_IMPLEMENTATION:
        raise StageIError("original Stage-H implementation identity changed")
    if original.get("stage_h_evidence") != ORIGINAL_STAGE_H_EVIDENCE:
        raise StageIError("original Stage-H evidence identity changed")
    if recovered.get("stage_h_implementation") != STAGE_H_IMPLEMENTATION:
        raise StageIError("recovered Stage-H implementation identity changed")
    report_hashes = payload.get("report_sha256")
    if not isinstance(report_hashes, Mapping) or report_hashes.get("stage_h") != STAGE_H_REPORT_SHA256:
        raise StageIError("recovered Stage-H report identity changed")
    governance = payload.get("governance")
    required_governance = {
        "science_rerun": False,
        "reports_regenerated": False,
        "source_files_exact_from_verified_packages": True,
        "report_files_exact_from_preserved_artifacts": True,
        "old_git_object_identity_recovered": False,
        "stage_g_frozen_probe_reaccessed": False,
        "selection_holdout_reaccessed": False,
    }
    if not isinstance(governance, Mapping):
        raise StageIError("Stage E-H recovery governance is missing")
    for key, expected in required_governance.items():
        if governance.get(key) != expected:
            raise StageIError("Stage E-H recovery governance changed: {}".format(key))
    return payload


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "parent": (_git(repo, "rev-parse", "HEAD^"), STAGE_H_EVIDENCE),
        "Stage-H evidence parent": (
            _git(repo, "rev-parse", STAGE_H_EVIDENCE + "^"),
            STAGE_H_IMPLEMENTATION,
        ),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise StageIError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if _git(repo, "show", "-s", "--format=%s", implementation_commit) != IMPLEMENTATION_SUBJECT:
        raise StageIError("Stage-I implementation subject changed")
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageIError("Stage-I implementation paths changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageIError("DeformableRavens worktree commit changed")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(submodule, "DeformableRavens")
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageIError("Stage-I source is missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageIError("Stage-I source differs from commit: {}".format(relative))
    stageh_path = repo / STAGE_H_REPORT
    committed_stageh = _git_bytes(repo, "show", "{}:{}".format(STAGE_H_EVIDENCE, STAGE_H_REPORT))
    if committed_stageh != stageh_path.read_bytes():
        raise StageIError("Stage-H report differs from evidence commit")
    validate_stageh_report(stageh_path)
    recovery_path = repo / RECOVERY_MANIFEST
    committed_recovery = _git_bytes(
        repo, "show", "{}:{}".format(STAGE_H_EVIDENCE, RECOVERY_MANIFEST)
    )
    if committed_recovery != recovery_path.read_bytes():
        raise StageIError("Stage E-H recovery manifest differs from evidence commit")
    validate_recovery_manifest(recovery_path)
    dataset_root = repo / DATASET_RELATIVE_ROOT
    if dataset_root.exists():
        raise StageIError("fresh evaluation dataset root already exists")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageIError("Stage-I report already exists: {}".format(relative))
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": STAGE_H_EVIDENCE,
        "stage_h_implementation": STAGE_H_IMPLEMENTATION,
        "stage_h_evidence": STAGE_H_EVIDENCE,
        "original_stage_h_implementation": ORIGINAL_STAGE_H_IMPLEMENTATION,
        "original_stage_h_evidence": ORIGINAL_STAGE_H_EVIDENCE,
        "recovery_manifest": RECOVERY_MANIFEST,
        "recovery_manifest_sha256": RECOVERY_MANIFEST_SHA256,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def expected_visible_seeds() -> Tuple[int, ...]:
    return tuple(range(FRESH_VISIBLE_SEED_START, FRESH_VISIBLE_SEED_STOP))


def validate_seed_contract() -> Mapping[str, Any]:
    seeds = expected_visible_seeds()
    if len(seeds) != FRESH_VISIBLE_SEED_COUNT:
        raise StageIError("fresh seed count changed")
    historical_ranges = (
        (400000, 400256),
        (410000, 410064),
        (420000, 420128),
    )
    for start, stop in historical_ranges:
        if set(seeds).intersection(range(start, stop)):
            raise StageIError("fresh seed range overlaps historical dataset")
    return {
        "split_name": FRESH_SPLIT_NAME,
        "visible_seed_start": FRESH_VISIBLE_SEED_START,
        "visible_seed_stop_exclusive": FRESH_VISIBLE_SEED_STOP,
        "visible_seed_count": FRESH_VISIBLE_SEED_COUNT,
        "visible_seeds_sha256": sha256_bytes(
            compact_json_bytes(list(seeds))
        ),
        "historical_seed_overlap": False,
    }


def _strict_manifest_fields(value: Mapping[str, Any]) -> None:
    required = {
        "environment_semantics_version": "ccda_hidden_slack_breakaway_v2",
        "observation_schema_version": "ccda_state_v2_position_proprio",
        "task": EXPECTED_TASK,
        "split": FRESH_SPLIT_NAME,
        "action_source_condition": "free",
        "contains_simulator_bead_velocity": False,
        "input_canonicalization": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise StageIError("raw manifest field changed: {}".format(key))


def audit_raw_dataset(
    raw_root: Path,
    *,
    implementation_commit: str,
    submodule_commit: str,
) -> Mapping[str, Any]:
    root = Path(raw_root)
    manifests = sorted(root.glob("*/*/*.manifest.json"))
    pickles = sorted(root.glob("*/*/*.pkl"))
    if len(manifests) != EXPECTED_EPISODE_COUNT:
        raise StageIError("fresh raw manifest count changed")
    if len(pickles) != EXPECTED_EPISODE_COUNT:
        raise StageIError("fresh raw episode count changed")
    expected_seed_set = set(expected_visible_seeds())
    observed: Dict[str, set] = {condition: set() for condition in EXPECTED_CONDITIONS}
    pair_action_hashes: Dict[int, set] = {seed: set() for seed in expected_seed_set}
    pair_groups: set = set()
    total_actions = 0
    for path in manifests:
        value = load_json(path)
        _strict_manifest_fields(value)
        condition = str(value.get("condition"))
        if condition not in EXPECTED_CONDITIONS:
            raise StageIError("unexpected fresh condition")
        seed = int(value.get("visible_seed"))
        if seed not in expected_seed_set:
            raise StageIError("fresh visible seed outside contract")
        if value.get("main_commit") != implementation_commit:
            raise StageIError("fresh episode main commit changed")
        if value.get("submodule_commit") != submodule_commit:
            raise StageIError("fresh episode submodule commit changed")
        pair_group = str(value.get("pair_group"))
        expected_pair_group = "phase313_{}_seed_{}".format(FRESH_SPLIT_NAME, seed)
        if pair_group != expected_pair_group:
            raise StageIError("fresh pair group changed")
        observed[condition].add(seed)
        pair_groups.add(pair_group)
        pair_action_hashes[seed].add(str(value.get("action_sequence_sha256")))
        num_actions = int(value.get("num_actions", 0))
        if num_actions <= 0:
            raise StageIError("fresh episode has no actions")
        total_actions += num_actions
    for condition in EXPECTED_CONDITIONS:
        if observed[condition] != expected_seed_set:
            raise StageIError("fresh condition seed coverage changed")
    if len(pair_groups) != EXPECTED_PAIR_GROUP_COUNT:
        raise StageIError("fresh pair-group count changed")
    if any(len(values) != 1 for values in pair_action_hashes.values()):
        raise StageIError("paired action sequence hashes differ")
    orphan_temporaries = sorted(root.glob("**/*.tmp"))
    if orphan_temporaries:
        raise StageIError("orphan temporary files remain in fresh raw root")
    return {
        "episode_count": len(pickles),
        "manifest_count": len(manifests),
        "pair_group_count": len(pair_groups),
        "condition_seed_counts": {
            condition: len(observed[condition]) for condition in EXPECTED_CONDITIONS
        },
        "paired_action_hashes_match": True,
        "total_episode_action_count": total_actions,
        "orphan_temporary_count": 0,
    }


def _strings(values: Iterable[str]) -> np.ndarray:
    rendered = [str(value) for value in values]
    width = max([1] + [len(value) for value in rendered])
    return np.asarray(rendered, dtype="<U{}".format(width))


def build_fresh_windows(root: Path) -> Mapping[str, Any]:
    from ccda_phase3.data_io import build_windows_from_dataset, save_action_template
    from ccda_phase3.schema_v2 import (
        ACTION_DIM,
        DEFAULT_TF,
        DEFAULT_TH,
        FORMAL_CONDITIONS,
        PAPER_X_DIM,
        STATE_ACTION_X_DIM,
        STATE_DIM,
    )

    if (
        STATE_DIM != EXPECTED_STATE_DIM
        or PAPER_X_DIM != EXPECTED_PAPER_X_DIM
        or STATE_ACTION_X_DIM != EXPECTED_STATE_ACTION_X_DIM
        or DEFAULT_TF != EXPECTED_FUTURE_HORIZON
        or ACTION_DIM != EXPECTED_ACTION_DIM
        or tuple(FORMAL_CONDITIONS) != EXPECTED_CONDITIONS
    ):
        raise StageIError("fresh window schema changed")
    repo = Path(root).resolve()
    raw_split = repo / RAW_RELATIVE_ROOT / FRESH_SPLIT_NAME
    rows, codec, sources = build_windows_from_dataset(
        FRESH_SPLIT_NAME,
        raw_split,
        DEFAULT_TH,
        DEFAULT_TF,
        codec=None,
        max_windows_per_episode=0,
        conditions=EXPECTED_CONDITIONS,
    )
    if not rows or codec is None:
        raise StageIError("fresh dataset produced no windows")
    arrays: Dict[str, np.ndarray] = {
        "paper_x": np.stack([row["paper_x"] for row in rows]).astype(np.float32),
        "state_action_x": np.stack([row["state_action_x"] for row in rows]).astype(np.float32),
        "y_state": np.stack([row["y_state"] for row in rows]).astype(np.float32),
        "y_final_state": np.stack([row["y_final_state"] for row in rows]).astype(np.float32),
        "y_action": np.stack([row["y_action"] for row in rows]).astype(np.float32),
        "condition_name": _strings(row["condition_name"] for row in rows),
        "visible_seed": np.asarray([row["visible_seed"] for row in rows], dtype=np.int64),
        "split_name": _strings(row["split_name"] for row in rows),
        "source_file": _strings(row["source_file"] for row in rows),
        "pair_group": _strings(row["pair_group"] for row in rows),
        "window_t": np.asarray([row["window_t"] for row in rows], dtype=np.int64),
        "success": np.asarray([row["success"] for row in rows], dtype=np.bool_),
        "final_fraction": np.asarray([row["final_fraction"] for row in rows], dtype=np.float32),
        "engagement_step": np.asarray([row["engagement_step"] for row in rows], dtype=np.int64),
        "release_step": np.asarray([row["release_step"] for row in rows], dtype=np.int64),
        "pre_engagement": np.asarray([row["pre_engagement"] for row in rows], dtype=np.bool_),
    }
    expected_tails = {
        "paper_x": (EXPECTED_PAPER_X_DIM,),
        "state_action_x": (EXPECTED_STATE_ACTION_X_DIM,),
        "y_state": (EXPECTED_FUTURE_HORIZON, EXPECTED_STATE_DIM),
        "y_final_state": (EXPECTED_STATE_DIM,),
        "y_action": (EXPECTED_ACTION_DIM,),
    }
    for name, tail in expected_tails.items():
        value = arrays[name]
        if value.shape[1:] != tail or value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise StageIError("invalid fresh window array: {}".format(name))
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise StageIError("object array detected in fresh windows")
    seeds = set(arrays["visible_seed"].tolist())
    if seeds != set(expected_visible_seeds()):
        raise StageIError("fresh window seed coverage changed")
    groups = set(arrays["pair_group"].astype(str).tolist())
    if len(groups) != EXPECTED_PAIR_GROUP_COUNT:
        raise StageIError("fresh window pair-group count changed")
    if set(arrays["condition_name"].astype(str).tolist()) != set(EXPECTED_CONDITIONS):
        raise StageIError("fresh window condition coverage changed")
    if set(arrays["split_name"].astype(str).tolist()) != {FRESH_SPLIT_NAME}:
        raise StageIError("fresh window split name changed")

    windows_dir = repo / WINDOWS_RELATIVE_ROOT
    windows_dir.mkdir(parents=True, exist_ok=False)
    template_path = repo / ACTION_TEMPLATE_RELATIVE
    temporary_template = template_path.with_name(".action_template.tmp.pkl")
    save_action_template(temporary_template, codec)
    os.replace(str(temporary_template), str(template_path))
    npz_path = repo / WINDOW_NPZ_RELATIVE
    temporary_npz = npz_path.with_name(".fresh_eval_windows.tmp.npz")
    np.savez_compressed(str(temporary_npz), **arrays)
    with np.load(str(temporary_npz), allow_pickle=False) as checked:
        if set(checked.files) != set(arrays):
            raise StageIError("fresh temporary NPZ key mismatch")
        for name in arrays:
            if checked[name].shape != arrays[name].shape or checked[name].dtype != arrays[name].dtype:
                raise StageIError("fresh temporary NPZ array changed: {}".format(name))
    os.replace(str(temporary_npz), str(npz_path))
    manifest = {
        "schema": "phase314b_r259_stagei_fresh_window_manifest_v1",
        "split_name": FRESH_SPLIT_NAME,
        "task": EXPECTED_TASK,
        "conditions": list(EXPECTED_CONDITIONS),
        "visible_seed_start": FRESH_VISIBLE_SEED_START,
        "visible_seed_stop_exclusive": FRESH_VISIBLE_SEED_STOP,
        "visible_seed_count": FRESH_VISIBLE_SEED_COUNT,
        "row_count": int(arrays["paper_x"].shape[0]),
        "pair_group_count": len(groups),
        "dimensions": {
            "state": EXPECTED_STATE_DIM,
            "paper_x": EXPECTED_PAPER_X_DIM,
            "state_action_x": EXPECTED_STATE_ACTION_X_DIM,
            "future": [EXPECTED_FUTURE_HORIZON, EXPECTED_STATE_DIM],
            "action": EXPECTED_ACTION_DIM,
        },
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "array_sha256": {name: sha256_array(value) for name, value in arrays.items()},
        "window_npz": WINDOW_NPZ_RELATIVE,
        "window_npz_sha256": sha256_file(npz_path),
        "action_template": ACTION_TEMPLATE_RELATIVE,
        "action_template_sha256": sha256_file(template_path),
        "robot_proxy_sources": dict(sources),
        "object_dtype_count": 0,
        "historical_dataset_loaded": False,
        "selection_holdout_loaded": False,
        "frozen_probe_loaded": False,
        "model_fit_or_evaluation_performed": False,
    }
    atomic_write_once(repo / WINDOW_MANIFEST_RELATIVE, stable_json_bytes(manifest))
    return manifest


def _inventory_paths(dataset_root: Path) -> List[Path]:
    root = Path(dataset_root)
    excluded = {
        root / INVENTORY_RELATIVE.split("/", 2)[-1],
        root / SEAL_RELATIVE.split("/", 2)[-1],
    }
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path not in excluded and not path.name.endswith(".tmp")
    ]
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def build_inventory(dataset_root: Path) -> Mapping[str, Any]:
    root = Path(dataset_root).resolve()
    records = []
    for path in _inventory_paths(root):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise StageIError("fresh artifact inventory is empty")
    inventory = {
        "schema": "phase314b_r259_stagei_fresh_artifact_inventory_v1",
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "records": records,
    }
    inventory_sha = sha256_bytes(stable_json_bytes(inventory))
    inventory = dict(inventory)
    inventory["inventory_sha256"] = inventory_sha
    return inventory


def validate_inventory(dataset_root: Path, inventory: Mapping[str, Any]) -> None:
    root = Path(dataset_root).resolve()
    records = inventory.get("records")
    if not isinstance(records, Sequence) or not records:
        raise StageIError("fresh inventory records are missing")
    observed_paths = []
    for record in records:
        if not isinstance(record, Mapping):
            raise StageIError("fresh inventory record is invalid")
        relative = str(record.get("path"))
        path = root / relative
        if not path.is_file():
            raise StageIError("fresh inventory file is missing: {}".format(relative))
        if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
            raise StageIError("fresh inventory size changed: {}".format(relative))
        if str(record.get("sha256")) != sha256_file(path):
            raise StageIError("fresh inventory SHA changed: {}".format(relative))
        observed_paths.append(relative)
    expected_paths = [path.relative_to(root).as_posix() for path in _inventory_paths(root)]
    if observed_paths != expected_paths:
        raise StageIError("fresh inventory path population changed")
    base = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if inventory.get("inventory_sha256") != sha256_bytes(stable_json_bytes(base)):
        raise StageIError("fresh inventory self-hash changed")


def build_seal(
    *,
    root: Path,
    repository: Mapping[str, Any],
    raw_audit: Mapping[str, Any],
    window_manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    dataset_root = repo / DATASET_RELATIVE_ROOT
    validate_inventory(dataset_root, inventory)
    seal = {
        "schema": "phase314b_r259_stagei_fresh_evaluation_seal_v1",
        "dataset_role": "fresh_untouched_final_risk_transfer_evaluation",
        "dataset_relative_root": DATASET_RELATIVE_ROOT,
        "repository": dict(repository),
        "seed_contract": validate_seed_contract(),
        "generation_contract": {
            "task": EXPECTED_TASK,
            "conditions": list(EXPECTED_CONDITIONS),
            "paired_action_source": "free",
            "paired_replay_condition": "hidden_slack_breakaway_pin_v2",
            "hz": EXPECTED_HZ,
            "episode_count": EXPECTED_EPISODE_COUNT,
            "pair_group_count": EXPECTED_PAIR_GROUP_COUNT,
        },
        "raw_audit": dict(raw_audit),
        "window_manifest_path": WINDOW_MANIFEST_RELATIVE,
        "window_manifest_sha256": sha256_file(repo / WINDOW_MANIFEST_RELATIVE),
        "window_row_count": int(window_manifest["row_count"]),
        "window_pair_group_count": int(window_manifest["pair_group_count"]),
        "window_npz_path": WINDOW_NPZ_RELATIVE,
        "window_npz_sha256": str(window_manifest["window_npz_sha256"]),
        "inventory_path": INVENTORY_RELATIVE,
        "inventory_sha256": str(inventory["inventory_sha256"]),
        "inventory_file_count": int(inventory["file_count"]),
        "inventory_total_size_bytes": int(inventory["total_size_bytes"]),
        "governance": {
            "generated_before_new_risk_repair_search": True,
            "targets_not_used_for_fit_selection_or_evaluation": True,
            "historical_selection_holdout_reaccessed": False,
            "historical_frozen_probe_reaccessed": False,
            "future_evaluation_count": 0,
            "future_evaluation_must_be_one_shot": True,
            "future_evaluation_requires_independently_locked_policy": True,
            "post_evaluation_retuning_forbidden": True,
            "fallback_after_evaluation_forbidden": True,
        },
    }
    seal["seal_sha256"] = sha256_bytes(stable_json_bytes(seal))
    validate_seal(seal)
    return seal


def validate_seal(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "phase314b_r259_stagei_fresh_evaluation_seal_v1":
        raise StageIError("fresh seal schema changed")
    if payload.get("dataset_role") != "fresh_untouched_final_risk_transfer_evaluation":
        raise StageIError("fresh dataset role changed")
    if payload.get("dataset_relative_root") != DATASET_RELATIVE_ROOT:
        raise StageIError("fresh dataset root changed")
    seed_contract = payload.get("seed_contract")
    if not isinstance(seed_contract, Mapping) or dict(seed_contract) != dict(validate_seed_contract()):
        raise StageIError("fresh seed contract changed")
    governance = payload.get("governance")
    if not isinstance(governance, Mapping):
        raise StageIError("fresh seal governance is missing")
    for key in (
        "generated_before_new_risk_repair_search",
        "targets_not_used_for_fit_selection_or_evaluation",
        "future_evaluation_must_be_one_shot",
        "future_evaluation_requires_independently_locked_policy",
        "post_evaluation_retuning_forbidden",
        "fallback_after_evaluation_forbidden",
    ):
        if governance.get(key) is not True:
            raise StageIError("fresh seal governance changed: {}".format(key))
    for key in (
        "historical_selection_holdout_reaccessed",
        "historical_frozen_probe_reaccessed",
    ):
        if governance.get(key) is not False:
            raise StageIError("fresh seal historical access changed: {}".format(key))
    if governance.get("future_evaluation_count") != 0:
        raise StageIError("fresh evaluation count changed")
    recomputed = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != "seal_sha256"})
    )
    if payload.get("seal_sha256") != recomputed:
        raise StageIError("fresh seal self-hash changed")


def build_summary(
    *,
    repository: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_seal(seal)
    summary: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "primary_failure_locus": "fresh_untouched_evaluation_surface_sealed",
        "root_cause": (
            "phase314b_r259_stagei_fresh_untouched_risk_transfer_evaluation_"
            "surface_generated_and_sealed_before_repair_search"
        ),
        "required_next_path": (
            "DESIGN_R259_OBJECTIVE_TRAIN_PREVALENCE_ANCHORED_RISK_REPAIR_"
            "AGAINST_SEALED_FRESH_EVALUATION_CONTRACT"
        ),
        "repository": dict(repository),
        "fresh_evaluation_seal": dict(seal),
        "execution_counts": {
            "historical_report_read_count": 1,
            "fresh_visible_seed_count": FRESH_VISIBLE_SEED_COUNT,
            "fresh_episode_generation_count": EXPECTED_EPISODE_COUNT,
            "fresh_window_build_count": 1,
            "model_fit_count": 0,
            "model_evaluation_count": 0,
            "historical_dataset_load_count": 0,
            "selection_holdout_access_count": 0,
            "frozen_probe_access_count": 0,
            "fresh_evaluation_access_count": 0,
        },
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_evaluation_count_added": 0,
        "selection_holdout_reaccessed": False,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "frozen_probe_reaccessed": False,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
    }
    for key in FALSE_BOUNDARIES:
        summary[key] = False
    summary["summary_sha256"] = sha256_bytes(stable_json_bytes(summary))
    validate_summary(summary)
    return summary


def validate_summary(payload: Mapping[str, Any]) -> None:
    required = {
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "selection_holdout_reaccessed": False,
        "frozen_probe_evaluation_count_added": 0,
        "frozen_probe_reaccessed": False,
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "rerun_authorized": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise StageIError("Stage-I summary field changed: {}".format(key))
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageIError("Stage-I crossed a forbidden boundary")
    seal = payload.get("fresh_evaluation_seal")
    if not isinstance(seal, Mapping):
        raise StageIError("Stage-I summary seal is missing")
    validate_seal(seal)
    recomputed = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != "summary_sha256"})
    )
    if payload.get("summary_sha256") != recomputed:
        raise StageIError("Stage-I summary self-hash changed")


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
    dataset_root: Path,
) -> Mapping[str, Any]:
    root = Path(dataset_root)
    marker = root / "attempt_started.json"
    seal_path = root / "seal_manifest.json"
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stagei_fresh_evaluation_generation_or_seal_failed",
        "required_next_path": "AUDIT_R259_STAGEI_PARTIAL_FRESH_EVALUATION_ARTIFACTS_BEFORE_ANY_RETRY",
        "primary_failure_locus": "fresh_untouched_evaluation_acquisition",
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "dataset_relative_root": DATASET_RELATIVE_ROOT,
        "attempt_marker_present": marker.is_file(),
        "seal_present": seal_path.is_file(),
        "dataset_root_present": root.exists(),
        "partial_artifacts_may_exist": root.exists() and not seal_path.is_file(),
        "fresh_evaluation_count_added": 0,
        "cumulative_fresh_evaluation_count": 0,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "selection_holdout_reaccessed": False,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_reaccessed": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
