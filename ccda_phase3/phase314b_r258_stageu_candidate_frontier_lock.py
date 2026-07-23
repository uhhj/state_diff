"""Stage-U: lock the unique tolerance-aligned OOF candidate frontier.

Stage-T Resume1 established, on objective-train grouped OOF only, that all 27
candidate cells have mechanism support, 13 cells have fidelity support, and a
single backbone (``segment_target_rr64_feasible``) is both matrix-eligible and
Pareto-optimal across t=10/25/50.

Stage-U performs no model fit, callback execution, candidate reconstruction,
CUDA probing, holdout access, or probe access.  It validates the immutable
Stage-T Resume1 summary and gate contract, then writes a train-only frontier
lock that:

* freezes the unique backbone and all three timesteps without cherry-picking;
* freezes the three candidate/selected-scale identities and scalar metrics;
* freezes the tolerance-aligned shadow gate identity;
* pre-registers a one-shot selection-holdout policy for the next stage;
* keeps selected_configuration and train_only_recommendation null.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage U"
SCHEMA = "phase314b_r258_stageu_candidate_frontier_lock_v1"
CONTRACT_SCHEMA = "phase314b_r258_stageu_candidate_frontier_contract_v1"
BLOCKED_SCHEMA = "phase314b_r258_stageu_candidate_frontier_lock_blocked_v1"

BASE_STAGET_RESUME1_IMPLEMENTATION_COMMIT = (
    "1831f1b89340441425dc2f17516871a3626553f5"
)
BASE_STAGET_RESUME1_EVIDENCE_COMMIT = (
    "50592b176e0a53c4b7cc51fe6c834933fb7baae3"
)
EXPECTED_STAGET_RESUME1_IMPLEMENTATION_PARENT = (
    "92d6ff422f32dcca9a75e2ea082b4763d28c5767"
)
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

BASE_SUMMARY = (
    "reports/phase3_14b_r258_staget_resume1_oof_candidate_matrix_summary.json"
)
BASE_GATE = (
    "reports/phase3_14b_r258_staget_resume1_tolerance_aligned_gate_contract.json"
)
EXPECTED_BASE_SUMMARY_FILE_SHA256 = (
    "c817c7353b779521963cbdc85ac3b6d39fcea0ce780487b7623270d1f4e4c419"
)
EXPECTED_BASE_GATE_FILE_SHA256 = (
    "8e9170ba5fb9c2cd395f40d7842241b6542568538d50f2318df19a8bd8a5efbf"
)
EXPECTED_BASE_SCIENTIFIC_SHA256 = (
    "0e49d648c9303dcfa30d90a51ed8dd594843eddbeb9d1508ef5f7f93c5a554b3"
)
EXPECTED_RECOVERED_RESULT_SHA256 = (
    "dc842b79446d1ff5e616de8be4c1b609d6ec9ce105a2609fa26236cedff1ebd2"
)
EXPECTED_RECOVERED_GATE_SHA256 = (
    "cd7c2df97ad5ff70c98b290fa614e9eb556493b508c2533717288e1f2e3a1b1b"
)
EXPECTED_CANDIDATE_MATRIX_SHA256 = (
    "ed600a8f30dcef78f3b7fd8d5eaedf69c71273959ca24ca19cdf2a5529359f90"
)
EXPECTED_FUNCTIONAL_PROJECTION_SHA256 = (
    "52c2bff55ab15ff9514c5cf349006fd73cd7c5b78bf84e1a08f0aa6453e92fff"
)
EXPECTED_CURRENT_FIT_PROJECTION_SHA256 = (
    "cebd647b7748caa1510e924b2b9eca98ebe2273affc467725284c3fbfcf376b4"
)
EXPECTED_INNER_GATE_CONTRACT_SHA256 = (
    "e9bef0054321ea74178d46e754cc5b93f42f20372b35381471bb450b4aae1175"
)
EXPECTED_INNER_GATE_PAYLOAD_SHA256 = (
    "e37bfb5318757fa61ab0061a404794866d14c397dc1be9a5b3465af6c047dbdc"
)
EXPECTED_FRONTIER_CONTRACT_SHA256 = (
    "c0bf2c477694cf1bda77f11c7ac407befb77d005afa8422e7e9bae2616714a8c"
)

EXPECTED_BASE_ROOT_CAUSE = (
    "phase314b_r258_staget_tolerance_aligned_gate_and_candidate_matrix_confirmed"
)
EXPECTED_BASE_NEXT_PATH = (
    "LOCK_TOLERANCE_ALIGNED_OOF_CANDIDATE_FRONTIER_ON_OBJECTIVE_TRAIN_ONLY"
)
ROOT_CAUSE = (
    "phase314b_r258_stageu_unique_three_timestep_candidate_frontier_locked"
)
NEXT_PATH = (
    "EVALUATE_LOCKED_TOLERANCE_ALIGNED_FRONTIER_ON_SELECTION_HOLDOUT_ONCE"
)

FRONTIER_BACKBONE = "segment_target_rr64_feasible"
LOCKED_TIMESTEPS = (10, 25, 50)
EXPECTED_MECHANISM_ELIGIBLE_CELLS = 27
EXPECTED_FIDELITY_ELIGIBLE_CELLS = 13
EXPECTED_MATRIX_ELIGIBLE_BACKBONES = 1

EXPECTED_BACKBONE_METRICS: Mapping[str, Any] = {
    "base_direction_id": FRONTIER_BACKBONE,
    "rank": 1,
    "timesteps": [10, 25, 50],
    "fidelity_eligible_timestep_count": 3,
    "matrix_eligible": True,
    "mean_acceptance_rate": 0.973876698014629,
    "minimum_acceptance_rate": 0.9608150470219435,
    "mean_overall_mse_ratio": 0.9063149060700774,
    "worst_overall_mse_ratio": 0.9568765463967528,
    "mean_positive_distance_reduction_rate": 0.6659658788027247,
    "minimum_positive_distance_reduction_rate": 0.6182707993474714,
    "mean_relative_distance_reduction_mean": 0.0632756426378945,
    "minimum_relative_distance_reduction_mean": 0.03731045290964939,
}

EXPECTED_CELL_LOCKS: Mapping[int, Mapping[str, Any]] = {
    10: {
        "base_direction_id": FRONTIER_BACKBONE,
        "timestep": 10,
        "feature_mode": "full_segment_constraint",
        "global_rank": 8,
        "per_timestep_rank": 1,
        "aligned_acceptance_rate": 0.9608150470219435,
        "aligned_candidate_sha256": (
            "f6445496b0e4c21679e15b79e594efda3c0d69a9285ea9723502a1028c198e94"
        ),
        "aligned_selected_scale_sha256": (
            "afb73b4752727bd54f992e516bb01daa19dd0e5c863da2bf760c2473d11ff8b7"
        ),
        "aligned_selected_scale_histogram": {"0": 25, "2": 613},
        "row_count": 638,
        "selected_row_count": 613,
        "overall_mse_ratio": 0.9568765463967528,
        "accepted_row_mse_ratio": 0.954716946762974,
        "positive_distance_reduction_rate": 0.6182707993474714,
        "relative_distance_reduction_mean": 0.03731045290964939,
    },
    25: {
        "base_direction_id": FRONTIER_BACKBONE,
        "timestep": 25,
        "feature_mode": "full_segment_constraint",
        "global_rank": 1,
        "per_timestep_rank": 1,
        "aligned_acceptance_rate": 0.9811912225705329,
        "aligned_candidate_sha256": (
            "596d9f2dda51998c8a8bc06bb35248fb0dadb455359dd9c15f0ed70abe2a9a9a"
        ),
        "aligned_selected_scale_sha256": (
            "e009101313018a92d67cca8ea4a2b2c3e1bd8e4c7ac361b0ac54b68c3e704d19"
        ),
        "aligned_selected_scale_histogram": {"0": 12, "2": 626},
        "row_count": 638,
        "selected_row_count": 626,
        "overall_mse_ratio": 0.8631487559799662,
        "accepted_row_mse_ratio": 0.859470986196053,
        "positive_distance_reduction_rate": 0.7332268370607029,
        "relative_distance_reduction_mean": 0.09348730898776085,
    },
    50: {
        "base_direction_id": FRONTIER_BACKBONE,
        "timestep": 50,
        "feature_mode": "full_segment_constraint",
        "global_rank": 4,
        "per_timestep_rank": 1,
        "aligned_acceptance_rate": 0.9796238244514106,
        "aligned_candidate_sha256": (
            "2faa532053cd3ce8cfc678e2da223391c020f8ecadd6a35e3e742aabb88914e0"
        ),
        "aligned_selected_scale_sha256": (
            "4aaa6cc2430e75d08943f41ffeccbfffc7703c58dc102294b2c13395c3429899"
        ),
        "aligned_selected_scale_histogram": {"0": 13, "2": 625},
        "row_count": 638,
        "selected_row_count": 625,
        "overall_mse_ratio": 0.8989194158335131,
        "accepted_row_mse_ratio": 0.8950161130212291,
        "positive_distance_reduction_rate": 0.6464,
        "relative_distance_reduction_mean": 0.059029166016273255,
    },
}

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stageu_candidate_frontier_lock_summary.json"
)
CONTRACT_REPORT = (
    "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stageu_candidate_frontier_lock_blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage U: lock tolerance-aligned OOF candidate frontier"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage U candidate-frontier lock evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage U blocked evidence"
)
BASE_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage T Resume1: restore process-topology schema"
)
BASE_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage T Resume1 candidate-matrix evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r258_stageu_candidate_frontier_lock.py"),
    ("A", "scripts/phase3_14b_r258_stageu_execute.py"),
    ("A", "tests/test_phase3_14b_r258_stageu_candidate_frontier_lock.py"),
)
BASE_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/"
        "phase314b_r258_staget_resume1_process_topology_schema_recovery.py",
    ),
    ("A", "scripts/phase3_14b_r258_staget_resume1_execute.py"),
    (
        "A",
        "tests/"
        "test_phase3_14b_r258_staget_resume1_process_topology_schema_recovery.py",
    ),
)
BASE_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", BASE_SUMMARY),
    ("A", BASE_GATE),
)
BASE_SOURCE_SHA256: Mapping[str, str] = {
    "ccda_phase3/phase314b_r258_staget_resume1_process_topology_schema_recovery.py": (
        "e039b6d535d186fd65fcaf9e526a6de157fd0afcb3979abc626c23b0ed60fb7e"
    ),
    "scripts/phase3_14b_r258_staget_resume1_execute.py": (
        "c32a43594742503ccd7d8ef532e3562e8f56d4aff0c6a76313f14caf2fca5d08"
    ),
    "tests/test_phase3_14b_r258_staget_resume1_process_topology_schema_recovery.py": (
        "3648e15d13b240afb1a179abab6da4352789956908d93325618a5ba526c44037"
    ),
}

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

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "selection_holdout_evaluated",
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
    "surrogate_weights_persisted",
    "prediction_tensor_persisted",
    "direction_tensor_persisted",
    "candidate_tensor_persisted",
    "predicate_tensor_persisted",
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageUError(RuntimeError):
    """Fail-closed Stage-U error."""


def stable_json_bytes(value: Any) -> bytes:
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageUError(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageUError(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise StageUError(f"{label} is not a sequence")
    return value


def _required_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    if key not in mapping:
        raise StageUError(f"{label} missing: {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise StageUError(f"{label} is not an integer: {key}={value!r}")
    return int(value)


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageUError(
                f"{label} boundary changed: {key}={mapping.get(key)!r}"
            )


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageUError(f"deterministic environment mismatch: {mismatch}")
    return dict(EXPECTED_ENV)


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise StageUError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _run_git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        commit,
    )
    records = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageUError("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise StageUError(f"{label} worktree is dirty")


def _exact_float(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise StageUError(f"{label} is not numeric: {actual!r}")
    if float(actual) != float(expected):
        raise StageUError(
            f"{label} changed: actual={actual!r} expected={expected!r}"
        )


def _exact_mapping_subset(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise StageUError(f"{label} missing: {key}")
        actual_value = actual[key]
        if isinstance(expected_value, float):
            _exact_float(actual_value, expected_value, f"{label}.{key}")
        elif actual_value != expected_value:
            raise StageUError(
                f"{label}.{key} changed: "
                f"actual={actual_value!r} expected={expected_value!r}"
            )


def validate_gate_wrapper(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    if gate.get("execution_verdict") != "PASS":
        raise StageUError("Stage-T Resume1 gate execution did not PASS")
    if gate.get("scientific_status") != "BLOCKED":
        raise StageUError("Stage-T Resume1 gate scientific status changed")
    if gate.get("schema_recovery_only") is not True:
        raise StageUError("Stage-T Resume1 gate is not schema-recovery-only")
    if gate.get("recovered_stage_t_gate_contract_sha256") != (
        EXPECTED_RECOVERED_GATE_SHA256
    ):
        raise StageUError("Stage-T Resume1 recovered gate SHA changed")
    if gate.get("selected_configuration") is not None:
        raise StageUError("Stage-T Resume1 gate selected a configuration")
    if gate.get("train_only_recommendation") is not None:
        raise StageUError("Stage-T Resume1 gate emitted a recommendation")
    require_false(gate, FALSE_BOUNDARIES, "Stage-T Resume1 gate wrapper")

    inner = _mapping(
        gate.get("recovered_stage_t_gate_contract"),
        "recovered Stage-T gate contract",
    )
    expected = {
        "schema": "phase314b_r258_staget_tolerance_aligned_gate_contract_v1",
        "contract_role": "shadow_frozen_not_written_to_stagee",
        "contract_sha256": EXPECTED_INNER_GATE_CONTRACT_SHA256,
        "contract_payload_sha256": EXPECTED_INNER_GATE_PAYLOAD_SHA256,
        "tolerance_factor": 5.0,
        "segment_tolerance": 2.5e-06,
        "length_tolerance": 1.25e-05,
        "position_shape": [4, 23],
        "upper_bound_sha256": (
            "afff5ceab890fed4cf83c015a0bfab6071080b562d717453abf51887c643bb43"
        ),
        "length_upper_sha256": (
            "5159d02ef16ef772e6a75b60ac179adbc6907541f56f167162ca2f88d986a1fc"
        ),
        "position_z_threshold_sha256": (
            "02175da1e304ed1d3a220fe665eafe7805aedaab556597cdfd4d45ad0affbabc"
        ),
        "objective_train_only": True,
        "worker_contract_byte_exact": True,
        "aligned_gate_written_to_stagee": False,
        "legacy_upper_gate_changed": False,
        "stagee_modified": False,
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    _exact_mapping_subset(inner, expected, "recovered Stage-T gate contract")
    require_false(inner, FALSE_BOUNDARIES, "recovered Stage-T gate contract")
    return inner


def _candidate_cell_map(matrix: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    records = _sequence(matrix.get("cell_records"), "candidate cell records")
    selected: Dict[int, Mapping[str, Any]] = {}
    for index, value in enumerate(records):
        record = _mapping(value, f"candidate cell {index}")
        if record.get("base_direction_id") != FRONTIER_BACKBONE:
            continue
        timestep = _required_int(record, "timestep", f"candidate cell {index}")
        if timestep in selected:
            raise StageUError(f"duplicate frontier cell at timestep {timestep}")
        selected[timestep] = record
    if tuple(sorted(selected)) != LOCKED_TIMESTEPS:
        raise StageUError(
            f"frontier timestep population changed: {tuple(sorted(selected))}"
        )
    return selected


def _per_timestep_rank(matrix: Mapping[str, Any], timestep: int) -> int:
    rankings = _mapping(matrix.get("per_timestep_ranking"), "per-timestep ranking")
    rows = _sequence(rankings.get(str(timestep)), f"t={timestep} ranking")
    matches = [
        _mapping(value, f"t={timestep} ranking row")
        for value in rows
        if isinstance(value, Mapping)
        and value.get("base_direction_id") == FRONTIER_BACKBONE
    ]
    if len(matches) != 1:
        raise StageUError(f"frontier ranking population changed at t={timestep}")
    rank = _required_int(matches[0], "rank", f"t={timestep} frontier ranking")
    if rank != 1:
        raise StageUError(f"frontier is not rank 1 at t={timestep}: rank={rank}")
    if matches[0].get("fidelity_eligible") is not True:
        raise StageUError(f"frontier lost fidelity eligibility at t={timestep}")
    return rank


def validate_frontier_cell(
    record: Mapping[str, Any],
    *,
    timestep: int,
    per_timestep_rank: int,
) -> Mapping[str, Any]:
    expected = EXPECTED_CELL_LOCKS[timestep]
    if record.get("matrix_reconstruction_exact") is not True:
        raise StageUError(f"frontier matrix reconstruction changed at t={timestep}")
    if record.get("legacy_functional_identity_exact") is not True:
        raise StageUError(f"frontier legacy identity changed at t={timestep}")
    for key in (
        "length_log_z_element_mismatch_count",
        "aligned_upper_element_failure_count",
        "strict_pass_aligned_fail_row_count",
    ):
        if _required_int(record, key, f"frontier t={timestep}") != 0:
            raise StageUError(f"frontier gate regression at t={timestep}: {key}")
    if _required_int(
        record,
        "matrix_reconstruction_attempt_count",
        f"frontier t={timestep}",
    ) != 7:
        raise StageUError(f"frontier reconstruction attempt count changed at t={timestep}")

    fidelity = _mapping(
        record.get("candidate_fidelity"), f"frontier fidelity t={timestep}"
    )
    if fidelity.get("mechanism_eligible") is not True:
        raise StageUError(f"frontier mechanism eligibility changed at t={timestep}")
    if fidelity.get("fidelity_eligible") is not True:
        raise StageUError(f"frontier fidelity eligibility changed at t={timestep}")
    accepted = _mapping(
        fidelity.get("accepted_rows"), f"frontier accepted rows t={timestep}"
    )
    relative = _mapping(
        accepted.get("relative_distance_reduction"),
        f"frontier relative reduction t={timestep}",
    )
    lock = {
        "base_direction_id": record.get("base_direction_id"),
        "timestep": _required_int(record, "timestep", "frontier timestep"),
        "feature_mode": record.get("feature_mode"),
        "global_rank": _required_int(record, "global_rank", "frontier global rank"),
        "per_timestep_rank": int(per_timestep_rank),
        "aligned_acceptance_rate": record.get("aligned_acceptance_rate"),
        "aligned_candidate_sha256": record.get("aligned_candidate_sha256"),
        "aligned_selected_scale_sha256": record.get(
            "aligned_selected_scale_sha256"
        ),
        "aligned_selected_scale_histogram": copy.deepcopy(
            record.get("aligned_selected_scale_histogram")
        ),
        "row_count": _required_int(fidelity, "row_count", "frontier row count"),
        "selected_row_count": _required_int(
            fidelity, "selected_row_count", "frontier selected row count"
        ),
        "overall_mse_ratio": fidelity.get("overall_mse_ratio"),
        "accepted_row_mse_ratio": accepted.get("mse_ratio"),
        "positive_distance_reduction_rate": accepted.get(
            "positive_distance_reduction_rate"
        ),
        "relative_distance_reduction_mean": relative.get("mean"),
        "mechanism_eligible": True,
        "fidelity_eligible": True,
        "matrix_reconstruction_exact": True,
        "legacy_functional_identity_exact": True,
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
    }
    _exact_mapping_subset(lock, expected, f"frontier lock t={timestep}")
    if sum(int(value) for value in lock["aligned_selected_scale_histogram"].values()) != (
        lock["row_count"]
    ):
        raise StageUError(f"selected-scale histogram population changed at t={timestep}")
    if lock["selected_row_count"] != int(
        lock["aligned_selected_scale_histogram"].get("2", -1)
    ):
        raise StageUError(f"selected-row/scale population changed at t={timestep}")
    return lock


def validate_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if summary.get("execution_verdict") != "PASS":
        raise StageUError("Stage-T Resume1 execution did not PASS")
    if summary.get("scientific_status") != "BLOCKED":
        raise StageUError("Stage-T Resume1 scientific status changed")
    if summary.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageUError("Stage-T Resume1 root cause changed")
    if summary.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageUError("Stage-T Resume1 next path changed")
    if summary.get("scientific_result_sha256") != EXPECTED_BASE_SCIENTIFIC_SHA256:
        raise StageUError("Stage-T Resume1 scientific SHA changed")
    if summary.get("recovered_stage_t_result_sha256") != EXPECTED_RECOVERED_RESULT_SHA256:
        raise StageUError("Stage-T Resume1 recovered-result SHA changed")
    if summary.get("recovered_stage_t_gate_contract_sha256") != EXPECTED_RECOVERED_GATE_SHA256:
        raise StageUError("Stage-T Resume1 recovered-gate SHA changed")
    if summary.get("selected_configuration") is not None:
        raise StageUError("Stage-T Resume1 selected a configuration")
    if summary.get("train_only_recommendation") is not None:
        raise StageUError("Stage-T Resume1 emitted a recommendation")
    require_false(summary, FALSE_BOUNDARIES, "Stage-T Resume1 summary")

    result = _mapping(summary.get("recovered_stage_t_result"), "recovered Stage-T result")
    if result.get("execution_verdict") != "PASS":
        raise StageUError("recovered Stage-T result did not PASS")
    if result.get("scientific_status") != "BLOCKED":
        raise StageUError("recovered Stage-T scientific status changed")
    if result.get("root_cause") != EXPECTED_BASE_ROOT_CAUSE:
        raise StageUError("recovered Stage-T root cause changed")
    if result.get("required_next_path") != EXPECTED_BASE_NEXT_PATH:
        raise StageUError("recovered Stage-T next path changed")
    if result.get("selected_configuration") is not None:
        raise StageUError("recovered Stage-T selected a configuration")
    if result.get("train_only_recommendation") is not None:
        raise StageUError("recovered Stage-T emitted a recommendation")
    require_false(result, FALSE_BOUNDARIES, "recovered Stage-T result")

    execution = _mapping(result.get("confirmation_execution"), "Stage-T execution")
    exact_execution = {
        "environment_probe_count": 1,
        "cold_science_worker_count": 2,
        "processes_distinct": True,
        "workers_sequential": True,
        "total_oof_fit_count": 108,
        "total_callback_pair_count": 54,
        "total_shadow_internal_scale_attempt_count": 378,
        "total_matrix_reconstruction_attempt_count": 378,
        "oracle_callback_rerun_count": 0,
        "worker_outputs_persisted": False,
        "candidate_matrix_sha256": EXPECTED_CANDIDATE_MATRIX_SHA256,
        "functional_projection_sha256": EXPECTED_FUNCTIONAL_PROJECTION_SHA256,
        "current_fit_projection_sha256": EXPECTED_CURRENT_FIT_PROJECTION_SHA256,
        "gate_contract_sha256": EXPECTED_INNER_GATE_CONTRACT_SHA256,
    }
    _exact_mapping_subset(execution, exact_execution, "Stage-T execution")
    comparison = _mapping(execution.get("worker_comparison"), "worker comparison")
    if comparison.get("all_exact") is not True:
        raise StageUError("Stage-T worker comparison changed")

    matrix = _mapping(result.get("candidate_matrix"), "candidate matrix")
    exact_matrix = {
        "backbone_count": 9,
        "timestep_count": 3,
        "cell_count": 27,
        "mechanism_eligible_cell_count": EXPECTED_MECHANISM_ELIGIBLE_CELLS,
        "fidelity_eligible_cell_count": EXPECTED_FIDELITY_ELIGIBLE_CELLS,
        "matrix_eligible_backbone_count": EXPECTED_MATRIX_ELIGIBLE_BACKBONES,
        "matrix_eligible_backbones": [FRONTIER_BACKBONE],
        "pareto_frontier_backbones": [FRONTIER_BACKBONE],
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    _exact_mapping_subset(matrix, exact_matrix, "candidate matrix")
    frontier = _sequence(result.get("candidate_matrix_frontier"), "candidate frontier")
    if list(frontier) != [FRONTIER_BACKBONE]:
        raise StageUError(f"candidate frontier changed: {list(frontier)!r}")

    rankings = _sequence(matrix.get("backbone_ranking"), "backbone ranking")
    frontier_rankings = [
        _mapping(value, "frontier backbone ranking")
        for value in rankings
        if isinstance(value, Mapping)
        and value.get("base_direction_id") == FRONTIER_BACKBONE
    ]
    if len(frontier_rankings) != 1:
        raise StageUError("frontier backbone ranking population changed")
    _exact_mapping_subset(
        frontier_rankings[0], EXPECTED_BACKBONE_METRICS, "frontier backbone ranking"
    )

    cell_map = _candidate_cell_map(matrix)
    locked_cells = []
    for timestep in LOCKED_TIMESTEPS:
        rank = _per_timestep_rank(matrix, timestep)
        locked_cells.append(
            validate_frontier_cell(
                cell_map[timestep],
                timestep=timestep,
                per_timestep_rank=rank,
            )
        )
    return {
        "result": result,
        "execution": execution,
        "matrix": matrix,
        "frontier_backbone_metrics": copy.deepcopy(dict(frontier_rankings[0])),
        "locked_cells": locked_cells,
    }


def build_frontier_contract(
    *,
    validated_summary: Mapping[str, Any],
    gate_inner: Mapping[str, Any],
) -> Mapping[str, Any]:
    cells = copy.deepcopy(list(validated_summary["locked_cells"]))
    contract: Dict[str, Any] = {
        "phase": PHASE,
        "schema": CONTRACT_SCHEMA,
        "contract_role": "objective_train_only_frontier_lock",
        "objective_train_only": True,
        "frontier_lock": {
            "backbone_id": FRONTIER_BACKBONE,
            "timesteps": list(LOCKED_TIMESTEPS),
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "backbone_search_reopened": False,
            "timestep_search_reopened": False,
            "fallback_backbone_allowed": False,
            "fallback_timestep_allowed": False,
            "matrix_eligible_backbone_count": 1,
            "pareto_frontier_backbone_count": 1,
            "frontier_is_unique": True,
            "all_locked_timesteps_fidelity_eligible": True,
            "frontier_backbone_metrics": copy.deepcopy(
                dict(validated_summary["frontier_backbone_metrics"])
            ),
            "locked_cells": cells,
            "locked_cells_sha256": sha256_bytes(stable_json_bytes(cells)),
        },
        "aligned_gate_lock": {
            "source_gate_contract_sha256": EXPECTED_RECOVERED_GATE_SHA256,
            "inner_gate_contract_sha256": EXPECTED_INNER_GATE_CONTRACT_SHA256,
            "inner_gate_payload_sha256": EXPECTED_INNER_GATE_PAYLOAD_SHA256,
            "contract_role": gate_inner["contract_role"],
            "tolerance_factor": gate_inner["tolerance_factor"],
            "segment_tolerance": gate_inner["segment_tolerance"],
            "length_tolerance": gate_inner["length_tolerance"],
            "position_shape": copy.deepcopy(gate_inner["position_shape"]),
            "upper_bound_sha256": gate_inner["upper_bound_sha256"],
            "length_upper_sha256": gate_inner["length_upper_sha256"],
            "position_z_threshold_sha256": gate_inner[
                "position_z_threshold_sha256"
            ],
            "aligned_gate_written_to_stagee": False,
            "legacy_upper_gate_changed": False,
            "deployment_authorized": False,
        },
        "next_stage_holdout_policy": {
            "evaluation_count": 1,
            "population": "locked_backbone_all_three_timesteps",
            "backbone_id": FRONTIER_BACKBONE,
            "timesteps": list(LOCKED_TIMESTEPS),
            "all_timesteps_must_pass": True,
            "post_holdout_backbone_selection_forbidden": True,
            "post_holdout_timestep_selection_forbidden": True,
            "threshold_changes_after_holdout_forbidden": True,
            "rerun_after_failure_forbidden": True,
            "per_timestep_pass_criteria": {
                "mechanism_eligible": True,
                "overall_mse_ratio_strictly_less_than": 1.0,
                "accepted_row_mse_ratio_strictly_less_than": 1.0,
                "positive_distance_reduction_rate_strictly_greater_than": 0.5,
                "relative_distance_reduction_mean_strictly_greater_than": 0.0,
                "length_log_z_element_mismatch_count": 0,
                "aligned_upper_element_failure_count": 0,
                "strict_pass_aligned_fail_row_count": 0,
            },
            "aggregate_metrics_are_diagnostic_only": True,
            "frozen_probe_remains_closed": True,
        },
        "source_evidence": {
            "stage_t_resume1_summary_file_sha256": (
                EXPECTED_BASE_SUMMARY_FILE_SHA256
            ),
            "stage_t_resume1_gate_file_sha256": EXPECTED_BASE_GATE_FILE_SHA256,
            "stage_t_resume1_scientific_result_sha256": (
                EXPECTED_BASE_SCIENTIFIC_SHA256
            ),
            "stage_t_recovered_result_sha256": EXPECTED_RECOVERED_RESULT_SHA256,
            "candidate_matrix_sha256": EXPECTED_CANDIDATE_MATRIX_SHA256,
            "functional_projection_sha256": (
                EXPECTED_FUNCTIONAL_PROJECTION_SHA256
            ),
            "current_fit_projection_sha256": (
                EXPECTED_CURRENT_FIT_PROJECTION_SHA256
            ),
        },
        "stageu_execution": {
            "environment_probe_count": 0,
            "science_worker_count": 0,
            "oof_fit_count": 0,
            "callback_pair_count": 0,
            "shadow_internal_scale_attempt_count": 0,
            "matrix_reconstruction_attempt_count": 0,
            "candidate_generation_count": 0,
            "gpu_used": False,
            "historical_reports_read_only": True,
        },
        "inherited_stage_t_execution": {
            "environment_probe_count": 1,
            "cold_science_worker_count": 2,
            "oof_fit_count": 108,
            "callback_pair_count": 54,
            "shadow_internal_scale_attempt_count": 378,
            "matrix_reconstruction_attempt_count": 378,
            "worker_comparison_all_exact": True,
        },
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    contract["contract_payload_sha256"] = sha256_bytes(stable_json_bytes(contract))
    return contract


def build_summary(
    *,
    repository: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    summary: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": ROOT_CAUSE,
        "required_next_path": NEXT_PATH,
        "primary_failure_locus": "locked_candidate_frontier",
        "repository": dict(repository),
        "frontier_contract_sha256": sha256_bytes(stable_json_bytes(contract)),
        "frontier_lock_summary": {
            "unique_frontier_backbone": FRONTIER_BACKBONE,
            "locked_timesteps": list(LOCKED_TIMESTEPS),
            "mechanism_eligible_cell_count": EXPECTED_MECHANISM_ELIGIBLE_CELLS,
            "fidelity_eligible_cell_count": EXPECTED_FIDELITY_ELIGIBLE_CELLS,
            "matrix_eligible_backbone_count": EXPECTED_MATRIX_ELIGIBLE_BACKBONES,
            "all_locked_timesteps_fidelity_eligible": True,
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "next_stage_holdout_evaluation_count": 1,
        },
        "stageu_execution": copy.deepcopy(contract["stageu_execution"]),
        "inherited_stage_t_execution": copy.deepcopy(
            contract["inherited_stage_t_execution"]
        ),
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    summary["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(summary))
    return summary


def execute_lock(
    *,
    root: Path,
    repository: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    validate_environment_variables()
    repo = Path(root).resolve()
    summary = load_json(repo / BASE_SUMMARY)
    gate = load_json(repo / BASE_GATE)
    validated = validate_summary(summary)
    gate_inner = validate_gate_wrapper(gate)
    contract = build_frontier_contract(
        validated_summary=validated,
        gate_inner=gate_inner,
    )
    observed_contract_sha = sha256_bytes(stable_json_bytes(contract))
    if observed_contract_sha != EXPECTED_FRONTIER_CONTRACT_SHA256:
        raise StageUError(
            "Stage-U frontier contract SHA changed: "
            f"{observed_contract_sha}"
        )
    result = build_summary(repository=repository, contract=contract)
    return result, contract


def _validate_base_provenance(repo: Path) -> Mapping[str, Any]:
    if _run_git(repo, "rev-parse", f"{BASE_STAGET_RESUME1_EVIDENCE_COMMIT}^") != (
        BASE_STAGET_RESUME1_IMPLEMENTATION_COMMIT
    ):
        raise StageUError("Stage-T Resume1 evidence parent changed")
    if _run_git(
        repo,
        "rev-parse",
        f"{BASE_STAGET_RESUME1_IMPLEMENTATION_COMMIT}^",
    ) != EXPECTED_STAGET_RESUME1_IMPLEMENTATION_PARENT:
        raise StageUError("Stage-T Resume1 implementation parent changed")
    if _run_git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGET_RESUME1_IMPLEMENTATION_COMMIT,
    ) != BASE_IMPLEMENTATION_SUBJECT:
        raise StageUError("Stage-T Resume1 implementation subject changed")
    if _run_git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_STAGET_RESUME1_EVIDENCE_COMMIT,
    ) != BASE_EVIDENCE_SUBJECT:
        raise StageUError("Stage-T Resume1 evidence subject changed")
    if commit_name_status(
        repo, BASE_STAGET_RESUME1_IMPLEMENTATION_COMMIT
    ) != tuple(sorted(BASE_IMPLEMENTATION_PATHS)):
        raise StageUError("Stage-T Resume1 implementation paths changed")
    if commit_name_status(
        repo, BASE_STAGET_RESUME1_EVIDENCE_COMMIT
    ) != tuple(sorted(BASE_EVIDENCE_PATHS)):
        raise StageUError("Stage-T Resume1 evidence paths changed")
    for relative, expected in BASE_SOURCE_SHA256.items():
        if sha256_file(repo / relative) != expected:
            raise StageUError(f"Stage-T Resume1 source changed: {relative}")
    if sha256_file(repo / BASE_SUMMARY) != EXPECTED_BASE_SUMMARY_FILE_SHA256:
        raise StageUError("Stage-T Resume1 summary file SHA changed")
    if sha256_file(repo / BASE_GATE) != EXPECTED_BASE_GATE_FILE_SHA256:
        raise StageUError("Stage-T Resume1 gate file SHA changed")
    return {
        "implementation_commit": BASE_STAGET_RESUME1_IMPLEMENTATION_COMMIT,
        "evidence_commit": BASE_STAGET_RESUME1_EVIDENCE_COMMIT,
        "summary_file_sha256": EXPECTED_BASE_SUMMARY_FILE_SHA256,
        "gate_file_sha256": EXPECTED_BASE_GATE_FILE_SHA256,
    }


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _run_git(repo, "branch", "--show-current") != "Experiment1":
        raise StageUError("Stage-U requires Experiment1")
    head = _run_git(repo, "rev-parse", "HEAD")
    if _run_git(repo, "rev-parse", f"{head}^") != BASE_STAGET_RESUME1_EVIDENCE_COMMIT:
        raise StageUError("Stage-U implementation parent changed")
    if _run_git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageUError("Stage-U implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageUError("Stage-U implementation paths changed")
    if _run_git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageUError("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _run_git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageUError("DeformableRavens commit changed")
    assert_clean_worktree(repo, "Stage-U")
    assert_clean_worktree(submodule, "DeformableRavens")
    base = _validate_base_provenance(repo)
    for relative in (SUCCESS_REPORT, CONTRACT_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageUError(f"Stage-U output already exists: {relative}")
    validated = validate_summary(load_json(repo / BASE_SUMMARY))
    validate_gate_wrapper(load_json(repo / BASE_GATE))
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_STAGET_RESUME1_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "base_stage_t_resume1": base,
        "frontier_backbone": FRONTIER_BACKBONE,
        "locked_timesteps": list(LOCKED_TIMESTEPS),
        "candidate_matrix_sha256": validated["execution"][
            "candidate_matrix_sha256"
        ],
    }


def blocked_report(
    *,
    repository: Optional[Mapping[str, Any]],
    error: BaseException,
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stageu_candidate_frontier_lock_failed",
        "required_next_path": "RESTORE_STAGEU_CANDIDATE_FRONTIER_LOCK",
        "primary_failure_locus": "frontier_lock_contract",
        "error_type": type(error).__name__,
        "error": str(error),
        "repository": None if repository is None else dict(repository),
        "stageu_environment_probe_count": 0,
        "stageu_science_worker_count": 0,
        "stageu_oof_fit_count": 0,
        "stageu_callback_pair_count": 0,
        "stageu_shadow_internal_scale_attempt_count": 0,
        "stageu_matrix_reconstruction_attempt_count": 0,
        "selection_holdout_evaluation_count": 0,
        "frontier_contract_written": False,
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **{key: False for key in FALSE_BOUNDARIES},
    }
