"""Phase3.14b-r2.5.7 Stage-D Resume3 calibration-trace attribution.

Resume2 corrected the immutable-schema nesting and reached the frozen Stage-C
pre-control gate.  That gate then proved that the manually reconstructed
calibration dictionary was not byte-exact to the Stage-C evidence.

Resume3 does not relax that gate and does not modify Stage C, Stage D, Resume1,
or Resume2.  It makes two isolated observations:

1. A disposable child process replays the Resume1 manual calibration path and
   records a bounded recursive diff against the frozen Stage-C contract.
2. The main worker invokes the frozen Stage-C ``run_calibration`` entry point
   itself.  Thin wrappers capture its original calibration and its completed
   diffusion-only control, then raise a sentinel before the first nonzero
   Stage-C candidate is trained.

The captured calibration and complete control record must be exactly equal to
the Stage-C evidence.  Only then is the frozen Resume1 executor resumed with
its ``exact_control_replay`` call replaced by the already verified captured
control.  All gated calibration, candidate training, selection, classification,
and forbidden-stage boundaries remain the original Resume1/Stage-D logic.

No checkpoint, weights, prediction tensor, candidate tensor, NPZ, cache,
image, or video is persisted.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3 import phase314b_r257_stageb_mechanism_audit as stageb_mechanism
from ccda_phase3 import phase314b_r257_stagec_topk_quadratic as stagec_topk
from ccda_phase3 import phase314b_r257_staged_timestep_gate as failed_staged
from ccda_phase3 import phase314b_r257_staged_resume1_control_trace as resume1
from ccda_phase3 import phase314b_r257_staged_resume2_schema as resume2

PHASE = "Phase3.14b-r2.5.7 Stage D Resume3"
PHASE_ID = "phase314b_r257_staged_resume3"

STAGED_IMPLEMENTATION_COMMIT = (
    "ccff30947a85028a2c7974edb5b008f3b3de2757"
)
STAGED_BLOCKED_EVIDENCE_COMMIT = (
    "630cff532d811c889702bb236766b659bdfd83d4"
)
RESUME1_IMPLEMENTATION_COMMIT = (
    "d34838bfebbde2376d128dff398dadcc97877191"
)
RESUME1_BLOCKED_EVIDENCE_COMMIT = (
    "85ec59d45c6445e9cd61e91b7e6d6d94fe729ecf"
)
RESUME2_IMPLEMENTATION_COMMIT = (
    "e68d069a8ad5135a10d8e4ab3a07f46049d596aa"
)
STAGEC_EVIDENCE_COMMIT = (
    "a002246558e66029bdd6279e09c95e1303e2356c"
)
EXPECTED_SUBMODULE_COMMIT = (
    "633a88752445cf5d6776ed374fdbbdb35f93050c"
)

RESUME2_BLOCKED = (
    "reports/"
    "phase3_14b_r257_staged_resume2_blocked_summary.json"
)
RESUME2_TEST_GATE = (
    "reports/"
    "phase3_14b_r257_staged_resume2_test_gate_summary.json"
)
EXPECTED_RESUME2_BLOCKED_SHA256 = (
    "f737cd10bcbe50b053df62e6e4840f357f0ea5257459846ad37b8d48014931d3"
)
EXPECTED_RESUME2_TEST_GATE_SHA256 = (
    "2cab7c00335472da24f1ac62894ac7b971dde6965037decd9d1b168b06cffe94"
)

RESUME2_IMPLEMENTATION_FILES = (
    (
        "ccda_phase3/"
        "phase314b_r257_staged_resume2_schema.py"
    ),
    "scripts/phase3_14b_r257_staged_resume2_worker.py",
    "scripts/phase3_14b_r257_staged_resume2_run.py",
    "scripts/phase3_14b_r257_staged_resume2_test_gate.py",
    "scripts/phase3_14b_r257_staged_resume2_blocked.py",
    "scripts/phase3_14b_r257_staged_resume2_run.sh",
    (
        "tests/"
        "test_phase3_14b_r257_staged_resume2_schema.py"
    ),
)

MAX_DIFF_RECORDS = 256


class Resume3TraceError(RuntimeError):
    """Raised when provenance, attribution, or exact capture fails."""


class _StopAfterStageCControl(RuntimeError):
    """Internal sentinel raised before the first nonzero Stage-C candidate."""


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("non-finite array cannot be serialized")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite scalar cannot be serialized")
        return value
    raise TypeError(
        "unsupported JSON value: {!r}".format(type(value))
    )


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            "refusing to overwrite write-once output: {}".format(
                target
            )
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (
        ".{}.{}.tmp".format(target.name, os.getpid())
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Resume3TraceError(
            "JSON root is not an object: {}".format(path)
        )
    return value


def assert_commit_ancestor(root: Path, commit: str) -> None:
    completed = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise Resume3TraceError(
            "required commit is not an ancestor: {}".format(
                commit
            )
        )


def assert_file_bound_to_commit(
    root: Path,
    relative: str,
    commit: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        [
            "git",
            "show",
            "{}:{}".format(commit, relative),
        ],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise Resume3TraceError(
            "bound file changed: {}".format(relative)
        )
    return sha256_bytes(observed)


def assert_tracked_current_blob(
    root: Path,
    relative: str,
) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    committed = subprocess.check_output(
        ["git", "show", "HEAD:{}".format(relative)],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise Resume3TraceError(
            "tracked report differs from HEAD: {}".format(
                relative
            )
        )
    return sha256_bytes(observed)


def validate_resume2_failure(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    for commit in (
        STAGEC_EVIDENCE_COMMIT,
        STAGED_IMPLEMENTATION_COMMIT,
        STAGED_BLOCKED_EVIDENCE_COMMIT,
        RESUME1_IMPLEMENTATION_COMMIT,
        RESUME1_BLOCKED_EVIDENCE_COMMIT,
        RESUME2_IMPLEMENTATION_COMMIT,
    ):
        assert_commit_ancestor(repository_root, commit)

    implementation_sha = {
        relative: assert_file_bound_to_commit(
            repository_root,
            relative,
            RESUME2_IMPLEMENTATION_COMMIT,
        )
        for relative in RESUME2_IMPLEMENTATION_FILES
    }

    blocked_sha = assert_tracked_current_blob(
        repository_root,
        RESUME2_BLOCKED,
    )
    gate_sha = assert_tracked_current_blob(
        repository_root,
        RESUME2_TEST_GATE,
    )
    if blocked_sha != EXPECTED_RESUME2_BLOCKED_SHA256:
        raise Resume3TraceError(
            "Resume2 blocked-summary SHA changed"
        )
    if gate_sha != EXPECTED_RESUME2_TEST_GATE_SHA256:
        raise Resume3TraceError(
            "Resume2 test-gate SHA changed"
        )

    blocked = load_json(repository_root / RESUME2_BLOCKED)
    expected_blocked = {
        "verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r257_staged_resume2_"
            "execution_failed_before_completion"
        ),
        "resume2_correction_applied": False,
        "control_replay_exact": False,
        "new_hyperparameter_candidate_run": False,
        "new_objective_variant_run": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
        "full_stageb_repaired_model_trained": False,
        "formal_pilot_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
    }
    for key, expected in expected_blocked.items():
        if blocked.get(key) != expected:
            raise Resume3TraceError(
                "Resume2 blocked field changed: {}".format(key)
            )

    gate = load_json(repository_root / RESUME2_TEST_GATE)
    if gate.get("verdict") != "PASS":
        raise Resume3TraceError(
            "Resume2 test gate is not PASS"
        )
    if int(gate.get("passed_test_count", -1)) != 784:
        raise Resume3TraceError(
            "Resume2 pass count changed"
        )
    if int(gate.get("test_file_count", -1)) != 44:
        raise Resume3TraceError(
            "Resume2 test-file count changed"
        )
    if int(
        gate.get("r257_staged_resume2_new_passed", -1)
    ) != 21:
        raise Resume3TraceError(
            "Resume2 new-test count changed"
        )

    return {
        "resume2_implementation_commit":
            RESUME2_IMPLEMENTATION_COMMIT,
        "resume2_implementation_sha256":
            implementation_sha,
        "resume2_blocked_path": RESUME2_BLOCKED,
        "resume2_blocked_sha256": blocked_sha,
        "resume2_test_gate_path": RESUME2_TEST_GATE,
        "resume2_test_gate_sha256": gate_sha,
        "resume2_blocked": blocked,
        "resume2_test_gate": gate,
    }


def summarize_value(value: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "type": type(value).__name__,
    }
    if isinstance(value, Mapping):
        result["key_count"] = len(value)
        result["sha256"] = sha256_bytes(
            stable_json_bytes(dict(value))
        )
    elif isinstance(value, (list, tuple)):
        result["length"] = len(value)
        result["sha256"] = sha256_bytes(
            stable_json_bytes({"value": list(value)})
        )
    elif isinstance(value, float):
        result["value"] = float(value)
    elif isinstance(value, (str, int, bool)) or value is None:
        result["value"] = value
    else:
        result["repr"] = repr(value)[:160]
    return result


def recursive_diff(
    expected: Any,
    observed: Any,
    *,
    path: str = "$",
    records: Optional[List[Dict[str, Any]]] = None,
    counter: Optional[List[int]] = None,
    limit: int = MAX_DIFF_RECORDS,
) -> Tuple[List[Dict[str, Any]], int]:
    output = [] if records is None else records
    total = [0] if counter is None else counter

    def add(kind: str, location: str, left: Any, right: Any) -> None:
        total[0] += 1
        if len(output) < int(limit):
            record: Dict[str, Any] = {
                "path": location,
                "kind": kind,
                "expected": summarize_value(left),
                "observed": summarize_value(right),
            }
            if isinstance(left, float) and isinstance(right, float):
                record["absolute_difference"] = abs(left - right)
                denominator = max(abs(left), abs(right), 1.0e-300)
                record["relative_difference"] = (
                    abs(left - right) / denominator
                )
            output.append(record)

    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        expected_keys = set(expected)
        observed_keys = set(observed)
        for key in sorted(expected_keys - observed_keys):
            add(
                "missing_observed_key",
                "{}.{}".format(path, key),
                expected[key],
                None,
            )
        for key in sorted(observed_keys - expected_keys):
            add(
                "unexpected_observed_key",
                "{}.{}".format(path, key),
                None,
                observed[key],
            )
        for key in sorted(expected_keys & observed_keys):
            recursive_diff(
                expected[key],
                observed[key],
                path="{}.{}".format(path, key),
                records=output,
                counter=total,
                limit=limit,
            )
    elif (
        isinstance(expected, (list, tuple))
        and isinstance(observed, (list, tuple))
    ):
        if len(expected) != len(observed):
            add("sequence_length", path, list(expected), list(observed))
        for index, (left, right) in enumerate(
            zip(expected, observed)
        ):
            recursive_diff(
                left,
                right,
                path="{}[{}]".format(path, index),
                records=output,
                counter=total,
                limit=limit,
            )
    elif type(expected) is not type(observed):
        # JSON round-trips may normalize tuple to list and NumPy scalar to
        # Python scalar; all expected contract values should already be JSON.
        add("type_mismatch", path, expected, observed)
    elif expected != observed:
        add("value_mismatch", path, expected, observed)

    return output, total[0]


def diff_category(path: str, kind: str) -> str:
    lowered = path.lower()
    if kind in (
        "missing_observed_key",
        "unexpected_observed_key",
        "sequence_length",
        "type_mismatch",
    ):
        return "structure"
    if "initial_model_sha256" in lowered:
        return "initial_model_identity"
    if "predicted_x0_sha256" in lowered:
        return "prediction_identity"
    if "selected_excess_sha256" in lowered:
        return "geometry_identity"
    if "calibration_sha256" in lowered:
        return "aggregate_calibration_identity"
    if "candidate_order" in lowered or "candidate_count" in lowered:
        return "candidate_schema"
    if ".gradient." in lowered:
        return "gradient_metric"
    if "row_contribution" in lowered:
        return "row_contribution_metric"
    if "selected_positions" in lowered:
        return "position_metric"
    if "selected_excess" in lowered:
        return "geometry_metric"
    if "diffusion_loss" in lowered or "geometry_loss" in lowered:
        return "loss_metric"
    return "other_scalar"


def classify_diff(
    records: Sequence[Mapping[str, Any]],
    total_count: int,
) -> Dict[str, Any]:
    categories: MutableMapping[str, int] = {}
    for record in records:
        category = diff_category(
            str(record["path"]),
            str(record["kind"]),
        )
        categories[category] = categories.get(category, 0) + 1

    identity_categories = {
        "initial_model_identity",
        "prediction_identity",
        "geometry_identity",
        "aggregate_calibration_identity",
        "candidate_schema",
        "structure",
    }
    identity_difference = any(
        categories.get(name, 0) > 0
        for name in identity_categories
    )
    if total_count == 0:
        locus = "none"
    elif identity_difference:
        locus = "calibration_identity_or_structure"
    elif categories.get("gradient_metric", 0) > 0:
        locus = "floating_gradient_metrics"
    elif (
        categories.get("loss_metric", 0) > 0
        or categories.get("geometry_metric", 0) > 0
        or categories.get("row_contribution_metric", 0) > 0
        or categories.get("position_metric", 0) > 0
    ):
        locus = "derived_calibration_metrics"
    else:
        locus = "other_scalar_metrics"
    return {
        "total_difference_count": int(total_count),
        "recorded_difference_count": len(records),
        "records_truncated": bool(
            total_count > len(records)
        ),
        "category_counts": dict(
            sorted(categories.items())
        ),
        "primary_difference_locus": locus,
        "identity_or_structure_difference":
            bool(identity_difference),
    }


def build_manual_calibration_context(
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    immutable = failed_staged.validate_immutable_inputs(
        repository_root
    )
    stagec_contract = immutable["stagec_contract"]
    canonical = stagec_topk.validate_immutable_inputs(
        repository_root
    )
    adapted = resume2.adapt_stagec_schema_for_resume1(
        canonical
    )
    objective_contract = (
        stageb_mechanism.load_objective_contract(
            adapted["stagea_contract"]
        )
    )

    stageb_spec = stageb.DiagnosticSpec()
    stageb_spec.validate()
    arrays = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=stageb.REQUIRED_TRAIN_KEYS,
    )
    condition = np.asarray(
        arrays["diffusion_condition_x"],
        dtype=np.float32,
    )
    target = np.asarray(
        arrays["diffusion_target_cable"],
        dtype=np.float32,
    )
    groups = np.asarray(
        arrays["episode_group_key"]
    ).astype(str)

    stageb_train, frozen_probe, _ = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb_spec.group_folds,
            probe_fold=stageb_spec.probe_fold,
        )
    )
    objective_train, selection_holdout, split = (
        stagea.deterministic_selection_split(
            groups,
            stageb_train,
            folds=(
                stagea.UpperObjectiveSpec()
                .selection_group_folds
            ),
            holdout_fold=(
                stagea.UpperObjectiveSpec()
                .selection_holdout_fold
            ),
        )
    )
    if np.any(
        frozen_probe
        & (objective_train | selection_holdout)
    ):
        raise Resume3TraceError(
            "frozen probe crossed manual-diff split"
        )
    condition_standardizer = stageb.fit_standardizer(
        condition[objective_train]
    )
    target_standardizer = stageb.fit_standardizer(
        target[objective_train]
    )
    diagnostic_batch = stageb_mechanism.fixed_diagnostic_batch(
        condition=condition[objective_train],
        target=target[objective_train],
        condition_standardizer=condition_standardizer,
        target_standardizer=target_standardizer,
        stageb_spec=stageb_spec,
        spec=stageb_mechanism.MechanismAuditSpec(),
    )
    return {
        "stageb_spec": stageb_spec,
        "objective_contract": objective_contract,
        "target_standardizer": target_standardizer,
        "diagnostic_batch": diagnostic_batch,
        "expected_calibration":
            stagec_contract["calibration"],
        "split": {
            **split,
            "stageb_training_rows":
                int(np.sum(stageb_train)),
            "objective_train_rows":
                int(np.sum(objective_train)),
            "selection_holdout_rows":
                int(np.sum(selection_holdout)),
            "frozen_probe_rows":
                int(np.sum(frozen_probe)),
            "frozen_probe_accessed": False,
        },
    }


def run_manual_calibration_diff(
    *,
    root: Path,
) -> Dict[str, Any]:
    context = build_manual_calibration_context(root)
    candidates, observed = stagec_topk.calibrate_candidates(
        stageb_spec=context["stageb_spec"],
        diagnostic_batch=context["diagnostic_batch"],
        target_standardizer=
            context["target_standardizer"],
        objective_contract=context["objective_contract"],
        spec=stagec_topk.TopKCalibrationSpec(),
    )
    expected = context["expected_calibration"]
    records, total = recursive_diff(
        expected,
        observed,
        limit=MAX_DIFF_RECORDS,
    )
    classification = classify_diff(records, total)
    return {
        "phase": PHASE,
        "schema":
            "phase314b_r257_staged_resume3_manual_diff_v1",
        "completed": True,
        "manual_replay_exact": bool(total == 0),
        "expected_calibration_sha256":
            expected["calibration_sha256"],
        "observed_calibration_sha256":
            observed["calibration_sha256"],
        "expected_payload_sha256":
            sha256_bytes(stable_json_bytes(expected)),
        "observed_payload_sha256":
            sha256_bytes(stable_json_bytes(observed)),
        "candidate_order": [
            candidate.candidate_id
            for candidate in candidates
        ],
        "split": context["split"],
        "difference_summary": classification,
        "differences": records,
        "training_performed": False,
        "control_replay_run": False,
        "new_candidate_run": False,
        "frozen_probe_accessed": False,
        "reverse_sampling_run": False,
    }


@contextlib.contextmanager
def capture_stagec_control_entrypoint() -> Iterator[MutableMapping[str, Any]]:
    """Capture Stage-C calibration/control and stop before candidate training."""
    state: MutableMapping[str, Any] = {
        "train_candidate_call_count": 0,
        "control_selection_record_count": 0,
    }
    original_calibrate = stagec_topk.calibrate_candidates
    original_train = stagec_topk.train_candidate
    original_selection_record = stagec_topk.selection_record

    def wrapped_calibrate(*args: Any, **kwargs: Any) -> Any:
        candidates, calibration = original_calibrate(
            *args,
            **kwargs,
        )
        state["calibration"] = calibration
        state["candidate_order"] = [
            candidate.candidate_id
            for candidate in candidates
        ]
        return candidates, calibration

    def wrapped_train(*args: Any, **kwargs: Any) -> Any:
        call_index = int(
            state["train_candidate_call_count"]
        )
        if call_index >= 1:
            state["stopped_before_first_nonzero_candidate"] = True
            raise _StopAfterStageCControl(
                "Stage-C control captured"
            )
        state["train_candidate_call_count"] = call_index + 1
        result = original_train(*args, **kwargs)
        state["control_training"] = result[1]
        state["control_diagnostics"] = result[2]
        return result

    def wrapped_selection_record(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        record = original_selection_record(
            *args,
            **kwargs,
        )
        if kwargs.get("candidate") is None:
            state["control_record"] = record
            state["control_selection_record_count"] = (
                int(
                    state[
                        "control_selection_record_count"
                    ]
                )
                + 1
            )
        return record

    stagec_topk.calibrate_candidates = wrapped_calibrate
    stagec_topk.train_candidate = wrapped_train
    stagec_topk.selection_record = wrapped_selection_record
    try:
        yield state
    finally:
        stagec_topk.calibrate_candidates = original_calibrate
        stagec_topk.train_candidate = original_train
        stagec_topk.selection_record = original_selection_record


def capture_exact_stagec_control(
    *,
    root: Path,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    immutable = failed_staged.validate_immutable_inputs(
        repository_root
    )
    expected_calibration = immutable[
        "stagec_contract"
    ]["calibration"]
    expected_control = immutable["control_record"]

    with capture_stagec_control_entrypoint() as state:
        try:
            stagec_topk.run_calibration(
                root=repository_root
            )
        except _StopAfterStageCControl:
            pass
        else:
            raise Resume3TraceError(
                "Stage-C capture did not stop before "
                "the first nonzero candidate"
            )

    required = (
        "calibration",
        "candidate_order",
        "control_record",
        "control_training",
        "control_diagnostics",
        "stopped_before_first_nonzero_candidate",
    )
    missing = [
        key
        for key in required
        if key not in state
    ]
    if missing:
        raise Resume3TraceError(
            "Stage-C capture is incomplete: {}".format(
                missing
            )
        )
    if int(state["train_candidate_call_count"]) != 1:
        raise Resume3TraceError(
            "Stage-C capture trained more than the control"
        )
    if int(
        state["control_selection_record_count"]
    ) != 1:
        raise Resume3TraceError(
            "Stage-C control record count changed"
        )

    calibration_records, calibration_count = recursive_diff(
        expected_calibration,
        state["calibration"],
        limit=MAX_DIFF_RECORDS,
    )
    if calibration_count:
        raise Resume3TraceError(
            "frozen Stage-C entrypoint calibration differs: "
            "{}".format(
                [
                    record["path"]
                    for record in calibration_records[:16]
                ]
            )
        )
    control_records, control_count = recursive_diff(
        expected_control,
        state["control_record"],
        limit=MAX_DIFF_RECORDS,
    )
    if control_count:
        raise Resume3TraceError(
            "frozen Stage-C entrypoint control differs: "
            "{}".format(
                [
                    record["path"]
                    for record in control_records[:16]
                ]
            )
        )

    control_record = copy.deepcopy(
        state["control_record"]
    )
    identity = {
        "capture_method":
            "frozen_stagec_run_calibration_entrypoint",
        "stopped_before_first_nonzero_candidate": True,
        "stagec_nonzero_candidate_training_count": 0,
        "calibration_exact": True,
        "calibration_sha256":
            state["calibration"]["calibration_sha256"],
        "calibration_payload_sha256":
            sha256_bytes(
                stable_json_bytes(
                    state["calibration"]
                )
            ),
        "candidate_order":
            list(state["candidate_order"]),
        "control_record_exact": True,
        "control_record_sha256":
            sha256_bytes(
                stable_json_bytes(control_record)
            ),
        "control_training_sha256":
            sha256_bytes(
                stable_json_bytes(
                    control_record["training"]
                )
            ),
        "control_model_sha256":
            control_record["training"][
                "final_model_sha256"
            ],
        "control_optimizer_sha256":
            control_record["training"][
                "final_optimizer_sha256"
            ],
        "control_loss_sha256":
            control_record["training"][
                "total_loss_history_sha256"
            ],
        "control_gradient_sha256":
            control_record["training"][
                "gradient_history_sha256"
            ],
        "control_exposure_sha256":
            control_record["training"][
                "source_exposure_sha256"
            ],
        "control_train_control_exact": True,
        "control_one_step_exact": True,
        "control_profiles_exact": True,
        "capture_wrappers_restored": bool(
            stagec_topk.calibrate_candidates
            is not None
            and stagec_topk.train_candidate
            is not None
            and stagec_topk.selection_record
            is not None
        ),
    }
    return {
        "control_bundle": control_record,
        "control_identity": identity,
    }


@contextlib.contextmanager
def install_captured_control(
    control_bundle: Mapping[str, Any],
    control_identity: Mapping[str, Any],
) -> Iterator[None]:
    original = resume1.exact_control_replay

    def captured(*args: Any, **kwargs: Any) -> Any:
        return (
            copy.deepcopy(dict(control_bundle)),
            copy.deepcopy(dict(control_identity)),
        )

    resume1.exact_control_replay = captured
    try:
        yield
    finally:
        resume1.exact_control_replay = original


def run_resume3(
    *,
    root: Path,
    manual_diff: Mapping[str, Any],
    spec: Optional[Any] = None,
) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    failure = validate_resume2_failure(
        repository_root
    )
    if manual_diff.get("completed") is not True:
        raise Resume3TraceError(
            "manual calibration diff did not complete"
        )
    if manual_diff.get("training_performed") is not False:
        raise Resume3TraceError(
            "manual diff unexpectedly trained a model"
        )
    if manual_diff.get("frozen_probe_accessed") is not False:
        raise Resume3TraceError(
            "manual diff accessed frozen probe"
        )

    captured = capture_exact_stagec_control(
        root=repository_root
    )
    original_exact_control = resume1.exact_control_replay
    original_validator = (
        resume1.stagec_topk.validate_immutable_inputs
    )
    with resume2.adapted_stagec_validator():
        with install_captured_control(
            captured["control_bundle"],
            captured["control_identity"],
        ):
            result = resume1.run_resume1(
                root=repository_root,
                spec=spec,
            )
    if resume1.exact_control_replay is not original_exact_control:
        raise Resume3TraceError(
            "Resume1 exact-control function was not restored"
        )
    if (
        resume1.stagec_topk.validate_immutable_inputs
        is not original_validator
    ):
        raise Resume3TraceError(
            "Stage-C validator was not restored"
        )

    result = copy.deepcopy(result)
    result["phase"] = PHASE
    result["phase_id"] = PHASE_ID
    result["schema"] = (
        "phase314b_r257_staged_resume3_worker_result_v1"
    )
    result["resume3_calibration_attribution"] = {
        "resume2_implementation_commit":
            RESUME2_IMPLEMENTATION_COMMIT,
        "resume2_blocked_sha256":
            EXPECTED_RESUME2_BLOCKED_SHA256,
        "resume2_test_gate_sha256":
            EXPECTED_RESUME2_TEST_GATE_SHA256,
        "manual_calibration_diff": copy.deepcopy(
            dict(manual_diff)
        ),
        "stagec_entrypoint_capture":
            captured["control_identity"],
        "manual_trace_relaxed": False,
        "frozen_stagec_entrypoint_required_exact": True,
        "resume1_exact_control_replaced_only_after_capture":
            True,
        "resume1_exact_control_function_restored": True,
        "stagec_validator_restored": True,
        "resume2_files_modified": False,
        "resume2_reports_modified": False,
        "correction_scope": (
            "diagnose the manual calibration mismatch and "
            "use the frozen Stage-C entry point as the exact "
            "control authority; no metric or SHA relaxation"
        ),
    }
    result["resume2_failed_attempt"] = failure
    result["resume3_correction_applied"] = True
    result["resume2_files_modified"] = False
    result["resume2_blocked_modified"] = False
    result["resume2_test_gate_modified"] = False
    result["control_replay_exact"] = True

    contract = result.get("calibration_contract")
    if not isinstance(contract, MutableMapping):
        raise Resume3TraceError(
            "Resume1 calibration contract is missing"
        )
    contract = copy.deepcopy(dict(contract))
    contract["resume3_calibration_attribution"] = (
        result["resume3_calibration_attribution"]
    )
    contract["correction_scope"] = (
        "manual replay is diagnostic only; exact calibration "
        "and control are captured from frozen Stage-C "
        "run_calibration before any nonzero Stage-C candidate"
    )
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = sha256_bytes(
        stable_json_bytes(contract)
    )
    result["calibration_contract"] = contract
    return result


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path":
            result["required_next_path"],
        "resume3_calibration_attribution":
            result["resume3_calibration_attribution"],
        "control_trace_correction":
            result["control_trace_correction"],
        "split": result["split"],
        "calibration_contract":
            result["calibration_contract"],
        "selection": result["selection"],
        "classification": result["classification"],
        "selected_configuration":
            result["selected_configuration"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(
        identity_projection(left)
    )
    right_payload = stable_json_bytes(
        identity_projection(right)
    )
    return {
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "manual_diff_exact": (
            left["resume3_calibration_attribution"][
                "manual_calibration_diff"
            ]
            == right["resume3_calibration_attribution"][
                "manual_calibration_diff"
            ]
        ),
        "stagec_capture_exact": (
            left["resume3_calibration_attribution"][
                "stagec_entrypoint_capture"
            ]
            == right["resume3_calibration_attribution"][
                "stagec_entrypoint_capture"
            ]
        ),
        "control_trace_exact": (
            left["control_trace_correction"]
            == right["control_trace_correction"]
        ),
        "calibration_contract_exact": (
            left["calibration_contract"]
            == right["calibration_contract"]
        ),
        "selection_exact": (
            left["selection"]
            == right["selection"]
        ),
        "classification_exact": (
            left["classification"]
            == right["classification"]
        ),
    }
