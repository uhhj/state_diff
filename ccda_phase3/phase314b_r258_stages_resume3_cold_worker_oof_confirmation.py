"""Phase3.14b-r2.5.8 Stage S Resume3 cold-worker OOF confirmation.

Stage-S Resume2A proved the only required CUDA process boundary:

* one disposable subprocess runs the portable environment probe and CUDA
  required-operation dry-run;
* fresh subprocesses can then consume the probe payload while retaining a cold
  CUDA context through the frozen control replay.

Resume3 applies that already-confirmed topology to the original Stage-S
confirmation objective.  It does not redesign the gate, OOF models, candidate
population, classification tree, or evidence schema.  One disposable probe is
run once, followed by two sequential cold science workers.  Each science worker
runs the unchanged Stage-R Resume1 portable OOF functional replay and projects
its result through the already-corrected Stage-S Resume1 historical schema
adapter.

No selection holdout, frozen probe, formal diffusion training, reverse
sampling, IDM, candidate execution, DeformableRavens execution, Phase4, or CPS
is permitted.  Worker JSON files and the environment payload are temporary and
must not survive the controller process.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.5.8 Stage S Resume3"
SCHEMA = "phase314b_r258_stages_resume3_cold_worker_oof_confirmation_v1"
WORKER_SCHEMA = "phase314b_r258_stages_resume3_cold_science_worker_v1"
BLOCKED_SCHEMA = (
    "phase314b_r258_stages_resume3_cold_worker_oof_confirmation_blocked_v1"
)

BASE_RESUME2A_IMPLEMENTATION_COMMIT = "edf85f19228bec61e500e7b9aa922f136e9d1d8d"
BASE_RESUME2A_EVIDENCE_COMMIT = "38b096bf5820cdb2f1cadde78149ce389c464352"
BASE_RESUME2A_PARENT = "28cfba64980ed311810ca314dcc249a6a5681fb3"
EXPECTED_REMOTE = "6758ea7ad800667a436b0243d3b1f6c63256d854"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

RESUME2A_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S Resume2A: smoke cold-CUDA process boundary"
)
RESUME2A_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume2A cold-CUDA smoke evidence"
)
RESUME2A_REPORT = (
    "reports/phase3_14b_r258_stages_resume2a_cold_cuda_process_boundary_"
    "smoke_summary.json"
)
EXPECTED_RESUME2A_REPORT_SHA256 = (
    "c06c594fd473cce3c7d690f3cddad791ea6ae6be27601a57ebc6ba8a5c0d1490"
)
EXPECTED_RESUME2A_SCIENTIFIC_SHA256 = (
    "9c635e3ef28fb31f6f27b13c06ae12fca6f64b729e507530d1e05e3f290565fb"
)
RESUME2A_IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py",
    ),
    ("A", "scripts/phase3_14b_r258_stages_resume2a_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume2a_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py",
    ),
)
RESUME2A_EVIDENCE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", RESUME2A_REPORT),
)
RESUME2A_SOURCE_SHA256: Mapping[str, str] = {
    (
        "ccda_phase3/phase314b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py"
    ): "456ef465ff95315ef3805a92d31919c1a2da3c4867a1086ed727fc5021a8fb00",
    "scripts/phase3_14b_r258_stages_resume2a_worker.py": (
        "096ffbedea6e4504ad01f77b637d812fb8a69c2cb8437c1a3aaa4cccf66a97d2"
    ),
    "scripts/phase3_14b_r258_stages_resume2a_execute.py": (
        "b30724d7270451bd3ac59a69f71408f28beced138bc3256180de256729ca5d51"
    ),
    (
        "tests/test_phase3_14b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py"
    ): "26fc9b0d19c76c91c24ad07aa2ecc5a053674c702446fd681a45e26eda31924e",
}

SUCCESS_REPORT = (
    "reports/phase3_14b_r258_stages_resume3_cold_worker_oof_confirmation_"
    "summary.json"
)
BLOCKED_REPORT = (
    "reports/phase3_14b_r258_stages_resume3_cold_worker_oof_confirmation_"
    "blocked_summary.json"
)

IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.5.8 Stage S Resume3: confirm OOF support in cold workers"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume3 OOF confirmation evidence"
)
BLOCKED_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.5.8 Stage S Resume3 blocked evidence"
)

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "A",
        "ccda_phase3/phase314b_r258_stages_resume3_"
        "cold_worker_oof_confirmation.py",
    ),
    ("A", "scripts/phase3_14b_r258_stages_resume3_worker.py"),
    ("A", "scripts/phase3_14b_r258_stages_resume3_execute.py"),
    (
        "A",
        "tests/test_phase3_14b_r258_stages_resume3_"
        "cold_worker_oof_confirmation.py",
    ),
)

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

EXPECTED_WORKER_COUNT = 2
EXPECTED_PROBE_PROCESS_COUNT = 1
EXPECTED_SCIENCE_FITS_PER_WORKER = 27
EXPECTED_REPEAT_FITS_PER_WORKER = 27
EXPECTED_FITS_PER_WORKER = 54
EXPECTED_CALLBACK_PAIRS_PER_WORKER = 27
EXPECTED_INTERNAL_ATTEMPTS_PER_WORKER = 189
EXPECTED_TOTAL_SCIENCE_FITS = 54
EXPECTED_TOTAL_REPEAT_FITS = 54
EXPECTED_TOTAL_FITS = 108
EXPECTED_TOTAL_CALLBACK_PAIRS = 54
EXPECTED_TOTAL_INTERNAL_ATTEMPTS = 378
EXPECTED_CELL_COUNT = 27
EXPECTED_NONZERO_SUPPORT_CELLS = 27
EXPECTED_ZERO_ACCEPTANCE_CELLS = 0
EXPECTED_ORACLE_LIKE_CELLS = 7
EXPECTED_DOMINANT_DISCRIMINATOR = "direction_retention"
EXPECTED_DOMINANT_SUPPORT = 2
EXPECTED_DOMINANT_BACKBONE_COVERAGE = 1
EXPECTED_DOMINANT_TIMESTEP_COVERAGE = 2

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
    "candidate_tensor_persisted",
    "predicate_tensor_persisted",
    "callback_event_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


class StageSResume3Error(RuntimeError):
    """Fail-closed Stage-S Resume3 execution error."""


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
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageSResume3Error(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise StageSResume3Error(f"JSON root is not a mapping: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageSResume3Error(f"{label} is not a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StageSResume3Error(f"{label} is not a sequence")
    return value


def validate_environment_variables() -> Mapping[str, str]:
    mismatch = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatch:
        raise StageSResume3Error(
            "deterministic environment mismatch: {}".format(mismatch)
        )
    return dict(EXPECTED_ENV)


def require_false(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if mapping.get(key) is not False:
            raise StageSResume3Error(
                f"{label} boundary changed: {key}={mapping.get(key)!r}"
            )


def _run_git(root: Path, *args: str, text: bool = True) -> Any:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(root),
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", "replace")
        raise StageSResume3Error(
            "git {} failed: {}".format(" ".join(args), str(stderr).strip())
        )
    return completed.stdout


def _git(root: Path, *args: str) -> str:
    return str(_run_git(root, *args, text=True)).strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return bytes(_run_git(root, *args, text=False))


def assert_clean_worktree(root: Path, label: str) -> None:
    raw = _git_bytes(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if raw:
        raise StageSResume3Error(f"{label} worktree is not clean")


def commit_name_status(root: Path, commit: str) -> Tuple[Tuple[str, str], ...]:
    output = _git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", commit)
    records: List[Tuple[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise StageSResume3Error("unexpected diff-tree record")
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def validate_resume2a_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("execution_verdict") != "PASS":
        raise StageSResume3Error("Resume2A smoke did not PASS")
    if report.get("scientific_status") != "BLOCKED":
        raise StageSResume3Error("Resume2A scientific status changed")
    if report.get("root_cause") != (
        "phase314b_r258_stages_resume2a_cold_cuda_process_boundary_confirmed"
    ):
        raise StageSResume3Error("Resume2A root cause changed")
    if report.get("required_next_path") != (
        "RUN_STAGES_RESUME3_CONFIRMATION_WITH_DISPOSABLE_PROBE_"
        "AND_COLD_SCIENCE_WORKERS"
    ):
        raise StageSResume3Error("Resume2A next path changed")
    if report.get("selected_configuration") is not None:
        raise StageSResume3Error("Resume2A selected a configuration")
    if report.get("train_only_recommendation") is not None:
        raise StageSResume3Error("Resume2A emitted a recommendation")
    smoke = _mapping(report.get("process_boundary_smoke"), "Resume2A smoke")
    required = {
        "environment_probe_process_count": 1,
        "cold_control_smoke_process_count": 1,
        "processes_distinct": True,
        "cold_cuda_before_control_replay": True,
        "frozen_control_replay_completed": True,
        "oof_fit_count": 0,
        "callback_pair_count": 0,
        "internal_scale_attempt_count": 0,
        "oracle_callback_rerun_count": 0,
    }
    for key, expected in required.items():
        if smoke.get(key) != expected:
            raise StageSResume3Error(
                f"Resume2A smoke changed: {key}={smoke.get(key)!r}"
            )
    require_false(report, FALSE_BOUNDARIES, "Resume2A report")
    expected_scientific = report.get("scientific_result_sha256")
    if expected_scientific != EXPECTED_RESUME2A_SCIENTIFIC_SHA256:
        raise StageSResume3Error("Resume2A scientific-result SHA changed")
    return report


def validate_repository(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    if _git(repo, "branch", "--show-current") != "Experiment1":
        raise StageSResume3Error("Stage-S Resume3 requires Experiment1")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "rev-parse", f"{head}^") != BASE_RESUME2A_EVIDENCE_COMMIT:
        raise StageSResume3Error("Resume3 implementation parent changed")
    if _git(repo, "show", "-s", "--format=%s", head) != IMPLEMENTATION_SUBJECT:
        raise StageSResume3Error("Resume3 implementation subject changed")
    if commit_name_status(repo, head) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageSResume3Error("Resume3 implementation paths changed")

    if _git(repo, "rev-parse", f"{BASE_RESUME2A_EVIDENCE_COMMIT}^") != (
        BASE_RESUME2A_IMPLEMENTATION_COMMIT
    ):
        raise StageSResume3Error("Resume2A evidence provenance changed")
    if _git(repo, "rev-parse", f"{BASE_RESUME2A_IMPLEMENTATION_COMMIT}^") != (
        BASE_RESUME2A_PARENT
    ):
        raise StageSResume3Error("Resume2A implementation provenance changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_RESUME2A_IMPLEMENTATION_COMMIT,
    ) != RESUME2A_IMPLEMENTATION_SUBJECT:
        raise StageSResume3Error("Resume2A implementation subject changed")
    if _git(
        repo,
        "show",
        "-s",
        "--format=%s",
        BASE_RESUME2A_EVIDENCE_COMMIT,
    ) != RESUME2A_EVIDENCE_SUBJECT:
        raise StageSResume3Error("Resume2A evidence subject changed")
    if commit_name_status(repo, BASE_RESUME2A_IMPLEMENTATION_COMMIT) != tuple(
        sorted(RESUME2A_IMPLEMENTATION_PATHS)
    ):
        raise StageSResume3Error("Resume2A implementation paths changed")
    if commit_name_status(repo, BASE_RESUME2A_EVIDENCE_COMMIT) != tuple(
        sorted(RESUME2A_EVIDENCE_PATHS)
    ):
        raise StageSResume3Error("Resume2A evidence paths changed")

    for relative, expected_sha in RESUME2A_SOURCE_SHA256.items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise StageSResume3Error(f"Resume2A source changed: {relative}")
        committed = _git_bytes(
            repo,
            "show",
            f"{BASE_RESUME2A_IMPLEMENTATION_COMMIT}:{relative}",
        )
        if committed != path.read_bytes():
            raise StageSResume3Error(
                f"Resume2A source differs from committed blob: {relative}"
            )

    report_path = repo / RESUME2A_REPORT
    if not report_path.is_file():
        raise StageSResume3Error("Resume2A report is missing")
    if sha256_file(report_path) != EXPECTED_RESUME2A_REPORT_SHA256:
        raise StageSResume3Error("Resume2A report SHA changed")
    committed_report = _git_bytes(
        repo,
        "show",
        f"{BASE_RESUME2A_EVIDENCE_COMMIT}:{RESUME2A_REPORT}",
    )
    if committed_report != report_path.read_bytes():
        raise StageSResume3Error("Resume2A report differs from committed blob")
    resume2a_report = validate_resume2a_report(load_json(report_path))

    if _git(repo, "rev-parse", "origin/Experiment1") != EXPECTED_REMOTE:
        raise StageSResume3Error("origin/Experiment1 changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageSResume3Error("DeformableRavens commit changed")
    assert_clean_worktree(submodule, "DeformableRavens")
    assert_clean_worktree(repo, "Stage-S Resume3")
    for relative in (SUCCESS_REPORT, BLOCKED_REPORT):
        if (repo / relative).exists():
            raise StageSResume3Error(f"Resume3 output already exists: {relative}")

    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages

    base_report = stages.validate_base_report(repo)
    return {
        "root": str(repo),
        "head": head,
        "parent": BASE_RESUME2A_EVIDENCE_COMMIT,
        "origin_experiment1": EXPECTED_REMOTE,
        "submodule_commit": EXPECTED_SUBMODULE,
        "resume2a_implementation_commit": BASE_RESUME2A_IMPLEMENTATION_COMMIT,
        "resume2a_evidence_commit": BASE_RESUME2A_EVIDENCE_COMMIT,
        "resume2a_report_sha256": EXPECTED_RESUME2A_REPORT_SHA256,
        "resume2a_scientific_result_sha256": EXPECTED_RESUME2A_SCIENTIFIC_SHA256,
        "resume2a_report": copy.deepcopy(dict(resume2a_report)),
        "base_report": base_report,
    }


def validate_probe_payload_for_science(
    payload: Mapping[str, Any],
    *,
    resume2a: Any,
) -> Mapping[str, Any]:
    environment = resume2a.validate_probe_payload(payload)
    wrapper = _mapping(payload, "probe wrapper")
    process_id = wrapper.get("process_id")
    if not isinstance(process_id, int):
        raise StageSResume3Error("probe process ID is invalid")
    if wrapper.get("process_disposable") is not True:
        raise StageSResume3Error("probe process is not disposable")
    if wrapper.get("cuda_initialization_allowed_in_this_process") is not True:
        raise StageSResume3Error("probe CUDA permission changed")
    expected_environment_sha = sha256_bytes(stable_json_bytes(environment))
    if wrapper.get("environment_sha256") != expected_environment_sha:
        raise StageSResume3Error("probe environment SHA is invalid")
    return environment


def _load_science_modules() -> Mapping[str, Any]:
    from ccda_phase3 import phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo
    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages
    from ccda_phase3 import phase314b_r258_stages_resume1_oracle_rerun_schema_recovery as stages_resume1
    from ccda_phase3 import phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a
    from ccda_phase3 import phase314b_r258_stager_resume1_portable_oof_functional_replay as stager_resume1

    return {
        "stageo": stageo,
        "stages": stages,
        "stages_resume1": stages_resume1,
        "resume2a": resume2a,
        "stager_resume1": stager_resume1,
    }


def validate_cold_science_worker_payload(
    payload: Mapping[str, Any],
    *,
    expected_environment_sha256: str,
    expected_base_projection_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    worker = _mapping(payload, "cold science worker")
    if worker.get("schema") != WORKER_SCHEMA:
        raise StageSResume3Error("cold science worker schema changed")
    if worker.get("execution_verdict") != "PASS":
        raise StageSResume3Error("cold science worker did not PASS")
    if worker.get("environment_sha256") != expected_environment_sha256:
        raise StageSResume3Error("cold science worker environment SHA changed")
    if expected_base_projection_sha256 is not None and worker.get(
        "base_functional_projection_sha256"
    ) != expected_base_projection_sha256:
        raise StageSResume3Error("worker base functional projection changed")
    cold = _mapping(worker.get("cold_cuda_precheck"), "cold CUDA precheck")
    if cold.get("torch_cuda_is_initialized") is not False:
        raise StageSResume3Error("science worker was not cold before replay")
    if worker.get("environment_probe_rerun_in_science_process") is not False:
        raise StageSResume3Error("science worker reran environment probe")
    if worker.get("functional_projection_matches_base") is not True:
        raise StageSResume3Error("science worker functional projection changed")
    required_counts = {
        "scientific_oof_fit_count": EXPECTED_SCIENCE_FITS_PER_WORKER,
        "repeat_identity_fit_count": EXPECTED_REPEAT_FITS_PER_WORKER,
        "total_oof_fit_count": EXPECTED_FITS_PER_WORKER,
        "callback_pair_count": EXPECTED_CALLBACK_PAIRS_PER_WORKER,
        "oracle_callback_rerun_count": 0,
        "internal_scale_attempt_count": EXPECTED_INTERNAL_ATTEMPTS_PER_WORKER,
    }
    for key, expected in required_counts.items():
        if worker.get(key) != expected:
            raise StageSResume3Error(
                f"science worker count changed: {key}={worker.get(key)!r}"
            )
    require_false(worker, FALSE_BOUNDARIES, "cold science worker")
    return worker


def cold_science_worker_payload(
    *,
    worker_id: str,
    root: Path,
    probe_payload: Mapping[str, Any],
    repository_head: str,
) -> Mapping[str, Any]:
    """Run in a fresh process that has never executed the CUDA probe."""
    validate_environment_variables()
    modules = _load_science_modules()
    stageo = modules["stageo"]
    stages = modules["stages"]
    stages_resume1 = modules["stages_resume1"]
    resume2a = modules["resume2a"]
    stager_resume1 = modules["stager_resume1"]

    environment = validate_probe_payload_for_science(
        probe_payload,
        resume2a=resume2a,
    )
    runtime_modules = stageo._runtime_modules()
    stagea = runtime_modules["stagel"].stagef.stagec258.stagea258
    stagea.validate_environment_payload(environment)
    cold_precheck = stagea.assert_cold_cuda_context_portable()

    repo = Path(root).resolve()
    base_report = stages.validate_base_report(repo)
    base_functional = stages_resume1.corrected_functional_projection(
        base_report,
        stages=stages,
    )
    base_functional_sha = stages.sha256_bytes(
        stages.stable_json_bytes(base_functional)
    )
    replay = stager_resume1.execute_recovery(
        root=repo,
        environment=environment,
        repository=_mapping(base_report.get("repository"), "base repository"),
    )
    functional = stages_resume1.corrected_functional_projection(
        replay,
        stages=stages,
    )
    fit_identity = stages.current_fit_projection(replay)
    functional_sha = stages.sha256_bytes(stages.stable_json_bytes(functional))
    if functional_sha != base_functional_sha:
        raise StageSResume3Error(
            "worker functional projection differs from Stage-R Resume1"
        )
    summary = stages.validate_confirmation_projection(functional)
    environment_sha = sha256_bytes(stable_json_bytes(environment))

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": WORKER_SCHEMA,
        "worker_id": str(worker_id),
        "process_id": os.getpid(),
        "execution_verdict": "PASS",
        "repository_head": str(repository_head),
        "environment": copy.deepcopy(dict(environment)),
        "environment_sha256": environment_sha,
        "cold_cuda_precheck": copy.deepcopy(dict(cold_precheck)),
        "environment_probe_rerun_in_science_process": False,
        "base_functional_projection_sha256": base_functional_sha,
        "functional_projection_sha256": functional_sha,
        "current_fit_projection_sha256": stages.sha256_bytes(
            stages.stable_json_bytes(fit_identity)
        ),
        "functional_projection_matches_base": True,
        "functional_projection": functional,
        "current_fit_projection": fit_identity,
        "confirmation_summary": summary,
        "scientific_oof_fit_count": EXPECTED_SCIENCE_FITS_PER_WORKER,
        "repeat_identity_fit_count": EXPECTED_REPEAT_FITS_PER_WORKER,
        "total_oof_fit_count": EXPECTED_FITS_PER_WORKER,
        "callback_pair_count": EXPECTED_CALLBACK_PAIRS_PER_WORKER,
        "oracle_callback_rerun_count": 0,
        "internal_scale_attempt_count": EXPECTED_INTERNAL_ATTEMPTS_PER_WORKER,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    validate_cold_science_worker_payload(
        result,
        expected_environment_sha256=environment_sha,
        expected_base_projection_sha256=base_functional_sha,
    )
    return result


def _run_child(
    command: Sequence[str],
    *,
    root: Path,
    label: str,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(command),
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
        check=False,
    )
    if completed.returncode != 0:
        raise StageSResume3Error(
            f"{label} rc={completed.returncode} "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return completed


def _worker_summary(worker: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "worker_id": worker.get("worker_id"),
        "process_id": worker.get("process_id"),
        "cold_cuda_before_replay": _mapping(
            worker.get("cold_cuda_precheck"), "worker cold precheck"
        ).get("torch_cuda_is_initialized")
        is False,
        "environment_sha256": worker.get("environment_sha256"),
        "functional_projection_sha256": worker.get(
            "functional_projection_sha256"
        ),
        "current_fit_projection_sha256": worker.get(
            "current_fit_projection_sha256"
        ),
        "functional_projection_matches_base": True,
        "scientific_oof_fit_count": worker.get("scientific_oof_fit_count"),
        "repeat_identity_fit_count": worker.get("repeat_identity_fit_count"),
        "total_oof_fit_count": worker.get("total_oof_fit_count"),
        "callback_pair_count": worker.get("callback_pair_count"),
        "oracle_callback_rerun_count": worker.get("oracle_callback_rerun_count"),
        "internal_scale_attempt_count": worker.get("internal_scale_attempt_count"),
    }


def validate_confirmation_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(summary, "confirmation summary")
    required = {
        "cell_count": EXPECTED_CELL_COUNT,
        "nonzero_support_cell_count": EXPECTED_NONZERO_SUPPORT_CELLS,
        "zero_acceptance_cell_count": EXPECTED_ZERO_ACCEPTANCE_CELLS,
        "oracle_like_cell_count": EXPECTED_ORACLE_LIKE_CELLS,
        "dominant_discriminator": EXPECTED_DOMINANT_DISCRIMINATOR,
        "dominant_support_count": EXPECTED_DOMINANT_SUPPORT,
        "dominant_backbone_coverage": EXPECTED_DOMINANT_BACKBONE_COVERAGE,
        "dominant_timestep_coverage": EXPECTED_DOMINANT_TIMESTEP_COVERAGE,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise StageSResume3Error(
                f"confirmation summary changed: {key}={value.get(key)!r}"
            )
    return value


def run_confirmation(
    *,
    root: Path,
    repository: Mapping[str, Any],
    python_bin: str,
) -> Mapping[str, Any]:
    validate_environment_variables()
    repo = Path(root).resolve()

    from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages
    from ccda_phase3 import phase314b_r258_stages_resume1_oracle_rerun_schema_recovery as stages_resume1
    from ccda_phase3 import phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a

    base_report = _mapping(repository.get("base_report"), "base report")
    base_projection = stages_resume1.corrected_functional_projection(
        base_report,
        stages=stages,
    )
    base_projection_sha = stages.sha256_bytes(
        stages.stable_json_bytes(base_projection)
    )
    probe_script = repo / "scripts/phase3_14b_r258_stages_resume2a_worker.py"
    science_script = repo / "scripts/phase3_14b_r258_stages_resume3_worker.py"

    with tempfile.TemporaryDirectory(
        prefix="phase314b_r258_stages_resume3_"
    ) as temporary:
        directory = Path(temporary)
        probe_path = directory / "environment_probe.json"
        probe_command = [
            str(python_bin),
            str(probe_script),
            "--mode",
            "environment-probe",
            "--root",
            str(repo),
            "--output",
            str(probe_path),
        ]
        _run_child(probe_command, root=repo, label="disposable environment probe")
        probe_payload = load_json(probe_path)
        environment = validate_probe_payload_for_science(
            probe_payload,
            resume2a=resume2a,
        )
        environment_sha = sha256_bytes(stable_json_bytes(environment))
        probe_pid = probe_payload.get("process_id")
        if not isinstance(probe_pid, int):
            raise StageSResume3Error("probe process ID is invalid")

        workers: List[Mapping[str, Any]] = []
        worker_paths: List[Path] = []
        for index in range(EXPECTED_WORKER_COUNT):
            output = directory / f"cold_science_worker_{index}.json"
            worker_paths.append(output)
            command = [
                str(python_bin),
                str(science_script),
                "--root",
                str(repo),
                "--worker-id",
                f"worker-{index + 1}",
                "--probe",
                str(probe_path),
                "--output",
                str(output),
            ]
            _run_child(
                command,
                root=repo,
                label=f"cold science worker {index + 1}",
            )
            worker = validate_cold_science_worker_payload(
                load_json(output),
                expected_environment_sha256=environment_sha,
                expected_base_projection_sha256=base_projection_sha,
            )
            if worker.get("repository_head") != repository.get("head"):
                raise StageSResume3Error("science worker repository HEAD changed")
            if worker.get("process_id") == probe_pid:
                raise StageSResume3Error(
                    "environment probe and science worker share a process"
                )
            workers.append(worker)

        probe_payload_sha = sha256_bytes(stable_json_bytes(probe_payload))
        worker_payload_shas = [
            sha256_bytes(stable_json_bytes(worker)) for worker in workers
        ]

    if len(workers) != EXPECTED_WORKER_COUNT:
        raise StageSResume3Error("science worker population changed")
    comparison = stages.compare_workers(workers[0], workers[1])
    if workers[0].get("functional_projection_sha256") != base_projection_sha:
        raise StageSResume3Error(
            "confirmed projection differs from frozen Stage-R Resume1"
        )
    summary = validate_confirmation_summary(
        _mapping(workers[0].get("confirmation_summary"), "confirmation summary")
    )
    classification = stages.classify_confirmation(summary)
    if classification.get("root_cause") != (
        "phase314b_r258_stages_tolerance_aligned_oof_support_confirmed"
    ):
        raise StageSResume3Error("Stage-S confirmation classification changed")
    if classification.get("required_next_path") != (
        "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_OOF_CANDIDATE_MATRIX_"
        "ON_OBJECTIVE_TRAIN_ONLY"
    ):
        raise StageSResume3Error("Stage-S confirmation next path changed")

    result: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stages_resume3_tolerance_aligned_oof_support_confirmed"
        ),
        "required_next_path": classification["required_next_path"],
        "primary_failure_locus": "confirmed_broad_oof_support",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": {
            key: value
            for key, value in repository.items()
            if key not in {"base_report", "resume2a_report"}
        },
        "immutable_inputs": {
            "resume2a_implementation_commit": BASE_RESUME2A_IMPLEMENTATION_COMMIT,
            "resume2a_evidence_commit": BASE_RESUME2A_EVIDENCE_COMMIT,
            "resume2a_report_sha256": EXPECTED_RESUME2A_REPORT_SHA256,
            "resume2a_scientific_result_sha256": (
                EXPECTED_RESUME2A_SCIENTIFIC_SHA256
            ),
            "stage_r_resume1_report_sha256": stages.EXPECTED_BASE_REPORT_SHA256,
            "stage_r_resume1_payload_sha256": (
                stages.EXPECTED_BASE_STAGE_R_PAYLOAD_SHA256
            ),
            "stage_r_resume1_scientific_result_sha256": (
                stages.EXPECTED_BASE_SCIENTIFIC_SHA256
            ),
            "objective_train_only": True,
            "oof_cell_count": EXPECTED_CELL_COUNT,
            "worker_count": EXPECTED_WORKER_COUNT,
        },
        "process_topology": {
            "environment_probe_process_count": EXPECTED_PROBE_PROCESS_COUNT,
            "cold_science_worker_process_count": EXPECTED_WORKER_COUNT,
            "probe_process_id": probe_pid,
            "science_worker_process_ids": [
                worker.get("process_id") for worker in workers
            ],
            "probe_distinct_from_each_science_worker": True,
            "workers_launched_sequentially": True,
            "environment_probe_rerun_in_science_workers": False,
            "environment_probe_payload_sha256": probe_payload_sha,
            "environment_sha256": environment_sha,
            "temporary_payloads_deleted": True,
        },
        "confirmation_execution": {
            "worker_count": EXPECTED_WORKER_COUNT,
            "workers": [_worker_summary(worker) for worker in workers],
            "worker_payload_sha256": worker_payload_shas,
            "worker_comparison": comparison,
            "workers_functional_projection_matches_base": True,
            "workers_current_fit_identity_byte_exact": True,
            "workers_cold_cuda_before_replay": True,
            "total_scientific_oof_fit_count": EXPECTED_TOTAL_SCIENCE_FITS,
            "total_repeat_identity_fit_count": EXPECTED_TOTAL_REPEAT_FITS,
            "total_oof_fit_count": EXPECTED_TOTAL_FITS,
            "total_callback_pair_count": EXPECTED_TOTAL_CALLBACK_PAIRS,
            "oracle_callback_rerun_count": 0,
            "total_internal_scale_attempt_count": EXPECTED_TOTAL_INTERNAL_ATTEMPTS,
            "worker_outputs_persisted": False,
        },
        "confirmation_summary": copy.deepcopy(dict(summary)),
        "confirmed_functional_projection_sha256": base_projection_sha,
        "confirmed_functional_projection": workers[0]["functional_projection"],
        "mechanism_boundary": {
            "stagee_modified": False,
            "stageh_modified": False,
            "stagek_modified": False,
            "stagel_modified": False,
            "stageq_modified": False,
            "stageq_resume1_modified": False,
            "stager_modified": False,
            "stager_resume1_modified": False,
            "stages_modified": False,
            "stages_resume1_modified": False,
            "stages_resume2a_modified": False,
            "legacy_upper_gate_changed": False,
            "aligned_upper_gate_changed": False,
            "aligned_upper_gate_written_to_stagee": False,
            "segment_reconstruction_changed": False,
            "integrator_thresholds_changed": False,
            "integrator_scale_search_changed": False,
            "external_scale_bank_changed": False,
            "oof_backbone_population_changed": False,
            "historical_prediction_sha_required": False,
            "cross_worker_current_fit_identity_required": True,
            "cross_worker_functional_identity_required": True,
            "cold_cuda_gate_removed": False,
            "required_operation_dry_run_skipped": False,
        },
        **{key: False for key in FALSE_BOUNDARIES},
    }
    result["scientific_result_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


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
        "root_cause": (
            "phase314b_r258_stages_resume3_cold_worker_confirmation_failed"
        ),
        "required_next_path": (
            "RESTORE_STAGES_RESUME3_COLD_WORKER_OOF_CONFIRMATION"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": None
        if repository is None
        else {
            key: value
            for key, value in repository.items()
            if key not in {"base_report", "resume2a_report"}
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        "resume2a_report_sha256": EXPECTED_RESUME2A_REPORT_SHA256,
        "stage_r_resume1_report_preserved": True,
        "complete_worker_population_claimed": False,
        "total_fit_count_claimed": False,
        "total_callback_count_claimed": False,
        "total_attempt_count_claimed": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
