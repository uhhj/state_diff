"""Stage-B byte-for-byte attestation of two independent external backups.

Stage A generated, audited, sealed and exported the r2.6.0 data universe but
left model access scientifically blocked until two independent external copies
were demonstrated.  Stage B never opens role NPZ files, episode pickles or
model targets.  It validates the Stage-A report and seal, derives a portable
backup contract for the seven required files, and adjudicates two independently
produced site attestations.

A site attestation is produced by streaming every required file at the backup
site.  Every deterministic tar archive is fully read and checked for unsafe
member paths, links, devices, duplicate names and truncation.  The site worker
also records filesystem/device observations and an explicit operator-declared
failure-domain identity.  The formal controller accepts the two attestations
only when all file identities match the Stage-A source contract and the two
failure domains are demonstrably distinct under the declared storage class.

The Stage-A seal is immutable and continues to state that backup had not yet
been attested at seal time.  Stage B does not rewrite it; the Stage-B evidence
report is the superseding governance record.  A successful report authorizes a
future, separately preregistered stage to open *objective_train only*.  It does
not itself load objective-train or authorize selection_holdout, frozen_probe or
final_evaluation access.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import plistlib
import re
import socket
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PHASE = "Phase3.14b-r2.6.0 Stage B"
SCHEMA = "phase314b_r260_stageb_external_backup_attestation_v1"
BLOCKED_SCHEMA = SCHEMA + "_blocked_v1"
CONTRACT_SCHEMA = "phase314b_r260_stageb_backup_contract_v1"
SITE_SCHEMA = "phase314b_r260_stageb_backup_site_attestation_v1"

BASE_HEAD = "636a2853db1a972073e2dd254e75a9724a5a1d17"
BASE_IMPLEMENTATION = "b7ca7bc3396c0879f13662fd953881a9f42e40f2"
EXPECTED_SUBMODULE = "633a88752445cf5d6776ed374fdbbdb35f93050c"

STAGEA_IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.6.0 Stage A: generate and seal new data universe"
)
STAGEA_EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.6.0 Stage A data-universe evidence"
)
IMPLEMENTATION_SUBJECT = (
    "Phase3.14b-r2.6.0 Stage B: attest independent external backups"
)
EVIDENCE_SUBJECT = (
    "Record Phase3.14b-r2.6.0 Stage B external-backup evidence"
)
BLOCKED_SUBJECT = "Record Phase3.14b-r2.6.0 Stage B blocked evidence"

IMPLEMENTATION_PATHS: Tuple[Tuple[str, str], ...] = (
    ("A", "ccda_phase3/phase314b_r260_stageb_external_backup_attestation.py"),
    ("A", "scripts/phase3_14b_r260_stageb_prepare_contract.py"),
    ("A", "scripts/phase3_14b_r260_stageb_site_inspect.py"),
    ("A", "scripts/phase3_14b_r260_stageb_execute.py"),
    ("A", "tests/test_phase3_14b_r260_stageb_external_backup_attestation.py"),
)

DATASET_RELATIVE_ROOT = "data/phase3_14b_r260_data_universe_v1"
SEAL_RELATIVE = DATASET_RELATIVE_ROOT + "/seal_manifest.json"
EXPORTS_RELATIVE_ROOT = DATASET_RELATIVE_ROOT + "/exports"
STAGEA_REPORT = "reports/phase3_14b_r260_stagea_data_universe_genesis_summary.json"
STAGEA_REPORT_FILE_SHA256 = (
    "60f9c2bc14d393dae061b2ac7ac3937e5ddf869c9841ba3ad834e300dfefd2c9"
)
SUCCESS_REPORT = "reports/phase3_14b_r260_stageb_external_backup_attestation_summary.json"
BLOCKED_REPORT = "reports/phase3_14b_r260_stageb_external_backup_attestation_blocked_summary.json"

EXPECTED_ARCHIVE_ROLES: Tuple[str, ...] = (
    "objective_train",
    "selection_holdout",
    "frozen_probe",
    "final_evaluation",
    "metadata_and_receipts",
)
EXPECTED_ARCHIVE_BASENAMES: Mapping[str, str] = {
    role: "{}.tar.gz".format(role) for role in EXPECTED_ARCHIVE_ROLES
}
EXPECTED_REQUIRED_FILE_COUNT = 7

FAULT_DOMAIN_KINDS: Tuple[str, ...] = (
    "physical_disk",
    "nas",
    "remote_object_store",
    "independent_host",
)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+\-=]{2,255}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


STAGEA_FALSE_BOUNDARIES: Tuple[str, ...] = (
    "legacy_r255_cache_loaded",
    "legacy_r256_cache_loaded",
    "legacy_stagei_data_loaded",
    "risk_fit_run",
    "direction_fit_run",
    "candidate_generation_run",
    "candidate_evaluation_run",
    "selection_holdout_model_evaluation_run",
    "frozen_probe_model_evaluation_run",
    "final_evaluation_model_evaluation_run",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_saved",
)

FALSE_BOUNDARIES: Tuple[str, ...] = (
    "objective_train_opened",
    "objective_train_model_fit_run",
    "selection_holdout_opened",
    "selection_holdout_model_evaluation_run",
    "frozen_probe_opened",
    "frozen_probe_model_evaluation_run",
    "final_evaluation_opened",
    "final_evaluation_model_evaluation_run",
    "role_npz_parsed",
    "episode_pickle_loaded",
    "risk_fit_run",
    "direction_fit_run",
    "candidate_generation_run",
    "candidate_evaluation_run",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_saved",
)


class StageBError(RuntimeError):
    """Fail-closed Stage-B error."""


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


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise StageBError("JSON root is not a mapping: {}".format(path))
    return value


def atomic_write_once(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise StageBError("write-once output exists: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        try:
            descriptor = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
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
            raise StageBError("unexpected commit path record: {!r}".format(line))
        records.append((fields[0], fields[1]))
    return tuple(sorted(records))


def assert_clean_worktree(root: Path, label: str) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise StageBError("{} worktree is dirty".format(label))


def _require_hex64(value: Any, label: str) -> str:
    rendered = str(value)
    if not HEX64_RE.fullmatch(rendered):
        raise StageBError("{} is not a lowercase SHA256".format(label))
    return rendered


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    observed = _require_hex64(payload.get(field), "{} {}".format(label, field))
    expected = sha256_bytes(
        stable_json_bytes({key: value for key, value in payload.items() if key != field})
    )
    if observed != expected:
        raise StageBError("{} self-hash changed".format(label))
    return observed


def _safe_relative(path: str) -> str:
    if not path or "\\" in path:
        raise StageBError("unsafe or non-POSIX relative path: {!r}".format(path))
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != path
    ):
        raise StageBError("unsafe relative path: {!r}".format(path))
    return pure.as_posix()


def validate_stagea_report_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "phase": "Phase3.14b-r2.6.0 Stage A",
        "schema": "phase314b_r260_stagea_data_universe_genesis_v1",
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "external_redundant_persistence_not_yet_attested",
        "required_next_path": (
            "ATTEST_TWO_INDEPENDENT_EXTERNAL_COPIES_OF_R260_DATA_UNIVERSE_"
            "BEFORE_ANY_MODEL_FIT_OR_HOLDOUT_ACCESS"
        ),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        "external_backup_attested": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageBError("Stage-A report field changed: {}".format(key))
    _validate_self_hash(payload, "summary_sha256", "Stage-A report")
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        raise StageBError("Stage-A repository record is missing")
    if repository.get("head") != BASE_IMPLEMENTATION:
        raise StageBError("Stage-A report implementation identity changed")
    if repository.get("parent") != "912870644734e49b9c0f34b6e87741c7fb63bda5":
        raise StageBError("Stage-A report parent identity changed")
    if repository.get("submodule_commit") != EXPECTED_SUBMODULE:
        raise StageBError("Stage-A report submodule identity changed")
    seal = payload.get("data_universe_seal")
    if not isinstance(seal, Mapping):
        raise StageBError("Stage-A embedded seal is missing")
    validate_stagea_seal_payload(seal)
    for key in STAGEA_FALSE_BOUNDARIES:
        if payload.get(key) is not False:
            raise StageBError("Stage-A report crossed forbidden boundary: {}".format(key))
    return seal


def validate_stagea_seal_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "phase314b_r260_stagea_data_universe_seal_v1":
        raise StageBError("Stage-A seal schema changed")
    if payload.get("dataset_relative_root") != DATASET_RELATIVE_ROOT:
        raise StageBError("Stage-A dataset root changed")
    _validate_self_hash(payload, "seal_sha256", "Stage-A seal")
    governance = payload.get("governance")
    if not isinstance(governance, Mapping):
        raise StageBError("Stage-A seal governance is missing")
    required_true = (
        "legacy_r255_r259_data_universe_abandoned",
        "legacy_cache_reconstruction_terminated",
        "new_role_seed_contract_frozen_before_generation",
        "role_regeneration_forbidden_after_terminal_report",
        "external_redundant_backup_required",
    )
    if any(governance.get(key) is not True for key in required_true):
        raise StageBError("Stage-A seal governance true field changed")
    if governance.get("external_redundant_backup_attested") is not False:
        raise StageBError("Stage-A seal was unexpectedly rewritten after generation")
    for key in (
        "objective_train_model_fit_count",
        "selection_holdout_model_evaluation_count",
        "frozen_probe_model_evaluation_count",
        "final_evaluation_model_evaluation_count",
    ):
        if governance.get(key) != 0:
            raise StageBError("Stage-A model access count changed")
    exports = payload.get("exports")
    if not isinstance(exports, Mapping):
        raise StageBError("Stage-A export manifest is missing")
    if exports.get("schema") != "phase314b_r260_stagea_export_manifest_v1":
        raise StageBError("Stage-A export schema changed")
    if exports.get("archive_count") != len(EXPECTED_ARCHIVE_ROLES):
        raise StageBError("Stage-A export archive count changed")
    if exports.get("external_copy_count_attested") != 0:
        raise StageBError("Stage-A export manifest unexpectedly attested a copy")
    records = exports.get("records")
    if not isinstance(records, Sequence) or len(records) != len(EXPECTED_ARCHIVE_ROLES):
        raise StageBError("Stage-A export records changed")
    observed_roles = []
    for record in records:
        if not isinstance(record, Mapping):
            raise StageBError("invalid Stage-A export record")
        role = str(record.get("role"))
        observed_roles.append(role)
        if role not in EXPECTED_ARCHIVE_ROLES:
            raise StageBError("unexpected Stage-A export role: {}".format(role))
        path = _safe_relative(str(record.get("path")))
        if Path(path).name != EXPECTED_ARCHIVE_BASENAMES[role]:
            raise StageBError("Stage-A export basename changed: {}".format(role))
        if int(record.get("size_bytes", -1)) <= 0:
            raise StageBError("Stage-A export size is invalid")
        _require_hex64(record.get("sha256"), "Stage-A export SHA")
    if tuple(observed_roles) != EXPECTED_ARCHIVE_ROLES:
        raise StageBError("Stage-A export role ordering changed")
    expected_population = sha256_bytes(stable_json_bytes(list(records)))
    if exports.get("all_archive_identity_sha256") != expected_population:
        raise StageBError("Stage-A export population hash changed")


def validate_repository(root: Path, implementation_commit: str) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    checks = {
        "branch": (_git(repo, "branch", "--show-current"), "Experiment1"),
        "HEAD": (_git(repo, "rev-parse", "HEAD"), implementation_commit),
        "Stage-B parent": (_git(repo, "rev-parse", "HEAD^"), BASE_HEAD),
        "Stage-A evidence parent": (
            _git(repo, "rev-parse", BASE_HEAD + "^"),
            BASE_IMPLEMENTATION,
        ),
        "Stage-A implementation subject": (
            _git(repo, "show", "-s", "--format=%s", BASE_IMPLEMENTATION),
            STAGEA_IMPLEMENTATION_SUBJECT,
        ),
        "Stage-A evidence subject": (
            _git(repo, "show", "-s", "--format=%s", BASE_HEAD),
            STAGEA_EVIDENCE_SUBJECT,
        ),
        "Stage-B implementation subject": (
            _git(repo, "show", "-s", "--format=%s", implementation_commit),
            IMPLEMENTATION_SUBJECT,
        ),
        "submodule gitlink": (
            _git(repo, "rev-parse", "HEAD:external/deformable-ravens"),
            EXPECTED_SUBMODULE,
        ),
    }
    for label, pair in checks.items():
        observed, expected = pair
        if observed != expected:
            raise StageBError(
                "{} changed: expected={} actual={}".format(label, expected, observed)
            )
    if commit_name_status(repo, implementation_commit) != tuple(sorted(IMPLEMENTATION_PATHS)):
        raise StageBError("Stage-B implementation population changed")
    submodule = repo / "external/deformable-ravens"
    if _git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE:
        raise StageBError("DeformableRavens worktree identity changed")
    assert_clean_worktree(repo, "main")
    assert_clean_worktree(submodule, "DeformableRavens")
    for _, relative in IMPLEMENTATION_PATHS:
        path = repo / relative
        if not path.is_file():
            raise StageBError("Stage-B implementation file missing: {}".format(relative))
        committed = _git_bytes(repo, "show", "{}:{}".format(implementation_commit, relative))
        if committed != path.read_bytes():
            raise StageBError("Stage-B implementation differs from commit: {}".format(relative))
    if (repo / SUCCESS_REPORT).exists() or (repo / BLOCKED_REPORT).exists():
        raise StageBError("Stage-B terminal report already exists")
    return {
        "root": str(repo),
        "head": implementation_commit,
        "parent": BASE_HEAD,
        "stage_a_implementation": BASE_IMPLEMENTATION,
        "stage_a_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def validate_stagea_source(root: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    repo = Path(root).resolve()
    report_path = repo / STAGEA_REPORT
    if not report_path.is_file():
        raise StageBError("Stage-A report is missing")
    if sha256_file(report_path) != STAGEA_REPORT_FILE_SHA256:
        raise StageBError("Stage-A report file SHA changed")
    committed = _git_bytes(repo, "show", "{}:{}".format(BASE_HEAD, STAGEA_REPORT))
    if committed != report_path.read_bytes():
        raise StageBError("Stage-A report differs from evidence commit")
    report = load_json(report_path)
    seal = validate_stagea_report_payload(report)
    seal_path = repo / SEAL_RELATIVE
    if not seal_path.is_file():
        raise StageBError("Stage-A seal file is missing")
    seal_file = load_json(seal_path)
    if seal_file != seal:
        raise StageBError("Stage-A seal file differs from embedded report seal")
    if stable_json_bytes(seal_file) != seal_path.read_bytes():
        raise StageBError("Stage-A seal file is not canonical JSON")
    exports = seal["exports"]
    for record in exports["records"]:
        path = repo / str(record["path"])
        if not path.is_file() or path.is_symlink():
            raise StageBError("Stage-A source export is missing or symlinked")
        if path.stat().st_size != int(record["size_bytes"]):
            raise StageBError("Stage-A source export size changed")
        if sha256_file(path) != str(record["sha256"]):
            raise StageBError("Stage-A source export SHA changed")
    return report, seal


def _validate_tar_member_name(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise StageBError("unsafe archive member name")
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != name
    ):
        raise StageBError("unsafe archive member path: {!r}".format(name))
    return pure.as_posix()


def scan_tar_gz(path: Path) -> Mapping[str, Any]:
    archive_path = Path(path)
    names: List[str] = []
    records: List[Mapping[str, Any]] = []
    seen = set()
    total_uncompressed = 0
    try:
        with tarfile.open(str(archive_path), mode="r:gz") as archive:
            for member in archive:
                name = _validate_tar_member_name(member.name)
                if name in seen:
                    raise StageBError("duplicate archive member: {}".format(name))
                seen.add(name)
                if not member.isfile():
                    raise StageBError("non-regular archive member: {}".format(name))
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise StageBError("unsafe archive member type: {}".format(name))
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise StageBError("archive member cannot be read: {}".format(name))
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                if size != int(member.size):
                    raise StageBError("archive member is truncated: {}".format(name))
                names.append(name)
                total_uncompressed += size
                records.append(
                    {
                        "path": name,
                        "size_bytes": size,
                        "sha256": digest.hexdigest(),
                        "mode": int(member.mode),
                        "mtime": int(member.mtime),
                        "uid": int(member.uid),
                        "gid": int(member.gid),
                    }
                )
    except (tarfile.TarError, EOFError, OSError) as error:
        raise StageBError("archive read failed: {}".format(error))
    if not records:
        raise StageBError("archive contains no regular files")
    return {
        "member_count": len(records),
        "total_uncompressed_bytes": total_uncompressed,
        "member_names_sha256": sha256_bytes(stable_json_bytes(names)),
        "member_content_manifest_sha256": sha256_bytes(stable_json_bytes(records)),
        "all_members_regular": True,
        "all_member_paths_safe": True,
        "duplicate_member_count": 0,
        "full_stream_read_completed": True,
    }


def build_backup_contract(root: Path) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    report, seal = validate_stagea_source(repo)
    required_files: List[Mapping[str, Any]] = []
    for record in seal["exports"]["records"]:
        role = str(record["role"])
        source_path = _safe_relative(str(record["path"]))
        path = repo / source_path
        required_files.append(
            {
                "logical_name": "archive:{}".format(role),
                "basename": EXPECTED_ARCHIVE_BASENAMES[role],
                "source_relative_path": source_path,
                "size_bytes": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
                "media_type": "application/gzip",
                "archive_scan": scan_tar_gz(path),
            }
        )
    seal_path = repo / SEAL_RELATIVE
    required_files.append(
        {
            "logical_name": "stage_a_seal",
            "basename": seal_path.name,
            "source_relative_path": SEAL_RELATIVE,
            "size_bytes": seal_path.stat().st_size,
            "sha256": sha256_file(seal_path),
            "media_type": "application/json",
            "json_self_hash_field": "seal_sha256",
            "json_self_sha256": seal["seal_sha256"],
        }
    )
    report_path = repo / STAGEA_REPORT
    required_files.append(
        {
            "logical_name": "stage_a_report",
            "basename": report_path.name,
            "source_relative_path": STAGEA_REPORT,
            "size_bytes": report_path.stat().st_size,
            "sha256": STAGEA_REPORT_FILE_SHA256,
            "media_type": "application/json",
            "json_self_hash_field": "summary_sha256",
            "json_self_sha256": report["summary_sha256"],
        }
    )
    if len(required_files) != EXPECTED_REQUIRED_FILE_COUNT:
        raise StageBError("backup required-file population changed")
    basenames = [str(item["basename"]) for item in required_files]
    if len(set(basenames)) != len(basenames):
        raise StageBError("backup required basenames are not unique")
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": CONTRACT_SCHEMA,
        "contract_role": "portable_byte_exact_backup_contract",
        "source_repository": {
            "stage_a_implementation": BASE_IMPLEMENTATION,
            "stage_a_evidence": BASE_HEAD,
            "submodule_commit": EXPECTED_SUBMODULE,
        },
        "source_report": {
            "path": STAGEA_REPORT,
            "file_sha256": STAGEA_REPORT_FILE_SHA256,
            "self_sha256": report["summary_sha256"],
        },
        "source_seal": {
            "path": SEAL_RELATIVE,
            "file_sha256": sha256_file(seal_path),
            "self_sha256": seal["seal_sha256"],
        },
        "data_universe_identity": {
            "dataset_relative_root": DATASET_RELATIVE_ROOT,
            "dataset_schema": seal["dataset_schema"],
            "role_contract": copy.deepcopy(seal["role_contract"]),
            "all_archive_identity_sha256": seal["exports"]["all_archive_identity_sha256"],
        },
        "required_files": required_files,
        "required_file_count": len(required_files),
        "archive_count": len(EXPECTED_ARCHIVE_ROLES),
        "backup_policy": {
            "required_independent_copy_count": 2,
            "same_physical_disk_partitions_are_one_failure_domain": True,
            "site_read_back_required": True,
            "byte_exact_file_identity_required": True,
            "archive_full_stream_read_required": True,
            "archive_safe_path_audit_required": True,
            "role_payload_semantics_must_not_be_parsed": True,
        },
    }
    payload["contract_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_backup_contract(payload)
    return payload


def validate_backup_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise StageBError("backup contract schema changed")
    if payload.get("contract_role") != "portable_byte_exact_backup_contract":
        raise StageBError("backup contract role changed")
    if payload.get("required_file_count") != EXPECTED_REQUIRED_FILE_COUNT:
        raise StageBError("backup contract file count changed")
    if payload.get("archive_count") != len(EXPECTED_ARCHIVE_ROLES):
        raise StageBError("backup contract archive count changed")
    source = payload.get("source_repository")
    if not isinstance(source, Mapping):
        raise StageBError("backup source repository is missing")
    expected_source = {
        "stage_a_implementation": BASE_IMPLEMENTATION,
        "stage_a_evidence": BASE_HEAD,
        "submodule_commit": EXPECTED_SUBMODULE,
    }
    if dict(source) != expected_source:
        raise StageBError("backup source repository identity changed")
    source_report = payload.get("source_report")
    source_seal = payload.get("source_seal")
    if not isinstance(source_report, Mapping) or not isinstance(source_seal, Mapping):
        raise StageBError("backup source report/seal identity is missing")
    if source_report.get("path") != STAGEA_REPORT:
        raise StageBError("backup source report path changed")
    if source_report.get("file_sha256") != STAGEA_REPORT_FILE_SHA256:
        raise StageBError("backup source report identity changed")
    _require_hex64(source_report.get("self_sha256"), "backup source report self SHA")
    if source_seal.get("path") != SEAL_RELATIVE:
        raise StageBError("backup source seal path changed")
    _require_hex64(source_seal.get("file_sha256"), "backup source seal file SHA")
    _require_hex64(source_seal.get("self_sha256"), "backup source seal self SHA")
    universe = payload.get("data_universe_identity")
    if not isinstance(universe, Mapping):
        raise StageBError("backup data-universe identity is missing")
    if universe.get("dataset_relative_root") != DATASET_RELATIVE_ROOT:
        raise StageBError("backup data-universe root changed")
    _require_hex64(
        universe.get("all_archive_identity_sha256"),
        "backup archive population SHA",
    )
    records = payload.get("required_files")
    if not isinstance(records, Sequence) or len(records) != EXPECTED_REQUIRED_FILE_COUNT:
        raise StageBError("backup contract records are missing")
    logical_names = []
    basenames = []
    archive_roles = []
    for record in records:
        if not isinstance(record, Mapping):
            raise StageBError("invalid backup contract file record")
        logical = str(record.get("logical_name"))
        basename = str(record.get("basename"))
        if Path(basename).name != basename or basename in ("", ".", ".."):
            raise StageBError("invalid backup basename")
        logical_names.append(logical)
        basenames.append(basename)
        if int(record.get("size_bytes", -1)) <= 0:
            raise StageBError("invalid backup file size")
        _require_hex64(record.get("sha256"), "backup contract file SHA")
        media_type = record.get("media_type")
        if media_type == "application/gzip":
            if not logical.startswith("archive:"):
                raise StageBError("archive logical name changed")
            role = logical.split(":", 1)[1]
            archive_roles.append(role)
            if role not in EXPECTED_ARCHIVE_ROLES:
                raise StageBError("unexpected contract archive role")
            if basename != EXPECTED_ARCHIVE_BASENAMES[role]:
                raise StageBError("contract archive basename changed")
            validate_archive_scan(record.get("archive_scan"))
        elif media_type == "application/json":
            _require_hex64(record.get("json_self_sha256"), "backup JSON self SHA")
        else:
            raise StageBError("unexpected backup media type")
    if len(set(logical_names)) != len(logical_names) or len(set(basenames)) != len(basenames):
        raise StageBError("backup contract contains duplicate records")
    if tuple(archive_roles) != EXPECTED_ARCHIVE_ROLES:
        raise StageBError("backup contract archive order changed")
    policy = payload.get("backup_policy")
    expected_policy = {
        "required_independent_copy_count": 2,
        "same_physical_disk_partitions_are_one_failure_domain": True,
        "site_read_back_required": True,
        "byte_exact_file_identity_required": True,
        "archive_full_stream_read_required": True,
        "archive_safe_path_audit_required": True,
        "role_payload_semantics_must_not_be_parsed": True,
    }
    if not isinstance(policy, Mapping) or dict(policy) != expected_policy:
        raise StageBError("backup policy changed")
    _validate_self_hash(payload, "contract_sha256", "backup contract")


def validate_archive_scan(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise StageBError("archive scan is missing")
    required_true = (
        "all_members_regular",
        "all_member_paths_safe",
        "full_stream_read_completed",
    )
    if any(value.get(key) is not True for key in required_true):
        raise StageBError("archive safety/readback check failed")
    if value.get("duplicate_member_count") != 0:
        raise StageBError("archive duplicate-member count changed")
    if int(value.get("member_count", 0)) <= 0:
        raise StageBError("archive member count is invalid")
    if int(value.get("total_uncompressed_bytes", 0)) <= 0:
        raise StageBError("archive uncompressed size is invalid")
    _require_hex64(value.get("member_names_sha256"), "archive names SHA")
    _require_hex64(value.get("member_content_manifest_sha256"), "archive content SHA")


def validate_contract_against_source(payload: Mapping[str, Any], root: Path) -> None:
    validate_backup_contract(payload)
    rebuilt = build_backup_contract(Path(root).resolve())
    if rebuilt != payload:
        raise StageBError("portable backup contract differs from current Stage-A source")


def _run_json_command(command: Sequence[str]) -> Optional[Any]:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def collect_storage_observation(root: Path) -> Mapping[str, Any]:
    path = Path(root).resolve()
    stat_result = path.stat()
    system = platform.system().lower()
    observation: Dict[str, Any] = {
        "schema": "phase314b_r260_stageb_storage_observation_v1",
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "hostname_sha256": sha256_bytes(socket.gethostname().encode("utf-8")),
        "backup_root_resolved_sha256": sha256_bytes(str(path).encode("utf-8")),
        "filesystem_device_number": int(stat_result.st_dev),
        "automatic_device_identity_available": False,
        "automatic_device_fingerprint": None,
        "automatic_details_sha256": None,
        "sensitive_storage_identifiers_persisted": False,
    }
    details: Dict[str, Any] = {}
    fingerprint_components: List[str] = []
    if system == "windows":
        drive = path.drive.rstrip(":\\/")
        if drive:
            script = (
                "$ErrorActionPreference='Stop';"
                "$p=Get-Partition -DriveLetter '" + drive + "';"
                "$d=$p|Get-Disk;"
                "[PSCustomObject]@{disk_number=$d.Number;serial_number=$d.SerialNumber;"
                "unique_id=$d.UniqueId;friendly_name=$d.FriendlyName;bus_type=$d.BusType;"
                "partition_number=$p.PartitionNumber;drive_letter=$p.DriveLetter}|"
                "ConvertTo-Json -Compress"
            )
            value = _run_json_command(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]
            )
            if isinstance(value, Mapping):
                details = dict(value)
                for key in ("unique_id", "serial_number", "disk_number"):
                    rendered = str(value.get(key) or "").strip()
                    if rendered:
                        fingerprint_components.append("{}={}".format(key, rendered))
    elif system == "linux":
        try:
            completed = subprocess.run(
                ["findmnt", "-J", "-T", str(path), "-o", "SOURCE,TARGET,FSTYPE,MAJ:MIN"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            try:
                value = json.loads(completed.stdout)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, Mapping):
                filesystems = value.get("filesystems")
                if isinstance(filesystems, Sequence) and filesystems:
                    row = filesystems[0]
                    if isinstance(row, Mapping):
                        details["findmnt"] = dict(row)
                        for key in ("source", "maj:min", "fstype"):
                            rendered = str(row.get(key) or "").strip()
                            if rendered:
                                fingerprint_components.append("{}={}".format(key, rendered))
        value = _run_json_command(
            [
                "lsblk",
                "-J",
                "-o",
                "NAME,KNAME,PKNAME,TYPE,MODEL,SERIAL,WWN,TRAN,MAJ:MIN,MOUNTPOINTS",
            ]
        )
        if isinstance(value, Mapping):
            details["lsblk"] = value
    elif system == "darwin":
        try:
            completed = subprocess.run(
                ["diskutil", "info", "-plist", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            completed = None
        if completed is not None and completed.returncode == 0:
            try:
                value = plistlib.loads(completed.stdout)
            except Exception:
                value = None
            if isinstance(value, Mapping):
                details = {str(key): value[key] for key in value}
                for key in ("DeviceIdentifier", "DiskUUID", "VolumeUUID", "MediaUUID"):
                    rendered = str(value.get(key) or "").strip()
                    if rendered:
                        fingerprint_components.append("{}={}".format(key, rendered))
    if details:
        observation["automatic_details_sha256"] = sha256_bytes(stable_json_bytes(details))
    if fingerprint_components:
        observation["automatic_device_identity_available"] = True
        observation["automatic_device_fingerprint"] = sha256_bytes(
            stable_json_bytes(sorted(fingerprint_components))
        )
    return observation


def _identifier_sha256(value: Any, label: str) -> str:
    rendered = str(value or "").strip()
    if not SAFE_ID_RE.fullmatch(rendered):
        raise StageBError("invalid stable identifier: {}".format(label))
    return sha256_bytes(rendered.encode("utf-8"))


def sanitize_failure_domain_declaration(value: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_failure_domain_declaration(value)
    kind = str(value["fault_domain_kind"])
    payload: Dict[str, Any] = {
        "site_id": str(value["site_id"]),
        "fault_domain_kind": kind,
        "fault_domain_id_sha256": _identifier_sha256(
            value["fault_domain_id"], "fault_domain_id"
        ),
        "physical_device_id_sha256": None,
        "service_or_system_id_sha256": None,
        "location_description": str(value["location_description"]),
        "copy_completed": True,
        "read_back_completed": True,
        "operator_certifies_independent_failure_domain": True,
        "operator_understands_same_disk_partitions_are_not_independent": True,
        "stable_identifiers_persisted_as_sha256_only": True,
    }
    if kind == "physical_disk":
        payload["physical_device_id_sha256"] = _identifier_sha256(
            value["physical_device_id"], "physical_device_id"
        )
    else:
        payload["service_or_system_id_sha256"] = _identifier_sha256(
            value["service_or_system_id"], "service_or_system_id"
        )
    return payload


def validate_sanitized_failure_domain(value: Mapping[str, Any]) -> None:
    required_true = (
        "copy_completed",
        "read_back_completed",
        "operator_certifies_independent_failure_domain",
        "operator_understands_same_disk_partitions_are_not_independent",
        "stable_identifiers_persisted_as_sha256_only",
    )
    if any(value.get(key) is not True for key in required_true):
        raise StageBError("sanitized site declaration is incomplete")
    if not SAFE_ID_RE.fullmatch(str(value.get("site_id") or "")):
        raise StageBError("sanitized site ID is invalid")
    kind = value.get("fault_domain_kind")
    if kind not in FAULT_DOMAIN_KINDS:
        raise StageBError("sanitized failure-domain kind is invalid")
    _require_hex64(value.get("fault_domain_id_sha256"), "fault-domain ID SHA")
    if not str(value.get("location_description") or "").strip():
        raise StageBError("sanitized location description is missing")
    if kind == "physical_disk":
        _require_hex64(value.get("physical_device_id_sha256"), "physical device ID SHA")
        if value.get("service_or_system_id_sha256") is not None:
            raise StageBError("physical disk unexpectedly has a service identity")
    else:
        _require_hex64(value.get("service_or_system_id_sha256"), "service/system ID SHA")
        if value.get("physical_device_id_sha256") is not None:
            raise StageBError("non-disk site unexpectedly has a physical device identity")


def validate_failure_domain_declaration(value: Mapping[str, Any]) -> None:
    required_true = (
        "copy_completed",
        "read_back_completed",
        "operator_certifies_independent_failure_domain",
        "operator_understands_same_disk_partitions_are_not_independent",
    )
    if any(value.get(key) is not True for key in required_true):
        raise StageBError("site operator declaration is incomplete")
    for key in ("site_id", "fault_domain_id"):
        rendered = str(value.get(key) or "")
        if not SAFE_ID_RE.fullmatch(rendered):
            raise StageBError("invalid site declaration ID: {}".format(key))
    kind = value.get("fault_domain_kind")
    if kind not in FAULT_DOMAIN_KINDS:
        raise StageBError("unsupported failure-domain kind")
    if not str(value.get("location_description") or "").strip():
        raise StageBError("site location description is missing")
    if kind == "physical_disk":
        if not SAFE_ID_RE.fullmatch(str(value.get("physical_device_id") or "")):
            raise StageBError("physical disk declaration requires stable device ID")
    else:
        if not SAFE_ID_RE.fullmatch(str(value.get("service_or_system_id") or "")):
            raise StageBError("non-disk declaration requires service/system ID")


def build_site_attestation(
    *,
    contract: Mapping[str, Any],
    backup_root: Path,
    declaration: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_backup_contract(contract)
    sanitized_declaration = sanitize_failure_domain_declaration(declaration)
    root = Path(backup_root).resolve()
    if not root.is_dir():
        raise StageBError("backup root is not a directory")
    file_records: List[Mapping[str, Any]] = []
    for expected in contract["required_files"]:
        path = root / str(expected["basename"])
        if not path.is_file() or path.is_symlink():
            raise StageBError("backup file is missing or symlinked: {}".format(path.name))
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != int(expected["size_bytes"]):
            raise StageBError("backup file size differs: {}".format(path.name))
        if actual_sha != str(expected["sha256"]):
            raise StageBError("backup file SHA differs: {}".format(path.name))
        record: Dict[str, Any] = {
            "logical_name": expected["logical_name"],
            "basename": expected["basename"],
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "byte_exact_to_source": True,
        }
        if expected["media_type"] == "application/gzip":
            scan = scan_tar_gz(path)
            if scan != expected["archive_scan"]:
                raise StageBError("backup archive scan differs: {}".format(path.name))
            record["archive_scan"] = scan
        else:
            payload = load_json(path)
            field = str(expected["json_self_hash_field"])
            if _validate_self_hash(payload, field, path.name) != expected["json_self_sha256"]:
                raise StageBError("backup JSON self identity differs")
            record["json_self_sha256"] = payload[field]
        file_records.append(record)
    observation = collect_storage_observation(root)
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SITE_SCHEMA,
        "attestation_role": "external_backup_site_readback",
        "contract_sha256": contract["contract_sha256"],
        "failure_domain": copy.deepcopy(dict(sanitized_declaration)),
        "storage_observation": observation,
        "required_file_count": len(file_records),
        "file_records": file_records,
        "all_required_files_byte_exact": True,
        "all_archives_fully_streamed": True,
        "all_archive_paths_safe": True,
        "role_payload_semantics_parsed": False,
        "role_npz_opened": False,
        "episode_pickle_loaded": False,
        "model_target_access_count": 0,
    }
    payload["attestation_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_site_attestation(payload, contract)
    return payload


def validate_site_attestation(payload: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    validate_backup_contract(contract)
    if payload.get("schema") != SITE_SCHEMA:
        raise StageBError("site attestation schema changed")
    if payload.get("attestation_role") != "external_backup_site_readback":
        raise StageBError("site attestation role changed")
    if payload.get("contract_sha256") != contract.get("contract_sha256"):
        raise StageBError("site attestation references a different contract")
    declaration = payload.get("failure_domain")
    if not isinstance(declaration, Mapping):
        raise StageBError("site failure-domain declaration is missing")
    validate_sanitized_failure_domain(declaration)
    observation = payload.get("storage_observation")
    if not isinstance(observation, Mapping):
        raise StageBError("site storage observation is missing")
    if observation.get("sensitive_storage_identifiers_persisted") is not False:
        raise StageBError("site storage observation persisted sensitive identifiers")
    _require_hex64(observation.get("hostname_sha256"), "site hostname SHA")
    _require_hex64(
        observation.get("backup_root_resolved_sha256"), "site backup-root SHA"
    )
    if observation.get("automatic_details_sha256") is not None:
        _require_hex64(
            observation.get("automatic_details_sha256"), "automatic details SHA"
        )
    if payload.get("required_file_count") != EXPECTED_REQUIRED_FILE_COUNT:
        raise StageBError("site required-file count changed")
    records = payload.get("file_records")
    if not isinstance(records, Sequence) or len(records) != EXPECTED_REQUIRED_FILE_COUNT:
        raise StageBError("site file records are missing")
    expected_by_name = {item["logical_name"]: item for item in contract["required_files"]}
    observed_names = []
    for record in records:
        if not isinstance(record, Mapping):
            raise StageBError("invalid site file record")
        logical = str(record.get("logical_name"))
        observed_names.append(logical)
        expected = expected_by_name.get(logical)
        if expected is None:
            raise StageBError("site contains unexpected logical file")
        for key in ("basename", "size_bytes", "sha256"):
            if record.get(key) != expected.get(key):
                raise StageBError("site file identity differs: {}".format(logical))
        if record.get("byte_exact_to_source") is not True:
            raise StageBError("site file is not byte-exact")
        if expected["media_type"] == "application/gzip":
            if record.get("archive_scan") != expected.get("archive_scan"):
                raise StageBError("site archive readback differs")
            validate_archive_scan(record.get("archive_scan"))
        else:
            if record.get("json_self_sha256") != expected.get("json_self_sha256"):
                raise StageBError("site JSON self identity differs")
    if len(set(observed_names)) != len(observed_names):
        raise StageBError("site file records are duplicated")
    required_true = (
        "all_required_files_byte_exact",
        "all_archives_fully_streamed",
        "all_archive_paths_safe",
    )
    if any(payload.get(key) is not True for key in required_true):
        raise StageBError("site readback boundary changed")
    required_false = (
        "role_payload_semantics_parsed",
        "role_npz_opened",
        "episode_pickle_loaded",
    )
    if any(payload.get(key) is not False for key in required_false):
        raise StageBError("site inspector crossed semantic data boundary")
    if payload.get("model_target_access_count") != 0:
        raise StageBError("site inspector accessed model targets")
    _validate_self_hash(payload, "attestation_sha256", "site attestation")


def _failure_domain_key(site: Mapping[str, Any]) -> str:
    declaration = site["failure_domain"]
    return "{}:{}".format(
        declaration["fault_domain_kind"], declaration["fault_domain_id_sha256"]
    )


def adjudicate_sites(
    contract: Mapping[str, Any],
    site_a: Mapping[str, Any],
    site_b: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_site_attestation(site_a, contract)
    validate_site_attestation(site_b, contract)
    declaration_a = site_a["failure_domain"]
    declaration_b = site_b["failure_domain"]
    if declaration_a["site_id"] == declaration_b["site_id"]:
        raise StageBError("backup site IDs are identical")
    key_a = _failure_domain_key(site_a)
    key_b = _failure_domain_key(site_b)
    if key_a == key_b:
        raise StageBError("backup failure-domain identities are identical")
    if declaration_a["fault_domain_id_sha256"] == declaration_b["fault_domain_id_sha256"]:
        raise StageBError("backup fault-domain IDs are identical")
    kind_a = declaration_a["fault_domain_kind"]
    kind_b = declaration_b["fault_domain_kind"]
    if kind_a == kind_b == "physical_disk":
        if declaration_a["physical_device_id_sha256"] == declaration_b["physical_device_id_sha256"]:
            raise StageBError("two physical-disk copies use the same device identity")
    if kind_a == kind_b and kind_a != "physical_disk":
        if declaration_a["service_or_system_id_sha256"] == declaration_b["service_or_system_id_sha256"]:
            raise StageBError("two non-disk copies use the same service/system identity")
    observation_a = site_a["storage_observation"]
    observation_b = site_b["storage_observation"]
    if kind_a != "remote_object_store" and kind_b != "remote_object_store":
        fingerprint_a = observation_a.get("automatic_device_fingerprint")
        fingerprint_b = observation_b.get("automatic_device_fingerprint")
        if fingerprint_a and fingerprint_b and fingerprint_a == fingerprint_b:
            raise StageBError("automatic storage observation identifies one failure domain")
        if (
            observation_a.get("hostname_sha256") == observation_b.get("hostname_sha256")
            and observation_a.get("filesystem_device_number")
            == observation_b.get("filesystem_device_number")
        ):
            raise StageBError("backup copies resolve to the same host filesystem device")
    if site_a["file_records"] != site_b["file_records"]:
        raise StageBError("external copies do not have identical file/readback records")
    return {
        "classification": "two_byte_exact_independent_external_failure_domains_attested",
        "independent_external_copy_count": 2,
        "site_ids": [declaration_a["site_id"], declaration_b["site_id"]],
        "failure_domain_keys": [key_a, key_b],
        "failure_domain_kinds": [kind_a, kind_b],
        "same_physical_disk_partitions_rejected": True,
        "all_required_files_byte_exact": True,
        "all_archives_fully_streamed_at_both_sites": True,
        "all_archive_paths_safe_at_both_sites": True,
        "role_payload_semantics_parsed": False,
        "objective_train_opened": False,
    }


def build_summary(
    *,
    repository: Mapping[str, Any],
    contract: Mapping[str, Any],
    site_a: Mapping[str, Any],
    site_b: Mapping[str, Any],
) -> Mapping[str, Any]:
    audit = adjudicate_sites(contract, site_a, site_b)
    payload: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "primary_failure_locus": "external_redundant_persistence_attested",
        "root_cause": (
            "phase314b_r260_stageb_two_independent_external_failure_domains_"
            "contain_byte_exact_read_back_verified_data_universe_exports"
        ),
        "required_next_path": (
            "PREREGISTER_R260_STAGEC_OBJECTIVE_TRAIN_OPENING_AND_BASELINE_STATISTICS"
        ),
        "repository": copy.deepcopy(dict(repository)),
        "backup_contract": copy.deepcopy(dict(contract)),
        "site_attestations": [copy.deepcopy(dict(site_a)), copy.deepcopy(dict(site_b))],
        "backup_audit": audit,
        "external_backup_attested": True,
        "independent_external_copy_count": 2,
        "objective_train_access_authorized_by_this_report": True,
        "selection_holdout_access_authorized_by_this_report": False,
        "frozen_probe_access_authorized_by_this_report": False,
        "final_evaluation_access_authorized_by_this_report": False,
        "execution_counts": {
            "source_report_read_count": 1,
            "source_seal_read_count": 1,
            "source_export_hash_count": len(EXPECTED_ARCHIVE_ROLES),
            "external_site_attestation_count": 2,
            "external_file_readback_count": 2 * EXPECTED_REQUIRED_FILE_COUNT,
            "external_archive_full_stream_count": 2 * len(EXPECTED_ARCHIVE_ROLES),
            "role_npz_parse_count": 0,
            "episode_pickle_load_count": 0,
            "model_fit_count": 0,
            "model_evaluation_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
    payload["summary_sha256"] = sha256_bytes(stable_json_bytes(payload))
    validate_summary(payload)
    return payload


def validate_summary(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "external_backup_attested": True,
        "independent_external_copy_count": 2,
        "objective_train_access_authorized_by_this_report": True,
        "selection_holdout_access_authorized_by_this_report": False,
        "frozen_probe_access_authorized_by_this_report": False,
        "final_evaluation_access_authorized_by_this_report": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise StageBError("Stage-B summary field changed: {}".format(key))
    contract = payload.get("backup_contract")
    sites = payload.get("site_attestations")
    if not isinstance(contract, Mapping):
        raise StageBError("Stage-B summary contract is missing")
    if not isinstance(sites, Sequence) or len(sites) != 2:
        raise StageBError("Stage-B summary site attestations are missing")
    audit = adjudicate_sites(contract, sites[0], sites[1])
    if payload.get("backup_audit") != audit:
        raise StageBError("Stage-B summary backup audit changed")
    if any(payload.get(key) is not False for key in FALSE_BOUNDARIES):
        raise StageBError("Stage-B crossed forbidden boundary")
    expected_counts = {
        "source_report_read_count": 1,
        "source_seal_read_count": 1,
        "source_export_hash_count": len(EXPECTED_ARCHIVE_ROLES),
        "external_site_attestation_count": 2,
        "external_file_readback_count": 2 * EXPECTED_REQUIRED_FILE_COUNT,
        "external_archive_full_stream_count": 2 * len(EXPECTED_ARCHIVE_ROLES),
        "role_npz_parse_count": 0,
        "episode_pickle_load_count": 0,
        "model_fit_count": 0,
        "model_evaluation_count": 0,
    }
    if payload.get("execution_counts") != expected_counts:
        raise StageBError("Stage-B execution counts changed")
    _validate_self_hash(payload, "summary_sha256", "Stage-B summary")


def blocked_report(
    repository: Optional[Mapping[str, Any]], error: BaseException
) -> Mapping[str, Any]:
    return {
        "phase": PHASE,
        "schema": BLOCKED_SCHEMA,
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "external_backup_attestation_execution_contract",
        "root_cause": "phase314b_r260_stageb_external_backup_attestation_failed",
        "required_next_path": (
            "REPAIR_R260_EXTERNAL_BACKUP_COPY_OR_FAILURE_DOMAIN_EVIDENCE_"
            "WITHOUT_OPENING_ANY_DATA_ROLE"
        ),
        "repository": None if repository is None else dict(repository),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "external_backup_attested": False,
        "objective_train_access_authorized_by_this_report": False,
        "selection_holdout_access_authorized_by_this_report": False,
        "frozen_probe_access_authorized_by_this_report": False,
        "final_evaluation_access_authorized_by_this_report": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        **{key: False for key in FALSE_BOUNDARIES},
    }
