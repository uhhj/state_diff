from __future__ import annotations

import copy
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r260_stageb_external_backup_attestation as stageb


def _write_tar(path: Path, members=None):
    members = members or [("root/a.txt", b"a"), ("root/b.txt", b"bb")]
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            archive.addfile(info, io.BytesIO(payload))


def _json_with_hash(field: str, **extra):
    value = dict(extra)
    value[field] = stageb.sha256_bytes(stageb.stable_json_bytes(value))
    return value


def _write_json(path: Path, value):
    path.write_bytes(stageb.stable_json_bytes(value))


def _valid_contract(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    required = []
    for role in stageb.EXPECTED_ARCHIVE_ROLES:
        path = root / stageb.EXPECTED_ARCHIVE_BASENAMES[role]
        _write_tar(path, [("{}/x.bin".format(role), role.encode("ascii"))])
        required.append(
            {
                "logical_name": "archive:{}".format(role),
                "basename": path.name,
                "source_relative_path": "exports/{}".format(path.name),
                "size_bytes": path.stat().st_size,
                "sha256": stageb.sha256_file(path),
                "media_type": "application/gzip",
                "archive_scan": stageb.scan_tar_gz(path),
            }
        )
    seal = _json_with_hash("seal_sha256", schema="seal")
    seal_path = root / "seal_manifest.json"
    _write_json(seal_path, seal)
    required.append(
        {
            "logical_name": "stage_a_seal",
            "basename": seal_path.name,
            "source_relative_path": stageb.SEAL_RELATIVE,
            "size_bytes": seal_path.stat().st_size,
            "sha256": stageb.sha256_file(seal_path),
            "media_type": "application/json",
            "json_self_hash_field": "seal_sha256",
            "json_self_sha256": seal["seal_sha256"],
        }
    )
    report = _json_with_hash("summary_sha256", schema="report")
    report_path = root / "phase3_14b_r260_stagea_data_universe_genesis_summary.json"
    _write_json(report_path, report)
    required.append(
        {
            "logical_name": "stage_a_report",
            "basename": report_path.name,
            "source_relative_path": stageb.STAGEA_REPORT,
            "size_bytes": report_path.stat().st_size,
            "sha256": stageb.sha256_file(report_path),
            "media_type": "application/json",
            "json_self_hash_field": "summary_sha256",
            "json_self_sha256": report["summary_sha256"],
        }
    )
    payload = {
        "phase": stageb.PHASE,
        "schema": stageb.CONTRACT_SCHEMA,
        "contract_role": "portable_byte_exact_backup_contract",
        "source_repository": {
            "stage_a_implementation": stageb.BASE_IMPLEMENTATION,
            "stage_a_evidence": stageb.BASE_HEAD,
            "submodule_commit": stageb.EXPECTED_SUBMODULE,
        },
        "source_report": {
            "path": stageb.STAGEA_REPORT,
            "file_sha256": stageb.STAGEA_REPORT_FILE_SHA256,
            "self_sha256": "a" * 64,
        },
        "source_seal": {
            "path": stageb.SEAL_RELATIVE,
            "file_sha256": "b" * 64,
            "self_sha256": "c" * 64,
        },
        "data_universe_identity": {
            "dataset_relative_root": stageb.DATASET_RELATIVE_ROOT,
            "dataset_schema": "test",
            "role_contract": {},
            "all_archive_identity_sha256": "d" * 64,
        },
        "required_files": required,
        "required_file_count": stageb.EXPECTED_REQUIRED_FILE_COUNT,
        "archive_count": len(stageb.EXPECTED_ARCHIVE_ROLES),
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
    payload["contract_sha256"] = stageb.sha256_bytes(stageb.stable_json_bytes(payload))
    stageb.validate_backup_contract(payload)
    return root, payload


def _copy_required(source: Path, destination: Path, contract):
    destination.mkdir()
    for record in contract["required_files"]:
        (destination / record["basename"]).write_bytes(
            (source / record["basename"]).read_bytes()
        )


def _declaration(site="site_a", domain="domain_a", device="disk_a"):
    return {
        "site_id": site,
        "fault_domain_kind": "physical_disk",
        "fault_domain_id": domain,
        "physical_device_id": device,
        "service_or_system_id": None,
        "location_description": "independent physical disk",
        "copy_completed": True,
        "read_back_completed": True,
        "operator_certifies_independent_failure_domain": True,
        "operator_understands_same_disk_partitions_are_not_independent": True,
    }


def _site_pair(tmp_path: Path):
    source, contract = _valid_contract(tmp_path)
    copy_a = tmp_path / "copy_a"
    copy_b = tmp_path / "copy_b"
    _copy_required(source, copy_a, contract)
    _copy_required(source, copy_b, contract)
    site_a = stageb.build_site_attestation(
        contract=contract,
        backup_root=copy_a,
        declaration=_declaration(),
    )
    site_b = stageb.build_site_attestation(
        contract=contract,
        backup_root=copy_b,
        declaration=_declaration("site_b", "domain_b", "disk_b"),
    )
    site_a = copy.deepcopy(site_a)
    site_b = copy.deepcopy(site_b)
    site_a["storage_observation"]["automatic_device_fingerprint"] = "1" * 64
    site_b["storage_observation"]["automatic_device_fingerprint"] = "2" * 64
    site_a["storage_observation"]["filesystem_device_number"] = 1
    site_b["storage_observation"]["filesystem_device_number"] = 2
    _rehash(site_a, "attestation_sha256")
    _rehash(site_b, "attestation_sha256")
    return contract, site_a, site_b


def _rehash(value, field):
    value[field] = stageb.sha256_bytes(
        stageb.stable_json_bytes({k: v for k, v in value.items() if k != field})
    )


def _valid_stagea_seal():
    records = []
    for role in stageb.EXPECTED_ARCHIVE_ROLES:
        records.append(
            {
                "role": role,
                "path": stageb.EXPORTS_RELATIVE_ROOT + "/" + stageb.EXPECTED_ARCHIVE_BASENAMES[role],
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
        )
    exports = {
        "schema": "phase314b_r260_stagea_export_manifest_v1",
        "records": records,
        "archive_count": 5,
        "all_archive_identity_sha256": stageb.sha256_bytes(stageb.stable_json_bytes(records)),
        "external_copy_count_attested": 0,
    }
    value = {
        "schema": "phase314b_r260_stagea_data_universe_seal_v1",
        "dataset_schema": "r260-test",
        "dataset_relative_root": stageb.DATASET_RELATIVE_ROOT,
        "repository": {},
        "source_lock_sha256": "b" * 64,
        "role_contract": {},
        "role_manifests": {},
        "audit": {},
        "inventory_sha256": "c" * 64,
        "inventory_file_count": 1,
        "inventory_total_size_bytes": 1,
        "exports": exports,
        "governance": {
            "legacy_r255_r259_data_universe_abandoned": True,
            "legacy_cache_reconstruction_terminated": True,
            "new_role_seed_contract_frozen_before_generation": True,
            "objective_train_model_fit_count": 0,
            "selection_holdout_model_evaluation_count": 0,
            "frozen_probe_model_evaluation_count": 0,
            "final_evaluation_model_evaluation_count": 0,
            "selection_holdout_may_not_be_opened_before_policy_lock": True,
            "frozen_probe_may_not_be_opened_before_selection_lock": True,
            "final_evaluation_may_not_be_opened_before_final_preregistration": True,
            "role_regeneration_forbidden_after_terminal_report": True,
            "external_redundant_backup_required": True,
            "external_redundant_backup_attested": False,
        },
    }
    _rehash(value, "seal_sha256")
    return value


def _valid_stagea_report():
    value = {
        "phase": "Phase3.14b-r2.6.0 Stage A",
        "schema": "phase314b_r260_stagea_data_universe_genesis_v1",
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "primary_failure_locus": "external_redundant_persistence_not_yet_attested",
        "root_cause": "x",
        "required_next_path": "ATTEST_TWO_INDEPENDENT_EXTERNAL_COPIES_OF_R260_DATA_UNIVERSE_BEFORE_ANY_MODEL_FIT_OR_HOLDOUT_ACCESS",
        "repository": {
            "head": stageb.BASE_IMPLEMENTATION,
            "parent": "912870644734e49b9c0f34b6e87741c7fb63bda5",
            "submodule_commit": stageb.EXPECTED_SUBMODULE,
        },
        "data_universe_seal": _valid_stagea_seal(),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "rerun_authorized": False,
        "resume_authorized": False,
        "external_backup_attested": False,
    }
    value.update({key: False for key in stageb.STAGEA_FALSE_BOUNDARIES})
    _rehash(value, "summary_sha256")
    return value


def test_stable_json_is_deterministic():
    assert stageb.stable_json_bytes({"b": 1, "a": 2}) == stageb.stable_json_bytes({"a": 2, "b": 1})


def test_atomic_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    stageb.atomic_write_once(path, b"x")
    assert path.read_bytes() == b"x"
    with pytest.raises(stageb.StageBError):
        stageb.atomic_write_once(path, b"y")


@pytest.mark.parametrize("value", ["../x", "/x", "x\\y", "./x", "x/../y", "x//y"])
def test_safe_relative_rejects_bad_paths(value):
    with pytest.raises(stageb.StageBError):
        stageb._safe_relative(value)


@pytest.mark.parametrize("value", ["../x", "/x", "x\\y", "./x", "x/../y", "x//y"])
def test_archive_member_name_rejects_bad_paths(value):
    with pytest.raises(stageb.StageBError):
        stageb._validate_tar_member_name(value)


def test_safe_relative_accepts_canonical_path():
    assert stageb._safe_relative("x/y") == "x/y"


def test_archive_member_name_accepts_canonical_path():
    assert stageb._validate_tar_member_name("x/y") == "x/y"


def test_scan_tar_gz_reads_all_members(tmp_path: Path):
    path = tmp_path / "x.tar.gz"
    _write_tar(path)
    result = stageb.scan_tar_gz(path)
    assert result["member_count"] == 2
    assert result["total_uncompressed_bytes"] == 3
    assert result["full_stream_read_completed"] is True


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_scan_tar_rejects_unsafe_member_types(tmp_path: Path, kind):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("x")
        info.type = kind
        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            info.linkname = "y"
        archive.addfile(info)
    with pytest.raises(stageb.StageBError):
        stageb.scan_tar_gz(path)


def test_scan_tar_rejects_duplicate_members(tmp_path: Path):
    path = tmp_path / "dup.tar.gz"
    _write_tar(path, [("x", b"a"), ("x", b"b")])
    with pytest.raises(stageb.StageBError):
        stageb.scan_tar_gz(path)


def test_scan_tar_rejects_empty_archive(tmp_path: Path):
    path = tmp_path / "empty.tar.gz"
    with tarfile.open(path, "w:gz"):
        pass
    with pytest.raises(stageb.StageBError):
        stageb.scan_tar_gz(path)


def test_validate_stagea_seal_accepts_valid():
    stageb.validate_stagea_seal_payload(_valid_stagea_seal())


@pytest.mark.parametrize(
    "field",
    [
        "legacy_r255_r259_data_universe_abandoned",
        "legacy_cache_reconstruction_terminated",
        "new_role_seed_contract_frozen_before_generation",
        "role_regeneration_forbidden_after_terminal_report",
        "external_redundant_backup_required",
    ],
)
def test_stagea_seal_rejects_required_false(field):
    value = _valid_stagea_seal()
    value["governance"][field] = False
    _rehash(value, "seal_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.validate_stagea_seal_payload(value)


@pytest.mark.parametrize(
    "field",
    [
        "objective_train_model_fit_count",
        "selection_holdout_model_evaluation_count",
        "frozen_probe_model_evaluation_count",
        "final_evaluation_model_evaluation_count",
    ],
)
def test_stagea_seal_rejects_model_access(field):
    value = _valid_stagea_seal()
    value["governance"][field] = 1
    _rehash(value, "seal_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.validate_stagea_seal_payload(value)


def test_validate_stagea_report_accepts_valid():
    stageb.validate_stagea_report_payload(_valid_stagea_report())


@pytest.mark.parametrize("field", stageb.STAGEA_FALSE_BOUNDARIES)
def test_stagea_report_rejects_boundary(field):
    value = _valid_stagea_report()
    value[field] = True
    _rehash(value, "summary_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.validate_stagea_report_payload(value)


def test_contract_accepts_valid(tmp_path: Path):
    _, contract = _valid_contract(tmp_path)
    stageb.validate_backup_contract(contract)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(schema="bad"),
        lambda x: x.update(required_file_count=6),
        lambda x: x.update(archive_count=4),
        lambda x: x["backup_policy"].update(site_read_back_required=False),
        lambda x: x["source_repository"].update(stage_a_evidence="0" * 40),
    ],
)
def test_contract_rejects_mutations(tmp_path: Path, mutation):
    _, contract = _valid_contract(tmp_path)
    value = copy.deepcopy(contract)
    mutation(value)
    _rehash(value, "contract_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.validate_backup_contract(value)


@pytest.mark.parametrize("kind", stageb.FAULT_DOMAIN_KINDS)
def test_failure_domain_kinds(kind):
    value = _declaration()
    value["fault_domain_kind"] = kind
    if kind != "physical_disk":
        value["physical_device_id"] = None
        value["service_or_system_id"] = "service_a"
    stageb.validate_failure_domain_declaration(value)


@pytest.mark.parametrize(
    "field",
    [
        "copy_completed",
        "read_back_completed",
        "operator_certifies_independent_failure_domain",
        "operator_understands_same_disk_partitions_are_not_independent",
    ],
)
def test_declaration_requires_true_fields(field):
    value = _declaration()
    value[field] = False
    with pytest.raises(stageb.StageBError):
        stageb.validate_failure_domain_declaration(value)


@pytest.mark.parametrize("field", ["site_id", "fault_domain_id"])
def test_declaration_rejects_bad_id(field):
    value = _declaration()
    value[field] = "bad id"
    with pytest.raises(stageb.StageBError):
        stageb.validate_failure_domain_declaration(value)


def test_build_site_attestation(tmp_path: Path):
    source, contract = _valid_contract(tmp_path)
    backup = tmp_path / "backup"
    _copy_required(source, backup, contract)
    value = stageb.build_site_attestation(
        contract=contract, backup_root=backup, declaration=_declaration()
    )
    stageb.validate_site_attestation(value, contract)
    assert value["required_file_count"] == 7
    assert value["role_npz_opened"] is False


def test_site_rejects_missing_file(tmp_path: Path):
    source, contract = _valid_contract(tmp_path)
    backup = tmp_path / "backup"
    _copy_required(source, backup, contract)
    (backup / contract["required_files"][0]["basename"]).unlink()
    with pytest.raises(stageb.StageBError):
        stageb.build_site_attestation(
            contract=contract, backup_root=backup, declaration=_declaration()
        )


def test_site_rejects_corrupt_file(tmp_path: Path):
    source, contract = _valid_contract(tmp_path)
    backup = tmp_path / "backup"
    _copy_required(source, backup, contract)
    with (backup / contract["required_files"][0]["basename"]).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(stageb.StageBError):
        stageb.build_site_attestation(
            contract=contract, backup_root=backup, declaration=_declaration()
        )


def test_site_rejects_symlink(tmp_path: Path):
    source, contract = _valid_contract(tmp_path)
    backup = tmp_path / "backup"
    _copy_required(source, backup, contract)
    name = contract["required_files"][0]["basename"]
    target = backup / name
    payload = target.read_bytes()
    target.unlink()
    actual = backup / "actual"
    actual.write_bytes(payload)
    target.symlink_to(actual)
    with pytest.raises(stageb.StageBError):
        stageb.build_site_attestation(
            contract=contract, backup_root=backup, declaration=_declaration()
        )


def test_adjudicate_accepts_independent_sites(tmp_path: Path):
    contract, site_a, site_b = _site_pair(tmp_path)
    result = stageb.adjudicate_sites(contract, site_a, site_b)
    assert result["independent_external_copy_count"] == 2
    assert result["objective_train_opened"] is False


@pytest.mark.parametrize(
    "field",
    ["site_id", "fault_domain_id_sha256", "physical_device_id_sha256"],
)
def test_adjudicate_rejects_same_physical_identity(tmp_path: Path, field):
    contract, site_a, site_b = _site_pair(tmp_path)
    site_b = copy.deepcopy(site_b)
    site_b["failure_domain"][field] = site_a["failure_domain"][field]
    _rehash(site_b, "attestation_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.adjudicate_sites(contract, site_a, site_b)


def test_adjudicate_rejects_same_automatic_fingerprint(tmp_path: Path):
    contract, site_a, site_b = _site_pair(tmp_path)
    site_a = copy.deepcopy(site_a)
    site_b = copy.deepcopy(site_b)
    site_a["storage_observation"]["automatic_device_fingerprint"] = "f" * 64
    site_b["storage_observation"]["automatic_device_fingerprint"] = "f" * 64
    _rehash(site_a, "attestation_sha256")
    _rehash(site_b, "attestation_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.adjudicate_sites(contract, site_a, site_b)


def test_adjudicate_rejects_same_remote_service_identity(tmp_path: Path):
    contract, site_a, site_b = _site_pair(tmp_path)
    for site in (site_a, site_b):
        site["failure_domain"]["fault_domain_kind"] = "remote_object_store"
        site["failure_domain"]["physical_device_id_sha256"] = None
        site["failure_domain"]["service_or_system_id_sha256"] = stageb.sha256_bytes(b"same_bucket")
        _rehash(site, "attestation_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.adjudicate_sites(contract, site_a, site_b)


def test_adjudicate_rejects_same_host_device_for_nonremote_kinds(tmp_path: Path):
    contract, site_a, site_b = _site_pair(tmp_path)
    site_b["failure_domain"]["fault_domain_kind"] = "independent_host"
    site_b["failure_domain"]["physical_device_id_sha256"] = None
    site_b["failure_domain"]["service_or_system_id_sha256"] = stageb.sha256_bytes(b"host_b")
    site_b["storage_observation"]["hostname_sha256"] = site_a["storage_observation"]["hostname_sha256"]
    site_b["storage_observation"]["filesystem_device_number"] = site_a["storage_observation"]["filesystem_device_number"]
    site_b["storage_observation"]["automatic_device_fingerprint"] = None
    site_a["storage_observation"]["automatic_device_fingerprint"] = None
    _rehash(site_a, "attestation_sha256")
    _rehash(site_b, "attestation_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.adjudicate_sites(contract, site_a, site_b)


def test_adjudicate_allows_distinct_remote_services(tmp_path: Path):
    contract, site_a, site_b = _site_pair(tmp_path)
    for index, site in enumerate((site_a, site_b), 1):
        site["failure_domain"]["fault_domain_kind"] = "remote_object_store"
        site["failure_domain"]["physical_device_id_sha256"] = None
        site["failure_domain"]["service_or_system_id_sha256"] = stageb.sha256_bytes(
            "bucket_{}".format(index).encode("utf-8")
        )
        _rehash(site, "attestation_sha256")
    result = stageb.adjudicate_sites(contract, site_a, site_b)
    assert result["failure_domain_kinds"] == ["remote_object_store", "remote_object_store"]


def test_build_summary_ready(tmp_path: Path):
    contract, site_a, site_b = _site_pair(tmp_path)
    summary = stageb.build_summary(
        repository={"root": "/data/state_diff2"},
        contract=contract,
        site_a=site_a,
        site_b=site_b,
    )
    assert summary["scientific_status"] == "READY"
    assert summary["objective_train_access_authorized_by_this_report"] is True
    assert summary["selection_holdout_access_authorized_by_this_report"] is False
    stageb.validate_summary(summary)


@pytest.mark.parametrize("field", stageb.FALSE_BOUNDARIES)
def test_summary_rejects_boundary(tmp_path: Path, field):
    contract, site_a, site_b = _site_pair(tmp_path)
    summary = stageb.build_summary(
        repository={}, contract=contract, site_a=site_a, site_b=site_b
    )
    summary[field] = True
    _rehash(summary, "summary_sha256")
    with pytest.raises(stageb.StageBError):
        stageb.validate_summary(summary)


def test_blocked_report_keeps_all_roles_closed():
    value = stageb.blocked_report(None, RuntimeError("x"))
    assert value["scientific_status"] == "BLOCKED"
    assert value["objective_train_access_authorized_by_this_report"] is False
    assert value["selection_holdout_access_authorized_by_this_report"] is False
    assert value["rerun_authorized"] is False


def test_implementation_paths_are_add_only():
    assert len(stageb.IMPLEMENTATION_PATHS) == 5
    assert all(status == "A" for status, _ in stageb.IMPLEMENTATION_PATHS)


def test_expected_file_population():
    assert stageb.EXPECTED_REQUIRED_FILE_COUNT == 7
    assert len(stageb.EXPECTED_ARCHIVE_ROLES) == 5
    assert set(stageb.EXPECTED_ARCHIVE_BASENAMES) == set(stageb.EXPECTED_ARCHIVE_ROLES)
