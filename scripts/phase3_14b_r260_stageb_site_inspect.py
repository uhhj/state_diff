#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_FROM_FILE = Path(__file__).resolve().parents[1]
if str(ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(ROOT_FROM_FILE))

from ccda_phase3 import phase314b_r260_stageb_external_backup_attestation as stageb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract")
    parser.add_argument("--backup-root")
    parser.add_argument("--output")
    parser.add_argument("--site-id")
    parser.add_argument("--fault-domain-kind", choices=stageb.FAULT_DOMAIN_KINDS)
    parser.add_argument("--fault-domain-id")
    parser.add_argument("--physical-device-id")
    parser.add_argument("--service-or-system-id")
    parser.add_argument("--location-description")
    parser.add_argument("--operator-certify-independent", action="store_true")
    parser.add_argument("--read-back-completed", action="store_true")
    parser.add_argument("--copy-completed", action="store_true")
    parser.add_argument("--understand-same-disk-partitions", action="store_true")
    parser.add_argument("--import-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.import_smoke:
        print(stageb.SITE_SCHEMA)
        return 0
    required = {
        "--contract": args.contract,
        "--backup-root": args.backup_root,
        "--output": args.output,
        "--site-id": args.site_id,
        "--fault-domain-kind": args.fault_domain_kind,
        "--fault-domain-id": args.fault_domain_id,
        "--location-description": args.location_description,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("missing required arguments: {}".format(", ".join(missing)))
    contract = stageb.load_json(Path(args.contract).resolve())
    declaration = {
        "site_id": args.site_id,
        "fault_domain_kind": args.fault_domain_kind,
        "fault_domain_id": args.fault_domain_id,
        "physical_device_id": args.physical_device_id,
        "service_or_system_id": args.service_or_system_id,
        "location_description": args.location_description,
        "copy_completed": bool(args.copy_completed),
        "read_back_completed": bool(args.read_back_completed),
        "operator_certifies_independent_failure_domain": bool(
            args.operator_certify_independent
        ),
        "operator_understands_same_disk_partitions_are_not_independent": bool(
            args.understand_same_disk_partitions
        ),
    }
    payload = stageb.build_site_attestation(
        contract=contract,
        backup_root=Path(args.backup_root).resolve(),
        declaration=declaration,
    )
    output = Path(args.output).resolve()
    stageb.atomic_write_once(output, stageb.stable_json_bytes(payload))
    print(
        json.dumps(
            {
                "attestation": str(output),
                "attestation_sha256": payload["attestation_sha256"],
                "site_id": declaration["site_id"],
                "fault_domain_id": declaration["fault_domain_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
