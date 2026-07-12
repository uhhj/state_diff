#!/usr/bin/env python3
"""Phase3.14 data-provenance gate for the regenerated formal dataset."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ccda_phase3.provenance_v2 import (
    strict_json_dump,
    strict_json_load,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--formal-root",
        default="data/phase3_state_v2_slack",
    )
    parser.add_argument(
        "--audit-summary",
        default=(
            "reports/"
            "phase3_13_r1_dataset_audit_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14_provenance_gate_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14_provenance_gate_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    intermediate = (
        root / "reports/phase3_14_provenance_verify.json"
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/phase3_13_r1_verify.py",
            "--root",
            str(root),
            "--formal-root",
            args.formal_root,
            "--audit-summary",
            args.audit_summary,
            "--output",
            str(intermediate.relative_to(root)),
            "--report",
            "reports/phase3_14_provenance_verify.md",
        ],
        cwd=str(root),
        check=True,
    )
    verified = strict_json_load(intermediate)
    if verified.get("verdict") != "PASS":
        raise RuntimeError("Phase3.13-r1 verification did not pass")

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314_data_provenance_supported",
        "phase313_r1_gate": verified,
        "formal_training_allowed": True,
        "deterministic_training_started": False,
        "ddpm_training_started": False,
        "idm_training_started": False,
        "candidate_execution_started": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.output, payload)
    lines = [
        "# Phase3.14 Provenance Gate",
        "",
        "- Verdict: `PASS`",
        "- Root cause: `phase314_data_provenance_supported`",
        "- All locked source hashes match: `True`",
        "- Windows hash matches manifest: `True`",
        "- Raw Merkle root matches attestation: `True`",
        "- Formal training allowed: `True`",
        "",
        "No deterministic/DDPM/IDM training or candidate execution "
        "was run by this gate. Phase4 and CPS remain blocked.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
