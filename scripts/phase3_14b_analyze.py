#!/usr/bin/env python3
"""Aggregate Phase3.14b while keeping IDM and Phase4 blocked."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccda_phase3.phase314a_contract import strict_json_dump, strict_json_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_preflight_summary.json"
    )
    training = strict_json_load(
        root / "reports/phase3_14b_training_summary.json"
    )
    candidate = strict_json_load(
        root / "reports/phase3_14b_candidate_summary.json"
    )

    if preflight.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314b_preflight_failed"
    elif training.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314b_ddpm_training_failed"
    else:
        verdict = str(candidate["verdict"])
        root_cause = str(candidate["root_cause"])

    next_step = (
        "Phase3.14c GT-future and predicted-future inverse dynamics"
        if verdict == "PASS"
        else "Diagnose Phase3.14b DDPM/candidate-support failure"
    )
    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "preflight": preflight["root_cause"],
        "training": training["root_cause"],
        "candidate_support": candidate["root_cause"],
        "selected_model": training["selected"],
        "gate_values": candidate.get("gate_values"),
        "next_step": next_step,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)
    lines = [
        "# Phase3.14b Final Report",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Next step: `{next_step}`",
        "",
        "No IDM training, query-local execution, Phase4, or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (root / "reports/phase3_14b_no_phase4_confirmation.md").write_text(
        "# Phase3.14b No Phase4 Confirmation\n\n"
        "- IDM training: `False`\n"
        "- Query-local candidate execution: `False`\n"
        "- Phase4: `False`\n"
        "- CPS: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
