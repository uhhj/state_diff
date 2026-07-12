#!/usr/bin/env python3
"""Aggregate Phase3.14b-r2 without authorizing IDM or Phase4 prematurely."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccda_phase3.phase314a_contract import (
    strict_json_dump,
    strict_json_load,
)


def optional_json(path: Path):
    return strict_json_load(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r2_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r2_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight = strict_json_load(
        root / "reports/phase3_14b_r2_preflight_summary.json"
    )
    smoke = strict_json_load(
        root / "reports/phase3_14b_r2_smoke_summary.json"
    )
    training = strict_json_load(
        root / "reports/phase3_14b_r2_training_summary.json"
    )
    selection = strict_json_load(
        root / "reports/phase3_14b_r2_selection_summary.json"
    )
    test = optional_json(
        root / "reports/phase3_14b_r2_test_summary.json"
    )

    if preflight.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314b_r2_preflight_failed"
    elif smoke.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314b_r2_runtime_smoke_failed"
    elif training.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314b_r2_training_failed"
    elif selection.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = str(selection["root_cause"])
    elif test is None:
        verdict = "FAIL"
        root_cause = "phase314b_r2_formal_test_missing"
    else:
        verdict = str(test["verdict"])
        root_cause = str(test["root_cause"])

    next_step = (
        "Phase3.14c repaired inverse dynamics"
        if verdict == "PASS"
        and root_cause == "phase314b_r2_candidate_support_repaired"
        else "Diagnose the Phase3.14b-r2 objective/schedule repair failure"
    )
    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "preflight": preflight["root_cause"],
        "smoke": smoke["root_cause"],
        "training": training["root_cause"],
        "selection": selection["root_cause"],
        "test": test["root_cause"] if test else None,
        "selected_model": (
            selection.get("selected_model")
            if selection.get("verdict") == "PASS"
            else None
        ),
        "next_step": next_step,
        "idm_training": False,
        "candidate_action_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r2 Final Report",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Next step: `{next_step}`",
        "",
        "- IDM training: `False`",
        "- Candidate action execution: `False`",
        "- Phase4: `False`",
        "- CPS: `False`",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (
        root / "reports/phase3_14b_r2_no_phase4_confirmation.md"
    ).write_text(
        "# Phase3.14b-r2 No Phase4 Confirmation\n\n"
        "- IDM training: `False`\n"
        "- Candidate action execution: `False`\n"
        "- Phase4: `False`\n"
        "- CPS: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
