#!/usr/bin/env python3
"""Aggregate Phase3.14a without authorizing DDPM or Phase4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccda_phase3.phase314a_contract import (
    strict_json_dump,
    strict_json_load,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--summary",
        default="reports/phase3_14a_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14a_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cache = strict_json_load(
        root / "reports/phase3_14a_cache_summary.json"
    )
    preflight = strict_json_load(
        root / "reports/phase3_14a_preflight_summary.json"
    )
    deterministic = strict_json_load(
        root / "reports/phase3_14a_deterministic_summary.json"
    )

    if cache.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314a_immutable_training_cache_failed"
    elif preflight.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314a_training_cache_preflight_failed"
    elif deterministic.get("verdict") != "PASS":
        verdict = "FAIL"
        root_cause = "phase314a_future_model_not_learnable"
    else:
        verdict = "PASS"
        root_cause = (
            "phase314a_state_v2_future_learnability_supported"
        )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "cache_gate": cache["root_cause"],
        "preflight_gate": preflight["root_cause"],
        "deterministic_gate": deterministic["root_cause"],
        "selected_model": deterministic.get("selection"),
        "next_step": (
            "Phase3.14b MLP-DDPM and Temporal-U-Net-DDPM"
            if verdict == "PASS"
            else "Diagnose deterministic future learnability"
        ),
        "ddpm_training": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14a Final Report",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Cache gate: `{cache['root_cause']}`",
        f"- Preflight gate: `{preflight['root_cause']}`",
        (
            "- Deterministic gate: "
            f"`{deterministic['root_cause']}`"
        ),
        f"- Next step: `{payload['next_step']}`",
        "",
        "No DDPM, IDM, candidate execution, Phase4, or CPS was run.",
    ]
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (root / "reports/phase3_14a_no_phase4_confirmation.md").write_text(
        "# Phase3.14a No Phase4 Confirmation\n\n"
        "- DDPM training: `False`\n"
        "- IDM training: `False`\n"
        "- Candidate support/execution: `False`\n"
        "- Phase4: `False`\n"
        "- CPS: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
