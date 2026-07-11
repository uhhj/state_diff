#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from phase3_12d_r22_integrity import strict_json_dump


def load(root: Path, name: str):
    path = root / "reports" / name
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    pre = load(root, "phase3_12d_r24_preflight_summary.json")
    env = load(root, "phase3_12d_r24_environment_audit_summary.json")
    obs = load(root, "phase3_12d_r24_observation_summary.json")
    snap = load(root, "phase3_12d_r24_snapshot_audit_summary.json")
    if pre.get("verdict") != "PASS": root_cause = "phase312d_r24_code_or_hook_integrity_failed"
    elif not env.get("checks", {}).get("direction_physics", False): root_cause = "phase312d_r24_numerical_stability_failed"
    elif not env.get("checks", {}).get("no_action_parity", False): root_cause = "phase312d_r24_no_action_parity_failed"
    elif obs.get("verdict") != "PASS": root_cause = "phase312d_r24_observation_leakage_detected"
    elif not env.get("checks", {}).get("sub_deadband_8_of_8", False) or not env.get("checks", {}).get("engagement_at_least_6_of_8", False): root_cause = "phase312d_r24_deadband_actionability_failed"
    elif not env.get("checks", {}).get("breakaway_8_of_8", False): root_cause = "phase312d_r24_breakaway_release_failed"
    elif snap.get("verdict") != "PASS": root_cause = "phase312d_r24_snapshot_restore_failed"
    else: root_cause = "phase312d_r24_slack_breakaway_v2_environment_supported"
    verdict = "PASS" if root_cause.endswith("environment_supported") else "FAIL"
    payload = {
        "verdict": verdict, "root_cause": root_cause,
        "preflight_verdict": pre.get("verdict"), "environment_verdict": env.get("verdict"),
        "observation_verdict": obs.get("verdict"), "snapshot_verdict": snap.get("verdict"),
        "state_v2_activated": False, "model_training_run": False, "candidate_matrix_run": False,
        "phase4_run": False, "cps_run": False,
        "next_step": "Phase3.13 state-v2 dataset regeneration" if verdict == "PASS" else "Remain blocked; repair the reported r2.4 environment gate and rerun r2.4.",
        "scientific_boundary": "PASS supports only no-action/deadband hiddenness, causal action divergence, breakaway release, and reproducible snapshot semantics. It does not establish StateDiff, candidate headroom, or CPS effectiveness.",
    }
    strict_json_dump(root / "reports/phase3_12d_r24_summary.json", payload)
    lines = ["# Phase3.12d-r2.4 Final Report", "", f"- Verdict: `{verdict}`", f"- Root cause: `{root_cause}`", f"- Next step: `{payload['next_step']}`", "", "## Audit Chain", "", "| Audit | Verdict |", "|---|---:|", f"| Preflight | `{pre.get('verdict')}` |", f"| Environment | `{env.get('verdict')}` |", f"| Observation | `{obs.get('verdict')}` |", f"| Snapshot | `{snap.get('verdict')}` |", "", "## Scientific Boundary", "", payload["scientific_boundary"], "", "State-v2 was not activated. No model training, candidate matrix, Phase4, or CPS was run."]
    (root / "reports/phase3_12d_r24_report.md").write_text("\n".join(lines) + "\n")
    confirmation = f"""# Phase3.12d-r2.4 No Phase4 Confirmation

- Timestamp: `{datetime.now(timezone.utc).isoformat()}`
- Phase4 run: `False`
- CPS run: `False`
- Candidate matrix run: `False`
- Model training run: `False`
- State-v2 activated: `False`
- The only permitted next step after PASS is Phase3.13 state-v2 dataset regeneration and baseline retraining.
"""
    (root / "reports/phase3_12d_r24_no_phase4_confirmation.md").write_text(confirmation)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
