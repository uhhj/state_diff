#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path("/data/state_diff2")
    reports = root / "reports"
    audit = json.loads((reports / "phase3_13_dataset_audit_summary.json").read_text())
    purge = json.loads((reports / "phase3_13_legacy_purge_manifest.json").read_text())
    fresh = json.loads((reports / "phase3_13_fresh_worktree_summary.json").read_text())
    purge_pass = purge.get("verdict") == "APPLIED" and not purge.get("unknown", {}).get("main") and not purge.get("unknown", {}).get("submodule")
    passed = audit.get("verdict") == "PASS" and purge_pass and fresh.get("verdict") == "PASS"
    root_cause = "phase313_state_v2_dataset_supported" if passed else (audit.get("root_cause") if audit.get("verdict") != "PASS" else "phase313_legacy_asset_purge_incomplete")
    payload = {"verdict": "PASS" if passed else "FAIL", "root_cause": root_cause, "dataset_audit": audit.get("verdict"), "legacy_purge": purge.get("verdict"), "fresh_worktree": fresh.get("verdict"), "future_model_training": False, "idm_training": False, "candidate_matrix": False, "phase4": False, "cps": False, "next_step": "Phase3.14 State-v2 StateDiff / IDM Retraining + Candidate-Set Oracle Headroom" if passed else "Remain in Phase3.13 and repair the failing gate."}
    (reports / "phase3_13_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (reports / "phase3_13_report.md").write_text(f"# Phase3.13 Final Report\n\n- Verdict: `{payload['verdict']}`\n- Root cause: `{root_cause}`\n- Next step: `{payload['next_step']}`\n\nNo future-model training, IDM training, candidate matrix, Phase4, or CPS was run.\n")
    (reports / "phase3_13_no_phase4_confirmation.md").write_text(f"# Phase3.13 No Phase4 Confirmation\n\n- Timestamp: `{datetime.now(timezone.utc).isoformat()}`\n- Future-model training: `False`\n- IDM training: `False`\n- Candidate matrix: `False`\n- Phase4: `False`\n- CPS: `False`\n")
    counts = purge.get("counts", {})
    (root / "LEGACY_ASSET_REMOVAL.md").write_text(f"# Legacy Asset Removal\n\n- Purge date: `{datetime.now(timezone.utc).date().isoformat()}`\n- Pre-purge main commit: `{purge.get('main_head_before')}`\n- Pre-purge submodule commit: `{purge.get('submodule_head_before')}`\n- Removed assets: `{sum(int(counts.get(key, 0)) for key in ('main_tracked','main_untracked','submodule_tracked','submodule_untracked'))}`\n- Recorded bytes removed: `{counts.get('bytes_total', 0)}`\n- Manifest: `reports/phase3_13_legacy_purge_manifest.json`\n- Git history rewritten: no\n- Tracked assets are recoverable from the pre-purge commits above.\n- Untracked assets were permanently deleted.\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if not passed: raise SystemExit("Phase3.13 final gate failed")


if __name__ == "__main__": main()
