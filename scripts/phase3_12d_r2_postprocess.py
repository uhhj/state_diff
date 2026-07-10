#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def infer_scoped_root(payload: Dict[str, Any]) -> str:
    original = str(payload.get("root_cause", ""))
    verdict = str(payload.get("verdict", ""))
    progress = payload.get("progress", {})
    progress_status = str(progress.get("status", payload.get("status", "")))

    if verdict == "FAIL" or progress_status not in {"", "completed"}:
        return "phase312d_r2_code_snapshot_or_execution_integrity_failed"

    lowered = original.lower()
    flags = payload.get("diagnostic_flags", {})
    raw_supported = bool(
        payload.get("raw_ddpm_headroom_supported", False)
        or flags.get("raw_ddpm_headroom_supported", False)
        or flags.get("candidate_oracle_headroom_supported", False)
    )
    sparse_masked = bool(
        payload.get("sparse_metric_mask_supported", False)
        or flags.get("sparse_metric_mask_supported", False)
        or "sparse" in lowered and "mask" in lowered
    )

    if raw_supported or ("headroom" in lowered and "supported" in lowered):
        return "phase312d_r2_raw_candidate_headroom_supported_under_repaired_environment"
    if sparse_masked:
        return "phase312d_r2_sparse_progress_metric_masks_dense_action_effect"
    if "no_actionable_headroom" in lowered or "no_headroom" in lowered:
        return "phase312d_r2_no_raw_headroom_under_legacy_checkpoint_bridge"
    return "phase312d_r2_candidate_oracle_completed_under_legacy_checkpoint_bridge"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--environment-audit", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    report_path = Path(args.report_md)
    audit_path = Path(args.environment_audit)

    payload = load_json(summary_path)
    audit = load_json(audit_path)
    original_root = str(payload.get("root_cause", ""))
    scoped_root = infer_scoped_root(payload)

    payload["original_analyzer_root_cause"] = original_root
    payload["root_cause"] = scoped_root
    payload["environment_semantics"] = "phase3_12d_r2_deferred_zero_offset_latent_arming"
    payload["checkpoint_semantics"] = "legacy_pre_settle_hidden_contact_phase3_windows"
    payload["legacy_checkpoint_bridge"] = True
    payload["environment_audit_verdict"] = audit.get("verdict")
    payload["environment_audit_root_cause"] = audit.get("root_cause")
    payload["scientific_scope_limit"] = (
        "The existing DDPM/IDM checkpoints were trained before the paired-visible "
        "arming repair. Positive oracle headroom is informative, but a negative "
        "result does not by itself prove that the repaired task has no actionable "
        "candidate support; it may reflect checkpoint/environment semantic shift."
    )
    payload["phase4_or_cps_allowed"] = False
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))

    appendix = f"""

## Phase3.12d-r2 checkpoint/environment bridge limitation

- Matrix label: `{args.label}`
- Environment semantics: `deferred zero-offset latent arming after common visible settle`
- Checkpoint/data semantics: `legacy pre-settle hidden-contact Phase3 windows`
- Original analyzer root cause: `{original_root}`
- Scoped r2 root cause: `{scoped_root}`
- Environment audit: `{audit.get('verdict')}` / `{audit.get('root_cause')}`
- Existing checkpoints are used only as a diagnostic bridge.
- A positive realized-effect oracle result is informative.
- A negative oracle result cannot be promoted to an architecture-level conclusion without regenerating data and retraining under the repaired task semantics.
- Phase4/CPS remains blocked.
"""
    text = report_path.read_text() if report_path.exists() else ""
    if "## Phase3.12d-r2 checkpoint/environment bridge limitation" not in text:
        report_path.write_text(text.rstrip() + appendix + "\n")

    print(json.dumps({
        "summary": str(summary_path),
        "report": str(report_path),
        "original_root_cause": original_root,
        "scoped_root_cause": scoped_root,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
