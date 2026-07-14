#!/usr/bin/env python3
"""Finalize the r2.5.4 shared-prior determinism audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r23_diagnostics import require_repository_state, write_json_once
from ccda_phase3.phase314b_r254_resume1_prior_determinism import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    SCHEMA,
    assert_only_allowed_worktree_paths,
    classify_prior_audit,
    source_sha256,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def compact_run(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": value["run_id"],
        "prior_state_sha256": value["prior_state_sha256"],
        "prior_prediction_sha256": value["prior_prediction_sha256"],
        "prior_z_mse": float(value["prior_z_mse"]),
        "prior_history": value["prior_history"],
        "state_summary": value["state_summary"],
        "environment_before_fit": value["environment_before_fit"],
        "environment_after_fit": value["environment_after_fit"],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    repeat = summary["same_device_repeats"]
    historical = summary["historical_functional_fingerprint"]
    lines = [
        "# Phase3.14b-r2.5.4 Resume1 Shared-Prior Determinism Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{summary['verdict']}` (train-only diagnostic chain completed)",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Functional prior contract supported: `{str(summary['functional_prior_contract_supported']).lower()}`",
        f"- Next: `{summary['next_stage']}`",
        "- Train-only recommendation: `None`",
        "- Selected configuration: `None`",
        "",
        "## Historical versus current prior",
        "",
        f"- Historical GPU: `{summary['historical_prior']['gpu_name']}`",
        f"- Current GPU: `{summary['environment']['gpu_name']}`",
        f"- Expected historical state SHA256: `{historical['expected_prior_state_sha256']}`",
        f"- Observed state SHA256 values: `{repeat['observed_state_sha256']}`",
        f"- Exact historical SHA matches: `{historical['exact_state_sha_match_count']}`",
        f"- Same-device state hashes all exact: `{str(repeat['all_state_sha_exact']).lower()}`",
        f"- Same-device prediction hashes all exact: `{str(repeat['all_prediction_sha_exact']).lower()}`",
        f"- Same-device functional equivalence: `{str(repeat['functional_equivalence_pass']).lower()}`",
        f"- Historical functional fingerprint: `{str(historical['functional_fingerprint_pass']).lower()}`",
        "",
        "## Repeated fits",
        "",
        "| Run | State SHA256 | Prediction SHA256 | Prior z MSE |",
        "|---|---|---|---:|",
    ]
    for value in summary["runs"]:
        lines.append(
            f"| `{value['run_id']}` | `{value['prior_state_sha256']}` | "
            f"`{value['prior_prediction_sha256']}` | {value['prior_z_mse']:.9g} |"
        )
    lines.extend([
        "",
        "## Same-device pairwise differences",
        "",
        "| Pair | State exact | Param max/RMSE | Prediction max/RMSE | z-MSE relative | PASS |",
        "|---|---:|---|---|---:|---:|",
    ])
    for name, value in repeat["pairwise"].items():
        state = value["state"]
        prediction = value["prediction"]
        prior_z = value["prior_z_mse"]
        lines.append(
            f"| `{name}` | {str(state['exact_equal']).lower()} | "
            f"{state['max_abs']:.4g}/{state['rmse']:.4g} | "
            f"{prediction['max_abs']:.4g}/{prediction['rmse']:.4g} | "
            f"{prior_z['relative_error']:.4g} | {str(value['pass']).lower()} |"
        )
    lines.extend([
        "",
        "## Historical functional fingerprint",
        "",
        f"- Historical prior z MSE: `{historical['prior_z_mse_expected']:.9g}`",
        f"- Current median prior z MSE: `{historical['prior_z_mse_observed_median']:.9g}`",
        f"- Prior z-MSE relative difference: `{historical['prior_z_mse_relative_error']:.6g}`",
        f"- Loss-curve relative error p95: `{historical['loss_curve_relative_error_p95']:.6g}`",
        f"- Final-loss relative error: `{historical['final_loss_relative_error']:.6g}`",
        "",
        "## Contract interpretation",
        "",
        "- Same-run loaded snapshot integrity remains exact-SHA protected.",
        "- Same-device reruns are judged by exact SHA plus strict parameter/prediction bounds.",
        "- Historical cross-device reproduction is judged by the committed prior functional fingerprint, not parameter bytes alone.",
        "- Robot-proxy attribution was not run or interpreted in this stage.",
        "",
        "## Boundaries",
        "",
        "- Reverse sampling rerun: `False`",
        "- Validation targets used: `False`",
        "- Formal test read: `False`",
        "- Formal training: `False`",
        "- Checkpoint/model weights saved: `False`",
        "- Formal IDM / candidate execution: `False`",
        "- Phase4 / CPS: `False`",
        "",
        "A PASS verdict means only that the shared-prior diagnostic chain completed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r254_resume1_preflight_summary.json",
    )
    parser.add_argument(
        "--evidence-report",
        default="reports/phase3_14b_r254_resume1_prior_repeat_evidence.json",
    )
    parser.add_argument(
        "--audit-report",
        default="reports/phase3_14b_r254_resume1_prior_audit_summary.json",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/phase3_14b_r254_resume1_summary.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/phase3_14b_r254_resume1_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight_path = resolve(root, args.preflight_report)
    evidence_path = resolve(root, args.evidence_report)
    audit_path = resolve(root, args.audit_report)
    summary_path = resolve(root, args.summary_output)
    markdown_path = resolve(root, args.markdown_output)
    if summary_path.exists() or markdown_path.exists():
        raise RuntimeError("refusing to overwrite Resume1 final artifacts")

    allowed = (
        preflight_path.relative_to(root).as_posix(),
        evidence_path.relative_to(root).as_posix(),
        audit_path.relative_to(root).as_posix(),
    )
    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, allowed)
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    preflight = load_json(preflight_path)
    evidence = load_json(evidence_path)
    audit = load_json(audit_path)
    if evidence.get("schema") != "phase314b_r254_resume1_prior_repeat_evidence_v1":
        raise RuntimeError("repeat evidence schema mismatch")
    if evidence.get("verdict") != "PASS":
        raise RuntimeError("repeat evidence did not complete")
    if preflight.get("verdict") != "PASS" or audit.get("verdict") != "PASS":
        raise RuntimeError("preflight or prior audit did not complete")
    if audit.get("schema") != SCHEMA:
        raise RuntimeError("prior audit schema mismatch")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("source changed after preflight")
    if evidence.get("source_sha256") != source_sha256(root):
        raise RuntimeError("repeat evidence source hash mismatch")
    if audit.get("source_sha256") != source_sha256(root):
        raise RuntimeError("audit source hash mismatch")
    if audit.get("repeat_evidence_report") != evidence_path.relative_to(root).as_posix():
        raise RuntimeError("audit repeat-evidence reference mismatch")
    if audit.get("runs") != evidence.get("runs"):
        raise RuntimeError("audit runs differ from write-once repeat evidence")
    if bool(audit.get("robot_proxy_attribution_run")):
        raise RuntimeError("Resume1 unexpectedly ran robot-proxy attribution")
    classification = classify_prior_audit(audit)
    for key in (
        "root_cause",
        "next_stage",
        "functional_prior_contract_supported",
        "train_only_recommendation",
        "selected_configuration",
    ):
        if audit.get(key) != classification.get(key):
            raise RuntimeError(f"prior classifier mismatch: {key}")
    for key in (
        "validation_targets_used",
        "formal_test_read",
        "formal_training",
        "checkpoint_saved",
        "idm",
        "candidate_execution",
        "phase4",
        "cps",
    ):
        if bool(audit.get(key)):
            raise RuntimeError(f"forbidden boundary crossed: {key}")

    summary = {
        "phase": PHASE,
        "verdict": "PASS",
        "root_cause": classification["root_cause"],
        "next_stage": classification["next_stage"],
        "functional_prior_contract_supported": classification[
            "functional_prior_contract_supported"
        ],
        "train_only_recommendation": None,
        "selected_configuration": None,
        "preflight_report": preflight_path.relative_to(root).as_posix(),
        "repeat_evidence_report": evidence_path.relative_to(root).as_posix(),
        "audit_report": audit_path.relative_to(root).as_posix(),
        "source_sha256": source_sha256(root),
        "environment": audit["environment"],
        "historical_prior": audit["historical_prior"],
        "runs": [compact_run(value) for value in audit["runs"]],
        "same_device_repeats": audit["same_device_repeats"],
        "historical_functional_fingerprint": audit[
            "historical_functional_fingerprint"
        ],
        "robot_proxy_attribution_run": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(summary_path, summary)
    write_text_once(markdown_path, render_report(summary))
    print(json.dumps({
        "verdict": "PASS",
        "root_cause": summary["root_cause"],
        "functional_prior_contract_supported": summary[
            "functional_prior_contract_supported"
        ],
        "summary": str(summary_path),
        "report": str(markdown_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
