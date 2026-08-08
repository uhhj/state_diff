"""Analyze Gate-4 definition and frozen control relevance for OHJ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_ohj_cable.analyze_control_relevance import (  # noqa: E402
    classify_control)
from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    load_config, report_dir, write_json)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True).strip()


def gate4_decomposition(pair, source_config):
    analysis = source_config["analysis"]
    future_peak = float(pair["future_peak_visible_rmse_m"])
    repeat_peak = float(pair["repeat_peak_visible_rmse_m"])
    absolute_floor = float(analysis["future_visible_rmse_min_m"])
    repeat_multiplier = float(analysis["future_vs_repeat_multiplier"])
    repeat_relative_floor = float(repeat_multiplier * repeat_peak)
    formal_threshold = max(absolute_floor, repeat_relative_floor)
    return {
        "future_peak_visible_rmse_m": future_peak,
        "repeat_peak_visible_rmse_m": repeat_peak,
        "absolute_floor_m": absolute_floor,
        "repeat_multiplier": repeat_multiplier,
        "repeat_relative_floor_m": repeat_relative_floor,
        "formal_threshold_m": formal_threshold,
        "absolute_pass": bool(future_peak >= absolute_floor),
        "repeat_relative_pass": bool(
            future_peak >= repeat_relative_floor),
        "formal_pass": bool(future_peak >= formal_threshold),
        "future_over_repeat": (
            None if repeat_peak <= 1e-12
            else float(future_peak / repeat_peak)),
        "absolute_margin_m": float(future_peak - absolute_floor),
        "repeat_relative_margin_m": float(
            future_peak - repeat_relative_floor),
    }


def choose_route(gate4, control):
    if not gate4["repeat_relative_pass"]:
        return "FUTURE_BRANCH_NOT_REPEAT_SEPARATED"
    if control["control_relevant"]:
        if gate4["formal_pass"]:
            return "PHASE0_INFORMATION_STRUCTURE_COMPLETE"
        if not gate4["absolute_pass"]:
            return "GATE4_ABSOLUTE_FLOOR_SCIENTIFIC_REVIEW_REQUIRED"
    return "OHJ_CONTROL_STRUCTURE_INSUFFICIENT"


def analyze(audit_config_path):
    audit = json.loads(Path(audit_config_path).read_text(encoding="utf-8"))
    source_path = REPO_ROOT / audit["source_config"]
    source_config = load_config(source_path)
    pair = json.loads(
        (report_dir(source_config) / "pair_metrics.json").read_text(
            encoding="utf-8"))
    raw_root = Path(audit["raw_output_root"])
    matrix_audit = json.loads(
        (raw_root / "CONTROL_MATRIX_AUDIT.json").read_text(
            encoding="utf-8"))
    control = classify_control(matrix_audit["matrix"], source_config)
    gate4 = gate4_decomposition(pair, source_config)
    route = choose_route(gate4, control)
    scientific = {
        "diagnostic_only": True,
        "formal_phase0_verdict_unchanged": True,
        "formal_gate5_executed": False,
        "source_config": audit["source_config"],
        "source_pair_verdict": pair["verdict"],
        "gate4_definition_audit": gate4,
        "control_relevance_audit": {
            **control,
            "formal_gate5": False,
            "interpretation": (
                "existing Gate-5 classifier applied diagnostically to "
                "the frozen action matrix"),
        },
        "route": route,
        "training_started": False,
    }
    write_json(raw_root / "SCIENTIFIC_AUDIT.json", scientific)
    committed = REPO_ROOT / audit["committed_report_dir"]
    evidence = {
        "phase_name": audit["phase_name"],
        "route": route,
        "repository": {
            "starting_main_sha": audit["provenance"]["starting_main_sha"],
            "ending_main_sha": _git("rev-parse", "HEAD"),
            "ending_submodule_sha": _git(
                "rev-parse", "HEAD:external/deformable-ravens"),
        },
        "scientific_audit": scientific,
        "training": {
            "B0": False, "B1": False, "CFPM": False, "IDM": False},
    }
    write_json(committed / "EVIDENCE.json", evidence)
    control_matrix = control["matrix"]
    result_text = f"""Verdict: PHASE0E_SCIENTIFIC_AUDIT_COMPLETE

Route: {route}

Source benchmark:
- Config: {audit["source_config"]}
- Source pair verdict: {pair["verdict"]}
- Formal Phase-0 verdict changed: No
- Formal Gate 5 executed: No

Gate-4 definition audit:
- Future peak: {gate4["future_peak_visible_rmse_m"]} m
- Repeat peak: {gate4["repeat_peak_visible_rmse_m"]} m
- Future/repeat: {gate4["future_over_repeat"]}
- Absolute floor: {gate4["absolute_floor_m"]} m
- Repeat multiplier: {gate4["repeat_multiplier"]}
- Repeat-relative floor: {gate4["repeat_relative_floor_m"]} m
- Formal threshold: {gate4["formal_threshold_m"]} m
- Absolute pass: {gate4["absolute_pass"]}
- Repeat-relative pass: {gate4["repeat_relative_pass"]}
- Formal pass: {gate4["formal_pass"]}
- Absolute margin: {gate4["absolute_margin_m"]} m
- Repeat-relative margin: {gate4["repeat_relative_margin_m"]} m

Frozen control-relevance audit:
- Diagnostic only: True
- Existing classifier reused: True
- Control relevant under frozen rule: {control["control_relevant"]}
- Same candidate commands across branches: {control["same_candidate_commands_across_branches"]}
- Best candidate: {control["best_candidate"]}
- JAM left progress advantage: {control["jam_left_progress_advantage_m"]} m
- FREE matrix: {control_matrix["free"]}
- JAM-R matrix: {control_matrix["jam_right"]}

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Interpretation:
- This audit does not lower or rewrite Gate 4.
- This audit does not count as formal Gate 5.
- If control relevance is present while only the absolute Gate-4 floor fails,
  the next step is an independent scientific justification/preregistration of
  Gate 4 before any training.
- If control relevance is absent, stop local OHJ tuning and review the
  benchmark structure instead of training around the problem.
"""
    committed.mkdir(parents=True, exist_ok=True)
    (committed / "RESULT.md").write_text(result_text, encoding="utf-8")
    return scientific


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = analyze(args.config)
    print("route={}".format(result["route"]))


if __name__ == "__main__":
    main()
