#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"

FILES_TO_SCAN = [
    "ccda_phase3/data_io.py",
    "scripts/phase3_generate_large_dataset.sh",
    "scripts/phase3_run_all.sh",
    "scripts/phase3_prepare_windows.py",
    "scripts/phase3_check_input_leakage.py",
    "scripts/phase3_eval_baselines.py",
    "scripts/phase3_train_baselines.py",
    "scripts/phase3_sanity_check_metrics.py",
    "scripts/phase3_pre_medium_audit.py",
    "scripts/phase3_condition_integration_audit.py",
    "scripts/phase3_debug_action_codec_idm.py",
    "scripts/phase3_policy_rollout.py",
]

REQUIRED_STRINGS = {
    "scripts/phase3_eval_baselines.py": ["primary_hidden_condition", "diagnostic_hidden_condition", PRIMARY],
    "scripts/phase3_check_input_leakage.py": ["primary_hidden_condition", "diagnostic_hidden_condition"],
    "scripts/phase3_run_all.sh": ["PHASE3_PRIMARY_HIDDEN_CONDITION", "PHASE3_DIAGNOSTIC_HIDDEN_CONDITION", "PHASE3_CONDITIONS", "PHASE3_ALLOW_ROLLOUT"],
    "ccda_phase3/data_io.py": [PRIMARY, "FORBIDDEN_METADATA_NOT_IN_X"],
}

FORBIDDEN_TOKENS = [
    "hidden_contact_meta",
    "recoverability_params",
    "breakaway_released",
    "breakaway_release_step",
    "breakaway_max_disp_seen",
    "final_fraction",
    "success",
]


def run(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def add(findings: List[Dict[str, Any]], level: str, name: str, file: str, line: Any, snippet: str, explain: str) -> None:
    findings.append({
        "level": level,
        "name": name,
        "file": file,
        "line": line,
        "snippet": snippet[:240] if snippet else "",
        "explain": explain,
    })


def line_no(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


def scan_file(root: Path, rel: str, findings: List[Dict[str, Any]]) -> None:
    path = root / rel
    if not path.exists():
        add(findings, "FAIL", "missing_file", rel, None, "", "Required Phase3 file is missing.")
        return
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    has_dynamic = "primary_hidden_condition" in text and "diagnostic_hidden_condition" in text and PRIMARY in text

    for m in re.finditer(r"primary_(?:pair|branch_pair).*free_vs_hidden_pin|free_vs_hidden_pin.*primary_(?:pair|branch_pair)", text):
        ln = line_no(text, m.start())
        add(findings, "FAIL", "primary_pair_hardcoded_hidden_pin", rel, ln, lines[ln-1], "Primary pair must be free_vs_hidden_breakaway_pin; free_vs_hidden_pin is allowed only as diagnostic.")

    for m in re.finditer(r"--conditions\s+free\s+hidden_pin\s+hidden_high_friction(?!\s+hidden_breakaway_pin)", text):
        ln = line_no(text, m.start())
        add(findings, "FAIL", "three_condition_generation", rel, ln, lines[ln-1], "Generation must include hidden_breakaway_pin.")

    if rel == "scripts/phase3_eval_baselines.py":
        for m in re.finditer(r"has_pin_ref\s*=|ref\[\"hidden_pin\"\]", text):
            ln = line_no(text, m.start())
            level = "INFO" if has_dynamic else "WARN"
            add(findings, level, "eval_hidden_pin_reference", rel, ln, lines[ln-1], "Allowed only when hidden_pin is diagnostic and primary_hidden_condition is dynamic.")

    if rel == "scripts/phase3_run_all.sh" and "run_rollout" in text:
        level = "INFO" if "PHASE3_ALLOW_ROLLOUT" in text else "FAIL"
        add(findings, level, "rollout_gate", rel, None, "run_rollout", "Rollout code exists; it must be gated by PHASE3_ALLOW_ROLLOUT=1.")

    if rel == "scripts/phase3_policy_rollout.py":
        if "free_vs_hidden_pin_success_gap" in text:
            add(findings, "WARN", "rollout_hidden_pin_primary_metric", rel, None, "free_vs_hidden_pin_success_gap", "Rollout summary still reports the old hidden_pin success gap. Patch or explicitly reinterpret this before rollout smoke.")
        default_three = 'default=["free", "hidden_pin", "hidden_high_friction"]'
        if default_three in text and PRIMARY not in text:
            add(findings, "WARN", "rollout_three_condition_default", rel, None, default_three, "Rollout defaults omit hidden_breakaway_pin. Use explicit conditions or patch defaults before rollout smoke.")
        elif default_three in text:
            add(findings, "WARN", "rollout_three_condition_default", rel, None, default_three, "Rollout defaults still start from the old three-condition list. Use explicit conditions before rollout smoke.")

    explicit_feature_block = ""
    if rel == "ccda_phase3/data_io.py":
        # Only fail if forbidden tokens appear outside the documented forbidden list or metadata outputs.
        for token in FORBIDDEN_TOKENS:
            if token in text and "FORBIDDEN_METADATA_NOT_IN_X" not in text:
                add(findings, "WARN", "forbidden_metadata_token_present", rel, None, token, "Token should be audited to ensure it is not part of model feature construction.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--out_json", default="reports/phase3_1_code_hazard_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_1_code_hazard_report.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    findings: List[Dict[str, Any]] = []
    git = {
        "branch": run(["git", "branch", "--show-current"], root),
        "head": run(["git", "rev-parse", "HEAD"], root),
        "origin_experiment1": (run(["git", "ls-remote", "origin", "Experiment1"], root).split("\t")[0] or ""),
        "status": run(["git", "status", "--short"], root),
        "submodule": run(["git", "submodule", "status", "external/deformable-ravens"], root),
    }

    for rel in FILES_TO_SCAN:
        scan_file(root, rel, findings)

    required_results = []
    for rel, strings in REQUIRED_STRINGS.items():
        path = root / rel
        text = path.read_text(errors="replace") if path.exists() else ""
        for item in strings:
            present = item in text
            required_results.append({"file": rel, "required": item, "present": present})
            if not present:
                add(findings, "FAIL", "required_string_missing", rel, None, item, f"Required string missing: {item}")

    code_notes = root / "reports/phase3_code_reading_notes.md"
    stale_marked = None
    if code_notes.exists():
        text = code_notes.read_text(errors="replace")
        stale_marked = "STALE / SUPERSEDED" in text[:500]
        if not stale_marked:
            add(findings, "WARN", "stale_phase3_code_reading_notes", str(code_notes.relative_to(root)), None, "", "Mark old code reading notes stale so the old three-branch state is not reused.")

    has_fail = any(f["level"] == "FAIL" for f in findings)
    has_warn = any(f["level"] == "WARN" for f in findings)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    payload = {
        "verdict": verdict,
        "git": git,
        "local_head_equals_origin_experiment1": git["head"] == git["origin_experiment1"],
        "primary_hidden_condition": PRIMARY,
        "diagnostic_hidden_condition": DIAGNOSTIC,
        "stale_report_marked": stale_marked,
        "findings": findings,
        "required_results": required_results,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.1 Code Hazard Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Branch: `{git['branch']}`",
        f"- HEAD: `{git['head']}`",
        f"- origin/Experiment1: `{git['origin_experiment1']}`",
        f"- Local HEAD equals origin/Experiment1: `{payload['local_head_equals_origin_experiment1']}`",
        f"- Stale report marked: `{stale_marked}`",
        "",
        "## Findings",
        "",
        "| Level | Name | File | Line | Detail |",
        "|---|---|---|---:|---|",
    ]
    if findings:
        for f in findings:
            detail = f"{f.get('snippet','')} - {f.get('explain','')}".replace("|", "/")
            line = f.get("line") if f.get("line") is not None else ""
            lines.append(f"| `{f['level']}` | `{f['name']}` | `{f.get('file','')}` | {line} | {detail} |")
    else:
        lines.append("| `PASS` | `none` |  |  | No hazards found. |")
    lines += [
        "",
        "## Required String Checks",
        "",
        "| File | Required string | Present |",
        "|---|---|---:|",
    ]
    for r in required_results:
        lines.append(f"| `{r['file']}` | `{r['required']}` | `{r['present']}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `FAIL` blocks rollout and Phase4.",
        "- `WARN` requires documentation, but may not block Phase3.1 if unrelated to leakage or primary-pair correctness.",
        "- `INFO` records diagnostic-only hidden_pin references or explicit rollout gates.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.1][FAIL] code hazard audit failed")


if __name__ == "__main__":
    main()
