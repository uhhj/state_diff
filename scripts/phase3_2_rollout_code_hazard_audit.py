#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]


def scan(path: Path):
    txt = path.read_text(errors="replace")
    findings = []

    def add(level, name, detail):
        findings.append({"level": level, "name": name, "detail": detail})

    old_default = 'default=["free", "hidden_pin", "hidden_high_friction"]'
    if old_default in txt:
        add("FAIL", "rollout_three_condition_default", "Rollout default omits hidden_breakaway_pin.")
    if "free hidden_pin hidden_high_friction" in txt and "hidden_breakaway_pin" not in txt:
        add("FAIL", "rollout_three_condition_shell_default", "Shell rollout default omits hidden_breakaway_pin.")
    is_policy = path.name == "phase3_policy_rollout.py"
    if is_policy and "primary_hidden_condition" not in txt:
        add("FAIL", "missing_primary_hidden_condition_arg", "Rollout must expose primary_hidden_condition.")
    if is_policy and "diagnostic_hidden_condition" not in txt:
        add("FAIL", "missing_diagnostic_hidden_condition_arg", "Rollout must expose diagnostic_hidden_condition.")
    if "hidden_breakaway_pin" not in txt:
        add("FAIL", "missing_hidden_breakaway_pin", "Rollout must include selected recoverable branch.")
    if is_policy and "free_vs_hidden_pin_success_gap" in txt and "diagnostic_success_gap" not in txt:
        add("FAIL", "old_hidden_pin_primary_metric", "free_vs_hidden_pin metric exists without diagnostic reinterpretation.")
    if is_policy and "primary_success_gap" not in txt:
        add("FAIL", "missing_primary_success_gap", "Rollout summary must include primary_success_gap.")
    if "PHASE3_ALLOW_ROLLOUT" not in txt:
        add("FAIL", "missing_rollout_gate_env", "Rollout must be explicitly gated.")
    if "PHASE3_ROLLOUT_CONFIRMED" not in txt:
        add("FAIL", "missing_user_confirmation_gate", "Rollout must require user confirmation gate.")

    dangerous_feature_terms = [
        "hidden_contact_meta",
        "recoverability_params",
        "breakaway_released",
        "breakaway_release_step",
        "breakaway_max_disp_seen",
        "final_fraction",
        "success",
        "condition_id",
        "condition_name",
    ]
    for term in dangerous_feature_terms:
        for match in re.finditer(term, txt):
            start = max(0, match.start() - 160)
            end = min(len(txt), match.end() + 160)
            ctx = txt[start:end]
            if any(marker in ctx for marker in ["policy_input", "obs_vec", "state_vec", "np.concatenate", "torch.cat", "x ="]):
                if term in {"final_fraction", "success"} and any(ok in ctx for ok in ["summary", "trials", "writerow", "success_rate"]):
                    continue
                add("FAIL", "possible_forbidden_metadata_in_policy_input", f"{term} appears near input construction: {ctx[:300]}")
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out_json", default="reports/phase3_2_rollout_code_hazard_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_2_rollout_code_hazard_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    files = [root / "scripts/phase3_policy_rollout.py", root / "scripts/phase3_run_all.sh"]
    findings = []
    for path in files:
        if not path.exists():
            findings.append({"level": "FAIL", "name": "missing_file", "detail": str(path), "file": str(path)})
            continue
        for finding in scan(path):
            finding["file"] = str(path.relative_to(root))
            findings.append(finding)

    has_fail = any(f["level"] == "FAIL" for f in findings)
    has_warn = any(f["level"] == "WARN" for f in findings)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    payload = {"verdict": verdict, "findings": findings, "required_conditions": REQUIRED_CONDITIONS}

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.2 Rollout Code Hazard Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        "",
        "## Findings",
        "",
        "| Level | Name | File | Detail |",
        "|---|---|---|---|",
    ]
    if findings:
        for f in findings:
            lines.append(f"| `{f['level']}` | `{f['name']}` | `{f.get('file','')}` | {str(f['detail']).replace('|','/')} |")
    else:
        lines.append("| `PASS` | `none` |  | No rollout hazards found. |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- FAIL blocks rollout smoke.",
        "- Rollout must use primary pair `free_vs_hidden_breakaway_pin`.",
        "- `free_vs_hidden_pin` is allowed only as diagnostic.",
        "- Rollout must require `PHASE3_ALLOW_ROLLOUT=1` and `PHASE3_ROLLOUT_CONFIRMED=1`.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.2][FAIL] rollout code hazard audit failed")


if __name__ == "__main__":
    main()
