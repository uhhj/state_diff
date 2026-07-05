#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

FORBIDDEN_IMPORT_PATTERNS = [
    (r"^\s*import\s+ravens\s*$", "top_level_import_ravens"),
    (r"^\s*from\s+ravens\s+import\s+", "from_ravens_import_top_level"),
    (r"^\s*import\s+tensorflow\b", "tensorflow_import"),
    (r"^\s*from\s+tensorflow\b", "tensorflow_import"),
    (r"ravens\.agents", "ravens_agents_reference"),
    (r"ravens\.models", "ravens_models_reference"),
    (r"ravens\.datasets", "ravens_datasets_reference"),
]

ALLOWED_REQUIRED_PATTERNS = [
    "ravens.tasks",
    "ravens.environment",
    "hidden-contact-cable-line",
    "primary_hidden_condition",
    "diagnostic_hidden_condition",
    "hidden_breakaway_pin",
    "PHASE3_ALLOW_ROLLOUT",
    "PHASE3_ROLLOUT_CONFIRMED",
    "assert_no_tensorflow_loaded",
]


def scan(path):
    txt = path.read_text(errors="replace")
    findings = []
    code_no_comments = "\n".join(line for line in txt.splitlines() if not line.lstrip().startswith("#"))

    for pattern, name in FORBIDDEN_IMPORT_PATTERNS:
        for match in re.finditer(pattern, code_no_comments, flags=re.MULTILINE):
            line_no = code_no_comments[:match.start()].count("\n") + 1
            line = code_no_comments.splitlines()[line_no - 1]
            if "FORBIDDEN_ROLLOUT_PREFIXES" in line or "name.startswith" in line:
                continue
            findings.append({
                "level": "FAIL",
                "name": name,
                "line": line_no,
                "snippet": line,
                "detail": "Rollout path must avoid top-level Ravens / TF / agents imports.",
            })

    for req in ALLOWED_REQUIRED_PATTERNS:
        if req not in txt:
            findings.append({
                "level": "FAIL",
                "name": "required_rollout_string_missing",
                "line": None,
                "snippet": "",
                "detail": f"Missing required string: {req}",
            })

    dangerous_x_terms = [
        "hidden_contact_meta",
        "recoverability_params",
        "breakaway_released",
        "breakaway_release_step",
        "breakaway_max_disp_seen",
        "success",
        "final_fraction",
        "condition_id",
        "condition_name",
    ]
    for term in dangerous_x_terms:
        for match in re.finditer(term, code_no_comments):
            start = max(0, match.start() - 140)
            end = min(len(code_no_comments), match.end() + 140)
            ctx = code_no_comments[start:end]
            if any(key in ctx for key in ["policy_input", "obs_vec", "state_vec", "np.concatenate", "torch.cat", "x ="]):
                findings.append({
                    "level": "FAIL",
                    "name": "possible_forbidden_metadata_in_policy_input",
                    "line": code_no_comments[:match.start()].count("\n") + 1,
                    "snippet": ctx.replace("\n", " ")[:260],
                    "detail": f"{term} appears near policy input construction.",
                })
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out_json", default="reports/phase3_3b_rollout_import_hazard_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_3b_rollout_import_hazard_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    target = root / "scripts/phase3_policy_rollout.py"
    if not target.exists():
        findings = [{"level": "FAIL", "name": "missing_phase3_policy_rollout", "line": None, "snippet": "", "detail": str(target)}]
    else:
        findings = scan(target)
    verdict = "FAIL" if any(f["level"] == "FAIL" for f in findings) else "PASS"
    payload = {"verdict": verdict, "target": str(target), "findings": findings}

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.3b Rollout Import Hazard Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        "",
        "## Findings",
        "",
        "| Level | Name | Line | Detail |",
        "|---|---|---:|---|",
    ]
    if findings:
        for f in findings:
            detail = str(f.get("detail", "")).replace("|", "/")
            snippet = str(f.get("snippet", "")).replace("|", "/")
            lines.append(f"| `{f['level']}` | `{f['name']}` | {f.get('line') or ''} | {detail} `{snippet}` |")
    else:
        lines.append("| `PASS` | `none` |  | No import hazards found. |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- FAIL blocks rollout smoke.",
        "- Rollout must avoid TensorFlow and Ravens agents.",
        "- `free_vs_hidden_pin` may exist only as diagnostic, not primary.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict != "PASS":
        raise SystemExit("[Phase3.3b][FAIL] rollout import hazard audit failed")


if __name__ == "__main__":
    main()
